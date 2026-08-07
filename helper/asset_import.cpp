#include "asset_import.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "helper/model_inspect.h"

namespace engine {
namespace helper {
namespace {

// Absolute -> project-relative (mirrors application.cpp's
// toPortableAssetPath: sidecars must never bake in a drive letter, or
// moving the project breaks every import with "loader returned null").
std::string portablePath(const std::string& p) {
    namespace fs = std::filesystem;
    if (p.empty()) return p;
    std::error_code ec;
    const fs::path root = fs::current_path(ec);
    if (ec) return p;
    fs::path abs = fs::path(p);
    if (abs.is_relative()) return abs.generic_string();
    const fs::path rel = fs::relative(abs, root, ec);
    if (ec || rel.empty()) return p;
    const std::string s = rel.generic_string();
    if (s.rfind("..", 0) == 0) return p;       // escapes the project root
    return s;
}

// ── Import progress sidecar ──────────────────────────────────────────────
// The headless import is ONE blocking call across a C ABI
// (rw_import_assets), so a bake that runs for an hour reports nothing until
// it returns and every watcher concludes it has hung.  That is not a corner
// case: an 800 MB instanced GLB bakes to ~46 000 objects, each with five
// QEM-decimated LOD levels, and the whole thing is a single call.
//
// Rather than widen the ABI with a callback pointer — a version bump and a
// rebuild for every host — the bake drops a sidecar the caller can poll:
//
//   <group_dir>/.import.progress    "<done> <total> <name>"
//
// Same shape as the terrain pipeline's own .progress files, so the stage
// runner reads it with the parsing it already has.  Its ABSENCE means "not
// running", which is why the destructor removes it.
class ImportProgress {
public:
    ImportProgress(std::filesystem::path dir, std::string name)
        : path_(dir / ".import.progress"), name_(std::move(name)) {}
    ~ImportProgress() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    ImportProgress(const ImportProgress&) = delete;
    ImportProgress& operator=(const ImportProgress&) = delete;

    void operator()(size_t done, size_t total) {
        // Throttled: at 46 000 objects an unthrottled rewrite would cost
        // more than the work it reports on.  The LAST object always
        // writes, so a poller cannot miss the finished count by landing
        // inside the throttle window.
        const auto now = std::chrono::steady_clock::now();
        if (done < total &&
            now - last_ < std::chrono::milliseconds(250)) return;
        last_ = now;
        std::error_code ec;
        std::filesystem::create_directories(path_.parent_path(), ec);
        std::ofstream f(path_, std::ios::trunc);
        if (f) f << done << ' ' << total << ' ' << name_ << '\n';
    }

private:
    std::filesystem::path                 path_;
    std::string                           name_;
    std::chrono::steady_clock::time_point last_{};
};

// Replace this group's rows in content/asset_index.tsv (stable IDs make
// re-imports idempotent), append the fresh ones.  Columns:
//   id <TAB> type <TAB> name <TAB> path
void updateAssetIndex(const std::string& group_rel,
                      const std::vector<std::array<std::string, 4>>& rows) {
    namespace fs = std::filesystem;
    const fs::path index_path = fs::path("content") / "asset_index.tsv";
    std::vector<std::string> kept;
    {
        std::ifstream in(index_path);
        std::string line;
        const std::string prefix = group_rel + "/";
        while (in && std::getline(in, line)) {
            if (line.empty()) continue;
            const size_t p3 = line.rfind('\t');
            const std::string lpath =
                (p3 == std::string::npos) ? std::string()
                                          : line.substr(p3 + 1);
            if (lpath == group_rel || lpath.rfind(prefix, 0) == 0)
                continue;                       // superseded by this import
            kept.push_back(line);
        }
    }
    std::ofstream outf(index_path, std::ios::trunc);
    for (const auto& l : kept) outf << l << "\n";
    for (const auto& r : rows)
        outf << r[0] << '\t' << r[1] << '\t' << r[2] << '\t' << r[3] << "\n";
}

}  // namespace

bool importOneAssetToContent(const std::string& chosen,
                             const std::string& import_dir,
                             const std::string& mode) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(chosen, ec)) {
        std::cerr << "[import] source not found: " << chosen << std::endl;
        return false;
    }
    fs::create_directories(import_dir, ec);
    const fs::path src(chosen);

    // "group" falls through to the bake path below with group
    // semantics; "copy" keeps the managed-glTF-group escape hatch for
    // callers that explicitly want a source-form payload.
    if (mode == "copy") {
        // ── MANAGED GROUP, character-pattern ─────────────────────────
        // Instanced files cannot be baked to .rwgeo — for them, glTF IS
        // the engine's native instanced representation (the renderer
        // reads the EXT_mesh_gpu_instancing accessors, the _lodtile_
        // band names and the world-manifest bindings straight from the
        // glTF).  So they import the way skinned CHARACTERS always
        // have, for the identical reason (a payload the static bake
        // cannot represent): a proper group folder with an
        // import.rwmeta sidecar whose main= names the payload —
        //   <dir>/<stem>/<stem>.glb
        //   <dir>/<stem>/import.rwmeta   (id, guid, provenance, main=)
        // — never a loose raw file dropped in content/.  The engine's
        // group placement already resolves main=, and the imported-copy
        // index finds the payload by name wherever it sits.
        const fs::path group_dir = fs::path(import_dir) / src.stem();
        std::error_code gec;
        fs::create_directories(group_dir, gec);
        const fs::path dst = group_dir / src.filename();
        {
            std::ifstream in(src, std::ios::binary);
            std::ofstream out(dst, std::ios::binary | std::ios::trunc);
            if (!in || !out) {
                std::cerr << "[import] cannot open '" << src.string()
                          << "' -> '" << dst.string() << "'" << std::endl;
                return false;
            }
            std::vector<char> buf(1u << 20);
            while (in) {
                in.read(buf.data(), (std::streamsize)buf.size());
                const std::streamsize n = in.gcount();
                if (n <= 0) break;
                out.write(buf.data(), n);
                if (!out) {
                    std::cerr << "[import] short write to '" << dst.string()
                              << "'" << std::endl;
                    return false;
                }
            }
        }
        // A LOOSE copy of the same file from an earlier import would
        // race this group in the imported-copy index — supersede it.
        {
            std::error_code lec;
            const fs::path loose = fs::path(import_dir) / src.filename();
            if (fs::exists(loose, lec) && fs::is_regular_file(loose, lec)) {
                fs::remove(loose, lec);
                if (!lec) {
                    std::cout << "[import] removed superseded loose copy "
                              << loose.generic_string() << std::endl;
                }
            }
        }
        std::error_code rrec2;
        fs::path group_rel_p =
            fs::relative(group_dir, fs::path("content"), rrec2);
        if (rrec2 || group_rel_p.empty()) group_rel_p = group_dir;
        const std::string group_rel = group_rel_p.generic_string();
        const std::string group_id =
            makeAssetId("group", group_rel, src.stem().string());
        std::mt19937_64 rng{ std::random_device{}() };
        char guid[33];
        std::snprintf(guid, sizeof(guid), "%016llx%016llx",
                      (unsigned long long)rng(), (unsigned long long)rng());
        const auto now_t = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        {
            std::ofstream meta((group_dir / "import.rwmeta").string(),
                               std::ios::trunc);
            meta << "id=" << group_id << "\n"
                 << "guid=" << guid << "\n"
                 << "source=" << portablePath(chosen) << "\n"
                 << "imported_unix=" << (long long)now_t << "\n"
                 << "type=instanced_model\n"
                 << "main=" << src.filename().string() << "\n";
        }
        const std::string payload_rel =
            group_rel + "/" + src.filename().string();
        updateAssetIndex(
            group_rel,
            { { group_id, "group", src.stem().string(), group_rel },
              { makeAssetId("model", payload_rel, src.stem().string()),
                "model", src.stem().string(), payload_rel } });
        std::cout << "[import] managed group '"
                  << group_dir.generic_string() << "' (instanced payload "
                  << src.filename().string() << ", main= placement, "
                  << "managed copy wins over assets/ by name)" << std::endl;
        return true;
    }

    // ── bake mode ────────────────────────────────────────────────────
    // Instanced files bake NATIVELY now: meshes to .rwgeo (stored once,
    // LOD levels included), textures to .rwtex, and the per-instance
    // transforms to instances.rwinst — the runtime's loadRwInstanced
    // rebuilds the instanced drawable from exactly these files, bands
    // and world-manifest bindings intact.  glTF is a SOURCE format; it
    // never lands in content/.
    const bool instanced = modelHasGpuInstancing(chosen);
    // Generated terrain is never skinned, but guard anyway: the static
    // bake would freeze a rig.
    if (!instanced && modelHasSkin(chosen)) {
        std::cout << "[import] '" << chosen << "' is skinned — bake would "
                  << "freeze it; importing as managed copy instead"
                  << std::endl;
        return importOneAssetToContent(chosen, import_dir, "copy");
    }
    const fs::path group_dir = fs::path(import_dir) / src.stem();
    std::vector<BakedObject> baked;
    // Publish per-object progress for whoever is watching (the stage
    // runner polls the sidecar while its blocking DLL call runs).  The
    // file is removed when this goes out of scope, success or not.
    ImportProgress prog(group_dir, src.filename().string());
    const bool ok =
        bakeModelToRenderReady(chosen, group_dir.string(), baked,
                               [&prog](size_t d, size_t t) { prog(d, t); });
    if (!ok) {
        std::cerr << "[import] bake FAILED for '" << chosen << "'"
                  << std::endl;
        return false;
    }
    std::error_code rrec;
    fs::path group_rel_p = fs::relative(group_dir, fs::path("content"), rrec);
    if (rrec || group_rel_p.empty()) group_rel_p = group_dir;
    const std::string group_rel = group_rel_p.generic_string();
    const std::string group_id =
        makeAssetId("group", group_rel, src.stem().string());

    std::mt19937_64 rng{ std::random_device{}() };
    char guid[33];
    std::snprintf(guid, sizeof(guid), "%016llx%016llx",
                  (unsigned long long)rng(), (unsigned long long)rng());
    const auto now_t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    // ── the instance tables / group marker ───────────────────────────
    // group semantics ("group" mode, or any instanced source): the whole
    // file loads as ONE native drawable (loadRwInstanced) — bands, gates
    // and name-keyed behaviour intact — instead of one object per node.
    // A non-instanced group still writes instances.rwinst as an empty
    // MARKER: node names and mesh ordinals already live in
    // hierarchy.rwhier; the marker is what routes the load.
    const bool group_mode = instanced || (mode == "group");
    bool wrote_rwinst = false;
    size_t n_inst_nodes = 0, n_inst_tables = 0;
    if (group_mode && !instanced) {
        const std::string ip = (group_dir / "instances.rwinst").string();
        wrote_rwinst = writeRwInst(ip, {}, {});
        if (!wrote_rwinst) {
            std::cerr << "[import] FAILED to write '" << ip << "'"
                      << std::endl;
            return false;
        }
    }
    if (instanced) {
        std::vector<RwInstArray> arrays;
        std::vector<RwInstNode> inst_nodes;
        if (readGpuInstancing(chosen, arrays, inst_nodes) &&
            !inst_nodes.empty()) {
            const std::string ip =
                (group_dir / "instances.rwinst").string();
            wrote_rwinst = writeRwInst(ip, arrays, inst_nodes);
            n_inst_nodes = inst_nodes.size();
            n_inst_tables = arrays.size();
            if (!wrote_rwinst) {
                std::cerr << "[import] FAILED to write '" << ip << "'"
                          << std::endl;
                return false;
            }
        } else {
            std::cerr << "[import] '" << chosen << "' reports gpu "
                      << "instancing but no tables decoded — refusing a "
                      << "bake that would lose the instances" << std::endl;
            return false;
        }
        // A leftover glTF payload from the interim managed-copy scheme
        // would shadow the native data — supersede it.
        std::error_code pec;
        const fs::path old_payload = group_dir / src.filename();
        if (fs::exists(old_payload, pec)) {
            fs::remove(old_payload, pec);
            if (!pec) {
                std::cout << "[import] removed superseded glTF payload "
                          << old_payload.generic_string() << std::endl;
            }
        }
    }
    {
        std::ofstream meta((group_dir / "import.rwmeta").string(),
                           std::ios::trunc);
        meta << "id=" << group_id << "\n"
             << "guid=" << guid << "\n"
             << "source=" << portablePath(chosen) << "\n"
             << "imported_unix=" << (long long)now_t << "\n";
        if (wrote_rwinst) {
            // main= is what the group placement resolves: ONE scene
            // object built by the native instanced loader — never one
            // object per node, which is what per-node .rwobj files
            // would produce for a group with thousands of instances.
            meta << "type=instanced_group\n"
                 << "main=instances.rwinst\n";
        } else {
            meta << "type=model_group\n";
        }
        meta << "subobjects_baked=1\n";
        for (const auto& b : baked) meta << "subobject=" << b.name << "\n";
    }

    auto sanitize = [](std::string s2) {
        for (auto& ch : s2) {
            if (ch == '/' || ch == '\\' || ch == ':' || ch == '*' ||
                ch == '?' || ch == '"' || ch == '<' || ch == '>' ||
                ch == '|') ch = '_';
        }
        if (s2.empty()) s2 = "object";
        return s2;
    };
    std::vector<std::array<std::string, 4>> index_rows;
    index_rows.push_back({ group_id, "group", src.stem().string(),
                           group_rel });
    if (wrote_rwinst) {
        const std::string irel = group_rel + "/instances.rwinst";
        index_rows.push_back(
            { makeAssetId("instances", irel, src.stem().string()),
              "instances", src.stem().string(), irel });
    }
    for (size_t k = 0; !wrote_rwinst && k < baked.size(); ++k) {
        char pre[16];
        std::snprintf(pre, sizeof(pre), "%03u_", (unsigned)k);
        const std::string fname =
            std::string(pre) + sanitize(baked[k].name) + ".rwobj";
        const fs::path op = group_dir / fname;
        const std::string obj_rel = group_rel + "/" + fname;
        const std::string obj_id =
            makeAssetId("object", obj_rel, baked[k].name);
        std::ofstream of(op, std::ios::trunc);
        of << "rwobj=1\n"
           << "id=" << obj_id << "\n"
           << "source=" << portablePath(chosen) << "\n"
           << "node=" << k << "\n"
           << "name=" << baked[k].name << "\n";
        if (!baked[k].rwgeo_rel.empty()) {
            of << "geo=" << baked[k].rwgeo_rel << "\n";
            const auto& bn = baked[k].bbox_min;
            const auto& bx = baked[k].bbox_max;
            of << "bbox=" << bn.x << ',' << bn.y << ',' << bn.z << ','
               << bx.x << ',' << bx.y << ',' << bx.z << "\n";
        }
        index_rows.push_back({ obj_id, "object", baked[k].name, obj_rel });
    }
    {
        std::error_code tec;
        const fs::path tdir = group_dir / "textures";
        for (auto& e : fs::directory_iterator(tdir, tec)) {
            if (e.path().extension() != ".rwtex") continue;
            const std::string trel =
                group_rel + "/textures/" + e.path().filename().string();
            const std::string tname = e.path().stem().string();
            index_rows.push_back(
                { makeAssetId("texture", trel, tname), "texture",
                  tname, trel });
        }
    }
    updateAssetIndex(group_rel, index_rows);
    std::cout << "[import] baked '" << chosen << "' -> "
              << baked.size() << " render-ready object(s) in '"
              << group_dir.generic_string() << "'  (group id " << group_id
              << ", " << index_rows.size() << " index rows";
    if (wrote_rwinst) {
        std::cout << ", instances.rwinst: " << n_inst_nodes
                  << " instanced node(s) over " << n_inst_tables
                  << " shared table(s)";
    }
    std::cout << ")" << std::endl;
    return true;
}

int importAssetsToContent(const std::string& sources,
                          const std::string& import_dir,
                          const std::string& mode) {
    int done = 0, failed = 0;
    size_t start = 0;
    for (;;) {
        const size_t semi = sources.find(';', start);
        const std::string one =
            (semi == std::string::npos) ? sources.substr(start)
                                        : sources.substr(start, semi - start);
        if (!one.empty()) {
            if (importOneAssetToContent(one, import_dir, mode)) ++done;
            else                                                ++failed;
        }
        if (semi == std::string::npos) break;
        start = semi + 1;
    }
    std::cout << "[import] finished: " << done << " ok, " << failed
              << " failed" << std::endl;
    return failed ? 1 : 0;
}

} // namespace helper
} // namespace engine
