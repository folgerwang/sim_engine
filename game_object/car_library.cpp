#include "car_library.h"

#include <cstring>
#include <fstream>
#include <iostream>

namespace engine {
namespace game_object {
namespace {

// ── A BOUNDS-CHECKED CURSOR OVER THE FILE ───────────────────────────
// The whole file is read once into memory (9 MB) and walked with this.
// Every read is checked, so a truncated or corrupt library returns
// false from one place instead of running off the end of a buffer —
// which matters more than usual here because the file is GENERATED and
// therefore changes whenever car_gen does.
struct Reader {
    const uint8_t* p = nullptr;
    const uint8_t* end = nullptr;
    bool ok = true;

    bool take(void* dst, size_t n) {
        if (!ok || size_t(end - p) < n) { ok = false; return false; }
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }
    uint32_t u32() { uint32_t v = 0; take(&v, 4); return v; }
    float    f32() { float v = 0.0f; take(&v, 4); return v; }
    std::string str() {
        const uint32_t n = u32();
        if (!ok || n > (1u << 20) || size_t(end - p) < n) { ok = false; return {}; }
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
    // Bulk read straight into a vector — the meshes are the bulk of the
    // file and a per-element take() would dominate the load time.
    template <typename T>
    bool vec(std::vector<T>& dst, size_t count) {
        if (!ok || count > (1u << 26) ||
            size_t(end - p) < count * sizeof(T)) { ok = false; return false; }
        dst.resize(count);
        if (count) std::memcpy(dst.data(), p, count * sizeof(T));
        p += count * sizeof(T);
        return true;
    }
};

bool readMesh(Reader& r, CarMesh& m) {
    const uint32_t nv = r.u32();
    const uint32_t ni = r.u32();
    if (!r.ok) return false;
    if (!r.vec(m.pos, nv) || !r.vec(m.nrm, nv) ||
        !r.vec(m.part, nv) || !r.vec(m.idx, ni)) {
        return false;
    }
    for (uint32_t i : m.idx) {
        if (i >= nv) return false;          // an index off the end
    }
    return true;
}

}  // namespace

std::vector<int> CarLibrary::byPrefix(const std::string& prefix) const {
    std::vector<int> out;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (samples[i].name.compare(0, prefix.size(), prefix) == 0) {
            out.push_back(int(i));
        }
    }
    return out;
}

bool loadCarLibrary(const std::string& path, CarLibrary& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cout << "[cars] no library at " << path
                  << " — falling back to the built-in hulls" << std::endl;
        return false;
    }
    const std::streamsize bytes = f.tellg();
    if (bytes <= 16) return false;
    // static_cast, not size_t(bytes): the functional cast makes this a
    // MOST VEXING PARSE — `size_t(bytes)` reads as a parameter
    // declaration, so the line declares a FUNCTION `blob` taking a
    // size_t and returning a vector, and every use of blob below then
    // fails with "left of '.data' must have class/struct/union".
    std::vector<uint8_t> blob(static_cast<size_t>(bytes));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(blob.data()), bytes)) return false;

    Reader r{blob.data(), blob.data() + blob.size(), true};
    char magic[8] = {};
    if (!r.take(magic, 8) || std::memcmp(magic, "RWCAR\0\0\0", 8) != 0) {
        std::cout << "[cars] '" << path << "' is not a car library"
                  << std::endl;
        return false;
    }
    const uint32_t version = r.u32();
    const uint32_t n_samples = r.u32();
    // Format 2 splits the interior into its own mesh so the engine can
    // tint it per instance.  Format 1 had it welded into the body and
    // this build cannot colour that, so it is refused rather than drawn
    // with seats the colour of the paint.
    if (version != 2) {
        std::cout << "[cars] library is format " << version
                  << ", this build reads 2 — re-run the cars stage"
                  << std::endl;
        return false;
    }
    if (!r.ok || n_samples == 0 || n_samples > 4096) return false;

    CarLibrary lib;
    const uint32_t n_paints = r.u32();
    if (!r.ok || n_paints > 1024) return false;
    lib.paints.resize(n_paints);
    for (CarPaint& p : lib.paints) {
        p.name = r.str();
        r.take(&p.base, sizeof(glm::vec3));
        p.metal = r.f32(); p.flake = r.f32(); p.coat = r.f32();
        r.take(&p.tint, sizeof(glm::vec3));
    }
    const uint32_t n_int = r.u32();
    if (!r.ok || n_int > 1024) return false;
    lib.interiors.resize(n_int);
    for (CarInterior& c : lib.interiors) {
        c.name = r.str();
        r.take(&c.rgb, sizeof(glm::vec3));
        c.roughness = r.f32();
    }
    if (!r.ok) return false;

    lib.samples.resize(n_samples);
    size_t tris = 0;
    for (CarSample& s : lib.samples) {
        s.name = r.str();
        s.L = r.f32(); s.W = r.f32(); s.H = r.f32();
        s.wheel_r = r.f32(); s.wheel_w = r.f32(); s.track_z = r.f32();
        const uint32_t n_axle = r.u32();
        if (!r.ok || n_axle == 0 || n_axle > 16) return false;
        s.axle_x.resize(n_axle);
        s.axle_dual.resize(n_axle);
        for (uint32_t k = 0; k < n_axle; ++k) {
            s.axle_x[k] = r.f32();
            s.axle_dual[k] = r.f32();
        }
        s.vmax = r.f32();
        s.bar = r.u32() != 0;
        s.bar_at_size.x = r.f32(); s.bar_at_size.y = r.f32();
        s.bar_at_size.z = r.f32(); s.bar_at_size.w = r.f32();
        const uint32_t n_pal = r.u32();
        if (!r.ok || n_pal > 1024) return false;
        s.paints.resize(n_pal);
        for (uint32_t& v : s.paints) {
            v = r.u32();
            if (v >= n_paints) return false;
        }
        if (!readMesh(r, s.body) || !readMesh(r, s.wheel) ||
            !readMesh(r, s.interior)) {
            std::cout << "[cars] '" << s.name << "' has a bad mesh"
                      << std::endl;
            return false;
        }
        tris += (s.body.idx.size() + s.wheel.idx.size() +
                 s.interior.idx.size()) / 3;
    }
    if (!r.ok) return false;

    lib.loaded = true;
    out = std::move(lib);
    std::cout << "[cars] library: " << out.samples.size() << " samples, "
              << out.paints.size() << " paints x " << out.interiors.size()
              << " interiors, " << tris << " triangles" << std::endl;
    return true;
}

}  // namespace game_object
}  // namespace engine
