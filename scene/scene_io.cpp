#include "scene/scene_io.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace engine {
namespace scene {

namespace {

// ── Portable asset paths ─────────────────────────────────────────────────
// Scenes must survive being moved to another drive or machine.  Asset
// references are therefore stored RELATIVE to the app root (the working
// directory), and reads accept either form: legacy scenes that recorded an
// absolute path (e.g. "E:\...\realworld\assets\x.gltf") are re-rooted on
// load, so a project that changed drives still opens instead of failing
// with "loader returned null".

std::string toPortableScenePath(const std::string& p) {
    namespace fs = std::filesystem;
    if (p.empty()) return p;
    std::error_code ec;
    const fs::path abs(p);
    if (abs.is_relative()) return abs.generic_string();
    const fs::path root = fs::current_path(ec);
    if (ec) return p;
    const fs::path rel = fs::relative(abs, root, ec);
    if (ec || rel.empty()) return p;
    const std::string s = rel.generic_string();
    if (s.rfind("..", 0) == 0) return p;    // outside the project root
    return s;
}

std::string resolveScenePath(const std::string& stored) {
    namespace fs = std::filesystem;
    if (stored.empty()) return stored;
    std::error_code ec;
    if (fs::exists(fs::path(stored), ec)) return stored;
    const fs::path p(stored);
    if (!p.is_absolute()) return stored;
    // Rebuild under the current root from the first "assets"/"content" part.
    std::vector<std::string> parts;
    for (const auto& c : p) parts.push_back(c.string());
    for (size_t i = 0; i < parts.size(); ++i) {
        if (parts[i] == "assets" || parts[i] == "content") {
            fs::path re;
            for (size_t j = i; j < parts.size(); ++j) re /= parts[j];
            std::error_code e2;
            if (fs::exists(re, e2)) {
                std::cout << "[scene] re-rooted stale asset path '" << stored
                          << "' -> '" << re.generic_string() << "'\n";
                return re.generic_string();
            }
            break;
        }
    }
    return stored;
}

template <typename T>
void writePod(std::ofstream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
bool readPod(std::ifstream& is, T& v) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

void writeStr(std::ofstream& os, const std::string& s) {
    const uint32_t n = static_cast<uint32_t>(s.size());
    writePod(os, n);
    if (n) {
        os.write(s.data(), static_cast<std::streamsize>(n));
    }
}

bool readStr(std::ifstream& is, std::string& s) {
    uint32_t n = 0;
    if (!readPod(is, n)) {
        return false;
    }
    // Sanity cap (16 MB) so a corrupt length can't trigger a huge allocation.
    if (n > (1u << 24)) {
        return false;
    }
    s.resize(n);
    if (n) {
        return static_cast<bool>(is.read(&s[0], static_cast<std::streamsize>(n)));
    }
    return true;
}

void writeXform(std::ofstream& os, const Transform& t) {
    writePod(os, t.translation.x);
    writePod(os, t.translation.y);
    writePod(os, t.translation.z);
    writePod(os, t.rotation.x);
    writePod(os, t.rotation.y);
    writePod(os, t.rotation.z);
    writePod(os, t.rotation.w);
    writePod(os, t.scale.x);
    writePod(os, t.scale.y);
    writePod(os, t.scale.z);
}

bool readXform(std::ifstream& is, Transform& t) {
    bool ok = true;
    ok = ok && readPod(is, t.translation.x);
    ok = ok && readPod(is, t.translation.y);
    ok = ok && readPod(is, t.translation.z);
    ok = ok && readPod(is, t.rotation.x);
    ok = ok && readPod(is, t.rotation.y);
    ok = ok && readPod(is, t.rotation.z);
    ok = ok && readPod(is, t.rotation.w);
    ok = ok && readPod(is, t.scale.x);
    ok = ok && readPod(is, t.scale.y);
    ok = ok && readPod(is, t.scale.z);
    return ok;
}

const char     kMagic[8] = { 'R', 'W', 'S', 'C', 'E', 'N', 'E', '\0' };
// v2: + music_path (string) + music_volume (float32) after the object list.
// v3: + per-object audio_clip (string) + audio_loop (u8) + audio_volume (f32)
//     in each object record (after its transform) — BGM objects.
// v4: + collision_map_path (string) after the music trailer — baked
//     walkable-collision map (.rwcmap) reference.
// v5: + per-object light_color (3×f32) + light_intensity (f32) +
//     light_radius (f32) in each object record (after the audio fields)
//     — point-light objects (.rwlight) for the ReSTIR lighting path.
const uint32_t kVersion  = 5;

} // namespace

bool saveSceneBinary(const std::string& path, const Scene& scene) {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
        return false;
    }

    os.write(kMagic, 8);
    writePod(os, kVersion);
    writeStr(os, scene.name);
    writeXform(os, scene.root);
    writePod(os, static_cast<uint32_t>(scene.objects.size()));

    for (const auto& o : scene.objects) {
        writeStr(os, o.name);
        writeStr(os, toPortableScenePath(o.asset_path));
        writePod(os, o.parent_index);
        writePod(os, o.source_node_index);
        writePod(os, static_cast<uint8_t>(o.is_group ? 1 : 0));
        writePod(os, static_cast<uint8_t>(o.visible ? 1 : 0));
        writeXform(os, o.transform);
        // v3 per-object audio (BGM objects).
        writeStr(os, toPortableScenePath(o.audio_clip));
        writePod(os, static_cast<uint8_t>(o.audio_loop ? 1 : 0));
        writePod(os, o.audio_volume);
        // v5 per-object light attributes (.rwlight objects).
        writePod(os, o.light_color.x);
        writePod(os, o.light_color.y);
        writePod(os, o.light_color.z);
        writePod(os, o.light_intensity);
        writePod(os, o.light_radius);
    }

    // v2 trailer: scene music.
    writeStr(os, toPortableScenePath(scene.music_path));
    writePod(os, scene.music_volume);

    // v4 trailer: baked collision-map reference.
    writeStr(os, toPortableScenePath(scene.collision_map_path));

    return static_cast<bool>(os);
}

bool loadSceneBinary(const std::string& path, Scene& out_scene) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        return false;
    }

    char magic[8] = { 0 };
    if (!is.read(magic, 8)) {
        return false;
    }
    if (std::memcmp(magic, kMagic, 8) != 0) {
        return false;
    }

    uint32_t version = 0;
    if (!readPod(is, version)) {
        return false;
    }
    if (version < 1 || version > kVersion) {
        return false;  // newer than this build understands
    }

    Scene s;
    if (!readStr(is, s.name)) {
        return false;
    }
    if (!readXform(is, s.root)) {
        return false;
    }

    uint32_t count = 0;
    if (!readPod(is, count)) {
        return false;
    }
    // Sanity cap (1M objects) against a corrupt count.
    if (count > (1u << 20)) {
        return false;
    }

    s.objects.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        Object& o = s.objects[i];
        uint8_t is_group = 0;
        uint8_t visible  = 0;
        if (!readStr(is, o.name)) {
            return false;
        }
        if (!readStr(is, o.asset_path)) {
            return false;
        }
        // Legacy scenes stored absolute paths; re-root them so a project
        // that moved drive/machine still resolves its assets.
        o.asset_path = resolveScenePath(o.asset_path);
        if (!readPod(is, o.parent_index)) {
            return false;
        }
        if (!readPod(is, o.source_node_index)) {
            return false;
        }
        if (!readPod(is, is_group)) {
            return false;
        }
        if (!readPod(is, visible)) {
            return false;
        }
        if (!readXform(is, o.transform)) {
            return false;
        }
        o.is_group = (is_group != 0);
        o.visible  = (visible != 0);
        // v3 per-object audio (absent in v1/v2 — defaults stand).
        if (version >= 3) {
            uint8_t loop = 1;
            if (!readStr(is, o.audio_clip)) return false;
            o.audio_clip = resolveScenePath(o.audio_clip);
            if (!readPod(is, loop)) return false;
            if (!readPod(is, o.audio_volume)) return false;
            o.audio_loop = (loop != 0);
        }
        // v5 per-object light attributes (absent pre-v5 — defaults stand).
        if (version >= 5) {
            if (!readPod(is, o.light_color.x)) return false;
            if (!readPod(is, o.light_color.y)) return false;
            if (!readPod(is, o.light_color.z)) return false;
            if (!readPod(is, o.light_intensity)) return false;
            if (!readPod(is, o.light_radius)) return false;
        }
    }

    // v2 trailer: scene music (absent in v1 files — defaults stand).
    if (version >= 2) {
        if (!readStr(is, s.music_path)) {
            return false;
        }
        s.music_path = resolveScenePath(s.music_path);
        if (!readPod(is, s.music_volume)) {
            return false;
        }
    }

    // v4 trailer: baked collision-map reference (absent before v4).
    if (version >= 4) {
        if (!readStr(is, s.collision_map_path)) {
            return false;
        }
        s.collision_map_path = resolveScenePath(s.collision_map_path);
    }

    out_scene = std::move(s);
    return true;
}

} // namespace scene
} // namespace engine
