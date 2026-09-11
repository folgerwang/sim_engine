#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

namespace engine::helper {
inline std::string unshardedGlbName(std::string name) {
    const auto part = name.rfind(".part");
    if (part != std::string::npos && name.size() > part + 9 &&
        name.compare(name.size() - 4, 4, ".glb") == 0 &&
        std::all_of(name.begin() + part + 5, name.end() - 4,
                    [](unsigned char c) { return c >= '0' && c <= '9'; })) {
        name.erase(part, name.size() - 4 - part);
    }
    return name;
}

inline bool pcgFamilyIsSharded(const std::string& filename) {
    namespace fs = std::filesystem;
    fs::path group(filename);
    if (group.filename() == "instances.rwinst") group = group.parent_path();
    else group.replace_extension();
    const std::string name = group.filename().string() + ".glb";
    std::error_code ec;
    return unshardedGlbName(name) != name ||
        fs::exists(group.string() + ".part1.glb", ec) ||
        fs::exists(fs::path(group.string() + ".part1") / "instances.rwinst", ec);
}
}
