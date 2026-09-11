#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace engine::helper {
// Cache identity includes missing optional inputs so installing a new
// companion map also starts a fresh cache. No cached terrain is deleted.
inline std::string terrainCacheRevision(const std::vector<std::string>& inputs) {
    uint64_t hash = 14695981039346656037ull;
    const auto add = [&hash](const std::string& value) {
        for (unsigned char c : value) { hash ^= c; hash *= 1099511628211ull; }
        hash ^= 0xff; hash *= 1099511628211ull;
    };
    add("terrain-detail-v2");
    for (const auto& input : inputs) {
        std::error_code ec;
        const std::filesystem::path path(input);
        add(path.lexically_normal().generic_string());
        const auto size = std::filesystem::file_size(path, ec);
        add(ec ? "missing" : std::to_string(size));
        ec.clear();
        const auto time = std::filesystem::last_write_time(path, ec);
        add(ec ? "missing" : std::to_string(time.time_since_epoch().count()));
    }
    return std::to_string(hash);
}
}
