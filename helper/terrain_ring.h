#pragma once
#include <glm/glm.hpp>
#include <vector>
namespace engine::helper {
struct TerrainRect { glm::vec2 min, max; };
// Partition a tile around finer coverage. Rectangles have disjoint interiors.
inline std::vector<TerrainRect> subtractTerrainRect(TerrainRect tile, TerrainRect fine) {
    std::vector<TerrainRect> out;
    auto emit = [&](glm::vec2 lo, glm::vec2 hi) {
        if (hi.x > lo.x && hi.y > lo.y) out.push_back({lo, hi});
    };
    auto lo = glm::max(tile.min, fine.min), hi = glm::min(tile.max, fine.max);
    if (lo.x >= hi.x || lo.y >= hi.y) { emit(tile.min, tile.max); return out; }
    emit(tile.min, {lo.x, tile.max.y});
    emit({hi.x, tile.min.y}, tile.max);
    emit({lo.x, tile.min.y}, {hi.x, lo.y});
    emit({lo.x, hi.y}, {hi.x, tile.max.y});
    return out;
}
}
