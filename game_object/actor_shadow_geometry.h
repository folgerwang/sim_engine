#pragma once
#include <vector>
#include <cstdint>
#include "renderer/renderer.h"
namespace engine::game_object {
// CPU copy of actor geometry; world-space frame data is consumed before
// the raster draw by the software BVH and hardware dynamic BLAS paths.
struct ActorShadowGeometry {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    template<class Transform>
    void append(const ActorShadowGeometry& mesh, Transform transform) {
        const uint32_t base = static_cast<uint32_t>(positions.size());
        for (const auto& p : mesh.positions) positions.push_back(transform(p));
        for (uint32_t i : mesh.indices) indices.push_back(base + i);
    }
};
}
