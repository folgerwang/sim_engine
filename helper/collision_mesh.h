#pragma once

#include <memory>
#include <vector>
#include "renderer/renderer.h"
#include "helper/bvh.h"
#include "helper/collision_debug_draw.h"

namespace engine {
namespace game_object { class DrawableObject; }
namespace helper {

// CollisionMesh — static-mesh CPU collision representation.
class CollisionMesh {
public:
    bool buildFromDrawable(const game_object::DrawableObject& drawable);

    bool resolveCapsule(
        glm::vec3& position,
        float radius,
        float height,
        glm::vec3& out_normal,
        int max_iterations = 4) const;

    bool empty() const { return indices_.empty(); }
    size_t triangleCount() const { return indices_.size() / 3; }
    const AABB& bounds() const { return bounds_; }

    // Read-only access to the flat (vertex, index) arrays so
    // CollisionDebugDraw can expand them per-triangle.
    const std::vector<glm::vec3>& debugVertices() const { return vertices_; }
    const std::vector<int>&       debugIndices()  const { return indices_; }

    // GPU-side debug buffers (positions + per-triangle ids), populated
    // lazily on first call to CollisionDebugDraw::uploadForMesh.
    CollisionDebugMeshBuffers&       debugBuffers()       { return debug_gpu_; }
    const CollisionDebugMeshBuffers& debugBuffers() const { return debug_gpu_; }

private:
    std::vector<glm::vec3>      vertices_;
    std::vector<int>            indices_;
    std::shared_ptr<BVHNode>    bvh_root_;
    AABB                        bounds_;

    // Debug GPU buffers — `mutable` so const draw paths can populate
    // them on first use.
    mutable CollisionDebugMeshBuffers debug_gpu_;

    void queryBVH(
        const AABB& query_box,
        std::vector<int>& out_tri_indices) const;

    bool resolveCapsuleStep(
        glm::vec3& position,
        float radius,
        float height,
        glm::vec3& accum_normal,
        int& contact_count) const;
};

class CollisionWorld {
public:
    void addMesh(std::shared_ptr<CollisionMesh> mesh) {
        if (mesh && !mesh->empty()) meshes_.push_back(std::move(mesh));
    }
    void clear() { meshes_.clear(); }
    bool empty() const { return meshes_.empty(); }
    size_t meshCount() const { return meshes_.size(); }

    bool resolveCapsule(
        glm::vec3& position,
        float radius,
        float height,
        glm::vec3& out_normal) const;

    // Draw every collision mesh as flat-shaded debug triangles.
    void drawDebug(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors) const;

    void destroyDebugBuffers(
        const std::shared_ptr<renderer::Device>& device);

private:
    std::vector<std::shared_ptr<CollisionMesh>> meshes_;
};

} // namespace helper
} // namespace engine
