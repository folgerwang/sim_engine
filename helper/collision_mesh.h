#pragma once

#include <memory>
#include <vector>
#include "renderer/renderer.h"
#include "helper/bvh.h"
#include "helper/collision_debug_draw.h"

namespace engine {
namespace game_object { class DrawableObject; }
namespace helper {

// How buildFromDrawablePrimitive shapes the resulting CollisionMesh.
//   None         -- keep the welded source triangle list verbatim.
//   Decimate     -- run OpenMesh QEM decimation (preserves
//                   silhouette, yields a low-poly mesh that still
//                   looks like the original).
//   VoxelCube    -- conservatively voxelise the welded mesh into a
//                   `voxel_size`-spaced grid, then emit a surface
//                   mesh of axis-aligned cubes for every occupied
//                   cell, culling the faces between two occupied
//                   cells. Result is a "Minecraft" view of the
//                   primitive -- chunky, simple, gap-free. Default
//                   shape; default voxel_size is 5cm.
//   VoxelSphere  -- same voxelisation, but each occupied cell emits
//                   a low-poly octahedron centered on the cell.
//                   Useful when the cube faces feel too rigid;
//                   triangle count is higher because there's no
//                   adjacency culling.
enum class CollisionShape {
    None,
    Decimate,
    VoxelCube,
    VoxelSphere,
};

// CollisionMesh — static-mesh CPU collision representation.
class CollisionMesh {
public:
    // Build the CPU triangle list (always) plus, optionally, the
    // multithreaded SAH BVH used by `resolveCapsule`. Pass
    // build_bvh=false when only the debug visualisation needs the
    // mesh -- the BVH build is O(N log N) with a worker-thread pool
    // and on a Bistro-sized scene takes seconds, which is not
    // acceptable on the render thread. With BVH skipped the call
    // completes in milliseconds; resolveCapsule() will early-return
    // (no bvh_root_) so static-mesh capsule collision is disabled
    // for that mesh.
    //
    // Concatenates ALL meshes inside the drawable into a single
    // CollisionMesh -- correct for capsule physics (one BVH per
    // drawable is plenty) but produces a coarse single-colour blob
    // in the segmentation viz. Use buildFromDrawableMesh() instead
    // when you want per-FBX-mesh granularity in the debug view.
    bool buildFromDrawable(
        const game_object::DrawableObject& drawable,
        bool build_bvh = true);

    // Build a CollisionMesh from a SINGLE mesh inside the drawable
    // (data.meshes_[mesh_idx]) so each FBX mesh can be drawn with
    // its own segmentation colour. Returns false if the drawable is
    // not ready, the index is out of range, the mesh has no
    // populated vertex_position_ / vertex_indices_, or the
    // resulting triangle list is too small. build_bvh has the same
    // meaning as in buildFromDrawable above.
    bool buildFromDrawableMesh(
        const game_object::DrawableObject& drawable,
        size_t mesh_idx,
        bool build_bvh = true);

    // Build a CollisionMesh from a SINGLE primitive (one material
    // part) of one mesh. Each primitive maps cleanly to one
    // material, so the resulting CollisionMesh carries a
    // self-contained `material_name_` that gameplay code can read
    // for surface-aware behaviour (footstep sounds, friction, decal
    // selection). The triangle list is also QEM-decimated to
    // `c_target_lod_ratio` of its original face count when
    // simplify=true, preserving silhouette while lowering both GPU
    // upload size and BVH-build cost.
    //
    // weld_eps: spatial-quantisation epsilon (in scene units --
    // metres for Bistro) used to merge near-coincident vertices.
    // Authoring tools and FBX export often leave gaps where two
    // parts meet; quantising to a `weld_eps` grid and de-duplicating
    // closes those gaps so the decimater sees a properly connected
    // surface and the resulting collision / viz mesh has no
    // hairline cracks. Default 0.1m (10cm) is aggressive -- it
    // collapses small props into a single chunk and bridges visible
    // seams between adjacent floor / wall sections, which is what
    // we want for a coarse collision proxy. Lower it (e.g. 1e-3f)
    // when you need finer fidelity. Pass 0.0f to disable welding
    // entirely.
    //
    // shape selects how the welded triangle list is finalised. The
    // default `VoxelCube` voxelises the primitive into 5cm cells and
    // emits a chunky surface mesh -- much simpler than the source
    // and gap-free. See the `CollisionShape` doc comment for
    // alternatives.
    //
    // voxel_size: edge length of each voxel cell, in scene units
    // (metres for Bistro). Only consulted by the VoxelCube /
    // VoxelSphere shapes. Smaller values preserve more surface
    // detail but multiply triangle count cubically. The total grid
    // size is capped internally so a single oversized primitive
    // can't blow up memory.
    bool buildFromDrawablePrimitive(
        const game_object::DrawableObject& drawable,
        size_t mesh_idx,
        size_t prim_idx,
        bool build_bvh = true,
        CollisionShape shape = CollisionShape::VoxelCube,
        float weld_eps = 0.1f,
        float voxel_size = 0.05f);

    const std::string& materialName() const { return material_name_; }

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

    // Source-asset material name for the primitive that produced
    // this CollisionMesh (only set by buildFromDrawablePrimitive).
    // Empty for the multi-mesh / multi-primitive build paths since
    // a single name can't summarise an aggregated triangle list.
    std::string                 material_name_;

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
