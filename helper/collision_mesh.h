#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
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

    // Cast a vertical ray straight down from `from` and return the
    // closest triangle hit within `max_distance`.  Walks this mesh's
    // BVH so the cost is O(log N) per leaf descent rather than O(N)
    // over the full triangle list.
    //
    // Returns false (and leaves out_hit / out_normal untouched) when:
    //   - the BVH wasn't built for this mesh (build_bvh=false at
    //     construction), or
    //   - the ray misses every triangle, or
    //   - the closest hit is further than max_distance below `from`.
    //
    // out_normal is the geometric face normal of the hit triangle,
    // computed from its vertex order — note: not necessarily the
    // "up-pointing" normal; for a downward ray hitting a floor it
    // typically points up, but for a ceiling it'll point down.
    // Caller should renormalize if it cares about sign.
    bool raycastDown(
        const glm::vec3& from,
        float max_distance,
        glm::vec3& out_hit,
        glm::vec3& out_normal) const;

    // Synchronously build the per-mesh SAH BVH from the already-
    // populated `vertices_` / `indices_` arrays. Safe to call from a
    // worker thread; uses a per-mesh mutex + an atomic ready flag so
    // concurrent raycastDown calls can either see a fully-built tree
    // or fall back to brute force without tearing on a half-set
    // shared_ptr. Returns true on success.  No-op (returns true) if
    // the BVH is already ready; returns false if the mesh is empty.
    bool buildBVH();

    // Cheap atomic test used by raycastDown to pick BVH vs brute
    // force.  Becomes true after buildBVH() finishes successfully.
    bool isBVHReady() const {
        return bvh_ready_.load(std::memory_order_acquire);
    }

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
    // `bvh_root_` is written exactly once (by buildBVH()) and read
    // many times (raycastDown / resolveCapsule).  The write must
    // happen-before any read that uses the tree, so we publish it
    // through `bvh_ready_` (release on set, acquire on test) and
    // copy the shared_ptr under `bvh_mutex_` to avoid tearing the
    // control-block pointer on platforms where shared_ptr
    // assignment isn't atomic.
    std::shared_ptr<BVHNode>    bvh_root_;
    mutable std::mutex          bvh_mutex_;
    std::atomic<bool>           bvh_ready_{false};
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
    ~CollisionWorld() { waitForBVHs(); }

    void addMesh(std::shared_ptr<CollisionMesh> mesh) {
        if (mesh && !mesh->empty()) meshes_.push_back(std::move(mesh));
    }
    // clear() tears down the mesh list; the async builder (if any)
    // could still be holding shared_ptrs to those meshes, so we have
    // to wait it out before releasing them.  Otherwise a concurrent
    // build would resurrect the mesh and we'd end up with a BVH for
    // a world that no longer exists.
    void clear() {
        waitForBVHs();
        meshes_.clear();
    }
    bool empty() const { return meshes_.empty(); }
    size_t meshCount() const { return meshes_.size(); }

    // Kick off a background thread that calls m->buildBVH() on every
    // mesh currently in this world.  Non-blocking: returns immediately.
    // While the build runs, CollisionMesh::raycastDown / resolveCapsule
    // fall back to brute force per mesh whose BVH isn't ready yet, so
    // foot IK keeps working from frame one and just speeds up as the
    // worker drains the list.  Safe to call multiple times; subsequent
    // calls while a build is in flight are silently ignored.  The
    // worker logs per-mesh + total build time.
    void buildBVHsAsync();

    // Join the async builder thread if it's running.  Called by
    // clear() and the destructor.  Safe to call when no build is in
    // flight (returns immediately).
    void waitForBVHs();

    bool resolveCapsule(
        glm::vec3& position,
        float radius,
        float height,
        glm::vec3& out_normal) const;

    // Cast a vertical ray downward from `from` against every mesh in
    // the world and return the closest hit within `max_distance`.
    // Uses per-mesh AABB early rejection (XZ overlap + Y range) before
    // descending each mesh's BVH, so meshes nowhere near the foot
    // column are skipped in O(1).  Cost scales with the number of
    // meshes that overlap the foot's vertical column, which for a
    // typical scene is a small handful even though the world holds
    // thousands of meshes.
    //
    // Returns false when no mesh is hit within `max_distance` below
    // `from`.  On hit, out_hit is the world-space intersection point
    // and out_normal is the hit triangle's face normal.
    bool raycastDown(
        const glm::vec3& from,
        float max_distance,
        glm::vec3& out_hit,
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

    // Async BVH builder bookkeeping.  `bvh_build_in_flight_` gates
    // re-entry from buildBVHsAsync() so a second call while a build
    // is already running is a no-op.  The thread is joined in
    // waitForBVHs() (called from clear() and the destructor).
    std::thread       bvh_builder_thread_;
    std::atomic<bool> bvh_build_in_flight_{false};
};

} // namespace helper
} // namespace engine
