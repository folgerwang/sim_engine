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
//                   primitive -- chunky, simple, gap-free.
//                   Default voxel_size is 5cm.
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

// Semantic role of a CollisionMesh in the gameplay world.  Assigned at
// build time by the LLM material classifier (see MaterialClassifier);
// application code can override it via setCategory().  Gameplay code can
// branch on this — Floor for navigation / foot IK targets, Wall for
// blocking, Door for interactable openings, Object for the per-thing
// physics pile (tables, chairs, bottles, …).  Also the colour key used
// by the segmentation debug draw, so the values double as the shader
// "category id".  Order matters: the values are reused as a stable
// integer key the fragment shader hashes / switches on.
enum class MeshCategory : uint32_t {
    Unknown    = 0,
    Floor      = 1,  // walkable: roads, sidewalks, interior floors, doormats
    Wall       = 2,  // vertical blocking: brick, plaster, stucco, cladding
    Door       = 3,  // openings: explicit door / gate / rollup
    Object     = 4,  // gameplay props: tables, chairs, bottles, lamps, signs
    Glass      = 5,  // see-through-but-blocking (window / glass)
    Ceiling    = 6,  // blocks-from-below: ceilings, interior roof underside
    Stairs     = 7,  // walkable but step-aware navigation (different gait)
    Vegetation = 8,  // non-collidable or soft: foliage, ivy, leaves, grass
    Elevator   = 9,  // walkable + vertical traversal (lift platforms)
    Ladder     = 10, // vertical traversal with hand-over-hand gait
};

// Stable, all-caps tags used for serialization (LLM responses, JSON
// caches, log lines).  Keep in lockstep with the enum above; the LLM
// is prompted to emit one of these exact strings.
const char* meshCategoryTag(MeshCategory c);
MeshCategory meshCategoryFromTag(const std::string& tag);

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
    // default `Decimate` runs OpenMesh QEM simplification on the
    // welded triangle list, producing a low-poly mesh that still
    // looks like the original. Compared to VoxelCube it preserves
    // the silhouette (good for capsule sliding along thin walls)
    // but can leave hairline gaps where (a) two primitives meet
    // and only weld within each side, or (b) authoring-time
    // vertex pairs straddle the weld grid boundary -- see the
    // 3x3x3 neighbour lookup below for the latter mitigation, and
    // the `CollisionShape` doc comment for alternatives if a
    // gap-free shell is required.
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
        CollisionShape shape = CollisionShape::Decimate,
        float weld_eps = 0.1f,
        float voxel_size = 0.05f);

    const std::string& materialName() const { return material_name_; }

    // FBX/glTF node name of the first node referencing the mesh this
    // primitive belongs to (e.g. "Bistro_Research_Exterior_Linde_
    // Tree_Large_linde_tree_large_4051").  Captured during the same
    // node-scan that bakes the world transform; an empty string means
    // no node referenced the mesh.  Useful as a SECOND classification
    // signal — bistro material names are sometimes generic ("Wood",
    // "Metal_Worn") while the node names carry the actual semantics
    // ("Stairs", "Linde_Tree", "Elevator_Shaft").
    const std::string& objectName() const { return object_name_; }

    // ── Source primitive identity (debug / comparison) ──────────────
    // The (drawable, mesh, primitive) this CollisionMesh was built from,
    // captured by buildFromDrawablePrimitive.  Lets debug tooling re-
    // extract the ORIGINAL, un-simplified source geometry for a given
    // collision-world index (e.g. the isolate overlay draws the original
    // background mesh next to the simplified one).  sourceDrawable() is
    // null for the aggregate build paths that don't track a single prim.
    const game_object::DrawableObject* sourceDrawable() const {
        return src_drawable_;
    }
    size_t sourceMeshIdx() const { return src_mesh_idx_; }
    size_t sourcePrimIdx() const { return src_prim_idx_; }

    // Semantic category — Floor / Wall / Door / Object / Glass /
    // Ceiling / Stairs / Vegetation / Elevator / Ladder / Unknown.
    // Left at the default Unknown by the build paths; the collision build
    // (application.cpp) stamps the final value from the LLM material
    // classifier's verdict via setCategory().
    MeshCategory category() const { return category_; }
    void setCategory(MeshCategory c) { category_ = c; }

    // Floor patches authored as a single primitive routinely bundle
    // non-walkable VERTICAL faces — curb sides, step risers, planter
    // and fountain walls, road-edge kerbs — into the same "road" /
    // "sidewalk" / "floor" material.  Those faces have a horizontal
    // surface normal (|normal.y| ~ 0) and must NOT be treated as
    // Floor: they block like a wall and would otherwise pollute
    // navigation / foot-IK raycasts with phantom walkable surfaces.
    //
    // This partitions the triangle list by per-face normal:
    //   |normal.y| >= up_threshold  → walkable, stays in THIS mesh
    //   |normal.y| <  up_threshold  → vertical, moved into a new mesh
    // and returns that new mesh (category Wall, same material / object
    // name, BVH not yet built) holding the vertical faces.
    //
    // Returns nullptr when no split is warranted:
    //   - this mesh is not currently classified Floor, or
    //   - every face is walkable (no vertical faces to peel off).
    // Special case: if EVERY face is vertical, this mesh is re-tagged
    // Wall in place and nullptr is returned (no second mesh needed).
    //
    // After a split THIS mesh's bounds are recomputed and any existing
    // BVH is invalidated (bvh_root_ cleared, ready flag reset) so the
    // caller must (re)build BVHs afterwards.  up_threshold defaults to
    // 0.5 — i.e. any face leaning more than 45° from horizontal is
    // considered a wall.
    std::shared_ptr<CollisionMesh> splitOffVerticalFaces(
        float up_threshold = 0.5f);

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

    // Triangle count of the ORIGINAL source primitive this collision mesh
    // was built from, captured pre-weld/pre-decimate by
    // buildFromDrawablePrimitive.  Lets debug tooling report the
    // simplification ratio (simplified triangleCount() vs this).  Zero on
    // the aggregate build paths that don't track a single source primitive.
    size_t originalTriangleCount() const { return orig_tri_count_; }
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

    // Source-asset node name (NodeInfo::name_) of the first node
    // referencing this mesh.  Same population path as the world
    // transform — set by buildFromDrawablePrimitive only.
    std::string                 object_name_;

    // Source (drawable, mesh, primitive) this mesh was built from --
    // see sourceDrawable()/sourceMeshIdx()/sourcePrimIdx().  Raw pointer:
    // the scene DrawableObjects outlive the collision world (which is
    // rebuilt whenever the scenes change), so this never dangles in
    // practice; it is used only by debug draw on the main thread.
    const game_object::DrawableObject* src_drawable_ = nullptr;
    size_t                      src_mesh_idx_ = 0;
    size_t                      src_prim_idx_ = 0;

    // Original (pre-weld / pre-decimate) triangle count of the source
    // primitive -- see originalTriangleCount().
    size_t                      orig_tri_count_ = 0;

    // Semantic classification — see MeshCategory above.  Default
    // Unknown until buildFromDrawablePrimitive resolves it.
    MeshCategory                category_ = MeshCategory::Unknown;

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

    // Read-only access to a single mesh by world-list index, for the
    // isolate-debug slider (scrub one mesh at a time to find a broken
    // one).  Returns nullptr if the index is out of range.
    const CollisionMesh* meshAt(size_t i) const {
        return (i < meshes_.size()) ? meshes_[i].get() : nullptr;
    }

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
    // isolate_index >= 0 draws ONLY meshes_[isolate_index] (the
    // isolate-debug slider) so a single mesh can be inspected in
    // isolation; -1 (default) draws the whole world.
    void drawDebug(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors,
        int isolate_index = -1) const;

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
