#pragma once
//
// cluster_mesh.h — "Nanite-lite" step 1: cluster sidecar data.
//
// This file defines a parallel, opt-in representation of a mesh that splits
// its triangles into small contiguous clusters (default cap: 128 tris each).
// It coexists with the existing engine::helper::Mesh — neither is modified,
// neither depends on the other once built.
//
// Usage pattern:
//
//     engine::helper::Mesh         mesh  = <loaded from gltf/obj/procgen>;
//     engine::helper::ClusterMesh  clust;
//     engine::helper::buildClusterMesh(mesh, clust /*, max_tris=128 */);
//
//     // Both `mesh` and `clust` can live side-by-side. Rendering code picks
//     // one based on engine::helper::clusterRenderingEnabled().
//
// No rendering change is wired up yet — this step only produces the data and
// exposes the toggle. Next step will consume `ClusterMesh` in a new draw
// path (mesh-shader / GPU-driven) behind the toggle.
//
#include <cstdint>
#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "helper/mesh_tool.h"   // engine::helper::Mesh, Face, VertexStruct
#include "helper/bvh.h"         // engine::helper::BVHNode, AABB

namespace engine {
namespace helper {

// ─── Per-cluster data ────────────────────────────────────────────────────────
// One cluster = up to `max_triangles_per_cluster` contiguous triangles from
// the source Mesh. `face_indices` are positions into Mesh::faces_ptr so the
// cluster doesn't duplicate geometry — switching on the cluster path is a
// pure bookkeeping change, not a geometry copy.
struct MeshletCluster {
    // Indices into the source Mesh's face array.
    std::vector<uint32_t> face_indices;

    // Unique vertex indices referenced by this cluster's faces. Useful for
    // the eventual mesh-shader path where each meshlet has its own small
    // vertex list (typical HW limit: 64 or 128 verts/meshlet).
    std::vector<uint32_t> vertex_indices;

    // ── Culling payload (kept CPU-side now, will be promoted to GPU later) ──
    // Bounding sphere of the cluster's vertex positions.
    glm::vec3 bounds_center = glm::vec3(0.0f);
    float     bounds_radius = 0.0f;
    // AABB.
    glm::vec3 aabb_min = glm::vec3( std::numeric_limits<float>::max());
    glm::vec3 aabb_max = glm::vec3(-std::numeric_limits<float>::max());
    // Normal cone for back-face culling whole clusters at once:
    //   axis    — averaged face normal (unit length)
    //   cutoff  — cos(max angle between axis and any face normal in cluster).
    // Cull rule: if dot(view_dir_from_cluster, cone_axis) >= cone_cutoff,
    // every triangle faces away from the camera and the cluster is skippable.
    glm::vec3 cone_axis   = glm::vec3(0.0f, 0.0f, 1.0f);
    float     cone_cutoff = -1.0f;  // -1 = "don't cull" (safe default)

    uint32_t triangleCount() const {
        return static_cast<uint32_t>(face_indices.size());
    }
    uint32_t vertexCount() const {
        return static_cast<uint32_t>(vertex_indices.size());
    }
};

// ─── Top-level cluster sidecar ──────────────────────────────────────────────
// Owned parallel to the Mesh it was built from. Holds only CPU-side data
// right now; GPU-side mirror (meshlet buffer, cluster BVH) lands in a later
// step once the render path consumes this.
struct ClusterMesh {
    // Weak back-reference to the source geometry. Cluster.face_indices index
    // into source->faces_ptr, so dropping the source would invalidate this.
    std::shared_ptr<const Mesh> source;

    std::vector<MeshletCluster> clusters;

    // ── Aggregate stats (handy for the HUD + debugging) ──
    uint32_t max_triangles_per_cluster_setting = 128;
    uint32_t total_triangles   = 0;
    uint32_t total_clusters    = 0;
    uint32_t min_tris_in_cluster = 0;
    uint32_t max_tris_in_cluster = 0;
    float    avg_tris_in_cluster = 0.0f;
    double   build_time_ms      = 0.0;

    // ── Cluster-level BVH ────────────────────────────────────────────────
    // Top-down binary tree over `clusters` (NOT over triangles). Each leaf
    // stores cluster indices in BVHNode::primitive_ref_indices, and each
    // internal node's AABB is the union of its children. This is the data
    // structure a GPU-driven culling pass will walk to skip whole subtrees
    // of meshlets at once (frustum / occlusion / LOD selection).
    //
    // Built at the end of buildClusterMesh; null if clusters is empty.
    std::shared_ptr<BVHNode> cluster_bvh_root;
    uint32_t cluster_bvh_node_count  = 0;
    uint32_t cluster_bvh_leaf_count  = 0;
    uint32_t cluster_bvh_depth       = 0;
    double   cluster_bvh_build_time_ms = 0.0;

    bool empty() const { return clusters.empty(); }
};

// ─── Build ──────────────────────────────────────────────────────────────────
// Partitions `mesh` into clusters of up to `max_triangles_per_cluster`
// triangles via greedy edge-adjacency BFS. Deterministic given identical
// input. Does NOT modify `mesh`.
//
// The current implementation is dependency-free (pure std::). We will swap
// in meshoptimizer's meshlet builder in a later step for better locality
// and normal-cone quality, but the API here won't change.
void buildClusterMesh(const Mesh& mesh,
                      ClusterMesh& out_clusters,
                      uint32_t max_triangles_per_cluster = 128);

// ─── Cluster BVH build ──────────────────────────────────────────────────────
// Builds a top-down binary BVH over the cluster list already present in
// `cm.clusters`. Uses median split on the longest centroid-extent axis
// (std::nth_element) and stores cluster indices in leaf nodes.
//
// `leaf_threshold` = max clusters per leaf before we stop splitting (small
// values give deeper trees with tighter bounds; 4 is a reasonable default
// for CPU traversal, GPU traversal will want smaller leaves).
//
// Invoked automatically at the end of buildClusterMesh(); also safe to call
// directly if callers rebuild cluster data through a different path.
void buildClusterBVH(ClusterMesh& cm, uint32_t leaf_threshold = 4);

// ─── Runtime switch ─────────────────────────────────────────────────────────
// Global flag that future draw-path code will consult. Defaults to OFF so
// that turning on the clustering build has no visible rendering change until
// the new path is wired up. Expose in ImGui debug panel / cvars as desired.
bool& clusterRenderingEnabled();

// True when the GPU indirect cluster draw is active.  When set, the forward
// pass skips clustered meshes entirely — the cluster renderer handles them.
bool& clusterIndirectActive();

}  // namespace helper
}  // namespace engine
