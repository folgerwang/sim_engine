//
// cluster_mesh.cpp — build a parallel cluster sidecar for engine::helper::Mesh.
//
// Algorithm (dependency-free, single pass):
//
//   1. Build an edge -> {face_a, face_b} adjacency table by iterating every
//      face and recording its 3 edges (sorted endpoint pair as the key).
//      This gives us, for any face, its up-to-3 neighbour faces that share
//      a full edge.
//
//   2. Greedy BFS seeded from the lowest-index unassigned face:
//
//        seed = first unassigned face
//        frontier = { seed }
//        while cluster.face_count < max && frontier not empty:
//            pop a face
//            add to cluster
//            push its unassigned edge-neighbours
//
//      This produces spatially contiguous clusters — neighbouring triangles
//      end up in the same cluster — which is exactly what we want for
//      cluster-level culling (tight bounds + tight normal cones).
//
//   3. For each cluster, collect the unique vertex set, then compute:
//        - AABB over all vertex positions
//        - bounding sphere (AABB center + farthest-vertex radius)
//        - normal cone: axis = normalized average face normal,
//                       cutoff = min(dot(axis, face_normal)) across faces
//          (this is the conservative "accept every face" cone — cheap to
//           build, good enough for a first pass; meshoptimizer does a
//           tighter Welzl-style solve, which we can swap in later).
//
// Determinism: seed selection and neighbour traversal both walk in ascending
// index order, so the same input mesh produces the same clusters every run.
//
#include "helper/cluster_mesh.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>

namespace engine {
namespace helper {

namespace {

// Pack a sorted (lo, hi) vertex-index pair into a single 64-bit key so we
// can use a flat std::unordered_map<uint64_t, ...> for the edge table.
inline uint64_t makeEdgeKey(uint32_t a, uint32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

// Lightweight per-edge record: up to 2 adjacent faces. Non-manifold edges
// with >2 faces are still handled (we just drop the 3rd+ from the adjacency
// — those triangles will still get clustered via their other edges).
struct EdgeAdj {
    uint32_t face_a = UINT32_MAX;
    uint32_t face_b = UINT32_MAX;
    void add(uint32_t f) {
        if (face_a == UINT32_MAX)      face_a = f;
        else if (face_b == UINT32_MAX) face_b = f;
        // else: non-manifold, ignore.
    }
};

// Compute the (unnormalized) face normal. We keep it unnormalized for the
// weighted average (larger triangles should dominate the cone axis), then
// normalize at the end.
inline glm::vec3 faceNormalWeighted(const VertexStruct& v0,
                                    const VertexStruct& v1,
                                    const VertexStruct& v2) {
    return glm::cross(v1.position - v0.position,
                      v2.position - v0.position);
}

// ── Cluster-level BVH helpers ──────────────────────────────────────────────
// One entry per cluster fed into the BVH builder. We keep the cluster index
// (into ClusterMesh::clusters) plus the cluster's AABB and centroid so the
// split code doesn't have to reach back through the ClusterMesh reference
// for every partition pass.
struct ClusterBVHItem {
    uint32_t  cluster_index;
    AABB      bounds;
    glm::vec3 centroid;
};

// Recursive top-down builder. Splits `[begin, end)` of `items` in-place.
// Each recursion picks the longest centroid-extent axis and partitions
// around the median via std::nth_element — fast, deterministic, and gives
// balanced trees even on irregular geometry.
static std::shared_ptr<BVHNode> buildClusterBVHRecursive(
        ClusterBVHItem* items,
        uint32_t        begin,
        uint32_t        end,
        uint32_t        leaf_threshold,
        uint32_t        depth,
        uint32_t&       max_depth,
        uint32_t&       node_count,
        uint32_t&       leaf_count) {
    auto node = std::make_shared<BVHNode>();
    ++node_count;
    max_depth = std::max(max_depth, depth);

    // Union AABB of all items in this range.
    AABB node_bounds;
    for (uint32_t i = begin; i < end; ++i) {
        node_bounds.extend(items[i].bounds);
    }
    node->bounds = node_bounds;

    const uint32_t count = end - begin;
    if (count <= leaf_threshold) {
        node->primitive_ref_indices.reserve(count);
        for (uint32_t i = begin; i < end; ++i) {
            node->primitive_ref_indices.push_back(
                static_cast<int>(items[i].cluster_index));
        }
        ++leaf_count;
        return node;
    }

    // Compute centroid AABB so split picks a meaningful axis even when one
    // cluster has a huge bounds box that would dominate the node bounds.
    AABB centroid_bounds;
    for (uint32_t i = begin; i < end; ++i) {
        centroid_bounds.extend(items[i].centroid);
    }
    const glm::vec3 ext = centroid_bounds.getExtent();
    int axis = 0;
    if (ext.y > ext.x) axis = 1;
    if (ext.z > ((axis == 0) ? ext.x : ext.y)) axis = 2;

    // Degenerate case: all centroids coincide — fall back to a leaf.
    if (ext[axis] <= 0.0f) {
        node->primitive_ref_indices.reserve(count);
        for (uint32_t i = begin; i < end; ++i) {
            node->primitive_ref_indices.push_back(
                static_cast<int>(items[i].cluster_index));
        }
        ++leaf_count;
        return node;
    }

    // Median split via nth_element. O(n) partitioning, deterministic
    // because std::nth_element is stable under a strict-weak ordering on
    // centroid[axis] (ties broken by cluster_index).
    const uint32_t mid = begin + count / 2;
    std::nth_element(items + begin, items + mid, items + end,
        [axis](const ClusterBVHItem& a, const ClusterBVHItem& b) {
            if (a.centroid[axis] != b.centroid[axis])
                return a.centroid[axis] < b.centroid[axis];
            return a.cluster_index < b.cluster_index;
        });

    node->left  = buildClusterBVHRecursive(
        items, begin, mid, leaf_threshold, depth + 1,
        max_depth, node_count, leaf_count);
    node->right = buildClusterBVHRecursive(
        items, mid, end, leaf_threshold, depth + 1,
        max_depth, node_count, leaf_count);
    return node;
}

} // anonymous namespace

// ─── buildClusterBVH() ──────────────────────────────────────────────────────
// Public entry point. Seeds one ClusterBVHItem per cluster, hands the buffer
// to the recursive builder, and records aggregate stats on the ClusterMesh.
void buildClusterBVH(ClusterMesh& cm, uint32_t leaf_threshold) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    cm.cluster_bvh_root.reset();
    cm.cluster_bvh_node_count     = 0;
    cm.cluster_bvh_leaf_count     = 0;
    cm.cluster_bvh_depth          = 0;
    cm.cluster_bvh_build_time_ms  = 0.0;

    if (cm.clusters.empty()) {
        return;
    }
    if (leaf_threshold == 0) leaf_threshold = 4;

    std::vector<ClusterBVHItem> items;
    items.reserve(cm.clusters.size());
    for (uint32_t i = 0; i < cm.clusters.size(); ++i) {
        const MeshletCluster& cl = cm.clusters[i];
        ClusterBVHItem it;
        it.cluster_index = i;
        it.bounds        = AABB(cl.aabb_min, cl.aabb_max);
        // Prefer the precomputed sphere center (matches AABB center for the
        // current clusterer, but this is the "correct" choice if we ever
        // swap in a Welzl-style sphere solver with off-AABB-center spheres).
        it.centroid      = cl.bounds_center;
        items.push_back(it);
    }

    uint32_t max_depth  = 0;
    uint32_t node_count = 0;
    uint32_t leaf_count = 0;
    cm.cluster_bvh_root = buildClusterBVHRecursive(
        items.data(), 0, static_cast<uint32_t>(items.size()),
        leaf_threshold, /*depth=*/0,
        max_depth, node_count, leaf_count);

    cm.cluster_bvh_node_count = node_count;
    cm.cluster_bvh_leaf_count = leaf_count;
    cm.cluster_bvh_depth      = max_depth;

    const auto t_end = clock::now();
    cm.cluster_bvh_build_time_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::printf(
        "[CLUSTER] BVH: %u clusters -> %u nodes (%u leaves, depth=%u) in %.2f ms\n",
        static_cast<uint32_t>(cm.clusters.size()),
        cm.cluster_bvh_node_count,
        cm.cluster_bvh_leaf_count,
        cm.cluster_bvh_depth,
        cm.cluster_bvh_build_time_ms);
}

// ─── clusterRenderingEnabled() ──────────────────────────────────────────────
// One global toggle. We keep it as a function-local static so the storage
// lives in this TU and multiple callers share the same bool.
bool& clusterRenderingEnabled() {
    static bool enabled = false;  // OFF by default — purely a build step for now.
    return enabled;
}

bool& clusterIndirectActive() {
    static bool active = false;
    return active;
}

// ─── buildClusterMesh() ─────────────────────────────────────────────────────
void buildClusterMesh(const Mesh& mesh,
                      ClusterMesh& out,
                      uint32_t max_triangles_per_cluster) {
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    // Reset output in case it was reused.
    out.clusters.clear();
    out.source.reset();
    out.max_triangles_per_cluster_setting = max_triangles_per_cluster;
    out.total_triangles       = 0;
    out.total_clusters        = 0;
    out.min_tris_in_cluster   = 0;
    out.max_tris_in_cluster   = 0;
    out.avg_tris_in_cluster   = 0.0f;
    out.build_time_ms         = 0.0;

    if (!mesh.isValid() || mesh.getFaceCount() == 0) {
        std::printf("[CLUSTER] skipped: empty or invalid mesh\n");
        return;
    }

    const auto& verts = *mesh.vertex_data_ptr;
    const auto& faces = *mesh.faces_ptr;
    const uint32_t face_count = static_cast<uint32_t>(faces.size());

    if (max_triangles_per_cluster == 0) max_triangles_per_cluster = 128;

    // ── 1) edge -> adjacent faces ──────────────────────────────────────────
    std::unordered_map<uint64_t, EdgeAdj> edge_map;
    edge_map.reserve(face_count * 2);  // ~3 edges/face, each shared by 2 faces
    for (uint32_t f = 0; f < face_count; ++f) {
        const Face& face = faces[f];
        if (face.isDegenerate()) continue;
        for (int e = 0; e < 3; ++e) {
            const uint32_t a = face.v_indices[e];
            const uint32_t b = face.v_indices[(e + 1) % 3];
            edge_map[makeEdgeKey(a, b)].add(f);
        }
    }

    // ── 2) greedy BFS clustering ───────────────────────────────────────────
    std::vector<uint8_t> assigned(face_count, 0u);
    std::vector<uint32_t> neighbour_scratch;  // reused per-face
    neighbour_scratch.reserve(3);

    // Pre-size the cluster list conservatively.
    out.clusters.reserve((face_count + max_triangles_per_cluster - 1) /
                         max_triangles_per_cluster);

    for (uint32_t seed = 0; seed < face_count; ++seed) {
        if (assigned[seed]) continue;
        if (faces[seed].isDegenerate()) {
            assigned[seed] = 1;  // silently skip degenerate triangles
            continue;
        }

        out.clusters.emplace_back();
        MeshletCluster& cl = out.clusters.back();
        cl.face_indices.reserve(max_triangles_per_cluster);

        // BFS frontier. We use a plain queue — the traversal order is
        // ascending because we push neighbours in ascending face-index
        // order after a small sort, which keeps builds deterministic.
        std::queue<uint32_t> frontier;
        frontier.push(seed);
        assigned[seed] = 1;

        while (!frontier.empty() &&
               cl.face_indices.size() < max_triangles_per_cluster) {
            const uint32_t f = frontier.front();
            frontier.pop();
            cl.face_indices.push_back(f);

            // Collect neighbours across the 3 edges, dedup, sort, then push.
            neighbour_scratch.clear();
            const Face& ff = faces[f];
            for (int e = 0; e < 3; ++e) {
                const uint32_t a = ff.v_indices[e];
                const uint32_t b = ff.v_indices[(e + 1) % 3];
                auto it = edge_map.find(makeEdgeKey(a, b));
                if (it == edge_map.end()) continue;
                const EdgeAdj& ea = it->second;
                for (uint32_t n : {ea.face_a, ea.face_b}) {
                    if (n == UINT32_MAX) continue;
                    if (n == f)          continue;
                    if (assigned[n])     continue;
                    if (faces[n].isDegenerate()) continue;
                    neighbour_scratch.push_back(n);
                }
            }
            std::sort(neighbour_scratch.begin(), neighbour_scratch.end());
            neighbour_scratch.erase(
                std::unique(neighbour_scratch.begin(), neighbour_scratch.end()),
                neighbour_scratch.end());

            for (uint32_t n : neighbour_scratch) {
                if (assigned[n]) continue;  // may have been claimed in-loop
                if (cl.face_indices.size() + frontier.size() >=
                    max_triangles_per_cluster) {
                    // Enough candidates queued to saturate this cluster.
                    break;
                }
                assigned[n] = 1;
                frontier.push(n);
            }
        }
    }

    // ── 3) per-cluster geometry payload (bounds + normal cone) ─────────────
    for (MeshletCluster& cl : out.clusters) {
        // Gather unique vertices.
        // Use a small local set — clusters are tiny (<=128 tris, <=~256 verts).
        std::vector<uint32_t>& vidx = cl.vertex_indices;
        vidx.clear();
        vidx.reserve(cl.face_indices.size() * 3);
        for (uint32_t f : cl.face_indices) {
            const Face& face = faces[f];
            vidx.push_back(face.v_indices[0]);
            vidx.push_back(face.v_indices[1]);
            vidx.push_back(face.v_indices[2]);
        }
        std::sort(vidx.begin(), vidx.end());
        vidx.erase(std::unique(vidx.begin(), vidx.end()), vidx.end());

        // AABB.
        glm::vec3 mn( std::numeric_limits<float>::max());
        glm::vec3 mx(-std::numeric_limits<float>::max());
        for (uint32_t vi : vidx) {
            const glm::vec3& p = verts[vi].position;
            mn = glm::min(mn, p);
            mx = glm::max(mx, p);
        }
        cl.aabb_min = mn;
        cl.aabb_max = mx;

        // Bounding sphere: center = AABB center; radius = farthest vertex.
        const glm::vec3 center = 0.5f * (mn + mx);
        float r2 = 0.0f;
        for (uint32_t vi : vidx) {
            const glm::vec3 d = verts[vi].position - center;
            r2 = std::max(r2, glm::dot(d, d));
        }
        cl.bounds_center = center;
        cl.bounds_radius = std::sqrt(r2);

        // Normal cone.
        glm::vec3 sum_n(0.0f);
        for (uint32_t f : cl.face_indices) {
            const Face& face = faces[f];
            sum_n += faceNormalWeighted(verts[face.v_indices[0]],
                                        verts[face.v_indices[1]],
                                        verts[face.v_indices[2]]);
        }
        const float axis_len2 = glm::dot(sum_n, sum_n);
        if (axis_len2 > 1e-12f) {
            cl.cone_axis = sum_n / std::sqrt(axis_len2);

            // Cutoff = min dot(axis, per-face unit normal). If any face
            // disagrees too much (triangle-pair clusters at sharp creases),
            // this may go negative — in which case back-face culling the
            // cluster is unsafe and we fall back to "don't cull".
            float min_dot = 1.0f;
            for (uint32_t f : cl.face_indices) {
                const Face& face = faces[f];
                glm::vec3 n = faceNormalWeighted(verts[face.v_indices[0]],
                                                 verts[face.v_indices[1]],
                                                 verts[face.v_indices[2]]);
                const float l2 = glm::dot(n, n);
                if (l2 <= 1e-20f) continue;  // skip degenerate
                n /= std::sqrt(l2);
                min_dot = std::min(min_dot, glm::dot(cl.cone_axis, n));
            }
            // Safety margin: if any face normal makes more than ~90° with the
            // axis (min_dot <= 0), the cone isn't useful for back-face
            // culling — disable it for this cluster.
            cl.cone_cutoff = (min_dot > 0.0f) ? min_dot : -1.0f;
        } else {
            // Degenerate cluster (e.g. all triangles collinear).
            cl.cone_axis   = glm::vec3(0.0f, 0.0f, 1.0f);
            cl.cone_cutoff = -1.0f;
        }
    }

    // ── 4) aggregate stats ─────────────────────────────────────────────────
    // Store an owning copy of the Mesh struct. The Mesh itself is lightweight
    // (two shared_ptrs) so the copy is cheap — it just bumps the ref counts
    // on vertex_data_ptr and faces_ptr, keeping the underlying data alive
    // as long as this ClusterMesh exists. The previous non-owning alias
    // (no-op deleter) dangled when the source Mesh was a stack local.
    out.source = std::make_shared<const Mesh>(mesh);

    out.total_clusters  = static_cast<uint32_t>(out.clusters.size());
    uint32_t sum_tris = 0;
    uint32_t min_tris = out.total_clusters ? UINT32_MAX : 0;
    uint32_t max_tris = 0;
    for (const MeshletCluster& cl : out.clusters) {
        const uint32_t n = cl.triangleCount();
        sum_tris += n;
        min_tris = std::min(min_tris, n);
        max_tris = std::max(max_tris, n);
    }
    out.total_triangles     = sum_tris;
    out.min_tris_in_cluster = min_tris == UINT32_MAX ? 0 : min_tris;
    out.max_tris_in_cluster = max_tris;
    out.avg_tris_in_cluster = out.total_clusters
        ? static_cast<float>(sum_tris) / static_cast<float>(out.total_clusters)
        : 0.0f;

    const auto t_end = clock::now();
    out.build_time_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::printf(
        "[CLUSTER] built: %u tris -> %u clusters "
        "(min=%u max=%u avg=%.1f, cap=%u) in %.2f ms\n",
        out.total_triangles,
        out.total_clusters,
        out.min_tris_in_cluster,
        out.max_tris_in_cluster,
        out.avg_tris_in_cluster,
        out.max_triangles_per_cluster_setting,
        out.build_time_ms);

    // ── 5) cluster-level BVH ───────────────────────────────────────────────
    // Builds a binary tree over `out.clusters` for future hierarchical
    // culling (frustum / occlusion / LOD). Safe no-op on empty clusters.
    buildClusterBVH(out);
}

}  // namespace helper
}  // namespace engine
