#pragma once
#include <algorithm>
#include <iostream>
#include "renderer/renderer_structs.h"

namespace engine {
namespace helper {

struct AABB {
    glm::vec3 min_bounds;
    glm::vec3 max_bounds;

    AABB() : min_bounds(std::numeric_limits<float>::max()),
        max_bounds(-std::numeric_limits<float>::max()) {
    }

    AABB(const glm::vec3& p) {
        min_bounds = p;
        max_bounds = p;
    }

    AABB(const glm::vec3& v_min, const glm::vec3& v_max) {
        min_bounds = min(v_min, v_max);
        max_bounds = max(v_min, v_max);
    }

    void extend(const glm::vec3& p) {
        min_bounds = min(min_bounds, p);
        max_bounds = max(max_bounds, p);
    }

    void extend(const AABB& other) {
        extend(other.min_bounds);
        extend(other.max_bounds);
    }

    float surfaceArea() const {
        glm::vec3 d = max_bounds - min_bounds;
        if (d.x < 0 || d.y < 0 || d.z < 0) return 0.0f;
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    glm::vec3 centroid() const {
        return (min_bounds + max_bounds) * 0.5f;
    }
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction; };

inline bool rayAABBIntersect(const Ray& ray, const AABB& box, float& t_near_hit, float& t_far_hit) {
    // It's often useful to compute 1.0f / ray.direction once
    glm::vec3 inv_dir =
        { 1.0f / ray.direction.x,
          1.0f / ray.direction.y,
          1.0f / ray.direction.z };

    float tx1 = (box.min_bounds.x - ray.origin.x) * inv_dir.x;
    float tx2 = (box.max_bounds.x - ray.origin.x) * inv_dir.x;

    float tmin = std::min(tx1, tx2);
    float tmax = std::max(tx1, tx2);

    float ty1 = (box.min_bounds.y - ray.origin.y) * inv_dir.y;
    float ty2 = (box.max_bounds.y - ray.origin.y) * inv_dir.y;

    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));

    float tz1 = (box.min_bounds.z - ray.origin.z) * inv_dir.z;
    float tz2 = (box.max_bounds.z - ray.origin.z) * inv_dir.z;

    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));

    // t_near_hit and t_far_hit are the intersection distances with the AABB itself
    t_near_hit = tmin;
    t_far_hit = tmax;

    // The ray intersects if tmax >= tmin AND tmax > 0 (intersection is in front)
    // We also need to consider the ray's maximum travel distance if it has one.
    return tmax >= tmin && tmax > 1e-6f; // tmax > 0 ensures it's in front
}

/**
 * @brief Checks if a ray intersects a triangle using the Möller–Trumbore algorithm.
 *
 * @param ray The ray to test.
 * @param v0 The first vertex of the triangle.
 * @param v1 The second vertex of the triangle.
 * @param v2 The third vertex of the triangle.
 * @param out_t Output parameter for the distance 't' along the ray to the intersection point.
 * @param out_u Output parameter for the barycentric coordinate 'u'.
 * @param out_v Output parameter for the barycentric coordinate 'v'.
 * @return true if the ray intersects the triangle within its bounds, false otherwise.
 */
inline bool rayTriangleIntersect(
    const Ray& ray,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float& out_t,        // Output: distance t along the ray
    float& out_u,        // Output: barycentric u
    float& out_v         // Output: barycentric v
) {
    // A small epsilon value for floating-point comparisons to avoid issues with
    // rays parallel to the triangle plane or hitting edges precisely.
    const float EPSILON = 1e-6f; // Adjust if needed based on your scene's scale

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    // Calculate determinant part 1: P = D x E2
    glm::vec3 pvec = cross(ray.direction, edge2);

    // Calculate determinant: det = E1 · P
    // If determinant is near zero, ray lies in plane of triangle or is parallel.
    float det = dot(edge1, pvec);

    // Culling check (optional): if determinant is negative, triangle is back-facing.
    // if (det < EPSILON) return false; // For culling back-faces
    // For double-sided intersection (no back-face culling):
    if (std::fabs(det) < EPSILON) { // Use fabs for double-sided
        return false; // Ray is parallel to the triangle or lies in its plane (missing)
    }

    float inv_det = 1.0f / det;

    // Calculate distance from V0 to ray origin: T = O - V0
    glm::vec3 tvec = ray.origin - v0;

    // Calculate u parameter and test bound: u = (T · P) * inv_det
    out_u = dot(tvec, pvec) * inv_det;
    if (out_u < 0.0f || out_u > 1.0f) {
        return false; // Intersection point is outside the V0-V1-V2_edge1 side
    }

    // Prepare to test v parameter: Q = T x E1
    glm::vec3 qvec = cross(tvec, edge1);

    // Calculate v parameter and test bound: v = (D · Q) * inv_det
    out_v = dot(ray.direction, qvec) * inv_det;
    if (out_v < 0.0f || out_u + out_v > 1.0f) {
        return false; // Intersection point is outside the V0-V1-V2_edge2 side or V1-V2 edge
    }

    // Calculate t, the distance along the ray to the intersection point: t = (E2 · Q) * inv_det
    out_t = dot(edge2, qvec) * inv_det;

    // Check if the intersection point is in front of the ray's origin
    // (and not practically at the origin if t is extremely small).
    if (out_t > EPSILON) {
        return true; // Ray intersects the triangle
    }
    else {
        // Line intersection but not a ray intersection (behind the origin)
        // or too close to be considered a valid hit.
        return false;
    }
}

// This structure holds precomputed info for each triangle
struct PrimitiveInfo {
    // Index i, for triangle (indices[3*i], indices[3*i+1], indices[3*i+2])
    int original_triangle_index;
    AABB bounds;
    glm::vec3 centroid_val;
    std::vector<glm::vec3> vertex_list;

    PrimitiveInfo(
        int tri_idx,
        const std::vector<glm::vec3>& v_list)
        : original_triangle_index(tri_idx),
          vertex_list(v_list){
        glm::vec3 sum_v(0);
        for (const auto& v : v_list) {
            bounds.extend(v);
            sum_v += v;
        }
        centroid_val = sum_v / float_t(v_list.size());
    }

    std::vector<glm::vec3> splitWithPlane(
        const glm::vec4& clip_plane,
        float clip_back = 1.0f) const {
        const uint32_t num_vertex = uint32_t(vertex_list.size());
        std::vector<float_t> dist_to_plane(num_vertex);
        for (uint32_t i = 0; i < num_vertex; i++) {
            dist_to_plane[i] =
                clip_back * dot(glm::vec4(vertex_list[i], 1.0f), clip_plane);
        }

        std::vector<glm::vec3> clipped_vertex_list;
        clipped_vertex_list.reserve(9);
        auto v0 = vertex_list[num_vertex - 1];
        auto d0 = dist_to_plane[num_vertex - 1];
        for (uint32_t i = 0; i < num_vertex; i++) {
            auto v1 = vertex_list[i];
            auto d1 = dist_to_plane[i];

            if (d1 > 0) {
                if (d0 > 0) {
                    clipped_vertex_list.push_back(v1);
                }
                else if (d0 < 0) {
                    float t = -d0 / (d1 - d0);
                    clipped_vertex_list.push_back(v0 + t * (v1 - v0));
                    clipped_vertex_list.push_back(v1);
                }
            }
            else if (d1 < 0) {
                if (d0 > 0) {
                    float t = -d0 / (d1 - d0);
                    clipped_vertex_list.push_back(v0 + t * (v1 - v0));
                }
            }
            else {
                clipped_vertex_list.push_back(v1);
            }

            v0 = v1;
            d0 = d1;
        }

        return clipped_vertex_list;
    }

    const AABB& getAABB() const { return bounds; }

    const glm::vec3& centroid() const { return centroid_val; }
};

struct HitInfo {
    bool hit = false;
    float t = std::numeric_limits<float>::max(); // Distance to intersection
    int triangle_index = -1;                     // Original index of the hit triangle
    float u, v;                                  // Barycentric coordinates
    glm::vec3 intersection_point; // Can be computed from ray.origin + ray.direction * t
};

struct BVHNode {
    AABB bounds;
    std::shared_ptr<BVHNode> left;
    std::shared_ptr<BVHNode> right;
    // Stores indices into the 'primitive_info_list' (see BVHBuilder)
    // OR directly stores 'original_triangle_index' from PrimitiveInfo
    std::vector<int> primitive_ref_indices;

    bool isLeaf() const {
        return left == nullptr && right == nullptr;
    }
};

// Forward declaration if rayTriangleIntersect is in another file/scope
// bool rayTriangleIntersect(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, float& out_t, float& out_u, float& out_v);

inline void intersectBVHRecursive(
    const Ray& ray,
    const std::shared_ptr<BVHNode>& node,
    const std::vector<glm::vec3>& vertices,      // Your global vertex stream
    const std::vector<int>& indices,        // Your global index stream
    HitInfo& closest_hit                     // Passed by reference to update
) {
    if (!node) {
        return;
    }

    // 1. Check if ray intersects the current node's AABB
    float t_aabb_near, t_aabb_far; // We might not need t_aabb_far here for basic traversal
    if (!rayAABBIntersect(ray, node->bounds, t_aabb_near, t_aabb_far)) {
        return; // Ray misses this node's entire volume
    }

    // Optimization: If the closest possible hit within this AABB (t_aabb_near)
    // is already further than a known hit, we can prune this branch.
    if (t_aabb_near >= closest_hit.t) {
        return;
    }

    // 2. If it's a Leaf Node
    if (node->isLeaf()) {
        for (int original_tri_idx : node->primitive_ref_indices) {
            // Get triangle vertices using original_tri_idx
            const glm::vec3& v0 = vertices[indices[3 * original_tri_idx + 0]];
            const glm::vec3& v1 = vertices[indices[3 * original_tri_idx + 1]];
            const glm::vec3& v2 = vertices[indices[3 * original_tri_idx + 2]];

            float current_t, current_u, current_v;
            if (rayTriangleIntersect(ray, v0, v1, v2, current_t, current_u, current_v)) {
                if (current_t < closest_hit.t && current_t > 1e-6f) { // Check if this hit is closer
                    closest_hit.hit = true;
                    closest_hit.t = current_t;
                    closest_hit.triangle_index = original_tri_idx;
                    closest_hit.u = current_u;
                    closest_hit.v = current_v;
                }
            }
        }
    }
    // 3. If it's an Internal Node
    else {
        // Recursively check children.
        // Optional Optimization: Check the child whose AABB is closer to the ray first.
        // This can lead to finding a closer hit sooner, allowing more pruning.
        float t_left_near, t_left_far, t_right_near, t_right_far;
        bool hit_left_aabb = node->left ? rayAABBIntersect(ray, node->left->bounds, t_left_near, t_left_far) : false;
        bool hit_right_aabb = node->right ? rayAABBIntersect(ray, node->right->bounds, t_right_near, t_right_far) : false;

        if (hit_left_aabb && hit_right_aabb) {
            if (t_left_near < t_right_near) {
                intersectBVHRecursive(ray, node->left, vertices, indices, closest_hit);
                // Only check right if it can still contain a closer hit
                if (t_right_near < closest_hit.t) {
                    intersectBVHRecursive(ray, node->right, vertices, indices, closest_hit);
                }
            }
            else {
                intersectBVHRecursive(ray, node->right, vertices, indices, closest_hit);
                // Only check left if it can still contain a closer hit
                if (t_left_near < closest_hit.t) {
                    intersectBVHRecursive(ray, node->left, vertices, indices, closest_hit);
                }
            }
        }
        else if (hit_left_aabb) {
            if (t_left_near < closest_hit.t) { // Only check if it can contain a closer hit
                intersectBVHRecursive(ray, node->left, vertices, indices, closest_hit);
            }
        }
        else if (hit_right_aabb) {
            if (t_right_near < closest_hit.t) { // Only check if it can contain a closer hit
                intersectBVHRecursive(ray, node->right, vertices, indices, closest_hit);
            }
        }
    }
}

class BVHBuilder {
public:
    std::shared_ptr<BVHNode> root_;
    std::vector<PrimitiveInfo> primitive_info_list_; // Stores AABB & centroid for each tri
    std::vector<int> build_indices_; // Indices into 'primitive_info_list' used during build

    const int MAX_PRIMS_IN_NODE = 4;
    const int SAH_BUCKET_COUNT = 12;

    BVHBuilder(
        const std::vector<glm::vec3>& vertices,
        const std::vector<int>& indices) {
        if (indices.empty() || vertices.empty()) return;

        size_t num_triangles = indices.size() / 3;
        primitive_info_list_.reserve(num_triangles);

        for (size_t i = 0; i < num_triangles; ++i) {
            primitive_info_list_.emplace_back(
                PrimitiveInfo(static_cast<int>(i),
                    {
                        vertices[indices[3 * i + 0]],
                        vertices[indices[3 * i + 1]],
                        vertices[indices[3 * i + 2]] }));
        }

        build_indices_.resize(primitive_info_list_.size());
        for (size_t i = 0; i < primitive_info_list_.size(); ++i) {
            build_indices_[i] = static_cast<int>(i);
        }

        root_ = buildRecursive(0, build_indices_);
    }

    ~BVHBuilder() {
        deleteNodeRecursive(root_);
    }

private:
    void deleteNodeRecursive(std::shared_ptr<BVHNode>& node) {
        if (!node) return;
        deleteNodeRecursive(node->left);
        deleteNodeRecursive(node->right);
        node = nullptr;
    }

    std::shared_ptr<BVHNode> buildRecursive(int level, const std::vector<int>& build_indices) {
        std::shared_ptr<BVHNode> node = std::make_shared<BVHNode>();
        int32_t num_primitives_in_node = int32_t(build_indices.size());

        // Calculate bounds for all primitives in this node
        for (int i = 0; i < num_primitives_in_node; ++i) {
            // build_indices[i] is an index into primitive_info_list
            node->bounds.extend(primitive_info_list_[build_indices[i]].getAABB());
        }

        // Base case: if few primitives, create a leaf node
        if (num_primitives_in_node <= MAX_PRIMS_IN_NODE) {
            for (int i = 0; i < num_primitives_in_node; ++i) {
                // Store the original triangle index directly, or the index into primitive_info_list
                node->primitive_ref_indices.push_back(
                    primitive_info_list_[build_indices[i]].original_triangle_index);
            }
            return node;
        }

        // --- Find the best split (SAH or other heuristic) ---
        AABB centroid_bounds;
        for (int i = 0; i < num_primitives_in_node; ++i) {
            centroid_bounds.extend(
                primitive_info_list_[build_indices[i]].centroid());
        }

        int split_axis = 0;
        glm::vec3 extent =
            centroid_bounds.max_bounds -
            centroid_bounds.min_bounds;

        if (extent.y > extent.x && extent.y > extent.z) split_axis = 1;
        else if (extent.z > extent.x && extent.z > extent.y) split_axis = 2;

        float split_coord_val;
        bool use_midpoint_fallback = false;
        if ((split_axis == 0 && extent.x < 1e-5f) ||
            (split_axis == 1 && extent.y < 1e-5f) ||
            (split_axis == 2 && extent.z < 1e-5f)) {
            use_midpoint_fallback = true;
        }

        glm::vec4 clip_plane(0);
        clip_plane[split_axis] = 1.0f;
        float split_ratio = 0.0f;

        if (use_midpoint_fallback || num_primitives_in_node < SAH_BUCKET_COUNT * 2) {
            split_coord_val = centroid_bounds.centroid()[split_axis];
        }
        else {
            // --- SAH Bucketing (Simplified) ---
            struct BucketInfo {
                int count = 0;
                AABB bounds;
            };
            std::vector<BucketInfo> buckets(SAH_BUCKET_COUNT);
            float min_c_axis = centroid_bounds.min_bounds[split_axis];
            float max_c_axis = centroid_bounds.max_bounds[split_axis];
            float extent_c_axis = max_c_axis - min_c_axis;

            if (extent_c_axis < 1e-6f) { // All centroids are basically at the same spot on this axis
                // Fallback if bucketing is not possible
                split_coord_val = min_c_axis; // or centroid_bounds.centroid().coord(split_axis)
            }
            else {
                for (int i = 0; i < num_primitives_in_node; ++i) {
                    const PrimitiveInfo& p_info = primitive_info_list_[build_indices[i]];
                    float centroid_coord = p_info.centroid()[split_axis];
                    int b = static_cast<int>(SAH_BUCKET_COUNT * ((centroid_coord - min_c_axis) / extent_c_axis));
                    b = std::max(0, std::min(SAH_BUCKET_COUNT - 1, b));
                    buckets[b].count++;
                    buckets[b].bounds.extend(p_info.getAABB());
                }

                std::vector<float> cost(SAH_BUCKET_COUNT - 1);
                for (int i = 0; i < SAH_BUCKET_COUNT - 1; ++i) {
                    AABB b0, b1;
                    int count0 = 0, count1 = 0;
                    for (int j = 0; j <= i; ++j) {
                        if (buckets[j].count > 0) {
                            b0.extend(buckets[j].bounds);
                            count0 += buckets[j].count;
                        }
                    }
                    for (int j = i + 1; j < SAH_BUCKET_COUNT; ++j) {
                        if (buckets[j].count) {
                            b1.extend(buckets[j].bounds);
                            count1 += buckets[j].count;
                        }
                    }
                    cost[i] = 0.125f + ((count0 > 0 ? count0 * b0.surfaceArea() : 0.0f) +
                        (count1 > 0 ? count1 * b1.surfaceArea() : 0.0f))
                        / node->bounds.surfaceArea();
                    if (count0 == 0 || count1 == 0) cost[i] = std::numeric_limits<float>::max(); // Penalize empty splits
                }

                float min_cost = std::numeric_limits<float>::max();
                int min_cost_split_bucket = 0;
                for (int i = 0; i < SAH_BUCKET_COUNT - 1; ++i) {
                    if (cost[i] < min_cost) {
                        min_cost = cost[i];
                        min_cost_split_bucket = i;
                    }
                }

                float leaf_cost = static_cast<float>(num_primitives_in_node);
                if (min_cost < leaf_cost && min_cost != std::numeric_limits<float>::max()) {
                    split_ratio = float_t(min_cost_split_bucket + 1) / float_t(SAH_BUCKET_COUNT);
                    split_coord_val = min_c_axis + extent_c_axis * split_ratio;
                    clip_plane[3] = -split_coord_val;
                }
                else {
                    for (int i = 0; i < num_primitives_in_node; ++i) {
                        node->primitive_ref_indices.push_back(
                            primitive_info_list_[build_indices[i]].original_triangle_index);
                    }
                    return node; // Make leaf
                }
            }
        }

        std::shared_ptr<std::vector<int32_t>> build_indices_left, build_indices_right;
        build_indices_left = std::make_shared<std::vector<int32_t>>();
        build_indices_right = std::make_shared<std::vector<int32_t>>();
        
        build_indices_left->reserve(num_primitives_in_node);
        build_indices_right->reserve(num_primitives_in_node);
        for (int32_t i = 0; i < num_primitives_in_node; i++) {
            const auto prim = primitive_info_list_[build_indices[i]];
            const auto& aabb = prim.getAABB();

            if (split_coord_val >= aabb.max_bounds[split_axis]) {
                build_indices_left->push_back(build_indices[i]);
            }
            else if (split_coord_val <= aabb.min_bounds[split_axis]) {
                build_indices_right->push_back(build_indices[i]);
            }
            else {
                auto left_tri_list = prim.splitWithPlane(clip_plane, -1.0f);
                if (left_tri_list.size() > 2) {
                    build_indices_left->push_back(static_cast<int32_t>(primitive_info_list_.size()));
                    primitive_info_list_.push_back(
                        PrimitiveInfo(prim.original_triangle_index, left_tri_list));
                }
                auto right_tri_list = prim.splitWithPlane(clip_plane, 1.0f);
                if (right_tri_list.size() > 2) {
                    build_indices_right->push_back(static_cast<int32_t>(primitive_info_list_.size()));
                    primitive_info_list_.push_back(
                        PrimitiveInfo(prim.original_triangle_index, right_tri_list));
                }
            }
        }

        std::cout << "level: " << level << ", split ratio: " << split_ratio << ", left: " << build_indices_left->size() << ", right: " << build_indices_right->size() << std::endl;

        node->left = buildRecursive(level + 1, *build_indices_left);
        node->right = buildRecursive(level + 1, *build_indices_right);

        return node;
    }

public:
    void printBVH(std::shared_ptr<BVHNode>& node, int depth = 0) const {
        if (node == nullptr) return;
        for (int i = 0; i < depth; ++i) std::cout << "  ";
        std::cout << "Node (Refs: " << node->primitive_ref_indices.size();
        if (!node->isLeaf()) {
            std::cout << ", SA: " << node->bounds.surfaceArea();
        }
        std::cout << ")" << std::endl;

        if (!node->isLeaf()) {
            printBVH(node->left, depth + 1);
            printBVH(node->right, depth + 1);
        }
        else {
            for (int i = 0; i < depth + 1; ++i)
                std::cout << "  ";
            std::cout << "Leaf Original Triangle Indices: ";
            for (int original_idx : node->primitive_ref_indices)
                std::cout << original_idx << " ";
            std::cout << std::endl;
        }
    }

    std::shared_ptr<BVHNode> getBvhNodeRoot() {
        return root_;
    }
};

// --- Main function to start the process ---
inline HitInfo findClosestHit(
    const Ray& ray,
    const BVHBuilder& bvh, // Assuming BVHBuilder holds the root and prim info
    const std::vector<glm::vec3>& vertices,
    const std::vector<int>& indices) {
    HitInfo closest_hit; // Initialized with hit = false, t = infinity
    if (bvh.root_) {
        intersectBVHRecursive(ray, bvh.root_, vertices, indices, closest_hit);
    }
    return closest_hit;
}

} // game_object
} // engine