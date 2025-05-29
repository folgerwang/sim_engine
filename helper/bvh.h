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
    glm::vec3 direction;
};

// This structure holds precomputed info for each triangle
struct PrimitiveInfo {
    // Index i, for triangle (indices[3*i], indices[3*i+1], indices[3*i+2])
    int original_triangle_index;
    AABB bounds;
    glm::vec3 centroid_val;
    std::vector<glm::vec3> vertex_list;

    PrimitiveInfo(
        int tri_idx,
        const std::vector<glm::vec3>& v_list);

    std::vector<glm::vec3> splitWithPlane(
        const glm::vec4& clip_plane,
        float clip_back = 1.0f) const;

    const AABB& getAABB() const { return bounds; }

    const glm::vec3& centroid() const { return centroid_val; }
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

class BVHBuilder {
public:
    std::shared_ptr<BVHNode> root_;
    std::vector<PrimitiveInfo> primitive_info_list_; // Stores AABB & centroid for each tri
    std::vector<int> build_indices_; // Indices into 'primitive_info_list' used during build

    const int MAX_PRIMS_IN_NODE = 4;
    const int SAH_BUCKET_COUNT = 12;

    BVHBuilder(
        const std::vector<glm::vec3>& vertices,
        const std::vector<int>& indices);

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

    std::shared_ptr<BVHNode> buildRecursive(
        int level,
        const std::vector<uint32_t>& tested_axises,
        const std::vector<int>& build_indices);

public:
    void printBVH(std::shared_ptr<BVHNode>& node, int depth = 0);

    std::shared_ptr<BVHNode> getBvhNodeRoot() {
        return root_;
    }
};

inline bool rayAABBIntersect(const Ray& ray, const AABB& box, float& t_near_hit, float& t_far_hit);

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
);

struct HitInfo {
    bool hit = false;
    float t = std::numeric_limits<float>::max(); // Distance to intersection
    int triangle_index = -1;                     // Original index of the hit triangle
    float u, v;                                  // Barycentric coordinates
    glm::vec3 intersection_point; // Can be computed from ray.origin + ray.direction * t
};

// Forward declaration if rayTriangleIntersect is in another file/scope
// bool rayTriangleIntersect(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, float& out_t, float& out_u, float& out_v);
inline void intersectBVHRecursive(
    const Ray& ray,
    const std::shared_ptr<BVHNode>& node,
    const std::vector<glm::vec3>& vertices,      // Your global vertex stream
    const std::vector<int>& indices,        // Your global index stream
    HitInfo& closest_hit                     // Passed by reference to update
);

// --- Main function to start the process ---
inline HitInfo findClosestHit(
    const Ray& ray,
    const BVHBuilder& bvh, // Assuming BVHBuilder holds the root and prim info
    const std::vector<glm::vec3>& vertices,
    const std::vector<int>& indices);

} // game_object
} // engine