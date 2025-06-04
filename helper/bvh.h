#pragma once
#include <algorithm>
#include <iostream>
#include <stack>
#include <limits>    // For std::numeric_limits
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
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

    glm::vec3 getExtent() const {
        return max_bounds - min_bounds;
    }

    // For printing
    friend std::ostream& operator<<(std::ostream& os, const AABB& box) {
        //os << "Min" << box.min_bounds << ", Max" << box.max_bounds;
        return os;
    }
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct PrimitiveInfo {
    int original_triangle_index;
    std::vector<glm::vec3> vertex_list; // Store vertices directly for splitting
    AABB bounds_val;
    glm::vec3 centroid_val;

    PrimitiveInfo(int tri_idx, const std::vector<glm::vec3>& v_list);
    const AABB& getAABB() const { return bounds_val; } // Kept as is, common getter style
    const glm::vec3& centroid() const { return centroid_val; } // Kept as is

    std::vector<glm::vec3> splitWithPlane(const glm::vec4& clip_plane, float clip_back = 1.0f) const;
};

struct BVHNode {
    AABB bounds;
    std::shared_ptr<BVHNode> left = nullptr;
    std::shared_ptr<BVHNode> right = nullptr;
    std::vector<int> primitive_ref_indices;

    bool isLeaf() const {
        return left == nullptr && right == nullptr;
    }
};

struct NodeTask {
    std::shared_ptr<BVHNode> node;
    std::shared_ptr<std::vector<int32_t>> build_indices;
    int level;
};

class BVHBuilder {
public:
    static const int MAX_PRIMS_IN_NODE = 8;
    static const int SAH_BUCKET_COUNT = 16;

    BVHBuilder(
        const std::vector<glm::vec3>& vertices,
        const std::vector<int>& indices,
        bool debug_mode = false);

    ~BVHBuilder();

    void build(); // Changed from build
    std::shared_ptr<BVHNode> getRoot() const { return root_; } // Changed from getRoot

private:
    std::shared_ptr<BVHNode> root_;
    std::vector<PrimitiveInfo> primitive_info_list_;
    std::vector<int32_t> initial_build_indices_;

    bool debug_mode_;

    std::vector<std::thread> thread_pool_;
    std::stack<NodeTask> task_queue_;
    std::mutex queue_mutex_;
    std::mutex prim_list_mutex_;
    std::mutex debug_mutex_; // Mutex for protecting std::cout
    std::condition_variable tasks_cv_;
    std::atomic<int> active_tasks_count_;
    std::atomic<bool> shutdown_threads_;

    void workerThreadLoop();
    void processNodeTask(NodeTask current_task);

    void fillNodeBounds(
        std::shared_ptr<BVHNode>& node,
        const std::vector<int>& task_build_indices);

    AABB getCentroidBounds(
        const std::vector<int>& task_build_indices);

    void fillLeafNode(
        std::shared_ptr<BVHNode>& leaf_node,
        const std::vector<int>& task_build_indices);

    uint32_t getSplitAxis(
        const glm::vec3& extent,
        const std::vector<uint32_t>& tested_axises);

    float calculateSahSplitCoord(
        int32_t split_axis,
        const glm::vec3& extent,
        const AABB& centroid_bounds,
        float_t node_surface_area,
        const std::vector<int>& task_build_indices);

    // Helper for thread-safe printing
    template<typename T>
    void printDebug(const T& message) {
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(debug_mutex_);
            std::cout << "[Thr:" << std::this_thread::get_id() << "] " << message << std::endl;
        }
    }
    void printDebug(const std::string& message) { // Overload for string literals
        if (debug_mode_) {
            std::lock_guard<std::mutex> lock(debug_mutex_);
            std::cout << "[Thr:" << std::this_thread::get_id() << "] " << message << std::endl;
        }
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