#pragma once
#include <algorithm>
#include <iostream>
#include "renderer/renderer_structs.h"

namespace engine {
namespace game_object {

struct AABB {
    glm::vec3 min_bounds;
    glm::vec3 max_bounds;

    AABB() : min_bounds(std::numeric_limits<float>::max()),
        max_bounds(-std::numeric_limits<float>::max()) {
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

// This structure holds precomputed info for each triangle
struct PrimitiveInfo {
    int original_triangle_index; // Index i, for triangle (indices[3*i], indices[3*i+1], indices[3*i+2])
    AABB bounds;
    glm::vec3 centroid_val;

    PrimitiveInfo(
        int tri_idx,
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2)
        : original_triangle_index(tri_idx) {
        bounds.extend(v0);
        bounds.extend(v1);
        bounds.extend(v2);
        centroid_val = (v0 + v1 + v2) * (1.0f / 3.0f); // Or use AABB centroid: bounds.centroid();
    }

    const AABB& getAABB() const { return bounds; }
    const glm::vec3& centroid() const { return centroid_val; }
};

struct BVHNode {
    AABB bounds;
    BVHNode* left = nullptr;
    BVHNode* right = nullptr;
    // Stores indices into the 'primitive_info_list' (see BVHBuilder)
    // OR directly stores 'original_triangle_index' from PrimitiveInfo
    std::vector<int> primitive_ref_indices;

    bool isLeaf() const {
        return left == nullptr && right == nullptr;
    }
};

class BVHBuilder {
public:
    BVHNode* root = nullptr;
    std::vector<PrimitiveInfo> primitive_info_list; // Stores AABB & centroid for each tri
    std::vector<int> build_indices; // Indices into 'primitive_info_list' used during build

    const int MAX_PRIMS_IN_NODE = 4;
    const int SAH_BUCKET_COUNT = 12;

    BVHBuilder(
        const std::vector<glm::vec3>& vertices,
        const std::vector<int>& indices) {
        if (indices.empty() || vertices.empty()) return;

        size_t num_triangles = indices.size() / 3;
        primitive_info_list.reserve(num_triangles);

        for (size_t i = 0; i < num_triangles; ++i) {
            const glm::vec3& v0 = vertices[indices[3 * i + 0]];
            const glm::vec3& v1 = vertices[indices[3 * i + 1]];
            const glm::vec3& v2 = vertices[indices[3 * i + 2]];
            primitive_info_list.emplace_back(static_cast<int>(i), v0, v1, v2);
        }

        build_indices.resize(primitive_info_list.size());
        for (size_t i = 0; i < primitive_info_list.size(); ++i) {
            build_indices[i] = static_cast<int>(i);
        }

        root = buildRecursive(0, static_cast<int>(primitive_info_list.size()));
    }

    ~BVHBuilder() {
        deleteNodeRecursive(root);
    }

private:
    void deleteNodeRecursive(BVHNode* node) {
        if (!node) return;
        deleteNodeRecursive(node->left);
        deleteNodeRecursive(node->right);
        delete node;
    }

    BVHNode* buildRecursive(int start_index, int end_index) {
        BVHNode* node = new BVHNode();
        int num_primitives_in_node = end_index - start_index;

        // Calculate bounds for all primitives in this node
        for (int i = start_index; i < end_index; ++i) {
            // build_indices[i] is an index into primitive_info_list
            node->bounds.extend(primitive_info_list[build_indices[i]].getAABB());
        }

        // Base case: if few primitives, create a leaf node
        if (num_primitives_in_node <= MAX_PRIMS_IN_NODE) {
            for (int i = start_index; i < end_index; ++i) {
                // Store the original triangle index directly, or the index into primitive_info_list
                node->primitive_ref_indices.push_back(
                    primitive_info_list[build_indices[i]].original_triangle_index
                );
            }
            return node;
        }

        // --- Find the best split (SAH or other heuristic) ---
        AABB centroid_bounds;
        for (int i = start_index; i < end_index; ++i) {
            centroid_bounds.extend(primitive_info_list[build_indices[i]].centroid());
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

        if (use_midpoint_fallback || num_primitives_in_node < SAH_BUCKET_COUNT * 2) {
            split_coord_val = (split_axis == 0) ? centroid_bounds.centroid().x :
                (split_axis == 1) ? centroid_bounds.centroid().y :
                centroid_bounds.centroid().z;
        }
        else {
            // --- SAH Bucketing (Simplified) ---
            struct BucketInfo {
                int count = 0;
                AABB bounds;
            };
            std::vector<BucketInfo> buckets(SAH_BUCKET_COUNT);
            float min_c_axis = (split_axis == 0) ? centroid_bounds.min_bounds.x : (split_axis == 1) ? centroid_bounds.min_bounds.y : centroid_bounds.min_bounds.z;
            float max_c_axis = (split_axis == 0) ? centroid_bounds.max_bounds.x : (split_axis == 1) ? centroid_bounds.max_bounds.y : centroid_bounds.max_bounds.z;
            float extent_c_axis = max_c_axis - min_c_axis;

            if (extent_c_axis < 1e-6f) { // All centroids are basically at the same spot on this axis
                // Fallback if bucketing is not possible
                split_coord_val = min_c_axis; // or centroid_bounds.centroid().coord(split_axis)
            }
            else {
                for (int i = start_index; i < end_index; ++i) {
                    const PrimitiveInfo& p_info = primitive_info_list[build_indices[i]];
                    float centroid_coord = (split_axis == 0) ? p_info.centroid().x :
                        (split_axis == 1) ? p_info.centroid().y :
                        p_info.centroid().z;
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
                        b0.extend(buckets[j].bounds);
                        count0 += buckets[j].count;
                    }
                    for (int j = i + 1; j < SAH_BUCKET_COUNT; ++j) {
                        b1.extend(buckets[j].bounds);
                        count1 += buckets[j].count;
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
                    split_coord_val = min_c_axis + extent_c_axis * (min_cost_split_bucket + 1) / SAH_BUCKET_COUNT;
                }
                else {
                    for (int i = start_index; i < end_index; ++i) {
                        node->primitive_ref_indices.push_back(primitive_info_list[build_indices[i]].original_triangle_index);
                    }
                    return node; // Make leaf
                }
            }
        }

        // Partition build_indices based on the split
        auto* p_mid = std::partition(&build_indices[start_index],
            &build_indices[end_index - 1] + 1,
            [&](int p_info_idx) {
                const PrimitiveInfo& current_prim_info = primitive_info_list[p_info_idx];
                float centroid_coord = (split_axis == 0) ? current_prim_info.centroid().x :
                    (split_axis == 1) ? current_prim_info.centroid().y :
                    current_prim_info.centroid().z;
                return centroid_coord < split_coord_val;
            });

        int mid_point_offset = static_cast<int>(p_mid - &build_indices[0]);

        if (mid_point_offset == start_index || mid_point_offset == end_index) {
            // Fallback if partitioning failed, just split in the middle of the current range
            mid_point_offset = start_index + num_primitives_in_node / 2;
        }

        node->left = buildRecursive(start_index, mid_point_offset);
        node->right = buildRecursive(mid_point_offset, end_index);

        return node;
    }

public:
    void printBVH(BVHNode* node, int depth = 0) const {
        if (!node) return;
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
            for (int i = 0; i < depth + 1; ++i) std::cout << "  ";
            std::cout << "Leaf Original Triangle Indices: ";
            for (int original_idx : node->primitive_ref_indices) std::cout << original_idx << " ";
            std::cout << std::endl;
        }
    }
};

} // game_object
} // engine