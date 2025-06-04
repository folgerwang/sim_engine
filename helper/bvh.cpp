#include "bvh.h"
#include <stack>

namespace engine {
namespace helper {

bool rayAABBIntersect(const Ray& ray, const AABB& box, float& t_near_hit, float& t_far_hit) {
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

bool rayTriangleIntersect(
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

    // Calculate determinant: det = E1 � P
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

    // Calculate u parameter and test bound: u = (T � P) * inv_det
    out_u = dot(tvec, pvec) * inv_det;
    if (out_u < 0.0f || out_u > 1.0f) {
        return false; // Intersection point is outside the V0-V1-V2_edge1 side
    }

    // Prepare to test v parameter: Q = T x E1
    glm::vec3 qvec = cross(tvec, edge1);

    // Calculate v parameter and test bound: v = (D � Q) * inv_det
    out_v = dot(ray.direction, qvec) * inv_det;
    if (out_v < 0.0f || out_u + out_v > 1.0f) {
        return false; // Intersection point is outside the V0-V1-V2_edge2 side or V1-V2 edge
    }

    // Calculate t, the distance along the ray to the intersection point: t = (E2 � Q) * inv_det
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

PrimitiveInfo::PrimitiveInfo(int tri_idx, const std::vector<glm::vec3>& v_list)
: original_triangle_index(tri_idx), vertex_list(v_list) {
    glm::vec3 sum_v(0);
    for (const auto& v : v_list) {
        bounds_val.extend(v);
        sum_v += v;
    }
    if (!v_list.empty()) {
        centroid_val = sum_v / static_cast<float_t>(v_list.size());
    }
}

std::vector<glm::vec3> PrimitiveInfo::splitWithPlane(
    const glm::vec4& clip_plane,
    float clip_back/* = 1.0f*/) const {
    // Was already PascalCase
    const uint32_t num_vertex = static_cast<uint32_t>(vertex_list.size());
    if (num_vertex < 3) return {};

    std::vector<float_t> dist_to_plane(num_vertex);
    for (uint32_t i = 0; i < num_vertex; i++) {
        dist_to_plane[i] = clip_back * dot(glm::vec4(vertex_list[i], 1.0f), clip_plane);
    }

    std::vector<glm::vec3> clipped_vertex_list;
    clipped_vertex_list.reserve(num_vertex + 1);

    for (uint32_t i = 0; i < num_vertex; ++i) {
        const glm::vec3& p1 = vertex_list[i];
        float d1 = dist_to_plane[i];
        const glm::vec3& p2 = vertex_list[(i + 1) % num_vertex];
        float d2 = dist_to_plane[(i + 1) % num_vertex];

        if (d1 >= 0) {
            clipped_vertex_list.push_back(p1);
        }
        if ((d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0)) {
            if (std::abs(d1 - d2) > 1e-6f) {
                float t = d1 / (d1 - d2);
                clipped_vertex_list.push_back(p1 + (p2 - p1) * t);
            }
        }
    }
    return clipped_vertex_list;
}

void intersectBVHRecursive(
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

BVHBuilder::BVHBuilder(
    const std::vector<glm::vec3>& vertices,
    const std::vector<int>& indices,
    bool debug_mode)
    : debug_mode_(debug_mode), active_tasks_count_(0), shutdown_threads_(false) {

    if (indices.empty() || vertices.empty() || indices.size() % 3 != 0) {
        // Use printDebug for consistency, though this is single-threaded here
        std::ostringstream oss;
        oss << "BVHBuilder: Invalid input geometry. Vertices: " << vertices.size() << ", Indices: " << indices.size();
        printDebug(oss.str());
        return;
    }

    size_t num_triangles = indices.size() / 3;
    primitive_info_list_.reserve(num_triangles * 2);

    for (size_t i = 0; i < num_triangles; ++i) {
        primitive_info_list_.emplace_back(
            PrimitiveInfo(static_cast<int>(i),
                {
                    vertices[indices[3 * i + 0]],
                    vertices[indices[3 * i + 1]],
                    vertices[indices[3 * i + 2]]
                }));
    }

    initial_build_indices_.resize(primitive_info_list_.size());
    for (size_t i = 0; i < primitive_info_list_.size(); ++i) {
        initial_build_indices_[i] = static_cast<int>(i);
    }
    std::ostringstream oss;
    oss << "BVHBuilder: Initialized with " << primitive_info_list_.size() << " primitives.";
    printDebug(oss.str());
}

BVHBuilder::~BVHBuilder() {
    printDebug("BVHBuilder: Destructor called. Shutting down threads.");
    shutdown_threads_.store(true);
    tasks_cv_.notify_all();
    for (auto& t : thread_pool_) {
        if (t.joinable()) {
            t.join();
        }
    }
    printDebug("BVHBuilder: All threads joined.");
}

void BVHBuilder::fillNodeBounds(
    std::shared_ptr<BVHNode>& node,
    const std::vector<int>& task_build_indices) {
    for (int prim_idx : task_build_indices) {
        node->bounds.extend(primitive_info_list_[prim_idx].getAABB());
    }
}

AABB BVHBuilder::getCentroidBounds(
    const std::vector<int>& task_build_indices) {
    AABB centroid_bounds;
    for (int prim_idx : task_build_indices) {
        centroid_bounds.extend(primitive_info_list_[prim_idx].centroid());
    }
    return centroid_bounds;
}

void BVHBuilder::fillLeafNode(
    std::shared_ptr<BVHNode>& leaf_node,
    const std::vector<int>& task_build_indices) {
    for (int prim_idx : task_build_indices) {
        leaf_node->primitive_ref_indices.push_back(
            primitive_info_list_[prim_idx].original_triangle_index);
    }
}

uint32_t BVHBuilder::getSplitAxis(
    const glm::vec3& extent,
    const std::vector<uint32_t>& tested_axises) {
    int split_axis = 0;
    glm::vec3 masked_extent = extent;
    for (auto axis : tested_axises) {
        if (axis < 3) masked_extent[axis] = -std::numeric_limits<float>::max();
    }

    if (masked_extent.y > masked_extent.x && masked_extent.y > masked_extent.z) split_axis = 1;
    else if (masked_extent.z > masked_extent.x && masked_extent.z > masked_extent.y) split_axis = 2;
    return static_cast<uint32_t>(split_axis);
}

float BVHBuilder::calculateSahSplitCoord(
    int32_t split_axis,
    const glm::vec3& extent,
    const AABB& centroid_bounds,
    float_t node_surface_area,
    const std::vector<int>& task_build_indices) {

    const int32_t num_primitives = static_cast<int32_t>(task_build_indices.size());
    float split_coord_val = std::numeric_limits<float>::max();

    bool use_midpoint_fallback = (extent[split_axis] < 1e-5f) || (num_primitives < SAH_BUCKET_COUNT * 2);

    if (use_midpoint_fallback) {
        split_coord_val = centroid_bounds.centroid()[split_axis];
        // printDebug("SAH: Using midpoint fallback for split coord.");
    }
    else {
        struct BucketInfo {
            int count = 0;
            AABB bounds;
        };
        std::vector<BucketInfo> buckets(SAH_BUCKET_COUNT);
        float min_c_axis = centroid_bounds.min_bounds[split_axis];
        float max_c_axis = centroid_bounds.max_bounds[split_axis];
        float extent_c_axis = max_c_axis - min_c_axis;

        if (extent_c_axis < 1e-6f) {
            split_coord_val = min_c_axis;
            // printDebug("SAH: Centroid extent too small, using min_c_axis.");
        }
        else {
            for (int prim_idx_in_node : task_build_indices) {
                const PrimitiveInfo& p_info_ref = primitive_info_list_[prim_idx_in_node];
                float centroid_coord = p_info_ref.centroid()[split_axis];
                int b = static_cast<int>(SAH_BUCKET_COUNT * ((centroid_coord - min_c_axis) / extent_c_axis));
                b = std::max(0, std::min(SAH_BUCKET_COUNT - 1, b));
                buckets[b].count++;
                buckets[b].bounds.extend(p_info_ref.getAABB());
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
                    if (buckets[j].count > 0) {
                        b1.extend(buckets[j].bounds);
                        count1 += buckets[j].count;
                    }
                }
                cost[i] = 0.125f + ((count0 > 0 ? count0 * b0.surfaceArea() : 0.0f) +
                    (count1 > 0 ? count1 * b1.surfaceArea() : 0.0f))
                    / node_surface_area;
                if (count0 == 0 || count1 == 0) cost[i] = std::numeric_limits<float>::max();
            }

            float min_cost = std::numeric_limits<float>::max();
            int min_cost_split_bucket = -1;
            for (int i = 0; i < SAH_BUCKET_COUNT - 1; ++i) {
                if (cost[i] < min_cost) {
                    min_cost = cost[i];
                    min_cost_split_bucket = i;
                }
            }

            float leaf_cost = static_cast<float>(num_primitives);
            if (min_cost_split_bucket != -1 && min_cost < leaf_cost) {
                float split_ratio = static_cast<float_t>(min_cost_split_bucket + 1) / static_cast<float_t>(SAH_BUCKET_COUNT);
                split_coord_val = min_c_axis + extent_c_axis * split_ratio;
                // std::ostringstream oss_sah;
                // oss_sah << "SAH: Found split. Bucket: " << min_cost_split_bucket << ", Cost: " << min_cost << ", Leaf Cost: " << leaf_cost << ", Coord: " << split_coord_val;
                // printDebug(oss_sah.str());
            }
            else {
                split_coord_val = std::numeric_limits<float>::max();
                // std::ostringstream oss_sah_leaf;
                // oss_sah_leaf << "SAH: No beneficial split found. Min SAH Cost: " << min_cost << ", Leaf Cost: " << leaf_cost;
                // printDebug(oss_sah_leaf.str());
            }
        }
    }
    return split_coord_val;
}


void BVHBuilder::processNodeTask(NodeTask current_task) {
    std::ostringstream oss_entry;
    oss_entry << "ProcessNodeTask Start. NodePtr: " << current_task.node.get()
        << ", Level: " << current_task.level
        << ", Prims: " << current_task.build_indices->size();
    printDebug(oss_entry.str());

    auto& node_to_build = current_task.node;
    const auto& task_build_indices = *current_task.build_indices;
    const int32_t num_primitives = static_cast<int32_t>(task_build_indices.size());

    AABB calculated_node_bounds; // For debug printing
    {
        std::lock_guard<std::mutex> lock(prim_list_mutex_);
        fillNodeBounds(node_to_build, task_build_indices);
        calculated_node_bounds = node_to_build->bounds; // Copy for printing after lock
    }
    std::ostringstream oss_bounds;
    oss_bounds << "  Node " << current_task.node.get() << " L" << current_task.level << " - Calculated Bounds: " << calculated_node_bounds;
    printDebug(oss_bounds.str());


    if (num_primitives <= MAX_PRIMS_IN_NODE) {
        {
            std::lock_guard<std::mutex> lock(prim_list_mutex_);
            fillLeafNode(node_to_build, task_build_indices);
        }
        std::ostringstream oss_leaf;
        oss_leaf << "  Node " << current_task.node.get() << " L" << current_task.level << " - Became LEAF (Max Prims). Prims: " << task_build_indices.size();
        printDebug(oss_leaf.str());

        active_tasks_count_.fetch_sub(1, std::memory_order_relaxed);
        tasks_cv_.notify_all();
        return;
    }

    AABB centroid_bounds;
    {
        std::lock_guard<std::mutex> lock(prim_list_mutex_);
        centroid_bounds = getCentroidBounds(task_build_indices);
    }
    std::ostringstream oss_centroid;
    oss_centroid << "  Node " << current_task.node.get() << " L" << current_task.level << " - Centroid Bounds: " << centroid_bounds;
    printDebug(oss_centroid.str());

    const glm::vec3 centroid_extent = centroid_bounds.getExtent();

    auto build_indices_left_ptr = std::make_shared<std::vector<int32_t>>();
    auto build_indices_right_ptr = std::make_shared<std::vector<int32_t>>();
    build_indices_left_ptr->reserve(num_primitives);
    build_indices_right_ptr->reserve(num_primitives);

    bool split_successful = false;
    std::vector<uint32_t> tested_axises_for_this_node;

    for (int attempt = 0; attempt < 3 && !split_successful; ++attempt) {
        uint32_t current_split_axis = getSplitAxis(centroid_extent, tested_axises_for_this_node);
        tested_axises_for_this_node.push_back(current_split_axis);

        std::ostringstream oss_attempt;
        oss_attempt << "  Node " << current_task.node.get() << " L" << current_task.level
            << " - Attempt " << attempt + 1 << " to split on axis " << current_split_axis;
        printDebug(oss_attempt.str());

        float split_coord_val;
        {
            std::lock_guard<std::mutex> lock(prim_list_mutex_);
            split_coord_val = calculateSahSplitCoord(
                current_split_axis,
                centroid_extent,
                centroid_bounds,
                node_to_build->bounds.surfaceArea(),
                task_build_indices);
        }
        std::ostringstream oss_sah_res;
        oss_sah_res << "    SAH Split Coord on axis " << current_split_axis << ": " << split_coord_val;
        printDebug(oss_sah_res.str());


        if (split_coord_val == std::numeric_limits<float>::max()) {
            printDebug("    SAH indicated no beneficial split on this axis.");
            continue;
        }

        build_indices_left_ptr->clear();
        build_indices_right_ptr->clear();
        int new_prims_added_count = 0;

        {
            std::lock_guard<std::mutex> lock(prim_list_mutex_);
            glm::vec4 clip_plane(0);
            clip_plane[current_split_axis] = 1.0f;
            clip_plane[3] = -split_coord_val;

            for (int prim_idx_in_node : task_build_indices) {
                PrimitiveInfo p_info_copy = primitive_info_list_[prim_idx_in_node];
                const AABB& prim_aabb = p_info_copy.getAABB();

                if (split_coord_val >= prim_aabb.max_bounds[current_split_axis]) {
                    build_indices_left_ptr->push_back(prim_idx_in_node);
                }
                else if (split_coord_val <= prim_aabb.min_bounds[current_split_axis]) {
                    build_indices_right_ptr->push_back(prim_idx_in_node);
                }
                else {
                    std::vector<glm::vec3> left_verts = p_info_copy.splitWithPlane(clip_plane, -1.0f);
                    if (left_verts.size() >= 3) {
                        build_indices_left_ptr->push_back(static_cast<int32_t>(primitive_info_list_.size()));
                        primitive_info_list_.emplace_back(p_info_copy.original_triangle_index, left_verts);
                        new_prims_added_count++;
                    }
                    std::vector<glm::vec3> right_verts = p_info_copy.splitWithPlane(clip_plane, 1.0f);
                    if (right_verts.size() >= 3) {
                        build_indices_right_ptr->push_back(static_cast<int32_t>(primitive_info_list_.size()));
                        primitive_info_list_.emplace_back(p_info_copy.original_triangle_index, right_verts);
                        new_prims_added_count++;
                    }
                }
            }
        }
        if (new_prims_added_count > 0) {
            std::ostringstream oss_split_add;
            oss_split_add << "    Split generated " << new_prims_added_count << " new primitives. Total now: " << primitive_info_list_.size();
            printDebug(oss_split_add.str());
        }


        if (build_indices_left_ptr->empty() || build_indices_right_ptr->empty() ||
            build_indices_left_ptr->size() == num_primitives || build_indices_right_ptr->size() == num_primitives) {
            std::ostringstream oss_bad_split;
            oss_bad_split << "    Bad split: Left (" << build_indices_left_ptr->size()
                << "), Right (" << build_indices_right_ptr->size()
                << ") vs Total (" << num_primitives << "). Trying next axis.";
            printDebug(oss_bad_split.str());
            if (tested_axises_for_this_node.size() < 3) {
                continue;
            }
            else {
                printDebug("    All axes tried, still bad split.");
                break;
            }
        }
        split_successful = true;
        std::ostringstream oss_good_split;
        oss_good_split << "    Successful split on axis " << current_split_axis
            << ". Left: " << build_indices_left_ptr->size()
            << ", Right: " << build_indices_right_ptr->size();
        printDebug(oss_good_split.str());
        break;
    }

    if (!split_successful || build_indices_left_ptr->empty() || build_indices_right_ptr->empty()) {
        {
            std::lock_guard<std::mutex> lock(prim_list_mutex_);
            fillLeafNode(node_to_build, task_build_indices);
        }
        std::ostringstream oss_fleaf;
        oss_fleaf << "  Node " << current_task.node.get() << " L" << current_task.level
            << " - Became LEAF (Failed Split/Empty Child). Prims: " << task_build_indices.size();
        printDebug(oss_fleaf.str());

        active_tasks_count_.fetch_sub(1, std::memory_order_relaxed);
        tasks_cv_.notify_all();
        return;
    }

    int tasks_added = 0;
    if (!build_indices_left_ptr->empty()) {
        NodeTask left_task;
        left_task.node = std::make_shared<BVHNode>();
        node_to_build->left = left_task.node;
        left_task.build_indices = build_indices_left_ptr;
        left_task.build_indices->shrink_to_fit();
        left_task.level = current_task.level + 1;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push(left_task);
        }
        tasks_added++;
        std::ostringstream oss_push_l;
        oss_push_l << "  Node " << current_task.node.get() << " L" << current_task.level
            << " - Pushed Left Child Task (NodePtr: " << left_task.node.get()
            << ", Prims: " << left_task.build_indices->size() << ")";
        printDebug(oss_push_l.str());
    }
    if (!build_indices_right_ptr->empty()) {
        NodeTask right_task;
        right_task.node = std::make_shared<BVHNode>();
        node_to_build->right = right_task.node;
        right_task.build_indices = build_indices_right_ptr;
        right_task.build_indices->shrink_to_fit();
        right_task.level = current_task.level + 1;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push(right_task);
        }
        tasks_added++;
        std::ostringstream oss_push_r;
        oss_push_r << "  Node " << current_task.node.get() << " L" << current_task.level
            << " - Pushed Right Child Task (NodePtr: " << right_task.node.get()
            << ", Prims: " << right_task.build_indices->size() << ")";
        printDebug(oss_push_r.str());
    }

    active_tasks_count_.fetch_add(tasks_added - 1, std::memory_order_relaxed);
    if (tasks_added > 0) {
        tasks_cv_.notify_one();
    }
    else {
        tasks_cv_.notify_all();
    }
    std::ostringstream oss_exit;
    oss_exit << "ProcessNodeTask End. NodePtr: " << current_task.node.get() << " L" << current_task.level << ". Tasks added: " << tasks_added;
    printDebug(oss_exit.str());
}

void BVHBuilder::workerThreadLoop() {
    std::ostringstream oss_start;
    oss_start << "Worker thread started.";
    printDebug(oss_start.str());

    while (true) {
        NodeTask current_task_to_process;
        bool task_fetched = false;
        void* task_node_ptr = nullptr; // For debug before processing
        int task_level = -1;
        size_t task_prims = 0;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // printDebug("Worker waiting for task or shutdown.");
            tasks_cv_.wait(lock, [this] {
                return shutdown_threads_.load() || !task_queue_.empty();
                });

            if (shutdown_threads_.load() && task_queue_.empty()) {
                printDebug("Worker shutting down (queue empty).");
                return;
            }

            if (!task_queue_.empty()) {
                current_task_to_process = task_queue_.top();
                task_queue_.pop();
                task_fetched = true;

                task_node_ptr = current_task_to_process.node.get();
                task_level = current_task_to_process.level;
                if (current_task_to_process.build_indices) task_prims = current_task_to_process.build_indices->size();

            }
            else if (shutdown_threads_.load()) { // Should be caught by first check, but as safeguard
                printDebug("Worker shutting down (spurious wakeup?).");
                return;
            }
        }

        if (task_fetched) {
            std::ostringstream oss_picked;
            oss_picked << "Worker picked up task. NodePtr: " << task_node_ptr << " L" << task_level << " Prims: " << task_prims;
            printDebug(oss_picked.str());
            processNodeTask(current_task_to_process);
        }
    }
}

void BVHBuilder::build() {
    if (primitive_info_list_.empty()) {
        printDebug("BVHBuilder::build - No primitives to build BVH for.");
        root_ = nullptr;
        return;
    }

    root_ = std::make_shared<BVHNode>();
    NodeTask root_task;
    root_task.node = root_;
    root_task.build_indices = std::make_shared<std::vector<int32_t>>(initial_build_indices_);
    root_task.level = 0;

    active_tasks_count_.store(1);
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(root_task);
    }
    std::ostringstream oss_root_pushed;
    oss_root_pushed << "BVHBuilder::build - Pushed root task. NodePtr: " << root_task.node.get()
        << ", Prims: " << root_task.build_indices->size();
    printDebug(oss_root_pushed.str());

    shutdown_threads_.store(false);

    unsigned int num_hw_threads = std::thread::hardware_concurrency();
    unsigned int num_worker_threads = (num_hw_threads > 0) ? num_hw_threads : 2;

    std::ostringstream oss_start_build;
    oss_start_build << "BVHBuilder: Starting build with " << num_worker_threads << " worker threads.";
    printDebug(oss_start_build.str());


    thread_pool_.reserve(num_worker_threads);
    for (unsigned int i = 0; i < num_worker_threads; ++i) {
        thread_pool_.emplace_back(&BVHBuilder::workerThreadLoop, this);
    }

    // Wait for all tasks to complete
    printDebug("BVHBuilder::build - Main thread waiting for tasks to complete...");
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_cv_.wait(lock, [this] {
            bool empty_q = task_queue_.empty();
            int active_c = active_tasks_count_.load();
            // if (debug_mode_ && (empty_q || active_c == 0)) { // Log check conditions
            //    std::lock_guard<std::mutex> dbg_lock(debug_mutex_);
            //    std::cout << "[Thr:" << std::this_thread::get_id() << "] "
            //              << "Wait check: queue_empty=" << empty_q << ", active_tasks=" << active_c << std::endl;
            // }
            return empty_q && (active_c == 0);
            });
    }

    printDebug("BVHBuilder: Build process completed by task counter. Shutting down threads.");

    shutdown_threads_.store(true);
    tasks_cv_.notify_all();
    for (auto& t : thread_pool_) {
        if (t.joinable()) {
            t.join();
        }
    }
    thread_pool_.clear();
    printDebug("BVHBuilder::build - All worker threads joined. Build finished.");
}

} // game_object
} // engine