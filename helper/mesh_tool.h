#pragma once
#include <algorithm>
#include <iostream>
#include <stack>
#include <limits>    // For std::numeric_limits
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>
#include "renderer/renderer_structs.h"

namespace engine {
namespace helper {

const float c_sharp_edge_angle_threshold_degrees = 75.0f; 
const float c_normal_weight = 10.0f; // How much dissimilarity in normals contributes to cost
const float c_uv_weight = 5.0f;    // How much distance in UVs contributes to cost

struct VertexStruct {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// --- Data Structures ---
class FaceInfo {
    uint32_t v_indices_[3]; 
    bool active_ = false;

public:
    FaceInfo(
        uint32_t v0 = 0,
        uint32_t v1 = 0,
        uint32_t v2 = 0) :
        active_(true) {
        v_indices_[0] = v0;
        v_indices_[1] = v1;
        v_indices_[2] = v2;
    }

    bool isDegenerate() const {
        return v_indices_[0] == v_indices_[1] ||
            v_indices_[1] == v_indices_[2] ||
            v_indices_[0] == v_indices_[2];
    }

    inline void setActive(bool active) {
        active_ = active;
    }

    inline bool isActive() const {
        return active_;
    }

    inline uint32_t getIndice(int order) const {
        return v_indices_[order % 3];
    }

    inline void setIndice(int order, uint32_t idx) {
        v_indices_[order % 3] = idx;
    }
};

struct Mesh {
    std::shared_ptr<std::vector<VertexStruct>> vertices;
    std::vector<FaceInfo> faces;
};

// --- Edge Structure for Simplification ---
class EdgeInternal {
    uint32_t v1_idx, v2_idx;
    float cost; // Combined cost
    // For debugging/understanding:
    float length_sq_contrib;
    float normal_penalty_contrib;
    float uv_penalty_contrib;
    bool is_sharp;

public:
    EdgeInternal(uint32_t u, uint32_t v,
        const std::shared_ptr<std::vector<VertexStruct>>& vertices_list,
        float normal_weight, float uv_weight);

    bool operator<(const EdgeInternal& other) const;
    inline void setSharp(bool sharp) {
        is_sharp = sharp;
    }

    inline uint32_t getV1Index() const {
        return v1_idx;
    }

    inline uint32_t getV2Index() const {
        return v2_idx;
    }
};

extern Mesh simplifyMeshActualButVeryBasic(
    const Mesh& input_mesh_const,
    int target_face_count,
    float sharp_edge_angle_degrees,
    float normal_similarity_weight,
    float uv_distance_weight);

} // game_object
} // engine