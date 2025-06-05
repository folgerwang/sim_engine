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
const uint32_t c_num_lods = 5;
const float c_target_lod_ratio = 0.3f;

// --- Vertex Structure ---
struct VertexStruct {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// --- Face Structure ---
struct Face {
    uint32_t v_indices[3];
    bool active = true;
    Face(unsigned int v0 = 0, unsigned int v1 = 0, unsigned int v2 = 0) : active(true) {
        v_indices[0] = v0; v_indices[1] = v1; v_indices[2] = v2;
    }
    // RENAMED
    bool isDegenerate() const {
        return v_indices[0] == v_indices[1] || v_indices[1] == v_indices[2] || v_indices[0] == v_indices[2];
    }
};

// --- Mesh Struct ---
struct Mesh {
    std::shared_ptr<std::vector<VertexStruct>> vertex_data_ptr;
    std::shared_ptr<std::vector<Face>> faces_ptr;

    Mesh() : vertex_data_ptr(std::make_shared<std::vector<VertexStruct>>()),
        faces_ptr(std::make_shared<std::vector<Face>>()) {
    }

    size_t getVertexCount() const { return vertex_data_ptr ? vertex_data_ptr->size() : 0; }
    size_t getFaceCount() const { return faces_ptr ? faces_ptr->size() : 0; }
    bool isValid() const { return vertex_data_ptr && faces_ptr; }
};

// --- Edge Structure for Simplification ---
struct EdgeInternal {
    uint32_t v1_idx, v2_idx;
    float cost;
    float length_sq_contrib;
    float normal_penalty_contrib;
    float uv_penalty_contrib;
    bool is_sharp;

    EdgeInternal(
        uint32_t u_param,
        uint32_t v_param,
        const std::vector<VertexStruct>& all_vertex_data,
        float normal_weight,
        float uv_weight);

    bool operator<(const EdgeInternal& other_edge) const;
};

// --- Forward declarations ---
extern Mesh simplifyMeshActualButVeryBasic(
    const Mesh& input_mesh,
    int target_face_count,
    float sharp_edge_angle_degrees,
    float normal_similarity_weight,
    float uv_distance_weight);

} // game_object
} // engine