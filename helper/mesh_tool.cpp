#include "mesh_tool.h"
#include <stack>

namespace engine {
namespace helper {

EdgeInternal::EdgeInternal(uint32_t u_param, uint32_t v_param,
    const std::vector<VertexStruct>& all_vertex_data,
    float normal_weight, float uv_weight)
    : is_sharp(false), cost(0.0f), length_sq_contrib(0.0f), normal_penalty_contrib(0.0f), uv_penalty_contrib(0.0f) {
    v1_idx = std::min(u_param, v_param);
    v2_idx = std::max(u_param, v_param);
    if (v1_idx < all_vertex_data.size() && v2_idx < all_vertex_data.size() && v1_idx != v2_idx) {
        const VertexStruct& vert1_data = all_vertex_data[v1_idx];
        const VertexStruct& vert2_data = all_vertex_data[v2_idx];
        glm::vec3 pos_diff = vert1_data.position - vert2_data.position;
        length_sq_contrib = glm::dot(pos_diff, pos_diff);
        cost += length_sq_contrib;

        glm::vec3 norm1 = glm::normalize(vert1_data.normal);
        glm::vec3 norm2 = glm::normalize(vert2_data.normal);
        if (glm::length(norm1) > 0.5f && glm::length(norm2) > 0.5f) {
            normal_penalty_contrib = (1.0f - glm::dot(norm1, norm2));
            cost += normal_penalty_contrib * normal_weight;
        }
        else {
            normal_penalty_contrib = 2.0f;
            cost += normal_penalty_contrib * normal_weight;
        }

        uv_penalty_contrib = glm::distance(vert1_data.uv, vert2_data.uv);
        cost += uv_penalty_contrib * uv_weight;
    }
    else {
        cost = length_sq_contrib = std::numeric_limits<float>::max();
    }
}

bool EdgeInternal::operator<(const EdgeInternal& other_edge) const {
    if (is_sharp != other_edge.is_sharp) return !is_sharp;
    if (cost != other_edge.cost) return cost < other_edge.cost;
    if (v1_idx != other_edge.v1_idx) return v1_idx < other_edge.v1_idx;
    return v2_idx < other_edge.v2_idx;
}

// --- Geometric Helper Functions ---
glm::vec3 calculateFaceNormal(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) {
    if (p0 == p1 || p1 == p2 || p0 == p2) return glm::vec3(0.0f);
    glm::vec3 edge1 = p1 - p0;
    glm::vec3 edge2 = p2 - p0;
    glm::vec3 calculated_normal = glm::cross(edge1, edge2);
    if (glm::length(calculated_normal) < 0.000001f) return glm::vec3(0.0f);
    return glm::normalize(calculated_normal);
}
float calculateDihedralAngleFromNormals(glm::vec3 n1, glm::vec3 n2) {
    if (glm::length(n1) < 0.000001f || glm::length(n2) < 0.000001f) return 180.0f;
    n1 = glm::normalize(n1);
    n2 = glm::normalize(n2);
    float dot_product_val = glm::dot(n1, n2);
    dot_product_val = std::max(-1.0f, std::min(1.0f, dot_product_val));
    float angle_radians = std::acos(dot_product_val);
    return angle_radians * (180.0f / glm::pi<float>());
}

// --- Simplification Core ---
Mesh simplifyMeshActualButVeryBasic(
    const Mesh& input_mesh,
    int target_face_count,
    float sharp_edge_angle_degrees,
    float normal_similarity_weight,
    float uv_distance_weight) {
    std::cout << "\n[Actual (But VERY BASIC) Simplification - Using VertexStruct, camelCase Funcs]" << std::endl;
    if (!input_mesh.isValid()) {
        return Mesh();
    }
    if (input_mesh.getVertexCount() == 0) {
        return Mesh();
    }
    if (input_mesh.getFaceCount() == 0 ||
        target_face_count >= static_cast<int>(input_mesh.getFaceCount())) {
        Mesh result_mesh;
        if (input_mesh.vertex_data_ptr) {
            *(result_mesh.vertex_data_ptr) = *(input_mesh.vertex_data_ptr);
        }
        if (input_mesh.faces_ptr) {
            *(result_mesh.faces_ptr) = *(input_mesh.faces_ptr);
        }
        if (result_mesh.faces_ptr) {
            for (Face& f_ref : *(result_mesh.faces_ptr)) {
                f_ref.active = true;
            }
        }
        return result_mesh;
    }
    if (target_face_count < 0) {
        target_face_count = 0;
    }

    std::vector<VertexStruct> current_vertex_data = *(input_mesh.vertex_data_ptr);
    std::vector<Face> current_faces_vector = *(input_mesh.faces_ptr);
    for (Face& f_ref : current_faces_vector) {
        f_ref.active = true;
    }

    int active_face_count_val = 0;
    for (const auto& f_const_ref : current_faces_vector) {
        if (f_const_ref.active && !f_const_ref.isDegenerate()) {
            active_face_count_val++;
        }
    } // CALL UPDATED NAME

    int outer_iterations = 0;
    const int max_outer_iterations_const = int(current_faces_vector.size()) + 10;
    uint32_t remained_faces_count = uint32_t(input_mesh.faces_ptr->size());
    uint32_t max_collapses_per_pass = remained_faces_count / 10;

    while (active_face_count_val > target_face_count &&
        outer_iterations < max_outer_iterations_const) {
        outer_iterations++;
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edge_to_face_map;
        std::vector<EdgeInternal> candidate_edges_list;

        for (uint32_t face_idx = 0; face_idx < current_faces_vector.size(); ++face_idx) {
            if (!current_faces_vector[face_idx].active ||
                current_faces_vector[face_idx].isDegenerate()) {
                continue; // CALL UPDATED NAME
            }
            const Face& current_f = current_faces_vector[face_idx];
            for (int i_loop = 0; i_loop < 3; ++i_loop) {
                uint32_t u_vert = current_f.v_indices[i_loop];
                uint32_t v_vert = current_f.v_indices[(i_loop + 1) % 3];
                if (u_vert == v_vert) {
                    continue;
                }
                std::pair<uint32_t, uint32_t> current_edge_key = {
                    std::min(u_vert, v_vert),
                    std::max(u_vert, v_vert)
                };
                edge_to_face_map[current_edge_key].push_back(face_idx);
            }
        }

        if (edge_to_face_map.empty()) {
            break;
        }

        for (const auto& map_entry : edge_to_face_map) {
            const auto& edge_key_val = map_entry.first;
            const auto& incident_face_indices_list = map_entry.second;
            if (edge_key_val.first >= current_vertex_data.size() ||
                edge_key_val.second >= current_vertex_data.size()) {
                continue;
            }

            EdgeInternal current_edge(
                edge_key_val.first,
                edge_key_val.second,
                current_vertex_data,
                normal_similarity_weight,
                uv_distance_weight);

            if (incident_face_indices_list.size() == 2) {
                const Face& f1_face_data = current_faces_vector[incident_face_indices_list[0]];
                const Face& f2_face_data = current_faces_vector[incident_face_indices_list[1]];
                glm::vec3 n_geom1_val =
                    calculateFaceNormal(
                        current_vertex_data[f1_face_data.v_indices[0]].position,
                        current_vertex_data[f1_face_data.v_indices[1]].position,
                        current_vertex_data[f1_face_data.v_indices[2]].position);
                glm::vec3 n_geom2_val =
                    calculateFaceNormal(
                        current_vertex_data[f2_face_data.v_indices[0]].position,
                        current_vertex_data[f2_face_data.v_indices[1]].position,
                        current_vertex_data[f2_face_data.v_indices[2]].position);
                if (glm::length(n_geom1_val) > 0.0001f && glm::length(n_geom2_val) > 0.0001f) {
                    if (calculateDihedralAngleFromNormals(n_geom1_val, n_geom2_val) > (180.0f - sharp_edge_angle_degrees + 0.01f)) {
                        current_edge.is_sharp = true;
                    }
                }
            }
            candidate_edges_list.push_back(current_edge);
        }

        if (candidate_edges_list.empty()) {
            break;
        }
        std::sort(
            candidate_edges_list.begin(),
            candidate_edges_list.end());

        std::vector<EdgeInternal> independent_set_to_collapse;
        std::vector<bool> vertex_locked_this_pass(current_vertex_data.size(), false);
        for (const EdgeInternal& edge_candidate : candidate_edges_list) {
            if (independent_set_to_collapse.size() >= max_collapses_per_pass) {
                break;
            }
            if (edge_candidate.v1_idx >= vertex_locked_this_pass.size() ||
                edge_candidate.v2_idx >= vertex_locked_this_pass.size()) {
                continue;
            }
            if (!vertex_locked_this_pass[edge_candidate.v1_idx] &&
                !vertex_locked_this_pass[edge_candidate.v2_idx]) {
                independent_set_to_collapse.push_back(edge_candidate);
                vertex_locked_this_pass[edge_candidate.v1_idx] = true;
                vertex_locked_this_pass[edge_candidate.v2_idx] = true;
            }
        }
        if (independent_set_to_collapse.empty()) {
            break;
        }

        int total_faces_removed_this_pass = 0;
        for (const EdgeInternal& edge_to_collapse_val : independent_set_to_collapse) {
            if (active_face_count_val <= target_face_count) {
                break;
            }
            uint32_t v_idx_remove = edge_to_collapse_val.v2_idx;
            uint32_t v_idx_keep = edge_to_collapse_val.v1_idx;
            if (v_idx_keep >= current_vertex_data.size() ||
                v_idx_remove >= current_vertex_data.size()) {
                continue;
            }

            VertexStruct& vert_to_keep = current_vertex_data[v_idx_keep];
            const VertexStruct& vert_to_remove = current_vertex_data[v_idx_remove];

            vert_to_keep.position = (vert_to_keep.position + vert_to_remove.position) * 0.5f;
            if (glm::length(vert_to_keep.normal) > 0.5f &&
                glm::length(vert_to_remove.normal) > 0.5f) {
                vert_to_keep.normal = glm::normalize(vert_to_keep.normal + vert_to_remove.normal);
            }
            else if (glm::length(vert_to_remove.normal) > 0.5f) {
                vert_to_keep.normal = glm::normalize(vert_to_remove.normal);
            }
            vert_to_keep.uv = (vert_to_keep.uv + vert_to_remove.uv) * 0.5f;

            int faces_removed_this_single_collapse = 0;
            for (uint32_t face_idx = 0; face_idx < current_faces_vector.size(); ++face_idx) {
                if (!current_faces_vector[face_idx].active) {
                    continue;
                }
                Face& f_ref = current_faces_vector[face_idx];
                for (int i_loop = 0; i_loop < 3; ++i_loop) {
                    if (f_ref.v_indices[i_loop] == v_idx_remove) {
                        f_ref.v_indices[i_loop] = v_idx_keep;
                    }
                }
                if (f_ref.isDegenerate()) { // CALL UPDATED NAME
                    if (current_faces_vector[face_idx].active) {
                        faces_removed_this_single_collapse++;
                    }
                    f_ref.active = false;
                }
            }
            active_face_count_val -= faces_removed_this_single_collapse;
            total_faces_removed_this_pass += faces_removed_this_single_collapse;
        }
        if (total_faces_removed_this_pass == 0 &&
            active_face_count_val > target_face_count) {
            break;
        }
        if (active_face_count_val <= 0 && target_face_count > 0) {
            break;
        }
        remained_faces_count -= total_faces_removed_this_pass;
        max_collapses_per_pass = std::max(remained_faces_count / 10, 10U);
    }
    if (outer_iterations >= max_outer_iterations_const) {
        std::cout << "  Reached max outer iterations. Stopping." << std::endl;
    }
    std::cout << "  Finished iterative collapse. Target: " << target_face_count << ", Actual active faces: " << active_face_count_val << std::endl;

    Mesh simplified_mesh_result;
    std::map<uint32_t, uint32_t> old_to_new_vertex_indices_map;
    uint32_t new_vertex_idx_counter = 0;
    std::vector<bool> final_vertex_is_referenced(current_vertex_data.size(), false);
    for (const auto& face_const_ref : current_faces_vector) {
        if (face_const_ref.active && !face_const_ref.isDegenerate()) { // CALL UPDATED NAME
            for (int i_loop = 0; i_loop < 3; ++i_loop) {
                if (face_const_ref.v_indices[i_loop] < current_vertex_data.size()) {
                    final_vertex_is_referenced[face_const_ref.v_indices[i_loop]] = true;
                }
            }
        }
    }

    for (uint32_t old_vertex_idx = 0; old_vertex_idx < current_vertex_data.size(); ++old_vertex_idx) {
        if (final_vertex_is_referenced[old_vertex_idx]) {
            simplified_mesh_result.vertex_data_ptr->push_back(current_vertex_data[old_vertex_idx]);
            old_to_new_vertex_indices_map[old_vertex_idx] = new_vertex_idx_counter++;
        }
    }
    for (const auto& face_const_ref : current_faces_vector) {
        if (face_const_ref.active && !face_const_ref.isDegenerate()) { // CALL UPDATED NAME
            Face new_built_face;
            bool all_indices_ok = true;
            uint32_t temp_face_indices[3];
            for (int i_loop = 0; i_loop < 3; ++i_loop) {
                auto map_iterator = old_to_new_vertex_indices_map.find(face_const_ref.v_indices[i_loop]);
                if (map_iterator != old_to_new_vertex_indices_map.end()) {
                    temp_face_indices[i_loop] = map_iterator->second;
                }
                else {
                    all_indices_ok = false;
                    break;
                }
            }
            if (all_indices_ok) {
                new_built_face.v_indices[0] = temp_face_indices[0];
                new_built_face.v_indices[1] = temp_face_indices[1];
                new_built_face.v_indices[2] = temp_face_indices[2];
                if (!new_built_face.isDegenerate()) { // CALL UPDATED NAME (final check after re-indexing)
                    simplified_mesh_result.faces_ptr->push_back(new_built_face);
                }
            }
        }
    }
    std::cout << "  Final simplified mesh: " << simplified_mesh_result.getVertexCount() << " Vertices, "
        << simplified_mesh_result.getFaceCount() << " Faces." << std::endl;
    return simplified_mesh_result;
}

// --- printMesh Function ---
void printMesh(const Mesh& mesh_to_print, const std::string& mesh_name) {
    if (!mesh_to_print.isValid()) {
        std::cout << "--- " << mesh_name << " (Mesh data pointers are null) ---" << std::endl;
        return;
    }
    std::cout << "--- " << mesh_name << " ---" << std::endl;
    std::cout << "Vertex Data entries (" << mesh_to_print.getVertexCount() << ")" << std::endl;
    if (mesh_to_print.getVertexCount() == 0) { std::cout << "  (No vertex data)" << std::endl; }
    for (size_t i_idx = 0; i_idx < mesh_to_print.getVertexCount(); ++i_idx) {
        const auto& current_vert = (*mesh_to_print.vertex_data_ptr)[i_idx];
        std::cout << "  V " << i_idx << ": P(" << current_vert.position.x << "," << current_vert.position.y << "," << current_vert.position.z << ")"
            << " N(" << current_vert.normal.x << "," << current_vert.normal.y << "," << current_vert.normal.z << ")"
            << " UV(" << current_vert.uv.x << "," << current_vert.uv.y << ")" << std::endl;
    }
    std::cout << "Faces (" << mesh_to_print.getFaceCount() << "):" << std::endl;
    if (mesh_to_print.getFaceCount() == 0) { std::cout << "  (No faces)" << std::endl; }
    for (size_t i_idx = 0; i_idx < mesh_to_print.getFaceCount(); ++i_idx) {
        const Face& current_f = (*mesh_to_print.faces_ptr)[i_idx];
        std::cout << "  F " << i_idx << ": (" << current_f.v_indices[0] << ", " << current_f.v_indices[1] << ", " << current_f.v_indices[2] << ")" << std::endl;
    }
    std::cout << "--------------------" << std::endl << std::endl;
}

// --- Multi-threaded Mesh Combining Function ---
struct MeshCombineTaskArgs {
    const Mesh* input_mesh_ptr;
    Mesh* output_mesh_ptr;
    uint32_t vertex_data_write_start_idx;
    uint32_t face_write_start_idx;
    uint32_t base_vertex_idx_offset;
};

void processSingleMeshForCombine(
    const MeshCombineTaskArgs& task_args) {
    if (!task_args.input_mesh_ptr->isValid() || !task_args.output_mesh_ptr->isValid()) {
        return;
    }
    const auto& input_vertex_data_vec = *(task_args.input_mesh_ptr->vertex_data_ptr);
    auto& output_vertex_data_vec = *(task_args.output_mesh_ptr->vertex_data_ptr);
    for (size_t i_loop = 0; i_loop < input_vertex_data_vec.size(); ++i_loop) {
        if (task_args.vertex_data_write_start_idx + i_loop < output_vertex_data_vec.size()) {
            output_vertex_data_vec[task_args.vertex_data_write_start_idx + i_loop] = input_vertex_data_vec[i_loop];
        }
    }
    const auto& input_faces_vec = *(task_args.input_mesh_ptr->faces_ptr);
    auto& output_faces_vec = *(task_args.output_mesh_ptr->faces_ptr);
    for (size_t i_loop = 0; i_loop < input_faces_vec.size(); ++i_loop) {
        const auto& original_face = input_faces_vec[i_loop];
        Face new_built_face(
            original_face.v_indices[0] + task_args.base_vertex_idx_offset,
            original_face.v_indices[1] + task_args.base_vertex_idx_offset,
            original_face.v_indices[2] + task_args.base_vertex_idx_offset);
        new_built_face.active = original_face.active;
        if (task_args.face_write_start_idx + i_loop < output_faces_vec.size()) {
            output_faces_vec[task_args.face_write_start_idx + i_loop] = new_built_face;
        }
    }
}

Mesh combineMeshesMultithreaded(
    const std::vector<Mesh>& meshes_to_combine_list,
    uint32_t num_threads_hint_val = 0) {
    if (meshes_to_combine_list.empty()) {
        return Mesh();
    }
    uint32_t actual_num_threads_to_use = num_threads_hint_val;
    if (actual_num_threads_to_use == 0) {
        actual_num_threads_to_use = std::thread::hardware_concurrency();
        if (actual_num_threads_to_use == 0) {
            actual_num_threads_to_use = 2;
        }
    }
    actual_num_threads_to_use =
        std::max(1u,
            std::min(
                actual_num_threads_to_use,
                static_cast<uint32_t>(meshes_to_combine_list.size())));
    Mesh combined_mesh_result;
    uint32_t total_vertex_data_items = 0; uint32_t total_face_items = 0;
    std::vector<uint32_t> vertex_data_write_starts(meshes_to_combine_list.size());
    std::vector<uint32_t> face_write_starts(meshes_to_combine_list.size());
    std::vector<uint32_t> base_vertex_idx_offsets_list(meshes_to_combine_list.size());
    uint32_t current_vertex_data_item_count = 0; uint32_t current_face_item_count = 0;
    for (size_t i_idx = 0; i_idx < meshes_to_combine_list.size(); ++i_idx) {
        if (!meshes_to_combine_list[i_idx].isValid()) {
            continue;
        }
        vertex_data_write_starts[i_idx] = current_vertex_data_item_count;
        face_write_starts[i_idx] = current_face_item_count;
        base_vertex_idx_offsets_list[i_idx] = current_vertex_data_item_count;
        total_vertex_data_items += static_cast<uint32_t>(meshes_to_combine_list[i_idx].getVertexCount());
        total_face_items += static_cast<uint32_t>(meshes_to_combine_list[i_idx].getFaceCount());
        current_vertex_data_item_count += static_cast<uint32_t>(meshes_to_combine_list[i_idx].getVertexCount());
        current_face_item_count += static_cast<uint32_t>(meshes_to_combine_list[i_idx].getFaceCount());
    }
    combined_mesh_result.vertex_data_ptr->resize(total_vertex_data_items);
    combined_mesh_result.faces_ptr->resize(total_face_items);
    if (meshes_to_combine_list.empty() ||
        (total_vertex_data_items == 0 && total_face_items == 0)) {
        return combined_mesh_result;
    }
    if (actual_num_threads_to_use == 1 || meshes_to_combine_list.size() <= 1) {
        for (size_t i_idx = 0; i_idx < meshes_to_combine_list.size(); ++i_idx) {
            if (!meshes_to_combine_list[i_idx].isValid()) {
                continue;
            }
            MeshCombineTaskArgs current_task_args = {
                &meshes_to_combine_list[i_idx],
                &combined_mesh_result,
                vertex_data_write_starts[i_idx],
                face_write_starts[i_idx],
                base_vertex_idx_offsets_list[i_idx] };
            processSingleMeshForCombine(current_task_args);
        }
    }
    else {
        std::vector<std::thread> worker_threads;
        worker_threads.reserve(actual_num_threads_to_use);
        for (uint32_t i_thread_idx = 0; i_thread_idx < actual_num_threads_to_use; ++i_thread_idx) {
            worker_threads.emplace_back([&, i_thread_idx, actual_num_threads_to_use]() {
                for (size_t k_mesh_idx = i_thread_idx; k_mesh_idx < meshes_to_combine_list.size(); k_mesh_idx += actual_num_threads_to_use) {
                    if (!meshes_to_combine_list[k_mesh_idx].isValid()) {
                        continue;
                    }
                    MeshCombineTaskArgs current_task_args = {
                        &meshes_to_combine_list[k_mesh_idx],
                        &combined_mesh_result,
                        vertex_data_write_starts[k_mesh_idx],
                        face_write_starts[k_mesh_idx],
                        base_vertex_idx_offsets_list[k_mesh_idx] };
                    processSingleMeshForCombine(current_task_args);
                }
                });
        }
        for (auto& current_thread_ref : worker_threads) if (current_thread_ref.joinable()) current_thread_ref.join();
    }
    return combined_mesh_result;
}

#if 0
// --- Main Function ---
int main() {
    Mesh cube_mesh_data;
    cube_mesh_data.vertex_data_ptr->assign({
        {{0,0,0}, glm::normalize(glm::vec3(-1,-1,-1)), {0,0}}, {{1,0,0}, glm::normalize(glm::vec3(1,-1,-1)), {1,0}},
        {{1,1,0}, glm::normalize(glm::vec3(1, 1,-1)), {1,1}}, {{0,1,0}, glm::normalize(glm::vec3(-1, 1,-1)), {0,1}},
        {{0,0,1}, glm::normalize(glm::vec3(-1,-1, 1)), {0,1}}, {{1,0,1}, glm::normalize(glm::vec3(1,-1, 1)), {1,1}},
        {{1,1,1}, glm::normalize(glm::vec3(1, 1, 1)), {1,0}}, {{0,1,1}, glm::normalize(glm::vec3(-1, 1, 1)), {0,0}} });
    cube_mesh_data.faces_ptr->assign({
        Face(0,1,2),Face(0,2,3),Face(4,5,6),Face(4,6,7), Face(0,1,5),Face(0,5,4),
        Face(1,2,6),Face(1,6,5),Face(2,3,7),Face(2,7,6),Face(3,0,4),Face(3,4,7) });
    printMesh(cube_mesh_data, "Test Mesh (Cube with VertexStruct, camelCase Funcs)");

    std::vector<Mesh> all_meshes_list; all_meshes_list.push_back(cube_mesh_data);

    auto start_time_combine_val = std::chrono::high_resolution_clock::now();
    Mesh merged_mesh_data = combineMeshesMultithreaded(all_meshes_list, 0);
    auto end_time_combine_val = std::chrono::high_resolution_clock::now();
    std::cout << "\nMesh combining took: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time_combine_val - start_time_combine_val).count() << " microseconds." << std::endl;
    printMesh(merged_mesh_data, "Mesh to be Simplified");

    std::cout << "\n--- Actual (BUT VERY BASIC) HLOD Simplification ---" << std::endl;
    float sharp_edge_angle_thresh_degrees = 75.0f;
    float normal_attr_weight = 20.0f;
    float uv_attr_weight = 10.0f;
    size_t collapses_per_pass_count = std::max(1UL, merged_mesh_data.getFaceCount() / 100);
    collapses_per_pass_count = std::min(collapses_per_pass_count, static_cast<size_t>(20));

    int original_face_count_val = merged_mesh_data.getFaceCount();
    int target_simplification_face_count = std::max(2, original_face_count_val / 4);
    if (original_face_count_val <= 4 && original_face_count_val > 0) target_simplification_face_count = std::max(1, original_face_count_val - 1);
    if (target_simplification_face_count == original_face_count_val && original_face_count_val > 0) target_simplification_face_count--;

    auto start_time_simplify_val = std::chrono::high_resolution_clock::now();
    Mesh hlod_proxy_mesh = simplifyMeshActualButVeryBasic(
        merged_mesh_data,
        target_simplification_face_count,
        sharp_edge_angle_thresh_degrees,
        normal_attr_weight,
        uv_attr_weight,
        collapses_per_pass_count);
    auto end_time_simplify_val = std::chrono::high_resolution_clock::now();
    std::cout << "\nActual (VERY BASIC) mesh simplification took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_simplify_val - start_time_simplify_val).count() << " milliseconds." << std::endl;
    printMesh(hlod_proxy_mesh, "Simplified HLOD Proxy");

    std::cout << "\nREMINDER: This is a TOY IMPLEMENTATION. Quality and robustness are limited." << std::endl;
    return 0;
}
#endif

} // game_object
} // engine