#include "mesh_tool.h"
#include <stack>

namespace engine {
namespace helper {

EdgeInternal::EdgeInternal(
    uint32_t u, uint32_t v,
    const std::shared_ptr<std::vector<VertexStruct>>& vertices_list,
    float normal_weight, float uv_weight) :
    is_sharp(false),
    cost(0.0f),
    length_sq_contrib(0.0f),
    normal_penalty_contrib(0.0f),
    uv_penalty_contrib(0.0f) {

    if (vertices_list == nullptr)
        return;

    uint32_t num_vertices = uint32_t(vertices_list->size());

    v1_idx = std::min(u, v);
    v2_idx = std::max(u, v);

    if (v1_idx < num_vertices && v2_idx < num_vertices && v1_idx != v2_idx) {
        glm::vec3 diff = (*vertices_list)[v1_idx].position - (*vertices_list)[v2_idx].position;
        length_sq_contrib = dot(diff, diff);
        cost += length_sq_contrib; // Base cost is geometric length

        // Normal similarity penalty (0 for same direction, up to 2 for opposite)
        glm::vec3 n1 = normalize((*vertices_list)[v1_idx].normal);
        glm::vec3 n2 = normalize((*vertices_list)[v2_idx].normal);
        if (length(n1) > 0.5f && length(n2) > 0.5f) { // Basic check if normals are somewhat valid
            normal_penalty_contrib = (1.0f - dot(n1, n2)); // Penalize dissimilarity
            cost += normal_penalty_contrib * normal_weight;
        }
        else { // If normals are zero, treat as max penalty to avoid collapsing bad data
            normal_penalty_contrib = 2.0f;
            cost += normal_penalty_contrib * normal_weight;
        }

        // UV distance penalty
        uv_penalty_contrib =
            distance((*vertices_list)[v1_idx].uv, (*vertices_list)[v2_idx].uv);
        cost += uv_penalty_contrib * uv_weight;

    }
    else {
        cost = length_sq_contrib = std::numeric_limits<float>::max();
    }
}

// Sort operator: non-sharp first, then by overall cost.
bool EdgeInternal::operator<(
    const EdgeInternal& other) const {
    if (is_sharp != other.is_sharp)
        return !is_sharp; // non-sharp (false) comes before sharp (true)
    if (cost != other.cost)
        return cost < other.cost;
    // Tie-breaking for consistent sort order
    if (v1_idx != other.v1_idx)
        return v1_idx < other.v1_idx;
    return v2_idx < other.v2_idx;
}

// --- Geometric Helper Functions ---
glm::vec3 calculateFaceNormal(
    const glm::vec3& p0,
    const glm::vec3& p1,
    const glm::vec3& p2) {
    if (p0 == p1 || p1 == p2 || p0 == p2)
        return glm::vec3(0.0f); 
    glm::vec3 edge1 = p1 - p0;
    glm::vec3 edge2 = p2 - p0;
    glm::vec3 normal = cross(edge1, edge2);
    if (length(normal) < 0.000001f) {
        return glm::vec3(0.0f);
    }
    return normalize(normal);
}

float calculateDihedralAngleFromNormals(
    glm::vec3 n1,
    glm::vec3 n2) {
    if (length(n1) < 0.000001f ||
        length(n2) < 0.000001f) {
        return 180.0f;
    }
    n1 = normalize(n1); 
    n2 = normalize(n2); 
    float dot_product = dot(n1, n2);
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product)); 
    float angle_rad = std::acos(dot_product);
    return angle_rad * (180.0f / glm::pi<float>()); 
}

// --- Simplification Core (VERY BASIC - now with normal/uv consideration) ---
Mesh simplifyMeshActualButVeryBasic(
    const Mesh& input_mesh_const,
    int target_face_count,
    float sharp_edge_angle_degrees,
    float normal_similarity_weight, 
    float uv_distance_weight)
{
    std::cout << "\n[Actual (But VERY BASIC) Simplification" << std::endl;
    std::cout << "  Normal Weight: " << normal_similarity_weight << ", UV Weight: " << uv_distance_weight << std::endl;

    if (input_mesh_const.vertices == nullptr ||
        input_mesh_const.vertices->empty()) {
        return Mesh();
    }

    if (input_mesh_const.faces.empty() ||
        target_face_count >= static_cast<int>(input_mesh_const.faces.size())) {
        Mesh resultMesh = input_mesh_const;
        for (FaceInfo& f : resultMesh.faces) {
            f.setActive(true);
        }
        return resultMesh;
    }

    if (target_face_count < 0)
        target_face_count = 0;

    auto current_vertices = *input_mesh_const.vertices;
    auto current_faces = input_mesh_const.faces;
    for (auto& f : current_faces) {
        f.setActive(true);
    }

    int active_face_count = 0;
    for (const auto& f : current_faces) {
        if (f.isActive() && !f.isDegenerate()) {
            active_face_count++;
        }
    }

    int iterations = 0;
    const int MAX_ITERATIONS = int(current_faces.size()) * 3; // Increased safety break slightly

    while (active_face_count > target_face_count &&
        iterations < MAX_ITERATIONS) {
        iterations++;
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edge_to_face_map;
        std::vector<EdgeInternal> current_edges;

        for (uint32_t face_idx = 0; face_idx < current_faces.size(); ++face_idx) {
            if (!current_faces[face_idx].isActive() ||
                current_faces[face_idx].isDegenerate())
                continue;

            const auto& f = current_faces[face_idx];
            for (int i = 0; i < 3; ++i) {
                uint32_t u = f.getIndice(i);
                uint32_t v = f.getIndice(i + 1);
                if (u == v) continue; 
                std::pair<uint32_t, uint32_t> edge_key = {
                    std::min(u, v),
                    std::max(u, v) };
                edge_to_face_map[edge_key].push_back(face_idx);
            }
        }

        if (edge_to_face_map.empty()) {
            break;
        }

        for (const auto& entry : edge_to_face_map) {
            const auto& edge_key = entry.first;
            const auto& incident_faces_indices = entry.second;
            if (edge_key.first >= current_vertices.size() ||
                edge_key.second >= current_vertices.size()) {
                continue;
            }

            EdgeInternal edge(
                edge_key.first,
                edge_key.second,
                input_mesh_const.vertices,
                normal_similarity_weight,
                uv_distance_weight);

            if (incident_faces_indices.size() == 2) { 
                const FaceInfo& f1_data = current_faces[incident_faces_indices[0]];
                const FaceInfo& f2_data = current_faces[incident_faces_indices[1]];
                glm::vec3 n_geom1 = calculateFaceNormal(
                    current_vertices[f1_data.getIndice(0)].position,
                    current_vertices[f1_data.getIndice(1)].position,
                    current_vertices[f1_data.getIndice(2)].position);
                glm::vec3 n_geom2 = calculateFaceNormal(
                    current_vertices[f2_data.getIndice(0)].position,
                    current_vertices[f2_data.getIndice(1)].position,
                    current_vertices[f2_data.getIndice(2)].position);

                if (glm::length(n_geom1) > 0.0001f && glm::length(n_geom2) > 0.0001f) { 
                    float angle_between_normals = calculateDihedralAngleFromNormals(n_geom1, n_geom2);
                    if (angle_between_normals > (180.0f - sharp_edge_angle_degrees + 0.01f) ) {
                         edge.setSharp(true);
                    }
                }
            }
            // Add additional sharpness criteria based on attribute discontinuities if desired
            // For example, if dot(vertex_normal1, vertex_normal2) < threshold, or distance(uv1, uv2) > threshold.
            // This would make the edge.is_sharp = true; or heavily increase its cost.
            // For now, is_sharp is purely geometric. The attribute differences are in the cost.
            current_edges.push_back(edge);
        }
        
        if (current_edges.empty()) {
            break;
        }
        std::sort(current_edges.begin(), current_edges.end()); 
        EdgeInternal* best_edge_to_collapse = nullptr;
        for(EdgeInternal& edge : current_edges) {
            if (edge.getV1Index() == edge.getV2Index()) {
                continue;
            }
                
            best_edge_to_collapse = &edge;
            break;
        }

        if (!best_edge_to_collapse) {
            break;
        }

        uint32_t v_remove = best_edge_to_collapse->getV2Index();
        uint32_t v_keep = best_edge_to_collapse->getV1Index();

        // Update attributes for v_keep (naive averaging)
        current_vertices[v_keep].position =
            (current_vertices[v_keep].position + current_vertices[v_remove].position) * 0.5f;
        if (v_keep < current_vertices.size() && v_remove < current_vertices.size()) { // Check bounds before access
            current_vertices[v_keep].normal =
                normalize(current_vertices[v_keep].normal + current_vertices[v_remove].normal);
        } else if (v_keep < current_vertices.size()) {
            // v_remove has no normal, v_keep keeps its normal (already normalized or should be)
            current_vertices[v_keep].normal =
                normalize(current_vertices[v_keep].normal);
        }

        if (v_keep < current_vertices.size() && v_remove < current_vertices.size()) { // Check bounds
            current_vertices[v_keep].uv =
                (current_vertices[v_keep].uv + current_vertices[v_remove].uv) * 0.5f;
        }

        // If one UV is missing, the other is kept. If both missing, UV at v_keep remains whatever it was.
        int faces_removed_this_iteration = 0;
        for (uint32_t face_idx = 0; face_idx < current_faces.size(); ++face_idx) {
            if (!current_faces[face_idx].isActive())
                continue;
            auto& f = current_faces[face_idx];
            for(int i=0; i<3; ++i) {
                if (f.getIndice(i) == v_remove) {
                    f.setIndice(i, v_keep);
                }
            }
            if (f.isDegenerate()) {
                if (current_faces[face_idx].isActive()) {
                    faces_removed_this_iteration++;
                }
                
                f.setActive(false);
            }
        }
        active_face_count -= faces_removed_this_iteration;
        
        if (faces_removed_this_iteration == 0 && active_face_count > target_face_count) { /* ... stall handling ... */
            // (Stall handling from previous code, slightly simplified for brevity)
            std::cout << "  Stalled in iteration " << iterations << ". Attempting basic recovery." << std::endl;
            const auto& incident_faces_it = edge_to_face_map.find({v_keep, v_remove});
            int extra_removed = 0;
            if (incident_faces_it != edge_to_face_map.end()) {
                for (uint32_t fi : incident_faces_it->second) {
                    if(current_faces[fi].isActive()) {
                        current_faces[fi].setActive(false);
                        extra_removed++;
                    }
                }
            }
            if(extra_removed > 0) {
                active_face_count -= extra_removed;
            } 
            else {
                std::cout << "    Stall unrecoverable. Stopping." << std::endl; break;
            }
        }
        if (active_face_count <= 0 && target_face_count > 0) {
            break;
        }
    }
     // ... (rest of loop and max iteration check from previous version) ...
    if (iterations >= MAX_ITERATIONS) {
        std::cout << "  Reached max iterations (" << MAX_ITERATIONS << "). Stopping." << std::endl;
    }
    std::cout << "  Finished iterative collapse. Active faces: " << active_face_count << std::endl;


    Mesh simplified_mesh;
    std::map<uint32_t, uint32_t> old_to_new_vertex_indices;
    uint32_t new_idx_counter = 0;

    simplified_mesh.vertices =
        std::make_shared<std::vector<VertexStruct>>();
    simplified_mesh.vertices->reserve(current_vertices.size());

    std::vector<bool> final_vertex_is_referenced(current_vertices.size(), false);
    for (const auto& face : current_faces) {
        if (face.isActive() && !face.isDegenerate()) {
            for (int i=0; i<3; ++i) {
                if (face.getIndice(i) < current_vertices.size()) {
                    final_vertex_is_referenced[face.getIndice(i)] = true;
                }
            }
        }
    }
    
    for (uint32_t old_idx = 0; old_idx < current_vertices.size(); ++old_idx) {
        if (final_vertex_is_referenced[old_idx]) {
            simplified_mesh.vertices->push_back(current_vertices[old_idx]);
            old_to_new_vertex_indices[old_idx] = new_idx_counter++;
        }
    }

    for (const auto& face : current_faces) {
        if (face.isActive() && !face.isDegenerate()) {
            FaceInfo new_face_data; 
            bool all_indices_mapped_and_valid = true;
            uint32_t temp_indices[3];
            for(int i=0; i<3; ++i) {
                auto it = old_to_new_vertex_indices.find(face.getIndice(i));
                if (it != old_to_new_vertex_indices.end()) {
                    temp_indices[i] = it->second;
                } 
                else {
                    all_indices_mapped_and_valid = false;
                    break;
                }
            }
            if (all_indices_mapped_and_valid) {
                new_face_data.setIndice(0, temp_indices[0]);
                new_face_data.setIndice(1, temp_indices[1]);
                new_face_data.setIndice(2, temp_indices[2]);
                if (!new_face_data.isDegenerate()) {
                    simplified_mesh.faces.push_back(new_face_data);
                }
            }
        }
    }
    std::cout << "  Final simplified mesh: " << simplified_mesh.vertices->size() << " vertices, "
              << simplified_mesh.faces.size() << " faces." << std::endl;
    return simplified_mesh;
}


// --- printMesh Function ---
void printMesh(
    const Mesh& mesh,
    const std::string& name) { 
    std::cout << "--- " << name << " ---" << std::endl;
    if (mesh.vertices == nullptr || mesh.vertices->empty()) {
        std::cout << "  (No vertices)" << std::endl;
    }
    auto& current_vertices = *mesh.vertices;
    std::cout << "Vertices (" << current_vertices.size() << std::endl;
    for (size_t i = 0; i < current_vertices.size(); ++i) {
        std::cout << "  V " << i << ": P(" << current_vertices[i].position.x << "," << current_vertices[i].position.y << "," << current_vertices[i].position.z << ")";
        std::cout << " N(" << current_vertices[i].normal.x << "," << current_vertices[i].normal.y << "," << current_vertices[i].normal.z << ")";
        std::cout << " UV(" << current_vertices[i].uv.x << "," << current_vertices[i].uv.y << ")";
        std::cout << std::endl;
    }
    std::cout << "Faces (" << mesh.faces.size() << "):" << std::endl;
    if (mesh.faces.empty()) { std::cout << "  (No faces)" << std::endl; }
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        const auto& f = mesh.faces[i]; 
        std::cout << "  F " << i << ": (" << f.getIndice(0) << ", " << f.getIndice(1) << ", " << f.getIndice(2) << ")" << std::endl;
    }
    std::cout << "--------------------" << std::endl << std::endl;
}


// --- Multi-threaded Mesh Combining Function ---
struct MeshCombineTaskArgs {
    const Mesh* input_mesh_ptr;
    Mesh* output_mesh_ptr;
    uint32_t vertex_write_start_idx;
    uint32_t face_write_start_idx;
    uint32_t base_vertex_idx_offset;
};

void processSingleMeshForCombine(const MeshCombineTaskArgs& args) { 
    if (args.input_mesh_ptr->vertices == nullptr ||
        args.output_mesh_ptr->vertices == nullptr) {
        return;
    }
    for (size_t i = 0; i < (*args.input_mesh_ptr->vertices).size(); ++i) {
        (*args.output_mesh_ptr->vertices)[args.vertex_write_start_idx + i] = (*args.input_mesh_ptr->vertices)[i];
    }
    for (size_t i = 0; i < args.input_mesh_ptr->faces.size(); ++i) {
        const auto& original_face = args.input_mesh_ptr->faces[i];
        FaceInfo new_face_data(
            original_face.getIndice(0) + args.base_vertex_idx_offset,
            original_face.getIndice(1) + args.base_vertex_idx_offset,
            original_face.getIndice(2) + args.base_vertex_idx_offset);
        new_face_data.setActive(original_face.isActive());
        args.output_mesh_ptr->faces[args.face_write_start_idx + i] = new_face_data;
    }
}

Mesh combineMeshesMultithreaded(
    const std::vector<Mesh>& meshes_to_combine,
    uint32_t num_threads_hint = 0) { 
    if (meshes_to_combine.empty()) { return Mesh(); }
    // ... (thread count logic from before) ...
    uint32_t actual_num_threads = num_threads_hint;
    if (actual_num_threads == 0) {
        actual_num_threads = std::thread::hardware_concurrency();
        if (actual_num_threads == 0)
            actual_num_threads = 2;
    }
    actual_num_threads =
        std::max(
            1u,
            std::min(
                actual_num_threads,
                static_cast<uint32_t>(meshes_to_combine.size())));

    Mesh combined_mesh; 
    uint32_t total_vertices = 0; uint32_t total_faces = 0;

    std::vector<uint32_t> vertex_write_start_indices(meshes_to_combine.size());
    std::vector<uint32_t> face_write_start_indices(meshes_to_combine.size());
    std::vector<uint32_t> base_vertex_index_offsets(meshes_to_combine.size());
    uint32_t current_vertex_count = 0;
    uint32_t current_face_count = 0;

    for (size_t i = 0; i < meshes_to_combine.size(); ++i) {
        vertex_write_start_indices[i] = current_vertex_count;
        face_write_start_indices[i] = current_face_count;
        base_vertex_index_offsets[i] = current_vertex_count;

        if (meshes_to_combine[i].vertices == nullptr) {
            continue;
        }
        
        total_vertices += static_cast<uint32_t>(meshes_to_combine[i].vertices->size());
        total_faces += static_cast<uint32_t>(meshes_to_combine[i].faces.size());

        current_vertex_count += static_cast<uint32_t>(meshes_to_combine[i].vertices->size());
        current_face_count += static_cast<uint32_t>(meshes_to_combine[i].faces.size());
    }
    
    if (combined_mesh.vertices == nullptr) {
        combined_mesh.vertices->resize(total_vertices);
    }
    combined_mesh.faces.resize(total_faces);

    if (meshes_to_combine.empty() || (total_vertices == 0 && total_faces == 0))
        return combined_mesh;

    if (actual_num_threads == 1 || meshes_to_combine.size() <= 1) {
        for (size_t i = 0; i < meshes_to_combine.size(); ++i) {
            MeshCombineTaskArgs args = {
                &meshes_to_combine[i],
                &combined_mesh,
                vertex_write_start_indices[i],
                face_write_start_indices[i],
                base_vertex_index_offsets[i]};
            processSingleMeshForCombine(args);
        }
    } else {
        std::vector<std::thread> threads; threads.reserve(actual_num_threads);
        for (uint32_t i = 0; i < actual_num_threads; ++i) {
            threads.emplace_back([&, i, actual_num_threads]() {
                for (size_t k = i; k < meshes_to_combine.size(); k += actual_num_threads) {
                     MeshCombineTaskArgs args = {
                         &meshes_to_combine[k],
                         &combined_mesh,
                         vertex_write_start_indices[k],
                         face_write_start_indices[k],
                         base_vertex_index_offsets[k]};
                    processSingleMeshForCombine(args);
                }
            });
        }
        for (auto& t : threads) { if (t.joinable()) { t.join(); } }
    }
    return combined_mesh;
}

#if 0
// --- Main Function (Now defines and uses meshes with normals and uvs) ---
int main() {
    Mesh cubeMesh; 
    // Define a cube with distinct normals per face (requires vertex duplication for attributes)
    // For this simple Mesh struct (attributes per vertex), we'll define smooth normals for shared vertices.
    // For distinct face normals, you'd typically have more vertices than geometric positions.
    // Let's define a cube with some example normals and UVs (assuming shared vertices where appropriate)
    cubeMesh.vertices = { // 8 vertices
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, 
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}  
    };
    // Example: Simple normals (can be refined for cube-like appearance if vertices were duplicated for hard edges)
    // For this example, let's just assign normalized position as normal (sphere-like normals for the cube corners)
    // or a generic up normal for all. Let's use distinct-ish normals for testing.
    cubeMesh.normals.resize(8);
    cubeMesh.normals[0] = glm::normalize(glm::vec3(-1,-1,-1)); cubeMesh.normals[1] = glm::normalize(glm::vec3( 1,-1,-1));
    cubeMesh.normals[2] = glm::normalize(glm::vec3( 1, 1,-1)); cubeMesh.normals[3] = glm::normalize(glm::vec3(-1, 1,-1));
    cubeMesh.normals[4] = glm::normalize(glm::vec3(-1,-1, 1)); cubeMesh.normals[5] = glm::normalize(glm::vec3( 1,-1, 1));
    cubeMesh.normals[6] = glm::normalize(glm::vec3( 1, 1, 1)); cubeMesh.normals[7] = glm::normalize(glm::vec3(-1, 1, 1));

    // Example UVs (very basic box mapping, might need duplication for proper unwrap)
    cubeMesh.uvs.resize(8);
    cubeMesh.uvs[0] = {0,0}; cubeMesh.uvs[1] = {1,0}; cubeMesh.uvs[2] = {1,1}; cubeMesh.uvs[3] = {0,1};
    cubeMesh.uvs[4] = {0,0}; cubeMesh.uvs[5] = {1,0}; cubeMesh.uvs[6] = {1,1}; cubeMesh.uvs[7] = {0,1}; // Top face UVs same as bottom for this simple example

    cubeMesh.faces = { // 12 faces
        Face(0,1,2), Face(0,2,3), Face(4,5,6), Face(4,6,7), 
        Face(0,1,5), Face(0,5,4), Face(1,2,6), Face(1,6,5), 
        Face(2,3,7), Face(2,7,6), Face(3,0,4), Face(3,4,7)  
    };
    printMesh(cubeMesh, "Test Mesh (Cube with attributes)");
    
    std::vector<Mesh> allMeshes;
    allMeshes.push_back(cubeMesh); 

    // ... (Combine meshes - though only one here for simplicity)
    Mesh mergedMesh = combineMeshesMultithreaded(allMeshes, 0); 
    printMesh(mergedMesh, "Mesh to be Simplified (with attributes)");

    std::cout << "\n--- Actual (BUT VERY BASIC) HLOD Simplification (with attribute consideration) ---" << std::endl;
    float sharpEdgeAngleThresholdDegrees = 75.0f; 
    float normalWeight = 10.0f; // How much dissimilarity in normals contributes to cost
    float uvWeight = 5.0f;    // How much distance in UVs contributes to cost

    int originalFaceCount = mergedMesh.faces.size();
    int targetSimplificationFaceCount = std::max(2, originalFaceCount / 3); 
    if (originalFaceCount <=2) targetSimplificationFaceCount = originalFaceCount;

    auto start_time_simplify = std::chrono::high_resolution_clock::now();
    Mesh hlodProxy = simplifyMeshActualButVeryBasic(
        mergedMesh,
        targetSimplificationFaceCount,
        sharpEdgeAngleThresholdDegrees,
        normalWeight,
        uvWeight
    );
    auto end_time_simplify = std::chrono::high_resolution_clock::now();
    std::cout << "\nActual (VERY BASIC) mesh simplification took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_simplify - start_time_simplify).count() << " milliseconds." << std::endl;
    printMesh(hlodProxy, "Simplified HLOD Proxy (Attributes Considered)");

    std::cout << "\nREMINDER: The simplification shown is a TOY IMPLEMENTATION." << std::endl;
    std::cout << "Attribute handling (normals, UVs) is very naive and will likely cause issues" << std::endl;
    std::cout << "like smoothed hard edges or distorted textures. UV seam preservation is NOT handled." << std::endl;

    return 0;
}
#endif

} // game_object
} // engine