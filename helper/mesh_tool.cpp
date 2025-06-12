#include <stack>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalFlippingT.hh>
#include "mesh_tool.h"

namespace engine {
namespace helper {

// --- OpenMesh HLOD Generation ---

// --- OpenMesh Structures (MyTraits, MyMesh) ---
// Unchanged, as these are type definitions.
struct MyTraits : public OpenMesh::DefaultTraits {
    VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge);
    FaceAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);
    VertexTraits {
    private:
        TexCoord2D tex_coord_;
    public:
        VertexT() : tex_coord_(TexCoord2D(0.0f, 0.0f)) {}
        const TexCoord2D& texcoord() const { return tex_coord_; }
        void set_texcoord(const TexCoord2D& _tex) { tex_coord_ = _tex; }
    };
};
typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> MyMesh;


/**
 * @brief Generates a simplified mesh using a robust method for vertex locking.
 * This version adheres to the specified naming conventions.
 *
 * @param input_mesh The original mesh to be simplified.
 * @param output_mesh The mesh structure to store the simplified result.
 * @param target_face_count The desired number of faces in the simplified mesh.
 * @param protected_vertex_indices A set of indices for vertices that should not be collapsed.
 */
void generateHLODWithSeamProtection(
    const Mesh& input_mesh,
    Mesh& output_mesh,
    size_t target_face_count,
    const std::set<uint32_t>& protected_vertex_indices) {

    // ... (Initial checks and mesh conversion logic remain the same) ...
    if (!input_mesh.isValid() || input_mesh.getFaceCount() == 0) { return; }
    if (target_face_count >= input_mesh.getFaceCount()) { output_mesh = input_mesh; return; }

    MyMesh om_mesh;
    om_mesh.request_vertex_status();
    om_mesh.request_edge_status();
    om_mesh.request_face_status();
    om_mesh.request_vertex_normals();
    om_mesh.request_vertex_texcoords2D();
    std::vector<MyMesh::VertexHandle> vertex_handles;
    for (const auto& input_vertex : *input_mesh.vertex_data_ptr) {
        MyMesh::VertexHandle vertex_handle = om_mesh.add_vertex(MyMesh::Point(input_vertex.position.x, input_vertex.position.y, input_vertex.position.z));
        om_mesh.set_normal(vertex_handle, MyMesh::Normal(input_vertex.normal.x, input_vertex.normal.y, input_vertex.normal.z));
        om_mesh.set_texcoord2D(vertex_handle, MyMesh::TexCoord2D(input_vertex.uv.x, input_vertex.uv.y));
        vertex_handles.push_back(vertex_handle);
    }
    for (const auto& input_face : *input_mesh.faces_ptr) {
        if (!input_face.isDegenerate()) {
            std::vector<MyMesh::VertexHandle> face_vertex_handles;
            face_vertex_handles.push_back(vertex_handles[input_face.v_indices[0]]);
            face_vertex_handles.push_back(vertex_handles[input_face.v_indices[1]]);
            face_vertex_handles.push_back(vertex_handles[input_face.v_indices[2]]);
            om_mesh.add_face(face_vertex_handles);
        }
    }
    for (uint32_t vertex_idx : protected_vertex_indices) {
        if (vertex_idx < vertex_handles.size()) {
            om_mesh.status(vertex_handles[vertex_idx]).set_locked(true);
        }
    }

    // --- Perform Decimation using the API from your header files ---
    OpenMesh::Decimater::DecimaterT<MyMesh> decimater(om_mesh);

    // Define the correct handle type for the quadric module.
    OpenMesh::Decimater::ModHandleT<OpenMesh::Decimater::ModQuadricT<MyMesh>> quadric_handle;

    // Register the module using the 'add' function found in BaseDecimaterT.hh
    decimater.add(quadric_handle); // This will create the module and populate the handle.

    // 2. Define a handle for the normal flipping module (to preserve features)
    OpenMesh::Decimater::ModHandleT<OpenMesh::Decimater::ModNormalFlippingT<MyMesh>> normal_flipping_handle;
    decimater.add(normal_flipping_handle); //

    // --- Configuration ---

    // Configure the normal flipping module
    // This tells the decimater to forbid any collapse that would change a face's
    // normal by more than 30 degrees. This is very effective at keeping sharp edges.
    decimater.module(normal_flipping_handle).set_max_normal_deviation(30.0f); //

    // Initialize the decimater. This should now succeed.
    decimater.initialize(); //

    std::cout << "Decimater initialized: " << std::boolalpha << decimater.is_initialized() << std::endl; //
    if (!decimater.is_initialized()) {
        std::cout << "[ERROR] Decimater failed to initialize." << std::endl;
        output_mesh = input_mesh;
        return;
    }

    // Call decimate_to_faces
    decimater.decimate_to_faces(0, target_face_count);

    om_mesh.garbage_collection();

    // ... (Conversion back to your format - Unchanged) ...
    output_mesh.vertex_data_ptr->clear();
    output_mesh.faces_ptr->clear();
    std::map<int, int> old_to_new_indices;
    int new_vertex_index = 0;
    for (MyMesh::VertexIter v_it = om_mesh.vertices_begin(); v_it != om_mesh.vertices_end(); ++v_it) {
        VertexStruct new_vertex;
        MyMesh::Point point = om_mesh.point(*v_it);
        new_vertex.position = glm::vec3(point[0], point[1], point[2]);
        MyMesh::Normal normal = om_mesh.normal(*v_it);
        new_vertex.normal = glm::vec3(normal[0], normal[1], normal[2]);
        MyMesh::TexCoord2D uv_coord = om_mesh.texcoord2D(*v_it);
        new_vertex.uv = glm::vec2(uv_coord[0], uv_coord[1]);
        output_mesh.vertex_data_ptr->push_back(new_vertex);
        old_to_new_indices[v_it->idx()] = new_vertex_index++;
    }
    for (MyMesh::FaceIter f_it = om_mesh.faces_begin(); f_it != om_mesh.faces_end(); ++f_it) {
        Face new_face;
        int i = 0;
        for (MyMesh::FaceVertexIter fv_it = om_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it, ++i) {
            new_face.v_indices[i] = old_to_new_indices[fv_it->idx()];
        }
        output_mesh.faces_ptr->push_back(new_face);
    }
}

} // game_object
} // engine