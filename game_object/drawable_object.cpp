#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <set>
#include <filesystem>
#include <mutex>

#include "helper/engine_helper.h"
#include "helper/bvh.h"
#include "helper/mesh_tool.h"
#include "helper/model_inspect.h"
#include "game_object/drawable_object.h"
#include "game_object/mesh_load_task_manager.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

// gltf
#include "tiny_gltf.h"
#include "helper/tiny_mtx2.h"

// fbx
#include "third_parties/fbx/ufbx.h"

static uint32_t num_draw_meshes = 0;
#define DEBUG_OUTPUT 1
#define HASH_CHECK 0

// ── Per-frame frustum cull state (set by application.cpp) ──────────────
static bool     s_frustum_cull_active = false;
static glm::vec4 s_frustum_planes[6];

namespace ego = engine::game_object;

// ── ECS material-dedup capture ──────────────────────────────────────────
// Fill MaterialInfo::desc_ (the renderer-free ecs::MaterialDesc dedup
// identity) from the just-built PbrMaterialParams UBO.  Called once per
// material by every loader (glTF / FBX / .rwobj / .rwchar) right after the
// UBO fill, so the desc always mirrors what the GPU material actually
// renders with.  Texture slots key by the texture's source filename when
// the loader recorded one (real path → dedups across assets); otherwise by
// "<asset_key>#<index>" (embedded textures → dedups across instances of
// the same asset, never falsely across different assets).
static void captureMaterialDesc(
    ego::MaterialInfo& m,
    const glsl::PbrMaterialParams& ubo,
    const std::vector<engine::renderer::TextureInfo>& textures,
    const std::string& asset_key) {
    auto tex_key = [&](int32_t idx) -> std::string {
        if (idx < 0) return std::string();
        if (idx < (int32_t)textures.size() &&
            !textures[idx].source_filename_.empty())
            return textures[idx].source_filename_;
        return asset_key + "#" + std::to_string(idx);
    };
    auto& d = m.desc_;
    d.base_color   = ubo.base_color_factor;
    d.emissive     = ubo.emissive_factor;
    d.metallic     = ubo.metallic_factor;
    d.roughness    = ubo.roughness_factor;
    d.alpha_cutoff = ubo.alpha_cutoff;
    d.alpha_mode   =
        m.alpha_mode_ == ego::AlphaMode::Blend ? engine::ecs::AlphaMode::kBlend :
        m.alpha_mode_ == ego::AlphaMode::Mask  ? engine::ecs::AlphaMode::kMask
                                               : engine::ecs::AlphaMode::kOpaque;
    d.base_color_tex         = tex_key(m.base_color_idx_);
    d.normal_tex             = tex_key(m.normal_idx_);
    d.metallic_roughness_tex = tex_key(m.metallic_roughness_idx_);
    d.emissive_tex           = tex_key(m.emissive_idx_);
    d.occlusion_tex          = tex_key(m.occlusion_idx_);
}
namespace engine {

namespace {

struct VertexHashInfo {
    uint32_t indice;
#if HASH_CHECK
    helper::VertexStruct vertex;
#endif
};


size_t hashCombine(
    size_t seed,
    size_t value) {
    // A variation based on Boost and CityHash
    const uint64_t kMul = 0x9ddfea08eb382d69ULL;
    uint64_t a = (value ^ seed) * kMul;
    a ^= (a >> 47);
    uint64_t b = (seed ^ a) * kMul;
    b ^= (b >> 47);
    b *= kMul;
    return b;
}

size_t hashCombine(
    const helper::VertexStruct& vertex) {
    uint64_t t0 =
        ((uint64_t)(*(uint32_t*)(&vertex.position.x)) << 32) |
        (uint64_t)(*(uint32_t*)(&vertex.normal.z));
    uint64_t t1 =
        ((uint64_t)(*(uint32_t*)(&vertex.position.y)) << 32) |
        (uint64_t)(*(uint32_t*)(&vertex.uv.x));
    uint64_t t2 =
        ((uint64_t)(*(uint32_t*)(&vertex.normal.y)) << 32) |
        (uint64_t)(*(uint32_t*)(&vertex.uv.y));
    uint64_t t3 =
        ((uint64_t)(*(uint32_t*)(&vertex.position.z)) << 32) |
        (uint64_t)(*(uint32_t*)(&vertex.normal.x));

    size_t hash = hashCombine(0, t0);
    hash = hashCombine(hash, t1);
    hash = hashCombine(hash, t2);
    hash = hashCombine(hash, t3);
    return hash;
}

// Caches a vertex index based on a hash of its data. If the vertex data already exists in the map, returns the existing index.
// Otherwise, adds the new vertex index to the match table and map, and returns the new index.
static uint32_t cacheVertexIndice(
    std::unordered_map<size_t, VertexHashInfo>& vertex_map,       // Map from vertex hash to index
    std::vector<uint32_t>& indice_match_table,              // Table mapping new indices to source vertex indices
    const helper::VertexStruct& vertex_data,                               // Pointer to vertex data (float array)
    uint32_t src_vert_index)                                // Source vertex index
{
    uint32_t new_indice = 0;
    // Compute a hash value for the vertex data
    auto hash_value = hashCombine(vertex_data);

    // Check if this vertex data already exists in the map
    auto it = vertex_map.find(hash_value);
    if (it != vertex_map.end()) {
        // Vertex already exists, use the existing index
        new_indice = it->second.indice;
#if HASH_CHECK
        assert(vertex_data.position.x == it->second.vertex.position.x);
        assert(vertex_data.position.y == it->second.vertex.position.y);
        assert(vertex_data.position.z == it->second.vertex.position.z);
        assert(vertex_data.normal.x == it->second.vertex.normal.x);
        assert(vertex_data.normal.y == it->second.vertex.normal.y);
        assert(vertex_data.normal.z == it->second.vertex.normal.z);
        assert(vertex_data.uv.x == it->second.vertex.uv.x);
        assert(vertex_data.uv.y == it->second.vertex.uv.y);
#endif
    }
    else {
        // Vertex is new, assign a new index and add to the match table and map
        new_indice = static_cast<uint32_t>(indice_match_table.size());
        indice_match_table.push_back(src_vert_index);
#if HASH_CHECK
        vertex_map[hash_value].vertex = vertex_data;
#endif
        vertex_map[hash_value].indice = static_cast<uint32_t>(new_indice);
    }

    return new_indice;
}

// Returns the new or existing index for a given input vertex index, updating the match table and original index list as needed.
static int32_t getNewIndice(
    std::vector<int32_t>& remap_table,      // Table mapping input indices to new indices
    std::vector<int32_t>& org_vertex_indice_list,  // List of original vertex indices in order of new indices
    uint32_t input_idx)                            // The input vertex index to map
{
    // If this input index hasn't been assigned a new index yet
    if (remap_table[input_idx] < 0) {
        // Assign the next available new index
        remap_table[input_idx] =
            int32_t(org_vertex_indice_list.size());
        // Store the original input index in the list
        org_vertex_indice_list.push_back(input_idx);
    }

    // Return the new index for this input index
    return remap_table[input_idx];
}

// Packs mesh indices: remaps input mesh vertex indices to a compacted list and updates faces accordingly.
// - output_mesh: mesh to write packed indices into
// - org_vertex_indice_list: output list of original vertex indices in new order
// - input_mesh: mesh to read from
static void packMeshPatchIndice(
    helper::Mesh& output_mesh,
    const helper::Mesh& input_mesh,
    std::vector<int32_t>& org_vertex_indice_list) {
    const auto& num_faces = input_mesh.faces_ptr->size();
    const auto& num_vertex = input_mesh.vertex_data_ptr->size();

    output_mesh.faces_ptr->resize(num_faces);
    std::vector<int32_t> remap_table(num_vertex);
    for (auto& idx : remap_table) {
        idx = -1;
    }

    org_vertex_indice_list.reserve(num_vertex);
    for (auto i_face = 0; i_face < num_faces; i_face++) {
        const auto& src_face = input_mesh.faces_ptr->at(i_face);
        auto& dest_face = output_mesh.faces_ptr->at(i_face);
        // Remap each vertex index in the face to a new compacted index
        dest_face = helper::Face(
            getNewIndice(
                remap_table,
                org_vertex_indice_list,
                src_face.v_indices[0]),
            getNewIndice(
                remap_table,
                org_vertex_indice_list,
                src_face.v_indices[1]),
            getNewIndice(
                remap_table,
                org_vertex_indice_list,
                src_face.v_indices[2]));

    }
}

// Packs mesh vertices: copies only the used vertices in the new order into output_mesh.
// - output_mesh: mesh to write packed vertices into
// - org_vertex_indice_list: list of original vertex indices in new order
// - input_mesh: mesh to read from
static void packMeshPatchVertex(
    helper::Mesh& output_mesh,
    const helper::Mesh& input_mesh,
    const std::vector<int32_t>& org_vertex_indice_list) {

    const auto& num_vertex = org_vertex_indice_list.size();
    output_mesh.vertex_data_ptr->resize(num_vertex);
    for (uint32_t i_vert = 0; i_vert < num_vertex; i_vert++) {
        output_mesh.vertex_data_ptr->at(i_vert) =
            input_mesh.vertex_data_ptr->at(org_vertex_indice_list[i_vert]);
    }
}

// Packs a mesh by compacting its vertices and remapping faces to the new indices.
// - output_mesh: mesh to write packed data into
// - input_mesh: mesh to read from
static void packMeshPatch(
    helper::Mesh& output_mesh,
    const helper::Mesh& input_mesh) {
    const auto& num_faces = input_mesh.faces_ptr->size();
    const auto& num_vertex = input_mesh.vertex_data_ptr->size();

    if (input_mesh.isValid() == false ||
        output_mesh.isValid() == false) {
        return;
    }

    std::vector<int32_t> org_vertex_indice_list;
    // Remap indices and collect used vertex indices
    packMeshPatchIndice(
        output_mesh,
        input_mesh,
        org_vertex_indice_list);

    // Copy only used vertices in new order
    packMeshPatchVertex(
        output_mesh,
        input_mesh,
        org_vertex_indice_list);
}

// Merges a mesh patch into an output mesh, deduplicating vertices and remapping indices.
// - vertex_map: hash map for deduplicating vertices
// - indice_match_table: table mapping new indices to source vertex indices
// - output_mesh: mesh to merge into
// - input_mesh: mesh patch to merge
static void mergeMeshPatch(
    helper::Mesh& output_mesh,
    const helper::Mesh& input_mesh,
    std::unordered_map<size_t, VertexHashInfo>& vertex_map,
    std::vector<uint32_t>& indice_match_table) {

    if (input_mesh.isValid() == false ||
        output_mesh.isValid() == false) {
        return;
    }

    // Pack input mesh to remove unused vertices and remap indices
    helper::Mesh packed_input_mesh;
    std::vector<int32_t> org_vertex_indice_list;
    packMeshPatchIndice(
        packed_input_mesh,
        input_mesh,
        org_vertex_indice_list);

    // For each packed vertex, deduplicate and add to output mesh if new
    std::vector<int32_t> deduped_vertex_indices(org_vertex_indice_list.size());
    for (uint32_t i_indice = 0; i_indice < org_vertex_indice_list.size(); i_indice++) {
        auto& cur_vertex = input_mesh.vertex_data_ptr->at(org_vertex_indice_list[i_indice]);
        auto old_table_size = indice_match_table.size();
        deduped_vertex_indices[i_indice] =
            cacheVertexIndice(
                vertex_map,
                indice_match_table,
                cur_vertex,
                uint32_t(output_mesh.vertex_data_ptr->size()));
        // When vertex is not in cache, add new vertex to vertex buffer.
        if (indice_match_table.size() > old_table_size) {
            output_mesh.vertex_data_ptr->push_back(cur_vertex);
        }
    }

    // Remap face indices to new deduplicated indices and add to output mesh
    for (auto& face: *packed_input_mesh.faces_ptr) {
        output_mesh.faces_ptr->push_back(
            helper::Face(
                deduped_vertex_indices[face.v_indices[0]],
                deduped_vertex_indices[face.v_indices[1]],
                deduped_vertex_indices[face.v_indices[2]]));
    }
}

static std::string getFilePathExtension(const std::string& file_name) {
    if (file_name.find_last_of(".") != std::string::npos)
        return file_name.substr(file_name.find_last_of(".") + 1);
    return "";
}

static glm::quat eulerToQuaternion(float roll, float pitch, float yaw) {
    // Convert degrees to radians
    roll = glm::radians(roll);
    pitch = glm::radians(pitch);
    yaw = glm::radians(yaw);

    // Create quaternion from Euler angles
    // GLM uses the yaw (Z), pitch (Y), and roll (X) order
    glm::quat quaternion = glm::quat(glm::vec3(pitch, yaw, roll));

    // Normalize the quaternion (optional, as GLM's constructor returns a normalized quaternion)
    return glm::normalize(quaternion);
}

static void transformBbox(
    const glm::mat4& mat,
    const glm::vec3& bbox_min,
    const glm::vec3& bbox_max,
    glm::vec3& output_bbox_min,
    glm::vec3& output_bbox_max) {

    glm::vec3 extent = bbox_max - bbox_min;
    glm::vec3 base = glm::vec3(mat * glm::vec4(bbox_min, 1.0f));
    output_bbox_min = base;
    output_bbox_max = base;
    auto mat_1 = glm::mat3(mat);
    glm::vec3 vec_x = mat_1 * glm::vec3(extent.x, 0, 0);
    glm::vec3 vec_y = mat_1 * glm::vec3(0, extent.y, 0);
    glm::vec3 vec_z = mat_1 * glm::vec3(0, 0, extent.z);

    glm::vec3 points[7];
    points[0] = base + vec_x;
    points[1] = base + vec_y;
    points[2] = base + vec_z;
    points[3] = points[0] + vec_y;
    points[4] = points[0] + vec_z;
    points[5] = points[1] + vec_z;
    points[6] = points[3] + vec_z;

    for (int i = 0; i < 7; i++) {
        output_bbox_min = min(output_bbox_min, points[i]);
        output_bbox_max = max(output_bbox_max, points[i]);
    }
}

static void calculateBbox(
    std::shared_ptr<ego::DrawableData>& drawable_object,
    int32_t node_idx,
    const glm::mat4& parent_matrix,
    glm::vec3& output_bbox_min,
    glm::vec3& output_bbox_max) {
    if (node_idx >= 0) {
        const auto& node = drawable_object->nodes_[node_idx];
        auto cur_matrix = parent_matrix;
        cur_matrix *= node.matrix_;

        if (node.mesh_idx_ >= 0) {
            glm::vec3 bbox_min, bbox_max;
            transformBbox(
                cur_matrix,
                drawable_object->meshes_[node.mesh_idx_].bbox_min_,
                drawable_object->meshes_[node.mesh_idx_].bbox_max_,
                bbox_min,
                bbox_max);
            output_bbox_min = min(output_bbox_min, bbox_min);
            output_bbox_max = max(output_bbox_max, bbox_max);
        }

        for (auto& child_idx : node.child_idx_) {
            calculateBbox(drawable_object, child_idx, cur_matrix, output_bbox_min, output_bbox_max);
        }
    }
}

static void setupMeshState(
    const std::shared_ptr<renderer::Device>& device,
    const tinygltf::Model& model,
    std::shared_ptr<ego::DrawableData>& drawable_object,
    const std::string& asset_key) {

    // Buffer
    {
        drawable_object->buffers_.resize(model.buffers.size());
        for (size_t i = 0; i < model.buffers.size(); i++) {
            auto buffer = model.buffers[i];
            renderer::Helper::createBuffer(
                device,
                SET_5_FLAG_BITS(
                    BufferUsage,
                    VERTEX_BUFFER_BIT,
                    INDEX_BUFFER_BIT,
                    SHADER_DEVICE_ADDRESS_BIT,
                    STORAGE_BUFFER_BIT,
                    ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
                SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
                drawable_object->buffers_[i].buffer,
                drawable_object->buffers_[i].memory,
                std::source_location::current(),
                buffer.data.size(),
                buffer.data.data());
        }
    }

    // Buffer views.
    {
        auto& buffer_views = drawable_object->buffer_views_;
        buffer_views.resize(model.bufferViews.size());

        for (size_t i = 0; i < model.bufferViews.size(); i++) {
            const tinygltf::BufferView& bufferView = model.bufferViews[i];
            buffer_views[i].buffer_idx = bufferView.buffer;
            buffer_views[i].offset = bufferView.byteOffset;
            buffer_views[i].range = bufferView.byteLength;
            buffer_views[i].stride = bufferView.byteStride;
        }
    }

    // allocate texture memory at first.
    drawable_object->textures_.resize(model.textures.size());

    // Material
    {
        drawable_object->materials_.resize(model.materials.size());
        for (size_t i_mat = 0; i_mat < model.materials.size(); i_mat++) {
            auto& dst_material = drawable_object->materials_[i_mat];
            const auto& src_material = model.materials[i_mat];

            // Source-asset material name for gameplay surface lookup.
            dst_material.name_ = src_material.name;

            dst_material.base_color_idx_ = src_material.pbrMetallicRoughness.baseColorTexture.index;
            dst_material.normal_idx_ = src_material.normalTexture.index;
            dst_material.metallic_roughness_idx_ = src_material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            dst_material.emissive_idx_ = src_material.emissiveTexture.index;
            dst_material.occlusion_idx_ = src_material.occlusionTexture.index;

            // ── KHR_materials_pbrSpecularGlossiness fallback ──────────
            // The standard glTF PBR workflow is pbrMetallicRoughness +
            // baseColorTexture.  Older Sketchfab / Mixamo / Substance
            // exports often ship with the KHR_materials_pbrSpecularGlossiness
            // extension instead, which puts the albedo under
            //   material.extensions["KHR_materials_pbrSpecularGlossiness"]
            //     .diffuseTexture.index
            // …and leaves pbrMetallicRoughness.baseColorTexture absent
            // (index == -1).  Without this fallback the material's
            // base_color_idx_ stays at -1, no albedo binds, and the
            // mesh renders with the default white baseColorFactor —
            // exactly the "untextured porcelain doll" symptom we hit on
            // scene-skinned.gltf.  Treat the spec-gloss diffuse* fields
            // as the albedo source when the standard fields are absent.
            //
            // We deliberately do NOT translate full spec-gloss lighting
            // into metallic-roughness here (proper conversion needs the
            // GGX→Phong remap which the renderer doesn't carry CPU-side).
            // The character still won't have correct specular/roughness
            // response under this fallback, but it WILL have the right
            // diffuse/albedo colour — which is what the user is asking
            // about ("fix texture issue on character").
            {
                auto ext_it = src_material.extensions.find(
                    "KHR_materials_pbrSpecularGlossiness");
                if (ext_it != src_material.extensions.end() &&
                    ext_it->second.IsObject()) {
                    const auto& sg = ext_it->second;

                    // diffuseTexture → base_color (only if standard
                    // baseColorTexture is absent, so a properly-authored
                    // dual-workflow asset wins through the standard path).
                    if (dst_material.base_color_idx_ < 0 &&
                        sg.Has("diffuseTexture")) {
                        const auto& dt = sg.Get("diffuseTexture");
                        if (dt.IsObject() && dt.Has("index") &&
                            dt.Get("index").IsNumber()) {
                            dst_material.base_color_idx_ =
                                dt.Get("index").GetNumberAsInt();
                        }
                    }

                    // specularGlossinessTexture → reuse the metallic-
                    // roughness slot (alpha = glossiness, RGB = specular
                    // colour).  The fragment shader's MR sampling will
                    // pull wrong channels but at least lights the surface
                    // instead of leaving the slot empty.  Better than
                    // nothing for a debug-mode character.
                    if (dst_material.metallic_roughness_idx_ < 0 &&
                        sg.Has("specularGlossinessTexture")) {
                        const auto& st = sg.Get("specularGlossinessTexture");
                        if (st.IsObject() && st.Has("index") &&
                            st.Get("index").IsNumber()) {
                            dst_material.metallic_roughness_idx_ =
                                st.Get("index").GetNumberAsInt();
                        }
                    }
                }
            }
            dst_material.alpha_cutoff_ = static_cast<float>(src_material.alphaCutoff);
            // ── Alpha mode (Opaque / Mask / Blend) ─────────────────────
            // Authoritative source: glTF alphaMode field. We then layer
            // a substring-match override on the material NAME so common
            // glass authoring (e.g. material called "GlassWindow") gets
            // promoted to Blend even when the asset author left
            // alphaMode at its OPAQUE default.
            if (src_material.alphaMode == "MASK") {
                dst_material.alpha_mode_ = ego::AlphaMode::Mask;
            } else if (src_material.alphaMode == "BLEND") {
                dst_material.alpha_mode_ = ego::AlphaMode::Blend;
            } else {
                dst_material.alpha_mode_ = ego::AlphaMode::Opaque;
            }
            {
                const std::string& mn = src_material.name;
                auto contains = [&mn](const char* needle) {
                    return mn.find(needle) != std::string::npos;
                };
                if (contains("glass")  || contains("Glass")  ||
                    contains("window") || contains("Window") ||
                    contains("transparent") || contains("Transparent")) {
                    dst_material.alpha_mode_   = ego::AlphaMode::Blend;
                    dst_material.glass_forced_ = true;
                }
            }
            dst_material.alpha_mask_ =
                (dst_material.alpha_mode_ == ego::AlphaMode::Mask);

            if (dst_material.base_color_idx_ >= 0) {
                drawable_object->textures_[dst_material.base_color_idx_].linear = false;
            }

            if (dst_material.emissive_idx_ >= 0) {
                drawable_object->textures_[dst_material.emissive_idx_].linear = false;
            }

            device->createBuffer(
                sizeof(glsl::PbrMaterialParams),
                SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
                0,
                dst_material.uniform_buffer_.buffer,
                dst_material.uniform_buffer_.memory,
                std::source_location::current());

            glsl::PbrMaterialParams ubo{};
            ubo.base_color_factor = glm::vec4(
                src_material.pbrMetallicRoughness.baseColorFactor[0],
                src_material.pbrMetallicRoughness.baseColorFactor[1],
                src_material.pbrMetallicRoughness.baseColorFactor[2],
                src_material.pbrMetallicRoughness.baseColorFactor[3]);

            // Spec-gloss diffuseFactor override (paired with the
            // diffuseTexture fallback above).  Only applies when the
            // material has the KHR_materials_pbrSpecularGlossiness
            // extension AND we already pulled an albedo from it (or
            // when the standard baseColorFactor is the default white
            // sentinel and the extension supplies a colour).
            {
                auto ext_it = src_material.extensions.find(
                    "KHR_materials_pbrSpecularGlossiness");
                if (ext_it != src_material.extensions.end() &&
                    ext_it->second.IsObject()) {
                    const auto& sg = ext_it->second;
                    if (sg.Has("diffuseFactor")) {
                        const auto& df = sg.Get("diffuseFactor");
                        if (df.IsArray() && df.ArrayLen() >= 4) {
                            glm::vec4 dv(
                                (float)df.Get(0).GetNumberAsDouble(),
                                (float)df.Get(1).GetNumberAsDouble(),
                                (float)df.Get(2).GetNumberAsDouble(),
                                (float)df.Get(3).GetNumberAsDouble());
                            // Replace base_color_factor when the
                            // standard slot is still at its (1,1,1,1)
                            // default (meaning the material author
                            // didn't fill in a base color factor).
                            // Otherwise both factors are present and we
                            // honor the standard one.
                            const auto& bc =
                                src_material.pbrMetallicRoughness.baseColorFactor;
                            const bool standard_is_default =
                                bc[0] == 1.0 && bc[1] == 1.0 &&
                                bc[2] == 1.0 && bc[3] == 1.0;
                            if (standard_is_default) {
                                ubo.base_color_factor = dv;
                            }
                        }
                    }
                }
            }

            // Force glass-tagged materials translucent if the asset
            // author left them at full alpha. 0.4 reads as obvious
            // glass without going invisible.
            if (dst_material.glass_forced_ && ubo.base_color_factor.a >= 0.95f) {
                ubo.base_color_factor.a = 0.4f;
            }

            ubo.glossiness_factor = 1.0f;
            ubo.metallic_roughness_specular_factor = 1.0f;
            ubo.metallic_factor = static_cast<float>(src_material.pbrMetallicRoughness.metallicFactor);
            ubo.roughness_factor = static_cast<float>(src_material.pbrMetallicRoughness.roughnessFactor);
            ubo.alpha_cutoff = static_cast<float>(src_material.alphaCutoff);
            ubo.mip_count = 11;
            ubo.normal_scale = static_cast<float>(src_material.normalTexture.scale);
            ubo.occlusion_strength = static_cast<float>(src_material.occlusionTexture.strength);

            ubo.emissive_factor = glm::vec3(
                src_material.emissiveFactor[0],
                src_material.emissiveFactor[1],
                src_material.emissiveFactor[2]);

            ubo.emissive_color = glm::vec3(1.0f);

            ubo.uv_set_flags = glm::vec4(0, 0, 0, 0);
            ubo.exposure = 1.0f;
            // Check dst_material.* (already populated with the
            // spec-gloss fallback) rather than the raw src_material
            // fields — otherwise a KHR_materials_pbrSpecularGlossiness
            // asset has the texture bound at the descriptor level but
            // FEATURE_HAS_BASE_COLOR_MAP unset, so the fragment shader
            // skips sampling it and falls back to base_color_factor.
            ubo.material_features = (dst_material.metallic_roughness_idx_ >= 0 ? (FEATURE_HAS_METALLIC_ROUGHNESS_MAP | FEATURE_HAS_METALLIC_CHANNEL) : 0) | FEATURE_MATERIAL_METALLICROUGHNESS;
            ubo.material_features |= (dst_material.base_color_idx_ >= 0 ? FEATURE_HAS_BASE_COLOR_MAP : 0);
            ubo.material_features |= (src_material.emissiveTexture.index >= 0 ? FEATURE_HAS_EMISSIVE_MAP : 0);
            ubo.material_features |= (src_material.occlusionTexture.index >= 0 ? FEATURE_HAS_OCCLUSION_MAP : 0);
            ubo.material_features |= (src_material.normalTexture.index >= 0 ? FEATURE_HAS_NORMAL_MAP : 0);
            // Propagate the CPU-side AlphaMode so the fragment shader can
            // dispatch on translucency (currently consumed by the
            // "Translucent" render-debug visualisation).  Glass-by-name
            // overrides earlier in the loader force AlphaMode::Blend, so
            // this also flags forced-glass materials as translucent.
            ubo.material_features |= (dst_material.alpha_mode_ == ego::AlphaMode::Blend ? FEATURE_MATERIAL_BLEND : 0);
            ubo.material_features |= (dst_material.alpha_mode_ == ego::AlphaMode::Mask  ? FEATURE_MATERIAL_ALPHA_MASK : 0);
            ubo.tonemap_type = TONEMAP_DEFAULT;
            ubo.specular_factor = glm::vec3(1.0f, 1.0f, 1.0f);
            ubo.specular_color = glm::vec3(1.0f, 1.0f, 1.0f);
            ubo.specular_exponent = 1.0f;

            device->updateBufferMemory(dst_material.uniform_buffer_.memory, sizeof(ubo), &ubo);

            // ECS dedup identity — mirrors exactly what was uploaded above.
            captureMaterialDesc(
                dst_material, ubo, drawable_object->textures_, asset_key);
        }
    }

    // Texture
    {
        for (size_t i_tex = 0; i_tex < model.textures.size(); i_tex++) {
            auto& dst_tex = drawable_object->textures_[i_tex];
            const auto& src_tex = model.textures[i_tex];
            const auto& src_img = model.images[i_tex];
            auto format = renderer::Format::R8G8B8A8_UNORM;
            renderer::Helper::create2DTextureImage(
                device,
                format,
                src_img.width,
                src_img.height,
//                src_img.component,
                src_img.image.data(),
                dst_tex.image,
                dst_tex.memory,
                std::source_location::current());

            // Populate dst_tex.size — the glTF loader had been
            // leaving it at the default glm::uvec3(0).  Mirror what
            // other loaders (renderer.cpp:572/830, engine_helper.cpp
            // :161) already do.
            dst_tex.size = glm::uvec3(
                uint32_t(src_img.width), uint32_t(src_img.height), 1u);

            // Stash the raw RGBA8 pixel data so the VT manager can
            // build its bordered tile pyramid from CPU memory at
            // registerMaterial time — no GPU readback needed.  Copy
            // (not move) since tinygltf still owns the model and
            // other code paths might read src_img.image too.
            dst_tex.cpu_pixels = std::make_shared<std::vector<uint8_t>>(
                src_img.image.begin(), src_img.image.end());

            dst_tex.view = device->createImageView(
                dst_tex.image,
                renderer::ImageViewType::VIEW_2D,
                format,
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                std::source_location::current());
        }
    }

    // NOTE: effective_opaque_ scan is deferred to the DrawableObject
    // constructor (so the FBX path gets it too) — see the
    // computeEffectiveOpaqueForMaterials(object_) call right after the
    // loadFbxModel/loadGltfModel branch.
}

static void setupMesh(
    const tinygltf::Model& model,
    const tinygltf::Mesh& src_mesh,
    ego::MeshInfo& mesh_info) {

    for (size_t i = 0; i < src_mesh.primitives.size(); i++) {
        const tinygltf::Primitive& primitive = src_mesh.primitives[i];

        ego::PrimitiveInfo primitive_info;
        primitive_info.tag_.restart_enable = false;
        primitive_info.tag_.double_sided = model.materials[primitive.material].doubleSided;
        primitive_info.material_idx_ = primitive.material;

        auto mode = renderer::PrimitiveTopology::MAX_ENUM;
        if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
            mode = renderer::PrimitiveTopology::TRIANGLE_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
            mode = renderer::PrimitiveTopology::TRIANGLE_STRIP;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) {
            mode = renderer::PrimitiveTopology::TRIANGLE_FAN;
        }
        else if (primitive.mode == TINYGLTF_MODE_POINTS) {
            mode = renderer::PrimitiveTopology::POINT_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE) {
            mode = renderer::PrimitiveTopology::LINE_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE_LOOP) {
            mode = renderer::PrimitiveTopology::LINE_STRIP;
        }
        else {
            assert(0);
        }

        primitive_info.tag_.topology = static_cast<uint32_t>(mode);

        if (primitive.indices < 0) return;

        std::map<std::string, int>::const_iterator it(primitive.attributes.begin());
        std::map<std::string, int>::const_iterator itEnd(primitive.attributes.end());

        uint32_t dst_binding = 0;
        for (; it != itEnd; it++) {
            assert(it->second >= 0);
            const tinygltf::Accessor& accessor = model.accessors[it->second];

            // Custom attributes (leading '_', e.g. the auto-rig's
            // _CLOSENESS_0/_CLOSENESS_1 debug payloads) have no vertex
            // input slot — skip them instead of asserting.
            if (!it->first.empty() && it->first[0] == '_') continue;

            assert(dst_binding < VINPUT_INSTANCE_BINDING_POINT);

            engine::renderer::VertexInputBindingDescription binding = {};
            binding.binding = dst_binding;
            binding.stride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
            binding.input_rate = renderer::VertexInputRate::VERTEX;
            primitive_info.binding_descs_.push_back(binding);

            engine::renderer::VertexInputAttributeDescription attribute = {};
            attribute.buffer_view = accessor.bufferView;
            attribute.binding = dst_binding;
            attribute.offset = 0;
            attribute.buffer_offset = accessor.byteOffset + model.bufferViews[accessor.bufferView].byteOffset;
            if (it->first.compare("POSITION") == 0) {
                attribute.location = VINPUT_POSITION;
                primitive_info.bbox_min_ = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
                primitive_info.bbox_max_ = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
                mesh_info.bbox_min_ = min(mesh_info.bbox_min_, primitive_info.bbox_min_);
                mesh_info.bbox_max_ = max(mesh_info.bbox_max_, primitive_info.bbox_max_);
            }
            else if (it->first.compare("TEXCOORD_0") == 0) {
                attribute.location = VINPUT_TEXCOORD0;
                primitive_info.tag_.has_texcoord_0 = true;
            }
            else if (it->first.compare("NORMAL") == 0) {
                attribute.location = VINPUT_NORMAL;
                primitive_info.tag_.has_normal = true;
            }
            else if (it->first.compare("TANGENT") == 0) {
                attribute.location = VINPUT_TANGENT;
                primitive_info.tag_.has_tangent = true;
            }
            else if (it->first.compare("TEXCOORD_1") == 0) {
                attribute.location = VINPUT_TEXCOORD1;
            }
            else if (it->first.compare("COLOR") == 0) {
                attribute.location = VINPUT_COLOR;
            }
            else if (it->first.compare("JOINTS_0") == 0) {
                attribute.location = VINPUT_JOINTS_0;
                primitive_info.tag_.has_skin_set_0 = true;
            }
            else if (it->first.compare("WEIGHTS_0") == 0) {
                attribute.location = VINPUT_WEIGHTS_0;
                primitive_info.tag_.has_skin_set_0 = true;
            }
            else if (it->first.compare("JOINTS_1") == 0) {
                // 8-bone skinning debug: second skin set (influences 4..7).
                attribute.location = VINPUT_JOINTS_1;
                primitive_info.tag_.has_skin_set_1 = true;
            }
            else if (it->first.compare("WEIGHTS_1") == 0) {
                attribute.location = VINPUT_WEIGHTS_1;
                primitive_info.tag_.has_skin_set_1 = true;
            }
            else {
                // add support here.
                assert(0);
            }

            if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_FLOAT) {
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    attribute.format = engine::renderer::Format::R32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    attribute.format = engine::renderer::Format::R32G32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    attribute.format = engine::renderer::Format::R32G32B32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    attribute.format = engine::renderer::Format::R32G32B32A32_SFLOAT;
                }
                else {
                    assert(0);
                }
            }
            else if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    attribute.format = engine::renderer::Format::R16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    attribute.format = engine::renderer::Format::R16G16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    attribute.format = engine::renderer::Format::R16G16B16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    attribute.format = engine::renderer::Format::R16G16B16A16_UINT;
                }
                else {
                    assert(0);
                }

            }
            else {
                // add support here.
                assert(0);
            }
            primitive_info.attribute_descs_.push_back(attribute);
            dst_binding++;
        }

        const auto& indexAccessor = model.accessors[primitive.indices];
        primitive_info.index_desc_.emplace_back();
        primitive_info.index_desc_.back().buffer_view = indexAccessor.bufferView;
        primitive_info.index_desc_.back().offset = indexAccessor.byteOffset;
        primitive_info.index_desc_.back().index_type =
            indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 
            renderer::IndexType::UINT16 : 
            renderer::IndexType::UINT32;
        primitive_info.index_desc_.back().index_count = indexAccessor.count;

        primitive_info.generateHash();
        mesh_info.primitives_.push_back(primitive_info);
    }
}

static void setupMeshes(
    const tinygltf::Model& model,
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    drawable_object->meshes_.resize(model.meshes.size());
    for (int i_mesh = 0; i_mesh < model.meshes.size(); i_mesh++) {
        setupMesh(model, model.meshes[i_mesh], drawable_object->meshes_[i_mesh]);
    }
}

std::hash<std::string> str_hash;
static void setupMeshState(
    const std::shared_ptr<renderer::Device>& device,
    const ufbx_abi ufbx_scene* fbx_scene,
    std::shared_ptr<ego::DrawableData>& drawable_object,
    const std::string& asset_key) {

    // allocate texture memory at first.
    std::vector<size_t> texture_hash_table;
    std::unordered_map<size_t, uint32_t> texture_map;
    std::vector<bool> texture_is_srgb;
    drawable_object->textures_.resize(fbx_scene->textures.count);
    texture_is_srgb.resize(fbx_scene->textures.count);

    // create texture name hash table.
    for (size_t i_tex = 0; i_tex < fbx_scene->textures.count; i_tex++) {
        const auto& src_tex = fbx_scene->textures[i_tex];
        auto hash_value = str_hash(src_tex->filename.data);
        texture_map[hash_value] = uint32_t(i_tex);
        texture_is_srgb[i_tex] = false;
    }

    // mark texture as srgb.
    for (size_t i_mat = 0; i_mat < fbx_scene->materials.count; i_mat++) {
        const auto& src_material = fbx_scene->materials[i_mat];

        for (int i = 0; i < src_material->textures.count; i++) {
            auto tex = src_material->textures[i];
            const auto& texture_string =
                std::string(tex.material_prop.data);
            const auto& texture_name =
                std::string(tex.texture->filename.data);

            auto tex_id = texture_map[str_hash(texture_name)];

            if (texture_string == "DiffuseColor" ||
                texture_string == "EmissiveColor") {
                texture_is_srgb[tex_id] = true;
            }
        }

        // ufbx PBR fallback: mark base-color and emissive textures sRGB
        // even when they use non-standard property names (Stingray PBS, etc.)
        const auto* bc_tex = src_material->pbr.base_color.texture;
        if (!bc_tex) bc_tex = src_material->fbx.diffuse_color.texture;
        if (bc_tex) texture_is_srgb[bc_tex->typed_id] = true;

        const auto* em_tex = src_material->pbr.emission_color.texture;
        if (!em_tex) em_tex = src_material->fbx.emission_color.texture;
        if (em_tex) texture_is_srgb[em_tex->typed_id] = true;
    }

    for (size_t i_tex = 0; i_tex < fbx_scene->textures.count; i_tex++) {
        auto& dst_tex = drawable_object->textures_[i_tex];
        const auto& src_tex = fbx_scene->textures[i_tex];

        // Preserve the on-disk filename string for downstream debug /
        // classifier code.  src_tex->filename.data is the absolute or
        // texture-pack-relative path captured by ufbx; we keep the
        // whole thing rather than basenaming so callers can do their
        // own basename extraction if they prefer.
        dst_tex.source_filename_ = src_tex->filename.data
            ? std::string(src_tex->filename.data) : std::string();

        glm::uvec3 size(0);
        auto format = renderer::Format::R8G8B8A8_UNORM;
        helper::createTextureImage(
            device,
            src_tex->filename.data,
            format,
            texture_is_srgb[i_tex],
            dst_tex,
            std::source_location::current(),
            /*cacheable=*/true);
    }

    // Material
    {
        drawable_object->materials_.resize(fbx_scene->materials.count);
        for (size_t i_mat = 0; i_mat < fbx_scene->materials.count; i_mat++) {
            auto& dst_material = drawable_object->materials_[i_mat];
            const auto& src_material = fbx_scene->materials[i_mat];

            // Capture the source-asset material name for gameplay use
            // (footstep sound lookup, surface-friction tables, etc.).
            // ufbx guarantees `name.data` is null-terminated; default
            // to empty when the source had no name set.
            dst_material.name_ = src_material->name.data
                ? std::string(src_material->name.data) : std::string();

            for (int i = 0; i < src_material->textures.count; i++) {
                auto tex = src_material->textures[i];
                const auto& texture_string =
                    std::string(tex.material_prop.data);
                const auto& texture_name =
                    std::string(tex.texture->filename.data);

                auto tex_id = texture_map[str_hash(texture_name)];

                if (texture_string == "DiffuseColor") {
                    dst_material.base_color_idx_ = tex_id;
                }
                else if (texture_string == "SpecularColor") {
                    dst_material.metallic_roughness_idx_ = tex_id;
                }
                else if (texture_string == "NormalMap") {
                    dst_material.normal_idx_ = tex_id;
                }
                else if (texture_string == "MetallicRoughness") {
                    dst_material.metallic_roughness_idx_ = tex_id;
                }
                else if (texture_string == "EmissiveColor") {
                    dst_material.emissive_idx_ = tex_id;
                }
                else if (texture_string == "Occlusion") {
                    dst_material.occlusion_idx_ = tex_id;
                }
                // else: unknown FBX texture property — skip silently.
                // Non-standard shaders (Stingray PBS, Arnold, etc.) use
                // namespaced property names that ufbx normalises into the
                // pbr / fbx struct.  We pick those up in the fallback below.
            }

            // ── ufbx PBR fallback ─────────────────────────────────────────
            // For materials that use non-standard FBX shaders (Stingray PBS,
            // 3dsMax|Parameters|base_color_map, …), the raw texture strings
            // don't match "DiffuseColor".  ufbx auto-maps those into
            // src_material->pbr.  Use that when the loop above found nothing.
            if (dst_material.base_color_idx_ < 0) {
                const auto* tex = src_material->pbr.base_color.texture;
                if (!tex) tex = src_material->fbx.diffuse_color.texture;
                if (tex) {
                    dst_material.base_color_idx_ =
                        static_cast<int32_t>(tex->typed_id);
                }
            }
            if (dst_material.normal_idx_ < 0) {
                const auto* tex = src_material->pbr.normal_map.texture;
                if (!tex) tex = src_material->fbx.normal_map.texture;
                if (tex) {
                    dst_material.normal_idx_ =
                        static_cast<int32_t>(tex->typed_id);
                }
            }
            if (dst_material.metallic_roughness_idx_ < 0) {
                const auto* tex = src_material->pbr.roughness.texture;
                if (tex) {
                    dst_material.metallic_roughness_idx_ =
                        static_cast<int32_t>(tex->typed_id);
                }
            }
            if (dst_material.emissive_idx_ < 0) {
                const auto* tex = src_material->pbr.emission_color.texture;
                if (!tex) tex = src_material->fbx.emission_color.texture;
                if (tex) {
                    dst_material.emissive_idx_ =
                        static_cast<int32_t>(tex->typed_id);
                }
            }
            if (dst_material.occlusion_idx_ < 0) {
                const auto* tex = src_material->pbr.ambient_occlusion.texture;
                if (tex) {
                    dst_material.occlusion_idx_ =
                        static_cast<int32_t>(tex->typed_id);
                }
            }
            // ─────────────────────────────────────────────────────────────

            if (dst_material.base_color_idx_ >= 0) {
                drawable_object->textures_[dst_material.base_color_idx_].linear = false;
            }

            if (dst_material.emissive_idx_ >= 0) {
                drawable_object->textures_[dst_material.emissive_idx_].linear = false;
            }

            // Alpha mask: base.frag has #define ALPHAMODE_MASK 1 always active,
            // meaning ALL FBX materials always discard fragments where alpha < 0.1.
            // Opaque textures have alpha=1.0 so the discard never fires for them.
            // Match this behaviour exactly in the cluster pass.
            dst_material.alpha_cutoff_ = 0.1f;
            dst_material.alpha_mask_ = true;
            dst_material.alpha_mode_ = ego::AlphaMode::Mask;

            // ── FBX glass-name override ────────────────────────────────────
            // Bistro authors many windows / bottles / display cases as
            // separate materials called "Glass*", "Window*", etc. Promote
            // those to true alpha-blend (drawn after opaque + mask, with
            // no depth-write).
            {
                std::string mn(src_material->name.data
                    ? src_material->name.data : "");
                auto contains = [&mn](const char* needle) {
                    return mn.find(needle) != std::string::npos;
                };
                if (contains("glass")  || contains("Glass")  ||
                    contains("window") || contains("Window") ||
                    contains("transparent") || contains("Transparent")) {
                    dst_material.alpha_mode_   = ego::AlphaMode::Blend;
                    dst_material.glass_forced_ = true;
                    dst_material.alpha_mask_   = false;
                }
            }

            device->createBuffer(
                sizeof(glsl::PbrMaterialParams),
                SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
                0,
                dst_material.uniform_buffer_.buffer,
                dst_material.uniform_buffer_.memory,
                std::source_location::current());

            glsl::PbrMaterialParams ubo{};
            const auto& base_factor = src_material->pbr.base_factor;
            ubo.base_color_factor = glm::vec4(0.0f);
            if (base_factor.has_value) {
                ubo.base_color_factor =
                    glm::vec4(float(base_factor.value_vec4.x));
                if (base_factor.value_components > 1) {
                    ubo.base_color_factor.y =
                        float(base_factor.value_vec4.y);
                }
                if (base_factor.value_components > 2) {
                    ubo.base_color_factor.z =
                        float(base_factor.value_vec4.z);
                }
                if (base_factor.value_components > 3) {
                    ubo.base_color_factor.w =
                        float(base_factor.value_vec4.w);
                }
            }

            // FBX glass override: clamp alpha so the material actually
            // reads as translucent.  Bistro bottles / windows often ship
            // with full-opaque base color + alpha=1 textures; without
            // this clamp the alpha-blend pipeline would blend an opaque
            // colour and look identical to opaque rendering.
            if (dst_material.glass_forced_ && ubo.base_color_factor.w >= 0.95f) {
                ubo.base_color_factor.w = 0.4f;
            }

            const auto& emission_factor = src_material->pbr.emission_factor;
            ubo.emissive_factor = glm::vec3(0.0f);
            if (emission_factor.has_value) {
                ubo.emissive_factor =
                    glm::vec4(float(emission_factor.value_vec4.x));
                if (emission_factor.value_components > 1) {
                    ubo.emissive_factor.y =
                        float(emission_factor.value_vec4.y);
                }
                if (emission_factor.value_components > 2) {
                    ubo.emissive_factor.z =
                        float(emission_factor.value_vec4.z);
                }
            }

            const auto& emission_color = src_material->pbr.emission_color;
            ubo.emissive_color = glm::vec3(0.0f);
            if (emission_color.has_value) {
                ubo.emissive_color =
                    glm::vec4(float(emission_color.value_vec4.x));
                if (emission_color.value_components > 1) {
                    ubo.emissive_color.y =
                        float(emission_color.value_vec4.y);
                }
                if (emission_color.value_components > 2) {
                    ubo.emissive_color.z =
                        float(emission_color.value_vec4.z);
                }
            }

            const auto& specular_factor = src_material->fbx.specular_factor;
            ubo.specular_factor = glm::vec3(1.0f);
            if (specular_factor.has_value) {
                ubo.specular_factor =
                    glm::vec4(float(specular_factor.value_vec4.x));
                if (specular_factor.value_components > 1) {
                    ubo.specular_factor.y =
                        float(specular_factor.value_vec4.y);
                }
                if (specular_factor.value_components > 2) {
                    ubo.specular_factor.z =
                        float(specular_factor.value_vec4.z);
                }
            }

            const auto& specular_color = src_material->fbx.specular_color;
            ubo.specular_color = glm::vec3(1.0f);
            if (specular_color.has_value) {
                ubo.specular_color =
                    glm::vec4(float(specular_color.value_vec4.x));
                if (specular_color.value_components > 1) {
                    ubo.specular_color.y =
                        float(specular_color.value_vec4.y);
                }
                if (specular_color.value_components > 2) {
                    ubo.specular_color.z =
                        float(specular_color.value_vec4.z);
                }
            }

            const auto& specular_exponent = src_material->fbx.specular_exponent;

            ubo.glossiness_factor =
                src_material->pbr.glossiness.has_value ?
                float(src_material->pbr.glossiness.value_real) : 
                1.0f;

            ubo.metallic_roughness_specular_factor =
                src_material->pbr.specular_factor.has_value ?
                float(src_material->pbr.specular_factor.value_real) :
                1.0f;

            ubo.metallic_factor =
                src_material->pbr.metalness.has_value ?
                float(src_material->pbr.metalness.value_real) :
                0.0f;

            ubo.roughness_factor =
                src_material->pbr.roughness.has_value ?
                float(src_material->pbr.roughness.value_real) :
                1.0f;

            ubo.metallic_roughness_specular_factor =
                src_material->pbr.specular_factor.has_value ?
                float(src_material->pbr.specular_factor.value_real) :
                1.0f;

            ubo.alpha_cutoff = 0.1f;
            ubo.mip_count = 11;
            ubo.normal_scale = 1.0f;
            ubo.uv_set_flags = glm::vec4(0, 0, 0, 0);
            ubo.exposure = 1.0f;
            ubo.occlusion_strength = 1.0f;
            ubo.tonemap_type = TONEMAP_DEFAULT;

            ubo.material_features = (dst_material.metallic_roughness_idx_ >= 0 ? (FEATURE_HAS_METALLIC_ROUGHNESS_MAP | FEATURE_HAS_METALLIC_CHANNEL) : 0) | FEATURE_MATERIAL_METALLICROUGHNESS;
            ubo.material_features |= (dst_material.base_color_idx_ >= 0 ? FEATURE_HAS_BASE_COLOR_MAP : 0);
            ubo.material_features |= (dst_material.emissive_idx_ >= 0 ? FEATURE_HAS_EMISSIVE_MAP : 0);
            ubo.material_features |= (dst_material.occlusion_idx_ >= 0 ? FEATURE_HAS_OCCLUSION_MAP : 0);
            ubo.material_features |= (dst_material.normal_idx_ >= 0 ? FEATURE_HAS_NORMAL_MAP : 0);
            ubo.material_features |= (dst_material.specular_color_idx_ >= 0 ? FEATURE_MATERIAL_SPECULARGLOSSINESS : 0);
            // FBX path mirror of the GLTF block above — feed the CPU-side
            // AlphaMode (including FBX glass-by-name overrides at line ~895)
            // into the shader so the "Translucent" debug mode can spot
            // them.  Glass-tagged materials always end up as Blend here.
            ubo.material_features |= (dst_material.alpha_mode_ == ego::AlphaMode::Blend ? FEATURE_MATERIAL_BLEND : 0);
            ubo.material_features |= (dst_material.alpha_mode_ == ego::AlphaMode::Mask  ? FEATURE_MATERIAL_ALPHA_MASK : 0);

            device->updateBufferMemory(dst_material.uniform_buffer_.memory, sizeof(ubo), &ubo);

            // ECS dedup identity — mirrors exactly what was uploaded above.
            captureMaterialDesc(
                dst_material, ubo, drawable_object->textures_, asset_key);
        }
    }
}

static void setupMesh(
    const std::shared_ptr<renderer::Device>& device,
    const ufbx_abi ufbx_scene* fbx_scene,
    std::shared_ptr<ego::DrawableData>& drawable_object,
    const uint32_t& mesh_idx,
    std::ostringstream& log_buf) {

    auto vertex_buffer_idx = mesh_idx * 2;
    auto indice_buffer_idx = mesh_idx * 2 + 1;

    const ufbx_mesh* src_mesh = fbx_scene->meshes[mesh_idx];
    auto& drawable_mesh = drawable_object->meshes_[mesh_idx];
    auto& vertex_buffer = drawable_object->buffers_[vertex_buffer_idx];
    auto& indice_buffer = drawable_object->buffers_[indice_buffer_idx];

    // ufbx is loaded WITHOUT triangulation (opts = {0}), so faces can be
    // n-gons and num_faces != num_triangles in general.  The face loop
    // below fan-triangulates each face, so the old all-triangle assert is
    // gone -- it fired in debug and, worse, was compiled out in release
    // where the previous raw-push silently corrupted the index list for
    // any quad / n-gon (the missing triangles in the collision overlay).
    assert(src_mesh->num_indices == src_mesh->vertex_position.indices.count);
    assert(src_mesh->num_indices == src_mesh->vertex_normal.indices.count);
    assert(src_mesh->num_indices == src_mesh->vertex_uv.indices.count);

    std::vector<glm::vec3> vertex_position(src_mesh->num_vertices);
    for (auto i = 0; i < src_mesh->num_vertices; i++) {
        vertex_position[i].x = float_t(src_mesh->vertex_position[i].x);
        vertex_position[i].y = float_t(src_mesh->vertex_position[i].y);
        vertex_position[i].z = float_t(src_mesh->vertex_position[i].z);
    }

    std::unordered_map<size_t, VertexHashInfo> vertex_map;
    std::vector<uint32_t> new_indices;
    std::vector<uint32_t> indice_match_table;
    new_indices.reserve(src_mesh->num_indices * 3);
    indice_match_table.reserve(src_mesh->num_indices * 3);

    helper::VertexStruct vertex_data;
    // (Removed: assert(num_faces * 3 == num_indices) -- only holds for an
    // all-triangle mesh; n-gons are fan-triangulated in the loop below.)

    size_t num_traingles = 0;
    size_t num_indices = 0;
    bool has_bvh_trees = false;
    for (int i_part = 0; i_part < src_mesh->material_parts.count; i_part++) {
        auto part = src_mesh->material_parts[i_part];
        auto mat = src_mesh->materials[part.index];

        bool create_bvh_tree = false;
        std::string mat_name_string(mat->name.data);
        if (mat_name_string.find("light") == std::string::npos &&
            mat_name_string.find("Light") == std::string::npos &&
            mat_name_string.find("BLENDSHADER") == std::string::npos &&
            mat_name_string.find("MASTER") == std::string::npos &&
            mat_name_string.find("Emissive") == std::string::npos &&
            mat_name_string.find("Banner") == std::string::npos &&
            mat_name_string.find("Sign") == std::string::npos &&
            mat_name_string.find("Pivot") == std::string::npos &&
            mat_name_string.find("Antenna") == std::string::npos &&
            mat_name_string.find("sign") == std::string::npos &&
            mat_name_string.find("NapkinHolder") == std::string::npos &&
            mat_name_string.find("Bottle") == std::string::npos &&
            mat_name_string.find("Flowers") == std::string::npos &&
            mat_name_string.find("ElectricBox") == std::string::npos &&
            mat_name_string.find("Fabric") == std::string::npos &&
            mat_name_string.find("Beams") == std::string::npos &&
            mat_name_string.find("Ashtray") == std::string::npos &&
            mat_name_string.find("Leaves") == std::string::npos &&
            mat_name_string.find("leaf") == std::string::npos &&
            mat_name_string.find("Foliage") == std::string::npos) {
            create_bvh_tree = true;
            has_bvh_trees = true;
        }

        // ── DEBUG: verify the RAW ufbx source for Pavement_Brick parts.
        // Mesh 46 (Pavement_Brick/Street_6185) came out as TWO disconnected
        // strips with a ~3m gap.  Compute the connected-component count of
        // each raw ufbx Pavement_Brick part (union-find over shared POSITION
        // indices -- transform-independent).  If a large part is ONE
        // connected piece, the source is continuous and our pipeline split
        // it (a real bug); if it is already two pieces, the source genuinely
        // has two strips.  Large parts (>200 faces) are also dumped raw to a
        // numbered OBJ for off-line inspection.  Remove when done.
        if (mat_name_string.find("Pavement_Brick") != std::string::npos &&
            part.num_faces > 50) {
            std::unordered_map<uint32_t,uint32_t> parent;
            auto find = [&](uint32_t x) -> uint32_t {
                while (parent[x]!=x){ parent[x]=parent[parent[x]]; x=parent[x]; }
                return x; };
            auto uni = [&](uint32_t a, uint32_t b){
                if (!parent.count(a)) parent[a]=a;
                if (!parent.count(b)) parent[b]=b;
                parent[find(a)] = find(b);
            };
            double mnz=1e30,mxz=-1e30,mnx=1e30,mxx=-1e30;
            for (int fi = 0; fi < part.num_faces; ++fi) {
                const auto& fc = src_mesh->faces[part.face_indices[fi]];
                if (fc.num_indices < 1) continue;
                uint32_t p0 = src_mesh->vertex_position.indices[fc.index_begin];
                for (uint32_t k = 0; k < fc.num_indices; ++k) {
                    uint32_t pidx = src_mesh->vertex_position.indices[fc.index_begin + k];
                    uni(p0, pidx);
                    const auto& pp = src_mesh->vertex_position.values[pidx];
                    mnz=std::min(mnz,(double)pp.z); mxz=std::max(mxz,(double)pp.z);
                    mnx=std::min(mnx,(double)pp.x); mxx=std::max(mxx,(double)pp.x);
                }
            }
            int components = 0;
            for (auto& kv : parent) if (find(kv.first) == kv.first) ++components;
            std::cout << "[mesh.fbx.raw] Pavement part faces=" << part.num_faces
                      << " positions=" << parent.size()
                      << " components=" << components
                      << " local_bbox x[" << mnx << "," << mxx << "] z["
                      << mnz << "," << mxz << "]"
                      << (components > 1 ? "  <== SOURCE ALREADY SPLIT" :
                                           "  (source connected)")
                      << std::endl;
            if (part.num_faces > 200) {
                static int s_pav = 0;
                std::ofstream raw(
                    "G:/work/procedure-world-sim/debug_source_raw_" +
                    std::to_string(s_pav++) + ".obj");
                if (raw.is_open()) {
                    raw << "# RAW ufbx, mat=" << mat_name_string
                        << " faces=" << part.num_faces
                        << " components=" << components << "\n";
                    std::unordered_map<uint32_t,uint32_t> pos_to_obj;
                    uint32_t next_obj = 1;
                    for (int fi = 0; fi < part.num_faces; ++fi) {
                        const auto& fc = src_mesh->faces[part.face_indices[fi]];
                        for (uint32_t k = 0; k < fc.num_indices; ++k) {
                            uint32_t pidx = src_mesh->vertex_position.indices[fc.index_begin + k];
                            if (!pos_to_obj.count(pidx)) {
                                const auto& p = src_mesh->vertex_position.values[pidx];
                                raw << "v " << p.x << " " << p.y << " " << p.z << "\n";
                                pos_to_obj[pidx] = next_obj++;
                            }
                        }
                    }
                    for (int fi = 0; fi < part.num_faces; ++fi) {
                        const auto& fc = src_mesh->faces[part.face_indices[fi]];
                        raw << "f";
                        for (uint32_t k = 0; k < fc.num_indices; ++k)
                            raw << " " << pos_to_obj[src_mesh->vertex_position.indices[fc.index_begin + k]];
                        raw << "\n";
                    }
                }
            }
        }

        ego::PrimitiveInfo primitive_info;
        primitive_info.bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
        // ::lowest() (most negative finite float), NOT ::min() (smallest
        // positive normalized).  See bbox_max_ comment in drawable_object.h.
        primitive_info.bbox_max_ = glm::vec3(std::numeric_limits<float>::lowest());

        // Range of indices contributed by this primitive within
        // `new_indices`.  Captured here (before the face loop) and used
        // after the loop to slice out a per-primitive collision index
        // list -- only for primitives that pass the BVH (collision)
        // gate above, so foliage / lights / banners do not contribute.
        const size_t part_index_start = new_indices.size();

        // ufbx is loaded WITHOUT triangulation (opts = {0}), so a face may
        // be an n-gon (quad / polygon) with face.num_indices > 3.  We cache
        // every corner, THEN fan-triangulate from corner 0:
        //   (0,1,2), (0,2,3), ..., (0, n-2, n-1)   -- n-2 triangles,
        // INCLUDING the closing (0, n-2, n-1).  The previous code pushed
        // all corners RAW and only asserted num_indices == 3; in release
        // (asserts compiled out) every n-gon's extra corners straddled the
        // group-of-3 boundary and corrupted / dropped triangles -- the
        // source of the missing triangles in the collision overlay.  For a
        // genuine triangle (num_indices == 3) this emits exactly one
        // triangle with the same three indices, i.e. byte-identical to the
        // old behaviour.
        std::vector<uint32_t> face_verts;
        for (int i_face = 0; i_face < part.num_faces; i_face++) {
            const auto& face_indice = part.face_indices[i_face];
            const auto& face = src_mesh->faces[face_indice];

            // Diagnostic: confirm whether this asset actually contains
            // n-gons (i.e. whether the missing-triangle bug was firing).
            // Throttled to the first 20 occurrences.
            if (face.num_indices != 3) {
                static int s_ngon_log = 0;
                if (s_ngon_log < 20) {
                    ++s_ngon_log;
                    std::cout << "[mesh.fbx] non-triangle face: num_indices="
                              << face.num_indices
                              << " -> fan-triangulated into "
                              << (face.num_indices >= 2
                                      ? face.num_indices - 2 : 0)
                              << " tri(s) (#" << s_ngon_log << "/20)"
                              << std::endl;
                }
            }

            face_verts.clear();
            for (uint32_t i_vert = 0; i_vert < face.num_indices; i_vert++) {
                const auto src_vert_index = face.index_begin + i_vert;
                auto pos_indice = src_mesh->vertex_position.indices[src_vert_index];
                const auto& position = src_mesh->vertex_position.values[pos_indice];
                auto position_packed = glm::vec3(position.x, position.y, position.z);

                primitive_info.bbox_min_ = glm::min(primitive_info.bbox_min_, position_packed);
                primitive_info.bbox_max_ = glm::max(primitive_info.bbox_max_, position_packed);

                vertex_data.position =
                    glm::vec3(
                        float(position.x),
                        float(position.y),
                        float(position.z));
                auto norm_indice = src_mesh->vertex_normal.indices[src_vert_index];
                const auto& normal = src_mesh->vertex_normal.values[norm_indice];
                vertex_data.normal =
                    glm::vec3(
                        float(normal.x),
                        float(normal.y),
                        float(normal.z));
                auto uv_indice = src_mesh->vertex_uv.indices[src_vert_index];
                const auto& uv = src_mesh->vertex_uv.values[uv_indice];
                vertex_data.uv =
                    glm::vec2(
                        float(uv.x),
                        float(uv.y));

                auto new_indice =
                    cacheVertexIndice(
                        vertex_map,
                        indice_match_table,
                        vertex_data,
                        src_vert_index);

                face_verts.push_back(new_indice);
            }

            // Fan-triangulate the cached corners.  Skips degenerate faces
            // (< 3 corners) automatically since the loop body never runs.
            for (size_t t = 1; t + 1 < face_verts.size(); ++t) {
                new_indices.push_back(face_verts[0]);
                new_indices.push_back(face_verts[t]);
                new_indices.push_back(face_verts[t + 1]);
            }
        }

        // Capture this primitive's index range as a CPU-side companion
        // array for the collision-mesh builder (CollisionMesh::
        // buildFromDrawable iterates per-primitive `vertex_indices_`
        // and skips primitives whose pointer is null).
        //
        // Capture for EVERY primitive with triangles, regardless of the
        // create_bvh_tree material-name gate above.  That gate was
        // tuned for a future BVH-collision path (excludes lights,
        // foliage, banners, ...) but its substring rules overreach
        // for visualisation -- "MASTER", "Fabric", "Beams" match the
        // Bistro's main Master_Paint_*, Fabric_Wall_*, Wood_Beams_*
        // floor / wall / structure materials, hiding the bulk of the
        // building.  For the collision-debug viz we want to see the
        // whole world; the create_bvh_tree flag is preserved on
        // has_bvh_trees so the BVH path can re-filter at build time
        // when it's re-enabled.
        const size_t part_index_end = new_indices.size();
        if (part_index_end > part_index_start) {
            auto part_idx_list = std::make_shared<std::vector<int32_t>>();
            part_idx_list->reserve(part_index_end - part_index_start);
            for (size_t k = part_index_start; k < part_index_end; ++k) {
                part_idx_list->push_back(static_cast<int32_t>(new_indices[k]));
            }
            primitive_info.vertex_indices_ = std::move(part_idx_list);
        }

#if 0
        if (create_bvh_tree) {
            bool debug_mode = false;

            std::cout << "mesh idx : " << mesh_idx
                << "/" << fbx_scene->meshes.count;

#if DEBUG_OUTPUT
            std::cout << 
                ", mat part : " << i_part <<
                ", num tris : " << vertex_indices.size() / 3 <<
                std::endl;
#else
            std::cout << std::endl;
#endif
            primitive_info.vertex_indices_ =
                std::make_shared<std::vector<int32_t>>(vertex_indices);

            std::shared_ptr<helper::BVHBuilder> builder =
                std::make_shared<helper::BVHBuilder>(
                    vertex_position,
                    vertex_indices,
                    debug_mode);

            //builder->build();

            primitive_info.bvh_root_ = builder->getRoot();
        }
#endif

        drawable_mesh.bbox_min_ =
            min(drawable_mesh.bbox_min_, primitive_info.bbox_min_);
        drawable_mesh.bbox_max_ =
            max(drawable_mesh.bbox_max_, primitive_info.bbox_max_);
        drawable_mesh.primitives_.push_back(primitive_info);
    }

    helper::Mesh full_lod_meshes;
    full_lod_meshes.vertex_data_ptr->reserve(indice_match_table.size() * 3);
    full_lod_meshes.faces_ptr->reserve(src_mesh->num_indices);

    for (uint32_t i_face = 0; i_face < new_indices.size() / 3; i_face++) {
        full_lod_meshes.faces_ptr->push_back(
            helper::Face(
                new_indices[i_face * 3],
                new_indices[i_face * 3 + 1],
                new_indices[i_face * 3 + 2]));
    }

    auto& drawable_vertices = full_lod_meshes.vertex_data_ptr;
    for (int i_vert = 0; i_vert < indice_match_table.size(); i_vert++) {
        const uint32_t& src_vert_idx = indice_match_table[i_vert];
        drawable_vertices->emplace_back();
        auto& vertex = drawable_vertices->back();
        auto position_idx = src_mesh->vertex_position.indices[src_vert_idx];
        const auto& position = src_mesh->vertex_position.values[position_idx];
        vertex.position = glm::vec3(position.x, position.y, position.z);
        auto normal_idx = src_mesh->vertex_normal.indices[src_vert_idx];
        const auto& normal = src_mesh->vertex_normal.values[normal_idx];
        vertex.normal = glm::vec3(normal.x, normal.y, normal.z);
        auto uv_idx = src_mesh->vertex_uv.indices[src_vert_idx];
        const auto& uv = src_mesh->vertex_uv.values[uv_idx];
        vertex.uv = glm::vec2(uv.x, uv.y);
    }

    // ── Cluster sidecar build (always-on for Nanite-lite GPU culling) ──
    // `full_lod_meshes` at this point holds exactly LOD-0 of the mesh —
    // new_indices.size()/3 faces and drawable_vertices populated above.
    // The HLOD loop below will APPEND higher-LOD faces onto the same arrays;
    // we want clusters for the base LOD only, so build them here first.
    helper::buildClusterMesh(full_lod_meshes, drawable_mesh.cluster_mesh_);
    // Only build the expanded per-triangle debug vertex buffer when the
    // cluster-debug visualisation is actually active.  Building it for every
    // FBX mesh unconditionally wastes hundreds of MB of GPU memory on large
    // scenes (Bistro: ~500 meshes × many triangles × 48 bytes/vertex).
    // The cluster indirect renderer (GPU culling path) does NOT need this
    // buffer — it uses cluster_global_mesh_idx_ >= 0 to identify its meshes.
    if (engine::helper::clusterRenderingEnabled()) {
        ego::ClusterDebugDraw::uploadForMesh(
            device,
            drawable_mesh.cluster_mesh_,
            drawable_mesh.cluster_debug_gpu_);
    }

    // ── Build cluster_prim_map_ ─────────────────────────────────────────────
    // Each cluster in the mesh-level cluster_mesh_ is assigned to the primitive
    // whose contiguous face range contains the cluster's first face index.
    // This keeps ONE cluster mesh per FBX mesh (efficient packing / low cluster
    // count) while still giving uploadMeshClusters the per-cluster material info
    // it needs — avoiding the 3-10× cluster explosion of per-primitive building.
    {
        // prim_face_start[i] = first global face index owned by primitive i.
        // prim_face_start[num_parts] = total face count (sentinel).
        std::vector<uint32_t> prim_face_start;
        prim_face_start.reserve(
            static_cast<size_t>(src_mesh->material_parts.count) + 1);
        uint32_t face_offset = 0;
        for (int i_part = 0; i_part < src_mesh->material_parts.count; i_part++) {
            prim_face_start.push_back(face_offset);
            face_offset += static_cast<uint32_t>(
                src_mesh->material_parts[i_part].num_faces);
        }
        prim_face_start.push_back(face_offset);  // sentinel

        drawable_mesh.cluster_prim_map_.clear();
        drawable_mesh.cluster_prim_map_.reserve(
            drawable_mesh.cluster_mesh_.clusters.size());
        for (const auto& cl : drawable_mesh.cluster_mesh_.clusters) {
            uint32_t prim_idx = 0;
            if (!cl.face_indices.empty()) {
                const uint32_t first_face = cl.face_indices[0];
                // Linear scan — number of material parts per mesh is small.
                for (uint32_t p = 0; p + 1 < prim_face_start.size(); ++p) {
                    if (first_face >= prim_face_start[p] &&
                        first_face <  prim_face_start[p + 1]) {
                        prim_idx = p;
                        break;
                    }
                }
            }
            drawable_mesh.cluster_prim_map_.push_back(prim_idx);
        }
    }

    // create HLOD
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>
        lod_indice_info(helper::c_num_lods + 1);
    {
#if DEBUG_OUTPUT
        log_buf << "====================" << std::endl;
        log_buf << "mesh idx : " << mesh_idx
            << "/" << fbx_scene->meshes.count
            << ", num tris : " << new_indices.size() / 3
            << std::endl;
#endif
        uint32_t face_idx_offset = 0;
        lod_indice_info[0].resize(src_mesh->material_parts.count);
        for (int i_part = 0; i_part < src_mesh->material_parts.count; i_part++) {
            auto part = src_mesh->material_parts[i_part];
            lod_indice_info[0][i_part].first = face_idx_offset;
            lod_indice_info[0][i_part].second = uint32_t(part.num_faces);
            face_idx_offset += uint32_t(part.num_faces);
        }
        for (int i_lod = 0; i_lod < helper::c_num_lods; i_lod++) {
            const auto& src_lod_info = lod_indice_info[i_lod];
            auto& dst_lod_info       = lod_indice_info[i_lod + 1];
            dst_lod_info.resize(src_mesh->material_parts.count);

            // Pack all parts of this LOD level into one local mesh so the QEM
            // decimater can see and preserve inter-part seam vertices, which
            // prevents T-junctions and stretched geometry at part boundaries.
            helper::Mesh combined;
            combined.vertex_data_ptr = std::make_shared<std::vector<helper::VertexStruct>>();
            combined.faces_ptr       = std::make_shared<std::vector<helper::Face>>();
            std::vector<int32_t> combined_part_ids;
            size_t total_src_faces = 0;

            // Map global drawable_vertices index → local packed index
            std::unordered_map<uint32_t, uint32_t> g2l;
            g2l.reserve(512);

            for (int i_part = 0; i_part < src_mesh->material_parts.count; i_part++) {
                int32_t face_start = src_lod_info[i_part].first;
                int32_t face_count = src_lod_info[i_part].second;
                total_src_faces += face_count;
                for (int32_t fi = 0; fi < face_count; fi++) {
                    const auto& sf = full_lod_meshes.faces_ptr->at(face_start + fi);
                    helper::Face lf;
                    for (int k = 0; k < 3; k++) {
                        uint32_t gi = sf.v_indices[k];
                        auto [it, ins] = g2l.emplace(gi, uint32_t(combined.vertex_data_ptr->size()));
                        if (ins)
                            combined.vertex_data_ptr->push_back((*drawable_vertices)[gi]);
                        lf.v_indices[k] = it->second;
                    }
                    combined.faces_ptr->push_back(lf);
                    combined_part_ids.push_back(i_part);
                }
            }

            size_t target_total = std::max(
                size_t(double(total_src_faces) * helper::c_target_lod_ratio), size_t(1));

            // Decimate the whole mesh with QEM
            helper::Mesh decimated;
            std::vector<int32_t> dec_part_ids;
            helper::decimateMesh(
                combined, combined_part_ids,
                decimated, dec_part_ids,
                target_total, log_buf);

#if DEBUG_OUTPUT
            log_buf << "  lod " << i_lod + 1 << ": "
                    << total_src_faces << " -> "
                    << decimated.getFaceCount() << " faces\n";
#endif

            // Append decimated vertices to the global vertex array
            uint32_t base_v = uint32_t(drawable_vertices->size());
            for (const auto& v : *decimated.vertex_data_ptr)
                drawable_vertices->push_back(v);

            // Bucket output faces by part, emit in part order
            std::vector<std::vector<int>> part_face_idx(src_mesh->material_parts.count);
            for (int fi = 0; fi < (int)decimated.faces_ptr->size(); fi++)
                part_face_idx[dec_part_ids[fi]].push_back(fi);

            for (int i_part = 0; i_part < src_mesh->material_parts.count; i_part++) {
                dst_lod_info[i_part].first  = face_idx_offset;
                dst_lod_info[i_part].second = uint32_t(part_face_idx[i_part].size());
                for (int fi : part_face_idx[i_part]) {
                    const auto& f = decimated.faces_ptr->at(fi);
                    full_lod_meshes.faces_ptr->push_back(helper::Face(
                        base_v + f.v_indices[0],
                        base_v + f.v_indices[1],
                        base_v + f.v_indices[2]));
                }
                face_idx_offset += dst_lod_info[i_part].second;
            }
        }
    }

    for (uint32_t i_face = uint32_t(new_indices.size()) / 3;
        i_face < uint32_t(full_lod_meshes.faces_ptr->size());
        i_face++) {
        const auto& face = full_lod_meshes.faces_ptr->at(i_face);
        new_indices.push_back(face.v_indices[0]);
        new_indices.push_back(face.v_indices[1]);
        new_indices.push_back(face.v_indices[2]);
    }

    renderer::Helper::createBuffer(
        device,
        SET_4_FLAG_BITS(
            BufferUsage,
            VERTEX_BUFFER_BIT,
            STORAGE_BUFFER_BIT,
            SHADER_DEVICE_ADDRESS_BIT,
            ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        vertex_buffer.buffer,
        vertex_buffer.memory,
        std::source_location::current(),
        drawable_vertices->size() * sizeof(helper::VertexStruct),
        drawable_vertices->data());

    auto total_num_indices = new_indices.size();
    auto use_16bits_index = total_num_indices < 65536;
    auto index_bytes_count = 2;
    auto index_type = renderer::IndexType::UINT16;

    if (use_16bits_index) {
        std::vector<uint16_t> indices_16;
        indices_16.resize(total_num_indices);
        for (int i_idx = 0; i_idx < total_num_indices; i_idx++) {
            indices_16[i_idx] = static_cast<uint16_t>(new_indices[i_idx]);
        }

        renderer::Helper::createBuffer(
            device,
            SET_4_FLAG_BITS(
                BufferUsage,
                INDEX_BUFFER_BIT,
                STORAGE_BUFFER_BIT,
                SHADER_DEVICE_ADDRESS_BIT,
                ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
            indice_buffer.buffer,
            indice_buffer.memory,
            std::source_location::current(),
            indices_16.size() * 2,
            indices_16.data());
    }
    else {
        renderer::Helper::createBuffer(
            device,
            SET_4_FLAG_BITS(
                BufferUsage,
                INDEX_BUFFER_BIT,
                STORAGE_BUFFER_BIT,
                SHADER_DEVICE_ADDRESS_BIT,
                ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
            indice_buffer.buffer,
            indice_buffer.memory,
            std::source_location::current(),
            new_indices.size() * 4,
            new_indices.data());

        index_bytes_count = 4;
        index_type = renderer::IndexType::UINT32;
    }

    // CPU-side position companion array for the collision-mesh
    // builder. Must use the *deduped* vertices that `new_indices`
    // (and therefore the per-primitive `vertex_indices_` captured
    // above) reference -- not the raw FBX `vertex_position` table.
    // `drawable_vertices` is the same buffer about to be uploaded
    // to the GPU vertex buffer, so positions are guaranteed to
    // match the GPU index buffer 1:1.
    //
    // Captured unconditionally (no longer gated on has_bvh_trees)
    // so the collision-debug viz can show every mesh, including
    // those whose every material part missed the BVH gate. The
    // memory cost is one CPU-side glm::vec3 per deduped vertex,
    // bounded by the size of the GPU vertex buffer we're about to
    // upload anyway.
    //
    // LOD-N (N>=1) vertices are appended to drawable_vertices later
    // in this function, but LOD0 indices reference only the front
    // portion of the array, so storing the full array is safe --
    // the LOD tail is just unreferenced extra memory.
    if (drawable_vertices && !drawable_vertices->empty()) {
        auto positions = std::make_shared<std::vector<glm::vec3>>();
        positions->reserve(drawable_vertices->size());
        for (const auto& v : *drawable_vertices) {
            positions->push_back(v.position);
        }
        drawable_mesh.vertex_position_ = std::move(positions);
    }

    num_traingles = 0;
    uint32_t buffer_offset = 0;
    auto pos_view_idx = mesh_idx * 4;
    drawable_object->buffer_views_[pos_view_idx].buffer_idx = vertex_buffer_idx;
    drawable_object->buffer_views_[pos_view_idx].offset = 0;
    drawable_object->buffer_views_[pos_view_idx].range = drawable_vertices->size() * sizeof(helper::VertexStruct);
    drawable_object->buffer_views_[pos_view_idx].stride = sizeof(helper::VertexStruct);

    auto normal_view_idx = mesh_idx * 4 + 1;
    drawable_object->buffer_views_[normal_view_idx].buffer_idx = vertex_buffer_idx;
    drawable_object->buffer_views_[normal_view_idx].offset = sizeof(glm::vec3);
    drawable_object->buffer_views_[normal_view_idx].range = drawable_vertices->size() * sizeof(helper::VertexStruct);
    drawable_object->buffer_views_[normal_view_idx].stride = sizeof(helper::VertexStruct);

    auto uv_view_idx = mesh_idx * 4 + 2;
    drawable_object->buffer_views_[uv_view_idx].buffer_idx = vertex_buffer_idx;
    drawable_object->buffer_views_[uv_view_idx].offset = sizeof(glm::vec3) * 2;
    drawable_object->buffer_views_[uv_view_idx].range = drawable_vertices->size() * sizeof(helper::VertexStruct);
    drawable_object->buffer_views_[uv_view_idx].stride = sizeof(helper::VertexStruct);

    auto indice_view_idx = mesh_idx * 4 + 3;
    drawable_object->buffer_views_[indice_view_idx].buffer_idx = indice_buffer_idx;
    drawable_object->buffer_views_[indice_view_idx].offset = 0;
    drawable_object->buffer_views_[indice_view_idx].range = src_mesh->num_indices * index_bytes_count;
    drawable_object->buffer_views_[indice_view_idx].stride = index_bytes_count;

    for (int i_part = 0; i_part < src_mesh->material_parts.count; i_part++) {
        uint32_t dst_binding = 0;

        auto part = src_mesh->material_parts[i_part];
        auto mat = src_mesh->materials[part.index];

        auto& primitive_info = drawable_mesh.primitives_[i_part];
        primitive_info.tag_.restart_enable = false;
        primitive_info.material_idx_ = mat->typed_id;

        auto mode = renderer::PrimitiveTopology::TRIANGLE_LIST;
        primitive_info.tag_.topology = static_cast<uint32_t>(mode);
        primitive_info.tag_.has_texcoord_0 = true;
        primitive_info.tag_.has_normal = true;
        primitive_info.tag_.double_sided = mat->features.double_sided.enabled;

        std::string name_string(mat->name.data);

        size_t found_pos = name_string.find("DoubleSided");
        if (found_pos != std::string::npos) {
            primitive_info.tag_.double_sided = true;
        }

        // position
        engine::renderer::VertexInputBindingDescription binding = {};
        binding.binding = dst_binding;
        binding.stride = sizeof(helper::VertexStruct);
        binding.input_rate = renderer::VertexInputRate::VERTEX;
        primitive_info.binding_descs_.push_back(binding);

        engine::renderer::VertexInputAttributeDescription attribute = {};
        attribute.buffer_view = pos_view_idx;
        attribute.binding = dst_binding;
        attribute.offset = 0;
        attribute.buffer_offset = 0;
        attribute.location = VINPUT_POSITION;
        attribute.format = engine::renderer::Format::R32G32B32_SFLOAT;

        primitive_info.attribute_descs_.push_back(attribute);
        dst_binding++;

        // normal
        binding.binding = dst_binding;
        binding.stride = sizeof(helper::VertexStruct);
        binding.input_rate = renderer::VertexInputRate::VERTEX;
        primitive_info.binding_descs_.push_back(binding);

        attribute.buffer_view = normal_view_idx;
        attribute.binding = dst_binding;
        attribute.offset = 0;
        attribute.buffer_offset = sizeof(glm::vec3);
        attribute.location = VINPUT_NORMAL;
        attribute.format = engine::renderer::Format::R32G32B32_SFLOAT;
        primitive_info.attribute_descs_.push_back(attribute);
        dst_binding++;

        // uv
        binding.binding = dst_binding;
        binding.stride = sizeof(helper::VertexStruct);
        binding.input_rate = renderer::VertexInputRate::VERTEX;
        primitive_info.binding_descs_.push_back(binding);

        attribute.buffer_view = uv_view_idx;
        attribute.binding = dst_binding;
        attribute.offset = 0;
        attribute.buffer_offset = 2 * sizeof(glm::vec3);
        attribute.location = VINPUT_TEXCOORD0;
        attribute.format = engine::renderer::Format::R32G32_SFLOAT;
        primitive_info.attribute_descs_.push_back(attribute);
        dst_binding++;

        // indices
        primitive_info.index_desc_.resize(helper::c_num_lods + 1);
        for (int32_t i_lod = 0; i_lod < helper::c_num_lods + 1; i_lod++) {
            primitive_info.index_desc_[i_lod].buffer_view = indice_view_idx;
            primitive_info.index_desc_[i_lod].offset =
                lod_indice_info[i_lod][i_part].first * 3 * index_bytes_count;
            primitive_info.index_desc_[i_lod].index_type = index_type;
            primitive_info.index_desc_[i_lod].index_count =
                lod_indice_info[i_lod][i_part].second * 3;
        }

        primitive_info.generateHash();
        num_traingles += part.num_faces;
    }
}

static void setupMeshes(
    const std::shared_ptr<renderer::Device>& device,
    const ufbx_abi ufbx_scene* fbx_scene,
    std::shared_ptr<ego::DrawableData>& drawable_object,
    std::ostringstream& log_buf) {
    drawable_object->meshes_.resize(fbx_scene->meshes.count);
    drawable_object->buffers_.resize(fbx_scene->meshes.count * 2);
    drawable_object->buffer_views_.resize(fbx_scene->meshes.count * 4);
    for (int i_mesh = 0; i_mesh < fbx_scene->meshes.count; i_mesh++) {
        setupMesh(device, fbx_scene, drawable_object, i_mesh, log_buf);
    }
}

static void setupAnimation(
    const tinygltf::Model& model,
    const tinygltf::Animation& src_anim,
    ego::AnimationInfo& anim_info) {

    // setup animation
    for (const auto& src_channel : src_anim.channels) {
        auto channel_info = std::make_shared<ego::AnimChannelInfo>();
        channel_info->node_idx_ = src_channel.target_node;

        const auto& src_sample = src_anim.samplers[src_channel.sampler];
        const auto& input_accessor = model.accessors[src_sample.input];
        const auto& output_accessor = model.accessors[src_sample.output];

        assert(output_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
        if (src_channel.target_path == "rotation") {
            channel_info->type_ = ego::AnimChannelInfo::kRotation;
            channel_info->data_buffer_.format = engine::renderer::Format::R32G32B32A32_SFLOAT;
            assert(output_accessor.type == TINYGLTF_TYPE_VEC4);
        }
        else if (src_channel.target_path == "translation") {
            channel_info->type_ = ego::AnimChannelInfo::kTranslation;
            channel_info->data_buffer_.format = engine::renderer::Format::R32G32B32_SFLOAT;
            assert(output_accessor.type == TINYGLTF_TYPE_VEC3);
        }
        else if (src_channel.target_path == "scale") {
            channel_info->type_ = ego::AnimChannelInfo::kScale;
            channel_info->data_buffer_.format = engine::renderer::Format::R32G32B32_SFLOAT;
            assert(output_accessor.type == TINYGLTF_TYPE_VEC3);
        }
        else {
            assert(0);
        }

        assert(input_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
        assert(input_accessor.type == TINYGLTF_TYPE_SCALAR);

        channel_info->sample_buffer_.buffer_view_idx = input_accessor.bufferView;
        channel_info->sample_buffer_.offset = input_accessor.byteOffset;
        channel_info->sample_buffer_.format = engine::renderer::Format::R32_SFLOAT;

        channel_info->data_buffer_.buffer_view_idx = output_accessor.bufferView;
        channel_info->data_buffer_.offset = output_accessor.byteOffset;

        const auto& sampler_buffer_view = model.bufferViews[input_accessor.bufferView];
        const auto& data_buffer_view = model.bufferViews[output_accessor.bufferView];
        const auto& sampler_buffer = model.buffers[sampler_buffer_view.buffer];
        const auto& data_buffer = model.buffers[data_buffer_view.buffer];
        assert(sampler_buffer_view.byteStride == 0);
        assert(data_buffer_view.byteStride == 0);
        const auto sampler_start = sampler_buffer.data.data() + input_accessor.byteOffset + sampler_buffer_view.byteOffset;
        const auto data_start = data_buffer.data.data() + output_accessor.byteOffset + data_buffer_view.byteOffset;
        std::vector<float> frames(input_accessor.count);
        std::memcpy(
            frames.data(),
            sampler_start,
            sizeof(float) * input_accessor.count);
        const float* frame_time = (const float*)frames.data();

        assert(input_accessor.count == output_accessor.count);
        auto sample_count = input_accessor.count;
        if (output_accessor.type == TINYGLTF_TYPE_VEC4) {
            std::vector<glm::vec4> channel_data(sample_count);
            std::memcpy(
                channel_data.data(),
                data_start,
                sizeof(glm::vec4) * sample_count);

            for (auto i = 0; i < sample_count; i++) {
                channel_info->samples_.push_back(std::make_pair(frames[i], channel_data[i]));
            }
        }
        else {
            std::vector<glm::vec3> channel_data(sample_count);
            std::memcpy(
                channel_data.data(),
                data_buffer.data.data() + output_accessor.byteOffset + data_buffer_view.byteOffset,
                sizeof(glm::vec3) * sample_count);

            for (auto i = 0; i < sample_count; i++) {
                channel_info->samples_.push_back(std::make_pair(frames[i], glm::vec4(channel_data[i], 0.0f)));
            }
        }

        anim_info.channels_.push_back(channel_info);
    }
}

static void setupAnimations(
    const tinygltf::Model& model,
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    drawable_object->animations_.resize(model.animations.size());
    for (int i_anim = 0; i_anim < model.animations.size(); i_anim++) {
        setupAnimation(model, model.animations[i_anim], drawable_object->animations_[i_anim]);
    }
}

static void setupSkin(
    const std::shared_ptr<renderer::Device>& device,
    const tinygltf::Model& model,
    const tinygltf::Skin& src_skin,
    ego::SkinInfo& skin_info) {

    skin_info.name_ = src_skin.name;
    // Find the root node of the skeleton
    skin_info.skeleton_root_ = src_skin.skeleton;

    // Find joint nodes
    for (auto joint_index : src_skin.joints) {
        skin_info.joints_.push_back(joint_index);
    }

    // Get the inverse bind matrices from the buffer associated to this skin
    if (src_skin.inverseBindMatrices > -1)
    {
        const tinygltf::Accessor& accessor = model.accessors[src_skin.inverseBindMatrices];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        const auto src_buffer_data = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
        const auto src_data_size = accessor.count * sizeof(glm::mat4);
        skin_info.inverse_bind_matrices_.resize(accessor.count);
        memcpy(skin_info.inverse_bind_matrices_.data(), src_buffer_data, src_data_size);

        renderer::Helper::createBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            skin_info.joints_buffer_.buffer,
            skin_info.joints_buffer_.memory,
            std::source_location::current(),
            src_data_size,
            src_buffer_data);
    }
}

static void setupSkins(
    const std::shared_ptr<renderer::Device>& device,
    const tinygltf::Model& model,
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    drawable_object->skins_.resize(model.skins.size());
    for (int i_skin = 0; i_skin < model.skins.size(); i_skin++) {
        setupSkin(
            device,
            model,
            model.skins[i_skin],
            drawable_object->skins_[i_skin]);
    }
}

// ── RT-shadow skeleton capture (glTF path) ───────────────────────────
// CPU snapshot of every skinned primitive's POSITION / JOINTS_0 /
// WEIGHTS_0 + indices, concatenated across meshes, decoded from the
// tinygltf accessors (u8/u16 joints, float/normalized-u8/u16 weights,
// u8/u16/u32 indices).  Consumed by ClusterRenderer::updateRtSkeletons,
// which CPU-skins it into world space per frame so the character casts
// shadows in both RT shadow modes.  The .rwchar loader captures the
// same structure from its own MeshData.
static void captureRtSkinSource(
    const tinygltf::Model& model,
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    if (model.skins.empty()) return;

    auto src = std::make_shared<ego::RtSkinSource>();
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            auto itp = prim.attributes.find("POSITION");
            auto itj = prim.attributes.find("JOINTS_0");
            auto itw = prim.attributes.find("WEIGHTS_0");
            if (itp == prim.attributes.end() ||
                itj == prim.attributes.end() ||
                itw == prim.attributes.end() ||
                prim.indices < 0) {
                continue;   // non-skinned primitive
            }
            // Skip non-opaque primitives: character assets often carry a
            // translucent shadow-catcher / ground quad (invisible in the
            // forward pass) — treating it as an opaque RT caster paints a
            // big black square under the character.  Parity rule: BLEND
            // never casts; near-zero constant base alpha never casts.
            if (prim.material >= 0 &&
                prim.material < (int)model.materials.size()) {
                const auto& mat = model.materials[prim.material];
                if (mat.alphaMode == "BLEND") continue;
                const auto& bcf =
                    mat.pbrMetallicRoughness.baseColorFactor;
                if (bcf.size() == 4 && bcf[3] < 0.5) continue;
            }

            const auto& pa = model.accessors[itp->second];
            const auto& ja = model.accessors[itj->second];
            const auto& wa = model.accessors[itw->second];
            if (pa.count == 0 ||
                ja.count != pa.count || wa.count != pa.count) continue;

            const uint32_t vbase = (uint32_t)src->positions.size();

            // POSITION — always float vec3 per the glTF spec.
            {
                const auto& bv = model.bufferViews[pa.bufferView];
                const uint8_t* base = model.buffers[bv.buffer].data.data() +
                                      pa.byteOffset + bv.byteOffset;
                const size_t stride =
                    bv.byteStride ? bv.byteStride : 3 * sizeof(float);
                for (size_t v = 0; v < pa.count; ++v) {
                    glm::vec3 p;
                    std::memcpy(&p, base + v * stride, sizeof(p));
                    src->positions.push_back(p);
                }
            }
            // JOINTS_0 — u8vec4 or u16vec4.
            {
                const auto& bv = model.bufferViews[ja.bufferView];
                const uint8_t* base = model.buffers[bv.buffer].data.data() +
                                      ja.byteOffset + bv.byteOffset;
                const bool u8 = ja.componentType ==
                                TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
                const size_t stride =
                    bv.byteStride ? bv.byteStride : (u8 ? 4u : 8u);
                for (size_t v = 0; v < ja.count; ++v) {
                    const uint8_t* d = base + v * stride;
                    if (u8) {
                        src->joints.push_back(
                            glm::u16vec4(d[0], d[1], d[2], d[3]));
                    } else {
                        uint16_t t[4];
                        std::memcpy(t, d, sizeof(t));
                        src->joints.push_back(
                            glm::u16vec4(t[0], t[1], t[2], t[3]));
                    }
                }
            }
            // WEIGHTS_0 — float4 or normalized u8/u16 vec4.
            {
                const auto& bv = model.bufferViews[wa.bufferView];
                const uint8_t* base = model.buffers[bv.buffer].data.data() +
                                      wa.byteOffset + bv.byteOffset;
                const int ct = wa.componentType;
                const size_t comp =
                    (ct == TINYGLTF_COMPONENT_TYPE_FLOAT)          ? 4u
                    : (ct == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) ? 2u
                                                                     : 1u;
                const size_t stride =
                    bv.byteStride ? bv.byteStride : comp * 4u;
                for (size_t v = 0; v < wa.count; ++v) {
                    const uint8_t* d = base + v * stride;
                    glm::vec4 w;
                    if (comp == 4) {
                        std::memcpy(&w, d, sizeof(w));
                    } else if (comp == 2) {
                        uint16_t t[4];
                        std::memcpy(t, d, sizeof(t));
                        w = glm::vec4(t[0], t[1], t[2], t[3]) / 65535.0f;
                    } else {
                        w = glm::vec4(d[0], d[1], d[2], d[3]) / 255.0f;
                    }
                    src->weights.push_back(w);
                }
            }
            // Indices — u8/u16/u32 scalar, rebased onto the concatenated
            // capture arrays.
            {
                const auto& ia = model.accessors[prim.indices];
                const auto& bv = model.bufferViews[ia.bufferView];
                const uint8_t* base = model.buffers[bv.buffer].data.data() +
                                      ia.byteOffset + bv.byteOffset;
                for (size_t i = 0; i < ia.count; ++i) {
                    uint32_t idx = 0;
                    switch (ia.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                        idx = base[i];
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        uint16_t t;
                        std::memcpy(&t, base + i * 2u, 2u);
                        idx = t;
                        break;
                    }
                    default:
                        std::memcpy(&idx, base + i * 4u, 4u);
                        break;
                    }
                    src->indices.push_back(idx + vbase);
                }
            }
        }
    }

    if (!src->positions.empty() && src->indices.size() >= 3) {
        std::printf(
            "[RT_SKEL] glTF skin capture: %zu verts, %zu tris\n",
            src->positions.size(), src->indices.size() / 3);
        drawable_object->rt_skin_source_ = std::move(src);
    }
}

static void setupNode(
    const tinygltf::Model& model,
    const uint32_t node_idx,
    std::shared_ptr<ego::DrawableData>& drawable_object) {

    const auto& node = model.nodes[node_idx];
    auto& node_info = drawable_object->nodes_[node_idx];
    node_info.name_ = node.name;

    if (node.matrix.size() == 16) {
        // Use 'matrix' attribute
        const auto& m = node.matrix.data();
        node_info.matrix_ =
            glm::mat4(
                m[0], m[1], m[2], m[3],
                m[4], m[5], m[6], m[7],
                m[8], m[9], m[10], m[11],
                m[12], m[13], m[14], m[15]);
    }

    if (node.scale.size() == 3) {
        node_info.scale_ =
            glm::vec3(
                node.scale[0],
                node.scale[1],
                node.scale[2]);
    }

    if (node.rotation.size() == 4) {
        node_info.rotation_ =
            glm::quat(
                static_cast<float>(node.rotation[3]),
                static_cast<float>(node.rotation[0]),
                static_cast<float>(node.rotation[1]),
                static_cast<float>(node.rotation[2]));
    }

    if (node.translation.size() == 3) {
        node_info.translation_ = 
            glm::vec3(
                node.translation[0],
                node.translation[1],
                node.translation[2]);
    }

    node_info.mesh_idx_ = node.mesh;
    node_info.skin_idx_ = node.skin;

    // Draw child nodes.
    node_info.child_idx_.resize(node.children.size());
    for (size_t i = 0; i < node.children.size(); i++) {
        assert(node.children[i] < model.nodes.size());
        node_info.child_idx_[i] = node.children[i];
        drawable_object->nodes_[node_info.child_idx_[i]].parent_idx_ = node_idx;
    }
}

static void setupNodes(
    const tinygltf::Model& model, 
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    drawable_object->nodes_.resize(model.nodes.size());
    for (int i_node = 0; i_node < model.nodes.size(); i_node++) {
        setupNode(model, i_node, drawable_object);
    }
}

static void setupModel(
    const tinygltf::Model& model,
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    assert(model.scenes.size() > 0);
    drawable_object->default_scene_ = model.defaultScene;
    drawable_object->scenes_.resize(model.scenes.size());
    for (uint32_t i_scene = 0; i_scene < model.scenes.size(); i_scene++) {
        const auto& src_scene = model.scenes[i_scene];
        auto& dst_scene = drawable_object->scenes_[i_scene];

        dst_scene.nodes_.resize(src_scene.nodes.size());
        for (size_t i_node = 0; i_node < src_scene.nodes.size(); i_node++) {
            dst_scene.nodes_[i_node] = src_scene.nodes[i_node];
        }
    }
}

static void setupNode(
    const ufbx_abi ufbx_scene* fbx_scene,
    const uint32_t node_idx,
    std::shared_ptr<ego::DrawableData>& drawable_object) {

    const auto& node = fbx_scene->nodes[node_idx];
    auto& node_info = drawable_object->nodes_[node_idx];
    node_info.name_ = node->name.data;

    auto& to_parent = node->node_to_parent;

    node_info.matrix_ =
        glm::mat4(
            to_parent.m00, to_parent.m10, to_parent.m20, 0,
            to_parent.m01, to_parent.m11, to_parent.m21, 0,
            to_parent.m02, to_parent.m12, to_parent.m22, 0,
            to_parent.m03, to_parent.m13, to_parent.m23, 1.0f);

    node_info.scale_ =
        glm::vec3(
            node->local_transform.scale.x,
            node->local_transform.scale.y,
            node->local_transform.scale.z);

    node_info.rotation_ =
        glm::quat(
            float(node->local_transform.rotation.x),
            float(node->local_transform.rotation.y),
            float(node->local_transform.rotation.z),
            float(node->local_transform.rotation.w));

    node_info.translation_ =
        glm::vec3(
            node->local_transform.translation.x,
            node->local_transform.translation.y,
            node->local_transform.translation.z);

    auto joint_mat =
        glm::translate(glm::mat4(1.0f), node_info.translation_) *
        glm::mat4(node_info.rotation_) *
        glm::scale(glm::mat4(1.0f), node_info.scale_);

    if (node->mesh) {
        for (int i_mesh = 0; i_mesh < fbx_scene->meshes.count; i_mesh++) {
            if (fbx_scene->meshes[i_mesh]->element_id == node->mesh->element_id) {
                node_info.mesh_idx_ = i_mesh;
                break;
            }
        }
    }

//    node_info.skin_idx_ = node.skin;

    node_info.parent_idx_ = -1;
    if (node->parent) {
        for (size_t i = 0; i < fbx_scene->nodes.count; i++) {
            if (node->parent->element_id == fbx_scene->nodes[i]->element_id) {
                node_info.parent_idx_ = int32_t(i);
                break;
            }
        }
    }

    // Draw child nodes.
    node_info.child_idx_.resize(node->children.count);
    for (size_t i_node = 0; i_node < node->children.count; i_node++) {
        node_info.child_idx_[i_node] = -1;
        for (size_t i = 0; i < fbx_scene->nodes.count; i++) {
            if (node->children[i_node]->element_id == fbx_scene->nodes[i]->element_id) {
                node_info.child_idx_[i_node] = int32_t(i);
                break;
            }
        }
        drawable_object->nodes_[node_info.child_idx_[i_node]].parent_idx_ = node_idx;
    }
}

static void setupNodes(
    const ufbx_abi ufbx_scene* fbx_scene,
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    drawable_object->nodes_.resize(fbx_scene->nodes.count);
    for (int i_node = 0; i_node < fbx_scene->nodes.count; i_node++) {
        setupNode(fbx_scene, i_node, drawable_object);
    }
}

static void setupModel(
    const ufbx_abi ufbx_scene* fbx_scene,
    std::shared_ptr<ego::DrawableData>& drawable_object) {

    drawable_object->default_scene_ = 0;
    drawable_object->scenes_.resize(1);
    auto& dst_scene = drawable_object->scenes_[0];

    dst_scene.nodes_.resize(1);
    dst_scene.nodes_[0] = 0;
}

static void setupRaytracing(
    std::shared_ptr<ego::DrawableData>& drawable_object) {

    for (auto& mesh : drawable_object->meshes_) {
        for (auto& prim : mesh.primitives_) {
            prim.as_geometry = std::make_shared<renderer::AccelerationStructureGeometry>();
            prim.as_geometry->flags = SET_FLAG_BIT(Geometry, OPAQUE_BIT_KHR);
            assert(prim.tag_.topology == static_cast<uint32_t>(renderer::PrimitiveTopology::TRIANGLE_LIST));
            prim.as_geometry->geometry_type = renderer::GeometryType::TRIANGLES_KHR;
            auto& dst_prim = prim.as_geometry->geometry.triangles;
            dst_prim.struct_type =
                renderer::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            bool has_position = false;
            for (int i = 0; i < prim.attribute_descs_.size(); i++) {
                auto& attr = prim.attribute_descs_[i];
                auto& vertex_buffer_view =
                    drawable_object->buffer_views_[attr.buffer_view];
                auto vertex_device_address =
                    drawable_object->buffers_[vertex_buffer_view.buffer_idx].buffer->getDeviceAddress();
                if (attr.location == VINPUT_POSITION) {
                    dst_prim.vertex_format = attr.format;
                    dst_prim.vertex_stride = prim.binding_descs_[i].stride;
                    assert((attr.buffer_offset % sizeof(float)) == 0);
                    dst_prim.vertex_data.device_address = vertex_device_address + attr.buffer_offset;
                    prim.as_geometry->position.base = static_cast<uint32_t>(attr.buffer_offset / sizeof(float));
                    prim.as_geometry->position.stride = prim.binding_descs_[i].stride / sizeof(float);
                    dst_prim.max_vertex = static_cast<uint32_t>(vertex_buffer_view.range / prim.binding_descs_[i].stride);
                    has_position = true;
                }
                else if (attr.location == VINPUT_NORMAL) {
                    assert((attr.buffer_offset % sizeof(float)) == 0);
                    prim.as_geometry->normal.base = static_cast<uint32_t>(attr.buffer_offset / sizeof(float));
                    prim.as_geometry->normal.stride = prim.binding_descs_[i].stride / sizeof(float);
                }
                else if (attr.location == VINPUT_TEXCOORD0) {
                    assert((attr.buffer_offset % sizeof(float)) == 0);
                    prim.as_geometry->uv.base = static_cast<uint32_t>(attr.buffer_offset / sizeof(float));
                    prim.as_geometry->uv.stride = prim.binding_descs_[i].stride / sizeof(float);
                }
                else if (attr.location == VINPUT_COLOR) {
                    assert((attr.buffer_offset % sizeof(float)) == 0);
                    prim.as_geometry->color.base = static_cast<uint32_t>(attr.buffer_offset / sizeof(float));
                    prim.as_geometry->color.stride = prim.binding_descs_[i].stride / sizeof(float);
                }
            }
            assert(has_position);

            uint32_t cur_lod = 0;
            auto& index_buffer_view = drawable_object->buffer_views_[prim.index_desc_[cur_lod].buffer_view];
            auto index_device_address = drawable_object->buffers_[index_buffer_view.buffer_idx].buffer->getDeviceAddress();
            auto index_offset = prim.index_desc_[cur_lod].offset + index_buffer_view.offset;
            dst_prim.index_type = prim.index_desc_[cur_lod].index_type;
            dst_prim.index_data.device_address = index_device_address + index_offset;
            prim.as_geometry->max_primitive_count = static_cast<uint32_t>(prim.index_desc_[cur_lod].index_count) / 3;
            prim.as_geometry->index_base = static_cast<uint32_t>(index_offset / sizeof(uint16_t));
            prim.as_geometry->index_by_bytes = 2;
        }
    }
}

static renderer::WriteDescriptorList addDrawableTextures(
    const std::shared_ptr<ego::DrawableData>& drawable_object,
    const ego::MaterialInfo& material,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex) {

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(12);
    auto& textures = drawable_object->textures_;

    auto& black_tex = renderer::Helper::getBlackTexture();
    auto& white_tex = renderer::Helper::getWhiteTexture();

    // Helper: a texture slot is usable only when it has a GPU view —
    // VT-only baked textures (.rwtex → cpu_pixels, no full-res image)
    // carry a valid index but a null view; bind the fallback for those
    // so the descriptor set stays complete.
    auto tex_or = [&textures](int32_t idx,
                              const renderer::TextureInfo& fallback)
        -> const renderer::TextureInfo& {
        return (idx < 0 ||
                idx >= static_cast<int32_t>(textures.size()) ||
                !textures[idx].view)
            ? fallback : textures[idx];
    };

    // base color.
    const auto& base_color_tex_view =
        tex_or(material.base_color_idx_, black_tex);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ALBEDO_TEX_INDEX,
        texture_sampler,
        base_color_tex_view.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // specular.
    const auto& specular_tex_view =
        tex_or(material.specular_color_idx_, black_tex);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SPECULAR_TEX_INDEX,
        texture_sampler,
        specular_tex_view.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // normal.
    const auto& normal_tex_view =
        tex_or(material.normal_idx_, black_tex);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        NORMAL_TEX_INDEX,
        texture_sampler,
        normal_tex_view.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // metallic roughness.
    const auto& metallic_roughness_tex =
        tex_or(material.metallic_roughness_idx_, black_tex);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        METAL_ROUGHNESS_TEX_INDEX,
        texture_sampler,
        metallic_roughness_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // emissive.
    const auto& emissive_tex =
        tex_or(material.emissive_idx_, black_tex);


    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        EMISSIVE_TEX_INDEX,
        texture_sampler,
        emissive_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // occlusion.
    const auto& occlusion_tex =
        tex_or(material.occlusion_idx_, white_tex);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        OCCLUSION_TEX_INDEX,
        texture_sampler,
        occlusion_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // thin_film_lut.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        THIN_FILM_LUT_INDEX,
        texture_sampler,
        thin_film_lut_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // ── Alpha-only companion (R8) for shadow / depth-only ─────────────
    // If this material's albedo texture has a real-cutout alpha
    // companion (Phase 1's scan built one and stored it on TextureInfo),
    // bind that view here so base_depthonly.frag can do its mask-discard
    // with 4× less bandwidth than re-sampling the full RGBA albedo.
    //
    // For materials without a companion (Opaque, Blend, or Mask with no
    // real cutout α), bind the global white fallback texture.  The
    // shadow pipeline for those materials usually has no fragment shader
    // at all (Phase 1 of the previous change), so this binding is never
    // sampled — the only reason we write it is to keep the descriptor
    // set complete (Vulkan rejects bound desc sets with unwritten
    // bindings, modulo VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT which
    // we don't use).  Reading .r of an all-white RGBA8 returns 1.0, so
    // any inadvertent sample produces "no discard" — safe.
    const auto* alpha_only_src = &white_tex;
    if (material.base_color_idx_ >= 0 &&
        material.base_color_idx_ <
            static_cast<int32_t>(textures.size())) {
        const auto& albedo = textures[material.base_color_idx_];
        if (albedo.alpha_only_view) {
            alpha_only_src = &albedo;
        }
    }
    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ALPHA_ONLY_TEX_INDEX,
        texture_sampler,
        alpha_only_src == &white_tex
            ? white_tex.view
            : alpha_only_src->alpha_only_view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::UNIFORM_BUFFER,
        PBR_CONSTANT_INDEX,
        material.uniform_buffer_.buffer,
        sizeof(glsl::PbrMaterialParams));

    return descriptor_writes;
}

static void updateDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& material_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& skin_desc_set_layout,
    const std::shared_ptr<ego::DrawableData>& drawable_object,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex)
{
    for (auto& material : drawable_object->materials_) {
        material.desc_set_ = device->createDescriptorSets(
            descriptor_pool, material_desc_set_layout, 1)[0];

        // create a global ibl texture descriptor set.
        auto material_descs = addDrawableTextures(
            drawable_object,
            material,
            texture_sampler,
            thin_film_lut_tex);

        device->updateDescriptorSets(material_descs);
    }

    for (auto& skin : drawable_object->skins_) {
        skin.desc_set_ = device->createDescriptorSets(
            descriptor_pool, skin_desc_set_layout, 1)[0];

        renderer::WriteDescriptorList skin_buffer_descs;
        renderer::Helper::addOneBuffer(
            skin_buffer_descs,
            skin.desc_set_,
            renderer::DescriptorType::STORAGE_BUFFER,
            JOINT_CONSTANT_INDEX,
            skin.joints_buffer_.buffer,
            static_cast<uint32_t>(skin.joints_.size() * sizeof(glm::mat4)));

        device->updateDescriptorSets(skin_buffer_descs);
    }
}

// ── Depth-only pipeline hash + opaque-bit distinguisher ───────────────────
// For the SHADOW pass we may build TWO different pipelines for what is
// otherwise the "same" primitive: one with the alpha-mask fragment shader
// + forced-double-sided (for Mask/Blend materials), and one with no
// fragment shader + authored-double-sided (for Opaque).  They differ ONLY
// in material alpha mode, so primitive.getDepthonlyHash() alone collides
// between the two.  We OR a single sentinel bit into the hash to keep
// them in the same map without collisions.
//
// The bit value is arbitrary as long as it doesn't overlap any bit
// getDepthonlyHash() actually uses (its return is built from the
// primitive's vertex layout flags + topology + buffer view indices, all
// of which fit comfortably in the low ~32 bits).  Using a high bit of a
// 64-bit size_t guarantees no collision on x86-64 / ARM64.
//
// Defined here (above drawMesh) so the draw-time hash lookup can call
// them; the pipeline-build sites further down in this file reuse them.
static constexpr size_t kDepthonlyHashOpaqueBit = size_t(1) << 63;

static inline bool isPrimitiveOpaque(
    const ego::PrimitiveInfo& primitive,
    const std::vector<ego::MaterialInfo>& materials) {
    // effective_opaque_ is set at material-load time by the post-load
    // texture-content scan (computeEffectiveOpaqueForMaterials below):
    //   • alpha_mode == Opaque  →  true
    //   • alpha_mode == Blend   →  false
    //   • alpha_mode == Mask    →  true iff albedo texture has no real
    //                              cutout α (every texel α ≈ 255).
    // No-material primitives are vacuously opaque.
    return (primitive.material_idx_ < 0) ||
        materials[primitive.material_idx_].effective_opaque_;
}

// ── Format-based alpha-channel detector ──────────────────────────────────
// Returns true iff `format` carries an alpha channel.  A texture stored in
// an RGB-only format (R8G8B8_UNORM, BC1_RGB, BC4_UNORM_BLOCK, etc.) cannot
// possibly contain cutout alpha — every "texel α" is implicitly 1.0 in the
// shader.  We use this to short-circuit the CPU-pixel scan for DDS/BC
// textures where cpu_pixels is unavailable: if the GPU format has no α
// channel, the material is definitively no-cutout regardless of how the
// asset author flagged its AlphaMode.
//
// Default for unknown formats is `true` (assume α present) so we stay on
// the safe side and force the frag-shader path.
static bool formatHasAlphaChannel(renderer::Format f) {
    using F = renderer::Format;
    switch (f) {
        // Single / two / three-channel uncompressed formats.
        case F::R8_UNORM:        case F::R8_SNORM:
        case F::R8_USCALED:      case F::R8_SSCALED:
        case F::R8_UINT:         case F::R8_SINT:        case F::R8_SRGB:
        case F::R8G8_UNORM:      case F::R8G8_SNORM:
        case F::R8G8_USCALED:    case F::R8G8_SSCALED:
        case F::R8G8_UINT:       case F::R8G8_SINT:      case F::R8G8_SRGB:
        case F::R8G8B8_UNORM:    case F::R8G8B8_SNORM:
        case F::R8G8B8_USCALED:  case F::R8G8B8_SSCALED:
        case F::R8G8B8_UINT:     case F::R8G8B8_SINT:    case F::R8G8B8_SRGB:
        case F::B8G8R8_UNORM:    case F::B8G8R8_SNORM:
        case F::B8G8R8_USCALED:  case F::B8G8R8_SSCALED:
        case F::B8G8R8_UINT:     case F::B8G8R8_SINT:    case F::B8G8R8_SRGB:
        case F::R5G6B5_UNORM_PACK16:  case F::B5G6R5_UNORM_PACK16:
        case F::R16_UNORM:       case F::R16_SNORM:
        case F::R16_USCALED:     case F::R16_SSCALED:
        case F::R16_UINT:        case F::R16_SINT:       case F::R16_SFLOAT:
        case F::R16G16_UNORM:    case F::R16G16_SNORM:
        case F::R16G16_USCALED:  case F::R16G16_SSCALED:
        case F::R16G16_UINT:     case F::R16G16_SINT:    case F::R16G16_SFLOAT:
        case F::R16G16B16_UNORM:    case F::R16G16B16_SNORM:
        case F::R16G16B16_USCALED:  case F::R16G16B16_SSCALED:
        case F::R16G16B16_UINT:     case F::R16G16B16_SINT:
        case F::R16G16B16_SFLOAT:
        case F::R32_UINT:    case F::R32_SINT:    case F::R32_SFLOAT:
        case F::R32G32_UINT: case F::R32G32_SINT: case F::R32G32_SFLOAT:
        case F::R32G32B32_UINT: case F::R32G32B32_SINT: case F::R32G32B32_SFLOAT:
        case F::B10G11R11_UFLOAT_PACK32:
        case F::E5B9G9R9_UFLOAT_PACK32:
        // BC1 RGB-only variants (BC1_RGBA_* DO carry 1-bit α).
        case F::BC1_RGB_UNORM_BLOCK:  case F::BC1_RGB_SRGB_BLOCK:
        // BC4 = single channel.  BC5 = two channel.  BC6H = RGB float.
        case F::BC4_UNORM_BLOCK: case F::BC4_SNORM_BLOCK:
        case F::BC5_UNORM_BLOCK: case F::BC5_SNORM_BLOCK:
        case F::BC6H_UFLOAT_BLOCK: case F::BC6H_SFLOAT_BLOCK:
        // ETC2 RGB-only (ETC2_R8G8B8A1 / A8 carry α).
        case F::ETC2_R8G8B8_UNORM_BLOCK: case F::ETC2_R8G8B8_SRGB_BLOCK:
        case F::EAC_R11_UNORM_BLOCK:  case F::EAC_R11_SNORM_BLOCK:
        case F::EAC_R11G11_UNORM_BLOCK: case F::EAC_R11G11_SNORM_BLOCK:
            return false;
        default:
            // RGBA8, BGRA8, BC1_RGBA, BC2/3/7, ETC2_R8G8B8A*, ASTC,
            // A*-prefixed packed formats, all 4-channel R*G*B*A* — they
            // all carry α.  Same for unknown / depth / stencil formats
            // (which don't appear in the albedo path anyway).
            return true;
    }
}

// ── Texture-content-aware opaque classifier ──────────────────────────────
// Walks every material on a freshly-loaded DrawableData and decides whether
// its shadow path can take the fast no-fragment-shader pipeline.  Runs once
// per asset, right after textures + materials are populated.
//
// Mask materials with effectively-opaque textures are the most interesting
// case: asset authors often default to AlphaMode::Mask "to be safe", but
// the texture itself contains no transparent texels (every α == 255).  For
// shadow purposes those behave identically to Opaque, and we want them on
// the fast path.  Mask materials with REAL cutout α (foliage, fences, etc.)
// stay on the alpha-discard frag-shader path so silhouettes are correct.
//
// The scan is O(W × H) per albedo texture and runs ONCE at load time, so
// even a 2K-texture-heavy scene amortises to a few ms total at startup.
//
// Threshold: any α < kOpaqueAlphaThreshold counts as "real transparency".
// 250/255 ≈ 0.98 leaves headroom for compression noise / JPEG-via-PNG
// artifacts where a fully-opaque texel comes out at 254 instead of 255.
static void computeEffectiveOpaqueForMaterials(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<ego::DrawableData>& drawable_object) {
    if (!drawable_object) return;

    constexpr uint8_t kOpaqueAlphaThreshold = 250;

    // effective_opaque_ means: "the shadow / depth-only pass can skip
    // the fragment shader entirely for this material."  It's false
    // whenever the depth-only frag shader needs to run its alpha-cutoff
    // discard — and importantly, we ERR ON THE SIDE OF RUNNING IT.
    //
    //   alpha_mode == Opaque                       → true   (no frag)
    //   alpha_mode == Blend                        → false  (frag still
    //                                                       discards on
    //                                                       per-fragment α)
    //   alpha_mode == Mask, no albedo              → true   (nothing to
    //                                                       discard against)
    //   alpha_mode == Mask, cpu_pixels missing     → false  (DDS / GPU-only
    //                                                       textures — can't
    //                                                       scan, but the
    //                                                       frag path still
    //                                                       samples the full
    //                                                       albedo's α
    //                                                       correctly.  This
    //                                                       is the path
    //                                                       Bistro foliage
    //                                                       takes.)
    //   alpha_mode == Mask, scan finds NO cutout   → true   (texture is
    //                                                       solid α — over-
    //                                                       flagged by author)
    //   alpha_mode == Mask, scan finds cutout      → false  (build alpha-only
    //                                                       companion for the
    //                                                       Phase-2 optim
    //                                                       once that's
    //                                                       wired up; for
    //                                                       now the frag
    //                                                       shader samples
    //                                                       baseColor.a
    //                                                       directly)
    //
    // The KEY DIFFERENCE from the previous "tighten" pass: missing
    // cpu_pixels no longer drops a Mask material into the no-frag
    // path.  That path was incorrect for DDS-loaded foliage (the only
    // signal of cutout is the GPU texture's alpha; skipping the frag
    // shader means the discard never fires and foliage casts solid
    // shadows).
    int stats_total = 0;
    int stats_opaque = 0, stats_mask = 0, stats_blend = 0;
    int stats_mask_with_cutout = 0;   // Mask materials that need frag shader
    int stats_mask_solid = 0;         // Mask materials whose texture had no α (scan upgrade)
    int stats_mask_unscanned = 0;     // Mask materials we couldn't scan (DDS path)
    int stats_alpha_companions_built = 0;
    size_t alpha_companion_bytes = 0;

    for (auto& mat : drawable_object->materials_) {
        ++stats_total;

        // Tally raw alpha_mode_ counts so the per-frame log line can
        // show actual Mask / Blend / Opaque mix regardless of whether
        // companions get built.
        if (mat.alpha_mode_ == ego::AlphaMode::Opaque) ++stats_opaque;
        else if (mat.alpha_mode_ == ego::AlphaMode::Blend) ++stats_blend;
        else ++stats_mask;

        // Easy classifications: Opaque is the only case where we can
        // confidently skip the shadow frag shader.  Blend still needs
        // the frag path so its per-fragment α can drive the discard
        // (Blend materials don't reach the shadow draw most of the
        // time anyway — the cull bucket filters them out — but if
        // they DO get drawn we want correct depth output).
        if (mat.alpha_mode_ == ego::AlphaMode::Opaque) {
            mat.effective_opaque_ = true;
            continue;
        }
        if (mat.alpha_mode_ == ego::AlphaMode::Blend) {
            mat.effective_opaque_ = false;
            continue;
        }

        // alpha_mode == Mask from here on.
        // Default to needing the frag path; only upgrade to no-frag
        // when we POSITIVELY prove there's no real cutout in the
        // texture (which requires a scannable cpu_pixels).
        mat.effective_opaque_ = false;

        if (mat.base_color_idx_ < 0 ||
            mat.base_color_idx_ >=
                static_cast<int32_t>(drawable_object->textures_.size())) {
            // No albedo — nothing to discard against, frag would be a
            // no-op.  Take the no-frag fast path.
            mat.effective_opaque_ = true;
            ++stats_mask_solid;
            continue;
        }

        auto& tex = drawable_object->textures_[mat.base_color_idx_];

        // Companion already built by an earlier material sharing this
        // texture — adopt the result.
        if (tex.alpha_only_view) {
            // companion exists → known real cutout
            ++stats_mask_with_cutout;
            continue;
        }

        if (!tex.cpu_pixels || tex.cpu_pixels->empty()) {
            // CAN'T SCAN content — typical of DDS / BC-compressed source.
            // BUT the texture's FORMAT alone is a perfect proxy: an RGB
            // (no-α) format physically cannot store cutout transparency,
            // so the AlphaMode::Mask flag is over-conservative.  Take
            // the no-frag fast path in that case.
            //
            // For α-bearing formats (RGBA8, BC1_RGBA, BC2/3/7, etc.)
            // we still can't tell whether the α channel actually has
            // cutout texels without scanning, so stay on the safe frag
            // path until a DDS-aware extractor lands.
            if (tex.image && !formatHasAlphaChannel(tex.image->getFormat())) {
                mat.effective_opaque_ = true;
                ++stats_mask_solid;
                continue;
            }
            ++stats_mask_unscanned;
            continue;
        }

        // Scan: every 4th byte (alpha channel of the RGBA8 source).
        const auto& pixels = *tex.cpu_pixels;
        bool has_transparency = false;
        for (size_t i = 3; i < pixels.size(); i += 4) {
            if (pixels[i] < kOpaqueAlphaThreshold) {
                has_transparency = true;
                break;
            }
        }

        if (!has_transparency) {
            // Texture is solid α — material was over-flagged.  Take
            // the no-frag fast path.
            mat.effective_opaque_ = true;
            ++stats_mask_solid;
            continue;
        }

        // Real cutout — confirmed.  Build the companion (data is
        // ready for the Phase-2 optimization once DDS support lands;
        // for now it's unused but harmless).
        ++stats_mask_with_cutout;
        if (!device) continue;
        const uint32_t w = tex.size.x;
        const uint32_t h = tex.size.y;
        if (w == 0 || h == 0) continue;

        std::vector<uint8_t> alpha_data(size_t(w) * h);
        for (size_t i = 0, n = alpha_data.size(); i < n; ++i) {
            alpha_data[i] = pixels[i * 4 + 3];
        }

        renderer::Helper::create2DTextureImage(
            device,
            renderer::Format::R8_UNORM,
            static_cast<int>(w),
            static_cast<int>(h),
            alpha_data.data(),
            tex.alpha_only_image,
            tex.alpha_only_memory,
            std::source_location::current());

        tex.alpha_only_view = device->createImageView(
            tex.alpha_only_image,
            renderer::ImageViewType::VIEW_2D,
            renderer::Format::R8_UNORM,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current());

        ++stats_alpha_companions_built;
        alpha_companion_bytes += alpha_data.size();
    }

    std::printf(
        "[drawable] materials: total=%d  opaque=%d  blend=%d  mask=%d "
        "(mask: with-cutout=%d, scanned-solid=%d, unscanned/DDS=%d)  "
        "companions built=%d (%.1f MB)\n",
        stats_total, stats_opaque, stats_blend, stats_mask,
        stats_mask_with_cutout, stats_mask_solid, stats_mask_unscanned,
        stats_alpha_companions_built,
        double(alpha_companion_bytes) / (1024.0 * 1024.0));

    // Engine-wide cumulative counters for the per-frame shadow log.
    // "Alpha-cutoff" = Mask materials that take the with-frag path
    // (have or might have real cutout).  Blend is reported separately.
    const int with_frag_this_call =
        stats_mask_with_cutout + stats_mask_unscanned + stats_blend;
    ego::DrawableObject::s_total_materials_count_.fetch_add(
        stats_total, std::memory_order_relaxed);
    ego::DrawableObject::s_alpha_cutoff_materials_count_.fetch_add(
        with_frag_this_call, std::memory_order_relaxed);
}

static inline size_t getDepthonlyHashForMaterial(
    const ego::PrimitiveInfo& primitive,
    const std::vector<ego::MaterialInfo>& materials) {
    return primitive.getDepthonlyHash() |
        (isPrimitiveOpaque(primitive, materials)
            ? kDepthonlyHashOpaqueBit : size_t(0));
}

static void drawMesh(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<ego::DrawableData>& drawable_object,
    const std::shared_ptr<renderer::PipelineLayout>& drawable_pipeline_layout,
    const renderer::DescriptorSetList& desc_set_list,
    const ego::MeshInfo& mesh_info,
    const ego::SkinInfo* skin_info,
    const glsl::ModelParams& model_params,
    std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>>& pipelines,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    bool depth_only,
    size_t& last_hash,
    // ── Mesh-shader CSM dispatch (DrawMode::kCsmMeshShader) ─────────
    // mesh_shader_csm_mode = true: `pipelines` IS the mesh-shader
    //   pipeline list.  For each primitive: if it has a mesh-shader
    //   descriptor set (mesh_shader_shadow_desc_set_ non-null) AND a
    //   hash-cached pipeline exists, dispatch via drawMeshTasks
    //   through the mesh-shader-shadow pipeline layout; otherwise
    //   fall back to mesh_shader_fallback_pipelines[hash] +
    //   drawIndexedIndirect using the standard pipeline layout.
    // mesh_shader_csm_mode = false: legacy behaviour — `pipelines` is
    //   the active list and the GS-style dispatch is used for every
    //   primitive.  fallback param ignored.
    bool mesh_shader_csm_mode = false,
    std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>>* mesh_shader_fallback_pipelines = nullptr) {

    // ── Debug reach: drawMesh entered ───────────────────────────────
    // Only one drawable in the scene has this flag (the player today),
    // so the branch is essentially never taken for other geometry.
    if (drawable_object->m_debug_force_red_ ||
        drawable_object->m_debug_log_draws_) {
        ++drawable_object->m_debug_draw_mesh_entered_;
    }

    // ── Per-mesh frustum culling (forward pass only) ────────────────────────
    // Transform the mesh's local-space bounding sphere into world space
    // using model_params.model_mat, then test against the frustum planes
    // set by application.cpp. Shadow / depth-only passes are NOT culled
    // so off-screen geometry still casts correct shadows.
    if (!depth_only && s_frustum_cull_active &&
        !drawable_object->m_use_node_transform_only_) {
        // ── Skinned-character bypass ────────────────────────────────
        // m_use_node_transform_only_ marks drawables whose world
        // pose is driven by PlayerController (or equivalent) and
        // typically contain a single skinned mesh.  For skinned meshes
        // the bind-pose `mesh_info.bbox_min_/bbox_max_` does NOT
        // necessarily bound the skinned vertices after the joint
        // matrices fire — even modest bone rotations can push verts
        // outside that bbox, and this sphere-vs-frustum test then
        // rejects the whole mesh while the shadow pass (which skips
        // the cull via depth_only) keeps drawing it.  That's the
        // root cause of "shadow stays, body disappears when I rotate
        // the camera".  We trade a few extra GPU vertices on this
        // single character for visual correctness.
        //
        // Compute local-space bounding sphere from AABB.
        glm::vec3 local_center = (mesh_info.bbox_min_ + mesh_info.bbox_max_) * 0.5f;
        glm::vec3 local_half   = (mesh_info.bbox_max_ - mesh_info.bbox_min_) * 0.5f;
        float local_radius     = glm::length(local_half);

        // Transform center to world space.
        glm::vec4 world_center4 = model_params.model_mat * glm::vec4(local_center, 1.0f);
        glm::vec3 world_center  = glm::vec3(world_center4);

        // Scale radius by the largest axis scale of the model matrix.
        float sx = glm::length(glm::vec3(model_params.model_mat[0]));
        float sy = glm::length(glm::vec3(model_params.model_mat[1]));
        float sz = glm::length(glm::vec3(model_params.model_mat[2]));
        float world_radius = local_radius * glm::max(sx, glm::max(sy, sz));

        // Sphere-vs-frustum: if the sphere is entirely behind any plane, cull.
        bool outside = false;
        for (int p = 0; p < 6; ++p) {
            float dist = glm::dot(glm::vec3(s_frustum_planes[p]), world_center)
                       + s_frustum_planes[p].w;
            if (dist < -world_radius) { outside = true; break; }
        }
        if (outside) {
            if (drawable_object->m_debug_force_red_ ||
                drawable_object->m_debug_log_draws_) {
                ++drawable_object->m_debug_draw_mesh_culled_frustum_;
            }
            return;
        }
    }

    // ── --cluster-debug override (forward pass only) ──────────────────────
    // When the CLI flag is live and this MeshInfo actually had its cluster
    // sidecar + GPU expansion built (only the FBX-path static meshes do so
    // far), bypass the whole primitive loop and paint the mesh with the
    // per-cluster flat-color helper.
    // When the cluster indirect draw is active this mesh is handled by the
    // cluster renderer pass — skip it entirely in BOTH the forward AND
    // shadow passes (the cluster renderer now provides drawClusterShadow,
    // a single drawIndexedIndirectCount that broadcasts to every cascade
    // via a GS, replacing the ~2400 individual per-mesh shadow draws that
    // used to dominate the shadow pass at ~17 ms CPU).  The shadow draw
    // is dispatched by the application around the existing shadow render
    // pass.  Both the forward and shadow paths read the same cull-output
    // indirect buffer, so consistency is automatic.
    // Use cluster_global_mesh_idx_ >= 0 (set during cluster upload) as the
    // "this mesh is owned by the cluster renderer" flag — independent of
    // whether debug GPU buffers happen to exist.
    if (engine::helper::clusterIndirectActive() &&
        mesh_info.cluster_global_mesh_idx_ >= 0) {
        if (drawable_object->m_debug_force_red_ ||
            drawable_object->m_debug_log_draws_) {
            ++drawable_object->m_debug_draw_mesh_taken_by_cluster_;
        }
        return;
    }

    if (!depth_only &&
        engine::helper::clusterRenderingEnabled() &&
        mesh_info.cluster_debug_gpu_.ready() &&
        ego::ClusterDebugDraw::ready()) {
        if (drawable_object->m_debug_force_red_ ||
            drawable_object->m_debug_log_draws_) {
            ++drawable_object->m_debug_draw_mesh_cluster_debug_path_;
        }
        // Cluster-debug pipeline layout only declares PBR_GLOBAL_PARAMS_SET
        // and VIEW_PARAMS_SET (same layout list as ShapeBase). Slice just
        // those two from the incoming list so the validation layers don't
        // flag a layout/set-count mismatch.
        renderer::DescriptorSetList cluster_desc_sets;
        if (desc_set_list.size() > VIEW_PARAMS_SET) {
            cluster_desc_sets = {
                desc_set_list[PBR_GLOBAL_PARAMS_SET],
                desc_set_list[VIEW_PARAMS_SET],
            };
        }
        ego::ClusterDebugDraw::draw(
            cmd_buf,
            cluster_desc_sets,
            mesh_info.cluster_debug_gpu_,
            model_params.model_mat,
            viewports,
            scissors);
        // Invalidate the normal-path's "last pipeline hash" cache so the
        // next non-debug drawMesh call correctly re-binds its pipeline.
        last_hash = 0;
        return;
    }

    // ── Sort primitives by alpha mode ─────────────────────────────────
    // Draw order within the mesh: Opaque -> Mask -> Blend. Stable
    // sort preserves source order within each bucket so cluster
    // hashing / pipeline batching still benefits from cache locality.
    // Per-mesh sorting (typically <50 primitives) is enough for
    // correctness; cross-mesh alpha sorting is the WBOIT resolve's job.
    std::vector<const ego::PrimitiveInfo*> sorted_prims;
    sorted_prims.reserve(mesh_info.primitives_.size());
    for (const auto& p : mesh_info.primitives_) sorted_prims.push_back(&p);
    std::stable_sort(
        sorted_prims.begin(), sorted_prims.end(),
        [&](const ego::PrimitiveInfo* a, const ego::PrimitiveInfo* b) {
            auto bucket = [&](const ego::PrimitiveInfo* pp) -> int {
                if (pp->material_idx_ < 0) return 0;
                switch (drawable_object->materials_[pp->material_idx_].alpha_mode_) {
                    case ego::AlphaMode::Opaque: return 0;
                    case ego::AlphaMode::Mask:   return 1;
                    case ego::AlphaMode::Blend:  return 2;
                }
                return 0;
            };
            return bucket(a) < bucket(b);
        });

    for (const auto* prim_ptr : sorted_prims) {
        const auto& prim = *prim_ptr;
        const auto& attrib_list = prim.attribute_descs_;
        if (drawable_object->m_debug_force_red_ ||
            drawable_object->m_debug_log_draws_) {
            ++drawable_object->m_debug_draw_prims_iterated_;
        }

        // Shadow pass needs the material-aware hash so opaque vs
        // mask/blend variants of the same vertex layout pick up the
        // right pipeline (opaque variant has no frag shader, mask
        // variant has the discard-capable one).  Forward pass keeps
        // the plain hash since the forward pipelines branch on alpha
        // mode inside the shader, not at pipeline level.
        auto cur_hash = depth_only
            ? getDepthonlyHashForMaterial(prim, drawable_object->materials_)
            : prim.getHash();

        // ── Mesh-shader CSM dispatch (eligible primitives only) ─────
        // Take this branch only when mesh-shader mode is active AND
        // this primitive has a populated per-primitive descriptor set
        // AND a hash-cached mesh-shader pipeline exists.  Otherwise
        // fall through to the GS-style dispatch below (using the
        // fallback pipeline list when in mesh-shader mode).
        const bool ms_eligible =
            mesh_shader_csm_mode &&
            prim.mesh_shader_shadow_desc_set_ != nullptr &&
            pipelines.find(cur_hash) != pipelines.end() &&
            pipelines[cur_hash] != nullptr;
        if (ms_eligible) {
            // Mesh-shader pipeline + layout.  Bind set 0 (this primitive's
            // VB/IB/instance SSBOs) and set RUNTIME_LIGHTS_PARAMS_SET
            // (cascade VPs UBO).  No IA bindings, no material / skin / PBR
            // sets — the mesh shader doesn't reference any of them.
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::GRAPHICS,
                pipelines[cur_hash]);
            cmd_buf->setViewports(viewports, 0, uint32_t(viewports.size()));
            cmd_buf->setScissors(scissors, 0, uint32_t(scissors.size()));
            // Force re-bind on the next non-mesh-shader primitive: the
            // pipeline layout we're about to bind differs from the GS
            // path's, so a same-hash fallback primitive after us must
            // still re-bind to refresh its descriptor-set bindings.
            last_hash = 0;

            const auto& ms_layout =
                ego::DrawableObject::getMeshShaderShadowPipelineLayout();
            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::GRAPHICS,
                ms_layout,
                { prim.mesh_shader_shadow_desc_set_ },
                /*first_set*/ 0);
            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::GRAPHICS,
                ms_layout,
                { desc_set_list[RUNTIME_LIGHTS_PARAMS_SET] },
                /*first_set*/ RUNTIME_LIGHTS_PARAMS_SET);

            // Push the MeshShadowPC (96 bytes — mat4 + 8 uints).
            struct MeshShadowPC {
                glm::mat4 model_mat;
                uint32_t  vertex_count;
                uint32_t  tri_count;
                uint32_t  vb_stride_floats;
                uint32_t  vb_position_offset_floats;
                uint32_t  ib_first_index;
                uint32_t  instance_stride_floats;
                uint32_t  pad0;
                uint32_t  pad1;
            } pc;
            pc.model_mat                 = model_params.model_mat;
            pc.vertex_count              = prim.mesh_shader_vertex_count_;
            pc.tri_count                 = prim.mesh_shader_tri_count_;
            pc.vb_stride_floats          = prim.mesh_shader_vb_stride_floats_;
            pc.vb_position_offset_floats = prim.mesh_shader_vb_position_offset_floats_;
            pc.ib_first_index            = prim.mesh_shader_ib_first_index_;
            pc.instance_stride_floats    =
                uint32_t(sizeof(glsl::InstanceDataInfo) / 4u);
            pc.pad0 = 0; pc.pad1 = 0;

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, MESH_BIT_EXT),
                ms_layout,
                &pc,
                sizeof(pc));

            // Task shader amplifies (1,1,1) → CSM_CASCADE_COUNT mesh WGs.
            cmd_buf->drawMeshTasks(1u, 1u, 1u);
            if (drawable_object->m_debug_force_red_ ||
                drawable_object->m_debug_log_draws_) {
                ++drawable_object->m_debug_draw_prim_mesh_shader_;
            }
            continue;
        }

        // ── GS-style dispatch (default + mesh-shader fallback) ──────
        // When mesh_shader_csm_mode is true and this primitive isn't
        // eligible, the caller's primary `pipelines` map (the mesh-
        // shader list) won't have an entry for cur_hash.  Substitute
        // the GS pipeline list passed via mesh_shader_fallback_pipelines.
        auto* dispatch_pipelines = &pipelines;
        if (mesh_shader_csm_mode && mesh_shader_fallback_pipelines) {
            dispatch_pipelines = mesh_shader_fallback_pipelines;
        }
        if (cur_hash != last_hash) {
            auto pipe_it = dispatch_pipelines->find(cur_hash);
            if (pipe_it == dispatch_pipelines->end() || !pipe_it->second) {
                // No pipeline cached for this primitive's hash.  Skip
                // the bind (and the eventual draw) — bindPipeline with
                // a null shared_ptr would crash inside the renderer.
                // Counts the case so the per-second [player.draw.reach]
                // print exposes "primitives reached the loop but had
                // no pipeline" as a distinct gating reason.
                if (drawable_object->m_debug_force_red_ ||
                    drawable_object->m_debug_log_draws_) {
                    ++drawable_object->m_debug_draw_prim_pipeline_null_;
                }
                continue;
            }
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::GRAPHICS,
                pipe_it->second);

            cmd_buf->setViewports(viewports, 0, uint32_t(viewports.size()));
            cmd_buf->setScissors(scissors, 0, uint32_t(scissors.size()));
            last_hash = cur_hash;
        }

        std::vector<std::shared_ptr<renderer::Buffer>> buffers(attrib_list.size());
        std::vector<uint64_t> offsets(attrib_list.size());
        for (int i_attrib = 0; i_attrib < attrib_list.size(); i_attrib++) {
            const auto& buffer_view = drawable_object->buffer_views_[attrib_list[i_attrib].buffer_view];
            buffers[i_attrib] = drawable_object->buffers_[buffer_view.buffer_idx].buffer;
            offsets[i_attrib] = attrib_list[i_attrib].buffer_offset;
        }
        cmd_buf->bindVertexBuffers(0, buffers, offsets);

        // TODO: replace with distance-based LOD selection.
        // cur_lod=1 was hardcoded which caused LOD 1 (30% faces) to always
        // render regardless of camera distance, producing stretched polygons
        // on close surfaces.  Use LOD 0 (original quality) until a proper
        // distance-based selector is wired up.
        uint32_t cur_lod = 0;
        const auto& index_buffer_view =
            drawable_object->buffer_views_[prim.index_desc_[cur_lod].buffer_view];

        cmd_buf->bindIndexBuffer(
            drawable_object->buffers_[index_buffer_view.buffer_idx].buffer,
            prim.index_desc_[cur_lod].offset + index_buffer_view.offset,
            prim.index_desc_[cur_lod].index_type);

        renderer::DescriptorSetList desc_sets = desc_set_list;
        if (prim.material_idx_ >= 0) {
            const auto& material =
                drawable_object->materials_[prim.material_idx_];
            desc_sets[PBR_MATERIAL_PARAMS_SET] = material.desc_set_;
        }

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::GRAPHICS,
            drawable_pipeline_layout,
            desc_sets);

        if (skin_info) {
            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::GRAPHICS,
                drawable_pipeline_layout,
                {skin_info->desc_set_},
                SKIN_PARAMS_SET);
        }

        // Push constants must list every stage that READS the range.
        // base.frag now reads model_params.debug_force_red /
        // debug_skip_skinning, so we must include FRAGMENT_BIT here —
        // otherwise the spec leaves fragment-stage reads of bytes
        // updated by a vertex-only push UNDEFINED.  Validation layers
        // may not catch the mismatch (the pipeline layout's range
        // includes both stages, and a subset-push is technically
        // legal), but on real hardware the fragment shader reads
        // garbage for the field and the debug_force_red override
        // silently does nothing.  This was the root cause of the
        // "no red even with debug_force_red set" symptom: the
        // pipeline layout's range correctly covered VERTEX+FRAGMENT,
        // but the per-draw push only updated VERTEX.
        cmd_buf->pushConstants(
            SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
            drawable_pipeline_layout,
            &model_params,
            sizeof(model_params));

        // Debug counter: when this drawable has the debug-force-red
        // flag set (currently only the PlayerController player),
        // tally how many primitive draws actually reach this point.
        // Reset at start of DrawableObject::draw, printed at end of
        // that function once per ~60 frames.  Confirms whether the
        // forward draw is firing N indirect draws for the player or
        // being entirely skipped upstream (frustum cull, isReady,
        // pipeline binding, etc.).
        if (drawable_object->m_debug_force_red_ ||
            drawable_object->m_debug_log_draws_) {
            ++drawable_object->m_debug_draw_call_count_;
        }

        //cmd_buf->drawIndexed(static_cast<uint32_t>(prim.index_desc_.index_count));
        cmd_buf->drawIndexedIndirect(
            drawable_object->indirect_draw_cmd_,
            prim.indirect_draw_cmd_ofs_);
    }
}

static void drawNodes(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<ego::DrawableData>& drawable_object,
    const std::shared_ptr<renderer::PipelineLayout>& drawable_pipeline_layout,
    const renderer::DescriptorSetList& desc_set_list,
    int32_t node_idx,
    std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>>& pipelines,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    bool depth_only,
    size_t& last_hash,
    uint32_t csm_cascade_idx,
    bool mesh_shader_csm_mode,
    std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>>* mesh_shader_fallback_pipelines) {
    if (node_idx >= 0) {
        if (drawable_object->m_debug_force_red_ ||
            drawable_object->m_debug_log_draws_) {
            ++drawable_object->m_debug_draw_nodes_visited_;
        }
        const auto& node = drawable_object->nodes_[node_idx];
        // Per-wrapper sub-object filter (staged by DrawableObject::draw):
        // when active, only the matching node's mesh is drawn.  Children
        // are still recursed below so a filtered node anywhere in the
        // hierarchy is reached.
        const bool node_filtered =
            drawable_object->m_only_render_node_ >= 0 &&
            drawable_object->m_only_render_node_ != node_idx;
        if (node.mesh_idx_ >= 0 && !node_filtered) {
            if (drawable_object->m_debug_force_red_ ||
                drawable_object->m_debug_log_draws_) {
                ++drawable_object->m_debug_draw_nodes_with_mesh_;
            }
            glsl::ModelParams model_params{};
            // Per-instance world override (set per-draw-call by
            // DrawableObject::draw); identity when the wrapper hasn't asked
            // for it, so non-shared drawables behave exactly as before.
            model_params.model_mat =
                drawable_object->m_current_instance_world_ *
                node.cached_matrix_;
            model_params.flip_uv_coord =
                (drawable_object->m_flip_u_ ? 0x01 : 0x00) |
                (drawable_object->m_flip_v_ ? 0x02 : 0x00);
            // Only consumed by the _CSMCASC vertex-shader permutation
            // (DrawMode::kCsmPerCascade pipelines).  Other pipelines
            // ignore this field — it sits in former-pad bytes.
            model_params.cascade_idx = csm_cascade_idx;
            // Debug "force red" override — see DrawableData::
            // m_debug_force_red_ comment.  Forwarded to base.frag via
            // push constants; depth-only / shadow / mesh-shader CSM
            // pipelines simply ignore the field (they don't read
            // model_params).
            // 1u = debug smoke-test red (whole drawable); 2u = editor
            // selection highlight for THIS node (or whole drawable when
            // m_highlight_node_ == -2).  base.frag tints amber for 2u.
            model_params.debug_force_red =
                drawable_object->m_debug_force_red_ ? 1u
                : ((drawable_object->m_highlight_node_ == node_idx ||
                    drawable_object->m_highlight_node_ == -2) ? 2u : 0u);
            // base.vert's skin-matrix multiplication is skipped when
            // this is set (see DrawableData::m_debug_skip_skinning_).
            // For depth-only / shadow / mesh-shader CSM permutations
            // the field is simply unread (those shaders don't include
            // the skinning branch).
            model_params.debug_skip_skinning =
                drawable_object->m_debug_skip_skinning_ ? 1u : 0u;

            drawMesh(cmd_buf,
                drawable_object,
                drawable_pipeline_layout,
                desc_set_list,
                drawable_object->meshes_[node.mesh_idx_],
                node.skin_idx_ > -1 ? &drawable_object->skins_[node.skin_idx_] : nullptr,
                model_params,
                pipelines,
                viewports,
                scissors,
                depth_only,
                last_hash,
                mesh_shader_csm_mode,
                mesh_shader_fallback_pipelines);

            num_draw_meshes++;
        }

        for (auto& child_idx : node.child_idx_) {
            drawNodes(cmd_buf,
                drawable_object,
                drawable_pipeline_layout,
                desc_set_list,
                child_idx,
                pipelines,
                viewports,
                scissors,
                depth_only,
                last_hash,
                csm_cascade_idx,
                mesh_shader_csm_mode,
                mesh_shader_fallback_pipelines);
        }
    }
}

// material texture descriptor set layout.
static std::shared_ptr<renderer::DescriptorSetLayout> createMaterialDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(7);

    renderer::DescriptorSetLayoutBinding ubo_pbr_layout_binding{};
    ubo_pbr_layout_binding.binding = PBR_CONSTANT_INDEX;
    ubo_pbr_layout_binding.descriptor_count = 1;
    ubo_pbr_layout_binding.descriptor_type = renderer::DescriptorType::UNIFORM_BUFFER;
    ubo_pbr_layout_binding.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    ubo_pbr_layout_binding.immutable_samplers = nullptr; // Optional
    bindings.push_back(ubo_pbr_layout_binding);

    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(ALBEDO_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(SPECULAR_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(NORMAL_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(METAL_ROUGHNESS_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(EMISSIVE_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(OCCLUSION_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(THIN_FILM_LUT_INDEX));
    // Alpha-only companion sampler used by base_depthonly.frag's mask
    // discard test (Mask materials only; written by Phase 1's scan in
    // computeEffectiveOpaqueForMaterials).  For materials without a real
    // companion, the binding is populated with a 1×1 R8 white fallback
    // so the layout always has a valid descriptor.
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(ALPHA_ONLY_TEX_INDEX));

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> createSkinDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {

    return device->createDescriptorSetLayout({
        renderer::helper::getBufferDescriptionSetLayoutBinding(
            JOINT_CONSTANT_INDEX,
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT),
            renderer::DescriptorType::STORAGE_BUFFER) });
}

// ─── Mesh-shader shadow descriptor set layout ──────────────────────────
// Used by the DrawMode::kCsmMeshShader pipeline (base_depthonly_csm.task
// + .mesh).  Set is bound at "set 0" of that pipeline's *own* layout —
// it does NOT collide with PBR_GLOBAL_PARAMS_SET because mesh-shader
// shadow pipelines have an isolated pipeline layout that doesn't expose
// the PBR / view / material / skin sets at all.  All three bindings
// are storage buffers consumed by the mesh shader stage:
//   binding 0: VB (position-only float view of the primitive's VB)
//   binding 1: IB (uint indices for the primitive's tri list)
//   binding 2: instance_buffer (per-instance TRS, slot 0 used)
static std::shared_ptr<renderer::DescriptorSetLayout> createMeshShaderShadowDescSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(3);
    for (uint32_t b = 0; b < 3; ++b) {
        bindings.push_back(
            renderer::helper::getBufferDescriptionSetLayoutBinding(
                b,
                SET_FLAG_BIT(ShaderStage, MESH_BIT_EXT),
                renderer::DescriptorType::STORAGE_BUFFER));
    }
    return device->createDescriptorSetLayout(bindings);
}

// Pipeline layout for the mesh-shader CSM shadow path.  Two descriptor
// sets:
//   set 0                          : mesh-shader VB/IB/instance SSBOs
//   set RUNTIME_LIGHTS_PARAMS_SET  : runtime-lights UBO (cascade VPs)
// Plus a 96-byte push constant range (MeshShadowPC) covering the
// MESH_BIT_EXT stage only.  Empty descriptor-set layouts fill the slots
// between 0 and RUNTIME_LIGHTS_PARAMS_SET to keep the layout list well
// formed (Vulkan requires every slot in the list to be a valid layout
// handle).
static std::shared_ptr<renderer::PipelineLayout> createMeshShaderShadowPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorSetLayout>& mesh_shadow_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& runtime_lights_desc_set_layout) {

    renderer::PushConstantRange pc{};
    pc.stage_flags = SET_FLAG_BIT(ShaderStage, MESH_BIT_EXT);
    pc.offset = 0;
    // Must match the push_constant block in base_depthonly_csm.mesh —
    // mat4 model_mat + 8 uints = 64 + 32 = 96 bytes.
    pc.size = 96;

    renderer::DescriptorSetLayoutList layouts(RUNTIME_LIGHTS_PARAMS_SET + 1);
    layouts[0] = mesh_shadow_desc_set_layout;
    layouts[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set_layout;
    // Fill nulls with empty layouts so Vulkan sees a valid handle at
    // every set index between 0 and RUNTIME_LIGHTS_PARAMS_SET.
    for (size_t i = 0; i < layouts.size(); ++i) {
        if (!layouts[i]) {
            layouts[i] = device->createDescriptorSetLayout({});
        }
    }
    return device->createPipelineLayout(
        layouts, { pc }, std::source_location::current());
}

static std::shared_ptr<renderer::PipelineLayout> createDrawablePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& material_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& skin_desc_set_layout) {
    renderer::PushConstantRange push_const_range{};
    // base.frag reads model_params.debug_force_red (and declares the
    // full ModelParams block to keep layouts identical across stages),
    // so the push-constant range must be visible to both VERTEX and
    // FRAGMENT.  Without the fragment bit the shader would compile
    // but the driver would reject the bind / produce a validation
    // error about a fragment-stage push-constant access outside the
    // declared range.
    push_const_range.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::ModelParams);

    renderer::DescriptorSetLayoutList desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.resize(MAX_NUM_PARAMS_SETS);
    desc_set_layouts[PBR_MATERIAL_PARAMS_SET] = material_desc_set_layout;
    desc_set_layouts[SKIN_PARAMS_SET] = skin_desc_set_layout;

    // Fill any remaining null slots with an empty descriptor set layout so that
    // the Vulkan pipeline layout has valid handles at every index.
    for (size_t i = 0; i < desc_set_layouts.size(); ++i) {
        if (!desc_set_layouts[i]) {
            desc_set_layouts[i] = device->createDescriptorSetLayout({});
        }
    }

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static renderer::ShaderModuleList getDrawableShaderModules(
    std::shared_ptr<renderer::Device> device,
    bool has_normals,
    bool has_tangent,
    bool has_texcoord_0,
    bool has_skin_set_0,
    bool has_material,
    bool has_double_sided,
    bool has_skin_set_1 = false) {
    renderer::ShaderModuleList shader_modules(2);
    auto vert_feature_str = std::string(has_texcoord_0 ? "_TEX" : "") +
        (has_tangent ? "_TN" : (has_normals ? "_N" : ""));
    vert_feature_str = has_material ? vert_feature_str : "_NOMTL";
    auto frag_feature_str = vert_feature_str;
    if (has_skin_set_0) {
        // _SKIN8 = both skin sets bound (8-bone debug); _SKIN = set 0 only.
        vert_feature_str += has_skin_set_1 ? "_SKIN8" : "_SKIN";
    }
    if (has_double_sided) {
        frag_feature_str += "_DS";
    }
    
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "base_vert" + vert_feature_str + ".spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());

    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "base_frag" + frag_feature_str + ".spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

    return shader_modules;
}

static renderer::ShaderModuleList getDrawableDepthonlyShaderModules(
    std::shared_ptr<renderer::Device> device,
    bool has_texcoord_0,
    bool has_skin_set_0,
    bool has_material,
    bool csm_layered = false,
    bool is_opaque = false,
    bool csm_per_cascade = false,
    bool has_skin_set_1 = false) {
    // csm_layered: when true, add the geometry shader that broadcasts
    // each input triangle to all CSM_CASCADE_COUNT depth-array layers
    // in a single pass, eliminating the per-cascade vertex transform.
    //
    // csm_per_cascade: when true, load the _CSMCASC permutation of the
    // VS, which reads light_view_proj[model_params.cascade_idx] from
    // the runtime-lights UBO instead of camera_info.view_proj from the
    // view-camera SSBO.  The host loops cascades and pushes cascade_idx
    // via ModelParams; no GS, no mesh shader.  Mutually exclusive with
    // csm_layered (caller's contract).
    //
    // is_opaque: when true, the fragment shader is OMITTED entirely.
    // base_depthonly.frag's only job is the alpha-mask discard
    // (ALPHAMODE_MASK is hardcoded at the top of the file).  For
    // Opaque materials that test never fires, so binding the frag
    // shader just wastes a per-fragment base-color texture sample ×
    // CSM_CASCADE_COUNT cascades.  Skipping it turns the pipeline into
    // a pure rasterizer-only depth-write, matching the cluster shadow
    // path (no frag stage).  Vulkan permits graphics pipelines with no
    // fragment stage as long as a depth attachment is bound — which
    // the CSM shadow render pass provides.
    //
    // Stage count breakdown (csm_per_cascade has the same stage count
    // as csm_layered=false; it just swaps the VS permutation):
    //          csm_layered  is_opaque   stages           layout
    //          F            F           2  (vert, frag)
    //          F            T           1  (vert)
    //          T            F           3  (vert, geom, frag)
    //          T            T           2  (vert, geom)
    const int stage_count =
        (csm_layered ? 1 : 0) + (is_opaque ? 0 : 1) + 1;
    renderer::ShaderModuleList shader_modules(stage_count);

    auto vert_feature_str = std::string(has_texcoord_0 ? "_TEX" : "");
    vert_feature_str = has_material ? vert_feature_str : "_NOMTL";
    auto frag_feature_str = vert_feature_str;
    if (has_skin_set_0) {
        // _SKIN8 = both skin sets bound (8-bone debug); _SKIN = set 0 only.
        vert_feature_str += has_skin_set_1 ? "_SKIN8" : "_SKIN";
    }
    // CSMCASC suffix only exists for the four permutations the CSM
    // shadow path actually uses (HAS_UV_SET0 / HAS_SKIN_SET_0 matrix,
    // no NOMTL).  See shaders-compile.cfg / CompileShaders.cmake.
    if (csm_per_cascade) {
        vert_feature_str += "_CSMCASC";
    }

    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "base_depthonly_vert" + vert_feature_str + ".spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());

    if (csm_layered) {
        // Geometry shader: amplifies each triangle to CSM_CASCADE_COUNT layers.
        shader_modules[1] =
            renderer::helper::loadShaderModule(
                device,
                "base_depthonly_csm_geom.spv",
                renderer::ShaderStageFlagBits::GEOMETRY_BIT,
                std::source_location::current());
    }

    if (!is_opaque) {
        // Mask / Blend materials: bind the frag shader so the
        // alpha-mask discard runs.  Index = stage_count - 1
        // (last slot) regardless of csm_layered.
        shader_modules[stage_count - 1] =
            renderer::helper::loadShaderModule(
                device,
                "base_depthonly_frag" + frag_feature_str + ".spv",
                renderer::ShaderStageFlagBits::FRAGMENT_BIT,
                std::source_location::current());
    }

    return shader_modules;
}

static std::shared_ptr<renderer::Pipeline> createDrawablePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const ego::PrimitiveInfo& primitive) {
    auto shader_modules = getDrawableShaderModules(
        device,
        primitive.tag_.has_normal,
        primitive.tag_.has_tangent,
        primitive.tag_.has_texcoord_0,
        primitive.tag_.has_skin_set_0,
        primitive.material_idx_ >= 0,
        primitive.tag_.double_sided,
        primitive.tag_.has_skin_set_1);

    renderer::PipelineInputAssemblyStateCreateInfo topology_info;
    topology_info.restart_enable = primitive.tag_.restart_enable;
    topology_info.topology = static_cast<renderer::PrimitiveTopology>(primitive.tag_.topology);

    auto binding_descs = primitive.binding_descs_;
    auto attribute_descs = primitive.attribute_descs_;

    renderer::VertexInputBindingDescription desc;
    desc.binding = VINPUT_INSTANCE_BINDING_POINT;
    desc.input_rate = renderer::VertexInputRate::INSTANCE;
    desc.stride = sizeof(glsl::InstanceDataInfo);
    binding_descs.push_back(desc);

    renderer::VertexInputAttributeDescription attr;
    attr.binding = VINPUT_INSTANCE_BINDING_POINT;
    attr.buffer_offset = 0;
    attr.format = renderer::Format::R32G32B32_SFLOAT;
    attr.buffer_view = 0;
    attr.location = IINPUT_MAT_ROT_0;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_rot_0);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_1;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_rot_1);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_2;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_rot_2);
    attribute_descs.push_back(attr);
    attr.format = renderer::Format::R32G32B32A32_SFLOAT;
    attr.location = IINPUT_MAT_POS_SCALE;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_pos_scale);
    attribute_descs.push_back(attr);

    renderer::RasterizationStateOverride rasterization_state_override;
    rasterization_state_override.override_double_sided = true;
    rasterization_state_override.double_sided = primitive.tag_.double_sided;

    auto drawable_pipeline = device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        topology_info,
        graphic_pipeline_info,
        shader_modules,
        renderbuffer_formats,
        rasterization_state_override,
        std::source_location::current());

    return drawable_pipeline;
}

static std::shared_ptr<renderer::Pipeline> createDrawableShadowPipelineInternal(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const ego::PrimitiveInfo& primitive,
    bool csm_layered,
    bool is_opaque,
    bool csm_per_cascade = false) {
    auto shader_modules = getDrawableDepthonlyShaderModules(
        device,
        primitive.tag_.has_texcoord_0,
        primitive.tag_.has_skin_set_0,
        primitive.material_idx_ >= 0,
        csm_layered,
        is_opaque,
        csm_per_cascade,
        primitive.tag_.has_skin_set_1);

    renderer::PipelineInputAssemblyStateCreateInfo topology_info;
    topology_info.restart_enable = primitive.tag_.restart_enable;
    topology_info.topology = static_cast<renderer::PrimitiveTopology>(primitive.tag_.topology);

    auto binding_descs = primitive.binding_descs_;
    auto attribute_descs = primitive.attribute_descs_;

    // The depth-only shadow shader (base_depthonly.vert) only reads
    // POSITION(0), optional TEXCOORD0(1), and optional JOINTS/WEIGHTS
    // (4,5 / 7,8).  Strip everything else so the validation layer does
    // not warn "Vertex attribute at location N not consumed by vertex shader".
    {
        std::set<uint32_t> kept_bindings;
        std::vector<renderer::VertexInputAttributeDescription> filtered;
        filtered.reserve(attribute_descs.size());
        for (auto& a : attribute_descs) {
            const uint32_t loc = a.location;
            const bool keep =
                (loc == VINPUT_POSITION) ||
                (loc == VINPUT_TEXCOORD0 && primitive.tag_.has_texcoord_0) ||
                (loc == VINPUT_JOINTS_0  && primitive.tag_.has_skin_set_0) ||
                (loc == VINPUT_WEIGHTS_0 && primitive.tag_.has_skin_set_0) ||
                (loc == VINPUT_JOINTS_1  && primitive.tag_.has_skin_set_1) ||
                (loc == VINPUT_WEIGHTS_1 && primitive.tag_.has_skin_set_1);
            if (keep) {
                filtered.push_back(a);
                kept_bindings.insert(a.binding);
            }
        }
        attribute_descs = std::move(filtered);

        std::vector<renderer::VertexInputBindingDescription> kept_binding_descs;
        kept_binding_descs.reserve(binding_descs.size());
        for (auto& b : binding_descs) {
            if (kept_bindings.count(b.binding)) kept_binding_descs.push_back(b);
        }
        binding_descs = std::move(kept_binding_descs);
    }

    renderer::VertexInputBindingDescription desc;
    desc.binding = VINPUT_INSTANCE_BINDING_POINT;
    desc.input_rate = renderer::VertexInputRate::INSTANCE;
    desc.stride = sizeof(glsl::InstanceDataInfo);
    binding_descs.push_back(desc);

    renderer::VertexInputAttributeDescription attr;
    attr.binding = VINPUT_INSTANCE_BINDING_POINT;
    attr.buffer_offset = 0;
    attr.format = renderer::Format::R32G32B32_SFLOAT;
    attr.buffer_view = 0;
    attr.location = IINPUT_MAT_ROT_0;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_rot_0);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_1;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_rot_1);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_2;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_rot_2);
    attribute_descs.push_back(attr);
    attr.format = renderer::Format::R32G32B32A32_SFLOAT;
    attr.location = IINPUT_MAT_POS_SCALE;
    attr.offset = offsetof(glsl::InstanceDataInfo, mat_pos_scale);
    attribute_descs.push_back(attr);
    renderer::RasterizationStateOverride rasterization_state_override;
    rasterization_state_override.override_depth_clamp_enable = true;
    rasterization_state_override.depth_clamp_enable = true;
    // ── Sidedness for the SHADOW pass ──────────────────────────────
    // Always respect the asset's authored double_sided flag — no more
    // is_opaque-vs-mask defensive override.  Asset bugs (single-sided
    // thin geometry not flagged double-sided) are the asset's problem;
    // the engine trusts the flag.  This matches the forward pass and
    // halves rasterised triangles for closed-solid materials.
    rasterization_state_override.override_double_sided = true;
    rasterization_state_override.double_sided = primitive.tag_.double_sided;
    auto drawable_pipeline = device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        topology_info,
        graphic_pipeline_info,
        shader_modules,
        renderbuffer_formats,
        rasterization_state_override,
        std::source_location::current());

    return drawable_pipeline;
}

// Single-cascade (per-layer) shadow pipeline — used as fallback / debug.
//   is_opaque = true → no fragment shader + respect authored double_sided.
//   is_opaque = false → full path: frag shader (alpha-mask discard) + force
//                       double_sided=true (cutout foliage safety).
static std::shared_ptr<renderer::Pipeline> createDrawableShadowPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const ego::PrimitiveInfo& primitive,
    bool is_opaque) {
    return createDrawableShadowPipelineInternal(
        device, renderbuffer_formats, pipeline_layout,
        graphic_pipeline_info, primitive, /*csm_layered*/ false, is_opaque);
}

// All-cascade (layered GS) shadow pipeline — used for the single-pass CSM path.
// is_opaque semantics match createDrawableShadowPipeline above.
static std::shared_ptr<renderer::Pipeline> createDrawableCsmLayeredPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const ego::PrimitiveInfo& primitive,
    bool is_opaque) {
    return createDrawableShadowPipelineInternal(
        device, renderbuffer_formats, pipeline_layout,
        graphic_pipeline_info, primitive, /*csm_layered*/ true, is_opaque);
}

// CSM per-cascade shadow pipeline — used for DrawMode::kCsmPerCascade,
// the "Regular" option on the shadow draw-mode menu.  Like
// createDrawableShadowPipeline (csm_layered=false, no GS), but the
// vertex shader is the _CSMCASC permutation that reads
// light_view_proj[model_params.cascade_idx] from the runtime-lights
// UBO — letting the host loop cascades and push cascade_idx per draw
// without rebinding view-camera descriptors.  is_opaque semantics
// match the other two helpers.
static std::shared_ptr<renderer::Pipeline> createDrawableCsmPerCascadePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const ego::PrimitiveInfo& primitive,
    bool is_opaque) {
    return createDrawableShadowPipelineInternal(
        device, renderbuffer_formats, pipeline_layout,
        graphic_pipeline_info, primitive, /*csm_layered*/ false, is_opaque,
        /*csm_per_cascade*/ true);
}

// CSM mesh-shader shadow pipeline — used for DrawMode::kCsmMeshShader
// (the "Mesh Shader" option on the shadow draw-mode menu).  Loads
// base_depthonly_csm.task + .mesh; no VS / GS / FS.  No vertex input
// state: the mesh shader fetches all data via the per-primitive SSBO
// descriptor set bound at set 0.  Uses the mesh-shader-shadow pipeline
// layout (NOT drawable_pipeline_layout_) so its set assignments don't
// collide with the forward / GS / per-cascade pipelines.
//
// Caller responsibility: only invoke for primitives that meet the
// mesh-shader eligibility criteria (see buildMeshShaderShadowResources).
// Ineligible primitives produce clamped / wrong output if dispatched
// through this pipeline.
static std::shared_ptr<renderer::Pipeline> createDrawableCsmMeshShaderPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const ego::PrimitiveInfo& primitive) {

    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] = renderer::helper::loadShaderModule(
        device,
        "base_depthonly_csm_task.spv",
        renderer::ShaderStageFlagBits::TASK_BIT_EXT,
        std::source_location::current());
    shader_modules[1] = renderer::helper::loadShaderModule(
        device,
        "base_depthonly_csm_mesh.spv",
        renderer::ShaderStageFlagBits::MESH_BIT_EXT,
        std::source_location::current());

    // Mesh-shader pipelines don't use the input assembler.  Empty
    // binding / attribute lists.
    std::vector<renderer::VertexInputBindingDescription> binding_descs;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs;

    renderer::PipelineInputAssemblyStateCreateInfo topology_info;
    topology_info.restart_enable = primitive.tag_.restart_enable;
    topology_info.topology =
        static_cast<renderer::PrimitiveTopology>(primitive.tag_.topology);

    renderer::RasterizationStateOverride raster;
    // The eligibility gate in buildMeshShaderShadowResources only lets
    // opaque primitives through, so we can respect the asset's authored
    // double_sided flag here (closed solids cull back faces, foliage-
    // class assets keep both sides) — matching the GS / per-cascade
    // opaque rule.
    raster.override_double_sided = true;
    raster.double_sided = primitive.tag_.double_sided;
    raster.override_depth_clamp_enable = true;
    raster.depth_clamp_enable = true;

    return device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        topology_info,
        graphic_pipeline_info,
        shader_modules,
        renderbuffer_formats,
        raster,
        std::source_location::current());
}

// ─── Mesh-shader CSM resources — per-primitive build ────────────────────
// For each primitive in the drawable, decide whether it's eligible for
// the mesh-shader CSM path and, if so:
//   • derive vb_stride_floats / vb_position_offset_floats from
//     attribute_descs_ + matching binding_descs_;
//   • derive ib_first_index from index_desc_[0] + the buffer view's
//     own offset (UINT32 indices only — UINT16 falls back to GS);
//   • derive vertex_count from the position buffer view's range;
//   • derive tri_count from index_count / 3;
//   • build a hash-cached mesh-shader pipeline;
//   • allocate a 3-binding descriptor set (VB, IB, instance) and write
//     the primitive's actual buffers into it.
//
// Eligibility gate (anything FAILS → leave mesh_shader_shadow_desc_set_
// null → drawMesh falls back to drawIndexedIndirect on the GS path for
// this primitive):
//   - opaque (no alpha-cutout discard needed)
//   - non-skinned (no joint matrices)
//   - has a POSITION attribute
//   - UINT32 indices
//   - vertex_count <= 256 AND tri_count <= 256
//   - position buffer view has a usable stride (>0)
//
// Skinning and cutout fall back to the GS path; extend the shader and
// the eligibility gate together if those need real mesh-shader support.
static void buildMeshShaderShadowResources(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& mesh_shadow_desc_set_layout,
    const std::shared_ptr<renderer::PipelineLayout>& mesh_shadow_pipeline_layout,
    const renderer::PipelineRenderbufferFormats& shadow_rb_formats,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<ego::DrawableData>& drawable,
    std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>>& pipeline_list) {

    if (!drawable || !mesh_shadow_desc_set_layout ||
        !mesh_shadow_pipeline_layout ||
        !drawable->instance_buffer_.buffer) {
        return;
    }

    for (auto& mesh : drawable->meshes_) {
        for (auto& prim : mesh.primitives_) {
            // ── Eligibility checks ────────────────────────────────────
            const bool is_opaque = isPrimitiveOpaque(prim, drawable->materials_);
            if (!is_opaque) continue;
            if (prim.tag_.has_skin_set_0) continue;
            if (prim.index_desc_.empty()) continue;
            const auto& idesc = prim.index_desc_[0];
            if (idesc.index_type != renderer::IndexType::UINT32) continue;
            if (idesc.index_count == 0 || idesc.index_count % 3 != 0) continue;
            const uint32_t tri_count = uint32_t(idesc.index_count / 3);
            if (tri_count > 256u) continue;

            // Find POSITION attribute.
            const renderer::VertexInputAttributeDescription* pos_attr = nullptr;
            for (auto& a : prim.attribute_descs_) {
                if (a.location == VINPUT_POSITION) { pos_attr = &a; break; }
            }
            if (!pos_attr) continue;

            // Find the binding descriptor matching the POSITION attr's binding.
            const renderer::VertexInputBindingDescription* pos_binding = nullptr;
            for (auto& b : prim.binding_descs_) {
                if (b.binding == pos_attr->binding) { pos_binding = &b; break; }
            }
            if (!pos_binding || pos_binding->stride == 0) continue;
            const uint32_t vb_stride_floats = uint32_t(pos_binding->stride / 4u);
            if (vb_stride_floats * 4u != pos_binding->stride) continue; // not float-aligned

            // VB buffer view (for position).
            if (pos_attr->buffer_view >= drawable->buffer_views_.size()) continue;
            const auto& vb_bv = drawable->buffer_views_[pos_attr->buffer_view];
            if (vb_bv.buffer_idx >= drawable->buffers_.size()) continue;
            auto vb_buffer = drawable->buffers_[vb_bv.buffer_idx].buffer;
            if (!vb_buffer) continue;
            // vertex_count from the buffer view's range; if range is 0 we
            // cannot bound the vertex emit loop safely, so skip.
            if (vb_bv.range == 0) continue;
            const uint32_t vb_view_stride =
                (vb_bv.stride > 0)
                    ? uint32_t(vb_bv.stride)
                    : uint32_t(pos_binding->stride);
            if (vb_view_stride == 0) continue;
            const uint32_t vertex_count =
                uint32_t(vb_bv.range / vb_view_stride);
            if (vertex_count == 0 || vertex_count > 256u) continue;

            // IB buffer view.
            if (idesc.buffer_view >= drawable->buffer_views_.size()) continue;
            const auto& ib_bv = drawable->buffer_views_[idesc.buffer_view];
            if (ib_bv.buffer_idx >= drawable->buffers_.size()) continue;
            auto ib_buffer = drawable->buffers_[ib_bv.buffer_idx].buffer;
            if (!ib_buffer) continue;
            // ib_first_index in uint32 strides; sums view + per-LOD byte
            // offsets, divided by 4 (sizeof uint32).
            const uint64_t ib_byte_off = ib_bv.offset + idesc.offset;
            if (ib_byte_off % 4u != 0) continue;
            const uint32_t ib_first_index = uint32_t(ib_byte_off / 4u);

            // Layout offsets (in FLOATS) for the position read inside
            // the shader.  attribute.buffer_offset is the array's start
            // in the buffer; attribute.offset is the offset within each
            // vertex (e.g. 0 for position-first interleaved).
            const uint64_t vb_byte_off = pos_attr->buffer_offset + pos_attr->offset;
            if (vb_byte_off % 4u != 0) continue;
            const uint32_t vb_position_offset_floats = uint32_t(vb_byte_off / 4u);

            // ── All checks passed — build pipeline (hash-cached) ─────
            const size_t hash_value =
                getDepthonlyHashForMaterial(prim, drawable->materials_);
            auto pl_it = pipeline_list.find(hash_value);
            if (pl_it == pipeline_list.end()) {
                pipeline_list[hash_value] =
                    createDrawableCsmMeshShaderPipeline(
                        device,
                        shadow_rb_formats,
                        mesh_shadow_pipeline_layout,
                        graphic_pipeline_info,
                        prim);
            }

            // ── Allocate per-primitive descriptor set + write bindings ─
            prim.mesh_shader_shadow_desc_set_ =
                device->createDescriptorSets(
                    descriptor_pool, mesh_shadow_desc_set_layout, 1)[0];

            renderer::WriteDescriptorList writes;
            renderer::Helper::addOneBuffer(
                writes,
                prim.mesh_shader_shadow_desc_set_,
                renderer::DescriptorType::STORAGE_BUFFER,
                /*binding*/ 0,
                vb_buffer,
                static_cast<uint32_t>(vb_buffer->getSize()));
            renderer::Helper::addOneBuffer(
                writes,
                prim.mesh_shader_shadow_desc_set_,
                renderer::DescriptorType::STORAGE_BUFFER,
                /*binding*/ 1,
                ib_buffer,
                static_cast<uint32_t>(ib_buffer->getSize()));
            renderer::Helper::addOneBuffer(
                writes,
                prim.mesh_shader_shadow_desc_set_,
                renderer::DescriptorType::STORAGE_BUFFER,
                /*binding*/ 2,
                drawable->instance_buffer_.buffer,
                static_cast<uint32_t>(
                    drawable->instance_buffer_.buffer->getSize()));
            device->updateDescriptorSets(writes);

            // Cache per-primitive layout descriptors for the push constant.
            prim.mesh_shader_vb_stride_floats_          = vb_stride_floats;
            prim.mesh_shader_vb_position_offset_floats_ = vb_position_offset_floats;
            prim.mesh_shader_ib_first_index_            = ib_first_index;
            prim.mesh_shader_vertex_count_              = vertex_count;
            prim.mesh_shader_tri_count_                 = tri_count;
        }
    }
}

// (Helpers `isPrimitiveOpaque` and `getDepthonlyHashForMaterial` are
// defined above drawMesh — they are needed by the draw-time hash lookup
// at line ~2360, which sits before this section.  See the comment block
// at that definition for details.)

renderer::WriteDescriptorList addDrawableIndirectDrawBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::BufferInfo& buffer) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_BUFFER,
        INDIRECT_DRAW_BUFFER_INDEX,
        buffer.buffer,
        buffer.buffer->getSize());

    return descriptor_writes;
}

renderer::WriteDescriptorList addUpdateInstanceBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::BufferInfo& game_objects_buffer,
    const renderer::BufferInfo& instance_buffer) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        GAME_OBJECTS_BUFFER_INDEX,
        game_objects_buffer.buffer,
        game_objects_buffer.buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        INSTANCE_BUFFER_INDEX,
        instance_buffer.buffer,
        instance_buffer.buffer->getSize());

    return descriptor_writes;
}

renderer::WriteDescriptorList addGameObjectsInfoBuffer(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::BufferInfo& game_object_buffer,
//    const renderer::BufferInfo& game_camera_buffer,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(6);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        GAME_OBJECTS_BUFFER_INDEX,
        game_object_buffer.buffer,
        game_object_buffer.buffer->getSize());

    // Always write a buffer at CAMERA_OBJECT_BUFFER_INDEX — the shader
    // (update_game_objects.comp) declares CameraInfoBuffer at this slot
    // and validation will fire VUID-vkCmdDispatch-None-08114 ("descriptor
    // … never updated") if the slot is left unwritten, regardless of
    // whether the dispatch actually reads from it at runtime.  Prefer
    // the app-supplied view-camera buffer (set via
    // DrawableObject::setViewCameraBufferForUpdate); if the application
    // hasn't called the setter yet (e.g. initStaticMembers runs before
    // the camera buffer is wired up), fall back to game_object_buffer
    // itself — wrong contents semantically, but it's a valid bound
    // STORAGE_BUFFER so validation is satisfied.  Once the application
    // calls updateGameObjectsCameraBuffer with the real camera buffer,
    // the binding is overwritten with the correct one.
    //
    // Read through the public getter — this function lives in an
    // anonymous namespace inside `engine::` (NOT `engine::game_object::`)
    // so it must qualify DrawableObject with its full namespace path to
    // reach the static method.
    const auto& view_cam_buf =
        engine::game_object::DrawableObject::getViewCameraBufferForUpdate();
    const auto& camera_slot_buffer =
        view_cam_buf ? view_cam_buf->buffer : game_object_buffer.buffer;
    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        CAMERA_OBJECT_BUFFER_INDEX,
        camera_slot_buffer,
        camera_slot_buffer->getSize());

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        texture_sampler,
        soil_water_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        WATER_FLOW_BUFFER_INDEX,
        texture_sampler,
        water_flow.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_AIRFLOW_INDEX,
        texture_sampler,
        airflow_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static std::shared_ptr<renderer::DescriptorSetLayout> createDrawableIndirectDrawDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        INDIRECT_DRAW_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> createUpdateGameObjectsDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(6);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        GAME_OBJECTS_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        CAMERA_OBJECT_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[2] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[3] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[4] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        WATER_FLOW_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[5] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_AIRFLOW_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> createInstanceBufferDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(2);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        GAME_OBJECTS_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        INSTANCE_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createDrawableIndirectDrawPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(uint32_t);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static std::shared_ptr<renderer::PipelineLayout> createGameObjectsPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::GameObjectsUpdateParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static std::shared_ptr<renderer::PipelineLayout> createInstanceBufferPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::InstanceBufferUpdateParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

} // namespace

namespace game_object {

// static member definition.
uint32_t DrawableObject::max_alloc_game_objects_in_buffer = kNumDrawableInstance;

// Engine-wide material classification counters — declared on
// DrawableObject so VirtualTextureManager can pull them for the
// per-frame vt_pool.log line.  Updated by
// computeEffectiveOpaqueForMaterials at every mesh load.
std::atomic<int> DrawableObject::s_total_materials_count_{0};
std::atomic<int> DrawableObject::s_alpha_cutoff_materials_count_{0};

std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::material_desc_set_layout_;
std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::skin_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DrawableObject::drawable_pipeline_layout_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> DrawableObject::drawable_pipeline_list_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> DrawableObject::drawable_shadow_pipeline_list_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> DrawableObject::drawable_csm_layered_pipeline_list_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> DrawableObject::drawable_csm_per_cascade_pipeline_list_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> DrawableObject::drawable_csm_mesh_shader_pipeline_list_;
std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::mesh_shader_shadow_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout>      DrawableObject::mesh_shader_shadow_pipeline_layout_;
std::unordered_map<std::string, std::shared_ptr<DrawableData>> DrawableObject::drawable_object_list_;
std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::drawable_indirect_draw_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DrawableObject::drawable_indirect_draw_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DrawableObject::drawable_indirect_draw_pipeline_;
std::shared_ptr<renderer::DescriptorSet> DrawableObject::update_game_objects_buffer_desc_set_[2];
std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::update_game_objects_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DrawableObject::update_game_objects_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DrawableObject::update_game_objects_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::update_instance_buffer_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DrawableObject::update_instance_buffer_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DrawableObject::update_instance_buffer_pipeline_;
std::shared_ptr<renderer::BufferInfo> DrawableObject::game_objects_buffer_;
std::shared_ptr<renderer::BufferInfo>
    DrawableObject::s_view_camera_buffer_for_update_;

void PrimitiveInfo::generateHash() {
    hash_ = std::hash<uint32_t>{}(tag_.data);
    hash_combine(hash_, material_idx_ >= 0 ? 0x0 : 0xffffffff);
    for (auto& item : binding_descs_) {
        uint64_t tmp_value = item.binding;
        tmp_value = (tmp_value << 32) | item.stride | (uint64_t(item.input_rate) << 31);
        hash_combine(hash_, tmp_value);
    }
    for (auto& item : attribute_descs_) {
        uint64_t tmp_value = item.binding;
        tmp_value = (tmp_value << 32) | uint64_t(item.format);
        hash_combine(hash_, tmp_value);
        hash_combine(hash_, item.location);
        hash_combine(hash_, item.offset);
    }

    auto depthonly_tag = tag_;
    depthonly_tag.has_normal = 0;
    depthonly_tag.has_tangent = 0;

    depthonly_hash_ = std::hash<uint32_t>{}(depthonly_tag.data);
    hash_combine(depthonly_hash_, material_idx_ >= 0 ? 0x0 : 0xffffffff);

    for (auto& item : binding_descs_) {
        uint64_t tmp_value = item.binding;
        tmp_value = (tmp_value << 32) | item.stride | (uint64_t(item.input_rate) << 31);
        hash_combine(depthonly_hash_, tmp_value);
    }
    for (auto& item : attribute_descs_) {
        if (item.location != VINPUT_NORMAL &&
            item.location != VINPUT_TANGENT &&
            item.location != VINPUT_COLOR &&
            item.location != VINPUT_TEXCOORD1) {
            uint64_t tmp_value = item.binding;
            tmp_value = (tmp_value << 32) | uint64_t(item.format);
            hash_combine(depthonly_hash_, tmp_value);
            hash_combine(depthonly_hash_, item.location);
            hash_combine(depthonly_hash_, item.offset);
        }
    }
}

void DrawableData::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    for (auto& texture : textures_) {
        if (!texture.borrowed_) texture.destroy(device);  // cache owns borrowed
    }

    for (auto& material : materials_) {
        material.uniform_buffer_.destroy(device);
    }

    for (auto& buffer : buffers_) {
        buffer.destroy(device);
    }

    for (int i_skin = 0; i_skin < skins_.size(); i_skin++) {
        skins_[i_skin].joints_buffer_.destroy(device);
    }

    indirect_draw_cmd_.destroy(device);
    instance_buffer_.destroy(device);
}

struct compare {
    bool operator()(const std::pair<int, glm::vec4>& value,
        const int& key)
    {
        return (value.first < key);
    }
    bool operator()(const int& key,
        const std::pair<int, glm::vec4> & value)
    {
        return (key < value.first);
    }
};

void AnimChannelInfo::update(DrawableData* object, float time, float time_scale/* = 1.0f*/, bool repeat/* = true */ ) {
    float scaled_time = time / time_scale;
    auto& last_one = samples_.back();
    
    float play_time = scaled_time;
    if (repeat && scaled_time > last_one.first) {
        play_time = glm::fract(scaled_time / last_one.first)* last_one.first;
    }

    auto result = std::lower_bound(samples_.begin(), samples_.end(), play_time,
        [](const std::pair<float, glm::vec4>& info, float value) {
            return info.first < value; });

    auto lower = result == samples_.begin() ? result : (result - 1);
    auto upper = result == samples_.end() ? (samples_.end() - 1) : result;

    auto step = upper->first - lower->first;

    float ratio = step == 0.0f ? 0.0f : glm::clamp((play_time - lower->first) / step, 0.0f, 1.0f);

    auto& target_node = object->nodes_[node_idx_];
    if (type_ == kTranslation)
    {
        target_node.translation_ = glm::mix(lower->second, upper->second, ratio);
    }
    else if (type_ == kRotation)
    {
        auto q1 = glm::quat(lower->second.w, lower->second.x, lower->second.y, lower->second.z);
        auto q2 = glm::quat(upper->second.w, upper->second.x, upper->second.y, upper->second.z);
        target_node.rotation_ = glm::normalize(glm::slerp(q1, q2, ratio));
    }
    else if (type_ == kScale)
    {
        target_node.scale_ = glm::mix(lower->second, upper->second, ratio);
    }
}

glm::mat4 NodeInfo::getLocalMatrix(
    bool use_local_matrix_only) {
    auto joint_mat =
        glm::translate(glm::mat4(1.0f), translation_) *
        glm::mat4(rotation_) *
        glm::scale(glm::mat4(1.0f), scale_);

    if (use_local_matrix_only)
        return matrix_;
    else 
        return joint_mat * matrix_;
}

glm::mat4 DrawableData::getNodeMatrix(
    const int32_t& node_idx,
    bool use_local_matrix_only) {
    if (node_idx < 0)
        return glm::mat4(1.0f);

    auto& node = nodes_[node_idx];
    glm::mat4 node_matrix =
        node.getLocalMatrix(use_local_matrix_only);
    auto parent_idx = node.parent_idx_;
    while (parent_idx >= 0) {
        auto& parent_node = nodes_[parent_idx];
        node_matrix =
            parent_node.getLocalMatrix(use_local_matrix_only) * node_matrix;
        parent_idx = parent_node.parent_idx_;
    }

    return node_matrix;
}

void DrawableData::updateJoints(
    const std::shared_ptr<renderer::Device>& device,
    int32_t node_idx) {
    auto& node = nodes_[node_idx];
    if (node.skin_idx_ > -1)
    {
        // Update the joint matrices
        auto inverse_transform = glm::inverse(node.getCachedMatrix());
        auto& skin = skins_[node.skin_idx_];
        auto num_joints = skin.joints_.size();
        std::vector<glm::mat4> joint_matrices(num_joints);
        for (size_t i = 0; i < num_joints; i++) {
            joint_matrices[i] =
                inverse_transform *
                nodes_[skin.joints_[i]].getCachedMatrix() *
                skin.inverse_bind_matrices_[i];
        }

        // Keep a CPU copy for the RT-shadow skeleton path (CPU-skins the
        // character into world space each frame from these).
        skin.joint_matrices_cpu_ = joint_matrices;

        renderer::Helper::updateBufferWithSrcData(
            device,
            joint_matrices.size() * sizeof(glm::mat4),
            joint_matrices.data(),
            skin.joints_buffer_.memory);
    }

    for (auto& child : node.child_idx_) {
        updateJoints(device, child);
    }
}

void DrawableData::update(
    const std::shared_ptr<renderer::Device>& device,
    const uint32_t& active_anim_idx,
    const float& time,
    bool use_local_matrix_only,
    bool skip_animations) {
    // ── Shared-data dedup ────────────────────────────────────────────
    // Many wrappers can share ONE DrawableData (every placed sub-object of
    // a group does).  Animations, the full node-hierarchy matrix refresh
    // and joint uploads only need to run ONCE per frame — without this
    // gate, placing N objects from a big FBX cost N × hierarchy-walk on
    // the CPU every frame.
    if (time == m_last_update_time_) return;
    m_last_update_time_ = time;
    // ── Animation channels ───────────────────────────────────────────
    // Skipped when the caller owns the node transforms (e.g. the
    // PlayerController-driven character).  Without this gate, an
    // animated rig like CesiumMan.gltf will replay its imported walk
    // animation every frame and CLOBBER whatever node values
    // setRootNodeTransform / setNodeRotationByName just wrote — making
    // the player render at the animation's authored origin instead of
    // its controller-driven spawn position, AND replaying the imported
    // limb pose on top of the controller's procedural arms/legs pose.
    if (!skip_animations && animations_.size() > 0) {
        assert(active_anim_idx < animations_.size());
        auto& anim = animations_[active_anim_idx];
        for (auto& channel : anim.channels_) {
            channel->update(this, time);
        }
    }

    // update hierarchy matrix
    for (auto i_node = 0; i_node < nodes_.size(); i_node++) {
        nodes_[i_node].cached_matrix_ =
            getNodeMatrix(i_node, use_local_matrix_only);
    }

    // update joints — gated on skins_ (not animations_) so procedurally-
    // posed rigs like scene-skinned.gltf still get their GPU joint
    // matrices uploaded each frame.
    if (skins_.size() > 0) {
        for (auto& scene : scenes_) {
            for (auto& node : scene.nodes_) {
                updateJoints(device, node);
            }
        }
    }
}

DrawableObject::DrawableObject(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::PipelineRenderbufferFormats* renderbuffer_formats,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex,
    const std::string& file_name,
    glm::mat4 location/* = glm::mat4(1.0f)*/)
    : location_(location){

    // Create a path object
    std::filesystem::path file_path(file_name);

    // Get the file extension
    auto extension = file_path.extension().string();

    auto result = drawable_object_list_.find(file_name);
    if (result == drawable_object_list_.end()) {
        if (extension == ".fbx") {
            object_ = loadFbxModel(device, file_name);
        }
        else if (extension == ".gltf" || extension == ".glb") {
            object_ = loadGltfModel(device, file_name);
        }

        // Texture-content-aware opaque classification + alpha-only
        // companion texture extraction.  Walks every loaded material,
        // scans its albedo texture's α channel, and:
        //   • sets effective_opaque_ (drives the no-fragment-shader
        //     shadow pipeline shape).
        //   • for Mask-with-real-cutout materials, allocates an
        //     R8_UNORM companion texture holding just the α channel
        //     on TextureInfo::alpha_only_*.  That companion is what
        //     the shadow / depth-only fragment shader will sample
        //     once Phase 2 wires it through the descriptor binding.
        //
        // Covers FBX too (the loader populates TextureInfo.cpu_pixels
        // for any path that runs createTextureImage / create2DTextureImage
        // via the file-load helper) — when cpu_pixels happens to be
        // missing, the helper degrades gracefully to the conservative
        // (Mask-kept) classification rather than crashing, and the
        // alpha companion is simply not built for that texture.
        computeEffectiveOpaqueForMaterials(device, object_);

        updateDescriptorSets(
            device,
            descriptor_pool,
            material_desc_set_layout_,
            skin_desc_set_layout_,
            object_,
            texture_sampler,
            thin_film_lut_tex);

        for (int i_mesh = 0; i_mesh < object_->meshes_.size(); i_mesh++) {
            for (int i_prim = 0; i_prim < object_->meshes_[i_mesh].primitives_.size(); i_prim++) {
                const auto& primitive = object_->meshes_[i_mesh].primitives_[i_prim];
                {
                    auto hash_value = primitive.getHash();
                    auto result = drawable_pipeline_list_.find(hash_value);
                    if (result == drawable_pipeline_list_.end()) {
                        drawable_pipeline_list_[hash_value] =
                            createDrawablePipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kForward)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive);
                    }
                }

                {
                    // Hash distinguishes opaque vs masked variants of
                    // the same primitive layout; is_opaque selects the
                    // no-frag-shader pipeline shape.
                    const bool is_opaque =
                        isPrimitiveOpaque(primitive, object_->materials_);
                    auto hash_value =
                        getDepthonlyHashForMaterial(
                            primitive, object_->materials_);
                    auto result = drawable_shadow_pipeline_list_.find(hash_value);
                    if (result == drawable_shadow_pipeline_list_.end()) {
                        drawable_shadow_pipeline_list_[hash_value] =
                            createDrawableShadowPipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive,
                                is_opaque);
                    }
                }

                {
                    // CSM layered pipeline: same as shadow but with GS for
                    // single-pass all-cascade rendering.
                    const bool is_opaque =
                        isPrimitiveOpaque(primitive, object_->materials_);
                    auto hash_value =
                        getDepthonlyHashForMaterial(
                            primitive, object_->materials_);
                    auto result = drawable_csm_layered_pipeline_list_.find(hash_value);
                    if (result == drawable_csm_layered_pipeline_list_.end()) {
                        drawable_csm_layered_pipeline_list_[hash_value] =
                            createDrawableCsmLayeredPipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive,
                                is_opaque);
                    }
                }

                {
                    // CSM per-cascade pipeline: depth-only VS (no GS) that
                    // reads light_view_proj[model_params.cascade_idx].
                    // Backs DrawMode::kCsmPerCascade — the "Regular"
                    // option on the shadow draw-mode menu.
                    const bool is_opaque =
                        isPrimitiveOpaque(primitive, object_->materials_);
                    auto hash_value =
                        getDepthonlyHashForMaterial(
                            primitive, object_->materials_);
                    auto result = drawable_csm_per_cascade_pipeline_list_.find(hash_value);
                    if (result == drawable_csm_per_cascade_pipeline_list_.end()) {
                        drawable_csm_per_cascade_pipeline_list_[hash_value] =
                            createDrawableCsmPerCascadePipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive,
                                is_opaque);
                    }
                }
            }
        }

        device->createBuffer(
            kNumDrawableInstance * sizeof(glsl::InstanceDataInfo),
            SET_2_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT, STORAGE_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            0,
            object_->instance_buffer_.buffer,
            object_->instance_buffer_.memory,
            std::source_location::current());

        object_->generateSharedDescriptorSet(
            device,
            descriptor_pool,
            drawable_indirect_draw_desc_set_layout_,
            update_instance_buffer_desc_set_layout_,
            game_objects_buffer_);

        // ── Mesh-shader CSM resources ─────────────────────────────────
        // Builds the hash-cached mesh-shader pipeline + per-primitive
        // descriptor sets for eligible primitives (opaque, non-skinned,
        // UINT32 indices, <=256 verts/tris).  Must run AFTER the
        // instance_buffer_ createBuffer above — buildMeshShaderShadow-
        // Resources writes instance_buffer_.buffer into binding 2 of
        // each per-primitive descriptor set.  Ineligible primitives
        // leave mesh_shader_shadow_desc_set_ null and dispatch falls
        // back to the GS path inside drawMesh.
        buildMeshShaderShadowResources(
            device,
            descriptor_pool,
            mesh_shader_shadow_desc_set_layout_,
            mesh_shader_shadow_pipeline_layout_,
            renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
            graphic_pipeline_info,
            object_,
            drawable_csm_mesh_shader_pipeline_list_);

        drawable_object_list_[file_name] = object_;

        // Sync path: everything is set up by the time we return, so
        // isReady() should report true immediately. Async path flips
        // this same flag at the end of phase 3 (see createAsync).
        object_->ready_.store(true, std::memory_order_release);
    }
    else {
        object_ = result->second;
    }
}

std::shared_ptr<DrawableObject> DrawableObject::createAsync(
    MeshLoadTaskManager& task_manager,
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::PipelineRenderbufferFormats* renderbuffer_formats,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex,
    const std::string& file_name,
    glm::mat4 location/* = glm::mat4(1.0f)*/) {

    // Shell object the caller can push into draw lists immediately.
    // object_ stays null (isReady() == false) until phase 3 finalizes.
    auto obj = std::shared_ptr<DrawableObject>(new DrawableObject(location));

    // Cache short-circuit: if this file has already been loaded,
    // reuse the finalized DrawableData directly — no async work needed.
    // drawable_object_list_ is only written on the main thread (from
    // the sync ctor's tail or phase 3 below), so this lookup is safe
    // without locking.
    auto cached = drawable_object_list_.find(file_name);
    if (cached != drawable_object_list_.end()) {
        obj->object_ = cached->second;
        // The cached DrawableData's ready_ is already set (either by
        // sync ctor or a previous phase 3), so isReady() returns true.
        return obj;
    }

    // ── In-flight load dedup ────────────────────────────────────────
    // The cache above is populated at the END of phase 3.  At startup,
    // multiple callers requesting the SAME asset (e.g. the 6 debug_cube
    // markers) all fire createAsync before phase 3 runs, so each one
    // misses the cache and kicks off its own file parse + GPU upload.
    // Track in-flight loads by filename: the first caller submits the
    // MeshLoadTask, every subsequent caller for the same filename
    // attaches as a "waiter" and gets its object_ populated by the SAME
    // phase 3 that finishes the first load.  Function-local static
    // because both this site and the phase 3 lambda below (same TU,
    // same enclosing function) need access; both run on the main
    // thread, no mutex needed.
    static std::unordered_map<
        std::string, std::vector<std::shared_ptr<DrawableObject>>>
        s_in_flight_loads_;
    auto in_flight_it = s_in_flight_loads_.find(file_name);
    if (in_flight_it != s_in_flight_loads_.end()) {
        in_flight_it->second.push_back(obj);
        // No task submission: phase 3 of the FIRST submitted load will
        // populate obj->object_ when it finishes.
        return obj;
    }
    s_in_flight_loads_[file_name].push_back(obj);

    // Shared mailbox: phase 2 (worker thread) writes `data`; phase 3
    // (main thread, after fence signals) reads it. The MeshLoadTaskManager
    // runs phase3 only after phase2's command buffer fence signals, so
    // there is a natural happens-before between the two lambdas.
    struct LoadState {
        std::shared_ptr<DrawableData> data;
    };
    auto state = std::make_shared<LoadState>();

    const std::string ext =
        std::filesystem::path(file_name).extension().string();

    MeshLoadTask::Phase2Fn phase2_fn =
        [device, file_name, ext, state](
            const std::shared_ptr<renderer::Device>& /*dev*/,
            const std::shared_ptr<renderer::CommandBuffer>& /*cmd_buf*/,
            std::string& err_out) -> bool {
            // NOTE: we deliberately do not record into the provided
            // cmd_buf. loadGltfModel / loadFbxModel perform their GPU
            // uploads via renderer::Helper::createBuffer, which goes
            // through the thread-routed transient channel
            // (VulkanDevice::setupTransientCommandBuffer dispatched
            // by loader_thread_id_). That keeps the existing helper
            // code unmodified. The outer cmd_buf is submitted empty,
            // and its fence signals near-instantly, which is what
            // drives phase 3 on the next main-thread poll().
            try {
                if (ext == ".fbx") {
                    state->data = loadFbxModel(device, file_name);
                } else if (ext == ".gltf" || ext == ".glb") {
                    state->data = loadGltfModel(device, file_name);
                } else if (ext == ".rwobj") {
                    // Native render-ready asset: small per-object load
                    // (baked .rwgeo + .rwtex), no source-model parse.
                    state->data = loadRwObjModel(device, file_name);
                    if (!state->data) {
                        err_out = "baked .rwobj load failed: " + file_name;
                        return false;
                    }
                } else if (ext == ".rwchar") {
                    // Native skinned character: one DrawableData built from
                    // the baked group (hierarchy + skinned .rwgeo + .rwanim).
                    state->data = loadRwCharacter(device, file_name);
                    if (!state->data) {
                        err_out = "baked .rwchar load failed: " + file_name;
                        return false;
                    }
                } else {
                    err_out = "unsupported mesh extension: " + ext;
                    return false;
                }
                // Mirror the sync constructor's effective_opaque_
                // scan + alpha-companion-texture build so async-
                // loaded meshes also benefit from the no-frag shadow
                // path AND have the R8 alpha companion ready for the
                // Phase-2 shadow shader.  Same routing comment as the
                // sync site.
                computeEffectiveOpaqueForMaterials(device, state->data);
                if (!state->data) {
                    err_out = "loader returned null for '" + file_name + "'";
                    return false;
                }
                return true;
            } catch (const std::exception& e) {
                err_out = std::string("load threw: ") + e.what();
                return false;
            }
        };

    // NOTE: renderbuffer_formats is a raw pointer into application-
    // owned storage (Application::renderbuffer_formats_). It must
    // outlive every in-flight async load, which it does for the
    // application lifetime. graphic_pipeline_info and thin_film_lut_tex
    // are captured by value.
    MeshLoadTask::Phase3Fn phase3_fn =
        [obj, state, device, descriptor_pool, renderbuffer_formats,
         graphic_pipeline_info, texture_sampler, thin_film_lut_tex,
         file_name]() {
            auto data = state->data;
            if (!data) {
                // Phase 2 failed; MeshLoadTaskManager already logged
                // the error. Leave obj->object_ null so isReady()
                // stays false — the caller can notice via the task
                // status if they held on to the MeshLoadTask handle.
                // Drop the in-flight entry so a later retry can submit
                // a fresh load instead of attaching to a dead slot.
                s_in_flight_loads_.erase(file_name);
                return;
            }

            // Descriptor-set + texture-image work. The descriptor pool
            // is not thread-safe per Vulkan spec, so this belongs on
            // the main thread.
            updateDescriptorSets(
                device,
                descriptor_pool,
                material_desc_set_layout_,
                skin_desc_set_layout_,
                data,
                texture_sampler,
                thin_film_lut_tex);

            // Pipeline creation (the per-primitive hashing + pipeline-cache
            // lookups match the sync ctor above — duplicated deliberately
            // to keep the two paths easy to diff).
            for (int i_mesh = 0;
                 i_mesh < static_cast<int>(data->meshes_.size());
                 i_mesh++) {
                for (int i_prim = 0;
                     i_prim < static_cast<int>(data->meshes_[i_mesh].primitives_.size());
                     i_prim++) {
                    const auto& primitive =
                        data->meshes_[i_mesh].primitives_[i_prim];
                    {
                        auto hash_value = primitive.getHash();
                        auto result = drawable_pipeline_list_.find(hash_value);
                        if (result == drawable_pipeline_list_.end()) {
                            drawable_pipeline_list_[hash_value] =
                                createDrawablePipeline(
                                    device,
                                    renderbuffer_formats[int(renderer::RenderPasses::kForward)],
                                    drawable_pipeline_layout_,
                                    graphic_pipeline_info,
                                    primitive);
                        }
                    }
                    {
                        const bool is_opaque =
                            isPrimitiveOpaque(primitive, data->materials_);
                        auto hash_value =
                            getDepthonlyHashForMaterial(
                                primitive, data->materials_);
                        auto result =
                            drawable_shadow_pipeline_list_.find(hash_value);
                        if (result == drawable_shadow_pipeline_list_.end()) {
                            drawable_shadow_pipeline_list_[hash_value] =
                                createDrawableShadowPipeline(
                                    device,
                                    renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                    drawable_pipeline_layout_,
                                    graphic_pipeline_info,
                                    primitive,
                                    is_opaque);
                        }
                    }
                    {
                        const bool is_opaque =
                            isPrimitiveOpaque(primitive, data->materials_);
                        auto hash_value =
                            getDepthonlyHashForMaterial(
                                primitive, data->materials_);
                        auto result =
                            drawable_csm_layered_pipeline_list_.find(hash_value);
                        if (result == drawable_csm_layered_pipeline_list_.end()) {
                            drawable_csm_layered_pipeline_list_[hash_value] =
                                createDrawableCsmLayeredPipeline(
                                    device,
                                    renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                    drawable_pipeline_layout_,
                                    graphic_pipeline_info,
                                    primitive,
                                    is_opaque);
                        }
                    }
                    {
                        // CSM per-cascade pipeline (DrawMode::kCsmPerCascade).
                        const bool is_opaque =
                            isPrimitiveOpaque(primitive, data->materials_);
                        auto hash_value =
                            getDepthonlyHashForMaterial(
                                primitive, data->materials_);
                        auto result =
                            drawable_csm_per_cascade_pipeline_list_.find(hash_value);
                        if (result == drawable_csm_per_cascade_pipeline_list_.end()) {
                            drawable_csm_per_cascade_pipeline_list_[hash_value] =
                                createDrawableCsmPerCascadePipeline(
                                    device,
                                    renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                    drawable_pipeline_layout_,
                                    graphic_pipeline_info,
                                    primitive,
                                    is_opaque);
                        }
                    }
                }
            }

            device->createBuffer(
                kNumDrawableInstance * sizeof(glsl::InstanceDataInfo),
                SET_2_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT, STORAGE_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
                0,
                data->instance_buffer_.buffer,
                data->instance_buffer_.memory,
                std::source_location::current());

            data->generateSharedDescriptorSet(
                device,
                descriptor_pool,
                drawable_indirect_draw_desc_set_layout_,
                update_instance_buffer_desc_set_layout_,
                game_objects_buffer_);

            // ── Mesh-shader CSM resources (parity with sync path) ─────
            // Builds the hash-cached mesh-shader pipeline + per-primitive
            // descriptor sets for eligible primitives (opaque, non-skinned,
            // UINT32 indices, ≤256 verts/tris).  MUST run AFTER the
            // instance_buffer_ createBuffer above — buildMeshShaderShadow-
            // Resources writes instance_buffer_.buffer into binding 2 of
            // each per-primitive descriptor set.  Without this call,
            // async-loaded primitives leave mesh_shader_shadow_desc_set_
            // null and dispatch falls back to the GS pipeline inside
            // drawMesh.  Sync path mirror at line ~4290.
            buildMeshShaderShadowResources(
                device,
                descriptor_pool,
                mesh_shader_shadow_desc_set_layout_,
                mesh_shader_shadow_pipeline_layout_,
                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                graphic_pipeline_info,
                data,
                drawable_csm_mesh_shader_pipeline_list_);

            // Publish into the global cache so subsequent loads of
            // the same file short-circuit in createAsync's cache
            // check above.
            drawable_object_list_[file_name] = data;

            // Attach to the shell DrawableObject and flip ready_.
            // The release store on ready_ pairs with the acquire
            // load in isReady(), synchronizing every preceding write
            // in this lambda with anything the render thread does
            // once it observes ready_ == true.
            obj->object_ = data;
            data->ready_.store(true, std::memory_order_release);

            // Fan out to every in-flight waiter that asked for the SAME
            // filename while this load was in flight.  The 6 debug-cube
            // markers all attach here — one file parse + one GPU upload,
            // six DrawableObject shells sharing the same DrawableData.
            // Each marker is then positioned independently via
            // setInstanceRootTransform (NOT setRootNodeTransform, which
            // would clobber every sibling on the shared nodes_).
            auto waiter_it = s_in_flight_loads_.find(file_name);
            if (waiter_it != s_in_flight_loads_.end()) {
                for (auto& w : waiter_it->second) {
                    if (w && w != obj) w->object_ = data;
                }
                s_in_flight_loads_.erase(waiter_it);
            }
        };

    task_manager.submit(
        file_name, std::move(phase2_fn), std::move(phase3_fn));

    return obj;
}

void DrawableData::generateSharedDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& drawable_indirect_draw_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& update_instance_buffer_desc_set_layout,
    const std::shared_ptr<renderer::BufferInfo>& game_objects_buffer) {

    // create indirect draw buffer set.
    indirect_draw_cmd_buffer_desc_set_ = device->createDescriptorSets(
        descriptor_pool, drawable_indirect_draw_desc_set_layout, 1)[0];

    // create a global ibl texture descriptor set.
    auto buffer_descs = addDrawableIndirectDrawBuffers(
        indirect_draw_cmd_buffer_desc_set_,
        indirect_draw_cmd_);
    device->updateDescriptorSets(buffer_descs);

    // update instance buffer set.
    update_instance_buffer_desc_set_ = device->createDescriptorSets(
        descriptor_pool, update_instance_buffer_desc_set_layout, 1)[0];

    // create a global ibl texture descriptor set.
    assert(game_objects_buffer);
    buffer_descs = addUpdateInstanceBuffers(
        update_instance_buffer_desc_set_,
        *game_objects_buffer,
        instance_buffer_);
    device->updateDescriptorSets(buffer_descs);
}

void DrawableObject::createGameObjectUpdateDescSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
//    const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::TextureInfo& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    // create a global ibl texture descriptor set.
    for (int soil_water = 0; soil_water < 2; soil_water++) {
        // game objects buffer update set.
        update_game_objects_buffer_desc_set_[soil_water] =
            device->createDescriptorSets(
                descriptor_pool, update_game_objects_desc_set_layout_, 1)[0];

        assert(game_objects_buffer_);
        // addGameObjectsInfoBuffer writes the CameraInfoBuffer slot
        // unconditionally — using s_view_camera_buffer_for_update_ when the
        // application has supplied it (via setViewCameraBufferForUpdate),
        // otherwise falling back to game_objects_buffer_ to keep the slot
        // bound to *something* valid so validation is satisfied even on
        // the early initStaticMembers path.
        auto write_descs = addGameObjectsInfoBuffer(
            update_game_objects_buffer_desc_set_[soil_water],
            texture_sampler,
            *game_objects_buffer_,
//            *game_camera_buffer,
            rock_layer,
            soil_water == 0 ? soil_water_layer_0 : soil_water_layer_1,
            water_flow,
            airflow_tex);

        device->updateDescriptorSets(write_descs);
    }
}

void DrawableObject::setFrustumCullPlanes(const glm::vec4 planes[6]) {
    for (int i = 0; i < 6; ++i)
        s_frustum_planes[i] = planes[i];
    s_frustum_cull_active = true;
}

void DrawableObject::clearFrustumCull() {
    s_frustum_cull_active = false;
}

void DrawableObject::initGameObjectBuffer(
    const std::shared_ptr<renderer::Device>& device) {
    if (!game_objects_buffer_) {
        game_objects_buffer_ = std::make_shared<renderer::BufferInfo>();
        device->createBuffer(
            kMaxNumObjects * sizeof(glsl::GameObjectInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            0,
            game_objects_buffer_->buffer,
            game_objects_buffer_->memory,
            std::source_location::current());
    }
}

void DrawableObject::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
//    const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::TextureInfo& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {

    assert(game_objects_buffer_);

    if (material_desc_set_layout_ == nullptr) {
        material_desc_set_layout_ =
            createMaterialDescriptorSetLayout(device);
    }

    if (skin_desc_set_layout_ == nullptr) {
        skin_desc_set_layout_ =
            createSkinDescriptorSetLayout(device);
    }

    if (drawable_indirect_draw_desc_set_layout_ == nullptr) {
        drawable_indirect_draw_desc_set_layout_ =
            createDrawableIndirectDrawDescriptorSetLayout(device);
    }

    if (update_game_objects_desc_set_layout_ == nullptr) {
        update_game_objects_desc_set_layout_ =
            createUpdateGameObjectsDescriptorSetLayout(device);
    }

    if (update_instance_buffer_desc_set_layout_ == nullptr) {
        update_instance_buffer_desc_set_layout_ =
            createInstanceBufferDescriptorSetLayout(device);
    }

    createStaticMembers(device, global_desc_set_layouts);

    createGameObjectUpdateDescSet(
        device,
        descriptor_pool,
        texture_sampler,
//        game_camera_buffer,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        water_flow,
        airflow_tex);
}

void DrawableObject::createStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {

    {
        if (drawable_pipeline_layout_) {
            device->destroyPipelineLayout(drawable_pipeline_layout_);
            drawable_pipeline_layout_ = nullptr;
        }

        if (drawable_pipeline_layout_ == nullptr) {
            assert(material_desc_set_layout_);
            assert(skin_desc_set_layout_);
            drawable_pipeline_layout_ =
                createDrawablePipelineLayout(
                    device,
                    global_desc_set_layouts,
                    material_desc_set_layout_,
                    skin_desc_set_layout_);
        }
    }

    // ── Mesh-shader shadow descriptor + pipeline layout ───────────────
    // Independent of drawable_pipeline_layout_: uses its own set 0
    // (VB/IB/instance SSBOs) and shares set RUNTIME_LIGHTS_PARAMS_SET
    // (cascade VPs).  Built unconditionally — setup cost is trivial and
    // it lets buildMeshShaderShadowResources allocate per-primitive
    // descriptor sets without a separate guard.
    {
        if (mesh_shader_shadow_desc_set_layout_ == nullptr) {
            mesh_shader_shadow_desc_set_layout_ =
                createMeshShaderShadowDescSetLayout(device);
        }
        if (mesh_shader_shadow_pipeline_layout_) {
            device->destroyPipelineLayout(mesh_shader_shadow_pipeline_layout_);
            mesh_shader_shadow_pipeline_layout_ = nullptr;
        }
        // global_desc_set_layouts[RUNTIME_LIGHTS_PARAMS_SET] is the
        // runtime_lights_desc_set_layout_ populated by application
        // before createStaticMembers is called.
        if (mesh_shader_shadow_pipeline_layout_ == nullptr) {
            assert(global_desc_set_layouts.size() > RUNTIME_LIGHTS_PARAMS_SET);
            assert(global_desc_set_layouts[RUNTIME_LIGHTS_PARAMS_SET]);
            mesh_shader_shadow_pipeline_layout_ =
                createMeshShaderShadowPipelineLayout(
                    device,
                    mesh_shader_shadow_desc_set_layout_,
                    global_desc_set_layouts[RUNTIME_LIGHTS_PARAMS_SET]);
        }
    }

    {
        if (drawable_indirect_draw_pipeline_layout_) {
            device->destroyPipelineLayout(drawable_indirect_draw_pipeline_layout_);
            drawable_indirect_draw_pipeline_layout_ = nullptr;
        }

        if (drawable_indirect_draw_pipeline_layout_ == nullptr) {
            drawable_indirect_draw_pipeline_layout_ =
                createDrawableIndirectDrawPipelineLayout(
                    device,
                    { drawable_indirect_draw_desc_set_layout_ });
        }
    }

    {
        if (drawable_indirect_draw_pipeline_) {
            device->destroyPipeline(drawable_indirect_draw_pipeline_);
            drawable_indirect_draw_pipeline_ = nullptr;
        }

        if (drawable_indirect_draw_pipeline_ == nullptr) {
            assert(drawable_indirect_draw_pipeline_layout_);
            drawable_indirect_draw_pipeline_ =
                renderer::helper::createComputePipeline(
                    device,
                    drawable_indirect_draw_pipeline_layout_,
                    "update_gltf_indirect_draw_comp.spv",
                    std::source_location::current());
        }
    }

    {
        if (update_game_objects_pipeline_layout_) {
            device->destroyPipelineLayout(update_game_objects_pipeline_layout_);
            update_game_objects_pipeline_layout_ = nullptr;
        }

        if (update_game_objects_pipeline_layout_ == nullptr) {
            update_game_objects_pipeline_layout_ =
                createGameObjectsPipelineLayout(
                    device,
                    { update_game_objects_desc_set_layout_ });
        }
    }

    {
        if (update_game_objects_pipeline_) {
            device->destroyPipeline(update_game_objects_pipeline_);
            update_game_objects_pipeline_ = nullptr;
        }

        if (update_game_objects_pipeline_ == nullptr) {
            assert(update_game_objects_pipeline_layout_);
            update_game_objects_pipeline_ =
                renderer::helper::createComputePipeline(
                    device,
                    update_game_objects_pipeline_layout_,
                    "update_game_objects_comp.spv",
                    std::source_location::current());
        }
    }

    {
        if (update_instance_buffer_pipeline_layout_) {
            device->destroyPipelineLayout(update_instance_buffer_pipeline_layout_);
            update_instance_buffer_pipeline_layout_ = nullptr;
        }

        if (update_instance_buffer_pipeline_layout_ == nullptr) {
            update_instance_buffer_pipeline_layout_ =
                createInstanceBufferPipelineLayout(
                    device,
                    { update_instance_buffer_desc_set_layout_ });
        }
    }

    {
        if (update_instance_buffer_pipeline_) {
            device->destroyPipeline(update_instance_buffer_pipeline_);
            update_instance_buffer_pipeline_ = nullptr;
        }

        if (update_instance_buffer_pipeline_ == nullptr) {
            assert(update_instance_buffer_pipeline_layout_);
            update_instance_buffer_pipeline_ =
                renderer::helper::createComputePipeline(
                    device,
                    update_instance_buffer_pipeline_layout_,
                    "update_instance_buffer_comp.spv",
                    std::source_location::current());
        }
    }
}

void DrawableObject::recreateStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats* renderbuffer_formats,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {

    createStaticMembers(device, global_desc_set_layouts);

    for (auto& pipeline : drawable_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_pipeline_list_.clear();

    for (auto& object : drawable_object_list_) {
        for (int i_mesh = 0; i_mesh < object.second->meshes_.size(); i_mesh++) {
            for (int i_prim = 0; i_prim < object.second->meshes_[i_mesh].primitives_.size(); i_prim++) {
                const auto& primitive = object.second->meshes_[i_mesh].primitives_[i_prim];
                auto hash_value = primitive.getHash();
                auto result = drawable_pipeline_list_.find(hash_value);
                if (result == drawable_pipeline_list_.end()) {
                    drawable_pipeline_list_[hash_value] =
                        createDrawablePipeline(
                            device,
                            renderbuffer_formats[int(renderer::RenderPasses::kForward)],
                            drawable_pipeline_layout_,
                            graphic_pipeline_info,
                            primitive);
                }
            }
        }
    }

    for (auto& pipeline : drawable_shadow_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_shadow_pipeline_list_.clear();

    for (auto& pipeline : drawable_csm_layered_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_csm_layered_pipeline_list_.clear();

    for (auto& pipeline : drawable_csm_per_cascade_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_csm_per_cascade_pipeline_list_.clear();

    for (auto& pipeline : drawable_csm_mesh_shader_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_csm_mesh_shader_pipeline_list_.clear();

    for (auto& object : drawable_object_list_) {
        for (int i_mesh = 0; i_mesh < object.second->meshes_.size(); i_mesh++) {
            for (int i_prim = 0; i_prim < object.second->meshes_[i_mesh].primitives_.size(); i_prim++) {
                const auto& primitive = object.second->meshes_[i_mesh].primitives_[i_prim];
                const bool is_opaque =
                    isPrimitiveOpaque(primitive, object.second->materials_);
                auto hash_value =
                    getDepthonlyHashForMaterial(
                        primitive, object.second->materials_);
                {
                    auto result = drawable_shadow_pipeline_list_.find(hash_value);
                    if (result == drawable_shadow_pipeline_list_.end()) {
                        drawable_shadow_pipeline_list_[hash_value] =
                            createDrawableShadowPipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive,
                                is_opaque);
                    }
                }
                {
                    auto result = drawable_csm_layered_pipeline_list_.find(hash_value);
                    if (result == drawable_csm_layered_pipeline_list_.end()) {
                        drawable_csm_layered_pipeline_list_[hash_value] =
                            createDrawableCsmLayeredPipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive,
                                is_opaque);
                    }
                }
                {
                    // CSM per-cascade pipeline (DrawMode::kCsmPerCascade).
                    auto result = drawable_csm_per_cascade_pipeline_list_.find(hash_value);
                    if (result == drawable_csm_per_cascade_pipeline_list_.end()) {
                        drawable_csm_per_cascade_pipeline_list_[hash_value] =
                            createDrawableCsmPerCascadePipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive,
                                is_opaque);
                    }
                }
            }
        }
    }
}

void DrawableObject::generateDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
//    const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
    const renderer::TextureInfo& thin_film_lut_tex,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::TextureInfo& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {

    for (auto& object : drawable_object_list_) {
        updateDescriptorSets(
            device,
            descriptor_pool,
            material_desc_set_layout_,
            skin_desc_set_layout_,
            object.second,
            texture_sampler,
            thin_film_lut_tex);

        object.second->generateSharedDescriptorSet(
            device,
            descriptor_pool,
            drawable_indirect_draw_desc_set_layout_,
            update_instance_buffer_desc_set_layout_,
            game_objects_buffer_);
    }

    createGameObjectUpdateDescSet(
        device,
        descriptor_pool,
        texture_sampler,
//        game_camera_buffer,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        water_flow,
        airflow_tex);
}

void DrawableObject::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    game_objects_buffer_->destroy(device);
    device->destroyDescriptorSetLayout(material_desc_set_layout_);
    device->destroyDescriptorSetLayout(skin_desc_set_layout_);
    device->destroyPipelineLayout(drawable_pipeline_layout_);
    for (auto& pipeline : drawable_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_pipeline_list_.clear();
    for (auto& pipeline : drawable_shadow_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_shadow_pipeline_list_.clear();
    for (auto& pipeline : drawable_csm_layered_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_csm_layered_pipeline_list_.clear();
    for (auto& pipeline : drawable_csm_per_cascade_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_csm_per_cascade_pipeline_list_.clear();
    for (auto& pipeline : drawable_csm_mesh_shader_pipeline_list_) {
        device->destroyPipeline(pipeline.second);
    }
    drawable_csm_mesh_shader_pipeline_list_.clear();
    if (mesh_shader_shadow_pipeline_layout_) {
        device->destroyPipelineLayout(mesh_shader_shadow_pipeline_layout_);
        mesh_shader_shadow_pipeline_layout_ = nullptr;
    }
    if (mesh_shader_shadow_desc_set_layout_) {
        device->destroyDescriptorSetLayout(mesh_shader_shadow_desc_set_layout_);
        mesh_shader_shadow_desc_set_layout_ = nullptr;
    }
    helper::destroyTextureCache(device);   // free cross-asset shared textures
    drawable_object_list_.clear();
    device->destroyDescriptorSetLayout(drawable_indirect_draw_desc_set_layout_);
    device->destroyPipelineLayout(drawable_indirect_draw_pipeline_layout_);
    device->destroyPipeline(drawable_indirect_draw_pipeline_);
    device->destroyDescriptorSetLayout(update_game_objects_desc_set_layout_);
    device->destroyPipelineLayout(update_game_objects_pipeline_layout_);
    device->destroyPipeline(update_game_objects_pipeline_);
    device->destroyDescriptorSetLayout(update_instance_buffer_desc_set_layout_);
    device->destroyPipelineLayout(update_instance_buffer_pipeline_layout_);
    device->destroyPipeline(update_instance_buffer_pipeline_);
}

void DrawableObject::updateGameObjectsCameraBuffer(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::BufferInfo>& view_camera_buffer) {
    assert(view_camera_buffer);
    for (int soil_water = 0; soil_water < 2; ++soil_water) {
        if (!update_game_objects_buffer_desc_set_[soil_water]) {
            continue;
        }
        renderer::WriteDescriptorList writes;
        renderer::Helper::addOneBuffer(
            writes,
            update_game_objects_buffer_desc_set_[soil_water],
            renderer::DescriptorType::STORAGE_BUFFER,
            CAMERA_OBJECT_BUFFER_INDEX,
            view_camera_buffer->buffer,
            view_camera_buffer->buffer->getSize());
        device->updateDescriptorSets(writes);
    }
}

void DrawableObject::updateGameObjectsBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::vec2& world_min,
    const glm::vec2& world_range,
    const glm::vec3& camera_pos,
    float air_flow_strength,
    float water_flow_strength,
    int update_frame_count,
    int soil_water,
    float delta_t,
    bool enble_airflow) {

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, update_game_objects_pipeline_);

    glsl::GameObjectsUpdateParams params;
    params.num_objects = max_alloc_game_objects_in_buffer;
    params.delta_t = delta_t;
    params.frame_count = update_frame_count;
    params.world_min = world_min;
    params.inv_world_range = 1.0f / world_range;
    params.enble_airflow = enble_airflow ? 1 : 0;
    params.air_flow_strength = air_flow_strength;
    params.water_flow_strength = water_flow_strength;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        update_game_objects_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        update_game_objects_pipeline_layout_,
        { update_game_objects_buffer_desc_set_[soil_water] });

    cmd_buf->dispatch((max_alloc_game_objects_in_buffer + 63) / 64, 1);

    cmd_buf->addBufferBarrier(
        game_objects_buffer_->buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        game_objects_buffer_->buffer->getSize());
}

void DrawableObject::updateInstanceBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    // Async-load safety: skip when the worker hasn't published the
    // instance_buffer_ yet. isReady() performs an acquire-load on
    // DrawableData::ready_ which is released at the end of phase 3.
    if (!isReady()) {
        return;
    }

    // Pre-dispatch barrier: WAR — the previous frame's reads of this
    // buffer must complete before we overwrite it.  Both consumers of
    // the instance buffer must be covered:
    //   • GS / Regular path  : input assembler reads at VERTEX_INPUT
    //   • Mesh-shader path   : storage-buffer reads at MESH_SHADER_BIT_EXT
    // Without MESH stage in the src mask, the compute write would race
    // the previous frame's mesh-shader read of the same buffer.
    cmd_buf->addBufferBarrier(
        object_->instance_buffer_.buffer,
        { SET_2_FLAG_BITS(Access, VERTEX_ATTRIBUTE_READ_BIT, SHADER_READ_BIT),
          SET_2_FLAG_BITS(PipelineStage, VERTEX_INPUT_BIT, MESH_SHADER_BIT_EXT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        object_->instance_buffer_.buffer->getSize());

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        update_instance_buffer_pipeline_);

    glsl::InstanceBufferUpdateParams params;
    params.num_instances = kNumDrawableInstance;
    // See setUseNodeTransformOnly() — when true, force the compute pass
    // to write identity transforms instead of reading the shared
    // game_objects_buffer_ position.  This is what fixes the player /
    // hand-placed-gltf double-transform bug: without it, every drawable
    // gets an additional (camera_pos@frame0 + physics drift) world-
    // space offset stacked on top of its node-hierarchy translation.
    params.force_identity = use_node_transform_only_ ? 1u : 0u;
    params.pad0 = 0u;
    params.pad1 = 0u;
    // Explicit transform for the force_identity path.  ALWAYS a true
    // identity: the editor placement TRS is applied on the OTHER side
    // of the vertex math — drawMesh pre-multiplies model_params.model_-
    // mat by m_current_instance_world_ (staged from the instance-root
    // TRS in draw()).  Writing the TRS here as well would apply the
    // placement twice.  The push-constant fields stay so the compute
    // can take an arbitrary transform later if a caller ever needs the
    // instance buffer (not model_mat) to carry placement.
    {
        const glm::mat4 forced(1.0f);
        params.forced_mat_0 = forced[0];
        params.forced_mat_1 = forced[1];
        params.forced_mat_2 = forced[2];
        params.forced_pos   = forced[3];
    }
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        update_instance_buffer_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        update_instance_buffer_pipeline_layout_,
        { object_->update_instance_buffer_desc_set_ });

    cmd_buf->dispatch((params.num_instances + 63) / 64, 1);

    // Post-dispatch barrier: RAW — compute writes must be visible to
    // EVERY downstream consumer of the instance buffer this frame:
    //   • GS / Regular path  : input assembler reads at VERTEX_INPUT
    //   • Mesh-shader path   : storage-buffer reads at MESH_SHADER_BIT_EXT
    // Without MESH stage in the dst mask the mesh shader can race the
    // compute write and read STALE / UNDEFINED instance transforms,
    // which manifests as missing or mis-positioned shadow casters in
    // the kCsmMeshShader draw mode (the GS / Regular paths are fine
    // because their VERTEX_INPUT_BIT dst is covered).
    cmd_buf->addBufferBarrier(
        object_->instance_buffer_.buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_2_FLAG_BITS(Access, VERTEX_ATTRIBUTE_READ_BIT, SHADER_READ_BIT),
          SET_2_FLAG_BITS(PipelineStage, VERTEX_INPUT_BIT, MESH_SHADER_BIT_EXT) },
        object_->instance_buffer_.buffer->getSize());
}

void DrawableObject::updateIndirectDrawBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    if (!isReady()) {
        return;
    }

    cmd_buf->addBufferBarrier(
        object_->indirect_draw_cmd_.buffer,
        { SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT), SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        object_->indirect_draw_cmd_.buffer->getSize());

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        drawable_indirect_draw_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        drawable_indirect_draw_pipeline_layout_,
        &object_->num_prims_,
        sizeof(object_->num_prims_));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        drawable_indirect_draw_pipeline_layout_,
        { object_->indirect_draw_cmd_buffer_desc_set_ });

    cmd_buf->dispatch((object_->num_prims_ + 63) / 64, 1);

    cmd_buf->addBufferBarrier(
        object_->indirect_draw_cmd_.buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT), SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) },
        object_->indirect_draw_cmd_.buffer->getSize());
}

void DrawableObject::updateBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {
    updateInstanceBuffer(cmd_buf);
    updateIndirectDrawBuffer(cmd_buf);
}

void DrawableObject::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    bool depth_only/* = false */,
    DrawMode draw_mode/* = DrawMode::kForward */,
    uint32_t csm_cascade_idx/* = 0 */) {

    // Per-wrapper visibility gate (set from app code, e.g. the Render
    // Debug menu's bone-only / character-only mode).  Skipping at the
    // very top means no instance-buffer bind, no node walk, no shadow
    // recording — exactly what we want when the user hides this drawable.
    if (!visible_) return;

    // ECS object-level frustum cull: a coarse early-out computed by
    // ecs::CullingSystem over the entity's WorldBounds.  FORWARD ONLY —
    // shadow / CSM / depth passes must still record geometry that is
    // outside the camera frustum (off-screen casters still throw
    // shadows into view), and the app only arms this hint around the
    // main forward pass anyway (cleared right after, so probe passes
    // never observe it).
    if (ecs_culled_hint_ && draw_mode == DrawMode::kForward) return;

    // Reset per-call debug counter BEFORE the isReady() guard so that
    // a not-ready drawable also prints "0 draws" rather than carrying
    // last frame's count.  Cheap; only used when m_debug_force_red_
    // is set on the drawable.
    if (object_ &&
        (object_->m_debug_force_red_ || object_->m_debug_log_draws_)) {
        object_->m_debug_draw_call_count_           = 0;
        object_->m_debug_draw_called_               = 1;
        object_->m_debug_draw_not_ready_            = 0;
        object_->m_debug_draw_nodes_visited_        = 0;
        object_->m_debug_draw_nodes_with_mesh_      = 0;
        object_->m_debug_draw_mesh_entered_         = 0;
        object_->m_debug_draw_mesh_culled_frustum_  = 0;
        object_->m_debug_draw_mesh_taken_by_cluster_= 0;
        object_->m_debug_draw_mesh_cluster_debug_path_ = 0;
        object_->m_debug_draw_prims_iterated_       = 0;
        object_->m_debug_draw_prim_pipeline_null_   = 0;
        object_->m_debug_draw_prim_mesh_shader_     = 0;
    }

    // ── Stage the per-instance world for this draw call ─────────────
    // drawNodes pre-multiplies model_mat by m_current_instance_world_, so a
    // wrapper that shares its DrawableData with other instances can pin its
    // own world position here without touching the shared nodes_ (which
    // every sibling marker would otherwise clobber).  When the wrapper
    // hasn't set an override, identity restores the original behaviour
    // (model_mat == cached_matrix).
    if (object_) {
        if (instance_root_valid_) {
            glm::mat4 m =
                glm::translate(glm::mat4(1.0f), instance_root_t_) *
                glm::mat4_cast(instance_root_r_) *
                glm::scale(glm::mat4(1.0f), instance_root_s_);
            // Fold m_debug_scale_ in here for shared-instance drawables.
            // setRootNodeTransform writes the debug scale into the root
            // node's scale_, but shared instances DON'T call it (would
            // clobber siblings), so the per-instance world has to carry
            // the scale instead.  Non-shared drawables (player rig, etc.)
            // keep getting scale via the node path unchanged.  Composing
            // this AFTER instance_root_s_ means link sticks can size
            // themselves in "(length/debug, thin/debug, thin/debug)" units
            // and end up at (length, thin, thin) on screen.
            if (object_->m_debug_scale_ > 0.0f) {
                m = m * glm::scale(
                    glm::mat4(1.0f),
                    glm::vec3(object_->m_debug_scale_));
            }
            object_->m_current_instance_world_ = m;
        } else {
            object_->m_current_instance_world_ = glm::mat4(1.0f);
        }

        // ── Stage the per-wrapper sub-object filter for this draw call ──
        // Lazily resolve the ordinal ("k-th node with a mesh", the order the
        // Outliner / Content Browser use) to a nodes_ index now that the
        // node table exists, then stage it into the SHARED DrawableData the
        // same way the instance world is staged — so sibling wrappers that
        // dedup onto one DrawableData can each render a different node.
        if (only_render_ordinal_ >= 0 && only_render_node_ < 0 && isReady()) {
            int k = 0;
            for (int ni = 0; ni < (int)object_->nodes_.size(); ++ni) {
                if (object_->nodes_[ni].mesh_idx_ < 0) continue;
                if (k == only_render_ordinal_) {
                    only_render_node_ = ni;
                    break;
                }
                ++k;
            }
            if (only_render_node_ < 0) {
                // Out-of-range ordinal — render nothing rather than
                // silently showing the whole file (no node has this index).
                only_render_node_ = 0x7fffffff;
            }
        }
        object_->m_only_render_node_ =
            (only_render_ordinal_ >= 0) ? only_render_node_ : -1;
    }

    // Skip while the async load is still in flight. The menu spinner
    // (see application.cpp HUD wiring) is the user-visible signal; the
    // object simply pops in the first frame after phase 3 finalizes.
    if (!isReady()) {
        return;
    }

    // Defensive: a destroyed-but-cached DrawableData (its GPU buffers
    // freed while ready_ stayed set) must not reach the bind below —
    // it would crash in bindVertexBuffers on a null buffer.
    if (!object_->instance_buffer_.buffer) {
        return;
    }

    auto& pipeline_list =
        (draw_mode == DrawMode::kCsmLayered)    ? drawable_csm_layered_pipeline_list_     :
        (draw_mode == DrawMode::kCsmPerCascade) ? drawable_csm_per_cascade_pipeline_list_ :
        (draw_mode == DrawMode::kCsmMeshShader) ? drawable_csm_mesh_shader_pipeline_list_ :
        depth_only                              ? drawable_shadow_pipeline_list_           :
                                                  drawable_pipeline_list_;

    std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
    std::vector<uint64_t> offsets(1);
    buffers[0] = object_->instance_buffer_.buffer;
    offsets[0] = 0;
    cmd_buf->bindVertexBuffers(VINPUT_INSTANCE_BINDING_POINT, buffers, offsets);

    num_draw_meshes = 0;
    size_t last_hash = 0;

    // In mesh-shader mode, drawMesh needs the GS pipeline list as a
    // fallback for ineligible primitives (skinned, cutout, UINT16
    // indices, oversized).  Outside mesh-shader mode the fallback is
    // unused.
    const bool mesh_shader_csm_mode = (draw_mode == DrawMode::kCsmMeshShader);
    std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>>*
        mesh_shader_fallback =
            mesh_shader_csm_mode ? &drawable_csm_layered_pipeline_list_ : nullptr;

    int32_t root_node =
        object_->default_scene_ >= 0 ? object_->default_scene_ : 0;
    if (object_->m_only_render_node_ >= 0) {
        // Sub-object wrapper: jump STRAIGHT to the filtered node instead of
        // walking the whole hierarchy.  With N placed objects sharing one
        // big FBX, the full walk cost N x total_nodes per pass on the CPU
        // (the CSM + forward spike); the filtered node's own subtree is
        // tiny, and drawNodes still skips any non-matching child meshes.
        // An out-of-range filter (unresolvable ordinal sentinel) draws
        // nothing at all.
        if (object_->m_only_render_node_ <
            (int32_t)object_->nodes_.size()) {
            drawNodes(
                cmd_buf,
                object_,
                drawable_pipeline_layout_,
                desc_set_list,
                object_->m_only_render_node_,
                pipeline_list,
                viewports,
                scissors,
                depth_only,
                last_hash,
                csm_cascade_idx,
                mesh_shader_csm_mode,
                mesh_shader_fallback);
        }
    } else {
        for (auto node_idx : object_->scenes_[root_node].nodes_) {
            drawNodes(
                cmd_buf,
                object_,
                drawable_pipeline_layout_,
                desc_set_list,
                node_idx,
                pipeline_list,
                viewports,
                scissors,
                depth_only,
                last_hash,
                csm_cascade_idx,
                mesh_shader_csm_mode,
                mesh_shader_fallback);
        }
    }

}

void DrawableObject::update(
    const std::shared_ptr<renderer::Device>& device,
    const float& time) {
    if (object_ && object_->ready_.load(std::memory_order_acquire)) {
        // use_node_transform_only_ is the same flag that opts the
        // drawable into identity-instance mode (see setUseNodeTransform-
        // Only).  Conceptually it means "external code owns the node
        // transforms" — for the player, that external code is
        // PlayerController::applyPose, which writes the root translation
        // and procedural limb rotations every frame.  If the imported
        // glTF animation channels also fire, they OVERWRITE those writes
        // (animation update runs AFTER applyPose, before joint matrices
        // are baked).  Net symptom: character renders at the animation's
        // authored origin (~0,0,0) instead of the spawn point — the
        // "character missing" bug.  Forwarding the flag as skip_anima-
        // tions stops the imported animation timeline from running while
        // still letting joint matrices be rebuilt from the controller's
        // node writes.
        object_->update(
            device,
            0,
            time,
            object_->m_use_local_matrix_only_,
            /*skip_animations=*/(use_node_transform_only_ ||
                                 external_animation_));
    }
}

void DrawableObject::setRootNodeTransform(
    const glm::vec3& translation,
    const glm::quat& rotation) {
    if (!object_ || !object_->ready_.load(std::memory_order_acquire))
        return;
    if (object_->nodes_.empty()) return;
    if (object_->default_scene_ < 0 ||
        object_->default_scene_ >= (int32_t)object_->scenes_.size()) return;
    const auto& scene = object_->scenes_[object_->default_scene_];

    // ── Apply override to EVERY scene root, not just nodes_[0] ──────
    // glTF allows multiple root nodes per scene.  A very common
    // export pattern (Blender, Mixamo, etc.) places the mesh node as
    // one root and the armature/skeleton as a SIBLING root rather
    // than a descendant.  Skinned meshes render through joint.cached_
    // matrix * inverse_bind, which means the bones — NOT the mesh
    // node — determine where the body actually appears on screen.  If
    // we only translate nodes_[0] (the mesh root), the mesh node's
    // cached_matrix moves but the skeleton stays at origin, and the
    // character renders wherever the bones were authored (typically
    // (0,0,0)) while our CPU-side bbox (which transforms mesh.bbox
    // through mesh_node.cached_matrix) reports the new spawn
    // location.  Net result: bbox / red marker high up, body on the
    // floor — exactly the symptom we just observed.
    //
    // Translating every root in the scene moves the mesh AND every
    // sibling armature in lockstep, so the rendered body follows.
    std::set<int32_t> touched_roots;
    auto apply_to_node = [&](int32_t idx) {
        if (idx < 0 || (size_t)idx >= object_->nodes_.size()) return;
        if (!touched_roots.insert(idx).second) return;
        auto& n = object_->nodes_[idx];
        n.translation_ = translation;
        n.rotation_    = rotation;
        // ── Debug giant-size override ────────────────────────────
        // When m_debug_scale_ > 0, force this root node's local
        // scale to that value.  Combined with the asset's own root-
        // node matrix (which for scene-skinned.gltf bakes a 0.1
        // uniform scale + 90° axis swap), the final on-screen size
        // is 0.1 × m_debug_scale_ × (asset units).  Set to 0 (or
        // below) to leave scale alone.
        if (object_->m_debug_scale_ > 0.0f) {
            n.scale_ = glm::vec3(object_->m_debug_scale_);
        }
    };
    for (auto root_idx : scene.nodes_) apply_to_node(root_idx);

    // Belt-and-braces: some exporters (older Mixamo, certain Maya
    // setups) reference the skeleton root via skin.skeleton_root_
    // but don't include it in scene.nodes_.  If we miss it, the
    // bones stay at origin and the rendered body follows them — the
    // exact symptom this override is meant to eliminate.  Walk the
    // skins and apply the same override to any skeleton_root_ we
    // haven't already touched (the set dedupes).
    for (const auto& skin : object_->skins_) {
        apply_to_node(skin.skeleton_root_);
    }
}

int DrawableObject::findNodeIndexByName(const std::string& name) const {
    if (!object_ || !object_->ready_.load(std::memory_order_acquire))
        return -1;
    for (size_t i = 0; i < object_->nodes_.size(); ++i) {
        if (object_->nodes_[i].name_ == name) return (int)i;
    }
    return -1;
}

bool DrawableObject::setNodeRotationByName(
    const std::string& name,
    const glm::quat& rotation) {
    int idx = findNodeIndexByName(name);
    if (idx < 0) return false;
    object_->nodes_[idx].rotation_ = rotation;
    return true;
}

glm::quat DrawableObject::getNodeRotationByName(
    const std::string& name) const {
    int idx = findNodeIndexByName(name);
    if (idx < 0) return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    return object_->nodes_[idx].rotation_;
}

glm::mat4 DrawableObject::getNodeWorldMatrixByName(
    const std::string& name) const {
    // findNodeIndexByName() already guards on object_ != nullptr +
    // async-load-complete; -1 from it means either condition failed
    // or the name didn't resolve.
    int idx = findNodeIndexByName(name);
    if (idx < 0) return glm::mat4(1.0f);
    // cached_matrix_ is written by DrawableData::update() each frame
    // as the parent-chain product of every ancestor's local matrix.
    // It's the same matrix the skinning path consumes when building
    // joint_matrices, so it's guaranteed in sync with the on-screen
    // pose.  No need to recompute here.
    return object_->nodes_[idx].cached_matrix_;
}

glm::vec3 DrawableObject::getModelBboxMin() const {
    if (!object_ || !object_->ready_.load(std::memory_order_acquire))
        return glm::vec3(0.0f);
    glm::vec3 mn(std::numeric_limits<float>::max());
    for (const auto& m : object_->meshes_) mn = glm::min(mn, m.bbox_min_);
    if (mn.x == std::numeric_limits<float>::max()) return glm::vec3(0.0f);
    return mn;
}

glm::vec3 DrawableObject::getModelBboxMax() const {
    if (!object_ || !object_->ready_.load(std::memory_order_acquire))
        return glm::vec3(0.0f);
    glm::vec3 mx(std::numeric_limits<float>::lowest());
    for (const auto& m : object_->meshes_) mx = glm::max(mx, m.bbox_max_);
    if (mx.x == std::numeric_limits<float>::lowest()) return glm::vec3(0.0f);
    return mx;
}

bool DrawableObject::getSkinnedModelAabb(
    glm::vec3& bmin, glm::vec3& bmax) const {
    if (!object_ || !object_->ready_.load(std::memory_order_acquire))
        return false;
    if (object_->skins_.empty()) return false;

    bmin = glm::vec3(std::numeric_limits<float>::max());
    bmax = glm::vec3(std::numeric_limits<float>::lowest());
    size_t joints_seen = 0;
    for (const auto& skin : object_->skins_) {
        for (const auto j : skin.joints_) {
            if (j < 0 || (size_t)j >= object_->nodes_.size()) continue;
            // cached_matrix_ column 3 = the joint's model-space
            // position — exactly where vertices weighted to this
            // joint render (joint.cached * inv_bind maps bind-pose
            // verts back into the joint's neighbourhood).
            const glm::vec3 p =
                glm::vec3(object_->nodes_[j].cached_matrix_[3]);
            bmin = glm::min(bmin, p);
            bmax = glm::max(bmax, p);
            ++joints_seen;
        }
    }
    if (joints_seen < 2) return false;

    // Joints are bone ORIGINS — the mesh surface sits a few cm to a
    // few dm outside them (skull above the head joint, fingertips past
    // the hand joint, clothing).  Pad each side by 15% of the largest
    // extent, at least 5 cm.
    const glm::vec3 ext = bmax - bmin;
    const float pad = std::max(
        0.05f, 0.15f * std::max(ext.x, std::max(ext.y, ext.z)));
    bmin -= glm::vec3(pad);
    bmax += glm::vec3(pad);
    return true;
}

std::shared_ptr<renderer::BufferInfo> DrawableObject::getGameObjectsBuffer() {
    return game_objects_buffer_;
}

std::shared_ptr<ego::DrawableData> DrawableObject::loadGltfModel(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& input_filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    std::string ext = getFilePathExtension(input_filename);

    bool ret = false;
    if (ext.compare("glb") == 0) {
        // assume binary glTF.
        ret =
            loader.LoadBinaryFromFile(&model, &err, &warn, input_filename.c_str());
    }
    else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, input_filename.c_str());
    }

    if (!warn.empty()) {
        std::cout << "Warn: " << warn.c_str() << std::endl;
    }

    if (!err.empty()) {
        std::cout << "ERR: " << err.c_str() << std::endl;
    }
    if (!ret) {
        std::cout << "Failed to load .glTF : " << input_filename << std::endl;
        return nullptr;
    }

    auto drawable_object = std::make_shared<ego::DrawableData>(device);
    drawable_object->meshes_.reserve(model.meshes.size());

    setupMeshState(device, model, drawable_object, input_filename);
    setupMeshes(model, drawable_object);
    setupAnimations(model, drawable_object);
    setupSkins(device, model, drawable_object);
    captureRtSkinSource(model, drawable_object);
    setupNodes(model, drawable_object);
    setupModel(model, drawable_object);
    for (auto& scene : drawable_object->scenes_) {
        for (auto& node : scene.nodes_) {
            calculateBbox(drawable_object, scene.nodes_[0], glm::mat4(1.0f), scene.bbox_min_, scene.bbox_max_);
        }
    }

    drawable_object->m_use_local_matrix_only_ = false;

    drawable_object->update(
        device,
        0,
        0.0f,
        drawable_object->m_use_local_matrix_only_);

    setupRaytracing(drawable_object);

    // init indirect draw buffer.
    uint32_t num_prims = 0;
    for (auto& mesh : drawable_object->meshes_) {
        for (auto& prim : mesh.primitives_) {
            prim.indirect_draw_cmd_ofs_ =
                num_prims * sizeof(renderer::DrawIndexedIndirectCommand) + INDIRECT_DRAW_BUF_OFS;
            num_prims++;
        }
    }

    std::vector<uint32_t> indirect_draw_cmd_buffer(
        sizeof(renderer::DrawIndexedIndirectCommand) / sizeof(uint32_t) * num_prims + 1);
    auto indirect_draw_buf = reinterpret_cast<renderer::DrawIndexedIndirectCommand*>(indirect_draw_cmd_buffer.data() + 1);

    // clear instance count = 0;
    indirect_draw_cmd_buffer[0] = 0;
    uint32_t prim_idx = 0;
    uint32_t cur_lod = 0;
    for (const auto& mesh : drawable_object->meshes_) {
        for (const auto& prim : mesh.primitives_) {
            indirect_draw_buf[prim_idx].first_index = 0;
            indirect_draw_buf[prim_idx].first_instance = 0;
            indirect_draw_buf[prim_idx].index_count =
                static_cast<uint32_t>(prim.index_desc_[cur_lod].index_count);
            indirect_draw_buf[prim_idx].instance_count = 0;
            indirect_draw_buf[prim_idx].vertex_offset = 0;
            prim_idx++;
        }
    }

    renderer::Helper::createBuffer(
        device,
        SET_2_FLAG_BITS(BufferUsage, INDIRECT_BUFFER_BIT, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        drawable_object->indirect_draw_cmd_.buffer,
        drawable_object->indirect_draw_cmd_.memory,
        std::source_location::current(),
        indirect_draw_cmd_buffer.size() * sizeof(uint32_t),
        indirect_draw_cmd_buffer.data());

    drawable_object->num_prims_ = num_prims;

    return drawable_object;
}

std::shared_ptr<ego::DrawableData> DrawableObject::loadFbxModel(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& input_filename)
{
    ufbx_load_opts opts = { 0 };
    ufbx_error error;
    ufbx_abi ufbx_scene* fbx_scene =
        ufbx_load_file(
            input_filename.c_str(),
            &opts, &error);

    auto drawable_object = std::make_shared<ego::DrawableData>(device);
    drawable_object->meshes_.reserve(fbx_scene->meshes.count);

    drawable_object->m_flip_v_ = true;

    setupMeshState(device, fbx_scene, drawable_object, input_filename);
    std::ostringstream log_buf;
    setupMeshes(device, fbx_scene, drawable_object, log_buf);
//    setupAnimations(model, drawable_object);
//    setupSkins(device, model, drawable_object);
    setupNodes(fbx_scene, drawable_object);
    setupModel(fbx_scene, drawable_object);
    for (auto& scene : drawable_object->scenes_) {
        for (auto& node : scene.nodes_) {
            calculateBbox(drawable_object, scene.nodes_[0], glm::mat4(1.0f), scene.bbox_min_, scene.bbox_max_);
        }
    }

    ufbx_free_scene(fbx_scene);

    // Write LOD generation log to a timestamped file.
    {
        auto now = std::chrono::system_clock::now();
        auto time_t_val = std::chrono::system_clock::to_time_t(now);
        std::tm tm_info{};
#ifdef _WIN32
        localtime_s(&tm_info, &time_t_val);
#else
        localtime_r(&time_t_val, &tm_info);
#endif
        std::ostringstream filename_ss;
        filename_ss << "logs/" << std::put_time(&tm_info, "%Y-%m-%d_%H-%M-%S") << ".log";
        std::filesystem::create_directories("logs");
        std::ofstream log_file(filename_ss.str());
        if (log_file.is_open()) {
            log_file << log_buf.str();
        }
    }

    drawable_object->m_use_local_matrix_only_ = true;

    drawable_object->update(
        device,
        0,
        0.0f,
        drawable_object->m_use_local_matrix_only_);

    setupRaytracing(drawable_object);

    // init indirect draw buffer.
    uint32_t num_prims = 0;
    for (auto& mesh : drawable_object->meshes_) {
        for (auto& prim : mesh.primitives_) {
            prim.indirect_draw_cmd_ofs_ =
                num_prims * sizeof(renderer::DrawIndexedIndirectCommand) + INDIRECT_DRAW_BUF_OFS;
            num_prims++;
        }
    }

    std::vector<uint32_t> indirect_draw_cmd_buffer(
        sizeof(renderer::DrawIndexedIndirectCommand) / sizeof(uint32_t) * num_prims + 1);
    auto indirect_draw_buf = reinterpret_cast<renderer::DrawIndexedIndirectCommand*>(indirect_draw_cmd_buffer.data() + 1);

    // clear instance count = 0;
    indirect_draw_cmd_buffer[0] = 0;
    uint32_t prim_idx = 0;
    for (const auto& mesh : drawable_object->meshes_) {
        for (const auto& prim : mesh.primitives_) {
            // TODO: replace with distance-based LOD selection (same as drawMesh).
            uint32_t cur_lod = 0;
            indirect_draw_buf[prim_idx].first_index = 0;
            indirect_draw_buf[prim_idx].first_instance = 0;
            indirect_draw_buf[prim_idx].index_count =
                static_cast<uint32_t>(prim.index_desc_[cur_lod].index_count);
            indirect_draw_buf[prim_idx].instance_count = 0;
            indirect_draw_buf[prim_idx].vertex_offset = 0;
            prim_idx++;
        }
    }

    renderer::Helper::createBuffer(
        device,
        SET_2_FLAG_BITS(BufferUsage, INDIRECT_BUFFER_BIT, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        drawable_object->indirect_draw_cmd_.buffer,
        drawable_object->indirect_draw_cmd_.memory,
        std::source_location::current(),
        indirect_draw_cmd_buffer.size() * sizeof(uint32_t),
        indirect_draw_cmd_buffer.data());

    drawable_object->num_prims_ = num_prims;

    return drawable_object;
}

// ─── loadRwObjModel ────────────────────────────────────────────────────────
// Build a DrawableData straight from a baked render-ready asset:
//   .rwobj  → names the object + points at its .rwgeo
//   .rwgeo  → world-space geometry + material SECTIONS (per-face materials
//             collapsed into contiguous index ranges at bake time)
//   .rwtex  → decoded RGBA8 base-colour textures (referenced by section)
// One mesh, one node (geometry is baked in source-world space, transforms
// already applied), one primitive + material per section.  Mirrors the
// FBX loader's GPU layout exactly (interleaved VertexStruct VB, shared IB,
// 4 buffer views, LOD0-only index ranges) so the forward pipeline, shadow
// pass, selection, collision and cluster upload all work unchanged.
std::shared_ptr<ego::DrawableData> DrawableObject::loadRwObjModel(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& input_filename) {

    // Resolve the reference.  geo_path comes back resolved against the
    // .rwobj's own folder; empty when the asset was never baked.
    std::string src_path, ref_name, geo_path;
    int sub_ordinal = -1;
    if (!helper::readRwObjRef(input_filename, src_path, sub_ordinal,
                              ref_name, geo_path) ||
        geo_path.empty()) {
        std::cout << "[rwobj] '" << input_filename
                  << "' has no baked geometry (geo= missing)" << std::endl;
        return nullptr;
    }

    helper::ModelPreviewData md;
    std::vector<std::string> tex_paths;   // parallel to md.textures
    if (!helper::loadRwGeo(geo_path, md, &tex_paths,
                           /*decode_textures=*/false) ||
        md.positions.empty() || md.indices.empty() || md.sections.empty()) {
        std::cout << "[rwobj] failed to read baked geometry '" << geo_path
                  << "'" << std::endl;
        return nullptr;
    }

    // Dedup triangle-soup vertices (files baked before the bake-side
    // dedup landed store one vertex per triangle corner — zero GPU
    // vertex-cache reuse, ~3× the vertex shading of the engine's own
    // FBX path).  Near no-op for freshly baked, already-indexed files.
    {
        const size_t pre_vc = md.positions.size();
        helper::dedupModelVertices(md);
        if (md.positions.size() < pre_vc) {
            std::cout << "[rwobj] vertex dedup: " << pre_vc << " -> "
                      << md.positions.size() << std::endl;
        }
    }

    auto drawable_object = std::make_shared<ego::DrawableData>(device);
    // Baked UVs are FINAL (the bake already applied the FBX V-flip), so
    // no shader-side flip — leave m_flip_u_/m_flip_v_ at their false
    // defaults.  Single static node → local-matrix-only transforms.
    drawable_object->m_use_local_matrix_only_ = true;

    // ── CPU-side mesh (retained: cluster sidecar, collision, selection) ──
    const uint32_t vtx_count =
        static_cast<uint32_t>(md.positions.size());
    helper::Mesh cpu_mesh;   // ctor allocates the shared vectors
    cpu_mesh.vertex_data_ptr->resize(vtx_count);
    for (uint32_t i = 0; i < vtx_count; ++i) {
        auto& v = cpu_mesh.vertex_data_ptr->at(i);
        v.position = md.positions[i];
        v.normal = (i < md.normals.size())
            ? md.normals[i] : glm::vec3(0.0f, 1.0f, 0.0f);
        v.uv = (i < md.uvs.size()) ? md.uvs[i] : glm::vec2(0.0f);
    }
    const uint32_t index_count =
        static_cast<uint32_t>(md.indices.size());
    cpu_mesh.faces_ptr->reserve(index_count / 3);
    for (uint32_t i = 0; i + 2 < index_count; i += 3) {
        cpu_mesh.faces_ptr->emplace_back(
            md.indices[i], md.indices[i + 1], md.indices[i + 2]);
    }

    // ── Textures — VT-ONLY, with a cross-object cache ──────────────────
    // Baked objects render through the cluster pipeline, whose materials
    // sample the Runtime Virtual Texture pool — registerMaterial builds
    // its BC7 tile cache straight from CPU pixels.  So NO full-res GPU
    // texture is created here at all: we keep only
    //   • cpu_pixels  — feeds VT registration + the effective-opaque scan
    //   • a small R8 alpha companion (cutout textures only) for the
    //     forward/shadow mask path
    // The forward pass (brief window before cluster takeover, and during
    // transform edits) renders these objects with their flat section
    // colour — the descriptor writer binds black/white fallbacks for
    // view-less textures.
    //
    // Entries are cached by canonical .rwtex path: baked objects of one
    // group share textures heavily, and per-OBJECT copies were the
    // "24 GB instead of 2" symptom.  Copies share cpu_pixels / the
    // companion via shared_ptr; entries live for the app's lifetime
    // (same policy as the drawable dedup cache).
    static std::mutex s_rwtex_cache_mutex;
    static std::unordered_map<std::string, renderer::TextureInfo>
        s_rwtex_gpu_cache;

    drawable_object->textures_.resize(md.textures.size());
    for (size_t ti = 0; ti < md.textures.size() && ti < tex_paths.size();
         ++ti) {
        auto& dst = drawable_object->textures_[ti];

        std::string key = std::filesystem::path(tex_paths[ti])
                              .lexically_normal().generic_string();
        for (auto& c : key) c = (char)std::tolower((unsigned char)c);

        {
            std::lock_guard<std::mutex> lk(s_rwtex_cache_mutex);
            auto it = s_rwtex_gpu_cache.find(key);
            if (it != s_rwtex_gpu_cache.end()) {
                dst = it->second;   // shares cpu_pixels / companion
                continue;
            }
        }

        helper::RwTexBaked tb;
        if (!helper::readRwTexBaked(tex_paths[ti], tb) ||
            tb.w <= 0 || tb.h <= 0) {
            continue;
        }
        dst.size = glm::uvec3(uint32_t(tb.w), uint32_t(tb.h), 1u);
        dst.linear = false;   // base colour → sRGB semantics
        dst.source_filename_ = tex_paths[ti];
        // Bake-time pre-encoded VT tile cache: registration becomes a
        // memcpy (no runtime CPU BC7 encode at all).
        dst.vt_bc7_tiles = tb.bc7_tiles;

        // R8 cutout companion — from the baked full-res alpha plane
        // (format 1) or a scan of the legacy full-res pixels (format
        // 0).  Built once per UNIQUE texture (cached);
        // computeEffectiveOpaqueForMaterials adopts it instead of
        // rebuilding per object.
        std::vector<uint8_t> alpha_data;
        int aw = tb.w, ah = tb.h;
        if (!tb.alpha.empty()) {
            alpha_data.assign(tb.alpha.begin(), tb.alpha.end());
        } else if (!tb.bc7_tiles &&
                   tb.preview_w == tb.w && tb.preview_h == tb.h) {
            // Legacy file: preview IS the full-res image — scan it.
            bool has_transparency = false;
            for (size_t i = 3; i < tb.preview_rgba.size(); i += 4) {
                if (tb.preview_rgba[i] < 250) {
                    has_transparency = true;
                    break;
                }
            }
            if (has_transparency) {
                alpha_data.resize(size_t(tb.w) * tb.h);
                for (size_t i = 0, n = alpha_data.size(); i < n; ++i) {
                    alpha_data[i] = tb.preview_rgba[i * 4 + 3];
                }
            }
        }
        if (!alpha_data.empty()) {
            renderer::Helper::create2DTextureImage(
                device,
                renderer::Format::R8_UNORM,
                aw,
                ah,
                alpha_data.data(),
                dst.alpha_only_image,
                dst.alpha_only_memory,
                std::source_location::current());
            dst.alpha_only_view = device->createImageView(
                dst.alpha_only_image,
                renderer::ImageViewType::VIEW_2D,
                renderer::Format::R8_UNORM,
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                std::source_location::current());
        }

        // CPU pixels: only needed as the VT registration source when
        // there's NO baked tile cache (legacy format 0 — full-res
        // pixels).  With a blob present we deliberately skip it: the
        // preview is low-res and would mismatch dst.size, and keeping
        // it would only waste RAM.
        if (!dst.vt_bc7_tiles &&
            tb.preview_w == tb.w && tb.preview_h == tb.h &&
            !tb.preview_rgba.empty()) {
            dst.cpu_pixels = std::make_shared<std::vector<uint8_t>>(
                tb.preview_rgba.begin(), tb.preview_rgba.end());
        }

        {
            std::lock_guard<std::mutex> lk(s_rwtex_cache_mutex);
            s_rwtex_gpu_cache.emplace(key, dst);
        }
    }

    // ── Materials: one per baked section ──────────────────────────────
    const size_t num_sections = md.sections.size();
    drawable_object->materials_.resize(num_sections);
    for (size_t si = 0; si < num_sections; ++si) {
        const auto& sec = md.sections[si];
        auto& dst_material = drawable_object->materials_[si];
        dst_material.name_ = ref_name + "_s" + std::to_string(si);
        if (sec.tex_index >= 0 &&
            static_cast<size_t>(sec.tex_index) <
                drawable_object->textures_.size()) {
            dst_material.base_color_idx_ = sec.tex_index;
        }
        // v5 bakes carry normal / metallic-roughness refs too — record
        // them for the future native skinned/forward render (the VT-only
        // texture entries have no GPU views yet, so the forward shader
        // feature flags stay off below).
        if (sec.nrm_index >= 0 &&
            static_cast<size_t>(sec.nrm_index) <
                drawable_object->textures_.size()) {
            dst_material.normal_idx_ = sec.nrm_index;
        }
        if (sec.mr_index >= 0 &&
            static_cast<size_t>(sec.mr_index) <
                drawable_object->textures_.size()) {
            dst_material.metallic_roughness_idx_ = sec.mr_index;
        }
        // Match the FBX path exactly: base.frag has ALPHAMODE_MASK
        // always active with cutoff 0.1 — opaque textures (alpha 1)
        // never trigger the discard, cutout foliage/signs do.
        dst_material.alpha_cutoff_ = 0.1f;
        dst_material.alpha_mask_   = true;
        dst_material.alpha_mode_   = ego::AlphaMode::Mask;

        device->createBuffer(
            sizeof(glsl::PbrMaterialParams),
            SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            dst_material.uniform_buffer_.buffer,
            dst_material.uniform_buffer_.memory,
            std::source_location::current());

        glsl::PbrMaterialParams ubo{};
        ubo.base_color_factor = sec.base_color;
        ubo.emissive_factor   = glm::vec3(0.0f);
        ubo.emissive_color    = glm::vec3(0.0f);
        ubo.specular_factor   = glm::vec3(1.0f);
        ubo.specular_color    = glm::vec3(1.0f);
        ubo.glossiness_factor = 1.0f;
        ubo.metallic_roughness_specular_factor = 1.0f;
        ubo.metallic_factor   = sec.metallic;
        ubo.roughness_factor  = sec.roughness;
        ubo.alpha_cutoff      = 0.1f;
        ubo.mip_count         = 11;
        ubo.normal_scale      = 1.0f;
        ubo.uv_set_flags      = glm::vec4(0.0f);
        ubo.exposure          = 1.0f;
        ubo.occlusion_strength = 1.0f;
        ubo.tonemap_type      = TONEMAP_DEFAULT;
        ubo.material_features = FEATURE_MATERIAL_METALLICROUGHNESS;
        // NO FEATURE_HAS_BASE_COLOR_MAP: baked textures are VT-only
        // (cpu_pixels, no full-res GPU image) — the forward pass renders
        // the flat section colour; the cluster pass samples the VT pool.
        ubo.material_features |= FEATURE_MATERIAL_ALPHA_MASK;

        device->updateBufferMemory(
            dst_material.uniform_buffer_.memory, sizeof(ubo), &ubo);

        // ECS dedup identity — .rwobj textures carry source paths, so two
        // placements of the same baked object intern to the same ids.
        captureMaterialDesc(
            dst_material, ubo, drawable_object->textures_, ref_name);
    }

    // ── GPU buffers + views (same layout as the FBX path) ─────────────
    drawable_object->meshes_.resize(1);
    drawable_object->buffers_.resize(2);
    drawable_object->buffer_views_.resize(4);
    auto& drawable_mesh = drawable_object->meshes_[0];
    auto& vertex_buffer = drawable_object->buffers_[0];
    auto& indice_buffer = drawable_object->buffers_[1];

    renderer::Helper::createBuffer(
        device,
        SET_4_FLAG_BITS(
            BufferUsage,
            VERTEX_BUFFER_BIT,
            STORAGE_BUFFER_BIT,
            SHADER_DEVICE_ADDRESS_BIT,
            ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        vertex_buffer.buffer,
        vertex_buffer.memory,
        std::source_location::current(),
        cpu_mesh.vertex_data_ptr->size() * sizeof(helper::VertexStruct),
        cpu_mesh.vertex_data_ptr->data());

    const bool use_16bits_index = vtx_count < 65536;
    uint32_t index_bytes_count = 4;
    auto index_type = renderer::IndexType::UINT32;
    if (use_16bits_index) {
        std::vector<uint16_t> indices_16(index_count);
        for (uint32_t i = 0; i < index_count; ++i) {
            indices_16[i] = static_cast<uint16_t>(md.indices[i]);
        }
        renderer::Helper::createBuffer(
            device,
            SET_4_FLAG_BITS(
                BufferUsage,
                INDEX_BUFFER_BIT,
                STORAGE_BUFFER_BIT,
                SHADER_DEVICE_ADDRESS_BIT,
                ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
            indice_buffer.buffer,
            indice_buffer.memory,
            std::source_location::current(),
            indices_16.size() * 2,
            indices_16.data());
        index_bytes_count = 2;
        index_type = renderer::IndexType::UINT16;
    } else {
        renderer::Helper::createBuffer(
            device,
            SET_4_FLAG_BITS(
                BufferUsage,
                INDEX_BUFFER_BIT,
                STORAGE_BUFFER_BIT,
                SHADER_DEVICE_ADDRESS_BIT,
                ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
            indice_buffer.buffer,
            indice_buffer.memory,
            std::source_location::current(),
            md.indices.size() * 4,
            md.indices.data());
    }

    const int pos_view_idx = 0, normal_view_idx = 1,
              uv_view_idx = 2, indice_view_idx = 3;
    drawable_object->buffer_views_[pos_view_idx].buffer_idx = 0;
    drawable_object->buffer_views_[pos_view_idx].offset = 0;
    drawable_object->buffer_views_[pos_view_idx].range =
        cpu_mesh.vertex_data_ptr->size() * sizeof(helper::VertexStruct);
    drawable_object->buffer_views_[pos_view_idx].stride =
        sizeof(helper::VertexStruct);

    drawable_object->buffer_views_[normal_view_idx] =
        drawable_object->buffer_views_[pos_view_idx];
    drawable_object->buffer_views_[normal_view_idx].offset = sizeof(glm::vec3);

    drawable_object->buffer_views_[uv_view_idx] =
        drawable_object->buffer_views_[pos_view_idx];
    drawable_object->buffer_views_[uv_view_idx].offset = 2 * sizeof(glm::vec3);

    drawable_object->buffer_views_[indice_view_idx].buffer_idx = 1;
    drawable_object->buffer_views_[indice_view_idx].offset = 0;
    drawable_object->buffer_views_[indice_view_idx].range =
        index_count * index_bytes_count;
    drawable_object->buffer_views_[indice_view_idx].stride = index_bytes_count;

    // ── Mesh bbox + CPU position table (collision / selection) ────────
    for (const auto& p : md.positions) {
        drawable_mesh.bbox_min_ = glm::min(drawable_mesh.bbox_min_, p);
        drawable_mesh.bbox_max_ = glm::max(drawable_mesh.bbox_max_, p);
    }
    {
        auto positions = std::make_shared<std::vector<glm::vec3>>(
            md.positions.begin(), md.positions.end());
        drawable_mesh.vertex_position_ = std::move(positions);
    }

    // ── Primitives: one per section ────────────────────────────────────
    drawable_mesh.primitives_.resize(num_sections);
    for (size_t si = 0; si < num_sections; ++si) {
        const auto& sec = md.sections[si];
        auto& primitive_info = drawable_mesh.primitives_[si];
        primitive_info.tag_.restart_enable = false;
        primitive_info.material_idx_ = static_cast<int32_t>(si);
        primitive_info.tag_.topology = static_cast<uint32_t>(
            renderer::PrimitiveTopology::TRIANGLE_LIST);
        primitive_info.tag_.has_texcoord_0 = true;
        primitive_info.tag_.has_normal = true;
        primitive_info.tag_.double_sided = false;

        uint32_t dst_binding = 0;
        engine::renderer::VertexInputBindingDescription binding = {};
        engine::renderer::VertexInputAttributeDescription attribute = {};

        // position
        binding.binding = dst_binding;
        binding.stride = sizeof(helper::VertexStruct);
        binding.input_rate = renderer::VertexInputRate::VERTEX;
        primitive_info.binding_descs_.push_back(binding);
        attribute.buffer_view = pos_view_idx;
        attribute.binding = dst_binding;
        attribute.offset = 0;
        attribute.buffer_offset = 0;
        attribute.location = VINPUT_POSITION;
        attribute.format = engine::renderer::Format::R32G32B32_SFLOAT;
        primitive_info.attribute_descs_.push_back(attribute);
        dst_binding++;

        // normal
        binding.binding = dst_binding;
        primitive_info.binding_descs_.push_back(binding);
        attribute.buffer_view = normal_view_idx;
        attribute.binding = dst_binding;
        attribute.offset = 0;
        attribute.buffer_offset = sizeof(glm::vec3);
        attribute.location = VINPUT_NORMAL;
        attribute.format = engine::renderer::Format::R32G32B32_SFLOAT;
        primitive_info.attribute_descs_.push_back(attribute);
        dst_binding++;

        // uv
        binding.binding = dst_binding;
        primitive_info.binding_descs_.push_back(binding);
        attribute.buffer_view = uv_view_idx;
        attribute.binding = dst_binding;
        attribute.offset = 0;
        attribute.buffer_offset = 2 * sizeof(glm::vec3);
        attribute.location = VINPUT_TEXCOORD0;
        attribute.format = engine::renderer::Format::R32G32_SFLOAT;
        primitive_info.attribute_descs_.push_back(attribute);
        dst_binding++;

        // indices — no baked LODs: every LOD slot points at the same
        // full-resolution section range.
        primitive_info.index_desc_.resize(helper::c_num_lods + 1);
        for (uint32_t i_lod = 0; i_lod < helper::c_num_lods + 1; i_lod++) {
            primitive_info.index_desc_[i_lod].buffer_view = indice_view_idx;
            primitive_info.index_desc_[i_lod].offset =
                sec.first_index * index_bytes_count;
            primitive_info.index_desc_[i_lod].index_type = index_type;
            primitive_info.index_desc_[i_lod].index_count = sec.index_count;
        }

        // Per-primitive bbox from the section's own vertices.
        glm::vec3 pmin(std::numeric_limits<float>::max());
        glm::vec3 pmax(std::numeric_limits<float>::lowest());
        const uint32_t sec_end =
            std::min(sec.first_index + sec.index_count, index_count);
        for (uint32_t ii = sec.first_index; ii < sec_end; ++ii) {
            const uint32_t vi = md.indices[ii];
            if (vi < vtx_count) {
                pmin = glm::min(pmin, md.positions[vi]);
                pmax = glm::max(pmax, md.positions[vi]);
            }
        }
        primitive_info.bbox_min_ = pmin;
        primitive_info.bbox_max_ = pmax;

        // CPU-side per-primitive index companion.  The collision-mesh
        // builder (CollisionMesh::buildFromDrawablePrimitive) iterates
        // `vertex_indices_` and SILENTLY SKIPS primitives whose pointer
        // is null — without this, every baked .rwobj placement dropped
        // out of the collision bake with reason=null_vertex_indices and
        // the bake produced an empty world.  Mirrors the FBX loader's
        // capture (see the part_idx_list block in loadFbxModel).
        if (sec_end > sec.first_index) {
            auto idx_list = std::make_shared<std::vector<int32_t>>();
            idx_list->reserve(sec_end - sec.first_index);
            for (uint32_t ii = sec.first_index; ii < sec_end; ++ii) {
                idx_list->push_back(static_cast<int32_t>(md.indices[ii]));
            }
            primitive_info.vertex_indices_ = std::move(idx_list);
        }

        primitive_info.generateHash();
    }

    // ── Cluster sidecar (Nanite-lite GPU culling path) ─────────────────
    helper::buildClusterMesh(cpu_mesh, drawable_mesh.cluster_mesh_);
    {
        // cluster → primitive map: sections are contiguous FACE ranges,
        // so the cluster's first face index resolves its section.
        std::vector<uint32_t> sec_face_start(num_sections + 1, 0);
        for (size_t si = 0; si < num_sections; ++si) {
            sec_face_start[si] = md.sections[si].first_index / 3;
        }
        sec_face_start[num_sections] = index_count / 3;
        drawable_mesh.cluster_prim_map_.clear();
        for (const auto& cluster : drawable_mesh.cluster_mesh_.clusters) {
            uint32_t prim_idx = 0;
            if (!cluster.face_indices.empty()) {
                const uint32_t first_face = cluster.face_indices[0];
                for (uint32_t p = 0; p + 1 < sec_face_start.size(); ++p) {
                    if (first_face >= sec_face_start[p] &&
                        first_face < sec_face_start[p + 1]) {
                        prim_idx = p;
                        break;
                    }
                }
            }
            drawable_mesh.cluster_prim_map_.push_back(prim_idx);
        }
    }

    // ── Node + scene (geometry is baked world-space → identity node) ──
    drawable_object->nodes_.resize(1);
    auto& node = drawable_object->nodes_[0];
    node.name_ = ref_name.empty() ? std::string("rwobj") : ref_name;
    node.mesh_idx_ = 0;
    node.matrix_ = glm::mat4(1.0f);
    drawable_object->scenes_.resize(1);
    drawable_object->scenes_[0].nodes_.resize(1);
    drawable_object->scenes_[0].nodes_[0] = 0;
    drawable_object->default_scene_ = 0;

    for (auto& scene : drawable_object->scenes_) {
        calculateBbox(drawable_object, scene.nodes_[0], glm::mat4(1.0f),
                      scene.bbox_min_, scene.bbox_max_);
    }

    drawable_object->update(
        device, 0, 0.0f, drawable_object->m_use_local_matrix_only_);

    setupRaytracing(drawable_object);

    // ── Indirect draw buffer (same as the FBX path) ───────────────────
    uint32_t num_prims = 0;
    for (auto& mesh : drawable_object->meshes_) {
        for (auto& prim : mesh.primitives_) {
            prim.indirect_draw_cmd_ofs_ =
                num_prims * sizeof(renderer::DrawIndexedIndirectCommand) +
                INDIRECT_DRAW_BUF_OFS;
            num_prims++;
        }
    }
    std::vector<uint32_t> indirect_draw_cmd_buffer(
        sizeof(renderer::DrawIndexedIndirectCommand) / sizeof(uint32_t) *
            num_prims + 1);
    auto indirect_draw_buf =
        reinterpret_cast<renderer::DrawIndexedIndirectCommand*>(
            indirect_draw_cmd_buffer.data() + 1);
    indirect_draw_cmd_buffer[0] = 0;
    uint32_t prim_idx = 0;
    for (const auto& mesh : drawable_object->meshes_) {
        for (const auto& prim : mesh.primitives_) {
            indirect_draw_buf[prim_idx].first_index = 0;
            indirect_draw_buf[prim_idx].first_instance = 0;
            indirect_draw_buf[prim_idx].index_count =
                static_cast<uint32_t>(prim.index_desc_[0].index_count);
            indirect_draw_buf[prim_idx].instance_count = 0;
            indirect_draw_buf[prim_idx].vertex_offset = 0;
            prim_idx++;
        }
    }
    renderer::Helper::createBuffer(
        device,
        SET_2_FLAG_BITS(BufferUsage, INDIRECT_BUFFER_BIT, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        drawable_object->indirect_draw_cmd_.buffer,
        drawable_object->indirect_draw_cmd_.memory,
        std::source_location::current(),
        indirect_draw_cmd_buffer.size() * sizeof(uint32_t),
        indirect_draw_cmd_buffer.data());
    drawable_object->num_prims_ = num_prims;

    std::cout << "[rwobj] loaded '" << ref_name << "' ("
              << vtx_count << " verts, " << index_count / 3 << " tris, "
              << num_sections << " section(s), "
              << md.textures.size() << " texture(s))" << std::endl;

    return drawable_object;
}

// ─── loadRwCharacter ───────────────────────────────────────────────────────
// Builds ONE skinned DrawableData from a baked character group (see the
// header doc).  Mirrors loadRwObjModel for geometry/material/texture and the
// glTF loader's setupNodes/setupSkins/setupAnimation for the rig — all sourced
// from raw data so no source model is needed at runtime.  Flows through the
// same Phase-3 (updateDescriptorSets builds skin sets from skins_; pipeline
// creation selects "_SKIN" from tag_.has_skin_set_0).
std::shared_ptr<ego::DrawableData> DrawableObject::loadRwCharacter(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& input_filename) {
    namespace fs = std::filesystem;

    const fs::path manifest(input_filename);
    const fs::path group_dir = manifest.parent_path();
    const std::string ref_name = manifest.stem().string();

    // ── Skeleton ──────────────────────────────────────────────────────
    std::vector<helper::RwHierNode> hier;
    if (!helper::loadRwHier((group_dir / "hierarchy.rwhier").string(), hier) ||
        hier.empty()) {
        std::cout << "[rwchar] missing/empty hierarchy for '" << ref_name
                  << "'" << std::endl;
        return nullptr;
    }

    // ── Baked objects: mesh_ordinal → objects/NNN_*.rwgeo ─────────────
    std::vector<std::pair<int, std::string>> ordinal_geo;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(group_dir / "objects", ec)) {
        if (e.path().extension() != ".rwgeo") continue;
        // Content-Browser "Enabled" toggle: a sibling .disabled marker
        // removes the sub-mesh from the loaded character entirely —
        // it never reaches the forward pass, CSM, or either RT shadow
        // path (the RT skin capture runs in the same load loop).
        {
            std::error_code dec;
            if (fs::exists(e.path().string() + ".disabled", dec)) {
                std::cout << "[rwchar] sub-mesh disabled, skipping: "
                          << e.path().filename().string() << std::endl;
                continue;
            }
        }
        int ord = -1;
        if (std::sscanf(e.path().filename().string().c_str(), "%d_", &ord)
                == 1 && ord >= 0)
            ordinal_geo.emplace_back(ord, e.path().string());
    }
    if (ordinal_geo.empty()) {
        std::cout << "[rwchar] no baked objects for '" << ref_name << "'"
                  << std::endl;
        return nullptr;
    }
    std::sort(ordinal_geo.begin(), ordinal_geo.end());

    auto drawable_object = std::make_shared<ego::DrawableData>(device);
    drawable_object->m_use_local_matrix_only_ = false;

    // Decompose a node-local matrix into TRS so animation channels (which set
    // translation_/rotation_/scale_) compose cleanly; matrix_ stays identity.
    auto decomposeTRS = [](const glm::mat4& m, glm::vec3& t, glm::quat& r,
                           glm::vec3& s) {
        t = glm::vec3(m[3]);
        const glm::vec3 c0(m[0]), c1(m[1]), c2(m[2]);
        s = glm::vec3(glm::length(c0), glm::length(c1), glm::length(c2));
        const glm::mat3 rot(
            s.x > 1e-8f ? glm::vec3(c0 / s.x) : glm::vec3(1, 0, 0),
            s.y > 1e-8f ? glm::vec3(c1 / s.y) : glm::vec3(0, 1, 0),
            s.z > 1e-8f ? glm::vec3(c2 / s.z) : glm::vec3(0, 0, 1));
        r = glm::quat_cast(rot);
    };

    drawable_object->nodes_.resize(hier.size());
    for (size_t i = 0; i < hier.size(); ++i) {
        auto& n = drawable_object->nodes_[i];
        n.name_ = hier[i].name;
        n.parent_idx_ = hier[i].parent;
        n.matrix_ = glm::mat4(1.0f);
        decomposeTRS(hier[i].local, n.translation_, n.rotation_, n.scale_);
    }
    std::vector<int32_t> roots;
    for (size_t i = 0; i < hier.size(); ++i) {
        const int p = hier[i].parent;
        if (p >= 0 && p < (int)hier.size())
            drawable_object->nodes_[p].child_idx_.push_back((int32_t)i);
        else
            roots.push_back((int32_t)i);
    }
    drawable_object->scenes_.resize(1);
    drawable_object->scenes_[0].nodes_ = roots;
    drawable_object->default_scene_ = 0;

    // Per-character VT texture cache (canonical .rwtex path → TextureInfo).
    std::unordered_map<std::string, renderer::TextureInfo> tex_gpu_cache;

    for (const auto& [ordinal, geo_path] : ordinal_geo) {
        int owner_node = -1;
        for (size_t i = 0; i < hier.size(); ++i)
            if (hier[i].mesh_ordinal == ordinal) { owner_node = (int)i; break; }

        helper::ModelPreviewData md;
        std::vector<std::string> tex_paths;
        if (!helper::loadRwGeo(geo_path, md, &tex_paths,
                               /*decode_textures=*/false) ||
            md.positions.empty() || md.indices.empty() || md.sections.empty())
            continue;
        helper::dedupModelVertices(md);

        const uint32_t vtx_count   = (uint32_t)md.positions.size();
        const uint32_t index_count = (uint32_t)md.indices.size();
        const bool skinned =
            md.joints.size()  == md.positions.size() &&
            md.weights.size() == md.positions.size() &&
            !md.skin_joint_nodes.empty();
        // 8-bone skinning debug: second skin set baked by the auto-rig.
        const bool skinned8 = skinned &&
            md.joints1.size()  == md.positions.size() &&
            md.weights1.size() == md.positions.size();

        // CPU interleaved vertex buffer (pos/normal/uv).
        helper::Mesh cpu_mesh;
        cpu_mesh.vertex_data_ptr->resize(vtx_count);
        for (uint32_t i = 0; i < vtx_count; ++i) {
            auto& v = cpu_mesh.vertex_data_ptr->at(i);
            v.position = md.positions[i];
            v.normal = i < md.normals.size() ? md.normals[i]
                                             : glm::vec3(0, 1, 0);
            v.uv = i < md.uvs.size() ? md.uvs[i] : glm::vec2(0);
        }
        cpu_mesh.faces_ptr->reserve(index_count / 3);
        for (uint32_t i = 0; i + 2 < index_count; i += 3)
            cpu_mesh.faces_ptr->emplace_back(
                md.indices[i], md.indices[i + 1], md.indices[i + 2]);

        // GPU buffers for this object (vertex + index [+ joints + weights]).
        const int vbuf = (int)drawable_object->buffers_.size();
        drawable_object->buffers_.emplace_back();   // vertex
        drawable_object->buffers_.emplace_back();   // index
        renderer::Helper::createBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
            drawable_object->buffers_[vbuf].buffer,
            drawable_object->buffers_[vbuf].memory,
            std::source_location::current(),
            vtx_count * sizeof(helper::VertexStruct),
            cpu_mesh.vertex_data_ptr->data());

        const bool use_16 = vtx_count < 65536;
        const uint32_t ibytes = use_16 ? 2u : 4u;
        const auto itype = use_16 ? renderer::IndexType::UINT16
                                  : renderer::IndexType::UINT32;
        if (use_16) {
            std::vector<uint16_t> idx16(index_count);
            for (uint32_t i = 0; i < index_count; ++i)
                idx16[i] = (uint16_t)md.indices[i];
            renderer::Helper::createBuffer(
                device, SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
                drawable_object->buffers_[vbuf + 1].buffer,
                drawable_object->buffers_[vbuf + 1].memory,
                std::source_location::current(), idx16.size() * 2,
                idx16.data());
        } else {
            renderer::Helper::createBuffer(
                device, SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
                drawable_object->buffers_[vbuf + 1].buffer,
                drawable_object->buffers_[vbuf + 1].memory,
                std::source_location::current(), md.indices.size() * 4,
                md.indices.data());
        }

        int jbuf = -1, wbuf = -1, jbuf1 = -1, wbuf1 = -1;
        if (skinned) {
            jbuf = (int)drawable_object->buffers_.size();
            drawable_object->buffers_.emplace_back();   // joints
            drawable_object->buffers_.emplace_back();   // weights
            wbuf = jbuf + 1;
            renderer::Helper::createBuffer(
                device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
                drawable_object->buffers_[jbuf].buffer,
                drawable_object->buffers_[jbuf].memory,
                std::source_location::current(),
                md.joints.size() * sizeof(glm::u16vec4), md.joints.data());
            renderer::Helper::createBuffer(
                device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
                drawable_object->buffers_[wbuf].buffer,
                drawable_object->buffers_[wbuf].memory,
                std::source_location::current(),
                md.weights.size() * sizeof(glm::vec4), md.weights.data());
        }
        if (skinned8) {
            jbuf1 = (int)drawable_object->buffers_.size();
            drawable_object->buffers_.emplace_back();   // joints set 1
            drawable_object->buffers_.emplace_back();   // weights set 1
            wbuf1 = jbuf1 + 1;
            renderer::Helper::createBuffer(
                device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
                drawable_object->buffers_[jbuf1].buffer,
                drawable_object->buffers_[jbuf1].memory,
                std::source_location::current(),
                md.joints1.size() * sizeof(glm::u16vec4), md.joints1.data());
            renderer::Helper::createBuffer(
                device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
                drawable_object->buffers_[wbuf1].buffer,
                drawable_object->buffers_[wbuf1].memory,
                std::source_location::current(),
                md.weights1.size() * sizeof(glm::vec4), md.weights1.data());
        }

        // Buffer views.
        const int vbase = (int)drawable_object->buffer_views_.size();
        drawable_object->buffer_views_.resize(
            vbase + (skinned8 ? 8 : (skinned ? 6 : 4)));
        const int pos_v = vbase + 0, nrm_v = vbase + 1, uv_v = vbase + 2,
                  idx_v = vbase + 3, jnt_v = vbase + 4, wgt_v = vbase + 5,
                  jnt1_v = vbase + 6, wgt1_v = vbase + 7;
        {
            auto& pv = drawable_object->buffer_views_[pos_v];
            pv.buffer_idx = vbuf; pv.offset = 0;
            pv.range = vtx_count * sizeof(helper::VertexStruct);
            pv.stride = sizeof(helper::VertexStruct);
            drawable_object->buffer_views_[nrm_v] = pv;
            drawable_object->buffer_views_[nrm_v].offset = sizeof(glm::vec3);
            drawable_object->buffer_views_[uv_v] = pv;
            drawable_object->buffer_views_[uv_v].offset = 2 * sizeof(glm::vec3);
            auto& iv = drawable_object->buffer_views_[idx_v];
            iv.buffer_idx = vbuf + 1; iv.offset = 0;
            iv.range = index_count * ibytes; iv.stride = ibytes;
            if (skinned) {
                auto& jv = drawable_object->buffer_views_[jnt_v];
                jv.buffer_idx = jbuf; jv.offset = 0;
                jv.range = vtx_count * sizeof(glm::u16vec4);
                jv.stride = sizeof(glm::u16vec4);
                auto& wv = drawable_object->buffer_views_[wgt_v];
                wv.buffer_idx = wbuf; wv.offset = 0;
                wv.range = vtx_count * sizeof(glm::vec4);
                wv.stride = sizeof(glm::vec4);
            }
            if (skinned8) {
                auto& jv1 = drawable_object->buffer_views_[jnt1_v];
                jv1.buffer_idx = jbuf1; jv1.offset = 0;
                jv1.range = vtx_count * sizeof(glm::u16vec4);
                jv1.stride = sizeof(glm::u16vec4);
                auto& wv1 = drawable_object->buffer_views_[wgt1_v];
                wv1.buffer_idx = wbuf1; wv1.offset = 0;
                wv1.range = vtx_count * sizeof(glm::vec4);
                wv1.stride = sizeof(glm::vec4);
            }
        }

        // Skin (one per skinned object); joints_ index into nodes_ directly
        // because skin_joint_nodes were baked as hierarchy.rwhier indices.
        int skin_index = -1;
        if (skinned) {
            skin_index = (int)drawable_object->skins_.size();
            drawable_object->skins_.emplace_back();
            auto& sk = drawable_object->skins_.back();
            sk.name_ = ref_name + "_skin";
            sk.skeleton_root_ = -1;
            sk.joints_.assign(md.skin_joint_nodes.begin(),
                              md.skin_joint_nodes.end());
            sk.inverse_bind_matrices_ = md.skin_inverse_bind;
            renderer::Helper::createBuffer(
                device, SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                HOST_COHERENT_BIT),
                0, sk.joints_buffer_.buffer, sk.joints_buffer_.memory,
                std::source_location::current(),
                sk.inverse_bind_matrices_.size() * sizeof(glm::mat4),
                sk.inverse_bind_matrices_.data());

            // Bind-pose snapshot for the RT-shadow skeleton path — the
            // auto-rig MeshData already has everything CPU-side, so this
            // is a straight copy (a few MB per character).  ClusterRenderer
            // CPU-skins it into world space each frame so the character
            // casts shadows in both RT shadow modes.
            // APPEND across the group's skinned sub-meshes (one .rwgeo
            // per sub-mesh — the loop visits each): indices are rebased
            // onto the concatenated arrays.  All sub-meshes of a baked
            // character share the hierarchy-indexed joint set, so one
            // joint-matrix array (skins_[0]) skins the whole snapshot.
            if (!drawable_object->rt_skin_source_) {
                drawable_object->rt_skin_source_ =
                    std::make_shared<RtSkinSource>();
            }
            auto& rt_src = *drawable_object->rt_skin_source_;
            const uint32_t rt_vbase = (uint32_t)rt_src.positions.size();
            rt_src.positions.insert(rt_src.positions.end(),
                                    md.positions.begin(),
                                    md.positions.end());
            rt_src.joints.insert(rt_src.joints.end(),
                                 md.joints.begin(), md.joints.end());
            rt_src.weights.insert(rt_src.weights.end(),
                                  md.weights.begin(), md.weights.end());
            if (skinned8) {
                // Set-1 arrays must stay PARALLEL to positions across the
                // whole concatenation — pad earlier meshes' gap if this
                // is the first skinned8 mesh.
                rt_src.joints1.resize(rt_vbase, glm::u16vec4(0));
                rt_src.weights1.resize(rt_vbase, glm::vec4(0.0f));
                rt_src.joints1.insert(rt_src.joints1.end(),
                                      md.joints1.begin(),
                                      md.joints1.end());
                rt_src.weights1.insert(rt_src.weights1.end(),
                                       md.weights1.begin(),
                                       md.weights1.end());
            } else if (!rt_src.joints1.empty()) {
                rt_src.joints1.resize(rt_src.positions.size(),
                                      glm::u16vec4(0));
                rt_src.weights1.resize(rt_src.positions.size(),
                                       glm::vec4(0.0f));
            }
            // Indices per SECTION, skipping non-opaque ones (translucent
            // shadow-catcher / glass sections must not cast — same rule
            // as the glTF capture and the cluster RT path).
            rt_src.indices.reserve(rt_src.indices.size() +
                                   md.indices.size());
            for (const auto& sec : md.sections) {
                if (sec.base_color.a < 0.5f) continue;
                const uint32_t iend = std::min(
                    (uint32_t)md.indices.size(),
                    sec.first_index + sec.index_count);
                for (uint32_t k = sec.first_index; k < iend; ++k) {
                    rt_src.indices.push_back(md.indices[k] + rt_vbase);
                }
            }
        }

        // Textures (VT-only) with global indices.
        const size_t tex_base = drawable_object->textures_.size();
        drawable_object->textures_.resize(tex_base + md.textures.size());
        for (size_t ti = 0; ti < md.textures.size() && ti < tex_paths.size();
             ++ti) {
            auto& dst = drawable_object->textures_[tex_base + ti];
            std::string key = fs::path(tex_paths[ti])
                                  .lexically_normal().generic_string();
            for (auto& c : key) c = (char)std::tolower((unsigned char)c);
            auto cit = tex_gpu_cache.find(key);
            if (cit != tex_gpu_cache.end()) { dst = cit->second; continue; }
            helper::RwTexBaked tb;
            if (!helper::readRwTexBaked(tex_paths[ti], tb) ||
                tb.w <= 0 || tb.h <= 0)
                continue;
            dst.size = glm::uvec3((uint32_t)tb.w, (uint32_t)tb.h, 1u);
            dst.linear = false;
            dst.source_filename_ = tex_paths[ti];
            dst.vt_bc7_tiles = tb.bc7_tiles;
            if (!dst.vt_bc7_tiles && tb.preview_w == tb.w &&
                tb.preview_h == tb.h && !tb.preview_rgba.empty())
                dst.cpu_pixels = std::make_shared<std::vector<uint8_t>>(
                    tb.preview_rgba.begin(), tb.preview_rgba.end());
            tex_gpu_cache.emplace(key, dst);
        }

        // Materials (one per section), global texture indices.
        const size_t mat_base = drawable_object->materials_.size();
        drawable_object->materials_.resize(mat_base + md.sections.size());
        for (size_t si = 0; si < md.sections.size(); ++si) {
            const auto& sec = md.sections[si];
            auto& mat = drawable_object->materials_[mat_base + si];
            mat.name_ = ref_name + "_o" + std::to_string(ordinal) + "_s" +
                        std::to_string(si);
            if (sec.tex_index >= 0)
                mat.base_color_idx_ = (int)tex_base + sec.tex_index;
            if (sec.nrm_index >= 0)
                mat.normal_idx_ = (int)tex_base + sec.nrm_index;
            if (sec.mr_index >= 0)
                mat.metallic_roughness_idx_ = (int)tex_base + sec.mr_index;
            mat.alpha_cutoff_ = 0.1f;
            mat.alpha_mask_ = true;
            mat.alpha_mode_ = ego::AlphaMode::Mask;
            device->createBuffer(
                sizeof(glsl::PbrMaterialParams),
                SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                HOST_COHERENT_BIT),
                0, mat.uniform_buffer_.buffer, mat.uniform_buffer_.memory,
                std::source_location::current());
            glsl::PbrMaterialParams ubo{};
            ubo.base_color_factor = sec.base_color;
            ubo.specular_factor = glm::vec3(1.0f);
            ubo.specular_color = glm::vec3(1.0f);
            ubo.glossiness_factor = 1.0f;
            ubo.metallic_roughness_specular_factor = 1.0f;
            ubo.metallic_factor = sec.metallic;
            ubo.roughness_factor = sec.roughness;
            ubo.alpha_cutoff = 0.1f;
            ubo.mip_count = 11;
            ubo.normal_scale = 1.0f;
            ubo.uv_set_flags = glm::vec4(0.0f);
            ubo.exposure = 1.0f;
            ubo.occlusion_strength = 1.0f;
            ubo.tonemap_type = TONEMAP_DEFAULT;
            ubo.material_features = FEATURE_MATERIAL_METALLICROUGHNESS;
            ubo.material_features |= FEATURE_MATERIAL_ALPHA_MASK;
            device->updateBufferMemory(mat.uniform_buffer_.memory,
                                       sizeof(ubo), &ubo);
            // ECS dedup identity — baked character sections.
            captureMaterialDesc(
                mat, ubo, drawable_object->textures_, ref_name);
        }

        // Mesh + primitives.
        const int mesh_index = (int)drawable_object->meshes_.size();
        drawable_object->meshes_.emplace_back();
        auto& mesh = drawable_object->meshes_[mesh_index];
        for (const auto& p : md.positions) {
            mesh.bbox_min_ = glm::min(mesh.bbox_min_, p);
            mesh.bbox_max_ = glm::max(mesh.bbox_max_, p);
        }
        mesh.vertex_position_ = std::make_shared<std::vector<glm::vec3>>(
            md.positions.begin(), md.positions.end());
        mesh.primitives_.resize(md.sections.size());
        for (size_t si = 0; si < md.sections.size(); ++si) {
            const auto& sec = md.sections[si];
            auto& prim = mesh.primitives_[si];
            prim.tag_.restart_enable = false;
            prim.material_idx_ = (int32_t)(mat_base + si);
            prim.tag_.topology =
                (uint32_t)renderer::PrimitiveTopology::TRIANGLE_LIST;
            prim.tag_.has_texcoord_0 = true;
            prim.tag_.has_normal = true;
            prim.tag_.double_sided = false;
            prim.tag_.has_skin_set_0 = skinned;
            prim.tag_.has_skin_set_1 = skinned8;   // 8-bone debug path

            uint32_t b = 0;
            renderer::VertexInputBindingDescription bd = {};
            renderer::VertexInputAttributeDescription ad = {};
            bd.input_rate = renderer::VertexInputRate::VERTEX;
            // position
            bd.binding = b; bd.stride = sizeof(helper::VertexStruct);
            prim.binding_descs_.push_back(bd);
            ad.buffer_view = pos_v; ad.binding = b; ad.offset = 0;
            ad.buffer_offset = 0; ad.location = VINPUT_POSITION;
            ad.format = renderer::Format::R32G32B32_SFLOAT;
            prim.attribute_descs_.push_back(ad); ++b;
            // normal
            bd.binding = b; bd.stride = sizeof(helper::VertexStruct);
            prim.binding_descs_.push_back(bd);
            ad.buffer_view = nrm_v; ad.binding = b; ad.offset = 0;
            ad.buffer_offset = sizeof(glm::vec3); ad.location = VINPUT_NORMAL;
            ad.format = renderer::Format::R32G32B32_SFLOAT;
            prim.attribute_descs_.push_back(ad); ++b;
            // uv
            bd.binding = b; bd.stride = sizeof(helper::VertexStruct);
            prim.binding_descs_.push_back(bd);
            ad.buffer_view = uv_v; ad.binding = b; ad.offset = 0;
            ad.buffer_offset = 2 * sizeof(glm::vec3);
            ad.location = VINPUT_TEXCOORD0;
            ad.format = renderer::Format::R32G32_SFLOAT;
            prim.attribute_descs_.push_back(ad); ++b;
            if (skinned) {
                // joints (u16vec4 → R16G16B16A16_UINT)
                bd.binding = b; bd.stride = sizeof(glm::u16vec4);
                prim.binding_descs_.push_back(bd);
                ad.buffer_view = jnt_v; ad.binding = b; ad.offset = 0;
                ad.buffer_offset = 0; ad.location = VINPUT_JOINTS_0;
                ad.format = renderer::Format::R16G16B16A16_UINT;
                prim.attribute_descs_.push_back(ad); ++b;
                // weights (vec4 → R32G32B32A32_SFLOAT)
                bd.binding = b; bd.stride = sizeof(glm::vec4);
                prim.binding_descs_.push_back(bd);
                ad.buffer_view = wgt_v; ad.binding = b; ad.offset = 0;
                ad.buffer_offset = 0; ad.location = VINPUT_WEIGHTS_0;
                ad.format = renderer::Format::R32G32B32A32_SFLOAT;
                prim.attribute_descs_.push_back(ad); ++b;
            }
            if (skinned8) {
                // second skin set (influences 4..7, 8-bone debug path)
                bd.binding = b; bd.stride = sizeof(glm::u16vec4);
                prim.binding_descs_.push_back(bd);
                ad.buffer_view = jnt1_v; ad.binding = b; ad.offset = 0;
                ad.buffer_offset = 0; ad.location = VINPUT_JOINTS_1;
                ad.format = renderer::Format::R16G16B16A16_UINT;
                prim.attribute_descs_.push_back(ad); ++b;
                bd.binding = b; bd.stride = sizeof(glm::vec4);
                prim.binding_descs_.push_back(bd);
                ad.buffer_view = wgt1_v; ad.binding = b; ad.offset = 0;
                ad.buffer_offset = 0; ad.location = VINPUT_WEIGHTS_1;
                ad.format = renderer::Format::R32G32B32A32_SFLOAT;
                prim.attribute_descs_.push_back(ad); ++b;
            }

            prim.index_desc_.resize(helper::c_num_lods + 1);
            for (uint32_t l = 0; l < helper::c_num_lods + 1; ++l) {
                prim.index_desc_[l].buffer_view = idx_v;
                prim.index_desc_[l].offset = sec.first_index * ibytes;
                prim.index_desc_[l].index_type = itype;
                prim.index_desc_[l].index_count = sec.index_count;
            }
            glm::vec3 pmin(std::numeric_limits<float>::max());
            glm::vec3 pmax(std::numeric_limits<float>::lowest());
            const uint32_t se =
                std::min(sec.first_index + sec.index_count, index_count);
            for (uint32_t ii = sec.first_index; ii < se; ++ii) {
                const uint32_t vi = md.indices[ii];
                if (vi < vtx_count) {
                    pmin = glm::min(pmin, md.positions[vi]);
                    pmax = glm::max(pmax, md.positions[vi]);
                }
            }
            prim.bbox_min_ = pmin; prim.bbox_max_ = pmax;
            prim.generateHash();
        }

        if (owner_node >= 0) {
            drawable_object->nodes_[owner_node].mesh_idx_ = mesh_index;
            drawable_object->nodes_[owner_node].skin_idx_ = skin_index;
        }
    }

    if (drawable_object->meshes_.empty()) return nullptr;

    // ── Animations from animation.rwanim → AnimChannelInfo samples ────
    {
        std::vector<helper::RwAnimClip> clips;
        if (helper::loadRwAnim((group_dir / "animation.rwanim").string(),
                               clips) && !clips.empty()) {
            drawable_object->animations_.resize(clips.size());
            for (size_t ci = 0; ci < clips.size(); ++ci) {
                auto& dst = drawable_object->animations_[ci];
                for (const auto& ch : clips[ci].channels) {
                    auto info = std::make_shared<ego::AnimChannelInfo>();
                    info->node_idx_ = (uint32_t)ch.node;
                    switch (ch.path) {
                    case helper::RwAnimPath::kTranslation:
                        info->type_ = ego::AnimChannelInfo::kTranslation;
                        break;
                    case helper::RwAnimPath::kScale:
                        info->type_ = ego::AnimChannelInfo::kScale;
                        break;
                    case helper::RwAnimPath::kRotation:
                    default:
                        info->type_ = ego::AnimChannelInfo::kRotation;
                        break;
                    }
                    info->samples_.reserve(ch.times.size());
                    for (size_t k = 0; k < ch.times.size() &&
                                       k < ch.values.size(); ++k)
                        info->samples_.emplace_back(ch.times[k], ch.values[k]);
                    dst.channels_.push_back(std::move(info));
                }
            }
        }
    }

    // Scene bbox.
    for (auto& scene : drawable_object->scenes_) {
        scene.bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
        scene.bbox_max_ = glm::vec3(std::numeric_limits<float>::lowest());
        for (auto root : scene.nodes_)
            calculateBbox(drawable_object, root, glm::mat4(1.0f),
                          scene.bbox_min_, scene.bbox_max_);
    }

    // Bind-pose hierarchy + first joint upload (skins_ gate runs updateJoints).
    drawable_object->update(device, 0, 0.0f, /*use_local_matrix_only=*/false);
    setupRaytracing(drawable_object);

    // ── Indirect draw buffer (same as the other native loaders) ───────
    uint32_t num_prims = 0;
    for (auto& mesh : drawable_object->meshes_)
        for (auto& prim : mesh.primitives_) {
            prim.indirect_draw_cmd_ofs_ =
                num_prims * sizeof(renderer::DrawIndexedIndirectCommand) +
                INDIRECT_DRAW_BUF_OFS;
            num_prims++;
        }
    std::vector<uint32_t> idc(
        sizeof(renderer::DrawIndexedIndirectCommand) / sizeof(uint32_t) *
            num_prims + 1);
    auto* cmds = reinterpret_cast<renderer::DrawIndexedIndirectCommand*>(
        idc.data() + 1);
    idc[0] = 0;
    uint32_t pi = 0;
    for (const auto& mesh : drawable_object->meshes_)
        for (const auto& prim : mesh.primitives_) {
            cmds[pi].first_index = 0;
            cmds[pi].first_instance = 0;
            cmds[pi].index_count = (uint32_t)prim.index_desc_[0].index_count;
            cmds[pi].instance_count = 0;
            cmds[pi].vertex_offset = 0;
            pi++;
        }
    renderer::Helper::createBuffer(
        device,
        SET_2_FLAG_BITS(BufferUsage, INDIRECT_BUFFER_BIT, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0,
        drawable_object->indirect_draw_cmd_.buffer,
        drawable_object->indirect_draw_cmd_.memory,
        std::source_location::current(),
        idc.size() * sizeof(uint32_t), idc.data());
    drawable_object->num_prims_ = num_prims;

    std::cout << "[rwchar] loaded '" << ref_name << "' ("
              << drawable_object->nodes_.size() << " nodes, "
              << drawable_object->meshes_.size() << " mesh(es), "
              << drawable_object->skins_.size() << " skin(s), "
              << drawable_object->animations_.size() << " clip(s))"
              << std::endl;
    return drawable_object;
}

} // namespace game_object
} // namespace engine
