#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <filesystem>

#include "helper/engine_helper.h"
#include "helper/bvh.h"
#include "helper/mesh_tool.h"
#include "game_object/drawable_object.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

// gltf
#include "tiny_gltf.h"
#include "helper/tiny_mtx2.h"

// fbx
#include "third_parties/fbx/ufbx.h"

static uint32_t num_draw_meshes = 0;
#define DEBUG_OUTPUT 0

namespace ego = engine::game_object;
namespace engine {

namespace {

size_t hashCombine(
    size_t seed,
    size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

size_t hashCombine(
    const uint32_t num_items,
    const float* items) {
    size_t hash = 0;

    // Quantize and hash position
    for (uint32_t i = 0; i < num_items; i++) {
        hash = hashCombine(hash, *(uint32_t*)(&items[i]));
    }
    return hash;
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
    std::shared_ptr<ego::DrawableData>& drawable_object) {

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

            dst_material.base_color_idx_ = src_material.pbrMetallicRoughness.baseColorTexture.index;
            dst_material.normal_idx_ = src_material.normalTexture.index;
            dst_material.metallic_roughness_idx_ = src_material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            dst_material.emissive_idx_ = src_material.emissiveTexture.index;
            dst_material.occlusion_idx_ = src_material.occlusionTexture.index;

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
            ubo.material_features = (src_material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0 ? (FEATURE_HAS_METALLIC_ROUGHNESS_MAP | FEATURE_HAS_METALLIC_CHANNEL) : 0) | FEATURE_MATERIAL_METALLICROUGHNESS;
            ubo.material_features |= (src_material.pbrMetallicRoughness.baseColorTexture.index >= 0 ? FEATURE_HAS_BASE_COLOR_MAP : 0);
            ubo.material_features |= (src_material.emissiveTexture.index >= 0 ? FEATURE_HAS_EMISSIVE_MAP : 0);
            ubo.material_features |= (src_material.occlusionTexture.index >= 0 ? FEATURE_HAS_OCCLUSION_MAP : 0);
            ubo.material_features |= (src_material.normalTexture.index >= 0 ? FEATURE_HAS_NORMAL_MAP : 0);
            ubo.tonemap_type = TONEMAP_DEFAULT;
            ubo.specular_factor = glm::vec3(1.0f, 1.0f, 1.0f);
            ubo.specular_color = glm::vec3(1.0f, 1.0f, 1.0f);
            ubo.specular_exponent = 1.0f;

            device->updateBufferMemory(dst_material.uniform_buffer_.memory, sizeof(ubo), &ubo);
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

            dst_tex.view = device->createImageView(
                dst_tex.image,
                renderer::ImageViewType::VIEW_2D,
                format,
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                std::source_location::current());
        }
    }
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
    std::shared_ptr<ego::DrawableData>& drawable_object) {

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
    }

    for (size_t i_tex = 0; i_tex < fbx_scene->textures.count; i_tex++) {
        auto& dst_tex = drawable_object->textures_[i_tex];
        const auto& src_tex = fbx_scene->textures[i_tex];

        glm::uvec3 size(0);
        auto format = renderer::Format::R8G8B8A8_UNORM;
        helper::createTextureImage(
            device, 
            src_tex->filename.data,
            format,
            texture_is_srgb[i_tex],
            dst_tex,
            std::source_location::current());
    }

    // Material
    {
        drawable_object->materials_.resize(fbx_scene->materials.count);
        for (size_t i_mat = 0; i_mat < fbx_scene->materials.count; i_mat++) {
            auto& dst_material = drawable_object->materials_[i_mat];
            const auto& src_material = fbx_scene->materials[i_mat];

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
                    assert(0);
                }
                else if (texture_string == "EmissiveColor") {
                    dst_material.emissive_idx_ = tex_id;
                }
                else if (texture_string == "Occlusion") {
                    dst_material.occlusion_idx_ = tex_id;
                }
                else {
                    assert(0);
                }
            }

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

            device->updateBufferMemory(dst_material.uniform_buffer_.memory, sizeof(ubo), &ubo);
        }
    }
}

static void setupMesh(
    const std::shared_ptr<renderer::Device>& device,
    const ufbx_abi ufbx_scene* fbx_scene,
    std::shared_ptr<ego::DrawableData>& drawable_object,
    const uint32_t& mesh_idx) {

    auto vertex_buffer_idx = mesh_idx * 2;
    auto indice_buffer_idx = mesh_idx * 2 + 1;

    const ufbx_mesh* src_mesh = fbx_scene->meshes[mesh_idx];
    auto& drawable_mesh = drawable_object->meshes_[mesh_idx];
    auto& vertex_buffer = drawable_object->buffers_[vertex_buffer_idx];
    auto& indice_buffer = drawable_object->buffers_[indice_buffer_idx];

    assert(src_mesh->num_faces == src_mesh->num_triangles);
    assert(src_mesh->num_indices == src_mesh->vertex_position.indices.count);
    assert(src_mesh->num_indices == src_mesh->vertex_normal.indices.count);
    assert(src_mesh->num_indices == src_mesh->vertex_uv.indices.count);

    std::vector<glm::vec3> vertex_position(src_mesh->num_vertices);
    for (auto i = 0; i < src_mesh->num_vertices; i++) {
        vertex_position[i].x = float_t(src_mesh->vertex_position[i].x);
        vertex_position[i].y = float_t(src_mesh->vertex_position[i].y);
        vertex_position[i].z = float_t(src_mesh->vertex_position[i].z);
    }

    std::unordered_map<size_t, uint32_t> vertex_map;
    std::vector<uint32_t> new_indices;
    std::vector<uint32_t> indice_match_table;
    new_indices.reserve(src_mesh->num_indices * 3);
    indice_match_table.reserve(src_mesh->num_indices);

    std::vector<float> vertex_data;
    vertex_data.reserve(20);

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

        ego::PrimitiveInfo primitive_info;
        primitive_info.bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
        primitive_info.bbox_max_ = glm::vec3(std::numeric_limits<float>::min());

        std::vector<int32_t> vertex_indices;
        vertex_indices.reserve(part.num_faces * 3);

        for (int i_face = 0; i_face < part.num_faces; i_face++) {
            const auto& face_indice = part.face_indices[i_face];
            const auto& face = src_mesh->faces[face_indice];
            assert(face.num_indices == 3);
            for (uint32_t i_vert = 0; i_vert < face.num_indices; i_vert++) {
                const auto src_vert_index = face.index_begin + i_vert;
                auto pos_indice = src_mesh->vertex_position.indices[src_vert_index];
                vertex_indices.push_back(pos_indice);
                const auto& position = src_mesh->vertex_position.values[pos_indice];
                auto position_packed = glm::vec3(position.x, position.y, position.z);

                primitive_info.bbox_min_ = glm::min(primitive_info.bbox_min_, position_packed);
                primitive_info.bbox_max_ = glm::max(primitive_info.bbox_max_, position_packed);

                vertex_data.clear();
                vertex_data.push_back(float(position.x));
                vertex_data.push_back(float(position.y));
                vertex_data.push_back(float(position.z));
                auto norm_indice = src_mesh->vertex_normal.indices[src_vert_index];
                const auto& normal = src_mesh->vertex_normal.values[norm_indice];
                vertex_data.push_back(float(normal.x));
                vertex_data.push_back(float(normal.y));
                vertex_data.push_back(float(normal.z));
                auto uv_indice = src_mesh->vertex_uv.indices[src_vert_index];
                const auto& uv = src_mesh->vertex_uv.values[uv_indice];
                vertex_data.push_back(float(uv.x));
                vertex_data.push_back(float(uv.y));
                auto hash_value =
                    hashCombine(
                        static_cast<uint32_t>(vertex_data.size()),
                        vertex_data.data());

                auto it = vertex_map.find(hash_value);
                if (it != vertex_map.end()) {
                    new_indices.push_back(it->second);
                }
                else {
                    auto new_index =
                        static_cast<uint32_t>(indice_match_table.size());
                    new_indices.push_back(new_index);
                    indice_match_table.push_back(src_vert_index);

                    vertex_map[hash_value] =
                        static_cast<uint32_t>(new_index);
                }
            }
        }

        if (create_bvh_tree) {
            bool debug_mode = false;

            std::cout << "mesh idx : " << mesh_idx
                << "/" << fbx_scene->meshes.count;

#if DEBUG_OUTPUT
            std::cout << 
                << ", mat part : " << i_part <<
                ", num tris : " << vertex_indices.size() / 3
                << std::endl;
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

        drawable_mesh.bbox_min_ =
            min(drawable_mesh.bbox_min_, primitive_info.bbox_min_);
        drawable_mesh.bbox_max_ =
            max(drawable_mesh.bbox_max_, primitive_info.bbox_max_);
        drawable_mesh.primitives_.push_back(primitive_info);
    }

    auto drawable_vertices =
        std::make_shared<std::vector<helper::VertexStruct>>();
    drawable_vertices->reserve(indice_match_table.size() * 3);
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

    assert(src_mesh->num_faces * 3 == src_mesh->num_indices);

    // create HLOD
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>
        lod_indice_info(helper::c_num_lods + 1);
    {
#if DEBUG_OUTPUT
        std::cout << "====================" << std::endl;
        std::cout << "mesh idx : " << mesh_idx
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
#if DEBUG_OUTPUT
            std::cout << "start lod : " << i_lod + 1 << std::endl;
#endif
            const auto& src_lod_indice_info = lod_indice_info[i_lod];
            auto& dst_lod_indice_info = lod_indice_info[i_lod + 1];
            dst_lod_indice_info.resize(src_mesh->material_parts.count);
            for (int i_part = 0; i_part < src_mesh->material_parts.count; i_part++) {
                int32_t src_face_start = src_lod_indice_info[i_part].first;
                int32_t src_face_count = src_lod_indice_info[i_part].second;
                int32_t target_face_count = std::max(int32_t(src_face_count * helper::c_target_lod_ratio), 1);

                helper::Mesh input_mesh_const;
                input_mesh_const.vertex_data_ptr = drawable_vertices;
                input_mesh_const.faces_ptr = std::make_shared<std::vector<helper::Face>>();
                input_mesh_const.faces_ptr->resize(src_face_count);
                for (int32_t i = 0; i < src_face_count; i++) {
                    int32_t src_face_idx = src_face_start + i;
                    input_mesh_const.faces_ptr->at(i) =
                        helper::Face(
                            new_indices[src_face_idx * 3 + 0],
                            new_indices[src_face_idx * 3 + 1],
                            new_indices[src_face_idx * 3 + 2]);
                }

                helper::Mesh mesh_lod =
                    helper::simplifyMeshActualButVeryBasic(
                        input_mesh_const,
                        target_face_count,
                        helper::c_sharp_edge_angle_threshold_degrees,
                        helper::c_normal_weight,
                        helper::c_uv_weight);

                uint32_t num_lod_vertex = uint32_t(mesh_lod.vertex_data_ptr->size());
                uint32_t vertex_index_offset = uint32_t(drawable_vertices->size());
                for (uint32_t i = 0; i < num_lod_vertex; i++) {
                    drawable_vertices->push_back(mesh_lod.vertex_data_ptr->at(i));
                }
                
                auto num_faces = uint32_t(mesh_lod.faces_ptr->size());
                for (uint32_t i = 0; i < num_faces; i++) {
                    auto idx0 = mesh_lod.faces_ptr->at(i).v_indices[0];
                    auto idx1 = mesh_lod.faces_ptr->at(i).v_indices[1];
                    auto idx2 = mesh_lod.faces_ptr->at(i).v_indices[2];
                    assert(idx0 < num_lod_vertex);
                    assert(idx1 < num_lod_vertex);
                    assert(idx2 < num_lod_vertex);
                    new_indices.push_back(vertex_index_offset + idx0);
                    new_indices.push_back(vertex_index_offset + idx1);
                    new_indices.push_back(vertex_index_offset + idx2);
                }

                dst_lod_indice_info[i_part].first = face_idx_offset;
                dst_lod_indice_info[i_part].second = num_faces;
                face_idx_offset += num_faces;
            }
        }
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

    if (has_bvh_trees) {
        drawable_mesh.vertex_position_ =
            std::make_shared<std::vector<glm::vec3>>(vertex_position);
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
    std::shared_ptr<ego::DrawableData>& drawable_object) {
    drawable_object->meshes_.resize(fbx_scene->meshes.count);
    drawable_object->buffers_.resize(fbx_scene->meshes.count * 2);
    drawable_object->buffer_views_.resize(fbx_scene->meshes.count * 4);
    for (int i_mesh = 0; i_mesh < fbx_scene->meshes.count; i_mesh++) {
        setupMesh(device, fbx_scene, drawable_object, i_mesh);
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

    // base color.
    auto& base_color_tex_view =
        material.base_color_idx_ < 0 ?
        black_tex :
        textures[material.base_color_idx_];

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ALBEDO_TEX_INDEX,
        texture_sampler,
        base_color_tex_view.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // normal.
    auto& specular_tex_view =
        material.specular_color_idx_ < 0 ?
        black_tex :
        textures[material.specular_color_idx_];

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SPECULAR_TEX_INDEX,
        texture_sampler,
        specular_tex_view.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // normal.
    auto& normal_tex_view =
        material.normal_idx_ < 0 ?
        black_tex :
        textures[material.normal_idx_];

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        NORMAL_TEX_INDEX,
        texture_sampler,
        normal_tex_view.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // metallic roughness.
    auto& metallic_roughness_tex =
        material.metallic_roughness_idx_ < 0 ?
        black_tex :
        textures[material.metallic_roughness_idx_];

    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        METAL_ROUGHNESS_TEX_INDEX,
        texture_sampler,
        metallic_roughness_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // emisive.
    auto& emissive_tex =
        material.emissive_idx_ < 0 ?
        black_tex :
        textures[material.emissive_idx_];


    renderer::Helper::addOneTexture(
        descriptor_writes,
        material.desc_set_,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        EMISSIVE_TEX_INDEX,
        texture_sampler,
        emissive_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // occlusion.
    auto& occlusion_tex =
        material.occlusion_idx_ < 0 ?
        white_tex :
        textures[material.occlusion_idx_];

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
    size_t& last_hash) {

    for (const auto& prim : mesh_info.primitives_) {
        const auto& attrib_list = prim.attribute_descs_;

        auto cur_hash = depth_only ? prim.getDepthonlyHash() : prim.getHash();
        if (cur_hash != last_hash) {
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::GRAPHICS,
                pipelines[cur_hash]);

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

        uint32_t cur_lod = std::min(0u, uint32_t(prim.index_desc_.size() - 1));
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

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT),
            drawable_pipeline_layout,
            &model_params,
            sizeof(model_params));

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
    size_t& last_hash) {
    if (node_idx >= 0) {
        const auto& node = drawable_object->nodes_[node_idx];
        if (node.mesh_idx_ >= 0) {
            glsl::ModelParams model_params{};
            model_params.model_mat = node.cached_matrix_;
            model_params.flip_uv_coord =
                (drawable_object->m_flip_u_ ? 0x01 : 0x00) |
                (drawable_object->m_flip_v_ ? 0x02 : 0x00);

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
                last_hash);

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
                last_hash);
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

static std::shared_ptr<renderer::PipelineLayout> createDrawablePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& material_desc_set_layout,
    const std::shared_ptr<renderer::DescriptorSetLayout>& skin_desc_set_layout) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, VERTEX_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::ModelParams);

    renderer::DescriptorSetLayoutList desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts[PBR_MATERIAL_PARAMS_SET] = material_desc_set_layout;
    desc_set_layouts[SKIN_PARAMS_SET] = skin_desc_set_layout;

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
    bool has_double_sided) {
    renderer::ShaderModuleList shader_modules(2);
    auto vert_feature_str = std::string(has_texcoord_0 ? "_TEX" : "") +
        (has_tangent ? "_TN" : (has_normals ? "_N" : ""));
    vert_feature_str = has_material ? vert_feature_str : "_NOMTL";
    auto frag_feature_str = vert_feature_str;
    if (has_skin_set_0) {
        vert_feature_str += "_SKIN";
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
    bool has_material) {
    renderer::ShaderModuleList shader_modules(2);
    auto vert_feature_str = std::string(has_texcoord_0 ? "_TEX" : "");
    vert_feature_str = has_material ? vert_feature_str : "_NOMTL";
    auto frag_feature_str = vert_feature_str;
    if (has_skin_set_0) {
        vert_feature_str += "_SKIN";
    }

    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "base_depthonly_vert" + vert_feature_str + ".spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());

    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "base_depthonly_frag" + frag_feature_str + ".spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

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
        primitive.tag_.double_sided);

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

static std::shared_ptr<renderer::Pipeline> createDrawableShadowPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const ego::PrimitiveInfo& primitive) {
    auto shader_modules = getDrawableDepthonlyShaderModules(
        device,
        primitive.tag_.has_texcoord_0,
        primitive.tag_.has_skin_set_0,
        primitive.material_idx_ >= 0);

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
    rasterization_state_override.override_depth_clamp_enable = true;
    rasterization_state_override.depth_clamp_enable = true;
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

/*    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        CAMERA_OBJECT_BUFFER_INDEX,
        game_camera_buffer.buffer,
        game_camera_buffer.buffer->getSize());*/

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

std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::material_desc_set_layout_;
std::shared_ptr<renderer::DescriptorSetLayout> DrawableObject::skin_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> DrawableObject::drawable_pipeline_layout_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> DrawableObject::drawable_pipeline_list_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> DrawableObject::drawable_shadow_pipeline_list_;
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
        texture.destroy(device);
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
    bool use_local_matrix_only) {
    // update all animations
    if (animations_.size() > 0) {
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

        // update joints
    if (animations_.size() > 0) {
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
                    auto hash_value = primitive.getDepthonlyHash();
                    auto result = drawable_shadow_pipeline_list_.find(hash_value);
                    if (result == drawable_shadow_pipeline_list_.end()) {
                        drawable_shadow_pipeline_list_[hash_value] =
                            createDrawableShadowPipeline(
                                device,
                                renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                                drawable_pipeline_layout_,
                                graphic_pipeline_info,
                                primitive);
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

        drawable_object_list_[file_name] = object_;
    }
    else {
        object_ = result->second;
    }
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

    for (auto& object : drawable_object_list_) {
        for (int i_mesh = 0; i_mesh < object.second->meshes_.size(); i_mesh++) {
            for (int i_prim = 0; i_prim < object.second->meshes_[i_mesh].primitives_.size(); i_prim++) {
                const auto& primitive = object.second->meshes_[i_mesh].primitives_[i_prim];
                auto hash_value = primitive.getDepthonlyHash();
                auto result = drawable_shadow_pipeline_list_.find(hash_value);
                if (result == drawable_shadow_pipeline_list_.end()) {
                    drawable_shadow_pipeline_list_[hash_value] =
                        createDrawableShadowPipeline(
                            device,
                            renderbuffer_formats[int(renderer::RenderPasses::kShadow)],
                            drawable_pipeline_layout_,
                            graphic_pipeline_info,
                            primitive);
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

    cmd_buf->addBufferBarrier(
        object_->instance_buffer_.buffer,
        { SET_FLAG_BIT(Access, VERTEX_ATTRIBUTE_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_INPUT_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        object_->instance_buffer_.buffer->getSize());

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        update_instance_buffer_pipeline_);

    glsl::InstanceBufferUpdateParams params;
    params.num_instances = kNumDrawableInstance;
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

    cmd_buf->addBufferBarrier(
        object_->instance_buffer_.buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, VERTEX_ATTRIBUTE_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_INPUT_BIT) },
        object_->instance_buffer_.buffer->getSize());
}

void DrawableObject::updateIndirectDrawBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

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
    bool depth_only/* = false */ ) {

    auto& pipeline_list =
        depth_only ? drawable_shadow_pipeline_list_ : drawable_pipeline_list_;

    std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
    std::vector<uint64_t> offsets(1);
    buffers[0] = object_->instance_buffer_.buffer;
    offsets[0] = 0;
    cmd_buf->bindVertexBuffers(VINPUT_INSTANCE_BINDING_POINT, buffers, offsets);

    num_draw_meshes = 0;
    size_t last_hash = 0;

    int32_t root_node =
        object_->default_scene_ >= 0 ? object_->default_scene_ : 0;
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
            last_hash);
    }
}

void DrawableObject::update(
    const std::shared_ptr<renderer::Device>& device,
    const float& time) {
    if (object_) {
        object_->update(
            device,
            0,
            time,
            object_->m_use_local_matrix_only_);
    }
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

    setupMeshState(device, model, drawable_object);
    setupMeshes(model, drawable_object);
    setupAnimations(model, drawable_object);
    setupSkins(device, model, drawable_object);
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

    setupMeshState(device, fbx_scene, drawable_object);
    setupMeshes(device, fbx_scene, drawable_object);
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
            uint32_t cur_lod = std::min(0u, uint32_t(prim.index_desc_.size() - 1));
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

} // game_object
} // engine