#include <string>
#include <sstream>
#include <unordered_map>
#include "object_file.h"
#include "engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {
static uint32_t getNewIndex(
    std::unordered_map<glm::uvec3, uint32_t>& index_map,
    std::vector<glm::vec3>& new_vertices,
    std::vector<glm::vec3>& new_normals,
    std::vector<glm::vec2>& new_uvs,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::vec2>& uvs,
    const glm::uvec3& idx) {
    uint32_t result = 0;
    if (index_map.find(idx) != index_map.end()) {
        result = index_map[idx];
    }
    else {
        auto i = static_cast<uint32_t>(new_vertices.size());
        index_map[idx] = i;
        result = i;
        new_vertices.push_back(vertices[idx.x-1]);
        if (idx.y <= normals.size()) {
            new_normals.push_back(normals[idx.y-1]);
        }
        if (idx.z <= uvs.size()) {
            new_uvs.push_back(uvs[idx.z-1]);
        }
    }

    return result;
}

static void createUnifiedMesh(
    std::vector<glm::vec3>& new_vertices,
    std::vector<glm::vec3>& new_normals,
    std::vector<glm::vec2>& new_uvs,
    std::vector<glm::uvec3>& new_faces,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::vec2>& uvs,
    const std::vector<glm::uvec4>& v_faces,
    const std::vector<glm::uvec4>& vn_faces,
    const std::vector<glm::uvec4>& vt_faces) {

    std::unordered_map<glm::uvec3, uint32_t> index_map;

    for (auto face_idx = 0; face_idx < v_faces.size(); face_idx++) {
        glm::uvec3 new_face;
        const auto& v_face = v_faces[face_idx];
        const auto& vn_face = vn_faces[face_idx];
        const auto& vt_face = vt_faces[face_idx];

        auto i0 = glm::uvec3(v_face.x, vn_face.x, vt_face.x);
        auto i1 = glm::uvec3(v_face.y, vn_face.y, vt_face.y);
        auto i2 = glm::uvec3(v_face.z, vn_face.z, vt_face.z);
        auto i3 = glm::uvec3(v_face.w, vn_face.w, vt_face.w);
        auto n0 = getNewIndex(index_map, new_vertices, new_normals, new_uvs, vertices, normals, uvs, i0);
        auto n1 = getNewIndex(index_map, new_vertices, new_normals, new_uvs, vertices, normals, uvs, i1);
        auto n2 = getNewIndex(index_map, new_vertices, new_normals, new_uvs, vertices, normals, uvs, i2);
        new_face.x = n0;
        new_face.y = n1;
        new_face.z = n2;
        new_faces.push_back(new_face);

        if (v_face.w != (uint32_t)-1) {
            new_face.x = n0;
            new_face.y = n2;
            new_face.z = getNewIndex(index_map, new_vertices, new_normals, new_uvs, vertices, normals, uvs, i3);
            new_faces.push_back(new_face);
        }
    }
}

static std::shared_ptr<game_object::Patch> createPatch(
    const std::shared_ptr<renderer::Device>& device,
    std::vector<glm::vec3> new_vertices,
    std::vector<glm::vec3> new_normals,
    std::vector<glm::vec2> new_uvs,
    std::vector<glm::uvec3> new_faces) {
    auto patch = std::make_shared<game_object::Patch>();

    renderer::VertexInputBindingDescription binding_desc;
    renderer::VertexInputAttributeDescription attribute_desc;
    std::vector<renderer::VertexInputBindingDescription> binding_descs;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs;
    assert(new_vertices.size() > 0);
    patch->setPositionBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            new_vertices.size() * sizeof(new_vertices[0]),
            new_vertices.data()));

    uint32_t binding_idx = 0;
    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(new_vertices[0]);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    binding_descs.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_POSITION;
    attribute_desc.format = renderer::Format::R32G32B32_SFLOAT;
    attribute_desc.offset = 0;
    attribute_descs.push_back(attribute_desc);
    binding_idx++;

    if (new_normals.size() > 0) {
        patch->setNormalBuffer(
            helper::createUnifiedMeshBuffer(
                device,
                SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                new_normals.size() * sizeof(new_normals[0]),
                new_normals.data()));

        binding_desc.binding = binding_idx;
        binding_desc.stride = sizeof(new_normals[0]);
        binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
        binding_descs.push_back(binding_desc);

        attribute_desc.binding = binding_idx;
        attribute_desc.location = VINPUT_NORMAL;
        attribute_desc.format = renderer::Format::R32G32B32_SFLOAT;
        attribute_desc.offset = 0;
        attribute_descs.push_back(attribute_desc);
        binding_idx++;
    }

    if (new_uvs.size() > 0) {
        patch->setUvBuffer(
            helper::createUnifiedMeshBuffer(
                device,
                SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                new_uvs.size() * sizeof(new_uvs[0]),
                new_uvs.data()));

        binding_desc.binding = binding_idx;
        binding_desc.stride = sizeof(new_uvs[0]);
        binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
        binding_descs.push_back(binding_desc);

        attribute_desc.binding = binding_idx;
        attribute_desc.location = VINPUT_TEXCOORD0;
        attribute_desc.format = renderer::Format::R32G32_SFLOAT;
        attribute_desc.offset = 0;
        attribute_descs.push_back(attribute_desc);
        binding_idx++;
    }

    patch->setBindingDescs(binding_descs);
    patch->setAttribDescs(attribute_descs);

    assert(new_faces.size() > 0);
    patch->setIndexBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
            new_faces.size() * sizeof(new_faces[0]),
            new_faces.data()));

    return patch;
}

static std::shared_ptr<game_object::Patch> createPatch(
    const std::shared_ptr<renderer::Device>& device,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::vec2>& uvs,
    std::vector<glm::uvec4>& v_faces,
    std::vector<glm::uvec4>& vn_faces,
    std::vector<glm::uvec4>& vt_faces) {
    std::vector<glm::vec3> new_vertices;
    std::vector<glm::vec3> new_normals;
    std::vector<glm::vec2> new_uvs;
    std::vector<glm::uvec3> new_faces;

    createUnifiedMesh(
        new_vertices,
        new_normals,
        new_uvs,
        new_faces,
        vertices,
        normals,
        uvs,
        v_faces,
        vn_faces,
        vt_faces);

    v_faces.clear();
    vn_faces.clear();
    vt_faces.clear();

    return createPatch(
        device,
        new_vertices,
        new_normals,
        new_uvs,
        new_faces);
}

static std::shared_ptr<renderer::PipelineLayout> createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& object_desc_set_layout) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::ModelParams);

    renderer::DescriptorSetLayoutList desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(object_desc_set_layout);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::DescriptorSetLayout> createObjectDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(4);

    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(DIFFUSE_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(NORMAL_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(GLOSSNESS_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(SPECULAR_TEX_INDEX));

    return device->createDescriptorSetLayout(bindings);
}

static renderer::WriteDescriptorList addObjectTextures(
    const std::shared_ptr<renderer::DescriptorSet>& desc_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& normal_tex,
    const renderer::TextureInfo& glossness_tex,
    const renderer::TextureInfo& diffuse_tex,
    const renderer::TextureInfo& specular_tex) {

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(4);

    // diffuse.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DIFFUSE_TEX_INDEX,
        texture_sampler,
        diffuse_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // normal.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        NORMAL_TEX_INDEX,
        texture_sampler,
        normal_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // glossness.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        GLOSSNESS_TEX_INDEX,
        texture_sampler,
        glossness_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // specular.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SPECULAR_TEX_INDEX,
        texture_sampler,
        specular_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

}

namespace game_object {

void ObjectMesh::loadObjectFile(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::string& object_name,
    const std::string& shader_name,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {
    uint64_t buffer_size = 0;
    auto buffer_data = engine::helper::readFile(object_name, buffer_size);
    std::string obj_string((char*)buffer_data.data(), buffer_size);
    std::stringstream obj_stream(std::move(obj_string));
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec4> v_faces;
    std::vector<glm::uvec4> vn_faces;
    std::vector<glm::uvec4> vt_faces;

    diffuse_tex_ = std::make_shared<renderer::TextureInfo>();
    normal_tex_ = std::make_shared<renderer::TextureInfo>();
    glossiness_tex_ = std::make_shared<renderer::TextureInfo>();
    specular_tex_ = std::make_shared<renderer::TextureInfo>();

    auto format = renderer::Format::R8G8B8A8_UNORM;
    engine::helper::createTextureImage(
        device,
        "assets/lungs/lungs_default_Diffuse.png",
        format,
        *diffuse_tex_);
    engine::helper::createTextureImage(
        device,
        "assets/lungs/lungs_default_Normal.png",
        format,
        *normal_tex_);
    engine::helper::createTextureImage(
        device,
        "assets/lungs/lungs_default_Glossiness.png",
        format,
        *glossiness_tex_);
    engine::helper::createTextureImage(
        device,
        "assets/lungs/lungs_default_Specular.png",
        format,
        *specular_tex_);

    bool start_process_faces = false;
    bool has_normal = false;
    bool has_texture = false;
    while (!obj_stream.eof()) {
        std::string line;
        getline(obj_stream, line);

        std::stringstream line_stream(line);

        std::string tag;
        line_stream >> tag;

        // end of process one object.
        if (start_process_faces && tag != "f") {
            patches_.push_back(
                createPatch(
                    device,
                    vertices,
                    normals,
                    uvs,
                    v_faces,
                    vn_faces,
                    vt_faces));

            start_process_faces = false;
        }

        if (tag == "#" || tag == "") {
            continue;
        }

        if (tag == "v") {
            float x, y, z;
            line_stream >> x >> y >> z;
            vertices.push_back(glm::vec3(x, y, z));
        }

        if (tag == "vn") {
            float x, y, z;
            line_stream >> x >> y >> z;
            normals.push_back(glm::vec3(x, y, z));
            has_normal = true;
        }

        if (tag == "vt") {
            float x, y;
            line_stream >> x >> y;
            uvs.push_back(glm::vec2(x, y));
            has_texture = true;
        }

        if (tag == "f") {
            start_process_faces = true;
            std::string is0, is1, is2, is3;
            line_stream >> is0 >> is1 >> is2 >> is3;
            glm::uvec4 i0, i1, i2;
            sscanf_s(is0.data(), "%u/%u/%u", &i0.x, &i1.x, &i2.x);
            sscanf_s(is1.data(), "%u/%u/%u", &i0.y, &i1.y, &i2.y);
            sscanf_s(is2.data(), "%u/%u/%u", &i0.z, &i1.z, &i2.z);
            if (is3 != "") {
                sscanf_s(is3.data(), "%u/%u/%u", &i0.w, &i1.w, &i2.w);
            }
            else {
                i0.w = -1;
                i1.w = -1;
                i2.w = -1;
            }

            v_faces.push_back(i0);
            vt_faces.push_back(i1);
            vn_faces.push_back(i2);
        }
    }

    if (v_faces.size() > 0) {
        patches_.push_back(
            createPatch(
                device,
                vertices,
                normals,
                uvs,
                v_faces,
                vn_faces,
                vt_faces));
    }

    object_desc_set_layout_ = createObjectDescriptorSetLayout(
        device);

    object_desc_set_ = device->createDescriptorSets(
        descriptor_pool, object_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto material_descs = addObjectTextures(
        object_desc_set_,
        texture_sampler,
        *normal_tex_,
        *glossiness_tex_,
        *diffuse_tex_,
        *specular_tex_);

    device->updateDescriptorSets(material_descs);

    object_pipeline_layout_ = createPipelineLayout(
        device,
        global_desc_set_layouts,
        object_desc_set_layout_);

    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            shader_name + "_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT);
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            shader_name + "_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT);

    object_pipeline_ = device->createPipeline(
        render_pass,
        object_pipeline_layout_,
        patches_[0]->getBindingDescs(),
        patches_[0]->getAttribDescs(),
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size);
}

void ObjectMesh::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    uint32_t draw_idx/* = -1 */ ) {

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, object_pipeline_);

    renderer::DescriptorSetList desc_sets = desc_set_list;
    desc_sets.push_back(object_desc_set_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        object_pipeline_layout_,
        desc_sets);

    glsl::ModelParams model_params{};
    model_params.model_mat =
        glm::mat4(
            glm::vec4(1, 0, 0, 0),
            glm::vec4(0, 0, 1, 0),
            glm::vec4(0, -1, 0, 0),
            glm::vec4(0, 0, 0, 1));

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        object_pipeline_layout_,
        &model_params,
        sizeof(model_params));

    for (auto i_patch = 0; i_patch < patches_.size(); i_patch++) {
        auto& patch = patches_[i_patch];
        if (draw_idx != (uint32_t)-1 && draw_idx != i_patch) {
            continue;
        }

        std::vector<std::shared_ptr<renderer::Buffer>> buffers;
        std::vector<uint64_t> offsets;
        if (patch->getPositionBuffer()) {
            buffers.push_back(patch->getPositionBuffer()->buffer);
            offsets.push_back(0);
        }

        if (patch->getNormalBuffer()) {
            buffers.push_back(patch->getNormalBuffer()->buffer);
            offsets.push_back(0);
        }

        if (patch->getUvBuffer()) {
            buffers.push_back(patch->getUvBuffer()->buffer);
            offsets.push_back(0);
        }

        cmd_buf->bindVertexBuffers(0, buffers, offsets);

        cmd_buf->bindIndexBuffer(
            patch->getIndexBuffer()->buffer,
            0,
            renderer::IndexType::UINT32);

        cmd_buf->drawIndexed(static_cast<uint32_t>(patch->getIndexBuffer()->buffer->getSize() / sizeof(uint32_t)));
    }
}

void ObjectMesh::destroy(const std::shared_ptr<renderer::Device>& device) {
    for (auto& patch : patches_) {
        patch->destroy(device);
    }

    if (normal_tex_) {
        normal_tex_->destroy(device);
    }
    if (glossiness_tex_) {
        glossiness_tex_->destroy(device);
    }
    if (diffuse_tex_) {
        diffuse_tex_->destroy(device);
    }
    if (specular_tex_) {
        specular_tex_->destroy(device);
    }

    device->destroyDescriptorSetLayout(object_desc_set_layout_);
    device->destroyPipelineLayout(object_pipeline_layout_);
    device->destroyPipeline(object_pipeline_);
}

} // game_object
} // engine