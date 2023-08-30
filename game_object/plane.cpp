#include "plane.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {

static void createUnifiedQuad(
    std::vector<glm::vec3>& new_vertices,
    std::vector<glm::vec3>& new_normals,
    std::vector<glm::vec2>& new_uvs,
    std::vector<glm::uvec3>& new_faces) {

    new_vertices.resize(4);
    new_normals.resize(4);
    new_uvs.resize(4);
    new_faces.resize(2);

    new_vertices[0] = glm::vec3(-1.0f, 0.0f, -1.0f);
    new_vertices[1] = glm::vec3(1.0f, 0.0f, -1.0f);
    new_vertices[2] = glm::vec3(1.0f, 0.0f, 1.0f);
    new_vertices[3] = glm::vec3(-1.0f, 0.0f, 1.0f);

    new_normals[0] = glm::vec3(0.0f, 1.0f, 0.0f);
    new_normals[1] = glm::vec3(0.0f, 1.0f, 0.0f);
    new_normals[2] = glm::vec3(0.0f, 1.0f, 0.0f);
    new_normals[3] = glm::vec3(0.0f, 1.0f, 0.0f);

    new_uvs[0] = glm::vec2(0.0f, 0.0f);
    new_uvs[1] = glm::vec2(1.0f, 0.0f);
    new_uvs[2] = glm::vec2(1.0f, 1.0f);
    new_uvs[3] = glm::vec2(0.0f, 1.0f);

    new_faces[0] = glm::uvec3(0, 2, 1);
    new_faces[1] = glm::uvec3(0, 3, 2);
}

static std::shared_ptr<renderer::BufferInfo> createUnifiedMeshBuffer(
    const renderer::DeviceInfo& device_info,
    const renderer::BufferUsageFlags& usage,
    const uint64_t& size,
    const void* data) {
    auto v_buffer = std::make_shared<renderer::BufferInfo>();
    renderer::Helper::createBuffer(
        device_info,
        usage,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        v_buffer->buffer,
        v_buffer->memory,
        size,
        data);

    return v_buffer;
}

} // namespace

namespace game_object {

Plane::Plane(const renderer::DeviceInfo& device_info)
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> faces;

    createUnifiedQuad(vertices, normals, uvs, faces);

    renderer::VertexInputBindingDescription binding_desc;
    renderer::VertexInputAttributeDescription attribute_desc;
    std::vector<renderer::VertexInputBindingDescription> binding_descs;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs;

    setPositionBuffer(createUnifiedMeshBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        vertices.size() * sizeof(vertices[0]),
        vertices.data()));
    uint32_t binding_idx = 0;
    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(vertices[0]);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    binding_descs.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_POSITION;
    attribute_desc.format = renderer::Format::R32G32B32_SFLOAT;
    attribute_desc.offset = 0;
    attribute_descs.push_back(attribute_desc);
    binding_idx++;

    setNormalBuffer(createUnifiedMeshBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        normals.size() * sizeof(normals[0]),
        normals.data()));
    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(normals[0]);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    binding_descs.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_NORMAL;
    attribute_desc.format = renderer::Format::R32G32B32_SFLOAT;
    attribute_desc.offset = 0;
    attribute_descs.push_back(attribute_desc);
    binding_idx++;

    setUvBuffer(createUnifiedMeshBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        uvs.size() * sizeof(uvs[0]),
        uvs.data()));
    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(uvs[0]);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    binding_descs.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_TEXCOORD0;
    attribute_desc.format = renderer::Format::R32G32_SFLOAT;
    attribute_desc.offset = 0;
    attribute_descs.push_back(attribute_desc);
    binding_idx++;

    setBindingDescs(binding_descs);
    setAttribDescs(attribute_descs);

    setIndexBuffer(createUnifiedMeshBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
        faces.size() * sizeof(faces[0]),
        faces.data()));
}

void Plane::draw(std::shared_ptr<renderer::CommandBuffer> cmd_buf) {
    std::vector<std::shared_ptr<renderer::Buffer>> buffers;
    std::vector<uint64_t> offsets;
    if (getPositionBuffer()) {
        buffers.push_back(getPositionBuffer()->buffer);
        offsets.push_back(0);
    }

    if (getNormalBuffer()) {
        buffers.push_back(getNormalBuffer()->buffer);
        offsets.push_back(0);
    }

    if (getUvBuffer()) {
        buffers.push_back(getUvBuffer()->buffer);
        offsets.push_back(0);
    }

    cmd_buf->bindVertexBuffers(0, buffers, offsets);

    cmd_buf->bindIndexBuffer(
        getIndexBuffer()->buffer,
        0,
        renderer::IndexType::UINT32);

    cmd_buf->drawIndexed(static_cast<uint32_t>(getIndexBuffer()->buffer->getSize() / sizeof(uint32_t)));
}

} // game_object
} // engine
