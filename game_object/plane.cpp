#include "plane.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {

// generate a plane with width segments x and height segments y
static void generatePlaneMesh(
    std::vector<glm::vec3>& vertices,
    std::vector<glm::vec2>& uvs,
    std::vector<glm::uvec3>& triangles,
    const std::array<glm::vec3, 4>& v,
    uint32_t segment_x,
    uint32_t segment_y) {

    uint32_t num_vertex =
        (segment_x + 1) * (segment_y + 1);

    uint32_t num_faces =
        segment_x * segment_y * 2;

    vertices.resize(num_vertex);
    uvs.resize(num_vertex);
    triangles.resize(num_faces);

    for (uint32_t y = 0; y < segment_y + 1; y++) {
        float factor_y = (float)y / (float)segment_y;
        auto x_start = mix(v[0], v[3], factor_y);
        auto x_end = mix(v[1], v[2], factor_y);
        for (uint32_t x = 0; x < segment_x + 1; x++) {
            uint32_t idx = y * (segment_x + 1) + x;
            float factor_x = (float)x / (float)segment_x;

            vertices[idx] = mix(x_start, x_end, factor_x);
            uvs[idx] = glm::vec2(factor_x, factor_y);
        }
    }

    for (uint32_t y = 0; y < segment_y; y++) {
        for (uint32_t x = 0; x < segment_x; x++) {
            uint32_t idx = y * segment_x + x;
            uint32_t quad_idx_00 = y * (segment_x + 1) + x;
            uint32_t quad_idx_01 = quad_idx_00 + 1;
            uint32_t quad_idx_10 = quad_idx_01 + segment_x;
            uint32_t quad_idx_11 = quad_idx_10 + 1;

            triangles[idx * 2] = glm::uvec3(
                quad_idx_00,
                quad_idx_01,
                quad_idx_10);
            triangles[idx * 2 + 1] = glm::uvec3(
                quad_idx_01,
                quad_idx_11,
                quad_idx_10);
        }
    }
}

void calculateNormalAndTangent(
    std::vector<glm::vec3>& normals,
    std::vector<glm::vec3>& tangents,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::vec2>& uvs,
    const std::vector<glm::uvec3>& triangles) {

    uint32_t num_vertex =
        static_cast<uint32_t>(vertices.size());
    uint32_t num_faces =
        static_cast<uint32_t>(triangles.size());

    normals.resize(num_vertex);
    tangents.resize(num_vertex);

    for (uint32_t n = 0; n < num_vertex; n++) {
        normals[n] = glm::vec3(0, 0, 0);
        tangents[n] = glm::vec3(0, 0, 0);
    }

    for (uint32_t f = 0; f < num_faces; f++) {
        auto& triangle = triangles[f];
        auto& v0 = vertices[triangle[0]];
        auto& v1 = vertices[triangle[1]];
        auto& v2 = vertices[triangle[2]];

        auto& uv0 = uvs[triangle[0]];
        auto& uv1 = uvs[triangle[1]];
        auto& uv2 = uvs[triangle[2]];

        auto edge10 = normalize(v1 - v0);
        auto edge20 = normalize(v2 - v0);
        auto edge21 = normalize(v2 - v1);

        float angle0 = acos(dot(edge10, edge20));
        float angle1 = acos(-dot(edge10, edge21));
        float angle2 = acos(dot(edge20, edge21));

        auto edge_uv10 = uv1 - uv0;
        auto edge_uv20 = uv2 - uv0;
        auto edge_uv21 = uv2 - uv1;

        glm::vec3 normal = cross(edge10, edge20);

        float r0 = 1.0f / (edge_uv10.x * edge_uv20.y - edge_uv10.y * edge_uv20.x);
        float r1 = 1.0f / (edge_uv10.x * edge_uv21.y - edge_uv10.y * edge_uv21.x);
        float r2 = 1.0f / (edge_uv20.x * edge_uv21.y - edge_uv20.y * edge_uv21.x);
        auto t0 = (edge10 * edge_uv20.y - edge20 * edge_uv10.y) * r0;
        auto t1 = (edge10 * edge_uv21.y - edge21 * edge_uv10.y) * r1;
        auto t2 = (edge20 * edge_uv21.y - edge21 * edge_uv20.y) * r2;

        auto b0 = (edge20 * edge_uv10.x - edge10 * edge_uv20.x) * r0;
        auto b1 = (edge21 * edge_uv10.x - edge10 * edge_uv21.x) * r1;
        auto b2 = (edge21 * edge_uv20.x - edge20 * edge_uv21.x) * r2;

        normals[triangle[0]] += normal * angle0;
        normals[triangle[1]] += normal * angle1;
        normals[triangle[2]] += normal * angle2;

        tangents[triangle[0]] += t0 * angle0;
        tangents[triangle[1]] += t1 * angle1;
        tangents[triangle[2]] += t2 * angle2;
    }

    for (uint32_t n = 0; n < num_vertex; n++) {
        normals[n] = normalize(normals[n]);
        tangents[n] = normalize(tangents[n]);
    }
}

} // namespace

namespace game_object {

Plane::Plane(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<std::array<glm::vec3, 4>> v,
    uint32_t split_num_x,
    uint32_t split_num_y,
    const std::source_location& src_location) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> faces;

    // 4 vertices
    if (v == nullptr) {
        v = std::make_shared<std::array<glm::vec3, 4>>();
        (*v)[0] = glm::vec3(-1.0f, 0.0f, -1.0f);
        (*v)[1] = glm::vec3(-1.0f, 0.0f, 1.0f);
        (*v)[2] = glm::vec3(1.0f, 0.0f, 1.0f);
        (*v)[3] = glm::vec3(1.0f, 0.0f, -1.0f);
    }

    generatePlaneMesh(
        vertices,
        uvs,
        faces,
        *v,
        split_num_x,
        split_num_y);

    calculateNormalAndTangent(
        normals,
        tangents,
        vertices,
        uvs,
        faces);

    renderer::VertexInputBindingDescription binding_desc;
    renderer::VertexInputAttributeDescription attribute_desc;
    std::vector<renderer::VertexInputBindingDescription> binding_descs;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs;

    setPositionBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            vertices.size() * sizeof(vertices[0]),
            vertices.data(),
            src_location));
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

    setNormalBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            normals.size() * sizeof(normals[0]),
            normals.data(),
            src_location));
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

    setTangentBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            tangents.size() * sizeof(tangents[0]),
            tangents.data(),
            src_location));
    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(tangents[0]);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    binding_descs.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_TANGENT;
    attribute_desc.format = renderer::Format::R32G32B32_SFLOAT;
    attribute_desc.offset = 0;
    attribute_descs.push_back(attribute_desc);
    binding_idx++;

    setUvBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            uvs.size() * sizeof(uvs[0]),
            uvs.data(),
            src_location));
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

    setIndexBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
            faces.size() * sizeof(faces[0]),
            faces.data(),
            src_location));
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

    if (getTangentBuffer()) {
        buffers.push_back(getTangentBuffer()->buffer);
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

    uint32_t index_count =
        static_cast<uint32_t>(getIndexBuffer()->buffer->getSize() / sizeof(uint32_t));
    cmd_buf->drawIndexed(index_count);
}

} // game_object
} // engine
