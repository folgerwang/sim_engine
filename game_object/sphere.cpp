#include <map>
#include "sphere.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {

// Global map to cache midpoints during subdivision to avoid duplicate vertices
// The key is a pair of vertex indices (sorted to ensure uniqueness for an edge)
using EdgeKey = std::pair<uint32_t, uint32_t>;
std::map<EdgeKey, uint32_t> g_midpoint_cache;

// Function to get or create a midpoint vertex for an edge
// Normalizes the midpoint to project it onto the unit sphere surface
static uint32_t getMidpointVertex(
    uint32_t p1_idx,
    uint32_t p2_idx,
    std::vector<glm::vec3>& vertices) {

    // Create a unique key for the edge (order doesn't matter)
    EdgeKey edge_key =
        std::make_pair(
            std::min(p1_idx, p2_idx),
            std::max(p1_idx, p2_idx));

    // Check if midpoint is already cached
    auto it = g_midpoint_cache.find(edge_key);
    if (it != g_midpoint_cache.end()) {
        return it->second; // Return existing midpoint index
    }

    // If not cached, calculate the midpoint
    const glm::vec3& v1 = vertices[p1_idx];
    const glm::vec3& v2 = vertices[p2_idx];
    glm::vec3 mid_point = (v1 + v2) / 2.0f;

    // Normalize the midpoint to place it on the surface of a unit sphere
    vertices.push_back(normalize(mid_point));
    uint32_t new_index = static_cast<uint32_t>(vertices.size() - 1);

    // Cache and return the new index
    g_midpoint_cache[edge_key] = new_index;
    return new_index;
}

// Function to generate an icosphere
static void generateIcosphere(
    std::vector<glm::vec3>& ov,
    std::vector<glm::uvec3>& ot,
    int subdivisions) {
    g_midpoint_cache.clear(); // Clear cache for new generation

    // --- 1. Create an Icosahedron (20 triangular faces, 12 vertices) ---
    // Golden ratio
    float t = (1.0f + std::sqrt(5.0f)) / 2.0f;

    std::vector<glm::uvec3> tris[2];
    ov.reserve(int32_t(12 + 20 * pow(3, subdivisions) / 2));
    tris[0].reserve(int32_t(20 * pow(4, subdivisions)));
    tris[1].reserve(int32_t(20 * pow(4, subdivisions)));

    // Add icosahedron vertices (normalized to unit sphere)
    ov.push_back(normalize(glm::vec3(-1, t, 0)));
    ov.push_back(normalize(glm::vec3(1, t, 0)));
    ov.push_back(normalize(glm::vec3(-1, -t, 0)));
    ov.push_back(normalize(glm::vec3(1, -t, 0)));

    ov.push_back(normalize(glm::vec3(0, -1, t)));
    ov.push_back(normalize(glm::vec3(0, 1, t)));
    ov.push_back(normalize(glm::vec3(0, -1, -t)));
    ov.push_back(normalize(glm::vec3(0, 1, -t)));

    ov.push_back(normalize(glm::vec3(t, 0, -1)));
    ov.push_back(normalize(glm::vec3(t, 0, 1)));
    ov.push_back(normalize(glm::vec3(-t, 0, -1)));
    ov.push_back(normalize(glm::vec3(-t, 0, 1)));

    // Add icosahedron faces (triangles)
    // 5 faces around point 0
    tris[0].push_back(glm::uvec3(0, 11, 5));
    tris[0].push_back(glm::uvec3(0, 5, 1));
    tris[0].push_back(glm::uvec3(0, 1, 7));
    tris[0].push_back(glm::uvec3(0, 7, 10));
    tris[0].push_back(glm::uvec3(0, 10, 11));

    // 5 adjacent faces
    tris[0].push_back(glm::vec3(1, 5, 9));
    tris[0].push_back(glm::vec3(5, 11, 4));
    tris[0].push_back(glm::vec3(11, 10, 2));
    tris[0].push_back(glm::vec3(10, 7, 6));
    tris[0].push_back(glm::vec3(7, 1, 8));

    // 5 faces around point 3
    tris[0].push_back(glm::vec3(3, 9, 4));
    tris[0].push_back(glm::vec3(3, 4, 2));
    tris[0].push_back(glm::vec3(3, 2, 6));
    tris[0].push_back(glm::vec3(3, 6, 8));
    tris[0].push_back(glm::vec3(3, 8, 9));

    // 5 adjacent faces
    tris[0].push_back(glm::vec3(4, 9, 5));
    tris[0].push_back(glm::vec3(2, 4, 11));
    tris[0].push_back(glm::vec3(6, 2, 10));
    tris[0].push_back(glm::vec3(8, 6, 7));
    tris[0].push_back(glm::vec3(9, 8, 1));

    // --- 2. Subdivide Triangles ---
    int src_index = 0;
    for (int i = 0; i < subdivisions; ++i) {
        auto& src_tris = tris[src_index];
        auto& dst_tris = tris[1 - src_index];
        dst_tris.clear();
        for (const auto& tri : src_tris) {
            // Get indices of original triangle's vertices
            uint32_t v1_idx = tri.x;
            uint32_t v2_idx = tri.y;
            uint32_t v3_idx = tri.z;

            // Calculate midpoints of each edge and add new vertices
            uint32_t m12_idx = getMidpointVertex(v1_idx, v2_idx, ov);
            uint32_t m23_idx = getMidpointVertex(v2_idx, v3_idx, ov);
            uint32_t m31_idx = getMidpointVertex(v3_idx, v1_idx, ov);

            // Create 4 new triangles from the original one
            dst_tris.push_back(glm::uvec3(v1_idx, m12_idx, m31_idx));
            dst_tris.push_back(glm::uvec3(v2_idx, m23_idx, m12_idx));
            dst_tris.push_back(glm::uvec3(v3_idx, m31_idx, m23_idx));
            dst_tris.push_back(glm::uvec3(m12_idx, m23_idx, m31_idx));
        }
        src_index = 1 - src_index;
    }

    auto& src_tris = tris[src_index];
    ot.reserve(src_tris.size());
    for (int i = 0; i < src_tris.size(); i++) {
        ot[i] = src_tris[i];
    }
}

void calculateNormalAndTangent(
    std::vector<glm::vec3>& tangents,
    std::vector<glm::vec2>& uvs,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::uvec3>& triangles) {

    uint32_t num_vertex =
        static_cast<uint32_t>(vertices.size());
    uint32_t num_faces =
        static_cast<uint32_t>(triangles.size());

    tangents.resize(num_vertex);

    for (uint32_t n = 0; n < num_vertex; n++) {
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

        tangents[triangle[0]] += t0 * angle0;
        tangents[triangle[1]] += t1 * angle1;
        tangents[triangle[2]] += t2 * angle2;
    }

    for (uint32_t n = 0; n < num_vertex; n++) {
        tangents[n] = normalize(tangents[n]);
    }
}

} // namespace

namespace game_object {

Sphere::Sphere(
    const std::shared_ptr<renderer::Device>& device,
    float radius,
    uint32_t subdivisions,
    const std::source_location& src_location) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> faces;

    generateIcosphere(
        vertices,
        faces,
        subdivisions);

    calculateNormalAndTangent(
        tangents,
        uvs,
        vertices,
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

void Sphere::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf) {
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

} //game_object
} //engine