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
    auto reserve_vertices_size = int32_t(12 + 20 * pow(3, subdivisions));
    auto reserve_triangle_size = int32_t(20 * pow(4, subdivisions));

    std::vector<glm::uvec3> tris[2];
    ov.reserve(reserve_vertices_size);
    tris[0].reserve(reserve_triangle_size);
    tris[1].reserve(reserve_triangle_size);

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
    ot.resize(src_tris.size());
    for (int i = 0; i < src_tris.size(); i++) {
        ot[i] = src_tris[i];
    }
}

void calculateNormalAndTangent(
    std::vector<glm::vec4>& tangents,
    std::vector<glm::vec2>& uvs,
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::uvec3>& triangles) {

    uint32_t num_vertex =
        static_cast<uint32_t>(vertices.size());
    uint32_t num_faces =
        static_cast<uint32_t>(triangles.size());

    // --- Calculate UVs and Normals (on unit sphere) ---
    uvs.resize(num_vertex);
    for (size_t i = 0; i < num_vertex; ++i) {
        const glm::vec3& unit_v = vertices[i]; // Vertices are already normalized (on unit sphere)

        // Spherical UV mapping
        // atan2 returns angle in radians from -PI to PI
        // acos returns angle in radians from 0 to PI
        float phi = std::atan2(unit_v.z, unit_v.x); // Azimuthal angle (longitude)
        float theta = std::acos(unit_v.y);          // Polar angle (colatitude)

        uvs[i].x = (phi + static_cast<float>(glm::pi<float>())) / (2.0f * static_cast<float>(glm::pi<float>())); // Normalize phi to [0, 1]
        uvs[i].y = theta / static_cast<float>(glm::pi<float>());                                 // Normalize theta to [0, 1]
    }

    // --- Calculate Tangents ---
    tangents.resize(num_vertex, glm::vec4(0, 0, 0, 0));
    std::vector<glm::vec3> tan1Accumulator(num_vertex, glm::vec3(0, 0, 0));
    std::vector<glm::vec3> tan2Accumulator(num_vertex, glm::vec3(0, 0, 0)); // For bitangents

    for (const auto& tri : triangles) {
        const glm::vec3& v0_pos = vertices[tri.x];
        const glm::vec3& v1_pos = vertices[tri.y];
        const glm::vec3& v2_pos = vertices[tri.z];

        const glm::vec2& uv0 = uvs[tri.x];
        const glm::vec2& uv1 = uvs[tri.y];
        const glm::vec2& uv2 = uvs[tri.z];

        glm::vec3 delta_pos1 = v1_pos - v0_pos;
        glm::vec3 delta_pos2 = v2_pos - v0_pos;

        glm::vec2 delta_uv1 = uv1 - uv0;
        glm::vec2 delta_uv2 = uv2 - uv0;

        float r = 1.0f / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
        if (std::isinf(r) || std::isnan(r)) { // Check for degenerate UVs
            r = 0.0f; // Avoid issues, tangent will be zero
        }


        glm::vec3 tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
        glm::vec3 bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

        // Accumulate for each vertex of the triangle
        tan1Accumulator[tri.x] = tan1Accumulator[tri.x] + tangent;
        tan1Accumulator[tri.y] = tan1Accumulator[tri.y] + tangent;
        tan1Accumulator[tri.z] = tan1Accumulator[tri.z] + tangent;

        tan2Accumulator[tri.x] = tan2Accumulator[tri.x] + bitangent;
        tan2Accumulator[tri.y] = tan2Accumulator[tri.y] + bitangent;
        tan2Accumulator[tri.z] = tan2Accumulator[tri.z] + bitangent;
    }

    for (size_t i = 0; i < num_vertex; ++i) {
        const glm::vec3& N = vertices[i];
        const glm::vec3& T_acc = tan1Accumulator[i];
        const glm::vec3& B_acc = tan2Accumulator[i];

        // Gram-Schmidt orthogonalize T_acc with respect to N
        glm::vec3 tangent = normalize(T_acc - N * dot(N, T_acc));

        // Calculate handedness (direction of B_acc relative to cross(N, T_acc))
        float handedness = (dot(cross(N, T_acc), B_acc) < 0.0f) ? -1.0f : 1.0f;

        tangents[i] = glm::vec4(tangent, handedness);
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
    std::vector<glm::vec4> tangents;
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

    setPositionBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            vertices.size() * sizeof(vertices[0]),
            vertices.data(),
            src_location));

    setNormalBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            vertices.size() * sizeof(vertices[0]),
            vertices.data(),
            src_location));

    setTangentBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            tangents.size() * sizeof(tangents[0]),
            tangents.data(),
            src_location));

    setUvBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            uvs.size() * sizeof(uvs[0]),
            uvs.data(),
            src_location));

    setIndexBuffer(
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
            faces.size() * sizeof(faces[0]),
            faces.data(),
            src_location));
}

void Sphere::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        s_base_shape_pipeline_);

    cmd_buf->setViewports(viewports, 0, uint32_t(viewports.size()));
    cmd_buf->setScissors(scissors, 0, uint32_t(scissors.size()));

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

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        s_base_shape_pipeline_layout_,
        desc_set_list);

    glsl::BaseShapeDrawParams params;
    params.transform =
        glm::mat4(
            glm::vec4(0.5f, 0, 0, 0),
            glm::vec4(0, 0.5f, 0, 0),
            glm::vec4(0, 0, 0.5f, 0),
            glm::vec4(-1.70988739f, 2.48692441f, -13.6786499f, 1));

    cmd_buf->pushConstants(
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
        s_base_shape_pipeline_layout_,
        &params,
        sizeof(params));

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