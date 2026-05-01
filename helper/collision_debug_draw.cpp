#include "collision_debug_draw.h"

#include "shaders/global_definition.glsl.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"   // createUnifiedMeshBuffer
#include "helper/collision_mesh.h"

// Vertex-input location for the per-vertex triangle id in
// collision_debug.vert. Matches the shader's `layout(location = 15)`
// and lives outside the standard mesh slots (0..9) and the instance
// slots (10..14) defined in global_definition.glsl.h.
static constexpr uint32_t kTriangleIdLocation = 15;

namespace engine {
namespace {

static auto createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    push_const_range.offset = 0;
    // Reuse ClusterDebugParams { mat4 transform; } — same shape as
    // what collision_debug.vert/frag expect.
    push_const_range.size = sizeof(glsl::ClusterDebugParams);

    renderer::DescriptorSetLayoutList desc_set_layouts =
        global_desc_set_layouts;

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static renderer::ShaderModuleList getShaderModules(
    std::shared_ptr<renderer::Device> device) {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "collision_debug_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "collision_debug_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    return shader_modules;
}

static auto createPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::vector<renderer::VertexInputBindingDescription>& binding_descs,
    const std::vector<renderer::VertexInputAttributeDescription>& attribute_descs,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    // Double-sided rasterisation: collision-debug geometry is
    // built from voxel surfaces, AABB faces, and source-mesh
    // triangles whose winding can vary with the upstream FBX
    // export.  Some single-cell-thick voxel slabs (flat ground
    // tiles, decals) end up showing only one face after culling
    // -- we'd rather always see them than lose ground tiles to
    // back-face culling.  Cheap on the GPU side because debug
    // geometry is low-poly, and avoids a class of "where did my
    // mesh go" reports.
    renderer::RasterizationStateOverride rasterization_state_info;
    rasterization_state_info.override_double_sided = true;
    rasterization_state_info.double_sided          = true;

    return device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        input_assembly,
        graphic_pipeline_info,
        getShaderModules(device),
        frame_buffer_format,
        rasterization_state_info,
        std::source_location::current());
}

static renderer::ShaderModuleList getWireShaderModules(
    std::shared_ptr<renderer::Device> device) {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "collision_debug_vert.spv",   // shared with the solid pass
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "collision_debug_wire_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    return shader_modules;
}

static auto createWirePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::vector<renderer::VertexInputBindingDescription>& binding_descs,
    const std::vector<renderer::VertexInputAttributeDescription>& attribute_descs,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::LINE_LIST;
    input_assembly.restart_enable = false;

    // Push wireframe lines toward the camera so they win the LESS
    // depth test against the solid fill drawn at the same Z. The
    // exact constants are chosen empirically: large enough to clear
    // depth quantisation noise on the Bistro's geometry, small
    // enough not to draw lines belonging to occluded back faces.
    // Also double-sided to match the solid-fill pipeline -- LINE_LIST
    // rasterisation isn't directly affected by cull_mode, but keeping
    // both pipelines on identical state simplifies reasoning.
    renderer::RasterizationStateOverride rasterization_state_info;
    rasterization_state_info.override_double_sided = true;
    rasterization_state_info.double_sided          = true;
    rasterization_state_info.override_depth_bias = true;
    rasterization_state_info.depth_bias_enable = true;
    rasterization_state_info.depth_bias_constant_factor = -2.0f;
    rasterization_state_info.depth_bias_slope_factor    = -2.0f;

    return device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        input_assembly,
        graphic_pipeline_info,
        getWireShaderModules(device),
        frame_buffer_format,
        rasterization_state_info,
        std::source_location::current());
}

} // anonymous namespace

namespace helper {

std::vector<renderer::VertexInputBindingDescription>   CollisionDebugDraw::s_binding_descs_;
std::vector<renderer::VertexInputAttributeDescription> CollisionDebugDraw::s_attrib_descs_;
std::shared_ptr<renderer::PipelineLayout>              CollisionDebugDraw::s_pipeline_layout_;
std::shared_ptr<renderer::Pipeline>                    CollisionDebugDraw::s_pipeline_;
std::shared_ptr<renderer::Pipeline>                    CollisionDebugDraw::s_wireframe_pipeline_;

void CollisionDebugMeshBuffers::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    if (position_buffer) {
        position_buffer->destroy(device);
        position_buffer.reset();
    }
    if (triangle_id_buffer) {
        triangle_id_buffer->destroy(device);
        triangle_id_buffer.reset();
    }
    if (line_index_buffer) {
        line_index_buffer->destroy(device);
        line_index_buffer.reset();
    }
    vertex_count = 0;
    line_index_count = 0;
}

void CollisionDebugDraw::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {

    // Binding 0 — tight vec3 position buffer (per-triangle-expanded).
    renderer::VertexInputBindingDescription bind0;
    bind0.binding    = 0;
    bind0.stride     = sizeof(glm::vec3);
    bind0.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(bind0);

    renderer::VertexInputAttributeDescription attr0;
    attr0.binding  = 0;
    attr0.location = VINPUT_POSITION;               // matches collision_debug.vert
    attr0.format   = renderer::Format::R32G32B32_SFLOAT;
    attr0.offset   = 0;
    s_attrib_descs_.push_back(attr0);

    // Binding 1 — tight uint32 triangle_id buffer.
    renderer::VertexInputBindingDescription bind1;
    bind1.binding    = 1;
    bind1.stride     = sizeof(uint32_t);
    bind1.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(bind1);

    renderer::VertexInputAttributeDescription attr1;
    attr1.binding  = 1;
    attr1.location = kTriangleIdLocation;            // == 15, matches shader
    attr1.format   = renderer::Format::R32_UINT;
    attr1.offset   = 0;
    s_attrib_descs_.push_back(attr1);

    s_pipeline_layout_ =
        createPipelineLayout(device, global_desc_set_layouts);
    s_pipeline_ = createPipeline(
        device,
        s_pipeline_layout_,
        graphic_pipeline_info,
        s_binding_descs_,
        s_attrib_descs_,
        frame_buffer_format);
    s_wireframe_pipeline_ = createWirePipeline(
        device,
        s_pipeline_layout_,
        graphic_pipeline_info,
        s_binding_descs_,
        s_attrib_descs_,
        frame_buffer_format);
}

void CollisionDebugDraw::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    if (s_pipeline_layout_) {
        device->destroyPipelineLayout(s_pipeline_layout_);
        s_pipeline_layout_.reset();
    }
    if (s_pipeline_) {
        device->destroyPipeline(s_pipeline_);
        s_pipeline_.reset();
    }
    if (s_wireframe_pipeline_) {
        device->destroyPipeline(s_wireframe_pipeline_);
        s_wireframe_pipeline_.reset();
    }
    s_binding_descs_.clear();
    s_attrib_descs_.clear();
}

void CollisionDebugDraw::uploadForMesh(
    const std::shared_ptr<renderer::Device>& device,
    const CollisionMesh& mesh,
    uint32_t mesh_id,
    CollisionDebugMeshBuffers& out_gpu) {

    if (out_gpu.ready()) return;          // already uploaded
    if (mesh.empty()) return;

    // Read flat triangle list from the CollisionMesh. The class doesn't
    // expose its internal vectors directly so we go through the public
    // accessor pair below — they're added to CollisionMesh.h alongside
    // this file.
    const auto& vertices = mesh.debugVertices();
    const auto& indices  = mesh.debugIndices();
    if (vertices.empty() || indices.size() < 3) return;

    const size_t tri_count = indices.size() / 3;
    std::vector<glm::vec3> positions;
    std::vector<uint32_t>  seg_ids;
    positions.reserve(tri_count * 3);
    seg_ids.reserve(tri_count * 3);

    for (size_t t = 0; t < tri_count; ++t) {
        // Pull all three corners and tag every vertex with the SAME
        // mesh_id. Each CollisionMesh becomes one solid colour after the
        // fragment shader hashes the id, so adjacent meshes appear as
        // distinct flat regions -- "instance segmentation" of the
        // physics world.
        for (int k = 0; k < 3; ++k) {
            int v_idx = indices[3 * t + k];
            if (v_idx < 0 || (size_t)v_idx >= vertices.size()) {
                // Malformed index — skip the whole triangle to keep
                // 3-vertex alignment in the non-indexed draw.
                positions.resize(t * 3);
                seg_ids.resize(t * 3);
                goto stop;
            }
            positions.push_back(vertices[v_idx]);
            seg_ids.push_back(mesh_id);
        }
    }
stop:
    if (positions.empty()) return;

    out_gpu.position_buffer =
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            positions.size() * sizeof(positions[0]),
            positions.data(),
            std::source_location::current());

    out_gpu.triangle_id_buffer =
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            seg_ids.size() * sizeof(seg_ids[0]),
            seg_ids.data(),
            std::source_location::current());

    out_gpu.vertex_count = static_cast<uint32_t>(positions.size());

    // Build the wireframe-overlay index buffer alongside the solid
    // fill data. Each non-indexed triangle (verts 3t, 3t+1, 3t+2)
    // produces three line segments connecting its corners, drawn
    // with LINE_LIST topology against the same position / id vertex
    // buffers as the solid fill -- 6 indices per triangle.
    const uint32_t tri_count_u32 =
        static_cast<uint32_t>(positions.size() / 3);
    if (tri_count_u32 > 0) {
        std::vector<uint32_t> line_indices;
        line_indices.reserve(static_cast<size_t>(tri_count_u32) * 6);
        for (uint32_t t = 0; t < tri_count_u32; ++t) {
            const uint32_t i0 = 3u * t + 0u;
            const uint32_t i1 = 3u * t + 1u;
            const uint32_t i2 = 3u * t + 2u;
            line_indices.push_back(i0); line_indices.push_back(i1);
            line_indices.push_back(i1); line_indices.push_back(i2);
            line_indices.push_back(i2); line_indices.push_back(i0);
        }
        out_gpu.line_index_buffer =
            helper::createUnifiedMeshBuffer(
                device,
                SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
                line_indices.size() * sizeof(line_indices[0]),
                line_indices.data(),
                std::source_location::current());
        out_gpu.line_index_count =
            static_cast<uint32_t>(line_indices.size());
    }
}

void CollisionDebugDraw::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const CollisionDebugMeshBuffers& gpu,
    const glm::mat4& model_transform,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {

    if (!ready() || !gpu.ready()) {
        return;
    }

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        s_pipeline_);

    cmd_buf->setViewports(viewports, 0, uint32_t(viewports.size()));
    cmd_buf->setScissors (scissors,  0, uint32_t(scissors.size()));

    std::vector<std::shared_ptr<renderer::Buffer>> buffers = {
        gpu.position_buffer->buffer,
        gpu.triangle_id_buffer->buffer,
    };
    std::vector<uint64_t> offsets = { 0, 0 };
    cmd_buf->bindVertexBuffers(0, buffers, offsets);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        s_pipeline_layout_,
        desc_set_list);

    glsl::ClusterDebugParams params{};
    params.transform = model_transform;
    cmd_buf->pushConstants(
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
        s_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->draw(gpu.vertex_count);
}

void CollisionDebugDraw::drawWireframe(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const CollisionDebugMeshBuffers& gpu,
    const glm::mat4& model_transform,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {

    if (!wireframeReady() || !gpu.wireframeReady()) {
        return;
    }

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        s_wireframe_pipeline_);

    cmd_buf->setViewports(viewports, 0, uint32_t(viewports.size()));
    cmd_buf->setScissors (scissors,  0, uint32_t(scissors.size()));

    // Same vertex bindings as the solid fill -- the wireframe vert
    // shader is collision_debug.vert (shared) so locations 0 and 15
    // must remain bound. The wire fragment shader ignores the id
    // input; binding it anyway keeps pipeline state consistent.
    std::vector<std::shared_ptr<renderer::Buffer>> buffers = {
        gpu.position_buffer->buffer,
        gpu.triangle_id_buffer->buffer,
    };
    std::vector<uint64_t> offsets = { 0, 0 };
    cmd_buf->bindVertexBuffers(0, buffers, offsets);

    cmd_buf->bindIndexBuffer(
        gpu.line_index_buffer->buffer,
        /*offset=*/0,
        renderer::IndexType::UINT32);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        s_pipeline_layout_,
        desc_set_list);

    glsl::ClusterDebugParams params{};
    params.transform = model_transform;
    cmd_buf->pushConstants(
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
        s_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->drawIndexed(gpu.line_index_count);
}

} // namespace helper
} // namespace engine
