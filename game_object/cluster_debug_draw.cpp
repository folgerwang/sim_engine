#include "cluster_debug_draw.h"
#include "shaders/global_definition.glsl.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"   // createUnifiedMeshBuffer
#include "helper/mesh_tool.h"       // helper::Mesh, Face, VertexStruct

// Vertex-input location for the per-vertex cluster id in cluster_debug.vert.
// Chosen to sit outside both the standard vertex-input slots (0..9) and the
// instance-input slots (10..14) defined in global_definition.glsl.h.
static constexpr uint32_t kClusterIdLocation = 15;

namespace engine {
namespace {

static auto createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    push_const_range.offset = 0;
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
            "cluster_debug_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "cluster_debug_frag.spv",
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
    renderer::RasterizationStateOverride rasterization_state_info;

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

} // anonymous namespace

namespace game_object {

std::vector<renderer::VertexInputBindingDescription>   ClusterDebugDraw::s_binding_descs_;
std::vector<renderer::VertexInputAttributeDescription> ClusterDebugDraw::s_attrib_descs_;
std::shared_ptr<renderer::PipelineLayout>              ClusterDebugDraw::s_pipeline_layout_;
std::shared_ptr<renderer::Pipeline>                    ClusterDebugDraw::s_pipeline_;

void ClusterDebugMeshBuffers::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    if (position_buffer) {
        position_buffer->destroy(device);
        position_buffer.reset();
    }
    if (cluster_id_buffer) {
        cluster_id_buffer->destroy(device);
        cluster_id_buffer.reset();
    }
    vertex_count = 0;
}

void ClusterDebugDraw::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {

    // Binding 0 -- tight vec3 position buffer (per-triangle-expanded).
    renderer::VertexInputBindingDescription bind0;
    bind0.binding    = 0;
    bind0.stride     = sizeof(glm::vec3);
    bind0.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(bind0);

    renderer::VertexInputAttributeDescription attr0;
    attr0.binding  = 0;
    attr0.location = VINPUT_POSITION;               // matches cluster_debug.vert
    attr0.format   = renderer::Format::R32G32B32_SFLOAT;
    attr0.offset   = 0;
    s_attrib_descs_.push_back(attr0);

    // Binding 1 -- tight uint32 cluster_id buffer (per-triangle-expanded).
    renderer::VertexInputBindingDescription bind1;
    bind1.binding    = 1;
    bind1.stride     = sizeof(uint32_t);
    bind1.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(bind1);

    renderer::VertexInputAttributeDescription attr1;
    attr1.binding  = 1;
    attr1.location = kClusterIdLocation;            // == 15, matches shader
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
}

void ClusterDebugDraw::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    if (s_pipeline_layout_) {
        device->destroyPipelineLayout(s_pipeline_layout_);
        s_pipeline_layout_.reset();
    }
    if (s_pipeline_) {
        device->destroyPipeline(s_pipeline_);
        s_pipeline_.reset();
    }
    s_binding_descs_.clear();
    s_attrib_descs_.clear();
}

void ClusterDebugDraw::uploadForMesh(
    const std::shared_ptr<renderer::Device>& device,
    const helper::ClusterMesh& cluster_mesh,
    ClusterDebugMeshBuffers& out_gpu) {

    // Early out if the sidecar is empty or disconnected from its source.
    if (cluster_mesh.empty() || !cluster_mesh.source ||
        !cluster_mesh.source->isValid()) {
        return;
    }

    const auto& source      = *cluster_mesh.source;
    const auto& source_verts = *source.vertex_data_ptr;
    const auto& source_faces = *source.faces_ptr;

    // Triangle-expanded layout: 3 fresh vertices per triangle, so flat
    // interpolation of cluster_id cannot be contaminated by cross-cluster
    // vertex sharing at boundaries.
    std::vector<glm::vec3> positions;
    std::vector<uint32_t>  cluster_ids;
    positions.reserve(cluster_mesh.total_triangles * 3u);
    cluster_ids.reserve(cluster_mesh.total_triangles * 3u);

    for (uint32_t c_idx = 0; c_idx < cluster_mesh.clusters.size(); ++c_idx) {
        const auto& cluster = cluster_mesh.clusters[c_idx];
        for (uint32_t face_idx : cluster.face_indices) {
            // Guard against a stale sidecar -- if the underlying Mesh was
            // rebuilt after clustering, this would be out of bounds.
            if (face_idx >= source_faces.size()) {
                continue;
            }
            const auto& face = source_faces[face_idx];
            for (int k = 0; k < 3; ++k) {
                uint32_t v_idx = face.v_indices[k];
                if (v_idx >= source_verts.size()) {
                    continue;
                }
                positions.push_back(source_verts[v_idx].position);
                cluster_ids.push_back(c_idx);
            }
        }
    }

    if (positions.empty()) {
        return;
    }

    out_gpu.position_buffer =
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            positions.size() * sizeof(positions[0]),
            positions.data(),
            std::source_location::current());

    out_gpu.cluster_id_buffer =
        helper::createUnifiedMeshBuffer(
            device,
            SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            cluster_ids.size() * sizeof(cluster_ids[0]),
            cluster_ids.data(),
            std::source_location::current());

    out_gpu.vertex_count = static_cast<uint32_t>(positions.size());
}

void ClusterDebugDraw::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const ClusterDebugMeshBuffers& gpu,
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
        gpu.cluster_id_buffer->buffer,
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

} // namespace game_object
} // namespace engine
