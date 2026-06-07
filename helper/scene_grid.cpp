#include "scene_grid.h"

#include <vector>

#include "shaders/global_definition.glsl.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"   // createUnifiedMeshBuffer

namespace engine {
namespace {

static auto createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    push_const_range.offset = 0;
    // Reuse ClusterDebugParams { mat4 transform; } — matches the shaders.
    push_const_range.size = sizeof(glsl::ClusterDebugParams);

    renderer::DescriptorSetLayoutList desc_set_layouts =
        global_desc_set_layouts;

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static renderer::ShaderModuleList getShaderModules(
    std::shared_ptr<renderer::Device> device,
    const char* vert_spv,
    const char* frag_spv) {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device, vert_spv,
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device, frag_spv,
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    return shader_modules;
}

// Analytic ground grid — single alpha-blended quad, no depth write.
static auto createGridPipeline(
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
    rasterization_state_info.override_double_sided = true;
    rasterization_state_info.double_sided          = true;
    rasterization_state_info.override_depth_bias = true;
    rasterization_state_info.depth_bias_enable = true;
    rasterization_state_info.depth_bias_constant_factor = -1.0f;
    rasterization_state_info.depth_bias_slope_factor    = -1.0f;

    renderer::GraphicPipelineInfo blended_info = graphic_pipeline_info;
    blended_info.blend_state_info =
        std::make_shared<renderer::PipelineColorBlendStateCreateInfo>(
            renderer::helper::fillPipelineColorBlendStateCreateInfo({
                renderer::helper::fillPipelineColorBlendAttachmentState(
                    SET_FLAG_BIT(ColorComponent, ALL_BITS),
                    /*blend_enable=*/true,
                    renderer::BlendFactor::SRC_ALPHA,
                    renderer::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    renderer::BlendOp::ADD,
                    renderer::BlendFactor::ONE,
                    renderer::BlendFactor::ZERO,
                    renderer::BlendOp::ADD)
            }));

    return device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        input_assembly,
        blended_info,
        getShaderModules(device, "scene_grid_vert.spv", "scene_grid_frag.spv"),
        frame_buffer_format,
        rasterization_state_info,
        std::source_location::current());
}

// Origin XYZ gizmo — solid colour-per-vertex arrows, depth-write ON.
static auto createAxisPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::vector<renderer::VertexInputBindingDescription>& binding_descs,
    const std::vector<renderer::VertexInputAttributeDescription>& attribute_descs,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    // Double-sided so winding doesn't matter for the generated boxes /
    // pyramids; depth state (write ON) is inherited from the supplied
    // graphic_pipeline_info so the arrows behave like real solid geometry.
    renderer::RasterizationStateOverride rasterization_state_info;
    rasterization_state_info.override_double_sided = true;
    rasterization_state_info.double_sided          = true;

    return device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        input_assembly,
        graphic_pipeline_info,
        getShaderModules(device, "scene_axis_vert.spv", "scene_axis_frag.spv"),
        frame_buffer_format,
        rasterization_state_info,
        std::source_location::current());
}

} // anonymous namespace

namespace helper {

std::vector<renderer::VertexInputBindingDescription>   SceneGrid::s_binding_descs_;
std::vector<renderer::VertexInputAttributeDescription> SceneGrid::s_attrib_descs_;
std::shared_ptr<renderer::PipelineLayout>              SceneGrid::s_pipeline_layout_;
std::shared_ptr<renderer::Pipeline>                    SceneGrid::s_pipeline_;
std::shared_ptr<renderer::BufferInfo>                  SceneGrid::s_position_buffer_;
uint32_t                                               SceneGrid::s_vertex_count_ = 0;

std::vector<renderer::VertexInputBindingDescription>   SceneGrid::s_axis_binding_descs_;
std::vector<renderer::VertexInputAttributeDescription> SceneGrid::s_axis_attrib_descs_;
std::shared_ptr<renderer::Pipeline>                    SceneGrid::s_axis_pipeline_;
std::shared_ptr<renderer::BufferInfo>                  SceneGrid::s_axis_position_buffer_;
std::shared_ptr<renderer::BufferInfo>                  SceneGrid::s_axis_color_buffer_;
uint32_t                                               SceneGrid::s_axis_vertex_count_ = 0;

void SceneGrid::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const renderer::GraphicPipelineInfo& grid_pipeline_info,
    const renderer::GraphicPipelineInfo& axis_pipeline_info,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {

    s_pipeline_layout_ =
        createPipelineLayout(device, global_desc_set_layouts);

    // ── Ground grid quad: binding 0 = vec3 position ──────────────────────
    {
        renderer::VertexInputBindingDescription bind0;
        bind0.binding    = 0;
        bind0.stride     = sizeof(glm::vec3);
        bind0.input_rate = renderer::VertexInputRate::VERTEX;
        s_binding_descs_.push_back(bind0);

        renderer::VertexInputAttributeDescription attr0;
        attr0.binding  = 0;
        attr0.location = VINPUT_POSITION;            // == 0
        attr0.format   = renderer::Format::R32G32B32_SFLOAT;
        attr0.offset   = 0;
        s_attrib_descs_.push_back(attr0);
    }
    s_pipeline_ = createGridPipeline(
        device, s_pipeline_layout_, grid_pipeline_info,
        s_binding_descs_, s_attrib_descs_, frame_buffer_format);

    // ── Origin gizmo: binding 0 = vec3 position, binding 1 = vec3 colour ─
    {
        renderer::VertexInputBindingDescription bind0;
        bind0.binding    = 0;
        bind0.stride     = sizeof(glm::vec3);
        bind0.input_rate = renderer::VertexInputRate::VERTEX;
        s_axis_binding_descs_.push_back(bind0);

        renderer::VertexInputAttributeDescription attr0;
        attr0.binding  = 0;
        attr0.location = VINPUT_POSITION;            // == 0
        attr0.format   = renderer::Format::R32G32B32_SFLOAT;
        attr0.offset   = 0;
        s_axis_attrib_descs_.push_back(attr0);

        renderer::VertexInputBindingDescription bind1;
        bind1.binding    = 1;
        bind1.stride     = sizeof(glm::vec3);
        bind1.input_rate = renderer::VertexInputRate::VERTEX;
        s_axis_binding_descs_.push_back(bind1);

        renderer::VertexInputAttributeDescription attr1;
        attr1.binding  = 1;
        attr1.location = VINPUT_COLOR;               // == 6
        attr1.format   = renderer::Format::R32G32B32_SFLOAT;
        attr1.offset   = 0;
        s_axis_attrib_descs_.push_back(attr1);
    }
    s_axis_pipeline_ = createAxisPipeline(
        device, s_pipeline_layout_, axis_pipeline_info,
        s_axis_binding_descs_, s_axis_attrib_descs_, frame_buffer_format);
}

void SceneGrid::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    if (s_position_buffer_)      { s_position_buffer_->destroy(device);      s_position_buffer_.reset(); }
    if (s_axis_position_buffer_) { s_axis_position_buffer_->destroy(device); s_axis_position_buffer_.reset(); }
    if (s_axis_color_buffer_)    { s_axis_color_buffer_->destroy(device);    s_axis_color_buffer_.reset(); }
    s_vertex_count_      = 0;
    s_axis_vertex_count_ = 0;

    if (s_pipeline_)      { device->destroyPipeline(s_pipeline_);      s_pipeline_.reset(); }
    if (s_axis_pipeline_) { device->destroyPipeline(s_axis_pipeline_); s_axis_pipeline_.reset(); }
    if (s_pipeline_layout_) { device->destroyPipelineLayout(s_pipeline_layout_); s_pipeline_layout_.reset(); }

    s_binding_descs_.clear();
    s_attrib_descs_.clear();
    s_axis_binding_descs_.clear();
    s_axis_attrib_descs_.clear();
}

void SceneGrid::buildGeometry(
    const std::shared_ptr<renderer::Device>& device) {
    if (s_position_buffer_ && s_vertex_count_ > 0 &&
        s_axis_position_buffer_ && s_axis_vertex_count_ > 0) {
        return;  // already built
    }

    // ── Ground grid quad (XZ plane, Y up) ────────────────────────────────
    {
        const float e = kHalfExtent;
        const std::vector<glm::vec3> positions = {
            glm::vec3(-e, 0.0f, -e),
            glm::vec3( e, 0.0f, -e),
            glm::vec3( e, 0.0f,  e),
            glm::vec3(-e, 0.0f, -e),
            glm::vec3( e, 0.0f,  e),
            glm::vec3(-e, 0.0f,  e),
        };
        s_position_buffer_ =
            helper::createUnifiedMeshBuffer(
                device,
                SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                positions.size() * sizeof(positions[0]),
                positions.data(),
                std::source_location::current());
        s_vertex_count_ = static_cast<uint32_t>(positions.size());
    }

    // ── Origin XYZ gizmo: three solid arrows (shaft box + arrowhead) ─────
    {
        std::vector<glm::vec3> apos;
        std::vector<glm::vec3> acol;

        auto addTri = [&](const glm::vec3& a, const glm::vec3& b,
                          const glm::vec3& c, const glm::vec3& col) {
            apos.push_back(a); apos.push_back(b); apos.push_back(c);
            acol.push_back(col); acol.push_back(col); acol.push_back(col);
        };
        auto addQuad = [&](const glm::vec3& a, const glm::vec3& b,
                           const glm::vec3& c, const glm::vec3& d,
                           const glm::vec3& col) {
            addTri(a, b, c, col);
            addTri(a, c, d, col);
        };

        // Build one arrow along orthonormal basis (A = along, U/V = perp).
        auto addArrow = [&](const glm::vec3& A, const glm::vec3& U,
                            const glm::vec3& V, const glm::vec3& col) {
            const float shaft_len = 2.6f;   // shaft length (m)
            const float r         = 0.03f;  // shaft half-thickness (m)
            const float head_len  = 0.5f;   // arrowhead length (m)
            const float hr        = 0.10f;  // arrowhead base half-width (m)

            auto P = [&](float s, float iu, float iv) {
                return A * s + U * (iu * r) + V * (iv * r);
            };
            const glm::vec3 c000 = P(0.0f, -1, -1), c001 = P(0.0f, -1, +1);
            const glm::vec3 c011 = P(0.0f, +1, +1), c010 = P(0.0f, +1, -1);
            const glm::vec3 c100 = P(shaft_len, -1, -1), c101 = P(shaft_len, -1, +1);
            const glm::vec3 c111 = P(shaft_len, +1, +1), c110 = P(shaft_len, +1, -1);

            // Six faces of the shaft box (double-sided → winding ignored).
            addQuad(c000, c010, c110, c100, col);
            addQuad(c001, c101, c111, c011, col);
            addQuad(c000, c100, c101, c001, col);
            addQuad(c010, c011, c111, c110, col);
            addQuad(c000, c001, c011, c010, col);   // near cap
            addQuad(c100, c110, c111, c101, col);   // far cap

            // Arrowhead pyramid at the shaft tip.
            const glm::vec3 b0 = A * shaft_len + U * (-hr) + V * (-hr);
            const glm::vec3 b1 = A * shaft_len + U * (+hr) + V * (-hr);
            const glm::vec3 b2 = A * shaft_len + U * (+hr) + V * (+hr);
            const glm::vec3 b3 = A * shaft_len + U * (-hr) + V * (+hr);
            const glm::vec3 apex = A * (shaft_len + head_len);
            addQuad(b0, b1, b2, b3, col);           // base
            addTri(b0, b1, apex, col);
            addTri(b1, b2, apex, col);
            addTri(b2, b3, apex, col);
            addTri(b3, b0, apex, col);
        };

        const glm::vec3 kRed  (0.85f, 0.18f, 0.18f);  // X
        const glm::vec3 kGreen(0.20f, 0.80f, 0.25f);  // Y (up)
        const glm::vec3 kBlue (0.18f, 0.34f, 0.90f);  // Z

        addArrow(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1), kRed);
        addArrow(glm::vec3(0, 1, 0), glm::vec3(0, 0, 1), glm::vec3(1, 0, 0), kGreen);
        addArrow(glm::vec3(0, 0, 1), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), kBlue);

        s_axis_position_buffer_ =
            helper::createUnifiedMeshBuffer(
                device,
                SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                apos.size() * sizeof(apos[0]),
                apos.data(),
                std::source_location::current());
        s_axis_color_buffer_ =
            helper::createUnifiedMeshBuffer(
                device,
                SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                acol.size() * sizeof(acol[0]),
                acol.data(),
                std::source_location::current());
        s_axis_vertex_count_ = static_cast<uint32_t>(apos.size());
    }
}

void SceneGrid::draw(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    const glm::mat4& model_transform) {

    if (!ready()) {
        return;
    }

    buildGeometry(device);

    cmd_buf->setViewports(viewports, 0, uint32_t(viewports.size()));
    cmd_buf->setScissors (scissors,  0, uint32_t(scissors.size()));

    glsl::ClusterDebugParams params{};
    params.transform = model_transform;

    // ── Ground grid quad ─────────────────────────────────────────────────
    if (s_position_buffer_ && s_vertex_count_ > 0) {
        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::GRAPHICS, s_pipeline_);

        std::vector<std::shared_ptr<renderer::Buffer>> buffers = {
            s_position_buffer_->buffer,
        };
        std::vector<uint64_t> offsets = { 0 };
        cmd_buf->bindVertexBuffers(0, buffers, offsets);
        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::GRAPHICS,
            s_pipeline_layout_, desc_set_list);
        cmd_buf->pushConstants(
            SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
            s_pipeline_layout_, &params, sizeof(params));
        cmd_buf->draw(s_vertex_count_);
    }

    // ── Origin XYZ gizmo ─────────────────────────────────────────────────
    if (s_axis_pipeline_ && s_axis_position_buffer_ &&
        s_axis_color_buffer_ && s_axis_vertex_count_ > 0) {
        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::GRAPHICS, s_axis_pipeline_);

        std::vector<std::shared_ptr<renderer::Buffer>> buffers = {
            s_axis_position_buffer_->buffer,
            s_axis_color_buffer_->buffer,
        };
        std::vector<uint64_t> offsets = { 0, 0 };
        cmd_buf->bindVertexBuffers(0, buffers, offsets);
        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::GRAPHICS,
            s_pipeline_layout_, desc_set_list);
        cmd_buf->pushConstants(
            SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
            s_pipeline_layout_, &params, sizeof(params));
        cmd_buf->draw(s_axis_vertex_count_);
    }
}

} // namespace helper
} // namespace engine
