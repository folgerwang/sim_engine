#pragma once
//
// scene_grid.h  --  Editor reference grid + ruler for "Create Scene" mode.
//
// Draws a flat ground grid centred at the world origin in the XZ plane
// (Y up), 200 x 200 metres (-100..+100 on both X and Z).  The grid is a
// single flat quad whose lines are computed analytically in the fragment
// shader (scene_grid.frag) using screen-space derivatives, so every line
// stays ~1 pixel wide at any distance -- no aliasing / moire like raw GPU
// line primitives.  Three tiers are drawn:
//
//   * minor   (every 1 m)              -- dim grey
//   * major   (every 10 m)             -- brighter grey
//   * centre axes (X at z=0, Z at x=0) -- red / blue respectively
//
// In addition a 3-D coordinate-system gizmo is drawn at the world origin:
// three solid colour-coded arrows (X = red, Y = green / up, Z = blue) so
// the axes and handedness are unambiguous in the editor.  These use a
// second solid-colour TRIANGLE_LIST pipeline (depth-write ON) so they read
// as real geometry, while the ground grid quad uses a no-depth-write
// alpha-blended pipeline.
//
// Rendering mirrors helper::CollisionDebugDraw: static graphics pipelines
// shared across the whole app, plus small vertex buffers built lazily the
// first time the grid is drawn.
//
#include "renderer/renderer.h"

namespace engine {
namespace helper {

class SceneGrid {
public:
    // One-time pipeline + pipeline-layout construction.  Call once from
    // the application after the descriptor-set layouts and framebuffer
    // formats are known (same site as CollisionDebugDraw::initStaticMembers).
    //   grid_pipeline_info — used for the alpha-blended ground grid quad;
    //                        pass the no-depth-write pipeline info so the
    //                        grid's transparent gaps don't occlude geometry.
    //   axis_pipeline_info — used for the solid origin gizmo arrows; pass the
    //                        normal (depth-write ON) pipeline info so the
    //                        arrows occlude correctly as real geometry.
    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const renderer::GraphicPipelineInfo& grid_pipeline_info,
        const renderer::GraphicPipelineInfo& axis_pipeline_info,
        const renderer::PipelineRenderbufferFormats& frame_buffer_format);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static bool ready() { return s_pipeline_ != nullptr; }

    // Draw the grid.  desc_set_list must include at least VIEW_PARAMS_SET so
    // the shaders can read the camera SSBO (view-proj in the vertex stage and
    // the camera position for the distance fade in the fragment stage).  Pass
    // the same { pbr_global, view_camera } pair the collision overlay uses.
    // The grid geometry is built on the first call.
    static void draw(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors,
        const glm::mat4& model_transform = glm::mat4(1.0f));

private:
    // Generate the grid quad + axis gizmo geometry (idempotent) into the
    // GPU vertex buffers.
    static void buildGeometry(const std::shared_ptr<renderer::Device>& device);

    // ── Ground grid quad (analytic, alpha-blended) ───────────────────────
    static std::vector<renderer::VertexInputBindingDescription>   s_binding_descs_;
    static std::vector<renderer::VertexInputAttributeDescription> s_attrib_descs_;
    static std::shared_ptr<renderer::PipelineLayout>              s_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline>                    s_pipeline_;
    static std::shared_ptr<renderer::BufferInfo>                  s_position_buffer_;
    static uint32_t                                               s_vertex_count_;

    // ── Origin XYZ gizmo (solid colour-per-vertex arrows) ────────────────
    static std::vector<renderer::VertexInputBindingDescription>   s_axis_binding_descs_;
    static std::vector<renderer::VertexInputAttributeDescription> s_axis_attrib_descs_;
    static std::shared_ptr<renderer::Pipeline>                    s_axis_pipeline_;
    static std::shared_ptr<renderer::BufferInfo>                  s_axis_position_buffer_;
    static std::shared_ptr<renderer::BufferInfo>                  s_axis_color_buffer_;
    static uint32_t                                               s_axis_vertex_count_;

    // Grid extent: the quad spans -kHalfExtent..+kHalfExtent on X and Z
    // (200 x 200 m).  Line spacing / major cadence live in scene_grid.frag.
    static constexpr float    kHalfExtent = 100.0f;  // metres each side of 0
};

} // namespace helper
} // namespace engine
