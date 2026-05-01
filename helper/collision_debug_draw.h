#pragma once
//
// collision_debug_draw.h  --  Debug-visualisation draw path for static
//                             collision meshes.
//
// Mirrors ClusterDebugDraw: owns a single static graphics pipeline +
// pipeline layout shared across every CollisionMesh, plus a per-mesh
// GPU buffer pair (positions + per-triangle ids) populated lazily on
// first draw. Renders each collision triangle flat-shaded with a hash
// of its triangle index so adjacent triangles look different and the
// covered surface is eyeball-readable.
//
// Memory note: the per-mesh debug buffers expand the mesh by a factor
// of 3 (one fresh vertex per triangle corner) so flat interpolation of
// the triangle id is correct without relying on provoking-vertex rules.
// Only allocated when the user enables collision debug, so the cost is
// limited to that mode.
//
#include "renderer/renderer.h"

namespace engine {
namespace helper {

class CollisionMesh;

// Per-CollisionMesh GPU state used only by the collision-debug draw
// path. Empty until CollisionDebugDraw::uploadForMesh() runs.
struct CollisionDebugMeshBuffers {
    std::shared_ptr<renderer::BufferInfo> position_buffer;
    std::shared_ptr<renderer::BufferInfo> triangle_id_buffer;
    uint32_t vertex_count = 0;          // == triangle_count * 3

    bool ready() const {
        return position_buffer && triangle_id_buffer && vertex_count > 0;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

class CollisionDebugDraw {
public:
    // One-time pipeline + pipeline-layout construction. Mirrors
    // ClusterDebugDraw::initStaticMembers — invoke once from the
    // application after the descriptor-set layouts and framebuffer
    // formats are known.
    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::PipelineRenderbufferFormats& frame_buffer_format);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    // Expand a CollisionMesh's flat triangle list into two dense GPU
    // vertex buffers:
    //   - positions   (vec3, triangle_count * 3)
    //   - triangle ids (uint, triangle_count * 3, same value across a tri)
    // Idempotent: if `out_gpu` is already populated, this is a no-op.
    static void uploadForMesh(
        const std::shared_ptr<renderer::Device>& device,
        const CollisionMesh& mesh,
        CollisionDebugMeshBuffers& out_gpu);

    // Draw one collision mesh with per-triangle hashed colour.
    // Bypasses the normal primitive loop — the caller should not have
    // bound a pipeline. desc_set_list must include at least
    // VIEW_PARAMS_SET so the vertex shader can read the camera SSBO.
    static void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const CollisionDebugMeshBuffers& gpu,
        const glm::mat4& model_transform,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors);

    static bool ready() { return s_pipeline_ != nullptr; }

private:
    static std::vector<renderer::VertexInputBindingDescription>   s_binding_descs_;
    static std::vector<renderer::VertexInputAttributeDescription> s_attrib_descs_;
    static std::shared_ptr<renderer::PipelineLayout>              s_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline>                    s_pipeline_;
};

} // namespace helper
} // namespace engine
