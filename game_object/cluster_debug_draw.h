#pragma once
//
// cluster_debug_draw.h  --  "Nanite-lite" cluster visualisation draw path.
//
// Parallel to ShapeBase: owns a single static graphics pipeline + pipeline
// layout (shared across every mesh) plus a per-MeshInfo GPU buffer pair
// that is only populated when --cluster-debug is active. Renders each
// triangle flat-shaded with a hash of the cluster id it was packed into
// by helper::buildClusterMesh(), so cluster boundaries are eyeball-visible.
//
// Memory note: the per-mesh buffers expand the mesh by a factor of 3 (one
// fresh vertex per triangle corner) so every triangle's three corners can
// carry the same cluster id without relying on the provoking-vertex rules.
// Only allocated in debug mode, so the waste is tolerable.
//
#include "renderer/renderer.h"
#include "helper/cluster_mesh.h"

namespace engine {
namespace game_object {

// Per-MeshInfo GPU state used only by the --cluster-debug draw path. Empty
// unless helper::clusterRenderingEnabled() was true at mesh load AND
// helper::buildClusterMesh() produced a non-empty sidecar.
struct ClusterDebugMeshBuffers {
    std::shared_ptr<renderer::BufferInfo> position_buffer;
    std::shared_ptr<renderer::BufferInfo> cluster_id_buffer;
    uint32_t vertex_count = 0;  // == total_triangles * 3

    bool ready() const {
        return position_buffer && cluster_id_buffer && vertex_count > 0;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

class ClusterDebugDraw {
public:
    // One-time pipeline + pipeline-layout construction. Mirrors
    // ShapeBase::initStaticMembers -- invoke once from the application
    // constructor after the descriptor-set layouts and framebuffer formats
    // are known.
    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::PipelineRenderbufferFormats& frame_buffer_format);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    // Expand the ClusterMesh sidecar into two dense GPU vertex buffers:
    //   - positions (vec3, total_triangles * 3)
    //   - cluster ids (uint, total_triangles * 3, same value across a tri)
    // No-op if the sidecar is empty.
    static void uploadForMesh(
        const std::shared_ptr<renderer::Device>& device,
        const helper::ClusterMesh& cluster_mesh,
        ClusterDebugMeshBuffers& out_gpu);

    // Draw one mesh with per-cluster flat color. Bypasses the normal
    // primitive loop -- the caller should not have bound a pipeline.
    // desc_set_list is expected to include at least the VIEW_PARAMS_SET
    // entry (camera info), matching the cluster_debug.vert SSBO binding.
    static void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const ClusterDebugMeshBuffers& gpu,
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

} // namespace game_object
} // namespace engine
