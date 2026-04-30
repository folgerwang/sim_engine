#pragma once
//
// cluster_renderer.h — GPU-driven cluster culling & rendering.
//
// "Nanite-lite" step 2: takes the CPU-side ClusterMesh sidecar built by
// helper::buildClusterMesh() and uploads it to GPU SSBOs. At render time,
// a compute shader frustum+cone-culls every cluster and emits visible
// clusters into an indirect draw buffer.
//
// All cluster data from all meshes is merged into single flat SSBOs so the
// cull pass is ONE compute dispatch (not one per mesh). This keeps the
// per-frame Vulkan command overhead constant regardless of mesh count.
//
// Lifecycle:
//   1. Construct once (creates pipelines, descriptor set layouts).
//   2. Call uploadMeshClusters() per drawable mesh — appends to CPU staging.
//   3. Call finalizeUploads() once all meshes are uploaded — creates GPU SSBOs.
//   4. Each frame: cull() → draw() in the forward pass.
//   5. recreate() on swap chain resize (re-alloc descriptor sets).
//   6. destroy() at shutdown.
//

#include "renderer/renderer.h"
#include "helper/cluster_mesh.h"
#include "shaders/global_definition.glsl.h"

#include <vector>
#include <memory>
#include <unordered_map>

namespace engine {

namespace game_object { struct DrawableData; }

namespace scene_rendering {

class ClusterRenderer {
    // Compute culling pipeline.
    std::shared_ptr<renderer::DescriptorSetLayout> cull_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout>      cull_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            cull_pipeline_;

    // Device + pool cached for descriptor allocation.
    std::shared_ptr<renderer::Device>         device_;
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool_;

    // ── CPU staging (populated by uploadMeshClusters, consumed by finalize) ──
    std::vector<glsl::ClusterCullInfo>        staging_cull_infos_;
    std::vector<glsl::ClusterDrawInfo>        staging_draw_infos_;
    std::vector<glsl::BindlessMaterialParams> staging_material_params_;
    // Base-colour texture views staged for the bindless sampler array (binding 2).
    std::vector<std::shared_ptr<renderer::ImageView>> staging_tex_views_;
    // Dedup map: ImageView raw pointer → slot index in staging_tex_views_.
    std::unordered_map<renderer::ImageView*, int> staging_tex_slot_map_;

    // Normal-map texture views staged for the bindless sampler array (binding 3).
    std::vector<std::shared_ptr<renderer::ImageView>> staging_normal_tex_views_;
    std::unordered_map<renderer::ImageView*, int>     staging_normal_slot_map_;

    uint32_t uploaded_mesh_count_ = 0;

    // ── Shared sampler + dummy 1×1 white texture (fallback for empty slots) ──
    std::shared_ptr<renderer::Sampler>    default_sampler_;
    renderer::TextureInfo                 dummy_texture_;

    // ── Merged VB/IB staging for bindless rendering ──
    // Vertices are transformed to world space during upload.
    struct BindlessVertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
    };
    std::vector<BindlessVertex> staging_vertices_;
    std::vector<uint32_t>       staging_indices_;

    // ── Per-mesh cluster tracking (for visibility feedback) ──
    struct MeshClusterRange {
        uint32_t cluster_start;
        uint32_t cluster_count;
    };
    std::vector<MeshClusterRange> mesh_cluster_ranges_;
    std::vector<uint32_t>         cluster_to_mesh_;    // global cluster idx → mesh idx
    std::vector<bool>             mesh_visible_;        // per-mesh visibility (prev frame)

    // ── Merged GPU buffers (created by finalizeUploads) ──
    renderer::BufferInfo cull_info_buffer_;       // ClusterCullInfo[]
    renderer::BufferInfo draw_info_buffer_;       // ClusterDrawInfo[]
    renderer::BufferInfo material_params_buffer_; // BindlessMaterialParams[]
    renderer::BufferInfo indirect_draw_buffer_; // DrawIndexedIndirectCommand[]
    renderer::BufferInfo draw_count_buffer_;    // single uint (host-visible)
    renderer::BufferInfo visible_buffer_;       // visible cluster indices
    std::shared_ptr<renderer::DescriptorSet> cull_desc_set_;

    // ── Merged vertex/index buffers for bindless rendering ──
    renderer::BufferInfo merged_vertex_buffer_; // BindlessVertex[]
    renderer::BufferInfo merged_index_buffer_;  // uint32_t[]
    uint32_t total_merged_vertices_ = 0;
    uint32_t total_merged_indices_  = 0;
    uint32_t total_materials_        = 0;  // set by finalizeUploads
    uint32_t total_textures_         = 0;  // set by finalizeUploads
    uint32_t total_normal_textures_  = 0;  // set by finalizeUploads

    // ── Bindless rendering pipeline ──
    std::shared_ptr<renderer::DescriptorSetLayout> bindless_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout>      bindless_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            bindless_pipeline_;
    std::shared_ptr<renderer::DescriptorSet>       bindless_desc_set_;

    bool gpu_ready_ = false;  // set after finalizeUploads()

    // Debug: small sample of cluster bounds for CPU-side frustum verification.
    static constexpr uint32_t kDebugSampleCount = 500;
    std::vector<glsl::ClusterCullInfo> debug_sample_clusters_;

    // Global stats for debug UI.
    uint32_t total_clusters_all_meshes_ = 0;
    uint32_t total_visible_all_meshes_ = 0;

    // Polygon (triangle) counts for debug UI.
    uint64_t total_triangles_all_meshes_ = 0;  // computed once at finalize
    uint64_t visible_triangles_ = 0;           // updated each frame from readback
    std::vector<uint32_t> cluster_tri_counts_;  // per-cluster triangle count

    bool enabled_ = true;   // master toggle (GPU cluster culling on by default)
    bool cpu_cull_mode_ = false;          // true = CPU frustum cull, false = GPU compute cull
    bool debug_draw_bbox_ = false;        // draw cluster bounding boxes
    bool debug_distance_cull_ = false;    // debug: set use_bvh=3 for distance-based cull

    // Debug: last-used VP matrix and camera pos for display.
    glm::mat4 debug_last_vp_ = glm::mat4(1.0f);
    glm::vec3 debug_last_cam_pos_ = glm::vec3(0.0f);

    // Debug: first cluster's center + radius (read back from GPU).
    glm::vec4 debug_first_cluster_bounds_ = glm::vec4(0.0f);

    // Debug: raw LOCAL-space data from the very first cluster uploaded,
    // BEFORE model_transform is applied. If this is NaN, buildClusterMesh
    // is the problem. If this is valid but debug_first_cluster_bounds_ is
    // NaN, the model_transform is the problem.
    glm::vec4 debug_first_local_bounds_ = glm::vec4(0.0f);
    glm::vec4 debug_first_model_diag_   = glm::vec4(0.0f);

public:
    ClusterRenderer(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Upload one mesh's cluster data (appends to CPU staging arrays).
    // Call after helper::buildClusterMesh() has run on the mesh.
    // cluster_prim_map[i] gives the primitive index for cluster i.
    // Allows one call per FBX mesh (not per primitive) while still assigning
    // the correct per-cluster material. Build with MeshInfo::cluster_prim_map_.
    void uploadMeshClusters(
        const helper::ClusterMesh& cluster_mesh,
        const game_object::DrawableData& drawable_data,
        uint32_t mesh_idx,
        const std::vector<uint32_t>& cluster_prim_map,
        const glm::mat4& model_transform);

    // Create merged GPU SSBOs + descriptor set from all staged data.
    // Call once after all uploadMeshClusters() calls are done.
    void finalizeUploads();

    // Initialise the bindless graphics pipeline. Call once after
    // finalizeUploads() and after descriptor-set layouts / render
    // formats are known. `global_desc_set_layouts` should include
    // PBR_GLOBAL_PARAMS_SET and VIEW_PARAMS_SET at minimum.
    void initBindlessPipeline(
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::PipelineRenderbufferFormats& framebuffer_format);

    // Dispatch cluster culling compute shader (single dispatch for all meshes).
    void cull(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::mat4& view_proj,
        const glm::vec3& camera_pos);

    // Bindless draw: single drawIndexedIndirectCount for all visible clusters.
    // Returns visible cluster count (from previous frame's readback).
    uint32_t draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors);

    // Swap chain resize.
    void recreate(
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    void destroy();

    // ── Indirect draw verification ────────────────────────────────────────
    // Call this once per frame (or just on the first few frames) AFTER
    // cull() has executed and the GPU fence for that frame has been waited.
    // It maps the HOST_VISIBLE buffers and validates every indirect command
    // the compute shader emitted, printing any anomaly to stderr.
    //
    // Returns true if all commands are clean, false if any problem is found.
    // Safe to call every frame during debugging — reads only, no GPU stall.
    bool verifyIndirectCommands(uint32_t max_commands_to_check = 32) const;

    // Reset debug stats to zero (called when cluster rendering is disabled).
    void resetStats() {
        total_visible_all_meshes_ = 0;
        visible_triangles_ = 0;
        mesh_visible_.assign(mesh_visible_.size(), false);
    }

    // Accessors for debug UI.
    bool& getEnabled() { return enabled_; }
    bool isEnabled() const { return enabled_; }
    bool& getCpuCullMode() { return cpu_cull_mode_; }
    bool isCpuCullMode() const { return cpu_cull_mode_; }
    bool& getDebugDrawBBox() { return debug_draw_bbox_; }
    bool& getDebugDistanceCull() { return debug_distance_cull_; }
    const std::vector<glsl::ClusterCullInfo>& getDebugSampleClusters() const {
        return debug_sample_clusters_;
    }
    const glm::mat4& getDebugVP() const { return debug_last_vp_; }
    const glm::vec3& getDebugCamPos() const { return debug_last_cam_pos_; }
    const glm::vec4& getDebugFirstCluster() const { return debug_first_cluster_bounds_; }
    const glm::vec4& getDebugFirstLocal() const { return debug_first_local_bounds_; }
    const glm::vec4& getDebugFirstModelDiag() const { return debug_first_model_diag_; }
    uint32_t getTotalClusters() const { return total_clusters_all_meshes_; }
    uint32_t getTotalVisible() const { return total_visible_all_meshes_; }
    uint64_t getTotalTriangles() const { return total_triangles_all_meshes_; }
    uint64_t getVisibleTriangles() const { return visible_triangles_; }
    float getCullPercentage() const {
        if (total_clusters_all_meshes_ == 0) return 0.0f;
        return 100.0f * (1.0f - float(total_visible_all_meshes_) /
                                 float(total_clusters_all_meshes_));
    }
    uint32_t getMeshCount() const { return uploaded_mesh_count_; }

    // Per-mesh visibility from previous frame's cluster cull readback.
    // Returns true (visible) for unknown mesh indices as a safe default.
    bool isMeshVisible(uint32_t global_mesh_idx) const {
        if (global_mesh_idx >= mesh_visible_.size()) return true;
        return mesh_visible_[global_mesh_idx];
    }
    uint32_t getRegisteredMeshCount() const {
        return static_cast<uint32_t>(mesh_cluster_ranges_.size());
    }
};

} // namespace scene_rendering
} // namespace engine
