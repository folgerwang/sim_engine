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
#include <optional>
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
    //   tangent.xyz = world-space tangent (already orthogonalised against normal)
    //   tangent.w   = bitangent sign so the fragment shader can derive
    //                 B = cross(N, tangent.xyz) * tangent.w
    // The tangent is computed at upload time from the source mesh's
    // per-triangle position+UV gradients (Lengyel-style accumulation,
    // normalised + Gram-Schmidt against the vertex normal).  Carrying it as
    // an interpolated vertex attribute lets cluster_bindless.frag apply
    // normal mapping without the per-fragment dFdx/dFdy reconstruction
    // that previously produced fine-grained sparkle on shaded surfaces.
    struct BindlessVertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec4 tangent;
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
    // Parallel buckets for AlphaMode::Blend (glass / windows).  The cull
    // compute routes each visible cluster into either the opaque indirect
    // list above or these translucent ones, based on
    // BindlessMaterialParams.flags & BINDLESS_MAT_TRANSLUCENT.  Drawn in
    // a second pass after the opaque draw with bindless_translucent_pipeline_
    // (alpha blend on, depth-write off).
    renderer::BufferInfo trans_indirect_draw_buffer_;
    renderer::BufferInfo trans_draw_count_buffer_;

    // ── Two-pass occlusion culling (Nanite-style) ────────────────────────
    // Persistent per-cluster visibility bit set across frames.  Sized for
    // ceil(total_clusters / 32) uints; bit N of element N/32 == "cluster N
    // was drawn last frame".  Phase A reads it (and only emits clusters
    // whose bit is set) so previously-visible clusters re-render first and
    // produce a partial depth buffer.  Phase B clears + atomically OR-writes
    // bits for every cluster that survives all three tests (frustum +
    // backface + Hi-Z), so the SAME buffer becomes next frame's Phase A
    // input.  Allocated zero-initialised in finalizeUploads — frame 1's
    // Phase A renders nothing, Phase B (with empty Hi-Z) renders everything,
    // and the buffer is correct from frame 2 onward.
    renderer::BufferInfo visibility_bit_buffer_;
    // Phase A indirect draw output.  Worst-case sized identically to the
    // single-pass indirect_draw_buffer_ (5 uints per cluster).
    renderer::BufferInfo indirect_draw_buffer_phase_a_;
    renderer::BufferInfo draw_count_buffer_phase_a_;

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
    // OIT (Weighted Blended) pipeline for the translucent draw.  Same
    // shader source as bindless_pipeline_ compiled with -DOIT_OUTPUT so
    // it writes vec4 accum + float reveal to two separate render targets
    // (oit_accum_tex_, oit_reveal_tex_) instead of a single colour.
    // Per-attachment blend setup:
    //   accum  attachment: src=ONE,  dst=ONE,                  op=ADD
    //   reveal attachment: src=ZERO, dst=ONE_MINUS_SRC_COLOR,  op=ADD
    // Depth test on, depth write off (translucent surfaces don't occlude
    // one another in the depth buffer).  Resolved by oit_composite_pipeline_
    // in a fullscreen pass after the OIT draw.
    std::shared_ptr<renderer::Pipeline>            bindless_translucent_pipeline_;

    // ── Deferred G-buffer variant ────────────────────────────────────────
    // Same pipeline layout (and bindless descriptor set) as the forward
    // bindless_pipeline_ but built against the application's 3-RT G-buffer
    // format and using cluster_bindless.frag compiled with -DGBUFFER_OUTPUT.
    // Lives lazily — only created when the application calls
    // initBindlessGBufferPipeline(); drawOpaqueGBuffer() no-ops if it
    // hasn't been initialised so legacy forward callers are unaffected.
    std::shared_ptr<renderer::Pipeline>            bindless_gbuffer_pipeline_;

    // (Re)create the OIT accum + reveal targets at the requested size,
    // and update the composite descriptor set to point at the new views.
    // Lazy: no-op when the size hasn't changed.  Called from draw() so
    // the targets always match the host color buffer's resolution
    // (handles swap-chain rebuild without an explicit recreate hook).
    void ensureOitTargets(const glm::uvec2& size);

    // ── OIT resources (created/recreated with screen size) ──────────────
    // accum  = RGBA16F, cleared to (0,0,0,0)
    // reveal = R8/R16F, cleared to (1,0,0,0)
    // Both are framebuffer-attachable AND sampler-readable so the resolve
    // pass can read them via the composite descriptor set.
    renderer::TextureInfo                          oit_accum_tex_;
    renderer::TextureInfo                          oit_reveal_tex_;
    glm::uvec2                                     oit_target_size_{0, 0};

    std::shared_ptr<renderer::Sampler>             oit_composite_sampler_;
    std::shared_ptr<renderer::DescriptorSetLayout> oit_composite_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>       oit_composite_desc_set_;
    std::shared_ptr<renderer::PipelineLayout>      oit_composite_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            oit_composite_pipeline_;
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
    // Toggle: Hi-Z occlusion culling for the cluster path.  When on,
    // Phase B's cull dispatch samples the Hi-Z pyramid (built from
    // Phase A's just-written depth) and rejects clusters whose
    // bounding-sphere nearest point is BEHIND every visible surface in
    // its screen footprint.  When off the cull pass falls back to
    // plain frustum + cone culling.  Plumbed end-to-end (menu →
    // renderer → push constant → shader) so toggling at runtime is
    // enough; no rebuild needed.
    //
    // Default ON: the two-pass orchestration + 4-tap test has been
    // verified to produce correct results in dense scenes (Bistro)
    // and the perf win from rejecting fully-occluded clusters in
    // Phase B easily covers the cost of the Hi-Z build dispatch.
    bool use_hiz_occlusion_cull_ = true;

    // Hi-Z pyramid handles supplied via setHiZTexture().  view + sampler
    // get bound at descriptor binding 11 of the cull set; size + mip
    // count are forwarded to the shader via push constants so it can
    // pick an appropriate mip and clamp the sample coordinates.
    std::shared_ptr<renderer::Sampler>   hiz_sampler_;
    std::shared_ptr<renderer::ImageView> hiz_view_;
    glm::uvec2                           hiz_size_      = glm::uvec2(0);
    uint32_t                             hiz_mip_count_ = 0;

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

    // Initialise the deferred G-buffer variant of the bindless pipeline.
    // Reuses the existing pipeline layout + descriptor set built by
    // initBindlessPipeline (must be called first), then compiles a 3-RT
    // pipeline against the supplied G-buffer format.  drawOpaqueGBuffer()
    // returns 0 if this hasn't been called.
    void initBindlessGBufferPipeline(
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::PipelineRenderbufferFormats& gbuffer_format);

    // Dispatch cluster culling compute shader (single dispatch for all meshes).
    // last_view_proj is kept in the signature for the legacy single-pass
    // path.  In the two-pass deferred path it's unused — Phase B
    // reprojects against the CURRENT view_proj because the Hi-Z pyramid
    // was built from this frame's Phase A depth, not last frame's.
    // hiz_cull_override:
    //   nullopt → fall back to the menu toggle use_hiz_occlusion_cull_
    //   false   → force-skip the Hi-Z test (probe / cubemap face passes
    //             that have no business sampling the main camera's pyramid)
    //   true    → force-enable (currently unused; reserved for tooling)
    void cull(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::mat4& view_proj,
        const glm::vec3& camera_pos,
        const glm::mat4& last_view_proj = glm::mat4(1.0f),
        std::optional<bool> hiz_cull_override = std::nullopt);

    // ── Two-pass occlusion culling (Nanite-style) ────────────────────────
    // Phase A: gated on visibility bits, frustum + backface only, emits
    // to indirect_draw_buffer_phase_a_.  No Hi-Z test (Phase A clusters
    // are the trusted "visible last frame" set, used as the depth-prepass
    // for Phase B's Hi-Z).  Sets visibility bits for survivors so they
    // continue carrying forward.
    void cullPhaseA(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::mat4& view_proj,
        const glm::vec3& camera_pos);

    // Phase B: tests ALL clusters with frustum + backface + Hi-Z (using
    // the pyramid built from Phase A's depth).  Emits to the standard
    // opaque indirect (overwriting Phase A's count — Phase B's emit set
    // is the COMPLETE set this frame, so the standard buffer carries
    // it for any consumer that wants "all visible clusters").  Also
    // emits to the translucent bucket (translucents only run in B).
    // Atomically OR's the visibility bit for every opaque survivor.
    void cullPhaseB(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::mat4& view_proj,
        const glm::vec3& camera_pos);

    // Indirect draw consumer for Phase A.  Same merged VB/IB + bindless
    // descriptor set as drawOpaqueGBuffer; the only difference is which
    // indirect buffer we read from.  Caller must have an active
    // dynamic-rendering pass with G-buffer attachments + depth bound.
    uint32_t drawOpaqueGBufferPhaseA(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors);

    // Phase B's indirect draw is exactly drawOpaqueGBuffer (same indirect
    // buffer); aliased here for symmetry / clarity at the call site.
    uint32_t drawOpaqueGBufferPhaseB(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors) {
        return drawOpaqueGBuffer(cmd_buf, desc_sets, viewports, scissors);
    }

    // Reset the persistent visibility bit set to all zeros.  Called
    // between Phase A and Phase B each frame so that Phase B's atomicOr
    // writes produce the correct "visible this frame" set without any
    // leak from the previous frame's bits.
    void clearVisibilityBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    // Provide the application's Hi-Z pyramid for last-frame occlusion
    // culling.  Call ONCE after pyramid creation and AGAIN whenever the
    // pyramid is recreated (swap-chain resize).  Caller passes the
    // sampler used to read the pyramid (point/CLAMP_TO_EDGE filter
    // recommended), the mip-0 image view, the mip-0 size in pixels, and
    // the total mip count so the shader can clamp its mip pick.
    void setHiZTexture(
        const std::shared_ptr<renderer::Sampler>& sampler,
        const std::shared_ptr<renderer::ImageView>& view,
        const glm::uvec2& size,
        uint32_t mip_count);

    // Bindless draw — issues two indirect draws and a fullscreen OIT
    // composite, all sharing the same merged VB/IB and bindless desc set:
    //   1. Opaque + alpha-mask clusters into the caller's currently-active
    //      dynamic rendering pass (color + depth).
    //   2. End caller's pass; begin internal OIT pass; translucent
    //      clusters write WBOIT (accum + reveal) into oit_accum_tex_ +
    //      oit_reveal_tex_ with depth-test on / depth-write off.
    //   3. End OIT pass; begin internal composite pass on color_view;
    //      fullscreen quad samples accum+reveal and blends the resolved
    //      colour over the existing scene with WBOIT's resolve formula.
    //   4. Re-begin the caller's dynamic rendering pass with LOAD/LOAD so
    //      the caller's matching endDynamicRendering() still has a pass
    //      open to close (preserves the API contract).
    //
    // `color_view` / `depth_view` are the caller's render targets (taken
    // from the existing forward dynamic-rendering setup).  `screen_size`
    // sizes the OIT accum/reveal targets — they're recreated lazily when
    // it changes (e.g. swap-chain rebuild).
    //
    // Returns previous frame's visible cluster count for HUD reporting.
    uint32_t draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors,
        const std::shared_ptr<renderer::ImageView>& color_view,
        const std::shared_ptr<renderer::ImageView>& depth_view,
        const glm::uvec2& screen_size);

    // Deferred opaque-only draw — issues the same indirect draw as
    // drawOpaqueOnly() but binds bindless_gbuffer_pipeline_, which
    // writes the 3-RT G-buffer instead of lit colour.  Caller must have
    // an active dynamic-rendering pass with the application's G-buffer
    // colour attachments (3) + the scene depth attachment bound.
    // Returns 0 if initBindlessGBufferPipeline() has not run.
    uint32_t drawOpaqueGBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors);

    // Forward-only translucent / OIT path for the deferred renderer.  The
    // deferred lighting compute writes opaque pixels into hdr_color_buffer_
    // before this is called; drawTranslucentForward() then runs the WBOIT
    // accum + reveal passes and the fullscreen composite (the same three
    // sub-passes draw() already runs internally), composited on top of
    // hdr_color_buffer_.  Caller must have the host pass on color_view +
    // depth_view OPEN (LOAD/LOAD), exactly like draw(); the function
    // re-opens it before returning.  Returns previous-frame visible count.
    uint32_t drawTranslucentForward(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors,
        const std::shared_ptr<renderer::ImageView>& color_view,
        const std::shared_ptr<renderer::ImageView>& depth_view,
        const glm::uvec2& screen_size);

    // Opaque-only draw for cube face captures (DynamicCubemap).  Skips
    // OIT + composite to avoid the size-dependent OIT target
    // reallocation hazard — see the implementation comment in
    // cluster_renderer.cpp for the full rationale.  Caller must have
    // an active dynamic-rendering pass with appropriate color + depth
    // attachments already bound.
    uint32_t drawOpaqueOnly(
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
    bool& getUseHiZOcclusionCull() { return use_hiz_occlusion_cull_; }
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
