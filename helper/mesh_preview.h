#pragma once
//
// mesh_preview.h  --  GPU offscreen PBR preview pass for the editor's Debug
//                     Display window.
//
// Renders a triangle mesh (positions + normals + UVs + base-colour texture,
// CPU-extracted by the UI from a loaded DrawableData or a content asset)
// into a persistent 512x512 RGBA8 target using the real graphics pipeline:
// Cook-Torrance GGX lit by a three-spot studio rig (preview_mesh.vert/.frag).
// The target is sampled by ImGui every frame; the pass itself is recorded
// only when the previewed object changes.
//
// Threading/lifetime notes:
//   * render() must be called OUTSIDE any render pass, on the frame's main
//     command buffer, BEFORE the ImGui pass that samples the result.
//   * Buffers/textures from the previous preview may still be referenced by
//     in-flight frames; they go onto a deferred-free list keyed by the
//     app's MONOTONIC frame counter (collectGarbage()).
//   * Descriptor sets rotate through a small ring so updating the texture
//     binding never touches a set a submitted frame still reads.
//   * The ImGui texture ID is registered lazily and must be re-registered
//     after the ImGui descriptor pool is rebuilt (reregisterImGui()).
//
#include "renderer/renderer.h"

namespace engine {
namespace helper {

// Everything one preview render needs.  Geometry plus MATERIAL SECTIONS —
// FBX assigns materials per face and glTF per primitive, so one object
// commonly spans several base-colour textures.  sections may be empty
// (a single default-material section spanning all indices is synthesised).
struct MeshPreviewTexture {
    std::vector<uint8_t> rgba;   // RGBA8
    int                  w = 0;
    int                  h = 0;
};
struct MeshPreviewSection {
    uint32_t  first_index = 0;
    uint32_t  index_count = 0;
    glm::vec4 base_color  = glm::vec4(1.0f);
    float     metallic    = 0.0f;
    float     roughness   = 0.6f;
    int       tex_index   = -1;  // albedo, into MeshPreviewPayload::textures
    int       nrm_index   = -1;  // normal map (-1 = none)
    int       mr_index    = -1;  // metallic-roughness map (G=rough, B=metal)
};
struct MeshPreviewPayload {
    std::vector<glm::vec3>          positions;
    std::vector<glm::vec3>          normals;
    std::vector<glm::vec2>          uvs;
    std::vector<uint32_t>           indices;
    std::vector<MeshPreviewTexture> textures;
    std::vector<MeshPreviewSection> sections;
};

class MeshPreview {
public:
    static constexpr uint32_t kSize = 512;   // offscreen target extent

    // One-time pipeline + render-target construction.  graphic_pipeline_info
    // should be the app's default (depth test + write ON, used by the mesh
    // pipeline); grid_pipeline_info the no-depth-write variant (the
    // alpha-blended reference grid must not occlude through its transparent
    // gaps).  descriptor_pool must be a PERSISTENT pool (the app's
    // drawable_descriptor_pool_); `sampler` is used both for the material
    // texture binding and the ImGui registration of the colour target.
    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::GraphicPipelineInfo& grid_pipeline_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& sampler);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static bool ready() { return s_pipeline_ != nullptr; }

    // Upload the payload and record the offscreen pass into cmd_buf.  The
    // camera auto-frames the mesh bounds (fixed orbit angle).  current_frame
    // keys the deferred free of the PREVIOUS preview's resources and must be
    // MONOTONIC across frames.
    static void render(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const MeshPreviewPayload& payload,
        uint64_t current_frame);

    // Free retired resources whose last-possible-use frame has passed.
    static void collectGarbage(
        const std::shared_ptr<renderer::Device>& device,
        uint64_t current_frame);

    // ── Orbit camera (mouse-driven, persists across previews) ──────────
    // The UI feeds drag deltas / wheel ticks; the pass re-records on the
    // next rerenderIfNeeded() call.
    static void orbit(float d_azimuth_deg, float d_elevation_deg);
    static void zoom(float wheel_ticks);
    // Right-drag: move the object in the view (pans the orbit target in the
    // camera plane; pixel deltas are scaled to the framing distance).
    static void pan(float dx_px, float dy_px);
    // Re-record the offscreen pass with the current camera when it changed
    // since the last render.  Same call-site constraints as render().
    static void rerenderIfNeeded(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    // ImGui handle of the colour target (0 until first render).
    static ImTextureID imguiId() { return s_im_id_; }
    // Re-register with a fresh ImGui descriptor pool (swapchain rebuild).
    static void reregisterImGui();
    // True once a preview has been rendered at least once.
    static bool hasImage() { return s_has_image_; }

    // Current offscreen target resolution (defaults to kSize x kSize).
    static uint32_t width()  { return s_width_; }
    static uint32_t height() { return s_height_; }
    // World→clip matrix of the LAST recorded pass (Vulkan clip space, Y
    // already flipped).  Lets the editor overlay 3D gizmos (e.g. the skeleton
    // debug draw) onto the preview image in screen space.
    static const glm::mat4& lastViewProj() { return s_view_proj_; }
    // Weight-debug render: when on, the mesh is drawn flat, coloured by the
    // per-vertex weight the caller packs into the payload's uv.x (no texturing
    // / lighting).  Toggled by the editor's bone/weight inspector.
    static void setWeightDebug(bool on) { s_weight_debug_ = on; }
    // Segmentation render: mesh drawn flat, coloured by the dominant bone index
    // the caller packs into uv.x (distinct hue per segment).
    static void setSegmentDebug(bool on) { s_segment_debug_ = on; }
    // Request a new target resolution (cheap; applied next render/rerender on
    // the main thread). Driven by the Debug Display panel size so the preview
    // re-renders at native resolution instead of upscaling a fixed 512^2.
    static void requestSize(uint32_t w, uint32_t h) { s_req_w_ = w; s_req_h_ = h; }

private:
    // Records the full offscreen pass (mesh + grid) with the current orbit
    // camera into cmd_buf.  Requires uploaded buffers.
    static void recordPass(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    static std::vector<renderer::VertexInputBindingDescription>   s_binding_descs_;
    static std::vector<renderer::VertexInputAttributeDescription> s_attrib_descs_;
    static std::shared_ptr<renderer::DescriptorSetLayout>         s_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout>              s_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline>                    s_pipeline_;
    static std::shared_ptr<renderer::Pipeline>                    s_grid_pipeline_;

    static std::shared_ptr<renderer::Sampler>        s_sampler_;
    static std::shared_ptr<renderer::DescriptorPool> s_pool_;
    static std::shared_ptr<renderer::TextureInfo>  s_color_;
    static std::shared_ptr<renderer::TextureInfo>  s_depth_;
    static std::shared_ptr<renderer::TextureInfo>  s_white_tex_;   // 1x1 fallback
    static std::shared_ptr<renderer::TextureInfo>  s_flatnrm_tex_; // 1x1 (128,128,255)
    static ImTextureID                             s_im_id_;
    static bool                                    s_has_image_;
    static bool                                    s_color_in_read_layout_;
    static uint32_t                                s_width_;
    static uint32_t                                s_height_;
    static uint32_t                                s_req_w_;
    static uint32_t                                s_req_h_;
    static uint64_t                                s_frame_;   // last monotonic frame
    // Recreate the offscreen target at the requested size (deferred-free of the
    // old one). Safe: called only from render()/rerenderIfNeeded (main thread,
    // outside any render pass).
    static void applyPendingResize(const std::shared_ptr<renderer::Device>& device);

    // Per-texture descriptor sets for the CURRENT preview, plus a recycler
    // so retired sets are reused once no in-flight frame references them
    // (sets come from the persistent pool — never freed, only recycled).
    static std::vector<std::shared_ptr<renderer::DescriptorSet>> s_active_sets_;
    struct RecycleSet {
        uint64_t                                 free_frame;
        std::shared_ptr<renderer::DescriptorSet> set;
    };
    static std::vector<RecycleSet>                 s_set_recycle_;

    // Material sections of the current preview (recordPass replays these).
    struct GpuSection {
        uint32_t  first_index = 0;
        uint32_t  index_count = 0;
        glm::vec4 base_color  = glm::vec4(1.0f);
        float     metallic    = 0.0f;
        float     roughness   = 0.6f;
        bool      has_tex     = false;
        bool      has_nrm     = false;   // normal map bound
        bool      has_mr      = false;   // metallic-roughness map bound
        size_t    set_index   = 0;   // into s_active_sets_
    };
    static std::vector<GpuSection>                 s_sections_;

    // Current mesh buffers / material textures + the deferred-free lists.
    static std::shared_ptr<renderer::BufferInfo>   s_pos_buf_;
    static std::shared_ptr<renderer::BufferInfo>   s_nrm_buf_;
    static std::shared_ptr<renderer::BufferInfo>   s_uv_buf_;
    static std::shared_ptr<renderer::BufferInfo>   s_idx_buf_;
    static std::shared_ptr<renderer::BufferInfo>   s_grid_buf_;   // ground quad
    static std::vector<std::shared_ptr<renderer::TextureInfo>> s_mat_texs_;
    static uint32_t                                s_index_count_;

    // Current payload framing (recordPass reads these; per-section material
    // params live in s_sections_).
    static glm::vec3 s_center_;
    static float     s_radius_;
    static bool      s_has_uv_;   // the uv stream carries real coordinates

    // Orbit camera state (degrees / bounding-radius multiplier) + the
    // right-drag pan offset added to the orbit target (reset per payload).
    static float     s_orbit_az_;
    static float     s_orbit_el_;
    static float     s_orbit_dist_;
    static glm::vec3 s_pan_;
    static bool      s_camera_dirty_;
    static glm::mat4 s_view_proj_;   // world→clip of the last recorded pass
    static bool      s_weight_debug_;  // flat per-vertex-weight render mode
    static bool      s_segment_debug_; // flat per-segment (bone index) render
    struct DeadBuffer {
        uint64_t                              free_frame;
        std::shared_ptr<renderer::BufferInfo> buf;
    };
    struct DeadTexture {
        uint64_t                               free_frame;
        std::shared_ptr<renderer::TextureInfo> tex;
    };
    static std::vector<DeadBuffer>                 s_dead_;
    static std::vector<DeadTexture>                s_dead_tex_;
};

} // namespace helper
} // namespace engine
