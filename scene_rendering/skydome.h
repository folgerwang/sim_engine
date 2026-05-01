#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class Skydome {
    glm::vec3 sun_dir_ = glm::vec3(0);

    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;

    float g_ = 0.562f;
    float rayleigh_scale_height_ = kRayleighScaleHeight;
    float mie_scale_height_ = kMieScaleHeight;
    float lut_rayleigh_scale_height_ = 0.0f;
    float lut_mie_scale_height_ = 0.0f;

    // Monotonic frame counter for the dithered "mini-buffer" sky update.
    // Used to pick one of 64 unique (dx, dy) offsets each frame so a full
    // 8x8 block is covered every 64 frames.
    uint32_t mini_frame_index_ = 0;

    renderer::TextureInfo sky_scattering_lut_tex_;
    renderer::TextureInfo sky_scattering_lut_sum_tex_;
    // Mini-buffer cubemap at (cube_size/8, cube_size/8, 6).  Holds the
    // per-frame sparse sample of the sky envmap before scattering into
    // the full-res envmap with an 8x8 dither offset.
    renderer::TextureInfo mini_envmap_tex_;

    std::shared_ptr<renderer::DescriptorSetLayout> skybox_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> skybox_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> skybox_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> cube_skybox_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> skybox_pipeline_;
    std::shared_ptr<renderer::Pipeline> cube_skybox_pipeline_;

    // ── Envmap background pipeline ─────────────────────────────────────────
    // Renders a fullscreen triangle at the far plane (z=1.0) that samples
    // the live sky envmap cubemap using a view direction reconstructed from
    // the push-constant inv_view_proj_relative matrix.  Depth test is
    // LESS_OR_EQUAL so the sky only fills pixels where no geometry was drawn.
    std::shared_ptr<renderer::DescriptorSetLayout> skybox_envmap_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>       skybox_envmap_desc_set_;
    std::shared_ptr<renderer::PipelineLayout>      skybox_envmap_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            skybox_envmap_pipeline_;

    // Compute pipeline / descriptors for the mini-buffer sky update.
    std::shared_ptr<renderer::DescriptorSetLayout> cube_skybox_mini_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> cube_skybox_mini_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> cube_skybox_mini_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cube_skybox_mini_pipeline_;

    std::shared_ptr<renderer::DescriptorSet> sky_scattering_lut_first_pass_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> sky_scattering_lut_sum_pass_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> sky_scattering_lut_final_pass_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> sky_scattering_lut_first_pass_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> sky_scattering_lut_sum_pass_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> sky_scattering_lut_final_pass_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> sky_scattering_lut_first_pass_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> sky_scattering_lut_sum_pass_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> sky_scattering_lut_final_pass_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> sky_scattering_lut_first_pass_pipeline_;
    std::shared_ptr<renderer::Pipeline> sky_scattering_lut_sum_pass_pipeline_;
    std::shared_ptr<renderer::Pipeline> sky_scattering_lut_final_pass_pipeline_;


public:
    Skydome(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size,
        const uint32_t& cube_size);

    inline glm::vec3 getSunDir() {
        return sun_dir_;
    }

    inline float& getRayleighScaleHeight() {
        return rayleigh_scale_height_;
    }

    inline float& getMieScaleHeight() {
        return mie_scale_height_;
    }

    inline float& getG() {
        return g_;
    }

    inline const std::shared_ptr<renderer::ImageView>& getScatteringLutTex() {
        return sky_scattering_lut_tex_.view;
    }

    // Read-only accessor for the (cube_size/8) per-face mini-buffer the
    // dithered sky update writes into every frame.  Exposed for the
    // Menu's IBL/sky debug window.
    inline const renderer::TextureInfo& getMiniEnvmapTexture() const {
        return mini_envmap_tex_;
    }

    // Reset the mini-buffer's temporal accumulation state.  Forces the
    // next `updateCubeSkyBoxMini` call into "first-touch" mode: every
    // texel is re-initialized in one cheap block-fill dispatch, then the
    // dither + integration cycle starts over from frame 0.
    //
    // Call this on big scene transitions where any temporal coherence
    // with the previous envmap is invalid - new game, player teleport,
    // a debug "skip to dawn" button, sun parameters changed by more
    // than the 64-frame integration window can absorb, etc.
    inline void resetMiniBuffer() { mini_frame_index_ = 0; }

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size);

    void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set);

    // Creates the fullscreen envmap background pipeline using the dynamic-
    // rendering format descriptors.  Call once after construction (and after
    // recreate) before the first drawEnvmap() call.  Also creates/updates the
    // envmap sampler descriptor set.
    void initEnvmapBackgroundPipeline(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats);

    // Draws the sky envmap as a fullscreen background into the currently active
    // dynamic rendering pass.  Only pixels at far-plane depth (no geometry) are
    // shaded.  inv_view_proj_relative must be inverse(proj * view_no_translation).
    void drawEnvmap(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::mat4& inv_view_proj_relative);

    void drawCubeSkyBox(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    // Dithered "mini-buffer" sky envmap update.  Renders a (cube_size/8)
    // resolution mini cubemap at full per-texel atmospheric quality and
    // scatters its texels into the full-res envmap mip 0 using a per-frame
    // 8x8 dither offset.  Over 64 consecutive frames the full envmap mip 0
    // is fully refreshed.  The full mip chain is regenerated afterwards so
    // the IBL convolution sees an updated input.
    //
    // This is roughly 64x cheaper per frame than `drawCubeSkyBox` at the
    // cost of a 64-frame propagation lag for sun/atmosphere parameter
    // changes - acceptable for slowly-varying sky lighting.
    void updateCubeSkyBoxMini(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::TextureInfo& rt_envmap_tex,
        const uint32_t& cube_size);

    // Bind/rebind the destination cubemap for the mini-buffer compute pass.
    // Must be called once after construction (and on recreate) so the
    // compute descriptor set knows where to scatter its samples.
    void bindMiniSkyBoxTargets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::TextureInfo& rt_envmap_tex);

    void update(float latitude, float longtitude, int d, int h, int m, int s);

    void updateSkyScatteringLut(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    // DEBUG: reads the scattering LUT back to CPU and prints a sample grid.
    // Call ONLY after the device is idle (e.g. waitIdle after the first frame).
    // Set enabled=false once the sky is working to avoid the GPU stall.
    void dumpScatteringLut(const std::shared_ptr<renderer::Device>& device);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
