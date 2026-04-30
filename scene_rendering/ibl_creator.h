#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class IblCreator {
    renderer::TextureInfo panorama_tex_;
    renderer::TextureInfo rt_envmap_tex_;
    renderer::TextureInfo tmp_ibl_diffuse_tex_;
    renderer::TextureInfo tmp_ibl_specular_tex_;
    renderer::TextureInfo tmp_ibl_sheen_tex_;
    renderer::TextureInfo rt_ibl_diffuse_tex_;
    renderer::TextureInfo rt_ibl_specular_tex_;
    renderer::TextureInfo rt_ibl_sheen_tex_;

    std::shared_ptr<renderer::DescriptorSet> envmap_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_diffuse_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_specular_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_sheen_tex_desc_set_;

    std::shared_ptr<renderer::DescriptorSetLayout> ibl_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> ibl_comp_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_comp_pipeline_layout_;

    std::shared_ptr<renderer::Pipeline> envmap_pipeline_;
    std::shared_ptr<renderer::Pipeline> lambertian_pipeline_;
    std::shared_ptr<renderer::Pipeline> ggx_pipeline_;
    std::shared_ptr<renderer::Pipeline> charlie_pipeline_;
    std::shared_ptr<renderer::Pipeline> blur_comp_pipeline_;

    // ── Dithered "mini-buffer" IBL convolution state ─────────────────────
    // Same idea as Skydome::updateCubeSkyBoxMini: per frame, only 1/64 of
    // each output IBL cubemap mip's texels are recomputed (at full sample
    // quality), at an 8x8 dither offset that cycles through 64 unique
    // positions over 64 frames.  After 64 frames each texel has been
    // refreshed exactly once.
    uint32_t mini_frame_index_ = 0;

    // Per-mip CUBE image views for storage-image writes.  We can't reuse
    // `texture.view` (which spans all mips) because GLSL imageStore on an
    // imageCube always targets the view's base mip; we need one view per
    // mip to write each mip independently.
    //   - diffuse: only mip 0 is convolved, so size 1.
    //   - specular / sheen: full mip chain.
    std::vector<std::shared_ptr<renderer::ImageView>> rt_ibl_diffuse_per_mip_cube_views_;
    std::vector<std::shared_ptr<renderer::ImageView>> rt_ibl_specular_per_mip_cube_views_;
    std::vector<std::shared_ptr<renderer::ImageView>> rt_ibl_sheen_per_mip_cube_views_;

    // Descriptor set layout / pipeline layout for the mini compute pass.
    // Bindings in the set:
    //   ENVMAP_TEX_INDEX (0) = source envmap, COMBINED_IMAGE_SAMPLER
    //   DST_TEX_INDEX    (4) = per-mip destination cubemap, STORAGE_IMAGE
    std::shared_ptr<renderer::DescriptorSetLayout> ibl_mini_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_mini_pipeline_layout_;

    // One compute pipeline per filter type.  Compiled from cube_ibl_mini.comp
    // with LAMBERTIAN_FILTER / GGX_FILTER / CHARLIE_FILTER #defines.
    std::shared_ptr<renderer::Pipeline> lambertian_mini_pipeline_;
    std::shared_ptr<renderer::Pipeline> ggx_mini_pipeline_;
    std::shared_ptr<renderer::Pipeline> charlie_mini_pipeline_;

    // Per-mip descriptor sets (one for each pipeline).  Index = mip level.
    std::vector<std::shared_ptr<renderer::DescriptorSet>> diffuse_mini_desc_sets_;   // size 1
    std::vector<std::shared_ptr<renderer::DescriptorSet>> specular_mini_desc_sets_;  // size num_mips
    std::vector<std::shared_ptr<renderer::DescriptorSet>> sheen_mini_desc_sets_;     // size num_mips

public:
    IblCreator(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const uint32_t& cube_size);

    inline const renderer::TextureInfo& getEnvmapTexture() const
    {
        return rt_envmap_tex_;
    }

    // Read-only accessors for the IBL output cubemaps.  Used by the
    // Menu's "IBL Debug" window to show face-by-face previews via
    // ImGui::Image.  TextureInfo::surface_views[mip][face] gives
    // sampleable 2D views for each cube face per mip - ideal for ImGui.
    inline const renderer::TextureInfo& getIblDiffuseTexture()  const { return rt_ibl_diffuse_tex_; }
    inline const renderer::TextureInfo& getIblSpecularTexture() const { return rt_ibl_specular_tex_; }
    inline const renderer::TextureInfo& getIblSheenTexture()    const { return rt_ibl_sheen_tex_; }

    // Reset the mini-buffer's temporal accumulation state across all
    // three IBL filters (Lambertian / GGX / Charlie).  Forces the next
    // updateIbl{Diffuse,Specular,Sheen}MapMini calls into "first-touch"
    // mode: every output texel is re-initialized in one cheap block-fill
    // dispatch, then the dither + integration cycle restarts from
    // frame 0.
    //
    // Call this on big scene transitions where the previously
    // accumulated IBL state is invalid: new game, camera teleport,
    // dramatic environment change, or any time the envmap has just been
    // fully replaced (e.g. paired with `Skydome::resetMiniBuffer()`).
    // For small parameter changes (slow sun motion, gentle weather) the
    // EMA blending will adapt on its own - prefer to NOT reset for those.
    inline void resetMiniBuffer() { mini_frame_index_ = 0; }

    inline std::shared_ptr<renderer::DescriptorSetLayout> getIblDescSetLayout() const
    {
        return ibl_desc_set_layout_;
    }

    inline std::shared_ptr<renderer::DescriptorSet> getEnvmapTexDescSet() const
    {
        return envmap_tex_desc_set_;
    }

    void createCubeTextures(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const uint32_t& cube_size);

    void addToGlobalTextures(
        renderer::WriteDescriptorList& descriptor_writes,
        const std::shared_ptr<renderer::DescriptorSet>& description_set,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    void createIblGraphicsPipelines(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
        const uint32_t& cube_size);

    void createDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    void drawEnvmapFromPanoramaImage(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void createIblDiffuseMap(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void createIblSpecularMap(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void createIblSheenMap(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void blurIblMaps(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const uint32_t& cube_size);

    // Dithered "mini-buffer" IBL convolution updates.  Each call dispatches
    // the compute shader once per relevant mip; per dispatch only 1/64 of
    // that mip's texels are recomputed (for mips with face_size >= 8) or
    // the whole mip is recomputed (for mips < 8x8 - cost is negligible).
    //
    // Together, these three calls replace `createIblDiffuseMap` /
    // `createIblSpecularMap` / `createIblSheenMap` from the second frame
    // onward.  The first frame must still call the full createIbl*Map
    // path to bootstrap the IBL cubemap contents - subsequent dithered
    // updates assume the existing texels are valid.
    //
    // Per-frame work drops to roughly 1/64 of the full convolution cost
    // (plus the small-mip full updates), at the cost of a 64-frame
    // propagation lag for envmap changes.
    void updateIblDiffuseMapMini(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const uint32_t& cube_size);

    void updateIblSpecularMapMini(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const uint32_t& cube_size);

    void updateIblSheenMapMini(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const uint32_t& cube_size);

    // (Re)allocate the per-mip cube views and per-(filter, mip) descriptor
    // sets used by the mini-buffer IBL passes.  Called by the constructor
    // and again from `recreate()` after the descriptor pool is rebuilt.
    void bindIblMiniTargets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const uint32_t& cube_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
