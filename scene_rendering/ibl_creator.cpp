#include <iostream>
#include <vector>

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "ibl_creator.h"

namespace {
namespace er = engine::renderer;

// VRAM cleanup (2026-08): the helpers that served the legacy full-res
// graphics IBL path (getIblShaderModules, createCubemapPipelineLayout,
// addIblTextures, addIblComputeTextures, createCubemapComputePipelineLayout)
// were removed together with the pipelines/textures they set up.  The
// public methods that used them are kept as logged no-op stubs — see the
// "STUBBED" banner in ibl_creator.h.

// Writes the PANORAMA_TEX_INDEX binding of envmap_tex_desc_set_.  The
// panorama HDR itself was removed as dead VRAM (drawEnvmapFromPanoramaImage
// is now a stub); the set is kept — it has a public accessor
// (getEnvmapTexDescSet) that unstaged UI/tooling code may still call —
// and is filled with a valid placeholder view so binding it is never
// undefined behavior.
er::WriteDescriptorList addPanoramaTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& panorama_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PANORAMA_TEX_INDEX,
        texture_sampler,
        panorama_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

// Warning for calls into the stubbed legacy IBL methods.  std::cerr is
// mirrored into the editor Output Log (see the logging note in
// cluster_renderer.cpp).  Each stub wraps this in its own
// function-local once-guard so a per-frame caller can't flood the log,
// while distinct stubbed methods each still get one line.
void warnStubbedIblCall(const char* method_name) {
    std::cerr << "[IblCreator] " << method_name
              << "() is a no-op stub - the legacy full-res IBL path "
                 "was removed (VRAM cleanup); use the *MapMini "
                 "compute path instead."
              << std::endl;
}

// Expands to a per-method once-guarded warnStubbedIblCall.
#define IBL_WARN_STUBBED_ONCE(method_name)          \
    do {                                            \
        static bool s_warned_once = false;          \
        if (!s_warned_once) {                       \
            s_warned_once = true;                   \
            warnStubbedIblCall(method_name);        \
        }                                           \
    } while (0)

// Pipeline layout for the mini-buffer IBL convolution compute shaders.
// Push constant carries IblMiniParams (roughness, mip metadata, dither
// stride/offset).  Single descriptor set has the source envmap (sampler)
// and the destination per-mip cubemap (storage image).
std::shared_ptr<er::PipelineLayout> createIblMiniPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& ibl_mini_desc_set_layout)
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblMiniParams);

    return device->createPipelineLayout(
        { ibl_mini_desc_set_layout },
        { push_const_range },
        std::source_location::current());
}

// Bit-reversal Van der Corput permutation of [0, 1024).  Sweeps the 32x32
// dither block in low-discrepancy order: 10-bit reversed index, lower 5 bits
// → x, upper 5 bits → y.  Skydome::ditherOffsetForFrame uses the smaller
// 8x8 block; the two are intentionally NOT in sync any more (the IBL
// convolution gets the slower 1024-frame fill window while the sky envmap
// stays on its tighter 64-frame cycle so sun motion shows up promptly in
// the visible sky).
glm::ivec2 iblDitherOffsetForFrame(uint32_t frame_index) {
    uint32_t i = frame_index & 1023u;
    uint32_t r = 0u;
    for (uint32_t b = 0u; b < 10u; ++b) {
        r |= ((i >> b) & 1u) << (9u - b);
    }
    return glm::ivec2(static_cast<int>(r & 31u),
                      static_cast<int>((r >> 5u) & 31u));
}

// Write descriptor entries for one mini-buffer IBL pass:
//   - source envmap (combined image sampler)
//   - destination per-mip cubemap (storage image)
er::WriteDescriptorList addIblMiniTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& envmap_view,
    const std::shared_ptr<er::ImageView>& dst_per_mip_cube_view) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ENVMAP_TEX_INDEX,
        texture_sampler,
        envmap_view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_per_mip_cube_view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

}

namespace engine {
namespace scene_rendering {

IblCreator::IblCreator(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const uint32_t& cube_size) {

    createCubeTextures(
        device,
        cube_render_pass,
        cube_size);

    // NOTE (VRAM cleanup, 2026-08): the panorama HDR load
    // ("assets/environments/doge2.hdr"), the ibl compute (blur)
    // descriptor set layout, the blur compute pipeline and the four
    // graphics pipelines (envmap / lambertian / ggx / charlie) that
    // used to be created here were removed — nothing in src/ calls the
    // methods that consumed them (drawEnvmapFromPanoramaImage,
    // createIbl*Map, blurIblMaps; all stubbed).  The runtime IBL path
    // is exclusively the *MapMini compute pipelines set up below.
    // The parameter stays on the signature so call sites (application.cpp)
    // don't churn.
    (void)cube_graphic_pipeline_info;

    // ibl texture descriptor set layout.  Kept: getIblDescSetLayout()
    // and getEnvmapTexDescSet() are public and may be referenced by
    // UI/tooling code outside this tree.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(PANORAMA_TEX_INDEX);
        //bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(ENVMAP_TEX_INDEX);

        ibl_desc_set_layout_ =
            device->createDescriptorSetLayout(bindings);
    }

    createDescriptorSets(
        device,
        descriptor_pool,
        texture_sampler);

    // ── Mini-buffer IBL convolution: descriptor set layout, pipeline
    // layout, and the three per-filter compute pipelines.  Per-mip
    // descriptor sets and per-mip cube views are bound in
    // bindIblMiniTargets, called next.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            ENVMAP_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);
        ibl_mini_desc_set_layout_ =
            device->createDescriptorSetLayout(bindings);
    }

    ibl_mini_pipeline_layout_ =
        createIblMiniPipelineLayout(device, ibl_mini_desc_set_layout_);

    lambertian_mini_pipeline_ =
        er::helper::createComputePipeline(
            device,
            ibl_mini_pipeline_layout_,
            "ibl_lambertian_mini_comp.spv",
            std::source_location::current());

    ggx_mini_pipeline_ =
        er::helper::createComputePipeline(
            device,
            ibl_mini_pipeline_layout_,
            "ibl_ggx_mini_comp.spv",
            std::source_location::current());

    charlie_mini_pipeline_ =
        er::helper::createComputePipeline(
            device,
            ibl_mini_pipeline_layout_,
            "ibl_charlie_mini_comp.spv",
            std::source_location::current());

    bindIblMiniTargets(device, descriptor_pool, texture_sampler, cube_size);
}

void IblCreator::bindIblMiniTargets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const uint32_t& cube_size) {

    const uint32_t num_mips =
        static_cast<uint32_t>(std::log2(cube_size) + 1);

    // Helper: build per-mip CUBE image views for one IBL output texture.
    // Each view targets a single mip across all 6 faces, so an `imageCube`
    // bound through it imageStore()s into that mip alone.
    auto rebuild_per_mip_cube_views = [&](
        std::vector<std::shared_ptr<er::ImageView>>& dst_views,
        const er::TextureInfo& tex,
        uint32_t mip_count) {
        dst_views.clear();
        dst_views.reserve(mip_count);
        for (uint32_t i = 0; i < mip_count; ++i) {
            dst_views.push_back(
                device->createImageView(
                    tex.image,
                    er::ImageViewType::VIEW_CUBE,
                    er::Format::R16G16B16A16_SFLOAT,
                    SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                    std::source_location::current(),
                    i, 1, 0, 6));
        }
    };

    rebuild_per_mip_cube_views(
        rt_ibl_diffuse_per_mip_cube_views_, rt_ibl_diffuse_tex_, 1);
    rebuild_per_mip_cube_views(
        rt_ibl_specular_per_mip_cube_views_, rt_ibl_specular_tex_, num_mips);
    rebuild_per_mip_cube_views(
        rt_ibl_sheen_per_mip_cube_views_, rt_ibl_sheen_tex_, num_mips);

    // Allocate one descriptor set per (filter, mip) and write the source
    // envmap + per-mip destination view into each.
    auto build_per_mip_desc_sets = [&](
        std::vector<std::shared_ptr<er::DescriptorSet>>& dst_sets,
        const std::vector<std::shared_ptr<er::ImageView>>& dst_views) {
        // persistent pool: allocate once, reuse across resize. The per-mip
        // cube views were rebuilt above (new ImageView objects), so each
        // reused set's binding-write must still be re-issued below.
        while (dst_sets.size() < dst_views.size())
            dst_sets.push_back(
                device->createDescriptorSets(
                    descriptor_pool, ibl_mini_desc_set_layout_, 1)[0]);
        for (size_t i = 0; i < dst_views.size(); ++i) {
            auto writes = addIblMiniTextures(
                dst_sets[i],
                texture_sampler,
                rt_envmap_tex_.view,
                dst_views[i]);
            device->updateDescriptorSets(writes);
        }
    };

    build_per_mip_desc_sets(diffuse_mini_desc_sets_,
                            rt_ibl_diffuse_per_mip_cube_views_);
    build_per_mip_desc_sets(specular_mini_desc_sets_,
                            rt_ibl_specular_per_mip_cube_views_);
    build_per_mip_desc_sets(sheen_mini_desc_sets_,
                            rt_ibl_sheen_per_mip_cube_views_);
}

void IblCreator::createCubeTextures(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const uint32_t& cube_size) {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    std::vector<renderer::BufferImageCopyInfo> dump_copies;

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_envmap_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_diffuse_tex_,
        std::source_location::current());

    // NOTE (VRAM cleanup, 2026-08): tmp_ibl_specular_tex_ and
    // tmp_ibl_sheen_tex_ (16 MiB each) were removed.  They only ever
    // served as blur sources for blurIblMaps(), which has no caller —
    // consumers bind rt_ibl_specular_tex_ / rt_ibl_sheen_tex_ directly
    // (see addToGlobalTextures).

    // Diffuse EMA accumulator at 2× linear / 4× area of the consumer-
    // facing diffuse cube.  The mini-buffer Lambertian convolution writes
    // mip 0 each frame at the larger size; we then run box-filter mipgen
    // to populate mip 1 — a proper 2×2-averaged 1× cube — and copy that
    // to tmp_ibl_diffuse_tex_ for consumers.
    //
    // Why 2× and not the previous 4× (VRAM cleanup, 2026-08):
    //   4× / 16×-area cost ~252 MiB of RGBA16F for what is a band-limited
    //   (~SH order 2) signal.  Nothing in the dither dispatch *requires*
    //   super-sampling — the 32×32 sparse-update path only needs
    //   face_size >= 32 — but the box downsample is the main spatial
    //   variance suppressor for the 16-sample Monte-Carlo estimates
    //   (this codebase has a history of visible diffuse speckle, see
    //   kIblTemporalAlpha).  2× keeps a true 2×2 box average (4
    //   independently-dithered EMA texels per consumer texel, ~4×
    //   variance reduction) at ~60 MiB — a 192 MiB saving.  If speckle
    //   ever shows up at 2×, prefer lowering kIblTemporalAlpha before
    //   re-growing this allocation.
    //
    // Why a mip chain and not a direct LINEAR blit to tmp_:
    //   Vulkan's LINEAR blit filter uses a 2×2 footprint regardless of
    //   the downsample ratio; a single 2:1 blit (or mipgen step) covers
    //   its full 2×2 source neighbourhood, so every accumulator texel
    //   contributes to the consumer cube — no dither-phase bias.
    //
    // Format must stay R16G16B16A16_SFLOAT: the mini compute shader uses
    // alpha == 0 as its "untouched texel" sentinel to pick
    // effective_alpha = 1.0 on a texel's literal first touch (see the
    // IblMiniParams::is_first_touch comment in global_definition.glsl.h),
    // so an alpha-less format (e.g. B10G11R11_UFLOAT) would break EMA
    // seeding — and the SPIR-V's declared storage-image format couldn't
    // be changed from here anyway.
    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size * 2,
        cube_size * 2,
        2,                      // mips 0 (2×), 1 (1×)
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_diffuse_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_specular_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_sheen_tex_,
        std::source_location::current());
}

// STUBBED (VRAM cleanup, 2026-08): the envmap / lambertian / ggx /
// charlie graphics pipelines only served drawEnvmapFromPanoramaImage and
// createIbl*Map, none of which have callers in src/.  Kept as a logged
// no-op in case out-of-tree UI/tooling code still calls it.
void IblCreator::createIblGraphicsPipelines(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const uint32_t& cube_size) {
    (void)device;
    (void)cube_render_pass;
    (void)cube_graphic_pipeline_info;
    (void)cube_size;
    IBL_WARN_STUBBED_ONCE("createIblGraphicsPipelines");
}

void IblCreator::createDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    // envmap
    {
        // only one descriptor layout.
        // persistent pool: allocate once, reuse across resize
        if (!envmap_tex_desc_set_)
            envmap_tex_desc_set_ =
                device->createDescriptorSets(
                    descriptor_pool, ibl_desc_set_layout_, 1)[0];

        // VRAM cleanup (2026-08): panorama_tex_ was removed (its only
        // reader, drawEnvmapFromPanoramaImage, is a stub).  Write the
        // runtime envmap cube as a placeholder so the set stays fully
        // written — getEnvmapTexDescSet() is public and out-of-tree code
        // binding an unwritten COMBINED_IMAGE_SAMPLER would be undefined
        // behavior.
        auto ibl_texture_descs = addPanoramaTextures(
            envmap_tex_desc_set_,
            texture_sampler,
            rt_envmap_tex_);
        device->updateDescriptorSets(ibl_texture_descs);
    }

    // VRAM cleanup (2026-08): the ibl_tex_desc_set_ (full-res graphics
    // convolution input) and the diffuse/specular/sheen blur-compute
    // descriptor sets were removed together with the stubbed
    // createIbl*Map / blurIblMaps paths and the tmp specular/sheen blur
    // cubes they bound.  The mini-buffer path's per-(filter, mip) sets
    // are built in bindIblMiniTargets().
}

void IblCreator::addToGlobalTextures(
    renderer::WriteDescriptorList& descriptor_writes,
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    // LAMBERTIAN_ENV_TEX_INDEX → tmp_ibl_diffuse_tex_ (the 1× cube that
    // updateIblDiffuseMapMini fills each frame via box-filter mipgen +
    // copy from the 2× EMA accumulator in rt_ibl_diffuse_tex_).  tmp_ is
    // a pure side-channel: nothing ever writes it back into rt_, so
    // successive frames' Monte-Carlo averages don't compound the spatial
    // downsample into themselves.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        LAMBERTIAN_ENV_TEX_INDEX,
        texture_sampler,
        tmp_ibl_diffuse_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        GGX_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_specular_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        CHARLIE_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_sheen_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
}

// STUBBED (VRAM cleanup, 2026-08): panorama_tex_ and envmap_pipeline_
// were removed — the runtime envmap is generated by
// Skydome::updateCubeSkyBoxMini instead.  Logged no-op for out-of-tree
// callers.
void IblCreator::drawEnvmapFromPanoramaImage(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size) {
    (void)cmd_buf;
    (void)cube_render_pass;
    (void)clear_values;
    (void)cube_size;
    IBL_WARN_STUBBED_ONCE("drawEnvmapFromPanoramaImage");
}

// STUBBED (VRAM cleanup, 2026-08): the full-res graphics convolution
// path was removed — updateIblDiffuseMapMini self-bootstraps via its
// first-touch block fill, so this bootstrap is no longer needed.
// Logged no-op for out-of-tree callers.
void IblCreator::createIblDiffuseMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    (void)cmd_buf;
    (void)cube_render_pass;
    (void)clear_values;
    (void)cube_size;
    IBL_WARN_STUBBED_ONCE("createIblDiffuseMap");
}

// STUBBED (VRAM cleanup, 2026-08): see createIblDiffuseMap — replaced
// by updateIblSpecularMapMini's first-touch block fill.
void IblCreator::createIblSpecularMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    (void)cmd_buf;
    (void)cube_render_pass;
    (void)clear_values;
    (void)cube_size;
    IBL_WARN_STUBBED_ONCE("createIblSpecularMap");
}

// STUBBED (VRAM cleanup, 2026-08): see createIblDiffuseMap — replaced
// by updateIblSheenMapMini's first-touch block fill.
void IblCreator::createIblSheenMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size) {
    (void)cmd_buf;
    (void)cube_render_pass;
    (void)clear_values;
    (void)cube_size;
    IBL_WARN_STUBBED_ONCE("createIblSheenMap");
}

namespace {

// EMA blend weight applied to each touched texel's new partial estimate.
// Lower = smoother (more samples averaged) but slower to react to sun /
// atmosphere parameter changes.
//
//   alpha = 1/8   ->  ~16 effective touches averaged  (~240 effective samples)
//                     convergence lag ~ 512 frames = 8.5 s @ 60 fps
//                     IBL visibly lags the sky; appears frozen at slow TOD speeds.
//
//   alpha = 1/2   ->  ~3 effective touches averaged   (~48 effective samples)
//                     90% convergence in ~192 frames = 3.2 s @ 60 fps
//
//   alpha = 1/4   ->  ~5 effective touches averaged  (~112 effective samples)
//                     90% convergence in ~320 frames = 5.3 s @ 60 fps
//                     At tod_advance_speed=100 the sun moves ~2.2° during that
//                     window — imperceptible at midday, very small near horizon.
//                     Enough temporal smoothing to suppress 16-sample variance.
//
//   alpha = 1.0   ->  no EMA; immediate replacement each touch
//                     IBL converges in one 64-frame dither cycle (≈1 s @ 60 fps)
//                     Causes visible per-pixel flicker on the diffuse buffer:
//                     the Lambertian kernel draws 16 random hemisphere samples
//                     whose variance is directly visible with no smoothing.
//
// alpha = 1/8 caused "IBL never changes" at tod_advance_speed=100: the sun only
// moves ~3.5° during the 512-frame lag, which is invisible at midday and barely
// perceptible near the horizon.
// alpha = 1.0 caused diffuse flicker: 16 samples/touch have high variance with
// no temporal smoothing.  1/4 is the middle ground: fast enough to track the
// sky, smooth enough to suppress noise.
constexpr float kIblTemporalAlpha = 1.0f / 4.0f;

// Dispatch one compute pass for a single mip of one IBL filter.  Handles
// the dither-stride decision (8 for big mips, 1 for small ones), the
// per-mip layout transitions, and pushing the per-frame stratification
// parameters (frame_index for sample-index advance, temporal_alpha for
// the EMA blend).  The image is left in SHADER_READ_ONLY_OPTIMAL on
// completion so the IBL fragment shaders can sample it next frame.
void dispatchIblMiniMip(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const std::shared_ptr<er::DescriptorSet>& desc_set,
    const std::shared_ptr<er::Image>& dst_image,
    uint32_t cube_size,
    uint32_t mip_level,
    uint32_t num_mips,
    const glm::ivec2& dither_offset,
    uint32_t frame_index) {

    const uint32_t mip_face_size =
        std::max(cube_size >> mip_level, 1u);

    // Sparse 32x32 dither only when the mip is at least one block tall.
    // Smaller mips fall back to a full update each frame - they have at
    // most 16x16 + 8x8 + 4x4 + 2x2 + 1 = 341 texels per face, which is
    // rounding error compared to the convolution work elsewhere.
    const bool sparse = mip_face_size >= 32u;
    const uint32_t dither_stride = sparse ? 32u : 1u;
    const uint32_t mini_size =
        sparse ? (mip_face_size / 32u) : mip_face_size;

    // Roughness is identical to the existing fragment-shader path: 0 at
    // the top mip, 1 at the smallest.  See createIblSpecularMap.
    const float roughness =
        (num_mips > 1)
            ? (static_cast<float>(mip_level) /
               static_cast<float>(num_mips - 1))
            : 0.0f;

    // First-touch detection: on the very first call (frame_index == 0)
    // the IBL image is still in UNDEFINED and holds garbage.  Tell the
    // shader to use its block-fill path (broadcast one estimate to every
    // texel in the dither block) instead of the EMA-blend path that
    // would otherwise blend against uninitialized memory.  This replaces
    // the heavyweight `createIbl*Map` graphics-pipeline bootstrap.
    const bool is_first_touch = (frame_index == 0u);

    glsl::IblMiniParams params = {};
    params.roughness        = roughness;
    params.currentMipLevel  = static_cast<int>(mip_level);
    params.width            = static_cast<int>(cube_size);
    params.lodBias          = 0.0f;
    params.mip_face_size    = static_cast<int>(mip_face_size);
    params.mini_size        = static_cast<int>(mini_size);
    params.dither_stride    = static_cast<int>(dither_stride);
    params.is_first_touch   = is_first_touch ? 1 : 0;
    params.dither_offset    = sparse ? dither_offset : glm::ivec2(0, 0);
    params.frame_index      = static_cast<int>(frame_index);
    params.temporal_alpha   = kIblTemporalAlpha;

    // Transition the destination mip to GENERAL so we can imageStore() into
    // it.  On first touch the image is in UNDEFINED (just allocated) and
    // we'll overwrite every texel via the block-fill path, so we use
    // getImageAsSource() (which encodes UNDEFINED) as the src layout.
    // After that, the previous frame's contents must be preserved (the
    // shader reads each touched texel for EMA blending), so we come from
    // SHADER_READ_ONLY_OPTIMAL.
    //
    // IMPORTANT: the non-first-touch path does imageLoad + imageStore (EMA).
    // Using getImageAsStore() (dstAccess = SHADER_WRITE only) does NOT make
    // prior writes visible to imageLoad reads — use getImageAsLoadStore()
    // (dstAccess = SHADER_READ | SHADER_WRITE) so the GPU flushes its shader
    // read cache before the compute dispatch.  Without this, imageLoad returns
    // stale/undefined values on frames where the cache was evicted, causing
    // sudden IBL intensity jumps.
    cmd_buf->addImageBarrier(
        dst_image,
        is_first_touch
            ? er::Helper::getImageAsSource()
            : er::Helper::getImageAsShaderSampler(),
        is_first_touch
            ? er::Helper::getImageAsStore()      // first touch: write-only block fill
            : er::Helper::getImageAsLoadStore(), // EMA path: imageLoad + imageStore
        mip_level, 1, 0, 6);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        pipeline_layout,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        { desc_set });

    const uint32_t group_xy = (mini_size + 7u) / 8u;
    cmd_buf->dispatch(group_xy, group_xy, 6u);

    cmd_buf->addImageBarrier(
        dst_image,
        er::Helper::getImageAsStore(),
        er::Helper::getImageAsShaderSampler(),
        mip_level, 1, 0, 6);
}
} // namespace

void IblCreator::updateIblDiffuseMapMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        lambertian_mini_pipeline_);

    const glm::ivec2 dither = iblDitherOffsetForFrame(mini_frame_index_);

    // The diffuse accumulator (rt_ibl_diffuse_tex_) is allocated at 2×
    // the consumer-facing size (4× area) with a 2-mip chain — see the
    // sizing rationale in createCubeTextures (shrunk from 4× to save
    // ~192 MiB of VRAM).  Each frame:
    //   1. Convolve mip 0 at the larger size (spatial super-sampling of
    //      the Monte-Carlo estimates).
    //   2. Box-filter mipgen mip 1 — one 2:1 LINEAR blit that covers its
    //      full 2×2 source neighbourhood, i.e. a true 2×2 box average of
    //      4 independently-dithered EMA texels per output texel.
    //   3. Copy mip 1 (now exactly cube_size sized) → tmp_ for consumers.
    const uint32_t accum_size = cube_size * 2;

    dispatchIblMiniMip(
        cmd_buf,
        ibl_mini_pipeline_layout_,
        diffuse_mini_desc_sets_[0],
        rt_ibl_diffuse_tex_.image,
        accum_size,
        /*mip_level*/ 0u,
        /*num_mips */ 1u,
        dither,
        mini_frame_index_);
    // dispatchIblMiniMip leaves rt_ mip 0 in SHADER_READ_ONLY_OPTIMAL.
    // Mip 1 is still in UNDEFINED on first frame (just allocated) and
    // SHADER_READ_ONLY_OPTIMAL on subsequent frames (left there by the
    // previous frame's mipgen tail).  generateMipmapLevels handles both
    // cases — it uses cur_image_layout for the source mip and
    // transitions destination mips internally.

    // Step 2: box-filter mipgen across mips 0 → 1.
    renderer::Helper::generateMipmapLevels(
        cmd_buf,
        rt_ibl_diffuse_tex_.image,
        /*mip_count*/ 2,
        accum_size, accum_size,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    // After mipgen, every mip of rt_ is in SHADER_READ_ONLY_OPTIMAL.

    // Step 3: copy mip 1 (1× size) → tmp_ mip 0 for consumers.
    er::ImageResourceInfo as_xfer_src = {
        er::ImageLayout::TRANSFER_SRC_OPTIMAL,
        SET_FLAG_BIT(Access, TRANSFER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    er::ImageResourceInfo as_xfer_dst = {
        er::ImageLayout::TRANSFER_DST_OPTIMAL,
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };

    // Only mip 1 of rt_ needs to flip to TRANSFER_SRC (mip 0 stays
    // in SHADER_READ_ONLY).
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsShaderSampler(),
        as_xfer_src,
        /*baseMip*/ 1, /*mipCount*/ 1, 0, 6);
    cmd_buf->addImageBarrier(
        tmp_ibl_diffuse_tex_.image,
        er::Helper::getImageAsSource(),
        as_xfer_dst,
        0, 1, 0, 6);

    er::ImageCopyInfo copy_region{};
    copy_region.src_subresource.aspect_mask      = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
    copy_region.src_subresource.mip_level        = 1;
    copy_region.src_subresource.base_array_layer = 0;
    copy_region.src_subresource.layer_count      = 6;
    copy_region.src_offset = glm::ivec3(0, 0, 0);
    copy_region.dst_subresource = copy_region.src_subresource;
    copy_region.dst_subresource.mip_level = 0;
    copy_region.dst_offset = glm::ivec3(0, 0, 0);
    copy_region.extent     = glm::uvec3(cube_size, cube_size, 1);

    cmd_buf->copyImage(
        rt_ibl_diffuse_tex_.image,  er::ImageLayout::TRANSFER_SRC_OPTIMAL,
        tmp_ibl_diffuse_tex_.image, er::ImageLayout::TRANSFER_DST_OPTIMAL,
        { copy_region });

    // tmp_ → SHADER_READ_ONLY for consumers (cluster bindless ambient,
    // base.frag IBL ambient, glass OIT).  rt_ mip 1 → SHADER_READ_ONLY
    // for next-frame state symmetry.
    cmd_buf->addImageBarrier(
        tmp_ibl_diffuse_tex_.image,
        as_xfer_dst,
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        as_xfer_src,
        er::Helper::getImageAsShaderSampler(),
        /*baseMip*/ 1, /*mipCount*/ 1, 0, 6);
}

void IblCreator::updateIblSpecularMapMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        ggx_mini_pipeline_);

    const uint32_t num_mips =
        static_cast<uint32_t>(std::log2(cube_size) + 1);
    const glm::ivec2 dither = iblDitherOffsetForFrame(mini_frame_index_);

    // Only mip 0 is convolved.  We pass num_mips=1 so the dispatch helper
    // uses roughness = 0 (sharp / mirror-like).  Higher-roughness response
    // comes from the box-filter mipgen below: mip i becomes a blurred
    // version of mip 0, which approximates - but is NOT identical to -
    // the per-mip GGX convolution the original path produced.  This trades
    // a small amount of physical accuracy at high roughness for a
    // ~num_mips-fold cost reduction per frame.
    dispatchIblMiniMip(
        cmd_buf,
        ibl_mini_pipeline_layout_,
        specular_mini_desc_sets_[0],
        rt_ibl_specular_tex_.image,
        cube_size,
        /*mip_level*/ 0u,
        /*num_mips */ 1u,
        dither,
        mini_frame_index_);

    // Box-filter mipgen for mips 1..N-1 ("downsample buffer will be
    // merged nearby pixel in downsample pass").  Mip 0 is left in
    // SHADER_READ_ONLY_OPTIMAL by the dispatch helper, which is exactly
    // what generateMipmapLevels expects as the input mip's current layout.
    if (num_mips > 1) {
        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_ibl_specular_tex_.image,
            num_mips,
            cube_size,
            cube_size,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
}

void IblCreator::updateIblSheenMapMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        charlie_mini_pipeline_);

    const uint32_t num_mips =
        static_cast<uint32_t>(std::log2(cube_size) + 1);
    const glm::ivec2 dither = iblDitherOffsetForFrame(mini_frame_index_);

    // See updateIblSpecularMapMini for the rationale: only mip 0 is
    // Charlie-convolved; higher mips come from box-filter mipgen.
    dispatchIblMiniMip(
        cmd_buf,
        ibl_mini_pipeline_layout_,
        sheen_mini_desc_sets_[0],
        rt_ibl_sheen_tex_.image,
        cube_size,
        /*mip_level*/ 0u,
        /*num_mips */ 1u,
        dither,
        mini_frame_index_);

    if (num_mips > 1) {
        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_ibl_sheen_tex_.image,
            num_mips,
            cube_size,
            cube_size,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    // Advance the dither + sample-stratification frame counter once per
    // "frame's IBL update".  Bumped here (after the last filter) so all
    // three filters - diffuse / specular / sheen - use the same per-frame
    // dither offset and sample stratum.  Caller expects this method to
    // run last in the IBL update sequence (see application.cpp).
    ++mini_frame_index_;
}

// STUBBED (VRAM cleanup, 2026-08): the Gaussian blur side-channel was
// superseded by the box-filter mipgen + copy inside
// updateIblDiffuseMapMini, and this method had no caller anywhere in
// src/.  Its blur pipeline, compute descriptor sets and the
// tmp specular/sheen blur-source cubes (16 MiB each) were removed.
// Logged no-op for out-of-tree callers.
void IblCreator::blurIblMaps(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {
    (void)cmd_buf;
    (void)cube_size;
    IBL_WARN_STUBBED_ONCE("blurIblMaps");
}

void IblCreator::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {

    createDescriptorSets(
        device,
        descriptor_pool,
        texture_sampler);

    // The descriptor pool was rebuilt by the application (swap-chain
    // recreate); our per-(filter, mip) mini-buffer descriptor sets are
    // therefore stale.  Reallocate them.  Per-mip image views are still
    // valid (they reference the IBL Image objects, which persist), but
    // bindIblMiniTargets() rebuilds them too for symmetry.
    if (ibl_mini_desc_set_layout_ != nullptr) {
        const uint32_t cube_size =
            static_cast<uint32_t>(rt_ibl_specular_tex_.size.x);
        bindIblMiniTargets(device, descriptor_pool, texture_sampler, cube_size);
    }
}

void IblCreator::destroy(
    const std::shared_ptr<renderer::Device>& device)
{
    // VRAM cleanup (2026-08): the blur / full-res graphics pipelines,
    // their pipeline layouts and the compute descriptor set layout no
    // longer exist — nothing to destroy for them.
    device->destroyDescriptorSetLayout(ibl_desc_set_layout_);

    // Mini-buffer IBL convolution.
    device->destroyPipeline(lambertian_mini_pipeline_);
    device->destroyPipeline(ggx_mini_pipeline_);
    device->destroyPipeline(charlie_mini_pipeline_);
    device->destroyPipelineLayout(ibl_mini_pipeline_layout_);
    device->destroyDescriptorSetLayout(ibl_mini_desc_set_layout_);
    // Per-mip cube views are owned ImageView shared_ptrs - clearing the
    // vectors releases them.  The underlying Images are owned by the
    // TextureInfo objects below.
    rt_ibl_diffuse_per_mip_cube_views_.clear();
    rt_ibl_specular_per_mip_cube_views_.clear();
    rt_ibl_sheen_per_mip_cube_views_.clear();

    rt_envmap_tex_.destroy(device);
    tmp_ibl_diffuse_tex_.destroy(device);
    rt_ibl_diffuse_tex_.destroy(device);
    rt_ibl_specular_tex_.destroy(device);
    rt_ibl_sheen_tex_.destroy(device);

}

}//namespace scene_rendering
}//namespace engine
