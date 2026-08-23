#include "lbm_water.h"

#include <cmath>
#include <iostream>

#include "helper/engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace game_object {

LbmWater::LbmWater(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool) {

    // ── Distribution + surface textures ──────────────────────────────
    const glm::uvec2 grid(kGridSize, kGridSize);
    for (int par = 0; par < 2; ++par) {
        for (int t = 0; t < 3; ++t) {
            renderer::Helper::create2DTextureImage(
                device,
                renderer::Format::R32G32B32A32_SFLOAT,
                grid,
                (uint32_t)-1,
                f_tex_[par][t],
                SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
                renderer::ImageLayout::GENERAL,
                std::source_location::current(),
                renderer::ImageTiling::OPTIMAL,
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));
        }
    }
    surface_tex_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16B16A16_SFLOAT,
        grid,
        (uint32_t)-1,
        *surface_tex_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        renderer::ImageLayout::GENERAL,
        std::source_location::current(),
        renderer::ImageTiling::OPTIMAL,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));

    // ── Region SSBO the WATER_ATTR pass reads ────────────────────────
    device->createBuffer(
        sizeof(glm::vec4),
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                        HOST_COHERENT_BIT),
        0,
        region_buffer_.buffer,
        region_buffer_.memory,
        std::source_location::current());

    // ── Water-level fallback (binding 7) ─────────────────────────────
    // A 1x1 ZERO texel (no water anywhere) keeps the descriptor valid
    // until the terrain apply binds the real <stem>_hydro_pondz.png
    // via setWaterLevelMap; the shader only derives force from it when
    // level_map_world.w > 0.
    flow_sampler_ = device->createSampler(
        renderer::Filter::LINEAR,
        renderer::SamplerAddressMode::CLAMP_TO_EDGE,
        renderer::SamplerMipmapMode::NEAREST,
        /*anisotropy*/ 0.0f,
        std::source_location::current());
    {
        const uint8_t zero[4] = { 0, 0, 0, 255 };
        renderer::Helper::create2DTextureImage(
            device,
            renderer::Format::R8G8B8A8_UNORM,
            1, 1,
            zero,
            flow_fallback_tex_.image,
            flow_fallback_tex_.memory,
            std::source_location::current());
        flow_fallback_tex_.size = glm::uvec3(1, 1, 1);
        flow_fallback_tex_.view = device->createImageView(
            flow_fallback_tex_.image,
            renderer::ImageViewType::VIEW_2D,
            renderer::Format::R8G8B8A8_UNORM,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current());
    }

    // ── GENERATED FLOWMAP output (binding 8) ─────────────────────────
    // The sim's macroscopic velocity per cell, written every step and
    // sampled by the water shading (set 3 binding 2) to advect the
    // surface detail along the current.
    flow_tex_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16_SFLOAT,
        grid,
        (uint32_t)-1,
        *flow_tex_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        renderer::ImageLayout::GENERAL,
        std::source_location::current(),
        renderer::ImageTiling::OPTIMAL,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));

    // ── Descriptor layout: 7 storage images (src A/B/C, dst A/B/C,
    //    surface out) + the water-level sampler + the generated-flow
    //    output — lbm_water.comp bindings 0..8 ───────────────────────
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(9);
    for (int i = 0; i < 7; ++i) {
        bindings[i] =
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                i, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                renderer::DescriptorType::STORAGE_IMAGE);
    }
    bindings[7] =
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            7, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[8] =
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            8, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::STORAGE_IMAGE);
    desc_set_layout_ = device->createDescriptorSetLayout(bindings);

    for (int par = 0; par < 2; ++par) {
        desc_sets_[par] = device->createDescriptorSets(
            descriptor_pool, desc_set_layout_, 1)[0];
        renderer::WriteDescriptorList writes;
        writes.reserve(7);
        const int src = par, dst = 1 - par;
        for (int t = 0; t < 3; ++t) {
            renderer::Helper::addOneTexture(
                writes, desc_sets_[par],
                renderer::DescriptorType::STORAGE_IMAGE, t,
                nullptr, f_tex_[src][t].view,
                renderer::ImageLayout::GENERAL);
        }
        for (int t = 0; t < 3; ++t) {
            renderer::Helper::addOneTexture(
                writes, desc_sets_[par],
                renderer::DescriptorType::STORAGE_IMAGE, 3 + t,
                nullptr, f_tex_[dst][t].view,
                renderer::ImageLayout::GENERAL);
        }
        renderer::Helper::addOneTexture(
            writes, desc_sets_[par],
            renderer::DescriptorType::STORAGE_IMAGE, 6,
            nullptr, surface_tex_->view,
            renderer::ImageLayout::GENERAL);
        renderer::Helper::addOneTexture(
            writes, desc_sets_[par],
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 7,
            flow_sampler_, flow_fallback_tex_.view,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        renderer::Helper::addOneTexture(
            writes, desc_sets_[par],
            renderer::DescriptorType::STORAGE_IMAGE, 8,
            nullptr, flow_tex_->view,
            renderer::ImageLayout::GENERAL);
        device->updateDescriptorSets(writes);
    }

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::LbmWaterParams);
    pipeline_layout_ = device->createPipelineLayout(
        { desc_set_layout_ },
        { push_const_range },
        std::source_location::current());

    pipeline_ = renderer::helper::createComputePipeline(
        device,
        pipeline_layout_,
        "lbm_water_comp.spv",
        std::source_location::current());
}

void LbmWater::setWaterLevelMap(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::ImageView>& level_view) {
    const auto& view = level_view ? level_view : flow_fallback_tex_.view;
    for (int par = 0; par < 2; ++par) {
        renderer::WriteDescriptorList writes;
        writes.reserve(1);
        renderer::Helper::addOneTexture(
            writes, desc_sets_[par],
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 7,
            flow_sampler_, view,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        device->updateDescriptorSets(writes);
    }
    level_map_bound_ = (level_view != nullptr);
    std::cout << "[lbm-water] water-level map "
              << (level_map_bound_
                      ? "bound — the sim now generates flow from the "
                        "surface gradient"
                      : "cleared (no force; legacy drift)")
              << std::endl;
}

void LbmWater::update(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const glm::vec3& camera_pos,
    const glm::vec2& flow_dir,
    float delta_t) {

    const float span = kGridSize * kCellM;

    // Recentre the patch on the camera, snapped to whole cells so the
    // pull-stream shift is exact.  A jump beyond half the patch resets
    // the lattice (teleport, map load) instead of streaming garbage.
    prev_origin_ws_ = origin_ws_;
    glm::vec2 want(camera_pos.x - span * 0.5f, camera_pos.z - span * 0.5f);
    origin_ws_ = glm::floor(want / kCellM) * kCellM;
    if (glm::length(origin_ws_ - prev_origin_ws_) > span * 0.5f) {
        needs_reset_ = true;
        prev_origin_ws_ = origin_ws_;
    }
    time_ += delta_t;

    // Region publish (host-coherent; tiny).
    // NOTE: y carries cell size, w the grid size — see LbmRegionBuf in
    // tile_water.frag's WATER_ATTR branch.
    glm::vec4 region(origin_ws_.x, kCellM, origin_ws_.y, float(kGridSize));

    device->updateBufferMemory(
        region_buffer_.memory, sizeof(region), &region, 0, true);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE, pipeline_);
    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        pipeline_layout_,
        { desc_sets_[parity_] });

    glsl::LbmWaterParams params{};
    params.origin_ws      = glm::vec4(origin_ws_.x, 0, origin_ws_.y, 0);
    params.prev_origin_ws = glm::vec4(prev_origin_ws_.x, 0,
                                      prev_origin_ws_.y, 0);
    params.flow_dir       = glm::vec4(flow_dir, 0, 0);
    // Water-level map mapping (covers the whole terrain footprint);
    // w = slope->flow gain in (m/s) per (m/m) of surface slope: river
    // surfaces run ~0.002-0.01 m/m, so 250 lands currents in the
    // 0.5-2 m/s band (the shader caps at 2).  0 = no map bound.
    params.level_map_world = level_map_bound_
        ? glm::vec4(-kTerrainMapMeters * 0.5f,
                    -kTerrainMapMeters * 0.5f,
                    1.0f / kTerrainMapMeters, 250.0f)
        : glm::vec4(0.0f);
    params.cell_m         = kCellM;
    // Fixed step: frame delta jitters, and LBM stability wants a
    // constant lattice speed.  One step/frame at 60 fps nominal.
    params.dt             = 1.0f / 60.0f;
    params.rest_depth     = 0.6f;
    params.time           = time_;
    params.flow_strength  = 0.35f;
    params.normal_amp     = 26.0f;
    params.grid_size      = kGridSize;
    params.reset          = needs_reset_ ? 1u : 0u;
    needs_reset_ = false;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->dispatch(kGridSize / 16, kGridSize / 16, 1);

    // Make this step's writes (dst distributions + surface) visible to
    // next frame's compute reads and this frame's fragment sampling.
    {
        renderer::BarrierList barrier_list;
        barrier_list.image_barriers.reserve(5);
        renderer::helper::addTexturesToBarrierList(
            barrier_list,
            { f_tex_[1 - parity_][0].image,
              f_tex_[1 - parity_][1].image,
              f_tex_[1 - parity_][2].image,
              surface_tex_->image,
              flow_tex_->image },
            renderer::ImageLayout::GENERAL,
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));
        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_2_FLAG_BITS(PipelineStage, COMPUTE_SHADER_BIT,
                            FRAGMENT_SHADER_BIT));
    }

    parity_ ^= 1u;
}

void LbmWater::destroy(const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(desc_set_layout_);
    device->destroyPipelineLayout(pipeline_layout_);
    device->destroyPipeline(pipeline_);
    for (int par = 0; par < 2; ++par)
        for (int t = 0; t < 3; ++t)
            f_tex_[par][t].destroy(device);
    if (surface_tex_) surface_tex_->destroy(device);
    if (flow_tex_) flow_tex_->destroy(device);
    flow_fallback_tex_.destroy(device);
    region_buffer_.destroy(device);
}

}  // namespace game_object
}  // namespace engine
