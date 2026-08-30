#include "wind_field.h"

#include <cmath>
#include <iostream>

#include "helper/engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace game_object {

WindField::WindField(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool)
    : descriptor_pool_(descriptor_pool) {

    // ── Distribution + output textures ───────────────────────────────
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
    // The published wind: SAMPLED (every consumer) + STORAGE (the sim
    // writes it).  Exists from construction so tile descriptor sets can
    // bind it before the first step; it reads as zero wind until then.
    wind_tex_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16_SFLOAT,
        grid,
        (uint32_t)-1,
        *wind_tex_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        renderer::ImageLayout::GENERAL,
        std::source_location::current(),
        renderer::ImageTiling::OPTIMAL,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));

    // ── Region SSBO consumers read (same layout as the LBM's) ────────
    device->createBuffer(
        sizeof(glm::vec4),
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                        HOST_COHERENT_BIT),
        0,
        region_buffer_.buffer,
        region_buffer_.memory,
        std::source_location::current());
    // Publish a dead region immediately: span 0 makes sampleWindFine()
    // return weight 0, so consumers fall back to the coarse tier (or
    // their constant) instead of sampling an unstepped lattice.
    {
        glm::vec4 dead(0.0f, 0.0f, 0.0f, 0.0f);
        device->updateBufferMemory(
            region_buffer_.memory, sizeof(dead), &dead, 0, true);
    }

    sampler_ = device->createSampler(
        renderer::Filter::LINEAR,
        renderer::SamplerAddressMode::CLAMP_TO_EDGE,
        renderer::SamplerMipmapMode::NEAREST,
        /*anisotropy*/ 0.0f,
        std::source_location::current());

    // ── Descriptor layout: wind_patch.comp bindings 0..8 ─────────────
    // 0-5 distributions (src/dst), 6 wind out, 7 airflow 3D, 8 rock.
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
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    desc_set_layout_ = device->createDescriptorSetLayout(bindings);

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::WindPatchParams);
    pipeline_layout_ = device->createPipelineLayout(
        { desc_set_layout_ },
        { push_const_range },
        std::source_location::current());

    pipeline_ = renderer::helper::createComputePipeline(
        device,
        pipeline_layout_,
        "weather/wind_patch_comp.spv",
        std::source_location::current());
}

void WindField::bindSources(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::ImageView>& airflow_view,
    const std::shared_ptr<renderer::ImageView>& rock_layer_view) {
    if (!airflow_view || !rock_layer_view) return;
    // Descriptor sets are created HERE, not in the constructor: the
    // compute set needs the airflow + rock views, and inventing 1x1
    // fallbacks for a sampler3D just to bind earlier buys nothing —
    // update() simply does not dispatch until this has run.  Re-binding
    // (a new terrain, a new rock layer) rewrites the same sets.
    for (int par = 0; par < 2; ++par) {
        if (desc_sets_[par] == nullptr) {
            desc_sets_[par] = device->createDescriptorSets(
                descriptor_pool_, desc_set_layout_, 1)[0];
        }
        renderer::WriteDescriptorList writes;
        writes.reserve(9);
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
            nullptr, wind_tex_->view,
            renderer::ImageLayout::GENERAL);
        renderer::Helper::addOneTexture(
            writes, desc_sets_[par],
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 7,
            sampler_, airflow_view,
            renderer::ImageLayout::GENERAL);
        renderer::Helper::addOneTexture(
            writes, desc_sets_[par],
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 8,
            sampler_, rock_layer_view,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        device->updateDescriptorSets(writes);
    }
    sources_bound_ = true;
    needs_reset_ = true;
    std::cout << "[wind] sources bound — local wind patch live "
              << "(512 m @ 1 m, D2Q9)" << std::endl;
}

void WindField::update(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const glm::vec3& camera_pos,
    const glm::vec3& world_min,
    const glm::vec3& world_range,
    float delta_t) {

    const float span = kGridSize * kCellM;

    prev_origin_ws_ = origin_ws_;
    glm::vec2 want(camera_pos.x - span * 0.5f, camera_pos.z - span * 0.5f);
    origin_ws_ = glm::floor(want / kCellM) * kCellM;
    if (glm::length(origin_ws_ - prev_origin_ws_) > span * 0.5f) {
        needs_reset_ = true;
        prev_origin_ws_ = origin_ws_;
    }
    time_ += delta_t;

    // Region publish — dead (w=0) until the sim can actually step, so
    // consumers never blend toward a zero lattice.
    // LAYOUT: (origin.x, origin.z, cell_m, grid_size) — what
    // sampleWindFine() in wind_field.glsl.h expects.  NOT the LBM
    // water's ordering (x, cell, z, grid); copying that here once put
    // the patch span at origin.z * grid and every consumer sampled
    // garbage UVs.  One header defines the contract; this publishes it.
    glm::vec4 region = sources_bound_
        ? glm::vec4(origin_ws_.x, origin_ws_.y, kCellM, float(kGridSize))
        : glm::vec4(0.0f);
    device->updateBufferMemory(
        region_buffer_.memory, sizeof(region), &region, 0, true);

    if (!sources_bound_ || desc_sets_[parity_] == nullptr) return;

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE, pipeline_);
    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        pipeline_layout_,
        { desc_sets_[parity_] });

    glsl::WindPatchParams params{};
    params.origin_ws      = glm::vec4(origin_ws_.x, 0, origin_ws_.y, 0);
    params.prev_origin_ws = glm::vec4(prev_origin_ws_.x, 0,
                                      prev_origin_ws_.y, 0);
    params.world_min      = glm::vec4(world_min, 0);
    params.world_range    = glm::vec4(world_range, 0);
    params.cell_m         = kCellM;
    // Fixed step for a fixed lattice speed, exactly as the water LBM.
    params.dt             = 1.0f / 60.0f;
    params.sample_height_m = 2.0f;
    params.time           = time_;
    params.inflow_gain    = 0.15f;
    params.gust_scale     = 1.2f;
    params.obstacle_margin_m = 4.0f;
    params.grid_size      = kGridSize;
    params.reset          = needs_reset_ ? 1u : 0u;
    needs_reset_ = false;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->dispatch(kGridSize / 16, kGridSize / 16, 1);

    {
        renderer::BarrierList barrier_list;
        barrier_list.image_barriers.reserve(4);
        renderer::helper::addTexturesToBarrierList(
            barrier_list,
            { f_tex_[1 - parity_][0].image,
              f_tex_[1 - parity_][1].image,
              f_tex_[1 - parity_][2].image,
              wind_tex_->image },
            renderer::ImageLayout::GENERAL,
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));
        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_3_FLAG_BITS(PipelineStage, COMPUTE_SHADER_BIT,
                            VERTEX_SHADER_BIT, FRAGMENT_SHADER_BIT));
    }

    parity_ ^= 1u;
}

void WindField::destroy(const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(desc_set_layout_);
    device->destroyPipelineLayout(pipeline_layout_);
    device->destroyPipeline(pipeline_);
    for (int par = 0; par < 2; ++par)
        for (int t = 0; t < 3; ++t)
            f_tex_[par][t].destroy(device);
    if (wind_tex_) wind_tex_->destroy(device);
    region_buffer_.destroy(device);
}

}  // namespace game_object
}  // namespace engine
