#include "wind_field.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "helper/engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace game_object {

std::vector<WindField::Injector> WindField::s_injectors_;

WindField::WindField(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool)
    : descriptor_pool_(descriptor_pool) {

    // ── Distribution textures, per level ─────────────────────────────
    const glm::uvec2 grid(kGridSize, kGridSize);
    for (uint32_t L = 0; L < kLevels; ++L) {
        for (int par = 0; par < 2; ++par) {
            for (int t = 0; t < 3; ++t) {
                renderer::Helper::create2DTextureImage(
                    device,
                    renderer::Format::R32G32B32A32_SFLOAT,
                    grid,
                    (uint32_t)-1,
                    f_tex_[L][par][t],
                    SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
                    renderer::ImageLayout::GENERAL,
                    std::source_location::current(),
                    renderer::ImageTiling::OPTIMAL,
                    SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));
            }
        }
    }

    // ── The published wind: one rg16f array, layer = level ───────────
    // SAMPLED (every consumer) + STORAGE (the sim writes it).  Exists
    // from construction so descriptor sets can bind it before the
    // first step; it reads as zero wind until then.
    const renderer::Format wind_fmt = renderer::Format::R16G16_SFLOAT;
    wind_image_ = device->createImage(
        renderer::ImageType::TYPE_2D,
        glm::uvec3(kGridSize, kGridSize, 1),
        wind_fmt,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        renderer::ImageTiling::OPTIMAL,
        renderer::ImageLayout::UNDEFINED,
        std::source_location::current(),
        0, false, 1, 1, kLevels);
    {
        auto mem_req = device->getImageMemoryRequirements(wind_image_);
        wind_memory_ = device->allocateMemory(
            mem_req.size, mem_req.memory_type_bits,
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT), 0);
        device->bindImageMemory(wind_image_, wind_memory_);
    }
    wind_array_view_ = device->createImageView(
        wind_image_, renderer::ImageViewType::VIEW_2D_ARRAY, wind_fmt,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT), std::source_location::current(),
        0, 1, 0, kLevels);
    for (uint32_t L = 0; L < kLevels; ++L) {
        wind_layer_views_[L] = device->createImageView(
            wind_image_, renderer::ImageViewType::VIEW_2D, wind_fmt,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT), std::source_location::current(),
            0, 1, L, 1);
    }
    renderer::Helper::transitionImageLayout(
        device, wind_image_, wind_fmt,
        renderer::ImageLayout::UNDEFINED, renderer::ImageLayout::GENERAL,
        0, 1, 0, kLevels);
    wind_tex_ = std::make_shared<renderer::TextureInfo>();
    wind_tex_->image = wind_image_;      // shared; destroyed once, below
    wind_tex_->view  = wind_layer_views_[0];
    wind_tex_->size  = glm::uvec3(kGridSize, kGridSize, 1);

    // ── Info SSBOs consumers read ────────────────────────────────────
    auto make_host_buffer = [&](renderer::BufferInfo& buf, uint64_t size) {
        device->createBuffer(
            size,
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                            HOST_COHERENT_BIT),
            0, buf.buffer, buf.memory,
            std::source_location::current());
    };
    make_host_buffer(region_buffer_, sizeof(glm::vec4));
    make_host_buffer(clip_info_buffer_, sizeof(glsl::WindClipInfo));
    make_host_buffer(injector_buffer_,
                     sizeof(glsl::WindInjector) * kWindInjectorMax);
    // Publish dead regions immediately: span 0 makes sampleWindFine() /
    // sampleWindClip() return weight 0, so consumers fall back to the
    // coarse tier (or their constant) instead of sampling an unstepped
    // lattice.
    {
        glm::vec4 dead(0.0f);
        device->updateBufferMemory(
            region_buffer_.memory, sizeof(dead), &dead, 0, true);
        glsl::WindClipInfo info{};
        device->updateBufferMemory(
            clip_info_buffer_.memory, sizeof(info), &info, 0, true);
        std::vector<glsl::WindInjector> zero(kWindInjectorMax);
        device->updateBufferMemory(
            injector_buffer_.memory, sizeof(glsl::WindInjector) * kWindInjectorMax,
            zero.data(), 0, true);
    }

    sampler_ = device->createSampler(
        renderer::Filter::LINEAR,
        renderer::SamplerAddressMode::CLAMP_TO_EDGE,
        renderer::SamplerMipmapMode::NEAREST,
        /*anisotropy*/ 0.0f,
        std::source_location::current());

    // ── Descriptor layout: wind_patch.comp bindings 0..10 ────────────
    // 0-5 distributions (src/dst), 6 wind out, 7 airflow 3D, 8 rock,
    // 9 coarser level's wind, 10 injectors.
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(11);
    for (int i = 0; i < 7; ++i) {
        bindings[i] =
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                i, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                renderer::DescriptorType::STORAGE_IMAGE);
    }
    for (int i = 7; i < 10; ++i) {
        bindings[i] =
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                i, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    }
    bindings[10] =
        renderer::helper::getBufferDescriptionSetLayoutBinding(
            10, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::STORAGE_BUFFER);
    desc_set_layout_ = device->createDescriptorSetLayout(bindings);

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::WindPatchParams);
    pipeline_layout_ = device->createPipelineLayout(
        { desc_set_layout_ },
        { push_const_range },
        std::source_location::current());

    // A build dir without wind_patch_comp.spv degrades to "no wind"
    // (dead regions, zero textures — consumers hold their constants),
    // never to a missing object: the global descriptor set binds the
    // clipmap unconditionally.
    try {
        pipeline_ = renderer::helper::createComputePipeline(
            device,
            pipeline_layout_,
            "weather/wind_patch_comp.spv",
            std::source_location::current());
        pipeline_ok_ = pipeline_ != nullptr;
    } catch (const std::exception& e) {
        pipeline_ = nullptr;
        pipeline_ok_ = false;
        std::cout << "[wind] wind_patch_comp.spv unavailable (" << e.what()
                  << ") — wind clipmap stays calm.  Re-run "
                     "GenerateProjectFiles.bat so wind_patch.comp is compiled."
                  << std::endl;
    }
}

void WindField::bindSources(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::ImageView>& airflow_view,
    const std::shared_ptr<renderer::ImageView>& rock_layer_view) {
    if (!airflow_view || !rock_layer_view || !pipeline_ok_) return;
    // Descriptor sets are created HERE, not in the constructor: the
    // compute set needs the airflow + rock views, and inventing 1x1
    // fallbacks for a sampler3D just to bind earlier buys nothing —
    // update() simply does not dispatch until this has run.  Re-binding
    // (a new terrain, a new rock layer) rewrites the same sets.
    for (uint32_t L = 0; L < kLevels; ++L) {
        for (int par = 0; par < 2; ++par) {
            auto& set = desc_sets_[L][par];
            if (set == nullptr) {
                set = device->createDescriptorSets(
                    descriptor_pool_, desc_set_layout_, 1)[0];
            }
            renderer::WriteDescriptorList writes;
            writes.reserve(11);
            const int src = par, dst = 1 - par;
            for (int t = 0; t < 3; ++t) {
                renderer::Helper::addOneTexture(
                    writes, set, renderer::DescriptorType::STORAGE_IMAGE, t,
                    nullptr, f_tex_[L][src][t].view,
                    renderer::ImageLayout::GENERAL);
            }
            for (int t = 0; t < 3; ++t) {
                renderer::Helper::addOneTexture(
                    writes, set, renderer::DescriptorType::STORAGE_IMAGE, 3 + t,
                    nullptr, f_tex_[L][dst][t].view,
                    renderer::ImageLayout::GENERAL);
            }
            renderer::Helper::addOneTexture(
                writes, set, renderer::DescriptorType::STORAGE_IMAGE, 6,
                nullptr, wind_layer_views_[L],
                renderer::ImageLayout::GENERAL);
            renderer::Helper::addOneTexture(
                writes, set, renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 7,
                sampler_, airflow_view,
                renderer::ImageLayout::GENERAL);
            renderer::Helper::addOneTexture(
                writes, set, renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 8,
                sampler_, rock_layer_view,
                renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            // The level above (the top level binds itself and ignores it).
            const uint32_t coarse = std::min(L + 1u, kLevels - 1u);
            renderer::Helper::addOneTexture(
                writes, set, renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 9,
                sampler_, wind_layer_views_[coarse],
                renderer::ImageLayout::GENERAL);
            renderer::Helper::addOneBuffer(
                writes, set, renderer::DescriptorType::STORAGE_BUFFER, 10,
                injector_buffer_.buffer,
                sizeof(glsl::WindInjector) * kWindInjectorMax);
            device->updateDescriptorSets(writes);
        }
    }
    sources_bound_ = true;
    needs_reset_ = true;
    std::cout << "[wind] sources bound — wind clipmap live ("
              << kLevels << " levels, 512 cells, 1 m .. "
              << cellMeters(kLevels - 1) << " m, D2Q9)" << std::endl;
}

void WindField::addInjector(const glm::vec2& pos_xz, const glm::vec2& vel_xz,
                            float radius_m, float coupling) {
    if (glm::dot(vel_xz, vel_xz) < 0.01f) return;      // standing still
    s_injectors_.push_back({ pos_xz, vel_xz, radius_m, coupling });
}

void WindField::update(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const glm::vec3& camera_pos,
    const glm::vec3& world_min,
    const glm::vec3& world_range,
    float delta_t) {

    time_ += delta_t;
    const bool live = sources_bound_ && pipeline_ok_;

    // ── Recentre every level on the camera (snapped to its cell) ─────
    glsl::WindClipInfo info{};
    for (uint32_t L = 0; L < kLevels; ++L) {
        const float cell = cellMeters(L);
        const float span = kGridSize * cell;
        prev_origin_ws_[L] = origin_ws_[L];
        glm::vec2 want(camera_pos.x - span * 0.5f, camera_pos.z - span * 0.5f);
        origin_ws_[L] = glm::floor(want / cell) * cell;
        if (glm::length(origin_ws_[L] - prev_origin_ws_[L]) > span * 0.5f) {
            needs_reset_ = true;
            prev_origin_ws_[L] = origin_ws_[L];
        }
        // LAYOUT: (origin.x, origin.z, cell_m, grid_size) — what
        // sampleWindFine() / sampleWindClip() expect.  Dead (w = 0)
        // until the sim can actually step, so consumers never blend
        // toward a zero lattice.
        info.region[L] = live
            ? glm::vec4(origin_ws_[L].x, origin_ws_[L].y, cell, float(kGridSize))
            : glm::vec4(0.0f);
    }
    info.params = glm::vec4(live ? float(kLevels) : 0.0f, 0.12f, time_, delta_t);
    device->updateBufferMemory(
        region_buffer_.memory, sizeof(glm::vec4), &info.region[0], 0, true);
    device->updateBufferMemory(
        clip_info_buffer_.memory, sizeof(info), &info, 0, true);

    // ── Injectors: the nearest kWindInjectorMax this frame ───────────
    uint32_t injector_count = 0;
    if (live && !s_injectors_.empty()) {
        const glm::vec2 cam_xz(camera_pos.x, camera_pos.z);
        if (s_injectors_.size() > size_t(kWindInjectorMax)) {
            std::nth_element(
                s_injectors_.begin(),
                s_injectors_.begin() + kWindInjectorMax,
                s_injectors_.end(),
                [&](const Injector& a, const Injector& b) {
                    return glm::dot(a.pos - cam_xz, a.pos - cam_xz) <
                           glm::dot(b.pos - cam_xz, b.pos - cam_xz);
                });
            s_injectors_.resize(kWindInjectorMax);
        }
        std::vector<glsl::WindInjector> up(s_injectors_.size());
        for (size_t i = 0; i < s_injectors_.size(); ++i) {
            const auto& in = s_injectors_[i];
            up[i].pos_r = glm::vec4(in.pos.x, in.pos.y, in.radius, in.coupling);
            up[i].vel   = glm::vec4(in.vel.x, in.vel.y, 0.0f, 0.0f);
        }
        injector_count = uint32_t(up.size());
        device->updateBufferMemory(
            injector_buffer_.memory, sizeof(glsl::WindInjector) * up.size(),
            up.data(), 0, true);
    }
    s_injectors_.clear();

    if (!live) return;

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, pipeline_);

    // Coarse to fine: each level's inflow is the level above it, so the
    // top steps first and the barrier after each dispatch publishes
    // its wind to the next one down (and to every consumer).
    for (int Li = int(kLevels) - 1; Li >= 0; --Li) {
        const uint32_t L = uint32_t(Li);
        if (desc_sets_[L][parity_] == nullptr) continue;
        const float cell = cellMeters(L);

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            pipeline_layout_,
            { desc_sets_[L][parity_] });

        glsl::WindPatchParams params{};
        params.origin_ws      = glm::vec4(origin_ws_[L].x, 0, origin_ws_[L].y, 0);
        params.prev_origin_ws = glm::vec4(prev_origin_ws_[L].x, 0,
                                          prev_origin_ws_[L].y, 0);
        params.coarse_region  = (L + 1u < kLevels) ? info.region[L + 1u]
                                                   : glm::vec4(0.0f);
        params.world_min      = glm::vec4(world_min, 0);
        params.world_range    = glm::vec4(world_range, 0);
        params.cell_m         = cell;
        // Fixed step for a fixed lattice speed, exactly as the water LBM.
        params.dt             = 1.0f / 60.0f;
        params.sample_height_m = 2.0f;
        params.time           = time_;
        params.inflow_gain    = 0.15f;
        // The procedural gust belongs to the finest level only: the
        // coarser ones carry the weather and the wakes, and a gust
        // term stacked per level would compound.
        params.gust_scale     = (L == 0u) ? 1.2f : 0.0f;
        // A 64 m cell cannot see a 4 m knoll; only ground standing
        // well above the slice at that scale should block.
        params.obstacle_margin_m = 4.0f + 2.0f * cell;
        params.grid_size      = kGridSize;
        params.reset          = needs_reset_ ? 1u : 0u;
        params.injector_count = injector_count;
        params.level          = L;
        params.has_coarse     = (L + 1u < kLevels) ? 1u : 0u;

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->dispatch(kGridSize / 16, kGridSize / 16, 1);

        renderer::BarrierList barrier_list;
        barrier_list.image_barriers.reserve(4);
        renderer::helper::addTexturesToBarrierList(
            barrier_list,
            { f_tex_[L][1 - parity_][0].image,
              f_tex_[L][1 - parity_][1].image,
              f_tex_[L][1 - parity_][2].image,
              wind_image_ },
            renderer::ImageLayout::GENERAL,
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));
        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_3_FLAG_BITS(PipelineStage, COMPUTE_SHADER_BIT,
                            VERTEX_SHADER_BIT, FRAGMENT_SHADER_BIT));
    }
    needs_reset_ = false;
    parity_ ^= 1u;
}

void WindField::destroy(const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(desc_set_layout_);
    device->destroyPipelineLayout(pipeline_layout_);
    if (pipeline_) device->destroyPipeline(pipeline_);
    for (uint32_t L = 0; L < kLevels; ++L)
        for (int par = 0; par < 2; ++par)
            for (int t = 0; t < 3; ++t)
                f_tex_[L][par][t].destroy(device);
    // wind_tex_ only aliases the array: release its handles without a
    // second destroy.
    if (wind_tex_) { wind_tex_->image = nullptr; wind_tex_->view = nullptr; }
    for (auto& v : wind_layer_views_) v.reset();
    wind_array_view_.reset();
    if (wind_image_) device->destroyImage(wind_image_);
    if (wind_memory_) device->freeMemory(wind_memory_);
    wind_image_ = nullptr;
    wind_memory_ = nullptr;
    region_buffer_.destroy(device);
    clip_info_buffer_.destroy(device);
    injector_buffer_.destroy(device);
}

}  // namespace game_object
}  // namespace engine
