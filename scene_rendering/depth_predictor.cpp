#include "depth_predictor.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"
#include <algorithm>
#include <iostream>

namespace engine::scene_rendering {
namespace er = renderer;

struct DepthPredictor::Impl {
    std::shared_ptr<er::Device> device;
    std::shared_ptr<er::DescriptorPool> pool;
    std::shared_ptr<er::DescriptorSetLayout> layout;
    std::shared_ptr<er::PipelineLayout> pipeline_layout;
    std::shared_ptr<er::Pipeline> pipeline;
    std::shared_ptr<er::DescriptorSet> desc;
    // R32_UINT scatter target (atomicMin) and the R32F dilated result
    // hiz_build reads as its mip-0 source.
    er::TextureInfo pred_u;
    er::TextureInfo pred_f;
    glm::uvec2 size{0, 0};
    uint32_t last_pixels = 0;
    bool primed = false;   // pred_f has been written at least once

    void destroyTargets() {
        if (pred_u.image) pred_u.destroy(device);
        if (pred_f.image) pred_f.destroy(device);
        pred_u = {}; pred_f = {};
        size = {0, 0};
        primed = false;
    }
};

DepthPredictor::DepthPredictor(const std::shared_ptr<er::Device>& device,
                               const std::shared_ptr<er::DescriptorPool>& pool)
    : impl_(std::make_unique<Impl>()) {
    auto& s = *impl_;
    s.device = device; s.pool = pool;
    std::vector<er::DescriptorSetLayoutBinding> b;
    b.push_back(er::helper::getBufferDescriptionSetLayoutBinding(
        0, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), er::DescriptorType::STORAGE_BUFFER));
    b.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        1, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), er::DescriptorType::COMBINED_IMAGE_SAMPLER));
    b.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        2, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), er::DescriptorType::COMBINED_IMAGE_SAMPLER));
    b.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        3, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), er::DescriptorType::STORAGE_IMAGE));
    b.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        4, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), er::DescriptorType::STORAGE_IMAGE));
    s.layout = device->createDescriptorSetLayout(b);
    s.pipeline_layout = er::helper::createComputePipelineLayout(device, { s.layout }, 32);
    s.pipeline = er::helper::createComputePipeline(
        device, s.pipeline_layout, "depth_predict_comp.spv", std::source_location::current());
    s.desc = device->createDescriptorSets(pool, s.layout, 1)[0];
    std::cout << "[DEPTH_PREDICT] frame-ahead depth predictor ready" << std::endl;
}

DepthPredictor::~DepthPredictor() {
    auto& s = *impl_;
    s.destroyTargets();
    if (s.pipeline) s.device->destroyPipeline(s.pipeline);
    if (s.pipeline_layout) s.device->destroyPipelineLayout(s.pipeline_layout);
    if (s.layout) s.device->destroyDescriptorSetLayout(s.layout);
}

void DepthPredictor::resize(const glm::uvec2& size) {
    auto& s = *impl_;
    if (size == s.size && s.pred_f.image) return;
    s.destroyTargets();
    if (size.x == 0 || size.y == 0) return;
    const auto usage = SET_2_FLAG_BITS(ImageUsage, STORAGE_BIT, SAMPLED_BIT);
    er::Helper::create2DTextureImage(
        s.device, er::Format::R32_UINT, size, 1, s.pred_u, usage,
        er::ImageLayout::GENERAL, std::source_location::current());
    er::Helper::create2DTextureImage(
        s.device, er::Format::R32_SFLOAT, size, 1, s.pred_f, usage,
        er::ImageLayout::GENERAL, std::source_location::current());
    s.size = size;
}

const er::TextureInfo& DepthPredictor::predicted() const { return impl_->pred_f; }
bool DepthPredictor::ready() const { return impl_->pred_f.image && impl_->primed; }
uint32_t DepthPredictor::lastDispatchPixels() const { return impl_->last_pixels; }

void DepthPredictor::recordInline(
    const std::shared_ptr<er::CommandBuffer>& cmd,
    const std::shared_ptr<er::Sampler>& sampler,
    const std::shared_ptr<er::ImageView>& src_depth,
    const glm::uvec2& src_size,
    const std::shared_ptr<er::ImageView>& src_motion3d,
    const std::shared_ptr<er::BufferInfo>& camera_buffer,
    bool use_motion) {
    auto& s = *impl_;
    if (!s.pred_f.image || !src_depth || !camera_buffer) return;
    if (use_motion && !src_motion3d) use_motion = false;

    er::WriteDescriptorList w;
    w.reserve(5);
    er::Helper::addOneBuffer(w, s.desc, er::DescriptorType::STORAGE_BUFFER, 0,
        camera_buffer->buffer, uint32_t(sizeof(glsl::ViewCameraInfo)));
    er::Helper::addOneTexture(w, s.desc, er::DescriptorType::COMBINED_IMAGE_SAMPLER, 1,
        sampler, src_depth, er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    // the motion binding must be valid even when unused; the depth view
    // stands in (never read: pc.use_motion == 0)
    er::Helper::addOneTexture(w, s.desc, er::DescriptorType::COMBINED_IMAGE_SAMPLER, 2,
        sampler, use_motion ? src_motion3d : src_depth, er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(w, s.desc, er::DescriptorType::STORAGE_IMAGE, 3,
        nullptr, s.pred_u.view, er::ImageLayout::GENERAL);
    er::Helper::addOneTexture(w, s.desc, er::DescriptorType::STORAGE_IMAGE, 4,
        nullptr, s.pred_f.view, er::ImageLayout::GENERAL);
    s.device->updateDescriptorSets(w);

    struct PC { uint32_t sx, sy, dx, dy, phase, use_motion, pad0, pad1; };
    cmd->beginDebugUtilsLabel("Frame-ahead depth predict");
    cmd->bindPipeline(er::PipelineBindPoint::COMPUTE, s.pipeline);
    cmd->bindDescriptorSets(er::PipelineBindPoint::COMPUTE, s.pipeline_layout, { s.desc });

    er::ImageResourceInfo rw = {
        er::ImageLayout::GENERAL,
        SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    auto dispatch = [&](uint32_t phase, const glm::uvec2& n) {
        PC pc{ src_size.x, src_size.y, s.size.x, s.size.y, phase, use_motion ? 1u : 0u, 0u, 0u };
        cmd->pushConstants(SET_FLAG_BIT(ShaderStage, COMPUTE_BIT), s.pipeline_layout, &pc, sizeof(pc));
        cmd->dispatch((n.x + 7u) / 8u, (n.y + 7u) / 8u, 1);
    };
    dispatch(0, s.size);                                   // clear to far
    cmd->addImageBarrier(s.pred_u.image, rw, rw, 0, 1, 0, 1);
    dispatch(1, src_size);                                 // scatter (atomicMin)
    cmd->addImageBarrier(s.pred_u.image, rw, rw, 0, 1, 0, 1);
    dispatch(2, s.size);                                   // dilate -> r32f
    // hiz_build samples pred_f through a sampler in GENERAL layout
    er::ImageResourceInfo to_sampled = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    cmd->addImageBarrier(s.pred_f.image, rw, to_sampled, 0, 1, 0, 1);
    cmd->endDebugUtilsLabel();
    s.last_pixels = src_size.x * src_size.y;
    s.primed = true;
}

}
