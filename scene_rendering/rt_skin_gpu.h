#pragma once
#include "renderer/renderer.h"
#include "rt_skin_types.h"
namespace engine::scene_rendering {
class RtSkinGpu {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    RtSkinGpu(const std::shared_ptr<renderer::Device>&, const std::shared_ptr<renderer::DescriptorPool>&);
    ~RtSkinGpu();
    // slot must have completed its prior graphics fence before prepare.
    void prepare(const std::vector<RtSkinBatch>& batches, uint32_t slot);
    bool changed() const;
    bool pending() const;
    uint32_t vertices() const;
    uint32_t indices() const;
    uint32_t chunks() const;
    const std::vector<uint32_t>& topology() const;
    std::shared_ptr<renderer::CommandBuffer> record(const std::shared_ptr<renderer::CommandBuffer>& graphics,
                renderer::BufferInfo& positions, renderer::BufferInfo& indices,
                renderer::BufferInfo& chunks, renderer::BufferInfo& headers);
    void finish();
    // Called after previous-frame fence + deferred host writes, before graphics
    // submission. Its semaphore must be waited by graphics at AS-build/compute.
    std::shared_ptr<renderer::Semaphore> submit();
};
}
