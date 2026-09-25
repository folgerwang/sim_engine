#pragma once
// depth_predictor.h — FRAME-AHEAD DEPTH PREDICTION (see depth_predict.comp)
//
// The simulation is committed one frame ahead of the picture.  Once
// frame N's depth is final, the camera for N+1 is exact and every
// dynamic surface has written its forward motion (G-buffer RT4), so the
// depth buffer frame N+1 will produce can be BUILT rather than drawn:
// reproject frame N's depth through next_view_proj, dynamic pixels
// offset by their motion.  hiz_build then turns it into the min/max
// pyramid the N+1 visibility cull tests against, which replaces the
// depth prepass as the occluder source.
//
// Where it runs.  The pass is written to run on the async compute
// queue beside frame N's shading tail (the RtSkinGpu pattern); the
// images it reads are graphics-owned, so until they are created with
// CONCURRENT sharing it records into the graphics command buffer at the
// tail of the frame instead (recordInline).  Same shader, same output.
#include "renderer/renderer.h"

namespace engine::scene_rendering {

class DepthPredictor {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    DepthPredictor(const std::shared_ptr<renderer::Device>& device,
                   const std::shared_ptr<renderer::DescriptorPool>& pool);
    ~DepthPredictor();

    // (Re)create the predicted-depth targets at `size` (the scene depth
    // size).  Call on init and on every resize.
    void resize(const glm::uvec2& size);

    // The predicted depth (R32F, GENERAL layout after a record) --
    // hand this to writeHiZBuildDescriptors as the mip-0 source.
    const renderer::TextureInfo& predicted() const;
    bool ready() const;

    // Record the three phases into `cmd`.  src_depth / src_motion3d are
    // sampled (SHADER_READ_ONLY); camera_buffer is the main camera UBO
    // (its next_view_proj is the N+1 camera).  use_motion = false runs
    // a camera-only prediction (motion view may then be null).
    void recordInline(const std::shared_ptr<renderer::CommandBuffer>& cmd,
                      const std::shared_ptr<renderer::Sampler>& sampler,
                      const std::shared_ptr<renderer::ImageView>& src_depth,
                      const glm::uvec2& src_size,
                      const std::shared_ptr<renderer::ImageView>& src_motion3d,
                      const std::shared_ptr<renderer::BufferInfo>& camera_buffer,
                      bool use_motion);

    // Stats for the profiler / debug menu.
    uint32_t lastDispatchPixels() const;
};

}
