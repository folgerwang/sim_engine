#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class ClusterRenderer;
class Skydome;

// ─── DynamicCubemap ──────────────────────────────────────────────────────
// Real-time dynamic reflection probe at the main camera.
//
// Per-frame strategy (Nanite-style temporal amortisation):
//   • One face out of six is freshly rendered each frame from the current
//     camera position with a 90° square view.  Captures opaque cluster
//     geometry + skydome into RGBA16F colour and a matching D32 depth
//     slice.  At 60 Hz this gives a fully-refreshed cubemap every 100 ms
//     (6-frame cycle).
//   • The other five faces are reprojected from their stored capture
//     position to the current camera position via a depth-aware compute
//     pass.  The shader walks each output texel's view ray, samples the
//     stored depth at the old origin, reconstructs the world point, and
//     re-resolves the colour at the new origin.  Distant geometry warps
//     correctly; near-camera geometry exhibits the standard re-projection
//     disocclusion artefacts.
//
// Output:
//   getColorCubeView() returns an RGBA16F sampler-ready cube view that
//   downstream consumers (glass OIT reflection, opaque IBL ambient) can
//   sample with a world-space direction.  The cubemap is conceptually
//   "centered on the main camera at the start of this frame" — sample it
//   the same way you'd sample a static IBL probe.
//
// Memory:
//   Two RGBA16F cube colour buffers (ping-pong) + one R32F cube depth
//   buffer at the configured edge size.  Default 512² ⇒ 6 MB colour ×2 +
//   3 MB depth = 15 MB total.
//
class DynamicCubemap {
public:
    // ── Per-face view matrix layout ─────────────────────────────────────
    // GLSL/Vulkan cubemap face order, matching VK_IMAGE_VIEW_TYPE_CUBE
    // layer indices (also matches how textureCube reads gl_Layer):
    //   0: +X   1: -X   2: +Y   3: -Y   2: +Z   5: -Z
    // CubeFaceView returns view * proj for face F at probe origin O.
    static constexpr uint32_t kNumFaces = 6;

    // Default edge size.  128² × 6 = ~400 KB per cube — small enough
    // that the per-frame face render (cluster_renderer opaque +
    // skydome at edge × edge) and the per-cycle SH projection both
    // stay cheap.  The downstream consumers (SH ambient + future
    // probe reprojection) only need low-frequency content, so a
    // larger resolution would be wasted work.  The constructor still
    // accepts an override if a high-resolution dynamic reflection
    // probe is ever needed.
    static constexpr uint32_t kDefaultEdge = 128;

    DynamicCubemap(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        // Cubemap render pass used for the per-face framebuffer
        // creation inside createCubemapTexture.  Not actually used by
        // the dynamic cubemap's own dispatch path (we drive the colour
        // cube as a storage image and the depth cube as a sampler),
        // but createCubemapTexture asserts on a non-null render_pass
        // when use_as_framebuffer is true.  The application should
        // pass its existing cubemap_render_pass_ here.
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        uint32_t edge = kDefaultEdge);

    // Drives the per-frame face render + reprojection.  Caller passes the
    // current main-camera world position as the probe origin and a
    // ClusterRenderer / Skydome pair that will be invoked to draw scene
    // content into the chosen face.  The cluster_desc_sets list is the
    // same descriptor-set list used for the main render path EXCEPT that
    // the VIEW_PARAMS_SET entry will be temporarily swapped to point at
    // this class's own per-face camera UBO so the cluster shaders see
    // the cube-face view+proj instead of the main camera's.
    //
    // The caller is responsible for providing a consistent set of input
    // descriptor sets across faces; lights, IBL, shadow maps etc. are
    // shared verbatim — only the camera info is overridden per face.
    void update(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::vec3& probe_pos,
        const std::shared_ptr<ClusterRenderer>& cluster_renderer,
        const std::shared_ptr<Skydome>& skydome,
        const renderer::DescriptorSetList& cluster_desc_sets);

    // Optional sky-envmap source for the per-frame face capture.  Set
    // once at startup with the IBL creator's envmap texture; on every
    // update() call the active face's content is copied from the
    // matching face of this envmap as a stand-in for full scene
    // capture.  Sky content is at infinity so it doesn't depend on
    // probe position — copying the envmap is functionally equivalent
    // to running the full skydome render at the probe origin per face,
    // at a fraction of the cost.  Pass null to disable sky capture
    // (the cubemap will then only carry whatever the reprojection pass
    // produces, useful for testing reprojection in isolation).
    inline void setSkyEnvmap(
        const std::shared_ptr<renderer::Image>& envmap_image,
        uint32_t envmap_edge) {
        sky_envmap_image_ = envmap_image;
        sky_envmap_edge_  = envmap_edge;
    }

    // Read access for downstream consumers.  view() returns a cube view
    // bound to the *current read* ping-pong buffer (the one not being
    // written this frame).
    inline const renderer::TextureInfo& getColorCube() const {
        return color_cubes_[current_read_idx_];
    }
    inline const std::shared_ptr<renderer::ImageView>& getColorCubeView() const {
        return color_cubes_[current_read_idx_].view;
    }
    inline const renderer::TextureInfo& getDepthCube() const {
        return depth_cube_;
    }

    // Diagnostic accessors used by the debug-viewer menu strip.
    inline uint32_t getEdge()           const { return edge_; }
    inline uint32_t getCurrentFace()    const { return current_face_; }
    inline uint64_t getFrameIndex()     const { return frame_index_; }
    inline int      getCurrentReadIdx() const { return current_read_idx_; }
    inline glm::vec3 getFaceCapturePos(uint32_t f) const {
        return face_capture_pos_[f];
    }

    // Per-face 2D layer views.  Used by the in-engine debug viewer to
    // register each face as an ImTextureID.  ping_pong_idx = 0 or 1 picks
    // which of the two ping-pong cube colour buffers is referenced; the
    // viewer should select whichever index `getCurrentReadIdx()` returns
    // at render time so the freshest content is shown.
    inline const std::shared_ptr<renderer::ImageView>& getColorFaceView(
        int ping_pong_idx, uint32_t face) const {
        return color_face_views_[ping_pong_idx][face];
    }

    // Swap the ping-pong read/write indices.  Caller (the ambient
    // probe system) invokes this at the end of each 6-face probe
    // cycle, after the SH projection has consumed the just-completed
    // write cube.  Post-swap, getColorCube{,View}() exposes the cube
    // that just finished being filled — guaranteed-consistent across
    // all 6 faces — and the OTHER cube becomes the next cycle's
    // write target.
    inline void swapPingPong() { current_read_idx_ ^= 1; }

    // Run box-filter mipgen on the current READ ping-pong cube
    // (color_cubes_[current_read_idx_]).  Should be called once per
    // probe cycle, immediately after swapPingPong(), so the just-
    // completed cube has a populated mip chain when sh_project.comp
    // (or any other consumer) samples via textureLod.  Mip 0 is left
    // unchanged; mips 1..N are 2:1 LINEAR-blit averages of mip 0.
    void generateMipsForRead(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void destroy(const std::shared_ptr<renderer::Device>& device);

private:
    // ── Resources ──────────────────────────────────────────────────────
    uint32_t edge_         = kDefaultEdge;
    uint32_t current_face_ = 0;     // round-robin face index this frame
    int      current_read_idx_ = 0; // which of color_cubes_[0/1] is the read
    uint64_t frame_index_  = 0;

    // Ping-pong RGBA16F cubes.  The "read" index is what consumers sample;
    // the "write" index is the destination of this frame's face render +
    // reprojection compute, then swapped at end of update().
    renderer::TextureInfo color_cubes_[2];

    // Per-face 2D image views (layer-sliced) for both ping-pong buffers,
    // used as the colour attachment for the per-face render pass.
    //   color_face_views_[ping_pong_idx][face_idx]
    std::shared_ptr<renderer::ImageView> color_face_views_[2][kNumFaces];

    // R32_SFLOAT depth cube.  Originally for the depth-aware reprojection
    // compute pass; currently unused by the active render path because
    // each probe captures all 6 faces over 6 consecutive frames at the
    // same origin, so reprojection isn't required.  Kept allocated to
    // preserve the descriptor-set wiring (the reproject pipeline still
    // references it).
    renderer::TextureInfo depth_cube_;
    std::shared_ptr<renderer::ImageView> depth_face_views_[kNumFaces];

    // Transient depth target for the per-face render pass.  Single 2D
    // D24S8 image — discarded between faces (we don't need depth
    // continuity across faces, since each face gets its own render
    // pass with CLEAR depth).  Format matches the application's
    // kForward renderbuffer depth format so the cluster_renderer's
    // bindless pipeline accepts it.
    renderer::TextureInfo face_depth_target_;

    // Per-face camera UBOs + descriptor sets.  Each frame the active
    // face's UBO is updated with view+proj matrices for the active
    // probe origin, and the matching descriptor set is substituted at
    // VIEW_PARAMS_SET in the cluster_renderer + skydome draw calls.
    renderer::BufferInfo face_camera_buffers_[kNumFaces];
    std::shared_ptr<renderer::DescriptorSet>
        face_view_desc_sets_[kNumFaces];

    // ── Depth-to-linear-distance compute pipeline ──────────────────────
    // Reads face_depth_target_ (D24S8 clip-space depth left by the face
    // render) and writes world-space distance from camera origin into
    // depth_cube_'s matching face slice (R32_SFLOAT).  Six pre-baked
    // descriptor sets (one per face), each binding face_depth_target_'s
    // view as the sampled input + the corresponding depth_face_views_[f]
    // as the storage-image output.
    std::shared_ptr<renderer::DescriptorSetLayout>
        depth_to_linear_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout>
        depth_to_linear_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>
        depth_to_linear_pipeline_;
    std::shared_ptr<renderer::DescriptorSet>
        depth_to_linear_desc_sets_[kNumFaces];
    std::shared_ptr<renderer::Sampler>
        depth_to_linear_depth_sampler_;

    // CPU-side capture position for each face: where the camera was when
    // that face was last freshly rendered.  Used by the reprojection
    // compute shader (uploaded via push constants or UBO each frame).
    glm::vec3 face_capture_pos_[kNumFaces] = {};

    // Per-face camera info UBO (one ViewCameraInfo per face) so any of
    // the 6 face renders can be driven by binding the matching slice.
    // Updated host-side at the start of each update() with the current
    // probe origin.
    renderer::BufferInfo face_camera_ubo_;

    // ── Reprojection compute pipeline ──────────────────────────────────
    std::shared_ptr<renderer::DescriptorSetLayout> reproject_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout>      reproject_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            reproject_pipeline_;

    // One descriptor set per (write_target_idx, face) — 2 × 6 = 12 sets.
    // Each binds the read cube as source + the write cube's per-face 2D
    // storage view as destination.
    std::shared_ptr<renderer::DescriptorSet>
        reproject_desc_sets_[2][kNumFaces];

    // Sampler shared with the reflection samplers — cube clamp/linear.
    std::shared_ptr<renderer::Sampler> sampler_;

    // Device cached for host-visible-memory mapping inside update().
    std::shared_ptr<renderer::Device> device_;

    // Optional sky envmap source for per-frame face capture.  If set,
    // the active face is overwritten each frame with a copy of the
    // matching face of this envmap (see setSkyEnvmap above for the
    // rationale on why this is sufficient for sky-only ambient).
    std::shared_ptr<renderer::Image> sky_envmap_image_;
    uint32_t sky_envmap_edge_ = 0;

    // ── Init helpers ────────────────────────────────────────────────────
    void createCubeResources(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass);
    void createFaceCameraResources(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);
    void createDepthToLinearPipeline(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);
    void createReprojectPipeline(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);
    void writeReprojectDescriptors(
        const std::shared_ptr<renderer::Device>& device);

    // Build the 6 view matrices for a probe at the given origin, using
    // the same handedness / Y-flip the main pipeline uses.
    static glm::mat4 cubeFaceView(uint32_t face, const glm::vec3& origin);
    // Right-handed perspective with reverse-Z disabled, 90° FOV, 1:1
    // aspect, near=0.1, far=1000.0 (matches the main pipeline's
    // GLM_FORCE_DEPTH_ZERO_TO_ONE convention).
    static glm::mat4 cubeFaceProj();

public:
    // Public helpers callable from AmbientProbeSystem to drive the
    // explicit per-face cluster cull right before the face render.
    static glm::mat4 cubeFaceViewProj(uint32_t face, const glm::vec3& origin) {
        return cubeFaceProj() * cubeFaceView(face, origin);
    }
};

}// namespace scene_rendering
}// namespace engine
