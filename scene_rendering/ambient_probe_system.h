#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class DynamicCubemap;
class Skydome;
class ClusterRenderer;

// ─── AmbientProbeSystem ──────────────────────────────────────────────────
// Real-time ambient lighting via temporally-amortised SH probes.
//
// HIGH-LEVEL FLOW:
//   1. At level load time, compute the world bounding box and place an
//      N×N×N grid of probe positions inside it.
//   2. One DynamicCubemap "scratch" probe (capacity for one cubemap +
//      depth, see scene_rendering/dynamic_cubemap.h) is moved to a
//      different grid probe each cycle.  It captures one cube face per
//      frame from the grid probe's world position; depth-aware
//      reprojection keeps the other 5 faces consistent as the scratch
//      cubemap moves.
//   3. After a probe's full cycle (6 frames), the scratch cubemap is
//      projected into 9 RGB SH coefficients via sh_project.comp and
//      written into that probe's slot in the SSBO.
//   4. Fragment shaders sample the 8 nearest probes (trilinear over the
//      grid) and evaluate SH for the surface normal, producing a
//      position-dependent diffuse ambient term.
//
// MEMORY:
//   Per probe:  vec4 position_active + vec4 sh[9] = 4 * (1 + 9) * 16 = 160 B
//   At 4³ = 64 probes:  ~10 KB.  Trivial.
//
// CYCLE TIMING:
//   With dither-style 1-face-per-frame capture, each probe takes 6 frames
//   to fully refresh.  At the default 4³ grid, full grid refresh is
//   64 * 6 = 384 frames ≈ 6.4s @ 60fps.  Faster grids (e.g. 2³ = 8
//   probes for tiny scenes) refresh in <1s.  For static-ish lighting
//   this is invisible; for fast-changing TOD it lags by half a cycle.
//
class AmbientProbeSystem {
public:
    // Shader-visible probe data layout — must match the GLSL struct in
    // shaders/ambient_probes.glsl.h exactly.  Aligned to 16 B for std430.
    struct ProbeData {
        glm::vec4 position;       // .xyz = world-space probe position
                                  // .w   = "fully baked" flag (>0 = ready)
        // 2nd-order spherical harmonics: 9 RGB coefficients = 9 vec4s
        // (the .w lane is unused — kept for std430 16-byte alignment).
        glm::vec4 sh[9];
    };

    static constexpr uint32_t kDefaultGridX = 4;
    static constexpr uint32_t kDefaultGridY = 4;
    static constexpr uint32_t kDefaultGridZ = 4;

    // Constructor allocates the SSBO + descriptor set + projection
    // pipeline.  The grid layout itself is empty until placeProbeGrid()
    // is called — the world bounding box has to come from outside,
    // since the full level isn't loaded at construction time.
    AmbientProbeSystem(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        uint32_t grid_x = kDefaultGridX,
        uint32_t grid_y = kDefaultGridY,
        uint32_t grid_z = kDefaultGridZ);

    // Place the probe grid inside `bbox_min..bbox_max`.  Probes are
    // distributed at uniform fractional positions along each axis so
    // that the corner probes sit at 1/(2N) inset from the bbox edges
    // — that places probes inside the level rather than on its very
    // surface, where they'd capture the inside of walls / sky behind.
    // Writes the positions through to the host-mapped probe SSBO and
    // resets every probe's SH coefficients + ready flag, so the next
    // updateOneProbe() cycle starts from a clean slate.
    void placeProbeGrid(
        const std::shared_ptr<renderer::Device>& device,
        const glm::vec3& bbox_min, const glm::vec3& bbox_max);

    // Per-frame entry point.  Drives the round-robin probe update:
    //   • Picks the active probe = (frame / 6) % num_probes.
    //   • Advances DynamicCubemap to that probe's world position.
    //   • Calls dynamic_cubemap_->update() so it captures one face this
    //     frame at the active probe's origin.
    //   • Once all 6 faces of the active probe are filled (every 6
    //     frames), runs sh_project.comp on the cubemap and writes the
    //     resulting 9 SH coefficients into the active probe's SSBO slot.
    void update(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<DynamicCubemap>& dynamic_cubemap,
        const std::shared_ptr<ClusterRenderer>& cluster_renderer,
        const std::shared_ptr<Skydome>& skydome,
        const renderer::DescriptorSetList& shared_desc_sets,
        const glm::vec3& camera_pos);

    // Build the debug-draw pipeline (one-time, after construction).
    // Caller passes the forward render pass's renderbuffer formats so
    // the pipeline matches the host pass it draws into.  Toggle the
    // visibility on per frame via the application + menu.
    void initDebugPipeline(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorSetLayout>&
            view_desc_set_layout,
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
        const renderer::GraphicPipelineInfo& pipeline_info);

    // Per-frame debug draw — instances `num_probes` icospheres at the
    // probe positions, colouring each by its SH-evaluated irradiance.
    // Call inside the application's forward dynamic-rendering pass
    // (LOAD/LOAD on color/depth) so the spheres show up on top of the
    // already-shaded scene.
    //
    // sphere_radius_world is the visual marker size in world units —
    // 0.3 m is a reasonable default for ~50-m-scale scenes.
    void drawDebug(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& shared_desc_sets,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors,
        float sphere_radius_world = 1.0f);

    // SSBO accessor for downstream consumers.  Bind this at the agreed
    // shader binding point along with the descriptor set layout below.
    inline const renderer::BufferInfo& getProbeBuffer() const {
        return probe_buffer_;
    }
    inline std::shared_ptr<renderer::DescriptorSetLayout>
        getProbeDescSetLayout() const { return probe_desc_set_layout_; }
    inline std::shared_ptr<renderer::DescriptorSet>
        getProbeDescSet() const { return probe_desc_set_; }

    // Layout / count accessors used by the fragment shader's grid lookup
    // and the debug visualiser.
    inline glm::vec3 getGridMin()    const { return grid_min_; }
    inline glm::vec3 getGridMax()    const { return grid_max_; }
    inline glm::uvec3 getGridDims()  const { return grid_dims_; }
    inline uint32_t   getProbeCount() const {
        return grid_dims_.x * grid_dims_.y * grid_dims_.z;
    }
    inline uint32_t   getCurrentProbe() const { return current_probe_; }
    inline uint32_t   getCurrentFace()  const { return current_face_; }
    inline uint64_t   getFrameIndex()   const { return frame_index_; }

    void destroy(const std::shared_ptr<renderer::Device>& device);

    // ── Swap-chain teardown hook ───────────────────────────────────
    // probe_desc_set_ and project_desc_set_ are allocated FROM the
    // application's main descriptor pool.  When the pool is destroyed
    // in cleanupSwapChain, those handles become dangling — calling
    // vkUpdateDescriptorSets against them (the per-frame
    // writeProjectDescriptorsForCube path) crashes inside the NVIDIA
    // driver dispatch.  Null the handles so consumers can't write to
    // them between pool destruction and recreate.
    void onDescriptorPoolDestroyed();

    // ── Re-allocate pool-owned descriptor sets from the fresh pool ──
    // Called from the application's recreateSwapChain after the new
    // descriptor pool has been built but before any consumer code
    // writes to a probe descriptor set.  Reallocates both sets and
    // re-issues their permanent SSBO bindings (probe_buffer_); the
    // cube-source bindings are refreshed every frame by
    // writeProjectDescriptorsForCube once these handles are alive.
    void recreateDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // ── Debug origin lock ──────────────────────────────────────────────
    // When enabled, ALL probes share the same world position (instead
    // of being distributed across the grid).  Useful for verifying the
    // face-render path: with the probe origin pinned at, say, the main
    // camera, the dynamic cubemap stops moving between frames and the
    // viewer shows a stable consistent cube.
    //
    // Pass `enable = false` (default) to restore the normal grid layout.
    inline void setLockedOrigin(bool enable, const glm::vec3& origin) {
        lock_origin_       = enable;
        locked_origin_     = origin;
    }
    inline bool isOriginLocked() const { return lock_origin_; }

    // Re-bind the source cubemap views that sh_project.comp samples.
    // Both the colour cube (current ping-pong read) and the linear-
    // distance cube (depth_cube_ from DynamicCubemap) are bound here.
    // Cheap (3 descriptor writes); the application calls this once
    // before each SH-projection dispatch in the per-frame update.
    void writeProjectDescriptorsForCube(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::ImageView>& color_cube_view,
        const std::shared_ptr<renderer::ImageView>& depth_cube_view);

private:
    // ── Grid layout ─────────────────────────────────────────────────────
    glm::uvec3 grid_dims_ = glm::uvec3(0);
    glm::vec3  grid_min_  = glm::vec3(0);
    glm::vec3  grid_max_  = glm::vec3(0);
    bool       grid_placed_ = false;

    // ── Update sequencer ────────────────────────────────────────────────
    uint64_t frame_index_   = 0;
    uint32_t current_probe_ = 0;
    uint32_t current_face_  = 0;

    // ── Debug origin lock ──────────────────────────────────────────────
    // setLockedOrigin sets these.  When `lock_origin_` is true the per-
    // frame probe_pos in update() becomes locked_origin_ regardless of
    // current_probe_'s grid coordinates.  All 64 probes share the same
    // world position, so the dynamic cubemap stays parked there.
    bool       lock_origin_   = false;
    glm::vec3  locked_origin_ = glm::vec3(0.0f);

    // ── Update interval (frames between face renders) ─────────────────
    // The cubemap face render + cluster cull is the dominant per-frame
    // cost of the probe system.  Only render one face every
    // update_interval_ application frames; the in-between frames just
    // skip the entire ambient_probe_system->update() body.  Default 4
    // means a full 6-face probe cycle takes 24 frames (~0.4 s @ 60 fps),
    // still smooth enough for slow-changing ambient lighting.
    int update_interval_ = 4;
    int update_phase_    = 0;      // counts modulo update_interval_

    // Camera-movement gating: when the camera hasn't moved more than
    // `motion_threshold_m_` since the last update, AND the current
    // 6-face cycle has already been completed at least once at this
    // location, we skip the entire update.  Ambient lighting doesn't
    // change when the player stands still, so skipping is free
    // visually.  Set the threshold to 0 to disable this gate.
    float     motion_threshold_m_ = 0.5f;
    glm::vec3 last_update_pos_    = glm::vec3(1e30f);
    bool      first_cycle_done_   = false;

public:
    inline int  getUpdateInterval() const { return update_interval_; }
    inline void setUpdateInterval(int n) {
        update_interval_ = (n < 1) ? 1 : n;
    }
    inline void setMotionThreshold(float meters) {
        motion_threshold_m_ = meters;
    }
    inline float getMotionThreshold() const { return motion_threshold_m_; }
private:

    // ── Probe SSBO ─────────────────────────────────────────────────────
    // ProbeData[N]; updated GPU-side by the SH projection compute pass,
    // and by host-side memcpy when placeProbeGrid() runs (only the
    // position field, then GPU writes start filling SH slots).
    renderer::BufferInfo probe_buffer_;

    // Descriptor set used by consumer fragment shaders to read the SSBO.
    std::shared_ptr<renderer::DescriptorSetLayout> probe_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>       probe_desc_set_;

    // ── SH projection compute pipeline ─────────────────────────────────
    // Reads the DynamicCubemap's color cube + writes 9 SH coefficients
    // into one probe slot of probe_buffer_.  Push constant carries the
    // target probe index.
    std::shared_ptr<renderer::DescriptorSetLayout> project_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout>      project_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            project_pipeline_;
    std::shared_ptr<renderer::DescriptorSet>       project_desc_set_;

    // Sampler reused for the cube read in the projection pass.
    std::shared_ptr<renderer::Sampler> sampler_;

    // ── Debug-draw graphics pipeline ───────────────────────────────────
    // Reuses the SH probe SSBO at descriptor set 0 (binding 0) and the
    // application's view-camera descriptor set at set VIEW_PARAMS_SET.
    // Procedural icosahedron — no vertex/index buffers.
    std::shared_ptr<renderer::PipelineLayout> debug_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>        debug_pipeline_;

    // ── Init helpers ────────────────────────────────────────────────────
    void createProbeBuffer(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);
    void createProjectPipeline(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Compute the world-space position of probe (i, j, k) in the placed
    // grid, with 1/(2N)-inset offsets so corner probes sit inside the
    // bounding box rather than on its faces.
    glm::vec3 probePosition(uint32_t i, uint32_t j, uint32_t k) const;
};

}// namespace scene_rendering
}// namespace engine
