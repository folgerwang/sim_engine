#include "ambient_probe_system.h"
#include "dynamic_cubemap.h"
#include "scene_rendering/cluster_renderer.h"

#include "renderer/renderer_helper.h"

namespace er = engine::renderer;

namespace engine {
namespace scene_rendering {

namespace {
// Push-constant block layout for sh_project.comp.  Keep in sync with
// the GLSL block at the top of that shader.
struct ShProjectPushConstants {
    glm::vec4 camera_pos_pad;   // .xyz = cubemap origin (camera)
    glm::vec4 probe_pos_pad;    // .xyz = active probe world position
    uint32_t  probe_idx;
    uint32_t  edge;
    float     pad0 = 0.0f;
    float     pad1 = 0.0f;
};
constexpr uint32_t kShProjectSrcBinding    = 0;
constexpr uint32_t kShProjectProbesBinding = 1;
constexpr uint32_t kShProjectDepthBinding  = 2;
}// namespace

// ─── Construction ───────────────────────────────────────────────────────
AmbientProbeSystem::AmbientProbeSystem(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z)
    : grid_dims_(grid_x, grid_y, grid_z), sampler_(texture_sampler) {
    createProbeBuffer(device, descriptor_pool);
    createProjectPipeline(device, descriptor_pool);
}

// ─── Probe buffer + consumer descriptor set ─────────────────────────────
void AmbientProbeSystem::createProbeBuffer(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) {
    const uint32_t n = getProbeCount();
    const VkDeviceSize buffer_size = sizeof(ProbeData) * n;

    // Host-visible | device-local — we want CPU writes (positions) to
    // land directly without staging, AND GPU reads/writes (SH baking)
    // to be fast.  HOST_COHERENT means we don't need explicit flushes.
    device->createBuffer(
        buffer_size,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        0,
        probe_buffer_.buffer,
        probe_buffer_.memory,
        std::source_location::current());

    // Initialise: every probe at origin, "not ready" flag = 0,
    // SH coefficients = 0.  placeProbeGrid() will overwrite positions
    // when it's called; SH coefficients stay zero until baked.
    std::vector<ProbeData> init(n);
    for (auto& p : init) {
        p.position = glm::vec4(0.0f);
        for (auto& c : p.sh) c = glm::vec4(0.0f);
    }
    void* mapped = device->mapMemory(probe_buffer_.memory, buffer_size, 0);
    if (mapped) {
        std::memcpy(mapped, init.data(), buffer_size);
        device->unmapMemory(probe_buffer_.memory);
    }

    // Consumer descriptor set: STORAGE_BUFFER at binding 0 of a
    // dedicated set.  Cluster bindless / forward shaders include the
    // matching layout via ambient_probes.glsl.h.
    std::vector<er::DescriptorSetLayoutBinding> bindings(1);
    bindings[0].binding         = 0;
    bindings[0].descriptor_count = 1;
    bindings[0].descriptor_type = er::DescriptorType::STORAGE_BUFFER;
    // VERTEX_BIT: required by probe_debug.vert (it reads probe
    // positions out of the SSBO to place each instanced icosphere).
    // FRAGMENT_BIT: required by probe_debug.frag (it reads SH
    // coefficients for the irradiance evaluation) AND by the future
    // hook in cluster_bindless.frag's USE_AMBIENT_PROBES path.
    // COMPUTE_BIT: required by sh_project.comp (it writes coefficients).
    bindings[0].stage_flags = SET_3_FLAG_BITS(ShaderStage,
        VERTEX_BIT, FRAGMENT_BIT, COMPUTE_BIT);
    probe_desc_set_layout_ = device->createDescriptorSetLayout(bindings);

    probe_desc_set_ = device->createDescriptorSets(
        descriptor_pool, probe_desc_set_layout_, 1)[0];

    er::WriteDescriptorList writes;
    er::Helper::addOneBuffer(writes, probe_desc_set_,
        er::DescriptorType::STORAGE_BUFFER,
        /*binding*/ 0,
        probe_buffer_.buffer,
        static_cast<uint32_t>(buffer_size));
    device->updateDescriptorSets(writes);
}

// ─── Projection compute pipeline ────────────────────────────────────────
void AmbientProbeSystem::createProjectPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        kShProjectSrcBinding,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1].binding         = kShProjectProbesBinding;
    bindings[1].descriptor_count = 1;
    bindings[1].descriptor_type = er::DescriptorType::STORAGE_BUFFER;
    bindings[1].stage_flags     = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    // Linear-distance cube — used by sh_project.comp for parallax-aware
    // per-probe reprojection.  Same shape (samplerCube) as src_color.
    bindings[2] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        kShProjectDepthBinding,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    project_desc_set_layout_ = device->createDescriptorSetLayout(bindings);

    er::PushConstantRange pcr;
    pcr.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    pcr.offset = 0;
    pcr.size   = sizeof(ShProjectPushConstants);
    project_pipeline_layout_ = device->createPipelineLayout(
        { project_desc_set_layout_ }, { pcr },
        std::source_location::current());

    auto cs = er::helper::loadShaderModule(
        device, "sh_project_comp.spv",
        er::ShaderStageFlagBits::COMPUTE_BIT,
        std::source_location::current());
    project_pipeline_ = device->createPipeline(
        project_pipeline_layout_, cs,
        std::source_location::current());

    project_desc_set_ = device->createDescriptorSets(
        descriptor_pool, project_desc_set_layout_, 1)[0];

    // Project descriptor set's SSBO binding is fixed (probe_buffer_);
    // the cube-source binding gets rewritten each time the dynamic
    // cubemap's read index swaps (writeProjectDescriptorsForCube).
    er::WriteDescriptorList writes;
    er::Helper::addOneBuffer(writes, project_desc_set_,
        er::DescriptorType::STORAGE_BUFFER,
        kShProjectProbesBinding,
        probe_buffer_.buffer,
        static_cast<uint32_t>(sizeof(ProbeData) * getProbeCount()));
    device->updateDescriptorSets(writes);
}

void AmbientProbeSystem::writeProjectDescriptorsForCube(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::ImageView>& color_cube_view,
    const std::shared_ptr<er::ImageView>& depth_cube_view) {
    er::WriteDescriptorList writes;
    er::Helper::addOneTexture(writes, project_desc_set_,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        kShProjectSrcBinding,
        sampler_, color_cube_view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    if (depth_cube_view) {
        er::Helper::addOneTexture(writes, project_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            kShProjectDepthBinding,
            sampler_, depth_cube_view,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
    device->updateDescriptorSets(writes);
}

// ─── Probe placement ────────────────────────────────────────────────────
glm::vec3 AmbientProbeSystem::probePosition(
    uint32_t i, uint32_t j, uint32_t k) const {
    // 1/(2N)-inset: probe at coordinate i ∈ [0, N) sits at
    //     min + (i + 0.5) / N · (max - min)
    // — i.e. the corner probes are inset by half a cell from the bbox
    // edges.  Avoids placing probes literally on / outside walls.
    glm::vec3 t(
        (float(i) + 0.5f) / float(grid_dims_.x),
        (float(j) + 0.5f) / float(grid_dims_.y),
        (float(k) + 0.5f) / float(grid_dims_.z));
    return glm::mix(grid_min_, grid_max_, t);
}

void AmbientProbeSystem::placeProbeGrid(
    const std::shared_ptr<er::Device>& device,
    const glm::vec3& bbox_min, const glm::vec3& bbox_max) {
    grid_min_ = bbox_min;
    grid_max_ = bbox_max;
    grid_placed_ = true;

    const uint32_t n = getProbeCount();
    std::vector<ProbeData> probes(n);
    for (uint32_t k = 0; k < grid_dims_.z; ++k) {
        for (uint32_t j = 0; j < grid_dims_.y; ++j) {
            for (uint32_t i = 0; i < grid_dims_.x; ++i) {
                const uint32_t idx =
                    (k * grid_dims_.y + j) * grid_dims_.x + i;
                probes[idx].position =
                    glm::vec4(probePosition(i, j, k), 0.0f);
                for (auto& c : probes[idx].sh) c = glm::vec4(0.0f);
            }
        }
    }

    // SAFE because probe_buffer_ is HOST_VISIBLE | HOST_COHERENT.  We're
    // overwriting the entire SSBO contents — no need to interleave with
    // GPU work since this happens during level load before any GPU
    // dispatches run against the buffer.
    const VkDeviceSize buffer_size = sizeof(ProbeData) * n;
    void* mapped = device->mapMemory(probe_buffer_.memory, buffer_size, 0);
    if (mapped) {
        std::memcpy(mapped, probes.data(), buffer_size);
        device->unmapMemory(probe_buffer_.memory);
    }

    // Reset sequencer so the first updateOneProbe() cycle starts at
    // probe 0 / face 0.
    frame_index_   = 0;
    current_probe_ = 0;
    current_face_  = 0;
}

// ─── Per-frame update ───────────────────────────────────────────────────
void AmbientProbeSystem::update(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const std::shared_ptr<DynamicCubemap>& dynamic_cubemap,
    const std::shared_ptr<ClusterRenderer>& cluster_renderer,
    const std::shared_ptr<Skydome>& skydome,
    const er::DescriptorSetList& shared_desc_sets,
    const glm::vec3& camera_pos) {
    if (!grid_placed_ || !dynamic_cubemap) return;

    const uint32_t n = getProbeCount();
    if (n == 0) return;

    // ── Update-interval gate ────────────────────────────────────────
    // Skip the entire probe pass on (update_interval_ - 1) of every
    // update_interval_ application frames.  The face render +
    // cluster cull are the dominant cost; the rest of the frame
    // (main render, UI, etc.) keeps running at full rate.  When we
    // skip, we don't increment frame_index_ either — that means the
    // 6-face cycle spans (6 * update_interval_) application frames,
    // and the SH projection at end-of-cycle still produces a fully
    // consistent cube.
    update_phase_ = (update_phase_ + 1) % update_interval_;
    if (update_phase_ != 0) return;

    // ── Camera-stationary gate ──────────────────────────────────────
    // Once we've finished at least one full 6-face cycle at the
    // current camera location AND the camera hasn't moved further
    // than the motion threshold since, the cube is already a valid
    // capture at this position.  Skip the update entirely.  As soon
    // as the camera moves past the threshold, fall back into the
    // full update path and re-run a 6-face cycle to refresh.
    if (motion_threshold_m_ > 0.0f && first_cycle_done_) {
        float d2 = glm::dot(camera_pos - last_update_pos_,
                             camera_pos - last_update_pos_);
        if (d2 < motion_threshold_m_ * motion_threshold_m_) {
            return;
        }
    }
    last_update_pos_ = camera_pos;

    // ── Architecture (revised) ──────────────────────────────────────
    // The dynamic cubemap is a SINGLE camera-centered probe — it never
    // moves between grid positions any more.  One face is rendered per
    // frame at the main camera's location; after the 6-face cycle
    // completes, the cubemap holds a coherent 360° view of the world
    // around the camera.
    //
    // The grid probes (placed at level-load time inside the level
    // bounding box) are then updated FROM THIS CAMERA CUBEMAP via
    // depth-aware reprojection: each probe's stored data is the result
    // of warping the camera's cubemap to that probe's position using
    // the captured depth.
    //
    // This update() function therefore:
    //   • Always feeds the camera position into dynamic_cubemap->update.
    //   • At the end of each 6-face cycle, swaps the cubemap's
    //     ping-pong (so the read view becomes the just-completed cube)
    //     and runs probe reprojection + SH projection per grid probe.
    //
    // current_probe_ + current_face_ are kept as diagnostic counters
    // (the viewer + frame_index_ display use them) — they don't gate
    // the work any more.
    current_probe_ = static_cast<uint32_t>((frame_index_ / 6) % n);
    current_face_  = static_cast<uint32_t>(frame_index_ % 6);

    // Camera origin.  The locked-origin override is the standard path
    // now; the application sets it to main_camera_object_->getCameraPosition()
    // each frame so the cubemap follows the player.
    glm::vec3 probe_pos = lock_origin_ ? locked_origin_ : glm::vec3(0.0f);

    // ── Per-face cluster cull ──────────────────────────────────────
    // Run cluster_renderer's cull dispatch with the active face's
    // view-projection BEFORE handing control to dynamic_cubemap->
    // update().  This rebuilds indirect_draw_buffer_ with clusters
    // visible from this specific face's 90° FOV at the camera origin
    // — without it the indirect set is whatever the previous frame
    // (or the main camera) populated, and clusters off the main
    // camera's frustum would be missing from the cube face.
    //
    // dynamic_cubemap->update() then issues the face render into the
    // active face slice, consuming the just-culled indirect set.
    if (cluster_renderer) {
        const uint32_t face = static_cast<uint32_t>(frame_index_ % 6);
        const glm::mat4 face_vp =
            DynamicCubemap::cubeFaceViewProj(face, probe_pos);
        // Force-disable Hi-Z occlusion for probe-face culls — the
        // pyramid is built from the main camera's depth and would
        // wildly mis-cull a cubemap face viewed from a different
        // origin / direction.  last_view_proj_ is unused when the
        // override forces the test off.
        cluster_renderer->cull(
            cmd_buf, face_vp, probe_pos,
            /*last_view_proj*/ face_vp,
            /*hiz_cull_override*/ std::optional<bool>(false));
    }

    // Drive the dynamic cubemap.  It renders face = (frame % 6) into
    // the WRITE-side ping-pong buffer (current_read_idx ^ 1 inside
    // DynamicCubemap).  Consumers reading getColorCubeView() see the
    // OTHER (last-completed) buffer until we swap below.
    dynamic_cubemap->update(
        cmd_buf, probe_pos,
        cluster_renderer, skydome, shared_desc_sets);

    // At the end of each 6-face cycle the camera cubemap holds a
    // complete 360° capture at the camera position.  We:
    //   1. Swap the ping-pong indices so the just-completed cube
    //      becomes the "read" buffer that consumers see.
    //   2. Re-bind the SH projection's source-cube descriptor to the
    //      newly-exposed read view.
    //   3. For each grid probe in the level, run per-probe SH
    //      projection on the cubemap.
    //
    // ── TODO: depth-aware probe reprojection ────────────────────────
    // The CORRECT per-probe path is to reproject the camera cubemap
    // from camera_pos to the probe's grid position using the captured
    // depth, THEN run SH projection on that reprojected cube.  This
    // requires:
    //   (a) Linear-distance depth in dynamic_cubemap's depth_cube_.
    //       Currently the face render writes D24S8 to a transient
    //       depth target; needs a small compute pass that converts
    //       that to R32F linear distance and stores it in depth_cube_.
    //   (b) Per-probe scratch cubemap (or in-shader parallax sampling)
    //       to hold the reprojected view.
    //   (c) Updated SH projection that takes a probe-origin push
    //       constant and uses it during sampling.
    //
    // For now (placeholder until those land), every probe gets the
    // SAME SH — the camera-centered SH.  This means probes don't yet
    // produce position-distinct ambient, but the SH update path runs
    // end-to-end so consumers can already start sampling.
    const bool finished_probe = (frame_index_ % 6) == 5;
    if (finished_probe) {
        first_cycle_done_ = true;
        dynamic_cubemap->swapPingPong();
        dynamic_cubemap->generateMipsForRead(cmd_buf);
        // Bind both the colour cube (just-completed read view) AND
        // the linear-distance cube — sh_project.comp reads both for
        // its parallax-aware per-probe sampling.
        writeProjectDescriptorsForCube(
            device,
            dynamic_cubemap->getColorCubeView(),
            dynamic_cubemap->getDepthCube().view);

        cmd_buf->bindPipeline(
            er::PipelineBindPoint::COMPUTE, project_pipeline_);
        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            project_pipeline_layout_,
            { project_desc_set_ });

        // Per-probe SH projection.  Each dispatch's push-constant
        // probe_pos drives parallax-aware sampling inside the shader,
        // so each probe ends up with a distinct SH integral matching
        // the radiance it would observe from its own grid position.
        const uint32_t edge_px = dynamic_cubemap->getEdge();
        for (uint32_t p = 0; p < n; ++p) {
            uint32_t pi = p % grid_dims_.x;
            uint32_t pj = (p / grid_dims_.x) % grid_dims_.y;
            uint32_t pk = p / (grid_dims_.x * grid_dims_.y);
            glm::vec3 ppos = probePosition(pi, pj, pk);

            ShProjectPushConstants pc{};
            pc.camera_pos_pad = glm::vec4(camera_pos, 0.0f);
            pc.probe_pos_pad  = glm::vec4(ppos,       0.0f);
            pc.probe_idx      = p;
            pc.edge           = edge_px;
            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                project_pipeline_layout_,
                &pc, sizeof(pc));
            cmd_buf->dispatch(1, 1, 1);
        }
    }

    ++frame_index_;
}

// ─── Debug-draw pipeline ────────────────────────────────────────────────
// Per-probe icosphere visualiser.  Geometry is generated in
// probe_debug.vert from a hardcoded 12-vertex icosahedron; the fragment
// shader evaluates the probe's SH coefficients in the surface-normal
// direction so each sphere shows the irradiance it would contribute.
//
// Pipeline layout:
//   set 0 — probe SSBO (this class's probe_desc_set_layout_)
//   set 1 — view-camera (ego::CameraObject's layout)
//   push constant — { float radius; float pad[3]; }
void AmbientProbeSystem::initDebugPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& view_desc_set_layout,
    const er::PipelineRenderbufferFormats& renderbuffer_formats,
    const er::GraphicPipelineInfo& pipeline_info) {
    er::PushConstantRange pcr;
    pcr.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    pcr.offset = 0;
    // 4 floats = 16 B (just `radius`, padded for std430).
    pcr.size = sizeof(float) * 4;

    debug_pipeline_layout_ = device->createPipelineLayout(
        { probe_desc_set_layout_, view_desc_set_layout },
        { pcr },
        std::source_location::current());

    er::ShaderModuleList shader_modules(2);
    shader_modules[0] = er::helper::loadShaderModule(
        device, "probe_debug_vert.spv",
        er::ShaderStageFlagBits::VERTEX_BIT,
        std::source_location::current());
    shader_modules[1] = er::helper::loadShaderModule(
        device, "probe_debug_frag.spv",
        er::ShaderStageFlagBits::FRAGMENT_BIT,
        std::source_location::current());

    er::PipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    // Override raster state to disable backface culling — the
    // hard-coded icosahedron index list in probe_debug.vert isn't
    // guaranteed to match the application's chosen front-face winding,
    // so culling could silently drop every triangle and the spheres
    // would never appear.  cull=NONE costs nothing for 60 verts × 64
    // probes and avoids the orientation guesswork.
    er::GraphicPipelineInfo dbg_info = pipeline_info;
    dbg_info.rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo(
                false, false,
                er::PolygonMode::FILL,
                SET_FLAG_BIT(CullMode, NONE)));

    debug_pipeline_ = device->createPipeline(
        debug_pipeline_layout_,
        /*vertex_input_bindings*/   {},
        /*vertex_input_attributes*/ {},
        input_assembly,
        dbg_info,
        shader_modules,
        renderbuffer_formats,
        er::RasterizationStateOverride{},
        std::source_location::current());
}

void AmbientProbeSystem::drawDebug(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const er::DescriptorSetList& shared_desc_sets,
    const std::vector<er::Viewport>& viewports,
    const std::vector<er::Scissor>& scissors,
    float sphere_radius_world) {
    if (!debug_pipeline_ || !grid_placed_) return;
    const uint32_t n = getProbeCount();
    if (n == 0) return;

    // Pull view set out of the caller's shared list.  shared_desc_sets
    // is the same list passed to update() — we only need the view
    // camera at index VIEW_PARAMS_SET.
    if (shared_desc_sets.size() <= VIEW_PARAMS_SET) return;
    auto view_set = shared_desc_sets[VIEW_PARAMS_SET];
    if (!view_set) return;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, debug_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        debug_pipeline_layout_,
        { probe_desc_set_, view_set });

    float pc[4] = { sphere_radius_world, 0, 0, 0 };
    cmd_buf->pushConstants(
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
        debug_pipeline_layout_,
        pc, sizeof(pc));

    // 240 vertex outputs per probe instance: a level-1 subdivided
    // icosphere = 20 base triangles × 4 sub-triangles × 3 vertices.
    // Geometry is fully procedural in probe_debug.vert (no vertex/
    // index buffer); the shader decodes gl_VertexIndex into
    // (parent_face, sub_triangle, corner) and computes the world-
    // sphere position from the canonical icosahedron vertex set plus
    // edge-midpoint subdivision.
    cmd_buf->draw(240, n);
}

// ─── Cleanup ────────────────────────────────────────────────────────────
void AmbientProbeSystem::destroy(const std::shared_ptr<er::Device>& device) {
    debug_pipeline_.reset();
    debug_pipeline_layout_.reset();
    project_pipeline_.reset();
    project_pipeline_layout_.reset();
    project_desc_set_layout_.reset();
    project_desc_set_.reset();
    probe_desc_set_.reset();
    probe_desc_set_layout_.reset();
    if (probe_buffer_.buffer)  probe_buffer_.destroy(device);
}

}// namespace scene_rendering
}// namespace engine
