#include "dynamic_cubemap.h"

#include "renderer/renderer_helper.h"
#include "renderer/vulkan/vk_renderer_helper.h"
#include "shaders/global_definition.glsl.h"
#include "scene_rendering/cluster_renderer.h"
#include "scene_rendering/skydome.h"
#include "game_object/camera_object.h"

namespace er = engine::renderer;

namespace engine {
namespace scene_rendering {

// ─── Constants ──────────────────────────────────────────────────────────
namespace {
// Match the application's forward-pass color format so the cluster
// bindless pipeline (and the OIT composite + sky envmap pipelines)
// can render into our cube face slices without format-mismatch
// validation errors.  See application.cpp::initVulkan: renderbuffer_
// formats_[kForward].color_formats[0].
constexpr er::Format kColorFormat = er::Format::B10G11R11_UFLOAT_PACK32;
// face_depth_target_ format — must match the forward pass so the
// cluster pipeline accepts it as a depth attachment.
constexpr er::Format kFaceDepthFormat = er::Format::D24_UNORM_S8_UINT;
// depth_cube_ format — single channel float, holds linear world-space
// distance from camera origin to the visible surface in each cube
// direction.  Written by depth_to_linear.comp after each face render
// and consumed by sh_project.comp for parallax-aware probe sampling.
constexpr er::Format kLinearDistanceFormat = er::Format::R32_SFLOAT;

// Push-constants layout for cube_reproject.comp.  Mirrors the GLSL block
// in cube_reproject.comp — keep the two definitions in sync.
struct ReprojectPushConstants {
    glm::vec3 src_origin;
    float     pad0 = 0.0f;
    glm::vec3 dst_origin;
    float     pad1 = 0.0f;
    int32_t   face_idx = 0;
    int32_t   edge     = 0;
    float     far_plane = 1000.0f;
    float     pad2 = 0.0f;
};

// Push-constants for depth_to_linear.comp.  Mirrors the GLSL block.
struct DepthToLinearPushConstants {
    glm::mat4 inv_view_proj;
    glm::vec4 camera_pos_pad;   // xyz = camera world position, w unused
    int32_t   edge = 0;
    int32_t   pad0 = 0;
    int32_t   pad1 = 0;
    int32_t   pad2 = 0;
};

// Reprojection compute descriptor-set bindings (see cube_reproject.comp).
constexpr uint32_t kReprojSrcColorBinding = 0;
constexpr uint32_t kReprojSrcDepthBinding = 1;
constexpr uint32_t kReprojDstColorBinding = 2;

constexpr uint32_t kComputeWorkgroupEdge = 8;

// 6 standard cube-face look-at directions for view matrices.
// Matches the same convention the cube_reproject.comp shader uses for
// face index → direction mapping (Vulkan/GL cubemap layer order).
struct FaceAxes {
    glm::vec3 forward;
    glm::vec3 up;
};
constexpr FaceAxes kFaceAxes[6] = {
    { { 1, 0,  0}, {0, -1,  0} },  // +X
    { {-1, 0,  0}, {0, -1,  0} },  // -X
    { { 0, 1,  0}, {0,  0,  1} },  // +Y
    { { 0,-1,  0}, {0,  0, -1} },  // -Y
    { { 0, 0,  1}, {0, -1,  0} },  // +Z
    { { 0, 0, -1}, {0, -1,  0} },  // -Z
};
}// namespace

// ─── ctor / init ────────────────────────────────────────────────────────
DynamicCubemap::DynamicCubemap(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::RenderPass>& cube_render_pass,
    uint32_t edge)
    : edge_(edge), sampler_(texture_sampler), device_(device) {
    createCubeResources(device, cube_render_pass);
    createFaceCameraResources(device, descriptor_pool);
    createDepthToLinearPipeline(device, descriptor_pool);
    createReprojectPipeline(device, descriptor_pool);
    writeReprojectDescriptors(device);
}

// ─── Resource creation ──────────────────────────────────────────────────
void DynamicCubemap::createCubeResources(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::RenderPass>& cube_render_pass) {
    // Two B10G11R11 cube color buffers (ping-pong) with a full mip
    // chain.  Mip 0 is what the per-face render pass writes into;
    // mips 1..N are box-filter mipgen'd (vkCmdBlitImage 2:1 LINEAR)
    // at end-of-cycle and consumed by sh_project.comp via textureLod
    // — sampling at mip log2(edge/grid) gives each SH grid cell a
    // hardware-box-filtered average of (edge/grid)² mip-0 texels,
    // which both improves the SH integral's accuracy and lets us use
    // a much smaller grid (e.g. 4 × 4 × 6 = 96 samples) without
    // aliasing.
    const uint32_t mip_count =
        static_cast<uint32_t>(std::log2(edge_) + 1);
    std::vector<er::BufferImageCopyInfo> empty_copies;
    for (int i = 0; i < 2; ++i) {
        er::Helper::createCubemapTexture(
            device,
            cube_render_pass,
            edge_, edge_, mip_count,
            kColorFormat,
            empty_copies,
            color_cubes_[i],
            std::source_location::current());
    }

    // R32_SFLOAT linear-distance cube — written by depth_to_linear.comp
    // after each face render with the world-space distance from the
    // camera origin to the surface visible at each pixel.  Used by
    // sh_project.comp for parallax-aware per-probe sampling: at each
    // sample direction d, the probe walks a distance distance_cube(d)
    // from the camera origin to find the corresponding 3D world point,
    // then reconstructs the colour the probe would see from that point.
    er::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        edge_, edge_, /*mip_count*/ 1,
        kLinearDistanceFormat,
        empty_copies,
        depth_cube_,
        std::source_location::current());

    // Per-face 2D layer views for both color cubes — used as colour
    // attachments by the per-face render pass and as storage targets
    // by the reprojection compute pass.
    for (int p = 0; p < 2; ++p) {
        for (uint32_t f = 0; f < kNumFaces; ++f) {
            color_face_views_[p][f] = device->createImageView(
                color_cubes_[p].image,
                er::ImageViewType::VIEW_2D,
                kColorFormat,
                SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                std::source_location::current(),
                /*baseMip*/ 0, /*mipCount*/ 1,
                /*baseLayer*/ f, /*layerCount*/ 1);
        }
    }
    // Per-face R32F storage views into depth_cube_ — used as the write
    // target by depth_to_linear.comp (one face per dispatch, post each
    // face render pass).
    for (uint32_t f = 0; f < kNumFaces; ++f) {
        depth_face_views_[f] = device->createImageView(
            depth_cube_.image,
            er::ImageViewType::VIEW_2D,
            kLinearDistanceFormat,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current(),
            0, 1, f, 1);
    }

    // ── Transient depth target for per-face render pass ─────────────
    // Allocated as a 2D D24S8 image at edge_ × edge_.  Reused for
    // every face: each face render pass uses CLEAR depth on entry, so
    // the contents don't need to persist between faces.  Format
    // matches the application's kForward renderbuffer depth format
    // so the cluster_renderer's bindless pipeline accepts it.
    face_depth_target_.image = device->createImage(
        er::ImageType::TYPE_2D,
        glm::uvec3(edge_, edge_, 1),
        kFaceDepthFormat,
        // SAMPLED_BIT lets depth_to_linear.comp read clip-space depth
        // out of this image after the face render pass completes.
        SET_2_FLAG_BITS(ImageUsage,
            DEPTH_STENCIL_ATTACHMENT_BIT, SAMPLED_BIT),
        er::ImageTiling::OPTIMAL,
        er::ImageLayout::UNDEFINED,
        std::source_location::current(),
        0, false, 1, 1, 1);
    {
        auto req = device->getImageMemoryRequirements(face_depth_target_.image);
        face_depth_target_.memory = device->allocateMemory(
            req.size, req.memory_type_bits,
            er::vk::helper::toVkMemoryPropertyFlags(
                SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT)),
            0);
        device->bindImageMemory(
            face_depth_target_.image, face_depth_target_.memory);
        face_depth_target_.view = device->createImageView(
            face_depth_target_.image,
            er::ImageViewType::VIEW_2D,
            kFaceDepthFormat,
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
            std::source_location::current(),
            0, 1, 0, 1);
    }

    // Initialise capture positions to the origin so the first reprojection
    // pass produces something sensible (nothing yet captured ⇒ source ==
    // dst, reprojection is identity).
    for (uint32_t f = 0; f < kNumFaces; ++f) {
        face_capture_pos_[f] = glm::vec3(0.0f);
    }
}

// ─── Per-face camera UBO + descriptor sets ──────────────────────────────
// One small storage buffer + one descriptor set per face.  The
// descriptor sets are allocated against the same layout the application
// uses for its main camera (CameraObject::getViewCameraDescriptorSetLayout)
// so they are drop-in substitutes for the cluster_renderer and skydome
// pipelines' VIEW_PARAMS_SET binding.  Each frame the active face's
// buffer is updated host-side with the face's view+proj at the active
// probe origin; the matching descriptor set is then bound in place of
// the main camera's set during the per-face render pass.
void DynamicCubemap::createFaceCameraResources(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) {
    const auto layout =
        engine::game_object::CameraObject::getViewCameraDescriptorSetLayout();

    for (uint32_t f = 0; f < kNumFaces; ++f) {
        device->createBuffer(
            sizeof(glsl::ViewCameraInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
            SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
            0,
            face_camera_buffers_[f].buffer,
            face_camera_buffers_[f].memory,
            std::source_location::current());

        face_view_desc_sets_[f] = device->createDescriptorSets(
            descriptor_pool, layout, 1)[0];

        er::WriteDescriptorList writes;
        er::Helper::addOneBuffer(writes, face_view_desc_sets_[f],
            er::DescriptorType::STORAGE_BUFFER,
            VIEW_CAMERA_BUFFER_INDEX,
            face_camera_buffers_[f].buffer,
            sizeof(glsl::ViewCameraInfo));
        device->updateDescriptorSets(writes);
    }
}

// ─── Depth → linear-distance pipeline ──────────────────────────────────
// Reads face_depth_target_ (D24S8) via a sampler and writes per-pixel
// linear distance from camera origin into one face slice of depth_cube_.
// Six descriptor sets are pre-baked at startup, one per face — they
// share the same depth-source binding (face_depth_target_) but each
// targets a different output face slice.
void DynamicCubemap::createDepthToLinearPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) {
    // Dedicated sampler for depth reads — NEAREST so we get raw clip-
    // depth bits without bilinear leak between texels.
    depth_to_linear_depth_sampler_ = device->createSampler(
        er::Filter::NEAREST,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::NEAREST,
        1.0f,
        std::source_location::current());

    // Layout: binding 0 = depth sampler, binding 1 = output face image2D.
    std::vector<er::DescriptorSetLayoutBinding> bindings(2);
    bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        0, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        1, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::STORAGE_IMAGE);
    depth_to_linear_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    er::PushConstantRange pcr;
    pcr.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    pcr.offset = 0;
    pcr.size   = sizeof(DepthToLinearPushConstants);
    depth_to_linear_pipeline_layout_ = device->createPipelineLayout(
        { depth_to_linear_desc_set_layout_ }, { pcr },
        std::source_location::current());

    auto cs = er::helper::loadShaderModule(
        device, "depth_to_linear_comp.spv",
        er::ShaderStageFlagBits::COMPUTE_BIT,
        std::source_location::current());
    depth_to_linear_pipeline_ = device->createPipeline(
        depth_to_linear_pipeline_layout_, cs,
        std::source_location::current());

    // 6 descriptor sets, one per output face slice.
    for (uint32_t f = 0; f < kNumFaces; ++f) {
        depth_to_linear_desc_sets_[f] = device->createDescriptorSets(
            descriptor_pool, depth_to_linear_desc_set_layout_, 1)[0];

        er::WriteDescriptorList writes;
        // Source: face_depth_target_ (sampled).  Same image for all 6
        // descriptor sets — face_depth_target_ is reused per face.
        er::Helper::addOneTexture(writes, depth_to_linear_desc_sets_[f],
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            /*binding*/ 0,
            depth_to_linear_depth_sampler_,
            face_depth_target_.view,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        // Destination: depth_cube_'s face f slice as a storage image.
        er::Helper::addOneTexture(writes, depth_to_linear_desc_sets_[f],
            er::DescriptorType::STORAGE_IMAGE,
            /*binding*/ 1,
            /*sampler*/ nullptr,
            depth_face_views_[f],
            er::ImageLayout::GENERAL);
        device->updateDescriptorSets(writes);
    }
}

// ─── Reprojection compute pipeline ──────────────────────────────────────
void DynamicCubemap::createReprojectPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) {
    // Descriptor layout:
    //   binding 0: samplerCube  src_color  (read)
    //   binding 1: samplerCube  src_depth  (read)
    //   binding 2: image2D      dst_color  (write, single face slice)
    std::vector<er::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        kReprojSrcColorBinding,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        kReprojSrcDepthBinding,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[2] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        kReprojDstColorBinding,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::STORAGE_IMAGE);
    reproject_desc_set_layout_ = device->createDescriptorSetLayout(bindings);

    // Pipeline layout with one push-constant range for ReprojectPushConstants.
    er::PushConstantRange pcr;
    pcr.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    pcr.offset = 0;
    pcr.size   = sizeof(ReprojectPushConstants);
    reproject_pipeline_layout_ = device->createPipelineLayout(
        { reproject_desc_set_layout_ }, { pcr },
        std::source_location::current());

    // Compute pipeline.
    auto cs = er::helper::loadShaderModule(
        device, "cube_reproject_comp.spv",
        er::ShaderStageFlagBits::COMPUTE_BIT,
        std::source_location::current());
    reproject_pipeline_ = device->createPipeline(
        reproject_pipeline_layout_, cs,
        std::source_location::current());

    // 12 descriptor sets (2 ping-pong × 6 faces).  Each set binds:
    //   - src color   = "the OTHER ping-pong buffer" (the read source)
    //   - src depth   = the single depth cube
    //   - dst color   = the per-face 2D layer view of the write target
    for (int p = 0; p < 2; ++p) {
        for (uint32_t f = 0; f < kNumFaces; ++f) {
            reproject_desc_sets_[p][f] = device->createDescriptorSets(
                descriptor_pool, reproject_desc_set_layout_, 1)[0];
        }
    }
}

void DynamicCubemap::writeReprojectDescriptors(
    const std::shared_ptr<er::Device>& device) {
    er::WriteDescriptorList writes;
    writes.reserve(2 * kNumFaces * 3);

    for (int p = 0; p < 2; ++p) {
        // When write target = ping_pong[p], the read source is ping_pong[1-p].
        const int read_idx = 1 - p;
        for (uint32_t f = 0; f < kNumFaces; ++f) {
            const auto& set = reproject_desc_sets_[p][f];

            er::Helper::addOneTexture(writes, set,
                er::DescriptorType::COMBINED_IMAGE_SAMPLER,
                kReprojSrcColorBinding,
                sampler_, color_cubes_[read_idx].view,
                er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            er::Helper::addOneTexture(writes, set,
                er::DescriptorType::COMBINED_IMAGE_SAMPLER,
                kReprojSrcDepthBinding,
                sampler_, depth_cube_.view,
                er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            er::Helper::addOneTexture(writes, set,
                er::DescriptorType::STORAGE_IMAGE,
                kReprojDstColorBinding,
                /*sampler*/ nullptr,
                color_face_views_[p][f],
                er::ImageLayout::GENERAL);
        }
    }
    device->updateDescriptorSets(writes);
}

// ─── Face view / projection helpers ────────────────────────────────────
glm::mat4 DynamicCubemap::cubeFaceView(uint32_t face, const glm::vec3& origin) {
    const auto& a = kFaceAxes[face];
    return glm::lookAtRH(origin, origin + a.forward, a.up);
}

glm::mat4 DynamicCubemap::cubeFaceProj() {
    // 90° FOV, 1:1 aspect, near=0.1, far=1000.0.  GLM is built with
    // GLM_FORCE_DEPTH_ZERO_TO_ONE in this project so perspectiveRH gives
    // Vulkan-conventioned depth.
    return glm::perspectiveRH(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
}

// ─── Per-frame update ──────────────────────────────────────────────────
// Renders one cube face at the active probe position by:
//   1. Updating the face's camera UBO with view+proj for the active probe.
//   2. Substituting the face camera descriptor set at VIEW_PARAMS_SET in
//      the cluster_desc_sets list.
//   3. Beginning a render pass on the active face's color slice +
//      transient depth.
//   4. Calling cluster_renderer->draw() to rasterize all visible
//      static meshes (opaque + glass via OIT) into the face.
//   5. Calling skydome->drawEnvmap() to fill remaining sky pixels.
//   6. Ending the render pass; transitioning the face slice to
//      SHADER_READ_ONLY for downstream consumers.
//
// Reprojection is intentionally NOT dispatched here.  Each probe is
// captured fresh over 6 consecutive frames at the same origin, and SH
// projection only runs at the end of each probe's cycle (frame % 6 ==
// 5), so the cubemap is always consistent at the moment it's read.
// The reprojection pipeline still exists in the class for future use
// (reflection probes that move with the camera) but isn't on the per-
// frame critical path.
void DynamicCubemap::update(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const glm::vec3& probe_pos,
    const std::shared_ptr<ClusterRenderer>& cluster_renderer,
    const std::shared_ptr<Skydome>& skydome,
    const er::DescriptorSetList& cluster_desc_sets) {
    const uint32_t face_to_render =
        static_cast<uint32_t>(frame_index_ % kNumFaces);
    current_face_ = face_to_render;

    // ── Ping-pong consistency invariant ────────────────────────────
    // Each probe cycle (6 frames) writes all 6 faces of color_cubes_
    // [write_idx_].  During those 6 frames the OTHER cube
    // (color_cubes_[1 - write_idx_]) is the previous cycle's
    // fully-baked output — so consumers reading via getColorCubeView
    // always see a consistent set of 6 faces from the *same probe
    // origin*.  At the end of frame 5 of each cycle we swap, exposing
    // the just-completed cube and starting the next cycle's writes
    // into the older buffer.
    //
    // Bootstrap: on the very first cycle (frame_index_ < 6) the read
    // buffer is still UNDEFINED; we partially-bootstrap by mirroring
    // each freshly-rendered face into both buffers during cycle 0,
    // so the viewer / SH projection see something coherent from
    // frame 0 onward.
    const int write_idx = current_read_idx_ ^ 1;

    // ── 1. Build face view+proj and write to host-visible UBO ───────
    glsl::ViewCameraInfo info{};
    info.view = cubeFaceView(face_to_render, probe_pos);
    info.proj = cubeFaceProj();
    info.view_proj     = info.proj * info.view;
    info.inv_view_proj = glm::inverse(info.view_proj);
    glm::mat4 view_no_translate = info.view;
    view_no_translate[3] = glm::vec4(0, 0, 0, 1);
    info.inv_view_proj_relative =
        glm::inverse(info.proj * view_no_translate);
    info.inv_view = glm::inverse(info.view);
    info.inv_proj = glm::inverse(info.proj);
    info.depth_params = glm::vec4(0.1f, 1000.0f, 0.0f, 0.0f);
    info.position = probe_pos;
    info.status   = 0;
    info.up_vector = glm::vec3(0, 1, 0);
    info.yaw = 0; info.pitch = 0;
    info.facing_dir = kFaceAxes[face_to_render].forward;
    info.mouse_pos  = glm::vec2(0, 0);
    info.camera_follow_dist = 0;
    info.input_features = 0;

    // Host-visible coherent mapping; buffer was created with
    // HOST_VISIBLE | HOST_COHERENT in createFaceCameraResources, so
    // a direct memcpy is visible to the GPU on the next read.
    if (device_ && face_camera_buffers_[face_to_render].memory) {
        void* mapped = device_->mapMemory(
            face_camera_buffers_[face_to_render].memory,
            sizeof(info), 0);
        if (mapped) {
            std::memcpy(mapped, &info, sizeof(info));
            device_->unmapMemory(
                face_camera_buffers_[face_to_render].memory);
        }
    }

    // ── 2. Build face cluster_desc_sets with VIEW_PARAMS_SET swapped ─
    er::DescriptorSetList face_desc_sets = cluster_desc_sets;
    if (face_desc_sets.size() <= VIEW_PARAMS_SET) {
        face_desc_sets.resize(VIEW_PARAMS_SET + 1, nullptr);
    }
    face_desc_sets[VIEW_PARAMS_SET] = face_view_desc_sets_[face_to_render];

    // ── 3. Begin render pass on the active face ─────────────────────
    // Initial layout transition: color cube face slice →
    // COLOR_ATTACHMENT_OPTIMAL (from UNDEFINED on first use, or
    // SHADER_READ_ONLY on subsequent frames).
    er::ImageResourceInfo as_undefined = {
        er::ImageLayout::UNDEFINED, 0,
        SET_FLAG_BIT(PipelineStage, TOP_OF_PIPE_BIT) };
    er::ImageResourceInfo as_color_attach = {
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
    er::ImageResourceInfo as_depth_attach = {
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        SET_2_FLAG_BITS(Access,
            DEPTH_STENCIL_ATTACHMENT_READ_BIT,
            DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
        SET_2_FLAG_BITS(PipelineStage,
            EARLY_FRAGMENT_TESTS_BIT, LATE_FRAGMENT_TESTS_BIT) };

    // Always treat the active face's source layout as UNDEFINED.  This
    // is safe because:
    //   • the render pass loads with CLEAR on color, so the prior
    //     contents are discarded anyway;
    //   • on the first cycle for this face the layout truly is UNDEFINED
    //     (allocated, never written);
    //   • on subsequent cycles it's SHADER_READ_ONLY but Vulkan allows
    //     UNDEFINED → COLOR_ATTACHMENT transitions, just dropping the
    //     prior contents (which we're about to clear anyway).
    cmd_buf->addImageBarrier(color_cubes_[write_idx].image,
        as_undefined, as_color_attach,
        0, 1, face_to_render, 1);
    cmd_buf->addImageBarrier(face_depth_target_.image,
        as_undefined, as_depth_attach,
        0, 1, 0, 1);

    er::RenderingAttachmentInfo color_att;
    color_att.image_view   = color_face_views_[write_idx][face_to_render];
    color_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    color_att.load_op      = er::AttachmentLoadOp::CLEAR;
    color_att.store_op     = er::AttachmentStoreOp::STORE;
    color_att.clear_value.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

    er::RenderingAttachmentInfo depth_att;
    depth_att.image_view   = face_depth_target_.view;
    depth_att.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_att.load_op      = er::AttachmentLoadOp::CLEAR;
    depth_att.store_op     = er::AttachmentStoreOp::STORE;
    depth_att.clear_value.depth_stencil = { 1.0f, 0 };

    er::RenderingInfo ri = {};
    ri.render_area_offset  = { 0, 0 };
    ri.render_area_extent  = { edge_, edge_ };
    ri.layer_count         = 1;
    ri.view_mask           = 0;
    ri.color_attachments   = { color_att };
    ri.depth_attachments   = { depth_att };
    ri.stencil_attachments = {};

    cmd_buf->beginDynamicRendering(ri);

    er::Viewport vp;
    vp.x = 0; vp.y = 0;
    vp.width = float(edge_); vp.height = float(edge_);
    vp.min_depth = 0; vp.max_depth = 1;
    er::Scissor sc;
    sc.offset = glm::ivec2(0);
    sc.extent = glm::uvec2(edge_, edge_);
    cmd_buf->setViewports({ vp });
    cmd_buf->setScissors({ sc });

    // ── 4. Draw cluster meshes for THIS face into the active face slice ──
    // Caller (AmbientProbeSystem::update) is expected to have just
    // run cluster_renderer_->cull() with this face's view-proj so the
    // indirect_draw_buffer_ contains the clusters visible from this
    // face's 90° FOV.  We just consume that here.  See AmbientProbeSystem
    // ::update for the cull-then-render-then-recull-for-main ordering.
    if (cluster_renderer) {
        cluster_renderer->drawOpaqueOnly(
            cmd_buf, face_desc_sets,
            { vp }, { sc });
    }

    // ── 5. Sky envmap on top (LESS_OR_EQUAL depth at z=1.0) ─────────
    if (skydome) {
        skydome->drawEnvmap(cmd_buf, info.inv_view_proj_relative);
    }

    cmd_buf->endDynamicRendering();

    // ── 5b. Depth → linear-distance compute pass for this face ─────
    // Reads face_depth_target_ (clip-space D24S8 the render pass just
    // wrote) and writes world-space distance from the camera origin
    // into depth_cube_'s matching face slice.  sh_project.comp picks
    // it up later when doing parallax-aware probe SH integration.
    {
        // face_depth_target_ → SHADER_READ_ONLY (it was DEPTH_STENCIL_
        // ATTACHMENT after the render pass ended).
        er::ImageResourceInfo from_depth_attach = {
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            SET_2_FLAG_BITS(Access,
                DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            SET_2_FLAG_BITS(PipelineStage,
                EARLY_FRAGMENT_TESTS_BIT, LATE_FRAGMENT_TESTS_BIT) };
        er::ImageResourceInfo to_shader_read = {
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
        cmd_buf->addImageBarrier(face_depth_target_.image,
            from_depth_attach, to_shader_read,
            0, 1, 0, 1);

        // depth_cube_ face slice → GENERAL for storage-image write.
        er::ImageResourceInfo to_storage_write = {
            er::ImageLayout::GENERAL,
            SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
        cmd_buf->addImageBarrier(depth_cube_.image,
            er::Helper::getImageAsSource(),
            to_storage_write,
            0, 1, face_to_render, 1);

        // Push constants: face's inv_view_proj + camera position + edge.
        DepthToLinearPushConstants dpc{};
        dpc.inv_view_proj  = info.inv_view_proj;
        dpc.camera_pos_pad = glm::vec4(probe_pos, 0.0f);
        dpc.edge           = static_cast<int32_t>(edge_);

        cmd_buf->bindPipeline(
            er::PipelineBindPoint::COMPUTE,
            depth_to_linear_pipeline_);
        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            depth_to_linear_pipeline_layout_,
            { depth_to_linear_desc_sets_[face_to_render] });
        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            depth_to_linear_pipeline_layout_,
            &dpc, sizeof(dpc));

        const uint32_t groups = (edge_ + 7u) / 8u;
        cmd_buf->dispatch(groups, groups, 1);

        // depth_cube_ face slice → SHADER_READ_ONLY for sh_project.comp.
        cmd_buf->addImageBarrier(depth_cube_.image,
            to_storage_write,
            er::Helper::getImageAsShaderSampler(),
            0, 1, face_to_render, 1);
        // face_depth_target_ → DEPTH_STENCIL_ATTACHMENT_OPTIMAL ready
        // for the next face render's CLEAR/STORE pass.
        cmd_buf->addImageBarrier(face_depth_target_.image,
            to_shader_read,
            from_depth_attach,
            0, 1, 0, 1);
    }

    // ── 6. Transitions to SHADER_READ_ONLY for downstream reads ────
    // The active face transitions from COLOR_ATTACHMENT (where the
    // render pass left it).  On frame 0 we ALSO transition the other
    // 5 faces from UNDEFINED → SHADER_READ_ONLY so ImGui's per-face
    // descriptor reads + the SH projection's samplerCube read aren't
    // touching layout-UNDEFINED memory on the first cycle.  The
    // contents of the not-yet-rendered faces are garbage, but ImGui
    // / SH projection both tolerate that — they just see noise.  By
    // frame 5 every face has been freshly rendered.
    cmd_buf->addImageBarrier(color_cubes_[write_idx].image,
        as_color_attach,
        er::Helper::getImageAsShaderSampler(),
        0, 1, face_to_render, 1);
    if (frame_index_ == 0) {
        // Other 5 faces of the WRITE cube — un-rendered until later in
        // this cycle.  Move them out of UNDEFINED so cube samplers can
        // safely read them (contents are still garbage; a few frames
        // of noise is acceptable during the first probe cycle).
        for (uint32_t f = 0; f < kNumFaces; ++f) {
            if (f == face_to_render) continue;
            cmd_buf->addImageBarrier(color_cubes_[write_idx].image,
                as_undefined,
                er::Helper::getImageAsShaderSampler(),
                0, 1, f, 1);
        }
        // Other ping-pong buffer — currently the *read* buffer that
        // consumers + ImGui see for the first 6 frames before the
        // first swap.  All 6 faces are UNDEFINED; transition all
        // together so cube samplers don't trip on layout-undefined
        // reads.
        const int other_idx = 1 - write_idx;
        cmd_buf->addImageBarrier(color_cubes_[other_idx].image,
            as_undefined,
            er::Helper::getImageAsShaderSampler(),
            0, 1, 0, kNumFaces);
        // Depth cube — unused by the active render path but the
        // reproject pipeline's descriptors reference it as a cube
        // sampler, so transition out of UNDEFINED for sampler safety.
        cmd_buf->addImageBarrier(depth_cube_.image,
            as_undefined,
            er::Helper::getImageAsShaderSampler(),
            0, 1, 0, kNumFaces);
    }

    face_capture_pos_[face_to_render] = probe_pos;
    ++frame_index_;
}

// ─── Mipgen for the read cube ──────────────────────────────────────────
// Called by AmbientProbeSystem at end-of-cycle (post-swap) so the
// just-completed cube has a full mip chain ready for SH projection.
// Mip 0 is left untouched (it holds the freshly-rendered face data);
// mips 1..N are populated by the standard 2:1 LINEAR-filter blit chain.
void DynamicCubemap::generateMipsForRead(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf) {
    const uint32_t mip_count =
        static_cast<uint32_t>(std::log2(edge_) + 1);
    if (mip_count <= 1) return;

    // The face render's tail barrier left mip 0 in
    // SHADER_READ_ONLY_OPTIMAL across all 6 faces.  Mips 1..N have
    // never been written and are in UNDEFINED.  generateMipmapLevels
    // will internally transition mip 0 → TRANSFER_SRC, mip i+1 →
    // TRANSFER_DST, blit, then transition all to SHADER_READ_ONLY at
    // the end.  We pass cur_image_layout = SHADER_READ_ONLY_OPTIMAL
    // for mip 0; mips 1..N's UNDEFINED state is acceptable for the
    // first transition (TRANSFER_DST discards the prior content).
    er::Helper::generateMipmapLevels(
        cmd_buf,
        color_cubes_[current_read_idx_].image,
        mip_count,
        edge_, edge_,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
}

// ─── Cleanup ────────────────────────────────────────────────────────────
void DynamicCubemap::destroy(const std::shared_ptr<er::Device>& device) {
    for (int p = 0; p < 2; ++p) {
        for (uint32_t f = 0; f < kNumFaces; ++f) {
            color_face_views_[p][f].reset();
        }
        if (color_cubes_[p].image) color_cubes_[p].destroy(device);
    }
    for (uint32_t f = 0; f < kNumFaces; ++f) {
        depth_face_views_[f].reset();
    }
    if (depth_cube_.image) depth_cube_.destroy(device);

    // Per-face camera UBOs + descriptor sets.
    for (uint32_t f = 0; f < kNumFaces; ++f) {
        if (face_camera_buffers_[f].buffer)
            face_camera_buffers_[f].destroy(device);
        face_view_desc_sets_[f].reset();
    }
    // Transient depth target.
    if (face_depth_target_.image) face_depth_target_.destroy(device);

    if (face_camera_ubo_.buffer) face_camera_ubo_.destroy(device);

    reproject_pipeline_.reset();
    reproject_pipeline_layout_.reset();
    reproject_desc_set_layout_.reset();
    for (int p = 0; p < 2; ++p) {
        for (uint32_t f = 0; f < kNumFaces; ++f) {
            reproject_desc_sets_[p][f].reset();
        }
    }
}

}// namespace scene_rendering
}// namespace engine
