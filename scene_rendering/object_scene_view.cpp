#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include "object_scene_view.h"
#include "helper/engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {

namespace scene_rendering {

ObjectSceneView::ObjectSceneView(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<ego::CameraObject>& camera_object,
    const std::shared_ptr<er::TextureInfo>& color_buffer/* = nullptr*/,
    const std::shared_ptr<er::TextureInfo>& depth_buffer/* = nullptr*/,
    const glm::uvec2& buffer_size/* = glm::uvec2(2560, 1440)*/,
    bool depth_only/* = false*/) :
    ViewObject(
        device,
        descriptor_pool,
        renderbuffer_formats,
        camera_object,
        color_buffer,
        depth_buffer,
        buffer_size,
        depth_only) {

    // tile params set.
    m_tile_res_desc_sets_.resize(2);
    for (int idx = 0; idx < 2; idx++) {
        m_tile_res_desc_sets_[idx] =
            device->createDescriptorSets(
                descriptor_pool,
                ego::TileObject::getTileResDescSetLayout(),
                1)[0];
    }
}

void ObjectSceneView::duplicateColorAndDepthBuffer(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf) {
    er::ImageResourceInfo color_src_info = {
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

    er::ImageResourceInfo color_dst_info = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

    er::ImageResourceInfo depth_src_info = {
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, EARLY_FRAGMENT_TESTS_BIT) };

    er::ImageResourceInfo depth_dst_info = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

    er::Helper::blitImage(
        cmd_buf,
        m_color_buffer_->image,
        m_color_buffer_copy_->image,
        color_src_info,
        color_src_info,
        color_dst_info,
        color_dst_info,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        m_color_buffer_->size,
        m_color_buffer_copy_->size);

    er::Helper::blitImage(
        cmd_buf,
        m_depth_buffer_->image,
        m_depth_buffer_copy_->image,
        depth_src_info,
        depth_src_info,
        depth_dst_info,
        depth_dst_info,
        SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
        SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
        m_depth_buffer_->size,
        m_depth_buffer_copy_->size);
}

void ObjectSceneView::duplicateDepthBuffer(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf) {
    if (!m_depth_buffer_ || !m_depth_buffer_copy_) {
        return;
    }

    er::ImageResourceInfo depth_src_info = {
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, LATE_FRAGMENT_TESTS_BIT) };

    er::ImageResourceInfo depth_dst_info = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

    // Both src states are passed twice (old == new) so the depth image
    // round-trips back to DEPTH_STENCIL_ATTACHMENT_OPTIMAL and the decal
    // pass that follows can keep depth-testing against it; likewise the
    // copy ends where the fragment shader expects to find it.
    er::Helper::blitImage(
        cmd_buf,
        m_depth_buffer_->image,
        m_depth_buffer_copy_->image,
        depth_src_info,
        depth_src_info,
        depth_dst_info,
        depth_dst_info,
        SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
        SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
        m_depth_buffer_->size,
        m_depth_buffer_copy_->size);
}

void ObjectSceneView::drawDepthPrepass(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_sets) {

    if (m_drawable_objects_.empty() || !m_depth_buffer_) {
        return;
    }

    renderer::DescriptorSetList desc_set_list = desc_sets;
    desc_set_list[VIEW_PARAMS_SET] =
        m_camera_object_->getViewCameraDescriptorSet();

    // Same eye publish as draw(): the prepass must select the same LOD
    // band per tile as the forward pass right after it, or the depth it
    // primes belongs to a tree the forward pass is not drawing.
    ego::DrawableObject::setPlantLodEye(
        m_camera_object_->getCameraViewInfo().position);

    {
        er::RenderingAttachmentInfo depth_attachment_info;
        depth_attachment_info.image_view = m_depth_buffer_->view;
        depth_attachment_info.image_layout =
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_info.load_op = er::AttachmentLoadOp::CLEAR;
        depth_attachment_info.store_op = er::AttachmentStoreOp::STORE;
        depth_attachment_info.clear_value.depth_stencil = { 1.0f, 0 };

        er::RenderingInfo renderingInfo = {};
        renderingInfo.render_area_offset = { 0, 0 };
        renderingInfo.render_area_extent = { m_buffer_size_.x, m_buffer_size_.y };
        renderingInfo.layer_count = 1;
        renderingInfo.view_mask = 0;
        renderingInfo.color_attachments = {};
        renderingInfo.depth_attachments = { depth_attachment_info };
        renderingInfo.stencil_attachments = {};

        cmd_buf->beginDynamicRendering(renderingInfo);
    }

    std::vector<er::Viewport> viewports(1);
    std::vector<er::Scissor> scissors(1);
    viewports[0].x = 0;
    viewports[0].y = 0;
    viewports[0].width = float(m_buffer_size_.x);
    viewports[0].height = float(m_buffer_size_.y);
    viewports[0].min_depth = 0.0f;
    viewports[0].max_depth = 1.0f;
    scissors[0].offset = glm::ivec2(0);
    scissors[0].extent = m_buffer_size_;

    // NEAR TO FAR.  Early-Z inside the prepass only rejects a fragment
    // when something nearer was recorded FIRST, so the order is most of
    // the benefit.  Coarse per-drawable ordering by world translation is
    // enough — the point is the houses before the forest behind them,
    // not exact per-triangle order — and a stable sort keeps equal-
    // distance drawables in registration order for cache locality.
    const glm::vec3 eye = m_camera_object_->getCameraViewInfo().position;
    std::vector<ego::DrawableObject*> order;
    order.reserve(m_drawable_objects_.size());
    for (auto& d : m_drawable_objects_) {
        order.push_back(d.get());
    }
    std::stable_sort(
        order.begin(), order.end(),
        [&eye](const ego::DrawableObject* a, const ego::DrawableObject* b) {
            const glm::vec3 da = a->getSortWorldPos() - eye;
            const glm::vec3 db = b->getSortWorldPos() - eye;
            return glm::dot(da, da) < glm::dot(db, db);
        });

    for (auto* drawable_obj : order) {
        drawable_obj->draw(
            cmd_buf,
            desc_set_list,
            viewports,
            scissors,
            false,
            ego::DrawableObject::DrawMode::kDepthPrepass,
            0u);
    }

    cmd_buf->endDynamicRendering();
}

void ObjectSceneView::drawGbuffer(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<std::shared_ptr<renderer::ImageView>>& gbuffer_views,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const glm::uvec2& buffer_size) {

    if (m_drawable_objects_.empty() || !depth_view) {
        return;
    }

    renderer::DescriptorSetList desc_set_list = desc_sets;
    desc_set_list[VIEW_PARAMS_SET] =
        m_camera_object_->getViewCameraDescriptorSet();

    {
        // LOAD everything: the cluster phases and the terrain G-buffer
        // pass already wrote these targets; the drawables add their
        // pixels on top, depth-gated against the forward pass's depth.
        std::vector<er::RenderingAttachmentInfo> color_attachment_infos;
        color_attachment_infos.reserve(gbuffer_views.size());
        for (const auto& view : gbuffer_views) {
            er::RenderingAttachmentInfo attachment_info;
            attachment_info.image_view = view;
            attachment_info.image_layout =
                er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            attachment_info.load_op = er::AttachmentLoadOp::LOAD;
            attachment_info.store_op = er::AttachmentStoreOp::STORE;
            color_attachment_infos.push_back(attachment_info);
        }

        er::RenderingAttachmentInfo depth_attachment_info;
        depth_attachment_info.image_view = depth_view;
        depth_attachment_info.image_layout =
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_info.load_op = er::AttachmentLoadOp::LOAD;
        depth_attachment_info.store_op = er::AttachmentStoreOp::STORE;

        er::RenderingInfo renderingInfo = {};
        renderingInfo.render_area_offset = { 0, 0 };
        renderingInfo.render_area_extent = { buffer_size.x, buffer_size.y };
        renderingInfo.layer_count = 1;
        renderingInfo.view_mask = 0;
        renderingInfo.color_attachments = color_attachment_infos;
        renderingInfo.depth_attachments = { depth_attachment_info };
        renderingInfo.stencil_attachments = {};

        cmd_buf->beginDynamicRendering(renderingInfo);
    }

    std::vector<er::Viewport> viewports(1);
    std::vector<er::Scissor> scissors(1);
    viewports[0].x = 0;
    viewports[0].y = 0;
    viewports[0].width = float(buffer_size.x);
    viewports[0].height = float(buffer_size.y);
    viewports[0].min_depth = 0.0f;
    viewports[0].max_depth = 1.0f;
    scissors[0].offset = glm::ivec2(0);
    scissors[0].extent = buffer_size;

    // Same eye publish as draw(): the LOD band memo keys on the eye, so
    // this pass re-uses the exact band selection the forward pass made
    // this frame — the G-buffer can never disagree with the forward
    // depth about which LOD is standing at a tile.
    ego::DrawableObject::setPlantLodEye(
        m_camera_object_->getCameraViewInfo().position);

    for (auto& drawable_obj : m_drawable_objects_) {
        drawable_obj->draw(
            cmd_buf,
            desc_set_list,
            viewports,
            scissors,
            false,
            ego::DrawableObject::DrawMode::kGBuffer,
            0u);
    }

    cmd_buf->endDynamicRendering();
}

// ── Deferred ground decals ───────────────────────────────────────────
// The deferred counterpart of drawDecals(): the same decal meshes, but
// re-rasterised into the cluster G-buffer BEFORE the resolve instead of
// blended over the finished image after it.  The decal's albedo lands on
// top of whatever the terrain-tile / drawable G-buffer passes already
// wrote, and deferred_resolve.comp then lights ground-plus-decal as ONE
// surface — so the decal inherits the ground's traced shadow, RT AO and
// RT GI for free.
//
// This exists because the post-resolve forward pass could not: the app
// raises FEATURE_INPUT_SHADOW_DISABLED the moment any RT technique arms
// (the CSM cascades are stale and must not be sampled), so base.frag's
// forward path shaded every decal fragment with shadow = 1.0 and no
// occlusion of any kind, while the terrain it sat on was fully
// RT-relit.  A brightly lit ribbon on shadowed ground reads as a decal
// hovering above the surface.
//
// MUST be issued AFTER drawGbuffer() and after the terrain tiles'
// G-buffer pass: the blend needs the ground albedo already in the
// target, and the ">= 0.5 written" sentinel it preserves has to be
// there to preserve.
void ObjectSceneView::drawDecalsGbuffer(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<std::shared_ptr<renderer::ImageView>>& gbuffer_views,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const glm::uvec2& buffer_size) {

    if (m_decal_objects_.empty() || !depth_view) {
        return;
    }

    // Same reason as drawDecals(): the DECAL permutations sample the
    // scene depth to compute their ground-contact fade, and sampling
    // m_depth_buffer_ while it is bound as the depth attachment would
    // trip VUID-vkCmdDraw-None-09600.  Must happen outside the render
    // pass, so before beginDynamicRendering below.
    duplicateDepthBuffer(cmd_buf);

    renderer::DescriptorSetList desc_set_list = desc_sets;
    desc_set_list[VIEW_PARAMS_SET] =
        m_camera_object_->getViewCameraDescriptorSet();

    {
        // LOAD every target — this pass blends onto them, and the three
        // it does not touch are write-masked off in the pipeline rather
        // than detached here, so the attachment list still has to match
        // the G-buffer pipeline's colour-attachment count.
        std::vector<er::RenderingAttachmentInfo> color_attachment_infos;
        color_attachment_infos.reserve(gbuffer_views.size());
        for (const auto& view : gbuffer_views) {
            er::RenderingAttachmentInfo attachment_info;
            attachment_info.image_view = view;
            attachment_info.image_layout =
                er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            attachment_info.load_op = er::AttachmentLoadOp::LOAD;
            attachment_info.store_op = er::AttachmentStoreOp::STORE;
            color_attachment_infos.push_back(attachment_info);
        }

        er::RenderingAttachmentInfo depth_attachment_info;
        depth_attachment_info.image_view = depth_view;
        depth_attachment_info.image_layout =
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_info.load_op = er::AttachmentLoadOp::LOAD;
        // STORE, not DONT_CARE: the pipeline does not write depth, but
        // the resolve and every later pass still read this buffer.
        depth_attachment_info.store_op = er::AttachmentStoreOp::STORE;

        er::RenderingInfo renderingInfo = {};
        renderingInfo.render_area_offset = { 0, 0 };
        renderingInfo.render_area_extent = { buffer_size.x, buffer_size.y };
        renderingInfo.layer_count = 1;
        renderingInfo.view_mask = 0;
        renderingInfo.color_attachments = color_attachment_infos;
        renderingInfo.depth_attachments = { depth_attachment_info };
        renderingInfo.stencil_attachments = {};

        cmd_buf->beginDynamicRendering(renderingInfo);
    }

    std::vector<er::Viewport> viewports(1);
    std::vector<er::Scissor> scissors(1);
    viewports[0].x = 0;
    viewports[0].y = 0;
    viewports[0].width = float(buffer_size.x);
    viewports[0].height = float(buffer_size.y);
    viewports[0].min_depth = 0.0f;
    viewports[0].max_depth = 1.0f;
    scissors[0].offset = glm::ivec2(0);
    scissors[0].extent = buffer_size;

    // Same eye publish drawDecals() makes, and for the same reason: the
    // ground-clutter distance cull on the CPU and the alpha ramp on the
    // GPU must agree about where the camera is, or tiles vanish while
    // still partly opaque.
    ego::DrawableObject::setViewerWorldPos(
        m_camera_object_->getCameraViewInfo().position);

    for (auto& decal_obj : m_decal_objects_) {
        decal_obj->draw(
            cmd_buf,
            desc_set_list,
            viewports,
            scissors,
            false,
            ego::DrawableObject::DrawMode::kDecalGBuffer,
            0u);
    }

    ego::DrawableObject::clearViewerWorldPos();

    cmd_buf->endDynamicRendering();
}

void ObjectSceneView::drawGlassForward(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::shared_ptr<renderer::ImageView>& color_view,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const glm::uvec2& buffer_size) {

    if (m_drawable_objects_.empty() || !color_view || !depth_view) {
        return;
    }

    renderer::DescriptorSetList desc_set_list = desc_sets;
    desc_set_list[VIEW_PARAMS_SET] =
        m_camera_object_->getViewCameraDescriptorSet();

    {
        // LOAD both: the whole point is to blend onto the finished
        // scene, and the depth (with writes off in the pipeline) only
        // gates panes behind walls.
        std::vector<er::RenderingAttachmentInfo> color_attachment_infos;
        er::RenderingAttachmentInfo attachment_info;
        attachment_info.image_view = color_view;
        attachment_info.image_layout =
            er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        attachment_info.load_op = er::AttachmentLoadOp::LOAD;
        attachment_info.store_op = er::AttachmentStoreOp::STORE;
        color_attachment_infos.push_back(attachment_info);

        er::RenderingAttachmentInfo depth_attachment_info;
        depth_attachment_info.image_view = depth_view;
        depth_attachment_info.image_layout =
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_info.load_op = er::AttachmentLoadOp::LOAD;
        depth_attachment_info.store_op = er::AttachmentStoreOp::STORE;

        er::RenderingInfo renderingInfo = {};
        renderingInfo.render_area_offset = { 0, 0 };
        renderingInfo.render_area_extent = { buffer_size.x, buffer_size.y };
        renderingInfo.layer_count = 1;
        renderingInfo.view_mask = 0;
        renderingInfo.color_attachments = color_attachment_infos;
        renderingInfo.depth_attachments = { depth_attachment_info };
        renderingInfo.stencil_attachments = {};

        cmd_buf->beginDynamicRendering(renderingInfo);
    }

    std::vector<er::Viewport> viewports(1);
    std::vector<er::Scissor> scissors(1);
    viewports[0].x = 0;
    viewports[0].y = 0;
    viewports[0].width = float(buffer_size.x);
    viewports[0].height = float(buffer_size.y);
    viewports[0].min_depth = 0.0f;
    viewports[0].max_depth = 1.0f;
    scissors[0].offset = glm::ivec2(0);
    scissors[0].extent = buffer_size;

    ego::DrawableObject::setPlantLodEye(
        m_camera_object_->getCameraViewInfo().position);

    for (auto& drawable_obj : m_drawable_objects_) {
        drawable_obj->draw(
            cmd_buf,
            desc_set_list,
            viewports,
            scissors,
            false,
            ego::DrawableObject::DrawMode::kGlassAttr,
            0u);
    }

    cmd_buf->endDynamicRendering();
}

void ObjectSceneView::drawDecals(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    int dbuf_idx,
    float delta_t,
    float cur_time) {

    if (m_decal_objects_.empty() || !m_color_buffer_ || !m_depth_buffer_) {
        return;
    }

    // Snapshot terrain+props depth into the sampleable copy.  Sampling
    // m_depth_buffer_ directly while it is bound as the depth attachment
    // would trip VUID-vkCmdDraw-None-09600.
    duplicateDepthBuffer(cmd_buf);

    renderer::DescriptorSetList desc_set_list = desc_sets;
    desc_set_list[VIEW_PARAMS_SET] =
        m_camera_object_->getViewCameraDescriptorSet();

    {
        std::vector<er::RenderingAttachmentInfo> color_attachment_infos;
        color_attachment_infos.reserve(1);
        er::RenderingAttachmentInfo attachment_info;
        attachment_info.image_view = m_color_buffer_->view;
        attachment_info.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        // LOAD, not CLEAR/DONT_CARE — the whole point of this pass is to
        // blend onto the terrain and props already sitting in there.
        attachment_info.load_op = er::AttachmentLoadOp::LOAD;
        attachment_info.store_op = er::AttachmentStoreOp::STORE;
        color_attachment_infos.push_back(attachment_info);

        er::RenderingAttachmentInfo depth_attachment_info;
        depth_attachment_info.image_view = m_depth_buffer_->view;
        depth_attachment_info.image_layout =
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_info.load_op = er::AttachmentLoadOp::LOAD;
        // STORE rather than DONT_CARE: the decal pipeline does not write
        // depth, but later passes in the frame still read this buffer,
        // so its terrain+props contents must survive the pass.
        depth_attachment_info.store_op = er::AttachmentStoreOp::STORE;

        er::RenderingInfo renderingInfo = {};
        renderingInfo.render_area_offset = { 0, 0 };
        renderingInfo.render_area_extent = { m_buffer_size_.x, m_buffer_size_.y };
        renderingInfo.layer_count = 1;
        renderingInfo.view_mask = 0;
        renderingInfo.color_attachments = color_attachment_infos;
        renderingInfo.depth_attachments = { depth_attachment_info };
        renderingInfo.stencil_attachments = {};

        cmd_buf->beginDynamicRendering(renderingInfo);
    }

    std::vector<er::Viewport> viewports(1);
    std::vector<er::Scissor> scissors(1);
    viewports[0].x = 0;
    viewports[0].y = 0;
    viewports[0].width = float(m_buffer_size_.x);
    viewports[0].height = float(m_buffer_size_.y);
    viewports[0].min_depth = 0.0f;
    viewports[0].max_depth = 1.0f;
    scissors[0].offset = glm::ivec2(0);
    scissors[0].extent = m_buffer_size_;

    // Publish the eye position for drawMesh's ground-clutter distance
    // cull.  Deliberately read from the SAME struct base.frag reads
    // (ViewCameraInfo::position, reached through the VIEW_PARAMS_SET
    // descriptor bound just above) so the CPU tile cull and the GPU
    // alpha ramp can never disagree about where the camera is — a
    // mismatch would show up as tiles vanishing while still partly
    // opaque, which is the exact artifact this pass exists to avoid.
    ego::DrawableObject::setViewerWorldPos(
        m_camera_object_->getCameraViewInfo().position);

    for (auto& decal_obj : m_decal_objects_) {
        decal_obj->draw(
            cmd_buf,
            desc_set_list,
            viewports,
            scissors,
            false,
            ego::DrawableObject::DrawMode::kDecal,
            0u);
    }

    // Scoped to this pass: no other draw path has a meaningful eye
    // position published, and a stale one silently culling geometry
    // elsewhere would be a nasty thing to debug.
    ego::DrawableObject::clearViewerWorldPos();

    cmd_buf->endDynamicRendering();
}

void ObjectSceneView::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    std::shared_ptr<ego::Sphere> sphere,
    int dbuf_idx,
    float delta_t,
    float cur_time,
    bool depth_only/* = false */,
    const std::shared_ptr<er::ImageView>& depth_layer_view/* = nullptr */,
    uint32_t layer_count/* = 1 */,
    bool preserve_depth/* = false */,
    int32_t csm_cascade_idx/* = -1 */,
    bool csm_use_mesh_shader/* = false */) {

    // layer_count > 1 means a single-pass layered CSM draw: the geometry shader
    // broadcasts each triangle to all layers; no per-layer view needed.
    const bool csm_layered = (layer_count > 1);
    // csm_cascade_idx >= 0 means a single-cascade per-pass draw using the
    // _CSMCASC pipeline (DrawMode::kCsmPerCascade — "Regular" shadow
    // draw mode).  Caller is responsible for setting up the per-cascade
    // depth_layer_view and looping k=0..CSM_CASCADE_COUNT-1.
    const bool csm_per_cascade = (csm_cascade_idx >= 0);
    // Mesh-shader CSM only applies to layered draws (single dispatch
    // amplified by task+mesh shaders to all cascade layers).  Ignored
    // when csm_per_cascade is requested (those two modes are mutually
    // exclusive — per-cascade does its own host-side cascade loop).
    const bool csm_mesh_shader =
        csm_use_mesh_shader && csm_layered && !csm_per_cascade;

    // ── Publish the eye for plant LOD band selection ────────────────────
    // Here rather than in drawDecals (which publishes the CLUTTER eye)
    // because the tree LOD has to select in this pass and in every shadow
    // cascade, and all of them have to select the SAME band for the same
    // tile — a cascade that disagreed would cast the shadow of a tree the
    // forward pass is not drawing.  Reading it once at the top of the one
    // entry point every pass shares is what guarantees that.
    //
    // Always the main view camera, never a cascade's ortho eye: the
    // cascades all render the player's surroundings, and selecting tree
    // detail from a directional light's viewpoint would swap meshes for
    // cards by sun angle.  m_camera_object_ is that camera in every pass
    // (the VIEW_PARAMS_SET bind below uses it for all of them).
    //
    // Not cleared afterwards, unlike setViewerWorldPos — see that
    // function's comment for the distinction: the clutter eye gates
    // whether geometry EXISTS and a stale one loses tiles, while this one
    // only picks between three drawings of the same tree.
    ego::DrawableObject::setPlantLodEye(
        m_camera_object_->getCameraViewInfo().position);

    renderer::DescriptorSetList desc_set_list = desc_sets;
    // For CSM layered pass there is no per-cascade camera descriptor — the GS
    // reads VP matrices directly from RuntimeLightsParams.  Use a nullptr slot
    // (the shadow vertex shader still reads from VIEW_PARAMS_SET for the
    // world-space transform, so keep the base shadow camera descriptor).
    desc_set_list[VIEW_PARAMS_SET] =
        m_camera_object_->getViewCameraDescriptorSet();

    {
        std::vector<er::RenderingAttachmentInfo> color_attachment_infos;
        color_attachment_infos.reserve(1);
        if (!depth_only) {
            er::RenderingAttachmentInfo attachment_info;
            attachment_info.image_view = m_color_buffer_->view;
            attachment_info.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            attachment_info.load_op = er::AttachmentLoadOp::CLEAR;
            attachment_info.store_op = er::AttachmentStoreOp::STORE;
            attachment_info.clear_value.color = { {0.3f, 0.3f, 0.3f, 1.0f} };
            color_attachment_infos.push_back(attachment_info);
        }
        er::RenderingAttachmentInfo depth_attachment_info;
        // CSM layered: use the full array view; single-cascade: use per-layer view.
        depth_attachment_info.image_view =
            depth_layer_view ? depth_layer_view : m_depth_buffer_->view;
        depth_attachment_info.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        // preserve_depth=true: the caller ran the CSM silhouette prepass
        // (or some other depth-priming pass) in its own dynamic-rendering
        // scope before us, and the depth buffer already contains the
        // intended pre-fill (0 outside silhouette, 1 inside).  Switch to
        // LOAD so we don't wipe that out.
        depth_attachment_info.load_op = preserve_depth
            ? er::AttachmentLoadOp::LOAD
            : er::AttachmentLoadOp::CLEAR;
        depth_attachment_info.store_op = er::AttachmentStoreOp::STORE;
        depth_attachment_info.clear_value.depth_stencil = { 1.0f, 0 };

        er::RenderingInfo renderingInfo = {};
        renderingInfo.render_area_offset = { 0, 0 };
        renderingInfo.render_area_extent = { m_buffer_size_.x, m_buffer_size_.y };
        renderingInfo.layer_count = layer_count;   // >1 for single-pass CSM GS
        renderingInfo.view_mask = 0;
        renderingInfo.color_attachments = color_attachment_infos;
        renderingInfo.depth_attachments = { depth_attachment_info };
        renderingInfo.stencil_attachments = {};

        cmd_buf->beginDynamicRendering(renderingInfo);
    }

    std::vector<er::Viewport> viewports(1);
    std::vector<er::Scissor> scissors(1);
    viewports[0].x = 0;
    viewports[0].y = 0;
    viewports[0].width = float(m_buffer_size_.x);
    viewports[0].height = float(m_buffer_size_.y);
    viewports[0].min_depth = 0.0f;
    viewports[0].max_depth = 1.0f;
    scissors[0].offset = glm::ivec2(0);
    scissors[0].extent = m_buffer_size_;

    const auto draw_mode =
        csm_per_cascade ? ego::DrawableObject::DrawMode::kCsmPerCascade :
        csm_mesh_shader ? ego::DrawableObject::DrawMode::kCsmMeshShader :
        csm_layered     ? ego::DrawableObject::DrawMode::kCsmLayered :
        depth_only      ? ego::DrawableObject::DrawMode::kShadow
                        : ego::DrawableObject::DrawMode::kForward;

    const uint32_t cascade_idx_arg =
        csm_per_cascade ? uint32_t(csm_cascade_idx) : 0u;

    for (auto& drawable_obj : m_drawable_objects_) {
        drawable_obj->draw(
            cmd_buf,
            desc_set_list,
            viewports,
            scissors,
            depth_only,
            draw_mode,
            cascade_idx_arg);
    }

    if (sphere) {
        sphere->draw(
            cmd_buf,
            { desc_set_list[PBR_GLOBAL_PARAMS_SET],
              desc_set_list[VIEW_PARAMS_SET] },
            viewports,
            scissors);
    }

    cmd_buf->endDynamicRendering();

    if (m_b_render_blend_) {
        duplicateColorAndDepthBuffer(cmd_buf);

        {
            std::vector<er::RenderingAttachmentInfo> color_attachment_infos;
            color_attachment_infos.reserve(1);
            if (!depth_only) {
                er::RenderingAttachmentInfo attachment_info;
                attachment_info.image_view = m_color_buffer_->view;
                attachment_info.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                attachment_info.load_op = er::AttachmentLoadOp::DONT_CARE;
                attachment_info.store_op = er::AttachmentStoreOp::STORE;
                color_attachment_infos.push_back(attachment_info);
            }
            er::RenderingAttachmentInfo depth_attachment_info;
            depth_attachment_info.image_view = m_depth_buffer_->view;
            depth_attachment_info.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment_info.load_op = er::AttachmentLoadOp::LOAD;
            depth_attachment_info.store_op = er::AttachmentStoreOp::DONT_CARE;

            er::RenderingInfo renderingInfo = {};
            renderingInfo.render_area_offset = { 0, 0 };
            renderingInfo.render_area_extent = { m_buffer_size_.x, m_buffer_size_.y };
            renderingInfo.layer_count = 1;
            renderingInfo.view_mask = 0;
            renderingInfo.color_attachments = color_attachment_infos;
            renderingInfo.depth_attachments = { depth_attachment_info }; // Or nullptr if no depth
            renderingInfo.stencil_attachments = {};

            cmd_buf->beginDynamicRendering(renderingInfo);
            
        }
        cmd_buf->endDynamicRendering();
    }
}

void ObjectSceneView::recreate(
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const glm::uvec2& new_buffer_size) {

    // Resize render buffers (destroys old, allocates at new size).
    resize(renderbuffer_formats, new_buffer_size);

    // Re-allocate tile-resource descriptor sets from the (new) pool.
    // m_descriptor_pool_ is a reference to the application's pool, so
    // it already points to the freshly-created pool.
    // persistent pool: allocate once, reuse across resize
    if (m_tile_res_desc_sets_.size() < 2) {
        m_tile_res_desc_sets_.resize(2);
    }
    for (int idx = 0; idx < 2; idx++) {
        if (!m_tile_res_desc_sets_[idx]) {
            m_tile_res_desc_sets_[idx] =
                m_device_->createDescriptorSets(
                    m_descriptor_pool_,
                    ego::TileObject::getTileResDescSetLayout(),
                    1)[0];
        }
    }
}

void ObjectSceneView::destroy(const std::shared_ptr<renderer::Device>& device) {

    ViewObject::destroy(device);
};

} // game_object
} // engine
