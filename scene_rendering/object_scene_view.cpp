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

void ObjectSceneView::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    std::shared_ptr<ego::Sphere> sphere,
    int dbuf_idx,
    float delta_t,
    float cur_time,
    bool depth_only/* = false */) {

    renderer::DescriptorSetList desc_set_list = desc_sets;
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
        depth_attachment_info.image_view = m_depth_buffer_->view;
        depth_attachment_info.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_info.load_op = er::AttachmentLoadOp::CLEAR;
        depth_attachment_info.store_op = er::AttachmentStoreOp::STORE;
        depth_attachment_info.clear_value.depth_stencil = { 1.0f, 0 };

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

    for (auto& drawable_obj : m_drawable_objects_) {
        drawable_obj->draw(
            cmd_buf,
            desc_set_list,
            viewports,
            scissors,
            depth_only);
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

void ObjectSceneView::destroy(const std::shared_ptr<renderer::Device>& device) {

    ViewObject::destroy(device);
};

} // game_object
} // engine