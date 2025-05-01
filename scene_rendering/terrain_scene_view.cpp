#include <string>
#include <sstream>
#include <unordered_map>
#include "terrain_scene_view.h"
#include "engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {

namespace scene_rendering {

TerrainSceneView::TerrainSceneView(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<ego::CameraObject>& camera_object,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<er::TextureInfo>& color_buffer = nullptr,
    const std::shared_ptr<er::TextureInfo>& depth_buffer = nullptr) :
    ViewObject(device, descriptor_pool, camera_object, color_buffer, depth_buffer) {

    auto color_no_blend_attachment =
        er::helper::fillPipelineColorBlendAttachmentState();

    std::vector<er::PipelineColorBlendAttachmentState>
        color_no_blend_attachments(1, color_no_blend_attachment);

    auto single_no_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(color_no_blend_attachments));

    auto cull_rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo());

    auto no_cull_rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo(
                false,
                false,
                er::PolygonMode::FILL,
                SET_FLAG_BIT(CullMode, NONE)));

    auto ms_info = std::make_shared<er::PipelineMultisampleStateCreateInfo>(
        er::helper::fillPipelineMultisampleStateCreateInfo());

    auto depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo());

    er::GraphicPipelineInfo graphic_pipeline_info;
    graphic_pipeline_info.blend_state_info = single_no_blend_state_info;
    graphic_pipeline_info.rasterization_info = cull_rasterization_info;
    graphic_pipeline_info.ms_info = ms_info;
    graphic_pipeline_info.depth_stencil_info = depth_stencil_info;

    er::GraphicPipelineInfo graphic_double_face_pipeline_info;
    graphic_double_face_pipeline_info.blend_state_info = single_no_blend_state_info;
    graphic_double_face_pipeline_info.rasterization_info = no_cull_rasterization_info;
    graphic_double_face_pipeline_info.ms_info = ms_info;
    graphic_double_face_pipeline_info.depth_stencil_info = depth_stencil_info;

    auto desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(ego::TileObject::getTileResDescSetLayout());

    if (m_tile_pipeline_layout_ == nullptr) {
        m_tile_pipeline_layout_ =
            ego::TileObject::createTilePipelineLayout(
                device,
                desc_set_layouts);
    }


    if (m_tile_grass_pipeline_layout_ == nullptr) {
        m_tile_grass_pipeline_layout_ =
            ego::TileObject::createTileGrassPipelineLayout(
                device,
                desc_set_layouts);
    }

    if (m_tile_pipeline_ == nullptr) {
        m_tile_pipeline_ =
            ego::TileObject::createTilePipeline(
                device,
                m_render_pass_,
                m_tile_pipeline_layout_,
                graphic_pipeline_info,
                m_buffer_size_,
                "terrain/tile_soil_vert.spv",
                "terrain/tile_frag.spv");
    }

    if (m_tile_water_pipeline_ == nullptr) {
        m_tile_water_pipeline_ =
            ego::TileObject::createTilePipeline(
                device,
                m_blend_render_pass_,
                m_tile_pipeline_layout_,
                graphic_pipeline_info,
                m_buffer_size_,
                "terrain/tile_water_vert.spv",
                "terrain/tile_water_frag.spv");
    }

    if (m_tile_grass_pipeline_ == nullptr) {
        m_tile_grass_pipeline_ =
            ego::TileObject::createGrassPipeline(
                device,
                m_render_pass_,
                m_tile_grass_pipeline_layout_,
                graphic_double_face_pipeline_info,
                m_buffer_size_);
    }

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

void TerrainSceneView::updateTileResDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& clamp_texture_sampler,
    const std::shared_ptr<renderer::Sampler>& repeat_texture_sampler,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_tex,
    const std::shared_ptr<renderer::ImageView>& map_mask_tex,
    const std::shared_ptr<renderer::ImageView>& detail_volume_noise_tex,
    const std::shared_ptr<renderer::ImageView>& rough_volume_noise_tex) {
        ego::TileObject::updateTileResDescriptorSet(
            device,
            descriptor_pool,
            m_tile_res_desc_sets_,
            clamp_texture_sampler,
            repeat_texture_sampler,
            m_color_buffer_copy_->view,
            m_depth_buffer_copy_->view,
            temp_tex,
            map_mask_tex,
            detail_volume_noise_tex,
            rough_volume_noise_tex);
}

void TerrainSceneView::duplicateColorAndDepthBuffer(
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

void TerrainSceneView::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& prt_desc_set,
    int dbuf_idx,
    float delta_t,
    float cur_time) {

    const renderer::DescriptorSetList& tile_desc_set_list =
        { prt_desc_set,
          m_camera_object_->getViewCameraDescriptorSet(),
          m_tile_res_desc_sets_[dbuf_idx] };

    cmd_buf->beginRenderPass(
        m_render_pass_,
        m_frame_buffer_,
        m_buffer_size_,
        m_clear_values_);

    // Draw all visible tiles
    if (m_b_render_terrain_) {
        for (auto& tile : m_visible_tiles_) {
            tile->draw(
                cmd_buf,
                m_tile_pipeline_layout_,
                m_tile_pipeline_,
                tile_desc_set_list,
                m_buffer_size_,
                delta_t,
                cur_time);

            if (m_b_render_grass_) {
                tile->drawGrass(
                    cmd_buf,
                    m_tile_grass_pipeline_layout_,
                    m_tile_grass_pipeline_,
                    tile_desc_set_list,
                    m_camera_object_->getCameraPosition(),
                    m_buffer_size_,
                    delta_t,
                    cur_time);
            }
        }
    }

    cmd_buf->endRenderPass();

    if (m_b_render_terrain_ && m_b_render_water_) {
        duplicateColorAndDepthBuffer(cmd_buf);

        cmd_buf->beginRenderPass(
            m_blend_render_pass_,
            m_frame_buffer_,
            m_buffer_size_,
            m_clear_values_);

        // Draw all visible tiles
        for (auto& tile : m_visible_tiles_) {
            tile->draw(
                cmd_buf,
                m_tile_pipeline_layout_,
                m_tile_water_pipeline_,
                tile_desc_set_list,
                m_buffer_size_,
                delta_t,
                cur_time);
        }

        cmd_buf->endRenderPass();
    }
}

void TerrainSceneView::destroy(const std::shared_ptr<renderer::Device>& device) {
    if (m_tile_pipeline_layout_) {
        device->destroyPipelineLayout(m_tile_pipeline_layout_);
        m_tile_pipeline_layout_ = nullptr;
    }

    if (m_tile_grass_pipeline_layout_) {
        device->destroyPipelineLayout(m_tile_grass_pipeline_layout_);
        m_tile_grass_pipeline_layout_ = nullptr;
    }

    if (m_tile_pipeline_) {
        device->destroyPipeline(m_tile_pipeline_);
        m_tile_pipeline_ = nullptr;
    }

    if (m_tile_water_pipeline_) {
        device->destroyPipeline(m_tile_water_pipeline_);
        m_tile_water_pipeline_ = nullptr;
    }

    if (m_tile_grass_pipeline_) {
        device->destroyPipeline(m_tile_grass_pipeline_);
        m_tile_grass_pipeline_ = nullptr;
    }

    ViewObject::destroy(device);
};

} // game_object
} // engine