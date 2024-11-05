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
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<er::TextureInfo>& color_buffer = nullptr,
    const std::shared_ptr<er::TextureInfo>& depth_buffer = nullptr) :
    ViewObject(device, descriptor_pool, color_buffer, depth_buffer) {

    m_view_camera_params_.world_min = ego::TileObject::getWorldMin();
    m_view_camera_params_.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
    m_view_camera_params_.init_camera_pos = glm::vec3(0, 500.0f, 0);
    m_view_camera_params_.init_camera_dir = glm::vec3(1.0f, 0.0f, 0.0f);
    m_view_camera_params_.init_camera_up = glm::vec3(0, 1, 0);

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

void TerrainSceneView::createCameraDescSetWithTerrain(
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {
    m_view_camera_->createViewCameraUpdateDescSet(
        m_device_,
        m_descriptor_pool_,
        texture_sampler,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        game_objects_buffer);
}

void TerrainSceneView::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const uint32_t& dbuf_idx,
    const int& input_key,
    const int& frame_count,
    const float& delta_t,
    const glm::vec2& last_mouse_pos,
    const float& mouse_wheel_offset,
    const bool& camera_rot_update) {

    const float s_camera_speed = 10.0f;

    m_view_camera_params_.camera_speed = s_camera_speed;

    m_view_camera_params_.yaw = 0.0f;
    m_view_camera_params_.pitch = -90.0f;
    m_view_camera_params_.init_camera_dir =
        normalize(ego::getDirectionByYawAndPitch(
            m_view_camera_params_.yaw,
            m_view_camera_params_.pitch));
    m_view_camera_params_.init_camera_up =
        abs(m_view_camera_params_.init_camera_dir.y) < 0.99f ?
        vec3(0, 1, 0) :
        vec3(1, 0, 0);

    m_view_camera_params_.z_near = 0.1f;
    m_view_camera_params_.z_far = 40000.0f;
    m_view_camera_params_.camera_follow_dist = 5.0f;
    m_view_camera_params_.key = input_key;
    m_view_camera_params_.frame_count = frame_count;
    m_view_camera_params_.delta_t = delta_t;
    m_view_camera_params_.mouse_pos = last_mouse_pos;
    m_view_camera_params_.fov = glm::radians(45.0f);
    m_view_camera_params_.aspect = m_buffer_size_.x / (float)m_buffer_size_.y;
    m_view_camera_params_.sensitivity = 0.2f;
    m_view_camera_params_.num_game_objs = static_cast<int32_t>(m_drawable_objects_.size());
    m_view_camera_params_.game_obj_idx = 0;
    m_view_camera_params_.camera_rot_update = camera_rot_update ? 1 : 0;
    m_view_camera_params_.mouse_wheel_offset = mouse_wheel_offset;

    m_view_camera_->updateViewCameraBuffer(
        cmd_buf,
        m_view_camera_params_,
        dbuf_idx);
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
          getViewCameraDescriptorSet(),
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
                    m_camera_pos_,
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