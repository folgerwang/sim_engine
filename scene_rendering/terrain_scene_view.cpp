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

    view_camera_params_.world_min = ego::TileObject::getWorldMin();
    view_camera_params_.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
    view_camera_params_.init_camera_pos = glm::vec3(0, 500.0f, 0);
    view_camera_params_.init_camera_dir = glm::vec3(1.0f, 0.0f, 0.0f);
    view_camera_params_.init_camera_up = glm::vec3(0, 1, 0);

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

    if (tile_pipeline_layout_ == nullptr) {
        tile_pipeline_layout_ =
            ego::TileObject::createTilePipelineLayout(
                device,
                desc_set_layouts);
    }


    if (tile_grass_pipeline_layout_ == nullptr) {
        tile_grass_pipeline_layout_ =
            ego::TileObject::createTileGrassPipelineLayout(
                device,
                desc_set_layouts);
    }

    if (tile_pipeline_ == nullptr) {
        tile_pipeline_ =
            ego::TileObject::createTilePipeline(
                device,
                render_pass_,
                tile_pipeline_layout_,
                graphic_pipeline_info,
                buffer_size_,
                "terrain/tile_soil_vert.spv",
                "terrain/tile_frag.spv");
    }

    if (tile_water_pipeline_ == nullptr) {
        tile_water_pipeline_ =
            ego::TileObject::createTilePipeline(
                device,
                blend_render_pass_,
                tile_pipeline_layout_,
                graphic_pipeline_info,
                buffer_size_,
                "terrain/tile_water_vert.spv",
                "terrain/tile_water_frag.spv");
    }

    if (tile_grass_pipeline_ == nullptr) {
        tile_grass_pipeline_ =
            ego::TileObject::createGrassPipeline(
                device,
                render_pass_,
                tile_grass_pipeline_layout_,
                graphic_double_face_pipeline_info,
                buffer_size_);
    }
}

void TerrainSceneView::createCameraDescSetWithTerrain(
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {
    view_camera_->createViewCameraUpdateDescSet(
        device_,
        descriptor_pool_,
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

    view_camera_params_.camera_speed = s_camera_speed;

    view_camera_params_.yaw = 0.0f;
    view_camera_params_.pitch = -90.0f;
    view_camera_params_.init_camera_dir =
        normalize(ego::getDirectionByYawAndPitch(
            view_camera_params_.yaw,
            view_camera_params_.pitch));
    view_camera_params_.init_camera_up =
        abs(view_camera_params_.init_camera_dir.y) < 0.99f ?
        vec3(0, 1, 0) :
        vec3(1, 0, 0);

    view_camera_params_.z_near = 0.1f;
    view_camera_params_.z_far = 40000.0f;
    view_camera_params_.camera_follow_dist = 5.0f;
    view_camera_params_.key = input_key;
    view_camera_params_.frame_count = frame_count;
    view_camera_params_.delta_t = delta_t;
    view_camera_params_.mouse_pos = last_mouse_pos;
    view_camera_params_.fov = glm::radians(45.0f);
    view_camera_params_.aspect = buffer_size_.x / (float)buffer_size_.y;
    view_camera_params_.sensitivity = 0.2f;
    view_camera_params_.num_game_objs = static_cast<int32_t>(drawable_objects_.size());
    view_camera_params_.game_obj_idx = 0;
    view_camera_params_.camera_rot_update = camera_rot_update ? 1 : 0;
    view_camera_params_.mouse_wheel_offset = mouse_wheel_offset;

    view_camera_->updateViewCameraBuffer(
        cmd_buf,
        view_camera_params_,
        dbuf_idx);
}

void TerrainSceneView::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    int dbuf_idx,
    float delta_t,
    float cur_time) {

    cmd_buf->beginRenderPass(
        render_pass_,
        frame_buffer_,
        buffer_size_,
        clear_values_);

    // Draw all visible tiles
    if (b_render_terrain_) {
        for (auto& tile : visible_tiles_) {
            tile->draw(
                cmd_buf,
                tile_pipeline_layout_,
                is_base_pass_ ? tile_pipeline_ : tile_water_pipeline_,
                desc_set_list,
                buffer_size_,
                dbuf_idx,
                delta_t,
                cur_time);

            if (is_base_pass_ && b_render_grass_) {
                tile->drawGrass(
                    cmd_buf,
                    tile_grass_pipeline_layout_,
                    tile_grass_pipeline_,
                    desc_set_list,
                    camera_pos_,
                    buffer_size_,
                    dbuf_idx,
                    delta_t,
                    cur_time);
            }
        }
    }

    cmd_buf->endRenderPass();
}

void TerrainSceneView::destroy(const std::shared_ptr<renderer::Device>& device) {
    if (tile_pipeline_layout_) {
        device->destroyPipelineLayout(tile_pipeline_layout_);
        tile_pipeline_layout_ = nullptr;
    }

    if (tile_grass_pipeline_layout_) {
        device->destroyPipelineLayout(tile_grass_pipeline_layout_);
        tile_grass_pipeline_layout_ = nullptr;
    }

    if (tile_pipeline_) {
        device->destroyPipeline(tile_pipeline_);
        tile_pipeline_ = nullptr;
    }

    if (tile_water_pipeline_) {
        device->destroyPipeline(tile_water_pipeline_);
        tile_water_pipeline_ = nullptr;
    }

    if (tile_grass_pipeline_) {
        device->destroyPipeline(tile_grass_pipeline_);
        tile_grass_pipeline_ = nullptr;
    }
};

} // game_object
} // engine