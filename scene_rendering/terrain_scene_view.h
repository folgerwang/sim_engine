#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "game_object/terrain.h"
#include "game_object/drawable_object.h"
#include "game_object/patch.h"
#include "game_object/view_object.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace scene_rendering {

class TerrainSceneView : public ego::ViewObject {
    std::shared_ptr<renderer::PipelineLayout> m_tile_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> m_tile_grass_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> m_tile_pipeline_;
    std::shared_ptr<renderer::Pipeline> m_tile_water_pipeline_;
    std::shared_ptr<renderer::Pipeline> m_tile_grass_pipeline_;

    std::vector<std::shared_ptr<ego::TileObject>> m_visible_tiles_;
    std::vector<std::shared_ptr<ego::DrawableObject>> m_visible_object_;
    std::vector<std::shared_ptr<ego::DrawableObject>> m_drawable_objects_;

    bool m_b_render_terrain_ = true;
    bool m_b_render_grass_ = false;
    bool m_b_render_water_ = false;

public:
    TerrainSceneView(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<er::TextureInfo>& color_buffer/* = nullptr*/,
        const std::shared_ptr<er::TextureInfo>& depth_buffer/* = nullptr*/);

    void updateTileResDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& clamp_texture_sampler,
        const std::shared_ptr<renderer::Sampler>& repeat_texture_sampler,
        const std::vector<std::shared_ptr<renderer::ImageView>>& temp_tex,
        const std::shared_ptr<renderer::ImageView>& map_mask_tex,
        const std::shared_ptr<renderer::ImageView>& detail_volume_noise_tex,
        const std::shared_ptr<renderer::ImageView>& rough_volume_noise_tex);

    void createCameraDescSetWithTerrain(
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::BufferInfo& game_objects_buffer);

    void setVisibleTiles(const std::vector<std::shared_ptr<ego::TileObject>>& visible_tiles) {
        m_visible_tiles_ = visible_tiles;
    }

    void duplicateColorAndDepthBuffer(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    virtual void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const uint32_t& dbuf_idx,
        const int& input_key,
        const int& frame_count,
        const float& delta_t,
        const glm::vec2& last_mouse_pos,
        const float& mouse_wheel_offset,
        const bool& camera_rot_update);

    virtual void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& prt_desc_set,
        int dbuf_idx,
        float delta_t,
        float cur_time);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine