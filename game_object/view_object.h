#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "game_object/terrain.h"
#include "game_object/drawable_object.h"
#include "patch.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace game_object {

class ViewObject {
    std::shared_ptr<ego::ViewCamera> view_camera_;
    const std::shared_ptr<renderer::Device>& device_;
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool_;
    std::vector<std::shared_ptr<TileObject>> visible_tiles_;
    std::vector<std::shared_ptr<ego::DrawableObject>> visible_object_;

    er::Format hdr_format_ = er::Format::B10G11R11_UFLOAT_PACK32;
    er::Format depth_format_ = er::Format::D24_UNORM_S8_UINT;
    glm::uvec2 buffer_size_ = glm::uvec2(1280, 720);
    glm::vec3 camera_pos_ = glm::vec3(0.0f, 0.0f, 0.0f);

    er::TextureInfo hdr_color_buffer_;
    er::TextureInfo hdr_color_buffer_copy_;
    er::TextureInfo depth_buffer_;
    er::TextureInfo depth_buffer_copy_;

    bool render_grass_ = false;
    bool base_pass_ = false;

public:
    ViewObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool);

    void createCameraDescSetWithTerrain(
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::BufferInfo& game_objects_buffer);

    void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        int dbuf_idx,
        float delta_t,
        float cur_time);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine