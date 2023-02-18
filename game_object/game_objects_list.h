#pragma once
#include <unordered_map>
#include "renderer/renderer.h"

namespace engine {
namespace game_object {

class GameObjectsList {
    enum {
        kMaxNumObjects = 10240
    };
    const renderer::DeviceInfo& device_info_;
    uint32_t                    num_objects_ = 0;

    std::shared_ptr<renderer::DescriptorSet> game_objects_update_buffer_desc_set_[2];
    std::shared_ptr<renderer::DescriptorSetLayout> game_objects_update_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> game_objects_update_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> game_objects_update_pipeline_;
    std::shared_ptr<renderer::DescriptorSetLayout> game_objects_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>  game_objects_desc_set_;

    std::shared_ptr<renderer::BufferInfo> game_objects_buffer_;
    std::shared_ptr<renderer::BufferInfo> game_objects_instance_buffer_;

public:
    GameObjectsList() = delete;
    GameObjectsList(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::TextureInfo& water_flow,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    void createGameObjectUpdateDescSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::TextureInfo& water_flow,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    void destroy(const std::shared_ptr<renderer::Device>& device);

    void updateGameObjectsBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::vec2& world_min,
        const glm::vec2& world_range,
        const glm::vec3& camera_pos,
        float air_flow_strength,
        float water_flow_strength,
        int update_frame_count,
        int soil_water,
        float delta_t,
        bool enble_airflow);

    inline std::shared_ptr<renderer::BufferInfo> getGameObjectsInstanceBuffer() {
        return game_objects_instance_buffer_;
    }

    inline std::shared_ptr<renderer::BufferInfo> getGameObjectsBuffer() {
        return game_objects_buffer_;
    }

    inline const std::shared_ptr<renderer::DescriptorSetLayout>& getGameObjectsDescSetLayout() {
        return game_objects_desc_set_layout_;
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getGameObjectsDescSet() {
        return game_objects_desc_set_;
    }
};

} // namespace game_object
} // namespace engine