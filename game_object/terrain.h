#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace game_object {

class TileObject {
    const renderer::DeviceInfo& device_info_;

    enum class TileConst{
        kRockLayerSize = 8192,
        kSoilLayerSize = 4096,
        kWaterlayerSize = 4096,
        kGrassSnowLayerSize = 2048,
        kCacheTileSize = 12,
        kVisibleTileSize = 10,
        kMinNumGrass = 1024,
        kMaxNumGrass = 8192,
        kSegmentCount = 32 - 1,
        kNumCachedBlocks = (kCacheTileSize * 2 + 1) * (kCacheTileSize * 2 + 1)
    };

    bool created = false;
    size_t hash_ = ~0x00;
    uint32_t block_idx_ = ~0x00;
    glm::ivec4  neighbors_;
    glm::vec2 min_;
    glm::vec2 max_;

    renderer::BufferInfo index_buffer_;

    // grass etc.
    renderer::BufferInfo grass_vertex_buffer_;
    renderer::BufferInfo grass_index_buffer_;
    renderer::BufferInfo grass_indirect_draw_cmd_;
    renderer::BufferInfo grass_instance_buffer_;

    static std::unordered_map<size_t, std::shared_ptr<TileObject>> tile_meshes_;
    static std::vector<std::shared_ptr<TileObject>> visible_tiles_;
    static std::vector<uint32_t> available_block_indexes_;
    static renderer::TextureInfo rock_layer_;
    static renderer::TextureInfo soil_water_layer_[2];
    static renderer::TextureInfo grass_snow_layer_;
    static renderer::TextureInfo water_normal_;
    static renderer::TextureInfo water_flow_;
    static std::shared_ptr<renderer::DescriptorSet> creator_buffer_desc_set_;
    static std::shared_ptr<renderer::DescriptorSet> tile_update_buffer_desc_set_[2]; // soil and water double buffer.
    static std::shared_ptr<renderer::DescriptorSet> tile_flow_update_buffer_desc_set_[2]; // soil and water double buffer.
    static std::shared_ptr<renderer::DescriptorSetLayout> tile_creator_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> tile_creator_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> tile_creator_pipeline_;
    static std::shared_ptr<renderer::DescriptorSetLayout> tile_update_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> tile_update_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> tile_update_pipeline_;
    static std::shared_ptr<renderer::DescriptorSetLayout> tile_flow_update_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> tile_flow_update_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> tile_flow_update_pipeline_;
    static std::shared_ptr<renderer::PipelineLayout> tile_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> tile_pipeline_;
    static std::shared_ptr<renderer::DescriptorSetLayout> tile_res_desc_set_layout_;
    static std::shared_ptr<renderer::DescriptorSet> tile_res_desc_set_[2];
    static std::shared_ptr<renderer::Pipeline> tile_water_pipeline_;
    static std::shared_ptr<renderer::Pipeline> tile_grass_pipeline_;

public:
    TileObject() = delete;
    TileObject(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
        const glm::vec2& min,
        const glm::vec2& max,
        const size_t& hash_value,
        const uint32_t& block_idx);

    ~TileObject() {
        destory();
    }

    inline size_t getHash() { return hash_; }

    inline void setNeighbors(const glm::ivec4& neighbors) {
        neighbors_ = neighbors;
    }

    void destory();

    static const renderer::TextureInfo& getRockLayer();
    static const renderer::TextureInfo& getSoilWaterLayer(int idx);
    static const renderer::TextureInfo& getWaterFlow();
    static glm::vec2 getWorldMin();
    static glm::vec2 getWorldRange();
    glm::vec2 getMin() { return min_; }
    glm::vec2 getMax() { return max_; }

    float getMinDistToCamera(const glm::vec2& camera_pos);

    static std::shared_ptr<TileObject> addOneTile(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
        const glm::vec2& min,
        const glm::vec2& max);

    static void initStaticMembers(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::RenderPass>& water_render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::GraphicPipelineInfo& graphic_double_face_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void createStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::RenderPass>& water_render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::GraphicPipelineInfo& graphic_double_face_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void recreateStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::RenderPass>& water_render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::GraphicPipelineInfo& graphic_double_face_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void destoryStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static void updateStaticDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& clamp_texture_sampler,
        const std::shared_ptr<renderer::Sampler>& repeat_texture_sampler,
        const std::shared_ptr<renderer::ImageView>& src_texture,
        const std::shared_ptr<renderer::ImageView>& src_depth,
        const std::vector<std::shared_ptr<renderer::ImageView>>& temp_tex,
        const std::shared_ptr<renderer::ImageView>& heightmap_tex,
        const std::shared_ptr<renderer::ImageView>& map_mask_tex,
        const std::shared_ptr<renderer::ImageView>& detail_volume_noise_tex,
        const std::shared_ptr<renderer::ImageView>& rough_volume_noise_tex);

    static void generateStaticDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& heightmap_tex);

    static void generateAllDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& heightmap_tex);

    static void generateTileBuffers(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    static void updateTileBuffers(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        float current_time,
        int dbuf_idx);

    static void updateTileFlowBuffers(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        float current_time,
        int dbuf_idx);

    static void drawAllVisibleTiles(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const glm::vec2& camera_pos,
        const glm::uvec2& display_size,
        int dbuf_idx,
        float delta_t,
        float cur_time,
        bool is_base_pass,
        bool render_grass = false);

    static void updateAllTiles(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
        const float& tile_size,
        const glm::vec2& camera_pos);

    static void destoryAllTiles();

    bool validTileBySize(
        const glm::ivec2& min_tile_idx,
        const glm::ivec2& max_tile_idx,
        const float& tile_size);

    void createMeshBuffers();

    void createGrassBuffers();
        
    void draw(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const glm::uvec2 display_size,
        int dbuf_idx,
        float delta_t,
        float cur_time,
        bool is_base_pass);

    void drawGrass(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const glm::vec2& camera_pos,
        const glm::uvec2& display_size,
        int dbuf_idx,
        float delta_t,
        float cur_time);
};

} // namespace game_object
} // namespace engine