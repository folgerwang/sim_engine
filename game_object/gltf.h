#pragma once
#include <unordered_map>
#include "renderer/renderer.h"

namespace engine {
namespace game_object {

struct MaterialInfo {
    int32_t                base_color_idx_ = -1;
    int32_t                normal_idx_ = -1;
    int32_t                metallic_roughness_idx_ = -1;
    int32_t                emissive_idx_ = -1;
    int32_t                occlusion_idx_ = -1;

    renderer::BufferInfo   uniform_buffer_;
    std::shared_ptr<renderer::DescriptorSet>  desc_set_;
};

struct BufferView {
    uint32_t                buffer_idx;
    uint64_t                stride;
    uint64_t                offset;
    uint64_t                range;
};

union PrimitiveHashTag {
    uint32_t                data = 0;
    struct {
        uint32_t                has_normal : 1;
        uint32_t                has_tangent : 1;
        uint32_t                has_texcoord_0 : 1;
        uint32_t                has_skin_set_0 : 1;
        uint32_t                restart_enable : 1;
        uint32_t                topology : 16;
    };
};

struct PrimitiveInfo {
private:
    size_t hash_ = 0;
public:
    int32_t                 material_idx_;
    int32_t                 indirect_draw_cmd_ofs_;
    PrimitiveHashTag        tag_;
    glm::vec3               bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3               bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
    std::shared_ptr<renderer::AccelerationStructureGeometry>  as_geometry;

    renderer::IndexInputBindingDescription  index_desc_;
    std::vector<renderer::VertexInputBindingDescription> binding_descs_;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs_;

    void generateHash();
    size_t getHash() const { return hash_; }
};

struct BufferViewInfo {
    uint32_t                buffer_view_idx;
    uint64_t                offset;
    renderer::Format        format;
};

struct GltfData;
struct AnimChannelInfo {
    enum AnimChannelType {
        kTranslation,
        kRotation,
        kScale,
        kMaxNumChannels,
    };

    AnimChannelType         type_;
    uint32_t                node_idx_;
    BufferViewInfo          sample_buffer_;
    BufferViewInfo          data_buffer_;
    std::vector<std::pair<float, glm::vec4>>    samples_;

    void update(GltfData* object, float time, float time_scale = 1.0f, bool repeat = true);
};

struct AnimationInfo {
    std::vector<std::shared_ptr<AnimChannelInfo>> channels_;
};

struct SkinInfo {
    std::string             name_;
    int32_t                 skeleton_root_;
    std::vector<int32_t>    joints_;
    std::vector<glm::mat4>  inverse_bind_matrices_;
    renderer::BufferInfo    joints_buffer_;
    std::shared_ptr<renderer::DescriptorSet>    desc_set_;
};

struct MeshInfo {
    std::vector<PrimitiveInfo>  primitives_;
    glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
};

struct NodeInfo {
    std::string                 name_;
    int32_t                     parent_idx_ = -1;
    std::vector<int32_t>        child_idx_;

    int32_t                     mesh_idx_ = -1;
    int32_t                     skin_idx_ = -1;

    glm::vec3                   translation_{};
    glm::vec3                   scale_{1.0f};
    glm::quat                   rotation_{};
    glm::mat4                   matrix_ = glm::mat4(1.0f);

    glm::mat4                   cached_matrix_ = glm::mat4(1.0f);
    glm::mat4 getLocalMatrix();
    const glm::mat4& getCachedMatrix() const {
        return cached_matrix_;
    }
};

struct SceneInfo {
    std::vector<int32_t>        nodes_;
    glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::min());
};

struct GltfData {
    const std::shared_ptr<renderer::Device>& device_;
    int32_t                     default_scene_;
    std::vector<SceneInfo>      scenes_;
    std::vector<NodeInfo>       nodes_;
    std::vector<MeshInfo>       meshes_;
    std::vector<AnimationInfo>  animations_;
    std::vector<SkinInfo>       skins_;
    std::vector<renderer::BufferInfo>     buffers_;
    std::vector<BufferView>     buffer_views_;

    std::vector<renderer::TextureInfo>    textures_;
    std::vector<MaterialInfo>   materials_;

    uint32_t                    num_prims_ = 0;
    renderer::BufferInfo        indirect_draw_cmd_;
    renderer::BufferInfo        instance_buffer_;

    std::shared_ptr<renderer::DescriptorSet> indirect_draw_cmd_buffer_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> update_instance_buffer_desc_set_;

public:
    GltfData(const std::shared_ptr<renderer::Device>& device) : device_(device) {}
    ~GltfData() {}

    void update(
        const renderer::DeviceInfo& device_info,
        const uint32_t& active_anim_idx,
        const float& time);

    glm::mat4 getNodeMatrix(const int32_t& node_idx);

    void updateJoints(
        const renderer::DeviceInfo& device_info,
        int32_t node_idx);

    void generateSharedDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::DescriptorSetLayout>& gltf_indirect_draw_desc_set_layout,
        const std::shared_ptr<renderer::DescriptorSetLayout>& update_instance_buffer_desc_set_layout);

    void destroy();
};

class GltfObject {
    enum {
        kMaxNumObjects = 10240
    };
    const renderer::DeviceInfo& device_info_;
    std::shared_ptr<GltfData>   object_;
    uint32_t                    num_instances_ = 0;

    static std::shared_ptr<renderer::DescriptorSetLayout> material_desc_set_layout_;
    static std::shared_ptr<renderer::DescriptorSetLayout> skin_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> gltf_pipeline_layout_;
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> gltf_pipeline_list_;
    static std::unordered_map<std::string, std::shared_ptr<GltfData>> object_list_;
    static std::shared_ptr<renderer::DescriptorSetLayout> gltf_indirect_draw_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> gltf_indirect_draw_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> gltf_indirect_draw_pipeline_;
    static std::shared_ptr<renderer::DescriptorSetLayout> update_instance_buffer_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> update_instance_buffer_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> update_instance_buffer_pipeline_;

public:
    GltfObject() = delete;
    GltfObject(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& thin_film_lut_tex,
        const std::string& file_name,
        const glm::uvec2& display_size,
        const uint32_t& num_instances);

    void updateInstanceBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const uint32_t& num_instances);

    void updateIndirectDrawBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void updateBuffers(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const uint32_t& num_instances);

    void draw(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::shared_ptr<renderer::DescriptorSet>& game_objects_desc_set);

    void update(const renderer::DeviceInfo& device_info, const float& time);

    void destroy() {
        // object_ data has been cached, no need to be destroyed here.
/*        if (object_) {
            object_->destroy();
        }*/
    }

    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::DescriptorSetLayout>& game_object_desc_set_layout,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    static void createStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::DescriptorSetLayout>& game_object_desc_set_layout);

    static void recreateStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::DescriptorSetLayout>& game_object_desc_set_layout,
        const glm::uvec2& display_size);

    static void generateDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& thin_film_lut_tex);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static std::shared_ptr<GltfData> loadGltfModel(
        const renderer::DeviceInfo& device_info,
        const std::string& input_filename);
};

} // namespace game_object
} // namespace engine