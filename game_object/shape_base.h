#pragma once
#include "renderer/renderer.h"
#include "helper/engine_helper.h"

namespace engine {
namespace game_object {

class ShapeBase {
protected:
    std::shared_ptr<renderer::BufferInfo> position_buffer_;
    std::shared_ptr<renderer::BufferInfo> uv_buffer_;
    std::shared_ptr<renderer::BufferInfo> normal_buffer_;
    std::shared_ptr<renderer::BufferInfo> tangent_buffer_;
    std::shared_ptr<renderer::BufferInfo> index_buffer_;
    static std::vector<renderer::VertexInputBindingDescription> s_binding_descs_;
    static std::vector<renderer::VertexInputAttributeDescription> s_attrib_descs_;
    static std::shared_ptr<renderer::PipelineLayout> s_base_shape_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> s_base_shape_pipeline_;

public:
    ShapeBase() {}

    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::PipelineRenderbufferFormats& frame_buffer_format);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    void setPositionBuffer(const std::shared_ptr<renderer::BufferInfo>& buffer) {
        position_buffer_ = buffer;
    }

    void setUvBuffer(const std::shared_ptr<renderer::BufferInfo>& buffer) {
        uv_buffer_ = buffer;
    }

    void setNormalBuffer(const std::shared_ptr<renderer::BufferInfo>& buffer) {
        normal_buffer_ = buffer;
    }

    void setTangentBuffer(const std::shared_ptr<renderer::BufferInfo>& buffer) {
        tangent_buffer_ = buffer;
    }

    void setIndexBuffer(const std::shared_ptr<renderer::BufferInfo>& buffer) {
        index_buffer_ = buffer;
    }

    const std::shared_ptr<renderer::BufferInfo>& getPositionBuffer() {
        return position_buffer_;
    }

    const std::shared_ptr<renderer::BufferInfo>& getUvBuffer() {
        return uv_buffer_;
    }

    const std::shared_ptr<renderer::BufferInfo>& getNormalBuffer() {
        return normal_buffer_;
    }

    const std::shared_ptr<renderer::BufferInfo>& getTangentBuffer() {
        return tangent_buffer_;
    }

    static const std::vector<renderer::VertexInputBindingDescription>& getBindingDescs() {
        return s_binding_descs_;
    }

    static const std::vector<renderer::VertexInputAttributeDescription>& getAttribDescs() {
        return s_attrib_descs_;
    }

    const std::shared_ptr<renderer::BufferInfo>& getIndexBuffer() {
        return index_buffer_;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine