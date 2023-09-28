#pragma once
#include "renderer/renderer.h"
#include "engine_helper.h"

namespace engine {
namespace game_object {

class ShapeBase {
    std::shared_ptr<renderer::BufferInfo> position_buffer_;
    std::shared_ptr<renderer::BufferInfo> uv_buffer_;
    std::shared_ptr<renderer::BufferInfo> normal_buffer_;
    std::shared_ptr<renderer::BufferInfo> tangent_buffer_;
    std::shared_ptr<renderer::BufferInfo> index_buffer_;
    std::vector<renderer::VertexInputBindingDescription> binding_descs_;
    std::vector<renderer::VertexInputAttributeDescription> attrib_descs_;

public:
    ShapeBase() {}

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

    void setBindingDescs(const std::vector<renderer::VertexInputBindingDescription>& binding_descs) {
        binding_descs_ = binding_descs;
    }

    void setAttribDescs(const std::vector<renderer::VertexInputAttributeDescription>& attrib_descs) {
        attrib_descs_ = attrib_descs;
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

    const std::vector<renderer::VertexInputBindingDescription>& getBindingDescs() {
        return binding_descs_;
    }

    const std::vector<renderer::VertexInputAttributeDescription>& getAttribDescs() {
        return attrib_descs_;
    }

    const std::shared_ptr<renderer::BufferInfo>& getIndexBuffer() {
        return index_buffer_;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device_);
};

} // game_object
} // engine