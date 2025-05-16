#include "box.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {
}

namespace game_object {
Box::Box(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<std::array<glm::vec3, 8>> v,
    uint32_t split_num_x,
    uint32_t split_num_y,
    uint32_t split_num_z,
    const std::source_location& src_location) {

    if (v == nullptr) {
        v = std::make_shared<std::array<glm::vec3, 8>>();
        (*v)[0] = glm::vec3(-1.0f, -1.0f, -1.0f);
        (*v)[1] = glm::vec3(1.0f, -1.0f, -1.0f);
        (*v)[2] = glm::vec3(1.0f, -1.0f, 1.0f);
        (*v)[3] = glm::vec3(-1.0f, -1.0f, 1.0f);
        (*v)[4] = glm::vec3(-1.0f, 1.0f, -1.0f);
        (*v)[5] = glm::vec3(1.0f, 1.0f, -1.0f);
        (*v)[6] = glm::vec3(1.0f, 1.0f, 1.0f);
        (*v)[7] = glm::vec3(-1.0f, 1.0f, 1.0f);
    }

    planes_[0] =
        std::make_shared<Plane>(
            device,
            std::make_shared<std::array<glm::vec3, 4>>(std::array<glm::vec3, 4>{(*v)[0], (*v)[1], (*v)[2], (*v)[3]}),
            split_num_x,
            split_num_z,
            src_location);

    planes_[1] =
        std::make_shared<Plane>(
            device,
            std::make_shared<std::array<glm::vec3, 4>>(std::array<glm::vec3, 4>{(*v)[5], (*v)[4], (*v)[7], (*v)[6]}),
            split_num_x,
            split_num_z,
            src_location);

    planes_[2] =
        std::make_shared<Plane>(
            device,
            std::make_shared<std::array<glm::vec3, 4>>(std::array<glm::vec3, 4>{(*v)[1], (*v)[0], (*v)[4], (*v)[5]}),
            split_num_x,
            split_num_y,
            src_location);

    planes_[3] =
        std::make_shared<Plane>(
            device,
            std::make_shared<std::array<glm::vec3, 4>>(std::array<glm::vec3, 4>{(*v)[3], (*v)[2], (*v)[6], (*v)[7]}),
            split_num_x,
            split_num_y,
            src_location);

    planes_[4] =
        std::make_shared<Plane>(
            device,
            std::make_shared<std::array<glm::vec3, 4>>(std::array<glm::vec3, 4>{(*v)[4], (*v)[0], (*v)[3], (*v)[7]}),
            split_num_y,
            split_num_z,
            src_location);

    planes_[5] =
        std::make_shared<Plane>(
            device,
            std::make_shared<std::array<glm::vec3, 4>>(std::array<glm::vec3, 4>{(*v)[1], (*v)[5], (*v)[6], (*v)[2]}),
            split_num_y,
            split_num_z,
            src_location);
}

void Box::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {
    for (auto& plane : planes_) {
        plane->draw(cmd_buf, viewports, scissors);
    }
}

void Box::destroy(const std::shared_ptr<renderer::Device>& device) {
    for (auto& plane : planes_) {
        plane->destroy(device);
    }
}

} // game_object
} // engine