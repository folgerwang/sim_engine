#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>

namespace engine::scene_rendering {
// Rest geometry is shared between instances. Only palettes/transforms change
// each frame; no caller needs to expand vertices into world space.
struct RtSkinBatch {
    bool world_space_dynamic = false;
    const std::vector<glm::vec3>* positions = nullptr;
    const std::vector<glm::u16vec4>* joints = nullptr;
    const std::vector<glm::vec4>* weights = nullptr;
    const std::vector<glm::u16vec4>* joints1 = nullptr;
    const std::vector<glm::vec4>* weights1 = nullptr;
    const std::vector<uint32_t>* indices = nullptr;
    const std::vector<glm::mat4>* joint_matrices = nullptr;
    glm::mat4 model{1.0f};
    std::vector<glm::mat4> palette;
    // 0: rigid, 1: joint skinning, 2: citizen taper/pivot blend,
    // 3: NPC skinning (unweighted vertices follow the first joint).
    uint32_t deformation = 0;
    glm::vec4 shape{0.0f};
};
}
