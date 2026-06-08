#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// scene_types.h — serializable scene authoring model.
//
// A Scene is an empty-by-default container with a world-space root transform
// (offset / rotation / scale).  Objects are imported from model files and held
// in a FLAT list; hierarchy is expressed through parent_index.  An imported
// file becomes a GROUP object (source_node_index == -1, is_group == true) and
// each renderable node inside the file becomes a child object
// (source_node_index >= 0, parent_index = the group's index).  That makes an
// FBX/glTF behave as a group whose individual sub-objects are independently
// selectable and movable, while still round-tripping to one source file.
//
// This header is intentionally free of any renderer/Vulkan types so the model
// can be (de)serialized and unit-reasoned about in isolation.  The live
// DrawableObject that backs each entry is owned by the application, NOT here.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace engine {
namespace scene {

// Decomposed transform: translation (offset), rotation (quaternion), scale.
struct Transform {
    glm::vec3 translation = glm::vec3(0.0f);
    // Identity quaternion.  glm::quat ctor order is (w, x, y, z).
    glm::quat rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale       = glm::vec3(1.0f);

    // Compose as T * R * S (scale first, then rotate, then translate).
    glm::mat4 toMatrix() const {
        const glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
        const glm::mat4 r = glm::mat4_cast(rotation);
        const glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }
};

// One entry in a scene's flat object list.
struct Object {
    std::string name;                 // display name (Outliner)
    std::string asset_path;           // source model file; shared by a group + its children
    int32_t     parent_index    = -1; // index into Scene::objects, or -1 for top-level
    int32_t     source_node_index = -1; // node within the asset; -1 = whole file / group
    bool        is_group        = false; // true for the imported-file root
    bool        visible         = true;
    Transform   transform;            // local to parent (world when parent_index == -1)
};

// The authored scene.  Default-constructed = empty scene named "Untitled" at
// the world origin with unit scale.
struct Scene {
    std::string         name = "Untitled";
    Transform           root;     // scene-level world offset / rotation / scale
    std::vector<Object> objects;

    // Background music: path to an audio clip (typically a generated .wav
    // under content/audio), looped on the music bus while the scene is
    // active.  Empty = no scene music.  Serialized from format v2 on.
    std::string         music_path;
    float               music_volume = 1.0f;
};

} // namespace scene
} // namespace engine
