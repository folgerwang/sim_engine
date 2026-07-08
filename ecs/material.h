#pragma once
// material.h — renderer-agnostic material description + reference component.
//
// A MaterialDesc is the authored PBR material (metallic-roughness model, the
// glTF default) with textures referenced by ASSET KEY (path/string), not GPU
// handles — so it lives in the renderer-free core and can be deduplicated and
// hashed. Many imported sub-meshes share identical materials; MaterialRef lets
// an entity point at one interned MaterialDesc (see MaterialCache), collapsing
// duplicates to a single entry the engine uploads once.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace engine {
namespace ecs {

enum class AlphaMode : uint8_t { kOpaque, kMask, kBlend };

struct MaterialDesc {
    glm::vec4   base_color   = glm::vec4(1.0f);
    glm::vec3   emissive     = glm::vec3(0.0f);
    float       metallic     = 1.0f;
    float       roughness    = 1.0f;
    float       alpha_cutoff = 0.5f;
    AlphaMode   alpha_mode   = AlphaMode::kOpaque;

    std::string base_color_tex;          // "" = none
    std::string normal_tex;
    std::string metallic_roughness_tex;
    std::string emissive_tex;
    std::string occlusion_tex;

    bool operator==(const MaterialDesc& o) const {
        return base_color == o.base_color && emissive == o.emissive &&
               metallic == o.metallic && roughness == o.roughness &&
               alpha_cutoff == o.alpha_cutoff && alpha_mode == o.alpha_mode &&
               base_color_tex == o.base_color_tex &&
               normal_tex == o.normal_tex &&
               metallic_roughness_tex == o.metallic_roughness_tex &&
               emissive_tex == o.emissive_tex &&
               occlusion_tex == o.occlusion_tex;
    }
    bool operator!=(const MaterialDesc& o) const { return !(*this == o); }

    // Stable content hash (for the dedup cache).
    size_t hash() const;
};

using MaterialId = uint32_t;
inline constexpr MaterialId kInvalidMaterial = 0xFFFFFFFFu;

// Component: the interned material an entity / sub-mesh renders with.
struct MaterialRef {
    MaterialId id = kInvalidMaterial;
};

// Component: ALL interned materials an entity's drawable renders with, in
// sub-material order (index-matched with DrawableData::materials_). Used
// while DrawableObject is still a monolith (pre-shred): one entity owns one
// drawable which owns N materials, so the entity carries the N interned ids
// and the GC cleanup hook release()s each when the entity dies.
struct MaterialSet {
    std::vector<MaterialId> ids;
};

}  // namespace ecs
}  // namespace engine
