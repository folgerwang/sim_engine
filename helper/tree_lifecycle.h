#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
namespace engine { namespace helper {
inline uint32_t treePaletteIndex(const std::string& name) {
    static const struct { const char* name; uint32_t index; } names[] = {
#define TREE_PALETTE(name, index) {name,index},
#include "tree_age_palette_names.inc"
#undef TREE_PALETTE
    };
    const bool grass=name.find("clutter_grass")!=std::string::npos || name.find("clutter_far_")!=std::string::npos;
    if(grass) for(const auto& entry:names) if(std::string(entry.name)=="cluttergrass") return entry.index;
    uint32_t best=0; size_t length=0;
    for (uint32_t i=1; i<sizeof(names)/sizeof(names[0]); ++i) {
        const std::string token(names[i].name);
        const size_t at=name.find(token);
        if (token.size()>length && at!=std::string::npos &&
            (at==0 || name[at-1]=='_') &&
            (at+token.size()==name.size() || name[at+token.size()]=='_')) {
            best=names[i].index; length=token.size();
        }
    }
    return best;
}
struct TreeDryColor { float r,g,b; };
inline TreeDryColor treeDryColor(uint32_t palette) {
    static const TreeDryColor colors[] = {
#define TREE_DRY(r,g,b) {r,g,b},
#include "tree_age_dry_colors.inc"
#undef TREE_DRY
    };
    return colors[std::min(palette,uint32_t(sizeof(colors)/sizeof(colors[0])-1))];
}
inline uint32_t plantSeasonMode(uint32_t palette) {
    static const uint32_t modes[] = {
#define PLANT_SEASON_MODE(mode) mode,
#include "plant_season_modes.inc"
#undef PLANT_SEASON_MODE
    };
    return modes[std::min(palette,uint32_t(sizeof(modes)/sizeof(modes[0])-1))];
}
inline bool plantShedCrossing(double before,double now,float seed,uint32_t group) {
    const double offset=std::clamp(-double(seed),0.,100.)*.0003+std::min(group,3u)*.012;
    // Each cohort detaches as its autumn coverage reaches zero, every year.
    return now>before && std::floor(now*.01-offset-.74)>std::floor(before*.01-offset-.74);
}
struct TreeLifeProfile { const char* family; float minimum, maximum; };
inline TreeLifeProfile treeLifeProfile(const std::string& species) {
    static const TreeLifeProfile profiles[] = {
#define TREE_LIFE(name, lo, hi) {name, float(lo), float(hi)},
#include "tree_lifecycle_profiles.inc"
#undef TREE_LIFE
    };
    for (const auto& p : profiles)
        if (species.find(p.family) != std::string::npos) return p;
    return profiles[sizeof(profiles)/sizeof(profiles[0])-1];
}
inline uint32_t treeLifeHash(uint64_t id, uint32_t salt) {
    uint32_t x = uint32_t(id) ^ uint32_t(id >> 32) ^ salt;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
inline float treeInitialAgeBase(uint64_t id) {
    return -float(treeLifeHash(id, 0xa341316cu) % 101u);
}
inline float treeInstanceLifespan(uint64_t id, const std::string& species) {
    const auto p = treeLifeProfile(species);
    const float u = float(treeLifeHash(id, 0xc8013ea4u) & 0xffffffu) / 16777215.f;
    return p.minimum + (p.maximum-p.minimum)*u;
}
enum class TreeLifeStage : uint8_t { Young, Mature, Old, Dead };
inline double treeAgeYears(double year, double birth) { return year+birth; }
inline TreeLifeStage treeLifeStage(double age, double lifespan) {
    if (age >= lifespan) return TreeLifeStage::Dead;
    if (age >= lifespan*.8) return TreeLifeStage::Old;
    if (age >= lifespan*.2) return TreeLifeStage::Mature;
    return TreeLifeStage::Young;
}
struct LeafLifeState {
    double age;
    float lifespan;
    bool expired;
};
inline float treeAgingRatio(const std::string& species) {
    const auto p = treeLifeProfile(species);
    return std::clamp(100.f / ((p.minimum+p.maximum)*.5f), .1f, 100.f);
}
inline LeafLifeState treeLeafLife(double tree_age_base, double global_age_time,
    double species_aging_ratio, float initial_leaf_age, float leaf_lifespan=100.f) {
    const double tree_age = std::max(tree_age_base + global_age_time *
        std::clamp(species_aging_ratio, .1, 100.), 0.0);
    const double age = std::max(tree_age + initial_leaf_age, 0.0);
    return {age, leaf_lifespan, age >= leaf_lifespan};
}
}} // namespace engine::helper
