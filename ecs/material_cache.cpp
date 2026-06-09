// material_cache.cpp — see material.h / material_cache.h.
#include "ecs/material_cache.h"

#include <cstring>

namespace engine {
namespace ecs {

namespace {
inline void mix(size_t& h, size_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);  // boost-style combine
}
inline size_t hashF(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<size_t>(u);
}
}  // namespace

size_t MaterialDesc::hash() const {
    size_t h = 1469598103934665603ull;
    for (int i = 0; i < 4; ++i) mix(h, hashF(base_color[i]));
    for (int i = 0; i < 3; ++i) mix(h, hashF(emissive[i]));
    mix(h, hashF(metallic));
    mix(h, hashF(roughness));
    mix(h, hashF(alpha_cutoff));
    mix(h, static_cast<size_t>(alpha_mode));
    std::hash<std::string> hs;
    mix(h, hs(base_color_tex));
    mix(h, hs(normal_tex));
    mix(h, hs(metallic_roughness_tex));
    mix(h, hs(emissive_tex));
    mix(h, hs(occlusion_tex));
    return h;
}

MaterialId MaterialCache::intern(const MaterialDesc& desc) {
    const size_t h = desc.hash();
    auto range = by_hash_.equal_range(h);
    for (auto it = range.first; it != range.second; ++it) {
        const MaterialId id = it->second;
        Entry& e = entries_[id];
        if (e.live && e.desc == desc) {  // exact match: dedup hit
            ++e.refs;
            return id;
        }
    }
    // New material: reuse a freed slot if available, else append.
    MaterialId id;
    if (!free_list_.empty()) {
        id = free_list_.back();
        free_list_.pop_back();
        entries_[id] = Entry{desc, 1, true};
    } else {
        id = static_cast<MaterialId>(entries_.size());
        entries_.push_back(Entry{desc, 1, true});
    }
    by_hash_.emplace(h, id);
    ++live_count_;
    return id;
}

void MaterialCache::addRef(MaterialId id) {
    if (id < entries_.size() && entries_[id].live) ++entries_[id].refs;
}

void MaterialCache::release(MaterialId id) {
    if (id >= entries_.size() || !entries_[id].live) return;
    Entry& e = entries_[id];
    if (--e.refs > 0) return;

    // Refcount hit zero: free the slot and drop its hash-map entry.
    const size_t h = e.desc.hash();
    auto range = by_hash_.equal_range(h);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == id) { by_hash_.erase(it); break; }
    }
    e.live = false;
    e.desc = MaterialDesc{};   // release any string storage
    free_list_.push_back(id);
    --live_count_;
}

const MaterialDesc* MaterialCache::get(MaterialId id) const {
    if (id >= entries_.size() || !entries_[id].live) return nullptr;
    return &entries_[id].desc;
}

uint32_t MaterialCache::refCount(MaterialId id) const {
    if (id >= entries_.size() || !entries_[id].live) return 0;
    return entries_[id].refs;
}

}  // namespace ecs
}  // namespace engine
