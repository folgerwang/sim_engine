#pragma once
// material_cache.h — refcounted interning cache for MaterialDesc.
//
// intern(desc) returns a stable MaterialId; identical descriptions collapse to
// the same id (and bump a refcount). release(id) drops a reference and frees the
// entry at zero, so the GC layer can reclaim materials whose last user went
// away. Renderer-agnostic and unit-tested; the engine pairs each live id with
// its uploaded GPU material (textures/descriptor set) in a parallel table.
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "ecs/material.h"

namespace engine {
namespace ecs {

class MaterialCache {
public:
    // Return the id for `desc`, creating it if new. Increments the refcount.
    MaterialId intern(const MaterialDesc& desc);

    void     addRef(MaterialId id);
    // Drop a reference; frees the entry (and its id) when the count hits zero.
    void     release(MaterialId id);

    const MaterialDesc* get(MaterialId id) const;  // nullptr if freed/invalid
    uint32_t refCount(MaterialId id) const;
    size_t   liveCount() const { return live_count_; }

private:
    struct Entry {
        MaterialDesc desc;
        uint32_t     refs = 0;
        bool         live = false;
    };
    std::vector<Entry>                             entries_;
    std::unordered_multimap<size_t, MaterialId>    by_hash_;   // hash -> id(s)
    std::vector<MaterialId>                        free_list_; // reclaimed slots
    size_t                                         live_count_ = 0;
};

}  // namespace ecs
}  // namespace engine
