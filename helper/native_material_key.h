#pragma once
#include <string>
#include <cstddef>
namespace engine::helper {
// Geometry sections and LOD nodes can share one material. Key the actual
// render inputs, not the section ordinal or source filename.
template<class Section, class Textures>
inline std::string nativeMaterialKey(const Section& sec,
        const Textures& textures,size_t base) {
    std::string key;
    auto append=[&](const auto& value) {key.append(reinterpret_cast<const char*>(&value),sizeof(value));};
    append(sec.base_color); append(sec.metallic); append(sec.roughness);
    append(sec.flags); append(sec.triplanar_tile_m);
    for(int index : {sec.tex_index,sec.nrm_index,sec.mr_index}) {
        const auto* image=(index>=0 && base+size_t(index)<textures.size()) ? textures[base+size_t(index)].image.get():nullptr;
        append(image);
        // No GPU image means missing/VT-only: retain source identity so
        // different unavailable textures are not accidentally merged.
        if(!image && index>=0 && base+size_t(index)<textures.size()) {
            const auto& path=textures[base+size_t(index)].source_filename_;
            const size_t length=path.size(); append(length); key.append(path);
        } else { const size_t length=0; append(length); }
    }
    return key;
}

}
