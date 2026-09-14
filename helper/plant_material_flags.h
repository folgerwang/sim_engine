#pragma once
#include <cstdint>

namespace engine { namespace helper {
// Bits 16..31 belong to the half-float relief scale. Seasonal palettes
// use bits 9..14, with bit 15 identifying this non-overlapping encoding.
// Leaf sections are always MASK, never BLEND. For leaves only, the
// otherwise-unused BLEND bit extends the cohort index without colliding
// with the palette, leaf-mask marker or half-float depth scale.
inline uint32_t packLeafGroup(uint32_t group) { return ((group & 3u)<<6) | (group & 4u); }
inline uint32_t unpackLeafGroup(uint32_t flags) { return ((flags>>6)&3u) | (flags&4u); }
inline bool sectionIsBlend(uint32_t flags) { return (flags & 4u) && !(flags & 0x20u); }
constexpr uint32_t kSecPlantPaletteV2 = 0x8000u;
inline uint32_t packPlantPalette(uint32_t palette) {
    return kSecPlantPaletteV2 | ((palette & 63u) << 9);
}
inline uint32_t unpackPlantPalette(uint32_t flags) {
    if (!(flags & 0x20u)) return 0u; // kSecLeafAge
    return flags & kSecPlantPaletteV2 ? (flags >> 9) & 63u
                                      : (flags >> 20) & 1023u;
}
}}
