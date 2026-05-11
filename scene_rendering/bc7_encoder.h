#pragma once
//
// bc7_encoder.h — minimal BC7 (Mode 6 only) encoder.
//
// Compresses RGBA8 → BC7 blocks for the Runtime Virtual Texture's
// albedo layer.  Mode 6 is the simplest BC7 mode that supports an
// alpha channel: single subset, two RGBA endpoints (7-bit each + a
// shared p-bit per endpoint), 16 4-bit per-pixel weight indices.
//
// Quality: clearly below a production encoder (no PCA endpoint search,
// no mode-selection across 8 modes, no rate-distortion optimisation),
// but produces VALID BC7 the GPU's hardware decompressor accepts.
// Each block costs 16 bytes vs RGBA8's 64 bytes — a 4× memory win for
// the pool.
//
// Speed: roughly 200-500 ms per 1024² texture in -O2 build.  Fine for
// scene-load-time; would need SIMD or a compute-shader encoder for
// per-frame use.
//

#include <cstdint>

namespace engine {
namespace scene_rendering {

// Compress an entire RGBA8 image into BC7 Mode 6.  The output buffer
// must be at least:
//     ((width  + 3) / 4) * ((height + 3) / 4) * 16   bytes
//
// Edges where width/height aren't multiples of 4 are CLAMPED — the
// missing pixels in the trailing block(s) are filled with the source's
// edge texels so the BC7 decoder doesn't pull garbage.
//
//   src_rgba   tightly-packed RGBA8 source, row-major.
//   width,
//   height     source dimensions in texels.
//   dst_bc7    destination buffer (caller-allocated).
void encodeBC7Mode6(
    const uint8_t* src_rgba,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       dst_bc7);

// Compress a 2-channel (RG) image into BC5_UNORM (= 2× BC4 channels
// concatenated, 16 bytes per 4×4 block).  Reads the R and G channels
// from a tightly-packed RGBA8 source (B and A are ignored — caller
// is responsible for placing the two normal channels in R + G).
//
// Output buffer must be at least:
//     ((width + 3) / 4) * ((height + 3) / 4) * 16   bytes
//
// Edge padding is the same as encodeBC7Mode6: blocks past the image
// right/bottom edge clamp to the rightmost/bottommost real texel.
//
// BC5 is the canonical hardware-decoded format for tangent-space
// normal maps — RG-only with 8-bit endpoints + 3-bit per-texel
// weights gives ~4× memory saving over RGBA8 with no visible quality
// loss for typical normal map content (smooth angle distributions).
//
//   src_rgba   tightly-packed RGBA8 source, row-major.
//   width,
//   height     source dimensions in texels.
//   dst_bc5    destination buffer (caller-allocated).
void encodeBC5UNorm(
    const uint8_t* src_rgba,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       dst_bc5);

}  // namespace scene_rendering
}  // namespace engine
