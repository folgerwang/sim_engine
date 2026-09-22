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

// Compress ONE channel of an RGBA8 source into BC4_UNORM (8 bytes per
// 4x4 block -- half of BC7, and a quarter of raw R8).
//
// This exists for the cutout alpha layer.  Alpha used to ride in the
// albedo's BC7 alpha channel, where mode 6 shares a single index set
// between colour and alpha: the encoder had to spend that index on
// whichever of the two it judged more important, and at a leaf's
// cutout edge -- exactly where colour and alpha disagree most -- the
// result was alpha smeared across the silhouette.  Given its own BC4
// block, alpha gets 8-bit endpoints and 3-bit indices of its own, so
// the edge is as sharp as the source.  The visibility pass then reads
// 8 bytes per block instead of 16, and never touches the albedo pool
// at all.
//
// Output buffer must be at least:
//     ((width + 3) / 4) * ((height + 3) / 4) * 8   bytes
//
// Edge padding matches encodeBC7Mode6 / encodeBC5UNorm: blocks past
// the image right/bottom edge clamp to the rightmost/bottommost real
// texel.
//
// The source is described by a base pointer and a texel stride rather
// than a channel index, so one function serves both callers: the alpha
// channel of an RGBA8 buffer is (rgba + 3, stride 4), and a tightly
// packed single-channel plane -- which is how a .rwtex bake stores its
// cutout alpha -- is (plane, stride 1).
//
//   src        first byte of the channel's first texel, row-major.
//   src_stride bytes between consecutive texels (4 for RGBA8, 1 for a
//              packed single-channel plane).
//   width,
//   height     source dimensions in texels.
//   dst_bc4    destination buffer (caller-allocated).
void encodeBC4UNorm(
    const uint8_t* src,
    uint32_t       src_stride,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       dst_bc4);

}  // namespace scene_rendering
}  // namespace engine
