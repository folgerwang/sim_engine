//
// bc7_encoder.cpp — RGBA8 → BC7 wrapper around Rich Geldreich's bc7enc.
//
// Replaces an earlier in-tree minimal Mode-6 encoder that produced
// visually-wrong colours on the GPU.  The streamer + page table were
// independently verified by switching the ALBEDO pool to RGBA8_UNORM —
// rendering was flawless — so the bug was localised to the encoder.
//
// We use bc7enc with default-quality params (mode 1 + mode 6, perceptual
// YCbCr weights, least-squares endpoint refinement).  Per-block cost on
// modern CPUs is tens of microseconds; the existing encode_pool_
// thread pool parallelises across all entries in a VT.
//
// API contract (encodeBC7Mode6 — the name is now historic; bc7enc may
// pick mode 1 OR mode 6 depending on the block content):
//   * Width / height clamped to image edge for trailing partial blocks.
//   * Output buffer must be at least ((w+3)/4)*((h+3)/4)*16 bytes.
//   * src_rgba is tightly packed RGBA8, row-major, R first in memory.
//

#include "bc7_encoder.h"

// bc7enc.h has its own `extern "C"` guards, so no external wrapping
// needed.  It also pulls in <stdlib.h> + <stdint.h> at file scope —
// don't wrap those in extern "C" by accident.
#include "third_parties/bc7enc/bc7enc.h"

#include <algorithm>
#include <cstring>
#include <mutex>

namespace engine {
namespace scene_rendering {

namespace {

// bc7enc_compress_block_init() builds large internal lookup tables and
// MUST be called exactly once before any compress_block call.  Guard
// with std::call_once so multiple VTs encoding in parallel during
// scene load are safe.
std::once_flag g_bc7enc_init_flag;
void ensureBc7EncInit() {
    std::call_once(g_bc7enc_init_flag, []() {
        bc7enc_compress_block_init();
    });
}

}  // namespace

void encodeBC7Mode6(
    const uint8_t* src_rgba,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       dst_bc7) {

    if (!src_rgba || !dst_bc7 || width == 0 || height == 0) return;

    ensureBc7EncInit();

    // Compression params — defaults give best quality (mode 1+6 with
    // perceptual weights and least-squares refinement).  For a faster
    // lower-quality preset, set m_max_partitions_mode = 0 and disable
    // m_try_least_squares.
    bc7enc_compress_block_params params;
    bc7enc_compress_block_params_init(&params);

    const uint32_t blocks_x = (width  + 3u) / 4u;
    const uint32_t blocks_y = (height + 3u) / 4u;

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            // Gather 4×4 RGBA pixels with edge clamp for blocks that
            // straddle the image right/bottom edge.
            uint8_t pixels[16 * 4];
            for (int py = 0; py < 4; ++py) {
                uint32_t y = std::min(by * 4u + uint32_t(py), height - 1u);
                for (int px = 0; px < 4; ++px) {
                    uint32_t x = std::min(bx * 4u + uint32_t(px), width - 1u);
                    const uint8_t* s = src_rgba + (size_t(y) * width + x) * 4u;
                    uint8_t*       d = pixels + (size_t(py) * 4 + px) * 4u;
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                }
            }

            uint8_t* block = dst_bc7 + (size_t(by) * blocks_x + bx) * 16u;
            bc7enc_compress_block(block, pixels, &params);
        }
    }
}

// ── BC4 single-channel block encode ────────────────────────────────
// Mode-6 layout (the simpler of BC4's two modes): two 8-bit endpoints
// where ep0 > ep1, six implicit interpolated values between, plus
// indices 6 and 7 reserved for the constants 0 and 255.  We use the
// 8-interp mode by always storing min as ep1 and max as ep0 — BC4
// auto-selects the 8-interp path when ep0 > ep1.  Block layout:
//   bytes 0..1: ep0, ep1
//   bytes 2..7: 16 × 3-bit indices, packed LSB-first into 48 bits
namespace {
inline void encodeBC4Block(const uint8_t v[16], uint8_t* dst8) {
    uint8_t mn = 255u, mx = 0u;
    for (int i = 0; i < 16; ++i) {
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
    }
    // Degenerate (single-value) block: ep0 == ep1, all indices 0.
    if (mn == mx) {
        dst8[0] = mx; dst8[1] = mx;
        std::memset(dst8 + 2, 0, 6);
        return;
    }
    // 8-interp mode: store ep0 = max, ep1 = min so ep0 > ep1.
    // Decoded values for indices 0..7:
    //   0: ep0 (=max),     1: ep1 (=min),
    //   2..7: lerp(ep0, ep1, k/7) for k=1..6
    // Encode each pixel by quantising its position along (max - min).
    const uint8_t ep0 = mx, ep1 = mn;
    const int range = int(ep0) - int(ep1);     // > 0 (guaranteed by branch)
    uint64_t bits = 0;
    for (int i = 0; i < 16; ++i) {
        // proj ∈ [0, 1] where 0 → ep0, 1 → ep1.
        // BC4 8-interp index mapping (per spec):
        //   idx 0   → ep0
        //   idx 1   → ep1
        //   idx 2..7 → (6-(idx-1))/7·ep0 + (idx-1)/7·ep1
        // So the natural lerp t ∈ [0..1] maps to:
        //   t = 0   → idx 0
        //   t = 1   → idx 1
        //   else    → idx = 1 + round(t * 7) ... not quite right.
        // Easier: enumerate all 8 weights and pick the closest.
        const int weights[8] = {
            int(ep0),                                                  // 0
            int(ep1),                                                  // 1
            (6 * int(ep0) + 1 * int(ep1) + 3) / 7,                     // 2
            (5 * int(ep0) + 2 * int(ep1) + 3) / 7,                     // 3
            (4 * int(ep0) + 3 * int(ep1) + 3) / 7,                     // 4
            (3 * int(ep0) + 4 * int(ep1) + 3) / 7,                     // 5
            (2 * int(ep0) + 5 * int(ep1) + 3) / 7,                     // 6
            (1 * int(ep0) + 6 * int(ep1) + 3) / 7,                     // 7
        };
        int  best_idx = 0;
        int  best_err = std::abs(int(v[i]) - weights[0]);
        for (int k = 1; k < 8; ++k) {
            int err = std::abs(int(v[i]) - weights[k]);
            if (err < best_err) { best_err = err; best_idx = k; }
        }
        bits |= (uint64_t(best_idx) & 0x7ull) << (i * 3);
        (void)range;
    }
    dst8[0] = ep0; dst8[1] = ep1;
    dst8[2] = uint8_t( bits        & 0xFFu);
    dst8[3] = uint8_t((bits >>  8) & 0xFFu);
    dst8[4] = uint8_t((bits >> 16) & 0xFFu);
    dst8[5] = uint8_t((bits >> 24) & 0xFFu);
    dst8[6] = uint8_t((bits >> 32) & 0xFFu);
    dst8[7] = uint8_t((bits >> 40) & 0xFFu);
}
}  // namespace

void encodeBC5UNorm(
    const uint8_t* src_rgba,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       dst_bc5) {

    if (!src_rgba || !dst_bc5 || width == 0 || height == 0) return;

    const uint32_t blocks_x = (width  + 3u) / 4u;
    const uint32_t blocks_y = (height + 3u) / 4u;

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            // Gather R and G channels of 16 pixels into separate
            // arrays.  Edge clamp matches the BC7 path so blocks
            // touching the image boundary repeat the rightmost /
            // bottommost real texel.
            uint8_t r[16], g[16];
            for (int py = 0; py < 4; ++py) {
                uint32_t y = std::min(by * 4u + uint32_t(py), height - 1u);
                for (int px = 0; px < 4; ++px) {
                    uint32_t x = std::min(bx * 4u + uint32_t(px), width - 1u);
                    const uint8_t* s = src_rgba + (size_t(y) * width + x) * 4u;
                    r[py * 4 + px] = s[0];
                    g[py * 4 + px] = s[1];
                }
            }
            // BC5 = BC4(R) || BC4(G), 8 + 8 = 16 bytes per block.
            uint8_t* block = dst_bc5 + (size_t(by) * blocks_x + bx) * 16u;
            encodeBC4Block(r, block);
            encodeBC4Block(g, block + 8);
        }
    }
}

}  // namespace scene_rendering
}  // namespace engine
