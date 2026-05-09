//
// bc7_encoder.cpp — BC7 Mode 6 RGBA encoder (simplified).
//
// Algorithm per 4×4 block:
//   1. Endpoints = component-wise min and max of the 16 pixels.  This
//      is an axis-aligned bounding-box endpoint heuristic — not as good
//      as PCA but cheap and adequate for diffuse content.
//   2. Quantise each endpoint channel to 7+1 bits, picking the p-bit
//      that minimises round-trip error per channel; majority-vote a
//      single p-bit per endpoint (Mode 6 has ONE shared p-bit per
//      endpoint, not per channel).
//   3. For each pixel, project onto the endpoint axis, divide by
//      axis-length-squared to get t ∈ [0, 1], round to the nearest
//      4-bit index 0..15.  Note: this produces a UNIFORM-spaced index,
//      which mismatches BC7's non-uniform decode weights table by ~4%
//      at extremes — visible only side-by-side.  An optimal encoder
//      would search all 16 weights per pixel.
//   4. Anchor the high bit of index[0] to 0 by swapping endpoints +
//      inverting indices when needed (Mode 6 anchor convention).
//   5. Bit-pack: 7 mode + 56 endpoint + 2 p-bit + 3 anchor + 60 index = 128.
//

#include "bc7_encoder.h"

#include <algorithm>
#include <cstring>

namespace engine {
namespace scene_rendering {

namespace {

// Decoder weight table for 4-bit indices in Mode 6.  Decoded color =
// (ep0 * (64 - w) + ep1 * w + 32) >> 6, where w = kMode6Weights[idx].
// Used only as documentation; the encoder's index-from-projection
// computation uses a uniform t * 15 mapping to keep the inner loop
// branch-free.
//
// constexpr int kMode6Weights[16] = {
//     0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64
// };

// Bit-packer for a 16-byte BC7 block.  Streams bits LSB-first; each
// `write(value, n)` appends n bits to the running cursor.
struct BitPacker {
    uint8_t* dst;       // 16-byte block (must be zero-initialised)
    uint32_t bit_pos;
    explicit BitPacker(uint8_t* d) : dst(d), bit_pos(0) {
        std::memset(dst, 0, 16);
    }
    void write(uint32_t value, uint32_t num_bits) {
        for (uint32_t i = 0; i < num_bits; ++i) {
            uint32_t bit = (value >> i) & 1u;
            dst[bit_pos / 8] |= uint8_t(bit << (bit_pos & 7u));
            ++bit_pos;
        }
    }
};

// Quantise an 8-bit value to a 7-bit "v7" + the channel's preferred
// p-bit (the LSB of the decoded value).  Decoded value:
//     decoded_8bit = (v7 << 1) | p
// Returns the v7 + p_bit pair that minimises round-trip squared error
// vs. the input.
inline void quantise7p(
    uint8_t  in,
    uint8_t& out_v7,
    uint8_t& out_p) {
    int best_err = 1 << 24;
    int best_v7  = 0;
    int best_p   = 0;
    for (int p = 0; p < 2; ++p) {
        // v7 such that (v7 << 1) | p approximates `in`.
        int v7 = (int(in) - p) >> 1;
        v7     = std::clamp(v7, 0, 127);
        int decoded = (v7 << 1) | p;
        int diff    = int(in) - decoded;
        int err     = diff * diff;
        if (err < best_err) {
            best_err = err;
            best_v7  = v7;
            best_p   = p;
        }
    }
    out_v7 = uint8_t(best_v7);
    out_p  = uint8_t(best_p);
}

}  // namespace

void encodeBC7Mode6(
    const uint8_t* src_rgba,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       dst_bc7) {

    if (!src_rgba || !dst_bc7 || width == 0 || height == 0) return;

    const uint32_t blocks_x = (width  + 3u) / 4u;
    const uint32_t blocks_y = (height + 3u) / 4u;

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {

            // ── Step 1: gather 16 pixels of this 4×4 block ─────────
            // Edge clamp so blocks past the image right/bottom edge
            // get the rightmost/bottommost real pixels rather than
            // garbage.
            uint8_t pix[16][4];
            for (int py = 0; py < 4; ++py) {
                uint32_t y = std::min(by * 4u + uint32_t(py), height - 1u);
                for (int px = 0; px < 4; ++px) {
                    uint32_t x = std::min(bx * 4u + uint32_t(px), width - 1u);
                    const uint8_t* s = src_rgba + (size_t(y) * width + x) * 4u;
                    pix[py * 4 + px][0] = s[0];
                    pix[py * 4 + px][1] = s[1];
                    pix[py * 4 + px][2] = s[2];
                    pix[py * 4 + px][3] = s[3];
                }
            }

            // ── Step 2: endpoints = component-wise min and max ─────
            uint8_t ep0[4] = {255, 255, 255, 255};
            uint8_t ep1[4] = {0, 0, 0, 0};
            for (int i = 0; i < 16; ++i) {
                for (int c = 0; c < 4; ++c) {
                    if (pix[i][c] < ep0[c]) ep0[c] = pix[i][c];
                    if (pix[i][c] > ep1[c]) ep1[c] = pix[i][c];
                }
            }

            // ── Step 3: quantise endpoints to 7 + 1 bits ───────────
            // Each channel suggests its own preferred p-bit; majority
            // vote among 4 channels picks the shared p-bit per
            // endpoint (Mode 6 stores ONE p-bit per endpoint).
            uint8_t ep0_v7[4], ep0_p[4];
            uint8_t ep1_v7[4], ep1_p[4];
            for (int c = 0; c < 4; ++c) {
                quantise7p(ep0[c], ep0_v7[c], ep0_p[c]);
                quantise7p(ep1[c], ep1_v7[c], ep1_p[c]);
            }
            int votes0 = ep0_p[0] + ep0_p[1] + ep0_p[2] + ep0_p[3];
            int votes1 = ep1_p[0] + ep1_p[1] + ep1_p[2] + ep1_p[3];
            uint8_t p0 = (votes0 >= 2) ? 1u : 0u;
            uint8_t p1 = (votes1 >= 2) ? 1u : 0u;

            // Re-quantise each channel with the chosen shared p-bit
            // (the per-channel preferred p-bit may have differed).
            for (int c = 0; c < 4; ++c) {
                int v7 = (int(ep0[c]) - int(p0)) >> 1;
                ep0_v7[c] = uint8_t(std::clamp(v7, 0, 127));
                v7 = (int(ep1[c]) - int(p1)) >> 1;
                ep1_v7[c] = uint8_t(std::clamp(v7, 0, 127));
            }

            // Reconstruct decoded endpoints (the 8-bit values the GPU
            // will use during interpolation) for index search.
            int dec0[4], dec1[4];
            for (int c = 0; c < 4; ++c) {
                dec0[c] = (int(ep0_v7[c]) << 1) | int(p0);
                dec1[c] = (int(ep1_v7[c]) << 1) | int(p1);
            }

            // ── Step 4: per-pixel 4-bit index ───────────────────────
            // Project each pixel along (ep1 - ep0).  4D dot product —
            // weighting all four channels equally.  Length-squared can
            // be zero for degenerate blocks (constant color); fall
            // back to index 0 in that case.
            int dir[4] = {
                dec1[0] - dec0[0],
                dec1[1] - dec0[1],
                dec1[2] - dec0[2],
                dec1[3] - dec0[3] };
            int len_sq = dir[0]*dir[0] + dir[1]*dir[1] +
                         dir[2]*dir[2] + dir[3]*dir[3];

            uint8_t indices[16];
            for (int i = 0; i < 16; ++i) {
                if (len_sq == 0) {
                    indices[i] = 0;
                    continue;
                }
                int p_minus_e0[4] = {
                    int(pix[i][0]) - dec0[0],
                    int(pix[i][1]) - dec0[1],
                    int(pix[i][2]) - dec0[2],
                    int(pix[i][3]) - dec0[3] };
                int proj = p_minus_e0[0]*dir[0] + p_minus_e0[1]*dir[1] +
                           p_minus_e0[2]*dir[2] + p_minus_e0[3]*dir[3];
                // t in [0, 15], rounded to nearest.  proj/len_sq is in
                // [0, 1] for in-range pixels; clamp to absorb numerical
                // overshoot at endpoints.
                int idx = (proj * 15 + len_sq / 2) / len_sq;
                idx     = std::clamp(idx, 0, 15);
                indices[i] = uint8_t(idx);
            }

            // ── Step 4b: anchor convention ─────────────────────────
            // Mode 6 stores index[0] as 3 bits (high bit implicit 0).
            // If the natural high bit is 1, we must swap endpoints AND
            // invert all indices to keep decoded values unchanged.
            if (indices[0] & 0x8u) {
                for (int c = 0; c < 4; ++c) {
                    std::swap(ep0_v7[c], ep1_v7[c]);
                }
                std::swap(p0, p1);
                for (int i = 0; i < 16; ++i) {
                    indices[i] = uint8_t(15 - indices[i]);
                }
            }

            // ── Step 5: bit-pack into 16 bytes ──────────────────────
            uint8_t* block = dst_bc7 + (size_t(by) * blocks_x + bx) * 16u;
            BitPacker bp(block);
            // Mode 6 header: bit-6 set, bits 0..5 zero.  Sequential
            // 7-bit write of value 0x40 puts the 1-bit at the right
            // position.
            bp.write(0x40u, 7);
            // Endpoints: R0,R1,G0,G1,B0,B1,A0,A1 (channel-interleaved
            // pairs, 7 bits each).
            bp.write(ep0_v7[0], 7); bp.write(ep1_v7[0], 7);
            bp.write(ep0_v7[1], 7); bp.write(ep1_v7[1], 7);
            bp.write(ep0_v7[2], 7); bp.write(ep1_v7[2], 7);
            bp.write(ep0_v7[3], 7); bp.write(ep1_v7[3], 7);
            // P-bits.
            bp.write(p0, 1);
            bp.write(p1, 1);
            // Anchor index: low 3 bits only (high bit is implicit 0).
            bp.write(indices[0] & 0x7u, 3);
            // Remaining 15 indices, 4 bits each.
            for (int i = 1; i < 16; ++i) {
                bp.write(indices[i], 4);
            }
        }
    }
}

}  // namespace scene_rendering
}  // namespace engine
