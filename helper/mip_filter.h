#pragma once
// ── 2x mip downsampling with a windowed-sinc kernel ──────────────────
// A 2x2 box is the cheapest 2x filter and the worst-looking one: it is
// a poor low-pass (aliasing survives it) and it blurs what it keeps, so
// high-detail textures (bark, leaf venation, terrain speckle) turn to
// mush a level early and shimmer a level late.  Every CPU mip chain in
// the engine (VT pyramid + slot mip 1, cutout stopgap chain, the alpha
// layer) goes through here instead: a separable Lanczos-3 kernel
// (12 taps per axis at 2x, edge-clamped), sRGB-correct for colour
// (decoded to linear, filtered, re-encoded), colour weighted by alpha
// so cutout fringes take the ink's colour rather than the transparent
// black beside it, alpha filtered plainly (it is coverage, not light).
// Negative lobes are clamped on output; coverage preservation for
// cutouts is a separate step the callers keep (cutout_mips.h).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace engine::helper {

namespace mipfilter_detail {
inline float sinc(float x) {
    if (std::fabs(x) < 1e-6f) return 1.0f;
    const float px = 3.14159265358979f * x;
    return std::sin(px) / px;
}
// Lanczos-3 weights for the 12 source taps of one destination texel at
// exact 2x decimation.  Destination texel x covers source [2x, 2x+2);
// its centre in source coordinates is 2x+1, source tap i (centre
// i+0.5) sits at t = (i + 0.5 - (2x+1)) / 2 destination texels.  With
// k = i - 2x the offsets are k = -5..6.  Normalised to sum 1.
struct Kernel {
    static constexpr int kFirst = -5;
    static constexpr int kTaps  = 12;
    float w[kTaps];
    Kernel() {
        float sum = 0.0f;
        for (int n = 0; n < kTaps; ++n) {
            const float t = (float(kFirst + n) - 0.5f) * 0.5f;
            w[n] = std::fabs(t) < 3.0f ? sinc(t) * sinc(t / 3.0f) : 0.0f;
            sum += w[n];
        }
        for (auto& v : w) v /= sum;
    }
};
inline const Kernel& kernel() { static const Kernel k; return k; }

struct SrgbLuts {
    float   to_linear[256];
    uint8_t to_srgb[4096 + 1];
    SrgbLuts() {
        for (int i = 0; i < 256; ++i) {
            const float c = float(i) / 255.0f;
            to_linear[i] = c <= 0.04045f ? c / 12.92f
                                         : std::pow((c + 0.055f) / 1.055f, 2.4f);
        }
        for (int i = 0; i <= 4096; ++i) {
            const float l = float(i) / 4096.0f;
            const float c = l <= 0.0031308f ? l * 12.92f
                                            : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
            to_srgb[i] = uint8_t(std::min(255.0f, std::max(0.0f, c * 255.0f + 0.5f)));
        }
    }
};
inline const SrgbLuts& srgbLuts() { static const SrgbLuts l; return l; }
inline uint8_t toByte(float v) {
    return uint8_t(std::min(255.0f, std::max(0.0f, v * 255.0f + 0.5f)));
}
}  // namespace mipfilter_detail

// RGBA8 -> RGBA8 at half size (max(1, w/2) x max(1, h/2)).  `srgb`:
// colour channels are sRGB-encoded (albedo); false for data textures.
inline void downsample2xRgba8(const uint8_t* src, uint32_t sw, uint32_t sh,
                              uint8_t* dst, bool srgb = true) {
    using namespace mipfilter_detail;
    const Kernel& K = kernel();
    const SrgbLuts& L = srgbLuts();
    const uint32_t dw = std::max(1u, sw >> 1), dh = std::max(1u, sh >> 1);
    // Pass 1 (horizontal): per source row, dw texels of
    // {premultiplied linear rgb, plain linear rgb, alpha} as floats.
    // 7 floats per texel.
    std::vector<float> tmp(size_t(dw) * sh * 7u);
    for (uint32_t y = 0; y < sh; ++y) {
        const uint8_t* row = src + size_t(y) * sw * 4u;
        float* out = tmp.data() + size_t(y) * dw * 7u;
        for (uint32_t x = 0; x < dw; ++x) {
            float pr = 0, pg = 0, pb = 0, r = 0, g = 0, b = 0, a = 0;
            for (int n = 0; n < Kernel::kTaps; ++n) {
                const int i = std::clamp(int(2u * x) + Kernel::kFirst + n, 0, int(sw) - 1);
                const uint8_t* p = row + size_t(i) * 4u;
                const float w = K.w[n];
                const float lr = srgb ? L.to_linear[p[0]] : p[0] * (1.0f / 255.0f);
                const float lg = srgb ? L.to_linear[p[1]] : p[1] * (1.0f / 255.0f);
                const float lb = srgb ? L.to_linear[p[2]] : p[2] * (1.0f / 255.0f);
                const float la = p[3] * (1.0f / 255.0f);
                pr += w * lr * la; pg += w * lg * la; pb += w * lb * la;
                r  += w * lr;      g  += w * lg;      b  += w * lb;
                a  += w * la;
            }
            float* o = out + size_t(x) * 7u;
            o[0] = pr; o[1] = pg; o[2] = pb; o[3] = r; o[4] = g; o[5] = b; o[6] = a;
        }
    }
    // Pass 2 (vertical) and encode.
    for (uint32_t y = 0; y < dh; ++y) {
        for (uint32_t x = 0; x < dw; ++x) {
            float acc[7] = {0, 0, 0, 0, 0, 0, 0};
            for (int n = 0; n < Kernel::kTaps; ++n) {
                const int j = std::clamp(int(2u * y) + Kernel::kFirst + n, 0, int(sh) - 1);
                const float* t = tmp.data() + (size_t(j) * dw + x) * 7u;
                const float w = K.w[n];
                for (int c = 0; c < 7; ++c) acc[c] += w * t[c];
            }
            const float a = std::clamp(acc[6], 0.0f, 1.0f);
            float rgb[3];
            if (a > 1e-4f) {
                for (int c = 0; c < 3; ++c) rgb[c] = std::clamp(acc[c] / a, 0.0f, 1.0f);
            } else {
                for (int c = 0; c < 3; ++c) rgb[c] = std::clamp(acc[3 + c], 0.0f, 1.0f);
            }
            uint8_t* d = dst + (size_t(y) * dw + x) * 4u;
            for (int c = 0; c < 3; ++c) {
                d[c] = srgb ? L.to_srgb[uint32_t(rgb[c] * 4096.0f + 0.5f)] : toByte(rgb[c]);
            }
            d[3] = toByte(a);
        }
    }
}

// Single-channel (alpha / mask) 8-bit -> half size, same kernel.
inline void downsample2xR8(const uint8_t* src, uint32_t sw, uint32_t sh, uint8_t* dst) {
    using namespace mipfilter_detail;
    const Kernel& K = kernel();
    const uint32_t dw = std::max(1u, sw >> 1), dh = std::max(1u, sh >> 1);
    std::vector<float> tmp(size_t(dw) * sh);
    for (uint32_t y = 0; y < sh; ++y) {
        const uint8_t* row = src + size_t(y) * sw;
        for (uint32_t x = 0; x < dw; ++x) {
            float a = 0;
            for (int n = 0; n < Kernel::kTaps; ++n) {
                const int i = std::clamp(int(2u * x) + Kernel::kFirst + n, 0, int(sw) - 1);
                a += K.w[n] * row[i];
            }
            tmp[size_t(y) * dw + x] = a;
        }
    }
    for (uint32_t y = 0; y < dh; ++y) {
        for (uint32_t x = 0; x < dw; ++x) {
            float a = 0;
            for (int n = 0; n < Kernel::kTaps; ++n) {
                const int j = std::clamp(int(2u * y) + Kernel::kFirst + n, 0, int(sh) - 1);
                a += K.w[n] * tmp[size_t(j) * dw + x];
            }
            dst[size_t(y) * dw + x] = uint8_t(std::min(255.0f, std::max(0.0f, a + 0.5f)));
        }
    }
}

}  // namespace engine::helper
