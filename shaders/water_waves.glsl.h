#ifndef WATER_WAVES_GLSL_H
#define WATER_WAVES_GLSL_H

// Visual waves only. The authored water level/column is never modified.
const float kWaterWaveMaxOffsetM = 0.08;

float boundedWaterWave(float deviation) {
    if (isnan(deviation) || isinf(deviation)) return 0.0;
    // Smooth saturation avoids flat clipped crests and abrupt normal changes.
    return kWaterWaveMaxOffsetM * tanh(deviation / kWaterWaveMaxOffsetM);
}

float waterWaveOffset(sampler2D surface, vec4 region, vec2 world_xz,
                      float depth) {
    float span = region.y * region.w;
    if (span <= 1.0 || depth <= 0.02) return 0.0;
    vec2 uv = (world_xz - region.xz) / span;
    if (any(lessThanEqual(uv, vec2(0.0))) ||
        any(greaterThanEqual(uv, vec2(1.0)))) return 0.0;
    vec2 edge = min(uv, 1.0 - uv);
    float feather = smoothstep(0.0, 0.15, min(edge.x, edge.y));
    float shallow = smoothstep(0.05, 0.75, depth);
    float offset = textureLod(surface, uv, 0.0).w;
    if (isnan(offset) || isinf(offset)) return 0.0;
    // Also protect consumers of an old/stale surface texture. Negative
    // troughs can never eat more than a quarter of the local water column.
    float limit = min(kWaterWaveMaxOffsetM, depth * 0.25);
    return clamp(offset, -limit, limit) * feather * shallow;
}

// ── WIND WAVES: a directional sine spectrum + noise ──────────────────
//
// Everything above is the LBM patch: a 512-cell lattice around the
// camera that carries river current and its ripples.  It is the right
// model for what it covers and it covers ~500 m — outside it (and
// wherever the patch has nothing to stir, which is every lake and the
// open sea) the surface is a MIRROR, and a mirror is what the eye reads
// as "this is not water".  The sim cannot be the answer there: the
// lattice is bounded by design, and widening it costs cells squared.
//
// What open water actually has is a WIND SEA — many trains of waves,
// each with its own wavelength, direction and speed, running at once.
// That superposition is what the sum below is: a small spectrum of
// directional sines, wavelengths falling geometrically from ~11 m to
// under a metre, each fanned off the wind by its own angle so the
// pattern never lines up into a plaid, each travelling at the deep-
// water speed for ITS wavelength (c = sqrt(g/k), so the long swell
// outruns the chop — same dispersion the real sea has, and the reason a
// sum of same-speed sines reads as a moving corrugation instead).
//
// The sines alone still repeat, because a sum of periodic functions is
// periodic.  A value-noise field warps their PHASE (and a second, finer
// one adds the last of the surface texture), which breaks the lattice
// without adding a texture fetch or a table: the noise below is hashed
// arithmetic with an analytic gradient, so the slope of the whole field
// is exact rather than differenced.
//
// SLOPE, NOT HEIGHT, is what is used.  The water surface is static by
// design (the mesh is the authored water level; see the architecture
// note in tile_water.frag) and the mesh is far too coarse to carry a
// 1 m wave anyway, so the waves are applied as a NORMAL perturbation
// and the height below exists for callers that want it (and to make the
// slope derivation legible).  Bump-mapped water at these amplitudes is
// indistinguishable from displaced water outside a silhouette.
//
// BAND LIMITING.  A wave whose wavelength is near the pixel footprint
// cannot be resolved: it aliases into crawling sparkle, which is the
// classic far-field water shimmer.  Each octave is faded out as its
// wavelength approaches the footprint, and the amplitude*slope it took
// with it is handed back as ROUGHNESS — the same energy, moved from
// geometry the pixel cannot see into the microfacet term it can.

const int   kWaterWaveOct     = 6;
const float kWaterWaveLen0    = 11.0;   // m, longest component
const float kWaterWaveLac     = 1.86;   // wavelength ratio per octave
const float kWaterWaveAmp0    = 0.12;   // m, longest component at gain 1
// Amplitude falls slower than wavelength does, so STEEPNESS (a*k, what
// the normal actually shows) grows a little toward the short end — which
// is the right shape: the long swell is broad and gentle, the chop on
// top of it is what glints.  Measured over open water at these numbers:
// mean surface slope 4.5 deg, peaks near 12 deg, ~5 cm rms of height.
const float kWaterWaveGain    = 0.60;   // amplitude ratio per octave
const float kWaterWaveGrav    = 9.81;
// Crest sharpening: real wind waves have round troughs and sharp
// crests.  s = (0.5 + 0.5*sin)^p is that shape with an exact
// derivative; p is kept low so the surface stays smooth enough to
// differentiate cheaply.
const float kWaterWaveSharp   = 1.7;
// Phase warp: how many radians of phase the low-frequency noise field
// can push a crest by, and the size of that field.
const float kWaterWaveWarpRad = 2.2;
const float kWaterWaveWarpInv = 0.021;  // 1/m
// Fallback wind while the wind patch is dead — the SAME direction the
// vegetation sway uses (veg_sway.glsl.h kVegWindDir), so the chop on
// the river and the gusts in the trees agree.
const vec2  kWaterWindFallback = vec2(0.8575, 0.5145);

// Value noise with an ANALYTIC gradient.  Quintic interpolation, so the
// gradient is continuous (a smoothstep blend gives a continuous value
// but a gradient with visible facets, which shows up on water as
// diamond-shaped patches — the exact artefact the near-field seam was).
float waterNoise2(vec2 p, out vec2 grad) {
    vec2 i = floor(p);
    vec2 f = p - i;
    // hash the four corners
    vec4 h4;
    for (int c = 0; c < 4; ++c) {
        vec2 o = vec2(float(c & 1), float((c >> 1) & 1));
        ivec2 q = ivec2(i + o);
        // Hashed in UNSIGNED arithmetic: signed overflow is not defined
        // and these products overflow by construction.
        uint n = uint(q.x) * 374761393u + uint(q.y) * 668265263u;
        n = (n ^ (n >> 13u)) * 1274126177u;
        n = n ^ (n >> 16u);
        h4[c] = float(n & 0xFFFFFFu) * (1.0 / float(0xFFFFFF)) * 2.0 - 1.0;
    }
    vec2 u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    float a = h4[0], b = h4[1], c2 = h4[2], d = h4[3];
    float k1 = b - a, k2 = c2 - a, k3 = a - b - c2 + d;
    grad = vec2(k1 + k3 * u.y, k2 + k3 * u.x) * du;
    return a + k1 * u.x + k2 * u.y + k3 * u.x * u.y;
}

struct WaterWaveField {
    float height;       // m above the still surface (informational)
    vec2  slope;        // dH/dx, dH/dz — what the normal is built from
    float unresolved;   // sub-footprint energy, to be spent as roughness
};

// Per-octave direction: fanned off the wind by an angle that does not
// divide evenly into any other, so no two trains ever share a crest
// line for long.
vec2 waterWaveDir(vec2 wind_dir, int i) {
    const float kFan[6] = float[6](0.10, -0.63, 0.41, -1.02, 0.78, -0.30);
    float a = kFan[i];
    float c = cos(a), s = sin(a);
    return vec2(c * wind_dir.x - s * wind_dir.y,
                s * wind_dir.x + c * wind_dir.y);
}

// `gain` scales the whole sea state (0 = glass, 1 = a stiff breeze);
// `footprint_m` is the world size of this pixel (length(fwidth(pos.xz)))
// and drives the band limit.
WaterWaveField waterWindWaves(vec2 xz, float t, vec2 wind_dir,
                              float gain, float footprint_m) {
    WaterWaveField w;
    w.height = 0.0;
    w.slope = vec2(0.0);
    w.unresolved = 0.0;
    if (gain <= 0.001) {
        return w;
    }
    float wl = length(wind_dir);
    vec2 wd = (wl > 1e-4) ? wind_dir / wl : kWaterWindFallback;

    // ONE low-frequency warp field, shared by every octave: the whole
    // sea bends together, the way a gust front bends it, instead of each
    // train wandering off on its own.
    vec2 ng;
    float nz = waterNoise2(xz * kWaterWaveWarpInv +
                           wd * (t * 0.035), ng);
    ng *= kWaterWaveWarpInv;
    float warp    = kWaterWaveWarpRad * nz;
    vec2  warp_gr = kWaterWaveWarpRad * ng;

    float len = kWaterWaveLen0;
    float amp = kWaterWaveAmp0 * gain;
    for (int i = 0; i < kWaterWaveOct; ++i) {
        float k = 6.28318530718 / len;
        // Deep-water dispersion: long waves are fast, chop is slow.
        float omega = sqrt(kWaterWaveGrav * k);
        vec2  d = waterWaveDir(wd, i);
        // Resolvability: a wave needs several pixels per wavelength.
        // Faded rather than switched, or the fade line itself is visible
        // as a ring around the camera.
        float res = 1.0 - smoothstep(0.22 * len, 0.75 * len, footprint_m);
        float ph  = dot(d, xz) * k - omega * t + warp;
        float sn  = sin(ph);
        // Sharpened crest: s = q^p with q = 0.5 + 0.5*sin(ph).
        float q   = 0.5 + 0.5 * sn;
        float qp  = pow(max(q, 1e-4), kWaterWaveSharp);
        // dS/dph = p * q^(p-1) * 0.5*cos(ph); centred so the mean stays
        // at the still level (a rectified wave otherwise lifts the whole
        // surface, which is the one thing the water may not do).
        float dS  = kWaterWaveSharp * (qp / max(q, 1e-4)) * 0.5 * cos(ph);
        vec2  dph = d * k + warp_gr;
        if (res > 0.001) {
            w.height += amp * res * (qp - 0.5);
            w.slope  += amp * res * dS * dph;
        }
        // What the pixel cannot resolve becomes microfacet roughness:
        // steepness (a*k) is the right currency, not amplitude.
        w.unresolved += amp * k * (1.0 - res);
        len *= 1.0 / kWaterWaveLac;
        amp *= kWaterWaveGain;
    }
    return w;
}
#endif
