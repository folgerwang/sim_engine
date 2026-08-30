#ifndef VEG_SWAY_GLSL_H_
#define VEG_SWAY_GLSL_H_

// ── Vegetation wind sway (MODEL_FLAG_VEGETATION_SWAY) ────────────────
// One definition, included by base.vert AND base_depthonly.vert: the
// shadow pass must displace exactly what the colour pass displaces, or
// every tree's shadow slides off its trunk in the gusts.
//
// PROCEDURAL on purpose.  The drawable path binds no wind texture, and
// threading the WindField patch through its pipeline layouts would
// touch every placed-object pipeline in the engine for a effect whose
// visible content is "gusts move through the canopy".  The gust model
// here is the same family the water waves use (a travelling front
// along a fixed wind direction + per-instance phase), and it shares
// their fallback direction, so trees and river chop read as the same
// weather.  When the water is upgraded to full sim wind near the
// camera the mismatch is a few degrees of direction — revisit then by
// binding the patch here too.
//
// The bend is quadratic in LOCAL height: roots planted, canopy carries
// the travel.  ~9 m is the reference canopy; a 2 m bush at the same
// wind bends (2/9)^2 = 5% of a tall tree's travel, which is about
// right, and a grass-clutter quad barely stirs — its motion mostly
// comes from the flutter term.
const vec2  kVegWindDir     = vec2(0.8575, 0.5145);   // = water fallback
const float kVegCanopyRefM  = 9.0;
const float kVegLeanM       = 0.22;   // canopy travel in a full gust
// Flutter is PER METRE of plant height (capped): a grass tuft's whole
// motion is flutter, and scaling it by the canopy-squared bend left
// the entire clutter layer visually frozen (measured 1 mm of travel on
// a 0.6 m tuft while trees moved 25 cm).  0.03/m puts ~2 cm of ripple
// on that tuft and ~4 cm of leaf shimmer atop a tree's lean.
const float kVegFlutterPerM = 0.03;
const float kVegFlutterCapM = 1.5;
const float kVegGustFreq    = 0.9;    // Hz-ish of the main gust front
const float kVegGustWaveInv = 0.055;  // 1/m: gust front spatial phase

vec3 vegSwayOffset(vec3 inst_t, float local_h, float t) {
    float ph = dot(inst_t.xz, kVegWindDir) * kVegGustWaveInv;
    // two incommensurate travelling waves so the field never breathes
    // in perfect unison
    float g = 0.55 + 0.45 * sin(t * kVegGustFreq - ph)
                   + 0.25 * sin(t * 2.3 - ph * 1.7 + inst_t.x * 0.11);
    float h01  = clamp(local_h * (1.0 / kVegCanopyRefM), 0.0, 1.2);
    float bend = h01 * h01;
    float lean = kVegLeanM * max(g, 0.0);
    // per-instance flutter phase from the translation itself — no hash
    // lookup, no divergence between passes
    float fl = kVegFlutterPerM * min(local_h, kVegFlutterCapM) *
               sin(t * (3.1 + fract(inst_t.x * 0.37 + inst_t.z * 0.53) *
                              2.0) +
                   local_h * 0.8);
    vec2 off = kVegWindDir * (lean * bend + fl);
    // the canopy dips slightly as it leans — an arc, not a shear
    return vec3(off.x, -0.30 * dot(off, off), off.y);
}

#endif  // VEG_SWAY_GLSL_H_
