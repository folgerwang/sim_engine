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
// ── Height profile shape ─────────────────────────────────────────────
// A pure h^2 cantilever is right for a rigid trunk and wrong for what
// these plants look like: at a third of the reference height it yields
// 11% of the canopy's travel, and under 1% near the base, so the bottom
// third of every plant read as frozen while the top swung.  Mixing in a
// LINEAR term keeps the root planted (still exactly 0 at h=0) and still
// grows toward the tip, but lets the lower trunk actually take part --
// 0 = pure quadratic (the old shape), 1 = pure linear.
const float kVegBendLinear  = 0.45;
// Height clamp, in canopy references.  Was 1.2, i.e. everything above
// ~10.8 m shared ONE bend value: on a 20 m pine that froze the entire
// upper half into a rigid slab that translated instead of swinging, and
// it is the other half of "the top doesn't swing more than the bottom".
// 2.2 lets a ~20 m tree keep opening up all the way to its crown.
const float kVegBendMaxH    = 2.2;
// Flutter is PER METRE of plant height (capped): a grass tuft's whole
// motion is flutter, and scaling it by the canopy-squared bend left
// the entire clutter layer visually frozen (measured 1 mm of travel on
// a 0.6 m tuft while trees moved 25 cm).  0.03/m puts ~2 cm of ripple
// on that tuft and ~4 cm of leaf shimmer atop a tree's lean.
const float kVegFlutterPerM = 0.03;
const float kVegFlutterCapM = 1.5;
const float kVegGustFreq    = 0.9;    // Hz-ish of the main gust front
const float kVegGustWaveInv = 0.055;  // 1/m: gust front spatial phase

// ── Downed timber does not sway ──────────────────────────────────────
// terrain_pcg.py's deadfall pass turns ~a fixed fraction of trees into
// the SAME MESH pitched 1.36-1.66 rad (78-95 deg) about its own base,
// carried in the per-instance quaternion.  Nothing downstream knew
// that, so a fallen trunk kept bending "up" its own local +Y -- which
// after the pitch points sideways along the ground -- and the tip of a
// 20 m log swept a quarter-metre arc across the forest floor every
// gust.  A tree on the ground has no root plate left to spring from
// and no canopy left to catch wind: it is inert.
//
// The test is the instance's OWN up axis after its transform, so it
// costs one dot product and needs no new vertex attribute, no CPU
// change and no asset change: cos(tilt) = 1 upright, ~0.21 down to
// -0.09 across the whole deadfall pitch range.  Faded rather than
// switched so a leaning-but-standing tree loses travel gradually; the
// window sits well clear of both populations (standing trees are
// yaw-only, so exactly 1.0).
const float kVegFallenCos   = 0.30;   // <= this: fully inert
const float kVegUprightCos  = 0.72;   // >= this: full sway (~44 deg)

// ── ONE PLANT = ONE RIGID SWING ──────────────────────────────────────
// A tree's fundamental bending mode has NO NODE between root and tip:
// every point of the trunk travels the same way at the same instant,
// with the amplitude growing toward the canopy.  Modes that DO reverse
// along the trunk (the ones that draw an S) are an order of magnitude
// higher in frequency and far weaker — you do not see them on a tree.
//
// The old flutter term carried `+ local_h * 0.8` INSIDE its phase, so
// the sine advanced 0.8 rad for every metre of height: 7.2 rad up a 9 m
// trunk, more than a full cycle.  Different heights of the SAME tree
// were therefore at opposite points of the swing — the S.  It was also
// signed, so low down (where lean * bend is nearly nothing) the flutter
// could exceed the lean and push the lower trunk UPWIND while the
// canopy leaned downwind, which is the same artefact from the other
// direction.
//
// Two rules keep the whole plant coherent, and both are enforced by
// construction rather than by tuning:
//
//   PHASE depends on the INSTANCE only, never on local_h — so every
//   vertex of one plant swings together, while neighbouring plants
//   still differ (the gust front crossing the field is between trees,
//   which is what you want, not within one).
//
//   AMPLITUDE is non-negative and non-decreasing in local_h — so the
//   displacement can only grow from root to tip and can only point
//   downwind.  No section can ever oppose another.
// `up_ws` is the instance's local +Y axis after the instance rotation
// (local_world_rot_mat * vec3(0,1,0)) -- taken as the image of the axis
// under the very matrix that transforms the vertex, so it cannot drift
// from whatever row/column convention InstanceDataInfo is packed in.
vec3 vegSwayOffset(vec3 inst_t, float local_h, float t, vec3 up_ws) {
    // Fallen/steeply-pitched instances: no swing at all.
    float upright = up_ws.y * inversesqrt(max(dot(up_ws, up_ws), 1e-8f));
    float stand = smoothstep(kVegFallenCos, kVegUprightCos, upright);
    if (stand <= 0.0f) {
        return vec3(0.0f);
    }
    float ph = dot(inst_t.xz, kVegWindDir) * kVegGustWaveInv;
    // two incommensurate travelling waves so the field never breathes
    // in perfect unison
    float g = 0.55 + 0.45 * sin(t * kVegGustFreq - ph)
                   + 0.25 * sin(t * 2.3 - ph * 1.7 + inst_t.x * 0.11);
    // Clamped at 0: a gust pushes or it does not; it never sucks the
    // canopy upwind.
    g = max(g, 0.0);

    // Roots planted, canopy carries the travel — but the whole stem
    // swings, not just the top third (see kVegBendLinear/kVegBendMaxH).
    // Still monotonically increasing in h and still exactly 0 at the
    // root, so the plant cannot fold into an S.
    float h01  = clamp(local_h * (1.0 / kVegCanopyRefM),
                       0.0, kVegBendMaxH);
    float bend = h01 * (kVegBendLinear + (1.0 - kVegBendLinear) * h01);
    float lean = kVegLeanM * g;

    // Per-instance flutter phase from the translation itself — no hash
    // lookup, no divergence between passes, and CRUCIALLY no local_h.
    float f_ph = t * (3.1 + fract(inst_t.x * 0.37 +
                                  inst_t.z * 0.53) * 2.0);
    // Rectified to [0,1] rather than [-1,1], then doubled to keep the
    // peak-to-peak travel the clutter was tuned at (a 0.6 m tuft still
    // gets ~3.6 cm, a canopy ~9 cm on top of its lean).  Rectifying is
    // what guarantees the flutter can only ever ADD downwind travel.
    float fl = 2.0 * kVegFlutterPerM *
               min(local_h, kVegFlutterCapM) * (0.5 + 0.5 * sin(f_ph));

    vec2 off = kVegWindDir * (lean * bend + fl) * stand;
    // the canopy dips slightly as it leans — an arc, not a shear
    return vec3(off.x, -0.30 * dot(off, off), off.y);
}

#endif  // VEG_SWAY_GLSL_H_
