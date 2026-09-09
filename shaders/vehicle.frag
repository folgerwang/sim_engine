#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "tonemap.glsl.h"

// ── VEHICLE PAINT ───────────────────────────────────────────────────
// Drawn through citizen.vert: the hull meshes are in METRES with the
// ground at y = 0, x forward, z to the right; in_local is the
// fragment's position on the hull, in_extra = (part kind, vehicle
// type, seed, light-bar clock).  Everything a car wears -- glass and
// pillars, head and tail lights, grille, bumpers, wheel arches, door
// seams, the trailer's corrugation, the liveries -- is painted here
// from that position and the face it is on (the face normal recovered
// from the position's screen derivatives, so the hull mesh carries
// nothing but positions).
layout(location = 0) in vec3 in_normal_ws;
layout(location = 1) in vec3 in_position_ws;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec3 in_local;
layout(location = 4) flat in vec4 in_extra;
// The hull's own normal, in the hull's own space (citizen.vert) — see
// the note there for why this exists.
layout(location = 5) in vec3 in_normal_ls;
// ── FROM THE LIBRARY ────────────────────────────────────────────────
// The baked cars (car_gen.py -> cars.rwcar) carry a PART ID per vertex
// and their paint per instance.  A negative part id means this draw is
// a built-in hull from VehicleSystem::buildHull — the fallback for when
// the library is missing — and everything below the part switch is that
// older path, which paints the same features by position and face.
layout(location = 6) flat in float in_part;
layout(location = 7) flat in vec4 in_paint;

// KEEP IN LOCKSTEP with car_gen.py's kPart* constants.
const int kPartBody = 0, kPartGlass = 1, kPartTrim = 2, kPartChrome = 3,
          kPartLampFront = 4, kPartLampRear = 5, kPartGrille = 6,
          kPartTyre = 7, kPartRim = 8, kPartInterior = 9,
          kPartLivery = 10, kPartLightBar = 11;

#include "citizen_gbuffer.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

const int kSedan = 0, kSuv = 1, kCyber = 2, kPickup = 3, kSemi = 4,
          kPolice = 5, kAmbulance = 6, kSchoolBus = 7, kFire = 8;

// per type: (belt, roof, side glass x0, side glass x1)
// KEEP IN LOCKSTEP with specs() in vehicle_system.cpp — the shader
// re-derives the hull's geometry from these to know where the glass,
// the arches and the lamps go, so a dimension changed in one place and
// not the other paints a window onto sheet metal.
const vec4 kGlass[9] = vec4[9](
    vec4(1.05, 1.50, -1.05, 0.95),   // sedan: DLO ends at the C pillar
    vec4(1.10, 1.65, -1.85, 1.15),   // crossover: long DLO to the tailgate
    vec4(1.14, 1.74, -1.30, 1.55),
    vec4(1.44, 1.90, -0.58, 1.20),   // pickup: cab glass only
    vec4(2.35, 3.45, -1.50, 1.50),
    vec4(1.12, 1.76, -2.05, 1.28),   // police utility
    vec4(1.40, 2.28, -2.30, 1.62),   // ambulance: cab + box rear window
    vec4(1.78, 2.72, -5.00, 2.95),
    vec4(2.10, 3.30, 0.75, 2.95));
// per type: (half length, half width, front wheel x, rear wheel x)
const vec4 kDims[9] = vec4[9](
    vec4(2.10, 0.88, 1.30, -1.30),   // sedan:     4.20 m, WB 2.60
    vec4(2.225, 0.93, 1.35, -1.35),  // crossover: 4.45 m, WB 2.70
    vec4(2.85, 1.00, 1.81, -1.81),   // stainless: 5.70 m, WB 3.62
    vec4(2.95, 1.025, 2.00, -1.68),  // pickup:    5.90 m, WB 3.68
    vec4(7.30, 1.275, 2.60, -1.05),
    vec4(2.50, 1.00, 1.52, -1.52),   // police:    5.00 m, WB 3.04
    vec4(2.65, 0.975, 1.75, -1.55),
    vec4(5.25, 1.25, 3.30, -2.90),
    vec4(4.75, 1.25, 3.40, -2.10));  // pumper: front axle under the cab
// per type: (wheel radius, headlight y0, headlight y1, tail y0)
const vec4 kLamp[9] = vec4[9](
    vec4(0.32, 0.60, 0.80, 0.70),    // sedan
    vec4(0.35, 0.76, 0.96, 0.86),    // crossover
    vec4(0.44, 0.93, 0.99, 1.04),
    vec4(0.43, 0.90, 1.12, 0.95),    // pickup
    vec4(0.52, 0.85, 1.05, 1.35),
    vec4(0.38, 0.80, 1.00, 0.90),    // police
    vec4(0.37, 0.78, 0.98, 0.95),    // ambulance
    vec4(0.52, 0.95, 1.15, 1.00),
    vec4(0.55, 1.10, 1.35, 1.10));

// ── A ROUNDED RECTANGLE, in a normalised box ────────────────────────
// q runs -1..1 over the feature's own extent, r is the corner radius as
// a fraction of that.  A lamp is not a rectangle: the corner radius is
// most of what makes one read as a moulded unit rather than as a decal,
// and having it as a mask means the bezel, the lens and the daytime
// strip are all the same shape at three sizes rather than three
// separately tuned boxes.
float rrect(vec2 q, float r) {
    vec2 d = abs(q) - vec2(1.0 - r);
    return step(length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0), r);
}

const vec3 kGlassCol = vec3(0.05, 0.065, 0.09);
// per type: (cabin floor y, dashboard x0) for the interior paint
const vec2 kCabin[9] = vec2[9](
    vec2(0.30, 1.18), vec2(0.36, 1.25), vec2(0.40, 1.30), vec2(0.45, 1.28),
    vec2(1.20, 1.15), vec2(0.38, 1.35), vec2(0.45, 1.72), vec2(0.55, 3.65),
    vec2(0.95, 3.00));
const vec3 kInteriorDark = vec3(0.085, 0.085, 0.095);
const vec3 kDark = vec3(0.045, 0.045, 0.05);
const vec3 kChrome = vec3(0.78, 0.80, 0.82);

void main() {
    const int kind = int(in_extra.x + 0.5);
    const int type = int(in_extra.y + 0.5);
    const float seed = in_extra.z;
    const float blink = in_extra.w;
    vec3 lp = in_local;
    // ── WHICH FACE IS THIS ──────────────────────────────────────────
    // The hull's own normal, interpolated from the mesh, in the hull's
    // own space.  It used to be reconstructed from the screen
    // derivatives of lp, which is free and correct on a big panel and
    // degenerate exactly where the details live: the end caps are
    // triangle fans whose triangles pinch to nothing at the centre, so
    // dFdx/dFdy vanish there and the reconstructed normal is noise —
    // the headlamps and grille broke into speckle and crawled with the
    // camera.  Derivatives stay as the fallback for any draw whose mesh
    // arrives without normals.
    vec3 fn = in_normal_ls;
    float fl = dot(fn, fn);
    fn = (fl > 1e-6) ? fn * inversesqrt(fl)
                     : normalize(cross(dFdx(lp), dFdy(lp)));
    vec3 an = abs(fn);

    vec3 base = in_color.rgb;
    vec3 albedo = base;
    vec3 emissive = vec3(0.0);
    bool glass = false;
    float alpha = 1.0;

    // ── THE LIBRARY PATH ────────────────────────────────────────────
    // A part id says what this fragment is, so nothing here has to be
    // deduced.  Everything the old path worked out from position, face
    // normal and a duplicated dimension table is simply known.
    int part = int(in_part + 0.5);
    if (in_part >= 0.0) {
        bool lib_glass = (part == kPartGlass);
        if (part == kPartBody) {
            albedo = base;
        } else if (part == kPartLivery) {
            // the second body colour: police doors, a trailer's box, a
            // bus panel.  White unless the instance says otherwise.
            albedo = vec3(0.90, 0.91, 0.92);
        } else if (part == kPartGlass) {
            albedo = kGlassCol;
        } else if (part == kPartTrim) {
            albedo = vec3(0.085, 0.088, 0.095);
        } else if (part == kPartChrome) {
            albedo = kChrome;
        } else if (part == kPartLampFront) {
            albedo = vec3(0.86, 0.86, 0.84);
            emissive = vec3(1.0, 0.98, 0.92) * 0.55;
        } else if (part == kPartLampRear) {
            albedo = vec3(0.52, 0.05, 0.035);
            emissive = vec3(1.0, 0.06, 0.03) * 0.55;
        } else if (part == kPartGrille) {
            albedo = vec3(0.045, 0.047, 0.05);
        } else if (part == kPartTyre) {
            albedo = vec3(0.032, 0.032, 0.035);
        } else if (part == kPartRim) {
            albedo = vec3(0.62, 0.64, 0.67);
        } else if (part == kPartInterior) {
            albedo = base;                     // the trim colour
        } else if (part == kPartLightBar) {
            // the bar alternates red and blue along its length and
            // flashes on the instance's clock
            float half_ = step(0.0, lp.z);
            vec3 c = mix(vec3(0.85, 0.06, 0.05), vec3(0.06, 0.16, 0.95), half_);
            float ph = fract(blink * 1.6 + half_ * 0.5);
            albedo = c * 0.35;
            emissive = (blink > 0.0 && ph < 0.5) ? c * 2.2 : c * 0.10;
        }
        glass = lib_glass;
        // The glass pass draws the same mesh again: windows only there,
        // everything but windows in the opaque pass.
        if (kind == 0 && lib_glass) discard;
        if (kind == 3) {
            if (!lib_glass) discard;
            alpha = 0.55;
        }
    } else if (kind == 4) {
        // ── THE INTERIOR: seats in the trim colour, the rest dark; the
        //    handles and mirrors outside the doors are chrome ─────────
        vec2 cb = kCabin[type];
        vec4 d = kDims[type];
        albedo = base;
        if (lp.y < cb.x + 0.02 || lp.x > cb.y) albedo = kInteriorDark;
        if (abs(lp.z) > d.y - 0.01) albedo = kChrome;
    } else if (kind == 0 || kind == 3) {
        // ── THE HULL ────────────────────────────────────────────────
        vec4 g = kGlass[type];
        vec4 d = kDims[type];
        vec4 lamp = kLamp[type];
        float belt = g.x, roof = g.y;
        float hl = d.x, hw = d.y;
        bool side = an.z > 0.62;
        bool top = an.y > 0.72;
        bool front = an.x > 0.66 && lp.x > 0.35 * hl;
        bool rear = an.x > 0.66 && lp.x < -0.35 * hl;
        bool slope = an.x > 0.28 && an.y > 0.30;      // windshield / rear glass
        bool semi_trailer = (type == kSemi && lp.x < -1.7);
        bool cab = !semi_trailer;

        // liveries first (the glass and lamps paint over them)
        if (type == kPolice) {
            albedo = kDark;                                  // black
            if (side && lp.x > -1.9 && lp.x < 0.95 && lp.y > 0.50 && lp.y < belt + 0.02)
                albedo = vec3(0.92);                         // white doors
            if (top && lp.y > roof - 0.02) albedo = vec3(0.92);
        } else if (type == kAmbulance) {
            albedo = vec3(0.93);
            if (lp.y > 1.16 && lp.y < 1.32 && !top) albedo = vec3(0.80, 0.06, 0.05);
            if (lp.y > 2.06 && lp.y < 2.18 && side) albedo = vec3(0.10, 0.25, 0.70);
        } else if (type == kSchoolBus) {
            albedo = vec3(0.95, 0.70, 0.10);
            if ((lp.y > 1.28 && lp.y < 1.38) || (lp.y > 0.80 && lp.y < 0.90)) albedo = kDark;
            if (lp.y < 0.62) albedo = kDark;                 // bumpers, skirt
            if (top && lp.y > roof + 0.2) albedo = vec3(0.92);   // white roof
        } else if (type == kFire) {
            albedo = vec3(0.75, 0.05, 0.04);
            if (lp.y > 1.55 && lp.y < 1.85 && !top) albedo = vec3(0.94);
            if (top && lp.x < 0.55 && lp.y > 3.15) albedo = kChrome;   // deck
            if (lp.y < 0.62) albedo = kChrome;               // bumper, running boards
        } else if (type == kSemi) {
            if (semi_trailer) {
                albedo = vec3(0.86, 0.87, 0.88);
                float rib = fract(lp.y * 2.5);
                if (!top && rib < 0.10) albedo *= 0.80;      // corrugation
                if (rear && abs(lp.z) < 0.025) albedo = kDark;   // door seam
                if (lp.y < 1.32) albedo = kDark;             // chassis
            } else {
                albedo = base;
                if (lp.y < 0.62) albedo = kChrome;           // bumper, tanks
            }
        } else if (type == kCyber) {
            albedo = vec3(0.70, 0.71, 0.73);
            if (lp.y < 0.62 && !top) albedo = kDark;         // lower cladding
        } else if (type == kSuv) {
            if (lp.y < 0.55 && !top) albedo = vec3(0.10, 0.10, 0.11);
        } else if (type == kPickup) {
            if (top && lp.x < -0.60 && lp.y > 1.25) albedo = vec3(0.14);   // bed cover
        }

        // ── wheel arches, door seams (sides) ────────────────────────
        if (side && cab) {
            // ── WHEEL ARCHES ────────────────────────────────────────
            // The hull now CUTS the opening (archTopAt in
            // vehicle_system.cpp): the skin recedes to the wheel-house
            // wall below the arc.  So this paint is no longer standing
            // in for a hole — it darkens the house that is really
            // there, and it must use the SAME ellipse the geometry cut
            // or the shading and the silhouette disagree by centimetres
            // all the way round the lip.  Keep in lockstep with
            // kArchMarginX / kArchMarginY.
            float wr = lamp.x;
            for (int k = 0; k < 2; ++k) {
                float wx = k == 0 ? d.z : d.w;
                float t = (lp.x - wx) / (wr + 0.10);
                if (abs(t) < 1.0) {
                    float top = wr + (wr + 0.13) * sqrt(1.0 - t * t);
                    if (lp.y < top) {
                        // the house: dark, and a shade darker deeper in
                        albedo = kDark;
                        // the lip itself keeps body colour for the last
                        // centimetre, so the arch reads as an edge of
                        // panel rather than as a painted band
                        if (lp.y > top - 0.02) albedo = mix(albedo, base, 0.65);
                    }
                }
            }
            // seams below the belt: the B pillar line and the door fronts
            float gx0 = g.z, gx1 = g.w;
            float pb = mix(gx0, gx1, 0.52);
            if (lp.y < belt && lp.y > 0.45 &&
                (abs(lp.x - pb) < 0.012 || abs(lp.x - (gx1 + 0.06)) < 0.012 ||
                 (type != kPickup && abs(lp.x - (gx0 - 0.05)) < 0.012)))
                albedo *= 0.55;
        }
        // ── glass ───────────────────────────────────────────────────
        if (cab) {
            float gx0 = g.z, gx1 = g.w;
            if (side && lp.y > belt + 0.02 && lp.y < roof - 0.03 &&
                lp.x > gx0 && lp.x < gx1) {
                glass = true;
                // pillars: cars have a B pillar, the long bodies one
                // every metre; the ambulance's rear box has a window too
                float pb = mix(gx0, gx1, 0.52);
                bool pillar = false;
                if (type == kSchoolBus || type == kAmbulance || type == kFire) {
                    pillar = fract((lp.x - gx0) / 1.05) < 0.07;
                    if (type == kAmbulance && lp.x < -0.4 && lp.y > belt + 0.55) pillar = true;
                } else {
                    pillar = abs(lp.x - pb) < 0.045;
                }
                if (pillar) { glass = false; albedo = kDark; }
            }
            if (slope && lp.y > belt + 0.04 && !side) glass = true;
            if (type == kCyber && slope && lp.x < -0.2) glass = false;   // the tonneau slope
        }
        if (glass) albedo = kGlassCol;
        if (kind == 0 && glass) discard;          // a window is a hole here
        if (kind == 3) {
            if (!glass) discard;                  // ...and only the window there
            alpha = 0.55;
        }
        // the inside of the hull, seen through a window: dark trim
        if (kind == 0 && !gl_FrontFacing) { albedo = kInteriorDark; glass = false; }

        // ── THE FACE: lamps, grille, plate, bumper ──────────────────
        // Everything here used to be one hard rectangle per feature,
        // which is why a headlight read as a white sticker: a real lamp
        // is a bezel around a lens, with a bright line along its top and
        // a hot spot where the projector is, and it is those three
        // steps of brightness — not the outline — that the eye uses.
        if (front) {
            float y0 = lamp.y, y1 = lamp.z;
            float az = abs(lp.z);
            float z0 = 0.28 * hw, z1 = 0.92 * hw;
            if (type == kCyber) {
                // the light BAR across the whole nose, this one's motif
                if (lp.y > y0 && lp.y < y1) {
                    albedo = vec3(0.75);
                    emissive = vec3(1.0, 0.97, 0.90) * 1.2;
                }
            } else if (lp.y > y0 - 0.02 && lp.y < y1 + 0.02 &&
                       az > z0 - 0.02 && az < z1 + 0.02) {
                vec2 q = vec2((lp.y - 0.5 * (y0 + y1)) / max(0.5 * (y1 - y0), 1e-3),
                              (az   - 0.5 * (z0 + z1)) / max(0.5 * (z1 - z0), 1e-3));
                float unit = rrect(q, 0.55);
                float lens = rrect(q * 1.26, 0.55);
                if (unit > 0.0) {
                    albedo = kDark * 1.4;                    // the bezel
                    if (lens > 0.0) {
                        albedo = vec3(0.78, 0.79, 0.80);     // the lens
                        emissive = vec3(0.95, 0.92, 0.85) * 0.55;
                        // daytime running strip along the lens top
                        if (q.x > 0.34 && q.x < 0.74) {
                            albedo = vec3(0.92);
                            emissive = vec3(1.0, 0.98, 0.92) * 1.15;
                        }
                        // the projector: a hot spot outboard of centre
                        if (length(vec2(q.x * 1.5, (q.y - 0.30) * 1.7)) < 0.42) {
                            albedo = vec3(0.95);
                            emissive = vec3(1.0, 0.98, 0.93) * 1.5;
                        }
                    }
                }
            } else if (lp.y > y0 - 0.06 && lp.y < y1 + 0.12 && az < 0.26 * hw) {
                // THE GRILLE: slats for everyone, not just the trucks —
                // a flat dark patch is the other half of why the face
                // read as painted on.
                albedo = (type == kSemi || type == kFire) ? kChrome : kDark;
                albedo *= 0.62 + 0.38 * step(0.45, fract(lp.y * 26.0));
                // and the bright line along its top edge
                if (lp.y > y1 + 0.04 && lp.y < y1 + 0.10) albedo = kChrome;
            }
            if (type == kSemi && lp.y > 1.15 && lp.y < 1.85 && az < 0.42 * hw)
                albedo = kChrome * (0.75 + 0.25 * step(0.5, fract(lp.y * 12.0)));
            if (lp.y < 0.50 && type != kSemi && type != kFire) {
                albedo = vec3(0.12);
                // number plate, low and central
                if (az < 0.24 * hw && lp.y > 0.30 && lp.y < 0.44)
                    albedo = vec3(0.86, 0.86, 0.82);
            }
        }
        if (rear && cab || (rear && semi_trailer)) {
            float t0 = lamp.w;
            float az = abs(lp.z);
            float z0 = 0.30 * hw, z1 = 0.94 * hw;
            if (type == kCyber) {
                if (lp.y > t0 && lp.y < t0 + 0.06) {
                    albedo = vec3(0.35, 0.03, 0.02);
                    emissive = vec3(1.0, 0.05, 0.02) * 1.3;
                }
            } else if (lp.y > t0 - 0.02 && lp.y < t0 + 0.20 &&
                       az > z0 - 0.02 && az < z1 + 0.02) {
                vec2 q = vec2((lp.y - (t0 + 0.09)) / 0.11,
                              (az - 0.5 * (z0 + z1)) / max(0.5 * (z1 - z0), 1e-3));
                float unit = rrect(q, 0.5);
                float lens = rrect(q * 1.24, 0.5);
                if (unit > 0.0) {
                    albedo = kDark * 1.3;
                    if (lens > 0.0) {
                        albedo = vec3(0.45, 0.045, 0.03);
                        emissive = vec3(0.85, 0.04, 0.02) * 0.5;
                        // the brake bar across the middle of the lens
                        if (abs(q.x) < 0.34) {
                            albedo = vec3(0.62, 0.05, 0.03);
                            emissive = vec3(1.0, 0.05, 0.02) * 0.95;
                        }
                        // reverse lamp: the inboard low corner, clear
                        if (q.y < -0.42 && q.x < -0.10) {
                            albedo = vec3(0.80, 0.79, 0.76);
                            emissive = vec3(0.0);
                        }
                    }
                }
            }
            if (lp.y < 0.50 && type != kSemi) {
                albedo = vec3(0.12);
                if (az < 0.24 * hw && lp.y > 0.30 && lp.y < 0.44)
                    albedo = vec3(0.86, 0.86, 0.82);
            }
        }
    } else if (kind == 1) {
        // ── A WHEEL: unit cylinder, axle along z ────────────────────
        float r = length(lp.xy);
        bool cap = an.z > 0.5;
        if (cap && r < 0.68) {
            float a = atan(lp.y, lp.x);
            bool spoke = abs(sin(a * 2.5)) > 0.62 || r < 0.18;
            albedo = spoke ? vec3(0.60, 0.61, 0.63) : vec3(0.10);
            if (r < 0.10) albedo = vec3(0.50);
        } else {
            albedo = vec3(0.035);
            if (!cap) { float a = atan(lp.y, lp.x); albedo *= 0.8 + 0.4 * step(0.5, fract(a * 6.0)); }
        }
    } else if (kind == 5) {
        // ── A SIGNAL: plain colour, lit by blink (v34: traffic-light
        //    poles, heads and lamps, stop signs) ───────────────────
        albedo = base;
        emissive = base * blink;
    } else {
        // ── A LIGHT BAR: unit box, red left / blue right ────────────
        vec3 lens = lp.x < 0.0 ? vec3(0.85, 0.05, 0.02) : vec3(0.05, 0.15, 0.95);
        albedo = lens * 0.6;
        if (blink > 0.5) {
            float phase = fract(blink * 2.4 + (lp.x < 0.0 ? 0.0 : 0.5));
            if (phase < 0.45) emissive = lens * 2.6;
        }
        if (an.y > 0.7 && lp.y < 0.0) albedo = kDark;   // underside
    }

    // ── shading: wrapped lambert + a car-paint highlight ────────────
    const vec3 kSunDir = normalize(vec3(-0.62, 0.62, -0.48));
    vec3 N = normalize(in_normal_ws);
    vec3 V = normalize(camera_info.position - in_position_ws);
    if (dot(N, V) < 0.0) N = -N;          // double-sided hulls
#ifdef GBUFFER_OUTPUT
    // Lamps keep their forward emissive colour; the packed G-buffer has
    // no RGB emission channel. Glass is drawn after the deferred resolve.
    if (dot(emissive, emissive) > 0.0) {
        out_albedo_ao = vec4(0.0); out_normal_rough = vec4(0.0);
        out_emissive_metal = vec4(0.0); out_velocity = vec2(0.0);
        return;
    }
    citizenGbuffer(albedo, N, in_position_ws, camera_info.view_proj,
                   camera_info.prev_view_proj, kind == 1 ? 0.9 : 0.35);
#else
    float nl = dot(N, kSunDir) * 0.5 + 0.5;
    vec3 H = normalize(kSunDir + V);
    float ndh = max(dot(N, H), 0.0);
    float ndv = max(dot(N, V), 0.0);
    float fres = pow(1.0 - ndv, 4.0);
    vec3 lit;
    // ── CAR PAINT: a basecoat under a clear coat ────────────────────
    // Everything else on the car is a plain surface; paint is not, and
    // the difference is why a repainted box still read as plastic.  A
    // real finish is two layers and the eye reads all three of their
    // signatures:
    //
    //   the BASECOAT's pigment, darkened as it turns away, plus — on a
    //   metallic — aluminium flake that returns light along the normal
    //   and dies at glancing angles, which is what makes silver look
    //   like metal and white look like paint under identical light;
    //
    //   the FLAKE's sparkle, which does the opposite: it only shows
    //   off-axis, and it is the fine glitter you see sweeping along a
    //   flank as a car turns;
    //
    //   the CLEAR COAT, a second and much sharper specular lobe with its
    //   own Fresnel sitting over both.  Its strength is the whole of the
    //   difference between gloss and a matte wrap, which is why it is a
    //   parameter of the paint (car_gen's `coat`) and not a constant.
    bool painted = (in_part >= 0.0) &&
                   (part == kPartBody || part == kPartLivery);
    if (painted) {
        float metal = in_paint.x, flake = in_paint.y, coat = in_paint.z;
        float grz = 1.0 - ndv;
        vec3 basecoat = albedo * (0.26 + 0.80 * nl) * (1.0 - 0.34 * metal);
        basecoat += albedo * metal * pow(nl, 2.0) * 0.85;
        basecoat += flake * pow(grz, 2.0) * nl * 1.25 * mix(albedo, vec3(1.0), 0.5);
        float clear = coat * pow(ndh, 220.0) * 1.7 + coat * fres * 0.32;
        lit = basecoat + clear * vec3(1.0, 0.99, 0.97);
    } else {
        float sp = pow(ndh, glass ? 90.0 : 48.0) * (glass ? 1.1 : 0.5);
        lit = albedo * (0.32 + 0.85 * nl);
        lit += sp * vec3(1.0, 0.98, 0.95);
        lit += fres * (glass ? 0.40 : 0.10) * vec3(0.55, 0.65, 0.80);
    }
    lit += emissive;
    outColor = vec4(sceneTonemapExposed(lit,
        sceneExposureScaleOf(camera_info.exposure_scale)), alpha);
#endif
}
