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

#include "citizen_gbuffer.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

const int kSedan = 0, kSuv = 1, kCyber = 2, kPickup = 3, kSemi = 4,
          kPolice = 5, kAmbulance = 6, kSchoolBus = 7, kFire = 8;

// per type: (belt, roof, side glass x0, side glass x1)
const vec4 kGlass[9] = vec4[9](
    vec4(1.00, 1.38, -1.55, 1.10),
    vec4(1.08, 1.60, -1.95, 1.05),
    vec4(1.14, 1.74, -1.30, 1.55),
    vec4(1.44, 1.90, -0.50, 1.00),
    vec4(2.35, 3.45, -1.50, 1.50),
    vec4(1.12, 1.72, -2.10, 1.15),
    vec4(1.38, 2.24, -2.30, 1.20),
    vec4(1.78, 2.72, -5.00, 2.95),
    vec4(2.10, 3.30, 0.75, 2.95));
// per type: (half length, half width, front wheel x, rear wheel x)
const vec4 kDims[9] = vec4[9](
    vec4(2.40, 0.925, 1.40, -1.40),
    vec4(2.22, 0.93, 1.33, -1.33),
    vec4(2.85, 1.00, 1.72, -1.72),
    vec4(2.95, 1.025, 1.85, -1.65),
    vec4(7.30, 1.275, 2.60, -1.05),
    vec4(2.50, 1.00, 1.50, -1.50),
    vec4(2.65, 0.975, 1.75, -1.55),
    vec4(5.25, 1.25, 3.30, -2.90),
    vec4(4.75, 1.25, 3.20, -2.30));
// per type: (wheel radius, headlight y0, headlight y1, tail y0)
const vec4 kLamp[9] = vec4[9](
    vec4(0.34, 0.62, 0.80, 0.68),
    vec4(0.36, 0.75, 0.92, 0.80),
    vec4(0.44, 0.93, 0.99, 1.04),
    vec4(0.43, 0.85, 1.05, 0.85),
    vec4(0.52, 0.85, 1.05, 1.35),
    vec4(0.38, 0.78, 0.95, 0.85),
    vec4(0.37, 0.75, 0.95, 0.90),
    vec4(0.52, 0.95, 1.15, 1.00),
    vec4(0.55, 1.10, 1.35, 1.10));

const vec3 kGlassCol = vec3(0.05, 0.065, 0.09);
// per type: (cabin floor y, dashboard x0) for the interior paint
const vec2 kCabin[9] = vec2[9](
    vec2(0.30, 0.95), vec2(0.36, 0.95), vec2(0.40, 1.30), vec2(0.45, 0.55),
    vec2(1.20, 1.15), vec2(0.38, 1.05), vec2(0.45, 1.55), vec2(0.55, 3.65),
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
    // the face normal in the hull's own space, from the position's
    // screen derivatives (sign is screen-facing dependent: use magnitudes)
    vec3 fn = normalize(cross(dFdx(lp), dFdy(lp)));
    vec3 an = abs(fn);

    vec3 base = in_color.rgb;
    vec3 albedo = base;
    vec3 emissive = vec3(0.0);
    bool glass = false;
    float alpha = 1.0;

    if (kind == 4) {
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
            float wr = lamp.x;
            for (int k = 0; k < 2; ++k) {
                float wx = k == 0 ? d.z : d.w;
                float dd = length(vec2(lp.x - wx, lp.y - wr));
                if (dd < wr + 0.10 && lp.y < wr + 0.12) albedo = kDark;
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

        // ── lamps, grille, bumpers ──────────────────────────────────
        if (front) {
            float y0 = lamp.y, y1 = lamp.z;
            float az = abs(lp.z);
            if (type == kCyber) {
                if (lp.y > y0 && lp.y < y1) emissive = vec3(1.0, 0.97, 0.90) * 1.2;
            } else if (lp.y > y0 && lp.y < y1 && az > 0.28 * hw && az < 0.92 * hw) {
                albedo = vec3(0.9); emissive = vec3(0.95, 0.92, 0.85) * 0.9;
            } else if (lp.y > y0 - 0.06 && lp.y < y1 + 0.12 && az < 0.26 * hw) {
                albedo = type == kSemi || type == kFire ? kChrome : kDark;   // grille
                if (type == kSemi || type == kFire)
                    albedo *= 0.75 + 0.25 * step(0.5, fract(lp.y * 12.0));
            }
            if (type == kSemi && lp.y > 1.15 && lp.y < 1.85 && az < 0.42 * hw)
                albedo = kChrome * (0.75 + 0.25 * step(0.5, fract(lp.y * 12.0)));
            if (lp.y < 0.50 && type != kSemi && type != kFire) albedo = vec3(0.12);
        }
        if (rear && cab || (rear && semi_trailer)) {
            float t0 = lamp.w;
            float az = abs(lp.z);
            if (type == kCyber) {
                if (lp.y > t0 && lp.y < t0 + 0.06) emissive = vec3(1.0, 0.05, 0.02) * 1.3;
            } else if (lp.y > t0 && lp.y < t0 + 0.18 && az > 0.30 * hw && az < 0.94 * hw) {
                albedo = vec3(0.55, 0.05, 0.03); emissive = vec3(0.9, 0.04, 0.02) * 0.6;
            }
            if (lp.y < 0.50 && type != kSemi) albedo = vec3(0.12);
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
    float sp = pow(max(dot(N, H), 0.0), glass ? 90.0 : 48.0) * (glass ? 1.1 : 0.5);
    float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    vec3 lit = albedo * (0.32 + 0.85 * nl);
    lit += sp * vec3(1.0, 0.98, 0.95);
    lit += fres * (glass ? 0.40 : 0.10) * vec3(0.55, 0.65, 0.80);
    lit += emissive;
    outColor = vec4(sceneTonemapExposed(lit,
        sceneExposureScaleOf(camera_info.exposure_scale)), alpha);
#endif
}
