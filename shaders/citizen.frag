#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "tonemap.glsl.h"

// Colour arrives as a varying now, not a push constant: the whole
// population is drawn in ONE instanced call, so per-part data has to
// ride the vertex stream.  See citizen.vert.
layout(location = 0) in vec3 in_normal_ws;
layout(location = 1) in vec3 in_position_ws;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec3 in_local;
layout(location = 4) flat in vec4 in_extra;

layout(location = 0) out vec4 outColor;

// Camera UBO (the vertex stage binds the same set): only the exposure
// scale is read here, so citizens follow the Camera & Lens exposure.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

// ── GARMENT PAINTING ────────────────────────────────────────────────
// Every part is a unit box (-1..1) in its own space, +y up, +z the
// front of the body, drawn on a cube (far / mid tiers, kind 0) or the
// rounded mesh (fine tier).  in_extra = (kind, style, packed accent,
// seed) says what the part is and what to draw on it; the base colour
// is in_color.rgb and the accent (skin under a sleeve, hair on a head,
// belt on the hips) is unpacked from in_extra.z.  Kinds mirror
// PartKind in citizen_system.cpp.
//   head   style: +1 long hair, +2 female, +4 child
//   torso  style: 0 tee, 1 buttoned shirt, 2 henley; +4 long sleeves
//   thigh / shin  style: 0 jeans, 1 slacks, 2 shorts (the shin is bare)
//   shoe   style: 1 heels (open top)
const float kHead = 1.0, kTorso = 2.0, kPelvis = 3.0, kSleeve = 4.0,
            kForearm = 5.0, kHand = 6.0, kThigh = 7.0, kShin = 8.0,
            kShoe = 9.0;

vec3 unpackRGB(float f) {
    float r = floor(f / 65536.0);
    float g = floor((f - r * 65536.0) / 256.0);
    float b = f - r * 65536.0 - g * 256.0;
    return vec3(r, g, b) / 255.0;
}

vec3 paint(vec3 base, vec3 lp, float kind, float style, vec3 accent,
           float seed) {
    vec3 c = base;
    const vec3 kButton = vec3(0.92, 0.90, 0.85);
    const vec3 kDark = vec3(0.06, 0.05, 0.045);
    bool front = lp.z > 0.55;
    if (kind == kHead) {
        // hair: over the top, down the back and the sides; the front
        // hairline sits higher on a man, lower with a fringe on a
        // woman; a child's face is rounder -- eyes larger and set
        // lower, the hairline low
        bool child = style >= 4.0;
        float hst = mod(style, 4.0);
        bool female = hst >= 2.0;
        float hairline = child ? 0.38 + 0.08 * seed
                       : female ? 0.30 + 0.10 * seed : 0.42 + 0.14 * seed;
        bool hair = lp.y > hairline ||
                    (lp.z < -0.10 && lp.y > -0.35) ||
                    (abs(lp.x) > 0.80 && lp.y > 0.05 && lp.z < 0.45);
        if (hair) {
            c = accent;
        } else {
            // face: two eyes, a mouth, a touch of colour on the cheeks
            float ey = child ? 0.02 : 0.12;
            float er = child ? 0.12 : 0.10;
            float ex = abs(abs(lp.x) - 0.36);
            if (front && ex < er && abs(lp.y - ey) < 0.07) {
                c = kDark;
                if (ex < 0.045 && abs(lp.y - ey) < 0.035) c = vec3(0.02);
            } else if (front && abs(lp.y + (child ? 0.34 : 0.40)) < 0.035 &&
                       abs(lp.x) < 0.22) {
                c = base * 0.62;
            } else if (front && abs(abs(lp.x) - 0.55) < 0.16 &&
                       abs(lp.y + 0.12) < 0.16) {
                c = base * vec3(1.0, 0.93, 0.92);
            }
        }
    } else if (kind == kTorso) {
        float ts = mod(style, 4.0);            // 0 tee, 1 shirt, 2 henley
        bool longsl = style >= 4.0;
        if (ts >= 0.5 && ts < 1.5) {
            // buttoned shirt: open collar, placket with buttons, pocket
            if (front && lp.y > 0.70 && abs(lp.x) < (lp.y - 0.70) * 1.9) {
                c = accent;                              // open neck
            } else if (lp.y > 0.80 && abs(lp.x) < 0.62 && lp.z > -0.2) {
                c = mix(base, vec3(1.0), 0.18);          // collar
            } else if (front && abs(lp.x) < 0.075) {
                c = base * 0.86;                         // placket
                float by = fract((lp.y + 1.0) / 0.34);
                if (abs(lp.x) < 0.035 && abs(by - 0.5) < 0.09) c = kButton;
            } else if (front && lp.x < -0.20 && lp.x > -0.64 &&
                       lp.y > 0.12 && lp.y < 0.50) {
                bool rim = lp.x > -0.24 || lp.x < -0.60 || lp.y < 0.16 ||
                           lp.y > 0.46;
                c = rim ? base * 0.84 : base * 0.95;    // breast pocket
            }
        } else if (ts >= 1.5) {
            // henley: round neck, a short placket with three buttons
            if (front && lp.y > 0.80 && abs(lp.x) < 0.30) {
                c = accent;
            } else if (front && abs(lp.x) < 0.07 && lp.y > 0.20) {
                c = base * 0.86;
                float by = fract((lp.y - 0.20) / 0.20);
                if (abs(lp.x) < 0.035 && abs(by - 0.5) < 0.15) c = kButton;
            }
        } else {
            // tee: round neck
            if (front && lp.y > 0.84 && abs(lp.x) < 0.30) c = accent;
        }
        if (lp.y < -0.94) c = base * 0.88;              // hem
        if (!longsl && abs(lp.x) > 0.96 && lp.y > 0.2) c = base * 0.92;
    } else if (kind == kPelvis) {
        if (lp.y > 0.68) {
            c = accent;                                  // belt
            if (front && abs(lp.x) < 0.10) c = vec3(0.75, 0.68, 0.45);
        } else if (front && abs(lp.x) < 0.03 && lp.y > -0.3) {
            c = base * 0.82;                             // fly
        }
    } else if (kind == kSleeve) {
        bool longsl = style >= 4.0;
        if (!longsl) {
            if (lp.y < -0.30) c = accent;                // bare arm
            else if (lp.y < -0.16) c = base * 0.88;      // sleeve hem
        }
    } else if (kind == kForearm) {
        bool longsl = style >= 4.0;
        if (longsl && lp.y < -0.80) c = mix(base, vec3(1.0), 0.15);   // cuff
    } else if (kind == kThigh) {
        if (style < 0.5) {
            // jeans: outer seam, a lighter wear line on the front
            if (abs(lp.x) > 0.93 && abs(lp.z) < 0.10) c = base * 1.18;
            if (front && abs(lp.x) < 0.35) c = mix(c, base * 1.08, 0.5);
        } else if (style < 1.5) {
            // slacks: front crease
            if (front && abs(lp.x) < 0.06) c = base * 1.12;
        } else {
            // shorts: a hem above the knee
            if (lp.y < -0.80) c = base * 0.85;
        }
    } else if (kind == kShin) {
        if (style < 0.5) {
            if (abs(lp.x) > 0.93 && abs(lp.z) < 0.10) c = base * 1.18;
            if (front && abs(lp.x) < 0.35) c = mix(c, base * 1.08, 0.5);
            if (lp.y < -0.90) c = base * 0.85;                       // cuff
        } else if (style < 1.5) {
            if (front && abs(lp.x) < 0.06) c = base * 1.12;
            if (lp.y < -0.90) c = base * 0.85;
        } else {
            // a bare shin (shorts): a sock above the shoe
            if (lp.y < -0.72) c = vec3(0.92, 0.92, 0.90);
        }
    } else if (kind == kShoe) {
        if (lp.y < -0.50) c = mix(base, vec3(0.30), 0.55);      // sole
        else if (lp.z > 0.55) c = mix(base, vec3(1.0), 0.06);   // toe cap
        if (style > 0.5 && lp.y > 0.70) c = accent;              // heel: open top
    }
    return c;
}

// Simple wrapped-lambert + ambient: citizens are gameplay markers first
// and PBR objects second, so a stable readable shade beats a full BRDF.
// Finished through the shared scene tonemap so they sit in the same
// exposure as the world around them.
const vec3 kSunDir = normalize(vec3(-0.62, 0.62, -0.48));

void main() {
    vec3 n = normalize(in_normal_ws);
    vec3 albedo = in_color.rgb;
    if (in_extra.x > 0.5) {
        albedo = paint(albedo, in_local, in_extra.x, in_extra.y,
                       unpackRGB(in_extra.z), in_extra.w);
    }
    float nl = dot(n, kSunDir) * 0.5 + 0.5;          // wrapped
    vec3 lit = albedo * (0.35 + 0.85 * nl);
    lit += albedo * in_color.a;               // readability lift
    outColor = vec4(sceneTonemapExposed(lit,
        sceneExposureScaleOf(camera_info.exposure_scale)), 1.0);
}
