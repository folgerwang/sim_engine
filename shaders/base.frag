#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#define ALPHAMODE_MASK 1

// Render-debug visualisation is now controlled at runtime via the
// FEATURE_INPUT_DEBUG_MODE bits of camera_info.input_features (set by the
// "Render Debug" combo in the menu) and dispatched at the bottom of main().
// The old compile-time DEBUG_BASE_COLOR / DEBUG_MIP_LEVEL toggles are
// removed because the runtime path covers their use cases.

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};
// Camera & Lens exposure for pbr_lighting.glsl.h's toneMap (see there).
#define SCENE_EXPOSURE_SCALE_EXPR sceneExposureScaleOf(camera_info.exposure_scale)

// Push constant — shared with base.vert.  We only read debug_force_red
// here; the vertex shader is the canonical reader of the matrix +
// flip_uv_coord + cascade_idx fields.  Declaring the whole struct
// keeps the layout identical across stages so the driver doesn't
// complain about a layout mismatch.
layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
};

#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PBR_CONSTANT_INDEX) uniform MaterialUniformBufferObject {
    PbrMaterialParams material;
};
#endif

layout(set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX) uniform RuntimeLightsUniformBufferObject {
    RuntimeLightsParams runtime_lights;
};

#include "ibl.glsl.h"

layout(location = 0) in ObjectVsPsData ps_in_data;

#ifdef GBUFFER_OUTPUT
// Deferred permutation (base_frag*_GBUF.spv): the classic drawable path
// re-rasterises into the cluster G-buffer instead of shading forward,
// exactly the way tile_gbuf_frag re-rasterises terrain.  Layout and
// packing MUST match cluster_bindless.frag's GBUFFER_OUTPUT branch and
// tile.frag — deferred_resolve.comp decodes all three identically.
layout(location = 0) out vec4 out_albedo_ao;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec4 out_emissive_metal;
layout(location = 3) out vec2 out_velocity;

// The forward path's code after our early return still names outColor;
// a plain global (not an output) lets it compile and the compiler
// dead-strips everything past the return.
vec4 outColor;

// Octahedral encode — identical to cluster_bindless.frag's copy, paired
// with octDecode in deferred_resolve.comp.
vec2 octEncodeDir(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy
                            : (1.0 - abs(n.yx)) * vec2(
                                  n.x >= 0.0 ? 1.0 : -1.0,
                                  n.y >= 0.0 ? 1.0 : -1.0);
    return oct * 0.5 + 0.5;
}
#elif defined(GLASS_ATTR)
// Glass/water attribute permutation (base_frag*_GLASS.spv): the glass
// panes of windows and doors re-rasterise into the two GLASS ATTRIBUTE
// targets after the opaque G-buffer passes; deferred_resolve.comp then
// shades them with REAL ray-traced reflection and refraction against
// the cluster TLAS/BVH.  Packing (decoded in the resolve):
//   attr0 = octEncode(N).xy, linear view depth (m), roughness
//   attr1 = transmission tint rgb, KIND — 0.25+0.5*alpha for glass
//           (encodes the material alpha), 1.0 for water (written by
//           the terrain water-attr pass, not this shader)
layout(location = 0) out vec4 out_glass_nr;
layout(location = 1) out vec4 out_glass_tint;

vec4 outColor;   // satisfies the dead code past our early return

vec2 octEncodeDir(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy
                            : (1.0 - abs(n.yx)) * vec2(
                                  n.x >= 0.0 ? 1.0 : -1.0,
                                  n.y >= 0.0 ? 1.0 : -1.0);
    return oct * 0.5 + 0.5;
}
#else
layout(location = 0) out vec4 outColor;
#endif

#ifdef DECAL
// ── Ground-decal soft blend ──────────────────────────────────────────
// Road ribbons are laid ON the terrain, so wherever the ribbon and the
// ground agree the decal should paint solid, and wherever the ribbon
// lifts away from -- or slices through -- what is already in the depth
// buffer it should dissolve instead of drawing a hard silhouette edge.
//
// The blend factor is the distance, in metres along the view ray,
// between this fragment and the scene depth already rendered (terrain
// first, then props).  DECAL_FADE_START is a tolerance rather than
// zero: build_road_spline_mesh crowns the carriageway ~0.20 m above the
// shoulder deliberately, for camber and for z-fight margin, so a fade
// starting at 0 would make the road CENTRE transparent and its EDGE
// opaque -- exactly backwards.  Inside the tolerance the ribbon is
// solid; past DECAL_FADE_END it is gone.
// Both are PERPENDICULAR metres -- see the ray_gap -> depth_gap
// conversion in main().  Reading them as "how far off the ground",
// independent of where the camera stands, is the whole point.
const float DECAL_FADE_START = 0.30;   // metres of intentional lift
const float DECAL_FADE_END   = 1.20;   // fully dissolved beyond this

// depth_params.xy are proj[2].z and proj[3].z, so this inverts the
// projection's z row without caring whether the depth range is [-1,1]
// or [0,1]:
//     clip.z = A*z_view + B,   clip.w = -z_view
//  => window_z = B/d - A    => d = B / (window_z + A)
// Returns positive metres in front of the eye.
float linearizeSceneDepth(float window_z) {
    return camera_info.depth_params.y /
           (window_z + camera_info.depth_params.x);
}

// ── Decal coverage ───────────────────────────────────────────────────
// The ground-contact fade times the ground-clutter distance fade, in
// [0, 1].  Factored out of main() because BOTH decal permutations need
// the identical number: the forward one (DECAL) multiplies it into
// outColor.a, and the deferred one (DECAL + GBUFFER_OUTPUT) hands it to
// the G-buffer pipeline's SRC_ALPHA blend.  Two copies of this would
// drift, and a drift here shows up as the road dissolving at a
// different distance depending on which shadow technique is selected —
// which is exactly the class of bug the deferred decal path exists to
// remove.
//
// The depth copy is the same resolution and the same viewport as this
// render target, so gl_FragCoord indexes it directly -- no screen-size
// uniform and no filtering wanted (a filtered depth read would
// interpolate across silhouettes and invent surfaces that are not
// there).
float decalCoverage(vec3 view_dir, vec3 geom_normal, vec3 world_pos) {
    float scene_win_z =
        texelFetch(scene_depth_sampler, ivec2(gl_FragCoord.xy), 0).r;
    // Positive when the decal floats in FRONT of the scene, which is
    // the only direction worth fading -- a decal behind the scene has
    // already been rejected by the depth test.
    float ray_gap =
        max(linearizeSceneDepth(scene_win_z) -
            linearizeSceneDepth(gl_FragCoord.z), 0.0);
    // ── Along-ray gap -> PERPENDICULAR gap ───────────────────────────
    // What DECAL_FADE_START/END want to measure is "how far is this
    // decal floating off the surface underneath it" -- a property of
    // the geometry alone.  ray_gap is not that: it is the separation
    // measured ALONG THE VIEW RAY, which for two near-parallel
    // surfaces is perpendicular_gap / |dot(V, N)| and therefore blows
    // up at grazing angles.  A ground-hugging quad lifted 9 cm, seen
    // from eye height 1.7 m at 20 m range, has sin(elevation) ~ 0.085
    // and so reads as a 1.06 m gap -- past DECAL_FADE_END, i.e. it
    // dissolves to nothing precisely where a player stands and looks
    // out across the ground.  That is why the road only ever looked
    // right close up and under the feet, and why ground clutter was
    // invisible entirely.
    //
    // Multiplying back by |dot(V, N)| recovers the perpendicular
    // separation, which is view-independent: a decal 9 cm off the
    // ground now reads as 9 cm from every angle.  view_dir is the
    // fragment->eye direction and geom_normal the GEOMETRIC normal,
    // both world-space and both already computed by the caller for the
    // lighting.
    //
    // Exactly edge-on (dot -> 0) the perpendicular gap tends to zero
    // and the decal stays solid, which is the correct limit: a surface
    // seen edge-on has no visible float to dissolve.  No epsilon floor
    // here on purpose -- a floor would reintroduce the angle
    // dependence this line exists to remove.
    float depth_gap = ray_gap * abs(dot(view_dir, geom_normal));
    float coverage =
        1.0 - smoothstep(DECAL_FADE_START, DECAL_FADE_END, depth_gap);

    // ── Ground-clutter distance fade ─────────────────────────────────
    // Zero end distance = "this decal does not fade with range" (the
    // road ribbon), so the whole thing costs one scalar compare there.
    // Distance is measured in world space from the eye to the FRAGMENT
    // rather than to the patch centre: a 3 m quad viewed edge-on at the
    // fade boundary would otherwise pop as a unit, and per-fragment
    // costs nothing extra since vertex_position is already interpolated
    // in for the lighting.
    if (model_params.clutter_fade_end_m > 0.0) {
        float view_dist = distance(camera_info.position.xyz, world_pos);
        coverage *= 1.0 - smoothstep(model_params.clutter_fade_start_m,
                                     model_params.clutter_fade_end_m,
                                     view_dist);
    }
    return coverage;
}

// ── Screen-door dissolve for MASK-mode decal cards ───────────────────
// The glTF authors the clutter/meadow cards as MASK, so the decal
// permutations cut instead of blending: texture alpha tests against the
// material cutoff, and the coverage above becomes a per-pixel
// interleaved-gradient threshold, so the distance fade survives without
// any partial transparency.  Returns true when this fragment should be
// discarded.  Shared for the same anti-drift reason as decalCoverage.
bool decalScreenDoorCull(float base_alpha, float cutoff, float coverage) {
    float ign = fract(52.9829189 *
                      fract(dot(gl_FragCoord.xy,
                                vec2(0.06711056, 0.00583715))));
    return base_alpha < cutoff || coverage <= ign;
}
#endif // DECAL

#include "pbr_lighting.glsl.h"

// PCSS soft shadow with cascade-consistent WORLD-SPACE blur radius.
// See deferred_resolve.comp for full tuning notes; constants MUST
// match across that file, cluster_bindless.frag, and base.frag.
const float CSM_NORMAL_BIAS_SCALE     = 0.05;
// Depth bias in WORLD units.  Converted per-cascade in
// calculateShadowFactor().  See deferred_resolve.comp for the full
// rationale.
const float CSM_DEPTH_BIAS_BASE_WORLD  = 0.05;
const float CSM_DEPTH_BIAS_SLOPE_WORLD = 0.20;
const float CSM_LIGHT_SIZE_WORLD      = 0.10;   // halved: tighter penumbra, less visible dither
const float CSM_BLOCKER_RADIUS_WORLD  = 0.20;   // halved with light size
const float CSM_MIN_PCF_RADIUS_WORLD  = 0.02;
const float CSM_MAX_PCF_RADIUS_WORLD  = 0.40;   // halved with light size
const float CSM_RANGE_FADE_START      = 0.85;  // fade shadows out over the last 15% of CSM range
const int   CSM_BLOCKER_SAMPLES       = 16;
const int   CSM_PCF_SAMPLES           = 16;

vec2 csmVogelDisk(int i, int n, float phi) {
    const float GOLDEN_ANGLE = 2.39996323;
    float r     = sqrt((float(i) + 0.5) / float(n));
    float theta = GOLDEN_ANGLE * float(i) + phi;
    return vec2(r * cos(theta), r * sin(theta));
}

float csmIGN(vec2 pixel) {
    return fract(52.9829189 *
                 fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

// ── CSM coverage helpers (out-of-range handling) ─────────────────────
// Fetch one shadow-map texel, treating anything outside the cascade's
// [0,1] UV box as "nothing between this point and the light".
//
// The shadow sampler is CLAMP_TO_EDGE and the CSM silhouette prepass
// clears the map to 0.0 outside each cascade's camera-frustum
// silhouette, so an out-of-bounds read returns 0.0 = "an occluder sits
// right at the light" — the blocker search then finds a fake blocker
// and the receiver resolves fully black.  Two places hit that path:
// border taps of the LAST cascade (no cascade left to be promoted
// into), and every receiver past the CSM's max range, whose projection
// lands far outside the last cascade's box.  The latter is what turns
// the whole far field (last split → camera far plane) black behind a
// hard, camera-dependent boundary.  With no data out there, "lit" is
// the only honest answer.
float csmSampleMap(vec2 tap_uv, int cascade) {
    if (any(lessThan(tap_uv, vec2(0.0))) ||
        any(greaterThan(tap_uv, vec2(1.0))))
        return 1.0;
    return texture(direct_shadow_sampler, vec3(tap_uv, float(cascade))).r;
}

// View-space distance at which the CSM runs out of data — the last
// cascade's far split (cascade_splits is packed vec4[2]).
float csmMaxRange() {
    return runtime_lights.cascade_splits[(CSM_CASCADE_COUNT - 1) >> 2]
                                        [(CSM_CASCADE_COUNT - 1) & 3];
}

// 0 well inside the covered range → 1 at/past the last split.  Fading
// the shadow term to fully lit across the tail of the last cascade
// keeps the edge of shadow coverage from reading as a visible ring.
float csmRangeFade(float view_depth) {
    float max_range  = csmMaxRange();
    float fade_begin = max_range * CSM_RANGE_FADE_START;
    return clamp((view_depth - fade_begin) /
                 max(max_range - fade_begin, 1e-4), 0.0, 1.0);
}

float calculateShadowFactor(
    vec3 position_world, vec3 normal_world, vec2 screen_pixel) {
    // Cascade selection by view-space depth.
    vec4 position_view = camera_info.view * vec4(position_world, 1.0);
    float view_depth = -position_view.z;

    // ── Out-of-max-range guard ──────────────────────────────────────
    // Past the last split no cascade covers this receiver: the depth
    // selection below falls through to the last cascade, the projection
    // lands outside its UV/depth box, and the clamped border reads make
    // the entire far field resolve as occluded.  Fade the shadow term
    // out over the tail of the last cascade and bail out completely
    // once we are past it.
    float range_fade = csmRangeFade(view_depth);
    if (range_fade >= 1.0) return 1.0;

    int cascade_idx = CSM_CASCADE_COUNT - 1;
    for (int i = 0; i < CSM_CASCADE_COUNT; ++i) {
        // cascade_splits is packed vec4[2]; index as [i/4][i%4].
        if (view_depth < runtime_lights.cascade_splits[i >> 2][i & 3]) {
            cascade_idx = i;
            break;
        }
    }

    // Normal-offset bias.
    vec3  N     = normalize(normal_world);
    vec3  L     = normalize(-runtime_lights.lights[0].direction);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    vec3  biased_world = position_world +
                         N * ((1.0 - NdotL) * CSM_NORMAL_BIAS_SCALE);

    // Per-cascade scale factors (see deferred_resolve.comp) + cascade
    // PROMOTION: a receiver on the outer rim of its depth-selected
    // cascade would sample the PCSS kernel outside the map (edge-clamp
    // prefill = "no blocker" = LIT band along the cascade boundary).
    // Step up until the kernel fits — mirrors deferred_resolve.comp.
    vec2  shadow_uv;
    float current_depth;
    float w2uv;
    for (;;) {
        vec4 position_light_clip =
            runtime_lights.light_view_proj[cascade_idx] *
            vec4(biased_world, 1.0);
        vec3 position_light_NDC =
            position_light_clip.xyz / position_light_clip.w;
        shadow_uv     = position_light_NDC.xy * 0.5 + 0.5;
        current_depth = position_light_NDC.z;
        w2uv = 0.5 *
            length(runtime_lights.light_view_proj[cascade_idx][0].xyz);

        if (cascade_idx >= CSM_CASCADE_COUNT - 1) {
            // Last cascade — nothing left to promote into.  If the
            // receiver projects outside its map or its depth slab there
            // is no shadow data for it, so return lit instead of
            // sampling clamped border texels.
            if (any(lessThan(shadow_uv, vec2(0.0))) ||
                any(greaterThan(shadow_uv, vec2(1.0))) ||
                current_depth < 0.0 || current_depth > 1.0)
                return 1.0;
            break;
        }

        float kernel_uv =
            max(CSM_BLOCKER_RADIUS_WORLD, CSM_MAX_PCF_RADIUS_WORLD) * w2uv
            + 1.0 / 2048.0;
        bool fits =
            all(greaterThanEqual(shadow_uv, vec2(kernel_uv))) &&
            all(lessThanEqual(shadow_uv, vec2(1.0 - kernel_uv))) &&
            current_depth >= 0.0 && current_depth <= 1.0;
        if (fits) break;
        ++cascade_idx;
    }
    float z_scale = length(runtime_lights.light_view_proj[cascade_idx][2].xyz);

    float depth_bias =
        (CSM_DEPTH_BIAS_BASE_WORLD +
         CSM_DEPTH_BIAS_SLOPE_WORLD * (1.0 - NdotL)) * z_scale;

    float blocker_radius_uv = CSM_BLOCKER_RADIUS_WORLD * w2uv;
    float light_size_uv     = CSM_LIGHT_SIZE_WORLD     * w2uv;
    float min_pcf_radius_uv = CSM_MIN_PCF_RADIUS_WORLD * w2uv;
    float max_pcf_radius_uv = CSM_MAX_PCF_RADIUS_WORLD * w2uv;

    float phi = csmIGN(screen_pixel) * 6.28318530718;

    // PCSS step 1: blocker search.
    float blocker_sum   = 0.0;
    int   blocker_count = 0;
    for (int i = 0; i < CSM_BLOCKER_SAMPLES; ++i) {
        vec2 off = csmVogelDisk(i, CSM_BLOCKER_SAMPLES, phi)
                       * blocker_radius_uv;
        float d = csmSampleMap(shadow_uv + off, cascade_idx);
        if (d < current_depth - depth_bias) {
            blocker_sum += d;
            ++blocker_count;
        }
    }
    if (blocker_count == 0) return 1.0;
    float avg_blocker_depth = blocker_sum / float(blocker_count);

    // PCSS step 2: penumbra estimate.
    float penumbra = (current_depth - avg_blocker_depth) /
                     max(avg_blocker_depth, 1e-4);
    float pcf_radius = clamp(penumbra * light_size_uv,
                             min_pcf_radius_uv,
                             max_pcf_radius_uv);

    // PCSS step 3: PCF at computed radius.
    float sum = 0.0;
    for (int i = 0; i < CSM_PCF_SAMPLES; ++i) {
        vec2 off = csmVogelDisk(i, CSM_PCF_SAMPLES, phi) * pcf_radius;
        float closest_depth = csmSampleMap(shadow_uv + off, cascade_idx);
        sum += (closest_depth < 1.0 &&
                current_depth > closest_depth + depth_bias)
               ? 0.0 : 1.0;
    }
    return mix(sum * (1.0 / float(CSM_PCF_SAMPLES)), 1.0, range_fade);
}

// ── Interior sky occlusion (forward path) ────────────────────────────
// The IBL cubes hold OPEN-SKY irradiance, and iblLighting() adds them
// unconditionally — no AO, no visibility term.  Outdoors that is right;
// inside a house it means a room with one small window receives exactly
// the same skylight as the lawn outside, which is why interiors render
// as bright as noon with no lights placed.
//
// The deferred path solves this properly: deferred_resolve.comp REPLACES
// the flat ambient with a traced 1-bounce GI estimate and scales the
// specular IBL by the traced sky visibility.  The forward path has no
// ray tracer bound (no BVH/TLAS in its descriptor sets), so it gets the
// cheap approximation instead: geometry the generator already marked as
// interior (MODEL_FLAG_INTERIOR) keeps this fraction of the sky term.
//
// 0.10 is the rule-of-thumb daylight factor for a room with modest
// glazing — real rooms sit around 2-10% of the outdoor horizontal
// illuminance.  Direct sun through the window still arrives at full
// strength via the punctual term below, so a sunbeam on the floor stays
// bright while the room around it falls back to a plausible interior
// level.  This is an APPROXIMATION, not a light transport solution: it
// cannot know where the windows are.
// 0.18 (was 0.10): with the panes now truly translucent the window
// itself reads bright, and a 10% interior against that contrast felt
// like a cellar.  Still inside the plausible daylight-factor band for
// a well-glazed room.
const float kInteriorSkyAmbient = 0.18;

// ── LOD band cross-fade dissolve ─────────────────────────────────────
// See ModelParams::lod_fade.  A tile inside a band transition is drawn
// by BOTH neighbouring bands; this screen-door test decides, per pixel,
// which one keeps it.  The two sides get exactly complementary
// comparisons against the same interleaved-gradient value, so the union
// of what they keep is every pixel and the intersection is (almost)
// none — no holes opening up mid-transition and no double-drawn
// silhouettes.  Costs one compare on ordinary geometry (lod_fade 1.0).
bool lodFadeDiscards(float w) {
    // 0 = no dissolve.  This is BOTH the common case and the value every
    // zero-initialised ModelParams carries, so it must mean "draw".
    if (w == 0.0) return false;
    if (w >= 0.999) return false;          // steady
    if (w <= -0.999) return false;         // fully faded in
    float ign = fract(52.9829189 *
                      fract(dot(gl_FragCoord.xy,
                                vec2(0.06711056, 0.00583715))));
    return (w >= 0.0) ? (w <= ign) : (-w <= 1.0 - ign);
}

void main() {
    // Band transition: dissolve before any shading work is done.
    // Per-instance mode (model_params_pad0 != 0, dense ground cover)
    // uses the weight base.vert computed from the instance's own
    // camera distance; |w| ~ 0 means the instance sits entirely
    // outside its band and drops whole.  The mid-transition dissolve
    // reuses lodFadeDiscards' complementary screen-door — both bands
    // derive w from the same shared instance translation, so their
    // kept pixels partition exactly.  Node-tile mode is unchanged.
    // The node-table path (MODEL_FLAG_NODE_TABLE) always carries its
    // dissolve weight in the varying: the push constant's lod_fade is
    // the bucket base slot there, not a weight.
    if (floatBitsToUint(model_params.model_params_pad0) != 0u ||
        (model_params.flip_uv_coord & MODEL_FLAG_NODE_TABLE) != 0u) {
        float w_ci = ps_in_data.vertex_ilod_fade;
        if (abs(w_ci) < 0.001) discard;
        if (lodFadeDiscards(w_ci)) discard;
    } else if (lodFadeDiscards(model_params.lod_fade)) {
        discard;
    }
    bool is_front_face = gl_FrontFacing;
#ifndef NO_MTL
    vec4 baseColor = getBaseColor(ps_in_data, material);
#ifdef ALPHAMODE_OPAQUE
    baseColor.a = 1.0;
#endif // ALPHAMODE_OPAQUE
#else
    vec4 baseColor = vec4(0);
#endif
    

#ifndef NO_MTL
#ifdef MATERIAL_UNLIT
    outColor = (vec4(linearTosRGB(baseColor.rgb), baseColor.a));
    return;
#endif // MATERIAL_UNLIT
    vec3 v = normalize(camera_info.position.xyz - ps_in_data.vertex_position);
    NormalInfo normal_info = getNormalInfo(ps_in_data, material, v, is_front_face);

    MaterialInfo material_info =
        setupMaterialInfo(
            ps_in_data,
            material,
            normal_info,
            v,
            baseColor.xyz);

#ifdef GLASS_ATTR
    // ── Glass: write reflection/refraction attributes and stop ───────
    // Only Blend-mode (glass-forced) primitives reach this pipeline —
    // DrawableObject::draw filters by material in DrawMode::kGlassAttr.
    // The pass depth-tests LEQUAL against the opaque depth with writes
    // off, so panes behind walls never shade; among overlapping panes
    // the last write wins (window glass rarely stacks on screen).
    {
        float glass_linz = camera_info.depth_params.y /
                           (camera_info.depth_params.x + gl_FragCoord.z);
        // Face the normal toward the viewer: a pane seen from either
        // side reflects on the side you look at.
        vec3 gn = normal_info.n;
        if (dot(gn, v) < 0.0) gn = -gn;
        // Real window glass is smooth whatever the albedo pipeline's
        // default roughnessFactor says — clamp so the traced
        // reflection stays a reflection.
        out_glass_nr = vec4(octEncodeDir(gn),
                            glass_linz,
                            min(material_info.perceptualRoughness, 0.08));
        float glass_alpha = clamp(baseColor.a, 0.0, 1.0);
        // Transmission tint: clear glass passes ~90% of the light with
        // only a HINT of the authored pane colour.  Multiplying by the
        // raw window-texture albedo (a dark blue) crushed everything
        // seen through a window toward black — the texture is how the
        // pane looked as an OPAQUE quad, not a transmittance spectrum.
        vec3 glass_tint = mix(vec3(0.92), normalize(baseColor.rgb + 1e-4)
                                              * 0.92, 0.35);
        out_glass_tint = vec4(glass_tint,
                              0.25 + 0.5 * glass_alpha);
    }
    return;
#endif // GLASS_ATTR

#ifdef GBUFFER_OUTPUT
    // ── Deferred: write material attributes and stop ─────────────────
    // No lighting runs in this permutation — deferred_resolve.comp does
    // PBR once per visible pixel with the traced shadow / RT-GI /
    // traced-sky-visibility path, which is the entire point of routing
    // these drawables through the G-buffer (a forward-shaded house gets
    // the raw unoccluded IBL cubes; a deferred one gets a room that
    // actually goes dark).  The LOD dissolve / per-instance-band
    // discards already ran at the top of main(), so the G-buffer
    // respects the same dithered band handoff the forward pass shows.
#if defined(ALPHAMODE_MASK) && !defined(DECAL)
    // The forward path's cutout discard sits AFTER its lighting; here it
    // must run before the writes or masked foliage would stamp opaque
    // rectangles into the G-buffer.
    if (baseColor.a < material.alpha_cutoff) {
        discard;
    }
#endif // ALPHAMODE_MASK

    // ── Deferred decals (DECAL + GBUFFER_OUTPUT) ─────────────────────
    // A ground decal is not a surface of its own: it is a change to the
    // ALBEDO of the surface it lies on.  Writing it into the G-buffer
    // and letting deferred_resolve.comp light the combined result once
    // is therefore both cheaper and more correct than lighting the
    // decal separately and compositing — and it is the only way the
    // decal can pick up the traced shadow / RT-AO / RT-GI the ground
    // under it gets.  Drawn forward AFTER the resolve (as it used to
    // be) the decal had no shadow at all in any RT mode.  (Historical
    // note: SHADOW_DISABLED used to be raised whenever an RT technique
    // armed; nowadays CSM stays alive for the forward writers in the
    // RT modes and only SSRT / the explicit shadow-off toggle raise it
    // — the G-buffer route stays the correct one for decals either
    // way.)
    //
    // The coverage goes in .a, which the pipeline's colour blend reads
    // as SRC_ALPHA.  The ALPHA channel's own blend is ZERO/ONE, so the
    // >= 0.5 "G-buffer written" sentinel the terrain already stamped
    // here survives untouched — see createDrawableDecalGbufferPipeline.
    // Consequence worth knowing: a decal over a pixel NO deferred
    // writer covered stays invisible (sentinel 0 → the resolve skips
    // it).  That fails safe, and every surface decals are authored onto
    // (terrain tiles, placed props) does go through the G-buffer.
    float gbuf_alpha = 1.0;
#ifdef DECAL
    {
        float coverage =
            decalCoverage(v, normal_info.ng, ps_in_data.vertex_position);
        if ((material.material_features & FEATURE_MATERIAL_ALPHA_MASK) != 0u) {
            // Foliage cards: cut, don't blend.  Survivors are fully
            // opaque so overlapping cards stop compositing into soup —
            // identical policy to the forward decal branch.
            if (decalScreenDoorCull(baseColor.a, material.alpha_cutoff,
                                    coverage)) {
                discard;
            }
            gbuf_alpha = 1.0;
        } else {
            // Road skirt & true decals: baseColor.a still carries the
            // material's own alpha (the road-fade skirt's texture ramp),
            // so the two multiply rather than one overriding the other.
            gbuf_alpha = baseColor.a * coverage;
        }
    }
#endif // DECAL

    // .a >= 0.5 is the resolve's "G-buffer written" sentinel — see
    // tile.frag, which compresses its AO into [0.5, 1] for the same
    // reason.  The drawable path has no baked AO, so a flat 1.0 (the
    // decal permutation overrides it with its coverage, above).
    out_albedo_ao = vec4(baseColor.rgb, gbuf_alpha);
    // flags.w = 0: no foliage-SSS classification on this path (parity
    // with the forward branch, which never applied foliageTranslucency
    // to classic drawables either).
    out_normal_rough = vec4(
        octEncodeDir(normal_info.n),
        material_info.perceptualRoughness, 0.0);
    vec2 oct_geom_dr = octEncodeDir(normal_info.ng);
    out_emissive_metal =
        vec4(oct_geom_dr.x, oct_geom_dr.y, 0.0, material_info.metallic);
    // Static-world velocity from the camera matrices, exactly like
    // tile.frag: this permutation is only built for NON-skinned vertex
    // layouts (see CompileShaders.cmake), so world positions are
    // camera-relative-constant and the matrix delta IS the velocity.
    {
        vec4 cur_clip =
            camera_info.view_proj *
            vec4(ps_in_data.vertex_position, 1.0);
        vec4 prev_clip =
            camera_info.prev_view_proj *
            vec4(ps_in_data.vertex_position, 1.0);
        out_velocity =
            cur_clip.xy / cur_clip.w - prev_clip.xy / prev_clip.w;
    }
    return;
#endif // GBUFFER_OUTPUT

#if !defined(GLASS_ATTR) && !defined(DECAL)
    // ── Deferred-relight fast path ───────────────────────────────────
    // When the deferred G-buffer re-rasterise + resolve is armed this
    // frame (FEATURE_INPUT_DEFERRED_RELIGHT) and THIS draw is one the
    // re-rasterise covers (MODEL_FLAG_DEFERRED_RELIGHT — CPU-set, never
    // on skinned nodes, which have no _GBUF permutation), every colour
    // this branch could produce is overwritten by deferred_resolve.comp.
    // Running the full IBL + punctual stack here was the largest slice
    // of the forward pass, shading pixels whose lighting was then thrown
    // away.  Emit a cheap flat approximation instead.  Depth, the LOD
    // dissolve discards at the top of main(), and the cutout discard all
    // still run, so depth and coverage stay bit-identical to the full
    // path — only the doomed colour is cheapened.
    if ((camera_info.input_features & FEATURE_INPUT_DEFERRED_RELIGHT) != 0u &&
        (model_params.flip_uv_coord & MODEL_FLAG_DEFERRED_RELIGHT) != 0u) {
#if defined(ALPHAMODE_MASK)
        // Same late cutout discard as the full path below.
        if (baseColor.a < material.alpha_cutoff) {
            discard;
        }
#endif // ALPHAMODE_MASK
        float fast_nl = 0.5f;
#ifdef USE_PUNCTUAL
        fast_nl = max(dot(normal_info.n,
                          normalize(-runtime_lights.lights[0].direction)),
                      0.0f);
#endif // USE_PUNCTUAL
        outColor = vec4(
            linearTosRGB(baseColor.rgb * (0.25f + 0.5f * fast_nl)), 1.0f);
        return;
    }
#endif // !GLASS_ATTR && !DECAL

    // Skip shadow sampling when the pass is disabled (avoids stale/zero CSM
    // texture reads that would incorrectly shadow the whole scene).
    float shadow = 1.0;
    if ((camera_info.input_features & FEATURE_INPUT_SHADOW_DISABLED) == 0u) {
        // Pass the GEOMETRIC normal (normal_info.ng) — not the
        // normal-mapped one — for shadow biasing.  Normal-mapped detail
        // can produce inconsistent bias at texel scale and re-introduce
        // acne on bumpy surfaces.  gl_FragCoord.xy is the dither key
        // for per-pixel Vogel-disk rotation.
        shadow = calculateShadowFactor(
            ps_in_data.vertex_position,
            normal_info.ng,
            gl_FragCoord.xy);
    }

    bool light_from_back = false;
#ifdef USE_PUNCTUAL    
    if (dot(normal_info.ng, runtime_lights.lights[0].direction) > 0)
        light_from_back = true;
#endif

#ifdef DOUBLE_SIDED
    // LIGHTING
    PbrLightsColorInfo back_color_info = initColorInfo();
    NormalInfo back_normal_info = normal_info;
    back_normal_info.ng = -normal_info.ng;
    back_normal_info.n = -normal_info.n;

    // Calculate lighting contribution from image based lighting source (IBL)
#ifdef USE_IBL
    iblLighting(
        back_color_info,
        material,
        material_info,
        back_normal_info, v);
    // Interior sky occlusion — see kInteriorSkyAmbient.  Applied HERE,
    // between the IBL and punctual calls, so it scales only the
    // environment term: the sun (and its shadow) is untouched.
    if ((model_params.flip_uv_coord & MODEL_FLAG_INTERIOR) != 0u ||
        ps_in_data.vertex_node_flags > 0.5) {
        back_color_info.f_specular  *= kInteriorSkyAmbient;
        back_color_info.f_diffuse   *= kInteriorSkyAmbient;
        back_color_info.f_clearcoat *= kInteriorSkyAmbient;
        back_color_info.f_sheen     *= kInteriorSkyAmbient;
    }
#endif // USE_IBL

	// Calculate lighting contribution from punctual light sources
#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i) {
        punctualLighting(
            back_color_info,
            ps_in_data,
            material,
            material_info,
            runtime_lights.lights[i],
            back_normal_info,
            v,
            shadow);
    }
#endif // !USE_PUNCTUAL
#endif

    // LIGHTING
    PbrLightsColorInfo color_info = initColorInfo();

    // Calculate lighting contribution from image based lighting source (IBL)
#ifdef USE_IBL
    iblLighting(
        color_info,
        material,
        material_info,
        normal_info, v);
    // Interior sky occlusion — see kInteriorSkyAmbient.
    if ((model_params.flip_uv_coord & MODEL_FLAG_INTERIOR) != 0u ||
        ps_in_data.vertex_node_flags > 0.5) {
        color_info.f_specular  *= kInteriorSkyAmbient;
        color_info.f_diffuse   *= kInteriorSkyAmbient;
        color_info.f_clearcoat *= kInteriorSkyAmbient;
        color_info.f_sheen     *= kInteriorSkyAmbient;
    }
#endif // USE_IBL

	// Calculate lighting contribution from punctual light sources
#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i) {
        punctualLighting(
            color_info,
            ps_in_data,
            material,
            material_info,
            runtime_lights.lights[i],
            normal_info,
            v,
            shadow);
    }
#endif // !USE_PUNCTUAL

#ifdef DOUBLE_SIDED
    float translucent_ratio = 0.2f;
    color_info.f_diffuse += back_color_info.f_diffuse * translucent_ratio;
    color_info.f_specular += back_color_info.f_specular * translucent_ratio * 0.1f;
#endif

    layerBlending(
        color_info,
        ps_in_data,
        material,
        material_info,
        normal_info,
        v);

    vec3 color =
        getFinalColor(
            color_info,
            ps_in_data,
            material,
            material_info,
            v,
            1.0f);


// A decal is alpha-BLENDED, not alpha-tested: its whole job is to be
// partially transparent at the margins, so the cutout branch (which
// discards low alpha and then forces a to 1.0) would destroy exactly
// the signal we need.  Skip it on that permutation only.
#if defined(ALPHAMODE_MASK) && !defined(DECAL)
    // Late discard to avoid samplig artifacts. See https://github.com/KhronosGroup/glTF-Sample-Viewer/issues/267
    if(baseColor.a < material.alpha_cutoff)
    {
        discard;
    }
    baseColor.a = 1.0;
#endif // ALPHAMODE_MASK

#ifdef DECAL
    // Ground-contact + distance fade.  Same call the deferred decal
    // branch above makes, so the two permutations dissolve identically.
    float decal_alpha =
        decalCoverage(v, normal_info.ng, ps_in_data.vertex_position);

    if ((material.material_features & FEATURE_MATERIAL_ALPHA_MASK) != 0u) {
        // Foliage cards: screen-door cut, survivors fully opaque (under
        // the decal pipeline's blend state alpha 1.0 is a full
        // overwrite, so overlapping cards stop compositing into soup).
        if (decalScreenDoorCull(baseColor.a, material.alpha_cutoff,
                                decal_alpha)) {
            discard;
        }
        outColor = vec4(toneMap(material, color), 1.0);
    } else {
        // Road skirt & true decals: alpha-blended as before.
        // baseColor.a still carries the material's own alpha here (the
        // road-fade skirt's texture ramp), so the two multiply rather
        // than one overriding the other.
        outColor = vec4(toneMap(material, color),
                        baseColor.a * decal_alpha);
    }
#else
    // regular shading
    outColor = vec4(toneMap(material, color), baseColor.a);
#endif // DECAL

    // ── Runtime render-debug override ────────────────────────────────────────
    // Driven by the "Render Debug" menu (packed into camera_info.input_features
    // bits 16..23 by application.cpp).  Mode 0 = the shaded path above, all
    // other modes overwrite outColor with a single intermediate channel so we
    // can visually inspect what each part of the pipeline is contributing.
    uint dbg_mode =
        (camera_info.input_features & FEATURE_INPUT_DEBUG_MODE_MASK)
            >> FEATURE_INPUT_DEBUG_MODE_SHIFT;
    if (dbg_mode == DEBUG_RENDER_MODE_ALBEDO) {
        outColor = vec4(baseColor.rgb, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_NORMAL) {
        outColor = vec4(normal_info.n * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_GEOMETRIC_NORMAL) {
        outColor = vec4(normal_info.ng * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_DIFFUSE) {
        outColor = vec4(color_info.f_diffuse, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SPECULAR) {
        outColor = vec4(color_info.f_specular, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SHADOW) {
        outColor = vec4(vec3(shadow), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_ROUGHNESS) {
        outColor = vec4(vec3(material_info.perceptualRoughness), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_METALLIC) {
        outColor = vec4(vec3(material_info.metallic), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_TRANSLUCENT) {
        // Tint by AlphaMode so it's instantly clear which materials are
        // tagged translucent (glass / windows), alpha-tested, or opaque.
        // Magenta = blend / glass, yellow = mask, dark grey = opaque.
        if ((material.material_features & FEATURE_MATERIAL_BLEND) != 0u) {
            outColor = vec4(1.0, 0.2, 1.0, 1.0);
        } else if ((material.material_features & FEATURE_MATERIAL_ALPHA_MASK) != 0u) {
            outColor = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            outColor = vec4(0.1, 0.1, 0.1, 1.0);
        }
    } else if (dbg_mode == DEBUG_RENDER_MODE_SSAO) {
        // White → ssao_apply.comp multiplies by ao → vec3(ao) on screen.
        // See cluster_bindless.frag's matching branch for the rationale.
        outColor = vec4(1.0, 1.0, 1.0, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_WEIGHT_SUM) {
        // Per-vertex skin weight sum (interpolated across the triangle).
        //   sum 0 → red, 1 → white, 2 → blue (clamped past 2).
        // A correctly skinned vertex sums to ~1 and reads white; red flags
        // unweighted/under-weighted verts, blue flags over-weighted ones.
        float ws = ps_in_data.vertex_weight_sum;
        if (ws < 0.0) {
            // Non-skinned geometry carries no weights — paint dark grey.
            outColor = vec4(0.05, 0.05, 0.05, 1.0);
        } else if (ws < 1.0) {
            outColor = vec4(mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0),
                                clamp(ws, 0.0, 1.0)), 1.0);
        } else {
            outColor = vec4(mix(vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 1.0),
                                clamp(ws - 1.0, 0.0, 1.0)), 1.0);
        }
    }
#else
    outColor = baseColor;
#endif // NO_MTL

    // ── Debug "force red" override ─────────────────────────────────
    // Last thing in the shader so it wins over every shaded / debug
    // branch above.  Drives "is this drawable actually rendering?"
    // smoke tests — the application sets debug_force_red=1 on a
    // specific DrawableObject (currently the PlayerController player)
    // via setDebugForceRed(true); every other drawable keeps the
    // field at 0 and is unaffected.
    if (model_params.debug_force_red == 1u) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (model_params.debug_force_red == 2u) {
        // Editor selection highlight: blend the lit colour toward a bright
        // amber so the picked object/sub-object reads as a highlight layer
        // rendered on top of the original mesh.  Set per-node by drawNodes()
        // when the drawable's m_highlight_node_ matches (or == -2 = whole).
        outColor.rgb = mix(outColor.rgb, vec3(1.0, 0.55, 0.08), 0.5);
    }
}