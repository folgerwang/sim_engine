#version 450
#extension GL_ARB_separate_shader_objects : enable
// Required for bindless texture array with non-uniform (per-cluster) indices.
// Without this, the GPU assumes tex_idx is uniform across the subgroup and
// picks one texture for the whole wave, producing wrong textures at material
// boundaries between adjacent clusters.
#extension GL_EXT_nonuniform_qualifier : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// ─── cluster_bindless.frag ──────────────────────────────────────────
// Bindless cluster fragment shader.
//
// Lighting matches base.frag structure:
//   • Directional sun from RuntimeLightsParams (set RUNTIME_LIGHTS_PARAMS_SET)
//   • Diffuse: Lambertian ÷π (simplified, no Fresnel energy conservation)
//   • Single CSM shadow sample (cheapest cascade, no PCF)
//   • Diffuse IBL ambient via lambertian_env_sampler (matches base.frag USE_IBL path)
//   • linearTosRGB tone mapping — matches base.frag default path
// ─────────────────────────────────────────────────────────────────────

// set 0 — IBL + shadow samplers (ibl.glsl.h declares all of set 0, including
// direct_shadow_sampler at DIRECT_SHADOW_INDEX — do not redeclare it below).
#include "ibl.glsl.h"

// set 1 — camera
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

// set 2 — cluster SSBOs + texture arrays
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = 0)
    readonly buffer DrawInfoBuffer {
    ClusterDrawInfo draw_infos[];
};
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = 1)
    readonly buffer MaterialParamsBuffer {
    BindlessMaterialParams material_params[];
};
layout(set = PBR_MATERIAL_PARAMS_SET, binding = 2)
    uniform sampler2D base_color_textures[MAX_CLUSTER_TEXTURES];
layout(set = PBR_MATERIAL_PARAMS_SET, binding = 3)
    uniform sampler2D normal_textures[MAX_CLUSTER_TEXTURES];

// ── Runtime Virtual Texture (RVT) bindings ──────────────────────────
// The four pool textures + page table + meta SSBO live on the same
// descriptor set as the rest of the cluster bindless data (set 2).
// vt_sample.glsl.h's helpers (`vtSampleAlbedo`, `vtSampleNormal`, …)
// reference the resource names declared below — the names must match
// exactly or the included file will fail to compile.
//
// When a material's *_vt_id is VT_INVALID_ID (== 0xFFFFFFFF), the
// shader falls back to the legacy bindless texture arrays declared
// above; otherwise it routes through `vtResolve` and samples from the
// pool texture matching the layer encoded in the upper bits of the
// id.  See virtual_texture.h for the encoding.
//
// Order of declarations matters here:
//   1. vt_types.glsl.h     — declares VirtualTextureMeta + constants.
//   2. SSBO/sampler bindings (use the struct from step 1).
//   3. vt_sample.glsl.h    — defines helpers that reference the
//                            bindings from step 2.
// GLSL needs every identifier in scope at parse time, so we cannot
// pull in the helpers until both the struct AND the bindings exist.
#include "vt_types.glsl.h"

layout(set = PBR_MATERIAL_PARAMS_SET, binding = 4)
    uniform sampler2D vt_pool_albedo;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = 5)
    uniform sampler2D vt_pool_normal;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = 6)
    uniform sampler2D vt_pool_mr_ao;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = 7)
    uniform sampler2D vt_pool_emissive;
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = 8)
    readonly buffer VtPageTableBuffer {
    uint vt_page_table[];
};
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = 9)
    readonly buffer VtMetaBuffer {
    VirtualTextureMeta vt_meta[];
};
// VT streaming feedback — one tile-key uint per 8×8 screen block.
// The cluster fragment shader writes its desired (vt, mip, page) key
// from the (0,0) fragment of each 8×8 block; the CPU streamer
// (VirtualTextureManager::tick) reads, dedupes, and acts on requests
// at frame end.  Buffer is laid out row-major
// (screen_w / VT_FEEDBACK_BLOCK_SIZE) × (screen_h / VT_FEEDBACK_BLOCK_SIZE)
// — the row stride lives in the push-constant `vt_feedback_pitch`
// below so the shader doesn't have to query the swapchain size.
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = 10)
    buffer VtFeedbackBuffer {
    uint vt_feedback[];
};

// Now that the resources above are in scope, pull in the VT sampling
// helpers.  The header declares `vtSampleAlbedo / vtSampleNormal /
// vtSampleMetalRoughAO / vtSampleEmissive` plus the `vtResolve`
// primitive — all of which reference the bindings above by name.
#include "vt_sample.glsl.h"

// set 4 — runtime lights
layout(set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX)
    uniform RuntimeLightsUniformBufferObject {
    RuntimeLightsParams runtime_lights;
};

// set 5 — ambient probe SSBO (AmbientProbeSystem grid).  Gated by the
// USE_AMBIENT_PROBES define so the shader still compiles when the probe
// pipeline isn't wired into the cluster layout yet.  When enabled, the
// fragment shader replaces the static IBL Lambertian ambient with a
// position-dependent SH evaluation from the 8 nearest probes.
//
// To enable end-to-end:
//   1. Add `AmbientProbeSystem::getProbeDescSetLayout()` to the layouts
//      passed to the cluster bindless pipeline (alongside the existing
//      sets above).
//   2. Bind ambient_probe_system_->getProbeDescSet() during draw().
//   3. Build cluster_bindless_frag.spv with `-DUSE_AMBIENT_PROBES`.
#ifdef USE_AMBIENT_PROBES
#define AMBIENT_PROBE_SET 5
#define AMBIENT_PROBE_BINDING 0
#include "ambient_probes.glsl.h"

// Grid bounds — the application could plumb these via push_constants
// or a UBO; for now the shader header's defaults are used (matching
// the kDefaultGrid* constants in AmbientProbeSystem) and the grid
// extents are passed as push constants alongside other per-frame state.
// If you don't have a UBO yet, call sampleAmbientProbeGrid with literal
// vec3 mins/maxes that match what placeProbeGrid was called with.
#endif

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) flat in uint v_cluster_idx;
// Interpolated world-space tangent + bitangent sign (see BindlessVertex doc
// in cluster_renderer.h).  Replaces the per-fragment dFdx/dFdy tangent
// reconstruction that produced sparkle on shaded surfaces.
layout(location = 4) in vec4 v_tangent;
// Current + previous-frame homogeneous clip positions, populated by
// cluster_bindless.vert from camera_info.view_proj and
// camera_info.prev_view_proj.  Only consumed by the GBUFFER_OUTPUT
// branch (velocity write); other variants ignore them but the vertex
// shader still computes them — the cost is two extra mat4*vec4 per
// vertex which is negligible relative to the rest of the cluster pass.
layout(location = 5) in vec4 v_cur_clip;
layout(location = 6) in vec4 v_prev_clip;

// FOUR output variants compiled from this single fragment source:
//   default              — single-RT lit colour for the legacy forward path
//                          (opaque + alpha-mask).  This is also where the
//                          per-frame debug-render overrides live.
//   OIT_OUTPUT           — McGuire-Bavoil WBOIT pair (accum + reveal) for
//                          translucent clusters; resolved by
//                          oit_composite.frag.  CURRENTLY UNUSED — the
//                          engine moved glass to the simpler alpha-blend
//                          variant below — but the variant is kept around
//                          for future opt-in (e.g., dense translucent
//                          scenes where order-independence matters).
//   ALPHA_BLEND_OUTPUT   — single-RT translucent for forward alpha
//                          blending.  Runs the same glass IBL math as
//                          OIT_OUTPUT but emits (rgb, α) to location 0
//                          for the pipeline's hardware src_alpha /
//                          one_minus_src_alpha blend.  No accum/reveal,
//                          no fullscreen composite — just porter-duff
//                          "over" in hardware.
//   GBUFFER_OUTPUT       — 3-RT G-buffer for the deferred path:
//                            RT0 RGBA8  albedo.rgb + ao.a
//                            RT1 RGBA8  octahedral normal.xy + roughness.z
//                                       + flags.w
//                            RT2 RGBA8  emissive.rgb + metallic.a
//                            RT3 RG16F  screen-space velocity (NDC delta)
//                          deferred_resolve.comp consumes these to run
//                          lighting once per pixel in compute, replacing
//                          the in-shader PBR math below.
//
// ⚠️ MAINTAINER NOTE — translucent-vs-opaque routing guard:
//   The early-frame guard at the top of main() discards fragments whose
//   BINDLESS_MAT_TRANSLUCENT flag doesn't match the variant's intent
//   (translucent variants discard non-translucent; opaque variants
//   discard translucent).  If you add a NEW variant, make sure the
//   guard's `#if defined(...)` correctly classifies it.  Forgetting
//   this caused the "alpha-blend glass completely missing" regression:
//   ALPHA_BLEND_OUTPUT initially fell into the opaque branch and
//   discarded every glass fragment.  Search for "MAINTAINER NOTE" in
//   the guard block for the matching commentary.
#ifdef OIT_OUTPUT
    layout(location = 0) out vec4 out_accum;   // RGBA16F: rgb = colour×alpha×w, a = alpha×w
    layout(location = 1) out float out_reveal; // R8/R16F: accumulates Π(1 − αᵢ)
#elif defined(GBUFFER_OUTPUT)
    layout(location = 0) out vec4 out_albedo_ao;
    layout(location = 1) out vec4 out_normal_rough;
    layout(location = 2) out vec4 out_emissive_metal;
    // RT3 — RG16F screen-space NDC-delta velocity.  Standard convention
    // for TAA: stores curNDC - prevNDC so reprojection is just
    // sample(prev, uv - velocity * 0.5) (NDC ∈ [-1,1] → UV ∈ [0,1]
    // halves the magnitude).  Sky / unwritten pixels remain at the
    // CLEAR value (0,0) so a downstream pass can treat zero velocity as
    // "no history" and fall back to current colour.
    layout(location = 3) out vec2 out_velocity;
#else
    // Shared by both `default` (opaque) and ALPHA_BLEND_OUTPUT — both
    // emit to a single colour attachment at location 0.  The pipeline
    // (opaque vs alpha-blend) decides whether hardware blending is
    // enabled and what the blend factors are.
    layout(location = 0) out vec4 out_color;
#endif

// Octahedral encode — paired with octDecode in deferred_resolve.comp.
// Maps a unit-length direction onto the 2D unit square with no
// singularities and ~7-8 bits of angular error per channel after RGBA8
// quantisation, which is well within the threshold for the diffuse +
// low-spec lighting we perform on the resolve side.
vec2 octEncodeDir(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy
                            : (1.0 - abs(n.yx)) * vec2(
                                  n.x >= 0.0 ? 1.0 : -1.0,
                                  n.y >= 0.0 ? 1.0 : -1.0);
    return oct * 0.5 + 0.5;
}

// ── Cheap single-cascade shadow ──────────────────────────────────────
// PCSS soft shadow with cascade-consistent WORLD-SPACE blur radius.
// All sizes are world units; per-cascade conversion to shadow-map UV
// happens inside shadowFactor().  Constants kept identical to
// deferred_resolve.comp and base.frag.
const float CSM_NORMAL_BIAS_SCALE     = 0.05;
// Depth bias in WORLD units.  Converted to per-cascade clip-Z inside
// shadowFactor() — see deferred_resolve.comp for full notes.  Without
// this conversion, coarser cascades silently get 5–10× more bias than
// fine cascades, producing a visible intensity step at boundaries.
const float CSM_DEPTH_BIAS_BASE_WORLD  = 0.05;
const float CSM_DEPTH_BIAS_SLOPE_WORLD = 0.20;
const float CSM_LIGHT_SIZE_WORLD      = 0.20;
const float CSM_BLOCKER_RADIUS_WORLD  = 0.40;
const float CSM_MIN_PCF_RADIUS_WORLD  = 0.02;
const float CSM_MAX_PCF_RADIUS_WORLD  = 0.80;
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

float shadowFactor(vec3 world_pos, vec3 world_normal, vec2 screen_pixel) {
    float view_depth = -(camera_info.view * vec4(world_pos, 1.0)).z;
    int cascade = CSM_CASCADE_COUNT - 1;
    for (int i = 0; i < CSM_CASCADE_COUNT; ++i) {
        // cascade_splits is packed vec4[2]; index as [i/4][i%4].
        if (view_depth < runtime_lights.cascade_splits[i >> 2][i & 3]) {
            cascade = i;
            break;
        }
    }

    vec3  N     = normalize(world_normal);
    vec3  L     = normalize(-runtime_lights.lights[0].direction);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    vec3  biased_pos = world_pos + N * ((1.0 - NdotL) * CSM_NORMAL_BIAS_SCALE);

    vec4 lclip = runtime_lights.light_view_proj[cascade] * vec4(biased_pos, 1.0);
    vec3 lndc  = lclip.xyz / lclip.w;
    vec2 uv    = lndc.xy * 0.5 + 0.5;
    float ref  = lndc.z;

    // Per-cascade world→UV (XY) and world→clip-Z (depth) scale factors.
    // See deferred_resolve.comp for the derivation.  Both are needed to
    // make BLUR RADIUS and DEPTH BIAS cascade-consistent in physical
    // (world) units.
    float w2uv    = 0.5 *
        length(runtime_lights.light_view_proj[cascade][0].xyz);
    float z_scale = length(runtime_lights.light_view_proj[cascade][2].xyz);

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
        float d = texture(direct_shadow_sampler,
                          vec3(uv + off, float(cascade))).r;
        if (d < ref - depth_bias) {
            blocker_sum += d;
            ++blocker_count;
        }
    }
    if (blocker_count == 0) return 1.0;
    float avg_blocker_depth = blocker_sum / float(blocker_count);

    // PCSS step 2: penumbra estimate.
    float penumbra = (ref - avg_blocker_depth) /
                     max(avg_blocker_depth, 1e-4);
    float pcf_radius = clamp(penumbra * light_size_uv,
                             min_pcf_radius_uv,
                             max_pcf_radius_uv);

    // PCSS step 3: PCF at the computed radius.
    float sum = 0.0;
    for (int i = 0; i < CSM_PCF_SAMPLES; ++i) {
        vec2 off = csmVogelDisk(i, CSM_PCF_SAMPLES, phi) * pcf_radius;
        float map = texture(direct_shadow_sampler,
                            vec3(uv + off, float(cascade))).r;
        sum += (map < 1.0 && ref > map + depth_bias) ? 0.0 : 1.0;
    }
    return sum * (1.0 / float(CSM_PCF_SAMPLES));
}

void main() {
    // Base colour — fetch first so we can discard early.
    uint mat_idx    = draw_infos[v_cluster_idx].material_idx;
    vec4 base_color = material_params[mat_idx].base_color_factor;
    int  tex_idx    = material_params[mat_idx].base_color_tex_idx;
    int  mat_flags  = material_params[mat_idx].flags;

#if defined(OIT_OUTPUT) || defined(ALPHA_BLEND_OUTPUT)
    // Translucent pipelines (WBOIT or forward alpha-blend): only
    // AlphaMode::Blend materials should ever reach this draw — the cull
    // shader routes by the BINDLESS_MAT_TRANSLUCENT flag into a separate
    // indirect bucket.  Defense-in-depth discard if something stale
    // sneaks through, so it can't darken accum/reveal (OIT path) or
    // leak into the colour buffer (alpha-blend path).
    //
    // NOTE: this guard MUST include ALPHA_BLEND_OUTPUT.  Without it,
    // the alpha-blend variant would fall into the `#else` opaque guard
    // below and discard EVERY translucent fragment — which is exactly
    // what happened when this variant was first added and "glass went
    // completely missing".
    if ((mat_flags & BINDLESS_MAT_TRANSLUCENT) == 0) {
        discard;
    }
#else
    // The opaque pipeline must NEVER draw translucent clusters — those go
    // through the translucent (OIT or alpha-blend) path.  Symmetric guard:
    // if the cull shader miswrites a translucent cluster into the opaque
    // indirect bucket, drop it here.
    if ((mat_flags & BINDLESS_MAT_TRANSLUCENT) != 0) {
        discard;
    }
#endif
    vec4 albedo4    = base_color;
    // ── Albedo sample: prefer VT, magenta-flag pool overflows ──────
    // The material may carry both an `albedo_vt_id` (VT registration)
    // and a `base_color_tex_idx` (legacy bindless slot).  Routing
    // priority:
    //
    //   1. has VT registration AND page resident  → VT (BC7 pool).
    //   2. has VT registration AND page UNRESIDENT
    //      → MAGENTA BLOCK (diagnostic — see below).  We deliberately
    //        do NOT fall back to legacy in this branch.
    //   3. no VT registration (albedo_vt == INVALID) → legacy bindless.
    //   4. no VT registration AND no legacy slot → leave as base_color.
    //
    // Why magenta instead of a silent legacy fallback for case 2:
    // the v1 pool is eager-resident with a fixed 4096-slot capacity,
    // and large scenes (Bistro etc.) overflow it almost immediately —
    // a single 2048² albedo eats 1024 slots, so 4 such textures fill
    // the pool and the remaining ~200 fall through to UNRESIDENT.
    // If we silently fell back to legacy in that case the scene
    // looked fine but VT would be nearly idle — there was no visual
    // signal to motivate raising the pool size, dropping tiny
    // textures, or implementing streaming.  Showing the overflow
    // textures as solid magenta makes the cost of being over-budget
    // immediately obvious, while case 1 still demonstrates the VT
    // path is structurally working for materials that do fit.
    //
    // Pool textures are BC7_SRGB_BLOCK so the VT-sampled value is
    // auto-de-gammafied on read.  Legacy texture views go through
    // hardware sRGB and are likewise linear on sample, so cases 1
    // and 3 feed albedo4 in the same colour space.
    uint albedo_vt       = material_params[mat_idx].albedo_vt_id;
    bool has_vt_alb      = (albedo_vt != VT_INVALID_ID);
    vec4 albedo_tex      = vec4(1.0);
    bool vt_alb_handled  = false;     // true once cases 1 or 2 handle the sample
    // Cache the picked VT mip + fractional LOD outside the if-block
    // so the normal-map path below can reuse them — both samples
    // share the same UV and the same source-resolution-driven LOD,
    // so they should always pick the same level.  This keeps albedo
    // and normal in visual lock-step (no shimmer where albedo picks
    // mip k and normal picks mip k+1).
    uint  vt_alb_mip  = 0u;
    float vt_alb_frac = 0.0;
    // Cache the resolved physical pool UV from the albedo path so the
    // normal-map (and any future MR_AO/EMISSIVE) sample below can
    // reuse it.  All four layer pools share a slot allocator, so the
    // same vt_index/mip/page resolves to identical phys_uv across
    // every layer — the redundant page-table SSBO read in each layer
    // sample was costing several percent of fragment-shader time.
    vec2  vt_alb_phys_uv  = vec2(0.0);
    bool  vt_alb_resolved = false;
    uint  vt_alb_walk_mip = 0u;
    if (has_vt_alb) {
        VirtualTextureMeta meta = vt_meta[vtIndexOf(albedo_vt)];
        // Continuous LOD → integer VT-mip + fractional in-pool LOD.
        // The frac drives the GPU's pool-mip-0/mip-1 trilinear lerp,
        // smoothly hiding the snap as LOD crosses an integer
        // boundary.  At the deepest VT mip there's no further mip to
        // lerp into, so frac is forced to 0 there.
        float lod_cont = vtComputeLod(meta, v_uv);
        uint  mip_max  = max(1u, meta.mip_count) - 1u;
        vt_alb_mip     = clamp(uint(lod_cont), 0u, mip_max);
        vt_alb_frac    = (vt_alb_mip == mip_max)
                         ? 0.0
                         : clamp(lod_cont - float(vt_alb_mip), 0.0, 1.0);

        // ── Streaming feedback ──────────────────────────────────────
        // Emit one tile-key per 8×8 screen block, from the block's
        // (0, 0)-corner fragment.  All four layer pools share the
        // same slot layout (registerMaterial uses a single shared
        // allocator), so one tile request covers albedo + normal +
        // mr_ao + emissive of the same material/page/mip.  CPU
        // streamer reads the buffer at frame end and decides which
        // pages to upload before the next frame.
        ivec2 pix = ivec2(gl_FragCoord.xy);
        if ((pix.x & 7) == 0 && (pix.y & 7) == 0) {
            uvec2 mip_pages = vtMipPagesXY(meta, vt_alb_mip);
            vec2  wrapped   = fract(v_uv);
            uvec2 page      = uvec2(floor(wrapped * vec2(mip_pages)));
            page.x = min(page.x, mip_pages.x - 1u);
            page.y = min(page.y, mip_pages.y - 1u);

            ivec2 block  = pix >> 3;
            uint  fb_idx = uint(block.y) * VT_FEEDBACK_PITCH + uint(block.x);
            vt_feedback[fb_idx] = vtMakeTileKey(
                vtIndexOf(albedo_vt), vt_alb_mip, page);
        }

        // Walk up the mip chain until we find a resident page.  The
        // streamer pins the smallest mip per VT, so this loop is
        // guaranteed to terminate before mip_count and we always
        // sample SOMETHING (worst case = a heavily blurred version).
        vec2 phys_uv;
        bool resolved = false;
        uint walk_mip = vt_alb_mip;
        for (uint i = 0u; i < VT_MAX_MIPS; ++i) {
            if (walk_mip >= meta.mip_count) break;
            if (vtResolve(albedo_vt, v_uv, meta, walk_mip, phys_uv)) {
                resolved = true;
                break;
            }
            ++walk_mip;
        }
        // Stash for the normal-map path below — all four layer pools
        // share the same slot, so the same phys_uv works for any of
        // them.  See vt_alb_phys_uv comment above.
        vt_alb_phys_uv  = phys_uv;
        vt_alb_resolved = resolved;
        vt_alb_walk_mip = walk_mip;
        if (resolved) {
            // If we walked up past the requested mip the in-pool
            // mip-1 lerp no longer corresponds to the desired LOD
            // — force frac to 0 so we just sample pool mip 0 of the
            // fallback slot rather than blurring twice.
            float frac = (walk_mip == vt_alb_mip) ? vt_alb_frac : 0.0;
            albedo_tex     = textureLod(vt_pool_albedo, phys_uv, frac);
            vt_alb_handled = true;
        } else {
            // No resident page anywhere up the chain (streamer hasn't
            // even pinned the smallest mip — should be rare/transient).
            // Magenta marker keeps it visually obvious.
            albedo_tex     = vec4(1.0, 0.0, 1.0, 1.0);
            vt_alb_handled = true;
        }
    }
    if (!vt_alb_handled && tex_idx >= 0) {
        // Case 3: no VT registration — legacy bindless path.
        // nonuniformEXT: tex_idx varies per cluster — the GPU must
        // not assume it is uniform across the subgroup (wave).
        // Without this, one texture is picked for the entire wave
        // and adjacent clusters with different materials get wrong
        // textures.
        albedo_tex = texture(base_color_textures[nonuniformEXT(tex_idx)], v_uv);
    }
    // Case 4 falls through with albedo_tex = vec4(1.0) → albedo4 = base_color.
    albedo4 *= albedo_tex;

    // Alpha mask discard — matches base.frag: if(baseColor.a < alpha_cutoff) discard.
    if ((mat_flags & BINDLESS_MAT_ALPHA_MASK) != 0) {
        if (albedo4.a < material_params[mat_idx].alpha_cutoff) discard;
    }

    vec3 albedo = albedo4.rgb;

    // Geometry normal — flip for back faces BEFORE building TBN so the cotangent
    // frame is oriented correctly for double-sided materials.
    // Backface culling is disabled for the cluster pipeline.
    vec3 N = normalize(v_normal);
    if ((mat_flags & BINDLESS_MAT_DOUBLE_SIDED) != 0 && !gl_FrontFacing) {
        N = -N;
    }
    // Stash the un-perturbed (geometric) normal before the normal map below
    // overwrites N.  Used by the GEOMETRIC_NORMAL debug visualisation; if we
    // sampled `N` after the TBN multiplication it would be identical to the
    // NORMAL mode, which is exactly the "they look the same" bug.
    vec3 N_geom = N;

    // Normal map — sample and rotate into world space via the interpolated
    // vertex tangent frame.  Tangents are precomputed at upload time
    // (computeMeshTangents() in cluster_renderer.cpp) and stored on the
    // merged vertex stream as vec4(T_world, bitangent_sign), which removes
    // the per-fragment dFdx/dFdy reconstruction that previously produced
    // fine-grained sparkle on shaded surfaces (visible in render-debug
    // mode 2 / 3, and in the final shaded image as specular speckle).
    int  norm_idx     = material_params[mat_idx].normal_tex_idx;
    uint normal_vt    = material_params[mat_idx].normal_vt_id;
    bool has_vt_norm  = (normal_vt != VT_INVALID_ID);
    if (has_vt_norm || norm_idx >= 0) {
        // Sample and decode the normal map.
        // Bistro (and most DCC tools) export DirectX-convention normal maps where
        // the green channel is inverted relative to OpenGL/GLSL tangent space.
        // base.frag applies n.y = -n.y for the same reason — we must match it.
        // Z is reconstructed from XY (more robust than reading the stored Z which
        // can be degraded by DXT5nm / BC5 compression).
        //
        // VT path: route through the page table (same UV mapping as the
        // legacy path) but inline the resolve so we keep the existing
        // DX-convention Y-flip and z-reconstruction intact.  Falling back
        // to the legacy bindless texture array when the page is
        // unresident keeps the surface lit until streaming catches up.
        vec4 raw_n;
        bool vt_resolved = false;
        if (has_vt_norm) {
            // Fast path: albedo already resolved this surface's
            // (vt_index, mip, page) → physical pool UV.  All four
            // layer pools share a slot allocator so that same
            // phys_uv directly addresses the normal pool's matching
            // slot — no second page-table SSBO read, no second LOD
            // calc, no second mip-walk.  This is the per-fragment
            // win the user asked for ("page-table ssbo read only one
            // for all 4 layers").
            if (has_vt_alb && vt_alb_resolved) {
                float frac = (vt_alb_walk_mip == vt_alb_mip)
                             ? vt_alb_frac : 0.0;
                raw_n       = textureLod(vt_pool_normal, vt_alb_phys_uv, frac);
                vt_resolved = true;
            } else {
                // Slow path: albedo isn't VT-registered (or didn't
                // resolve), so do the full pick + resolve from the
                // normal's own meta entry.  Rare in practice.
                VirtualTextureMeta meta = vt_meta[vtIndexOf(normal_vt)];
                float lod_cont = vtComputeLod(meta, v_uv);
                uint  mip_max  = max(1u, meta.mip_count) - 1u;
                uint  nrm_mip  = clamp(uint(lod_cont), 0u, mip_max);
                float nrm_frac = (nrm_mip == mip_max)
                                 ? 0.0
                                 : clamp(lod_cont - float(nrm_mip), 0.0, 1.0);
                vec2 phys_uv;
                if (vtResolve(normal_vt, v_uv, meta, nrm_mip, phys_uv)) {
                    raw_n       = textureLod(vt_pool_normal, phys_uv, nrm_frac);
                    vt_resolved = true;
                }
            }
        }
        if (!vt_resolved) {
            // Legacy bindless path — used when (a) material isn't VT-
            // registered for normal, or (b) the requested page isn't
            // resident yet (v1 is eager-resident, but the fallback keeps
            // v2 streaming graceful).
            if (norm_idx >= 0) {
                raw_n = texture(normal_textures[nonuniformEXT(norm_idx)], v_uv);
            } else {
                // No legacy slot AND no resident VT page → fall through
                // with raw_n that decodes to a flat tangent-space normal.
                raw_n = vec4(0.5, 0.5, 1.0, 1.0);
            }
        }
        vec2 nxy   = raw_n.xy * 2.0 - 1.0;
        nxy.y      = -nxy.y;   // DX → GL convention flip
        vec3 ts_n  = vec3(nxy, sqrt(max(1.0 - dot(nxy, nxy), 0.0)));

        // Reconstruct an orthonormal TBN from the interpolated tangent.
        // The vertex tangent was orthogonalised against the OBJECT-space
        // normal at upload, but interpolation across the triangle and the
        // potentially-non-rigid model transform can leave a small component
        // along N — re-orthogonalise per fragment to keep TBN orthonormal.
        vec3 T = normalize(v_tangent.xyz - dot(v_tangent.xyz, N) * N);
        vec3 B = cross(N, T) * v_tangent.w;

        N = normalize(mat3(T, B, N) * ts_n);
    }

#ifdef GBUFFER_OUTPUT
    // Deferred path: stash material attributes into the 3-RT G-buffer and
    // bail out before any lighting math runs.  deferred_resolve.comp will
    // consume these targets to do PBR once per visible pixel.
    //
    // Slot 0: albedo.rgb + ao.a     — ao is 1.0 placeholder; an SSAO pass
    //                                  may later route into RT0.a.
    // Slot 1: octEncode(N).xy +
    //         roughness.z + flags.w — roughness is a 0.5 default until we
    //                                  plumb metal-rough textures through
    //                                  BindlessMaterialParams; flags.w
    //                                  reserved for shading-model bits.
    // Slot 2: emissive.rgb + metallic.a — both default 0 for the same
    //                                     reason; cluster materials don't
    //                                     yet author either channel.
    out_albedo_ao      = vec4(albedo, 1.0);
    out_normal_rough   = vec4(octEncodeDir(N), 0.5, 0.0);
    // Slot 2 was emissive.rgb + metallic.a, but cluster materials don't
    // author either yet — both default to 0 in the deferred resolve.
    // Repurpose .rg for the octahedral-encoded GEOMETRIC normal so the
    // resolve pass can distinguish NORMAL (perturbed by normal map)
    // from GEOMETRIC_NORMAL in the debug visualisation.  Keep .b for
    // future emissive intensity, .a for metallic.  A future material
    // upgrade that wants emissive RGB will need a 4th GBuffer
    // attachment; for now this is the cheapest place to land it.
    vec2 oct_geom = octEncodeDir(N_geom);
    out_emissive_metal = vec4(oct_geom.x, oct_geom.y, 0.0, 0.0);

    // Screen-space NDC velocity = curNDC.xy - prevNDC.xy.  Both clip
    // positions came from per-vertex matrix multiplies in the vertex
    // shader; perspective-divide here so each fragment's velocity is
    // its own current-vs-previous projected location.  When the
    // application has just initialised, prev_view_proj is mirrored
    // from view_proj (see ViewCamera::updateViewCameraInfo) so the
    // first frame writes 0 — correct "no history yet".
    vec2 cur_ndc  = v_cur_clip.xy  / v_cur_clip.w;
    vec2 prev_ndc = v_prev_clip.xy / v_prev_clip.w;
    out_velocity  = cur_ndc - prev_ndc;
    return;
#endif

    vec3 V = normalize(camera_info.position - v_world_pos);

    // Sun light direction (away from surface → sun).
    vec3 L = normalize(-runtime_lights.lights[0].direction);
    // Full physical intensity — tone mapping below handles highlight compression.
    vec3 light_col = runtime_lights.lights[0].color *
                     runtime_lights.lights[0].intensity;

    float NdotL = max(dot(N, L), 0.0);

    // Respect FEATURE_INPUT_SHADOW_DISABLED, matching base.frag.
    float shad = 1.0;
    if ((camera_info.input_features & FEATURE_INPUT_SHADOW_DISABLED) == 0u) {
        // Pass the un-perturbed vertex normal (v_normal) for shadow
        // biasing, not N which may already include normal-map detail.
        // gl_FragCoord.xy is the dither key for per-pixel Vogel-disk
        // rotation.
        shad = shadowFactor(v_world_pos, v_normal, gl_FragCoord.xy);
    }

    // Diffuse IBL ambient — samples the pre-convolved lambertian irradiance
    // cubemap with the surface normal, matching base.frag's USE_IBL path.
    // This gives direction-dependent ambient (brighter sky-facing surfaces,
    // darker ground-facing ones) instead of the old flat 3% approximation.
#ifdef USE_AMBIENT_PROBES
    // Probe-driven ambient: trilinear-blend SH evaluation from the 8
    // nearest probes in the grid.  Each probe's coefficients were baked
    // from a moving DynamicCubemap that visited that probe's world
    // position over 6 frames and projected the captured cube into 9
    // RGB SH coefficients.  Cosine convolution is already baked into
    // the stored coefficients (sh_project.comp), so this is a simple
    // dot-product evaluation — no extra cosine factor needed.
    //
    // Grid bounds match what AmbientProbeSystem::placeProbeGrid was
    // called with in application.cpp.  If you change those, update
    // these constants OR plumb them via a UBO.
    const vec3 kProbeGridMin = vec3(-50.0, -10.0, -50.0);
    const vec3 kProbeGridMax = vec3( 50.0,  30.0,  50.0);
    vec3 probe_irradiance =
        sampleAmbientProbeGrid(v_world_pos, N,
                               kProbeGridMin, kProbeGridMax);
    vec3 ambient = albedo * probe_irradiance;
#else
    vec3 ambient = getIBLRadianceLambertian(N, albedo);
#endif

    // Lambertian diffuse (÷π matches PBR BRDF_lambertian normalisation).
    vec3 diffuse = albedo * light_col * (NdotL * shad / M_PI);

    // Blinn-Phong specular — cheap substitute for GGX, no texture lookup needed.
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * shad;
    vec3  specular = light_col * (0.05 * spec / M_PI);

    vec3 color = ambient + diffuse + specular;

#if defined(OIT_OUTPUT) || defined(ALPHA_BLEND_OUTPUT)
    // ── Image-based glass shading ───────────────────────────────────────
    // Activated for BOTH the WBOIT path (OIT_OUTPUT) and the simpler
    // forward alpha-blended path (ALPHA_BLEND_OUTPUT).  The math is
    // identical — the only difference downstream is whether the result
    // accumulates into a (accum, reveal) pair for a composite pass, or
    // gets written straight to the scene colour buffer with hardware
    // src_alpha / one_minus_src_alpha blending.
    // Glass surfaces aren't well represented by the diffuse + Blinn-Phong
    // model used for opaque cluster geometry.  In the OIT (translucent)
    // path we replace `color` with a Fresnel-blended mix of:
    //
    //   • Reflection : IBL GGX specular cube sampled in reflect(-V, N) at
    //                  low roughness (sharp environment reflection).
    //                  Standard getIBLRadianceGGX with F0 = 0.04 gives a
    //                  dielectric reflection that's faint at near-normal
    //                  angles and intensifies at grazing — exactly the
    //                  Fresnel behaviour real glass shows.
    //
    //   • Refraction : IBL specular cube sampled in a *slightly* bent
    //                  view direction.  Pure refract(-V, N, 1/η) at glass
    //                  IOR 1.5 produces a strong sphere-like warp; for a
    //                  thin pane we want the bend to be barely
    //                  perceptible — just enough for the eye to read
    //                  "this surface has thickness, what's behind shifts
    //                  with the surface curvature."  We blend the
    //                  geometric view direction (-V) with the fully-
    //                  refracted direction by glass_thickness ∈ [0, 1]:
    //                  0  = no refraction (looks like a flat decal),
    //                  1  = full IOR=1.5 refraction (sphere-style warp).
    //                  ~0.25 reads as "thick window pane" without making
    //                  the world warp obviously.
    //
    // Schlick Fresnel mixes refraction (near-normal) with reflection
    // (grazing).  base_color tints the refraction so stained-glass windows
    // and tinted bottles shift the colour of what's seen through them.
    // 25 % of the direct-lighting term is layered on top so a sun on
    // glass still leaves a Blinn-Phong specular highlight and the surface
    // isn't pitch-dark under unlit IBL.
    {
        // ── Glass parameters ─────────────────────────────────────────────
        const float glass_ior            = 1.5;    // standard window glass
        const float glass_thickness      = 0.30;   // 0 = no refraction, 1 = sphere-style bend
        const float glass_tint_strength  = 0.20;   // 0 = neutral, 1 = full albedo tint on transmission
        const float glass_F0             = 0.04;   // dielectric Fresnel reflectance at normal incidence
        const float glass_chromatic_aber = 0.015;  // R/B index spread (gives slight prism fringe)
        // LOD to sample the GGX environment cube at for glass.  LOD 0
        // (mip 0, mirror-sharp) is what flat polished glass "should"
        // sample, BUT the runtime IBL "mini" path updates only a
        // SPARSE DITHER of mip-0 texels per frame (~1/32 of them) and
        // EMA-blends new samples toward the GGX-convolved target.  At
        // any one moment, mip 0 is a checkerboard of "freshly updated"
        // and "stale-but-converging" texels, with the dither pattern
        // moving every frame.  On glass — which samples the cube
        // directly with no per-fragment averaging, and on three
        // slightly-different directions per pixel for chromatic
        // aberration — that checkerboard turns into visible coloured
        // flicker / sparkle that shifts as the camera moves or the
        // dither rolls.  Opaque PBR doesn't see it because its
        // textureLod is at a roughness-derived LOD ≥ ~0.5 and the
        // box-filter mipgen above mip 0 averages the noise out.
        //
        // Sampling here at LOD 2 reads a 4×4-averaged downsample of
        // mip 0, which collapses the per-texel jitter while keeping
        // the reflection clearly directional.  Push to LOD 3 if any
        // flicker remains; pull back toward LOD 1 once the IBL mini
        // path is upgraded to dense per-frame updates.
        const float glass_env_lod        = 2.0;

        // ── Reflection — env sample at the mirror direction ────────────
        // Deliberately skipping getIBLRadianceGGX: that helper multiplies
        // by (f0·brdf.r + brdf.g) ≈ 0.04 for clear-glass f0, which
        // collapses the reflection to nearly nothing.  We want the full
        // env to appear in the mirror direction with strength controlled
        // by Fresnel below.
        vec3 reflect_dir     = normalize(reflect(-V, N));
        vec3 reflection_term = textureLod(ggx_env_sampler, reflect_dir, glass_env_lod).rgb;

        // ── Refraction — slightly bent ray to simulate pane thickness ──
        // refract() gives the direction the view ray continues *inside*
        // the glass.  At η = 1.5 a pure refract is a strong sphere-style
        // warp; for a thin pane we want a barely-perceptible bend, so we
        // interpolate from the straight-through view direction (-V) to
        // the fully-refracted direction by `glass_thickness`.
        // Per-channel IORs give a tiny chromatic split — red bends less
        // than blue, producing the subtle prism fringe real glass shows.
        vec3 ref_dir_r = refract(-V, N, 1.0 / (glass_ior - glass_chromatic_aber));
        vec3 ref_dir_g = refract(-V, N, 1.0 / glass_ior);
        vec3 ref_dir_b = refract(-V, N, 1.0 / (glass_ior + glass_chromatic_aber));
        if (dot(ref_dir_r, ref_dir_r) < 1e-6) ref_dir_r = -V;
        if (dot(ref_dir_g, ref_dir_g) < 1e-6) ref_dir_g = -V;
        if (dot(ref_dir_b, ref_dir_b) < 1e-6) ref_dir_b = -V;

        vec3 view_r = normalize(mix(-V, ref_dir_r, glass_thickness));
        vec3 view_g = normalize(mix(-V, ref_dir_g, glass_thickness));
        vec3 view_b = normalize(mix(-V, ref_dir_b, glass_thickness));

        vec3 refraction_term = vec3(
            textureLod(ggx_env_sampler, view_r, glass_env_lod).r,
            textureLod(ggx_env_sampler, view_g, glass_env_lod).g,
            textureLod(ggx_env_sampler, view_b, glass_env_lod).b);

        // Albedo tints what's seen through the glass — a green pane
        // greens the world behind it, clear glass leaves it untouched.
        vec3 glass_tint   = mix(vec3(1.0), albedo, glass_tint_strength);
        refraction_term  *= glass_tint;

        // ── Schlick Fresnel ──────────────────────────────────────────────
        // F0 = 0.04 for dielectric glass.  At normal incidence the surface
        // transmits ~96 %; at grazing it reflects nearly 100 % — giving
        // the classic "looking up at a tall glass tower" effect where the
        // upper panels mirror the sky and the lower panels show what's
        // inside.
        float NdotV   = max(dot(N, V), 0.0);
        float fresnel = glass_F0 + (1.0 - glass_F0) * pow(1.0 - NdotV, 5.0);

        // ── Final composition ────────────────────────────────────────────
        // mix(refraction, reflection, fresnel) is the physically-motivated
        // glass term.  We layer a small fraction of the direct-lighting
        // result so a sun glint on glass still produces a Blinn-Phong
        // highlight on top of the IBL reflection.
        color = mix(refraction_term, reflection_term, fresnel)
              + specular * 0.5;
    }
#endif

    // Gamma-correct to sRGB — matches base.frag's TONEMAP_DEFAULT path (exposure=1.0,
    // linearTosRGB only). FBX materials set tonemap_type=TONEMAP_DEFAULT by default.
    //
    // Output alpha = albedo4.a (texture × factor) rather than just base_color.a
    // so semi-transparent textures contribute correctly to the translucent
    // pass blend equation.  For opaque materials albedo4.a ≈ 1.0, so the
    // translucent pipeline (drawn second) is a no-op for them — the cull
    // shader filters them out anyway.
    vec4 final_color = vec4(linearTosRGB(color), albedo4.a);

#if defined(GBUFFER_OUTPUT)
    // The GBUFFER_OUTPUT branch already wrote out_albedo_ao /
    // out_normal_rough / out_emissive_metal and `return`'d above before
    // any of the lighting / sRGB / debug-override code below ran.  We
    // guard this region anyway so the compiler doesn't try to type-check
    // the out_color / out_accum / out_reveal references against a layout
    // qualifier set that doesn't declare them in this variant.
#elif defined(OIT_OUTPUT)
    // ── Weighted Blended OIT output ─────────────────────────────────────
    // McGuire & Bavoil 2013, "weight 7" formulation.  The weight emphasises
    // near-camera fragments (1 − z * 0.9) and high-alpha fragments (alpha
    // ramp), and the clamp keeps the magnitude inside FP16 range so the
    // RGBA16F accumulator never overflows.  Linear-space colour is fed in
    // (we pass linearTosRGB-encoded; for true OIT we'd skip sRGB encoding,
    // but matching the opaque path here keeps colour spaces consistent).
    //
    // Alpha is clamped into [0.2, 0.85] before the weight calc.  This:
    //   • prevents the upper-bound case (a == 1.0) from setting reveal to
    //     zero — that drives the composite to emit `vec4(rgb, 1.0)` which
    //     fully replaces the scene colour and reads as "all black" if the
    //     lit colour is small.  α = 0.85 also keeps Fresnel reflection
    //     punchy at grazing where glass should look quite opaque.
    //   • prevents the lower-bound case (a ~ 0) from contributing nothing
    //     visible to the composite — assets sometimes ship glass with
    //     base_color_factor.a ≈ 0.05, which after blending leaves only
    //     5 % of the IBL reflection / refraction visible.  A 0.2 floor
    //     keeps glass legible while still letting most of the scene show
    //     through.
    float a = clamp(final_color.a, 0.2, 0.85);
    float z = gl_FragCoord.z;
    float weight = clamp(
        pow(min(1.0, a * 10.0) + 0.01, 3.0) * 1e8 *
        pow(1.0 - z * 0.9, 3.0),
        1e-2, 3e3);

    out_accum  = vec4(final_color.rgb * a, a) * weight;
    out_reveal = a;  // pipeline blend is dst*(1−src) so this accumulates Π(1−αᵢ)

    // Debug-mode override is intentionally skipped in OIT_OUTPUT — the
    // visualisation only makes sense in the single-target opaque path.
    return;
#elif defined(ALPHA_BLEND_OUTPUT)
    // ── Forward alpha-blended glass ────────────────────────────────────
    // Single colour attachment, hardware src_alpha / one_minus_src_alpha
    // blending.  We emit (rgb, α) directly; the pipeline blends it over
    // the scene colour buffer with the standard porter-duff "over"
    // formula:  final = src.rgb * src.a + dst.rgb * (1 - src.a)
    //
    // Floor alpha at 0.15 so any asset that ships glass with
    // base_color_factor.a ≈ 0 still gets a faint Fresnel-tinted
    // visible pane instead of disappearing entirely.  Ceiling at 0.95
    // so even the most opaque glass still lets a sliver of the scene
    // behind show through (real flat glass never fully occludes).
    float blend_a = clamp(final_color.a, 0.15, 0.95);
    out_color = vec4(final_color.rgb, blend_a);
    return;
#else
    out_color = final_color;

    // ── Runtime render-debug override ────────────────────────────────────────
    // Mirrors base.frag.  The cluster bindless path uses Blinn-Phong specular
    // (no IBL specular cube), so the SPECULAR mode here shows that term;
    // ROUGHNESS / METALLIC are not authored per-cluster and are emitted as
    // constants so the visualisation is still well-defined.
    uint dbg_mode =
        (camera_info.input_features & FEATURE_INPUT_DEBUG_MODE_MASK)
            >> FEATURE_INPUT_DEBUG_MODE_SHIFT;
    if (dbg_mode == DEBUG_RENDER_MODE_ALBEDO) {
        out_color = vec4(albedo, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_NORMAL) {
        out_color = vec4(N * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_GEOMETRIC_NORMAL) {
        out_color = vec4(N_geom * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_DIFFUSE) {
        out_color = vec4(ambient + diffuse, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SPECULAR) {
        out_color = vec4(specular, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SHADOW) {
        out_color = vec4(vec3(shad), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_ROUGHNESS) {
        out_color = vec4(vec3(0.5), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_METALLIC) {
        out_color = vec4(vec3(0.0), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_TRANSLUCENT) {
        // Tint by alpha mode (matches base.frag's mapping):
        //   magenta = BINDLESS_MAT_TRANSLUCENT (glass / windows / Blend)
        //   yellow  = BINDLESS_MAT_ALPHA_MASK
        //   dark    = opaque
        // Lets you visually verify that glass-by-name overrides reached
        // the cluster path (since cluster bindless currently still draws
        // these without alpha blending, this debug mode is the only way
        // to see the tagging is correct upstream of the rendering).
        if ((mat_flags & BINDLESS_MAT_TRANSLUCENT) != 0) {
            out_color = vec4(1.0, 0.2, 1.0, 1.0);
        } else if ((mat_flags & BINDLESS_MAT_ALPHA_MASK) != 0) {
            out_color = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            out_color = vec4(0.1, 0.1, 0.1, 1.0);
        }
    } else if (dbg_mode == DEBUG_RENDER_MODE_VELOCITY) {
        // Forward path: there is no velocity G-buffer to sample, but the
        // vertex shader still provides cur/prev clip — compute velocity
        // inline so the visualisation matches the deferred path 1:1
        // (same scale, same encoding) and toggling pipelines doesn't
        // change colours under this debug mode.
        const float kVelocityViewScale = 50.0;
        vec2 cur_ndc  = v_cur_clip.xy  / v_cur_clip.w;
        vec2 prev_ndc = v_prev_clip.xy / v_prev_clip.w;
        vec2 v        = (cur_ndc - prev_ndc) * kVelocityViewScale;
        out_color = vec4(v * 0.5 + 0.5, 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SSAO) {
        // Output pure white so when ssao_apply.comp runs (force-enabled
        // for this mode in application.cpp + SSAO::render) it multiplies
        // white * ao = vec3(ao) and the screen displays the raw AO
        // factor.  Works even when ssao_apply.comp's own debug-branch
        // isn't recompiled, and gives a clear "SSAO is off" indicator
        // (uniform white) if the apply pass somehow doesn't run.
        out_color = vec4(1.0, 1.0, 1.0, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_CATEGORY) {
        // MeshCategory solid-colour overlay.  Category id is packed
        // into mat_flags bits 8..15 by ClusterRenderer::applyMaterial
        // Categories after the LLM classifier runs; if it hasn't run
        // yet (or this material wasn't classified) the field reads
        // zero and we display Unknown=grey.  Colour table must match
        // categoryColor() in collision_debug.frag so the rendered-mesh
        // overlay and the collision-proxy overlay read identically.
        uint cat = (uint(mat_flags) & uint(BINDLESS_MAT_CATEGORY_MASK))
                       >> BINDLESS_MAT_CATEGORY_SHIFT;

        // Per-fragment Floor/Wall disambiguation by TRUE surface
        // orientation.  The per-material category is a single name-based
        // verdict, but Bistro reuses the same concrete material on both
        // the ground and building facades, so any one verdict is wrong
        // on the other.  Geometry resolves it unambiguously: among
        // Floor/Wall, an up- or down-facing fragment (|N.y| high) is a
        // horizontal surface -> Floor; a sideways-facing fragment
        // (|N.y| low) is vertical -> Wall.  N_geom is the un-perturbed
        // geometric world normal (before normal mapping), Y-up, so its
        // y component is the orientation we want.  Scoped to Floor<->
        // Wall ONLY: Door / Glass / Object / Vegetation / Ceiling /
        // Stairs / etc. keep their semantic colour regardless of facing.
        if (cat == 1u || cat == 2u) {
            cat = (abs(N_geom.y) >= 0.5) ? 1u : 2u;
        }

        vec3 cat_color;
        if      (cat == 1u)  cat_color = vec3(0.20, 0.75, 0.30); // Floor
        else if (cat == 2u)  cat_color = vec3(0.80, 0.20, 0.20); // Wall
        else if (cat == 3u)  cat_color = vec3(0.95, 0.75, 0.10); // Door
        else if (cat == 4u)  cat_color = vec3(0.55, 0.25, 0.80); // Object
        else if (cat == 5u)  cat_color = vec3(0.40, 0.85, 0.95); // Glass
        else if (cat == 6u)  cat_color = vec3(0.25, 0.45, 0.85); // Ceiling
        else if (cat == 7u)  cat_color = vec3(0.95, 0.50, 0.10); // Stairs
        else if (cat == 8u)  cat_color = vec3(0.55, 0.65, 0.20); // Vegetation
        else if (cat == 9u)  cat_color = vec3(0.95, 0.20, 0.65); // Elevator
        else if (cat == 10u) cat_color = vec3(0.75, 0.55, 0.35); // Ladder
        else                 cat_color = vec3(0.55, 0.55, 0.55); // Unknown
        out_color = vec4(cat_color, 1.0);
    }
#endif // OIT_OUTPUT
}
