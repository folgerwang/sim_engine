#ifndef VT_SAMPLE_GLSL_H
#define VT_SAMPLE_GLSL_H

// ─── Runtime Virtual Texture sampling shim ────────────────────────
// Drop-in replacement for `texture(sampler2D_var, uv)` at material
// sample sites.  Resolves a VirtualTextureId through the page table
// to a physical pool UV, samples the appropriate layer pool, and
// (in v2) writes a feedback request for the page actually used.
//
// Mirrors virtual_texture.h on the C++ side — these constants and
// the VirtualTextureMeta layout MUST match.  Static asserts on the
// C++ side guard against drift.
//
// ── Include order requirement ──────────────────────────────────────
// The helpers below reference SSBO and sampler bindings by name, so
// the including shader MUST declare them BEFORE this file is included.
// The required include order is:
//
//   #include "vt_types.glsl.h"    // VirtualTextureMeta + constants
//   layout(set=..., binding=..) uniform sampler2D vt_pool_albedo;
//   layout(set=..., binding=..) uniform sampler2D vt_pool_normal;
//   layout(set=..., binding=..) uniform sampler2D vt_pool_mr_ao;
//   layout(set=..., binding=..) uniform sampler2D vt_pool_emissive;
//   layout(...) readonly buffer { uint vt_page_table[]; };
//   layout(...) readonly buffer { VirtualTextureMeta vt_meta[]; };
//   #include "vt_sample.glsl.h"   // helpers below
//
// (vt_types.glsl.h is split out so the SSBO declaration can use
// VirtualTextureMeta without forward-declaring the struct here.)

#include "vt_types.glsl.h"

// ── Required descriptor bindings — provided by the host ─────────────
//
// The including shader is responsible for declaring these bindings at
// the right (set, binding) slot for its pipeline.  By convention we
// use a single dedicated set for VT bindings to avoid colliding with
// existing material descriptors — the host wires it in initVulkan
// once and every material pipeline binds the same set.
//
// Expected resources:
//   sampler2D vt_pool_albedo;    // VtLayer::ALBEDO
//   sampler2D vt_pool_normal;    // VtLayer::NORMAL
//   sampler2D vt_pool_mr_ao;     // VtLayer::METAL_ROUGH_AO
//   sampler2D vt_pool_emissive;  // VtLayer::EMISSIVE
//   readonly buffer VtPageTableBuffer { uint vt_page_table[]; };
//   readonly buffer VtMetaBuffer      { VirtualTextureMeta vt_meta[]; };

// ── Core: virtual UV → physical pool UV ────────────────────────────
// Translate a (vt_id, uv) sample request into a physical pool UV.
// Returns false if the page is unresident — caller should fall back
// to a default value (typically a neutral debug color).
//
// In v2 this function will also append the requested page ID to the
// feedback ring buffer; for v1 the streaming branch is empty.
bool vtResolve(
    uint vt_id,
    vec2 uv,
    in VirtualTextureMeta meta,
    out vec2 phys_uv) {

    // Per-VT page coords.  uv is in [0, 1]; multiply by pages_{x,y}
    // to get fractional page coords, floor for the page index.  Wrap
    // negative or >1 UVs via fract — most material UVs are tiled.
    vec2 wrapped_uv = fract(uv);
    vec2 page_uv    = wrapped_uv * vec2(float(meta.pages_x), float(meta.pages_y));
    uvec2 page_xy   = uvec2(floor(page_uv));
    page_xy.x       = min(page_xy.x, meta.pages_x - 1u);
    page_xy.y       = min(page_xy.y, meta.pages_y - 1u);

    // Per-page intra-UV — fractional part within the chosen page.
    vec2 in_page_uv = fract(page_uv);

    // Page table lookup.
    uint entry_idx = meta.page_table_offset + page_xy.y * meta.pages_x + page_xy.x;
    uint packed    = vt_page_table[entry_idx];
    VtPageEntry e  = vtUnpackEntry(packed);
    if (!e.resident) {
        phys_uv = vec2(0.0);
        return false;
    }

    // Physical pool UV = page origin + intra-page offset, all in
    // pool's [0, 1] UV space.
    vec2 page_origin_uv = vec2(
        float(e.phys_x) * VT_PAGE_TO_POOL_X,
        float(e.phys_y) * VT_PAGE_TO_POOL_Y);
    phys_uv = page_origin_uv + in_page_uv * vec2(VT_PAGE_TO_POOL_X, VT_PAGE_TO_POOL_Y);
    return true;
}

// ── Per-layer convenience samplers ─────────────────────────────────
// Each takes a VirtualTextureId + UV and returns the sampled value
// (or a sensible default for unresident pages).

vec4 vtSampleAlbedo(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec4(1.0);
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, phys_uv)) {
        // ── Diagnostic: unresident page fallback ───────────────────
        // Bright magenta so any surface that fell through to the
        // "page is missing" branch is visually obvious.  Restore to
        // neutral grey (0.5,0.5,0.5) once residency is debugged.
        return vec4(1.0, 0.0, 1.0, 1.0);
    }
    return texture(vt_pool_albedo, phys_uv);
}

vec3 vtSampleNormal(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec3(0.0, 0.0, 1.0);  // tangent-space up
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, phys_uv)) {
        return vec3(0.0, 0.0, 1.0);
    }
    // RG channels store the tangent-space x/y; reconstruct z under
    // the assumption of a unit normal.  Pool stores [0, 1] UNORM, so
    // remap to [-1, 1] before the sqrt.
    vec2 nxy = texture(vt_pool_normal, phys_uv).rg * 2.0 - 1.0;
    float nz = sqrt(max(0.0, 1.0 - dot(nxy, nxy)));
    return vec3(nxy, nz);
}

// metallic, roughness, ao packed into the .b, .g, .r channels (glTF
// convention for the metallic-roughness texture, augmented with AO
// in .r as the engine convention).  Returns RGBA so callers can pick
// the channels they need without an extra texture call.
vec4 vtSampleMetalRoughAO(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec4(1.0, 1.0, 0.0, 1.0);  // ao=1, rough=1, metal=0
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, phys_uv)) {
        return vec4(1.0, 1.0, 0.0, 1.0);
    }
    return texture(vt_pool_mr_ao, phys_uv);
}

vec4 vtSampleEmissive(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec4(0.0);  // no emission by default
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, phys_uv)) {
        return vec4(0.0);
    }
    return texture(vt_pool_emissive, phys_uv);
}

#endif  // VT_SAMPLE_GLSL_H
