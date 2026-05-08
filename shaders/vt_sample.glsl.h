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

const uint VT_PAGE_SIZE          = 128u;
const uint VT_POOL_WIDTH         = 4096u;
const uint VT_POOL_HEIGHT        = 4096u;
const uint VT_PAGES_ACROSS       = VT_POOL_WIDTH  / VT_PAGE_SIZE;   // 32
const uint VT_PAGES_DOWN         = VT_POOL_HEIGHT / VT_PAGE_SIZE;   // 32
const uint VT_PAGES_PER_LAYER    = VT_PAGES_ACROSS * VT_PAGES_DOWN; // 1024

const float VT_INV_POOL_WIDTH    = 1.0 / float(VT_POOL_WIDTH);
const float VT_INV_POOL_HEIGHT   = 1.0 / float(VT_POOL_HEIGHT);
const float VT_PAGE_TO_POOL_X    = float(VT_PAGE_SIZE) * VT_INV_POOL_WIDTH;
const float VT_PAGE_TO_POOL_Y    = float(VT_PAGE_SIZE) * VT_INV_POOL_HEIGHT;

// Layer enum — must match C++ VtLayer.
const uint VT_LAYER_ALBEDO         = 0u;
const uint VT_LAYER_NORMAL         = 1u;
const uint VT_LAYER_METAL_ROUGH_AO = 2u;
const uint VT_LAYER_EMISSIVE       = 3u;
const uint VT_LAYER_COUNT          = 4u;

// VirtualTextureId encoding — must match makeVtId/vtLayer/vtIndex
// in virtual_texture.h.
//   bits 30..31  layer (2 bits)
//   bits  0..29  vt_index (30 bits)
const uint VT_INVALID_ID = 0xFFFFFFFFu;
uint vtLayerOf(uint vt_id) { return (vt_id >> 30) & 0x3u; }
uint vtIndexOf(uint vt_id) { return vt_id & 0x3FFFFFFFu; }

// VirtualTextureMeta — 32 B, must match C++ layout exactly.
struct VirtualTextureMeta {
    uint width_px;
    uint height_px;
    uint pages_x;
    uint pages_y;
    uint page_table_offset;
    uint mip_count;
    uint pad0;
    uint pad1;
};

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
//
// The actual layout(set=..., binding=...) lines live in vt_bindings.glsl.h
// (or wherever the host wires them in) so this file can be included
// from many different pipelines without re-declaring bindings.

// ── Page-table entry decode ────────────────────────────────────────
// Matches packPageEntry() in virtual_texture.h:
//   bits  0..7   phys_x (page coord, 0..255)
//   bits  8..15  phys_y (page coord, 0..255)
//   bits 16      resident (1 = mapped)
struct VtPageEntry {
    uint phys_x;
    uint phys_y;
    bool resident;
};
VtPageEntry vtUnpackEntry(uint packed) {
    VtPageEntry e;
    e.phys_x   =  packed        & 0xFFu;
    e.phys_y   = (packed >>  8) & 0xFFu;
    e.resident = ((packed >> 16) & 0x1u) != 0u;
    return e;
}

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
        // Default: neutral grey so missing pages are visible without
        // breaking lighting math.  v2 streams the page in.
        return vec4(0.5, 0.5, 0.5, 1.0);
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
