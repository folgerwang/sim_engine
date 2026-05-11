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
//   sampler2D vt_pool_albedo;        // VtLayer::ALBEDO
//   sampler2D vt_pool_normal;        // VtLayer::NORMAL
//   sampler2D vt_pool_mr_ao;         // VtLayer::METAL_ROUGH_AO
//   sampler2D vt_pool_emissive;      // VtLayer::EMISSIVE
//   readonly  buffer VtPageTableBuffer { uint vt_page_table[]; };
//   readonly  buffer VtMetaBuffer      { VirtualTextureMeta vt_meta[]; };
//   buffer    VtFeedbackBuffer         { uint vt_feedback[]; };
//   layout(push_constant) uniform     { uvec2 vt_feedback_dim; ... };
//
// The feedback buffer is laid out row-major in (screen_w/8) × (screen_h/8)
// uints.  Cleared to VT_TILE_KEY_INVALID at frame start; one key
// written per 8×8 screen block by the cluster_bindless.frag block-
// representative fragment.  The CPU streamer reads the buffer at
// frame end, dedupes the keys, and uploads the matching pages.

// ── LOD selection ─────────────────────────────────────────────────
// Compute the continuous LOD value for the sample at `uv`.  Standard
// Vulkan formula: ρ² = max(|dUV/dx|², |dUV/dy|²) in source-texel
// units, lod = 0.5 * log₂(ρ²).  Returns a non-negative float — the
// caller splits it into an integer VT-mip (which page-table window
// to read) and a fractional part (the lerp weight between pool mip
// 0 and pool mip 1, which gives smooth transition near integer LOD
// boundaries — see the Stage 1.5 "+1 pool mip" change).
//
// Note: dFdx/dFdy must be called from a fragment shader and the
// derivatives must be uniform across the quad for the result to be
// meaningful.  All current callers (cluster_bindless.frag's albedo /
// normal sample sites) satisfy this.
float vtComputeLod(in VirtualTextureMeta meta, vec2 uv) {
    vec2 src_uv = uv * vec2(float(meta.width_px), float(meta.height_px));
    vec2 dx     = dFdx(src_uv);
    vec2 dy     = dFdy(src_uv);
    float rho2  = max(dot(dx, dx), dot(dy, dy));
    // 0.5 * log2(rho2) = log2(rho).  Floor at 0 for sub-texel
    // derivatives (we'd just sample mip 0 anyway).
    return 0.5 * log2(max(rho2, 1.0));
}

// Convenience: continuous LOD clamped to the VT's mip range and
// rounded down to the integer VT-mip whose page should be looked
// up.  The FRAC component for the in-pool trilinear lerp is the
// caller's responsibility — typically `clamp(lod - float(mip), 0,
// 1)` with frac forced to 0 at the highest mip (no further lerp).
uint vtPickMip(in VirtualTextureMeta meta, vec2 uv) {
    float lod    = vtComputeLod(meta, uv);
    uint  mip_max = max(1u, meta.mip_count) - 1u;
    return clamp(uint(lod), 0u, mip_max);
}

// ── Core: virtual UV → physical pool UV ────────────────────────────
// Translate a (vt_id, uv, mip) sample request into a physical pool
// UV.  Returns false if the page is unresident — caller should
// either fall back to a higher mip (Stage 5 walk) or to a default.
//
// `mip` is computed by vtPickMip() at the call site; passing it in
// rather than recomputing inside lets the caller pick once and try
// the same mip for albedo + normal of the same surface, OR walk to
// higher mips on miss without redoing the LOD math.
//
// The streaming feedback write (Stage 3) will also live in this
// function — every resolve, hit or miss, atomicOr's a bit in the
// feedback bitmap so the streamer knows what's been requested.
bool vtResolve(
    uint vt_id,
    vec2 uv,
    in VirtualTextureMeta meta,
    uint mip,
    out vec2 phys_uv) {

    // Per-VT page coords AT THE CHOSEN MIP.  Mip k's grid is
    // (pages_x>>k, pages_y>>k), each clamped to ≥ 1.  Wrap neg/>1
    // UVs via fract — most material UVs are tiled.
    uvec2 mip_pages = vtMipPagesXY(meta, mip);
    vec2 wrapped_uv = fract(uv);
    vec2 page_uv    = wrapped_uv * vec2(float(mip_pages.x), float(mip_pages.y));
    uvec2 page_xy   = uvec2(floor(page_uv));
    page_xy.x       = min(page_xy.x, mip_pages.x - 1u);
    page_xy.y       = min(page_xy.y, mip_pages.y - 1u);

    // Per-page intra-UV — fractional part within the chosen page.
    //
    // No edge inset is needed any more: each pool slot now stores a
    // VT_TILE_SIZE × VT_TILE_SIZE block (= VT_PAGE_SIZE content +
    // VT_TILE_BORDER on every side), where the border holds real
    // neighbor pixels from the source texture.  Bilinear samples at
    // in_page_uv == 0.0 / 1.0 land 0..0.5 texels inside the border
    // and pick up correct neighbor data instead of bleeding into a
    // different VT's content (which was the problem when slots were
    // tightly packed without guttering).
    vec2 in_page_uv = fract(page_uv);

    // Page table lookup at the chosen mip's window.
    uint mip_offset = vtMipPageTableOffset(meta, mip);
    uint entry_idx  = mip_offset + page_xy.y * mip_pages.x + page_xy.x;
    uint packed    = vt_page_table[entry_idx];
    VtPageEntry e  = vtUnpackEntry(packed);
    if (!e.resident) {
        phys_uv = vec2(0.0);
        return false;
    }

    // Physical pool UV.  The slot starts at (phys_x * SLOT, phys_y *
    // SLOT) and includes a VT_TILE_BORDER-pixel border on the top
    // and left; the page's content sits inside at +BORDER.  Sampling
    // [0,1] of the page maps to [BORDER..BORDER+PAGE]/POOL_W of the
    // slot, so bilinear samples just outside [0,1] still hit border
    // data and avoid edge clamps.
    vec2 slot_origin_uv = vec2(
        float(e.phys_x) * VT_SLOT_TO_POOL_X + VT_BORDER_TO_POOL_X,
        float(e.phys_y) * VT_SLOT_TO_POOL_Y + VT_BORDER_TO_POOL_Y);
    phys_uv = slot_origin_uv + in_page_uv * vec2(VT_PAGE_TO_POOL_X, VT_PAGE_TO_POOL_Y);
    return true;
}

// ── Per-layer convenience samplers ─────────────────────────────────
// Each takes a VirtualTextureId + UV and returns the sampled value
// (or a sensible default for unresident pages).

// NOTE: the convenience samplers below are kept for completeness
// but the cluster_bindless.frag main path now does the resolve
// inline so it can pick mip ONCE, reuse it across albedo + normal,
// fall back to higher mips on miss (Stage 5), and route to legacy
// bindless when the VT result is missing.  These wrappers pick mip
// internally — adequate for ad-hoc shaders that want a one-call
// VT sample.  All four use textureLod with the fractional LOD so
// the GPU's pool-mip-0/1 trilinear lerp engages.

// Helper: split continuous LOD into (int VT-mip, in-pool frac).
// Frac is forced to 0 at the deepest VT mip (no further lerp).
void vtPickMipAndFrac(in VirtualTextureMeta meta, vec2 uv,
                      out uint mip, out float frac) {
    float lod_cont = vtComputeLod(meta, uv);
    uint  mip_max  = max(1u, meta.mip_count) - 1u;
    mip            = clamp(uint(lod_cont), 0u, mip_max);
    frac           = (mip == mip_max)
                     ? 0.0
                     : clamp(lod_cont - float(mip), 0.0, 1.0);
}

// One-shot resolve: pick mip + frac AND do the page-table lookup in
// a single call, returning the physical pool UV + LOD frac that all
// four layer pool samplers can reuse.  Since the four pools share a
// slot allocator (the same vt_id resolves to the same physical slot
// in all four pools), there's no need to repeat the page-table read
// per layer — one resolve, four textureLods.  Saves three SSBO
// reads + three vtComputeLod calls per fragment compared to calling
// vtSampleAlbedo + vtSampleNormal + vtSampleMetalRoughAO +
// vtSampleEmissive separately.
//
// Caller pattern:
//   vec2 phys_uv; float frac;
//   if (vtResolveSharedSlot(vt_id, uv, phys_uv, frac)) {
//       vec4 albedo = textureLod(vt_pool_albedo, phys_uv, frac);
//       vec3 normal = textureLod(vt_pool_normal, phys_uv, frac).rgb * 2 - 1;
//       vec4 mr_ao  = textureLod(vt_pool_mr_ao,  phys_uv, frac);
//       vec4 emiss  = textureLod(vt_pool_emissive, phys_uv, frac);
//   } else { /* fallback */ }
//
// vt_id can be any of the four layer ids — they share the same
// vt_index, so vtIndexOf(any of them) returns the same meta entry.
bool vtResolveSharedSlot(
    uint vt_id, vec2 uv,
    out vec2 phys_uv, out float frac) {
    if (vt_id == VT_INVALID_ID) {
        phys_uv = vec2(0.0); frac = 0.0;
        return false;
    }
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    uint mip;
    vtPickMipAndFrac(meta, uv, mip, frac);
    return vtResolve(vt_id, uv, meta, mip, phys_uv);
}

vec4 vtSampleAlbedo(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec4(1.0);
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    uint mip; float frac;
    vtPickMipAndFrac(meta, uv, mip, frac);
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, mip, phys_uv)) {
        return vec4(1.0, 0.0, 1.0, 1.0);  // magenta diagnostic
    }
    return textureLod(vt_pool_albedo, phys_uv, frac);
}

vec3 vtSampleNormal(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec3(0.0, 0.0, 1.0);
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    uint mip; float frac;
    vtPickMipAndFrac(meta, uv, mip, frac);
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, mip, phys_uv)) {
        return vec3(0.0, 0.0, 1.0);
    }
    vec2 nxy = textureLod(vt_pool_normal, phys_uv, frac).rg * 2.0 - 1.0;
    float nz = sqrt(max(0.0, 1.0 - dot(nxy, nxy)));
    return vec3(nxy, nz);
}

vec4 vtSampleMetalRoughAO(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec4(1.0, 1.0, 0.0, 1.0);
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    uint mip; float frac;
    vtPickMipAndFrac(meta, uv, mip, frac);
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, mip, phys_uv)) {
        return vec4(1.0, 1.0, 0.0, 1.0);
    }
    return textureLod(vt_pool_mr_ao, phys_uv, frac);
}

vec4 vtSampleEmissive(uint vt_id, vec2 uv) {
    if (vt_id == VT_INVALID_ID) return vec4(0.0);
    VirtualTextureMeta meta = vt_meta[vtIndexOf(vt_id)];
    uint mip; float frac;
    vtPickMipAndFrac(meta, uv, mip, frac);
    vec2 phys_uv;
    if (!vtResolve(vt_id, uv, meta, mip, phys_uv)) {
        return vec4(0.0);
    }
    return textureLod(vt_pool_emissive, phys_uv, frac);
}

#endif  // VT_SAMPLE_GLSL_H
