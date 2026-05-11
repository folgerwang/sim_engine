#ifndef VT_TYPES_GLSL_H
#define VT_TYPES_GLSL_H

// ─── Runtime Virtual Texture — types & constants ──────────────────
//
// Split out of vt_sample.glsl.h so that fragment shaders can declare
// SSBO blocks of `VirtualTextureMeta vt_meta[]` BEFORE pulling in the
// helper functions (which reference those bindings by name).  GLSL
// requires every identifier inside a function body to be in scope at
// parse time, so the helpers in vt_sample.glsl.h cannot be included
// until after the bindings are declared — but the bindings need this
// struct.  Two-file split breaks the cycle:
//
//   #include "vt_types.glsl.h"      // VirtualTextureMeta + constants
//   layout(...) buffer { VirtualTextureMeta vt_meta[]; };  // bindings
//   #include "vt_sample.glsl.h"     // helpers using the above
//
// Mirrors virtual_texture.h on the C++ side — these constants and the
// VirtualTextureMeta layout MUST match.  Static asserts on the C++
// side guard against drift.

// VT_PAGE_SIZE is the LOGICAL page size — the addressable content
// area each tile represents in the source texture.  VT_TILE_BORDER
// is the per-side guttering: each physical pool slot is
// VT_TILE_SIZE = VT_PAGE_SIZE + 2 * VT_TILE_BORDER pixels wide, with
// the inner VT_PAGE_SIZE×VT_PAGE_SIZE block holding the page's
// content and the surrounding border holding pixels from the
// neighboring tiles in the source texture.  Bilinear sampling at
// page UV [0,1] therefore lands inside the border at edges and
// reads real neighbor data instead of clamping to the slot edge or
// bleeding into a different VT's content.
const uint VT_PAGE_SIZE          = 64u;
const uint VT_TILE_BORDER        = 4u;
const uint VT_TILE_SIZE          = VT_PAGE_SIZE + 2u * VT_TILE_BORDER;  // 72
// Pool grid: 114 × 57 slots → 8208 × 4104 px image (~physical 8K×4K).
// Must match kVtPagesAcross / kVtPagesDown / kVtPoolWidth / kVtPoolHeight
// in virtual_texture.h.
const uint VT_PAGES_ACROSS       = 114u;
const uint VT_PAGES_DOWN         = 57u;
const uint VT_PAGES_PER_LAYER    = VT_PAGES_ACROSS * VT_PAGES_DOWN;     // 6498
const uint VT_POOL_WIDTH         = VT_PAGES_ACROSS * VT_TILE_SIZE;      // 8208
const uint VT_POOL_HEIGHT        = VT_PAGES_DOWN   * VT_TILE_SIZE;      // 4104

// Hard cap on mip levels per virtual texture.  At 64-px page size,
// 8 mips covers source dimensions up to 64 << 7 = 8192 px, which is
// past every texture in the engine's content pipeline.  The cap also
// bounds the loop in vtMipPageTableOffset / fallback walks, keeping
// the shader's instruction count predictable.
const uint VT_MAX_MIPS           = 8u;

// ── Streaming feedback ────────────────────────────────────────────
// Each fragment that wants to sample a VT page can encode the tile
// it touched as a 32-bit key and hand it to the CPU streamer via the
// vt_feedback SSBO.  To avoid every fragment writing (4M atomic writes
// per frame at 1080p), the cluster fragment shader writes ONE key per
// VT_FEEDBACK_BLOCK_SIZE × VT_FEEDBACK_BLOCK_SIZE screen block — the
// block-corner fragment is the representative.  At 1920×1080 with
// block size 8 that's 32400 keys per frame ≈ 130 KB feedback buffer.
//
// Tile key layout (32 bits):
//   bits  0..13   vt_index   (14 bits → 16384 VTs)
//   bits 14..17   mip        (4 bits  → 16 levels)
//   bits 18..24   page_x     (7 bits  → 128 pages)
//   bits 25..31   page_y     (7 bits  → 128 pages)
const uint VT_FEEDBACK_BLOCK_SIZE = 8u;
const uint VT_TILE_KEY_INVALID    = 0xFFFFFFFFu;
// Feedback buffer is laid out as a fixed-stride 2D grid of tile keys
// (one key per 8×8 screen block).  The pitch is sized to cover up to
// a 4K-wide screen (4096 / 8 = 512 keys per row); for smaller
// screens the tail of each row stays at VT_TILE_KEY_INVALID and the
// CPU dedupe pass skips them cheaply.  Total buffer size:
// 512² × 4 = 1 MB, cleared via vkCmdFillBuffer at frame start.
const uint VT_FEEDBACK_PITCH      = 512u;
const uint VT_FEEDBACK_TOTAL      = VT_FEEDBACK_PITCH * VT_FEEDBACK_PITCH;

uint vtMakeTileKey(uint vt_idx, uint mip, uvec2 page) {
    return ( vt_idx     & 0x3FFFu)
         | ((mip        & 0x000Fu) << 14)
         | ((page.x     & 0x007Fu) << 18)
         | ((page.y     & 0x007Fu) << 25);
}

// Decode helpers — useful for shader-side debug viz.  CPU side has a
// matching set in virtual_texture.h.
uint vtTileKeyVtIndex(uint key) { return  key         & 0x3FFFu; }
uint vtTileKeyMip    (uint key) { return (key >> 14)  & 0x000Fu; }
uvec2 vtTileKeyPage  (uint key) {
    return uvec2((key >> 18) & 0x007Fu, (key >> 25) & 0x007Fu);
}

const float VT_INV_POOL_WIDTH    = 1.0 / float(VT_POOL_WIDTH);
const float VT_INV_POOL_HEIGHT   = 1.0 / float(VT_POOL_HEIGHT);
// Two scale factors now: SLOT-to-pool is the physical 72-px stride,
// PAGE-to-pool is the logical 64-px content stride.  vtResolve uses
// SLOT for the slot origin and PAGE for the in-page UV mapping —
// shaper bilinear samples at edges land in the border instead of
// crossing into another slot's content.
const float VT_SLOT_TO_POOL_X    = float(VT_TILE_SIZE) * VT_INV_POOL_WIDTH;
const float VT_SLOT_TO_POOL_Y    = float(VT_TILE_SIZE) * VT_INV_POOL_HEIGHT;
const float VT_PAGE_TO_POOL_X    = float(VT_PAGE_SIZE) * VT_INV_POOL_WIDTH;
const float VT_PAGE_TO_POOL_Y    = float(VT_PAGE_SIZE) * VT_INV_POOL_HEIGHT;
const float VT_BORDER_TO_POOL_X  = float(VT_TILE_BORDER) * VT_INV_POOL_WIDTH;
const float VT_BORDER_TO_POOL_Y  = float(VT_TILE_BORDER) * VT_INV_POOL_HEIGHT;

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
//
// Page-table layout for a registered VT:
//   [page_table_offset .. + pages_x*pages_y)                 ← mip 0
//   [.. + (pages_x>>1)*(pages_y>>1))                          ← mip 1
//   ...
//   [.. + 1)                                                  ← mip (mip_count-1)
// So the offset to a given mip's window is
//   mip_off(k) = page_table_offset
//              + sum_{j<k} (max(1, pages_x>>j) * max(1, pages_y>>j))
// computed by the helper vtMipPageTableOffset() below.
//
// pages_x / pages_y are mip-0 dimensions in pages.  Higher mips use
// `max(1u, pages_x >> mip)` etc — never 0.  When a mip's page count
// would round to less than 1 it stays at 1 and the source is held
// in a corner of the 64×64 pool slot (in_page_uv stays in the valid
// sub-rect via the half-texel clamp; sub-page mips only happen when
// the source is smaller than a page, in which case mip_count == 1
// anyway).
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

// Compute the page-table offset of a specific mip's window inside
// the VT's contiguous page-table region.  Cheap: at most VT_MAX_MIPS
// = 8 iterations, each one mul + ceil-half + add.  Inlines well.
//
// CEIL-halve (not floor) is required so non-POT mip-0 counts still
// give every mip a grid that fully covers the downsampled source —
// see virtual_texture.h::vtMipPagesAt for the worked example.
uint vtMipPageTableOffset(in VirtualTextureMeta meta, uint mip) {
    uint offset = meta.page_table_offset;
    uint px     = meta.pages_x;
    uint py     = meta.pages_y;
    for (uint k = 0u; k < mip; ++k) {
        offset += px * py;
        px = max(1u, (px + 1u) >> 1);
        py = max(1u, (py + 1u) >> 1);
    }
    return offset;
}

// Page grid dimensions at a given mip level.  Both clamped to ≥ 1
// so the divide-by-zero / wrap-on-shift issue can't crop up when
// the source is smaller than 2^mip pages.  Ceil-halve, see above.
//
// Bit form: ceil(p / 2^k) = (p + (1<<k) - 1) >> k, min 1.
uvec2 vtMipPagesXY(in VirtualTextureMeta meta, uint mip) {
    uint denom_minus_1 = (1u << mip) - 1u;
    uint px = (meta.pages_x + denom_minus_1) >> mip;
    uint py = (meta.pages_y + denom_minus_1) >> mip;
    return uvec2(max(1u, px), max(1u, py));
}

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

#endif  // VT_TYPES_GLSL_H
