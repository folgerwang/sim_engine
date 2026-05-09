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
