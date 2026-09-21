#ifndef CLUSTER_BINDLESS_BINDINGS_GLSL_H
#define CLUSTER_BINDLESS_BINDINGS_GLSL_H
// ── The cluster bindless descriptor set, declared ONCE ───────────────
// Set PBR_MATERIAL_PARAMS_SET, bindings 0..10: cluster draw infos,
// material params, the legacy bindless texture arrays, and the whole
// Runtime Virtual Texture side (four pools + page table + meta +
// feedback).  Extracted from cluster_bindless.frag so the visibility
// pass and the deferred material pass bind the SAME set by the same
// names -- three copies of a descriptor layout is the drift that ends
// in a wrong texture on a wave.  vt_sample.glsl.h documents that it
// references these by name, so including this header is what satisfies
// it.
//
// C++ side: ClusterRenderer builds this set (bindless_desc_set_); any
// new pipeline that includes this header must bind that same set at
// PBR_MATERIAL_PARAMS_SET.

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

#endif // CLUSTER_BINDLESS_BINDINGS_GLSL_H
