#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
// ── VISIBILITY PASS ──────────────────────────────────────────────────
// Rasterises exactly the same clusters as cluster_bindless.frag's
// GBUFFER_OUTPUT permutation, over the same vertex shader and the same
// indirect draws, but writes only WHICH TRIANGLE won the pixel:
//
//     R32G32_UINT = (cluster_idx + 1, gl_PrimitiveID)
//
// and lets visbuffer_material.comp evaluate the material ONCE per
// surviving pixel.
//
// Why this exists (Nsight GPU Trace, 151 ms frame): the G-buffer cluster
// pass discards, so it gets no early-Z, and a pixel inside a crown runs
// the full material chain -- VT page-table read, page walk, four pool
// layers, the leaf-age cohort mask -- for EVERY one of the 10-30 leaf
// fragments stacked on it.  L1TEX saturated, SM throughput a hairline.
// Here a hidden fragment costs one alpha fetch and a 8-byte write.
//
// The alpha test has to stay in this pass: with a cutout material you
// cannot know which card owns the pixel without evaluating its alpha.
// It is one fetch instead of eight, and once leaf materials go opaque
// (design/NANITE_VISIBILITY_BUFFER.md phase 2) it disappears entirely
// and the hardware runs early-Z natively.
#include "global_definition.glsl.h"
#include "cluster_bindless_bindings.glsl.h"
#include "vt_sample.glsl.h"
#include "leaf_age.glsl.h"

// Declared HERE, before cluster_cutout.glsl.h, not after: that header
// reads camera_info.global_leaf_age, and glslang resolves identifiers
// at parse time, so a later declaration is not visible to it.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer ViewCameraInfoBuffer {
    ViewCameraInfo camera_info;
};

#include "cluster_cutout.glsl.h"
#include "visbuffer_common.glsl.h"

// Interface kept identical to cluster_bindless.vert's outputs so the
// SAME vertex shader feeds both passes -- a second vertex permutation
// that dropped the unused varyings would save a little bandwidth and is
// a later optimisation, not a correctness matter.
layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) flat in uint v_cluster_idx;
layout(location = 4) in vec4 v_tangent;
layout(location = 5) in vec4 v_cur_clip;
layout(location = 6) in vec4 v_prev_clip;

layout(location = 0) out uvec2 out_vis;

void main() {
    vt_lod_bias_g = camera_info.vt_lod_bias;   // Render Debug > Leaf cutout mip lerp
    uint mat_idx = draw_infos[v_cluster_idx].material_idx;
    cluster_tree_age_g = clusterTreeAgeOf(v_cluster_idx);
    int  flags   = material_params[mat_idx].flags;

    // Translucent geometry does not belong in a visibility buffer: one
    // pixel would have to remember several triangles.  Glass keeps the
    // forward/OIT path, exactly as the G-buffer permutation leaves it
    // to the translucent pipeline.
    if ((flags & BINDLESS_MAT_TRANSLUCENT) != 0) discard;

    // The cutout, from the one shared implementation the material pass
    // also calls -- see cluster_cutout.glsl.h.  clusterCutoutAlpha
    // reaches the SAME value clusterCutoutAlbedo puts in .a, but off
    // the dedicated BC4 alpha layer, so this pass never touches the
    // albedo pool.  That is the point: this runs per fragment, and a
    // crown pixel has 10-30 of them.
    //
    // Hardware mip selection here (have_grad = false): this pass has
    // real derivatives; the compute pass does not.
    float alpha = clusterCutoutAlpha(mat_idx, flags, v_uv,
                                     vec2(0.0), vec2(0.0), false);
    if (clusterCutoutRejects(mat_idx, flags, alpha)) discard;

    out_vis = vbPack(v_cluster_idx, uint(gl_PrimitiveID));
}
