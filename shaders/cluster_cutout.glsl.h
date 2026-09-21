#ifndef CLUSTER_CUTOUT_GLSL_H
#define CLUSTER_CUTOUT_GLSL_H
// ── The cutout decision, in ONE place ────────────────────────────────
// Which triangle owns a pixel is decided by the alpha test, and with a
// visibility buffer that decision is made in one pass (cluster_visbuffer
// .frag) and the shading in another (visbuffer_material.comp).  If the
// two ever compute alpha differently, a pixel is shaded from a triangle
// the depth test rejected -- so neither may own a private copy of this.
//
// The order below is load-bearing and is NOT symmetric between the two
// aging paths; it reproduces cluster_bindless.frag exactly:
//
//   * BINDLESS_MAT_LEAF_AGE without LEAF_MASK ages the base-colour
//     FACTOR, before the texture multiply, with the group packed in the
//     material flags (UNPACK_BINDLESS_LEAF_GROUP);
//   * BINDLESS_MAT_LEAF_MASK ages the PRODUCT, after the multiply, with
//     the group read per texel from the cohort map in normal_textures
//     (falling back to the packed group when no map is bound).
//
// NOT yet a drop-in for cluster_bindless.frag: that shader's albedo
// fetch also carries the magenta unresident-page diagnostic and the
// DEPTH_SURFACE triplanar sample, neither of which matters to a cutout
// (the diagnostic path yields alpha 1, and depth surfaces are opaque).
// Switching cluster_bindless.frag onto this function is worth doing once
// the vis path is proven, so there is exactly one copy; until then the
// divergence is confined to those two cases.
//
// Requires the cluster bindless set (cluster_bindless_bindings.glsl.h),
// vt_sample.glsl.h, leaf_age.glsl.h and a `camera_info` in scope.

// Albedo RGBA as the cutout sees it: factor x texture, aged.
// `lod_uv_ddx/ddy` are the analytic UV gradients from the visibility
// buffer's barycentric derivation; pass vec2(0) from a fragment shader
// to let the hardware pick the mip.
vec4 clusterCutoutAlbedo(uint mat_idx, int mat_flags, vec2 uv,
                         vec2 lod_uv_ddx, vec2 lod_uv_ddy,
                         bool have_grad) {
    vec4 base = material_params[mat_idx].base_color_factor;
    bool leaf_mask = (mat_flags & BINDLESS_MAT_LEAF_MASK) != 0;
    uint profile = (uint(mat_flags) >> BINDLESS_MAT_TREE_PROFILE_SHIFT) & 1023u;
    float life = material_params[mat_idx].tree_life.x;

    if (!leaf_mask && (mat_flags & BINDLESS_MAT_LEAF_AGE) != 0) {
        base = leafAgedColor(base, UNPACK_BINDLESS_LEAF_GROUP(uint(mat_flags)),
                             camera_info.global_leaf_age, profile, life);
    }

    vec4 tex = vec4(1.0);
    uint vt      = material_params[mat_idx].albedo_vt_id;
    int  tex_idx = material_params[mat_idx].base_color_tex_idx;
    if (vt != 0u) {
        tex = vtSampleAlbedo(vt, uv);
    } else if (tex_idx >= 0) {
        tex = have_grad
            ? textureGrad(base_color_textures[nonuniformEXT(tex_idx)], uv,
                          lod_uv_ddx, lod_uv_ddy)
            : texture(base_color_textures[nonuniformEXT(tex_idx)], uv);
    }
    vec4 albedo4 = base * tex;

    if (leaf_mask) {
        int  age_idx = material_params[mat_idx].normal_tex_idx;
        uint group   = UNPACK_BINDLESS_LEAF_GROUP(uint(mat_flags));
        if (age_idx >= 0) {
            // NEAREST-ish by construction: the cohort id is a bin centre,
            // so it must not be blended across a leaf boundary.  The
            // gradient form keeps the same mip the colour fetch used.
            float code = have_grad
                ? textureGrad(normal_textures[nonuniformEXT(age_idx)], uv,
                              lod_uv_ddx, lod_uv_ddy).b
                : texture(normal_textures[nonuniformEXT(age_idx)], uv).b;
            group = leafMaskGroup(code);
        }
        albedo4 = leafAgedColor(albedo4, group, camera_info.global_leaf_age,
                                profile, life);
    }
    return albedo4;
}

// True when this fragment/pixel is cut away.
bool clusterCutoutRejects(uint mat_idx, int mat_flags, float alpha) {
    if ((mat_flags & BINDLESS_MAT_ALPHA_MASK) == 0) return false;
    return alpha < DBG_CUTOFF(material_params[mat_idx].alpha_cutoff);
}

#endif // CLUSTER_CUTOUT_GLSL_H
