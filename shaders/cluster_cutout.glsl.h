#ifndef CLUSTER_CUTOUT_GLSL_H
#define CLUSTER_CUTOUT_GLSL_H
// A compute unit defines VT_NO_DERIVATIVES, which compiles the hardware
// samplers out; the macro keeps this one source compiling in both stages
// without an #ifdef around every call site.
// The implicit-LOD forms (vtSampleAlbedo, texture()) are fragment-only in
// GLSL -- glslang rejects them in a compute unit whether or not the
// branch containing them can be taken at run time, since it compiles
// both sides.  So they are macros: real calls in a fragment unit, dead
// constants in a compute unit, where have_grad is always true and the
// gradient forms beside them are what actually run.
// A consumer that declares no alpha pool (terrain, say) still has to
// compile this file.  Both alpha entry points then read 1.0 and the
// BINDLESS_MAT_ALPHA_VT branch is dead -- which is correct, since
// nothing would have bound the pool for it anyway.
#ifndef VT_HAS_ALPHA_POOL
#define vtSampleAlphaGrad(id, uv, ddx, ddy) 1.0
#define VT_SAMPLE_ALPHA_HW(id, uv)          1.0
#endif

#ifdef VT_NO_DERIVATIVES
#define VT_SAMPLE_ALBEDO_HW(id, uv) vec4(1.0)
#define TEX_SAMPLE_HW(s, uv)        vec4(1.0)
#else
#define VT_SAMPLE_ALBEDO_HW(id, uv) vtSampleAlbedo(id, uv)
#define TEX_SAMPLE_HW(s, uv)        texture(s, uv)
#endif
#if !defined(VT_NO_DERIVATIVES) && defined(VT_HAS_ALPHA_POOL)
#define VT_SAMPLE_ALPHA_HW(id, uv)  vtSampleAlpha(id, uv)
#elif !defined(VT_SAMPLE_ALPHA_HW)
#define VT_SAMPLE_ALPHA_HW(id, uv)  1.0
#endif
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
    float life = clusterTreeLife(mat_idx);

    if (!leaf_mask && (mat_flags & BINDLESS_MAT_LEAF_AGE) != 0) {
        base = leafAgedColor(base, UNPACK_BINDLESS_LEAF_GROUP(uint(mat_flags)),
                             camera_info.global_leaf_age, profile, life);
    }

    vec4 tex = vec4(1.0);
    uint vt      = material_params[mat_idx].albedo_vt_id;
    int  tex_idx = material_params[mat_idx].base_color_tex_idx;
    if (vt != VT_INVALID_ID) {
        // Gradient form in the compute pass (no dFdx there), hardware
        // form in the raster pass -- same rho^2 metric either way, so
        // both passes land on the same mip and the same page.
        tex = have_grad ? vtSampleAlbedoGrad(vt, uv, lod_uv_ddx, lod_uv_ddy)
                        : VT_SAMPLE_ALBEDO_HW(vt, uv);
    }
    // NO LEGACY BINDLESS FALLBACK.  A material without a VT id used to
    // sample base_color_textures[] here.  It now falls through with
    // tex = vec4(1.0), so albedo4 = base_color_factor alone -- a flat
    // correctly-tinted surface instead of a second texture path.
    //
    // That is a deliberate trade.  vtResolveWalk climbs to the pinned
    // mip tail, so "registered but not streamed yet" already lands on
    // real texels; the only case reaching here is "never registered"
    // (pool full), which should be rare and is better fixed by sizing
    // the pool than by carrying a parallel sampler array for it.
    // With a dedicated BC4 alpha layer the albedo was encoded OPAQUE,
    // so tex.a is 255 and meaningless -- the real cutout alpha comes
    // from the alpha pool.  It shares this material's vt_id and pool
    // slot, so it resolves to the same page; only the sampler differs.
    // One extra small fetch, paid once per pixel in the material pass.
    if ((mat_flags & BINDLESS_MAT_ALPHA_VT) != 0) {
        tex.a = have_grad
            ? vtSampleAlphaGrad(vt, uv, lod_uv_ddx, lod_uv_ddy)
            : VT_SAMPLE_ALPHA_HW(vt, uv);
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
                : TEX_SAMPLE_HW(normal_textures[nonuniformEXT(age_idx)], uv).b;
            group = leafMaskGroup(code);
        }
        albedo4 = leafAgedColor(albedo4, group, camera_info.global_leaf_age,
                                profile, life);
    }
    return albedo4;
}

// ── Cutout alpha ALONE -- the visibility pass's whole job ────────────
// Same value clusterCutoutAlbedo would put in .a, reached without
// fetching albedo RGB at all.  That matters because this runs per
// FRAGMENT: under dense foliage the alpha test executes 10-30x per
// pixel, so it is the one sample whose cost multiplies by the
// overdraw.  A BC4 tap is 8 bytes per block against BC7's 16, and the
// big albedo pool never enters L1TEX during the pass that was measured
// saturating it.
//
// The aging algebra below is NOT redundant with clusterCutoutAlbedo --
// it has to match it exactly, or the visibility pass would keep a
// different set of fragments than the raster path and a pixel could be
// shaded from a triangle the depth test rejected.  It is reproducible
// cheaply because leafAgedColor's ALPHA output depends only on the
// incoming alpha (plus group / season / profile), never on RGB: see
// leafSilhouette and the `sil * vis` return in leaf_age.glsl.h.  So we
// feed it a black vec4 carrying just the alpha.  The RGB math it does
// on that black is wasted ALU and costs nothing here -- this pass is
// latency-bound on texture fetches, not arithmetic.
float clusterCutoutAlpha(uint mat_idx, int mat_flags, vec2 uv,
                         vec2 lod_uv_ddx, vec2 lod_uv_ddy,
                         bool have_grad) {
    // No dedicated alpha layer: the alpha still rides in the albedo
    // (an asset baked before the split, or a legacy bindless texture),
    // so there is nothing cheaper available and we take the full path.
    if ((mat_flags & BINDLESS_MAT_ALPHA_VT) == 0) {
        return clusterCutoutAlbedo(mat_idx, mat_flags, uv,
                                   lod_uv_ddx, lod_uv_ddy, have_grad).a;
    }

    float base_a   = material_params[mat_idx].base_color_factor.a;
    bool  leaf_mask = (mat_flags & BINDLESS_MAT_LEAF_MASK) != 0;
    uint  profile  = (uint(mat_flags) >> BINDLESS_MAT_TREE_PROFILE_SHIFT) & 1023u;
    float life     = clusterTreeLife(mat_idx);

    // Asymmetric, exactly as clusterCutoutAlbedo: LEAF_AGE without
    // LEAF_MASK ages the FACTOR before the texture multiply.
    if (!leaf_mask && (mat_flags & BINDLESS_MAT_LEAF_AGE) != 0) {
        base_a = leafAgedColor(vec4(0.0, 0.0, 0.0, base_a),
                               UNPACK_BINDLESS_LEAF_GROUP(uint(mat_flags)),
                               camera_info.global_leaf_age, profile, life).a;
    }

    uint vt = material_params[mat_idx].albedo_vt_id;
    float a = base_a * (have_grad
        ? vtSampleAlphaGrad(vt, uv, lod_uv_ddx, lod_uv_ddy)
        : VT_SAMPLE_ALPHA_HW(vt, uv));

    // ... and LEAF_MASK ages the PRODUCT afterwards, with the cohort
    // read per texel from the map in normal_textures.  That fetch is
    // unavoidable here -- the raster path pays it too -- but it is a
    // small legacy bindless texture, not the VT albedo chain.
    if (leaf_mask) {
        int  age_idx = material_params[mat_idx].normal_tex_idx;
        uint group   = UNPACK_BINDLESS_LEAF_GROUP(uint(mat_flags));
        if (age_idx >= 0) {
            float code = have_grad
                ? textureGrad(normal_textures[nonuniformEXT(age_idx)], uv,
                              lod_uv_ddx, lod_uv_ddy).b
                : TEX_SAMPLE_HW(normal_textures[nonuniformEXT(age_idx)], uv).b;
            group = leafMaskGroup(code);
        }
        a = leafAgedColor(vec4(0.0, 0.0, 0.0, a), group,
                          camera_info.global_leaf_age, profile, life).a;
    }
    return a;
}

// True when this fragment/pixel is cut away.
bool clusterCutoutRejects(uint mat_idx, int mat_flags, float alpha) {
    if ((mat_flags & BINDLESS_MAT_ALPHA_MASK) == 0) return false;
    return alpha < DBG_CUTOFF(material_params[mat_idx].alpha_cutoff);
}

#endif // CLUSTER_CUTOUT_GLSL_H
