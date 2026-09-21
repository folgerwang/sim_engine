#ifndef VISBUFFER_COMMON_GLSL_H
#define VISBUFFER_COMMON_GLSL_H
// ── Visibility buffer: triangle fetch + analytic barycentrics ────────
//
// The vis pass stores (cluster_idx, prim_id) per pixel and NOTHING else.
// This header is what lets the material pass rebuild everything the
// old fragment shader got from the rasterizer's interpolators:
//
//   * the triangle, from ClusterDrawInfo + the merged index/vertex
//     buffers (both already STORAGE_BUFFER_BIT for the mesh-shader CSM
//     path, so no new uploads);
//   * perspective-correct barycentrics AND their screen-space
//     gradients, derived analytically from the three clip positions --
//     Deferred Attribute Interpolation Shading (Schied & Dachsbacher,
//     HPG 2015).  The gradients are what replace dFdx/dFdy, which do
//     not exist in compute: without them every texture fetch would
//     have to pick a mip by hand and the whole point (cheap, correct
//     texturing) is lost.
//
// This is only sound because the cluster path does NO vertex animation
// (cluster_bindless.vert: v_world_pos = in_position, positions are
// pre-baked world space).  If that ever changes, the reconstruction
// must replay the same displacement or the barycentrics go wrong.

struct VbBary {
    vec3 lambda;   // perspective-correct barycentrics
    vec3 ddx;      // d(lambda)/d(screen x)
    vec3 ddy;      // d(lambda)/d(screen y)
};

// `ndc` is the pixel centre in [-1,1]; `win` the render size in pixels.
VbBary vbCalcFullBary(vec4 p0, vec4 p1, vec4 p2, vec2 ndc, vec2 win) {
    VbBary r;
    vec3 inv_w = 1.0 / vec3(p0.w, p1.w, p2.w);
    vec2 n0 = p0.xy * inv_w.x;
    vec2 n1 = p1.xy * inv_w.y;
    vec2 n2 = p2.xy * inv_w.z;

    float inv_det = 1.0 / determinant(mat2(n2 - n1, n0 - n1));
    r.ddx = vec3(n1.y - n2.y, n2.y - n0.y, n0.y - n1.y) * inv_det * inv_w;
    r.ddy = vec3(n2.x - n1.x, n0.x - n2.x, n1.x - n0.x) * inv_det * inv_w;
    float ddx_sum = dot(r.ddx, vec3(1.0));
    float ddy_sum = dot(r.ddy, vec3(1.0));

    vec2  delta       = ndc - n0;
    float interp_invw = inv_w.x + delta.x * ddx_sum + delta.y * ddy_sum;
    float interp_w    = 1.0 / interp_invw;

    r.lambda = vec3(
        interp_w * (inv_w.x + delta.x * r.ddx.x + delta.y * r.ddy.x),
        interp_w * (0.0     + delta.x * r.ddx.y + delta.y * r.ddy.y),
        interp_w * (0.0     + delta.x * r.ddx.z + delta.y * r.ddy.z));

    // NDC -> pixel gradients.  y flips: NDC grows up, pixels grow down.
    r.ddx    *= (2.0 / win.x);
    r.ddy    *= (2.0 / win.y);
    ddx_sum  *= (2.0 / win.x);
    ddy_sum  *= (2.0 / win.y);
    r.ddy    *= -1.0;
    ddy_sum  *= -1.0;

    float w_ddx = 1.0 / (interp_invw + ddx_sum);
    float w_ddy = 1.0 / (interp_invw + ddy_sum);
    r.ddx = w_ddx * (r.lambda * interp_invw + r.ddx) - r.lambda;
    r.ddy = w_ddy * (r.lambda * interp_invw + r.ddy) - r.lambda;
    return r;
}

// Scalar attribute: .x = value, .y = d/dx, .z = d/dy
vec3 vbInterp(VbBary b, vec3 attr_per_vertex) {
    return vec3(dot(attr_per_vertex, b.lambda),
                dot(attr_per_vertex, b.ddx),
                dot(attr_per_vertex, b.ddy));
}

// Value only, for attributes whose gradients nothing needs (normals,
// tangents, world position).
float vbInterp1(VbBary b, vec3 attr_per_vertex) {
    return dot(attr_per_vertex, b.lambda);
}
vec3 vbInterp3(VbBary b, vec3 a0, vec3 a1, vec3 a2) {
    return a0 * b.lambda.x + a1 * b.lambda.y + a2 * b.lambda.z;
}

// ── Visibility id packing ────────────────────────────────────────────
// R32G32_UINT: x = cluster_idx + 1 (0 = empty pixel), y = prim_id.
// Clusters are whole glTF primitives here (up to 21000 triangles), so
// prim_id needs 15 bits and cluster count is unbounded -- a single
// 32-bit word is only safe once clusters are re-cut to meshlet size.
#define VB_EMPTY 0u
uvec2 vbPack(uint cluster_idx, uint prim_id) {
    return uvec2(cluster_idx + 1u, prim_id);
}
bool vbIsEmpty(uvec2 v)      { return v.x == VB_EMPTY; }
uint vbCluster(uvec2 v)      { return v.x - 1u; }
uint vbPrim(uvec2 v)         { return v.y; }

#endif // VISBUFFER_COMMON_GLSL_H
