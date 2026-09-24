#ifndef TERRAIN_DEPTH_PBR_GLSL
#define TERRAIN_DEPTH_PBR_GLSL
// R=occlusion, G=roughness, B=metallic, A=normalized front-surface depth.
// The three projection tiles must not filter into one another. Relief affects
// lighting only: card coverage, shadow geometry and fragment depth are unchanged.
vec2 depthPbrUV(vec2 uv, vec2 texel, float tile) {
    return clamp(uv, vec2(tile / 3.0 + texel.x * 0.5, texel.y * 0.5),
                     vec2((tile + 1.0) / 3.0 - texel.x * 0.5, 1.0 - texel.y * 0.5));
}
// ── Explicit-LOD cores ───────────────────────────────────────────────
// textureQueryLod, texture() and dFdx/dFdy are all fragment-only, and
// glslang rejects them in a compute unit whether or not the call can be
// reached.  The visibility-buffer material pass (visbuffer_material.comp)
// is compute and carries analytic gradients from the barycentric
// derivation, so every function here is split: a core taking the LOD (or
// the gradients) explicitly, and a fragment-only wrapper that derives it
// from the hardware.  Both stages run the SAME core, so the two shading
// paths cannot drift.
//
// depthPbrLodFromGrad is the mip textureQueryLod would have returned:
// log2 of the longest UV gradient in texels, the standard isotropic
// rho.  Same clamp range the wrapper applied.
float depthPbrLodFromGrad(sampler2D tex, vec2 ddx, vec2 ddy) {
    vec2 size = vec2(textureSize(tex, 0));
    vec2 dx = ddx * size;
    vec2 dy = ddy * size;
    float rho2 = max(dot(dx, dx), dot(dy, dy));
    return 0.5 * log2(max(rho2, 1e-12));
}

vec3 depthPbrNormalLod(sampler2D atlas, vec2 uv, float raw_lod) {
    vec2 size = vec2(textureSize(atlas, 0));
    vec2 texel = 1.0 / size;
    // Use the same explicit LOD for all taps; gradients at tiny impostor sizes
    // otherwise change abruptly at mip boundaries.
    float lod = clamp(raw_lod, 0.0, 4.0);
    vec2 stepUV = texel * exp2(lod);
    float tile = clamp(floor(uv.x*3.0),0.0,2.0);
    float l = textureLod(atlas, depthPbrUV(uv-vec2(stepUV.x,0),stepUV,tile),lod).a;
    float r = textureLod(atlas, depthPbrUV(uv+vec2(stepUV.x,0),stepUV,tile),lod).a;
    float u = textureLod(atlas, depthPbrUV(uv-vec2(0,stepUV.y),stepUV,tile),lod).a;
    float d = textureLod(atlas, depthPbrUV(uv+vec2(0,stepUV.y),stepUV,tile),lod).a;
    vec2 slope = vec2((r-l)/(6.0*stepUV.x), (d-u)/(2.0*stepUV.y));
    // Disconnected leaves create depth discontinuities. Bound their slope so
    // a silhouette cannot become a black grazing-angle lighting spike.
    slope = clamp(slope,vec2(-2.0),vec2(2.0));
    return normalize(vec3(-slope,1.0));
}
#ifndef VT_NO_DERIVATIVES
vec3 depthPbrNormal(sampler2D atlas, vec2 uv) {
    return depthPbrNormalLod(atlas, uv, textureQueryLod(atlas, uv).x);
}
#endif
// Repeating PCG surfaces use their original height amplitude, not the
// three-view card clamp. Native previews/mips may change resolution; scale is
// per UV repeat so the same material retains its relief at every resolution.
vec3 depthSurfaceNormalLod(sampler2D tex, vec2 uv, float scale,
                           float raw_lod) {
    if (scale <= 0.0) return vec3(0,0,1);
    float lod=max(raw_lod,0.0);
    vec2 stepUV=exp2(lod)/vec2(textureSize(tex,0));
    float l=textureLod(tex,uv-vec2(stepUV.x,0),lod).a;
    float r=textureLod(tex,uv+vec2(stepUV.x,0),lod).a;
    float u=textureLod(tex,uv-vec2(0,stepUV.y),lod).a;
    float d=textureLod(tex,uv+vec2(0,stepUV.y),lod).a;
    return normalize(vec3(-vec2(r-l,d-u)*scale/(2.0*stepUV),1.0));
}
#ifndef VT_NO_DERIVATIVES
vec3 depthSurfaceNormal(sampler2D tex, vec2 uv, float scale) {
    return depthSurfaceNormalLod(tex, uv, scale, textureQueryLod(tex,uv).x);
}
#endif
vec4 depthTriplanarSampleLod(sampler2D tex,vec3 p,vec3 n,float tile,
                             float lod) {
    vec3 w=max(abs(n)-0.28,vec3(0));w/=max(w.x+w.y+w.z,1e-5);
    p/=max(tile,1e-3);
    vec4 c=vec4(0);
    if(w.x>0.0)c+=textureLod(tex,p.zy,lod)*w.x;
    if(w.y>0.0)c+=textureLod(tex,p.xz,lod)*w.y;
    if(w.z>0.0)c+=textureLod(tex,p.xy,lod)*w.z;
    return c;
}
#ifndef VT_NO_DERIVATIVES
vec4 depthTriplanarSample(sampler2D tex,vec3 p,vec3 n,float tile) {
    vec3 w=max(abs(n)-0.28,vec3(0));w/=max(w.x+w.y+w.z,1e-5);
    vec3 q=p/max(tile,1e-3);
    vec4 c=vec4(0);
    if(w.x>0.0)c+=texture(tex,q.zy)*w.x;
    if(w.y>0.0)c+=texture(tex,q.xz)*w.y;
    if(w.z>0.0)c+=texture(tex,q.xy)*w.z;
    return c;
}
#endif
// Explicit-gradient core: `x`/`y` are d(world position)/d(screen x,y) and
// `hx`/`hy` the matching derivatives of `height`.  The fragment wrapper
// below fills them from dFdx/dFdy; the compute path gets `x`/`y` from the
// triangle's analytic barycentric gradients and `hx`/`hy` from a pair of
// finite differences.
vec3 depthWorldNormalGrad(vec3 n,float scaleMetres,
                          vec3 x,vec3 y,float hx,float hy) {
    vec3 rx=cross(y,n),ry=cross(n,x);
    float det=dot(x,rx);
    vec3 grad=(hx*rx+hy*ry)*scaleMetres;
    if(abs(det)<1e-10)return n;
    return normalize(n-grad/det);
}
#ifndef VT_NO_DERIVATIVES
vec3 depthWorldNormal(float height,vec3 p,vec3 n,float scaleMetres) {
    return depthWorldNormalGrad(n,scaleMetres,
                                dFdx(p),dFdy(p),dFdx(height),dFdy(height));
}
#endif
// ── Parallax occlusion for repeating DEPTH_SURFACE materials ─────────
// UV path only: the triplanar branch has no UV frame to march in.  The
// ORM texture's .a is height (1 = the proud face of a stone, 0 = the back
// of a joint) and `scale` is the relief measured in UV units -- the
// material's depth_scale, height_range_m / tile_m -- so the march is
// metres-true at any UV density.  x/y are d(world)/d(screen) and uvx/uvy
// d(uv)/d(screen): dFdx in the raster paths, the analytic barycentric
// gradients in visbuffer_material.comp, so all three shading paths run
// this one core.  `lod` is fixed from the UNdisplaced uv.  Returns the
// displaced uv.  Shading only: depth, silhouette and shadow geometry keep
// the flat face.  Faded out between DEPTH_POM_NEAR_M and _FAR_M, where
// the relief is sub-pixel anyway and the taps would be wasted.
#ifndef DEPTH_POM_NEAR_M
#define DEPTH_POM_NEAR_M 18.0
#endif
#ifndef DEPTH_POM_FAR_M
#define DEPTH_POM_FAR_M 42.0
#endif
vec2 depthSurfaceParallaxGrad(sampler2D tex, vec2 uv, vec2 uvx, vec2 uvy,
                              vec3 x, vec3 y, vec3 n, vec3 v, float dist,
                              float scale, float lod) {
    float fade = 1.0 - smoothstep(DEPTH_POM_NEAR_M, DEPTH_POM_FAR_M, dist);
    float vn = dot(n, v);
    if (scale <= 0.0 || fade <= 0.0 || vn <= 0.05) return uv;
    // in-plane gradients of u and v in world space (cotangent frame)
    vec3 c2 = cross(y, n), c1 = cross(n, x);
    float det = dot(x, c2);
    if (abs(det) < 1e-14) return uv;
    vec3 gu = (c2 * uvx.x + c1 * uvy.x) / det;
    vec3 gv = (c2 * uvx.y + c1 * uvy.y) / det;
    float g2 = 0.5 * (dot(gu, gu) + dot(gv, gv));
    if (g2 < 1e-12) return uv;
    float relief_m = scale * inversesqrt(g2) * fade;
    // sideways metres per metre of depth down the view ray (grazing capped)
    vec3 side = -(v - n * vn) / max(vn, 0.3);
    vec2 span = vec2(dot(gu, side), dot(gv, side)) * relief_m;
    int steps = int(mix(20.0, 6.0, clamp(vn, 0.0, 1.0)));
    float dt = 1.0 / float(steps);
    float t = 0.0;
    float d = 1.0 - textureLod(tex, uv, lod).a;   // depth below the crest
    if (d <= 0.0) return uv;
    float prev = d;
    for (int i = 0; i < 24; ++i) {
        if (i >= steps || d <= t) break;
        prev = d - t;
        t += dt;
        d = 1.0 - textureLod(tex, uv + span * t, lod).a;
    }
    float cur = d - t;                            // <= 0 once the ray is in
    float w = cur < 0.0 ? prev / max(prev - cur, 1e-5) : 1.0;
    return uv + span * clamp(t - dt + w * dt, 0.0, 1.0);
}
#endif
