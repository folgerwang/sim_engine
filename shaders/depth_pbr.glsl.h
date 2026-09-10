#ifndef TERRAIN_DEPTH_PBR_GLSL
#define TERRAIN_DEPTH_PBR_GLSL
// R=occlusion, G=roughness, B=metallic, A=normalized front-surface depth.
// The three projection tiles must not filter into one another. Relief affects
// lighting only: card coverage, shadow geometry and fragment depth are unchanged.
vec2 depthPbrUV(vec2 uv, vec2 texel, float tile) {
    return clamp(uv, vec2(tile / 3.0 + texel.x * 0.5, texel.y * 0.5),
                     vec2((tile + 1.0) / 3.0 - texel.x * 0.5, 1.0 - texel.y * 0.5));
}
vec3 depthPbrNormal(sampler2D atlas, vec2 uv) {
    vec2 size = vec2(textureSize(atlas, 0));
    vec2 texel = 1.0 / size;
    // Use the same explicit LOD for all taps; gradients at tiny impostor sizes
    // otherwise change abruptly at mip boundaries.
    float lod = clamp(textureQueryLod(atlas, uv).x, 0.0, 4.0);
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
// Repeating PCG surfaces use their original height amplitude, not the
// three-view card clamp. Native previews/mips may change resolution; scale is
// per UV repeat so the same material retains its relief at every resolution.
vec3 depthSurfaceNormal(sampler2D tex, vec2 uv, float scale) {
    if (scale <= 0.0) return vec3(0,0,1);
    float lod=max(textureQueryLod(tex,uv).x,0.0);
    vec2 stepUV=exp2(lod)/vec2(textureSize(tex,0));
    float l=textureLod(tex,uv-vec2(stepUV.x,0),lod).a;
    float r=textureLod(tex,uv+vec2(stepUV.x,0),lod).a;
    float u=textureLod(tex,uv-vec2(0,stepUV.y),lod).a;
    float d=textureLod(tex,uv+vec2(0,stepUV.y),lod).a;
    return normalize(vec3(-vec2(r-l,d-u)*scale/(2.0*stepUV),1.0));
}
vec4 depthTriplanarSample(sampler2D tex,vec3 p,vec3 n,float tile) {
    vec3 w=max(abs(n)-0.28,vec3(0));w/=max(w.x+w.y+w.z,1e-5);
    p/=max(tile,1e-3);
    vec4 c=vec4(0);
    if(w.x>0.0)c+=texture(tex,p.zy)*w.x;
    if(w.y>0.0)c+=texture(tex,p.xz)*w.y;
    if(w.z>0.0)c+=texture(tex,p.xy)*w.z;
    return c;
}
vec3 depthWorldNormal(float height,vec3 p,vec3 n,float scaleMetres) {
    vec3 x=dFdx(p),y=dFdy(p);
    vec3 rx=cross(y,n),ry=cross(n,x);
    float det=dot(x,rx);
    vec3 grad=(dFdx(height)*rx+dFdy(height)*ry)*scaleMetres;
    if(abs(det)<1e-10)return n;
    return normalize(n-grad/det);
}
#endif
