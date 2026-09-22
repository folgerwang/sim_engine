#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
layout(std430,set=VIEW_PARAMS_SET,binding=VIEW_CAMERA_BUFFER_INDEX)
readonly buffer CameraInfoBuffer { ViewCameraInfo camera_info; };
struct WorldProbe { vec4 cell_epoch; vec4 state; uvec2 radiance[6]; uint occlusion[6]; };
layout(std430,set=0,binding=0) readonly buffer ProbeBuffer {
    vec4 bounds[2048]; uvec4 state[2048]; uvec4 lookup[4096]; WorldProbe probes[];
};
layout(push_constant) uniform Params {
    uint epoch; float range; float pixels; uint mode;
    vec4 viewport; // width, height, xray, reserved
} pc;
layout(location=0) out vec2 disk;
layout(location=1) flat out vec3 color;
layout(location=2) flat out vec3 status_color;
layout(location=3) flat out vec3 rad_px;   // +x  -x  ... the six cube-axis radiances
layout(location=4) flat out vec3 rad_nx;
layout(location=5) flat out vec3 rad_py;
layout(location=6) flat out vec3 rad_ny;
layout(location=7) flat out vec3 rad_pz;
layout(location=8) flat out vec3 rad_nz;
layout(location=9) flat out float lit_ball;   // 1: shade the disk as a sphere lit by its own radiance
// A probe is drawn as a WORLD-SIZED ball (pc.pixels is its radius in
// metres), so it shrinks with distance and is depth-tested like any
// other object (unless the x-ray box is ticked).
const float kBallRadiusM = 2.0;
void main() {
    const vec2 corners[6]=vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    disk=corners[gl_VertexIndex]; color=vec3(0);status_color=vec3(0);lit_ball=0.0;
    rad_px=rad_nx=rad_py=rad_ny=rad_pz=rad_nz=vec3(0);
    gl_Position=vec4(2,2,2,1);
    uint index=uint(gl_InstanceIndex),tile=index/32u,sub=index%32u;
    if(state[tile].y==0u) return;
    vec2 xz=mix(bounds[tile].xy,bounds[tile].zw,(vec2(sub%2u,(sub/2u)%2u)+0.5)/2.0);
    float y=uintBitsToFloat(state[tile].z)+float(sub/4u)*96.0;   // anchored to the tile's ground
    vec3 pos=vec3(xz.x,y,xz.y);
    if(distance(pos,camera_info.position.xyz)>pc.range) return;
    vec4 clip=camera_info.view_proj*vec4(pos,1);
    if(clip.w<=0.0 || clip.z<0.0 || clip.z>clip.w) return;
    WorldProbe p=probes[index];
    uint epoch=pc.epoch^(state[tile].x*2654435761u);
    bool current=all(equal(p.cell_epoch.xyz,pos)) && floatBitsToUint(p.cell_epoch.w)==epoch;
    float readiness=current?p.state.y:0.0;
    status_color=readiness<0.0?vec3(1,0.1,0.1):(readiness>0.5?vec3(0.1,1,0.2):vec3(1,0.8,0.05));
    color=status_color;
    if(pc.mode!=0u && readiness>0.5) {
        vec3 r[6];
        float occlusion=0.0;
        for(int k=0;k<6;++k) {
            r[k]=vec3(unpackHalf2x16(p.radiance[k].x),unpackHalf2x16(p.radiance[k].y).x);
            occlusion+=unpackHalf2x16(p.occlusion[k]).x/6.0;
        }
        if(pc.mode==1u) {
            rad_px=r[0];rad_nx=r[1];rad_py=r[2];rad_ny=r[3];rad_pz=r[4];rad_nz=r[5];
            lit_ball=1.0;
        }
        else if(pc.mode==2u) color=vec3(clamp(p.state.x,0,1));
        else color=vec3(occlusion);
    }
    // Perspective size: a ball of kBallRadiusM metres.  The billboard
    // spans the ball's projected radius (clip.w is the view depth).
    const float half_h_px = kBallRadiusM * camera_info.proj[1][1] * 0.5 * pc.viewport.y;
    const float px_r = max(half_h_px / max(clip.w, 0.05), 3.0);
    clip.xy+=disk*(2.0*px_r/pc.viewport.xy)*clip.w;
    if(pc.viewport.z>0.5) clip.z=0.0;
    gl_Position=clip;
}
