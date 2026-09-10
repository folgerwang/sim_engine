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
void main() {
    const vec2 corners[6]=vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    disk=corners[gl_VertexIndex]; color=vec3(0);status_color=vec3(0);
    gl_Position=vec4(2,2,2,1);
    uint index=uint(gl_InstanceIndex),tile=index/32u,sub=index%32u;
    if(state[tile].y==0u) return;
    vec2 xz=mix(bounds[tile].xy,bounds[tile].zw,(vec2(sub%2u,(sub/2u)%2u)+0.5)/2.0);
    float y=(floor(camera_info.position.y/96.0)-4.0+float(sub/4u)+0.5)*96.0;
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
        vec3 light=vec3(0);float occlusion=0.0;
        for(int k=0;k<6;++k) {
            light+=vec3(unpackHalf2x16(p.radiance[k].x),unpackHalf2x16(p.radiance[k].y).x)/6.0;
            occlusion+=unpackHalf2x16(p.occlusion[k]).x/6.0;
        }
        if(pc.mode==1u) color=pow(light/(vec3(1)+light),vec3(1.0/2.2));
        else if(pc.mode==2u) color=vec3(clamp(p.state.x,0,1));
        else color=vec3(occlusion);
    }
    clip.xy+=disk*(2.0*pc.pixels/pc.viewport.xy)*clip.w;
    if(pc.viewport.z>0.5) clip.z=0.0;
    gl_Position=clip;
}
