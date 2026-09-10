#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "veg_sway.glsl.h"
layout(std430,set=VIEW_PARAMS_SET,binding=VIEW_CAMERA_BUFFER_INDEX)
readonly buffer CameraInfoBuffer { ViewCameraInfo camera_info; };
struct Leaf { vec4 vertices[12]; vec4 instance_birth; vec4 up_bits; vec4 color; vec4 ground_wind; };
layout(std430,set=0,binding=0) readonly buffer Leaves { Leaf leaves[]; };
layout(location=0) out vec3 color;
vec3 birthPoint(Leaf leaf, int v) {
    vec4 p=leaf.vertices[v];
    return p.xyz+vegSwayVec(vegSwayTravel(leaf.instance_birth.xyz,p.w,
        leaf.instance_birth.w,leaf.up_bits.xyz,floatBitsToUint(leaf.up_bits.w)));
}
void main() {
    Leaf leaf=leaves[gl_InstanceIndex];
    float t=max(camera_info.time_s-leaf.instance_birth.w,0.);
    gl_Position=vec4(2,2,2,1); color=leaf.color.rgb;
    if(t>90.) return;
    vec3 center=vec3(0);
    for(int i=0;i<12;++i) center+=birthPoint(leaf,i)/12.;
    vec3 p=birthPoint(leaf,gl_VertexIndex)-center;
    // Strong air drag (3/s) and lift (8 m/s²) give a slow ~0.60 m/s
    // terminal descent. Exact linear-drag integration avoids frame-step spikes.
    float drag=3.,response=t-(1.-exp(-drag*t))/drag;
    vec3 wind=vec3(kVegWindDir.x,0,kVegWindDir.y)*leaf.ground_wind.y;
    vec3 displacement=(wind+vec3(0,(8.-9.81)/drag,0))*response;
    // Zero displacement at birth; bounded flutter rides the same wind direction.
    float phase=leaf.ground_wind.z;
    displacement+=vec3(-kVegWindDir.y,.12,kVegWindDir.x)*
        ((sin(t*2.1+phase)-sin(phase))*.14*(1.-exp(-t))*min(leaf.ground_wind.y,1.));
    float landed=clamp((center.y+displacement.y-leaf.ground_wind.x)/.15,0.,1.);
    float angle=t*(1.2+.4*sin(phase))*landed;
    vec3 axis=normalize(vec3(cos(phase),.35,sin(phase)));
    p=p*cos(angle)+cross(axis,p)*sin(angle)+axis*dot(axis,p)*(1.-cos(angle));
    vec3 world=center+displacement+p;
    world.y=max(world.y,leaf.ground_wind.x+.015);
    gl_Position=camera_info.view_proj*vec4(world,1);
}
