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
    // ── v52: A LEAF SPIRALS DOWN AND STAYS WHERE IT LANDS ───────────
    // Before: the wind term grew with time for the whole 90 s life, so a
    // leaf kept sliding downwind after it had reached the ground (the
    // ground clamp only held its height), and the flutter was a flat
    // sideways sway.  Now the descent time is solved from the height
    // above ground, every motion term stops at that instant, and the
    // path is a helix: a growing circle around the descent line, whose
    // radius and rate vary per leaf, on top of a modest downwind drift.
    float drag=3.;
    float v_down=(9.81-8.)/drag;                            // ~0.60 m/s terminal descent
    float h=max(center.y-leaf.ground_wind.x,0.);
    // exact linear-drag descent: y(t)=v_down*(t-(1-e^-dt)/d); invert for t_land
    float t_land=h/v_down+1./drag;
    float ta=min(t,t_land);                                  // airborne time
    float response=ta-(1.-exp(-drag*ta))/drag;
    float phase=leaf.ground_wind.z;
    float gust=min(leaf.ground_wind.y,1.);
    vec3 wind=vec3(kVegWindDir.x,0,kVegWindDir.y)*(.35*gust);   // gentle downwind drift
    vec3 displacement=(wind+vec3(0,-v_down,0))*response;
    // the helix: radius eases in from 0 over the first second, then grows
    // slowly; one turn every ~3-5 s, direction per leaf
    float turn=(1.3+.6*sin(phase*3.1))*(phase>3.14159?1.:-1.);
    float radius=(.25+.2*sin(phase*1.7))*(1.-exp(-ta))*(1.+.08*ta);
    float ang=turn*ta+phase;
    displacement+=radius*(vec3(cos(ang),0,sin(ang))-vec3(cos(phase),0,sin(phase)));
    // a little bob riding the helix
    displacement.y+=.06*sin(turn*ta*2.+phase)*(1.-exp(-ta));
    // spin only while airborne, easing out over the last metre
    float landed=clamp(h-v_down*response,0.,1.);
    float angle=(1.2+.4*sin(phase))*response*landed+(1.2+.4*sin(phase))*(1.-landed)*t_land*.0;
    vec3 axis=normalize(vec3(cos(phase),.35,sin(phase)));
    p=p*cos(angle)+cross(axis,p)*sin(angle)+axis*dot(axis,p)*(1.-cos(angle));
    // on the ground the leaf lies FLAT: squash its profile toward the ground plane
    p.y*= mix(.15,1.,landed);
    vec3 world=center+displacement+p;
    world.y=max(world.y,leaf.ground_wind.x+.015);
    gl_Position=camera_info.view_proj*vec4(world,1);
}
