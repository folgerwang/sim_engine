#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
layout(std430,set=VIEW_PARAMS_SET,binding=VIEW_CAMERA_BUFFER_INDEX)
readonly buffer CameraInfoBuffer { ViewCameraInfo camera_info; };
layout(location=0) in vec2 disk;
layout(location=1) flat in vec3 color;
layout(location=2) flat in vec3 status_color;
layout(location=3) flat in vec3 rad_px;
layout(location=4) flat in vec3 rad_nx;
layout(location=5) flat in vec3 rad_py;
layout(location=6) flat in vec3 rad_ny;
layout(location=7) flat in vec3 rad_pz;
layout(location=8) flat in vec3 rad_nz;
layout(location=9) flat in float lit_ball;
layout(location=0) out vec4 out_color;
void main() {
    float r=dot(disk,disk);
    if(r>1.0) discard;
    if(lit_ball<0.5) {
        out_color=vec4(r>0.6?status_color:color,1.0);
        return;
    }
    // A sphere: the disk point's normal in VIEW space, turned into the
    // world, then the probe's cube-axis radiance blended by that normal
    // (the same n^2 weighting the resolve's gather uses) -- the ball shows
    // the light the probe stores, from every side, at its true colour.
    vec3 nv=vec3(disk.x,disk.y,sqrt(max(1.0-r,0.0)));
    vec3 n=normalize(mat3(camera_info.inv_view)*nv);
    vec3 E = (n.x>0.0?rad_px:rad_nx)*n.x*n.x
           + (n.y>0.0?rad_py:rad_ny)*n.y*n.y
           + (n.z>0.0?rad_pz:rad_nz)*n.z*n.z;
    vec3 c = E/(vec3(1.0)+E);           // simple tonemap
    c = pow(max(c,vec3(0)),vec3(1.0/2.2));
    out_color=vec4(c,1.0);
}
