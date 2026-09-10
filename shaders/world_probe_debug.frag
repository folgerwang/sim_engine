#version 450
layout(location=0) in vec2 disk;
layout(location=1) flat in vec3 color;
layout(location=2) flat in vec3 status_color;
layout(location=0) out vec4 out_color;
void main() {
    float r=dot(disk,disk);
    if(r>1.0) discard;
    out_color=vec4(r>0.6?status_color:color,1.0);
}
