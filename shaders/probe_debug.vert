#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ─── probe_debug.vert ───────────────────────────────────────────────────
// Per-probe debug-draw vertex shader.  Renders a level-1 subdivided
// icosphere at every AmbientProbeSystem probe position.  The fragment
// shader evaluates the probe's SH coefficients in the surface-normal
// direction so each sphere visually shows the irradiance it would
// contribute.
//
// Geometry: a 12-vertex icosahedron (20 base triangles), each base
// triangle subdivided once into 4 sub-triangles by inserting normalized
// midpoints.  Result is 80 triangles ⇒ 240 vertex outputs per probe.
// No vertex/index buffer needed — geometry is generated entirely from
// gl_VertexIndex via the constant arrays below.  Caller draws
// `vkCmdDraw(240, num_probes, ...)` (instanced).
//
// 240 = 20 (base faces) × 4 (sub-triangles per face) × 3 (vertices
// per sub-triangle).

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer { ViewCameraInfo camera_info; };

struct ProbeData { vec4 position; vec4 sh[9]; };
layout(std430, set = 0, binding = 0)
    readonly buffer ProbeBuffer { ProbeData probes[]; };

// Standard icosahedron — 12 vertices on the corners of three mutually
// perpendicular golden-ratio rectangles.  Vertex ordering matches the
// canonical index list in ICO_I below.
const float t = 1.61803398875;  // (1 + sqrt(5)) / 2
const vec3 ICO_V[12] = vec3[12](
    vec3(-1,  t, 0),    //  0
    vec3( 1,  t, 0),    //  1
    vec3(-1, -t, 0),    //  2
    vec3( 1, -t, 0),    //  3
    vec3( 0, -1,  t),   //  4
    vec3( 0,  1,  t),   //  5
    vec3( 0, -1, -t),   //  6
    vec3( 0,  1, -t),   //  7
    vec3( t,  0, -1),   //  8
    vec3( t,  0,  1),   //  9
    vec3(-t,  0, -1),   // 10
    vec3(-t,  0,  1)    // 11
);

// 20 triangular faces — canonical CCW-from-outside winding.
const uint ICO_I[60] = uint[60](
    0, 11,  5,   0,  5,  1,   0,  1,  7,   0,  7, 10,   0, 10, 11,
    1,  5,  9,   5, 11,  4,  11, 10,  2,  10,  7,  6,   7,  1,  8,
    3,  9,  4,   3,  4,  2,   3,  2,  6,   3,  6,  8,   3,  8,  9,
    4,  9,  5,   2,  4, 11,   6,  2, 10,   8,  6,  7,   9,  8,  1
);

// Sub-triangle layout for the 4-way subdivision.  Each parent triangle
// has corners {A, B, C} (indices 0/1/2) and edge midpoints {AB, BC, CA}
// (indices 3/4/5).  The 4 sub-triangles tile the parent (CCW from
// outside):
//   • corner near A: A, AB, CA
//   • corner near B: AB, B, BC
//   • corner near C: CA, BC, C
//   • inverted centre: AB, BC, CA   (note: inverted winding cancels
//                                    out via the pipeline's cull=NONE)
const uint SUB_I[12] = uint[12](
    0, 3, 5,        // sub 0
    3, 1, 4,        // sub 1
    5, 4, 2,        // sub 2
    3, 4, 5         // sub 3 (centre)
);

layout(push_constant) uniform PushConstants {
    float radius;
    float pad0, pad1, pad2;
} pc;

layout(location = 0) out vec3 v_normal;
layout(location = 1) flat out uint v_probe_idx;

void main() {
    // Decode gl_VertexIndex into (parent face, vertex within parent's
    // 4-subdivision = 12 outputs).
    uint pf  = uint(gl_VertexIndex) / 12u;   // parent face 0..19
    uint pfv = uint(gl_VertexIndex) % 12u;   // 0..11 within parent

    // Three corners of the parent icosahedron face (normalized).
    vec3 A = normalize(ICO_V[ICO_I[pf * 3u + 0u]]);
    vec3 B = normalize(ICO_V[ICO_I[pf * 3u + 1u]]);
    vec3 C = normalize(ICO_V[ICO_I[pf * 3u + 2u]]);

    // Midpoints of the parent edges, projected back onto the unit
    // sphere.  This is what makes the subdivision approximate a sphere
    // rather than a flat triangle subdivision.
    vec3 AB = normalize((A + B) * 0.5);
    vec3 BC = normalize((B + C) * 0.5);
    vec3 CA = normalize((C + A) * 0.5);

    // Pick the actual vertex for this output via the SUB_I table,
    // which maps each of the 12 parent-face outputs to a slot in the
    // {A, B, C, AB, BC, CA} array below.
    vec3 P[6] = vec3[6](A, B, C, AB, BC, CA);
    vec3 n = P[SUB_I[pfv]];

    uint probe_idx = uint(gl_InstanceIndex);
    vec3 probe_pos = probes[probe_idx].position.xyz;
    vec3 world_pos = probe_pos + n * pc.radius;

    gl_Position  = camera_info.view_proj * vec4(world_pos, 1.0);
    v_normal     = n;
    v_probe_idx  = probe_idx;
}
