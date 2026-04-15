#version 450
#extension GL_ARB_separate_shader_objects : enable

// Push constant: which cascade layer to visualise (0–3).
layout(push_constant) uniform CsmDebugParams {
    int cascade_idx;
} params;

// The full 4-layer CSM depth texture array.
// Bound with layout DEPTH_READ_ONLY_OPTIMAL; sampled value comes from .r.
layout(set = 0, binding = 0) uniform sampler2DArray csm_depth_sampler;

layout(location = 0) in  vec2 v_UV;
layout(location = 0) out vec4 out_color;

void main() {
    float d = texture(csm_depth_sampler,
                      vec3(v_UV, float(params.cascade_idx))).r;
    // Raw NDC depth: 0 = near-plane (black), 1 = far-plane (white).
    // Invert so near objects appear bright and are easier to read.
    float vis = 1.0 - d;
    out_color = vec4(vis, vis, vis, 1.0);
}
