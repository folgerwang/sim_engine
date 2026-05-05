#version 450
#extension GL_ARB_separate_shader_objects : enable

// ─── oit_composite.frag ────────────────────────────────────────────────
// Resolve pass for Weighted Blended Order-Independent Transparency
// (McGuire & Bavoil 2013).  Runs as a fullscreen triangle (full_screen.vert)
// after the cluster_bindless.frag OIT_OUTPUT variant has populated the
// accum (RGBA16F) and reveal (R8/R16F) targets.
//
// Translucent geometry was rasterised with these per-attachment blend
// equations:
//   accum  = Σᵢ (colourᵢ × αᵢ × wᵢ, αᵢ × wᵢ)            // additive
//   reveal = Π_i (1 − αᵢ)                                // multiplicative
//
// The standard WBOIT resolve recovers an order-independent estimate of
// the back-to-front blended colour:
//   resolved.rgb = accum.rgb / max(accum.a, ε)
//   resolved.a   = (1 − reveal)            // total transparency coverage
//
// This shader writes vec4(resolved.rgb, resolved.a).  The composite
// pipeline blends the result over the existing scene colour with
// (src=SRC_ALPHA, dst=ONE_MINUS_SRC_ALPHA), giving:
//   final = resolved.rgb * (1 − reveal) + dst * reveal
// which matches the classic "transparency over opaque" formula.

layout(set = 0, binding = 0) uniform sampler2D u_accum;
layout(set = 0, binding = 1) uniform sampler2D u_reveal;

layout(location = 0) in  vec2 v_UV;
layout(location = 0) out vec4 outColor;

void main() {
    vec4  accum  = textureLod(u_accum,  v_UV, 0.0);
    float reveal = textureLod(u_reveal, v_UV, 0.0).r;

    // ── Robust "no-glass-here" rejection ────────────────────────────
    // Two independent signals must both indicate that some glass actually
    // wrote to this pixel before we fire the composite, otherwise the
    // fullscreen pass would paint over and stamp depth across the entire
    // scene whenever either accumulator drifts slightly off its clear
    // value (numerical noise from drawing zero glass clusters, residual
    // FP16 from a prior frame's content, etc.).
    //   • reveal must be < 0.99   (cleared to 1.0; only true glass
    //     fragments multiply it down via dst*(1−SRC_COLOR))
    //   • accum.a must be > 1e-3  (glass writes α·weight which is ≥
    //     0.05·1e-2 = 5e-4 even for the most transparent case after
    //     the shader's α-clamp)
    // Both conditions are loose enough that real glass always passes
    // and tight enough that empty-OIT pixels never sneak through.
    if (reveal > 0.99 || accum.a < 1e-3) {
        discard;
    }

    vec3 resolved = accum.rgb / max(accum.a, 1e-5);
    outColor = vec4(resolved, 1.0 - reveal);

    // Stamp depth to a value just below the cleared far plane (1.0).
    // The downstream sky-envmap pass uses LESS_OR_EQUAL against sky's
    // far-plane depth (1.0).  At glass pixels we just wrote 0.99999, so
    // 1.0 <= 0.99999 is false — sky is correctly rejected and our glass
    // colour survives.  At non-glass pixels we discarded above, so depth
    // remains untouched and sky still draws over empty pixels normally.
    gl_FragDepth = 0.99999;
}
