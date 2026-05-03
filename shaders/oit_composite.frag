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

    // Pixels with no translucent contribution have accum == 0 and
    // reveal == 1, which would produce 0/0 in the divide.  Skip them
    // entirely so the composite blend leaves the scene colour intact
    // and we don't stamp depth where there's no glass (otherwise the
    // sky pass below would be blocked from drawing over empty sky).
    if (reveal >= 1.0 - 1e-5) {
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
