#version 450
#extension GL_ARB_separate_shader_objects : enable

// ─── oit_pp_composite.frag ────────────────────────────────────────────
// Resolve pass for the sorted per-pixel A-buffer OIT path.
// Runs as a fullscreen triangle (full_screen.vert) after the translucent
// cluster pipeline (cluster_bindless.frag with OIT_PP_OUTPUT) has
// populated the per-pixel storage:
//   u_pp_count  R32_UINT       : fragment count for this pixel, 0..K.
//   u_pp_color  RGBA16_SFLOAT  : K layers, slot[i] = (rgb*α, α) for the
//                                i'th captured fragment.  RGB is
//                                premultiplied; alpha is kept explicit so
//                                we can rebuild the transmittance chain.
//   u_pp_depth  R32_SFLOAT     : K layers, slot[i] = sort key (typically
//                                gl_FragCoord.z, NDC depth) for fragment i.
//
// Algorithm:
//   1. If count == 0 → discard (no glass at this pixel, sky / opaque
//      content survives untouched).
//   2. Read all K slots (we cap the loop at K so it stays static; the
//      shader compiler unrolls a K=4 loop cleanly).
//   3. Insertion sort by depth, DESCENDING (farthest first → nearest
//      last) so the iteration order below is back-to-front.
//   4. Back-to-front blend with premultiplied-alpha "over":
//          accum_rgb = src.rgb + accum_rgb * (1 - src.a)
//          accum_a   = src.a   + accum_a   * (1 - src.a)
//      starting from (accum_rgb, accum_a) = (0, 0).  Output the final
//      (accum_rgb, accum_a) and let the pipeline's hardware blend put it
//      over the existing scene colour with SRC_ALPHA / ONE_MINUS_SRC_ALPHA.
//   5. Stamp gl_FragDepth = 0.99999 so the downstream sky envmap pass's
//      LESS_OR_EQUAL test against far-plane depth 1.0 fails at every
//      glass pixel.  Matches the existing oit_composite.frag trick.
//
// Overflow note: when more than K fragments hit a pixel, the translucent
// shader replaced the FARTHEST slot with the new fragment (so we always
// keep the K nearest).  Anything farther than the K'th nearest is lost.
// Visually this is barely noticeable — the lost fragments would have been
// the most occluded ones in the back-to-front chain.

// K must match ClusterRenderer::kOitPerPixelDepth in cluster_renderer.h.
// Hard-coding here for clarity; if you change one, change the other.
const int K = 4;

// Sentinel value oit_pp_clear.comp writes into every depth slot at the
// start of each frame.  A real fragment's gl_FragCoord.z ∈ [0,1]; the
// translucent shader uses the unified depth-compare-against-max
// algorithm where empty slots ARE the max (because sentinel > 1.0), so
// not-full lists work without a separate code path.  Resolve filters
// slots by `depth < sentinel` to pick up only the real entries.
const float OIT_PP_SENTINEL_DEPTH = 2.0;

// Image bindings.  These three are READ-ONLY here (the translucent shader
// writes them; this resolve only reads).
layout(set = 0, binding = 0, r32ui)         uniform uimage2D  u_pp_count;
layout(set = 0, binding = 1, rgba16f)       uniform image3D   u_pp_color;
layout(set = 0, binding = 2, r32f)          uniform image3D   u_pp_depth;

layout(location = 0) in  vec2 v_UV;  // unused; we use gl_FragCoord
layout(location = 0) out vec4 outColor;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);

    // ── 1. Bail out at pixels that had no glass fragments ───────────
    // The count image is advisory: the translucent shader bumps it
    // every time a fragment passes the depth-vs-max guard and writes a
    // slot.  Zero means no fragment ever wrote here, so we can early-
    // out without paying for the K imageLoads.  Non-zero just means
    // "something was inserted at some point" — the depth filter below
    // is what tells us which slots actually contain real data.
    uint count = imageLoad(u_pp_count, pixel).r;
    if (count == 0u) {
        discard;
    }

    // ── 2. Read all K slots, filtering by depth < sentinel ──────────
    // Slot order is irrelevant — the translucent shader replaces the
    // farthest slot, not the slot at any particular position.  So a
    // half-full list can have its real entries at any indices, with
    // sentinel-bearing slots interleaved.  We compact into a dense
    // [0..n) prefix here so the sort + blend below can use a single
    // bound.
    vec4  fr_color[K];
    float fr_depth[K];
    int   n = 0;
    for (int i = 0; i < K; ++i) {
        float d = imageLoad(u_pp_depth, ivec3(pixel, i)).r;
        if (d < OIT_PP_SENTINEL_DEPTH) {
            fr_color[n] = imageLoad(u_pp_color, ivec3(pixel, i));
            fr_depth[n] = d;
            ++n;
        }
    }
    // Second early-out: count was non-zero but every slot reads as
    // sentinel.  This shouldn't be reachable in practice (the
    // translucent shader bumps count only when it writes a slot, and
    // the slot write commits the real depth) but is here as a defensive
    // guard so we never feed an empty list into the back-to-front
    // blend below.
    if (n == 0) {
        discard;
    }

    // ── 3. Insertion sort by depth DESCENDING ───────────────────────
    // We only need to sort the first `n` entries.  Insertion sort is
    // optimal for very small n; for K=4 the inner loop is bounded by
    // 6 compare-swaps in the worst case (n=4) and compiles to a small
    // straight-line sequence after unrolling.
    //
    // Loop bound has to be a compile-time constant for GLSL unroll
    // (loops with a dynamic upper bound work too but produce more
    // varied codegen).  We loop to K and guard inside on (i < n) and
    // (j > 0 && fr_depth[j-1] < key_depth) to ignore unused slots.
    for (int i = 1; i < K; ++i) {
        if (i >= n) break;
        vec4  key_color = fr_color[i];
        float key_depth = fr_depth[i];
        int j = i - 1;
        // Shift entries whose depth is SMALLER than key (so the larger
        // one ends up first → descending order).  We want farthest at
        // index 0, nearest at index n-1.
        while (j >= 0 && fr_depth[j] < key_depth) {
            fr_color[j + 1] = fr_color[j];
            fr_depth[j + 1] = fr_depth[j];
            --j;
        }
        fr_color[j + 1] = key_color;
        fr_depth[j + 1] = key_depth;
    }

    // ── 4. Back-to-front blend with premultiplied alpha ─────────────
    // Slot layout after the sort: 0 = farthest, n-1 = nearest.  We
    // iterate front-to-back of the SORTED array, which is back-to-front
    // of the scene.  Each step does the standard premultiplied-alpha
    // "over" operator:
    //     dst = src + dst * (1 - src.a)
    // Starting from (0, 0) the result at the end is the fully-resolved
    // translucent layer in premultiplied form.
    vec4 accum = vec4(0.0);
    for (int i = 0; i < K; ++i) {
        if (i >= n) break;
        vec4 src = fr_color[i];        // src.rgb already premultiplied
        accum = src + accum * (1.0 - src.a);
    }

    // ── 5. Output + depth stamp ─────────────────────────────────────
    // The composite pipeline's blend is SRC_ALPHA / ONE_MINUS_SRC_ALPHA,
    // so we need to "un-premultiply" before emitting — or, equivalently,
    // emit (accum.rgb / accum.a, accum.a).  Guard against accum.a == 0
    // (which would only happen if every slot had α == 0, in which case
    // discard would actually be cleaner; we play safe instead of
    // discarding because at this point we've already paid for the sort).
    float a = max(accum.a, 1e-5);
    outColor = vec4(accum.rgb / a, accum.a);

    // Stamp depth just inside the far plane so sky envmap's
    // LESS_OR_EQUAL(0.99999, 1.0) → false at this pixel.
    gl_FragDepth = 0.99999;
}
