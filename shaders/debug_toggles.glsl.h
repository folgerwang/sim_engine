#ifndef DEBUG_TOGGLES_GLSL_H
#define DEBUG_TOGGLES_GLSL_H
// ── Diagnostic kill-switches (compile time) ──────────────────────────
// Flip these to 1, rebuild (the shaders are compiled by CMake), and the
// named effect is removed engine-wide.  They exist to ISOLATE the source
// of the dithered / screen-door speckle seen on plants: each switch
// removes exactly one thing that can produce a per-pixel pattern.
//
//   DEBUG_NO_ALPHA_CUTOFF  every ALPHA_MASK / cutout test is defeated.
//                          Nothing is discarded for low alpha anywhere
//                          (forward, G-buffer, depth prepass, shadow
//                          depth, cluster bindless, RT any-hit in
//                          deferred_resolve): leaf and grass cards draw
//                          as whole opaque quads.  If speckle survives
//                          this, it is NOT the cutout.
//
//   DEBUG_NO_LEAF_AGE      leafAgedColor() is a pass-through: no season
//                          tint, no cohort growth window, no autumn
//                          drop, no alpha*10 silhouette rescale.  This
//                          is the "age fadeout".
//
//   DEBUG_NO_LOD_DITHER    the LOD band cross-fade screen door
//                          (lodFadeDiscards) and the decal/clutter
//                          distance screen door (decalScreenDoorCull)
//                          both stop discarding.  Left at 0 on purpose:
//                          with the two above off, if the dither is
//                          still there, this is the one that owns it —
//                          set it to 1 to confirm.
#define DEBUG_NO_ALPHA_CUTOFF 0
#define DEBUG_NO_LEAF_AGE     0
#define DEBUG_NO_LOD_DITHER   0

// A cutoff of -1 defeats the test in BOTH senses used in the codebase:
//   a <  cutoff  -> never true  -> nothing discarded
//   a >= cutoff  -> always true -> the texel counts as solid
#if DEBUG_NO_ALPHA_CUTOFF
#define DBG_CUTOFF(x) (-1.0)
#else
#define DBG_CUTOFF(x) (x)
#endif
#endif // DEBUG_TOGGLES_GLSL_H
