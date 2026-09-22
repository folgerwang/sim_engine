#ifndef LEAF_AGE_GLSL_H
#define LEAF_AGE_GLSL_H
#include "debug_toggles.glsl.h"
#include "tree_age_palette_data.glsl.h"
#include "plant_season.glsl.h"
// Historical function/buffer names are retained for asset and shader ABI compatibility.
// THE SILHOUETTE, NOISE-TOLERANT.  The albedo alpha is the leaf shape
// times a root-to-tip ramp whose floor is .1, and the old test was
// clamp(a*10): anything at or above .1 is the leaf, fine -- but it also
// turned alpha NOISE of 3/255 into "visible".  The cards are BC7 in the
// VT pool, and mode 6 shares ONE index between colour and alpha, so a
// block where colour and alpha do not move together (any leaf edge,
// and every transparent texel once its colour is bled from the ink)
// carries exactly that noise.  Measured on the needle spray: 12% of
// transparent texels passed the cutout with the old black-backed bake,
// 54% with the bled one.  A ramp from .07 to .11 keeps the tip (.1 ->
// .75, well over the .09 cutoff) and drops everything a block encode
// can smear under it: 1% leak, 99.5% of the ink kept.
float leafSilhouette(float a) { return smoothstep(.07, .11, a); }
vec4 leafAgedColor(vec4 color, uint group, float seasonTime, uint profile, float timingSeed) {
#if DEBUG_NO_LEAF_AGE
    return color;   // diagnostic: age/season fadeout disabled
#else
    profile=min(profile,TREE_PALETTE_COUNT-1u);
    uint mode=treePalette_seasonMode[profile];
    float phase=plantSeasonPhase(seasonTime,timingSeed,group);
    if(mode==2u) return vec4(grassSeasonColor(color.rgb,seasonTime),color.a);
    if(mode==1u) {
        // Evergreen needles stay attached, but still follow the year:
        // fresh spring green, summer green, subtle species-specific autumn
        // bronzing and cooler winter foliage. Alpha is never changed.
        float spring=smoothstep(0.,.08,phase)*(1.-smoothstep(.16,.30,phase));
        float autumn=smoothstep(.40,.55,phase)*(1.-smoothstep(.72,.85,phase));
        vec3 tinted=color.rgb*mix(vec3(1),vec3(.85,.92,.88),plantSeasonDormancy(phase));
        tinted*=mix(vec3(1),vec3(1.04,1.12,.88),spring);
        vec3 target=treePalette_aging[profile];
        target*=dot(color.rgb,vec3(.2126,.7152,.0722))/max(dot(target,vec3(.2126,.7152,.0722)),.01);
        // (the root-to-tip ramp is taken out of it, as below: at a mip where
        // leaf and gap average together the tip's .1 fell under the cutoff
        // and every needle spray lost its outer half)
        return vec4(mix(tinted,target,.12*autumn),leafSilhouette(color.a));
    }
    // WHOLE LEAVES, NOT STUBS.  Albedo alpha is silhouette x a root-to-tip
    // ramp (root 1, tip .1).  The old test, max(a+coverage-1,0), slid ONE
    // threshold along every leaf of the tree at once, so for the quarter
    // of the year that coverage is partial every leaf was cut to the same
    // stub -- rows of identical stubs along a twig are the "comb".  (A
    // first repair unrolled each leaf from its root during the spring;
    // that is the same comb for a few weeks and it is gone too.)
    //   silhouette  alpha*10: the tip's .1 already saturates, so the ramp
    //               no longer leaks into the cutout edge;
    //   a leaf is either THERE or NOT: its cohort (the group -- per
    //   texel under LEAF_MASK, per material otherwise) and the tree's
    //   own seed give it a slot in [0,1); it appears when the spring
    //   has reached that slot and drops when the autumn has.
    float sil=leafSilhouette(color.a);
    float vis=1.;
    // v54: cohort 0 is a cohort like the others.  Stage 0 is the BABY
    // leaves at every branch tip (leaf_growth_stage: youngest at the
    // tip, oldest at the base), and the old "group 0 never drops" rule
    // -- meant for ground clutter, which the mode==2 return above
    // already handles -- kept an eighth of every deciduous crown, the
    // tips, in leaf all winter and fully grown before spring began.
    {
        float slot=fract(float(group)/8.*.875+fract(abs(timingSeed)*.37));
        float sp=clamp((phase-.02)/.14,0.,1.);
        float fall=clamp((phase-.62)/.12,0.,1.);
        // v53: SMOOTH, NOT A SWITCH.  Each cohort GROWS over its own
        // window (35% of the spring) and WITHERS over its own window
        // (30% of the autumn), both staggered by slot.  Growth unrolls
        // the leaf from its root: the albedo's root-to-tip ramp (root 1,
        // tip .1) says how far along the leaf a texel lies, so keeping
        // texels up to the grown fraction is a leaf getting longer, not a
        // leaf popping in.  Withering runs the same ramp back from the
        // tip.  The stubs-on-every-leaf comb of the very first version
        // came from ALL cohorts growing in lockstep; the slots spread
        // that out, and the cohort/seed offsets keep neighbours apart.
        // v55: the ramp is now a SCALE metric (terrain_pcg leaflet():
        // the texel's alpha says which root-anchored, same-shaped copy
        // of the leaf first contains it), so "keep the texels with
        // ramp >= 1-grow" is the leaf drawn at `grow` of its full size,
        // not the leaf with its tip sawn off at a straight line.
        // Withering never cuts: a leaf goes brown and DROPS WHOLE, at a
        // moment of its own (slot + seed jitter) inside the autumn window,
        // so the crown thins card by card instead of every leaf being
        // trimmed shorter in lockstep.
        float ramp=clamp((color.a-.1)/.9,0.,1.);           // 1 root .. 0 tip
        float grow=clamp((sp-slot*.6)/.35,0.,1.);
        float dropAt=slot*.7+.3*fract(abs(timingSeed)*11.7+float(group)*.61);
        if(ramp<1.-grow) vis=0.;
        if(fall>0. && fall>=dropAt) vis=0.;
    }
    // Each year's newborn starts at baby color, then reaches its cohort's
    // authored initial color; autumn coloring starts only after maturation.
    float babyToInitial=smoothstep(.02,.25,phase);
    vec3 fresh=mix(color.rgb*vec3(.85,1.10,.72),color.rgb,babyToInitial);
    vec3 target=mix(treePalette_aging[profile],treePalette_dry[profile],smoothstep(.60,.74,phase));
    float detail=dot(color.rgb,vec3(.2126,.7152,.0722));
    target*=detail/max(dot(target,vec3(.2126,.7152,.0722)),.01);
    return vec4(mix(fresh,target,smoothstep(.45,.62,phase)),sil*vis);
#endif
}
vec4 leafAgedColor(vec4 c,uint g,float t,uint p) {return leafAgedColor(c,g,t,p,0.);}
vec4 leafAgedColor(vec4 c,uint g,float t) {return leafAgedColor(c,g,t,0u,0.);}
uint leafMaskGroup(float code) {return min(uint(clamp(code,0.,1.)*8.),7u);}
float leafMaskCoverage(float code,float t) {return leafMaskGroup(code)==0u ? 1. : plantSeasonCoverage(plantSeasonPhase(t,0.,leafMaskGroup(code)));}
#endif
