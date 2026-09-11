#ifndef LEAF_AGE_GLSL_H
#define LEAF_AGE_GLSL_H
#include "tree_age_palette_data.glsl.h"
#include "plant_season.glsl.h"
// Historical function/buffer names are retained for asset and shader ABI compatibility.
vec4 leafAgedColor(vec4 color, uint group, float seasonTime, uint profile, float timingSeed) {
    profile=min(profile,TREE_PALETTE_COUNT-1u);
    uint mode=treePalette_seasonMode[profile];
    float phase=plantSeasonPhase(seasonTime,timingSeed,group);
    if(mode==2u) return vec4(grassSeasonColor(color.rgb,seasonTime),color.a);
    if(mode==1u) return vec4(color.rgb*mix(vec3(1),vec3(.85,.92,.88),plantSeasonDormancy(phase)),color.a);
    float coverage=plantSeasonCoverage(phase);
    // Each year's newborn starts at baby color, then reaches its cohort's
    // authored initial color; autumn coloring starts only after maturation.
    float babyToInitial=smoothstep(.02,.25,phase);
    vec3 fresh=mix(color.rgb*vec3(.85,1.10,.72),color.rgb,babyToInitial);
    vec3 target=mix(treePalette_aging[profile],treePalette_dry[profile],smoothstep(.60,.74,phase));
    float detail=dot(color.rgb,vec3(.2126,.7152,.0722));
    target*=detail/max(dot(target,vec3(.2126,.7152,.0722)),.01);
    return vec4(mix(fresh,target,smoothstep(.45,.62,phase)),max(color.a+coverage-1.,0.));
}
vec4 leafAgedColor(vec4 c,uint g,float t,uint p) {return leafAgedColor(c,g,t,p,0.);}
vec4 leafAgedColor(vec4 c,uint g,float t) {return leafAgedColor(c,g,t,0u,0.);}
uint leafMaskGroup(float code) {return min(uint(clamp(code,0.,1.)*4.),3u);}
float leafMaskCoverage(float code,float t) {return plantSeasonCoverage(plantSeasonPhase(t,0.,leafMaskGroup(code)));}
#endif
