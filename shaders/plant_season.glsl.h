#ifndef PLANT_SEASON_GLSL_H
#define PLANT_SEASON_GLSL_H
// One common year: spring 0, summer .25, autumn .5, winter .75.
float plantSeasonPhase(float time, float seed, uint group) {
    float offset = clamp(-seed,0.,100.)*.0003 + float(min(group,3u))*.012;
    return fract(time*.01-offset);
}
float plantSeasonDormancy(float phase) {
    return smoothstep(.45,.72,phase)*(1.-smoothstep(.90,1.,phase));
}
float plantSeasonCoverage(float phase) {
    return smoothstep(.02,.16,phase)*(1.-smoothstep(.62,.74,phase));
}
vec3 grassSeasonColor(vec3 color, float time) {
    float phase=plantSeasonPhase(time,0.,0u);
    float luma=dot(color,vec3(.2126,.7152,.0722));
    vec3 straw=vec3(.300,.232,.082);
    straw *= luma/max(dot(straw,vec3(.2126,.7152,.0722)),.01);
    return mix(color,straw,.85*plantSeasonDormancy(phase));
}
#endif
