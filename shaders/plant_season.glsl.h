#ifndef PLANT_SEASON_GLSL_H
#define PLANT_SEASON_GLSL_H
// One common year: spring 0, summer .25, autumn .5, winter .75.
float plantSeasonPhase(float time, float seed, uint group) {
    float offset = clamp(-seed,0.,100.)*.0003 + float(min(group,7u))*(.036/7.);
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
    // Shared by terrain blades, meadow clumps and yard turf. Colours are
    // linear, with continuous transitions including winter -> spring.
    const vec3 palette[4] = vec3[4](
        vec3(.120,.240,.045), // spring: fresh green
        vec3(.095,.165,.038), // summer: mature green
        vec3(.300,.232,.082), // autumn: golden dry grass
        vec3(.190,.175,.130)  // winter: muted dormant turf
    );
    float quarter = phase * 4.0;
    int season = int(floor(quarter));
    vec3 tint = mix(palette[season], palette[(season + 1) % 4],
                    smoothstep(0.0, 1.0, fract(quarter)));
    tint *= luma / max(dot(tint,vec3(.2126,.7152,.0722)),.01);
    return mix(color,tint,.85);
}
// Garden turf reserves grass cohort 7. World coordinates prevent the metre
// scale variation and snow from repeating with the small blade texture.
float gardenTurfNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    vec4 h = fract(sin(vec4(dot(i,vec2(127.1,311.7)),
        dot(i+vec2(1,0),vec2(127.1,311.7)),
        dot(i+vec2(0,1),vec2(127.1,311.7)),
        dot(i+vec2(1,1),vec2(127.1,311.7))))*43758.5453);
    return mix(mix(h.x,h.y,f.x),mix(h.z,h.w,f.x),f.y);
}
vec3 gardenTurfSurface(vec3 color, vec2 worldXZ, float time) {
    float broad = gardenTurfNoise(worldXZ*.085);
    float nap = sin(dot(worldXZ,vec2(.8,.6))*1.8 + broad*.4);
    color *= .96 + .06*broad + .025*nap;
    float phase = plantSeasonPhase(time,0.,0u);
    float winter = smoothstep(.58,.74,phase)*(1.-smoothstep(.84,1.,phase));
    float patches = .65*gardenTurfNoise(worldXZ*1.3)
                  + .35*gardenTurfNoise(worldXZ*4.7+vec2(31.7,9.2));
    float snow = smoothstep(.43,.62,patches) * winter;
    // Patchy cover leaves dormant blades visible as in the winter reference.
    return mix(color,vec3(.78,.81,.84),snow);
}
#endif
