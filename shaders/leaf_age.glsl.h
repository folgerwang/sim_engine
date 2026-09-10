#ifndef LEAF_AGE_GLSL_H
#define LEAF_AGE_GLSL_H
// Initial group ages are 1..50; the application's shared float is 0..100.
float leafGroupAge(uint group, float globalAge) {
    const float initialAge[4] = float[4](1., 17., 34., 50.);
    return initialAge[min(group, 3u)] + clamp(globalAge, 0., 100.);
}
vec4 leafAgedColor(vec4 color, uint group, float globalAge) {
    float age = leafGroupAge(group, globalAge);
    if (age >= 100.) return vec4(0);
    vec3 fresh = color.rgb * mix(vec3(.85,1.10,.72), vec3(1), smoothstep(1.,25.,age));
    const vec3 dying[3] = vec3[3](vec3(.80,.55,.08), vec3(.62,.12,.06), vec3(.35));
    vec3 hue = dying[group % 3u];
    float luminance = dot(color.rgb, vec3(.2126,.7152,.0722));
    vec3 dry = hue * (luminance * .75 / max(dot(hue,vec3(.2126,.7152,.0722)),.01));
    return vec4(mix(fresh, dry, smoothstep(55.,100.,age)), color.a);
}
uint leafMaskGroup(float code) {
    return min(uint(clamp(code, 0., 1.) * 4.), 3u);
}
float leafMaskCoverage(float code, float globalAge) {
    return leafGroupAge(leafMaskGroup(code), globalAge) < 100. ? 1. : 0.;
}
#endif
