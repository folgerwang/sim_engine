#ifdef GBUFFER_OUTPUT
layout(location = 0) out vec4 out_albedo_ao;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec4 out_emissive_metal;
layout(location = 3) out vec2 out_velocity;
vec2 citizenOct(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    vec2 e = n.xy;
    if (n.z < 0.0) e = (1.0 - abs(e.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(e, vec2(0.0)));
    return e * 0.5 + 0.5;
}
void citizenGbuffer(vec3 albedo, vec3 normal_ws, vec3 position_ws,
                    mat4 current_vp, mat4 previous_vp, float roughness) {
    vec2 n = citizenOct(normalize(normal_ws));
    out_albedo_ao = vec4(albedo, 1.0);
    out_normal_rough = vec4(n, roughness, 0.0);
    out_emissive_metal = vec4(n, 0.0, 0.0);
    vec4 cur = current_vp * vec4(position_ws, 1.0);
    vec4 prev = previous_vp * vec4(position_ws, 1.0);
    out_velocity = cur.xy / cur.w - prev.xy / prev.w;
}
#else
layout(location = 0) out vec4 outColor;
#endif
