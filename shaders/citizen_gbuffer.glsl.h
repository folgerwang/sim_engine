#ifdef GBUFFER_OUTPUT
layout(location = 0) out vec4 out_albedo_ao;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec4 out_emissive_metal;
layout(location = 3) out vec2 out_velocity;
// RT4 — world-space FORWARD motion (frame N -> N+1), metres.  Written by
// the dynamic actors only (citizens, NPCs, vehicles); static geometry
// leaves the clear value (0) and the depth predictor moves it by camera
// alone.  See depth_predict.comp.
layout(location = 4) out vec4 out_motion3d;
vec2 citizenOct(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    vec2 e = n.xy;
    if (n.z < 0.0) e = (1.0 - abs(e.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(e, vec2(0.0)));
    return e * 0.5 + 0.5;
}
// FRAME-AHEAD CONTRACT.  The simulation is committed one frame ahead of
// the picture: the instance record carries the pose for frame N+1 and the
// pose for frame N (the one being drawn).  position_ws is the DRAWN (N)
// world position, position_next_ws the N+1 one, position_prev_ws the N-1
// one.  Screen velocity is the true N-1 -> N delta (TAA / motion blur),
// motion3d the N -> N+1 world delta (depth prediction).
void citizenGbufferMotion(vec3 albedo, vec3 normal_ws,
                          vec3 position_ws, vec3 position_prev_ws,
                          vec3 position_next_ws,
                          mat4 current_vp, mat4 previous_vp, float roughness) {
    vec2 n = citizenOct(normalize(normal_ws));
    out_albedo_ao = vec4(albedo, 1.0);
    out_normal_rough = vec4(n, roughness, 0.0);
    out_emissive_metal = vec4(n, 0.0, 0.0);
    vec4 cur = current_vp * vec4(position_ws, 1.0);
    vec4 prev = previous_vp * vec4(position_prev_ws, 1.0);
    out_velocity = cur.xy / cur.w - prev.xy / prev.w;
    out_motion3d = vec4(position_next_ws - position_ws, 1.0);
}
// Legacy entry (camera-only velocity, no forward motion): kept for any
// caller that has no pose history.
void citizenGbuffer(vec3 albedo, vec3 normal_ws, vec3 position_ws,
                    mat4 current_vp, mat4 previous_vp, float roughness) {
    citizenGbufferMotion(albedo, normal_ws, position_ws, position_ws,
                         position_ws, current_vp, previous_vp, roughness);
    out_motion3d = vec4(0.0);
}
#else
layout(location = 0) out vec4 outColor;
#endif
