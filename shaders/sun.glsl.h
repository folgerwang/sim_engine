#if RENDER_SUN

#ifndef _SUN_GLSL_H_

#define _SUN_GLSL_H_

const vec3 kSunColor = vec3(0.905, 0.772, 0.368);

vec3 renderSun(vec3 eye_dir, vec3 sun_pos) {
  vec3 rd = eye_dir;
  float sun_amount = max(dot(rd, sun_pos), 0.0);
  vec3 sky = kSunColor * min(pow(sun_amount, 9500.0) * 4.0, 1.0);
  return sky;
}

#endif // _SUN_GLSL_H_

#endif // RENDER_SUN
