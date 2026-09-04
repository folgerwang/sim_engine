#ifndef TONEMAP_GLSL_H
#define TONEMAP_GLSL_H

// ── Scene tonemap ──────────────────────────────────────────────────────
// ONE shared exposure + filmic curve for every final-colour writer:
// deferred_resolve.comp, cluster_bindless.frag (forward), base.frag's
// default toneMap() path, terrain tile.frag / tile_water.frag and the
// two skybox shaders.  They all render into the same LDR target and the
// deferred resolve leaves forward pixels untouched, so any writer using
// a different curve shows up as a visible seam between deferred and
// forward content.
//
// Previously each writer did a bare linearTosRGB(): HDR radiance above
// 1.0 (sun-lit ground, the whole sky) simply clipped, which is why the
// frame read as washed-out white.  ACES pins mid-gray almost exactly
// (0.18 in → ~0.18 out at exposure 1) while rolling highlights off to a
// filmic shoulder, so lowering exposure here darkens the blown range
// without crushing the shadows.
//
// Standalone header (rather than functions.glsl.h) because the sky
// shaders can't include functions.glsl.h — its rsi() collides with
// sunlight_scattering.glsl.h's.  functions.glsl.h includes this, so
// every existing includer sees sceneTonemap unchanged.
//
// NOT for texture/IBL prebake passes (cube_skybox.frag, cube_ibl.frag,
// sky LUTs…) — those must stay linear; they are inputs to lighting, not
// display output.
// 0.58 (was 0.75): with near-white ground albedo the noon sum of sun +
// full-sky IBL still pushed most of the frame past the ACES shoulder at
// 0.75 — shadow terms and surface detail read as one blown white.  ACES
// pins mid-gray, so this darkens the blown range ~25% while shadows
// keep their footing.  ONE shared constant for every final-colour
// writer — tune here, never per-shader.
const float kSceneExposure = 0.58;

// ACES filmic fit (Narkowicz 2016).
vec3 sceneAcesFilm(vec3 x)
{
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
}

vec3 sceneTonemap(vec3 hdr)
{
    // 1/2.2 gamma matches functions.glsl.h's linearTosRGB exactly.
    return pow(sceneAcesFilm(hdr * kSceneExposure), vec3(1.0 / 2.2));
}

// ── Physical-camera exposure ────────────────────────────────────────────
// The same curve with the per-frame exposure scale the camera UBO
// carries (ViewCameraInfo::exposure_scale — shutter / f-stop / ISO from
// the Camera & Lens panel, relative to the calibrated default).  Every
// final-colour writer that binds the camera buffer passes
// sceneExposureScaleOf(camera_info.exposure_scale); a writer without
// the camera buffer keeps sceneTonemap() (scale 1).
float sceneExposureScaleOf(float v)
{
    return (v > 0.0) ? v : 1.0;
}

vec3 sceneTonemapExposed(vec3 hdr, float exposure_scale)
{
    return pow(sceneAcesFilm(hdr * (kSceneExposure * exposure_scale)),
               vec3(1.0 / 2.2));
}

#endif  // TONEMAP_GLSL_H
