#pragma once
// ── THE BAKED CAR LIBRARY ───────────────────────────────────────────
// Reads "cars.rwcar", the file tools/terrain/car_gen.py bakes through
// the "cars" stage.  Cars used to be generated in C++ at engine start
// by lofting a side profile (VehicleSystem::buildHull), with every
// feature a car actually has — grille, lamps, mirrors, handles, glass,
// rims — PAINTED on that shell by vehicle.frag from the fragment's
// position.  Paint cannot make a silhouette, and the shader had to keep
// a duplicate copy of every dimension so it could guess where the
// features belonged.
//
// The geometry is baked offline now, and each triangle carries a PART
// ID.  That id is the entire interface: the shader asks what a fragment
// IS rather than deducing it, so a wing mirror is geometry rather than
// a rule, and the duplicated tables have nothing left to do.
//
// This header deliberately knows nothing about the renderer — it is
// parsing and plain arrays, so it can be unit-tested and so a malformed
// file fails HERE rather than somewhere inside a pipeline.  When it
// fails, VehicleSystem falls back to buildHull and the world still has
// cars in it; that is why every failure below returns false quietly
// rather than throwing.
#include <cstdint>
#include <string>
#include <vector>

#include "glm/glm.hpp"

namespace engine {
namespace game_object {

// KEEP IN LOCKSTEP with car_gen.py's kPart* and vehicle.frag's.
enum CarPart : uint32_t {
    kCarPartBody = 0, kCarPartGlass, kCarPartTrim, kCarPartChrome,
    kCarPartLampFront, kCarPartLampRear, kCarPartGrille, kCarPartTyre,
    kCarPartRim, kCarPartInterior, kCarPartLivery, kCarPartLightBar,
    kCarPartCount
};

// A paint is a basecoat under a clear coat; see car_gen's palette note.
struct CarPaint {
    std::string name;
    glm::vec3   base = glm::vec3(0.8f);
    float       metal = 0.0f;    // aluminium-flake basecoat
    float       flake = 0.0f;    // off-axis sparkle
    float       coat = 0.9f;     // clear coat's specular strength
    glm::vec3   tint = glm::vec3(0.8f);   // pearl shift
};

struct CarInterior {
    std::string name;
    glm::vec3   rgb = glm::vec3(0.1f);
    float       roughness = 0.9f;
};

struct CarMesh {
    std::vector<glm::vec3> pos, nrm;
    std::vector<float>     part;     // one part id per vertex
    std::vector<uint32_t>  idx;
    bool empty() const { return idx.empty(); }
};

struct CarSample {
    std::string           name;
    float                 L = 4.0f, W = 1.8f, H = 1.5f;
    float                 wheel_r = 0.32f, wheel_w = 0.2f, track_z = 0.75f;
    std::vector<float>    axle_x, axle_dual;
    float                 vmax = 14.0f;
    bool                  bar = false;
    glm::vec4             bar_at_size = glm::vec4(0.0f);  // x, y, len, width
    std::vector<uint32_t> paints;    // indices into CarLibrary::paints
    CarMesh               body, wheel, interior;
};

struct CarLibrary {
    std::vector<CarPaint>    paints;
    std::vector<CarInterior> interiors;
    std::vector<CarSample>   samples;
    bool loaded = false;

    // Every sample whose name starts with `prefix` ("saloon_", "police").
    std::vector<int> byPrefix(const std::string& prefix) const;
};

// Returns false (and leaves `out` untouched) on any problem at all: the
// file missing, a bad magic, a version this build does not know, a
// truncated stream.  The caller falls back to the built-in hulls.
bool loadCarLibrary(const std::string& path, CarLibrary& out);

}  // namespace game_object
}  // namespace engine
