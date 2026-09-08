#include "vehicle_system.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>

#include <glm/gtc/matrix_transform.hpp>

#include "json.hpp"   // vendored at third_parties/tinygltf/json.hpp
#include "helper/engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace er = engine::renderer;

namespace engine {
namespace game_object {
namespace {

// ── tuning ───────────────────────────────────────────────────────────
constexpr float kSimRadius = 1600.0f;      // vehicles ticked, per frame
constexpr float kDrawRadius = 1400.0f;     // vehicles drawn
constexpr float kClampRadius = 450.0f;     // exact ground query
constexpr float kCurbSearchM = 170.0f;     // how far a car may be from a road
constexpr float kVisualCapMs = 22.0f;      // fastest a car ever LOOKS
constexpr float kCarLenGap = 4.5f;         // bumper-to-bumper allowance
constexpr float kAccel = 2.6f, kBrake = 6.5f;
// Ambient traffic lives in a ring around the camera: spawned 45-620 m
// out (260-900 m put most of the forty cars out of frame on a road
// network that is 16% of the map's houses within 120 m of a road),
// recycled once 1.1 km away, destinations 250-1500 m off so a car
// stays in the neighbourhood it was spawned for.
constexpr float kSpawnMin = 45.0f, kSpawnMax = 620.0f;
constexpr float kRecycleM = 1100.0f;
constexpr float kDestMin = 250.0f, kDestMax = 1500.0f;
constexpr float kParkedShare = 0.38f;      // ambient cars kerbed on spawn
constexpr float kVergeM = 2.25f;           // kerbed: this far past the curb spot
// ── v34: lanes and junction control ──────────────────────────────────
// Two lanes EACH WAY needs a genuine arterial.  At 4.4 m every paved
// street in the world qualified (they are all ~4.95 m half-width) and
// got four lanes on 8.2 m of pavement; a residential street is two
// lanes, one each way, which is what laneOffset gives below that.
constexpr float kTwoLaneHalfM = 7.0f;      // half-width from which a road has 2 lanes a side
constexpr float kStopLineM = 5.0f;         // the stop line, before the node
constexpr float kStopWaitS = 1.0f;         // a stop sign: stand this long
constexpr float kCycleS = 36.0f;           // lights: A 0-14 green 14-17 yellow, B 18-32 green 32-35 yellow
constexpr float kSignalDrawM = 500.0f;     // signals drawn inside this
constexpr float kClaimHoldS = 7.0f;        // a node claim expires after this
constexpr float kYieldS = 9.0f;            // held this long at a junction: go
// ── WHERE THE PAVEMENT IS (v36) ─────────────────────────────────────
// The road ribbon is not all road: its outer part is shoulder, blended
// into the terrain, and only the inner fraction is surfaced and
// painted.  These are the SAME fractions terrain_pcg's road recipe
// paints to, picked by the same half-width thresholds its mesher sorts
// the surface classes by (under 1.5 m dirt track, under 2.5 m gravel,
// wider asphalt) -- so a lane centre here is a lane centre there, and
// the markings land between the cars instead of under them.
//
// This is what the v34 two-lane change got wrong: its outer lane sat
// at 0.72 of the HALF-WIDTH, which on every one of these roads is out
// on the verge.
float carriageFrac(float h) {
    return h < 1.5f ? 0.50f : (h < 2.5f ? 0.62f : 0.82f);
}

// Lateral offset of a lane's centre from the road's centre line.  One
// lane each way on a narrow road, centred in its half of the pavement;
// on a wide one two, at a quarter and three quarters of it -- either
// side of the dashed lane line the texture paints at its midpoint.
float laneOffset(float h, float lane) {
    const float pave = carriageFrac(h) * h;      // pavement half-width
    if (h < kTwoLaneHalfM) return 0.5f * pave;
    return pave * (0.25f + 0.50f * glm::clamp(lane, 0.0f, 1.0f));
}
bool twoLane(float h) { return h >= kTwoLaneHalfM; }
constexpr float kSpawnGapM = 8.0f;         // never spawn onto another car
constexpr float kRoadPerCarM = 22.0f;      // ambient cars: one per this much road
constexpr int   kAmbientMin = 12;          // ...but never fewer, where any road is
constexpr float kPtCell = 32.0f;           // road-point hash cell
constexpr float kNodeCell = 6.0f;
constexpr float kNodeMerge = 2.5f;
constexpr float kTeeSnapM = 9.0f;

// ── a vehicle type ───────────────────────────────────────────────────
struct Spec {
    float L, W;
    std::vector<glm::vec2> profile;     // side profile, metres (x fwd, y up)
    float tumble;                       // width lost at the roof (fraction)
    float belt, roof;
    float wheel_r, wheel_w, wheel_z;
    std::vector<float> wheel_x;
    std::vector<float> wheel_dual;      // width factor per axle
    float vmax;                         // m/s
    bool  bar;
    glm::vec3 bar_at;                   // light bar centre (x, y)
    glm::vec2 bar_size;                 // (length, width)
};

const std::vector<Spec>& specs() {
    static std::vector<Spec> s;
    if (!s.empty()) return s;
    s.resize(VehicleSystem::kTypeCount);
    // Acura-style sedan
    s[VehicleSystem::kSedan] = {
        4.80f, 1.85f,
        {{-2.40f, 0.32f}, {-2.40f, 0.78f}, {-2.25f, 0.96f}, {-1.60f, 1.00f},
         {-0.85f, 1.38f}, {0.40f, 1.42f}, {1.15f, 1.04f}, {2.10f, 0.90f},
         {2.40f, 0.76f}, {2.40f, 0.34f}, {2.05f, 0.20f}, {-2.05f, 0.20f}},
        0.10f, 1.00f, 1.38f, 0.34f, 0.22f, 0.78f, {1.40f, -1.40f},
        {1.0f, 1.0f}, 14.0f, false, {0, 0, 0}, {0, 0}};
    // compact SUV (XC40-style)
    s[VehicleSystem::kSuv] = {
        4.45f, 1.86f,
        {{-2.22f, 0.40f}, {-2.22f, 1.00f}, {-2.10f, 1.25f}, {-1.95f, 1.60f},
         {-0.95f, 1.65f}, {0.45f, 1.65f}, {1.10f, 1.22f}, {1.95f, 1.06f},
         {2.22f, 0.88f}, {2.22f, 0.40f}, {1.90f, 0.26f}, {-1.90f, 0.26f}},
        0.12f, 1.08f, 1.60f, 0.36f, 0.24f, 0.78f, {1.33f, -1.33f},
        {1.0f, 1.0f}, 14.0f, false, {0, 0, 0}, {0, 0}};
    // the angular stainless pickup
    s[VehicleSystem::kCyber] = {
        5.70f, 2.00f,
        {{-2.85f, 0.42f}, {-2.85f, 1.12f}, {-2.30f, 1.28f}, {0.05f, 1.80f},
         {1.75f, 1.22f}, {2.85f, 0.98f}, {2.85f, 0.42f}, {2.45f, 0.30f},
         {-2.45f, 0.30f}},
        0.16f, 1.14f, 1.74f, 0.44f, 0.30f, 0.86f, {1.72f, -1.72f},
        {1.0f, 1.0f}, 14.0f, false, {0, 0, 0}, {0, 0}};
    // F-150-style pickup
    s[VehicleSystem::kPickup] = {
        5.90f, 2.05f,
        {{-2.95f, 0.42f}, {-2.95f, 1.30f}, {-0.55f, 1.30f}, {-0.55f, 1.95f},
         {0.60f, 1.95f}, {1.10f, 1.46f}, {2.60f, 1.26f}, {2.95f, 1.06f},
         {2.95f, 0.42f}, {2.50f, 0.32f}, {-2.50f, 0.32f}},
        0.08f, 1.44f, 1.90f, 0.43f, 0.28f, 0.86f, {1.85f, -1.65f},
        {1.0f, 1.0f}, 14.0f, false, {0, 0, 0}, {0, 0}};
    // semi tractor + trailer (one rigid body)
    s[VehicleSystem::kSemi] = {
        14.6f, 2.55f,
        {{-11.0f, 1.20f}, {-11.0f, 4.05f}, {-1.70f, 4.05f}, {-1.70f, 3.55f},
         {1.55f, 3.55f}, {1.75f, 2.95f}, {1.75f, 1.95f}, {3.40f, 1.85f},
         {3.60f, 1.05f}, {3.60f, 0.55f}, {3.10f, 0.45f}, {-10.6f, 0.45f}},
        0.02f, 2.35f, 3.45f, 0.52f, 0.32f, 1.10f,
        {2.60f, -0.40f, -1.70f, -8.00f, -9.30f},
        {1.0f, 1.9f, 1.9f, 1.9f, 1.9f}, 11.0f, false, {0, 0, 0}, {0, 0}};
    // police SUV
    s[VehicleSystem::kPolice] = {
        5.00f, 2.00f,
        {{-2.50f, 0.42f}, {-2.50f, 1.05f}, {-2.35f, 1.30f}, {-2.20f, 1.72f},
         {-1.00f, 1.76f}, {0.55f, 1.76f}, {1.25f, 1.30f}, {2.20f, 1.12f},
         {2.50f, 0.92f}, {2.50f, 0.42f}, {2.10f, 0.28f}, {-2.10f, 0.28f}},
        0.12f, 1.12f, 1.72f, 0.38f, 0.25f, 0.84f, {1.50f, -1.50f},
        {1.0f, 1.0f}, 15.0f, true, {0.0f, 1.84f, 0.0f}, {1.20f, 0.28f}};
    // ambulance van
    s[VehicleSystem::kAmbulance] = {
        5.30f, 1.95f,
        {{-2.65f, 0.45f}, {-2.65f, 2.32f}, {0.65f, 2.36f}, {1.30f, 1.78f},
         {2.05f, 1.18f}, {2.65f, 0.98f}, {2.65f, 0.45f}, {2.30f, 0.32f},
         {-2.30f, 0.32f}},
        0.04f, 1.38f, 2.24f, 0.37f, 0.24f, 0.82f, {1.75f, -1.55f},
        {1.0f, 1.0f}, 15.0f, true, {0.40f, 2.44f, 0.0f}, {1.30f, 0.30f}};
    // school bus
    s[VehicleSystem::kSchoolBus] = {
        10.5f, 2.50f,
        {{-5.25f, 0.55f}, {-5.25f, 3.10f}, {3.05f, 3.10f}, {3.45f, 2.40f},
         {3.55f, 1.72f}, {5.25f, 1.62f}, {5.25f, 0.55f}, {4.90f, 0.42f},
         {-4.90f, 0.42f}},
        0.05f, 1.78f, 2.72f, 0.52f, 0.35f, 1.05f, {3.30f, -2.90f},
        {1.0f, 1.8f}, 10.0f, false, {0, 0, 0}, {0, 0}};
    // fire engine (cab-over pumper)
    s[VehicleSystem::kFireEngine] = {
        9.50f, 2.50f,
        {{-4.75f, 0.55f}, {-4.75f, 3.25f}, {0.55f, 3.25f}, {0.75f, 3.40f},
         {3.05f, 3.40f}, {3.45f, 2.55f}, {4.75f, 2.25f}, {4.75f, 0.55f},
         {4.40f, 0.42f}, {-4.40f, 0.42f}},
        0.04f, 2.10f, 3.30f, 0.55f, 0.40f, 1.05f, {3.20f, -2.30f},
        {1.0f, 1.8f}, 12.0f, true, {1.90f, 3.48f, 0.0f}, {1.60f, 0.30f}};
    return s;
}

// ── small helpers ────────────────────────────────────────────────────
uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16; return x;
}
float rnd01(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return float(hash32(state) & 0xFFFFFFu) / 16777216.0f;
}
float h01(uint32_t id, uint32_t salt) {
    return float(hash32(id * 0x9E3779B1u + salt * 0x85EBCA6Bu) & 0xFFFFFFu) /
           16777216.0f;
}
uint64_t cellKey(float x, float z, float cell) {
    const int64_t ix = int64_t(std::floor(x / cell)) + (1ll << 30);
    const int64_t iz = int64_t(std::floor(z / cell)) + (1ll << 30);
    return (uint64_t(ix) << 32) ^ uint64_t(iz & 0xFFFFFFFF);
}

// ── 2D ear clipping (simple polygon, any orientation) ────────────────
float signedArea(const std::vector<glm::vec2>& p) {
    float a = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) {
        const glm::vec2& u = p[i];
        const glm::vec2& v = p[(i + 1) % p.size()];
        a += u.x * v.y - v.x * u.y;
    }
    return 0.5f * a;
}
bool pointInTri(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b,
                const glm::vec2& c) {
    auto cr = [](const glm::vec2& o, const glm::vec2& u, const glm::vec2& v) {
        return (u.x - o.x) * (v.y - o.y) - (u.y - o.y) * (v.x - o.x);
    };
    const float d1 = cr(a, b, p), d2 = cr(b, c, p), d3 = cr(c, a, p);
    const bool neg = d1 < 0 || d2 < 0 || d3 < 0;
    const bool pos = d1 > 0 || d2 > 0 || d3 > 0;
    return !(neg && pos);
}
std::vector<glm::ivec3> earClip(std::vector<glm::vec2> p) {
    std::vector<glm::ivec3> out;
    std::vector<int> idx(p.size());
    for (size_t i = 0; i < p.size(); ++i) idx[i] = int(i);
    if (signedArea(p) < 0.0f) std::reverse(idx.begin(), idx.end());
    int guard = 0;
    while (idx.size() > 3 && guard++ < 4000) {
        bool cut = false;
        for (size_t i = 0; i < idx.size(); ++i) {
            const int ia = idx[(i + idx.size() - 1) % idx.size()];
            const int ib = idx[i];
            const int ic = idx[(i + 1) % idx.size()];
            const glm::vec2 a = p[ia], b = p[ib], c = p[ic];
            const float cross = (b.x - a.x) * (c.y - b.y) -
                                (b.y - a.y) * (c.x - b.x);
            if (cross <= 1e-9f) continue;          // reflex
            bool inside = false;
            for (int j : idx) {
                if (j == ia || j == ib || j == ic) continue;
                if (pointInTri(p[j], a, b, c)) { inside = true; break; }
            }
            if (inside) continue;
            out.push_back({ia, ib, ic});
            idx.erase(idx.begin() + int(i));
            cut = true;
            break;
        }
        if (!cut) break;
    }
    if (idx.size() == 3) out.push_back({idx[0], idx[1], idx[2]});
    return out;
}

// ── meshes: flat-shaded triangles, one vertex each ───────────────────
struct MeshData {
    std::vector<glm::vec3> pos, nrm;
    std::vector<uint32_t> idx;
    void tri(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
        glm::vec3 n = glm::cross(b - a, c - a);
        const float l = glm::length(n);
        n = l > 1e-9f ? n / l : glm::vec3(0, 1, 0);
        const uint32_t base = uint32_t(pos.size());
        pos.push_back(a); pos.push_back(b); pos.push_back(c);
        nrm.push_back(n); nrm.push_back(n); nrm.push_back(n);
        idx.push_back(base); idx.push_back(base + 1); idx.push_back(base + 2);
    }
    void quad(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
              const glm::vec3& d) { tri(a, b, c); tri(a, c, d); }
};

// min / max y of the profile polygon on the vertical line at x
bool profileYRange(const std::vector<glm::vec2>& prof, float x, float& yb, float& yt) {
    bool any = false;
    yb = 1e9f; yt = -1e9f;
    const size_t n = prof.size();
    for (size_t i = 0; i < n; ++i) {
        const glm::vec2 a = prof[i], b = prof[(i + 1) % n];
        if ((a.x <= x && x <= b.x) || (b.x <= x && x <= a.x)) {
            float y0, y1;
            if (std::abs(b.x - a.x) < 1e-9f) { y0 = a.y; y1 = b.y; }
            else { const float t = (x - a.x) / (b.x - a.x); y0 = y1 = a.y + (b.y - a.y) * t; }
            yb = std::min(yb, std::min(y0, y1));
            yt = std::max(yt, std::max(y0, y1));
            any = true;
        }
    }
    return any;
}

// One cross-section ring (y, z) at station x, from the bottom centre
// round the right side, over the roof, and down the left: rounded
// sill, side up to the belt, tumblehome to a rounded shoulder, roof.
// Always the same vertex count, so consecutive rings loft.
constexpr int kRingSill = 4, kRingSide = 4, kRingTumble = 4, kRingShoulder = 6, kRingRoof = 3;
constexpr int kRingHalf = kRingSill + 1 + kRingSide + kRingTumble + kRingShoulder + kRingRoof;
constexpr int kRingN = 2 * kRingHalf - 2;

bool sectionRing(const Spec& sp, const std::vector<glm::vec2>& prof, float x,
                 float wscale, std::vector<glm::vec2>& ring) {
    float yb, yt;
    if (!profileYRange(prof, x, yb, yt)) return false;
    const float W2 = 0.5f * sp.W * wscale;
    const float h = std::max(yt - yb, 0.02f);
    const float rb = std::min(0.10f, 0.30f * h);          // sill radius
    const float rt = std::min(0.16f, 0.30f * h);          // shoulder radius
    auto w_at = [&](float y) {
        const float t = glm::clamp((y - sp.belt) / std::max(sp.roof - sp.belt, 0.05f), 0.0f, 1.0f);
        return W2 * (1.0f - sp.tumble * t);
    };
    std::vector<glm::vec2> right;
    right.reserve(kRingHalf);
    for (int k = 0; k <= kRingSill; ++k) {
        const float a = 1.5707963f * float(k) / kRingSill;
        right.push_back({yb + rb - rb * std::cos(a), (w_at(yb + rb) - rb) + rb * std::sin(a)});
    }
    const float y_lo = yb + rb;
    const float y_belt = std::max(std::min(sp.belt, yt - rt), y_lo);
    for (int k = 1; k <= kRingSide; ++k) {
        const float y = y_lo + (y_belt - y_lo) * float(k) / kRingSide;
        right.push_back({y, w_at(y)});
    }
    const float y_sh = yt - rt;
    for (int k = 1; k <= kRingTumble; ++k) {
        const float y = y_belt + (y_sh - y_belt) * float(k) / kRingTumble;
        right.push_back({y, w_at(y)});
    }
    const float wsh = w_at(y_sh);
    for (int k = 1; k <= kRingShoulder; ++k) {
        const float a = 1.5707963f * float(k) / kRingShoulder;
        right.push_back({y_sh + rt * std::sin(a), (wsh - rt) + rt * std::cos(a)});
    }
    for (int k = 1; k <= kRingRoof; ++k)
        right.push_back({yt, (wsh - rt) * (1.0f - float(k) / kRingRoof)});
    ring.clear();
    ring.reserve(kRingN);
    for (const glm::vec2& p : right) ring.push_back(p);
    for (int i = int(right.size()) - 2; i >= 1; --i) ring.push_back({right[size_t(i)].x, -right[size_t(i)].y});
    return int(ring.size()) == kRingN;
}

// THE HULL: a loft of rounded sections along the side profile.  The
// old extrusion was a slab with the profile as its silhouette; this
// rounds the sills and shoulders, tapers the body a little toward the
// bumpers with rounded corners, keeps the tumblehome, and shades the
// sweep with smooth vertex normals (the bumper faces stay flat, on
// their own vertices).  vehicle.frag paints by local position, so the
// glass, lights, grille and seams land where they did.
MeshData buildHull(const Spec& sp) {
    MeshData m;
    std::vector<glm::vec2> prof = sp.profile;
    if (signedArea(prof) < 0.0f) std::reverse(prof.begin(), prof.end());
    float xmin = 1e9f, xmax = -1e9f;
    for (const glm::vec2& p : prof) { xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x); }
    const float L = std::max(xmax - xmin, 0.1f);
    // stations: cosine-spaced (dense at the ends) plus every profile x
    std::vector<float> st;
    const int n_st = 34;
    for (int k = 0; k <= n_st; ++k) {
        const float t = float(k) / n_st;
        st.push_back(xmin + L * (0.5f - 0.5f * std::cos(3.14159265f * t)));
    }
    for (const glm::vec2& p : prof) st.push_back(p.x);
    std::sort(st.begin(), st.end());
    st.erase(std::unique(st.begin(), st.end(), [](float a, float b) { return std::abs(a - b) < 1e-4f; }), st.end());
    std::vector<float> xs;
    std::vector<std::vector<glm::vec2>> rings;
    for (float x : st) {
        const float t = (x - 0.5f * (xmin + xmax)) / (0.5f * L);
        const float edge = std::max(0.0f, std::abs(t) - 0.84f) / 0.16f;
        const float at = std::abs(t);
        const float wscale = (1.0f - 0.04f * at * at * at) * (1.0f - 0.11f * edge * edge);
        const float xx = glm::clamp(x, xmin + 1e-4f, xmax - 1e-4f);
        std::vector<glm::vec2> ring;
        if (!sectionRing(sp, prof, xx, wscale, ring)) continue;
        xs.push_back(x);
        rings.push_back(std::move(ring));
    }
    if (rings.size() < 2) return m;
    const int M = kRingN;
    // the sweep: shared vertices, smooth normals
    const uint32_t base = uint32_t(m.pos.size());
    for (size_t i = 0; i < rings.size(); ++i)
        for (int j = 0; j < M; ++j)
            m.pos.push_back({xs[i], rings[i][size_t(j)].x, rings[i][size_t(j)].y});
    m.nrm.assign(m.pos.size(), glm::vec3(0.0f));
    auto push_tri = [&](uint32_t a, uint32_t b, uint32_t c) {
        const glm::vec3 n = glm::cross(m.pos[b] - m.pos[a], m.pos[c] - m.pos[a]);
        m.idx.push_back(a); m.idx.push_back(b); m.idx.push_back(c);
        m.nrm[a] += n; m.nrm[b] += n; m.nrm[c] += n;
    };
    for (size_t i = 0; i + 1 < rings.size(); ++i) {
        for (int j = 0; j < M; ++j) {
            const uint32_t a = base + uint32_t(i * M + j);
            const uint32_t b = base + uint32_t(i * M + (j + 1) % M);
            const uint32_t c = base + uint32_t((i + 1) * M + (j + 1) % M);
            const uint32_t d = base + uint32_t((i + 1) * M + j);
            push_tri(a, b, c);
            push_tri(a, c, d);
        }
    }
    // orientation: normals must point OUT; test one roof triangle
    {
        const size_t mid = rings.size() / 2;
        const uint32_t a = base + uint32_t(mid * M + (M / 2));    // roof centre-ish
        if (m.nrm[a].y < 0.0f) {
            // flip every sweep triangle and the accumulated normals
            for (size_t k = 0; k + 2 < m.idx.size(); k += 3) std::swap(m.idx[k + 1], m.idx[k + 2]);
            for (glm::vec3& n : m.nrm) n = -n;
        }
    }
    for (glm::vec3& n : m.nrm) {
        const float l = glm::length(n);
        n = l > 1e-9f ? n / l : glm::vec3(0, 1, 0);
    }
    // the bumper faces: flat fans on their own vertices
    for (int end = 0; end < 2; ++end) {
        const size_t i = end == 0 ? 0 : rings.size() - 1;
        const float x = xs[i];
        float yb = 1e9f, yt = -1e9f;
        for (const glm::vec2& p : rings[i]) { yb = std::min(yb, p.x); yt = std::max(yt, p.x); }
        const glm::vec3 c(x, 0.5f * (yb + yt), 0.0f);
        for (int j = 0; j < M; ++j) {
            const glm::vec3 a(x, rings[i][size_t(j)].x, rings[i][size_t(j)].y);
            const glm::vec3 b(x, rings[i][size_t((j + 1) % M)].x, rings[i][size_t((j + 1) % M)].y);
            // winding chosen so the fan's normal points out: -x at
            // the tail (end 0), +x at the nose
            if (end == 0) m.tri(c, a, b); else m.tri(c, b, a);
        }
    }
    return m;
}

// unit cylinder along z: radius 1, z in -1..1
MeshData buildWheel() {
    MeshData m;
    const int S = 22;
    for (int i = 0; i < S; ++i) {
        const float a0 = 6.2831853f * float(i) / S, a1 = 6.2831853f * float(i + 1) / S;
        const glm::vec3 p0(std::cos(a0), std::sin(a0), 0.0f), p1(std::cos(a1), std::sin(a1), 0.0f);
        const glm::vec3 z(0, 0, 1);
        // tread: two triangles with radial normals (flat per face)
        m.quad(p0 - z, p1 - z, p1 + z, p0 + z);
        // caps
        m.tri(z, p0 + z, p1 + z);
        m.tri(-z, p1 - z, p0 - z);
    }
    return m;
}

// unit box, -1..1
MeshData buildBox() {
    MeshData m;
    const glm::vec3 c[8] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                            {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    m.quad(c[4], c[5], c[6], c[7]);   // +z
    m.quad(c[1], c[0], c[3], c[2]);   // -z
    m.quad(c[5], c[1], c[2], c[6]);   // +x
    m.quad(c[0], c[4], c[7], c[3]);   // -x
    m.quad(c[7], c[6], c[2], c[3]);   // +y
    m.quad(c[0], c[1], c[5], c[4]);   // -y
    return m;
}

// ── the cabin: what is inside the hull ───────────────────────────────
// Model space like the hull (x forward, y up, z right, ground y = 0).
struct Cabin {
    float floor_y;                      // the cabin floor
    float x0, x1;                       // cabin extent along x
    float half_z;                       // half width inside the doors
    glm::vec4 dash;                     // (x0, x1, y0, y1)
    glm::vec3 wheel;                    // steering wheel centre
    float wheel_r;
    std::vector<glm::vec3> seats;       // (x, cushion top y, z); 0 = driver
    float seat_w, seat_d;               // cushion width (z) and depth (x)
    float back_h;                       // backrest height over the cushion
};

const Cabin& cabinOf(int type) {
    static std::vector<Cabin> c;
    if (c.empty()) {
        c.resize(VehicleSystem::kTypeCount);
        c[VehicleSystem::kSedan] = {
            0.30f, -1.35f, 1.05f, 0.78f, {0.95f, 1.40f, 0.70f, 0.96f},
            {0.72f, 0.86f, -0.38f}, 0.18f,
            {{0.15f, 0.52f, -0.38f}, {0.15f, 0.52f, 0.38f},
             {-0.85f, 0.52f, -0.38f}, {-0.85f, 0.52f, 0.38f}},
            0.50f, 0.48f, 0.60f};
        c[VehicleSystem::kSuv] = {
            0.36f, -1.30f, 1.00f, 0.78f, {0.95f, 1.45f, 0.82f, 1.08f},
            {0.75f, 0.98f, -0.38f}, 0.18f,
            {{0.20f, 0.62f, -0.38f}, {0.20f, 0.62f, 0.38f},
             {-0.80f, 0.62f, -0.38f}, {-0.80f, 0.62f, 0.38f}},
            0.50f, 0.48f, 0.62f};
        c[VehicleSystem::kCyber] = {
            0.40f, -0.40f, 1.40f, 0.86f, {1.30f, 1.75f, 0.85f, 1.10f},
            {1.05f, 1.02f, -0.42f}, 0.18f,
            {{0.55f, 0.66f, -0.42f}, {0.55f, 0.66f, 0.42f},
             {-0.30f, 0.66f, -0.42f}, {-0.30f, 0.66f, 0.42f}},
            0.52f, 0.48f, 0.62f};
        c[VehicleSystem::kPickup] = {
            0.45f, -0.55f, 0.60f, 0.88f, {0.55f, 1.05f, 0.95f, 1.22f},
            {0.35f, 1.12f, -0.42f}, 0.19f,
            {{-0.05f, 0.75f, -0.42f}, {-0.05f, 0.75f, 0.42f}},
            0.52f, 0.48f, 0.62f};
        c[VehicleSystem::kSemi] = {
            1.20f, -1.50f, 1.50f, 1.10f, {1.15f, 1.70f, 1.75f, 2.05f},
            {0.80f, 1.95f, -0.55f}, 0.22f,
            {{0.15f, 1.55f, -0.55f}, {0.15f, 1.55f, 0.55f}},
            0.55f, 0.50f, 0.70f};
        c[VehicleSystem::kPolice] = {
            0.38f, -1.45f, 1.10f, 0.86f, {1.05f, 1.55f, 0.85f, 1.12f},
            {0.85f, 1.00f, -0.40f}, 0.18f,
            {{0.20f, 0.64f, -0.40f}, {0.20f, 0.64f, 0.40f},
             {-0.85f, 0.64f, -0.40f}, {-0.85f, 0.64f, 0.40f}},
            0.50f, 0.48f, 0.62f};
        c[VehicleSystem::kAmbulance] = {
            0.45f, 0.30f, 1.60f, 0.84f, {1.55f, 2.05f, 0.95f, 1.25f},
            {1.35f, 1.12f, -0.45f}, 0.19f,
            {{1.00f, 0.75f, -0.45f}, {1.00f, 0.75f, 0.45f}},
            0.52f, 0.48f, 0.64f};
        c[VehicleSystem::kSchoolBus] = {
            0.55f, -4.80f, 3.60f, 1.08f, {3.65f, 4.20f, 1.00f, 1.42f},
            {3.75f, 1.35f, -0.62f}, 0.24f,
            {{3.30f, 0.95f, -0.62f}}, 0.90f, 0.45f, 0.62f};
        // bench rows down the bus, two a row
        for (float x = 2.35f; x > -4.7f; x -= 0.85f) {
            c[VehicleSystem::kSchoolBus].seats.push_back({x, 0.95f, -0.62f});
            c[VehicleSystem::kSchoolBus].seats.push_back({x, 0.95f, 0.62f});
        }
        c[VehicleSystem::kFireEngine] = {
            0.95f, 0.85f, 3.25f, 1.10f, {3.00f, 3.40f, 1.55f, 1.90f},
            {2.90f, 1.82f, -0.60f}, 0.22f,
            {{2.55f, 1.25f, -0.60f}, {2.55f, 1.25f, 0.60f},
             {1.35f, 1.25f, -0.60f}, {1.35f, 1.25f, 0.60f}},
            0.55f, 0.50f, 0.70f};
    }
    return c[size_t(type)];
}

void addBox(MeshData& m, const glm::vec3& c, const glm::vec3& h) {
    const glm::vec3 p[8] = {
        c + glm::vec3(-h.x, -h.y, -h.z), c + glm::vec3(h.x, -h.y, -h.z),
        c + glm::vec3(h.x, h.y, -h.z),   c + glm::vec3(-h.x, h.y, -h.z),
        c + glm::vec3(-h.x, -h.y, h.z),  c + glm::vec3(h.x, -h.y, h.z),
        c + glm::vec3(h.x, h.y, h.z),    c + glm::vec3(-h.x, h.y, h.z)};
    m.quad(p[4], p[5], p[6], p[7]);   // +z
    m.quad(p[1], p[0], p[3], p[2]);   // -z
    m.quad(p[5], p[1], p[2], p[6]);   // +x
    m.quad(p[0], p[4], p[7], p[3]);   // -x
    m.quad(p[7], p[6], p[2], p[3]);   // +y
    m.quad(p[0], p[1], p[5], p[4]);   // -y
}

// Seats, dashboard, steering wheel and floor inside; door handles and
// mirrors outside.  vehicle.frag paints it by position (kind 4).
MeshData buildInterior(const Spec& sp, const Cabin& cb, int type) {
    MeshData m;
    // floor
    m.quad({cb.x0, cb.floor_y, -cb.half_z}, {cb.x1, cb.floor_y, -cb.half_z},
           {cb.x1, cb.floor_y, cb.half_z}, {cb.x0, cb.floor_y, cb.half_z});
    // dashboard
    addBox(m, {0.5f * (cb.dash.x + cb.dash.y), 0.5f * (cb.dash.z + cb.dash.w), 0.0f},
           {0.5f * (cb.dash.y - cb.dash.x), 0.5f * (cb.dash.w - cb.dash.z),
            cb.half_z - 0.02f});
    // steering wheel: a ring of 10 blocks, tilted toward the driver
    {
        const int S = 10;
        const float r = cb.wheel_r;
        for (int i = 0; i < S; ++i) {
            const float a0 = 6.2831853f * float(i) / S;
            const float a1 = 6.2831853f * float(i + 1) / S;
            // ring in the (y, z) plane tilted 35 deg back along x
            auto P = [&](float a, float ro) {
                const float cy = std::cos(a) * ro, cz = std::sin(a) * ro;
                return glm::vec3(cb.wheel.x - cy * 0.57f, cb.wheel.y + cy * 0.82f,
                                 cb.wheel.z + cz);
            };
            const glm::vec3 a = P(a0, r - 0.02f), b = P(a1, r - 0.02f);
            const glm::vec3 c2 = P(a1, r + 0.02f), d = P(a0, r + 0.02f);
            const glm::vec3 t(0.02f * 0.82f, 0.02f * 0.57f, 0.0f);
            m.quad(a + t, b + t, c2 + t, d + t);
            m.quad(d - t, c2 - t, b - t, a - t);
            m.quad(a - t, b - t, b + t, a + t);
            m.quad(d + t, c2 + t, c2 - t, d - t);
        }
        // the column
        addBox(m, {cb.wheel.x + 0.10f, cb.wheel.y - 0.06f, cb.wheel.z},
               {0.10f, 0.03f, 0.03f});
    }
    // seats
    for (const glm::vec3& s : cb.seats) {
        const float hw = 0.5f * cb.seat_w, hd = 0.5f * cb.seat_d;
        addBox(m, {s.x, s.y - 0.06f, s.z}, {hd, 0.06f, hw});          // cushion
        addBox(m, {s.x - hd + 0.05f, s.y + 0.5f * cb.back_h, s.z},
               {0.05f, 0.5f * cb.back_h, hw});                         // back
        if (type != VehicleSystem::kSchoolBus)
            addBox(m, {s.x - hd + 0.05f, s.y + cb.back_h + 0.10f, s.z},
                   {0.045f, 0.09f, 0.13f});                            // headrest
    }
    // ── outside: door handles and mirrors ───────────────────────────
    {
        const float hw = 0.5f * sp.W;
        const float belt = sp.belt;
        // the glass table in the shader knows the door split; here the
        // handles sit a hand below the belt, front door and rear door
        const float gx0 = cb.x0 + 0.15f, gx1 = cb.x1 - 0.15f;
        const float pb = 0.5f * (gx0 + gx1);
        for (int side = -1; side <= 1; side += 2) {
            const float z = float(side) * (hw + 0.012f);
            addBox(m, {pb - 0.35f, belt - 0.16f, z}, {0.085f, 0.014f, 0.012f});
            if (cb.seats.size() > 2 && type != VehicleSystem::kSchoolBus &&
                type != VehicleSystem::kFireEngine)
                addBox(m, {pb + 0.45f, belt - 0.16f, z}, {0.085f, 0.014f, 0.012f});
            // mirror: a stalk and a head at the front of the door glass
            const float mx = cb.x1 - 0.05f, my = belt + 0.16f;
            addBox(m, {mx, my, float(side) * (hw + 0.06f)}, {0.03f, 0.02f, 0.06f});
            addBox(m, {mx, my + 0.02f, float(side) * (hw + 0.16f)},
                   {0.045f, 0.065f, 0.075f});
        }
    }
    return m;
}

// streams: hull per type, interior per type, wheel, bar, then the glass
// pass per type (the hull mesh again, blended)
constexpr int kMeshInterior0 = VehicleSystem::kTypeCount;
constexpr int kMeshWheel = VehicleSystem::kTypeCount * 2;
constexpr int kMeshBar = VehicleSystem::kTypeCount * 2 + 1;
constexpr int kMeshCount = VehicleSystem::kTypeCount * 2 + 2;
constexpr int kStreamGlass0 = kMeshCount;
constexpr int kStreamCount = kMeshCount + VehicleSystem::kTypeCount;
int streamMesh(int stream) {
    return stream >= kStreamGlass0 ? stream - kStreamGlass0 : stream;
}

glm::vec3 paletteColor(uint32_t seed) {
    static const glm::vec3 kPal[10] = {
        {0.72f, 0.73f, 0.75f}, {0.92f, 0.92f, 0.90f}, {0.05f, 0.05f, 0.06f},
        {0.15f, 0.35f, 0.75f}, {0.62f, 0.08f, 0.08f}, {0.35f, 0.36f, 0.38f},
        {0.10f, 0.16f, 0.35f}, {0.12f, 0.30f, 0.20f}, {0.80f, 0.78f, 0.72f},
        {0.55f, 0.20f, 0.10f}};
    return kPal[int(h01(seed, 0x11u) * 10.0f) % 10];
}

}  // namespace

// ── statics ──────────────────────────────────────────────────────────
std::shared_ptr<er::Pipeline> VehicleSystem::s_gbuf_pipeline_;
std::shared_ptr<er::PipelineLayout>  VehicleSystem::s_pipeline_layout_;
std::shared_ptr<er::Pipeline>        VehicleSystem::s_pipeline_;
std::vector<VehicleSystem::Mesh>     VehicleSystem::s_meshes_;
std::shared_ptr<er::Device>          VehicleSystem::s_device_;
std::shared_ptr<er::BufferInfo>      VehicleSystem::s_inst_buf_;
std::shared_ptr<er::Pipeline>        VehicleSystem::s_glass_pipeline_;
uint32_t                             VehicleSystem::s_inst_capacity_ = 0;

void VehicleSystem::initStaticMembers(
    const std::shared_ptr<er::Device>& device,
    const er::DescriptorSetLayoutList& global_desc_set_layouts,
    const er::GraphicPipelineInfo& graphic_pipeline_info,
    const er::PipelineRenderbufferFormats& frame_buffer_format,
    const er::PipelineRenderbufferFormats& gbuffer_format) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::CitizenDrawParams);
    s_pipeline_layout_ = device->createPipelineLayout(
        global_desc_set_layouts, {push_const_range},
        std::source_location::current());
    s_device_ = device;

    // the citizens' cube-pipeline vertex layout: POSITION, NORMAL, and
    // the six per-instance vec4s at locations 10-15 citizen.vert reads
    std::vector<er::VertexInputBindingDescription> bindings(3);
    std::vector<er::VertexInputAttributeDescription> attribs(8);
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(glm::vec3);
    bindings[0].input_rate = er::VertexInputRate::VERTEX;
    attribs[0].binding = 0;
    attribs[0].location = VINPUT_POSITION;
    attribs[0].format = er::Format::R32G32B32_SFLOAT;
    attribs[0].offset = 0;
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(glm::vec3);
    bindings[1].input_rate = er::VertexInputRate::VERTEX;
    attribs[1].binding = 1;
    attribs[1].location = VINPUT_NORMAL;
    attribs[1].format = er::Format::R32G32B32_SFLOAT;
    attribs[1].offset = 0;
    bindings[2].binding = 2;
    bindings[2].stride = sizeof(PartInstance);
    bindings[2].input_rate = er::VertexInputRate::INSTANCE;
    for (int k = 0; k < 6; ++k) {
        attribs[2 + k].binding = 2;
        attribs[2 + k].location = uint32_t(10 + k);
        attribs[2 + k].format = er::Format::R32G32B32A32_SFLOAT;
        attribs[2 + k].offset = uint32_t(k * sizeof(glm::vec4));
    }
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    er::RasterizationStateOverride raster_override;
    raster_override.override_double_sided = true;
    raster_override.double_sided = true;
    er::ShaderModuleList shader_modules(2);
    shader_modules[0] = er::helper::loadShaderModule(
        device, "citizen_vert.spv", er::ShaderStageFlagBits::VERTEX_BIT,
        std::source_location::current());
    shader_modules[1] = er::helper::loadShaderModule(
        device, "vehicle_frag.spv", er::ShaderStageFlagBits::FRAGMENT_BIT,
        std::source_location::current());
    s_pipeline_ = device->createPipeline(
        s_pipeline_layout_, bindings, attribs, input_assembly,
        graphic_pipeline_info, shader_modules, frame_buffer_format,
        raster_override, std::source_location::current());
    {
        er::ShaderModuleList gbuf_modules = shader_modules;
        gbuf_modules[1] = er::helper::loadShaderModule(device, "vehicle_gbuf_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT, std::source_location::current());
        er::GraphicPipelineInfo gbuf_info = graphic_pipeline_info;
        auto att = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS), false);
        gbuf_info.blend_state_info = std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(
                std::vector<er::PipelineColorBlendAttachmentState>(4, att)));
        s_gbuf_pipeline_ = device->createPipeline(
            s_pipeline_layout_, bindings, attribs, input_assembly,
            gbuf_info, gbuf_modules, gbuffer_format,
            raster_override, std::source_location::current());
    }

    // THE GLASS PASS: the same shaders and layout, alpha-blended, depth
    // tested against the opaque pass but not written (two panes of
    // one car overlap and must not occlude each other).
    {
        er::GraphicPipelineInfo glass_info = graphic_pipeline_info;
        auto blend_att = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS), true,
            er::BlendFactor::SRC_ALPHA, er::BlendFactor::ONE_MINUS_SRC_ALPHA,
            er::BlendOp::ADD, er::BlendFactor::ONE, er::BlendFactor::ZERO,
            er::BlendOp::ADD);
        std::vector<er::PipelineColorBlendAttachmentState> atts(1, blend_att);
        glass_info.blend_state_info =
            std::make_shared<er::PipelineColorBlendStateCreateInfo>(
                er::helper::fillPipelineColorBlendStateCreateInfo(atts));
        glass_info.depth_stencil_info =
            std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
                er::helper::fillPipelineDepthStencilStateCreateInfo(true, false));
        try {
            s_glass_pipeline_ = device->createPipeline(
                s_pipeline_layout_, bindings, attribs, input_assembly,
                glass_info, shader_modules, frame_buffer_format,
                raster_override, std::source_location::current());
        } catch (const std::exception& e) {
            std::cout << "[vehicle] glass pipeline failed (" << e.what()
                      << ") -- windows stay opaque" << std::endl;
            s_glass_pipeline_ = nullptr;
        }
    }

    // meshes: one hull + one interior per type, the wheel, the light bar
    s_meshes_.assign(kMeshCount, Mesh{});
    auto upload = [&](int i, const MeshData& md) {
        Mesh& m = s_meshes_[i];
        m.shadow.positions = md.pos;
        m.shadow.indices = md.idx;
        m.pos = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            md.pos.size() * sizeof(glm::vec3), md.pos.data(),
            std::source_location::current());
        m.nrm = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            md.nrm.size() * sizeof(glm::vec3), md.nrm.data(),
            std::source_location::current());
        m.idx = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
            md.idx.size() * sizeof(uint32_t), md.idx.data(),
            std::source_location::current());
        m.count = uint32_t(md.idx.size());
    };
    for (int t = 0; t < kTypeCount; ++t) {
        upload(t, buildHull(specs()[t]));
        upload(kMeshInterior0 + t, buildInterior(specs()[t], cabinOf(t), t));
    }
    upload(kMeshWheel, buildWheel());
    upload(kMeshBar, buildBox());
    std::cout << "[vehicle] pipeline + " << kMeshCount << " meshes ready"
              << std::endl;
}

void VehicleSystem::destroyStaticMembers(
    const std::shared_ptr<er::Device>& device) {
    if (s_pipeline_layout_) device->destroyPipelineLayout(s_pipeline_layout_);
    s_pipeline_layout_ = nullptr;
    if (s_gbuf_pipeline_) device->destroyPipeline(s_gbuf_pipeline_);
    s_gbuf_pipeline_ = nullptr;
    if (s_pipeline_) device->destroyPipeline(s_pipeline_);
    s_pipeline_ = nullptr;
    if (s_glass_pipeline_) device->destroyPipeline(s_glass_pipeline_);
    s_glass_pipeline_ = nullptr;
    for (auto& m : s_meshes_) {
        if (m.pos) m.pos->destroy(device);
        if (m.nrm) m.nrm->destroy(device);
        if (m.idx) m.idx->destroy(device);
    }
    s_meshes_.clear();
    if (s_inst_buf_) { s_inst_buf_->destroy(device); s_inst_buf_ = nullptr; }
    s_inst_capacity_ = 0;
    s_device_ = nullptr;
}

// ── the road graph ───────────────────────────────────────────────────
bool VehicleSystem::loadRoads(const std::string& roads_json) {
    edges_.clear(); nodes_.clear(); pt_grid_.clear(); node_grid_.clear();
    vehicles_.clear(); node_claim_.clear();
    std::ifstream f(roads_json);
    if (!f) {
        std::cout << "[vehicle] no roads json at " << roads_json
                  << " -- no traffic" << std::endl;
        return false;
    }
    nlohmann::json j;
    try { f >> j; } catch (const std::exception& e) {
        std::cout << "[vehicle] roads json unreadable (" << e.what() << ")"
                  << std::endl;
        return false;
    }
    if (!j.contains("splines")) return false;
    std::vector<std::vector<glm::vec3>> splines;
    std::vector<std::vector<float>> halves;
    for (const auto& sp : j["splines"]) {
        if (!sp.contains("pts")) continue;
        const auto& pts = sp["pts"];
        const size_t n = pts.size();
        if (n < 2) continue;
        std::vector<glm::vec3> p; p.reserve(n);
        std::vector<float> h; h.reserve(n);
        const bool has_y = sp.contains("y") && sp["y"].size() == n;
        const bool has_h = sp.contains("half") && sp["half"].size() == n;
        for (size_t i = 0; i < n; ++i) {
            const float x = pts[i][0].get<float>();
            const float z = pts[i][1].get<float>();
            const float y = has_y ? sp["y"][i].get<float>() : 0.0f;
            p.push_back({x, y, z});
            h.push_back(has_h ? sp["half"][i].get<float>() : 4.0f);
        }
        splines.push_back(std::move(p));
        halves.push_back(std::move(h));
    }
    if (splines.empty()) return false;
    buildGraph(splines, halves);
    std::cout << "[vehicle] roads: " << splines.size() << " splines -> "
              << edges_.size() << " edges, " << nodes_.size() << " nodes"
              << std::endl;
    return !edges_.empty();
}

int VehicleSystem::nodeAt(const glm::vec3& p) {
    // an existing node within kNodeMerge, else a new one
    const int cx = int(std::floor(p.x / kNodeCell));
    const int cz = int(std::floor(p.z / kNodeCell));
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            const uint64_t key = cellKey(float(cx + dx) * kNodeCell,
                                         float(cz + dz) * kNodeCell, kNodeCell);
            auto it = node_grid_.find(key);
            if (it == node_grid_.end()) continue;
            // a cell may hold several nodes: chain through the list
            for (int ni = it->second; ni >= 0;) {
                const Node& n = nodes_[ni];
                const float ddx = n.pos.x - p.x, ddz = n.pos.z - p.z;
                if (ddx * ddx + ddz * ddz < kNodeMerge * kNodeMerge) return ni;
                break;
            }
        }
    }
    Node n; n.pos = p;
    nodes_.push_back(n);
    const int id = int(nodes_.size()) - 1;
    // one node per cell in the grid; a second node in the same cell
    // (> 2.5 m apart) is only missed for merging, which is harmless
    node_grid_.emplace(cellKey(p.x, p.z, kNodeCell), id);
    return id;
}

void VehicleSystem::buildGraph(std::vector<std::vector<glm::vec3>>& splines,
                               std::vector<std::vector<float>>& halves) {
    // 1. T-junctions: a spline whose END lies on another spline's
    //    length (the residential streets the mesh stage joined to the
    //    network) snaps onto it, and that spline is cut there.
    std::unordered_map<uint64_t, std::vector<RoadPt>> grid;
    for (size_t s = 0; s < splines.size(); ++s) {
        for (size_t i = 0; i < splines[s].size(); ++i) {
            const glm::vec3& p = splines[s][i];
            grid[cellKey(p.x, p.z, 16.0f)].push_back({int(s), int(i)});
        }
    }
    std::vector<std::vector<int>> cuts(splines.size());
    for (size_t s = 0; s < splines.size(); ++s) {
        for (int end = 0; end < 2; ++end) {
            glm::vec3& ep = end == 0 ? splines[s].front() : splines[s].back();
            float best = kTeeSnapM * kTeeSnapM;
            RoadPt hit{-1, -1};
            const int cx = int(std::floor(ep.x / 16.0f));
            const int cz = int(std::floor(ep.z / 16.0f));
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dx = -1; dx <= 1; ++dx) {
                    auto it = grid.find(cellKey(float(cx + dx) * 16.0f,
                                                float(cz + dz) * 16.0f, 16.0f));
                    if (it == grid.end()) continue;
                    for (const RoadPt& rp : it->second) {
                        if (rp.edge == int(s)) continue;
                        const auto& o = splines[rp.edge];
                        if (rp.index <= 0 || rp.index >= int(o.size()) - 1)
                            continue;              // interior points only
                        const float ddx = o[rp.index].x - ep.x;
                        const float ddz = o[rp.index].z - ep.z;
                        const float d2 = ddx * ddx + ddz * ddz;
                        if (d2 < best) { best = d2; hit = rp; }
                    }
                }
            }
            // an end already touching another spline's END merges by
            // the node grid below; only a true T needs a cut
            if (hit.edge >= 0) {
                cuts[hit.edge].push_back(hit.index);
                ep = splines[hit.edge][hit.index];
            }
        }
    }
    // 2. cut the splines into edges, nodes at every end and cut
    for (size_t s = 0; s < splines.size(); ++s) {
        std::vector<int>& c = cuts[s];
        c.push_back(0);
        c.push_back(int(splines[s].size()) - 1);
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        for (size_t k = 0; k + 1 < c.size(); ++k) {
            const int i0 = c[k], i1 = c[k + 1];
            if (i1 - i0 < 1) continue;
            Edge e;
            e.pts.assign(splines[s].begin() + i0, splines[s].begin() + i1 + 1);
            e.half.assign(halves[s].begin() + i0, halves[s].begin() + i1 + 1);
            e.s.resize(e.pts.size());
            e.s[0] = 0.0f;
            for (size_t i = 1; i < e.pts.size(); ++i) {
                const glm::vec3 d = e.pts[i] - e.pts[i - 1];
                e.s[i] = e.s[i - 1] + std::sqrt(d.x * d.x + d.z * d.z);
            }
            e.len = e.s.back();
            if (e.len < 0.5f) continue;
            e.a = nodeAt(e.pts.front());
            e.b = nodeAt(e.pts.back());
            if (e.a == e.b && e.len < 20.0f) continue;   // a loop of nothing
            const int id = int(edges_.size());
            edges_.push_back(std::move(e));
            nodes_[edges_[id].a].edges.push_back(id);
            nodes_[edges_[id].b].edges.push_back(id);
        }
    }
    node_claim_.assign(nodes_.size(), -1);
    node_claim_age_.assign(nodes_.size(), 0.0f);
    buildSignals();
    // 2b. islands: union-find over the nodes, then every edge's island
    {
        std::vector<int> parent(nodes_.size());
        for (size_t i = 0; i < parent.size(); ++i) parent[i] = int(i);
        auto find = [&](int x) {
            while (parent[size_t(x)] != x) {
                parent[size_t(x)] = parent[size_t(parent[size_t(x)])];
                x = parent[size_t(x)];
            }
            return x;
        };
        for (const Edge& e : edges_) {
            const int ra = find(e.a), rb = find(e.b);
            if (ra != rb) parent[size_t(ra)] = rb;
        }
        std::unordered_map<int, int> root_to_comp;
        edge_comp_.assign(edges_.size(), -1);
        comp_edges_.clear();
        for (size_t ei = 0; ei < edges_.size(); ++ei) {
            const int r = find(edges_[ei].a);
            auto it = root_to_comp.find(r);
            if (it == root_to_comp.end()) {
                it = root_to_comp.emplace(r, int(comp_edges_.size())).first;
                comp_edges_.emplace_back();
            }
            edge_comp_[ei] = it->second;
            comp_edges_[size_t(it->second)].push_back(int(ei));
        }
        size_t biggest = 0;
        for (const auto& c : comp_edges_) biggest = std::max(biggest, c.size());
        std::cout << "[vehicle] road islands: " << comp_edges_.size()
                  << " (largest " << biggest << " edges)" << std::endl;
    }
    // 3. the road-point hash for curb and spawn queries
    {
        double len = 0.0; size_t npts = 0;
        for (size_t ei = 0; ei < edges_.size(); ++ei) {
            len += edges_[ei].len;
            npts += edges_[ei].pts.size();
            for (size_t i = 0; i < edges_[ei].pts.size(); ++i) {
                const glm::vec3& p = edges_[ei].pts[i];
                pt_grid_[cellKey(p.x, p.z, kPtCell)].push_back({int(ei), int(i)});
            }
        }
        if (npts > 0 && len > 0.0)
            pt_spacing_m_ = glm::clamp(float(len / double(npts)), 0.2f, 20.0f);
    }
}

bool VehicleSystem::nearestRoadPt(const glm::vec3& p, float radius,
                                  RoadPt& out, float* out_dist) const {
    const int r = int(std::ceil(radius / kPtCell));
    const int cx = int(std::floor(p.x / kPtCell));
    const int cz = int(std::floor(p.z / kPtCell));
    float best = radius * radius;
    bool found = false;
    for (int dz = -r; dz <= r; ++dz) {
        for (int dx = -r; dx <= r; ++dx) {
            auto it = pt_grid_.find(cellKey(float(cx + dx) * kPtCell,
                                            float(cz + dz) * kPtCell, kPtCell));
            if (it == pt_grid_.end()) continue;
            for (const RoadPt& rp : it->second) {
                const glm::vec3& q = edges_[rp.edge].pts[rp.index];
                const float ddx = q.x - p.x, ddz = q.z - p.z;
                const float d2 = ddx * ddx + ddz * ddz;
                if (d2 < best) { best = d2; out = rp; found = true; }
            }
        }
    }
    if (found && out_dist) *out_dist = std::sqrt(best);
    return found;
}

float VehicleSystem::roadLengthNear(const glm::vec3& p, float radius) const {
    const int r = int(std::ceil(radius / kPtCell));
    const int cx = int(std::floor(p.x / kPtCell));
    const int cz = int(std::floor(p.z / kPtCell));
    const float r2 = radius * radius;
    size_t n = 0;
    for (int dz = -r; dz <= r; ++dz) {
        for (int dx = -r; dx <= r; ++dx) {
            auto it = pt_grid_.find(cellKey(float(cx + dx) * kPtCell,
                                            float(cz + dz) * kPtCell, kPtCell));
            if (it == pt_grid_.end()) continue;
            for (const RoadPt& rp : it->second) {
                const glm::vec3& q = edges_[rp.edge].pts[rp.index];
                const float ddx = q.x - p.x, ddz = q.z - p.z;
                if (ddx * ddx + ddz * ddz < r2) ++n;
            }
        }
    }
    return float(n) * pt_spacing_m_;
}

glm::vec3 VehicleSystem::edgePoint(const Edge& e, float s, glm::vec3* tangent,
                                   float* half) const {
    s = glm::clamp(s, 0.0f, e.len);
    // binary search the segment
    size_t hi = std::upper_bound(e.s.begin(), e.s.end(), s) - e.s.begin();
    if (hi == 0) hi = 1;
    if (hi >= e.s.size()) hi = e.s.size() - 1;
    const size_t lo = hi - 1;
    const float seg = std::max(e.s[hi] - e.s[lo], 1e-4f);
    const float t = glm::clamp((s - e.s[lo]) / seg, 0.0f, 1.0f);
    const glm::vec3 p = glm::mix(e.pts[lo], e.pts[hi], t);
    if (tangent) {
        glm::vec3 d = e.pts[hi] - e.pts[lo];
        d.y = 0.0f;
        const float l = glm::length(d);
        *tangent = l > 1e-6f ? d / l : glm::vec3(0, 0, 1);
    }
    if (half) *half = glm::mix(e.half[lo], e.half[hi], t);
    return p;
}

glm::vec3 VehicleSystem::lanePos(const Edge& e, float s, float dir, float extra,
                                 glm::vec3* tangent, float* half,
                                 float lane_k) const {
    glm::vec3 t; float h;
    const glm::vec3 p = edgePoint(e, s, &t, &h);
    if (dir < 0.0f) t = -t;
    const float lane = laneOffset(h, lane_k) + extra;
    const glm::vec3 right(-t.z, 0.0f, t.x);      // right of travel, y up
    if (tangent) *tangent = t;
    if (half) *half = h;
    return p + right * lane;
}

bool VehicleSystem::routeBetween(const RoadPt& from, const RoadPt& to,
                                 std::vector<Leg>& out) const {
    out.clear();
    const Edge& e0 = edges_[from.edge];
    const Edge& e1 = edges_[to.edge];
    const float s0 = e0.s[from.index], s1 = e1.s[to.index];
    if (from.edge == to.edge) {
        if (std::abs(s1 - s0) < 1.0f) return false;
        out.push_back({from.edge, s0, s1});
        return true;
    }
    // A* from both ends of e0 to either end of e1
    const int n = int(nodes_.size());
    std::vector<float> g(n, std::numeric_limits<float>::max());
    std::vector<int> parent_edge(n, -1), parent_node(n, -1);
    std::vector<char> closed(n, 0);
    auto heur = [&](int ni) {
        const glm::vec3& p = nodes_[ni].pos;
        const glm::vec3& q = e1.pts[to.index];
        return std::sqrt((p.x - q.x) * (p.x - q.x) + (p.z - q.z) * (p.z - q.z));
    };
    using QE = std::pair<float, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> open;
    g[e0.a] = s0;              parent_edge[e0.a] = -2;   // -2: from the start edge
    g[e0.b] = e0.len - s0;     parent_edge[e0.b] = -2;
    open.push({g[e0.a] + heur(e0.a), e0.a});
    open.push({g[e0.b] + heur(e0.b), e0.b});
    int goal = -1;
    int guard = 0;
    while (!open.empty() && guard++ < 200000) {
        const auto [f, ni] = open.top(); open.pop();
        if (closed[ni]) continue;
        closed[ni] = 1;
        if (ni == e1.a || ni == e1.b) { goal = ni; break; }
        for (int ei : nodes_[ni].edges) {
            const Edge& e = edges_[ei];
            const int nj = e.a == ni ? e.b : e.a;
            if (nj == ni || closed[nj]) continue;
            const float ng = g[ni] + e.len;
            if (ng < g[nj]) {
                g[nj] = ng; parent_edge[nj] = ei; parent_node[nj] = ni;
                open.push({ng + heur(nj), nj});
            }
        }
    }
    if (goal < 0) return false;
    // unwind: goal .. start
    std::vector<Leg> rev;
    int cur = goal;
    while (parent_edge[cur] >= 0) {
        const Edge& e = edges_[parent_edge[cur]];
        const int prev = parent_node[cur];
        // travelling prev -> cur along e
        const float sa = e.a == prev ? 0.0f : e.len;
        const float sb = e.a == prev ? e.len : 0.0f;
        rev.push_back({parent_edge[cur], sa, sb});
        cur = prev;
    }
    // first leg: from s0 to the start node `cur` along e0
    out.push_back({from.edge, s0, cur == e0.a ? 0.0f : e0.len});
    for (auto it = rev.rbegin(); it != rev.rend(); ++it) out.push_back(*it);
    // last leg: from the goal node along e1 to s1
    out.push_back({to.edge, goal == e1.a ? 0.0f : e1.len, s1});
    // drop zero-length legs
    out.erase(std::remove_if(out.begin(), out.end(), [](const Leg& l) {
        return std::abs(l.s_to - l.s_from) < 0.05f; }), out.end());
    return !out.empty();
}

// ── v34 junction control ─────────────────────────────────────────────
void VehicleSystem::buildSignals() {
    signals_.clear();
    node_signal_.assign(nodes_.size(), -1);
    int n_lights = 0, n_stops = 0;
    for (size_t ni = 0; ni < nodes_.size(); ++ni) {
        const Node& nd = nodes_[ni];
        if (nd.edges.size() < 3) continue;
        // the heading of each road leaving the node (a chord 8 m out)
        std::vector<glm::vec2> dirs;
        bool ok = true;
        for (int ei : nd.edges) {
            const Edge& e = edges_[size_t(ei)];
            if (e.a == e.b || e.len < 2.0f) { ok = false; break; }
            const bool at_a = e.a == int(ni);
            const float s = at_a ? std::min(8.0f, e.len) : std::max(e.len - 8.0f, 0.0f);
            const glm::vec3 p = edgePoint(e, s, nullptr, nullptr);
            glm::vec2 d(p.x - nd.pos.x, p.z - nd.pos.z);
            const float l = glm::length(d);
            if (l < 0.5f) { ok = false; break; }
            dirs.push_back(d / l);
        }
        if (!ok) continue;
        Signal sg;
        sg.node = int(ni);
        if (nd.edges.size() >= 4) {
            sg.kind = 1;
            const glm::vec2 axis = dirs[0];
            for (size_t k = 0; k < dirs.size(); ++k)
                if (std::abs(glm::dot(dirs[k], axis)) > 0.5f)
                    sg.group_a.push_back(nd.edges[k]);
            if (sg.group_a.size() == nd.edges.size()) continue;   // nothing crosses
            sg.phase = h01(uint32_t(ni), 0x99u) * kCycleS;
            ++n_lights;
        } else {
            sg.kind = 2;
            // the through pair is the two most opposite headings; the
            // third road is the stem
            float best = 1e9f; size_t bi = 0, bj = 1;
            for (size_t i = 0; i < dirs.size(); ++i)
                for (size_t j = i + 1; j < dirs.size(); ++j) {
                    const float d = glm::dot(dirs[i], dirs[j]);
                    if (d < best) { best = d; bi = i; bj = j; }
                }
            for (size_t k = 0; k < dirs.size(); ++k)
                if (k != bi && k != bj) sg.stems.push_back(nd.edges[k]);
            if (sg.stems.empty()) continue;
            ++n_stops;
        }
        node_signal_[ni] = int(signals_.size());
        signals_.push_back(std::move(sg));
    }
    std::cout << "[vehicle] junctions: " << n_lights << " with traffic lights, "
              << n_stops << " with a stop sign" << std::endl;
}

int VehicleSystem::lightState(const Signal& sg, int edge) const {
    const bool in_a = std::find(sg.group_a.begin(), sg.group_a.end(), edge) !=
                      sg.group_a.end();
    const float t = std::fmod(anim_t_ + sg.phase, kCycleS);
    if (in_a) return t < 14.0f ? 0 : t < 17.0f ? 1 : 2;
    return (t >= 18.0f && t < 32.0f) ? 0 : (t >= 32.0f && t < 35.0f) ? 1 : 2;
}

// The poles, heads, lamps and signs: boxes from the bar mesh, paint
// kind 5 (plain colour, emissive by `glow`), on the near-side kerb
// kStopLineM before the node of every controlled approach.
void VehicleSystem::emitSignals(const glm::vec3& camera_pos) {
    if (frame_.size() != size_t(kStreamCount)) return;
    auto box = [&](const glm::vec3& c, const glm::vec3& half, float yaw,
                   const glm::vec3& rgb, float glow) {
        const glm::mat4 M =
            glm::translate(glm::mat4(1.0f), c) *
            glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0, 1, 0)) *
            glm::scale(glm::mat4(1.0f), half);
        frame_[kMeshBar].push_back({M, glm::vec4(rgb, 0.0f),
                                    glm::vec4(5.0f, 0.0f, 0.0f, glow)});
    };
    const glm::vec3 kPole(0.36f, 0.37f, 0.39f), kHead(0.07f, 0.07f, 0.08f);
    const glm::vec3 kLampOn[3] = {{1.0f, 0.08f, 0.05f}, {1.0f, 0.72f, 0.10f}, {0.10f, 1.0f, 0.35f}};
    const glm::vec3 kLampOff[3] = {{0.28f, 0.04f, 0.03f}, {0.30f, 0.22f, 0.04f}, {0.03f, 0.26f, 0.09f}};
    auto head = [&](const glm::vec3& c, const glm::vec3& t, float yaw, int state) {
        box(c, {0.16f, 0.50f, 0.14f}, yaw, kHead, 0.0f);
        for (int k = 0; k < 3; ++k) {
            const bool on = (state == 2 && k == 0) || (state == 1 && k == 1) ||
                            (state == 0 && k == 2);
            box(c + glm::vec3(0.0f, 0.30f - 0.30f * float(k), 0.0f) - t * 0.16f,
                {0.10f, 0.10f, 0.06f}, yaw, on ? kLampOn[k] : kLampOff[k],
                on ? 1.8f : 0.0f);
        }
    };
    for (const Signal& sg : signals_) {
        const glm::vec3& np = nodes_[size_t(sg.node)].pos;
        const float dx = np.x - camera_pos.x, dz = np.z - camera_pos.z;
        if (dx * dx + dz * dz > kSignalDrawM * kSignalDrawM) continue;
        for (int ei : nodes_[size_t(sg.node)].edges) {
            if (sg.kind == 2 &&
                std::find(sg.stems.begin(), sg.stems.end(), ei) == sg.stems.end())
                continue;
            const Edge& e = edges_[size_t(ei)];
            const bool at_b = e.b == sg.node;
            const float dir = at_b ? 1.0f : -1.0f;      // travel toward the node
            const float s_node = at_b ? e.len : 0.0f;
            const float s = glm::clamp(s_node - dir * kStopLineM, 0.0f, e.len);
            glm::vec3 t; float h;
            const glm::vec3 pc = edgePoint(e, s, &t, &h);
            if (dir < 0.0f) t = -t;
            const glm::vec3 right(-t.z, 0.0f, t.x);
            const glm::vec3 p = pc + right * (h + 0.8f);
            const float yaw = std::atan2(t.x, t.z);
            if (sg.kind == 1) {
                const int st = lightState(sg, ei);
                box(p + glm::vec3(0.0f, 2.2f, 0.0f), {0.06f, 2.2f, 0.06f}, yaw, kPole, 0.0f);
                head(p + glm::vec3(0.0f, 3.4f, 0.0f), t, yaw, st);
                // the arm over the road, with a second head above the
                // inner lane
                const float arm = h * 0.5f + 0.8f;
                box(p + glm::vec3(0.0f, 4.4f, 0.0f) - right * (0.5f * arm),
                    {0.5f * arm, 0.05f, 0.05f}, yaw, kPole, 0.0f);
                head(p - right * arm + glm::vec3(0.0f, 3.85f, 0.0f), t, yaw, st);
            } else {
                box(p + glm::vec3(0.0f, 1.15f, 0.0f), {0.035f, 1.15f, 0.035f}, yaw, kPole, 0.0f);
                box(p + glm::vec3(0.0f, 2.45f, 0.0f) + t * 0.012f, {0.40f, 0.40f, 0.012f},
                    yaw, {0.92f, 0.92f, 0.90f}, 0.05f);
                box(p + glm::vec3(0.0f, 2.45f, 0.0f) - t * 0.012f, {0.34f, 0.34f, 0.012f},
                    yaw, {0.78f, 0.04f, 0.04f}, 0.25f);
            }
        }
    }
}

// ── vehicles ─────────────────────────────────────────────────────────
void VehicleSystem::parkAt(Vehicle& v, const RoadPt& rp) {
    const Edge& e = edges_[rp.edge];
    const float s = e.s[rp.index];
    glm::vec3 t; float h;
    // parked at the curb, facing along the edge
    v.pos = lanePos(e, s, 1.0f, 0.0f, &t, &h);
    // the curb spot; on a two-lane road that is the outer lane, so
    // the car goes onto the verge instead (where the kerbed ones are)
    const float extra = std::max(0.0f, h - 1.1f - laneOffset(h, 1.0f)) +
                        (twoLane(h) ? kVergeM : 0.0f);
    v.pos = lanePos(e, s, 1.0f, extra, &t, &h);
    v.yaw = std::atan2(t.x, t.z);
    v.parked = true;
    v.speed = 0.0f;
    v.route.clear();
    v.leg = -1;
    v.s = s;
    if (v.claim >= 0 && v.claim < int(node_claim_.size()) &&
        node_claim_[v.claim] == (&v - vehicles_.data())) {
        node_claim_[v.claim] = -1;
    }
    v.claim = -1;
}

void VehicleSystem::parkVerge(Vehicle& v, const RoadPt& rp, float dir) {
    const Edge& e = edges_[rp.edge];
    const float s = e.s[rp.index];
    glm::vec3 t; float h;
    edgePoint(e, s, &t, &h);
    const float extra = std::max(0.0f, h - 1.1f - laneOffset(h, 1.0f));
    v.pos = lanePos(e, s, dir, extra + kVergeM, &t, &h);
    v.yaw = std::atan2(t.x, t.z);
    v.parked = true;
    v.kerbed = true;
    v.speed = 0.0f;
    v.steer = 0.0f;
    v.route.clear();
    v.leg = -1;
    v.s = s;
    v.claim = -1;
}

void VehicleSystem::strollPlace(Stroll& st) const {
    const Edge& e = edges_[size_t(st.edge)];
    glm::vec3 t; float h;
    const glm::vec3 p = edgePoint(e, st.s, &t, &h);
    if (st.dir < 0.0f) t = -t;
    const glm::vec3 right(-t.z, 0.0f, t.x);
    st.pos = p + right * std::max(0.6f, h - st.side);
    st.yaw = std::atan2(t.x, t.z);
}

bool VehicleSystem::strollStartNear(const glm::vec3& near, float radius,
                                    uint32_t seed, Stroll& out) const {
    if (!loaded()) return false;
    RoadPt rp;
    if (!nearestRoadPt(near, radius, rp)) return false;
    uint32_t st = seed;
    out.edge = rp.edge;
    out.s = edges_[size_t(rp.edge)].s[size_t(rp.index)];
    out.dir = rnd01(st) < 0.5f ? 1.0f : -1.0f;
    out.side = 0.35f + 0.35f * rnd01(st);
    strollPlace(out);
    return true;
}

bool VehicleSystem::strollAdvance(Stroll& st, float ds, uint32_t& rng) const {
    if (st.edge < 0 || st.edge >= int(edges_.size())) return false;
    const Edge* e = &edges_[size_t(st.edge)];
    st.s += ds * st.dir;
    for (int guard = 0; guard < 4; ++guard) {
        const bool past_end = st.dir > 0.0f ? st.s > e->len : st.s < 0.0f;
        if (!past_end) break;
        const float over = st.dir > 0.0f ? st.s - e->len : -st.s;
        const int node = st.dir > 0.0f ? e->b : e->a;
        int   next = st.edge;
        float ndir = -st.dir;                          // dead end: back
        if (node >= 0 && node < int(nodes_.size()) &&
            nodes_[size_t(node)].edges.size() > 1) {
            const std::vector<int>& ne = nodes_[size_t(node)].edges;
            for (int tries = 0; tries < 6; ++tries) {
                const int c = ne[size_t(int(rnd01(rng) * float(ne.size())) % ne.size())];
                if (c != st.edge) { next = c; break; }
            }
            if (next != st.edge)
                ndir = edges_[size_t(next)].a == node ? 1.0f : -1.0f;
        }
        st.edge = next;
        e = &edges_[size_t(next)];
        st.dir = ndir;
        st.s = ndir > 0.0f ? std::min(over, e->len)
                           : std::max(e->len - over, 0.0f);
    }
    st.s = glm::clamp(st.s, 0.0f, e->len);
    strollPlace(st);
    return true;
}

void VehicleSystem::clearVehicles() {
    vehicles_.clear();
    std::fill(node_claim_.begin(), node_claim_.end(), -1);
}

int VehicleSystem::spawnCar(uint32_t seed, const glm::vec3& near_pos) {
    if (!loaded()) return -1;
    RoadPt rp;
    if (!nearestRoadPt(near_pos, kCurbSearchM, rp)) return -1;
    Vehicle v;
    v.seed = seed;
    const float r = h01(seed, 0x21u);
    v.type = r < 0.42f ? kSedan : r < 0.72f ? kSuv : r < 0.88f ? kPickup : kCyber;
    v.color = v.type == kCyber ? glm::vec3(0.72f, 0.73f, 0.75f) : paletteColor(seed);
    v.ambient = false;
    parkAt(v, rp);
    vehicles_.push_back(v);
    return int(vehicles_.size()) - 1;
}

bool VehicleSystem::dispatch(int car, const glm::vec3& dest) {
    if (car < 0 || car >= int(vehicles_.size())) return false;
    Vehicle& v = vehicles_[car];
    RoadPt from, to;
    if (!nearestRoadPt(v.pos, kCurbSearchM, from)) return false;
    if (!nearestRoadPt(dest, kCurbSearchM, to)) return false;
    std::vector<Leg> route;
    if (!routeBetween(from, to, route)) return false;
    v.route = std::move(route);
    v.leg = 0;
    v.s = v.route[0].s_from;
    v.parked = false;
    v.speed = 0.0f;
    v.lane = 1; v.lane_x = 1.0f; v.stop_t = 0.0f;
    return true;
}

bool VehicleSystem::parked(int car) const {
    return car >= 0 && car < int(vehicles_.size()) && vehicles_[car].parked;
}
glm::vec3 VehicleSystem::position(int car) const {
    return (car >= 0 && car < int(vehicles_.size())) ? vehicles_[car].pos
                                                     : glm::vec3(0.0f);
}
float VehicleSystem::yaw(int car) const {
    return (car >= 0 && car < int(vehicles_.size())) ? vehicles_[car].yaw : 0.0f;
}
glm::vec3 VehicleSystem::stepOut(int car) const {
    if (car < 0 || car >= int(vehicles_.size())) return glm::vec3(0.0f);
    const Vehicle& v = vehicles_[car];
    const glm::vec3 right(-std::cos(v.yaw), 0.0f, std::sin(v.yaw));
    return v.pos + right * 1.6f;
}
glm::vec3 seatWorld(const glm::vec3& pos, float yaw, const glm::vec3& seat) {
    // model (x fwd, z right) -> world: the same quarter turn emit() uses
    const float a = yaw - 1.5707963f;
    const float ca = std::cos(a), sa = std::sin(a);
    return {pos.x + seat.x * ca + seat.z * sa,
            pos.y + seat.y - 0.45f,
            pos.z - seat.x * sa + seat.z * ca};
}

glm::vec3 VehicleSystem::seatPos(int car, int seat) const {
    if (car < 0 || car >= int(vehicles_.size())) return glm::vec3(0.0f);
    const Vehicle& v = vehicles_[size_t(car)];
    const Cabin& cb = cabinOf(v.type);
    if (cb.seats.empty()) return v.pos;
    const glm::vec3& s = cb.seats[size_t(std::max(0, std::min(seat, int(cb.seats.size()) - 1)))];
    return seatWorld(v.pos, v.yaw, s);
}

void VehicleSystem::occupants(const glm::vec3& cam, float radius,
                              std::vector<Occupant>& out) const {
    out.clear();
    const float r2 = radius * radius;
    const bool school_hours = (tod_hours_ > 6.8f && tod_hours_ < 9.2f) ||
                              (tod_hours_ > 14.0f && tod_hours_ < 16.5f);
    for (const Vehicle& v : vehicles_) {
        if (!v.ambient || v.kerbed) continue;          // kerbed: empty
        const float dx = v.pos.x - cam.x, dz = v.pos.z - cam.z;
        if (dx * dx + dz * dz > r2) continue;
        const Cabin& cb = cabinOf(v.type);
        for (size_t k = 0; k < cb.seats.size(); ++k) {
            bool in = (k == 0);                            // the driver
            if (k == 1) in = h01(v.seed, 0x61u) < (v.type == kPolice ? 0.55f : 0.40f);
            if (k >= 2) {
                if (v.type == kSchoolBus) in = school_hours && h01(v.seed, 0x70u + uint32_t(k)) < 0.7f;
                else if (v.type == kFireEngine) in = true;
                else in = h01(v.seed, 0x70u + uint32_t(k)) < 0.15f;
            }
            if (!in) continue;
            Occupant o;
            o.pos = seatWorld(v.pos, v.yaw, cb.seats[k]);
            o.yaw = v.yaw;
            o.seed = hash32(v.seed * 31u + uint32_t(k) * 977u + 0x99u);
            o.seat = int(k);
            o.type = v.type;
            out.push_back(o);
        }
    }
}

void VehicleSystem::recall(int car, const glm::vec3& pos) {
    if (car < 0 || car >= int(vehicles_.size())) return;
    RoadPt rp;
    if (nearestRoadPt(pos, kCurbSearchM, rp)) parkAt(vehicles_[car], rp);
}

bool VehicleSystem::randomDestination(const glm::vec3& from, float min_m,
                                      float max_m, uint32_t seed,
                                      RoadPt& out, int comp,
                                      int min_comp_edges) const {
    if (edges_.empty()) return false;
    const std::vector<int>* pool = nullptr;
    if (comp >= 0 && comp < int(comp_edges_.size())) pool = &comp_edges_[size_t(comp)];
    const int npool = pool ? int(pool->size()) : int(edges_.size());
    if (npool <= 0) return false;
    uint32_t st = seed;
    auto dist_to = [&](const glm::vec3& p) {
        return std::sqrt((p.x - from.x) * (p.x - from.x) +
                         (p.z - from.z) * (p.z - from.z));
    };
    // 1. inside the window, on a big enough island
    for (int tries = 0; tries < 80; ++tries) {
        const int k = int(rnd01(st) * float(npool)) % npool;
        const int ei = pool ? (*pool)[size_t(k)] : k;
        if (!pool && min_comp_edges > 1 && ei < int(edge_comp_.size()) &&
            int(comp_edges_[size_t(edge_comp_[size_t(ei)])].size()) < min_comp_edges)
            continue;
        const Edge& e = edges_[size_t(ei)];
        const int idx = int(rnd01(st) * float(e.pts.size())) % int(e.pts.size());
        const float d = dist_to(e.pts[size_t(idx)]);
        if (d >= min_m && d <= max_m) { out = {ei, idx}; return true; }
    }
    // 2. on a small island the window is often empty: the farthest of
    //    a few draws, as long as it is a drive at all
    if (pool) {
        float best = 40.0f; bool found = false;
        for (int tries = 0; tries < 40; ++tries) {
            const int ei = (*pool)[size_t(int(rnd01(st) * float(npool)) % npool)];
            const Edge& e = edges_[size_t(ei)];
            for (const int idx : {0, int(e.pts.size()) - 1,
                                  int(rnd01(st) * float(e.pts.size())) % int(e.pts.size())}) {
                const float d = dist_to(e.pts[size_t(idx)]);
                if (d > best && d <= max_m * 1.5f) { best = d; out = {ei, idx}; found = true; }
            }
        }
        return found;
    }
    return false;
}

void VehicleSystem::spawnAmbient(const glm::vec3& camera_pos, uint32_t seed) {
    RoadPt rp;
    bool ok = false;
    // 1. the camera's OWN island, most of the time, when it is a real
    //    network: a car spawned there drives past the camera; one on
    //    another island is a car for someone else
    RoadPt cam_rp;
    if (h01(seed, 0x11u) < 0.6f && nearestRoadPt(camera_pos, 500.0f, cam_rp) &&
        size_t(cam_rp.edge) < edge_comp_.size()) {
        const int comp = edge_comp_[size_t(cam_rp.edge)];
        if (comp >= 0 && comp < int(comp_edges_.size()) &&
            int(comp_edges_[size_t(comp)].size()) >= 6)
            ok = randomDestination(camera_pos, kSpawnMin, kSpawnMax, seed, rp, comp);
    }
    // 2. anywhere in the window, big islands first
    if (!ok) ok = randomDestination(camera_pos, kSpawnMin, kSpawnMax, seed ^ 0x33u, rp, -1, 3);
    if (!ok) ok = randomDestination(camera_pos, kSpawnMin, kSpawnMax, seed ^ 0x77u, rp);
    if (!ok) return;
    // never onto another car
    {
        const glm::vec3& q = edges_[size_t(rp.edge)].pts[size_t(rp.index)];
        for (const Vehicle& o : vehicles_) {
            const float dx = o.pos.x - q.x, dz = o.pos.z - q.z;
            if (dx * dx + dz * dz < kSpawnGapM * kSpawnGapM) return;
        }
    }
    // a semi wants a wide road
    Vehicle v;
    v.seed = seed;
    v.ambient = true;
    const float r = h01(seed, 0x31u);
    const bool school_hours = (tod_hours_ > 6.8f && tod_hours_ < 9.2f) ||
                              (tod_hours_ > 14.0f && tod_hours_ < 16.5f);
    if (r < 0.30f)      v.type = kSedan;
    else if (r < 0.52f) v.type = kSuv;
    else if (r < 0.64f) v.type = kPickup;
    else if (r < 0.70f) v.type = kCyber;
    else if (r < 0.80f) v.type = kSemi;
    else if (r < 0.87f) v.type = school_hours ? kSchoolBus : kSedan;
    else if (r < 0.93f) v.type = kPolice;
    else if (r < 0.97f) v.type = kAmbulance;
    else                v.type = kFireEngine;
    if (v.type == kSemi && edges_[rp.edge].half[rp.index] < 4.5f) v.type = kPickup;
    switch (v.type) {
    case kCyber:      v.color = {0.72f, 0.73f, 0.75f}; break;
    case kSemi:       v.color = {0.12f, 0.13f, 0.18f}; break;
    case kPolice:     v.color = {0.05f, 0.05f, 0.06f}; break;
    case kAmbulance:  v.color = {0.93f, 0.93f, 0.92f}; break;
    case kSchoolBus:  v.color = {0.95f, 0.70f, 0.10f}; break;
    case kFireEngine: v.color = {0.75f, 0.05f, 0.04f}; break;
    default:          v.color = paletteColor(seed); break;
    }
    const float lr = h01(seed, 0x41u);
    v.lights = (v.type == kPolice && lr < 0.30f) ||
               (v.type == kAmbulance && lr < 0.50f) ||
               (v.type == kFireEngine && lr < 0.60f);
    // ── KERBED: parked on the verge, empty, for a few minutes ────────
    // The cars a street has standing along it.  Not the buses and the
    // emergency fleet, which are only ever seen on the move.
    const bool kerb = h01(seed, 0x45u) < kParkedShare &&
                      v.type != kSchoolBus && v.type != kFireEngine &&
                      v.type != kAmbulance && v.type != kSemi;
    if (kerb) {
        parkVerge(v, rp, h01(seed, 0x47u) < 0.5f ? 1.0f : -1.0f);
        v.lights = false;
        v.idle_t = 120.0f + 600.0f * h01(seed, 0x46u);
        vehicles_.push_back(v);
        return;
    }
    parkAt(v, rp);
    RoadPt to;
    if (randomDestination(v.pos, kDestMin, kDestMax, seed ^ 0x5bd1e995u, to,
                          edge_comp_[size_t(rp.edge)])) {
        std::vector<Leg> route;
        if (routeBetween(rp, to, route)) {
            v.route = std::move(route);
            v.leg = 0;
            v.s = v.route[0].s_from;
            v.parked = false;
            v.speed = specs()[v.type].vmax * 0.6f;
            // spawned moving: either lane (the big ones outer)
            v.lane = (specs()[v.type].L > 7.0f || h01(seed, 0x3Cu) < 0.6f) ? 1 : 0;
            v.lane_x = float(v.lane); v.stop_t = 0.0f;
        }
    }
    vehicles_.push_back(v);
}

void VehicleSystem::tick(Vehicle& v, int vi, float dt, float speed_scale,
                         const std::unordered_map<int, std::vector<int>>& on_edge) {
    const Spec& sp = specs()[v.type];
    if (v.parked) {
        if (v.ambient) {
            v.idle_t -= dt;
            if (v.idle_t <= 0.0f) {
                RoadPt from, to;
                if (nearestRoadPt(v.pos, kCurbSearchM, from) &&
                    randomDestination(v.pos, kDestMin, kDestMax,
                                      v.seed ^ uint32_t(anim_t_ * 977.0f), to,
                                      edge_comp_[size_t(from.edge)])) {
                    std::vector<Leg> route;
                    if (routeBetween(from, to, route)) {
                        v.route = std::move(route);
                        v.leg = 0; v.s = v.route[0].s_from;
                        v.parked = false; v.speed = 0.0f;
                        v.kerbed = false;              // pulls out
                        v.lane = 1; v.lane_x = 1.0f; v.stop_t = 0.0f;
                    }
                }
                v.idle_t = 6.0f;
            }
        }
        return;
    }
    if (v.leg < 0 || v.leg >= int(v.route.size())) { v.parked = true; return; }
    const Leg& leg = v.route[v.leg];
    const Edge& e = edges_[leg.edge];
    const float dir = leg.s_to >= leg.s_from ? 1.0f : -1.0f;
    const float remain_leg = std::max(0.0f, (leg.s_to - v.s) * dir);
    float remain_total = remain_leg;
    for (int k = v.leg + 1; k < int(v.route.size()); ++k)
        remain_total += std::abs(v.route[k].s_to - v.route[k].s_from);

    // ── speed limits ────────────────────────────────────────────────
    float vlim = sp.vmax * (v.lights ? 1.25f : 1.0f);
    // stop at the destination
    vlim = std::min(vlim, std::sqrt(2.0f * 3.0f * std::max(remain_total, 0.0f)) + 0.3f);
    // corners: heading change into the next leg
    const bool has_next = v.leg + 1 < int(v.route.size());
    const int node_ahead = dir > 0.0f ? e.b : e.a;
    if (has_next && remain_leg < 22.0f) {
        glm::vec3 t0, t1;
        edgePoint(e, leg.s_to, &t0, nullptr);
        if (dir < 0.0f) t0 = -t0;
        const Leg& nl = v.route[v.leg + 1];
        const Edge& ne = edges_[nl.edge];
        edgePoint(ne, nl.s_from, &t1, nullptr);
        if (nl.s_to < nl.s_from) t1 = -t1;
        const float c = glm::clamp(glm::dot(t0, t1), -1.0f, 1.0f);
        if (c < 0.85f) vlim = std::min(vlim, c < 0.3f ? 3.5f : 5.5f);
        // ── junction control (v35): lights, a stop sign, or the claim
        // WHO CLAIMS.  Only traffic that has to give way queues for the
        // node: at a stop sign that is the STEM alone (the through road
        // has priority, and making it queue was half of the deadlock),
        // at traffic lights nobody (the phases already exclude), and at
        // an uncontrolled multi-way node everybody, as before.
        bool need_claim = node_ahead >= 0 && nodes_[node_ahead].edges.size() > 2;
        const int sgi = node_ahead >= 0 && node_ahead < int(node_signal_.size())
                            ? node_signal_[node_ahead] : -1;
        if (sgi >= 0 && signals_[sgi].kind == 1) {
            // TRAFFIC LIGHTS: stop at the line on red; on yellow only
            // if there is room to; inside the box, go on through.
            const int st = lightState(signals_[sgi], leg.edge);
            const float to_line = remain_leg - kStopLineM;
            const bool inside = remain_leg < kStopLineM - 2.0f;
            const bool commit = st == 1 && to_line < 1.2f * v.speed + 1.0f;
            if (st != 0 && !inside && !commit && to_line > -0.5f)
                vlim = std::min(vlim, std::sqrt(2.0f * 4.5f * std::max(to_line, 0.0f)));
            need_claim = false;
        } else if (sgi >= 0 && signals_[sgi].kind == 2) {
            const bool stem =
                std::find(signals_[sgi].stems.begin(), signals_[sgi].stems.end(),
                          leg.edge) != signals_[sgi].stems.end();
            if (stem) {
                // STOP SIGN: halt at the line, stand a moment, then
                // take the junction under the claim.
                const float to_line = remain_leg - kStopLineM;
                if (v.stop_t < kStopWaitS && to_line > -0.5f) {
                    vlim = std::min(vlim, std::sqrt(2.0f * 4.5f * std::max(to_line, 0.0f)));
                    if (to_line < 2.0f && v.speed < 0.6f) v.stop_t += dt;
                }
            } else {
                need_claim = false;          // the through road: priority
            }
        }
        // THE CLAIM: one vehicle through the node at a time.  The
        // holder is validated every frame -- it must still be a live,
        // moving vehicle that names THIS node as its claim -- because
        // the ambient ring pops slots off vehicles_, and an index left
        // pointing past the end used to block the node for ever.  A
        // claim also expires (kClaimHoldS), and a vehicle held here too
        // long (kYieldS) takes the node regardless: whatever goes
        // wrong, traffic never stands still permanently.
        if (need_claim) {
            int& claim = node_claim_[node_ahead];
            const bool held =
                claim >= 0 && claim != vi && claim < int(vehicles_.size()) &&
                !vehicles_[claim].parked && !vehicles_[claim].dormant &&
                vehicles_[claim].claim == node_ahead &&
                node_claim_age_[size_t(node_ahead)] < kClaimHoldS;
            if (!held || v.block_t > kYieldS) {
                if (claim != vi) node_claim_age_[size_t(node_ahead)] = 0.0f;
                claim = vi;
                v.claim = node_ahead;
                v.block_t = 0.0f;
            } else {
                v.block_t += dt;
                vlim = remain_leg > 9.0f ? std::min(vlim, 3.0f) : 0.0f;
            }
        } else {
            v.block_t = 0.0f;
        }
    }
    // release a claim once the node is well behind
    if (v.claim >= 0 && v.claim != node_ahead) {
        const glm::vec3& np = nodes_[v.claim].pos;
        const float ddx = np.x - v.pos.x, ddz = np.z - v.pos.z;
        if (ddx * ddx + ddz * ddz > 16.0f * 16.0f) {
            if (node_claim_[v.claim] == vi) node_claim_[v.claim] = -1;
            v.claim = -1;
        }
    }
    // ── LANES (v34) ─────────────────────────────────────────────────
    // On a wide road: pull out to pass a slower car when the inner
    // lane is clear ahead and behind; drift back to the outer lane
    // when that is clear.  No changes in the last 30 m before a node.
    float h_here = 0.0f;
    edgePoint(e, v.s, nullptr, &h_here);
    const bool two_lane = twoLane(h_here);
    v.lane_cool = std::max(0.0f, v.lane_cool - dt);
    auto it = on_edge.find(leg.edge);
    if (!two_lane) {
        v.lane = 1;
    } else if (v.lane_cool <= 0.0f && remain_leg > 30.0f && v.speed > 1.0f) {
        float gap[2] = {1e9f, 1e9f}, back[2] = {1e9f, 1e9f}, osp[2] = {99.0f, 99.0f};
        if (it != on_edge.end()) {
            for (int oi : it->second) {
                if (oi == vi) continue;
                const Vehicle& o = vehicles_[oi];
                if (o.parked || o.leg < 0 || o.leg >= int(o.route.size())) continue;
                const Leg& ol = o.route[o.leg];
                if ((ol.s_to >= ol.s_from ? 1.0f : -1.0f) != dir) continue;
                const float g = (o.s - v.s) * dir;
                for (int l = 0; l < 2; ++l) {
                    if (std::abs(o.lane_x - float(l)) > 0.55f) continue;
                    if (g > 0.0f) { if (g < gap[l]) { gap[l] = g; osp[l] = o.speed; } }
                    else if (-g < back[l]) back[l] = -g;
                }
            }
        }
        const int cur = v.lane, oth = 1 - v.lane;
        const bool big = sp.L > 7.0f;                 // buses, trucks: outer
        if (gap[cur] < 22.0f && osp[cur] < sp.vmax - 2.0f &&
            gap[oth] > 30.0f && back[oth] > 14.0f && !(big && oth == 0)) {
            v.lane = oth; v.lane_cool = 6.0f;
        } else if (cur == 0 && gap[1] > 35.0f && back[1] > 12.0f) {
            v.lane = 1; v.lane_cool = 6.0f;
        }
    }
    v.lane_x += glm::clamp(float(v.lane) - v.lane_x, -0.7f * dt, 0.7f * dt);
    // the car ahead on this edge, same direction (and lane)
    if (it != on_edge.end()) {
        float best_gap = 1e9f;
        for (int oi : it->second) {
            if (oi == vi) continue;
            const Vehicle& o = vehicles_[oi];
            if (o.parked || o.leg < 0 || o.leg >= int(o.route.size())) continue;
            const Leg& ol = o.route[o.leg];
            const float odir = ol.s_to >= ol.s_from ? 1.0f : -1.0f;
            if (odir != dir) continue;
            // v34: the other lane is not in the way
            if (two_lane && std::abs(o.lane_x - v.lane_x) > 0.55f) continue;
            const float gap = (o.s - v.s) * dir;
            if (gap > 0.0f && gap < best_gap) best_gap = gap;
        }
        if (best_gap < 1e8f) {
            const float free = best_gap - kCarLenGap - 0.5f * sp.L;
            vlim = std::min(vlim, std::max(0.0f, free / 1.4f));
        }
    }
    // ── integrate ───────────────────────────────────────────────────
    const float scale_vis = std::min(speed_scale, kVisualCapMs / std::max(sp.vmax, 1.0f));
    const float dv = vlim - v.speed;
    v.speed += glm::clamp(dv, -kBrake * dt, kAccel * dt);
    v.speed = std::max(v.speed, 0.0f);
    const float ds = v.speed * scale_vis * dt;
    v.wheel += ds / sp.wheel_r;
    v.s += ds * dir;
    if ((leg.s_to - v.s) * dir <= 0.0f) {
        // next leg, or arrived
        if (has_next) {
            ++v.leg;
            v.s = v.route[v.leg].s_from;
            v.stop_t = 0.0f;
            v.block_t = 0.0f;
        } else {
            RoadPt rp;
            // park at the destination: the nearest road point to where
            // the last leg ends
            const glm::vec3 end = edgePoint(e, leg.s_to, nullptr, nullptr);
            if (nearestRoadPt(end, 30.0f, rp)) parkAt(v, rp);
            else { v.parked = true; v.speed = 0.0f; v.leg = -1; v.route.clear(); }
            v.idle_t = v.ambient ? 4.0f + 8.0f * h01(v.seed, 0x77u) : 0.0f;
            return;
        }
    }
    const Leg& cl = v.route[v.leg];
    const Edge& ce = edges_[cl.edge];
    const float cdir = cl.s_to >= cl.s_from ? 1.0f : -1.0f;
    glm::vec3 t;
    const glm::vec3 np = lanePos(ce, v.s, cdir, 0.0f, &t, nullptr, v.lane_x);
    const float target_yaw = std::atan2(t.x, t.z);
    float dyaw = target_yaw - v.yaw;
    while (dyaw > 3.14159265f) dyaw -= 6.2831853f;
    while (dyaw < -3.14159265f) dyaw += 6.2831853f;
    const float turn = glm::clamp(dyaw, -2.5f * dt, 2.5f * dt);
    v.yaw += turn;
    v.steer += (glm::clamp(dyaw * 1.5f, -0.5f, 0.5f) - v.steer) * std::min(1.0f, 6.0f * dt);
    v.pos.x = np.x; v.pos.z = np.z;
    v.pos.y = np.y;
}

void VehicleSystem::update(float delta_t, float speed_scale,
                           const glm::vec3& camera_pos,
                           const GroundQueryFn& ground) {
    if (!loaded()) return;
    const float dt = std::min(delta_t, 0.1f);
    anim_t_ += dt;
    camera_pos_ = camera_pos;
    // ── ambient traffic around the camera ───────────────────────────
    {
        // ambient vehicles that drifted out of range are rebuilt IN
        // PLACE (owned cars are referenced by index, so nothing moves)
        // how many this ring can carry: by the road inside it
        ring_timer_ -= dt;
        if (ring_timer_ <= 0.0f) {
            ring_timer_ = 2.0f;
            const float road_m = roadLengthNear(camera_pos, kSpawnMax);
            ambient_now_ = road_m < 30.0f ? 0
                         : std::min(ambient_target_,
                                    std::max(kAmbientMin, int(road_m / kRoadPerCarM)));
        }
        // Owned cars are referenced by index, so an ambient slot is
        // never erased: one the ring no longer needs goes DORMANT
        // (parked a long way off, which every radius test then skips)
        // and is the first slot the next spawn reuses.
        int alive = 0;
        for (const Vehicle& v : vehicles_)
            if (v.ambient && !v.dormant) ++alive;
        std::vector<size_t> spare;
        for (size_t i = 0; i < vehicles_.size(); ++i) {
            Vehicle& v = vehicles_[i];
            if (!v.ambient) continue;
            if (v.dormant) { spare.push_back(i); continue; }
            const float dx = v.pos.x - camera_pos.x, dz = v.pos.z - camera_pos.z;
            if (dx * dx + dz * dz > kRecycleM * kRecycleM) {
                const uint32_t seed = hash32(v.seed * 3u + 0x51u + uint32_t(anim_t_ * 13.0f));
                if (v.claim >= 0 && node_claim_[v.claim] == int(i)) node_claim_[v.claim] = -1;
                v.claim = -1;
                if (alive > ambient_now_) {
                    v.dormant = true;
                    v.parked = true; v.kerbed = false; v.lights = false;
                    v.route.clear(); v.leg = -1; v.speed = 0.0f;
                    v.pos = glm::vec3(1.0e7f, 0.0f, 1.0e7f);
                    spare.push_back(i);
                    --alive;
                    continue;
                }
                // rebuild in place
                const size_t before = vehicles_.size();
                spawnAmbient(camera_pos, seed);
                if (vehicles_.size() > before) {
                    vehicles_[i] = vehicles_.back();
                    vehicles_.pop_back();
                }
            }
        }
        int budget = 2;
        while (alive < ambient_now_ && budget-- > 0) {
            const size_t before = vehicles_.size();
            spawnAmbient(camera_pos, hash32(rng_ += 0x9E3779B9u));
            if (vehicles_.size() == before) break;
            if (!spare.empty()) {
                vehicles_[spare.back()] = vehicles_.back();
                vehicles_.pop_back();
                spare.pop_back();
            }
            ++alive;
        }
    }
    // v35: age every node claim once a frame (see kClaimHoldS)
    if (node_claim_age_.size() != node_claim_.size())
        node_claim_age_.assign(node_claim_.size(), 0.0f);
    for (float& t : node_claim_age_) t += dt;
    // ── who is on which edge (car following) ────────────────────────
    std::unordered_map<int, std::vector<int>> on_edge;
    const float sim2 = kSimRadius * kSimRadius;
    for (size_t i = 0; i < vehicles_.size(); ++i) {
        const Vehicle& v = vehicles_[i];
        if (v.parked || v.leg < 0 || v.leg >= int(v.route.size())) continue;
        const float dx = v.pos.x - camera_pos.x, dz = v.pos.z - camera_pos.z;
        if (dx * dx + dz * dz > sim2) continue;
        on_edge[v.route[v.leg].edge].push_back(int(i));
    }
    // ── tick ────────────────────────────────────────────────────────
    for (size_t i = 0; i < vehicles_.size(); ++i) {
        Vehicle& v = vehicles_[i];
        const float dx = v.pos.x - camera_pos.x, dz = v.pos.z - camera_pos.z;
        const float d2 = dx * dx + dz * dz;
        if (d2 > sim2) continue;
        tick(v, int(i), dt, speed_scale, on_edge);
        if (ground && d2 < kClampRadius * kClampRadius) {
            float gy; glm::vec3 gn;
            if (ground(v.pos.x, v.pos.z, v.pos.y, gy, gn) &&
                std::abs(gy - v.pos.y) < 4.0f) {
                v.pos.y = gy + 0.02f;
            }
        }
    }
    // ── emit the frame ──────────────────────────────────────────────
    if (frame_.size() != size_t(kStreamCount)) frame_.resize(kStreamCount);
    for (auto& f : frame_) f.clear();
    const float draw2 = kDrawRadius * kDrawRadius;
    for (const Vehicle& v : vehicles_) {
        const float dx = v.pos.x - camera_pos.x, dz = v.pos.z - camera_pos.z;
        if (dx * dx + dz * dz > draw2) continue;
        emit(v);
    }
    emitSignals(camera_pos);
    dbg_timer_ += dt;
    if (dbg_timer_ > 10.0f) {
        dbg_timer_ = 0.0f;
        size_t moving = 0, amb = 0, kerbed = 0, near = 0, stuck = 0;
        for (const auto& v : vehicles_) {
            if (v.dormant) continue;
            if (!v.parked) ++moving;
            if (v.ambient) ++amb;
            if (v.kerbed) ++kerbed;
            if (!v.parked && v.speed < 0.2f) ++stuck;
            const float dx = v.pos.x - camera_pos.x, dz = v.pos.z - camera_pos.z;
            if (dx * dx + dz * dz < 300.0f * 300.0f) ++near;
        }
        std::cout << "[vehicle] " << vehicles_.size() << " slots (" << amb
                  << " ambient of " << ambient_now_ << " wanted, " << kerbed
                  << " kerbed), " << moving << " moving (" << stuck
                  << " standing), " << near
                  << " within 300 m" << std::endl;
    }
}

void VehicleSystem::emit(const Vehicle& v) {
    const Spec& sp = specs()[v.type];
    // The hull's FORWARD is model +x (the side profile's x), and yaw
    // is atan2(t.x, t.z) -- the angle that turns +z onto the road.
    // Turning the model a quarter back first puts +x on the road;
    // without it every vehicle stood across its lane.
    const glm::mat4 root =
        glm::translate(glm::mat4(1.0f), v.pos) *
        glm::rotate(glm::mat4(1.0f), v.yaw - 1.5707963f, glm::vec3(0, 1, 0));
    const float seed01 = h01(v.seed, 0x91u);
    const float blink = v.lights ? 1.0f + anim_t_ : 0.0f;
    // hull: the mesh is in metres with the ground at y = 0
    frame_[v.type].push_back({root, glm::vec4(v.color, 0.0f),
                              glm::vec4(0.0f, float(v.type), seed01, blink)});
    // the same hull again in the glass pass (kind 3: windows only)
    if (s_glass_pipeline_)
        frame_[kStreamGlass0 + v.type].push_back(
            {root, glm::vec4(v.color, 0.0f),
             glm::vec4(3.0f, float(v.type), seed01, blink)});
    // the interior (kind 4): seats in one of four trims by the seed
    {
        const float tr = h01(v.seed, 0x81u);
        const glm::vec3 trim = tr < 0.45f ? glm::vec3(0.10f, 0.10f, 0.11f)
                             : tr < 0.70f ? glm::vec3(0.30f, 0.30f, 0.32f)
                             : tr < 0.88f ? glm::vec3(0.46f, 0.36f, 0.26f)
                                          : glm::vec3(0.62f, 0.58f, 0.50f);
        frame_[kMeshInterior0 + v.type].push_back(
            {root, glm::vec4(trim, 0.0f),
             glm::vec4(4.0f, float(v.type), seed01, blink)});
    }
    // wheels
    for (size_t k = 0; k < sp.wheel_x.size(); ++k) {
        const float dual = k < sp.wheel_dual.size() ? sp.wheel_dual[k] : 1.0f;
        const bool front = k == 0;
        for (int side = 0; side < 2; ++side) {
            const float z = (side == 0 ? -1.0f : 1.0f) * (sp.wheel_z - 0.5f * sp.wheel_w * (dual - 1.0f));
            glm::mat4 M = root *
                glm::translate(glm::mat4(1.0f), glm::vec3(sp.wheel_x[k], sp.wheel_r, z));
            if (front && v.steer != 0.0f)
                M = M * glm::rotate(glm::mat4(1.0f), -v.steer, glm::vec3(0, 1, 0));
            M = M * glm::rotate(glm::mat4(1.0f), -v.wheel, glm::vec3(0, 0, 1)) *
                glm::scale(glm::mat4(1.0f),
                           glm::vec3(sp.wheel_r, sp.wheel_r, 0.5f * sp.wheel_w * dual));
            frame_[kMeshWheel].push_back({M, glm::vec4(0.04f, 0.04f, 0.045f, 0.0f),
                                          glm::vec4(1.0f, float(v.type), seed01, 0.0f)});
        }
    }
    if (sp.bar) {
        const glm::mat4 M = root *
            glm::translate(glm::mat4(1.0f), sp.bar_at) *
            glm::scale(glm::mat4(1.0f), glm::vec3(0.5f * sp.bar_size.x, 0.07f, 0.5f * sp.bar_size.y));
        frame_[kMeshBar].push_back({M, glm::vec4(0.15f, 0.15f, 0.18f, 0.0f),
                                    glm::vec4(2.0f, float(v.type), seed01, blink)});
    }
}

void VehicleSystem::collectShadowGeometry(ActorShadowGeometry& out) const {
    out.positions.clear(); out.indices.clear();
    if (!loaded() || !s_pipeline_) return;
    // Glass duplicates hull triangles, so submit each hull only once.
    for (size_t stream = 0; stream < frame_.size() && stream < kStreamGlass0; ++stream) {
        const int mesh = streamMesh(int(stream));
        if (mesh < 0 || mesh >= int(s_meshes_.size())) continue;
        for (const auto& instance : frame_[stream])
            out.append(s_meshes_[mesh].shadow, [&](const glm::vec3& p) {
                return glm::vec3(instance.xform * glm::vec4(p, 1.0f));
            });
    }
}

void VehicleSystem::draw(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const er::DescriptorSetList& desc_sets,
    const std::shared_ptr<er::ImageView>& color_view,
    const std::shared_ptr<er::ImageView>& depth_view,
    const glm::uvec2& buffer_size,
    const std::vector<std::shared_ptr<er::ImageView>>& gbuffer,
    bool glass_only, bool opaque_only) {
    const bool deferred = !gbuffer.empty();
    if (deferred && (gbuffer.size() != 4 || !s_gbuf_pipeline_)) return;
    if (!loaded() || !s_pipeline_ || !s_device_ || !s_pipeline_layout_) return;
    if (!color_view || !depth_view) return;
    size_t total = 0;
    for (const auto& f : frame_) total += f.size();
    if (!total) return;
    for (const auto& ds : desc_sets) if (!ds) return;
    // ── upload every stream into one buffer ─────────────────────────
    {
        const uint32_t need = uint32_t(total);
        if (!s_inst_buf_ || need > s_inst_capacity_) {
            uint32_t cap = s_inst_capacity_ ? s_inst_capacity_ : 1024u;
            while (cap < need) cap *= 2u;
            if (s_inst_buf_) s_inst_buf_->destroy(s_device_);
            s_inst_buf_ = std::make_shared<er::BufferInfo>();
            er::Helper::createBuffer(
                s_device_,
                SET_2_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT, TRANSFER_DST_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
                0, s_inst_buf_->buffer, s_inst_buf_->memory,
                std::source_location::current(),
                uint64_t(cap) * sizeof(PartInstance), nullptr);
            s_inst_capacity_ = cap;
        }
        uint64_t off = 0;
        for (const auto& f : frame_) {
            if (!f.empty()) {
                s_device_->updateBufferMemory(
                    s_inst_buf_->memory, uint64_t(f.size()) * sizeof(PartInstance),
                    f.data(), off);
            }
            off += uint64_t(f.size()) * sizeof(PartInstance);
        }
    }
    er::RenderingAttachmentInfo color_att;
    color_att.image_view = color_view;
    color_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    color_att.load_op = er::AttachmentLoadOp::LOAD;
    color_att.store_op = er::AttachmentStoreOp::STORE;
    er::RenderingAttachmentInfo depth_att;
    depth_att.image_view = depth_view;
    depth_att.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_att.load_op = er::AttachmentLoadOp::LOAD;
    depth_att.store_op = er::AttachmentStoreOp::STORE;
    er::RenderingInfo ri{};
    ri.render_area_offset = {0, 0};
    ri.render_area_extent = {buffer_size.x, buffer_size.y};
    ri.layer_count = 1;
    ri.view_mask = 0;
    ri.color_attachments = {color_att};
    if (deferred) {
        ri.color_attachments.clear();
        for (const auto& view : gbuffer) {
            auto att = color_att; att.image_view = view;
            ri.color_attachments.push_back(att);
        }
    }
    ri.depth_attachments = {depth_att};
    cmd_buf->beginDynamicRendering(ri);
    std::vector<er::Viewport> viewports(1);
    std::vector<er::Scissor> scissors(1);
    viewports[0].x = 0; viewports[0].y = 0;
    viewports[0].width = float(buffer_size.x);
    viewports[0].height = float(buffer_size.y);
    viewports[0].min_depth = 0.0f; viewports[0].max_depth = 1.0f;
    scissors[0].offset = {0, 0};
    scissors[0].extent = {buffer_size.x, buffer_size.y};
    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, deferred ? s_gbuf_pipeline_ : s_pipeline_);
    cmd_buf->setViewports(viewports, 0, 1);
    cmd_buf->setScissors(scissors, 0, 1);
    cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS,
                                s_pipeline_layout_, desc_sets);
    std::vector<uint64_t> offs = {0, 0, 0};
    uint32_t first = 0;
    bool glass_bound = false;
    for (size_t m = 0; m < frame_.size(); ++m) {
        const uint32_t n = uint32_t(frame_[m].size());
        const int mi = streamMesh(int(m));
        if (mi < 0 || mi >= int(s_meshes_.size())) { first += n; continue; }
        if (glass_only && int(m) < kStreamGlass0) { first += n; continue; }
        if ((deferred || opaque_only) && int(m) >= kStreamGlass0) break;
        if (int(m) >= kStreamGlass0 && !glass_bound) {
            if (!s_glass_pipeline_) break;     // no glass pass this build
            cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, s_glass_pipeline_);
            cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS,
                                        s_pipeline_layout_, desc_sets);
            glass_bound = true;
        }
        const Mesh& mesh = s_meshes_[size_t(mi)];
        if (n && mesh.pos && mesh.nrm && mesh.idx) {
            std::vector<std::shared_ptr<er::Buffer>> vbs = {
                mesh.pos->buffer, mesh.nrm->buffer, s_inst_buf_->buffer};
            cmd_buf->bindVertexBuffers(0, vbs, offs);
            cmd_buf->bindIndexBuffer(mesh.idx->buffer, 0, er::IndexType::UINT32);
            cmd_buf->drawIndexed(mesh.count, n, 0, 0, first);
        }
        first += n;
    }
    cmd_buf->endDynamicRendering();
}

void VehicleSystem::destroy(const std::shared_ptr<er::Device>& device) {
    (void)device;
    vehicles_.clear();
    frame_.clear();
    edges_.clear();
    nodes_.clear();
    pt_grid_.clear();
    node_grid_.clear();
    node_claim_.clear();
}

}  // namespace game_object
}  // namespace engine
