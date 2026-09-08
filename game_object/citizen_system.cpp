#include "citizen_system.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <filesystem>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <glm/gtc/matrix_transform.hpp>

#include "json.hpp"   // vendored at third_parties/tinygltf/json.hpp
#include "drawable_object.h"   // PcgInstanceRegistry (furniture)
#include "helper/engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace er = engine::renderer;

namespace engine {
namespace game_object {
namespace {

// ── enums kept as ints in the structs so the header stays light ──────
enum Activity {
    kActIdle = 0, kActWalk, kActSit, kActCook, kActBrowse, kActCare,
    kActDeskWork, kActPlay, kActSleepish,
    // Washing up at the kitchen sink.  APPENDED, not inserted: these
    // values are what activityOf() maps city_sim's schedule strings
    // onto, and renumbering them would silently re-label every
    // activity in every existing map.
    kActWash
};
// Which piece of furniture a home anchor resolved to.  kAnchorNone
// is "this house has none of it" — the caller falls back.
enum AnchorKind { kAnchorNone = 0, kAnchorBed, kAnchorStove, kAnchorSeat,
                  kAnchorSink };
enum Duty {
    kDutyResident = 0, kDutyDoctor, kDutyNurse, kDutyTeacher,
    kDutyOfficial, kDutyPolice, kDutyFire, kDutyChef, kDutyWaiter,
    kDutyShop, kDutyOffice, kDutyHomemaker, kDutyStudent, kDutyInfant,
    kDutyRetiree, kDutyCommuter, kDutyChildcare, kDutyWorker
};

// What citizen.frag paints on a part, from its unit-space position.
// Kind 0 is a plain box (the far tier, the mid tier, hair, heels).
enum PartKind {
    kPartPlain = 0, kPartHead, kPartTorso, kPartPelvis, kPartSleeve,
    kPartForearm, kPartHand, kPartThigh, kPartShin, kPartShoe
};

// ── A SKELETON ───────────────────────────────────────────────────────
// Sixteen joints in a tree.  Each has a REST PIVOT in body space
// (metres; y up, +z the front, the root between the feet) and a local
// rotation as Euler angles -- x flexion (the hinge every limb had
// before), z abduction (an arm held out from the body), y twist (the
// spine turning against the pelvis, the head looking round).  A part
// hangs off one joint at an offset from its pivot, so a rotation at
// the hip carries the thigh, the shin, the foot and the shoe.  The
// fine tier solves this per person per frame; the seven-cube tier
// keeps its four flat hinges.  Parents precede children in the enum,
// so one forward pass solves the tree.
enum Joint {
    jPelvis = 0, jSpine, jNeck, jHead,
    jShoulderL, jElbowL, jWristL, jShoulderR, jElbowR, jWristR,
    jHipL, jKneeL, jAnkleL, jHipR, jKneeR, jAnkleR, jCount
};
static const int kJointParent[jCount] = {
    -1, jPelvis, jSpine, jNeck,
    jSpine, jShoulderL, jElbowL, jSpine, jShoulderR, jElbowR,
    jPelvis, jHipL, jKneeL, jPelvis, jHipR, jKneeR};

struct Skeleton {
    glm::vec3 pivot[jCount];      // rest pivots, body space
    glm::vec3 rot[jCount];        // local (x flex, y twist, z abduct)
    glm::mat4 world[jCount];      // solved: joint space -> world
};

// twist, then abduction, then flexion (flexion innermost, so a hinge
// reads the same as before whatever else is set)
glm::mat4 eulerJoint(const glm::vec3& r) {
    glm::mat4 M(1.0f);
    if (r.y != 0.0f) M = glm::rotate(M, r.y, glm::vec3(0, 1, 0));
    if (r.z != 0.0f) M = glm::rotate(M, r.z, glm::vec3(0, 0, 1));
    if (r.x != 0.0f) M = glm::rotate(M, r.x, glm::vec3(1, 0, 0));
    return M;
}

void solveSkeleton(Skeleton& sk, const glm::mat4& root) {
    for (int j = 0; j < jCount; ++j) {
        const int pj = kJointParent[j];
        const glm::vec3 off =
            pj < 0 ? sk.pivot[j] : sk.pivot[j] - sk.pivot[pj];
        const glm::mat4& parent = pj < 0 ? root : sk.world[pj];
        sk.world[j] = parent * glm::translate(glm::mat4(1.0f), off) *
                      eulerJoint(sk.rot[j]);
    }
}

// ── A LOOK ───────────────────────────────────────────────────────────
// Skin, hair and an outfit per person, hashed from the person id so it
// is the same every frame and every session.  The wardrobe is a short
// list of everyday combinations (the two reference photos first: a
// grey henley over blue jeans with heels, a white short-sleeve shirt
// over navy slacks with black shoes) and the uniformed duties override
// it.  Everything here is a colour or a flag the shader reads; no
// geometry differs between looks except long hair and heels.
struct Look {
    glm::vec3 skin{0.9f, 0.75f, 0.62f};
    glm::vec3 hair{0.2f, 0.12f, 0.08f};
    glm::vec3 top{0.6f, 0.6f, 0.62f};
    glm::vec3 bottom{0.22f, 0.33f, 0.55f};
    glm::vec3 shoe{0.1f, 0.1f, 0.1f};
    bool female = false;
    bool long_sleeve = false;
    bool long_hair = false;
    bool heels = false;
    bool stick = false;        // senior: a walking stick, right hand
    int  top_style = 0;        // 0 tee, 1 buttoned shirt, 2 henley
    int  bottom_style = 0;     // 0 jeans, 1 slacks
};

float lookRand(uint32_t id, uint32_t salt) {
    uint32_t x = id * 0x9E3779B1u + salt * 0x85EBCA6Bu;
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return float(x & 0xFFFFFFu) / 16777216.0f;
}

// rgb (0..1) -> one float the shader unpacks exactly (24 bits < 2^24).
float packRGB(const glm::vec3& c) {
    const glm::vec3 q = glm::clamp(c, 0.0f, 1.0f) * 255.0f;
    return float(int(q.r + 0.5f) * 65536 + int(q.g + 0.5f) * 256 +
                 int(q.b + 0.5f));
}

int ageEnum(const std::string& a) {
    if (a == "child") return 1;
    if (a == "toddler" || a == "infant") return 2;
    if (a == "senior") return 3;
    return 0;
}

// Who keeps a car (v25): an adult or a senior whose duty takes them out
// of the house -- about two in three of them.  Decided once, from the
// person id, so it is the same every session.
bool wantsCar(int duty, int age, uint32_t pid) {
    if (age == 1 || age == 2) return false;
    switch (duty) {
    case kDutyStudent: case kDutyInfant: case kDutyHomemaker:
    case kDutyChildcare: case kDutyRetiree:
        return false;
    default: break;
    }
    return lookRand(pid, 0x0CA2u) < 0.62f;
}

int dutyEnum(const std::string& d) {
    if (d == "doctor") return kDutyDoctor;
    if (d == "nurse") return kDutyNurse;
    if (d == "teacher") return kDutyTeacher;
    if (d == "city_official") return kDutyOfficial;
    if (d == "police_officer") return kDutyPolice;
    if (d == "firefighter") return kDutyFire;
    if (d == "chef") return kDutyChef;
    if (d == "waiter") return kDutyWaiter;
    if (d == "shop_worker") return kDutyShop;
    if (d == "office_worker") return kDutyOffice;
    if (d == "homemaker") return kDutyHomemaker;
    if (d == "student") return kDutyStudent;
    if (d == "infant") return kDutyInfant;
    if (d == "retiree") return kDutyRetiree;
    if (d == "commuter") return kDutyCommuter;
    if (d == "childcare_worker") return kDutyChildcare;
    return kDutyWorker;
}

glm::vec3 dutyColor(int duty) {
    static const glm::vec3 table[] = {
        {0.6f, 0.6f, 0.6f},                    // resident
        {0.92f, 0.95f, 0.98f}, {0.95f, 0.75f, 0.80f},
        {0.25f, 0.60f, 0.35f}, {0.16f, 0.22f, 0.45f},
        {0.10f, 0.16f, 0.38f}, {0.75f, 0.15f, 0.10f},
        {0.92f, 0.90f, 0.85f}, {0.30f, 0.28f, 0.30f},
        {0.85f, 0.55f, 0.15f}, {0.35f, 0.42f, 0.55f},
        {0.55f, 0.35f, 0.60f}, {0.85f, 0.75f, 0.20f},
        {0.90f, 0.70f, 0.75f}, {0.55f, 0.55f, 0.55f},
        {0.40f, 0.35f, 0.30f}, {0.70f, 0.55f, 0.75f},
        {0.80f, 0.50f, 0.20f},
    };
    if (duty < 0 || duty >= int(sizeof(table) / sizeof(table[0])))
        return table[0];
    return table[duty];
}

Look lookOf(int pid, int duty, int age) {
    const uint32_t id = uint32_t(pid);
    auto r = [&](uint32_t s) { return lookRand(id, s); };
    Look L;
    L.female = r(1) < 0.5f;
    static const glm::vec3 kSkins[4] = {
        {0.93f, 0.78f, 0.66f}, {0.85f, 0.66f, 0.52f},
        {0.66f, 0.46f, 0.32f}, {0.42f, 0.28f, 0.20f}};
    L.skin = kSkins[int(r(2) * 4.0f) & 3];
    static const glm::vec3 kHairs[5] = {
        {0.08f, 0.06f, 0.05f}, {0.25f, 0.15f, 0.09f},
        {0.45f, 0.28f, 0.14f}, {0.72f, 0.55f, 0.30f},
        {0.50f, 0.20f, 0.10f}};
    L.hair = kHairs[int(r(3) * 5.0f) % 5];
    if (age == 3) {                              // senior: grey to white
        L.hair = glm::mix(glm::vec3(0.62f), glm::vec3(0.88f), r(4));
    }
    L.long_hair = L.female ? r(5) < 0.75f : r(5) < 0.08f;
    L.heels = L.female && age == 0 && r(6) < 0.45f;
    struct Outfit { glm::vec3 top, bottom, shoe; bool ls; int ts, bs; };
    static const Outfit kWardrobe[10] = {
        // grey henley, blue jeans, nude heels (reference photo 1)
        {{0.62f, 0.62f, 0.64f}, {0.22f, 0.33f, 0.55f}, {0.85f, 0.78f, 0.70f}, true,  2, 0},
        // white short-sleeve shirt, navy slacks, black shoes (photo 2)
        {{0.95f, 0.95f, 0.94f}, {0.12f, 0.15f, 0.25f}, {0.06f, 0.06f, 0.06f}, false, 1, 1},
        {{0.55f, 0.70f, 0.85f}, {0.55f, 0.48f, 0.36f}, {0.30f, 0.20f, 0.12f}, true,  1, 1},
        {{0.20f, 0.40f, 0.30f}, {0.10f, 0.10f, 0.12f}, {0.08f, 0.08f, 0.08f}, false, 0, 0},
        {{0.75f, 0.20f, 0.18f}, {0.30f, 0.30f, 0.32f}, {0.10f, 0.10f, 0.10f}, false, 0, 1},
        {{0.96f, 0.93f, 0.85f}, {0.16f, 0.24f, 0.42f}, {0.35f, 0.22f, 0.14f}, true,  1, 0},
        {{0.15f, 0.18f, 0.30f}, {0.24f, 0.30f, 0.45f}, {0.90f, 0.90f, 0.90f}, true,  0, 0},
        {{0.85f, 0.55f, 0.25f}, {0.20f, 0.20f, 0.22f}, {0.15f, 0.12f, 0.10f}, false, 0, 0},
        {{0.30f, 0.55f, 0.60f}, {0.40f, 0.35f, 0.30f}, {0.20f, 0.15f, 0.10f}, false, 1, 1},
        {{0.92f, 0.80f, 0.85f}, {0.55f, 0.62f, 0.75f}, {0.90f, 0.85f, 0.80f}, true,  2, 0},
    };
    const Outfit& o = kWardrobe[int(r(7) * 10.0f) % 10];
    L.top = o.top; L.bottom = o.bottom; L.shoe = o.shoe;
    L.long_sleeve = o.ls; L.top_style = o.ts; L.bottom_style = o.bs;
    switch (duty) {
    case kDutyDoctor:                            // white coat
        L.top = {0.95f, 0.95f, 0.96f}; L.long_sleeve = true;
        L.top_style = 1; L.bottom = {0.20f, 0.22f, 0.30f};
        L.bottom_style = 1; break;
    case kDutyNurse:                             // scrubs
        L.top = {0.45f, 0.62f, 0.80f}; L.bottom = L.top;
        L.long_sleeve = false; L.top_style = 0; L.bottom_style = 1;
        L.shoe = {0.9f, 0.9f, 0.9f}; L.heels = false; break;
    case kDutyPolice:
        L.top = {0.12f, 0.16f, 0.30f}; L.bottom = {0.10f, 0.12f, 0.22f};
        L.long_sleeve = true; L.top_style = 1; L.bottom_style = 1;
        L.shoe = {0.05f, 0.05f, 0.05f}; L.heels = false; break;
    case kDutyFire:
        L.top = {0.60f, 0.15f, 0.12f}; L.bottom = {0.15f, 0.15f, 0.17f};
        L.long_sleeve = true; L.top_style = 1; L.heels = false; break;
    case kDutyChef:
        L.top = {0.95f, 0.95f, 0.95f}; L.bottom = {0.20f, 0.20f, 0.22f};
        L.long_sleeve = true; L.top_style = 1; L.heels = false; break;
    case kDutyWaiter:
        L.top = {0.95f, 0.95f, 0.95f}; L.bottom = {0.08f, 0.08f, 0.09f};
        L.long_sleeve = true; L.top_style = 1; L.bottom_style = 1;
        L.shoe = {0.05f, 0.05f, 0.05f}; break;
    case kDutyOfficial: case kDutyOffice:        // office wear
        L.top = r(8) < 0.5f ? glm::vec3(0.95f, 0.95f, 0.94f)
                            : glm::vec3(0.70f, 0.78f, 0.88f);
        L.bottom = {0.16f, 0.18f, 0.26f}; L.long_sleeve = true;
        L.top_style = 1; L.bottom_style = 1; L.shoe = {0.08f, 0.06f, 0.05f};
        break;
    case kDutyInfant:
        L.top = {0.95f, 0.75f, 0.35f}; L.bottom = {0.35f, 0.55f, 0.80f};
        L.long_sleeve = false; L.top_style = 0; L.heels = false; break;
    default: break;
    }
    if (age == 1 || age == 2) {
        // children: tees, shorts on most, sneakers on half; a toddler
        // in short sleeves whatever the wardrobe said
        L.heels = false;
        L.long_hair = L.female && r(9) < 0.6f;
        L.top_style = L.top_style == 1 ? 0 : L.top_style;
        if (r(11) < (age == 2 ? 0.7f : 0.5f)) L.bottom_style = 2;
        if (age == 2) { L.long_sleeve = false; L.top_style = 0; }
        if (r(13) < 0.5f) L.shoe = {0.90f, 0.90f, 0.88f};
    }
    L.stick = age == 3 && r(12) < 0.4f;
    return L;
}

float parseClock(const std::string& hhmm) {
    if (hhmm.size() < 4) return 0.0f;
    return float(std::atoi(hhmm.substr(0, 2).c_str())) * 60.0f +
           float(std::atoi(hhmm.substr(3).c_str()));
}

int activityOf(const std::string& act, const std::string& place,
               int duty) {
    if (act == "work" || act == "shift_start") {
        if (duty == kDutyDoctor || duty == kDutyNurse) return kActCare;
        if (duty == kDutyOffice || duty == kDutyOfficial)
            return kActDeskWork;
        if (duty == kDutyTeacher) return kActIdle;    // stands, teaching
        return kActIdle;
    }
    if (act == "class" || act == "lunch" || act == "dinner_out")
        return kActSit;
    if (act == "breakfast" || act == "dinner" ||
        (act == "chores" && place == "home")) return kActCook;
    if (act == "groceries" || act == "errand" || act == "outing" ||
        act == "daycare_dropoff") return kActBrowse;
    if (act == "daycare" || act == "play") return kActPlay;
    if (act == "walk" || act == "leisure") return kActWalk;
    if (act == "rest" || act == "pickup" || act == "home" ||
        act == "off_work" || act == "shift_end") return kActIdle;
    return kActIdle;
}

// Avalanche an index before it is used as a hash key (murmur3's
// finalizer).  h01 alone is fine over a whole map but has strong LOCAL
// structure on sequential inputs: feeding it raw house indices put
// 2.7% homemakers in a 4000-house town where the thresholds ask for
// 12%, and clumped whole contiguous streets onto one occupation.
// Mixing first holds every band to its intended share at any town size.
uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x85EBCA6Bu;
    x ^= x >> 13; x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

float h01(uint32_t a, uint32_t b) {
    uint32_t h = 2166136261u;
    h = (h ^ a) * 16777619u;
    h = (h ^ b) * 16777619u;
    return float(h) / 4294967296.0f;
}

constexpr float kClockScale = 60.0f;      // 1 real s = 1 game minute
// EVERYONE is simulated, but at two rates.  Inside kNearSimRadius a
// person gets the full per-frame tick — walking interpolation, yaw,
// gait phase, door gestures.  Beyond it, a round-robin ring
// (kFarSimPerFrame persons/frame) snaps each person to their current
// schedule step's anchor: at 150k+ persons the full walk tick every
// frame is real milliseconds, and a commuter 3 km away lerping between
// anchors is indistinguishable from one teleported there once a
// second.  Approach them and the near tick resumes mid-schedule.
//
// Rendering stays tiered by distance: full seven-box articulation
// inside kDetailRadius (capped at kMaxDetailed nearest so a packed
// district can't explode the draw count); a single person-box out to
// kShowRadius (10 km per the design brief).  kMinAngular skips far
// persons whose box would land under ~a third of a pixel — and it is
// only the FLOOR of an adaptive threshold: with a whole town's
// population in view the far tier caps itself at kMaxFarParts draws by
// raising the cutoff until the crowd fits (nearest, tallest figures
// keep priority as the angular test is height/dist).
constexpr float kNearSimRadius = 700.0f;
constexpr size_t kFarSimPerFrame = 8192;
constexpr float kDetailRadius = 300.0f;
// ── THE BUDGETS THAT WERE REALLY DRAW-CALL BUDGETS ──────────────────
// 320 detailed figures and 12000 far parts were sized against a draw
// path that issued one pushConstants + one drawIndexed PER BOX: 320
// figures is 2240 calls and the far tier another 12000, which is
// already an unreasonable number of draws for gameplay markers.  The
// stream is instanced now — the entire visible population is ONE
// drawIndexed — so the limits can be what the GEOMETRY can carry
// rather than what the command buffer could.
//
// A box is 12 triangles.  The whole 119k population as far-tier boxes
// is 1.4 M triangles in one call, which is less than a single tree
// band, and the per-frame cost is the 80 B/part memcpy into the
// instance buffer: 9.5 MB for everyone at once, and far less in
// practice because kShowRadius and the angular test still apply.
//
// kMaxFarParts stays as a SAFETY VALVE, not a look decision: the
// adaptive far_thresh_ below still raises the cutoff if a vantage ever
// puts more than this in view, so a pathological frame degrades
// instead of stalling.  At 200k it never engages for this map's
// population, which is the point — "visible when the ground under them
// is visible" is the rule now.
constexpr size_t kMaxDetailed = 4096;
constexpr size_t kMaxFarParts = 200000;   // far-tier safety valve
// The fine tier (rounded sixteen-part figures with clothes and faces):
// the nearest kMaxFine persons inside kFineRadius.  800 triangles a
// part (both windings), ~17 parts a person -- 192 of them is ~2.6 M
// triangles at the worst, which is what caps it.
// v3: ~5.6k triangles a figure on the skinned meshes -- 256 of them
// is ~1.4 M, so the band widens.
constexpr float  kFineRadius = 90.0f;
// The CHARACTER MESHES (v33): the nearest kMaxNpc fine persons inside
// this radius are drawn as baked scans instead of the parts puppet.
constexpr float  kNpcRadius = 60.0f;
// Driving (v25): a trip longer than this on foot is taken by car when
// the person's car is parked within kBoardR of them.
constexpr float  kDriveMinM = 320.0f;
constexpr float  kBoardR = 60.0f;
constexpr size_t kMaxFine = 256;
constexpr float kShowRadius = 10000.0f;   // 10 km
// Sub-pixel cutoff, not a budget: 0.0003 rad of height is about a third
// of a pixel at 1440p, so this only drops people who could not put a
// fragment on the screen anyway.  It used to be the FLOOR of a
// threshold the far-part budget kept ratcheting upward, which is what
// made whole crowds disappear from a hilltop view; with the budget
// effectively lifted it is once again just the sub-pixel test it reads
// as.
constexpr float kMinAngular = 0.0002f;    // height/dist cutoff (floor)
constexpr float kGroundClampRadius = 400.0f;
// ...but the exact clamp is a TERRAIN QUERY, the one thing in this
// tick that does not scale, and a household is 3-5 people now.  In a
// dense village 400 m can hold a few thousand residents, so the exact
// clamps are budgeted per frame.
//
// TWO tiers, because a budget alone starves people: the loop runs in
// index order, so a fixed budget taken from index 0 would hand the
// same winners an exact clamp every frame forever and leave everyone
// else on the slow ring — including someone standing 30 m from the
// camera in full articulation.  So anyone inside kAlwaysClampR is
// clamped unconditionally (they are the ones you can see the ground
// under), and the rest share kNearClampPerFrame from a start index
// that resumes where the previous frame ran out — so the ring cycles
// in ceil(residents_in_ring / kNearClampPerFrame) frames, which at
// ordinary density is one.  The budget is a spike guard for a dense
// village centre, not the common path.
constexpr float  kAlwaysClampR      = 70.0f;
constexpr size_t kNearClampPerFrame = 1536;
// The clock runs 60x real time, so schedule-accurate commuting needs
// faster-than-life legs: at kWalkTimeScale 6 a 500 m commute costs one
// game hour (about a real minute of visible walking) instead of six.
// Full 60x would be teleport-sprinting; 6x reads as "people getting
// places" while staying watchable up close.
constexpr float kWalkTimeScale = 6.0f;

// Wrap period for the real-time POSE clock (CitizenSystem::anim_t_).
// 8*pi seconds: every pose frequency below is a multiple of 0.5 rad/s,
// and 0.5 * 8*pi = 4*pi is a whole number of cycles — so the wrap
// never pops a limb.  Wrapping at all is what keeps the sine arguments
// in a precise part of the float range over a long session.
constexpr float kAnimWrap = 25.13274123f;   // 8 * pi
// Staggered height refresh for everyone the near tier did not reach.
// Sized against the population this system now carries (~4 residents
// per house): at 1024 a 228k-person town took ~220 frames to come
// round, long enough that a far commuter's height visibly snaps.
constexpr size_t kFarClampPerFrame = 4096;
// How many houses per grid cell get promoted to destinations by
// synthesizeResidents (first four workplaces, the rest shops).  Lives
// out here because a LOCAL class may not declare a static data member
// (MSVC C2258 / [class.local]) — the cell picker is a local struct.
constexpr int kCellAnchors = 6;
// ...and how many of a school cell's picks actually become schools.
constexpr int kSchoolsPerCell = 4;
// House avoidance.  kHouseBlockR is the radius of the disc a house is
// treated as while steering — the archetypes run ~11-14 m across at the
// shipped scale, so ~6.5 m plus a shoulder keeps a walker off the walls
// without carving a wide berth through a tight street.  kAvoidLook is
// how far ahead the walker checks, kAvoidCell the spatial-hash cell.
constexpr float kHouseBlockR = 7.0f;
constexpr float kAvoidLook   = 9.0f;
constexpr float kAvoidCell   = 48.0f;
// HOUSEHOLD SIZE.  A house holds a FAMILY, not a lone occupant: the
// population is house_count x [kHouseholdMin, kHouseholdMax] drawn per
// house, so a 57k-house town carries ~230k people rather than 57k.
constexpr int kHouseholdMin = 3;
constexpr int kHouseholdMax = 5;
// Rosette radius for a household at home and out in the yard.  Houses
// generate around a ~8 m footprint, so 2.2 m keeps five people spread
// but still inside the shell; outdoors they can use the whole garden.
// ── FURNITURE (see harvestFurniture / furnitureAnchor) ──────────────
// How far from a house centre a placed object still counts as being
// INSIDE that house.  Village houses here sit 30-70 m apart and their
// room decals land within a couple of metres of the shell, so 12 m
// binds every one of them and never reaches a neighbour's.
// 25 m, up from 12: 12 m was sized for the distance BETWEEN houses,
// not for the size of one.  A chair in the far corner of a large
// promoted workplace is 12-15 m from its own house's centre and was
// dropped on the floor by the radius test, so that building had no
// seats and everyone in it sat in mid-air.  Nearest-centre is the real
// criterion and it is unambiguous at any radius here — village houses
// stand 30-70 m apart — so the cap only needs to be loose enough to
// cover one building's own footprint.
constexpr float kFurnitureBindR = 25.0f;
// Two to a double bed, offset either side of its centreline.  A
// household of four in a house with one bed used to be four figures
// stacked in the same volume; sharing in pairs is both what the
// geometry can carry and what a bedroom looks like.  Residents past
// the last bed do not lie down at all — see furnitureAnchor.
constexpr float kBedShareOffsetM = 0.32f;
// Mattress top above the bed instance's origin, and half the mattress
// length.  The sleeper's ROOT goes at the FOOT end: the lying pose in
// emitPerson tips the whole figure flat about the root, so the body
// extends from there toward the pillow.
constexpr float kBedTopY    = 0.55f;
constexpr float kBedHalfLen = 0.85f;
// If sleepers come out lying ACROSS their beds instead of along them,
// the bed model's yaw runs along its WIDTH: put half a turn
// (1.5707963f) here and every bed in the map lines up.  Nothing else
// reads this.
constexpr float kBedYawFix  = 0.0f;
// Where a cook stands relative to the cooktop: out in front of it by
// this much, turned back to face it.
constexpr float kStoveStand = 0.75f;
// Where someone washing up stands relative to the sink, and which way
// they face: the same arrangement as the cooktop — out in front of it,
// turned back into it.
constexpr float kSinkStand = 0.70f;
// Above this many buildings, furniture is not binned to buildings at
// all — see the note in harvestFurniture.  A city-json district is
// ~15 buildings; the synthesized path promotes ~10k houses and does
// not need the pass.
constexpr size_t kCivicScanMax = 512;
// Rooms per house are single digits; the next-hop matrix is rooms^2
// int16s, so this bounds it at 512 B per archetype and refuses to build
// one for anything pathological rather than quietly eating memory.
constexpr int kNavMaxRooms = 24;
// How near a waypoint counts as reached.  A doorway is ~0.9 m wide and
// a box person is 0.4 m across, so half a metre is "through it".
constexpr float kNavReachM = 0.55f;
// How far past a doorway the aim point sits — inside the room being
// ENTERED, so reaching it means having crossed the threshold rather
// than having arrived at it.
constexpr float kNavDoorPushM = 0.7f;
// How far outside its room a walker may stray before its carried route
// room is dropped and re-derived.  Big enough to absorb a walk that
// clips a corner or a crowd shove, small enough that a person moved to
// another building resyncs on the first tick there.
constexpr float kNavStickM = 1.5f;
// How far from a civic building's centre a placed object still counts
// as being inside it.  Sized for the district's largest plate (the
// 96 x 54 m mall) rather than a house.
constexpr float kCivicBindR = 70.0f;
// If seated citizens face AWAY from their table, the chair model's yaw
// points out of the seat rather than into it: put half a turn
// (3.14159265f) here and every chair in the map turns round.  Same
// escape hatch as kBedYawFix, same reason — the furniture library's
// axis convention is not this system's to assume.
constexpr float kSeatYawFix = 0.0f;
// The night window.  At home inside it everyone is asleep, whatever
// their schedule step nominally says — nobody cooks at 03:00.
constexpr float kNightStartMin = 22.5f * 60.0f;
constexpr float kNightEndMin   =  6.0f * 60.0f;

constexpr float kHomeSpreadR = 2.2f;
constexpr float kYardSpreadR = 4.0f;

}  // namespace

std::shared_ptr<er::Pipeline> CitizenSystem::s_gbuf_pipeline_;
std::shared_ptr<er::Pipeline> CitizenSystem::s_skin_gbuf_pipeline_;
std::shared_ptr<er::Pipeline> CitizenSystem::s_npc_gbuf_pipeline_;
ActorShadowGeometry CitizenSystem::s_shadow_cube_, CitizenSystem::s_shadow_tube_,
    CitizenSystem::s_shadow_blob_, CitizenSystem::s_shadow_ball_;
std::shared_ptr<er::PipelineLayout> CitizenSystem::s_pipeline_layout_;
std::shared_ptr<er::Pipeline>       CitizenSystem::s_pipeline_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_pos_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_nrm_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_idx_;
uint32_t                            CitizenSystem::s_cube_index_count_ = 0;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_round_pos_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_round_nrm_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_round_idx_;
uint32_t                            CitizenSystem::s_round_index_count_ = 0;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_tube_pos_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_tube_nrm_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_tube_idx_;
uint32_t                            CitizenSystem::s_tube_index_count_ = 0;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_ball_pos_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_ball_nrm_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_ball_idx_;
uint32_t                            CitizenSystem::s_ball_index_count_ = 0;
std::shared_ptr<er::Pipeline>       CitizenSystem::s_skin_pipeline_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_skin_buf_;
uint32_t                            CitizenSystem::s_skin_capacity_ = 0;
std::shared_ptr<er::Device>         CitizenSystem::s_device_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_inst_buf_;
uint32_t                            CitizenSystem::s_inst_capacity_ = 0;
CitizenSystem::NpcAsset             CitizenSystem::s_npc_[2];
bool                                CitizenSystem::s_npc_ready_ = false;
std::shared_ptr<er::PipelineLayout> CitizenSystem::s_npc_layout_;
std::shared_ptr<er::Pipeline>       CitizenSystem::s_npc_pipeline_;
std::shared_ptr<er::DescriptorSetLayout> CitizenSystem::s_npc_desc_layout_;
std::shared_ptr<er::DescriptorPool> CitizenSystem::s_npc_pool_;
std::shared_ptr<er::Sampler>        CitizenSystem::s_npc_sampler_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_npc_inst_buf_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_npc_palette_buf_;

namespace {
// The .npcmesh file npc_bake.py writes: "NPCM", u32 version, nv, ni,
// nj; nj x (i32 parent, 3 f32 bind pos); nv x pos, nrm (f32x3), uv
// (f32x2), joints (u8x4), weights (u8x4); ni x u32 index.
struct NpcVertex {
    glm::vec3 pos;
    glm::vec3 nrm;
    glm::vec2 uv;
    uint8_t   joints[4];
    uint8_t   weights[4];
};
bool loadNpcMesh(const std::string& path, std::vector<NpcVertex>& verts,
                 std::vector<uint32_t>& idx, glm::vec3* bind, int nj_want) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "NPCM") return false;
    uint32_t ver = 0, nv = 0, ni = 0, nj = 0;
    f.read(reinterpret_cast<char*>(&ver), 4);
    f.read(reinterpret_cast<char*>(&nv), 4);
    f.read(reinterpret_cast<char*>(&ni), 4);
    f.read(reinterpret_cast<char*>(&nj), 4);
    if (ver != 1 || int(nj) != nj_want || nv == 0 || ni == 0 ||
        nv > 2000000u || ni > 6000000u) return false;
    for (uint32_t j = 0; j < nj; ++j) {
        int32_t parent = 0;
        f.read(reinterpret_cast<char*>(&parent), 4);
        f.read(reinterpret_cast<char*>(&bind[j]), 12);
    }
    std::vector<glm::vec3> pos(nv), nrm(nv);
    std::vector<glm::vec2> uv(nv);
    std::vector<uint8_t> jn(nv * 4), wt(nv * 4);
    f.read(reinterpret_cast<char*>(pos.data()), std::streamsize(nv) * 12);
    f.read(reinterpret_cast<char*>(nrm.data()), std::streamsize(nv) * 12);
    f.read(reinterpret_cast<char*>(uv.data()), std::streamsize(nv) * 8);
    f.read(reinterpret_cast<char*>(jn.data()), std::streamsize(nv) * 4);
    f.read(reinterpret_cast<char*>(wt.data()), std::streamsize(nv) * 4);
    idx.resize(ni);
    f.read(reinterpret_cast<char*>(idx.data()), std::streamsize(ni) * 4);
    if (!f) return false;
    verts.resize(nv);
    for (uint32_t i = 0; i < nv; ++i) {
        NpcVertex& v = verts[i];
        v.pos = pos[i];
        v.nrm = nrm[i];
        v.uv = uv[i];
        for (int k = 0; k < 4; ++k) {
            v.joints[k] = uint8_t(std::min<uint32_t>(jn[i * 4 + k], nj - 1));
            v.weights[k] = wt[i * 4 + k];
        }
    }
    for (uint32_t& k : idx) if (k >= nv) return false;
    return true;
}
}  // namespace

void CitizenSystem::initStaticMembers(
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
        global_desc_set_layouts, { push_const_range },
        std::source_location::current());

    s_device_ = device;
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
    // ── THE INSTANCE STREAM ─────────────────────────────────────────
    // One PartInstance per box: the four columns of its transform and
    // its colour, at INSTANCE rate.  This replaces a push constant that
    // forced one draw call per box — see the note at the top of
    // citizen.vert.  Locations 10-14 are free here; the VINPUT_* block
    // only reaches 9.
    bindings[2].binding = 2;
    bindings[2].stride = sizeof(PartInstance);
    bindings[2].input_rate = er::VertexInputRate::INSTANCE;
    // ...plus the sixth vec4 at location 15: (kind, style, packed
    // accent colour, seed) for the garment painting in citizen.frag.
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

    er::ShaderModuleList shader_modules(2);
    shader_modules[0] = er::helper::loadShaderModule(
        device, "citizen_vert.spv", er::ShaderStageFlagBits::VERTEX_BIT,
        std::source_location::current());
    shader_modules[1] = er::helper::loadShaderModule(
        device, "citizen_frag.spv", er::ShaderStageFlagBits::FRAGMENT_BIT,
        std::source_location::current());
    s_pipeline_ = device->createPipeline(
        s_pipeline_layout_, bindings, attribs, input_assembly,
        graphic_pipeline_info, shader_modules, frame_buffer_format,
        raster_override, std::source_location::current());
    {
        er::ShaderModuleList gbuf_modules = shader_modules;
        gbuf_modules[1] = er::helper::loadShaderModule(device, "citizen_gbuf_frag.spv",
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


    // ── THE SKIN PIPELINE (fine tier, v3) ───────────────────────────
    // Same layout and fragment shader; a 12-vec4 instance stream at
    // locations 3-14 (three 3x4 transforms, colour, paint record,
    // -- clear of VINPUT_POSITION 0 and VINPUT_NORMAL 2 --
    // shape) and citizen_skin.vert, which blends each vertex between
    // the three transforms by its unit height.  Drawn DOUBLE-SIDED:
    // the meshes carry one winding, and a skinned surface can fold
    // its back to the camera on the inside of a sharp bend.
    {
        std::vector<er::VertexInputBindingDescription> sb(3);
        std::vector<er::VertexInputAttributeDescription> sa(2 + 12);
        sb[0] = bindings[0]; sb[1] = bindings[1];
        sa[0] = attribs[0];  sa[1] = attribs[1];
        sb[2].binding = 2;
        sb[2].stride = sizeof(SkinInstance);
        sb[2].input_rate = er::VertexInputRate::INSTANCE;
        for (int k = 0; k < 12; ++k) {
            sa[2 + k].binding = 2;
            sa[2 + k].location = uint32_t(3 + k);
            sa[2 + k].format = er::Format::R32G32B32A32_SFLOAT;
            sa[2 + k].offset = uint32_t(k * sizeof(glm::vec4));
        }
        er::RasterizationStateOverride two_sided;
        two_sided.override_double_sided = true;
        two_sided.double_sided = true;
        er::ShaderModuleList sm(2);
        sm[0] = er::helper::loadShaderModule(
            device, "citizen_skin_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
        sm[1] = shader_modules[1];
        s_skin_pipeline_ = device->createPipeline(
            s_pipeline_layout_, sb, sa, input_assembly,
            graphic_pipeline_info, sm, frame_buffer_format,
            two_sided, std::source_location::current());
        {
            er::ShaderModuleList gbuf_modules = sm;
            gbuf_modules[1] = er::helper::loadShaderModule(device, "citizen_gbuf_frag.spv",
                er::ShaderStageFlagBits::FRAGMENT_BIT, std::source_location::current());
            er::GraphicPipelineInfo gbuf_info = graphic_pipeline_info;
            auto att = er::helper::fillPipelineColorBlendAttachmentState(
                SET_FLAG_BIT(ColorComponent, ALL_BITS), false);
            gbuf_info.blend_state_info = std::make_shared<er::PipelineColorBlendStateCreateInfo>(
                er::helper::fillPipelineColorBlendStateCreateInfo(
                    std::vector<er::PipelineColorBlendAttachmentState>(4, att)));
            s_skin_gbuf_pipeline_ = device->createPipeline(
                s_pipeline_layout_, sb, sa, input_assembly,
                gbuf_info, gbuf_modules, gbuffer_format,
                two_sided, std::source_location::current());
        }

    }

    // Unit cube centred at origin, half-extent 1, 24 verts so every
    // face gets its own flat normal.
    static const glm::vec3 face_n[6] = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1},
        {0, 0, -1}};
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> nrm;
    std::vector<uint32_t> idx;
    for (int f = 0; f < 6; ++f) {
        glm::vec3 n = face_n[f];
        glm::vec3 u = glm::abs(n.y) > 0.5f ? glm::vec3(1, 0, 0)
                                           : glm::vec3(0, 1, 0);
        glm::vec3 v = glm::cross(n, u);
        u = glm::cross(v, n);
        uint32_t base = uint32_t(pos.size());
        for (int k = 0; k < 4; ++k) {
            float su = (k == 1 || k == 2) ? 1.0f : -1.0f;
            float sv = (k >= 2) ? 1.0f : -1.0f;
            pos.push_back(n + u * su + v * sv);
            nrm.push_back(n);
        }
        idx.insert(idx.end(), {base, base + 1, base + 2,
                               base, base + 2, base + 3});
        // both windings so no cull mode can hide a face
        idx.insert(idx.end(), {base, base + 2, base + 1,
                               base, base + 3, base + 2});
    }
    s_cube_pos_ = helper::createUnifiedMeshBuffer(
        device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        pos.size() * sizeof(pos[0]), pos.data(),
        std::source_location::current());
    s_cube_nrm_ = helper::createUnifiedMeshBuffer(
        device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        nrm.size() * sizeof(nrm[0]), nrm.data(),
        std::source_location::current());
    s_cube_idx_ = helper::createUnifiedMeshBuffer(
        device, SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
        idx.size() * sizeof(idx[0]), idx.data(),
        std::source_location::current());
    s_cube_index_count_ = uint32_t(idx.size());
    s_shadow_cube_.positions = pos;
    s_shadow_cube_.indices = idx;

    // ── The fine tier's meshes (v3): tube, blob, ball ───────────────
    // Superellipsoids with the same +-1 extents as the cube, so a part
    // transform means the same thing on any of them.  e1 shapes the
    // ends (small = flat with a rounded edge, 1 = a dome), e2 the
    // cross-section (1 = round, small = a rounded square).  One
    // winding, counter-clockwise from outside; the skin pipeline draws
    // them double-sided.  The tube has the most rings along its length
    // because that is the direction it bends in.
    auto make_shape = [&](float e1, float e2, int kLat, int kLon,
                          std::shared_ptr<er::BufferInfo>& pos_b,
                          std::shared_ptr<er::BufferInfo>& nrm_b,
                          std::shared_ptr<er::BufferInfo>& idx_b,
                          uint32_t& count) {
        std::vector<glm::vec3> rp, rn;
        std::vector<uint32_t> ri;
        auto sp = [](float x, float p) {
            return (x < 0.0f ? -1.0f : 1.0f) * std::pow(std::abs(x), p);
        };
        for (int i = 0; i <= kLat; ++i) {
            const float th = -1.5707963f + 3.14159265f * float(i) / kLat;
            const float ct = std::cos(th), st = std::sin(th);
            for (int j = 0; j <= kLon; ++j) {
                const float ph = 6.2831853f * float(j) / kLon;
                const float cp = std::cos(ph), sn = std::sin(ph);
                rp.push_back({sp(ct, e1) * sp(cp, e2), sp(st, e1),
                              sp(ct, e1) * sp(sn, e2)});
                glm::vec3 n(sp(ct, 2.0f - e1) * sp(cp, 2.0f - e2),
                            sp(st, 2.0f - e1),
                            sp(ct, 2.0f - e1) * sp(sn, 2.0f - e2));
                const float ln = glm::length(n);
                rn.push_back(ln > 1e-6f ? n / ln : glm::vec3(0, 1, 0));
            }
        }
        for (int i = 0; i < kLat; ++i) {
            for (int j = 0; j < kLon; ++j) {
                const uint32_t a = uint32_t(i * (kLon + 1) + j);
                const uint32_t b = a + 1;
                const uint32_t c = a + uint32_t(kLon + 1);
                const uint32_t d = c + 1;
                ri.insert(ri.end(), {a, c, b, b, c, d});
            }
        }
        pos_b = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            rp.size() * sizeof(rp[0]), rp.data(),
            std::source_location::current());
        nrm_b = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            rn.size() * sizeof(rn[0]), rn.data(),
            std::source_location::current());
        idx_b = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
            ri.size() * sizeof(ri[0]), ri.data(),
            std::source_location::current());
        count = uint32_t(ri.size());
        ActorShadowGeometry* shadow = &s_shadow_ball_;
        if (&pos_b == &s_tube_pos_) shadow = &s_shadow_tube_;
        else if (&pos_b == &s_round_pos_) shadow = &s_shadow_blob_;
        shadow->positions = rp;
        shadow->indices = ri;
    };
    make_shape(0.50f, 1.00f, 14, 12, s_tube_pos_, s_tube_nrm_, s_tube_idx_,
               s_tube_index_count_);                       // limbs, neck
    make_shape(0.55f, 0.70f, 10, 16, s_round_pos_, s_round_nrm_,
               s_round_idx_, s_round_index_count_);        // torso, pelvis
    make_shape(0.85f, 0.90f, 10, 16, s_ball_pos_, s_ball_nrm_, s_ball_idx_,
               s_ball_index_count_);                       // head, hands

    // ── THE CHARACTER MESHES (v33) ──────────────────────────────────
    // Fail-soft as a whole: without the assets or the shaders the
    // puppets carry on exactly as before.
    s_npc_ready_ = false;
    try {
        static const char* kNames[2] = {"man", "woman"};
        std::vector<NpcVertex> verts;
        std::vector<uint32_t> idx;
        int loaded = 0;
        for (int c = 0; c < 2; ++c) {
            NpcAsset& as = s_npc_[c];
            as = NpcAsset();
            const std::string base =
                std::string("assets/Characters/npc/") + kNames[c];
            if (!loadNpcMesh(base + ".npcmesh", verts, idx, as.bind,
                             kNpcJoints)) {
                std::cout << "[citizen] npc mesh missing/unreadable: "
                          << base << ".npcmesh" << std::endl;
                continue;
            }
            as.vb = helper::createUnifiedMeshBuffer(
                device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
                verts.size() * sizeof(NpcVertex), verts.data(),
                std::source_location::current());
            as.ib = helper::createUnifiedMeshBuffer(
                device, SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
                idx.size() * sizeof(uint32_t), idx.data(),
                std::source_location::current());
            as.shadow.indices = idx;
            for (const auto& v : verts) {
                as.shadow.positions.push_back(v.pos);
                as.shadow_joints.emplace_back(v.joints[0], v.joints[1], v.joints[2], v.joints[3]);
                as.shadow_weights.emplace_back(v.weights[0], v.weights[1], v.weights[2], v.weights[3]);
            }
            as.index_count = uint32_t(idx.size());
            as.tri_count = as.index_count / 3u;
            engine::helper::createTextureImage(
                device, base + "_albedo.png", er::Format::R8G8B8A8_UNORM,
                true, as.albedo, std::source_location::current());
            as.ok = as.vb && as.ib && as.albedo.view != nullptr;
            if (as.ok) ++loaded;
        }
        if (loaded) {
            // set 2: the palette (vertex) and the albedo (fragment)
            std::vector<er::DescriptorSetLayoutBinding> b;
            b.push_back(er::helper::getBufferDescriptionSetLayoutBinding(
                0, SET_FLAG_BIT(ShaderStage, VERTEX_BIT),
                er::DescriptorType::STORAGE_BUFFER));
            b.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
                1, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER));
            s_npc_desc_layout_ = device->createDescriptorSetLayout(
                b, std::source_location::current());
            er::DescriptorSetLayoutList layouts = global_desc_set_layouts;
            layouts.push_back(s_npc_desc_layout_);
            er::PushConstantRange pcr{};
            pcr.stage_flags =
                SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
            pcr.offset = 0;
            pcr.size = sizeof(glsl::CitizenDrawParams);
            s_npc_layout_ = device->createPipelineLayout(
                layouts, { pcr }, std::source_location::current());
            s_npc_pool_ = device->createDescriptorPool(
                std::source_location::current());
            s_npc_sampler_ = device->createSampler(
                er::Filter::LINEAR, er::SamplerAddressMode::REPEAT,
                er::SamplerMipmapMode::LINEAR, 0.0f,
                std::source_location::current());
            // the palette and the instance stream: fixed capacity, so
            // the descriptor never has to be rewritten
            s_npc_palette_buf_ = std::make_shared<er::BufferInfo>();
            er::Helper::createBuffer(
                device,
                SET_2_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT,
                                TRANSFER_DST_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                HOST_COHERENT_BIT),
                0, s_npc_palette_buf_->buffer, s_npc_palette_buf_->memory,
                std::source_location::current(),
                uint64_t(kMaxNpc) * kNpcRows * sizeof(glm::vec4), nullptr);
            s_npc_inst_buf_ = std::make_shared<er::BufferInfo>();
            er::Helper::createBuffer(
                device,
                SET_2_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT,
                                TRANSFER_DST_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                HOST_COHERENT_BIT),
                0, s_npc_inst_buf_->buffer, s_npc_inst_buf_->memory,
                std::source_location::current(),
                uint64_t(kMaxNpc) * sizeof(NpcInstance), nullptr);
            for (int c = 0; c < 2; ++c) {
                NpcAsset& as = s_npc_[c];
                if (!as.ok) continue;
                as.desc = device->createDescriptorSets(
                    s_npc_pool_, s_npc_desc_layout_, 1,
                    std::source_location::current())[0];
                er::WriteDescriptorList w;
                er::Helper::addOneBuffer(
                    w, as.desc, er::DescriptorType::STORAGE_BUFFER, 0,
                    s_npc_palette_buf_->buffer,
                    uint32_t(uint64_t(kMaxNpc) * kNpcRows * sizeof(glm::vec4)));
                er::Helper::addOneTexture(
                    w, as.desc, er::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    1, s_npc_sampler_, as.albedo.view,
                    er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                device->updateDescriptorSets(w);
            }
            // the pipeline: one interleaved vertex stream + the
            // per-instance vec4; double-sided (a scan's winding is
            // not guaranteed), citizen depth and formats
            std::vector<er::VertexInputBindingDescription> nb(2);
            std::vector<er::VertexInputAttributeDescription> na(6);
            nb[0].binding = 0;
            nb[0].stride = sizeof(NpcVertex);
            nb[0].input_rate = er::VertexInputRate::VERTEX;
            nb[1].binding = 1;
            nb[1].stride = sizeof(NpcInstance);
            nb[1].input_rate = er::VertexInputRate::INSTANCE;
            const er::Format fmts[6] = {
                er::Format::R32G32B32_SFLOAT, er::Format::R32G32B32_SFLOAT,
                er::Format::R32G32_SFLOAT, er::Format::R8G8B8A8_UINT,
                er::Format::R8G8B8A8_UNORM, er::Format::R32G32B32A32_SFLOAT};
            const uint32_t offs[6] = {
                uint32_t(offsetof(NpcVertex, pos)),
                uint32_t(offsetof(NpcVertex, nrm)),
                uint32_t(offsetof(NpcVertex, uv)),
                uint32_t(offsetof(NpcVertex, joints)),
                uint32_t(offsetof(NpcVertex, weights)), 0u};
            for (int k = 0; k < 6; ++k) {
                na[k].binding = k < 5 ? 0 : 1;
                na[k].location = uint32_t(k);
                na[k].format = fmts[k];
                na[k].offset = offs[k];
            }
            er::PipelineInputAssemblyStateCreateInfo ia;
            ia.topology = er::PrimitiveTopology::TRIANGLE_LIST;
            ia.restart_enable = false;
            er::RasterizationStateOverride two_sided;
            two_sided.override_double_sided = true;
            two_sided.double_sided = true;
            er::ShaderModuleList sm(2);
            sm[0] = er::helper::loadShaderModule(
                device, "citizen_npc_vert.spv",
                er::ShaderStageFlagBits::VERTEX_BIT,
                std::source_location::current());
            sm[1] = er::helper::loadShaderModule(
                device, "citizen_npc_frag.spv",
                er::ShaderStageFlagBits::FRAGMENT_BIT,
                std::source_location::current());
            s_npc_pipeline_ = device->createPipeline(
                s_npc_layout_, nb, na, ia, graphic_pipeline_info, sm,
                frame_buffer_format, two_sided,
                std::source_location::current());
            {
                er::ShaderModuleList gbuf_modules = sm;
                gbuf_modules[1] = er::helper::loadShaderModule(device, "citizen_npc_gbuf_frag.spv",
                    er::ShaderStageFlagBits::FRAGMENT_BIT, std::source_location::current());
                er::GraphicPipelineInfo gbuf_info = graphic_pipeline_info;
                auto att = er::helper::fillPipelineColorBlendAttachmentState(
                    SET_FLAG_BIT(ColorComponent, ALL_BITS), false);
                gbuf_info.blend_state_info = std::make_shared<er::PipelineColorBlendStateCreateInfo>(
                    er::helper::fillPipelineColorBlendStateCreateInfo(
                        std::vector<er::PipelineColorBlendAttachmentState>(4, att)));
                s_npc_gbuf_pipeline_ = device->createPipeline(
                    s_npc_layout_, nb, na, ia, gbuf_info, gbuf_modules,
                    gbuffer_format, two_sided,
                    std::source_location::current());
            }

            s_npc_ready_ = s_npc_pipeline_ != nullptr;
            std::cout << "[citizen] character meshes: man "
                      << (s_npc_[0].ok ? s_npc_[0].tri_count : 0u)
                      << " tris, woman "
                      << (s_npc_[1].ok ? s_npc_[1].tri_count : 0u)
                      << " tris (" << (s_npc_ready_ ? "on" : "OFF")
                      << ")" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "[citizen] character meshes disabled: " << e.what()
                  << std::endl;
        s_npc_ready_ = false;
    }
}

void CitizenSystem::destroyStaticMembers(
    const std::shared_ptr<er::Device>& device) {
    if (s_pipeline_layout_) device->destroyPipelineLayout(
        s_pipeline_layout_);
    s_pipeline_layout_ = nullptr;
    if (s_gbuf_pipeline_) device->destroyPipeline(s_gbuf_pipeline_);
    s_gbuf_pipeline_ = nullptr;
    if (s_pipeline_) device->destroyPipeline(s_pipeline_);
    s_pipeline_ = nullptr;
    if (s_inst_buf_) {
        s_inst_buf_->destroy(device);
        s_inst_buf_ = nullptr;
    }
    s_inst_capacity_ = 0;
    s_device_ = nullptr;
    if (s_cube_pos_) s_cube_pos_->destroy(device);
    if (s_cube_nrm_) s_cube_nrm_->destroy(device);
    if (s_cube_idx_) s_cube_idx_->destroy(device);
    s_cube_pos_ = s_cube_nrm_ = s_cube_idx_ = nullptr;
    if (s_round_pos_) s_round_pos_->destroy(device);
    if (s_round_nrm_) s_round_nrm_->destroy(device);
    if (s_round_idx_) s_round_idx_->destroy(device);
    s_round_pos_ = s_round_nrm_ = s_round_idx_ = nullptr;
    s_round_index_count_ = 0;
    if (s_tube_pos_) s_tube_pos_->destroy(device);
    if (s_tube_nrm_) s_tube_nrm_->destroy(device);
    if (s_tube_idx_) s_tube_idx_->destroy(device);
    s_tube_pos_ = s_tube_nrm_ = s_tube_idx_ = nullptr;
    s_tube_index_count_ = 0;
    if (s_ball_pos_) s_ball_pos_->destroy(device);
    if (s_ball_nrm_) s_ball_nrm_->destroy(device);
    if (s_ball_idx_) s_ball_idx_->destroy(device);
    s_ball_pos_ = s_ball_nrm_ = s_ball_idx_ = nullptr;
    s_ball_index_count_ = 0;
    if (s_skin_gbuf_pipeline_) device->destroyPipeline(s_skin_gbuf_pipeline_);
    s_skin_gbuf_pipeline_ = nullptr;
    if (s_skin_pipeline_) device->destroyPipeline(s_skin_pipeline_);
    s_skin_pipeline_ = nullptr;
    if (s_skin_buf_) s_skin_buf_->destroy(device);
    s_skin_buf_ = nullptr;
    s_skin_capacity_ = 0;
    // the character meshes
    s_npc_ready_ = false;
    if (s_npc_gbuf_pipeline_) device->destroyPipeline(s_npc_gbuf_pipeline_);
    s_npc_gbuf_pipeline_ = nullptr;
    if (s_npc_pipeline_) device->destroyPipeline(s_npc_pipeline_);
    s_npc_pipeline_ = nullptr;
    if (s_npc_layout_) device->destroyPipelineLayout(s_npc_layout_);
    s_npc_layout_ = nullptr;
    for (auto& as : s_npc_) {
        if (as.vb) as.vb->destroy(device);
        if (as.ib) as.ib->destroy(device);
        if (as.albedo.image) as.albedo.destroy(device);
        as = NpcAsset();
    }
    if (s_npc_pool_) device->destroyDescriptorPool(s_npc_pool_);
    s_npc_pool_ = nullptr;
    if (s_npc_desc_layout_) device->destroyDescriptorSetLayout(s_npc_desc_layout_);
    s_npc_desc_layout_ = nullptr;
    if (s_npc_sampler_) device->destroySampler(s_npc_sampler_);
    s_npc_sampler_ = nullptr;
    if (s_npc_inst_buf_) s_npc_inst_buf_->destroy(device);
    s_npc_inst_buf_ = nullptr;
    if (s_npc_palette_buf_) s_npc_palette_buf_->destroy(device);
    s_npc_palette_buf_ = nullptr;
}

bool CitizenSystem::loadCity(const std::string& city_json_path,
                             const std::string& world_json_path,
                             const std::string& indoor_json_path) {
    using nlohmann::json;
    loaded_ = false;
    houses_.clear();
    house_school_seats_.clear();
    graphs_.clear();
    house_graph_.clear();
    house_yaw_.clear();
    house_scale_.clear();
    buildings_.clear();
    persons_.clear();
    sim_.clear();
    std::error_code ec;
    // ── WHAT IS ACTUALLY REQUIRED ────────────────────────────────────
    // Only the WORLD manifest: it carries the house transforms, which
    // is all a resident needs to exist somewhere.  The CITY json is
    // OPTIONAL — city_sim.py writes it only for maps that also got a
    // civic district (it reads <stem>_pcg_city_buildings.json, and
    // bails when that is absent), so on an ordinary settlement map the
    // file never appeared, loadCity returned false, and the entire
    // system stayed inert: every house empty, no citizens anywhere.
    // Without the city json we synthesize a HOUSEHOLD of 3-5 per
    // house instead (see synthesizeResidents).
    // LOUD on purpose: a silent false here cost a debugging round —
    // "why no npc has been rendered" with no line to grep for.
    // std::cout, not printf: only std::cout reaches the engine log
    // (main.cpp swaps its rdbuf; printf bypasses it entirely).
    if (!std::filesystem::exists(world_json_path, ec)) {
        std::cout << "[citizen] world manifest not found ("
                  << world_json_path << ") — no citizens this map.  "
                     "Run the place stage to generate it."
                  << std::endl;
        return false;
    }
    const bool have_city = std::filesystem::exists(city_json_path, ec);
    if (!have_city) {
        std::cout << "[citizen] no city json (" << city_json_path
                  << ") — populating from the world manifest instead: "
                     "a household of 3-5 per house."
                  << std::endl;
    }
    try {
        json world;
        {
            std::ifstream f(world_json_path);
            f >> world;
        }
        const auto& hs = world.at("instances").at("houses");
        const auto& t = hs.at("t");
        const auto& yw = hs.at("yaw");
        size_t n = yw.size();
        houses_.reserve(n);
        house_yaw_.reserve(n);
        house_scale_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            houses_.emplace_back(float(t[3 * i]), float(t[3 * i + 1]),
                                 float(t[3 * i + 2]));
            house_yaw_.push_back(float(yw[i]));
        }
        // Per-axis scale, needed because the indoor graph is in
        // ARCHETYPE-local metres and a placed house is stretched to the
        // footprint it was traced from — routing a stretched house with
        // unscaled room rectangles walks people into their own walls.
        if (hs.contains("s") && hs.at("s").size() == n * 3) {
            const auto& sc = hs.at("s");
            for (size_t i = 0; i < n; ++i) {
                house_scale_.emplace_back(float(sc[3 * i]),
                                          float(sc[3 * i + 2]));
            }
        } else {
            house_scale_.assign(n, glm::vec2(1.0f));
        }
        // Which archetype each house is an instance of — the key the
        // indoor graphs are filed under.
        std::vector<std::string> node_names;
        std::vector<int> node_idx;
        if (hs.contains("nodes") && hs.contains("ni")) {
            for (const auto& nm : hs.at("nodes")) {
                node_names.push_back(nm.get<std::string>());
            }
            const auto& ni = hs.at("ni");
            node_idx.reserve(ni.size());
            for (size_t i = 0; i < ni.size(); ++i) {
                node_idx.push_back(int(ni[i].get<int>()));
            }
        }
        // ── REAL SCHOOLS ────────────────────────────────────────────
        // terrain_pcg picks a campus of house shells per catchment,
        // furnishes them as classrooms and writes their pupil-seat
        // count here.  It has to be the pipeline's decision: the
        // capacity IS the number of chairs the furniture solver managed
        // to fit, which nothing on this side can know.
        if (hs.contains("school")) {
            const auto& sc = hs.at("school");
            if (sc.size() == n) {
                house_school_seats_.resize(n, 0);
                size_t n_sch = 0, n_seat = 0;
                for (size_t i = 0; i < n; ++i) {
                    const int s = sc[i].get<int>();
                    house_school_seats_[i] = s;
                    if (s > 0) { ++n_sch; n_seat += size_t(s); }
                }
                std::cout << "[citizen] " << n_sch
                          << " school building(s) in the manifest, "
                          << n_seat << " pupil seats" << std::endl;
            }
        }
        // The whole city-json parse is conditional now.  The body is
        // NOT re-indented so this stays a reviewable diff — it is the
        // same code, one scope deeper.
        //
        // Its OWN try: the parse is all json::at(), so a stale or
        // schema-drifted city json throws — and letting that escape to
        // the outer handler would skip the synthesized fallback below,
        // leaving the map with zero citizens for exactly the reason the
        // fallback exists.  Discard the half-parsed city and populate
        // from the manifest instead.
        if (have_city) {
        try {
        json city;
        {
            std::ifstream f(city_json_path);
            f >> city;
        }
        for (const auto& b : city.at("buildings")) {
            Building bd;
            bd.type = b.at("type").get<std::string>();
            bd.entrance = {float(b.at("entrance")[0]),
                           float(b.at("entrance")[1])};
            bd.centre = {float(b.at("centre")[0]),
                         float(b.at("centre")[1])};
            bd.yaw = float(b.at("yaw_rad"));
            bd.base_y = float(b.at("base_y"));
            // Arrival spread from the building's REAL plate rather
            // than the 6 m house default — 180 mall workers scattered
            // over a 6 m box is a human pillar in the doorway.  0.7 of
            // the shorter side keeps the Vogel fan inside the walls.
            if (b.contains("size_m") && b["size_m"].size() >= 2) {
                const float pw = b["size_m"][0].get<float>();
                const float pd = b["size_m"][1].get<float>();
                bd.spread = std::max(6.0f, std::min(pw, pd) * 0.7f);
            }
            // Jobs as the initial headcount hint; the occupancy count
            // after the persons parse raises it to who actually comes.
            if (b.contains("jobs")) {
                bd.headcount = std::max(1, b["jobs"].get<int>());
            }
            buildings_.push_back(bd);
        }
        auto bindex = [this](const std::string& place) -> int {
            if (place == "home") return -1;
            if (place == "outdoors") return -2;
            for (size_t i = 0; i < buildings_.size(); ++i) {
                if (buildings_[i].type == place) return int(i);
            }
            return -1;                  // unknown place: stay home
        };
        auto parseSched = [&](const json& arr,
                              int duty) -> std::vector<Step> {
            std::vector<Step> out;
            if (!arr.is_array()) return out;
            for (const auto& st : arr) {
                Step s;
                s.minutes = parseClock(st[0].get<std::string>());
                const std::string act = st[1].get<std::string>();
                const std::string plc = st[2].get<std::string>();
                s.place = bindex(plc);
                s.activity = activityOf(act, plc, duty);
                out.push_back(s);
            }
            std::sort(out.begin(), out.end(),
                      [](const Step& a, const Step& b) {
                          return a.minutes < b.minutes;
                      });
            // ── A DAY ENDS AT HOME ──────────────────────────────────
            // city_sim writes some evening errands with times PAST the
            // final home step (e.g. groceries 17:45 after off_work
            // 17:30); sorted, the errand becomes the LAST step of the
            // day and the person then stands in the supermarket all
            // night — the night rule only puts people to sleep at
            // place == -1.  If the sorted day ends anywhere but home,
            // append the walk home 45 minutes later, clamped inside
            // the day (a step at >= 1440 never fires).
            if (!out.empty() && out.back().place != -1) {
                Step home;
                home.minutes = std::min(out.back().minutes + 45.0f,
                                        1435.0f);
                home.activity = kActIdle;
                home.place = -1;
                out.push_back(home);
            }
            return out;
        };
        for (const auto& pj : city.at("persons")) {
            Person p;
            p.house = pj.at("house").get<int>();
            if (p.house < 0 || p.house >= int(houses_.size())) continue;
            const std::string duty =
                pj.contains("duty") ? pj["duty"].get<std::string>()
                                    : std::string("worker");
            p.duty = dutyEnum(duty);
            if (pj.contains("body")) {
                p.height = pj["body"].value("height_m", 1.7f);
                const std::string st = pj["body"].value(
                    "status", std::string("fit"));
                p.bulk = st == "heavy" ? 1.25f
                       : st == "slim" ? 0.88f : 1.0f;
                p.speed = st == "frail" ? 0.8f
                        : st == "heavy" ? 1.1f : 1.35f;
            }
            if (p.duty == kDutyStudent) p.speed = 1.5f;
            p.age = ageEnum(pj.value("age", std::string("adult")));
            if (pj.contains("schedule")) {
                const auto& sc = pj["schedule"];
                p.works_weekend = sc.value("works_weekend", false);
                p.weekday = parseSched(sc["weekday"], p.duty);
                p.weekend = parseSched(sc["weekend"], p.duty);
            } else if (pj.contains("routine")) {
                p.weekday = parseSched(pj["routine"], p.duty);
            }
            if (p.weekday.empty()) continue;
            persons_.push_back(std::move(p));
        }
        // Seat every city-json resident within their household the
        // same way synthesizeResidents does.  placePos fans a family
        // out around the house centre by (hslot, hcount); left at the
        // 0/1 default a json household of four would resolve to one
        // identical point and read as a single person.
        {
            std::unordered_map<int, int> household;
            for (const Person& q : persons_) ++household[q.house];
            std::unordered_map<int, int> seat;
            for (Person& q : persons_) {
                q.hslot  = seat[q.house]++;
                q.hcount = household[q.house];
            }
        }
        // ── COUNTED SLOTS + NEIGHBOURHOOD SCHOOLS (city-json path) ──
        // Two long-standing gaps against the synthesized path, and
        // both read as "the routine system doesn't work":
        //
        //  1. Every city-json person carried slot 0, so EVERYONE at a
        //     building resolved to the same chair (furnitureAnchor
        //     indexes chairs by slot) and the same arrival spot — a
        //     whole shift stacked inside one figure, nobody visibly
        //     "sitting on a chair".  Count slots per destination the
        //     way synthesizeResidents always has.
        //
        //  2. Every student attended THE one civic-district school,
        //     however far away, while the manifest's real campuses
        //     (houses furnished as classrooms, with measured seats)
        //     stood empty.  Re-seat each pupil at the nearest campus
        //     building, preferring ones with seats left.
        if (!persons_.empty() && !buildings_.empty()) {
            int civic_school = -1;
            for (size_t i = 0; i < buildings_.size(); ++i) {
                if (buildings_[i].type == "school") {
                    civic_school = int(i);
                    break;
                }
            }
            std::vector<int> campus_bi;      // building index
            std::vector<int> campus_left;    // seats remaining
            if (civic_school >= 0 &&
                house_school_seats_.size() == houses_.size()) {
                for (size_t hi = 0; hi < houses_.size(); ++hi) {
                    if (house_school_seats_[hi] <= 0) continue;
                    Building cb;
                    cb.type = "school";
                    cb.house = int(hi);      // its classroom chairs
                    cb.centre = {houses_[hi].x, houses_[hi].z};
                    cb.entrance = cb.centre;
                    cb.base_y = houses_[hi].y;
                    cb.headcount =
                        std::max(1, house_school_seats_[hi]);
                    campus_bi.push_back(int(buildings_.size()));
                    campus_left.push_back(house_school_seats_[hi]);
                    buildings_.push_back(cb);
                }
            }
            if (!campus_bi.empty()) {
                size_t reseated = 0;
                for (Person& q : persons_) {
                    // Only people whose schedule actually visits the
                    // district school pay the campus scan.
                    bool attends = false;
                    for (const Step& st : q.weekday) {
                        if (st.place == civic_school) {
                            attends = true;
                            break;
                        }
                    }
                    if (!attends) {
                        for (const Step& st : q.weekend) {
                            if (st.place == civic_school) {
                                attends = true;
                                break;
                            }
                        }
                    }
                    if (!attends) continue;
                    const glm::vec3& hp = houses_[size_t(q.house)];
                    int pick = -1;
                    float best = 3.0e38f;
                    for (size_t c = 0; c < campus_bi.size(); ++c) {
                        const Building& cb =
                            buildings_[size_t(campus_bi[c])];
                        const float dx = cb.centre.x - hp.x;
                        const float dz = cb.centre.y - hp.z;
                        float d2 = dx * dx + dz * dz;
                        // A full campus only loses to an emptier one
                        // nearby — the seat count is a preference,
                        // not a wall (better an over-full class than
                        // a cross-map commute for a child).
                        if (campus_left[size_t(c)] <= 0) d2 *= 9.0f;
                        if (d2 < best) { best = d2; pick = int(c); }
                    }
                    if (pick < 0) break;
                    bool used = false;
                    auto reseat = [&](std::vector<Step>& sched) {
                        for (Step& st : sched) {
                            if (st.place == civic_school) {
                                st.place = campus_bi[size_t(pick)];
                                used = true;
                            }
                        }
                    };
                    reseat(q.weekday);
                    reseat(q.weekend);
                    if (used) {
                        --campus_left[size_t(pick)];
                        ++reseated;
                    }
                }
                std::cout << "[citizen] " << reseated
                          << " pupil(s) re-seated from the district "
                             "school to " << campus_bi.size()
                          << " neighbourhood campus building(s)"
                          << std::endl;
            }
            // Counted slots.  Primary destination = the building whose
            // steps cover the most weekday minutes; the first OTHER
            // building visited takes the errand slot (shop_b), for the
            // same reason synthesizeResidents gives it one — a slot is
            // only meaningful at the building it was counted at.
            std::vector<int>   occupancy(buildings_.size(), 0);
            std::vector<float> mins(buildings_.size(), 0.0f);
            for (Person& q : persons_) {
                std::fill(mins.begin(), mins.end(), 0.0f);
                const std::vector<Step>& wd = q.weekday;
                for (size_t si = 0; si < wd.size(); ++si) {
                    const int pl = wd[si].place;
                    if (pl < 0 || pl >= int(buildings_.size())) {
                        continue;
                    }
                    const float t0 = wd[si].minutes;
                    const float t1 = (si + 1 < wd.size())
                                         ? wd[si + 1].minutes
                                         : 1440.0f;
                    mins[size_t(pl)] += std::max(0.0f, t1 - t0);
                }
                int prim = -1;
                float pm = 0.0f;
                for (size_t b = 0; b < mins.size(); ++b) {
                    if (mins[b] > pm) { pm = mins[b]; prim = int(b); }
                }
                if (prim >= 0) {
                    q.slot = occupancy[size_t(prim)]++;
                }
                for (size_t b = 0; b < mins.size(); ++b) {
                    if (int(b) != prim && mins[b] > 0.0f) {
                        q.shop_b = int(b);
                        q.shop_slot = occupancy[b]++;
                        break;
                    }
                }
            }
            // Raise each building's headcount to who actually comes,
            // so the Vogel arrival fan packs evenly at real density.
            for (size_t b = 0; b < buildings_.size(); ++b) {
                buildings_[b].headcount =
                    std::max(buildings_[b].headcount, occupancy[b]);
            }
        }
        } catch (const std::exception& ce) {
            std::cout << "[citizen] city json unusable (" << ce.what()
                      << ") — falling back to synthesized residents"
                      << std::endl;
            buildings_.clear();
            persons_.clear();
        }
        }   // if (have_city)

        // Indoor route graphs, if the map has them.  After the houses
        // (it indexes by house) and before anyone is placed.
        if (!indoor_json_path.empty()) {
            loadIndoor(indoor_json_path, node_names, node_idx);
        }
        // Houses are the obstacle field for walk steering.
        buildHouseGrid();
        // ...and, through that same grid, the owner of every bed,
        // cooktop and chair the placement stage put on the level.
        // AFTER buildHouseGrid (it uses the grid), BEFORE anyone is
        // placed.  The registry is loaded at terrain apply, well ahead
        // of this call.
        harvestFurniture();

        // No city json, or one that yielded nobody (an old export, or a
        // map whose households never got allocated): give every house
        // a household so the town is inhabited from the moment the
        // terrain finishes loading.
        if (persons_.empty() && !houses_.empty()) {
            synthesizeResidents();
        }
        loaded_ = !persons_.empty();
        if (loaded_) addOutings();
        district_centre_ = glm::vec2(0.0f);
        if (!buildings_.empty()) {
            for (const auto& b : buildings_) district_centre_ += b.entrance;
            district_centre_ /= float(buildings_.size());
        }
        std::cout << "[citizen] loaded " << persons_.size()
                  << " persons, " << buildings_.size() << " buildings, "
                  << houses_.size() << " houses from "
                  << (have_city
                          ? std::filesystem::path(city_json_path)
                                .filename().string()
                          : std::string("the world manifest "
                                        "(synthesized residents)"))
                  << " — civic district around ("
                  << int(district_centre_.x) << ", "
                  << int(district_centre_.y) << ")"
                  << std::endl;
        if (have_city && persons_.size() < houses_.size() / 4) {
            std::cout << "[citizen] NOTE: only " << persons_.size()
                      << " person records for " << houses_.size()
                      << " houses — this city json predates the "
                         "full-population export.  Re-run "
                         "tools/terrain/city_sim.py to populate every "
                         "household." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "[citizen] city load failed: " << e.what()
                  << std::endl;
        loaded_ = false;
    }
    return loaded_;
}

void CitizenSystem::synthesizeResidents() {
    // ── WHY THIS EXISTS ──────────────────────────────────────────────
    // The city json (city_sim.py) is what normally allocates households
    // to houses and jobs/school seats to civic buildings.  It only gets
    // written for maps that also got a civic district, so on an
    // ordinary settlement it is absent and every house stood empty.
    // Here we build BOTH halves from the world manifest alone: a set of
    // destination anchors picked out of the house field, and one
    // household per house with a real day built around each of them.
    persons_.clear();
    buildings_.clear();
    if (houses_.empty()) return;

    // ── 1. DESTINATIONS ──────────────────────────────────────────────
    // A village's workplaces and shops ARE buildings on its streets, so
    // promote a few houses to that role rather than inventing floating
    // waypoints.  Grid the world and let each cell nominate the house
    // nearest its centre (workplace) and the runner-up (shop); a
    // coarser grid nominates schools.  This is O(houses), it is local
    // by construction — nobody commutes across the map because their
    // anchor is always in their own cell — and it is deterministic.
    constexpr float kWorkCellM   = 420.0f;    // ~5 min walk across
    constexpr float kSchoolCellM = 1680.0f;   // one school per 4x4 cells
    auto cellKey = [](float x, float z, float m) -> uint64_t {
        const int32_t cx = int32_t(std::floor(x / m));
        const int32_t cz = int32_t(std::floor(z / m));
        return (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz);
    };
    // SEVERAL anchors per cell, not one.  A 420 m cell in a real
    // settlement holds a hundred-plus houses; sending all of their
    // workers to a single doorway stacks bodies inside one another.
    // Keep the six houses nearest the cell centre — four become
    // workplaces, two become shops — and hash residents across them.
    struct CellPick {
        // Only entries [0, n) are ever read, so the arrays need no
        // initializer list — which also keeps them correct if
        // kCellAnchors changes.
        int   idx[kCellAnchors];
        float d2[kCellAnchors];
        int   n = 0;
        void offer(int house, float dist2) {
            int p = 0;
            while (p < n && d2[p] <= dist2) ++p;
            if (p >= kCellAnchors) return;       // worse than all kept
            const int last = (n < kCellAnchors ? n : kCellAnchors - 1);
            for (int k = last; k > p; --k) {
                idx[k] = idx[k - 1];
                d2[k]  = d2[k - 1];
            }
            idx[p] = house;
            d2[p]  = dist2;
            if (n < kCellAnchors) ++n;
        }
    };
    std::unordered_map<uint64_t, CellPick> work_cells, school_cells;
    auto nominate = [&](std::unordered_map<uint64_t, CellPick>& cells,
                        float cm, size_t hi_idx, const glm::vec3& h) {
        const float ccx = (std::floor(h.x / cm) + 0.5f) * cm;
        const float ccz = (std::floor(h.z / cm) + 0.5f) * cm;
        const float d2 = (h.x - ccx) * (h.x - ccx) +
                         (h.z - ccz) * (h.z - ccz);
        cells[cellKey(h.x, h.z, cm)].offer(int(hi_idx), d2);
    };
    for (size_t i = 0; i < houses_.size(); ++i) {
        nominate(work_cells,   kWorkCellM,   i, houses_[i]);
        nominate(school_cells, kSchoolCellM, i, houses_[i]);
    }
    // Sorted key order: unordered_map iteration order is an
    // implementation detail, and building INDICES are what the
    // schedules below bake in — so walk the keys sorted to keep two
    // runs of the same map identical.
    auto sortedKeys = [](const std::unordered_map<uint64_t, CellPick>& m) {
        std::vector<uint64_t> ks;
        ks.reserve(m.size());
        for (const auto& kv : m) ks.push_back(kv.first);
        std::sort(ks.begin(), ks.end());
        return ks;
    };
    std::unordered_map<uint64_t, std::vector<int>> work_bi, shop_bi;
    // MULTIPLE schools per cell, for the same reason workplaces get
    // four: with 3-5 residents per house the seats-2-and-up rule makes
    // students a third of the town, and a single anchor per 1680 m
    // cell had ~880 children resolving to one point.
    std::unordered_map<uint64_t, std::vector<int>> school_bi;
    auto addBuilding = [&](int house_idx, const char* type) -> int {
        const glm::vec3& h = houses_[house_idx];
        Building b;
        b.type = type;
        // Remember WHICH house this was promoted from: its furniture is
        // this workplace's furniture (see Building::house).
        b.house = house_idx;
        b.centre = {h.x, h.z};
        // INSIDE the building, not on its doorstep.  This anchor is
        // where a person AT work / school / the shop stands all day,
        // and offsetting it clear of the shell is what had whole
        // shifts sitting on the grass outside — "why are those people
        // sitting outside the room?".  These are promoted HOUSES with
        // walkable interiors, so the centre is a room, and the small
        // scatter below keeps a handful of people inside the footprint
        // instead of spilling into the garden.
        b.entrance = {h.x, h.z};
        b.yaw = 0.0f;
        b.base_y = h.y;
        buildings_.push_back(b);
        return int(buildings_.size()) - 1;
    };
    for (uint64_t k : sortedKeys(work_cells)) {
        const CellPick& c = work_cells[k];
        const int n_work = std::min(c.n, 4);
        for (int j = 0; j < n_work; ++j) {
            work_bi[k].push_back(addBuilding(c.idx[j], "work"));
        }
        for (int j = n_work; j < c.n; ++j) {
            shop_bi[k].push_back(addBuilding(c.idx[j], "shop"));
        }
    }
    // ── SCHOOLS: THE PIPELINE'S, WHEN THERE ARE ANY ─────────────────
    // A manifest that carries the school column has REAL school
    // buildings — houses furnished as classrooms, with a measured seat
    // count — so use those and do not promote anything.  Promoting
    // arbitrary cottages is the fallback for maps generated before the
    // column existed, and it is what put a hundred children in a
    // living room: it invented four schools per 1680 m cell whatever
    // the population, with no building behind them.
    const bool real_schools =
        house_school_seats_.size() == houses_.size() &&
        std::any_of(house_school_seats_.begin(), house_school_seats_.end(),
                    [](int s) { return s > 0; });
    if (real_schools) {
        for (size_t i = 0; i < houses_.size(); ++i) {
            if (house_school_seats_[i] <= 0) continue;
            const glm::vec3& h = houses_[i];
            const int bi = addBuilding(int(i), "school");
            school_bi[cellKey(h.x, h.z, kSchoolCellM)].push_back(bi);
        }
    } else {
        for (uint64_t k : sortedKeys(school_cells)) {
            const CellPick& c = school_cells[k];
            const int n_school = std::min(c.n, kSchoolsPerCell);
            for (int j = 0; j < n_school; ++j) {
                school_bi[k].push_back(addBuilding(c.idx[j], "school"));
            }
        }
    }
    // Pick one of a cell's anchors for this resident — spreading the
    // cell's workforce over its workplaces instead of one doorway.
    auto pick = [](const std::unordered_map<uint64_t,
                                            std::vector<int>>& m,
                   uint64_t k, uint32_t salt) -> int {
        auto it = m.find(k);
        if (it == m.end() || it->second.empty()) return -1;
        return it->second[mix32(salt) % it->second.size()];
    };
    // Headcount per building, so the arrival scatter can widen with it.
    std::vector<int> occupancy(buildings_.size(), 0);

    // ── 2. RESIDENTS ─────────────────────────────────────────────────
    const int hh_avg = (kHouseholdMin + kHouseholdMax + 1) / 2;
    persons_.reserve(houses_.size() * size_t(hh_avg));
    for (size_t i = 0; i < houses_.size(); ++i) {
        // A SCHOOL IS NOT A DWELLING.  Its rooms hold desks, not beds,
        // so nobody lives here — and letting a household move in would
        // put four people asleep on a classroom floor and, worse, count
        // its own children toward the catchment it exists to serve.
        if (real_schools && i < house_school_seats_.size() &&
                house_school_seats_[i] > 0) {
            continue;
        }
        const glm::vec3& home = houses_[i];
        // Household size, drawn per HOUSE and uniform over
        // [kHouseholdMin, kHouseholdMax].
        const uint32_t hh_hash = mix32(static_cast<uint32_t>(i));
        const int hh = kHouseholdMin +
            int(h01(hh_hash, 0x484Fu) *
                float(kHouseholdMax - kHouseholdMin + 1));
        const int hh_n = std::min(std::max(hh, kHouseholdMin),
                                  kHouseholdMax);
        for (int j = 0; j < hh_n; ++j) {
            // Mixed id per PERSON (house index and seat), so household
            // members differ from one another as well as from the street —
            // see mix32.  hh_n <= 8 keeps the two fields from colliding.
            const uint32_t hi = mix32(static_cast<uint32_t>(i) * 8u +
                                      static_cast<uint32_t>(j));
            Person p;
            p.house = static_cast<int>(i);
            p.hslot = j;
            p.hcount = hh_n;
            // ── WHO LIVES HERE ──────────────────────────────────────────
            // A household reads as a family rather than five strangers who
            // happen to share an address: seat 0 is an earner, seat 1 a
            // second adult (often an earner, sometimes keeping the house or
            // retired), and the rest are mostly children at school with the
            // occasional grown-up still at home.  Duty drives the body
            // colour in emitPerson, where the person spends their day, and
            // what they do once they get there.
            const float du = h01(hi, 0x51A1u);
            if (j == 0) {                       // primary earner
                p.duty = du < 0.45f ? kDutyWorker
                       : du < 0.75f ? kDutyOffice
                                    : kDutyShop;
            } else if (j == 1) {                // second adult
                p.duty = du < 0.30f ? kDutyWorker
                       : du < 0.50f ? kDutyOffice
                       : du < 0.60f ? kDutyShop
                       : du < 0.82f ? kDutyHomemaker
                                    : kDutyRetiree;
            } else {                            // children, mostly
                p.duty = du < 0.72f ? kDutyStudent
                       : du < 0.86f ? kDutyWorker
                       : du < 0.95f ? kDutyOffice
                                    : kDutyShop;
            }
            p.height = 1.55f + h01(hi, 0x0077u) * 0.35f;
            p.bulk   = 0.88f + h01(hi, 0x009Bu) * 0.40f;
            p.speed  = (p.duty == kDutyStudent) ? 1.50f
                     : (p.duty == kDutyRetiree) ? 0.90f
                     : 1.15f + h01(hi, 0x00C3u) * 0.35f;
            p.works_weekend = (p.duty == kDutyShop) &&
                              h01(hi, 0x002Du) < 0.5f;

            // Where this person's day happens.  A missing anchor (a lone
            // house with no cell neighbours) falls back to OUTDOORS (-2),
            // which placePos resolves to a spot beside the house — so a
            // schedule always has somewhere to send them.
            const uint64_t wk = cellKey(home.x, home.z, kWorkCellM);
            const uint64_t sk = cellKey(home.x, home.z, kSchoolCellM);
            // Salted per PERSON, not per house: a family that all worked
            // at the same desk would walk the street in lockstep.
            int work   = pick(work_bi, wk, hi * 2654435761u);
            int shop   = pick(shop_bi, wk, hi * 40503u + 7u);
            int school = pick(school_bi, sk, hi * 2246822519u + 13u);
            if (shop < 0)   shop = work;
            if (work < 0)   work = -2;
            if (shop < 0)   shop = -2;
            if (school < 0) school = work;

            int dest = work;
            int dest_act = kActDeskWork;
            switch (p.duty) {
                case kDutyStudent:   dest = school; dest_act = kActSit;    break;
                case kDutyShop:      dest = shop;   dest_act = kActBrowse; break;
                case kDutyHomemaker: dest = shop;   dest_act = kActBrowse; break;
                case kDutyRetiree:   dest = shop;   dest_act = kActSit;    break;
                case kDutyOffice:    dest = work;   dest_act = kActDeskWork; break;
                default:             dest = work;   dest_act = kActIdle;   break;
            }
            // The person's standing SLOT at that destination: their index
            // among everyone who reports there, read before the bump.
            if (dest >= 0 && dest < int(occupancy.size())) {
                p.slot = occupancy[dest];
                ++occupancy[dest];
            }
            // ...and a SECOND slot at the errand shop, because slot is
            // an index at the PRIMARY destination and means nothing
            // anywhere else.  Everyone runs a mid-day errand (see the
            // schedule below), so without this an office worker's
            // slot-128 arrives at a 64-seat shop, wraps to residue 0
            // and stands inside the shop's own slot-0 regular for the
            // whole 45 minutes — the "couple of people crowded
            // together" report, moved from the workplace to the shop.
            // Counting errands in occupancy also makes headcount the
            // real peak presence, so the arrival scatter widens to
            // match.
            if (shop >= 0 && shop != dest &&
                shop < int(occupancy.size())) {
                p.shop_b = shop;
                p.shop_slot = occupancy[shop];
                ++occupancy[shop];
            }
            const int home_act = (p.duty == kDutyRetiree) ? kActSit
                               : (p.duty == kDutyStudent) ? kActPlay
                                                          : kActIdle;

            // Day shape by duty.  Students start earliest and finish
            // mid-afternoon; the employed keep office hours; the retired
            // and homemakers make a late-morning errand and are home well
            // before dark.  Minutes since midnight.
            float wake, leave, ret, bed;
            if (p.duty == kDutyStudent) {
                wake =  6.75f * 60.0f; leave =  7.70f * 60.0f;
                ret  = 15.60f * 60.0f; bed   = 21.50f * 60.0f;
            } else if (p.duty == kDutyHomemaker || p.duty == kDutyRetiree) {
                wake =  7.50f * 60.0f; leave = 10.00f * 60.0f;
                ret  = 13.00f * 60.0f; bed   = 22.00f * 60.0f;
            } else {
                wake =  6.50f * 60.0f; leave =  7.60f * 60.0f;
                ret  = 17.60f * 60.0f; bed   = 22.50f * 60.0f;
            }
            // ── OVERTIME ─────────────────────────────────────────────
            // Not everyone comes home at the same hour, and the jitter
            // on `ret` is symmetric noise — it makes early leavers as
            // often as late ones, and never a genuinely long day.  A
            // slice of the employed stay on 1-4 hours past their normal
            // finish; the rest of the evening slides with them, capped
            // so a hinge can never land past midnight (a step at
            // minutes >= 1440 would simply never fire).  Drawn per
            // PERSON, so who works late is a fact about them rather
            // than a coin flipped each evening — a village knows who
            // is never home before dark.
            const bool ot_duty = (p.duty == kDutyWorker ||
                                  p.duty == kDutyOffice ||
                                  p.duty == kDutyShop);
            float ot = 0.0f;
            if (ot_duty && h01(hi, 0x0E7u) < 0.28f) {
                ot = 60.0f + h01(hi, 0x0E8u) * 180.0f;   // 1-4 h
            }
            ret = std::min(ret + ot, 1380.0f);           // <= 23:00
            bed = std::min(bed + ot * 0.35f, 1425.0f);   // <= 23:45
            auto jit = [&](float base_min, float span_min, uint32_t k) {
                return base_min + (h01(hi, k) - 0.5f) * span_min;
            };
            auto step = [](float minutes, int activity, int place) {
                Step st;
                st.minutes = minutes;
                st.activity = activity;
                st.place = place;
                return st;
            };
            // Walking is not scheduled: update() walks a person toward
            // whatever anchor their CURRENT step names, and emitPerson
            // overrides the pose with the walk cycle while they are in
            // transit — so "leave for work at 07:40" is one step whose
            // place is the workplace, and the commute happens on its own.
            // ── WHY THE HINGES ARE SMEARED THIS WIDE ─────────────────
            // A realistic town leaves for work inside a 40-minute window,
            // and that makes a DEAD town to look at: outside those few
            // windows not one person in 57,000 is in transit, so the
            // player sees a field of statues.  Measured in-engine: at
            // 15:40 every sampled citizen reported "idle", because the
            // next hinge was two game-hours away and the world clock
            // advances ~4 game-minutes per real minute at the default TOD
            // speed — a 20+ real-minute wait for anyone to take a step.
            //
            // So the commute is smeared over hours (+/- 2.5 h on the
            // outbound, +/- 3 h on the return) and every resident gets two
            // extra ERRANDS at independently drawn times.  At any hour of
            // the day some slice of the town is walking, whatever speed
            // the clock runs at, and an individual's day still reads as
            // sleep -> out -> home -> sleep.
            const float errand1 = jit(11.0f * 60.0f, 300.0f, 0x111u);
            const float errand2 = jit(16.0f * 60.0f, 300.0f, 0x112u);
            p.weekday = {
                step(0.0f,                              kActSleepish, -1),
                step(jit(wake,   90.0f, 0x101u),        home_act,     -1),
                step(jit(leave, 150.0f, 0x102u),        dest_act,     dest),
                // Mid-day errand: out to the shops and back, so the
                // streets are never empty between the two commutes.
                step(errand1,                           kActBrowse,   shop),
                step(errand1 + 45.0f,                   dest_act,     dest),
                step(errand2,                           kActWalk,     -2),
                step(errand2 + 40.0f,                   dest_act,     dest),
                step(jit(ret,   180.0f, 0x103u),        home_act,     -1),
                // Dinner is an EVENING hinge, not "100 minutes after
                // whenever you got home": the homemaker/retiree branch
                // returns at 13:00, and ret+100 had a fifth of the town
                // cooking from mid-afternoon until bed.
                step(jit(std::max(ret + 100.0f, 18.5f * 60.0f),
                         60.0f, 0x104u),               kActCook,     -1),
                // Washing up after dinner.  This is the ONLY thing that
                // ever sends anyone to a kitchen sink, so without it
                // those four thousand basins are scenery.
                step(jit(std::max(ret + 145.0f, 19.25f * 60.0f),
                         60.0f, 0x106u),               kActWash,     -1),
                step(jit(bed,    50.0f, 0x105u),        kActSleepish, -1),
            };
            // Weekend: a lie-in, an errand or a stroll, home for the
            // evening.  Whoever works weekends keeps the weekday shape
            // (scheduleOf only reaches for this list when they do not).
            p.weekend = {
                step(0.0f,                              kActSleepish, -1),
                step(jit( 9.00f * 60.0f, 150.0f, 0x201u), home_act,   -1),
                step(jit(11.00f * 60.0f, 240.0f, 0x202u), kActBrowse, shop),
                step(jit(13.00f * 60.0f, 240.0f, 0x207u), home_act,   -1),
                step(jit(15.00f * 60.0f, 300.0f, 0x203u), kActWalk,   -2),
                step(jit(17.50f * 60.0f, 240.0f, 0x204u), home_act,   -1),
                step(jit(19.50f * 60.0f, 120.0f, 0x205u), kActCook,   -1),
                step(jit(20.25f * 60.0f, 120.0f, 0x208u), kActWash,   -1),
                step(jit(22.50f * 60.0f, 60.0f, 0x206u),  kActSleepish, -1),
            };
            // Overtime plus jitter can push an evening hinge past
            // midnight, and a step at minutes >= 1440 never fires at
            // all (tod is always < 1440) — the person would be stuck on
            // whatever they were doing at 23:59 until the 00:00 step.
            // Clamp into the day.  Two hinges landing on the same
            // clamped minute is harmless: at home after 22:30 the night
            // rule in resolveActivity puts them to bed regardless of
            // which one won.
            auto clamp_day = [](std::vector<Step>& v) {
                for (Step& s2 : v) {
                    s2.minutes = std::min(std::max(s2.minutes, 0.0f),
                                          1439.0f);
                }
            };
            clamp_day(p.weekday);
            clamp_day(p.weekend);
            // Jitter can reorder two hinges that started close together;
            // currentStep walks the list assuming ascending minutes.
            auto by_time = [](const Step& a, const Step& b) {
                return a.minutes < b.minutes;
            };
            std::sort(p.weekday.begin(), p.weekday.end(), by_time);
            std::sort(p.weekend.begin(), p.weekend.end(), by_time);
            persons_.push_back(std::move(p));
        }   // household seat
    }   // house
    // Scatter inside the building.  Grid density works out at roughly
    // half a dozen people per anchor (230k residents over ~9k anchors),
    // so a footprint-sized box holds a workplace comfortably; it still
    // widens with sqrt(headcount) for the rare crowded anchor, but is
    // CAPPED so a busy one cannot push its staff out through the walls
    // — past the cap they simply stand closer together, which is what
    // a busy room looks like anyway.
    for (size_t b = 0; b < buildings_.size() && b < occupancy.size(); ++b) {
        const int hc = std::max(1, occupancy[b]);
        // A school is not a ROOM.  Even split kSchoolsPerCell ways it
        // gathers a couple of hundred children, so its footprint is
        // GROUNDS and it gets a campus-sized cap; a workplace or shop
        // holds a few dozen and stays room-sized so nobody is packed
        // out through a wall.
        const float cap_m =
            buildings_[b].type == "school" ? 26.0f : 9.0f;
        buildings_[b].spread =
            std::min(cap_m, 4.0f + 1.0f * std::sqrt(float(hc)));
        buildings_[b].headcount = hc;
    }
    const double avg_hh = houses_.empty() ? 0.0
        : double(persons_.size()) / double(houses_.size());
    const std::streamsize prec0 = std::cout.precision();
    std::cout << "[citizen] synthesized " << persons_.size()
              << " resident(s) in " << houses_.size() << " household(s) ("
              << std::fixed << std::setprecision(2) << avg_hh
              << " per house, range " << kHouseholdMin << "-"
              << kHouseholdMax << ") around "
              << buildings_.size() << " workplace/shop/school anchor(s)"
              << std::endl;
    // Precision is SEPARATE state from the float field: unsetting the
    // field alone leaves every later float in the engine log at two
    // significant digits.
    std::cout.unsetf(std::ios::floatfield);
    std::cout.precision(prec0);
}

void CitizenSystem::setTimeOfDayHours(float hours) {
    clock_external_ = true;
    const float tod = std::fmod(hours * 60.0f + 1440.0f, 1440.0f);
    const float prev = std::fmod(clock_min_, 1440.0f);
    int day = int(clock_min_ / 1440.0f) % 7;
    // Forward wrap only, and only a SMALL forward step: late evening
    // -> small hours across midnight is a new day, but a user dragging
    // the slider (or hitting the Midnight button) is a scrub, not a
    // day passing.  fwd is the forward distance in minutes, so a real
    // 23:59 -> 00:01 tick reads as 2 while a 23:00 -> 00:00 jump reads
    // as 60 and is rejected.
    const float fwd = tod - prev + (tod < prev ? 1440.0f : 0.0f);
    if (prev > 18.0f * 60.0f && tod < 6.0f * 60.0f && fwd < 60.0f) {
        day = (day + 1) % 7;
    }
    clock_min_ = float(day) * 1440.0f + tod;
}

void CitizenSystem::placeAll() {
    // Wipe the per-person sim state: pos 0, inited false, cur_step -1.
    // The next update() then places every person at the anchor their
    // schedule names for the CURRENT clock (near ones immediately, the
    // rest over the next few frames via the far ring) — which is the
    // whole spawn-on-Play behaviour, without a second placement path
    // that could disagree with the simulation's own.
    sim_.assign(persons_.size(), SimState{});
    // every car is re-issued at the next placement
    car_of_.assign(persons_.size(), -2);
    if (vehicles_) vehicles_->clearVehicles();
    // Restart the clock-rate measurement: the gap across an edit-mode
    // pause is not a rate sample, and a rate carried over from a
    // faster session would sprint everyone for its first second.
    prev_clock_min_ = -1.0f;
    clock_rate_ = 0.0f;
    const float tod = std::fmod(clock_min_, 1440.0f);
    std::cout << "[citizen] placing " << persons_.size()
              << " citizen(s) at day " << dayOfWeek() << " "
              << int(tod / 60.0f) << ":"
              << (int(tod) % 60 < 10 ? "0" : "")
              << int(tod) % 60 << std::endl;
}

void CitizenSystem::buildHouseGrid() {
    house_grid_.clear();
    house_grid_.reserve(houses_.size() / 4 + 16);
    for (size_t i = 0; i < houses_.size(); ++i) {
        const glm::vec3& h = houses_[i];
        const int32_t cx = int32_t(std::floor(h.x / kAvoidCell));
        const int32_t cz = int32_t(std::floor(h.z / kAvoidCell));
        const uint64_t k =
            (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz);
        house_grid_[k].push_back(int(i));
    }
}

glm::vec2 CitizenSystem::steerAroundHouses(const glm::vec2& pos,
                                           const glm::vec2& dir,
                                           int exempt_house,
                                           int exempt_dest) const {
    if (house_grid_.empty()) return dir;
    // The two buildings this walker is allowed to be inside: the house
    // they live in, and the house they are walking to.  Everything else
    // is an obstacle — walking through the neighbours is what this
    // exists to stop.
    int dest_house = -1;
    if (exempt_dest >= 0 && exempt_dest < int(buildings_.size())) {
        // Synthesized buildings ARE houses; match by position so the
        // exemption works without threading the house index through.
        const Building& b = buildings_[exempt_dest];
        const int32_t cx = int32_t(std::floor(b.centre.x / kAvoidCell));
        const int32_t cz = int32_t(std::floor(b.centre.y / kAvoidCell));
        const uint64_t k =
            (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz);
        auto it = house_grid_.find(k);
        if (it != house_grid_.end()) {
            float best = 4.0f * 4.0f;
            for (int hi : it->second) {
                const glm::vec3& h = houses_[hi];
                const float dx = h.x - b.centre.x;
                const float dz = h.z - b.centre.y;
                const float d2 = dx * dx + dz * dz;
                if (d2 < best) { best = d2; dest_house = hi; }
            }
        }
    }

    glm::vec2 steer(0.0f);
    const int32_t c0x = int32_t(std::floor((pos.x - kAvoidLook) / kAvoidCell));
    const int32_t c1x = int32_t(std::floor((pos.x + kAvoidLook) / kAvoidCell));
    const int32_t c0z = int32_t(std::floor((pos.y - kAvoidLook) / kAvoidCell));
    const int32_t c1z = int32_t(std::floor((pos.y + kAvoidLook) / kAvoidCell));
    for (int32_t cx = c0x; cx <= c1x; ++cx) {
        for (int32_t cz = c0z; cz <= c1z; ++cz) {
            const uint64_t k =
                (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz);
            auto it = house_grid_.find(k);
            if (it == house_grid_.end()) continue;
            for (int hi : it->second) {
                if (hi == exempt_house || hi == dest_house) continue;
                const glm::vec3& h = houses_[hi];
                const glm::vec2 to_h(h.x - pos.x, h.z - pos.y);
                const float d2 = to_h.x * to_h.x + to_h.y * to_h.y;
                if (d2 > (kAvoidLook + kHouseBlockR) *
                         (kAvoidLook + kHouseBlockR)) {
                    continue;
                }
                const float d = std::sqrt(std::max(d2, 1e-6f));
                if (d < kHouseBlockR) {
                    // ALREADY inside a wall (spawned there, or shoved
                    // by a previous frame): push straight out, hardest
                    // the deeper they are.  This is what unsticks a
                    // walker rather than letting them grind along.
                    steer += (-to_h / d) * (2.0f * (kHouseBlockR - d));
                    continue;
                }
                // Ahead of us, and close enough to the line to clip?
                const float along = to_h.x * dir.x + to_h.y * dir.y;
                if (along <= 0.0f || along > kAvoidLook) continue;
                const glm::vec2 perp = to_h - dir * along;
                const float side_d =
                    std::sqrt(std::max(perp.x * perp.x + perp.y * perp.y,
                                       1e-6f));
                const float clear = kHouseBlockR + 0.8f;
                if (side_d >= clear) continue;
                // Slide along the tangent, to whichever side we are
                // already leaning; weight rises as the wall nears and
                // as the obstacle gets closer.
                glm::vec2 tang(-dir.y, dir.x);
                if (perp.x * tang.x + perp.y * tang.y > 0.0f) tang = -tang;
                const float w = (clear - side_d) / clear *
                                (1.0f - along / kAvoidLook);
                steer += tang * (2.5f * w);
            }
        }
    }
    if (steer.x == 0.0f && steer.y == 0.0f) return dir;
    glm::vec2 out = dir + steer;
    const float L = std::sqrt(out.x * out.x + out.y * out.y);
    if (L < 1e-4f) return dir;
    return out / L;
}

bool CitizenSystem::loadIndoor(const std::string& path,
                               const std::vector<std::string>& house_node,
                               const std::vector<int>& house_node_idx) {
    using nlohmann::json;
    house_graph_.assign(houses_.size(), -1);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::cout << "[citizen] no indoor route graph (" << path
                  << ") — citizens walk straight lines indoors"
                  << std::endl;
        return false;
    }
    try {
        json doc;
        {
            std::ifstream f(path);
            f >> doc;
        }
        const auto& arch = doc.at("archetypes");
        std::unordered_map<std::string, int> by_name;
        for (auto it = arch.begin(); it != arch.end(); ++it) {
            IndoorGraph g;
            for (const auto& r : it.value().at("rooms")) {
                NavRoom nr;
                nr.c = glm::vec2(float(r.at("c")[0]), float(r.at("c")[1]));
                nr.hw = float(r.at("hw"));
                nr.hd = float(r.at("hd"));
                nr.yaw = float(r.at("yaw"));
                nr.storey = int(r.value("storey", 0));
                g.rooms.push_back(nr);
            }
            if (g.rooms.size() > size_t(kNavMaxRooms)) continue;
            for (const auto& dj : it.value().at("doors")) {
                NavDoor nd;
                nd.p = glm::vec2(float(dj.at("p")[0]),
                                 float(dj.at("p")[1]));
                nd.storey = int(dj.value("storey", 0));
                nd.a = int(dj.at("rooms")[0]);
                nd.b = int(dj.at("rooms")[1]);
                if (nd.a < 0 || nd.b < 0) {
                    g.street.push_back(int(g.doors.size()));
                }
                g.doors.push_back(nd);
            }
            // ── NEXT-HOP MATRIX ─────────────────────────────────────
            // One BFS per room over the doorway graph, storing only the
            // FIRST door of each shortest route.  A citizen then never
            // searches: it looks up next_[here * n + there], walks to
            // that doorway, and repeats from the room it lands in.  The
            // route is implicit in the table and cannot go stale.
            const int nr = int(g.rooms.size());
            g.next_.assign(size_t(nr) * size_t(nr), int16_t(-1));
            g.dist_.assign(size_t(nr) * size_t(nr), int16_t(-1));
            // NOT `adj(size_t(nr))`: `T v(size_t(nr));` is the vexing
            // parse — `size_t(nr)` reads as a parameter declaration, so
            // the line declares a FUNCTION and every later subscript of
            // it fails.  A named size sidesteps it for all three.
            const size_t nrz = static_cast<size_t>(nr);
            std::vector<std::vector<glm::ivec2>> adj(nrz);
            for (size_t di = 0; di < g.doors.size(); ++di) {
                const NavDoor& nd = g.doors[di];
                if (nd.a >= 0 && nd.b >= 0 && nd.a < nr && nd.b < nr) {
                    adj[size_t(nd.a)].push_back(
                        glm::ivec2(int(di), nd.b));
                    adj[size_t(nd.b)].push_back(
                        glm::ivec2(int(di), nd.a));
                }
            }
            std::vector<int> prev_door(nrz);
            std::vector<int> prev_room(nrz);
            for (int src = 0; src < nr; ++src) {
                std::fill(prev_door.begin(), prev_door.end(), -1);
                std::fill(prev_room.begin(), prev_room.end(), -1);
                std::vector<int> q{src};
                std::vector<uint8_t> seen(size_t(nr), 0);
                std::vector<int> hops(size_t(nr), 0);
                seen[size_t(src)] = 1;
                for (size_t qi = 0; qi < q.size(); ++qi) {
                    const int cur = q[qi];
                    for (const glm::ivec2& e : adj[size_t(cur)]) {
                        if (seen[size_t(e.y)]) continue;
                        seen[size_t(e.y)] = 1;
                        prev_door[size_t(e.y)] = e.x;
                        prev_room[size_t(e.y)] = cur;
                        hops[size_t(e.y)] = hops[size_t(cur)] + 1;
                        q.push_back(e.y);
                    }
                }
                g.dist_[size_t(src) * size_t(nr) + size_t(src)] = 0;
                // Walk each destination back to src; the last door on
                // that walk is the first one to take from src.
                for (int dst = 0; dst < nr; ++dst) {
                    if (dst == src || !seen[size_t(dst)]) continue;
                    g.dist_[size_t(src) * size_t(nr) + size_t(dst)] =
                        int16_t(hops[size_t(dst)]);
                    int r = dst, dr = prev_door[size_t(dst)];
                    while (prev_room[size_t(r)] != src &&
                           prev_room[size_t(r)] >= 0) {
                        r = prev_room[size_t(r)];
                        dr = prev_door[size_t(r)];
                    }
                    g.next_[size_t(src) * size_t(nr) + size_t(dst)] =
                        int16_t(dr);
                }
            }
            by_name[it.key()] = int(graphs_.size());
            graphs_.push_back(std::move(g));
        }
        size_t bound = 0;
        for (size_t i = 0; i < houses_.size(); ++i) {
            if (i >= house_node_idx.size()) break;
            const int ni = house_node_idx[i];
            if (ni < 0 || ni >= int(house_node.size())) continue;
            auto bit = by_name.find(house_node[size_t(ni)]);
            if (bit == by_name.end()) continue;
            house_graph_[i] = bit->second;
            ++bound;
        }
        std::cout << "[citizen] indoor routes: " << graphs_.size()
                  << " archetype graph(s), " << bound << " of "
                  << houses_.size() << " houses bound" << std::endl;
        return bound > 0;
    } catch (const std::exception& e) {
        std::cout << "[citizen] indoor route graph unusable ("
                  << e.what() << ") — straight lines indoors"
                  << std::endl;
        graphs_.clear();
        house_graph_.assign(houses_.size(), -1);
        return false;
    }
}

glm::vec2 CitizenSystem::worldToLocal(int hi, const glm::vec2& w) const {
    const glm::vec3& h = houses_[size_t(hi)];
    const float yaw = house_yaw_[size_t(hi)];
    const glm::vec2 s = house_scale_[size_t(hi)];
    const float dx = w.x - h.x, dz = w.y - h.z;
    // Inverse of localToWorld below.
    const float ca = std::cos(yaw), sa = std::sin(yaw);
    const float lx = dx * ca - dz * sa;
    const float lz = dx * sa + dz * ca;
    return glm::vec2(lx / std::max(s.x, 1e-3f),
                     lz / std::max(s.y, 1e-3f));
}

glm::vec2 CitizenSystem::localToWorld(int hi, const glm::vec2& l) const {
    const glm::vec3& h = houses_[size_t(hi)];
    const float yaw = house_yaw_[size_t(hi)];
    const glm::vec2 s = house_scale_[size_t(hi)];
    const float sx = l.x * s.x, sz = l.y * s.y;
    const float ca = std::cos(yaw), sa = std::sin(yaw);
    return glm::vec2(h.x + sx * ca + sz * sa,
                     h.z - sx * sa + sz * ca);
}

int CitizenSystem::roomAt(int hi, const glm::vec2& local,
                          int storey, int toward) const {
    if (hi < 0 || hi >= int(house_graph_.size())) return -1;
    const int gi = house_graph_[size_t(hi)];
    if (gi < 0) return -1;
    const IndoorGraph& g = graphs_[size_t(gi)];
    // ── ROOMS OVERLAP ───────────────────────────────────────────────
    // Where two wings meet, the floor belongs to both of their room
    // rectangles, and taking whichever came first in the list gave the
    // walker a room it was leaving rather than the one it was in — the
    // aim point past the doorway then landed back in the first room
    // and the walk stalled on the threshold.  When a destination is
    // known, the containing room NEARER IT in the doorway graph is the
    // one that makes the route shorter, and it is the right answer for
    // both halves of the overlap.  Ties, and the no-destination case,
    // go to the room the point is deepest inside.
    const int n = int(g.rooms.size());
    int best = -1;
    int best_d = 0;
    float best_in = 0.0f;
    for (size_t i = 0; i < g.rooms.size(); ++i) {
        const NavRoom& r = g.rooms[i];
        if (r.storey != storey) continue;
        const float ca = std::cos(-r.yaw), sa = std::sin(-r.yaw);
        const float ex = local.x - r.c.x, ez = local.y - r.c.y;
        const float lu = std::abs(ex * ca - ez * sa);
        const float lv = std::abs(ex * sa + ez * ca);
        const float inside = std::min(r.hw - lu, r.hd - lv);
        if (inside < 0.0f) continue;
        int d = 0;
        if (toward >= 0 && toward < n && !g.dist_.empty()) {
            const int16_t h =
                g.dist_[size_t(i) * size_t(n) + size_t(toward)];
            d = (h < 0) ? 9999 : int(h);
        }
        if (best < 0 || d < best_d || (d == best_d && inside > best_in)) {
            best = int(i);
            best_d = d;
            best_in = inside;
        }
    }
    return best;
}

int CitizenSystem::anchorHouse(const Person& p, const Step& s) const {
    if (s.place == -1) return p.house;              // home
    if (s.place >= 0 && s.place < int(buildings_.size())) {
        return buildings_[size_t(s.place)].house;   // -1 when civic
    }
    return -1;                                      // -2: outdoors
}

namespace {
// The aim point for a doorway: the opening itself, pushed into the
// room being entered.  Aiming AT the threshold is reached while still
// in the room you are leaving, so the same doorway resolves again and
// the walk stalls in the door.
glm::vec2 doorAim(const glm::vec2& dp, const glm::vec2& far_c) {
    const glm::vec2 fwd = far_c - dp;
    const float fl = glm::length(fwd);
    return (fl > 1e-3f) ? (dp + (fwd / fl) * kNavDoorPushM) : dp;
}
}  // namespace

bool CitizenSystem::indoorWaypoint(int hi, const glm::vec3& pos_world,
                                   const glm::vec3& dst_world,
                                   int16_t& nav_room,
                                   glm::vec3& out_wp) const {
    if (hi < 0 || hi >= int(house_graph_.size())) { nav_room = -1;
                                                    return false; }
    const int gi = house_graph_[size_t(hi)];
    if (gi < 0) { nav_room = -1; return false; }
    const IndoorGraph& g = graphs_[size_t(gi)];
    if (g.rooms.empty()) { nav_room = -1; return false; }
    const int n = int(g.rooms.size());
    const glm::vec2 lp = worldToLocal(hi, glm::vec2(pos_world.x,
                                                    pos_world.z));
    const glm::vec2 ld = worldToLocal(hi, glm::vec2(dst_world.x,
                                                    dst_world.z));
    // Which storey the destination is on, from its height over the
    // house base.  Rooms are filed per storey and a ground-floor route
    // through an upstairs room is nonsense.
    const float rel_y = dst_world.y - houses_[size_t(hi)].y;
    const int dst_st = int(std::max(0.0f, rel_y) / 3.0f + 0.5f);
    const int to = roomAt(hi, ld, dst_st);
    if (to < 0) { nav_room = -1; return false; }   // target is nowhere
                                                   // we have a room for
    // ── THE ROUTE ROOM IS CARRIED, NOT RE-DERIVED ───────────────────
    // Re-deriving it from the position every tick is what made the
    // walk flip at every wall: the two rooms either side of a doorway
    // each aim the walker back through it.  So the room is remembered
    // and only ever ADVANCES — to the far side of a doorway, once the
    // walker has actually reached the aim point 0.7 m inside it.  It
    // is dropped only when the walker is nowhere near it any more,
    // which is what a change of building or a respawn looks like.
    if (nav_room >= 0) {
        if (nav_room >= n || g.rooms[size_t(nav_room)].storey != dst_st) {
            nav_room = -1;
        } else {
            const NavRoom& r = g.rooms[size_t(nav_room)];
            const float ca = std::cos(-r.yaw), sa = std::sin(-r.yaw);
            const float ex = lp.x - r.c.x, ez = lp.y - r.c.y;
            const float gu = std::max(0.0f,
                                      std::abs(ex * ca - ez * sa) - r.hw);
            const float gv = std::max(0.0f,
                                      std::abs(ex * sa + ez * ca) - r.hd);
            if (gu * gu + gv * gv > kNavStickM * kNavStickM) nav_room = -1;
        }
    }
    if (nav_room < 0) {
        const int cur = roomAt(hi, lp, dst_st, to);
        if (cur >= 0) {
            nav_room = int16_t(cur);
        } else {
            // NOT IN A ROOM ON THAT STOREY.  Either outside the house —
            // in which case the way in is a street door, and heading
            // for the anchor instead is precisely what walks people
            // through the wall — or on a DIFFERENT storey, where the
            // route would need the stair and this pass does not yet
            // use it.  Distinguish them so an upstairs sleeper is not
            // marched back out of the front door: only route from
            // outside.
            for (const NavRoom& r : g.rooms) {
                const float ca = std::cos(-r.yaw), sa = std::sin(-r.yaw);
                const float ex = lp.x - r.c.x, ez = lp.y - r.c.y;
                if (std::abs(ex * ca - ez * sa) <= r.hw &&
                    std::abs(ex * sa + ez * ca) <= r.hd) {
                    return false;        // inside, just not on this storey
                }
            }
            // The nearest street door THAT LEADS ANYWHERE USEFUL.  A
            // house can have several ways in and the closest one is not
            // always on a path to the room wanted; preferring a door
            // whose room can actually reach the target costs one table
            // lookup and avoids entering by a door that then has to be
            // left again.
            float best = std::numeric_limits<float>::max();
            float best_any = std::numeric_limits<float>::max();
            int door = -1, any_door = -1;
            for (int di : g.street) {
                const NavDoor& sd = g.doors[size_t(di)];
                const int entry = (sd.a >= 0) ? sd.a : sd.b;
                if (entry < 0) continue;
                const float d2 = glm::dot(sd.p - lp, sd.p - lp);
                if (d2 < best_any) { best_any = d2; any_door = di; }
                const bool useful =
                    (entry == to) ||
                    (g.next_[size_t(entry) * size_t(n) + size_t(to)] >= 0);
                if (useful && d2 < best) { best = d2; door = di; }
            }
            if (door < 0) door = any_door;
            if (door < 0) return false;
            const NavDoor& nd = g.doors[size_t(door)];
            const int entry = (nd.a >= 0) ? nd.a : nd.b;
            const glm::vec2 wp = localToWorld(
                hi, doorAim(nd.p, g.rooms[size_t(entry)].c));
            out_wp = glm::vec3(wp.x, dst_world.y, wp.y);
            return true;                 // still outdoors: nav_room stays -1
        }
    }
    // AIM THROUGH THE OPENING, INTO THE ROOM BEYOND — not at the
    // threshold, which is reached while still in the room you started
    // in, and not at the final target, which on a corner lies back
    // through the room you came from.
    for (int guard = 0; guard <= n; ++guard) {
        if (nav_room == int16_t(to)) return false;   // walk straight at it
        const int door =
            int(g.next_[size_t(nav_room) * size_t(n) + size_t(to)]);
        if (door < 0 || door >= int(g.doors.size())) return false;
        const NavDoor& nd = g.doors[size_t(door)];
        const int far = (nd.a == nav_room) ? nd.b : nd.a;
        if (far < 0 || far >= n) return false;
        const glm::vec2 aim = doorAim(nd.p, g.rooms[size_t(far)].c);
        if (glm::dot(aim - lp, aim - lp) <= kNavReachM * kNavReachM) {
            nav_room = int16_t(far);     // arrived: the route advances
            continue;
        }
        const glm::vec2 wp = localToWorld(hi, aim);
        out_wp = glm::vec3(wp.x, dst_world.y, wp.y);
        return true;
    }
    return false;
}

void CitizenSystem::harvestFurniture() {
    beds_.clear();   stoves_.clear();   seats_.clear();   sinks_.clear();
    // ── ONE ANCHOR INDEX SPACE: HOUSES, THEN CIVIC BUILDINGS ────────
    // [0, houses_.size())            a house
    // [houses_.size(), + buildings_) a city-json civic building
    //
    // The district's mall, towers and hospital are NOT houses — the
    // business centre evicts every house inside its lot — so binning
    // their furniture to the nearest house finds nothing within the
    // radius and their shelves, desks and beds stay scenery.  Giving
    // buildings their own slot in the same table is what lets a shop
    // worker stand at a real counter.
    //
    // Only the CITY-JSON path reaches this: on the synthesized path
    // buildings_ is still empty here (synthesizeResidents fills it
    // afterwards) and its buildings are promoted houses anyway, so the
    // house half already covers them.  Count only the LEADING run of
    // house-less buildings as civic: the neighbourhood school campus
    // buildings appended after the city-json parse are promoted
    // houses (house >= 0), already covered by the house half — and
    // counting them here used to push the total past kCivicScanMax,
    // which silently zeroed n_civic and unbound the mall's own
    // shelves and desks.  The size guard keeps the linear scan below
    // honest — a city district is a couple of dozen buildings.
    size_t n_civic_raw = 0;
    while (n_civic_raw < buildings_.size() &&
           buildings_[n_civic_raw].house < 0) {
        ++n_civic_raw;
    }
    const size_t n_civic =
        (n_civic_raw <= kCivicScanMax) ? n_civic_raw : 0;
    const size_t n_slot = houses_.size() + n_civic;
    house_beds_.assign(n_slot, glm::ivec2(0));
    house_stoves_.assign(n_slot, glm::ivec2(0));
    house_seats_.assign(n_slot, glm::ivec2(0));
    house_sinks_.assign(n_slot, glm::ivec2(0));
    if (houses_.empty()) return;

    PcgInstanceRegistry& reg = PcgInstanceRegistry::get();
    if (reg.size() == 0) {
        // No <map>_pcg_instances.json for this map (or it failed to
        // parse).  Not an error: the household rosette in placePos is
        // the same fallback it always was, people just stand rather
        // than lie down.
        std::cout << "[citizen] no instance registry — no furniture "
                     "anchors; home stays the household rosette"
                  << std::endl;
        return;
    }
    // Nearest house within kFurnitureBindR, through the same 48 m grid
    // the walk steering uses (buildHouseGrid runs first).  A bed is
    // metres from its own house and tens of metres from the next, so
    // nearest-centre is unambiguous — no need to know the footprint.
    auto nearestHouse = [this, n_civic](const glm::vec3& p) -> int {
        // Civic buildings first and by a WIDER radius: a mall's own
        // shelving is 40 m from its centre and would otherwise bind to
        // whichever house happens to sit nearest the district edge.
        int c_best = -1;
        float c_d2 = kCivicBindR * kCivicBindR;
        for (size_t b = 0; b < n_civic; ++b) {
            const float dx = buildings_[b].centre.x - p.x;
            const float dz = buildings_[b].centre.y - p.z;
            const float d2 = dx * dx + dz * dz;
            if (d2 < c_d2) { c_d2 = d2; c_best = int(b); }
        }
        if (c_best >= 0) return int(houses_.size()) + c_best;
        const int32_t cx = int32_t(std::floor(p.x / kAvoidCell));
        const int32_t cz = int32_t(std::floor(p.z / kAvoidCell));
        int best = -1;
        float best_d2 = kFurnitureBindR * kFurnitureBindR;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dx = -1; dx <= 1; ++dx) {
                const uint64_t k =
                    (uint64_t(uint32_t(cx + dx)) << 32) |
                    uint32_t(cz + dz);
                auto it = house_grid_.find(k);
                if (it == house_grid_.end()) continue;
                for (int hi : it->second) {
                    const glm::vec3& h = houses_[size_t(hi)];
                    const float ddx = h.x - p.x, ddz = h.z - p.z;
                    const float d2 = ddx * ddx + ddz * ddz;
                    if (d2 < best_d2) { best_d2 = d2; best = hi; }
                }
            }
        }
        return best;
    };
    // Counting sort into per-house slices: one pass to bin, one prefix
    // sum, one to fill.  A vector-per-house would be 30k allocations
    // for a village and three times that for the three prefixes.
    auto harvest = [&](const char* prefix,
                       std::vector<Furniture>& out,
                       std::vector<glm::ivec2>& slice) -> int {
        const std::vector<PcgInstanceRecord> recs =
            reg.queryByNodePrefix(prefix, /*category=*/4);
        std::vector<int> owner(recs.size(), -1);
        std::vector<int> count(n_slot, 0);
        for (size_t i = 0; i < recs.size(); ++i) {
            const int hi = nearestHouse(recs[i].t);
            owner[i] = hi;
            if (hi >= 0) ++count[size_t(hi)];
        }
        int run = 0;
        for (size_t h = 0; h < n_slot; ++h) {
            slice[h] = glm::ivec2(run, count[h]);
            run += count[h];
        }
        out.assign(size_t(run), Furniture{});
        std::vector<int> cursor(n_slot, 0);
        for (size_t i = 0; i < recs.size(); ++i) {
            const int hi = owner[i];
            if (hi < 0) continue;
            Furniture f;
            f.t     = recs[i].t;
            f.yaw   = recs[i].yaw;
            f.scale = recs[i].scale;
            out[size_t(slice[size_t(hi)].x + cursor[size_t(hi)]++)] = f;
        }
        return run;
    };
    const int nb = harvest("obj_bed",     beds_,   house_beds_);
    const int ns = harvest("obj_cooktop", stoves_, house_stoves_);
    const int nc = harvest("obj_chair",   seats_,  house_seats_);
    const int nk = harvest("obj_sink",    sinks_,  house_sinks_);
    std::cout << "[citizen] furniture anchors: " << nb << " bed(s), "
              << ns << " cooktop(s), " << nc << " chair(s), "
              << nk << " sink(s) bound to "
              << houses_.size() << " house(s)" << std::endl;
}

int CitizenSystem::resolveActivity(const Step& s) const {
    if (s.place != -1) return s.activity;
    const float tod = std::fmod(clock_min_, 1440.0f);
    // At home, at night: asleep, whatever the step nominally says.  The
    // rule used to live only in emitPerson, which posed people asleep
    // while placePos still had them standing at the household rosette.
    if (s.activity == kActSleepish ||
        tod < kNightEndMin || tod > kNightStartMin) {
        return kActSleepish;
    }
    return s.activity;
}

CitizenSystem::Anchor CitizenSystem::furnitureAnchor(
    const Person& p, const Step& s, int activity) const {
    Anchor a;
    // ── WHOSE FURNITURE APPLIES ──────────────────────────────────────
    // At home it is their own house.  At a workplace, school or shop it
    // is the house that building was promoted from — which is the fix
    // for the report this function exists to answer: a child at school
    // and a worker at a desk are at a `place >= 0`, this used to bail
    // out on them, and they sat in the air inside a perfectly furnished
    // room.  A city-json civic building is not a house and still bails.
    int house = -1;
    int seat_i = 0;
    if (s.place == -1) {
        house = p.house;
        seat_i = p.hslot;
    } else if (s.place >= 0 && s.place < int(buildings_.size())) {
        // A promoted house resolves to that house; a city-json civic
        // building resolves to its OWN slot past the houses (see the
        // index-space note in harvestFurniture), which is how a person
        // in the mall or the tower reaches furniture that belongs to no
        // house at all.
        house = buildings_[size_t(s.place)].house;
        if (house < 0) house = int(houses_.size()) + s.place;
        // Their counted slot AT THIS building — the same index placePos
        // uses for the arrival scatter, so two people never resolve to
        // one chair while another stands empty.
        seat_i = (s.place == p.shop_b) ? p.shop_slot : p.slot;
    }
    if (house < 0) return a;
    const std::vector<Furniture>*  arr = nullptr;
    const std::vector<glm::ivec2>* sl  = nullptr;
    int kind = kAnchorNone;
    switch (activity) {
    case kActSleepish: arr = &beds_;   sl = &house_beds_;
                       kind = kAnchorBed;   break;
    case kActCook:     arr = &stoves_; sl = &house_stoves_;
                       kind = kAnchorStove; break;
    case kActWash:     arr = &sinks_;  sl = &house_sinks_;
                       kind = kAnchorSink;  break;
    case kActSit:
    case kActDeskWork: arr = &seats_;  sl = &house_seats_;
                       kind = kAnchorSeat;  break;
    default: return a;
    }
    if (house >= int(sl->size())) return a;
    const glm::ivec2 sp = (*sl)[size_t(house)];
    if (sp.y <= 0) return a;
    // ── ONE PERSON PER PIECE (two to a bed) ──────────────────────────
    // The modulo this used to do handed the same chair to every fourth
    // resident and the same mattress to all of them, which is how four
    // figures ended up inside one another.  Index straight instead, and
    // when the index runs past what the room actually holds, return
    // NONE: the caller then leaves them standing rather than posing
    // them on furniture that is not there.
    int idx = seat_i;
    float side = 0.0f;
    if (kind == kAnchorBed) {
        idx  = seat_i / 2;                       // two to a double bed
        side = (seat_i & 1) ? kBedShareOffsetM : -kBedShareOffsetM;
    }
    if (idx >= sp.y) return a;
    const Furniture& f = (*arr)[size_t(sp.x + idx)];
    a.kind = kind;
    const float fwd_x = std::sin(f.yaw);
    const float fwd_z = std::cos(f.yaw);
    switch (kind) {
    case kAnchorBed: {
        // Root at the FOOT of the mattress, on top of it; the lying
        // pose runs the body from here toward the pillow.  `side`
        // shifts a bed's second sleeper across its width — the
        // perpendicular of the same forward axis — so a couple lies
        // beside each other instead of inside each other.
        a.yaw = f.yaw + kBedYawFix;
        const float rgt_x =  fwd_z;      // perpendicular to forward
        const float rgt_z = -fwd_x;
        a.pos = glm::vec3(f.t.x + fwd_x * kBedHalfLen * f.scale +
                              rgt_x * side * f.scale,
                          f.t.y + kBedTopY * f.scale,
                          f.t.z + fwd_z * kBedHalfLen * f.scale +
                              rgt_z * side * f.scale);
        break;
    }
    case kAnchorStove:
        // In front of the cooktop, facing back into it.
        a.yaw = f.yaw + 3.14159265f;
        a.pos = glm::vec3(f.t.x + fwd_x * kStoveStand * f.scale,
                          f.t.y,
                          f.t.z + fwd_z * kStoveStand * f.scale);
        break;
    case kAnchorSink:
        // At the basin, same arrangement as the cooktop.
        a.yaw = f.yaw + 3.14159265f;
        a.pos = glm::vec3(f.t.x + fwd_x * kSinkStand * f.scale,
                          f.t.y,
                          f.t.z + fwd_z * kSinkStand * f.scale);
        break;
    default:
        // On the chair, facing the way it faces.  The sitting pose
        // already drops the root to seat height off the floor, which is
        // where the chair's own origin sits.
        a.yaw = f.yaw + kSeatYawFix;
        a.pos = f.t;
        break;
    }
    return a;
}

glm::vec3 CitizenSystem::yardSpot(const Person& p, int pid, int bi,
                                  float out_extra) const {
    // ── OUTSIDE A BUILDING ───────────────────────────────────────────
    // On the doorstep of building bi: 2-5 m out from its entrance,
    // spread along the front.
    if (bi >= 0 && bi < int(buildings_.size())) {
        const Building& b = buildings_[size_t(bi)];
        glm::vec2 out = b.entrance - b.centre;
        const float l = glm::length(out);
        out = l > 0.2f ? out / l : glm::vec2(0.0f, 1.0f);
        const glm::vec2 side(out.y, -out.x);
        const glm::vec2 sp = b.entrance +
            out * (2.0f + 3.0f * h01(uint32_t(pid), 0x33u)) +
            side * ((h01(uint32_t(pid), 0x34u) - 0.5f) * 6.0f);
        return {sp.x, b.base_y, sp.y};
    }
    // ── THE FRONT YARD ───────────────────────────────────────────────
    // In front of the house's own front door when the map shipped the
    // indoor graph (it knows the street doors and the rooms, so the
    // house's extent and the door's outward side are both known): 2.4
    // to 5 m out from the door, up to 2.5 m either side of it.  The
    // old spot was the house centre + (12, 12) m, which on a fenced
    // lot with 6 m to the neighbour is the neighbour's living room.
    const glm::vec3& h = houses_[size_t(p.house)];
    const int gi = (size_t(p.house) < house_graph_.size())
                       ? house_graph_[size_t(p.house)] : -1;
    if (gi >= 0 && gi < int(graphs_.size()) &&
        !graphs_[size_t(gi)].street.empty()) {
        const IndoorGraph& g = graphs_[size_t(gi)];
        float hw = 1.0f, hd = 1.0f;
        for (const NavRoom& r : g.rooms) {
            hw = std::max(hw, std::abs(r.c.x) + r.hw);
            hd = std::max(hd, std::abs(r.c.y) + r.hd);
        }
        // the FRONT door: the street door furthest along local +z
        // (house_gen puts the entrance on the +z long wall)
        int di = g.street.front();
        for (int k : g.street) {
            if (g.doors[size_t(k)].storey == 0 &&
                g.doors[size_t(k)].p.y > g.doors[size_t(di)].p.y) di = k;
        }
        const glm::vec2 dp = g.doors[size_t(di)].p;
        glm::vec2 out;
        if (std::abs(dp.y) / hd >= std::abs(dp.x) / hw)
            out = glm::vec2(0.0f, dp.y >= 0.0f ? 1.0f : -1.0f);
        else
            out = glm::vec2(dp.x >= 0.0f ? 1.0f : -1.0f, 0.0f);
        const glm::vec2 side(out.y, -out.x);
        const glm::vec2 lp = dp +
            out * (2.4f + 2.6f * h01(uint32_t(pid), 0x35u) + out_extra) +
            side * ((h01(uint32_t(pid), 0x36u) - 0.5f) * 5.0f);
        const glm::vec2 w = localToWorld(p.house, lp);
        return {w.x, h.y, w.y};
    }
    // No graph: the seat-based fan beside the house, at garden spacing.
    const float jx = (h01(uint32_t(pid), 11u) - 0.5f) * 6.0f;
    const float jz = (h01(uint32_t(pid), 23u) - 0.5f) * 6.0f;
    const float hcap = float(std::max(1, p.hcount));
    const float hsi  = float(p.hslot) + 0.5f;
    const float hth  = hsi * 2.39996323f;
    const float rr = kYardSpreadR * std::sqrt(hsi / hcap);
    return {h.x + 8.0f + std::cos(hth) * rr + jx * 0.5f,
            h.y,
            h.z + 8.0f + std::sin(hth) * rr + jz * 0.5f};
}

void CitizenSystem::emitOccupants(const glm::vec3& camera_pos) {
    if (!vehicles_ || !vehicles_->loaded()) return;
    std::vector<VehicleSystem::Occupant> occ;
    vehicles_->occupants(camera_pos, kDetailRadius, occ);
    for (const auto& o : occ) {
        Person tp;
        const float r = h01(o.seed, 0x51u);
        // who rides: the school bus carries children behind its driver,
        // the police cruiser police, the fire engine its crew
        if (o.type == VehicleSystem::kSchoolBus && o.seat >= 1) {
            tp.duty = kDutyStudent; tp.age = 1;
        } else if (o.type == VehicleSystem::kPolice) {
            tp.duty = kDutyPolice; tp.age = 0;
        } else if (o.type == VehicleSystem::kFireEngine) {
            tp.duty = kDutyFire; tp.age = 0;
        } else if (o.type == VehicleSystem::kAmbulance) {
            tp.duty = kDutyNurse; tp.age = 0;
        } else {
            tp.duty = r < 0.35f ? kDutyWorker : r < 0.6f ? kDutyOffice
                    : r < 0.8f ? kDutyResident : kDutyShop;
            tp.age = (o.seat >= 1 && h01(o.seed, 0x55u) < 0.12f) ? 3 : 0;
        }
        tp.height = tp.age == 1 ? 1.25f + 0.2f * h01(o.seed, 0x53u)
                                : 1.60f + 0.24f * h01(o.seed, 0x53u);
        tp.bulk = 0.88f + 0.28f * h01(o.seed, 0x54u);
        SimState ta;
        ta.pos = o.pos;
        ta.yaw = o.yaw;
        ta.ride = 2;
        ta.inited = true;
        ta.cur_step = -1;
        const float dx = o.pos.x - camera_pos.x, dz = o.pos.z - camera_pos.z;
        const bool fine = dx * dx + dz * dz < kFineRadius * kFineRadius;
        emitPerson(int(0x40000000u | (o.seed & 0x3FFFFFFFu)), ta, tp, true, fine);
    }
}

void CitizenSystem::addOutings() {
    // ── STREET LIFE ──────────────────────────────────────────────────
    // With city_sim schedules a noon town is a town of closed doors:
    // everyone is at work or at home, indoors, and the log's nearest
    // citizen is "idle" 200 m away.  Every resident gets THREE short
    // spells outdoors a day at hashed hours -- on the doorstep of
    // wherever the schedule has them (home: the front yard; work,
    // school, shop: outside that building), then back to what they
    // were doing.  At any daytime hour a slice of every street is out,
    // and the walk there and back is a walk.  Both schedule sources
    // get it (city json and synthesized), applied after either loads.
    size_t added = 0;
    auto inject = [&](std::vector<Step>& sched, uint32_t hi, uint32_t salt) {
        if (sched.empty()) return;
        // four a day (v30: was three), 20-45 min each -- about a fifth
        // of any daytime hour spent outdoors, so a street has people
        // on its doorsteps and not only the strollers passing through
        const float t0[4]   = {8.75f * 60.0f, 11.0f * 60.0f, 13.75f * 60.0f,
                               16.75f * 60.0f};
        const float span[4] = {120.0f, 150.0f, 150.0f, 150.0f};
        auto at = [&](float t) {
            int cur = -1;
            for (size_t i = 0; i < sched.size(); ++i)
                if (sched[i].minutes <= t) cur = int(i);
            return cur;
        };
        std::vector<Step> add;
        for (int k = 0; k < 4; ++k) {
            const uint32_t key = salt + 0x310u + uint32_t(k) * 7u;
            const float t = t0[k] + (h01(hi, key) - 0.5f) * span[k];
            const float dur = 20.0f + 25.0f * h01(hi, key + 1u);
            if (t < 6.5f * 60.0f || t + dur >= 21.0f * 60.0f) continue;
            const int c0 = at(t), c1 = at(t + dur);
            if (c0 < 0 || c1 < 0) continue;
            const Step& s0 = sched[size_t(c0)];
            const Step& s1 = sched[size_t(c1)];
            if (s0.activity == kActSleepish || s0.place < -1 ||
                s1.activity == kActSleepish || s1.place < -1) continue;
            Step out;
            out.minutes = t;
            out.activity = kActIdle;
            out.place = (s0.place >= 0 && s0.place < int(buildings_.size()))
                            ? -(100 + s0.place) : -2;
            Step back = s1;
            back.minutes = t + dur;
            add.push_back(out);
            add.push_back(back);
        }
        if (add.empty()) return;
        sched.insert(sched.end(), add.begin(), add.end());
        std::sort(sched.begin(), sched.end(),
                  [](const Step& a, const Step& b) {
                      return a.minutes < b.minutes;
                  });
        added += add.size() / 2;
    };
    for (size_t i = 0; i < persons_.size(); ++i) {
        inject(persons_[i].weekday, uint32_t(i), 0u);
        inject(persons_[i].weekend, uint32_t(i), 0x40u);
    }
    std::cout << "[citizen] street life: " << added
              << " outdoor spell(s) added across " << persons_.size()
              << " schedule(s)" << std::endl;
}

glm::vec3 CitizenSystem::placePos(const Person& p, const Step& s,
                                  int pid) const {
    // deterministic per-person jitter so a crowd at one entrance
    // spreads instead of z-fighting inside one another
    const float jx = (h01(pid, 11u) - 0.5f) * 6.0f;
    const float jz = (h01(pid, 23u) - 0.5f) * 6.0f;
    // A household is 3-5 people now, so HOME needs the same even
    // packing the workplaces got: two hashes inside a +/-1.5 m box put
    // a family of five in one another's ribs.  Same Vogel layout, keyed
    // on the seat within the household instead of the seat at work.
    const float hcap = float(std::max(1, p.hcount));
    const float hsi  = float(p.hslot) + 0.5f;
    const float hth  = hsi * 2.39996323f;       // golden angle (radians)
    if (s.place == -1 || s.place >= int(buildings_.size())) {
        // Indoors.  REAL FURNITURE FIRST: the bed they sleep on, the
        // cooktop they cook at, the chair they eat on — placed by the
        // same pipeline that placed the house, so this is the actual
        // room rather than a guess at where its middle is.
        {
            const Anchor an = furnitureAnchor(p, s, resolveActivity(s));
            if (an.kind != kAnchorNone) return an.pos;
        }
        // No such furniture in this house: the household rosette.
        // kHomeSpreadR keeps it inside the shell so nobody is packed
        // through a wall.
        const glm::vec3& h = houses_[p.house];
        const float rr = kHomeSpreadR * std::sqrt(hsi / hcap);
        return {h.x + std::cos(hth) * rr + jx * 0.15f,
                h.y,
                h.z + std::sin(hth) * rr + jz * 0.15f};
    }
    if (s.place == -2) {
        return yardSpot(p, pid, -1);
    }
    if (s.place <= -100) {
        // outside a building: its doorstep
        return yardSpot(p, pid, -s.place - 100);
    }
    // A WORKPLACE, SCHOOL OR SHOP.  These are promoted houses, so they
    // have real furniture too: seat the person on one of its chairs
    // before falling back to the arrival scatter.  Sitting at a desk on
    // nothing, inside a room that has chairs in it, is the report this
    // answers.
    {
        const Anchor an = furnitureAnchor(p, s, resolveActivity(s));
        if (an.kind != kAnchorNone) return an.pos;
    }
    const Building& b = buildings_[s.place];
    // EVEN PACKING, not a jitter box.  Two independent hashes put six
    // people in a +/-3 m square on top of one another often enough to
    // read as a pile ("a couple of people crowded together").  A
    // sunflower/Vogel layout gives every slot its own ring position, so
    // n people in a room are n visibly separate people; the golden
    // angle keeps successive slots apart, and the sqrt radius keeps
    // density even instead of clumping at the centre.
    // A slot is only meaningful at the building it was counted at.
    // p.slot is the index at the PRIMARY destination; carried into a
    // smaller shop it lands at rr = spread/2 * sqrt(220/16) — tens of
    // metres out, in a neighbour's living room — which is why the
    // errand shop gets its own counted slot.
    const int   cap_i = std::max(1, b.headcount);
    const float cap = float(cap_i);
    // Their standing spot HERE.  shop_slot when this is their errand
    // shop, slot otherwise; the modulo is a belt-and-braces guard for
    // the city-json path, whose slots are all 0, and for any anchor a
    // schedule reaches that nobody was counted into.
    const int   raw = (s.place == p.shop_b) ? p.shop_slot : p.slot;
    const float si = float(raw % cap_i) + 0.5f;
    const float rr = (b.spread * 0.5f) * std::sqrt(si / cap);
    const float th = si * 2.39996323f;          // golden angle (radians)
    // A whisper of hash jitter so a row of identical rooms does not
    // show the same rosette in each.
    const float wob = 0.35f;
    return {b.entrance.x + std::cos(th) * rr + jx * 0.1f * wob,
            b.base_y,
            b.entrance.y + std::sin(th) * rr + jz * 0.1f * wob};
}

int CitizenSystem::currentStep(const std::vector<Step>& sched,
                               float tod) const {
    int cur = -1;
    for (size_t i = 0; i < sched.size(); ++i) {
        if (sched[i].minutes <= tod) cur = int(i);
    }
    // before the first step of the day: still at (or heading) home
    return cur;
}


// ── STROLLERS ────────────────────────────────────────────────────────
namespace {
constexpr float kStrollRingMin   = 25.0f;   // spawn ring round the camera
constexpr float kStrollRingMax   = 480.0f;
constexpr float kStrollRecycleM  = 640.0f;
constexpr float kStrollPerHouse  = 1.3f;    // walkers per house in the ring
constexpr int   kStrollMax       = 320;
constexpr float kStrollHopMin    = 30.0f;   // house-to-house leg
constexpr float kStrollHopMax    = 150.0f;
constexpr float kStrollRoadP     = 0.45f;   // share on the road when one is near
constexpr float kStrollRoadNearM = 90.0f;
constexpr float kStrollFenceOutM = 1.6f;    // pause past the fence, not in the garden

float strollRnd(uint32_t& s) {
    if (s == 0u) s = 0x9E3779B9u;              // xorshift never leaves 0
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return float(s & 0xFFFFFFu) / 16777216.0f;
}
}  // namespace

int CitizenSystem::countHousesNear(const glm::vec3& c, float radius) const {
    if (house_grid_.empty()) return 0;
    const int r = int(std::ceil(radius / kAvoidCell));
    const int32_t cx = int32_t(std::floor(c.x / kAvoidCell));
    const int32_t cz = int32_t(std::floor(c.z / kAvoidCell));
    const float r2 = radius * radius;
    int n = 0;
    for (int dz = -r; dz <= r; ++dz) {
        for (int dx = -r; dx <= r; ++dx) {
            const uint64_t k =
                (uint64_t(uint32_t(cx + dx)) << 32) | uint32_t(cz + dz);
            auto it = house_grid_.find(k);
            if (it == house_grid_.end()) continue;
            for (int hi : it->second) {
                const glm::vec3& h = houses_[size_t(hi)];
                const float ddx = h.x - c.x, ddz = h.z - c.z;
                if (ddx * ddx + ddz * ddz < r2) ++n;
            }
        }
    }
    return n;
}

bool CitizenSystem::randomHouseNear(const glm::vec3& c, float min_m,
                                    float max_m, uint32_t& rng,
                                    int& out) const {
    if (house_grid_.empty()) return false;
    const int r = int(std::ceil(max_m / kAvoidCell));
    const int32_t cx = int32_t(std::floor(c.x / kAvoidCell));
    const int32_t cz = int32_t(std::floor(c.z / kAvoidCell));
    for (int tries = 0; tries < 28; ++tries) {
        const int dx = int(strollRnd(rng) * float(2 * r + 1)) - r;
        const int dz = int(strollRnd(rng) * float(2 * r + 1)) - r;
        const uint64_t k =
            (uint64_t(uint32_t(cx + dx)) << 32) | uint32_t(cz + dz);
        auto it = house_grid_.find(k);
        if (it == house_grid_.end() || it->second.empty()) continue;
        const int hi = it->second[size_t(int(strollRnd(rng) *
                                             float(it->second.size())) %
                                         it->second.size())];
        const glm::vec3& h = houses_[size_t(hi)];
        const float ddx = h.x - c.x, ddz = h.z - c.z;
        const float d2 = ddx * ddx + ddz * ddz;
        if (d2 < min_m * min_m || d2 > max_m * max_m) continue;
        out = hi;
        return true;
    }
    return false;
}

bool CitizenSystem::nextHop(Stroller& w) {
    int hi;
    for (int tries = 0; tries < 3; ++tries) {
        if (!randomHouseNear(w.pos, kStrollHopMin, kStrollHopMax,
                             stroll_rng_, hi)) continue;
        if (hi == w.house) continue;
        Person tp;
        tp.house = hi;
        w.house = hi;
        w.target = yardSpot(tp, int((w.seed ^ uint32_t(hi) * 2654435761u) &
                                    0x7FFFFFFFu), -1, kStrollFenceOutM);
        w.walking = true;
        return true;
    }
    return false;
}

bool CitizenSystem::spawnStroller(const glm::vec3& camera_pos, Stroller& w) {
    int hi;
    if (!randomHouseNear(camera_pos, kStrollRingMin, kStrollRingMax,
                         stroll_rng_, hi))
        return false;
    w.seed = mix32(stroll_rng_ += 0x9E3779B9u) | 1u;
    // ── WHO ──────────────────────────────────────────────────────────
    // Children out of school hours, seniors by day, the rest a mix of
    // residents and people on their way somewhere.
    const float tod = std::fmod(clock_min_, 1440.0f);
    const bool school_hours = !isWeekend() && tod > 8.5f * 60.0f &&
                              tod < 15.0f * 60.0f;
    const float r = h01(w.seed, 0x11u);
    Person& p = w.look;
    if (r < (school_hours ? 0.06f : 0.26f)) {
        p.duty = kDutyStudent; p.age = 1;
    } else if (r < 0.50f) {
        p.duty = kDutyResident; p.age = h01(w.seed, 0x12u) < 0.35f ? 3 : 0;
    } else if (r < 0.62f) {
        p.duty = kDutyRetiree; p.age = 3;
    } else if (r < 0.74f) {
        p.duty = kDutyHomemaker; p.age = 0;
    } else if (r < 0.88f) {
        p.duty = kDutyWorker; p.age = 0;
    } else {
        p.duty = kDutyOffice; p.age = 0;
    }
    p.height = p.age == 1 ? 1.15f + 0.35f * h01(w.seed, 0x13u)
             : p.age == 3 ? 1.52f + 0.22f * h01(w.seed, 0x13u)
                          : 1.58f + 0.28f * h01(w.seed, 0x13u);
    p.bulk  = 0.86f + 0.30f * h01(w.seed, 0x14u);
    p.speed = (p.age == 3 ? 0.85f : 1.15f) + 0.40f * h01(w.seed, 0x15u);
    p.house = hi;
    // one outdoors step, so emitPerson never reaches for a bed or a
    // chair: standing idle at a pause (a child: playing)
    Step out;
    out.minutes = 0.0f;
    out.activity = (p.age == 1 && h01(w.seed, 0x19u) < 0.6f) ? kActPlay
                                                              : kActIdle;
    out.place = -2;
    p.weekday.assign(1, out);
    p.weekend.clear();
    w.life_t = 90.0f + 240.0f * h01(w.seed, 0x16u);
    w.phase = h01(w.seed, 0x17u) * 6.2831853f;
    w.wait_t = 0.0f;
    // ── WHERE ────────────────────────────────────────────────────────
    // The front yard of the house drawn, then house to house; or the
    // road by it, when one runs within reach.
    Person tp;
    tp.house = hi;
    w.pos = yardSpot(tp, int(w.seed & 0x7FFFFFFFu), -1, kStrollFenceOutM);
    w.house = hi;
    w.kind = 0;
    if (vehicles_ && vehicles_->loaded() && h01(w.seed, 0x18u) < kStrollRoadP &&
        vehicles_->strollStartNear(w.pos, kStrollRoadNearM, w.seed, w.road)) {
        w.kind = 1;
        w.pos = w.road.pos;
        w.yaw = w.road.yaw;
        w.walking = true;
    } else if (!nextHop(w)) {
        return false;
    } else {
        const glm::vec3 d = w.target - w.pos;
        w.yaw = std::atan2(d.x, d.z);
    }
    if (ground_) {
        float gy; glm::vec3 gn;
        if (ground_(w.pos.x, w.pos.z, w.pos.y + 2.0f, gy, gn)) w.pos.y = gy;
    }
    return true;
}

void CitizenSystem::tickStroller(Stroller& w, float dt, float walk_scale) {
    const float v = w.look.speed * walk_scale;
    if (w.kind == 1) {
        if (!vehicles_ || !vehicles_->strollAdvance(w.road, v * dt, stroll_rng_)) {
            w.life_t = 0.0f;                        // stale: recycle
            return;
        }
        w.pos.x = w.road.pos.x;
        w.pos.z = w.road.pos.z;
        w.pos.y += (w.road.pos.y - w.pos.y) * std::min(1.0f, 4.0f * dt);
        w.yaw = w.road.yaw;
        w.phase += dt * v * 1.7f;
        w.walking = true;
        return;
    }
    if (!w.walking) {
        w.wait_t -= dt;
        if (w.wait_t <= 0.0f && !nextHop(w)) w.wait_t = 4.0f;
        return;
    }
    glm::vec3 d = w.target - w.pos;
    d.y = 0.0f;
    const float dist = glm::length(d);
    if (dist < 0.6f) {
        w.walking = false;
        w.wait_t = 2.0f + 9.0f * h01(w.seed, 0x21u + uint32_t(w.house));
        return;
    }
    glm::vec3 dir = d / dist;
    if (dist > kHouseBlockR + 3.0f) {
        const glm::vec2 sd = steerAroundHouses(
            glm::vec2(w.pos.x, w.pos.z), glm::vec2(dir.x, dir.z),
            w.house, -1);
        dir.x = sd.x;
        dir.z = sd.y;
    }
    w.pos += dir * std::min(v * dt, dist);
    w.yaw = std::atan2(dir.x, dir.z);
    w.phase += dt * v * 1.7f;
    w.pos.y += (w.target.y - w.pos.y) *
               std::min(1.0f, v * dt / std::max(dist, 1e-3f));
}

void CitizenSystem::updateStrollers(float delta_t, const glm::vec3& camera_pos,
                                    float walk_scale) {
    if (house_grid_.empty() || houses_.empty()) return;
    // the target follows the houses round the camera, once a second
    stroll_timer_ -= delta_t;
    if (stroll_timer_ <= 0.0f) {
        stroll_timer_ = 1.0f;
        stroll_houses_ = countHousesNear(camera_pos, kStrollRingMax);
        stroll_target_ = std::min(kStrollMax,
                                  int(float(stroll_houses_) * kStrollPerHouse));
    }
    // recycle the far and the expired
    for (size_t i = 0; i < strollers_.size();) {
        Stroller& w = strollers_[i];
        const float dx = w.pos.x - camera_pos.x, dz = w.pos.z - camera_pos.z;
        w.life_t -= delta_t;
        if (dx * dx + dz * dz > kStrollRecycleM * kStrollRecycleM ||
            w.life_t <= 0.0f) {
            if (i + 1 < strollers_.size()) strollers_[i] = std::move(strollers_.back());
            strollers_.pop_back();
            continue;
        }
        ++i;
    }
    // over target (the camera left a town): drop the farthest, a
    // couple a frame
    for (int k = 0; k < 2 && int(strollers_.size()) > stroll_target_; ++k) {
        size_t far_i = 0; float far_d2 = -1.0f;
        for (size_t i = 0; i < strollers_.size(); ++i) {
            const float dx = strollers_[i].pos.x - camera_pos.x;
            const float dz = strollers_[i].pos.z - camera_pos.z;
            const float d2 = dx * dx + dz * dz;
            if (d2 > far_d2) { far_d2 = d2; far_i = i; }
        }
        if (far_i + 1 < strollers_.size()) strollers_[far_i] = std::move(strollers_.back());
        strollers_.pop_back();
    }
    int budget = 6;
    while (int(strollers_.size()) < stroll_target_ && budget-- > 0) {
        Stroller w;
        if (!spawnStroller(camera_pos, w)) continue;
        strollers_.push_back(std::move(w));
    }
    // tick + ground: every frame inside the clamp radius, every fourth
    // frame beyond it (a far walker's height is beneath notice)
    const float clamp2 = kGroundClampRadius * kGroundClampRadius;
    const uint32_t frame_k = uint32_t(anim_t_ * 60.0f);
    for (size_t i = 0; i < strollers_.size(); ++i) {
        Stroller& w = strollers_[i];
        tickStroller(w, delta_t, walk_scale);
        if (!ground_) continue;
        const float dx = w.pos.x - camera_pos.x, dz = w.pos.z - camera_pos.z;
        if (dx * dx + dz * dz > clamp2 && ((i + frame_k) & 3u) != 0u) continue;
        float gy; glm::vec3 gn;
        if (ground_(w.pos.x, w.pos.z, w.pos.y + 1.0f, gy, gn) &&
            std::abs(gy - w.pos.y) < 6.0f)
            w.pos.y = gy;
    }
}

void CitizenSystem::update(float delta_t, const glm::vec3& camera_pos,
                           const GroundQueryFn& ground) {
    if (!loaded_) return;
    ground_ = ground;
    // Pose clock (see anim_t_ in the header): REAL seconds, advanced
    // here and nowhere else, deliberately independent of clock_min_
    // and of whatever multiplier the time-of-day slider is on.
    anim_t_ = std::fmod(anim_t_ + delta_t, kAnimWrap);
    // kClockScale game-seconds per real second -> minutes here.  Skipped
    // when the application drives the clock (setTimeOfDayHours): two
    // clocks running at different rates is how the sun and the town
    // drift apart.
    if (!clock_external_) {
        clock_min_ = std::fmod(clock_min_ + delta_t * (kClockScale / 60.0f),
                               7.0f * 1440.0f);
    }
    const float tod = std::fmod(clock_min_, 1440.0f);

    // ── HOW FAST IS THE CLOCK? ───────────────────────────────────────
    // kWalkTimeScale was tuned against this class's own 60x clock (1
    // real second = 1 game minute).  Driven from the world clock the
    // rate is whatever the TOD slider says — at its default 5x that is
    // 0.083 game-minutes per real second, and a fixed 6x leg speed
    // would have everyone sprinting at 8 m/s across the village.
    // Measure the rate and scale legs with it, floored at life speed
    // (walking must always LOOK like walking) and capped at the old
    // 6x (past that it reads as teleporting).  Wraps and slider jumps
    // are discarded rather than smoothed.
    if (delta_t > 1e-4f && prev_clock_min_ >= 0.0f) {
        const float dmin = clock_min_ - prev_clock_min_;
        const float a_ = std::min(1.0f, delta_t * 2.0f);
        if (dmin > 0.0f && dmin < 60.0f) {
            const float inst = dmin / delta_t;
            clock_rate_ += (inst - clock_rate_) * a_;
        } else if (dmin <= 0.0f) {
            // Clock PAUSED (auto-advance off) or scrubbed backwards.
            // Decay toward zero rather than holding the last rate —
            // a frozen clock left over from a 500x session would
            // otherwise keep everyone sprinting at the 6x cap.
            clock_rate_ += (0.0f - clock_rate_) * a_;
        }
    }
    prev_clock_min_ = clock_min_;
    const float walk_scale =
        std::min(kWalkTimeScale,
                 std::max(1.0f, kWalkTimeScale * clock_rate_));

    if (sim_.size() != persons_.size()) sim_.resize(persons_.size());
    if (car_of_.size() != persons_.size()) car_of_.assign(persons_.size(), -2);
    walk_scale_ = walk_scale;

    // ── EVERYONE is simulated; only the NEAR ring pays per frame ─────
    // Inside kNearSimRadius: the full walk tick, every frame.  Beyond:
    // the round-robin ring below snaps persons to their schedule
    // anchors.  What does NOT scale at all is the terrain ground
    // query, so that runs per frame only inside kGroundClampRadius and
    // as its own slow ring (kFarClampPerFrame/frame) — a far
    // commuter's height refreshes every couple of seconds, which at
    // 700 m+ is beneath notice.
    const size_t n = persons_.size();
    const float near2 = kNearSimRadius * kNearSimRadius;
    size_t near_clamps = 0;
    bool clamp_budget_hit = false;
    // Rotating start so the budgeted clamp tier is fair over frames.
    const size_t near_start = n ? (near_clamp_cursor_ % n) : 0;
    for (size_t k = 0; k < n; ++k) {
        size_t i = near_start + k;
        if (i >= n) i -= n;
        SimState& a = sim_[i];
        if (a.inited) {
            const float dcx0 = a.pos.x - camera_pos.x;
            const float dcz0 = a.pos.z - camera_pos.z;
            if (dcx0 * dcx0 + dcz0 * dcz0 > near2) continue;
        }
        const Person& p = persons_[i];
        const auto& sched = scheduleOf(p);
        int cs = currentStep(sched, tod);
        const Step home_step{};
        const Step& st = cs >= 0 ? sched[cs] : home_step;
        if (!a.inited) {
            a.inited = true;
            a.pos = placePos(p, st, int(i));
            a.cur_step = cs;
            a.yaw = h01(uint32_t(i), 77u) * 6.2831853f;
        }
        if (cs != a.cur_step) {
            a.cur_step = cs;
            a.gesture_t = 0.0f;
        }
        glm::vec3 target = placePos(p, st, int(i));
        // ── INDOOR ROUTING ───────────────────────────────────────────
        // Replace the destination with the next DOORWAY on the way to
        // it while one is still needed.  Near tier only, and only when
        // the map shipped a graph: without it this is a no-op and the
        // walk is the straight line it always was.
        if (!graphs_.empty()) {
            const int nav_h = anchorHouse(p, st);
            if (nav_h >= 0) {
                glm::vec3 wp;
                if (indoorWaypoint(nav_h, a.pos, target, a.nav_room, wp)) {
                    target = wp;
                }
            } else {
                a.nav_room = -1;         // outdoors — no route room to hold
            }
        }
        glm::vec3 d = target - a.pos;
        d.y = 0.0f;
        float dist = glm::length(d);
        // ── DRIVING (v25) ────────────────────────────────────────────
        // A person with a car at hand and a long way to go takes the
        // car: they vanish into it, the vehicle system drives it to the
        // curb nearest their destination, and they step out there and
        // walk the rest.  The car is theirs from their first placement,
        // parked at the curb by wherever they were then.
        if (vehicles_ && vehicles_->loaded() && i < car_of_.size()) {
            int& car = car_of_[i];
            if (car == -2) {
                car = wantsCar(p.duty, p.age, uint32_t(i))
                          ? vehicles_->spawnCar(
                                uint32_t(i) * 2654435761u + 7u, a.pos)
                          : -1;
            }
            if (car >= 0) {
                if (a.ride == 2) {
                    if (vehicles_->parked(car)) {
                        a.ride = 0;
                        a.pos = vehicles_->stepOut(car);
                        d = target - a.pos;
                        d.y = 0.0f;
                        dist = glm::length(d);
                    } else {
                        // on the driver's seat, facing the car's way:
                        // emitPerson draws them there, seated
                        a.pos = vehicles_->seatPos(car, 0);
                        a.yaw = vehicles_->yaw(car);
                        a.walking = false;
                        a.nav_room = -1;
                        continue;
                    }
                } else if (dist > kDriveMinM && vehicles_->parked(car)) {
                    const glm::vec3 cp = vehicles_->position(car);
                    const float cdx = cp.x - a.pos.x;
                    const float cdz = cp.z - a.pos.z;
                    if (cdx * cdx + cdz * cdz < kBoardR * kBoardR &&
                        vehicles_->dispatch(car, target)) {
                        a.ride = 2;
                        a.pos = cp;
                        a.walking = false;
                        a.nav_room = -1;
                        continue;
                    }
                }
            }
        }
        a.walking = dist > 0.6f;
        if (a.walking) {
            glm::vec3 dir = d / dist;
            // Walk AROUND the neighbours' houses rather than through
            // them.  Near tier only: this is the tier that actually
            // interpolates a walk, and the only one anybody can see.
            //
            // Only while EN ROUTE.  On the final approach the target
            // itself is usually inside a building — their workplace,
            // or an outdoors spot that happens to sit in a neighbour's
            // footprint — and steering away from it there would have
            // them orbit it forever, never arriving and never leaving
            // the walk pose.  Inside the last few metres, go straight.
            if (dist > kHouseBlockR + 3.0f) {
                const glm::vec2 sd = steerAroundHouses(
                    glm::vec2(a.pos.x, a.pos.z),
                    glm::vec2(dir.x, dir.z),
                    p.house,
                    st.place <= -100 ? (-st.place - 100) : st.place);
                dir.x = sd.x;
                dir.z = sd.y;
            }
            const float v = p.speed * walk_scale;
            a.pos += dir * std::min(v * delta_t, dist) * 1.0f;
            a.yaw = std::atan2(dir.x, dir.z);
            a.phase += delta_t * v * 1.7f;
            // walking between anchors: carry Y by blending the two
            // endpoints' base heights so far commuters don't tunnel;
            // the ground clamp below refines it when it is their turn
            a.pos.y += (target.y - a.pos.y) *
                       std::min(1.0f, p.speed * delta_t /
                                          std::max(dist, 1e-3f));
        } else if (dist < 0.9f && a.gesture_t < 1.2f &&
                   st.place == -1) {
            a.gesture_t += delta_t;    // door-open pause at home
        }
        // near-camera: exact terrain clamp — unconditional up close,
        // budgeted + round-robin out to kGroundClampRadius (see
        // kAlwaysClampR / kNearClampPerFrame).
        if (ground_) {
            const float dcx = a.pos.x - camera_pos.x;
            const float dcz = a.pos.z - camera_pos.z;
            const float dc2 = dcx * dcx + dcz * dcz;
            const bool always = dc2 < kAlwaysClampR * kAlwaysClampR;
            if (dc2 < kGroundClampRadius * kGroundClampRadius &&
                (always || !clamp_budget_hit)) {
                float gy;
                glm::vec3 gn;
                if (!always && ++near_clamps >= kNearClampPerFrame) {
                    // Resume HERE next frame.  The cursor advances by
                    // CANDIDATES CONSUMED, not by a fixed index
                    // stride: strided, the served window would slide
                    // by budget * |candidates| / population per frame
                    // and a 400 m ring would take ~150 frames to come
                    // round — slower than the blanket far ring below,
                    // which would make this whole tier pointless.
                    clamp_budget_hit = true;
                    near_clamp_cursor_ = i + 1;
                }
                if (ground_(a.pos.x, a.pos.z, a.pos.y + 1.0f, gy, gn)) {
                    a.pos.y = gy;
                }
            }
        }
    }
    // Budget never bound: everyone in the ring was clamped this frame,
    // so there is nothing to resume from.
    if (!clamp_budget_hit) near_clamp_cursor_ = 0;
    // far persons: staggered schedule ring.  Each visit snaps the
    // person to their CURRENT step's anchor — no walking interpolation
    // out here (a lerp nobody can resolve is a lerp nobody pays for).
    // With kFarSimPerFrame per frame a 150k-person town fully
    // refreshes in ~20 frames — well inside one game-minute tick.
    if (n) {
        for (size_t k = 0; k < kFarSimPerFrame; ++k) {
            const size_t i = (sim_cursor_ + k) % n;
            SimState& a = sim_[i];
            if (a.inited) {
                const float dcx = a.pos.x - camera_pos.x;
                const float dcz = a.pos.z - camera_pos.z;
                if (dcx * dcx + dcz * dcz <= near2) continue;
            }
            const Person& p = persons_[i];
            const auto& sched = scheduleOf(p);
            const int cs = currentStep(sched, tod);
            const Step home_step{};
            const Step& st = cs >= 0 ? sched[cs] : home_step;
            if (!a.inited) {
                a.inited = true;
                a.yaw = h01(uint32_t(i), 77u) * 6.2831853f;
            }
            if (cs != a.cur_step || glm::length(a.pos) < 1e-6f) {
                a.cur_step = cs;
                a.gesture_t = 0.0f;
                a.pos = placePos(p, st, int(i));
            }
            a.walking = false;
            // snapped out of a trip: the car comes with them
            if (a.ride) {
                a.ride = 0;
                if (vehicles_ && i < car_of_.size() && car_of_[i] >= 0) {
                    vehicles_->recall(car_of_[i], a.pos);
                }
            }
        }
        sim_cursor_ = (sim_cursor_ + kFarSimPerFrame) % n;
    }
    // far persons: staggered ground refresh ring
    if (ground_ && n) {
        for (size_t k = 0; k < kFarClampPerFrame; ++k) {
            const size_t i = (clamp_cursor_ + k) % n;
            SimState& a = sim_[i];
            float gy;
            glm::vec3 gn;
            if (a.inited &&
                ground_(a.pos.x, a.pos.z, a.pos.y + 2.0f, gy, gn)) {
                a.pos.y = gy;
            }
        }
        clamp_cursor_ = (clamp_cursor_ + kFarClampPerFrame) % n;
    }

    // ── the ambient crowd round the camera ───────────────────────────
    updateStrollers(delta_t, camera_pos, walk_scale);

    // ── RENDER TIERS ─────────────────────────────────────────────────
    // Everyone inside kShowRadius emits; the nearest kMaxDetailed
    // inside kDetailRadius get the full articulated figure, the rest a
    // single person-box (or nothing once they fall under kMinAngular).
    frame_parts_.clear();
    std::vector<std::pair<float, int>>& near_ids = near_ids_;
    near_ids.clear();
    for (size_t i = 0; i < n; ++i) {
        if (!sim_[i].inited) continue;
        const float dx = sim_[i].pos.x - camera_pos.x;
        const float dz = sim_[i].pos.z - camera_pos.z;
        const float d2 = dx * dx + dz * dz;
        if (d2 < kDetailRadius * kDetailRadius) {
            near_ids.emplace_back(d2, int(i));
        }
    }
    size_t keep = std::min(near_ids.size(), kMaxDetailed);
    std::partial_sort(near_ids.begin(), near_ids.begin() + keep,
                      near_ids.end());
    near_ids.resize(keep);
    std::vector<uint8_t>& is_detailed = is_detailed_;
    is_detailed.assign(n, 0);
    for (const auto& [d2, i] : near_ids) is_detailed[i] = 1;
    // The FINE tier: the nearest kMaxFine of those, inside kFineRadius
    // (near_ids is sorted nearest first).  Sixteen rounded parts each,
    // so the count is what keeps the figure budget flat.
    std::vector<uint8_t>& is_fine = is_fine_;
    is_fine.assign(n, 0);
    {
        size_t n_fine = 0;
        for (const auto& [d2, i] : near_ids) {
            if (d2 >= kFineRadius * kFineRadius || n_fine >= kMaxFine)
                break;
            is_fine[i] = 1;
            ++n_fine;
        }
    }
    // The CHARACTER MESHES: the nearest kMaxNpc of the fine tier
    // inside kNpcRadius (residents first; the strollers take what is
    // left of the palette).
    std::vector<uint8_t>& is_npc = is_npc_;
    is_npc.assign(n, 0);
    if (s_npc_ready_) {
        size_t n_npc = 0;
        for (const auto& [d2, i] : near_ids) {
            if (d2 >= kNpcRadius * kNpcRadius || n_npc >= kMaxNpc) break;
            if (!is_fine[i]) continue;
            is_npc[i] = 1;
            ++n_npc;
        }
    }
    frame_tube_.clear();
    frame_blob_.clear();
    frame_ball_.clear();
    frame_npc_[0].clear();
    frame_npc_[1].clear();
    frame_palette_.clear();

    if (far_thresh_ < kMinAngular) far_thresh_ = kMinAngular;
    size_t far_emitted = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!sim_[i].inited) continue;
        const float dx = sim_[i].pos.x - camera_pos.x;
        const float dz = sim_[i].pos.z - camera_pos.z;
        const float d2 = dx * dx + dz * dz;
        if (d2 > kShowRadius * kShowRadius) continue;
        if (sim_[i].ride == 2 && !is_detailed[i]) continue;  // in their car
        if (is_detailed[i]) {
            emitPerson(int(i), sim_[i], persons_[i], true,
                       is_fine[i] != 0, is_npc[i] != 0);
        } else {
            const float dist = std::sqrt(std::max(d2, 1.0f));
            if (persons_[i].height / dist < far_thresh_) continue;
            // hard ceiling so the first frame at a new vantage cannot
            // burst-draw the whole town before the controller reacts
            if (far_emitted >= kMaxFarParts + kMaxFarParts / 2) continue;
            ++far_emitted;
            emitPerson(int(i), sim_[i], persons_[i], false, false);
        }
    }
    // ── the strollers ────────────────────────────────────────────────
    // Same tiers by distance; the fine budget is theirs on top of the
    // residents' (a few dozen at most inside 90 m).
    for (const Stroller& w : strollers_) {
        const float dx = w.pos.x - camera_pos.x;
        const float dz = w.pos.z - camera_pos.z;
        const float d2 = dx * dx + dz * dz;
        if (d2 > kShowRadius * kShowRadius) continue;
        SimState ta;
        ta.pos = w.pos;
        ta.yaw = w.yaw;
        ta.phase = w.phase;
        ta.walking = w.walking;
        ta.inited = true;
        ta.cur_step = 0;
        const bool detailed = d2 < kDetailRadius * kDetailRadius;
        const bool fine = d2 < kFineRadius * kFineRadius;
        emitPerson(int(0x20000000u | (w.seed & 0x1FFFFFFFu)), ta, w.look,
                   detailed, detailed && fine,
                   detailed && fine && s_npc_ready_ &&
                       d2 < kNpcRadius * kNpcRadius);
    }
    // ── the people in the ambient cars ───────────────────────────────
    emitOccupants(camera_pos);
    // Far-tier draw budget: a whole town in frame is >100k one-box
    // persons — more push-constant draws than the pass can afford.
    // Nudge the angular cutoff until the emitted count sits inside
    // kMaxFarParts (and relax it back when the crowd thins), so the
    // nearest / largest figures always win the budget.
    if (far_emitted > kMaxFarParts) {
        far_thresh_ *= 1.25f;
    } else if (far_emitted < kMaxFarParts / 2 &&
               far_thresh_ > kMinAngular) {
        far_thresh_ = glm::max(kMinAngular, far_thresh_ * 0.9f);
    }
    far_thresh_ = glm::min(far_thresh_, 0.02f);

    // ── Telemetry: one [citizen] line every ~5 real seconds ──────────
    // Answers "where are the citizens" from the log alone: game clock,
    // how many are inited, how many sit within the detail / near-sim /
    // show radii of the camera, the nearest person's position and
    // distance, how many parts this frame actually emitted, and where
    // the civic district is relative to the camera.  goes through
    // std::cout so it lands in logs/engine_stdout_*.log.
    dbg_timer_ += delta_t;
    if (dbg_timer_ >= 5.0f) {
        dbg_timer_ = 0.0f;
        size_t n_init = 0, in_detail = 0, in_near = 0, in_show = 0;
        float best_d2 = std::numeric_limits<float>::max();
        int best_i = -1;
        for (size_t i = 0; i < n; ++i) {
            if (!sim_[i].inited) continue;
            ++n_init;
            const float dx = sim_[i].pos.x - camera_pos.x;
            const float dz = sim_[i].pos.z - camera_pos.z;
            const float d2 = dx * dx + dz * dz;
            if (d2 < kDetailRadius * kDetailRadius) ++in_detail;
            if (d2 < kNearSimRadius * kNearSimRadius) ++in_near;
            if (d2 < kShowRadius * kShowRadius) ++in_show;
            if (d2 < best_d2) { best_d2 = d2; best_i = int(i); }
        }
        const int day = dayOfWeek();
        const int hh = int(std::fmod(clock_min_, 1440.0f)) / 60;
        const int mm = int(std::fmod(clock_min_, 1440.0f)) % 60;
        const float ddx = district_centre_.x - camera_pos.x;
        const float ddz = district_centre_.y - camera_pos.z;
        std::cout << "[citizen] day " << day << " "
                  << (hh < 10 ? "0" : "") << hh << ":"
                  << (mm < 10 ? "0" : "") << mm
                  << " | pop " << n << " (inited " << n_init << ")"
                  << " | <300m " << in_detail
                  << "  <700m " << in_near
                  << "  <10km " << in_show
                  << " | drawn parts " << frame_parts_.size()
                  << " (far_thresh " << far_thresh_ << ")"
                  << " | strollers " << strollers_.size() << "/"
                  << stroll_target_ << " (" << stroll_houses_
                  << " houses in " << int(kStrollRingMax) << " m)";
        if (best_i >= 0) {
            std::cout << " | nearest #" << best_i << " at ("
                      << int(sim_[best_i].pos.x) << ", "
                      << int(sim_[best_i].pos.y) << ", "
                      << int(sim_[best_i].pos.z) << ") d="
                      << int(std::sqrt(best_d2)) << "m"
                      << (sim_[best_i].walking ? " walking" : " idle");
        }
        std::cout << " | cam (" << int(camera_pos.x) << ", "
                  << int(camera_pos.z) << ") district "
                  << int(std::sqrt(ddx * ddx + ddz * ddz)) << "m away";
        // ── WHERE IS THE NEAREST SCHOOL ─────────────────────────────
        // Schools are a couple of percent of the houses on the map, so
        // walking into one by chance essentially does not happen and
        // "I never see a classroom" is the expected experience even
        // when every one of them is furnished correctly.  Print the
        // nearest one's position so it can be flown to directly.
        {
            int best_s = -1;
            float best_sd2 = std::numeric_limits<float>::max();
            for (size_t b = 0; b < buildings_.size(); ++b) {
                if (buildings_[b].type != "school") continue;
                const float sx = buildings_[b].centre.x - camera_pos.x;
                const float sz = buildings_[b].centre.y - camera_pos.z;
                const float d2 = sx * sx + sz * sz;
                if (d2 < best_sd2) { best_sd2 = d2; best_s = int(b); }
            }
            if (best_s >= 0) {
                std::cout << " | nearest school ("
                          << int(buildings_[best_s].centre.x) << ", "
                          << int(buildings_[best_s].centre.y) << ") "
                          << int(std::sqrt(best_sd2)) << "m, "
                          << buildings_[best_s].headcount << " pupil(s)";
            }
        }
        std::cout << std::endl;
    }
}

void CitizenSystem::emitPerson(int pid_i, const SimState& a,
                               const Person& p, bool detailed,
                               bool fine, bool npc) {
    if (!detailed) {
        // FAR TIER: one box, person-sized, duty-tinted — a figure at a
        // distance, not a puppet.  Slight walk bob keeps crowds alive.
        const float s = p.height / 1.75f;
        float bob = a.walking
                        ? std::abs(std::cos(a.phase)) * 0.03f * s : 0.0f;
        glm::mat4 M =
            glm::translate(glm::mat4(1.0f),
                           {a.pos.x, a.pos.y + bob, a.pos.z}) *
            glm::rotate(glm::mat4(1.0f), a.yaw, glm::vec3(0, 1, 0)) *
            glm::translate(glm::mat4(1.0f), {0.0f, 0.875f * s, 0.0f}) *
            glm::scale(glm::mat4(1.0f),
                       {0.20f * s * p.bulk, 0.875f * s,
                        0.13f * s * p.bulk});
        frame_parts_.push_back({M, glm::vec4(dutyColor(p.duty), 0.12f),
                                glm::vec4(0.0f)});
        return;
    }

    const auto& sched = scheduleOf(p);
    const Step home_step{};
    const Step& st = a.cur_step >= 0 ? sched[a.cur_step] : home_step;
    // resolveActivity carries the night rule, and it is the SAME call
    // placePos made when it decided where to put this person — so the
    // pose and the spot always agree.
    int act = a.walking ? kActWalk : resolveActivity(st);
    // Door-open gesture on arrival home — but never in place of going
    // to bed: at 03:00 an arm reaching for a door handle is not what
    // anybody is doing.
    if (!a.walking && act != kActSleepish && a.gesture_t > 0.0f &&
        a.gesture_t < 1.2f && st.place == -1) {
        act = kActBrowse;              // arm-forward: opening the door
    }

    // ── ON THE FURNITURE ─────────────────────────────────────────────
    // At home, the anchor carries both the spot and the FACING: a bed
    // decides which way its sleeper lies, a cooktop which way its cook
    // turns.  The body is drawn at the anchor rather than at a.pos:
    // walking stops within 0.6 m of the target and the ground clamp
    // rewrites y every frame, and neither of those belongs on a
    // mattress.  Costs one array index, and only for the few hundred
    // figures the detail tier draws.
    glm::vec3 body_pos = a.pos;
    float     body_yaw = a.yaw;
    bool      lying    = false;
    bool      seated   = false;
    if (!a.walking) {
        const Anchor an = furnitureAnchor(p, st, act);
        if (an.kind != kAnchorNone) {
            body_pos = an.pos;
            body_yaw = an.yaw;
            lying    = (an.kind == kAnchorBed);
            seated   = (an.kind == kAnchorSeat);
        }
    }

    if (a.ride == 2) {
        // IN A CAR: on the seat the vehicle system put them on (a.pos
        // is the floor point under it), facing the car's heading.
        body_pos = a.pos;
        body_yaw = a.yaw;
        seated   = true;
        lying    = false;
        act      = kActSit;
    }
    const float s = p.height / 1.75f;
    const float bw = p.bulk;
    const Look look = lookOf(pid_i, p.duty, p.age);
    glm::vec4 col(dutyColor(p.duty), 0.12f);

    const float swing = std::sin(a.phase);
    // Per-person offset on the shared real-time pose clock, so two
    // neighbours standing in the same doorway are not one puppet
    // mirrored.  0.7315 is just an irrational-ish stride through the
    // id space; the offset is scaled by each frequency below, which
    // keeps the 8*pi wrap seamless for every half-integer rate.
    const float anim_phase = anim_t_ + float(pid_i) * 0.7315f;
    float root_y = body_pos.y;
    float torso_pitch = 0.0f;
    float leg_l = -0.0f, leg_r = 0.0f, arm_l = 0.0f, arm_r = 0.0f;
    // Joint bends for the fine figure: knees (positive = shin swings
    // back under the thigh) and elbows (negative = forearm comes
    // forward).  The seven-cube figure ignores them.
    float knee_l = 0.0f, knee_r = 0.0f, elbow_l = -0.25f, elbow_r = -0.25f;
    bool sitting = false;
    switch (act) {
    case kActWalk:
        leg_l = swing * 0.55f;
        leg_r = -swing * 0.55f;
        arm_l = -swing * 0.45f;
        arm_r = swing * 0.45f;
        // the knee flexes most while the leg SWINGS THROUGH (moving
        // forward under the body -- the shin tucks back to clear the
        // ground) and is near straight at heel strike and push-off
        {
            const float cph = std::cos(a.phase);
            knee_l = 0.12f + 0.80f * std::max(0.0f, -cph);
            knee_r = 0.12f + 0.80f * std::max(0.0f, cph);
        }
        elbow_l = elbow_r = -0.40f;
        root_y += std::abs(std::cos(a.phase)) * 0.03f * s;
        break;
    case kActSleepish:
        if (lying) {
            // LYING DOWN.  R below is rotY(yaw) * rotX(pitch) applied
            // about the ROOT, which sits between the feet — so a
            // quarter turn of pitch tips the whole figure flat, on its
            // back, extending from the root toward its head.
            // furnitureAnchor put the root at the foot of the mattress
            // exactly this reason.  Limbs stay straight; the shared
            // breath below is the only motion.
            torso_pitch = -1.5707963f;
            arm_l = arm_r = 0.0f;
            leg_l = leg_r = 0.0f;
            // The quarter turn also stands the torso's DEPTH on end:
            // local +Z (half-extent 0.11) becomes world up, so the
            // body's centreline has to rise by that much or half the
            // sleeper is inside the mattress.
            root_y += 0.12f * s;
        } else {
            // No bed in this house — dozing upright, the old behaviour.
            arm_l = arm_r = 0.04f * std::sin(anim_phase * 0.5f);
        }
        break;
    case kActSit:
    case kActDeskWork:
        if (seated) {
            sitting = true;
            leg_l = leg_r = -1.45f;    // thighs forward
            knee_l = knee_r = 1.45f;   // ...shins hang from the seat
            arm_l = arm_r = act == kActDeskWork ? -0.9f : -0.4f;
            elbow_l = elbow_r = act == kActDeskWork ? -0.6f : -0.9f;
        } else {
            // NOTHING TO SIT ON.  The pose used to be unconditional, so
            // a person whose room had no free chair — or none at all —
            // was drawn folded into a sitting shape in mid-air, which
            // is the single most conspicuous thing this system did.  A
            // figure standing where it should be sitting is a modelling
            // shortfall; a figure sitting on nothing is a bug, and only
            // one of the two reads as a mistake.
            arm_l = arm_r = 0.10f * std::sin(anim_phase * 0.5f);
        }
        break;
    case kActCook:
        arm_r = -1.1f + 0.25f * std::sin(anim_t_ * 4.0f +
                                         float(pid_i));
        arm_l = -0.5f;
        elbow_r = -0.4f; elbow_l = -0.9f;
        torso_pitch = 0.12f;
        break;
    case kActBrowse:
        arm_r = -0.9f;
        elbow_r = -0.3f;
        torso_pitch = 0.08f;
        break;
    case kActWash:
        // Washing up: both hands down in the basin, a slight lean over
        // it, and a small scrub off the real-time pose clock.  TWO arms
        // rather than the cook's one is the whole read — from behind,
        // one arm out is stirring and two is washing.
        arm_l = -1.15f + 0.10f * std::sin(anim_phase * 3.0f);
        arm_r = -1.15f - 0.10f * std::sin(anim_phase * 3.0f);
        elbow_l = elbow_r = -0.35f;
        torso_pitch = 0.15f;
        break;
    case kActCare:
        // rounds: slow sway + attending arm
        arm_l = -0.7f + 0.2f * std::sin(anim_phase * 2.0f);
        elbow_l = -0.6f;
        torso_pitch = 0.16f;
        break;
    case kActPlay:
        arm_l = std::sin(anim_t_ * 5.0f + float(pid_i)) * 0.8f;
        arm_r = -std::sin(anim_t_ * 5.0f + float(pid_i)) * 0.8f;
        elbow_l = elbow_r = -0.5f;
        break;
    default: {
        // STANDING IDLE.  A 0.06 rad arm sway was the whole of it, and
        // on a box figure that is a 3 cm displacement nobody reads as
        // motion — hence "they don't move at all" even when the sim is
        // ticking.  Give it a slow weight shift instead, at an
        // amplitude that survives being seen from ten metres.
        arm_l = arm_r = 0.10f * std::sin(anim_phase * 0.5f);
        break;
    }
    }
    // Everyone who is not WALKING still breathes: a small torso pitch
    // and root rise, ~14 cycles a minute, desynchronized per person so
    // a street does not inhale in unison.  Without it a seated desk
    // worker, a browsing shopper and a sleeper are all mannequins —
    // the walk cycle was the only motion this system had.
    if (act != kActWalk) {
        const float br = std::sin(anim_phase * 1.5f);
        // A sleeper's breath is the RISE only: adding it to the pitch
        // would rock a flat body end over end about its feet, which at
        // 1.7 m of lever arm is a visible see-saw rather than a breath.
        if (!lying) torso_pitch += 0.015f * br;
        root_y += 0.008f * s * br;
    }
    if (sitting) root_y -= 0.42f * s;

    // The head grows on a child: a toddler's is nearer a third of its
    // height than a fifth.  (The rest of the figure scales with s.)
    const float hs = p.age == 1 ? 1.15f : p.age == 2 ? 1.30f : 1.0f;
    // Heels: the foot pitches about its toe (below), which raises the
    // heel end and the shin on it; the figure rises with it.
    if (fine && look.heels && !lying) root_y += 0.03f * s;

    const glm::mat4 R =
        glm::rotate(glm::mat4(1.0f), body_yaw, glm::vec3(0, 1, 0)) *
        glm::rotate(glm::mat4(1.0f), torso_pitch, glm::vec3(1, 0, 0));
    const glm::mat4 root =
        glm::translate(glm::mat4(1.0f),
                       glm::vec3(body_pos.x, root_y, body_pos.z)) * R;
    auto hinge = [](const glm::mat4& frame, glm::vec3 pivot, float rot) {
        if (rot == 0.0f) return frame;
        return frame * glm::translate(glm::mat4(1.0f), pivot) *
               glm::rotate(glm::mat4(1.0f), rot, glm::vec3(1, 0, 0)) *
               glm::translate(glm::mat4(1.0f), -pivot);
    };
    const glm::vec3 skin = look.skin;
    const glm::vec3 sleeve_lo = look.long_sleeve ? look.top : skin;

    if (fine) {
        // ── THE FINE FIGURE: rounded parts on a SKELETON ────────────
        const float shw = look.female ? 0.92f : 1.0f;   // shoulders
        const float hpw = look.female ? 1.06f : 1.0f;   // hips
        const float ax = 0.21f * s * bw * shw;          // shoulder span
        const float lx = 0.09f * s;                     // hip span
        Skeleton sk;
        sk.pivot[jPelvis] = {0.0f, 0.98f * s, 0.0f};
        sk.pivot[jSpine]  = {0.0f, 1.06f * s, 0.0f};    // the waist
        sk.pivot[jNeck]   = {0.0f, 1.45f * s, 0.0f};
        sk.pivot[jHead]   = {0.0f, 1.51f * s, 0.0f};
        for (int side = 0; side < 2; ++side) {
            const float sx = side == 0 ? -1.0f : 1.0f;
            const int sh = side == 0 ? jShoulderL : jShoulderR;
            sk.pivot[sh]     = {sx * ax, 1.40f * s, 0.0f};
            sk.pivot[sh + 1] = {sx * ax, 1.13f * s, 0.0f};   // elbow
            sk.pivot[sh + 2] = {sx * ax, 0.885f * s, 0.0f};  // wrist
            const int hp = side == 0 ? jHipL : jHipR;
            sk.pivot[hp]     = {sx * lx, 0.90f * s, 0.0f};
            sk.pivot[hp + 1] = {sx * lx, 0.47f * s, 0.0f};   // knee
            sk.pivot[hp + 2] = {sx * lx, 0.07f * s, 0.0f};   // ankle
        }
        for (auto& r : sk.rot) r = glm::vec3(0.0f);

        // ── the pose: the activity's hinges first ───────────────────
        auto& rot = sk.rot;
        rot[jShoulderL].x = arm_l;   rot[jShoulderR].x = arm_r;
        rot[jElbowL].x = elbow_l;    rot[jElbowR].x = elbow_r;
        rot[jHipL].x = leg_l;        rot[jHipR].x = leg_r;
        rot[jKneeL].x = knee_l;      rot[jKneeR].x = knee_r;
        rot[jShoulderL].z = -0.08f;  rot[jShoulderR].z = 0.08f;  // clear the torso
        float lateral = 0.0f;                  // root side-step, body x
        const bool senior = p.age == 3, toddler = p.age == 2;
        const float ph = anim_phase;

        // ── ...then the layers ──────────────────────────────────────
        if (act == kActWalk) {
            const float sn = std::sin(a.phase);
            // stride and swing by age
            const float stride = senior ? 0.70f : toddler ? 0.80f : 1.0f;
            const float swing_k = senior ? 0.60f : 1.0f;
            rot[jHipL].x *= stride;      rot[jHipR].x *= stride;
            rot[jShoulderL].x *= swing_k; rot[jShoulderR].x *= swing_k;
            // the forward-swinging arm bends more at the elbow
            rot[jElbowL].x -= 0.15f * std::max(0.0f, sn);
            rot[jElbowR].x -= 0.15f * std::max(0.0f, -sn);
            // the pelvis turns with the leading leg, the spine turns
            // against it and the neck undoes the rest, so the head
            // faces the way the feet go; a little pelvic roll and a
            // side-step with it
            rot[jPelvis].y = 0.06f * sn;
            rot[jPelvis].z = 0.035f * sn;
            rot[jSpine].y = -0.10f * sn;
            rot[jSpine].x = 0.03f;
            rot[jNeck].y = 0.04f * sn;
            rot[jHead].x = 0.02f * std::cos(2.0f * a.phase);
            lateral = 0.012f * s * sn;
            // the foot: toe down at push-off, lifted while the knee tucks
            rot[jAnkleL].x = 0.30f * std::max(0.0f, sn) - 0.3f * knee_l;
            rot[jAnkleR].x = 0.30f * std::max(0.0f, -sn) - 0.3f * knee_r;
            if (toddler) {          // arms out for balance, wide stance
                rot[jShoulderL].z = -0.35f; rot[jShoulderR].z = 0.35f;
                rot[jElbowL].x = rot[jElbowR].x = -0.6f;
                rot[jHipL].z = -0.10f;      rot[jHipR].z = 0.10f;
                rot[jPelvis].z = 0.07f * sn;
            }
        } else if (!lying) {
            // standing or seated: a breath in the chest, a slow weight
            // shift with the legs planted (the hips undo the pelvic
            // tilt), and the head looking about on its own clock
            rot[jSpine].x += 0.010f * std::sin(ph * 1.5f);
            if (!sitting) {
                const float ws = std::sin(ph * 0.5f);
                rot[jPelvis].z = 0.025f * ws;
                rot[jSpine].z = -0.025f * ws;
                rot[jHipL].z = rot[jHipR].z = -0.025f * ws;
                lateral = 0.02f * s * ws;
            }
            rot[jNeck].y = 0.40f * std::sin(ph * 0.31f) *
                           std::sin(ph * 0.17f + 1.0f);
            rot[jHead].x = 0.04f * std::sin(ph * 0.23f);
        }
        switch (act) {                     // where the work is
        case kActDeskWork:
            rot[jHead].x += 0.18f; rot[jSpine].x += 0.05f;
            rot[jElbowL].x += 0.05f * std::sin(anim_t_ * 6.0f + float(pid_i));
            rot[jElbowR].x -= 0.05f * std::sin(anim_t_ * 6.0f + float(pid_i));
            break;
        case kActCook:   rot[jHead].x += 0.25f; break;
        case kActWash:   rot[jHead].x += 0.35f; break;
        case kActCare:   rot[jHead].x += 0.20f; break;
        case kActBrowse:
            rot[jHead].x += 0.12f;
            rot[jNeck].y = 0.25f * std::sin(ph * 0.5f);
            break;
        case kActPlay:
            rot[jNeck].y = 0.25f * std::sin(anim_t_ * 5.0f + float(pid_i));
            break;
        default: break;
        }
        if (senior && !lying) {
            // the stoop: spine forward, neck back to keep the eyes up;
            // a little give in the knees; the stick hand hardly swings
            const float stoop = 0.10f + 0.08f * lookRand(uint32_t(pid_i), 0x5Au);
            rot[jSpine].x += stoop;
            rot[jNeck].x -= stoop * 0.6f;
            if (!sitting) { rot[jKneeL].x += 0.08f; rot[jKneeR].x += 0.08f; }
            if (look.stick && !sitting) {
                rot[jShoulderR].x = -0.12f + 0.4f * rot[jShoulderR].x;
                rot[jElbowR].x = -0.30f;
            }
        }
        solveSkeleton(sk, root * glm::translate(glm::mat4(1.0f),
                                                glm::vec3(lateral, 0.0f, 0.0f)));

        // ── THE CHARACTER MESH (v33) ────────────────────────────────
        // A baked scan instead of the parts: the same skeleton, the
        // same pose, re-solved on the ASSET's joint positions (scaled
        // to this person's height) so the mesh bends at its own knees
        // and elbows; every asset joint follows one citizen joint.
        // Children keep the puppet -- a scaled-down adult is not a
        // child.
        if (npc && s_npc_ready_ && p.age != 1 && p.age != 2 &&
            frame_palette_.size() + size_t(kNpcRows) <=
                size_t(kMaxNpc) * size_t(kNpcRows)) {
            const int ch = look.female ? 1 : 0;
            const NpcAsset& as = s_npc_[ch].ok ? s_npc_[ch]
                                                : s_npc_[1 - ch];
            if (as.ok) {
                // asset joint -> citizen joint (asset order: hips,
                // spine, chest, neck, head, L shoulder, L upper arm, L
                // forearm, L hand, R shoulder, R upper arm, R forearm,
                // R hand, L thigh, L shin, L foot, R thigh, R shin, R
                // foot; the asset's LEFT is +x = the citizen's R side)
                static const int kA2C[kNpcJoints] = {
                    jPelvis, jSpine, jSpine, jNeck, jHead,
                    jSpine, jShoulderR, jElbowR, jWristR,
                    jSpine, jShoulderL, jElbowL, jWristL,
                    jHipR, jKneeR, jAnkleR, jHipL, jKneeL, jAnkleL};
                // citizen joint -> the asset joint whose rest position
                // it takes
                static const int kC2A[jCount] = {
                    0, 1, 3, 4, 10, 11, 12, 6, 7, 8, 16, 17, 18, 13, 14, 15};
                const float H = p.height;
                Skeleton sk2;
                for (int j = 0; j < jCount; ++j) {
                    sk2.pivot[j] = as.bind[kC2A[j]] * H;
                    sk2.rot[j] = sk.rot[j];
                }
                // the scan's arms already clear its torso
                sk2.rot[jShoulderL].z = 0.0f;
                sk2.rot[jShoulderR].z = 0.0f;
                solveSkeleton(sk2, root * glm::translate(glm::mat4(1.0f),
                                  glm::vec3(lateral, 0.0f, 0.0f)));
                const uint32_t base_row = uint32_t(frame_palette_.size());
                const glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(H));
                for (int k = 0; k < kNpcJoints; ++k) {
                    const int j = kA2C[k];
                    const glm::mat4 M = sk2.world[j] *
                        glm::translate(glm::mat4(1.0f), -sk2.pivot[j]) * S;
                    for (int r = 0; r < 3; ++r)
                        frame_palette_.push_back(
                            glm::vec4(M[0][r], M[1][r], M[2][r], M[3][r]));
                }
                NpcInstance ni;
                ni.a = glm::vec4(float(base_row),
                                 lookRand(uint32_t(pid_i), 0x51u),
                                 0.10f, 0.0f);
                frame_npc_[&as == &s_npc_[0] ? 0 : 1].push_back(ni);
                return;
            }
        }

        // ── the parts, SKINNED onto the joints (v3) ─────────────────
        // A part hangs off joint j; its top end skins to joint `up`,
        // its bottom end to joint `lo` (-1: none).  The three
        // transforms handed to the shader all map the part's unit box
        // to the world -- hung off j, off `up`, off `lo` -- and agree
        // exactly at the pivot the two joints share, so blending
        // between them around that pivot bends the surface without
        // opening it.  Adjacent parts overlap past their shared pivot
        // by about a radius and match cross-sections there (taper),
        // which closes the seams.  blend: half-width of each blend,
        // body metres.  mesh: 0 tube, 1 blob, 2 ball.
        auto rows = [](const glm::mat4& M, glm::vec4* r) {
            for (int i = 0; i < 3; ++i)
                r[i] = glm::vec4(M[0][i], M[1][i], M[2][i], M[3][i]);
        };
        auto spart = [&](int j, int up, int lo, glm::vec3 centre,
                        glm::vec3 half, float taper, float blend,
                        int mesh, int kind, glm::vec3 rgb, float style,
                        glm::vec3 accent,
                        glm::vec3 pre_pivot = glm::vec3(0.0f),
                        float pre_rot = 0.0f) {
            glm::mat4 L;
            if (pre_rot != 0.0f) {
                L = glm::translate(glm::mat4(1.0f), pre_pivot - sk.pivot[j]) *
                    glm::rotate(glm::mat4(1.0f), pre_rot, glm::vec3(1, 0, 0)) *
                    glm::translate(glm::mat4(1.0f), centre - pre_pivot);
            } else {
                L = glm::translate(glm::mat4(1.0f), centre - sk.pivot[j]);
            }
            L = L * glm::scale(glm::mat4(1.0f), half);
            const glm::mat4 M_self = sk.world[j] * L;
            auto other = [&](int k) {
                if (k < 0) return M_self;
                return sk.world[k] *
                       glm::translate(glm::mat4(1.0f),
                                      sk.pivot[j] - sk.pivot[k]) * L;
            };
            // the pivot the two joints share is the CHILD's
            auto meet_y = [&](int k) {
                return kJointParent[k] == j ? sk.pivot[k].y : sk.pivot[j].y;
            };
            const float hy = std::max(half.y, 1e-4f);
            SkinInstance si;
            rows(M_self, si.self);
            rows(other(up), si.up);
            rows(other(lo), si.lo);
            si.color = glm::vec4(rgb, 0.10f);
            si.extra = glm::vec4(float(kind), style, packRGB(accent),
                                 lookRand(uint32_t(pid_i), 0x51u));
            si.shape = glm::vec4(
                up >= 0 ? (meet_y(up) - centre.y) / hy : 10.0f,
                lo >= 0 ? (meet_y(lo) - centre.y) / hy : -10.0f,
                blend / hy, taper);
            (mesh == 0 ? frame_tube_ : mesh == 1 ? frame_blob_
                                                 : frame_ball_).push_back(si);
        };
        const float top_style =
            float(look.top_style) + (look.long_sleeve ? 4.0f : 0.0f);
        const float head_style =
            (look.long_hair ? 1.0f : 0.0f) + (look.female ? 2.0f : 0.0f) +
            (p.age == 1 || p.age == 2 ? 4.0f : 0.0f);
        const float bot_style = float(look.bottom_style);
        const bool shorts = look.bottom_style == 2;
        const glm::vec3 belt(0.12f, 0.10f, 0.08f);
        // the widths the parts meet at
        const float Ww = 0.135f * s * bw;                 // waist
        const float Wh = 0.165f * s * bw * hpw;           // hips
        const float Wc = 0.17f * s * bw * shw;            // chest
        // pelvis: waist at the top, hips at the bottom; torso: chest at
        // the top, waist at the bottom; the two overlap at the waist
        spart(jPelvis, jSpine, -1, {0.0f, 0.995f * s, 0.0f},
             {Ww, 0.135f * s, 0.095f * s * bw}, Wh / Ww, 0.05f * s, 1,
             kPartPelvis, look.bottom, bot_style, belt);
        spart(jSpine, -1, jPelvis, {0.0f, 1.225f * s, 0.0f},
             {Wc, 0.235f * s, 0.105f * s * bw}, Ww / Wc, 0.06f * s, 1,
             kPartTorso, look.top, top_style, skin);
        // v31: the neck flares into the shoulders (taper > 1: wider
        // at its foot), and the head is an EGG, narrower at the chin
        spart(jNeck, jHead, jSpine, {0.0f, 1.48f * s, 0.0f},
             {0.048f * s, 0.06f * s, 0.05f * s}, 1.35f, 0.03f * s, 0,
             kPartPlain, skin, 0.0f, skin);
        const float hy = 1.51f * s + 0.12f * s * hs;      // head centre
        spart(jHead, -1, jNeck, {0.0f, hy, 0.0f},
             {0.10f * s * hs, 0.12f * s * hs, 0.105f * s * hs}, 0.84f,
             0.03f * s, 2, kPartHead, skin, head_style, look.hair);
        // ears, nose: small, but they are what makes a head a face
        // from the side and behind
        for (int side = 0; side < 2; ++side) {
            const float ex = (side == 0 ? -1.0f : 1.0f) * 0.104f * s * hs;
            spart(jHead, -1, -1, {ex, hy - 0.012f * s * hs, -0.012f * s * hs},
                 {0.016f * s * hs, 0.026f * s * hs, 0.012f * s * hs}, 1.0f,
                 0.0f, 2, kPartPlain, skin, 0.0f, skin);
        }
        spart(jHead, -1, -1, {0.0f, hy - 0.028f * s * hs, 0.104f * s * hs},
             {0.013f * s * hs, 0.019f * s * hs, 0.016f * s * hs}, 1.0f,
             0.0f, 2, kPartPlain, skin, 0.0f, skin);
        // a HAIR VOLUME over the crown and back (the paint alone left
        // the head a bald ball in silhouette); seniors: some men bald
        const bool bald = p.age == 3 && !look.female &&
                          lookRand(uint32_t(pid_i), 0x61u) < 0.35f;
        if (!bald) {
            const float hv = look.long_hair ? 1.0f : 0.92f;
            spart(jHead, -1, -1, {0.0f, hy + 0.028f * s * hs, -0.022f * s * hs},
                 {0.106f * s * hs, 0.092f * s * hs * hv, 0.112f * s * hs},
                 0.90f, 0.0f, 1, kPartPlain, look.hair, 0.0f, look.hair);
        }
        if (look.long_hair) {
            spart(jHead, -1, -1, {0.0f, hy - 0.11f * s * hs, -0.065f * s * hs},
                 {0.115f * s * hs, 0.19f * s * hs, 0.055f * s * hs}, 1.0f,
                 0.0f, 1, kPartPlain, look.hair, 0.0f, look.hair);
        }
        // arms: upper arm from inside the shoulder past the elbow,
        // forearm from inside the elbow past the wrist, the hand
        for (int side = 0; side < 2; ++side) {
            const float x = (side == 0 ? -1.0f : 1.0f) * ax;
            const int sh = side == 0 ? jShoulderL : jShoulderR;
            spart(sh, jSpine, sh + 1, {x, 1.25f * s, 0.0f},
                 {0.06f * s, 0.18f * s, 0.06f * s}, 0.83f, 0.05f * s, 0,
                 kPartSleeve, look.top, top_style, skin);
            spart(sh + 1, sh, sh + 2, {x, 1.02f * s, 0.0f},
                 {0.05f * s, 0.17f * s, 0.05f * s}, 0.80f, 0.045f * s, 0,
                 kPartForearm, sleeve_lo, top_style, skin);
            // v31: a flat, tapered palm and a thumb instead of a ball
            spart(sh + 2, sh + 1, -1, {x, 0.83f * s, 0.005f * s},
                 {0.041f * s, 0.078f * s, 0.021f * s}, 0.72f, 0.03f * s, 1,
                 kPartHand, skin, 0.0f, skin);
            spart(sh + 2, -1, -1, {x - (side == 0 ? -1.0f : 1.0f) * 0.034f * s,
                                   0.865f * s, 0.024f * s},
                 {0.013f * s, 0.024f * s, 0.013f * s}, 0.8f, 0.0f, 2,
                 kPartPlain, skin, 0.0f, skin);
            // the DELTOID: the arm grows out of a shoulder, not out of
            // the side of the torso.  Hung off the spine so it stays
            // on the torso whatever the arm does.
            spart(jSpine, -1, -1, {x - (side == 0 ? -1.0f : 1.0f) * 0.012f * s,
                                   1.372f * s, 0.0f},
                 {0.064f * s * bw, 0.060f * s, 0.064f * s * bw}, 1.0f, 0.0f,
                 2, kPartSleeve, look.top, top_style, skin);
            // the ELBOW: a ball on the hinge, so a bent arm stays one
            // limb instead of two tubes meeting at an angle
            spart(sh + 1, -1, -1, {x, 1.13f * s, 0.0f},
                 {0.047f * s, 0.048f * s, 0.047f * s}, 1.0f, 0.0f, 2,
                 kPartPlain, sleeve_lo, 0.0f, sleeve_lo);
        }
        if (senior && look.stick && !sitting && !lying) {
            // a stick from the right hand to the ground: it hangs off
            // the wrist but is turned back through the arm's own
            // flexion (every hinge above it is about x), so it stays
            // near vertical, leaning a little ahead, and its foot
            // stays near the ground while the hand moves
            const glm::vec3 wood(0.36f, 0.24f, 0.13f);
            const float arm_flex = rot[jShoulderR].x + rot[jElbowR].x +
                                   rot[jSpine].x + rot[jPelvis].x;
            spart(jWristR, -1, -1, {ax, 0.845f * s - 0.41f * s, 0.03f * s},
                 {0.012f * s, 0.41f * s, 0.012f * s}, 1.0f, 0.0f, 1,
                 kPartPlain, wood, 0.0f, wood,
                 {ax, 0.845f * s, 0.005f * s}, -arm_flex - 0.08f);
        }
        // legs: thigh from inside the pelvis past the knee, shin from
        // inside the knee past the ankle, the shoe up over the ankle
        for (int side = 0; side < 2; ++side) {
            const float x = (side == 0 ? -1.0f : 1.0f) * lx;
            const int hp = side == 0 ? jHipL : jHipR;
            spart(hp, jPelvis, hp + 1, {x, 0.685f * s, 0.0f},
                 {0.085f * s * bw, 0.285f * s, 0.085f * s * bw}, 0.73f,
                 0.06f * s, 0, kPartThigh, look.bottom, bot_style, skin);
            spart(hp + 1, hp, hp + 2, {x, 0.285f * s, 0.0f},
                 {0.062f * s * bw, 0.255f * s, 0.062f * s * bw}, 0.72f,
                 0.05f * s, 0, kPartShin, shorts ? skin : look.bottom,
                 bot_style, skin);
            // v31: the KNEE, a ball on the hinge
            spart(hp + 1, -1, -1, {x, 0.47f * s, 0.005f * s},
                 {0.059f * s * bw, 0.060f * s, 0.059f * s * bw}, 1.0f, 0.0f,
                 2, kPartPlain, shorts ? skin : look.bottom, 0.0f,
                 shorts ? skin : look.bottom);
            if (look.heels) {
                // the foot pitched about its toe: the heel end stands
                // on a heel block, the toe stays on the ground
                const glm::vec3 toe(x, 0.0f, 0.13f * s);
                spart(hp + 2, -1, -1, {x, 0.025f * s, 0.03f * s},
                     {0.045f * s, 0.025f * s, 0.11f * s}, 1.0f, 0.0f, 1,
                     kPartShoe, look.shoe, 1.0f, skin, toe, 0.30f);
                spart(hp + 2, -1, -1, {x, -0.03f * s, -0.06f * s},
                     {0.016f * s, 0.04f * s, 0.016f * s}, 1.0f, 0.0f, 1,
                     kPartPlain, look.shoe, 0.0f, look.shoe, toe, 0.30f);
            } else {
                spart(hp + 2, hp + 1, -1, {x, 0.04f * s, 0.035f * s},
                     {0.055f * s, 0.045f * s, 0.12f * s}, 1.0f, 0.03f * s,
                     1, kPartShoe, look.shoe, 0.0f, skin);
            }
        }
        return;
    }

    // ── THE SEVEN-CUBE FIGURE (mid tier), wearing the same look ─────
    auto part = [&](glm::vec3 centre, glm::vec3 half, float pivot_rot,
                    glm::vec3 pivot, glm::vec3 rgb) {
        glm::mat4 M = hinge(root, pivot, pivot_rot) *
            glm::translate(glm::mat4(1.0f), centre) *
            glm::scale(glm::mat4(1.0f), half);
        frame_parts_.push_back({M, glm::vec4(rgb, 0.12f),
                                glm::vec4(0.0f)});
    };
    (void)col;
    // torso / head keep the walk-neutral frame
    part({0.0f, 1.18f * s, 0.0f},
         {0.17f * s * bw, 0.27f * s, 0.11f * s * bw}, 0.0f, {}, look.top);
    part({0.0f, 1.505f * s + 0.115f * s * hs, 0.0f},
         {0.105f * s * hs, 0.115f * s * hs, 0.105f * s * hs}, 0.0f, {},
         skin);
    // limbs swing about their pivots
    part({-0.235f * s * bw, 1.14f * s, 0.0f},
         {0.05f * s, 0.27f * s, 0.05f * s},
         arm_l, {-0.235f * s * bw, 1.40f * s, 0.0f}, sleeve_lo);
    part({0.235f * s * bw, 1.14f * s, 0.0f},
         {0.05f * s, 0.27f * s, 0.05f * s},
         arm_r, {0.235f * s * bw, 1.40f * s, 0.0f}, sleeve_lo);
    part({-0.09f * s, 0.47f * s, 0.0f},
         {0.07f * s * bw, 0.44f * s, 0.07f * s * bw},
         leg_l, {-0.09f * s, 0.90f * s, 0.0f}, look.bottom);
    part({0.09f * s, 0.47f * s, 0.0f},
         {0.07f * s * bw, 0.44f * s, 0.07f * s * bw},
         leg_r, {0.09f * s, 0.90f * s, 0.0f}, look.bottom);
}

void CitizenSystem::collectShadowGeometry(ActorShadowGeometry& out) const {
    out.positions.clear(); out.indices.clear();
    if (!loaded_ || !s_pipeline_) return;
    for (const auto& part : frame_parts_)
        out.append(s_shadow_cube_, [&](const glm::vec3& p) {
            return glm::vec3(part.xform * glm::vec4(p, 1.0f));
        });
    auto transform = [](const glm::vec4* rows, const glm::vec4& p) {
        return glm::vec3(glm::dot(rows[0], p), glm::dot(rows[1], p), glm::dot(rows[2], p));
    };
    auto skin = [&](const std::vector<SkinInstance>& parts, const ActorShadowGeometry& mesh) {
        for (const auto& part : parts) out.append(mesh, [&](const glm::vec3& v) {
            // Same taper and pivot blend as citizen_skin.vert.
            const float f = glm::mix(part.shape.w, 1.0f, (v.y + 1.0f) * 0.5f);
            const glm::vec4 p(v.x * f, v.y, v.z * f, 1.0f);
            float wu = 0.0f, wl = 0.0f;
            if (part.shape.z > 0.0f) {
                wu = glm::clamp(0.5f + (v.y - part.shape.x) / (2.0f * part.shape.z), 0.0f, 1.0f);
                wl = glm::clamp(0.5f - (v.y - part.shape.y) / (2.0f * part.shape.z), 0.0f, 1.0f);
            }
            return std::max(0.0f, 1.0f - wu - wl) * transform(part.self, p)
                 + wu * transform(part.up, p) + wl * transform(part.lo, p);
        });
    };
    if (s_skin_pipeline_) {
        skin(frame_tube_, s_shadow_tube_); skin(frame_blob_, s_shadow_blob_);
        skin(frame_ball_, s_shadow_ball_);
    }
    if (!s_npc_ready_) return;
    for (int c = 0; c < 2; ++c) {
        const auto& asset = s_npc_[c];
        if (!asset.ok) continue;
        for (const auto& inst : frame_npc_[c]) {
            const size_t row = size_t(inst.a.x + 0.5f);
            if (row + kNpcRows > frame_palette_.size()) continue;
            const uint32_t base = uint32_t(out.positions.size());
            for (size_t v = 0; v < asset.shadow.positions.size(); ++v) {
                const glm::vec4 p(asset.shadow.positions[v], 1.0f);
                glm::vec3 world(0.0f); float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    const float w = float(asset.shadow_weights[v][k]);
                    if (w <= 0.0f) continue;
                    const size_t joint = asset.shadow_joints[v][k];
                    world += w * transform(frame_palette_.data() + row + joint * 3, p);
                    sum += w;
                }
                out.positions.push_back(sum > 0.0f ? world / sum : transform(frame_palette_.data() + row, p));
            }
            for (uint32_t i : asset.shadow.indices) out.indices.push_back(base + i);
        }
    }
}

void CitizenSystem::draw(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const er::DescriptorSetList& desc_sets,
    const std::shared_ptr<er::ImageView>& color_view,
    const std::shared_ptr<er::ImageView>& depth_view,
    const glm::uvec2& buffer_size,
    const std::vector<std::shared_ptr<er::ImageView>>& gbuffer) {
    const bool deferred = !gbuffer.empty();
    if (deferred && (gbuffer.size() != 4 || !s_gbuf_pipeline_)) return;
    if (!loaded_ || !s_pipeline_) return;
    if (frame_parts_.empty() && frame_tube_.empty() &&
        frame_blob_.empty() && frame_ball_.empty() &&
        frame_npc_[0].empty() && frame_npc_[1].empty()) return;
    if (!color_view || !depth_view) return;
    if (!s_device_) return;
    // ── UPLOAD THE INSTANCE STREAM ──────────────────────────────────
    // Grown in powers of two and never shrunk: the crowd in view swings
    // frame to frame as the camera turns, and reallocating a
    // multi-megabyte buffer on every swing would cost more than the
    // headroom it reclaims.  HOST_VISIBLE | HOST_COHERENT so the fill
    // is a memcpy with no staging copy and no barrier.
    {
        const uint32_t need = uint32_t(frame_parts_.size());
        if (!s_inst_buf_ || need > s_inst_capacity_) {
            uint32_t cap = s_inst_capacity_ ? s_inst_capacity_ : 4096u;
            while (cap < need) cap *= 2u;
            if (s_inst_buf_) s_inst_buf_->destroy(s_device_);
            s_inst_buf_ = std::make_shared<er::BufferInfo>();
            er::Helper::createBuffer(
                s_device_,
                SET_2_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT,
                                TRANSFER_DST_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                HOST_COHERENT_BIT),
                0,
                s_inst_buf_->buffer,
                s_inst_buf_->memory,
                std::source_location::current(),
                uint64_t(cap) * sizeof(PartInstance),
                nullptr);
            s_inst_capacity_ = cap;
            std::cout << "[citizen] instance buffer -> " << cap
                      << " parts ("
                      << (uint64_t(cap) * sizeof(PartInstance)) / 1024
                      << " KiB)" << std::endl;
        }
        if (!frame_parts_.empty()) {
            s_device_->updateBufferMemory(
                s_inst_buf_->memory,
                uint64_t(frame_parts_.size()) * sizeof(PartInstance),
                frame_parts_.data());
        }
    }
    // ── THE SKIN STREAM: tube, blob, ball parts in one buffer ───────
    const uint32_t n_tube = uint32_t(frame_tube_.size());
    const uint32_t n_blob = uint32_t(frame_blob_.size());
    const uint32_t n_ball = uint32_t(frame_ball_.size());
    {
        const uint32_t need = n_tube + n_blob + n_ball;
        if (need && (!s_skin_buf_ || need > s_skin_capacity_)) {
            uint32_t cap = s_skin_capacity_ ? s_skin_capacity_ : 1024u;
            while (cap < need) cap *= 2u;
            if (s_skin_buf_) s_skin_buf_->destroy(s_device_);
            s_skin_buf_ = std::make_shared<er::BufferInfo>();
            er::Helper::createBuffer(
                s_device_,
                SET_2_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT,
                                TRANSFER_DST_BIT),
                SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                HOST_COHERENT_BIT),
                0,
                s_skin_buf_->buffer,
                s_skin_buf_->memory,
                std::source_location::current(),
                uint64_t(cap) * sizeof(SkinInstance),
                nullptr);
            s_skin_capacity_ = cap;
            std::cout << "[citizen] skin instance buffer -> " << cap
                      << " parts ("
                      << (uint64_t(cap) * sizeof(SkinInstance)) / 1024
                      << " KiB)" << std::endl;
        }
        uint64_t off = 0;
        for (const auto* v : {&frame_tube_, &frame_blob_, &frame_ball_}) {
            if (!v->empty()) {
                s_device_->updateBufferMemory(
                    s_skin_buf_->memory,
                    uint64_t(v->size()) * sizeof(SkinInstance),
                    v->data(), off);
            }
            off += uint64_t(v->size()) * sizeof(SkinInstance);
        }
    }
    if (!s_pipeline_layout_ || !s_cube_pos_ || !s_cube_nrm_ ||
        !s_cube_idx_) {
        return;
    }
    // CONTRACT: desc_sets must be exactly the sets s_pipeline_layout_
    // was created with, in order — { PBR global, view camera } — and
    // none of them null.  A longer list, or a null handle, is undefined
    // behaviour in vkCmdBindDescriptorSets and crashes inside the
    // driver rather than failing anywhere we can see it.  Degrade to
    // "no citizens this frame" instead; the caller's log tells us why.
    for (const auto& ds : desc_sets) {
        if (!ds) {
            static bool s_warned = false;
            if (!s_warned) {
                s_warned = true;
                std::cout << "[citizen] draw skipped: null descriptor "
                             "set in the bind list (expected { PBR "
                             "global, view camera })" << std::endl;
            }
            return;
        }
    }

    er::RenderingAttachmentInfo color_att;
    color_att.image_view = color_view;
    color_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    color_att.load_op = er::AttachmentLoadOp::LOAD;
    color_att.store_op = er::AttachmentStoreOp::STORE;
    er::RenderingAttachmentInfo depth_att;
    depth_att.image_view = depth_view;
    depth_att.image_layout =
        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
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
    viewports[0].x = 0;
    viewports[0].y = 0;
    viewports[0].width = float(buffer_size.x);
    viewports[0].height = float(buffer_size.y);
    viewports[0].min_depth = 0.0f;
    viewports[0].max_depth = 1.0f;
    scissors[0].offset = {0, 0};
    scissors[0].extent = {buffer_size.x, buffer_size.y};

    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, deferred ? s_gbuf_pipeline_ : s_pipeline_);
    cmd_buf->setViewports(viewports, 0, 1);
    cmd_buf->setScissors(scissors, 0, 1);
    cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS,
                                s_pipeline_layout_, desc_sets);
    std::vector<std::shared_ptr<er::Buffer>> vbs = {
        s_cube_pos_->buffer, s_cube_nrm_->buffer,
        s_inst_buf_->buffer};
    std::vector<uint64_t> offs = {0, 0, 0};
    cmd_buf->bindVertexBuffers(0, vbs, offs);
    cmd_buf->bindIndexBuffer(s_cube_idx_->buffer, 0,
                             er::IndexType::UINT32);
    // ONE CALL FOR THE WHOLE TOWN.  This was a pushConstants and a
    // drawIndexed per box, which is what actually capped the visible
    // population: the cost was never the 12 triangles of a box, it was
    // the draw call in front of them.
    if (!frame_parts_.empty()) {
        cmd_buf->drawIndexed(s_cube_index_count_,
                             uint32_t(frame_parts_.size()));
    }
    // ...and the SKINNED figures: the skin pipeline, three draws (one
    // per mesh) over one instance buffer at their offsets.
    if (n_tube + n_blob + n_ball && s_skin_pipeline_ && s_skin_buf_) {
        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS,
                              deferred ? s_skin_gbuf_pipeline_ : s_skin_pipeline_);
        cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS,
                                    s_pipeline_layout_, desc_sets);
        struct Draw { const std::shared_ptr<er::BufferInfo>* pos;
                      const std::shared_ptr<er::BufferInfo>* nrm;
                      const std::shared_ptr<er::BufferInfo>* idx;
                      uint32_t count; uint32_t n; };
        const Draw draws[3] = {
            {&s_tube_pos_, &s_tube_nrm_, &s_tube_idx_, s_tube_index_count_, n_tube},
            {&s_round_pos_, &s_round_nrm_, &s_round_idx_, s_round_index_count_, n_blob},
            {&s_ball_pos_, &s_ball_nrm_, &s_ball_idx_, s_ball_index_count_, n_ball}};
        uint32_t first = 0;
        for (const auto& d : draws) {
            if (d.n && *d.pos && *d.nrm && *d.idx) {
                std::vector<std::shared_ptr<er::Buffer>> vbs2 = {
                    (*d.pos)->buffer, (*d.nrm)->buffer, s_skin_buf_->buffer};
                cmd_buf->bindVertexBuffers(0, vbs2, offs);
                cmd_buf->bindIndexBuffer((*d.idx)->buffer, 0,
                                         er::IndexType::UINT32);
                cmd_buf->drawIndexed(d.count, d.n, 0, 0, first);
            }
            first += d.n;
        }
    }
    // ...and the CHARACTER MESHES: the palette and the instance
    // stream (man instances, then woman) go up, one draw per
    // character with its own descriptor set behind the two globals.
    const uint32_t n_npc0 = uint32_t(frame_npc_[0].size());
    const uint32_t n_npc1 = uint32_t(frame_npc_[1].size());
    if (n_npc0 + n_npc1 && s_npc_ready_ && s_npc_pipeline_ &&
        s_npc_inst_buf_ && s_npc_palette_buf_ && !frame_palette_.empty() &&
        n_npc0 + n_npc1 <= kMaxNpc && desc_sets.size() >= 2) {
        s_device_->updateBufferMemory(
            s_npc_palette_buf_->memory,
            uint64_t(frame_palette_.size()) * sizeof(glm::vec4),
            frame_palette_.data());
        uint64_t off = 0;
        for (int c = 0; c < 2; ++c) {
            if (frame_npc_[c].empty()) continue;
            s_device_->updateBufferMemory(
                s_npc_inst_buf_->memory,
                uint64_t(frame_npc_[c].size()) * sizeof(NpcInstance),
                frame_npc_[c].data(), off);
            off += uint64_t(frame_npc_[c].size()) * sizeof(NpcInstance);
        }
        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS,
                              deferred ? s_npc_gbuf_pipeline_ : s_npc_pipeline_);
        uint32_t first = 0;
        for (int c = 0; c < 2; ++c) {
            const uint32_t n = uint32_t(frame_npc_[c].size());
            const NpcAsset& as = s_npc_[c];
            if (n && as.ok && as.desc) {
                er::DescriptorSetList sets = {desc_sets[0], desc_sets[1],
                                              as.desc};
                cmd_buf->bindDescriptorSets(er::PipelineBindPoint::GRAPHICS,
                                            s_npc_layout_, sets);
                std::vector<std::shared_ptr<er::Buffer>> vbs3 = {
                    as.vb->buffer, s_npc_inst_buf_->buffer};
                std::vector<uint64_t> offs3 = {0, 0};
                cmd_buf->bindVertexBuffers(0, vbs3, offs3);
                cmd_buf->bindIndexBuffer(as.ib->buffer, 0,
                                         er::IndexType::UINT32);
                cmd_buf->drawIndexed(as.index_count, n, 0, 0, first);
            }
            first += n;
        }
    }
    cmd_buf->endDynamicRendering();
}

void CitizenSystem::destroy(const std::shared_ptr<er::Device>& device) {
    (void)device;
    persons_.clear();
    sim_.clear();
    frame_parts_.clear();
    for (auto* v : {&frame_tube_, &frame_blob_, &frame_ball_}) {
        v->clear();
        v->shrink_to_fit();
    }
    frame_npc_[0].clear();
    frame_npc_[1].clear();
    frame_palette_.clear();
    frame_palette_.shrink_to_fit();
    is_npc_.clear();
    is_detailed_.clear();
    is_detailed_.shrink_to_fit();
    is_fine_.clear();
    is_fine_.shrink_to_fit();
    near_ids_.clear();
    near_ids_.shrink_to_fit();
    loaded_ = false;
}

}  // namespace game_object
}  // namespace engine
