#include "citizen_system.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <filesystem>
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
    kActDeskWork, kActPlay, kActSleepish
};
// Which piece of furniture a home anchor resolved to.  kAnchorNone
// is "this house has none of it" — the caller falls back.
enum AnchorKind { kAnchorNone = 0, kAnchorBed, kAnchorStove, kAnchorSeat };
enum Duty {
    kDutyResident = 0, kDutyDoctor, kDutyNurse, kDutyTeacher,
    kDutyOfficial, kDutyPolice, kDutyFire, kDutyChef, kDutyWaiter,
    kDutyShop, kDutyOffice, kDutyHomemaker, kDutyStudent, kDutyInfant,
    kDutyRetiree, kDutyCommuter, kDutyChildcare, kDutyWorker
};

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

std::shared_ptr<er::PipelineLayout> CitizenSystem::s_pipeline_layout_;
std::shared_ptr<er::Pipeline>       CitizenSystem::s_pipeline_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_pos_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_nrm_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_idx_;
uint32_t                            CitizenSystem::s_cube_index_count_ = 0;
std::shared_ptr<er::Device>         CitizenSystem::s_device_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_inst_buf_;
uint32_t                            CitizenSystem::s_inst_capacity_ = 0;

void CitizenSystem::initStaticMembers(
    const std::shared_ptr<er::Device>& device,
    const er::DescriptorSetLayoutList& global_desc_set_layouts,
    const er::GraphicPipelineInfo& graphic_pipeline_info,
    const er::PipelineRenderbufferFormats& frame_buffer_format) {

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
    std::vector<er::VertexInputAttributeDescription> attribs(7);
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
    for (int k = 0; k < 5; ++k) {
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
}

void CitizenSystem::destroyStaticMembers(
    const std::shared_ptr<er::Device>& device) {
    if (s_pipeline_layout_) device->destroyPipelineLayout(
        s_pipeline_layout_);
    s_pipeline_layout_ = nullptr;
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
}

bool CitizenSystem::loadCity(const std::string& city_json_path,
                             const std::string& world_json_path) {
    using nlohmann::json;
    loaded_ = false;
    houses_.clear();
    house_school_seats_.clear();
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
        size_t n = hs.at("yaw").size();
        houses_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            houses_.emplace_back(float(t[3 * i]), float(t[3 * i + 1]),
                                 float(t[3 * i + 2]));
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
        } catch (const std::exception& ce) {
            std::cout << "[citizen] city json unusable (" << ce.what()
                      << ") — falling back to synthesized residents"
                      << std::endl;
            buildings_.clear();
            persons_.clear();
        }
        }   // if (have_city)

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
        if (real_schools && house_school_seats_[i] > 0) continue;
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

void CitizenSystem::harvestFurniture() {
    beds_.clear();   stoves_.clear();   seats_.clear();
    house_beds_.assign(houses_.size(), glm::ivec2(0));
    house_stoves_.assign(houses_.size(), glm::ivec2(0));
    house_seats_.assign(houses_.size(), glm::ivec2(0));
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
    auto nearestHouse = [this](const glm::vec3& p) -> int {
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
        std::vector<int> count(houses_.size(), 0);
        for (size_t i = 0; i < recs.size(); ++i) {
            const int hi = nearestHouse(recs[i].t);
            owner[i] = hi;
            if (hi >= 0) ++count[size_t(hi)];
        }
        int run = 0;
        for (size_t h = 0; h < houses_.size(); ++h) {
            slice[h] = glm::ivec2(run, count[h]);
            run += count[h];
        }
        out.assign(size_t(run), Furniture{});
        std::vector<int> cursor(houses_.size(), 0);
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
    std::cout << "[citizen] furniture anchors: " << nb << " bed(s), "
              << ns << " cooktop(s), " << nc << " chair(s) bound to "
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
        house = buildings_[size_t(s.place)].house;
        // Their counted slot AT THIS building — the same index placePos
        // uses for the arrival scatter, so two people never resolve to
        // one chair while another stands empty.
        seat_i = (s.place == p.shop_b) ? p.shop_slot : p.slot;
    }
    if (house < 0 || house >= int(houses_.size())) return a;
    const std::vector<Furniture>*  arr = nullptr;
    const std::vector<glm::ivec2>* sl  = nullptr;
    int kind = kAnchorNone;
    switch (activity) {
    case kActSleepish: arr = &beds_;   sl = &house_beds_;
                       kind = kAnchorBed;   break;
    case kActCook:     arr = &stoves_; sl = &house_stoves_;
                       kind = kAnchorStove; break;
    case kActSit:
    case kActDeskWork: arr = &seats_;  sl = &house_seats_;
                       kind = kAnchorSeat;  break;
    default: return a;
    }
    if (sl->size() != houses_.size()) return a;
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
        // Out in the yard beside the house — same seat-based fan so a
        // family stepping outside does not stand in a single column,
        // just at garden spacing rather than room spacing.
        const glm::vec3& h = houses_[p.house];
        const float rr = kYardSpreadR * std::sqrt(hsi / hcap);
        return {h.x + 12.0f + std::cos(hth) * rr + jx * 0.5f,
                h.y,
                h.z + 12.0f + std::sin(hth) * rr + jz * 0.5f};
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
        glm::vec3 d = target - a.pos;
        d.y = 0.0f;
        float dist = glm::length(d);
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
                    p.house, st.place);
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

    if (far_thresh_ < kMinAngular) far_thresh_ = kMinAngular;
    size_t far_emitted = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!sim_[i].inited) continue;
        const float dx = sim_[i].pos.x - camera_pos.x;
        const float dz = sim_[i].pos.z - camera_pos.z;
        const float d2 = dx * dx + dz * dz;
        if (d2 > kShowRadius * kShowRadius) continue;
        if (is_detailed[i]) {
            emitPerson(int(i), sim_[i], persons_[i], true);
        } else {
            const float dist = std::sqrt(std::max(d2, 1.0f));
            if (persons_[i].height / dist < far_thresh_) continue;
            // hard ceiling so the first frame at a new vantage cannot
            // burst-draw the whole town before the controller reacts
            if (far_emitted >= kMaxFarParts + kMaxFarParts / 2) continue;
            ++far_emitted;
            emitPerson(int(i), sim_[i], persons_[i], false);
        }
    }
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
                  << " (far_thresh " << far_thresh_ << ")";
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
                  << int(std::sqrt(ddx * ddx + ddz * ddz)) << "m away"
                  << std::endl;
    }
}

void CitizenSystem::emitPerson(int pid_i, const SimState& a,
                               const Person& p, bool detailed) {
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
        frame_parts_.push_back({M, glm::vec4(dutyColor(p.duty), 0.12f)});
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

    const float s = p.height / 1.75f;
    const float bw = p.bulk;
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
    bool sitting = false;
    switch (act) {
    case kActWalk:
        leg_l = swing * 0.55f;
        leg_r = -swing * 0.55f;
        arm_l = -swing * 0.45f;
        arm_r = swing * 0.45f;
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
            arm_l = arm_r = act == kActDeskWork ? -0.9f : -0.4f;
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
        torso_pitch = 0.12f;
        break;
    case kActBrowse:
        arm_r = -0.9f;
        torso_pitch = 0.08f;
        break;
    case kActCare:
        // rounds: slow sway + attending arm
        arm_l = -0.7f + 0.2f * std::sin(anim_phase * 2.0f);
        torso_pitch = 0.16f;
        break;
    case kActPlay:
        arm_l = std::sin(anim_t_ * 5.0f + float(pid_i)) * 0.8f;
        arm_r = -std::sin(anim_t_ * 5.0f + float(pid_i)) * 0.8f;
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

    const glm::mat4 R =
        glm::rotate(glm::mat4(1.0f), body_yaw, glm::vec3(0, 1, 0)) *
        glm::rotate(glm::mat4(1.0f), torso_pitch, glm::vec3(1, 0, 0));
    auto part = [&](glm::vec3 centre, glm::vec3 half, float pivot_rot,
                    glm::vec3 pivot) {
        glm::mat4 M = glm::translate(glm::mat4(1.0f),
                                     glm::vec3(body_pos.x, root_y,
                                               body_pos.z)) * R;
        if (pivot_rot != 0.0f) {
            M = M * glm::translate(glm::mat4(1.0f), pivot) *
                glm::rotate(glm::mat4(1.0f), pivot_rot,
                            glm::vec3(1, 0, 0)) *
                glm::translate(glm::mat4(1.0f), -pivot);
        }
        M = M * glm::translate(glm::mat4(1.0f), centre) *
            glm::scale(glm::mat4(1.0f), half);
        frame_parts_.push_back({M, col});
    };
    // torso / head keep the walk-neutral frame
    part({0.0f, 1.18f * s, 0.0f},
         {0.17f * s * bw, 0.27f * s, 0.11f * s * bw}, 0.0f, {});
    glm::vec4 head_col = glm::vec4(glm::mix(
        glm::vec3(0.87f, 0.72f, 0.58f), glm::vec3(col), 0.15f), 0.12f);
    {
        glm::vec4 keep = col;
        col = head_col;
        part({0.0f, 1.62f * s, 0.0f},
             {0.105f * s, 0.115f * s, 0.105f * s}, 0.0f, {});
        col = keep;
    }
    // limbs swing about their pivots
    part({-0.235f * s * bw, 1.14f * s, 0.0f},
         {0.05f * s, 0.27f * s, 0.05f * s},
         arm_l, {-0.235f * s * bw, 1.40f * s, 0.0f});
    part({0.235f * s * bw, 1.14f * s, 0.0f},
         {0.05f * s, 0.27f * s, 0.05f * s},
         arm_r, {0.235f * s * bw, 1.40f * s, 0.0f});
    part({-0.09f * s, 0.47f * s, 0.0f},
         {0.07f * s * bw, 0.44f * s, 0.07f * s * bw},
         leg_l, {-0.09f * s, 0.90f * s, 0.0f});
    part({0.09f * s, 0.47f * s, 0.0f},
         {0.07f * s * bw, 0.44f * s, 0.07f * s * bw},
         leg_r, {0.09f * s, 0.90f * s, 0.0f});
}

void CitizenSystem::draw(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const er::DescriptorSetList& desc_sets,
    const std::shared_ptr<er::ImageView>& color_view,
    const std::shared_ptr<er::ImageView>& depth_view,
    const glm::uvec2& buffer_size) {
    if (!loaded_ || frame_parts_.empty() || !s_pipeline_) return;
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
        s_device_->updateBufferMemory(
            s_inst_buf_->memory,
            uint64_t(need) * sizeof(PartInstance),
            frame_parts_.data());
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

    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, s_pipeline_);
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
    cmd_buf->drawIndexed(s_cube_index_count_,
                         uint32_t(frame_parts_.size()));
    cmd_buf->endDynamicRendering();
}

void CitizenSystem::destroy(const std::shared_ptr<er::Device>& device) {
    (void)device;
    persons_.clear();
    sim_.clear();
    frame_parts_.clear();
    is_detailed_.clear();
    is_detailed_.shrink_to_fit();
    near_ids_.clear();
    near_ids_.shrink_to_fit();
    loaded_ = false;
}

}  // namespace game_object
}  // namespace engine
