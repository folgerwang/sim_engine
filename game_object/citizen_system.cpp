#include "citizen_system.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include <glm/gtc/matrix_transform.hpp>

#include "json.hpp"   // vendored at third_parties/tinygltf/json.hpp
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
constexpr size_t kMaxDetailed = 320;
constexpr size_t kMaxFarParts = 12000;    // far-tier draw budget
constexpr float kShowRadius = 10000.0f;   // 10 km
constexpr float kMinAngular = 0.0003f;    // height/dist cutoff (floor)
constexpr float kGroundClampRadius = 400.0f;
// The clock runs 60x real time, so schedule-accurate commuting needs
// faster-than-life legs: at kWalkTimeScale 6 a 500 m commute costs one
// game hour (about a real minute of visible walking) instead of six.
// Full 60x would be teleport-sprinting; 6x reads as "people getting
// places" while staying watchable up close.
constexpr float kWalkTimeScale = 6.0f;
constexpr size_t kFarClampPerFrame = 1024; // staggered height refresh
// How many houses per grid cell get promoted to destinations by
// synthesizeResidents (first four workplaces, the rest shops).  Lives
// out here because a LOCAL class may not declare a static data member
// (MSVC C2258 / [class.local]) — the cell picker is a local struct.
constexpr int kCellAnchors = 6;

}  // namespace

std::shared_ptr<er::PipelineLayout> CitizenSystem::s_pipeline_layout_;
std::shared_ptr<er::Pipeline>       CitizenSystem::s_pipeline_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_pos_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_nrm_;
std::shared_ptr<er::BufferInfo>     CitizenSystem::s_cube_idx_;
uint32_t                            CitizenSystem::s_cube_index_count_ = 0;

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

    std::vector<er::VertexInputBindingDescription> bindings(2);
    std::vector<er::VertexInputAttributeDescription> attribs(2);
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
    // Without the city json we synthesize one resident per house
    // instead (see synthesizeResidents).
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
                     "one resident per house."
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
        } catch (const std::exception& ce) {
            std::cout << "[citizen] city json unusable (" << ce.what()
                      << ") — falling back to synthesized residents"
                      << std::endl;
            buildings_.clear();
            persons_.clear();
        }
        }   // if (have_city)

        // No city json, or one that yielded nobody (an old export, or a
        // map whose households never got allocated): give every house a
        // resident so the town is inhabited from the moment the terrain
        // finishes loading.
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
    // resident per house with a real day built around them.
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
    std::unordered_map<uint64_t, int> school_bi;
    auto addBuilding = [&](int house_idx, const char* type) -> int {
        const glm::vec3& h = houses_[house_idx];
        Building b;
        b.type = type;
        b.centre = {h.x, h.z};
        // Stand arrivals just OUTSIDE the shell — placePos jitters
        // +/-3 m around this, and an entrance ON the house centre would
        // park half the staff inside the walls.
        b.entrance = {h.x + 4.5f, h.z + 4.5f};
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
    for (uint64_t k : sortedKeys(school_cells)) {
        const CellPick& c = school_cells[k];
        if (c.n > 0) school_bi[k] = addBuilding(c.idx[0], "school");
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
    auto lookup1 = [](const std::unordered_map<uint64_t, int>& m,
                      uint64_t k) -> int {
        auto it = m.find(k);
        return it == m.end() ? -1 : it->second;
    };
    // Headcount per building, so the arrival scatter can widen with it.
    std::vector<int> occupancy(buildings_.size(), 0);

    // ── 2. RESIDENTS ─────────────────────────────────────────────────
    persons_.reserve(houses_.size());
    for (size_t i = 0; i < houses_.size(); ++i) {
        // Mixed id for every attribute draw below — see mix32.
        const uint32_t hi = mix32(static_cast<uint32_t>(i));
        const glm::vec3& home = houses_[i];
        Person p;
        p.house = static_cast<int>(i);
        // Duty drives the body colour in emitPerson, where the person
        // spends their day, and what they do once they get there.
        const float du = h01(hi, 0x51A1u);
        p.duty = du < 0.34f ? kDutyWorker
               : du < 0.48f ? kDutyOffice
               : du < 0.60f ? kDutyShop
               : du < 0.76f ? kDutyStudent
               : du < 0.88f ? kDutyHomemaker
                            : kDutyRetiree;
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
        int work   = pick(work_bi, wk, uint32_t(i) * 2654435761u);
        int shop   = pick(shop_bi, wk, uint32_t(i) * 40503u + 7u);
        int school = lookup1(school_bi, sk);
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
        if (dest >= 0 && dest < int(occupancy.size())) ++occupancy[dest];
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
        // Jitter can reorder two hinges that started close together;
        // currentStep walks the list assuming ascending minutes.
        auto by_time = [](const Step& a, const Step& b) {
            return a.minutes < b.minutes;
        };
        std::sort(p.weekday.begin(), p.weekday.end(), by_time);
        std::sort(p.weekend.begin(), p.weekend.end(), by_time);
        persons_.push_back(std::move(p));
    }
    // A doorway that 80 people report to needs a forecourt, not a
    // 6 m jitter box: widen with sqrt(headcount) so density stays
    // roughly constant however many a cell sends.
    for (size_t b = 0; b < buildings_.size() && b < occupancy.size(); ++b) {
        buildings_[b].spread =
            6.0f + 1.8f * std::sqrt(float(occupancy[b]));
    }
    std::cout << "[citizen] synthesized " << persons_.size()
              << " resident(s) — one per house — around "
              << buildings_.size() << " workplace/shop/school anchor(s)"
              << std::endl;
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

glm::vec3 CitizenSystem::placePos(const Person& p, const Step& s,
                                  int pid) const {
    // deterministic per-person jitter so a crowd at one entrance
    // spreads instead of z-fighting inside one another
    const float jx = (h01(pid, 11u) - 0.5f) * 6.0f;
    const float jz = (h01(pid, 23u) - 0.5f) * 6.0f;
    if (s.place == -1 || s.place >= int(buildings_.size())) {
        const glm::vec3& h = houses_[p.house];
        return {h.x + jx * 0.5f, h.y, h.z + jz * 0.5f};
    }
    if (s.place == -2) {
        const glm::vec3& h = houses_[p.house];
        return {h.x + 12.0f + jx, h.y, h.z + 12.0f + jz};
    }
    const Building& b = buildings_[s.place];
    // jx/jz span +/-3 m (a 6 m box); rescale to this building's own
    // forecourt width (6 m for city-json buildings, wider for a
    // synthesized workplace with a big headcount).
    const float sc = b.spread / 6.0f;
    return {b.entrance.x + jx * sc, b.base_y, b.entrance.y + jz * sc};
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
    for (size_t i = 0; i < n; ++i) {
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
        // near-camera: exact terrain clamp every frame
        if (ground_) {
            const float dcx = a.pos.x - camera_pos.x;
            const float dcz = a.pos.z - camera_pos.z;
            if (dcx * dcx + dcz * dcz <
                kGroundClampRadius * kGroundClampRadius) {
                float gy;
                glm::vec3 gn;
                if (ground_(a.pos.x, a.pos.z, a.pos.y + 1.0f, gy, gn)) {
                    a.pos.y = gy;
                }
            }
        }
    }
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
    std::vector<std::pair<float, int>> near_ids;
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
    std::vector<uint8_t> is_detailed(n, 0);
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

    const float tod = std::fmod(clock_min_, 1440.0f);
    const auto& sched = scheduleOf(p);
    const Step home_step{};
    const Step& st = a.cur_step >= 0 ? sched[a.cur_step] : home_step;
    int act = a.walking ? kActWalk : st.activity;
    if (!a.walking && a.gesture_t > 0.0f && a.gesture_t < 1.2f &&
        st.place == -1) {
        act = kActBrowse;              // arm-forward: opening the door
    }
    // night: everyone home is asleep-ish (idle, no cooking at 03:00)
    if (!a.walking && st.place == -1 &&
        (tod < 6.0f * 60.0f || tod > 22.5f * 60.0f)) {
        act = kActIdle;
    }

    const float s = p.height / 1.75f;
    const float bw = p.bulk;
    glm::vec4 col(dutyColor(p.duty), 0.12f);

    const float swing = std::sin(a.phase);
    float root_y = a.pos.y;
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
    case kActSit:
    case kActDeskWork:
        sitting = true;
        leg_l = leg_r = -1.45f;        // thighs forward
        arm_l = arm_r = act == kActDeskWork ? -0.9f : -0.4f;
        break;
    case kActCook:
        arm_r = -1.1f + 0.25f * std::sin(clock_min_ * 4.0f +
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
        arm_l = -0.7f + 0.2f * std::sin(clock_min_ * 2.0f);
        torso_pitch = 0.16f;
        break;
    case kActPlay:
        arm_l = std::sin(clock_min_ * 5.0f + pid_i) * 0.8f;
        arm_r = -std::sin(clock_min_ * 5.0f + pid_i) * 0.8f;
        break;
    default:
        arm_l = arm_r = 0.06f * std::sin(clock_min_ * 1.5f +
                                         float(pid_i));
        break;
    }
    if (sitting) root_y -= 0.42f * s;

    const glm::mat4 R =
        glm::rotate(glm::mat4(1.0f), a.yaw, glm::vec3(0, 1, 0)) *
        glm::rotate(glm::mat4(1.0f), torso_pitch, glm::vec3(1, 0, 0));
    auto part = [&](glm::vec3 centre, glm::vec3 half, float pivot_rot,
                    glm::vec3 pivot) {
        glm::mat4 M = glm::translate(glm::mat4(1.0f),
                                     glm::vec3(a.pos.x, root_y,
                                               a.pos.z)) * R;
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
        s_cube_pos_->buffer, s_cube_nrm_->buffer};
    std::vector<uint64_t> offs = {0, 0};
    cmd_buf->bindVertexBuffers(0, vbs, offs);
    cmd_buf->bindIndexBuffer(s_cube_idx_->buffer, 0,
                             er::IndexType::UINT32);
    for (const auto& part : frame_parts_) {
        glsl::CitizenDrawParams params{};
        params.transform = part.xform;
        params.color = part.color;
        cmd_buf->pushConstants(
            SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
            s_pipeline_layout_, &params, sizeof(params));
        cmd_buf->drawIndexed(s_cube_index_count_);
    }
    cmd_buf->endDynamicRendering();
}

void CitizenSystem::destroy(const std::shared_ptr<er::Device>& device) {
    (void)device;
    persons_.clear();
    sim_.clear();
    frame_parts_.clear();
    loaded_ = false;
}

}  // namespace game_object
}  // namespace engine
