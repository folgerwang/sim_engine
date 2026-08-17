#include "citizen_system.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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

float h01(uint32_t a, uint32_t b) {
    uint32_t h = 2166136261u;
    h = (h ^ a) * 16777619u;
    h = (h ^ b) * 16777619u;
    return float(h) / 4294967296.0f;
}

constexpr float kClockScale = 60.0f;      // 1 real s = 1 game minute
// EVERYONE simulates every frame; rendering is tiered by distance.
// Full seven-box articulation inside kDetailRadius (capped at
// kMaxDetailed nearest so a packed district can't explode the draw
// count); a single person-box out to kShowRadius (10 km per the design
// brief).  kMinAngular skips far persons whose box would land under
// ~a third of a pixel — at 10 km a 1.7 m person is invisible anyway,
// so this is where the 10 km budget actually stops costing.
constexpr float kDetailRadius = 300.0f;
constexpr size_t kMaxDetailed = 320;
constexpr float kShowRadius = 10000.0f;   // 10 km
constexpr float kMinAngular = 0.0003f;    // height/dist cutoff
constexpr float kGroundClampRadius = 400.0f;
// The clock runs 60x real time, so schedule-accurate commuting needs
// faster-than-life legs: at kWalkTimeScale 6 a 500 m commute costs one
// game hour (about a real minute of visible walking) instead of six.
// Full 60x would be teleport-sprinting; 6x reads as "people getting
// places" while staying watchable up close.
constexpr float kWalkTimeScale = 6.0f;
constexpr size_t kFarClampPerFrame = 256; // staggered height refresh

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
    if (!std::filesystem::exists(city_json_path, ec) ||
        !std::filesystem::exists(world_json_path, ec)) {
        // LOUD on purpose: a silent false here cost a debugging round —
        // "why no npc has been rendered" with no line to grep for.
        printf("[citizen] city data not found (%s / %s) — no citizens "
               "this map.  Run the place stage to generate them.\n",
               city_json_path.c_str(), world_json_path.c_str());
        return false;
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
        loaded_ = !persons_.empty();
        printf("[citizen] loaded %zu persons, %zu buildings, %zu "
               "houses from %s\n",
               persons_.size(), buildings_.size(), houses_.size(),
               std::filesystem::path(city_json_path)
                   .filename().string().c_str());
    } catch (const std::exception& e) {
        printf("[citizen] city load failed: %s\n", e.what());
        loaded_ = false;
    }
    return loaded_;
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
    return {b.entrance.x + jx, b.base_y, b.entrance.y + jz};
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
    // kClockScale game-seconds per real second -> minutes here.
    clock_min_ = std::fmod(clock_min_ + delta_t * (kClockScale / 60.0f),
                           7.0f * 1440.0f);
    const float tod = std::fmod(clock_min_, 1440.0f);

    if (sim_.size() != persons_.size()) sim_.resize(persons_.size());

    // ── EVERYONE ticks, every frame ──────────────────────────────────
    // 8k persons x a compare, a lerp and a normalize is well under a
    // millisecond; what does NOT scale is the terrain ground query, so
    // that runs per frame only inside kGroundClampRadius and as a slow
    // round-robin ring (kFarClampPerFrame/frame) beyond it — a far
    // commuter's height refreshes every couple of seconds, which at
    // 300 m+ is beneath notice.
    const size_t n = persons_.size();
    for (size_t i = 0; i < n; ++i) {
        const Person& p = persons_[i];
        SimState& a = sim_[i];
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
            const float v = p.speed * kWalkTimeScale;
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
            if (persons_[i].height / dist < kMinAngular) continue;
            emitPerson(int(i), sim_[i], persons_[i], false);
        }
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
