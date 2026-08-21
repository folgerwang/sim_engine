#pragma once

// CitizenSystem — box-people living the city simulation.
//
// Reads the two files the placement pipeline writes:
//   <map>_pcg_city.json         persons: home, duty, age, body sheet,
//                               weekday/weekend schedules (city_sim.py)
//   <map>_pcg_world.json        house instance transforms (columnar)
//
// and turns the persons nearest the camera into ARTICULATED BOX FIGURES
// — pelvis, torso, head, two arms, two legs, seven unit cubes per
// person, posed procedurally per activity: walking (limb swing), sitting
// (chairs at work/school/restaurants), cooking at home, browsing in the
// mall, care rounds for nurses and doctors, a door-open gesture on
// every house entry.  A game clock (accelerated real time, weekday /
// weekend aware) advances each person along their schedule; they WALK
// between home, school, daycare and their workplace, terrain-clamped
// through the same ground query the player's foot IK uses.
//
// Rendering is one pipeline + one shared unit-cube mesh + one push
// constant (transform + role colour) per part, drawn into the forward
// colour/depth buffers in a LOAD-op dynamic-rendering pass right after
// the terrain tiles — depth-tested against the world, tonemapped with
// the shared scene curve.  Citizens are DELIBERATELY absent from the
// shadow/RT paths: they are gameplay markers, and thousands of
// 7-box figures in a TLAS rebuilt per frame is exactly the cost this
// engine spent the week avoiding.

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "renderer/renderer.h"

namespace engine {
namespace game_object {

class CitizenSystem {
public:
    using GroundQueryFn = std::function<
        bool(float x, float z, float y_hint,
             float& out_y, glm::vec3& out_nrm)>;

    // Pipeline + shared unit-cube mesh.  Call once after the device is
    // up, with the same descriptor-set layouts / formats ShapeBase gets
    // (set 0 PBR-global unused, set 1 the camera SSBO citizen.vert
    // reads).
    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::PipelineRenderbufferFormats& frame_buffer_format);
    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    // Parse the city + world manifests.  Safe to call again on map
    // change (replaces everything).  Returns false when either file is
    // missing — the system then stays inert.
    bool loadCity(const std::string& city_json_path,
                  const std::string& world_json_path,
                  const std::string& indoor_json_path = std::string());
    bool loaded() const { return loaded_; }

    // (Re)place EVERY citizen at the anchor their schedule puts them at
    // for the current game clock — the Play button's spawn.  Called on
    // the edit->play edge: the town is populated the instant play
    // starts, at whatever hour the clock says (asleep at home at 03:00,
    // at work at 10:00), instead of drifting in from wherever the last
    // session left them.  Cheap: it just clears the per-person sim
    // state, and the next update() places everyone from their schedule.
    void placeAll();

    // Drive the citizen day from the WORLD clock (the menu's
    // time-of-day, which also drives the sun and the clock face) so the
    // town and the sky can never disagree — people asleep under a noon
    // sun is the failure this prevents.  Hours in [0, 24); the weekday
    // advances on each midnight wrap.  Once called, update() stops
    // running its own clock.
    void setTimeOfDayHours(float hours);

    // Advance the game clock and every active citizen; refresh which
    // persons are active around the camera.
    void update(float delta_t, const glm::vec3& camera_pos,
                const GroundQueryFn& ground);

    // Record the citizen draw into the forward colour/depth buffers
    // (LOAD-op dynamic rendering, own pass — call after the terrain
    // tiles so citizens depth-test against the ground they stand on).
    void draw(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
              const renderer::DescriptorSetList& desc_sets,
              const std::shared_ptr<renderer::ImageView>& color_view,
              const std::shared_ptr<renderer::ImageView>& depth_view,
              const glm::uvec2& buffer_size);

    void destroy(const std::shared_ptr<renderer::Device>& device);

    // Game-clock introspection (minutes since Monday 00:00, day 0-6).
    float clockMinutes() const { return clock_min_; }
    int   dayOfWeek() const { return int(clock_min_ / 1440.0f) % 7; }
    bool  isWeekend() const { return dayOfWeek() >= 5; }
    // The ENTIRE city population is loaded and simulated (near persons
    // at frame rate, the rest on a snap-to-schedule ring); rendering is
    // what stays distance-tiered.
    size_t populationLoaded() const { return persons_.size(); }
    size_t activeCount() const { return persons_.size(); }

private:
    struct Step {
        float minutes = 0.0f;          // time of day, minutes
        int   activity = 0;            // Activity enum
        int   place = -1;              // building index, -1 = home,
                                       // -2 = outdoors
    };
    struct Person {
        int   house = 0;
        int   slot = 0;                // index among this destination's
                                       // residents — their standing spot
        int   shop_b = -1;             // errand shop, when it differs
        int   shop_slot = 0;           // and their standing spot THERE:
                                       // slot is an index at the PRIMARY
                                       // destination and means nothing
                                       // at another building
        int   hslot = 0;               // index within their household,
        int   hcount = 1;              // and how many share the house —
                                       // packs a family apart at home
                                       // the same way slot does at work
        int   duty = 0;                // Duty enum
        float height = 1.7f;
        float bulk = 1.0f;             // width factor from body status
        float speed = 1.35f;           // walk m/s
        bool  works_weekend = false;
        std::vector<Step> weekday;
        std::vector<Step> weekend;
    };
    struct Building {
        std::string type;
        // Which HOUSE this building was promoted from, or -1 for a
        // city-json civic building that is not a house at all.  The
        // synthesized workplaces, shops and schools ARE houses (see
        // synthesizeResidents), which is the whole reason a worker at
        // a desk or a child at a school can be seated on a real chair:
        // the furniture the placement stage put in that house is the
        // furniture of that workplace.
        int house = -1;
        glm::vec2 entrance{0.0f};
        glm::vec2 centre{0.0f};
        float yaw = 0.0f;
        float base_y = 0.0f;
        // Full width (metres) of the arrival scatter placePos spreads
        // people over.  6 m is the historical fixed jitter and stays
        // the default for city-json buildings; synthesized workplaces
        // widen it with their headcount so a cell's whole workforce
        // does not stack inside one doorway.
        float spread = 6.0f;
        // How many residents report here.  With the slot index below it
        // turns the arrival scatter from a random jitter box (where six
        // people routinely land on top of one another) into an even
        // packing.
        int   headcount = 1;
    };
    struct SimState {                   // per-person, ALWAYS ticking
        glm::vec3 pos{0.0f};
        float yaw = 0.0f;
        float phase = 0.0f;            // walk cycle
        float gesture_t = 0.0f;        // door-open timer
        int   cur_step = -1;
        // WHICH ROOM THE INDOOR ROUTE IS IN, -1 = not routing.
        // The route room ADVANCES ONLY when the walker reaches the aim
        // point past a doorway; it is never re-derived from the
        // position each tick.  Rooms are rectangles that meet at a
        // wall, and re-deriving means the walker flips between the two
        // rooms either side of that wall as it crosses — each flip
        // pointing it back where it came from.  Measured over 480
        // archetypes: re-deriving stalled 1.2% of routes outright and
        // left 3.7% ping-ponging; advancing arrives on all of them.
        int16_t nav_room = -1;
        bool  walking = false;
        bool  inited = false;
    };

    // ── FURNITURE ANCHORS ────────────────────────────────────────────
    // The placement stage puts REAL furniture inside these houses —
    // obj_bed*, obj_cooktop*, obj_chair* out of room_decals.glb — and
    // PcgInstanceRegistry already holds every one of them with the
    // transform it was placed at.  Harvesting them once at load turns
    // "at home" from a rosette of standing points around the house
    // centre into the actual room: the sleeper LIES ON A BED, the cook
    // stands at the stove, a meal happens on a chair.  Without the
    // registry (a map placed before it existed) every lookup misses and
    // the rosette is still there underneath.
    struct Furniture {
        glm::vec3 t{0.0f};
        float yaw = 0.0f;
        float scale = 1.0f;
    };
    // Flat arrays plus one [first, count) slice per house.  A household
    // resolves its own furniture from its house index alone, so Person
    // — of which there are 120k+ — grows by nothing at all.
    std::vector<Furniture>  beds_, stoves_, seats_, sinks_;
    std::vector<glm::ivec2> house_beds_, house_stoves_, house_seats_,
                            house_sinks_;
    void harvestFurniture();

    // Where a person AT HOME stands or lies for one activity, and which
    // way they face there.  kind == 0 (kAnchorNone) means this house
    // has no such furniture: fall back to the household rosette.
    struct Anchor {
        glm::vec3 pos{0.0f};
        float yaw  = 0.0f;
        int   kind = 0;
    };
    // Where a person doing `activity` at step `s` should stand, sit or
    // lie, and which way they face.  Resolves the house whose furniture
    // applies — their own when they are home, the promoted house behind
    // a workplace/school/shop otherwise — so a schoolchild sits on the
    // school's chairs rather than in the air above its floor.
    // kind == 0 (kAnchorNone): no such furniture reachable, and the
    // CALLER MUST NOT POSE AS IF THERE WERE.
    Anchor furnitureAnchor(const Person& p, const Step& s,
                           int activity) const;
    // ONE definition of "what is this person actually doing" — the
    // night rule included.  placePos and emitPerson both go through it,
    // so the spot someone is put on and the pose they are drawn in can
    // never disagree (a sleeper posed asleep at the STANDING anchor is
    // exactly the bug this prevents).
    int resolveActivity(const Step& s) const;

    const std::vector<Step>& scheduleOf(const Person& p) const {
        return (isWeekend() && !p.works_weekend && !p.weekend.empty())
                   ? p.weekend : p.weekday;
    }
    // Fallback population: a HOUSEHOLD of 3-5 per house, synthesized
    // from the world manifest alone.  Household size is drawn per house
    // and the seats read as a family (earner, second adult, students),
    // so a town's population is house_count x 3 .. house_count x 5.  The city json (city_sim.py) is optional —
    // it only exists for maps that got a civic district, and without it
    // every house used to stand empty (loadCity returned false and the
    // whole system went inert, which is the "where are the people?"
    // report).  It also promotes a few houses per neighbourhood to
    // workplaces, shops and schools, so a synthesized day is a real
    // commute; everything else — bodies, walking, ground clamping,
    // rendering — is the same code path.
    void synthesizeResidents();
    glm::vec3 placePos(const Person& p, const Step& s, int pid) const;

    // ── Walking around buildings ─────────────────────────────────────
    // Citizens walked the straight line between anchors, which took
    // them clean through their neighbours' houses.  There is no nav
    // mesh here (and ~230k agents could not afford one), so this is local
    // steering: houses are treated as discs in a coarse spatial hash,
    // and a walker whose next step would enter one slides along its
    // tangent instead.  `exempt_*` are the buildings they are allowed
    // to be inside — their own home, and wherever they are heading.
    void buildHouseGrid();
    glm::vec2 steerAroundHouses(const glm::vec2& pos, const glm::vec2& dir,
                                int exempt_house, int exempt_dest) const;
    std::unordered_map<uint64_t, std::vector<int>> house_grid_;
    int currentStep(const std::vector<Step>& sched, float tod) const;
    void emitPerson(int pid, const SimState& a, const Person& p,
                    bool detailed);

    // static render objects
    static std::shared_ptr<renderer::PipelineLayout> s_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline>       s_pipeline_;
    static std::shared_ptr<renderer::BufferInfo>     s_cube_pos_;
    static std::shared_ptr<renderer::BufferInfo>     s_cube_nrm_;
    static std::shared_ptr<renderer::BufferInfo>     s_cube_idx_;
    static uint32_t                                  s_cube_index_count_;
    // ── PER-FRAME INSTANCE STREAM ───────────────────────────────────
    // frame_parts_ uploaded once and drawn with ONE instanced call.
    // The device is kept because draw() is handed a command buffer and
    // nothing else, and the buffer has to grow with the crowd.
    static std::shared_ptr<renderer::Device>         s_device_;
    static std::shared_ptr<renderer::BufferInfo>     s_inst_buf_;
    static uint32_t                                  s_inst_capacity_;

    bool loaded_ = false;
    std::vector<glm::vec3> houses_;
    // Pupil SEATS in each house, from the world manifest's
    // instances.houses.school column: 0 for an ordinary dwelling, and
    // for a school building the number of classroom chairs terrain_pcg
    // actually laid out in it.  Empty when the map predates the column,
    // in which case synthesizeResidents falls back to promoting houses
    // the way it always did.
    //
    // NOT house_seats_ — that name is already taken above by the
    // furniture slice table for obj_chair, which is a completely
    // different thing (where the chairs ARE, not how many pupils a
    // building holds).  The two collided and the compiler resolved the
    // uses to whichever it saw first.
    std::vector<int> house_school_seats_;

    // ── INDOOR ROUTE GRAPHS ─────────────────────────────────────────
    // <map>_pcg_indoor.json, one graph per ARCHETYPE in house-local
    // metres (terrain_pcg.build_indoor_graph).  A house names its
    // archetype in the world manifest, so a graph is shared by every
    // instance of it and a house costs one index plus the transform it
    // already had.
    //
    // The routing itself is table-driven rather than searched: rooms
    // per house are few (<= kNavMaxRooms), so a next-hop matrix built
    // once at load answers "standing in room A, heading for room B,
    // which doorway next" in one lookup — no per-agent path storage
    // and no per-frame search for a quarter of a million people.
    struct NavRoom {
        glm::vec2 c{0.0f};
        float hw = 0.0f, hd = 0.0f, yaw = 0.0f;
        int storey = 0;
    };
    struct NavDoor {
        glm::vec2 p{0.0f};
        int storey = 0;
        int a = -1, b = -1;          // rooms joined; -1 == outdoors
    };
    struct IndoorGraph {
        std::vector<NavRoom> rooms;
        std::vector<NavDoor> doors;
        std::vector<int>     street;   // door indices reaching outside
        // next_[from * rooms + to] = door index to head for, or -1
        std::vector<int16_t> next_;
        // dist_[from * rooms + to] = doorways still to cross, or -1.
        // Same BFS, kept because ROOMS OVERLAP: wings share floor, so
        // a point can be inside two rooms at once and "which room am I
        // in" has no geometric answer.  The one nearer the destination
        // in the graph is the useful answer, and this is the table
        // that says which that is.
        std::vector<int16_t> dist_;
    };
    std::vector<IndoorGraph> graphs_;
    std::vector<int>   house_graph_;   // -1 = no graph for this house
    std::vector<float> house_yaw_;
    std::vector<glm::vec2> house_scale_;   // x/z only

    bool loadIndoor(const std::string& path,
                    const std::vector<std::string>& house_node,
                    const std::vector<int>& house_node_idx);
    // House-local <-> world for house `hi` (yaw + per-axis scale).
    glm::vec2 worldToLocal(int hi, const glm::vec2& w) const;
    glm::vec2 localToWorld(int hi, const glm::vec2& l) const;
    // Which room of house `hi` contains a house-local point, or -1.
    // `toward` breaks the tie when overlapping rooms both contain it:
    // the one closer to room `toward` in the doorway graph wins, then
    // the one the point is deepest inside.  -1 = no destination yet.
    int roomAt(int hi, const glm::vec2& local, int storey,
               int toward = -1) const;
    // The house whose interior step `s` refers to, or -1 for outdoors.
    int anchorHouse(const Person& p, const Step& s) const;
    // Next waypoint toward `dst_world` for a person at `pos_world`
    // heading into house `hi`; false when no routing is needed.
    // `nav_room` is the walker's own route room, carried between ticks
    // and advanced here (see SimState::nav_room).
    bool indoorWaypoint(int hi, const glm::vec3& pos_world,
                        const glm::vec3& dst_world,
                        int16_t& nav_room,
                        glm::vec3& out_wp) const;
    std::vector<Building>  buildings_;
    std::vector<Person>    persons_;
    std::vector<SimState>  sim_;       // parallel to persons_
    size_t clamp_cursor_ = 0;          // far-person ground refresh ring
    size_t sim_cursor_ = 0;            // far-person schedule ring
    size_t near_clamp_cursor_ = 0;     // budgeted near-ring clamp start
    float  far_thresh_ = 0.0f;         // adaptive far-tier angular cutoff
                                       // (0 = seed from kMinAngular)
    float  dbg_timer_ = 0.0f;          // [citizen] telemetry cadence
    // ── POSE CLOCK: REAL seconds, not game minutes ──────────────────
    // Idle sway, the cooking stir, a child's arm swing — every pose
    // cycle in emitPerson used to read clock_min_, which is GAME
    // minutes.  Driven from the world clock at its default 5x speed
    // that advances ~0.08 per REAL second, so a 1.5 rad/game-minute
    // sway became 0.12 rad/s: mathematically animated, visually a
    // statue.  (Near clock_min_'s week-long range, ~10080, float
    // spacing is ~0.001, so what motion survived was also quantized.)
    // Poses are BODY motion — they belong to real time and must look
    // the same whatever rate the world clock runs at.  Wrapped at
    // kAnimWrap = 8*pi, which is a whole number of cycles for every
    // half-integer frequency the poses use, so the wrap is seamless.
    float  anim_t_ = 0.0f;             // real seconds, wrapped at 8*pi
    glm::vec2 district_centre_{0.0f};  // civic-building centroid (log aid)

    float clock_min_ = 8.0f * 60.0f + 2.0f * 1440.0f;   // Wed 08:00
    bool  clock_external_ = false;     // driven by setTimeOfDayHours
    // Measured clock rate (game-minutes per real second) and the walk
    // scale derived from it — see the kWalkTimeScale note in the .cpp:
    // legs have to keep up with whatever speed the world clock runs at,
    // without turning into a sprint when it runs slowly.
    float prev_clock_min_ = -1.0f;
    float clock_rate_ = 0.0f;          // 0 = unmeasured -> life speed
    GroundQueryFn ground_;

    struct PartInstance { glm::mat4 xform; glm::vec4 color; };
    std::vector<PartInstance> frame_parts_;
    // Per-frame scratch, kept as members so the render-tier pass does
    // not heap-allocate (and free) a population-sized buffer every
    // frame — with 3-5 residents per house that is a quarter-megabyte
    // malloc per frame, for nothing.
    std::vector<uint8_t> is_detailed_;
    std::vector<std::pair<float, int>> near_ids_;
};

}  // namespace game_object
}  // namespace engine
