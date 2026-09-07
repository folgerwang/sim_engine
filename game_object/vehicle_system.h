#pragma once
// ── VEHICLES: cars the citizens drive ───────────────────────────────
//
// The road network the mesh stage solved (<map>_roads.json: splines
// with per-point half-width and graded height) becomes a GRAPH here --
// nodes where splines meet, edges along them -- and vehicles drive it
// on the right-hand lane: a route is found with A*, a car follows the
// one ahead of it, slows for corners and junctions, yields at a busy
// node, and parks at the curb nearest its destination.
//
// Two kinds of vehicle share the system.  OWNED cars belong to citizens
// (CitizenSystem hands a person a car at their first placement, parked
// at the curb by wherever they are; when that person's schedule sends
// them somewhere far and the car is at hand they board, the car drives
// there, they step out and walk the last stretch).  AMBIENT vehicles
// keep the roads alive around the camera whoever is watching: sedans
// and SUVs, pickups, a semi on the wide roads, the school bus in its
// hours, a police cruiser, an ambulance and a fire engine with their
// light bars going now and then.
//
// Rendering is one instanced pipeline in the citizens' style (the
// citizen vertex shader is reused: unit-space position + a paint
// record per instance): nine HULL meshes -- one per vehicle type, a
// side profile extruded across the width with a little tumblehome
// above the belt line -- plus a wheel and a light-bar mesh, and
// vehicle.frag paints glass, pillars, lights, grille, arches, seams
// and livery onto them from the fragment's position on the hull.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "renderer/renderer.h"

namespace engine {
namespace game_object {

class VehicleSystem {
public:
    using GroundQueryFn = std::function<
        bool(float x, float z, float y_hint,
             float& out_y, glm::vec3& out_nrm)>;

    enum Type {
        kSedan = 0, kSuv, kCyber, kPickup, kSemi, kPolice, kAmbulance,
        kSchoolBus, kFireEngine, kTypeCount
    };

    // Pipeline + the shared meshes.  Same contract as
    // CitizenSystem::initStaticMembers (set 0 PBR-global unused, set 1
    // the camera SSBO the vertex shader reads).
    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::PipelineRenderbufferFormats& frame_buffer_format);
    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    // The road network.  Returns false (and stays inert) when the file
    // is missing or holds no splines.
    bool loadRoads(const std::string& roads_json);
    bool loaded() const { return !edges_.empty(); }
    // Forget every vehicle (the owners are being re-placed).
    void clearVehicles();

    // ── owned cars (CitizenSystem) ───────────────────────────────────
    // A household car parked at the curb nearest `near_pos`; -1 when
    // no road runs within reach.
    int  spawnCar(uint32_t seed, const glm::vec3& near_pos);
    // Drive `car` to the curb nearest `dest`.  False when no route.
    bool dispatch(int car, const glm::vec3& dest);
    bool parked(int car) const;
    glm::vec3 position(int car) const;
    float yaw(int car) const;
    // Where the driver stands after getting out: beside the car, curb
    // side.
    glm::vec3 stepOut(int car) const;
    // Move a parked car to the curb nearest `pos` (its owner was
    // snapped elsewhere by the far ring).
    void recall(int car, const glm::vec3& pos);

    // ── who is in the cars ───────────────────────────────────────────
    // Where a seated figure's ROOT goes for seat `seat` of `car` (the
    // floor point under the seat: cushion top - 0.45 m), facing the
    // car's heading.  The citizen system draws the owner there.
    glm::vec3 seatPos(int car, int seat) const;
    // The synthetic people in the AMBIENT vehicles within `radius` of
    // `cam`: a driver always, passengers by the car's seed, the school
    // bus its children in school hours.  pos/yaw as seatPos gives them.
    struct Occupant {
        glm::vec3 pos{0.0f};
        float yaw = 0.0f;
        uint32_t seed = 0;
        int seat = 0;          // 0 driver, 1 front passenger, 2+ rear
        int type = 0;
    };
    void occupants(const glm::vec3& cam, float radius,
                   std::vector<Occupant>& out) const;

    // ── the road edge, for the citizens' STROLLERS ───────────────────
    // There is no pavement mesh, so a walker keeps just inside the
    // asphalt edge on the right of travel (the lane centre is 1.4-2.6
    // m out, a car's flank at most 1.25 m past that; the walker sits
    // 0.35-0.7 m in from the edge of a road at least 2.8 m wide).
    struct Stroll {
        int   edge = -1;
        float s = 0.0f;                // arc position on the edge
        float dir = 1.0f;              // +1 towards pts.back
        float side = 0.5f;             // metres inside the road edge
        glm::vec3 pos{0.0f};           // world, y = graded road height
        float yaw = 0.0f;              // atan2(t.x, t.z), as vehicles
    };
    // Start on the road nearest `near` (within `radius`), facing either
    // way.  False when no road runs that close.
    bool strollStartNear(const glm::vec3& near, float radius,
                         uint32_t seed, Stroll& out) const;
    // Walk `ds` metres on; at a junction take a random other road, at
    // a dead end turn back.  False only for a stale record.
    bool strollAdvance(Stroll& st, float ds, uint32_t& rng) const;

    void setTimeOfDayHours(float h) { tod_hours_ = h; }
    void setAmbientCount(int n) { ambient_target_ = n; }

    // speed_scale: the citizens' walk scale (legs keep up with the
    // world clock); vehicles use it too, capped so a car still looks
    // like a car.
    void update(float delta_t, float speed_scale,
                const glm::vec3& camera_pos, const GroundQueryFn& ground);
    void draw(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
              const renderer::DescriptorSetList& desc_sets,
              const std::shared_ptr<renderer::ImageView>& color_view,
              const std::shared_ptr<renderer::ImageView>& depth_view,
              const glm::uvec2& buffer_size);
    void destroy(const std::shared_ptr<renderer::Device>& device);

    size_t vehicleCount() const { return vehicles_.size(); }
    size_t nodeCount() const { return nodes_.size(); }
    size_t edgeCount() const { return edges_.size(); }

private:
    // ── the road graph ───────────────────────────────────────────────
    struct Edge {
        std::vector<glm::vec3> pts;    // world, y = graded road height
        std::vector<float>     s;      // cumulative length at each pt
        std::vector<float>     half;   // half-width at each pt
        int a = -1, b = -1;            // nodes at pts.front / pts.back
        float len = 0.0f;
    };
    struct Node {
        glm::vec3 pos{0.0f};
        std::vector<int> edges;
    };
    struct RoadPt { int edge; int index; };    // a point of the network
    struct Leg { int edge; float s_from; float s_to; };

    // ── a vehicle ────────────────────────────────────────────────────
    struct Vehicle {
        int   type = kSedan;
        glm::vec3 color{0.7f};
        uint32_t seed = 0;
        bool  ambient = false;
        bool  lights = false;          // emergency lights on
        bool  parked = true;
        // Ambient, KERBED: parked on the verge for a few minutes,
        // empty -- the cars a street has standing along it.
        bool  kerbed = false;
        // Ambient, DORMANT: a slot the ring does not need right now,
        // parked 10,000 km off (see update); reused by the next spawn.
        bool  dormant = false;
        int   owner = -1;
        std::vector<Leg> route;
        int   leg = -1;
        float s = 0.0f;                // arc position on the leg's edge
        float speed = 0.0f;
        float steer = 0.0f;
        float wheel = 0.0f;            // wheel spin, radians
        float idle_t = 0.0f;           // ambient: pause at a destination
        glm::vec3 pos{0.0f};
        float yaw = 0.0f;
        float y_ground = 0.0f;         // last exact clamp
        int   claim = -1;              // junction node claimed
    };

    struct PartInstance { glm::mat4 xform; glm::vec4 color;
                          glm::vec4 extra; };
    struct Mesh {
        std::shared_ptr<renderer::BufferInfo> pos, nrm, idx;
        uint32_t count = 0;
    };

    // graph building
    void buildGraph(std::vector<std::vector<glm::vec3>>& splines,
                    std::vector<std::vector<float>>& halves);
    int  nodeAt(const glm::vec3& p);
    bool nearestRoadPt(const glm::vec3& p, float radius, RoadPt& out,
                       float* out_dist = nullptr) const;
    bool routeBetween(const RoadPt& from, const RoadPt& to,
                      std::vector<Leg>& out) const;
    glm::vec3 edgePoint(const Edge& e, float s, glm::vec3* tangent,
                        float* half) const;
    glm::vec3 lanePos(const Edge& e, float s, float dir, float extra,
                      glm::vec3* tangent, float* half) const;
    void parkAt(Vehicle& v, const RoadPt& rp);
    // Parked on the VERGE: kVergeM further out than the curb spot,
    // facing along the road in direction `dir` (its own side of the
    // road for either direction, as a street of parked cars is).
    void parkVerge(Vehicle& v, const RoadPt& rp, float dir);
    void strollPlace(Stroll& st) const;
    void spawnAmbient(const glm::vec3& camera_pos, uint32_t seed);
    // A road point min_m..max_m from `from`.  comp >= 0 restricts it
    // to that island of the network (a car can only drive where its
    // road goes); min_comp_edges skips the tiny islands when spawning.
    bool randomDestination(const glm::vec3& from, float min_m, float max_m,
                           uint32_t seed, RoadPt& out, int comp = -1,
                           int min_comp_edges = 1) const;
    void tick(Vehicle& v, int vi, float dt, float speed_scale,
              const std::unordered_map<int, std::vector<int>>& on_edge);
    void emit(const Vehicle& v);

    std::vector<Edge> edges_;
    std::vector<Node> nodes_;
    // spatial hash of every road point, for curb / spawn queries
    std::unordered_map<uint64_t, std::vector<RoadPt>> pt_grid_;
    std::unordered_map<uint64_t, int> node_grid_;
    std::vector<Vehicle> vehicles_;
    std::vector<int> node_claim_;       // vehicle holding the node, -1
    // Connected components ("islands") of the graph.  The mesh stage's
    // roads are per settlement -- 2171 splines came out as 531
    // islands, the largest 143 edges -- so a destination is only ever
    // picked on the island the car is on.
    std::vector<int> edge_comp_;
    std::vector<std::vector<int>> comp_edges_;
    // The CAP on ambient cars; what actually drives is scaled to the
    // road in the spawn ring (one car per kRoadPerCarM of it, kerbed
    // ones included), re-measured every couple of seconds -- a
    // country lane gets a dozen, a town grid the whole hundred and
    // thirty.
    int   ambient_target_ = 130;
    int   ambient_now_ = 0;
    float ring_timer_ = 0.0f;
    // Metres of road inside `radius` of `p`, from the point hash
    // (points x the mean point spacing, measured at load).
    float roadLengthNear(const glm::vec3& p, float radius) const;
    float pt_spacing_m_ = 1.7f;
    float tod_hours_ = 8.0f;
    float anim_t_ = 0.0f;
    float dbg_timer_ = 0.0f;
    uint32_t rng_ = 0x2545F491u;
    glm::vec3 camera_pos_{0.0f};

    // per-frame instance streams, one per mesh
    std::vector<std::vector<PartInstance>> frame_;

    static std::shared_ptr<renderer::PipelineLayout> s_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline>       s_pipeline_;
    // the glass pass: the same hull meshes, blended, depth-tested but
    // not written, windows only (vehicle.frag kind 3)
    static std::shared_ptr<renderer::Pipeline>       s_glass_pipeline_;
    static std::vector<Mesh>                         s_meshes_;
    static std::shared_ptr<renderer::Device>         s_device_;
    static std::shared_ptr<renderer::BufferInfo>     s_inst_buf_;
    static uint32_t                                  s_inst_capacity_;
};

}  // namespace game_object
}  // namespace engine
