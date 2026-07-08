// ─────────────────────────────────────────────────────────────────────────────
// ecs_core_tests.cpp — standalone unit tests for the renderer-agnostic ECS core.
//
// Builds against the real EnTT on the engine, or the minimal EnTT-API stand-in
// in the sandbox. Exercises: deferred-deleter GC timing, transform hierarchy
// propagation + world bounds, generational handle invalidation, and the
// streaming state machine (with a mock streamer). No Vulkan required.
//
// Build (sandbox):
//   g++ -std=c++20 -I<sim_engine> -I<stub-entt-dir> -I<glm-dir> \
//       ecs/tests/ecs_core_tests.cpp ecs/transform_system.cpp \
//       ecs/streaming_system.cpp ecs/lifetime_system.cpp ecs/world.cpp -o tests
// ─────────────────────────────────────────────────────────────────────────────
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ecs/asset_streamer.h"
#include "ecs/components.h"
#include "ecs/deferred_deleter.h"
#include "ecs/animation_system.h"
#include "ecs/material_cache.h"
#include "ecs/culling_system.h"
#include "ecs/streaming_system.h"
#include "ecs/transform_system.h"
#include "ecs/world.h"

using namespace engine::ecs;

static int g_checks = 0;
#define CHECK(cond)                                                           \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(cond)) {                                                        \
            std::printf("FAIL: %s  (line %d)\n", #cond, __LINE__);            \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static bool approx(float a, float b) { return std::fabs(a - b) < 1e-4f; }
static bool approx(const glm::vec3& a, const glm::vec3& b) {
    return approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z);
}

// ── 1. Deferred deleter (GC ring) timing ─────────────────────────────────────
static void test_deferred_deleter() {
    DeferredDeleter d(/*frames_in_flight=*/2);
    int freed = 0;
    d.schedule([&] { ++freed; });
    CHECK(d.pending() == 1);
    // Must survive frames_in_flight begin-frames; freed on the (F+1)-th.
    CHECK(d.beginFrame() == 0); CHECK(freed == 0);  // frame 1
    CHECK(d.beginFrame() == 0); CHECK(freed == 0);  // frame 2
    CHECK(d.beginFrame() == 1); CHECK(freed == 1);  // frame 3 -> runs
    CHECK(d.pending() == 0);

    // shutdown flushes everything immediately.
    int freed2 = 0;
    d.schedule([&] { ++freed2; });
    d.shutdown();
    CHECK(freed2 == 1);
    std::printf("  [ok] deferred deleter GC timing\n");
}

// ── 2. Transform hierarchy propagation + world bounds ────────────────────────
static void test_transform_hierarchy() {
    World w;
    auto& reg = w.registry();

    LocalTransform root_t; root_t.translation = {10, 0, 0};
    Entity root = w.createAt(root_t);

    LocalTransform child_t; child_t.translation = {0, 5, 0};
    Entity child = w.createAt(child_t, root);

    LocalTransform gc_t; gc_t.translation = {0, 0, 2};
    Entity grandchild = w.createAt(gc_t, child);

    // Local bounds on the grandchild to verify world-bounds transform.
    reg.emplace<LocalBounds>(grandchild, LocalBounds{glm::vec3(0), glm::vec3(1)});

    w.updateTransforms();
    CHECK(approx(reg.get<WorldTransform>(root).position(),       {10, 0, 0}));
    CHECK(approx(reg.get<WorldTransform>(child).position(),      {10, 5, 0}));
    CHECK(approx(reg.get<WorldTransform>(grandchild).position(), {10, 5, 2}));
    CHECK(reg.all_of<WorldBounds>(grandchild));
    CHECK(approx(reg.get<WorldBounds>(grandchild).center, {10, 5, 2}));

    // Move root -> whole subtree must follow after marking root dirty.
    reg.get<LocalTransform>(root).translation = {20, 0, 0};
    w.markDirty(root);
    w.updateTransforms();
    CHECK(approx(reg.get<WorldTransform>(child).position(),      {20, 5, 0}));
    CHECK(approx(reg.get<WorldTransform>(grandchild).position(), {20, 5, 2}));

    // Dirty flags cleared -> no-op pass.
    CHECK(w.updateTransforms() == 0);
    std::printf("  [ok] transform hierarchy + world bounds\n");
}

// ── 3. Generational handle invalidation (GC safety) ──────────────────────────
static void test_generational_gc() {
    World w;
    auto& reg = w.registry();

    int cleaned = 0;
    w.setCleanupHook([&](entt::registry&, Entity) { ++cleaned; });

    Entity e = w.createAt(LocalTransform{});
    CHECK(w.valid(e));

    Entity stale = e;            // a dangling copy someone might hold
    w.destroy(e);                // deferred: still valid until collectGarbage
    CHECK(w.valid(e));
    CHECK(w.collectGarbage() == 1);
    CHECK(cleaned == 1);
    CHECK(!w.valid(e));          // destroyed
    CHECK(!w.valid(stale));      // stale copy is now invalid -> no dangling use

    // Recycled slot gets a bumped version => a different handle value.
    Entity reused = w.createAt(LocalTransform{});
    CHECK(w.valid(reused));
    CHECK(entt::to_integral(reused) != entt::to_integral(stale));
    std::printf("  [ok] generational handle invalidation\n");
}

// ── 4. Streaming state machine (mock streamer) ───────────────────────────────
class MockStreamer : public IAssetStreamer {
public:
    int loads = 0, unloads = 0;
    int ready_after = 2;  // becomes resident after N polls

    StreamHandle beginLoad(Entity, const std::string&, const glm::mat4&) override {
        ++loads;
        const StreamHandle h = next_++;
        polls_[h] = 0;
        return h;
    }
    AssetState poll(StreamHandle h) override {
        return (++polls_[h] >= ready_after) ? AssetState::kResident
                                            : AssetState::kLoading;
    }
    void setWorld(StreamHandle, const glm::mat4&) override { ++set_world_calls; }
    void unload(StreamHandle h) override { ++unloads; polls_.erase(h); }

    int set_world_calls = 0;
private:
    StreamHandle next_ = 1;
    std::unordered_map<StreamHandle, int> polls_;
};

static void test_streaming() {
    World w;
    auto& reg = w.registry();
    MockStreamer streamer;
    w.setStreamer(&streamer);

    Entity e = w.createAt(LocalTransform{});  // at origin
    StreamingComponent sc;
    sc.asset_path = "assets/rock.glb";
    sc.load_radius = 50.0f;
    sc.unload_radius = 80.0f;
    reg.emplace<StreamingComponent>(e, sc);
    w.updateTransforms();

    auto& live = reg.get<StreamingComponent>(e);

    // Far away -> stays unloaded.
    w.updateStreaming(glm::vec3(100, 0, 0));
    CHECK(live.state == AssetState::kUnloaded);
    CHECK(streamer.loads == 0);

    // Enter load radius -> begins loading (mock: ready after 2 polls).
    w.updateStreaming(glm::vec3(10, 0, 0));      // beginLoad
    CHECK(live.state == AssetState::kLoading);
    CHECK(streamer.loads == 1);
    w.updateStreaming(glm::vec3(10, 0, 0));      // poll 1 -> still loading
    CHECK(live.state == AssetState::kLoading);
    w.updateStreaming(glm::vec3(10, 0, 0));      // poll 2 -> resident
    CHECK(live.state == AssetState::kResident);

    // Hysteresis: in the dead-band (between load & unload radius) stay resident.
    w.updateStreaming(glm::vec3(60, 0, 0));
    CHECK(live.state == AssetState::kResident);

    // Beyond unload radius -> unload.
    w.updateStreaming(glm::vec3(200, 0, 0));
    CHECK(live.state == AssetState::kUnloaded);
    CHECK(streamer.unloads == 1);

    // Dead-band while unloaded -> stays unloaded (no thrash).
    w.updateStreaming(glm::vec3(60, 0, 0));
    CHECK(live.state == AssetState::kUnloaded);
    CHECK(streamer.loads == 1);
    std::printf("  [ok] streaming state machine + hysteresis\n");
}

// ── 5. Frustum culling over WorldBounds ──────────────────────────────────────
static void test_culling() {
    World w;
    auto& reg = w.registry();

    auto mk = [&](glm::vec3 pos) {
        LocalTransform lt; lt.translation = pos;
        Entity e = w.createAt(lt);
        reg.emplace<LocalBounds>(e, LocalBounds{glm::vec3(0), glm::vec3(1)});
        return e;
    };
    Entity inside = mk({0, 0, 0});
    Entity outside = mk({1000, 0, 0});
    w.updateTransforms();  // compute WorldBounds

    // Orthographic box frustum covering [-10,10]^3 (right-handed, GL depth).
    glm::mat4 vp = glm::ortho(-10.f, 10.f, -10.f, 10.f, -10.f, 10.f);
    auto frustum = FrustumPlanes::fromViewProj(vp);

    auto stats = CullingSystem::update(reg, frustum);
    CHECK(stats.tested == 2);
    CHECK(reg.all_of<Culled>(outside));
    CHECK(!reg.all_of<Culled>(inside));
    CHECK(stats.culled == 1 && stats.visible == 1);

    // Move the outside entity into view -> Culled cleared next pass.
    reg.get<LocalTransform>(outside).translation = glm::vec3(0, 0, 0);
    w.markDirty(outside);
    w.updateTransforms();
    CullingSystem::update(reg, frustum);
    CHECK(!reg.all_of<Culled>(outside));

    // A straddling box (half in) is NOT culled.
    Entity straddle = mk({10, 0, 0});  // extents 1 -> spans x in [9,11]
    w.updateTransforms();
    CullingSystem::update(reg, frustum);
    CHECK(!reg.all_of<Culled>(straddle));
    std::printf("  [ok] frustum culling over WorldBounds\n");
}

// ── 6. Animation sampling + playback ─────────────────────────────────────────
static void test_animation() {
    AnimationClip clip;
    clip.name = "test";
    clip.duration = 1.0f;
    // node 0: translation 0->(10,0,0) linear over [0,1]
    {
        AnimChannel ch;
        ch.target_node = 0; ch.path = AnimPath::kTranslation;
        ch.interp = AnimInterp::kLinear;
        ch.times = {0.0f, 1.0f};
        ch.vec   = {glm::vec3(0), glm::vec3(10, 0, 0)};
        clip.channels.push_back(ch);
    }
    // node 1: rotation identity->90deg about Y, linear
    {
        AnimChannel ch;
        ch.target_node = 1; ch.path = AnimPath::kRotation;
        ch.interp = AnimInterp::kLinear;
        ch.times = {0.0f, 1.0f};
        ch.quat  = {glm::quat(1,0,0,0),
                    glm::angleAxis(glm::radians(90.0f), glm::vec3(0,1,0))};
        clip.channels.push_back(ch);
    }

    AnimPose pose;
    // midpoint: translation interpolates to (5,0,0)
    AnimationSystem::sample(clip, 0.5f, pose);
    CHECK(pose.nodes.size() == 2);
    CHECK(pose.nodes[0].has_t);
    CHECK(approx(pose.nodes[0].translation, {5, 0, 0}));
    // rotation slerps to ~45deg about Y: rotating +Z gives (sin45,0,cos45)
    {
        glm::vec3 v = pose.nodes[1].rotation * glm::vec3(0, 0, 1);
        CHECK(approx(v, glm::vec3(0.70710678f, 0.0f, 0.70710678f)));
    }

    // clamp before/after range
    AnimationSystem::sample(clip, -1.0f, pose);
    CHECK(approx(pose.nodes[0].translation, {0, 0, 0}));
    AnimationSystem::sample(clip, 5.0f, pose);
    CHECK(approx(pose.nodes[0].translation, {10, 0, 0}));

    // STEP interpolation holds the left keyframe
    clip.channels[0].interp = AnimInterp::kStep;
    AnimationSystem::sample(clip, 0.5f, pose);
    CHECK(approx(pose.nodes[0].translation, {0, 0, 0}));

    // Playback: time advances and loops past duration.
    World w;
    auto& reg = w.registry();
    Entity e = w.create();
    AnimationPlayer pl;
    pl.clip = &clip; pl.time = 0.9f; pl.speed = 1.0f; pl.loop = true;
    reg.emplace<AnimationPlayer>(e, pl);
    CHECK(AnimationSystem::update(reg, 0.2f) == 1);
    CHECK(approx(reg.get<AnimationPlayer>(e).time, 0.1f));  // fmod(1.1,1)
    CHECK(reg.all_of<AnimPose>(e));

    // Non-loop clamps at duration.
    reg.get<AnimationPlayer>(e).loop = false;
    reg.get<AnimationPlayer>(e).time = 0.95f;
    AnimationSystem::update(reg, 1.0f);
    CHECK(approx(reg.get<AnimationPlayer>(e).time, 1.0f));
    std::printf("  [ok] animation sampling + playback\n");
}

// ── 7. Material dedup cache ───────────────────────────────────────────────────
static void test_material_cache() {
    MaterialCache cache;

    MaterialDesc red;
    red.base_color = glm::vec4(1, 0, 0, 1);
    red.base_color_tex = "tex/red.png";

    MaterialDesc red2 = red;                 // identical content
    MaterialDesc blue = red;
    blue.base_color = glm::vec4(0, 0, 1, 1); // different

    MaterialId a = cache.intern(red);
    MaterialId b = cache.intern(red2);       // dedup -> same id
    MaterialId c = cache.intern(blue);       // distinct

    CHECK(a == b);
    CHECK(a != c);
    CHECK(cache.liveCount() == 2);
    CHECK(cache.refCount(a) == 2);           // two interns of red
    CHECK(cache.refCount(c) == 1);
    CHECK(cache.get(a)->base_color == glm::vec4(1, 0, 0, 1));

    // Release one red ref: still live.
    cache.release(a);
    CHECK(cache.refCount(a) == 1);
    CHECK(cache.get(a) != nullptr);
    CHECK(cache.liveCount() == 2);

    // Release the last red ref: freed.
    cache.release(a);
    CHECK(cache.get(a) == nullptr);
    CHECK(cache.refCount(a) == 0);
    CHECK(cache.liveCount() == 1);

    // Re-intern red: reuses the freed slot, fresh refcount.
    MaterialId d = cache.intern(red);
    CHECK(cache.get(d) != nullptr);
    CHECK(cache.refCount(d) == 1);
    CHECK(cache.liveCount() == 2);
    CHECK(d == a);                           // freed id 'a' was recycled

    // blue still intact and independent.
    CHECK(cache.get(c)->base_color == glm::vec4(0, 0, 1, 1));
    CHECK(cache.refCount(c) == 1);
    std::printf("  [ok] material dedup cache\n");
}

// ── 8. MaterialSet entity lifecycle ──────────────────────────────────────────
// Mirrors the engine wiring (application.cpp updateEcsMaterials + the GC
// cleanup hook): entities carry a MaterialSet of interned ids; identical
// materials across entities collapse to one id; the World cleanup hook
// releases refs on entity death so the last user frees the entry.
static void test_material_set_lifecycle() {
    World w;
    auto& reg = w.registry();

    MaterialCache cache;
    w.setCleanupHook([&cache](entt::registry& r, Entity e) {
        if (auto* ms = r.try_get<MaterialSet>(e)) {
            for (auto id : ms->ids) cache.release(id);
            ms->ids.clear();
        }
    });

    MaterialDesc wood;  wood.base_color_tex = "tex/wood.png";
    MaterialDesc metal; metal.metallic = 1.0f; metal.base_color_tex = "tex/m.png";

    // Two "drawable" entities sharing the wood material (e.g. two placements
    // of the same asset), one of which also uses metal.
    Entity e1 = w.create();
    Entity e2 = w.create();
    reg.emplace<MaterialSet>(e1, MaterialSet{{cache.intern(wood)}});
    reg.emplace<MaterialSet>(
        e2, MaterialSet{{cache.intern(wood), cache.intern(metal)}});

    CHECK(cache.liveCount() == 2);                      // wood + metal, deduped
    CHECK(reg.get<MaterialSet>(e1).ids[0] ==
          reg.get<MaterialSet>(e2).ids[0]);             // same interned wood id
    const MaterialId wood_id = reg.get<MaterialSet>(e1).ids[0];
    CHECK(cache.refCount(wood_id) == 2);

    // Kill e2: metal (last user) is freed, wood survives via e1.
    w.destroy(e2);
    w.collectGarbage();
    CHECK(cache.liveCount() == 1);
    CHECK(cache.refCount(wood_id) == 1);

    // Kill e1: everything released.
    w.destroy(e1);
    w.collectGarbage();
    CHECK(cache.liveCount() == 0);
    CHECK(cache.get(wood_id) == nullptr);
    std::printf("  [ok] MaterialSet entity lifecycle\n");
}

int main() {
    std::printf("ECS core tests:\n");
    test_deferred_deleter();
    test_transform_hierarchy();
    test_generational_gc();
    test_streaming();
    test_culling();
    test_animation();
    test_material_cache();
    test_material_set_lifecycle();
    std::printf("ALL PASSED (%d checks)\n", g_checks);
    return 0;
}
