# ECS Layer — Design & Migration Plan

An entity-component-system layer for game objects, built on **EnTT** (vendored,
fetched at CMake configure time). It provides the abstract lifetime layer you
asked for — **GC**, **streaming**, and cache-friendly **efficiency** — while the
existing renderer (`DrawableObject`, `ObjectSceneView`, the cluster renderer)
keeps working unchanged.

The core is deliberately split from the renderer so it is unit-testable with no
Vulkan dependency. `ecs/tests/ecs_core_tests.cpp` compiles and passes 36 checks
covering GC timing, transform hierarchy, generational invalidation, and the
streaming state machine.

---

## 1. Module layout

```
ecs/
  entity.h                 Entity = entt::entity (generational handle)
  components.h             POD components: LocalTransform, WorldTransform, Parent,
                           LocalBounds, WorldBounds, Name, PersistentId, + tags
                           (Active, Visible, Static, DirtyTransform, PendingDestroy)
  asset_streamer.h         IAssetStreamer contract + StreamingComponent + AssetState
  deferred_deleter.h       GC ring: frame-delayed, GPU-safe resource reclamation
  transform_system.{h,cpp} hierarchy propagation -> WorldTransform / WorldBounds
  streaming_system.{h,cpp} distance-based load/unload state machine
  lifetime_system.{h,cpp}  deferred destroy + generational invalidation
  world.{h,cpp}            facade: owns the registry + GC, orders the systems
  engine/                  Vulkan-bearing bridge (compiled only in the engine):
    render_components.h       Renderable { shared_ptr<DrawableObject> }, CameraRef
    drawable_asset_streamer.* IAssetStreamer over DrawableObject::createAsync +
                              MeshLoadTaskManager + DeferredDeleter
    render_system.*           gather visible drawables for ObjectSceneView
  tests/ecs_core_tests.cpp standalone, renderer-free unit tests
```

**Dependency rule:** everything outside `ecs/engine/` is renderer-free. Vulkan
types appear only in `ecs/engine/`. This is what keeps the core testable and the
build fast.

---

## 2. Why a thin component over DrawableObject first (not a full shred on day 1)

You chose a full data-oriented ECS as the destination. Getting there safely on a
~350 K-LOC engine is a phased migration, not a big-bang rewrite. The layer
shipped here is **phase 1+2**: real EnTT storage, generational lifetime, GC,
streaming, and decoupled transform iteration — with `DrawableObject` wrapped in a
`Renderable` component rather than torn apart. `DrawableObject` is already a
self-contained render package (nodes, meshes, materials, skins, instance
buffer), so wrapping it yields the lifetime/streaming/iteration wins immediately
and at zero risk to the renderer. Phase 3 (below) then peels data off it into
dedicated component pools incrementally, one system at a time, while every frame
still renders.

---

## 3. Lifetime & GC model

Two independent guarantees:

**(a) Generational handles (no dangling references).** `Entity` is EnTT's 32-bit
handle = index + version. `World::destroy(e)` tags the entity `PendingDestroy`;
`World::collectGarbage()` (once per frame) runs a cleanup hook and calls
`registry.destroy(e)`, which recycles the index with a **bumped version**. Any
stale `Entity` copy held elsewhere now fails `World::valid()` — there are no raw
pointers to invalidate. Destruction is deferred so a system iterating entities
can never have an object deleted out from under it mid-pass.

**(b) GPU-safe resource reclamation (`DeferredDeleter`).** The GPU lags the CPU by
up to `frames_in_flight` frames, so freeing a Vulkan buffer the instant an entity
dies risks a use-after-free. `DeferredDeleter` is a ring of `frames_in_flight + 1`
buckets; `schedule(fn)` drops a release closure into the current bucket, and
`beginFrame()` flushes the bucket that aged out exactly `frames_in_flight` frames
ago — by which point the GPU has provably finished those command buffers. The
`DrawableAssetStreamer` schedules `drawable->destroy(device)` through it on
unload. `flushAll()` (after `Device::waitIdle()`) drains everything at shutdown.

*Verified:* `test_deferred_deleter` asserts a closure survives `frames_in_flight`
begin-frames and runs on the next; `test_generational_gc` asserts stale handles
go invalid and recycled slots get a new handle value.

---

## 4. Streaming

`StreamingComponent` carries `asset_path`, `load_radius`, `unload_radius`, and
live `state` + `handle`. `StreamingSystem::update(registry, streamer, focus)`
walks every streamed entity and drives the state machine by distance to `focus`
(the camera):

```
Unloaded --(dist <= load_radius)--> Loading --(streamer ready)--> Resident
   ^                                   |                              |
   |          (dist > unload_radius, cancel)        (dist > unload_radius)
   +-----------------------------------+------------------------------+
```

`load_radius < unload_radius` gives hysteresis so an entity hovering at the
boundary doesn't thrash load/unload. All loading goes through `IAssetStreamer`;
the engine's `DrawableAssetStreamer` implements it with the **existing async
pipeline** — `DrawableObject::createAsync()` (phase 2 on a worker, phase 3 on the
main thread via `MeshLoadTaskManager`) — and attaches a `Renderable` to the
entity when the load finalizes. Unload detaches `Renderable` and schedules the
GPU teardown via the deferred deleter.

*Verified:* `test_streaming` exercises far/near transitions, the 2-poll load
latency, hysteresis in the dead-band, and unload, with a mock streamer.

This composes with your existing terrain tile streaming (`TileObject::
updateAllTiles`) rather than replacing it — terrain stays as-is; ECS streaming
handles discrete placed assets (props, NPCs, imported scene objects).

---

## 5. Transform system

`LocalTransform` (authored TRS) is the source of truth; `WorldTransform` (cached
matrix) is derived. Editing a transform calls `World::markDirty(e)`, which tags
`DirtyTransform`. `TransformSystem::update` builds a parent->children adjacency,
propagates dirtiness down each affected subtree, recomputes world matrices
shallow->deep (memoized so a parent is computed before its children), refreshes
`WorldBounds` from `LocalBounds`, then clears the dirty tags. Clean frames are a
near-free early-out. This mirrors the existing `scene::Object::parent_index`
hierarchy but in entity space.

*Verified:* `test_transform_hierarchy` checks a 3-level chain, subtree
propagation when a root moves, and world-AABB transform.

---

## 6. Per-frame integration (application.cpp)

In `drawFrame()`, around the existing work:

```cpp
world.beginFrame();                       // advance GC ring, flush aged GPU frees
world.updateTransforms();                 // dirty subtrees -> world matrices
world.updateStreaming(camera_position);   // load/unload by distance
world.collectGarbage();                   // destroy PendingDestroy entities

// hand ECS drawables to the existing renderer:
auto drawables = ecs::RenderSystem::gather(world.registry());
object_scene_view_->setDrawables(drawables);   // or append to its list
// ... existing drawScene() proceeds unchanged ...
```

One-time setup after the device/pool/formats exist:

```cpp
ecs::DrawableAssetStreamer::PipelineContext ctx{ device_, drawable_descriptor_pool_,
    &renderbuffer_formats_, graphic_pipeline_info_, texture_sampler_, thin_film_lut_ };
streamer_ = std::make_unique<ecs::DrawableAssetStreamer>(
    world.registry(), *mesh_load_task_manager_, world.deleter(), ctx);
world.setStreamer(streamer_.get());
world.setCleanupHook([&](entt::registry& r, ecs::Entity e){
    // e.g. if (auto* s = r.try_get<ecs::StreamingComponent>(e); s && s->handle)
    //          streamer_->unload(s->handle);
});
```

`World` should be a member of `RealWorldApplication` and `frames_in_flight`
should match the renderer's value (e.g. 2).

---

## 7. Phased migration to full data-oriented ECS

- **Phase 1 — Foundation (done).** EnTT integrated; core components; GC,
  streaming, transform systems; tested.
- **Phase 2 — Adopt at the edges (done in bridge).** Imported scene objects
  (`scene_.objects` / `imported_objects_`) become entities with
  `LocalTransform + Parent + StreamingComponent + Renderable`. `scene_io`
  round-trips via `PersistentId`. Wire `World` into `drawFrame`.
- **Phase 3 — Shred DrawableObject into pools, one system at a time.** Move
  independently-iterated data out of `DrawableObject` into components:
  `MeshComponent` (buffers + primitive list), `MaterialComponent`,
  `SkinComponent` / `AnimationComponent`, `BoundsComponent`. Add an
  `AnimationSystem` that updates only skinned entities, a `CullingSystem` over
  `WorldBounds`. Each move is behind a flag and validated against the current
  frame output before deleting the old path.
- **Phase 4 — Data-oriented hot loops.** Replace the transform adjacency with
  EnTT relationship storage + a depth-sorted pool; group render data so the draw
  gather is a linear sweep feeding the cluster renderer's indirect path. Convert
  PlayerController to write into an entity's components instead of poking the
  drawable directly.

Each phase keeps the engine shippable; nothing requires a stop-the-world rewrite.

---

## 8. Scaling notes

- **Transform system** currently allocates a per-pass `unordered_map` adjacency.
  Fine for thousands of entities; for tens of thousands, switch to EnTT's
  `reactive`/relationship storage and a depth-sorted dense pool so propagation is
  a contiguous sweep with no hashing. Public API is unchanged.
- **Streaming** uses linear distance scans. For very large worlds, feed it from a
  spatial grid / BVH (you already have `scene_grid` and `bvh`) so only nearby
  entities are considered.
- **GC ring** is O(closures freed); negligible.

---

## 9. Build integration

- EnTT `${ENTT_VER}` single header is downloaded by CMake into
  `third_parties/entt/entt/entt.hpp` (re-fetched if missing or truncated) and is
  git-ignored — same convention as miniaudio/sherpa. `${TP_DIR}/entt` is on the
  include path, so code uses `#include <entt/entt.hpp>`.
- The six `ecs/**/*.cpp` files are compiled into the existing `engine` static lib.
- C++20, no new link-time dependencies (EnTT is header-only).

## 10. Running the core tests

```
g++ -std=c++20 -I<sim_engine> -I<entt-include> -I<glm-include> \
    ecs/tests/ecs_core_tests.cpp ecs/transform_system.cpp ecs/streaming_system.cpp \
    ecs/lifetime_system.cpp ecs/world.cpp -o ecs_tests && ./ecs_tests
```

The core depends only on EnTT + GLM, so this runs anywhere — no Vulkan, no engine.

## 11. EnTT API surface used (for review)

`registry::{create, destroy, valid, emplace, emplace_or_replace, get, try_get,
all_of, remove, view}`, `view::each`, `entt::{null, to_integral}`. All stable
across EnTT 3.x.

---

## 12. CullingSystem (added) + the render-wiring caveat

`culling_system.{h,cpp}` is a pure, renderer-agnostic system: `FrustumPlanes::
fromViewProj(vp)` (Gribb–Hartmann) + `CullingSystem::update(reg, frustum)` tests
each entity's `WorldBounds` AABB against the 6 planes and toggles a `Culled`
tag. Unit-tested (inside / outside / straddling / re-enter). `LocalBounds` is
populated in the app from `DrawableObject::getModelBboxMin/Max()` (or the
skinned joint AABB), so `WorldBounds` is real and the system has data.

**Why it is NOT yet wired into rendering (do this with a build/test loop):**
Acting on `Culled` in this engine is not safe to do blind, because both obvious
hooks conflict with existing systems:
- Toggling **scene-view membership** (add/remove from `ObjectSceneView`) churns
  the list every time an object crosses the frustum edge, and `ObjectSceneView`
  is add/remove-only (O(n) erase).
- Toggling the drawable's **`visible_` flag** fights the cluster renderer, which
  already sets `visible_ = false` on placed objects when they are cluster-
  resident (to avoid forward/cluster double-draw). A per-frame cull override
  would re-enable forward drawing and double-draw them.

The engine also already frustum-culls per-mesh at draw time, so object-level ECS
culling is a *coarser early-out*, not a correctness fix. The clean way to wire
it is to make rendering fully ECS-list-driven (gather rebuilds the per-frame
draw list, with `Culled` filtered out) — which is the same refactor as the
mesh/material/skin shred below. Until then `Culled` is computed and available
for tools / spatial queries.

## 13. Remaining: DrawableObject shred (compile-in-the-loop work)

`DrawableObject` is a ~1500-line monolith holding nodes, meshes, materials,
skins, animations and GPU buffers, and the renderer (forward + cluster + shadow
+ skinning + instance buffers) reads its internals directly. Splitting it into
`MeshComponent` / `MaterialComponent` / `SkinComponent` / `AnimationComponent`
pools rewires the render core, so it must be done incrementally **with a
build + visual test after each move** — not in one blind pass. Suggested order
(least-coupled first):
1. `AnimationComponent` + `AnimationSystem` — animation playback is already
   fairly self-contained; move channel evaluation out of `DrawableObject::update`.
2. `BoundsComponent` is already done (LocalBounds/WorldBounds) — switch the
   draw-time per-mesh cull to consult `Culled` and retire the static
   `setFrustumCullPlanes` path.
3. `MaterialComponent` — pull material/texture refs into a pool shared by
   instances (dedup).
4. `MeshComponent` — the big one: GPU buffers + primitive list. Do last, behind
   a flag, validated against the cluster path frame-by-frame.

---

## 14. Status & wiring guide (current)

**Core systems — built, unit-tested (71 checks), renderer-free:**
transform, streaming, lifetime/GC, deferred-deleter, culling, animation,
material dedup cache.

Build/run the suite (sandbox stand-in or real EnTT):
```
g++ -std=c++20 -I<sim_engine> -I<entt-include> -I<glm-include> \
    ecs/tests/ecs_core_tests.cpp \
    ecs/transform_system.cpp ecs/streaming_system.cpp ecs/lifetime_system.cpp \
    ecs/culling_system.cpp ecs/animation_system.cpp ecs/material_cache.cpp \
    ecs/world.cpp -o ecs_tests && ./ecs_tests
```

**Wired into the engine:** lifetime/GC, streaming, transform/hierarchy, render
membership (reconcile), LocalBounds population.

**Built but NOT wired (need a build-and-see loop):** animation, culling,
material dedup. Each rewires a piece of the render core, so do them one at a
time with a visual check. Concrete hooks:

### Animation
1. Build an `AnimationClip` cache at import: the channels are already parsed in
   `DrawableObject`; copy them into `ecs::AnimationClip` keyed by asset+clip name.
2. Give animated entities an `AnimationPlayer{clip,...}`.
3. In `tickEcs`, call `AnimationSystem::update(reg, dt)` (after transforms,
   before the imported push).
4. Add a small adapter: for each entity with an `AnimPose`, write
   `pose.nodes[i]` (where `has_t/r/s`) onto the matching `DrawableObject` node
   TRS, then let the existing joint-matrix/skinning path run. Retire the channel
   evaluation currently inside `DrawableObject::update`.

### Culling
1. `auto frustum = FrustumPlanes::fromViewProj(main_cam.proj * main_cam.view);`
2. `CullingSystem::update(reg, frustum);` each frame after `updateTransforms`.
3. Consume `Culled`: the clean path is to filter it inside
   `reconcileImportedSceneViewMembership` (gather only un-culled), BUT first make
   reconcile rebuild-based or it will churn add/remove at the frustum edge.
   Do NOT toggle `DrawableObject::visible_` for culling — it fights the cluster
   renderer's forward-hide. Easiest safe win: use `Culled` only as an early-out
   in the forward draw loop, leaving membership alone.

### Material dedup
1. At import, fill a `MaterialDesc` per material and `intern()` it into a
   `MaterialCache` member; store the returned `MaterialId` on the sub-mesh.
2. Keep a parallel `MaterialId -> GPU material (textures + descriptor set)` table
   built once per unique id — this is where the upload/VRAM savings land.
3. On object destroy, `release()` each `MaterialId`; free the GPU material when
   `refCount` hits 0 (via the DeferredDeleter).

**Remaining (largest, deferred):** `MeshComponent` — moving the GPU vertex/index
buffers + primitive list out of `DrawableObject`. This is the deepest render-core
change (cluster + forward + shadow + skinning all read those buffers) and should
be done last, behind a flag, validated frame-by-frame against the cluster path.
