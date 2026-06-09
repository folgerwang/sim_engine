#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// asset_streamer.h — streaming contract + component (renderer-agnostic).
//
// The streaming system decides WHEN an entity's asset should be resident
// (distance-based), but it does not know HOW to load one — that requires the
// renderer (DrawableObject::createAsync + MeshLoadTaskManager + GPU upload).
// We bridge the two with the IAssetStreamer interface: the core depends only
// on this abstraction, and the engine supplies a concrete DrawableAssetStreamer
// (ecs/engine/drawable_asset_streamer.h). A trivial mock implements it in the
// unit tests, so the load/unload state machine is verifiable without Vulkan.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>

#include <glm/glm.hpp>

#include "ecs/entity.h"

namespace engine {
namespace ecs {

// Opaque handle the streamer hands back from beginLoad(); the streaming system
// stores it on the component and passes it to poll()/unload().
using StreamHandle = uint64_t;
inline constexpr StreamHandle kInvalidStream = 0;

// Residency state of one streamed asset.
enum class AssetState : uint8_t {
    kUnloaded = 0,  // not in memory; outside load radius
    kLoading  = 1,  // load kicked off, GPU upload in flight
    kResident = 2,  // ready, drawable attached
};

// The renderer-side loader the streaming system drives. Implementations must be
// safe to call from the main thread once per frame.
class IAssetStreamer {
public:
    virtual ~IAssetStreamer() = default;

    // Begin async load of `asset_path` for `entity`, placed at `world`.
    // Returns a handle (kInvalidStream on immediate failure). The concrete
    // streamer is responsible for attaching the renderable to the entity once
    // the load finishes — the core never touches Vulkan components.
    virtual StreamHandle beginLoad(Entity entity,
                                   const std::string& asset_path,
                                   const glm::mat4& world) = 0;

    // Poll an in-flight (or resident) handle. Returns the current state.
    // kLoading -> kResident transition happens here when the GPU upload and
    // main-thread finalize (phase 3) have completed.
    virtual AssetState poll(StreamHandle handle) = 0;

    // Push an updated world transform to a resident asset (e.g. parent moved).
    virtual void setWorld(StreamHandle handle, const glm::mat4& world) = 0;

    // Release the asset's resources (detaches the renderable and schedules a
    // GC-safe GPU free via the deferred deleter). The handle is invalid after.
    virtual void unload(StreamHandle handle) = 0;
};

// Per-entity streaming policy + live state. Attach to any entity whose asset
// should load/unload by camera distance.
struct StreamingComponent {
    std::string  asset_path;
    float        load_radius   = 64.0f;   // enter at <= this distance
    float        unload_radius = 96.0f;   // leave at >  this distance (hysteresis)
    AssetState   state         = AssetState::kUnloaded;
    StreamHandle handle        = kInvalidStream;
};

}  // namespace ecs
}  // namespace engine
