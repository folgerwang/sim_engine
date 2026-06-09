#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// drawable_asset_streamer.h — concrete IAssetStreamer over the existing
// async mesh loader (engine side only).
//
// Implements the core's streaming contract using the engine's production
// loading path: DrawableObject::createAsync() + MeshLoadTaskManager (phase 2 on
// a worker, phase 3 on the main thread). On unload it detaches the Renderable
// component and schedules the drawable's GPU teardown through the World's
// DeferredDeleter, so the buffers/images survive until the GPU is provably done
// with them (frames-in-flight delay).
//
// One instance is owned by the application and registered via
// World::setStreamer(). It holds the render-pipeline parameters createAsync()
// needs (device, descriptor pool, formats, sampler, LUT) captured once at
// init, mirroring how application.cpp already calls createAsync today.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <glm/glm.hpp>

#include "ecs/asset_streamer.h"
#include "ecs/deferred_deleter.h"
#include "game_object/drawable_object.h"
#include "game_object/mesh_load_task_manager.h"
#include "renderer/renderer.h"

namespace engine {
namespace ecs {

class DrawableAssetStreamer : public IAssetStreamer {
public:
    // Parameters needed to build a DrawableObject, captured once. These are
    // exactly the arguments application.cpp passes to createAsync().
    struct PipelineContext {
        std::shared_ptr<renderer::Device>         device;
        std::shared_ptr<renderer::DescriptorPool> descriptor_pool;
        const renderer::PipelineRenderbufferFormats* renderbuffer_formats = nullptr;
        renderer::GraphicPipelineInfo             graphic_pipeline_info;
        std::shared_ptr<renderer::Sampler>        texture_sampler;
        renderer::TextureInfo                     thin_film_lut_tex;
    };

    DrawableAssetStreamer(entt::registry& reg,
                          game_object::MeshLoadTaskManager& task_manager,
                          DeferredDeleter& deleter,
                          PipelineContext ctx)
        : reg_(reg),
          task_manager_(task_manager),
          deleter_(deleter),
          ctx_(std::move(ctx)) {}

    StreamHandle beginLoad(Entity entity,
                           const std::string& asset_path,
                           const glm::mat4& world) override;
    AssetState   poll(StreamHandle handle) override;
    void         setWorld(StreamHandle handle, const glm::mat4& world) override;
    void         unload(StreamHandle handle) override;

    // Optional render-list hooks. The bridge stays engine-generic (no
    // ObjectSceneView dependency); the application wires these to add/remove
    // the drawable from its scene view(s) when an asset becomes resident or is
    // unloaded. on_resident fires once, right after the Renderable is attached;
    // on_unload fires before the GPU teardown is scheduled.
    using RenderHook =
        std::function<void(const std::shared_ptr<game_object::DrawableObject>&)>;
    void setRenderHooks(RenderHook on_resident, RenderHook on_unload) {
        on_resident_ = std::move(on_resident);
        on_unload_   = std::move(on_unload);
    }

private:
    struct Slot {
        Entity                                       entity = kNull;
        std::shared_ptr<game_object::DrawableObject> drawable;
        glm::mat4                                    world{1.0f};
        bool                                         attached = false;  // Renderable added
    };

    entt::registry&                          reg_;
    game_object::MeshLoadTaskManager&        task_manager_;
    DeferredDeleter&                         deleter_;
    PipelineContext                          ctx_;
    std::unordered_map<StreamHandle, Slot>   slots_;
    StreamHandle                             next_handle_ = 1;  // 0 = invalid
    RenderHook                               on_resident_;
    RenderHook                               on_unload_;
};

}  // namespace ecs
}  // namespace engine
