// drawable_asset_streamer.cpp — see drawable_asset_streamer.h.
#include "ecs/engine/drawable_asset_streamer.h"

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>

#include <entt/entt.hpp>

#include "ecs/engine/render_components.h"

namespace engine {
namespace ecs {

namespace {
// Decompose a world matrix into the (translation, rotation, scale) triple that
// DrawableObject::setInstanceRootTransform expects.
void decompose(const glm::mat4& m, glm::vec3& t, glm::quat& r, glm::vec3& s) {
    glm::vec3 skew;
    glm::vec4 persp;
    glm::quat q;
    if (glm::decompose(m, s, q, t, skew, persp)) {
        r = glm::normalize(q);
    } else {
        t = glm::vec3(m[3]);
        r = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        s = glm::vec3(1.0f);
    }
}
}  // namespace

StreamHandle DrawableAssetStreamer::beginLoad(Entity entity,
                                              const std::string& asset_path,
                                              const glm::mat4& world) {
    auto drawable = game_object::DrawableObject::createAsync(
        task_manager_, ctx_.device, ctx_.descriptor_pool,
        ctx_.renderbuffer_formats, ctx_.graphic_pipeline_info,
        ctx_.texture_sampler, ctx_.thin_film_lut_tex, asset_path, world);
    if (!drawable) return kInvalidStream;

    const StreamHandle h = next_handle_++;
    Slot slot;
    slot.entity   = entity;
    slot.drawable = std::move(drawable);
    slot.world    = world;
    slots_.emplace(h, std::move(slot));
    return h;
}

AssetState DrawableAssetStreamer::poll(StreamHandle handle) {
    auto it = slots_.find(handle);
    if (it == slots_.end()) return AssetState::kUnloaded;
    Slot& slot = it->second;

    if (!slot.drawable || !slot.drawable->isReady()) return AssetState::kLoading;

    // Ready: place it at the requested world and attach the Renderable so the
    // render gather picks it up. Only attach once.
    if (!slot.attached) {
        glm::vec3 t, s;
        glm::quat r;
        decompose(slot.world, t, r, s);
        slot.drawable->setInstanceRootTransform(t, r, s);

        if (reg_.valid(slot.entity)) {
            reg_.emplace_or_replace<Renderable>(slot.entity,
                                                Renderable{slot.drawable});
        }
        slot.attached = true;
        if (on_resident_) on_resident_(slot.drawable);
    }
    return AssetState::kResident;
}

void DrawableAssetStreamer::setWorld(StreamHandle handle, const glm::mat4& world) {
    auto it = slots_.find(handle);
    if (it == slots_.end()) return;
    Slot& slot = it->second;
    slot.world = world;
    if (slot.drawable && slot.drawable->isReady()) {
        glm::vec3 t, s;
        glm::quat r;
        decompose(world, t, r, s);
        slot.drawable->setInstanceRootTransform(t, r, s);
    }
}

void DrawableAssetStreamer::unload(StreamHandle handle) {
    auto it = slots_.find(handle);
    if (it == slots_.end()) return;
    Slot slot = std::move(it->second);
    slots_.erase(it);

    // Detach from the entity so the render gather stops drawing it now.
    if (reg_.valid(slot.entity) && reg_.all_of<Renderable>(slot.entity)) {
        reg_.remove<Renderable>(slot.entity);
    }
    if (on_unload_ && slot.drawable) on_unload_(slot.drawable);

    // GC-safe GPU teardown: defer destroy() by frames-in-flight so the GPU is
    // done reading the buffers/images recorded in earlier command buffers.
    auto drawable = slot.drawable;
    auto device   = ctx_.device;
    deleter_.schedule([drawable, device]() mutable {
        if (drawable) drawable->destroy(device);
        drawable.reset();
    });
}

}  // namespace ecs
}  // namespace engine
