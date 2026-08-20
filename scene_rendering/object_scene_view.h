#pragma once
#include <algorithm>
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "plugins/terrain_gen/terrain.h"
#include "game_object/drawable_object.h"
#include "game_object/patch.h"
#include "game_object/sphere.h"
#include "game_object/view_object.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace scene_rendering {

class ObjectSceneView : public ego::ViewObject {
    std::vector<std::shared_ptr<ego::DrawableObject>> m_drawable_objects_;

    // Ground decals (currently the road ribbon + fade skirt loaded from
    // <map>_pcg_decal.glb).  Deliberately a SEPARATE list rather than a
    // flag checked inside the main loop: decals must be drawn in their
    // own pass, at a different point in the frame (after the terrain
    // tiles, so there is completed ground depth to blend against) and
    // with a different pipeline state (alpha blend on, depth write off).
    std::vector<std::shared_ptr<ego::DrawableObject>> m_decal_objects_;

    bool m_b_render_blend_ = false;

public:
    ObjectSceneView(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
        const std::shared_ptr<ego::CameraObject>& camera_object,
        const std::shared_ptr<er::TextureInfo>& color_buffer = nullptr,
        const std::shared_ptr<er::TextureInfo>& depth_buffer = nullptr,
        const glm::uvec2& buffer_size = glm::uvec2(2560, 1440),
        bool depth_only = false);

    void addDrawableObject(
        const std::shared_ptr<ego::DrawableObject>& drawable_object) {
        m_drawable_objects_.push_back(drawable_object);
    }

    void removeDrawableObject(
        const std::shared_ptr<ego::DrawableObject>& drawable_object) {
        m_drawable_objects_.erase(
            std::remove(m_drawable_objects_.begin(), m_drawable_objects_.end(),
                        drawable_object),
            m_drawable_objects_.end());
    }

    void addDecalObject(
        const std::shared_ptr<ego::DrawableObject>& decal_object) {
        m_decal_objects_.push_back(decal_object);
    }

    void removeDecalObject(
        const std::shared_ptr<ego::DrawableObject>& decal_object) {
        m_decal_objects_.erase(
            std::remove(m_decal_objects_.begin(), m_decal_objects_.end(),
                        decal_object),
            m_decal_objects_.end());
    }

    bool hasDecalObjects() const { return !m_decal_objects_.empty(); }

    void duplicateColorAndDepthBuffer(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    // Depth-only variant of the above.  The decal pass needs scene depth
    // but not scene colour, and the colour blit is a full-resolution
    // copy of an HDR target every frame — not something to pay for when
    // nothing reads it.
    void duplicateDepthBuffer(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    // ── Ground-decal pass ────────────────────────────────────────────
    // Call AFTER the forward pass AND after the terrain tiles have been
    // drawn into this view's color/depth buffers.  Blits depth into
    // m_depth_buffer_copy_ (so the fragment shader can sample the scene
    // depth it is simultaneously depth-testing against), then re-opens
    // colour and depth with LOAD_OP_LOAD and draws every registered
    // decal through DrawMode::kDecal — depth test ON so houses still
    // occlude the road, depth write OFF so decals never occlude each
    // other, and alpha blend on so the depth-difference factor computed
    // in base.frag's DECAL permutation actually reaches the framebuffer.
    //
    // The caller must have written this view's depth copy into
    // SCENE_DEPTH_TEX_INDEX of the set-0 descriptor passed in desc_sets.
    // Deferred re-rasterise: draw every registered drawable into the
    // cluster G-buffer (4 RTs + the depth this view's forward pass
    // stamped earlier this frame, all LOAD).  DrawMode::kGBuffer
    // pipelines depth-test LESS_OR_EQUAL with writes off, so only
    // visible surfaces write attributes; deferred_resolve.comp then
    // lights those pixels with the traced shadow / RT-GI path.  The
    // decal list is deliberately excluded — alpha-blended geometry has
    // no place in an opaque G-buffer (same rule terrain applies to its
    // grass and water).
    void drawGbuffer(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<std::shared_ptr<renderer::ImageView>>& gbuffer_views,
        const std::shared_ptr<renderer::ImageView>& depth_view,
        const glm::uvec2& buffer_size);

    // Forward translucent GLASS pass: draw ONLY the Blend/glass
    // primitives of every registered drawable, alpha-blended over the
    // finished scene colour (depth test on, writes off).  Runs AFTER
    // the resolve + sky + decals, so what shows through a pane is the
    // real rendered world — rasterised translucency IS the thin-pane
    // transmission term.
    void drawGlassForward(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::shared_ptr<renderer::ImageView>& color_view,
        const std::shared_ptr<renderer::ImageView>& depth_view,
        const glm::uvec2& buffer_size);

    // Forward ground-decal pass: blends the decal meshes over the
    // finished scene colour.  Used by the PURE-FORWARD configuration
    // only — in deferred mode the decals go through
    // drawDecalsGbuffer() instead, because a forward decal drawn after
    // the resolve gets no shadow in any RT mode (see that function).
    void drawDecals(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        int dbuf_idx,
        float delta_t,
        float cur_time);

    // Deferred ground-decal pass: the same meshes re-rasterised into
    // the cluster G-buffer BEFORE the resolve, so deferred_resolve.comp
    // lights ground-plus-decal as one surface and the decal inherits
    // the ground's traced shadow / RT AO / RT GI.  Issue AFTER
    // drawGbuffer() and the terrain tiles' G-buffer pass.
    void drawDecalsGbuffer(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        const std::vector<std::shared_ptr<renderer::ImageView>>& gbuffer_views,
        const std::shared_ptr<renderer::ImageView>& depth_view,
        const glm::uvec2& buffer_size);

    virtual void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        std::shared_ptr<ego::Sphere> sphere,
        int dbuf_idx,
        float delta_t,
        float cur_time,
        bool depth_only = false,
        const std::shared_ptr<er::ImageView>& depth_layer_view = nullptr,
        uint32_t layer_count = 1,
        // When true, the depth attachment is opened with LOAD_OP_LOAD
        // instead of LOAD_OP_CLEAR so a prior render pass (e.g. the
        // CSM silhouette prepass that fills in-frustum texels with
        // depth=1 over a 0-clear) is preserved.  Caller is responsible
        // for having initialised the depth contents.  Color attachment
        // load behaviour is unaffected.
        bool preserve_depth = false,
        // Set to >=0 to drive the CSM per-cascade pipeline (DrawMode
        // ::kCsmPerCascade — "Regular" shadow draw mode).  When >=0,
        // layer_count MUST be 1, depth_layer_view MUST point at a
        // single-layer view of the right cascade, and the host is
        // expected to loop k=0..CSM_CASCADE_COUNT-1 calling this draw
        // once per cascade.  Default -1 = "not a per-cascade pass",
        // legacy dispatch (kShadow / kCsmLayered / kForward) applies.
        int32_t csm_cascade_idx = -1,
        // When true AND layer_count > 1 (i.e. layered CSM draw),
        // selects DrawMode::kCsmMeshShader instead of kCsmLayered so
        // DrawableObject::draw routes through the task+mesh shader
        // path.  Ineligible primitives (skinned, cutout, etc.) fall
        // back to the GS pipeline inside drawMesh.  Ignored unless
        // layered CSM is in effect.
        bool csm_use_mesh_shader = false);

    // Re-allocate descriptor sets from the (new) descriptor pool and
    // resize render buffers after a swap chain recreation.
    void recreate(
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
        const glm::uvec2& new_buffer_size);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine