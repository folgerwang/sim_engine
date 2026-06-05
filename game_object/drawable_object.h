#pragma once
#include <atomic>
#include <unordered_map>
#include "renderer/renderer.h"
#include "helper/bvh.h"
#include "helper/cluster_mesh.h"  // Optional "Nanite-lite" cluster sidecar.
#include "cluster_debug_draw.h"   // Per-mesh GPU buffers for the cluster debug draw path.

namespace engine {
namespace game_object {

class MeshLoadTaskManager;  // fwd-decl for async load API.

// glTF / FBX alpha-mode categorisation.
//   Opaque - fully opaque, no alpha test, no blending. Default.
//   Mask   - alpha test against alpha_cutoff_. Still depth-write & opaque-bin.
//   Blend  - alpha-blend with framebuffer. Drawn after opaque + mask, no
//            depth-write so transparent surfaces don't occlude each other.
//
// Promoted from the previous bool (alpha_mask_) so we can express BLEND
// as a first-class state. The legacy `alpha_mask_` field is kept as a
// derived alias for downstream code that hasn't migrated yet.
enum class AlphaMode : uint8_t {
    Opaque = 0,
    Mask   = 1,
    Blend  = 2,
};

struct MaterialInfo {
    // Source-asset material name (e.g. "MASTER_BistroFloor",
    // "Wood_Beams_Dark"). Populated at FBX/GLTF load directly from
    // the source material -- gameplay code can read it to decide
    // surface-specific behaviour (footstep sounds, friction,
    // bullet impact decals, etc.). Empty if the source had no name.
    std::string            name_;

    int32_t                base_color_idx_ = -1;
    int32_t                normal_idx_ = -1;
    int32_t                metallic_roughness_idx_ = -1;
    int32_t                specular_color_idx_ = -1;
    int32_t                emissive_idx_ = -1;
    int32_t                occlusion_idx_ = -1;

    // Cluster renderer needs these CPU-side (avoids re-reading the GPU UBO).
    float                  alpha_cutoff_ = 0.0f;  // >0 enables alpha-mask discard
    AlphaMode              alpha_mode_   = AlphaMode::Opaque;
    bool                   alpha_mask_   = false; // legacy alias = (alpha_mode_ == Mask)

    // True if the loader detected this material as glass-like by name
    // (substring match on "glass" / "window" / "transparent"). Promoted
    // to AlphaMode::Blend with a low base-color alpha. The flag is kept
    // around so debug UIs can highlight forced-glass materials.
    bool                   glass_forced_ = false;

    // ── Effective-opaque flag (texture-content-aware) ─────────────────
    // The CPU loader sets alpha_mode_ from the asset's metadata (gltf
    // material.alphaMode, FBX heuristics, glass-by-name overrides).
    // That classification is conservative: many assets ship materials
    // flagged Mask "just in case" but whose albedo texture is actually
    // fully opaque (every texel α == 255).  For shadow rendering we
    // can treat those as truly opaque and route them through the
    // no-fragment-shader pipeline — a measurable win in heavy scenes.
    //
    // This flag is initialised to true and downgraded to false after
    // load if EITHER:
    //   • alpha_mode_ == Blend (real translucency, no shortcut), OR
    //   • alpha_mode_ == Mask AND the albedo texture contains at
    //     least one texel with α below the opacity threshold (i.e.
    //     real cutout exists in the texture).
    // It stays true when:
    //   • alpha_mode_ == Opaque (the easy case), OR
    //   • alpha_mode_ == Mask but the albedo texture has no real
    //     translucency (every α == 255 — asset author over-flagged).
    //
    // Consumers: isPrimitiveOpaque() in drawable_object.cpp uses this
    // to decide whether to bind the no-frag shadow pipeline.
    bool                   effective_opaque_ = true;

    renderer::BufferInfo   uniform_buffer_;
    std::shared_ptr<renderer::DescriptorSet>  desc_set_;
};

struct BufferView {
    uint32_t                buffer_idx;
    uint64_t                stride;
    uint64_t                offset;
    uint64_t                range;
};

union PrimitiveHashTag {
    uint32_t                data = 0;
    struct {
        uint32_t                has_normal : 1;
        uint32_t                has_tangent : 1;
        uint32_t                has_texcoord_0 : 1;
        uint32_t                has_skin_set_0 : 1;
        uint32_t                restart_enable : 1;
        uint32_t                double_sided : 1;
        uint32_t                topology : 16;
    };
};

struct PrimitiveInfo {
private:
    size_t hash_ = 0;
    size_t depthonly_hash_ = 0;
public:
    int32_t                 material_idx_;
    int32_t                 indirect_draw_cmd_ofs_;
    PrimitiveHashTag        tag_;
    glm::vec3               bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    // NOTE: numeric_limits<float>::min() is the smallest POSITIVE
    // normalized float (~1.18e-38), NOT the most negative — using it
    // as a "lower than any real max" init silently breaks the
    // max-aggregation for any axis whose true max is negative, leaving
    // bbox_max stuck at that ~0 sentinel.  Use ::lowest() (the most
    // negative finite float) so the standard max(bbox, vert) update
    // correctly overrides on the first vertex.
    glm::vec3               bbox_max_ = glm::vec3(std::numeric_limits<float>::lowest());
    std::shared_ptr<renderer::AccelerationStructureGeometry>  as_geometry;
    std::shared_ptr<std::vector<int32_t>> vertex_indices_;
    std::shared_ptr<helper::BVHNode> bvh_root_;

    std::vector<renderer::IndexInputBindingDescription>  index_desc_;
    std::vector<renderer::VertexInputBindingDescription> binding_descs_;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs_;

    // ── Mesh-shader CSM path (DrawMode::kCsmMeshShader) ─────────────
    // Non-null when the primitive is eligible for the mesh-shader
    // shadow path (opaque, non-skinned, UINT32 indices, vertex_count
    // and tri_count both <= 256) AND a mesh-shader pipeline was
    // successfully built.  Bound at set 0 of the mesh-shader-shadow
    // pipeline layout; references the primitive's VB + IB + the owning
    // drawable's instance_buffer as storage buffers.  nullptr means
    // "not eligible — drawMesh falls back to the GS pipeline".
    std::shared_ptr<renderer::DescriptorSet> mesh_shader_shadow_desc_set_;
    // Layout descriptors consumed by the MeshShadowPC push constant.
    // All values in FLOATS (not bytes) — buildMeshShaderShadowResources
    // converts byte offsets to float strides at pipeline-build time.
    uint32_t mesh_shader_vb_stride_floats_          = 0;
    uint32_t mesh_shader_vb_position_offset_floats_ = 0;
    uint32_t mesh_shader_ib_first_index_            = 0;
    uint32_t mesh_shader_vertex_count_              = 0;
    uint32_t mesh_shader_tri_count_                 = 0;

    // (Per-primitive cluster_mesh_ removed — see MeshInfo::cluster_prim_map_)

    void generateHash();
    size_t getHash() const { return hash_; }
    size_t getDepthonlyHash() const { return depthonly_hash_; }
};

struct BufferViewInfo {
    uint32_t                buffer_view_idx;
    uint64_t                offset;
    renderer::Format        format;
};

struct DrawableData;
struct AnimChannelInfo {
    enum AnimChannelType {
        kTranslation,
        kRotation,
        kScale,
        kMaxNumChannels,
    };

    AnimChannelType         type_;
    uint32_t                node_idx_;
    BufferViewInfo          sample_buffer_;
    BufferViewInfo          data_buffer_;
    std::vector<std::pair<float, glm::vec4>>    samples_;

    void update(
        DrawableData* object,
        float time,
        float time_scale = 1.0f,
        bool repeat = true);
};

struct AnimationInfo {
    std::vector<std::shared_ptr<AnimChannelInfo>> channels_;
};

struct SkinInfo {
    std::string             name_;
    int32_t                 skeleton_root_;
    std::vector<int32_t>    joints_;
    std::vector<glm::mat4>  inverse_bind_matrices_;
    renderer::BufferInfo    joints_buffer_;
    std::shared_ptr<renderer::DescriptorSet>    desc_set_;
};

struct MeshInfo {
    std::vector<PrimitiveInfo>  primitives_;
    std::shared_ptr<std::vector<glm::vec3>> vertex_position_;
    glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    // See PrimitiveInfo::bbox_max_ comment — ::min() is the smallest
    // positive normalized float, NOT the most negative.  Use ::lowest().
    glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::lowest());

    // "Nanite-lite" cluster sidecar. Built at load time *only* when
    // engine::helper::clusterRenderingEnabled() is true (i.e. the
    // --cluster-debug CLI flag was set). Otherwise it stays empty and costs
    // nothing. See helper/cluster_mesh.h for the data layout.
    helper::ClusterMesh         cluster_mesh_;

    // GPU-side companion buffers for the cluster debug draw path. Populated
    // by ClusterDebugDraw::uploadForMesh() immediately after cluster_mesh_
    // is built, and consumed by ClusterDebugDraw::draw() in place of the
    // normal primitive loop while --cluster-debug is active. Empty and
    // zero-cost when the flag is off.
    ClusterDebugMeshBuffers     cluster_debug_gpu_;

    // Per-cluster primitive index: cluster_prim_map_[i] = primitive index
    // whose material cluster i belongs to. Built alongside cluster_mesh_ by
    // checking which primitive's face range each cluster's first face falls in.
    // Passed to uploadMeshClusters() so it can assign per-cluster materials
    // while keeping ONE cluster mesh per FBX mesh (no per-primitive BVH overhead).
    std::vector<uint32_t>       cluster_prim_map_;

    // ClusterRenderer per-mesh index. Set during cluster upload in
    // application.cpp. -1 means this mesh has no cluster data.
    int32_t cluster_global_mesh_idx_ = -1;
};

struct NodeInfo {
    std::string                 name_;
    int32_t                     parent_idx_ = -1;
    std::vector<int32_t>        child_idx_;

    int32_t                     mesh_idx_ = -1;
    int32_t                     skin_idx_ = -1;

    glm::vec3                   translation_{};
    glm::vec3                   scale_{1.0f};
    glm::quat                   rotation_{};
    glm::mat4                   matrix_ = glm::mat4(1.0f);

    glm::mat4                   cached_matrix_ = glm::mat4(1.0f);
    glm::mat4 getLocalMatrix(bool use_local_matrix_only);
    const glm::mat4& getCachedMatrix() const {
        return cached_matrix_;
    }
};

struct SceneInfo {
    std::vector<int32_t>        nodes_;
    glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    // See PrimitiveInfo::bbox_max_ comment — ::min() is the smallest
    // positive normalized float, NOT the most negative.  Use ::lowest().
    glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::lowest());
};

struct DrawableData {
    const std::shared_ptr<renderer::Device>& device_;
    bool m_use_local_matrix_only_ = false;
    bool m_flip_u_ = false;
    bool m_flip_v_ = false;
    // Mirror of DrawableObject::use_node_transform_only_, kept on
    // DrawableData so the static drawMesh helper (which only receives
    // DrawableData*) can read it for the skinned-character per-mesh
    // frustum-cull bypass.  Written by DrawableObject::setUseNode-
    // TransformOnly.  See that setter's doc-comment for the full
    // semantics (identity instance buffer + skip imported animations).
    bool m_use_node_transform_only_ = false;
    // "Force red" debug override — when set, drawNodes pushes
    // model_params.debug_force_red=1 for every primitive draw belonging
    // to this DrawableData.  base.frag's final pass overrides outColor
    // with vec4(1,0,0,1) when that flag is non-zero.  Used to verify
    // that a "missing" drawable's draw is actually reaching the
    // rasterizer (versus being silently culled / never bound).  Set
    // via DrawableObject::setDebugForceRed.
    bool m_debug_force_red_ = false;
    // Editor selection highlight target node index.  -1 = no highlight,
    // -2 = highlight the WHOLE drawable, >=0 = highlight just that node.
    // drawNodes() pushes debug_force_red=2 for matching nodes and base.frag
    // tints them amber — a real highlight layer on the original mesh.
    int32_t m_highlight_node_ = -1;
    // "Skip skinning" debug override — when set, drawNodes pushes
    // model_params.debug_skip_skinning=1 for every primitive draw on
    // this drawable.  base.vert then renders the mesh in its glTF
    // bind pose (skin_matrix collapses to identity) instead of
    // deforming it through joint matrices.  Useful when a skinned
    // character is invisible — bypasses every potential skin-math
    // bug (degenerate inv_bind, mis-sized joint_matrices, bad node
    // hierarchy) and renders the raw bind-pose mesh at the root
    // node's world transform, so the only thing that can hide the
    // draw is genuine culling / pipeline / material breakage.
    bool m_debug_skip_skinning_ = false;
    // Force-override the root node's local scale when > 0.  Wired by
    // setRootNodeTransform so the override travels through every
    // subsequent applyPose call from PlayerController without needing
    // a separate sync path.  See DrawableObject::setDebugScale.
    float m_debug_scale_ = 0.0f;
    // Counter of indirect-draw submissions issued for THIS drawable
    // per DrawableObject::draw() call.  Only incremented when
    // m_debug_force_red_ is set (currently the player).  Reset at the
    // top of DrawableObject::draw, read at the bottom for the per-
    // second diagnostic print.
    uint32_t m_debug_draw_call_count_ = 0;
    // ── "where did the draw go?" reach counters ────────────────────
    // Companion to m_debug_draw_call_count_.  When m_debug_log_draws_
    // is set, every gating point in the draw path that can early-out
    // a primitive bumps the matching counter.  Printed alongside
    // indirect_draws_issued in the per-second [player.draw] log so
    // we can see WHY an invisible drawable issued 0 draws.  Reset at
    // the top of DrawableObject::draw().
    uint32_t m_debug_draw_called_         = 0; // entered draw() at all
    uint32_t m_debug_draw_not_ready_      = 0; // exited via !isReady() guard
    uint32_t m_debug_draw_nodes_visited_  = 0; // drawNodes() entries (any node)
    uint32_t m_debug_draw_nodes_with_mesh_ = 0; // nodes that had mesh_idx_ >= 0
    uint32_t m_debug_draw_mesh_entered_   = 0; // drawMesh() entries
    uint32_t m_debug_draw_mesh_culled_frustum_      = 0;
    uint32_t m_debug_draw_mesh_taken_by_cluster_    = 0; // cluster_global_mesh_idx_ >= 0
    uint32_t m_debug_draw_mesh_cluster_debug_path_  = 0; // cluster_debug GPU path
    uint32_t m_debug_draw_prims_iterated_           = 0; // primitive loop iterations
    uint32_t m_debug_draw_prim_pipeline_null_       = 0; // (*pipelines)[cur_hash] == null
    uint32_t m_debug_draw_prim_mesh_shader_         = 0; // dispatched via drawMeshTasks
    // Independent "log this drawable's draw calls to stdout" flag.
    // Decoupled from m_debug_force_red_ so we can see the per-second
    // [player.draw] indirect-draws-issued tally WITHOUT also tinting
    // the character red.  setDebugLogDraws(true) wires both the
    // counter increment and the per-second print.
    bool m_debug_log_draws_ = false;

    // ── Per-draw instance-world scratchpad ───────────────────────────
    // Written by DrawableObject::draw() at the top of every draw call so
    // drawNodes can apply the per-instance world transform on top of the
    // shared cached_matrix (model_mat = m_current_instance_world_ * cached).
    // For drawables that DON'T share their DrawableData (the player rig,
    // standalone meshes) this stays identity and the existing model_mat =
    // cached_matrix behaviour is preserved.  For SHARED-instance drawables
    // (e.g. the debug-cube markers, which all share one loaded mesh) each
    // wrapper writes its own world here before recording its draw commands;
    // since push-constants are baked into the cmd stream at record time,
    // each marker's recorded model_mat carries its own per-instance world.
    glm::mat4 m_current_instance_world_ = glm::mat4(1.0f);

    int32_t                     default_scene_ = 0;
    std::vector<SceneInfo>      scenes_;
    std::vector<NodeInfo>       nodes_;
    std::vector<MeshInfo>       meshes_;
    std::vector<AnimationInfo>  animations_;
    std::vector<SkinInfo>       skins_;
    std::vector<renderer::BufferInfo>     buffers_;
    std::vector<BufferView>     buffer_views_;

    std::vector<renderer::TextureInfo>    textures_;
    std::vector<MaterialInfo>   materials_;

    uint32_t                    num_prims_ = 0;
    renderer::BufferInfo        indirect_draw_cmd_;
    renderer::BufferInfo        instance_buffer_;

    std::shared_ptr<renderer::DescriptorSet> indirect_draw_cmd_buffer_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> update_instance_buffer_desc_set_;

    // Set true on the main thread at the end of async phase3, after
    // descriptor-set / pipeline / instance-buffer creation. The render
    // loop must check this before touching any of the members above —
    // they may still be default-constructed while an async load is in
    // flight. See DrawableObject::createAsync and DrawableObject::isReady.
    std::atomic<bool> ready_{false};

public:
    DrawableData(const std::shared_ptr<renderer::Device>& device) : device_(device) {}
    ~DrawableData() {}

    // skip_animations: when true, the imported glTF animation channels
    // are NOT evaluated for this frame.  Node transforms keep whatever
    // values the caller has just written (e.g. PlayerController::
    // applyPose for procedurally-driven rigs).  Joint matrices are still
    // recomputed from the current node hierarchy, so the rig still
    // animates — just from the caller's writes instead of the imported
    // animation timeline.
    void update(
        const std::shared_ptr<renderer::Device>& device,
        const uint32_t& active_anim_idx,
        const float& time,
        bool use_local_matrix_only,
        bool skip_animations = false);

    glm::mat4 getNodeMatrix(
        const int32_t& node_idx,
        bool use_local_matrix_only);

    void updateJoints(
        const std::shared_ptr<renderer::Device>& device,
        int32_t node_idx);

    void generateSharedDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::DescriptorSetLayout>& gltf_indirect_draw_desc_set_layout,
        const std::shared_ptr<renderer::DescriptorSetLayout>& update_instance_buffer_desc_set_layout,
        const std::shared_ptr<renderer::BufferInfo>& game_objects_buffer);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

class DrawableObject {
    enum {
        kMaxNumObjects = 10240
    };
    std::shared_ptr<DrawableData>   object_;
    glm::mat4                   location_;

    // See setUseNodeTransformOnly() in the public section for what this
    // flag does and why the player controller has to set it.
    // NOTE: the canonical storage lives on DrawableData as
    // m_use_node_transform_only_ so the static drawMesh helpers (which
    // only receive DrawableData*) can read it for the skinned-character
    // frustum-cull bypass.  This field on the outer wrapper is kept as
    // a cache to make getUseNodeTransformOnly() valid even before
    // object_ is populated; the setter writes both.
    bool                        use_node_transform_only_ = false;

    // ── Per-instance world override (for shared-mesh drawables) ──────
    // When the same loaded mesh (one shared DrawableData) is rendered as
    // multiple instances, each outer wrapper holds its own world transform
    // here.  DrawableObject::draw() copies it into DrawableData::
    // m_current_instance_world_ at the top of each draw call, so the
    // per-instance world gets baked into that draw's push-constants.  The
    // shared DrawableData::nodes_ stay untouched -- writing to
    // setRootNodeTransform on a shared mesh would clobber every sibling
    // instance.  setInstanceRootTransform() sets these fields; when
    // instance_root_valid_ is false the override is identity (existing
    // single-instance behaviour).
    glm::vec3                   instance_root_t_ = glm::vec3(0.0f);
    glm::quat                   instance_root_r_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    // Per-instance non-uniform scale.  Composed AFTER rotation in draw():
    //   instance_world = translate(t) * mat4_cast(r) * scale(s)
    // followed by the existing m_debug_scale_ multiplier (so a bone-link
    // stick can ask for a (length/debug, thin/debug, thin/debug) here and
    // the final on-screen dimensions come out (length, thin, thin) after
    // the shared debug_scale is applied).  Default (1,1,1) preserves the
    // existing behaviour for plain bone-marker cubes.
    glm::vec3                   instance_root_s_ = glm::vec3(1.0f);
    bool                        instance_root_valid_ = false;

    // Per-wrapper visibility gate.  When false, DrawableObject::draw()
    // early-returns BEFORE doing any work (no instance buffer bind, no
    // node walk, no shadow recording either, since the same draw() runs
    // for both passes via the depth_only / DrawMode params).  Set per
    // frame from application code — used by the Render Debug menu's
    // bone-only / character-only / both selector to toggle the player
    // mesh and the 19 bone-marker cubes without removing them from the
    // scene-view lists.  Defaults to true so existing drawables behave
    // exactly as before.
    bool                        visible_ = true;

    // static members.
    static uint32_t max_alloc_game_objects_in_buffer;

public:
    // ── Engine-wide material classification counters ──────────────────
    // Maintained by computeEffectiveOpaqueForMaterials whenever a new
    // mesh's materials are scanned.  Counts are CUMULATIVE — every load
    // adds to them; we never decrement on unload (no unload path
    // exists yet, and the counters represent "total materials ever
    // classified" rather than "materials currently resident").
    //
    // Read by VirtualTextureManager::tick to include in the per-frame
    // vt_pool.log line so the user can see, at a glance, how many of
    // the scene's materials are taking the slow alpha-cutoff shadow
    // path vs the fast no-frag path.
    //
    // Atomic because async mesh loads call computeEffectiveOpaqueForMaterials
    // off the main thread (see Phase2Fn in drawable_object.cpp).
    static std::atomic<int> s_total_materials_count_;
    static std::atomic<int> s_alpha_cutoff_materials_count_;
private:

    static std::shared_ptr<renderer::DescriptorSetLayout> material_desc_set_layout_;
    static std::shared_ptr<renderer::DescriptorSetLayout> skin_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> drawable_pipeline_layout_;
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_pipeline_list_;
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_shadow_pipeline_list_;
    // Single-pass CSM shadow pipeline — geometry shader broadcasts to all layers.
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_csm_layered_pipeline_list_;
    // Per-cascade CSM shadow pipeline (no GS, no mesh shader).  The host
    // loops cascades and pushes ModelParams.cascade_idx per draw; the VS
    // reads light_view_proj[cascade_idx] from the runtime-lights UBO.
    // Selected by DrawMode::kCsmPerCascade ("Regular" shadow draw mode).
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_csm_per_cascade_pipeline_list_;
    // Mesh-shader CSM shadow pipeline (task+mesh stages, no VS/GS).
    // Selected by DrawMode::kCsmMeshShader.  Per-primitive descriptor
    // sets (binding 0=VB SSBO, 1=IB SSBO, 2=instance_buffer SSBO) are
    // allocated at pipeline-build time and stored on PrimitiveInfo.
    // Supports opaque non-skinned UINT32-indexed primitives with
    // <=256 verts/tris; everything else falls back to the GS pipeline
    // inside drawMesh.
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_csm_mesh_shader_pipeline_list_;
    static std::shared_ptr<renderer::DescriptorSetLayout> mesh_shader_shadow_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout>      mesh_shader_shadow_pipeline_layout_;
    static std::unordered_map<std::string, std::shared_ptr<DrawableData>> drawable_object_list_;
    static std::shared_ptr<renderer::DescriptorSetLayout> drawable_indirect_draw_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> drawable_indirect_draw_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> drawable_indirect_draw_pipeline_;
    static std::shared_ptr<renderer::DescriptorSet> update_game_objects_buffer_desc_set_[2];
    static std::shared_ptr<renderer::DescriptorSetLayout> update_game_objects_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> update_game_objects_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> update_game_objects_pipeline_;
    static std::shared_ptr<renderer::DescriptorSetLayout> update_instance_buffer_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> update_instance_buffer_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> update_instance_buffer_pipeline_;
    static std::shared_ptr<renderer::BufferInfo> game_objects_buffer_;

    // View-camera buffer the application supplies for the
    // update_game_objects compute path.  Set once at init via the public
    // setter below; consumed by createGameObjectUpdateDescSet (binds it
    // at CAMERA_OBJECT_BUFFER_INDEX every time the descset is (re)created)
    // and by updateGameObjectsCameraBuffer (for late-arrival updates).
    // Static so the helper can find it without threading a parameter
    // through every caller chain.
    static std::shared_ptr<renderer::BufferInfo>
        s_view_camera_buffer_for_update_;

public:
    // Public accessor so the application (and other compilation units) can
    // wire the view-camera buffer into the game-objects update path before
    // generateDescriptorSet runs.  Setting this before the descset is
    // created causes createGameObjectUpdateDescSet to bind it atomically
    // alongside the other slots, eliminating the
    // VUID-vkCmdDispatch-None-08114 ("descriptor … never updated")
    // warning that would otherwise fire every frame the
    // update_game_objects compute fires.
    static void setViewCameraBufferForUpdate(
        const std::shared_ptr<renderer::BufferInfo>& buf) {
        s_view_camera_buffer_for_update_ = buf;
    }
    // Read-only accessor for the file-scope addGameObjectsInfoBuffer
    // helper to consult before deciding what buffer to bind at
    // CAMERA_OBJECT_BUFFER_INDEX.  Free function can't reach the static
    // directly because it lives at namespace scope (not as a class
    // member), so it goes through this public getter.
    static const std::shared_ptr<renderer::BufferInfo>&
    getViewCameraBufferForUpdate() {
        return s_view_camera_buffer_for_update_;
    }

private:
    // Used by createAsync() to build a shell whose object_ will be
    // populated by phase 3 on the main thread once the worker finishes.
    explicit DrawableObject(glm::mat4 location) : location_(location) {}

public:
    DrawableObject() = delete;
    DrawableObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::PipelineRenderbufferFormats* renderbuffer_formats,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& thin_film_lut_tex,
        const std::string& file_name,
        glm::mat4 location = glm::mat4(1.0f));

    // Async factory. Submits a MeshLoadTask to `task_manager` and
    // returns a shared_ptr<DrawableObject> whose object_ is nullptr
    // until the worker finishes file parsing + GPU upload and the
    // main-thread phase 3 (descriptor sets, pipelines, instance
    // buffer) runs during the next task_manager.poll() call.
    //
    // Callers must check isReady() before draw()/update()/updateBuffers();
    // those methods early-return when the object is not yet ready.
    // The returned shared_ptr is safe to push into draw lists immediately.
    static std::shared_ptr<DrawableObject> createAsync(
        MeshLoadTaskManager& task_manager,
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::PipelineRenderbufferFormats* renderbuffer_formats,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& thin_film_lut_tex,
        const std::string& file_name,
        glm::mat4 location = glm::mat4(1.0f));

    // True once phase 3 has published the populated DrawableData.
    // Uses memory_order_acquire so the caller sees all of phase3's
    // writes (descriptor sets, pipelines, buffers) synchronized
    // with the ready_ flag.
    bool isReady() const {
        return object_ && object_->ready_.load(std::memory_order_acquire);
    }

    // Access mesh info for cluster upload. Only valid when isReady().
    const std::vector<MeshInfo>& getMeshes() const { return object_->meshes_; }
    std::vector<MeshInfo>& getMutableMeshes() { return object_->meshes_; }
    const DrawableData& getDrawableData() const { return *object_; }
    const glm::mat4& getLocation() const { return location_; }

    // ── Player / procedural-pose helpers ─────────────────────────────────
    // scene-skinned.gltf has no animation channels, so PlayerController
    // drives the rig procedurally. These accessors expose just enough of
    // the node hierarchy to (a) place the model in the world and
    // (b) rotate individual bones each frame.
    void setRootNodeTransform(
        const glm::vec3& translation,
        const glm::quat& rotation);

    // Per-instance world override.  Use this — NOT setRootNodeTransform —
    // when the DrawableObject shares its underlying DrawableData with
    // other wrappers (i.e. multiple debug markers driven by one loaded
    // mesh).  setRootNodeTransform would write into the shared node TRS
    // and every sibling instance would jump to whichever wrapper called
    // it last.  This instead stores the per-wrapper world here, and
    // DrawableObject::draw() composes it into model_mat at record time.
    void setInstanceRootTransform(
        const glm::vec3& translation,
        const glm::quat& rotation,
        const glm::vec3& scale = glm::vec3(1.0f)) {
        instance_root_t_     = translation;
        instance_root_r_     = rotation;
        instance_root_s_     = scale;
        instance_root_valid_ = true;
    }

    // Read-back of the per-instance world override, for the editor transform
    // panel and scene serialization.  Meaningful once setInstanceRootTransform
    // has run (hasInstanceRoot() == true).
    const glm::vec3& getInstanceRootTranslation() const { return instance_root_t_; }
    const glm::quat& getInstanceRootRotation()    const { return instance_root_r_; }
    const glm::vec3& getInstanceRootScale()       const { return instance_root_s_; }
    bool             hasInstanceRoot()            const { return instance_root_valid_; }

    // Per-wrapper visibility gate (forward + shadow).  See the visible_
    // field comment above.  False -> draw() does nothing; true (default)
    // -> existing behaviour.
    void setVisible(bool v) { visible_ = v; }
    bool isVisible() const  { return visible_; }

    // Opt the drawable into "external code fully owns the rig" mode.
    // PlayerController-driven drawables (and any future similar
    // procedurally-posed character) MUST enable this.  Has TWO effects,
    // both required to make the player render at the controller's
    // spawn position with the controller's procedural pose:
    //
    //  1. updateInstanceBuffer's compute pass writes an IDENTITY
    //     instance transform (zero translation, unit rotation, scale 1)
    //     instead of reading the shared game_objects_buffer_'s slot 0
    //     position.  That slot is initialised to camera_pos on frame 0
    //     and then drifts via gravity — without this flag, base.vert
    //     compounds it with the node translation, double-transforming
    //     the drawable and pushing it 2× the camera offset off-screen.
    //
    //  2. The imported glTF animation channels are NOT evaluated each
    //     frame.  For rigs with baked animations (e.g. CesiumMan's walk
    //     cycle), the animation update would otherwise run AFTER
    //     PlayerController::applyPose and clobber the controller's just-
    //     written root translation + limb rotations — making the
    //     character render at the animation's authored origin (~0,0,0)
    //     and replay its imported limb pose on top of the controller's.
    //     Joint matrices are still recomputed from the current node
    //     hierarchy, so the rig still animates — just from the
    //     controller's writes instead of the imported timeline.
    void setUseNodeTransformOnly(bool v) {
        use_node_transform_only_ = v;          // cache on outer wrapper
        if (object_) object_->m_use_node_transform_only_ = v;  // mirror to data
    }
    bool getUseNodeTransformOnly() const { return use_node_transform_only_; }

    // ── Debug "force red" override ──────────────────────────────────
    // When true, every primitive of this drawable renders as a flat
    // bright red pixel through base.frag (debug_force_red is pushed
    // via model_params).  Smoke test for "is this draw actually
    // reaching the rasterizer?".  Safe to leave on indefinitely —
    // affects only this object's color output, nothing else.
    void setDebugForceRed(bool v) {
        if (object_) object_->m_debug_force_red_ = v;
    }
    bool getDebugForceRed() const {
        return object_ ? object_->m_debug_force_red_ : false;
    }
    // Editor selection highlight: -1 none, -2 whole drawable, >=0 a node.
    void setHighlightNode(int n) {
        if (object_) object_->m_highlight_node_ = n;
    }

    // ── Debug "skip skinning" override ───────────────────────────────
    // When true, every primitive of this drawable renders in its
    // glTF bind pose (base.vert's skin_matrix multiplication is
    // gated off via the matching push-constant flag).  Useful to
    // isolate whether a missing skinned character is failing in the
    // skin math vs. the draw not running.  The mesh appears at its
    // node's world transform with no bone deformation — looks like a
    // statue in the T-pose / bind pose the asset was authored at.
    void setDebugSkipSkinning(bool v) {
        if (object_) object_->m_debug_skip_skinning_ = v;
    }
    bool getDebugSkipSkinning() const {
        return object_ ? object_->m_debug_skip_skinning_ : false;
    }

    // ── Debug giant-size override ────────────────────────────────────
    // Force-overrides the root node's scale to (s, s, s) on every
    // subsequent setRootNodeTransform call (which PlayerController
    // invokes every frame).  Combined with whatever scale the asset's
    // own root matrix bakes in (0.1 for scene-skinned.gltf), the
    // rendered size becomes 0.1 × s × asset_units.  Pass 0 to disable
    // the override and let the asset's authored scale stand.  Use 30
    // to "blow the character up so it pokes through any wall" when
    // visibility-debugging.
    void setDebugScale(float s) {
        if (object_) object_->m_debug_scale_ = s;
    }
    float getDebugScale() const {
        return object_ ? object_->m_debug_scale_ : 0.0f;
    }

    // ── Debug: log per-second [player.draw] tally ───────────────────
    // When true, DrawableObject::draw prints the indirect-draws-
    // issued count (and "SKIPPED !isReady" if applicable) once every
    // ~60 frames.  Independent of setDebugForceRed (which affects
    // visual output).  Use this to confirm whether all primitives are
    // being culled vs. truly reaching the rasterizer, without
    // tinting the character.
    void setDebugLogDraws(bool v) {
        if (object_) object_->m_debug_log_draws_ = v;
    }
    bool getDebugLogDraws() const {
        return object_ ? object_->m_debug_log_draws_ : false;
    }

    int  findNodeIndexByName(const std::string& name) const;
    // Read the current local rotation of a node by its asset-side
    // name.  Returns identity when the name doesn't resolve OR when
    // the drawable shell isn't ready yet (object_ == nullptr or
    // async load still in flight) — caller decides whether identity
    // is acceptable.  Used by PlayerController to snapshot each
    // named bone's bind-pose rotation on first frame so subsequent
    // animation writes can COMPOSE against the bind rather than
    // overwriting it (raw `setNodeRotationByName(bone, swing_q)`
    // discards the asset's authored bone orientation and produces
    // distorted poses — twist where it should be swing, etc.).
    glm::quat getNodeRotationByName(const std::string& name) const;
    bool setNodeRotationByName(
        const std::string& name,
        const glm::quat& rotation);

    // Returns the WORLD-space transform of the named node as of the
    // most recent DrawableObject::update().  This is the cached
    // parent-chain product that the skinning path already maintains
    // (DrawableData::update writes nodes_[i].cached_matrix_ each
    // frame).  Returns identity when:
    //   - the drawable shell isn't ready (object_ == nullptr or
    //     async load still in flight), or
    //   - no node with that name exists on this rig.
    // Caller is responsible for combining with any external "root
    // node override" if it cares — the cached matrix already
    // includes the setRootNodeTransform override applied in the
    // last frame, so for the player rig this returns positions in
    // world space that match the on-screen render.  Used by the
    // foot-marker debug visualization in application.cpp.
    glm::mat4 getNodeWorldMatrixByName(const std::string& name) const;
    glm::vec3 getModelBboxMin() const;
    glm::vec3 getModelBboxMax() const;

    void updateInstanceBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void updateIndirectDrawBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void updateBuffers(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    // How the object should be drawn.
    enum class DrawMode {
        kForward,       // regular lit forward pass
        kShadow,        // per-cascade depth-only pass (legacy, 4 separate draws)
        kCsmLayered,    // single-pass depth-only with GS broadcasting to all CSM layers
        kCsmPerCascade, // per-cascade depth-only pass; VS reads
                        // light_view_proj[ModelParams.cascade_idx] from
                        // the runtime-lights UBO.  Used by the "Regular"
                        // option on the shadow draw-mode menu.
        kCsmMeshShader, // Single-pass depth-only via task+mesh shaders.
                        // task amplifies one drawcall into
                        // CSM_CASCADE_COUNT mesh workgroups; the mesh
                        // shader fetches VB/IB/instance via per-primitive
                        // SSBOs.  Ineligible primitives (skinned,
                        // cutout, UINT16 indices, >256 verts/tris) fall
                        // back to the GS pipeline inside drawMesh.
    };

    void draw(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors,
        bool depth_only = false,
        DrawMode draw_mode = DrawMode::kForward,
        // Only consumed by DrawMode::kCsmPerCascade — written into each
        // drawcall's ModelParams.cascade_idx so the _CSMCASC vertex
        // shader picks the right cascade VP from the runtime-lights UBO.
        // Ignored by every other DrawMode.
        uint32_t csm_cascade_idx = 0);

    // Static accessor for the mesh-shader shadow pipeline layout.
    // Needed by drawMesh (file-scope static) which can't reach the
    // private static directly.
    static const std::shared_ptr<renderer::PipelineLayout>&
        getMeshShaderShadowPipelineLayout() {
        return mesh_shader_shadow_pipeline_layout_;
    }

    void update(
        const std::shared_ptr<renderer::Device>& device,
        const float& time);

    void destroy(
        const std::shared_ptr<renderer::Device>& device) {
        if (object_) {
            object_->destroy(device);
        }
    }

    static void createGameObjectUpdateDescSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
//        const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::TextureInfo& water_flow,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    // Per-frame frustum cull state. Set once per frame before draw(),
    // used inside drawMesh() to skip meshes whose world-space bounding
    // sphere is entirely outside the view frustum. The planes are in
    // world space (Gribb-Hartmann, normalised).
    static void setFrustumCullPlanes(const glm::vec4 planes[6]);
    static void clearFrustumCull();

    static void initGameObjectBuffer(
        const std::shared_ptr<renderer::Device>& device);

    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
//        const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::TextureInfo& water_flow,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    static void createStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts);

    static void recreateStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::PipelineRenderbufferFormats* renderbuffer_formats,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts);

    static void generateDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
//        const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
        const renderer::TextureInfo& thin_film_lut_tex,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::TextureInfo& water_flow,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static void updateGameObjectsCameraBuffer(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::BufferInfo>& view_camera_buffer);

    static void updateGameObjectsBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::vec2& world_min,
        const glm::vec2& world_range,
        const glm::vec3& camera_pos,
        float air_flow_strength,
        float water_flow_strength,
        int update_frame_count,
        int soil_water,
        float delta_t,
        bool enble_airflow);

    static std::shared_ptr<renderer::BufferInfo> getGameObjectsBuffer();

    static std::shared_ptr<DrawableData> loadGltfModel(
        const std::shared_ptr<renderer::Device>& device,
        const std::string& input_filename);

    static std::shared_ptr<DrawableData> loadFbxModel(
        const std::shared_ptr<renderer::Device>& device,
        const std::string& input_filename);
};

} // namespace game_object
} // namespace engine