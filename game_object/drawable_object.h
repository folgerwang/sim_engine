#pragma once
#include <atomic>
#include <mutex>          // PcgInstanceRegistry guards its tables
#include <unordered_map>
#include <utility>        // std::pair, for DrawableData::mesh_instance_range_
#include "renderer/renderer.h"
#include "helper/bvh.h"
#include "helper/cluster_mesh.h"  // Optional "Nanite-lite" cluster sidecar.
#include "cluster_debug_draw.h"   // Per-mesh GPU buffers for the cluster debug draw path.
#include "ecs/material.h"         // Renderer-free MaterialDesc (dedup identity).

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

    // ── ECS dedup identity ─────────────────────────────────────────────
    // Renderer-free description of this material, captured at load time
    // right after the PbrMaterialParams UBO is filled (all four loaders).
    // Texture slots are keyed "<asset>#<texture-index>" so two instances
    // of the same asset (or two sub-materials sharing a texture) hash
    // identically.  Interned into the app's ecs::MaterialCache; the
    // refcount tells us how many live uses each unique material has —
    // the foundation for the shared MaterialId→GPU-material table
    // (ECS_DESIGN.md §14, material dedup steps 2–3).
    ecs::MaterialDesc      desc_;

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
        // 8-bone skinning debug: JOINTS_1/WEIGHTS_1 streams present
        // (implies has_skin_set_0; selects the _SKIN8 shader permutations).
        uint32_t                has_skin_set_1 : 1;
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
    // CPU copy of the last joint matrices uploaded to joints_buffer_
    // (updateJoints).  Consumed by the RT-shadow skeleton path, which
    // CPU-skins the character into world space each frame.
    std::vector<glm::mat4>  joint_matrices_cpu_;
};

// ── RT-shadow skeleton source ────────────────────────────────────────
// Bind-pose snapshot of a skinned character, captured at .rwchar load
// time (the auto-rig path keeps all of this on the CPU in MeshData).
// ClusterRenderer::updateRtSkeletons CPU-skins it into world space each
// frame so both RT shadow modes (software BVH loop + hardware BLAS) can
// treat the character as a shadow caster.  Topology is static; only
// positions deform.
struct RtSkinSource {
    std::vector<glm::vec3>    positions;       // bind pose
    std::vector<glm::u16vec4> joints;          // 4 influences
    std::vector<glm::vec4>    weights;
    std::vector<glm::u16vec4> joints1;         // optional set 1 (8-bone)
    std::vector<glm::vec4>    weights1;
    std::vector<uint32_t>     indices;         // triangle list
};

struct MeshInfo {
    std::vector<PrimitiveInfo>  primitives_;
    std::shared_ptr<std::vector<glm::vec3>> vertex_position_;
    glm::vec3                   bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    // See PrimitiveInfo::bbox_max_ comment — ::min() is the smallest
    // positive normalized float, NOT the most negative.  Use ::lowest().
    glm::vec3                   bbox_max_ = glm::vec3(std::numeric_limits<float>::lowest());

    // ── Instance-expanded bounds (EXT_mesh_gpu_instancing) ───────────
    // bbox_min_/bbox_max_ above bound the mesh's VERTEX DATA.  For a
    // GPU-instanced mesh that is the wrong box to cull against: the
    // vertices are one tree modelled around the origin, while the
    // thousands of copies the draw actually puts on screen live wherever
    // the instance buffer says.  Culling the origin-sized box makes the
    // WHOLE species blink out the moment the world origin leaves the
    // frustum — which is exactly the "trees appear and disappear when I
    // move the camera" symptom, and why it took a camera *move* rather
    // than a camera *distance* to trigger it.
    //
    // bakeInstanceTransforms fills these with the union of the vertex
    // bbox transformed by every baked instance, in the same space
    // model_mat expects (i.e. post-instance, pre-node), so the existing
    // `model_mat * local_center` cull math needs no other change.
    //
    // has_inst_bbox_ stays false for every mesh in every file that does
    // NOT carry the extension, and cullBboxMin/Max then return the
    // vertex bbox byte-for-byte — so this is a strict no-op for all
    // existing assets.
    glm::vec3                   inst_bbox_min_ = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3                   inst_bbox_max_ = glm::vec3(std::numeric_limits<float>::lowest());
    bool                        has_inst_bbox_ = false;

    // The box any CULL or BOUNDS query should use.  Prefer these over
    // bbox_min_/bbox_max_ anywhere the question is "where on screen does
    // this mesh end up"; use the raw pair only when the question is
    // genuinely about the vertex data itself.
    glm::vec3 cullBboxMin() const {
        return has_inst_bbox_ ? inst_bbox_min_ : bbox_min_;
    }
    glm::vec3 cullBboxMax() const {
        return has_inst_bbox_ ? inst_bbox_max_ : bbox_max_;
    }

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

// One per-instance world transform, byte-identical to glsl::Instance-
// DataInfo (see shaders/global_definition.glsl.h) so a vector of these
// can be memcpy'd straight into instance_buffer_.
//
// 48 B, not 64: the old fourth vec4 (mat_pos_scale) and the three .w
// pads were dead weight — every consumer read only the .xyz of the
// basis columns and the .xyz of the translation, so at 17 M baked
// instances the four unused lanes alone cost ~260 MiB of VRAM.  The
// translation now rides in the .w lane of each basis column instead:
//     mat_rot_0 = (basis[0], T.x)
//     mat_rot_1 = (basis[1], T.y)
//     mat_rot_2 = (basis[2], T.z)
//
// Column convention is base.vert's, NOT a glm::mat4's rows:
//     position_ws = mat3(mat_rot_0.xyz, mat_rot_1.xyz, mat_rot_2.xyz)
//                 * position_ls
//                 + vec3(mat_rot_0.w, mat_rot_1.w, mat_rot_2.w)
// GLSL's mat3x3(a,b,c) builds from COLUMNS, so mat_rot_i.xyz is column
// i of the rotation*scale basis (scale premultiplied at bake time) and
// the three .w lanes together are the translation.  That matches glm's
// column-major mat4 exactly: mat_rot_i.xyz == vec3(M[i]) for
// M = translate(T) * mat4_cast(R) * scale(S).  Get this wrong and every
// instance renders transposed — sheared and mirrored, not obviously
// "rotated", which is a miserable thing to debug.
struct BakedInstanceXform {
    glm::vec4                   mat_rot_0{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4                   mat_rot_1{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4                   mat_rot_2{0.0f, 0.0f, 1.0f, 0.0f};
};

struct NodeInfo {
    std::string                 name_;
    int32_t                     parent_idx_ = -1;
    std::vector<int32_t>        child_idx_;

    int32_t                     mesh_idx_ = -1;
    int32_t                     skin_idx_ = -1;

    // ── EXT_mesh_gpu_instancing ──────────────────────────────────────
    // Accessor indices read straight off node.extensions in setupNode.
    // -1 = the node did not carry the extension.  ROTATION and SCALE are
    // optional per the extension spec (a node may instance translations
    // only), TRANSLATION in practice is always present; bakeInstance-
    // Transforms substitutes identity for whichever is missing.
    int32_t                     inst_translation_acc_ = -1;
    int32_t                     inst_rotation_acc_ = -1;
    int32_t                     inst_scale_acc_ = -1;
    // Filled by bakeInstanceTransforms: this node's slice of
    // DrawableData::baked_instances_.  inst_count_ == 0 means "not
    // instanced" and the drawable's single identity slot 0 is used.
    uint32_t                    inst_offset_ = 0;
    uint32_t                    inst_count_ = 0;
    // ── THIS NODE'S INDIRECT COMMANDS ────────────────────────────────
    // Byte offset of the first of this node's draw commands, one per
    // primitive of its mesh, contiguous.  -1 = this drawable still uses
    // the per-primitive commands in PrimitiveInfo.
    //
    // A command has to belong to the NODE, not the primitive, or two
    // nodes cannot share a mesh: first_instance/instance_count are a
    // property of the node's slice of the baked transform table, and a
    // command living on the primitive can only hold one node's slice.
    // That is what forced one mesh per instanced node, and therefore one
    // GPU upload of the same geometry per node -- 817 486 tree nodes
    // over 803 distinct meshes on a full world.  With the command here,
    // the mesh is just geometry and every node that draws it supplies
    // its own instance range.
    int32_t                     indirect_cmd_ofs_ = -1;

    // ── Plant LOD band (parsed once at load from name_) ───────────────
    // terrain_pcg.py names every tree node
    //     tree_<species>_lodtile_<i>_<j>_<tile>_<near>_<far>
    // with the last five fields whole metres, and each (tile, band,
    // species) is its own node.  parsePlantLodBands turns that into the
    // tile's XZ RECTANGLE and the band's distance range; selectPlant-
    // LodBands then draws, for each tile, the first band whose far_
    // exceeds the distance from the eye to that rectangle.
    //
    // The rectangle comes from the NAME and not from the node's instance
    // bounds on purpose.  A tile holds a different subset of species in
    // each band, so its instance bounds differ per band, and two bands
    // that measure different distances to the same ground disagree about
    // which of them owns it — the tile is then drawn twice or not at all,
    // and the hole moves with the camera.  Named rectangles are identical
    // across the bands by construction, so the partition is exact.
    //
    // lod_tile_m_ == 0 means "this node is not part of a LOD chain" and
    // the node draws unconditionally, which is every node in every other
    // file the engine loads.
    float                       lod_tile_m_ = 0.0f;
    float                       lod_near_m_ = 0.0f;
    float                       lod_far_m_ = 0.0f;
    // Which rung of its OWN chain this node is, 0 = nearest, and which
    // chain that is.  An ordinal, not a distance, so it survives any
    // runtime remap of the metres.  -1 = not a LOD node.
    //
    // The chain matters because one file carries several INDEPENDENT
    // ones: terrain_pcg.py merges bushes and rocks into the tree stream
    // to share meshes, so new-world_pcg_trees.glb holds _TREE_LOD
    // (0/50/90/140/200/2600), _SHRUB_LOD (0/25/40/55/75/260) and
    // _ROCK_LOD (0/50/100/170/260/700) at once.  Ordering every authored
    // (near, far) pair in the file by `near` would interleave the three
    // into one nonsense ladder — and applying a tree schedule to grass
    // would push 4 m tuft clumps out to a 600 m card edge.  The chain is
    // therefore keyed by the node name's leading token ("tree", "bush",
    // "rock", "ground", ...), which is exactly the category the
    // generator names them by.
    int                         lod_band_idx_ = -1;
    int                         lod_cat_idx_ = -1;
    // The tile rectangle in WORLD space, cached by selectPlantLodBands
    // for the drawable's current instance world (see DrawableData::
    // lod_world_from_).  Recomputed only when that transform changes,
    // so the per-frame work per node is two subtractions and a compare.
    float                       lod_wx0_ = 0.0f, lod_wx1_ = 0.0f;
    float                       lod_wz0_ = 0.0f, lod_wz1_ = 0.0f;
    // Tile origin in the file's own space, kept so the world rectangle
    // can be rebuilt if the wrapper moves.
    float                       lod_lx0_ = 0.0f, lod_lz0_ = 0.0f;
    // ── Per-instance LOD (dense ground cover) ─────────────────────
    // Category "ground": its band widths (20-40 m) are far smaller
    // than the 256 m export tile, so the per-NODE rect test flipped a
    // whole tile's cover at once — visible square LOD seams.  Flagged
    // nodes resolve the band PER INSTANCE in the vertex shader
    // (near/far packed into ModelParams::model_params_pad0 by
    // drawNodeMesh); the CPU pass keeps only a conservative
    // node-level cull.  Set by parsePlantLodBands.
    uint8_t                     lod_per_instance_ = 0;
    // ── Building interior (forward-path sky occlusion) ────────────
    // 1 for geometry the generator marked as INSIDE a building: the
    // house "_int_lodtile_" band (interior shell, door leaves) and the
    // "_pgate_" mode-0 room props (furniture, which exists only when
    // the eye is at the house).  drawNodeMesh forwards it to base.frag
    // as MODEL_FLAG_INTERIOR, which scales the environment term down —
    // the forward path has no ray tracer, so without this a room gets
    // full open-sky irradiance and renders as bright as the lawn.
    // Set by parsePlantLodBands alongside the LOD/gate parse.
    uint8_t                     interior_ = 0;
    // ── Proximity gate (house interiors / door leaves) ────────────
    // Parsed from a "_pgate_<x_dm>_<z_dm>_<r_dm>_<mode>" marker in the
    // node name (decimetre ints).  mode 0: node draws ONLY when the eye
    // is within r of (x, z) — house interiors, open door leaves.
    // mode 1: node draws only when the eye is FARTHER than r — closed
    // door leaves, which "open" (swap for the mode-0 leaf) as the
    // player walks up.  gate_r_ == 0 means no gate.
    float                       gate_x_ = 0.0f, gate_z_ = 0.0f;
    float                       gate_r_ = 0.0f;
    int                         gate_mode_ = 0;

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
    // "Only render this node" filter — STAGED PER DRAW CALL by
    // DrawableObject::draw() from the wrapper's sub-object selection (same
    // shared-data staging pattern as m_current_instance_world_, so several
    // wrappers can share one DrawableData and each render a different
    // node).  -1 = draw every node (default).  drawNodes() skips the mesh
    // draw for any other node but still recurses children, so a filtered
    // node deep in the hierarchy is reached regardless of its parents.
    int32_t m_only_render_node_ = -1;
    // Same-frame dedup stamp for DrawableData::update() — many wrappers can
    // share one DrawableData (placed sub-objects); the hierarchy/animation/
    // joint refresh only needs to run once per frame.
    float m_last_update_time_ = -1.0e30f;
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
    // ── Ground-clutter distance fade (DrawMode::kDecal only) ─────────
    // Pushed to base.frag as ModelParams::clutter_fade_{start,end}_m and
    // used on the CPU by drawMesh to reject whole per-tile MeshInfos
    // beyond the end distance.  0/0 = no fade, which is what every
    // drawable except the <map>_pcg_clutter.glb import carries — the
    // road ribbon shares the decal pipeline and must NOT dissolve with
    // range.
    //
    // STAGING SLOT, not the authority.  DrawableData is shared between
    // every wrapper that loaded the same file (see the dedup cache in
    // createAsync), so a per-drawable property cannot be stored here and
    // survive; the owning wrapper copies its own values in at the top of
    // DrawableObject::draw, exactly as m_current_instance_world_ and
    // m_only_render_node_ are staged.  Set via DrawableObject::
    // setClutterFade.
    float m_clutter_fade_start_m_ = 0.0f;
    float m_clutter_fade_end_m_ = 0.0f;
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
    // Bind-pose CPU snapshot for the RT-shadow skeleton path — non-null
    // only for skinned characters loaded through the .rwchar path.
    std::shared_ptr<RtSkinSource> rt_skin_source_;
    std::vector<renderer::BufferInfo>     buffers_;
    std::vector<BufferView>     buffer_views_;

    std::vector<renderer::TextureInfo>    textures_;
    std::vector<MaterialInfo>   materials_;

    uint32_t                    num_prims_ = 0;
    renderer::BufferInfo        indirect_draw_cmd_;
    renderer::BufferInfo        instance_buffer_;

    // ── Baked GPU instancing (EXT_mesh_gpu_instancing) ───────────────
    // Non-empty only for assets that carried the extension — today the
    // PCG tree scatter, "<map>_pcg_trees.glb", where 30 species meshes
    // stand in for several thousand drawn trees.
    //
    // The engine's DEFAULT instancing is a crowd system: every primitive
    // of every drawable gets the same instance_count (kNumDrawableInstance,
    // which is 1) and the same transform, rewritten every frame by
    // update_instance_buffer.comp from the shared game_objects_buffer_.
    // Baked instancing is the opposite shape — a fixed, per-node,
    // load-time table that nothing rewrites.  The two cannot coexist on
    // one drawable, which is what has_baked_instances_ arbitrates: when
    // it is set, BOTH per-frame compute passes (updateInstanceBuffer and
    // updateIndirectDrawBuffer) skip this drawable entirely, because
    // either one would overwrite the baked table — the first with the
    // game-object transform, the second by stamping kNumDrawableInstance
    // over every primitive's instance_count.
    //
    // Slot 0 is ALWAYS a reserved identity transform, so a mixed file
    // (some nodes instanced, some not) still renders its plain nodes at
    // first_instance=0, count=1 — i.e. exactly as before.
    std::vector<BakedInstanceXform> baked_instances_;
    // mesh index -> (first_instance, instance_count), first writer wins.
    // Empty when nothing is instanced.  REPORTING ONLY: the indirect
    // draw fill used to walk meshes and read this, which made a mesh
    // shared by two instanced nodes ambiguous and forced the writer to
    // emit one mesh entry per instanced node.  An instanced drawable now
    // gets one command block per NODE (NodeInfo::indirect_cmd_ofs_), so
    // sharing is expected and this map only feeds the load summary.
    std::unordered_map<int32_t, std::pair<uint32_t, uint32_t>>
                                mesh_instance_range_;
    bool                        has_baked_instances_ = false;
    // Last geometry-LOD override the indirect commands were written
    // for; updateIndirectDrawBuffer rewrites per-prim index counts
    // when DrawableObject::forced_geo_lod_ moves away from this.
    int32_t                     applied_geo_lod_ = 0;

    // ── Plant LOD band selection ─────────────────────────────────────
    // NOT a staging slot, unlike m_clutter_fade_*_m_ above, and the
    // difference is worth stating because the two look alike.  A fade
    // distance is a per-WRAPPER property, so a shared DrawableData can
    // only hold one wrapper's copy at a time.  A band selection is a pure
    // function of (the node table, the instance world, the eye), and the
    // memo below is keyed by both inputs that can change — so two
    // wrappers of the same file at different positions still each get
    // their own correct answer; they would merely recompute alternately
    // rather than share.  (Today there is one tree wrapper, placed at
    // identity, so they do share.)  A drawable is typically drawn five
    // times a frame
    // (forward + four cascades) and the memo is what makes the other
    // four free — and, more importantly, what makes them agree: the
    // bands partition distance, so a cascade selecting differently from
    // the forward pass would shadow a tree that is not there.
    //
    // 1 = draw this node this frame.  Sized to nodes_ and left empty on
    // every file that carries no _lodtile_ names, i.e. all but the tree
    // GLB, which is also what has_plant_lod_ gates on.
    std::vector<uint8_t>        lod_node_visible_;
    // ── LOD cross-fade weight, parallel to lod_node_visible_ ─────────
    // 1 = fully this band's tile; 0 = not drawn; in between = inside a
    // band TRANSITION, where this tile's mesh dissolves out (or in) via
    // a screen-door threshold in base.frag instead of vanishing whole.
    // Signed to say WHICH side of the transition the node is on, so the
    // two neighbouring bands can use exactly complementary dither tests
    // and never both-draw or both-drop a pixel:
    //   > 0  fading OUT (or steady at 1.0)  → draw where  w >  ign
    //   < 0  fading IN,  magnitude is alpha → draw where |w| > 1-ign
    // A hard per-tile swap is what made whole 128 m squares of trees and
    // grass change character in one frame — the tile-shaped LOD pop.
    std::vector<float>          lod_node_fade_;
    // Parallel again: the ORIGINAL hard band test, `d in [near, far)` —
    // exactly one band owns each tile, no transition zone.  DEPTH-ONLY
    // passes (shadow / CSM cascades) use this instead of the cross-fade,
    // because those pipelines do not run the dissolve: picking by
    // "whichever side of the transition is winning" made a tile's SHADOW
    // pop on and off as the camera drifted across the halfway point,
    // which is shadow flicker that the pre-cross-fade engine never had.
    // Stable ownership costs nothing and is invisible in soft cascades.
    std::vector<uint8_t>        lod_node_owner_;
    bool                        has_plant_lod_ = false;
    // Memo keys.  lod_eye_valid_ false means "no eye has been published
    // yet"; see selectPlantLodBands for what is drawn then.
    glm::mat4                   lod_world_from_ = glm::mat4(0.0f);
    glm::vec3                   lod_eye_ = glm::vec3(0.0f);
    bool                        lod_eye_valid_ = false;
    // Largest far_ over all LOD nodes, i.e. which band is the outermost.
    // Only read by the no-eye-yet fallback, which draws that band alone.
    float                       lod_far_max_ = 0.0f;
    // Per chain: its category name, and the (near, far) pairs it
    // authored in rung order.  Kept so a runtime override can be
    // reverted, and so a chain with more rungs than its override table
    // still has something sane for the rungs the table does not cover.
    std::vector<std::string>              lod_cat_names_;
    std::vector<std::vector<glm::vec2>>   lod_band_authored_;
    // Value of plantLodBandGeneration() the node distances were last
    // written for.  Bumped by setPlantLodBands(), so the per-frame pass
    // rewrites 690k node distances only when the table actually changed.
    uint32_t                    lod_band_gen_ = 0xFFFFFFFFu;
    // Diagnostics for the per-second print: nodes kept by the last
    // selection, and how many tiles fell in each band.
    uint32_t                    lod_nodes_drawn_ = 0;

    // ── Flattened mesh-node list (draw-path fast lane) ────────────────
    // DFS-ordered indices of every node with mesh_idx_ >= 0, over all
    // roots of the default scene.  Built lazily on first draw (node
    // topology is immutable once ready_) and iterated INSTEAD of the
    // recursive whole-tree walk: procedurally generated files carry
    // hundreds of thousands of empty grouping nodes around a few hundred
    // mesh nodes (587k nodes / 180 meshes was measured for one placed
    // house layout), and the recursive visit of every one of them —
    // ~1.5M nodes per pass across the scene — was the dominant CPU cost
    // of command recording (~35 ms/frame in the forward pass alone).
    // Node transforms are pre-baked into node.cached_matrix_, so the
    // draw never needed the parent chain — only the mesh nodes.
    // The per-wrapper sub-object filter path (m_only_render_node_) keeps
    // the old targeted recursion and never touches this list.
    std::vector<int32_t>        mesh_node_flat_;
    // Parallel to mesh_node_flat_: each node's cull sphere in DRAWABLE
    // space (cached_matrix_ pre-applied) — xyz = center, w = radius.
    // Lets the draw loop frustum/distance-reject a node with one
    // mat4×vec4 + a few dots BEFORE staging ModelParams (a full
    // mat4×mat4) and calling into drawMesh, which is what the ~50k-node
    // ground-clutter file needs: nearly all of its tiles are past the
    // fade distance every frame, and rejecting them used to cost more
    // than drawing the survivors.  Conservative radii (per-axis max
    // scale) so the fine tests in drawMesh never disagree toward
    // dropping something this test kept.
    std::vector<glm::vec4>      mesh_node_sphere_;
    bool                        mesh_node_flat_built_ = false;

    // ── Tile grid over the flat mesh-node list ────────────────────────
    // The flat list is bucketed into a world-aligned 2D grid sized to
    // match the 128 m terrain tiles and REORDERED so every cell's nodes
    // are contiguous.  Each cell keeps its node range and the union AABB
    // of its members' cull spheres, which lets a pass reject a whole
    // tile's worth of nodes with one box test instead of one test per
    // node.  Because instanced nodes' cull bbox already spans all their
    // instances, a cell's AABB automatically covers the tile AND
    // everything placed on it.
    // Built once alongside mesh_node_flat_ / mesh_node_sphere_, from the
    // same drawable-space spheres, so it is only meaningful for the same
    // static drawables those are (no skins, no animations).
    // Empty means "no useful partition" (a single cell, or too few
    // nodes to be worth it) and the draw path falls back to the plain
    // flat loop.
    struct NodeCell {
        glm::vec3 bmin;    // drawable space, spheres expanded by radius
        glm::vec3 bmax;
        float     maxr;    // largest member sphere radius, drawable space
        uint32_t  first;   // index into mesh_node_flat_
        uint32_t  count;
    };
    std::vector<NodeCell>       mesh_node_cells_;

    // ── LOD SURVIVOR LISTS ────────────────────────────────────────────
    // The plant-LOD band table rejects ~99.4% of the mesh nodes in a
    // procedural world (measured: 2,215,770 of 2,228,507).  Testing it
    // per node per pass is one byte load, but it is one byte load at a
    // scattered address 2.2 million times, five passes a frame — ~13 ns
    // each, ~29 ms per pass, and it was the whole of the CSM and forward
    // recording cost.
    //
    // The table only changes when selectPlantLodBands actually
    // recomputes (the eye or the instance world moved).  So compact it
    // ONCE per change into two lists of surviving mesh_node_flat_
    // indices — one per predicate, because the depth passes use the
    // single-owner test and the colour passes use the cross-fade
    // visibility test — and let every pass in between iterate ~12k
    // entries instead of 2.2M.
    //
    // lod_tables_version_ is bumped by selectPlantLodBands on every
    // recompute; lod_pass_version_ / lod_pass_flat_n_ record what the
    // lists were built from, so a stale pair rebuilds itself.
    // drawNodeMesh still applies the band test itself, so a list that is
    // somehow stale can only cost work, never draw something hidden.
    uint64_t                    lod_tables_version_ = 0;
    uint64_t                    lod_pass_version_ = ~uint64_t(0);
    size_t                      lod_pass_flat_n_ = 0;
    std::vector<uint32_t>       lod_pass_fwd_;    // colour-pass survivors
    std::vector<uint32_t>       lod_pass_depth_;  // depth-pass survivors

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

    // Read-only view of the loaded materials (incl. the ECS MaterialDesc
    // captured at load).  Only meaningful once ready_ is set — callers must
    // gate on DrawableObject::isReady() first (async phase-2/3 may still be
    // populating the vector on another thread before that).
    const std::vector<MaterialInfo>& materials() const { return materials_; }

    // ── RT-shadow skeleton access ─────────────────────────────────────
    // Source is non-null only for .rwchar-loaded skinned characters.
    // The world matrix is the cached matrix of the node that OWNS the
    // skin — controller/animation placement flows through the node
    // hierarchy, so this is the full world placement the raster path
    // uses.  Joint matrices are the CPU copies cached by updateJoints.
    const std::shared_ptr<RtSkinSource>& getRtSkinSource() const {
        return rt_skin_source_;
    }
    glm::mat4 getRtSkinWorldMatrix() const {
        for (const auto& n : nodes_) {
            if (n.skin_idx_ > -1) return n.getCachedMatrix();
        }
        return glm::mat4(1.0f);
    }
    const std::vector<glm::mat4>* getRtSkinJointMatrices() const {
        return skins_.empty() ? nullptr : &skins_[0].joint_matrices_cpu_;
    }

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

// ── RUNTIME PLANT LOD BANDS ──────────────────────────────────────────
// The generator bakes two different things and only one of them has to
// stay baked.  The MESHES must be — a decimated tree and its impostor
// card are minutes of offline work each, not a per-frame operation.  The
// SWITCH DISTANCES were baked only because they happened to ride along
// in the node name (`_lodtile_<i>_<j>_<tile>_<near>_<far>`), and they are
// just numbers: changing them needs no new geometry, only a different
// comparison.  Anything tunable by eye belongs at runtime, and a band
// schedule is the definition of tunable by eye.
//
// setPlantLodBands({50, 150, 300, 600, 1200, 2600}) sets the rung EDGES:
// rung 0 draws [0, 50), rung 1 [50, 150), and so on, with the last rung
// running to the final edge.  A file with more rungs than the table has
// edges keeps its authored metres for the rungs past the end, so a short
// table degrades instead of blanking the far field.  Pass {} to go back
// to exactly what the generator authored.
//
// This works on an ALREADY-BAKED world: it re-labels distances the file
// already has meshes for.  What it cannot do is invent a rung — pushing
// the impostor edge out on a 5-rung bake means the rung below it covers
// more ground with the detail it was built at, not with new detail.
// `category` is the node name's leading token — "tree", "bush", "rock",
// "ground", and whatever else a bake introduces.  Pass "" to set the
// fallback used by any category without a table of its own.  Each
// category is independent, which is the point: grass, shrubs, rocks and
// canopy stop being resolvable at wildly different distances, and one
// shared schedule is wrong for at least three of them.
void  setPlantLodBands(const std::string& category,
                       const std::vector<float>& edges_m);
const std::vector<float>& plantLodBands(const std::string& category);

// ── INTERIOR PROP VISIBILITY ─────────────────────────────────────────
// House contents ride _pgate_ nodes (mode 0: draw only INSIDE radius r
// of the gate centre; mode 1: only outside — the closed door leaf that
// swaps for the open one).  r is baked per house as that house's own
// radius, which is a few metres.  This raises every gate radius to at
// least `m`, so furniture is visible from further out.
//
// It must apply to BOTH modes or the pair stops partitioning: raising
// only the open leaf's radius would leave a ring where the open and
// closed leaves both draw.
void  setInteriorGateRadiusM(float m);
float interiorGateRadiusM();
// Bumped on every set; drawables compare it against lod_band_gen_ so the
// rewrite over every node happens on change, not every frame.
uint32_t plantLodBandGeneration();

class DrawableObject {
    enum {
        kMaxNumObjects = 10240
    };
    std::shared_ptr<DrawableData>   object_;
    glm::mat4                   location_;

    // Set (main thread, phase 3) when this wrapper's async load FAILED
    // permanently — phase 2 produced no DrawableData (unreadable /
    // corrupt source file).  object_ stays null so isReady() is false
    // forever; loadFailed() lets waiters distinguish "still streaming"
    // from "never coming".  See DrawableObject::createAsync.
    std::atomic<bool>           load_failed_{false};

    // See setUseNodeTransformOnly() in the public section for what this
    // flag does and why the player controller has to set it.
    // NOTE: the canonical storage lives on DrawableData as
    // m_use_node_transform_only_ so the static drawMesh helpers (which
    // only receive DrawableData*) can read it for the skinned-character
    // frustum-cull bypass.  This field on the outer wrapper is kept as
    // a cache to make getUseNodeTransformOnly() valid even before
    // object_ is populated; the setter writes both.
    bool                        use_node_transform_only_ = false;
    bool                        external_animation_ = false;

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

    // Per-wrapper ground-clutter distance fade, in metres.  0/0 = no
    // fade (the default, and what every drawable except the clutter glb
    // keeps).  Lives here rather than in the shared DrawableData for the
    // same reason instance_root_* does — see setClutterFade.
    float                       clutter_fade_start_m_ = 0.0f;
    float                       clutter_fade_end_m_ = 0.0f;

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

    // Transient per-frame ECS frustum-cull hint — see setEcsCulledHint()
    // in the public section.  Only honoured by draw() when the mode is
    // kForward; every other pass ignores it.  The application clears it
    // right after the main forward pass each frame.
    bool                        ecs_culled_hint_ = false;

    // PlayerController (external code) owns this wrapper's WORLD
    // placement.  While set, the ECS RenderSystem must NOT push the
    // scene-authored transform into the instance root each frame — the
    // controller drives the rig through the root node and the two would
    // double-transform / fight.  Set when the editor's scene player is
    // adopted for play mode; cleared on scene unload.
    bool                        controller_driven_ = false;

    // ── Per-wrapper "render only ONE sub-object" filter ────────────────
    // Set via setOnlyRenderSubObject(ordinal) where `ordinal` is the k-th
    // renderable (mesh) node — the same enumeration the Outliner children
    // and the Content Browser .rwobj files use.  The ordinal is resolved
    // to an actual nodes_ index lazily on the first draw after the async
    // load completes (the node table doesn't exist before that), then
    // staged into DrawableData::m_only_render_node_ each draw call so
    // dedup-shared DrawableData instances can each show different nodes.
    // -1 = no filter (whole drawable, default).
    int32_t                     only_render_ordinal_ = -1;
    int32_t                     only_render_node_    = -1;  // resolved index

    // static members.
    static uint32_t max_alloc_game_objects_in_buffer;

public:
    // ── Runtime geometry-LOD override ─────────────────────────────────
    // Which baked LOD level the CPU draw paths bind: 0 (default) = full
    // detail; 1..c_num_lods = the progressively decimated levels baked
    // into v6 .rwgeo files (and built at load for FBX).  Clamped per
    // primitive to the slots it actually has, so assets without baked
    // LODs simply keep rendering full detail.  Tweaked from the editor
    // UI ("Forced geometry LOD").
    static int32_t forced_geo_lod_;

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
    // Ground-decal forward pipeline.  Same vertex layout, same layout
    // object and the same forward renderbuffer formats as
    // drawable_pipeline_list_; what differs is the fragment permutation
    // (base_frag*_DECAL.spv, which fades itself out by the distance
    // between its own depth and the scene depth already in the buffer)
    // and the fixed-function state (alpha blend ON, depth write OFF,
    // depth test still ON so houses occlude the road).  Keyed by the
    // SAME PrimitiveInfo hash as the forward list — the two never share
    // a map, so there is no collision to disambiguate, and a primitive
    // drawn both ways gets one entry in each.
    // Selected by DrawMode::kDecal; populated only for objects the host
    // registered via ObjectSceneView::addDecalObject.
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_decal_pipeline_list_;
    // ── Deferred-decal pipelines (DrawMode::kDecalGBuffer) ────────────
    // base_vert_* + base_frag_*_DECAL_GBUF against the 4-RT cluster
    // G-buffer formats.  Same depth state as the plain G-buffer list
    // (LESS_OR_EQUAL, writes OFF); what differs is the blend state —
    // attachment 0 blends the decal albedo over the ground albedo with
    // the textbook "over" factors, its ALPHA channel is ZERO/ONE so the
    // deferred sentinel the terrain stamped survives, and attachments
    // 1..3 are write-masked off entirely so the ground keeps its normal,
    // geometric normal and velocity.  Populated at the same sites and
    // under the same skip rules as drawable_gbuffer_pipeline_list_
    // (needs setGbufferRenderbufferFormats(), non-skinned, has material).
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_decal_gbuffer_pipeline_list_;
    // ── G-buffer pipelines (DrawMode::kGBuffer) ───────────────────────
    // base_vert_* + base_frag_*_GBUF against the 4-RT cluster G-buffer
    // formats: depth test LESS_OR_EQUAL, depth writes OFF — the pass
    // re-rasterises the drawables over the depth the forward pass
    // already stamped, adding material attributes only where they are
    // the visible surface (same contract as terrain's tile_gbuf pass).
    // Populated lazily at the same three sites as the forward list, but
    // only for non-skinned primitives WITH a material — skinned
    // characters keep forward shading (static-world velocity would be
    // wrong for them), and _NOMTL has no _GBUF permutation compiled.
    // Requires setGbufferRenderbufferFormats() to have been called; the
    // application does so right after it builds the G-buffer formats.
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_gbuffer_pipeline_list_;
    static renderer::PipelineRenderbufferFormats gbuffer_renderbuffer_formats_;
    static bool gbuffer_formats_valid_;
    // ── Glass-attribute pipelines (DrawMode::kGlassAttr) ──────────────
    // base_vert_* + base_frag_*_GLASS against the two glass targets:
    // depth test LESS_OR_EQUAL vs the forward pass's depth, writes off.
    // Same population sites and skip rules as the G-buffer list.
    static std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> drawable_glass_pipeline_list_;
    static renderer::PipelineRenderbufferFormats glass_renderbuffer_formats_;
    static bool glass_formats_valid_;
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

    // True when the async load for this wrapper failed permanently
    // (isReady() will never become true).  Systems that wait for
    // "every placed object ready" — e.g. the placed→cluster merge in
    // syncPlacedObjectsToClusters — must treat a failed wrapper as
    // skippable instead of stalling forever behind it.
    bool loadFailed() const {
        return load_failed_.load(std::memory_order_acquire);
    }

    // RT-shadow skeleton forwarders (see DrawableData) — only valid when
    // isReady().  Source is non-null only for .rwchar skinned characters.
    const std::shared_ptr<RtSkinSource>& getRtSkinSource() const {
        static const std::shared_ptr<RtSkinSource> s_null;
        return isReady() ? object_->getRtSkinSource() : s_null;
    }
    glm::mat4 getRtSkinWorldMatrix() const {
        return isReady() ? object_->getRtSkinWorldMatrix() : glm::mat4(1.0f);
    }
    const std::vector<glm::mat4>* getRtSkinJointMatrices() const {
        return isReady() ? object_->getRtSkinJointMatrices() : nullptr;
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

    // ── ECS frustum-cull hint (forward pass only) ────────────────────
    // Set per frame by the application from the ECS CullingSystem right
    // before the main forward pass and cleared right after it, so probe /
    // shadow / CSM passes are never affected.  When true, draw() early-
    // outs for DrawMode::kForward only — a coarse object-level early-out
    // on top of the existing per-mesh frustum cull.  Deliberately
    // DISTINCT from visible_ (a persistent user/tool gate) and from the
    // cluster renderer's forward-hide (which also rides visible_): this
    // flag is transient and owned exclusively by the ECS cull pass.
    void setEcsCulledHint(bool v) { ecs_culled_hint_ = v; }
    bool isEcsCulledHint() const  { return ecs_culled_hint_; }

    // Loaded materials (with their load-time ECS MaterialDescs), or nullptr
    // until the async load has finalized (isReady()).  Used by the app's
    // ECS material-dedup pass to intern each sub-material.
    const std::vector<MaterialInfo>* getMaterials() const {
        return (object_ && isReady()) ? &object_->materials() : nullptr;
    }

    // External-controller transform ownership — see controller_driven_.
    void setControllerDriven(bool v) { controller_driven_ = v; }
    bool isControllerDriven() const  { return controller_driven_; }

    // Restrict rendering to the `ordinal`-th renderable sub-object (the
    // k-th node with a mesh — the same order the Outliner children and the
    // Content Browser's exploded .rwobj assets use).  Pass -1 to clear.
    // Safe to call on a not-yet-loaded shell: the ordinal is resolved to a
    // node index lazily on the first draw after the load completes.
    void setOnlyRenderSubObject(int32_t ordinal) {
        only_render_ordinal_ = ordinal;
        only_render_node_    = -1;   // re-resolve on next draw
    }
    int32_t onlyRenderSubObject() const { return only_render_ordinal_; }

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

    // ── ECS-driven animation ──────────────────────────────────────────────
    // When true, DrawableObject::update() skips the imported glTF channel
    // evaluation (like use_node_transform_only_) but WITHOUT the identity-
    // instance side effect — so a placed/instanced object keeps its world
    // placement while the ECS AnimationSystem owns its node TRS (written via
    // setNodeLocalTRS before update()). Default false = unchanged behaviour.
    void setExternalAnimation(bool v) { external_animation_ = v; }
    bool getExternalAnimation() const { return external_animation_; }

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

    // ── Ground-clutter distance fade ─────────────────────────────────
    // start_m: last distance at which the drawable is fully opaque.
    // end_m:   distance at which it has completely dissolved; whole
    //          MeshInfos whose bounding sphere lies entirely beyond it
    //          are skipped on the CPU before any draw is recorded.
    // Pass (0, 0) to disable — that is the default, and it is what the
    // road decal keeps.  Only meaningful for drawables rendered through
    // DrawMode::kDecal; base.frag reads the values in its DECAL branch
    // and no other permutation does.
    //
    // Stored on the WRAPPER, deliberately: this is callable the instant
    // createAsync returns, long before object_ exists, and it must not be
    // clobbered by a sibling wrapper that deduped onto the same
    // DrawableData.  draw() copies it down each frame.
    void setClutterFade(float start_m, float end_m) {
        clutter_fade_start_m_ = start_m;
        clutter_fade_end_m_ = end_m;
    }
    float getClutterFadeStart() const { return clutter_fade_start_m_; }
    float getClutterFadeEnd() const { return clutter_fade_end_m_; }

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

    // ── ECS animation seam ────────────────────────────────────────────────
    // The ECS AnimationSystem samples clips (read from getDrawableData()) and
    // writes the resulting per-node local TRS back through setNodeLocalTRS().
    // Pair with skipping the imported channel evaluation (DrawableData::update
    // skip_animations) so the ECS pose is not overwritten the same frame.
    // All index-based, bounds- and ready-checked; no-ops on a not-yet-loaded
    // shell. No behaviour change unless a caller actually drives them.
    uint32_t getNodeCount() const {
        return object_ ? static_cast<uint32_t>(object_->nodes_.size()) : 0u;
    }
    bool getNodeLocalTRS(uint32_t node_idx, glm::vec3& t, glm::quat& r,
                         glm::vec3& s) const {
        if (!object_ || node_idx >= object_->nodes_.size()) return false;
        const auto& n = object_->nodes_[node_idx];
        t = n.translation_; r = n.rotation_; s = n.scale_;
        return true;
    }
    void setNodeLocalTRS(uint32_t node_idx, const glm::vec3& t,
                         const glm::quat& r, const glm::vec3& s) {
        if (!object_ || node_idx >= object_->nodes_.size()) return;
        auto& n = object_->nodes_[node_idx];
        n.translation_ = t; n.rotation_ = r; n.scale_ = s;
    }

    // Model-space AABB of a SKINNED character derived from its joint
    // positions (cached node matrices), padded for the flesh around
    // the bones.  The raw mesh bbox_min_/bbox_max_ are PRE-skin
    // accessor bounds: for rigs whose skeleton is authored away from
    // the mesh node (Mixamo/Blender sibling-armature exports) they
    // land nowhere near the rendered body.  Joint origins follow the
    // same math the vertex shader applies (joint.cached * inv_bind),
    // so this box tracks what's actually on screen.  Returns false
    // for non-skinned or not-ready drawables — callers fall back to
    // the mesh bbox.
    bool getSkinnedModelAabb(glm::vec3& bmin, glm::vec3& bmax) const;
    // True when this drawable carries skinning data (skeleton-driven
    // deformation). Used by the Debug Display to badge Static vs Skeletal.
    bool isSkinned() const { return object_ && !object_->skins_.empty(); }

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
        kGlassAttr,     // Glass/water attribute pass: ONLY Blend/glass
                        // primitives, into the two glass targets the
                        // resolve ray-traces reflection/refraction from.
                        // Uses drawable_glass_pipeline_list_.
        kGBuffer,       // Deferred re-rasterise: material attributes into
                        // the 4-RT cluster G-buffer (depth-gated LEQUAL
                        // against the forward pass's depth, no depth
                        // writes).  deferred_resolve.comp lights the
                        // pixels.  Uses drawable_gbuffer_pipeline_list_.
        kDecal,         // Ground-decal forward pass.  Lit exactly like
                        // kForward — same VS, same material bindings —
                        // but runs AFTER the terrain tiles, with alpha
                        // blend on and depth write off, and uses the
                        // _DECAL fragment permutation which samples the
                        // scene depth copy at SCENE_DEPTH_TEX_INDEX and
                        // dissolves itself wherever the decal mesh
                        // separates from the ground already rendered
                        // there.  That is what makes the road ribbon
                        // blend into the terrain instead of drawing a
                        // hard silhouette edge over it.
                        //
                        // DEFERRED MODE DOES NOT USE THIS — see
                        // kDecalGBuffer below.  kDecal remains the path
                        // for the pure-forward (CSM / no-deferred)
                        // configuration.
        kDecalGBuffer,  // Deferred ground decals: the same decal meshes
                        // re-rasterised into the cluster G-buffer
                        // BEFORE the resolve, blending their albedo
                        // over whatever the terrain tiles / drawables
                        // wrote there.  deferred_resolve.comp then
                        // lights ground-plus-decal as ONE surface, so
                        // the decal inherits the traced shadow, RT AO
                        // and RT GI of the ground it lies on.
                        //
                        // Why this mode exists: drawn forward AFTER the
                        // resolve (kDecal), a decal got shadow = 1.0 in
                        // every RT mode — the app raises
                        // FEATURE_INPUT_SHADOW_DISABLED as soon as an
                        // RT technique arms (the CSM cascades are stale
                        // and must not be sampled), and nothing else
                        // re-lit those pixels.  The result was a fully
                        // lit road ribbon sitting on correctly shadowed
                        // terrain, which reads as a sticker floating
                        // above the ground.
                        //
                        // Uses drawable_decal_gbuffer_pipeline_list_:
                        // base_frag*_DECAL_GBUF.spv, albedo blended
                        // SRC_ALPHA/ONE_MINUS_SRC_ALPHA, alpha channel
                        // ZERO/ONE so the resolve's ">= 0.5 G-buffer
                        // written" sentinel survives, and the other
                        // three G-buffer targets write-masked off so
                        // the ground keeps its own normal / geometric
                        // normal / velocity.
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
    // Hand the drawable path the G-buffer attachment formats so the
    // population sites can build DrawMode::kGBuffer pipelines.  Call
    // once at init (application.cpp, right after the formats are
    // assembled) and again on any format change; idempotent.
    static void setGbufferRenderbufferFormats(
        const renderer::PipelineRenderbufferFormats& formats) {
        gbuffer_renderbuffer_formats_ = formats;
        gbuffer_formats_valid_ = true;
    }

    // Same contract for the glass-attribute targets (octN/linZ/rough +
    // tint/kind).  Call once at init, before assets import.
    static void setGlassRenderbufferFormats(
        const renderer::PipelineRenderbufferFormats& formats) {
        glass_renderbuffer_formats_ = formats;
        glass_formats_valid_ = true;
    }

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
    // Live state of the frustum test — lets a caller record WHETHER
    // culling was armed for a pass, not just how many nodes it lost.
    static bool frustumCullArmed();

    // ── SHADOW-CASTER CULL (depth-only passes) ───────────────────────
    // The frustum test above is deliberately skipped on depth_only
    // passes, and correctly so: a caster BEHIND the camera still throws
    // a shadow into view, so culling shadow geometry against the MAIN
    // camera frustum drops real shadows.
    //
    // That reasoning does not extend to the LIGHT's own bounds.  A
    // caster outside every cascade's light frustum cannot write a texel
    // into any cascade's depth map — there is no view from which its
    // shadow appears.  Measured: the CSM pass walks the same ~2M mesh
    // nodes the main passes do, with no rejection whatsoever, and in
    // "Regular" mode it walks them ONCE PER CASCADE, which is ~28 ms of
    // the frame on its own.
    //
    // So depth-only passes get their own volume: ONE light frustum
    // fitted to the whole camera frustum out to the last cascade split
    // — the union volume computeCascadeMatrices already builds for
    // ClusterRenderer::cullShadow.  Every cascade is a slice of that
    // same view frustum, so a caster no cascade can see is a caster
    // this volume cannot see.  Per-cascade volumes would be tighter,
    // but they buy nothing: the drawable list is walked ONCE for the
    // layered draw, and six plane sets would be six times the per-node
    // work for one pass.
    //
    // FOUR PLANES, NOT SIX — the side planes only, and this is the
    // subtle part.  The cascade ortho near plane sits at the nearest
    // frustum corner, and casters in FRONT of it (between the sun and
    // the slab) still shadow the slab: they clip to z < 0 and are kept
    // by depthClampEnable on the shadow pipelines.  Culling against a
    // near plane would delete exactly those casters — the tall tree
    // just off the top of the cascade whose shadow falls across it.
    // Along the light direction the depth clamp already handles both
    // ends, so the only sound test is the lateral one.
    //
    // planes points at 4 world-space Gribb-Hartmann side planes
    // (left, right, bottom, top), normalised, in the same sign
    // convention setFrustumCullPlanes uses.  The array is copied, so
    // the caller may reuse its storage immediately.
    //
    // Unarmed (the default, and every non-CSM depth pass) the test is
    // skipped entirely and behaviour is exactly as before.
    static void setShadowCullVolume(const glm::vec4 planes[4]);
    static void clearShadowCull();

    // ── CPU RECORDING COST INSTRUMENTATION ───────────────────────────────
    // The GPU renders this scene in ~7 ms while drawScene spends ~76 ms
    // RECORDING it, so the interesting number is not "how long did a pass
    // take" but "how many things did it walk, and how many did culling
    // actually throw away".  A wall-clock scope cannot tell a pass that
    // recorded 300k draw calls apart from one that recorded 3k and stalled;
    // these counters can.
    //
    // Plain (non-atomic) counters ON PURPOSE: every one of these is
    // incremented from the single thread that records the command buffer.
    // If that ever stops being true these turn into a data race, which is
    // exactly the tripwire you want when someone parallelises recording.
    struct DrawStats {
        uint64_t drawables = 0;     // DrawableObject::draw() bodies entered
        uint64_t nodes = 0;         // mesh nodes considered (flat lane)
        uint64_t sub_lane = 0;      // drawables that took the recursive
                                    // sub-object lane, whose nodes the
                                    // counter above does not see
        uint64_t cull_frustum = 0;  // ...rejected by the frustum sphere test
        uint64_t cull_dist = 0;     // ...rejected by the clutter-fade test
        uint64_t cull_lod = 0;      // ...rejected by the LOD band table
        uint64_t cull_shadow = 0;   // ...rejected by the shadow cull volume
        uint64_t cull_tile = 0;     // ...rejected wholesale with their tile
        uint64_t tiles = 0;         // tile cells tested
        uint64_t tiles_culled = 0;  // ...of which rejected whole
        uint64_t lod_rebuilds = 0;  // LOD survivor-list rebuilds (see
                                    // lod_tables_version_) — should be at
                                    // most one per frame, not one per pass
        uint64_t prims = 0;         // drawIndexedIndirect calls recorded
        uint64_t desc_binds = 0;    // bindDescriptorSets calls recorded
    };
    static void resetDrawStats();
    static const DrawStats& drawStats();

    // Per-frame eye position in world space, used by drawMesh's clutter
    // distance cull.  Kept separate from the frustum planes because the
    // decal pass runs at a different point in the frame than the forward
    // pass and the cull must be live for it; ObjectSceneView::drawDecals
    // sets it from its camera object.  Until it is set the distance cull
    // is inert (nothing is culled), which is the safe default.
    static void setViewerWorldPos(const glm::vec3& pos);
    static void clearViewerWorldPos();

    // Eye position for plant LOD band selection.  A SECOND eye, and not
    // the one above, for one reason: setViewerWorldPos is scoped to the
    // decal pass and cleared straight after (deliberately — a stale eye
    // silently culling geometry is nasty to debug), while the tree LOD
    // has to select in the forward pass and in every shadow cascade.
    // The distinction is that the clutter cull DROPS geometry, so being
    // wrong there loses a tile; band selection only chooses between three
    // drawings of the same tree, so being a metre stale changes nothing
    // anyone can see.  Never cleared, for the same reason: a cascade that
    // lost the eye mid-frame would disagree with the forward pass about
    // which band owns a tile, and shadow a tree that is not drawn.
    // Published by ObjectSceneView::draw (all passes) and mirrored from
    // setViewerWorldPos so the decal path keeps it warm too.
    static void setPlantLodEye(const glm::vec3& pos);

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

    // Native render-ready asset load: a .rwobj object reference whose
    // baked geometry (.rwgeo v3, sections + .rwtex textures) was written
    // at import time.  Builds a single-mesh DrawableData directly from
    // the baked data — no source-model parse, so each placed object
    // streams independently (per-object loading wheel, individual
    // pop-in) instead of riding one giant shared FBX load.  Returns
    // nullptr when the reference has no baked geometry.
    static std::shared_ptr<DrawableData> loadRwObjModel(
        const std::shared_ptr<renderer::Device>& device,
        const std::string& input_filename);

    // Native skinned-character load: a .rwchar manifest names a baked group
    // (content/.../<name>/) that holds hierarchy.rwhier (skeleton), every
    // objects/*.rwgeo (v4: geometry + per-vertex joints/weights + skin
    // table) and animation.rwanim (clips).  Builds ONE skinned DrawableData
    // — nodes_, skins_ (+joint buffers), skinned vertex attributes and
    // animations_ — entirely from raw data, so a character animates with NO
    // source model present.  Flows through the same Phase-3 descriptor /
    // _SKIN-pipeline path as the glTF loader.  Returns nullptr on failure.
    static std::shared_ptr<DrawableData> loadRwCharacter(
        const std::shared_ptr<renderer::Device>& device,
        const std::string& input_filename);

    // Native INSTANCED GROUP loader (input: <group>/instances.rwinst).
    // Builds ONE static DrawableData from a baked group — hierarchy from
    // hierarchy.rwhier (node names intact, so _lodtile_ bands, _pgate_
    // gates and world-manifest bindings all work), geometry from
    // objects/NNN_*.rwgeo (v6 LOD levels honoured), textures from
    // .rwtex, and the per-instance transform tables from
    // instances.rwinst.  The native replacement for loading the
    // generated glTF: content/ holds engine formats only.  An EMPTY
    // rwinst (marker) loads the group as one non-instanced drawable —
    // used for the merged generated files (clutter, decals, roads).
    static std::shared_ptr<DrawableData> loadRwInstanced(
        const std::shared_ptr<renderer::Device>& device,
        const std::string& input_filename);
};

// ── PCG world-manifest instance overrides ───────────────────────────
// The terrain apply reads <map>_pcg_world.json and, before importing a
// sample LIBRARY glb (houses / plants / game_objects), registers the
// manifest's per-node instance transforms here keyed by the library
// file name.  The glTF loader consumes the entry in
// bakeInstanceTransforms: nodes named in the map draw at the manifest
// transforms instead of their sample-sheet slot, and nodes NOT named
// are hidden — the library becomes the level's placed content, linked
// by node name, with no re-baked geometry in between.
struct PcgInstanceOverride {
    std::vector<float> t;   // 3 floats per instance
    std::vector<float> r;   // 4 floats per instance (xyzw quaternion)
    std::vector<float> s;   // 3 floats per instance
};
// key: node name, matched exactly or as a "<key>_" prefix
using PcgOverrideMap = std::map<std::string, PcgInstanceOverride>;
void setPcgInstanceOverrides(const std::string& asset_file_name,
                             PcgOverrideMap overrides);
// consume (and erase) the registered map for one asset; empty if none
PcgOverrideMap takePcgInstanceOverrides(
    const std::string& asset_file_name);
// Parse <map>_pcg_world.json (the placement stage's level manifest).
// out: library key ("houses"/"plants"/"objects") -> (library glb path,
// node-name -> instance transforms).  Returns false on any parse
// failure; implemented beside the glTF loader so the vendored JSON
// parser is reused rather than re-included elsewhere.
bool loadPcgWorldManifest(
    const std::string& manifest_path,
    std::map<std::string, std::pair<std::string, PcgOverrideMap>>&
        out_by_library);

// ── PCG PER-INSTANCE PHYSICAL REGISTRY ──────────────────────────────────
// One record per PHYSICAL PROP the placement stage put on the level —
// rocks, trees, bushes, houses, game objects — read from
// <map>_pcg_instances.json (terrain_pcg._write_instance_registry).  Each
// record carries the prop's stable 53-bit id, its weight in kg and its
// placed transform; every prop starts STATIC, and the two runtime states
// (moving, destroyed) exist so gameplay can push a rock downhill or
// remove it without the pipeline knowing.
//
// HOW A RECORD REACHES THE GPU.  Placed props render as baked
// EXT_mesh_gpu_instancing instances (DrawableData::baked_instances_, a
// table uploaded once at load and then never rewritten — see
// updateInstanceBuffer's baked opt-out).  At bake time the loader hands
// every freshly created instance range to bindBakedRange, which matches
// slots to records by NODE NAME PREFIX plus decimetre-quantised plan
// position — position, not row order, because the same record appears
// once per LOD band on the library path and per-tile-subsetted on the
// baked-glb path, and order survives neither.  setTransform/setState
// then queue the affected slots, and flushDirty rewrites them in place
// through vkCmdUpdateBuffer (64 B per slot) from the per-frame update
// pass, with the same barriers every other in-place rewrite uses.
//
// Thread contract: load/bind run on asset-load threads, set*/find on
// gameplay, flushDirty on the render thread — one mutex covers all of
// it; every call is short.
struct PcgInstanceRecord {
    uint64_t  id = 0;
    uint32_t  node = 0;          // index into the loaded node-name table
    uint8_t   category = 0;      // 0 rocks 1 trees 2 bushes 3 houses 4 objects
    uint8_t   state = 0;         // 0 static 1 moving 2 destroyed
    float     weight_kg = 0.0f;
    glm::vec3 t{0.0f};
    float     yaw = 0.0f;
    float     scale = 1.0f;
};

class PcgInstanceRegistry {
public:
    static PcgInstanceRegistry& get();

    // Parse <map>_pcg_instances.json; replaces the previous registry and
    // drops every binding (call at terrain apply, BEFORE the placed
    // glbs / libraries import).  Returns false on a parse failure.
    bool load(const std::string& json_path);
    void clear();
    size_t size() const;

    const PcgInstanceRecord* find(uint64_t id) const;   // nullptr if unknown
    // ids within `radius` of `p` (plan distance), optionally one
    // category only.  Linear scan — fine for occasional gameplay
    // queries, wrong inside a per-frame loop over many callers.
    std::vector<uint64_t> queryRadius(const glm::vec3& p, float radius,
                                      int category = -1) const;

    // Every record whose NODE NAME starts with `prefix`, optionally in
    // one category — "obj_bed" over category 4 is every bed the
    // placement stage put in the level, at the transform it was placed
    // at.  Returned BY VALUE: the caller (CitizenSystem, binning
    // furniture to houses at load) keeps the snapshot, and handing out
    // pointers into recs_ past the lock would be a lie.  Linear scan
    // like queryRadius — a load-time query, not a per-frame one.
    std::vector<PcgInstanceRecord> queryByNodePrefix(
        const std::string& prefix, int category = -1) const;

    // Runtime state.  setTransform re-places the prop (marks it moving);
    // setState(2) hides every bound GPU slot via a zero-scale rewrite,
    // and setState(0|1) after a destroy restores the stored transform.
    bool setState(uint64_t id, uint8_t state);
    bool setTransform(uint64_t id, const glm::vec3& t, float yaw,
                      float scale);

    // Loader-side hooks — not for gameplay code.
    void bindBakedRange(const std::shared_ptr<DrawableData>& data,
                        const std::string& node_name,
                        uint32_t first_slot,
                        const std::vector<float>& translations,
                        size_t count);
    void flushDirty(const std::shared_ptr<DrawableData>& data,
                    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

private:
    PcgInstanceRegistry() = default;
    uint64_t posKey(uint32_t node, float x, float z) const;
    BakedInstanceXform xformOf(const PcgInstanceRecord& r) const;
    void queueRecord(uint32_t rec_idx, const BakedInstanceXform& x);

    mutable std::mutex mu_;
    std::vector<std::string> nodes_;
    std::vector<PcgInstanceRecord> recs_;
    std::unordered_map<uint64_t, uint32_t> by_id_;
    std::unordered_map<uint64_t, uint32_t> by_pos_;   // posKey -> rec idx
    struct Binding {
        std::weak_ptr<DrawableData> data;
        uint32_t slot;
    };
    std::unordered_map<uint32_t, std::vector<Binding>> binds_;
    std::unordered_map<const DrawableData*,
                       std::vector<std::pair<uint32_t, BakedInstanceXform>>>
        dirty_;
};

} // namespace game_object
} // namespace engine