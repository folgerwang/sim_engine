#pragma once
//
// model_inspect.h  --  CPU-only structural peek at model files.
//
// Returns the names of a model file's renderable sub-objects (nodes with a
// mesh attached) WITHOUT creating any GPU resources — the same rule the
// editor Outliner uses for a loaded DrawableObject (multi-mesh files become
// a group whose mesh nodes are the children; single-mesh files return an
// empty list).  Used by the Content Browser to present an imported asset as
// a virtual folder of its sub-objects, and by the import worker to bake the
// list into the asset's .rwmeta sidecar.
//
// Supported: .gltf / .glb (tinygltf), .fbx (ufbx), .obj ('o ' / 'g ' scan).
// Anything else (or a parse failure) returns an empty list.
//
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>   // glm::u16vec4 (skin joint indices)

namespace engine {
namespace helper {

std::vector<std::string> listModelSubObjects(const std::string& path);

// Everything the Debug Display's PBR preview pass needs from a model file:
// geometry (world-space) plus MATERIAL SECTIONS — FBX assigns materials
// per-face and glTF per-primitive, so one object commonly spans several
// base-colour textures.  Each section is a contiguous index range with its
// own factors + (optional) texture.
struct PreviewTexture {
    std::vector<uint8_t> rgba;   // decoded RGBA8
    int                  w = 0;
    int                  h = 0;
};
struct PreviewSection {
    uint32_t  first_index = 0;   // into ModelPreviewData::indices
    uint32_t  index_count = 0;
    glm::vec4 base_color  = glm::vec4(1.0f);
    float     metallic    = 0.0f;
    float     roughness   = 0.6f;
    int       tex_index   = -1;  // albedo, into ModelPreviewData::textures
    int       nrm_index   = -1;  // normal map (-1 = none)
    int       mr_index    = -1;  // metallic-roughness map (glTF G=rough B=metal)
};
struct ModelPreviewData {
    std::vector<glm::vec3>      positions;
    std::vector<glm::vec3>      normals;   // recomputed when absent
    std::vector<glm::vec2>      uvs;       // zero-filled when absent
    std::vector<uint32_t>       indices;
    std::vector<PreviewTexture> textures;  // dedup'd across sections
    std::vector<PreviewSection> sections;  // >= 1 after a successful load

    // ── Skinned meshes only (empty otherwise) ─────────────────────────
    // Per-vertex skinning data, parallel to `positions`: 4 joint indices
    // (into the skin table below) + 4 normalized weights.
    std::vector<glm::u16vec4>   joints;
    std::vector<glm::vec4>      weights;
    // Optional: baked distance-derived closeness for the same 4 joints (auto-rig
    // _CLOSENESS_0).  Parallel to weights when present; empty otherwise.  Used by
    // the debug display so it renders the baked distance, not a runtime recompute.
    std::vector<glm::vec4>      closeness;
    // ── 8-bone skinning debug path: second skin set (influences 4..7) ──
    // glTF JOINTS_1/WEIGHTS_1/_CLOSENESS_1.  Parallel to joints/weights/
    // closeness when present; empty otherwise (4-bone legacy assets).
    std::vector<glm::u16vec4>   joints1;
    std::vector<glm::vec4>      weights1;
    std::vector<glm::vec4>      closeness1;
    // Skin table: per joint, the node it binds to (index into the
    // group's hierarchy.rwhier node array — glTF node order) and the
    // inverse bind matrix.  Together with joints/weights this is the
    // complete data a native skinned renderer needs.
    std::vector<int32_t>        skin_joint_nodes;
    std::vector<glm::mat4>      skin_inverse_bind;
};

// CPU-only preview load (Debug Display): fills world-space triangles +
// shading data for the sub_index-th renderable sub-object — same enumeration
// order as listModelSubObjects.  sub_index < 0 loads the WHOLE file.
// Supports .gltf/.glb (tinygltf) and .fbx (ufbx); returns false otherwise
// or on parse failure.
bool loadModelPreviewData(const std::string& path, int sub_index,
                          ModelPreviewData& out);

// External texture files a model depends on.  Each entry is
//   { dst_relative_path, src_absolute_path }
// where dst_relative_path preserves the model's own (sanitised) relative
// reference, so copying each file to <model_dir>/<dst_relative_path> keeps
// the model's texture resolution working from the copied location.  Used by
// the Content Browser import to bring textures along with the model.
// Embedded textures (GLB buffers, FBX embedded content) need no copying and
// are not listed.
std::vector<std::pair<std::string, std::string>>
listModelTextureDependencies(const std::string& path);

// True when the model contains a skin (skeleton-driven deformation):
// glTF skins / FBX skin deformers.  Skinned models are CHARACTER assets —
// the static render-ready bake would freeze them, so the import copies
// them source-form instead.
bool modelHasSkin(const std::string& path);

// All external files the model needs to load from a copied location:
// listModelTextureDependencies PLUS (for .gltf) the binary buffer files
// (.bin).  Same {dst_relative, src_absolute} pair convention.
std::vector<std::pair<std::string, std::string>>
listModelFileDependencies(const std::string& path);

// Decode the top mip of a .dds file to RGBA8 (DXT1/3/5 + 32bpp
// uncompressed).  Returns false for DX10/BC7 and exotic formats.  Shared by
// the preview texture loader and the Content Browser thumbnails.
bool decodeDdsToRgba(const std::string& path, int& w, int& h,
                     std::vector<unsigned char>& rgba);

// Load any supported image file to RGBA8: stb formats (png/jpg/tga/bmp/…)
// plus .dds via decodeDdsToRgba.  Returns false when unreadable.
bool loadImageFileRgba(const std::string& path, int& w, int& h,
                       std::vector<unsigned char>& rgba);

// .rwobj — tiny text "object reference" asset written when a multi-object
// model is imported: it names ONE sub-object of a source model file so the
// engine can treat each sub-object as an independently placeable drawable.
//   rwobj=1
//   source=<model file, relative to the .rwobj's folder>
//   node=<sub-object ordinal (k-th mesh node)>
//   name=<display name>
//   geo=<render-ready .rwgeo, relative to the .rwobj's folder (optional)>
// Reads the file and resolves `source` / `geo` against its location.
// out_geo_path is empty when the object has no baked data.
bool readRwObjRef(const std::string& rwobj_path,
                  std::string& out_source_path,
                  int&         out_sub_index,
                  std::string& out_name,
                  std::string& out_geo_path);

// Source-space bounds recorded in the .rwobj at bake time (bbox= line).
// Returns false for refs baked before bounds existed.
bool readRwObjBounds(const std::string& rwobj_path,
                     glm::vec3& out_min, glm::vec3& out_max);

// ── Render-ready bake ────────────────────────────────────────────────────
// Converts an imported model ONCE into engine-ready binary assets under the
// model's group folder, the unit a future streaming system loads in/out:
//
//   <group_dir>/objects/<NNN_name>.rwgeo   — geometry + material per object
//   <group_dir>/textures/<name>.rwtex      — decoded RGBA8 base colour
//
// One source parse bakes everything; textures are dedup'd across objects.
// Returns the baked objects in sub-object enumeration order (the same order
// listModelSubObjects produces).  `log` (optional) receives one line per
// bake step — useful in the import log.
struct BakedObject {
    std::string name;        // display name (mesh node)
    std::string rwgeo_rel;   // path relative to group_dir, e.g. "objects/000_Wall.rwgeo"
    // Source-space bounds — placement uses these to cancel the source
    // offset so a placed object lands AT the drop point.
    glm::vec3   bbox_min = glm::vec3(0.0f);
    glm::vec3   bbox_max = glm::vec3(0.0f);
};
// `progress` (optional) is called after each baked object with
// (objects_done, objects_total) — drives the import progress bar.
bool bakeModelToRenderReady(
    const std::string& source_path,
    const std::string& group_dir,
    std::vector<BakedObject>& out_objects,
    const std::function<void(size_t, size_t)>& progress = {});

// Read one baked object (geometry + material + its .rwtex texture) back
// into preview data.  tex paths inside the .rwgeo are relative to the
// group folder (= parent of the objects/ folder the .rwgeo lives in).
//
// out_texture_paths (optional): receives the RESOLVED .rwtex path for
// every entry of out.textures (parallel arrays) — lets callers key a
// cross-object texture cache.  decode_textures=false skips reading the
// .rwtex pixel data entirely: out.textures gets empty placeholder
// entries (w=h=0) so section tex_index values stay valid, and the
// caller loads pixels itself (via readRwTex) only on cache misses.
bool loadRwGeo(const std::string& rwgeo_path, ModelPreviewData& out,
               std::vector<std::string>* out_texture_paths = nullptr,
               bool decode_textures = true);

// Raw .rwtex read (RGBA8).  For format-0 (legacy RGBA8) files this is
// the full-resolution image; for format-1 (BC7 VT tile cache) files it
// returns the embedded RGBA8 PREVIEW mip (≤256 px) — previews and
// thumbnails don't need more.
bool readRwTex(const std::string& path, int& w, int& h,
               std::vector<unsigned char>& rgba);

// Full .rwtex read for the engine's native .rwobj loader.
//   format 0 (legacy): preview_* = full-res pixels, bc7_tiles null.
//   format 1 (baked) : preview_* = small RGBA8 preview, alpha = full-res
//                      alpha plane (cutout textures only, else empty),
//                      bc7_tiles = the pre-encoded VT albedo tile cache
//                      (null if the file's tile geometry doesn't match
//                      the engine's current VT constants).
struct RwTexBaked {
    int w = 0, h = 0;                       // full-resolution dims
    int preview_w = 0, preview_h = 0;
    std::vector<unsigned char> preview_rgba;
    std::vector<unsigned char> alpha;       // w*h, empty when opaque
    std::shared_ptr<std::vector<uint8_t>> bc7_tiles;
};
bool readRwTexBaked(const std::string& path, RwTexBaked& out);

// Dedup triangle-soup vertices in place (exact bit-equality of
// position/normal/uv) and remap indices; index ORDER is preserved so
// section ranges stay valid.  .rwgeo files baked before the bake-side
// dedup landed store one vertex per triangle corner — run this after
// loadRwGeo so the GPU gets indexed, cache-friendly vertices.
void dedupModelVertices(ModelPreviewData& d);

// ── Node hierarchy (render-ready) ────────────────────────────────────────
// Written at bake time as <group_dir>/hierarchy.rwhier: the source model's
// FULL node tree — including transform-only intermediate nodes — with each
// node's parent link and LOCAL (node-to-parent) transform.  Baked .rwgeo
// geometry is stored in node-local space, so a renderer composes
// world(node) = local(root) * … * local(node) and can articulate/animate
// the hierarchy.  mesh_ordinal maps a node to the k-th baked object (the
// same enumeration .rwobj files use); -1 = no mesh on this node.
struct RwHierNode {
    int32_t     parent       = -1;   // index into the same array, -1 = root
    int32_t     mesh_ordinal = -1;   // k-th baked object, -1 = transform-only
    glm::mat4   local        = glm::mat4(1.0f);   // node_to_parent
    std::string name;
};
bool loadRwHier(const std::string& path, std::vector<RwHierNode>& out);

// ── Skeletal / node animation (render-ready) ──────────────────────────────
// Baked beside hierarchy.rwhier as <group_dir>/animation.rwanim: every clip
// in the source model, resampled to LINEARLY-interpolable keyframes.  Each
// channel targets a hierarchy.rwhier node index (glTF node order / ufbx
// scene->nodes order), so a baked character animates with NO source file.
// Rotation values are stored quaternion XYZW; translation/scale use XYZ
// (w unused).  This makes a character fully self-contained post-import.
enum class RwAnimPath : uint8_t { kTranslation = 0, kRotation = 1, kScale = 2 };
struct RwAnimChannel {
    int32_t                node = -1;   // hierarchy.rwhier node index
    RwAnimPath             path = RwAnimPath::kTranslation;
    uint8_t                step = 0;    // 1 = constant/step, 0 = linear/slerp
    std::vector<float>     times;       // seconds, ascending; parallel to values
    std::vector<glm::vec4> values;      // T/S: xyz (w=0); R: quat xyzw
};
struct RwAnimClip {
    std::string                name;
    float                      duration = 0.0f;   // seconds
    std::vector<RwAnimChannel> channels;
};
// Read every clip baked for a group.  Returns false (out left empty) when the
// file is absent — the group simply has no animation in that case.
bool loadRwAnim(const std::string& path, std::vector<RwAnimClip>& out);

// ── Stable asset identity ────────────────────────────────────────────────
// Deterministic 64-bit FNV-1a hash — unlike std::hash, identical across
// runs, builds and machines, so IDs survive re-imports and can key
// external systems (the ML search index, caches, …).
uint64_t stableAssetHash(const std::string& s);

// Canonical asset ID: hash of "type|path|name" with the path lower-cased
// and slash-normalised.  `canonical_path` should be relative to the
// content root so the ID is independent of where the project lives.
// Returns 16 lowercase hex characters.
std::string makeAssetId(const std::string& type,
                        const std::string& canonical_path,
                        const std::string& name);

} // namespace helper
} // namespace engine
