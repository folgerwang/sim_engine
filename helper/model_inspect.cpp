#include "model_inspect.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "tiny_gltf.h"               // implementation lives in engine_helper.cpp
#include "third_parties/fbx/ufbx.h"  // ufbx.c compiled with the engine
// Bake-time BC7 VT tile-cache encode (static, CPU-only) + the VT tile
// geometry constants stamped into format-1 .rwtex files.
#include "scene_rendering/virtual_texture.h"

// stbi_load / stbi_image_free are compiled into the project via tinygltf
// (STB_IMAGE_IMPLEMENTATION lives in engine_helper.cpp); declare just what
// the FBX base-colour image load needs.
extern "C" {
    unsigned char* stbi_load(const char*, int*, int*, int*, int);
    unsigned char* stbi_load_from_memory(const unsigned char*, int,
                                         int*, int*, int*, int);
    void stbi_image_free(void*);
    // Implementation compiled in engine_helper.cpp — used to drop a
    // human-checkable .png next to every baked .rwtex.
    int stbi_write_png(const char*, int, int, int, const void*, int);
}

namespace engine {
namespace helper {
namespace {

std::string lowerExt(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext;
}

} // anonymous namespace

std::vector<std::string> listModelSubObjects(const std::string& path) {
    std::vector<std::string> names;
    const std::string ext = lowerExt(path);

    if (ext == ".gltf" || ext == ".glb") {
        tinygltf::Model    model;
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        const bool ok = (ext == ".glb")
            ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
            : loader.LoadASCIIFromFile(&model, &err, &warn, path);
        if (!ok) return names;

        // Mirror the Outliner's grouping rule: only multi-mesh files are
        // groups with children; a single mesh node stays a flat leaf.
        int mesh_nodes = 0;
        for (const auto& n : model.nodes) if (n.mesh >= 0) ++mesh_nodes;
        if (mesh_nodes > 1) {
            names.reserve((size_t)mesh_nodes);
            for (size_t i = 0; i < model.nodes.size(); ++i) {
                if (model.nodes[i].mesh < 0) continue;
                std::string nm = model.nodes[i].name;
                if (nm.empty()) nm = "node " + std::to_string(i);
                names.push_back(std::move(nm));
            }
        }
    } else if (ext == ".fbx") {
        ufbx_load_opts opts = { 0 };
        ufbx_error     error;
        ufbx_scene* scene = ufbx_load_file(path.c_str(), &opts, &error);
        if (!scene) return names;

        int mesh_nodes = 0;
        for (size_t i = 0; i < scene->nodes.count; ++i)
            if (scene->nodes[i]->mesh) ++mesh_nodes;
        if (mesh_nodes > 1) {
            names.reserve((size_t)mesh_nodes);
            for (size_t i = 0; i < scene->nodes.count; ++i) {
                if (!scene->nodes[i]->mesh) continue;
                std::string nm = scene->nodes[i]->name.data
                               ? scene->nodes[i]->name.data : "";
                if (nm.empty()) nm = "node " + std::to_string(i);
                names.push_back(std::move(nm));
            }
        }
        ufbx_free_scene(scene);
    } else if (ext == ".obj") {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.size() > 2 &&
                (line[0] == 'o' || line[0] == 'g') && line[1] == ' ') {
                std::string nm = line.substr(2);
                while (!nm.empty() &&
                       (nm.back() == '\r' || nm.back() == ' '))
                    nm.pop_back();
                if (!nm.empty()) names.push_back(std::move(nm));
            }
        }
        if (names.size() <= 1) names.clear();   // single object = flat leaf
    }

    return names;
}

namespace {

// Local transform of a glTF node (matrix, or T*R*S composition).
glm::mat4 gltfNodeLocal(const tinygltf::Node& n) {
    if (!n.matrix.empty()) {
        glm::mat4 m(1.0f);
        for (int i = 0; i < 16; ++i) (&m[0][0])[i] = (float)n.matrix[i];
        return m;
    }
    glm::mat4 T(1.0f), R(1.0f), S(1.0f);
    if (!n.translation.empty())
        T = glm::translate(glm::mat4(1.0f),
            glm::vec3((float)n.translation[0], (float)n.translation[1],
                      (float)n.translation[2]));
    if (!n.rotation.empty()) {
        glm::quat q((float)n.rotation[3], (float)n.rotation[0],
                    (float)n.rotation[1], (float)n.rotation[2]);
        R = glm::mat4_cast(q);
    }
    if (!n.scale.empty())
        S = glm::scale(glm::mat4(1.0f),
            glm::vec3((float)n.scale[0], (float)n.scale[1],
                      (float)n.scale[2]));
    return T * R * S;
}

// Read a float vec2/vec3 attribute accessor (POSITION / NORMAL / TEXCOORD_0).
const float* gltfAccessorPtr(const tinygltf::Model& model, int acc_idx,
                             size_t& out_count, size_t& out_stride,
                             int components) {
    if (acc_idx < 0 || acc_idx >= (int)model.accessors.size()) return nullptr;
    const auto& acc = model.accessors[acc_idx];
    if (acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) return nullptr;
    const auto& bv  = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    out_count  = acc.count;
    out_stride = bv.byteStride ? bv.byteStride / sizeof(float)
                               : (size_t)components;
    return reinterpret_cast<const float*>(
        buf.data.data() + bv.byteOffset + acc.byteOffset);
}

// Append one glTF PRIMITIVE (vertices + indices) transformed by `world`.
void appendGltfPrim(const tinygltf::Model& model,
                    const tinygltf::Primitive& prim,
                    const glm::mat4& world,
                    std::vector<glm::vec3>& positions,
                    std::vector<glm::vec3>& normals,
                    std::vector<glm::vec2>& uvs,
                    std::vector<uint32_t>&  indices,
                    std::vector<glm::u16vec4>* out_joints = nullptr,
                    std::vector<glm::vec4>*    out_weights = nullptr,
                    std::vector<glm::vec4>*    out_closeness = nullptr,
                    // 8-bone debug path: second skin set (JOINTS_1 /
                    // WEIGHTS_1 / _CLOSENESS_1, influences 4..7).
                    std::vector<glm::u16vec4>* out_joints1 = nullptr,
                    std::vector<glm::vec4>*    out_weights1 = nullptr,
                    std::vector<glm::vec4>*    out_closeness1 = nullptr) {
    const glm::mat3 nrm_mat(world);   // approximate (ok for previews)
    if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1) return;
    auto pit = prim.attributes.find("POSITION");
    if (pit == prim.attributes.end()) return;

    const uint32_t base = (uint32_t)positions.size();
    size_t pc = 0, ps = 0;
    const float* p = gltfAccessorPtr(model, pit->second, pc, ps, 3);
    if (!p) return;
    for (size_t i = 0; i < pc; ++i) {
        const glm::vec4 wp = world * glm::vec4(
            p[i * ps + 0], p[i * ps + 1], p[i * ps + 2], 1.0f);
        positions.push_back(glm::vec3(wp));
    }

    // NORMAL — transformed + renormalised; zero-filled when absent
    // (the caller recomputes face normals in that case).
    const float* n = nullptr;
    size_t nc = 0, ns = 0;
    auto nit = prim.attributes.find("NORMAL");
    if (nit != prim.attributes.end())
        n = gltfAccessorPtr(model, nit->second, nc, ns, 3);
    for (size_t i = 0; i < pc; ++i) {
        glm::vec3 nv(0.0f);
        if (n && i < nc) {
            nv = nrm_mat * glm::vec3(n[i * ns + 0], n[i * ns + 1],
                                     n[i * ns + 2]);
            const float l = glm::length(nv);
            if (l > 1e-6f) nv /= l;
        }
        normals.push_back(nv);
    }

    // TEXCOORD_0 — zero-filled when absent so all streams stay aligned.
    const float* t = nullptr;
    size_t tc = 0, ts = 0;
    auto tit = prim.attributes.find("TEXCOORD_0");
    if (tit != prim.attributes.end())
        t = gltfAccessorPtr(model, tit->second, tc, ts, 2);
    for (size_t i = 0; i < pc; ++i) {
        uvs.push_back((t && i < tc)
            ? glm::vec2(t[i * ts + 0], t[i * ts + 1])
            : glm::vec2(0.0f));
    }

    // JOINTS_0 + WEIGHTS_0 — skinned meshes only, and only when the
    // caller asked for them.  Always pushes pc entries so the arrays
    // stay parallel to `positions` even for unskinned primitives mixed
    // into a skinned bake (identity binding: joint 0, weight 1).
    if (out_joints && out_weights) {
        auto rawAccessor = [&](int acc_idx, const uint8_t*& raw,
                               size_t& count, size_t& stride,
                               int& comp_type) -> bool {
            raw = nullptr; count = 0; stride = 0; comp_type = 0;
            if (acc_idx < 0 || acc_idx >= (int)model.accessors.size())
                return false;
            const auto& a = model.accessors[acc_idx];
            if (a.bufferView < 0 ||
                a.bufferView >= (int)model.bufferViews.size())
                return false;
            const auto& bv = model.bufferViews[a.bufferView];
            const auto& bf = model.buffers[bv.buffer];
            raw = bf.data.data() + bv.byteOffset + a.byteOffset;
            count = a.count;
            comp_type = a.componentType;
            const int comp_sz =
                tinygltf::GetComponentSizeInBytes(a.componentType);
            stride = bv.byteStride ? bv.byteStride : (size_t)comp_sz * 4;
            return true;
        };

        const uint8_t* jraw = nullptr; size_t jc = 0, js = 0; int jt = 0;
        const uint8_t* wraw = nullptr; size_t wc = 0, ws = 0; int wt = 0;
        const uint8_t* craw = nullptr; size_t cc = 0, cs = 0; int ct = 0;
        const uint8_t* jraw1 = nullptr; size_t jc1 = 0, js1 = 0; int jt1 = 0;
        const uint8_t* wraw1 = nullptr; size_t wc1 = 0, ws1 = 0; int wt1 = 0;
        const uint8_t* craw1 = nullptr; size_t cc1 = 0, cs1 = 0; int ct1 = 0;
        auto jit = prim.attributes.find("JOINTS_0");
        auto wit = prim.attributes.find("WEIGHTS_0");
        auto cit = prim.attributes.find("_CLOSENESS_0");   // auto-rig custom attr
        auto jit1 = prim.attributes.find("JOINTS_1");      // 8-bone debug path
        auto wit1 = prim.attributes.find("WEIGHTS_1");
        auto cit1 = prim.attributes.find("_CLOSENESS_1");
        const bool has_j = jit != prim.attributes.end() &&
            rawAccessor(jit->second, jraw, jc, js, jt);
        const bool has_w = wit != prim.attributes.end() &&
            rawAccessor(wit->second, wraw, wc, ws, wt);
        const bool has_c = out_closeness && cit != prim.attributes.end() &&
            rawAccessor(cit->second, craw, cc, cs, ct);
        const bool has_j1 = out_joints1 && jit1 != prim.attributes.end() &&
            rawAccessor(jit1->second, jraw1, jc1, js1, jt1);
        const bool has_w1 = out_weights1 && wit1 != prim.attributes.end() &&
            rawAccessor(wit1->second, wraw1, wc1, ws1, wt1);
        const bool has_c1 = out_closeness1 && cit1 != prim.attributes.end() &&
            rawAccessor(cit1->second, craw1, cc1, cs1, ct1);

        auto readJoints = [](const uint8_t* p4, int t) -> glm::u16vec4 {
            glm::u16vec4 J(0);
            for (int c = 0; c < 4; ++c) {
                J[c] = (t == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    ? reinterpret_cast<const uint16_t*>(p4)[c]
                    : (uint16_t)p4[c];   // UNSIGNED_BYTE
            }
            return J;
        };
        auto readWeights = [](const uint8_t* p4, int t) -> glm::vec4 {
            glm::vec4 W(0.0f);
            for (int c = 0; c < 4; ++c) {
                float v = 0.0f;
                if (t == TINYGLTF_COMPONENT_TYPE_FLOAT)
                    v = reinterpret_cast<const float*>(p4)[c];
                else if (t == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    v = reinterpret_cast<const uint16_t*>(p4)[c] / 65535.0f;
                else if (t == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                    v = p4[c] / 255.0f;
                W[c] = v;
            }
            return W;
        };

        for (size_t i = 0; i < pc; ++i) {
            glm::u16vec4 J(0), J1(0);
            glm::vec4    W(1.0f, 0.0f, 0.0f, 0.0f), W1(0.0f);
            glm::vec4    C(0.0f), C1(0.0f);
            if (has_j && i < jc)   J  = readJoints(jraw + i * js, jt);
            if (has_j1 && i < jc1) J1 = readJoints(jraw1 + i * js1, jt1);
            if (has_w && i < wc) {
                W = readWeights(wraw + i * ws, wt);
                if (has_w1 && i < wc1) W1 = readWeights(wraw1 + i * ws1, wt1);
                // Renormalise across BOTH sets (quantised weights rarely sum
                // to 1; an 8-bone vertex's set 0 alone sums to < 1 by design).
                const float sum = W.x + W.y + W.z + W.w +
                                  W1.x + W1.y + W1.z + W1.w;
                if (sum > 1e-6f) { W /= sum; W1 /= sum; }
                else { W = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); W1 = glm::vec4(0.0f); }
            }
            // _CLOSENESS_* is float vec4, kept RAW (no renormalize).
            if (has_c && i < cc) {
                const uint8_t* p4 = craw + i * cs;
                for (int c = 0; c < 4; ++c)
                    C[c] = (ct == TINYGLTF_COMPONENT_TYPE_FLOAT)
                        ? reinterpret_cast<const float*>(p4)[c] : 0.0f;
            }
            if (has_c1 && i < cc1) {
                const uint8_t* p4 = craw1 + i * cs1;
                for (int c = 0; c < 4; ++c)
                    C1[c] = (ct1 == TINYGLTF_COMPONENT_TYPE_FLOAT)
                        ? reinterpret_cast<const float*>(p4)[c] : 0.0f;
            }
            out_joints->push_back(J);
            out_weights->push_back(W);
            if (out_closeness) out_closeness->push_back(C);
            if (out_joints1)    out_joints1->push_back(J1);
            if (out_weights1)   out_weights1->push_back(W1);
            if (out_closeness1) out_closeness1->push_back(C1);
        }
    }

    if (prim.indices >= 0) {
        const auto& ia  = model.accessors[prim.indices];
        const auto& ib  = model.bufferViews[ia.bufferView];
        const auto& ibf = model.buffers[ib.buffer];
        const uint8_t* raw =
            ibf.data.data() + ib.byteOffset + ia.byteOffset;
        for (size_t i = 0; i < ia.count; ++i) {
            uint32_t idx = 0;
            switch (ia.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                    idx = reinterpret_cast<const uint16_t*>(raw)[i];
                    break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                    idx = reinterpret_cast<const uint32_t*>(raw)[i];
                    break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                    idx = raw[i];
                    break;
                default: break;
            }
            indices.push_back(base + idx);
        }
    } else {
        for (uint32_t i = 0; i < (uint32_t)pc; ++i)
            indices.push_back(base + i);
    }
}

// glTF PBR factors of a material index (defaults when out of range).
void gltfMaterialFactors(const tinygltf::Model& model, int material_idx,
                         glm::vec4& base_color, float& metallic,
                         float& roughness) {
    base_color = glm::vec4(1.0f);
    metallic   = 0.0f;
    roughness  = 0.6f;
    if (material_idx < 0 || material_idx >= (int)model.materials.size())
        return;
    const auto& mr = model.materials[material_idx].pbrMetallicRoughness;
    if (mr.baseColorFactor.size() == 4) {
        base_color = glm::vec4(
            (float)mr.baseColorFactor[0], (float)mr.baseColorFactor[1],
            (float)mr.baseColorFactor[2], (float)mr.baseColorFactor[3]);
    }
    metallic  = (float)mr.metallicFactor;
    roughness = (float)mr.roughnessFactor;
}

// Albedo texture index of a glTF material, with the
// KHR_materials_pbrSpecularGlossiness diffuseTexture fallback — older
// Sketchfab / Mixamo exports (e.g. scene-skinned.gltf) put the albedo
// there and leave pbrMetallicRoughness.baseColorTexture at -1.  The
// engine's glTF loader carries the same fallback; without it those
// assets bake/preview untextured.
int gltfAlbedoTextureIndex(const tinygltf::Model& model, int material_idx) {
    if (material_idx < 0 || material_idx >= (int)model.materials.size())
        return -1;
    const auto& m = model.materials[material_idx];
    int tex = m.pbrMetallicRoughness.baseColorTexture.index;
    if (tex < 0) {
        auto it = m.extensions.find("KHR_materials_pbrSpecularGlossiness");
        if (it != m.extensions.end() && it->second.IsObject() &&
            it->second.Has("diffuseTexture")) {
            const auto& dt = it->second.Get("diffuseTexture");
            if (dt.IsObject() && dt.Has("index") &&
                dt.Get("index").IsNumber()) {
                tex = dt.Get("index").GetNumberAsInt();
            }
        }
    }
    if (tex < 0 || tex >= (int)model.textures.size()) return -1;
    return tex;
}

// ufbx 3x4 matrix → glm column-major 4x4.
glm::mat4 ufbxToGlm(const ufbx_matrix& m) {
    glm::mat4 r(1.0f);
    r[0] = glm::vec4((float)m.cols[0].x, (float)m.cols[0].y,
                     (float)m.cols[0].z, 0.0f);
    r[1] = glm::vec4((float)m.cols[1].x, (float)m.cols[1].y,
                     (float)m.cols[1].z, 0.0f);
    r[2] = glm::vec4((float)m.cols[2].x, (float)m.cols[2].y,
                     (float)m.cols[2].z, 0.0f);
    r[3] = glm::vec4((float)m.cols[3].x, (float)m.cols[3].y,
                     (float)m.cols[3].z, 1.0f);
    return r;
}

// FBX PBR factors + base-colour texture of a ufbx material.
void fbxMaterialFactors(const ufbx_material* mat, glm::vec4& base_color,
                        float& metallic, float& roughness,
                        const ufbx_texture*& tex) {
    base_color = glm::vec4(1.0f);
    metallic   = 0.0f;
    roughness  = 0.6f;
    tex        = nullptr;
    if (!mat) return;
    const auto& bc = mat->pbr.base_color;
    base_color = glm::vec4(
        (float)bc.value_vec4.x, (float)bc.value_vec4.y,
        (float)bc.value_vec4.z,
        bc.value_vec4.w > 0.0 ? (float)bc.value_vec4.w : 1.0f);
    metallic  = (float)mat->pbr.metalness.value_real;
    roughness = (float)mat->pbr.roughness.value_real;
    if (roughness <= 0.0f) roughness = 0.6f;
    tex = bc.texture;
    if (!tex) tex = mat->fbx.diffuse_color.texture;
    // Texture-driven colour with an unset (black) FBX factor → white.
    if (tex && glm::length(glm::vec3(base_color)) < 0.01f)
        base_color = glm::vec4(1.0f);
}

// Append one ufbx mesh as FLATTENED per-corner vertices (FBX normals/UVs are
// per-index attributes), transformed by the node's geometry_to_world.
// Triangles are BUCKETED BY MATERIAL (FBX assigns materials per face):
// out_mats[k] is the material of bucket out_buckets[k] (may be null).
// world_space=true bakes node->geometry_to_world into the vertices (direct
// source previews); false keeps geometry in NODE-LOCAL space (the
// hierarchical render-ready bake — world comes from hierarchy.rwhier).
void appendFbxMesh(const ufbx_node* node,
                   std::vector<glm::vec3>& positions,
                   std::vector<glm::vec3>& normals,
                   std::vector<glm::vec2>& uvs,
                   std::vector<const ufbx_material*>&  out_mats,
                   std::vector<std::vector<uint32_t>>& out_buckets,
                   bool world_space = true,
                   const ufbx_skin_deformer* skin = nullptr,
                   std::vector<glm::u16vec4>* out_joints = nullptr,
                   std::vector<glm::vec4>*    out_weights = nullptr) {
    const ufbx_mesh* m = node->mesh;
    if (!m) return;
    const ufbx_matrix& xform =
        world_space ? node->geometry_to_world : node->geometry_to_node;

    // Resolve a per-face material slot: per-instance node materials first,
    // then the mesh's own list (exporters vary).
    auto mat_of = [&](uint32_t mi) -> const ufbx_material* {
        if (mi < node->materials.count) return node->materials.data[mi];
        if (mi < m->materials.count)    return m->materials.data[mi];
        if (node->materials.count > 0)  return node->materials.data[0];
        if (m->materials.count > 0)     return m->materials.data[0];
        return nullptr;
    };
    auto bucket_of = [&](const ufbx_material* mat) -> std::vector<uint32_t>& {
        for (size_t i = 0; i < out_mats.size(); ++i)
            if (out_mats[i] == mat) return out_buckets[i];
        out_mats.push_back(mat);
        out_buckets.emplace_back();
        return out_buckets.back();
    };

    const bool has_n  = m->vertex_normal.exists;
    const bool has_uv = m->vertex_uv.exists;
    const glm::mat3 nrm_mat(
        glm::vec3((float)xform.cols[0].x, (float)xform.cols[0].y,
                  (float)xform.cols[0].z),
        glm::vec3((float)xform.cols[1].x, (float)xform.cols[1].y,
                  (float)xform.cols[1].z),
        glm::vec3((float)xform.cols[2].x, (float)xform.cols[2].y,
                  (float)xform.cols[2].z));

    std::vector<uint32_t> tri(64 * 3);
    for (size_t fi = 0; fi < m->faces.count; ++fi) {
        const ufbx_face face = m->faces.data[fi];
        const size_t need =
            (size_t)(face.num_indices >= 3 ? face.num_indices - 2 : 0) * 3;
        if (need == 0) continue;
        if (tri.size() < need) tri.resize(need);
        const uint32_t ntri =
            ufbx_triangulate_face(tri.data(), tri.size(), m, face);

        const uint32_t mat_slot =
            (fi < m->face_material.count) ? m->face_material.data[fi] : 0u;
        std::vector<uint32_t>& bucket = bucket_of(mat_of(mat_slot));

        for (uint32_t t = 0; t < ntri * 3; ++t) {
            const uint32_t ix = tri[t];
            const ufbx_vec3 p = ufbx_transform_position(
                &xform,
                ufbx_get_vertex_vec3(&m->vertex_position, ix));
            bucket.push_back((uint32_t)positions.size());
            positions.push_back(
                glm::vec3((float)p.x, (float)p.y, (float)p.z));

            glm::vec3 nv(0.0f);
            if (has_n) {
                const ufbx_vec3 n =
                    ufbx_get_vertex_vec3(&m->vertex_normal, ix);
                nv = nrm_mat * glm::vec3((float)n.x, (float)n.y, (float)n.z);
                const float l = glm::length(nv);
                if (l > 1e-6f) nv /= l;
            }
            normals.push_back(nv);

            glm::vec2 uv(0.0f);
            if (has_uv) {
                const ufbx_vec2 u = ufbx_get_vertex_vec2(&m->vertex_uv, ix);
                // FBX UVs are V-up (origin bottom-left); image rows are
                // top-down — flip V, mirroring the engine loader's
                // m_flip_v_ behaviour.  Verified on the street-sign FRONT
                // face: without the flip its text renders upside down.
                uv = glm::vec2((float)u.x, 1.0f - (float)u.y);
            }
            uvs.push_back(uv);

            // Skin: per corner, look up the control point's weights.  Local
            // joint index == cluster index (the bake writes skin_joint_nodes
            // parallel to clusters).  Pushes one entry per emitted vertex so
            // joints/weights stay parallel to positions (identity binding for
            // unskinned verts in a skinned mesh).
            if (out_joints && out_weights) {
                glm::u16vec4 J(0);
                glm::vec4    W(1.0f, 0.0f, 0.0f, 0.0f);
                if (skin) {
                    const uint32_t cp = m->vertex_position.indices.data[ix];
                    if (cp < skin->vertices.count) {
                        const ufbx_skin_vertex sv = skin->vertices.data[cp];
                        glm::u16vec4 jj(0);
                        glm::vec4    ww(0.0f);
                        const uint32_t nw =
                            sv.num_weights < 4u ? sv.num_weights : 4u;
                        for (uint32_t w = 0; w < nw; ++w) {
                            const ufbx_skin_weight sw =
                                skin->weights.data[sv.weight_begin + w];
                            jj[(int)w] = (uint16_t)sw.cluster_index;
                            ww[(int)w] = (float)sw.weight;
                        }
                        const float sum = ww.x + ww.y + ww.z + ww.w;
                        if (sum > 1e-6f) { W = ww / sum; J = jj; }
                    }
                }
                out_joints->push_back(J);
                out_weights->push_back(W);
            }
        }
    }
}

// Area-weighted normal recompute for files that carried none.
void recomputeNormalsCpu(const std::vector<glm::vec3>& positions,
                         const std::vector<uint32_t>&  indices,
                         std::vector<glm::vec3>&       normals) {
    normals.assign(positions.size(), glm::vec3(0.0f));
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        if (a >= positions.size() || b >= positions.size() ||
            c >= positions.size()) continue;
        const glm::vec3 fn = glm::cross(positions[b] - positions[a],
                                        positions[c] - positions[a]);
        normals[a] += fn; normals[b] += fn; normals[c] += fn;
    }
    for (auto& n : normals) {
        const float l = glm::length(n);
        n = (l > 1e-6f) ? (n / l) : glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

} // anonymous namespace

bool loadModelPreviewData(const std::string& path, int sub_index,
                          ModelPreviewData& out) {
    out = ModelPreviewData{};
    const std::string ext = lowerExt(path);

    if (ext == ".gltf" || ext == ".glb") {
        tinygltf::Model    model;
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        const bool ok = (ext == ".glb")
            ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
            : loader.LoadASCIIFromFile(&model, &err, &warn, path);
        if (!ok) return false;

        // World transform per node (scene-root walk).
        std::vector<glm::mat4> node_world(model.nodes.size(),
                                          glm::mat4(1.0f));
        {
            std::function<void(int, const glm::mat4&)> walk =
                [&](int idx, const glm::mat4& parent) {
                    node_world[idx] =
                        parent * gltfNodeLocal(model.nodes[idx]);
                    for (int c : model.nodes[idx].children)
                        walk(c, node_world[idx]);
                };
            std::vector<bool> is_child(model.nodes.size(), false);
            for (const auto& n : model.nodes)
                for (int c : n.children) is_child[c] = true;
            for (int i = 0; i < (int)model.nodes.size(); ++i)
                if (!is_child[i]) walk(i, glm::mat4(1.0f));
        }

        // In-memory texture cache: glTF image index → out.textures index.
        std::unordered_map<int, int> img_cache;
        auto tex_of_material = [&](int material_idx) -> int {
            const int tex_idx =
                gltfAlbedoTextureIndex(model, material_idx);
            if (tex_idx < 0) return -1;
            const int src = model.textures[tex_idx].source;
            if (src < 0 || src >= (int)model.images.size()) return -1;
            auto it = img_cache.find(src);
            if (it != img_cache.end()) return it->second;
            const auto& img = model.images[src];
            if (img.image.empty() || img.width <= 0 || img.height <= 0 ||
                (img.component != 3 && img.component != 4)) return -1;
            PreviewTexture pt;
            pt.w = img.width;
            pt.h = img.height;
            const size_t n = (size_t)img.width * img.height;
            pt.rgba.resize(n * 4);
            for (size_t i = 0; i < n; ++i) {
                const uint8_t* sp = img.image.data() + i * img.component;
                uint8_t* dp = pt.rgba.data() + i * 4;
                dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
                dp[3] = (img.component == 4) ? sp[3] : 255;
            }
            const int idx = (int)out.textures.size();
            out.textures.push_back(std::move(pt));
            img_cache.emplace(src, idx);
            return idx;
        };

        // Mesh nodes in declaration order — the SAME enumeration that
        // listModelSubObjects uses, so sub_index lines up with the names.
        // Each glTF PRIMITIVE becomes a material section.
        int k = 0;
        for (int i = 0; i < (int)model.nodes.size(); ++i) {
            if (model.nodes[i].mesh < 0) continue;
            if (sub_index < 0 || k == sub_index) {
                const auto& mesh = model.meshes[model.nodes[i].mesh];
                for (const auto& prim : mesh.primitives) {
                    const uint32_t first = (uint32_t)out.indices.size();
                    appendGltfPrim(model, prim, node_world[i],
                                   out.positions, out.normals, out.uvs,
                                   out.indices);
                    const uint32_t count =
                        (uint32_t)out.indices.size() - first;
                    if (count < 3) continue;
                    PreviewSection sec;
                    sec.first_index = first;
                    sec.index_count = count;
                    gltfMaterialFactors(model, prim.material,
                                        sec.base_color, sec.metallic,
                                        sec.roughness);
                    sec.tex_index = tex_of_material(prim.material);
                    out.sections.push_back(sec);
                }
                if (sub_index >= 0) break;
            }
            ++k;
        }
    } else if (ext == ".fbx") {
        // RAW load options — identical to the engine's loadFbxModel.  The
        // axis/unit conversion we used before can MIRROR the geometry when
        // the source handedness differs (the street-sign test showed
        // left-right flipped text), making the preview disagree with the
        // engine renderer.
        ufbx_load_opts opts{};
        ufbx_error error;
        ufbx_scene* scene = ufbx_load_file(path.c_str(), &opts, &error);
        if (!scene) return false;

        // In-memory texture cache: ufbx texture → out.textures index.
        // Decode order: embedded content → ufbx-resolved path → absolute
        // path → <model_dir>/<basename> (dds-aware via loadImageFileRgba).
        std::unordered_map<const ufbx_texture*, int> tex_cache;
        auto tex_of = [&](const ufbx_texture* tex) -> int {
            if (!tex) return -1;
            auto it = tex_cache.find(tex);
            if (it != tex_cache.end()) return it->second;

            namespace fs = std::filesystem;
            int w = 0, h = 0;
            std::vector<unsigned char> rgba;
            bool loaded = false;
            std::string tried;
            if (tex->content.size > 0 && tex->content.data) {
                int comp = 0;
                unsigned char* px = stbi_load_from_memory(
                    (const unsigned char*)tex->content.data,
                    (int)tex->content.size, &w, &h, &comp, 4);
                if (px && w > 0 && h > 0) {
                    rgba.assign(px, px + (size_t)w * h * 4);
                    loaded = true;
                }
                if (px) stbi_image_free(px);
                tried = "<embedded>";
            }
            if (!loaded && tex->filename.data && tex->filename.length > 0) {
                tried = tex->filename.data;
                loaded = loadImageFileRgba(tried, w, h, rgba);
            }
            if (!loaded && tex->absolute_filename.data &&
                tex->absolute_filename.length > 0) {
                tried = tex->absolute_filename.data;
                loaded = loadImageFileRgba(tried, w, h, rgba);
            }
            if (!loaded && tex->filename.data && tex->filename.length > 0) {
                const fs::path flat =
                    fs::path(path).parent_path() /
                    fs::path(tex->filename.data).filename();
                tried = flat.string();
                loaded = loadImageFileRgba(tried, w, h, rgba);
            }
            std::cout << "[preview] fbx base-colour texture '"
                      << (tex->filename.data ? tex->filename.data : "?")
                      << "' -> " << (loaded ? "loaded" : "FAILED")
                      << " (last tried: " << tried << ")" << std::endl;
            int idx = -1;
            if (loaded) {
                PreviewTexture pt;
                pt.w = w;
                pt.h = h;
                pt.rgba = std::move(rgba);
                idx = (int)out.textures.size();
                out.textures.push_back(std::move(pt));
            }
            tex_cache.emplace(tex, idx);
            return idx;
        };

        int k = 0;
        for (size_t i = 0; i < scene->nodes.count; ++i) {
            const ufbx_node* node = scene->nodes.data[i];
            if (!node->mesh) continue;
            if (sub_index < 0 || k == sub_index) {
                // Geometry arrives bucketed by material (FBX assigns
                // materials per face) — one section per bucket.
                std::vector<const ufbx_material*>  mats;
                std::vector<std::vector<uint32_t>> buckets;
                appendFbxMesh(node, out.positions, out.normals, out.uvs,
                              mats, buckets);
                for (size_t b = 0; b < buckets.size(); ++b) {
                    if (buckets[b].size() < 3) continue;
                    PreviewSection sec;
                    sec.first_index = (uint32_t)out.indices.size();
                    sec.index_count = (uint32_t)buckets[b].size();
                    out.indices.insert(out.indices.end(),
                                       buckets[b].begin(),
                                       buckets[b].end());
                    const ufbx_texture* tex = nullptr;
                    fbxMaterialFactors(mats[b], sec.base_color,
                                       sec.metallic, sec.roughness, tex);
                    sec.tex_index = tex_of(tex);
                    out.sections.push_back(sec);
                }
                if (sub_index >= 0) break;
            }
            ++k;
        }
        ufbx_free_scene(scene);
    } else {
        return false;
    }

    if (out.positions.empty() || out.indices.size() < 3) return false;

    // Guarantee at least one section spanning everything.
    if (out.sections.empty()) {
        PreviewSection sec;
        sec.first_index = 0;
        sec.index_count = (uint32_t)out.indices.size();
        out.sections.push_back(sec);
    }

    // Files without authored normals (all-zero stream) get face normals.
    bool any_normal = false;
    for (const auto& n : out.normals) {
        if (glm::length(n) > 0.5f) { any_normal = true; break; }
    }
    if (out.normals.size() != out.positions.size() || !any_normal) {
        recomputeNormalsCpu(out.positions, out.indices, out.normals);
    }
    // Drop the UV stream only when NOTHING is textured.
    if (out.textures.empty()) {
        out.uvs.clear();
    }
    return true;
}

// ── DDS decode (top mip → RGBA8) ────────────────────────────────────────────
namespace {

constexpr uint32_t ddsFourCC(char a, char b, char c, char d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
           ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}
inline uint32_t ddsRd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline void ddsRgb565(uint16_t c, int& r, int& g, int& b) {
    int r5 = (c >> 11) & 0x1F, g6 = (c >> 5) & 0x3F, b5 = c & 0x1F;
    r = (r5 << 3) | (r5 >> 2);
    g = (g6 << 2) | (g6 >> 4);
    b = (b5 << 3) | (b5 >> 2);
}
inline void ddsDecodeBC1(const uint8_t* b, uint8_t out[64],
                         bool punchthrough) {
    uint16_t c0 = (uint16_t)(b[0] | (b[1] << 8));
    uint16_t c1 = (uint16_t)(b[2] | (b[3] << 8));
    int r[4], g[4], bl[4], a[4] = { 255, 255, 255, 255 };
    ddsRgb565(c0, r[0], g[0], bl[0]);
    ddsRgb565(c1, r[1], g[1], bl[1]);
    if (c0 > c1 || !punchthrough) {
        r[2] = (2*r[0]+r[1])/3; g[2] = (2*g[0]+g[1])/3; bl[2] = (2*bl[0]+bl[1])/3;
        r[3] = (r[0]+2*r[1])/3; g[3] = (g[0]+2*g[1])/3; bl[3] = (bl[0]+2*bl[1])/3;
    } else {
        r[2] = (r[0]+r[1])/2; g[2] = (g[0]+g[1])/2; bl[2] = (bl[0]+bl[1])/2;
        r[3] = g[3] = bl[3] = 0; a[3] = 0;
    }
    uint32_t bits = ddsRd32(b + 4);
    for (int i = 0; i < 16; ++i) {
        int idx = (bits >> (i * 2)) & 3;
        out[i*4+0] = (uint8_t)r[idx];  out[i*4+1] = (uint8_t)g[idx];
        out[i*4+2] = (uint8_t)bl[idx]; out[i*4+3] = (uint8_t)a[idx];
    }
}
inline void ddsDecodeBC3Alpha(const uint8_t* b, uint8_t alpha[16]) {
    int a0 = b[0], a1 = b[1], al[8]; al[0] = a0; al[1] = a1;
    if (a0 > a1) { for (int i = 1; i < 7; ++i) al[i+1] = ((7-i)*a0 + i*a1)/7; }
    else { for (int i = 1; i < 5; ++i) al[i+1] = ((5-i)*a0 + i*a1)/5;
           al[6] = 0; al[7] = 255; }
    uint64_t bits = 0;
    for (int i = 0; i < 6; ++i) bits |= (uint64_t)b[2+i] << (8*i);
    for (int i = 0; i < 16; ++i) alpha[i] = (uint8_t)al[(bits >> (i*3)) & 7];
}

} // anonymous namespace

bool decodeDdsToRgba(const std::string& path, int& W, int& H,
                     std::vector<unsigned char>& rgba) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (d.size() < 128 || ddsRd32(d.data()) != 0x20534444u) return false;
    const uint8_t* hdr = d.data() + 4;
    const uint32_t height = ddsRd32(hdr + 8), width = ddsRd32(hdr + 12);
    if (width == 0 || height == 0 || width > 16384 || height > 16384)
        return false;
    const uint8_t* pf = hdr + 72;                       // DDS_PIXELFORMAT
    const uint32_t pfFlags = ddsRd32(pf + 4), fourCC = ddsRd32(pf + 8),
                   rgbBits = ddsRd32(pf + 12);
    const uint32_t rMask = ddsRd32(pf + 16), gMask = ddsRd32(pf + 20),
                   bMask = ddsRd32(pf + 24), aMask = ddsRd32(pf + 28);
    const bool isFourCC = (pfFlags & 0x4) != 0;
    if (isFourCC && fourCC == ddsFourCC('D','X','1','0')) return false;

    W = (int)width; H = (int)height;
    rgba.assign((size_t)W * H * 4, 255);
    const uint8_t* src = d.data() + 128;
    const size_t avail = d.size() - 128;
    const int bw = (W + 3) / 4, bh = (H + 3) / 4;

    auto putBlock = [&](int bx, int by, const uint8_t px[64]) {
        for (int py = 0; py < 4; ++py) for (int pxx = 0; pxx < 4; ++pxx) {
            int X = bx*4 + pxx, Y = by*4 + py;
            if (X >= W || Y >= H) continue;
            const uint8_t* s = px + (py*4 + pxx)*4;
            uint8_t* o = &rgba[((size_t)Y*W + X)*4];
            o[0]=s[0]; o[1]=s[1]; o[2]=s[2]; o[3]=s[3];
        }
    };

    if (isFourCC && fourCC == ddsFourCC('D','X','T','1')) {
        if (avail < (size_t)bw*bh*8) return false;
        for (int by = 0; by < bh; ++by) for (int bx = 0; bx < bw; ++bx) {
            uint8_t px[64];
            ddsDecodeBC1(src + ((size_t)by*bw + bx)*8, px, true);
            putBlock(bx, by, px);
        }
        return true;
    }
    if (isFourCC && (fourCC == ddsFourCC('D','X','T','5') ||
                     fourCC == ddsFourCC('D','X','T','3'))) {
        const bool dxt5 = (fourCC == ddsFourCC('D','X','T','5'));
        if (avail < (size_t)bw*bh*16) return false;
        for (int by = 0; by < bh; ++by) for (int bx = 0; bx < bw; ++bx) {
            const uint8_t* blk = src + ((size_t)by*bw + bx)*16;
            uint8_t px[64];
            ddsDecodeBC1(blk + 8, px, false);
            if (dxt5) {
                uint8_t al[16];
                ddsDecodeBC3Alpha(blk, al);
                for (int i = 0; i < 16; ++i) px[i*4+3] = al[i];
            } else {
                for (int i = 0; i < 16; ++i) {
                    int byte = blk[i/2];
                    int a4 = (i&1) ? ((byte>>4)&0xF) : (byte&0xF);
                    px[i*4+3] = (uint8_t)(a4*17);
                }
            }
            putBlock(bx, by, px);
        }
        return true;
    }
    if (!isFourCC && rgbBits == 32) {
        if (avail < (size_t)W*H*4) return false;
        auto shift = [](uint32_t m){ int s=0; if(!m) return 0;
                                     while(!(m&1)){m>>=1;++s;} return s; };
        const int sr = shift(rMask), sg = shift(gMask),
                  sb = shift(bMask), sa = shift(aMask);
        for (int i = 0; i < W*H; ++i) {
            uint32_t p = ddsRd32(src + (size_t)i*4);
            rgba[i*4+0] = (uint8_t)((p & rMask) >> sr);
            rgba[i*4+1] = (uint8_t)((p & gMask) >> sg);
            rgba[i*4+2] = (uint8_t)((p & bMask) >> sb);
            rgba[i*4+3] = aMask ? (uint8_t)((p & aMask) >> sa) : 255;
        }
        return true;
    }
    return false;
}

bool loadImageFileRgba(const std::string& path, int& w, int& h,
                       std::vector<unsigned char>& rgba) {
    std::string ext = lowerExt(path);
    if (ext == ".dds") {
        return decodeDdsToRgba(path, w, h, rgba);
    }
    int comp = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!px || w <= 0 || h <= 0) {
        if (px) stbi_image_free(px);
        // Last resort: some pipelines hide DDS data behind other
        // extensions — sniff the magic.
        return decodeDdsToRgba(path, w, h, rgba);
    }
    rgba.assign(px, px + (size_t)w * h * 4);
    stbi_image_free(px);
    return true;
}

std::vector<std::pair<std::string, std::string>>
listModelTextureDependencies(const std::string& path) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, std::string>> deps;
    const std::string ext = lowerExt(path);
    const fs::path src_dir = fs::path(path).parent_path();

    // Reject absolute / parent-escaping relatives — fall back to basename
    // so the copy stays inside the import folder.
    auto sanitize_rel = [](const std::string& rel) -> std::string {
        fs::path p(rel);
        if (p.is_absolute() || rel.find("..") != std::string::npos)
            return p.filename().string();
        return p.generic_string();
    };
    auto add_dep = [&](const std::string& rel_raw,
                       const fs::path& resolved_src) {
        std::error_code ec;
        if (rel_raw.empty()) return;
        if (!fs::exists(resolved_src, ec) ||
            !fs::is_regular_file(resolved_src, ec)) return;
        const std::string rel = sanitize_rel(rel_raw);
        for (const auto& d : deps)
            if (d.first == rel) return;   // dedup
        deps.emplace_back(rel, resolved_src.string());
    };

    if (ext == ".gltf") {
        // GLB embeds its images; only .gltf references external files.
        tinygltf::Model    model;
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        if (!loader.LoadASCIIFromFile(&model, &err, &warn, path))
            return deps;
        for (const auto& img : model.images) {
            if (img.uri.empty()) continue;
            if (img.uri.rfind("data:", 0) == 0) continue;   // embedded
            add_dep(img.uri, src_dir / fs::path(img.uri));
        }
    } else if (ext == ".fbx") {
        ufbx_load_opts opts{};
        ufbx_error error;
        ufbx_scene* scene = ufbx_load_file(path.c_str(), &opts, &error);
        if (!scene) return deps;
        for (size_t i = 0; i < scene->textures.count; ++i) {
            const ufbx_texture* tex = scene->textures.data[i];
            if (tex->content.size > 0) continue;   // embedded — no file
            std::string rel;
            if (tex->relative_filename.data &&
                tex->relative_filename.length > 0)
                rel = tex->relative_filename.data;
            else if (tex->filename.data && tex->filename.length > 0)
                rel = fs::path(tex->filename.data).filename().string();
            if (rel.empty()) continue;
            // Resolve the source file: ufbx's filename (already resolved
            // against the model's own location), else absolute, else
            // model_dir + relative.
            fs::path resolved;
            std::error_code ec;
            if (tex->filename.data && tex->filename.length > 0 &&
                fs::exists(fs::path(tex->filename.data), ec)) {
                resolved = fs::path(tex->filename.data);
            } else if (tex->absolute_filename.data &&
                       tex->absolute_filename.length > 0 &&
                       fs::exists(fs::path(tex->absolute_filename.data),
                                  ec)) {
                resolved = fs::path(tex->absolute_filename.data);
            } else {
                resolved = src_dir / fs::path(rel);
            }
            add_dep(rel, resolved);
        }
        ufbx_free_scene(scene);
    } else if (ext == ".obj") {
        // The .mtl companion is copied by the importer already; pulling the
        // map_Kd files would need an .mtl parse — punt for now.
    }
    return deps;
}

bool modelHasSkin(const std::string& path) {
    const std::string ext = lowerExt(path);
    if (ext == ".gltf" || ext == ".glb") {
        tinygltf::Model    model;
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        const bool ok = (ext == ".glb")
            ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
            : loader.LoadASCIIFromFile(&model, &err, &warn, path);
        if (!ok) return false;
        return !model.skins.empty();
    }
    if (ext == ".fbx") {
        ufbx_load_opts opts{};
        ufbx_error error;
        ufbx_scene* scene = ufbx_load_file(path.c_str(), &opts, &error);
        if (!scene) return false;
        const bool has_skin = scene->skin_deformers.count > 0;
        ufbx_free_scene(scene);
        return has_skin;
    }
    return false;
}

std::vector<std::pair<std::string, std::string>>
listModelFileDependencies(const std::string& path) {
    namespace fs = std::filesystem;
    auto deps = listModelTextureDependencies(path);
    const std::string ext = lowerExt(path);
    if (ext == ".gltf") {
        // Binary buffer files (.bin) — same sanitize/dedup rules as the
        // texture dependencies.
        const fs::path src_dir = fs::path(path).parent_path();
        tinygltf::Model    model;
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        if (loader.LoadASCIIFromFile(&model, &err, &warn, path)) {
            for (const auto& buf : model.buffers) {
                if (buf.uri.empty()) continue;
                if (buf.uri.rfind("data:", 0) == 0) continue;   // embedded
                fs::path rel(buf.uri);
                std::string rel_s =
                    (rel.is_absolute() ||
                     buf.uri.find("..") != std::string::npos)
                        ? rel.filename().string()
                        : rel.generic_string();
                const fs::path resolved = src_dir / fs::path(buf.uri);
                std::error_code ec;
                if (rel_s.empty() || !fs::exists(resolved, ec) ||
                    !fs::is_regular_file(resolved, ec))
                    continue;
                bool dup = false;
                for (const auto& d : deps)
                    if (d.first == rel_s) { dup = true; break; }
                if (!dup) deps.emplace_back(rel_s, resolved.string());
            }
        }
    }
    return deps;
}

bool readRwObjRef(const std::string& rwobj_path,
                  std::string& out_source_path,
                  int&         out_sub_index,
                  std::string& out_name,
                  std::string& out_geo_path) {
    out_source_path.clear();
    out_sub_index = -1;
    out_name.clear();
    out_geo_path.clear();

    std::ifstream in(rwobj_path);
    if (!in) return false;

    bool is_rwobj = false;
    std::string source_rel;
    std::string geo_rel;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("rwobj=", 0) == 0) {
            is_rwobj = true;
        } else if (line.rfind("source=", 0) == 0) {
            source_rel = line.substr(7);
        } else if (line.rfind("node=", 0) == 0) {
            out_sub_index = std::atoi(line.c_str() + 5);
        } else if (line.rfind("name=", 0) == 0) {
            out_name = line.substr(5);
        } else if (line.rfind("geo=", 0) == 0) {
            geo_rel = line.substr(4);
        }
    }
    if (!is_rwobj || out_sub_index < 0) return false;

    // Resolve `source` / `geo` relative to the .rwobj's own folder (an
    // absolute `source` — the import-time original — passes through as-is).
    // A missing source is tolerated as long as the baked geo exists: the
    // preview runs entirely from the render-ready data.
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::path(rwobj_path).parent_path();
    if (!source_rel.empty()) {
        fs::path resolved =
            (base / fs::path(source_rel)).lexically_normal();
        if (fs::exists(resolved, ec)) out_source_path = resolved.string();
    }
    if (!geo_rel.empty()) {
        fs::path g = (base / fs::path(geo_rel)).lexically_normal();
        if (fs::exists(g, ec)) out_geo_path = g.string();
    }
    return !out_source_path.empty() || !out_geo_path.empty();
}

// ── Render-ready formats (.rwtex / .rwgeo) ──────────────────────────────────
namespace {

constexpr char kRwTexMagic[8] = {'R','W','T','E','X','0','0','1'};
constexpr char kRwGeoMagic[8]  = {'R','W','G','E','O','0','0','1'};
constexpr char kRwGeoMagic2[8] = {'R','W','G','E','O','0','0','2'};
constexpr char kRwGeoMagic3[8] = {'R','W','G','E','O','0','0','3'};
// v4 = v3 + skinning: flags bit1 = has_skin → a joint table (hier node +
// inverse bind per joint) after the section table, and per-vertex
// u16vec4 joints + vec4 weights blobs between uvs and indices.
constexpr char kRwGeoMagic4[8] = {'R','W','G','E','O','0','0','4'};
// v5 = v4 + full PBR material refs: every section carries TWO more
// texlen+rel pairs after the albedo one — normal map, then
// metallic-roughness map (either may be empty).
constexpr char kRwGeoMagic5[8] = {'R','W','G','E','O','0','0','5'};
constexpr char kRwHierMagic[8] = {'R','W','H','I','E','R','0','1'};
constexpr char kRwAnimMagic[8] = {'R','W','A','N','I','M','0','1'};

template <typename T>
void wrPod(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
bool rdPod(std::ifstream& f, T& v) {
    return (bool)f.read(reinterpret_cast<char*>(&v), sizeof(T));
}

std::string sanitizeFileName(std::string s) {
    for (auto& ch : s) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '*' ||
            ch == '?' || ch == '"' || ch == '<' || ch == '>' || ch == '|')
            ch = '_';
    }
    if (s.empty()) s = "unnamed";
    return s;
}

// Simple ½× box downsample (RGBA8), edge-safe for odd dimensions.
void halveRgba8(const std::vector<unsigned char>& src, int sw, int sh,
                std::vector<unsigned char>& dst, int& dw, int& dh) {
    dw = std::max(1, sw / 2);
    dh = std::max(1, sh / 2);
    dst.assign((size_t)dw * dh * 4, 0);
    for (int y = 0; y < dh; ++y) {
        const int sy0 = std::min(y * 2,     sh - 1);
        const int sy1 = std::min(y * 2 + 1, sh - 1);
        for (int x = 0; x < dw; ++x) {
            const int sx0 = std::min(x * 2,     sw - 1);
            const int sx1 = std::min(x * 2 + 1, sw - 1);
            const unsigned char* p00 = src.data() + ((size_t)sy0 * sw + sx0) * 4;
            const unsigned char* p01 = src.data() + ((size_t)sy0 * sw + sx1) * 4;
            const unsigned char* p10 = src.data() + ((size_t)sy1 * sw + sx0) * 4;
            const unsigned char* p11 = src.data() + ((size_t)sy1 * sw + sx1) * 4;
            unsigned char* d = dst.data() + ((size_t)y * dw + x) * 4;
            for (int c = 0; c < 4; ++c) {
                d[c] = (unsigned char)((p00[c] + p01[c] + p10[c] + p11[c] + 2) / 4);
            }
        }
    }
}

// .rwtex writer — bakes the ENGINE format:
//   format 1 = pre-encoded Virtual Texture BC7 tile cache + small RGBA8
//   preview (≤256 px, for thumbnails / Debug Display) + full-res alpha
//   plane (cutout textures only, for the forward/shadow mask companion).
// The runtime VT registration adopts the blob directly — no CPU BC7
// encode at load.  Falls back to legacy format 0 (raw RGBA8) when the
// encode fails.
bool writeRwTex(const std::string& path, int w, int h,
                const std::vector<unsigned char>& rgba) {
    namespace sr = engine::scene_rendering;

    // BC7 VT tile cache (the expensive part — runs on the import
    // worker thread, multi-threaded internally).
    std::vector<uint8_t> blob;
    const bool bc7_ok = sr::VirtualTextureManager::encodeAlbedoTileCacheCpu(
        rgba.data(), (uint32_t)w, (uint32_t)h, blob);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(kRwTexMagic, 8);
    wrPod(f, (uint32_t)w);
    wrPod(f, (uint32_t)h);

    if (!bc7_ok) {
        // Legacy fallback: raw RGBA8.
        wrPod(f, (uint32_t)0);   // format 0 = RGBA8
        wrPod(f, (uint32_t)1);   // mip count
        f.write(reinterpret_cast<const char*>(rgba.data()),
                (std::streamsize)((size_t)w * h * 4));
        return (bool)f;
    }

    wrPod(f, (uint32_t)1);       // format 1 = BC7 VT tile cache
    {
        const uint32_t px = ((uint32_t)w + sr::kVtPageSize - 1) / sr::kVtPageSize;
        const uint32_t py = ((uint32_t)h + sr::kVtPageSize - 1) / sr::kVtPageSize;
        wrPod(f, sr::vtComputeMipCount(px, py));   // mip count
    }

    // Small RGBA8 preview (halve until longest side ≤ 256).
    std::vector<unsigned char> preview = rgba;
    int pw = w, ph = h;
    while (std::max(pw, ph) > 256) {
        std::vector<unsigned char> next;
        int nw = 0, nh = 0;
        halveRgba8(preview, pw, ph, next, nw, nh);
        preview = std::move(next);
        pw = nw; ph = nh;
    }
    wrPod(f, (uint32_t)pw);
    wrPod(f, (uint32_t)ph);
    f.write(reinterpret_cast<const char*>(preview.data()),
            (std::streamsize)preview.size());

    // Full-res alpha plane — only when the texture has real cutout.
    bool has_alpha = false;
    for (size_t i = 3; i < rgba.size(); i += 4) {
        if (rgba[i] < 250) { has_alpha = true; break; }
    }
    wrPod(f, (uint32_t)(has_alpha ? 1 : 0));
    if (has_alpha) {
        std::vector<unsigned char> alpha((size_t)w * h);
        for (size_t i = 0, n = alpha.size(); i < n; ++i) {
            alpha[i] = rgba[i * 4 + 3];
        }
        f.write(reinterpret_cast<const char*>(alpha.data()),
                (std::streamsize)alpha.size());
    }

    // Tile geometry stamp (loader validates against engine constants).
    wrPod(f, sr::kVtPageSize);
    wrPod(f, sr::kVtTileBorder);

    // The BC7 tile blob.
    wrPod(f, (uint64_t)blob.size());
    f.write(reinterpret_cast<const char*>(blob.data()),
            (std::streamsize)blob.size());
    return (bool)f;
}

} // anonymous namespace

// Public (declared in model_inspect.h).  Format 0 → full-res RGBA8;
// format 1 (BC7 VT tile cache) → the embedded RGBA8 preview.
bool readRwTex(const std::string& path, int& w, int& h,
               std::vector<unsigned char>& rgba) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    if (!f.read(magic, 8) || std::memcmp(magic, kRwTexMagic, 8) != 0)
        return false;
    uint32_t uw = 0, uh = 0, fmt = 0, mips = 0;
    if (!rdPod(f, uw) || !rdPod(f, uh) || !rdPod(f, fmt) || !rdPod(f, mips))
        return false;
    if (uw == 0 || uh == 0 || uw > 16384 || uh > 16384)
        return false;
    if (fmt == 0) {
        rgba.resize((size_t)uw * uh * 4);
        if (!f.read(reinterpret_cast<char*>(rgba.data()),
                    (std::streamsize)rgba.size()))
            return false;
        w = (int)uw;
        h = (int)uh;
        return true;
    }
    if (fmt == 1) {
        uint32_t pw = 0, ph = 0;
        if (!rdPod(f, pw) || !rdPod(f, ph) ||
            pw == 0 || ph == 0 || pw > 16384 || ph > 16384)
            return false;
        rgba.resize((size_t)pw * ph * 4);
        if (!f.read(reinterpret_cast<char*>(rgba.data()),
                    (std::streamsize)rgba.size()))
            return false;
        w = (int)pw;
        h = (int)ph;
        return true;
    }
    return false;
}

// Full read — see header doc.
bool readRwTexBaked(const std::string& path, RwTexBaked& out) {
    out = RwTexBaked{};
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    if (!f.read(magic, 8) || std::memcmp(magic, kRwTexMagic, 8) != 0)
        return false;
    uint32_t uw = 0, uh = 0, fmt = 0, mips = 0;
    if (!rdPod(f, uw) || !rdPod(f, uh) || !rdPod(f, fmt) || !rdPod(f, mips))
        return false;
    if (uw == 0 || uh == 0 || uw > 16384 || uh > 16384)
        return false;
    out.w = (int)uw;
    out.h = (int)uh;

    if (fmt == 0) {
        // Legacy raw RGBA8 — full pixels double as the "preview".
        out.preview_rgba.resize((size_t)uw * uh * 4);
        if (!f.read(reinterpret_cast<char*>(out.preview_rgba.data()),
                    (std::streamsize)out.preview_rgba.size()))
            return false;
        out.preview_w = (int)uw;
        out.preview_h = (int)uh;
        return true;
    }
    if (fmt != 1) return false;

    uint32_t pw = 0, ph = 0;
    if (!rdPod(f, pw) || !rdPod(f, ph) ||
        pw == 0 || ph == 0 || pw > 16384 || ph > 16384)
        return false;
    out.preview_rgba.resize((size_t)pw * ph * 4);
    if (!f.read(reinterpret_cast<char*>(out.preview_rgba.data()),
                (std::streamsize)out.preview_rgba.size()))
        return false;
    out.preview_w = (int)pw;
    out.preview_h = (int)ph;

    uint32_t has_alpha = 0;
    if (!rdPod(f, has_alpha)) return false;
    if (has_alpha) {
        out.alpha.resize((size_t)uw * uh);
        if (!f.read(reinterpret_cast<char*>(out.alpha.data()),
                    (std::streamsize)out.alpha.size()))
            return false;
    }

    uint32_t page_size = 0, tile_border = 0;
    uint64_t blob_bytes = 0;
    if (!rdPod(f, page_size) || !rdPod(f, tile_border) ||
        !rdPod(f, blob_bytes))
        return false;
    // Tile-geometry + size validation: a mismatch (engine constants
    // changed since the bake) silently drops the blob — the texture
    // then registers from the preview... which is low-res, so warn.
    namespace sr = engine::scene_rendering;
    const uint64_t expected =
        sr::VirtualTextureManager::albedoTileCacheBytes(uw, uh);
    if (page_size != sr::kVtPageSize ||
        tile_border != sr::kVtTileBorder ||
        blob_bytes != expected || blob_bytes == 0) {
        std::cout << "[rwtex] '" << path << "': baked BC7 tile cache "
                     "doesn't match engine VT constants — re-import to "
                     "re-bake." << std::endl;
        return true;   // still usable (preview + alpha)
    }
    auto blob = std::make_shared<std::vector<uint8_t>>(blob_bytes);
    if (!f.read(reinterpret_cast<char*>(blob->data()),
                (std::streamsize)blob_bytes))
        return false;
    out.bc7_tiles = std::move(blob);
    return true;
}

namespace {

// One material section as written into a .rwgeo (texture by group-relative
// .rwtex path; empty = untextured).
struct GeoSectionOut {
    uint32_t    first_index = 0;
    uint32_t    index_count = 0;
    glm::vec4   base_color  = glm::vec4(1.0f);
    float       metallic    = 0.0f;
    float       roughness   = 0.6f;
    std::string tex_rel;   // albedo
    std::string nrm_rel;   // normal map (may be empty)
    std::string mr_rel;    // metallic-roughness map (may be empty)
};

// Dedup triangle-soup geometry by exact (position, normal, uv) bits.
// The source-parse helpers (appendFbxMesh / appendGltfPrim) emit one
// vertex PER TRIANGLE CORNER — fine for the CPU preview, terrible for
// the GPU: every index is unique, so the post-transform vertex cache
// never hits and the hardware shades ~3× the vertices the engine's own
// FBX loader produces for identical content (measured as the
// "placed scene slower than New Game" regression).  Index ORDER is
// preserved (only values change), so section ranges stay valid.
void dedupSoupVertices(std::vector<glm::vec3>& positions,
                       std::vector<glm::vec3>& normals,
                       std::vector<glm::vec2>& uvs,
                       std::vector<uint32_t>&  indices,
                       std::vector<glm::u16vec4>* joints = nullptr,
                       std::vector<glm::vec4>*    weights = nullptr,
                       std::vector<glm::vec4>*    closeness = nullptr,
                       // 8-bone debug path: second skin set, rides parallel.
                       std::vector<glm::u16vec4>* joints1 = nullptr,
                       std::vector<glm::vec4>*    weights1 = nullptr,
                       std::vector<glm::vec4>*    closeness1 = nullptr) {
    const size_t vc = positions.size();
    if (vc == 0) return;
    const bool has_n  = normals.size() == vc;
    const bool has_uv = uvs.size() == vc;
    const bool has_skin =
        joints && weights &&
        joints->size() == vc && weights->size() == vc;
    // Closeness rides parallel to weights (first-occurrence wins); it is NOT
    // part of the dedup key — identical joints/weights imply ~identical closeness.
    const bool has_close = has_skin && closeness && closeness->size() == vc;
    const bool has_skin1 = has_skin && joints1 && weights1 &&
        joints1->size() == vc && weights1->size() == vc;
    const bool has_close1 =
        has_skin1 && closeness1 && closeness1->size() == vc;

    auto vert_hash = [&](uint32_t v) -> uint64_t {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](const void* p, size_t n) {
            const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
            for (size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 1099511628211ull;
            }
        };
        mix(&positions[v], sizeof(glm::vec3));
        if (has_n)  mix(&normals[v], sizeof(glm::vec3));
        if (has_uv) mix(&uvs[v], sizeof(glm::vec2));
        if (has_skin) {
            mix(&(*joints)[v], sizeof(glm::u16vec4));
            mix(&(*weights)[v], sizeof(glm::vec4));
        }
        if (has_skin1) {
            mix(&(*joints1)[v], sizeof(glm::u16vec4));
            mix(&(*weights1)[v], sizeof(glm::vec4));
        }
        return h;
    };

    std::vector<glm::vec3> dpos;  dpos.reserve(vc / 2);
    std::vector<glm::vec3> dnrm;  if (has_n)  dnrm.reserve(vc / 2);
    std::vector<glm::vec2> duv;   if (has_uv) duv.reserve(vc / 2);
    std::vector<glm::u16vec4> djnt;
    std::vector<glm::vec4>    dwgt;
    std::vector<glm::vec4>    dcls;
    std::vector<glm::u16vec4> djnt1;
    std::vector<glm::vec4>    dwgt1;
    std::vector<glm::vec4>    dcls1;
    if (has_skin) { djnt.reserve(vc / 2); dwgt.reserve(vc / 2); }
    if (has_close) dcls.reserve(vc / 2);
    if (has_skin1) { djnt1.reserve(vc / 2); dwgt1.reserve(vc / 2); }
    if (has_close1) dcls1.reserve(vc / 2);
    std::vector<uint32_t> remap(vc);
    std::unordered_map<uint64_t, std::vector<uint32_t>> buckets;
    buckets.reserve(vc);

    for (uint32_t v = 0; v < (uint32_t)vc; ++v) {
        auto& cands = buckets[vert_hash(v)];
        uint32_t found = 0xFFFFFFFFu;
        for (uint32_t c : cands) {
            if (dpos[c] == positions[v] &&
                (!has_n  || dnrm[c] == normals[v]) &&
                (!has_uv || duv[c] == uvs[v]) &&
                (!has_skin ||
                 (djnt[c] == (*joints)[v] && dwgt[c] == (*weights)[v])) &&
                (!has_skin1 ||
                 (djnt1[c] == (*joints1)[v] && dwgt1[c] == (*weights1)[v]))) {
                found = c;
                break;
            }
        }
        if (found == 0xFFFFFFFFu) {
            found = (uint32_t)dpos.size();
            dpos.push_back(positions[v]);
            if (has_n)  dnrm.push_back(normals[v]);
            if (has_uv) duv.push_back(uvs[v]);
            if (has_skin) {
                djnt.push_back((*joints)[v]);
                dwgt.push_back((*weights)[v]);
            }
            if (has_close) dcls.push_back((*closeness)[v]);
            if (has_skin1) {
                djnt1.push_back((*joints1)[v]);
                dwgt1.push_back((*weights1)[v]);
            }
            if (has_close1) dcls1.push_back((*closeness1)[v]);
            cands.push_back(found);
        }
        remap[v] = found;
    }

    for (auto& ix : indices) {
        if (ix < vc) ix = remap[ix];
    }
    positions = std::move(dpos);
    if (has_n)  normals = std::move(dnrm);
    if (has_uv) uvs = std::move(duv);
    if (has_skin) {
        *joints  = std::move(djnt);
        *weights = std::move(dwgt);
    }
    if (has_close) *closeness = std::move(dcls);
    if (has_skin1) {
        *joints1  = std::move(djnt1);
        *weights1 = std::move(dwgt1);
    }
    if (has_close1) *closeness1 = std::move(dcls1);
}

bool writeRwGeo(const std::string& path, const ModelPreviewData& d,
                const std::vector<GeoSectionOut>& sections,
                const glm::mat4& node_to_world) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    bool any_tex = false;
    for (const auto& s : sections) {
        any_tex |= !s.tex_rel.empty() || !s.nrm_rel.empty() ||
                   !s.mr_rel.empty();
    }
    const bool has_uv = any_tex && d.uvs.size() == d.positions.size();
    const bool has_skin =
        d.joints.size() == d.positions.size() &&
        d.weights.size() == d.positions.size() &&
        !d.skin_joint_nodes.empty() &&
        d.skin_joint_nodes.size() == d.skin_inverse_bind.size();
    // Optional baked closeness (auto-rig distance field), parallel to weights.
    const bool has_close = has_skin &&
        d.closeness.size() == d.positions.size();
    // 8-bone debug path: second skin set (influences 4..7).
    const bool has_skin1 = has_skin &&
        d.joints1.size() == d.positions.size() &&
        d.weights1.size() == d.positions.size();
    const bool has_close1 = has_skin1 &&
        d.closeness1.size() == d.positions.size();

    // Dedup the parse helpers' triangle soup before writing — indexed,
    // shared vertices are what the GPU (and the cluster builder) want.
    std::vector<glm::vec3>    positions = d.positions;
    std::vector<glm::vec3>    normals   = d.normals;
    std::vector<glm::vec2>    uvs       = has_uv ? d.uvs
                                                 : std::vector<glm::vec2>();
    std::vector<uint32_t>     indices   = d.indices;
    std::vector<glm::u16vec4> joints    = has_skin
        ? d.joints  : std::vector<glm::u16vec4>();
    std::vector<glm::vec4>    weights   = has_skin
        ? d.weights : std::vector<glm::vec4>();
    std::vector<glm::vec4>    closeness = has_close
        ? d.closeness : std::vector<glm::vec4>();
    std::vector<glm::u16vec4> joints1   = has_skin1
        ? d.joints1  : std::vector<glm::u16vec4>();
    std::vector<glm::vec4>    weights1  = has_skin1
        ? d.weights1 : std::vector<glm::vec4>();
    std::vector<glm::vec4>    closeness1 = has_close1
        ? d.closeness1 : std::vector<glm::vec4>();
    dedupSoupVertices(positions, normals, uvs, indices,
                      has_skin ? &joints : nullptr,
                      has_skin ? &weights : nullptr,
                      has_close ? &closeness : nullptr,
                      has_skin1 ? &joints1 : nullptr,
                      has_skin1 ? &weights1 : nullptr,
                      has_close1 ? &closeness1 : nullptr);

    // v5: full PBR section refs (albedo + normal + metallic-roughness);
    // flags bit1 marks the optional skin blobs.  Older files (v1-v4)
    // remain readable.
    f.write(kRwGeoMagic5, 8);
    wrPod(f, (uint32_t)positions.size());
    wrPod(f, (uint32_t)indices.size());
    // bit3 (8u) = second skin set blobs (8-bone debug), bit4 (16u) =
    // second closeness blob.  Readers older than the 8-bone path ignore
    // unknown bits only if they're clear — engine and bake ship together.
    wrPod(f, (uint32_t)((has_uv ? 1u : 0u) | (has_skin ? 2u : 0u) |
                        (has_close ? 4u : 0u) | (has_skin1 ? 8u : 0u) |
                        (has_close1 ? 16u : 0u)));
    // v3: geometry is NODE-LOCAL; this matrix places it in source-world
    // space (standalone consumers apply it; hierarchical renderers compose
    // the rwhier chain instead).
    wrPod(f, node_to_world);
    wrPod(f, (uint32_t)sections.size());
    for (const auto& s : sections) {
        wrPod(f, s.first_index);
        wrPod(f, s.index_count);
        wrPod(f, s.base_color);
        wrPod(f, s.metallic);
        wrPod(f, s.roughness);
        wrPod(f, (uint32_t)s.tex_rel.size());
        f.write(s.tex_rel.data(), (std::streamsize)s.tex_rel.size());
        // v5: normal + metallic-roughness refs (empty = none).
        wrPod(f, (uint32_t)s.nrm_rel.size());
        f.write(s.nrm_rel.data(), (std::streamsize)s.nrm_rel.size());
        wrPod(f, (uint32_t)s.mr_rel.size());
        f.write(s.mr_rel.data(), (std::streamsize)s.mr_rel.size());
    }
    // Skin joint table (v4): per joint, the hierarchy.rwhier node index
    // it binds to + the inverse bind matrix.
    if (has_skin) {
        wrPod(f, (uint32_t)d.skin_joint_nodes.size());
        for (size_t j = 0; j < d.skin_joint_nodes.size(); ++j) {
            wrPod(f, d.skin_joint_nodes[j]);
            wrPod(f, d.skin_inverse_bind[j]);
        }
    }

    f.write(reinterpret_cast<const char*>(positions.data()),
            (std::streamsize)(positions.size() * sizeof(glm::vec3)));
    f.write(reinterpret_cast<const char*>(normals.data()),
            (std::streamsize)(normals.size() * sizeof(glm::vec3)));
    if (has_uv) {
        f.write(reinterpret_cast<const char*>(uvs.data()),
                (std::streamsize)(uvs.size() * sizeof(glm::vec2)));
    }
    if (has_skin) {
        f.write(reinterpret_cast<const char*>(joints.data()),
                (std::streamsize)(joints.size() * sizeof(glm::u16vec4)));
        f.write(reinterpret_cast<const char*>(weights.data()),
                (std::streamsize)(weights.size() * sizeof(glm::vec4)));
    }
    // Closeness block (flag bit 4): parallel to weights, written between the
    // weights and the indices so older readers that know bit2 but not bit4 stay
    // in sync only when bit4 is clear (it isn't set for legacy assets).
    if (has_close) {
        f.write(reinterpret_cast<const char*>(closeness.data()),
                (std::streamsize)(closeness.size() * sizeof(glm::vec4)));
    }
    // 8-bone debug path: second skin set blobs (flag bits 8/16), between
    // the closeness block and the indices.
    if (has_skin1) {
        f.write(reinterpret_cast<const char*>(joints1.data()),
                (std::streamsize)(joints1.size() * sizeof(glm::u16vec4)));
        f.write(reinterpret_cast<const char*>(weights1.data()),
                (std::streamsize)(weights1.size() * sizeof(glm::vec4)));
    }
    if (has_close1) {
        f.write(reinterpret_cast<const char*>(closeness1.data()),
                (std::streamsize)(closeness1.size() * sizeof(glm::vec4)));
    }
    f.write(reinterpret_cast<const char*>(indices.data()),
            (std::streamsize)(indices.size() * sizeof(uint32_t)));
    return (bool)f;
}

bool writeRwHier(const std::string& path,
                 const std::vector<RwHierNode>& nodes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(kRwHierMagic, 8);
    wrPod(f, (uint32_t)nodes.size());
    for (const auto& n : nodes) {
        wrPod(f, n.parent);
        wrPod(f, n.mesh_ordinal);
        wrPod(f, n.local);
        wrPod(f, (uint32_t)n.name.size());
        f.write(n.name.data(), (std::streamsize)n.name.size());
    }
    return (bool)f;
}

// Write every clip to one animation.rwanim.  Layout (after the 8-byte magic):
//   u32 clip_count
//   per clip: u32 name_len, char[name_len], f32 duration, u32 channel_count
//     per channel: i32 node, u8 path, u8 step, u32 key_count, u8 comps(3|4),
//                  f32 times[key_count], f32 values[key_count*comps]
bool writeRwAnim(const std::string& path,
                 const std::vector<RwAnimClip>& clips) {
    if (clips.empty()) return false;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(kRwAnimMagic, 8);
    wrPod(f, (uint32_t)clips.size());
    for (const auto& clip : clips) {
        wrPod(f, (uint32_t)clip.name.size());
        f.write(clip.name.data(), (std::streamsize)clip.name.size());
        wrPod(f, clip.duration);
        wrPod(f, (uint32_t)clip.channels.size());
        for (const auto& ch : clip.channels) {
            const uint8_t comps =
                (ch.path == RwAnimPath::kRotation) ? 4u : 3u;
            const uint32_t kc = (uint32_t)ch.times.size();
            wrPod(f, ch.node);
            wrPod(f, (uint8_t)ch.path);
            wrPod(f, ch.step);
            wrPod(f, kc);
            wrPod(f, comps);
            if (kc) {
                f.write(reinterpret_cast<const char*>(ch.times.data()),
                        (std::streamsize)(kc * sizeof(float)));
                for (const auto& v : ch.values)
                    f.write(reinterpret_cast<const char*>(&v),
                            (std::streamsize)(comps * sizeof(float)));
            }
        }
    }
    return (bool)f;
}

} // anonymous namespace

// Public wrapper over the bake-side soup dedup — see header doc.
void dedupModelVertices(ModelPreviewData& d) {
    // uvs / skin attributes may legitimately be absent — the helper
    // tolerates any combination of matching array sizes.
    const bool has_skin =
        d.joints.size() == d.positions.size() &&
        d.weights.size() == d.positions.size();
    const bool has_close = has_skin &&
        d.closeness.size() == d.positions.size();
    const bool has_skin1 = has_skin &&
        d.joints1.size() == d.positions.size() &&
        d.weights1.size() == d.positions.size();
    const bool has_close1 = has_skin1 &&
        d.closeness1.size() == d.positions.size();
    dedupSoupVertices(d.positions, d.normals, d.uvs, d.indices,
                      has_skin ? &d.joints : nullptr,
                      has_skin ? &d.weights : nullptr,
                      has_close ? &d.closeness : nullptr,
                      has_skin1 ? &d.joints1 : nullptr,
                      has_skin1 ? &d.weights1 : nullptr,
                      has_close1 ? &d.closeness1 : nullptr);
}

bool loadRwHier(const std::string& path, std::vector<RwHierNode>& out) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    if (!f.read(magic, 8) || std::memcmp(magic, kRwHierMagic, 8) != 0)
        return false;
    uint32_t count = 0;
    if (!rdPod(f, count) || count == 0 || count > 1'000'000u) return false;
    out.resize(count);
    for (auto& n : out) {
        uint32_t name_len = 0;
        if (!rdPod(f, n.parent) || !rdPod(f, n.mesh_ordinal) ||
            !rdPod(f, n.local) || !rdPod(f, name_len) ||
            name_len > 4096u) {
            out.clear();
            return false;
        }
        n.name.resize(name_len);
        if (name_len && !f.read(n.name.data(), name_len)) {
            out.clear();
            return false;
        }
    }
    return true;
}

bool loadRwAnim(const std::string& path, std::vector<RwAnimClip>& out) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    if (!f.read(magic, 8) || std::memcmp(magic, kRwAnimMagic, 8) != 0)
        return false;
    uint32_t clip_count = 0;
    if (!rdPod(f, clip_count) || clip_count > 100'000u) return false;
    out.resize(clip_count);
    for (auto& clip : out) {
        uint32_t name_len = 0;
        if (!rdPod(f, name_len) || name_len > 4096u) { out.clear(); return false; }
        clip.name.resize(name_len);
        if (name_len && !f.read(clip.name.data(), name_len)) {
            out.clear(); return false;
        }
        uint32_t ch_count = 0;
        if (!rdPod(f, clip.duration) || !rdPod(f, ch_count) ||
            ch_count > 1'000'000u) { out.clear(); return false; }
        clip.channels.resize(ch_count);
        for (auto& ch : clip.channels) {
            uint32_t kc = 0;
            uint8_t  path = 0, comps = 0;
            if (!rdPod(f, ch.node) || !rdPod(f, path) || !rdPod(f, ch.step) ||
                !rdPod(f, kc) || !rdPod(f, comps) ||
                kc > 10'000'000u || (comps != 3 && comps != 4)) {
                out.clear(); return false;
            }
            ch.path = (RwAnimPath)path;
            ch.times.resize(kc);
            ch.values.resize(kc);
            if (kc) {
                if (!f.read(reinterpret_cast<char*>(ch.times.data()),
                            (std::streamsize)(kc * sizeof(float)))) {
                    out.clear(); return false;
                }
                for (auto& v : ch.values) {
                    v = glm::vec4(0.0f);
                    if (!f.read(reinterpret_cast<char*>(&v),
                                (std::streamsize)(comps * sizeof(float)))) {
                        out.clear(); return false;
                    }
                }
            }
        }
    }
    return true;
}

bool loadRwGeo(const std::string& rwgeo_path, ModelPreviewData& out,
               std::vector<std::string>* out_texture_paths,
               bool decode_textures) {
    namespace fs = std::filesystem;
    out = ModelPreviewData{};
    if (out_texture_paths) out_texture_paths->clear();
    std::ifstream f(rwgeo_path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    if (!f.read(magic, 8)) return false;
    const bool v5 = std::memcmp(magic, kRwGeoMagic5, 8) == 0;
    const bool v4 = std::memcmp(magic, kRwGeoMagic4, 8) == 0 || v5;
    const bool v3 = std::memcmp(magic, kRwGeoMagic3, 8) == 0 || v4;
    const bool v2 = std::memcmp(magic, kRwGeoMagic2, 8) == 0;
    const bool v1 = std::memcmp(magic, kRwGeoMagic,  8) == 0;
    if (!v1 && !v2 && !v3) return false;

    uint32_t vc = 0, ic = 0, flags = 0;
    if (!rdPod(f, vc) || !rdPod(f, ic) || !rdPod(f, flags)) return false;
    if (vc == 0 || ic < 3 || vc > 50'000'000u || ic > 150'000'000u)
        return false;
    const bool has_uv    = (flags & 1u) != 0;
    const bool has_skin  = v4 && (flags & 2u) != 0;
    const bool has_close = v4 && (flags & 4u) != 0;   // baked closeness block
    const bool has_skin1  = v4 && (flags & 8u) != 0;  // 8-bone second set
    const bool has_close1 = v4 && (flags & 16u) != 0;

    // v3+: node-local geometry + the node's world matrix (re-applied below
    // so standalone consumers keep seeing source-world coordinates).
    glm::mat4 node_to_world(1.0f);
    if (v3 && !rdPod(f, node_to_world)) return false;

    // Section table (v2) / single legacy material header (v1).
    struct SecIn {
        uint32_t first = 0, count = 0;
        glm::vec4 color = glm::vec4(1.0f);
        float metallic = 0.0f, roughness = 0.6f;
        std::string tex_rel;
        std::string nrm_rel;   // v5
        std::string mr_rel;    // v5
    };
    std::vector<SecIn> secs;
    if (v2 || v3) {
        uint32_t sc = 0;
        if (!rdPod(f, sc) || sc == 0 || sc > 100000u) return false;
        secs.resize(sc);
        for (auto& s : secs) {
            uint32_t texlen = 0;
            if (!rdPod(f, s.first) || !rdPod(f, s.count) ||
                !rdPod(f, s.color) || !rdPod(f, s.metallic) ||
                !rdPod(f, s.roughness) || !rdPod(f, texlen))
                return false;
            s.tex_rel.resize(texlen);
            if (texlen && !f.read(s.tex_rel.data(), texlen)) return false;
            if (v5) {
                uint32_t nlen = 0, mlen = 0;
                if (!rdPod(f, nlen)) return false;
                s.nrm_rel.resize(nlen);
                if (nlen && !f.read(s.nrm_rel.data(), nlen)) return false;
                if (!rdPod(f, mlen)) return false;
                s.mr_rel.resize(mlen);
                if (mlen && !f.read(s.mr_rel.data(), mlen)) return false;
            }
        }
    } else {
        SecIn s;
        uint32_t texlen = 0;
        if (!rdPod(f, s.color) || !rdPod(f, s.metallic) ||
            !rdPod(f, s.roughness) || !rdPod(f, texlen))
            return false;
        s.tex_rel.resize(texlen);
        if (texlen && !f.read(s.tex_rel.data(), texlen)) return false;
        s.first = 0;
        s.count = ic;
        secs.push_back(std::move(s));
    }

    // Skin joint table (v4, has_skin).
    if (has_skin) {
        uint32_t jc = 0;
        if (!rdPod(f, jc) || jc == 0 || jc > 65536u) return false;
        out.skin_joint_nodes.resize(jc);
        out.skin_inverse_bind.resize(jc);
        for (uint32_t j = 0; j < jc; ++j) {
            if (!rdPod(f, out.skin_joint_nodes[j]) ||
                !rdPod(f, out.skin_inverse_bind[j]))
                return false;
        }
    }

    out.positions.resize(vc);
    out.normals.resize(vc);
    if (!f.read(reinterpret_cast<char*>(out.positions.data()),
                (std::streamsize)(vc * sizeof(glm::vec3))))
        return false;
    if (!f.read(reinterpret_cast<char*>(out.normals.data()),
                (std::streamsize)(vc * sizeof(glm::vec3))))
        return false;
    if (has_uv) {
        out.uvs.resize(vc);
        if (!f.read(reinterpret_cast<char*>(out.uvs.data()),
                    (std::streamsize)(vc * sizeof(glm::vec2))))
            return false;
    }
    if (has_skin) {
        out.joints.resize(vc);
        out.weights.resize(vc);
        if (!f.read(reinterpret_cast<char*>(out.joints.data()),
                    (std::streamsize)(vc * sizeof(glm::u16vec4))))
            return false;
        if (!f.read(reinterpret_cast<char*>(out.weights.data()),
                    (std::streamsize)(vc * sizeof(glm::vec4))))
            return false;
    }
    if (has_close) {
        out.closeness.resize(vc);
        if (!f.read(reinterpret_cast<char*>(out.closeness.data()),
                    (std::streamsize)(vc * sizeof(glm::vec4))))
            return false;
    }
    // 8-bone debug path: second skin set blobs (flag bits 8/16).
    if (has_skin1) {
        out.joints1.resize(vc);
        out.weights1.resize(vc);
        if (!f.read(reinterpret_cast<char*>(out.joints1.data()),
                    (std::streamsize)(vc * sizeof(glm::u16vec4))))
            return false;
        if (!f.read(reinterpret_cast<char*>(out.weights1.data()),
                    (std::streamsize)(vc * sizeof(glm::vec4))))
            return false;
    }
    if (has_close1) {
        out.closeness1.resize(vc);
        if (!f.read(reinterpret_cast<char*>(out.closeness1.data()),
                    (std::streamsize)(vc * sizeof(glm::vec4))))
            return false;
    }
    out.indices.resize(ic);
    if (!f.read(reinterpret_cast<char*>(out.indices.data()),
                (std::streamsize)(ic * sizeof(uint32_t))))
        return false;

    // v3: re-apply the node's world transform so previews and placement see
    // source-world coordinates (the hierarchical renderer will compose the
    // rwhier chain instead and skip this).
    if (v3 && node_to_world != glm::mat4(1.0f)) {
        const glm::mat3 nm(node_to_world);
        for (auto& p : out.positions)
            p = glm::vec3(node_to_world * glm::vec4(p, 1.0f));
        for (auto& n : out.normals) {
            n = nm * n;
            const float l = glm::length(n);
            if (l > 1e-6f) n /= l;
        }
    }

    // Textures: group-relative .rwtex files (group = parent of objects/),
    // dedup'd by relative path.  Shared resolver for the albedo / normal /
    // metallic-roughness refs.
    const fs::path group = fs::path(rwgeo_path).parent_path().parent_path();
    std::unordered_map<std::string, int> tex_cache;
    auto resolve_tex = [&](const std::string& rel) -> int {
        if (rel.empty()) return -1;
        auto it = tex_cache.find(rel);
        if (it != tex_cache.end()) return it->second;
        int slot = -1;
        const std::string tex_path = (group / fs::path(rel)).string();
        if (decode_textures) {
            PreviewTexture pt;
            if (readRwTex(tex_path, pt.w, pt.h, pt.rgba)) {
                slot = (int)out.textures.size();
                out.textures.push_back(std::move(pt));
                if (out_texture_paths)
                    out_texture_paths->push_back(tex_path);
            } else {
                std::cout << "[rwgeo] texture missing: " << tex_path
                          << std::endl;
            }
        } else {
            // Paths-only mode: reserve the slot (empty entry) so the
            // index stays valid; the caller loads pixels on cache
            // misses via readRwTex.
            std::error_code tec;
            if (fs::exists(tex_path, tec)) {
                slot = (int)out.textures.size();
                out.textures.emplace_back();
                if (out_texture_paths)
                    out_texture_paths->push_back(tex_path);
            } else {
                std::cout << "[rwgeo] texture missing: " << tex_path
                          << std::endl;
            }
        }
        tex_cache.emplace(rel, slot);
        return slot;
    };
    for (const auto& s : secs) {
        PreviewSection sec;
        sec.first_index = s.first;
        sec.index_count = s.count;
        sec.base_color  = s.color;
        sec.metallic    = s.metallic;
        sec.roughness   = s.roughness;
        sec.tex_index   = resolve_tex(s.tex_rel);
        sec.nrm_index   = resolve_tex(s.nrm_rel);
        sec.mr_index    = resolve_tex(s.mr_rel);
        out.sections.push_back(sec);
    }
    if (out.textures.empty()) out.uvs.clear();
    return true;
}

bool bakeModelToRenderReady(
    const std::string& source_path,
    const std::string& group_dir,
    std::vector<BakedObject>& out_objects,
    const std::function<void(size_t, size_t)>& progress) {
    namespace fs = std::filesystem;
    out_objects.clear();
    const std::string ext = lowerExt(source_path);
    std::error_code ec;
    fs::create_directories(fs::path(group_dir) / "objects", ec);
    fs::create_directories(fs::path(group_dir) / "textures", ec);

    auto geo_rel_for = [&](size_t k, const std::string& name) {
        char pre[16];
        std::snprintf(pre, sizeof(pre), "%03u_", (unsigned)k);
        return std::string("objects/") + pre + sanitizeFileName(name) +
               ".rwgeo";
    };

    if (ext == ".gltf" || ext == ".glb") {
        tinygltf::Model    model;
        tinygltf::TinyGLTF loader;
        std::string err, warn;
        const bool ok = (ext == ".glb")
            ? loader.LoadBinaryFromFile(&model, &err, &warn, source_path)
            : loader.LoadASCIIFromFile(&model, &err, &warn, source_path);
        if (!ok) return false;

        std::vector<glm::mat4> node_world(model.nodes.size(),
                                          glm::mat4(1.0f));
        std::vector<int>       node_parent(model.nodes.size(), -1);
        {
            std::function<void(int, const glm::mat4&)> walk =
                [&](int idx, const glm::mat4& parent) {
                    node_world[idx] =
                        parent * gltfNodeLocal(model.nodes[idx]);
                    for (int c : model.nodes[idx].children) {
                        node_parent[c] = idx;
                        walk(c, node_world[idx]);
                    }
                };
            std::vector<bool> is_child(model.nodes.size(), false);
            for (const auto& n : model.nodes)
                for (int c : n.children) is_child[c] = true;
            for (int i = 0; i < (int)model.nodes.size(); ++i)
                if (!is_child[i]) walk(i, glm::mat4(1.0f));
        }

        // Texture bake cache: glTF image index → textures/<name>.rwtex.
        std::unordered_map<int, std::string> tex_cache;
        // Bake ONE glTF texture (by texture index) to textures/<name>.rwtex
        // — used for albedo, normal and metallic-roughness refs alike.
        auto bake_gltf_texture = [&](int tex_idx) -> std::string {
            if (tex_idx < 0 || tex_idx >= (int)model.textures.size())
                return {};
            const int src = model.textures[tex_idx].source;
            if (src < 0 || src >= (int)model.images.size()) return {};
            auto it = tex_cache.find(src);
            if (it != tex_cache.end()) return it->second;
            const auto& img = model.images[src];
            if (img.image.empty() || img.width <= 0 || img.height <= 0 ||
                (img.component != 3 && img.component != 4)) return {};
            std::vector<unsigned char> rgba((size_t)img.width *
                                            img.height * 4);
            for (size_t i = 0; i < (size_t)img.width * img.height; ++i) {
                const uint8_t* sp = img.image.data() + i * img.component;
                uint8_t* dp = rgba.data() + i * 4;
                dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
                dp[3] = (img.component == 4) ? sp[3] : 255;
            }
            std::string nm = img.uri.empty()
                ? ("image" + std::to_string(src))
                : fs::path(img.uri).stem().string();
            const std::string rel =
                "textures/" + sanitizeFileName(nm) + ".rwtex";
            if (!writeRwTex((fs::path(group_dir) / rel).string(),
                            img.width, img.height, rgba)) return {};
            // Viewable copy of EXACTLY what was baked (orientation check +
            // Content Browser thumbnail).
            stbi_write_png(
                ((fs::path(group_dir) / rel).string() + ".png").c_str(),
                img.width, img.height, 4, rgba.data(), img.width * 4);
            tex_cache.emplace(src, rel);
            return rel;
        };

        size_t total = 0;
        for (const auto& n : model.nodes) if (n.mesh >= 0) ++total;
        int k = 0;
        for (int i = 0; i < (int)model.nodes.size(); ++i) {
            if (model.nodes[i].mesh < 0) continue;
            std::string name = model.nodes[i].name;
            if (name.empty()) name = "node " + std::to_string(i);

            // One section per glTF primitive (per-primitive materials).
            // Geometry stays NODE-LOCAL (glTF vertices already are); the
            // node's world matrix goes into the .rwgeo header and the full
            // hierarchy into hierarchy.rwhier.
            ModelPreviewData d;
            std::vector<GeoSectionOut> secs;
            const auto& mesh = model.meshes[model.nodes[i].mesh];

            // Skinned node: extract per-vertex joints/weights and the
            // skin's joint table (joint → glTF node index, which is
            // EXACTLY the hierarchy.rwhier node index — the hier below
            // is written 1:1 in glTF node order — plus the inverse bind
            // matrix).  The baked .rwgeo then carries everything a
            // native skinned renderer needs.
            const int skin_idx = model.nodes[i].skin;
            const bool node_skinned =
                skin_idx >= 0 && skin_idx < (int)model.skins.size();
            if (node_skinned) {
                const auto& skin = model.skins[skin_idx];
                d.skin_joint_nodes.assign(skin.joints.begin(),
                                          skin.joints.end());
                d.skin_inverse_bind.assign(skin.joints.size(),
                                           glm::mat4(1.0f));
                if (skin.inverseBindMatrices >= 0 &&
                    skin.inverseBindMatrices <
                        (int)model.accessors.size()) {
                    const auto& a =
                        model.accessors[skin.inverseBindMatrices];
                    if (a.bufferView >= 0 &&
                        a.bufferView < (int)model.bufferViews.size()) {
                        const auto& bv = model.bufferViews[a.bufferView];
                        const auto& bf = model.buffers[bv.buffer];
                        const float* m = reinterpret_cast<const float*>(
                            bf.data.data() + bv.byteOffset + a.byteOffset);
                        const size_t n = std::min(
                            (size_t)a.count, skin.joints.size());
                        for (size_t j = 0; j < n; ++j) {
                            std::memcpy(&d.skin_inverse_bind[j],
                                        m + j * 16, 16 * sizeof(float));
                        }
                    }
                }
            }

            for (const auto& prim : mesh.primitives) {
                const uint32_t first = (uint32_t)d.indices.size();
                appendGltfPrim(model, prim, glm::mat4(1.0f),
                               d.positions, d.normals, d.uvs, d.indices,
                               node_skinned ? &d.joints     : nullptr,
                               node_skinned ? &d.weights    : nullptr,
                               node_skinned ? &d.closeness  : nullptr,
                               node_skinned ? &d.joints1    : nullptr,
                               node_skinned ? &d.weights1   : nullptr,
                               node_skinned ? &d.closeness1 : nullptr);
                const uint32_t count = (uint32_t)d.indices.size() - first;
                if (count < 3) continue;
                GeoSectionOut s;
                s.first_index = first;
                s.index_count = count;
                gltfMaterialFactors(model, prim.material, s.base_color,
                                    s.metallic, s.roughness);
                s.tex_rel = bake_gltf_texture(
                    gltfAlbedoTextureIndex(model, prim.material));
                // Full PBR refs: normal + metallic-roughness maps.
                if (prim.material >= 0 &&
                    prim.material < (int)model.materials.size()) {
                    const auto& m = model.materials[prim.material];
                    s.nrm_rel = bake_gltf_texture(
                        m.normalTexture.index);
                    s.mr_rel = bake_gltf_texture(
                        m.pbrMetallicRoughness
                            .metallicRoughnessTexture.index);
                }
                secs.push_back(std::move(s));
            }
            if (d.positions.empty() || d.indices.size() < 3 ||
                secs.empty()) { ++k; continue; }
            bool any_n = false;
            for (const auto& n : d.normals)
                if (glm::length(n) > 0.5f) { any_n = true; break; }
            if (!any_n)
                recomputeNormalsCpu(d.positions, d.indices, d.normals);
            // Placement bounds are WORLD-space.
            glm::vec3 bmn(1e30f), bmx(-1e30f);
            for (const auto& p : d.positions) {
                const glm::vec3 wp =
                    glm::vec3(node_world[i] * glm::vec4(p, 1.0f));
                bmn = glm::min(bmn, wp);
                bmx = glm::max(bmx, wp);
            }
            const std::string geo_rel = geo_rel_for((size_t)k, name);
            if (writeRwGeo((fs::path(group_dir) / geo_rel).string(),
                           d, secs, node_world[i])) {
                out_objects.push_back({ name, geo_rel, bmn, bmx });
            } else {
                out_objects.push_back({ name, std::string(), bmn, bmx });
            }
            ++k;
            if (progress) progress((size_t)k, total);
        }

        // ── Hierarchy: every node (incl. transform-only), local TRS ─────
        {
            std::vector<RwHierNode> hier(model.nodes.size());
            int mk = 0;
            for (int i = 0; i < (int)model.nodes.size(); ++i) {
                hier[i].parent = node_parent[i];
                hier[i].local  = gltfNodeLocal(model.nodes[i]);
                hier[i].name   = model.nodes[i].name.empty()
                    ? ("node " + std::to_string(i)) : model.nodes[i].name;
                hier[i].mesh_ordinal =
                    (model.nodes[i].mesh >= 0) ? mk++ : -1;
            }
            writeRwHier(
                (fs::path(group_dir) / "hierarchy.rwhier").string(), hier);
        }

        // ── Animation: every glTF clip → animation.rwanim ───────────────
        // glTF is already keyframed; channels target node indices that match
        // hierarchy.rwhier 1:1.  Rotation stored quaternion XYZW.
        {
            std::vector<RwAnimClip> clips;
            clips.reserve(model.animations.size());
            for (const auto& anim : model.animations) {
                RwAnimClip clip;
                clip.name = anim.name;
                float dur = 0.0f;
                for (const auto& chan : anim.channels) {
                    if (chan.sampler < 0 ||
                        chan.sampler >= (int)anim.samplers.size() ||
                        chan.target_node < 0)
                        continue;
                    RwAnimChannel oc;
                    oc.node = chan.target_node;
                    if (chan.target_path == "translation")
                        oc.path = RwAnimPath::kTranslation;
                    else if (chan.target_path == "scale")
                        oc.path = RwAnimPath::kScale;
                    else if (chan.target_path == "rotation")
                        oc.path = RwAnimPath::kRotation;
                    else
                        continue;   // morph "weights" / unsupported
                    const auto& samp = anim.samplers[chan.sampler];
                    const bool cubic = (samp.interpolation == "CUBICSPLINE");
                    oc.step = (samp.interpolation == "STEP") ? 1u : 0u;
                    const int ncomp =
                        (oc.path == RwAnimPath::kRotation) ? 4 : 3;
                    size_t tc = 0, ts = 0, vc = 0, vs = 0;
                    const float* tp =
                        gltfAccessorPtr(model, samp.input, tc, ts, 1);
                    const float* vp =
                        gltfAccessorPtr(model, samp.output, vc, vs, ncomp);
                    if (!tp || !vp || tc == 0) continue;
                    oc.times.reserve(tc);
                    oc.values.reserve(tc);
                    for (size_t k = 0; k < tc; ++k) {
                        const float t = tp[k * ts];
                        oc.times.push_back(t);
                        if (t > dur) dur = t;
                        // CUBICSPLINE packs [inTangent, value, outTangent]
                        // per key — take the middle (value) sample.
                        const size_t vk = cubic ? (k * 3 + 1) : k;
                        glm::vec4 val(0.0f);
                        for (int c = 0; c < ncomp; ++c)
                            val[c] = vp[vk * vs + c];
                        oc.values.push_back(val);
                    }
                    clip.channels.push_back(std::move(oc));
                }
                clip.duration = dur;
                if (!clip.channels.empty())
                    clips.push_back(std::move(clip));
            }
            if (!clips.empty()) {
                writeRwAnim(
                    (fs::path(group_dir) / "animation.rwanim").string(),
                    clips);
                std::cout << "[bake] baked " << clips.size()
                          << " animation clip(s)" << std::endl;
            }
        }

        // Character manifest — lets the engine load this baked group as ONE
        // skinned DrawableObject straight from raw data (no source needed).
        if (!model.skins.empty()) {
            const std::string leaf = fs::path(group_dir).filename().string();
            std::ofstream mf(
                (fs::path(group_dir) / (leaf + ".rwchar")).string(),
                std::ios::trunc);
            if (mf)
                mf << "rwchar=1\nname=" << leaf
                   << "\nhierarchy=hierarchy.rwhier"
                   << "\nanimation=animation.rwanim\n";
        }
    } else if (ext == ".fbx") {
        // RAW load options — identical to the engine's loadFbxModel (axis
        // conversion can mirror geometry; see loadModelPreviewData note).
        ufbx_load_opts opts{};
        ufbx_error error;
        ufbx_scene* scene =
            ufbx_load_file(source_path.c_str(), &opts, &error);
        if (!scene) return false;

        // Texture bake cache: ufbx texture → textures/<name>.rwtex.
        std::unordered_map<const ufbx_texture*, std::string> tex_cache;
        auto bake_fbx_texture =
            [&](const ufbx_texture* tex) -> std::string {
            if (!tex) return {};
            auto it = tex_cache.find(tex);
            if (it != tex_cache.end()) return it->second;

            int w = 0, h = 0;
            std::vector<unsigned char> rgba;
            bool loaded = false;
            if (tex->content.size > 0 && tex->content.data) {
                int comp = 0;
                unsigned char* px = stbi_load_from_memory(
                    (const unsigned char*)tex->content.data,
                    (int)tex->content.size, &w, &h, &comp, 4);
                if (px && w > 0 && h > 0) {
                    rgba.assign(px, px + (size_t)w * h * 4);
                    loaded = true;
                }
                if (px) stbi_image_free(px);
            }
            if (!loaded && tex->filename.data && tex->filename.length > 0)
                loaded = loadImageFileRgba(tex->filename.data, w, h, rgba);
            if (!loaded && tex->absolute_filename.data &&
                tex->absolute_filename.length > 0)
                loaded = loadImageFileRgba(tex->absolute_filename.data,
                                           w, h, rgba);
            if (!loaded && tex->relative_filename.data &&
                tex->relative_filename.length > 0) {
                const fs::path p =
                    fs::path(source_path).parent_path() /
                    fs::path(tex->relative_filename.data);
                loaded = loadImageFileRgba(p.string(), w, h, rgba);
            }
            std::cout << "[bake] fbx texture '"
                      << (tex->filename.data ? tex->filename.data : "?")
                      << "' -> " << (loaded ? "baked" : "FAILED")
                      << std::endl;
            if (!loaded) {
                tex_cache.emplace(tex, std::string());
                return {};
            }
            std::string nm = "texture";
            if (tex->filename.data && tex->filename.length > 0)
                nm = fs::path(tex->filename.data).stem().string();
            std::string rel =
                "textures/" + sanitizeFileName(nm) + ".rwtex";
            // Avoid same-stem collisions from different folders.
            int suffix = 1;
            std::error_code xec;
            while (fs::exists(fs::path(group_dir) / rel, xec)) {
                rel = "textures/" + sanitizeFileName(nm) + "_" +
                      std::to_string(suffix++) + ".rwtex";
            }
            if (!writeRwTex((fs::path(group_dir) / rel).string(),
                            w, h, rgba)) {
                tex_cache.emplace(tex, std::string());
                return {};
            }
            // Viewable copy of EXACTLY what was baked (orientation check +
            // Content Browser thumbnail).
            stbi_write_png(
                ((fs::path(group_dir) / rel).string() + ".png").c_str(),
                w, h, 4, rgba.data(), w * 4);
            tex_cache.emplace(tex, rel);
            return rel;
        };

        size_t total = 0;
        for (size_t i = 0; i < scene->nodes.count; ++i)
            if (scene->nodes.data[i]->mesh) ++total;
        int k = 0;
        for (size_t i = 0; i < scene->nodes.count; ++i) {
            const ufbx_node* node = scene->nodes.data[i];
            if (!node->mesh) continue;
            std::string name = node->name.data ? node->name.data : "";
            if (name.empty()) name = "node " + std::to_string(i);

            // Geometry bucketed by per-face material → one section each.
            // NODE-LOCAL space; the node's world matrix goes into the
            // .rwgeo header, the full tree into hierarchy.rwhier.
            ModelPreviewData d;
            std::vector<const ufbx_material*>  mats;
            std::vector<std::vector<uint32_t>> buckets;

            // Skinned node: build the skin joint table — per cluster, the bone
            // node's scene index (== hierarchy.rwhier index, via typed_id)
            // plus the node-local inverse bind (mesh_node_to_bone, matching the
            // NODE-LOCAL geometry baked below).  Mirrors the glTF skin bake so
            // the .rwgeo carries the full rig with no source dependency.
            const ufbx_skin_deformer* skin =
                (node->mesh->skin_deformers.count > 0)
                    ? node->mesh->skin_deformers.data[0] : nullptr;
            if (skin) {
                d.skin_joint_nodes.resize(skin->clusters.count);
                d.skin_inverse_bind.resize(skin->clusters.count);
                for (size_t ci = 0; ci < skin->clusters.count; ++ci) {
                    const ufbx_skin_cluster* cl = skin->clusters.data[ci];
                    d.skin_joint_nodes[ci] =
                        (cl && cl->bone_node)
                            ? (int32_t)cl->bone_node->typed_id : -1;
                    d.skin_inverse_bind[ci] = cl
                        ? ufbxToGlm(cl->mesh_node_to_bone) : glm::mat4(1.0f);
                }
            }
            appendFbxMesh(node, d.positions, d.normals, d.uvs,
                          mats, buckets, /*world_space=*/false,
                          skin,
                          skin ? &d.joints  : nullptr,
                          skin ? &d.weights : nullptr);
            std::vector<GeoSectionOut> secs;
            for (size_t b = 0; b < buckets.size(); ++b) {
                if (buckets[b].size() < 3) continue;
                GeoSectionOut s;
                s.first_index = (uint32_t)d.indices.size();
                s.index_count = (uint32_t)buckets[b].size();
                d.indices.insert(d.indices.end(),
                                 buckets[b].begin(), buckets[b].end());
                const ufbx_texture* tex = nullptr;
                fbxMaterialFactors(mats[b], s.base_color, s.metallic,
                                   s.roughness, tex);
                s.tex_rel = bake_fbx_texture(tex);
                // Normal map (PBR preview) — ufbx pbr slot first, then
                // the classic FBX property.
                if (mats[b]) {
                    const ufbx_texture* nrm =
                        mats[b]->pbr.normal_map.texture;
                    if (!nrm) nrm = mats[b]->fbx.normal_map.texture;
                    s.nrm_rel = bake_fbx_texture(nrm);
                }
                secs.push_back(std::move(s));
            }
            if (d.positions.empty() || d.indices.size() < 3 ||
                secs.empty()) { ++k; continue; }
            bool any_n = false;
            for (const auto& n : d.normals)
                if (glm::length(n) > 0.5f) { any_n = true; break; }
            if (!any_n)
                recomputeNormalsCpu(d.positions, d.indices, d.normals);

            // Placement bounds are WORLD-space.
            const glm::mat4 node_world = ufbxToGlm(node->node_to_world);
            glm::vec3 bmn(1e30f), bmx(-1e30f);
            for (const auto& p : d.positions) {
                const glm::vec3 wp =
                    glm::vec3(node_world * glm::vec4(p, 1.0f));
                bmn = glm::min(bmn, wp);
                bmx = glm::max(bmx, wp);
            }
            const std::string geo_rel = geo_rel_for((size_t)k, name);
            if (writeRwGeo((fs::path(group_dir) / geo_rel).string(),
                           d, secs, node_world)) {
                out_objects.push_back({ name, geo_rel, bmn, bmx });
            } else {
                out_objects.push_back({ name, std::string(), bmn, bmx });
            }
            ++k;
            if (progress) progress((size_t)k, total);
        }

        // ── Hierarchy: every node (incl. transform-only), local TRS ─────
        {
            std::unordered_map<const ufbx_node*, int> node_index;
            for (size_t i = 0; i < scene->nodes.count; ++i)
                node_index.emplace(scene->nodes.data[i], (int)i);
            std::vector<RwHierNode> hier(scene->nodes.count);
            int mk = 0;
            for (size_t i = 0; i < scene->nodes.count; ++i) {
                const ufbx_node* node = scene->nodes.data[i];
                auto pit = node->parent
                    ? node_index.find(node->parent) : node_index.end();
                hier[i].parent =
                    (pit != node_index.end()) ? pit->second : -1;
                hier[i].local = ufbxToGlm(node->node_to_parent);
                hier[i].name  = (node->name.data && node->name.length)
                    ? std::string(node->name.data)
                    : ("node " + std::to_string(i));
                hier[i].mesh_ordinal = node->mesh ? mk++ : -1;
            }
            writeRwHier(
                (fs::path(group_dir) / "hierarchy.rwhier").string(), hier);
        }

        // ── Animation: bake every FBX anim stack → animation.rwanim ─────
        // ufbx_bake_anim resamples each stack to per-node TRS keyframes;
        // baked_node.typed_id == scene->nodes index == hierarchy.rwhier
        // index.  Rotation stored quaternion XYZW.
        {
            std::vector<RwAnimClip> clips;
            for (size_t si = 0; si < scene->anim_stacks.count; ++si) {
                const ufbx_anim_stack* stack = scene->anim_stacks.data[si];
                if (!stack || !stack->anim) continue;
                ufbx_bake_opts bopts{};
                ufbx_error berr;
                ufbx_baked_anim* bake =
                    ufbx_bake_anim(scene, stack->anim, &bopts, &berr);
                if (!bake) continue;
                RwAnimClip clip;
                clip.name = (stack->name.data && stack->name.length)
                    ? std::string(stack->name.data)
                    : ("clip " + std::to_string(si));
                clip.duration = (float)bake->playback_duration;
                for (size_t ni = 0; ni < bake->nodes.count; ++ni) {
                    const ufbx_baked_node& bn = bake->nodes.data[ni];
                    const int32_t node_idx = (int32_t)bn.typed_id;
                    auto push_vec = [&](const ufbx_baked_vec3_list& keys,
                                        RwAnimPath path) {
                        if (!keys.count) return;
                        RwAnimChannel ch;
                        ch.node = node_idx; ch.path = path; ch.step = 0;
                        ch.times.reserve(keys.count);
                        ch.values.reserve(keys.count);
                        for (size_t k = 0; k < keys.count; ++k) {
                            ch.times.push_back((float)keys.data[k].time);
                            ch.values.emplace_back(
                                (float)keys.data[k].value.x,
                                (float)keys.data[k].value.y,
                                (float)keys.data[k].value.z, 0.0f);
                        }
                        clip.channels.push_back(std::move(ch));
                    };
                    push_vec(bn.translation_keys, RwAnimPath::kTranslation);
                    push_vec(bn.scale_keys,       RwAnimPath::kScale);
                    if (bn.rotation_keys.count) {
                        RwAnimChannel ch;
                        ch.node = node_idx;
                        ch.path = RwAnimPath::kRotation;
                        ch.step = 0;
                        ch.times.reserve(bn.rotation_keys.count);
                        ch.values.reserve(bn.rotation_keys.count);
                        for (size_t k = 0; k < bn.rotation_keys.count; ++k) {
                            const auto& kf = bn.rotation_keys.data[k];
                            ch.times.push_back((float)kf.time);
                            ch.values.emplace_back(
                                (float)kf.value.x, (float)kf.value.y,
                                (float)kf.value.z, (float)kf.value.w);
                        }
                        clip.channels.push_back(std::move(ch));
                    }
                }
                ufbx_free_baked_anim(bake);
                if (!clip.channels.empty())
                    clips.push_back(std::move(clip));
            }
            if (!clips.empty()) {
                writeRwAnim(
                    (fs::path(group_dir) / "animation.rwanim").string(),
                    clips);
                std::cout << "[bake] baked " << clips.size()
                          << " animation clip(s)" << std::endl;
            }
        }

        // Character manifest — see the glTF branch.
        if (scene->skin_deformers.count > 0) {
            const std::string leaf = fs::path(group_dir).filename().string();
            std::ofstream mf(
                (fs::path(group_dir) / (leaf + ".rwchar")).string(),
                std::ios::trunc);
            if (mf)
                mf << "rwchar=1\nname=" << leaf
                   << "\nhierarchy=hierarchy.rwhier"
                   << "\nanimation=animation.rwanim\n";
        }
        ufbx_free_scene(scene);
    } else {
        return false;
    }

    std::cout << "[bake] '" << source_path << "' -> " << out_objects.size()
              << " render-ready object(s) in '" << group_dir << "'"
              << std::endl;
    return !out_objects.empty();
}

bool readRwObjBounds(const std::string& rwobj_path,
                     glm::vec3& out_min, glm::vec3& out_max) {
    std::ifstream in(rwobj_path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("bbox=", 0) != 0) continue;
        float v[6];
        if (std::sscanf(line.c_str() + 5, "%f,%f,%f,%f,%f,%f",
                        &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) {
            out_min = glm::vec3(v[0], v[1], v[2]);
            out_max = glm::vec3(v[3], v[4], v[5]);
            return out_min.x <= out_max.x && out_min.y <= out_max.y &&
                   out_min.z <= out_max.z;
        }
    }
    return false;
}

uint64_t stableAssetHash(const std::string& s) {
    // FNV-1a 64-bit — deterministic across runs/builds/platforms.
    uint64_t h = 14695981039346656037ull;          // offset basis
    for (unsigned char c : s) {
        h ^= (uint64_t)c;
        h *= 1099511628211ull;                     // FNV prime
    }
    return h;
}

std::string makeAssetId(const std::string& type,
                        const std::string& canonical_path,
                        const std::string& name) {
    std::string key;
    key.reserve(type.size() + canonical_path.size() + name.size() + 2);
    key += type;
    key += '|';
    for (char c : canonical_path) {
        if (c == '\\') c = '/';
        key += (char)std::tolower((unsigned char)c);
    }
    key += '|';
    for (char c : name) {
        key += (char)std::tolower((unsigned char)c);
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  (unsigned long long)stableAssetHash(key));
    return std::string(buf);
}

} // namespace helper
} // namespace engine
