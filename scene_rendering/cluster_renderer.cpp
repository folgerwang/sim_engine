//
// cluster_renderer.cpp — GPU-driven cluster culling & rendering.
//
// All cluster data is merged into single flat SSBOs. The cull pass is a
// single compute dispatch regardless of how many source meshes were uploaded.
//
#include "cluster_renderer.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "helper/material_classifier.h"
#include "game_object/drawable_object.h"
#include "virtual_texture.h"

#include <cstdio>
#include <source_location>
#include <cstring>
#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace er  = engine::renderer;

namespace engine {
namespace scene_rendering {

// ─── Push-constant struct layout sanity checks ────────────────────────
// The same ClusterCullPushConstants struct is consumed by both C++ (via
// glsl::ClusterCullPushConstants from global_definition.glsl.h) and the
// GLSL compute shader.  glslc + Vulkan std430 push-constant rules give
// known offsets; if the C++ side ever drifts (e.g. someone removes a
// pad uint, or glm changes default vec4 alignment), we want a build
// failure rather than silent corruption — symptom is hard-to-diagnose
// "diagnostics in Phase B's Hi-Z block don't fire even though Phase B
// pushes cull_phase=2u".
static_assert(offsetof(glsl::ClusterCullPushConstants, view_proj)         == 0,   "view_proj offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, camera_pos_pad)    == 64,  "camera_pos_pad offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, total_clusters)    == 80,  "total_clusters offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, total_bvh_nodes)   == 84,  "total_bvh_nodes offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, lod_error_threshold)==88,  "lod_error_threshold offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, use_bvh)           == 92,  "use_bvh offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, use_hiz_cull)      == 96,  "use_hiz_cull offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, cull_phase)        == 100, "cull_phase offset MISMATCH — std430 expects 100");
static_assert(offsetof(glsl::ClusterCullPushConstants, pad0)              == 104, "pad0 offset");
static_assert(offsetof(glsl::ClusterCullPushConstants, pad1)              == 108, "pad1 offset (must precede mat4 to push it to offset 112)");
static_assert(offsetof(glsl::ClusterCullPushConstants, last_view_proj)    == 112, "last_view_proj offset (std430 mat4 alignment forces 16B-aligned start)");
static_assert(offsetof(glsl::ClusterCullPushConstants, hiz_size_mips_pad) == 176, "hiz_size_mips_pad offset");
static_assert(sizeof(glsl::ClusterCullPushConstants) == 192,                       "ClusterCullPushConstants total size");

// ─── BindlessMaterialParams layout check ──────────────────────────────
// std430 layout (storage buffer) — vec4 needs 16-byte align, uint
// 4-byte align.  See global_definition.glsl.h.  Misalignment between
// C++ and GLSL would silently corrupt every material's data.
static_assert(offsetof(glsl::BindlessMaterialParams, base_color_factor)  ==  0, "");
static_assert(offsetof(glsl::BindlessMaterialParams, base_color_tex_idx) == 16, "");
static_assert(offsetof(glsl::BindlessMaterialParams, alpha_cutoff)       == 20, "");
static_assert(offsetof(glsl::BindlessMaterialParams, flags)              == 24, "");
static_assert(offsetof(glsl::BindlessMaterialParams, normal_tex_idx)     == 28, "");
static_assert(offsetof(glsl::BindlessMaterialParams, albedo_vt_id)       == 32, "");
static_assert(offsetof(glsl::BindlessMaterialParams, normal_vt_id)       == 36, "");
static_assert(offsetof(glsl::BindlessMaterialParams, mr_ao_vt_id)        == 40, "");
static_assert(offsetof(glsl::BindlessMaterialParams, emissive_vt_id)     == 44, "");
static_assert(sizeof(glsl::BindlessMaterialParams)                       == 48, "");

// Flip to 1 to re-enable the diagnostic upload / finalize lines that
// otherwise printed for every ClusterRenderer mesh upload + a few
// per-frame fprintf'd lines.  Keep these silent in normal runs;
// they're only useful when debugging the bindless cluster pipeline
// at startup.
#ifndef CLUSTER_RENDERER_VERBOSE
#  define CLUSTER_RENDERER_VERBOSE 0
#endif

// ─── Per-source-vertex tangent computation ────────────────────────────────
//
// Lengyel-style accumulation: for each triangle, derive (T, B) from position
// and UV gradients, then accumulate into each of the triangle's three vertex
// slots.  After visiting every face, normalise each vertex's accumulated T,
// orthogonalise it against the vertex normal (Gram-Schmidt), and stash the
// bitangent sign in .w so the fragment shader can recover the bitangent as
// `B = cross(N, T) * tangent.w`.
//
// This is the same scheme MikkT uses at low quality settings — good enough
// for normal-map rendering without the per-fragment dFdx/dFdy reconstruction
// that produces speckle in cluster_bindless.frag.  Returns one vec4 per
// source vertex in object space; the caller transforms .xyz to world space.
namespace {

std::vector<glm::vec4> computeMeshTangents(
    const std::vector<engine::helper::VertexStruct>& verts,
    const std::vector<engine::helper::Face>&         faces) {

    const size_t nv = verts.size();
    std::vector<glm::vec3> tan_acc(nv, glm::vec3(0.0f));
    std::vector<glm::vec3> bit_acc(nv, glm::vec3(0.0f));

    for (const auto& f : faces) {
        if (f.isDegenerate()) continue;
        if (f.v_indices[0] >= nv || f.v_indices[1] >= nv ||
            f.v_indices[2] >= nv) continue;

        const auto& v0 = verts[f.v_indices[0]];
        const auto& v1 = verts[f.v_indices[1]];
        const auto& v2 = verts[f.v_indices[2]];

        glm::vec3 e1  = v1.position - v0.position;
        glm::vec3 e2  = v2.position - v0.position;
        glm::vec2 du1 = v1.uv - v0.uv;
        glm::vec2 du2 = v2.uv - v0.uv;

        // Determinant of the UV Jacobian; sign encodes UV handedness.
        // Tiny |det| means the triangle is UV-degenerate (collapsed to a
        // line in texture space) — skip rather than divide by ~0.
        float det = du1.x * du2.y - du2.x * du1.y;
        if (std::abs(det) < 1e-8f) continue;
        float r = 1.0f / det;

        glm::vec3 T = r * ( du2.y * e1 - du1.y * e2);
        glm::vec3 B = r * (-du2.x * e1 + du1.x * e2);

        for (int k = 0; k < 3; ++k) {
            uint32_t vi = f.v_indices[k];
            tan_acc[vi] += T;
            bit_acc[vi] += B;
        }
    }

    std::vector<glm::vec4> out(nv);
    for (size_t i = 0; i < nv; ++i) {
        glm::vec3 N = verts[i].normal;
        float n_len = glm::length(N);
        if (n_len < 1e-6f) {
            // Degenerate vertex normal — emit an arbitrary frame.  The
            // fragment shader will still work; the surface just won't show
            // normal-map detail (which is consistent with degenerate input).
            out[i] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            continue;
        }
        N /= n_len;

        glm::vec3 T = tan_acc[i];
        // Orthogonalise the accumulated tangent against the vertex normal.
        T = T - glm::dot(T, N) * N;
        float t_len = glm::length(T);
        if (t_len < 1e-6f) {
            // No usable UV info on any incident face — pick an arbitrary
            // perpendicular to N so downstream math (cross, normalize) is
            // stable.  Picks the world axis least aligned with N.
            glm::vec3 up = std::abs(N.y) < 0.99f
                ? glm::vec3(0.0f, 1.0f, 0.0f)
                : glm::vec3(1.0f, 0.0f, 0.0f);
            T = glm::normalize(glm::cross(up, N));
        } else {
            T /= t_len;
        }

        // Bitangent sign: if cross(N, T) aligns with the accumulated B,
        // sign is +1; otherwise -1.  Shader recomputes B = cross(N, T) * w.
        float w = glm::dot(glm::cross(N, T), bit_acc[i]) >= 0.0f
            ? 1.0f : -1.0f;
        out[i] = glm::vec4(T, w);
    }
    return out;
}

}  // anonymous namespace

// ─── Descriptor set layout for the cull compute pass ───────────────

namespace {

std::shared_ptr<er::DescriptorSetLayout> createCullDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    // binding 0: ClusterCullInfo[]               (SSBO, readonly)
    // binding 1: ClusterDrawInfo[]               (SSBO, readonly)
    // binding 2: IndirectDrawBuffer (opaque)     (SSBO, writeonly)
    // binding 3: DrawCountBuffer    (opaque)     (SSBO, read/write — atomicAdd)
    // binding 4: VisibleClusterBuffer (opaque)   (SSBO, writeonly)
    // binding 5: BindlessMaterialParams[]        (SSBO, readonly — for flags)
    // binding 6: TransIndirectDrawBuffer         (SSBO, writeonly)
    // binding 7: TransDrawCountBuffer            (SSBO, read/write — atomicAdd)
    // binding 8: VisibilityBitBuffer             (SSBO, read+atomicOr) — Nanite-style
    //              persistent visibility from previous frame.  Phase A reads,
    //              Phase B writes (atomicOr per visible cluster).
    // binding 9: IndirectDrawBuffer (Phase A)    (SSBO, writeonly) — only used by
    //              Phase A cull; Phase B writes the regular binding 2.
    // binding 10: DrawCountBuffer    (Phase A)   (SSBO, read/write — atomicAdd)
    // binding 11: Hi-Z pyramid sampler           (sampler2D, read-only)
    //              Bound to the application's hiz_pyramid_ texture so the
    //              cull pass can reproject each cluster's bounding sphere
    //              through last_view_proj and reject clusters fully behind
    //              the previous frame's depth (Nanite-style occlusion).
    //              When use_hiz_occlusion_cull_ is off the shader skips
    //              the sample, so a dummy 1×1 white texture is fine — but
    //              the binding MUST be valid even then because Vulkan
    //              validates descriptor presence at dispatch time.
    std::vector<er::DescriptorSetLayoutBinding> bindings(12);
    for (int i = 0; i < 11; ++i) {
        bindings[i] = er::helper::getBufferDescriptionSetLayoutBinding(
            i, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_BUFFER);
    }
    bindings[11] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        11, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    return device->createDescriptorSetLayout(bindings);
}

void writeCullDescriptors(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSet>& desc_set,
    uint32_t total_clusters,
    const renderer::BufferInfo& cull_info_buffer,
    const renderer::BufferInfo& draw_info_buffer,
    const renderer::BufferInfo& indirect_draw_buffer,
    const renderer::BufferInfo& draw_count_buffer,
    const renderer::BufferInfo& visible_buffer,
    const renderer::BufferInfo& material_params_buffer,
    uint32_t                    total_materials,
    const renderer::BufferInfo& trans_indirect_draw_buffer,
    const renderer::BufferInfo& trans_draw_count_buffer,
    const renderer::BufferInfo& visibility_bit_buffer,
    const renderer::BufferInfo& indirect_draw_buffer_phase_a,
    const renderer::BufferInfo& draw_count_buffer_phase_a,
    const std::shared_ptr<er::Sampler>& hiz_sampler,
    const std::shared_ptr<er::ImageView>& hiz_view) {

    er::WriteDescriptorList writes;
    writes.reserve(12);

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 0,
        cull_info_buffer.buffer,
        static_cast<uint32_t>(total_clusters * sizeof(glsl::ClusterCullInfo)));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 1,
        draw_info_buffer.buffer,
        static_cast<uint32_t>(total_clusters * sizeof(glsl::ClusterDrawInfo)));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 2,
        indirect_draw_buffer.buffer,
        static_cast<uint32_t>(total_clusters * 5u * sizeof(uint32_t)));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 3,
        draw_count_buffer.buffer,
        sizeof(uint32_t));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 4,
        visible_buffer.buffer,
        static_cast<uint32_t>(total_clusters * sizeof(uint32_t)));

    // Material params — same buffer the bindless render pass uses.  Cull
    // shader reads .flags only, but we expose the whole struct so the
    // layout matches across the two consumers.  The buffer always has at
    // least one entry (the white-fallback initialisation in finalizeUploads).
    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 5,
        material_params_buffer.buffer,
        static_cast<uint32_t>(std::max(total_materials, 1u)
                              * sizeof(glsl::BindlessMaterialParams)));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 6,
        trans_indirect_draw_buffer.buffer,
        static_cast<uint32_t>(total_clusters * 5u * sizeof(uint32_t)));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 7,
        trans_draw_count_buffer.buffer,
        sizeof(uint32_t));

    // Two-pass occlusion bindings.  Sized so the buffer's whole range is
    // visible to the shader; the cull compute decides per-cluster which
    // 32-bit element to touch via cluster_idx >> 5.
    const uint32_t vis_uint_count = (total_clusters + 31u) / 32u;
    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 8,
        visibility_bit_buffer.buffer,
        static_cast<uint32_t>(vis_uint_count * sizeof(uint32_t)));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 9,
        indirect_draw_buffer_phase_a.buffer,
        static_cast<uint32_t>(total_clusters * 5u * sizeof(uint32_t)));

    er::Helper::addOneBuffer(writes, desc_set,
        er::DescriptorType::STORAGE_BUFFER, 10,
        draw_count_buffer_phase_a.buffer,
        sizeof(uint32_t));

    // Hi-Z pyramid for last-frame occlusion cull.  Caller must pass a
    // valid sampler + view: nullptrs would emit a non-conforming
    // descriptor write.  When the application hasn't built the pyramid
    // yet (e.g. very first frame) it should pass its dummy 1×1 white
    // texture and Hi-Z sample in the shader will read 1.0 (= far
    // plane) so no cluster is rejected.
    //
    // IMPORTANT: the Hi-Z pyramid stays in GENERAL layout for the
    // entire frame (storage-image writes during build, sampler reads
    // here in the cull compute, sampler reads in the deferred resolve
    // for visualization).  The descriptor write must match — declaring
    // SHADER_READ_ONLY_OPTIMAL when the image is actually GENERAL is
    // a validation-layer warning AND in practice on NV produced
    // undefined sampler results, so the test always read 1.0 and no
    // cluster was ever rejected (toggle on/off was a no-op).
    if (hiz_sampler && hiz_view) {
        er::Helper::addOneTexture(writes, desc_set,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER, 11,
            hiz_sampler, hiz_view,
            er::ImageLayout::GENERAL);
    }

    device->updateDescriptorSets(writes);
}

// Create a GPU buffer with initial data upload.
renderer::BufferInfo createSSBO(
    const std::shared_ptr<er::Device>& device,
    uint64_t size,
    const void* data) {

    // HOST_VISIBLE so the CPU culling path can read cluster data
    // and write indirect draw commands. Slight perf cost on
    // discrete GPUs but keeps both paths working from a single buffer.
    renderer::BufferInfo info;
    er::Helper::createBuffer(
        device,
        SET_2_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT, TRANSFER_DST_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        info.buffer,
        info.memory,
        std::source_location::current(),
        size,
        data);
    return info;
}

// Create a GPU buffer for indirect draw commands.
renderer::BufferInfo createIndirectSSBO(
    const std::shared_ptr<er::Device>& device,
    uint64_t size) {

    // HOST_VISIBLE so the CPU culling path can write indirect commands.
    renderer::BufferInfo info;
    er::Helper::createBuffer(
        device,
        SET_3_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT,
                        INDIRECT_BUFFER_BIT, TRANSFER_DST_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        info.buffer,
        info.memory,
        std::source_location::current(),
        size);
    return info;
}

// Create a small host-visible buffer for the draw count (need CPU readback).
// INDIRECT_BUFFER_BIT is required by vkCmdDrawIndexedIndirectCount for the
// count buffer — omitting it is a Vulkan spec violation that causes undefined
// behaviour (and validation errors on debug layers).
renderer::BufferInfo createCounterBuffer(
    const std::shared_ptr<er::Device>& device) {

    renderer::BufferInfo info;
    uint32_t zero = 0;
    er::Helper::createBuffer(
        device,
        SET_3_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT,
                        INDIRECT_BUFFER_BIT, TRANSFER_DST_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        info.buffer,
        info.memory,
        std::source_location::current(),
        sizeof(uint32_t),
        &zero);
    return info;
}

} // anonymous namespace

// ─── Constructor ───────────────────────────────────────────────────

ClusterRenderer::ClusterRenderer(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool)
    : device_(device)
    , descriptor_pool_(descriptor_pool) {

    cull_desc_set_layout_ = createCullDescSetLayout(device);

    cull_pipeline_layout_ = er::helper::createComputePipelineLayout(
        device,
        { cull_desc_set_layout_ },
        sizeof(glsl::ClusterCullPushConstants));

    cull_pipeline_ = er::helper::createComputePipeline(
        device,
        cull_pipeline_layout_,
        "cluster_cull_comp.spv",
        std::source_location::current());

    // ── Default linear sampler (shared by all clustered textures) ──
    default_sampler_ = device->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR,
        16.0f,
        std::source_location::current());

    // ── Dummy 1×1 white texture (fills unused slots in the texture array) ──
    constexpr uint8_t kWhitePixel[4] = { 255, 255, 255, 255 };
    er::Helper::create2DTextureImage(
        device,
        er::Format::R8G8B8A8_UNORM,
        1, 1,
        kWhitePixel,
        dummy_texture_.image,
        dummy_texture_.memory,
        std::source_location::current());
    dummy_texture_.view = device->createImageView(
        dummy_texture_.image,
        er::ImageViewType::VIEW_2D,
        er::Format::R8G8B8A8_UNORM,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        std::source_location::current());

#if CLUSTER_RENDERER_VERBOSE
    std::printf("[CLUSTER_RENDERER] Initialized GPU cluster culling pipeline.\n");
#endif
}

// ─── Upload mesh clusters (CPU staging only) ──────────────────────

void ClusterRenderer::uploadMeshClusters(
    const helper::ClusterMesh& cluster_mesh,
    const game_object::DrawableData& drawable_data,
    uint32_t mesh_idx,
    const std::vector<uint32_t>& cluster_prim_map,
    const glm::mat4& model_transform) {

    if (cluster_mesh.empty() || !cluster_mesh.source) {
        return;
    }

    const uint32_t num_clusters =
        static_cast<uint32_t>(cluster_mesh.clusters.size());

    // Correct normal-space transform = transpose(inverse(upper3×3)).
    // The raw upper3×3 is only valid for orthogonal (uniform-scale + rotation)
    // matrices. Non-uniform scale shears normals in the wrong direction.
    // We always normalize the result, so the overall scale factor doesn't
    // matter — only the direction needs to be correct.
    // Use the max column scale to conservatively inflate radii.
    const glm::mat3 normal_mat =
        glm::transpose(glm::inverse(glm::mat3(model_transform)));
    const float scale_x = glm::length(glm::vec3(model_transform[0]));
    const float scale_y = glm::length(glm::vec3(model_transform[1]));
    const float scale_z = glm::length(glm::vec3(model_transform[2]));
    const float max_scale = std::max({scale_x, scale_y, scale_z});

    // Diagnostic: save source data for first mesh uploaded (shown in ImGui).
    // The two fprintf summaries are guarded by CLUSTER_RENDERER_VERBOSE
    // (default off); the bookkeeping that captures debug_first_*_bounds_
    // for the ImGui inspector still runs unconditionally.
    if (uploaded_mesh_count_ < 3 && num_clusters > 0) {
        const auto& c0 = cluster_mesh.clusters[0];
#if CLUSTER_RENDERER_VERBOSE
        fprintf(stderr,
            "[CLUSTER_UPLOAD #%u] mesh_idx=%u clusters=%u "
            "LOCAL center=(%.3f,%.3f,%.3f) r=%.3f "
            "modelDiag=(%.3f,%.3f,%.3f,%.3f)\n",
            uploaded_mesh_count_, mesh_idx, num_clusters,
            c0.bounds_center.x, c0.bounds_center.y,
            c0.bounds_center.z, c0.bounds_radius,
            model_transform[0][0], model_transform[1][1],
            model_transform[2][2], model_transform[3][3]);
        fprintf(stderr,
            "  modelCol3=(%.3f,%.3f,%.3f,%.3f) "
            "LOCAL aabb_min=(%.3f,%.3f,%.3f) aabb_max=(%.3f,%.3f,%.3f)\n",
            model_transform[3][0], model_transform[3][1],
            model_transform[3][2], model_transform[3][3],
            c0.aabb_min.x, c0.aabb_min.y, c0.aabb_min.z,
            c0.aabb_max.x, c0.aabb_max.y, c0.aabb_max.z);
#endif

        if (uploaded_mesh_count_ == 0) {
            debug_first_local_bounds_ = glm::vec4(
                c0.bounds_center, c0.bounds_radius);
            debug_first_model_diag_ = glm::vec4(
                model_transform[0][0], model_transform[1][1],
                model_transform[2][2], model_transform[3][3]);
        }
    }

    // ── Per-cluster material lookup ────────────────────────────────────────
    // One material entry is created per unique primitive index encountered in
    // cluster_prim_map. prim_to_mat_idx caches prim_idx → staging index so
    // clusters that share a primitive reuse the same entry (no duplication).
    std::unordered_map<uint32_t, uint32_t> prim_to_mat_idx;
    const auto& meshes = drawable_data.meshes_;

    // First node that references this mesh wins as the "object name"
    // for every primitive inside it.  Same convention used by
    // CollisionMesh::buildFromDrawablePrimitive — keeps the AI
    // classifier's per-mesh lookup consistent across the collision
    // proxy and the cluster-rendered draw.  Empty when no node owns
    // the mesh (rare but possible for procedurally-generated meshes).
    std::string mesh_object_name;
    for (const auto& node : drawable_data.nodes_) {
        if (node.mesh_idx_ == static_cast<int32_t>(mesh_idx)) {
            mesh_object_name = node.name_;
            break;
        }
    }

    auto getMaterialIdx = [&](uint32_t prim_idx) -> uint32_t {
        auto cache_it = prim_to_mat_idx.find(prim_idx);
        if (cache_it != prim_to_mat_idx.end()) return cache_it->second;

        glm::vec4 base_color(1.0f);  // white default
        glsl::BindlessMaterialParams mp{};
        mp.base_color_tex_idx = -1;
        mp.normal_tex_idx     = -1;
        // VT IDs default to INVALID until registerMaterial assigns
        // one (per layer that actually had a source image).  Shader
        // checks against kInvalidVtId to fall back to the legacy
        // bindless path when VT registration was skipped (pool full,
        // manager not wired, layer had no source image, etc.).
        mp.albedo_vt_id   = 0xFFFFFFFFu;
        mp.normal_vt_id   = 0xFFFFFFFFu;
        mp.mr_ao_vt_id    = 0xFFFFFFFFu;
        mp.emissive_vt_id = 0xFFFFFFFFu;

        if (mesh_idx < meshes.size()) {
            const auto& prims = meshes[mesh_idx].primitives_;
            if (prim_idx < prims.size()) {
                int32_t mat_idx = prims[prim_idx].material_idx_;
                if (mat_idx >= 0 &&
                    static_cast<size_t>(mat_idx) < drawable_data.materials_.size()) {
                    const auto& mat = drawable_data.materials_[mat_idx];
                    // uniform_buffer_ is HOST_VISIBLE|HOST_COHERENT — map, read, unmap.
                    if (mat.uniform_buffer_.memory) {
                        constexpr uint64_t kVec4Size = sizeof(glm::vec4);
                        void* ptr = device_->mapMemory(
                            mat.uniform_buffer_.memory, kVec4Size, 0);
                        if (ptr) {
                            std::memcpy(&base_color, ptr, kVec4Size);
                            device_->unmapMemory(mat.uniform_buffer_.memory);
                        }
                    }
                    // Stage the base-color texture (binding 2).
                    int32_t tex_idx = mat.base_color_idx_;
                    // Resolve albedo TextureInfo and add to legacy
                    // bindless array (parallel path, see VT block
                    // below).  Capture for the combined VT
                    // registration call.
                    const renderer::TextureInfo* albedo_tex = nullptr;
                    if (tex_idx >= 0 &&
                        static_cast<size_t>(tex_idx) < drawable_data.textures_.size()) {
                        albedo_tex = &drawable_data.textures_[tex_idx];
                        if (albedo_tex->view) {
                            auto tex_it = staging_tex_slot_map_.find(albedo_tex->view.get());
                            if (tex_it != staging_tex_slot_map_.end()) {
                                mp.base_color_tex_idx = tex_it->second;
                            } else if (staging_tex_views_.size() < MAX_CLUSTER_TEXTURES) {
                                int slot = static_cast<int>(staging_tex_views_.size());
                                staging_tex_slot_map_[albedo_tex->view.get()] = slot;
                                staging_tex_views_.push_back(albedo_tex->view);
                                mp.base_color_tex_idx = slot;
                            }
                            // else: over MAX_CLUSTER_TEXTURES — idx stays -1
                        }
                    }
                    // Stage the normal-map texture (binding 3).
                    int32_t norm_idx = mat.normal_idx_;
                    const renderer::TextureInfo* normal_tex = nullptr;
                    if (norm_idx >= 0 &&
                        static_cast<size_t>(norm_idx) < drawable_data.textures_.size()) {
                        normal_tex = &drawable_data.textures_[norm_idx];
                        if (normal_tex->view) {
                            auto n_it = staging_normal_slot_map_.find(normal_tex->view.get());
                            if (n_it != staging_normal_slot_map_.end()) {
                                mp.normal_tex_idx = n_it->second;
                            } else if (staging_normal_tex_views_.size() < MAX_CLUSTER_TEXTURES) {
                                int slot = static_cast<int>(staging_normal_tex_views_.size());
                                staging_normal_slot_map_[normal_tex->view.get()] = slot;
                                staging_normal_tex_views_.push_back(normal_tex->view);
                                mp.normal_tex_idx = slot;
                            }
                        }
                    }
                    // ── Unified VT registration (per-material) ────
                    // Single registerMaterial call uploads albedo +
                    // (optionally) normal into the matching layer
                    // pools at IDENTICAL physical slot positions.
                    // The same vt_id then resolves correctly across
                    // both layers in cluster_bindless.frag — slot N
                    // in vt_pool_albedo is material X's albedo, slot
                    // N in vt_pool_normal is material X's normal.
                    //
                    // Cache by albedo Image* — Bistro materials that
                    // share an albedo image typically share the rest
                    // too, so this de-dupes successfully.  If a
                    // future asset reuses an albedo with a different
                    // normal, the cache would return the FIRST
                    // material's normal data; revisit with a
                    // tuple-keyed cache when that case appears.
                    // VT-only baked textures carry CPU pixels and/or a
                    // bake-time pre-encoded BC7 tile blob but NO GPU
                    // image — registerMaterial handles all three
                    // sources.  Cache key = image pointer when present,
                    // else the shared cpu_pixels / blob pointer.
                    const bool albedo_has_cpu =
                        albedo_tex && albedo_tex->cpu_pixels &&
                        !albedo_tex->cpu_pixels->empty();
                    const bool albedo_has_blob =
                        albedo_tex && albedo_tex->vt_bc7_tiles &&
                        !albedo_tex->vt_bc7_tiles->empty();
                    if (vt_manager_ && albedo_tex &&
                        (albedo_tex->image || albedo_has_cpu ||
                         albedo_has_blob) &&
                        albedo_tex->size.x > 0 && albedo_tex->size.y > 0) {
                        const void* vt_key = albedo_tex->image
                            ? static_cast<const void*>(albedo_tex->image.get())
                            : albedo_tex->cpu_pixels
                                ? static_cast<const void*>(albedo_tex->cpu_pixels.get())
                                : static_cast<const void*>(albedo_tex->vt_bc7_tiles.get());
                        auto vt_it = vt_albedo_id_cache_.find(vt_key);
                        uint32_t vid = kInvalidVtId;
                        if (vt_it != vt_albedo_id_cache_.end()) {
                            vid = vt_it->second;
                        } else {
                            const auto& nrm_img =
                                (normal_tex && normal_tex->image)
                                    ? normal_tex->image
                                    : std::shared_ptr<renderer::Image>();
                            // Prefer CPU pixels when the loader stashed them
                            // (glTF + non-DDS engine_helper paths).  Fall
                            // back to the GPU image — registerMaterial does
                            // a one-time blit-decode + readback for BC-only
                            // sources to materialise RGBA8.
                            const uint8_t* px = albedo_has_cpu
                                ? albedo_tex->cpu_pixels->data()
                                : nullptr;
                            vid = vt_manager_->registerMaterial(
                                px,
                                albedo_tex->image,
                                nrm_img,
                                /*mr_ao*/   nullptr,
                                /*emissive*/nullptr,
                                albedo_tex->size.x,
                                albedo_tex->size.y,
                                albedo_tex->vt_bc7_tiles);
                            if (vid != kInvalidVtId) {
                                vt_albedo_id_cache_[vt_key] = vid;
                                if (nrm_img) {
                                    vt_normal_id_cache_[nrm_img.get()] = vid;
                                }
                            }
                        }
                        if (vid != kInvalidVtId) {
                            mp.albedo_vt_id = vid;
                            if (normal_tex && normal_tex->image) {
                                mp.normal_vt_id = vid;
                            }
                        }
                    }
                    // Alpha mask — discard fragments below cutoff.
                    if (mat.alpha_mask_ && mat.alpha_cutoff_ > 0.0f) {
                        mp.alpha_cutoff = mat.alpha_cutoff_;
                        mp.flags |= BINDLESS_MAT_ALPHA_MASK;
                    }
                    // Translucent (glass / windows / asset-authored Blend).
                    // The cluster pipeline currently still draws these as
                    // opaque (no separate translucent pass yet), but we
                    // flag them so the "Translucent" render-debug
                    // visualisation can highlight them, and so the data
                    // is in place when a real translucent pass is added.
                    if (mat.alpha_mode_ == game_object::AlphaMode::Blend) {
                        mp.flags |= BINDLESS_MAT_TRANSLUCENT;
                    }
                }
                // Double-sided — primitive property, not material-level.
                if (prims[prim_idx].tag_.double_sided) {
                    mp.flags |= BINDLESS_MAT_DOUBLE_SIDED;
                }
            }
        }

        mp.base_color_factor = base_color;
        uint32_t idx = static_cast<uint32_t>(staging_material_params_.size());
        staging_material_params_.push_back(mp);

        // Capture (material_name, object_name) alongside the params
        // entry so applyMaterialCategories() can later look up the
        // category from the LLM classifier without re-walking the
        // drawable data.  Material name comes from MaterialInfo; the
        // object name is the first owning-node's name resolved above.
        std::string material_name;
        if (mesh_idx < meshes.size()) {
            const auto& prims = meshes[mesh_idx].primitives_;
            if (prim_idx < prims.size()) {
                const int32_t mat_idx = prims[prim_idx].material_idx_;
                if (mat_idx >= 0 &&
                    static_cast<size_t>(mat_idx) <
                        drawable_data.materials_.size()) {
                    material_name =
                        drawable_data.materials_[mat_idx].name_;
                }
            }
        }
        staging_material_names_.emplace_back(
            std::move(material_name), mesh_object_name);

        prim_to_mat_idx[prim_idx] = idx;
        return idx;
    };

    // ── Per-source-mesh tangent precomputation ────────────────────────────
    // Computed ONCE per uploadMeshClusters call (not per cluster) and
    // looked up by source vertex index inside the cluster loop below.  Each
    // entry is a vec4(T_object, bitangent_sign).  Empty if the source mesh
    // has no CPU-side geometry (e.g. it was already freed after the
    // original draw path's GPU upload) — the per-cluster code below already
    // bails out in that case, so we just leave src_tangents empty.
    std::vector<glm::vec4> src_tangents;
    if (cluster_mesh.source->vertex_data_ptr &&
        cluster_mesh.source->faces_ptr &&
        !cluster_mesh.source->vertex_data_ptr->empty()) {
        src_tangents = computeMeshTangents(
            *cluster_mesh.source->vertex_data_ptr,
            *cluster_mesh.source->faces_ptr);
    }
    // Direction-vector transform for tangents.  Tangents lie IN the surface
    // plane and transform like position differences (mat3 of model_transform),
    // NOT like normals (inverse-transpose).  For rigid / uniform-scale
    // instances the two are identical; for shears they differ.
    const glm::mat3 tangent_mat3 = glm::mat3(model_transform);

    for (uint32_t c = 0; c < num_clusters; ++c) {
        const auto& cl = cluster_mesh.clusters[c];

        // Transform cluster bounds from local space → world space.
        glm::vec3 ws_center = glm::vec3(
            model_transform * glm::vec4(cl.bounds_center, 1.0f));
        float ws_radius = cl.bounds_radius * max_scale;

        glm::vec3 ws_aabb_min = glm::vec3(
            model_transform * glm::vec4(cl.aabb_min, 1.0f));
        glm::vec3 ws_aabb_max = glm::vec3(
            model_transform * glm::vec4(cl.aabb_max, 1.0f));
        // Fix min/max after transform (rotation can swap them).
        glm::vec3 fixed_min = glm::min(ws_aabb_min, ws_aabb_max);
        glm::vec3 fixed_max = glm::max(ws_aabb_min, ws_aabb_max);

        // Transform cone axis by the normal matrix.
        // Guard against zero-length result — normalize of zero produces NaN.
        glm::vec3 raw_cone = normal_mat * cl.cone_axis;
        float cone_len2 = glm::dot(raw_cone, raw_cone);
        glm::vec3 ws_cone_axis = (cone_len2 > 1e-12f)
            ? raw_cone / std::sqrt(cone_len2)
            : glm::vec3(0.0f, 0.0f, 1.0f);

        glsl::ClusterCullInfo cull_info;
        cull_info.bounds_sphere = glm::vec4(ws_center, ws_radius);
        cull_info.cone_axis_cutoff = glm::vec4(ws_cone_axis, cl.cone_cutoff);
        cull_info.aabb_min_pad = glm::vec4(fixed_min, 0.0f);
        cull_info.aabb_max_pad = glm::vec4(fixed_max, 0.0f);
        staging_cull_infos_.push_back(cull_info);

        // Print first few world-space results for diagnosis.
        if (uploaded_mesh_count_ == 0 && c < 3) {
            std::printf("  WS cluster[%u] center=(%.3f,%.3f,%.3f) r=%.3f "
                        "aabb=(%.3f..%.3f, %.3f..%.3f, %.3f..%.3f)\n",
                        c,
                        cull_info.bounds_sphere.x, cull_info.bounds_sphere.y,
                        cull_info.bounds_sphere.z, cull_info.bounds_sphere.w,
                        cull_info.aabb_min_pad.x, cull_info.aabb_max_pad.x,
                        cull_info.aabb_min_pad.y, cull_info.aabb_max_pad.y,
                        cull_info.aabb_min_pad.z, cull_info.aabb_max_pad.z);
        }

        // ── Merge vertex/index data for this cluster into global arrays ──
        // Skip if the source mesh geometry has been released (e.g. after
        // GPU upload of the original draw path freed the CPU-side data).
        if (!cluster_mesh.source->vertex_data_ptr ||
            !cluster_mesh.source->faces_ptr ||
            cluster_mesh.source->vertex_data_ptr->empty()) {
            // Still need a draw info entry to keep cluster indexing consistent.
            glsl::ClusterDrawInfo draw_info{};
            // Tag with this mesh's object id even though the cluster draws
            // nothing (index_count == 0) — keeps object_idx meaningful for
            // every entry and avoids a stale/garbage value if the cluster
            // is ever revived.
            draw_info.object_idx = uploaded_mesh_count_;
            staging_draw_infos_.push_back(draw_info);
            continue;
        }

        // Base vertex offset in the merged VB for this cluster's vertices.
        const uint32_t merged_vertex_base =
            static_cast<uint32_t>(staging_vertices_.size());
        const uint32_t merged_index_base =
            static_cast<uint32_t>(staging_indices_.size());

        // Build the vertex remap directly from face data (not from
        // cl.vertex_indices, which may be incomplete). This guarantees
        // every vertex referenced by a triangle is in the remap.
        const auto& src_verts = *cluster_mesh.source->vertex_data_ptr;
        const auto& src_faces = *cluster_mesh.source->faces_ptr;
        const uint32_t src_vert_count =
            static_cast<uint32_t>(src_verts.size());
        const uint32_t src_face_count =
            static_cast<uint32_t>(src_faces.size());
        // Same inverse-transpose as normal_mat above — reuse it so we don't
        // compute another matrix inverse per cluster (one per mesh is fine).
        const glm::mat3& normal_mat3 = normal_mat;

        // Pass 1: scan all faces in this cluster, collect unique vertices,
        // transform to world space, and push into staging VB.
        std::unordered_map<uint32_t, uint32_t> vert_remap;
        vert_remap.reserve(cl.vertex_indices.size());

        for (uint32_t fi : cl.face_indices) {
            if (fi >= src_face_count) continue;
            const auto& face = src_faces[fi];
            for (int k = 0; k < 3; ++k) {
                uint32_t vi = face.v_indices[k];
                if (vi >= src_vert_count || vert_remap.count(vi)) continue;
                uint32_t new_idx = merged_vertex_base +
                    static_cast<uint32_t>(vert_remap.size());
                vert_remap[vi] = new_idx;

                const auto& sv = src_verts[vi];
                BindlessVertex bv;
                bv.position = glm::vec3(
                    model_transform * glm::vec4(sv.position, 1.0f));
                bv.normal = glm::normalize(normal_mat3 * sv.normal);
                bv.uv = sv.uv;
                // Apply the same UV flip that base.vert applies via model_params.flip_uv_coord.
                // FBX files always set m_flip_v_=true (V-axis is inverted vs OpenGL/Vulkan).
                if (drawable_data.m_flip_u_) bv.uv.x = 1.0f - bv.uv.x;
                if (drawable_data.m_flip_v_) bv.uv.y = 1.0f - bv.uv.y;
                // World-space tangent + bitangent sign.  src_tangents[vi].xyz is
                // the orthogonalised object-space tangent from
                // computeMeshTangents(); we transform direction-only with
                // tangent_mat3.  Renormalise after the transform because
                // non-uniform scale would otherwise leave |T| != 1.  The .w
                // bitangent sign is preserved unchanged.
                if (vi < src_tangents.size()) {
                    glm::vec3 ws_T =
                        glm::normalize(tangent_mat3 * glm::vec3(src_tangents[vi]));
                    bv.tangent = glm::vec4(ws_T, src_tangents[vi].w);
                } else {
                    // Defensive fallback for out-of-range indices.
                    bv.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                }
                staging_vertices_.push_back(bv);
            }
        }

        // Pass 2: emit remapped indices for each face.
        for (uint32_t fi : cl.face_indices) {
            if (fi >= src_face_count) continue;
            const auto& face = src_faces[fi];
            auto it0 = vert_remap.find(face.v_indices[0]);
            auto it1 = vert_remap.find(face.v_indices[1]);
            auto it2 = vert_remap.find(face.v_indices[2]);
            if (it0 == vert_remap.end() || it1 == vert_remap.end() ||
                it2 == vert_remap.end()) continue;
            staging_indices_.push_back(it0->second);
            staging_indices_.push_back(it1->second);
            staging_indices_.push_back(it2->second);
        }

        glsl::ClusterDrawInfo draw_info;
        draw_info.index_offset  = merged_index_base;  // firstIndex into merged IB
        draw_info.index_count   = static_cast<uint32_t>(
            staging_indices_.size()) - merged_index_base;  // actual emitted count
        draw_info.vertex_offset = 0;  // absolute indices, no base vertex needed
        {
            uint32_t prim_idx = (c < cluster_prim_map.size())
                ? cluster_prim_map[c] : 0u;
            draw_info.material_idx = getMaterialIdx(prim_idx);
        }
        // Per-object id for DEBUG_RENDER_MODE_OBJECT_ID.  uploaded_mesh_count_
        // is the running global mesh-upload counter (incremented once at the
        // end of this call), so it's stable for every cluster of this mesh
        // and unique across all meshes of all drawables — unlike mesh_idx,
        // which restarts at 0 per drawable and would collide between the
        // exterior and interior scenes.  draw_info here is NOT
        // zero-initialised, so this assignment is required (not just tidy).
        draw_info.object_idx = uploaded_mesh_count_;
        staging_draw_infos_.push_back(draw_info);
    }

    // Record per-mesh cluster range for visibility feedback.
    mesh_cluster_ranges_.push_back({
        static_cast<uint32_t>(staging_cull_infos_.size() - num_clusters),
        num_clusters
    });

    // Stash this mesh's prim_idx → global material_idx map, keyed by the
    // mesh's global object id (== uploaded_mesh_count_ for this upload,
    // since the counter is bumped just below and matches the object_idx
    // baked into every ClusterDrawInfo above).  Lets the collision
    // isolate-debug overlay (application.cpp) translate an isolated
    // collision mesh's (drawable, mesh, prim) into the cluster
    // material_idx its fragments carry, so the textured background can be
    // restricted to that exact source primitive.  prim_to_mat_idx already
    // holds exactly the (prim → material) entries that produced clusters
    // for this mesh.
    if (mesh_prim_material_.size() <= uploaded_mesh_count_)
        mesh_prim_material_.resize(uploaded_mesh_count_ + 1);
    mesh_prim_material_[uploaded_mesh_count_] = prim_to_mat_idx;

    ++uploaded_mesh_count_;
}

// ─── preRegisterVtMaterials ────────────────────────────────────────────────
// Budgeted VT cache warm-up — see header doc.  Mirrors the registration
// logic inside uploadMeshClusters::getMaterialIdx (same cache keys), so a
// later upload of the same textures is a pure cache hit.
int ClusterRenderer::preRegisterVtMaterials(
    const game_object::DrawableData& drawable_data,
    int max_new) {
    if (!vt_manager_ || max_new <= 0) return 0;

    int registered = 0;
    for (const auto& mat : drawable_data.materials_) {
        if (registered >= max_new) break;

        const renderer::TextureInfo* albedo_tex = nullptr;
        if (mat.base_color_idx_ >= 0 &&
            static_cast<size_t>(mat.base_color_idx_) <
                drawable_data.textures_.size()) {
            albedo_tex = &drawable_data.textures_[mat.base_color_idx_];
        }
        if (!albedo_tex) continue;
        const bool has_cpu =
            albedo_tex->cpu_pixels && !albedo_tex->cpu_pixels->empty();
        const bool has_blob =
            albedo_tex->vt_bc7_tiles && !albedo_tex->vt_bc7_tiles->empty();
        if (!(albedo_tex->image || has_cpu || has_blob) ||
            albedo_tex->size.x == 0 || albedo_tex->size.y == 0) {
            continue;
        }
        const void* vt_key = albedo_tex->image
            ? static_cast<const void*>(albedo_tex->image.get())
            : albedo_tex->cpu_pixels
                ? static_cast<const void*>(albedo_tex->cpu_pixels.get())
                : static_cast<const void*>(albedo_tex->vt_bc7_tiles.get());
        if (vt_albedo_id_cache_.find(vt_key) != vt_albedo_id_cache_.end()) {
            continue;   // already warm (valid OR known-failed)
        }

        const renderer::TextureInfo* normal_tex = nullptr;
        if (mat.normal_idx_ >= 0 &&
            static_cast<size_t>(mat.normal_idx_) <
                drawable_data.textures_.size()) {
            normal_tex = &drawable_data.textures_[mat.normal_idx_];
        }
        const auto& nrm_img =
            (normal_tex && normal_tex->image)
                ? normal_tex->image
                : std::shared_ptr<renderer::Image>();

        const uint8_t* px = has_cpu
            ? albedo_tex->cpu_pixels->data()
            : nullptr;
        const uint32_t vid = vt_manager_->registerMaterial(
            px,
            albedo_tex->image,
            nrm_img,
            /*mr_ao*/   nullptr,
            /*emissive*/nullptr,
            albedo_tex->size.x,
            albedo_tex->size.y,
            albedo_tex->vt_bc7_tiles);
        // Cache EVEN failed registrations (pool full, …) so the warm-up
        // doesn't retry the same texture every frame; uploadMeshClusters
        // checks vid validity before using a cached entry.
        vt_albedo_id_cache_[vt_key] = vid;
        if (vid != kInvalidVtId && nrm_img) {
            vt_normal_id_cache_[nrm_img.get()] = vid;
        }
        ++registered;
    }
    return registered;
}

// ─── countPendingVtMaterials ───────────────────────────────────────────────
int ClusterRenderer::countPendingVtMaterials(
    const game_object::DrawableData& drawable_data) const {
    if (!vt_manager_) return 0;
    int pending = 0;
    std::unordered_set<const void*> seen;
    for (const auto& mat : drawable_data.materials_) {
        const renderer::TextureInfo* albedo_tex = nullptr;
        if (mat.base_color_idx_ >= 0 &&
            static_cast<size_t>(mat.base_color_idx_) <
                drawable_data.textures_.size()) {
            albedo_tex = &drawable_data.textures_[mat.base_color_idx_];
        }
        if (!albedo_tex) continue;
        const bool has_cpu =
            albedo_tex->cpu_pixels && !albedo_tex->cpu_pixels->empty();
        const bool has_blob =
            albedo_tex->vt_bc7_tiles && !albedo_tex->vt_bc7_tiles->empty();
        if (!(albedo_tex->image || has_cpu || has_blob) ||
            albedo_tex->size.x == 0 || albedo_tex->size.y == 0) {
            continue;
        }
        const void* vt_key = albedo_tex->image
            ? static_cast<const void*>(albedo_tex->image.get())
            : albedo_tex->cpu_pixels
                ? static_cast<const void*>(albedo_tex->cpu_pixels.get())
                : static_cast<const void*>(albedo_tex->vt_bc7_tiles.get());
        if (!seen.insert(vt_key).second) continue;
        if (vt_albedo_id_cache_.find(vt_key) == vt_albedo_id_cache_.end()) {
            ++pending;
        }
    }
    return pending;
}

// ─── Finalize uploads (create merged GPU SSBOs) ───────────────────

void ClusterRenderer::finalizeUploads() {
    total_clusters_all_meshes_ =
        static_cast<uint32_t>(staging_cull_infos_.size());

    // ── Re-finalize support (editor incremental uploads) ──────────────
    // When merged buffers from a previous finalize exist, drain the GPU
    // first: every buffer below is REPLACED (the shared_ptr reassignment
    // frees the old allocation) and no in-flight frame may still
    // reference it.  This is an editor-time hitch (object placement /
    // transform commit), not a per-frame cost.
    const bool refinalize = (cull_info_buffer_.buffer != nullptr);
    if (refinalize) {
        device_->waitIdle();
    }

    // Log VT registration outcome so we know the pool got populated.
    // Counts unique source images registered (cache size).  If this is
    // zero, vt_manager_ wasn't wired or no materials had textures.
    std::printf("[RVT] registered %zu albedo textures, %zu normal textures with VT pool.\n",
                vt_albedo_id_cache_.size(),
                vt_normal_id_cache_.size());

    if (total_clusters_all_meshes_ == 0) {
        if (refinalize) {
            // Everything was removed (e.g. the editor's last placed
            // object was deleted on top of an empty base).  Park the
            // renderer: draw()/cull() gate on gpu_ready_, so flipping it
            // off retires the stale buffers without destroying them.
            // The next finalize with content re-arms everything.
            gpu_ready_ = false;
            total_merged_vertices_ = 0;
            total_merged_indices_  = 0;
            total_visible_all_meshes_ = 0;
            cluster_to_mesh_.clear();
            mesh_visible_.clear();
        }
        std::printf("[CLUSTER_RENDERER] No clusters to upload.\n");
        return;
    }

    // Create cull_info as HOST_VISIBLE so we can verify the data.
    {
        const uint64_t sz = total_clusters_all_meshes_ * sizeof(glsl::ClusterCullInfo);
        er::Helper::createBuffer(
            device_,
            SET_2_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT, TRANSFER_DST_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            cull_info_buffer_.buffer,
            cull_info_buffer_.memory,
            std::source_location::current(),
            sz,
            staging_cull_infos_.data());

        // Read back and verify first few entries.
        void* mapped = device_->mapMemory(cull_info_buffer_.memory, sz, 0);
        if (mapped) {
            auto* gpu_data = reinterpret_cast<const glsl::ClusterCullInfo*>(mapped);
            const uint32_t n_check = std::min(5u, total_clusters_all_meshes_);
            for (uint32_t i = 0; i < n_check; ++i) {
                std::printf(
                    "  GPU[%u] center=(%.2f,%.2f,%.2f) r=%.2f\n",
                    i,
                    gpu_data[i].bounds_sphere.x,
                    gpu_data[i].bounds_sphere.y,
                    gpu_data[i].bounds_sphere.z,
                    gpu_data[i].bounds_sphere.w);
            }
            device_->unmapMemory(cull_info_buffer_.memory);
        }
    }

    draw_info_buffer_ = createSSBO(
        device_,
        total_clusters_all_meshes_ * sizeof(glsl::ClusterDrawInfo),
        staging_draw_infos_.data());

    // Material params SSBO — one entry per unique material (or per upload call).
    if (!staging_material_params_.empty()) {
        material_params_buffer_ = createSSBO(
            device_,
            staging_material_params_.size() * sizeof(glsl::BindlessMaterialParams),
            staging_material_params_.data());
        std::printf("[CLUSTER_RENDERER] Uploaded %zu material(s) to GPU.\n",
                    staging_material_params_.size());
    } else {
        // Fallback: single white material entry so the shader always has data.
        glsl::BindlessMaterialParams white;
        white.base_color_factor = glm::vec4(1.0f);
        material_params_buffer_ = createSSBO(
            device_,
            sizeof(glsl::BindlessMaterialParams),
            &white);
    }

    // Indirect draw buffer: worst case all clusters visible (opaque bucket).
    indirect_draw_buffer_ = createIndirectSSBO(
        device_,
        total_clusters_all_meshes_ * 5 * sizeof(uint32_t));

    // Atomic draw count buffer (host visible for readback).
    draw_count_buffer_ = createCounterBuffer(device_);

    // Visible cluster indices buffer (opaque bucket only).
    visible_buffer_ = createSSBO(
        device_,
        total_clusters_all_meshes_ * sizeof(uint32_t),
        nullptr);

    // Translucent bucket — same worst-case sizing as opaque so a 100%-glass
    // scene wouldn't overflow.  In practice these allocations are dwarfed
    // by the merged VB/IB so the doubled indirect-buffer cost is fine.
    trans_indirect_draw_buffer_ = createIndirectSSBO(
        device_,
        total_clusters_all_meshes_ * 5 * sizeof(uint32_t));
    trans_draw_count_buffer_ = createCounterBuffer(device_);

    // ── Shadow cull scratch buffers ──────────────────────────────────
    // The shadow cull dispatch needs a separate place to land its
    // translucent-bucket writes so they don't clobber the main path's
    // trans_indirect_draw_buffer_ before drawTranslucentForward reads
    // it.  drawClusterShadow never reads these — they're write-only
    // scratch.  Worst-case sized like the regular translucent buffer.
    shadow_cull_trans_indirect_buffer_ = createIndirectSSBO(
        device_,
        total_clusters_all_meshes_ * 5 * sizeof(uint32_t));
    shadow_cull_trans_count_buffer_ = createCounterBuffer(device_);

    // Scratch visible-buffer for the shadow cull dispatches — see the
    // header doc for why this is necessary (avoids clobbering the main
    // cull's visible_buffer_, which is read back next frame for
    // per-mesh visibility tracking).
    shadow_cull_visible_buffer_ = createSSBO(
        device_,
        total_clusters_all_meshes_ * sizeof(uint32_t),
        nullptr);

    // ── Per-cascade shadow indirect buffers (Option B) ───────────────
    // Allocate one (indirect, count) pair per CSM cascade.  cullShadow
    // populates them per frame (one cluster_cull.comp dispatch per
    // cascade with that cascade's tight VP); drawClusterShadow consumes
    // each cascade's pair via vkCmdDrawIndexedIndirectCount inside its
    // own dynamic-rendering pass on the cascade's depth layer.
    //
    // Each indirect buffer is worst-case sized for "all clusters land
    // in this cascade"; in practice each cascade retains ~5-90% of
    // clusters depending on which slab of the camera frustum it covers.
    //
    // The count buffers are seeded with total_clusters_all_meshes_ as a
    // first-frame fallback — if cullShadow hasn't run yet (e.g. before
    // clusterIndirectActive flips true), drawClusterShadow can still
    // produce correct output by drawing every cluster.  Subsequent
    // frames overwrite this with the actual cull-survivor count.
    //
    // Index/instance/vertex offsets in the indirect commands are NOT
    // pre-populated — cluster_cull.comp writes them per surviving
    // cluster.  The static seed at finalize would draw nothing if
    // cullShadow never runs, but that's an acceptable startup state
    // since clusterIndirectActive being false means the per-mesh
    // shadow path in shadow_object_scene_view_->draw is handling the
    // load anyway.
    for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
        shadow_indirect_draw_buffers_[k] = createIndirectSSBO(
            device_,
            total_clusters_all_meshes_ * 5 * sizeof(uint32_t));
        shadow_draw_count_buffers_[k] = createCounterBuffer(device_);
    }

    // ── Two-pass occlusion buffers ──────────────────────────────────────
    // Persistent visibility bits — one bit per cluster.  Zero-initialised
    // so frame 1's Phase A culls everything (renders nothing) and frame
    // 1's Phase B (with empty Hi-Z) renders the full frustum-visible set,
    // populating the bits for frame 2.
    {
        const uint32_t vis_uint_count =
            (total_clusters_all_meshes_ + 31u) / 32u;
        std::vector<uint32_t> zeros(vis_uint_count, 0u);
        visibility_bit_buffer_ = createSSBO(
            device_,
            vis_uint_count * sizeof(uint32_t),
            zeros.data());
    }
    // Phase A indirect output — same worst-case size as the single-pass
    // opaque buffer.  In steady state Phase A's count converges on the
    // visible-this-frame count, but we size for "all clusters were
    // visible last frame" to avoid overflow during teleport / cut events.
    indirect_draw_buffer_phase_a_ = createIndirectSSBO(
        device_,
        total_clusters_all_meshes_ * 5 * sizeof(uint32_t));
    draw_count_buffer_phase_a_ = createCounterBuffer(device_);

    // Allocate (once — reused on re-finalize) and write descriptor set.
    if (!cull_desc_set_) {
        cull_desc_set_ = device_->createDescriptorSets(
            descriptor_pool_, cull_desc_set_layout_, 1)[0];
    }

    // material_params_buffer_ was just created above; staging_material_params_
    // still has the source data for the size lookup.  total_materials_ isn't
    // assigned until later in finalizeUploads, so use the staging size here.
    const uint32_t mat_count = static_cast<uint32_t>(
        std::max(size_t(1), staging_material_params_.size()));
    writeCullDescriptors(
        device_, cull_desc_set_,
        total_clusters_all_meshes_,
        cull_info_buffer_, draw_info_buffer_,
        indirect_draw_buffer_, draw_count_buffer_,
        visible_buffer_,
        material_params_buffer_, mat_count,
        trans_indirect_draw_buffer_, trans_draw_count_buffer_,
        visibility_bit_buffer_,
        indirect_draw_buffer_phase_a_,
        draw_count_buffer_phase_a_,
        // Hi-Z binding: pass through whatever the application has
        // already provided via setHiZTexture().  May be null on the
        // very first finalizeUploads if the app hasn't called it yet
        // — the shader's sample is gated on use_hiz_cull anyway, so
        // a null binding is fine until the first descriptor refresh.
        hiz_sampler_, hiz_view_);

    // ── Per-cascade shadow cull descriptor sets (Option B) ───────────
    // One descriptor set per cascade; same layout as cull_desc_set_, but
    // bindings 2/3 point at the cascade's dedicated shadow indirect+count
    // pair (so each cascade's dispatch writes its own survivors).
    // Bindings 6/7 point at the shared scratch trans pair — we only
    // ever write into it, never read, so cascades sharing the scratch
    // is fine.  Other bindings reuse the main-cull SSBOs.
    for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
        if (!cull_desc_sets_shadow_[k]) {
            cull_desc_sets_shadow_[k] = device_->createDescriptorSets(
                descriptor_pool_, cull_desc_set_layout_, 1)[0];
        }
        writeCullDescriptors(
            device_, cull_desc_sets_shadow_[k],
            total_clusters_all_meshes_,
            cull_info_buffer_, draw_info_buffer_,
            shadow_indirect_draw_buffers_[k],
            shadow_draw_count_buffers_[k],
            shadow_cull_visible_buffer_,  // scratch, not visible_buffer_
            material_params_buffer_, mat_count,
            shadow_cull_trans_indirect_buffer_,
            shadow_cull_trans_count_buffer_,
            visibility_bit_buffer_,
            indirect_draw_buffer_phase_a_,
            draw_count_buffer_phase_a_,
            hiz_sampler_, hiz_view_);
    }

    // ── DEBUG: validate merged staging data — write to file ──
    {
        FILE* dbg = std::fopen("cluster_debug_dump.txt", "w");
        if (dbg) {
            const uint32_t n_verts = static_cast<uint32_t>(staging_vertices_.size());
            const uint32_t n_idx   = static_cast<uint32_t>(staging_indices_.size());
            const uint32_t n_draw  = static_cast<uint32_t>(staging_draw_infos_.size());
            uint32_t bad_idx = 0, nan_vert = 0;
            for (uint32_t i = 0; i < n_idx; ++i) {
                if (staging_indices_[i] >= n_verts) ++bad_idx;
            }
            for (uint32_t i = 0; i < n_verts; ++i) {
                const auto& v = staging_vertices_[i];
                if (std::isnan(v.position.x) || std::isnan(v.position.y) ||
                    std::isnan(v.position.z)) ++nan_vert;
            }
            std::fprintf(dbg, "VALIDATION: %u verts, %u indices, "
                        "%u draw infos | bad_idx=%u nan_vert=%u\n",
                        n_verts, n_idx, n_draw, bad_idx, nan_vert);

            // Dump first 5 clusters' draw info + ALL indices + vertex positions.
            for (uint32_t ci = 0; ci < std::min(5u, n_draw); ++ci) {
                const auto& di = staging_draw_infos_[ci];
                std::fprintf(dbg, "\ncluster[%u] idx_off=%u idx_cnt=%u vtx_off=%u mat=%u\n",
                            ci, di.index_offset, di.index_count,
                            di.vertex_offset, di.material_idx);
                uint32_t end = std::min(di.index_offset + di.index_count, n_idx);
                for (uint32_t j = di.index_offset; j < end; ++j) {
                    uint32_t vi = staging_indices_[j];
                    if (vi < n_verts) {
                        const auto& v = staging_vertices_[vi];
                        std::fprintf(dbg, "  idx[%u]=%u pos=(%.4f,%.4f,%.4f) "
                                    "n=(%.3f,%.3f,%.3f)\n",
                                    j, vi, v.position.x, v.position.y, v.position.z,
                                    v.normal.x, v.normal.y, v.normal.z);
                    } else {
                        std::fprintf(dbg, "  idx[%u]=%u OUT_OF_RANGE!\n", j, vi);
                    }
                }
            }

            // Also dump vertex position range (AABB of all merged verts).
            glm::vec3 vmin(1e30f), vmax(-1e30f);
            for (uint32_t i = 0; i < n_verts; ++i) {
                vmin = glm::min(vmin, staging_vertices_[i].position);
                vmax = glm::max(vmax, staging_vertices_[i].position);
            }
            std::fprintf(dbg, "\nAll verts AABB: min=(%.3f,%.3f,%.3f) "
                        "max=(%.3f,%.3f,%.3f)\n",
                        vmin.x, vmin.y, vmin.z,
                        vmax.x, vmax.y, vmax.z);

            std::fclose(dbg);
            std::printf("[CLUSTER_RENDERER] Debug dump written to "
                        "cluster_debug_dump.txt\n");
        }
    }

    // ── Create merged vertex/index buffers for bindless rendering ──
    total_merged_vertices_ = static_cast<uint32_t>(staging_vertices_.size());
    total_merged_indices_  = static_cast<uint32_t>(staging_indices_.size());

    if (total_merged_vertices_ > 0) {
        // HOST_VISIBLE for debugging — bypass staging transfer to rule out
        // GPU upload corruption.
        // STORAGE_BUFFER_BIT lets the cluster mesh-shader CSM path read
        // the merged VB as an SSBO (cluster_bindless_shadow.mesh).  The
        // forward / G-buffer paths still bind it through VERTEX_BUFFER_BIT
        // via the input assembler.
        er::Helper::createBuffer(
            device_,
            SET_3_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT,
                            STORAGE_BUFFER_BIT, TRANSFER_DST_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            merged_vertex_buffer_.buffer,
            merged_vertex_buffer_.memory,
            std::source_location::current(),
            total_merged_vertices_ * sizeof(BindlessVertex),
            staging_vertices_.data());
    }

    if (total_merged_indices_ > 0) {
        // STORAGE_BUFFER_BIT for the mesh-shader shadow path (reads indices
        // as a plain uint[] SSBO and computes its own primitive emission).
        er::Helper::createBuffer(
            device_,
            SET_3_FLAG_BITS(BufferUsage, INDEX_BUFFER_BIT,
                            STORAGE_BUFFER_BIT, TRANSFER_DST_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            merged_index_buffer_.buffer,
            merged_index_buffer_.memory,
            std::source_location::current(),
            total_merged_indices_ * sizeof(uint32_t),
            staging_indices_.data());
    }

    gpu_ready_ = true;

    // ── Cluster mesh-shader descriptor set ───────────────────────────
    // The mesh-shader CSM path reads these SSBOs.  Visible to both
    // TASK_BIT_EXT (task shader's per-cluster cull) and MESH_BIT_EXT
    // (mesh shader's emission).
    //   binding 0  cluster cull infos       (frustum + cone bounds)
    //   binding 1  cluster draw infos       (index off/cnt + material_idx)
    //   binding 2  merged vertex buffer     (BindlessVertex[] as float[])
    //   binding 3  merged index buffer      (uint[])
    //   binding 4  bindless material params (flags for double-sided /
    //                                        translucent backface-cull
    //                                        skip — same source the
    //                                        cluster_cull.comp main-camera
    //                                        path reads)
    // The layout is independent of the cull compute's layout (which has
    // 12 bindings and is COMPUTE-visible only).
    if (total_merged_vertices_ > 0 && total_merged_indices_ > 0) {
        if (!cluster_mesh_data_desc_set_layout_) {
            std::vector<er::DescriptorSetLayoutBinding> mesh_bindings(5);
            for (int i = 0; i < 5; ++i) {
                mesh_bindings[i] = er::helper::getBufferDescriptionSetLayoutBinding(
                    i, SET_2_FLAG_BITS(ShaderStage, TASK_BIT_EXT, MESH_BIT_EXT),
                    er::DescriptorType::STORAGE_BUFFER);
            }
            cluster_mesh_data_desc_set_layout_ =
                device_->createDescriptorSetLayout(mesh_bindings);
        }
        if (!cluster_mesh_data_desc_set_) {
            cluster_mesh_data_desc_set_ = device_->createDescriptorSets(
                descriptor_pool_, cluster_mesh_data_desc_set_layout_, 1)[0];
        }

        er::WriteDescriptorList mesh_writes;
        mesh_writes.reserve(5);
        er::Helper::addOneBuffer(mesh_writes, cluster_mesh_data_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 0,
            cull_info_buffer_.buffer,
            static_cast<uint32_t>(
                total_clusters_all_meshes_ * sizeof(glsl::ClusterCullInfo)));
        er::Helper::addOneBuffer(mesh_writes, cluster_mesh_data_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 1,
            draw_info_buffer_.buffer,
            static_cast<uint32_t>(
                total_clusters_all_meshes_ * sizeof(glsl::ClusterDrawInfo)));
        er::Helper::addOneBuffer(mesh_writes, cluster_mesh_data_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 2,
            merged_vertex_buffer_.buffer,
            static_cast<uint32_t>(
                total_merged_vertices_ * sizeof(BindlessVertex)));
        er::Helper::addOneBuffer(mesh_writes, cluster_mesh_data_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 3,
            merged_index_buffer_.buffer,
            static_cast<uint32_t>(total_merged_indices_ * sizeof(uint32_t)));
        // Material params (same buffer the cluster_cull.comp main-camera
        // cull reads at its set=0, binding=5).  Always has at least one
        // entry (white-fallback in the early branch above).
        const uint32_t mat_count_safe = std::max(
            static_cast<uint32_t>(staging_material_params_.size()),
            1u);
        er::Helper::addOneBuffer(mesh_writes, cluster_mesh_data_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 4,
            material_params_buffer_.buffer,
            static_cast<uint32_t>(
                mat_count_safe * sizeof(glsl::BindlessMaterialParams)));
        device_->updateDescriptorSets(mesh_writes);
    }

    std::printf(
        "[CLUSTER_RENDERER] Finalized: %u clusters from %u meshes, "
        "merged VB: %u verts (%zu KB), IB: %u indices (%zu KB).\n",
        total_clusters_all_meshes_, uploaded_mesh_count_,
        total_merged_vertices_,
        total_merged_vertices_ * sizeof(BindlessVertex) / 1024,
        total_merged_indices_,
        total_merged_indices_ * sizeof(uint32_t) / 1024);

    // Store first cluster for display in ImGui.
    if (!staging_cull_infos_.empty()) {
        debug_first_cluster_bounds_ = staging_cull_infos_[0].bounds_sphere;
        fprintf(stderr,
            "[CLUSTER_FINALIZE] staging[0] bounds_sphere=(%.3f,%.3f,%.3f) r=%.3f\n",
            staging_cull_infos_[0].bounds_sphere.x,
            staging_cull_infos_[0].bounds_sphere.y,
            staging_cull_infos_[0].bounds_sphere.z,
            staging_cull_infos_[0].bounds_sphere.w);
    }

    // Debug: save evenly-spaced sample of cluster bounds for bbox drawing.
    {
        debug_sample_clusters_.clear();
        const uint32_t total = total_clusters_all_meshes_;
        const uint32_t want = std::min(kDebugSampleCount, total);
        if (want > 0) {
            const uint32_t step = std::max(1u, total / want);
            for (uint32_t i = 0; i < total && debug_sample_clusters_.size() < want;
                 i += step) {
                debug_sample_clusters_.push_back(staging_cull_infos_[i]);
            }
        }

        for (uint32_t i = 0; i < std::min(5u, (uint32_t)debug_sample_clusters_.size()); ++i) {
            const auto& c = debug_sample_clusters_[i];
            std::printf(
                "  [%u] center=(%.2f,%.2f,%.2f) r=%.2f\n",
                i,
                c.bounds_sphere.x, c.bounds_sphere.y, c.bounds_sphere.z,
                c.bounds_sphere.w);
        }
    }

    // Build per-cluster triangle count array and compute total polygon count.
    cluster_tri_counts_.resize(total_clusters_all_meshes_);
    total_triangles_all_meshes_ = 0;
    for (uint32_t ci = 0; ci < total_clusters_all_meshes_; ++ci) {
        uint32_t tris = staging_draw_infos_[ci].index_count / 3;
        cluster_tri_counts_[ci] = tris;
        total_triangles_all_meshes_ += tris;
    }

    // Build cluster→mesh lookup table for per-mesh visibility readback.
    cluster_to_mesh_.resize(total_clusters_all_meshes_);
    for (uint32_t mi = 0; mi < static_cast<uint32_t>(mesh_cluster_ranges_.size()); ++mi) {
        const auto& r = mesh_cluster_ranges_[mi];
        for (uint32_t ci = r.cluster_start;
             ci < r.cluster_start + r.cluster_count; ++ci) {
            cluster_to_mesh_[ci] = mi;
        }
    }
    // Initialize all meshes as visible until the first readback.
    mesh_visible_.assign(mesh_cluster_ranges_.size(), true);

    std::printf("[CLUSTER_RENDERER] Built cluster→mesh LUT: %u meshes, "
                "%llu total triangles.\n",
                static_cast<uint32_t>(mesh_cluster_ranges_.size()),
                static_cast<unsigned long long>(total_triangles_all_meshes_));

    // Record counts (used by initBindlessPipeline / descriptor rewrites).
    total_materials_ = static_cast<uint32_t>(
        std::max(size_t(1), staging_material_params_.size()));
    // NOTE: the texture staging arrays are NOT padded to
    // MAX_CLUSTER_TEXTURES any more — the descriptor-write loops bounds-
    // check and fall back to dummy_texture_ for empty slots, and in-place
    // padding would break the slot dedup on later (editor) uploads.
    total_textures_ = static_cast<uint32_t>(staging_tex_views_.size());
    total_normal_textures_ = static_cast<uint32_t>(staging_normal_tex_views_.size());

    // Backup material_params (with VT ids populated) so setVtEnabled()
    // can restore them after a user-driven VT-off → VT-on transition.
    // ~48 B per material × ~3K materials = ~150 KB CPU; trivial.
    material_params_backup_ = staging_material_params_;
    vt_enabled_ = true;  // freshly uploaded with VT ids → VT is on.

    // CPU staging is RETAINED (not cleared) so the editor's incremental
    // flow can resetToBaseUploads() + re-stage placed objects + call
    // finalizeUploads() again.  Costs one CPU copy of the merged VB/IB —
    // acceptable for the editor; revisit with a "shipping" mode flag if
    // the retained copy ever matters for the packaged game.
    if (!base_marked_) {
        // Auto-snapshot: the first finalize defines the base scene unless
        // the application marked it explicitly beforehand.
        markBaseUploads();
    }

    // On a RE-finalize the bindless graphics set (if the pipeline is
    // already up) still points at the old, just-replaced buffers —
    // re-point bindings 0..3 now.  Safe: refinalize did waitIdle above.
    if (refinalize && bindless_desc_set_) {
        rewriteBindlessDescriptorsAfterRefinalize();
    }
}

// ─── markBaseUploads / resetToBaseUploads ──────────────────────────────────
// Editor incremental-upload bookkeeping — see header doc.

void ClusterRenderer::markBaseUploads() {
    base_cluster_count_  = staging_cull_infos_.size();
    base_material_count_ = staging_material_params_.size();
    base_vertex_count_   = staging_vertices_.size();
    base_index_count_    = staging_indices_.size();
    base_tex_count_      = staging_tex_views_.size();
    base_normal_tex_count_ = staging_normal_tex_views_.size();
    base_mesh_count_     = uploaded_mesh_count_;
    base_marked_         = true;
    std::printf("[CLUSTER_RENDERER] Base uploads marked: %u meshes, "
                "%zu clusters, %zu verts.\n",
                base_mesh_count_, base_cluster_count_, base_vertex_count_);
}

void ClusterRenderer::resetToBaseUploads() {
    if (!base_marked_) {
        markBaseUploads();
        return;
    }
    // staging_draw_infos_ is 1:1 with staging_cull_infos_ (one entry per
    // cluster, including geometry-less placeholders) — same truncation.
    staging_cull_infos_.resize(base_cluster_count_);
    staging_draw_infos_.resize(base_cluster_count_);
    staging_material_params_.resize(base_material_count_);
    staging_material_names_.resize(base_material_count_);
    staging_vertices_.resize(base_vertex_count_);
    staging_indices_.resize(base_index_count_);
    mesh_cluster_ranges_.resize(base_mesh_count_);
    if (mesh_prim_material_.size() > base_mesh_count_) {
        mesh_prim_material_.resize(base_mesh_count_);
    }
    uploaded_mesh_count_ = base_mesh_count_;

    // Drop the placed tail's texture views too: a removed object's
    // drawable (and its GPU textures) may be destroyed right after this
    // call, and retaining the dead views would feed them back into the
    // descriptor rewrite on the next finalize.  The dedup maps lose the
    // corresponding entries so re-staged objects re-register cleanly.
    // (VT pool registrations are NOT reclaimed — editor-grade leak,
    // bounded by the pool size; a full scene reload resets it.)
    if (staging_tex_views_.size() > base_tex_count_) {
        staging_tex_views_.resize(base_tex_count_);
        for (auto it = staging_tex_slot_map_.begin();
             it != staging_tex_slot_map_.end();) {
            if (it->second >= static_cast<int>(base_tex_count_)) {
                it = staging_tex_slot_map_.erase(it);
            } else {
                ++it;
            }
        }
    }
    if (staging_normal_tex_views_.size() > base_normal_tex_count_) {
        staging_normal_tex_views_.resize(base_normal_tex_count_);
        for (auto it = staging_normal_slot_map_.begin();
             it != staging_normal_slot_map_.end();) {
            if (it->second >= static_cast<int>(base_normal_tex_count_)) {
                it = staging_normal_slot_map_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// ─── setMeshClustersHidden ─────────────────────────────────────────────────
// Poison / restore one mesh's cluster bounding spheres in the HOST_VISIBLE
// cull-info buffer.  See header doc.
void ClusterRenderer::setMeshClustersHidden(
    uint32_t global_mesh_idx, bool hidden) {
    if (!gpu_ready_ || !cull_info_buffer_.memory) return;
    if (global_mesh_idx >= mesh_cluster_ranges_.size()) return;
    const auto& r = mesh_cluster_ranges_[global_mesh_idx];
    if (r.cluster_count == 0) return;
    const uint64_t total_sz =
        total_clusters_all_meshes_ * sizeof(glsl::ClusterCullInfo);
    if ((r.cluster_start + r.cluster_count) > total_clusters_all_meshes_)
        return;
    void* mapped = device_->mapMemory(cull_info_buffer_.memory, total_sz, 0);
    if (!mapped) return;
    auto* infos = reinterpret_cast<glsl::ClusterCullInfo*>(mapped);
    for (uint32_t ci = r.cluster_start;
         ci < r.cluster_start + r.cluster_count; ++ci) {
        if (hidden) {
            infos[ci].bounds_sphere.w = -1e30f;  // fails every frustum test
        } else if (ci < staging_cull_infos_.size()) {
            infos[ci].bounds_sphere = staging_cull_infos_[ci].bounds_sphere;
        }
    }
    device_->unmapMemory(cull_info_buffer_.memory);
}

// ─── rewriteBindlessDescriptorsAfterRefinalize ─────────────────────────────
// Mirrors the descriptor writes at the end of initBindlessPipeline for
// bindings 0..3 only.  The VT bindings (4..10) point at VirtualTexture-
// Manager resources that survive a refinalize untouched.  Caller already
// did device_->waitIdle(), so updating the live set is safe.
void ClusterRenderer::rewriteBindlessDescriptorsAfterRefinalize() {
    er::WriteDescriptorList writes;
    writes.reserve(2 + MAX_CLUSTER_TEXTURES * 2);

    er::Helper::addOneBuffer(writes, bindless_desc_set_,
        er::DescriptorType::STORAGE_BUFFER, 0,
        draw_info_buffer_.buffer,
        static_cast<uint32_t>(
            std::max(1u, total_clusters_all_meshes_) *
            sizeof(glsl::ClusterDrawInfo)));

    er::Helper::addOneBuffer(writes, bindless_desc_set_,
        er::DescriptorType::STORAGE_BUFFER, 1,
        material_params_buffer_.buffer,
        static_cast<uint32_t>(
            total_materials_ * sizeof(glsl::BindlessMaterialParams)));

    for (uint32_t ti = 0; ti < MAX_CLUSTER_TEXTURES; ++ti) {
        auto tex_write = std::make_shared<er::TextureDescriptor>();
        tex_write->binding           = 2;
        tex_write->dst_array_element = ti;
        tex_write->desc_type         = er::DescriptorType::COMBINED_IMAGE_SAMPLER;
        tex_write->desc_set          = bindless_desc_set_;
        tex_write->image_layout      = er::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        tex_write->sampler           = default_sampler_;
        tex_write->texture           =
            (ti < staging_tex_views_.size() && staging_tex_views_[ti])
                ? staging_tex_views_[ti]
                : dummy_texture_.view;
        writes.push_back(tex_write);
    }
    for (uint32_t ti = 0; ti < MAX_CLUSTER_TEXTURES; ++ti) {
        auto nw = std::make_shared<er::TextureDescriptor>();
        nw->binding           = 3;
        nw->dst_array_element = ti;
        nw->desc_type         = er::DescriptorType::COMBINED_IMAGE_SAMPLER;
        nw->desc_set          = bindless_desc_set_;
        nw->image_layout      = er::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        nw->sampler           = default_sampler_;
        nw->texture           =
            (ti < staging_normal_tex_views_.size() && staging_normal_tex_views_[ti])
                ? staging_normal_tex_views_[ti]
                : dummy_texture_.view;
        writes.push_back(nw);
    }

    device_->updateDescriptorSets(writes);
}

// ─── setVtEnabled ──────────────────────────────────────────────────────────
// Toggle the runtime VT path on/off without restarting the scene.  Walks
// material_params_backup_ (the original CPU copy with VT ids populated),
// either uses it as-is (enable) or copies with all four vt_id fields
// overwritten to VT_INVALID_ID (disable), and writes the result to the
// HOST_VISIBLE material_params SSBO.  The shader's existing fallback to
// the legacy bindless texture arrays handles the "all VT ids invalid"
// case automatically — no shader rebuild, no descriptor-set rewrite.
void ClusterRenderer::setVtEnabled(bool enabled) {
    if (enabled == vt_enabled_) return;
    if (material_params_backup_.empty() ||
        !material_params_buffer_.memory) {
        vt_enabled_ = enabled;
        return;
    }
    constexpr uint32_t kInvalidVtId = 0xFFFFFFFFu;
    const size_t bytes =
        material_params_backup_.size() * sizeof(glsl::BindlessMaterialParams);
    if (enabled) {
        // Restore the original VT ids (BIN-equal copy of the upload).
        device_->updateBufferMemory(
            material_params_buffer_.memory,
            bytes,
            material_params_backup_.data());
    } else {
        // Build a temp copy with vt_ids cleared and upload that.  The
        // shader sees VT_INVALID_ID for every layer and routes through
        // base_color_textures[]/normal_textures[] bindless arrays.
        std::vector<glsl::BindlessMaterialParams> off_copy =
            material_params_backup_;
        for (auto& mp : off_copy) {
            mp.albedo_vt_id   = kInvalidVtId;
            mp.normal_vt_id   = kInvalidVtId;
            mp.mr_ao_vt_id    = kInvalidVtId;
            mp.emissive_vt_id = kInvalidVtId;
        }
        device_->updateBufferMemory(
            material_params_buffer_.memory,
            bytes,
            off_copy.data());
    }
    vt_enabled_ = enabled;
    std::printf("[CLUSTER_RENDERER] VT %s — material_params re-uploaded "
                "(%zu materials, %zu KB)\n",
                enabled ? "ENABLED" : "DISABLED",
                material_params_backup_.size(), bytes / 1024u);
}

// ─── applyMaterialCategories ───────────────────────────────────────────────
// Re-upload the per-material flags SSBO with MeshCategory bits packed
// into bits 8..15.  Lookups go through the LLM-backed classifier; the
// (material_name, object_name) pair to query is recorded in
// staging_material_names_[i] alongside staging_material_params_[i] at
// upload time.  The shader's DEBUG_RENDER_MODE_CATEGORY branch reads
// the bits via BINDLESS_MAT_CATEGORY_SHIFT / _MASK and maps each enum
// value to a fixed RGB (same table as collision_debug.frag).
//
// We patch the CPU-side material_params_backup_ (the source of truth
// post-finalize) and then re-upload the GPU buffer in one shot.
//
// Idempotent and safe to call repeatedly; each call wipes the previous
// category bits before OR-ing in the fresh lookup.

void ClusterRenderer::applyMaterialCategories(
    const helper::MaterialClassifier& cls) {
    if (material_params_backup_.empty() ||
        !material_params_buffer_.memory) {
        std::printf(
            "[CLUSTER_RENDERER] applyMaterialCategories: no params to "
            "patch (backup_empty=%d memory_null=%d)\n",
            static_cast<int>(material_params_backup_.empty()),
            static_cast<int>(!material_params_buffer_.memory));
        return;
    }

    // staging_material_names_ should match material_params_backup_ in
    // length and order — they were filled side-by-side in
    // uploadMeshClusters().  Defend against mismatch (e.g. partial
    // upload) by truncating to the shorter length and logging.
    const size_t pair_count = std::min(
        material_params_backup_.size(), staging_material_names_.size());
    if (pair_count != material_params_backup_.size() ||
        pair_count != staging_material_names_.size()) {
        std::printf(
            "[CLUSTER_RENDERER] applyMaterialCategories: name list "
            "length mismatch (params=%zu names=%zu), patching first %zu\n",
            material_params_backup_.size(),
            staging_material_names_.size(),
            pair_count);
    }

    int patched = 0;
    int matched = 0;
    std::array<int, 11> per_cat{};
    for (size_t i = 0; i < pair_count; ++i) {
        const auto& [mat_name, obj_name] = staging_material_names_[i];
        // Category is the LLM classifier's verdict alone -- the
        // name-string surface guard was removed.
        const auto cat = cls.lookup(mat_name, obj_name);
        // Always wipe the previous category bits before writing, so
        // a second call with a different classifier overrides cleanly.
        int& flags = material_params_backup_[i].flags;
        flags &= ~BINDLESS_MAT_CATEGORY_MASK;
        const uint32_t cat_u = static_cast<uint32_t>(cat);
        flags |= static_cast<int>(
            (cat_u << BINDLESS_MAT_CATEGORY_SHIFT) &
            BINDLESS_MAT_CATEGORY_MASK);
        if (cat_u < per_cat.size()) ++per_cat[cat_u];
        if (cat != helper::MeshCategory::Unknown) ++matched;
        ++patched;
    }

    // Re-upload the patched backup to the live SSBO.  Same path the
    // VT toggle uses; the buffer is HOST_VISIBLE | HOST_COHERENT so
    // the write is visible to the next draw without a barrier (apart
    // from the swapchain's natural per-frame synchronisation).
    const size_t bytes =
        material_params_backup_.size() * sizeof(glsl::BindlessMaterialParams);
    device_->updateBufferMemory(
        material_params_buffer_.memory,
        bytes,
        material_params_backup_.data());

    categories_applied_ = true;

    std::printf(
        "[CLUSTER_RENDERER] applyMaterialCategories: patched %d / %d "
        "entries (%d classified non-Unknown). "
        "Unknown=%d Floor=%d Wall=%d Door=%d Object=%d Glass=%d "
        "Ceiling=%d Stairs=%d Vegetation=%d Elevator=%d Ladder=%d\n",
        patched, static_cast<int>(material_params_backup_.size()),
        matched,
        per_cat[0], per_cat[1], per_cat[2], per_cat[3], per_cat[4],
        per_cat[5], per_cat[6], per_cat[7], per_cat[8], per_cat[9],
        per_cat[10]);
}

// ─── setHiZTexture ─────────────────────────────────────────────────────────
// Stash the Hi-Z handles and refresh the cull descriptor's binding 11
// pointer so subsequent dispatches see the new pyramid.  Idempotent and
// cheap; safe to call every swap-chain rebuild.

void ClusterRenderer::setHiZTexture(
    const std::shared_ptr<renderer::Sampler>& sampler,
    const std::shared_ptr<renderer::ImageView>& view,
    const glm::uvec2& size,
    uint32_t mip_count) {

    hiz_sampler_   = sampler;
    hiz_view_      = view;
    hiz_size_      = size;
    hiz_mip_count_ = mip_count;

    // If the cull descriptor set hasn't been allocated yet (called before
    // finalizeUploads), the next writeCullDescriptors inside finalize will
    // pick up the stored handles.  Otherwise patch binding 11 in place.
    //
    // Layout MUST be GENERAL to match the Hi-Z pyramid's actual layout
    // (storage-image writes during build keep it in GENERAL the whole
    // frame; sampler reads tolerate GENERAL).  Declaring
    // SHADER_READ_ONLY_OPTIMAL here used to silently break the cull
    // dispatch on NV — the validation layer's complaint is non-fatal
    // but the sampler reads come back as undefined data, and the
    // diagnostic returns we put in the Hi-Z block never produced any
    // visible cluster-count change because the dispatch ran in a
    // degenerate state.
    if (cull_desc_set_ && sampler && view) {
        er::WriteDescriptorList writes;
        writes.reserve(1);
        er::Helper::addOneTexture(writes, cull_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER, 11,
            sampler, view,
            er::ImageLayout::GENERAL);
        device_->updateDescriptorSets(writes);
    }
}

// ─── Cull (single dispatch for ALL clusters) ──────────────────────

// ── Debug-readback prologue ─────────────────────────────────────────────
// Pulled out of cull() so the deferred path — which now skips the legacy
// cull dispatches entirely in favor of cullPhaseA/B — can still refresh
// the Smart Mesh ImGui stats once per frame.  Reads HOST_VISIBLE memory
// only; no command-buffer recording.  Safe to call at the same point in
// the frame as cull() used to be (after the previous-frame fence wait).
void ClusterRenderer::pollDebugReadback() {
    if (!enabled_ || !gpu_ready_) return;

    // Read visible count (4 bytes).
    void* mapped = device_->mapMemory(
        draw_count_buffer_.memory, sizeof(uint32_t), 0);
    if (mapped) {
        std::memcpy(&total_visible_all_meshes_, mapped, sizeof(uint32_t));
        device_->unmapMemory(draw_count_buffer_.memory);
    }

    // Read visible cluster indices for per-mesh visibility + triangle stats.
    visible_triangles_ = 0;
    if (!mesh_cluster_ranges_.empty() && !cluster_to_mesh_.empty()) {
        mesh_visible_.assign(mesh_cluster_ranges_.size(), false);
        const uint32_t vis_count = std::min(
            total_visible_all_meshes_, total_clusters_all_meshes_);
        if (vis_count > 0) {
            void* vis_mapped = device_->mapMemory(
                visible_buffer_.memory,
                vis_count * sizeof(uint32_t), 0);
            if (vis_mapped) {
                auto* vis_indices = reinterpret_cast<const uint32_t*>(vis_mapped);
                for (uint32_t i = 0; i < vis_count; ++i) {
                    uint32_t ci = vis_indices[i];
                    if (ci < cluster_to_mesh_.size())
                        mesh_visible_[cluster_to_mesh_[ci]] = true;
                    if (ci < cluster_tri_counts_.size())
                        visible_triangles_ += cluster_tri_counts_[ci];
                }
                device_->unmapMemory(visible_buffer_.memory);
            }
        }
    }
}

void ClusterRenderer::cull(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::mat4& view_proj,
    const glm::vec3& camera_pos,
    const glm::mat4& last_view_proj,
    std::optional<bool> hiz_cull_override) {

    if (!enabled_ || !gpu_ready_) return;

    // Store for debug display in ImGui.
    debug_last_vp_ = view_proj;
    debug_last_cam_pos_ = camera_pos;

    // Debug-readback prologue.  Kept inline here so existing callers
    // (probe per-face cull, forward path's main cull) get stats refresh
    // for free without an extra explicit call.  Deferred path now skips
    // cull() entirely and calls pollDebugReadback() directly instead.
    pollDebugReadback();

    if (cpu_cull_mode_) {
        // ── CPU frustum culling path ─────────────────────────────────
        // Extract frustum planes on CPU (same Gribb-Hartmann method as
        // the compute shader) and test each cluster's bounding sphere.
        // Results are written directly into the host-visible indirect
        // draw and counter buffers.

        auto* cull_data = reinterpret_cast<const glsl::ClusterCullInfo*>(
            device_->mapMemory(cull_info_buffer_.memory,
                               total_clusters_all_meshes_ * sizeof(glsl::ClusterCullInfo), 0));

        auto* draw_data = reinterpret_cast<const glsl::ClusterDrawInfo*>(
            device_->mapMemory(draw_info_buffer_.memory,
                               total_clusters_all_meshes_ * sizeof(glsl::ClusterDrawInfo), 0));

        glm::vec4 planes[6];
        planes[0] = glm::vec4(view_proj[0][3]+view_proj[0][0], view_proj[1][3]+view_proj[1][0],
                               view_proj[2][3]+view_proj[2][0], view_proj[3][3]+view_proj[3][0]);
        planes[1] = glm::vec4(view_proj[0][3]-view_proj[0][0], view_proj[1][3]-view_proj[1][0],
                               view_proj[2][3]-view_proj[2][0], view_proj[3][3]-view_proj[3][0]);
        // Bottom — Y-flip (proj[1].y *= -1) swaps bottom/top planes relative
        // to standard Gribb-Hartmann: bottom = row3-row1, top = row3+row1.
        planes[2] = glm::vec4(view_proj[0][3]-view_proj[0][1], view_proj[1][3]-view_proj[1][1],
                               view_proj[2][3]-view_proj[2][1], view_proj[3][3]-view_proj[3][1]);
        // Top (see note above)
        planes[3] = glm::vec4(view_proj[0][3]+view_proj[0][1], view_proj[1][3]+view_proj[1][1],
                               view_proj[2][3]+view_proj[2][1], view_proj[3][3]+view_proj[3][1]);
        // Near — Vulkan depth Z∈[0,1] → near plane = row2 only (not row3+row2).
        planes[4] = glm::vec4(view_proj[0][2], view_proj[1][2],
                               view_proj[2][2], view_proj[3][2]);
        planes[5] = glm::vec4(view_proj[0][3]-view_proj[0][2], view_proj[1][3]-view_proj[1][2],
                               view_proj[2][3]-view_proj[2][2], view_proj[3][3]-view_proj[3][2]);
        for (int i = 0; i < 6; ++i) {
            float len = glm::length(glm::vec3(planes[i]));
            if (len > 0.0001f) planes[i] /= len;
        }

        auto* indirect = reinterpret_cast<uint32_t*>(
            device_->mapMemory(indirect_draw_buffer_.memory,
                               total_clusters_all_meshes_ * 5 * sizeof(uint32_t), 0));

        uint32_t visible_count = 0;

        if (cull_data && draw_data && indirect) {
            for (uint32_t ci = 0; ci < total_clusters_all_meshes_; ++ci) {
                glm::vec3 center(cull_data[ci].bounds_sphere);
                float radius = cull_data[ci].bounds_sphere.w;

                bool visible = true;
                for (int p = 0; p < 6; ++p) {
                    float d = glm::dot(glm::vec3(planes[p]), center) + planes[p].w;
                    if (d < -radius) { visible = false; break; }
                }
                if (!visible) continue;

                // Backface cone cull (mirrors cluster_cull.comp::isBackfacing).
                // cone_cutoff stores sin(θ_max) — see cluster_mesh.cpp derivation.
                // Cull when dot(cam→cluster, cone_axis) ≥ sin(θ_max).
                glm::vec3 cone_axis(cull_data[ci].cone_axis_cutoff);
                float cone_cutoff = cull_data[ci].cone_axis_cutoff.w;
                if (cone_cutoff >= 0.0f) {
                    glm::vec3 view_dir = glm::normalize(center - camera_pos);
                    if (glm::dot(view_dir, cone_axis) >= cone_cutoff)
                        continue;
                }

                uint32_t base = visible_count * 5;
                indirect[base + 0] = draw_data[ci].index_count;
                indirect[base + 1] = 1;
                indirect[base + 2] = draw_data[ci].index_offset;
                indirect[base + 3] = draw_data[ci].vertex_offset;
                indirect[base + 4] = 0;
                ++visible_count;
            }
        }

        if (indirect)
            device_->unmapMemory(indirect_draw_buffer_.memory);
        if (draw_data)
            device_->unmapMemory(draw_info_buffer_.memory);
        if (cull_data)
            device_->unmapMemory(cull_info_buffer_.memory);

        // Write visible count.
        {
            void* mapped = device_->mapMemory(
                draw_count_buffer_.memory, sizeof(uint32_t), 0);
            if (mapped) {
                std::memcpy(mapped, &visible_count, sizeof(uint32_t));
                device_->unmapMemory(draw_count_buffer_.memory);
            }
        }
        // The CPU cull path doesn't bucket clusters by alpha mode (yet);
        // explicitly zero the translucent counter so the second indirect
        // draw in draw() is a no-op rather than reading stale commands
        // left over from a prior GPU-cull frame.  Glass simply won't
        // render under CPU cull until this path learns about the bucket.
        {
            uint32_t zero = 0;
            void* mapped = device_->mapMemory(
                trans_draw_count_buffer_.memory, sizeof(uint32_t), 0);
            if (mapped) {
                std::memcpy(mapped, &zero, sizeof(uint32_t));
                device_->unmapMemory(trans_draw_count_buffer_.memory);
            }
        }
        total_visible_all_meshes_ = visible_count;

        // Barrier: host writes → indirect draw reads.
        er::BufferResourceInfo ssbo_host_write = {
            SET_FLAG_BIT(Access, HOST_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, HOST_BIT) };
        er::BufferResourceInfo indirect_read = {
            SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
            SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };

        cmd_buf->addBufferBarrier(
            indirect_draw_buffer_.buffer,
            ssbo_host_write, indirect_read);
        cmd_buf->addBufferBarrier(
            draw_count_buffer_.buffer,
            ssbo_host_write, indirect_read);
        // Translucent counter was zeroed above — also needs a host→indirect
        // barrier so draw()'s drawIndexedIndirectCount sees the zero.
        cmd_buf->addBufferBarrier(
            trans_draw_count_buffer_.buffer,
            ssbo_host_write, indirect_read);

    } else {
        // ── GPU compute culling path ─────────────────────────────────

        // Zero the counters via vkCmdFillBuffer (GPU-side).
        //
        // We used to do this with a host-side mapMemory + memcpy, which
        // works fine when cull() runs once per frame but races when
        // it's called multiple times per frame.  Specifically: the
        // probe pass calls cull() THEN drawIndexedIndirectCount; the
        // main cull THEN re-issues cull() (host-zeroing the same
        // buffer) BEFORE the GPU has finished the probe's indirect
        // read.  HOST_COHERENT memory is visible to the GPU at
        // arbitrary times between barriers, so the host-zero from the
        // second cull can clobber the count the probe's
        // drawIndexedIndirectCount is about to read — making the
        // probe pass occasionally render zero clusters and the cube
        // face go blank for one frame.  Whole-scene flicker.
        //
        // vkCmdFillBuffer is a GPU command, properly ordered with
        // surrounding compute / indirect operations via the ssbo
        // barriers below.  No host involvement, no race.
        cmd_buf->fillBuffer(draw_count_buffer_.buffer,
                            0, sizeof(uint32_t), 0);
        cmd_buf->fillBuffer(trans_draw_count_buffer_.buffer,
                            0, sizeof(uint32_t), 0);

        // Barrier: counter-zero (GPU transfer) + previous indirect
        // reads → compute.
        er::BufferResourceInfo ssbo_transfer_write = {
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        er::BufferResourceInfo ssbo_compute_rw = {
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
        er::BufferResourceInfo prev_indirect_read = {
            SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
            SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };

        cmd_buf->addBufferBarrier(
            draw_count_buffer_.buffer,
            ssbo_transfer_write,
            ssbo_compute_rw);
        cmd_buf->addBufferBarrier(
            trans_draw_count_buffer_.buffer,
            ssbo_transfer_write,
            ssbo_compute_rw);

        cmd_buf->addBufferBarrier(
            indirect_draw_buffer_.buffer,
            prev_indirect_read,
            ssbo_compute_rw);
        cmd_buf->addBufferBarrier(
            trans_indirect_draw_buffer_.buffer,
            prev_indirect_read,
            ssbo_compute_rw);

        // Bind pipeline + descriptor set + push constants — ONCE.
        cmd_buf->bindPipeline(
            er::PipelineBindPoint::COMPUTE,
            cull_pipeline_);

        glsl::ClusterCullPushConstants push{};
        push.view_proj = view_proj;
        push.camera_pos_pad = glm::vec4(camera_pos, 0.0f);
        push.total_clusters = total_clusters_all_meshes_;
        push.total_bvh_nodes = 0;
        push.lod_error_threshold = 1.0f;
        push.use_bvh = debug_distance_cull_ ? 3 : 0;
        // Hi-Z occlusion cull is gated by the menu toggle; the shader
        // currently treats this as a no-op stub but the flag is plumbed
        // through end-to-end so wiring up the actual sampler later is
        // a shader-only change.
        const bool effective_hiz_cull =
            hiz_cull_override.value_or(use_hiz_occlusion_cull_);
        push.use_hiz_cull = effective_hiz_cull ? 1u : 0u;
        push.pad0 = 0u;
        push.last_view_proj = last_view_proj;
        push.hiz_size_mips_pad = glm::vec4(
            float(hiz_size_.x), float(hiz_size_.y),
            float(hiz_mip_count_), 0.0f);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            cull_pipeline_layout_,
            &push,
            sizeof(push));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            cull_pipeline_layout_,
            { cull_desc_set_ });

        // Single dispatch for ALL clusters.
        uint32_t groups = (total_clusters_all_meshes_ + 63) / 64;
        cmd_buf->dispatch(groups, 1);

        // Barrier: compute writes → indirect draw reads.  Both buckets.
        er::BufferResourceInfo indirect_read = {
            SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
            SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };

        cmd_buf->addBufferBarrier(
            indirect_draw_buffer_.buffer,
            ssbo_compute_rw,
            indirect_read);
        cmd_buf->addBufferBarrier(
            draw_count_buffer_.buffer,
            ssbo_compute_rw,
            indirect_read);
        cmd_buf->addBufferBarrier(
            trans_indirect_draw_buffer_.buffer,
            ssbo_compute_rw,
            indirect_read);
        cmd_buf->addBufferBarrier(
            trans_draw_count_buffer_.buffer,
            ssbo_compute_rw,
            indirect_read);
    }
}

// ─── Per-cascade shadow cull (Option B) ──────────────────────────────────
// CSM_CASCADE_COUNT compute dispatches, each running cluster_cull.comp
// against one cascade's tight VP and emitting survivors into that
// cascade's dedicated shadow indirect/count buffer.  This is the
// optimisation that makes per-cascade rendering a win: cascade 0 only
// retains ~5% of clusters, cascade 5 retains most; summed across all
// cascades the total triangle work is ~2× the union, not 6× as it
// would be for any "draw everything to every layer" approach (GS
// broadcast, multiview).
//
// Each dispatch shares pipeline / per-cluster SSBOs with the main cull
// but binds its own cull_desc_sets_shadow_[k], whose bindings 2/3
// point at shadow_indirect_draw_buffers_[k] / shadow_draw_count_
// buffers_[k].  Translucent emits all land in the shared scratch trans
// pair which drawClusterShadow ignores.
//
// light_dir = FROM-sun-TO-scene unit vector.  Translated into a
// synthetic camera_pos at infinity along -light_dir so the shader's
// cone test becomes a directional-light backface check (a cluster
// whose normals all face away from the sun is self-occluded by its
// own front face and is safe to drop from shadow rendering).
void ClusterRenderer::cullShadow(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::array<glm::mat4, CSM_CASCADE_COUNT>& cascade_vps,
    const glm::vec3& light_dir) {

    if (!enabled_ || !gpu_ready_) return;
    if (cpu_cull_mode_) return;  // CPU cull path doesn't currently support shadow output
    if (!cull_desc_sets_shadow_[0]) return;  // never finalized

    er::BufferResourceInfo ssbo_transfer_write = {
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    er::BufferResourceInfo ssbo_compute_rw = {
        SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::BufferResourceInfo prev_indirect_read = {
        SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
        SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };
    er::BufferResourceInfo indirect_read = {
        SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
        SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };

    // Zero EVERY cascade's counter once before any dispatch — the shared
    // scratch translucent counter too.  Using vkCmdFillBuffer keeps the
    // resets in the GPU command stream (proper ordering against prior
    // indirect reads from last frame's draw).
    for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
        cmd_buf->fillBuffer(shadow_draw_count_buffers_[k].buffer,
                            0, sizeof(uint32_t), 0);
    }
    cmd_buf->fillBuffer(shadow_cull_trans_count_buffer_.buffer,
                        0, sizeof(uint32_t), 0);

    // Sync those resets, plus last frame's indirect reads on every
    // cascade's indirect buffer, with the compute work that follows.
    for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
        cmd_buf->addBufferBarrier(
            shadow_draw_count_buffers_[k].buffer,
            ssbo_transfer_write, ssbo_compute_rw);
        cmd_buf->addBufferBarrier(
            shadow_indirect_draw_buffers_[k].buffer,
            prev_indirect_read, ssbo_compute_rw);
    }
    cmd_buf->addBufferBarrier(
        shadow_cull_trans_count_buffer_.buffer,
        ssbo_transfer_write, ssbo_compute_rw);
    cmd_buf->addBufferBarrier(
        shadow_cull_trans_indirect_buffer_.buffer,
        prev_indirect_read, ssbo_compute_rw);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        cull_pipeline_);

    // Synthetic camera_pos for directional-light backface culling — see
    // function header for the derivation.  1e6 is comfortably beyond any
    // plausible world bounds without overflowing the cone-test dot.
    const glm::vec3 light_unit = glm::normalize(light_dir);
    const glm::vec3 synth_cam = -light_unit * 1.0e6f;

    const uint32_t groups = (total_clusters_all_meshes_ + 63) / 64;

    // One dispatch per cascade.  Push constants change per dispatch
    // (view_proj only); descriptor set changes per dispatch (each
    // cascade has its own indirect/count target).
    for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
        glsl::ClusterCullPushConstants push{};
        push.view_proj           = cascade_vps[k];
        push.camera_pos_pad      = glm::vec4(synth_cam, 0.0f);
        push.total_clusters      = total_clusters_all_meshes_;
        push.total_bvh_nodes     = 0;
        push.lod_error_threshold = 1.0f;
        push.use_bvh             = 0;
        push.use_hiz_cull        = 0;     // no shadow-space Hi-Z yet
        push.cull_phase          = 0;
        push.pad0                = 0;
        push.pad1                = 0;
        push.last_view_proj      = glm::mat4(1.0f);
        push.hiz_size_mips_pad   = glm::vec4(0.0f);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            cull_pipeline_layout_,
            &push, sizeof(push));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            cull_pipeline_layout_,
            { cull_desc_sets_shadow_[k] });

        cmd_buf->dispatch(groups, 1);
    }

    // Make every cascade's cull writes visible to the subsequent
    // drawIndexedIndirectCount in drawClusterShadow.
    for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
        cmd_buf->addBufferBarrier(
            shadow_indirect_draw_buffers_[k].buffer,
            ssbo_compute_rw, indirect_read);
        cmd_buf->addBufferBarrier(
            shadow_draw_count_buffers_[k].buffer,
            ssbo_compute_rw, indirect_read);
    }
}

// ─── Two-pass occlusion culling (Nanite-style) ────────────────────────────
// Phase A: cluster_cull.comp with cull_phase=1.  Reads visibility bits
// from last frame's Phase B output, emits opaque draws to the dedicated
// phase-A indirect buffer, no Hi-Z test, no translucents.
//
// Phase B: cluster_cull.comp with cull_phase=2.  Tests every cluster
// against frustum + backface + Hi-Z (built from Phase A's depth between
// the two cull dispatches).  Writes the canonical "visible this frame"
// set via atomicOr into visibility_bit_buffer_, fills the standard
// opaque + translucent indirect buffers.
//
// Both share the same compute pipeline / descriptor set as cull() —
// only the push-constant cull_phase value differs.  Buffer barriers
// reset the per-phase counters and ensure ordering between the two
// cull dispatches and their respective indirect-draw consumers.

namespace {

// Helper for the per-phase cull dispatch — builds and pushes a
// ClusterCullPushConstants with the requested phase, leaving the rest
// (view_proj, camera_pos, etc.) at the supplied values.  The descriptor
// set, pipeline, and counter-reset barriers are caller-supplied so
// each phase can sequence them differently.
}  // anonymous namespace

void ClusterRenderer::cullPhaseA(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::mat4& view_proj,
    const glm::vec3& camera_pos) {

    if (!enabled_ || !gpu_ready_) return;
    if (cpu_cull_mode_) return;   // CPU path doesn't support two-pass

    // Reset Phase A's draw counter.  The visibility-bit buffer is
    // expected to hold last frame's Phase B output — we DON'T clear it
    // here; the orchestrator does so between Phase A and Phase B.
    cmd_buf->fillBuffer(draw_count_buffer_phase_a_.buffer,
                        0, sizeof(uint32_t), 0);

    er::BufferResourceInfo ssbo_transfer_write = {
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    er::BufferResourceInfo ssbo_compute_rw = {
        SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::BufferResourceInfo prev_indirect_read = {
        SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
        SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };

    cmd_buf->addBufferBarrier(
        draw_count_buffer_phase_a_.buffer,
        ssbo_transfer_write, ssbo_compute_rw);
    cmd_buf->addBufferBarrier(
        indirect_draw_buffer_phase_a_.buffer,
        prev_indirect_read, ssbo_compute_rw);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE, cull_pipeline_);

    glsl::ClusterCullPushConstants push{};
    push.view_proj = view_proj;
    push.camera_pos_pad = glm::vec4(camera_pos, 0.0f);
    push.total_clusters = total_clusters_all_meshes_;
    push.use_bvh = 0;
    push.use_hiz_cull = 0u;     // Phase A skips Hi-Z; the shader also
                                // gates this internally on cull_phase != 1.
    push.cull_phase = 1u;
    push.last_view_proj = glm::mat4(1.0f);
    push.hiz_size_mips_pad = glm::vec4(0.0f);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        cull_pipeline_layout_, &push, sizeof(push));
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        cull_pipeline_layout_, { cull_desc_set_ });

    uint32_t groups = (total_clusters_all_meshes_ + 63) / 64;
    cmd_buf->dispatch(groups, 1);

    er::BufferResourceInfo indirect_read = {
        SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
        SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };
    cmd_buf->addBufferBarrier(
        indirect_draw_buffer_phase_a_.buffer,
        ssbo_compute_rw, indirect_read);
    cmd_buf->addBufferBarrier(
        draw_count_buffer_phase_a_.buffer,
        ssbo_compute_rw, indirect_read);
}

void ClusterRenderer::cullPhaseB(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::mat4& view_proj,
    const glm::vec3& camera_pos) {

    if (!enabled_ || !gpu_ready_) return;
    if (cpu_cull_mode_) return;

    // Reset standard opaque + translucent counters.  Visibility bits are
    // cleared separately by clearVisibilityBuffer() — that has to happen
    // BEFORE this dispatch but AFTER Phase A consumed the previous bits.
    cmd_buf->fillBuffer(draw_count_buffer_.buffer,
                        0, sizeof(uint32_t), 0);
    cmd_buf->fillBuffer(trans_draw_count_buffer_.buffer,
                        0, sizeof(uint32_t), 0);

    er::BufferResourceInfo ssbo_transfer_write = {
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    er::BufferResourceInfo ssbo_compute_rw = {
        SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::BufferResourceInfo prev_indirect_read = {
        SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
        SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };

    cmd_buf->addBufferBarrier(
        draw_count_buffer_.buffer,
        ssbo_transfer_write, ssbo_compute_rw);
    cmd_buf->addBufferBarrier(
        trans_draw_count_buffer_.buffer,
        ssbo_transfer_write, ssbo_compute_rw);
    cmd_buf->addBufferBarrier(
        indirect_draw_buffer_.buffer,
        prev_indirect_read, ssbo_compute_rw);
    cmd_buf->addBufferBarrier(
        trans_indirect_draw_buffer_.buffer,
        prev_indirect_read, ssbo_compute_rw);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE, cull_pipeline_);

    glsl::ClusterCullPushConstants push{};
    push.view_proj = view_proj;
    push.camera_pos_pad = glm::vec4(camera_pos, 0.0f);
    push.total_clusters = total_clusters_all_meshes_;
    push.use_bvh = debug_distance_cull_ ? 3u : 0u;
    // Hi-Z occlusion test is gated on the menu toggle so the default
    // two-pass behaviour is "frustum + backface only" (stable).  Flip
    // use_hiz_occlusion_cull_ in the cluster debug menu to evaluate the
    // current Hi-Z heuristic.  The pyramid is built from THIS frame's
    // Phase A depth, so the shader's reprojection uses CURRENT
    // view_proj — no last-frame matrix needed.
    push.use_hiz_cull = use_hiz_occlusion_cull_ ? 1u : 0u;
    push.cull_phase = 2u;
    // Phase B reprojects against the CURRENT frame's view-proj because
    // the Hi-Z pyramid was built from THIS frame's Phase A depth (not
    // last frame's).  So pass view_proj for both.
    push.last_view_proj = view_proj;
    push.hiz_size_mips_pad = glm::vec4(
        float(hiz_size_.x), float(hiz_size_.y),
        float(hiz_mip_count_), 0.0f);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        cull_pipeline_layout_, &push, sizeof(push));
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        cull_pipeline_layout_, { cull_desc_set_ });

    uint32_t groups = (total_clusters_all_meshes_ + 63) / 64;
    cmd_buf->dispatch(groups, 1);

    er::BufferResourceInfo indirect_read = {
        SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT),
        SET_FLAG_BIT(PipelineStage, DRAW_INDIRECT_BIT) };
    cmd_buf->addBufferBarrier(
        indirect_draw_buffer_.buffer,
        ssbo_compute_rw, indirect_read);
    cmd_buf->addBufferBarrier(
        draw_count_buffer_.buffer,
        ssbo_compute_rw, indirect_read);
    cmd_buf->addBufferBarrier(
        trans_indirect_draw_buffer_.buffer,
        ssbo_compute_rw, indirect_read);
    cmd_buf->addBufferBarrier(
        trans_draw_count_buffer_.buffer,
        ssbo_compute_rw, indirect_read);
}

void ClusterRenderer::clearVisibilityBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {
    if (!gpu_ready_ || !visibility_bit_buffer_.buffer) return;

    const uint32_t vis_uint_count =
        (total_clusters_all_meshes_ + 31u) / 32u;
    const uint64_t bytes = uint64_t(vis_uint_count) * sizeof(uint32_t);
    if (bytes == 0) return;

    // ── Pre-fill barrier ─────────────────────────────────────────────────
    // Phase A's compute dispatch (which ran BEFORE this clear) reads the
    // visibility bits.  Without this barrier the GPU is free to reorder
    // Phase A's reads with this transfer write, which can let the fill
    // race the read and produce frames where Phase A sees partially-
    // cleared bits — those frames render nothing (or a sparse subset)
    // for Phase A, the Hi-Z is built from incomplete depth, Phase B
    // compensates by drawing extra, and the next frame inverts the bug
    // because the bits are correct again.  Net visual: flicker.
    er::BufferResourceInfo ssbo_compute_read = {
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::BufferResourceInfo ssbo_transfer_write = {
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    cmd_buf->addBufferBarrier(
        visibility_bit_buffer_.buffer,
        ssbo_compute_read, ssbo_transfer_write);

    cmd_buf->fillBuffer(
        visibility_bit_buffer_.buffer,
        /*offset*/ 0,
        /*size*/ bytes,
        /*data*/ 0u);

    // ── Post-fill barrier ────────────────────────────────────────────────
    // Make the clear visible to the cull compute that runs next (Phase B).
    er::BufferResourceInfo ssbo_compute_rw = {
        SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    cmd_buf->addBufferBarrier(
        visibility_bit_buffer_.buffer,
        ssbo_transfer_write, ssbo_compute_rw);
}

uint32_t ClusterRenderer::drawOpaqueGBufferPhaseA(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {
    if (!gpu_ready_ || !bindless_gbuffer_pipeline_ || !bindless_desc_set_) {
        return 0;
    }

    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= PBR_MATERIAL_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets[PBR_MATERIAL_PARAMS_SET] = bindless_desc_set_;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, bindless_gbuffer_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        bindless_pipeline_layout_, all_desc_sets);
    cmd_buf->bindVertexBuffers(
        0, { merged_vertex_buffer_.buffer }, { 0 });
    cmd_buf->bindIndexBuffer(
        merged_index_buffer_.buffer, 0, er::IndexType::UINT32);

    cmd_buf->drawIndexedIndirectCount(
        indirect_draw_buffer_phase_a_, 0,
        draw_count_buffer_phase_a_, 0,
        total_clusters_all_meshes_);
    return total_visible_all_meshes_;
}

// ─── Init bindless graphics pipeline ──────────────────────────────

void ClusterRenderer::initBindlessPipeline(
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::PipelineRenderbufferFormats& framebuffer_format) {

    if (!gpu_ready_ || total_merged_vertices_ == 0) {
        std::printf("[CLUSTER_RENDERER] Skipping bindless pipeline init — "
                    "no merged geometry.\n");
        return;
    }

    // ── Bindless descriptor set layout ──
    // binding 0: ClusterDrawInfo[]        SSBO (index/count per cluster)
    // binding 1: BindlessMaterialParams[] SSBO (base colour + tex idx per material)
    // binding 2: sampler2D[MAX_CLUSTER_TEXTURES]  (base colour texture array — legacy bindless)
    // binding 3: sampler2D[MAX_CLUSTER_TEXTURES]  (normal map texture array — legacy bindless)
    // ── RVT (Runtime Virtual Texture) bindings ──
    // The bindings below expose the VirtualTextureManager's pool textures
    // and page tables to the fragment shader so material samples can flow
    // through `vtSampleAlbedo / vtSampleNormal / ...` instead of the
    // legacy bindless arrays.  vt_sample.glsl.h declares the shader-side
    // resource names used by the helpers (vt_pool_albedo, vt_page_table,
    // etc.); the actual `layout(set=..., binding=...)` decorations live
    // in cluster_bindless.frag and reference these slots.
    //
    // binding 4: sampler2D vt_pool_albedo
    // binding 5: sampler2D vt_pool_normal
    // binding 6: sampler2D vt_pool_mr_ao
    // binding 7: sampler2D vt_pool_emissive
    // binding 8:  VtPageTable SSBO (uint[])
    // binding 9:  VtMeta      SSBO (VirtualTextureMeta[])
    // binding 10: VtFeedback  SSBO (uint[])  — streaming requests
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(11);
        bindings[0] = er::helper::getBufferDescriptionSetLayoutBinding(
            0, SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
               SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::STORAGE_BUFFER);
        bindings[1] = er::helper::getBufferDescriptionSetLayoutBinding(
            1, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::STORAGE_BUFFER);
        // Base-colour texture array.
        auto tex_binding = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            2, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        tex_binding.descriptor_count = MAX_CLUSTER_TEXTURES;
        bindings[2] = tex_binding;
        // Normal-map texture array.
        auto norm_binding = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            3, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        norm_binding.descriptor_count = MAX_CLUSTER_TEXTURES;
        bindings[3] = norm_binding;
        // VT pool samplers (one per layer — albedo, normal, mr_ao, emissive).
        for (uint32_t l = 0; l < 4; ++l) {
            auto vt_pool_binding = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
                4u + l, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER);
            bindings[4u + l] = vt_pool_binding;
        }
        // VT page table SSBO.
        bindings[8] = er::helper::getBufferDescriptionSetLayoutBinding(
            8, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::STORAGE_BUFFER);
        // VT meta SSBO.
        bindings[9] = er::helper::getBufferDescriptionSetLayoutBinding(
            9, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::STORAGE_BUFFER);
        // VT feedback SSBO — fragment shader writes one tile-key per
        // 8×8 screen block; the streamer reads at frame end.
        bindings[10] = er::helper::getBufferDescriptionSetLayoutBinding(
            10, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::STORAGE_BUFFER);
        bindless_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // ── Pipeline layout ──
    // Build the final descriptor set layout list:
    //   sets 0..PBR_MATERIAL_PARAMS_SET-1  — from global_desc_set_layouts
    //   set  PBR_MATERIAL_PARAMS_SET (2)   — our bindless cluster set
    //   sets PBR_MATERIAL_PARAMS_SET+1..N  — remainder of global (SKIN=3,
    //                                         RUNTIME_LIGHTS=4, etc.)
    // This allows the caller to provide RUNTIME_LIGHTS at set 4 without
    // displacing our bindless set.
    er::DescriptorSetLayoutList all_layouts;
    for (uint32_t i = 0; i < PBR_MATERIAL_PARAMS_SET; ++i) {
        all_layouts.push_back(
            i < static_cast<uint32_t>(global_desc_set_layouts.size())
                ? global_desc_set_layouts[i] : nullptr);
    }
    const uint32_t bindless_set_index = PBR_MATERIAL_PARAMS_SET;
    all_layouts.push_back(bindless_desc_set_layout_);
    for (uint32_t i = PBR_MATERIAL_PARAMS_SET + 1;
         i < static_cast<uint32_t>(global_desc_set_layouts.size()); ++i) {
        all_layouts.push_back(global_desc_set_layouts[i]);
    }

    // Vulkan requires every set slot in the pipeline layout to have a valid
    // (non-null) descriptor set layout — gaps are not allowed.  Replace any
    // nullptr slots (e.g. SKIN_PARAMS_SET = 3, which we don't use) with an
    // empty layout so createPipelineLayout doesn't crash on .get().
    auto empty_layout = device_->createDescriptorSetLayout({});
    for (auto& layout : all_layouts) {
        if (!layout) layout = empty_layout;
    }

    // No push constants needed — camera comes from VIEW_PARAMS_SET UBO,
    // vertices are already in world space.
    bindless_pipeline_layout_ = device_->createPipelineLayout(
        all_layouts, {}, std::source_location::current());

    // ── Shader modules ──
    er::ShaderModuleList shader_modules(2);
    shader_modules[0] = er::helper::loadShaderModule(
        device_, "cluster_bindless_vert.spv",
        er::ShaderStageFlagBits::VERTEX_BIT,
        std::source_location::current());
    shader_modules[1] = er::helper::loadShaderModule(
        device_, "cluster_bindless_frag.spv",
        er::ShaderStageFlagBits::FRAGMENT_BIT,
        std::source_location::current());

    // ── Vertex input description (matches BindlessVertex struct) ──
    std::vector<er::VertexInputBindingDescription> binding_descs(1);
    binding_descs[0].binding = 0;
    binding_descs[0].stride = sizeof(BindlessVertex);
    binding_descs[0].input_rate = er::VertexInputRate::VERTEX;

    std::vector<er::VertexInputAttributeDescription> attrib_descs(4);
    // location 0: vec3 position
    attrib_descs[0].binding  = 0;
    attrib_descs[0].location = 0;
    attrib_descs[0].format   = er::Format::R32G32B32_SFLOAT;
    attrib_descs[0].offset   = offsetof(BindlessVertex, position);
    // location 1: vec3 normal
    attrib_descs[1].binding  = 0;
    attrib_descs[1].location = 1;
    attrib_descs[1].format   = er::Format::R32G32B32_SFLOAT;
    attrib_descs[1].offset   = offsetof(BindlessVertex, normal);
    // location 2: vec2 uv
    attrib_descs[2].binding  = 0;
    attrib_descs[2].location = 2;
    attrib_descs[2].format   = er::Format::R32G32_SFLOAT;
    attrib_descs[2].offset   = offsetof(BindlessVertex, uv);
    // location 3: vec4 tangent (xyz = world-space tangent, w = bitangent sign).
    // Used by cluster_bindless.frag to apply normal mapping without the per-
    // fragment dFdx/dFdy reconstruction that produced sparkle on shaded
    // surfaces.  See BindlessVertex doc + computeMeshTangents() for details.
    attrib_descs[3].binding  = 0;
    attrib_descs[3].location = 3;
    attrib_descs[3].format   = er::Format::R32G32B32A32_SFLOAT;
    attrib_descs[3].offset   = offsetof(BindlessVertex, tangent);

    // ── Graphics pipeline ──
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology      = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    // The cluster pipeline renders all geometry in one draw — some primitives are
    // double-sided (plants, leaves). Disable backface culling globally; the
    // fragment shader flips N via gl_FrontFacing for back faces of double-sided
    // materials, and single-sided back faces are lit but not discarded (acceptable
    // for an approximate cluster pass).
    er::RasterizationStateOverride raster_override{};
    raster_override.override_double_sided = true;
    raster_override.double_sided          = true;

    bindless_pipeline_ = device_->createPipeline(
        bindless_pipeline_layout_,
        binding_descs,
        attrib_descs,
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        framebuffer_format,
        raster_override,
        std::source_location::current());

    // ── OIT translucent pipeline (Weighted Blended OIT) ──────────────
    // McGuire & Bavoil 2013.  cluster_bindless.frag is recompiled with
    // -DOIT_OUTPUT into cluster_bindless_oit_frag.spv, which writes:
    //   • location 0  (accum ): vec4(rgb*α*w, α*w)   — additive blend
    //   • location 1  (reveal): float α              — Π(1−α) blend
    //
    // Per-attachment blends:
    //   accum : src=ONE,  dst=ONE,                 op=ADD
    //   reveal: src=ZERO, dst=ONE_MINUS_SRC_COLOR, op=ADD  (multiplicative)
    //
    // Depth-test ON (so opaque geometry in front rejects translucent),
    // depth-write OFF (translucent fragments don't occlude one another in
    // the depth buffer).  Resolved by oit_composite_pipeline_ below.
    //
    // Renders into the kOitAccumFormat / kOitRevealFormat attachments —
    // NOT into the host's color buffer directly.  draw() begins a fresh
    // dynamic rendering pass on those targets between the opaque indirect
    // draw and the composite.
    constexpr er::Format kOitAccumFormat  = er::Format::R16G16B16A16_SFLOAT;
    constexpr er::Format kOitRevealFormat = er::Format::R8_UNORM;
    {
        // ── Forward alpha-blended translucent pipeline ─────────────────
        //
        // Replaces the prior WBOIT (Weighted Blended OIT) pipeline + the
        // separate fullscreen composite pass.  Glass now uses standard
        // hardware alpha blending against the scene's single colour
        // attachment, with no OIT accum/reveal targets and no composite.
        //
        // Why we switched: the OIT path's accum/reveal accumulators +
        // composite resolve were producing unpredictable results on
        // glass — "fully translucent" frames where the discard rule
        // (reveal > 0.99 OR accum.a < 1e-3) fired even when glass should
        // have been visible, plus weird per-frame colour shifts from
        // the IBL mini-cube's mip-0 temporal dither feeding the
        // chromatic refraction lookup.  WBOIT is correct in principle
        // but fragile here: any NaN in accum poisons the whole pixel,
        // and reveal's multiplicative blend can saturate to 1.0 in a
        // way that's indistinguishable from "no glass drew here."
        //
        // Standard alpha blending is simpler, depth-order-dependent
        // (acceptable for a cluster scene where the translucent set is
        // small and per-cluster sort by depth is good enough for the
        // visual fidelity bar), and gives stable predictable output
        // every frame.  Glass cluster z-sort happens at cull time via
        // the per-cluster bounds; the cluster set is small enough that
        // the residual ordering errors are rare.
        //
        // Blend equation (matches base.frag's translucent path):
        //   src color:    SRC_ALPHA
        //   dst color:    ONE_MINUS_SRC_ALPHA
        //   src alpha:    ONE                  (so accumulated alpha is
        //   dst alpha:    ONE_MINUS_SRC_ALPHA   reported correctly for
        //                                       any post-effect that
        //                                       reads dst alpha)
        auto ab_blend = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS),
            /*blend_enable*/ true,
            /*src color*/    er::BlendFactor::SRC_ALPHA,
            /*dst color*/    er::BlendFactor::ONE_MINUS_SRC_ALPHA,
            /*color op*/     er::BlendOp::ADD,
            /*src alpha*/    er::BlendFactor::ONE,
            /*dst alpha*/    er::BlendFactor::ONE_MINUS_SRC_ALPHA,
            /*alpha op*/     er::BlendOp::ADD);

        // Historical note: PipelineColorBlendStateCreateInfo USED TO
        // hold a raw pointer (color_blending.attachments = vec.data())
        // into the input vector, which dangled when callers passed an
        // inline initializer-list — calling with `{ ab_blend }` created
        // a temporary that died at the end of the statement, leaving
        // a wild pointer that createPipeline later dereferenced.  That
        // bug made glass invisible whenever two translucent pipelines
        // were created back-to-back (the second's temporary landed in
        // the first's freed memory and corrupted it).  The struct was
        // changed to OWN its attachments vector, so this is now safe
        // either inline or via a named vector.
        std::vector<er::PipelineColorBlendAttachmentState>
            ab_blend_attachments = { ab_blend };
        auto ab_blend_state =
            std::make_shared<er::PipelineColorBlendStateCreateInfo>(
                er::helper::fillPipelineColorBlendStateCreateInfo(
                    ab_blend_attachments));

        // Depth test on (opaque geometry in front rejects glass behind);
        // depth write OFF (so multiple glass surfaces in depth can all
        // contribute via the blend — if we wrote depth, only the
        // front-most glass would survive and any glass behind it
        // would be rejected by the next translucent cluster's depth
        // test, which would defeat the alpha blend entirely).
        auto ab_depth_stencil =
            std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
                er::helper::fillPipelineDepthStencilStateCreateInfo(
                    /*depth_test_enable */ true,
                    /*depth_write_enable*/ false,
                    er::CompareOp::LESS_OR_EQUAL));

        er::GraphicPipelineInfo ab_info = graphic_pipeline_info;
        ab_info.blend_state_info   = ab_blend_state;
        ab_info.depth_stencil_info = ab_depth_stencil;

        // Same vertex shader, swap fragment to the ALPHA_BLEND_OUTPUT
        // variant.  That variant runs the same glass IBL math as the
        // OIT variant did, but writes a single (rgb, α) to location 0
        // for the hardware blend above to consume.
        er::ShaderModuleList ab_shader_modules(2);
        ab_shader_modules[0] = shader_modules[0];   // cluster_bindless_vert.spv
        ab_shader_modules[1] = er::helper::loadShaderModule(
            device_, "cluster_bindless_alphablend_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

        // Framebuffer format: ONE colour attachment (the scene colour
        // buffer's format, same as the opaque forward pipeline).
        er::PipelineRenderbufferFormats ab_fmt;
        ab_fmt.color_formats = { framebuffer_format.color_formats.empty()
                                   ? er::Format::R16G16B16A16_SFLOAT
                                   : framebuffer_format.color_formats[0] };
        ab_fmt.depth_format  = framebuffer_format.depth_format;

        bindless_translucent_pipeline_ = device_->createPipeline(
            bindless_pipeline_layout_,   // same layout — same descriptor sets
            binding_descs,
            attrib_descs,
            input_assembly,
            ab_info,
            ab_shader_modules,
            ab_fmt,
            raster_override,
            std::source_location::current());
    }

    // ── OIT (WBOIT) translucent pipeline — McGuire & Bavoil 2013 ──────
    // Coexists with the alpha-blend pipeline above.  Selected by the
    // application via drawTranslucentOit() vs drawTranslucentForward()
    // — independent code paths, no in-function dispatch.
    //
    // Root-cause fix: PipelineColorBlendStateCreateInfo now owns its
    // attachments vector (previously a raw pointer into the caller's
    // input vector — see the historical-note comment in the alpha-
    // blend block above for what that bug looked like).  Creating
    // this second pipeline now safely coexists with the first.
    {
        auto accum_blend = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS),
            /*blend_enable*/ true,
            /*src color*/    er::BlendFactor::ONE,
            /*dst color*/    er::BlendFactor::ONE,
            /*color op*/     er::BlendOp::ADD,
            /*src alpha*/    er::BlendFactor::ONE,
            /*dst alpha*/    er::BlendFactor::ONE,
            /*alpha op*/     er::BlendOp::ADD);
        auto reveal_blend = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, R_BIT),  // single-channel
            /*blend_enable*/ true,
            /*src color*/    er::BlendFactor::ZERO,
            /*dst color*/    er::BlendFactor::ONE_MINUS_SRC_COLOR,
            /*color op*/     er::BlendOp::ADD,
            /*src alpha*/    er::BlendFactor::ZERO,
            /*dst alpha*/    er::BlendFactor::ONE_MINUS_SRC_ALPHA,
            /*alpha op*/     er::BlendOp::ADD);

        // Named-vector pattern — see the alpha-blend block for why
        // an inline initializer-list here would corrupt
        // PipelineColorBlendStateCreateInfo's dangling pointer.
        std::vector<er::PipelineColorBlendAttachmentState>
            oit_blend_attachments = { accum_blend, reveal_blend };
        auto oit_blend_state =
            std::make_shared<er::PipelineColorBlendStateCreateInfo>(
                er::helper::fillPipelineColorBlendStateCreateInfo(
                    oit_blend_attachments));

        auto oit_depth_stencil =
            std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
                er::helper::fillPipelineDepthStencilStateCreateInfo(
                    /*depth_test_enable */ true,
                    /*depth_write_enable*/ false,
                    er::CompareOp::LESS_OR_EQUAL));

        er::GraphicPipelineInfo oit_info = graphic_pipeline_info;
        oit_info.blend_state_info   = oit_blend_state;
        oit_info.depth_stencil_info = oit_depth_stencil;

        er::ShaderModuleList oit_shader_modules(2);
        oit_shader_modules[0] = shader_modules[0];  // vertex (shared SPV)
        oit_shader_modules[1] = er::helper::loadShaderModule(
            device_, "cluster_bindless_oit_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

        er::PipelineRenderbufferFormats oit_fmt;
        oit_fmt.color_formats = { kOitAccumFormat, kOitRevealFormat };
        oit_fmt.depth_format  = framebuffer_format.depth_format;

        bindless_translucent_oit_pipeline_ = device_->createPipeline(
            bindless_pipeline_layout_,
            binding_descs,
            attrib_descs,
            input_assembly,
            oit_info,
            oit_shader_modules,
            oit_fmt,
            raster_override,
            std::source_location::current());
    }

    // ── OIT composite resources ──────────────────────────────────────
    // The composite is a fullscreen pass that samples accum + reveal and
    // blends the resolved colour over the existing scene.  Owns its own
    // sampler, descriptor set layout, pipeline layout, and pipeline; the
    // descriptor set itself is rewritten whenever the OIT targets are
    // (re)created in ensureOitTargets().
    {
        oit_composite_sampler_ = device_->createSampler(
            er::Filter::NEAREST,                         // 1:1 fullscreen
            er::SamplerAddressMode::CLAMP_TO_EDGE,
            er::SamplerMipmapMode::NEAREST,
            0.0f,
            std::source_location::current());

        std::vector<er::DescriptorSetLayoutBinding> comp_bindings(2);
        comp_bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            0, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        comp_bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            1, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        oit_composite_desc_set_layout_ =
            device_->createDescriptorSetLayout(comp_bindings);

        oit_composite_pipeline_layout_ = device_->createPipelineLayout(
            { oit_composite_desc_set_layout_ }, {},
            std::source_location::current());

        // Composite pipeline: fullscreen triangle, no depth, alpha-blend
        // resolved colour over the scene.  src=SRC_ALPHA / dst=ONE_MINUS_SRC_ALPHA
        // gives:  out = resolved.rgb * (1 − reveal) + dst.rgb * reveal
        auto comp_blend_attach = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS),
            /*blend_enable*/ true,
            er::BlendFactor::SRC_ALPHA, er::BlendFactor::ONE_MINUS_SRC_ALPHA,
            er::BlendOp::ADD,
            er::BlendFactor::ONE, er::BlendFactor::ZERO,
            er::BlendOp::ADD);
        auto comp_blend_state =
            std::make_shared<er::PipelineColorBlendStateCreateInfo>(
                er::helper::fillPipelineColorBlendStateCreateInfo(
                    { comp_blend_attach }));
        // Depth-write ON with ALWAYS compare: the composite shader emits
        // gl_FragDepth = 0.99999 at every glass pixel (everywhere it doesn't
        // discard), which marks those pixels as "translucent layer present"
        // in the depth buffer.  The downstream sky envmap pass uses
        // LESS_OR_EQUAL against a sky depth of 1.0; with our 0.99999 stamp
        // sky's test becomes 1.0 <= 0.99999 → false, so sky no longer
        // overwrites resolved-glass colour.  Without this, glass with sky
        // visible behind it (windows, glass facades) gets fully painted
        // over by the subsequent sky pass and reads on screen as "missing".
        auto comp_depth_stencil =
            std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
                er::helper::fillPipelineDepthStencilStateCreateInfo(
                    /*depth_test_enable */ true,
                    /*depth_write_enable*/ true,
                    er::CompareOp::ALWAYS));
        auto comp_raster = std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo(
                false, false, er::PolygonMode::FILL,
                SET_FLAG_BIT(CullMode, NONE)));
        auto comp_ms = std::make_shared<er::PipelineMultisampleStateCreateInfo>(
            er::helper::fillPipelineMultisampleStateCreateInfo());

        er::GraphicPipelineInfo comp_info;
        comp_info.blend_state_info   = comp_blend_state;
        comp_info.rasterization_info = comp_raster;
        comp_info.ms_info            = comp_ms;
        comp_info.depth_stencil_info = comp_depth_stencil;

        er::ShaderModuleList comp_shaders(2);
        comp_shaders[0] = er::helper::loadShaderModule(
            device_, "full_screen_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
        comp_shaders[1] = er::helper::loadShaderModule(
            device_, "oit_composite_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

        // Composite framebuffer format: host color + host depth.  Depth is
        // attached so the shader can stamp gl_FragDepth = 0.99999 at glass
        // pixels (see comp_depth_stencil above for why this is required).
        er::PipelineRenderbufferFormats comp_fmt;
        comp_fmt.color_formats = { framebuffer_format.color_formats[0] };
        comp_fmt.depth_format  = framebuffer_format.depth_format;

        er::PipelineInputAssemblyStateCreateInfo comp_input_assembly;
        comp_input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
        comp_input_assembly.restart_enable = false;

        oit_composite_pipeline_ = device_->createPipeline(
            oit_composite_pipeline_layout_,
            {}, {},                      // no vertex bindings — VS generates a fullscreen tri
            comp_input_assembly,
            comp_info,
            comp_shaders,
            comp_fmt,
            er::RasterizationStateOverride{},
            std::source_location::current());
    }

    // ── Allocate the bindless descriptor set ──
    auto desc_sets = device_->createDescriptorSets(
        descriptor_pool_, bindless_desc_set_layout_, 1);
    bindless_desc_set_ = desc_sets[0];

    // ── Write descriptors ──
    {
        er::WriteDescriptorList writes;
        writes.reserve(2 + MAX_CLUSTER_TEXTURES * 2);

        // binding 0: ClusterDrawInfo SSBO
        er::Helper::addOneBuffer(writes, bindless_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 0,
            draw_info_buffer_.buffer,
            static_cast<uint32_t>(
                std::max(1u, total_clusters_all_meshes_) *
                sizeof(glsl::ClusterDrawInfo)));

        // binding 1: BindlessMaterialParams SSBO
        er::Helper::addOneBuffer(writes, bindless_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 1,
            material_params_buffer_.buffer,
            static_cast<uint32_t>(
                total_materials_ * sizeof(glsl::BindlessMaterialParams)));

        // binding 2: base_color_textures[MAX_CLUSTER_TEXTURES]
        for (uint32_t ti = 0; ti < MAX_CLUSTER_TEXTURES; ++ti) {
            auto tex_write = std::make_shared<er::TextureDescriptor>();
            tex_write->binding           = 2;
            tex_write->dst_array_element = ti;
            tex_write->desc_type         = er::DescriptorType::COMBINED_IMAGE_SAMPLER;
            tex_write->desc_set          = bindless_desc_set_;
            tex_write->image_layout      = er::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            tex_write->sampler           = default_sampler_;
            tex_write->texture           =
                (ti < staging_tex_views_.size() && staging_tex_views_[ti])
                    ? staging_tex_views_[ti]
                    : dummy_texture_.view;
            writes.push_back(tex_write);
        }

        // binding 3: normal_textures[MAX_CLUSTER_TEXTURES]
        for (uint32_t ti = 0; ti < MAX_CLUSTER_TEXTURES; ++ti) {
            auto nw = std::make_shared<er::TextureDescriptor>();
            nw->binding           = 3;
            nw->dst_array_element = ti;
            nw->desc_type         = er::DescriptorType::COMBINED_IMAGE_SAMPLER;
            nw->desc_set          = bindless_desc_set_;
            nw->image_layout      = er::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            nw->sampler           = default_sampler_;
            nw->texture           =
                (ti < staging_normal_tex_views_.size() && staging_normal_tex_views_[ti])
                    ? staging_normal_tex_views_[ti]
                    : dummy_texture_.view;
            writes.push_back(nw);
        }

        // ── VT bindings 4..9 ─────────────────────────────────────────
        // If a VT manager is wired up, point the descriptor at its pool
        // textures + page-table / meta SSBOs.  Otherwise fall back to the
        // dummy texture / a small placeholder buffer just so the
        // descriptor set is fully written (Vulkan requires every binding
        // referenced by the pipeline layout to be valid even if the
        // shader never reads it on a given invocation).
        if (vt_manager_) {
            const auto vt_sampler = vt_manager_->getPoolSampler();
            for (uint32_t l = 0; l < 4; ++l) {
                auto layer = static_cast<VtLayer>(l);
                er::Helper::addOneTexture(writes, bindless_desc_set_,
                    er::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    4u + l,
                    vt_sampler,
                    vt_manager_->getPoolImageView(layer),
                    er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
            // Page table SSBO.  Bind the full buffer (UINT32_MAX → driver
            // clamps to actual buffer size).  Sizes match what the
            // VirtualTextureManager allocates in its constructor.
            er::Helper::addOneBuffer(writes, bindless_desc_set_,
                er::DescriptorType::STORAGE_BUFFER, 8,
                vt_manager_->getPageTableBuffer(),
                vt_manager_->getPageTableBufferBytes());
            // Meta SSBO.
            er::Helper::addOneBuffer(writes, bindless_desc_set_,
                er::DescriptorType::STORAGE_BUFFER, 9,
                vt_manager_->getMetaBuffer(),
                vt_manager_->getMetaBufferBytes());
            // VT streaming feedback SSBO.  Fragment shader writes one
            // tile-key per 8×8 screen block; VirtualTextureManager::tick
            // drains the buffer at frame end.
            er::Helper::addOneBuffer(writes, bindless_desc_set_,
                er::DescriptorType::STORAGE_BUFFER, 10,
                vt_manager_->getFeedbackBuffer(),
                vt_manager_->getFeedbackBufferBytes());
        } else {
            // Fallback: dummy texture for the four pool slots.  No
            // safe placeholder for the SSBOs without allocating one;
            // shaders gate on `mat.albedo_vt_id != INVALID` so the SSBO
            // reads are dead code in this branch — the descriptor set
            // still has to exist though, which means we'd need a real
            // buffer.  In practice vt_manager_ is always non-null in
            // the engine's main flow (initialised in initVulkan), so
            // this branch logs a warning and skips writing the SSBO
            // bindings — the resulting validation error makes the
            // misconfiguration obvious during development.
            for (uint32_t l = 0; l < 4; ++l) {
                er::Helper::addOneTexture(writes, bindless_desc_set_,
                    er::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    4u + l, default_sampler_, dummy_texture_.view,
                    er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
            std::printf("[CLUSTER_RENDERER] WARNING: vt_manager_ not set "
                        "during writeBindlessDescriptors. VT bindings 8/9 "
                        "left UNWRITTEN — call setVtManager() before "
                        "initBindlessPipeline().\n");
        }

        device_->updateDescriptorSets(writes);
    }

    // NOTE: texture staging views are RETAINED (they used to be cleared
    // here) — the editor's re-finalize path needs them to (a) keep the
    // slot-dedup maps meaningful for later uploads and (b) rewrite the
    // bindless texture arrays after the merged buffers are replaced.
    // The views are shared_ptrs into DrawableData textures that stay
    // alive anyway; the extra cost is just the vector itself.

    std::printf("[CLUSTER_RENDERER] Bindless graphics pipeline created. "
                "Bindless set at index %u.\n", bindless_set_index);
}

// ─── ensureOitTargets ──────────────────────────────────────────────
// Lazily (re)create the OIT accum + reveal targets at the requested size.
// First call allocates them; subsequent calls reallocate only when the
// size changes (e.g. window resize).  Also (re)allocates the composite
// descriptor set so it points at the current views.
//
// Both targets are created with COLOR_ATTACHMENT | SAMPLED so the OIT
// translucent pass can write them, then the composite pass can sample
// them.  Initial layout is UNDEFINED; the OIT pass's load_op = CLEAR
// transitions them to COLOR_ATTACHMENT_OPTIMAL on first use.
void ClusterRenderer::ensureOitTargets(const glm::uvec2& size) {
    if (oit_target_size_ == size && oit_accum_tex_.image && oit_reveal_tex_.image) {
        return;
    }

    if (oit_accum_tex_.image)  oit_accum_tex_.destroy(device_);
    if (oit_reveal_tex_.image) oit_reveal_tex_.destroy(device_);

    const auto col_usage =
        SET_FLAG_BIT(ImageUsage, COLOR_ATTACHMENT_BIT) |
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT);

    er::Helper::create2DTextureImage(
        device_, er::Format::R16G16B16A16_SFLOAT, size, 1,
        oit_accum_tex_, col_usage,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    er::Helper::create2DTextureImage(
        device_, er::Format::R8_UNORM, size, 1,
        oit_reveal_tex_, col_usage,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    oit_target_size_ = size;

    // (Re)write the composite descriptor set against the new views.
    // persistent pool: allocate once, reuse across resize — the OIT
    // targets are rebuilt at the new size above, so the set's bindings
    // are re-pointed at the fresh accum/reveal views below, but the set
    // handle itself is reused rather than reallocated (reallocating on
    // every size change would leak into the now-persistent pool).
    if (descriptor_pool_ && oit_composite_desc_set_layout_) {
        if (!oit_composite_desc_set_) {
            oit_composite_desc_set_ = device_->createDescriptorSets(
                descriptor_pool_, oit_composite_desc_set_layout_, 1)[0];
        }

        er::WriteDescriptorList writes;
        er::Helper::addOneTexture(writes, oit_composite_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER, 0,
            oit_composite_sampler_, oit_accum_tex_.view,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        er::Helper::addOneTexture(writes, oit_composite_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER, 1,
            oit_composite_sampler_, oit_reveal_tex_.view,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        device_->updateDescriptorSets(writes);
    }
}

// ─── Draw — opaque pass + WBOIT translucent pass + composite ──────────────

// ─── drawOpaqueOnly ─────────────────────────────────────────────────────
// Opaque-only draw used by the dynamic cubemap face capture path.
//
// The full draw() method below also runs OIT + composite, which need
// accum/reveal targets sized to the caller's screen_size.  When the
// dynamic cubemap calls draw() with edge_×edge_ = 512×512 *and* the
// main render path then calls draw() with the swapchain size, the OIT
// helper destroys+reallocates the accum/reveal cubes between the two
// calls.  That destruction happens during command-buffer recording but
// the recorded probe-pass commands still hold VkImage handles for the
// freed targets, which causes DEVICE_LOST at submission time.
//
// Glass / OIT content is irrelevant for ambient SH probes (they
// integrate diffuse irradiance over the hemisphere; the contribution
// of partly-transparent surfaces is already captured implicitly by
// what's behind them).  So we skip OIT here entirely — only the
// opaque cluster draw runs, into the caller's already-bound render
// pass.  No size-dependent state, no reallocation hazard.
uint32_t ClusterRenderer::drawOpaqueOnly(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {
    if (!gpu_ready_ || !bindless_pipeline_ || !bindless_desc_set_) {
        return 0;
    }

    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= PBR_MATERIAL_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets[PBR_MATERIAL_PARAMS_SET] = bindless_desc_set_;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, bindless_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        bindless_pipeline_layout_,
        all_desc_sets);
    cmd_buf->bindVertexBuffers(
        0, { merged_vertex_buffer_.buffer }, { 0 });
    cmd_buf->bindIndexBuffer(
        merged_index_buffer_.buffer, 0, er::IndexType::UINT32);

    cmd_buf->drawIndexedIndirectCount(
        indirect_draw_buffer_, 0,
        draw_count_buffer_, 0,
        total_clusters_all_meshes_);
    return total_visible_all_meshes_;
}

uint32_t ClusterRenderer::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    const std::shared_ptr<renderer::ImageView>& color_view,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const glm::uvec2& screen_size) {

    if (!gpu_ready_ || !bindless_pipeline_ || !bindless_desc_set_) {
        return 0;
    }

    // total_visible_all_meshes_ was already set by cull() at the start of
    // this frame.  See the long comment below for why we don't re-read.
    const uint32_t prev_visible = total_visible_all_meshes_;

    // ── Build full descriptor set list ──
    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= PBR_MATERIAL_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets[PBR_MATERIAL_PARAMS_SET] = bindless_desc_set_;

    auto bind_geometry_state = [&](const std::shared_ptr<er::Pipeline>& pipe) {
        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, pipe);
        cmd_buf->setViewports(viewports);
        cmd_buf->setScissors(scissors);
        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::GRAPHICS,
            bindless_pipeline_layout_,
            all_desc_sets);
        cmd_buf->bindVertexBuffers(0, { merged_vertex_buffer_.buffer }, { 0 });
        cmd_buf->bindIndexBuffer(
            merged_index_buffer_.buffer, 0, er::IndexType::UINT32);
    };

    // ── Opaque + alpha-mask only ───────────────────────────────────────
    // ClusterRenderer::draw() is now the OPAQUE entry point only.  The
    // translucent (alpha-blended glass) draw is issued separately by
    // the application AFTER the sky envmap pass — see the glass-draw
    // block in application.cpp::drawScene.  This split ensures glass
    // alpha-blends correctly OVER the sky background at pixels where
    // no opaque geometry sits behind the glass; if we drew glass here
    // (before sky), the sky envmap's LESS_OR_EQUAL-vs-depth=1.0 test
    // would overwrite glass pixels whose depth stayed at the cleared
    // far value (depth_write is OFF on the alpha-blend pipeline, by
    // design, so multi-layer glass can blend without depth-rejecting
    // itself).
    bind_geometry_state(bindless_pipeline_);
    cmd_buf->drawIndexedIndirectCount(
        indirect_draw_buffer_, 0,
        draw_count_buffer_, 0,
        total_clusters_all_meshes_);

    (void)color_view;
    (void)depth_view;
    (void)screen_size;
    return prev_visible;
}

// ─── Deferred G-buffer pipeline + draw ────────────────────────────────────
// Builds a third variant of the bindless pipeline using the same vertex
// shader and the cluster_bindless.frag GBUFFER_OUTPUT branch (3 RTs, no
// in-shader lighting).  Reuses bindless_pipeline_layout_ / bindless_desc_
// set_ — only the colour-attachment formats and the fragment SPV change.
//
// drawOpaqueGBuffer() then issues exactly the same indirect draw as the
// forward opaque pipeline, but with this G-buffer pipeline bound, so the
// per-cluster cull SSBOs / vertex layout / texture arrays remain shared.
void ClusterRenderer::initBindlessGBufferPipeline(
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::PipelineRenderbufferFormats& gbuffer_format) {

    // No-op if the forward init didn't run (e.g. no merged geometry) — the
    // application will simply fall back to the forward path.
    if (!gpu_ready_ || !bindless_pipeline_layout_) {
        return;
    }
    if (bindless_gbuffer_pipeline_) {
        return;   // already built
    }

    // Vertex input description — must match BindlessVertex exactly.  Mirror
    // the layout from initBindlessPipeline rather than refactoring it out
    // to keep this addition minimally invasive.
    std::vector<er::VertexInputBindingDescription> binding_descs(1);
    binding_descs[0].binding    = 0;
    binding_descs[0].stride     = sizeof(BindlessVertex);
    binding_descs[0].input_rate = er::VertexInputRate::VERTEX;

    // VertexInputAttributeDescription has no list-init operator= — assign
    // each field explicitly, mirroring initBindlessPipeline above.
    std::vector<er::VertexInputAttributeDescription> attrib_descs(4);
    attrib_descs[0].binding  = 0;
    attrib_descs[0].location = 0;
    attrib_descs[0].format   = er::Format::R32G32B32_SFLOAT;
    attrib_descs[0].offset   = offsetof(BindlessVertex, position);
    attrib_descs[1].binding  = 0;
    attrib_descs[1].location = 1;
    attrib_descs[1].format   = er::Format::R32G32B32_SFLOAT;
    attrib_descs[1].offset   = offsetof(BindlessVertex, normal);
    attrib_descs[2].binding  = 0;
    attrib_descs[2].location = 2;
    attrib_descs[2].format   = er::Format::R32G32_SFLOAT;
    attrib_descs[2].offset   = offsetof(BindlessVertex, uv);
    attrib_descs[3].binding  = 0;
    attrib_descs[3].location = 3;
    attrib_descs[3].format   = er::Format::R32G32B32A32_SFLOAT;
    attrib_descs[3].offset   = offsetof(BindlessVertex, tangent);

    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology       = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    er::RasterizationStateOverride raster_override{};
    raster_override.override_double_sided = true;
    raster_override.double_sided          = true;

    // 3 colour attachments — no blending: each G-buffer texel is owned by a
    // single fragment after depth test.  The application supplies the
    // attachment formats; we build a no-blend state with N attachments to
    // match.
    std::vector<er::PipelineColorBlendAttachmentState> gbuf_no_blend(
        gbuffer_format.color_formats.size(),
        er::helper::fillPipelineColorBlendAttachmentState());
    auto gbuf_blend_state =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(gbuf_no_blend));

    er::GraphicPipelineInfo gbuf_info = graphic_pipeline_info;
    gbuf_info.blend_state_info = gbuf_blend_state;

    er::ShaderModuleList gbuf_shader_modules(2);
    gbuf_shader_modules[0] = er::helper::loadShaderModule(
        device_, "cluster_bindless_vert.spv",
        er::ShaderStageFlagBits::VERTEX_BIT,
        std::source_location::current());
    gbuf_shader_modules[1] = er::helper::loadShaderModule(
        device_, "cluster_bindless_gbuf_frag.spv",
        er::ShaderStageFlagBits::FRAGMENT_BIT,
        std::source_location::current());

    bindless_gbuffer_pipeline_ = device_->createPipeline(
        bindless_pipeline_layout_,
        binding_descs,
        attrib_descs,
        input_assembly,
        gbuf_info,
        gbuf_shader_modules,
        gbuffer_format,
        raster_override,
        std::source_location::current());
}

uint32_t ClusterRenderer::drawOpaqueGBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {
    if (!gpu_ready_ || !bindless_gbuffer_pipeline_ || !bindless_desc_set_) {
        return 0;
    }

    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= PBR_MATERIAL_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets[PBR_MATERIAL_PARAMS_SET] = bindless_desc_set_;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, bindless_gbuffer_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        bindless_pipeline_layout_,
        all_desc_sets);
    cmd_buf->bindVertexBuffers(
        0, { merged_vertex_buffer_.buffer }, { 0 });
    cmd_buf->bindIndexBuffer(
        merged_index_buffer_.buffer, 0, er::IndexType::UINT32);

    cmd_buf->drawIndexedIndirectCount(
        indirect_draw_buffer_, 0,
        draw_count_buffer_, 0,
        total_clusters_all_meshes_);
    return total_visible_all_meshes_;
}

// ─── Cluster CSM shadow pipeline + draw ──────────────────────────────────
// Depth-only single-pass CSM rendering for cluster-owned meshes via a
// mesh shader (cluster_bindless_shadow.mesh).  One mesh-shader workgroup
// per (cluster, cascade) pair; each workgroup frustum-culls against its
// cascade's VP and emits the cluster's surviving triangles directly to
// the matching depth-array layer (gl_Layer = cascade_idx).
//
// Perf history that landed on the mesh-shader path (Bistro, desktop NV):
//   A. Per-cluster vkCmdDrawIndexedIndirectCount + GS broadcast    ~28 ms
//   B. Single drawIndexed over merged VB/IB + GS broadcast        ~10 ms
//   C. Single drawIndexed + multiview                              ~56 ms (VS replication)
//   D. Per-cascade rendering + per-cascade cull                    ~26 ms
//   E. Mesh shader, one workgroup per (cluster, cascade)            ←
//
// Why mesh shader wins:
//   • Bypasses the GS amplification stage entirely — no triangle
//     duplication through fixed-function output, no GS invocation
//     overhead per input primitive.
//   • Per-(cluster, cascade) frustum cull skips ~75% of total
//     workgroups outright (each cascade keeps ~5-90% of clusters; the
//     union across cascades is ~2× a single frustum-culled set, vs 6×
//     for any "broadcast everything to every layer" path).
//   • One workgroup = one cluster of up to 128 triangles with private
//     vertex output; mesh-shader hardware schedules them in parallel.
//
// Why a separate pipeline / layout (not reusing initBindlessGBufferPipeline):
//   • No fragment shader (depth-only).
//   • No bindless material descriptor set needed.  The slim layout
//     carries: VIEW_PARAMS_SET (compatibility), RUNTIME_LIGHTS_PARAMS_
//     SET (mesh reads light_view_proj[cascade]), and
//     CLUSTER_MESH_DATA_SET (mesh reads cull-info / draw-info /
//     merged VB / merged IB SSBOs).
//   • Mesh-shader stage replaces VS+GS; no vertex input bindings.
void ClusterRenderer::initBindlessShadowPipeline(
    const renderer::DescriptorSetLayoutList& shadow_desc_set_layouts,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::Format& shadow_depth_format) {

    // No-op if the cluster GPU state isn't ready (no merged geometry → no
    // indirect buffer to draw against anyway).
    if (!gpu_ready_) {
        return;
    }
    if (bindless_shadow_pipeline_) {
        return;   // already built
    }
    if (!cluster_mesh_data_desc_set_layout_) {
        // The mesh-shader data set wasn't allocated (e.g. zero clusters
        // uploaded) — no point building a pipeline that can't draw.
        return;
    }

    // ── Pipeline layout ──────────────────────────────────────────────
    // Build a slim layout that exposes only the descriptor sets the
    // mesh shader actually touches:
    //   set VIEW_PARAMS_SET             — present for set-list compatibility
    //   set RUNTIME_LIGHTS_PARAMS_SET   — light_view_proj[cascade] UBO
    //   set CLUSTER_MESH_DATA_SET (+1)  — cluster SSBOs (cull/draw/VB/IB)
    //
    // Vulkan requires every set slot in the pipeline layout to have a
    // valid (non-null) descriptor set layout — gaps are not allowed.
    // Replace any nullptr slots with an empty layout so createPipelineLayout
    // doesn't crash on .get().
    const uint32_t cluster_mesh_data_set = RUNTIME_LIGHTS_PARAMS_SET + 1;
    er::DescriptorSetLayoutList all_layouts;
    for (uint32_t i = 0; i <= cluster_mesh_data_set; ++i) {
        std::shared_ptr<er::DescriptorSetLayout> layout;
        if (i == cluster_mesh_data_set) {
            layout = cluster_mesh_data_desc_set_layout_;
        } else if (i < shadow_desc_set_layouts.size()) {
            layout = shadow_desc_set_layouts[i];
        }
        all_layouts.push_back(layout);
    }
    auto empty_layout = device_->createDescriptorSetLayout({});
    for (auto& layout : all_layouts) {
        if (!layout) layout = empty_layout;
    }

    // Push constants: task shader needs total_clusters to clamp its
    // per-lane cluster_idx (last task workgroup is partial when the
    // cluster count isn't a multiple of CLUSTERS_PER_TASK_WG = 32).
    er::PushConstantRange shadow_pc{};
    shadow_pc.stage_flags = SET_FLAG_BIT(ShaderStage, TASK_BIT_EXT);
    shadow_pc.offset      = 0;
    shadow_pc.size        = sizeof(uint32_t);  // one uint: total_clusters
    bindless_shadow_pipeline_layout_ = device_->createPipelineLayout(
        all_layouts, { shadow_pc }, std::source_location::current());

    // ── Vertex input: none (mesh shader fetches vertices from SSBO) ──
    std::vector<er::VertexInputBindingDescription>   binding_descs;
    std::vector<er::VertexInputAttributeDescription> attrib_descs;

    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology       = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    // ── Rasterization state ──────────────────────────────────────────
    // depth_clamp_enable=true: keep fragments outside the cascade depth
    // range from being clipped (matches createDrawableShadowPipelineInternal).
    //
    // No more force-double-sided override.  The cluster shadow pipeline
    // is shared across all cluster materials (single pipeline per CSM
    // draw mode, no per-material variant), so the rasterizer ends up at
    // its default cull mode (back-face cull).  This matches the drawable
    // shadow path's per-material behaviour for single-sided assets and
    // gives us cluster-level back-face dropping for free.
    //
    // Trade-off: any cluster whose material is authored as double_sided
    // (foliage, fences) loses its back-face shadow on this path because
    // the rasterizer here can't tell per-cluster what the material
    // flag says.  Asset authors who need double-sided shadow coverage
    // for thin geometry should keep that geometry on the drawable
    // (non-cluster) path, which has per-material pipelines.  In the
    // MeshShader path the task shader additionally cone-culls clusters
    // whose normal cone is fully aligned with the sun (perf), gated on
    // the material flag — see cluster_bindless_shadow.task.
    er::RasterizationStateOverride raster_override{};
    raster_override.override_depth_clamp_enable = true;
    raster_override.depth_clamp_enable          = true;

    // ── Depth-stencil state ─────────────────────────────────────────
    // Standard shadow depth-pass state: test + write enabled, LESS_OR_EQUAL.
    auto shadow_depth_stencil =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                /*depth_test_enable */ true,
                /*depth_write_enable*/ true,
                er::CompareOp::LESS_OR_EQUAL));

    // No colour attachments → an empty blend state with zero attachments.
    std::vector<er::PipelineColorBlendAttachmentState> shadow_no_blend;
    auto shadow_blend_state =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(shadow_no_blend));

    er::GraphicPipelineInfo shadow_info = graphic_pipeline_info;
    shadow_info.blend_state_info   = shadow_blend_state;
    shadow_info.depth_stencil_info = shadow_depth_stencil;

    // ── Shader modules: TASK + MESH, no fragment ─────────────────────
    // Task shader culls clusters against all CSM_CASCADE_COUNT cascade
    // frustums in one pass and emits mesh workgroups ONLY for surviving
    // (cluster, cascade) pairs.  Mesh shader then emits triangles
    // without any further cull.  See cluster_bindless_shadow.task and
    // .mesh for the per-stage rationale.
    er::ShaderModuleList shader_modules(2);
    shader_modules[0] = er::helper::loadShaderModule(
        device_, "cluster_bindless_shadow_task.spv",
        er::ShaderStageFlagBits::TASK_BIT_EXT,
        std::source_location::current());
    shader_modules[1] = er::helper::loadShaderModule(
        device_, "cluster_bindless_shadow_mesh.spv",
        er::ShaderStageFlagBits::MESH_BIT_EXT,
        std::source_location::current());

    // ── Framebuffer format ──────────────────────────────────────────
    // No colour attachments; single depth-array attachment.
    // view_mask stays at 0 (no multiview); the mesh shader writes
    // gl_Layer per-primitive and the matching RenderingInfo uses
    // layer_count = CSM_CASCADE_COUNT.
    er::PipelineRenderbufferFormats shadow_fmt;
    shadow_fmt.color_formats = {};
    shadow_fmt.depth_format  = shadow_depth_format;

    bindless_shadow_pipeline_ = device_->createPipeline(
        bindless_shadow_pipeline_layout_,
        binding_descs,
        attrib_descs,
        input_assembly,
        shadow_info,
        shader_modules,
        shadow_fmt,
        raster_override,
        std::source_location::current());

    // ── Auxiliary cluster shadow pipelines (GS + per-cascade) ────────
    // Built alongside the task+mesh pipeline so the menu's "Drawable
    // shadow draw mode" 3-way toggle can drive the cluster path too.
    // Both share the renderbuffer format / depth-stencil / blend /
    // raster-override scaffolding above.

    // Slim layout for both — RUNTIME_LIGHTS_PARAMS_SET only (the GS
    // and per-cascade VS read light_view_proj from the UBO).  We do
    // NOT include CLUSTER_MESH_DATA_SET here: these pipelines bind the
    // merged VB/IB through the input assembler, not as SSBOs.
    er::DescriptorSetLayoutList aux_layouts;
    auto aux_empty_layout = device_->createDescriptorSetLayout({});
    for (uint32_t i = 0; i <= RUNTIME_LIGHTS_PARAMS_SET; ++i) {
        std::shared_ptr<er::DescriptorSetLayout> layout =
            (i < shadow_desc_set_layouts.size())
                ? shadow_desc_set_layouts[i]
                : nullptr;
        aux_layouts.push_back(layout ? layout : aux_empty_layout);
    }

    // Vertex input: position only (matches BindlessVertex.position
    // location 0).  Same layout the mesh path uses internally; the
    // input assembler simply pulls position from the merged VB.
    std::vector<er::VertexInputBindingDescription> aux_binding_descs(1);
    aux_binding_descs[0].binding    = 0;
    aux_binding_descs[0].stride     = sizeof(BindlessVertex);
    aux_binding_descs[0].input_rate = er::VertexInputRate::VERTEX;

    std::vector<er::VertexInputAttributeDescription> aux_attrib_descs(1);
    aux_attrib_descs[0].binding  = 0;
    aux_attrib_descs[0].location = 0;
    aux_attrib_descs[0].format   = er::Format::R32G32B32_SFLOAT;
    aux_attrib_descs[0].offset   = offsetof(BindlessVertex, position);

    // ─── GS pipeline (kGeometryShader) ─────────────────────────────
    // VS passes world position; GS broadcasts each tri to all 6 layers
    // applying lights_params.light_view_proj[cascade] per emit.  Layered
    // FB (CSM_CASCADE_COUNT) — same as the mesh-shader pipeline.
    {
        bindless_shadow_gs_pipeline_layout_ = device_->createPipelineLayout(
            aux_layouts, {}, std::source_location::current());

        er::ShaderModuleList gs_modules(2);
        gs_modules[0] = er::helper::loadShaderModule(
            device_, "cluster_bindless_shadow_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
        gs_modules[1] = er::helper::loadShaderModule(
            device_, "cluster_bindless_shadow_geom.spv",
            er::ShaderStageFlagBits::GEOMETRY_BIT,
            std::source_location::current());

        bindless_shadow_gs_pipeline_ = device_->createPipeline(
            bindless_shadow_gs_pipeline_layout_,
            aux_binding_descs,
            aux_attrib_descs,
            input_assembly,
            shadow_info,
            gs_modules,
            shadow_fmt,
            raster_override,
            std::source_location::current());
    }

    // ─── Per-cascade VS pipeline (kRegular) ───────────────────────
    // Push constant: uint cascade_idx (4 bytes, VS visibility).  Host
    // loops cascades, each iteration pushes its index and renders into
    // a single-layer view of csm_shadow_tex_.
    {
        er::PushConstantRange per_cascade_pc{};
        per_cascade_pc.stage_flags = SET_FLAG_BIT(ShaderStage, VERTEX_BIT);
        per_cascade_pc.offset      = 0;
        per_cascade_pc.size        = sizeof(uint32_t);

        bindless_shadow_per_cascade_pipeline_layout_ =
            device_->createPipelineLayout(
                aux_layouts, { per_cascade_pc },
                std::source_location::current());

        er::ShaderModuleList vs_modules(1);
        vs_modules[0] = er::helper::loadShaderModule(
            device_, "cluster_bindless_shadow_per_cascade_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());

        // Per-cascade FB has only ONE layer (csm_layer_views_[k]).  No
        // change to shadow_fmt is needed for that — the format struct
        // doesn't carry layer count; that's a runtime RenderingInfo field.
        bindless_shadow_per_cascade_pipeline_ = device_->createPipeline(
            bindless_shadow_per_cascade_pipeline_layout_,
            aux_binding_descs,
            aux_attrib_descs,
            input_assembly,
            shadow_info,
            vs_modules,
            shadow_fmt,
            raster_override,
            std::source_location::current());
    }
}

// ─── CSM silhouette prepass pipeline + draw ──────────────────────────────
// See csm_silhouette_prepass.mesh for the algorithm rationale.  This is
// a small mesh-shader pipeline that fills each cascade's in-camera-
// frustum interior with depth=1.0 so that out-of-frustum texels (still
// at the cleared 0.0) reject every later shadow caster via LESS_OR_EQUAL.
//
// The depth-test state for this pipeline is ALWAYS (so the forced z=1
// unconditionally replaces the cleared 0.0).  Subsequent shadow draws
// must use load_op = LOAD so the prepass result is preserved.
void ClusterRenderer::initCsmSilhouettePrepassPipeline(
    const std::shared_ptr<renderer::DescriptorSetLayout>&
        runtime_lights_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::Format& shadow_depth_format) {

    if (silhouette_prepass_pipeline_) {
        return;   // already built
    }
    if (!runtime_lights_desc_set_layout) {
        return;
    }

    // ── Pipeline layout ──────────────────────────────────────────────
    // The mesh shader only reads from RUNTIME_LIGHTS_PARAMS_SET (cascade
    // VPs + slab corners).  Pad earlier set slots with an empty layout
    // so createPipelineLayout doesn't choke on null entries.
    er::DescriptorSetLayoutList all_layouts;
    auto empty_layout = device_->createDescriptorSetLayout({});
    for (uint32_t i = 0; i <= RUNTIME_LIGHTS_PARAMS_SET; ++i) {
        all_layouts.push_back(
            i == RUNTIME_LIGHTS_PARAMS_SET
                ? runtime_lights_desc_set_layout
                : empty_layout);
    }

    silhouette_prepass_pipeline_layout_ = device_->createPipelineLayout(
        all_layouts, {}, std::source_location::current());

    // No vertex input — mesh shader fetches everything from the UBO.
    std::vector<er::VertexInputBindingDescription>   binding_descs;
    std::vector<er::VertexInputAttributeDescription> attrib_descs;

    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology       = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    // Cull none: we want both hex faces to cover the silhouette.  Depth
    // clamp on so any vertex projected outside [0,1] is clamped rather
    // than clipped (the mesh shader forces z=w → 1.0 NDC, so this is
    // belt-and-suspenders against future shader changes).
    er::RasterizationStateOverride raster_override{};
    raster_override.override_depth_clamp_enable = true;
    raster_override.depth_clamp_enable          = true;
    raster_override.override_double_sided       = true;
    raster_override.double_sided                = true;

    // Depth test ALWAYS: the prepass unconditionally writes 1.0 over
    // the cleared 0.0 inside the silhouette.  Depth write ON.
    auto silhouette_depth_stencil =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                /*depth_test_enable */ true,
                /*depth_write_enable*/ true,
                er::CompareOp::ALWAYS));

    std::vector<er::PipelineColorBlendAttachmentState> no_blend;
    auto blend_state =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(no_blend));

    er::GraphicPipelineInfo info = graphic_pipeline_info;
    info.blend_state_info   = blend_state;
    info.depth_stencil_info = silhouette_depth_stencil;

    // Single mesh shader stage.
    er::ShaderModuleList shader_modules(1);
    shader_modules[0] = er::helper::loadShaderModule(
        device_, "csm_silhouette_prepass_mesh.spv",
        er::ShaderStageFlagBits::MESH_BIT_EXT,
        std::source_location::current());

    er::PipelineRenderbufferFormats fmt;
    fmt.color_formats = {};
    fmt.depth_format  = shadow_depth_format;

    silhouette_prepass_pipeline_ = device_->createPipeline(
        silhouette_prepass_pipeline_layout_,
        binding_descs,
        attrib_descs,
        input_assembly,
        info,
        shader_modules,
        fmt,
        raster_override,
        std::source_location::current());
}

void ClusterRenderer::drawCsmSilhouettePrepass(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& runtime_lights_desc_set,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {
    if (!silhouette_prepass_pipeline_ || !runtime_lights_desc_set) {
        return;
    }

    er::DescriptorSetList all_desc_sets(RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
    all_desc_sets[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, silhouette_prepass_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        silhouette_prepass_pipeline_layout_,
        all_desc_sets);

    // CSM_CASCADE_COUNT workgroups (one per cascade), each emits the
    // 12 hex faces of its slab.
    cmd_buf->drawMeshTasks(static_cast<uint32_t>(CSM_CASCADE_COUNT), 1u, 1u);
}

// ─── Cluster CSM shadow draw (mesh-shader path) ──────────────────────────
// Dispatches one (cluster, cascade) mesh-shader workgroup per pair.
// Each workgroup frustum-culls against its cascade's VP and emits the
// cluster's surviving triangles directly to the matching depth-array
// layer (gl_Layer = cascade_idx).  Replaces the prior VS+GS broadcast
// path — see the comment at initBindlessShadowPipeline for the perf
// history.
//
// Caller MUST have an active dynamic-rendering pass with the full
// CSM_CASCADE_COUNT-layer depth attachment bound and
// layer_count = CSM_CASCADE_COUNT on the RenderingInfo so the mesh
// shader's gl_Layer writes land on the right cascade.  No fragment
// shader runs.
uint32_t ClusterRenderer::drawClusterShadow(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {
    if (!gpu_ready_ || !bindless_shadow_pipeline_) {
        return 0;
    }
    if (!cluster_mesh_data_desc_set_) {
        return 0;
    }

    // Pad the desc-set list to include the cluster mesh-data set at
    // index RUNTIME_LIGHTS_PARAMS_SET + 1.
    const uint32_t cluster_mesh_data_set = RUNTIME_LIGHTS_PARAMS_SET + 1;
    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= cluster_mesh_data_set) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets.resize(cluster_mesh_data_set + 1);
    all_desc_sets[cluster_mesh_data_set] = cluster_mesh_data_desc_set_;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, bindless_shadow_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        bindless_shadow_pipeline_layout_,
        all_desc_sets);

    // Push total_clusters so the task shader can clamp lane_idx on the
    // last (partial) task workgroup.
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, TASK_BIT_EXT),
        bindless_shadow_pipeline_layout_,
        &total_clusters_all_meshes_,
        sizeof(uint32_t),
        0);

    // Task workgroup count = ceil(total_clusters / CLUSTERS_PER_TASK_WG).
    // CLUSTERS_PER_TASK_WG = 32, matching the task shader's local_size_x.
    // Each task workgroup tests its 32 clusters against all 6 cascades,
    // emits mesh workgroups only for the surviving pairs.  Cull-failed
    // (cluster, cascade) pairs cost only the task-shader bit test, no
    // mesh-workgroup launch.
    constexpr uint32_t kClustersPerTaskWg = 32u;
    const uint32_t task_wg_count =
        (total_clusters_all_meshes_ + kClustersPerTaskWg - 1u) /
        kClustersPerTaskWg;
    cmd_buf->drawMeshTasks(task_wg_count, 1u, 1u);
    return total_clusters_all_meshes_;
}

// ─── Cluster CSM shadow draw — VS+GS broadcast (kGeometryShader) ────────
// Single drawIndexed over the entire merged VB/IB; the GS broadcasts
// each triangle to all CSM_CASCADE_COUNT depth-array layers.  No per-
// cluster cull on this path — every cluster's tris go to every cascade
// (the per-cascade rasterizer eventually clips fragments outside the
// cascade XY extent + the silhouette-prepass depth-rejection takes care
// of out-of-frustum waste).
//
// Caller MUST have an active dynamic-rendering pass with the full
// CSM_CASCADE_COUNT-layer depth attachment bound and
// layer_count = CSM_CASCADE_COUNT on the RenderingInfo so the GS's
// gl_Layer writes land on the right cascade.
uint32_t ClusterRenderer::drawClusterShadowGs(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) {
    if (!gpu_ready_ || !bindless_shadow_gs_pipeline_) {
        return 0;
    }

    // Pad / truncate to the GS pipeline layout's slot count
    // (RUNTIME_LIGHTS_PARAMS_SET + 1 entries, no cluster mesh-data set).
    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= RUNTIME_LIGHTS_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets.resize(RUNTIME_LIGHTS_PARAMS_SET + 1);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, bindless_shadow_gs_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        bindless_shadow_gs_pipeline_layout_,
        all_desc_sets);
    cmd_buf->bindVertexBuffers(
        0, { merged_vertex_buffer_.buffer }, { 0 });
    cmd_buf->bindIndexBuffer(
        merged_index_buffer_.buffer, 0, er::IndexType::UINT32);

    cmd_buf->drawIndexed(
        total_merged_indices_,
        /*instance_count*/ 1,
        /*first_index*/    0,
        /*vertex_offset*/  0,
        /*first_instance*/ 0);
    return total_clusters_all_meshes_;
}

// ─── Cluster CSM shadow draw — VS-only per-cascade (kRegular) ──────────
// Single-cascade single-layer draw.  Caller MUST have an active dynamic-
// rendering pass with a SINGLE-LAYER view (csm_layer_views_[cascade_idx])
// bound and layer_count = 1.  Caller is expected to loop k=0..N-1, opening
// a fresh single-layer scope each iteration and calling this with k.
//
// VS reads lights_params.light_view_proj[cascade_idx] — picked via the
// uint push constant.  No cull, no GS, no mesh shader.  Same triangle
// work as the GS path, but split across CSM_CASCADE_COUNT separate
// drawIndexed calls + render passes.
uint32_t ClusterRenderer::drawClusterShadowPerCascade(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    uint32_t cascade_idx) {
    if (!gpu_ready_ || !bindless_shadow_per_cascade_pipeline_) {
        return 0;
    }

    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= RUNTIME_LIGHTS_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets.resize(RUNTIME_LIGHTS_PARAMS_SET + 1);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS,
        bindless_shadow_per_cascade_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        bindless_shadow_per_cascade_pipeline_layout_,
        all_desc_sets);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT),
        bindless_shadow_per_cascade_pipeline_layout_,
        &cascade_idx,
        sizeof(uint32_t),
        0);

    cmd_buf->bindVertexBuffers(
        0, { merged_vertex_buffer_.buffer }, { 0 });
    cmd_buf->bindIndexBuffer(
        merged_index_buffer_.buffer, 0, er::IndexType::UINT32);

    cmd_buf->drawIndexed(
        total_merged_indices_,
        /*instance_count*/ 1,
        /*first_index*/    0,
        /*vertex_offset*/  0,
        /*first_instance*/ 0);
    return total_clusters_all_meshes_;
}

uint32_t ClusterRenderer::drawTranslucentForward(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    const std::shared_ptr<renderer::ImageView>& color_view,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const glm::uvec2& screen_size) {
    // Translucent (glass) draw path, NOW using standard alpha blending
    // straight into the caller's color/depth pass — no WBOIT accum/reveal
    // targets, no composite resolve.  The previous WBOIT implementation
    // was producing unpredictable glass: "fully translucent" frames where
    // the composite's discard rule (reveal > 0.99 OR accum.a < 1e-3)
    // fired even when glass should have been visible, plus the
    // chromatic refraction of the glass IBL math poisoning whole-pixel
    // accum values via the IBL mini-cube mip-0 dither feeding NaN /
    // wild colour deltas into the accumulator.
    //
    // The new design:
    //   1. The cull pass already emits a TRANSLUCENT-only indirect draw
    //      list (trans_indirect_draw_buffer_) keyed on the
    //      BINDLESS_MAT_TRANSLUCENT alpha-mode tag — unchanged.
    //   2. We bind the ALPHA_BLEND_OUTPUT shader variant + the alpha-
    //      blend pipeline (created in initBindlessPipeline above), which
    //      writes a single (rgb, α) to location 0.
    //   3. The pipeline does porter-duff "over" via hardware src_alpha /
    //      one_minus_src_alpha blending against the scene colour buffer.
    //   4. Depth test ON, depth write OFF — see the pipeline depth-stencil
    //      comment for why writing depth here would break multi-layer
    //      glass.
    //
    // Alpha-blend only.  WBOIT was briefly re-added as a runtime-
    // switchable second mode but caused glass to disappear (regression
    // we couldn't pin down quickly), so it was reverted to keep the
    // engine in a known-good state.  The OIT pipeline + composite
    // resources still exist in this file but are no longer reachable
    // from this draw path.  If WBOIT is needed again later, prefer
    // adding it back as a separate explicit entry point (e.g.
    // drawTranslucentOit) rather than dispatching inside this one —
    // the in-place dispatch was the version that broke.
    (void)color_view;
    (void)depth_view;
    (void)screen_size;

    if (!gpu_ready_ || !bindless_pipeline_ || !bindless_desc_set_) {
        return 0;
    }
    if (!bindless_translucent_pipeline_) {
        // Shader compile failed at init — nothing we can do; opaque pass
        // already drew, the frame is incomplete but won't crash.
        return total_visible_all_meshes_;
    }

    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= PBR_MATERIAL_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets[PBR_MATERIAL_PARAMS_SET] = bindless_desc_set_;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS, bindless_translucent_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        bindless_pipeline_layout_,
        all_desc_sets);
    cmd_buf->bindVertexBuffers(
        0, { merged_vertex_buffer_.buffer }, { 0 });
    cmd_buf->bindIndexBuffer(
        merged_index_buffer_.buffer, 0, er::IndexType::UINT32);

    cmd_buf->drawIndexedIndirectCount(
        trans_indirect_draw_buffer_, 0,
        trans_draw_count_buffer_, 0,
        total_clusters_all_meshes_);

    return total_visible_all_meshes_;
}

// ─── WBOIT translucent draw ────────────────────────────────────────────────
// Standalone entry point — caller MUST NOT have a render pass open when
// it invokes this function.  We open every pass we need internally and
// leave NO pass open on return.  The application's caller handles its
// own outer pass orchestration; for the glass dispatch in drawScene the
// rule is simply "alpha-blend path opens a pass around the call, OIT
// path does NOT" — see the glass block in application.cpp.
//
// This is deliberately a separate entry point from drawTranslucentForward
// (and not a dispatch inside it) because the earlier in-function dispatch
// design managed to regress alpha-blend glass in ways we couldn't pin
// down quickly.  Split-entry guarantees the two pipelines stay on
// independent code paths.
uint32_t ClusterRenderer::drawTranslucentOit(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_sets,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors,
    const std::shared_ptr<renderer::ImageView>& color_view,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const glm::uvec2& screen_size) {
    if (!gpu_ready_ || !bindless_pipeline_ || !bindless_desc_set_) {
        return 0;
    }
    if (!bindless_translucent_oit_pipeline_ || !oit_composite_pipeline_) {
        // OIT pipeline didn't init — glass simply doesn't draw in this
        // path; opaque scene remains untouched.  Caller's outer code
        // is responsible for falling back to alpha-blend if it wants
        // to (typically by reading getTranslucentMode() and dispatching
        // before calling either function).
        return total_visible_all_meshes_;
    }

    ensureOitTargets(screen_size);
    if (!oit_composite_desc_set_) {
        return total_visible_all_meshes_;
    }

    er::DescriptorSetList all_desc_sets = desc_sets;
    while (all_desc_sets.size() <= PBR_MATERIAL_PARAMS_SET) {
        all_desc_sets.push_back(nullptr);
    }
    all_desc_sets[PBR_MATERIAL_PARAMS_SET] = bindless_desc_set_;

    auto bind_geometry_state = [&](const std::shared_ptr<er::Pipeline>& pipe) {
        cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, pipe);
        cmd_buf->setViewports(viewports);
        cmd_buf->setScissors(scissors);
        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::GRAPHICS,
            bindless_pipeline_layout_,
            all_desc_sets);
        cmd_buf->bindVertexBuffers(0, { merged_vertex_buffer_.buffer }, { 0 });
        cmd_buf->bindIndexBuffer(
            merged_index_buffer_.buffer, 0, er::IndexType::UINT32);
    };

    // ── Pass A: WBOIT accum+reveal draw ──────────────────────────────
    er::RenderingAttachmentInfo accum_att;
    accum_att.image_view   = oit_accum_tex_.view;
    accum_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    accum_att.load_op      = er::AttachmentLoadOp::CLEAR;
    accum_att.store_op     = er::AttachmentStoreOp::STORE;
    accum_att.clear_value.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

    er::RenderingAttachmentInfo reveal_att;
    reveal_att.image_view   = oit_reveal_tex_.view;
    reveal_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    reveal_att.load_op      = er::AttachmentLoadOp::CLEAR;
    reveal_att.store_op     = er::AttachmentStoreOp::STORE;
    reveal_att.clear_value.color = { { 1.0f, 0.0f, 0.0f, 0.0f } };

    er::RenderingAttachmentInfo oit_depth_att;
    oit_depth_att.image_view   = depth_view;
    oit_depth_att.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    oit_depth_att.load_op      = er::AttachmentLoadOp::LOAD;
    oit_depth_att.store_op     = er::AttachmentStoreOp::STORE;

    er::RenderingInfo oit_ri = {};
    oit_ri.render_area_offset = { 0, 0 };
    oit_ri.render_area_extent = screen_size;
    oit_ri.layer_count        = 1;
    oit_ri.view_mask          = 0;
    oit_ri.color_attachments  = { accum_att, reveal_att };
    oit_ri.depth_attachments  = { oit_depth_att };
    oit_ri.stencil_attachments = {};

    cmd_buf->beginDynamicRendering(oit_ri);
    bind_geometry_state(bindless_translucent_oit_pipeline_);
    cmd_buf->drawIndexedIndirectCount(
        trans_indirect_draw_buffer_, 0,
        trans_draw_count_buffer_, 0,
        total_clusters_all_meshes_);
    cmd_buf->endDynamicRendering();

    // ── Transition accum + reveal: COLOR_ATTACHMENT → SHADER_READ ────
    er::ImageResourceInfo from_color_att = {
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
    er::ImageResourceInfo to_shader_read = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
    cmd_buf->addImageBarrier(oit_accum_tex_.image,  from_color_att, to_shader_read, 0, 1, 0, 1);
    cmd_buf->addImageBarrier(oit_reveal_tex_.image, from_color_att, to_shader_read, 0, 1, 0, 1);

    // ── Pass B: fullscreen composite onto host colour buffer ────────
    er::RenderingAttachmentInfo comp_color_att;
    comp_color_att.image_view   = color_view;
    comp_color_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    comp_color_att.load_op      = er::AttachmentLoadOp::LOAD;
    comp_color_att.store_op     = er::AttachmentStoreOp::STORE;

    er::RenderingAttachmentInfo comp_depth_att;
    comp_depth_att.image_view   = depth_view;
    comp_depth_att.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    comp_depth_att.load_op      = er::AttachmentLoadOp::LOAD;
    comp_depth_att.store_op     = er::AttachmentStoreOp::STORE;

    er::RenderingInfo comp_ri = {};
    comp_ri.render_area_offset  = { 0, 0 };
    comp_ri.render_area_extent  = screen_size;
    comp_ri.layer_count         = 1;
    comp_ri.view_mask           = 0;
    comp_ri.color_attachments   = { comp_color_att };
    comp_ri.depth_attachments   = { comp_depth_att };
    comp_ri.stencil_attachments = {};

    cmd_buf->beginDynamicRendering(comp_ri);
    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, oit_composite_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        oit_composite_pipeline_layout_,
        { oit_composite_desc_set_ });
    cmd_buf->draw(3);   // fullscreen triangle from full_screen.vert
    cmd_buf->endDynamicRendering();

    return total_visible_all_meshes_;
}

// ─── Pool-owned descriptor cleanup (swap chain teardown) ──────────────────
//
// When the application destroys the descriptor pool every descriptor set
// allocated from it becomes a dangling Vulkan handle.  C++ shared_ptr
// reference counts don't keep the Vulkan object alive — vkDestroy-
// DescriptorPool is unconditional.  Any subsequent vkUpdateDescriptor-
// Sets that touches one of those dead handles is undefined behaviour
// (NVIDIA's driver crashes inside the validation layer's dispatch
// shim — the backtrace surfaces as nvoglv64.dll → UpdateDescriptorSets).
//
// The recreateSwapChain flow in application.cpp inadvertently writes
// to several of these sets BEFORE recreate() repopulates them: the
// recreateRenderBuffer → createGBuffer → createHiZPyramid chain calls
// setHiZTexture (which patches cull_desc_set_'s binding 11) WAY before
// the new descriptor_pool_ + cluster_renderer_->recreate(...) pair
// runs later in recreateSwapChain.  We null the handles here so the
// "if (cull_desc_set_ && ...)" guard at the top of setHiZTexture
// short-circuits during that window, then the freshly-allocated set
// from recreate() / the next finalizeUploads picks up the cached
// hiz_sampler_ / hiz_view_ stored on this object.
void ClusterRenderer::onDescriptorPoolDestroyed() {
    bindless_desc_set_.reset();
    cull_desc_set_.reset();
    for (auto& s : cull_desc_sets_shadow_) s.reset();
    cluster_mesh_data_desc_set_.reset();
    oit_composite_desc_set_.reset();
}

// ─── Recreate (swap chain resize) ─────────────────────────────────────────

void ClusterRenderer::recreate(
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool) {

    if (!gpu_ready_ || !bindless_desc_set_layout_) return;

    descriptor_pool_ = descriptor_pool;

    // Re-allocate bindless descriptor set.
    // persistent pool: allocate once, reuse across resize
    if (!bindless_desc_set_) {
        auto desc_sets = device_->createDescriptorSets(
            descriptor_pool_, bindless_desc_set_layout_, 1);
        bindless_desc_set_ = desc_sets[0];
    }

    // Re-write SSBO descriptors.
    er::WriteDescriptorList writes;
    writes.reserve(2);
    er::Helper::addOneBuffer(writes, bindless_desc_set_,
        er::DescriptorType::STORAGE_BUFFER, 0,
        draw_info_buffer_.buffer,
        static_cast<uint32_t>(
            total_clusters_all_meshes_ * sizeof(glsl::ClusterDrawInfo)));
    er::Helper::addOneBuffer(writes, bindless_desc_set_,
        er::DescriptorType::STORAGE_BUFFER, 1,
        material_params_buffer_.buffer,
        static_cast<uint32_t>(
            total_materials_ * sizeof(glsl::BindlessMaterialParams)));

    // Texture descriptors are baked into dummy_texture_ — re-write using it.
    for (uint32_t ti = 0; ti < MAX_CLUSTER_TEXTURES; ++ti) {
        auto tex_write = std::make_shared<er::TextureDescriptor>();
        tex_write->binding           = 2;
        tex_write->dst_array_element = ti;
        tex_write->desc_type         = er::DescriptorType::COMBINED_IMAGE_SAMPLER;
        tex_write->desc_set          = bindless_desc_set_;
        tex_write->image_layout      = er::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        tex_write->sampler           = default_sampler_;
        tex_write->texture           = dummy_texture_.view;
        writes.push_back(tex_write);
    }
    for (uint32_t ti = 0; ti < MAX_CLUSTER_TEXTURES; ++ti) {
        auto nw = std::make_shared<er::TextureDescriptor>();
        nw->binding           = 3;
        nw->dst_array_element = ti;
        nw->desc_type         = er::DescriptorType::COMBINED_IMAGE_SAMPLER;
        nw->desc_set          = bindless_desc_set_;
        nw->image_layout      = er::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        nw->sampler           = default_sampler_;
        nw->texture           = dummy_texture_.view;
        writes.push_back(nw);
    }

    // ── VT bindings 4..9 ─────────────────────────────────────────────
    // Same logic as initBindlessPipeline above — the bindless layout
    // includes these slots so the descriptor set we just allocated must
    // write them or shaders will hit a validation error reading dangling
    // bindings.
    if (vt_manager_) {
        const auto vt_sampler = vt_manager_->getPoolSampler();
        for (uint32_t l = 0; l < 4; ++l) {
            auto layer = static_cast<VtLayer>(l);
            er::Helper::addOneTexture(writes, bindless_desc_set_,
                er::DescriptorType::COMBINED_IMAGE_SAMPLER,
                4u + l, vt_sampler,
                vt_manager_->getPoolImageView(layer),
                er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        er::Helper::addOneBuffer(writes, bindless_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 8,
            vt_manager_->getPageTableBuffer(),
            vt_manager_->getPageTableBufferBytes());
        er::Helper::addOneBuffer(writes, bindless_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 9,
            vt_manager_->getMetaBuffer(),
            vt_manager_->getMetaBufferBytes());
        er::Helper::addOneBuffer(writes, bindless_desc_set_,
            er::DescriptorType::STORAGE_BUFFER, 10,
            vt_manager_->getFeedbackBuffer(),
            vt_manager_->getFeedbackBufferBytes());
    }

    device_->updateDescriptorSets(writes);

    // ── Reallocate the cull descriptor set ──────────────────────────
    // cull_desc_set_ was nulled by onDescriptorPoolDestroyed because
    // its parent pool was destroyed; without re-allocating it here the
    // cluster cull dispatch would skip silently every frame after a
    // resize (cull() / cullPhaseA / cullPhaseB all bind cull_desc_set_).
    // Only proceed if finalizeUploads has run — its layout is built
    // there alongside the per-cluster buffers.  All the source state
    // (cluster counts, indirect buffers, visibility bits, Hi-Z view
    // and sampler) is cached on this object and survives the swap
    // chain teardown.
    if (cull_desc_set_layout_ && total_clusters_all_meshes_ > 0) {
        // persistent pool: allocate once, reuse across resize
        if (!cull_desc_set_) {
            cull_desc_set_ = device_->createDescriptorSets(
                descriptor_pool_, cull_desc_set_layout_, 1)[0];
        }
        const uint32_t mat_count = std::max<uint32_t>(1u, total_materials_);
        writeCullDescriptors(
            device_, cull_desc_set_,
            total_clusters_all_meshes_,
            cull_info_buffer_, draw_info_buffer_,
            indirect_draw_buffer_, draw_count_buffer_,
            visible_buffer_,
            material_params_buffer_, mat_count,
            trans_indirect_draw_buffer_, trans_draw_count_buffer_,
            visibility_bit_buffer_,
            indirect_draw_buffer_phase_a_,
            draw_count_buffer_phase_a_,
            hiz_sampler_, hiz_view_);
    }
}

// ─── Destroy ──────────────────────────────────────────────────────────────

void ClusterRenderer::destroy() {
    bindless_pipeline_.reset();
    bindless_translucent_pipeline_.reset();
    bindless_translucent_oit_pipeline_.reset();
    bindless_gbuffer_pipeline_.reset();
    bindless_shadow_pipeline_.reset();
    bindless_shadow_pipeline_layout_.reset();
    bindless_pipeline_layout_.reset();
    bindless_desc_set_.reset();
    bindless_desc_set_layout_.reset();
    cull_pipeline_.reset();
    cull_pipeline_layout_.reset();
    cull_desc_set_.reset();
    for (auto& s : cull_desc_sets_shadow_) s.reset();
    cull_desc_set_layout_.reset();

    // Mesh-shader CSM cluster data set (allocated in finalizeUploads).
    cluster_mesh_data_desc_set_.reset();
    cluster_mesh_data_desc_set_layout_.reset();

    // CSM silhouette prepass pipeline.
    silhouette_prepass_pipeline_.reset();
    silhouette_prepass_pipeline_layout_.reset();

    // Cluster shadow GS + per-cascade pipelines (kGeometryShader / kRegular).
    bindless_shadow_gs_pipeline_.reset();
    bindless_shadow_gs_pipeline_layout_.reset();
    bindless_shadow_per_cascade_pipeline_.reset();
    bindless_shadow_per_cascade_pipeline_layout_.reset();

    // ── OIT resources ──
    oit_composite_pipeline_.reset();
    oit_composite_pipeline_layout_.reset();
    oit_composite_desc_set_.reset();
    oit_composite_desc_set_layout_.reset();
    oit_composite_sampler_.reset();
    if (oit_accum_tex_.image)  oit_accum_tex_.destroy(device_);
    if (oit_reveal_tex_.image) oit_reveal_tex_.destroy(device_);
    oit_target_size_ = glm::uvec2(0, 0);

    // GPU buffers (including trans_indirect_draw_buffer_ /
    // trans_draw_count_buffer_) are released when their shared_ptrs go
    // out of scope alongside the rest of the cluster state.
    gpu_ready_ = false;
}

// ─── Verify indirect commands (debug helper) ──────────────────────────────

bool ClusterRenderer::verifyIndirectCommands(uint32_t max_commands_to_check) const {
    if (!draw_count_buffer_.memory || !indirect_draw_buffer_.memory) {
        std::fprintf(stderr, "[VERIFY] Failed to map draw_count_buffer_.\n");
        return false;
    }

    // Read draw count.
    uint32_t draw_count = 0;
    {
        const void* ptr = device_->mapMemory(
            draw_count_buffer_.memory, sizeof(uint32_t), 0);
        if (!ptr) {
            std::fprintf(stderr, "[VERIFY] Failed to map draw_count_buffer_.\n");
            return false;
        }
        std::memcpy(&draw_count, ptr, sizeof(uint32_t));
        device_->unmapMemory(draw_count_buffer_.memory);
    }
    std::fprintf(stderr,
        "[VERIFY] draw_count=%u  total_clusters=%u  "
        "total_merged_verts=%u  total_merged_indices=%u\n",
        draw_count, total_clusters_all_meshes_,
        total_merged_vertices_, total_merged_indices_);

    if (draw_count == 0) {
        std::fprintf(stderr, "[VERIFY] draw_count is 0 — no clusters visible.\n");
        return true;
    }
    if (draw_count > total_clusters_all_meshes_) {
        std::fprintf(stderr,
            "[VERIFY] CORRUPTED: draw_count=%u > total_clusters=%u\n",
            draw_count, total_clusters_all_meshes_);
        return false;
    }

    // Read indirect draw buffer.
    const uint32_t stride = sizeof(er::DrawIndexedIndirectCommand);
    const uint32_t n_check = std::min(draw_count, max_commands_to_check);
    const uint64_t map_size = n_check * stride;

    const void* indirect_ptr = device_->mapMemory(
        indirect_draw_buffer_.memory, map_size, 0);
    if (!indirect_ptr) {
        std::fprintf(stderr, "[VERIFY] Failed to map indirect or index buffer.\n");
        return false;
    }

    bool all_ok = true;
    const auto* cmds =
        reinterpret_cast<const er::DrawIndexedIndirectCommand*>(indirect_ptr);
    for (uint32_t i = 0; i < n_check; ++i) {
        const auto& cmd = cmds[i];
        bool bad = (cmd.first_index + cmd.index_count > total_merged_indices_) ||
                   (cmd.vertex_offset < 0 ||
                    static_cast<uint32_t>(cmd.vertex_offset) >= total_merged_vertices_);
        if (bad) {
            std::fprintf(stderr,
                "[VERIFY] cmd[%u]: indexCount=%u firstIndex=%u "
                "vertexOffset=%d firstInstance=%u — OUT OF RANGE\n",
                i, cmd.index_count, cmd.first_index,
                cmd.vertex_offset, cmd.first_instance);
            all_ok = false;
        }
    }
    device_->unmapMemory(indirect_draw_buffer_.memory);
    return all_ok;
}

} // namespace scene_rendering
} // namespace engine
