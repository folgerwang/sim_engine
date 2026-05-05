//
// cluster_renderer.cpp — GPU-driven cluster culling & rendering.
//
// All cluster data is merged into single flat SSBOs. The cull pass is a
// single compute dispatch regardless of how many source meshes were uploaded.
//
#include "cluster_renderer.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "game_object/drawable_object.h"

#include <cstdio>
#include <source_location>
#include <cstring>
#include <unordered_map>

namespace er  = engine::renderer;

namespace engine {
namespace scene_rendering {

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
    std::vector<er::DescriptorSetLayoutBinding> bindings(8);
    for (int i = 0; i < 8; ++i) {
        bindings[i] = er::helper::getBufferDescriptionSetLayoutBinding(
            i, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_BUFFER);
    }
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
    const renderer::BufferInfo& trans_draw_count_buffer) {

    er::WriteDescriptorList writes;
    writes.reserve(8);

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

    std::printf("[CLUSTER_RENDERER] Initialized GPU cluster culling pipeline.\n");
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
    // Always print the first 3 calls to stderr (which flushes immediately).
    if (uploaded_mesh_count_ < 3 && num_clusters > 0) {
        const auto& c0 = cluster_mesh.clusters[0];
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

    auto getMaterialIdx = [&](uint32_t prim_idx) -> uint32_t {
        auto cache_it = prim_to_mat_idx.find(prim_idx);
        if (cache_it != prim_to_mat_idx.end()) return cache_it->second;

        glm::vec4 base_color(1.0f);  // white default
        glsl::BindlessMaterialParams mp{};
        mp.base_color_tex_idx = -1;
        mp.normal_tex_idx     = -1;

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
                    if (tex_idx >= 0 &&
                        static_cast<size_t>(tex_idx) < drawable_data.textures_.size()) {
                        const auto& tex = drawable_data.textures_[tex_idx];
                        if (tex.view) {
                            auto tex_it = staging_tex_slot_map_.find(tex.view.get());
                            if (tex_it != staging_tex_slot_map_.end()) {
                                mp.base_color_tex_idx = tex_it->second;
                            } else if (staging_tex_views_.size() < MAX_CLUSTER_TEXTURES) {
                                int slot = static_cast<int>(staging_tex_views_.size());
                                staging_tex_slot_map_[tex.view.get()] = slot;
                                staging_tex_views_.push_back(tex.view);
                                mp.base_color_tex_idx = slot;
                            }
                            // else: over MAX_CLUSTER_TEXTURES — idx stays -1
                        }
                    }
                    // Stage the normal-map texture (binding 3).
                    int32_t norm_idx = mat.normal_idx_;
                    if (norm_idx >= 0 &&
                        static_cast<size_t>(norm_idx) < drawable_data.textures_.size()) {
                        const auto& ntex = drawable_data.textures_[norm_idx];
                        if (ntex.view) {
                            auto n_it = staging_normal_slot_map_.find(ntex.view.get());
                            if (n_it != staging_normal_slot_map_.end()) {
                                mp.normal_tex_idx = n_it->second;
                            } else if (staging_normal_tex_views_.size() < MAX_CLUSTER_TEXTURES) {
                                int slot = static_cast<int>(staging_normal_tex_views_.size());
                                staging_normal_slot_map_[ntex.view.get()] = slot;
                                staging_normal_tex_views_.push_back(ntex.view);
                                mp.normal_tex_idx = slot;
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
        staging_draw_infos_.push_back(draw_info);
    }

    // Record per-mesh cluster range for visibility feedback.
    mesh_cluster_ranges_.push_back({
        static_cast<uint32_t>(staging_cull_infos_.size() - num_clusters),
        num_clusters
    });

    ++uploaded_mesh_count_;
}

// ─── Finalize uploads (create merged GPU SSBOs) ───────────────────

void ClusterRenderer::finalizeUploads() {
    total_clusters_all_meshes_ =
        static_cast<uint32_t>(staging_cull_infos_.size());

    if (total_clusters_all_meshes_ == 0) {
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

    // Allocate and write descriptor set.
    cull_desc_set_ = device_->createDescriptorSets(
        descriptor_pool_, cull_desc_set_layout_, 1)[0];

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
        trans_indirect_draw_buffer_, trans_draw_count_buffer_);

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
        er::Helper::createBuffer(
            device_,
            SET_2_FLAG_BITS(BufferUsage, VERTEX_BUFFER_BIT, TRANSFER_DST_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            merged_vertex_buffer_.buffer,
            merged_vertex_buffer_.memory,
            std::source_location::current(),
            total_merged_vertices_ * sizeof(BindlessVertex),
            staging_vertices_.data());
    }

    if (total_merged_indices_ > 0) {
        er::Helper::createBuffer(
            device_,
            SET_2_FLAG_BITS(BufferUsage, INDEX_BUFFER_BIT, TRANSFER_DST_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            merged_index_buffer_.buffer,
            merged_index_buffer_.memory,
            std::source_location::current(),
            total_merged_indices_ * sizeof(uint32_t),
            staging_indices_.data());
    }

    gpu_ready_ = true;

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

    // Record counts before clearing staging (used by initBindlessPipeline).
    total_materials_ = static_cast<uint32_t>(
        std::max(size_t(1), staging_material_params_.size()));
    // Texture staging is kept alive until initBindlessPipeline writes descriptors.
    total_textures_ = static_cast<uint32_t>(staging_tex_views_.size());
    // Pad unused base-colour texture slots with the dummy white texture.
    while (staging_tex_views_.size() < MAX_CLUSTER_TEXTURES) {
        staging_tex_views_.push_back(dummy_texture_.view);
    }
    // Same for normal-map slots.
    total_normal_textures_ = static_cast<uint32_t>(staging_normal_tex_views_.size());
    while (staging_normal_tex_views_.size() < MAX_CLUSTER_TEXTURES) {
        staging_normal_tex_views_.push_back(dummy_texture_.view);
    }

    // Free CPU staging memory.
    staging_cull_infos_.clear();
    staging_cull_infos_.shrink_to_fit();
    staging_draw_infos_.clear();
    staging_draw_infos_.shrink_to_fit();
    staging_material_params_.clear();
    staging_material_params_.shrink_to_fit();
    staging_tex_slot_map_.clear();
    staging_normal_slot_map_.clear();
    staging_vertices_.clear();
    staging_vertices_.shrink_to_fit();
    staging_indices_.clear();
    staging_indices_.shrink_to_fit();
}

// ─── Cull (single dispatch for ALL clusters) ──────────────────────

void ClusterRenderer::cull(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::mat4& view_proj,
    const glm::vec3& camera_pos) {

    if (!enabled_ || !gpu_ready_) return;

    // Store for debug display in ImGui.
    debug_last_vp_ = view_proj;
    debug_last_cam_pos_ = camera_pos;

    // ── Debug readback (every frame, non-blocking) ────────────────────
    // Read back PREVIOUS frame's results for debug UI stats.
    // The GPU has finished the previous frame by the time we begin
    // recording the current frame (fence waited), so this is safe.
    // Reading HOST_VISIBLE memory does NOT stall the GPU pipeline —
    // it only reads already-complete data.
    {
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
    // binding 2: sampler2D[MAX_CLUSTER_TEXTURES]  (base colour texture array)
    // binding 3: sampler2D[MAX_CLUSTER_TEXTURES]  (normal map texture array)
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(4);
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
        // accum attachment: additive (ONE / ONE).
        auto accum_blend = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS),
            /*blend_enable*/ true,
            /*src color*/    er::BlendFactor::ONE,
            /*dst color*/    er::BlendFactor::ONE,
            /*color op*/     er::BlendOp::ADD,
            /*src alpha*/    er::BlendFactor::ONE,
            /*dst alpha*/    er::BlendFactor::ONE,
            /*alpha op*/     er::BlendOp::ADD);
        // reveal attachment: multiplicative (ZERO / 1−SRC_COLOR).
        // Each fragment's α multiplies the accumulated visibility, so a
        // sequence of α₁, α₂, … produces dst = Π(1 − αᵢ).
        auto reveal_blend = er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, R_BIT),  // single-channel
            /*blend_enable*/ true,
            /*src color*/    er::BlendFactor::ZERO,
            /*dst color*/    er::BlendFactor::ONE_MINUS_SRC_COLOR,
            /*color op*/     er::BlendOp::ADD,
            /*src alpha*/    er::BlendFactor::ZERO,
            /*dst alpha*/    er::BlendFactor::ONE_MINUS_SRC_ALPHA,
            /*alpha op*/     er::BlendOp::ADD);
        auto oit_blend_state =
            std::make_shared<er::PipelineColorBlendStateCreateInfo>(
                er::helper::fillPipelineColorBlendStateCreateInfo(
                    { accum_blend, reveal_blend }));

        auto oit_depth_stencil =
            std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
                er::helper::fillPipelineDepthStencilStateCreateInfo(
                    /*depth_test_enable */ true,
                    /*depth_write_enable*/ false,
                    er::CompareOp::LESS_OR_EQUAL));

        er::GraphicPipelineInfo oit_info = graphic_pipeline_info;
        oit_info.blend_state_info   = oit_blend_state;
        oit_info.depth_stencil_info = oit_depth_stencil;

        // OIT shader pair: reuse the existing vertex shader, swap fragment.
        er::ShaderModuleList oit_shader_modules(2);
        oit_shader_modules[0] = shader_modules[0];   // cluster_bindless_vert.spv
        oit_shader_modules[1] = er::helper::loadShaderModule(
            device_, "cluster_bindless_oit_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

        // OIT framebuffer format: 2 color attachments + same depth as host.
        er::PipelineRenderbufferFormats oit_fmt;
        oit_fmt.color_formats = { kOitAccumFormat, kOitRevealFormat };
        oit_fmt.depth_format  = framebuffer_format.depth_format;

        bindless_translucent_pipeline_ = device_->createPipeline(
            bindless_pipeline_layout_,   // same layout — same descriptor sets
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

        device_->updateDescriptorSets(writes);
    }

    // Texture staging views are no longer needed after descriptor write.
    staging_tex_views_.clear();
    staging_tex_views_.shrink_to_fit();
    staging_normal_tex_views_.clear();
    staging_normal_tex_views_.shrink_to_fit();

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

    // (Re)allocate the composite descriptor set against the new views.
    if (descriptor_pool_ && oit_composite_desc_set_layout_) {
        oit_composite_desc_set_ = device_->createDescriptorSets(
            descriptor_pool_, oit_composite_desc_set_layout_, 1)[0];

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

    // ── Pass 1: opaque + alpha-mask in caller's currently-active pass ──
    bind_geometry_state(bindless_pipeline_);
    cmd_buf->drawIndexedIndirectCount(
        indirect_draw_buffer_, 0,
        draw_count_buffer_, 0,
        total_clusters_all_meshes_);

    // If OIT resources aren't ready yet (e.g. shader compile failed), skip
    // the translucent draw entirely; opaque-only is already in the buffer.
    if (!bindless_translucent_pipeline_ || !oit_composite_pipeline_) {
        return prev_visible;
    }

    ensureOitTargets(screen_size);
    if (!oit_composite_desc_set_) {
        return prev_visible;
    }

    // ── End the caller's pass; we're about to begin our own ──
    cmd_buf->endDynamicRendering();

    // ── Pass 2: WBOIT translucent ──
    // Two color attachments (accum, reveal) + caller's depth (read-only).
    // Clear values: accum=(0,0,0,0), reveal=(1,0,0,0).  After the OIT
    // shaders' per-attachment blends, accum holds Σ(c·α·w, α·w) and
    // reveal holds Π(1−α).  See cluster_bindless.frag (OIT_OUTPUT path).
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
    bind_geometry_state(bindless_translucent_pipeline_);
    cmd_buf->drawIndexedIndirectCount(
        trans_indirect_draw_buffer_, 0,
        trans_draw_count_buffer_, 0,
        total_clusters_all_meshes_);
    cmd_buf->endDynamicRendering();

    // ── Transition accum/reveal: COLOR_ATTACHMENT → SHADER_READ_ONLY ──
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

    // ── Pass 3: fullscreen composite onto the host's color buffer ──
    // Now also attaches the host depth buffer so the composite fragment
    // shader can stamp gl_FragDepth = 0.99999 at glass pixels — this
    // prevents the downstream sky pass (LESS_OR_EQUAL vs depth=1.0) from
    // painting sky over the freshly-resolved glass colour at pixels where
    // no opaque geometry is behind the glass.
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
    cmd_buf->draw(3);   // fullscreen triangle generated in full_screen.vert
    cmd_buf->endDynamicRendering();

    // ── Re-begin the caller's pass with LOAD/LOAD ──
    // Preserves the API contract: caller's matching endDynamicRendering()
    // still has a pass open to close.
    er::RenderingAttachmentInfo host_color_att;
    host_color_att.image_view   = color_view;
    host_color_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    host_color_att.load_op      = er::AttachmentLoadOp::LOAD;
    host_color_att.store_op     = er::AttachmentStoreOp::STORE;

    er::RenderingAttachmentInfo host_depth_att;
    host_depth_att.image_view   = depth_view;
    host_depth_att.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    host_depth_att.load_op      = er::AttachmentLoadOp::LOAD;
    host_depth_att.store_op     = er::AttachmentStoreOp::STORE;

    er::RenderingInfo host_ri = {};
    host_ri.render_area_offset  = { 0, 0 };
    host_ri.render_area_extent  = screen_size;
    host_ri.layer_count         = 1;
    host_ri.view_mask           = 0;
    host_ri.color_attachments   = { host_color_att };
    host_ri.depth_attachments   = { host_depth_att };
    host_ri.stencil_attachments = {};
    cmd_buf->beginDynamicRendering(host_ri);

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

uint32_t ClusterRenderer::drawTranslucentForward(
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
    if (!bindless_translucent_pipeline_ || !oit_composite_pipeline_) {
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

    // ── End host pass (caller's pass on color_view + depth_view) ──
    cmd_buf->endDynamicRendering();

    // ── WBOIT translucent into accum + reveal ──
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

    cmd_buf->beginDynamicRendering(oit_ri);
    bind_geometry_state(bindless_translucent_pipeline_);
    cmd_buf->drawIndexedIndirectCount(
        trans_indirect_draw_buffer_, 0,
        trans_draw_count_buffer_, 0,
        total_clusters_all_meshes_);
    cmd_buf->endDynamicRendering();

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

    // ── Composite ──
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

    cmd_buf->beginDynamicRendering(comp_ri);
    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, oit_composite_pipeline_);
    cmd_buf->setViewports(viewports);
    cmd_buf->setScissors(scissors);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        oit_composite_pipeline_layout_,
        { oit_composite_desc_set_ });
    cmd_buf->draw(3);
    cmd_buf->endDynamicRendering();

    // ── Re-open caller's pass with LOAD/LOAD ──
    er::RenderingAttachmentInfo host_color_att;
    host_color_att.image_view   = color_view;
    host_color_att.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    host_color_att.load_op      = er::AttachmentLoadOp::LOAD;
    host_color_att.store_op     = er::AttachmentStoreOp::STORE;

    er::RenderingAttachmentInfo host_depth_att;
    host_depth_att.image_view   = depth_view;
    host_depth_att.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    host_depth_att.load_op      = er::AttachmentLoadOp::LOAD;
    host_depth_att.store_op     = er::AttachmentStoreOp::STORE;

    er::RenderingInfo host_ri = {};
    host_ri.render_area_offset  = { 0, 0 };
    host_ri.render_area_extent  = screen_size;
    host_ri.layer_count         = 1;
    host_ri.view_mask           = 0;
    host_ri.color_attachments   = { host_color_att };
    host_ri.depth_attachments   = { host_depth_att };
    cmd_buf->beginDynamicRendering(host_ri);

    return total_visible_all_meshes_;
}

// ─── Recreate (swap chain resize) ─────────────────────────────────────────

void ClusterRenderer::recreate(
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool) {

    if (!gpu_ready_ || !bindless_desc_set_layout_) return;

    descriptor_pool_ = descriptor_pool;

    // Re-allocate bindless descriptor set.
    auto desc_sets = device_->createDescriptorSets(
        descriptor_pool_, bindless_desc_set_layout_, 1);
    bindless_desc_set_ = desc_sets[0];

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

    device_->updateDescriptorSets(writes);
}

// ─── Destroy ──────────────────────────────────────────────────────────────

void ClusterRenderer::destroy() {
    bindless_pipeline_.reset();
    bindless_translucent_pipeline_.reset();
    bindless_gbuffer_pipeline_.reset();
    bindless_pipeline_layout_.reset();
    bindless_desc_set_.reset();
    bindless_desc_set_layout_.reset();
    cull_pipeline_.reset();
    cull_pipeline_layout_.reset();
    cull_desc_set_.reset();
    cull_desc_set_layout_.reset();

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
