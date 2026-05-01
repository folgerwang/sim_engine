#include "collision_mesh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stack>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "game_object/drawable_object.h"
#include "helper/bvh.h"

namespace engine {
namespace helper {

namespace {

glm::vec3 closestPtPointSegment(
    const glm::vec3& p,
    const glm::vec3& a,
    const glm::vec3& b,
    float* t_out = nullptr) {
    glm::vec3 ab = b - a;
    float ab_len2 = glm::dot(ab, ab);
    float t = (ab_len2 > 0.0f) ? glm::dot(p - a, ab) / ab_len2 : 0.0f;
    t = glm::clamp(t, 0.0f, 1.0f);
    if (t_out) *t_out = t;
    return a + t * ab;
}

glm::vec3 closestPtPointTriangle(
    const glm::vec3& p,
    const glm::vec3& a,
    const glm::vec3& b,
    const glm::vec3& c) {
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 ap = p - a;
    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

float closestPtSegmentSegment(
    const glm::vec3& p1, const glm::vec3& q1,
    const glm::vec3& p2, const glm::vec3& q2,
    glm::vec3& c1, glm::vec3& c2) {
    glm::vec3 d1 = q1 - p1;
    glm::vec3 d2 = q2 - p2;
    glm::vec3 r  = p1 - p2;
    float a = glm::dot(d1, d1);
    float e = glm::dot(d2, d2);
    float f = glm::dot(d2, r);

    const float kEps = 1e-8f;
    float s, t;
    if (a <= kEps && e <= kEps) {
        c1 = p1; c2 = p2;
        return glm::dot(c1 - c2, c1 - c2);
    }
    if (a <= kEps) {
        s = 0.0f;
        t = glm::clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = glm::dot(d1, r);
        if (e <= kEps) {
            t = 0.0f;
            s = glm::clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = glm::dot(d1, d2);
            float denom = a * e - b * b;
            s = (denom != 0.0f)
                ? glm::clamp((b * f - c * e) / denom, 0.0f, 1.0f)
                : 0.0f;
            float t_unc = (b * s + f) / e;
            if (t_unc < 0.0f) {
                t = 0.0f;
                s = glm::clamp(-c / a, 0.0f, 1.0f);
            } else if (t_unc > 1.0f) {
                t = 1.0f;
                s = glm::clamp((b - c) / a, 0.0f, 1.0f);
            } else {
                t = t_unc;
            }
        }
    }
    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return glm::dot(c1 - c2, c1 - c2);
}

void closestPtSegmentTriangle(
    const glm::vec3& s_a, const glm::vec3& s_b,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    glm::vec3& out_seg, glm::vec3& out_tri) {
    float best = std::numeric_limits<float>::max();
    auto consider = [&](const glm::vec3& sp, const glm::vec3& tp) {
        glm::vec3 d = sp - tp;
        float dd = glm::dot(d, d);
        if (dd < best) { best = dd; out_seg = sp; out_tri = tp; }
    };
    consider(s_a, closestPtPointTriangle(s_a, v0, v1, v2));
    consider(s_b, closestPtPointTriangle(s_b, v0, v1, v2));
    glm::vec3 c1, c2;
    closestPtSegmentSegment(s_a, s_b, v0, v1, c1, c2); consider(c1, c2);
    closestPtSegmentSegment(s_a, s_b, v1, v2, c1, c2); consider(c1, c2);
    closestPtSegmentSegment(s_a, s_b, v2, v0, c1, c2); consider(c1, c2);

    glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
    float nn = glm::dot(n, n);
    if (nn > 1e-12f) {
        float d_a = glm::dot(s_a - v0, n);
        float d_b = glm::dot(s_b - v0, n);
        if (d_a * d_b < 0.0f) {
            float t = d_a / (d_a - d_b);
            glm::vec3 pi = s_a + t * (s_b - s_a);
            glm::vec3 v0v1 = v1 - v0, v0v2 = v2 - v0, v0p = pi - v0;
            float dot00 = glm::dot(v0v2, v0v2);
            float dot01 = glm::dot(v0v2, v0v1);
            float dot02 = glm::dot(v0v2, v0p);
            float dot11 = glm::dot(v0v1, v0v1);
            float dot12 = glm::dot(v0v1, v0p);
            float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
            float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
            if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
                consider(pi, pi);
            }
        }
    }
}

bool aabbOverlap(const AABB& a, const AABB& b) {
    return  a.min_bounds.x <= b.max_bounds.x &&
            a.max_bounds.x >= b.min_bounds.x &&
            a.min_bounds.y <= b.max_bounds.y &&
            a.max_bounds.y >= b.min_bounds.y &&
            a.min_bounds.z <= b.max_bounds.z &&
            a.max_bounds.z >= b.min_bounds.z;
}

} // namespace

bool CollisionMesh::buildFromDrawable(
    const game_object::DrawableObject& drawable,
    bool build_bvh) {

    if (!drawable.isReady()) return false;
    const auto& data = drawable.getDrawableData();
    if (data.meshes_.empty()) return false;

    size_t est_verts = 0;
    size_t est_indices = 0;
    for (const auto& m : data.meshes_) {
        if (!m.vertex_position_) continue;
        est_verts += m.vertex_position_->size();
        for (const auto& p : m.primitives_) {
            if (p.vertex_indices_) est_indices += p.vertex_indices_->size();
        }
    }
    if (est_indices < 3) return false;
    vertices_.reserve(est_verts);
    indices_.reserve(est_indices);

    bounds_ = AABB();

    // Build mesh_idx → world-ish-space transform table. NodeInfo::
    // cached_matrix_ already includes the full parent hierarchy
    // (DrawableData::update bakes it). First node referencing each
    // mesh wins, mirroring the cluster-upload pattern in application
    // .cpp -- handles the common case of one node per mesh in FBX
    // bistro scenes. Without this, every mesh's positions land at the
    // origin in mesh-local space and the debug draw shows a tangled
    // pile of overlapping geometry instead of the actual scene layout.
    std::vector<glm::mat4> mesh_transforms(
        data.meshes_.size(), glm::mat4(1.0f));
    std::vector<bool> mesh_has_xform(data.meshes_.size(), false);
    for (const auto& node : data.nodes_) {
        if (node.mesh_idx_ >= 0 &&
            static_cast<size_t>(node.mesh_idx_) < data.meshes_.size()) {
            const size_t mi = static_cast<size_t>(node.mesh_idx_);
            if (!mesh_has_xform[mi]) {
                mesh_transforms[mi] = node.cached_matrix_;
                mesh_has_xform[mi] = true;
            }
        }
    }

    for (size_t mi = 0; mi < data.meshes_.size(); ++mi) {
        const auto& mesh = data.meshes_[mi];
        if (!mesh.vertex_position_) continue;
        const auto& positions = *mesh.vertex_position_;
        if (positions.empty()) continue;

        const glm::mat4& xform = mesh_transforms[mi];
        const int vert_offset = (int)vertices_.size();
        for (const auto& p : positions) {
            const glm::vec3 wp = glm::vec3(xform * glm::vec4(p, 1.0f));
            vertices_.push_back(wp);
            bounds_.extend(wp);
        }
        for (const auto& prim : mesh.primitives_) {
            if (!prim.vertex_indices_) continue;
            const auto& src = *prim.vertex_indices_;
            if (src.size() < 3 || (src.size() % 3) != 0) continue;
            for (int idx : src) indices_.push_back(idx + vert_offset);
        }
    }

    if (indices_.size() < 3) {
        vertices_.clear();
        indices_.clear();
        return false;
    }

    if (!build_bvh) {
        // Debug-visualisation path: caller doesn't need spatial queries,
        // so skip the multithreaded BVH build (seconds on Bistro-sized
        // input). The mesh is still drawable -- only resolveCapsule()
        // becomes a no-op for this mesh because bvh_root_ stays null.
        bvh_root_ = nullptr;
        return true;
    }

    BVHBuilder builder(vertices_, indices_, /*debug_mode=*/false);
    builder.build();
    bvh_root_ = builder.getRoot();
    return bvh_root_ != nullptr;
}

void CollisionMesh::queryBVH(
    const AABB& query_box,
    std::vector<int>& out_tri_indices) const {
    if (!bvh_root_) return;
    if (!aabbOverlap(query_box, bvh_root_->bounds)) return;

    std::vector<const BVHNode*> stack;
    stack.reserve(64);
    stack.push_back(bvh_root_.get());
    while (!stack.empty()) {
        const BVHNode* node = stack.back();
        stack.pop_back();
        if (!aabbOverlap(query_box, node->bounds)) continue;
        if (node->isLeaf()) {
            for (int tri_idx : node->primitive_ref_indices) {
                out_tri_indices.push_back(tri_idx);
            }
        } else {
            if (node->left)  stack.push_back(node->left.get());
            if (node->right) stack.push_back(node->right.get());
        }
    }
}

bool CollisionMesh::resolveCapsuleStep(
    glm::vec3& position,
    float radius,
    float height,
    glm::vec3& accum_normal,
    int& contact_count) const {
    if (indices_.empty() || !bvh_root_) return false;

    const float seg_top_y = std::max(height - radius, radius);
    const glm::vec3 seg_a = position + glm::vec3(0.0f, radius,    0.0f);
    const glm::vec3 seg_b = position + glm::vec3(0.0f, seg_top_y, 0.0f);

    AABB cap_box;
    cap_box.extend(position - glm::vec3(radius, 0.0f, radius));
    cap_box.extend(position + glm::vec3(radius, height, radius));
    const float kSlop = 1e-3f;
    cap_box.min_bounds -= glm::vec3(kSlop);
    cap_box.max_bounds += glm::vec3(kSlop);

    static thread_local std::vector<int> tri_scratch;
    tri_scratch.clear();
    queryBVH(cap_box, tri_scratch);
    if (tri_scratch.empty()) return false;

    glm::vec3 push_total(0.0f);
    glm::vec3 normal_total(0.0f);
    int hits = 0;

    for (int tri_idx : tri_scratch) {
        const glm::vec3& v0 = vertices_[indices_[3 * tri_idx + 0]];
        const glm::vec3& v1 = vertices_[indices_[3 * tri_idx + 1]];
        const glm::vec3& v2 = vertices_[indices_[3 * tri_idx + 2]];

        glm::vec3 seg_pt, tri_pt;
        closestPtSegmentTriangle(seg_a, seg_b, v0, v1, v2, seg_pt, tri_pt);

        glm::vec3 d = seg_pt - tri_pt;
        float dist2 = glm::dot(d, d);
        if (dist2 >= radius * radius) continue;

        float dist = std::sqrt(dist2);
        glm::vec3 n;
        if (dist > 1e-6f) {
            n = d / dist;
        } else {
            glm::vec3 face_n = glm::cross(v1 - v0, v2 - v0);
            float fn_len = glm::length(face_n);
            n = (fn_len > 1e-6f) ? (face_n / fn_len) : glm::vec3(0, 1, 0);
        }
        float pen = radius - dist;
        push_total   += n * pen;
        normal_total += n;
        ++hits;
    }

    if (hits == 0) return false;

    position    += push_total;
    accum_normal += normal_total;
    contact_count += hits;
    return true;
}

bool CollisionMesh::resolveCapsule(
    glm::vec3& position,
    float radius,
    float height,
    glm::vec3& out_normal,
    int max_iterations) const {
    if (empty()) return false;

    AABB cap_box;
    cap_box.extend(position - glm::vec3(radius, 0.0f, radius));
    cap_box.extend(position + glm::vec3(radius, height, radius));
    if (!aabbOverlap(cap_box, bounds_)) return false;

    glm::vec3 accum_normal(0.0f);
    int contacts = 0;
    bool any_hit = false;
    for (int it = 0; it < max_iterations; ++it) {
        bool touched = resolveCapsuleStep(
            position, radius, height, accum_normal, contacts);
        if (!touched) break;
        any_hit = true;
    }

    if (any_hit && contacts > 0) {
        float len = glm::length(accum_normal);
        out_normal = (len > 1e-6f) ? (accum_normal / len)
                                   : glm::vec3(0.0f, 1.0f, 0.0f);
    } else {
        out_normal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    return any_hit;
}

bool CollisionMesh::buildFromDrawableMesh(
    const game_object::DrawableObject& drawable,
    size_t mesh_idx,
    bool build_bvh) {

    if (!drawable.isReady()) return false;
    const auto& data = drawable.getDrawableData();
    if (mesh_idx >= data.meshes_.size()) return false;

    const auto& mesh = data.meshes_[mesh_idx];
    if (!mesh.vertex_position_) return false;
    const auto& positions = *mesh.vertex_position_;
    if (positions.empty()) return false;

    // First node referencing this mesh wins, mirroring the multi-mesh
    // path in buildFromDrawable() and the cluster-upload pattern in
    // application.cpp. Bakes parent hierarchy into vertices_ so the
    // per-draw model transform stays identity.
    glm::mat4 xform(1.0f);
    for (const auto& node : data.nodes_) {
        if (node.mesh_idx_ == static_cast<int32_t>(mesh_idx)) {
            xform = node.cached_matrix_;
            break;
        }
    }

    size_t est_indices = 0;
    for (const auto& p : mesh.primitives_) {
        if (p.vertex_indices_) est_indices += p.vertex_indices_->size();
    }
    if (est_indices < 3) return false;

    vertices_.reserve(positions.size());
    indices_.reserve(est_indices);
    bounds_ = AABB();

    for (const auto& p : positions) {
        const glm::vec3 wp = glm::vec3(xform * glm::vec4(p, 1.0f));
        vertices_.push_back(wp);
        bounds_.extend(wp);
    }
    for (const auto& prim : mesh.primitives_) {
        if (!prim.vertex_indices_) continue;
        const auto& src = *prim.vertex_indices_;
        if (src.size() < 3 || (src.size() % 3) != 0) continue;
        for (int idx : src) indices_.push_back(idx);
    }

    if (indices_.size() < 3) {
        vertices_.clear();
        indices_.clear();
        return false;
    }

    if (!build_bvh) {
        bvh_root_ = nullptr;
        return true;
    }

    BVHBuilder builder(vertices_, indices_, /*debug_mode=*/false);
    builder.build();
    bvh_root_ = builder.getRoot();
    return bvh_root_ != nullptr;
}

void CollisionWorld::drawDebug(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const std::vector<renderer::Viewport>& viewports,
    const std::vector<renderer::Scissor>& scissors) const {
    if (!CollisionDebugDraw::ready()) return;
    for (size_t i = 0; i < meshes_.size(); ++i) {
        const auto& m = meshes_[i];
        if (!m || m->empty()) continue;
        // Lazy upload of per-mesh GPU debug buffers on first draw. The
        // mesh_id we pass is just the world-list index -- stable across
        // frames, unique per mesh -- which the fragment shader hashes
        // into a flat segmentation colour.
        CollisionDebugDraw::uploadForMesh(
            device, *m, static_cast<uint32_t>(i), m->debugBuffers());
        if (!m->debugBuffers().ready()) continue;
        // Triangles are already in world space (FBX bake), so the
        // per-draw model transform is identity.
        CollisionDebugDraw::draw(
            cmd_buf,
            desc_set_list,
            m->debugBuffers(),
            glm::mat4(1.0f),
            viewports,
            scissors);
    }
}

void CollisionWorld::destroyDebugBuffers(
    const std::shared_ptr<renderer::Device>& device) {
    for (auto& m : meshes_) {
        if (m) m->debugBuffers().destroy(device);
    }
}

bool CollisionWorld::resolveCapsule(
    glm::vec3& position,
    float radius,
    float height,
    glm::vec3& out_normal) const {
    glm::vec3 accum(0.0f);
    int hits = 0;
    bool any = false;
    for (const auto& m : meshes_) {
        glm::vec3 n;
        if (m->resolveCapsule(position, radius, height, n)) {
            accum += n;
            ++hits;
            any = true;
        }
    }
    if (any && hits > 0) {
        float len = glm::length(accum);
        out_normal = (len > 1e-6f) ? (accum / len) : glm::vec3(0, 1, 0);
    } else {
        out_normal = glm::vec3(0, 1, 0);
    }
    return any;
}

} // namespace helper
} // namespace engine
