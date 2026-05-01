#include "collision_mesh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stack>
#include <unordered_map>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "game_object/drawable_object.h"
#include "helper/bvh.h"
#include "helper/mesh_tool.h"   // c_target_lod_ratio, decimateMesh, helper::Mesh

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

namespace {

// Conservatively voxelise a welded triangle soup into an occupancy
// grid. Each triangle marks every cell its world-space AABB
// overlaps -- not the tightest possible (a true tri-vs-AABB SAT
// would skip a few corner cells) but plenty accurate at the 5cm
// scale we use here, and runs in microseconds per primitive. The
// returned `grid_size` and `grid_origin` describe the cell layout;
// `occ` is a flat byte vector of size grid_size.x*y*z, 0=empty,
// 1=occupied.
struct VoxelGrid {
    glm::ivec3 grid_size;
    glm::vec3  origin;
    float      cell_size;
    std::vector<uint8_t> occ;

    bool inBounds(int i, int j, int k) const {
        return i >= 0 && j >= 0 && k >= 0 &&
               i < grid_size.x && j < grid_size.y && k < grid_size.z;
    }
    size_t idx(int i, int j, int k) const {
        return (static_cast<size_t>(k) * grid_size.y + j) * grid_size.x + i;
    }
    uint8_t get(int i, int j, int k) const {
        return inBounds(i, j, k) ? occ[idx(i, j, k)] : uint8_t(0);
    }
};

// Cap on total cells. Past this we coarsen `cell_size` upward so
// even a giant primitive (entire bistro floor as one part, say)
// produces a manageable grid. 1M cells = 1MB occupancy, ~milliseconds
// to fill.
static constexpr size_t kMaxVoxels = 1u << 20;

static bool buildVoxelGrid(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    float requested_cell_size,
    VoxelGrid& out) {

    if (positions.empty() || indices.size() < 3) return false;

    glm::vec3 lo( std::numeric_limits<float>::max());
    glm::vec3 hi(-std::numeric_limits<float>::max());
    for (const auto& p : positions) {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    const glm::vec3 ext_raw = hi - lo;

    // Expand any near-zero axis to a single cell so flat decals /
    // walls still produce a one-cell-thick volume.
    glm::vec3 ext = glm::max(ext_raw, glm::vec3(requested_cell_size * 0.5f));

    // Coarsen cell_size if the requested resolution would exceed the
    // global voxel cap. We pick the smallest scale factor s such that
    // (ext / (cell_size * s))^3 <= kMaxVoxels.
    float cell_size = requested_cell_size;
    {
        const double total =
            static_cast<double>(ext.x) *
            static_cast<double>(ext.y) *
            static_cast<double>(ext.z);
        const double per_cell =
            static_cast<double>(cell_size) *
            static_cast<double>(cell_size) *
            static_cast<double>(cell_size);
        if (total / per_cell > static_cast<double>(kMaxVoxels)) {
            const double scale = std::cbrt(
                total / (per_cell * static_cast<double>(kMaxVoxels)));
            cell_size = static_cast<float>(cell_size * scale);
        }
    }

    out.cell_size = cell_size;
    out.origin    = lo;
    out.grid_size = glm::ivec3(
        std::max(1, static_cast<int>(std::ceil(ext.x / cell_size))),
        std::max(1, static_cast<int>(std::ceil(ext.y / cell_size))),
        std::max(1, static_cast<int>(std::ceil(ext.z / cell_size))));
    const size_t total_cells =
        static_cast<size_t>(out.grid_size.x) *
        static_cast<size_t>(out.grid_size.y) *
        static_cast<size_t>(out.grid_size.z);
    if (total_cells == 0 || total_cells > kMaxVoxels) return false;
    out.occ.assign(total_cells, uint8_t(0));

    const float inv_cell = 1.0f / cell_size;
    const size_t tri_count = indices.size() / 3;
    for (size_t t = 0; t < tri_count; ++t) {
        const glm::vec3& v0 = positions[indices[3 * t + 0]];
        const glm::vec3& v1 = positions[indices[3 * t + 1]];
        const glm::vec3& v2 = positions[indices[3 * t + 2]];

        const glm::vec3 tmin = glm::min(glm::min(v0, v1), v2);
        const glm::vec3 tmax = glm::max(glm::max(v0, v1), v2);

        const glm::ivec3 vmin = glm::clamp(
            glm::ivec3(glm::floor((tmin - out.origin) * inv_cell)),
            glm::ivec3(0),
            out.grid_size - 1);
        const glm::ivec3 vmax = glm::clamp(
            glm::ivec3(glm::floor((tmax - out.origin) * inv_cell)),
            glm::ivec3(0),
            out.grid_size - 1);

        for (int k = vmin.z; k <= vmax.z; ++k)
        for (int j = vmin.y; j <= vmax.y; ++j)
        for (int i = vmin.x; i <= vmax.x; ++i) {
            out.occ[out.idx(i, j, k)] = 1;
        }
    }
    return true;
}

// Surface-mesh extraction: for every occupied cell, emit a face for
// each of the 6 sides whose outward neighbour is empty (or off-grid).
// Each face is 4 fresh vertices + 2 triangles; we don't dedupe so the
// wireframe overlay shows clean per-face outlines instead of merging
// them into long strips.
static void emitVoxelCubeSurface(
    const VoxelGrid& g,
    std::vector<glm::vec3>& out_pos,
    std::vector<uint32_t>& out_ind) {

    auto push_face = [&](const glm::vec3& a, const glm::vec3& b,
                         const glm::vec3& c, const glm::vec3& d) {
        const uint32_t base = static_cast<uint32_t>(out_pos.size());
        out_pos.push_back(a);
        out_pos.push_back(b);
        out_pos.push_back(c);
        out_pos.push_back(d);
        // CCW from outside: (a, b, c) and (a, c, d).
        out_ind.push_back(base + 0);
        out_ind.push_back(base + 1);
        out_ind.push_back(base + 2);
        out_ind.push_back(base + 0);
        out_ind.push_back(base + 2);
        out_ind.push_back(base + 3);
    };

    for (int k = 0; k < g.grid_size.z; ++k)
    for (int j = 0; j < g.grid_size.y; ++j)
    for (int i = 0; i < g.grid_size.x; ++i) {
        if (!g.occ[g.idx(i, j, k)]) continue;

        const glm::vec3 lo = g.origin +
            g.cell_size * glm::vec3(i, j, k);
        const glm::vec3 hi = lo + g.cell_size;

        // Cell corners, indexed as bxyz in {lo,hi}.
        const glm::vec3 c000(lo.x, lo.y, lo.z);
        const glm::vec3 c100(hi.x, lo.y, lo.z);
        const glm::vec3 c010(lo.x, hi.y, lo.z);
        const glm::vec3 c110(hi.x, hi.y, lo.z);
        const glm::vec3 c001(lo.x, lo.y, hi.z);
        const glm::vec3 c101(hi.x, lo.y, hi.z);
        const glm::vec3 c011(lo.x, hi.y, hi.z);
        const glm::vec3 c111(hi.x, hi.y, hi.z);

        if (!g.get(i - 1, j, k)) push_face(c000, c001, c011, c010);  // -X
        if (!g.get(i + 1, j, k)) push_face(c100, c110, c111, c101);  // +X
        if (!g.get(i, j - 1, k)) push_face(c000, c100, c101, c001);  // -Y
        if (!g.get(i, j + 1, k)) push_face(c010, c011, c111, c110);  // +Y
        if (!g.get(i, j, k - 1)) push_face(c000, c010, c110, c100);  // -Z
        if (!g.get(i, j, k + 1)) push_face(c001, c101, c111, c011);  // +Z
    }
}

// Per-cell octahedron: 6 verts, 8 triangles, fits within the cell
// inscribed sphere of radius cell_size/2. Cheap, recognisable as a
// blob, no adjacency culling so total tris == 8 * occupied cells.
static void emitVoxelSphereCloud(
    const VoxelGrid& g,
    std::vector<glm::vec3>& out_pos,
    std::vector<uint32_t>& out_ind) {

    static const glm::vec3 oct[6] = {
        { 1.0f,  0.0f,  0.0f}, {-1.0f,  0.0f,  0.0f},
        { 0.0f,  1.0f,  0.0f}, { 0.0f, -1.0f,  0.0f},
        { 0.0f,  0.0f,  1.0f}, { 0.0f,  0.0f, -1.0f},
    };
    static const uint32_t tris[24] = {
        0, 2, 4,   2, 1, 4,   1, 3, 4,   3, 0, 4,
        2, 0, 5,   1, 2, 5,   3, 1, 5,   0, 3, 5,
    };
    const float r = g.cell_size * 0.5f;

    for (int k = 0; k < g.grid_size.z; ++k)
    for (int j = 0; j < g.grid_size.y; ++j)
    for (int i = 0; i < g.grid_size.x; ++i) {
        if (!g.occ[g.idx(i, j, k)]) continue;
        const glm::vec3 center = g.origin +
            g.cell_size * (glm::vec3(i, j, k) + 0.5f);
        const uint32_t base = static_cast<uint32_t>(out_pos.size());
        for (int v = 0; v < 6; ++v) {
            out_pos.push_back(center + r * oct[v]);
        }
        for (int t = 0; t < 24; ++t) {
            out_ind.push_back(base + tris[t]);
        }
    }
}

} // anonymous namespace

bool CollisionMesh::buildFromDrawablePrimitive(
    const game_object::DrawableObject& drawable,
    size_t mesh_idx,
    size_t prim_idx,
    bool build_bvh,
    CollisionShape shape,
    float weld_eps,
    float voxel_size) {

    if (!drawable.isReady()) return false;
    const auto& data = drawable.getDrawableData();
    if (mesh_idx >= data.meshes_.size()) return false;
    const auto& mesh = data.meshes_[mesh_idx];
    if (prim_idx >= mesh.primitives_.size()) return false;
    const auto& prim = mesh.primitives_[prim_idx];
    if (!prim.vertex_indices_) return false;
    if (!mesh.vertex_position_) return false;

    const auto& src_indices = *prim.vertex_indices_;
    const auto& src_positions = *mesh.vertex_position_;
    if (src_indices.size() < 3 || (src_indices.size() % 3) != 0) {
        return false;
    }
    if (src_positions.empty()) return false;

    // Material name comes straight from the drawable's MaterialInfo
    // (populated at FBX/GLTF load). Negative material_idx_ is a
    // valid "no material" marker -- leave material_name_ empty.
    if (prim.material_idx_ >= 0 &&
        static_cast<size_t>(prim.material_idx_) < data.materials_.size()) {
        material_name_ = data.materials_[prim.material_idx_].name_;
    }

    // First node referencing this mesh wins -- bakes parent
    // hierarchy into vertices_ so the per-draw model transform
    // stays identity.
    glm::mat4 xform(1.0f);
    for (const auto& node : data.nodes_) {
        if (node.mesh_idx_ == static_cast<int32_t>(mesh_idx)) {
            xform = node.cached_matrix_;
            break;
        }
    }

    // Build a tightly packed (vertices, indices) for just this
    // primitive's referenced subset. src_positions covers the WHOLE
    // mesh (all primitives), so we walk src_indices, dedupe via a
    // map, and remap indices into the compact array. This keeps the
    // decimater working on minimum input -- a primitive that touches
    // 200 vertices doesn't ship the mesh's other 5000.
    std::unordered_map<int, uint32_t> remap;
    remap.reserve(src_indices.size() / 2);
    std::vector<glm::vec3> packed_positions;
    packed_positions.reserve(src_indices.size() / 2);
    std::vector<uint32_t> packed_indices;
    packed_indices.reserve(src_indices.size());

    for (int idx : src_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= src_positions.size()) {
            // Malformed source -- skip whole triangle to keep tri
            // alignment for downstream non-indexed expansion.
            const size_t partial = packed_indices.size() % 3;
            if (partial) packed_indices.resize(packed_indices.size() - partial);
            continue;
        }
        auto it = remap.find(idx);
        uint32_t new_idx;
        if (it == remap.end()) {
            new_idx = static_cast<uint32_t>(packed_positions.size());
            remap[idx] = new_idx;
            const glm::vec3 wp =
                glm::vec3(xform * glm::vec4(src_positions[idx], 1.0f));
            packed_positions.push_back(wp);
        } else {
            new_idx = it->second;
        }
        packed_indices.push_back(new_idx);
    }

    if (packed_indices.size() < 3) return false;

    // ── Vertex welding ────────────────────────────────────────────────
    // Authored / exported geometry frequently has sub-mm gaps where
    // two parts meet (separate FBX nodes, floating-point quantisation
    // at part boundaries, asset-pipeline rounding). Quantise each
    // vertex position to a `weld_eps`-sized grid and dedupe -- vertices
    // that round to the same cell collapse into one, closing the gap.
    // Then drop triangles whose three corners welded to <=2 distinct
    // vertices (they have zero area and would confuse the decimater).
    //
    // Done BEFORE decimation so what were boundary edges across a gap
    // become interior edges in the welded mesh, which the QEM pass
    // can then collapse across.
    if (weld_eps > 0.0f && packed_indices.size() >= 3) {
        struct IVec3Hash {
            size_t operator()(const glm::ivec3& v) const noexcept {
                // 3-way mix from boost::hash_combine -- adequate
                // dispersion for small grids and fast enough that
                // welding adds <1% to per-primitive build time.
                size_t h = std::hash<int32_t>()(v.x);
                h ^= std::hash<int32_t>()(v.y) + 0x9e3779b9u + (h << 6) + (h >> 2);
                h ^= std::hash<int32_t>()(v.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
                return h;
            }
        };
        const float inv_eps = 1.0f / weld_eps;
        auto quantize = [inv_eps](const glm::vec3& p) {
            return glm::ivec3(
                static_cast<int32_t>(std::lround(p.x * inv_eps)),
                static_cast<int32_t>(std::lround(p.y * inv_eps)),
                static_cast<int32_t>(std::lround(p.z * inv_eps)));
        };

        std::unordered_map<glm::ivec3, uint32_t, IVec3Hash> spatial;
        spatial.reserve(packed_positions.size());
        std::vector<uint32_t> remap_to_welded(packed_positions.size());
        std::vector<glm::vec3> welded_positions;
        welded_positions.reserve(packed_positions.size());

        for (size_t i = 0; i < packed_positions.size(); ++i) {
            const glm::ivec3 key = quantize(packed_positions[i]);
            auto it = spatial.find(key);
            if (it == spatial.end()) {
                const uint32_t new_idx =
                    static_cast<uint32_t>(welded_positions.size());
                welded_positions.push_back(packed_positions[i]);
                spatial.emplace(key, new_idx);
                remap_to_welded[i] = new_idx;
            } else {
                remap_to_welded[i] = it->second;
            }
        }

        // Remap the index list onto welded indices, dropping any
        // triangle that collapsed to a degenerate (line / point).
        std::vector<uint32_t> welded_indices;
        welded_indices.reserve(packed_indices.size());
        for (size_t t = 0; t < packed_indices.size(); t += 3) {
            const uint32_t a = remap_to_welded[packed_indices[t + 0]];
            const uint32_t b = remap_to_welded[packed_indices[t + 1]];
            const uint32_t c = remap_to_welded[packed_indices[t + 2]];
            if (a == b || b == c || c == a) continue;
            welded_indices.push_back(a);
            welded_indices.push_back(b);
            welded_indices.push_back(c);
        }
        if (welded_indices.size() < 3) return false;

        packed_positions = std::move(welded_positions);
        packed_indices   = std::move(welded_indices);
    }

    // Shape finalisation: turn the welded triangle list into the
    // requested debug / collision proxy.
    switch (shape) {
    case CollisionShape::None:
        // Pass-through. Welding above already happened.
        break;

    case CollisionShape::Decimate: {
        // OpenMesh QEM decimation via helper::decimateMesh
        // (mesh_tool.cpp). One material part per CollisionMesh so
        // face_part_ids is a flat zero vector. Target
        // c_target_lod_ratio (~30%) of original face count, floor 4.
        if (packed_indices.size() < 3 * 12) break;
        Mesh src;
        src.vertex_data_ptr->reserve(packed_positions.size());
        for (const auto& p : packed_positions) {
            VertexStruct v{};
            v.position = p;
            v.normal   = glm::vec3(0.0f, 1.0f, 0.0f);
            v.uv       = glm::vec2(0.0f);
            src.vertex_data_ptr->push_back(v);
        }
        const size_t face_count = packed_indices.size() / 3;
        src.faces_ptr->reserve(face_count);
        for (size_t f = 0; f < face_count; ++f) {
            src.faces_ptr->emplace_back(
                packed_indices[3 * f + 0],
                packed_indices[3 * f + 1],
                packed_indices[3 * f + 2]);
        }
        std::vector<int32_t> part_ids(face_count, 0);
        Mesh dst;
        std::vector<int32_t> dst_part_ids;
        const size_t target = std::max<size_t>(
            4, static_cast<size_t>(face_count * c_target_lod_ratio));
        std::ostringstream silent;
        decimateMesh(src, part_ids, dst, dst_part_ids, target, silent);
        if (dst.isValid() && dst.getFaceCount() >= 1 &&
            !dst.vertex_data_ptr->empty()) {
            packed_positions.clear();
            packed_indices.clear();
            packed_positions.reserve(dst.getVertexCount());
            for (const auto& v : *dst.vertex_data_ptr) {
                packed_positions.push_back(v.position);
            }
            packed_indices.reserve(dst.getFaceCount() * 3);
            for (const auto& f : *dst.faces_ptr) {
                packed_indices.push_back(f.v_indices[0]);
                packed_indices.push_back(f.v_indices[1]);
                packed_indices.push_back(f.v_indices[2]);
            }
        }
        break;
    }

    case CollisionShape::VoxelCube:
    case CollisionShape::VoxelSphere: {
        // Voxelise the welded triangle list at `voxel_size` and
        // re-mesh from the occupancy grid. VoxelCube emits a
        // face-culled cube surface ("Minecraft" view); VoxelSphere
        // emits a low-poly octahedron per occupied cell. Both modes
        // produce a clean gap-free shell of the primitive at a
        // visibly chunky resolution.
        VoxelGrid grid;
        if (!buildVoxelGrid(packed_positions, packed_indices,
                            voxel_size, grid)) {
            return false;
        }
        std::vector<glm::vec3> vox_pos;
        std::vector<uint32_t>  vox_ind;
        if (shape == CollisionShape::VoxelCube) {
            emitVoxelCubeSurface(grid, vox_pos, vox_ind);
        } else {
            emitVoxelSphereCloud(grid, vox_pos, vox_ind);
        }
        if (vox_ind.size() < 3) return false;
        packed_positions = std::move(vox_pos);
        packed_indices   = std::move(vox_ind);
        break;
    }
    }

    // Move into the CollisionMesh's storage (signed int32 to match
    // the existing API used by debugIndices() / queryBVH()).
    vertices_ = std::move(packed_positions);
    indices_.clear();
    indices_.reserve(packed_indices.size());
    for (uint32_t i : packed_indices) indices_.push_back(static_cast<int>(i));

    bounds_ = AABB();
    for (const auto& v : vertices_) bounds_.extend(v);

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
    // Two passes per CollisionMesh: solid hashed-colour fill first,
    // then a white wireframe overlay so the user can see each mesh's
    // triangulation against the segmentation colour. Drawing the
    // wireframe pass per-mesh (instead of as one combined sweep at
    // the end) keeps each mesh's edges next to its fill in the
    // command stream -- minor pipeline-binding overhead but cleaner
    // depth-bias behaviour because the relevant fill is the most
    // recent thing in the depth buffer.
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
        // Triangles are already in world space (node transforms baked
        // in during buildFromDrawablePrimitive), so the per-draw model
        // transform is identity.
        CollisionDebugDraw::draw(
            cmd_buf,
            desc_set_list,
            m->debugBuffers(),
            glm::mat4(1.0f),
            viewports,
            scissors);
        CollisionDebugDraw::drawWireframe(
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
