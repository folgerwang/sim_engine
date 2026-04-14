#include <stack>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalFlippingT.hh>
#include "mesh_tool.h"

namespace engine {
namespace helper {

// --- OpenMesh HLOD Generation ---

// --- OpenMesh Structures (MyTraits, MyMesh) ---
// Unchanged, as these are type definitions.
struct MyTraits : public OpenMesh::DefaultTraits {
    VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge);
    FaceAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);
    VertexTraits {
    private:
        TexCoord2D tex_coord_;
    public:
        VertexT() : tex_coord_(TexCoord2D(0.0f, 0.0f)) {}
        const TexCoord2D& texcoord() const { return tex_coord_; }
        void set_texcoord(const TexCoord2D& _tex) { tex_coord_ = _tex; }
    };
};
typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> MyMesh;


/**
 * @brief Generates a simplified mesh using a robust method for vertex locking.
 * This version adheres to the specified naming conventions.
 *
 * @param input_mesh The original mesh to be simplified.
 * @param output_mesh The mesh structure to store the simplified result.
 * @param target_face_count The desired number of faces in the simplified mesh.
 * @param protected_vertex_indices A set of indices for vertices that should not be collapsed.
 */
void generateHLODWithSeamProtection(
    const Mesh& input_mesh,
    Mesh& output_mesh,
    size_t target_face_count,
    const std::set<uint32_t>& protected_vertex_indices,
    std::ostream& log) {

    // ... (Initial checks and mesh conversion logic remain the same) ...
    if (!input_mesh.isValid() || input_mesh.getFaceCount() == 0) { return; }
    if (target_face_count >= input_mesh.getFaceCount()) { output_mesh = input_mesh; return; }

    MyMesh om_mesh;
    om_mesh.request_vertex_status();
    om_mesh.request_edge_status();
    om_mesh.request_face_status();
    om_mesh.request_vertex_normals();
    om_mesh.request_vertex_texcoords2D();

    // Build OM vertices with position-based deduplication.
    // UV-seam vertices share the same 3-D position but differ in normal/UV.
    // Treating them as separate OM vertices causes every shared edge to appear
    // twice, triggering "complex edge" errors.  We map by position so that
    // topologically adjacent faces always reference the same vertex handle.
    std::map<std::tuple<float, float, float>, MyMesh::VertexHandle> pos_to_handle;
    std::vector<MyMesh::VertexHandle> vertex_handles;
    vertex_handles.reserve(input_mesh.vertex_data_ptr->size());

    for (const auto& v : *input_mesh.vertex_data_ptr) {
        auto key = std::make_tuple(v.position.x, v.position.y, v.position.z);
        auto [it, inserted] = pos_to_handle.emplace(key, MyMesh::VertexHandle{});
        if (inserted) {
            MyMesh::VertexHandle vh = om_mesh.add_vertex(
                MyMesh::Point(v.position.x, v.position.y, v.position.z));
            om_mesh.set_normal(vh, MyMesh::Normal(v.normal.x, v.normal.y, v.normal.z));
            om_mesh.set_texcoord2D(vh, MyMesh::TexCoord2D(v.uv.x, v.uv.y));
            it->second = vh;
        }
        vertex_handles.push_back(it->second);
    }

    // Lock protected vertices (indices still valid after merging).
    for (uint32_t vertex_idx : protected_vertex_indices) {
        if (vertex_idx < vertex_handles.size()) {
            om_mesh.status(vertex_handles[vertex_idx]).set_locked(true);
        }
    }

    // Add faces.  Redirect std::cerr so any residual OpenMesh topology warnings
    // go to our log stream instead of the console.
    std::streambuf* saved_cerr = std::cerr.rdbuf(log.rdbuf());
    int skipped_faces = 0;
    for (const auto& input_face : *input_mesh.faces_ptr) {
        if (input_face.isDegenerate()) { ++skipped_faces; continue; }

        MyMesh::VertexHandle vh0 = vertex_handles[input_face.v_indices[0]];
        MyMesh::VertexHandle vh1 = vertex_handles[input_face.v_indices[1]];
        MyMesh::VertexHandle vh2 = vertex_handles[input_face.v_indices[2]];

        // After position-merging, distinct indices may map to the same handle.
        if (vh0 == vh1 || vh1 == vh2 || vh0 == vh2) { ++skipped_faces; continue; }

        auto fh = om_mesh.add_face({ vh0, vh1, vh2 });
        if (!fh.is_valid()) { ++skipped_faces; }
    }
    std::cerr.rdbuf(saved_cerr);  // restore cerr

    if (skipped_faces > 0) {
        log << "[HLOD] " << skipped_faces
            << " face(s) skipped (non-manifold or degenerate after position-merge).\n";
    }

    // --- Perform Decimation using the API from your header files ---
    OpenMesh::Decimater::DecimaterT<MyMesh> decimater(om_mesh);

    // Define the correct handle type for the quadric module.
    OpenMesh::Decimater::ModHandleT<OpenMesh::Decimater::ModQuadricT<MyMesh>> quadric_handle;

    // Register the module using the 'add' function found in BaseDecimaterT.hh
    decimater.add(quadric_handle); // This will create the module and populate the handle.

    // 2. Define a handle for the normal flipping module (to preserve features)
    OpenMesh::Decimater::ModHandleT<OpenMesh::Decimater::ModNormalFlippingT<MyMesh>> normal_flipping_handle;
    decimater.add(normal_flipping_handle); //

    // --- Configuration ---

    // Configure the normal flipping module
    // This tells the decimater to forbid any collapse that would change a face's
    // normal by more than 30 degrees. This is very effective at keeping sharp edges.
    decimater.module(normal_flipping_handle).set_max_normal_deviation(30.0f); //

    // Initialize the decimater. This should now succeed.
    decimater.initialize(); //

    log << "Decimater initialized: " << std::boolalpha << decimater.is_initialized() << std::endl;
    if (!decimater.is_initialized()) {
        log << "[ERROR] Decimater failed to initialize." << std::endl;
        output_mesh = input_mesh;
        return;
    }

    // Call decimate_to_faces
    decimater.decimate_to_faces(0, target_face_count);

    om_mesh.garbage_collection();

    // ... (Conversion back to your format - Unchanged) ...
    output_mesh.vertex_data_ptr->clear();
    output_mesh.faces_ptr->clear();
    std::map<int, int> old_to_new_indices;
    int new_vertex_index = 0;
    for (MyMesh::VertexIter v_it = om_mesh.vertices_begin(); v_it != om_mesh.vertices_end(); ++v_it) {
        VertexStruct new_vertex;
        MyMesh::Point point = om_mesh.point(*v_it);
        new_vertex.position = glm::vec3(point[0], point[1], point[2]);
        MyMesh::Normal normal = om_mesh.normal(*v_it);
        new_vertex.normal = glm::vec3(normal[0], normal[1], normal[2]);
        MyMesh::TexCoord2D uv_coord = om_mesh.texcoord2D(*v_it);
        new_vertex.uv = glm::vec2(uv_coord[0], uv_coord[1]);
        output_mesh.vertex_data_ptr->push_back(new_vertex);
        old_to_new_indices[v_it->idx()] = new_vertex_index++;
    }
    for (MyMesh::FaceIter f_it = om_mesh.faces_begin(); f_it != om_mesh.faces_end(); ++f_it) {
        Face new_face;
        int i = 0;
        for (MyMesh::FaceVertexIter fv_it = om_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it, ++i) {
            new_face.v_indices[i] = old_to_new_indices[fv_it->idx()];
        }
        output_mesh.faces_ptr->push_back(new_face);
    }
}

// =============================================================================
// Local Quadric Error Metrics (QEM) decimation — no external library needed.
// Processes all material parts as one object so seams are never broken.
// =============================================================================

namespace {

// ----- 4×4 symmetric quadric stored as 10 upper-triangle doubles -------------
// Layout: q[00,01,02,03, 11,12,13, 22,23, 33]
struct Quadric {
    double q[10] = {};

    void addPlane(double a, double b, double c, double d) {
        q[0]+=a*a; q[1]+=a*b; q[2]+=a*c; q[3]+=a*d;
        q[4]+=b*b; q[5]+=b*c; q[6]+=b*d;
        q[7]+=c*c; q[8]+=c*d;
        q[9]+=d*d;
    }
    void operator+=(const Quadric& o) {
        for (int i=0;i<10;i++) q[i]+=o.q[i];
    }
    double eval(double x, double y, double z) const {
        return q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x
             + q[4]*y*y + 2*q[5]*y*z + 2*q[6]*y
             + q[7]*z*z + 2*q[8]*z + q[9];
    }
    // Solve [q00 q01 q02; q01 q11 q12; q02 q12 q22]*v = -[q03;q13;q23]
    bool optimal(double& ox, double& oy, double& oz) const {
        double a00=q[0],a01=q[1],a02=q[2];
        double          a11=q[4],a12=q[5];
        double                   a22=q[7];
        double b0=-q[3],b1=-q[6],b2=-q[8];
        double det = a00*(a11*a22-a12*a12)
                   - a01*(a01*a22-a12*a02)
                   + a02*(a01*a12-a11*a02);
        if (std::abs(det) < 1e-12) return false;
        double inv = 1.0/det;
        ox = inv*(b0*(a11*a22-a12*a12) + a01*(-b1*a22+a12*b2) + a02*(b1*a12-a11*b2));
        oy = inv*(a00*(b1*a22-a12*b2)  + b0*(-a01*a22+a12*a02) + a02*(a01*b2-b1*a02));
        oz = inv*(a00*(a11*b2-b1*a12)  + a01*(-a01*b2+b1*a02)  + b0*(a01*a12-a11*a02));
        return true;
    }
};

struct QEMVert {
    glm::dvec3 pos;
    Quadric    Q;
    bool       locked  = false;
    bool       deleted = false;
    std::vector<int> adj_faces;
    std::vector<int> adj_verts;
};

struct QEMFace {
    int      v[3];
    int32_t  part_id;
    bool     deleted = false;
};

struct CollapseEntry {
    float      cost;
    int        v0, v1;   // canonical: v0 < v1
    int        version;
    glm::dvec3 new_pos;
    bool operator>(const CollapseEntry& o) const { return cost > o.cost; }
};

static inline uint64_t edgeKey(int a, int b) {
    return ((uint64_t)(unsigned)std::min(a,b) << 32) | (uint32_t)std::max(a,b);
}

static void facePlane(const glm::dvec3& p0, const glm::dvec3& p1,
                      const glm::dvec3& p2,
                      double& a, double& b, double& c, double& d) {
    glm::dvec3 n = glm::cross(p1-p0, p2-p0);
    double len = glm::length(n);
    if (len < 1e-15) { a=b=c=d=0.0; return; }
    n /= len;
    a = n.x; b = n.y; c = n.z;
    d = -glm::dot(n, p0);
}

static CollapseEntry computeCollapse(
    const std::vector<QEMVert>& verts, int v0, int v1,
    const std::unordered_map<uint64_t,int>& edge_ver)
{
    CollapseEntry e;
    e.v0 = v0; e.v1 = v1;
    auto it = edge_ver.find(edgeKey(v0,v1));
    e.version = (it != edge_ver.end()) ? it->second : 0;

    // If either endpoint is locked, use its position directly — no need to solve.
    if (verts[v0].locked) {
        e.new_pos = verts[v0].pos;
        Quadric Q = verts[v0].Q; Q += verts[v1].Q;
        e.cost = (float)std::max(0.0, Q.eval(e.new_pos.x, e.new_pos.y, e.new_pos.z));
        return e;
    }
    if (verts[v1].locked) {
        e.new_pos = verts[v1].pos;
        Quadric Q = verts[v0].Q; Q += verts[v1].Q;
        e.cost = (float)std::max(0.0, Q.eval(e.new_pos.x, e.new_pos.y, e.new_pos.z));
        return e;
    }

    Quadric Q = verts[v0].Q;
    Q += verts[v1].Q;

    glm::dvec3 p0 = verts[v0].pos, p1 = verts[v1].pos;
    glm::dvec3 mid = (p0 + p1) * 0.5;

    double ox, oy, oz;
    if (Q.optimal(ox, oy, oz)) {
        e.new_pos = {ox, oy, oz};

        // Overshoot guard: if the optimal position is farther than 3× the edge
        // length from both endpoints the quadric is ill-conditioned — fall back
        // to whichever endpoint or midpoint gives lower cost.
        double edge_sq = glm::dot(p1 - p0, p1 - p0);
        double thresh  = edge_sq * 9.0;   // (3× edge length)²
        bool d0_far = glm::dot(e.new_pos - p0, e.new_pos - p0) > thresh;
        bool d1_far = glm::dot(e.new_pos - p1, e.new_pos - p1) > thresh;
        if (d0_far && d1_far) {
            // fall through to endpoint/midpoint selection below
            double c0 = Q.eval(p0.x, p0.y, p0.z);
            double c1 = Q.eval(p1.x, p1.y, p1.z);
            double cm = Q.eval(mid.x, mid.y, mid.z);
            if      (c0 <= c1 && c0 <= cm) e.new_pos = p0;
            else if (c1 <= cm)             e.new_pos = p1;
            else                           e.new_pos = mid;
        }
    } else {
        // Singular matrix — pick best of endpoints and midpoint
        double c0 = Q.eval(p0.x, p0.y, p0.z);
        double c1 = Q.eval(p1.x, p1.y, p1.z);
        double cm = Q.eval(mid.x, mid.y, mid.z);
        if      (c0 <= c1 && c0 <= cm) e.new_pos = p0;
        else if (c1 <= cm)             e.new_pos = p1;
        else                           e.new_pos = mid;
    }
    double cost = Q.eval(e.new_pos.x, e.new_pos.y, e.new_pos.z);
    e.cost = (float)std::max(0.0, cost);
    return e;
}

} // anonymous namespace

// -----------------------------------------------------------------------------

void decimateMesh(
    const Mesh& input_mesh,
    const std::vector<int32_t>& face_part_ids,
    Mesh& output_mesh,
    std::vector<int32_t>& output_face_part_ids,
    size_t target_face_count,
    std::ostream& log)
{
    const auto& in_verts = *input_mesh.vertex_data_ptr;
    const auto& in_faces = *input_mesh.faces_ptr;
    const int nv = (int)in_verts.size();
    const int nf = (int)in_faces.size();

    if (nf == 0 || target_face_count >= (size_t)nf) {
        output_mesh = input_mesh;
        output_face_part_ids = face_part_ids;
        return;
    }

    // ── Build working arrays ──────────────────────────────────────────────────
    std::vector<QEMVert> verts(nv);
    std::vector<QEMFace> faces(nf);

    for (int i = 0; i < nv; ++i)
        verts[i].pos = glm::dvec3(in_verts[i].position);

    for (int i = 0; i < nf; ++i) {
        faces[i].v[0]    = (int)in_faces[i].v_indices[0];
        faces[i].v[1]    = (int)in_faces[i].v_indices[1];
        faces[i].v[2]    = (int)in_faces[i].v_indices[2];
        faces[i].part_id = face_part_ids[i];
    }

    // ── Vertex → face adjacency ───────────────────────────────────────────────
    for (int fi = 0; fi < nf; ++fi)
        for (int k = 0; k < 3; ++k)
            verts[faces[fi].v[k]].adj_faces.push_back(fi);

    // ── Edge → face map (for boundary / seam detection) ───────────────────────
    std::unordered_map<uint64_t, std::vector<int>> edge_face_map;
    edge_face_map.reserve((size_t)nf * 3);
    for (int fi = 0; fi < nf; ++fi)
        for (int k = 0; k < 3; ++k) {
            int a = faces[fi].v[k], b = faces[fi].v[(k+1)%3];
            edge_face_map[edgeKey(a,b)].push_back(fi);
        }

    // ── Lock boundary and inter-part seam vertices ────────────────────────────
    // Boundary edge  : appears in only 1 face  → lock both endpoints
    // Seam edge      : shared by faces of different parts → lock both endpoints
    for (auto& [key, flist] : edge_face_map) {
        bool lock = (flist.size() == 1); // open boundary
        if (!lock) {
            int32_t p0 = faces[flist[0]].part_id;
            for (size_t i = 1; i < flist.size(); ++i)
                if (faces[flist[i]].part_id != p0) { lock = true; break; }
        }
        if (lock) {
            int va = (int)(key >> 32);
            int vb = (int)(uint32_t)key;
            verts[va].locked = verts[vb].locked = true;
        }
    }

    // ── Lock UV-seam vertices ─────────────────────────────────────────────────
    // A UV seam has multiple input vertices at the same 3-D position (one per
    // UV shell edge).  If QEM moves one of them but not the others, a visible
    // crack appears.  Locking every vertex whose position is shared by another
    // vertex prevents the split — the seam stays geometrically intact.
    {
        std::map<std::tuple<float,float,float>, int> pos_first;
        for (int i = 0; i < nv; ++i) {
            auto key = std::make_tuple(
                in_verts[i].position.x,
                in_verts[i].position.y,
                in_verts[i].position.z);
            auto [it, ins] = pos_first.emplace(key, i);
            if (!ins) {
                // Duplicate position — lock both this vertex and the first seen
                verts[i].locked        = true;
                verts[it->second].locked = true;
            }
        }
    }

    // ── Per-vertex quadrics from face planes ──────────────────────────────────
    for (int fi = 0; fi < nf; ++fi) {
        const auto& f = faces[fi];
        double a, b, c, d;
        facePlane(verts[f.v[0]].pos, verts[f.v[1]].pos, verts[f.v[2]].pos, a,b,c,d);
        for (int k = 0; k < 3; ++k)
            verts[f.v[k]].Q.addPlane(a,b,c,d);
    }

    // ── Build edge neighbor lists + initial priority queue ────────────────────
    std::unordered_map<uint64_t,int> edge_ver;
    edge_ver.reserve((size_t)nf * 3);
    std::priority_queue<CollapseEntry,
                        std::vector<CollapseEntry>,
                        std::greater<CollapseEntry>> pq;

    std::unordered_set<uint64_t> seen;
    seen.reserve((size_t)nf * 3);
    for (int fi = 0; fi < nf; ++fi)
        for (int k = 0; k < 3; ++k) {
            int a = faces[fi].v[k], b = faces[fi].v[(k+1)%3];
            uint64_t key = edgeKey(a,b);
            if (!seen.insert(key).second) continue;
            verts[a].adj_verts.push_back(b);
            verts[b].adj_verts.push_back(a);
            if (verts[a].locked && verts[b].locked) continue; // skip fully-locked
            edge_ver[key] = 0;
            pq.push(computeCollapse(verts, std::min(a,b), std::max(a,b), edge_ver));
        }

    // ── Main collapse loop ────────────────────────────────────────────────────
    int alive         = nf;
    int collapses_done = 0;

    while (alive > (int)target_face_count && !pq.empty()) {
        auto entry = pq.top(); pq.pop();
        int v0 = entry.v0, v1 = entry.v1;

        // Stale check
        {
            auto it = edge_ver.find(edgeKey(v0,v1));
            if (it == edge_ver.end() || it->second != entry.version) continue;
        }
        if (verts[v0].deleted || verts[v1].deleted)   continue;
        if (verts[v0].locked  && verts[v1].locked)    continue;

        // Snap to locked endpoint if needed
        glm::dvec3 new_pos = entry.new_pos;
        if      (verts[v0].locked) new_pos = verts[v0].pos;
        else if (verts[v1].locked) new_pos = verts[v1].pos;

        // ── Geometry-quality safety check ────────────────────────────────────
        // Returns true if moving vi_move to new_pos would degrade any surviving
        // face (one that does NOT also contain vi_other, those will be deleted).
        // Three rejection criteria:
        //   1. Normal flip       – prevents inside-out faces
        //   2. Near-zero area    – prevents degenerate / zero-area faces
        //   3. High aspect ratio – prevents stretched needle triangles
        //      threshold: max_edge / min_altitude > k  (k = 20 means 20:1 ratio)
        static constexpr double kStretchSq = 20.0; // max_edge² / (2·area) > k

        auto wouldDegrade = [&](int vi_move, int vi_other) -> bool {
            for (int fi : verts[vi_move].adj_faces) {
                if (faces[fi].deleted) continue;
                const auto& f = faces[fi];
                bool has_other = (f.v[0]==vi_other||f.v[1]==vi_other||f.v[2]==vi_other);
                if (has_other) continue;

                glm::dvec3 p[3] = {verts[f.v[0]].pos, verts[f.v[1]].pos, verts[f.v[2]].pos};
                glm::dvec3 old_n = glm::cross(p[1]-p[0], p[2]-p[0]);
                for (int k = 0; k < 3; ++k) if (f.v[k] == vi_move) p[k] = new_pos;
                glm::dvec3 new_n = glm::cross(p[1]-p[0], p[2]-p[0]);

                // 2. Near-zero area → degenerate triangle
                double area2 = glm::length(new_n); // = 2 * area
                if (area2 < 1e-12) return true;

                // 1. Normal flip (skip if old face was already degenerate)
                double old_area2 = glm::length(old_n);
                if (old_area2 > 1e-12 && glm::dot(old_n, new_n) < 0.0) return true;

                // 3. Aspect ratio: max_edge² / (2·area) > threshold
                double e0sq = glm::dot(p[1]-p[0], p[1]-p[0]);
                double e1sq = glm::dot(p[2]-p[1], p[2]-p[1]);
                double e2sq = glm::dot(p[0]-p[2], p[0]-p[2]);
                double max_esq = std::max({e0sq, e1sq, e2sq});
                if (max_esq > area2 * kStretchSq) return true;
            }
            return false;
        };

        // ── Link-condition check ──────────────────────────────────────────────
        // The number of vertices adjacent to BOTH v0 and v1 must equal exactly
        // the number of faces shared by edge (v0,v1).  A higher count means the
        // collapse would weld together unrelated surface sheets at a single vertex
        // ("bow-tie" / non-manifold), which renders as severe diagonal streaks.
        {
            // Count faces that contain both v0 and v1 (these will be deleted)
            int shared_faces = 0;
            for (int fi : verts[v0].adj_faces) {
                if (faces[fi].deleted) continue;
                const auto& f = faces[fi];
                if (f.v[0]==v1 || f.v[1]==v1 || f.v[2]==v1) ++shared_faces;
            }

            // Collect v0's live unique neighbors (excluding v1)
            std::unordered_set<int> v0_nbrs;
            v0_nbrs.reserve(verts[v0].adj_verts.size());
            for (int vn : verts[v0].adj_verts)
                if (!verts[vn].deleted && vn != v1) v0_nbrs.insert(vn);

            // Count how many of v1's live neighbors are also v0's neighbors
            std::unordered_set<int> v1_seen;
            int shared_nbrs = 0;
            for (int vn : verts[v1].adj_verts) {
                if (verts[vn].deleted || vn == v0) continue;
                if (v1_seen.insert(vn).second && v0_nbrs.count(vn)) ++shared_nbrs;
            }

            if (shared_nbrs != shared_faces) continue; // would create non-manifold
        }

        if (wouldDegrade(v1, v0)) continue;
        if (glm::length(new_pos - verts[v0].pos) > 1e-12 && wouldDegrade(v0, v1)) continue;

        // ── Execute collapse: merge v1 into v0 ───────────────────────────────
        verts[v0].pos      = new_pos;
        verts[v0].Q       += verts[v1].Q;
        verts[v1].deleted  = true;

        for (int fi : verts[v1].adj_faces) {
            if (faces[fi].deleted) continue;
            auto& f = faces[fi];
            bool has_v0 = (f.v[0]==v0||f.v[1]==v0||f.v[2]==v0);
            if (has_v0) {
                faces[fi].deleted = true; --alive;
            } else {
                for (int k=0;k<3;++k) if (f.v[k]==v1) f.v[k]=v0;
                verts[v0].adj_faces.push_back(fi);
            }
        }
        verts[v1].adj_faces.clear();

        // Merge v1's neighbor list into v0 (lazy — keeps duplicates, skip on use)
        for (int vn : verts[v1].adj_verts) {
            if (vn == v0 || verts[vn].deleted) continue;
            verts[v0].adj_verts.push_back(vn);
            for (auto& nb : verts[vn].adj_verts) if (nb == v1) nb = v0;
        }
        verts[v1].adj_verts.clear();

        // Remove the collapsed edge
        edge_ver.erase(edgeKey(v0,v1));

        // Recompute all edges incident to v0
        for (int vn : verts[v0].adj_verts) {
            if (verts[vn].deleted || vn == v0) continue;
            if (verts[v0].locked && verts[vn].locked) continue;
            uint64_t key = edgeKey(v0,vn);
            ++edge_ver[key];
            pq.push(computeCollapse(verts, std::min(v0,vn), std::max(v0,vn), edge_ver));
        }
        ++collapses_done;
    }

    log << "[QEM] collapses=" << collapses_done
        << "  faces: " << nf << " -> " << alive << "\n";

    // ── Build output mesh ─────────────────────────────────────────────────────
    std::vector<int> old_to_new(nv, -1);
    output_mesh.vertex_data_ptr->clear();
    output_mesh.faces_ptr->clear();
    output_face_part_ids.clear();

    // First: assign output vertex indices (only for alive faces)
    for (int fi = 0; fi < nf; ++fi) {
        if (faces[fi].deleted) continue;
        for (int k = 0; k < 3; ++k) {
            int vi = faces[fi].v[k];
            if (old_to_new[vi] == -1) {
                old_to_new[vi] = (int)output_mesh.vertex_data_ptr->size();
                VertexStruct vs   = in_verts[vi];       // copy UV + normal
                vs.position       = glm::vec3(verts[vi].pos); // QEM position
                output_mesh.vertex_data_ptr->push_back(vs);
            }
        }
    }
    // Second: emit faces
    for (int fi = 0; fi < nf; ++fi) {
        if (faces[fi].deleted) continue;
        output_mesh.faces_ptr->push_back(Face(
            (uint32_t)old_to_new[faces[fi].v[0]],
            (uint32_t)old_to_new[faces[fi].v[1]],
            (uint32_t)old_to_new[faces[fi].v[2]]));
        output_face_part_ids.push_back(faces[fi].part_id);
    }
}

} // game_object
} // engine