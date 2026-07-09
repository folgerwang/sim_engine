#pragma once
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <memory>
#include "renderer/renderer.h"

namespace engine {
namespace helper {

// ---------------------------------------------------------------------------
//  GpuProfiler – Vulkan timestamp-based GPU frame profiler with a
//  Nsight-style interactive flame-chart timeline.
//
//  Features:
//    • Multi-frame ring-buffer history (128 frames)
//    • Flame-chart lanes ordered by nesting depth
//    • Minimap strip showing all frames colour-coded by total GPU time
//    • Mouse-scroll wheel zoom, left-drag pan
//    • Auto-pause on interaction so you can inspect any frame
//    • Adaptive time ruler, hover tooltips, "RESUME" button
//
//  Usage:
//    // Init once
//    profiler.init(device, frames_in_flight, max_scopes_per_frame);
//
//    // Each frame:
//    profiler.beginFrame(cmd_buf, frame_index);
//    {
//        auto s = profiler.beginScope(cmd_buf, "Shadow Pass");
//        // ... draw shadow ...
//        profiler.endScope(cmd_buf, s);
//
//        auto s2 = profiler.beginScope(cmd_buf, "Forward Pass");
//        {
//            auto s3 = profiler.beginScope(cmd_buf, "Terrain");
//            profiler.endScope(cmd_buf, s3);
//        }
//        profiler.endScope(cmd_buf, s2);
//    }
//    profiler.endFrame(cmd_buf, frame_index);
//
//    // After GPU finishes (typically 1-frame delay):
//    profiler.collectResults(device, prev_frame_index);
//
//    // Render UI (call inside an ImGui frame):
//    profiler.drawImGui();
// ---------------------------------------------------------------------------

// One recorded GPU scope (stored after result collection).
struct ScopeDisplay {
    std::string name;
    float       begin_ms = 0.0f;  // relative to frame-start tick
    float       end_ms   = 0.0f;
    int         depth    = 0;
    uint32_t    color    = 0;     // ABGR packed ImGui color
};

// One complete frame of collected GPU data.
struct FrameRecord {
    std::vector<ScopeDisplay> scopes;       // GPU scopes (relative to GPU frame-start tick)
    std::vector<ScopeDisplay> cpu_scopes;   // CPU scopes (relative to CPU frame-start time)
    float total_ms = 0.0f;       // sum of depth-0 GPU scopes
    float total_cpu_ms = 0.0f;   // sum of depth-0 CPU scopes
};

class GpuProfiler {
public:
    static constexpr int   kHistorySize      = 128;   // frames of ring history
    static constexpr int   kMaxScopesPerFrame = 64;   // max named scopes per frame
    static constexpr float kFrameGapMs       = 0.3f;  // visual gap between frames (ms)

    GpuProfiler() = default;
    ~GpuProfiler() = default;

    // Call once after device creation.
    void init(
        const std::shared_ptr<renderer::Device>& device,
        uint32_t frames_in_flight,
        uint32_t max_scopes = kMaxScopesPerFrame);

    // Call at the very start of each frame's command buffer recording.
    // Writes a frame-start TOP_OF_PIPE timestamp at query slot 0.
    void beginFrame(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        uint32_t frame_index);

    // Returns an opaque scope handle.  Pass it to endScope().
    // Returns UINT32_MAX if capacity exceeded.
    uint32_t beginScope(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const char* name);

    void endScope(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        uint32_t scope_handle);

    // Call at the end of each frame's command buffer recording.
    void endFrame(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        uint32_t frame_index);

    // ── CPU scope API (parallel to the GPU one) ────────────────────
    // Uses std::chrono::high_resolution_clock; no command buffer or
    // query pool needed since the timing is host-side.  Call
    // beginCpuFrame at the very start of frame N's CPU work, then
    // wrap any CPU phase in begin/endCpuScope, then endCpuFrame
    // before the application moves on to the next frame.  The
    // recorded CPU scopes are stashed into the FrameState's slot
    // for `frame_index` and merged into the FrameRecord by the
    // matching call to collectResults — so CPU lanes line up with
    // GPU lanes for the same logical frame in the timeline UI.
    void beginCpuFrame(uint32_t frame_index);
    uint32_t beginCpuScope(const char* name);
    void endCpuScope(uint32_t scope_handle);
    void endCpuFrame(uint32_t frame_index);

    // Call AFTER the GPU has finished the frame (typically with a 1-frame
    // delay to avoid stalls).  frame_index is the frame-in-flight slot.
    void collectResults(
        const std::shared_ptr<renderer::Device>& device,
        uint32_t frame_index);

    // ImGui flame-chart timeline display.
    void drawImGui();

    void destroy(const std::shared_ptr<renderer::Device>& device);

    // Window visibility: the ImGui window's [X] close button clears
    // m_show_window_ via Begin's p_open; the menu re-opens it with
    // setWindowOpen(true) and mirrors windowOpen() back into its own
    // checkmark so the two stay in sync.
    bool windowOpen() const { return m_show_window_; }
    void setWindowOpen(bool v) { m_show_window_ = v; }

    // Toggle/set pause state of the flame-chart display. When paused, the
    // timeline stops auto-scrolling so you can inspect any frame.
    void togglePause() { m_paused_ = !m_paused_; }
    void setPaused(bool p) { m_paused_ = p; }
    bool isPaused() const { return m_paused_; }

    // Returns a pointer to the most recently collected FrameRecord, or nullptr
    // if no frame has been collected yet.  The pointer is valid until the next
    // call to collectResults() (ring-buffer slot may be reused after 128 frames).
    const FrameRecord* getLatestFrame() const {
        if (m_frame_count_ <= 0) return nullptr;
        int idx = (m_frame_write_idx_ - 1 + kHistorySize) % kHistorySize;
        return &m_frames_[idx];
    }

    // ── Combined CPU + GPU scope (RAII) ─────────────────────────────
    // Records a CPU scope AND a GPU scope with the same name so the
    // profiler shows two bars at the same horizontal position — the
    // CPU bar measures recording time, the GPU bar measures execution
    // time.  Use:
    //   {
    //     auto _t = gpu_profiler_.scope(cmd_buf, "My Pass");
    //     // ... record + execute work ...
    //   }   // scope auto-closes both
    class Scope {
    public:
        Scope(GpuProfiler& p, const std::shared_ptr<renderer::CommandBuffer>& cb,
              const char* name)
            : p_(p), cb_(cb) {
            cpu_h_ = p.beginCpuScope(name);
            gpu_h_ = p.beginScope(cb, name);
        }
        ~Scope() {
            p_.endScope(cb_, gpu_h_);
            p_.endCpuScope(cpu_h_);
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        // No move — RAII end is bound to lexical scope.
        Scope(Scope&&) = delete;
        Scope& operator=(Scope&&) = delete;
    private:
        GpuProfiler& p_;
        std::shared_ptr<renderer::CommandBuffer> cb_;
        uint32_t cpu_h_ = UINT32_MAX;
        uint32_t gpu_h_ = UINT32_MAX;
    };

private:
    // ----- Recording state (per frame-in-flight slot) ----------------------

    struct FrameState {
        std::shared_ptr<renderer::QueryPool> query_pool;

        struct ScopeEntry {
            std::string name;
            int         depth       = 0;
            uint32_t    begin_query = 0;  // index into query pool
            uint32_t    end_query   = 0;
        };
        std::vector<ScopeEntry> recorded_scopes;
        uint32_t                next_query = 0;  // slot 0 = frame-start marker
        int                     open_depth = 0;
        bool                    active     = false;

        // ── CPU scopes (recorded host-side via chrono) ──────────────
        struct CpuScopeEntry {
            std::string name;
            int  depth = 0;
            std::chrono::high_resolution_clock::time_point begin{};
            std::chrono::high_resolution_clock::time_point end{};
        };
        std::vector<CpuScopeEntry> cpu_recorded;
        int  cpu_open_depth = 0;
        bool cpu_active     = false;
        std::chrono::high_resolution_clock::time_point cpu_frame_start{};

        // ── Completed snapshot, read by collectResults ──────────────
        // endCpuFrame moves cpu_recorded into cpu_completed (and the
        // matching cpu_frame_start into cpu_completed_frame_start),
        // so collectResults can read FULLY CLOSED scopes from the
        // previous use of this slot.  Without this, collectResults
        // (called early in the frame, after the fence wait) saw the
        // CURRENT frame's in-progress cpu_recorded — only the very
        // first sub-scope that had finished by that point was readable;
        // the still-open parent scopes ("drawFrame", "Fence Wait +
        // Acquire") had end == begin and rendered as zero-width bars.
        std::vector<CpuScopeEntry> cpu_completed;
        std::chrono::high_resolution_clock::time_point cpu_completed_frame_start{};
    };

    // Frame-in-flight slot currently being CPU-recorded.  Set by
    // beginCpuFrame, used by begin/endCpuScope so the caller doesn't
    // have to thread a frame_index through every scope call.
    uint32_t m_cpu_active_frame_idx_ = 0;

    uint32_t m_frames_in_flight_ = 0;
    uint32_t m_max_scopes_       = kMaxScopesPerFrame;
    float    m_timestamp_period_ = 1.0f;  // ns per tick

    std::vector<FrameState> m_frame_states_;

    // ----- Collected history -----------------------------------------------

    std::array<FrameRecord, kHistorySize> m_frames_{};
    int m_frame_write_idx_ = 0;   // next slot to write (ring)
    int m_frame_count_     = 0;   // how many valid frames in ring

    // Global timeline X-position (in ms) for the *start* of each ring slot.
    // Recomputed after every collectResults call.
    std::array<float, kHistorySize> m_frame_global_start_{};

    void recomputeGlobalPositions();

    // ----- View state (for the flame-chart UI) -----------------------------

    float m_view_start_ms_  = 0.0f;   // left edge of visible region (ms)
    float m_view_range_ms_  = 50.0f;  // visible time span (ms)
    bool  m_paused_         = false;  // true = frozen, no auto-scroll
    bool  m_show_window_    = true;

    // ----- Helpers ---------------------------------------------------------

    // Return a stable ABGR colour for a given scope name.
    static uint32_t colorForName(const std::string& name);

    // Linear interpolate between two ABGR colours.
    static uint32_t lerpColor(uint32_t a, uint32_t b, float t);

    // Map a GPU time (ms, relative to its frame-start) to canvas X pixel.
    // Defined inline here for use in drawImGui.
    float timeToX(float global_ms, float canvas_x, float canvas_w) const {
        return canvas_x + (global_ms - m_view_start_ms_) / m_view_range_ms_ * canvas_w;
    }
};

} // namespace helper
} // namespace engine
