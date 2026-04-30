#pragma once
#include <string>
#include <vector>
#include <array>
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
    std::vector<ScopeDisplay> scopes;
    float total_ms = 0.0f;  // sum of depth-0 scopes
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

    // Call AFTER the GPU has finished the frame (typically with a 1-frame
    // delay to avoid stalls).  frame_index is the frame-in-flight slot.
    void collectResults(
        const std::shared_ptr<renderer::Device>& device,
        uint32_t frame_index);

    // ImGui flame-chart timeline display.
    void drawImGui();

    void destroy(const std::shared_ptr<renderer::Device>& device);

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
    };

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
