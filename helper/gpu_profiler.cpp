// ---------------------------------------------------------------------------
//  windows.h MUST come before anything that may pull in GLFW (via the
//  renderer headers) to avoid APIENTRY redefinition, and it MUST have
//  NOMINMAX defined so its min/max macros don't poison std::min / std::max.
// ---------------------------------------------------------------------------
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>    // OutputDebugStringA
#endif

#include "gpu_profiler.h"
#include "imgui.h"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdarg>

// ---------------------------------------------------------------------------
//  GPU_PROF_DEBUG    – 1 = verbose tracing to stderr + debugger output
//  GPU_PROF_ASSERT   – light assertions that log instead of aborting
// ---------------------------------------------------------------------------
// Default to silent — the trace path emits per-frame messages from
// collectResults / drawImGui / zoom handlers, which floods the console
// once a build is up and stable.  Flip to 1 only when actively
// debugging the profiler internals.
#ifndef GPU_PROF_DEBUG
#  define GPU_PROF_DEBUG 0
#endif

namespace engine {
namespace helper {

// ---- Tracing helpers ------------------------------------------------------
static void gpu_prof_trace(const char* fmt, ...) {
#if GPU_PROF_DEBUG
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    buf[sizeof(buf) - 1] = '\0';
    fprintf(stderr, "[GpuProfiler] %s\n", buf);
#  ifdef _WIN32
    char dbuf[600];
    snprintf(dbuf, sizeof(dbuf), "[GpuProfiler] %s\n", buf);
    OutputDebugStringA(dbuf);
#  endif
#else
    (void)fmt;
#endif
}

#define GPU_PROF_CHECK(cond, ...) \
    do { if (!(cond)) { gpu_prof_trace("CHECK FAILED: " #cond " @ %s:%d -- " __VA_ARGS__, __FILE__, __LINE__); return; } } while (0)

#define GPU_PROF_CHECK_RET(cond, retv, ...) \
    do { if (!(cond)) { gpu_prof_trace("CHECK FAILED: " #cond " @ %s:%d -- " __VA_ARGS__, __FILE__, __LINE__); return retv; } } while (0)

static inline bool gp_is_finite(float v) {
    return std::isfinite(v);
}

// Clamp a pixel coordinate to a sane range to prevent ImGui vertex overflow.
static inline float gp_sanitize_px(float v, float fallback) {
    if (!std::isfinite(v)) return fallback;
    if (v >  1e6f) return  1e6f;
    if (v < -1e6f) return -1e6f;
    return v;
}


// ============================================================================
//  Init / Destroy
// ============================================================================

void GpuProfiler::init(
    const std::shared_ptr<renderer::Device>& device,
    uint32_t frames_in_flight,
    uint32_t max_scopes)
{
    m_frames_in_flight_ = frames_in_flight;
    m_max_scopes_       = max_scopes;
    m_timestamp_period_ = device->getTimestampPeriod();

    // Query layout per frame:
    //   slot 0          : frame-start TOP_OF_PIPE timestamp
    //   slots 1..2*N    : scope begin/end pairs (N = max_scopes)
    const uint32_t queries_per_frame = 1 + max_scopes * 2;

    m_frame_states_.resize(frames_in_flight);
    for (auto& fs : m_frame_states_) {
        fs.query_pool = device->createQueryPool(queries_per_frame);
        fs.recorded_scopes.reserve(max_scopes);
        fs.active = false;
    }

    m_frame_write_idx_ = 0;
    m_frame_count_     = 0;
    m_frames_          = {};
    m_frame_global_start_.fill(0.0f);

    gpu_prof_trace("init: frames_in_flight=%u max_scopes=%u ts_period=%.3f ns/tick",
        frames_in_flight, max_scopes, m_timestamp_period_);
}

void GpuProfiler::destroy(const std::shared_ptr<renderer::Device>& device)
{
    gpu_prof_trace("destroy: releasing %zu frame states", m_frame_states_.size());

    for (auto& fs : m_frame_states_) {
        if (fs.query_pool) {
            device->destroyQueryPool(fs.query_pool);
            fs.query_pool.reset();
        }
        fs.recorded_scopes.clear();
        fs.active = false;
    }
    m_frame_states_.clear();

    // Drop any collected data so a stray drawImGui() after destroy becomes
    // a no-op (instead of dereferencing freed query pools through pointers
    // that never leave this class, but still: belt & braces).
    m_frame_count_     = 0;
    m_frame_write_idx_ = 0;
    for (auto& f : m_frames_) f.scopes.clear();
}

// ============================================================================
//  Per-frame recording
// ============================================================================

void GpuProfiler::beginFrame(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    uint32_t frame_index)
{
    if (m_frame_states_.empty() || m_frames_in_flight_ == 0) return;
    auto& fs = m_frame_states_[frame_index % m_frames_in_flight_];
    fs.recorded_scopes.clear();
    fs.next_query = 0;
    fs.open_depth = 0;
    fs.active     = true;

    const uint32_t queries_per_frame = 1 + m_max_scopes_ * 2;
    cmd_buf->resetQueryPool(fs.query_pool, 0, queries_per_frame);

    // Slot 0: frame-start marker.
    cmd_buf->writeTimestamp(fs.query_pool, fs.next_query++, /*after=*/false);
}

uint32_t GpuProfiler::beginScope(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const char* name)
{
    FrameState* fs = nullptr;
    for (auto& s : m_frame_states_) {
        if (s.active) { fs = &s; break; }
    }
    if (!fs) return UINT32_MAX;
    if (fs->next_query + 2 > 1 + m_max_scopes_ * 2) return UINT32_MAX;

    uint32_t scope_idx = static_cast<uint32_t>(fs->recorded_scopes.size());

    FrameState::ScopeEntry entry;
    entry.name        = name;
    entry.depth       = fs->open_depth;
    entry.begin_query = fs->next_query++;
    entry.end_query   = 0;
    fs->recorded_scopes.push_back(std::move(entry));

    cmd_buf->writeTimestamp(
        fs->query_pool,
        fs->recorded_scopes.back().begin_query,
        /*after=*/false);

    fs->open_depth++;
    return scope_idx;
}

void GpuProfiler::endScope(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    uint32_t scope_handle)
{
    if (scope_handle == UINT32_MAX) return;

    FrameState* fs = nullptr;
    for (auto& s : m_frame_states_) {
        if (s.active) { fs = &s; break; }
    }
    if (!fs || scope_handle >= fs->recorded_scopes.size()) return;

    auto& entry     = fs->recorded_scopes[scope_handle];
    entry.end_query = fs->next_query++;
    fs->open_depth  = std::max(0, fs->open_depth - 1);

    cmd_buf->writeTimestamp(fs->query_pool, entry.end_query, /*after=*/true);
}

void GpuProfiler::endFrame(
    const std::shared_ptr<renderer::CommandBuffer>& /*cmd_buf*/,
    uint32_t /*frame_index*/)
{
    for (auto& fs : m_frame_states_) {
        if (fs.active) { fs.active = false; break; }
    }
}

// ============================================================================
//  CPU scope API — chrono-based, parallel to the GPU one
// ============================================================================

void GpuProfiler::beginCpuFrame(uint32_t frame_index)
{
    if (m_frame_states_.empty() || m_frames_in_flight_ == 0) return;
    auto& fs = m_frame_states_[frame_index % m_frames_in_flight_];
    fs.cpu_recorded.clear();
    fs.cpu_open_depth   = 0;
    fs.cpu_active       = true;
    fs.cpu_frame_start  = std::chrono::high_resolution_clock::now();
    m_cpu_active_frame_idx_ = frame_index;
}

uint32_t GpuProfiler::beginCpuScope(const char* name)
{
    if (m_frame_states_.empty() || m_frames_in_flight_ == 0) return UINT32_MAX;
    auto& fs = m_frame_states_[m_cpu_active_frame_idx_ % m_frames_in_flight_];
    if (!fs.cpu_active) return UINT32_MAX;
    if (fs.cpu_recorded.size() >= m_max_scopes_) return UINT32_MAX;

    uint32_t scope_idx = static_cast<uint32_t>(fs.cpu_recorded.size());
    FrameState::CpuScopeEntry entry;
    entry.name  = name ? name : "<null>";
    entry.depth = fs.cpu_open_depth;
    entry.begin = std::chrono::high_resolution_clock::now();
    entry.end   = entry.begin;  // placeholder
    fs.cpu_recorded.push_back(std::move(entry));
    fs.cpu_open_depth++;
    return scope_idx;
}

void GpuProfiler::endCpuScope(uint32_t scope_handle)
{
    if (scope_handle == UINT32_MAX) return;
    if (m_frame_states_.empty() || m_frames_in_flight_ == 0) return;
    auto& fs = m_frame_states_[m_cpu_active_frame_idx_ % m_frames_in_flight_];
    if (!fs.cpu_active || scope_handle >= fs.cpu_recorded.size()) return;

    fs.cpu_recorded[scope_handle].end =
        std::chrono::high_resolution_clock::now();
    fs.cpu_open_depth = std::max(0, fs.cpu_open_depth - 1);
}

void GpuProfiler::endCpuFrame(uint32_t frame_index)
{
    if (m_frame_states_.empty() || m_frames_in_flight_ == 0) return;
    auto& fs = m_frame_states_[frame_index % m_frames_in_flight_];
    fs.cpu_active = false;

    // Snapshot the FULLY-CLOSED recording into cpu_completed so the
    // next time this slot is hit by beginCpuFrame (FIF cycles from
    // now), collectResults can read this complete frame's data
    // even though cpu_recorded is about to be cleared/reused.
    //
    // Move (not copy) — the entries contain std::strings that we'd
    // rather not allocate twice per frame.  cpu_recorded is left
    // empty and ready for the next beginCpuFrame to start fresh.
    fs.cpu_completed             = std::move(fs.cpu_recorded);
    fs.cpu_completed_frame_start = fs.cpu_frame_start;
    fs.cpu_recorded.clear();   // move-from leaves it valid-but-unspecified
}

// ============================================================================
//  Result collection
// ============================================================================

void GpuProfiler::collectResults(
    const std::shared_ptr<renderer::Device>& device,
    uint32_t frame_index)
{
    if (m_frame_states_.empty() || m_frames_in_flight_ == 0) return;

    // When paused, stop ingesting new frames so the flame chart freezes.
    if (m_paused_) return;

    auto& fs = m_frame_states_[frame_index % m_frames_in_flight_];
    if (fs.recorded_scopes.empty()) return;

    // Queries used = 1 (frame-start) + scope_count * 2
    const uint32_t total_queries =
        1 + static_cast<uint32_t>(fs.recorded_scopes.size()) * 2;

    std::vector<uint64_t> raw;
    bool ready = device->getQueryPoolResults(fs.query_pool, 0, total_queries, raw);
    if (!ready) return;

    const float ms_factor = m_timestamp_period_ * 1e-6f;
    const uint64_t frame_start_tick = raw[0];

    FrameRecord rec;
    rec.scopes.reserve(fs.recorded_scopes.size());

    for (auto& entry : fs.recorded_scopes) {
        ScopeDisplay sd;
        sd.name  = entry.name;
        sd.depth = entry.depth;
        sd.color = colorForName(entry.name);

        if (entry.begin_query < raw.size() && entry.end_query < raw.size()) {
            sd.begin_ms = static_cast<float>(
                raw[entry.begin_query] - frame_start_tick) * ms_factor;
            sd.end_ms   = static_cast<float>(
                raw[entry.end_query]   - frame_start_tick) * ms_factor;
        }
        rec.scopes.push_back(sd);
    }

    // Total from depth-0 scopes.
    for (auto& sd : rec.scopes) {
        if (sd.depth == 0) rec.total_ms += (sd.end_ms - sd.begin_ms);
    }

    // ── Merge CPU scopes for the same frame slot ─────────────────────
    // Read from cpu_completed (snapshot taken by endCpuFrame at the
    // end of the PREVIOUS use of this slot, FIF frames ago).  This
    // is the matched-pair to the GPU queries we just ingested, since
    // both the GPU's frame-N-FIF queries and the CPU's frame-N-FIF
    // scopes ended up parked in this slot until we got around to
    // reading them.  Reading cpu_recorded (the live one) would race
    // the current frame's ongoing scopes — half open, half closed.
    rec.cpu_scopes.reserve(fs.cpu_completed.size());
    for (auto& cpu : fs.cpu_completed) {
        ScopeDisplay sd;
        sd.name  = cpu.name;
        sd.depth = cpu.depth;
        sd.color = colorForName(cpu.name);
        const auto begin_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                cpu.begin - fs.cpu_completed_frame_start).count();
        const auto end_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                cpu.end   - fs.cpu_completed_frame_start).count();
        sd.begin_ms = float(begin_ns) * 1e-6f;
        sd.end_ms   = float(end_ns)   * 1e-6f;
        rec.cpu_scopes.push_back(sd);
    }
    for (auto& sd : rec.cpu_scopes) {
        if (sd.depth == 0) rec.total_cpu_ms += (sd.end_ms - sd.begin_ms);
    }

    // Sanity-check before committing to the ring buffer: reject any
    // frame whose timestamps produced NaN/inf or negative durations so a
    // single bad sample cannot blow up the renderer.
    bool frame_ok = std::isfinite(rec.total_ms) && rec.total_ms >= 0.0f;
    if (frame_ok) {
        for (auto& sd : rec.scopes) {
            if (!std::isfinite(sd.begin_ms) || !std::isfinite(sd.end_ms) ||
                sd.end_ms < sd.begin_ms) {
                frame_ok = false;
                break;
            }
        }
    }
    if (!frame_ok) {
        gpu_prof_trace("collectResults: rejected corrupt frame (total=%.3f, scopes=%zu)",
            rec.total_ms, rec.scopes.size());
        return;
    }

    m_frames_[m_frame_write_idx_] = std::move(rec);
    m_frame_write_idx_ = (m_frame_write_idx_ + 1) % kHistorySize;
    if (m_frame_count_ < kHistorySize) ++m_frame_count_;

    recomputeGlobalPositions();

#if GPU_PROF_DEBUG
    static int s_collect_counter = 0;
    if ((++s_collect_counter % 60) == 0) {
        int latest = (m_frame_write_idx_ - 1 + kHistorySize) % kHistorySize;
        gpu_prof_trace("collect: write_idx=%d count=%d latest_total=%.3fms scopes=%zu view=[%.2f,+%.2f]ms paused=%d",
            m_frame_write_idx_, m_frame_count_,
            m_frames_[latest].total_ms, m_frames_[latest].scopes.size(),
            m_view_start_ms_, m_view_range_ms_, (int)m_paused_);
    }
#endif

    // Auto-scroll to the latest frame unless paused.
    if (!m_paused_ && m_frame_count_ > 0) {
        // Find the last valid frame's global end.
        float total_span = 0.0f;
        for (int i = 0; i < m_frame_count_; ++i) {
            int slot = (m_frame_write_idx_ - 1 - i + kHistorySize) % kHistorySize;
            float frame_end = m_frame_global_start_[slot] +
                              std::max(m_frames_[slot].total_ms,
                                       m_frames_[slot].total_cpu_ms);
            if (frame_end > total_span) total_span = frame_end;
        }
        // Keep the right edge of the view at the end of the last frame.
        m_view_start_ms_ = total_span - m_view_range_ms_;
        if (m_view_start_ms_ < 0.0f) m_view_start_ms_ = 0.0f;
    }
}

void GpuProfiler::recomputeGlobalPositions()
{
    // Walk the ring in chronological order: oldest → newest.
    // The "oldest" slot is m_frame_write_idx_ when the ring is full,
    // or 0 when it isn't.
    float cursor = 0.0f;
    const int count = m_frame_count_;
    const int base  = (m_frame_count_ < kHistorySize)
                    ? 0
                    : m_frame_write_idx_;

    for (int i = 0; i < count; ++i) {
        int slot = (base + i) % kHistorySize;
        m_frame_global_start_[slot] = cursor;
        // Frame width is the max of CPU and GPU totals — otherwise
        // the longer side overflows into the next frame slot in the
        // timeline.  Both stacks share the same global X axis so that
        // the user sees CPU and GPU bars for the same frame stacked
        // vertically with consistent horizontal extents.
        float dt = std::max(m_frames_[slot].total_ms,
                            m_frames_[slot].total_cpu_ms);
        if (!std::isfinite(dt) || dt < 0.0f) dt = 0.0f;
        cursor += dt + kFrameGapMs;
    }
}

// ============================================================================
//  Colour helpers
// ============================================================================

uint32_t GpuProfiler::colorForName(const std::string& name)
{
    // Hash the name to pick a hue; keep saturation and value fixed.
    uint32_t h = 2166136261u;
    for (char c : name) {
        h ^= static_cast<uint8_t>(c);
        h *= 16777619u;
    }

    // Map hash → hue [0, 360)
    float hue = static_cast<float>(h % 360);
    float s   = 0.75f;
    float v   = 0.90f;

    // HSV → RGB
    float c2 = v * s;
    float x  = c2 * (1.0f - fabsf(fmodf(hue / 60.0f, 2.0f) - 1.0f));
    float m  = v - c2;

    float r, g, b;
    int   sector = static_cast<int>(hue / 60.0f) % 6;
    switch (sector) {
        case 0:  r = c2; g = x;  b = 0;  break;
        case 1:  r = x;  g = c2; b = 0;  break;
        case 2:  r = 0;  g = c2; b = x;  break;
        case 3:  r = 0;  g = x;  b = c2; break;
        case 4:  r = x;  g = 0;  b = c2; break;
        default: r = c2; g = 0;  b = x;  break;
    }

    auto to8 = [](float f) -> uint32_t {
        return static_cast<uint32_t>(
            std::min(std::max(f, 0.0f), 1.0f) * 255.0f + 0.5f);
    };

    uint32_t ri = to8(r + m);
    uint32_t gi = to8(g + m);
    uint32_t bi = to8(b + m);
    // ImGui IM_COL32 format: ABGR
    return 0xFF000000u | (bi << 16) | (gi << 8) | ri;
}

uint32_t GpuProfiler::lerpColor(uint32_t a, uint32_t b, float t)
{
    auto lerp8 = [&](int shift) -> uint32_t {
        uint32_t ca = (a >> shift) & 0xFF;
        uint32_t cb = (b >> shift) & 0xFF;
        return static_cast<uint32_t>(ca + (cb - (int)ca) * t) & 0xFF;
    };
    return (lerp8(24) << 24) | (lerp8(16) << 16) | (lerp8(8) << 8) | lerp8(0);
}

// ============================================================================
//  ImGui flame-chart draw
// ============================================================================

void GpuProfiler::drawImGui()
{
#if GPU_PROF_DEBUG
    static int s_draw_counter = 0;
    ++s_draw_counter;
    const int draw_id = s_draw_counter;
    gpu_prof_trace("drawImGui ENTER #%d  view=[%.3f, +%.3f]ms paused=%d frames=%d",
        draw_id, m_view_start_ms_, m_view_range_ms_,
        (int)m_paused_, m_frame_count_);
#endif

    // Respect the window's close button: ImGui::Begin's p_open writes
    // m_show_window_ = false when the user clicks [X]; stay closed until
    // the caller re-opens via setWindowOpen(true) (the menu item does).
    // (This used to force m_show_window_ = true every call, which reset
    // the [X] within the same frame and made the window unclosable.)
    if (!m_show_window_) return;

    // (Space-bar pause/resume is handled in the application's GLFW
    // key callback — it sets a static flag that drawFrame consumes
    // by calling gpu_profiler_.togglePause(), which avoids fighting
    // with ImGui's own key state and works even when ImGui doesn't
    // have window focus.  Don't add a second handler here — that
    // would double-toggle every press and the pause would appear
    // to do nothing.)

    // --- Sanitize view state up front (defensive) -----------------------
    if (!std::isfinite(m_view_start_ms_))  m_view_start_ms_  = 0.0f;
    if (!std::isfinite(m_view_range_ms_))  m_view_range_ms_  = 50.0f;
    if (m_view_range_ms_ < 0.05f)          m_view_range_ms_  = 0.05f;
    if (m_view_range_ms_ > 10000.0f)       m_view_range_ms_  = 10000.0f;
    if (m_view_start_ms_ < 0.0f)           m_view_start_ms_  = 0.0f;

    ImGui::SetNextWindowSize(ImVec2(900, 520), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(500, 300), ImVec2(FLT_MAX, FLT_MAX));

    ImGuiWindowFlags wflags =
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
        ImGuiWindowFlags_NoDocking;

    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
    if (!ImGui::Begin("GPU Profiler", &m_show_window_, wflags)) {
        ImGui::End();
        return;
    }

    // ---- Header row --------------------------------------------------------

    // Latest-frame stats (last written slot).
    const int latest_slot =
        (m_frame_write_idx_ - 1 + kHistorySize) % kHistorySize;
    const float latest_ms = (m_frame_count_ > 0)
                          ? m_frames_[latest_slot].total_ms : 0.0f;

    const float latest_cpu_ms = (m_frame_count_ > 0)
                              ? m_frames_[latest_slot].total_cpu_ms : 0.0f;
    ImGui::Text("CPU: %.3f ms (%.1f fps)   GPU: %.3f ms (%.1f fps)",
        latest_cpu_ms,
        (latest_cpu_ms > 0.0f) ? 1000.0f / latest_cpu_ms : 0.0f,
        latest_ms,
        (latest_ms > 0.0f) ? 1000.0f / latest_ms : 0.0f);

    ImGui::SameLine(0, 20);

    // Pause/Resume toggle — ALWAYS visible (used to be conditionally
    // hidden behind the LIVE label, which made it hard to click out
    // of pause when zoom/pan auto-pause kept re-engaging).  Big
    // colored button so it can't be missed.  Space-bar shortcut
    // toggles the same flag (handled below).
    if (m_paused_) {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.80f, 0.40f, 0.05f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.00f, 0.55f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.65f, 0.30f, 0.02f, 1.0f));
        if (ImGui::Button("  RESUME  (Space)  ")) {
            m_paused_ = false;
        }
        ImGui::PopStyleColor(3);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.20f, 0.50f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.65f, 0.30f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.15f, 0.40f, 0.15f, 1.0f));
        if (ImGui::Button("  PAUSE   (Space)  ")) {
            m_paused_ = true;
        }
        ImGui::PopStyleColor(3);
    }

    // (Space-bar shortcut handled at the very top of drawImGui,
    // before ImGui::Begin, so it still works when this window is
    // collapsed or unfocused.)

    ImGui::SameLine(0, 20);
    ImGui::TextDisabled("Wheel=zoom (live)  |  Drag=pan (auto-pause)  |  Space=toggle");

    ImGui::Separator();

    if (m_frame_count_ == 0) {
        ImGui::TextDisabled("(no GPU data collected yet)");
        ImGui::End();
        return;
    }

    // ---- Layout constants --------------------------------------------------
    // Two stacks: CPU lanes on top, GPU lanes below, separated by a
    // labelled divider so it's instantly obvious which is which.
    const float kMinimapH      = 32.0f;   // minimap strip height
    const float kRulerH        = 30.0f;   // time ruler height
    const float kLaneH         = 30.0f;   // height per depth lane
    const int   kMaxDepth      = 8;
    const int   kCpuMaxDepth   = 6;       // CPU stacks tend shallower
    const float kCpuFlameH     = kLaneH * kCpuMaxDepth;
    const float kGpuFlameH     = kLaneH * kMaxDepth;
    const float kDividerH      = 18.0f;   // CPU/GPU label strip between stacks
    const float kFlameH        = kCpuFlameH + kDividerH + kGpuFlameH;
    const float kPadding       = 4.0f;

    ImDrawList* dl    = ImGui::GetWindowDrawList();
    ImVec2      cp    = ImGui::GetCursorScreenPos();
    float       winW  = ImGui::GetContentRegionAvail().x;

    // Guard against degenerate sizes (collapsed window / tiny layout).
    if (winW < 16.0f) {
        ImGui::TextDisabled("(window too small)");
        ImGui::End();
        return;
    }

    // ---- Minimap -----------------------------------------------------------
    {
        ImVec2 mm_min = cp;
        ImVec2 mm_max = ImVec2(cp.x + winW, cp.y + kMinimapH);

        // Background
        dl->AddRectFilled(mm_min, mm_max, IM_COL32(30, 30, 30, 220));
        dl->AddRect      (mm_min, mm_max, IM_COL32(80, 80, 80, 200));

        // Compute total global span for minimap scaling.  Use max of
        // CPU + GPU totals so a CPU-bound frame still extends the
        // minimap to its full width.
        float total_span = 0.0f;
        for (int i = 0; i < m_frame_count_; ++i) {
            int slot = (m_frame_count_ < kHistorySize)
                     ? i
                     : (m_frame_write_idx_ + i) % kHistorySize;
            float fe = m_frame_global_start_[slot] +
                       std::max(m_frames_[slot].total_ms,
                                m_frames_[slot].total_cpu_ms);
            if (fe > total_span) total_span = fe;
        }
        if (total_span < 1e-6f) total_span = 1.0f;

        // Draw each frame bar colour-coded by duration.
        // Green (fast) → Red (slow), threshold ≈ 33 ms (30 fps).
        const uint32_t col_fast = IM_COL32( 50, 200,  50, 220);
        const uint32_t col_slow = IM_COL32(200,  50,  50, 220);
        const float slow_thresh = 33.3f;

        for (int i = 0; i < m_frame_count_; ++i) {
            int slot = (m_frame_count_ < kHistorySize)
                     ? i
                     : (m_frame_write_idx_ + i) % kHistorySize;

            float gs  = m_frame_global_start_[slot];
            float dur = std::max(m_frames_[slot].total_ms,
                                 m_frames_[slot].total_cpu_ms);

            if (!std::isfinite(gs) || !std::isfinite(dur) || dur < 0.0f) continue;
            float x0 = mm_min.x + (gs       / total_span) * winW;
            float x1 = mm_min.x + ((gs+dur) / total_span) * winW;
            x0 = gp_sanitize_px(x0, mm_min.x);
            x1 = gp_sanitize_px(x1, mm_max.x);
            if (x1 < x0 + 1.0f) x1 = x0 + 1.0f;

            float t   = std::min(dur / slow_thresh, 1.0f);
            uint32_t c = lerpColor(col_fast, col_slow, t);

            float y0 = mm_min.y + 2.0f;
            float y1 = mm_max.y - 2.0f;
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), c);
        }

        // Viewport overlay on minimap.
        {
            float vx0 = mm_min.x + (m_view_start_ms_ / total_span) * winW;
            float vx1 = mm_min.x + ((m_view_start_ms_ + m_view_range_ms_) / total_span) * winW;
            vx0 = std::max(vx0, mm_min.x);
            vx1 = std::min(vx1, mm_max.x);
            dl->AddRectFilled(ImVec2(vx0, mm_min.y), ImVec2(vx1, mm_max.y),
                              IM_COL32(255, 255, 255, 40));
            dl->AddRect      (ImVec2(vx0, mm_min.y), ImVec2(vx1, mm_max.y),
                              IM_COL32(255, 255, 255, 160), 0.0f, 0, 1.5f);
        }

        // Click on minimap to navigate.
        ImGui::SetCursorScreenPos(mm_min);
        ImGui::InvisibleButton("##minimap", ImVec2(winW, kMinimapH));
        if (ImGui::IsItemClicked()) {
            float mx = ImGui::GetIO().MousePos.x - mm_min.x;
            float frac = mx / winW;
            m_view_start_ms_ = total_span * frac - m_view_range_ms_ * 0.5f;
            if (m_view_start_ms_ < 0.0f) m_view_start_ms_ = 0.0f;
            m_paused_ = true;
        }
    }

    cp.y += kMinimapH + kPadding;

    // ---- Main timeline canvas ---------------------------------------------
    float canvas_h = kRulerH + kFlameH + kPadding * 2.0f;
    ImVec2 cv_min  = cp;
    ImVec2 cv_max  = ImVec2(cp.x + winW, cp.y + canvas_h);
    float  canvas_w = winW;
    if (canvas_w < 1.0f) canvas_w = 1.0f;

    // Reserve the full canvas as an invisible button so we get proper hover
    // detection and layout advancement in one shot.
    ImGui::SetCursorScreenPos(cv_min);
    ImGui::InvisibleButton("##timeline_canvas", ImVec2(canvas_w, canvas_h));
    const bool canvas_hovered = ImGui::IsItemHovered();

    // Background
    dl->AddRectFilled(cv_min, cv_max, IM_COL32(20, 20, 20, 230));

    // Clip everything to the canvas.
    dl->PushClipRect(cv_min, cv_max, true);

    // ---- Ruler -------------------------------------------------------------
    {
        float ruler_y = cv_min.y;
        dl->AddRectFilled(ImVec2(cv_min.x, ruler_y),
                          ImVec2(cv_max.x, ruler_y + kRulerH),
                          IM_COL32(40, 40, 40, 255));

        // Adaptive tick interval: pick the smallest "nice" value such that
        // ticks are at least 60 px apart.  Fall back to a power-of-10 when
        // even our largest preset (100 ms) is too dense, so the loop below
        // never explodes.
        float ms_per_px   = m_view_range_ms_ / canvas_w;
        float min_tick_ms = ms_per_px * 150.0f;
        if (min_tick_ms < 1e-4f) min_tick_ms = 1e-4f;

        float nice[]  = {0.05f, 0.1f, 0.25f, 0.5f, 1.0f, 2.0f, 5.0f,
                         10.0f, 20.0f, 50.0f, 100.0f};
        float tick_ms = 0.0f;
        for (float n : nice) { if (n >= min_tick_ms) { tick_ms = n; break; } }
        if (tick_ms <= 0.0f) {
            // Above 100 ms — round up to the next power-of-10 multiple.
            float p10 = 100.0f;
            while (p10 < min_tick_ms) p10 *= 10.0f;
            tick_ms = p10;
        }

        // Hard cap on tick count as a last line of defence.
        const int kMaxTicks = 400;
        int tick_budget = kMaxTicks;

        float first_tick = ceilf(m_view_start_ms_ / tick_ms) * tick_ms;
        for (float t = first_tick;
             t <= m_view_start_ms_ + m_view_range_ms_ && tick_budget-- > 0;
             t += tick_ms) {
            float px = timeToX(t, cv_min.x, canvas_w);
            if (!std::isfinite(px)) continue;
            px = gp_sanitize_px(px, cv_min.x);
            dl->AddLine(ImVec2(px, ruler_y),
                        ImVec2(px, cv_max.y),
                        IM_COL32(60, 60, 60, 180));
            dl->AddLine(ImVec2(px, ruler_y),
                        ImVec2(px, ruler_y + kRulerH),
                        IM_COL32(140, 140, 140, 220));

            char lbl[32];
            if (tick_ms < 1.0f)
                snprintf(lbl, sizeof(lbl), "%.2fms", t);
            else if (tick_ms < 10.0f)
                snprintf(lbl, sizeof(lbl), "%.1fms", t);
            else
                snprintf(lbl, sizeof(lbl), "%.0fms", t);

            float lbl_h = ImGui::GetFontSize();
            float lbl_y = ruler_y + (kRulerH - lbl_h) * 0.5f;
            dl->AddText(ImVec2(px + 2.0f, lbl_y),
                        IM_COL32(200, 200, 200, 255), lbl);
        }
    }

    // ---- Flame chart lanes -------------------------------------------------
    // Layout (top to bottom inside the timeline canvas):
    //   ruler          (kRulerH)
    //   CPU lanes      (kCpuFlameH)   ← new
    //   CPU/GPU label  (kDividerH)    ← new
    //   GPU lanes      (kGpuFlameH)
    const float cpu_flame_y = cv_min.y + kRulerH;
    const float divider_y   = cpu_flame_y + kCpuFlameH;
    const float gpu_flame_y = divider_y + kDividerH;

    // CPU lane background stripes.
    for (int d = 0; d < kCpuMaxDepth; ++d) {
        uint32_t bg = (d & 1) ? IM_COL32(20, 24, 30, 255)
                               : IM_COL32(15, 18, 22, 255);
        dl->AddRectFilled(ImVec2(cv_min.x, cpu_flame_y + d * kLaneH),
                          ImVec2(cv_max.x, cpu_flame_y + (d + 1) * kLaneH),
                          bg);
    }
    // CPU/GPU divider strip with labels.
    dl->AddRectFilled(ImVec2(cv_min.x, divider_y),
                      ImVec2(cv_max.x, divider_y + kDividerH),
                      IM_COL32(50, 50, 60, 255));
    dl->AddText(ImVec2(cv_min.x + 6.0f,
                       divider_y + (kDividerH - ImGui::GetFontSize()) * 0.5f),
                IM_COL32(180, 200, 230, 255), "CPU ↑   GPU ↓");
    // GPU lane background stripes.
    for (int d = 0; d < kMaxDepth; ++d) {
        uint32_t bg = (d & 1) ? IM_COL32(28, 28, 28, 255)
                               : IM_COL32(22, 22, 22, 255);
        dl->AddRectFilled(ImVec2(cv_min.x, gpu_flame_y + d * kLaneH),
                          ImVec2(cv_max.x, gpu_flame_y + (d + 1) * kLaneH),
                          bg);
    }
    // Backwards-compat alias for the loop below — GPU lanes anchor.
    float flame_y = gpu_flame_y;

    // Draw frame scopes.
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const char*  tooltip_name = nullptr;
    float        tooltip_ms   = 0.0f;
    const char*  tooltip_kind = nullptr;   // "CPU" or "GPU"

    // ── Pass 1: CPU lanes (top half) ────────────────────────────────
    // Same hit-testing + label logic as the GPU pass below; we just
    // anchor at cpu_flame_y and clamp depth to kCpuMaxDepth.
    for (int i = 0; i < m_frame_count_; ++i) {
        int slot = (m_frame_count_ < kHistorySize)
                 ? i
                 : (m_frame_write_idx_ + i) % kHistorySize;
        float frame_gs = m_frame_global_start_[slot];
        const FrameRecord& fr = m_frames_[slot];
        for (auto& sd : fr.cpu_scopes) {
            if (sd.depth < 0 || sd.depth >= kCpuMaxDepth) continue;
            if (!std::isfinite(sd.begin_ms) || !std::isfinite(sd.end_ms)) continue;
            if (sd.end_ms < sd.begin_ms) continue;
            float g_begin = frame_gs + sd.begin_ms;
            float g_end   = frame_gs + sd.end_ms;
            float x0 = timeToX(g_begin, cv_min.x, canvas_w);
            float x1 = timeToX(g_end,   cv_min.x, canvas_w);
            if (!std::isfinite(x0) || !std::isfinite(x1)) continue;
            if (x1 < cv_min.x || x0 > cv_max.x) continue;
            x0 = std::max(x0, cv_min.x);
            x1 = std::min(x1, cv_max.x);
            if (x1 <= x0) continue;
            x0 = gp_sanitize_px(x0, cv_min.x);
            x1 = gp_sanitize_px(x1, cv_max.x);
            float y0 = cpu_flame_y + sd.depth * kLaneH + 2.0f;
            float y1 = cpu_flame_y + (sd.depth + 1) * kLaneH - 2.0f;
            float bx0 = x0 + 1.0f;
            float bx1 = x1 - 1.0f;
            if (bx1 <= bx0) { bx0 = x0; bx1 = x1; }
            uint32_t fill_col   = sd.color;
            uint32_t border_col = lerpColor(sd.color, IM_COL32(0, 0, 0, 255), 0.4f);
            dl->AddRectFilled(ImVec2(bx0, y0), ImVec2(bx1, y1), fill_col, 2.0f);
            dl->AddRect      (ImVec2(bx0, y0), ImVec2(bx1, y1), border_col, 2.0f);
            float bar_w = bx1 - bx0;
            if (bar_w > 30.0f) {
                char label[128];
                float dur_ms = sd.end_ms - sd.begin_ms;
                snprintf(label, sizeof(label), "%s  %.2fms", sd.name.c_str(), dur_ms);
                float text_h = ImGui::GetFontSize();
                float text_y = y0 + (y1 - y0 - text_h) * 0.5f;
                dl->AddText(ImVec2(bx0 + 3.0f, text_y),
                            IM_COL32(255, 255, 255, 230), label);
            }
            if (mouse.x >= bx0 && mouse.x <= bx1 &&
                mouse.y >= y0 && mouse.y <= y1) {
                tooltip_name = sd.name.c_str();
                tooltip_ms   = sd.end_ms - sd.begin_ms;
                tooltip_kind = "CPU";
            }
        }
    }

    // ── Pass 2: GPU lanes (bottom half) ─────────────────────────────
    for (int i = 0; i < m_frame_count_; ++i) {
        int slot = (m_frame_count_ < kHistorySize)
                 ? i
                 : (m_frame_write_idx_ + i) % kHistorySize;

        float frame_gs = m_frame_global_start_[slot];
        const FrameRecord& fr = m_frames_[slot];

        // Frame separator line — spans both stacks.
        float sep_x = timeToX(frame_gs, cv_min.x, canvas_w);
        if (std::isfinite(sep_x) && sep_x >= cv_min.x && sep_x <= cv_max.x) {
            dl->AddLine(ImVec2(sep_x, cpu_flame_y),
                        ImVec2(sep_x, cv_max.y),
                        IM_COL32(100, 100, 100, 120), 1.0f);
        }

        for (auto& sd : fr.scopes) {
            if (sd.depth < 0 || sd.depth >= kMaxDepth) continue;
            if (!std::isfinite(sd.begin_ms) || !std::isfinite(sd.end_ms)) continue;
            if (sd.end_ms < sd.begin_ms) continue;

            float g_begin = frame_gs + sd.begin_ms;
            float g_end   = frame_gs + sd.end_ms;

            float x0 = timeToX(g_begin, cv_min.x, canvas_w);
            float x1 = timeToX(g_end,   cv_min.x, canvas_w);

            // Sanitize: reject non-finite coordinates (possible when
            // view_range goes to 0 through some weird path).
            if (!std::isfinite(x0) || !std::isfinite(x1)) continue;

            // Cull if fully outside viewport.
            if (x1 < cv_min.x || x0 > cv_max.x) continue;
            x0 = std::max(x0, cv_min.x);
            x1 = std::min(x1, cv_max.x);
            if (x1 <= x0) continue;  // zero-width bar

            // Clamp to a sane pixel range so ImGui's vertex buffer stays healthy.
            x0 = gp_sanitize_px(x0, cv_min.x);
            x1 = gp_sanitize_px(x1, cv_max.x);

            float y0 = flame_y + sd.depth * kLaneH + 2.0f;
            float y1 = flame_y + (sd.depth + 1) * kLaneH - 2.0f;

            // Small horizontal gap between bars for visual separation.
            float bx0 = x0 + 1.0f;
            float bx1 = x1 - 1.0f;
            if (bx1 <= bx0) { bx0 = x0; bx1 = x1; } // too narrow, skip gap

            // Darken colour slightly for the border.
            uint32_t fill_col   = sd.color;
            uint32_t border_col = lerpColor(sd.color, IM_COL32(0, 0, 0, 255), 0.4f);

            dl->AddRectFilled(ImVec2(bx0, y0), ImVec2(bx1, y1), fill_col, 2.0f);
            dl->AddRect      (ImVec2(bx0, y0), ImVec2(bx1, y1), border_col, 2.0f);

            // Label: draw name + duration, vertically centred in the bar.
            float bar_w = bx1 - bx0;
            if (bar_w > 30.0f) {
                char label[128];
                float dur_ms = sd.end_ms - sd.begin_ms;
                snprintf(label, sizeof(label), "%s  %.2fms", sd.name.c_str(), dur_ms);
                float text_h = ImGui::GetFontSize();
                float text_y = y0 + (y1 - y0 - text_h) * 0.5f;
                dl->AddText(ImVec2(bx0 + 3.0f, text_y),
                            IM_COL32(255, 255, 255, 230),
                            label);
            }

            // Hover check for tooltip.
            if (mouse.x >= bx0 && mouse.x <= bx1 &&
                mouse.y >= y0 && mouse.y <= y1) {
                tooltip_name = sd.name.c_str();
                tooltip_ms   = sd.end_ms - sd.begin_ms;
                tooltip_kind = "GPU";
            }
        }
    }

    dl->PopClipRect();

    // ---- Canvas border -----------------------------------------------------
    dl->AddRect(cv_min, cv_max, IM_COL32(80, 80, 80, 200));

    // ---- Tooltip -----------------------------------------------------------
    if (tooltip_name) {
        ImGui::SetTooltip("%s [%s]\n%.3f ms",
                          tooltip_name,
                          tooltip_kind ? tooltip_kind : "",
                          tooltip_ms);
    }

    // ---- Interaction -------------------------------------------------------
    // Advance cursor past the canvas for the next widget (if any).
    // ImGui requires an item submission AFTER SetCursorScreenPos for the
    // parent window's content rect to actually grow to include the new
    // position; otherwise it asserts "Code uses SetCursorPos() to extend
    // window/parent boundaries".  A zero-size Dummy is the standard way
    // to register a no-op item there.
    ImGui::SetCursorScreenPos(ImVec2(cv_min.x, cv_max.y + kPadding));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));

    if (canvas_hovered) {
        float scroll = ImGui::GetIO().MouseWheel;
        if (fabsf(scroll) > 0.01f) {
            // Zoom around the mouse X position.
            float mouse_frac = (mouse.x - cv_min.x) / canvas_w;
            if (!std::isfinite(mouse_frac)) mouse_frac = 0.5f;
            mouse_frac = std::max(0.0f, std::min(mouse_frac, 1.0f));

            float ms_at_mouse = m_view_start_ms_ + mouse_frac * m_view_range_ms_;

            float factor = (scroll > 0.0f) ? 0.80f : 1.25f;
            float new_range = m_view_range_ms_ * factor;
            // Clamp: min 0.05 ms (50 µs) so at 900 px a pixel is ~55 ns,
            // max 10,000 ms so we can always see the entire ring history.
            new_range = std::max(0.05f, std::min(new_range, 10000.0f));
            if (!std::isfinite(new_range)) new_range = 50.0f;

            float new_start = ms_at_mouse - mouse_frac * new_range;
            if (!std::isfinite(new_start)) new_start = 0.0f;
            if (new_start < 0.0f) new_start = 0.0f;

            gpu_prof_trace("zoom: scroll=%.2f frac=%.3f ms_at_mouse=%.3f "
                "range %.3f->%.3f start %.3f->%.3f canvas_w=%.1f",
                scroll, mouse_frac, ms_at_mouse,
                m_view_range_ms_, new_range,
                m_view_start_ms_, new_start, canvas_w);

            m_view_range_ms_ = new_range;
            m_view_start_ms_ = new_start;
            // NOTE: deliberately do NOT auto-pause on zoom anymore.
            // Auto-pause-on-scroll was the main reason RESUME felt
            // broken — every accidental wheel event over the canvas
            // would freeze the timeline, and the user had to find
            // and click the small button to recover.  Zoom changes
            // m_view_range_ms_ but auto-scroll (in collectResults)
            // still keeps the right edge anchored to the latest
            // frame, so live zoom-in works naturally — you see the
            // last N ms of frames in detail, updating each frame.
            // To FREEZE the view for inspection, hit Space or click
            // the big PAUSE button.
        }

        // Left-drag pan.
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 3.0f)) {
            float dx = ImGui::GetIO().MouseDelta.x;
            float ms_per_px = m_view_range_ms_ / canvas_w;
            m_view_start_ms_ -= dx * ms_per_px;
            if (m_view_start_ms_ < 0.0f) m_view_start_ms_ = 0.0f;
            m_paused_ = true;
        }

        // Double-click: zoom out by one frame (one display-buffer-flip duration).
        // Manual detection — ImGui::IsMouseDoubleClicked can be swallowed by
        // InvisibleButton, so we track it ourselves.
        {
            static float s_last_click_time = -1.0f;
            static ImVec2 s_last_click_pos = ImVec2(-1, -1);
            constexpr float kDblClickTime = 0.35f;
            constexpr float kDblClickDist = 6.0f;

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                float now = static_cast<float>(ImGui::GetTime());
                float dx = mouse.x - s_last_click_pos.x;
                float dy = mouse.y - s_last_click_pos.y;
                bool close_enough = (dx * dx + dy * dy) < kDblClickDist * kDblClickDist;

                if (now - s_last_click_time < kDblClickTime && close_enough) {
                    // --- Double-click detected ---
                    float frame_ms = (m_frame_count_ > 0)
                        ? m_frames_[latest_slot].total_ms
                        : 16.6f;
                    if (frame_ms < 0.01f) frame_ms = 0.01f;

                    float mouse_frac = (mouse.x - cv_min.x) / canvas_w;
                    if (!std::isfinite(mouse_frac)) mouse_frac = 0.5f;
                    mouse_frac = std::max(0.0f, std::min(mouse_frac, 1.0f));

                    float ms_at_mouse = m_view_start_ms_ + mouse_frac * m_view_range_ms_;
                    float new_range = m_view_range_ms_ + frame_ms;
                    new_range = std::max(0.05f, std::min(new_range, 10000.0f));

                    float new_start = ms_at_mouse - mouse_frac * new_range;
                    if (!std::isfinite(new_start)) new_start = 0.0f;
                    if (new_start < 0.0f) new_start = 0.0f;

                    gpu_prof_trace("dblclick zoom-out: frame_ms=%.3f range %.3f->%.3f",
                        frame_ms, m_view_range_ms_, new_range);

                    m_view_range_ms_ = new_range;
                    m_view_start_ms_ = new_start;
                    m_paused_ = true;

                    s_last_click_time = -1.0f; // reset so triple-click won't re-trigger
                } else {
                    s_last_click_time = now;
                    s_last_click_pos = mouse;
                }
            }
        }
    }

    ImGui::End();

#if GPU_PROF_DEBUG
    gpu_prof_trace("drawImGui EXIT  #%d", draw_id);
#endif
}

} // namespace helper
} // namespace engine
