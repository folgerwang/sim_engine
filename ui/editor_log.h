#pragma once
// ---------------------------------------------------------------------------
//  EditorLog — a tiny thread-safe ring buffer of console lines.
//
//  The console-routing buffers in main.cpp push every completed std::cout /
//  std::cerr line here; the editor's Output Log panel (menu.cpp) reads a
//  snapshot each frame and renders it UE5-style.  Bounded so a long-running
//  session can't grow without limit.  All access is mutex-guarded because log
//  lines arrive from background threads (async mesh load, BVH build, etc.).
// ---------------------------------------------------------------------------
#include <deque>
#include <string>
#include <vector>
#include <mutex>
#include <cstdint>

namespace engine {
namespace ui {

class EditorLog {
public:
    static EditorLog& get() {
        static EditorLog s_instance;
        return s_instance;
    }

    // Append one line (trailing '\n' is stripped for display).
    void push(std::string line) {
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::lock_guard<std::mutex> lk(mutex_);
        if (lines_.size() >= kMaxLines) lines_.pop_front();
        lines_.push_back(std::move(line));
        ++version_;
    }

    // Copy a snapshot of all lines + the current version (so the UI can detect
    // changes and only auto-scroll when something new arrived).
    void snapshot(std::vector<std::string>& out, std::uint64_t& version) {
        std::lock_guard<std::mutex> lk(mutex_);
        out.assign(lines_.begin(), lines_.end());
        version = version_;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mutex_);
        lines_.clear();
        ++version_;
    }

private:
    EditorLog() = default;
    static constexpr std::size_t kMaxLines = 5000;
    std::mutex                   mutex_;
    std::deque<std::string>      lines_;
    std::uint64_t                version_ = 0;
};

}  // namespace ui
}  // namespace engine
