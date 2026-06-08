// ─────────────────────────────────────────────────────────────────────────────
// tts_engine.cpp — sherpa-onnx offline TTS behind a worker thread.
//
// The sherpa-onnx SDK (DLL + import lib + C API header) is auto-downloaded
// by CMake into third_parties/sherpa-onnx/; HAS_SHERPA_ONNX is defined only
// when it was found, so this file degrades to no-op stubs on machines where
// the download failed — the engine always builds.
// ─────────────────────────────────────────────────────────────────────────────
#include "audio/tts_engine.h"
#include "audio/audio_engine.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

#if defined(HAS_SHERPA_ONNX)
#include "sherpa-onnx/c-api/c-api.h"
#endif

namespace engine {
namespace audio {

#if defined(HAS_SHERPA_ONNX)

namespace {

struct Request {
    uint64_t    id;
    std::string text;
    int         speaker_id;
    float       speed;
};

struct TtsState {
    std::string             model_dir;       // resolved voice directory
    std::string             model_onnx;      // <dir>/<name>.onnx
    std::string             tokens_txt;      // <dir>/tokens.txt
    std::string             espeak_dir;      // <dir>/espeak-ng-data (optional)
    bool                    model_found = false;

    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<Request>     queue;
    bool                    quit = false;
    bool                    worker_started = false;

    std::atomic<uint64_t>   playing_handle{0};
    std::atomic<bool>       synthesizing{false};
    uint64_t                next_id = 1;
};

TtsState& S() {
    static TtsState s;
    return s;
}

// Locate a usable voice: `dir` itself, or its first subdirectory, holding a
// .onnx model + tokens.txt.  Records the paths in S().
bool resolveModelDir(const std::string& dir) {
    namespace fs = std::filesystem;
    TtsState& s = S();
    auto try_dir = [&](const fs::path& d) -> bool {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) return false;
        fs::path onnx, tokens;
        for (auto& e : fs::directory_iterator(d, ec)) {
            if (e.path().extension() == ".onnx") onnx = e.path();
            if (e.path().filename() == "tokens.txt") tokens = e.path();
        }
        if (onnx.empty() || tokens.empty()) return false;
        s.model_dir  = d.string();
        s.model_onnx = onnx.string();
        s.tokens_txt = tokens.string();
        const fs::path espeak = d / "espeak-ng-data";
        s.espeak_dir = fs::is_directory(espeak, ec) ? espeak.string() : "";
        return true;
    };
    if (try_dir(dir)) return true;
    std::error_code ec;
    if (std::filesystem::is_directory(dir, ec)) {
        for (auto& e : std::filesystem::directory_iterator(dir, ec)) {
            if (e.is_directory() && try_dir(e.path())) return true;
        }
    }
    return false;
}

// Worker thread: owns the sherpa-onnx instance for its whole lifetime so
// the (heavy, ~1s) model load never touches the render thread.
void workerMain() {
    TtsState& s = S();

    SherpaOnnxOfflineTtsConfig config;
    std::memset(&config, 0, sizeof(config));
    config.model.vits.model         = s.model_onnx.c_str();
    config.model.vits.tokens        = s.tokens_txt.c_str();
    config.model.vits.data_dir =
        s.espeak_dir.empty() ? nullptr : s.espeak_dir.c_str();
    config.model.vits.noise_scale   = 0.667f;
    config.model.vits.noise_scale_w = 0.8f;
    config.model.vits.length_scale  = 1.0f;
    config.model.num_threads        = 2;       // CPU synth; keep the cores
    config.model.provider           = "cpu";   // for the engine
    config.max_num_sentences        = 2;       // chunked synthesis

    const SherpaOnnxOfflineTts* tts = SherpaOnnxCreateOfflineTts(&config);
    if (!tts) {
        std::printf("[tts] failed to create sherpa-onnx TTS from '%s'\n",
                    s.model_dir.c_str());
        return;
    }
    std::printf("[tts] voice ready: %s\n", s.model_onnx.c_str());

    for (;;) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(s.mtx);
            s.cv.wait(lk, [&] { return s.quit || !s.queue.empty(); });
            if (s.quit) break;
            req = std::move(s.queue.front());
            s.queue.pop_front();
        }

        s.synthesizing.store(true);
        const SherpaOnnxGeneratedAudio* audio =
            SherpaOnnxOfflineTtsGenerate(tts, req.text.c_str(),
                                         req.speaker_id, req.speed);
        s.synthesizing.store(false);
        if (!audio || audio->n <= 0) {
            if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
            std::printf("[tts] synthesis produced no audio\n");
            continue;
        }

        // Dialog semantics: a new line replaces whatever is still playing.
        const uint64_t prev = s.playing_handle.exchange(0);
        if (prev) AudioEngine::stop(prev);

        const uint64_t h = AudioEngine::playPcm(
            audio->samples, (size_t)audio->n, (uint32_t)audio->sample_rate,
            AudioEngine::Bus::kVoice);
        s.playing_handle.store(h);
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    }

    SherpaOnnxDestroyOfflineTts(tts);
}

// Caller must hold S().mtx.
bool ensureWorkerLocked(const std::string& model_dir_hint) {
    TtsState& s = S();
    if (s.worker_started) return s.model_found;
    s.worker_started = true;

    s.model_found =
        resolveModelDir(model_dir_hint.empty() ? "assets/models/tts"
                                               : model_dir_hint);
    if (!s.model_found) {
        std::printf("[tts] no voice model under '%s' — text-to-voice "
                    "disabled (run Setup.bat to download one)\n",
                    model_dir_hint.empty() ? "assets/models/tts"
                                           : model_dir_hint.c_str());
        return false;
    }
    s.worker = std::thread(workerMain);
    return true;
}

} // namespace

bool TtsEngine::init(const std::string& model_dir) {
    TtsState& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    return ensureWorkerLocked(model_dir);
}

void TtsEngine::shutdown() {
    TtsState& s = S();
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        if (!s.worker_started) return;
        s.quit = true;
    }
    s.cv.notify_all();
    if (s.worker.joinable()) s.worker.join();
}

bool TtsEngine::available() {
    TtsState& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!s.worker_started) return resolveModelDir("assets/models/tts");
    return s.model_found;
}

uint64_t TtsEngine::speak(const std::string& text, int speaker_id,
                          float speed) {
    if (text.empty()) return 0;
    TtsState& s = S();
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        if (!ensureWorkerLocked("")) return 0;
        id = s.next_id++;
        // Dialog semantics: drop anything still waiting — only the newest
        // line matters.
        s.queue.clear();
        s.queue.push_back(Request{id, text, speaker_id, speed});
    }
    s.cv.notify_one();
    return id;
}

void TtsEngine::stop() {
    TtsState& s = S();
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        s.queue.clear();
    }
    const uint64_t h = s.playing_handle.exchange(0);
    if (h) AudioEngine::stop(h);
}

bool TtsEngine::speaking() {
    TtsState& s = S();
    if (s.synthesizing.load()) return true;
    const uint64_t h = s.playing_handle.load();
    return h != 0 && AudioEngine::isPlaying(h);
}

#else // !HAS_SHERPA_ONNX — stubs (engine builds without the SDK)

bool TtsEngine::init(const std::string&) {
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::printf("[tts] sherpa-onnx SDK not present at build time — "
                    "text-to-voice disabled (re-run CMake configure)\n");
    }
    return false;
}
void     TtsEngine::shutdown() {}
bool     TtsEngine::available() { return false; }
uint64_t TtsEngine::speak(const std::string&, int, float) {
    init("");
    return 0;
}
void TtsEngine::stop() {}
bool TtsEngine::speaking() { return false; }

#endif // HAS_SHERPA_ONNX

} // namespace audio
} // namespace engine
