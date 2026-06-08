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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(HAS_SHERPA_ONNX)
#include "sherpa-onnx/c-api/c-api.h"
#endif

namespace engine {
namespace audio {

#if defined(HAS_SHERPA_ONNX)

namespace {

namespace fs = std::filesystem;

struct Request {
    uint64_t    id;
    std::string text;
    int         speaker_id;
    float       speed;
};

// sherpa-onnx ships several offline-TTS model FAMILIES, each needing a
// different config struct.  Feeding a non-VITS model into the VITS config
// makes onnxruntime fault — so we detect the family and configure the right
// one, and refuse families we don't support (rather than crash).
enum class VoiceType { kUnsupported = 0, kVits, kKokoro, kKitten, kMatcha };

struct VoiceInfo {
    VoiceType   type = VoiceType::kUnsupported;
    std::string model;     // vits/kokoro/kitten model .onnx, or matcha acoustic
    std::string tokens;    // tokens.txt
    std::string voices;    // kokoro/kitten voices.bin
    std::string vocoder;   // matcha vocoder .onnx
    std::string espeak;    // espeak-ng-data dir (optional)
    std::string dict_dir;  // jieba/dict dir for CJK models (optional)
    std::string lexicon;   // comma-joined lexicon*.txt (optional)
};

// Inspect a folder and decide which family it is + the files needed to load
// it.  Returns false (kUnsupported) for families we can't drive — those are
// hidden from the picker so selecting them can't crash.
bool analyzeVoiceDir(const fs::path& d, VoiceInfo& out) {
    std::error_code ec;
    if (!fs::is_directory(d, ec)) return false;

    std::vector<fs::path> onnx;
    std::vector<fs::path> lexicons;
    fs::path tokens, voices, vocoder, espeak, dict_dir;
    for (auto& e : fs::directory_iterator(d, ec)) {
        const std::string fn = e.path().filename().string();
        std::string lf = fn;
        for (auto& c : lf) c = (char)std::tolower((unsigned char)c);
        if (e.path().extension() == ".onnx") {
            if (lf.find("vocos") != std::string::npos ||
                lf.find("vocoder") != std::string::npos ||
                lf.find("hifigan") != std::string::npos)
                vocoder = e.path();
            else
                onnx.push_back(e.path());
        } else if (fn == "tokens.txt") {
            tokens = e.path();
        } else if (lf == "voices.bin" || lf == "voices.json") {
            voices = e.path();
        } else if (e.is_directory() && fn == "espeak-ng-data") {
            espeak = e.path();
        } else if (e.is_directory() && (fn == "dict" || fn == "dict_dir")) {
            dict_dir = e.path();
        } else if (lf.rfind("lexicon", 0) == 0 &&
                   e.path().extension() == ".txt") {
            lexicons.push_back(e.path());   // lexicon.txt / lexicon-zh.txt / ...
        }
    }
    if (onnx.empty() || tokens.empty()) return false;

    std::string name = d.filename().string();
    for (auto& c : name) c = (char)std::tolower((unsigned char)c);
    auto has = [&](const char* s) { return name.find(s) != std::string::npos; };

    // Non-VITS families (kokoro / kitten / matcha) have per-model quirks that
    // make sherpa-onnx hard-EXIT the process on load (bad style_dim, missing
    // per-language lexicon, etc.) — uncatchable from C++.  VITS/Piper is the
    // reliable workhorse (300+ voices, every language, espeak G2P), so the
    // picker shows VITS only by default.  Set RW_TTS_EXPERIMENTAL=1 to also
    // surface the exotic families (at your own risk of a crash).
    static const bool experimental = [] {
        const char* e = std::getenv("RW_TTS_EXPERIMENTAL");
        return e && e[0] && e[0] != '0';
    }();

    out = VoiceInfo{};
    out.tokens = tokens.string();
    out.espeak = espeak.empty() ? "" : espeak.string();
    out.dict_dir = dict_dir.empty() ? "" : dict_dir.string();
    out.model  = onnx.front().string();
    std::sort(lexicons.begin(), lexicons.end());
    for (size_t i = 0; i < lexicons.size(); ++i) {
        if (i) out.lexicon += ",";
        out.lexicon += lexicons[i].string();
    }

    // Non-VITS families — opt-in only (see RW_TTS_EXPERIMENTAL above).
    if (has("matcha")) {
        if (!experimental || vocoder.empty()) return false;
        out.type = VoiceType::kMatcha;
        out.vocoder = vocoder.string();
        return true;
    }
    if (has("kokoro")) {
        if (!experimental || voices.empty()) return false;
        const bool multilang = has("multi-lang") || has("multi_lang");
        if (multilang && (out.lexicon.empty() || out.dict_dir.empty()))
            return false;
        if (!multilang && out.espeak.empty() && out.lexicon.empty())
            return false;
        out.type = VoiceType::kKokoro;
        out.voices = voices.string();
        return true;
    }
    if (has("kitten")) {
        if (!experimental || voices.empty()) return false;
        out.type = VoiceType::kKitten;
        out.voices = voices.string();
        return true;
    }
    // Families we don't drive — hide them rather than crash.
    if (has("zipvoice") || has("supertonic") || has("pocket")) return false;

    // VITS family.  Only the self-contained G2P paths are reliable:
    //   • espeak-ng-data  (piper, all languages) — robust, no extra files;
    //   • Latin character/phoneme tokens (coqui / mms / ljs / vctk).
    // CJK models (vits-zh / cantonese / aishell / melo-zh) drive their text
    // through a jieba dict + lexicon and hard-EXIT sherpa at generate time
    // (especially on the Latin dialog text we feed them), so they're hidden.
    const bool char_based =
        has("coqui") || has("mms") || has("-ljs") || has("vctk") ||
        has("ljspeech");
    const bool cjk =
        has("-zh") || has("zh-") || has("zh_") || has("cantonese") ||
        has("aishell") || has("melo");
    if (cjk) return false;
    if (out.espeak.empty() && !char_based) return false;
    out.type = VoiceType::kVits;
    return true;
}

struct TtsState {
    std::string             voices_root;     // assets/ml_models/tts (for listing)
    std::string             model_dir;       // resolved voice directory
    VoiceInfo               voice;           // resolved family + file paths
    bool                    model_found = false;

    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<Request>     queue;
    bool                    quit = false;
    bool                    worker_started = false;
    bool                    reload_voice = false;  // worker rebuilds tts

    // Last spoken line, kept so the editor voice picker can re-audition.
    std::string             last_text;
    int                     last_speaker = 0;
    float                   last_speed   = 1.0f;

    std::atomic<uint64_t>   playing_handle{0};
    std::atomic<bool>       synthesizing{false};
    uint64_t                next_id = 1;
};

TtsState& S() {
    static TtsState s;
    return s;
}

// Locate a usable voice holding a .onnx model + tokens.txt.  Records the
// paths in S().  Selection order:
//   1) `dir` itself if it directly contains a voice;
//   2) the RW_TTS_VOICE env var matched against subdirectory names (so you
//      can keep several voices installed — amy, ryan, ... — and switch with
//      one env var, no re-download).  Substring match, case-insensitive,
//      so RW_TTS_VOICE=ryan or RW_TTS_VOICE=male-ish both work;
//   3) RANDOM among the installed voices when several are present (a
//      different NPC voice each launch).  Disable with RW_TTS_RANDOM=0,
//      which falls back to (4);
//   4) the first subdirectory that contains a voice (alphabetical).
//
// The random pick is made ONCE per process and cached, so available() and
// init() — which both call this — always agree on the chosen voice.
bool resolveModelDir(const std::string& dir) {
    namespace fs = std::filesystem;
    TtsState& s = S();
    auto try_dir = [&](const fs::path& d) -> bool {
        VoiceInfo vi;
        if (!analyzeVoiceDir(d, vi)) return false;   // unsupported / not a voice
        s.model_dir = d.string();
        s.voice     = vi;
        return true;
    };
    s.voices_root = dir;   // remembered for listVoices() / setVoice()
    if (try_dir(dir)) return true;

    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return false;

    // Collect voice subdirectories, sorted for stable "first" semantics.
    std::vector<fs::path> voices;
    for (auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.is_directory()) voices.push_back(e.path());
    }
    std::sort(voices.begin(), voices.end());
    if (voices.empty()) return false;

    // RW_TTS_VOICE preference (substring, case-insensitive) — explicit
    // choice wins over the random pick.
    const char* want = std::getenv("RW_TTS_VOICE");
    if (want && want[0]) {
        std::string w(want);
        for (auto& c : w) c = (char)std::tolower((unsigned char)c);
        for (const auto& v : voices) {
            std::string nm = v.filename().string();
            for (auto& c : nm) c = (char)std::tolower((unsigned char)c);
            if (nm.find(w) != std::string::npos && try_dir(v)) return true;
        }
        std::printf("[tts] RW_TTS_VOICE='%s' not found under '%s' — falling "
                    "back to auto-select\n", want, dir.c_str());
    }

    // Random voice per run (default when >1 installed).  Cached so every
    // call this process returns the same pick.  Opt out with RW_TTS_RANDOM=0.
    const char* rnd_env = std::getenv("RW_TTS_RANDOM");
    const bool randomize =
        voices.size() > 1 && !(rnd_env && rnd_env[0] == '0');
    if (randomize) {
        static std::string s_random_pick;   // process-lifetime cache
        if (s_random_pick.empty()) {
            // Fresh seed each launch so the choice varies run to run.
            std::mt19937 rng(static_cast<uint32_t>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
            std::uniform_int_distribution<size_t> dist(0, voices.size() - 1);
            s_random_pick = voices[dist(rng)].string();
        }
        if (try_dir(s_random_pick)) {
            std::printf("[tts] random voice this run: %s\n",
                        fs::path(s_random_pick).filename().string().c_str());
            return true;
        }
    }

    for (const auto& v : voices) {
        if (try_dir(v)) return true;
    }
    return false;
}

// (Re)build a sherpa-onnx TTS instance from S()'s current model paths.
// Caller passes a snapshot of the paths (taken under lock) so the heavy
// SherpaOnnxCreateOfflineTts call runs without holding the mutex.
const SherpaOnnxOfflineTts* buildTts(const VoiceInfo& v) {
    if (v.type == VoiceType::kUnsupported) return nullptr;

    SherpaOnnxOfflineTtsConfig config;
    std::memset(&config, 0, sizeof(config));
    const char* espeak = v.espeak.empty() ? nullptr : v.espeak.c_str();

    const char* dict = v.dict_dir.empty() ? nullptr : v.dict_dir.c_str();
    const char* lex  = v.lexicon.empty()  ? nullptr : v.lexicon.c_str();
    switch (v.type) {
    case VoiceType::kVits:
        config.model.vits.model         = v.model.c_str();
        config.model.vits.tokens        = v.tokens.c_str();
        config.model.vits.data_dir      = espeak;
        config.model.vits.lexicon       = lex;     // CJK models ship one
        config.model.vits.dict_dir      = dict;    // jieba dict for zh
        config.model.vits.noise_scale   = 0.667f;
        config.model.vits.noise_scale_w = 0.8f;
        config.model.vits.length_scale  = 1.0f;
        break;
    case VoiceType::kKokoro:
        config.model.kokoro.model        = v.model.c_str();
        config.model.kokoro.voices       = v.voices.c_str();
        config.model.kokoro.tokens       = v.tokens.c_str();
        config.model.kokoro.data_dir     = espeak;
        config.model.kokoro.lexicon      = lex;
        config.model.kokoro.dict_dir     = dict;
        config.model.kokoro.length_scale = 1.0f;
        break;
    case VoiceType::kKitten:
        config.model.kitten.model        = v.model.c_str();
        config.model.kitten.voices       = v.voices.c_str();
        config.model.kitten.tokens       = v.tokens.c_str();
        config.model.kitten.data_dir     = espeak;
        config.model.kitten.length_scale = 1.0f;
        break;
    case VoiceType::kMatcha:
        config.model.matcha.acoustic_model = v.model.c_str();
        config.model.matcha.vocoder        = v.vocoder.c_str();
        config.model.matcha.tokens         = v.tokens.c_str();
        config.model.matcha.data_dir       = espeak;
        config.model.matcha.noise_scale    = 0.667f;
        config.model.matcha.length_scale   = 1.0f;
        break;
    default:
        return nullptr;
    }
    config.model.num_threads = 2;       // CPU synth; keep the cores
    config.model.provider    = "cpu";   // for the engine
    config.max_num_sentences = 2;       // chunked synthesis
    return SherpaOnnxCreateOfflineTts(&config);
}

// Worker thread: owns the sherpa-onnx instance.  Rebuilds it on demand when
// the editor switches voices (reload_voice), so the (heavy, ~1s) model load
// never touches the render thread.
void workerMain() {
    TtsState& s = S();

    auto load = [&]() -> const SherpaOnnxOfflineTts* {
        VoiceInfo vi;
        std::string dir;
        {
            std::lock_guard<std::mutex> lk(s.mtx);
            vi = s.voice; dir = s.model_dir;
        }
        const SherpaOnnxOfflineTts* t = buildTts(vi);
        if (t) std::printf("[tts] voice ready: %s\n",
                           fs::path(dir).filename().string().c_str());
        else   std::printf("[tts] failed to create TTS from '%s'\n",
                           dir.c_str());
        return t;
    };

    const SherpaOnnxOfflineTts* tts = load();
    if (!tts) return;

    for (;;) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(s.mtx);
            s.cv.wait(lk, [&] {
                return s.quit || s.reload_voice || !s.queue.empty();
            });
            if (s.quit) break;
            if (s.reload_voice) {
                s.reload_voice = false;
                lk.unlock();
                // Stop the current line, swap the model, then fall through
                // to drain any queued (re-audition) request with the new
                // voice.
                const uint64_t prev = s.playing_handle.exchange(0);
                if (prev) AudioEngine::stop(prev);
                SherpaOnnxDestroyOfflineTts(tts);
                tts = load();
                if (!tts) return;
                lk.lock();
            }
            if (s.queue.empty()) continue;
            req = std::move(s.queue.front());
            s.queue.pop_front();
        }

        s.synthesizing.store(true);
        SherpaOnnxGenerationConfig gen_cfg;
        std::memset(&gen_cfg, 0, sizeof(gen_cfg));
        gen_cfg.sid   = req.speaker_id;
        gen_cfg.speed = req.speed;
        const SherpaOnnxGeneratedAudio* audio =
            SherpaOnnxOfflineTtsGenerateWithConfig(
                tts, req.text.c_str(), &gen_cfg, nullptr, nullptr);
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
        resolveModelDir(model_dir_hint.empty() ? "assets/ml_models/tts"
                                               : model_dir_hint);
    if (!s.model_found) {
        std::printf("[tts] no voice model under '%s' — text-to-voice "
                    "disabled (run Setup.bat to download one)\n",
                    model_dir_hint.empty() ? "assets/ml_models/tts"
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
    if (!s.worker_started) return resolveModelDir("assets/ml_models/tts");
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
        // Remember the line so the editor voice picker can re-audition it.
        s.last_text = text; s.last_speaker = speaker_id; s.last_speed = speed;
        // Dialog semantics: drop anything still waiting — only the newest
        // line matters.
        s.queue.clear();
        s.queue.push_back(Request{id, text, speaker_id, speed});
    }
    s.cv.notify_one();
    return id;
}

std::vector<std::string> TtsEngine::listVoices() {
    namespace fs = std::filesystem;
    TtsState& s = S();
    std::string root;
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        root = s.voices_root.empty() ? "assets/ml_models/tts" : s.voices_root;
    }
    std::vector<std::string> out;
    std::error_code ec;
    if (fs::is_directory(root, ec)) {
        for (auto& e : fs::directory_iterator(root, ec)) {
            if (!e.is_directory()) continue;
            // Only LOADABLE voices (a family we can drive) — hide the rest
            // so selecting them can't crash onnxruntime.
            VoiceInfo vi;
            if (analyzeVoiceDir(e.path(), vi))
                out.push_back(e.path().filename().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string TtsEngine::currentVoice() {
    namespace fs = std::filesystem;
    TtsState& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (s.model_dir.empty()) return std::string();
    return fs::path(s.model_dir).filename().string();
}

bool TtsEngine::setVoice(const std::string& voice_name) {
    namespace fs = std::filesystem;
    TtsState& s = S();
    bool ok = false;
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        // Make sure the worker exists (and voices_root is known).
        if (!ensureWorkerLocked("")) return false;
        const std::string root =
            s.voices_root.empty() ? "assets/ml_models/tts" : s.voices_root;
        const fs::path vd = fs::path(root) / voice_name;
        // Detect family + files; refuse anything we can't load so the worker
        // never feeds onnxruntime a mismatched config (the crash).
        VoiceInfo vi;
        if (!analyzeVoiceDir(vd, vi)) {
            std::printf("[tts] voice '%s' is an unsupported model family — "
                        "ignored\n", voice_name.c_str());
            return false;
        }

        s.model_dir  = vd.string();
        s.voice      = vi;
        s.reload_voice = true;

        // Re-audition with the new voice: the last dialog line if one has
        // played, otherwise a default sample so the picker always speaks.
        if (s.last_text.empty()) {
            s.last_text = "The quick brown fox jumps over the lazy dog.";
            s.last_speaker = 0;
            s.last_speed = 1.0f;
        }
        s.queue.clear();
        s.queue.push_back(Request{s.next_id++, s.last_text,
                                  s.last_speaker, s.last_speed});
        ok = true;
    }
    s.cv.notify_one();
    return ok;
}

uint64_t TtsEngine::repeatLast() {
    TtsState& s = S();
    std::string t; int spk; float spd;
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        if (s.last_text.empty()) return 0;
        t = s.last_text; spk = s.last_speaker; spd = s.last_speed;
    }
    return speak(t, spk, spd);
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
std::vector<std::string> TtsEngine::listVoices() { return {}; }
std::string TtsEngine::currentVoice() { return std::string(); }
bool     TtsEngine::setVoice(const std::string&) { return false; }
uint64_t TtsEngine::repeatLast() { return 0; }

#endif // HAS_SHERPA_ONNX

} // namespace audio
} // namespace engine
