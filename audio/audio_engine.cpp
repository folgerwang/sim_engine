// ─────────────────────────────────────────────────────────────────────────────
// audio_engine.cpp — miniaudio-backed implementation of AudioEngine.
//
// miniaudio.h is fetched by CMake into third_parties/miniaudio/ at configure
// time (same pattern as the LibTorch auto-download).  If it is missing the
// whole backend compiles to no-op stubs so the engine never fails to build
// over an audio dependency.
// ─────────────────────────────────────────────────────────────────────────────
#include "audio/audio_engine.h"

#if __has_include("third_parties/miniaudio/miniaudio.h")
#define RW_HAS_MINIAUDIO 1
// Single-TU implementation — keep MINIAUDIO_IMPLEMENTATION in THIS file only.
// Built-in decoders (wav/flac/mp3) are all enabled.
#define MA_NO_ENCODING        // we never write audio files from the engine
#define MINIAUDIO_IMPLEMENTATION
#include "third_parties/miniaudio/miniaudio.h"
#else
#define RW_HAS_MINIAUDIO 0
#endif

#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace engine {
namespace audio {

#if RW_HAS_MINIAUDIO

namespace {

struct Voice {
    std::unique_ptr<ma_sound>        sound;
    // Owning copy of raw PCM for playPcm() voices (file voices leave these
    // empty).  ma_audio_buffer refers into pcm — keep both alive together.
    std::unique_ptr<ma_audio_buffer> buffer;
    std::vector<float>               pcm;
    bool                             looping = false;
    int                              bus     = 0;
};

struct State {
    ma_engine                          engine{};
    ma_sound_group                     groups[AudioEngine::kNumBuses]{};
    float                              volumes[AudioEngine::kNumBuses] =
                                           {1.0f, 1.0f, 1.0f};
    std::unordered_map<uint64_t, Voice> voices;
    uint64_t                           next_handle = 1;
    bool                               inited      = false;
    bool                               failed      = false;
    std::mutex                         mtx;
};

State& S() {
    static State s;
    return s;
}

// Caller must hold S().mtx.
bool initLocked() {
    State& s = S();
    if (s.inited) return true;
    if (s.failed) return false;
    if (ma_engine_init(nullptr, &s.engine) != MA_SUCCESS) {
        std::printf("[audio] ma_engine_init failed — audio disabled\n");
        s.failed = true;
        return false;
    }
    for (int i = 0; i < AudioEngine::kNumBuses; ++i) {
        if (ma_sound_group_init(&s.engine, 0, nullptr, &s.groups[i]) !=
            MA_SUCCESS) {
            std::printf("[audio] bus %d init failed — audio disabled\n", i);
            for (int j = 0; j < i; ++j) ma_sound_group_uninit(&s.groups[j]);
            ma_engine_uninit(&s.engine);
            s.failed = true;
            return false;
        }
        ma_sound_group_set_volume(&s.groups[i], s.volumes[i]);
    }
    s.inited = true;
    std::printf("[audio] engine initialised (%u Hz, %u ch)\n",
                ma_engine_get_sample_rate(&s.engine),
                ma_engine_get_channels(&s.engine));
    return true;
}

// Caller must hold S().mtx.
void reapFinishedLocked() {
    State& s = S();
    for (auto it = s.voices.begin(); it != s.voices.end();) {
        Voice& v = it->second;
        if (!v.looping && v.sound &&
            !ma_sound_is_playing(v.sound.get()) &&
            ma_sound_at_end(v.sound.get())) {
            ma_sound_uninit(v.sound.get());
            if (v.buffer) ma_audio_buffer_uninit(v.buffer.get());
            it = s.voices.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace

bool AudioEngine::init() {
    std::lock_guard<std::mutex> lk(S().mtx);
    return initLocked();
}

void AudioEngine::shutdown() {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!s.inited) return;
    for (auto& [h, v] : s.voices) {
        if (v.sound) ma_sound_uninit(v.sound.get());
        if (v.buffer) ma_audio_buffer_uninit(v.buffer.get());
    }
    s.voices.clear();
    for (int i = 0; i < kNumBuses; ++i) ma_sound_group_uninit(&s.groups[i]);
    ma_engine_uninit(&s.engine);
    s.inited = false;
}

bool AudioEngine::ready() {
    std::lock_guard<std::mutex> lk(S().mtx);
    return S().inited;
}

uint64_t AudioEngine::playFile(const std::string& path, Bus bus, bool loop,
                               float volume) {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!initLocked()) return 0;
    reapFinishedLocked();

    Voice v;
    v.sound   = std::make_unique<ma_sound>();
    v.looping = loop;
    v.bus     = (int)bus;
    const ma_result r = ma_sound_init_from_file(
        &s.engine, path.c_str(),
        MA_SOUND_FLAG_NO_SPATIALIZATION,
        &s.groups[(int)bus], nullptr, v.sound.get());
    if (r != MA_SUCCESS) {
        std::printf("[audio] failed to load '%s' (err %d)\n",
                    path.c_str(), (int)r);
        return 0;
    }
    ma_sound_set_volume(v.sound.get(), volume);
    ma_sound_set_looping(v.sound.get(), loop ? MA_TRUE : MA_FALSE);
    ma_sound_start(v.sound.get());

    const uint64_t h = s.next_handle++;
    s.voices.emplace(h, std::move(v));
    return h;
}

uint64_t AudioEngine::playPcm(const float* samples, size_t sample_count,
                              uint32_t sample_rate, Bus bus, float volume) {
    if (!samples || sample_count == 0) return 0;
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!initLocked()) return 0;
    reapFinishedLocked();

    Voice v;
    v.bus = (int)bus;
    v.pcm.assign(samples, samples + sample_count);
    v.buffer = std::make_unique<ma_audio_buffer>();
    ma_audio_buffer_config cfg = ma_audio_buffer_config_init(
        ma_format_f32, /*channels=*/1, sample_count, v.pcm.data(), nullptr);
    cfg.sampleRate = sample_rate;
    if (ma_audio_buffer_init(&cfg, v.buffer.get()) != MA_SUCCESS) {
        std::printf("[audio] playPcm: buffer init failed\n");
        return 0;
    }
    v.sound = std::make_unique<ma_sound>();
    if (ma_sound_init_from_data_source(
            &s.engine, v.buffer.get(), MA_SOUND_FLAG_NO_SPATIALIZATION,
            &s.groups[(int)bus], v.sound.get()) != MA_SUCCESS) {
        ma_audio_buffer_uninit(v.buffer.get());
        std::printf("[audio] playPcm: sound init failed\n");
        return 0;
    }
    ma_sound_set_volume(v.sound.get(), volume);
    ma_sound_start(v.sound.get());

    const uint64_t h = s.next_handle++;
    s.voices.emplace(h, std::move(v));
    return h;
}

void AudioEngine::stop(uint64_t handle) {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    auto it = s.voices.find(handle);
    if (it == s.voices.end()) return;
    if (it->second.sound) ma_sound_uninit(it->second.sound.get());
    if (it->second.buffer) ma_audio_buffer_uninit(it->second.buffer.get());
    s.voices.erase(it);
}

void AudioEngine::stopAll() {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    for (auto& [h, v] : s.voices) {
        if (v.sound) ma_sound_uninit(v.sound.get());
        if (v.buffer) ma_audio_buffer_uninit(v.buffer.get());
    }
    s.voices.clear();
}

void AudioEngine::stopBus(Bus bus) {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!s.inited) return;
    for (auto it = s.voices.begin(); it != s.voices.end();) {
        Voice& v = it->second;
        if (v.bus == (int)bus) {
            ma_sound_uninit(v.sound.get());
            if (v.buffer) ma_audio_buffer_uninit(v.buffer.get());
            it = s.voices.erase(it);
        } else {
            ++it;
        }
    }
}

bool AudioEngine::isPlaying(uint64_t handle) {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    auto it = s.voices.find(handle);
    return it != s.voices.end() && it->second.sound &&
           ma_sound_is_playing(it->second.sound.get());
}

void AudioEngine::setVolume(uint64_t handle, float v) {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    auto it = s.voices.find(handle);
    if (it != s.voices.end() && it->second.sound)
        ma_sound_set_volume(it->second.sound.get(), v);
}

void AudioEngine::setBusVolume(Bus bus, float v) {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    s.volumes[(int)bus] = v;
    if (s.inited) ma_sound_group_set_volume(&s.groups[(int)bus], v);
}

float AudioEngine::busVolume(Bus bus) {
    std::lock_guard<std::mutex> lk(S().mtx);
    return S().volumes[(int)bus];
}

void AudioEngine::update() {
    State& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (s.inited) reapFinishedLocked();
}

#else // !RW_HAS_MINIAUDIO — stub backend (engine builds without audio)

bool AudioEngine::init() {
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::printf("[audio] miniaudio.h not present at build time — "
                    "audio disabled (re-run CMake configure to download)\n");
    }
    return false;
}
void     AudioEngine::shutdown() {}
bool     AudioEngine::ready() { return false; }
uint64_t AudioEngine::playFile(const std::string&, Bus, bool, float) {
    init();
    return 0;
}
uint64_t AudioEngine::playPcm(const float*, size_t, uint32_t, Bus, float) {
    init();
    return 0;
}
void  AudioEngine::stop(uint64_t) {}
void  AudioEngine::stopAll() {}
void  AudioEngine::stopBus(Bus) {}
bool  AudioEngine::isPlaying(uint64_t) { return false; }
void  AudioEngine::setVolume(uint64_t, float) {}
void  AudioEngine::setBusVolume(Bus, float) {}
float AudioEngine::busVolume(Bus) { return 1.0f; }
void  AudioEngine::update() {}

#endif // RW_HAS_MINIAUDIO

} // namespace audio
} // namespace engine
