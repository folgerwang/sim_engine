#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// audio_engine.h — engine audio playback core (miniaudio-backed).
//
// Static singleton: lazily initialised on first use, torn down via
// shutdown() (safe to skip — the OS reclaims the device at process exit).
//
// Three mix buses with independent volumes:
//   kMusic — background / generated music tracks
//   kSfx   — sound effects + Content Browser audio previews
//   kVoice — runtime TTS dialog lines (fed via playPcm)
//
// File playback supports .wav / .mp3 / .flac (miniaudio built-in decoders).
// playPcm() exists for the ML text-to-voice path: a TTS backend synthesises
// mono float32 PCM and hands it straight to the voice bus without touching
// disk.
//
// Implementation detail: if third_parties/miniaudio/miniaudio.h is absent
// (CMake downloads it at configure time), every call degrades to a no-op
// returning failure — the engine still builds and runs silently.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>

namespace engine {
namespace audio {

class AudioEngine {
public:
    enum class Bus : int { kMusic = 0, kSfx = 1, kVoice = 2 };
    static constexpr int kNumBuses = 3;

    // Idempotent; called automatically by play*().  Returns false when the
    // backend is unavailable (no miniaudio.h at build time, or no output
    // device at runtime).
    static bool init();
    static void shutdown();
    static bool ready();

    // Start a file on a bus.  Returns a handle (0 = failure).  Non-looping
    // sounds self-release once finished (reaped inside update()).
    static uint64_t playFile(const std::string& path, Bus bus,
                             bool loop = false, float volume = 1.0f);

    // Mono float32 PCM playback (the TTS path).  The samples are COPIED, the
    // caller's buffer can be freed immediately.
    static uint64_t playPcm(const float* samples, size_t sample_count,
                            uint32_t sample_rate, Bus bus,
                            float volume = 1.0f);

    static void stop(uint64_t handle);
    static void stopAll();
    static void stopBus(Bus bus);
    static bool isPlaying(uint64_t handle);

    // Live per-sound volume (0..1).  No-op if the handle has finished.
    static void setVolume(uint64_t handle, float volume);

    static void  setBusVolume(Bus bus, float v);   // 0..1
    static float busVolume(Bus bus);

    // Per-frame housekeeping: releases finished one-shot sounds.  Cheap;
    // call once per frame from the app loop.
    static void update();
};

} // namespace audio
} // namespace engine
