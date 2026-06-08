#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// tts_engine.h — runtime text-to-voice (sherpa-onnx + Piper-style VITS voice).
//
// Static singleton over a background synthesis thread: speak() enqueues a
// line, the worker synthesizes float PCM via sherpa-onnx (CPU, real-time
// factor << 1 for Piper voices) and plays it on AudioEngine's kVoice bus.
// A new speak() supersedes the currently playing line — dialog semantics.
//
// Voice model: first directory under assets/models/tts/ containing a .onnx
// + tokens.txt (Setup.bat downloads vits-piper-en_US-amy-medium by default;
// drop any sherpa-onnx VITS voice next to it and pass its dir to init()).
//
// Backend availability: compiled in only when CMake found the sherpa-onnx
// SDK (HAS_SHERPA_ONNX).  Without it — or without a voice model on disk —
// every call is a safe no-op and available() returns false.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>

namespace engine {
namespace audio {

class TtsEngine {
public:
    // Lazily called by speak(); explicit init lets the app choose the model
    // directory.  Heavy model load happens on the worker thread, so this
    // never blocks the frame.
    static bool init(const std::string& model_dir = "");
    static void shutdown();

    // Backend compiled in AND a voice model directory was found.
    static bool available();

    // Queue a line for synthesis + playback on the voice bus.  Cancels the
    // currently playing line first.  speaker_id selects the voice in multi-
    // speaker models (0 for single-speaker); speed 1.0 = natural.
    // Returns a request id (0 = backend unavailable).
    static uint64_t speak(const std::string& text, int speaker_id = 0,
                          float speed = 1.0f);

    // Stop playback and drop any queued lines.
    static void stop();

    // True while a line is being synthesized or played.
    static bool speaking();
};

} // namespace audio
} // namespace engine
