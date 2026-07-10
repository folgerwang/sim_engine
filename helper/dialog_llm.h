#pragma once
//
// dialog_llm.h — NPC dialogue backed by a LOCAL Ollama chat model
// (the Jin Yong fine-tune).
//
// Same transport + conventions as helper/material_classifier:
//   * talks to http://localhost:11434 (override with OLLAMA_HOST)
//   * model resolution, first hit wins:
//       1. RW_DIALOG_MODEL env var (exact Ollama tag)
//       2. first tag from /api/tags containing "jinyong"
//       3. OLLAMA_MODEL env var
//       4. "qwen3.5:2b" (the base model the classifier uses)
//   * RW_DIALOG_SYSTEM overrides the built-in system prompt.
//
// All calls are non-blocking: the availability probe and each chat
// turn run on background threads; ChatBox polls revision()/reply()
// per frame.  Conversation history is kept process-wide (one NPC
// conversation at a time — matches the single ChatBox).
//
#include <cstdint>
#include <string>

namespace engine {
namespace helper {

class DialogLlm {
public:
    // Kicks the (one-time) daemon probe on first call; returns the
    // cached result afterwards.  False until the probe finishes.
    static bool available();

    // Resolved Ollama tag ("" until the probe finishes).
    static std::string modelName();

    // Queue one user utterance.  Returns false if a reply is already
    // in flight (single outstanding request by design).
    static bool send(const std::string& user_text);

    static bool busy();

    // Monotonic change counter: bumps when reply() content changes.
    static uint64_t revision();

    // Latest assistant reply, UTF-8 (Chinese for the Jin Yong model).
    static std::string reply();

    // Drop the conversation history (keeps the resolved model).
    static void resetConversation();
};

}  // namespace helper
}  // namespace engine
