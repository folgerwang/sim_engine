// =============================================================================
// http_post.h — minimal blocking plain-HTTP client for POSIX platforms
//
// Shared transport for the local-Ollama consumers (material_classifier,
// dialog_llm, auto_rig text-to-animation) on Linux / macOS, mirroring the
// WinHTTP path those files use on Windows: plaintext HTTP 1.1, no proxy,
// generous receive timeout (local LLM generation can take minutes on CPU).
//
// Windows builds keep their original WinHTTP implementations; this TU
// compiles to nothing there.
// =============================================================================
#pragma once

#ifndef _WIN32

#include <atomic>
#include <cstddef>
#include <string>

namespace engine {
namespace helper {

struct SimpleHttpResult {
    unsigned int status = 0;   // 0 = transport failure (see err)
    std::string  body;
    std::string  err;          // human-readable failure reason
};

// Single-shot synchronous request to http://<host>:<port><path>.
//   method             "GET" / "POST" / ...
//   body               request payload ("" sends no body)
//   connect_timeout_ms socket connect ceiling
//   recv_timeout_ms    per-recv idle ceiling (NOT total transfer time —
//                      matches WinHTTP's receive-timeout semantics well
//                      enough for a token-streaming local daemon)
//   bytes_received     optional live progress counter (relaxed stores)
// Handles Content-Length, chunked transfer-encoding, and EOF-terminated
// bodies. No TLS — the Ollama daemon speaks bare HTTP on localhost.
SimpleHttpResult simpleHttpRequest(
    const std::string& host,
    unsigned short     port,
    const char*        method,
    const std::string& path,
    const std::string& body,
    int                connect_timeout_ms,
    int                recv_timeout_ms,
    std::atomic<size_t>* bytes_received = nullptr);

}  // namespace helper
}  // namespace engine

#endif  // !_WIN32
