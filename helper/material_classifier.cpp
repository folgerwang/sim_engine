// material_classifier.cpp — Ollama /api/chat backed classifier for
// static collision mesh categorisation.  See material_classifier.h
// for the overview; this file is the WinHTTP + nlohmann/json glue.
//
// Talks to a LOCAL Ollama daemon (default http://localhost:11434).
// Model + host are configurable via env vars:
//
//   OLLAMA_HOST  — e.g. "localhost:11434" (default) or "192.168.1.5:11434"
//                  Hostname[:port] only; no scheme.  HTTPS is not
//                  supported because Ollama doesn't ship one by default.
//   OLLAMA_MODEL — e.g. "qwen3.5:2b" (default).  Whatever tag is
//                  installed locally; the prompt expects a model
//                  competent at producing valid JSON output.  The
//                  default tag must be pulled locally (`ollama pull
//                  qwen3.5:2b`); if the daemon can't resolve the
//                  manifest, the /api/tags pre-flight below logs the
//                  installed tags and the procedural classifier carries
//                  the day instead of blocking the run.
//
// On failure (daemon unreachable, model not pulled, malformed reply)
// classifyAll() returns false and the procedural classifier carries
// the day.
//
#include "material_classifier.h"

#include <atomic>
#include <cctype>      // std::isdigit
#include <cstdlib>     // _dupenv_s (Windows-safe env var read)
#include <chrono>
#include <iostream>
#include <sstream>

// Windows.h must come BEFORE winhttp.h.  application.cpp already pulls
// in Windows.h transitively, but this TU is built on its own so we
// include it explicitly here.  NOMINMAX suppresses the legacy `min`
// and `max` preprocessor macros that Windows.h would otherwise drop
// on us — without it, any std::min(...) / std::max(...) call later
// in this file expands to `std::((a)<(b)?(a):(b))(...)` and the
// compiler trips on the spurious `(` after the scope-resolution `::`
// (C2589: "illegal token on right side of '::'").
#define NOMINMAX
#include "Windows.h"
#include "winhttp.h"

#include "json.hpp"   // vendored at third_parties/tinygltf/json.hpp,
                      // include path "${TP_DIR}/tinygltf" is on the
                      // engine target.

// Tell the linker to pull in winhttp.lib without touching the CMake
// link list manually.  Either works; the pragma is the lower-friction
// path because it lives next to the only call site.
#pragma comment(lib, "winhttp.lib")

namespace engine {
namespace helper {

namespace {

// ── Live progress counters (shared between worker & main thread) ─────
// The HTTP receive loop on the worker thread bumps g_bytes_received
// every chunk; the main-thread progress-bar reads it with relaxed
// memory order (the value is monotonic non-negative integer — no
// happens-before requirements with other state).  Reset at the
// start of every classifyAll() so subsequent runs don't carry
// previous progress.
std::atomic<size_t> g_classifier_bytes_received{0};
// Snapshot of the body upload size — gives the bar a sensible
// baseline before the response starts arriving so the bar shows
// "sending" rather than "0".
std::atomic<size_t> g_classifier_bytes_sent{0};

// ── env var helper ────────────────────────────────────────────────────
// Returns the requested env var, or `fallback` when it isn't set.
// _dupenv_s is the Windows-safe alternative to getenv() — MSVC's
// secure CRT warns about getenv().
std::string getEnv(const char* name, const char* fallback) {
    char*  buf  = nullptr;
    size_t len  = 0;
    const errno_t err = _dupenv_s(&buf, &len, name);
    if (err != 0 || buf == nullptr || len == 0) {
        if (buf) free(buf);
        return std::string(fallback);
    }
    std::string s(buf, len > 0 ? len - 1 : 0);
    free(buf);
    return s;
}

// Parse "host[:port]" into (host, port).  Default port if absent: 11434
// (Ollama's listen port).  Drop a leading "http://" if the user pasted
// one (we don't accept HTTPS — Ollama doesn't speak it natively).
struct OllamaTarget { std::string host; unsigned short port = 11434; };
OllamaTarget parseHost(const std::string& raw) {
    OllamaTarget t;
    std::string s = raw;
    const std::string http_scheme  = "http://";
    const std::string https_scheme = "https://";
    if (s.rfind(http_scheme, 0) == 0)  s.erase(0, http_scheme.size());
    if (s.rfind(https_scheme, 0) == 0) s.erase(0, https_scheme.size());
    auto colon = s.find(':');
    if (colon == std::string::npos) {
        t.host = s;
    } else {
        t.host = s.substr(0, colon);
        try {
            const int p = std::stoi(s.substr(colon + 1));
            if (p > 0 && p < 65536) t.port = static_cast<unsigned short>(p);
        } catch (...) { /* keep default */ }
    }
    if (t.host.empty()) t.host = "localhost";
    return t;
}

// UTF-8 → UTF-16 helper for the WinHTTP API surface (which is wide-
// char only).  WinHTTP API names like WinHttpOpen take LPCWSTR for
// hostnames, paths, and the user-agent string.
std::wstring toW(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(
        CP_UTF8, 0, s.data(),
        static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, s.data(),
        static_cast<int>(s.size()), out.data(), n);
    return out;
}

// ── WinHTTP request ──────────────────────────────────────────────────
// Single-shot synchronous plaintext-HTTP request to <host>:<port><path>.
// Returns (status, body) on success; (0, "") on any failure (with a
// reason printed to stdout for debugging).  Caller checks status >=
// 200 && < 300.
//
// Method is passed as a wide string ("POST" / "GET" / …) because that's
// what WinHttpOpenRequest takes.  When body is empty (e.g. a GET),
// the WinHttpSendRequest call passes WINHTTP_NO_REQUEST_DATA / 0
// instead of a copy of the empty string, which keeps the wire format
// strictly correct (no trailing CRLF with zero bytes).
//
// Plaintext (no WINHTTP_FLAG_SECURE) because Ollama serves bare HTTP
// on localhost.  WinHttpOpen with NO_PROXY because a system-wide proxy
// configured for browser traffic will typically refuse a connection
// to localhost on a non-standard port.
struct HttpResult { unsigned int status = 0; std::string body; };
HttpResult httpRequest(
    const std::string& host,
    unsigned short port,
    const wchar_t*     method,
    const std::string& path,
    const std::string& body) {

    HttpResult r{};

    // Per-phase progress marker.  Forces std::endl so the line is
    // flushed even when stdout is piped to a file or VS Output window
    // with line buffering.  Tagged "[mat.cls.http]" so it grep-greps
    // cleanly out of the larger [mat.cls] / [collision] log spam.
    auto mark = [](const char* msg) {
        std::cout << "[mat.cls.http] " << msg << std::endl;
    };

    mark("WinHttpOpen ...");
    HINTERNET hSession = WinHttpOpen(
        L"RealWorld-MaterialClassifier/1.0",
        WINHTTP_ACCESS_TYPE_NO_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0);
    if (!hSession) {
        std::cout << "[mat.cls.http] WinHttpOpen FAILED err="
                  << GetLastError() << std::endl;
        return r;
    }

    // Local LLM inference can take a while on CPU.  Bistro's bistro
    // produces ~200 materials + ~300 objects → ~15K output tokens,
    // which a 3B model at ~30 tok/s on CPU takes 5–8 minutes to
    // generate.  We pick 10 minutes as the receive ceiling so the
    // first run "just works" without manual intervention; connect /
    // send timeouts stay tight because those reflect socket-level
    // reachability, not model generation time.
    WinHttpSetTimeouts(hSession, 5000, 10000, 10000, 600000);

    std::cout << "[mat.cls.http] WinHttpConnect " << host << ":" << port
              << std::endl;
    HINTERNET hConnect = WinHttpConnect(
        hSession, toW(host).c_str(), port, 0);
    if (!hConnect) {
        std::cout << "[mat.cls.http] WinHttpConnect FAILED err="
                  << GetLastError() << " (host=" << host
                  << " port=" << port << ")" << std::endl;
        WinHttpCloseHandle(hSession);
        return r;
    }

    // Narrow the wide method back to a printable string just for the
    // log line — the underlying WinHTTP call still consumes the wide
    // pointer directly.
    {
        std::string method_narrow;
        for (const wchar_t* p = method; p && *p; ++p) {
            method_narrow.push_back(static_cast<char>(*p));
        }
        std::cout << "[mat.cls.http] WinHttpOpenRequest "
                  << method_narrow << " " << path << std::endl;
    }
    HINTERNET hRequest = WinHttpOpenRequest(
        hConnect, method, toW(path).c_str(),
        nullptr, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES,
        /*flags=*/0);  // 0 = plain HTTP; no WINHTTP_FLAG_SECURE.
    if (!hRequest) {
        std::cout << "[mat.cls.http] WinHttpOpenRequest FAILED err="
                  << GetLastError() << std::endl;
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return r;
    }

    // Ollama just needs content-type.  No auth, no API version header.
    std::wstring headers;
    headers += L"content-type: application/json\r\n";

    std::cout << "[mat.cls.http] WinHttpSendRequest body_bytes="
              << body.size() << std::endl;
    g_classifier_bytes_sent.store(
        body.size(), std::memory_order_relaxed);
    // For an empty body (GET pre-flight), feed WINHTTP_NO_REQUEST_DATA
    // + 0 length so we don't smuggle a phantom byte onto the wire.
    // We use an if/else rather than ternaries because the LPVOID vs
    // NULL type-deduction inside a `?:` warns under MSVC's stricter
    // levels.
    LPCWSTR headers_ptr;
    DWORD   headers_len;
    LPVOID  body_ptr;
    DWORD   body_len;
    if (body.empty()) {
        headers_ptr = WINHTTP_NO_ADDITIONAL_HEADERS;
        headers_len = 0;
        body_ptr    = WINHTTP_NO_REQUEST_DATA;
        body_len    = 0;
    } else {
        headers_ptr = headers.c_str();
        headers_len = static_cast<DWORD>(headers.size());
        body_ptr    = const_cast<char*>(body.data());
        body_len    = static_cast<DWORD>(body.size());
    }
    BOOL ok = WinHttpSendRequest(
        hRequest,
        headers_ptr,
        headers_len,
        body_ptr,
        body_len,
        body_len,
        0);
    if (!ok) {
        std::cout << "[mat.cls.http] WinHttpSendRequest FAILED err="
                  << GetLastError() << std::endl;
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return r;
    }

    mark("WinHttpReceiveResponse waiting (this is the long blocking "
         "call -- model is generating tokens here)");
    if (!WinHttpReceiveResponse(hRequest, nullptr)) {
        std::cout << "[mat.cls.http] WinHttpReceiveResponse FAILED err="
                  << GetLastError() << std::endl;
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return r;
    }
    mark("WinHttpReceiveResponse returned -- headers are in");

    // HTTP status.
    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(
        hRequest,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size,
        WINHTTP_NO_HEADER_INDEX);
    r.status = status_code;
    std::cout << "[mat.cls.http] status=" << status_code
              << " -- draining body..." << std::endl;

    // Drain the response body in chunks.  Log every chunk so the user
    // sees real-time progress; this is the most reassuring thing for
    // a 5-minute call.  WinHttpQueryDataAvailable() blocks until at
    // least one byte is ready OR the receive timeout fires, so each
    // iteration represents actual data arriving (not a busy-loop).
    size_t total_bytes = 0;
    int    chunk_idx   = 0;
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(hRequest, &avail)) {
            std::cout << "[mat.cls.http] WinHttpQueryDataAvailable FAILED "
                         "err=" << GetLastError() << " after "
                      << total_bytes << " bytes" << std::endl;
            break;
        }
        if (avail == 0) {
            std::cout << "[mat.cls.http] end-of-stream (total "
                      << total_bytes << " bytes in "
                      << chunk_idx << " chunks)" << std::endl;
            break;
        }
        std::string buf(avail, '\0');
        DWORD read = 0;
        if (!WinHttpReadData(hRequest, buf.data(), avail, &read)) {
            std::cout << "[mat.cls.http] WinHttpReadData FAILED err="
                      << GetLastError() << " after " << total_bytes
                      << " bytes" << std::endl;
            break;
        }
        if (read == 0) {
            std::cout << "[mat.cls.http] zero-byte read (treating as EOF "
                         "after " << total_bytes << " bytes)"
                      << std::endl;
            break;
        }
        r.body.append(buf.data(), read);
        total_bytes += read;
        ++chunk_idx;
        // Publish to the live progress counter so the UI bar moves
        // in real time as bytes arrive.  Relaxed order — the
        // consumer only reads for display, no synchronisation
        // requirements with anything else.
        g_classifier_bytes_received.store(
            total_bytes, std::memory_order_relaxed);
        std::cout << "[mat.cls.http] chunk #" << chunk_idx
                  << " size=" << read
                  << " total=" << total_bytes << " bytes"
                  << std::endl;
    }

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    mark("transport done, handles closed");
    return r;
}

// ── Prompt construction ───────────────────────────────────────────────
// Build the JSON request body for /v1/messages.  We use Haiku because
// the classification task is trivial relative to the model's capacity
// and Haiku is roughly 10× cheaper than Sonnet for the same throughput.
//
// The system prompt enumerates the categories with one-line gameplay
// definitions; the user message is the list of material+albedo pairs
// as compact JSON.  We ask the model to respond with a JSON object
// mapping each material name to a category tag.
std::string buildRequestBody(
    const std::string& model,
    const std::string& scene_label,
    const std::unordered_map<std::string, std::string>& mat_collected,
    const std::unordered_map<std::string, std::string>& obj_collected) {

    using nlohmann::json;

    const std::string system_prompt = R"(You classify 3D scene strings into ONE of eleven gameplay collision categories. You receive two lists in the user message: "materials" (asset material identifiers) and "objects" (FBX/glTF node names). Reply with a single JSON object containing two sub-objects, "materials" and "objects", each mapping the input names to one of the all-caps tags below.

Categories:
- FLOOR: walkable horizontal surfaces underfoot. ALL outdoor ground — roads, streets, asphalt, pavement, sidewalks, cobblestones, plazas, courtyards, paths — plus interior floors, floor tiles, and doormats. If a surface is ground you could stand on, it is FLOOR, never CEILING.
- WALL: blocking vertical surfaces. Brick, plaster, stucco, exterior cladding, building facades, low barriers, railings, fences, roof shingles (treat roofs as Wall for blocking). IMPORTANT: plaster and stucco are ALWAYS walls, never floors. Concrete is FLOOR only when the name says it is ground (pavement/road/sidewalk/floor-tile); a bare "Concrete" / "Concrete_White" / "Concrete_Plaster" material is building cladding → WALL.
- DOOR: traversable openings. Doors, gates, rollups.
- OBJECT: gameplay props. Tables, chairs, bottles, cups, plates, lamps, signs, registers, awnings, banners, paintings, radiators.
- GLASS: see-through-but-blocking. Window glass, glass panels, frosted glass.
- CEILING: INTERIOR overhead surfaces only — the underside of a roof or upper floor, seen from below while standing inside a room. Interior ceilings, ceiling fans (only the ceiling part), interior roof underside. NEVER use CEILING for outdoor ground, roads, pavement, or any surface you walk on top of — those are FLOOR.
- STAIRS: walkable but step-aware. Staircases, stair treads.
- VEGETATION: any plant matter. Trees (trunk, bark, branches, twigs, limbs, roots), leaves, foliage, canopy, hedges, shrubs, bushes, ivy, vines, ferns, moss, grass, lawn, turf, weeds, flowers, petals, palms, saplings, atlas-billboard leaf cards. Tag bark / trunk / wood-grain materials as VEGETATION when the object name suggests a tree (Linde_Tree, Oak, Maple, Palm) — only fall back to OBJECT for cut-lumber wood like planks, boards, table-tops, crates, and barrels.
- ELEVATOR: walkable + vertical traversal. Lift platforms, elevator cars / cabs.
- LADDER: vertical hand-over-hand traversal. Ladders, fire escape ladders.
- UNKNOWN: anything you can't confidently place. Better to mark UNKNOWN than to guess.

Rules:
- "objects" entries are FBX/node names like "Bistro_Research_Exterior_Linde_Tree_Large" — treat as the strongest single signal when the name is descriptive.
- "materials" entries are shader/material identifiers like "MASTER_Wood_Brown".
- The optional "albedo" texture filename on a material can disambiguate.
- Output ONLY the JSON object, no prose, no markdown. Schema:
  {"materials": {<name>: <TAG>, ...}, "objects": {<name>: <TAG>, ...}})";

    json materials = json::array();
    materials.get_ref<json::array_t&>().reserve(mat_collected.size());
    for (const auto& [name, albedo] : mat_collected) {
        json entry;
        entry["name"] = name;
        if (!albedo.empty()) entry["albedo"] = albedo;
        materials.push_back(std::move(entry));
    }

    json objects = json::array();
    objects.get_ref<json::array_t&>().reserve(obj_collected.size());
    for (const auto& [name, _unused] : obj_collected) {
        json entry;
        entry["name"] = name;
        objects.push_back(std::move(entry));
    }

    json user_payload;
    user_payload["scene"]     = scene_label;
    user_payload["materials"] = std::move(materials);
    user_payload["objects"]   = std::move(objects);
    const std::string user_text = user_payload.dump();

    // Ollama /api/chat envelope.  Differences from the previous
    // Anthropic shape:
    //   • `model` is whatever local tag the user has pulled.
    //   • `system` lives inside the messages array as a role="system"
    //     entry, not as a top-level field.
    //   • `stream`:false → single JSON response instead of NDJSON
    //     chunks (which WinHTTP can't easily consume).
    //   • `format`:"json" → Ollama post-processes the model's reply
    //     to make sure it parses as JSON.  Useful for smaller models
    //     that occasionally drift into prose or markdown fences.
    //   • `options.num_ctx` raised so the long materials+objects
    //     payload doesn't get truncated on models with small defaults.
    //   • `options.temperature` low so the model commits to a tag
    //     instead of hedging.
    json system_msg;
    system_msg["role"]    = "system";
    system_msg["content"] = system_prompt;
    json user_msg;
    user_msg["role"]    = "user";
    user_msg["content"] = user_text;

    json body;
    body["model"]    = model;
    body["stream"]   = true;
    body["format"]   = "json";
    body["think"]    = false;
    // num_ctx must cover BOTH the input prompt AND the generated reply
    // for a SINGLE batch.  With batching at 10 items per request, a
    // request body is ~3 KB ≈ ~2k tokens of prompt, and the reply
    // adds ~10 entries × ~30 tokens = ~300 tokens.  4096 leaves
    // generous headroom for both.
    //
    // CRITICAL: Ollama pre-allocates a KV cache buffer of size
    // num_ctx on the first call to the model.  At 32K, that's ~3 GB
    // for a 2B model — slow to allocate, can fail to fit in VRAM
    // and bounce the model to CPU, and produces a multi-minute first-
    // batch stall with zero visible response progress.  At 4K it's
    // ~400 MB, allocates in seconds, and easily fits in any modern
    // GPU's VRAM.
    body["options"]  = {
        {"num_ctx",     4096},
        {"temperature", 0.1},
    };
    // Pin the model in VRAM for 10 minutes between requests.  Ollama's
    // default keep_alive is 5 minutes, which is normally fine, but a
    // 52-batch run that takes >5 minutes (e.g. on a CPU-bound machine
    // or while the user has the Inspector window blocking the main
    // thread for a few seconds) can otherwise see the model unloaded
    // mid-run and pay another ~3-second reload cost on the next batch.
    // "10m" is the string format Ollama expects (also accepts "0" to
    // never cache, or "-1" to keep forever).
    body["keep_alive"] = "10m";
    body["messages"] = json::array();
    body["messages"].push_back(std::move(system_msg));
    body["messages"].push_back(std::move(user_msg));
    return body.dump();
}

// ── Response parsing ──────────────────────────────────────────────────
// Ollama /api/chat (with stream:false) returns:
//   { "model":"…", "created_at":"…",
//     "message": { "role":"assistant", "content":"<JSON STRING>" },
//     "done":true, "done_reason":"stop", … }
// The model's reply text is in message.content; we re-parse THAT as the
// actual classification JSON.  Expected inner schema:
//   { "materials": {name:TAG}, "objects": {name:TAG} }
// — we extract both sub-maps.  Backward-compatible with the legacy
// flat {name:TAG} schema (whole reply → materials map, objects empty).
//
// Defensive: even with format:"json" forcing valid JSON, smaller
// models occasionally wrap the reply in ```json fences — strip those.
struct ParsedReply {
    std::unordered_map<std::string, std::string> materials;
    std::unordered_map<std::string, std::string> objects;
};
ParsedReply parseResponse(const std::string& body) {
    using nlohmann::json;
    ParsedReply out;

    // Reassemble the assistant reply from NDJSON frames.  Walk the body
    // line by line; each non-blank line should be one streaming frame.
    // Accumulate every frame's message.content into `text`.  Surface any
    // {"error":...} envelope.  A stream:false body is a single line, so
    // this also covers the non-streaming case.
    std::string text;
    bool saw_frame = false;
    {
        size_t pos = 0;
        while (pos <= body.size()) {
            const size_t nl = body.find('\n', pos);
            std::string line = (nl == std::string::npos)
                ? body.substr(pos)
                : body.substr(pos, nl - pos);
            pos = (nl == std::string::npos) ? body.size() + 1 : nl + 1;
            // Trim trailing CR/space and skip blank lines.
            while (!line.empty() &&
                   (line.back() == '\r' || line.back() == ' ')) {
                line.pop_back();
            }
            if (line.empty()) continue;

            json frame = json::parse(line, nullptr,
                                     /*allow_exceptions=*/false);
            if (frame.is_discarded()) continue;  // partial / non-JSON line
            saw_frame = true;
            if (frame.contains("error") && frame["error"].is_string()) {
                std::cout << "[mat.cls] ollama error: "
                          << frame["error"].get<std::string>() << std::endl;
            }
            if (frame.contains("message") && frame["message"].is_object() &&
                frame["message"].contains("content") &&
                frame["message"]["content"].is_string()) {
                text += frame["message"]["content"].get<std::string>();
            }
        }
    }

    if (!saw_frame) {
        std::cout << "[mat.cls] response had no parseable JSON frames"
                  << std::endl;
        return out;
    }
    if (text.empty()) {
        std::cout << "[mat.cls] accumulated message.content was empty"
                  << std::endl;
        return out;
    }

    try {
        // Strip ```json ... ``` fences if the model added them.
        const auto strip_fence = [](std::string s) {
            auto t0 = s.find("```");
            if (t0 != std::string::npos) {
                auto nl = s.find('\n', t0);
                if (nl != std::string::npos) s.erase(0, nl + 1);
                auto t1 = s.rfind("```");
                if (t1 != std::string::npos) s.erase(t1);
            }
            return s;
        };
        text = strip_fence(text);

        json inner = json::parse(text, nullptr, false);
        if (inner.is_discarded() || !inner.is_object()) {
            std::cout << "[mat.cls] inner reply not a JSON object"
                      << std::endl;
            return out;
        }

        auto pull = [](const json& src,
                       std::unordered_map<std::string, std::string>& dst) {
            for (auto it = src.begin(); it != src.end(); ++it) {
                if (!it.value().is_string()) continue;
                dst.emplace(it.key(), it.value().get<std::string>());
            }
        };

        const bool has_mats = inner.contains("materials") &&
                              inner["materials"].is_object();
        const bool has_objs = inner.contains("objects") &&
                              inner["objects"].is_object();
        if (has_mats || has_objs) {
            if (has_mats) pull(inner["materials"], out.materials);
            if (has_objs) pull(inner["objects"],   out.objects);
        } else {
            // Backward-compatible flat schema: treat the whole object
            // as the materials map.
            pull(inner, out.materials);
        }
    } catch (const std::exception& e) {
        std::cout << "[mat.cls] parse exception: " << e.what()
                  << std::endl;
    }
    return out;
}

} // anonymous namespace

std::string MaterialClassifier::normalizeObjectName(const std::string& s) {
    // Strip trailing _<digits> and trailing runs of '_' until the
    // tail stabilises.  Bistro node names instance the same prop
    // hundreds of times with patterns like "Ashtray_01_mesh_4073",
    // "Ashtray_01_mesh__", "Ashtray_01_mesh___10__" — this collapses
    // all of them down to "Ashtray_01_mesh" so the LLM sees ONE
    // canonical identifier per logical prop.
    std::string r = s;
    while (!r.empty()) {
        // Peel a trailing _<digits>.
        size_t i = r.size();
        while (i > 0 && std::isdigit(static_cast<unsigned char>(r[i-1]))) {
            --i;
        }
        if (i < r.size() && i > 0 && r[i-1] == '_') {
            r.resize(i - 1);
            continue;
        }
        // Peel a trailing run of underscores.
        size_t j = r.size();
        while (j > 0 && r[j-1] == '_') --j;
        if (j < r.size()) {
            r.resize(j);
            continue;
        }
        break;
    }
    return r;
}

void MaterialClassifier::collect(
    const std::string& material_name,
    const std::string& albedo_filename,
    const std::string& object_name) {
    if (!material_name.empty()) {
        // First observation wins; in practice the same material name
        // always maps to the same albedo so subsequent collects are
        // redundant.
        mat_collected_.emplace(material_name, albedo_filename);
    }
    if (!object_name.empty()) {
        const std::string norm = normalizeObjectName(object_name);
        if (!norm.empty()) {
            obj_collected_.emplace(norm, std::string());
        }
    }
}

bool MaterialClassifier::classifyAll(const std::string& scene_label) {
    was_run_ = true;
    // Reset all state under the lock so any frame that races against
    // us at the very start sees the maps as cleared rather than
    // half-populated.
    {
        std::lock_guard<std::mutex> lock(classified_mu_);
        mat_classified_.clear();
        obj_classified_.clear();
        mat_classified_count_.store(0, std::memory_order_release);
        obj_classified_count_.store(0, std::memory_order_release);
    }
    g_classifier_bytes_received.store(0, std::memory_order_relaxed);
    g_classifier_bytes_sent.store(0,     std::memory_order_relaxed);

    if (mat_collected_.empty() && obj_collected_.empty()) {
        std::cout << "[mat.cls] nothing collected, skipping" << std::endl;
        return false;
    }

    const OllamaTarget target =
        parseHost(getEnv("OLLAMA_HOST", "localhost:11434"));
    // Default model: qwen3.5:2b — small and fast.  NOTE: this tag must
    // be pulled locally (`ollama pull qwen3.5:2b`) for classification to
    // run; if Ollama can't resolve the manifest the model-availability
    // check below logs the installed tags and skips the LLM pass.
    // History: an earlier multi-minute first-batch hang inside
    // WinHttpSendRequest was caused by the request using stream:false
    // (the WinHttp call stalled waiting for the whole response) — NOT by
    // the model tag.  That was fixed by switching /api/chat to
    // stream:true + NDJSON frame parsing, so the model default is free to
    // be whatever local tag you prefer.  Override with OLLAMA_MODEL.
    const std::string model = getEnv("OLLAMA_MODEL", "qwen3.5:2b");

    // Batch size — environment-overridable, default 10.  Smaller
    // batches give finer progress granularity and faster per-request
    // turnaround (so the Inspector populates progressively); larger
    // batches amortise the per-request HTTP overhead better.  We
    // clamp to (1..200) to keep the request body inside a reasonable
    // num_ctx and to make sure a misconfigured "0" doesn't divide-
    // by-zero downstream.
    size_t batch_size = 10;
    {
        const std::string raw = getEnv("OLLAMA_BATCH_SIZE", "10");
        try {
            const int n = std::stoi(raw);
            if (n >= 1 && n <= 200) {
                batch_size = static_cast<size_t>(n);
            }
        } catch (...) { /* keep default */ }
    }

    // Build a flat list of (kind, name, albedo) tuples.  Materials
    // first then objects so within each batch the materials sit
    // together — produces a less confusing batch-mix that the LLM
    // handles more cleanly than "mat-obj-mat-obj-…".
    struct Item {
        bool        is_material;
        std::string name;
        std::string albedo;
    };
    std::vector<Item> items;
    items.reserve(mat_collected_.size() + obj_collected_.size());
    for (const auto& [name, albedo] : mat_collected_) {
        items.push_back({true,  name, albedo});
    }
    for (const auto& [name, _albedo] : obj_collected_) {
        items.push_back({false, name, std::string()});
    }

    const size_t total_items   = items.size();
    const size_t total_batches =
        (total_items + batch_size - 1) / batch_size;

    std::cout << "[mat.cls] starting batched classify: " << total_items
              << " items in " << total_batches << " batches of "
              << batch_size << " (scene='" << scene_label
              << "', model='" << model << "', host="
              << target.host << ":" << target.port << ")"
              << std::endl;

    // ── Pre-flight: GET /api/tags ────────────────────────────────────
    // Cheap (<1 ms locally) sanity check that:
    //   1. The Ollama daemon is actually listening on OLLAMA_HOST, and
    //   2. The requested OLLAMA_MODEL is pulled locally.
    //
    // Without this, a tag that isn't pulled locally makes /api/chat
    // accept the POST and then wedge during manifest resolution.  From
    // the client's side that can look like a long hang inside
    // WinHttpSendRequest with no error or diagnostic to act on.
    // /api/tags, by contrast, returns instantly with the list of
    // locally-pulled models, so we can fail loudly and helpfully and
    // fall back to the procedural classifier rather than blocking.
    {
        std::cout << "[mat.cls] pre-flight: GET /api/tags ..." << std::endl;
        const auto t_pf_0 = std::chrono::steady_clock::now();
        const HttpResult tags = httpRequest(
            target.host, target.port, L"GET", "/api/tags", std::string());
        const auto t_pf_1 = std::chrono::steady_clock::now();
        const double pf_ms = std::chrono::duration<double, std::milli>(
            t_pf_1 - t_pf_0).count();

        if (tags.status == 0) {
            std::cout << "[mat.cls] PRE-FLIGHT FAILED: cannot reach "
                      << target.host << ":" << target.port
                      << " — is `ollama serve` running?  (Test with:"
                      << "  curl http://" << target.host << ":"
                      << target.port << "/api/tags)" << std::endl;
            return false;
        }
        if (tags.status != 200) {
            std::cout << "[mat.cls] PRE-FLIGHT FAILED: /api/tags HTTP "
                      << tags.status << " after " << pf_ms << "ms";
            if (!tags.body.empty()) {
                std::cout << " body: " << tags.body.substr(0, 256);
            }
            std::cout << std::endl;
            return false;
        }

        // Parse the list and see whether `model` is locally present.
        // Ollama's /api/tags returns:
        //   { "models": [ { "name": "qwen2.5:3b", "model": "...", ... }, ... ] }
        // We accept either an exact match against "name" OR — to be
        // forgiving — a match where the user said "qwen2.5" but Ollama
        // lists "qwen2.5:latest", which is the same thing.
        std::vector<std::string> available;
        bool model_present = false;
        try {
            const auto j = nlohmann::json::parse(tags.body);
            if (j.contains("models") && j["models"].is_array()) {
                for (const auto& m : j["models"]) {
                    if (!m.contains("name") || !m["name"].is_string()) {
                        continue;
                    }
                    const std::string name = m["name"].get<std::string>();
                    available.push_back(name);
                    if (name == model) {
                        model_present = true;
                    } else if (model.find(':') == std::string::npos &&
                               name.rfind(model + ":", 0) == 0) {
                        // User passed bare "qwen2.5", server has "qwen2.5:latest".
                        model_present = true;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cout << "[mat.cls] PRE-FLIGHT WARNING: could not "
                      << "parse /api/tags response (" << e.what()
                      << ") — continuing anyway, but failures from "
                      << "here on are likely model-tag related."
                      << std::endl;
        }

        std::cout << "[mat.cls] pre-flight ok in " << pf_ms << "ms, "
                  << available.size() << " local tags: ";
        for (size_t i = 0; i < available.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << available[i];
        }
        std::cout << std::endl;

        if (!model_present && !available.empty()) {
            std::cout << "[mat.cls] PRE-FLIGHT FAILED: model '" << model
                      << "' is NOT pulled locally.  Either:" << std::endl
                      << "  - run `ollama pull " << model << "` to fetch it, or"
                      << std::endl
                      << "  - set OLLAMA_MODEL env var to one of the tags "
                      << "listed above and relaunch RealWorld." << std::endl;
            return false;
        }
    }

    int batches_ok     = 0;
    int batches_failed = 0;
    const auto t_all_0 = std::chrono::steady_clock::now();

    for (size_t batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
        const size_t start = batch_idx * batch_size;
        const size_t end =
            std::min(total_items, start + batch_size);

        // Re-split this batch back into the (materials, objects)
        // shape buildRequestBody / parseResponse already expect, so
        // we don't have to rewrite the prompt schema.
        std::unordered_map<std::string, std::string> batch_mats;
        std::unordered_map<std::string, std::string> batch_objs;
        for (size_t k = start; k < end; ++k) {
            if (items[k].is_material) {
                batch_mats.emplace(items[k].name, items[k].albedo);
            } else {
                batch_objs.emplace(items[k].name, std::string());
            }
        }

        const std::string body = buildRequestBody(
            model, scene_label, batch_mats, batch_objs);
        std::cout << "[mat.cls.batch] " << (batch_idx + 1) << "/"
                  << total_batches << " sending (mats="
                  << batch_mats.size() << " objs=" << batch_objs.size()
                  << ", body=" << body.size() << "B)"
                  << std::endl;

        const auto t_batch_0 = std::chrono::steady_clock::now();
        const HttpResult res = httpRequest(
            target.host, target.port, L"POST", "/api/chat", body);
        const auto t_batch_1 = std::chrono::steady_clock::now();
        const double batch_ms = std::chrono::duration<double, std::milli>(
            t_batch_1 - t_batch_0).count();

        if (res.status < 200 || res.status >= 300) {
            std::cout << "[mat.cls.batch] " << (batch_idx + 1) << "/"
                      << total_batches << " FAILED HTTP "
                      << res.status << " after " << batch_ms << "ms";
            if (res.status == 0) {
                std::cout
                    << " (could not reach " << target.host << ":"
                    << target.port << " — is `ollama serve` running "
                    "and model '" << model
                    << "' pulled with `ollama pull " << model << "`?)";
            }
            if (!res.body.empty()) {
                std::cout << " body: " << res.body.substr(0, 256);
            }
            std::cout << std::endl;
            ++batches_failed;
            continue;
        }

        const ParsedReply reply = parseResponse(res.body);
        if (reply.materials.empty() && reply.objects.empty()) {
            std::cout << "[mat.cls.batch] " << (batch_idx + 1) << "/"
                      << total_batches
                      << " parsed empty after " << batch_ms << "ms"
                      << std::endl;
            ++batches_failed;
            continue;
        }

        // Merge under the mutex.  After unlocking, bump the atomic
        // counters with release semantics so the main thread's
        // acquire load sees both the map writes AND the bumped count
        // when it observes the new count.
        size_t new_mat_count = 0;
        size_t new_obj_count = 0;
        {
            std::lock_guard<std::mutex> lock(classified_mu_);
            for (const auto& [name, tag] : reply.materials) {
                mat_classified_.emplace(
                    name, meshCategoryFromTag(tag));
            }
            for (const auto& [name, tag] : reply.objects) {
                obj_classified_.emplace(
                    name, meshCategoryFromTag(tag));
            }
            new_mat_count = mat_classified_.size();
            new_obj_count = obj_classified_.size();
        }
        mat_classified_count_.store(new_mat_count,
                                    std::memory_order_release);
        obj_classified_count_.store(new_obj_count,
                                    std::memory_order_release);
        ++batches_ok;

        std::cout << "[mat.cls.batch] " << (batch_idx + 1) << "/"
                  << total_batches << " OK in " << batch_ms
                  << "ms (this batch: mats+="
                  << reply.materials.size() << " objs+="
                  << reply.objects.size() << "; running total: "
                  << new_mat_count << " mats, "
                  << new_obj_count << " objs)" << std::endl;
    }

    const auto t_all_1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(
        t_all_1 - t_all_0).count();
    std::cout << "[mat.cls] batched classify done in " << total_ms
              << "ms: " << batches_ok << "/" << total_batches
              << " batches OK, " << batches_failed
              << " batches failed, "
              << mat_classified_count_.load() << " mats + "
              << obj_classified_count_.load() << " objs classified"
              << std::endl;

    // Treat ANY successful batch as overall success — the caller
    // checks classifiedMaterialCount() / classifiedObjectCount() to
    // decide whether to apply categories.
    return batches_ok > 0;
}

size_t MaterialClassifier::bytesSent() {
    return g_classifier_bytes_sent.load(std::memory_order_relaxed);
}

size_t MaterialClassifier::bytesReceived() {
    return g_classifier_bytes_received.load(std::memory_order_relaxed);
}

MeshCategory MaterialClassifier::lookup(
    const std::string& material_name,
    const std::string& object_name) const {
    // Category is derived SOLELY from the object name.  One material
    // (concrete, glass, plaster) is reused across floors, walls and
    // props, so a single material verdict is necessarily wrong on every
    // surface but the one it was judged for; the object name is the unit
    // that actually maps to a category.  material_name is intentionally
    // ignored (kept in the signature so call sites need not change).
    (void)material_name;
    if (!object_name.empty()) {
        const std::string norm = normalizeObjectName(object_name);
        auto it = obj_classified_.find(norm);
        if (it != obj_classified_.end()) {
            return it->second;  // object has the only say (incl. Unknown)
        }
    }
    return MeshCategory::Unknown;
}

} // namespace helper
} // namespace engine
