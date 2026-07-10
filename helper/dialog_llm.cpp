//
// dialog_llm.cpp — Ollama /api/chat client for the in-game dialogue box.
// See dialog_llm.h for the contract.  Transport is a trimmed copy of
// material_classifier.cpp's WinHTTP path (same daemon, same reasons:
// NO_PROXY so a system proxy can't eat localhost, plain HTTP).
//
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "Windows.h"
#include "winhttp.h"
#pragma comment(lib, "winhttp.lib")

#include "json.hpp"   // vendored at third_parties/tinygltf/json.hpp

#include "dialog_llm.h"

namespace engine {
namespace helper {
namespace {

using nlohmann::json;

// ── Shared state ─────────────────────────────────────────────────────
struct Msg { std::string role, content; };
struct State {
    std::mutex             mtx;
    std::atomic<bool>      probe_started{false};
    std::atomic<bool>      probe_done{false};
    std::atomic<bool>      is_available{false};
    std::atomic<bool>      in_flight{false};
    std::atomic<uint64_t>  revision{0};
    std::string            model;
    std::string            host = "localhost";
    unsigned short         port = 11434;
    std::vector<Msg>       history;   // excludes the system prompt
    std::string            last_reply;
};
State& S() {
    static State s;
    return s;
}

std::string getEnvStr(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::string(v) : std::string(fallback);
}

std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());   // host names are ASCII
}

// Minimal WinHTTP round-trip (blocking — only ever called from the
// worker threads below, never the render thread).
struct HttpResult { unsigned int status = 0; std::string body; };
HttpResult httpRequest(
    const std::string& host, unsigned short port,
    const wchar_t* method, const std::string& path,
    const std::string& body) {
    HttpResult r{};
    HINTERNET session = WinHttpOpen(
        L"RealWorld-DialogLlm/1.0",
        WINHTTP_ACCESS_TYPE_NO_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) return r;
    // Generation on a 2B model can take a while on first load — give
    // the daemon a generous receive window.
    WinHttpSetTimeouts(session, 5000, 5000, 120000, 120000);
    HINTERNET conn = WinHttpConnect(
        session, widen(host).c_str(), port, 0);
    if (conn) {
        HINTERNET req = WinHttpOpenRequest(
            conn, method, widen(path).c_str(),
            nullptr, WINHTTP_NO_REFERER,
            WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
        if (req) {
            const wchar_t* hdrs =
                L"Content-Type: application/json\r\n";
            BOOL ok = WinHttpSendRequest(
                req, hdrs, (DWORD)-1,
                body.empty() ? WINHTTP_NO_REQUEST_DATA
                             : (LPVOID)body.data(),
                (DWORD)body.size(), (DWORD)body.size(), 0);
            if (ok) ok = WinHttpReceiveResponse(req, nullptr);
            if (ok) {
                DWORD status = 0, sz = sizeof(status);
                WinHttpQueryHeaders(
                    req,
                    WINHTTP_QUERY_STATUS_CODE |
                        WINHTTP_QUERY_FLAG_NUMBER,
                    WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz,
                    WINHTTP_NO_HEADER_INDEX);
                r.status = status;
                for (;;) {
                    DWORD avail = 0;
                    if (!WinHttpQueryDataAvailable(req, &avail) ||
                        avail == 0) {
                        break;
                    }
                    std::string chunk(avail, '\0');
                    DWORD got = 0;
                    if (!WinHttpReadData(req, chunk.data(), avail,
                                         &got) || got == 0) {
                        break;
                    }
                    chunk.resize(got);
                    r.body += chunk;
                }
            }
            WinHttpCloseHandle(req);
        }
        WinHttpCloseHandle(conn);
    }
    WinHttpCloseHandle(session);
    return r;
}

std::string systemPrompt() {
    // Default keeps the fine-tune in character; override with
    // RW_DIALOG_SYSTEM for other personas.
    return getEnvStr(
        "RW_DIALOG_SYSTEM",
        "\xE4\xBD\xA0\xE6\x98\xAF\xE9\x87\x91\xE5\xBA\xB8\xE6\xAD\xA6"
        "\xE4\xBE\xA0\xE4\xB8\x96\xE7\x95\x8C\xE4\xB8\xAD\xE7\x9A\x84"
        "\xE4\xB8\x80\xE4\xBD\x8D\xE4\xBE\xA0\xE5\xAE\xA2\xEF\xBC\x8C"
        "\xE8\xAF\xB7\xE7\x94\xA8\xE7\xAE\x80\xE7\x9F\xAD\xE7\x9A\x84"
        "\xE6\xAD\xA6\xE4\xBE\xA0\xE9\xA3\x8E\xE6\xA0\xBC\xE5\x9B\x9E"
        "\xE7\xAD\x94\xE7\x8E\xA9\xE5\xAE\xB6\xE3\x80\x82");
        // "你是金庸武侠世界中的一位侠客，请用简短的武侠风格回答玩家。"
}

// One-time daemon probe + model resolution (background thread).
void runProbe() {
    auto& s = S();
    // OLLAMA_HOST override, "host[:port]".
    const std::string host_env = getEnvStr("OLLAMA_HOST", "");
    if (!host_env.empty()) {
        const size_t colon = host_env.find(':');
        std::lock_guard<std::mutex> lk(s.mtx);
        s.host = host_env.substr(0, colon);
        if (colon != std::string::npos) {
            s.port = (unsigned short)std::atoi(
                host_env.c_str() + colon + 1);
        }
    }
    std::string host;
    unsigned short port;
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        host = s.host;
        port = s.port;
    }

    const HttpResult tags =
        httpRequest(host, port, L"GET", "/api/tags", "");
    std::string resolved;
    if (tags.status == 200) {
        // 1. Exact tag from env.
        resolved = getEnvStr("RW_DIALOG_MODEL", "");
        // 2. Auto-discover a "jinyong" tag.
        if (resolved.empty()) {
            try {
                const json j = json::parse(tags.body);
                for (const auto& m : j.value("models", json::array())) {
                    std::string name = m.value("name", "");
                    std::string low = name;
                    for (auto& c : low) {
                        c = (char)std::tolower((unsigned char)c);
                    }
                    if (low.find("jinyong") != std::string::npos ||
                        low.find("jin-yong") != std::string::npos ||
                        low.find("jin_yong") != std::string::npos) {
                        resolved = name;
                        break;
                    }
                }
            } catch (...) {}
        }
        // 3./4. Fall back to the classifier's model.
        if (resolved.empty()) {
            resolved = getEnvStr("OLLAMA_MODEL", "qwen3.5:2b");
            std::cout << "[dialog.llm] no 'jinyong' tag found — "
                         "falling back to '" << resolved
                      << "' (set RW_DIALOG_MODEL to pick the "
                         "fine-tune explicitly)" << std::endl;
        }
    }

    {
        std::lock_guard<std::mutex> lk(s.mtx);
        s.model = resolved;
    }
    s.is_available.store(!resolved.empty());
    s.probe_done.store(true);
    if (!resolved.empty()) {
        std::cout << "[dialog.llm] dialogue model: " << resolved
                  << " @ " << host << ":" << port << std::endl;
    } else {
        std::cout << "[dialog.llm] Ollama unreachable — dialogue box "
                     "keeps its scripted lines" << std::endl;
    }
}

}  // namespace

bool DialogLlm::available() {
    auto& s = S();
    if (!s.probe_started.exchange(true)) {
        std::thread(runProbe).detach();
    }
    return s.is_available.load();
}

std::string DialogLlm::modelName() {
    auto& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    return s.model;
}

bool DialogLlm::busy() { return S().in_flight.load(); }

uint64_t DialogLlm::revision() { return S().revision.load(); }

std::string DialogLlm::reply() {
    auto& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    return s.last_reply;
}

void DialogLlm::resetConversation() {
    auto& s = S();
    std::lock_guard<std::mutex> lk(s.mtx);
    s.history.clear();
    s.last_reply.clear();
    s.revision.fetch_add(1);
}

bool DialogLlm::send(const std::string& user_text) {
    auto& s = S();
    if (!available() || user_text.empty()) return false;
    if (s.in_flight.exchange(true)) return false;

    std::thread([user_text]() {
        auto& s = S();
        std::string host, model;
        unsigned short port;
        json messages = json::array();
        {
            std::lock_guard<std::mutex> lk(s.mtx);
            host  = s.host;
            port  = s.port;
            model = s.model;
            messages.push_back(
                { {"role", "system"}, {"content", systemPrompt()} });
            for (const auto& m : s.history) {
                messages.push_back(
                    { {"role", m.role}, {"content", m.content} });
            }
            messages.push_back(
                { {"role", "user"}, {"content", user_text} });
        }

        json body_j = {
            {"model", model},
            {"messages", messages},
            {"stream", false},
        };
        const HttpResult res = httpRequest(
            host, port, L"POST", "/api/chat", body_j.dump());

        std::string reply_text;
        if (res.status == 200) {
            try {
                const json j = json::parse(res.body);
                reply_text = j.value("message", json::object())
                                 .value("content", "");
            } catch (...) {}
        }
        if (reply_text.empty()) {
            reply_text =
                "(\xE6\xB2\x89\xE9\xBB\x98\xE4\xB8\x8D\xE8\xAF\xAD)";
            // "(沉默不语)" — visible marker for a failed generation.
            std::cout << "[dialog.llm] chat request failed, status "
                      << res.status << std::endl;
        }

        {
            std::lock_guard<std::mutex> lk(s.mtx);
            s.history.push_back({"user", user_text});
            s.history.push_back({"assistant", reply_text});
            // Cap the context so a long session can't outgrow the
            // model's window: keep the last 16 turns.
            const size_t kMaxMsgs = 32;
            if (s.history.size() > kMaxMsgs) {
                s.history.erase(
                    s.history.begin(),
                    s.history.begin() +
                        (s.history.size() - kMaxMsgs));
            }
            s.last_reply = reply_text;
        }
        s.revision.fetch_add(1);
        s.in_flight.store(false);
    }).detach();
    return true;
}

}  // namespace helper
}  // namespace engine
