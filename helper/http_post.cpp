// =============================================================================
// http_post.cpp — POSIX sockets implementation of http_post.h
// (Linux / macOS only; Windows uses the original WinHTTP paths.)
// =============================================================================
#ifndef _WIN32

#include "http_post.h"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <sstream>

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace engine {
namespace helper {

namespace {

// Connect with a timeout: non-blocking connect + poll, then back to blocking.
int connectWithTimeout(const std::string& host, unsigned short port,
                       int timeout_ms, std::string& err) {
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* res = nullptr;
    const std::string port_s = std::to_string(port);
    int rc = getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
    if (rc != 0 || !res) {
        err = "getaddrinfo(" + host + ") failed: " + gai_strerror(rc);
        return -1;
    }

    int fd = -1;
    for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        const int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);

        rc = connect(fd, ai->ai_addr, (socklen_t)ai->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            struct pollfd pfd;
            pfd.fd = fd;
            pfd.events = POLLOUT;
            rc = poll(&pfd, 1, timeout_ms);
            if (rc == 1) {
                int soerr = 0;
                socklen_t slen = sizeof(soerr);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &slen);
                rc = (soerr == 0) ? 0 : -1;
                if (rc != 0) errno = soerr;
            } else {
                rc = -1;                       // timeout or poll error
                if (rc != 0 && errno == 0) errno = ETIMEDOUT;
            }
        }
        if (rc == 0) {
            fcntl(fd, F_SETFL, flags);         // back to blocking
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0 && err.empty()) {
        err = "connect to " + host + ":" + port_s + " failed: " +
              std::strerror(errno ? errno : ECONNREFUSED);
    }
    return fd;
}

bool sendAll(int fd, const char* data, size_t len) {
    size_t off = 0;
    while (off < len) {
        const ssize_t n = send(fd, data + off, len - off,
#ifdef MSG_NOSIGNAL
                               MSG_NOSIGNAL
#else
                               0
#endif
        );
        if (n <= 0) {
            if (n < 0 && (errno == EINTR)) continue;
            return false;
        }
        off += (size_t)n;
    }
    return true;
}

}  // namespace

SimpleHttpResult simpleHttpRequest(
    const std::string& host,
    unsigned short     port,
    const char*        method,
    const std::string& path,
    const std::string& body,
    int                connect_timeout_ms,
    int                recv_timeout_ms,
    std::atomic<size_t>* bytes_received) {

    SimpleHttpResult r;

    const int fd = connectWithTimeout(host, port, connect_timeout_ms, r.err);
    if (fd < 0) return r;

    // Per-recv idle timeout — approximates WinHTTP's receive timeout.
    struct timeval tv;
    tv.tv_sec  = recv_timeout_ms / 1000;
    tv.tv_usec = (recv_timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    // ── Request ──────────────────────────────────────────────────────────
    std::ostringstream req;
    req << method << ' ' << (path.empty() ? "/" : path) << " HTTP/1.1\r\n"
        << "Host: " << host << ':' << port << "\r\n"
        << "Connection: close\r\n"
        << "Accept: application/json\r\n";
    if (!body.empty()) {
        req << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n";
    }
    req << "\r\n";
    const std::string head = req.str();

    if (!sendAll(fd, head.data(), head.size()) ||
        (!body.empty() && !sendAll(fd, body.data(), body.size()))) {
        r.err = std::string("send failed: ") + std::strerror(errno);
        close(fd);
        return r;
    }

    // ── Response: read everything until EOF / idle timeout ───────────────
    std::string raw;
    raw.reserve(16 * 1024);
    // Parse the header block as soon as it's complete so we can decode the
    // body incrementally and publish live progress.
    size_t header_end = std::string::npos;
    size_t body_start = 0;
    bool   chunked = false;
    long long content_length = -1;
    std::string decoded;                    // de-chunked body
    size_t chunk_scan = 0;                  // parse cursor into raw (chunked)

    auto parseHeaders = [&](void) {
        // Status line: "HTTP/1.1 200 OK"
        const size_t sp = raw.find(' ');
        if (sp != std::string::npos) {
            r.status = (unsigned int)std::atoi(raw.c_str() + sp + 1);
        }
        // Headers we care about.
        std::string lower = raw.substr(0, header_end);
        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
        const size_t te = lower.find("transfer-encoding:");
        if (te != std::string::npos &&
            lower.find("chunked", te) != std::string::npos &&
            lower.find("chunked", te) < lower.find('\n', te)) {
            chunked = true;
        }
        const size_t cl = lower.find("content-length:");
        if (cl != std::string::npos) {
            content_length = std::atoll(lower.c_str() + cl + 15);
        }
        body_start = header_end + 4;
        chunk_scan = body_start;
    };

    // Incremental chunked-transfer decoder over raw[chunk_scan..].
    auto drainChunks = [&](void) {
        for (;;) {
            const size_t nl = raw.find("\r\n", chunk_scan);
            if (nl == std::string::npos) return;             // need more data
            const long sz = strtol(raw.c_str() + chunk_scan, nullptr, 16);
            if (sz <= 0) { chunk_scan = raw.size(); return; }  // final chunk
            const size_t data_at = nl + 2;
            if (raw.size() < data_at + (size_t)sz + 2) return; // partial
            decoded.append(raw, data_at, (size_t)sz);
            chunk_scan = data_at + (size_t)sz + 2;             // skip CRLF
        }
    };

    char buf[16 * 1024];
    for (;;) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;                    // EOF, timeout, or error
        raw.append(buf, (size_t)n);

        if (header_end == std::string::npos) {
            header_end = raw.find("\r\n\r\n");
            if (header_end != std::string::npos) parseHeaders();
        }
        if (header_end != std::string::npos) {
            if (chunked) drainChunks();
            if (bytes_received) {
                const size_t got = chunked ? decoded.size()
                                           : raw.size() - body_start;
                bytes_received->store(got, std::memory_order_relaxed);
            }
            // Stop early once a known-length body is complete.
            if (!chunked && content_length >= 0 &&
                raw.size() - body_start >= (size_t)content_length) {
                break;
            }
        }
    }
    close(fd);

    if (header_end == std::string::npos) {
        r.err = raw.empty() ? "empty response (connection closed / timeout)"
                            : "malformed HTTP response";
        r.status = 0;
        return r;
    }

    r.body = chunked ? std::move(decoded) : raw.substr(body_start);
    if (r.status == 0) r.err = "missing HTTP status line";
    return r;
}

}  // namespace helper
}  // namespace engine

#endif  // !_WIN32
