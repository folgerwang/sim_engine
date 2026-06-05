#include "vram_cuda.h"

// Isolated translation unit so platform headers (windows.h) never leak into the
// large UI/menu sources.  Everything here is resolved by DYNAMIC LOAD at runtime
// — no link-time CUDA/NVML dependency and no vendor headers required.

#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

#include <cstddef>

namespace engine {
namespace {

// ── tiny cross-platform dynamic-library helpers ─────────────────────────────
#if defined(_WIN32)
using LibHandle = HMODULE;
LibHandle loadLib(const char* name) {
    LibHandle h = GetModuleHandleA(name);   // already loaded? (e.g. by LibTorch)
    return h ? h : LoadLibraryA(name);
}
void* sym(LibHandle h, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(h, name));
}
#else
using LibHandle = void*;
LibHandle loadLib(const char* name) { return dlopen(name, RTLD_NOW | RTLD_GLOBAL); }
void* sym(LibHandle h, const char* name) { return dlsym(h, name); }
#endif

// ── NVML: true device-wide usage across ALL processes (matches nvidia-smi) ──
// This is the RIGHT source on Windows: under the WDDM memory model
// cudaMemGetInfo only reflects (roughly) the calling process, so a separate
// process like the FLUX.2 Python generator is invisible to it.  NVML reports
// the board's physical total/used regardless of which process allocated it.
struct nvmlMemory_t { unsigned long long total, free, used; };
using nvmlInit_t      = int (*)();
using nvmlGetHandle_t = int (*)(unsigned int, void**);
using nvmlGetMem_t    = int (*)(void*, nvmlMemory_t*);

bool tryNvml(unsigned long long& free_b, unsigned long long& total_b) {
    static bool          tried  = false;
    static bool          ok     = false;
    static nvmlGetMem_t  getMem = nullptr;
    static void*         dev    = nullptr;

    if (!tried) {
        tried = true;
#if defined(_WIN32)
        static const char* kNames[] = { "nvml.dll" };
#else
        static const char* kNames[] = { "libnvidia-ml.so.1", "libnvidia-ml.so" };
#endif
        for (const char* n : kNames) {
            LibHandle h = loadLib(n);
            if (!h) continue;
            auto init = reinterpret_cast<nvmlInit_t>(sym(h, "nvmlInit_v2"));
            if (!init) init = reinterpret_cast<nvmlInit_t>(sym(h, "nvmlInit"));
            auto getH = reinterpret_cast<nvmlGetHandle_t>(
                            sym(h, "nvmlDeviceGetHandleByIndex_v2"));
            if (!getH) getH = reinterpret_cast<nvmlGetHandle_t>(
                            sym(h, "nvmlDeviceGetHandleByIndex"));
            getMem = reinterpret_cast<nvmlGetMem_t>(
                            sym(h, "nvmlDeviceGetMemoryInfo"));
            // Device 0 (single-GPU machines; the engine renders on the primary).
            if (init && getH && getMem && init() == 0 && getH(0, &dev) == 0) {
                ok = true;
            } else {
                getMem = nullptr;
            }
            break;
        }
    }

    if (!ok || !getMem || !dev) return false;
    nvmlMemory_t m{};
    if (getMem(dev, &m) != 0 || m.total == 0) return false;   // non-zero == error
    free_b  = m.free;
    total_b = m.total;
    return true;
}

// ── cudaMemGetInfo fallback (device-wide on Linux / TCC; coarse on WDDM) ─────
// Kept as a backstop when NVML is unavailable (e.g. non-NVIDIA, or a stripped
// driver without nvml.dll).
using cudaMemGetInfo_t = int (*)(std::size_t*, std::size_t*);

bool tryCuda(unsigned long long& free_b, unsigned long long& total_b) {
    static bool             tried = false;
    static cudaMemGetInfo_t fn    = nullptr;
    if (!tried) {
        tried = true;
#if defined(_WIN32)
        static const char* kNames[] = {
            "cudart64_12.dll", "cudart64_120.dll", "cudart64_128.dll",
            "cudart64_110.dll", "cudart64_11.dll",
        };
#else
        static const char* kNames[] = {
            "libcudart.so", "libcudart.so.12", "libcudart.so.11.0",
        };
#endif
        for (const char* n : kNames) {
            LibHandle h = loadLib(n);
            if (!h) continue;
            fn = reinterpret_cast<cudaMemGetInfo_t>(sym(h, "cudaMemGetInfo"));
            if (fn) break;
        }
    }
    if (!fn) return false;
    std::size_t f = 0, t = 0;
    if (fn(&f, &t) != 0 || t == 0) return false;
    free_b  = static_cast<unsigned long long>(f);
    total_b = static_cast<unsigned long long>(t);
    return true;
}

}  // namespace

bool queryDeviceWideVramBytes(unsigned long long& free_bytes,
                              unsigned long long& total_bytes) {
    // NVML first (true cross-process), then CUDA as a fallback.
    if (tryNvml(free_bytes, total_bytes)) return true;
    if (tryCuda(free_bytes, total_bytes)) return true;
    return false;
}

}  // namespace engine
