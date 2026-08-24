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
  #include <unistd.h>
#endif
#if defined(__APPLE__)
  #include <mach/mach.h>
  #include <sys/sysctl.h>
#endif

#include <cstdio>

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

#if defined(__APPLE__)
// ── Apple Silicon: unified memory — "device-wide VRAM" IS system RAM ────────
// The GPU shares one pool with the CPU/OS, so the honest device-wide numbers
// are total physical memory and how much of it the system currently holds.
// "Used" mirrors Activity Monitor's Memory Used (active + wired + compressed);
// inactive/speculative/purgeable pages are reclaimable on demand and count as
// free — that is the memory the GPU could actually get.
bool tryAppleUnified(unsigned long long& free_b, unsigned long long& total_b) {
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) != 0 ||
        memsize == 0)
        return false;
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm),
                          &count) != KERN_SUCCESS)
        return false;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS || page == 0)
        page = 16384;
    const unsigned long long used =
        (static_cast<unsigned long long>(vm.active_count) +
         static_cast<unsigned long long>(vm.wire_count) +
         static_cast<unsigned long long>(vm.compressor_page_count)) *
        static_cast<unsigned long long>(page);
    total_b = memsize;
    free_b  = used < memsize ? memsize - used : 0;
    return true;
}
#endif

}  // namespace

bool queryDeviceWideVramBytes(unsigned long long& free_bytes,
                              unsigned long long& total_bytes) {
#if defined(__APPLE__)
    // Unified memory: NVML/CUDA never exist here; mach statistics are the
    // device-wide truth.
    if (tryAppleUnified(free_bytes, total_bytes)) return true;
#endif
    // NVML first (true cross-process), then CUDA as a fallback.
    if (tryNvml(free_bytes, total_bytes)) return true;
    if (tryCuda(free_bytes, total_bytes)) return true;
    return false;
}

// ── System RAM + this process's footprint (HUD RAM meter) ───────────────────
bool querySystemMemoryBytes(unsigned long long& free_bytes,
                            unsigned long long& total_bytes) {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return false;
    total_bytes = ms.ullTotalPhys;
    free_bytes  = ms.ullAvailPhys;
    return total_bytes != 0;
#elif defined(__APPLE__)
    return tryAppleUnified(free_bytes, total_bytes);
#else
    // MemAvailable is the kernel's own "claimable without swapping" estimate
    // (free + reclaimable page cache + reclaimable slab).
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return false;
    unsigned long long total_kb = 0, avail_kb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        std::sscanf(line, "MemTotal: %llu", &total_kb);
        std::sscanf(line, "MemAvailable: %llu", &avail_kb);
    }
    std::fclose(f);
    if (total_kb == 0) return false;
    total_bytes = total_kb * 1024ull;
    free_bytes  = avail_kb * 1024ull;
    return true;
#endif
}

bool queryProcessMemoryBytes(unsigned long long& rss_bytes) {
#if defined(_WIN32)
    // PROCESS_MEMORY_COUNTERS, declared locally and resolved through
    // kernel32's K32GetProcessMemoryInfo so no psapi.h / psapi.lib
    // dependency is added.
    struct ProcMemCounters {
        DWORD  cb;
        DWORD  PageFaultCount;
        SIZE_T PeakWorkingSetSize;
        SIZE_T WorkingSetSize;
        SIZE_T QuotaPeakPagedPoolUsage;
        SIZE_T QuotaPagedPoolUsage;
        SIZE_T QuotaPeakNonPagedPoolUsage;
        SIZE_T QuotaNonPagedPoolUsage;
        SIZE_T PagefileUsage;
        SIZE_T PeakPagefileUsage;
    };
    using GetMemInfo_t = BOOL (WINAPI*)(HANDLE, ProcMemCounters*, DWORD);
    static GetMemInfo_t fn = []() -> GetMemInfo_t {
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        return k32 ? reinterpret_cast<GetMemInfo_t>(
                         GetProcAddress(k32, "K32GetProcessMemoryInfo"))
                   : nullptr;
    }();
    if (!fn) return false;
    ProcMemCounters pmc{};
    pmc.cb = sizeof(pmc);
    if (!fn(GetCurrentProcess(), &pmc, sizeof(pmc))) return false;
    rss_bytes = pmc.WorkingSetSize;
    return true;
#elif defined(__APPLE__)
    task_vm_info_data_t ti{};
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&ti), &cnt) != KERN_SUCCESS)
        return false;
    // phys_footprint is what Activity Monitor charges the process —
    // anonymous + compressed + IOKit (Metal/GPU) memory.
    rss_bytes = ti.phys_footprint;
    return true;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return false;
    unsigned long long pages_total = 0, pages_rss = 0;
    const int n = std::fscanf(f, "%llu %llu", &pages_total, &pages_rss);
    std::fclose(f);
    if (n != 2) return false;
    rss_bytes = pages_rss *
        static_cast<unsigned long long>(sysconf(_SC_PAGESIZE));
    return true;
#endif
}

}  // namespace engine
