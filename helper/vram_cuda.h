#pragma once

namespace engine {

// Device-wide GPU memory, resolved by DYNAMIC LOAD at runtime (no link-time
// dependency and no vendor headers required).  Returns true on success with
// byte counts for the primary GPU.
//
// Prefers NVML (nvmlDeviceGetMemoryInfo) — the same source nvidia-smi uses,
// which reports the board's PHYSICAL usage across ALL processes.  This matters
// on Windows: under the WDDM memory model cudaMemGetInfo only reflects (roughly)
// the calling process, so a separate process such as the FLUX.2 Python
// generator would be invisible to it.  Falls back to cudaMemGetInfo when NVML
// is unavailable.
//
// Unlike the Vulkan VK_EXT_memory_budget query (which only sees THIS process's
// Vulkan allocations), this reflects every VRAM consumer on the card: the
// engine's Vulkan + LibTorch tensors, plus the FLUX.2 / Ollama subprocesses.
//
// Returns false if no NVIDIA driver/runtime is present (the caller should then
// fall back to the Vulkan-only numbers).
bool queryDeviceWideVramBytes(unsigned long long& free_bytes,
                              unsigned long long& total_bytes);

}  // namespace engine
