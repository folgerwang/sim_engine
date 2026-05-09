#pragma once
//
// virtual_texture.h — Runtime Virtual Texture (RVT) system.
//
// Why: large scenes (Bistro, Sponza, etc.) have hundreds of unique
// material textures.  Binding them all simultaneously is impractical
// (descriptor budget) and uploading them all to a single bindless
// array uses huge amounts of VRAM for textures that may not even be
// visible from the camera.  RVT solves both:
//
//   * Per-layer physical pool textures (one each for albedo / normal /
//     metallic-roughness-AO / emissive) hold a fixed pool of 128×128
//     PAGES.  All four pools share the same page-coord layout, so a
//     single allocation moves all four channels in lockstep.
//
//   * A page table SSBO maps virtual page IDs (mesh × material × mip ×
//     page-x × page-y) to physical pool slots.  Sampling shaders
//     translate UVs through this table at runtime.
//
//   * Streaming (v2): fragment shaders write the page IDs they wanted
//     into a feedback buffer.  CPU reads it, evicts LRU pages, uploads
//     fresh ones via async queue.  v1 is "eager-resident" — the pool
//     must be large enough to hold every page of every registered
//     mesh.
//
// Layer formats:
//   ALBEDO         — RGBA8 SRGB
//   NORMAL         — RG8  UNORM   (z reconstructed in shader from xy)
//   METAL_ROUGH_AO — RGB8 UNORM packed into RGBA8
//   EMISSIVE       — RGBA8 UNORM
//
// Pool dimensions:
//   kPoolWidth = kPoolHeight = 4096  →  32 × 32 = 1024 pages per layer.
//   kPageSize  = 128.
//
// A VirtualTextureId is an opaque 32-bit handle:
//   bits  0..29  virtual_texture_index   (1 billion textures)
//   bits 30..31  layer (0=albedo, 1=normal, 2=mr_ao, 3=emissive)
//
// Public API on the manager:
//   registerTexture(layer, src_image, w, h)  → returns a VirtualTextureId
//   getPageTableBuffer()                     → bind to material descriptors
//   getPoolImageView(layer)                  → bind to material descriptors
//
// All four pool textures live in GENERAL layout so storage-image writes
// (when a streaming page upload uses vkCmdCopyImage to a sub-region)
// don't need per-frame transitions.
//

#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"
#include "helper/thread_pool.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace engine {
namespace scene_rendering {

// ── Compile-time constants — must match vt_sample.glsl.h ───────────
constexpr uint32_t kVtPageSize  = 128u;
constexpr uint32_t kVtPoolWidth = 4096u;          // 32 pages across
constexpr uint32_t kVtPoolHeight = 4096u;          // 32 pages down
constexpr uint32_t kVtPagesAcross = kVtPoolWidth  / kVtPageSize;
constexpr uint32_t kVtPagesDown   = kVtPoolHeight / kVtPageSize;
constexpr uint32_t kVtPagesPerLayer = kVtPagesAcross * kVtPagesDown;

enum class VtLayer : uint32_t {
    ALBEDO         = 0,
    NORMAL         = 1,
    METAL_ROUGH_AO = 2,
    EMISSIVE       = 3,
    COUNT          = 4,
};

// 32-bit packed virtual-texture handle.  Layout above.
using VirtualTextureId = uint32_t;
constexpr VirtualTextureId kInvalidVtId = 0xFFFFFFFFu;

inline VirtualTextureId makeVtId(VtLayer layer, uint32_t vt_index) {
    return (static_cast<uint32_t>(layer) << 30) | (vt_index & 0x3FFFFFFFu);
}
inline VtLayer vtLayer(VirtualTextureId id) {
    return static_cast<VtLayer>((id >> 30) & 0x3u);
}
inline uint32_t vtIndex(VirtualTextureId id) {
    return id & 0x3FFFFFFFu;
}

// One entry per registered virtual texture — describes its dimensions
// in pages and where its page-table window starts in the global page
// table buffer.  Indexed by vtIndex(id).
struct VirtualTextureMeta {
    uint32_t width_px;       // source texture width in pixels
    uint32_t height_px;      // source texture height in pixels
    uint32_t pages_x;        // ceil(width / kVtPageSize)
    uint32_t pages_y;        // ceil(height / kVtPageSize)
    uint32_t page_table_offset;  // first slot in page_table_ for this VT
    uint32_t mip_count;      // 1 in v1 (no mip support yet)
    uint32_t pad0;
    uint32_t pad1;
};
static_assert(sizeof(VirtualTextureMeta) == 32, "match vt_sample.glsl.h");

// Page-table entry: physical page coords + residency bit.  16 bits
// is plenty for either coord (max 32 across × down at 4K pool), so
// pack everything into a u32.
//
//   bits  0..7   phys_x (page coord, 0..255)
//   bits  8..15  phys_y (page coord, 0..255)
//   bits 16      resident (1 = mapped)
//   bits 17..31  reserved (LRU age, format flags, etc. for v2)
inline uint32_t packPageEntry(uint32_t phys_x, uint32_t phys_y, bool resident) {
    return (phys_x & 0xFFu) | ((phys_y & 0xFFu) << 8u) |
           (resident ? (1u << 16u) : 0u);
}
constexpr uint32_t kPageEntryUnresident = 0u;

class VirtualTextureManager {
public:
    VirtualTextureManager(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    void destroy();

    // Register a CPU-side texture's pixels with the VT system.  The
    // pixels are immediately sliced into 128×128 pages and copied to
    // the appropriate layer pool (eager-resident in v1).  Returns a
    // VirtualTextureId the caller stores in their material struct.
    //
    //   layer      — which layer pool to use.
    //   pixels     — tightly-packed RGBA8 pixels (4 bytes per texel).
    //                Other layer formats convert internally.
    //   width      — source texture width in pixels.
    //   height     — source texture height in pixels.
    //
    // Returns kInvalidVtId on out-of-pool-space (caller should fall
    // back to a default texture).  v2 will instead enqueue eviction.
    VirtualTextureId registerTexture(
        VtLayer layer,
        const uint8_t* pixels,
        uint32_t width,
        uint32_t height);

    // Same as above, but the source pixels live on the GPU (already
    // uploaded as a sampled image during mesh load).  This is the path
    // used by the existing material-upload code: the loader has long
    // since freed the CPU-side blob, but the GPU image is alive in
    // SHADER_READ_ONLY_OPTIMAL layout.  We do GPU→GPU vkCmdCopyImage
    // straight into the pool's sub-regions, avoiding any CPU bounce.
    //
    //   src_image, src_layout — must be SHADER_READ_ONLY_OPTIMAL on
    //                           entry.  The image is restored to that
    //                           layout on return (we transition it to
    //                           TRANSFER_SRC and back internally).
    //                           Must have been created with
    //                           TRANSFER_SRC_BIT usage; loader-created
    //                           textures do via createTextureImage.
    //
    // Format mismatch (e.g. source RGBA8 → pool RG8 for normals) is
    // handled by reading only the channels Vulkan-blits naturally;
    // for Layer::NORMAL the .b/.a are dropped on the GPU via the
    // ImageBlit's destination format, no CPU work needed.
    VirtualTextureId registerTextureFromImage(
        VtLayer layer,
        const std::shared_ptr<renderer::Image>& src_image,
        uint32_t width,
        uint32_t height);

    // Bind handles for material descriptor sets.
    std::shared_ptr<renderer::ImageView> getPoolImageView(VtLayer layer) const;
    std::shared_ptr<renderer::Sampler>   getPoolSampler() const { return pool_sampler_; }
    std::shared_ptr<renderer::Buffer>    getPageTableBuffer() const { return page_table_buffer_; }
    std::shared_ptr<renderer::Buffer>    getMetaBuffer()      const { return meta_buffer_; }

    // Buffer byte sizes for descriptor writes.  Match what the
    // constructor passes to createBuffer — the values are derived from
    // the kMax* constants in virtual_texture.cpp; exposed here so
    // descriptor write sites don't have to depend on those internals.
    uint32_t getPageTableBufferBytes() const;
    uint32_t getMetaBufferBytes() const;

    // Push-constant / shader-side constants (matching vt_sample.glsl.h).
    static uint32_t poolWidth()  { return kVtPoolWidth; }
    static uint32_t poolHeight() { return kVtPoolHeight; }
    static uint32_t pageSize()   { return kVtPageSize; }

private:
    // Convert source RGBA8 pixels into the layer's native channel layout
    // and write a single page (128×128 sub-region) into the pool.
    void uploadPage(
        VtLayer layer,
        uint32_t phys_x,
        uint32_t phys_y,
        const uint8_t* page_pixels,   // 128×128 × stride bytes
        uint32_t src_stride_bytes);   // bytes per row of source

    // BC7 specialised registration: GPU readback the source RGBA8,
    // encode each page to BC7 Mode 6 on the CPU, upload the BC7 blob
    // into the ALBEDO pool via staging buffer.  Used by registerTextureFromImage
    // when layer == ALBEDO since vkCmdCopyImage can't convert source
    // RGBA8 → destination BC7_SRGB_BLOCK directly.
    VirtualTextureId registerAlbedoBC7(
        const std::shared_ptr<renderer::Image>& src_image,
        uint32_t width,
        uint32_t height);

    // Allocate the next free page slot.  v1 = monotonic counter; v2 =
    // LRU eviction.  Returns kVtPagesPerLayer when full.
    uint32_t allocatePageSlot();

    std::shared_ptr<renderer::Device>         device_;
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool_;

    // Per-layer physical pool textures.  Indexed by VtLayer enum.
    struct LayerPool {
        renderer::TextureInfo texture;
        // No per-layer sampler — pool_sampler_ is shared.
    };
    LayerPool                            layer_pools_[uint32_t(VtLayer::COUNT)];
    std::shared_ptr<renderer::Sampler>   pool_sampler_;

    // Global page table.  One u32 per virtual page across all
    // registered VTs, indexed by VirtualTextureMeta::page_table_offset
    // + page_y * pages_x + page_x.  HOST_VISIBLE so registerTexture()
    // can write entries directly without a staging copy.
    std::shared_ptr<renderer::Buffer>            page_table_buffer_;
    std::shared_ptr<renderer::DeviceMemory>      page_table_memory_;
    std::vector<uint32_t>                        page_table_cpu_;

    // Per-VT metadata (one entry per registerTexture call).
    std::shared_ptr<renderer::Buffer>            meta_buffer_;
    std::shared_ptr<renderer::DeviceMemory>      meta_memory_;
    std::vector<VirtualTextureMeta>              meta_cpu_;

    // Free-page allocator.  v1: monotonic.  v2: replace with LRU.
    uint32_t next_free_slot_ = 0;
    // Tail pointer into page_table_cpu_ for the next registerTexture
    // call's contiguous window.  v1: monotonic; v2: free-list.
    uint32_t next_page_table_tail_ = 0;

    // ── CPU thread pool for parallel BC7 encoding ──────────────────
    // Sized to hardware_concurrency().  Owned by the manager so its
    // lifetime matches.  Used inside registerAlbedoBC7 to encode
    // multiple pages concurrently — a 1024² albedo has 64 pages, so
    // an 8-core machine processes ~8× faster than serial encoding.
    std::unique_ptr<engine::helper::ThreadPool> encode_pool_;

    // ── Async streaming worker ──────────────────────────────────────
    // Single dedicated thread that drains a request queue.  Async
    // semantics: registerTextureFromImage() pre-allocates the VT id +
    // page-table window on the calling thread (synchronous bookkeeping
    // — fast), pushes the GPU readback / encode / upload work onto
    // pending_work_, then returns.  The worker thread later does the
    // heavy lifting (parallel BC7 encode via encode_pool_) and flips
    // page-table entries to RESIDENT once each page is uploaded.
    //
    // Until that flip, the shader sees pages as not-resident and uses
    // its fallback (default colour for albedo, up-vector for normal).
    // From the user's perspective: textures fade in over a few frames
    // post-scene-load instead of blocking the load screen.
    //
    // The worker registers itself as the device's loader thread on
    // startup, so its setupTransientCommandBuffer() calls are routed
    // to the loader queue (separate from main thread's compute queue),
    // avoiding cross-thread Vulkan submission races.
    struct PendingWork {
        VtLayer  layer;
        std::shared_ptr<renderer::Image> src_image;
        uint32_t width;
        uint32_t height;
        uint32_t table_offset;   // pre-allocated page-table window start
        uint32_t pages_x;
        uint32_t pages_y;
        // Per-page allocated phys coords (one entry per page in the
        // window).  Saved here so the worker doesn't need to re-run
        // the slot allocator (which isn't thread-safe).
        std::vector<uint32_t> phys_x;
        std::vector<uint32_t> phys_y;
        std::vector<uint8_t>  ok;     // 1 if slot was allocated, 0 if pool full
    };
    std::thread             worker_thread_;
    std::queue<PendingWork> pending_work_;
    std::mutex              work_mtx_;
    std::condition_variable work_cv_;
    std::atomic<bool>       worker_should_stop_{false};

    void workerThreadLoop();
    void processPendingWork(PendingWork&& w);
};

}  // namespace scene_rendering
}  // namespace engine
