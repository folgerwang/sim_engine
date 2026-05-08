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

#include <cstdint>
#include <memory>
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

    // Bind handles for material descriptor sets.
    std::shared_ptr<renderer::ImageView> getPoolImageView(VtLayer layer) const;
    std::shared_ptr<renderer::Sampler>   getPoolSampler() const { return pool_sampler_; }
    std::shared_ptr<renderer::Buffer>    getPageTableBuffer() const { return page_table_buffer_; }
    std::shared_ptr<renderer::Buffer>    getMetaBuffer()      const { return meta_buffer_; }

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
};

}  // namespace scene_rendering
}  // namespace engine
