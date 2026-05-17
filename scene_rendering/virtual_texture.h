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
//   kPoolWidth = kPoolHeight = 4096  →  64 × 64 = 4096 pages per layer.
//   kPageSize  = 64.
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

#include <algorithm>
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

// ── Compile-time constants — must match vt_types.glsl.h ───────────
// Tile size 64 (down from 128) gives 4× more slots per layer pool
// (4096 instead of 1024) at the same VRAM footprint.  64 stays a
// multiple of 4 (BC7 block alignment) and a multiple of any vendor
// CopyBufferToImage row-pitch granularity we'd reasonably hit.
// `phys_x`/`phys_y` are still 8-bit fields in packPageEntry (max
// 255), so 64 pages-across (max index 63) leaves plenty of headroom.
// kVtPageSize is the LOGICAL page size — the addressable content
// area each tile represents in the source texture.  Each physical
// pool slot is kVtTileSize = kVtPageSize + 2 * kVtTileBorder pixels
// wide, with the inner kVtPageSize×kVtPageSize block holding the
// page's content and the surrounding border holding pixels from
// neighboring tiles in the source texture.  See vtResolve in
// vt_sample.glsl.h for the corresponding GPU UV math.
constexpr uint32_t kVtPageSize    = 64u;
constexpr uint32_t kVtTileBorder  = 4u;
constexpr uint32_t kVtTileSize    = kVtPageSize + 2u * kVtTileBorder;   // 72
// Pool dimensions: 8208 × 4104 (114 × 57 slots) — closest tile-
// aligned dimensions to a "physical 8K × 4K" pool image.  6498
// slots/layer is ~1.59× the previous 4096 — large enough to hold
// the resident working set of Bistro at typical camera positions
// without LRU thrashing.  packPageEntry encodes phys_x/phys_y in
// 8-bit fields, so the cap is 256 slots per dimension; we stay
// well below that.
constexpr uint32_t kVtPagesAcross = 114u;
constexpr uint32_t kVtPagesDown   = 57u;
constexpr uint32_t kVtPagesPerLayer = kVtPagesAcross * kVtPagesDown;    // 6498
constexpr uint32_t kVtPoolWidth   = kVtPagesAcross * kVtTileSize;       // 8208
constexpr uint32_t kVtPoolHeight  = kVtPagesDown   * kVtTileSize;       // 4104
static_assert(kVtTileSize % 4 == 0,
              "kVtTileSize must be a multiple of 4 (BC7 block size).");
static_assert(kVtTileSize / 2 % 4 == 0,
              "kVtTileSize/2 must be a multiple of 4 (BC7 block size at pool mip 1).");
static_assert(kVtPagesAcross < 256u && kVtPagesDown < 256u,
              "phys_x/phys_y in packPageEntry are 8-bit; pool grid "
              "must stay below 256 in each dimension.");

// Hard cap on mip levels per VT — must equal VT_MAX_MIPS in
// vt_types.glsl.h.  At 64-px page size this covers source dimensions
// up to 64 << 7 = 8192 px; the largest texture in the engine's
// content set is 4096 px, so 7 mips would suffice and 8 leaves room
// for one larger asset class.
constexpr uint32_t kVtMaxMips = 8u;

// Pool image mip levels per slot.  Mip 0 holds the page at full
// 64×64 res; mip 1 holds the same page downsampled to 32×32.
// Combined with a LINEAR mipmap sampler this gives a smooth lerp
// within a VT-mip-k slot as the continuous LOD goes from k.0 to
// k.99, hiding the snap that would otherwise occur when LOD
// crosses an integer VT-mip boundary.  See virtual_texture.cpp
// processPendingWork / registerTextureFromImage for the upload
// patterns that fill both pool mips per page.
constexpr uint32_t kVtPoolMipLevels = 2u;

// Mip-aware page-grid helpers.  Mip 0 has (pages_x, pages_y) pages;
// each subsequent mip's grid is the CEIL-halve of the previous,
// floored at 1.  Ceil-halve (not floor) is required for non-power-
// of-2 mip-0 counts so every higher mip's grid still fully covers
// the downsampled source — e.g. pages_x_0 = 3 (a 192-px-wide
// source) needs ceil(3/2) = 2 pages at mip 1, not 3>>1 = 1.  The
// shader (vtMipPagesXY in vt_types.glsl.h) mirrors this exactly —
// both sides MUST agree for the page-table indexing math to line up.
//
// Bit form: ceil(p / 2^k) = (p + (1<<k) - 1) >> k, with a min of 1.
inline uint32_t vtMipPagesAt(uint32_t pages_dim_mip0, uint32_t mip) {
    if (pages_dim_mip0 == 0u) return 1u;
    const uint32_t shifted = (pages_dim_mip0 + ((1u << mip) - 1u)) >> mip;
    return shifted == 0u ? 1u : shifted;
}

// Number of mip levels needed for a VT whose mip-0 page grid is
// (pages_x, pages_y).  Stops at the smallest mip whose grid is
// 1×1 — going further would just store identical 1×1 page entries.
inline uint32_t vtComputeMipCount(uint32_t pages_x, uint32_t pages_y) {
    uint32_t max_pages = std::max(pages_x, pages_y);
    uint32_t mips      = 1u;
    while (max_pages > 1u && mips < kVtMaxMips) {
        max_pages = (max_pages + 1u) / 2u;   // ceil-half — match shader
        ++mips;
    }
    return mips;
}

// Total page-table entries (and pool slots, since each entry is
// backed by one slot) for a VT with the given mip-0 grid and mip
// count.  Sum over k of (pages_x>>k) * (pages_y>>k).
inline uint32_t vtTotalPagesAllMips(
    uint32_t pages_x, uint32_t pages_y, uint32_t mip_count) {
    uint32_t total = 0u;
    for (uint32_t k = 0u; k < mip_count; ++k) {
        total += vtMipPagesAt(pages_x, k) * vtMipPagesAt(pages_y, k);
    }
    return total;
}

// Page-table offset of a specific mip's window inside a VT's
// contiguous page-table region (relative to the VT's
// page_table_offset).  Matches vtMipPageTableOffset in
// vt_types.glsl.h.
inline uint32_t vtMipOffsetWithinVt(
    uint32_t pages_x, uint32_t pages_y, uint32_t mip) {
    uint32_t off = 0u;
    for (uint32_t k = 0u; k < mip; ++k) {
        off += vtMipPagesAt(pages_x, k) * vtMipPagesAt(pages_y, k);
    }
    return off;
}

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

// Page-table entry: physical page coords + residency bit.  8 bits
// is plenty for either coord (max 64 across × down at the current
// 4K pool / 64-px page sizing — leaves headroom for an 8K pool or
// 32-px pages without changing the bit layout), so pack everything
// into a u32.
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

    // ── Swap-chain teardown hook ───────────────────────────────────
    // compact_desc_set_ is allocated from the application's main
    // descriptor pool.  When the pool is destroyed in cleanupSwapChain
    // the handle becomes dangling — and tick() / compactFeedback()
    // run every frame writing to it, so the next frame after a resize
    // crashes inside vkCmdBindDescriptorSets / vkUpdateDescriptorSets.
    // Null the handle here so the lazy re-allocation in
    // recreateDescriptorSets can pick it up.
    void onDescriptorPoolDestroyed();

    // Re-allocate compact_desc_set_ from the fresh pool and re-write
    // the (stable) feedback-buffer bindings.  Called from the
    // application's recreateSwapChain after the new descriptor pool
    // has been built but before tick / compactFeedback fires.  No-op
    // if VT pipeline wasn't initialised (compact_desc_set_layout_
    // null), so callers don't need to guard.
    void recreateDescriptorSets(
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Register a single material with the VT system.  The four
    // source images (any of which may be null for "this material
    // doesn't have that layer") are uploaded into the matching layer
    // pool images at IDENTICAL physical slot positions across the
    // pools.  This is what makes the four pool layouts the same and
    // what lets a single vt_id resolve correctly across layers via
    // the per-layer pool samplers in the shader.
    //
    //   albedo_pixels:   required.  CPU RGBA8 source data, sliced
    //                    on the CPU into bordered tiles, BC7-encoded
    //                    via bc7enc, and uploaded to the BC7 ALBEDO
    //                    pool on demand.  No GPU readback.
    //   normal_image:    optional.  RGBA8 SHADER_READ_ONLY_OPTIMAL.
    //                    GPU-blitted into the NORMAL pool with the
    //                    same per-page layout.  Caller passes null
    //                    when the material has no normal map; the
    //                    shader keeps that layer's vt_id at INVALID
    //                    so it falls back to legacy bindless or the
    //                    flat tangent-up default.
    //   mr_ao_image,
    //   emissive_image:  same convention as normal_image.
    //   width, height:   shared by all four images.  All four
    //                    sources MUST match these dimensions —
    //                    mismatched sizes are a caller error in v1.
    //
    // Returns a vt_id valid for every PROVIDED layer (caller stores
    // it in the matching mp.*_vt_id field for layers it passed in,
    // and leaves the others at kInvalidVtId).  Returns kInvalidVtId
    // outright if the pool is full or any required input is missing
    // — the caller should then fall back entirely to legacy bindless.
    // Register a material with the VT manager.
    //   * albedo_pixels: PREFERRED.  CPU-side RGBA8 pixel data for
    //     the source albedo, width × height × 4 bytes, row-major.
    //     When provided, the VT manager builds its per-tile BC7
    //     cache directly from these CPU bytes — no GPU work to get
    //     RGBA8.  Caller's lifetime: the pointer must stay valid for
    //     the duration of this call only.
    //   * albedo_image: FALLBACK.  Used when albedo_pixels is null
    //     (typical for BC-compressed DDS sources where we don't
    //     have CPU-side RGBA8 around).  The VT manager does a one-
    //     time GPU blit from this image into a temporary RGBA8
    //     image (Vulkan's blit decompresses BC formats), reads it
    //     back, and uses those bytes as the source.
    //   * normal_image / mr_ao_image / emissive_image: GPU images
    //     (kept alive for streaming-time blits).
    //   * width / height: source albedo dimensions in texels.
    VirtualTextureId registerMaterial(
        const uint8_t* albedo_pixels,
        const std::shared_ptr<renderer::Image>& albedo_image,
        const std::shared_ptr<renderer::Image>& normal_image,
        const std::shared_ptr<renderer::Image>& mr_ao_image,
        const std::shared_ptr<renderer::Image>& emissive_image,
        uint32_t width,
        uint32_t height);

    // Bind handles for material descriptor sets.
    std::shared_ptr<renderer::ImageView> getPoolImageView(VtLayer layer) const;
    std::shared_ptr<renderer::Sampler>   getPoolSampler() const { return pool_sampler_; }
    // Bind handles for ImGui debug viewer.  Use the mip-0-only view
    // and the debug sampler so ImGui's quad-sampling at very high
    // LOD doesn't fall off the end of the multi-mip view's mip range
    // (which the trilinear pool view exposes only mip 0 + mip 1).
    std::shared_ptr<renderer::ImageView> getPoolDebugView(VtLayer layer) const;
    std::shared_ptr<renderer::Sampler>   getPoolDebugSampler() const { return pool_debug_sampler_; }
    std::shared_ptr<renderer::Buffer>    getPageTableBuffer() const { return page_table_buffer_; }
    std::shared_ptr<renderer::Buffer>    getMetaBuffer()      const { return meta_buffer_; }

    // Buffer byte sizes for descriptor writes.  Match what the
    // constructor passes to createBuffer — the values are derived from
    // the kMax* constants in virtual_texture.cpp; exposed here so
    // descriptor write sites don't have to depend on those internals.
    uint32_t getPageTableBufferBytes() const;
    uint32_t getMetaBufferBytes() const;

    // ── Per-slot status snapshot for the debug viewer ──────────────
    // Bit-encoded (one byte per slot, length kVtPagesPerLayer):
    //   bit 0 (0x01): slot is RESIDENT (has a tile cached in it)
    //   bit 1 (0x02): slot was ACCESSED by the GPU this frame
    //                 (its tile_key appeared in the feedback buffer)
    //   bit 2 (0x04): slot is PINNED (always-resident fallback tile)
    // Updated each call to tick() after the feedback dedupe.  Cells
    // are laid out row-major in (kVtPagesAcross × kVtPagesDown).
    // The menu's debug viewer reads this every frame and renders one
    // coloured rectangle per cell — orders of magnitude more useful
    // than the raw pool-image atlas for spotting "the camera is
    // looking at 200 tiles but the pool is sitting on 4090".
    static constexpr uint8_t kSlotStatusResident = 0x01;
    static constexpr uint8_t kSlotStatusActive   = 0x02;
    static constexpr uint8_t kSlotStatusPinned   = 0x04;
    const std::vector<uint8_t>& getSlotStatusGrid() const {
        return slot_status_;
    }
    uint32_t getSlotGridWidth()  const { return kVtPagesAcross; }
    uint32_t getSlotGridHeight() const { return kVtPagesDown; }
    // Counts derived from the most recent tick(): handy for a one-line
    // header above the grid.
    uint32_t getSlotsActive()   const { return slots_active_; }
    uint32_t getSlotsResident() const { return slots_resident_; }
    uint32_t getSlotsPinned()   const { return slots_pinned_; }

    // Streaming feedback buffer — bound by the cluster bindless pipeline
    // at PBR_MATERIAL_PARAMS_SET binding 10.  Cluster fragment shader
    // writes one tile-key per 8×8 screen block; CPU drains it in
    // tick() at frame end.  Sized for VT_FEEDBACK_PITCH² uints (1 MB),
    // covering up to a 4096-wide screen at block size 8.
    std::shared_ptr<renderer::Buffer> getFeedbackBuffer() const { return feedback_buffer_; }
    uint32_t getFeedbackBufferBytes() const;

    // Per-frame streamer.  CPU side: reads the GPU-compacted request
    // list (a tiny few-KB buffer the previous frame's compactFeedback
    // pass produced) and queues uploads.  Records all upload barriers
    // and copies into `cmd_buf` — no transient submit/wait, so this
    // call is essentially free CPU-side.
    //
    // Call AT FRAME START, BEFORE the cluster bindless draw of the
    // current frame.  The frame's normal submit will sequence the
    // upload commands ahead of the cluster sample.  Pair with a
    // matching compactFeedback() call AFTER the cluster bindless draw.
    void tick(const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
              uint64_t frame_index);

    // Dispatch the GPU compute pass that scans the 1 MB feedback
    // buffer, compacts non-invalid tile keys into the per-FIF compact
    // buffer slot at (frame_index % FIF), and clears the feedback
    // buffer for the next frame.  Replaces the previous CPU-side
    // 262144-entry walk + std::unordered_set dedupe at the start of
    // tick().  CPU still dedupes the much smaller compacted list
    // (typically a few thousand entries vs. 262144) — that part is
    // fast and friendly to the cache.
    //
    // Call AFTER the cluster bindless draw (so the shader's tile-key
    // writes are visible to the compaction pass) and BEFORE the next
    // frame's tick() / cluster draw (so the next frame sees the
    // cleared feedback buffer).  Records into `cmd_buf` — runs as
    // part of the frame's normal submit, no extra sync needed.
    void compactFeedback(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        uint64_t frame_index);

    // Push-constant / shader-side constants (matching vt_sample.glsl.h).
    static uint32_t poolWidth()  { return kVtPoolWidth; }
    static uint32_t poolHeight() { return kVtPoolHeight; }
    static uint32_t pageSize()   { return kVtPageSize; }

    // Tile-key encode/decode — must match vt_types.glsl.h.  See the
    // shader-side vtMakeTileKey / vtTileKey* helpers; both sides
    // share the bit layout.
    static constexpr uint32_t kVtTileKeyInvalid = 0xFFFFFFFFu;
    static constexpr uint32_t kVtFeedbackBlockSize = 8u;
    static constexpr uint32_t kVtFeedbackPitch     = 512u;     // covers up to 4096-wide screen
    static constexpr uint32_t kVtFeedbackTotal     = kVtFeedbackPitch * kVtFeedbackPitch;

    // ── GPU feedback compaction ────────────────────────────────────
    // Number of FIF slots in feedback_compact_buffer_.  Must equal
    // kMaxFramesInFlight in application.h — frame N's compaction
    // dispatch writes to slot (N % kVtCompactSlots), and tick()
    // (N + kVtCompactSlots) frames later reads from that same slot
    // after its fence has been waited on.  Hardcoded to 2 to match
    // the engine's current FIF; if FIF grows, bump this and verify.
    static constexpr uint32_t kVtCompactSlots = 2u;
    // Maximum compacted entries the GPU pass will write per slot
    // before clamping.  At a 4K screen with 8×8 feedback blocks
    // we have at most 32400 raw writes; after dedupe these typically
    // collapse to a few thousand unique tiles.  8K is a safe cap
    // that handles pathological camera spins; overflow is rare and
    // benign (extra requests dropped this frame, re-requested next).
    static constexpr uint32_t kVtCompactMaxEntries = 8192u;
    // Per-slot byte size: 1 counter + kVtCompactMaxEntries entries.
    static constexpr uint32_t kVtCompactSlotBytes =
        (1u + kVtCompactMaxEntries) * sizeof(uint32_t);
    static inline uint32_t makeTileKey(
        uint32_t vt_idx, uint32_t mip, uint32_t page_x, uint32_t page_y) {
        return  ( vt_idx & 0x3FFFu)
             | ((mip    & 0x000Fu) << 14)
             | ((page_x & 0x007Fu) << 18)
             | ((page_y & 0x007Fu) << 25);
    }
    static inline void decodeTileKey(uint32_t key,
                                     uint32_t& vt_idx, uint32_t& mip,
                                     uint32_t& page_x, uint32_t& page_y) {
        vt_idx  =  key         & 0x3FFFu;
        mip     = (key >> 14)  & 0x000Fu;
        page_x  = (key >> 18)  & 0x007Fu;
        page_y  = (key >> 25)  & 0x007Fu;
    }

private:
    // BC7 specialised registration: GPU readback the source RGBA8,
    // encode each (mip, page) to BC7 Mode 6 on the CPU, upload the
    // resulting blob into the ALBEDO pool via staging buffer.
    // Internally allocates pool slots from the SHARED allocator and
    // claims one contiguous page-table window — that allocation is
    // what registerMaterial reuses for the other layer uploads.
    // vkCmdCopyImage can't bridge RGBA8 (source) ↔ BC7 (pool dest)
    // directly, which is why ALBEDO needs its own CPU-encode path
    // separate from the GPU-blit path used for the other layers.
    VirtualTextureId registerAlbedoBC7(
        const std::shared_ptr<renderer::Image>& src_image,
        uint32_t width,
        uint32_t height);

    // Upload one non-ALBEDO layer's data to the slots already
    // allocated for an existing VT id.  Reads phys (X, Y) for every
    // entry from the page-table window the albedo pass populated,
    // then issues vkCmdBlitImage from `src_image` (RGBA8) into the
    // matching layer pool image (RGBA8) at those same slots.  Both
    // pool mip levels written per entry — see Stage 1.5 for why.
    void uploadLayerByBlitToSlots(
        VtLayer layer,
        const std::shared_ptr<renderer::Image>& src_image,
        uint32_t width,
        uint32_t height,
        uint32_t vt_index);

    // ── Streaming helpers (Phase B) ─────────────────────────────────
    // Pop a free slot, or evict the LRU back of resident_lru_ if no
    // free slots remain.  Returns kVtPagesPerLayer if even the LRU
    // back is pinned (pool is full of pinned slots — shouldn't
    // happen for sane pin counts but bail safely).  Caller is
    // responsible for filling the slot's data and calling
    // markSlotResident afterwards.
    uint32_t allocSlot();
    // Mark slot S as holding tile_key T at page-table entry E.
    // If `pinned` is true the slot is added straight to slot_info_
    // (not the LRU) so it can never be evicted.  Updates
    // tile_to_slot_ + page_table_cpu_ + GPU page table.
    void markSlotResident(uint32_t slot, uint32_t tile_key,
                          uint32_t entry_idx, bool pinned);
    // Move slot S to the LRU front (most-recently-used).  No-op if
    // pinned or free.
    void touchSlot(uint32_t slot);
    // Build the per-VT BC7 albedo cache + stash source images.
    // GPU-readback the source albedo, build mip pyramid, encode
    // every (mip, page) into vt_cache_[vt_index].bc7_albedo.  Does
    // NOT touch the pool image or allocate slots — those happen on
    // demand in tick().
    // Build the per-VT BC7 tile cache from CPU RGBA8 pixels.  No GPU
    // readback.  For every (mip k, page x, y) entry, extract a
    // kVtTileSize × kVtTileSize block of the source mip with proper
    // edge clamping (the inner kVtPageSize × kVtPageSize is the
    // page's content; the surrounding kVtTileBorder pixels come from
    // neighboring tiles in the source mip), bc7enc encode it, and
    // also encode the half-resolution version for pool mip 1.  All
    // tiles for all mips of this VT end up in vt_cache_[vt_index].
    // bc7_albedo, contiguous in entry-index order.
    void encodeAndCacheVt(
        uint32_t vt_index,
        const uint8_t* albedo_pixels,
        const std::shared_ptr<renderer::Image>& albedo_image,
        const std::shared_ptr<renderer::Image>& normal_image,
        const std::shared_ptr<renderer::Image>& mr_ao_image,
        const std::shared_ptr<renderer::Image>& emissive_image,
        uint32_t width, uint32_t height,
        uint32_t pages_x, uint32_t pages_y, uint32_t mip_count,
        uint32_t page_table_offset);

    // Decompress an arbitrary-format GPU image into a CPU RGBA8
    // buffer.  Uses vkCmdBlitImage to decompress BC formats into a
    // temporary RGBA8 image (the driver handles BC1/BC3/BC5/BC7 decode
    // implicitly during the blit), then copyImageToBuffer to read the
    // RGBA8 bytes back.  Returns the bytes; one-time cost paid only
    // when registerMaterial is called without a CPU pixel buffer.
    //
    //   dst_is_srgb:
    //     true  → temp image is R8G8B8A8_SRGB.  The blit's source
    //             sRGB→linear decode is canceled by the dest's
    //             linear→sRGB encode, so the readback bytes are the
    //             SAME bytes that were originally stored in the BC*
    //             source.  Use this for ALBEDO (sRGB-encoded data
    //             that bc7enc must encode opaquely so the BC7_SRGB
    //             pool's sample-time decode produces correct linear
    //             values).
    //     false → temp image is R8G8B8A8_UNORM.  The blit reads source
    //             as linear and writes raw bytes — preserves the
    //             linear values from any UNORM-source format (BC5
    //             normal maps, BC7_UNORM mr_ao, etc.).  Use this for
    //             NORMAL / METAL_ROUGH_AO / EMISSIVE source images
    //             whose values are already linear.
    std::vector<uint8_t> readbackImageToRgba8(
        const std::shared_ptr<renderer::Image>& src_image,
        uint32_t width, uint32_t height,
        bool     dst_is_srgb);
    // Upload a single tile (vt_index, mip, page_x, page_y) into
    // pool slot `slot` across all four layer pools.  Reads BC7 from
    // the per-VT cache for ALBEDO, blits from cached source images
    // for the other layers.  Issues all work on the supplied
    // command buffer; caller is responsible for surrounding image
    // barriers (pool to TRANSFER_DST, back to SHADER_READ).
    //   * staging_slot: index in [0, kStreamerUploadsPerFrame).  The
    //     ALBEDO BC7 bytes get written to byte offset
    //     staging_slot * kBc7BytesPerEntry inside the persistent
    //     upload_staging_buffer_.  Caller is responsible for
    //     allocating one staging slot per upload in the current tick.
    void uploadTileAllLayers(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        uint32_t slot, uint32_t vt_index,
        uint32_t mip, uint32_t page_x, uint32_t page_y,
        uint32_t staging_slot);

    std::shared_ptr<renderer::Device>         device_;
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool_;

    // Per-layer physical pool textures.  Indexed by VtLayer enum.
    struct LayerPool {
        renderer::TextureInfo texture;
        // Single-mip-0 view of the pool, used by external consumers
        // (e.g. the ImGui debug viewer) that don't want the trilinear
        // pool-mip-0/1 lerp behaviour of the multi-mip view.
        std::shared_ptr<renderer::ImageView> mip0_view;
        // No per-layer sampler — pool_sampler_ is shared.
    };
    LayerPool                            layer_pools_[uint32_t(VtLayer::COUNT)];
    std::shared_ptr<renderer::Sampler>   pool_sampler_;
    // Companion sampler used only by the debug viewer: NEAREST mipmap
    // mode + maxLod = 0, so ImGui samples a fixed mip without any
    // implicit LOD selection that could land outside the view's mip
    // range and return black.
    std::shared_ptr<renderer::Sampler>   pool_debug_sampler_;

    // Global page table.  One u32 per virtual page across all
    // registered VTs, indexed by VirtualTextureMeta::page_table_offset
    // + page_y * pages_x + page_x.  HOST_VISIBLE so registerTexture()
    // can write entries directly without a staging copy.
    std::shared_ptr<renderer::Buffer>            page_table_buffer_;
    std::shared_ptr<renderer::DeviceMemory>      page_table_memory_;
    std::vector<uint32_t>                        page_table_cpu_;

    // Per-VT metadata (one entry per registerMaterial call).
    std::shared_ptr<renderer::Buffer>            meta_buffer_;
    std::shared_ptr<renderer::DeviceMemory>      meta_memory_;
    std::vector<VirtualTextureMeta>              meta_cpu_;

    // Persistent CPU mappings of the two HOST_VISIBLE | HOST_COHERENT
    // SSBOs.  Repeatedly calling mapMemory(…, 4, entry_idx*4) per
    // page-table entry was unsafe — Vulkan implementations have
    // non-trivial alignment requirements (minMemoryMapAlignment,
    // typically 64 B), and on some drivers the per-entry map/unmap
    // pattern corrupts the mapping state.  We map once at
    // construction, write through the pointer for every update, and
    // unmap at destroy().  HOST_COHERENT means writes are visible to
    // the GPU on the next vkQueueSubmit.
    //
    // page_table_mapped_ is only used by registerMaterial (synchronous,
    // pre-render).  Per-frame tick() updates go through
    // page_table_dirty_entries_ + vkCmdUpdateBuffer so they're properly
    // sequenced with cmd_buf execution and don't race in-flight
    // previous-frame shader reads.
    uint32_t*                                    page_table_mapped_ = nullptr;
    VirtualTextureMeta*                          meta_mapped_       = nullptr;

    // ── Per-frame dirty page-table entries ───────────────────────────
    // Every tick() call that evicts a slot or marks a new tile
    // resident appends (entry_idx, packed_value) here on the CPU
    // side; tick() then drains the vector by recording one
    // vkCmdUpdateBuffer per entry into the frame's main command
    // buffer (with surrounding TRANSFER → SHADER_READ barriers).
    //
    // Why this is necessary: before this fix, those updates went
    // through page_table_mapped_ (HOST_COHERENT), which made them
    // visible to the GPU IMMEDIATELY — including to the previous
    // frame's still-executing cluster-bindless draw.  Result: that
    // frame would read "tile T is now resident at slot S" while
    // slot S still held the OLD tile's data (this frame's
    // copyBufferToImage hadn't run yet) → the wrong texture
    // appeared for one or two frames → visible flicker.
    struct PageTableDirty {
        uint32_t entry_idx;
        uint32_t packed_value;
    };
    std::vector<PageTableDirty> page_table_dirty_entries_;

    // Apply all queued dirty page-table entries via a TRANSIENT
    // command buffer + submit-and-wait, so by the time this returns
    // the GPU's page_table_buffer_ reflects every entry recently
    // written by allocSlot/markSlotResident.  Used by registerMaterial
    // — it's synchronous (no concurrent rendering of these new
    // entries), so blocking on a transient submit is fine and is
    // exactly what we need to avoid the "one frame of magenta /
    // pinned-mip fallback right after a material loads" symptom.
    //
    // Per-frame tick() does NOT call this; it drains the same
    // dirty queue via vkCmdUpdateBuffer into the frame's MAIN cmd
    // buffer instead, sequenced with pool uploads + barriers.
    void flushDirtyPageTableEntriesViaTransient();

    // ── Streaming feedback ──────────────────────────────────────────
    // GPU-only SSBO; cluster_bindless.frag writes one tile-key per
    // 8×8 screen block.  No host visibility — the CPU never reads
    // this buffer directly anymore (replaced by the GPU compaction
    // pass below).  Cleared via vkCmdFillBuffer to kVtTileKeyInvalid
    // by compactFeedback() at end-of-frame, after the compaction
    // pass has read it.
    std::shared_ptr<renderer::Buffer>            feedback_buffer_;
    std::shared_ptr<renderer::DeviceMemory>      feedback_memory_;

    // ── GPU feedback compaction output ─────────────────────────────
    // HOST_VISIBLE | HOST_COHERENT.  Sized for kVtCompactSlots *
    // kVtCompactSlotBytes — one slot per frame-in-flight, addressed
    // by frame_index % kVtCompactSlots.  Per slot:
    //   uint count
    //   uint entries[kVtCompactMaxEntries]
    // The compute pass atomicAdd's `count` and writes entries; the
    // CPU side of tick() reads min(count, kVtCompactMaxEntries) keys
    // and dedupes them into upload requests.  Replaces the previous
    // mapMemory + 262144-entry scan + std::unordered_set dedupe at
    // the head of tick(), which dominated VT's CPU cost.
    std::shared_ptr<renderer::Buffer>            feedback_compact_buffer_;
    std::shared_ptr<renderer::DeviceMemory>      feedback_compact_memory_;
    uint32_t*                                    feedback_compact_mapped_ = nullptr;

    // Compute pipeline + descriptor set for vt_compact_feedback.comp.
    // Single descriptor set bound for every dispatch (the buffer
    // bindings don't change frame-to-frame); the per-frame slot
    // selection is via push constant `compact_slot_offset_uints`.
    std::shared_ptr<renderer::DescriptorSetLayout> compact_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>       compact_desc_set_;
    std::shared_ptr<renderer::PipelineLayout>      compact_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            compact_pipeline_;
    void initFeedbackCompactPipeline();

    // ── Streaming: LRU pool slot allocator ──────────────────────────
    // Slots can be: free (in free_slots_), resident-evictable (in
    // resident_lru_), or resident-pinned (smallest mip per VT —
    // never evicted, guarantees the shader's mip-walk fallback
    // always lands on something).  allocSlot pops free, then evicts
    // LRU back if necessary; touchSlot moves to LRU front.
    struct SlotInfo {
        // INVALID = slot is free.  Otherwise = the tile-key currently
        // resident in this slot, used to back-update tile_to_slot_
        // and the page-table entry on eviction.
        uint32_t tile_key  = kVtTileKeyInvalid;
        // Page-table index for the resident tile — saved here so
        // eviction doesn't need to re-derive it from the tile_key.
        uint32_t entry_idx = 0xFFFFFFFFu;
        bool     pinned    = false;
        std::list<uint32_t>::iterator lru_iter;
    };
    std::vector<SlotInfo>           slot_info_;          // [kVtPagesPerLayer]

    // Debug-viewer slot-activity snapshot (see getSlotStatusGrid).
    std::vector<uint8_t>            slot_status_;
    uint32_t                        slots_active_   = 0;
    uint32_t                        slots_resident_ = 0;
    uint32_t                        slots_pinned_   = 0;
    std::vector<uint32_t>           free_slots_;
    std::list<uint32_t>             resident_lru_;       // front=newest, back=oldest
    std::unordered_map<uint32_t, uint32_t> tile_to_slot_; // key → slot

    // Per-VT pre-encoded cache.  Built once at registerMaterial,
    // consumed by the streamer's per-tile uploads.  Albedo holds the
    // BC7 mip chain for all pool slots (kBc7BytesPerEntry per entry,
    // entries laid out same order as the page-table window).  Source
    // images for the other layers are kept alive on the GPU and
    // blitted into pool slots on demand.
    struct VtCacheEntry {
        uint32_t width   = 0;
        uint32_t height  = 0;
        uint32_t pages_x = 0;
        uint32_t pages_y = 0;
        uint32_t mip_count = 0;
        uint32_t page_table_offset = 0;
        // ALBEDO is BC7-encoded (via bc7enc) into bc7_albedo at
        // registerMaterial time; uploadTileAllLayers feeds it to the
        // BC7_SRGB pool slot via copyBufferToImage.  The other three
        // layers stay on the GPU and are blitted from their kept-
        // alive source images (no readback / encode).  albedo_src
        // is kept around for diagnostic format swaps but is unused
        // by the live BC7 path.
        std::vector<uint8_t> bc7_albedo;
        // Per-VT BC5 cache for the NORMAL layer.  Same per-tile entry
        // size as bc7_albedo (kBc7BytesPerEntry = 6480 B) — BC5 and
        // BC7 are both 16 B per 4×4 block, so the layout math is
        // identical and we can reuse the same constants for offset
        // arithmetic in uploadTileAllLayers.  Built at registerMaterial
        // time from the normal source's RGBA8 readback (only RG used).
        std::vector<uint8_t> bc5_normal;
        std::shared_ptr<renderer::Image> albedo_src;
        std::shared_ptr<renderer::Image> normal_src;
        std::shared_ptr<renderer::Image> mr_ao_src;
        std::shared_ptr<renderer::Image> emissive_src;
    };
    std::vector<VtCacheEntry>       vt_cache_;           // by vt_index

    // Streamer per-frame upload budget.  Bumped from 32 to 128 once
    // tick() stopped doing a synchronous transient submit-and-wait —
    // upload commands now ride on the frame's main command buffer
    // and their cost is amortized by the frame's normal submit, so
    // there's no longer a per-tile CPU stall and we can stream the
    // working set in faster after a camera move.
    static constexpr uint32_t kStreamerUploadsPerFrame = 128u;

    // Persistent staging buffer for streamed tile uploads.  Sized to
    // kStreamerUploadsPerFrame * kBc7BytesPerEntry; one HOST_VISIBLE +
    // HOST_COHERENT allocation made at construction, mapped once and
    // never unmapped.  Each tick uses the first N tile slots (where N
    // = number of uploads queued this frame) — slot i gets a known
    // offset i * kBc7BytesPerEntry, the CPU memcpy's the BC7 bytes
    // there, then copyBufferToImage uses (offset, offset+kBc7BytesMip0)
    // for mip 0 and mip 1.  Replaces the per-tile vkCreateBuffer +
    // vkAllocateMemory pair that used to dominate streaming overhead.
    std::shared_ptr<renderer::Buffer>       upload_staging_buffer_;
    std::shared_ptr<renderer::DeviceMemory> upload_staging_memory_;
    uint8_t*                                upload_staging_mapped_ = nullptr;
    // Byte offset within upload_staging_buffer_ at which THIS frame's
    // FIF slice begins.  Set by tick() for the duration of a frame's
    // upload pass; uploadTileAllLayers adds it to its per-tile slot
    // offset to land in the right slice.  Without this offset,
    // back-to-back frames would race: frame N+1's CPU memcpy would
    // overwrite frame N's still-being-read staging bytes.
    uint64_t                                active_staging_base_off_ = 0;

    // BC7 byte sizes used by the cache + tile-upload paths.  Each pool
    // slot holds a kVtTileSize × kVtTileSize block (content + border)
    // at mip 0 and a half-resolution version at mip 1.
    //   72×72 BC7 mip 0 = 18×18 blocks × 16 B = 5184 B per tile
    //   36×36 BC7 mip 1 =  9×9  blocks × 16 B = 1296 B per tile
    //   Total BC7 per tile = 6480 B
    static constexpr uint32_t kBc7BytesMip0     = (kVtTileSize / 4u) * (kVtTileSize / 4u) * 16u;
    static constexpr uint32_t kBc7BytesMip1     = ((kVtTileSize / 2u) / 4u) * ((kVtTileSize / 2u) / 4u) * 16u;
    static constexpr uint32_t kBc7BytesPerEntry = kBc7BytesMip0 + kBc7BytesMip1;
    // Tail pointer into page_table_cpu_ for the next registerTexture
    // call's contiguous window.  v1: monotonic; v2: free-list.
    uint32_t next_page_table_tail_ = 0;

    // ── CPU thread pool for parallel BC7 encoding ──────────────────
    // Sized to hardware_concurrency().  Owned by the manager so its
    // lifetime matches.  Used inside registerAlbedoBC7 to encode
    // multiple pages concurrently — a 1024² albedo has 256 pages
    // (16×16) at the current 64-px page size, so an 8-core machine
    // processes ~8× faster than serial encoding.
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
        uint32_t width;            // mip-0 source width
        uint32_t height;           // mip-0 source height
        uint32_t table_offset;     // start of mip 0's window inside the
                                   // VT's contiguous page-table region
        uint32_t pages_x;          // mip-0 page grid
        uint32_t pages_y;          //
        uint32_t mip_count;        // total mip levels in this VT's chain
        // Per-PAGE allocated phys coords / residency flags / mip level.
        // Length = vtTotalPagesAllMips(pages_x, pages_y, mip_count).
        // Layout: mip 0's pages first (pages_x * pages_y entries), then
        // mip 1 ((pages_x>>1) * (pages_y>>1)), … same as page-table
        // windowing — so phys_*[i] corresponds to the i-th entry in
        // the per-VT page-table region.  The `mip` array stores which
        // mip level each entry belongs to so processPendingWork knows
        // which downsampled image to encode from.
        std::vector<uint32_t> phys_x;
        std::vector<uint32_t> phys_y;
        std::vector<uint8_t>  mip;    // 0..mip_count-1 per entry
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
