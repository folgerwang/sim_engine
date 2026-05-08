//
// virtual_texture.cpp — RVT pool + page-table allocator (v1).
//

#include "virtual_texture.h"
#include "renderer/renderer_helper.h"

#include <cassert>
#include <cstring>

namespace er = engine::renderer;

namespace engine {
namespace scene_rendering {

// Capacity assumed at startup.  Pre-allocate page-table + meta buffers
// for this many entries; can be raised if we hit it.  Each VT entry
// costs 32 B (meta) and pages_x*pages_y*4 B (page table); typical 1024
// VTs averaging 16 pages each ≈ 64 KB meta + 64 KB page table — tiny.
static constexpr uint32_t kMaxVirtualTextures   = 1024u;
static constexpr uint32_t kMaxPageTableEntries  = 1024u * 1024u;  // ~4 MB

static er::Format layerFormat(VtLayer layer) {
    // Match comments in virtual_texture.h:
    //   ALBEDO         — RGBA8 SRGB        (gamma-corrected sampling)
    //   NORMAL         — RG8  UNORM        (z reconstructed in shader)
    //   METAL_ROUGH_AO — RGBA8 UNORM       (.a unused, reserved for IOR/specular)
    //   EMISSIVE       — RGBA8 UNORM       (linear; tone-mapped at output)
    switch (layer) {
        case VtLayer::ALBEDO:         return er::Format::R8G8B8A8_SRGB;
        case VtLayer::NORMAL:         return er::Format::R8G8_UNORM;
        case VtLayer::METAL_ROUGH_AO: return er::Format::R8G8B8A8_UNORM;
        case VtLayer::EMISSIVE:       return er::Format::R8G8B8A8_UNORM;
        default:                      return er::Format::R8G8B8A8_UNORM;
    }
}

VirtualTextureManager::VirtualTextureManager(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool)
    : device_(device)
    , descriptor_pool_(descriptor_pool) {

    // ── 1. Pool textures (one per layer) ───────────────────────────
    // GENERAL layout throughout — sub-region copies (vkCmdCopyImage
    // from CPU staging) and sampled reads both work without per-frame
    // transitions.  SAMPLED + TRANSFER_DST_BIT gives us both.  No
    // mip chain in v1; we'll add a per-page mip pyramid in v2 once
    // streaming works.
    glm::uvec2 pool_size(kVtPoolWidth, kVtPoolHeight);
    auto pool_usage = SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, TRANSFER_DST_BIT);
    for (uint32_t l = 0; l < uint32_t(VtLayer::COUNT); ++l) {
        VtLayer layer = static_cast<VtLayer>(l);
        er::Helper::create2DTextureImage(
            device_,
            layerFormat(layer),
            pool_size,
            1,                              // mip count (v1)
            layer_pools_[l].texture,
            pool_usage,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    // ── 2. Shared sampler for all four pools ───────────────────────
    // LINEAR + CLAMP_TO_EDGE.  Page borders aren't replicated in v1,
    // so bilinear filtering across page boundaries shows ~1-texel
    // seams.  v2 will add per-page borders to fix this; for now CLAMP
    // at least prevents wrap-around catastrophe.
    pool_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::NEAREST,
        /*anisotropy*/ 0.0f,
        std::source_location::current());

    // ── 3. Page table buffer ───────────────────────────────────────
    // HOST_VISIBLE | HOST_COHERENT so registerTexture() can write
    // entries from the CPU without a staging copy.  Each entry is a
    // packed u32 (see packPageEntry in the header).
    page_table_cpu_.assign(kMaxPageTableEntries, kPageEntryUnresident);
    er::Helper::createBuffer(
        device_,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        page_table_buffer_,
        page_table_memory_,
        std::source_location::current(),
        kMaxPageTableEntries * sizeof(uint32_t),
        page_table_cpu_.data());

    // ── 4. Per-VT metadata buffer ──────────────────────────────────
    // One VirtualTextureMeta per registered VT.  Indexed by vtIndex(id)
    // in the sampling shader to recover the page-table window.
    meta_cpu_.reserve(kMaxVirtualTextures);
    er::Helper::createBuffer(
        device_,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        meta_buffer_,
        meta_memory_,
        std::source_location::current(),
        kMaxVirtualTextures * sizeof(VirtualTextureMeta),
        nullptr);
}

void VirtualTextureManager::destroy() {
    for (uint32_t l = 0; l < uint32_t(VtLayer::COUNT); ++l) {
        layer_pools_[l].texture.destroy(device_);
    }
    pool_sampler_.reset();
    if (page_table_buffer_) {
        device_->destroyBuffer(page_table_buffer_);
        device_->freeMemory(page_table_memory_);
    }
    if (meta_buffer_) {
        device_->destroyBuffer(meta_buffer_);
        device_->freeMemory(meta_memory_);
    }
}

uint32_t VirtualTextureManager::allocatePageSlot() {
    // v1: monotonic counter.  v2: replace with LRU eviction queue.
    // Returns kVtPagesPerLayer when the pool is full — caller must
    // bail (or in v2: trigger eviction).
    if (next_free_slot_ >= kVtPagesPerLayer) {
        return kVtPagesPerLayer;
    }
    return next_free_slot_++;
}

VirtualTextureId VirtualTextureManager::registerTexture(
    VtLayer layer,
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height) {

    if (!pixels || width == 0 || height == 0) {
        return kInvalidVtId;
    }
    if (meta_cpu_.size() >= kMaxVirtualTextures) {
        // Hit the per-layer registration cap.  Return invalid; caller
        // should fall back to an inline color.
        return kInvalidVtId;
    }

    // ── Compute page grid ──────────────────────────────────────────
    // Round up so source textures whose dimensions aren't multiples
    // of kVtPageSize still get covered.  The rightmost / bottommost
    // page slots will have unused border pixels — we fill those with
    // the source's edge texels (CLAMP) below.
    const uint32_t pages_x = (width  + kVtPageSize - 1) / kVtPageSize;
    const uint32_t pages_y = (height + kVtPageSize - 1) / kVtPageSize;
    const uint32_t total_pages = pages_x * pages_y;

    // Need contiguous page-table window for this VT (so the shader
    // can compute table_offset + page_y * pages_x + page_x in one
    // load).  page_table_cpu_ is pre-allocated; just bump a tail
    // pointer in the next free region.
    static uint32_t next_table_offset = 0;
    if (next_table_offset + total_pages > kMaxPageTableEntries) {
        return kInvalidVtId;
    }
    const uint32_t table_offset = next_table_offset;
    next_table_offset += total_pages;

    // ── Per-page allocation + upload ───────────────────────────────
    // For each (page_x, page_y) we allocate a physical slot in the
    // shared pool, then upload the corresponding 128×128 sub-region
    // of the source texture into that slot.  Page-table entry is
    // written in lockstep so the GPU sees a consistent state.
    //
    // v1 limitation: no streaming.  If the pool fills up partway
    // through a registration, we mark remaining pages unresident and
    // the shader's residency check will fall back to a default.
    for (uint32_t py = 0; py < pages_y; ++py) {
        for (uint32_t px = 0; px < pages_x; ++px) {
            const uint32_t slot = allocatePageSlot();
            const uint32_t entry_index = table_offset + py * pages_x + px;

            if (slot >= kVtPagesPerLayer) {
                page_table_cpu_[entry_index] = kPageEntryUnresident;
                continue;
            }

            const uint32_t phys_page_x = slot % kVtPagesAcross;
            const uint32_t phys_page_y = slot / kVtPagesAcross;

            // TODO(rvt-v1-upload): copy the (px, py) source page into
            // pool slot (phys_page_x, phys_page_y) for this layer.
            // Requires a command buffer + staging buffer for the copy.
            // For the foundational scaffolding pass, we leave the
            // pool texels at their cleared default and only write the
            // page-table entry — sampling will return the default
            // color until upload is hooked up.  See uploadPage below.
            (void)pixels;

            page_table_cpu_[entry_index] =
                packPageEntry(phys_page_x, phys_page_y, /*resident*/ true);
        }
    }

    // ── Mirror page-table updates to the GPU buffer ────────────────
    // HOST_COHERENT so a memcpy is enough; the GPU sees writes on the
    // next access.  We could optimise to only push the modified
    // window, but page-table updates happen at scene-load time, not
    // per-frame, so a full memcpy of the changed region is fine.
    {
        void* mapped = device_->mapMemory(
            page_table_memory_,
            total_pages * sizeof(uint32_t),
            table_offset * sizeof(uint32_t));
        if (mapped) {
            std::memcpy(mapped,
                        page_table_cpu_.data() + table_offset,
                        total_pages * sizeof(uint32_t));
            device_->unmapMemory(page_table_memory_);
        }
    }

    // ── Append the meta entry ──────────────────────────────────────
    VirtualTextureMeta meta{};
    meta.width_px          = width;
    meta.height_px         = height;
    meta.pages_x           = pages_x;
    meta.pages_y           = pages_y;
    meta.page_table_offset = table_offset;
    meta.mip_count         = 1;
    meta.pad0              = 0;
    meta.pad1              = 0;
    const uint32_t vt_index = static_cast<uint32_t>(meta_cpu_.size());
    meta_cpu_.push_back(meta);

    // Write only the new entry to the GPU meta buffer.
    {
        void* mapped = device_->mapMemory(
            meta_memory_,
            sizeof(VirtualTextureMeta),
            vt_index * sizeof(VirtualTextureMeta));
        if (mapped) {
            std::memcpy(mapped, &meta, sizeof(VirtualTextureMeta));
            device_->unmapMemory(meta_memory_);
        }
    }

    return makeVtId(layer, vt_index);
}

std::shared_ptr<er::ImageView> VirtualTextureManager::getPoolImageView(
    VtLayer layer) const {
    return layer_pools_[uint32_t(layer)].texture.view;
}

void VirtualTextureManager::uploadPage(
    VtLayer layer,
    uint32_t phys_x,
    uint32_t phys_y,
    const uint8_t* page_pixels,
    uint32_t src_stride_bytes) {
    // TODO(rvt-v1-upload): implement vkCmdCopyBufferToImage from a
    // staging buffer into the pool texture's (phys_x, phys_y) sub-
    // region.  Needs:
    //   1. A reusable staging buffer pool (one per frame to avoid
    //      stalls) sized for at least a few pages × bytesPerTexel.
    //   2. Format conversion for VtLayer::NORMAL (RGBA→RG) and
    //      METAL_ROUGH_AO (channel reorder if source is unusual).
    //   3. A command-buffer entry point — currently the manager has
    //      no per-frame command-buffer hook; will add one when we
    //      wire registerTexture into mesh upload (Task #39).
    (void)layer; (void)phys_x; (void)phys_y;
    (void)page_pixels; (void)src_stride_bytes;
}

}  // namespace scene_rendering
}  // namespace engine
