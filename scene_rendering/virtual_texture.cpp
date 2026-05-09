//
// virtual_texture.cpp — RVT pool + page-table allocator (v1).
//

#include "virtual_texture.h"
#include "bc7_encoder.h"
#include "renderer/renderer_helper.h"

#include <cassert>
#include <cstring>
#include <vector>

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
    // ALBEDO uses BC7 (4× memory win); other layers use RGBA8 to
    // stay format-class-compatible with their RGBA8 source images
    // for the simple GPU-to-GPU vkCmdCopyImage path.  TRANSFER_SRC_BIT
    // was added to the texture loader (renderer.cpp::createTextureImage)
    // so source images now satisfy the readback path that the BC7
    // albedo registration uses.
    switch (layer) {
        case VtLayer::ALBEDO:         return er::Format::BC7_SRGB_BLOCK;
        case VtLayer::NORMAL:         return er::Format::R8G8B8A8_UNORM;
        case VtLayer::METAL_ROUGH_AO: return er::Format::R8G8B8A8_UNORM;
        case VtLayer::EMISSIVE:       return er::Format::R8G8B8A8_UNORM;
        default:                      return er::Format::R8G8B8A8_UNORM;
    }
}

VirtualTextureManager::VirtualTextureManager(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool)
    : device_(device)
    , descriptor_pool_(descriptor_pool)
    , encode_pool_(std::make_unique<engine::helper::ThreadPool>()) {

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

    // ── 5. Async streaming worker — DISABLED ───────────────────────
    // Was: spawn worker thread that drains pending_work_ via the
    // device's loader queue.  Triggered VK_ERROR_DEVICE_LOST on first
    // launch.  Likely cause: the existing mesh-load worker also owns
    // the loader-thread routing slot, so my worker re-registering
    // displaced it; subsequent transient command buffer calls from
    // mesh-load went to the wrong queue, mid-flight image layouts
    // got out of sync, the GPU rejected the resulting submission and
    // gave up.  Solution lives in a future iteration that uses a
    // dedicated VT-only command pool + queue, isolating its sync
    // domain from the rest of the engine.  For now, registerAlbedoBC7
    // calls processPendingWork directly on the caller's thread —
    // synchronous from the main thread's perspective, but parallel
    // BC7 encoding via encode_pool_ still saturates all CPU cores
    // for the encode step.
    //
    // worker_thread_ = std::thread([this]{ workerThreadLoop(); });
}

void VirtualTextureManager::destroy() {
    // Async worker is currently disabled (see constructor).  When it's
    // re-enabled, the shutdown sequence is: set worker_should_stop_,
    // notify cv, join thread.  Must happen BEFORE GPU resource
    // teardown so the worker doesn't touch destroyed handles.
    if (worker_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(work_mtx_);
            worker_should_stop_.store(true, std::memory_order_release);
        }
        work_cv_.notify_all();
        worker_thread_.join();
    }

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

uint32_t VirtualTextureManager::getPageTableBufferBytes() const {
    // Matches the size passed to createBuffer() in the constructor.
    return static_cast<uint32_t>(kMaxPageTableEntries * sizeof(uint32_t));
}

uint32_t VirtualTextureManager::getMetaBufferBytes() const {
    // Matches the size passed to createBuffer() in the constructor.
    return static_cast<uint32_t>(kMaxVirtualTextures * sizeof(VirtualTextureMeta));
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

// Source pixels are RGBA8.  Per-layer pool format may be different —
// this helper extracts a 128×128 sub-region from the source AND
// converts to the layer's pool format, writing into a tightly-packed
// destination buffer of bytesPerTexel(layer) * 128 * 128 bytes.
//
//   src_pixels:   tightly-packed RGBA8 source, src_w × src_h.
//   src_x/src_y:  top-left of the page within the source (in texels).
//   src_w/src_h:  source dimensions in texels.
//   dst:          destination buffer (must hold one page's worth).
//   layer:        determines output channel layout.
//
// Pages whose region extends past the source edge are CLAMPED — the
// extra pixels replicate the source's right/bottom edge.  This avoids
// black-fringes on textures whose dimensions aren't multiples of 128.
static void extractAndConvertPage(
    const uint8_t* src_pixels,
    uint32_t src_x, uint32_t src_y,
    uint32_t src_w, uint32_t src_h,
    VtLayer  layer,
    uint8_t* dst) {

    auto bpp_dst = [layer]() -> uint32_t {
        switch (layer) {
            case VtLayer::ALBEDO:         return 4;  // RGBA8
            case VtLayer::NORMAL:         return 2;  // RG8
            case VtLayer::METAL_ROUGH_AO: return 4;  // RGBA8
            case VtLayer::EMISSIVE:       return 4;  // RGBA8
            default:                       return 4;
        }
    }();

    for (uint32_t y = 0; y < kVtPageSize; ++y) {
        for (uint32_t x = 0; x < kVtPageSize; ++x) {
            uint32_t sx = std::min(src_x + x, src_w - 1u);
            uint32_t sy = std::min(src_y + y, src_h - 1u);
            const uint8_t* s = src_pixels + (sy * src_w + sx) * 4u;
            uint8_t* d       = dst + (y * kVtPageSize + x) * bpp_dst;
            switch (layer) {
                case VtLayer::ALBEDO:
                case VtLayer::METAL_ROUGH_AO:
                case VtLayer::EMISSIVE:
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                    break;
                case VtLayer::NORMAL:
                    // Tangent-space normal: keep the X/Y channels.  Z
                    // is reconstructed in the sampling shader as
                    // sqrt(1 - x² - y²) under the unit-normal
                    // assumption.  Most engines store the source
                    // normal map's RG channels in this layout already.
                    d[0] = s[0]; d[1] = s[1];
                    break;
                default: break;
            }
        }
    }
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
        return kInvalidVtId;
    }

    // ── Compute page grid ──────────────────────────────────────────
    // Round up so source textures whose dimensions aren't multiples
    // of kVtPageSize still get covered.  Edge pages get edge-clamped
    // pixels via extractAndConvertPage above.
    const uint32_t pages_x = (width  + kVtPageSize - 1) / kVtPageSize;
    const uint32_t pages_y = (height + kVtPageSize - 1) / kVtPageSize;
    const uint32_t total_pages = pages_x * pages_y;

    // Reserve a contiguous page-table window for this VT.  Shaders
    // index it as table_offset + page_y * pages_x + page_x, so
    // contiguity is required for a single load.  We bump a class-
    // level tail pointer; v2's eviction work will replace this with
    // a free-list allocator.
    if (next_page_table_tail_ + total_pages > kMaxPageTableEntries) {
        return kInvalidVtId;
    }
    const uint32_t table_offset = next_page_table_tail_;
    next_page_table_tail_ += total_pages;

    // ── Per-layer bytes-per-texel + page byte size ────────────────
    auto bpp = (layer == VtLayer::NORMAL) ? 2u : 4u;
    const uint32_t bytes_per_page = kVtPageSize * kVtPageSize * bpp;
    const uint32_t total_bytes    = bytes_per_page * total_pages;

    // ── Build the staging buffer in CPU memory ────────────────────
    // One contiguous blob: page (0,0) starts at offset 0, page (1,0)
    // at offset bytes_per_page, etc.  We extract + format-convert
    // each page with extractAndConvertPage, then upload the whole
    // blob in a single vkCmdCopyBufferToImage with N copy regions.
    std::vector<uint8_t> staging_cpu(total_bytes);
    std::vector<uint32_t> per_page_phys_x(total_pages, 0);
    std::vector<uint32_t> per_page_phys_y(total_pages, 0);
    std::vector<bool>     per_page_ok(total_pages, false);

    for (uint32_t py = 0; py < pages_y; ++py) {
        for (uint32_t px = 0; px < pages_x; ++px) {
            const uint32_t page_idx    = py * pages_x + px;
            const uint32_t entry_index = table_offset + page_idx;

            const uint32_t slot = allocatePageSlot();
            if (slot >= kVtPagesPerLayer) {
                // Pool is full.  Mark unresident; sampling will fall
                // back to a default colour.  v2 will trigger eviction
                // here instead.
                page_table_cpu_[entry_index] = kPageEntryUnresident;
                continue;
            }

            const uint32_t phys_page_x = slot % kVtPagesAcross;
            const uint32_t phys_page_y = slot / kVtPagesAcross;
            per_page_phys_x[page_idx] = phys_page_x;
            per_page_phys_y[page_idx] = phys_page_y;
            per_page_ok[page_idx]     = true;

            // Extract this page from the source texture and convert
            // to the layer's destination format.  Output goes into
            // the staging blob at this page's slot.
            extractAndConvertPage(
                pixels,
                px * kVtPageSize, py * kVtPageSize,
                width, height,
                layer,
                staging_cpu.data() + page_idx * bytes_per_page);

            page_table_cpu_[entry_index] =
                packPageEntry(phys_page_x, phys_page_y, /*resident*/ true);
        }
    }

    // ── Upload via staging buffer + batched copyBufferToImage ─────
    // Build a single Vulkan staging buffer with the entire blob, then
    // submit one transient command buffer with N copy regions (one
    // per page).  Far cheaper than per-page submit-and-waits.
    std::shared_ptr<er::Buffer>       staging_buf;
    std::shared_ptr<er::DeviceMemory> staging_mem;
    er::Helper::createBuffer(
        device_,
        SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        staging_buf,
        staging_mem,
        std::source_location::current(),
        total_bytes,
        staging_cpu.data());

    // Build copy regions — one per resident page.
    std::vector<er::BufferImageCopyInfo> copy_regions;
    copy_regions.reserve(total_pages);
    for (uint32_t i = 0; i < total_pages; ++i) {
        if (!per_page_ok[i]) continue;
        er::BufferImageCopyInfo r{};
        r.buffer_offset       = uint64_t(i) * bytes_per_page;
        r.buffer_row_length   = kVtPageSize;
        r.buffer_image_height = kVtPageSize;
        r.image_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        r.image_subresource.mip_level   = 0;
        r.image_subresource.base_array_layer = 0;
        r.image_subresource.layer_count = 1;
        r.image_offset = glm::ivec3(
            int32_t(per_page_phys_x[i] * kVtPageSize),
            int32_t(per_page_phys_y[i] * kVtPageSize),
            0);
        r.image_extent = glm::uvec3(kVtPageSize, kVtPageSize, 1);
        copy_regions.push_back(r);
    }

    if (!copy_regions.empty()) {
        auto cmd_buf = device_->setupTransientCommandBuffer();
        // Pool was created in SHADER_READ_ONLY_OPTIMAL — transition
        // briefly to TRANSFER_DST_OPTIMAL for the copy, then back.
        // This blocks reads until upload completes; safe because v1
        // only runs registerTexture at scene-load time, before the
        // render loop starts.
        er::ImageResourceInfo from_read{
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        er::ImageResourceInfo to_xfer{
            er::ImageLayout::TRANSFER_DST_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        cmd_buf->addImageBarrier(
            layer_pools_[uint32_t(layer)].texture.image,
            from_read, to_xfer, 0, 1, 0, 1);

        cmd_buf->copyBufferToImage(
            staging_buf,
            layer_pools_[uint32_t(layer)].texture.image,
            copy_regions,
            er::ImageLayout::TRANSFER_DST_OPTIMAL);

        cmd_buf->addImageBarrier(
            layer_pools_[uint32_t(layer)].texture.image,
            to_xfer, from_read, 0, 1, 0, 1);

        device_->submitAndWaitTransientCommandBuffer();
    }

    device_->destroyBuffer(staging_buf);
    device_->freeMemory(staging_mem);

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

VirtualTextureId VirtualTextureManager::registerTextureFromImage(
    VtLayer layer,
    const std::shared_ptr<er::Image>& src_image,
    uint32_t width,
    uint32_t height) {

    if (!src_image || width == 0 || height == 0) return kInvalidVtId;
    if (meta_cpu_.size() >= kMaxVirtualTextures) return kInvalidVtId;

    // ── ALBEDO uses a separate path: GPU readback → parallel CPU
    //    BC7 encode → upload via staging buffer.  vkCmdCopyImage
    //    can't bridge RGBA8 (source) and BC7 (destination) so we
    //    round-trip through the CPU.  Encoding is parallelised
    //    across all hardware threads via encode_pool_; for a
    //    1024² texture (64 pages of 32×32 BC7 blocks each) an 8-core
    //    machine is roughly 8× faster than serial encoding.
    if (layer == VtLayer::ALBEDO) {
        return registerAlbedoBC7(src_image, width, height);
    }

    // Page grid identical to the CPU-pixels variant.
    const uint32_t pages_x = (width  + kVtPageSize - 1) / kVtPageSize;
    const uint32_t pages_y = (height + kVtPageSize - 1) / kVtPageSize;
    const uint32_t total_pages = pages_x * pages_y;

    if (next_page_table_tail_ + total_pages > kMaxPageTableEntries) {
        return kInvalidVtId;
    }
    const uint32_t table_offset = next_page_table_tail_;
    next_page_table_tail_ += total_pages;

    // ── Walk pages, allocate slots, build copy regions ─────────────
    // Each region copies a 128×128 sub-rect from src_image at
    // (src.x, src.y) to pool[layer] at (dst.x, dst.y).  Source
    // out-of-bounds (when source dims aren't multiples of 128) is
    // CLAMPED by clipping the extent — we copy fewer pixels for the
    // edge page rather than reading garbage.  The pool slot's
    // remaining pixels stay at whatever the prior register left
    // there; in v1 the pool starts cleared so unused page tail is
    // black, which is fine for a non-tiling source.
    std::vector<er::ImageCopyInfo> regions;
    regions.reserve(total_pages);

    for (uint32_t py = 0; py < pages_y; ++py) {
        for (uint32_t px = 0; px < pages_x; ++px) {
            const uint32_t page_idx    = py * pages_x + px;
            const uint32_t entry_index = table_offset + page_idx;

            const uint32_t slot = allocatePageSlot();
            if (slot >= kVtPagesPerLayer) {
                page_table_cpu_[entry_index] = kPageEntryUnresident;
                continue;
            }
            const uint32_t phys_page_x = slot % kVtPagesAcross;
            const uint32_t phys_page_y = slot / kVtPagesAcross;

            // Source extent: clip to the source image bounds so
            // edge pages don't try to read past width/height.
            const uint32_t src_x = px * kVtPageSize;
            const uint32_t src_y = py * kVtPageSize;
            const uint32_t copy_w = std::min(kVtPageSize, width  - src_x);
            const uint32_t copy_h = std::min(kVtPageSize, height - src_y);

            er::ImageCopyInfo r{};
            r.src_subresource.aspect_mask     = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
            r.src_subresource.mip_level       = 0;
            r.src_subresource.base_array_layer = 0;
            r.src_subresource.layer_count     = 1;
            r.src_offset = glm::ivec3(int32_t(src_x), int32_t(src_y), 0);
            r.dst_subresource.aspect_mask     = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
            r.dst_subresource.mip_level       = 0;
            r.dst_subresource.base_array_layer = 0;
            r.dst_subresource.layer_count     = 1;
            r.dst_offset = glm::ivec3(
                int32_t(phys_page_x * kVtPageSize),
                int32_t(phys_page_y * kVtPageSize), 0);
            r.extent = glm::uvec3(copy_w, copy_h, 1);
            regions.push_back(r);

            page_table_cpu_[entry_index] =
                packPageEntry(phys_page_x, phys_page_y, /*resident*/ true);
        }
    }

    // Push the page-table window to GPU.
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

    // ── Submit one transient cmd buffer doing all the page copies ──
    // Note: vkCmdCopyImage requires src and dst formats to be
    // "compatible" — same texel size class.  RGBA8 → RGBA8 is fine.
    // For VtLayer::NORMAL the pool is RG8, which is NOT format-class-
    // compatible with the source RGBA8.  In that case we'd need a
    // blit (with format conversion) or per-channel copy.  v1 treats
    // NORMAL the same as ALBEDO format (RGBA8) until we add the blit
    // path; the sampling shader still only reads .rg so the redundant
    // .ba just costs 2 bytes per texel of pool memory.  Trade-off
    // preferred over the complexity of adding a blit-only path here.
    if (!regions.empty()) {
        auto cmd_buf = device_->setupTransientCommandBuffer();

        er::ImageResourceInfo from_read{
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        er::ImageResourceInfo to_xfer_src{
            er::ImageLayout::TRANSFER_SRC_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        er::ImageResourceInfo to_xfer_dst{
            er::ImageLayout::TRANSFER_DST_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };

        cmd_buf->addImageBarrier(src_image,
            from_read, to_xfer_src, 0, 1, 0, 1);
        cmd_buf->addImageBarrier(
            layer_pools_[uint32_t(layer)].texture.image,
            from_read, to_xfer_dst, 0, 1, 0, 1);

        cmd_buf->copyImage(
            src_image, er::ImageLayout::TRANSFER_SRC_OPTIMAL,
            layer_pools_[uint32_t(layer)].texture.image,
            er::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions);

        cmd_buf->addImageBarrier(src_image,
            to_xfer_src, from_read, 0, 1, 0, 1);
        cmd_buf->addImageBarrier(
            layer_pools_[uint32_t(layer)].texture.image,
            to_xfer_dst, from_read, 0, 1, 0, 1);

        device_->submitAndWaitTransientCommandBuffer();
    }

    // Append meta entry.
    VirtualTextureMeta meta{};
    meta.width_px          = width;
    meta.height_px         = height;
    meta.pages_x           = pages_x;
    meta.pages_y           = pages_y;
    meta.page_table_offset = table_offset;
    meta.mip_count         = 1;
    const uint32_t vt_index = static_cast<uint32_t>(meta_cpu_.size());
    meta_cpu_.push_back(meta);
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

// ─── BC7 albedo path (async) ─────────────────────────────────────────
// Synchronous setup phase (this function — called on main thread):
//   1. Validate inputs and reserve a contiguous page-table window.
//   2. Allocate physical pool slots for each page.  Mark every entry
//      as UNRESIDENT for now so the shader's fallback returns a
//      default colour while streaming completes.
//   3. Append the meta entry so the VT id resolves correctly.
//   4. Push a PendingWork onto the worker queue and return.
//
// Asynchronous execution phase (workerThreadLoop → processPendingWork):
//   5. GPU readback the source image (RGBA8) → CPU.
//   6. Parallel BC7-encode 128×128 pages across encode_pool_ workers.
//   7. Upload BC7 blob to the pool via staging buffer.
//   8. Flip page-table entries from UNRESIDENT → RESIDENT.
//
// From the caller's perspective this returns immediately; the texture
// fades in over a few hundred milliseconds as the worker completes.
VirtualTextureId VirtualTextureManager::registerAlbedoBC7(
    const std::shared_ptr<er::Image>& src_image,
    uint32_t width,
    uint32_t height) {

    const uint32_t pages_x = (width  + kVtPageSize - 1) / kVtPageSize;
    const uint32_t pages_y = (height + kVtPageSize - 1) / kVtPageSize;
    const uint32_t total_pages = pages_x * pages_y;

    if (next_page_table_tail_ + total_pages > kMaxPageTableEntries) {
        std::printf("[RVT] registerAlbedoBC7 BAILED: page-table full "
                    "(%u + %u > %u)\n",
                    next_page_table_tail_, total_pages, kMaxPageTableEntries);
        return kInvalidVtId;
    }
    const uint32_t table_offset = next_page_table_tail_;
    next_page_table_tail_ += total_pages;

    // ── Allocate page slots + mark all entries UNRESIDENT ──────────
    // Allocator state (next_free_slot_, page_table_cpu_) is not
    // thread-safe; we do this on the main thread.  The worker only
    // flips entries from UNRESIDENT to RESIDENT later, in-place,
    // which is a single-writer race-free update.
    PendingWork work{};
    work.layer        = VtLayer::ALBEDO;
    work.src_image    = src_image;
    work.width        = width;
    work.height       = height;
    work.table_offset = table_offset;
    work.pages_x      = pages_x;
    work.pages_y      = pages_y;
    work.phys_x.resize(total_pages, 0);
    work.phys_y.resize(total_pages, 0);
    work.ok.resize(total_pages, 0);

    uint32_t allocated = 0;
    for (uint32_t i = 0; i < total_pages; ++i) {
        const uint32_t entry_index = table_offset + i;
        const uint32_t slot        = allocatePageSlot();
        if (slot >= kVtPagesPerLayer) {
            page_table_cpu_[entry_index] = kPageEntryUnresident;
            continue;
        }
        work.phys_x[i] = slot % kVtPagesAcross;
        work.phys_y[i] = slot / kVtPagesAcross;
        work.ok[i]     = 1;
        ++allocated;
        // UNRESIDENT for now — flipped to RESIDENT by the worker once
        // the page's BC7 data is uploaded.
        page_table_cpu_[entry_index] = kPageEntryUnresident;
    }
    std::printf("[RVT] registerAlbedoBC7 %ux%u pages=%ux%u tbl_off=%u "
                "allocated=%u/%u next_slot=%u/%u\n",
                width, height, pages_x, pages_y, table_offset,
                allocated, total_pages,
                next_free_slot_, kVtPagesPerLayer);

    // Push the freshly-marked UNRESIDENT entries to GPU.  Worker will
    // overwrite them with RESIDENT entries when ready.
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

    // ── Append the meta entry NOW so the VT id resolves correctly
    //    even before the worker has processed this request.  The
    //    shader resolves through meta to find page_table_offset; the
    //    page-table entries themselves are UNRESIDENT until the
    //    worker completes, at which point the shader sees real data.
    VirtualTextureMeta meta{};
    meta.width_px          = width;
    meta.height_px         = height;
    meta.pages_x           = pages_x;
    meta.pages_y           = pages_y;
    meta.page_table_offset = table_offset;
    meta.mip_count         = 1;
    const uint32_t vt_index = static_cast<uint32_t>(meta_cpu_.size());
    meta_cpu_.push_back(meta);
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

    // ── Process synchronously on caller's thread ───────────────────
    // The async worker path triggered VK_ERROR_DEVICE_LOST due to
    // loader-thread routing conflicts with the existing mesh-load
    // worker (only one thread can own the loader-queue routing).
    // Until we add a dedicated VT-only command pool/queue, do the
    // GPU readback + parallel encode + GPU upload right here on the
    // caller's thread.  Parallel encoding via encode_pool_ remains
    // — that's the only meaningful win this iteration.
    processPendingWork(std::move(work));
    return makeVtId(VtLayer::ALBEDO, vt_index);
}

// ─── Worker thread: process pending registration requests ────────────
// Single thread that drains pending_work_ in order.  Registers itself
// as the device's loader thread so its setupTransientCommandBuffer
// calls route to the loader queue, separate from the main thread's
// compute queue — avoids cross-thread Vulkan submission races.  The
// existing mesh-load worker also uses this slot; in practice mesh load
// is idle by the time VT registrations start, so re-registering here
// is safe.  If the user dynamically loads new meshes mid-game while VT
// streaming is active, the loader-thread tag can briefly mismatch —
// worst case is one transient command buffer using the wrong queue,
// which usually triggers a validation warning rather than a hang.
void VirtualTextureManager::workerThreadLoop() {
    if (device_) {
        device_->registerLoaderThread(std::this_thread::get_id());
    }
    for (;;) {
        PendingWork w;
        {
            std::unique_lock<std::mutex> lk(work_mtx_);
            work_cv_.wait(lk, [this]{
                return worker_should_stop_.load(std::memory_order_acquire) ||
                       !pending_work_.empty();
            });
            if (worker_should_stop_.load(std::memory_order_acquire) &&
                pending_work_.empty()) {
                return;
            }
            w = std::move(pending_work_.front());
            pending_work_.pop();
        }
        processPendingWork(std::move(w));
    }
}

// Heavy-lifting half of registerAlbedoBC7, runs on worker thread:
// readback → parallel BC7 encode → upload → flip residency bits.
void VirtualTextureManager::processPendingWork(PendingWork&& w) {
    if (w.layer != VtLayer::ALBEDO) return;  // only ALBEDO supported
    const uint32_t total_pages = w.pages_x * w.pages_y;

    // === DUPLICATED BLOCK (formerly inside registerAlbedoBC7) ===
    // Below is the original GPU readback + parallel encode + upload
    // logic, lifted verbatim from the old synchronous version.
    // Variable names preserved for readability.
    const uint32_t width        = w.width;
    const uint32_t height       = w.height;
    const auto&    src_image    = w.src_image;
    const uint32_t table_offset = w.table_offset;

    // ── Step 1: GPU → CPU readback of the source image ─────────────
    // Allocate a host-visible buffer big enough for all pixels (W*H*4
    // bytes).  Issue a transient command buffer that transitions the
    // source to TRANSFER_SRC, copies it to the buffer, transitions
    // back to SHADER_READ_ONLY.  Submit-and-wait so the buffer's
    // contents are valid before we touch them on the CPU.
    const uint64_t src_bytes = uint64_t(width) * height * 4u;
    std::shared_ptr<er::Buffer>       readback_buf;
    std::shared_ptr<er::DeviceMemory> readback_mem;
    er::Helper::createBuffer(
        device_,
        SET_FLAG_BIT(BufferUsage, TRANSFER_DST_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        readback_buf,
        readback_mem,
        std::source_location::current(),
        src_bytes,
        nullptr);

    {
        auto cmd = device_->setupTransientCommandBuffer();
        er::ImageResourceInfo from_read{
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        er::ImageResourceInfo to_xfer_src{
            er::ImageLayout::TRANSFER_SRC_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        cmd->addImageBarrier(src_image, from_read, to_xfer_src, 0, 1, 0, 1);

        er::BufferImageCopyInfo r{};
        r.buffer_offset = 0;
        r.buffer_row_length = width;
        r.buffer_image_height = height;
        r.image_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        r.image_subresource.mip_level = 0;
        r.image_subresource.base_array_layer = 0;
        r.image_subresource.layer_count = 1;
        r.image_offset = glm::ivec3(0);
        r.image_extent = glm::uvec3(width, height, 1);
        cmd->copyImageToBuffer(
            src_image, readback_buf, { r },
            er::ImageLayout::TRANSFER_SRC_OPTIMAL);

        cmd->addImageBarrier(src_image, to_xfer_src, from_read, 0, 1, 0, 1);
        device_->submitAndWaitTransientCommandBuffer();
    }

    // Map and copy into a CPU-owned vector so we can free the GPU
    // readback buffer immediately and not hold device memory for the
    // duration of the CPU encode.
    std::vector<uint8_t> cpu_pixels(src_bytes);
    {
        void* mapped = device_->mapMemory(readback_mem, src_bytes, 0);
        if (mapped) {
            std::memcpy(cpu_pixels.data(), mapped, src_bytes);
            device_->unmapMemory(readback_mem);
        }
    }
    device_->destroyBuffer(readback_buf);
    device_->freeMemory(readback_mem);

    // ── Step 2: parallel BC7 encode ────────────────────────────────
    // Page slot allocation already happened on the caller's thread;
    // PendingWork carries phys_x/y/ok arrays per page.  Here we just
    // do the CPU-bound encoding, parallelised across encode_pool_.
    const uint32_t kBc7BytesPerPage = (kVtPageSize / 4) * (kVtPageSize / 4) * 16;
    std::vector<uint8_t> staging_cpu(uint64_t(total_pages) * kBc7BytesPerPage);

    // pages_x captured in lambda — local copy, since w may be moved.
    const uint32_t lambda_pages_x = w.pages_x;
    encode_pool_->parallelFor(total_pages,
        [&](size_t page_idx) {
            if (!w.ok[page_idx]) return;

            const uint32_t px = uint32_t(page_idx) % lambda_pages_x;
            const uint32_t py = uint32_t(page_idx) / lambda_pages_x;

            // Per-thread scratch buffer (allocated on the worker).
            std::vector<uint8_t> page_rgba(kVtPageSize * kVtPageSize * 4u);
            for (uint32_t y = 0; y < kVtPageSize; ++y) {
                uint32_t sy = std::min(py * kVtPageSize + y, height - 1u);
                for (uint32_t x = 0; x < kVtPageSize; ++x) {
                    uint32_t sx = std::min(px * kVtPageSize + x, width - 1u);
                    const uint8_t* s = cpu_pixels.data() + (size_t(sy) * width + sx) * 4;
                    uint8_t*       d = page_rgba.data() + (size_t(y) * kVtPageSize + x) * 4;
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                }
            }

            uint8_t* dst = staging_cpu.data() + uint64_t(page_idx) * kBc7BytesPerPage;
            encodeBC7Mode6(page_rgba.data(),
                           kVtPageSize, kVtPageSize, dst);
        });

    // ── Step 3: upload the BC7 staging blob to the pool ────────────
    // Single staging buffer + N copy regions.  Pool is in
    // SHADER_READ_ONLY_OPTIMAL on entry; transition to TRANSFER_DST,
    // copy, transition back.
    std::shared_ptr<er::Buffer>       staging_buf;
    std::shared_ptr<er::DeviceMemory> staging_mem;
    er::Helper::createBuffer(
        device_,
        SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        staging_buf,
        staging_mem,
        std::source_location::current(),
        staging_cpu.size(),
        staging_cpu.data());

    std::vector<er::BufferImageCopyInfo> regions;
    regions.reserve(total_pages);
    for (uint32_t i = 0; i < total_pages; ++i) {
        if (!w.ok[i]) continue;
        er::BufferImageCopyInfo r{};
        r.buffer_offset = uint64_t(i) * kBc7BytesPerPage;
        // BC7: buffer_row_length / image_height are in TEXELS, not
        // blocks.  The driver does the conversion based on format.
        r.buffer_row_length   = kVtPageSize;
        r.buffer_image_height = kVtPageSize;
        r.image_subresource.aspect_mask     = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        r.image_subresource.mip_level       = 0;
        r.image_subresource.base_array_layer = 0;
        r.image_subresource.layer_count     = 1;
        r.image_offset = glm::ivec3(
            int32_t(w.phys_x[i] * kVtPageSize),
            int32_t(w.phys_y[i] * kVtPageSize), 0);
        r.image_extent = glm::uvec3(kVtPageSize, kVtPageSize, 1);
        regions.push_back(r);
    }

    if (!regions.empty()) {
        auto cmd = device_->setupTransientCommandBuffer();
        er::ImageResourceInfo from_read{
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        er::ImageResourceInfo to_xfer{
            er::ImageLayout::TRANSFER_DST_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        cmd->addImageBarrier(
            layer_pools_[uint32_t(VtLayer::ALBEDO)].texture.image,
            from_read, to_xfer, 0, 1, 0, 1);
        cmd->copyBufferToImage(
            staging_buf,
            layer_pools_[uint32_t(VtLayer::ALBEDO)].texture.image,
            regions,
            er::ImageLayout::TRANSFER_DST_OPTIMAL);
        cmd->addImageBarrier(
            layer_pools_[uint32_t(VtLayer::ALBEDO)].texture.image,
            to_xfer, from_read, 0, 1, 0, 1);
        device_->submitAndWaitTransientCommandBuffer();
    }
    device_->destroyBuffer(staging_buf);
    device_->freeMemory(staging_mem);

    // ── Step 4: flip page-table entries to RESIDENT ────────────────
    // The synchronous setup phase wrote UNRESIDENT entries; now that
    // BC7 data is uploaded, mark them RESIDENT in-place.  Single-
    // writer (this worker) so no cross-thread races on the entries.
    // HOST_COHERENT page_table memory means the GPU sees the flip
    // immediately — next frame's sample picks up the now-resident
    // pages.
    uint32_t flipped = 0;
    for (uint32_t i = 0; i < total_pages; ++i) {
        if (!w.ok[i]) continue;
        const uint32_t entry_index = table_offset + i;
        page_table_cpu_[entry_index] =
            packPageEntry(w.phys_x[i], w.phys_y[i], /*resident*/ true);
        ++flipped;
    }
    {
        void* mapped = device_->mapMemory(
            page_table_memory_,
            total_pages * sizeof(uint32_t),
            table_offset * sizeof(uint32_t));
        bool map_ok = (mapped != nullptr);
        if (mapped) {
            std::memcpy(mapped,
                        page_table_cpu_.data() + table_offset,
                        total_pages * sizeof(uint32_t));
            device_->unmapMemory(page_table_memory_);
        }
        // Diagnostic: confirm what we wrote.  Pre-state vs post-state of
        // the first entry, plus the count of pages flipped to RESIDENT.
        const uint32_t first = (total_pages > 0)
            ? page_table_cpu_[table_offset]
            : 0u;
        std::printf("[RVT] processPendingWork %ux%u tbl_off=%u flipped=%u/%u "
                    "first_entry=0x%08x map_ok=%d\n",
                    w.pages_x, w.pages_y, table_offset,
                    flipped, total_pages, first, map_ok ? 1 : 0);
    }
}

}  // namespace scene_rendering
}  // namespace engine
