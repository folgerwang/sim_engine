//
// virtual_texture.cpp — RVT pool + page-table allocator (v1).
//

#include "virtual_texture.h"
#include "bc7_encoder.h"
#include "renderer/renderer_helper.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_set>
#include <vector>

namespace er = engine::renderer;

namespace engine {
namespace scene_rendering {

// ── vt_pool.log helper ──────────────────────────────────────────────────
// Shared by VirtualTextureManager::tick (per-frame stats line) and
// registerMaterial (one-shot per material registration).  Truncates on
// the first call of the process so each run starts with a clean file.
// Returns the same ofstream on every subsequent call — opened once,
// closes implicitly at program exit.
//
// Each writer is expected to format a complete line (including '\n')
// and call flush() so a `tail -f vt_pool.log` sees updates live.
namespace {
std::ofstream& vtLog() {
    static std::ofstream s_log("vt_pool.log",
                               std::ios::out | std::ios::trunc);
    return s_log;
}
}  // namespace

// Capacity assumed at startup.  Pre-allocate page-table + meta buffers
// for this many entries; can be raised if we hit it.  With mip-aware
// VTs each registration claims `vtTotalPagesAllMips(...)` page-table
// entries — for a 4096² source that's 1+4+16+64+256+1024+4096+(remainder)
// ≈ 5461 entries.  1 M total still fits ~190 such VTs, well past the
// 200-textures-per-scene workload.
static constexpr uint32_t kMaxVirtualTextures   = 1024u;
static constexpr uint32_t kMaxPageTableEntries  = 1024u * 1024u;  // ~4 MB

// ── CPU box-filter mip downsample (RGBA8 → RGBA8) ─────────────────
// Source is `src_w × src_h` RGBA8.  Writes a (src_w/2 × src_h/2)
// RGBA8 result to `dst`.  Linear-space averaging — fine for diffuse
// since the BC7 pool format is _SRGB and the GPU de-gammafies on
// sample, but technically the mip should be generated in linear
// space for sRGB content.  Two-stage proper version (sRGB → linear
// → average → linear → sRGB) is a v2 quality bump; the visible
// difference at typical viewing mip levels is small.
static void boxDownsampleRgba8(
    const uint8_t* src, uint32_t src_w, uint32_t src_h,
    uint8_t* dst) {
    const uint32_t dst_w = std::max(1u, src_w >> 1);
    const uint32_t dst_h = std::max(1u, src_h >> 1);
    for (uint32_t y = 0; y < dst_h; ++y) {
        for (uint32_t x = 0; x < dst_w; ++x) {
            const uint32_t sx0 = std::min(x * 2u,         src_w - 1u);
            const uint32_t sx1 = std::min(x * 2u + 1u,    src_w - 1u);
            const uint32_t sy0 = std::min(y * 2u,         src_h - 1u);
            const uint32_t sy1 = std::min(y * 2u + 1u,    src_h - 1u);
            const uint8_t* s00 = src + (sy0 * src_w + sx0) * 4u;
            const uint8_t* s10 = src + (sy0 * src_w + sx1) * 4u;
            const uint8_t* s01 = src + (sy1 * src_w + sx0) * 4u;
            const uint8_t* s11 = src + (sy1 * src_w + sx1) * 4u;
            uint8_t*       d   = dst + (y   * dst_w + x  ) * 4u;
            for (int c = 0; c < 4; ++c) {
                d[c] = uint8_t(
                    (uint32_t(s00[c]) + s10[c] + s01[c] + s11[c] + 2u) >> 2);
            }
        }
    }
}

static er::Format layerFormat(VtLayer layer) {
    // ALBEDO is BC7_SRGB_BLOCK (4× VRAM win over RGBA8) — encoded
    // CPU-side via Rich Geldreich's bc7enc.  The other three layers
    // stay RGBA8_UNORM since they're blitted directly from source
    // images on the GPU (no CPU encoding).  See encodeAndCacheVt for
    // the ALBEDO encode pipeline and uploadTileAllLayers for the
    // BC7-staging vs blit upload split.
    switch (layer) {
        case VtLayer::ALBEDO:         return er::Format::BC7_SRGB_BLOCK;
        // NORMAL is BC5_UNORM — 2-channel block-compressed format
        // designed exactly for tangent-space normals.  Only RG is
        // stored on the GPU; the shader reconstructs Z via
        // sqrt(1 - x² - y²).  4× memory saving over RGBA8_UNORM
        // (~168 MB → ~42 MB at 8208×4104) with no visible quality
        // loss for typical normal-map content.  CPU-encoded per
        // tile via encodeBC5UNorm; same per-entry byte layout as
        // BC7 so the existing kBc7BytesMip0/Mip1 constants apply.
        case VtLayer::NORMAL:         return er::Format::BC5_UNORM_BLOCK;
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
    // Pool has TWO mip levels per slot — mip 0 is the full 64×64
    // page, mip 1 is a 32×32 half-res of that same page.  This isn't
    // a normal mip chain over the whole pool image (each slot's
    // mip-0 and mip-1 represent the same source content at two
    // resolutions; neighbouring slots in mip 1 belong to different
    // VTs the same way they do in mip 0).  Combined with a LINEAR
    // mipmap sampler and explicit textureLod(..., lod_frac) in the
    // shader, this lets us smoothly lerp within the VT-mip-k slot
    // when our continuous LOD is fractional — which hides the snap
    // when the LOD crosses an integer VT-mip boundary.  Cost: pool
    // VRAM grows by 1/4 (the mip-1 area) and each page upload does
    // one extra BC7 encode / blit at half res.  TRANSFER_SRC_BIT is
    // added so future code can blit pool mip 0 → mip 1 in-place if
    // we ever want to skip the per-page CPU mip-1 generation.
    glm::uvec2 pool_size(kVtPoolWidth, kVtPoolHeight);
    auto pool_usage = SET_3_FLAG_BITS(ImageUsage,
        SAMPLED_BIT, TRANSFER_DST_BIT, TRANSFER_SRC_BIT);
    for (uint32_t l = 0; l < uint32_t(VtLayer::COUNT); ++l) {
        VtLayer layer = static_cast<VtLayer>(l);
        er::Helper::create2DTextureImage(
            device_,
            layerFormat(layer),
            pool_size,
            kVtPoolMipLevels,               // 2: full + half-res per slot
            layer_pools_[l].texture,
            pool_usage,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
        // Replace the auto-generated single-mip view with one that
        // covers both pool mips.  The default Helper::create2DTextureImage
        // always builds a 1-mip view (mip_count param defaults to 1
        // in createImageView), but we need the trilinear sampler to
        // actually see pool mip 1 — without this the GPU would clamp
        // every sample to mip 0 and the LOD lerp would no-op.
        layer_pools_[l].texture.view = device_->createImageView(
            layer_pools_[l].texture.image,
            er::ImageViewType::VIEW_2D,
            layerFormat(layer),
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current(),
            /*base_mip*/   0,
            /*mip_count*/  kVtPoolMipLevels,
            /*base_layer*/ 0,
            /*layer_count*/1);
        // Companion mip-0-only view for the ImGui debug viewer.  ImGui
        // draws the pool as a small thumbnail quad; the GPU's implicit
        // LOD calculation would otherwise pick mip ~log2(POOL/THUMB)
        // which is way past the trilinear view's mip range (only mip
        // 0 + mip 1) and on some drivers returns black instead of
        // clamping to the highest available mip.  Binding ImGui to a
        // single-mip view side-steps that entirely.
        layer_pools_[l].mip0_view = device_->createImageView(
            layer_pools_[l].texture.image,
            er::ImageViewType::VIEW_2D,
            layerFormat(layer),
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current(),
            /*base_mip*/   0,
            /*mip_count*/  1,
            /*base_layer*/ 0,
            /*layer_count*/1);
    }

    // ── 2. Shared sampler for all four pools ───────────────────────
    // LINEAR min/mag + LINEAR mipmap + CLAMP_TO_EDGE.  Mipmap mode
    // LINEAR is required for the trilinear pool-mip-0/1 lerp that
    // hides VT-mip transitions.  Page borders still aren't
    // replicated in v1 — the half-texel inset clamp in vtResolve
    // (now 1.0/page_size, increased to cover the wider mip-1
    // bilinear footprint) prevents bleed across slot boundaries
    // for both pool mip levels.
    pool_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::LINEAR,
        /*anisotropy*/ 0.0f,
        std::source_location::current());
    // Debug-viewer sampler: NEAREST mipmap so the sampler picks one
    // mip without interpolating into a non-existent neighbour.
    pool_debug_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::NEAREST,
        /*anisotropy*/ 0.0f,
        std::source_location::current());

    // ── 3. Page table buffer ───────────────────────────────────────
    // HOST_VISIBLE | HOST_COHERENT so registerMaterial can write
    // entries from the CPU at startup (synchronous, before any
    // rendering uses them).  TRANSFER_DST_BIT lets tick() update
    // entries via vkCmdUpdateBuffer at frame time — that path
    // sequences the page-table update with the rest of the frame's
    // GPU work via standard Vulkan barriers, instead of relying on
    // HOST_COHERENT visibility (which would race in-flight
    // previous-frame cluster-bindless reads → flicker).
    page_table_cpu_.assign(kMaxPageTableEntries, kPageEntryUnresident);
    er::Helper::createBuffer(
        device_,
        SET_2_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT, TRANSFER_DST_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        page_table_buffer_,
        page_table_memory_,
        std::source_location::current(),
        kMaxPageTableEntries * sizeof(uint32_t),
        page_table_cpu_.data());

    // ── 3b. Persistent tile-upload staging buffer ──────────────────
    // Single HOST_VISIBLE+HOST_COHERENT buffer of
    // kStreamerUploadsPerFrame * kBc7BytesPerEntry bytes; one alloc
    // at construction, mapped once for the lifetime of the manager.
    // Each per-tick upload memcpy's to a known offset inside this
    // buffer instead of vkCreateBuffer/vkAllocateMemory'ing fresh per
    // tile.  The driver-side allocation overhead per buffer is
    // ~100 µs on most desktop drivers, so for 32 uploads/frame this
    // alone saves ~3 ms of CPU time per tick.
    {
        // Two BC layers (ALBEDO BC7 + NORMAL BC5) both consume one
        // kBc7BytesPerEntry-sized region per upload, so each frame
        // needs 2 × N regions.  We allocate kVtCompactSlots (= FIF)
        // copies of that to make CPU writes safe while the GPU is
        // still reading from the previous frame's slice.
        //
        // Layout per frame slot (frame_index % FIF):
        //   slot N → bytes [frame_base + N*2 + 0] = ALBEDO
        //          → bytes [frame_base + N*2 + 1] = NORMAL
        // where frame_base = (frame_index % FIF) * kPerFrameBytes.
        //
        // Without this multi-buffering the previous design relied on
        // submitAndWaitTransientCommandBuffer() to fence GPU reads
        // before the next CPU memcpy — that wait alone was the bulk
        // of "VT tick (CPU)" cost.  Now uploads ride on the frame's
        // command buffer; with FIF separate slices, no synchronization
        // beyond the frame fence is needed.
        const uint64_t kPerFrameBytes =
            uint64_t(kStreamerUploadsPerFrame) * 2u * kBc7BytesPerEntry;
        const uint64_t upload_bytes = kPerFrameBytes * kVtCompactSlots;
        er::Helper::createBuffer(
            device_,
            SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            upload_staging_buffer_,
            upload_staging_memory_,
            std::source_location::current(),
            upload_bytes, nullptr);
        upload_staging_mapped_ = static_cast<uint8_t*>(
            device_->mapMemory(upload_staging_memory_, upload_bytes, 0));
        if (!upload_staging_mapped_) {
            std::printf("[RVT] FATAL: upload_staging_buffer mapMemory "
                        "returned null — streaming will be broken.\n");
        }
    }

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

    // ── 4b. Streaming feedback buffer ──────────────────────────────
    // Shader writes one tile-key per 8×8 screen block.  GPU-only:
    // the CPU no longer reads this buffer directly — instead, the
    // compactFeedback() compute pass scans it on the GPU and writes
    // a tiny compact list to feedback_compact_buffer_ which the CPU
    // reads instead.  Sized to kVtFeedbackTotal uints (= 512² = 1 MB),
    // covering up to a 4096-wide screen with 8-px blocks.
    // DEVICE_LOCAL gives the GPU peak bandwidth on the read path.
    // TRANSFER_DST_BIT for the per-frame vkCmdFillBuffer clear.
    {
        std::vector<uint32_t> init(kVtFeedbackTotal, kVtTileKeyInvalid);
        er::Helper::createBuffer(
            device_,
            SET_2_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT, TRANSFER_DST_BIT),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            0,
            feedback_buffer_,
            feedback_memory_,
            std::source_location::current(),
            uint64_t(kVtFeedbackTotal) * sizeof(uint32_t),
            init.data());
    }

    // ── 4b'. GPU feedback-compaction output buffer ─────────────────
    // FIF slots laid out back-to-back in one buffer; the compaction
    // compute shader's push constant `compact_slot_offset_uints`
    // selects which slot it writes to.  HOST_VISIBLE | HOST_COHERENT
    // so tick() can read it directly with no map/unmap each frame.
    // STORAGE_BUFFER_BIT for the compute shader; TRANSFER_DST_BIT
    // for the per-slot counter clear (vkCmdFillBuffer of the first
    // 4 bytes of the slot before each dispatch).
    {
        const uint64_t total_bytes =
            uint64_t(kVtCompactSlots) * uint64_t(kVtCompactSlotBytes);
        er::Helper::createBuffer(
            device_,
            SET_2_FLAG_BITS(BufferUsage, STORAGE_BUFFER_BIT, TRANSFER_DST_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            feedback_compact_buffer_,
            feedback_compact_memory_,
            std::source_location::current(),
            total_bytes, nullptr);
        feedback_compact_mapped_ = static_cast<uint32_t*>(
            device_->mapMemory(feedback_compact_memory_, total_bytes, 0));
        if (!feedback_compact_mapped_) {
            std::printf("[RVT] FATAL: feedback_compact_buffer mapMemory "
                        "returned null — VT streaming will be broken.\n");
        } else {
            // Zero the counter slot at start; entries are uninitialized
            // (compute pass writes them and counter tells CPU how many
            // are valid).
            std::memset(feedback_compact_mapped_, 0, total_bytes);
        }
    }

    // ── 4b. Persistently map the two HOST_COHERENT SSBOs ───────────
    // Page-table + meta both updated incrementally throughout the
    // session.  Mapping once and writing through the pointer is the
    // only safe and fast pattern; per-entry mapMemory at non-aligned
    // offsets has been observed to corrupt state on some drivers.
    //
    // If the underlying memory was left mapped by Helper::createBuffer
    // (some implementations do that for HOST_VISIBLE init data), the
    // map below will fail.  We log loudly so the misconfiguration is
    // visible — without a valid persistent pointer the streamer's
    // page-table writes silently no-op and surfaces never become
    // resident.
    {
        void* p = device_->mapMemory(
            page_table_memory_,
            uint64_t(kMaxPageTableEntries) * sizeof(uint32_t), 0);
        page_table_mapped_ = static_cast<uint32_t*>(p);
        if (!page_table_mapped_) {
            std::printf("[RVT] FATAL: page_table_buffer persistent mapMemory "
                        "returned null — streamer page-table writes will be "
                        "lost.  Check that Helper::createBuffer unmaps after "
                        "the init-data memcpy.\n");
        }
    }
    {
        void* p = device_->mapMemory(
            meta_memory_,
            uint64_t(kMaxVirtualTextures) * sizeof(VirtualTextureMeta), 0);
        meta_mapped_ = static_cast<VirtualTextureMeta*>(p);
        if (!meta_mapped_) {
            std::printf("[RVT] FATAL: meta_buffer persistent mapMemory "
                        "returned null — registerMaterial meta writes will "
                        "be lost.\n");
        }
    }

    // ── 4c. Streamer state ─────────────────────────────────────────
    // Slot 0..kVtPagesPerLayer-1 all start FREE.  Pushed in reverse
    // so that pop_back returns ascending slot indices first — keeps
    // the pool's debug visualization legible (early registrations
    // show up at top-left rather than bottom-right).
    slot_info_.assign(kVtPagesPerLayer, SlotInfo{});
    free_slots_.reserve(kVtPagesPerLayer);
    for (uint32_t i = 0; i < kVtPagesPerLayer; ++i) {
        free_slots_.push_back(kVtPagesPerLayer - 1u - i);
    }
    vt_cache_.reserve(kMaxVirtualTextures);

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

    // ── 6. GPU feedback compaction pipeline ────────────────────────
    // Built last because it consumes the feedback_buffer_ +
    // feedback_compact_buffer_ created above.  The compute pass is
    // dispatched once per frame from compactFeedback() (after the
    // cluster bindless draw) and produces the per-FIF-slot compact
    // request list that the next-next frame's tick() reads.
    initFeedbackCompactPipeline();
}

// ── GPU feedback compaction: pipeline init ───────────────────────────
// Single descriptor set (one set, two SSBO bindings) + push constants.
// Pipeline binds for every dispatch; the per-frame slot offset is a
// push-constant, not a separate descriptor set per frame.
void VirtualTextureManager::initFeedbackCompactPipeline() {
    // Descriptor set layout: two storage buffers, both compute-stage.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = er::helper::getBufferDescriptionSetLayoutBinding(
            0, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_BUFFER);
        bindings[1] = er::helper::getBufferDescriptionSetLayoutBinding(
            1, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_BUFFER);
        compact_desc_set_layout_ =
            device_->createDescriptorSetLayout(bindings);
    }

    // Push constants: feedback_total + max_entries + slot_offset_uints
    // + pad → 16 B (vendor-portable minimum push range; well below the
    // 128 B guaranteed limit).
    struct CompactPC {
        uint32_t feedback_total;
        uint32_t max_entries;
        uint32_t compact_slot_offset_uints;
        uint32_t pad0;
    };
    compact_pipeline_layout_ = er::helper::createComputePipelineLayout(
        device_, { compact_desc_set_layout_ }, sizeof(CompactPC));

    compact_pipeline_ = er::helper::createComputePipeline(
        device_, compact_pipeline_layout_,
        "vt_compact_feedback_comp.spv",
        std::source_location::current());

    // Allocate the (single) descriptor set and write the buffer
    // bindings once — they don't change frame-to-frame.  feedback_buffer_
    // and feedback_compact_buffer_ are stable for the lifetime of the
    // manager.
    compact_desc_set_ = device_->createDescriptorSets(
        descriptor_pool_, compact_desc_set_layout_, 1)[0];

    er::WriteDescriptorList writes;
    writes.reserve(2);
    er::Helper::addOneBuffer(
        writes,
        compact_desc_set_,
        er::DescriptorType::STORAGE_BUFFER,
        0,
        feedback_buffer_,
        feedback_buffer_->getSize());
    er::Helper::addOneBuffer(
        writes,
        compact_desc_set_,
        er::DescriptorType::STORAGE_BUFFER,
        1,
        feedback_compact_buffer_,
        feedback_compact_buffer_->getSize());
    device_->updateDescriptorSets(writes);
}

// ─── Pool-owned descriptor cleanup / re-allocation ─────────────────────
//
// Same playbook as ClusterRenderer::onDescriptorPoolDestroyed /
// recreateDescriptorSets.  When the application's main descriptor pool
// is torn down in cleanupSwapChain, compact_desc_set_ becomes a
// dangling Vulkan handle.  tick() and compactFeedback() bind that set
// every frame, so the very first frame after a resize crashes inside
// the NVIDIA driver dispatch with an `EXCEPTION_ACCESS_VIOLATION_READ`
// at a tiny offset (e.g. 0x38) — reading a member of a destroyed
// descriptor-pool internal struct via a handle that's been freed.
void VirtualTextureManager::onDescriptorPoolDestroyed() {
    compact_desc_set_.reset();
}

void VirtualTextureManager::recreateDescriptorSets(
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) {
    // Skip if init never happened (VT pipeline not built yet) — the
    // layout is constructed by the same code that allocated the
    // original set, so its absence means there's nothing to restore.
    if (!compact_desc_set_layout_) return;
    descriptor_pool_ = descriptor_pool;
    compact_desc_set_ = device_->createDescriptorSets(
        descriptor_pool_, compact_desc_set_layout_, 1)[0];
    // Re-write the (stable) feedback-buffer bindings — same writes
    // the initial allocation site issues.  feedback_buffer_ and
    // feedback_compact_buffer_ are owned by this manager and survive
    // the swap-chain teardown, so the same buffer handles are valid.
    er::WriteDescriptorList writes;
    writes.reserve(2);
    er::Helper::addOneBuffer(
        writes, compact_desc_set_,
        er::DescriptorType::STORAGE_BUFFER, 0,
        feedback_buffer_, feedback_buffer_->getSize());
    er::Helper::addOneBuffer(
        writes, compact_desc_set_,
        er::DescriptorType::STORAGE_BUFFER, 1,
        feedback_compact_buffer_,
        feedback_compact_buffer_->getSize());
    device_->updateDescriptorSets(writes);
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
        if (page_table_mapped_) {
            device_->unmapMemory(page_table_memory_);
            page_table_mapped_ = nullptr;
        }
        device_->destroyBuffer(page_table_buffer_);
        device_->freeMemory(page_table_memory_);
    }
    if (meta_buffer_) {
        if (meta_mapped_) {
            device_->unmapMemory(meta_memory_);
            meta_mapped_ = nullptr;
        }
        device_->destroyBuffer(meta_buffer_);
        device_->freeMemory(meta_memory_);
    }
    if (feedback_buffer_) {
        device_->destroyBuffer(feedback_buffer_);
        device_->freeMemory(feedback_memory_);
    }
    if (feedback_compact_buffer_) {
        if (feedback_compact_mapped_) {
            device_->unmapMemory(feedback_compact_memory_);
            feedback_compact_mapped_ = nullptr;
        }
        device_->destroyBuffer(feedback_compact_buffer_);
        device_->freeMemory(feedback_compact_memory_);
    }
    // Compact compute pipeline + descriptor set layout are owned via
    // shared_ptr's; resetting drops the device-side handles.  The
    // descriptor set itself was allocated from descriptor_pool_ and
    // is freed when the pool is destroyed by the application.
    compact_pipeline_.reset();
    compact_pipeline_layout_.reset();
    compact_desc_set_.reset();
    compact_desc_set_layout_.reset();
}

uint32_t VirtualTextureManager::getPageTableBufferBytes() const {
    // Matches the size passed to createBuffer() in the constructor.
    return static_cast<uint32_t>(kMaxPageTableEntries * sizeof(uint32_t));
}

uint32_t VirtualTextureManager::getFeedbackBufferBytes() const {
    return static_cast<uint32_t>(kVtFeedbackTotal * sizeof(uint32_t));
}

uint32_t VirtualTextureManager::getMetaBufferBytes() const {
    // Matches the size passed to createBuffer() in the constructor.
    return static_cast<uint32_t>(kMaxVirtualTextures * sizeof(VirtualTextureMeta));
}

// ── Per-frame streamer (Phase A: log-only) ────────────────────────────
// Reads the GPU feedback buffer (one tile-key per 8×8 screen block),
// dedupes the keys with a small unordered_set, prints stats, then
// clears the buffer for the next frame via vkCmdFillBuffer.
//
// Called from application.cpp's render loop AFTER the cluster
// bindless GBuffer pass (so the shader's tile-key writes are visible
// to the host) and BEFORE the next frame's command buffer starts
// recording sample work (so the cleared buffer + future Phase-B
// uploads are sequenced ahead of the next sample).  In Phase B this
// becomes the actual streamer: dedupe → upload BC7 / blit → flip
// page-table entries to RESIDENT.
void VirtualTextureManager::tick(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    uint64_t frame_index) {

    if (!feedback_compact_buffer_ || !feedback_compact_mapped_) return;

    // ── Read the GPU-compacted request list ──────────────────────────
    // The compactFeedback() pass dispatched (kVtCompactSlots-1) frames
    // ago wrote to slot ((frame_index - kVtCompactSlots) % kVtCompactSlots)
    // — we waited on its fence at frame start (the standard FIF cycle),
    // so its writes are visible to the host now.  Reading is just
    // pointer arithmetic into the persistent host mapping; no
    // mapMemory call, no 1 MB cold walk, no big std::unordered_set.
    //
    // Slot layout: [counter][entry_0][entry_1] ... [entry_max-1]
    const uint32_t read_slot =
        static_cast<uint32_t>(frame_index % kVtCompactSlots);
    const uint32_t kSlotUints = kVtCompactSlotBytes / sizeof(uint32_t);
    const uint32_t* slot_base =
        feedback_compact_mapped_ + uint64_t(read_slot) * kSlotUints;
    const uint32_t raw_count = slot_base[0];
    // Clamp to capacity in case the GPU saw more requests than we can
    // hold — extras were dropped by the shader (atomic counter still
    // counted them, useful for the diag log to spot pathological frames).
    const uint32_t valid_count = std::min(raw_count, kVtCompactMaxEntries);
    const uint32_t* keys = slot_base + 1u;
    const uint64_t total_writes = raw_count;

    // ── Tiny CPU dedupe over the compacted list ──────────────────────
    // Compaction doesn't dedupe (the GPU shader just bump-allocates
    // and writes), so multiple 8×8 blocks landing on the same VT tile
    // each contribute an entry.  Typical frames see a few thousand
    // entries collapsing to a few hundred uniques — small enough that
    // a vector + sort + unique is faster than std::unordered_set.
    std::vector<uint32_t> unique_keys;
    unique_keys.reserve(valid_count);
    for (uint32_t i = 0; i < valid_count; ++i) {
        const uint32_t k = keys[i];
        if (k != kVtTileKeyInvalid) unique_keys.push_back(k);
    }
    std::sort(unique_keys.begin(), unique_keys.end());
    unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()),
                      unique_keys.end());

    // ── Streaming: touch resident, separate misses ───────────────────
    // First pass: every resident request bumps its slot's LRU age;
    // every miss goes onto a separate list we'll prioritize before
    // the upload-budget cap is applied.
    std::vector<uint32_t> miss_keys;
    miss_keys.reserve(unique_keys.size());
    for (uint32_t key : unique_keys) {
        auto it = tile_to_slot_.find(key);
        if (it != tile_to_slot_.end()) {
            touchSlot(it->second);
        } else {
            miss_keys.push_back(key);
        }
    }

    // ── Coarse-to-fine prioritization ────────────────────────────────
    // Sort misses by MIP DESCENDING — higher mip = more downsampled =
    // covers a larger source-texture area per tile = typically what a
    // surface viewed from far away requests.  We satisfy those first
    // because:
    //   1. One high-mip tile replaces the "pinned smallest mip" blurry
    //      fallback for an entire surface, which is what the eye
    //      notices most after a camera jump.
    //   2. Coarse tiles are the SAME 6480 B BC payload as fine tiles
    //      (per-tile cost is constant in this design), so prioritizing
    //      them costs nothing — it's just a re-ordering of the
    //      bounded budget.
    //   3. Once the coarse coverage is in, subsequent frames stream
    //      finer detail in (mip 0, 1, ...) without the user noticing
    //      the blur step-down — the higher mip is already a
    //      reasonable approximation, so refinement is incremental
    //      rather than a hard pop from "magenta / pinned 1×1" to
    //      "full detail".
    //
    // Tile-key bit layout (must match makeTileKey / vt_types.glsl.h):
    //   bits  0..13   vt_index
    //   bits 14..17   mip        ← extracted with (key >> 14) & 0xF
    //   bits 18..24   page_x
    //   bits 25..31   page_y
    // Tiebreak by raw key for a deterministic order across frames
    // (helps the LRU behave consistently under repeated camera poses).
    std::sort(miss_keys.begin(), miss_keys.end(),
        [](uint32_t a, uint32_t b) {
            const uint32_t mip_a = (a >> 14) & 0xFu;
            const uint32_t mip_b = (b >> 14) & 0xFu;
            if (mip_a != mip_b) return mip_a > mip_b;   // higher mip first
            return a < b;
        });

    // Take up to kStreamerUploadsPerFrame from the priority-sorted
    // miss list.  Anything past the cap gets dropped this frame and
    // re-requested by the shader next frame; if the camera keeps
    // moving, the new requests preempt the old.
    std::vector<uint32_t> to_upload;
    to_upload.reserve(kStreamerUploadsPerFrame);
    for (uint32_t key : miss_keys) {
        if (to_upload.size() >= kStreamerUploadsPerFrame) break;
        to_upload.push_back(key);
    }

    // ── Process upload queue ─────────────────────────────────────────
    // KEY CHANGE FROM PREVIOUS DESIGN: uploads are recorded into the
    // PASSED-IN frame command buffer (cmd_buf), NOT a separate
    // transient command buffer that we'd then submit-and-wait on.
    // The frame's normal vkQueueSubmit picks them up; the per-frame
    // staging-buffer slice (indexed by frame_index % FIF) keeps CPU
    // memcpy's safe against the GPU's still-in-flight reads from the
    // previous slot.  This single change eliminates the >20 ms
    // stall that previously dominated VT tick on bottlenecked frames.
    //
    // Sequencing: tick() is called at frame start, BEFORE the cluster
    // bindless draw later in the frame samples the pool.  Recording
    // the pool transitions + copyBufferToImage / blitImage at the head
    // of cmd_buf naturally orders the upload before any sampling.
    uint32_t uploads_done = 0;
    if (!to_upload.empty()) {
        er::ImageResourceInfo from_read{
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        er::ImageResourceInfo to_xfer_dst{
            er::ImageLayout::TRANSFER_DST_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        er::ImageResourceInfo to_xfer_src{
            er::ImageLayout::TRANSFER_SRC_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };

        for (uint32_t l = 0; l < uint32_t(VtLayer::COUNT); ++l) {
            cmd_buf->addImageBarrier(layer_pools_[l].texture.image,
                from_read, to_xfer_dst, 0, kVtPoolMipLevels, 0, 1);
        }
        // Track which source images we've transitioned to TRANSFER_SRC
        // so each gets one barrier in + one out per tick.
        std::unordered_set<er::Image*> src_transitioned;
        auto trans_src_once = [&](const std::shared_ptr<er::Image>& im) {
            if (!im) return;
            if (src_transitioned.insert(im.get()).second) {
                cmd_buf->addImageBarrier(im, from_read, to_xfer_src, 0, 1, 0, 1);
            }
        };

        // Per-tick upload counter doubles as the staging-buffer slot
        // index inside this frame's FIF slice.  uploadTileAllLayers
        // computes the absolute byte offset by adding the frame slice
        // base (stashed in m_active_staging_base_off_ for the duration
        // of the call).
        const uint64_t kPerFrameStagingBytes =
            uint64_t(kStreamerUploadsPerFrame) * 2u * kBc7BytesPerEntry;
        const uint64_t frame_staging_base =
            uint64_t(frame_index % kVtCompactSlots) * kPerFrameStagingBytes;
        active_staging_base_off_ = frame_staging_base;

        uint32_t staging_slot = 0;
        for (uint32_t key : to_upload) {
            uint32_t vt, mip, px, py;
            decodeTileKey(key, vt, mip, px, py);
            if (vt >= vt_cache_.size() || vt_cache_[vt].mip_count == 0) continue;
            const VtCacheEntry& cache = vt_cache_[vt];
            trans_src_once(cache.mr_ao_src);
            trans_src_once(cache.emissive_src);

            uint32_t s = allocSlot();
            if (s >= kVtPagesPerLayer) {
                // Pool exhausted (all pinned, or LRU back was pinned
                // and we've nothing to evict).  Skip rest of frame's
                // upload queue — try again next frame after some
                // tiles age out.
                break;
            }
            uploadTileAllLayers(cmd_buf, s, vt, mip, px, py, staging_slot);
            ++staging_slot;

            const uint32_t mip_off = vtMipOffsetWithinVt(
                cache.pages_x, cache.pages_y, mip);
            const uint32_t local_idx =
                py * vtMipPagesAt(cache.pages_x, mip) + px;
            const uint32_t entry_idx = cache.page_table_offset + mip_off + local_idx;
            markSlotResident(s, key, entry_idx, /*pinned*/ false);
            ++uploads_done;
        }

        for (uint32_t l = 0; l < uint32_t(VtLayer::COUNT); ++l) {
            cmd_buf->addImageBarrier(layer_pools_[l].texture.image,
                to_xfer_dst, from_read, 0, kVtPoolMipLevels, 0, 1);
        }
        for (er::Image* p : src_transitioned) {
            // Reverse barrier — find the shared_ptr by walking caches
            // (cheap; ≤ N source images).
            for (auto& c : vt_cache_) {
                if (c.mr_ao_src.get()    == p) cmd_buf->addImageBarrier(c.mr_ao_src,    to_xfer_src, from_read, 0, 1, 0, 1);
                if (c.emissive_src.get() == p) cmd_buf->addImageBarrier(c.emissive_src, to_xfer_src, from_read, 0, 1, 0, 1);
            }
        }
        // No transient submit, no wait — uploads ride the frame's
        // normal command buffer.  Persistent staging buffer is reused
        // every frame; the FIF slicing above keeps CPU memcpy's safe.
    }

    // ── Publish queued page-table entry updates to the GPU ───────────
    // After pool slots have been transitioned back to SHADER_READ,
    // record one vkCmdUpdateBuffer per dirty entry that
    // allocSlot/markSlotResident appended.  The updates are inline-
    // data writes in the command buffer (the renderer wrapper passes
    // a copy through to vkCmdUpdateBuffer), executed at GPU TRANSFER
    // stage in submission order — AFTER the previous frame's cluster
    // bindless reads complete (queue ordering) and AFTER this frame's
    // pool uploads have populated the new tile data (the dirty
    // updates run after the pool-back-to-SHADER_READ barriers above).
    //
    // Wrap with TRANSFER_WRITE → SHADER_READ buffer barriers so the
    // cluster bindless draw later in this frame sees the updated
    // entries via a real Vulkan memory dependency, not by relying on
    // the buffer's HOST_COHERENT property (which we've stopped using
    // from the per-frame path entirely).
    if (!page_table_dirty_entries_.empty()) {
        // Pre-barrier: anything earlier in this cmd buffer that read
        // page_table_buffer_ as a shader storage read must complete
        // before our transfer writes overwrite those entries.
        er::BufferResourceInfo from_shader_read{
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_2_FLAG_BITS(PipelineStage,
                FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT) };
        er::BufferResourceInfo to_xfer_write{
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        cmd_buf->addBufferBarrier(
            page_table_buffer_, from_shader_read, to_xfer_write);

        // One vkCmdUpdateBuffer per dirty entry.  Each is 4 bytes
        // inline; for a typical 100–250 dirty entries per frame
        // that's a few KB of cmd buffer space, negligible.
        for (const auto& d : page_table_dirty_entries_) {
            const uint64_t off = uint64_t(d.entry_idx) * sizeof(uint32_t);
            cmd_buf->updateBuffer(
                page_table_buffer_,
                off,
                /*size*/ sizeof(uint32_t),
                /*data*/ &d.packed_value);
        }
        page_table_dirty_entries_.clear();

        // Post-barrier: subsequent shader reads (cluster bindless,
        // VT compaction, etc.) must wait for our transfer writes.
        er::BufferResourceInfo from_xfer_write{
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        er::BufferResourceInfo to_shader_read{
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_2_FLAG_BITS(PipelineStage,
                FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT) };
        cmd_buf->addBufferBarrier(
            page_table_buffer_, from_xfer_write, to_shader_read);
    }

    // ── Build per-slot status snapshot for the debug viewer ─────────
    // One byte per slot, row-major.  Bit 0 = resident, bit 1 = active
    // this frame (had a feedback request), bit 2 = pinned.  The view
    // queries this each frame and draws a coloured grid; orders of
    // magnitude more useful than the raw pool-image atlas for seeing
    // "the camera only touches 200 of the 4090 resident tiles".
    if (slot_status_.size() != kVtPagesPerLayer) {
        slot_status_.assign(kVtPagesPerLayer, 0u);
    } else {
        std::fill(slot_status_.begin(), slot_status_.end(), uint8_t(0));
    }
    slots_active_ = 0;
    slots_resident_ = 0;
    slots_pinned_ = 0;
    // 1) Mark every resident slot (and pinned-vs-not) — walk
    //    tile_to_slot_ which holds (tile_key → slot_index).
    for (const auto& kv : tile_to_slot_) {
        uint32_t s = kv.second;
        if (s >= kVtPagesPerLayer) continue;
        uint8_t b = kSlotStatusResident;
        if (s < slot_info_.size() && slot_info_[s].pinned) {
            b |= kSlotStatusPinned;
            ++slots_pinned_;
        }
        slot_status_[s] |= b;
        ++slots_resident_;
    }
    // 2) Mark every slot whose tile_key was requested by the GPU
    //    this frame — straight from the deduped feedback set.
    for (uint32_t key : unique_keys) {
        auto it = tile_to_slot_.find(key);
        if (it != tile_to_slot_.end() && it->second < kVtPagesPerLayer) {
            if (!(slot_status_[it->second] & kSlotStatusActive)) {
                slot_status_[it->second] |= kSlotStatusActive;
                ++slots_active_;
            }
        }
    }

    // ── Log stats ────────────────────────────────────────────────────
    if ((frame_index % 60u) == 0u) {
        // Pool usage = (slots in use) / (slots per layer).  100% means
        // the LRU is evicting on every upload — the working set is
        // larger than the pool can hold, and tiles will get streamed
        // in/out repeatedly as the camera moves (thrashing).  Healthy
        // steady-state is below ~95%; chronic 100% means raise
        // kVtPagesPerLayer or reduce camera/scene density.
        const size_t used = tile_to_slot_.size();
        const float  pct_resident = (kVtPagesPerLayer == 0u) ? 0.0f
                          : (100.0f * float(used) / float(kVtPagesPerLayer));
        const float  pct_active = (used == 0u) ? 0.0f
                          : (100.0f * float(slots_active_) / float(used));
        // Mip distribution of THIS FRAME'S uploads: shows whether the
        // coarse-first prioritization is doing its job.  After a
        // camera jump the first few frames should be dominated by
        // high-mip uploads, then taper to mip 0 as the working set
        // refines.  Format: "mips[hi..lo]=N,N,...,N" (only printed
        // for non-zero counts so a steady state with all-mip-0
        // requests stays readable).
        uint32_t mip_hist[16] = {};
        uint32_t highest_mip_seen = 0;
        for (uint32_t k : to_upload) {
            uint32_t m = (k >> 14) & 0xFu;
            if (m < 16) {
                ++mip_hist[m];
                highest_mip_seen = std::max(highest_mip_seen, m);
            }
        }
        char mip_buf[128];
        int  mip_off = std::snprintf(mip_buf, sizeof(mip_buf), "mips[");
        for (int m = int(highest_mip_seen); m >= 0; --m) {
            mip_off += std::snprintf(mip_buf + mip_off,
                                     sizeof(mip_buf) - mip_off,
                                     "%s%u", (m == int(highest_mip_seen) ? "" : ","),
                                     mip_hist[m]);
        }
        mip_off += std::snprintf(mip_buf + mip_off,
                                 sizeof(mip_buf) - mip_off, "]");
        // ── Per-frame VT stats line → vt_pool.log ──────────────────
        // Used to printf to stdout every frame.  Now appended via the
        // shared vtLog() helper so the registerMaterial line shows up
        // in the same file with consistent ordering.
        {
            auto& log = vtLog();
            if (log.is_open()) {
                char line[512];
                std::snprintf(line, sizeof(line),
                    "[RVT] frame=%llu unique=%zu writes=%llu uploads=%u "
                    "free=%zu lru=%zu resident=%zu/%u (%.1f%%) "
                    "active=%u (%.1f%% of resident) pinned=%u  %s\n",
                    static_cast<unsigned long long>(frame_index),
                    unique_keys.size(),
                    static_cast<unsigned long long>(total_writes),
                    uploads_done,
                    free_slots_.size(),
                    resident_lru_.size(),
                    used, kVtPagesPerLayer, pct_resident,
                    slots_active_, pct_active,
                    slots_pinned_,
                    mip_buf);
                log << line;
                log.flush();
            }
        }
    }

    // The feedback buffer's per-frame clear lives in compactFeedback()
    // now — the compute pass reads the buffer first (so we can't clear
    // it before the dispatch), then we fill it for the next frame's
    // fragment-shader writes.  See VirtualTextureManager::compactFeedback.
}

// ── GPU feedback compaction: dispatch + clear ────────────────────────
// Runs after the cluster bindless draw of the current frame.  Writes
// the compact list into feedback_compact_buffer_'s slot
// (frame_index % kVtCompactSlots), then clears feedback_buffer_ so
// the NEXT frame's shader writes start from a clean slate.
//
// Sequencing inside cmd_buf:
//   1. Buffer barrier: cluster fragment shader writes → compute read
//   2. Clear current frame's compact slot counter to 0
//   3. Buffer barrier: counter clear → compute atomicAdd
//   4. Dispatch vt_compact_feedback.comp
//   5. Buffer barrier: compute writes → host read (frame fence
//      makes them host-visible at next frame's wait point)
//   6. Clear feedback_buffer_ for next frame
//   7. Buffer barrier: clear → next frame's fragment writes
void VirtualTextureManager::compactFeedback(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    uint64_t frame_index) {
    if (!compact_pipeline_ || !feedback_buffer_ ||
        !feedback_compact_buffer_) return;

    const uint32_t write_slot =
        static_cast<uint32_t>(frame_index % kVtCompactSlots);
    const uint32_t kSlotUints = kVtCompactSlotBytes / sizeof(uint32_t);
    const uint64_t slot_byte_off = uint64_t(write_slot) * kVtCompactSlotBytes;
    const uint64_t fb_bytes =
        uint64_t(kVtFeedbackTotal) * sizeof(uint32_t);

    // 1. Cluster shader's atomic-store writes → compute shader reads.
    {
        er::BufferResourceInfo from_frag_rw{
            SET_2_FLAG_BITS(Access, SHADER_WRITE_BIT, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        er::BufferResourceInfo to_compute_read{
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
        cmd_buf->addBufferBarrier(
            feedback_buffer_, from_frag_rw, to_compute_read);
    }

    // 2. Reset this frame's compact slot counter.  Only the first
    //    4 bytes need clearing (the entries[] tail is overwritten
    //    in-order by the atomicAdd indices, with whatever's beyond
    //    the count being garbage we don't read).  vkCmdFillBuffer
    //    requires a multiple of 4 bytes, which 4 satisfies.
    cmd_buf->fillBuffer(feedback_compact_buffer_,
                        slot_byte_off,
                        /*size*/ sizeof(uint32_t),
                        /*data*/ 0u);

    {
        er::BufferResourceInfo from_xfer{
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        er::BufferResourceInfo to_compute_rw{
            SET_2_FLAG_BITS(Access, SHADER_WRITE_BIT, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
        cmd_buf->addBufferBarrier(
            feedback_compact_buffer_, from_xfer, to_compute_rw);
    }

    // 3. Bind + push constants + dispatch.
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE, compact_pipeline_);
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        compact_pipeline_layout_,
        { compact_desc_set_ });

    struct CompactPC {
        uint32_t feedback_total;
        uint32_t max_entries;
        uint32_t compact_slot_offset_uints;
        uint32_t pad0;
    } pc;
    pc.feedback_total            = kVtFeedbackTotal;
    pc.max_entries               = kVtCompactMaxEntries;
    pc.compact_slot_offset_uints = uint32_t(write_slot) * kSlotUints;
    pc.pad0                      = 0;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        compact_pipeline_layout_, &pc, sizeof(pc));

    const uint32_t kLocalSize = 64u;
    const uint32_t group_x =
        (kVtFeedbackTotal + kLocalSize - 1u) / kLocalSize;
    cmd_buf->dispatch(group_x, 1, 1);

    // 4. Compute writes → host read (HOST_VISIBLE | HOST_COHERENT
    //    backing memory makes the writes visible to the host once the
    //    frame's submit fence is waited on at next-frame start).
    {
        er::BufferResourceInfo from_compute_w{
            SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
        er::BufferResourceInfo to_host_read{
            SET_FLAG_BIT(Access, HOST_READ_BIT),
            SET_FLAG_BIT(PipelineStage, HOST_BIT) };
        cmd_buf->addBufferBarrier(
            feedback_compact_buffer_, from_compute_w, to_host_read);
    }

    // 5. Clear feedback_buffer_ for next frame's fragment writes.
    cmd_buf->fillBuffer(feedback_buffer_,
                        /*offset*/ 0,
                        /*size*/   fb_bytes,
                        /*data*/   kVtTileKeyInvalid);

    {
        er::BufferResourceInfo from_xfer{
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        er::BufferResourceInfo to_frag_rw{
            SET_2_FLAG_BITS(Access, SHADER_WRITE_BIT, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        cmd_buf->addBufferBarrier(
            feedback_buffer_, from_xfer, to_frag_rw);
    }
}

// ── LRU pool slot allocator ───────────────────────────────────────────
// Pop a free slot first.  If none free, evict the LRU back of the
// resident list — that slot's tile-key gets removed from
// tile_to_slot_, its page-table entry is marked UNRESIDENT (CPU and
// GPU side, since page-table memory is HOST_COHERENT), and the slot
// is recycled for the new caller.  Returns kVtPagesPerLayer iff
// nothing is evictable (every slot is pinned — fail safe).
uint32_t VirtualTextureManager::allocSlot() {
    if (!free_slots_.empty()) {
        uint32_t s = free_slots_.back();
        free_slots_.pop_back();
        return s;
    }
    if (resident_lru_.empty()) {
        return kVtPagesPerLayer;
    }
    uint32_t s = resident_lru_.back();
    SlotInfo& info = slot_info_[s];

    // Evict — clear the tile_to_slot_ mapping, mark the page-table
    // entry UNRESIDENT in CPU shadow + queue a dirty entry for
    // tick() to apply via vkCmdUpdateBuffer.
    //
    // CRITICAL: do NOT write through page_table_mapped_ here.  That
    // path is HOST_COHERENT, meaning the write is visible to the
    // GPU immediately — including to the in-flight previous-frame
    // cluster bindless draw.  If THAT frame samples this slot's old
    // tile and we yank its page-table entry out from under it mid-
    // shader, you get torn reads / wrong-texture flicker.  Queueing
    // the update for cmd_buf-based application instead sequences
    // the change with the rest of this frame's GPU work, AFTER the
    // previous frame's reads have completed.
    auto it = tile_to_slot_.find(info.tile_key);
    if (it != tile_to_slot_.end() && it->second == s) {
        tile_to_slot_.erase(it);
    }
    if (info.entry_idx < page_table_cpu_.size()) {
        page_table_cpu_[info.entry_idx] = kPageEntryUnresident;
        page_table_dirty_entries_.push_back(
            { info.entry_idx, kPageEntryUnresident });
    }
    resident_lru_.pop_back();
    info.tile_key  = kVtTileKeyInvalid;
    info.entry_idx = 0xFFFFFFFFu;
    info.pinned    = false;
    return s;
}

void VirtualTextureManager::markSlotResident(
    uint32_t slot, uint32_t tile_key,
    uint32_t entry_idx, bool pinned) {
    SlotInfo& info = slot_info_[slot];
    info.tile_key  = tile_key;
    info.entry_idx = entry_idx;
    info.pinned    = pinned;
    if (!pinned) {
        resident_lru_.push_front(slot);
        info.lru_iter = resident_lru_.begin();
    }
    tile_to_slot_[tile_key] = slot;

    // Pack the page-table entry.  Update the CPU shadow now (so the
    // VT cache logic stays consistent), and queue a dirty entry that
    // tick() will publish to the GPU via vkCmdUpdateBuffer in the
    // frame's command buffer — sequenced AFTER the matching pool
    // upload's TRANSFER_DST → SHADER_READ barrier, so the shader
    // never sees "tile is now resident at slot S" before slot S
    // actually contains the new tile data.
    const uint32_t phys_x = slot % kVtPagesAcross;
    const uint32_t phys_y = slot / kVtPagesAcross;
    const uint32_t packed =
        packPageEntry(phys_x, phys_y, /*resident*/ true);
    page_table_cpu_[entry_idx] = packed;
    page_table_dirty_entries_.push_back({ entry_idx, packed });
}

// Flush all queued page-table entry updates to the GPU buffer via
// a TRANSIENT command buffer + submit-and-wait.  Used by
// registerMaterial right before it returns, so the pinned-mip
// page-table entries it just wrote are visible to the very next
// frame's cluster bindless draw — no "magenta / pinned-mip
// fallback" flash on the frame a material first becomes available.
//
// No-op when the dirty queue is empty (no allocations happened).
void VirtualTextureManager::flushDirtyPageTableEntriesViaTransient() {
    if (page_table_dirty_entries_.empty()) return;
    if (!page_table_buffer_) return;

    auto cmd = device_->setupTransientCommandBuffer();

    // Match the in-flight barrier order used by tick():
    // SHADER_READ → TRANSFER_WRITE → SHADER_READ around the updates.
    // The SHADER_READ source is conservative — registerMaterial runs
    // synchronously and there's no concurrent in-flight cluster
    // bindless draw at this point, but the barrier costs nothing and
    // keeps the sync semantics identical to the per-frame path.
    er::BufferResourceInfo from_shader_read{
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_2_FLAG_BITS(PipelineStage,
            FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT) };
    er::BufferResourceInfo to_xfer_write{
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    cmd->addBufferBarrier(
        page_table_buffer_, from_shader_read, to_xfer_write);

    for (const auto& d : page_table_dirty_entries_) {
        const uint64_t off = uint64_t(d.entry_idx) * sizeof(uint32_t);
        cmd->updateBuffer(
            page_table_buffer_,
            off,
            /*size*/ sizeof(uint32_t),
            /*data*/ &d.packed_value);
    }
    page_table_dirty_entries_.clear();

    er::BufferResourceInfo from_xfer_write{
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    er::BufferResourceInfo to_shader_read{
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_2_FLAG_BITS(PipelineStage,
            FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT) };
    cmd->addBufferBarrier(
        page_table_buffer_, from_xfer_write, to_shader_read);

    device_->submitAndWaitTransientCommandBuffer();
}

void VirtualTextureManager::touchSlot(uint32_t slot) {
    SlotInfo& info = slot_info_[slot];
    if (info.pinned || info.tile_key == kVtTileKeyInvalid) return;
    // Move to LRU front.  std::list::splice is O(1).
    resident_lru_.splice(resident_lru_.begin(), resident_lru_, info.lru_iter);
    info.lru_iter = resident_lru_.begin();
}

// Source pixels are RGBA8.  Per-layer pool format may be different —
// this helper extracts a kVtPageSize sub-region from the source AND
// converts to the layer's pool format, writing into a tightly-packed
// destination buffer of bytesPerTexel(layer) * page_size² bytes.
//
//   src_pixels:   tightly-packed RGBA8 source, src_w × src_h.
//   src_x/src_y:  top-left of the page within the source (in texels).
//   src_w/src_h:  source dimensions in texels.
//   dst:          destination buffer (must hold one page's worth).
//   layer:        determines output channel layout.
//
// Pages whose region extends past the source edge are CLAMPED — the
// extra pixels replicate the source's right/bottom edge.
//
// CURRENTLY UNUSED — the production path is registerTextureFromImage
// which works with GPU-resident source images via vkCmdBlitImage.
// Kept as reference for a future CPU-pixels registerTexture rewrite
// that will need its own per-mip extraction (call this once per mip
// after boxDownsampleRgba8).
[[maybe_unused]] static void extractAndConvertPage(
    const uint8_t* src_pixels,
    uint32_t src_x, uint32_t src_y,
    uint32_t src_w, uint32_t src_h,
    VtLayer  layer,
    uint8_t* dst) {

    // Destination bytes-per-pixel must match the pool image's format
    // (see layerFormat() — all four layers are 4-channel RGBA8 in v1
    // for vkCmdCopyImage compatibility with RGBA8 source images).
    // Even NORMAL uses 4 bpp here despite only the .rg channels being
    // meaningful: vkCmdCopyBufferToImage interprets the staging
    // buffer through the destination IMAGE's format, so a 2 bpp
    // staging row would be read by the driver at 4 bpp and shift
    // every subsequent row by half — corrupting the upload.  Pad
    // .b/.a to 0 and let the sampler read .rg only.
    const uint32_t bpp_dst = 4u;

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
                    // Tangent-space normal: keep the X/Y channels in
                    // .rg; .ba zero-padded.  Z is reconstructed in
                    // the sampling shader as sqrt(1 - x² - y²) under
                    // the unit-normal assumption.
                    d[0] = s[0]; d[1] = s[1]; d[2] = 0; d[3] = 0;
                    break;
                default: break;
            }
        }
    }
}

std::shared_ptr<er::ImageView> VirtualTextureManager::getPoolImageView(
    VtLayer layer) const {
    return layer_pools_[uint32_t(layer)].texture.view;
}

std::shared_ptr<er::ImageView> VirtualTextureManager::getPoolDebugView(
    VtLayer layer) const {
    return layer_pools_[uint32_t(layer)].mip0_view;
}

// ── Helper: upload one non-ALBEDO layer to already-allocated slots
//
// Used by registerMaterial after the shared slot allocator (called by
// registerAlbedoBC7) has populated the per-VT page-table window with
// (phys_x, phys_y, resident=1) for every successfully-allocated entry.
// We walk that window, recover each entry's mip + (px, py) from the
// meta, build vkCmdBlitImage regions covering both pool mip levels
// per entry, and submit one transient command buffer.
//
// This is what makes the four pool layouts identical: every layer's
// upload uses the SAME phys positions for the SAME (vt, mip, page),
// so slot (X, Y) in any layer pool image always belongs to the same
// (vt, page) tuple.
void VirtualTextureManager::uploadLayerByBlitToSlots(
    VtLayer layer,
    const std::shared_ptr<er::Image>& src_image,
    uint32_t width,
    uint32_t height,
    uint32_t vt_index) {

    const VirtualTextureMeta& meta = meta_cpu_[vt_index];
    const uint32_t mip_count   = meta.mip_count;
    const uint32_t pages_x     = meta.pages_x;
    const uint32_t pages_y     = meta.pages_y;
    const uint32_t total_pages =
        vtTotalPagesAllMips(pages_x, pages_y, mip_count);

    std::vector<er::ImageBlitInfo> regions;
    regions.reserve(uint64_t(total_pages) * 2u);     // mip 0 + mip 1 per entry

    uint32_t entry_local = 0;
    for (uint32_t k = 0; k < mip_count; ++k) {
        const uint32_t mip_px      = vtMipPagesAt(pages_x, k);
        const uint32_t mip_py      = vtMipPagesAt(pages_y, k);
        const uint32_t src_page_sz = kVtPageSize << k;
        for (uint32_t py = 0; py < mip_py; ++py) {
            for (uint32_t px = 0; px < mip_px; ++px) {
                const uint32_t entry_index = meta.page_table_offset + entry_local;
                const uint32_t packed = page_table_cpu_[entry_index];
                ++entry_local;
                // Skip slots the albedo pass couldn't allocate.
                if (((packed >> 16u) & 1u) == 0u) continue;

                const uint32_t phys_page_x = packed & 0xFFu;
                const uint32_t phys_page_y = (packed >> 8u) & 0xFFu;

                const uint32_t src_x0 = px * src_page_sz;
                const uint32_t src_y0 = py * src_page_sz;
                const uint32_t src_x1 = std::min(src_x0 + src_page_sz, width);
                const uint32_t src_y1 = std::min(src_y0 + src_page_sz, height);

                er::ImageBlitInfo r0{};
                r0.src_subresource.aspect_mask      = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
                r0.src_subresource.mip_level        = 0;
                r0.src_subresource.base_array_layer = 0;
                r0.src_subresource.layer_count      = 1;
                r0.src_offsets[0] = glm::ivec3(int32_t(src_x0), int32_t(src_y0), 0);
                r0.src_offsets[1] = glm::ivec3(int32_t(src_x1), int32_t(src_y1), 1);
                r0.dst_subresource.aspect_mask      = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
                r0.dst_subresource.mip_level        = 0;
                r0.dst_subresource.base_array_layer = 0;
                r0.dst_subresource.layer_count      = 1;
                r0.dst_offsets[0] = glm::ivec3(
                    int32_t(phys_page_x * kVtPageSize),
                    int32_t(phys_page_y * kVtPageSize), 0);
                r0.dst_offsets[1] = glm::ivec3(
                    int32_t((phys_page_x + 1) * kVtPageSize),
                    int32_t((phys_page_y + 1) * kVtPageSize), 1);
                regions.push_back(r0);

                er::ImageBlitInfo r1 = r0;
                r1.dst_subresource.mip_level = 1;
                r1.dst_offsets[0] = glm::ivec3(
                    int32_t(phys_page_x * (kVtPageSize / 2)),
                    int32_t(phys_page_y * (kVtPageSize / 2)), 0);
                r1.dst_offsets[1] = glm::ivec3(
                    int32_t((phys_page_x + 1) * (kVtPageSize / 2)),
                    int32_t((phys_page_y + 1) * (kVtPageSize / 2)), 1);
                regions.push_back(r1);
            }
        }
    }

    if (regions.empty()) return;

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
        from_read, to_xfer_dst, 0, kVtPoolMipLevels, 0, 1);

    cmd_buf->blitImage(
        src_image, er::ImageLayout::TRANSFER_SRC_OPTIMAL,
        layer_pools_[uint32_t(layer)].texture.image,
        er::ImageLayout::TRANSFER_DST_OPTIMAL,
        regions, er::Filter::LINEAR);

    cmd_buf->addImageBarrier(src_image,
        to_xfer_src, from_read, 0, 1, 0, 1);
    cmd_buf->addImageBarrier(
        layer_pools_[uint32_t(layer)].texture.image,
        to_xfer_dst, from_read, 0, kVtPoolMipLevels, 0, 1);

    device_->submitAndWaitTransientCommandBuffer();
}

// ── Public: register one material across all four layer pools ────
//
// Allocates ONE set of pool slots (in registerAlbedoBC7) and uploads
// every provided layer's data into those same slots across the
// matching pool images.  The four layer pool layouts are therefore
// identical, and the same vt_id resolves correctly through any
// layer's pool sampler in the shader.  See registerMaterial's
// declaration in virtual_texture.h for caller semantics.
// ── Build per-VT BC7 cache + stash source images ─────────────────────
// GPU-readback the source albedo, build the RGBA8 mip pyramid in CPU
// memory, parallel-encode every (mip, page) at full + half res into
// vt_cache_[vt_index].bc7_albedo.  Total cache size per VT:
// total_pages × kBc7BytesPerEntry  (≈ 7 MB for a 2048² source).
// Source images for non-ALBEDO layers are kept by shared_ptr — the
// streamer issues vkCmdBlitImage from those on demand.
// Decompress any GPU image to CPU RGBA8 via a blit-through-RGBA8
// detour.  vkCmdBlitImage decompresses BC formats during the blit
// (driver-side), so the destination RGBA8 image holds true colour
// data we can then copy back to the host.  Caller pays one transient
// command-buffer + small staging-buffer cost per call.
std::vector<uint8_t> VirtualTextureManager::readbackImageToRgba8(
    const std::shared_ptr<er::Image>& src_image,
    uint32_t width, uint32_t height,
    bool dst_is_srgb) {

    const uint64_t bytes = uint64_t(width) * height * 4u;
    std::vector<uint8_t> out(bytes, 0);
    if (!src_image || width == 0 || height == 0) return out;

    // ── 1. Allocate temp RGBA8 image (TRANSFER_DST + TRANSFER_SRC).
    // The temp's format determines what sRGB conversion (if any) the
    // blit's destination write performs:
    //   * R8G8B8A8_SRGB: source decode → linear → re-encode as sRGB.
    //     Round-trip preserves sRGB-encoded bytes.  Use for ALBEDO.
    //   * R8G8B8A8_UNORM: source decode → linear → write raw bytes.
    //     Preserves linear values.  Use for NORMAL / MR_AO / EMISSIVE.
    // Picking the wrong one for a given source double-encodes (sRGB
    // for linear data) or double-decodes (UNORM for sRGB data) the
    // values and produces wrong-coloured / wrong-direction outputs.
    er::TextureInfo temp_tex;
    // create2DTextureImage unconditionally builds an ImageView, which
    // requires usage to include at least one of SAMPLED / STORAGE /
    // {COLOR,DEPTH,INPUT}_ATTACHMENT.  Adding SAMPLED_BIT satisfies
    // the spec (VUID-VkImageViewCreateInfo-image-04441); we don't
    // actually sample temp_tex — the blit dst + buffer copy only need
    // TRANSFER_{SRC,DST}_BIT — but the helper insists on a view.
    er::Helper::create2DTextureImage(
        device_,
        dst_is_srgb ? er::Format::R8G8B8A8_SRGB
                    : er::Format::R8G8B8A8_UNORM,
        glm::uvec2(width, height),
        /*mip_levels*/ 1,
        temp_tex,
        SET_3_FLAG_BITS(ImageUsage, TRANSFER_DST_BIT, TRANSFER_SRC_BIT, SAMPLED_BIT),
        er::ImageLayout::TRANSFER_DST_OPTIMAL,
        std::source_location::current());

    // ── 2. Allocate readback buffer (HOST_VISIBLE/COHERENT) ─────────
    std::shared_ptr<er::Buffer>       rb_buf;
    std::shared_ptr<er::DeviceMemory> rb_mem;
    er::Helper::createBuffer(
        device_,
        SET_FLAG_BIT(BufferUsage, TRANSFER_DST_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0, rb_buf, rb_mem,
        std::source_location::current(),
        bytes, nullptr);

    // ── 3. Record + submit blit + copy on a transient command buffer
    auto cmd = device_->setupTransientCommandBuffer();
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

    // Source: SHADER_READ_ONLY → TRANSFER_SRC for the blit.
    cmd->addImageBarrier(src_image, from_read, to_xfer_src, 0, 1, 0, 1);

    // Blit src → temp; LINEAR filter so BC's bilinear-ish decode is
    // sane (NEAREST would still work but might miss sub-block detail).
    //
    // The blit's src region must fit inside the source image (Vulkan
    // VUID-vkCmdBlitImage-srcOffset-00243/00244/pRegions-00215).  Some
    // call paths pass a (width, height) that exceeds the source's
    // mip-0 extent — e.g. a 1×1 placeholder image standing in for a
    // missing albedo while the caller still wants the VT's "logical"
    // width × height.  Clamp the src region to the source's actual
    // extent (recorded by Device::createImage on every Image); the
    // LINEAR-filter blit will upscale to fill the dst.  Upscaling a
    // 1×1 source to N×N produces a constant-colour image — the right
    // behaviour for a placeholder, and no more validation spam.
    const glm::uvec3 src_extent = src_image->getExtent();
    const uint32_t src_w = src_extent.x ? src_extent.x : width;
    const uint32_t src_h = src_extent.y ? src_extent.y : height;
    const int32_t  src_blit_w = int32_t(std::min(width,  src_w));
    const int32_t  src_blit_h = int32_t(std::min(height, src_h));

    std::vector<er::ImageBlitInfo> blits(1);
    blits[0].src_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
    blits[0].src_subresource.mip_level   = 0;
    blits[0].src_subresource.layer_count = 1;
    blits[0].src_offsets[0] = glm::ivec3(0);
    blits[0].src_offsets[1] = glm::ivec3(src_blit_w, src_blit_h, 1);
    blits[0].dst_subresource = blits[0].src_subresource;
    blits[0].dst_offsets[0] = glm::ivec3(0);
    blits[0].dst_offsets[1] = glm::ivec3(int32_t(width), int32_t(height), 1);
    cmd->blitImage(src_image, er::ImageLayout::TRANSFER_SRC_OPTIMAL,
                   temp_tex.image, er::ImageLayout::TRANSFER_DST_OPTIMAL,
                   blits, er::Filter::LINEAR);

    // Restore source to its original layout.
    cmd->addImageBarrier(src_image, to_xfer_src, from_read, 0, 1, 0, 1);

    // Temp: TRANSFER_DST → TRANSFER_SRC for the buffer copy.
    cmd->addImageBarrier(temp_tex.image, to_xfer_dst, to_xfer_src, 0, 1, 0, 1);

    er::BufferImageCopyInfo r{};
    r.buffer_offset       = 0;
    r.buffer_row_length   = 0;        // tightly packed
    r.buffer_image_height = 0;
    r.image_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
    r.image_subresource.mip_level   = 0;
    r.image_subresource.layer_count = 1;
    r.image_offset = glm::ivec3(0);
    r.image_extent = glm::uvec3(width, height, 1);
    cmd->copyImageToBuffer(temp_tex.image, rb_buf, { r },
                           er::ImageLayout::TRANSFER_SRC_OPTIMAL);
    device_->submitAndWaitTransientCommandBuffer();

    // ── 4. Map staging memory + memcpy into out ─────────────────────
    void* mapped = device_->mapMemory(rb_mem, bytes, 0);
    if (mapped) {
        std::memcpy(out.data(), mapped, bytes);
        device_->unmapMemory(rb_mem);
    }

    // ── 5. Cleanup: free temp image + readback buffer ───────────────
    device_->destroyBuffer(rb_buf);
    device_->freeMemory(rb_mem);
    temp_tex.destroy(device_);

    return out;
}

void VirtualTextureManager::encodeAndCacheVt(
    uint32_t vt_index,
    const uint8_t* albedo_pixels,
    const std::shared_ptr<er::Image>& albedo_image,
    const std::shared_ptr<er::Image>& normal_image,
    const std::shared_ptr<er::Image>& mr_ao_image,
    const std::shared_ptr<er::Image>& emissive_image,
    uint32_t width, uint32_t height,
    uint32_t pages_x, uint32_t pages_y, uint32_t mip_count,
    uint32_t page_table_offset) {

    while (vt_cache_.size() <= vt_index) {
        vt_cache_.emplace_back();
    }
    VtCacheEntry& cache = vt_cache_[vt_index];
    cache.width   = width;
    cache.height  = height;
    cache.pages_x = pages_x;
    cache.pages_y = pages_y;
    cache.mip_count = mip_count;
    cache.page_table_offset = page_table_offset;
    cache.albedo_src   = nullptr;       // not used in CPU-pixels path
    cache.normal_src   = normal_image;
    cache.mr_ao_src    = mr_ao_image;
    cache.emissive_src = emissive_image;

    // Fallback path: if the caller didn't have CPU-side pixels (the
    // common case for BC-compressed DDS sources), do a one-time GPU
    // blit-decode + readback to materialise RGBA8 bytes.  The local
    // buffer holds the pixels for the rest of this function only —
    // the real long-lived storage is the per-tile BC7 cache built
    // below.
    std::vector<uint8_t> fallback_pixels;
    if (!albedo_pixels && albedo_image) {
        // ALBEDO: source is sRGB-encoded BC7_SRGB / BC1_SRGB / etc.
        // Use the SRGB temp so the round-trip preserves bytes; bc7enc
        // then encodes those preserved bytes opaquely and the
        // BC7_SRGB pool's sample-time sRGB→linear gives the right
        // colour.
        fallback_pixels = readbackImageToRgba8(albedo_image, width, height,
                                               /*dst_is_srgb*/ true);
        albedo_pixels = fallback_pixels.data();
    }

    // ── ALBEDO pipeline: input CPU pixels → per-mip CPU pyramid →
    //    per-tile bordered RGBA8 extraction → bc7enc → cache.bc7_albedo.
    // No GPU readback.  Each tile is kVtTileSize × kVtTileSize at
    // pool mip 0 (kVtPageSize content + kVtTileBorder per side) and
    // half that at pool mip 1.  Border samples are clamped to the
    // mip's edge for tiles that straddle the source boundary.
    const uint32_t total_pages =
        vtTotalPagesAllMips(pages_x, pages_y, mip_count);
    cache.bc7_albedo.assign(uint64_t(total_pages) * kBc7BytesPerEntry, 0);

    if (!albedo_pixels) {
        std::printf("[RVT] encodeAndCacheVt: null albedo_pixels for vt=%u\n",
                    vt_index);
        return;
    }

    // ── Build CPU mip pyramid from input pixels ──────────────────────
    std::vector<std::vector<uint8_t>> mip_pixels(mip_count);
    std::vector<uint32_t> mip_w(mip_count), mip_h(mip_count);
    mip_pixels[0].assign(albedo_pixels,
                         albedo_pixels + uint64_t(width) * height * 4u);
    mip_w[0] = width; mip_h[0] = height;
    for (uint32_t k = 1; k < mip_count; ++k) {
        mip_w[k] = std::max(1u, mip_w[k - 1] >> 1);
        mip_h[k] = std::max(1u, mip_h[k - 1] >> 1);
        mip_pixels[k].resize(uint64_t(mip_w[k]) * mip_h[k] * 4u);
        boxDownsampleRgba8(mip_pixels[k - 1].data(),
                           mip_w[k - 1], mip_h[k - 1],
                           mip_pixels[k].data());
    }

    // ── Parallel BC7 encode every (mip, page) tile ────────────────
    // Each tile is kVtTileSize² RGBA8 with edge-clamped border, then
    // encoded twice (mip 0 of slot + mip 1 of slot) into 6480 bytes.
    encode_pool_->parallelFor(total_pages,
        [&](size_t entry_idx) {
            // Map entry_idx → (mip k, page px, page py).  Same walk
            // as vtMipOffsetWithinVt produces, just in reverse.
            uint32_t local = uint32_t(entry_idx);
            uint32_t k = 0;
            while (k < mip_count) {
                uint32_t mp = vtMipPagesAt(pages_x, k) * vtMipPagesAt(pages_y, k);
                if (local < mp) break;
                local -= mp;
                ++k;
            }
            if (k >= mip_count) return;
            const uint32_t mpx = vtMipPagesAt(pages_x, k);
            const uint32_t px  = local % mpx;
            const uint32_t py  = local / mpx;
            const uint32_t mw  = mip_w[k];
            const uint32_t mh  = mip_h[k];
            const uint8_t* mip_data = mip_pixels[k].data();

            // Extract kVtTileSize × kVtTileSize RGBA8 region centred
            // on the page (px, py) of mip k.  The page's content
            // covers source pixels [px*PAGE..px*PAGE+PAGE) × [..]; we
            // also include kVtTileBorder pixels on each side, edge-
            // clamped where the source ends.  Source coordinate for
            // tile pixel (x, y) is (px*PAGE - BORDER + x, py*PAGE -
            // BORDER + y), clamped to [0, mw-1] × [0, mh-1].
            std::vector<uint8_t> tile_rgba(kVtTileSize * kVtTileSize * 4u);
            const int32_t origin_x = int32_t(px * kVtPageSize) - int32_t(kVtTileBorder);
            const int32_t origin_y = int32_t(py * kVtPageSize) - int32_t(kVtTileBorder);
            for (uint32_t ty = 0; ty < kVtTileSize; ++ty) {
                int32_t sy_signed = origin_y + int32_t(ty);
                uint32_t sy = uint32_t(std::clamp(sy_signed,
                                                  0, int32_t(mh) - 1));
                for (uint32_t tx = 0; tx < kVtTileSize; ++tx) {
                    int32_t sx_signed = origin_x + int32_t(tx);
                    uint32_t sx = uint32_t(std::clamp(sx_signed,
                                                      0, int32_t(mw) - 1));
                    const uint8_t* s = mip_data + (size_t(sy) * mw + sx) * 4u;
                    uint8_t* d = tile_rgba.data() + (size_t(ty) * kVtTileSize + tx) * 4u;
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                }
            }

            // Encode the bordered tile to BC7 at full resolution
            // (kVtTileSize × kVtTileSize → kBc7BytesMip0 bytes).
            uint8_t* dst0 = cache.bc7_albedo.data()
                          + uint64_t(entry_idx) * kBc7BytesPerEntry;
            encodeBC7Mode6(tile_rgba.data(),
                           kVtTileSize, kVtTileSize, dst0);

            // Half-resolution tile for pool mip 1 — box-downsample
            // the full bordered tile, then encode that.  The mip 1
            // slot is kVtTileSize/2 × kVtTileSize/2 = 36×36 → 1296 B.
            std::vector<uint8_t> tile_rgba_half(
                (kVtTileSize/2) * (kVtTileSize/2) * 4u);
            boxDownsampleRgba8(tile_rgba.data(),
                               kVtTileSize, kVtTileSize,
                               tile_rgba_half.data());
            encodeBC7Mode6(tile_rgba_half.data(),
                           kVtTileSize/2, kVtTileSize/2,
                           dst0 + kBc7BytesMip0);
        });

    // ── NORMAL pipeline: identical structure to ALBEDO above but
    //    encoded as BC5_UNORM (RG only) instead of BC7.  The pool's
    //    mip 0 + mip 1 byte sizes happen to match exactly because
    //    BC5 and BC7 are both 16 B per 4×4 block.  We readback the
    //    normal source via the same blit-decode helper used for
    //    DDS-loaded BC sources, then per-tile slice + encode.
    if (normal_image) {
        // NORMAL: source is BC5_UNORM / BC7_UNORM with linear data.
        // Use the UNORM temp so the blit preserves linear bytes —
        // sRGB-encoding tangent-space normal RG channels would skew
        // the angles and cause exactly the "debug normal vs geometry
        // normal mismatch" you'd see in the render-debug viz.
        std::vector<uint8_t> nrm_pixels =
            readbackImageToRgba8(normal_image, width, height,
                                 /*dst_is_srgb*/ false);
        if (!nrm_pixels.empty()) {
            cache.bc5_normal.assign(uint64_t(total_pages) * kBc7BytesPerEntry, 0);

            // CPU mip pyramid for the normal source.
            std::vector<std::vector<uint8_t>> nm_pixels(mip_count);
            std::vector<uint32_t> nm_w(mip_count), nm_h(mip_count);
            nm_pixels[0] = std::move(nrm_pixels);
            nm_w[0] = width; nm_h[0] = height;
            for (uint32_t k = 1; k < mip_count; ++k) {
                nm_w[k] = std::max(1u, nm_w[k - 1] >> 1);
                nm_h[k] = std::max(1u, nm_h[k - 1] >> 1);
                nm_pixels[k].resize(uint64_t(nm_w[k]) * nm_h[k] * 4u);
                boxDownsampleRgba8(nm_pixels[k - 1].data(),
                                   nm_w[k - 1], nm_h[k - 1],
                                   nm_pixels[k].data());
            }

            // Same parallel per-tile encode loop as ALBEDO but
            // calling encodeBC5UNorm on the gathered RG.
            encode_pool_->parallelFor(total_pages,
                [&](size_t entry_idx) {
                    uint32_t local = uint32_t(entry_idx);
                    uint32_t k = 0;
                    while (k < mip_count) {
                        uint32_t mp = vtMipPagesAt(pages_x, k) * vtMipPagesAt(pages_y, k);
                        if (local < mp) break;
                        local -= mp;
                        ++k;
                    }
                    if (k >= mip_count) return;
                    const uint32_t mpx = vtMipPagesAt(pages_x, k);
                    const uint32_t px  = local % mpx;
                    const uint32_t py  = local / mpx;
                    const uint32_t mw  = nm_w[k];
                    const uint32_t mh  = nm_h[k];
                    const uint8_t* mip_data = nm_pixels[k].data();

                    std::vector<uint8_t> tile_rgba(kVtTileSize * kVtTileSize * 4u);
                    const int32_t origin_x = int32_t(px * kVtPageSize) - int32_t(kVtTileBorder);
                    const int32_t origin_y = int32_t(py * kVtPageSize) - int32_t(kVtTileBorder);
                    for (uint32_t ty = 0; ty < kVtTileSize; ++ty) {
                        int32_t sy_signed = origin_y + int32_t(ty);
                        uint32_t sy = uint32_t(std::clamp(sy_signed,
                                                          0, int32_t(mh) - 1));
                        for (uint32_t tx = 0; tx < kVtTileSize; ++tx) {
                            int32_t sx_signed = origin_x + int32_t(tx);
                            uint32_t sx = uint32_t(std::clamp(sx_signed,
                                                              0, int32_t(mw) - 1));
                            const uint8_t* s = mip_data + (size_t(sy) * mw + sx) * 4u;
                            uint8_t* d = tile_rgba.data() + (size_t(ty) * kVtTileSize + tx) * 4u;
                            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                        }
                    }

                    uint8_t* dst0 = cache.bc5_normal.data()
                                  + uint64_t(entry_idx) * kBc7BytesPerEntry;
                    encodeBC5UNorm(tile_rgba.data(),
                                   kVtTileSize, kVtTileSize, dst0);

                    std::vector<uint8_t> tile_rgba_half(
                        (kVtTileSize/2) * (kVtTileSize/2) * 4u);
                    boxDownsampleRgba8(tile_rgba.data(),
                                       kVtTileSize, kVtTileSize,
                                       tile_rgba_half.data());
                    encodeBC5UNorm(tile_rgba_half.data(),
                                   kVtTileSize/2, kVtTileSize/2,
                                   dst0 + kBc7BytesMip0);
                });
        }
    }
}

// ── Upload one tile (mip, page_x, page_y) of a VT to a pool slot ─────
// Issues into the supplied command buffer.  ALBEDO comes from the
// per-VT BC7 cache via copyBufferToImage; non-ALBEDO blits from the
// stashed source GPU image.  Caller surrounds with TRANSFER_DST
// barrier on each pool image.  Created staging buffers are pushed to
// the keepalive vectors and released after submit completes.
void VirtualTextureManager::uploadTileAllLayers(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    uint32_t slot, uint32_t vt_index,
    uint32_t mip, uint32_t page_x, uint32_t page_y,
    uint32_t staging_slot) {

    if (vt_index >= vt_cache_.size()) return;
    const VtCacheEntry& cache = vt_cache_[vt_index];

    const uint32_t phys_page_x = slot % kVtPagesAcross;
    const uint32_t phys_page_y = slot / kVtPagesAcross;
    const uint32_t mip_offset_local = vtMipOffsetWithinVt(
        cache.pages_x, cache.pages_y, mip);
    const uint32_t local_idx =
        page_y * vtMipPagesAt(cache.pages_x, mip) + page_x;
    const uint32_t entry_idx_in_cache = mip_offset_local + local_idx;

    // ── ALBEDO: BC7 from cache → persistent staging buffer slot →
    //    pool slot (mip 0 + mip 1).  No per-tile vkCreateBuffer; we
    //    memcpy into upload_staging_buffer_'s byte slice for this
    //    upload's staging_slot, then point copyBufferToImage at the
    //    same offset.  The persistent buffer is HOST_COHERENT, so the
    //    write is visible to the GPU as soon as the next submit runs.
    if (entry_idx_in_cache < cache.bc7_albedo.size() / kBc7BytesPerEntry &&
        upload_staging_mapped_ &&
        staging_slot < kStreamerUploadsPerFrame) {
        // Per-slot staging layout: ALBEDO uses subslot 0, NORMAL uses
        // subslot 1.  Both subslots are kBc7BytesPerEntry-sized.
        // The frame's FIF slice base (active_staging_base_off_) is
        // added on so back-to-back frames don't overlap.
        const uint64_t base_off =
            active_staging_base_off_ +
            uint64_t(staging_slot) * 2u * kBc7BytesPerEntry;
        std::memcpy(
            upload_staging_mapped_ + base_off,
            cache.bc7_albedo.data() + uint64_t(entry_idx_in_cache) * kBc7BytesPerEntry,
            kBc7BytesPerEntry);

        std::vector<er::BufferImageCopyInfo> regions(2);
        regions[0].buffer_offset = base_off;
        regions[0].buffer_row_length   = kVtTileSize;
        regions[0].buffer_image_height = kVtTileSize;
        regions[0].image_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        regions[0].image_subresource.mip_level   = 0;
        regions[0].image_subresource.layer_count = 1;
        regions[0].image_offset = glm::ivec3(int32_t(phys_page_x*kVtTileSize),
                                              int32_t(phys_page_y*kVtTileSize), 0);
        regions[0].image_extent = glm::uvec3(kVtTileSize, kVtTileSize, 1);
        regions[1] = regions[0];
        regions[1].buffer_offset = base_off + kBc7BytesMip0;
        regions[1].buffer_row_length   = kVtTileSize / 2;
        regions[1].buffer_image_height = kVtTileSize / 2;
        regions[1].image_subresource.mip_level = 1;
        regions[1].image_offset = glm::ivec3(int32_t(phys_page_x*(kVtTileSize/2)),
                                              int32_t(phys_page_y*(kVtTileSize/2)), 0);
        regions[1].image_extent = glm::uvec3(kVtTileSize/2, kVtTileSize/2, 1);
        cmd_buf->copyBufferToImage(
            upload_staging_buffer_,
            layer_pools_[uint32_t(VtLayer::ALBEDO)].texture.image,
            regions, er::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    // ── NORMAL: BC5 from cache → persistent staging buffer subslot 1
    //    → pool slot.  Same per-tile structure as ALBEDO above; the
    //    NORMAL pool was switched from R8G8B8A8_UNORM (blit-fed) to
    //    BC5_UNORM (CPU-encoded) for a 4× memory saving.
    if (!cache.bc5_normal.empty() &&
        entry_idx_in_cache < cache.bc5_normal.size() / kBc7BytesPerEntry &&
        upload_staging_mapped_ &&
        staging_slot < kStreamerUploadsPerFrame) {
        // FIF base + per-slot subslot 1 (NORMAL).
        const uint64_t base_off =
            active_staging_base_off_ +
            (uint64_t(staging_slot) * 2u + 1u) * kBc7BytesPerEntry;
        std::memcpy(
            upload_staging_mapped_ + base_off,
            cache.bc5_normal.data() + uint64_t(entry_idx_in_cache) * kBc7BytesPerEntry,
            kBc7BytesPerEntry);

        std::vector<er::BufferImageCopyInfo> regions(2);
        regions[0].buffer_offset = base_off;
        regions[0].buffer_row_length   = kVtTileSize;
        regions[0].buffer_image_height = kVtTileSize;
        regions[0].image_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        regions[0].image_subresource.mip_level   = 0;
        regions[0].image_subresource.layer_count = 1;
        regions[0].image_offset = glm::ivec3(int32_t(phys_page_x*kVtTileSize),
                                              int32_t(phys_page_y*kVtTileSize), 0);
        regions[0].image_extent = glm::uvec3(kVtTileSize, kVtTileSize, 1);
        regions[1] = regions[0];
        regions[1].buffer_offset = base_off + kBc7BytesMip0;
        regions[1].buffer_row_length   = kVtTileSize / 2;
        regions[1].buffer_image_height = kVtTileSize / 2;
        regions[1].image_subresource.mip_level = 1;
        regions[1].image_offset = glm::ivec3(int32_t(phys_page_x*(kVtTileSize/2)),
                                              int32_t(phys_page_y*(kVtTileSize/2)), 0);
        regions[1].image_extent = glm::uvec3(kVtTileSize/2, kVtTileSize/2, 1);
        cmd_buf->copyBufferToImage(
            upload_staging_buffer_,
            layer_pools_[uint32_t(VtLayer::NORMAL)].texture.image,
            regions, er::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    // ── Non-ALBEDO/Non-NORMAL: blit (page_size + 2*border) << mip
    // from source into a kVtTileSize×kVtTileSize pool slot.  Source
    // region is
    // CENTRED on the page's content with a kVtTileBorder<<mip border
    // on each side (clamped to the source extent for edge tiles),
    // and the blit's bilinear filter handles the (<<mip) downsample
    // to fit in the slot.  Pool mip 1 gets a half-resolution copy.
    auto blitOne = [&](VtLayer layer, const std::shared_ptr<er::Image>& src) {
        if (!src) return;
        const int32_t src_page_sz   = int32_t(kVtPageSize)   << mip;
        const int32_t src_border    = int32_t(kVtTileBorder) << mip;
        const int32_t src_x_centre0 = int32_t(page_x) * src_page_sz;
        const int32_t src_y_centre0 = int32_t(page_y) * src_page_sz;
        const int32_t src_x0 = std::max(src_x_centre0 - src_border, 0);
        const int32_t src_y0 = std::max(src_y_centre0 - src_border, 0);
        const int32_t src_x1 = std::min(src_x_centre0 + src_page_sz + src_border,
                                        int32_t(cache.width));
        const int32_t src_y1 = std::min(src_y_centre0 + src_page_sz + src_border,
                                        int32_t(cache.height));
        std::vector<er::ImageBlitInfo> regs(2);
        regs[0].src_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        regs[0].src_subresource.mip_level   = 0;
        regs[0].src_subresource.layer_count = 1;
        regs[0].src_offsets[0] = glm::ivec3(src_x0, src_y0, 0);
        regs[0].src_offsets[1] = glm::ivec3(src_x1, src_y1, 1);
        regs[0].dst_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        regs[0].dst_subresource.mip_level   = 0;
        regs[0].dst_subresource.layer_count = 1;
        regs[0].dst_offsets[0] = glm::ivec3(int32_t(phys_page_x*kVtTileSize),
                                              int32_t(phys_page_y*kVtTileSize), 0);
        regs[0].dst_offsets[1] = glm::ivec3(int32_t((phys_page_x+1)*kVtTileSize),
                                              int32_t((phys_page_y+1)*kVtTileSize), 1);
        regs[1] = regs[0];
        regs[1].dst_subresource.mip_level = 1;
        regs[1].dst_offsets[0] = glm::ivec3(int32_t(phys_page_x*(kVtTileSize/2)),
                                              int32_t(phys_page_y*(kVtTileSize/2)), 0);
        regs[1].dst_offsets[1] = glm::ivec3(int32_t((phys_page_x+1)*(kVtTileSize/2)),
                                              int32_t((phys_page_y+1)*(kVtTileSize/2)), 1);
        cmd_buf->blitImage(src, er::ImageLayout::TRANSFER_SRC_OPTIMAL,
                           layer_pools_[uint32_t(layer)].texture.image,
                           er::ImageLayout::TRANSFER_DST_OPTIMAL,
                           regs, er::Filter::LINEAR);
    };
    blitOne(VtLayer::METAL_ROUGH_AO, cache.mr_ao_src);
    blitOne(VtLayer::EMISSIVE,       cache.emissive_src);
}

// ── Public: register a material (Phase B: metadata-only + pin) ───────
// Allocates the page-table window, builds the per-VT BC7 cache and
// stashes source images, then PINS the smallest mip (1×1 page = 1
// pool slot per VT) so the shader's mip-walk fallback always finds
// at least the most-blurred version of every registered material.
// All other (mip, page) entries start UNRESIDENT and stream in via
// tick() based on shader feedback.
VirtualTextureId VirtualTextureManager::registerMaterial(
    const uint8_t* albedo_pixels,
    const std::shared_ptr<er::Image>& albedo_image,
    const std::shared_ptr<er::Image>& normal_image,
    const std::shared_ptr<er::Image>& mr_ao_image,
    const std::shared_ptr<er::Image>& emissive_image,
    uint32_t width,
    uint32_t height) {

    // Either CPU pixels OR a GPU image must be available — we use
    // the pixels directly when present, fall back to GPU decode-blit
    // + readback when only the image is available (DDS/BC sources).
    if ((!albedo_pixels && !albedo_image) || width == 0 || height == 0) {
        return kInvalidVtId;
    }
    if (meta_cpu_.size() >= kMaxVirtualTextures) return kInvalidVtId;

    const uint32_t pages_x   = (width  + kVtPageSize - 1) / kVtPageSize;
    const uint32_t pages_y   = (height + kVtPageSize - 1) / kVtPageSize;
    const uint32_t mip_count = vtComputeMipCount(pages_x, pages_y);
    const uint32_t total_pages =
        vtTotalPagesAllMips(pages_x, pages_y, mip_count);

    if (next_page_table_tail_ + total_pages > kMaxPageTableEntries) {
        return kInvalidVtId;
    }
    const uint32_t table_offset = next_page_table_tail_;
    next_page_table_tail_ += total_pages;

    // All entries start UNRESIDENT; tick() will fill them on demand.
    for (uint32_t i = 0; i < total_pages; ++i) {
        page_table_cpu_[table_offset + i] = kPageEntryUnresident;
    }
    if (page_table_mapped_) {
        std::memcpy(page_table_mapped_ + table_offset,
                    page_table_cpu_.data() + table_offset,
                    uint64_t(total_pages) * sizeof(uint32_t));
    }

    // ── Append meta entry ──────────────────────────────────────────
    VirtualTextureMeta meta{};
    meta.width_px          = width;
    meta.height_px         = height;
    meta.pages_x           = pages_x;
    meta.pages_y           = pages_y;
    meta.page_table_offset = table_offset;
    meta.mip_count         = mip_count;
    const uint32_t vt_index = static_cast<uint32_t>(meta_cpu_.size());
    meta_cpu_.push_back(meta);
    if (meta_mapped_) {
        meta_mapped_[vt_index] = meta;
    }

    // ── Build CPU BC7 cache + stash source images ─────────────────
    encodeAndCacheVt(vt_index, albedo_pixels, albedo_image,
                     normal_image, mr_ao_image, emissive_image,
                     width, height,
                     pages_x, pages_y, mip_count, table_offset);

    // ── Pin the smallest mip (always-resident fallback) ───────────
    // Smallest mip is mip_count - 1 with vtMipPagesAt = 1×1 = 1 page.
    // Allocate one slot, upload via uploadTileAllLayers, mark
    // RESIDENT + pinned.  Issued through a transient command buffer.
    {
        const uint32_t k = mip_count - 1u;
        const uint32_t mip_off = vtMipOffsetWithinVt(pages_x, pages_y, k);
        const uint32_t mip_px  = vtMipPagesAt(pages_x, k);
        const uint32_t mip_py  = vtMipPagesAt(pages_y, k);

        auto cmd = device_->setupTransientCommandBuffer();
        // Transition all four pool images and any non-null source
        // images for the upcoming copy + blit.
        er::ImageResourceInfo from_read{
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };
        er::ImageResourceInfo to_xfer_dst{
            er::ImageLayout::TRANSFER_DST_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        er::ImageResourceInfo to_xfer_src{
            er::ImageLayout::TRANSFER_SRC_OPTIMAL,
            SET_FLAG_BIT(Access, TRANSFER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
        for (uint32_t l = 0; l < uint32_t(VtLayer::COUNT); ++l) {
            cmd->addImageBarrier(layer_pools_[l].texture.image,
                from_read, to_xfer_dst, 0, kVtPoolMipLevels, 0, 1);
        }
        // ALBEDO + NORMAL upload via copyBufferToImage from per-VT
        // BC7/BC5 caches — no source-image transition required for
        // either.  Only MR_AO + EMISSIVE still go through the GPU
        // blit path and need TRANSFER_SRC_OPTIMAL.
        if (mr_ao_image)    cmd->addImageBarrier(mr_ao_image,
            from_read, to_xfer_src, 0, 1, 0, 1);
        if (emissive_image) cmd->addImageBarrier(emissive_image,
            from_read, to_xfer_src, 0, 1, 0, 1);

        // Pin block: at most kStreamerUploadsPerFrame tiles get
        // pinned (we cap mip_px*mip_py × number of materials to
        // something well below this).  Reuse the persistent staging
        // buffer's first N slots — same submit-and-wait pattern as
        // tick(), so the buffer is safe to reuse on the next call.
        uint32_t pinned_count = 0;
        uint32_t pin_staging_slot = 0;
        for (uint32_t py = 0; py < mip_py; ++py) {
            for (uint32_t px = 0; px < mip_px; ++px) {
                const uint32_t entry_local = mip_off + py * mip_px + px;
                const uint32_t entry_idx   = table_offset + entry_local;
                uint32_t s = allocSlot();
                if (s >= kVtPagesPerLayer) continue;
                uploadTileAllLayers(cmd, s, vt_index, k, px, py,
                                    pin_staging_slot);
                ++pin_staging_slot;
                markSlotResident(s,
                    makeTileKey(vt_index, k, px, py),
                    entry_idx, /*pinned*/ true);
                ++pinned_count;
            }
        }

        for (uint32_t l = 0; l < uint32_t(VtLayer::COUNT); ++l) {
            cmd->addImageBarrier(layer_pools_[l].texture.image,
                to_xfer_dst, from_read, 0, kVtPoolMipLevels, 0, 1);
        }
        if (mr_ao_image)    cmd->addImageBarrier(mr_ao_image,
            to_xfer_src, from_read, 0, 1, 0, 1);
        if (emissive_image) cmd->addImageBarrier(emissive_image,
            to_xfer_src, from_read, 0, 1, 0, 1);
        device_->submitAndWaitTransientCommandBuffer();
        // Persistent staging buffer — no per-call destroy.

        // Redirected to vt_pool.log — see vtLog() helper above for
        // why and how.  Same line format as the original stdout
        // printf, so any user tail/grep just needs to point at the
        // log file instead of the console.
        {
            auto& log = vtLog();
            if (log.is_open()) {
                char line[256];
                std::snprintf(line, sizeof(line),
                    "[RVT] registerMaterial vt=%u %ux%u mip0=%ux%u mips=%u "
                    "pinned=%u/%u total_pages=%u free_slots=%zu\n",
                    vt_index, width, height, pages_x, pages_y, mip_count,
                    pinned_count, mip_px * mip_py, total_pages,
                    free_slots_.size());
                log << line;
                log.flush();
            }
        }
    }

    // Synchronously publish the page-table entry updates queued by
    // the pinned-mip allocations above (markSlotResident appended
    // each into page_table_dirty_entries_ but did NOT touch the
    // host-mapped buffer — that path is reserved for tick(), which
    // applies updates via the frame's main cmd buffer to avoid the
    // race with in-flight previous-frame shader reads).  Since
    // registerMaterial is synchronous (no concurrent rendering of
    // these new pages yet), we flush via a transient submit-and-
    // wait so by the time this returns the GPU buffer is consistent
    // with what cluster_bindless.frag will read on the very next
    // frame — otherwise the first frame after a material load
    // would see all the new pages as UNRESIDENT, the mip-walk would
    // find nothing resident anywhere (pinned slot's page-table
    // entry hadn't been published yet), and the fragment shader
    // would return its diagnostic magenta — visible as a full-
    // screen colored flicker as the streamer warms up.
    flushDirtyPageTableEntriesViaTransient();

    return makeVtId(VtLayer::ALBEDO, vt_index);
}

// ── Old per-layer registerTextureFromImage was here ───────────────
// Replaced by registerMaterial above.  The cluster_renderer loader
// no longer issues two separate registrations (one for albedo, one
// for normal) — both go through registerMaterial which uses the
// shared slot allocator and the uploadLayerByBlitToSlots helper.
// The deleted body's mip-0-grid + reserve + per-page allocation
// logic is now inside registerAlbedoBC7 (since that's where the
// shared allocator runs); the deleted body's blit-region build +
// transient-command-buffer logic is in uploadLayerByBlitToSlots.
#if 0
VirtualTextureId VirtualTextureManager::registerTextureFromImage_REMOVED(
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

    // ── Mip-0 grid + full mip chain page count ────────────────────
    const uint32_t pages_x   = (width  + kVtPageSize - 1) / kVtPageSize;
    const uint32_t pages_y   = (height + kVtPageSize - 1) / kVtPageSize;
    const uint32_t mip_count = vtComputeMipCount(pages_x, pages_y);
    const uint32_t total_pages =
        vtTotalPagesAllMips(pages_x, pages_y, mip_count);

    if (next_page_table_tail_ + total_pages > kMaxPageTableEntries) {
        return kInvalidVtId;
    }
    const uint32_t table_offset = next_page_table_tail_;
    next_page_table_tail_ += total_pages;

    // ── Walk all mip levels, allocate slots, build blit regions ───
    // For each (mip, page), the source region spans (page_size << mip)
    // pixels of the original image — vkCmdBlitImage with a LINEAR
    // filter does the downsample on the GPU, so we don't need to
    // round-trip through the CPU.  Source format (RGBA8) and pool
    // format (RGBA8) are blit-compatible.  This stays GPU-resident
    // and is much faster than a CPU readback path.
    //
    // Note: BC7-format pool images do NOT support BLIT_DST, which is
    // why ALBEDO has its own registerAlbedoBC7 path that goes through
    // CPU encode.  Here we have RGBA8 pool layers (NORMAL / MR_AO /
    // EMISSIVE) which DO support blit-as-dst per the Vulkan spec.
    std::vector<er::ImageBlitInfo> regions;
    regions.reserve(total_pages);

    uint32_t entry_local = 0;        // 0..total_pages-1 (local to this VT)
    for (uint32_t k = 0; k < mip_count; ++k) {
        const uint32_t mip_px      = vtMipPagesAt(pages_x, k);
        const uint32_t mip_py      = vtMipPagesAt(pages_y, k);
        const uint32_t src_page_sz = kVtPageSize << k;  // src texels per page at mip k
        for (uint32_t py = 0; py < mip_py; ++py) {
            for (uint32_t px = 0; px < mip_px; ++px) {
                const uint32_t entry_index = table_offset + entry_local;
                const uint32_t slot = allocatePageSlot(layer);
                if (slot >= kVtPagesPerLayer) {
                    page_table_cpu_[entry_index] = kPageEntryUnresident;
                    ++entry_local;
                    continue;
                }
                const uint32_t phys_page_x = slot % kVtPagesAcross;
                const uint32_t phys_page_y = slot / kVtPagesAcross;

                // Source region in mip-0 pixel coordinates,
                // clamped so the right/bottom edge pages don't
                // read past the image.  vkCmdBlitImage takes a
                // box in src_offsets[0]..src_offsets[1] and
                // downsamples that into the dst box; for non-
                // power-of-2 source dims the right column /
                // bottom row of the deepest mips just collapse a
                // partial source box, which is fine.
                const uint32_t src_x0 = px * src_page_sz;
                const uint32_t src_y0 = py * src_page_sz;
                const uint32_t src_x1 = std::min(src_x0 + src_page_sz, width);
                const uint32_t src_y1 = std::min(src_y0 + src_page_sz, height);

                // Mip 0 region — full 64×64 page.  Linear-filter
                // downsamples (64<<k) source texels into 64 dest.
                er::ImageBlitInfo r0{};
                r0.src_subresource.aspect_mask      = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
                r0.src_subresource.mip_level        = 0;
                r0.src_subresource.base_array_layer = 0;
                r0.src_subresource.layer_count      = 1;
                r0.src_offsets[0] = glm::ivec3(int32_t(src_x0), int32_t(src_y0), 0);
                r0.src_offsets[1] = glm::ivec3(int32_t(src_x1), int32_t(src_y1), 1);
                r0.dst_subresource.aspect_mask      = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
                r0.dst_subresource.mip_level        = 0;
                r0.dst_subresource.base_array_layer = 0;
                r0.dst_subresource.layer_count      = 1;
                r0.dst_offsets[0] = glm::ivec3(
                    int32_t(phys_page_x * kVtPageSize),
                    int32_t(phys_page_y * kVtPageSize), 0);
                r0.dst_offsets[1] = glm::ivec3(
                    int32_t((phys_page_x + 1) * kVtPageSize),
                    int32_t((phys_page_y + 1) * kVtPageSize), 1);
                regions.push_back(r0);

                // Mip 1 region — same source box, half-res 32×32
                // dest at the slot's mip-1 location.  Pool mip 1 is
                // a 2048×2048 image so phys coords halve.  The
                // linear filter does a 2× wider downsample than
                // mip 0 (effectively giving the slot a 1-extra-
                // mip preview the trilinear lerp uses to bridge
                // VT-mip transitions).
                er::ImageBlitInfo r1 = r0;
                r1.dst_subresource.mip_level = 1;
                r1.dst_offsets[0] = glm::ivec3(
                    int32_t(phys_page_x * (kVtPageSize / 2)),
                    int32_t(phys_page_y * (kVtPageSize / 2)), 0);
                r1.dst_offsets[1] = glm::ivec3(
                    int32_t((phys_page_x + 1) * (kVtPageSize / 2)),
                    int32_t((phys_page_y + 1) * (kVtPageSize / 2)), 1);
                regions.push_back(r1);

                page_table_cpu_[entry_index] =
                    packPageEntry(phys_page_x, phys_page_y, /*resident*/ true);
                ++entry_local;
            }
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

    // ── Submit one transient cmd buffer doing all the blits ───────
    // RGBA8 source → RGBA8 dest, Filter::LINEAR for in-driver
    // bilinear downsample at higher mip levels.  Mip 0 entries blit
    // 1:1 (src and dst extents both = page_size); mip k blits a
    // (page_size << k)² source box into a page_size² dst box, with
    // the linear filter doing 2^k×2^k area averaging.
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
        // Both pool mip levels participate in the blit (regions
        // include dst_mip 0 and dst_mip 1), so the barrier needs to
        // cover the full mip range — not just mip 0.
        cmd_buf->addImageBarrier(
            layer_pools_[uint32_t(layer)].texture.image,
            from_read, to_xfer_dst, 0, kVtPoolMipLevels, 0, 1);

        cmd_buf->blitImage(
            src_image, er::ImageLayout::TRANSFER_SRC_OPTIMAL,
            layer_pools_[uint32_t(layer)].texture.image,
            er::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions, er::Filter::LINEAR);

        cmd_buf->addImageBarrier(src_image,
            to_xfer_src, from_read, 0, 1, 0, 1);
        cmd_buf->addImageBarrier(
            layer_pools_[uint32_t(layer)].texture.image,
            to_xfer_dst, from_read, 0, kVtPoolMipLevels, 0, 1);

        device_->submitAndWaitTransientCommandBuffer();
    }

    // Append meta entry.
    VirtualTextureMeta meta{};
    meta.width_px          = width;
    meta.height_px         = height;
    meta.pages_x           = pages_x;       // mip 0 grid; higher mips
    meta.pages_y           = pages_y;       // computed by vtMipPagesXY
    meta.page_table_offset = table_offset;
    meta.mip_count         = mip_count;
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
#endif  // 0 — registerTextureFromImage_REMOVED reference body

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

    // ── Mip-0 page grid + full-mip-chain page count ────────────────
    // Mip 0 has ceil(W/64) × ceil(H/64) pages.  Higher mips are
    // ceil-halved each level; mip_count stops at 1×1 (or VT_MAX_MIPS
    // whichever first).  total_pages_all_mips counts every page-
    // table entry the VT consumes — these are claimed contiguously
    // from page_table_cpu_, mip 0 first, then mip 1, etc.
    const uint32_t pages_x = (width  + kVtPageSize - 1) / kVtPageSize;
    const uint32_t pages_y = (height + kVtPageSize - 1) / kVtPageSize;
    const uint32_t mip_count = vtComputeMipCount(pages_x, pages_y);
    const uint32_t total_pages =
        vtTotalPagesAllMips(pages_x, pages_y, mip_count);

    if (next_page_table_tail_ + total_pages > kMaxPageTableEntries) {
        std::printf("[RVT] registerAlbedoBC7 BAILED: page-table full "
                    "(%u + %u > %u)\n",
                    next_page_table_tail_, total_pages, kMaxPageTableEntries);
        return kInvalidVtId;
    }
    const uint32_t table_offset = next_page_table_tail_;
    next_page_table_tail_ += total_pages;

    // ── Allocate page slots + mark all entries UNRESIDENT ──────────
    // The work struct's per-entry arrays are sized for
    // `total_pages_all_mips`, indexed in the same order as the
    // page-table region: mip 0 first, then mip 1, etc.  We track
    // which mip each entry belongs to so processPendingWork knows
    // which downsampled mip image to encode for that entry.
    PendingWork work{};
    work.layer        = VtLayer::ALBEDO;
    work.src_image    = src_image;
    work.width        = width;
    work.height       = height;
    work.table_offset = table_offset;
    work.pages_x      = pages_x;
    work.pages_y      = pages_y;
    work.mip_count    = mip_count;
    work.phys_x.resize(total_pages, 0);
    work.phys_y.resize(total_pages, 0);
    work.mip.resize(total_pages, 0);
    work.ok.resize(total_pages, 0);

    uint32_t allocated   = 0;
    uint32_t entry_local = 0;        // 0..total_pages-1 (local to this VT)
    for (uint32_t k = 0; k < mip_count; ++k) {
        const uint32_t mip_px = vtMipPagesAt(pages_x, k);
        const uint32_t mip_py = vtMipPagesAt(pages_y, k);
        for (uint32_t py = 0; py < mip_py; ++py) {
            for (uint32_t px = 0; px < mip_px; ++px) {
                const uint32_t entry_index = table_offset + entry_local;
                // NOTE: registerAlbedoBC7 is now ORPHAN code — Phase B
                // routes everything through registerMaterial, which uses
                // encodeAndCacheVt + uploadTileAllLayers for streaming.
                // We keep this function compilable (replaced the deleted
                // allocatePageSlot with allocSlot) so the BC7 encode +
                // upload reference logic is still here for diff context.
                const uint32_t slot = allocSlot();
                work.mip[entry_local] = uint8_t(k);
                if (slot >= kVtPagesPerLayer) {
                    page_table_cpu_[entry_index] = kPageEntryUnresident;
                    ++entry_local;
                    continue;
                }
                work.phys_x[entry_local] = slot % kVtPagesAcross;
                work.phys_y[entry_local] = slot / kVtPagesAcross;
                work.ok[entry_local]     = 1;
                ++allocated;
                // UNRESIDENT for now — flipped to RESIDENT by
                // processPendingWork once the BC7 data is uploaded.
                page_table_cpu_[entry_index] = kPageEntryUnresident;
                ++entry_local;
            }
        }
    }
    std::printf("[RVT] registerAlbedoBC7(orphan) %ux%u mip0=%ux%u mips=%u "
                "tbl_off=%u allocated=%u/%u free=%zu\n",
                width, height, pages_x, pages_y, mip_count, table_offset,
                allocated, total_pages, free_slots_.size());

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
    //    even before processPendingWork has populated the pages.
    //    The shader resolves through meta to find page_table_offset
    //    + per-mip offsets; entries themselves are UNRESIDENT until
    //    the upload completes.
    VirtualTextureMeta meta{};
    meta.width_px          = width;
    meta.height_px         = height;
    meta.pages_x           = pages_x;
    meta.pages_y           = pages_y;
    meta.page_table_offset = table_offset;
    meta.mip_count         = mip_count;
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

// Heavy-lifting half of registerAlbedoBC7, runs on caller thread
// (worker is currently disabled — see constructor):
// readback → CPU mip-chain build → parallel BC7 encode every mip's
// pages → upload → flip residency bits.
void VirtualTextureManager::processPendingWork(PendingWork&& w) {
    if (w.layer != VtLayer::ALBEDO) return;  // only ALBEDO supported
    // total_pages now spans the full mip chain — registerAlbedoBC7
    // sized phys_x / phys_y / mip / ok to vtTotalPagesAllMips(...).
    const uint32_t total_pages  = uint32_t(w.phys_x.size());
    const uint32_t mip_count    = w.mip_count;

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

    // ── Step 2: build the CPU mip chain ────────────────────────────
    // Box-filter cpu_pixels (mip 0) down through mip_count-1 and
    // store each mip's RGBA8 pixels.  Linear-space averaging — see
    // the boxDownsampleRgba8 helper note for the gamma caveat.
    // mip_pixels[k] holds mip k's W*H*4 bytes, where W/H are the
    // mip's full source dimensions (not the page-aligned size).
    std::vector<std::vector<uint8_t>> mip_pixels(mip_count);
    std::vector<uint32_t> mip_widths (mip_count);
    std::vector<uint32_t> mip_heights(mip_count);
    mip_pixels[0]  = std::move(cpu_pixels);
    mip_widths[0]  = width;
    mip_heights[0] = height;
    for (uint32_t k = 1; k < mip_count; ++k) {
        mip_widths[k]  = std::max(1u, mip_widths [k - 1] >> 1);
        mip_heights[k] = std::max(1u, mip_heights[k - 1] >> 1);
        mip_pixels[k].resize(uint64_t(mip_widths[k]) * mip_heights[k] * 4u);
        boxDownsampleRgba8(
            mip_pixels[k - 1].data(),
            mip_widths [k - 1], mip_heights[k - 1],
            mip_pixels[k].data());
    }

    // ── Step 3: parallel BC7 encode (every page of every mip × pool mips)
    // PendingWork's per-entry arrays are sized to total_pages =
    // vtTotalPagesAllMips, ordered mip 0 first then mip 1, etc.  For
    // each entry we know which VT mip it belongs to (w.mip[i]) and
    // therefore which downsampled image to extract its 64×64 page
    // from.  Per entry we now produce TWO BC7 blobs — one for pool
    // mip 0 (64×64) and one for pool mip 1 (32×32 half-res of the
    // page) — and lay them out contiguously in the staging buffer:
    //   [entry_idx * (kBc7BytesMip0 + kBc7BytesMip1)
    //                                + 0]                ← pool mip 0
    //   [entry_idx * (...) + kBc7BytesMip0]              ← pool mip 1
    // The upload step below will emit two copyBufferToImage regions
    // per entry pointing at those two offsets.
    const uint32_t kBc7BytesMip0 = (kVtPageSize / 4) * (kVtPageSize / 4) * 16;            // 4096
    const uint32_t kBc7BytesMip1 = ((kVtPageSize / 2) / 4) * ((kVtPageSize / 2) / 4) * 16; // 1024
    const uint32_t kBc7BytesPerEntry = kBc7BytesMip0 + kBc7BytesMip1;                      // 5120
    std::vector<uint8_t> staging_cpu(uint64_t(total_pages) * kBc7BytesPerEntry);

    // Capture local copies of pages_x/y for the parallelFor body.
    const uint32_t lambda_pages_x_0 = w.pages_x;
    const uint32_t lambda_pages_y_0 = w.pages_y;
    encode_pool_->parallelFor(total_pages,
        [&](size_t entry_idx) {
            if (!w.ok[entry_idx]) return;
            const uint32_t k       = w.mip[entry_idx];
            const uint32_t mip_w   = mip_widths [k];
            const uint32_t mip_h   = mip_heights[k];
            const uint32_t mip_px  = vtMipPagesAt(lambda_pages_x_0, k);
            const uint32_t mip_off = vtMipOffsetWithinVt(
                lambda_pages_x_0, lambda_pages_y_0, k);
            const uint32_t local   = uint32_t(entry_idx) - mip_off;
            const uint32_t px      = local % mip_px;
            const uint32_t py      = local / mip_px;

            // Per-thread scratch — full-res 64×64 RGBA8 page extracted
            // from the appropriate downsampled VT-mip image.
            std::vector<uint8_t> page_rgba(kVtPageSize * kVtPageSize * 4u);
            const uint8_t* mip_data = mip_pixels[k].data();
            for (uint32_t y = 0; y < kVtPageSize; ++y) {
                uint32_t sy = std::min(py * kVtPageSize + y, mip_h - 1u);
                for (uint32_t x = 0; x < kVtPageSize; ++x) {
                    uint32_t sx = std::min(px * kVtPageSize + x, mip_w - 1u);
                    const uint8_t* s = mip_data + (size_t(sy) * mip_w + sx) * 4;
                    uint8_t*       d = page_rgba.data() + (size_t(y) * kVtPageSize + x) * 4;
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                }
            }

            // BC7 encode the full-res page into pool-mip-0's slot.
            uint8_t* dst0 = staging_cpu.data()
                          + uint64_t(entry_idx) * kBc7BytesPerEntry;
            encodeBC7Mode6(page_rgba.data(),
                           kVtPageSize, kVtPageSize, dst0);

            // Box-downsample the page to 32×32 and BC7-encode that
            // into pool-mip-1's slot.  This gives the GPU's trilinear
            // sampler a half-res tap inside the same VT-mip slot,
            // which is what hides the snap when the LOD crosses an
            // integer VT-mip boundary.
            std::vector<uint8_t> page_rgba_half(
                (kVtPageSize / 2) * (kVtPageSize / 2) * 4u);
            boxDownsampleRgba8(
                page_rgba.data(), kVtPageSize, kVtPageSize,
                page_rgba_half.data());
            uint8_t* dst1 = dst0 + kBc7BytesMip0;
            encodeBC7Mode6(page_rgba_half.data(),
                           kVtPageSize / 2, kVtPageSize / 2, dst1);
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
    regions.reserve(uint64_t(total_pages) * 2u);    // mip 0 + mip 1 per entry
    for (uint32_t i = 0; i < total_pages; ++i) {
        if (!w.ok[i]) continue;

        // Pool mip 0 — full 64×64 page at (phys_x*64, phys_y*64).
        er::BufferImageCopyInfo r0{};
        r0.buffer_offset = uint64_t(i) * kBc7BytesPerEntry;
        // BC7: buffer_row_length / image_height are in TEXELS, not
        // blocks.  The driver does the conversion based on format.
        r0.buffer_row_length   = kVtPageSize;
        r0.buffer_image_height = kVtPageSize;
        r0.image_subresource.aspect_mask      = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        r0.image_subresource.mip_level        = 0;
        r0.image_subresource.base_array_layer = 0;
        r0.image_subresource.layer_count      = 1;
        r0.image_offset = glm::ivec3(
            int32_t(w.phys_x[i] * kVtPageSize),
            int32_t(w.phys_y[i] * kVtPageSize), 0);
        r0.image_extent = glm::uvec3(kVtPageSize, kVtPageSize, 1);
        regions.push_back(r0);

        // Pool mip 1 — half-res 32×32 at (phys_x*32, phys_y*32) of
        // pool mip 1 (which is a 2048×2048 image).  Buffer offset
        // points just past the entry's mip-0 BC7 blob.
        er::BufferImageCopyInfo r1 = r0;
        r1.buffer_offset            = uint64_t(i) * kBc7BytesPerEntry + kBc7BytesMip0;
        r1.buffer_row_length        = kVtPageSize / 2;
        r1.buffer_image_height      = kVtPageSize / 2;
        r1.image_subresource.mip_level = 1;
        r1.image_offset = glm::ivec3(
            int32_t(w.phys_x[i] * (kVtPageSize / 2)),
            int32_t(w.phys_y[i] * (kVtPageSize / 2)), 0);
        r1.image_extent = glm::uvec3(kVtPageSize / 2, kVtPageSize / 2, 1);
        regions.push_back(r1);
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
        // Both pool mip levels participate in the upload — barrier
        // covers the full 0..kVtPoolMipLevels range.
        cmd->addImageBarrier(
            layer_pools_[uint32_t(VtLayer::ALBEDO)].texture.image,
            from_read, to_xfer, 0, kVtPoolMipLevels, 0, 1);
        cmd->copyBufferToImage(
            staging_buf,
            layer_pools_[uint32_t(VtLayer::ALBEDO)].texture.image,
            regions,
            er::ImageLayout::TRANSFER_DST_OPTIMAL);
        cmd->addImageBarrier(
            layer_pools_[uint32_t(VtLayer::ALBEDO)].texture.image,
            to_xfer, from_read, 0, kVtPoolMipLevels, 0, 1);
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
        std::printf("[RVT] processPendingWork mip0=%ux%u mips=%u "
                    "tbl_off=%u flipped=%u/%u first_entry=0x%08x map_ok=%d\n",
                    w.pages_x, w.pages_y, mip_count, table_offset,
                    flipped, total_pages, first, map_ok ? 1 : 0);
    }
}

}  // namespace scene_rendering
}  // namespace engine
