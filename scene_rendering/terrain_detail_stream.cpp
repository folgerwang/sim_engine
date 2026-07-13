#include "terrain_detail_stream.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "game_object/mesh_load_task_manager.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

#include "stb_image.h"

namespace fs = std::filesystem;
namespace er = engine::renderer;

namespace engine {
namespace scene_rendering {

namespace {

constexpr uint32_t kTileRes   = kDetailTileRes;          // 2049
constexpr uint32_t kSlots     = kDetailCacheSlots;       // 9 (3x3)
constexpr int      kTilesSide = kDetailTilesPerSide;     // 16

double nowSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::string tilePngPath(const std::string& dir, int tx, int ty) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "tile_%02d_%02d.png", tx, ty);
    return (fs::path(dir) / buf).string();
}

std::string reqPath(const std::string& dir, int tx, int ty) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "req_%02d_%02d", tx, ty);
    return (fs::path(dir) / buf).string();
}

}  // namespace

TerrainDetailStream::TerrainDetailStream(
    const std::shared_ptr<renderer::Device>& device) {
    // R16 array, one 2049^2 layer per cache slot (~8.4 MB each).
    height_array_image_ = device->createImage(
        er::ImageType::TYPE_2D,
        glm::uvec3(kTileRes, kTileRes, 1),
        er::Format::R16_UNORM,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, TRANSFER_DST_BIT),
        er::ImageTiling::OPTIMAL,
        er::ImageLayout::UNDEFINED,
        std::source_location::current(),
        /*flags*/ 0,
        /*sharing*/ false,
        /*num_samples*/ 1,
        /*num_mips*/ 1,
        /*num_layers*/ kSlots);
    // Bind memory the same way Helper does for 2D textures.
    auto mem_requirements = device->getImageMemoryRequirements(
        height_array_image_);
    height_array_memory_ = device->allocateMemory(
        mem_requirements.size,
        mem_requirements.memory_type_bits,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        /*allocate_flags*/ 0);
    device->bindImageMemory(height_array_image_, height_array_memory_);
    height_array_view_ = device->createImageView(
        height_array_image_,
        er::ImageViewType::VIEW_2D_ARRAY,
        er::Format::R16_UNORM,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        std::source_location::current(),
        /*base_mip*/ 0, /*mip_count*/ 1,
        /*base_layer*/ 0, /*layer_count*/ kSlots);
    // All layers to SHADER_READ_ONLY so sampling is defined before any
    // upload (slot table is all -1 anyway).
    er::Helper::transitionImageLayout(
        device,
        height_array_image_,
        er::Format::R16_UNORM,
        er::ImageLayout::UNDEFINED,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        0, 1, 0, kSlots);

    // 1 m albedo tile array (RGBA8, 2048^2 per slot — see the worker's
    // generate_color_tile; ~16 MB per slot, 9 slots).
    const uint32_t kColorRes = 2048;
    color_array_image_ = device->createImage(
        er::ImageType::TYPE_2D,
        glm::uvec3(kColorRes, kColorRes, 1),
        er::Format::R8G8B8A8_UNORM,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, TRANSFER_DST_BIT),
        er::ImageTiling::OPTIMAL,
        er::ImageLayout::UNDEFINED,
        std::source_location::current(),
        /*flags*/ 0,
        /*sharing*/ false,
        /*num_samples*/ 1,
        /*num_mips*/ 1,
        /*num_layers*/ kSlots);
    auto color_reqs = device->getImageMemoryRequirements(color_array_image_);
    color_array_memory_ = device->allocateMemory(
        color_reqs.size,
        color_reqs.memory_type_bits,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        /*allocate_flags*/ 0);
    device->bindImageMemory(color_array_image_, color_array_memory_);
    color_array_view_ = device->createImageView(
        color_array_image_,
        er::ImageViewType::VIEW_2D_ARRAY,
        er::Format::R8G8B8A8_UNORM,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        std::source_location::current(),
        0, 1, 0, kSlots);
    er::Helper::transitionImageLayout(
        device,
        color_array_image_,
        er::Format::R8G8B8A8_UNORM,
        er::ImageLayout::UNDEFINED,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        0, 1, 0, kSlots);

    // Slot table SSBO (host-visible: tiny, updated rarely).
    er::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        table_buffer_,
        table_memory_,
        std::source_location::current(),
        sizeof(TableCpu));
    for (auto& s : table_cpu_.slot_map) s = -1;
    for (auto& s : table_cpu_.color_slot) s = -1;
    uploadTable(device);
}

void TerrainDetailStream::uploadTable(
    const std::shared_ptr<renderer::Device>& device) {
    device->updateBufferMemory(
        table_memory_, sizeof(TableCpu), &table_cpu_);
}

void TerrainDetailStream::setSource(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& tiles_dir,
    const std::string& worker_spawn_cmd) {
    tiles_dir_ = tiles_dir;
    worker_cmd_ = worker_spawn_cmd;
    requested_.clear();
    for (auto& s : slots_) s = Slot{};
    for (auto& s : table_cpu_.slot_map) s = -1;
    for (auto& s : table_cpu_.color_slot) s = -1;
    uploadTable(device);
    if (!tiles_dir_.empty()) {
        std::error_code ec;
        fs::create_directories(tiles_dir_, ec);
        if (!worker_cmd_.empty()) {
            std::cout << "[terrain-detail] spawning worker: "
                      << worker_cmd_ << std::endl;
            std::system(worker_cmd_.c_str());
            last_worker_check_ = nowSeconds();
        }
    }
}

int TerrainDetailStream::findFreeSlot() const {
    for (int i = 0; i < (int)kSlots; ++i) {
        if (slots_[i].tile_x < 0 && !slots_[i].loading) return i;
    }
    return -1;
}

void TerrainDetailStream::requestTile(int tx, int ty) {
    const int key = ty * kTilesSide + tx;
    if (requested_.count(key)) return;
    std::error_code ec;
    const std::string rp = reqPath(tiles_dir_, tx, ty);
    if (!fs::exists(rp, ec)) {
        std::ofstream f(rp, std::ios::binary);   // touch
    }
    requested_.insert(key);
}

void TerrainDetailStream::update(
    const std::shared_ptr<renderer::Device>& device,
    const glm::vec2& camera_pos_xz,
    game_object::MeshLoadTaskManager* loader) {
    if (tiles_dir_.empty() || !loader) return;

    // Which 3x3 block of detail tiles do we want resident?
    const float half = kTerrainMapMeters * 0.5f;
    const int cx = (int)std::floor((camera_pos_xz.x + half) / kDetailTileMeters);
    const int cy = (int)std::floor((camera_pos_xz.y + half) / kDetailTileMeters);

    bool table_dirty = false;
    bool want[kTilesSide * kTilesSide] = {};
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int tx = cx + dx, ty = cy + dy;
            if (tx < 0 || ty < 0 || tx >= kTilesSide || ty >= kTilesSide)
                continue;
            want[ty * kTilesSide + tx] = true;
        }
    }

    // Evict resident slots we no longer want (never evict mid-load).
    for (int i = 0; i < (int)kSlots; ++i) {
        Slot& s = slots_[i];
        if (s.tile_x >= 0 && !s.loading &&
            !want[s.tile_y * kTilesSide + s.tile_x]) {
            table_cpu_.slot_map[s.tile_y * kTilesSide + s.tile_x] = -1;
            table_cpu_.color_slot[s.tile_y * kTilesSide + s.tile_x] = -1;
            requested_.erase(s.tile_y * kTilesSide + s.tile_x);
            s = Slot{};
            table_dirty = true;
        }
    }

    // Request / load wanted tiles.
    std::error_code ec;
    for (int idx = 0; idx < kTilesSide * kTilesSide; ++idx) {
        if (!want[idx]) continue;
        if (table_cpu_.slot_map[idx] >= 0) continue;          // resident
        const int tx = idx % kTilesSide, ty = idx / kTilesSide;
        bool loading = false;
        for (const auto& s : slots_)
            if (s.loading && s.tile_x == tx && s.tile_y == ty) loading = true;
        if (loading) continue;

        const std::string png = tilePngPath(tiles_dir_, tx, ty);
        if (!fs::exists(png, ec)) {
            requestTile(tx, ty);
            continue;
        }
        const int slot = findFreeSlot();
        if (slot < 0) continue;              // all 9 busy; retry next frame

        slots_[slot].tile_x = tx;
        slots_[slot].tile_y = ty;
        slots_[slot].loading = true;
        submitTileLoad(device, loader, tx, ty, slot);
    }

    // ── Colour retry ────────────────────────────────────────────────
    // A tile can go resident without its 1 m colour (the worker writes
    // the colour AFTER... older workers wrote the height PNG first, and
    // ML colour tiles can simply take longer).  Re-load such tiles once
    // the colour file shows up, else the near field stays on the blurry
    // global map forever.
    {
        static double s_last_color_scan = 0.0;
        const double now = nowSeconds();
        if (now - s_last_color_scan > 2.0) {
            s_last_color_scan = now;
            for (int i = 0; i < (int)kSlots; ++i) {
                Slot& s = slots_[i];
                if (s.tile_x < 0 || s.loading) continue;
                const int key = s.tile_y * kTilesSide + s.tile_x;
                if (table_cpu_.slot_map[key] != i) continue;
                if (table_cpu_.color_slot[key] >= 0) continue;
                const std::string cpng =
                    tilePngPath(tiles_dir_, s.tile_x, s.tile_y);
                const std::string color_png =
                    cpng.substr(0, cpng.size() - 4) + "_color.png";
                if (fs::exists(color_png, ec)) {
                    // Unmap while reloading: the upload rewrites this
                    // array layer on the loader queue, and in-flight
                    // frames must not keep sampling it meanwhile (the
                    // height falls back to the base map for a moment).
                    table_cpu_.slot_map[key] = -1;
                    table_dirty = true;
                    s.loading = true;
                    submitTileLoad(device, loader, s.tile_x, s.tile_y, i);
                }
            }
        }
    }

    if (table_dirty) uploadTable(device);
    updateWorkerHealth(ec);
}

void TerrainDetailStream::submitTileLoad(
    const std::shared_ptr<renderer::Device>& device,
    game_object::MeshLoadTaskManager* loader,
    int tx, int ty, int slot) {
    const std::string png = tilePngPath(tiles_dir_, tx, ty);

    // Hot-load through the async loader: decode + upload on the
    // loader queue (Phase 2, worker thread), publish the table entry
    // on the main thread (Phase 3).  The matching 1 m colour tile
    // (tile_XX_YY_color.png) rides along when present.
    auto staging = std::make_shared<er::BufferInfo>();
        auto color_staging = std::make_shared<er::BufferInfo>();
        auto has_color = std::make_shared<bool>(false);
        TerrainDetailStream* self = this;
        auto image = height_array_image_;
        auto color_image = color_array_image_;
        const std::string color_png =
            png.substr(0, png.size() - 4) + "_color.png";
        loader->submit(
            png,
            /*phase2*/
            [png, color_png, staging, color_staging, has_color,
             image, color_image, slot](
                const std::shared_ptr<er::Device>& dev,
                const std::shared_ptr<er::CommandBuffer>& cmd_buf,
                std::string& error_out) -> bool {
                int w = 0, h = 0, comp = 0;
                stbi_us* pixels = stbi_load_16(png.c_str(), &w, &h, &comp,
                                               STBI_grey);
                if (!pixels) {
                    error_out = "stbi_load_16 failed: " + png;
                    return false;
                }
                if (w != (int)kTileRes || h != (int)kTileRes) {
                    stbi_image_free(pixels);
                    error_out = "unexpected detail tile size: " + png;
                    return false;
                }
                const uint64_t bytes = (uint64_t)w * h * sizeof(uint16_t);
                er::Helper::createBuffer(
                    dev,
                    SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
                    SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                    HOST_COHERENT_BIT),
                    0,
                    staging->buffer,
                    staging->memory,
                    std::source_location::current(),
                    bytes,
                    pixels);
                stbi_image_free(pixels);

                // UNDEFINED source layout: the helper doesn't support
                // SHADER_READ_ONLY -> TRANSFER_DST ("unsupported layout
                // transition"), and the layer is fully overwritten.
                er::Helper::transitionImageLayout(
                    cmd_buf, image, er::Format::R16_UNORM,
                    er::ImageLayout::UNDEFINED,
                    er::ImageLayout::TRANSFER_DST_OPTIMAL,
                    0, 1, slot, 1);
                er::BufferImageCopyInfo region = {};
                region.buffer_offset = 0;
                region.buffer_row_length = 0;
                region.buffer_image_height = 0;
                region.image_subresource.aspect_mask =
                    SET_FLAG_BIT(ImageAspect, COLOR_BIT);
                region.image_subresource.mip_level = 0;
                region.image_subresource.base_array_layer = slot;
                region.image_subresource.layer_count = 1;
                region.image_offset = glm::ivec3(0);
                region.image_extent = glm::uvec3(kTileRes, kTileRes, 1);
                cmd_buf->copyBufferToImage(
                    staging->buffer, image, { region },
                    er::ImageLayout::TRANSFER_DST_OPTIMAL);
                er::Helper::transitionImageLayout(
                    cmd_buf, image, er::Format::R16_UNORM,
                    er::ImageLayout::TRANSFER_DST_OPTIMAL,
                    er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    0, 1, slot, 1);

                // Optional 1 m albedo tile (2048^2 RGBA8).
                int cw = 0, ch = 0, ccomp = 0;
                stbi_uc* cpix = stbi_load(color_png.c_str(),
                                          &cw, &ch, &ccomp, STBI_rgb_alpha);
                if (cpix && cw == 2048 && ch == 2048) {
                    const uint64_t cbytes = (uint64_t)cw * ch * 4;
                    er::Helper::createBuffer(
                        dev,
                        SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
                        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT,
                                        HOST_COHERENT_BIT),
                        0,
                        color_staging->buffer,
                        color_staging->memory,
                        std::source_location::current(),
                        cbytes,
                        cpix);
                    er::Helper::transitionImageLayout(
                        cmd_buf, color_image, er::Format::R8G8B8A8_UNORM,
                        er::ImageLayout::UNDEFINED,
                        er::ImageLayout::TRANSFER_DST_OPTIMAL,
                        0, 1, slot, 1);
                    er::BufferImageCopyInfo cregion = region;
                    cregion.image_extent = glm::uvec3(cw, ch, 1);
                    cmd_buf->copyBufferToImage(
                        color_staging->buffer, color_image, { cregion },
                        er::ImageLayout::TRANSFER_DST_OPTIMAL);
                    er::Helper::transitionImageLayout(
                        cmd_buf, color_image, er::Format::R8G8B8A8_UNORM,
                        er::ImageLayout::TRANSFER_DST_OPTIMAL,
                        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        0, 1, slot, 1);
                    *has_color = true;
                }
                if (cpix) stbi_image_free(cpix);
                return true;
            },
            /*phase3*/
            [self, staging, color_staging, has_color, slot, tx, ty,
             device]() {
                Slot& s = self->slots_[slot];
                s.loading = false;
                if (s.tile_x == tx && s.tile_y == ty) {
                    self->table_cpu_.slot_map[ty * kTilesSide + tx] = slot;
                    self->table_cpu_.color_slot[ty * kTilesSide + tx] =
                        *has_color ? slot : -1;
                    self->uploadTable(device);
                    std::cout << "[terrain-detail] tile (" << tx << "," << ty
                              << ") resident in slot " << slot
                              << (*has_color ? " (+1m albedo)" : "")
                              << std::endl;
                }
                staging->destroy(device);
                color_staging->destroy(device);
            });
}

void TerrainDetailStream::updateWorkerHealth(std::error_code& ec) {
    // Worker health: if we're waiting on tiles and the heartbeat is
    // stale, respawn (detached worker may have idle-exited).
    if (!requested_.empty() && !worker_cmd_.empty()) {
        const double now = nowSeconds();
        if (now - last_worker_check_ > 20.0) {
            last_worker_check_ = now;
            const std::string alive =
                (fs::path(tiles_dir_) / "worker.alive").string();
            bool stale = true;
            if (fs::exists(alive, ec)) {
                const auto t = fs::last_write_time(alive, ec);
                if (!ec) {
                    const auto age = std::chrono::duration_cast<
                        std::chrono::seconds>(
                        fs::file_time_type::clock::now() - t).count();
                    stale = age > 15;
                }
            }
            if (stale) {
                std::cout << "[terrain-detail] worker heartbeat stale — "
                             "respawning" << std::endl;
                std::system(worker_cmd_.c_str());
            }
        }
    }
}

void TerrainDetailStream::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    if (height_array_view_)  device->destroyImageView(height_array_view_);
    if (height_array_image_) device->destroyImage(height_array_image_);
    if (height_array_memory_) device->freeMemory(height_array_memory_);
    if (color_array_view_)  device->destroyImageView(color_array_view_);
    if (color_array_image_) device->destroyImage(color_array_image_);
    if (color_array_memory_) device->freeMemory(color_array_memory_);
    if (table_buffer_) device->destroyBuffer(table_buffer_);
    if (table_memory_) device->freeMemory(table_memory_);
    height_array_view_.reset();
    height_array_image_.reset();
    height_array_memory_.reset();
    color_array_view_.reset();
    color_array_image_.reset();
    color_array_memory_.reset();
    table_buffer_.reset();
    table_memory_.reset();
}

}  // namespace scene_rendering
}  // namespace engine
