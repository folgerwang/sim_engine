#pragma once
//
// terrain_detail_stream.h — runtime 1 m terrain detail tile streaming.
//
// The terrain renders from an 8192^2 BASE heightmap covering the 32 km
// world (4 m/texel).  Near the camera the tile shaders want 1 m data:
// the world is split into 16x16 DETAIL TILES of 2048 m (2049^2 texels,
// centers on integer world meters — see terrain_detail_worker.py).
//
// This class keeps a 3x3 ring of detail tiles resident around the camera
// in an R16 texture ARRAY (kDetailCacheSlots layers) plus a small SSBO
// mapping world tile index -> array slice (-1 = not resident).  Tiles are
// produced on demand by a detached python worker
// (tools/terrain/terrain_detail_worker.py): we touch "req_XX_YY" in the
// tiles dir, the worker answers with "tile_XX_YY.png" (atomic rename),
// and we hot-load it through MeshLoadTaskManager (decode + GPU upload on
// the loader queue, publish on the main thread).
//
// Seamlessness: tile content is a pure function of world position (base
// Catmull-Rom + world-anchored noise), and border texels of adjacent
// tiles are bit-identical, so no gutters or cross-tile blending are
// needed.  The shader fades detail -> base by camera distance
// (kDetailFadeStart/EndMeters) so the boundary of the RESIDENT region
// never shows a step.
//
#include <memory>
#include <string>
#include <unordered_set>

#include "renderer/renderer.h"

namespace engine {
namespace game_object { class MeshLoadTaskManager; }
namespace scene_rendering {

class TerrainDetailStream {
public:
    // GPU-side TerrainDetailTable mirror (layout must match the SSBO in
    // tile_detail.glsl.h: two int[256] arrays back-to-back).
    struct TableCpu {
        int32_t slot_map[16 * 16];    // world tile -> array slice (-1)
        int32_t color_slot[16 * 16];  // -1 when no 1 m colour tile
    };

    TerrainDetailStream(
        const std::shared_ptr<renderer::Device>& device);

    // New terrain applied: forget everything, point at the new tiles dir
    // and remember how to (re)spawn the worker.  Passing empty strings
    // disables streaming (no terrain created yet).
    void setSource(
        const std::shared_ptr<renderer::Device>& device,
        const std::string& tiles_dir,
        const std::string& worker_spawn_cmd);

    // Per-frame (main thread, OUTSIDE command recording): request/evict
    // tiles for the camera position and hot-load finished ones.
    void update(
        const std::shared_ptr<renderer::Device>& device,
        const glm::vec2& camera_pos_xz,
        game_object::MeshLoadTaskManager* loader);

    const std::shared_ptr<renderer::ImageView>& getHeightArrayView() const {
        return height_array_view_;
    }
    const std::shared_ptr<renderer::ImageView>& getColorArrayView() const {
        return color_array_view_;
    }
    const std::shared_ptr<renderer::Buffer>& getTableBuffer() const {
        return table_buffer_;
    }
    static uint32_t tableBytes() { return sizeof(TableCpu); }

    void destroy(const std::shared_ptr<renderer::Device>& device);

private:
    struct Slot {
        int32_t tile_x = -1;      // world detail-tile index, -1 = free
        int32_t tile_y = -1;
        bool    loading = false;
    };

    void uploadTable(const std::shared_ptr<renderer::Device>& device);
    void requestTile(int tx, int ty);
    int  findFreeSlot() const;
    // Decode + upload tile (tx,ty) into `slot` via the async loader
    // (height + optional 1 m colour).  Used for first loads and for
    // colour retries (see update()).
    void submitTileLoad(
        const std::shared_ptr<renderer::Device>& device,
        game_object::MeshLoadTaskManager* loader,
        int tx, int ty, int slot);
    void updateWorkerHealth(std::error_code& ec);

    // GPU resources.
    std::shared_ptr<renderer::Image>       height_array_image_;
    std::shared_ptr<renderer::DeviceMemory> height_array_memory_;
    std::shared_ptr<renderer::ImageView>   height_array_view_;
    // 1 m albedo tiles (RGBA8 2048^2 per slot, streamed with heights).
    std::shared_ptr<renderer::Image>       color_array_image_;
    std::shared_ptr<renderer::DeviceMemory> color_array_memory_;
    std::shared_ptr<renderer::ImageView>   color_array_view_;
    std::shared_ptr<renderer::Buffer>      table_buffer_;
    std::shared_ptr<renderer::DeviceMemory> table_memory_;

    Slot     slots_[9];
    TableCpu table_cpu_;

    std::string tiles_dir_;
    std::string worker_cmd_;
    // Tiles with an outstanding req_ file (avoid re-touching every frame).
    std::unordered_set<int> requested_;
    double last_worker_check_ = 0.0;
};

}  // namespace scene_rendering
}  // namespace engine
