#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// scene_io.h — custom binary (de)serialization for the scene authoring model.
//
// Format (little-endian, native x64):
//   char    magic[8] = "RWSCENE\0"
//   uint32  version  = 1
//   string  scene name           (uint32 length + raw bytes)
//   Transform root               (10 float32: tx ty tz  rx ry rz rw  sx sy sz)
//   uint32  object count
//   repeat object count times:
//       string   name
//       string   asset_path
//       int32    parent_index
//       int32    source_node_index
//       uint8    is_group
//       uint8    visible
//       Transform transform      (10 float32)
//
// v2+: music_path (string) + music_volume (f32) trailer.
// v3+: per-object audio_clip (string) + audio_loop (u8) + audio_volume (f32).
// v4+: collision_map_path (string) trailer — baked .rwcmap reference.
//
// Returns false on any I/O error, bad magic, or unknown version.
// ─────────────────────────────────────────────────────────────────────────────
#include <string>

#include "scene/scene_types.h"

namespace engine {
namespace scene {

bool saveSceneBinary(const std::string& path, const Scene& scene);
bool loadSceneBinary(const std::string& path, Scene& out_scene);

} // namespace scene
} // namespace engine
