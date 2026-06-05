#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// native_file_dialog.h — minimal OS "open file" dialogs.
//
// Each returns the selected absolute path, or an empty string if the user
// cancels.  `owner_hwnd` is an optional native window handle (passed as void*
// so this header pulls in no platform headers); pass the app window's HWND to
// keep the dialog modal to it, or nullptr.
// ─────────────────────────────────────────────────────────────────────────────
#include <string>

namespace engine {
namespace scene {

// Filtered to model files (*.gltf;*.glb;*.fbx;*.obj).
std::string openModelFileDialog(void* owner_hwnd = nullptr);

// Filtered to scene files (*.scene).
std::string openSceneFileDialog(void* owner_hwnd = nullptr);

} // namespace scene
} // namespace engine
