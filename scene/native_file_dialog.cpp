#include "scene/native_file_dialog.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#pragma comment(lib, "comdlg32.lib")
#endif

namespace engine {
namespace scene {

#ifdef _WIN32
namespace {
// `filter` is a double-null-terminated set of "label\0pattern\0" pairs.
std::string openWithFilter(const char* filter, const char* title,
                           void* owner_hwnd,
                           const char* initial_dir = nullptr) {
    char file_buf[MAX_PATH] = { 0 };

    // Explicit ...A (ANSI) variants so we work with std::string regardless of
    // the project's global UNICODE define.
    OPENFILENAMEA ofn = {};
    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = reinterpret_cast<HWND>(owner_hwnd);
    ofn.lpstrFilter  = filter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFile    = file_buf;
    ofn.nMaxFile     = MAX_PATH;
    ofn.lpstrTitle   = title;
    // Starting folder (e.g. content/scene for scene files).  NULL keeps
    // the OS default (last-visited folder).
    ofn.lpstrInitialDir =
        (initial_dir && initial_dir[0]) ? initial_dir : nullptr;
    // OFN_NOCHANGEDIR: the engine loads assets via paths relative to the
    // working directory; without this the common dialog would change CWD to
    // the browsed folder and break subsequent relative asset loads.
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(file_buf);
    }
    return std::string();
}
} // namespace
#endif

std::string openModelFileDialog(void* owner_hwnd) {
#ifdef _WIN32
    return openWithFilter(
        "3D models (*.gltf;*.glb;*.fbx;*.obj)\0*.gltf;*.glb;*.fbx;*.obj\0"
        "All files (*.*)\0*.*\0",
        "Import model into scene", owner_hwnd);
#else
    (void)owner_hwnd;
    return std::string();
#endif
}

std::string openSceneFileDialog(void* owner_hwnd,
                                const char* initial_dir) {
#ifdef _WIN32
    return openWithFilter(
        "Scene files (*.scene)\0*.scene\0"
        "All files (*.*)\0*.*\0",
        "Load scene", owner_hwnd, initial_dir);
#else
    (void)owner_hwnd;
    (void)initial_dir;
    return std::string();
#endif
}

} // namespace scene
} // namespace engine
