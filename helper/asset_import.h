#pragma once
//
// asset_import.h  --  headless GLB/model import into the content/ tree.
//
// The ONE implementation of "this generated file becomes managed content".
// Compiled into the asset_importer plugin DLL — the terrain pipeline calls
// that in-process via ctypes, and RealWorld's --import-asset flag loads
// it per call (falling back to this compiled-in copy when the DLL is
// absent).  Everything here is CPU file work — no window, no
// Vulkan device — which is the entire point: importing an asset must
// never be able to boot an editor.
//
//   bake — full render-ready conversion: <dir>/<stem>/objects/NNN.rwgeo,
//          textures/*.rwtex, one placeable .rwobj per object,
//          import.rwmeta, and rows in content/asset_index.tsv.  For
//          static geometry (the terrain surface mesh).
//   copy — a MANAGED GROUP (the character-import pattern):
//          <dir>/<stem>/<stem>.glb + import.rwmeta with main=, plus
//          group/model index rows.  For instanced/banded files whose
//          in-engine behaviour lives in the glTF loader
//          (EXT_mesh_gpu_instancing transforms, _lodtile_ bands,
//          world-manifest overrides): for those, glTF IS the native
//          runtime representation — .rwgeo cannot carry instancing —
//          so the payload stays glTF but lives inside a managed group,
//          never as a loose file.  Skinned and instanced sources
//          handed to bake mode fall back to this automatically.
//
// Paths in the sidecars are written project-relative (see the portable-
// path notes in application.cpp); content/asset_index.tsv is resolved
// against the CURRENT WORKING DIRECTORY, which callers set to the
// project root (realworld/).
//
#include <string>

namespace engine {
namespace helper {

// Import one file.  `mode` is "bake" or "copy" (anything else = bake).
// Returns false on any failure; diagnostics go to stdout/stderr.
bool importOneAssetToContent(const std::string& source,
                             const std::string& import_dir,
                             const std::string& mode);

// Import a ';'-separated list.  Returns a process exit code
// (0 = every source imported).
int importAssetsToContent(const std::string& sources,
                          const std::string& import_dir,
                          const std::string& mode);

} // namespace helper
} // namespace engine
