#pragma once
//
// material_classifier.h — LLM-backed mesh-category classifier.
//
// Collects (material_name, albedo_filename, object_name) triples from
// the loaded drawables, fires a single HTTP request to a LOCAL Ollama
// daemon at world-build time, parses the response, and exposes a
// lookup that gameplay / collision code can use to override the
// substring-based classifier in collision_mesh.cpp.
//
// Configuration (read from env vars at classifyAll() time):
//   OLLAMA_HOST   — "localhost:11434" by default.  Hostname[:port].
//   OLLAMA_MODEL  — "qwen3.5:2b" by default.  Any tag pulled locally
//                   via `ollama pull <tag>`.
//
// Designed to be best-effort and fail-quiet: when the daemon is
// unreachable, the model isn't pulled, the network request errors, or
// the response is malformed, classifyAll() returns false and lookup()
// yields MeshCategory::Unknown for every name — the caller is
// expected to fall back to the procedural classifier already in
// place.
//
// Threading: collect() is NOT thread-safe; call it serially during the
// scene-walk phase.  classifyAll() runs synchronously on the calling
// thread and typically takes 1–3 s for a few hundred materials.
// lookup() is safe to call concurrently with itself once classifyAll()
// has returned.
//
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "helper/collision_mesh.h"  // MeshCategory

namespace engine {
namespace helper {

class MaterialClassifier {
public:
    // Record one (material name, albedo filename, object name) triple.
    // Any field may be empty.  Material and object are deduped
    // separately — the same material is typically referenced by many
    // objects, and the same object can carry several materials, so
    // both indices grow independently.  Object names are normalised
    // (trailing _NNNN suffixes and runs of underscores stripped)
    // before insertion, which collapses thousands of instance names
    // like "Ashtray_01_mesh_4073" / "Ashtray_01_mesh___10__" down to
    // a single "Ashtray_01_mesh" identifier.
    void collect(
        const std::string& material_name,
        const std::string& albedo_filename,
        const std::string& object_name);

    // Fire one classification request via WinHTTP to the local Ollama
    // daemon's /api/chat endpoint, parse the JSON response into the
    // classified_ maps.  scene_label is included in the prompt (e.g.
    // "Bistro") so the model can use it as additional context for
    // ambiguous strings.  Model and host come from OLLAMA_MODEL and
    // OLLAMA_HOST env vars (defaults qwen3.5:2b and localhost:11434).
    //
    // The prompt contains TWO arrays — materials and objects — and
    // asks the LLM to return TWO maps; lookup() reads both and lets
    // the object verdict win when it disagrees with the material
    // verdict.
    //
    // Returns true on full success.  Returns false (and leaves the
    // classified maps empty, so every lookup() returns Unknown) when:
    //   - the Ollama daemon isn't reachable on OLLAMA_HOST,
    //   - the requested OLLAMA_MODEL isn't pulled locally,
    //   - WinHTTP returned a non-2xx status,
    //   - the response was not valid JSON, or
    //   - the JSON contained no recognisable category tags.
    //
    // Blocking — typically 1–30 s depending on model size and how
    // many names were collected; the first call after `ollama serve`
    // starts also pays a one-shot model-load cost (~seconds).
    bool classifyAll(const std::string& scene_label);

    // After classifyAll() succeeds, look up the LLM-assigned category
    // for a (material, object) pair.  Precedence:
    //   1. object verdict (if non-Unknown) — most specific signal,
    //   2. material verdict (if non-Unknown),
    //   3. Unknown — caller falls back to the substring classifier.
    // The object_name is normalised internally so the caller can
    // pass the raw NodeInfo::name_ string verbatim.
    MeshCategory lookup(
        const std::string& material_name,
        const std::string& object_name) const;

    size_t collectedMaterialCount() const { return mat_collected_.size(); }
    size_t collectedObjectCount()   const { return obj_collected_.size(); }

    // LIVE counts maintained by the worker thread with atomic stores.
    // Safe to read concurrently with classifyAll() running on the
    // worker — the running total of classified items grows
    // monotonically as each batch lands.  Used by the progress-bar
    // push so the UI ticks up batch-by-batch instead of jumping
    // from 0 to 511 at the very end.
    size_t classifiedMaterialCount() const {
        return mat_classified_count_.load(std::memory_order_acquire);
    }
    size_t classifiedObjectCount() const {
        return obj_classified_count_.load(std::memory_order_acquire);
    }

    // Read-only access to the classified maps for UI / logging.
    // These return references into the underlying storage and are
    // NOT thread-safe to call while classifyAll() is still running
    // on a worker thread.  Use them ONLY after the worker's future
    // has been joined (status == ready), or from the worker thread
    // itself.  Per-frame UI code on the main thread should prefer
    // snapshotClassified() below, which copies under the internal
    // mutex.
    using TagMap = std::unordered_map<std::string, MeshCategory>;
    const TagMap& classifiedMaterials() const { return mat_classified_; }
    const TagMap& classifiedObjects()   const { return obj_classified_; }

    // Thread-safe snapshot: copies the current contents of the
    // classified maps into the caller's TagMaps under the internal
    // mutex.  Safe to call concurrently with classifyAll(), e.g.
    // every frame from the main thread to feed the Mesh-Category
    // Inspector window incrementally as each batch lands.
    void snapshotClassified(TagMap& out_mats, TagMap& out_objs) const {
        std::lock_guard<std::mutex> lock(classified_mu_);
        out_mats = mat_classified_;
        out_objs = obj_classified_;
    }

    // True after classifyAll() has been called at least once
    // (regardless of success), so the caller can decide whether to
    // run the substring fallback for everything.
    bool wasRun() const { return was_run_; }

    // Live progress (worker-thread-published, main-thread-readable).
    // Use these for a UI progress bar — neither has happens-before
    // requirements with anything else, so the underlying atomics
    // are loaded with std::memory_order_relaxed.  Both reset at
    // every classifyAll() entry so the values reflect ONLY the
    // current run.
    //
    //   bytesSent()      — total request body bytes queued via
    //                      WinHttpSendRequest (one-shot, jumps to
    //                      the full body size as soon as the
    //                      transport queues it).
    //   bytesReceived()  — running total of response bytes drained
    //                      so far.  Climbs chunk-by-chunk as the
    //                      LLM streams tokens back through Ollama.
    static size_t bytesSent();
    static size_t bytesReceived();

    // Public so tests / callers can normalise names the same way the
    // classifier does (e.g. for logging).  Strips trailing _<digits>
    // and runs of underscores repeatedly until the tail stabilises.
    static std::string normalizeObjectName(const std::string& s);

private:
    // Material name → albedo texture filename (may be empty).
    std::unordered_map<std::string, std::string> mat_collected_;
    // Normalised object name → presence sentinel (we only need the set,
    // but using a map keeps the value slot free for future per-object
    // metadata like AABB or instance count).
    std::unordered_map<std::string, std::string> obj_collected_;

    // Material / object name → category, populated by classifyAll().
    // Written by the worker thread under classified_mu_ as each
    // batch's response is parsed; read by the main thread via
    // snapshotClassified() (also under the mutex) or via
    // classifiedMaterials() / classifiedObjects() once the worker
    // has finished.  The atomic count fields below mirror the map
    // sizes for lock-free progress reads.
    std::unordered_map<std::string, MeshCategory> mat_classified_;
    std::unordered_map<std::string, MeshCategory> obj_classified_;
    mutable std::mutex                            classified_mu_;
    std::atomic<size_t>                           mat_classified_count_{0};
    std::atomic<size_t>                           obj_classified_count_{0};
    bool was_run_ = false;
};

} // namespace helper
} // namespace engine
