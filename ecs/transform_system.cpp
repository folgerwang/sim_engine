// transform_system.cpp — see transform_system.h.
#include "ecs/transform_system.h"

#include <functional>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>

#include "ecs/components.h"

namespace engine {
namespace ecs {

namespace {

// World AABB of a local AABB transformed by `m`. Standard absolute-matrix
// trick: the world half-extents are |M3x3| * local_extents.
WorldBounds transformBounds(const LocalBounds& lb, const glm::mat4& m) {
    WorldBounds wb;
    wb.center = glm::vec3(m * glm::vec4(lb.center, 1.0f));
    const glm::mat3 abs3{
        glm::abs(glm::vec3(m[0])),
        glm::abs(glm::vec3(m[1])),
        glm::abs(glm::vec3(m[2])),
    };
    wb.extents = abs3 * lb.extents;
    return wb;
}

// Write WorldTransform (+ WorldBounds if present) for `e` given its world mat.
void writeWorld(entt::registry& reg, Entity e, const glm::mat4& world) {
    reg.get<WorldTransform>(e).matrix = world;
    if (auto* lb = reg.try_get<LocalBounds>(e)) {
        reg.emplace_or_replace<WorldBounds>(e, transformBounds(*lb, world));
    }
}

}  // namespace

glm::mat4 TransformSystem::computeWorld(entt::registry& reg, Entity e) {
    const auto* lt = reg.try_get<LocalTransform>(e);
    const glm::mat4 local = lt ? lt->toMatrix() : glm::mat4(1.0f);
    if (const auto* p = reg.try_get<Parent>(e); p && p->parent != kNull &&
                                                reg.valid(p->parent)) {
        return computeWorld(reg, p->parent) * local;
    }
    return local;
}

size_t TransformSystem::update(entt::registry& reg) {
    // Nothing dirty → cheap early-out.
    auto dirty_view = reg.view<DirtyTransform>();
    if (dirty_view.begin() == dirty_view.end()) return 0;

    // 1) Build parent→children adjacency from the Parent pool.
    std::unordered_map<Entity, std::vector<Entity>> children;
    for (auto [e, parent] : reg.view<Parent>().each()) {
        if (parent.parent != kNull) children[parent.parent].push_back(e);
    }

    // 2) Propagate dirtiness down: a dirty entity dirties its whole subtree.
    //    Collect the set of entities that need recompute.
    std::vector<Entity> roots_to_walk;
    for (auto e : dirty_view) roots_to_walk.push_back(e);

    std::unordered_map<Entity, bool> need;  // entity -> needs recompute
    std::vector<Entity> stack = roots_to_walk;
    while (!stack.empty()) {
        Entity e = stack.back();
        stack.pop_back();
        if (need.count(e)) continue;
        need[e] = true;
        if (auto it = children.find(e); it != children.end()) {
            for (Entity c : it->second) stack.push_back(c);
        }
    }

    // 3) Recompute each needed entity, memoizing world matrices. A recursive
    //    resolver guarantees parents are computed before children regardless
    //    of iteration order.
    std::unordered_map<Entity, glm::mat4> world_cache;
    std::function<glm::mat4(Entity)> resolve = [&](Entity e) -> glm::mat4 {
        if (auto it = world_cache.find(e); it != world_cache.end())
            return it->second;

        const auto* lt = reg.try_get<LocalTransform>(e);
        const glm::mat4 local = lt ? lt->toMatrix() : glm::mat4(1.0f);

        glm::mat4 world = local;
        if (const auto* p = reg.try_get<Parent>(e);
            p && p->parent != kNull && reg.valid(p->parent)) {
            // Parent that itself needs recompute is resolved here; a parent
            // that is already clean uses its cached WorldTransform.
            if (need.count(p->parent)) {
                world = resolve(p->parent) * local;
            } else if (const auto* pw = reg.try_get<WorldTransform>(p->parent)) {
                world = pw->matrix * local;
            }
        }
        world_cache[e] = world;
        return world;
    };

    size_t updated = 0;
    for (auto& [e, _] : need) {
        if (!reg.all_of<WorldTransform>(e)) reg.emplace<WorldTransform>(e);
        writeWorld(reg, e, resolve(e));
        ++updated;
    }

    // 4) Clear dirty flags (only the originally-flagged ones carry the tag).
    for (Entity e : roots_to_walk) {
        if (reg.valid(e)) reg.remove<DirtyTransform>(e);
    }
    return updated;
}

}  // namespace ecs
}  // namespace engine
