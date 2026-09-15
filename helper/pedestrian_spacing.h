#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

namespace engine::game_object {
// Local occupancy, shared by residents and ambient walkers. Swept checks
// prevent a large simulation step from jumping through someone in a queue.
class PedestrianSpacing {
    struct Entry { uint64_t id; glm::vec3 p; };
    std::unordered_map<uint64_t,std::vector<Entry>> cells_;
    static uint64_t key(int x,int z) { return (uint64_t(uint32_t(x))<<32)|uint32_t(z); }
    static int cell(float x) { return int(std::floor(x/2.f)); }
public:
    void clear() { cells_.clear(); }
    void add(uint64_t id,glm::vec3 p) { cells_[key(cell(p.x),cell(p.z))].push_back({id,p}); }
    bool allow(uint64_t id,glm::vec3 from,glm::vec3 to) const {
        const glm::vec2 a(from.x,from.z), d(to.x-from.x,to.z-from.z);
        const float len2=glm::dot(d,d);
        if(len2<1e-10f) return true;
        constexpr float radius=.68f;
        for(int z=cell(std::min(from.z,to.z)-radius);z<=cell(std::max(from.z,to.z)+radius);++z)
        for(int x=cell(std::min(from.x,to.x)-radius);x<=cell(std::max(from.x,to.x)+radius);++x) {
            auto it=cells_.find(key(x,z)); if(it==cells_.end()) continue;
            for(const auto& e:it->second) {
                if(e.id==id || std::abs(e.p.y-from.y)>1.5f) continue;
                glm::vec2 q(e.p.x,e.p.z); auto v=a-q;
                float start=glm::dot(v,v);
                // Let an existing overlap escape, but never move further in.
                if(start<radius*radius && glm::dot(v,d)>=0.f) continue;
                float t=glm::clamp(-glm::dot(v,d)/len2,0.f,1.f);
                auto closest=v+d*t;
                if(glm::dot(closest,closest)<radius*radius) return false;
            }
        }
        return true;
    }
    void move(uint64_t id,glm::vec3 from,glm::vec3 to) {
        auto it=cells_.find(key(cell(from.x),cell(from.z)));
        if(it!=cells_.end()) {
            auto& entries=it->second;
            entries.erase(std::remove_if(entries.begin(),entries.end(),[&](const Entry& e){return e.id==id;}),entries.end());
        }
        add(id,to);
    }
};
inline bool pedestrianMoving(glm::vec3 a,glm::vec3 b,float dt) {
    const glm::vec2 d(b.x-a.x,b.z-a.z);
    return dt>0.f && glm::dot(d,d)>.0004f*dt*dt;
}
}
