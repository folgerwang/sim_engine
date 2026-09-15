#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <glm/glm.hpp>

namespace engine::helper {
// Split true at-grade crossings before signal and marking generation. A spatial
// grid bounds comparisons; bridges at different elevations remain disconnected.
inline std::vector<std::vector<int>> splitRoadCrossings(
    std::vector<std::vector<glm::vec3>>& paths,
    std::vector<std::vector<float>>& widths) {
    struct Segment { size_t path, index; };
    struct Cut { float station; glm::vec3 point; float width; bool junction; };
    std::vector<Segment> segments;
    std::vector<std::vector<Cut>> additions(paths.size());
    std::unordered_map<uint64_t,std::vector<size_t>> grid;
    auto key=[](int x,int z) { return (uint64_t(uint32_t(x))<<32)|uint32_t(z); };
    auto cross=[](glm::vec2 a,glm::vec2 b) {return a.x*b.y-a.y*b.x;};
    for(size_t s=0;s<paths.size();++s) for(size_t i=1;i<paths[s].size();++i) {
        const auto a=paths[s][i-1], b=paths[s][i];
        const glm::vec2 aa(a.x,a.z), d(b.x-a.x,b.z-a.z);
        int x0=int(std::floor(std::min(a.x,b.x)/32.f)), x1=int(std::floor(std::max(a.x,b.x)/32.f));
        int z0=int(std::floor(std::min(a.z,b.z)/32.f)), z1=int(std::floor(std::max(a.z,b.z)/32.f));
        std::unordered_set<size_t> visited;
        for(int z=z0;z<=z1;++z) for(int x=x0;x<=x1;++x) {
            auto& bin=grid[key(x,z)];
            for(size_t id:bin) {
                if(!visited.insert(id).second) continue;
                const auto other=segments[id];
                if(other.path==s) continue;
                const auto c=paths[other.path][other.index], e=paths[other.path][other.index+1];
                glm::vec2 cc(c.x,c.z), v(e.x-c.x,e.z-c.z);
                float det=cross(d,v);
                if(std::abs(det)<1e-5f) continue;
                float t=cross(cc-aa,v)/det, u=cross(cc-aa,d)/det;
                if(t<0.f||t>1.f||u<0.f||u>1.f) continue;
                glm::vec3 pa=glm::mix(a,b,t), pb=glm::mix(c,e,u);
                if(std::abs(pa.y-pb.y)>2.f) continue;
                auto point=(pa+pb)*.5f;
                additions[s].push_back({float(i-1)+t,point,glm::mix(widths[s][i-1],widths[s][i],t),true});
                additions[other.path].push_back({float(other.index)+u,point,
                    glm::mix(widths[other.path][other.index],widths[other.path][other.index+1],u),true});
            }
        }
        const size_t id=segments.size();segments.push_back({s,i-1});
        for(int z=z0;z<=z1;++z) for(int x=x0;x<=x1;++x) grid[key(x,z)].push_back(id);
    }
    std::vector<std::vector<int>> cuts(paths.size());
    for(size_t s=0;s<paths.size();++s) {
        if(additions[s].empty()) continue;
        auto& samples=additions[s];
        for(size_t i=0;i<paths[s].size();++i) samples.push_back({float(i),paths[s][i],widths[s][i],false});
        std::sort(samples.begin(),samples.end(),[](const Cut&a,const Cut&b){
            return a.station==b.station ? a.junction>b.junction : a.station<b.station;});
        paths[s].clear();widths[s].clear();float last=-10.f;
        for(const auto& c:samples) {
            if(c.station-last<1e-5f) continue;
            if(c.junction) cuts[s].push_back(int(paths[s].size()));
            paths[s].push_back(c.point);widths[s].push_back(c.width);last=c.station;
        }
    }
    return cuts;
}
}
