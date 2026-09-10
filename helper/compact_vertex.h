#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <string_view>
#include <algorithm>
#include <cctype>
#include <vector>
#include <limits>
#include <cstring>
namespace engine::helper {
// Explicit exclusions supplement the geometric precision test. Match the
// asset/mesh name, not a generic /terrain/ parent shared by all PCG assets.
inline bool requiresFloatObjectPositions(std::string_view asset) {
    std::string name(asset);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto has = [&](const char* text) { return name.find(text) != std::string::npos; };
    const auto slash = name.find_last_of("/\\");
    const auto base = name.substr(slash == std::string::npos ? 0 : slash + 1);
    return has("house") || has("building") || has("road") || has("terrain_tile") ||
           has("huge_rock") || has("large_rock") || has("giant_rock") || has("crag") ||
           base.rfind("tile_", 0) == 0;
}

// Vulkan A2B10G10R10_SNORM_PACK32: x in low 10 bits, y then z.
struct CompactVertex {
    float position[3];
    uint32_t normal;
    float uv[2];
};
static_assert(sizeof(CompactVertex) == 24);
static_assert(offsetof(CompactVertex, normal) == 12);
static_assert(offsetof(CompactVertex, uv) == 16);
inline CompactVertex compactVertex(const glm::vec3& p, const glm::vec3& n,
                                   const glm::vec2& uv) {
    const float len2 = glm::dot(n, n);
    const glm::vec3 unit = std::isfinite(len2) && len2 > 1e-20f
        ? n / std::sqrt(len2) : glm::vec3(0, 1, 0);
    return {{p.x, p.y, p.z}, glm::packSnorm3x10_1x2(glm::vec4(unit, 0)), {uv.x, uv.y}};
}
// 20-byte fallback for large plant bounds; UV remains FP16.
struct FloatPlantVertex {
    float position[3];
    uint32_t normal;
    uint32_t uv;
};
static_assert(sizeof(FloatPlantVertex) == 20);
inline bool halfPlantUvSafe(const glm::vec2& uv) {
    for (int i=0;i<2;++i) {
        const float q=glm::unpackHalf1x16(glm::packHalf1x16(uv[i]));
        if (!std::isfinite(uv[i]) || !std::isfinite(q) || std::abs(q-uv[i])>0.001f) return false;
    }
    return true;
}
inline bool halfPlantBoundsSafe(const glm::vec3& lo, const glm::vec3& hi) {
    // Within [-32,32] m, FP16 position rounding stays below 1 cm.
    // Extent alone is insufficient: a small box far from the origin
    // still loses precision without an explicit per-mesh origin decode.
    for (int i=0;i<3;++i)
        if (!std::isfinite(lo[i]) || !std::isfinite(hi[i]) || hi[i]<lo[i] ||
            hi[i]-lo[i]>64.0f || lo[i]<-32.0f || hi[i]>32.0f) return false;
    return true;
}
inline FloatPlantVertex floatPlantVertex(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
    return {{p.x,p.y,p.z}, compactVertex(p,n,uv).normal, glm::packHalf2x16(uv)};
}
struct QuantizedVertex {
    uint32_t position_xy, position_zw, normal, uv;
};
struct QuantizedBounds {
    float bias[4], scale[4];
    float as_transform[12]; // row-major VkTransformMatrixKHR
};
static_assert(sizeof(QuantizedVertex)==16 && sizeof(QuantizedBounds)==80);
inline QuantizedBounds quantizedBounds(const glm::vec3& lo, const glm::vec3& hi) {
    const glm::vec3 extent=hi-lo;
    const glm::vec3 scale(extent.x>0?extent.x:1, extent.y>0?extent.y:1, extent.z>0?extent.z:1);
    return {{lo.x,lo.y,lo.z,0},{scale.x,scale.y,scale.z,0},
        {scale.x,0,0,lo.x,0,scale.y,0,lo.y,0,0,scale.z,lo.z}};
}
inline bool quantizedBoundsSafe(const glm::vec3& lo,const glm::vec3& hi) {
    for(int i=0;i<3;++i) if(!std::isfinite(lo[i]) || !std::isfinite(hi[i]) ||
        hi[i]<lo[i] || hi[i]-lo[i]>64.0f) return false;
    return true;
}
inline QuantizedVertex quantizedVertex(const glm::vec3& p,const glm::vec3& n,
                                       const glm::vec2& uv,const QuantizedBounds& bounds) {
    glm::vec3 q(0);
    for(int i=0;i<3;++i) if(bounds.scale[i]>0)
        q[i]=glm::clamp((p[i]-bounds.bias[i])/bounds.scale[i],0.0f,1.0f);
    return {glm::packUnorm2x16(glm::vec2(q)),glm::packUnorm2x16(glm::vec2(q.z,1)),
        compactVertex(p,n,uv).normal,glm::packHalf2x16(uv)};
}
struct HalfPlantVertex {
    uint32_t position_xy, position_zw;
    uint32_t normal;
    uint32_t uv;
};
static_assert(sizeof(HalfPlantVertex) == 16);
inline bool halfPlantVertexSafe(const glm::vec3& p, const glm::vec2& uv) {
    // Local plant geometry only: never silently quantize large world positions.
    for (int i=0;i<3;++i) {
        float q=glm::unpackHalf1x16(glm::packHalf1x16(p[i]));
        if (!std::isfinite(p[i]) || !std::isfinite(q) || std::abs(q-p[i])>0.01f) return false;
    }
    for (int i=0;i<2;++i) {
        float q=glm::unpackHalf1x16(glm::packHalf1x16(uv[i]));
        if (!std::isfinite(uv[i]) || !std::isfinite(q) || std::abs(q-uv[i])>0.001f) return false;
    }
    return true;
}
inline HalfPlantVertex halfPlantVertex(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
    return {glm::packHalf2x16(glm::vec2(p)), glm::packHalf2x16(glm::vec2(p.z,1)),
            compactVertex(p,n,uv).normal, glm::packHalf2x16(uv)};
}
struct PackedObjectVertices {
    std::vector<uint8_t> bytes;
    uint32_t stride=24, vertex_offset=0, normal_offset=12, uv_offset=16;
    bool quantized=false, half_uv=false;
};
template<class Vertices>
PackedObjectVertices packObjectVertices(const Vertices& vertices, bool format_supported,
                                        std::string_view asset) {
    PackedObjectVertices out;
    if(vertices.empty()) return out;
    glm::vec3 lo(std::numeric_limits<float>::max()),hi(std::numeric_limits<float>::lowest());
    bool finite=true, uv_ok=true;
    for(const auto& v:vertices) {
        for(int i=0;i<3;++i) finite &= std::isfinite(v.position[i]);
        lo=glm::min(lo,v.position); hi=glm::max(hi,v.position);
        uv_ok &= halfPlantUvSafe(v.uv);
    }
    out.half_uv=format_supported && uv_ok;
    out.quantized=out.half_uv && finite && !requiresFloatObjectPositions(asset) && quantizedBoundsSafe(lo,hi);
    out.stride=out.quantized?16u:(out.half_uv?20u:24u);
    out.normal_offset=out.quantized?8u:12u; out.uv_offset=out.quantized?12u:16u;
    auto append=[&](const auto& value) {
        const auto* b=reinterpret_cast<const uint8_t*>(&value);
        out.bytes.insert(out.bytes.end(),b,b+sizeof(value));
    };
    out.bytes.reserve(vertices.size()*out.stride+(out.quantized?80:0));
    const auto bounds=quantizedBounds(lo,hi);
    if(out.quantized) {append(bounds);out.vertex_offset=sizeof(bounds);}
    for(const auto& v:vertices) {
        if(out.quantized) append(quantizedVertex(v.position,v.normal,v.uv,bounds));
        else if(out.half_uv) append(floatPlantVertex(v.position,v.normal,v.uv));
        else append(compactVertex(v.position,v.normal,v.uv));
    }
    return out;
}
} // namespace engine::helper
