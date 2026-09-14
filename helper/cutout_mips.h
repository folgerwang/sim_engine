#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdint>
#include <vector>
namespace engine::helper {
inline float cutoutCoverage(const uint8_t* pixels,size_t count,uint8_t cutoff=23) {
    size_t covered=0;
    for(size_t i=0;i<count;++i) covered+=pixels[i*4+3]>=cutoff;
    return count?float(covered)/float(count):0.f;
}
inline void preserveCutoutCoverage(std::vector<uint8_t>& pixels,float coverage,uint8_t cutoff=23) {
    if(coverage<=0.f || pixels.empty()) return;
    const size_t n=pixels.size()/4;
    const size_t wanted=std::min(n,std::max(size_t(1),size_t(std::lround(coverage*n))));
    std::vector<uint8_t> alpha(n);
    for(size_t i=0;i<n;++i) alpha[i]=pixels[i*4+3];
    std::nth_element(alpha.begin(),alpha.begin()+wanted-1,alpha.end(),std::greater<uint8_t>());
    const uint8_t edge=alpha[wanted-1];
    // Empty texels stay empty. Never amplify all pixels merely because
    // the target coverage cannot fit in this coarse level.
    if(!edge) return;
    const float scale=std::max(1.f,float(cutoff)/float(edge));
    for(size_t i=0;i<n;++i)
        pixels[i*4+3]=uint8_t(std::min(255.f,std::round(pixels[i*4+3]*scale)));
}
// Match the bilinear sampler, not just isolated texel centers. A mip
// whose brightest texel is exactly the cutoff loses nearly all coverage
// when that texel is blended with its transparent neighbors.
inline std::vector<float> filteredCutoutAlpha(const uint8_t* pixels,int w,int h) {
    std::vector<float> samples;
    samples.reserve(size_t(w)*h*4);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        for(int sy=0;sy<2;++sy) for(int sx=0;sx<2;++sx) {
            const float px=x+(sx?0.25f:-0.25f), py=y+(sy?0.25f:-0.25f);
            const int ix=int(std::floor(px)), iy=int(std::floor(py));
            const float tx=px-ix,ty=py-iy;
            auto a=[&](int xx,int yy) {
                xx=(xx+w)%w; yy=(yy+h)%h; // repeat sampler
                return float(pixels[(size_t(yy)*w+xx)*4+3]);
            };
            samples.push_back((a(ix,iy)*(1-tx)+a(ix+1,iy)*tx)*(1-ty)+
                              (a(ix,iy+1)*(1-tx)+a(ix+1,iy+1)*tx)*ty);
        }
    return samples;
}
inline float filteredCutoutCoverage(const uint8_t* pixels,int w,int h,float cutoff=22.95f) {
    const auto samples=filteredCutoutAlpha(pixels,w,h);
    return samples.empty()?0.f:float(std::count_if(samples.begin(),samples.end(),
        [=](float a){return a>=cutoff;}))/float(samples.size());
}
inline void preserveFilteredCutoutCoverage(std::vector<uint8_t>& pixels,int w,int h,float coverage) {
    if(coverage<=0.f || pixels.empty()) return;
    auto samples=filteredCutoutAlpha(pixels.data(),w,h);
    const size_t wanted=std::min(samples.size(),std::max(size_t(1),size_t(std::lround(coverage*samples.size()))));
    std::nth_element(samples.begin(),samples.begin()+wanted-1,samples.end(),std::greater<float>());
    const float edge=samples[wanted-1];
    if(edge<=0.f) return;
    // Half a byte of headroom avoids rounding back below the cutoff.
    const float scale=std::max(1.f,23.5f/edge);
    for(size_t i=3;i<pixels.size();i+=4)
        pixels[i]=uint8_t(std::min(255.f,std::round(pixels[i]*scale)));
}
struct CutoutMipChain {std::vector<uint8_t> pixels;uint32_t levels=0;};
inline CutoutMipChain cutoutMipChain(const uint8_t* rgba,int w,int h) {
    CutoutMipChain chain;
    std::vector<uint8_t> current(rgba,rgba+size_t(w)*h*4);
    const float coverage=filteredCutoutCoverage(rgba,w,h);
    for(;;) {
        chain.pixels.insert(chain.pixels.end(),current.begin(),current.end());++chain.levels;
        if(w==1 && h==1) break;
        const int nw=std::max(1,w/2),nh=std::max(1,h/2);
        std::vector<uint8_t> next(size_t(nw)*nh*4);
        for(int y=0;y<nh;++y) for(int x=0;x<nw;++x) {
            unsigned sum[4]={};unsigned count=0;
            for(int sy=y*h/nh;sy<(y+1)*h/nh;++sy)
                for(int sx=x*w/nw;sx<(x+1)*w/nw;++sx) {
                    for(int c=0;c<4;++c) sum[c]+=current[(size_t(sy)*w+sx)*4+c];++count;
                }
            for(int c=0;c<4;++c) next[(size_t(y)*nw+x)*4+c]=uint8_t((sum[c]+count/2)/count);
        }
        preserveFilteredCutoutCoverage(next,nw,nh,coverage);
        current.swap(next);w=nw;h=nh;
    }
    return chain;
}
}
