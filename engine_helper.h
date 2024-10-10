#pragma once
#define __STDC_LIB_EXT1__
#include <vector>
#include "renderer/renderer.h"

namespace engine {
namespace helper {

#pragma pack(push, 1)
struct DDS_PIXELFORMAT {
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwFourCC;
    uint32_t dwRGBBitCount;
    uint32_t dwRBitMask;
    uint32_t dwGBitMask;
    uint32_t dwBBitMask;
    uint32_t dwABitMask;
};

struct DDS_HEADER {
    uint32_t           dwNameTag;
    uint32_t           dwSize;
    uint32_t           dwFlags;
    uint32_t           dwHeight;
    uint32_t           dwWidth;
    uint32_t           dwPitchOrLinearSize;
    uint32_t           dwDepth;
    uint32_t           dwMipMapCount;
    uint32_t           dwReserved1[11];
    DDS_PIXELFORMAT    ddspf;
    uint32_t           dwCaps;
    uint32_t           dwCaps2;
    uint32_t           dwCaps3;
    uint32_t           dwCaps4;
    uint32_t           dwReserved2;
};
#pragma pack(pop)

void readFile(
    const std::string& file_name,
    uint64_t& file_size,
    std::vector<char>& buffer);

void loadDdsTexture(
    const glm::uvec3& size,
    const void* image_data,
    const std::string& input_filename);

void storeImageFileWithHeader(
    const std::string& file_name,
    const uint32_t& header_size,
    const void* header_data,
    const uint32_t& image_size,
    const void* image_data);

void createTextureImage(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& file_name,
    renderer::Format format,
    renderer::TextureInfo& texture,
    const std::source_location& src_location);

std::shared_ptr<renderer::BufferInfo> createUnifiedMeshBuffer(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::BufferUsageFlags& usage,
    const uint64_t& size,
    const void* data,
    const std::source_location& src_location);

void loadMtx2Texture(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cubemap_render_pass,
    const std::string& input_filename,
    renderer::TextureInfo& texture,
    const std::source_location& src_location);

void saveDdsTexture(
    const glm::uvec3& size,
    const void* image_data,
    const std::string& input_filename);

uint32_t inline popcnt(uint32_t x)
{
    x -= ((x >> 1) & 0x55555555);
    x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
    x = (((x >> 4) + x) & 0x0f0f0f0f);
    x += (x >> 8);
    x += (x >> 16);
    return x & 0x0000003f;
}
uint32_t inline clz(uint32_t x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return 32 - popcnt(x);
}

std::pair<std::string, int> exec(const char* cmd);

std::string compileGlobalShaders();

std::string initCompileGlobalShaders(
    const std::string& src_shader_path,
    const std::string& output_path,
    const std::string& compiler_path);

} // namespace helper
} // namespace engine
