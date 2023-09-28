#pragma once
#define __STDC_LIB_EXT1__
#include <vector>
#include "renderer/renderer.h"

namespace engine {
namespace helper {

std::vector<uint64_t> readFile(
    const std::string& file_name,
    uint64_t& file_size);

void writeImageFile(
    const std::string& file_name,
    const uint32_t& header_size,
    const void* header_data,
    const uint32_t& image_size,
    const void* image_data);

void createTextureImage(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& file_name,
    renderer::Format format,
    renderer::TextureInfo& texture);

std::shared_ptr<renderer::BufferInfo> createUnifiedMeshBuffer(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::BufferUsageFlags& usage,
    const uint64_t& size,
    const void* data);

void loadMtx2Texture(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cubemap_render_pass,
    const std::string& input_filename,
    renderer::TextureInfo& texture);

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
