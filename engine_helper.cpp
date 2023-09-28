#include <fstream>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <array>
#include <memory>
#include <vector>
#include <string>

#include "engine_helper.h"
#include "renderer/renderer.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
#include "tiny_mtx2.h"

namespace engine {
namespace helper {
static std::string s_src_shader_path;
static std::string s_output_path;
static std::string s_compiler_path;

std::vector<uint64_t> readFile(const std::string& file_name, uint64_t& file_size) {
    std::ifstream file(file_name, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::string error_message = std::string("failed to open file! :") + file_name;
        throw std::runtime_error(error_message);
    }

    file_size = (uint64_t)file.tellg();
    std::vector<uint64_t> buffer((file_size + sizeof(uint64_t) - 1) / sizeof(uint64_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    file.close();
    return buffer;
}

void writeImageFile(
    const std::string& file_name,
    const uint32_t& header_size,
    const void* header_data,
    const uint32_t& image_size,
    const void* image_data) {
    std::ofstream file(file_name, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::string error_message = std::string("failed to open file! :") + file_name;
        throw std::runtime_error(error_message);
    }

    file.write(reinterpret_cast<const char*>(header_data), header_size);
    file.write(reinterpret_cast<const char*>(image_data), image_size);

    file.close();
}

void createTextureImage(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& file_name,
    renderer::Format format,
    renderer::TextureInfo& texture) {
    int tex_width, tex_height, tex_channels;
    void* void_pixels = nullptr;
    if (format == engine::renderer::Format::R16_UNORM) {
        stbi_us* pixels =
            stbi_load_16(
                file_name.c_str(),
                &tex_width,
                &tex_height,
                &tex_channels,
                STBI_grey);
        void_pixels = pixels;
    }
    else {
        stbi_uc* pixels =
            stbi_load(
                file_name.c_str(),
                &tex_width,
                &tex_height,
                &tex_channels,
                STBI_rgb_alpha);
        void_pixels = pixels;
    }

    if (!void_pixels) {
        throw std::runtime_error("failed to load texture image!");
    }
    renderer::Helper::create2DTextureImage(
        device,
        format,
        tex_width,
        tex_height,
        tex_channels,
        void_pixels,
        texture.image,
        texture.memory);

    texture.size = { tex_width, tex_height, 1.0f };

    stbi_image_free(void_pixels);

    texture.view = device->createImageView(
        texture.image,
        renderer::ImageViewType::VIEW_2D,
        format,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT));
}

std::shared_ptr<renderer::BufferInfo> createUnifiedMeshBuffer(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::BufferUsageFlags& usage,
    const uint64_t& size,
    const void* data) {
    auto v_buffer = std::make_shared<renderer::BufferInfo>();
    renderer::Helper::createBuffer(
        device,
        usage,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        v_buffer->buffer,
        v_buffer->memory,
        size,
        data);

    return v_buffer;
}

void loadMtx2Texture(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cubemap_render_pass,
    const std::string& input_filename,
    renderer::TextureInfo& texture) {
    uint64_t buffer_size;
    auto mtx2_data = engine::helper::readFile(input_filename, buffer_size);
    auto src_data = (char*)mtx2_data.data();

    // header block
    Mtx2HeaderBlock* header_block = reinterpret_cast<Mtx2HeaderBlock*>(src_data);
    src_data += sizeof(Mtx2HeaderBlock);

    assert(header_block->format == renderer::Format::R16G16B16A16_SFLOAT);

    // index block
    Mtx2IndexBlock* index_block = reinterpret_cast<Mtx2IndexBlock*>(src_data);
    src_data += sizeof(Mtx2IndexBlock);

    uint32_t width = header_block->pixel_width;
    uint32_t height = header_block->pixel_height;
    // level index block.
    uint32_t num_level_blocks = std::max(1u, header_block->level_count);
    std::vector<renderer::BufferImageCopyInfo> copy_regions(num_level_blocks);
    for (uint32_t i_level = 0; i_level < num_level_blocks; i_level++) {
        Mtx2LevelIndexBlock* level_block = reinterpret_cast<Mtx2LevelIndexBlock*>(src_data);

        auto& region = copy_regions[i_level];
        region.buffer_offset = level_block->byte_offset;
        region.buffer_row_length = 0;
        region.buffer_image_height = 0;

        region.image_subresource.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        region.image_subresource.mip_level = i_level;
        region.image_subresource.base_array_layer = 0;
        region.image_subresource.layer_count = 6;

        region.image_offset = glm::ivec3(0, 0, 0);
        region.image_extent = glm::uvec3(width, height, 1);
        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);

        src_data += sizeof(Mtx2LevelIndexBlock);
    }

    char* dfd_data_start = (char*)mtx2_data.data() + index_block->dfd_byte_offset;
    uint32_t dfd_total_size = *reinterpret_cast<uint32_t*>(dfd_data_start);
    src_data += sizeof(uint32_t);

    char* kvd_data_start = (char*)mtx2_data.data() + index_block->kvd_byte_offset;
    uint32_t key_value_byte_length = *reinterpret_cast<uint32_t*>(kvd_data_start);
    uint8_t* key_value = reinterpret_cast<uint8_t*>(kvd_data_start + 4);
    for (uint32_t i = 0; i < key_value_byte_length; i++) {
        auto result = key_value[i];
        int hit = 1;
    }

    char* sgd_data_start = nullptr;
    if (index_block->sgd_byte_length > 0) {
        sgd_data_start = (char*)mtx2_data.data() + index_block->sgd_byte_offset;
    }

    renderer::Helper::createCubemapTexture(
        device,
        cubemap_render_pass,
        header_block->pixel_width,
        header_block->pixel_height,
        num_level_blocks,
        header_block->format,
        copy_regions,
        texture,
        buffer_size,
        mtx2_data.data());
}

void saveDdsTexture(
    const glm::uvec3& size,
    const void* image_data,
    const std::string& input_filename) {

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
        DDS_PIXELFORMAT ddspf;
        uint32_t           dwCaps;
        uint32_t           dwCaps2;
        uint32_t           dwCaps3;
        uint32_t           dwCaps4;
        uint32_t           dwReserved2;
    };

    DDS_HEADER dds_header;
    dds_header.ddspf.dwSize = sizeof(DDS_PIXELFORMAT);
    dds_header.ddspf.dwFlags = 0x41;
    dds_header.ddspf.dwFourCC = 0;
    dds_header.ddspf.dwRGBBitCount = 32;
    dds_header.ddspf.dwRBitMask = 0x000000ff;
    dds_header.ddspf.dwGBitMask = 0x0000ff00;
    dds_header.ddspf.dwBBitMask = 0x00ff0000;
    dds_header.ddspf.dwABitMask = 0xff000000;
    dds_header.dwNameTag = 0x20534444;
    dds_header.dwSize = 124;
    dds_header.dwFlags = 0x100f | (size.z > 1 ? 0x800000 : 0x00);
    dds_header.dwHeight = size.y;
    dds_header.dwWidth = size.x;
    dds_header.dwPitchOrLinearSize = size.x * 4;
    dds_header.dwDepth = size.z;
    dds_header.dwMipMapCount = 1;
    dds_header.dwCaps = 0x1000; // texture.
    dds_header.dwCaps2 = size.z > 1 ? 0x200000 : 0x00; // volume texture.

    writeImageFile(
        input_filename,
        sizeof(DDS_HEADER),
        &dds_header,
        dds_header.dwPitchOrLinearSize* size.y* size.z,
        image_data);
}

std::pair<std::string, int> exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    int return_code = -1;
    auto pclose_wrapper = [&return_code](FILE* cmd) { return_code = _pclose(cmd); };
    { // scope is important, have to make sure the ptr goes out of scope first
        const std::unique_ptr<FILE, decltype(pclose_wrapper)> pipe(_popen(cmd, "rt"), pclose_wrapper);
        if (pipe) {
            while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
                result += buffer.data();
            }
        }
    }
    return make_pair(result, return_code);
}

static void analyzeCommandLine(
    const std::string& line,
    std::string& input_name,
    std::string& output_name,
    std::string& params_str) {

    auto o0 = line.rfind(' ');
    auto o1 = line.rfind('\t');
    if (o1 != std::string::npos) {
        if (o0 != std::string::npos) {
            o0 = std::max(o0, o1);
        }
    }
    auto e0 = line.rfind('\r');
    auto e1 = line.rfind('\n');
    if (e1 != std::string::npos) {
        if (e0 != std::string::npos) {
            e0 = std::max(e0, e1);
        }
    }

    output_name = "\\" + line.substr(o0 + 1, e0 - (o0 + 1));

    auto i0 = line.find(' ');
    auto i1 = line.find('\t');
    if (i0 != std::string::npos && i1 != std::string::npos) {
        i0 = std::max(i0, i1);
    }
    input_name = "\\" + line.substr(0, i0);

    params_str = line.substr(i0, o0 - i0);
}

std::string compileGlobalShaders() {
    std::string error_strings;
    auto input_folder_exist = std::filesystem::exists(s_src_shader_path);
    auto output_folder_exist = std::filesystem::exists(s_output_path);
    if (input_folder_exist) {
        if (!output_folder_exist) {
            output_folder_exist = std::filesystem::create_directory(s_output_path);
        }
        if (output_folder_exist) {
            std::fstream fs;
            fs.open(s_src_shader_path + "\\shaders-compile.cfg", std::ios::in | std::ios::binary | std::ios::ate);

            std::string buffer;
            if (fs.is_open()) {
                fs.seekg(0, fs.end);
                auto size = fs.tellg();
                fs.seekg(0, fs.beg);
                buffer.resize(size);
                fs.read(buffer.data(), size);
            }
            fs.close();

            std::istringstream buf_str(buffer);
            for (std::string line; std::getline(buf_str, line); ) {
                std::string input_name, output_name, params_str;
                analyzeCommandLine(line, input_name, output_name, params_str);
                input_name = s_src_shader_path + input_name;
                output_name = s_output_path + output_name;

                struct stat input_attrib, output_attrib;
                stat(input_name.c_str(), &input_attrib);
                stat(output_name.c_str(), &output_attrib);

                if (true/*input_attrib.st_mtime > output_attrib.st_mtime*/) {
                    auto cmd_str = s_compiler_path + "\\glslc.exe " + input_name + " " + params_str + " " + output_name;

                    auto result = exec((cmd_str + " 2>&1").c_str());

                    if (result.second != 0) {
                        error_strings += cmd_str + "\n" + result.first + "\n";
                    }
                }
            }
        }
    }

    return error_strings;
}

static std::unordered_map<std::string, bool> s_folder_check;
void checkAndAddFolder(const std::string& path_name) {
    if (!std::filesystem::exists(path_name)) {
        bool check_path = std::filesystem::create_directory(path_name);
        assert(check_path);
    }
}

size_t findFolderSplit(const std::string& path_name, const size_t& pos) {
    auto pos_1 = path_name.find("/", pos);
    auto pos_2 = path_name.find("\\", pos);
    
    return std::min(pos_1, pos_2);
}

void analyzeAndSplitFilePath(const std::string& path_name) {
    size_t split_pos = std::string::npos;
    size_t offset = 0;
    std::vector<std::string> folder_list;
    while((split_pos = findFolderSplit(path_name, offset)) != std::string::npos) {
        folder_list.push_back(path_name.substr(0, split_pos));
        offset = split_pos + 1;
    };

    for (const auto& folder : folder_list) {
        if (s_folder_check.find(folder) == s_folder_check.end()) {
            checkAndAddFolder(folder);
            s_folder_check[folder] = true;
        }
    }
}

std::string  initCompileGlobalShaders(
    const std::string& src_shader_path,
    const std::string& output_path,
    const std::string& compiler_path) {
    s_src_shader_path = src_shader_path;
    s_output_path = output_path;
    s_compiler_path = compiler_path;

    std::string error_strings;
    const auto shader_compiler_str = s_compiler_path + "\\glslc.exe ";

    std::fstream fs;
    fs.open(s_src_shader_path + "\\shaders-compile.cfg", std::ios::in | std::ios::binary | std::ios::ate);

    std::string buffer;
    if (fs.is_open()) {
        fs.seekg(0, fs.end);
        auto size = fs.tellg();
        fs.seekg(0, fs.beg);
        buffer.resize(size);
        fs.read(buffer.data(), size);
    }
    fs.close();

    std::istringstream buf_str(buffer);
    for (std::string line; std::getline(buf_str, line); ) {
        std::string input_name, output_name, params_str;
        analyzeCommandLine(line, input_name, output_name, params_str);
        input_name = s_src_shader_path + input_name;
        output_name = s_output_path + output_name;

        struct stat input_attrib, output_attrib;
        stat(input_name.c_str(), &input_attrib);
        auto exist = stat(output_name.c_str(), &output_attrib);

        analyzeAndSplitFilePath(input_name);
        analyzeAndSplitFilePath(output_name);

        bool shader_tobe_rebuilt = false;
        if (std::filesystem::exists(input_name)) {
            if (!std::filesystem::exists(output_name)) {
                shader_tobe_rebuilt = true;
            }
            else if (
                std::filesystem::last_write_time(input_name) >
                std::filesystem::last_write_time(output_name)) {
                shader_tobe_rebuilt = true;
            }
        }

        if (shader_tobe_rebuilt) {
            auto cmd_str = shader_compiler_str + input_name + " " + params_str + " " + output_name;
            auto result = exec((cmd_str + " 2>&1").c_str());

            if (result.second != 0) {
                error_strings += cmd_str + "\n" + result.first + "\n";
            }
        }
    }

    return error_strings;
}

} // namespace helper
} // namespace engine
