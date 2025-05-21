/*
// Data Format Descriptor 
UInt32 dfdTotalSize
continue
dfDes
until dfdByteLength read

// Key/Value Data 
continue
UInt32   keyAndValueByteLength
Byte     keyAndValue[keyAndValueByteLength]
align(4) valuePadding

until kvdByteLength read
if (sgdByteLength > 0)
align(8) sgdPadding

// Supercompression Global Data 
Byte supercompressionGlobalData[sgdByteLength]

// Mip Level Array
for each mip_level in levelCount
Byte     levelImages[bytesOfLevelImages]
end*/

#pragma once

#include "renderer/renderer.h"

namespace engine {

struct Mtx2HeaderBlock {
    uint8_t             identifier[12];
    engine::renderer::Format    format;
    uint32_t            type_size;
    uint32_t            pixel_width;
    uint32_t            pixel_height;
    uint32_t            pixel_depth;
    uint32_t            layer_count;
    uint32_t            face_count;
    uint32_t            level_count;
    uint32_t            supercompression_scheme;
};

struct Mtx2IndexBlock {
    uint32_t            dfd_byte_offset;
    uint32_t            dfd_byte_length;
    uint32_t            kvd_byte_offset;
    uint32_t            kvd_byte_length;
    uint64_t            sgd_byte_offset;
    uint64_t            sgd_byte_length;
};

struct Mtx2LevelIndexBlock {
    uint64_t            byte_offset;
    uint64_t            byte_length;
    uint64_t            uncompressed_byte_length;
};

} //namespace engine