#include "mesh_preview.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include "glm/gtc/matrix_transform.hpp"

#include "shaders/global_definition.glsl.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"   // createUnifiedMeshBuffer

namespace engine {
namespace {

constexpr renderer::Format kColorFormat = renderer::Format::R8G8B8A8_UNORM;
constexpr renderer::Format kDepthFormat = renderer::Format::D24_UNORM_S8_UINT;

static renderer::ShaderModuleList getShaderModules(
    std::shared_ptr<renderer::Device> device) {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "preview_mesh_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "preview_mesh_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    return shader_modules;
}

static auto createPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::vector<renderer::VertexInputBindingDescription>& binding_descs,
    const std::vector<renderer::VertexInputAttributeDescription>& attribute_descs) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    // Double-sided: preview geometry comes from arbitrary DCC exports whose
    // winding varies; the fragment shader flips normals toward the camera.
    renderer::RasterizationStateOverride rasterization_state_info;
    rasterization_state_info.override_double_sided = true;
    rasterization_state_info.double_sided          = true;

    renderer::PipelineRenderbufferFormats fb_format;
    fb_format.color_formats = { kColorFormat };
    fb_format.depth_format  = kDepthFormat;

    return device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        input_assembly,
        graphic_pipeline_info,
        getShaderModules(device),
        fb_format,
        rasterization_state_info,
        std::source_location::current());
}

// Alpha-blended pipeline for the reference grid quad (position-only vertex
// stream; analytic grid in the fragment shader).  graphic_pipeline_info
// should be the no-depth-write variant so the grid's transparent gaps don't
// occlude anything.
static auto createGridPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::vector<renderer::VertexInputBindingDescription>& binding_descs,
    const std::vector<renderer::VertexInputAttributeDescription>& attribute_descs) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    renderer::RasterizationStateOverride rasterization_state_info;
    rasterization_state_info.override_double_sided = true;
    rasterization_state_info.double_sided          = true;

    renderer::GraphicPipelineInfo blended_info = graphic_pipeline_info;
    blended_info.blend_state_info =
        std::make_shared<renderer::PipelineColorBlendStateCreateInfo>(
            renderer::helper::fillPipelineColorBlendStateCreateInfo({
                renderer::helper::fillPipelineColorBlendAttachmentState(
                    SET_FLAG_BIT(ColorComponent, ALL_BITS),
                    /*blend_enable=*/true,
                    renderer::BlendFactor::SRC_ALPHA,
                    renderer::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    renderer::BlendOp::ADD,
                    renderer::BlendFactor::ONE,
                    renderer::BlendFactor::ZERO,
                    renderer::BlendOp::ADD)
            }));

    renderer::PipelineRenderbufferFormats fb_format;
    fb_format.color_formats = { kColorFormat };
    fb_format.depth_format  = kDepthFormat;

    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device, "preview_grid_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device, "preview_grid_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

    return device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        input_assembly,
        blended_info,
        shader_modules,
        fb_format,
        rasterization_state_info,
        std::source_location::current());
}

// Sampled RGBA8 texture from CPU pixels (material base colour / fallback).
static std::shared_ptr<renderer::TextureInfo> createPixelTexture(
    const std::shared_ptr<renderer::Device>& device,
    int w, int h, const uint8_t* rgba) {
    auto info = std::make_shared<renderer::TextureInfo>();
    info->mip_levels = 1;
    renderer::Helper::create2DTextureImage(
        device, kColorFormat, w, h, rgba,
        info->image, info->memory, std::source_location::current());
    info->view = device->createImageView(
        info->image, renderer::ImageViewType::VIEW_2D, kColorFormat,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        std::source_location::current());
    return info;
}

} // anonymous namespace

namespace helper {

std::vector<renderer::VertexInputBindingDescription>   MeshPreview::s_binding_descs_;
std::vector<renderer::VertexInputAttributeDescription> MeshPreview::s_attrib_descs_;
std::shared_ptr<renderer::DescriptorSetLayout>         MeshPreview::s_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout>              MeshPreview::s_pipeline_layout_;
std::shared_ptr<renderer::Pipeline>                    MeshPreview::s_pipeline_;
std::shared_ptr<renderer::Pipeline>                    MeshPreview::s_grid_pipeline_;
std::shared_ptr<renderer::Sampler>                     MeshPreview::s_sampler_;
std::shared_ptr<renderer::DescriptorPool>              MeshPreview::s_pool_;
std::shared_ptr<renderer::TextureInfo>                 MeshPreview::s_color_;
std::shared_ptr<renderer::TextureInfo>                 MeshPreview::s_depth_;
std::shared_ptr<renderer::TextureInfo>                 MeshPreview::s_white_tex_;
std::shared_ptr<renderer::TextureInfo>                 MeshPreview::s_flatnrm_tex_;
ImTextureID                                            MeshPreview::s_im_id_ = 0;
bool                                                   MeshPreview::s_has_image_ = false;
bool                                                   MeshPreview::s_color_in_read_layout_ = false;
std::vector<std::shared_ptr<renderer::DescriptorSet>>  MeshPreview::s_active_sets_;
std::vector<MeshPreview::RecycleSet>                   MeshPreview::s_set_recycle_;
std::vector<MeshPreview::GpuSection>                   MeshPreview::s_sections_;
std::shared_ptr<renderer::BufferInfo>                  MeshPreview::s_pos_buf_;
std::shared_ptr<renderer::BufferInfo>                  MeshPreview::s_nrm_buf_;
std::shared_ptr<renderer::BufferInfo>                  MeshPreview::s_uv_buf_;
std::shared_ptr<renderer::BufferInfo>                  MeshPreview::s_idx_buf_;
std::shared_ptr<renderer::BufferInfo>                  MeshPreview::s_grid_buf_;
std::vector<std::shared_ptr<renderer::TextureInfo>>    MeshPreview::s_mat_texs_;
uint32_t                                               MeshPreview::s_index_count_ = 0;
std::vector<MeshPreview::DeadBuffer>                   MeshPreview::s_dead_;
std::vector<MeshPreview::DeadTexture>                  MeshPreview::s_dead_tex_;
glm::vec3 MeshPreview::s_center_     = glm::vec3(0.0f);
float     MeshPreview::s_radius_     = 1.0f;
bool      MeshPreview::s_has_uv_     = false;
float     MeshPreview::s_orbit_az_   = 35.0f;
float     MeshPreview::s_orbit_el_   = 20.0f;
float     MeshPreview::s_orbit_dist_ = 2.4f;
glm::vec3 MeshPreview::s_pan_        = glm::vec3(0.0f);
bool      MeshPreview::s_camera_dirty_ = false;

void MeshPreview::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::GraphicPipelineInfo& grid_pipeline_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& sampler) {

    s_sampler_ = sampler;

    // Binding 0 — vec3 position; binding 1 — vec3 normal; binding 2 — vec2 uv.
    {
        renderer::VertexInputBindingDescription bind0;
        bind0.binding    = 0;
        bind0.stride     = sizeof(glm::vec3);
        bind0.input_rate = renderer::VertexInputRate::VERTEX;
        s_binding_descs_.push_back(bind0);

        renderer::VertexInputAttributeDescription attr0;
        attr0.binding  = 0;
        attr0.location = VINPUT_POSITION;            // == 0
        attr0.format   = renderer::Format::R32G32B32_SFLOAT;
        attr0.offset   = 0;
        s_attrib_descs_.push_back(attr0);

        renderer::VertexInputBindingDescription bind1;
        bind1.binding    = 1;
        bind1.stride     = sizeof(glm::vec3);
        bind1.input_rate = renderer::VertexInputRate::VERTEX;
        s_binding_descs_.push_back(bind1);

        renderer::VertexInputAttributeDescription attr1;
        attr1.binding  = 1;
        attr1.location = VINPUT_NORMAL;              // == 2
        attr1.format   = renderer::Format::R32G32B32_SFLOAT;
        attr1.offset   = 0;
        s_attrib_descs_.push_back(attr1);

        renderer::VertexInputBindingDescription bind2;
        bind2.binding    = 2;
        bind2.stride     = sizeof(glm::vec2);
        bind2.input_rate = renderer::VertexInputRate::VERTEX;
        s_binding_descs_.push_back(bind2);

        renderer::VertexInputAttributeDescription attr2;
        attr2.binding  = 2;
        attr2.location = VINPUT_TEXCOORD0;           // == 1
        attr2.format   = renderer::Format::R32G32_SFLOAT;
        attr2.offset   = 0;
        s_attrib_descs_.push_back(attr2);
    }

    // Set 0: binding 0 = albedo, 1 = normal map, 2 = metallic-roughness —
    // full PBR material set per section (fallbacks: white / flat-normal /
    // white).
    {
        std::vector<renderer::DescriptorSetLayoutBinding> bindings(3);
        for (uint32_t b = 0; b < 3; ++b) {
            bindings[b] =
                renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                    b, SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
                    renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
        }
        s_desc_set_layout_ = device->createDescriptorSetLayout(bindings);
    }
    // Sets are allocated on demand (one per texture of the current preview)
    // and recycled through s_set_recycle_ — keep the persistent pool around.
    s_pool_ = descriptor_pool;

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::PreviewMeshParams);
    s_pipeline_layout_ = device->createPipelineLayout(
        { s_desc_set_layout_ },
        { push_const_range },
        std::source_location::current());

    s_pipeline_ = createPipeline(
        device, s_pipeline_layout_, graphic_pipeline_info,
        s_binding_descs_, s_attrib_descs_);

    // Grid pipeline: position-only vertex stream, alpha-blended, shares the
    // pipeline layout (its shaders use only the push constants).
    {
        std::vector<renderer::VertexInputBindingDescription> gbinds;
        std::vector<renderer::VertexInputAttributeDescription> gattrs;
        renderer::VertexInputBindingDescription gb0;
        gb0.binding    = 0;
        gb0.stride     = sizeof(glm::vec3);
        gb0.input_rate = renderer::VertexInputRate::VERTEX;
        gbinds.push_back(gb0);
        renderer::VertexInputAttributeDescription ga0;
        ga0.binding  = 0;
        ga0.location = VINPUT_POSITION;
        ga0.format   = renderer::Format::R32G32B32_SFLOAT;
        ga0.offset   = 0;
        gattrs.push_back(ga0);
        s_grid_pipeline_ = createGridPipeline(
            device, s_pipeline_layout_, grid_pipeline_info,
            gbinds, gattrs);
    }

    // 1x1 white fallback for untextured meshes.
    const uint8_t white[4] = { 255, 255, 255, 255 };
    s_white_tex_ = createPixelTexture(device, 1, 1, white);
    // 1x1 flat-normal fallback (+Z tangent space) for unmapped sections.
    const uint8_t flatn[4] = { 128, 128, 255, 255 };
    s_flatnrm_tex_ = createPixelTexture(device, 1, 1, flatn);

    // Persistent offscreen targets.
    s_color_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device,
        kColorFormat,
        glm::uvec2(kSize, kSize),
        1,
        *s_color_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, COLOR_ATTACHMENT_BIT),
        renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());
    s_color_in_read_layout_ = false;

    s_depth_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::createDepthResources(
        device,
        kDepthFormat,
        glm::uvec2(kSize, kSize),
        *s_depth_,
        std::source_location::current());
}

void MeshPreview::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    for (auto& d : s_dead_)     { if (d.buf) d.buf->destroy(device); }
    for (auto& d : s_dead_tex_) { if (d.tex) d.tex->destroy(device); }
    s_dead_.clear();
    s_dead_tex_.clear();
    if (s_pos_buf_)  { s_pos_buf_->destroy(device);  s_pos_buf_.reset(); }
    if (s_nrm_buf_)  { s_nrm_buf_->destroy(device);  s_nrm_buf_.reset(); }
    if (s_uv_buf_)   { s_uv_buf_->destroy(device);   s_uv_buf_.reset(); }
    if (s_idx_buf_)  { s_idx_buf_->destroy(device);  s_idx_buf_.reset(); }
    if (s_grid_buf_) { s_grid_buf_->destroy(device); s_grid_buf_.reset(); }
    for (auto& t : s_mat_texs_) { if (t) t->destroy(device); }
    s_mat_texs_.clear();
    s_active_sets_.clear();
    s_set_recycle_.clear();
    s_sections_.clear();
    s_pool_.reset();
    s_index_count_ = 0;
    if (s_white_tex_) { s_white_tex_->destroy(device); s_white_tex_.reset(); }
    if (s_flatnrm_tex_) {
        s_flatnrm_tex_->destroy(device);
        s_flatnrm_tex_.reset();
    }
    if (s_color_) { s_color_->destroy(device); s_color_.reset(); }
    if (s_depth_) { s_depth_->destroy(device); s_depth_.reset(); }
    if (s_pipeline_) { device->destroyPipeline(s_pipeline_); s_pipeline_.reset(); }
    if (s_grid_pipeline_) {
        device->destroyPipeline(s_grid_pipeline_);
        s_grid_pipeline_.reset();
    }
    if (s_pipeline_layout_) {
        device->destroyPipelineLayout(s_pipeline_layout_);
        s_pipeline_layout_.reset();
    }
    s_desc_set_layout_.reset();
    s_binding_descs_.clear();
    s_attrib_descs_.clear();
    s_im_id_ = 0;
    s_has_image_ = false;
    s_sampler_.reset();
}

void MeshPreview::reregisterImGui() {
    if (s_color_ && s_color_->view && s_sampler_) {
        s_im_id_ = renderer::Helper::addImTextureID(s_sampler_, s_color_->view);
    } else {
        s_im_id_ = 0;
    }
}

void MeshPreview::collectGarbage(
    const std::shared_ptr<renderer::Device>& device,
    uint64_t current_frame) {
    for (auto it = s_dead_.begin(); it != s_dead_.end();) {
        if (current_frame >= it->free_frame) {
            if (it->buf) it->buf->destroy(device);
            it = s_dead_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = s_dead_tex_.begin(); it != s_dead_tex_.end();) {
        if (current_frame >= it->free_frame) {
            if (it->tex) it->tex->destroy(device);
            it = s_dead_tex_.erase(it);
        } else {
            ++it;
        }
    }
}

void MeshPreview::render(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const MeshPreviewPayload& payload,
    uint64_t current_frame) {

    const auto& positions = payload.positions;
    const auto& normals   = payload.normals;
    const auto& indices   = payload.indices;
    if (!ready() || positions.empty() || indices.size() < 3 ||
        normals.size() != positions.size()) {
        return;
    }

    // Retire the previous mesh buffers / material texture — in-flight
    // frames may still reference them, so free a few frames from now.
    auto retire_buf = [&](std::shared_ptr<renderer::BufferInfo>& b) {
        if (b) {
            s_dead_.push_back({ current_frame + 4, b });
            b.reset();
        }
    };
    retire_buf(s_pos_buf_);
    retire_buf(s_nrm_buf_);
    retire_buf(s_uv_buf_);
    retire_buf(s_idx_buf_);
    for (auto& t : s_mat_texs_) {
        if (t) s_dead_tex_.push_back({ current_frame + 4, t });
    }
    s_mat_texs_.clear();
    // Retired descriptor sets become reusable once in-flight frames drain.
    for (auto& s : s_active_sets_) {
        s_set_recycle_.push_back({ current_frame + 4, s });
    }
    s_active_sets_.clear();
    s_sections_.clear();

    s_pos_buf_ = helper::createUnifiedMeshBuffer(
        device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        positions.size() * sizeof(positions[0]), positions.data(),
        std::source_location::current());
    s_nrm_buf_ = helper::createUnifiedMeshBuffer(
        device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        normals.size() * sizeof(normals[0]), normals.data(),
        std::source_location::current());
    // UV stream: zero-filled when the payload has none, so the pipeline's
    // binding 2 is always backed.
    const bool has_uv =
        payload.uvs.size() == positions.size() &&
        !payload.textures.empty();
    if (has_uv) {
        s_uv_buf_ = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            payload.uvs.size() * sizeof(payload.uvs[0]), payload.uvs.data(),
            std::source_location::current());
    } else {
        const std::vector<glm::vec2> zeros(positions.size(),
                                           glm::vec2(0.0f));
        s_uv_buf_ = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            zeros.size() * sizeof(zeros[0]), zeros.data(),
            std::source_location::current());
    }
    s_idx_buf_ = helper::createUnifiedMeshBuffer(
        device, SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
        indices.size() * sizeof(indices[0]), indices.data(),
        std::source_location::current());
    s_index_count_ = (uint32_t)indices.size();
    s_has_uv_ = has_uv;

    // ── Material textures + per-texture descriptor sets ──────────────────
    // One GPU texture per payload texture; one descriptor set per DISTINCT
    // bound texture (sections sharing a texture share its set).  Sets come
    // from the recycler when no in-flight frame can still reference them.
    auto acquire_set = [&]() -> std::shared_ptr<renderer::DescriptorSet> {
        for (auto it = s_set_recycle_.begin();
             it != s_set_recycle_.end(); ++it) {
            if (current_frame >= it->free_frame) {
                auto set = it->set;
                s_set_recycle_.erase(it);
                return set;
            }
        }
        return device->createDescriptorSets(
            s_pool_, s_desc_set_layout_, 1)[0];
    };
    // Binds the FULL PBR set: 0 = albedo (white fallback), 1 = normal
    // map (flat-normal fallback), 2 = metallic-roughness (white).
    auto bind_material_set =
        [&](const std::shared_ptr<renderer::TextureInfo>& albedo,
            const std::shared_ptr<renderer::TextureInfo>& nrm,
            const std::shared_ptr<renderer::TextureInfo>& mr) -> size_t {
        auto set = acquire_set();
        renderer::WriteDescriptorList writes;
        writes.reserve(3);
        renderer::Helper::addOneTexture(
            writes, set,
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 0,
            s_sampler_, (albedo ? albedo : s_white_tex_)->view,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        renderer::Helper::addOneTexture(
            writes, set,
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 1,
            s_sampler_, (nrm ? nrm : s_flatnrm_tex_)->view,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        renderer::Helper::addOneTexture(
            writes, set,
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER, 2,
            s_sampler_, (mr ? mr : s_white_tex_)->view,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        device->updateDescriptorSets(writes);
        s_active_sets_.push_back(set);
        return s_active_sets_.size() - 1;
    };

    for (const auto& pt : payload.textures) {
        if (!pt.rgba.empty() && pt.w > 0 && pt.h > 0) {
            s_mat_texs_.push_back(
                createPixelTexture(device, pt.w, pt.h, pt.rgba.data()));
        } else {
            s_mat_texs_.push_back(nullptr);
        }
    }
    auto tex_at = [&](int idx) -> std::shared_ptr<renderer::TextureInfo> {
        return (idx >= 0 && idx < (int)s_mat_texs_.size())
            ? s_mat_texs_[idx] : nullptr;
    };
    const size_t white_set =
        bind_material_set(nullptr, nullptr, nullptr);

    // ── Sections (synthesise one default section when none given) ────────
    // One descriptor set per section with any real texture; sections with
    // identical (albedo, nrm, mr) triples share a set via this cache.
    std::map<std::tuple<int, int, int>, size_t> combo_cache;
    if (payload.sections.empty()) {
        GpuSection sec;
        sec.first_index = 0;
        sec.index_count = s_index_count_;
        sec.set_index   = white_set;
        s_sections_.push_back(sec);
    } else {
        for (const auto& ps : payload.sections) {
            if (ps.index_count < 3) continue;
            GpuSection sec;
            sec.first_index = ps.first_index;
            sec.index_count = ps.index_count;
            sec.base_color  = ps.base_color;
            sec.metallic    = glm::clamp(ps.metallic,  0.0f, 1.0f);
            sec.roughness   = glm::clamp(ps.roughness, 0.0f, 1.0f);
            const auto albedo = has_uv ? tex_at(ps.tex_index) : nullptr;
            const auto nrm    = has_uv ? tex_at(ps.nrm_index) : nullptr;
            const auto mrtex  = has_uv ? tex_at(ps.mr_index)  : nullptr;
            sec.has_tex = albedo != nullptr;
            sec.has_nrm = nrm    != nullptr;
            sec.has_mr  = mrtex  != nullptr;
            if (!albedo && !nrm && !mrtex) {
                sec.set_index = white_set;
            } else {
                const auto key = std::make_tuple(
                    albedo ? ps.tex_index : -1,
                    nrm    ? ps.nrm_index : -1,
                    mrtex  ? ps.mr_index  : -1);
                auto it = combo_cache.find(key);
                sec.set_index = (it != combo_cache.end())
                    ? it->second
                    : (combo_cache[key] =
                           bind_material_set(albedo, nrm, mrtex));
            }
            s_sections_.push_back(sec);
        }
        if (s_sections_.empty()) {
            GpuSection sec;
            sec.first_index = 0;
            sec.index_count = s_index_count_;
            sec.set_index   = white_set;
            s_sections_.push_back(sec);
        }
    }

    // ── Framing for recordPass ───────────────────────────────────────────
    glm::vec3 bmin(1e30f), bmax(-1e30f);
    for (const auto& p : positions) {
        bmin = glm::min(bmin, p);
        bmax = glm::max(bmax, p);
    }
    s_center_ = (bmin + bmax) * 0.5f;
    s_radius_ = glm::max(glm::length(bmax - bmin) * 0.5f, 1e-4f);
    s_pan_    = glm::vec3(0.0f);   // fresh object — recentre the view

    // ── Reference-grid quad: origin at the OBJECT CENTRE, plane at the
    // object's base, sized to the grid's fade radius ─────────────────────
    {
        if (s_grid_buf_) {
            s_dead_.push_back({ current_frame + 4, s_grid_buf_ });
            s_grid_buf_.reset();
        }
        const float e = glm::max(s_radius_ * 3.0f, 4.0f);
        const float gy = bmin.y;
        const float cx = s_center_.x, cz = s_center_.z;
        const glm::vec3 quad[6] = {
            { cx - e, gy, cz - e }, { cx + e, gy, cz - e },
            { cx + e, gy, cz + e },
            { cx - e, gy, cz - e }, { cx + e, gy, cz + e },
            { cx - e, gy, cz + e },
        };
        s_grid_buf_ = helper::createUnifiedMeshBuffer(
            device, SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
            sizeof(quad), quad, std::source_location::current());
    }

    recordPass(cmd_buf);
}

void MeshPreview::orbit(float d_azimuth_deg, float d_elevation_deg) {
    s_orbit_az_ += d_azimuth_deg;
    if (s_orbit_az_ > 360.0f) s_orbit_az_ -= 360.0f;
    if (s_orbit_az_ < 0.0f)   s_orbit_az_ += 360.0f;
    s_orbit_el_ = glm::clamp(s_orbit_el_ + d_elevation_deg, -85.0f, 85.0f);
    s_camera_dirty_ = true;
}

void MeshPreview::zoom(float wheel_ticks) {
    s_orbit_dist_ =
        glm::clamp(s_orbit_dist_ * std::pow(0.9f, wheel_ticks),
                   1.1f, 10.0f);
    s_camera_dirty_ = true;
}

void MeshPreview::pan(float dx_px, float dy_px) {
    // Camera basis at the current orbit angles.
    const float az = glm::radians(s_orbit_az_);
    const float el = glm::radians(s_orbit_el_);
    const glm::vec3 dir(
        std::cos(el) * std::sin(az),
        std::sin(el),
        std::cos(el) * std::cos(az));
    const glm::vec3 fwd   = -dir;                       // camera → target
    const glm::vec3 right = glm::normalize(
        glm::cross(fwd, glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 up    = glm::cross(right, fwd);

    // Pixel → world scale at the framing distance (~vertical FOV span).
    const float world_per_px =
        (s_radius_ * s_orbit_dist_) * 1.5f / (float)kSize;

    // Dragging moves the OBJECT with the cursor: shift the orbit target the
    // opposite way (screen Y grows downward).
    s_pan_ += (-right * dx_px + up * dy_px) * world_per_px;
    s_camera_dirty_ = true;
}

void MeshPreview::rerenderIfNeeded(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {
    (void)device;
    if (!s_camera_dirty_ || !ready() || s_index_count_ == 0) return;
    recordPass(cmd_buf);
}

void MeshPreview::recordPass(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {
    if (!ready() || !s_pos_buf_ || s_index_count_ == 0) return;

    // ── Orbit camera around the (pan-shifted) object centre ──────────────
    const float az = glm::radians(s_orbit_az_);
    const float el = glm::radians(s_orbit_el_);
    const glm::vec3 dir(
        std::cos(el) * std::sin(az),
        std::sin(el),
        std::cos(el) * std::cos(az));
    const glm::vec3 target = s_center_ + s_pan_;
    const glm::vec3 eye = target + dir * (s_radius_ * s_orbit_dist_);

    const glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(
        glm::radians(40.0f), 1.0f,
        glm::max(s_radius_ * 0.02f, 0.001f), s_radius_ * 30.0f);
    proj[1][1] *= -1.0f;   // Vulkan clip-space Y

    glsl::PreviewMeshParams params{};
    params.view_proj         = proj * view;
    params.center_radius     = glm::vec4(s_center_, s_radius_);
    params.camera_pos        = glm::vec4(eye, 0.0f);
    params.base_color_factor = glm::vec4(1.0f);
    params.pbr_params        = glm::vec4(0.0f, 0.6f, 0.0f, 0.0f);

    // ── Transition colour target to attachment, render, back to sampled ──
    const renderer::ImageResourceInfo as_attachment = {
        renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
    const renderer::ImageResourceInfo as_sampled = {
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

    if (s_color_in_read_layout_) {
        cmd_buf->addImageBarrier(
            s_color_->image, as_sampled, as_attachment, 0, 1, 0, 1);
    }

    renderer::RenderingAttachmentInfo color_attach;
    color_attach.image_view   = s_color_->view;
    color_attach.image_layout = renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    color_attach.load_op      = renderer::AttachmentLoadOp::CLEAR;
    color_attach.store_op     = renderer::AttachmentStoreOp::STORE;
    // Neutral gray studio backdrop.
    color_attach.clear_value.color = { {0.45f, 0.45f, 0.46f, 1.0f} };

    renderer::RenderingAttachmentInfo depth_attach;
    depth_attach.image_view   = s_depth_->view;
    depth_attach.image_layout =
        renderer::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attach.load_op      = renderer::AttachmentLoadOp::CLEAR;
    depth_attach.store_op     = renderer::AttachmentStoreOp::STORE;
    depth_attach.clear_value.depth_stencil = { 1.0f, 0 };

    renderer::RenderingInfo ri = {};
    ri.render_area_offset  = { 0, 0 };
    ri.render_area_extent  = { kSize, kSize };
    ri.layer_count         = 1;
    ri.view_mask           = 0;
    ri.color_attachments   = { color_attach };
    ri.depth_attachments   = { depth_attach };
    ri.stencil_attachments = {};
    cmd_buf->beginDynamicRendering(ri);

    renderer::Viewport vp;
    vp.x = 0.0f; vp.y = 0.0f;
    vp.width  = (float)kSize;
    vp.height = (float)kSize;
    vp.min_depth = 0.0f; vp.max_depth = 1.0f;
    renderer::Scissor sc;
    sc.offset = glm::ivec2(0);
    sc.extent = glm::uvec2(kSize, kSize);
    std::vector<renderer::Viewport> viewports = { vp };
    std::vector<renderer::Scissor>  scissors  = { sc };
    cmd_buf->setViewports(viewports, 0, 1);
    cmd_buf->setScissors (scissors,  0, 1);

    // ── Mesh (opaque PBR, depth write) — one draw per material section ──
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, s_pipeline_);
    {
        std::vector<std::shared_ptr<renderer::Buffer>> buffers = {
            s_pos_buf_->buffer,
            s_nrm_buf_->buffer,
            s_uv_buf_->buffer,
        };
        std::vector<uint64_t> offsets = { 0, 0, 0 };
        cmd_buf->bindVertexBuffers(0, buffers, offsets);
    }
    cmd_buf->bindIndexBuffer(
        s_idx_buf_->buffer, 0, renderer::IndexType::UINT32);

    for (const auto& sec : s_sections_) {
        if (sec.set_index >= s_active_sets_.size()) continue;
        renderer::DescriptorSetList desc_sets = {
            s_active_sets_[sec.set_index] };
        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::GRAPHICS,
            s_pipeline_layout_, desc_sets);
        params.base_color_factor = sec.base_color;
        // pbr_params: x=metallic, y=roughness, z=has albedo map,
        // w = map flags (bit0 normal map, bit1 metallic-roughness map).
        params.pbr_params = glm::vec4(
            sec.metallic, sec.roughness,
            sec.has_tex ? 1.0f : 0.0f,
            (sec.has_nrm ? 1.0f : 0.0f) + (sec.has_mr ? 2.0f : 0.0f));
        cmd_buf->pushConstants(
            SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
            s_pipeline_layout_,
            &params,
            sizeof(params));
        cmd_buf->drawIndexed(sec.index_count, 1, sec.first_index);
    }

    // ── Reference grid (alpha-blended, depth-tested, no depth write) ─────
    if (s_grid_pipeline_ && s_grid_buf_) {
        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::GRAPHICS, s_grid_pipeline_);
        std::vector<std::shared_ptr<renderer::Buffer>> gbuffers = {
            s_grid_buf_->buffer,
        };
        std::vector<uint64_t> goffsets = { 0 };
        cmd_buf->bindVertexBuffers(0, gbuffers, goffsets);
        cmd_buf->draw(6);
    }

    cmd_buf->endDynamicRendering();

    cmd_buf->addImageBarrier(
        s_color_->image, as_attachment, as_sampled, 0, 1, 0, 1);
    s_color_in_read_layout_ = true;
    s_camera_dirty_ = false;

    // Register with ImGui on first use.
    if (!s_im_id_) reregisterImGui();
    s_has_image_ = true;
}

} // namespace helper
} // namespace engine
