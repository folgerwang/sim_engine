#include <vector>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <source_location>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <ctime>
#include <queue>          // geodesic weight de-bleed (Dijkstra)
#include <chrono>         // monotonic clock for the one-time viewport intro fade
#include <functional>     // std::greater
#include <unordered_map>
#include <cstring>        // std::memcpy (exact-position vertex weld)

// stbi_load / stbi_image_free are already compiled into the project via
// tinygltf (STB_IMAGE_IMPLEMENTATION lives in engine_helper.cpp).
// Pull in just the declarations so we can scan the background PNG.
extern "C" {
    unsigned char* stbi_load(const char*, int*, int*, int*, int);
    void stbi_image_free(void*);
    // STB_IMAGE_WRITE_IMPLEMENTATION is compiled in engine_helper.cpp, so the
    // symbol is linked — declaring it here lets us write sidecar thumbnails.
    int stbi_write_png(const char*, int, int, int, const void*, int);
}
#ifndef STBI_rgb_alpha
#define STBI_rgb_alpha 4
#endif

#include <glm/gtc/matrix_transform.hpp>  // selection helpers compose the
#include <glm/gtc/quaternion.hpp>        // wrapper instance TRS

#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder API for the editor default layout
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "game_object/drawable_object.h"  // Details panel reads object transform
#include "editor_log.h"                    // Output Log panel

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "renderer/vulkan/vk_device.h"   // VulkanDevice/VulkanPhysicalDevice for VRAM query
#include "helper/engine_helper.h"
#include "helper/vram_cuda.h"             // device-wide VRAM (counts ML/CUDA + other processes)
#include "helper/model_inspect.h"         // sub-object names for content assets
#include "helper/mesh_preview.h"          // GPU offscreen Debug Display preview
#include "ecs/animation_system.h"         // clip sampling for the animated preview
#include "audio/audio_engine.h"           // Content Browser audio preview
#include "audio/tts_engine.h"             // Audio menu voice picker
#include "scene/native_file_dialog.h"     // reference-image picker (FLUX.2)

#include "menu.h"
#include "plugins/plugin_manager.h"
#include "plugins/auto_rig/simple_rasterizer.h"  // model-thumbnail mesh loader + CPU rasterizer
#include "plugins/auto_rig/auto_rig_plugin.h"     // Debug Display "skin layer only" source
#include "game_object/mesh_load_task_manager.h"
#include "scene_rendering/ssao.h"
#include "scene_rendering/cluster_renderer.h"
#include "scene_rendering/virtual_texture.h"
#include "helper/cluster_mesh.h"

namespace er = engine::renderer;

namespace {
    static void check_vk_result(VkResult err)
    {
        if (err == 0)
            return;
        fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
        if (err < 0)
            abort();
    }

    // ------------------------------------------------------------------
    // VRAM profiler helpers.  Use the driver-reported VK_EXT_memory_budget
    // numbers, which account for EVERY Vulkan allocation made by this process
    // (engine buffers/images, render targets, swapchain, and ImGui's own
    // vertex/index buffers + font atlas) — no manual bookkeeping needed.
    // ------------------------------------------------------------------
    struct VramQuery { double used_mb, budget_mb, total_mb; bool valid; };

    static VkPhysicalDevice getVkPhysicalDevice(
            const std::shared_ptr<engine::renderer::Device>& dev) {
        // The engine only ever instantiates the Vulkan backend, and the
        // PhysicalDevice base is non-polymorphic (no vtable), so use
        // static_cast rather than dynamic_cast.
        if (!dev) return VK_NULL_HANDLE;
        auto* vkdev = static_cast<engine::renderer::vk::VulkanDevice*>(dev.get());
        const auto& pd = vkdev->getPhysicalDevice();
        if (!pd) return VK_NULL_HANDLE;
        auto* vkpd =
            static_cast<engine::renderer::vk::VulkanPhysicalDevice*>(pd.get());
        return vkpd->get();
    }

    static VramQuery queryVram(VkPhysicalDevice phys) {
        VramQuery q{0.0, 0.0, 0.0, false};
        if (phys == VK_NULL_HANDLE) return q;

        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp{};
        mp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp.pNext = &budget;
        // Core in Vulkan 1.1+ (engine requests 1.3), statically linked.
        vkGetPhysicalDeviceMemoryProperties2(phys, &mp);

        const VkPhysicalDeviceMemoryProperties& props = mp.memoryProperties;
        double used = 0.0, bud = 0.0, total = 0.0;
        bool any_budget = false;
        for (uint32_t i = 0; i < props.memoryHeapCount; ++i) {
            if (!(props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
                continue;                                // VRAM heaps only
            total += static_cast<double>(props.memoryHeaps[i].size);
            // heapUsage/heapBudget are only meaningful when the extension was
            // actually enabled at device creation (budget reads back non-zero).
            if (budget.heapBudget[i] != 0) {
                used += static_cast<double>(budget.heapUsage[i]);
                bud  += static_cast<double>(budget.heapBudget[i]);
                any_budget = true;
            }
        }
        const double MB = 1024.0 * 1024.0;
        q.total_mb = total / MB;
        if (any_budget) {
            q.used_mb   = used / MB;
            q.budget_mb = bud  / MB;
            q.valid     = true;
        } else {
            // Extension unavailable — show capacity only (no usage readback).
            q.budget_mb = q.total_mb;
            q.valid     = (total > 0.0);
        }
        return q;
    }

    // ------------------------------------------------------------------
    // Fantasy aesthetic — ported from realworld/design/fantasy_menu.html.
    // Deep indigo panels, muted gold accents, low-alpha borders. Applied
    // once after ImGui init so every widget (menu bar, popups, sliders,
    // plugin panels) picks up the look without per-call styling.
    // ------------------------------------------------------------------
    static void applyFantasyStyle() {
        ImGuiStyle& s = ImGui::GetStyle();

        const ImVec4 gold        = ImVec4(0.83f, 0.69f, 0.35f, 1.00f);
        const ImVec4 gold_dim    = ImVec4(0.55f, 0.45f, 0.22f, 0.80f);
        const ImVec4 gold_bright = ImVec4(0.95f, 0.86f, 0.60f, 1.00f);
        const ImVec4 cream       = ImVec4(0.93f, 0.88f, 0.72f, 1.00f);

        s.Colors[ImGuiCol_Text]              = cream;
        s.Colors[ImGuiCol_TextDisabled]      = ImVec4(0.55f, 0.52f, 0.42f, 1.00f);
        s.Colors[ImGuiCol_WindowBg]          = ImVec4(0.05f, 0.06f, 0.12f, 0.92f);
        s.Colors[ImGuiCol_ChildBg]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        // Popups (menu dropdowns, context menus): same raised indigo as
        // the tabs so menu items read on the same solid background.
        s.Colors[ImGuiCol_PopupBg]           = ImVec4(0.22f, 0.22f, 0.34f, 0.98f);
        s.Colors[ImGuiCol_Border]            = ImVec4(0.55f, 0.45f, 0.22f, 0.40f);
        s.Colors[ImGuiCol_BorderShadow]      = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        s.Colors[ImGuiCol_FrameBg]           = ImVec4(0.10f, 0.10f, 0.18f, 0.85f);
        s.Colors[ImGuiCol_FrameBgHovered]    = ImVec4(0.18f, 0.16f, 0.28f, 1.00f);
        s.Colors[ImGuiCol_FrameBgActive]     = ImVec4(0.25f, 0.22f, 0.38f, 1.00f);
        s.Colors[ImGuiCol_TitleBg]           = ImVec4(0.04f, 0.05f, 0.10f, 1.00f);
        s.Colors[ImGuiCol_TitleBgActive]     = ImVec4(0.08f, 0.08f, 0.15f, 1.00f);
        s.Colors[ImGuiCol_TitleBgCollapsed]  = ImVec4(0.04f, 0.05f, 0.10f, 0.75f);
        // Menu bar: matches the tab background (one consistent "chrome"
        // colour across the top bar and the panel tab strips).
        s.Colors[ImGuiCol_MenuBarBg]         = ImVec4(0.22f, 0.22f, 0.34f, 1.00f);
        s.Colors[ImGuiCol_ScrollbarBg]       = ImVec4(0.02f, 0.02f, 0.06f, 0.60f);
        s.Colors[ImGuiCol_ScrollbarGrab]     = ImVec4(0.20f, 0.18f, 0.30f, 1.00f);
        s.Colors[ImGuiCol_ScrollbarGrabHovered] = gold_dim;
        s.Colors[ImGuiCol_ScrollbarGrabActive]  = gold;
        s.Colors[ImGuiCol_CheckMark]         = gold_bright;
        s.Colors[ImGuiCol_SliderGrab]        = gold_dim;
        s.Colors[ImGuiCol_SliderGrabActive]  = gold_bright;
        s.Colors[ImGuiCol_Button]            = ImVec4(0.10f, 0.10f, 0.18f, 0.85f);
        s.Colors[ImGuiCol_ButtonHovered]     = gold_dim;
        s.Colors[ImGuiCol_ButtonActive]      = gold;
        s.Colors[ImGuiCol_Header]            = ImVec4(0.10f, 0.10f, 0.18f, 0.85f);
        s.Colors[ImGuiCol_HeaderHovered]     = gold_dim;
        s.Colors[ImGuiCol_HeaderActive]      = gold;
        s.Colors[ImGuiCol_Separator]         = ImVec4(0.35f, 0.30f, 0.18f, 0.60f);
        s.Colors[ImGuiCol_SeparatorHovered]  = gold_dim;
        s.Colors[ImGuiCol_SeparatorActive]   = gold;
        // Tabs: SOLID, clearly visible backgrounds — the previous values
        // were so close to the window bg that tab strips read as bare
        // text floating over the viewport.  Selected tab = muted gold
        // block; UNSELECTED tabs = noticeably brighter raised indigo so
        // they separate from the panel/menu background at a glance.
        s.Colors[ImGuiCol_Tab]               = ImVec4(0.22f, 0.22f, 0.34f, 1.00f);
        s.Colors[ImGuiCol_TabHovered]        = ImVec4(0.42f, 0.35f, 0.18f, 1.00f);
        s.Colors[ImGuiCol_TabActive]         = ImVec4(0.33f, 0.27f, 0.14f, 1.00f);
        s.Colors[ImGuiCol_TabUnfocused]      = ImVec4(0.19f, 0.19f, 0.30f, 1.00f);
        s.Colors[ImGuiCol_TabUnfocusedActive]= ImVec4(0.26f, 0.22f, 0.15f, 1.00f);
        s.Colors[ImGuiCol_ResizeGrip]        = ImVec4(0.35f, 0.30f, 0.18f, 0.40f);
        s.Colors[ImGuiCol_ResizeGripHovered] = gold_dim;
        s.Colors[ImGuiCol_ResizeGripActive]  = gold;

        s.WindowRounding    = 2.0f;
        s.FrameRounding     = 2.0f;
        s.GrabRounding      = 2.0f;
        s.PopupRounding     = 2.0f;
        s.ScrollbarRounding = 2.0f;
        s.TabRounding       = 2.0f;
        s.WindowBorderSize  = 1.0f;
        s.FrameBorderSize   = 0.0f;
        s.PopupBorderSize   = 1.0f;
        s.WindowPadding     = ImVec2(12.0f, 10.0f);
        s.ItemSpacing       = ImVec2(8.0f, 6.0f);
        s.IndentSpacing     = 18.0f;
    }

    // ------------------------------------------------------------------
    // Dual counter-rotating rune loader ring. Mirrors the CSS animation
    // in the HTML mockup: an outer dashed ring slowly clockwise, an
    // inner ring counter-rotating faster, plus a bright sweep arc and a
    // pulsing central dot. Time-driven so the cadence is stable across
    // framerates.
    // ------------------------------------------------------------------
    static void drawRuneLoader(
        ImDrawList* dl,
        const ImVec2& c,
        float radius,
        float elapsed) {

        const float kTwoPi = 6.28318530718f;
        const ImU32 gold       = IM_COL32(212, 175,  90, 255);
        const ImU32 gold_dim   = IM_COL32(180, 140,  70, 160);
        const ImU32 cream      = IM_COL32(242, 220, 170, 230);
        const ImU32 pulse_col  = IM_COL32(230, 200, 130, 200);

        // Outer + inner base rings.
        dl->AddCircle(c, radius,          gold_dim, 64, 1.4f);
        dl->AddCircle(c, radius * 0.66f,  gold_dim, 48, 1.0f);

        // Outer ring ticks, rotating clockwise.
        const float outer_angle = elapsed * 0.6f;
        const int   n_outer     = 12;
        for (int i = 0; i < n_outer; ++i) {
            const float f = float(i) / float(n_outer);
            const float a = outer_angle + f * kTwoPi;
            const float len = (i % 3 == 0) ? 6.0f : 3.0f;
            const ImVec2 p0 = { c.x + std::cos(a) * radius,
                                c.y + std::sin(a) * radius };
            const ImVec2 p1 = { c.x + std::cos(a) * (radius + len),
                                c.y + std::sin(a) * (radius + len) };
            dl->AddLine(p0, p1, gold, 1.4f);
        }

        // Inner ring ticks, counter-rotating and faster.
        const float inner_angle = -elapsed * 0.95f;
        const int   n_inner     = 8;
        const float r_inner     = radius * 0.66f;
        for (int i = 0; i < n_inner; ++i) {
            const float f = float(i) / float(n_inner);
            const float a = inner_angle + f * kTwoPi;
            const float len = 4.0f;
            const ImVec2 p0 = { c.x + std::cos(a) * r_inner,
                                c.y + std::sin(a) * r_inner };
            const ImVec2 p1 = { c.x + std::cos(a) * (r_inner - len),
                                c.y + std::sin(a) * (r_inner - len) };
            dl->AddLine(p0, p1, gold, 1.2f);
        }

        // Bright ~90 degree sweep arc — the "rune is being read" hint.
        const float sweep = outer_angle;
        dl->PathArcTo(c, radius, sweep, sweep + kTwoPi * 0.25f, 24);
        dl->PathStroke(cream, ImDrawFlags_None, 2.2f);

        // Pulsing central dot.
        const float pulse = 0.5f + 0.5f * std::sin(elapsed * 3.0f);
        const float dot_r = 2.0f + 3.0f * pulse;
        dl->AddCircleFilled(c, dot_r, pulse_col);
    }

    // Wide twilight backdrop strip that anchors the top of the screen
    // behind the main menu bar. Everything cosmetic for the top header
    // row — gradient, starfield, distant moon, title, subtitle, version
    // stamp — is painted inside this single window so it all shares
    // one proven-working draw list (earlier attempts with separate
    // NoBackground windows dropped content under multi-viewport mode).
    static void drawFantasyTopBackdrop(
        const ImVec2& vp_pos,
        float width,
        float height,
        float menu_height,
        const char* title,
        const char* subtitle,
        const char* version) {

        ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
        ImGui::SetNextWindowPos(vp_pos);
        ImGui::SetNextWindowSize(ImVec2(width, height));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        const bool opened = ImGui::Begin(
            "##fantasy_top_backdrop",
            nullptr,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoBackground |
            ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoNav |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (!opened) {
            ImGui::End();
            ImGui::PopStyleVar(3);
            return;
        }

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p0 = vp_pos;
        const ImVec2 p1 = { vp_pos.x + width, vp_pos.y + height };

        // Vertical twilight gradient split into two bands for the
        // indigo → plum → dusty-rose transition from the HTML mockup.
        // Alphas are kept near-opaque so the gradient survives
        // composition over the brightly-lit game scene — at lower
        // alpha the strip was invisible against a bright sky.
        const float mid_y = vp_pos.y + height * 0.55f;
        const float low_y = p1.y;
        const ImU32 c_top_l = IM_COL32( 26, 18,  54, 255);
        const ImU32 c_top_r = IM_COL32( 38, 22,  72, 255);
        const ImU32 c_mid_l = IM_COL32( 58, 35,  87, 250);
        const ImU32 c_mid_r = IM_COL32( 90, 45,  98, 250);
        const ImU32 c_bot_l = IM_COL32(125, 59,  94, 210);
        const ImU32 c_bot_r = IM_COL32(201,118,  90, 190);

        dl->AddRectFilledMultiColor(
            p0, ImVec2(p1.x, mid_y),
            c_top_l, c_top_r, c_mid_r, c_mid_l);
        dl->AddRectFilledMultiColor(
            ImVec2(p0.x, mid_y), ImVec2(p1.x, low_y),
            c_mid_l, c_mid_r, c_bot_r, c_bot_l);

        // (Stars are now detected from the background image and drawn
        // as a twinkling overlay in Menu::draw — see detected_stars_.)

        // Distant moon on the far right — small enough not to crowd
        // the fps widget or the mesh-load HUD to its left.
        const ImVec2 far_moon = {
            vp_pos.x + width * 0.84f,
            vp_pos.y + height * 0.28f };
        dl->AddCircleFilled(far_moon, 18.0f, IM_COL32(253, 243, 209,  40), 32);
        dl->AddCircleFilled(far_moon, 12.0f, IM_COL32(253, 243, 209,  80), 32);
        dl->AddCircleFilled(far_moon,  7.0f, IM_COL32(253, 243, 209, 210), 32);

        // Thin gold underline separating the backdrop from the scene.
        dl->AddLine(
            ImVec2(p0.x, low_y - 1.0f),
            ImVec2(p1.x, low_y - 1.0f),
            IM_COL32(180, 140, 70, 140), 1.0f);

        // --- Title banner, drawn inline so it shares this draw list ---
        // Anchored below the menu bar, top-left corner.
        const float banner_x0 = vp_pos.x + 10.0f;
        const float banner_y0 = vp_pos.y + menu_height + 6.0f;
        const float banner_w  = 260.0f;
        const float banner_h  = 40.0f;
        const ImVec2 b0 = { banner_x0, banner_y0 };
        const ImVec2 b1 = { banner_x0 + banner_w, banner_y0 + banner_h };

        // A slightly darker inner panel so the banner has definition
        // against the gradient backdrop.
        dl->AddRectFilledMultiColor(
            b0, b1,
            IM_COL32( 18, 14,  38, 180),
            IM_COL32( 36, 26,  60, 160),
            IM_COL32( 16, 12,  30, 180),
            IM_COL32( 10,  8,  24, 210));
        // Gold frame.
        dl->AddRect(b0, b1, IM_COL32(200, 158, 80, 200), 2.0f, 0, 1.2f);

        // Little moon disc inside the banner (mirrors fantasy_menu.html).
        const ImVec2 m_c = { b0.x + 22.0f, b0.y + banner_h * 0.5f };
        dl->AddCircle      (m_c, 12.0f, IM_COL32(240, 230, 190,  80), 24, 2.0f);
        dl->AddCircle      (m_c,  9.0f, IM_COL32(240, 230, 190, 140), 24, 1.0f);
        dl->AddCircleFilled(m_c,  6.5f, IM_COL32(240, 230, 190, 240), 24);
        dl->AddCircleFilled(
            ImVec2(m_c.x - 1.5f, m_c.y - 1.5f),
            1.4f, IM_COL32(210, 200, 160, 220), 12);

        // Title + subtitle text.
        if (title) {
            dl->AddText(
                ImVec2(b0.x + 44.0f, b0.y +  4.0f),
                IM_COL32(240, 222, 172, 255),
                title);
        }
        if (subtitle) {
            dl->AddText(
                ImVec2(b0.x + 44.0f, b0.y + 20.0f),
                IM_COL32(205, 178, 122, 230),
                subtitle);
        }

        // Version stamp just below the banner.
        if (version) {
            dl->AddText(
                ImVec2(banner_x0 + 4.0f, banner_y0 + banner_h + 4.0f),
                IM_COL32(180, 155, 95, 220),
                version);
        }

        ImGui::End();
        ImGui::PopStyleVar(3);
    }

    // (Earlier drawFantasyTitleBanner was folded into
    // drawFantasyTopBackdrop to keep all header-row chrome on a single
    // proven-working draw list.)

    static void drawViewport(
        const ImTextureID& texture_id,
        const ImVec2& offset,
        const ImVec2& size,
        const float& aspect) {

        ImGui::BeginGroup();
        ImGui::Begin(
            "Viewport",
            nullptr,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar);

        ImVec2 available_size = ImGui::GetContentRegionAvail();
        ImVec2 clamped_size = available_size;
        clamped_size.y = std::min(available_size.y, available_size.x * aspect);
        clamped_size.x = clamped_size.y / aspect;

        ImGui::SetWindowPos(offset);

        ImGui::SetCursorPos(ImVec2((available_size.x - clamped_size.x) / 2, (available_size.y - clamped_size.y) / 2));
        ImGui::Image(texture_id, clamped_size);

        // Calculate position to render the icon on top of the base image
        ImVec2 icon_position;
        icon_position.x = offset.x + 10; // Adjust X offset for icon placement
        icon_position.y = offset.y + 10;  // Adjust Y offset for icon placement

        // Set the new cursor position for the icon
        ImGui::SetCursorScreenPos(icon_position);

        // Set a color with transparency (RGBA) for the icon (semi-transparent, e.g., 50% opacity)
        ImVec4 icon_tint = ImVec4(1.0f, 1.0f, 1.0f, 0.5f);  // White with 50% opacity

        // Render the icon on top of the base image with semi-transparency
        //ImGui::Image(icon_texture, ImVec2(50, 50), ImVec2(0, 0), ImVec2(1, 1), icon_tint);
        ImGui::Button("Button", ImVec2(150, 50));

        ImGui::End();
        ImGui::EndGroup();
    }
}

namespace engine {
namespace ui {

Menu::Menu(
    GLFWwindow* window,
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::Instance>& instance,
    const renderer::QueueFamilyList& queue_family_list,
    const renderer::SwapChainInfo& swap_chain_info,
    const std::shared_ptr<renderer::Queue>& graphics_queue,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::Sampler>& sampler,
    const std::shared_ptr<renderer::ImageView>& rt_image_view,
    const std::shared_ptr<renderer::ImageView>& main_image_view) {
    std::string path = "assets";
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        auto path_string = entry.path();
        auto ext_string = std::filesystem::path(path_string).extension();
        if (ext_string == ".glb" || ext_string == ".gltf") {
            gltf_file_names_.push_back(path_string.filename().string());
        }
    }

    renderer::Helper::initImgui(
        window,
        device,
        instance,
        queue_family_list,
        swap_chain_info,
        graphics_queue,
        descriptor_pool,
        render_pass);

    // Keep references for re-registering ImGui textures after swap chain
    // recreation and for cleanup on shutdown.
    device_  = device;
    sampler_ = sampler;

    // Initialise time-of-day from the player's local wall clock so the
    // sky lighting and the clock overlay both reflect the local time of day.
    {
        std::time_t now = std::time(nullptr);
        struct tm local_tm{};
#if defined(_WIN32)
        localtime_s(&local_tm, &now);
#else
        localtime_r(&now, &local_tm);
#endif
        const float local_hours =
            static_cast<float>(local_tm.tm_hour) +
            static_cast<float>(local_tm.tm_min)  / 60.0f +
            static_cast<float>(local_tm.tm_sec)  / 3600.0f;
        tod_hours_      = local_hours;
        tod_prev_hours_ = local_hours;
    }

    // Fantasy aesthetic (deep indigo panels, gold accents) picked up by
    // all subsequent ImGui draws. See applyFantasyStyle() in this file
    // for the palette; it ports the values from
    // realworld/design/fantasy_menu.html so the mockup and the live
    // game stay visually in sync.
    applyFantasyStyle();

    // Load title-screen layout from XML. If the file is missing we
    // fall back to hardcoded values so the title screen always works.
    title_config_ = loadTitleScreenConfig("assets/ui/title_screen.xml");
    if (!title_config_.loaded) {
        fprintf(stderr, "[title_screen] XML not found — using hardcoded fallback.\n");
        title_config_.title    = "Ashes of Eldra";
        title_config_.subtitle = "chronicles of the fallen realm";
        title_config_.version  = "v0.4.2  pre-alpha";
        title_config_.top_bar_height_extra = 72.0f;
        title_config_.background_image = "assets/ui/fantasy_bg.png";
        title_config_.menu_items = {
            {"New Game",        "new_game"},
            {"Load Game",       "load_game"},
            {"Settings",        "settings"},
            {"Quit to Desktop", "quit"}
        };
        title_config_.new_game_meshes = {
            "assets/Bistro_v5_2/BistroExterior.fbx",
            "assets/Bistro_v5_2/BistroInterior.fbx"
        };
        title_config_.loaded = true;
    }

    const float kTempScale = 1.0f;
    const float kMoistScale = 1.0f;

    weather_controls_.mix_rate = 0.92f;
    weather_controls_.sea_level_temperature = 30.0f;
    // temperature changes by energy from sun minus energy vapored from land surface.
    weather_controls_.soil_temp_adj = 0.0436f * kTempScale;
    // temperature changes by energy from sun minus energy vapored from water surface.
    weather_controls_.water_temp_adj = 0.0337f * kTempScale;
    // temperature changes by energy from sun minus energy vapored from water droplet.
    weather_controls_.moist_temp_convert = 0.0001f;
    weather_controls_.soil_moist_adj = 0.1124f * kMoistScale;
    weather_controls_.water_moist_adj = 0.5173f * kMoistScale;
    weather_controls_.transfer_ratio = 0.4f;// 0.8f;
    weather_controls_.transfer_noise_weight = 0.2f;
    weather_controls_.cloud_forming_ratio = 0.5f;
    weather_controls_.frozen_ext_factor = 2.0f;
    weather_controls_.frozen_pow_curve = 1.0f / 2.0f;

    // Convert the Vulkan image to an ImGui texture
    if (rt_image_view) {
        rt_texture_id_ =
            renderer::Helper::addImTextureID(sampler, rt_image_view);
    }

    if (main_image_view) {
        main_texture_id_ =
            renderer::Helper::addImTextureID(sampler, main_image_view);
    }

    // Fantasy twilight landscape backdrop. 1920x1080 PNG, painted
    // behind everything else on each frame. Load is best-effort —
    // if the file is missing we just won't draw it, rather than
    // taking the engine down.
    try {
        bg_texture_info_ = std::make_shared<renderer::TextureInfo>();
        engine::helper::createTextureImage(
            device,
            "assets/ui/fantasy_bg.png",
            renderer::Format::R8G8B8A8_UNORM,
            true,  // is_srgb — the image was painted in sRGB space
            *bg_texture_info_,
            std::source_location::current());
        if (bg_texture_info_->view) {
            bg_texture_id_ = renderer::Helper::addImTextureID(
                sampler, bg_texture_info_->view);
        }
    } catch (...) {
        // Swallow — a missing background shouldn't crash the game.
        bg_texture_info_.reset();
        bg_texture_id_ = ImTextureID(0);
    }

    // ---- Analog clock face texture ----------------------------------------
    try {
        clock_tex_info_ = std::make_shared<renderer::TextureInfo>();
        engine::helper::createTextureImage(
            device,
            "assets/ui/clock_face.png",
            renderer::Format::R8G8B8A8_UNORM,
            true,
            *clock_tex_info_,
            std::source_location::current());
        if (clock_tex_info_->view) {
            clock_tex_id_ = renderer::Helper::addImTextureID(
                sampler, clock_tex_info_->view);
        }
    } catch (...) {
        clock_tex_info_.reset();
        clock_tex_id_ = ImTextureID(0);
    }

    // ---- Viewport Play button icon ----------------------------------------
    try {
        play_icon_info_ = std::make_shared<renderer::TextureInfo>();
        engine::helper::createTextureImage(
            device,
            "assets/icon/play.png",
            renderer::Format::R8G8B8A8_UNORM,
            true,
            *play_icon_info_,
            std::source_location::current());
        if (play_icon_info_->view) {
            play_icon_id_ = renderer::Helper::addImTextureID(
                sampler, play_icon_info_->view);
        }
    } catch (...) {
        play_icon_info_.reset();
        play_icon_id_ = ImTextureID(0);
    }

    // ---- Scan the background PNG for bright star-like pixels ----
    // Load the image a second time (CPU side only) to read pixel data.
    // We look for small clusters of bright pixels (luminance > threshold)
    // surrounded by darker sky, record their normalised positions, then
    // free the pixel buffer. The cost is a one-time PNG decode at startup.
    {
        int w = 0, h = 0, ch = 0;
        unsigned char* px = stbi_load(
            "assets/ui/fantasy_bg.png", &w, &h, &ch, STBI_rgb_alpha);
        if (px && w > 0 && h > 0) {
            // Luminance threshold — anything above this is a candidate.
            // Tuned for the twilight landscape: sky is dark (L < 0.35),
            // stars are small bright dots (L > 0.65).
            constexpr float kLumThreshold = 0.65f;
            // Minimum spacing² between detected stars (in normalised coords)
            // to avoid clustering many hits from the same bright patch.
            constexpr float kMinDistSq = 0.012f * 0.012f;

            // Scan at 2-pixel stride for speed; stars are small so we
            // won't miss them.
            for (int y = 0; y < h; y += 2) {
                for (int x = 0; x < w; x += 2) {
                    const int idx = (y * w + x) * 4;
                    const float r = px[idx + 0] / 255.0f;
                    const float g = px[idx + 1] / 255.0f;
                    const float b = px[idx + 2] / 255.0f;
                    const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

                    if (lum < kLumThreshold) continue;

                    // Only detect stars in the upper 45% (sky region).
                    if (float(y) / float(h) > 0.45f) continue;

                    // A real star is a tiny bright dot surrounded by DARK sky.
                    // Check a wider ring (radius 6–10 px) — if the average
                    // luminance there is also high, this pixel is part of a
                    // large bright object (moon, cloud) and not a star.
                    float ring_sum = 0.0f;
                    int   ring_n   = 0;
                    for (int dy = -10; dy <= 10; dy += 4) {
                        for (int dx = -10; dx <= 10; dx += 4) {
                            const int d2 = dx * dx + dy * dy;
                            if (d2 < 36 || d2 > 100) continue; // ring 6..10 px
                            int rx = x + dx, ry = y + dy;
                            if (rx < 0 || ry < 0 || rx >= w || ry >= h) continue;
                            const int ri = (ry * w + rx) * 4;
                            ring_sum += 0.2126f * px[ri] / 255.0f +
                                        0.7152f * px[ri+1] / 255.0f +
                                        0.0722f * px[ri+2] / 255.0f;
                            ++ring_n;
                        }
                    }
                    // If the surrounding ring is brighter than dark sky,
                    // this is moon/cloud, not a star. Skip it.
                    if (ring_n > 0 && (ring_sum / float(ring_n)) > 0.35f) continue;

                    // Also verify this pixel is a local maximum (brighter
                    // than its immediate neighbours).
                    bool is_local_peak = true;
                    for (int dy = -2; dy <= 2 && is_local_peak; dy += 2) {
                        for (int dx = -2; dx <= 2 && is_local_peak; dx += 2) {
                            if (dx == 0 && dy == 0) continue;
                            int nx2 = x + dx, ny2 = y + dy;
                            if (nx2 < 0 || ny2 < 0 || nx2 >= w || ny2 >= h) continue;
                            const int ni = (ny2 * w + nx2) * 4;
                            const float nl = 0.2126f * px[ni] / 255.0f +
                                             0.7152f * px[ni+1] / 255.0f +
                                             0.0722f * px[ni+2] / 255.0f;
                            if (nl > lum) is_local_peak = false;
                        }
                    }
                    if (!is_local_peak) continue;

                    const float fnx = float(x) / float(w);
                    const float fny = float(y) / float(h);

                    // Reject if too close to an already-detected star.
                    bool too_close = false;
                    for (const auto& s : detected_stars_) {
                        float ddx = fnx - s.nx, ddy = fny - s.ny;
                        if (ddx * ddx + ddy * ddy < kMinDistSq) {
                            too_close = true;
                            break;
                        }
                    }
                    if (too_close) continue;

                    DetectedStar star;
                    star.nx = fnx;
                    star.ny = fny;
                    star.brightness = lum;
                    // Estimate radius from how many adjacent pixels are
                    // also bright — cheap 1D horizontal scan.
                    int span = 0;
                    for (int sx = x + 1; sx < w && sx < x + 8; ++sx) {
                        const int si = (y * w + sx) * 4;
                        const float sl = 0.2126f * px[si] / 255.0f +
                                         0.7152f * px[si+1] / 255.0f +
                                         0.0722f * px[si+2] / 255.0f;
                        if (sl < kLumThreshold * 0.7f) break;
                        ++span;
                    }
                    star.radius = 1.0f + float(span) * 0.5f;

                    // Skip large bright blobs (the moon, glowing clouds, etc).
                    // Real stars are tiny — anything wider than ~4 px is not a star.
                    if (star.radius > 4.0f) continue;

                    // Unique twinkle parameters derived from position.
                    star.speed = 0.8f + 2.0f * (float(std::abs(x * 7 + y * 13) % 100) / 100.0f);
                    star.phase = 6.28f * (float(std::abs(x * 3 + y * 17) % 100) / 100.0f);

                    detected_stars_.push_back(star);
                }
            }

            stbi_image_free(px);
            fprintf(stderr, "[title_screen] Detected %zu stars in background image.\n",
                    detected_stars_.size());
        }
    }

    chat_box_ = std::make_shared<ChatBox>();
}

void Menu::init(
    GLFWwindow* window,
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::Instance>& instance,
    const renderer::QueueFamilyList& queue_family_list,
    const renderer::SwapChainInfo& swap_chain_info,
    const std::shared_ptr<renderer::Queue>& graphics_queue,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass) {

    renderer::Helper::initImgui(
        window,
        device,
        instance,
        queue_family_list,
        swap_chain_info,
        graphics_queue,
        descriptor_pool,
        render_pass);

    // Re-apply the fantasy colour theme — the ImGui context was
    // destroyed and recreated during swap chain recreation, so the
    // style reverts to default.
    applyFantasyStyle();

    // Re-register the background texture with the fresh ImGui/Vulkan
    // descriptor pool. The Vulkan image + view survive swap chain
    // recreation, but the ImGui-side descriptor does not.
    if (bg_texture_info_ && bg_texture_info_->view && sampler_) {
        bg_texture_id_ = renderer::Helper::addImTextureID(
            sampler_, bg_texture_info_->view);
    }

    // Same hazard for the analog-clock face texture.  clock_tex_id_ is
    // built once in the title-screen constructor from a pool-allocated
    // VkDescriptorSet, and the always-on-top clock block in Menu::draw
    // (~line 2437) uses it on EVERY frame.  After cleanupSwapChain
    // destroys descriptor_pool_, this descriptor handle is dangling;
    // the next ImGui_ImplVulkan_RenderDrawData binds it and faults in
    // the NVIDIA driver (most reliably reproduced via fullscreen ↔
    // windowed toggles, which trigger several rapid resizes).
    if (clock_tex_info_ && clock_tex_info_->view && sampler_) {
        clock_tex_id_ = renderer::Helper::addImTextureID(
            sampler_, clock_tex_info_->view);
    } else {
        // No clock face image available — null out so the fallback
        // circle in Menu::draw is used instead of a stale handle.
        clock_tex_id_ = ImTextureID(0);
    }

    // Same hazard for the viewport Play button icon.
    if (play_icon_info_ && play_icon_info_->view && sampler_) {
        play_icon_id_ = renderer::Helper::addImTextureID(
            sampler_, play_icon_info_->view);
    } else {
        play_icon_id_ = ImTextureID(0);   // fall back to the drawn triangle
    }

    // rt_texture_id_ and main_texture_id_ are also pool-allocated and
    // would be dangling here.  They're only referenced inside `#if 0`
    // blocks today (drawViewport for the disabled raytrace preview),
    // but null them out defensively so any future re-enabling doesn't
    // accidentally bind a dead handle from the old pool.
    rt_texture_id_   = ImTextureID(0);
    main_texture_id_ = ImTextureID(0);

    // Content-browser thumbnail textures hit the SAME hazard: their
    // ImTextureIDs are VkDescriptorSets allocated from the descriptor pool
    // that cleanupSwapChain just destroyed.  The underlying Vulkan images
    // (TextureInfo) survive recreation, so re-register each surviving view
    // with the fresh pool; drop any that lost their view.  Without this the
    // grid binds a dangling descriptor on the next frame and the NVIDIA
    // driver faults inside ImGui_ImplVulkan_RenderDrawData (seen on resize).
    for (auto& kv : thumb_cache_) {
        ThumbTex& t = kv.second;
        if (t.failed) continue;
        if (t.info && t.info->view)
            t.id = renderer::Helper::addImTextureID(sampler_, t.info->view);
        else
            t.failed = true;
    }
    // Retired (superseded) thumbnail textures are no longer referenced by the
    // UI; the device is idle here, so free them outright rather than
    // re-registering dead handles.
    retired_thumbs_.clear();

    // Selection-mask texture hits the same hazard.  Re-register its surviving
    // view with the fresh pool; drop the deferred-free list WITHOUT calling
    // RemoveTexture (those descriptors belonged to the now-destroyed pool).
    // Reset the change key so it regenerates for the (likely changed) viewport.
    if (sel_mask_tex_ && sel_mask_tex_->view)
        sel_mask_id_ = renderer::Helper::addImTextureID(sampler_, sel_mask_tex_->view);
    else
        sel_mask_id_ = 0;
    sel_mask_dead_.clear();
    sel_mask_obj_  = nullptr;
    sel_mask_node_ = -3;

    // Debug Display GPU preview target: same descriptor-pool hazard as the
    // selection mask — re-register its view with the fresh pool, and drop its
    // deferred-free descriptor list WITHOUT RemoveTexture (those belonged to
    // the now-destroyed pool).
    engine::helper::MeshPreview::dropDeadImGuiDescriptors();
    engine::helper::MeshPreview::reregisterImGui();
}

bool Menu::draw(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::Framebuffer>& framebuffer,
    const glm::uvec2& screen_size,
    const std::shared_ptr<scene_rendering::Skydome>& skydome,
    bool& dump_volume_noise,
    const float& delta_t) {

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ── UE-style editor dock layout ───────────────────────────────────
    // Drawn first (as a transparent pass-through host) so the 3D scene
    // shows through the central viewport and the Outliner/Details/Content
    // panels dock around it.  Active from app start in editor mode (so the
    // title screen + loading all happen inside the viewport), not just
    // in-game.
    if (editor_enabled_) {
        drawEditorDockSpace();
    }

    // ── Main viewport intro: hold solid gray 1.5s, then fade in over 3s ──
    // Plays once per program run.  A function-local static monotonic clock is
    // used (not ImGui::GetTime or a Menu member) so it does NOT replay when the
    // window is resized — those events recreate the swapchain/Menu/ImGui frame
    // but leave this static origin untouched.
    {
        constexpr double kHold = 0.75;  // fully gray
        constexpr double kFade = 3.0;   // gray -> live render
        static const std::chrono::steady_clock::time_point kFadeOrigin =
            std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - kFadeOrigin).count();
        if (elapsed < kHold + kFade) {
            double opacity = 1.0;                       // hold phase
            if (elapsed > kHold)
                opacity = 1.0 - (elapsed - kHold) / kFade;  // fade phase
            if (opacity < 0.0) opacity = 0.0;
            ImVec2 vp_pos, vp_size, vp_c;
            getViewportScreenRect(vp_pos, vp_size, vp_c);
            const int a = (int)(opacity * 255.0 + 0.5);
            ImGui::GetBackgroundDrawList(ImGui::GetMainViewport())->AddRectFilled(
                vp_pos, ImVec2(vp_pos.x + vp_size.x, vp_pos.y + vp_size.y),
                IM_COL32(96, 96, 102, a));
        }
    }

    // ── Hover-object tooltip ──────────────────────────────────────────
    // application.cpp ray-picks the scene object under the mouse each
    // frame and pushes its name via setHoverObjectName().  Draw it in a
    // small box pinned to the cursor.  Skipped while the pointer is over a
    // menu window (WantCaptureMouse) so it doesn't fight the UI, and when
    // the name is empty (pointing at sky / nothing).
    if (!hover_object_name_.empty() &&
        !ImGui::GetIO().WantCaptureMouse) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();
        const ImVec2 mp = ImGui::GetMousePos();
        const ImVec2 ts = ImGui::CalcTextSize(hover_object_name_.c_str());
        const ImVec2 pad(6.0f, 4.0f);
        // Offset down-right of the cursor so the pointer tip stays visible.
        const ImVec2 org(mp.x + 16.0f, mp.y + 10.0f);
        fg->AddRectFilled(
            ImVec2(org.x - pad.x, org.y - pad.y),
            ImVec2(org.x + ts.x + pad.x, org.y + ts.y + pad.y),
            IM_COL32(0, 0, 0, 190), 4.0f);
        fg->AddText(org, IM_COL32(255, 235, 120, 255),
                    hover_object_name_.c_str());
    }

    // ── Player debug overlay ──────────────────────────────────────────
    // Always-on locator widget — paints a red square at the player's
    // projected screen position plus a HUD readout of the world coords
    // and camera-relative distance.  Surface for when the 3D character
    // doesn't appear visually but the controller state is fine: lets
    // the user see at a glance whether the player is off-screen, behind
    // them, or just very far away.
    // Default OFF — set to true to re-enable the cyan bbox wireframe +
    // red position marker + HUD text.  Was drawing unconditionally
    // before; user asked for the debug overlay off by default.
    static bool s_show_player_overlay_ = false;
    if (s_show_player_overlay_ && has_player_debug_info_) {
        // Non-const because GetForegroundDrawList takes a non-const
        // ImGuiViewport*.  The viewport returned by GetMainViewport()
        // is owned by Dear ImGui — we're only reading from it.
        ImGuiViewport* mvp = ImGui::GetMainViewport();
        const float sw = mvp->Size.x, sh = mvp->Size.y;
        const ImVec2 spos = mvp->Pos;

        // Project world position to clip space.
        glm::vec4 clip = player_view_proj_ *
                         glm::vec4(player_world_pos_, 1.0f);
        bool on_screen = (clip.w > 0.0f);
        ImVec2 screen_px(0, 0);
        if (on_screen) {
            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            // NDC in [-1,1] → pixel coords.  Vulkan's clip-space Y is
            // negated by the projection's flip, so ndc.y already has
            // image-down-positive direction; just remap [-1,1]→[0,1].
            float px = (ndc.x * 0.5f + 0.5f) * sw + spos.x;
            float py = (ndc.y * 0.5f + 0.5f) * sh + spos.y;
            screen_px = ImVec2(px, py);
            // Off-screen check: only the XY frustum bounds matter for
            // "is the point inside the rendered image".  Don't reject
            // on ndc.z — when the player straddles the near plane (e.g.
            // you turn around fast and the spawn point's projection is
            // just past z=1) the marker would flicker out for a frame
            // even though the screen-space position is still meaningful.
            // We still rejected clip.w <= 0 above, which covers
            // "behind the camera" — the only z-related case that
            // actually invalidates the screen position.
            //
            // Hysteresis pad on the edge: a stationary character near
            // the viewport edge oscillates between [on-screen marker]
            // and [edge arrow] when ndc.x/y wiggles past ±1 due to
            // floating-point noise.  Pad the on-screen bound out to
            // ±1.02 (and keep the off-screen state until truly past
            // ±1.1 in the AddText branch below if we add it later).
            // Without this the marker flickers at the edges every
            // frame.
            constexpr float kOnScreenPad = 0.02f;
            on_screen = on_screen &&
                ndc.x >= -1.0f - kOnScreenPad && ndc.x <= 1.0f + kOnScreenPad &&
                ndc.y >= -1.0f - kOnScreenPad && ndc.y <= 1.0f + kOnScreenPad;
        }

        ImDrawList* fg = ImGui::GetForegroundDrawList(mvp);
        const ImU32 kRed     = IM_COL32(255,  40,  40, 255);
        const ImU32 kRedSoft = IM_COL32(255,  40,  40, 120);

        if (on_screen) {
            // 24×24 px hollow red square + filled centre dot, plus a
            // larger faint halo so it's findable even against busy
            // backgrounds.
            const float half  = 12.0f;
            const float halo  = 32.0f;
            fg->AddCircleFilled(screen_px, halo, kRedSoft, 24);
            fg->AddRect(
                ImVec2(screen_px.x - half, screen_px.y - half),
                ImVec2(screen_px.x + half, screen_px.y + half),
                kRed, 0.0f, 0, 3.0f);
            fg->AddCircleFilled(screen_px, 3.0f, kRed, 12);
        } else {
            // Player is off-screen — pin the marker to the centre of
            // the matching screen edge with an arrow nub pointing at
            // the player.  Compute direction in camera space.
            glm::vec4 cs = player_view_proj_ *
                           glm::vec4(player_world_pos_, 1.0f);
            // Behind camera or way off — fall back to using the X/Y of
            // the clip-space coordinate (clamped to ±1) so the marker
            // sticks to an edge facing the player's azimuth.
            glm::vec2 dir(cs.x, cs.y);
            float L = glm::length(dir);
            if (L > 1e-4f) dir /= L;
            else dir = glm::vec2(1.0f, 0.0f);
            float edge_x = spos.x + sw * (0.5f + dir.x * 0.45f);
            float edge_y = spos.y + sh * (0.5f + dir.y * 0.45f);
            ImVec2 edge_px(edge_x, edge_y);
            fg->AddCircle(edge_px, 16.0f, kRed, 16, 3.0f);
            fg->AddText(
                ImVec2(edge_x + 18.0f, edge_y - 8.0f),
                kRed, "Player off-screen");
        }

        // ── World-space AABB wireframe overlay ───────────────────────
        // Projects the 8 corners of the player's world AABB through
        // the same view_proj used for the position marker and draws
        // the 12 edges as ImGui lines.  We CLIP each edge against
        // the camera near plane (w = w_eps) before projecting — the
        // earlier "skip the whole edge if either endpoint has w <= 0"
        // approach made edges pop in/out every time the camera
        // wiggled past a corner, which read as flickering even
        // though the box itself was stable.  Proper clipping draws
        // a partial line up to the near plane so the wireframe
        // stays continuous as the camera moves.
        const ImU32 kBoxColor = IM_COL32( 60, 220, 255, 220);
        if (player_bbox_valid_) {
            const glm::vec3& bmin = player_bbox_min_;
            const glm::vec3& bmax = player_bbox_max_;
            // Corner ordering: bit 0 = X (min/max), bit 1 = Y, bit 2 = Z.
            // So corner i's coords are (X[i&1], Y[(i>>1)&1], Z[(i>>2)&1]).
            const float Xc[2] = { bmin.x, bmax.x };
            const float Yc[2] = { bmin.y, bmax.y };
            const float Zc[2] = { bmin.z, bmax.z };
            // Stash clip-space (not NDC) for all 8 corners.  Clipping
            // happens per-edge below before we divide by w.
            glm::vec4 clip[8];
            for (int i = 0; i < 8; ++i) {
                glm::vec3 wc(Xc[i & 1],
                             Yc[(i >> 1) & 1],
                             Zc[(i >> 2) & 1]);
                clip[i] = player_view_proj_ * glm::vec4(wc, 1.0f);
            }
            // Near-plane epsilon: anything with w smaller than this
            // is treated as "behind" and gets clipped.  Picked to be
            // tighter than any reasonable camera near plane (= 0.01m)
            // so we don't accidentally clip valid geometry.
            constexpr float kWEps = 1e-3f;

            // Helper: project a (post-clip) clip-space point to
            // pixel coordinates.  Caller must guarantee cc.w >= kWEps.
            auto toPx = [&](const glm::vec4& cc) -> ImVec2 {
                glm::vec3 ndc = glm::vec3(cc) / cc.w;
                return ImVec2(
                    (ndc.x * 0.5f + 0.5f) * sw + spos.x,
                    (ndc.y * 0.5f + 0.5f) * sh + spos.y);
            };

            // 12 edges by corner-index pairs.  An edge connects two
            // corners that differ in exactly one bit of the index.
            const int edges[12][2] = {
                {0,1},{2,3},{4,5},{6,7},   // X-axis edges
                {0,2},{1,3},{4,6},{5,7},   // Y-axis edges
                {0,4},{1,5},{2,6},{3,7},   // Z-axis edges
            };
            for (int e = 0; e < 12; ++e) {
                glm::vec4 a = clip[edges[e][0]];
                glm::vec4 b = clip[edges[e][1]];
                const bool a_in = a.w >= kWEps;
                const bool b_in = b.w >= kWEps;
                if (!a_in && !b_in) continue; // whole edge behind camera
                if (!a_in) {
                    // Clip A to the near plane: find t in [0,1] on the
                    // line A → B where (1-t)*a.w + t*b.w = kWEps.
                    // (b.w - a.w) is non-zero because b is in front.
                    float t = (kWEps - a.w) / (b.w - a.w);
                    a = a + t * (b - a);
                }
                if (!b_in) {
                    float t = (kWEps - b.w) / (a.w - b.w);
                    b = b + t * (a - b);
                }
                fg->AddLine(toPx(a), toPx(b), kBoxColor, 2.0f);
            }
        }

        // HUD text — top-right corner, always visible.  Adds the
        // bbox range when valid so we can read the actual numbers
        // alongside the wireframe.
        const float dist = glm::length(
            player_world_pos_ - player_cam_pos_);
        char buf[512];
        if (player_bbox_valid_) {
            std::snprintf(buf, sizeof(buf),
                "Player: (%.2f, %.2f, %.2f)  dist=%.2fm  %s\n"
                "BBox:   (%.2f, %.2f, %.2f) .. (%.2f, %.2f, %.2f)\n"
                "Size:    %.2f x %.2f x %.2f m",
                player_world_pos_.x, player_world_pos_.y, player_world_pos_.z,
                dist, on_screen ? "[on-screen]" : "[off-screen]",
                player_bbox_min_.x, player_bbox_min_.y, player_bbox_min_.z,
                player_bbox_max_.x, player_bbox_max_.y, player_bbox_max_.z,
                player_bbox_max_.x - player_bbox_min_.x,
                player_bbox_max_.y - player_bbox_min_.y,
                player_bbox_max_.z - player_bbox_min_.z);
        } else {
            std::snprintf(buf, sizeof(buf),
                "Player: (%.2f, %.2f, %.2f)  dist=%.2fm  %s",
                player_world_pos_.x, player_world_pos_.y, player_world_pos_.z,
                dist, on_screen ? "[on-screen]" : "[off-screen]");
        }
        const ImVec2 ts = ImGui::CalcTextSize(buf);
        const ImVec2 pad(8.0f, 4.0f);
        ImVec2 pos(spos.x + sw - ts.x - pad.x * 2.0f - 12.0f,
                   spos.y + 36.0f);
        fg->AddRectFilled(
            ImVec2(pos.x - pad.x, pos.y - pad.y),
            ImVec2(pos.x + ts.x + pad.x, pos.y + ts.y + pad.y),
            IM_COL32(0, 0, 0, 160), 4.0f);
        fg->AddText(pos, IM_COL32(255, 255, 255, 255), buf);
    }

    // ── Outliner selected-object highlight + mask ─────────────────────────
    // Bright bounding box + a translucent screen-space "mask" around the
    // object (or sub-object) selected in the editor Outliner, projected into
    // the viewport through the camera's view_proj (same one the player marker
    // uses, fed every frame).  Double-clicking an Outliner item ALSO teleports
    // the camera to frame it — that part is consumed app-side via
    // takeEditorFocus().  The box pulses so it reads as a live selection.
    if (editor_enabled_) {
        // Refresh the exact silhouette mask if the selection / camera changed.
        updateSelectionMask();

        ImVec2 hvp_pos, hvp_size, hvp_c;
        getViewportScreenRect(hvp_pos, hvp_size, hvp_c);
        const float hsw = hvp_size.x, hsh = hvp_size.y;
        ImDrawList* hfg = ImGui::GetForegroundDrawList();

        // Pulsing colour so the selection reads as "live".
        const float pulse = 0.5f + 0.5f * std::sin((float)ImGui::GetTime() * 4.0f);
        const int   edge_a = 180 + (int)(75.0f * pulse);
        const ImU32 kEdge = IM_COL32(255, 190, 50, edge_a);
        const ImU32 kMask = IM_COL32(255, 190, 50, 70);

        // Confine ALL highlight drawing to the viewport rect so it can't bleed
        // over the docked panels (Outliner / Content Browser / Output Log).
        hfg->PushClipRect(hvp_pos,
            ImVec2(hvp_pos.x + hsw, hvp_pos.y + hsh), true);

        // ── Scene-camera frustum gizmo ───────────────────────────────────
        // Selecting a camera object (.rwcam row) draws a wireframe frustum
        // pyramid + up-triangle at the camera's pose so the captured view
        // point/direction is visible in the scene.
        const bool cam_selected =
            editor_selected_ >= 0 &&
            editor_selected_ < (int)editor_objects_.size() &&
            editor_objects_[editor_selected_].is_camera &&
            editor_objects_[editor_selected_].scene_xform;
        // Suppress the gizmo while looking THROUGH this camera (view eye
        // at/near the camera position — the frustum would hang right in
        // front of the lens): apex clip-w ≈ view-space depth.
        bool cam_gizmo_visible = cam_selected;
        if (cam_selected) {
            const glm::vec4 apex_cc = player_view_proj_ * glm::vec4(
                editor_objects_[editor_selected_].scene_xform->translation,
                1.0f);
            if (apex_cc.w > -0.75f && apex_cc.w < 0.75f)
                cam_gizmo_visible = false;
        }
        if (cam_gizmo_visible) {
            const auto& xf = *editor_objects_[editor_selected_].scene_xform;
            const glm::vec3 cpos = xf.translation;
            const glm::mat3 R   = glm::mat3_cast(xf.rotation);
            const glm::vec3 cr  = R[0];           // right
            const glm::vec3 cu  = R[1];           // up
            const glm::vec3 cf  = -R[2];          // forward (looks down -Z)

            // Compact frustum: 0.6 m deep, 60° vertical FOV, 16:9.
            const float dist = 0.6f;
            const float hh   = dist * std::tan(glm::radians(60.0f * 0.5f));
            const float hw   = hh * (16.0f / 9.0f);
            const glm::vec3 c  = cpos + cf * dist;
            const glm::vec3 q0 = c - cr * hw - cu * hh;
            const glm::vec3 q1 = c + cr * hw - cu * hh;
            const glm::vec3 q2 = c + cr * hw + cu * hh;
            const glm::vec3 q3 = c - cr * hw + cu * hh;
            // "Up" indicator triangle above the far rect.
            const glm::vec3 t0 = c - cr * hw * 0.35f + cu * hh;
            const glm::vec3 t1 = c + cr * hw * 0.35f + cu * hh;
            const glm::vec3 t2 = c + cu * hh * 1.55f;

            constexpr float kWEps = 1e-3f;
            auto camToPx = [&](const glm::vec3& wp, ImVec2& out) -> bool {
                glm::vec4 cc = player_view_proj_ * glm::vec4(wp, 1.0f);
                if (cc.w < kWEps) return false;
                glm::vec3 ndc = glm::vec3(cc) / cc.w;
                out = ImVec2((ndc.x * 0.5f + 0.5f) * hsw + hvp_pos.x,
                             (ndc.y * 0.5f + 0.5f) * hsh + hvp_pos.y);
                return true;
            };
            auto camLine = [&](const glm::vec3& a, const glm::vec3& b,
                               ImU32 col, float th) {
                ImVec2 pa, pb;
                if (camToPx(a, pa) && camToPx(b, pb))
                    hfg->AddLine(pa, pb, col, th);
            };
            // Pulsing, fully bright cyan + a translucent fill on the far
            // rect so the gizmo pops against any scene content.
            const int   cam_a  = 200 + (int)(55.0f * pulse);
            const ImU32 kCam     = IM_COL32(0, 220, 255, cam_a);
            const ImU32 kCamFill = IM_COL32(0, 220, 255, 60);
            const ImU32 kCamUp   = IM_COL32(255, 190, 50, cam_a);
            // Translucent far-rect + up-triangle fills.
            {
                ImVec2 p0, p1, p2, p3;
                if (camToPx(q0, p0) && camToPx(q1, p1) &&
                    camToPx(q2, p2) && camToPx(q3, p3)) {
                    const ImVec2 quad[4] = { p0, p1, p2, p3 };
                    hfg->AddConvexPolyFilled(quad, 4, kCamFill);
                }
                ImVec2 u0, u1, u2;
                if (camToPx(t0, u0) && camToPx(t1, u1) && camToPx(t2, u2)) {
                    hfg->AddTriangleFilled(u0, u1, u2,
                                           IM_COL32(255, 190, 50, 150));
                }
            }
            // Apex → far corners.
            camLine(cpos, q0, kCam, 3.0f);
            camLine(cpos, q1, kCam, 3.0f);
            camLine(cpos, q2, kCam, 3.0f);
            camLine(cpos, q3, kCam, 3.0f);
            // Far rectangle.
            camLine(q0, q1, kCam, 3.0f);
            camLine(q1, q2, kCam, 3.0f);
            camLine(q2, q3, kCam, 3.0f);
            camLine(q3, q0, kCam, 3.0f);
            // Up triangle outline.
            camLine(t0, t1, kCamUp, 3.0f);
            camLine(t1, t2, kCamUp, 3.0f);
            camLine(t2, t0, kCamUp, 3.0f);
            // Filled apex dot + ring so the camera position reads from any
            // angle (and from behind, where the frustum is nearly edge-on).
            ImVec2 apx;
            if (camToPx(cpos, apx)) {
                hfg->AddCircleFilled(apx, 5.0f, kCam, 12);
                hfg->AddCircle(apx, 9.0f, kCam, 16, 2.5f);
            }
        }

        // Preferred: the EXACT silhouette mask (CPU-rasterised this object from
        // the camera).  It already fills the viewport (rendered at the camera
        // view), so draw it 1:1 over the viewport rect.  Falls back to the
        // convex-hull outline, then the bounding box, when no mask is ready.
        std::vector<ImVec2> hull;
        glm::vec3 hb_min, hb_max;
        if (cam_selected) {
            // Camera gizmo already drawn — skip the mask/hull/bbox path.
        } else if (sel_mask_id_) {
            hfg->AddImage(sel_mask_id_, hvp_pos,
                ImVec2(hvp_pos.x + hsw, hvp_pos.y + hsh));
        } else if (selectedScreenHull(hull)) {
            hfg->AddConvexPolyFilled(hull.data(), (int)hull.size(), kMask);
            hfg->AddPolyline(hull.data(), (int)hull.size(), kEdge,
                             ImDrawFlags_Closed, 2.5f);
        } else if (selectedWorldAabb(hb_min, hb_max)) {
            // Fallback (e.g. a whole-FBX group selection): bounding box.
            const float Xc[2] = { hb_min.x, hb_max.x };
            const float Yc[2] = { hb_min.y, hb_max.y };
            const float Zc[2] = { hb_min.z, hb_max.z };
            glm::vec4 clip[8];
            for (int i = 0; i < 8; ++i)
                clip[i] = player_view_proj_ * glm::vec4(
                    Xc[i & 1], Yc[(i >> 1) & 1], Zc[(i >> 2) & 1], 1.0f);
            constexpr float kWEps = 1e-3f;
            auto toPx = [&](const glm::vec4& cc) -> ImVec2 {
                glm::vec3 ndc = glm::vec3(cc) / cc.w;
                return ImVec2((ndc.x * 0.5f + 0.5f) * hsw + hvp_pos.x,
                              (ndc.y * 0.5f + 0.5f) * hsh + hvp_pos.y);
            };
            static const int edges[12][2] = {
                {0,1},{2,3},{4,5},{6,7}, {0,2},{1,3},{4,6},{5,7},
                {0,4},{1,5},{2,6},{3,7} };
            for (int e = 0; e < 12; ++e) {
                glm::vec4 A = clip[edges[e][0]];
                glm::vec4 B = clip[edges[e][1]];
                const bool ain = A.w >= kWEps, bin = B.w >= kWEps;
                if (!ain && !bin) continue;
                if (!ain) A = A + ((kWEps - A.w) / (B.w - A.w)) * (B - A);
                if (!bin) B = B + ((kWEps - B.w) / (A.w - B.w)) * (A - B);
                hfg->AddLine(toPx(A), toPx(B), kEdge, 2.5f);
            }
        }
        hfg->PopClipRect();
    }

    // ── Scene-grid origin axis labels (X / Y / Z at the gizmo tips) ──────
    // Companion to SceneGrid's solid origin arrows: project each arrow-tip
    // world position through the camera (player_view_proj_, fed every frame
    // by setPlayerDebugInfo) into the viewport and paint a colour-matched
    // glyph there.  Same NDC→pixel mapping as the selection bbox above; the
    // Vulkan projection already flips Y, so [-1,1]→[0,1] remap is enough.
    // Hidden in play mode alongside the grid/gizmo draw itself (the
    // show_scene_grid_ toggle latches on from "Create Scene" and must
    // not leak editor aids into gameplay).
    if (show_scene_grid_ && !play_mode_) {
        ImVec2 avp_pos, avp_size, avp_c;
        getViewportScreenRect(avp_pos, avp_size, avp_c);
        if (avp_size.x > 1.0f && avp_size.y > 1.0f) {
            // BACKGROUND draw list: the labels composite over the 3D viewport
            // but stay BEHIND ImGui windows, so menus/popups aren't overdrawn
            // by the X/Y/Z glyphs.  (Foreground would paint over the menu.)
            ImDrawList* afg = ImGui::GetBackgroundDrawList();
            afg->PushClipRect(
                avp_pos,
                ImVec2(avp_pos.x + avp_size.x, avp_pos.y + avp_size.y),
                true);

            // Slightly past the 3.1 m arrow tips (shaft 2.6 + head 0.5 in
            // scene_grid.cpp) so the glyph floats clear of the arrowhead.
            struct AxisLabel {
                glm::vec3   world;
                ImU32       color;
                const char* text;
            };
            const AxisLabel axis_labels[3] = {
                { glm::vec3(3.45f, 0.0f, 0.0f), IM_COL32(235,  70,  70, 255), "X" },
                { glm::vec3(0.0f, 3.45f, 0.0f), IM_COL32( 75, 220,  90, 255), "Y" },
                { glm::vec3(0.0f, 0.0f, 3.45f), IM_COL32( 80, 130, 245, 255), "Z" },
            };
            for (const auto& L : axis_labels) {
                glm::vec4 clip = player_view_proj_ * glm::vec4(L.world, 1.0f);
                if (clip.w <= 1e-3f) continue;       // behind the camera
                glm::vec3 ndc = glm::vec3(clip) / clip.w;
                if (ndc.x < -1.05f || ndc.x > 1.05f ||
                    ndc.y < -1.05f || ndc.y > 1.05f) continue;
                const ImVec2 px(
                    (ndc.x * 0.5f + 0.5f) * avp_size.x + avp_pos.x,
                    (ndc.y * 0.5f + 0.5f) * avp_size.y + avp_pos.y);

                // Centre the glyph on the projected point with a soft dark
                // backing pill so it reads over any background.  Drawn at
                // 2x the default font size so the labels read clearly.
                ImFont*     fnt   = ImGui::GetFont();
                const float fsize = ImGui::GetFontSize() * 2.0f;
                ImVec2 ts = ImGui::CalcTextSize(L.text);
                ts.x *= 2.0f;
                ts.y *= 2.0f;
                const ImVec2 tp(px.x - ts.x * 0.5f, px.y - ts.y * 0.5f);
                afg->AddRectFilled(
                    ImVec2(tp.x - 4.0f, tp.y - 2.0f),
                    ImVec2(tp.x + ts.x + 4.0f, tp.y + ts.y + 2.0f),
                    IM_COL32(0, 0, 0, 140), 4.0f);
                afg->AddText(fnt, fsize, tp, L.color, L.text);
            }
            afg->PopClipRect();
        }
    }

    // ── Classifier progress bar (top-pinned overlay) ─────────────────
    // Full-width ImGui window pinned to the very top of the viewport
    // so it's the first thing the user sees during a long classify.
    // Two rows:
    //   1. ImGui::ProgressBar driven by bytes_received / estimated
    //      total.  Capped at 100 % so an overshoot doesn't render
    //      wildly.  Shows a 0.0 fraction at the moment kick-off
    //      happens but BEFORE the daemon has produced any bytes;
    //      jumps to the upload-size fraction the moment the
    //      WinHttpSendRequest completes.
    //   2. A summary line ("LLM: pending — 12.4s, 4096/77000 bytes
    //      (5%)" etc.).
    // Drawn unconditionally except in the Idle state so the title
    // screen stays clean.
    // Hide BOTH progress bars once the collision ("simplified mesh") build
    // has finished — the user wants a clean screen at that point.  The
    // collision bar below is gated to Building-only for the same reason.
    //
    // RE-ENABLED with the bake feature: the classifier only runs as part
    // of Tools > Bake Collision Map now, so this bar IS the bake UI's
    // first phase (the collision-build bar below is the second).
    constexpr bool kShowMeshCategoryProgressBar = true;
    if (kShowMeshCategoryProgressBar &&
        (classifier_status_ == ClassifierStatus::Pending ||
         classifier_status_ == ClassifierStatus::Ready   ||
         classifier_status_ == ClassifierStatus::Failed) &&
        collision_build_status_ != CollisionBuildStatus::Done) {
        ImVec2 vp_pos, vp_size, vp_c;
        getViewportScreenRect(vp_pos, vp_size, vp_c);
        const float bar_h = 44.0f;  // two rows + padding
        // Half-width, horizontally centred on the VIEWPORT's top edge so the
        // bar stays inside the 3D viewport rather than spanning the editor.
        const float bar_w = vp_size.x * 0.5f;
        const float bar_x = vp_pos.x + (vp_size.x - bar_w) * 0.5f;
        ImGui::SetNextWindowPos(
            ImVec2(bar_x, vp_pos.y + 6.0f),
            ImGuiCond_Always);
        ImGui::SetNextWindowSize(
            ImVec2(bar_w, bar_h),
            ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.80f);
        if (ImGui::Begin("##llm_progress_bar", nullptr,
            ImGuiWindowFlags_NoTitleBar  |
            ImGuiWindowFlags_NoResize    |
            ImGuiWindowFlags_NoMove      |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav))
        {
            // Progress fraction — cap at 1.0 so overshoots don't
            // render past the bar end.  When estimated_total is 0
            // (kick-off frame), show an "indeterminate" feel by
            // forcing 0.05 so the bar is visibly alive.
            float frac = 0.0f;
            char overlay[96] = {0};
            if (classifier_status_ == ClassifierStatus::Ready) {
                frac = 1.0f;
                std::snprintf(overlay, sizeof(overlay),
                    "%d materials + %d objects classified",
                    classifier_mats_done_, classifier_objs_done_);
            } else if (classifier_status_ == ClassifierStatus::Failed) {
                frac = 1.0f;
                std::snprintf(overlay, sizeof(overlay), "FAILED");
            } else {
                if (classifier_bytes_estimated_total_ > 0) {
                    frac = float(classifier_bytes_received_) /
                           float(classifier_bytes_estimated_total_);
                    if (frac > 0.99f) frac = 0.99f;  // never claim
                                                      // done until
                                                      // status flips
                                                      // to Ready
                }
                if (frac < 0.02f) frac = 0.02f;  // visible kick-off
                std::snprintf(overlay, sizeof(overlay),
                    "%zu / %zu items",
                    classifier_bytes_received_,
                    classifier_bytes_estimated_total_);
            }

            // Colour the bar fill the same way the text banner is
            // coloured (amber / green / red) so the two overlays
            // read as one widget.
            ImVec4 bar_col;
            switch (classifier_status_) {
                case ClassifierStatus::Ready:
                    bar_col = ImVec4(0.31f, 0.86f, 0.47f, 1.0f); break;
                case ClassifierStatus::Failed:
                    bar_col = ImVec4(1.0f,  0.35f, 0.35f, 1.0f); break;
                default:
                    bar_col = ImVec4(1.0f,  0.78f, 0.31f, 1.0f); break;
            }
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_col);
            ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f), overlay);
            ImGui::PopStyleColor();

            // Summary line below the bar.
            const char* status_word =
                classifier_status_ == ClassifierStatus::Ready  ? "ready"   :
                classifier_status_ == ClassifierStatus::Failed ? "failed"  :
                                                                  "classifying";
            ImGui::Text(
                "LLM: %s — %.1fs elapsed, %d mats + %d objs classified",
                status_word, classifier_elapsed_s_,
                classifier_mats_done_, classifier_objs_done_);
        }
        ImGui::End();
    }

    // ── Collision-mesh build progress bar (stacked below the LLM bar) ─
    // Second top-pinned overlay, same width/centering as the classifier
    // bar but pinned one row lower so the two read as a stacked pair.
    // Driven by setCollisionBuildStatus() (done/total primitives + the
    // running Floor-mesh count), pushed every frame by application.cpp
    // while the collision world builds incrementally.  Shown while
    // Building and after Done (full green, as confirmation); hidden in
    // Idle so the pre-classify phase stays clean.
    // Only while actively building; once Done, hide it (and the classifier
    // bar above) so both vanish the moment the simplified-mesh build ends.
    if (collision_build_status_ == CollisionBuildStatus::Building) {
        ImVec2 vp_pos, vp_size, vp_c;
        getViewportScreenRect(vp_pos, vp_size, vp_c);
        const float bar_h = 44.0f;
        const float bar_w = vp_size.x * 0.5f;
        const float bar_x = vp_pos.x + (vp_size.x - bar_w) * 0.5f;
        // Stack one bar-height + a small gap below the LLM bar (which sits at
        // the viewport top), so the two pills don't overlap.
        const float gap = 6.0f;
        const float bar_y = vp_pos.y + 6.0f + bar_h + gap;
        ImGui::SetNextWindowPos(ImVec2(bar_x, bar_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(bar_w, bar_h), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.80f);
        if (ImGui::Begin("##collision_build_bar", nullptr,
            ImGuiWindowFlags_NoTitleBar  |
            ImGuiWindowFlags_NoResize    |
            ImGuiWindowFlags_NoMove      |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav))
        {
            // Fraction = primitives processed / total queued.  Force a
            // visible sliver at kick-off and a full bar once Done.
            float frac = 0.0f;
            if (collision_build_total_ > 0) {
                frac = float(collision_build_done_) /
                       float(collision_build_total_);
                if (frac > 1.0f) frac = 1.0f;
            }
            if (collision_build_status_ == CollisionBuildStatus::Done) {
                frac = 1.0f;
            } else if (frac < 0.02f) {
                frac = 0.02f;
            }

            char overlay[96] = {0};
            std::snprintf(overlay, sizeof(overlay),
                "%zu / %zu primitives",
                collision_build_done_, collision_build_total_);

            // Blue while building, green when done — distinct from the
            // LLM bar's amber so the two bars are easy to tell apart.
            ImVec4 bar_col =
                (collision_build_status_ == CollisionBuildStatus::Done)
                    ? ImVec4(0.31f, 0.86f, 0.47f, 1.0f)   // green
                    : ImVec4(0.30f, 0.68f, 0.95f, 1.0f);  // blue
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_col);
            ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f), overlay);
            ImGui::PopStyleColor();

            const char* word =
                (collision_build_status_ == CollisionBuildStatus::Done)
                    ? "ready" : "building";
            ImGui::Text(
                "Collision: %s — %zu floor mesh(es) from %zu/%zu prims",
                word, collision_build_meshes_,
                collision_build_done_, collision_build_total_);
        }
        ImGui::End();
    }

    // ── "No collision map" banner ────────────────────────────────────
    // Pushed per-frame by the application: play mode wants to walk the
    // player, but no collision world exists (no baked .rwcmap on the
    // scene, no bake run) — without it the player never spawns.  Point
    // the user at the bake action instead of failing silently.
    if (no_collision_map_warning_) {
        ImVec2 vp_pos, vp_size, vp_c;
        getViewportScreenRect(vp_pos, vp_size, vp_c);
        const float bar_h = 44.0f;
        const float bar_w = vp_size.x * 0.5f;
        const float bar_x = vp_pos.x + (vp_size.x - bar_w) * 0.5f;
        const float bar_y = vp_pos.y + 6.0f;
        ImGui::SetNextWindowPos(ImVec2(bar_x, bar_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(bar_w, bar_h), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.80f);
        if (ImGui::Begin("##no_collision_map_banner", nullptr,
            ImGuiWindowFlags_NoTitleBar  |
            ImGuiWindowFlags_NoResize    |
            ImGuiWindowFlags_NoMove      |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav))
        {
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(1.0f, 0.72f, 0.25f, 1.0f));
            ImGui::TextUnformatted(
                "No collision map — player can't spawn/walk");
            ImGui::PopStyleColor();
            ImGui::TextUnformatted(
                "Tools > Bake Collision Map (Player Walk), then Save Scene");
        }
        ImGui::End();
    }

    // ── Isolate-mesh identity overlay ────────────────────────────────
    // When the isolate-debug slider is on, show the currently-isolated
    // collision mesh's identity (index / category / tris / material /
    // object), pushed each frame by the app.  Pinned top-left so it
    // doesn't fight the centred progress bars.  Lets you read off the
    // index of the broken mesh while scrubbing the slider.
    if (collision_isolate_enabled_ && !collision_isolate_info_.empty()) {
        ImVec2 vp_pos, vp_size, vp_c;
        getViewportScreenRect(vp_pos, vp_size, vp_c);
        // Pinned to the viewport's top-left, pushed DOWN below the clock
        // widget so they don't overlap.  Left/Right arrows scrub the index.
        ImGui::SetNextWindowPos(
            ImVec2(vp_pos.x + 12.0f, vp_pos.y + 150.0f),
            ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.85f);
        if (ImGui::Begin("##coll_isolate_info", nullptr,
            ImGuiWindowFlags_NoTitleBar  |
            ImGuiWindowFlags_NoResize    |
            ImGuiWindowFlags_NoMove      |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav))
        {
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(1.0f, 0.85f, 0.30f, 1.0f));
            ImGui::TextUnformatted("ISOLATED COLLISION MESH");
            ImGui::PopStyleColor();
            ImGui::TextUnformatted(collision_isolate_info_.c_str());
        }
        ImGui::End();
    }

    // (The earlier centred text-only banner was superseded by the
    // top-pinned ProgressBar window above; same data, denser layout,
    // can't be hidden by other windows.)

    std::vector<er::ClearValue> clear_values;
    clear_values.resize(2);
    clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

    float menu_height = ImGui::GetFrameHeight(); // Gets the height of the menu bar

    // ---- Title-screen elements (background + top bar + menu items) ----
    // Only drawn when we are on the title screen or still loading.
    // Once InGame, the background and title elements are hidden.
    const bool show_title_ui =
        (game_state_ == GameState::TitleScreen ||
         game_state_ == GameState::Loading);

    // Full-screen fantasy landscape background.
    // Use the background draw list (behind all windows) so the image
    // is never clipped by any ImGui window's client rect and always
    // covers the entire framebuffer.
    if (show_title_ui && bg_enabled_ && bg_texture_id_ != ImTextureID(0)) {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        {
            ImDrawList* dl = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
            const ImVec2 p0 = vp->Pos;
            const ImVec2 p1 = ImVec2(vp->Pos.x + vp->Size.x,
                                     vp->Pos.y + vp->Size.y);
            dl->AddImage(bg_texture_id_, p0, p1);

            // Twinkle overlay — 4-point diffraction spikes + core glow.
            // Drawn with standard AddLine / AddCircleFilled which
            // handle the texture switch from the bg image correctly.
            // Each spike fades by drawing 2–3 layered lines from thick
            // bright to thin dim, simulating a soft taper.
            if (!detected_stars_.empty()) {
                const float t = (float)ImGui::GetTime();
                const float scr_w = vp->Size.x;
                const float scr_h = vp->Size.y;

                for (const auto& s : detected_stars_) {
                    const float pulse = 0.5f + 0.5f * std::sin(t * s.speed + s.phase);
                    const ImVec2 c = {
                        vp->Pos.x + s.nx * scr_w,
                        vp->Pos.y + s.ny * scr_h };

                    const uint8_t cr = (uint8_t)(230 + s.brightness * 25);
                    const uint8_t cg = (uint8_t)(225 + s.brightness * 20);
                    const uint8_t cb = (uint8_t)(200 + s.brightness * 15);

                    // Short subtle cross spikes — kept small so they don't
                    // stray beyond the sky region into the landscape.
                    const float spike = s.radius * (1.5f + pulse * 3.0f);
                    const uint8_t spike_a = (uint8_t)(30 + pulse * 90);
                    const ImU32 spike_col = IM_COL32(cr, cg, cb, spike_a);
                    dl->AddLine(ImVec2(c.x - spike, c.y), ImVec2(c.x + spike, c.y), spike_col, 1.0f);
                    dl->AddLine(ImVec2(c.x, c.y - spike), ImVec2(c.x, c.y + spike), spike_col, 1.0f);

                    // Bright core dot.
                    const uint8_t core_a = (uint8_t)(80 + pulse * 175);
                    dl->AddCircleFilled(c, s.radius * (0.5f + pulse * 0.5f),
                        IM_COL32(cr, cg, cb, core_a), 10);
                }
            }
        }
    }

    // Twilight backdrop + title banner — text now driven by XML config.
    if (show_title_ui) {
        // Span the editor Viewport (or full window outside editor mode) so the
        // title header sits at the top of the 3D view, correctly sized.
        ImVec2 vp_pos, vp_sz, vp_c;
        getViewportScreenRect(vp_pos, vp_sz, vp_c);
        const float vp_w = vp_sz.x > 0.0f ? vp_sz.x : float(screen_size.x);

        const char* t_title    = title_config_.loaded ? title_config_.title.c_str()    : "Ashes of Eldra";
        const char* t_subtitle = title_config_.loaded ? title_config_.subtitle.c_str() : "chronicles of the fallen realm";
        const char* t_version  = title_config_.loaded ? title_config_.version.c_str()  : "v0.4.2  pre-alpha";
        const float t_extra    = title_config_.loaded ? title_config_.top_bar_height_extra : 72.0f;

        drawFantasyTopBackdrop(
            vp_pos,
            vp_w,
            menu_height + t_extra,
            menu_height,
            t_title,
            t_subtitle,
            t_version);
    }

    // ---- Title-screen main menu (centred vertical list) ----
    if (game_state_ == GameState::TitleScreen && title_config_.loaded &&
        !title_config_.menu_items.empty()) {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        // Centre the menu in the editor Viewport (or full window otherwise).
        ImVec2 mvp_pos, mvp_sz, mvp_c;
        getViewportScreenRect(mvp_pos, mvp_sz, mvp_c);

        // Menu block: centred horizontally, lower-third vertically.
        const float item_w = 320.0f;
        const float item_h = 48.0f;
        const float spacing = 12.0f;
        const int   n_items = (int)title_config_.menu_items.size();
        const float block_h = n_items * item_h + (n_items - 1) * spacing;
        const float start_x = mvp_pos.x + (mvp_sz.x - item_w) * 0.5f;
        const float start_y = mvp_pos.y + mvp_sz.y * 0.52f;

        ImGui::SetNextWindowViewport(vp->ID);
        ImGui::SetNextWindowPos(ImVec2(start_x, start_y));
        ImGui::SetNextWindowSize(ImVec2(item_w, block_h + 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));

        ImGui::Begin(
            "##title_menu",
            nullptr,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_AlwaysAutoResize);

        // Fantasy-styled buttons with staggered glow pulse.
        // Each button gets its own phase offset so they shimmer
        // one after another like runes lighting up in sequence.
        static float s_title_time = 0.0f;
        s_title_time += delta_t;

        const ImVec4 btn_bg_base   = ImVec4(0.08f, 0.07f, 0.16f, 0.80f);
        const ImVec4 btn_hovered   = ImVec4(0.55f, 0.45f, 0.22f, 0.90f);
        const ImVec4 btn_active    = ImVec4(0.83f, 0.69f, 0.35f, 1.00f);
        const ImVec4 text_base     = ImVec4(0.93f, 0.88f, 0.72f, 1.00f);

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);

        for (int i = 0; i < n_items; ++i) {
            const auto& mi = title_config_.menu_items[i];

            // Staggered sine pulse: each item is offset by ~0.7 s
            // so the glow cascades down the menu list.
            const float phase = s_title_time * 2.2f - float(i) * 1.5f;
            const float pulse = 0.5f + 0.5f * std::sin(phase);

            // Interpolate button bg from deep indigo → dim gold.
            const ImVec4 btn_bg = ImVec4(
                btn_bg_base.x + pulse * 0.12f,
                btn_bg_base.y + pulse * 0.10f,
                btn_bg_base.z + pulse * 0.04f,
                btn_bg_base.w);

            // Border brightens with the pulse.
            const ImVec4 btn_border = ImVec4(
                0.55f + pulse * 0.28f,
                0.45f + pulse * 0.24f,
                0.22f + pulse * 0.13f,
                0.40f + pulse * 0.40f);

            // Text glows from cream to bright gold.
            const ImVec4 text_col = ImVec4(
                text_base.x + pulse * 0.07f,
                text_base.y + pulse * 0.04f,
                text_base.z - pulse * 0.12f,
                text_base.w);

            ImGui::PushStyleColor(ImGuiCol_Button,        btn_bg);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  btn_hovered);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,   btn_active);
            ImGui::PushStyleColor(ImGuiCol_Border,         btn_border);
            ImGui::PushStyleColor(ImGuiCol_Text,           text_col);

            if (ImGui::Button(mi.label.c_str(), ImVec2(item_w, item_h))) {
                // Dispatch action.
                if (mi.action == "new_game") {
                    new_game_mesh_requests_ = title_config_.new_game_meshes;
                    new_game_requested_ = true;
                    game_state_ = GameState::Loading;
                }
                else if (mi.action == "quit") {
                    // Request window close.
                    if (auto* win = ImGui::GetIO().BackendPlatformUserData) {
                        // Fallback: set GLFW should-close via the io ctx.
                    }
                    // We'll let application poll this via game state.
                }
                else if (mi.action == "load_game") {
                    // TODO: implement load game
                }
                else if (mi.action == "settings") {
                    // TODO: implement settings
                }
            }
            ImGui::PopStyleColor(5); // per-button colors

            if (i < n_items - 1)
                ImGui::Dummy(ImVec2(0, spacing - ImGui::GetStyle().ItemSpacing.y));
        }

        ImGui::PopStyleVar(2);   // FrameRounding, FrameBorderSize
        ImGui::End();
        ImGui::PopStyleColor();  // WindowBg
        ImGui::PopStyleVar(3);   // WindowPadding, BorderSize, Rounding
    }

    cmd_buf->beginRenderPass(
        render_pass,
        framebuffer,
        screen_size,
        clear_values);

    static bool s_select_load_gltf = false;
    static bool s_show_skydome = false;
    static bool s_show_weather = false;
    static bool s_show_shader_error_message = false;
    static std::string s_shader_error_message;
    static bool s_show_gpu_profiler = false;
    bool compile_shaders = false;

    bool test_true = true;
    // Place the FPS widget at the top-right of the VIEWPORT (editor mode) or
    // the full window (otherwise).  The viewport rect already sits below the
    // menu bar, so only the full-window fallback needs the menu_height offset.
    ImVec2 fps_vp_pos, fps_vp_size, fps_vp_c;
    getViewportScreenRect(fps_vp_pos, fps_vp_size, fps_vp_c);
    const float fps_top_pad = isViewportValid() ? 8.0f : menu_height;
    // Standalone FPS box only in game mode; in editor mode FPS is merged into
    // the top-right VRAM HUD panel (see Menu::draw's profiler overlay).
    if (!editor_enabled_) {
    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
    ImGui::Begin(
        "fps",
        &test_true,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoDocking);
    ImGui::SetWindowPos(ImVec2(
        fps_vp_pos.x + fps_vp_size.x - 220.0f, fps_vp_pos.y + fps_top_pad));
    ImGui::SetWindowSize(ImVec2((float)210, (float)47));
    ImGui::BeginChild("fps", ImVec2(0, 0), false);
    // Rolling average over the last 60 frames for a stable reading.
    static float s_fps_history[60] = {};
    static int   s_fps_idx = 0;
    float instant_fps = delta_t > 0.0f ? 1.0f / delta_t : 0.0f;
    s_fps_history[s_fps_idx] = instant_fps;
    s_fps_idx = (s_fps_idx + 1) % 60;
    float fps_sum = 0.0f;
    for (int i = 0; i < 60; ++i) fps_sum += s_fps_history[i];
    float fps = fps_sum / 60.0f;
    ImGui::Text("fps : %8.2f", fps);
    ImGui::EndChild();
    ImGui::End();
    }

    // (Title banner + version stamp now drawn inline inside the
    // twilight backdrop window earlier in this frame — see the call
    // to drawFantasyTopBackdrop() above.)

    // Async mesh-load HUD — dual counter-rotating rune ring + filename
    // list. Sits just under the fps widget and stays hidden when
    // nothing is loading. NoInputs so it can't steal clicks from the
    // menu. See drawRuneLoader() above for the ring geometry; the
    // aesthetic mirrors the loader in
    // realworld/design/fantasy_menu.html.
    if (mesh_load_task_manager_ || !scene_place_pending_.empty() ||
        vt_warm_active_) {
        std::vector<std::string> in_flight;
        if (mesh_load_task_manager_) {
            in_flight = mesh_load_task_manager_->inFlightFilenames();
        }
        // Scene placements still streaming in (drag & drop / Add to
        // Scene) spin the same rune ring, listed alongside the raw mesh
        // loads — so dropping an object always gives immediate feedback,
        // even when its source dedups onto an already-in-flight load.
        const size_t raw_loads = in_flight.size();
        for (const auto& n : scene_place_pending_) {
            in_flight.push_back(n);
        }
        if (!in_flight.empty() || vt_warm_active_) {
            // Cap the displayed list so a large batch load doesn't
            // blow past the screen edge; the count shown beside the
            // loader still reflects the full set.
            constexpr size_t kMaxListed = 6;

            // Time-driven rotation so the cadence is stable across
            // framerates. Main thread only, so the static is safe.
            static float s_elapsed = 0.0f;
            s_elapsed += delta_t;

            const float kLoaderSize = 52.0f;
            // Mesh-load HUD: top-right of the viewport (editor) or window.
            ImVec2 mlv_pos, mlv_size, mlv_c;
            getViewportScreenRect(mlv_pos, mlv_size, mlv_c);
            const float mlv_top = isViewportValid() ? 40.0f
                                                    : (menu_height + 40.0f);
            const ImVec2 mesh_win_pos = {
                mlv_pos.x + mlv_size.x - 380.0f,
                mlv_pos.y + mlv_top };

            ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
            ImGui::SetNextWindowPos(mesh_win_pos);
            ImGui::Begin(
                "mesh_loads",
                nullptr,
                ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_AlwaysAutoResize);

            // Rune ring is drawn into this window's own draw list
            // (not a foreground list) so it renders via the exact
            // same path as the filename text below — proven visible.
            // Reserve a square of kLoaderSize first, then paint the
            // ring centered inside it. SameLine() after puts the
            // text flush to its right.
            {
                const ImVec2 slot_tl = ImGui::GetCursorScreenPos();
                const ImVec2 loader_c = {
                    slot_tl.x + kLoaderSize * 0.5f,
                    slot_tl.y + kLoaderSize * 0.5f };
                drawRuneLoader(
                    ImGui::GetWindowDrawList(),
                    loader_c,
                    kLoaderSize * 0.45f,
                    s_elapsed);
            }
            ImGui::Dummy(ImVec2(kLoaderSize, kLoaderSize));

            // Title + first filename sit to the right of the rings.
            ImGui::SameLine();
            ImGui::BeginGroup();
            if (in_flight.empty()) {
                // Loads done — only the VT texture warm-up remains.
                ImGui::TextColored(
                    ImVec4(0.95f, 0.86f, 0.60f, 1.0f),
                    "Preparing textures");
            } else if (raw_loads == 0) {
                // Pure placement wait (source already loading elsewhere
                // or shared) — word it as what the user just did.
                ImGui::TextColored(
                    ImVec4(0.95f, 0.86f, 0.60f, 1.0f),
                    "Placing %zu object%s",
                    in_flight.size(),
                    in_flight.size() == 1 ? "" : "s");
            } else {
                ImGui::TextColored(
                    ImVec4(0.95f, 0.86f, 0.60f, 1.0f),
                    "Loading %zu mesh%s",
                    in_flight.size(),
                    in_flight.size() == 1 ? "" : "es");
            }
            // Inline the first stem so a single-mesh load reads as one
            // self-contained line ("Loading 1 mesh\nbistro.fbx").
            if (!in_flight.empty()) {
                const auto& name = in_flight[0];
                auto pos = name.find_last_of("/\\");
                std::string stem =
                    pos == std::string::npos ? name : name.substr(pos + 1);
                ImGui::TextColored(
                    ImVec4(0.70f, 0.65f, 0.50f, 1.0f),
                    "%s", stem.c_str());
            }
            ImGui::EndGroup();

            // Remaining filenames below as bulleted list, capped.
            if (in_flight.size() > 1) {
                ImGui::Separator();
                for (size_t i = 1;
                     i < in_flight.size() && i < kMaxListed;
                     ++i) {
                    const auto& name = in_flight[i];
                    auto pos = name.find_last_of("/\\");
                    std::string stem = pos == std::string::npos
                        ? name : name.substr(pos + 1);
                    ImGui::BulletText("%s", stem.c_str());
                }
                if (in_flight.size() > kMaxListed) {
                    ImGui::BulletText("... (+%zu more)",
                        in_flight.size() - kMaxListed);
                }
            }

            // ── VT warm-up progress bar ───────────────────────────────
            // Placed objects' textures being BC7-encoded into the
            // Virtual Texture pool (one per frame) — objects render
            // flat until this finishes and the cluster bake snaps in.
            if (vt_warm_active_) {
                ImGui::Separator();
                ImGui::TextColored(
                    ImVec4(0.70f, 0.65f, 0.50f, 1.0f),
                    "%s", vt_warm_label_.c_str());
                ImGui::ProgressBar(vt_warm_frac_, ImVec2(280.0f, 0.0f));
            }
            ImGui::End();
        }
    }

// turn off ray tracing view window.
#if 0
    // Define the rectangle's position and size

     // Set up the size for the first child window
    ImVec2 child1_size(300, 200); // Initial size of the child window (width, height)
    ImGui::BeginChild("Child Window 1", child1_size, true); // 'true' enables the border
    float aspect = (float)screen_size.y / (float)screen_size.x;
    drawViewport(
        rt_texture_id_,
        ImVec2(0, menu_height),
        ImVec2(screen_size.x / 2.0f, screen_size.y / 2.0f),
        aspect);
    ImGui::EndChild();

    // Add some spacing between the sub-windows
    ImGui::Spacing();

    // Set up the size for the second child window
    ImVec2 child2_size(300, 200);
    ImGui::BeginChild("Child Window 2", child2_size, true);
    ImGui::Text("This is another adjustable sub-window.");
    ImGui::EndChild();

#endif
    // The dev menu bar is only visible once gameplay has started (or
    // during mesh loading). On the title screen the XML-driven menu
    // items are shown instead.
    if (game_state_ != GameState::TitleScreen && ImGui::BeginMainMenuBar())
    {
        // ── Scene menu (first — the editor's primary menu) ────────────────
        if (ImGui::BeginMenu("Scene"))
        {
            ImGui::InputText("Name", scene_name_buf_, sizeof(scene_name_buf_));
            ImGui::Separator();
            // Create Scene mode: empties the current scene and shows a flat
            // reference grid (200 x 200 m, 1 m spacing, Y up) to build on.
            if (ImGui::MenuItem("Create Scene")) {
                scene_create_request_ = true;
                show_scene_grid_      = true;
                scene_node_active_    = true;   // surface the scene node in
                                                // the Outliner immediately
            }
            // Camera creation lives on the viewport right-click menu
            // ("Add Camera Object") — no Scene-menu duplicate.
            ImGui::MenuItem("Show Grid", NULL, &show_scene_grid_);
            ImGui::Separator();
            // Model import happens through the Content Browser (Import
            // button / drag-and-drop) — no Scene-menu duplicate.
            if (ImGui::MenuItem("Save Scene"))      { scene_save_request_   = true; }
            if (ImGui::MenuItem("Load Scene..."))   { scene_load_request_   = true; }
            // Recent Scenes: quick reload of recently opened scenes without the
            // OS file dialog.  Entries show the file's basename; the full path
            // is sent back via recent_scene_load_request_.
            if (ImGui::BeginMenu("Recent Scenes")) {
                if (recent_scenes_.empty()) {
                    ImGui::MenuItem("(none)", NULL, false, /*enabled=*/false);
                } else {
                    for (const std::string& path : recent_scenes_) {
                        const std::string label =
                            std::filesystem::path(path).stem().string();
                        if (ImGui::MenuItem(label.c_str())) {
                            recent_scene_load_request_ = path;
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("%s", path.c_str());
                        }
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        // ── Rendering menu ────────────────────────────────────────────────
        // Render-path + environment rendering options under one roof:
        // raytracing, terrain passes, skydome, weather system.
        if (ImGui::BeginMenu("Rendering"))
        {
            if (ImGui::BeginMenu("Raytracing"))
            {
                if (ImGui::MenuItem("Turn off ray tracing", NULL,
                                    turn_off_ray_tracing_)) {
                    turn_off_ray_tracing_ = !turn_off_ray_tracing_;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Terrain"))
            {
                if (ImGui::MenuItem("Turn off water pass", NULL,
                                    turn_off_water_pass_)) {
                    turn_off_water_pass_ = !turn_off_water_pass_;
                }
                if (ImGui::MenuItem("Turn off grass pass", NULL,
                                    turn_off_grass_pass_)) {
                    turn_off_grass_pass_ = !turn_off_grass_pass_;
                }
                ImGui::EndMenu();
            }
            // Plain CLICK items (not submenus): as submenus they auto-
            // opened on hover, popping the Skydome / Weather windows the
            // moment the mouse passed over them on the way to another
            // entry.  The checkmark mirrors the window's open state.
            if (ImGui::MenuItem("Skydome", NULL, s_show_skydome)) {
                s_show_skydome = !s_show_skydome;
            }
            if (ImGui::MenuItem("Weather System", NULL, s_show_weather)) {
                s_show_weather = !s_show_weather;
            }
            if (ImGui::BeginMenu("Shadow"))
            {
                if (ImGui::MenuItem("Turn off shadow pass", NULL,
                                    turn_off_shadow_pass_)) {
                    turn_off_shadow_pass_ = !turn_off_shadow_pass_;
                }

                // ── Shadow technique ─────────────────────────────────
                // CSM (default): cascaded shadow maps, rendered by the
                // shadow pass and sampled everywhere.
                // Screen-space raytraced (software): NO shadow maps at
                // all — deferred_resolve.comp ray-marches the G-buffer
                // depth toward the sun per pixel.  Experimental / for
                // perf comparison: only deferred (cluster) pixels get
                // shadows, and only on-screen geometry can cast them.
                // Compare "CSM Shadow" vs "Deferred Resolve Compute" in
                // the GPU profiler to read the cost delta.
                if (ImGui::BeginMenu("Shadow technique")) {
                    const ShadowTechnique cur = shadow_technique_;
                    if (ImGui::MenuItem("CSM (cascaded shadow maps)", NULL,
                                        cur == ShadowTechnique::kCsm)) {
                        shadow_technique_ = ShadowTechnique::kCsm;
                    }
                    if (ImGui::MenuItem(
                            "Screen-space raytraced (software, experimental)",
                            NULL, cur == ShadowTechnique::kSsrt)) {
                        shadow_technique_ = ShadowTechnique::kSsrt;
                    }
                    // TRUE software RT: per-pixel rays traced against the
                    // cluster BVH in world space — off-screen casters
                    // work.  Needs the cluster path finalized; the app
                    // falls back to unshadowed until the BVH is built.
                    if (ImGui::MenuItem(
                            "World-space raytraced (software BVH, experimental)",
                            NULL, cur == ShadowTechnique::kRtBvh)) {
                        shadow_technique_ = ShadowTechnique::kRtBvh;
                    }
                    // HARDWARE RT: ray queries against a TLAS over the
                    // same cluster geometry (alpha-cutoff casters live in
                    // a non-opaque BLAS geometry and are texture-tested).
                    // Falls back to unshadowed until the AS is built.
                    if (ImGui::MenuItem(
                            "World-space raytraced (hardware, ray query)",
                            NULL, cur == ShadowTechnique::kHwRt)) {
                        shadow_technique_ = ShadowTechnique::kHwRt;
                    }
                    // ReSTIR DI: the sun becomes an AREA light (soft
                    // penumbras) and authored point lights (Outliner →
                    // Add Point Light) shade with ONE shadow ray/pixel
                    // via reservoir temporal reuse.  Uses the HW ray-
                    // query backend when available, else the software
                    // BVH.
                    if (ImGui::MenuItem(
                            "ReSTIR (soft sun + point lights, experimental)",
                            NULL, cur == ShadowTechnique::kRestir)) {
                        shadow_technique_ = ShadowTechnique::kRestir;
                    }
                    ImGui::EndMenu();
                }

                // Edge-aware bilateral filter over the RT shadow + AO
                // (kills the per-pixel dither grain).  RT techniques
                // only; ReSTIR excluded.
                if (ImGui::MenuItem("Smooth RT shadow/AO", NULL,
                                    rt_smoothing_)) {
                    rt_smoothing_ = !rt_smoothing_;
                }

                // CSM silhouette prepass — see the member field comment in
                // menu.h and csm_silhouette_prepass.mesh's header.  Off →
                // the shadow pass clears depth to 1.0 (legacy) and skips
                // the prepass dispatch, for A/B timing comparisons.
                if (ImGui::MenuItem("CSM silhouette prepass", NULL,
                                     csm_silhouette_prepass_enabled_)) {
                    csm_silhouette_prepass_enabled_ =
                        !csm_silhouette_prepass_enabled_;
                }

                // CSM drawable-shadow draw mode — three mutually-exclusive
                // picks for how the drawable shadow path amplifies geometry
                // across the cascades (see the enum comment in menu.h).
                // The cluster shadow path is unaffected (always task+mesh).
                if (ImGui::BeginMenu("Drawable shadow draw mode")) {
                    const CsmDrawMode cur = csm_draw_mode_;
                    if (ImGui::MenuItem("Regular (per-cascade passes)", NULL,
                                         cur == CsmDrawMode::kRegular)) {
                        csm_draw_mode_ = CsmDrawMode::kRegular;
                    }
                    if (ImGui::MenuItem("Geometry shader (layered)", NULL,
                                         cur == CsmDrawMode::kGeometryShader)) {
                        csm_draw_mode_ = CsmDrawMode::kGeometryShader;
                    }
                    if (ImGui::MenuItem("Mesh shader (task+mesh)", NULL,
                                         cur == CsmDrawMode::kMeshShader)) {
                        csm_draw_mode_ = CsmDrawMode::kMeshShader;
                    }
                    ImGui::EndMenu();
                }

                ImGui::Separator();

                if (ImGui::MenuItem("Debug Cascades", NULL, show_csm_debug_)) {
                    show_csm_debug_ = !show_csm_debug_;
                }

                ImGui::Separator();

                if (ssao_) {
                    if (ImGui::MenuItem("SSAO Enabled", NULL, ssao_->enabled)) {
                        ssao_->enabled = !ssao_->enabled;
                    }
                    ImGui::SliderFloat("AO Radius",    &ssao_->radius,      0.01f, 5.0f,  "%.3f");
                    ImGui::SliderFloat("AO Bias",      &ssao_->bias,        0.001f, 0.1f, "%.4f");
                    ImGui::SliderFloat("AO Power",     &ssao_->power,       0.5f,  4.0f,  "%.2f");
                    ImGui::SliderFloat("AO Intensity", &ssao_->intensity,   0.1f,  3.0f,  "%.2f");
                    ImGui::SliderFloat("AO Strength",  &ssao_->strength,    0.0f,  1.0f,  "%.2f");
                    ImGui::SliderInt("AO Samples",     &ssao_->kernel_size, 4,     64);
                }

                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("IBL Debug")) {
                if (ImGui::MenuItem("Open IBL / Sky Debug")) {
                    show_ibl_debug_ = true;
                }
                ImGui::EndMenu();
            }
            // Smart Mesh (cluster-rendering stats/controls floating window).
            if (cluster_renderer_ &&
                ImGui::MenuItem("Smart Mesh", NULL,
                                show_smart_mesh_window_)) {
                show_smart_mesh_window_ = !show_smart_mesh_window_;
            }
            // Render Debug — debug visualisation modes, pipeline toggle,
            // viewers, glass mode (see drawRenderDebugMenuContent()).
            if (ImGui::BeginMenu("Render Debug")) {
                drawRenderDebugMenuContent();
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        // ── Audio menu ────────────────────────────────────────────────────
        // Bus volumes + the text-to-voice picker.  The voice list is the
        // installed sherpa-onnx voices under assets/ml_models/tts; selecting one
        // switches live and re-speaks the last dialog line so you can audition
        // them one by one.
        if (ImGui::BeginMenu("Audio"))
        {
            namespace au = engine::audio;

            // Master-ish bus volumes (0..1).  Stored in the engine so they
            // persist across clips played this session.
            float vMusic = au::AudioEngine::busVolume(
                au::AudioEngine::Bus::kMusic);
            if (ImGui::SliderFloat("Music##bus", &vMusic, 0.0f, 1.0f, "%.2f"))
                au::AudioEngine::setBusVolume(au::AudioEngine::Bus::kMusic,
                                              vMusic);
            float vSfx = au::AudioEngine::busVolume(
                au::AudioEngine::Bus::kSfx);
            if (ImGui::SliderFloat("SFX##bus", &vSfx, 0.0f, 1.0f, "%.2f"))
                au::AudioEngine::setBusVolume(au::AudioEngine::Bus::kSfx,
                                              vSfx);
            float vVoice = au::AudioEngine::busVolume(
                au::AudioEngine::Bus::kVoice);
            if (ImGui::SliderFloat("Voice##bus", &vVoice, 0.0f, 1.0f, "%.2f"))
                au::AudioEngine::setBusVolume(au::AudioEngine::Bus::kVoice,
                                              vVoice);

            ImGui::Separator();
            if (ImGui::BeginMenu("Voice (text-to-speech)"))
            {
                // Cache the voice list — a directory scan every frame the
                // menu is open is wasteful.  Rebuilt when first opened.
                if (tts_voice_cache_.empty())
                    tts_voice_cache_ = au::TtsEngine::listVoices();
                const std::string cur = au::TtsEngine::currentVoice();

                if (tts_voice_cache_.empty()) {
                    ImGui::TextDisabled("(no voices installed)");
                    ImGui::TextDisabled("Get more:  Setup.bat -tts-voice=all");
                    ImGui::TextDisabled("or  tools/tts/download_voices.py --all");
                } else {
                    ImGui::TextDisabled("%zu voice(s) — click to audition",
                                        tts_voice_cache_.size());
                    // Filter box (type to narrow a long list, e.g. "en_US").
                    ImGui::SetNextItemWidth(280.0f);
                    ImGui::InputTextWithHint("##voice_filter", "filter...",
                        tts_voice_filter_, sizeof(tts_voice_filter_));
                    std::string flt = tts_voice_filter_;
                    for (auto& c : flt)
                        c = (char)std::tolower((unsigned char)c);

                    // Scrollable list so 100+ voices stay on-screen.  Use
                    // Selectables (not MenuItems) so picking one does NOT
                    // close the popup — you can audition several in a row.
                    ImGui::BeginChild("##voice_list", ImVec2(300.0f, 360.0f),
                                      true);
                    for (const auto& v : tts_voice_cache_) {
                        if (!flt.empty()) {
                            std::string lv = v;
                            for (auto& c : lv)
                                c = (char)std::tolower((unsigned char)c);
                            if (lv.find(flt) == std::string::npos) continue;
                        }
                        const bool sel = (v == cur);
                        if (ImGui::Selectable(v.c_str(), sel))
                            au::TtsEngine::setVoice(v);
                    }
                    ImGui::EndChild();

                    if (ImGui::Button("Repeat last line"))
                        au::TtsEngine::repeatLast();
                    ImGui::SameLine();
                    if (ImGui::Button("Stop"))
                        au::TtsEngine::stop();
                }
                ImGui::EndMenu();
            } else {
                // Menu closed — drop the cache so it refreshes next open
                // (picks up voices added to the folder mid-session).
                tts_voice_cache_.clear();
            }
            ImGui::EndMenu();
        }

        // ("Game Objects" menu removed — player objects are now authored
        // scene objects: right-click the viewport → "Add Player Object".)

        // ── Physics menu ─────────────────────────────────────────────────
        // Collects toggles related to the static-mesh CollisionWorld and
        // any future physics-driven bits (rigid bodies, debug draws,
        // pause/step). Entries here drive engine state, not just UI.
        if (ImGui::BeginMenu("Physics"))
        {
            // Collision-mesh debug visualisation. Equivalent to pressing
            // F1; the application reads `isCollisionDebugOn()` each
            // frame and swaps in CollisionWorld::drawDebug for the
            // regular object forward pass when this is on.
            if (ImGui::MenuItem(
                    "Debug Draw Collision Meshes (F1)", NULL,
                    show_collision_debug_)) {
                show_collision_debug_ = !show_collision_debug_;
            }

            // Collision-shape selector. Switching here flips
            // `collision_world_dirty_`; application.cpp drains the
            // flag the next frame, calls device_->waitIdle(), tears
            // down the existing CollisionWorld, and rebuilds it
            // with the new shape. Brief stall (~tens of ms on
            // Bistro), expected for a debug mode change.
            if (ImGui::BeginMenu("Collision Shape")) {
                const auto cur = collision_debug_shape_;
                if (ImGui::MenuItem(
                        "Original mesh", NULL,
                        cur == CollisionDebugShape::Original)) {
                    setCollisionDebugShape(
                        CollisionDebugShape::Original);
                }
                if (ImGui::MenuItem(
                        "Gap-filled simplified", NULL,
                        cur == CollisionDebugShape::Simplified)) {
                    setCollisionDebugShape(
                        CollisionDebugShape::Simplified);
                }
                if (ImGui::MenuItem(
                        "Volume (5cm voxels)", NULL,
                        cur == CollisionDebugShape::Volume)) {
                    setCollisionDebugShape(
                        CollisionDebugShape::Volume);
                }
                ImGui::EndMenu();
            }

            // ── Isolate single mesh (debug slider) ───────────────────
            // Scrub one collision mesh at a time to hunt down a broken /
            // missing one.  When enabled, drawDebug shows ONLY the mesh
            // at the slider index; the top overlay shows that mesh's
            // identity (index / category / tris / material / object) so
            // you can read off the index of the bad one.
            ImGui::Separator();
            ImGui::MenuItem("Isolate single mesh", NULL,
                            &collision_isolate_enabled_);
            if (collision_isolate_enabled_ && collision_mesh_count_ > 0) {
                const int max_idx =
                    static_cast<int>(collision_mesh_count_) - 1;
                if (collision_isolate_index_ > max_idx)
                    collision_isolate_index_ = max_idx;
                if (collision_isolate_index_ < 0)
                    collision_isolate_index_ = 0;
                ImGui::SetNextItemWidth(220.0f);
                ImGui::SliderInt("##coll_iso_idx",
                                 &collision_isolate_index_,
                                 0, max_idx, "mesh %d");
                if (ImGui::SmallButton("- prev") &&
                    collision_isolate_index_ > 0)
                    --collision_isolate_index_;
                ImGui::SameLine();
                if (ImGui::SmallButton("next +") &&
                    collision_isolate_index_ < max_idx)
                    ++collision_isolate_index_;
                ImGui::SameLine();
                ImGui::Text("/ %d", max_idx);
            }

            // ── Foot IK (live tuning) ────────────────────────────────
            // Two-bone (hip+knee) inverse kinematics that plants each
            // foot on the ground, tilts it to the surface, and drops the
            // pelvis when a foot cannot reach.  application.cpp reads
            // these every frame and pushes them into PlayerController.
            ImGui::Separator();
            if (ImGui::BeginMenu("Foot IK")) {
                ImGui::Checkbox("Enabled", &foot_ik_params_.enabled);
                ImGui::SetNextItemWidth(220.0f);
                ImGui::SliderFloat("Stride", &foot_ik_params_.stride_amp,
                                   0.0f, 0.6f, "%.2f m");
                ImGui::SetNextItemWidth(220.0f);
                ImGui::SliderFloat("Lift", &foot_ik_params_.lift_amp,
                                   0.0f, 0.4f, "%.2f m");
                ImGui::SetNextItemWidth(220.0f);
                ImGui::SliderFloat("Sole drop", &foot_ik_params_.sole_drop,
                                   0.0f, 0.3f, "%.3f m");
                ImGui::SetNextItemWidth(220.0f);
                ImGui::SliderFloat("Foot tilt", &foot_ik_params_.tilt_weight,
                                   0.0f, 1.0f, "%.2f");
                ImGui::SetNextItemWidth(220.0f);
                ImGui::SliderFloat("Pelvis drop max",
                                   &foot_ik_params_.pelvis_drop_max,
                                   0.0f, 1.5f, "%.2f m");
                ImGui::Text("Live pelvis drop: %.3f m",
                            foot_ik_pelvis_drop_live_);
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // (Skydome / Weather System / IBL Debug / Shadow / Smart Mesh all
        // moved under the Rendering menu.)

        // (Render Debug moved under the Rendering menu — content lives in
        // drawRenderDebugMenuContent().)


        if (ImGui::BeginMenu("Tools"))
        {
            if (ImGui::MenuItem("Compile Shaders", NULL)) {
                compile_shaders = true;
            }

            if (ImGui::MenuItem("Dump noise volumetric texture", NULL)) {
                dump_volume_noise = true;
            }

            // Bake the runtime-built walkable collision world to
            // content/maps/<scene>.rwcmap and reference it from the scene
            // (Save Scene persists the reference; loading the scene then
            // restores collision instantly so the player can spawn and
            // walk with WASD without waiting for the classifier build).
            if (ImGui::MenuItem("Bake Collision Map (Player Walk)", NULL)) {
                bake_collision_map_requested_ = true;
            }

            if (ImGui::MenuItem("GPU Profiler", NULL, s_show_gpu_profiler)) {
                s_show_gpu_profiler = !s_show_gpu_profiler;
            }

            // AI terrain: text → FLUX satellite heightmap → erosion model
            // → 16-bit terrain map (optionally installed as assets/map.png).
            if (ImGui::MenuItem("Generate Terrain (AI)...")) {
                terrain_gen_popup_open_ = true;
            }

            // ── FPS cap ───────────────────────────────────────────────
            // MAILBOX presents uncapped; capping frees GPU time slices
            // for ML jobs sharing the card (classifier / fine-tune /
            // image gen) so presentation stops flashing black.  While
            // the classifier runs, the app clamps to 30 regardless.
            ImGui::Separator();
            if (ImGui::BeginMenu("FPS Cap")) {
                static const int   kCaps[]   = {0, 120, 60, 30};
                static const char* kLabels[] = {"Off (uncapped)", "120",
                                                "60 (default)", "30"};
                for (int ci = 0; ci < 4; ++ci) {
                    if (ImGui::MenuItem(kLabels[ci], NULL,
                                        fps_cap_ == kCaps[ci])) {
                        fps_cap_ = kCaps[ci];
                    }
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (plugin_manager_ && ImGui::BeginMenu("Plugins"))
        {
            for (auto& p : plugin_manager_->plugins()) {
                bool vis = p->isVisible();
                if (ImGui::MenuItem(p->getName(), NULL, vis)) {
                    p->setVisible(!vis);
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // ── Smart Mesh floating window (visible while moving around) ────
    if (cluster_renderer_ && show_smart_mesh_window_) {
        ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Smart Mesh", &show_smart_mesh_window_,
                         ImGuiWindowFlags_NoFocusOnAppearing |
                         ImGuiWindowFlags_AlwaysAutoResize)) {
            // Master toggle: cluster rendering vs direct mesh rendering.
            bool cluster_on = cluster_renderer_->isEnabled();
            if (ImGui::Checkbox("Enable Cluster Rendering", &cluster_on)) {
                cluster_renderer_->getEnabled() = cluster_on;
                if (!cluster_on) cluster_renderer_->resetStats();
            }

            // Culling method radio (only when cluster rendering is on).
            if (cluster_on) {
                bool gpu_mode = !cluster_renderer_->getCpuCullMode();
                if (ImGui::RadioButton("GPU Culling", gpu_mode)) {
                    cluster_renderer_->getCpuCullMode() = false;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("CPU Culling", !gpu_mode)) {
                    cluster_renderer_->getCpuCullMode() = true;
                }
            }

            ImGui::Separator();
            ImGui::Text("Uploaded Meshes: %u", cluster_renderer_->getMeshCount());

            // Helper lambda: format large numbers with K/M suffix.
            auto fmtCount = [](uint64_t n, char* buf, size_t sz) {
                if (n >= 1000000)
                    snprintf(buf, sz, "%.2fM", n / 1000000.0);
                else if (n >= 1000)
                    snprintf(buf, sz, "%.1fK", n / 1000.0);
                else
                    snprintf(buf, sz, "%llu", static_cast<unsigned long long>(n));
            };

            {
                char totalBuf[32], visBuf[32];
                uint64_t totalTri = cluster_renderer_->getTotalTriangles();
                uint64_t visTri   = cluster_renderer_->getVisibleTriangles();
                fmtCount(totalTri, totalBuf, sizeof(totalBuf));
                fmtCount(visTri,   visBuf,   sizeof(visBuf));

                ImGui::Text("Total Clusters:  %u (%s tris)",
                            cluster_renderer_->getTotalClusters(), totalBuf);
                ImGui::Text("Visible:         %u (%s tris)",
                            cluster_renderer_->getTotalVisible(), visBuf);
            }
            ImGui::Text("Culled:          %.1f%%", cluster_renderer_->getCullPercentage());
            // Show per-mesh visibility stats.
            {
                uint32_t total_m = cluster_renderer_->getRegisteredMeshCount();
                uint32_t vis_m = 0;
                for (uint32_t i = 0; i < total_m; ++i)
                    if (cluster_renderer_->isMeshVisible(i)) ++vis_m;
                ImGui::Text("Meshes Visible:  %u / %u", vis_m, total_m);
            }

            ImGui::Separator();
            ImGui::Text("Debug");
            bool cluster_debug = engine::helper::clusterRenderingEnabled();
            if (ImGui::Checkbox("Cluster Debug Draw", &cluster_debug)) {
                engine::helper::clusterRenderingEnabled() = cluster_debug;
            }
            bool bbox_draw = cluster_renderer_->getDebugDrawBBox();
            if (ImGui::Checkbox("Cluster Bound Box Draw", &bbox_draw)) {
                cluster_renderer_->getDebugDrawBBox() = bbox_draw;
            }
            // Hi-Z occlusion-cull toggle: when on, Phase B samples the
            // Hi-Z pyramid (built from this frame's Phase A depth) and
            // rejects clusters whose bounding-sphere nearest point is
            // BEHIND every visible surface in its screen footprint.
            // Off = plain frustum + backface cone cull (the default).
            bool hiz_cull = cluster_renderer_->getUseHiZOcclusionCull();
            if (ImGui::Checkbox("Hi-Z Occlusion Cull", &hiz_cull)) {
                cluster_renderer_->getUseHiZOcclusionCull() = hiz_cull;
            }
            ImGui::Separator();
            const auto& vp = cluster_renderer_->getDebugVP();
            const auto& cp = cluster_renderer_->getDebugCamPos();
            ImGui::Text("VP diag: %.2f, %.2f, %.2f, %.2f",
                        vp[0][0], vp[1][1], vp[2][2], vp[3][3]);
            ImGui::Text("Cam: %.1f, %.1f, %.1f", cp.x, cp.y, cp.z);
        }
        ImGui::End();
    }

    // ── Cluster bounding box wireframe overlay ──────────────────────
    // When enabled, projects cluster AABB corners to screen space and
    // draws wireframe boxes via ImGui's background draw list so they
    // appear on top of the 3D scene but behind other UI.
    if (cluster_renderer_ && cluster_renderer_->getDebugDrawBBox()) {
        const auto& samples = cluster_renderer_->getDebugSampleClusters();
        const auto& vp_mat  = cluster_renderer_->getDebugVP();

        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        ImVec2 display_size = ImGui::GetIO().DisplaySize;
        float half_w = display_size.x * 0.5f;
        float half_h = display_size.y * 0.5f;

        // Project a world-space point to screen pixel coords.
        // Returns false if behind camera (clip.w <= 0).
        auto projectToScreen = [&](const glm::vec3& world_pos,
                                   ImVec2& out_screen) -> bool {
            glm::vec4 clip = vp_mat * glm::vec4(world_pos, 1.0f);
            if (clip.w <= 0.001f) return false;
            float inv_w = 1.0f / clip.w;
            float ndc_x = clip.x * inv_w;
            float ndc_y = clip.y * inv_w;
            out_screen.x = (ndc_x * 0.5f + 0.5f) * display_size.x;
            out_screen.y = (ndc_y * 0.5f + 0.5f) * display_size.y;
            return true;
        };

        // 12 edges of a box: pairs of corner indices (0..7)
        static const int edges[12][2] = {
            {0,1},{1,3},{3,2},{2,0},  // bottom face
            {4,5},{5,7},{7,6},{6,4},  // top face
            {0,4},{1,5},{2,6},{3,7}   // vertical edges
        };

        const ImU32 bbox_color = IM_COL32(0, 255, 0, 160);  // green wireframe

        for (const auto& ci : samples) {
            glm::vec3 mn(ci.aabb_min_pad.x, ci.aabb_min_pad.y, ci.aabb_min_pad.z);
            glm::vec3 mx(ci.aabb_max_pad.x, ci.aabb_max_pad.y, ci.aabb_max_pad.z);

            // Skip degenerate / zero AABBs
            if (mn == mx) continue;

            // 8 corners of the AABB
            glm::vec3 corners[8] = {
                {mn.x, mn.y, mn.z},  // 0
                {mx.x, mn.y, mn.z},  // 1
                {mn.x, mx.y, mn.z},  // 2
                {mx.x, mx.y, mn.z},  // 3
                {mn.x, mn.y, mx.z},  // 4
                {mx.x, mn.y, mx.z},  // 5
                {mn.x, mx.y, mx.z},  // 6
                {mx.x, mx.y, mx.z},  // 7
            };

            // Project all 8 corners
            ImVec2 screen_pts[8];
            bool visible[8];
            int visible_count = 0;
            for (int i = 0; i < 8; ++i) {
                visible[i] = projectToScreen(corners[i], screen_pts[i]);
                if (visible[i]) ++visible_count;
            }

            // Only draw if at least 2 corners are on screen
            if (visible_count < 2) continue;

            // Draw 12 edges
            for (int e = 0; e < 12; ++e) {
                int a = edges[e][0];
                int b = edges[e][1];
                if (visible[a] && visible[b]) {
                    draw_list->AddLine(screen_pts[a], screen_pts[b],
                                       bbox_color, 1.0f);
                }
            }
        }
    }

    // Camera pause: gate on what ImGui actually WANTS, not on window focus.
    // Focus persists after ANY panel interaction (placing an object focuses
    // the browser/Outliner) and right-clicking the viewport never clears
    // it — which froze the camera until a stray left-click on empty space.
    // WantCaptureMouse is false whenever the cursor is over the
    // pass-through viewport; WantTextInput keeps WASD from fighting an
    // active text field.
    ImGuiIO& focus_io = ImGui::GetIO();
    bool in_focus = focus_io.WantCaptureMouse || focus_io.WantTextInput;

    // Pin all menu windows to the main viewport so they stay inside the
    // main application window (only the AutoRig plugin overrides this).
    ImGuiID main_vp_id = ImGui::GetMainViewport()->ID;

    if (s_show_skydome) {
        ImGui::SetNextWindowViewport(main_vp_id);
        if (ImGui::Begin("Skydome", &s_show_skydome, ImGuiWindowFlags_NoDocking)) {
            // ---- Time of day -------------------------------------------
            // Slider drives Skydome::update each frame; the application
            // calls consumeTodJump() to detect big skips (debug "go to
            // dawn" etc.) and reset the mini-buffer accumulators.
            ImGui::SeparatorText("Time of day");
            ImGui::SliderFloat("Hour (0..24)", &tod_hours_, 0.0f, 24.0f, "%.2f h");
            // Quick-jump buttons for common test cases.
            if (ImGui::Button("Dawn (6h)"))    { setTimeOfDayHours(6.0f); }
            ImGui::SameLine();
            if (ImGui::Button("Noon (12h)"))   { setTimeOfDayHours(12.0f); }
            ImGui::SameLine();
            if (ImGui::Button("Dusk (18h)"))   { setTimeOfDayHours(18.0f); }
            ImGui::SameLine();
            if (ImGui::Button("Midnight (0h)")){ setTimeOfDayHours(0.0f); }
            ImGui::Checkbox("Auto-advance", &tod_auto_advance_);
            if (tod_auto_advance_) {
                ImGui::SameLine();
                // Speed = game-time / real-time.  100x by default
                // (1 real-second = 100 game-seconds, full day in ~14.4 min).
                ImGui::SliderFloat("speed (x real-time)",
                    &tod_advance_speed_, 1.0f, 10000.0f, "%.0fx",
                    ImGuiSliderFlags_Logarithmic);
            }
            ImGui::SliderFloat("Jump threshold (h)",
                &tod_jump_threshold_, 0.05f, 6.0f, "%.2f");

            ImGui::SeparatorText("Atmosphere");
            ImGui::SliderFloat("phase func g", &skydome->getG(), -1.0f, 2.0f);
            ImGui::SliderFloat("rayleigh scale height", &skydome->getRayleighScaleHeight(), 0.0f, 16000.0f);
            ImGui::SliderFloat("mei scale height", &skydome->getMieScaleHeight(), 0.0f, 2400.0f);

            // ---- Sky envmap fragment-shader debug visualisation -------
            // Drives SkyboxEnvmapParams::debug_mode in skybox_envmap.frag
            // each frame.  Internally the shader's switch is sparse (no
            // mode "2"), so we keep the same numeric values and label
            // them explicitly via the items[] array.
            ImGui::SeparatorText("Debug");
            static const char* kSkyDebugItems[] = {
                "0: Normal (tone-mapped)",
                "1: Solid red (FS smoke test)",
                "3: view_dir RGB",
                "4: Envmap raw (HDR)",
                "5: Envmap x10000",
            };
            static const int kSkyDebugValues[] = { 0, 1, 3, 4, 5 };
            int& debug_sky_mode = skydome->getDebugSkyMode();
            int current_idx = 0;
            for (int i = 0; i < IM_ARRAYSIZE(kSkyDebugValues); ++i) {
                if (kSkyDebugValues[i] == debug_sky_mode) { current_idx = i; break; }
            }
            if (ImGui::Combo("Debug mode", &current_idx,
                             kSkyDebugItems, IM_ARRAYSIZE(kSkyDebugItems))) {
                debug_sky_mode = kSkyDebugValues[current_idx];
            }
        }
        ImGui::End();
    }

    if (s_show_weather) {
        ImGui::SetNextWindowViewport(main_vp_id);
        if (ImGui::Begin("Weather", &s_show_weather, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoDocking)) {
            ImGui::Checkbox("Turn off volume moist", &turn_off_volume_moist_);
            ImGui::Checkbox("Turn on airflow effect", &turn_on_airflow_);

            ImGui::Separator();

            const char* items[] = { "no debug draw", "debug temperature", "debug moisture" };
            if (ImGui::BeginCombo("##debugmode", items[debug_draw_type_])) // The second parameter is the label previewed before opening the combo.
            {
                for (int n = 0; n < IM_ARRAYSIZE(items); n++)
                {
                    bool is_selected = (items[debug_draw_type_] == items[n]); // You can store your selection however you want, outside or inside your objects
                    if (ImGui::Selectable(items[n], is_selected)) {
                        debug_draw_type_ = n;
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Separator();

            ImGui::SliderFloat("Light Extinct Rate", &light_ext_factor_, 0.0f, 0.1f);
            ImGui::SliderFloat("View Extinct Rate", &view_ext_factor_, 0.0f, 2.0f);
            ImGui::SliderFloat("View Extinct Exponent", &view_ext_exponent_, 0.0f, 2.0f);
            ImGui::SliderFloat("Cloud Ambient Intensity", &cloud_ambient_intensity_, 0.0f, 2.0f);
            ImGui::SliderFloat("Cloud Phase Intensity", &cloud_phase_intensity_, 0.0f, 2.0f);

            ImGui::Separator();

            ImGui::SliderFloat("water flow strength", &water_flow_strength_, 0.0f, 10.0f);
            ImGui::SliderFloat("air flow strength", &air_flow_strength_, 0.0f, 100.0f);

            ImGui::Separator();

            ImGui::SliderFloat("mix rate", &weather_controls_.mix_rate, 0.0f, 1.0f);
            ImGui::SliderFloat("sea level temperature", &weather_controls_.sea_level_temperature, -40.0f, 40.0f);
            ImGui::SliderFloat("global flow dir", &global_flow_dir_, 0.0f, 360.0f);
            ImGui::SliderFloat("global flow speed (m/f)", &global_flow_speed_, 0.0f, 10.0f);

            ImGui::Separator();

            ImGui::SliderFloat("soil moist regenerate", &weather_controls_.soil_moist_adj, 0.0f, 1.0f);
            ImGui::SliderFloat("water moist regenerate", &weather_controls_.water_moist_adj, 0.0f, 1.0f);

            ImGui::Separator();

            ImGui::SliderFloat("transfer rate", &weather_controls_.transfer_ratio, 0.0f, 2.0f);
            ImGui::SliderFloat("transfer noise level", &weather_controls_.transfer_noise_weight, 0.0f, 1.0f);

            ImGui::Separator();

            ImGui::SliderFloat("cloud forming ratio", &weather_controls_.cloud_forming_ratio, 0.0f, 1.0f);
            ImGui::SliderFloat("frozen ext factor", &weather_controls_.frozen_ext_factor, 0.0f, 10.0f);
            ImGui::SliderFloat("frozen power curve", &weather_controls_.frozen_pow_curve, 0.0f, 10.0f);

            ImGui::Separator();
            ImGui::SliderFloat4("cloud noise weight 0", cloud_noise_weight_[0], 0.0f, 5.0f);
            ImGui::SliderFloat4("cloud noise weight 1", cloud_noise_weight_[1], 0.0f, 5.0f);
            ImGui::SliderFloat("cloud noise thresold", &cloud_noise_thresold_, 0.0f, 1.0f);
            ImGui::SliderFloat("cloud noise scrolling speed", &cloud_noise_scrolling_speed_, 0.0f, 20.0f);
            ImGui::SliderFloat2("cloud noise scale", cloud_noise_scale_, 0.0f, 10.0f);
        }
        ImGui::End();
    }

    if (s_select_load_gltf) {
        ImGui::OpenPopup("select gltf object");
        if (ImGui::BeginPopup("select gltf object"))
        {
            std::vector<const char*> listbox_items;
            for (const auto& name : gltf_file_names_) {
                listbox_items.push_back(name.c_str());
            }
            static int s_listbox_item_current = -1;
            ImGui::ListBox(" ",
                &s_listbox_item_current, listbox_items.data(),
                static_cast<int>(listbox_items.size()),
                static_cast<int>(listbox_items.size()));

            if (s_listbox_item_current >= 0) {
                auto file_name = "assets/" + gltf_file_names_[s_listbox_item_current];
                to_load_gltf_names_.push_back(file_name);
                s_listbox_item_current = -1;
                s_select_load_gltf = false;
            }

            ImGui::EndPopup();
        }

        if (!s_select_load_gltf) {
            ImGui::CloseCurrentPopup();
        }
    }

    // (Legacy s_spawn_player → spawn_gltf_name_ trigger removed along
    // with the menu item that set it; the player is eager-loaded once
    // at application startup with a fixed asset.)

    if (compile_shaders) {
        auto error_strings = engine::helper::compileGlobalShaders();
        s_shader_error_message = error_strings;
        if (error_strings.length() > 0) {
            s_show_shader_error_message = true;
        }
    }

    if (s_show_shader_error_message) {
        ImGui::SetNextWindowViewport(main_vp_id);
        if (ImGui::Begin("Shader Compile Error", &s_show_shader_error_message, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoDocking)) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%s", s_shader_error_message.c_str());
        }
        ImGui::End();
    }

    // ---- IBL / Sky cubemap debug — popup float window ---------------------
    // ---- IBL / Sky cubemap debug — floating tool window -------------------
    // Regular Begin (not modal) so the main scene stays fully interactive.
    if (show_ibl_debug_) {
        const float thumb = ibl_debug_thumb_size_;
        const float row_w = thumb * 6.0f + 60.0f;
        const float row_h = thumb + 60.0f;  // +slider line per row
        const float win_w = row_w + 40.0f;
        const float win_h = row_h * 5.0f + 120.0f;

        // Centre on first appearance; user can move/resize freely after.
        const ImVec2 vp_centre = viewportCenter();
        ImGui::SetNextWindowPos(vp_centre, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(win_w, win_h), ImGuiCond_Appearing);
        if (ImGui::Begin("IBL / Sky Debug", &show_ibl_debug_)) {
            ImGui::SliderFloat("Thumbnail size",
                &ibl_debug_thumb_size_, 32.0f, 256.0f, "%.0f px");
            // Exposure slider: values up to 32x cover atmospheric mid-
            // tones (~0.05 linear -> 1.6 displayed).  Logarithmic so
            // small movements near 1.0 give intuitive control.
            ImGui::SliderFloat("Exposure",
                &ibl_debug_exposure_, 0.1f, 32.0f, "%.2fx",
                ImGuiSliderFlags_Logarithmic);
            ImGui::SameLine();
            if (ImGui::Button("1x")) ibl_debug_exposure_ = 1.0f;

            // Apply exposure as the ImGui::Image tint multiplier.  Alpha
            // stays at 1 so we don't accidentally key transparency off
            // the cubemap's alpha channel.
            const ImVec4 tint(
                ibl_debug_exposure_,
                ibl_debug_exposure_,
                ibl_debug_exposure_,
                1.0f);
            const ImVec4 border(0, 0, 0, 0);

            const char* face_names[6] = { "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

            // One helper for ALL rows: label + per-row mip slider + 6
            // face thumbnails at the selected mip.  Rows whose cubemap
            // has only one mip get a degenerate [0, 0] slider that
            // ImGui still draws so the layout stays uniform across rows.
            auto draw_mip_row = [&](
                const char* row_id,
                const char* label,
                const IblDebugMipFaceArray& mip_face_ids,
                int num_mips,
                int* current_mip) {
                ImGui::SeparatorText(label);
                ImGui::PushID(row_id);
                const int max_mip = std::max(0, num_mips - 1);
                ImGui::SetNextItemWidth(thumb * 4.0f);
                ImGui::SliderInt("mip", current_mip, 0, max_mip);
                ImGui::PopID();
                const int picked = std::clamp(
                    *current_mip, 0, kIblDebugMaxMips - 1);
                const auto& faces = mip_face_ids[picked];

                for (int f = 0; f < 6; ++f) {
                    if (f > 0) ImGui::SameLine();
                    ImGui::BeginGroup();
                    ImGui::Text("%s", face_names[f]);
                    ImTextureID id = faces[f];
                    if (id) {
                        ImGui::Image(id, ImVec2(thumb, thumb),
                            ImVec2(0, 0), ImVec2(1, 1), tint, border);
                        if (ImGui::IsItemHovered()) {
                            ImGui::BeginTooltip();
                            ImGui::Image(id, ImVec2(384.0f, 384.0f),
                                ImVec2(0, 0), ImVec2(1, 1), tint, border);
                            ImGui::EndTooltip();
                        }
                    } else {
                        ImGui::Dummy(ImVec2(thumb, thumb));
                    }
                    ImGui::EndGroup();
                }
            };

            // Sky envmap (full mip chain - useful for verifying mipgen
            // smoothing of the dithered partial updates).
            draw_mip_row(
                "envmap", "Sky envmap",
                envmap_face_mip_tex_ids_,
                ibl_debug_envmap_num_mips_,
                &ibl_debug_envmap_mip_);

            // Sky mini-buffer (size/8 cubemap, single mip - slider goes
            // 0..0 but stays in the UI for layout consistency).
            draw_mip_row(
                "mini_envmap", "Sky mini-buffer (size/8)",
                mini_envmap_face_mip_tex_ids_,
                ibl_debug_mini_envmap_num_mips_,
                &ibl_debug_mini_envmap_mip_);

            // IBL diffuse (Lambertian, single mip).  The application
            // registers the *blurred consumer-facing* cube here (see
            // IblCreator::getIblDiffuseTexture, which now returns
            // tmp_ibl_diffuse_tex_ — the post-blur output bound at
            // LAMBERTIAN_ENV_TEX_INDEX).  This is what cluster /
            // forward / glass shaders actually sample, so the panel
            // matches what's on screen.
            draw_mip_row(
                "ibl_diffuse", "IBL diffuse (Lambertian, post-blur)",
                diffuse_face_mip_tex_ids_,
                ibl_debug_diffuse_num_mips_,
                &ibl_debug_diffuse_mip_);

            // IBL specular (GGX, full mip chain - higher mips are box-
            // filter mipgen of mip 0; visible blur progression).
            draw_mip_row(
                "ibl_specular", "IBL specular (GGX, +mipgen)",
                specular_face_mip_tex_ids_,
                ibl_debug_specular_num_mips_,
                &ibl_debug_specular_mip_);

            // IBL sheen (Charlie, full mip chain).
            draw_mip_row(
                "ibl_sheen", "IBL sheen (Charlie, +mipgen)",
                sheen_face_mip_tex_ids_,
                ibl_debug_sheen_num_mips_,
                &ibl_debug_sheen_mip_);
        }
        ImGui::End();
    }

    // ---- CSM cascade debug window ------------------------------------------
    if (show_csm_debug_) {
        constexpr float kThumbSize = 240.0f;
        // Fixed window size: 4 images side-by-side + padding/spacing.
        constexpr float kWinW = kThumbSize * CSM_CASCADE_COUNT + 40.0f;
        constexpr float kWinH = kThumbSize + 80.0f;
        ImGui::SetNextWindowSize(ImVec2(kWinW, kWinH));
        ImGui::SetNextWindowViewport(main_vp_id);
        if (ImGui::Begin("CSM Debug", &show_csm_debug_, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking)) {
            // Build the cascade label per-iteration so this scales with
            // CSM_CASCADE_COUNT.  The previous code hard-coded a 4-entry
            // labels[] array, which crashed (OOB read → garbage pointer
            // fed into ImGui::Text("%s", ...) → CRT format-string parse
            // crash) once CSM_CASCADE_COUNT was bumped to 6 — see the
            // csm_cascade_splits array in application.cpp.
            char label_buf[32];
            for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
                if (k > 0) ImGui::SameLine();
                ImGui::BeginGroup();
                // Label above each image — mark the ends so the user
                // can orient at a glance.
                const char* suffix =
                    (k == 0) ? "\n(near)"
                    : (k == CSM_CASCADE_COUNT - 1) ? "\n(far)"
                    : "";
                std::snprintf(label_buf, sizeof(label_buf),
                              "Cascade %d%s", k, suffix);
                ImGui::Text("%s", label_buf);
                if (csm_debug_tex_ids_[k]) {
                    ImGui::Image(csm_debug_tex_ids_[k], ImVec2(kThumbSize, kThumbSize));
                    if (ImGui::IsItemHovered()) {
                        // Show a larger tooltip preview.
                        ImGui::BeginTooltip();
                        ImGui::Image(csm_debug_tex_ids_[k], ImVec2(300, 300));
                        ImGui::EndTooltip();
                    }
                } else {
                    ImGui::Dummy(ImVec2(kThumbSize, kThumbSize));
                }
                ImGui::EndGroup();
            }
        }
        ImGui::End();
    }
    // ------------------------------------------------------------------------

    // ---- RVT pool viewer ---------------------------------------------------
    // Shows the four 4096² layer pool textures.  If RVT registration
    // is working you'll see a tiled atlas of 128×128 pages (one per
    // registered material texture).  Empty / unregistered pages stay
    // black.  Toggled via Tools → "Show VT Pool" in the menu.
    if (show_vt_pool_debug_) {
        // Window is sized for the activity grid up top + the four
        // 240-px pool atlas thumbnails below.
        constexpr float kThumbSize = 240.0f;
        constexpr float kWinW      = kThumbSize * 4 + 60.0f;
        constexpr float kWinH      = kThumbSize + 320.0f;
        ImGui::SetNextWindowSize(ImVec2(kWinW, kWinH));
        ImGui::SetNextWindowViewport(main_vp_id);
        if (ImGui::Begin("VT Pool Debug", &show_vt_pool_debug_,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking)) {
            // ── VT on/off toggle ───────────────────────────────────
            // Re-uploads material_params with all vt_id fields cleared
            // when toggled off; the shader's existing fallback to the
            // legacy bindless texture arrays takes over.  Side-by-side
            // toggle is the cleanest way to A/B the VT path's quality
            // and perf against the bindless baseline.
            if (cluster_renderer_) {
                bool vt_on = cluster_renderer_->isVtEnabled();
                if (ImGui::Checkbox("VT Enabled", &vt_on)) {
                    cluster_renderer_->setVtEnabled(vt_on);
                }
                ImGui::SameLine();
                ImGui::TextDisabled(
                    "(off → legacy bindless texture arrays)");
                // (The Glass translucent-mode picker lives in the main
                // menu bar under Render Debug → Glass: ... — it was
                // previously parked here but is unrelated to VT.)
                ImGui::Separator();
            }

            // ── Activity grid: per-slot colour-coded heatmap ──
            // One coloured rectangle per pool slot.  Colour mapping:
            //   BLACK         free / unallocated
            //   DARK GREY     resident, idle (cached but unused this frame)
            //   BRIGHT GREEN  resident + accessed this frame (the
            //                 working set the camera is actually
            //                 sampling)
            //   GOLD          pinned (always-resident smallest-mip
            //                 fallback per VT)
            //   BRIGHT YELLOW pinned + accessed this frame
            // The ratio of green-to-grey tells you whether the pool
            // is sized right: lots of grey = pool is bigger than
            // working set (memory wasted); 100% green = thrashing
            // boundary; pinned-only-yellow with no green means the
            // streamer hasn't caught up to the camera.
            if (vt_manager_) {
                const auto& grid = vt_manager_->getSlotStatusGrid();
                const uint32_t cols = vt_manager_->getSlotGridWidth();
                const uint32_t rows = vt_manager_->getSlotGridHeight();
                const uint32_t total = uint32_t(grid.size());
                if (cols > 0 && rows > 0 && total >= cols * rows) {
                    const uint32_t a = vt_manager_->getSlotsActive();
                    const uint32_t r = vt_manager_->getSlotsResident();
                    const uint32_t p = vt_manager_->getSlotsPinned();
                    ImGui::Text("Slots: %u total | %u resident (%.1f%%) | "
                                "%u active this frame (%.1f%% of resident) | "
                                "%u pinned",
                        total, r, 100.0f * float(r) / float(total),
                        a, r > 0 ? 100.0f * float(a) / float(r) : 0.0f,
                        p);

                    // Pick a cell pixel size that fits the grid into
                    // the available content width.  Min 4 px so the
                    // colours are still legible at 114 cols.
                    const float avail_w = ImGui::GetContentRegionAvail().x - 8.0f;
                    float cell_px = std::floor(avail_w / float(cols));
                    if (cell_px < 4.0f) cell_px = 4.0f;
                    if (cell_px > 16.0f) cell_px = 16.0f;
                    const float grid_w = cell_px * float(cols);
                    const float grid_h = cell_px * float(rows);

                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    // Background frame.
                    dl->AddRectFilled(origin,
                        ImVec2(origin.x + grid_w, origin.y + grid_h),
                        IM_COL32(20, 20, 20, 255));
                    // Per-cell colour.
                    for (uint32_t y = 0; y < rows; ++y) {
                        for (uint32_t x = 0; x < cols; ++x) {
                            uint8_t b = grid[y * cols + x];
                            const bool resident =
                                (b & engine::scene_rendering::
                                       VirtualTextureManager::kSlotStatusResident);
                            const bool active   =
                                (b & engine::scene_rendering::
                                       VirtualTextureManager::kSlotStatusActive);
                            const bool pinned   =
                                (b & engine::scene_rendering::
                                       VirtualTextureManager::kSlotStatusPinned);
                            ImU32 col;
                            if (!resident) {
                                col = IM_COL32(0, 0, 0, 255);
                            } else if (pinned && active) {
                                col = IM_COL32(255, 235, 0, 255);  // bright yellow
                            } else if (pinned) {
                                col = IM_COL32(180, 130, 0, 255);  // gold
                            } else if (active) {
                                col = IM_COL32(60, 220, 80, 255);  // bright green
                            } else {
                                col = IM_COL32(70, 70, 80, 255);   // dark grey
                            }
                            ImVec2 p0(origin.x + float(x) * cell_px,
                                      origin.y + float(y) * cell_px);
                            ImVec2 p1(p0.x + cell_px - 0.5f,
                                      p0.y + cell_px - 0.5f);
                            dl->AddRectFilled(p0, p1, col);
                        }
                    }
                    ImGui::Dummy(ImVec2(grid_w, grid_h));
                }
                ImGui::Separator();
            } else {
                ImGui::TextDisabled("(VT manager not bound — slot grid unavailable)");
                ImGui::Separator();
            }

            // ── Pool image atlas thumbnails (collapsible) ──
            // Kept for content inspection — useful for confirming
            // tiles are encoded correctly.  Less useful day-to-day
            // than the activity grid above.
            if (ImGui::CollapsingHeader("Pool Image Atlases")) {
                // Layer format labels — must match VtLayer enum order.
                const char* labels[] = {
                    "Albedo (BC7 sRGB)",
                    "Normal (BC5 UNorm)",
                    "Metal/Rough/AO (RGBA8)",
                    "Emissive (RGBA8)" };
                for (int k = 0; k < 4; ++k) {
                    if (k > 0) ImGui::SameLine();
                    ImGui::BeginGroup();
                    ImGui::Text("%s", labels[k]);
                    if (vt_pool_tex_ids_[k]) {
                        ImGui::Image(vt_pool_tex_ids_[k],
                                     ImVec2(kThumbSize, kThumbSize));
                        if (ImGui::IsItemHovered()) {
                            ImGui::BeginTooltip();
                            ImGui::Image(vt_pool_tex_ids_[k], ImVec2(600, 600));
                            ImGui::EndTooltip();
                        }
                    } else {
                        ImGui::Dummy(ImVec2(kThumbSize, kThumbSize));
                        ImGui::TextDisabled("(not bound)");
                    }
                    ImGui::EndGroup();
                }
            }
        }
        ImGui::End();
    }
    // ------------------------------------------------------------------------

    // ---- Hi-Z pyramid debug viewer -----------------------------------------
    // Strip of per-mip thumbnails for the cluster Z-cull pyramid built
    // each frame from depth_buffer_copy_.  Useful for verifying that
    // the per-frame Hi-Z build dispatch is firing AND producing sane
    // values BEFORE chasing further bugs in the cluster cull's sample
    // math.  The texture format is R32F (max-Z reduction); ImGui will
    // sample it as grey-scale, so:
    //   black  → near plane (closest geometry)
    //   white  → far plane (sky / unwritten / cleared 1.0)
    //   grey   → real depth values
    // If every mip is uniform white the build isn't writing the
    // pyramid; if uniform black the binding is reading 0; mips that
    // look like a fuzzy depth map across the whole strip mean the
    // build is working and we can trust the sample side.
    if (show_hiz_debug_) {
        const ImVec2 vp_centre = viewportCenter();
        ImGui::SetNextWindowPos(vp_centre, ImGuiCond_Appearing,
                                ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(800.0f, 220.0f), ImGuiCond_Appearing);
        if (ImGui::Begin("Hi-Z Pyramid Debug", &show_hiz_debug_)) {
            if (hiz_debug_tex_ids_.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                    "Hi-Z pyramid viewer not yet wired into application.");
                ImGui::Spacing();
                ImGui::TextWrapped(
                    "The application needs to register one ImTextureID "
                    "per pyramid mip (using Helper::addImTextureID on "
                    "each entry of hiz_mip_views_) and pass them via "
                    "menu_->setHiZDebugTextureIds(...).");
            } else {
                ImGui::Text("Mip 0 = full-res depth on the left.  "
                            "Mips halve in size each step to the right.");
                ImGui::Text("Black = near, white = far.  Uniform white "
                            "across all mips usually means the build "
                            "dispatch isn't writing.");
                ImGui::Spacing();

                // Each mip is half the resolution of the previous one;
                // the displayed thumbnail width also halves so the
                // proportions stay correct as you scan across.
                float w = 256.0f;
                for (size_t m = 0; m < hiz_debug_tex_ids_.size(); ++m) {
                    if (m > 0) ImGui::SameLine();
                    if (hiz_debug_tex_ids_[m]) {
                        ImGui::Image(
                            hiz_debug_tex_ids_[m],
                            ImVec2(std::max(w, 8.0f),
                                   std::max(w, 8.0f)));
                    } else {
                        ImGui::Dummy(ImVec2(std::max(w, 8.0f),
                                            std::max(w, 8.0f)));
                    }
                    w *= 0.5f;
                }
            }
        }
        ImGui::End();
    }

    // ---- Mesh-Category inspector window ------------------------------------
    // Two scrollable ImGui tables showing the LLM classifier's verdict
    // for every collected (material, object) name.  Snapshot pushed
    // from application.cpp once the async classifyAll() finishes;
    // entries are pre-sorted alphabetically in setMaterialCategory
    // Snapshot so the order is stable across frames.
    //
    // Each row: a 12x12 colour swatch + the source name + the
    // category tag string ("FLOOR", "WALL", …).  Colour table is
    // duplicated from the shader / collision_debug.frag so the
    // inspector reads identically to the on-screen overlay; the
    // comment on each case calls this out as a drift hazard.
    //
    // Filter box at the top runs a case-insensitive substring match
    // against the name column so the user can pinpoint a specific
    // material in the 500+-row scrollback.
    if (show_mesh_category_inspector_ && mesh_category_snapshot_valid_) {
        const ImVec2 vp_centre = viewportCenter();
        ImGui::SetNextWindowPos(vp_centre, ImGuiCond_Appearing,
                                ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(620.0f, 540.0f),
                                 ImGuiCond_Appearing);

        if (ImGui::Begin("Mesh-Category Inspector",
                         &show_mesh_category_inspector_)) {
            ImGui::TextDisabled(
                "%zu materials  •  %zu objects classified",
                mesh_category_materials_.size(),
                mesh_category_objects_.size());

            ImGui::InputTextWithHint(
                "##filter", "Filter by substring (case-insensitive)",
                mesh_category_filter_,
                sizeof(mesh_category_filter_));

            // Colour table — uint32_t MeshCategory cast → RGB.  KEEP
            // IN SYNC with categoryColor() in collision_debug.frag
            // and the switch in cluster_bindless.frag, OR every
            // colour-coded debug surface stops agreeing with each
            // other.
            auto cat_color_and_tag = [](uint32_t cat,
                                        ImVec4& out_rgb,
                                        const char*& out_tag) {
                switch (cat) {
                    case 1:  out_rgb = ImVec4(0.20f, 0.75f, 0.30f, 1.0f); out_tag = "WALKABLE_SURFACE";      break;
                    case 2:  out_rgb = ImVec4(0.80f, 0.20f, 0.20f, 1.0f); out_tag = "WALL";       break;
                    case 3:  out_rgb = ImVec4(0.95f, 0.75f, 0.10f, 1.0f); out_tag = "DOOR";       break;
                    case 4:  out_rgb = ImVec4(0.55f, 0.25f, 0.80f, 1.0f); out_tag = "OBJECT";     break;
                    case 5:  out_rgb = ImVec4(0.40f, 0.85f, 0.95f, 1.0f); out_tag = "GLASS";      break;
                    case 6:  out_rgb = ImVec4(0.25f, 0.45f, 0.85f, 1.0f); out_tag = "CEILING";    break;
                    case 7:  out_rgb = ImVec4(0.95f, 0.50f, 0.10f, 1.0f); out_tag = "STAIRS";     break;
                    case 8:  out_rgb = ImVec4(0.55f, 0.65f, 0.20f, 1.0f); out_tag = "VEGETATION"; break;
                    case 9:  out_rgb = ImVec4(0.95f, 0.20f, 0.65f, 1.0f); out_tag = "ELEVATOR";   break;
                    case 10: out_rgb = ImVec4(0.75f, 0.55f, 0.35f, 1.0f); out_tag = "LADDER";     break;
                    default: out_rgb = ImVec4(0.55f, 0.55f, 0.55f, 1.0f); out_tag = "UNKNOWN";    break;
                }
            };

            // Case-insensitive substring filter.  Build a lowered
            // needle once per frame; per-row test lower-cases the
            // name into a local buffer (names are short so a 256-byte
            // stack buffer is fine; longer names truncate which is
            // acceptable for a filter).
            std::string needle = mesh_category_filter_;
            std::transform(needle.begin(), needle.end(), needle.begin(),
                [](unsigned char c){ return std::tolower(c); });
            auto matches = [&](const std::string& name) {
                if (needle.empty()) return true;
                std::string low = name;
                std::transform(low.begin(), low.end(), low.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                return low.find(needle) != std::string::npos;
            };

            auto draw_table = [&](
                const char* title,
                const std::vector<std::pair<std::string, uint32_t>>& rows) {
                ImGui::Spacing();
                ImGui::SeparatorText(title);
                // Fixed-height child for a nested scroll region so the
                // user can scroll materials independently of objects.
                // NB: spelling is `ImGuiChildFlags_Borders` (plural).
                // It was renamed from `_Border` in ImGui 1.91.1
                // (August 2024) — the legacy spelling no longer
                // resolves in this project's 1.92.8 build.
                if (ImGui::BeginChild(
                        title, ImVec2(0.0f, 220.0f),
                        ImGuiChildFlags_Borders,
                        ImGuiWindowFlags_HorizontalScrollbar)) {
                    for (const auto& [name, cat] : rows) {
                        if (!matches(name)) continue;
                        ImVec4 rgb;
                        const char* tag = nullptr;
                        cat_color_and_tag(cat, rgb, tag);
                        ImGui::ColorButton(
                            "##sw", rgb,
                            ImGuiColorEditFlags_NoTooltip |
                                ImGuiColorEditFlags_NoPicker |
                                ImGuiColorEditFlags_NoLabel,
                            ImVec2(12, 12));
                        ImGui::SameLine();
                        // Category tag in fixed-width column then name.
                        ImGui::Text("%-10s", tag);
                        ImGui::SameLine();
                        ImGui::TextUnformatted(name.c_str());
                    }
                }
                ImGui::EndChild();
            };

            draw_table("Materials", mesh_category_materials_);
            draw_table("Objects",   mesh_category_objects_);
        }
        ImGui::End();
    }

    // ---- Dynamic camera-positioned cubemap debug viewer --------------------
    // Mirrors the IBL Debug window layout but for the per-frame DynamicCubemap
    // probe.  Rows shown:
    //   • "Read buffer" — the ping-pong slice currently being sampled by
    //     downstream consumers (glass OIT, opaque IBL).  Each face index's
    //     thumbnail comes from the per-face 2D layer view registered with
    //     ImGui at startup.  The face that was freshly rendered THIS frame
    //     is highlighted with a yellow border so the round-robin update
    //     pattern is visible at a glance.
    //   • "Write buffer" — the other ping-pong slice (next frame's read
    //     target).  Useful for comparing reprojection output against the
    //     previously-stable buffer.
    // Per-face capture-position metadata is shown in a tooltip on hover.
    if (show_dynamic_cube_debug_) {
        const float thumb = dynamic_cube_thumb_size_;
        const float win_w = thumb * 6.0f + 80.0f;
        const float win_h = thumb * 2.0f + 200.0f;

        const ImVec2 vp_centre = viewportCenter();
        ImGui::SetNextWindowPos(vp_centre, ImGuiCond_Appearing,
                                ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(win_w, win_h), ImGuiCond_Appearing);
        if (ImGui::Begin("Dynamic Cubemap Debug", &show_dynamic_cube_debug_)) {
            // Detect "not yet wired" state — every entry is a default-
            // constructed ImTextureID(0).  Show a hint instead of an
            // empty grid of Dummy slots so it's obvious why the panel
            // is blank.
            bool any_id_set = false;
            for (int p = 0; p < 2 && !any_id_set; ++p) {
                for (int f = 0; f < 6; ++f) {
                    if (dynamic_cube_face_tex_ids_[p][f]) {
                        any_id_set = true;
                        break;
                    }
                }
            }
            if (!any_id_set) {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                    "DynamicCubemap not yet wired into application.");
                ImGui::Spacing();
                ImGui::TextWrapped(
                    "The class skeleton, reprojection compute shader and "
                    "viewer UI are all in place, but the application has "
                    "not yet:");
                ImGui::Bullet(); ImGui::TextWrapped("instantiated DynamicCubemap;");
                ImGui::Bullet(); ImGui::TextWrapped(
                    "registered each face view via Helper::addImTextureID;");
                ImGui::Bullet(); ImGui::TextWrapped(
                    "called setDynamicCubeFaceTextureIds() / setDynamicCubeFrameInfo();");
                ImGui::Bullet(); ImGui::TextWrapped(
                    "called cubemap->update() each frame.");
                ImGui::Spacing();
                ImGui::TextWrapped(
                    "Once that wiring is added, this panel will show "
                    "the live face strip with hover tooltips and "
                    "yellow-bordered \"freshly rendered\" highlights.");
            } else {
                ImGui::Text("Frame %llu, face just rendered: %d",
                    static_cast<unsigned long long>(dynamic_cube_frame_index_),
                    dynamic_cube_current_face_);
                ImGui::Text("Read ping-pong index: %d",
                    dynamic_cube_current_read_idx_);
                ImGui::SliderFloat("Thumbnail size",
                    &dynamic_cube_thumb_size_, 32.0f, 256.0f, "%.0f px");
                ImGui::SliderFloat("Exposure",
                    &dynamic_cube_exposure_, 0.1f, 32.0f, "%.2fx",
                    ImGuiSliderFlags_Logarithmic);
                ImGui::SameLine();
                if (ImGui::Button("1x")) dynamic_cube_exposure_ = 1.0f;

                const ImVec4 tint(
                    dynamic_cube_exposure_,
                    dynamic_cube_exposure_,
                    dynamic_cube_exposure_,
                    1.0f);
                const ImVec4 border_normal(0, 0, 0, 0);
                const ImVec4 border_hot(1.0f, 0.85f, 0.1f, 1.0f); // yellow

                const char* face_names[6] =
                    { "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

                // Helper: draw one labelled row of 6 face thumbnails.
                // `is_read_buffer` controls whether the freshly-rendered-face
                // highlight + capture-position tooltip is applied.
                auto draw_cube_row = [&](
                    const char* row_id, const char* label,
                    int ping_pong_idx,
                    bool is_read_buffer) {
                    ImGui::SeparatorText(label);
                    ImGui::PushID(row_id);
                    const auto& faces =
                        dynamic_cube_face_tex_ids_[ping_pong_idx];
                    for (int f = 0; f < 6; ++f) {
                        if (f > 0) ImGui::SameLine();
                        ImGui::BeginGroup();
                        ImGui::Text("%s", face_names[f]);
                        ImTextureID id = faces[f];
                        const bool is_fresh = is_read_buffer &&
                            (f == dynamic_cube_current_face_);
                        const ImVec4& brd = is_fresh
                            ? border_hot : border_normal;
                        if (id) {
                            ImGui::Image(id, ImVec2(thumb, thumb),
                                ImVec2(0, 0), ImVec2(1, 1), tint, brd);
                            if (ImGui::IsItemHovered()) {
                                ImGui::BeginTooltip();
                                ImGui::Image(id, ImVec2(384.0f, 384.0f),
                                    ImVec2(0, 0), ImVec2(1, 1),
                                    tint, border_normal);
                                const auto& p = dynamic_cube_face_capture_pos_[f];
                                ImGui::Text(
                                    "face %s (%d)\nlast captured at "
                                    "(%.2f, %.2f, %.2f)",
                                    face_names[f], f, p.x, p.y, p.z);
                                if (is_fresh) {
                                    ImGui::TextColored(
                                        ImVec4(1, 0.8f, 0, 1),
                                        "freshly rendered this frame");
                                }
                                ImGui::EndTooltip();
                            }
                        } else {
                            ImGui::Dummy(ImVec2(thumb, thumb));
                        }
                        ImGui::EndGroup();
                    }
                    ImGui::PopID();
                };

                const int read_idx  = dynamic_cube_current_read_idx_;
                const int write_idx = 1 - read_idx;

                draw_cube_row(
                    "##dyn_cube_read", "Read buffer (consumed this frame)",
                    read_idx,  /*is_read_buffer*/ true);
                draw_cube_row(
                    "##dyn_cube_write", "Other ping-pong (next frame's read)",
                    write_idx, /*is_read_buffer*/ false);
            }
        }
        ImGui::End();
    }
    // ------------------------------------------------------------------------

    // ---- GPU Profiler window -----------------------------------------------
    if (s_show_gpu_profiler && gpu_profiler_) {
        gpu_profiler_->setWindowOpen(true);
        gpu_profiler_->drawImGui();
        // The window's [X] clears the profiler's own flag inside
        // drawImGui — mirror it back so the menu checkmark follows and
        // the window actually stays closed next frame.
        s_show_gpu_profiler = gpu_profiler_->windowOpen();
    }

    // ---- AI terrain generation popup (Tools > Generate Terrain) ------------
    drawTerrainGenPopup();

    // ---- Full-size image viewer (double-click an image tile) ---------------
    drawImageViewer();
    // ------------------------------------------------------------------------

    // ---- Plugin windows ----------------------------------------------------
    if (plugin_manager_) {
        plugin_manager_->drawAllImGui();
    }
    // ------------------------------------------------------------------------

    // Chat box / dialogue UI only during gameplay.  In editor mode confine it
    // to the Viewport rect; otherwise it spans the full window (default args).
    if (game_state_ == GameState::InGame) {
        ImVec2 cvp_pos, cvp_size, cvp_c;
        getViewportScreenRect(cvp_pos, cvp_size, cvp_c);
        const glm::vec2 vp_org  = isViewportValid()
            ? glm::vec2(cvp_pos.x, cvp_pos.y) : glm::vec2(0.0f);
        const glm::vec2 vp_size = isViewportValid()
            ? glm::vec2(cvp_size.x, cvp_size.y) : glm::vec2(0.0f);
        chat_box_->draw(cmd_buf, render_pass, framebuffer, screen_size,
            skydome, dump_volume_noise, delta_t, vp_org, vp_size);
    }

    // ── Viewport Play / Stop button (editor only, top-left corner) ──────────
    // Toggles play mode: Play (green ▶) enters play mode (character is
    // controllable, clock visible); Stop (red ■) returns to edit mode (player
    // frozen, camera flies the level).
    //
    // No longer gated on CollisionBuildStatus::Done: collision is BAKE-
    // driven now, so a scene that restores a baked .rwcmap never runs the
    // incremental build (status stays Idle) — the old gate hid the game UI
    // forever.  The button shows whenever the editor is in game; pressing
    // Play without a collision world surfaces the "No collision map" banner
    // (pointing at Tools > Bake Collision Map) instead of silently no-oping.
    if (editor_enabled_ &&
        game_state_ == GameState::InGame) {
        ImVec2 tb_pos, tb_size, tb_c;
        getViewportScreenRect(tb_pos, tb_size, tb_c);
        ImGui::SetNextWindowPos(ImVec2(tb_pos.x + 8.0f, tb_pos.y + 8.0f));
        ImGui::SetNextWindowBgAlpha(0.35f);
        if (ImGui::Begin("##viewport_toolbar", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing)) {
            const float bs = 90.0f;   // 3× the original 30 px button
            const ImVec2 bp = ImGui::GetCursorScreenPos();
            const bool clicked = ImGui::Button("##playstop", ImVec2(bs, bs));
            ImDrawList* d = ImGui::GetWindowDrawList();
            const ImVec2 cc(bp.x + bs * 0.5f, bp.y + bs * 0.5f);
            // Semi-transparent (~55%) so the big 90 px button doesn't
            // block the viewport content underneath; full opacity on
            // hover for clear affordance.
            const int icon_a = ImGui::IsItemHovered() ? 255 : 140;
            if (play_mode_) {
                d->AddRectFilled(ImVec2(cc.x - 18, cc.y - 18),
                                 ImVec2(cc.x + 18, cc.y + 18),
                                 IM_COL32(240, 80, 80, icon_a), 6.0f); // ■ stop
            } else if (play_icon_id_) {
                // assets/icon/play.png (falls back to the drawn triangle below).
                d->AddImage(play_icon_id_, bp, ImVec2(bp.x + bs, bp.y + bs),
                            ImVec2(0, 0), ImVec2(1, 1),
                            IM_COL32(255, 255, 255, icon_a));
            } else {
                d->AddTriangleFilled(ImVec2(cc.x - 15, cc.y - 24),
                                     ImVec2(cc.x - 15, cc.y + 24),
                                     ImVec2(cc.x + 24, cc.y),
                                     IM_COL32(90, 220, 120, icon_a));  // ▶ play
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(play_mode_ ? "Stop — return to edit mode"
                                             : "Play — control the character");
            if (clicked) play_mode_ = !play_mode_;
        }
        ImGui::End();
    }

    // ---- Analog clock overlay (top-left, always on top) --------------------
    // Drawn directly on the foreground draw list so it is always visible
    // above every ImGui window, regardless of window z-order.  Only shown in
    // PLAY mode (the time-of-day clock is a gameplay element).
    if (play_mode_) {
        constexpr float kFaceSize = 96.0f;
        constexpr float kPad      = 4.0f;
        const float kMenuH = ImGui::GetFrameHeight();

        // Top-left anchor: the editor Viewport's top-left (already below the
        // menu bar) or the full window below the menu bar otherwise.
        ImVec2 clk_pos, clk_size, clk_c;
        getViewportScreenRect(clk_pos, clk_size, clk_c);
        // Shifted DOWN by half the face size (leaves room for the Play button
        // above it in the editor viewport's top-left corner).
        const float clk_top =
            (isViewportValid() ? 8.0f : (kMenuH + 8.0f)) + kFaceSize * 0.5f;
        const ImVec2 origin(clk_pos.x + 14.0f, clk_pos.y + clk_top);
        const ImVec2 ctr(origin.x + kPad + kFaceSize * 0.5f,
                         origin.y + kPad + kFaceSize * 0.5f);
        const float  R = kFaceSize * 0.5f;

        ImDrawList* dl = ImGui::GetForegroundDrawList();

        // ── clock face image (or fallback circle) ─────────────────────────
        const ImVec2 img_min(origin.x + kPad, origin.y + kPad);
        const ImVec2 img_max(img_min.x + kFaceSize, img_min.y + kFaceSize);
        if (clock_tex_id_) {
            // Tint alpha = 230 → 90 % opaque
            dl->AddImage(clock_tex_id_, img_min, img_max,
                         ImVec2(0,0), ImVec2(1,1), IM_COL32(255, 255, 255, 230));
        } else {
            dl->AddCircleFilled(ctr, R, IM_COL32(18, 14, 52, 230), 64);
            dl->AddCircle      (ctr, R, IM_COL32(191, 157, 45, 230), 64, 2.0f);
        }

        // ── decompose tod_hours_ into hand angles ─────────────────────────
        const float tod  = tod_hours_;
        const float h12  = std::fmod(tod, 12.0f);
        const float mins = std::fmod(tod * 60.0f, 60.0f);
        constexpr float k2Pi  = 2.0f * 3.14159265f;
        constexpr float kBase = -3.14159265f * 0.5f;   // 12 o'clock at top
        const float ang_h = kBase + (h12  / 12.0f) * k2Pi;
        const float ang_m = kBase + (mins / 60.0f) * k2Pi;

        // ── clock hands ───────────────────────────────────────────────────
        auto drawHand = [&](float ang, float len, float stub,
                            float thick, ImU32 col) {
            const float ca = std::cos(ang), sa = std::sin(ang);
            dl->AddLine(ctr,
                        ImVec2(ctr.x - ca * stub, ctr.y - sa * stub),
                        col, thick * 0.6f);
            dl->AddLine(ctr,
                        ImVec2(ctr.x + ca * len, ctr.y + sa * len),
                        col, thick);
        };
        drawHand(ang_h, R * 0.50f, R * 0.12f, 5.0f, IM_COL32(220, 185,  60, 230));
        drawHand(ang_m, R * 0.75f, R * 0.15f, 3.0f, IM_COL32(210, 210, 210, 230));

        // Centre cap
        dl->AddCircleFilled(ctr, 5.0f, IM_COL32(255, 220,  80, 230), 16);
        dl->AddCircle      (ctr, 5.0f, IM_COL32(120,  90,  20, 230), 16, 1.5f);

        // ── HH:MM digital readout below the dial ─────────────────────────
        const int disp_h = static_cast<int>(tod) % 24;
        const int disp_m = static_cast<int>(mins);
        char timebuf[8];
        snprintf(timebuf, sizeof(timebuf), "%02d:%02d", disp_h, disp_m);
        const ImVec2 txt_sz = ImGui::CalcTextSize(timebuf);
        const ImVec2 txt_pos(ctr.x - txt_sz.x * 0.5f,
                             img_max.y + 3.0f);
        dl->AddText(txt_pos, IM_COL32(242, 209, 89, 230), timebuf);
    }
    // ------------------------------------------------------------------------

    // ── VRAM usage profiler (editor overlay, top-right of viewport) ─────────
    // Driver-reported device-local heap usage vs capacity.  Covers ALL VRAM
    // allocations by this process, ImGui's buffers included (see queryVram).
    if (editor_enabled_) {
        const double now = ImGui::GetTime();
        if (!vram_valid_ || now - vram_last_poll_ > 0.25) {   // throttle to 4 Hz
            // (1) Vulkan: this process's own VRAM (engine + ImGui).
            const VramQuery q = queryVram(getVkPhysicalDevice(device_));
            vram_used_mb_   = q.used_mb;
            vram_budget_mb_ = q.budget_mb;
            vram_total_mb_  = q.total_mb;
            vram_valid_     = q.valid;
            // (2) CUDA: device-wide usage, so ML passes (LibTorch in-process +
            //     the FLUX/Ollama subprocesses) and other apps are included.
            unsigned long long dwf = 0, dwt = 0;
            if (engine::queryDeviceWideVramBytes(dwf, dwt) && dwt != 0) {
                const double MB = 1024.0 * 1024.0;
                vram_dev_total_mb_ = (double)dwt / MB;
                vram_dev_used_mb_  = (double)(dwt - dwf) / MB;
                vram_dev_valid_    = true;
            } else {
                vram_dev_valid_ = false;
            }
            vram_last_poll_ = now;
        }

        // Prefer the device-wide (CUDA) numbers; fall back to Vulkan-only.
        const bool   dev    = vram_dev_valid_ && vram_dev_total_mb_ > 0.0;
        const double cap_mb = dev ? vram_dev_total_mb_ : vram_total_mb_;
        const double use_mb = dev ? vram_dev_used_mb_  : vram_used_mb_;
        const double mine_mb = vram_valid_ ? vram_used_mb_ : 0.0;
        const bool   vram_ok = (dev || vram_valid_) && cap_mb > 0.0;

        // FPS line (always) + VRAM line + bar (when the query succeeded).
        char fbuf[48];
        snprintf(fbuf, sizeof(fbuf), "FPS  %.0f", ImGui::GetIO().Framerate);
        char vbuf[128] = {0};
        if (vram_ok) {
            const double free_mb = std::max(0.0, cap_mb - use_mb);
            if (dev && vram_valid_)
                snprintf(vbuf, sizeof(vbuf),
                         "VRAM %.1f / %.0f GB  ·  engine %.1f  ·  %.1f free",
                         use_mb / 1024.0, cap_mb / 1024.0,
                         mine_mb / 1024.0, free_mb / 1024.0);
            else
                snprintf(vbuf, sizeof(vbuf),
                         "VRAM %.1f / %.0f GB  ·  %.1f free%s",
                         use_mb / 1024.0, cap_mb / 1024.0, free_mb / 1024.0,
                         dev ? "" : "  (engine only)");
        }

        // ── Tidy HUD panel: FPS on top, then (optional) VRAM text + bar ─────
        ImVec2 vp_pos, vp_size, vp_c;
        getViewportScreenRect(vp_pos, vp_size, vp_c);
        const float kMenuH = ImGui::GetFrameHeight();
        const float inpad = 7.0f, gap = 5.0f, bar_h = 9.0f, line_gap = 2.0f;
        const float txt_h = ImGui::GetTextLineHeight();
        const ImVec2 fps_ts  = ImGui::CalcTextSize(fbuf);
        const ImVec2 vram_ts = vram_ok ? ImGui::CalcTextSize(vbuf) : ImVec2(0, 0);
        const float cont_w = std::max({fps_ts.x, vram_ts.x,
                                       vram_ok ? 210.0f : 70.0f});
        const float panel_w = cont_w + inpad * 2.0f;
        float panel_h = inpad * 2.0f + txt_h;                 // FPS line
        if (vram_ok) panel_h += line_gap + txt_h + gap + bar_h;
        const float margin = 10.0f;
        const float right  = vp_pos.x + vp_size.x;
        const float ptop   = vp_pos.y +
            (isViewportValid() ? 8.0f : (kMenuH + 8.0f));
        const ImVec2 p0(right - margin - panel_w, ptop);
        const ImVec2 p1(p0.x + panel_w, p0.y + panel_h);

        ImDrawList* dl = ImGui::GetForegroundDrawList();
        dl->AddRectFilled(p0, p1, IM_COL32(16, 16, 22, 170), 4.0f);
        dl->AddRect      (p0, p1, IM_COL32(0, 0, 0, 120), 4.0f);

        // FPS line.
        const ImVec2 fp(p0.x + inpad, p0.y + inpad);
        dl->AddText(ImVec2(fp.x + 1.0f, fp.y + 1.0f), IM_COL32(0, 0, 0, 190), fbuf);
        dl->AddText(fp, IM_COL32(232, 232, 240, 245), fbuf);

        if (vram_ok) {
            const ImVec2 tp(p0.x + inpad, fp.y + txt_h + line_gap);
            dl->AddText(ImVec2(tp.x + 1.0f, tp.y + 1.0f),
                        IM_COL32(0, 0, 0, 190), vbuf);          // shadow
            dl->AddText(tp, IM_COL32(232, 232, 240, 245), vbuf);

            const ImVec2 b0(p0.x + inpad, tp.y + txt_h + gap);
            const ImVec2 b1(p1.x - inpad, b0.y + bar_h);
            const float  bw = b1.x - b0.x;
            dl->AddRectFilled(b0, b1, IM_COL32(40, 40, 48, 220), 2.0f);

            const float fu = (float)std::min(1.0, std::max(0.0, use_mb / cap_mb));
            const ImU32  used_col = fu < 0.70f ? IM_COL32( 90, 200, 120, 240)
                                  : fu < 0.90f ? IM_COL32(235, 190,  70, 240)
                                               : IM_COL32(235,  90,  80, 240);
            dl->AddRectFilled(b0, ImVec2(b0.x + bw * fu, b1.y), used_col, 2.0f);

            // Engine's own Vulkan share overlaid in blue.
            if (dev && mine_mb > 0.0) {
                const float fm = (float)std::min((double)fu, mine_mb / cap_mb);
                dl->AddRectFilled(b0, ImVec2(b0.x + bw * fm, b1.y),
                                  IM_COL32(80, 160, 245, 245), 2.0f);
            }
            dl->AddRect(b0, b1, IM_COL32(0, 0, 0, 140), 2.0f);
        }
    }
    // ------------------------------------------------------------------------

    renderer::Helper::addImGuiToCommandBuffer(cmd_buf);

    cmd_buf->endRenderPass();

    // Multi-viewport: update and render platform windows (separate OS windows).
    ImGuiIO& vp_io = ImGui::GetIO();
    if (vp_io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }

    return in_focus;
}

void Menu::destroy() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Menu::destroyResources() {
    // Release the background image GPU resources. Called once at
    // application shutdown (NOT during swap chain recreation, where
    // the texture needs to survive).
    if (bg_texture_info_ && device_) {
        bg_texture_info_->destroy(device_);
        bg_texture_info_.reset();
        bg_texture_id_ = ImTextureID(0);
    }
    if (clock_tex_info_ && device_) {
        clock_tex_info_->destroy(device_);
        clock_tex_info_.reset();
        clock_tex_id_ = ImTextureID(0);
    }
    if (play_icon_info_ && device_) {
        play_icon_info_->destroy(device_);
        play_icon_info_.reset();
        play_icon_id_ = ImTextureID(0);
    }
}

// ============================================================================
//  Editor UI (UE-style docked layout)
//
//  A fullscreen, transparent DockSpace host covers the main viewport with a
//  PASS-THROUGH central node, so the existing fullscreen 3D scene shows
//  through the empty centre (the "Viewport"), while real, resizable panels
//  dock around it: Outliner + Details tabbed top-right, Content Browser along
//  the bottom.  No offscreen render target is needed; camera input is already
//  gated by ImGui's WantCaptureMouse whenever the cursor is over a panel.
// ============================================================================
void Menu::getViewportScreenRect(ImVec2& pos, ImVec2& size, ImVec2& center) const {
    ImGuiViewport* mvp = ImGui::GetMainViewport();
    if (viewport_valid_ && editor_enabled_) {
        pos  = ImVec2(mvp->Pos.x + viewport_pos_.x, mvp->Pos.y + viewport_pos_.y);
        size = ImVec2(viewport_size_.x, viewport_size_.y);
    } else {
        pos = mvp->Pos; size = mvp->Size;
    }
    center = ImVec2(pos.x + size.x * 0.5f, pos.y + size.y * 0.5f);
}

ImVec2 Menu::viewportCenter() const {
    ImVec2 p, s, c; getViewportScreenRect(p, s, c); return c;
}

void Menu::drawEditorDockSpace() {
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);

    const ImGuiWindowFlags host_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize   | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDocking;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("##EditorDockHost", nullptr, host_flags);
    ImGui::PopStyleVar(3);

    const ImGuiID dock_id = ImGui::GetID("EditorDockSpace");
    ImGui::DockSpace(dock_id, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);

    // One-time default arrangement: full-width Content Browser along the
    // bottom, Outliner+Details tabbed on the top-right, viewport = the
    // remaining (pass-through) centre/top-left.
    if (!editor_layout_built_) {
        editor_layout_built_ = true;
        ImGui::DockBuilderRemoveNode(dock_id);
        ImGui::DockBuilderAddNode(dock_id,
            ImGuiDockNodeFlags_DockSpace |
            ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::DockBuilderSetNodeSize(dock_id, vp->WorkSize);

        ImGuiID center = dock_id;
        ImGuiID bottom = ImGui::DockBuilderSplitNode(
            center, ImGuiDir_Down, 0.26f, nullptr, &center);
        ImGuiID right = ImGui::DockBuilderSplitNode(
            center, ImGuiDir_Right, 0.24f, nullptr, &center);

        ImGui::DockBuilderDockWindow("Content Browser", bottom);
        ImGui::DockBuilderDockWindow("File Browser", bottom);  // tab w/ Content Browser
        ImGui::DockBuilderDockWindow("Output Log", bottom);    // tab w/ Content Browser
        ImGui::DockBuilderDockWindow("Outliner", right);
        ImGui::DockBuilderDockWindow("Debug Display", right);  // tab w/ Outliner
        ImGui::DockBuilderFinish(dock_id);
    }

    // Track the central node rect (the 3D Viewport region) so the app can
    // match the camera aspect / scene render size to it.  Coords are in the
    // main viewport's screen space; convert to window-local pixels.
    if (ImGuiDockNode* cn = ImGui::DockBuilderGetCentralNode(dock_id)) {
        viewport_pos_  = glm::vec2(cn->Pos.x - vp->Pos.x, cn->Pos.y - vp->Pos.y);
        viewport_size_ = glm::vec2(cn->Size.x, cn->Size.y);
        viewport_valid_ = (cn->Size.x > 4.0f && cn->Size.y > 4.0f);
    } else {
        viewport_valid_ = false;
    }
    ImGui::End();

    drawOutlinerPanel();
    drawDebugDisplayPanel();
    drawContentBrowserPanel();
    drawFileBrowserPanel();
    drawOutputPanel();
    // Shared FLUX.2 popup — drawn ONCE per frame (both browsers can open it).
    drawFluxGeneratePopup();
    // Shared text-to-audio popup — same single-instance rule.
    drawAudioGeneratePopup();
    // Shared text-to-animation popup (Qwen → .anim).
    drawAnimGeneratePopup();

    // ── Drag & drop placement into the 3D viewport ──────────────────────
    // The viewport is the pass-through central dock node (no ImGui window),
    // so a regular drop target can't live there.  And ImGui clears the
    // payload in NewFrame ON the release frame — before this code runs —
    // so we remember the path while the drag is in flight and detect the
    // drop as "payload vanished + mouse released this frame".
    if (const ImGuiPayload* pl = ImGui::GetDragDropPayload();
        pl && pl->IsDataType("RW_CONTENT_ASSET") && pl->Data) {
        // Drag in flight: remember the asset + show the drop hint.
        drag_asset_path_.assign((const char*)pl->Data);
        ImVec2 vp_pos, vp_size, vp_c;
        getViewportScreenRect(vp_pos, vp_size, vp_c);
        const ImVec2 m = ImGui::GetIO().MousePos;
        const bool in_viewport =
            m.x >= vp_pos.x && m.x < vp_pos.x + vp_size.x &&
            m.y >= vp_pos.y && m.y < vp_pos.y + vp_size.y;
        const bool over_panel =
            ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
        if (in_viewport && !over_panel) {
            ImGui::GetForegroundDrawList()->AddText(
                ImVec2(m.x + 14.0f, m.y + 10.0f),
                IM_COL32(120, 255, 140, 255), "+ place in scene");
        }
    } else if (!drag_asset_path_.empty()) {
        // Drag ended this frame (release or cancel).
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            ImVec2 vp_pos, vp_size, vp_c;
            getViewportScreenRect(vp_pos, vp_size, vp_c);
            const ImVec2 m = ImGui::GetIO().MousePos;
            const bool in_viewport =
                m.x >= vp_pos.x && m.x < vp_pos.x + vp_size.x &&
                m.y >= vp_pos.y && m.y < vp_pos.y + vp_size.y;
            const bool over_panel =
                ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
            if (in_viewport && !over_panel) {
                place_asset_request_ = drag_asset_path_;
            }
        }
        drag_asset_path_.clear();
    }

    // ── Viewport right-click context menu ────────────────────────────────
    // The viewport is the pass-through central dock node (no ImGui window),
    // so the popup is opened manually: a right-button CLICK (release within
    // a few pixels of the press — drags are camera-look) over the viewport
    // and not over any panel.
    if (editor_enabled_) {
        const ImGuiIO& vio = ImGui::GetIO();
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            viewport_rclick_pos_ = vio.MousePos;
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right) &&
            !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
            !ImGui::IsAnyItemHovered()) {
            ImVec2 vp_pos, vp_size, vp_c;
            getViewportScreenRect(vp_pos, vp_size, vp_c);
            const ImVec2 m = vio.MousePos;
            const float dx = m.x - viewport_rclick_pos_.x;
            const float dy = m.y - viewport_rclick_pos_.y;
            const bool in_viewport =
                m.x >= vp_pos.x && m.x < vp_pos.x + vp_size.x &&
                m.y >= vp_pos.y && m.y < vp_pos.y + vp_size.y;
            if (in_viewport && (dx * dx + dy * dy) < 16.0f) {
                ImGui::OpenPopup("##viewport_ctx");
            }
        }
        if (ImGui::BeginPopup("##viewport_ctx")) {
            // Player object: organizational node with a skeleton-mesh
            // slot (assign a character from the Content Browser's player
            // folder in Details), placed in front of the camera.
            if (ImGui::MenuItem("Add Player Object")) {
                player_create_request_ = true;
                scene_node_active_     = true;
            }
            // Collision-map object: holds the scene's baked .rwcmap
            // reference (assign one in Details, or Tools > Bake Collision
            // Map fills it in automatically).
            if (ImGui::MenuItem("Add Collision Map")) {
                collision_create_request_ = true;
                scene_node_active_        = true;
            }
            // Capture the CURRENT editor view (position + facing) as a
            // scene camera in the Outliner under World.
            if (ImGui::MenuItem("Add Camera Object")) {
                camera_create_request_ = true;
                scene_node_active_     = true;
            }
            // Background-music object: drag a clip onto it (Details) and
            // set repeat / volume.
            if (ImGui::MenuItem("Add BGM Object")) {
                bgm_create_request_ = true;
                scene_node_active_  = true;
            }
            // Point light for the ReSTIR direct-lighting path: position
            // via the standard transform gizmo; colour / intensity /
            // radius in Details.  Placed 2.5 m in front of the camera.
            if (ImGui::MenuItem("Add Point Light")) {
                light_create_request_ = true;
                scene_node_active_    = true;
            }
            ImGui::EndPopup();
        }
    }
}

void Menu::drawOutlinerPanel() {
    if (ImGui::Begin("Outliner")) {
        ImGui::TextDisabled("%d object(s)", (int)editor_objects_.size());

        // ── Resizable list / details split ────────────────────────────
        // A draggable bar between the object list (top) and the selected
        // object's attributes (bottom).  The list height is persisted in
        // outliner_list_h_ and clamped so neither pane can collapse.
        constexpr float kSplitterH = 6.0f;
        constexpr float kMinPane   = 48.0f;
        const float total = ImGui::GetContentRegionAvail().y;
        if (outliner_list_h_ <= 0.0f) outliner_list_h_ = total * 0.5f;  // default
        float list_h = outliner_list_h_;
        const float max_list = total - kMinPane - kSplitterH;
        if (list_h < kMinPane) list_h = kMinPane;
        if (max_list > kMinPane && list_h > max_list) list_h = max_list;

        // Object list — a tree rooted at "World".  The authored scene
        // (Create Scene / Load Scene) appears as a child node named after
        // the scene, with its objects nested underneath; engine-level
        // objects (player, NPCs, debug drawables) list under World directly.
        // Multi-mesh drawables (the Bistro FBX files) become collapsible
        // GROUPS named after the asset, with their many sub-objects (named
        // mesh nodes) nested underneath; single-mesh objects render as
        // flat leaves.
        ImGui::BeginChild("##outliner_list", ImVec2(0, list_h), true);
        // Deeper per-level indent than the global 18 px style — with the
        // editor's monospace font that was under two characters, which
        // made group/child nesting almost invisible in deep scenes.
        // Scoped to the outliner list only.
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 32.0f);

        // Renders one editor object (leaf or group) at its outliner index.
        auto draw_object_row = [&](int i) {
            EditorSceneObject& eo = editor_objects_[i];
            const std::string base =
                eo.name.empty() ? ("Object " + std::to_string(i)) : eo.name;
            const auto& kids = outlinerChildren(eo);
            const bool dbl = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

            if (kids.empty()) {
                // Leaf object.  Single click selects; double click teleports.
                const std::string label = base + "##obj" + std::to_string(i);
                const bool sel = (editor_selected_ == i && editor_selected_child_ < 0);
                if (ImGui::Selectable(label.c_str(), sel)) {
                    editor_selected_ = i;
                    editor_selected_child_ = -1;
                }
                if (ImGui::IsItemHovered() && dbl) requestEditorFocus(i, -1);
                if (editor_scroll_to_selected_ && sel) ImGui::SetScrollHereY(0.5f);
                return;
            }

            // Group node (FBX file).  Clicking the label (not the arrow)
            // selects the group; expanding lists its sub-objects.
            ImGuiTreeNodeFlags gflags = ImGuiTreeNodeFlags_OpenOnArrow |
                                        ImGuiTreeNodeFlags_SpanAvailWidth;
            if (editor_selected_ == i && editor_selected_child_ < 0)
                gflags |= ImGuiTreeNodeFlags_Selected;
            // "###"-keyed ID (note: "##" still hashes the WHOLE label —
            // including the live sub-object count — so any count change
            // minted a new ID and reopened collapsed groups; "###"
            // restricts the ID to the stable name suffix).
            const std::string glabel = base + "  (" +
                std::to_string(kids.size()) + ")###grp_" + base;
            // Revealing a pick-selected sub-object: force its group open.
            if (editor_scroll_to_selected_ && editor_selected_ == i &&
                editor_selected_child_ >= 0)
                ImGui::SetNextItemOpen(true);
            const bool open = ImGui::TreeNodeEx(glabel.c_str(), gflags);
            if (editor_scroll_to_selected_ && editor_selected_ == i &&
                editor_selected_child_ < 0)
                ImGui::SetScrollHereY(0.5f);
            const bool ghover = ImGui::IsItemHovered();
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                editor_selected_ = i;
                editor_selected_child_ = -1;
            }
            if (ghover && dbl && !ImGui::IsItemToggledOpen())
                requestEditorFocus(i, -1);
            if (open) {
                for (int c = 0; c < (int)kids.size(); ++c) {
                    const bool csel =
                        (editor_selected_ == i && editor_selected_child_ == c);
                    ImGuiTreeNodeFlags cflags =
                        ImGuiTreeNodeFlags_Leaf |
                        ImGuiTreeNodeFlags_NoTreePushOnOpen |
                        ImGuiTreeNodeFlags_SpanAvailWidth;
                    if (csel) cflags |= ImGuiTreeNodeFlags_Selected;
                    const std::string clabel = kids[c].first + "##o" +
                        std::to_string(i) + "_" + std::to_string(c);
                    ImGui::TreeNodeEx(clabel.c_str(), cflags);
                    if (ImGui::IsItemClicked()) {
                        editor_selected_ = i;
                        editor_selected_child_ = c;
                    }
                    if (ImGui::IsItemHovered() && dbl) requestEditorFocus(i, c);
                    if (editor_scroll_to_selected_ && csel)
                        ImGui::SetScrollHereY(0.5f);
                }
                ImGui::TreePop();
            }
        };

        // If a scene-member object is being revealed (scene-view pick),
        // force the World / scene nodes open so the row is reachable.
        if (editor_scroll_to_selected_) {
            ImGui::SetNextItemOpen(true);
        } else {
            ImGui::SetNextItemOpen(true, ImGuiCond_Once);
        }
        if (ImGui::TreeNodeEx("World##world_root",
                              ImGuiTreeNodeFlags_SpanAvailWidth)) {
            // ── Authored scene node (after Create Scene / Load Scene) ────
            // Also derived from the DATA: placed objects (in_scene rows)
            // are only ever rendered inside this node, so placing into a
            // fresh session without clicking "Create Scene" first must
            // still surface it — otherwise the Outliner shows a non-zero
            // object count above an empty tree.
            if (!scene_node_active_) {
                for (const auto& eo : editor_objects_) {
                    if (eo.in_scene) {
                        scene_node_active_ = true;
                        break;
                    }
                }
            }
            if (scene_node_active_) {
                const bool reveal_scene_member =
                    editor_scroll_to_selected_ &&
                    editor_selected_ >= 0 &&
                    editor_selected_ < (int)editor_objects_.size() &&
                    editor_objects_[editor_selected_].in_scene;
                if (reveal_scene_member) {
                    ImGui::SetNextItemOpen(true);
                } else {
                    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
                }
                int scene_count = 0;
                for (const auto& eo : editor_objects_)
                    if (eo.in_scene) ++scene_count;
                // "###" — the live object count must not participate in
                // the ID, or every scene change re-expands a collapsed
                // scene node.
                const std::string slabel =
                    std::string(scene_name_buf_) + "  (" +
                    std::to_string(scene_count) + ")###scene_node";
                ImGuiTreeNodeFlags _sflags =
                    ImGuiTreeNodeFlags_SpanAvailWidth |
                    ImGuiTreeNodeFlags_OpenOnArrow;
                if (editor_scene_root_selected_ && editor_selected_ < 0)
                    _sflags |= ImGuiTreeNodeFlags_Selected;
                const bool _scene_open =
                    ImGui::TreeNodeEx(slabel.c_str(), _sflags);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    editor_scene_root_selected_ = true;
                    editor_selected_       = -1;
                    editor_selected_child_ = -1;
                }
                if (_scene_open) {
                    // Children of a scene row (placed objects under their
                    // source-group node).
                    auto scene_children_of = [&](int pi) {
                        std::vector<int> out;
                        for (int j = 0; j < (int)editor_objects_.size(); ++j)
                            if (editor_objects_[j].in_scene &&
                                editor_objects_[j].parent == pi)
                                out.push_back(j);
                        return out;
                    };
                    for (int i = 0; i < (int)editor_objects_.size(); ++i) {
                        if (!editor_objects_[i].in_scene) continue;
                        if (editor_objects_[i].parent >= 0) continue;
                        const auto group_kids = scene_children_of(i);
                        if (group_kids.empty()) {
                            draw_object_row(i);
                            continue;
                        }
                        // Source-group node: individual placed objects
                        // nest underneath it.
                        ImGuiTreeNodeFlags gflags2 =
                            ImGuiTreeNodeFlags_OpenOnArrow |
                            ImGuiTreeNodeFlags_SpanAvailWidth |
                            ImGuiTreeNodeFlags_DefaultOpen;
                        if (editor_selected_ == i &&
                            editor_selected_child_ < 0)
                            gflags2 |= ImGuiTreeNodeFlags_Selected;
                        // "###"-keyed ID — "##" hashes the whole label
                        // including the live child count, so any count
                        // change minted a new ID and reopened collapsed
                        // groups; "###" keys on the stable name only.
                        const std::string gname2 =
                            editor_objects_[i].name.empty()
                                ? std::string("Group")
                                : editor_objects_[i].name;
                        const std::string glabel2 =
                            gname2 + "  (" +
                            std::to_string(group_kids.size()) +
                            ")###sgrp_" + gname2;
                        const bool gopen =
                            ImGui::TreeNodeEx(glabel2.c_str(), gflags2);
                        if (ImGui::IsItemClicked() &&
                            !ImGui::IsItemToggledOpen()) {
                            editor_selected_       = i;
                            editor_selected_child_ = -1;
                        }
                        if (gopen) {
                            for (int j : group_kids) draw_object_row(j);
                            ImGui::TreePop();
                        }
                    }
                    if (scene_count == 0)
                        ImGui::TextDisabled("(empty scene)");
                    ImGui::TreePop();
                }
            }

            // ── Engine-level objects (player, NPCs, debug drawables) ─────
            for (int i = 0; i < (int)editor_objects_.size(); ++i)
                if (!editor_objects_[i].in_scene) draw_object_row(i);

            ImGui::TreePop();
        }
        editor_scroll_to_selected_ = false;   // one-shot consumed this draw
        if (editor_objects_.empty() && !scene_node_active_)
            ImGui::TextDisabled("(no scene objects yet)");
        ImGui::PopStyleVar();   // IndentSpacing (outliner list scope)
        ImGui::EndChild();
        // Dropping a content asset anywhere on the Outliner list adds it to
        // the scene — same as dropping into the viewport / Add to Scene.
        // (EndChild submits the list as an item, so it can host a target.)
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* pl =
                    ImGui::AcceptDragDropPayload("RW_CONTENT_ASSET")) {
                if (pl->Data) {
                    place_asset_request_.assign(
                        (const char*)pl->Data);
                }
            }
            ImGui::EndDragDropTarget();
        }

        // Draggable splitter bar.
        ImGui::InvisibleButton("##outliner_splitter", ImVec2(-1.0f, kSplitterH));
        const bool split_active  = ImGui::IsItemActive();
        const bool split_hovered = ImGui::IsItemHovered();
        if (split_active || split_hovered)
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        if (split_active)
            outliner_list_h_ = list_h + ImGui::GetIO().MouseDelta.y;
        {   // grip line
            const ImVec2 rmin = ImGui::GetItemRectMin();
            const ImVec2 rmax = ImGui::GetItemRectMax();
            const float  cy   = (rmin.y + rmax.y) * 0.5f;
            const ImU32  col  = ImGui::GetColorU32(
                split_active  ? ImGuiCol_SeparatorActive  :
                split_hovered ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator);
            ImGui::GetWindowDrawList()->AddLine(
                ImVec2(rmin.x + 2.0f, cy), ImVec2(rmax.x - 2.0f, cy), col, 2.0f);
        }

        // Attributes of the highlighted object (fills the rest).
        ImGui::SeparatorText("Details");
        ImGui::BeginChild("##outliner_details", ImVec2(0, 0), false);
        drawDetailsContent();
        ImGui::EndChild();
    }
    ImGui::End();
}

// Build (once, when the object is ready) and cache the list of sub-object
// names for a drawable.  A drawable is treated as a GROUP only if it has more
// than one mesh node (so single-mesh objects like the player stay flat).
// Returns an empty list for not-yet-ready or single-mesh objects.
const std::vector<std::pair<std::string, int>>&
Menu::outlinerChildren(const EditorSceneObject& eo) {
    static const std::vector<std::pair<std::string, int>> kEmpty;
    if (!eo.obj) return kEmpty;
    auto it = outliner_children_cache_.find(eo.obj);
    if (it != outliner_children_cache_.end()) return it->second;
    if (!eo.obj->isReady()) return kEmpty;   // not loaded yet — retry next frame

    std::vector<std::pair<std::string, int>> kids;
    const auto& data = eo.obj->getDrawableData();
    int mesh_nodes = 0;
    for (const auto& n : data.nodes_) if (n.mesh_idx_ >= 0) ++mesh_nodes;
    // A wrapper restricted to ONE sub-object (.rwobj placement) is a leaf in
    // the Outliner even though its shared DrawableData has many mesh nodes.
    if (mesh_nodes > 1 && eo.obj->onlyRenderSubObject() < 0) {
        kids.reserve((size_t)mesh_nodes);
        for (int ni = 0; ni < (int)data.nodes_.size(); ++ni) {
            if (data.nodes_[ni].mesh_idx_ < 0) continue;
            std::string nm = data.nodes_[ni].name_;
            if (nm.empty()) nm = "node " + std::to_string(ni);
            kids.emplace_back(std::move(nm), ni);
        }
    }
    auto res = outliner_children_cache_.emplace(eo.obj, std::move(kids));
    return res.first->second;
}

// World-space AABB of the selected object (group) or sub-object node.
bool Menu::selectedWorldAabb(glm::vec3& bmin, glm::vec3& bmax) {
    if (editor_selected_ < 0 ||
        editor_selected_ >= (int)editor_objects_.size())
        return false;
    EditorSceneObject& eo = editor_objects_[editor_selected_];
    if (!eo.obj || !eo.obj->isReady()) return false;

    glm::vec3 lmin, lmax;   // local/model-space box
    glm::mat4 world(1.0f);

    // Per-instance world (render path: instance_world * cached_matrix_).
    glm::mat4 inst(1.0f);
    if (eo.obj->hasInstanceRoot()) {
        inst = glm::translate(glm::mat4(1.0f),
                              eo.obj->getInstanceRootTranslation()) *
               glm::mat4_cast(eo.obj->getInstanceRootRotation()) *
               glm::scale(glm::mat4(1.0f),
                          eo.obj->getInstanceRootScale());
    }

    const auto& kids = outlinerChildren(eo);
    if (editor_selected_child_ >= 0 &&
        editor_selected_child_ < (int)kids.size()) {
        // Sub-object node: mesh bbox in the node's world matrix.
        const auto& data = eo.obj->getDrawableData();
        const int ni = kids[editor_selected_child_].second;
        if (ni < 0 || ni >= (int)data.nodes_.size()) return false;
        const auto& node = data.nodes_[ni];
        if (node.mesh_idx_ < 0 ||
            node.mesh_idx_ >= (int)data.meshes_.size()) return false;
        const auto& mesh = data.meshes_[node.mesh_idx_];
        if (mesh.bbox_min_.x > mesh.bbox_max_.x) return false;
        lmin = mesh.bbox_min_; lmax = mesh.bbox_max_;
        world = inst * node.cached_matrix_;
    } else if (eo.obj->onlyRenderSubObject() >= 0) {
        // Placed sub-object wrapper: frame ITS node, not the whole shared
        // source file (whose bbox spans e.g. all of Bistro).
        const auto& data = eo.obj->getDrawableData();
        int k = 0, ni = -1;
        for (int i = 0; i < (int)data.nodes_.size(); ++i) {
            if (data.nodes_[i].mesh_idx_ < 0) continue;
            if (k == eo.obj->onlyRenderSubObject()) { ni = i; break; }
            ++k;
        }
        if (ni < 0) return false;
        const auto& node = data.nodes_[ni];
        if (node.mesh_idx_ < 0 ||
            node.mesh_idx_ >= (int)data.meshes_.size()) return false;
        const auto& mesh = data.meshes_[node.mesh_idx_];
        if (mesh.bbox_min_.x > mesh.bbox_max_.x) return false;
        lmin = mesh.bbox_min_; lmax = mesh.bbox_max_;
        world = inst * node.cached_matrix_;
    } else if (eo.obj->getSkinnedModelAabb(lmin, lmax)) {
        // Skinned character: box from JOINT positions, not the raw
        // mesh bbox.  The pre-skin accessor bounds live in mesh space
        // and for sibling-armature exports (Mixamo/Blender) sit far
        // from where the skinned body actually renders — the joint-
        // derived box follows the same joint.cached * inv_bind math
        // the vertex shader uses, so it hugs the on-screen body.
        world = eo.obj->hasInstanceRoot() ? inst : eo.obj->getLocation();
    } else {
        // Whole object: model bbox in the instance world transform.
        lmin = eo.obj->getModelBboxMin();
        lmax = eo.obj->getModelBboxMax();
        if (lmin.x > lmax.x) return false;
        world = eo.obj->hasInstanceRoot() ? inst : eo.obj->getLocation();
    }

    // Transform the 8 corners and re-fit an axis-aligned world box.
    const float Xc[2] = { lmin.x, lmax.x };
    const float Yc[2] = { lmin.y, lmax.y };
    const float Zc[2] = { lmin.z, lmax.z };
    bmin = glm::vec3(std::numeric_limits<float>::max());
    bmax = glm::vec3(std::numeric_limits<float>::lowest());
    for (int i = 0; i < 8; ++i) {
        glm::vec4 w = world * glm::vec4(Xc[i & 1], Yc[(i >> 1) & 1],
                                        Zc[(i >> 2) & 1], 1.0f);
        glm::vec3 p(w);
        bmin = glm::min(bmin, p);
        bmax = glm::max(bmax, p);
    }
    return bmax.x >= bmin.x;
}

// Convex-hull outline of the selected object's projected mesh vertices.
bool Menu::selectedScreenHull(std::vector<ImVec2>& hull) {
    hull.clear();
    if (editor_selected_ < 0 ||
        editor_selected_ >= (int)editor_objects_.size())
        return false;
    EditorSceneObject& eo = editor_objects_[editor_selected_];
    if (!eo.obj || !eo.obj->isReady()) return false;
    const auto& data = eo.obj->getDrawableData();

    const auto& kids = outlinerChildren(eo);
    const bool is_group = !kids.empty();
    const bool child_sel = (editor_selected_child_ >= 0 &&
                            editor_selected_child_ < (int)kids.size());
    // Groups with no chosen sub-object would mean projecting the whole FBX
    // (hundreds of thousands of verts) — skip and let the caller use the bbox.
    if (is_group && !child_sel) return false;

    ImVec2 vp_pos, vp_size, vp_c;
    getViewportScreenRect(vp_pos, vp_size, vp_c);
    const float sw = vp_size.x, sh = vp_size.y;

    // Per-instance world (render path: instance_world * cached_matrix_).
    glm::mat4 inst(1.0f);
    if (eo.obj->hasInstanceRoot()) {
        inst = glm::translate(glm::mat4(1.0f),
                              eo.obj->getInstanceRootTranslation()) *
               glm::mat4_cast(eo.obj->getInstanceRootRotation()) *
               glm::scale(glm::mat4(1.0f),
                          eo.obj->getInstanceRootScale());
    }

    std::vector<ImVec2> pts;
    auto projectMesh = [&](int mesh_idx, const glm::mat4& world) {
        if (mesh_idx < 0 || mesh_idx >= (int)data.meshes_.size()) return;
        const auto& mesh = data.meshes_[mesh_idx];
        if (!mesh.vertex_position_) return;
        const auto& V = *mesh.vertex_position_;
        // Cap the vertex count we project so huge meshes stay cheap.
        const size_t stride = (V.size() > 4000) ? (V.size() / 4000 + 1) : 1;
        for (size_t i = 0; i < V.size(); i += stride) {
            glm::vec4 clip = player_view_proj_ * (world * glm::vec4(V[i], 1.0f));
            if (clip.w < 1e-3f) continue;          // behind camera
            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            pts.push_back(ImVec2((ndc.x * 0.5f + 0.5f) * sw + vp_pos.x,
                                 (ndc.y * 0.5f + 0.5f) * sh + vp_pos.y));
        }
    };

    if (child_sel) {
        const int ni = kids[editor_selected_child_].second;
        if (ni >= 0 && ni < (int)data.nodes_.size())
            projectMesh(data.nodes_[ni].mesh_idx_,
                        inst * data.nodes_[ni].cached_matrix_);
    } else if (eo.obj->onlyRenderSubObject() >= 0) {
        // Placed sub-object wrapper: project ONLY its filtered node — the
        // shared DrawableData is the whole source file, and walking every
        // node here ran per frame while selected.
        int k = 0;
        for (const auto& node : data.nodes_) {
            if (node.mesh_idx_ < 0) continue;
            if (k == eo.obj->onlyRenderSubObject()) {
                projectMesh(node.mesh_idx_, inst * node.cached_matrix_);
                break;
            }
            ++k;
        }
    } else {
        // Single-mesh object (player / NPC / standalone drawable).
        for (const auto& node : data.nodes_)
            if (node.mesh_idx_ >= 0)
                projectMesh(node.mesh_idx_, inst * node.cached_matrix_);
    }
    if (pts.size() < 3) return false;

    // Andrew's monotone-chain convex hull (counter-clockwise).
    std::sort(pts.begin(), pts.end(), [](const ImVec2& a, const ImVec2& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y); });
    auto cross = [](const ImVec2& O, const ImVec2& A, const ImVec2& B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x); };
    const int n = (int)pts.size();
    std::vector<ImVec2> H(2 * n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(H[k - 2], H[k - 1], pts[i]) <= 0.0f) --k;
        H[k++] = pts[i];
    }
    for (int i = n - 2, t = k + 1; i >= 0; --i) {
        while (k >= t && cross(H[k - 2], H[k - 1], pts[i]) <= 0.0f) --k;
        H[k++] = pts[i];
    }
    H.resize((size_t)std::max(0, k - 1));
    if (H.size() < 3) return false;
    hull = std::move(H);
    return true;
}

// Regenerate the exact silhouette mask when the selection or camera changed.
void Menu::updateSelectionMask() {
    ++sel_mask_frame_;
    // Free retired mask textures whose in-flight window has elapsed.
    for (auto it = sel_mask_dead_.begin(); it != sel_mask_dead_.end();) {
        if (it->free_frame <= sel_mask_frame_) {
            if (it->id)
                ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)it->id);
            it = sel_mask_dead_.erase(it);
        } else {
            ++it;
        }
    }

    auto dropCurrent = [&]() {
        if (sel_mask_id_ || sel_mask_tex_)
            sel_mask_dead_.push_back(
                { sel_mask_frame_ + 4, sel_mask_id_, sel_mask_tex_ });
        sel_mask_id_ = 0; sel_mask_tex_.reset();
        sel_mask_obj_ = nullptr; sel_mask_node_ = -3;
    };

    if (!editor_enabled_ || !device_ || !sampler_) { dropCurrent(); return; }

    engine::game_object::DrawableObject* obj = nullptr;
    int node_idx = -1;
    if (!getSelectedHighlight(obj, node_idx) || !obj || !obj->isReady()) {
        dropCurrent();
        return;
    }

    ImVec2 vp_pos, vp_size, vp_c;
    getViewportScreenRect(vp_pos, vp_size, vp_c);
    const glm::vec2 vpsz(vp_size.x, vp_size.y);
    if (vpsz.x < 8.0f || vpsz.y < 8.0f) return;

    // The render path composes instance_world * node.cached_matrix_ —
    // mirror that here, otherwise placed objects' masks appear at the
    // raw geometry position (identity node for native baked assets)
    // instead of where the instance actually sits.  Part of the cache
    // key too, so dragging the object refreshes the mask.
    glm::mat4 inst(1.0f);
    if (obj->hasInstanceRoot()) {
        inst = glm::translate(glm::mat4(1.0f),
                              obj->getInstanceRootTranslation()) *
               glm::mat4_cast(obj->getInstanceRootRotation()) *
               glm::scale(glm::mat4(1.0f), obj->getInstanceRootScale());
    }

    const bool changed =
        (obj != sel_mask_obj_) || (node_idx != sel_mask_node_) ||
        (player_view_proj_ != sel_mask_vp_) || (vpsz != sel_mask_vpsz_) ||
        (inst != sel_mask_inst_);
    if (!changed) return;   // reuse the cached mask

    // ── Build a world-space triangle mesh from the selected node(s) ──
    plugins::auto_rig::TriangleMesh mesh;
    const auto& data = obj->getDrawableData();
    auto addNode = [&](int ni) {
        if (ni < 0 || ni >= (int)data.nodes_.size()) return;
        const auto& node = data.nodes_[ni];
        if (node.mesh_idx_ < 0 || node.mesh_idx_ >= (int)data.meshes_.size()) return;
        const auto& m = data.meshes_[node.mesh_idx_];
        if (!m.vertex_position_) return;
        const auto& V = *m.vertex_position_;
        const glm::mat4 W = inst * node.cached_matrix_;
        const uint32_t base = (uint32_t)mesh.positions.size();
        mesh.positions.reserve(mesh.positions.size() + V.size());
        for (const auto& v : V)
            mesh.positions.push_back(glm::vec3(W * glm::vec4(v, 1.0f)));
        for (const auto& prim : m.primitives_) {
            if (!prim.vertex_indices_) continue;
            for (int32_t idx : *prim.vertex_indices_)
                mesh.indices.push_back(base + (uint32_t)idx);
        }
    };
    // A placed sub-object wrapper shares the WHOLE source file's
    // DrawableData — rasterising all of it (e.g. the entire Bistro) froze
    // the app for seconds on selection.  Restrict to the wrapper's single
    // filtered node.
    int eff_node = node_idx;
    if (obj->onlyRenderSubObject() >= 0) {
        int k = 0;
        eff_node = -1;
        for (int ni = 0; ni < (int)data.nodes_.size(); ++ni) {
            if (data.nodes_[ni].mesh_idx_ < 0) continue;
            if (k == obj->onlyRenderSubObject()) { eff_node = ni; break; }
            ++k;
        }
        if (eff_node < 0) {
            sel_mask_obj_ = obj; sel_mask_node_ = node_idx;
            sel_mask_vp_ = player_view_proj_; sel_mask_vpsz_ = vpsz;
            sel_mask_inst_ = inst;
            dropCurrent();
            return;
        }
    }

    if (eff_node >= 0) addNode(eff_node);
    else for (int ni = 0; ni < (int)data.nodes_.size(); ++ni)
        if (data.nodes_[ni].mesh_idx_ >= 0) addNode(ni);

    // Record the change key now so an unrenderable selection isn't retried
    // every frame.
    sel_mask_obj_ = obj; sel_mask_node_ = node_idx;
    sel_mask_vp_ = player_view_proj_; sel_mask_vpsz_ = vpsz;
    sel_mask_inst_ = inst;

    if (mesh.positions.empty() || mesh.indices.size() < 3) { dropCurrent(); return; }

    // Triangle budget: the mask is a CPU rasterisation that re-runs on
    // every camera move while selected — beyond this it costs frames (or
    // seconds).  Oversized selections fall back to the hull/bbox highlight.
    constexpr size_t kMaxMaskIndices = 600'000;   // ~200k triangles
    if (mesh.indices.size() > kMaxMaskIndices) { dropCurrent(); return; }

    mesh.recomputeBounds();
    mesh.recomputeNormals();   // rasteriser shades with normals — avoid empties

    // ── Rasterise the silhouette from the CURRENT camera ──────────────────
    // The mesh is already in world space, so pass identity as the view and the
    // camera's view_proj as the projection.  Cap the resolution to bound cost.
    float scale = 1.0f;
    const float mxdim = std::max(vpsz.x, vpsz.y);
    if (mxdim > 1280.0f) scale = 1280.0f / mxdim;
    const int rw = std::max(16, (int)(vpsz.x * scale));
    const int rh = std::max(16, (int)(vpsz.y * scale));
    plugins::auto_rig::SimpleRasterizer rast;
    plugins::auto_rig::ViewCapture cap =
        rast.render(mesh, rw, rh, glm::mat4(1.0f), player_view_proj_);
    if ((int)cap.silhouette.size() < rw * rh) { dropCurrent(); return; }

    // ── Amber RGBA from the coverage mask ──
    std::vector<uint8_t> rgba((size_t)rw * rh * 4, 0);
    for (int i = 0; i < rw * rh; ++i) {
        if (cap.silhouette[i]) {
            rgba[i*4+0] = 255; rgba[i*4+1] = 190; rgba[i*4+2] = 50; rgba[i*4+3] = 110;
        }
    }

    // ── Upload + register (ImGui's own descriptor pool) ──
    std::shared_ptr<renderer::TextureInfo> info;
    ImTextureID id = 0;
    try {
        info = std::make_shared<renderer::TextureInfo>();
        info->mip_levels = 1;
        renderer::Helper::create2DTextureImage(
            device_, renderer::Format::R8G8B8A8_UNORM, rw, rh, rgba.data(),
            info->image, info->memory, std::source_location::current());
        info->view = device_->createImageView(
            info->image, renderer::ImageViewType::VIEW_2D,
            renderer::Format::R8G8B8A8_UNORM,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current());
        if (info->view)
            id = renderer::Helper::addImTextureID(sampler_, info->view);
    } catch (...) {
        info.reset();
        id = 0;
    }

    // Retire the previous mask, install the new one.
    if (sel_mask_id_ || sel_mask_tex_)
        sel_mask_dead_.push_back(
            { sel_mask_frame_ + 4, sel_mask_id_, sel_mask_tex_ });
    sel_mask_tex_ = info; sel_mask_id_ = id; sel_mask_w_ = rw; sel_mask_h_ = rh;
}

void Menu::requestEditorFocus(int obj_idx, int child_idx) {
    editor_selected_       = obj_idx;
    editor_selected_child_ = child_idx;

    // Double-clicking a CAMERA object looks THROUGH it: stage the camera's
    // exact pose for the app to snap the view camera to (instead of the
    // orbit-the-bbox focus used for meshes).
    if (obj_idx >= 0 && obj_idx < (int)editor_objects_.size() &&
        editor_objects_[obj_idx].is_camera &&
        editor_objects_[obj_idx].scene_xform) {
        const auto& xf = *editor_objects_[obj_idx].scene_xform;
        const glm::mat3 R = glm::mat3_cast(xf.rotation);
        editor_campose_pos_     = xf.translation;
        editor_campose_dir_     = -R[2];   // camera looks down local -Z
        editor_campose_follow_  =
            editor_objects_[obj_idx].follow_link
                ? *editor_objects_[obj_idx].follow_link : -1;
        editor_campose_pending_ = true;
        return;
    }

    glm::vec3 bmin, bmax;
    if (!selectedWorldAabb(bmin, bmax)) return;
    editor_focus_center_ = (bmin + bmax) * 0.5f;
    editor_focus_radius_ =
        glm::max(0.25f, glm::length(bmax - bmin) * 0.5f);
    editor_focus_pending_ = true;
}

bool Menu::getSelectedHighlight(engine::game_object::DrawableObject*& obj,
                                int& node_idx) {
    if (editor_selected_ < 0 ||
        editor_selected_ >= (int)editor_objects_.size())
        return false;
    obj = editor_objects_[editor_selected_].obj;
    if (!obj) return false;
    const auto& kids = outlinerChildren(editor_objects_[editor_selected_]);
    if (editor_selected_child_ >= 0 &&
        editor_selected_child_ < (int)kids.size())
        node_idx = kids[editor_selected_child_].second;  // a specific sub-object
    else if (kids.empty())
        node_idx = -2;   // single-mesh object → highlight the whole thing
    else
        node_idx = -1;   // group selected, no sub-object → bbox only, no tint
    return true;
}

void Menu::selectByPick(engine::game_object::DrawableObject* obj, int node_idx) {
    if (!obj) return;
    for (int i = 0; i < (int)editor_objects_.size(); ++i) {
        if (editor_objects_[i].obj != obj) continue;
        editor_selected_       = i;
        editor_selected_child_ = -1;
        if (node_idx >= 0) {
            const auto& kids = outlinerChildren(editor_objects_[i]);
            for (int c = 0; c < (int)kids.size(); ++c) {
                if (kids[c].second == node_idx) {
                    editor_selected_child_ = c;
                    break;
                }
            }
        }
        editor_scroll_to_selected_ = true;   // reveal it in the list next draw
        return;
    }
}

void Menu::drawDetailsContent() {
    // ── Scene root (Outliner "Scene" node): Name + Transform, like an object.
    if (editor_selected_ < 0 && editor_scene_root_selected_) {
        ImGui::SetNextItemWidth(220.0f);
        const bool entered = ImGui::InputText(
            "Name", scene_name_buf_, sizeof(scene_name_buf_),
            ImGuiInputTextFlags_EnterReturnsTrue);
        if ((entered || ImGui::IsItemDeactivatedAfterEdit()) &&
            scene_name_buf_[0] != '\0') {
            scene_root_rename_pending_ = true;
            scene_root_name_ = scene_name_buf_;
        }
        ImGui::Separator();
        if (scene_root_xform_) {
            ImGui::SeparatorText("Transform");
            glm::vec3 pos = scene_root_xform_->translation;
            glm::vec3 scl = scene_root_xform_->scale;
            glm::vec3 euler_deg =
                glm::degrees(glm::eulerAngles(scene_root_xform_->rotation));
            bool changed = false;
            changed = ImGui::DragFloat3("Offset",   &pos.x, 0.05f) || changed;
            changed = ImGui::DragFloat3("Rotation", &euler_deg.x, 0.5f) || changed;
            changed = ImGui::DragFloat3("Scale",    &scl.x, 0.01f, 0.001f,
                                        1000.0f) || changed;
            if (changed) {
                scene_root_xform_->translation = pos;
                scene_root_xform_->rotation = glm::quat(glm::radians(euler_deg));
                scene_root_xform_->scale    = scl;
                scene_xform_dirty_ = true;
            }
        } else {
            ImGui::TextDisabled("(no scene loaded)");
        }
        return;
    }
    if (editor_selected_ >= 0 &&
        editor_selected_ < (int)editor_objects_.size()) {
        EditorSceneObject& o = editor_objects_[editor_selected_];

        // Sub-object (FBX node) selected within a group: show which one, then
        // fall through to the parent FBX's transform/visibility below.
        const auto& kids = outlinerChildren(o);
        if (editor_selected_child_ >= 0 &&
            editor_selected_child_ < (int)kids.size()) {
            ImGui::Text("Object: %s", kids[editor_selected_child_].first.c_str());
            ImGui::TextDisabled("in group: %s", o.name.c_str());
            ImGui::Separator();
            // (Per-node transform editing is a follow-up; the group transform
            // below moves the whole FBX.)
        } else {
            if (o.in_scene && o.scene_index >= 0) {
                // Scene objects (placed meshes, groups, cameras, players)
                // are renameable: commit on Enter or focus loss; the app
                // writes the new name into scene_.objects (persisted via
                // scene Save).
                static int  s_name_idx = -1;
                static char s_name_buf[128] = {0};
                if (s_name_idx != o.scene_index) {
                    s_name_idx = o.scene_index;
                    std::snprintf(s_name_buf, sizeof(s_name_buf), "%s",
                                  o.name.c_str());
                }
                ImGui::SetNextItemWidth(220.0f);
                const bool entered = ImGui::InputText(
                    "Name", s_name_buf, sizeof(s_name_buf),
                    ImGuiInputTextFlags_EnterReturnsTrue);
                if ((entered || ImGui::IsItemDeactivatedAfterEdit()) &&
                    s_name_buf[0] != '\0' && o.name != s_name_buf) {
                    scene_rename_pending_ = true;
                    scene_rename_idx_     = o.scene_index;
                    scene_rename_name_    = s_name_buf;
                }
            } else {
                ImGui::Text("Name: %s", o.name.c_str());
            }
            if (!kids.empty())
                ImGui::TextDisabled("%d sub-objects", (int)kids.size());
            ImGui::Separator();
        }

        if (o.obj) {
            ImGui::SeparatorText("Transform");
            // Per-instance world override (offset / rotation / scale).  Fall
            // back to the createAsync location translation if no instance
            // override has been set yet (e.g. player / npc).
            glm::vec3 pos, scl;
            glm::quat rot;
            if (o.obj->hasInstanceRoot()) {
                pos = o.obj->getInstanceRootTranslation();
                rot = o.obj->getInstanceRootRotation();
                scl = o.obj->getInstanceRootScale();
            } else {
                const glm::mat4& m = o.obj->getLocation();
                pos = glm::vec3(m[3].x, m[3].y, m[3].z);
                rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                scl = glm::vec3(1.0f);
            }
            glm::vec3 euler_deg = glm::degrees(glm::eulerAngles(rot));
            bool changed = false;
            changed = ImGui::DragFloat3("Offset",   &pos.x, 0.05f) || changed;
            changed = ImGui::DragFloat3("Rotation", &euler_deg.x, 0.5f) || changed;
            changed = ImGui::DragFloat3("Scale",    &scl.x, 0.01f, 0.001f, 1000.0f) || changed;
            if (changed) {
                o.obj->setInstanceRootTransform(
                    pos, glm::quat(glm::radians(euler_deg)), scl);
            }
            ImGui::SeparatorText("Rendering");
            bool vis = o.obj->isVisible();
            if (ImGui::Checkbox("Visible", &vis)) o.obj->setVisible(vis);
            ImGui::Text("Loaded: %s", o.obj->isReady() ? "yes" : "(streaming)");

            if (anim_has_) {
                ImGui::SeparatorText("Animation");
                bool playing = anim_playing_;
                if (ImGui::Checkbox("Play", &playing)) {
                    anim_playing_ = playing; anim_ctrl_pending_ = true;
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                float speed = anim_speed_;
                if (ImGui::SliderFloat("Speed", &speed, 0.0f, 3.0f)) {
                    anim_speed_ = speed; anim_ctrl_pending_ = true;
                }
                float t = anim_time_;
                const float maxt = anim_dur_ > 0.0f ? anim_dur_ : 1.0f;
                if (ImGui::SliderFloat("Time", &t, 0.0f, maxt, "%.2fs")) {
                    anim_time_ = t; anim_scrub_ = true; anim_ctrl_pending_ = true;
                }
            }
        } else if (o.scene_xform) {
            // ── Group node: edit the GROUP transform ─────────────────────
            // Children store local-to-parent transforms; the app composes
            // parent * local into each child drawable when this changes —
            // so dragging these values moves/rotates the whole group.
            ImGui::SeparatorText("Group Transform");
            glm::vec3 pos = o.scene_xform->translation;
            glm::vec3 scl = o.scene_xform->scale;
            glm::vec3 euler_deg =
                glm::degrees(glm::eulerAngles(o.scene_xform->rotation));
            bool changed = false;
            changed = ImGui::DragFloat3("Offset",   &pos.x, 0.05f) || changed;
            changed = ImGui::DragFloat3("Rotation", &euler_deg.x, 0.5f) || changed;
            changed = ImGui::DragFloat3("Scale",    &scl.x, 0.01f,
                                        0.001f, 1000.0f) || changed;
            if (changed) {
                o.scene_xform->translation = pos;
                o.scene_xform->rotation    = glm::quat(glm::radians(euler_deg));
                o.scene_xform->scale       = scl;
                scene_xform_dirty_ = true;
            }

            // ── Player: skeleton-mesh assignment ──────────────────────
            // Players are created EMPTY; pick a character (imported into
            // the Content Browser's player folder) here to attach it —
            // or None to detach.  Mirrors the camera's Follow link.
            if (o.is_player) {
                ImGui::SeparatorText("Player");
                std::string cur = "None";
                for (const auto& ce : editor_objects_) {
                    if (ce.in_scene && ce.parent == editor_selected_) {
                        cur = ce.name;
                        break;
                    }
                }
                if (ImGui::BeginCombo("Skeleton Mesh", cur.c_str())) {
                    if (ImGui::Selectable("None", cur == "None")) {
                        player_mesh_assign_pending_ = true;
                        player_mesh_assign_idx_  = o.scene_index;
                        player_mesh_assign_path_.clear();
                    }
                    // Characters under content/player/ (import.rwmeta
                    // with type=character).  CONTENT-ONLY RENDERING:
                    // resolve to the baked character manifest
                    // (<group>/<leaf>.rwchar) written at import — the
                    // original model (main=/source=) is import INPUT
                    // only and is never loaded for rendering.  Groups
                    // imported before the .rwchar bake existed won't
                    // have one: they're skipped here and need a
                    // re-import.
                    namespace fs = std::filesystem;
                    std::error_code ec;
                    for (auto& e :
                         fs::directory_iterator("content/player", ec)) {
                        if (!e.is_directory(ec)) continue;
                        std::ifstream meta(
                            (e.path() / "import.rwmeta").string());
                        if (!meta) continue;
                        std::string line;
                        bool is_char = false;
                        while (std::getline(meta, line)) {
                            if (line == "type=character") { is_char = true; break; }
                        }
                        std::string model_path;
                        {
                            const std::string leaf =
                                e.path().filename().string();
                            const fs::path rwchar =
                                e.path() / (leaf + ".rwchar");
                            if (fs::exists(rwchar, ec))
                                model_path = rwchar.string();
                        }
                        if (!is_char || model_path.empty()) continue;
                        const std::string label =
                            e.path().filename().string();
                        if (ImGui::Selectable(label.c_str(),
                                              cur.rfind(label, 0) == 0)) {
                            player_mesh_assign_pending_ = true;
                            player_mesh_assign_idx_  = o.scene_index;
                            player_mesh_assign_path_ = model_path;
                        }
                    }
                    ImGui::EndCombo();
                }
                // The mesh slot is ALSO a drag-drop target: drag a
                // character (its group folder or the model file) from
                // the Content Browser straight onto the combo.
                if (ImGui::BeginDragDropTarget()) {
                    if (const ImGuiPayload* pl =
                            ImGui::AcceptDragDropPayload(
                                "RW_CONTENT_ASSET")) {
                        std::string p((const char*)pl->Data);
                        namespace fs = std::filesystem;
                        std::error_code ec;
                        // A FILE dragged from inside a character group
                        // (e.g. one of its baked objects/*.rwgeo tiles)
                        // resolves UP to the group folder — the WHOLE
                        // character attaches as one unit (a single
                        // skeleton is a group of skeleton meshes; you
                        // can't bind just one of them).
                        if (!fs::is_directory(p, ec)) {
                            fs::path dir = fs::path(p).parent_path();
                            for (int up = 0;
                                 up < 3 && !dir.empty();
                                 ++up, dir = dir.parent_path()) {
                                std::ifstream gm(
                                    (dir / "import.rwmeta").string());
                                if (!gm) continue;
                                std::string gl;
                                bool gchar = false;
                                while (std::getline(gm, gl)) {
                                    if (gl == "type=character") {
                                        gchar = true;
                                        break;
                                    }
                                }
                                if (gchar) {
                                    p = dir.string();
                                    break;
                                }
                            }
                        }
                        if (fs::is_directory(p, ec)) {
                            // Group folder → CONTENT-ONLY: resolve to
                            // the baked <leaf>.rwchar manifest.  The
                            // original model (main=/source=) is import
                            // input only — never loaded for rendering.
                            std::ifstream meta(
                                (fs::path(p) / "import.rwmeta").string());
                            std::string line;
                            bool is_char = false;
                            while (meta && std::getline(meta, line)) {
                                if (line == "type=character") {
                                    is_char = true;
                                    break;
                                }
                            }
                            if (is_char) {
                                const std::string leaf =
                                    fs::path(p).filename().string();
                                const fs::path rwchar =
                                    fs::path(p) / (leaf + ".rwchar");
                                if (fs::exists(rwchar, ec)) {
                                    p = rwchar.string();
                                } else {
                                    EditorLog::get().push(
                                        "[player] '" + leaf +
                                        "' has no baked .rwchar — "
                                        "re-import the character "
                                        "(originals are import-only).");
                                    p.clear();
                                }
                            } else {
                                p.clear();
                            }
                        }
                        // Hard content-only gate: whatever path arrived
                        // (dragged file, resolved group), only a baked
                        // .rwchar may bind to the skeleton-mesh slot.
                        if (!p.empty() &&
                            fs::path(p).extension() != ".rwchar") {
                            EditorLog::get().push(
                                "[player] rejected '" + p +
                                "' — rendering uses baked content only; "
                                "import the character first.");
                            p.clear();
                        }
                        if (!p.empty()) {
                            player_mesh_assign_pending_ = true;
                            player_mesh_assign_idx_  = o.scene_index;
                            player_mesh_assign_path_ = p;
                        }
                    }
                    ImGui::EndDragDropTarget();
                }
                ImGui::TextDisabled(
                    "(pick above or drag a character onto the slot)");
            }

            // ── Collision object: held .rwcmap + assignment combo ────────
            // Mirrors the player's Skeleton Mesh slot: the combo lists the
            // baked maps under content/maps/collision; picking one swaps
            // the live collision world immediately (the app consumes
            // consumeCollisionMapAssign).  "None" clears the reference —
            // the scene loads with no collision until a map is assigned
            // or baked again.
            if (o.is_collision) {
                ImGui::SeparatorText("Collision");
                std::string cur = "None";
                if (o.collision_map && !o.collision_map->empty() &&
                    o.collision_map->size() > 7 &&
                    o.collision_map->compare(
                        o.collision_map->size() - 7, 7, ".rwcmap") == 0) {
                    cur = std::filesystem::path(*o.collision_map)
                              .filename().string();
                }
                if (ImGui::BeginCombo("Collision Map", cur.c_str())) {
                    if (ImGui::Selectable("None", cur == "None")) {
                        collision_map_assign_pending_ = true;
                        collision_map_assign_idx_  = o.scene_index;
                        collision_map_assign_path_.clear();
                    }
                    namespace fs = std::filesystem;
                    std::error_code ec;
                    if (fs::is_directory("content/maps/collision", ec)) {
                        for (auto& e : fs::directory_iterator(
                                 "content/maps/collision", ec)) {
                            if (e.path().extension() != ".rwcmap") continue;
                            const std::string label =
                                e.path().filename().string();
                            if (ImGui::Selectable(label.c_str(),
                                                  label == cur)) {
                                collision_map_assign_pending_ = true;
                                collision_map_assign_idx_  = o.scene_index;
                                collision_map_assign_path_ =
                                    e.path().generic_string();
                            }
                        }
                    }
                    ImGui::EndCombo();
                }
                if (o.collision_map && cur != "None") {
                    ImGui::TextDisabled("%s", o.collision_map->c_str());
                }
                ImGui::TextDisabled(
                    "(Tools > Bake Collision Map fills this in)");
            }

            // ── Camera: follow-target link ────────────────────────────
            // Points the camera at a player object.  Linked → looking
            // through this camera follows the player; empty → the camera
            // is a free view pose.
            if (o.is_camera && o.follow_link) {
                ImGui::SeparatorText("Camera");
                std::string cur_label = "None (free)";
                for (const auto& pe : editor_objects_) {
                    if (pe.is_player && pe.scene_index == *o.follow_link) {
                        cur_label = pe.name;
                        break;
                    }
                }
                if (ImGui::BeginCombo("Follow", cur_label.c_str())) {
                    if (ImGui::Selectable("None (free)",
                                          *o.follow_link < 0)) {
                        *o.follow_link = -1;
                    }
                    for (const auto& pe : editor_objects_) {
                        if (!pe.is_player || pe.scene_index < 0) continue;
                        const bool sel =
                            (pe.scene_index == *o.follow_link);
                        if (ImGui::Selectable(pe.name.c_str(), sel)) {
                            *o.follow_link = pe.scene_index;
                        }
                    }
                    ImGui::EndCombo();
                }
            }

            // ── BGM object: clip + repeat + volume ────────────────────
            // Drag a music clip from the Content Browser onto the slot;
            // the app loops/plays it on the music bus.  Repeat + volume
            // edit the scene object directly (synced live).
            if (o.is_bgm && o.audio_clip) {
                ImGui::SeparatorText("Background Music");
                const std::string cur =
                    o.audio_clip->empty()
                        ? std::string("(none — drag a clip here)")
                        : std::filesystem::path(*o.audio_clip)
                              .filename().string();
                ImGui::Button(cur.c_str(),
                              ImVec2(ImGui::GetContentRegionAvail().x, 0));
                if (ImGui::BeginDragDropTarget()) {
                    if (const ImGuiPayload* pl =
                            ImGui::AcceptDragDropPayload("RW_CONTENT_ASSET")) {
                        std::string p((const char*)pl->Data);
                        std::string ext =
                            std::filesystem::path(p).extension().string();
                        for (auto& c : ext)
                            c = (char)std::tolower((unsigned char)c);
                        if (ext == ".wav" || ext == ".mp3" || ext == ".flac")
                            *o.audio_clip = p;
                    }
                    ImGui::EndDragDropTarget();
                }
                if (!o.audio_clip->empty()) {
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Clear")) o.audio_clip->clear();
                }
                if (o.audio_loop)
                    ImGui::Checkbox("Repeat", o.audio_loop);
                if (o.audio_volume) {
                    ImGui::SetNextItemWidth(
                        ImGui::GetContentRegionAvail().x * 0.6f);
                    ImGui::SliderFloat("Volume", o.audio_volume,
                                       0.0f, 1.0f, "%.2f");
                }
            }

            // ── Point-light attributes (.rwlight rows) ────────────────
            // Live pointers into the scene object — edits apply next
            // frame via updateRestirLights() and persist with Save Scene.
            if (o.is_light && o.light_color) {
                ImGui::SeparatorText("Point Light");
                ImGui::ColorEdit3("Color",
                                  &o.light_color->x,
                                  ImGuiColorEditFlags_Float);
                if (o.light_intensity) {
                    ImGui::SetNextItemWidth(
                        ImGui::GetContentRegionAvail().x * 0.6f);
                    ImGui::DragFloat("Intensity", o.light_intensity,
                                     0.25f, 0.0f, 10000.0f, "%.2f");
                }
                if (o.light_radius) {
                    ImGui::SetNextItemWidth(
                        ImGui::GetContentRegionAvail().x * 0.6f);
                    ImGui::DragFloat("Radius (m)", o.light_radius,
                                     0.1f, 0.1f, 500.0f, "%.1f");
                }
                ImGui::TextDisabled(
                    "Shades via Rendering > Shadow > ReSTIR technique");
            }
        } else if (o.is_bgm && o.audio_clip) {
            // BGM object with no transform pane shown above — render the
            // clip/repeat/volume editor on its own.
            ImGui::SeparatorText("Background Music");
            const std::string cur =
                o.audio_clip->empty()
                    ? std::string("(none — drag a clip here)")
                    : std::filesystem::path(*o.audio_clip)
                          .filename().string();
            ImGui::Button(cur.c_str(),
                          ImVec2(ImGui::GetContentRegionAvail().x, 0));
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* pl =
                        ImGui::AcceptDragDropPayload("RW_CONTENT_ASSET")) {
                    std::string p((const char*)pl->Data);
                    std::string ext =
                        std::filesystem::path(p).extension().string();
                    for (auto& c : ext)
                        c = (char)std::tolower((unsigned char)c);
                    if (ext == ".wav" || ext == ".mp3" || ext == ".flac")
                        *o.audio_clip = p;
                }
                ImGui::EndDragDropTarget();
            }
            if (!o.audio_clip->empty()) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear")) o.audio_clip->clear();
            }
            if (o.audio_loop) ImGui::Checkbox("Repeat", o.audio_loop);
            if (o.audio_volume) {
                ImGui::SetNextItemWidth(
                    ImGui::GetContentRegionAvail().x * 0.6f);
                ImGui::SliderFloat("Volume", o.audio_volume,
                                   0.0f, 1.0f, "%.2f");
            }
        } else {
            ImGui::TextDisabled("(no drawable bound)");
        }
    } else {
        ImGui::TextDisabled("Select an object above.");
    }
}

void Menu::drawOutputPanel() {
    if (ImGui::Begin("Output Log")) {
        static std::vector<std::string> s_lines;
        static std::uint64_t s_version = (std::uint64_t)-1;
        static bool s_autoscroll = true;
        static char s_filter[64] = {0};

        std::uint64_t cur_version = s_version;
        EditorLog::get().snapshot(s_lines, cur_version);
        const bool changed = (cur_version != s_version);
        s_version = cur_version;

        // Toolbar: clear / copy / autoscroll / filter.
        if (ImGui::SmallButton("Clear")) { EditorLog::get().clear(); s_lines.clear(); }
        ImGui::SameLine();
        if (ImGui::SmallButton("Copy")) {
            // Copy all (filtered) lines to the clipboard.
            std::string all;
            const bool hf = (s_filter[0] != '\0');
            for (const std::string& ln : s_lines) {
                if (hf && ln.find(s_filter) == std::string::npos) continue;
                all += ln; all += '\n';
            }
            ImGui::SetClipboardText(all.c_str());
        }
        ImGui::SameLine();
        ImGui::Checkbox("Auto-scroll", &s_autoscroll);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(180.0f);
        ImGui::InputTextWithHint("##log_filter", "filter", s_filter, sizeof(s_filter));
        ImGui::SameLine();
        ImGui::TextDisabled("%d lines", (int)s_lines.size());
        ImGui::Separator();

        ImGui::BeginChild("##log_scroll", ImVec2(0, 0), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 1));
        const bool has_filter = (s_filter[0] != '\0');
        int li = 0;
        for (const std::string& ln : s_lines) {
            ++li;
            if (has_filter && ln.find(s_filter) == std::string::npos) continue;
            // Subtle colour-coding by tag for readability.
            ImVec4 col(0.86f, 0.86f, 0.86f, 1.0f);
            if (ln.find("error") != std::string::npos ||
                ln.find("ERROR") != std::string::npos ||
                ln.find("FAIL")  != std::string::npos)
                col = ImVec4(1.0f, 0.45f, 0.40f, 1.0f);
            else if (ln.find("warn") != std::string::npos ||
                     ln.find("WARN") != std::string::npos)
                col = ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
            else if (!ln.empty() && ln[0] == '[')
                col = ImVec4(0.55f, 0.78f, 1.0f, 1.0f);   // [tag] lines
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            // Selectable line: click copies just this line to the clipboard.
            // "###" keeps the visible label = the full line, ID = the index.
            const std::string lbl = ln + "###logln" + std::to_string(li);
            if (ImGui::Selectable(lbl.c_str()))
                ImGui::SetClipboardText(ln.c_str());
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Click to copy this line");
        }
        ImGui::PopStyleVar();
        if (s_autoscroll && changed)
            ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();
    }
    ImGui::End();
}

// ── DDS decode helpers (top mip → RGBA8) ────────────────────────────────────
namespace {

constexpr uint32_t kFourCC(char a, char b, char c, char d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
           ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}
inline uint32_t rd32le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline void rgb565(uint16_t c, int& r, int& g, int& b) {
    int r5 = (c >> 11) & 0x1F, g6 = (c >> 5) & 0x3F, b5 = c & 0x1F;
    r = (r5 << 3) | (r5 >> 2); g = (g6 << 2) | (g6 >> 4); b = (b5 << 3) | (b5 >> 2);
}
// BC1 colour block (8 bytes) → 16 RGBA texels, row-major.
inline void decodeBC1(const uint8_t* b, uint8_t out[64], bool punchthrough) {
    uint16_t c0 = (uint16_t)(b[0] | (b[1] << 8)), c1 = (uint16_t)(b[2] | (b[3] << 8));
    int r[4], g[4], bl[4], a[4] = {255, 255, 255, 255};
    rgb565(c0, r[0], g[0], bl[0]); rgb565(c1, r[1], g[1], bl[1]);
    if (c0 > c1 || !punchthrough) {
        r[2] = (2*r[0]+r[1])/3; g[2] = (2*g[0]+g[1])/3; bl[2] = (2*bl[0]+bl[1])/3;
        r[3] = (r[0]+2*r[1])/3; g[3] = (g[0]+2*g[1])/3; bl[3] = (bl[0]+2*bl[1])/3;
    } else {
        r[2] = (r[0]+r[1])/2; g[2] = (g[0]+g[1])/2; bl[2] = (bl[0]+bl[1])/2;
        r[3] = g[3] = bl[3] = 0; a[3] = 0;
    }
    uint32_t bits = rd32le(b + 4);
    for (int i = 0; i < 16; ++i) {
        int idx = (bits >> (i * 2)) & 3;
        out[i*4+0] = (uint8_t)r[idx]; out[i*4+1] = (uint8_t)g[idx];
        out[i*4+2] = (uint8_t)bl[idx]; out[i*4+3] = (uint8_t)a[idx];
    }
}
// BC3/DXT5 alpha block (8 bytes) → 16 alpha values.
inline void decodeBC3Alpha(const uint8_t* b, uint8_t alpha[16]) {
    int a0 = b[0], a1 = b[1], al[8]; al[0] = a0; al[1] = a1;
    if (a0 > a1) { for (int i = 1; i < 7; ++i) al[i+1] = ((7-i)*a0 + i*a1)/7; }
    else { for (int i = 1; i < 5; ++i) al[i+1] = ((5-i)*a0 + i*a1)/5; al[6] = 0; al[7] = 255; }
    uint64_t bits = 0; for (int i = 0; i < 6; ++i) bits |= (uint64_t)b[2+i] << (8*i);
    for (int i = 0; i < 16; ++i) alpha[i] = (uint8_t)al[(bits >> (i*3)) & 7];
}

// Decode the top mip of a .dds to RGBA8.  Handles DXT1/3/5 and 32bpp
// uncompressed (via channel masks).  Returns false for DX10/BC7 / exotic
// formats — the caller then falls back to a type tile.
bool decodeDdsToRgba(const std::string& path, int& W, int& H,
                     std::vector<unsigned char>& rgba) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (d.size() < 128 || rd32le(d.data()) != 0x20534444u) return false;  // "DDS "
    const uint8_t* hdr = d.data() + 4;
    const uint32_t height = rd32le(hdr + 8), width = rd32le(hdr + 12);
    if (width == 0 || height == 0 || width > 16384 || height > 16384) return false;
    const uint8_t* pf = hdr + 72;                       // DDS_PIXELFORMAT
    const uint32_t pfFlags = rd32le(pf + 4), fourCC = rd32le(pf + 8),
                   rgbBits = rd32le(pf + 12);
    const uint32_t rMask = rd32le(pf + 16), gMask = rd32le(pf + 20),
                   bMask = rd32le(pf + 24), aMask = rd32le(pf + 28);
    const bool isFourCC = (pfFlags & 0x4) != 0;
    if (isFourCC && fourCC == kFourCC('D','X','1','0')) return false;  // DX10/BC7

    W = (int)width; H = (int)height;
    rgba.assign((size_t)W * H * 4, 255);
    const uint8_t* src = d.data() + 128;
    const size_t avail = d.size() - 128;
    const int bw = (W + 3) / 4, bh = (H + 3) / 4;

    auto putBlock = [&](int bx, int by, const uint8_t px[64]) {
        for (int py = 0; py < 4; ++py) for (int pxx = 0; pxx < 4; ++pxx) {
            int X = bx*4 + pxx, Y = by*4 + py; if (X >= W || Y >= H) continue;
            const uint8_t* s = px + (py*4 + pxx)*4;
            uint8_t* o = &rgba[((size_t)Y*W + X)*4];
            o[0]=s[0]; o[1]=s[1]; o[2]=s[2]; o[3]=s[3];
        }
    };

    if (isFourCC && fourCC == kFourCC('D','X','T','1')) {
        if (avail < (size_t)bw*bh*8) return false;
        for (int by = 0; by < bh; ++by) for (int bx = 0; bx < bw; ++bx) {
            uint8_t px[64]; decodeBC1(src + ((size_t)by*bw + bx)*8, px, true);
            putBlock(bx, by, px);
        }
        return true;
    }
    if (isFourCC && (fourCC == kFourCC('D','X','T','5') ||
                     fourCC == kFourCC('D','X','T','3'))) {
        const bool dxt5 = (fourCC == kFourCC('D','X','T','5'));
        if (avail < (size_t)bw*bh*16) return false;
        for (int by = 0; by < bh; ++by) for (int bx = 0; bx < bw; ++bx) {
            const uint8_t* blk = src + ((size_t)by*bw + bx)*16;
            uint8_t px[64]; decodeBC1(blk + 8, px, false);
            if (dxt5) { uint8_t al[16]; decodeBC3Alpha(blk, al);
                        for (int i = 0; i < 16; ++i) px[i*4+3] = al[i]; }
            else { for (int i = 0; i < 16; ++i) { int byte = blk[i/2];
                        int a4 = (i&1) ? ((byte>>4)&0xF) : (byte&0xF);
                        px[i*4+3] = (uint8_t)(a4*17); } }
            putBlock(bx, by, px);
        }
        return true;
    }
    if (!isFourCC && rgbBits == 32) {
        if (avail < (size_t)W*H*4) return false;
        auto shift = [](uint32_t m){ int s=0; if(!m) return 0; while(!(m&1)){m>>=1;++s;} return s; };
        const int sr = shift(rMask), sg = shift(gMask), sb = shift(bMask), sa = shift(aMask);
        for (int i = 0; i < W*H; ++i) {
            uint32_t p = rd32le(src + (size_t)i*4);
            rgba[i*4+0] = (uint8_t)((p & rMask) >> sr);
            rgba[i*4+1] = (uint8_t)((p & gMask) >> sg);
            rgba[i*4+2] = (uint8_t)((p & bMask) >> sb);
            rgba[i*4+3] = aMask ? (uint8_t)((p & aMask) >> sa) : 255;
        }
        return true;
    }
    return false;
}

}  // namespace

// Produce/refresh the sidecar thumbnail for `src` and return its path (or "").
// Thumbnails live in a hidden ".thumbnails" folder next to the asset, named
// "<filename>.png".  Regeneration is incremental: an existing thumbnail at
// least as new as the source is kept untouched; only a stale/missing one is
// rebuilt.  Dispatches by type: images (stb) and .dds (decoded) are
// box-downscaled; .gltf/.glb/.fbx are rasterised to an orbit view.
std::string Menu::ensureThumbnail(const std::string& src) {
    namespace fs = std::filesystem;
    namespace ar = plugins::auto_rig;
    std::error_code ec;
    constexpr int kThumb = 128;

    fs::path srcP(src);
    std::string ext = srcP.extension().string();
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    const bool is_img   = (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
                           ext == ".bmp" || ext == ".tga");
    const bool is_dds   = (ext == ".dds");
    const bool is_model = (ext == ".gltf" || ext == ".glb" || ext == ".fbx");
    if (!is_img && !is_dds && !is_model) return "";

    const fs::path thumbDir = srcP.parent_path() / ".thumbnails";
    const fs::path thumb     = thumbDir / (srcP.filename().string() + ".png");

    // Incremental: keep the existing thumbnail unless the source is newer.
    if (fs::exists(thumb, ec)) {
        const auto t_src = fs::last_write_time(src, ec);
        const auto t_thb = fs::last_write_time(thumb, ec);
        if (!ec && t_src <= t_thb) return thumb.string();
    }
    fs::create_directories(thumbDir, ec);

    // ── 3D model → rasterise one auto-framed orbit view ──
    if (is_model) {
        ar::TriangleMesh mesh;
        if (!ar::loadMeshForThumbnail(src, mesh)) return "";
        ar::SimpleRasterizer rast;
        auto caps = rast.captureOrbit(mesh, /*views*/1, kThumb,
                                      /*elevation*/15.0f, /*radius_mult*/1.5f);
        if (caps.empty() || caps[0].color.empty()) return "";
        const auto& c = caps[0];
        if (!stbi_write_png(thumb.string().c_str(), c.width, c.height, 3,
                            c.color.data(), c.width * 3))
            return "";
        return thumb.string();
    }

    // ── Image / DDS → RGBA pixels, then box-downscale to <=kThumb ──
    int w = 0, h = 0;
    std::vector<unsigned char> rgba;
    if (is_dds) {
        if (!decodeDdsToRgba(src, w, h, rgba)) return "";
    } else {
        int comp = 0;
        unsigned char* px = stbi_load(src.c_str(), &w, &h, &comp, STBI_rgb_alpha);
        if (!px || w <= 0 || h <= 0) { if (px) stbi_image_free(px); return ""; }
        rgba.assign(px, px + (size_t)w * h * 4);
        stbi_image_free(px);
    }

    int dw = w, dh = h;
    if (w > kThumb || h > kThumb) {
        const float s = (float)kThumb / (float)std::max(w, h);
        dw = std::max(1, (int)(w * s));
        dh = std::max(1, (int)(h * s));
    }
    std::vector<unsigned char> out((size_t)dw * dh * 4);
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const int sx = std::min(w - 1, (int)(((float)x + 0.5f) * w / dw));
            const int sy = std::min(h - 1, (int)(((float)y + 0.5f) * h / dh));
            const unsigned char* sp = rgba.data() + ((size_t)sy * w + sx) * 4;
            unsigned char* dp = out.data() + ((size_t)y * dw + x) * 4;
            dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2]; dp[3] = sp[3];
        }
    }
    if (!stbi_write_png(thumb.string().c_str(), dw, dh, 4, out.data(), dw * 4))
        return "";
    return thumb.string();
}

// ── Full-size image viewer ──────────────────────────────────────────────────
void Menu::openImageViewer(const std::string& path) {
    // Retire any previous full-size texture (same GPU-safety pattern as
    // stale thumbnails: keep the shared_ptr alive until a safe drain).
    if (image_viewer_info_) retired_thumbs_.push_back(image_viewer_info_);
    image_viewer_info_.reset();
    image_viewer_id_   = ImTextureID(0);
    image_viewer_path_ = path;
    image_viewer_open_ = true;
}

void Menu::drawImageViewer() {
    if (!image_viewer_open_) {
        // Window closed — release the texture (retired, GPU-safe).
        if (image_viewer_info_) {
            retired_thumbs_.push_back(image_viewer_info_);
            image_viewer_info_.reset();
            image_viewer_id_ = ImTextureID(0);
            image_viewer_path_.clear();
        }
        return;
    }

    // Lazy full-res load.  Same guard as getThumbnail: never do main-thread
    // GPU uploads while the async mesh loader is busy (queue/pool hazard);
    // just retry on a later frame — the window shows "Loading..." meanwhile.
    if (!image_viewer_id_ && !image_viewer_path_.empty() &&
        !(mesh_load_task_manager_ &&
          mesh_load_task_manager_->inFlightCount() > 0)) {
        try {
            image_viewer_info_ = std::make_shared<renderer::TextureInfo>();
            engine::helper::createTextureImage(
                device_, image_viewer_path_,
                renderer::Format::R8G8B8A8_UNORM, true,
                *image_viewer_info_, std::source_location::current());
            if (image_viewer_info_->view) {
                image_viewer_id_ = renderer::Helper::addImTextureID(
                    sampler_, image_viewer_info_->view);
            }
        } catch (...) {
            image_viewer_info_.reset();
            EditorLog::get().push("[viewer] failed to load image: " +
                                  image_viewer_path_);
            image_viewer_open_ = false;
            return;
        }
    }

    const std::string title =
        std::filesystem::path(image_viewer_path_).filename().string() +
        "###image_viewer";
    // Fit the window to the image, capped at ~85% of the main viewport;
    // the image scales down uniformly to fit (never upscaled past 1:1).
    const ImVec2 vp = ImGui::GetMainViewport()->WorkSize;
    ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowPos(
        ImVec2(ImGui::GetMainViewport()->WorkPos.x + vp.x * 0.5f,
               ImGui::GetMainViewport()->WorkPos.y + vp.y * 0.5f),
        ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::Begin(title.c_str(), &image_viewer_open_,
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoDocking)) {
        if (image_viewer_id_ && image_viewer_info_) {
            const float iw = float(image_viewer_info_->size.x);
            const float ih = float(image_viewer_info_->size.y);
            const float max_w = vp.x * 0.85f;
            const float max_h = vp.y * 0.85f;
            float scale = 1.0f;
            if (iw > max_w) scale = max_w / iw;
            if (ih * scale > max_h) scale = max_h / ih;
            ImGui::Image(image_viewer_id_,
                         ImVec2(iw * scale, ih * scale));
            ImGui::TextDisabled("%.0f x %.0f%s", iw, ih,
                                scale < 1.0f ? "  (fit to screen)" : "");
        } else {
            ImGui::TextDisabled("Loading...");
        }
    }
    ImGui::End();
}

ImTextureID Menu::getThumbnail(const std::string& src) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto src_time = fs::last_write_time(src, ec);

    auto it = thumb_cache_.find(src);
    if (it != thumb_cache_.end()) {
        // Reuse the cached texture unless the source changed since we built it.
        if (it->second.failed) return 0;
        if (ec || src_time == it->second.src_time) return it->second.id;
        // Source is newer — retire the stale GPU texture (kept alive so ImGui
        // never touches a freed descriptor) and rebuild below.
        if (it->second.info) retired_thumbs_.push_back(it->second.info);
        thumb_cache_.erase(it);
    }

    if (thumb_budget_ <= 0) return 0;   // budget spent — retry next frame

    // CRITICAL: do NOT create thumbnails while the async mesh loader is busy.
    // Thumbnail generation does GPU work on the MAIN thread — ensureThumbnail()
    // loads/rasterises a model (tinygltf/ufbx) and createTextureImage() records
    // a command buffer + submits to a queue + allocates device memory.  The
    // mesh-load WORKER thread does the same kinds of GPU work concurrently, and
    // Vulkan requires external synchronisation for queues / command pools / the
    // allocator.  Running both at once corrupts driver state and faults later
    // inside vkAllocateDescriptorSets (the createDescriptorSets crash).  Defer
    // all thumbnail creation until the loader has drained; cached thumbnails
    // still display (the cache-hit path above returns before this point).
    if (mesh_load_task_manager_ &&
        mesh_load_task_manager_->inFlightCount() > 0) {
        return 0;   // loader active — retry on a later, idle frame
    }

    // Hard cap on cached thumbnails (bounds ImGui's own descriptor pool).
    constexpr size_t kMaxThumbs = 192;
    if (thumb_cache_.size() >= kMaxThumbs) return 0;

    --thumb_budget_;

    ThumbTex t;
    t.src_time = src_time;
    const std::string thumb = ensureThumbnail(src);   // incremental: only if stale
    if (thumb.empty()) {
        t.failed = true;
    } else {
        try {
            t.info = std::make_shared<renderer::TextureInfo>();
            engine::helper::createTextureImage(
                device_, thumb, renderer::Format::R8G8B8A8_UNORM, true,
                *t.info, std::source_location::current());
            if (t.info && t.info->view)
                t.id = renderer::Helper::addImTextureID(sampler_, t.info->view);
            else
                t.failed = true;
        } catch (...) {
            t.failed = true;
            t.info.reset();
        }
    }
    const ImTextureID id = t.failed ? 0 : t.id;
    thumb_cache_.emplace(src, std::move(t));
    return id;
}

void Menu::drawFolderTree(const std::string& dir, int depth,
                          std::string& cur_dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return;

    std::vector<fs::directory_entry> subs;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_directory()) continue;
        // Hide dot-folders (.git, .thumbnails, .flux_tmp, …) — internal
        // bookkeeping, never browsable assets.
        const std::string fn = e.path().filename().string();
        if (!fn.empty() && fn[0] == '.') continue;
        subs.push_back(e);
    }
    std::sort(subs.begin(), subs.end(), [](const auto& a, const auto& b) {
        return a.path().filename().string() < b.path().filename().string(); });

    std::string name = fs::path(dir).filename().string();
    if (name.empty()) name = dir;

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                               ImGuiTreeNodeFlags_SpanAvailWidth;
    if (cur_dir == dir) flags |= ImGuiTreeNodeFlags_Selected;
    if (depth == 0)          flags |= ImGuiTreeNodeFlags_DefaultOpen;
    if (subs.empty())
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

    const bool open = ImGui::TreeNodeEx(name.c_str(), flags);
    // Click on the label (not the open/close arrow) selects the folder.
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        cur_dir = dir;
    if (open && !subs.empty()) {
        for (auto& s : subs)
            drawFolderTree(s.path().string(), depth + 1, cur_dir);
        ImGui::TreePop();
    }
}

// Launch the FLUX.2 generator for one image (detached subprocess).
// ImGui's InputTextMultiline has no native word-wrap: a long line just scrolls
// horizontally and gets clipped.  This callback soft-wraps the buffer to the box
// width by converting spaces to newlines (and back), so wrapping is purely
// visual and reversible.  Cursor stays aligned because we only swap chars in
// place (' ' <-> '\n'), never insert or delete, so the buffer length is stable.
namespace {
struct GenWrapUD { float wrap_w; };

int GenPromptWrapCallback(ImGuiInputTextCallbackData* data) {
    GenWrapUD* ud = static_cast<GenWrapUD*>(data->UserData);
    const float wrap_w = ud ? ud->wrap_w : 400.0f;
    if (wrap_w <= 1.0f || data->BufTextLen <= 0) return 0;

    // Flatten existing soft-wraps back to spaces; record the cursor as a count
    // of kept characters so we can restore it after re-wrapping.
    std::string raw;
    raw.reserve(data->BufTextLen);
    int raw_cursor = 0;
    for (int i = 0; i < data->BufTextLen; ++i) {
        char c = data->Buf[i];
        if (c == '\r') continue;
        if (c == '\n') c = ' ';
        if (i < data->CursorPos) ++raw_cursor;
        raw.push_back(c);
    }

    // Greedy pixel word-wrap: convert the last space on an over-long line to a
    // newline.  Length is preserved, so cursor index maps 1:1.
    std::string wrapped = raw;
    int line_start = 0;
    int last_space = -1;
    for (int i = 0; i < (int)wrapped.size(); ++i) {
        if (wrapped[i] == ' ') last_space = i;
        const char* ls = wrapped.c_str() + line_start;
        const char* le = wrapped.c_str() + (i + 1);
        if (ImGui::CalcTextSize(ls, le).x > wrap_w && last_space > line_start) {
            wrapped[last_space] = '\n';
            line_start = last_space + 1;
            last_space = -1;
        }
    }

    // Only rewrite if something actually changed (avoids fighting the cursor
    // every frame under CallbackAlways).
    bool same = ((int)wrapped.size() == data->BufTextLen);
    for (int i = 0; same && i < (int)wrapped.size(); ++i)
        if (wrapped[i] != data->Buf[i]) same = false;
    if (same) return 0;

    if ((int)wrapped.size() < data->BufSize) {
        for (int i = 0; i < (int)wrapped.size(); ++i) data->Buf[i] = wrapped[i];
        data->Buf[wrapped.size()] = '\0';
        data->BufTextLen = (int)wrapped.size();
        data->BufDirty = true;
        data->CursorPos = data->SelectionStart = data->SelectionEnd = raw_cursor;
    }
    return 0;
}
} // namespace

void Menu::launchImageGen(const std::string& folder, const std::string& prompt,
                          int width, int height) {
    namespace fs = std::filesystem;
    std::string p = prompt;
    // The prompt box soft-wraps by inserting newlines purely for display; the
    // model wants a clean single line, so collapse interior breaks to spaces.
    for (char& c : p) if (c == '\n' || c == '\r') c = ' ';
    while (!p.empty() && (p.back() == ' ' || p.back() == '\t'))
        p.pop_back();
    if (p.empty()) { gen_status_ = 3; gen_err_ = "empty prompt"; return; }

    std::error_code ec;
    fs::create_directories(folder, ec);

    static int s_n = 0;
    // Optional name from the popup (gen_name_): sanitize + drop a typed
    // ".png"; empty -> a unique auto name (a reused name overwrites).
    std::string base;
    {
        std::string clean;
        for (char ch : std::string(gen_name_)) {
            if (ch=='/'||ch=='\\'||ch==':'||ch=='*'||ch=='?'||ch=='"'||
                ch=='<'||ch=='>'||ch=='|'||ch=='\n'||ch=='\r'||ch=='\t')
                continue;
            clean.push_back(ch);
        }
        while (!clean.empty() && clean.front()==' ') clean.erase(clean.begin());
        while (!clean.empty() && clean.back()==' ')  clean.pop_back();
        if (clean.size() >= 4) {
            std::string e = clean.substr(clean.size()-4);
            for (char& ec : e) if (ec>='A'&&ec<='Z') ec=(char)(ec+32);
            if (e == ".png") clean.resize(clean.size()-4);
        }
        base = clean.empty()
             ? ("flux_" + std::to_string((long long)std::time(nullptr)) +
                "_" + std::to_string(s_n++))
             : clean;
    }
    gen_out_path_ = (fs::path(folder) / (base + ".png")).string();
    fs::path flux_tmp = fs::path(folder) / ".flux_tmp";
    fs::create_directories(flux_tmp, ec);
    const std::string pfile =
        (flux_tmp / (base + ".png.prompt.txt")).string();
    { std::ofstream pf(pfile, std::ios::binary); pf << p; }

    gen_last_prompt_ = p; gen_last_w_ = width; gen_last_h_ = height;
    gen_err_.clear();

    // Reference images: FLUX.2 klein's native image conditioning — the
    // prompt steers edits / style / structure relative to them (ControlNet-
    // style guidance with the base weights, no extra checkpoints).
    std::string ref_args;
    for (const auto& rp : gen_ref_images_) {
        ref_args += " --ref-image \"" + rp + "\"";
    }

    // Use the SAME interpreter Setup.bat configured (system "python" with the
    // CUDA torch it installed) — not a separate venv.
#ifdef _WIN32
    const std::string script = "tools\\flux\\flux_generate.py";
    const std::string cmd =
        "cmd /c start \"flux\" /B python \"" + script + "\""
        " --prompt-file \"" + pfile + "\" --out \"" + gen_out_path_ + "\""
        " --width "  + std::to_string(width) +
        " --height " + std::to_string(height) + ref_args;
#else
    const std::string script = "tools/flux/flux_generate.py";
    const std::string cmd =
        "python3 \"" + script + "\""
        " --prompt-file \"" + pfile + "\" --out \"" + gen_out_path_ + "\""
        " --width "  + std::to_string(width) +
        " --height " + std::to_string(height) + ref_args + " &";
#endif
    EditorLog::get().push("[flux] launching generation -> " + gen_out_path_);
    std::system(cmd.c_str());
    gen_status_ = 1;   // running
}

// ── AI terrain generation popup (Tools > Generate Terrain) ──────────────────
// Text → FLUX satellite-view heightmap → torch erosion conversion → 16-bit
// terrain PNG, optionally installed as assets/map.png.  Detached python
// process (same launch/poll contract as the FLUX image popup: output PNG =
// success, "<out>.err" = failure).
void Menu::drawTerrainGenPopup() {
    namespace fs = std::filesystem;

    // Poll a running generation.
    if (terrain_gen_status_ == 1) {
        std::error_code ec;
        if (fs::exists(terrain_gen_out_, ec)) {
            terrain_gen_status_ = 2;
            EditorLog::get().push(
                "[terrain] heightmap ready: " + terrain_gen_out_ +
                (terrain_gen_install_
                     ? "  (installed as assets/map.png — restart the "
                       "engine to apply)"
                     : ""));
        } else if (fs::exists(terrain_gen_out_ + ".err", ec)) {
            terrain_gen_status_ = 3;
            std::ifstream ef(terrain_gen_out_ + ".err", std::ios::binary);
            terrain_gen_err_.assign(std::istreambuf_iterator<char>(ef),
                                    std::istreambuf_iterator<char>());
            EditorLog::get().push("[terrain] generation FAILED: " +
                                  terrain_gen_err_);
        }
    }

    if (!terrain_gen_popup_open_) return;
    ImGui::SetNextWindowSize(ImVec2(480, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Generate Terrain (AI)", &terrain_gen_popup_open_,
                     ImGuiWindowFlags_NoDocking |
                     ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(
            "Describe the terrain.  FLUX renders a satellite-view "
            "heightmap; a GPU erosion model converts it into a terrain "
            "heightfield (2048x2048, 16-bit).");
        // Soft-wrap to the box width (same callback the image-gen prompt
        // uses) so long prompts wrap instead of scrolling off the right edge.
        const float tp_box_w = ImGui::GetContentRegionAvail().x;
        GenWrapUD tp_ud;
        tp_ud.wrap_w = tp_box_w - ImGui::GetStyle().FramePadding.x * 2.0f
                                - ImGui::GetStyle().ScrollbarSize - 4.0f;
        ImGui::InputTextMultiline("##terrain_prompt", terrain_prompt_buf_,
                                  sizeof(terrain_prompt_buf_),
                                  ImVec2(tp_box_w, 64.0f),
                                  ImGuiInputTextFlags_CallbackAlways,
                                  &GenPromptWrapCallback, &tp_ud);
        ImGui::Checkbox("Install as active terrain (assets/map.png)",
                        &terrain_gen_install_);
        // Second FLUX pass conditioned on the finished heightfield —
        // produces <name>_color.png with matching layout (rock/snow on
        // peaks, vegetation in valleys, water in channels).
        ImGui::Checkbox("Also generate color satellite map",
                        &terrain_gen_color_);

        const bool busy = (terrain_gen_status_ == 1);
        if (busy) ImGui::BeginDisabled();
        if (ImGui::Button("Generate", ImVec2(120, 0))) {
            const std::string p(terrain_prompt_buf_);
            if (!p.empty()) {
                std::error_code ec;
                fs::create_directories("content/terrain/.terrain_tmp", ec);
                const std::string base =
                    "terrain_" +
                    std::to_string((long long)std::time(nullptr));
                terrain_gen_out_ =
                    (fs::path("content/terrain") / (base + ".png")).string();
                const std::string pfile =
                    (fs::path("content/terrain/.terrain_tmp") /
                     (base + ".prompt.txt")).string();
                { std::ofstream pf(pfile, std::ios::binary); pf << p; }
                terrain_gen_err_.clear();
#ifdef _WIN32
                const std::string cmd =
                    "cmd /c start \"terrain\" /B python "
                    "\"tools\\terrain\\terrain_from_text.py\""
                    " --prompt-file \"" + pfile + "\""
                    " --out \"" + terrain_gen_out_ + "\"" +
                    (terrain_gen_install_ ? " --install" : "") +
                    (terrain_gen_color_   ? " --color"   : "");
#else
                const std::string cmd =
                    "python3 \"tools/terrain/terrain_from_text.py\""
                    " --prompt-file \"" + pfile + "\""
                    " --out \"" + terrain_gen_out_ + "\"" +
                    (terrain_gen_install_ ? " --install" : "") +
                    (terrain_gen_color_   ? " --color"   : "") + " &";
#endif
                EditorLog::get().push(
                    "[terrain] launching generation -> " + terrain_gen_out_);
                std::system(cmd.c_str());
                terrain_gen_status_ = 1;
            }
        }
        if (busy) ImGui::EndDisabled();

        if (terrain_gen_status_ != 1) ImGui::SameLine();
        if (terrain_gen_status_ == 1) {
            // Progress: the script overwrites "<out>.progress" with
            // "<frac> <stage label>" (diffusion steps included — it parses
            // FLUX's tqdm output).  Poll it here; missing file = starting.
            // Full-width bar on its OWN row under the (disabled) button.
            float       frac  = 0.01f;
            std::string label = "starting python...";
            {
                std::ifstream pf(terrain_gen_out_ + ".progress",
                                 std::ios::binary);
                if (pf) {
                    pf >> frac;
                    std::getline(pf, label);
                    if (!label.empty() && label[0] == ' ')
                        label.erase(0, 1);
                }
            }
            char overlay[192];
            std::snprintf(overlay, sizeof(overlay), "%s  %.0f%%",
                          label.c_str(), frac * 100.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                  ImVec4(1.0f, 0.78f, 0.31f, 1.0f));
            ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f), overlay);
            ImGui::PopStyleColor();
        } else if (terrain_gen_status_ == 2) {
            ImGui::TextColored(ImVec4(0.45f, 0.9f, 0.5f, 1.0f), "Done.");
            if (terrain_gen_install_) {
                ImGui::SameLine();
                // Hot reload: the app re-reads assets/map.png and re-runs
                // the tile creator compute — new terrain, no restart.
                if (ImGui::Button("Rebuild Terrain Now")) {
                    terrain_apply_request_ = true;
                }
            }
        } else if (terrain_gen_status_ == 3) {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f),
                               "Failed — see Editor Log.");
        }
        // The generated PNG lands in content/terrain/, so it also shows
        // up in the Content Browser for preview / reuse as a ref image.
    }
    ImGui::End();
}

// ── Shared directory-browser body ───────────────────────────────────────────
// Folder tree (left) + thumbnail grid (right).  Drives BOTH bottom panels:
//   * File Browser    — raw disk view rooted at the project folder; lets you
//                       inspect what's on disk (is_content == false).
//   * Content Browser — the game project's asset view rooted at content/;
//                       Import… copies external files in (with a .rwmeta
//                       sidecar) and tiles gain "Add to Scene"
//                       (is_content == true).
void Menu::drawBrowserBody(const std::string& tree_root,
                           std::string& cur_dir,
                           float& left_w,
                           bool is_content) {
    namespace fs = std::filesystem;
    {
        thumb_budget_ = 3;   // cap texture loads per frame so big folders don't stall

        // ── Streaming-import progress (Content Browser only) ───────────
        // Shown while the app's background thread copies the chosen file
        // into content/; the new tile appears in the grid when it's done.
        if (is_content && content_import_active_) {
            char overlay[160];
            std::snprintf(overlay, sizeof(overlay), "Importing %s  %.0f%%",
                          content_import_label_.c_str(),
                          content_import_frac_ * 100.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                  ImVec4(1.0f, 0.78f, 0.31f, 1.0f));
            ImGui::ProgressBar(content_import_frac_, ImVec2(-1.0f, 0.0f),
                               overlay);
            ImGui::PopStyleColor();
        }

        const float total_w = ImGui::GetContentRegionAvail().x;
        constexpr float kMinPane = 90.0f, kSplit = 6.0f;
        if (left_w <= 0.0f) left_w = total_w * 0.22f;
        if (left_w < kMinPane) left_w = kMinPane;
        if (total_w - kMinPane - kSplit > kMinPane &&
            left_w > total_w - kMinPane - kSplit)
            left_w = total_w - kMinPane - kSplit;

        // ── Left: folder tree (names only) ────────────────────────────
        ImGui::BeginChild("##cb_tree", ImVec2(left_w, 0), true);
        drawFolderTree(tree_root, 0, cur_dir);
        ImGui::EndChild();

        // Vertical splitter between the two panes.
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::InvisibleButton("##cb_split", ImVec2(kSplit, -1.0f));
        if (ImGui::IsItemActive() || ImGui::IsItemHovered())
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        if (ImGui::IsItemActive())
            left_w += ImGui::GetIO().MouseDelta.x;
        {
            const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
            const float cx = (a.x + b.x) * 0.5f;
            ImGui::GetWindowDrawList()->AddLine(
                ImVec2(cx, a.y + 2), ImVec2(cx, b.y - 2),
                ImGui::GetColorU32(ImGui::IsItemActive() ? ImGuiCol_SeparatorActive
                                                         : ImGuiCol_Separator), 1.5f);
        }
        ImGui::SameLine(0.0f, 0.0f);

        // ── Right: thumbnail grid of the selected folder ──────────────
        // NoNav window flag: arrow keys must MOVE THE SELECTION, never
        // scroll-shift the panel (ImGui's default nav response).
        ImGui::BeginChild("##cb_grid", ImVec2(0, 0), true,
                          ImGuiWindowFlags_NoNav);
        // Keep ImGui's keyboard-nav focus OFF the tiles: its blue focus
        // ring (set by clicking a button) would linger on the last clicked
        // tile while the amber preview-selection border moves with the
        // arrow keys — two competing highlights.  With NoNav the amber
        // border is the single selection indicator for mouse AND arrows.
        ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);

        // Arrow keys work from THIS panel too (not just the Debug Display):
        // after clicking a tile the browser keeps focus, and arrows should
        // still step the selection.
        if (is_content &&
            ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
            preview_nav_ != PreviewNav::None) {
            handlePreviewArrowNav();
        }

        if (ImGui::ArrowButton("##cb_up", ImGuiDir_Up)) {
            fs::path p(cur_dir);
            // Never navigate above the browser's root.
            if (cur_dir != tree_root &&
                p.has_parent_path() && !p.parent_path().empty())
                cur_dir = p.parent_path().string();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Up one level");
        ImGui::SameLine();
        ImGui::TextUnformatted(cur_dir.c_str());
        ImGui::Separator();

        std::error_code ec;
        if (is_content && fs::exists(cur_dir, ec) &&
            fs::is_regular_file(cur_dir, ec)) {
            // ── Virtual asset folder ──────────────────────────────────
            // cur_dir is a model FILE the user opened like a directory:
            // list its renderable sub-objects (mesh nodes) as tiles —
            // the same group/children structure the Outliner shows for
            // the placed asset.  Names come from the .rwmeta sidecar
            // (baked at import) or a one-time CPU parse, cached either way.
            const auto& subs = assetSubObjects(cur_dir);
            const float cell  = 80.0f;
            const float avail = ImGui::GetContentRegionAvail().x;
            int cols = (int)(avail / (cell + 14.0f));
            if (cols < 1) cols = 1;
            if (is_content) browser_grid_cols_ = cols;
            // Fixed column origin: tiles are pinned to grid_x0 + k*(cell+14)
            // so a label wider than the tile can't shift the next column.
            const float grid_x0 = ImGui::GetCursorPosX();
            for (size_t i = 0; i < subs.size(); ++i) {
                ImGui::PushID((int)i);
                ImGui::BeginGroup();
                ImGui::PushStyleColor(ImGuiCol_Button,
                                      ImVec4(0.33f, 0.30f, 0.42f, 1.0f));
                ImGui::Button("MESH", ImVec2(cell, cell));
                ImGui::PopStyleColor();
                std::string disp = subs[i];
                if (ImGui::CalcTextSize(disp.c_str()).x > cell) {
                    while (disp.size() > 1 &&
                           ImGui::CalcTextSize((disp + "..").c_str()).x > cell)
                        disp.pop_back();
                    disp += "..";
                }
                ImGui::TextUnformatted(disp.c_str());
                ImGui::EndGroup();

                // Selection highlight: this sub-object is the one currently
                // previewed in the Debug Display (click or arrow keys).
                if (preview_nav_ == PreviewNav::SubObjects &&
                    preview_nav_path_ == cur_dir &&
                    preview_nav_index_ == (int)i) {
                    const ImVec2 ra = ImGui::GetItemRectMin();
                    const ImVec2 rb = ImGui::GetItemRectMax();
                    ImGui::GetWindowDrawList()->AddRect(
                        ImVec2(ra.x - 2.0f, ra.y - 2.0f),
                        ImVec2(rb.x + 2.0f, rb.y + 2.0f),
                        IM_COL32(255, 190, 50, 255), 4.0f, 0, 2.5f);
                    // Arrow navigation: keep the selection in view.
                    if (browser_scroll_to_selected_) {
                        ImGui::SetScrollHereY(0.5f);
                        browser_scroll_to_selected_ = false;
                    }
                }

                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", subs[i].c_str());
                // Double-click a sub-object → preview it in Debug Display.
                if (ImGui::IsItemHovered() &&
                    ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    buildAssetPreview(cur_dir, (int)i,
                        fs::path(cur_dir).filename().string() +
                        "  /  " + subs[i]);
                }
                ImGui::PopID();
                if ((int)((i + 1) % (size_t)cols) != 0)
                    ImGui::SameLine(grid_x0 +
                        (float)((i + 1) % (size_t)cols) * (cell + 14.0f));
            }
            if (subs.empty())
                ImGui::TextDisabled("(no sub-objects — single-mesh asset)");
        } else if (fs::exists(cur_dir, ec) && fs::is_directory(cur_dir, ec)) {
            // Multi-selection resets when the browsed folder changes, so stale
            // paths from another directory can't linger as highlights.
            if (is_content && content_browser_last_dir_ != cur_dir) {
                content_selected_.clear();
                content_sel_anchor_.clear();
                content_browser_last_dir_ = cur_dir;
            }
            std::vector<fs::directory_entry> items;   // dirs first, then files
            std::vector<fs::directory_entry> dirs, files;
            for (auto& e : fs::directory_iterator(cur_dir, ec)) {
                // Hide dot-entries (.thumbnails, .flux_tmp, .git, …).
                const std::string fn = e.path().filename().string();
                if (!fn.empty() && fn[0] == '.') continue;
                // Sidecar metadata / the asset index stay invisible in the
                // asset view.
                if (is_content &&
                    e.path().extension() == ".rwmeta") continue;
                if (is_content &&
                    e.path().filename() == "asset_index.tsv") continue;
                // ".disabled" markers are the persistent form of the
                // per-tile Enabled toggle — never shown as tiles.
                if (e.path().extension() == ".disabled") continue;
                (e.is_directory() ? dirs : files).push_back(e);
            }
            auto byname = [](const fs::directory_entry& a, const fs::directory_entry& b) {
                return a.path().filename().string() < b.path().filename().string(); };
            std::sort(dirs.begin(), dirs.end(), byname);
            std::sort(files.begin(), files.end(), byname);
            items.insert(items.end(), dirs.begin(), dirs.end());
            items.insert(items.end(), files.begin(), files.end());

            const float cell  = 80.0f;
            const float avail = ImGui::GetContentRegionAvail().x;
            int cols = (int)(avail / (cell + 14.0f));
            if (cols < 1) cols = 1;
            if (is_content) browser_grid_cols_ = cols;
            // Fixed column origin (see the sub-object grid above): pin each
            // column to grid_x0 + k*(cell+14) so wide labels can't skew the row.
            const float grid_x0 = ImGui::GetCursorPosX();

            for (size_t i = 0; i < items.size(); ++i) {
                const fs::directory_entry& e = items[i];
                const std::string name = e.path().filename().string();
                std::string ext = e.path().extension().string();
                for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
                const bool is_dir = e.is_directory();
                const bool is_img = (ext == ".png" || ext == ".jpg" ||
                                     ext == ".jpeg" || ext == ".bmp" ||
                                     ext == ".tga"  || ext == ".dds");
                const bool is_model = (ext == ".gltf" || ext == ".glb" ||
                                       ext == ".obj"  || ext == ".fbx");
                // Exploded sub-object reference asset (one object of a
                // multi-object import) — placeable + previewable.
                const bool is_object = (ext == ".rwobj");
                // Baked render-ready geometry (objects/NNN.rwgeo) —
                // previewable directly.  Character (bake-only) imports
                // have no .rwobj refs, so this is how their skeleton
                // meshes preview in the Debug Display.
                const bool is_geo = (ext == ".rwgeo");
                // Audio clips (generated or imported) — click to preview
                // through the engine's SFX bus, click again to stop.
                const bool is_audio = (ext == ".wav" || ext == ".mp3" ||
                                       ext == ".flac");
                // Skeletal animation clip (Qwen-generated or baked).  Double-
                // click previews it on the standard rig in the Debug Display.
                const bool is_anim = (ext == ".anim");
                // Saved scene file (Scene → Save Scene).  Double-click opens
                // it — the app no-ops if it is already the loaded scene.
                const bool is_scene = (ext == ".scene");
                // Import GROUP folder (holds import.rwmeta + .rwobj files):
                // placeable as a whole — every object, original layout.
                bool is_group_dir = false;
                if (is_content && is_dir) {
                    std::error_code gec;
                    is_group_dir =
                        fs::exists(e.path() / "import.rwmeta", gec);
                }

                // Anything with a generatable thumbnail goes through getThumbnail;
                // it returns 0 (→ type tile) for formats we can't render yet.
                ImTextureID tex = (!is_dir && (is_img || is_model))
                                  ? getThumbnail(e.path().string()) : 0;

                ImGui::PushID((int)i);
                // Debug enable flag: disabled assets are greyed out in the
                // grid.  Persistent form is a "<path>.disabled" sidecar
                // marker, which the .rwchar loader honours at load time —
                // a disabled sub-mesh is SKIPPED everywhere (forward, CSM,
                // both RT shadow paths).  The in-memory set mirrors the
                // sidecars for cheap per-frame dimming.
                const std::string item_path = e.path().string();
                {
                    std::error_code dec;
                    if (fs::exists(item_path + ".disabled", dec))
                        content_disabled_.insert(item_path);
                }
                const bool tile_disabled =
                    content_disabled_.find(item_path) != content_disabled_.end();
                if (tile_disabled)
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha,
                                        ImGui::GetStyle().Alpha * 0.35f);
                ImGui::BeginGroup();
                bool clicked = false;
                if (tex) {
                    clicked = ImGui::ImageButton("##t", tex, ImVec2(cell, cell));
                } else {
                    // Audio tiles flip to a brighter "PLAYING" tint while
                    // their clip is on the preview bus.
                    const bool audio_playing =
                        is_audio && audio_preview_handle_ != 0 &&
                        audio_preview_path_ == e.path().string() &&
                        engine::audio::AudioEngine::isPlaying(
                            audio_preview_handle_);
                    const ImVec4 c = is_dir    ? ImVec4(0.24f, 0.31f, 0.45f, 1.0f)
                                   : is_model  ? ImVec4(0.28f, 0.40f, 0.27f, 1.0f)
                                   : is_object ? ImVec4(0.33f, 0.30f, 0.42f, 1.0f)
                                   : is_geo    ? ImVec4(0.30f, 0.36f, 0.42f, 1.0f)
                                   : audio_playing
                                               ? ImVec4(0.45f, 0.33f, 0.18f, 1.0f)
                                   : is_audio  ? ImVec4(0.40f, 0.30f, 0.22f, 1.0f)
                                   : is_anim   ? ImVec4(0.20f, 0.38f, 0.42f, 1.0f)
                                   : is_scene  ? ImVec4(0.42f, 0.36f, 0.20f, 1.0f)
                                               : ImVec4(0.28f, 0.28f, 0.33f, 1.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, c);
                    std::string glyph = is_dir    ? "DIR"
                                      : is_object ? "OBJ"
                                      : is_geo    ? "GEO"
                                      : audio_playing ? "||>"
                                      : is_audio  ? "SND"
                                      : is_anim   ? "ANM"
                                      : is_scene  ? "SCN"
                        : (ext.size() > 1 ? ext.substr(1) : "?");
                    for (auto& ch : glyph) ch = (char)std::toupper((unsigned char)ch);
                    clicked = ImGui::Button(glyph.c_str(), ImVec2(cell, cell));
                    ImGui::PopStyleColor();
                }
                // Double-click opens a folder (including group folders) or a
                // model as a virtual sub-object folder. Detected independently
                // of the single click below: ImGui::Button reports on mouse
                // RELEASE while a double-click registers on the second PRESS,
                // so the two never land in the same frame -- gating the "enter
                // folder" on `clicked` (as before) never fired for groups.
                const bool tile_dbl =
                    ImGui::IsItemHovered() &&
                    ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
                if (tile_dbl && (is_dir || (is_content && is_model))) {
                    cur_dir = e.path().string();
                }
                // Double-click a .anim → preview it on the standard rig in the
                // right-side Debug Display (works in either browser).
                if (tile_dbl && is_anim) {
                    buildStandardRigAnimPreview(
                        e.path().string(),
                        e.path().stem().string());
                }
                // Double-click a .scene → ask the app to open it.  The app
                // skips the load when this file is already the live scene,
                // so a stray double-click can't dump unsaved edits.
                if (tile_dbl && is_scene) {
                    content_scene_open_request_ = e.path().string();
                }
                // Double-click an image → full-size viewer (the tile only
                // shows the downscaled thumbnail).
                if (tile_dbl && is_img) {
                    openImageViewer(e.path().string());
                }

                // Drag a placeable tile into the 3D viewport to add it to
                // the scene — same effect as right-click → Add to Scene
                // (the drop is handled at the end of the editor pass).
                // Group FOLDERS are draggable too (places every object).
                // Images are draggable too — onto the Generate Image popup
                // as a reference (FLUX.2 conditioning), not into the scene.
                if (is_content &&
                    (is_model || is_object || is_group_dir || is_img ||
                     is_audio) &&
                    ImGui::BeginDragDropSource()) {
                    const std::string pay = e.path().string();
                    ImGui::SetDragDropPayload("RW_CONTENT_ASSET",
                                              pay.c_str(), pay.size() + 1);
                    ImGui::TextUnformatted(name.c_str());
                    ImGui::EndDragDropSource();
                }
                // Truncated one-line label (keeps the grid aligned):
                // trimmed by pixel width so it never exceeds the tile.
                std::string disp = name;
                if (ImGui::CalcTextSize(disp.c_str()).x > cell) {
                    while (disp.size() > 1 &&
                           ImGui::CalcTextSize((disp + "..").c_str()).x > cell)
                        disp.pop_back();
                    disp += "..";
                }
                ImGui::TextUnformatted(disp.c_str());
                ImGui::EndGroup();
                if (tile_disabled) ImGui::PopStyleVar();   // end grey-out

                // ── Selection highlight ───────────────────────────────
                // The tile whose object is currently shown in the Debug
                // Display (clicked or arrow-stepped) gets an amber border,
                // matching the editor's selection colour.
                bool tile_selected = false;
                if (is_content && is_object &&
                    preview_nav_ == PreviewNav::RwObjSiblings) {
                    tile_selected =
                        (e.path().string() == preview_nav_path_);
                } else if (is_content && is_model) {
                    tile_selected =
                        (dbg_asset_key_ == e.path().string() + "#-1");
                } else if (is_content && is_geo) {
                    tile_selected =
                        (dbg_asset_key_ == e.path().string() + "#geo");
                }
                if (tile_selected) {
                    const ImVec2 ra = ImGui::GetItemRectMin();
                    const ImVec2 rb = ImGui::GetItemRectMax();
                    ImGui::GetWindowDrawList()->AddRect(
                        ImVec2(ra.x - 2.0f, ra.y - 2.0f),
                        ImVec2(rb.x + 2.0f, rb.y + 2.0f),
                        IM_COL32(255, 190, 50, 255), 4.0f, 0, 2.5f);
                    // Arrow navigation: keep the selection in view.
                    if (browser_scroll_to_selected_) {
                        ImGui::SetScrollHereY(0.5f);
                        browser_scroll_to_selected_ = false;
                    }
                }

                // Multi-selection highlight (blue border), on top of the amber
                // preview highlight above.
                if (is_content && content_selected_.count(item_path)) {
                    const ImVec2 sra = ImGui::GetItemRectMin();
                    const ImVec2 srb = ImGui::GetItemRectMax();
                    ImGui::GetWindowDrawList()->AddRect(
                        ImVec2(sra.x - 2.0f, sra.y - 2.0f),
                        ImVec2(srb.x + 2.0f, srb.y + 2.0f),
                        IM_COL32(90, 160, 255, 255), 4.0f, 0, 2.5f);
                }

                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name.c_str());
                // Right-clicking a tile NOT in the current selection replaces
                // the selection with just that tile, so the menu acts sensibly.
                if (is_content &&
                    ImGui::IsItemClicked(ImGuiMouseButton_Right) &&
                    !content_selected_.count(item_path)) {
                    content_selected_.clear();
                    content_selected_.insert(item_path);
                    content_sel_anchor_ = item_path;
                }
                // Right-click a tile → context actions.
                if (ImGui::BeginPopupContextItem("##cbctx")) {
                    std::vector<std::string> sel_set;
                    if (is_content && content_selected_.count(item_path))
                        sel_set.assign(content_selected_.begin(),
                                       content_selected_.end());
                    else
                        sel_set.push_back(item_path);
                    const int  sel_n = (int)sel_set.size();
                    const bool multi = sel_n > 1;

                    if (ImGui::MenuItem("Enabled", nullptr, !tile_disabled)) {
                        const bool enable = tile_disabled;
                        std::error_code mec;
                        for (const auto& sp : sel_set) {
                            if (enable) {
                                content_disabled_.erase(sp);
                                fs::remove(fs::path(sp + ".disabled"), mec);
                            } else {
                                content_disabled_.insert(sp);
                                std::ofstream(sp + ".disabled") << "1\n";
                            }
                        }
                        dbg_asset_key_.clear();
                        EditorLog::get().push(
                            std::string("[content] ") +
                            (enable ? "enabled " : "disabled ") +
                            std::to_string(sel_n) + " item(s) — already-loaded "
                            "characters need an app restart to pick this up.");
                    }
                    ImGui::Separator();
                    if (is_content &&
                        (is_model || is_object || is_group_dir) &&
                        ImGui::MenuItem(multi
                            ? (std::string("Add to Scene (") +
                               std::to_string(sel_n) + ")").c_str()
                            : "Add to Scene")) {
                        for (const auto& sp : sel_set) {
                            std::string se = fs::path(sp).extension().string();
                            for (auto& c : se)
                                c = (char)std::tolower((unsigned char)c);
                            const bool placeable =
                                se == ".gltf" || se == ".glb" ||
                                se == ".obj"  || se == ".fbx" ||
                                se == ".rwobj" ||
                                (fs::is_directory(sp, ec) &&
                                 fs::exists(fs::path(sp) / "import.rwmeta", ec));
                            if (placeable) place_asset_queue_.push_back(sp);
                        }
                    }
                    if (ImGui::MenuItem("Rename")) {
                        rename_target_ = e.path().string();
                        std::snprintf(rename_buf_, sizeof(rename_buf_), "%s",
                                      name.c_str());
                        rename_open_ = true;
                    }
                    if (is_content && ImGui::MenuItem(multi
                            ? (std::string("Delete (") +
                               std::to_string(sel_n) + ")").c_str()
                            : "Delete")) {
                        delete_paths_.assign(sel_set.begin(), sel_set.end());
                        delete_open_ = true;
                    }
                    ImGui::EndPopup();
                }
                ImGui::PopID();

                // ── Multi-selection click (Ctrl / Shift / plain) ───────────
                ImGuiIO& io_sel = ImGui::GetIO();
                const bool mod_ctrl  = is_content && io_sel.KeyCtrl;
                const bool mod_shift = is_content && io_sel.KeyShift;
                if (clicked && is_content && mod_ctrl) {
                    if (content_selected_.count(item_path))
                        content_selected_.erase(item_path);
                    else
                        content_selected_.insert(item_path);
                    content_sel_anchor_ = item_path;
                } else if (clicked && is_content && mod_shift) {
                    int ai = -1, ci = -1;
                    for (int k = 0; k < (int)items.size(); ++k) {
                        const std::string sp = items[k].path().string();
                        if (sp == content_sel_anchor_) ai = k;
                        if (sp == item_path)           ci = k;
                    }
                    if (ci >= 0) {
                        if (ai < 0) ai = ci;
                        content_selected_.clear();
                        const int lo = std::min(ai, ci), hi = std::max(ai, ci);
                        for (int k = lo; k <= hi; ++k)
                            content_selected_.insert(items[k].path().string());
                    }
                } else if (clicked && is_content) {
                    content_selected_.clear();
                    content_selected_.insert(item_path);
                    content_sel_anchor_ = item_path;
                }
                const bool plain_click = clicked && !mod_ctrl && !mod_shift;

                // Folders open on click.  Content-Browser model assets are
                // Explorer-style: a click previews the asset in the Debug
                // Display; a double-click ALSO opens it as a virtual folder
                // of its sub-objects.  .rwobj object assets preview their
                // single sub-object on click / double-click.  "Add to
                // Scene" lives in the tile's right-click menu.
                if (plain_click && is_dir) {
                    if (is_group_dir && is_content) {
                        // Group folder, Explorer-style: a single CLICK previews
                        // the assembled collection in the Debug Display; a
                        // DOUBLE-click (handled above) opens the folder.
                        buildRwGroupPreview(e.path().string(), name);
                    } else {
                        cur_dir = e.path().string();   // plain folder: enter
                    }
                } else if (plain_click && is_content && is_model) {
                    buildAssetPreview(e.path().string(), -1, name);
                } else if (plain_click && is_content && is_object) {
                    // Prefers the baked .rwgeo/.rwtex render-ready data.
                    buildRwObjPreview(e.path().string(), name);
                } else if (plain_click && is_content && is_geo) {
                    buildRwGeoPreview(e.path().string(), name);
                } else if (plain_click && is_audio) {
                    // Toggle preview: clicking the playing clip stops it;
                    // clicking another clip switches to it.
                    const std::string p = e.path().string();
                    const bool was_this =
                        audio_preview_handle_ != 0 &&
                        audio_preview_path_ == p &&
                        engine::audio::AudioEngine::isPlaying(
                            audio_preview_handle_);
                    if (audio_preview_handle_ != 0) {
                        engine::audio::AudioEngine::stop(
                            audio_preview_handle_);
                        audio_preview_handle_ = 0;
                    }
                    if (!was_this) {
                        audio_preview_handle_ =
                            engine::audio::AudioEngine::playFile(
                                p, engine::audio::AudioEngine::Bus::kSfx);
                        audio_preview_path_ = p;
                        if (audio_preview_handle_ == 0)
                            EditorLog::get().push(
                                "[audio] preview failed: " + p);
                    }
                }

                if ((int)((i + 1) % (size_t)cols) != 0)
                    ImGui::SameLine(grid_x0 +
                        (float)((i + 1) % (size_t)cols) * (cell + 14.0f));
            }
            if (items.empty()) ImGui::TextDisabled("(empty)");
        } else {
            ImGui::TextDisabled("(folder not found: %s)", cur_dir.c_str());
        }

        // Left-click empty grid space clears the multi-selection (Content
        // Browser only; not while Ctrl/Shift are held for additive picking).
        if (is_content && ImGui::IsWindowHovered() &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
            !ImGui::IsAnyItemHovered() &&
            !ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift) {
            content_selected_.clear();
            content_sel_anchor_.clear();
        }
        // Right-click the grid background (not a tile) → context menu.
        // Content Browser: Import / New Folder live here (no toolbar).
        // Both browsers: FLUX image generation into the current folder.
        if (ImGui::IsWindowHovered() &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
            !ImGui::IsAnyItemHovered()) {
            ImGui::OpenPopup("##cb_bg_ctx");
        }
        if (ImGui::BeginPopup("##cb_bg_ctx")) {
            if (is_content) {
                // Enable/Disable every item in the OPEN folder, using the same
                // content_disabled_ set the per-tile right-click toggle uses
                // (disabled items render dimmed and are skipped by the loader /
                // assembler at build time).  Keys are the entry path strings,
                // exactly as inserted per-tile (e.path().string()).
                auto setFolderEnabled = [&](bool enabled) {
                    std::error_code lec;
                    int n = 0;
                    for (auto& e : fs::directory_iterator(cur_dir, lec)) {
                        if (lec) break;
                        if (e.path().extension() == ".disabled") continue;
                        const std::string ip = e.path().string();
                        std::error_code mec;
                        if (enabled) {
                            if (content_disabled_.erase(ip) > 0) ++n;
                            fs::remove(fs::path(ip + ".disabled"), mec);
                        } else {
                            if (content_disabled_.insert(ip).second) ++n;
                            std::ofstream(ip + ".disabled") << "1\n";
                        }
                    }
                    EditorLog::get().push(
                        std::string("[content] ") + (enabled ? "enabled " : "disabled ") +
                        std::to_string(n) + " item(s) in " + cur_dir);
                };
                if (ImGui::MenuItem("Enable All (this folder)"))  setFolderEnabled(true);
                if (ImGui::MenuItem("Disable All (this folder)")) setFolderEnabled(false);
                ImGui::Separator();
                if (ImGui::MenuItem("Import...")) {
                    content_import_request_ = true;
                    content_import_dir_     = cur_dir;
                }
                if (ImGui::MenuItem("New Folder...")) {
                    newfolder_buf_[0] = '\0';
                    newfolder_open_   = true;
                }
                ImGui::Separator();
            }
            if (ImGui::MenuItem("Generate Image...")) {
                gen_popup_pending_ = true;
                gen_folder_        = cur_dir;
            }
            if (ImGui::MenuItem("Generate Audio...")) {
                agen_popup_pending_ = true;
                agen_folder_        = cur_dir;
            }
            if (ImGui::MenuItem("Generate Animation...")) {
                nanim_popup_pending_ = true;
                nanim_folder_        = cur_dir;
            }
            ImGui::EndPopup();
        }
        ImGui::PopItemFlag();   // ImGuiItemFlags_NoNav (grid tiles)
        ImGui::EndChild();

        // ── Rename dialog (right-click a tile → Rename) ────────────────────
        if (rename_open_) { ImGui::OpenPopup("Rename##cb"); rename_open_ = false; }
        if (ImGui::BeginPopupModal("Rename##cb", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextDisabled("%s",
                std::filesystem::path(rename_target_).filename().string().c_str());
            ImGui::TextUnformatted("New name:");
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(320.0f);
            const bool enter = ImGui::InputText("##rn_in", rename_buf_,
                sizeof(rename_buf_), ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::Spacing();
            const bool ok = ImGui::Button("Rename", ImVec2(120, 0)) || enter;
            ImGui::SameLine();
            const bool cancel = ImGui::Button("Cancel", ImVec2(120, 0));

            if (ok && rename_buf_[0] != '\0' && !rename_target_.empty()) {
                fs::path src(rename_target_);
                fs::path dst = src.parent_path() / rename_buf_;
                std::error_code rec2;
                if (fs::exists(dst, rec2)) {
                    EditorLog::get().push("[rename] target already exists: " +
                                          dst.filename().string());
                } else {
                    fs::rename(src, dst, rec2);
                    if (rec2) {
                        EditorLog::get().push("[rename] failed: " + rec2.message());
                    } else {
                        EditorLog::get().push("[rename] " +
                            src.filename().string() + " -> " +
                            std::string(rename_buf_));
                        if (cur_dir == rename_target_)
                            cur_dir = dst.string();
                        // Keep an imported asset's sidecar travelling with it.
                        std::error_code mec;
                        const std::string meta_src = rename_target_ + ".rwmeta";
                        if (is_content && fs::exists(meta_src, mec)) {
                            fs::rename(meta_src,
                                       dst.string() + ".rwmeta", mec);
                        }
                    }
                }
                ImGui::CloseCurrentPopup();
            }
            if (cancel) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // ── Delete confirmation (right-click a tile → Delete) ──────────────
        if (delete_open_) { ImGui::OpenPopup("Delete##cb"); delete_open_ = false; }
        if (ImGui::BeginPopupModal("Delete##cb", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            std::vector<std::string> del_list = delete_paths_;
            if (del_list.empty() && !delete_target_.empty())
                del_list.push_back(delete_target_);
            const fs::path tgt(del_list.empty() ? std::string() : del_list[0]);
            if (del_list.size() > 1)
                ImGui::Text("Delete %d selected items?", (int)del_list.size());
            else
                ImGui::Text("Delete '%s'?", tgt.filename().string().c_str());
            if (del_list.size() > 1) {
                ImGui::TextDisabled("Folders remove everything inside; models "
                                    "also drop rwmeta / exploded / thumbnail.");
                ImGui::BeginChild("##dl", ImVec2(360,120), true);
                for (const auto& dp : del_list)
                    ImGui::BulletText("%s", fs::path(dp).filename().string().c_str());
                ImGui::EndChild();
            } else if (delete_is_dir_) {
                ImGui::TextDisabled(
                    "The folder and EVERYTHING inside it will be removed.");
            } else {
                std::string lext = tgt.extension().string();
                for (auto& c : lext) c = (char)std::tolower((unsigned char)c);
                const bool del_model =
                    (lext == ".gltf" || lext == ".glb" ||
                     lext == ".obj"  || lext == ".fbx");
                if (del_model) {
                    ImGui::TextDisabled(
                        "Also removes its .rwmeta sidecar, the exploded\n"
                        "'%s/' object folder, and its cached thumbnail.",
                        tgt.stem().string().c_str());
                }
                ImGui::TextDisabled(
                    "Objects already placed in the scene keep rendering,\n"
                    "but will fail to load after a scene reload.");
            }
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.70f, 0.20f, 0.20f, 1.0f));
            const bool del_ok = ImGui::Button("Delete", ImVec2(120, 0));
            ImGui::PopStyleColor();
            ImGui::SameLine();
            const bool del_cancel = ImGui::Button("Cancel", ImVec2(120, 0));

            if (del_ok && !del_list.empty()) {
                auto deleteOne = [&](const std::string& path) {
                    const fs::path t(path);
                    std::error_code dec;
                    if (fs::is_directory(t, dec)) {
                        fs::remove_all(t, dec);
                    } else {
                        fs::remove(t, dec);
                        std::error_code dec2;
                        fs::remove(fs::path(path + ".rwmeta"), dec2);
                        const fs::path exploded = t.parent_path() / t.stem();
                        if (fs::is_directory(exploded, dec2))
                            fs::remove_all(exploded, dec2);
                        const fs::path thumb = t.parent_path() / ".thumbnails" /
                            (t.filename().string() + ".png");
                        fs::remove(thumb, dec2);
                    }
                    if (dec)
                        EditorLog::get().push("[delete] failed: " +
                            t.filename().string() + " — " + dec.message());
                    else
                        EditorLog::get().push("[delete] removed '" +
                            t.filename().string() + "'");
                    asset_children_cache_.erase(path);
                    auto itc = thumb_cache_.find(path);
                    if (itc != thumb_cache_.end()) {
                        if (itc->second.info)
                            retired_thumbs_.push_back(itc->second.info);
                        thumb_cache_.erase(itc);
                    }
                    if (cur_dir.rfind(path, 0) == 0) cur_dir = tree_root;
                    content_selected_.erase(path);
                };
                for (const auto& dp : del_list) deleteOne(dp);
                delete_paths_.clear();
                delete_target_.clear();
                content_sel_anchor_.clear();
                ImGui::CloseCurrentPopup();
            }
            if (del_cancel) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // ── New Folder dialog (Content Browser toolbar) ────────────────────
        if (is_content) {
            if (newfolder_open_) {
                ImGui::OpenPopup("New Folder##cb");
                newfolder_open_ = false;
            }
            if (ImGui::BeginPopupModal("New Folder##cb", nullptr,
                                       ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::TextDisabled("in %s", cur_dir.c_str());
                ImGui::TextUnformatted("Folder name:");
                if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
                ImGui::SetNextItemWidth(320.0f);
                const bool nf_enter = ImGui::InputText("##nf_in",
                    newfolder_buf_, sizeof(newfolder_buf_),
                    ImGuiInputTextFlags_EnterReturnsTrue);
                ImGui::Spacing();
                const bool nf_ok = ImGui::Button("Create", ImVec2(120, 0)) ||
                                   nf_enter;
                ImGui::SameLine();
                const bool nf_cancel = ImGui::Button("Cancel", ImVec2(120, 0));
                if (nf_ok && newfolder_buf_[0] != '\0') {
                    std::error_code nec;
                    const fs::path np = fs::path(cur_dir) / newfolder_buf_;
                    if (fs::exists(np, nec)) {
                        EditorLog::get().push(
                            "[content] folder already exists: " + np.string());
                    } else {
                        fs::create_directories(np, nec);
                        if (nec) {
                            EditorLog::get().push(
                                "[content] create folder failed: " +
                                nec.message());
                        } else {
                            EditorLog::get().push(
                                "[content] created folder: " + np.string());
                            cur_dir = np.string();
                        }
                    }
                    ImGui::CloseCurrentPopup();
                }
                if (nf_cancel) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
        }
    }
}

// ── Panel wrappers ───────────────────────────────────────────────────────────
void Menu::drawContentBrowserPanel() {
    // Ensure the project's content root exists so the tree / grid / import
    // target always have a home (cheap no-op when already present).
    std::error_code ec;
    std::filesystem::create_directories("content", ec);
    if (content_dir_.empty()) content_dir_ = "content";
    if (ImGui::Begin("Content Browser")) {
        drawBrowserBody("content", content_dir_, content_left_w_,
                        /*is_content=*/true);
    }
    ImGui::End();
}

void Menu::drawFileBrowserPanel() {
    if (ImGui::Begin("File Browser")) {
        drawBrowserBody(".", file_dir_, file_left_w_,
                        /*is_content=*/false);
    }
    ImGui::End();
}

// ── FLUX.2 generate-image popup ──────────────────────────────────────────────
// Single instance shared by both browsers (each sets gen_folder_ when its
// grid background is right-clicked).  Drawn once per frame from the editor
// panel pass so the two panels can't double-submit the same window.
void Menu::drawFluxGeneratePopup() {
    namespace fs = std::filesystem;

    // Poll a running generation: the output PNG appears (success) or
    // "<out>.err" (failure).  The grid's thumbnail system shows the PNG.
    if (gen_status_ == 1) {
        std::error_code pec;
        // Sidecars (.err) live in <folder>/.flux_tmp/<name>.err
        fs::path _op(gen_out_path_);
        const std::string err_path =
            (_op.parent_path() / ".flux_tmp" /
             (_op.filename().string() + ".err")).string();
        if (fs::exists(gen_out_path_, pec)) {
            gen_status_ = 2;
            EditorLog::get().push("[flux] image ready: " + gen_out_path_);
        } else if (fs::exists(err_path, pec)) {
            gen_status_ = 3;
            std::ifstream ef(err_path, std::ios::binary);
            gen_err_.assign(std::istreambuf_iterator<char>(ef),
                            std::istreambuf_iterator<char>());
            EditorLog::get().push("[flux] generation FAILED: " + gen_err_);
        }
    }

    {
        static bool s_gen_open = false;
        if (gen_popup_pending_) {
            gen_popup_pending_ = false;
            if (gen_folder_.empty()) gen_folder_ = content_dir_;
            s_gen_open = true;
        }
        if (s_gen_open) {
            // Appearing (not Always) so it's resizable; tall enough for the
            // reference-image list + button row with the fantasy font.
            ImGui::SetNextWindowSize(ImVec2(624.0f, 640.0f),
                                     ImGuiCond_Appearing);
            ImGui::SetNextWindowSizeConstraints(ImVec2(360.0f, 360.0f),
                                                ImVec2(8192.0f, 8192.0f));
            if (ImGui::Begin("Generate Image", &s_gen_open,
                             ImGuiWindowFlags_NoCollapse)) {
            ImGui::TextDisabled("Generate image into:");
            ImGui::TextUnformatted(gen_folder_.c_str());
            ImGui::Separator();

            ImGui::TextUnformatted("Prompt");
            // Fill the window width so the box grows when the window is resized.
            const float box_w = ImGui::GetContentRegionAvail().x;
            GenWrapUD genud;
            genud.wrap_w = box_w - ImGui::GetStyle().FramePadding.x * 2.0f
                                 - ImGui::GetStyle().ScrollbarSize - 4.0f;
            // CallbackAlways: re-wrap EVERY line each frame (and on resize).
            ImGui::InputTextMultiline("##gen_prompt", gen_prompt_,
                sizeof(gen_prompt_),
                ImVec2(box_w, ImGui::GetTextLineHeightWithSpacing() * 7.0f),
                ImGuiInputTextFlags_CallbackAlways, &GenPromptWrapCallback,
                &genud);

            ImGui::Spacing();
            ImGui::TextUnformatted("Name (optional)");
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            ImGui::InputTextWithHint("##gen_name",
                "leave empty for an auto name", gen_name_, sizeof(gen_name_));
            ImGui::Spacing();

            struct Sz { const char* label; int w, h; };
            static const Sz kSizes[] = {
                {"32 x 32 (icon)",            32,   32},
                {"64 x 64 (icon)",            64,   64},
                {"128 x 128 (icon)",         128,  128},
                {"256 x 256 (icon)",         256,  256},
                {"512 x 512",                512,  512},
                {"768 x 768",                768,  768},
                {"1024 x 1024",             1024, 1024},
                {"1024 x 1536 (portrait)",  1024, 1536},
                {"1536 x 1024 (landscape)", 1536, 1024},
            };
            const int nSz = (int)(sizeof(kSizes) / sizeof(kSizes[0]));
            if (gen_size_idx_ < 0 || gen_size_idx_ >= nSz) gen_size_idx_ = 6;
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            if (ImGui::BeginCombo("##gen_size", kSizes[gen_size_idx_].label)) {
                for (int i = 0; i < nSz; ++i)
                    if (ImGui::Selectable(kSizes[i].label, gen_size_idx_ == i))
                        gen_size_idx_ = i;
                ImGui::EndCombo();
            }

            // ── Reference images (FLUX.2 native conditioning) ─────────────
            // Up to 4 images guide structure / style / edits; the prompt
            // says HOW to use them ("same character, new pose", "in the
            // style of this", "recolor this").  Add via picker or by
            // dragging Content Browser image tiles onto this window.
            ImGui::Spacing();
            ImGui::TextUnformatted("Reference images (optional, max 4)");
            for (size_t i = 0; i < gen_ref_images_.size(); ++i) {
                ImGui::PushID((int)(1000 + i));
                if (ImGui::SmallButton("X")) {
                    gen_ref_images_.erase(gen_ref_images_.begin() + i);
                    ImGui::PopID();
                    break;
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(std::filesystem::path(gen_ref_images_[i])
                                           .filename().string().c_str());
                ImGui::PopID();
            }
            const bool full = gen_ref_images_.size() >= 4;
            if (full) ImGui::BeginDisabled();
            if (ImGui::SmallButton("Add Reference...")) {
                const std::string img =
                    engine::scene::openImageFileDialog(nullptr,
                                                       gen_folder_.c_str());
                if (!img.empty() && gen_ref_images_.size() < 4)
                    gen_ref_images_.push_back(img);
            }
            if (full) ImGui::EndDisabled();
            if (!gen_ref_images_.empty()) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Clear")) gen_ref_images_.clear();
            }
            // Accept Content Browser image tiles dropped anywhere in the
            // window (the tile drag source emits "RW_CONTENT_ASSET").
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* pl =
                        ImGui::AcceptDragDropPayload("RW_CONTENT_ASSET")) {
                    const std::string dropped((const char*)pl->Data);
                    std::string ext =
                        std::filesystem::path(dropped).extension().string();
                    for (auto& c : ext)
                        c = (char)std::tolower((unsigned char)c);
                    const bool is_img = (ext == ".png" || ext == ".jpg" ||
                                         ext == ".jpeg" || ext == ".bmp");
                    if (is_img && gen_ref_images_.size() < 4)
                        gen_ref_images_.push_back(dropped);
                }
                ImGui::EndDragDropTarget();
            }

            ImGui::Separator();
            // Split the row evenly so neither button's label is clipped
            // (the fantasy font makes "Regenerate" wider than a fixed 120px).
            const float btn_w = (ImGui::GetContentRegionAvail().x
                                 - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
            const bool busy = (gen_status_ == 1);
            if (busy) ImGui::BeginDisabled();
            if (ImGui::Button("Generate", ImVec2(btn_w, 0)))
                launchImageGen(gen_folder_, gen_prompt_,
                               kSizes[gen_size_idx_].w, kSizes[gen_size_idx_].h);
            ImGui::SameLine();
            const bool can_regen = !gen_last_prompt_.empty();
            if (!can_regen) ImGui::BeginDisabled();
            if (ImGui::Button("Regenerate", ImVec2(btn_w, 0)))
                launchImageGen(gen_folder_, gen_last_prompt_,
                               gen_last_w_, gen_last_h_);
            if (!can_regen) ImGui::EndDisabled();
            if (busy) ImGui::EndDisabled();

            if (gen_status_ == 1)
                ImGui::TextColored(ImVec4(1.0f, 0.80f, 0.30f, 1.0f),
                    "Generating... (FLUX.2 — this can take a while)");
            else if (gen_status_ == 2)
                ImGui::TextColored(ImVec4(0.40f, 0.90f, 0.50f, 1.0f),
                    "Done — image added to this folder.");
            else if (gen_status_ == 3)
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f),
                    "Failed — see Output Log.");
            }
            ImGui::End();
        }
    }
}

// ── Text-to-audio generation (Stable Audio Open via tools/audiogen) ─────────
// Same contract as the FLUX image tool: write the prompt to a sidecar file,
// spawn the python generator detached, poll for the output WAV (success) or
// "<out>.err" (failure traceback).
void Menu::launchAudioGen(const std::string& folder, const std::string& prompt,
                          int type_idx, float duration_sec) {
    namespace fs = std::filesystem;
    std::string p = prompt;
    for (char& c : p) if (c == '\n' || c == '\r') c = ' ';
    while (!p.empty() && (p.back() == ' ' || p.back() == '\t'))
        p.pop_back();
    if (p.empty()) { agen_status_ = 3; agen_err_ = "empty prompt"; return; }

    std::error_code ec;
    fs::create_directories(folder, ec);

    static int s_n = 0;
    // Optional name from the popup (agen_name_): sanitize + drop a typed
    // ".wav"; empty -> a unique auto name (a reused name overwrites).
    std::string base;
    {
        std::string clean;
        for (char ch : std::string(agen_name_)) {
            if (ch=='/'||ch=='\\'||ch==':'||ch=='*'||ch=='?'||ch=='"'||
                ch=='<'||ch=='>'||ch=='|'||ch=='\n'||ch=='\r'||ch=='\t')
                continue;
            clean.push_back(ch);
        }
        while (!clean.empty() && clean.front()==' ') clean.erase(clean.begin());
        while (!clean.empty() && clean.back()==' ')  clean.pop_back();
        if (clean.size() >= 4) {
            std::string e = clean.substr(clean.size()-4);
            for (char& c2 : e) if (c2>='A'&&c2<='Z') c2=(char)(c2+32);
            if (e == ".wav") clean.resize(clean.size()-4);
        }
        base = clean.empty()
             ? (std::string(type_idx == 1 ? "sfx_" : "music_") +
                std::to_string((long long)std::time(nullptr)) +
                "_" + std::to_string(s_n++))
             : clean;
    }
    agen_out_path_ = (fs::path(folder) / (base + ".wav")).string();
    // Stale results from a previous run with the same name would satisfy the
    // poll instantly — clear them first.
    fs::remove(agen_out_path_, ec);
    fs::remove(agen_out_path_ + ".err", ec);

    fs::path tmp_dir = fs::path(folder) / ".audiogen_tmp";
    fs::create_directories(tmp_dir, ec);
    const std::string pfile =
        (tmp_dir / (base + ".wav.prompt.txt")).string();
    { std::ofstream pf(pfile, std::ios::binary); pf << p; }

    agen_last_prompt_ = p;
    agen_last_type_   = type_idx;
    agen_last_dur_    = duration_sec;
    agen_err_.clear();

    const char* type_arg = (type_idx == 1) ? "sfx" : "music";
    char dur[32];
    std::snprintf(dur, sizeof(dur), "%.1f", duration_sec);

    // Same interpreter Setup.bat configured (shared CUDA torch install).
#ifdef _WIN32
    const std::string script = "tools\\audiogen\\audiogen_generate.py";
    const std::string cmd =
        "cmd /c start \"audiogen\" /B python \"" + script + "\""
        " --prompt-file \"" + pfile + "\" --out \"" + agen_out_path_ + "\""
        " --type " + type_arg + " --duration " + dur;
#else
    const std::string script = "tools/audiogen/audiogen_generate.py";
    const std::string cmd =
        "python3 \"" + script + "\""
        " --prompt-file \"" + pfile + "\" --out \"" + agen_out_path_ + "\""
        " --type " + std::string(type_arg) + " --duration " + dur + " &";
#endif
    EditorLog::get().push("[audiogen] launching generation -> " +
                          agen_out_path_);
    std::system(cmd.c_str());
    agen_status_ = 1;   // running
}

// Generate-audio popup — single instance shared by both browsers, drawn once
// per frame next to drawFluxGeneratePopup().
void Menu::drawAudioGeneratePopup() {
    namespace fs = std::filesystem;

    // Poll a running generation: the WAV appears (success) or "<out>.err"
    // (failure, traceback inside).
    if (agen_status_ == 1) {
        std::error_code pec;
        if (fs::exists(agen_out_path_, pec)) {
            agen_status_ = 2;
            EditorLog::get().push("[audiogen] clip ready: " + agen_out_path_);
        } else if (fs::exists(agen_out_path_ + ".err", pec)) {
            agen_status_ = 3;
            std::ifstream ef(agen_out_path_ + ".err", std::ios::binary);
            agen_err_.assign(std::istreambuf_iterator<char>(ef),
                             std::istreambuf_iterator<char>());
            EditorLog::get().push("[audiogen] generation FAILED: " +
                                  agen_err_);
        }
    }

    static bool s_open = false;
    if (agen_popup_pending_) {
        agen_popup_pending_ = false;
        if (agen_folder_.empty()) agen_folder_ = content_dir_;
        s_open = true;
    }
    if (!s_open) return;

    // Appearing (not Always) so the user can resize; tall enough that the
    // button row + status line fit below the prompt with the fantasy font.
    ImGui::SetNextWindowSize(ImVec2(624.0f, 560.0f), ImGuiCond_Appearing);
    ImGui::SetNextWindowSizeConstraints(ImVec2(360.0f, 320.0f),
                                        ImVec2(8192.0f, 8192.0f));
    if (ImGui::Begin("Generate Audio", &s_open,
                     ImGuiWindowFlags_NoCollapse)) {
        ImGui::TextDisabled("Generate audio into:");
        ImGui::TextUnformatted(agen_folder_.c_str());
        ImGui::Separator();

        ImGui::TextUnformatted("Prompt");
        const float box_w = ImGui::GetContentRegionAvail().x;
        GenWrapUD genud;
        genud.wrap_w = box_w - ImGui::GetStyle().FramePadding.x * 2.0f
                             - ImGui::GetStyle().ScrollbarSize - 4.0f;
        ImGui::InputTextMultiline("##agen_prompt", agen_prompt_,
            sizeof(agen_prompt_),
            ImVec2(box_w, ImGui::GetTextLineHeightWithSpacing() * 5.0f),
            ImGuiInputTextFlags_CallbackAlways, &GenPromptWrapCallback,
            &genud);

        ImGui::Spacing();
        ImGui::TextUnformatted("Name (optional)");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputTextWithHint("##agen_name",
            "leave empty for an auto name", agen_name_, sizeof(agen_name_));

        ImGui::Spacing();
        const char* kTypes[] = { "Music", "Sound Effect" };
        if (agen_type_idx_ < 0 || agen_type_idx_ > 1) agen_type_idx_ = 0;
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::BeginCombo("##agen_type", kTypes[agen_type_idx_])) {
            for (int i = 0; i < 2; ++i)
                if (ImGui::Selectable(kTypes[i], agen_type_idx_ == i))
                    agen_type_idx_ = i;
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::SliderFloat("##agen_dur", &agen_duration_, 1.0f, 47.0f,
                           "%.0f s");

        ImGui::Separator();
        const float btn_w = (ImGui::GetContentRegionAvail().x
                             - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        const bool busy = (agen_status_ == 1);
        if (busy) ImGui::BeginDisabled();
        if (ImGui::Button("Generate", ImVec2(btn_w, 0)))
            launchAudioGen(agen_folder_, agen_prompt_, agen_type_idx_,
                           agen_duration_);
        ImGui::SameLine();
        const bool can_regen = !agen_last_prompt_.empty();
        if (!can_regen) ImGui::BeginDisabled();
        if (ImGui::Button("Regenerate", ImVec2(btn_w, 0)))
            launchAudioGen(agen_folder_, agen_last_prompt_,
                           agen_last_type_, agen_last_dur_);
        if (!can_regen) ImGui::EndDisabled();
        if (busy) ImGui::EndDisabled();

        if (agen_status_ == 1)
            ImGui::TextColored(ImVec4(1.0f, 0.80f, 0.30f, 1.0f),
                "Generating... (Stable Audio Open — this can take a while)");
        else if (agen_status_ == 2) {
            ImGui::TextColored(ImVec4(0.40f, 0.90f, 0.50f, 1.0f),
                "Done — clip added to this folder.");
            ImGui::SameLine();
            if (ImGui::SmallButton("Play")) {
                if (audio_preview_handle_ != 0)
                    engine::audio::AudioEngine::stop(audio_preview_handle_);
                audio_preview_handle_ =
                    engine::audio::AudioEngine::playFile(
                        agen_out_path_,
                        engine::audio::AudioEngine::Bus::kSfx);
                audio_preview_path_ = agen_out_path_;
            }
        } else if (agen_status_ == 3)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f),
                "Failed — see Output Log.");
    }
    ImGui::End();
}

// ── Text-to-animation popup (Qwen via Ollama → binary .anim) ────────────────
// Unlike FLUX/audio (detached Python subprocesses), this runs the generator
// in-process on a worker thread (plugins::auto_rig::generateAnimationFile).
void Menu::drawAnimGeneratePopup() {
    namespace fs = std::filesystem;

    // Poll the worker.
    if (nanim_status_ == 1 && nanim_future_.valid() &&
        nanim_future_.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
        bool ok = false;
        try { ok = nanim_future_.get(); }
        catch (const std::exception& e) { nanim_err_ = e.what(); ok = false; }
        if (ok) {
            nanim_status_ = 2;
            EditorLog::get().push("[anim] generated: " + nanim_out_path_);
        } else {
            nanim_status_ = 3;
            EditorLog::get().push("[anim] generation FAILED: " + nanim_err_);
        }
    }

    static bool s_open = false;
    if (nanim_popup_pending_) {
        nanim_popup_pending_ = false;
        if (nanim_folder_.empty()) nanim_folder_ = content_dir_;
        s_open = true;
    }
    if (!s_open) return;

    ImGui::SetNextWindowSize(ImVec2(560.0f, 380.0f), ImGuiCond_Appearing);
    ImGui::SetNextWindowSizeConstraints(ImVec2(360.0f, 280.0f),
                                        ImVec2(8192.0f, 8192.0f));
    if (ImGui::Begin("Generate Animation", &s_open,
                     ImGuiWindowFlags_NoCollapse)) {
        ImGui::TextDisabled("Generate .anim into:");
        ImGui::TextUnformatted(nanim_folder_.c_str());
        ImGui::TextDisabled("Local model: Qwen via Ollama  (needs `ollama serve`)");
        ImGui::Separator();

        ImGui::TextUnformatted("Motion prompt");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputText("##nanim_prompt", nanim_prompt_, sizeof(nanim_prompt_));
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("e.g. \"walk forward\", \"wave right hand\", "
                              "\"jump\", \"idle breathing\"");

        ImGui::Spacing();
        ImGui::TextUnformatted("Name (optional)");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputTextWithHint("##nanim_name",
            "leave empty for an auto name", nanim_name_, sizeof(nanim_name_));

        ImGui::Spacing();
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderInt("Seconds", &nanim_seconds_, 1, 10);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderInt("FPS", &nanim_fps_, 12, 30);

        ImGui::Separator();
        const bool busy = (nanim_status_ == 1);
        if (busy) ImGui::BeginDisabled();
        if (ImGui::Button("Generate", ImVec2(160, 0)) && nanim_prompt_[0]) {
            std::string base = nanim_name_[0]
                ? std::string(nanim_name_)
                : ("anim_" + std::to_string((long long)std::time(nullptr)));
            fs::path bp(base);
            if (bp.extension() != ".anim") bp.replace_extension(".anim");
            nanim_out_path_ = (fs::path(nanim_folder_) / bp).string();
            const std::string prompt = nanim_prompt_;
            const int secs = nanim_seconds_, fps = nanim_fps_;
            const std::string outp = nanim_out_path_;
            nanim_err_.clear();
            nanim_status_ = 1;
            EditorLog::get().push("[anim] generating '" + prompt + "' -> " + outp);
            nanim_future_ = std::async(std::launch::async,
                [this, prompt, secs, fps, outp]() {
                    return plugins::auto_rig::generateAnimationFile(
                        prompt, secs, fps, outp, nanim_err_);
                });
        }
        if (busy) ImGui::EndDisabled();

        if (nanim_status_ == 1)
            ImGui::TextColored(ImVec4(1.0f, 0.80f, 0.30f, 1.0f),
                "Generating... (Qwen — CPU inference can take a while)");
        else if (nanim_status_ == 2) {
            ImGui::TextColored(ImVec4(0.40f, 0.90f, 0.50f, 1.0f),
                "Done — %s added.",
                fs::path(nanim_out_path_).filename().string().c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton("Preview"))
                buildStandardRigAnimPreview(nanim_out_path_,
                    fs::path(nanim_out_path_).stem().string());
            ImGui::TextDisabled("(double-click the .anim tile to preview anytime)");
        } else if (nanim_status_ == 3)
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f),
                "Failed — see Output Log.");
    }
    ImGui::End();
}

// ── Animated standard-rig skeleton preview from a .anim file ────────────────
// Builds a canonical 19-joint humanoid (T-pose), a procedural "bone" mesh
// rigged 1:1 to it, converts the AnimClip into the engine's RwAnimClip, and
// populates the same preview_* members buildRwGroupPreview uses — so the
// existing tickPreviewAnimation() plays it (with the bone overlay) in the
// right-side Debug Display.
void Menu::buildStandardRigAnimPreview(const std::string& anim_path,
                                       const std::string& caption) {
    namespace ar = plugins::auto_rig;
    ar::AnimClip clip;
    if (!ar::loadAnimClip(clip, anim_path)) {
        EditorLog::get().push("[anim] preview: failed to load " + anim_path);
        return;
    }
    const std::vector<std::string>& names   = ar::getStandardJointNames();
    const std::vector<int>&         parents = ar::getStandardJointParents();
    const int N = (int)names.size();

    // Canonical T-pose bind positions (metres; Y up, +X = character's left).
    std::vector<glm::vec3> bind(N, glm::vec3(0.0f));
    bind[0]  = {  0.00f, 1.00f, 0.0f };   // hips
    bind[1]  = {  0.00f, 1.15f, 0.0f };   // spine
    bind[2]  = {  0.00f, 1.35f, 0.0f };   // chest
    bind[3]  = {  0.00f, 1.52f, 0.0f };   // neck
    bind[4]  = {  0.00f, 1.66f, 0.0f };   // head
    bind[5]  = {  0.12f, 1.45f, 0.0f };   // left_shoulder
    bind[6]  = {  0.25f, 1.45f, 0.0f };   // left_upper_arm
    bind[7]  = {  0.50f, 1.45f, 0.0f };   // left_lower_arm
    bind[8]  = {  0.72f, 1.45f, 0.0f };   // left_hand
    bind[9]  = { -0.12f, 1.45f, 0.0f };   // right_shoulder
    bind[10] = { -0.25f, 1.45f, 0.0f };   // right_upper_arm
    bind[11] = { -0.50f, 1.45f, 0.0f };   // right_lower_arm
    bind[12] = { -0.72f, 1.45f, 0.0f };   // right_hand
    bind[13] = {  0.10f, 0.92f, 0.0f };   // left_upper_leg
    bind[14] = {  0.10f, 0.52f, 0.0f };   // left_lower_leg
    bind[15] = {  0.10f, 0.08f, 0.05f };  // left_foot (toe fwd)
    bind[16] = { -0.10f, 0.92f, 0.0f };   // right_upper_leg
    bind[17] = { -0.10f, 0.52f, 0.0f };   // right_lower_leg
    bind[18] = { -0.10f, 0.08f, 0.05f };  // right_foot

    // Skeleton node arrays + bind world matrices (rotation = identity).
    preview_node_parent_.assign(N, -1);
    preview_node_name_.assign(N, std::string());
    preview_node_bind_local_.assign(N, glm::mat4(1.0f));
    preview_node_invbind_.assign(N, glm::mat4(1.0f));
    preview_node_rot_limit_.assign(N, 3.14159265f);
    for (int i = 0; i < N; ++i) {
        preview_node_parent_[i] = parents[i];
        preview_node_name_[i]   = names[i];
        const glm::vec3 off =
            (parents[i] >= 0) ? (bind[i] - bind[parents[i]]) : bind[i];
        preview_node_bind_local_[i] = glm::translate(glm::mat4(1.0f), off);
        preview_node_invbind_[i] =
            glm::translate(glm::mat4(1.0f), -bind[i]);   // inverse(T(bind))
        // Rotation cap (radians) so garbage angles can't fold a limb through
        // the body, while allowing the FULL natural human range (arms reach
        // down to the sides and overhead).  MUST match rot_cap_deg() in the
        // training pipeline (ml_training/anim_finetune/anim_common.py).
        const std::string& n = names[i];
        auto has = [&](const char* s){ return n.find(s) != std::string::npos; };
        float lim = 3.14159265f;
        if      (has("lower_leg"))                  lim = 2.62f;  // 150 deg
        else if (has("lower_arm"))                  lim = 2.53f;  // 145 deg
        else if (has("shoulder") || has("upper_arm")) lim = 2.88f; // 165 deg
        else if (has("upper_leg"))                  lim = 1.92f;  // 110 deg
        else if (has("spine") || has("chest") ||
                 has("neck")  || has("head"))       lim = 0.70f;  // 40 deg
        else if (has("hand")  || has("foot"))       lim = 1.05f;  // 60 deg
        preview_node_rot_limit_[i] = lim;
    }
    preview_mesh_node_inv_ = glm::mat4(1.0f);

    // Procedural geometry: octahedron marker at each joint (weighted to that
    // joint) + a "bone" diamond per parent→child segment (weighted to PARENT,
    // since a rigid bone rotates with its parent joint).
    engine::helper::MeshPreviewPayload p;
    std::vector<glm::uvec4> sj;
    std::vector<glm::vec4>  sw, sc;
    auto addVert = [&](const glm::vec3& pos, int joint) {
        p.positions.push_back(pos);
        p.normals.push_back(glm::vec3(0.0f));       // recomputed below
        p.uvs.push_back(glm::vec2(0.0f));
        sj.push_back(glm::uvec4((uint32_t)joint, 0, 0, 0));
        sw.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        sc.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    };
    auto addTri = [&](uint32_t a, uint32_t b, uint32_t c) {
        p.indices.push_back(a); p.indices.push_back(b); p.indices.push_back(c);
    };
    auto addOcta = [&](const glm::vec3& c, float r, int joint) {
        const uint32_t b = (uint32_t)p.positions.size();
        addVert(c + glm::vec3( r, 0, 0), joint);   // 0 +x
        addVert(c + glm::vec3(-r, 0, 0), joint);   // 1 -x
        addVert(c + glm::vec3( 0, r, 0), joint);   // 2 +y
        addVert(c + glm::vec3( 0,-r, 0), joint);   // 3 -y
        addVert(c + glm::vec3( 0, 0, r), joint);   // 4 +z
        addVert(c + glm::vec3( 0, 0,-r), joint);   // 5 -z
        addTri(b+0,b+2,b+4); addTri(b+2,b+1,b+4); addTri(b+1,b+3,b+4); addTri(b+3,b+0,b+4);
        addTri(b+2,b+0,b+5); addTri(b+1,b+2,b+5); addTri(b+3,b+1,b+5); addTri(b+0,b+3,b+5);
    };
    auto addBone = [&](const glm::vec3& a, const glm::vec3& bpt, float r, int joint) {
        glm::vec3 axis = bpt - a;
        const float len = glm::length(axis);
        if (len < 1e-5f) return;
        axis /= len;
        glm::vec3 ref = (std::fabs(axis.y) < 0.9f) ? glm::vec3(0,1,0)
                                                   : glm::vec3(1,0,0);
        const glm::vec3 u = glm::normalize(glm::cross(axis, ref));
        const glm::vec3 v = glm::normalize(glm::cross(axis, u));
        const glm::vec3 mid = a + axis * (len * 0.18f);   // ring near the base
        const uint32_t base = (uint32_t)p.positions.size();
        addVert(a, joint);                                 // 0 tip-base
        addVert(mid + u * r, joint);                       // 1 ring
        addVert(mid + v * r, joint);                       // 2
        addVert(mid - u * r, joint);                       // 3
        addVert(mid - v * r, joint);                       // 4
        addVert(bpt, joint);                               // 5 tip-end
        // base pyramid (a → ring)
        addTri(base+0,base+1,base+2); addTri(base+0,base+2,base+3);
        addTri(base+0,base+3,base+4); addTri(base+0,base+4,base+1);
        // end pyramid (ring → b)
        addTri(base+5,base+2,base+1); addTri(base+5,base+3,base+2);
        addTri(base+5,base+4,base+3); addTri(base+5,base+1,base+4);
    };

    // Scale features to the rig size.
    const float jr = 0.022f, br = 0.014f;
    for (int i = 0; i < N; ++i) addOcta(bind[i], jr, i);
    for (int i = 0; i < N; ++i)
        if (parents[i] >= 0) addBone(bind[parents[i]], bind[i], br, parents[i]);

    // Recompute vertex normals from faces (smooth) for shading.
    for (size_t t = 0; t + 2 < p.indices.size(); t += 3) {
        const uint32_t i0 = p.indices[t], i1 = p.indices[t+1], i2 = p.indices[t+2];
        const glm::vec3 fn = glm::cross(p.positions[i1] - p.positions[i0],
                                        p.positions[i2] - p.positions[i0]);
        p.normals[i0] += fn; p.normals[i1] += fn; p.normals[i2] += fn;
    }
    for (auto& n : p.normals) { float l = glm::length(n); if (l > 1e-9f) n /= l; }

    engine::helper::MeshPreviewSection sec;
    sec.first_index = 0;
    sec.index_count = (uint32_t)p.indices.size();
    p.sections.push_back(sec);

    // Skin arrays (single influence each; no second set).
    preview_skin_joints_    = std::move(sj);
    preview_skin_weights_   = std::move(sw);
    preview_skin_closeness_ = std::move(sc);
    preview_skin_joints1_.clear();
    preview_skin_weights1_.clear();
    preview_skin_closeness1_.clear();
    preview_anim_base_ = p;                      // BIND-pose copy before move

    // Weight-debug bone list = all joints.
    preview_weight_bones_.clear();
    for (int i = 0; i < N; ++i) preview_weight_bones_.push_back(i);
    preview_weight_sel_ = 0;
    preview_bone_weight_scale_.assign(N, 1.0f);

    // Distance-debug weld cache unused here.
    preview_weld_id_.clear(); preview_weld_pos_.clear();
    preview_weld_adj_.clear(); preview_weld_tris_.clear();
    preview_joint_bind_world_.assign(N, glm::vec3(0.0f));
    for (int i = 0; i < N; ++i) preview_joint_bind_world_[i] = bind[i];
    preview_dist_sel_cached_ = -999;
    preview_surface_close_.clear();

    // Convert AnimClip → RwAnimClip (rotation per joint; root translation
    // becomes an ABSOLUTE hips translation = bind + offset).
    engine::helper::RwAnimClip rc;
    rc.name     = clip.name;
    rc.duration = clip.duration > 0.0f ? clip.duration : 1.0f;
    for (const auto& tr : clip.tracks) {
        if (tr.joint < 0 || tr.joint >= N) continue;
        engine::helper::RwAnimChannel ch;
        ch.node = tr.joint;
        ch.path = engine::helper::RwAnimPath::kRotation;
        ch.step = 0;
        for (const auto& k : tr.rot) {
            ch.times.push_back(k.time);
            ch.values.push_back(glm::vec4(k.rot.x, k.rot.y, k.rot.z, k.rot.w));
        }
        rc.channels.push_back(std::move(ch));
    }
    if (!clip.root_pos.empty()) {
        engine::helper::RwAnimChannel ch;
        ch.node = 0;                                  // hips (root)
        ch.path = engine::helper::RwAnimPath::kTranslation;
        ch.step = 0;
        for (const auto& k : clip.root_pos) {
            ch.times.push_back(k.time);
            ch.values.push_back(glm::vec4(bind[0] + k.v, 0.0f));
        }
        rc.channels.push_back(std::move(ch));
    }
    preview_clip_ = std::move(rc);

    // Stage + enable playback (mirrors buildRwGroupPreview ordering).
    preview_nav_ = PreviewNav::None;
    stagePreviewPayload(std::move(p));        // clears preview_anim_ready_
    finishAssetPreview(anim_path,
        "Anim: " + caption + "  (standard rig: " + std::to_string(N) +
        " joints)");
    dbg_preview_skinned_   = true;
    preview_show_skeleton_ = true;
    preview_anim_time_     = 0.0f;
    preview_anim_speed_    = 1.0f;            // a prior scrub may have left it 0
    preview_anim_playing_  = true;
    preview_anim_ready_    = true;            // AFTER stage (which cleared it)

    // Diagnostic: report the clip's actual content so a "no motion" preview is
    // easy to triage (0 tracks = generation/parse issue; tiny max-angle = the
    // model returned near-bind rotations).
    float tmin = 1e30f, tmax = -1e30f, maxdeg = 0.0f;
    for (const auto& tr : clip.tracks)
        for (const auto& k : tr.rot) {
            tmin = std::min(tmin, k.time); tmax = std::max(tmax, k.time);
            glm::quat q = glm::normalize(k.rot);
            const float ang = glm::degrees(2.0f *
                std::acos(glm::clamp(std::fabs(q.w), 0.0f, 1.0f)));
            maxdeg = std::max(maxdeg, ang);
        }
    if (clip.tracks.empty()) { tmin = 0.0f; tmax = 0.0f; }
    char dbg[256];
    std::snprintf(dbg, sizeof(dbg),
        "[anim] preview '%s': rig=%d joints, %d tracks, %d keys, %d channels, "
        "dur=%.2fs, key t=[%.2f..%.2f], maxRot=%.1f deg %s",
        clip.name.c_str(), N, (int)clip.tracks.size(), (int)clip.keyCount(),
        (int)preview_clip_.channels.size(), clip.duration, tmin, tmax, maxdeg,
        (clip.tracks.empty() ? "(EMPTY clip — generation produced no tracks)"
         : maxdeg < 2.0f ? "(near-bind rotations — model gave ~0 motion)" : ""));
    EditorLog::get().push(dbg);
}

// ── Sub-object names for a content asset (virtual-folder view) ──────────────
// Preferred source is the .rwmeta sidecar baked at import time (instant);
// assets that predate the bake (or were dropped into content/ by hand) get a
// one-time CPU parse via helper::listModelSubObjects.  Either way the result
// is cached for the session.
const std::vector<std::string>&
Menu::assetSubObjects(const std::string& path) {
    auto it = asset_children_cache_.find(path);
    if (it != asset_children_cache_.end()) return it->second;

    std::vector<std::string> names;
    bool baked = false;
    {
        std::ifstream meta(path + ".rwmeta");
        std::string line;
        while (meta && std::getline(meta, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "subobjects_baked=1") {
                baked = true;
            } else if (line.rfind("subobject=", 0) == 0) {
                names.push_back(line.substr(10));
            }
        }
    }
    if (!baked) {
        // No baked list — parse the model itself (may hitch once on a very
        // large file; the result is cached so it only ever happens once).
        names = engine::helper::listModelSubObjects(path);
    }

    auto res = asset_children_cache_.emplace(path, std::move(names));
    return res.first->second;
}

// ── Debug Display: shaded orbit preview of the clicked object ───────────────
// Rebuilds only when the selection key (object pointer + node index) changes.
// The mesh is pulled from the loaded DrawableData exactly like the selection
// mask, then rendered by the CPU rasteriser as one auto-framed orbit view.
void Menu::updateDebugDisplay(engine::game_object::DrawableObject* obj,
                              int node_idx) {
    if (obj == dbg_disp_obj_ && node_idx == dbg_disp_node_) return;
    dbg_disp_obj_  = obj;
    dbg_disp_node_ = node_idx;

    if (!obj || !obj->isReady()) { retireDebugPreview(); return; }
    dbg_preview_skinned_ = obj->isSkinned();

    // A wrapper restricted to one sub-object (.rwobj placement) previews
    // ONLY that node, regardless of how it was selected.
    int eff_node = node_idx;
    if (obj->onlyRenderSubObject() >= 0) {
        const auto& d2 = obj->getDrawableData();
        int k = 0;
        eff_node = -1;
        for (int ni = 0; ni < (int)d2.nodes_.size(); ++ni) {
            if (d2.nodes_[ni].mesh_idx_ < 0) continue;
            if (k == obj->onlyRenderSubObject()) { eff_node = ni; break; }
            ++k;
        }
        if (eff_node < 0) { retireDebugPreview(); return; }
    }

    // World-space triangle mesh from the selected node — or, for a group /
    // whole-object selection (node_idx < 0), every mesh node.
    plugins::auto_rig::TriangleMesh mesh;
    const auto& data = obj->getDrawableData();
    auto addNode = [&](int ni) {
        if (ni < 0 || ni >= (int)data.nodes_.size()) return;
        const auto& node = data.nodes_[ni];
        if (node.mesh_idx_ < 0 ||
            node.mesh_idx_ >= (int)data.meshes_.size()) return;
        const auto& m = data.meshes_[node.mesh_idx_];
        if (!m.vertex_position_) return;
        const auto& V = *m.vertex_position_;
        const glm::mat4& W = node.cached_matrix_;
        const uint32_t base = (uint32_t)mesh.positions.size();
        mesh.positions.reserve(mesh.positions.size() + V.size());
        for (const auto& v : V)
            mesh.positions.push_back(glm::vec3(W * glm::vec4(v, 1.0f)));
        for (const auto& prim : m.primitives_) {
            if (!prim.vertex_indices_) continue;
            for (int32_t idx : *prim.vertex_indices_)
                mesh.indices.push_back(base + (uint32_t)idx);
        }
    };
    if (eff_node >= 0) {
        addNode(eff_node);
    } else {
        for (int ni = 0; ni < (int)data.nodes_.size(); ++ni)
            if (data.nodes_[ni].mesh_idx_ >= 0) addNode(ni);
    }
    if (mesh.positions.empty() || mesh.indices.size() < 3) {
        retireDebugPreview();
        return;
    }
    dbg_asset_key_.clear();   // the scene selection owns the preview now
    installDebugPreview(mesh);
}

void Menu::retireDebugPreview() {
    preview_pending_ = false;
    preview_active_  = false;
    preview_payload_ = engine::helper::MeshPreviewPayload{};
    preview_nav_     = PreviewNav::None;
    preview_nav_path_.clear();
    preview_nav_index_ = -1;
    dbg_preview_skinned_ = false;
    preview_anim_ready_ = false;
    engine::helper::MeshPreview::setWeightDebug(false);
    engine::helper::MeshPreview::setSegmentDebug(false);
}

// Per-frame CPU skinning for the animated Debug Display preview.  Samples the
// baked clip (via the unit-tested ecs::AnimationSystem), composes the skeleton
// world matrices, linear-blend-skins the bind pose and re-stages the payload —
// so the existing static preview pass renders the animation.  Assumes an
// identity skinned-mesh node (the common rigged-character case).
void Menu::tickPreviewAnimation() {
    if (!preview_anim_ready_) return;
    const size_t nn = preview_node_parent_.size();
    const size_t vc = preview_anim_base_.positions.size();
    if (nn == 0 || vc == 0 || preview_skin_joints_.size() != vc) return;

    ImGuiIO& io = ImGui::GetIO();
    const float dur =
        preview_clip_.duration > 0.0f ? preview_clip_.duration : 1.0f;
    if (preview_anim_playing_) {
        preview_anim_time_ += io.DeltaTime * preview_anim_speed_;
        if (preview_anim_time_ > dur)
            preview_anim_time_ = std::fmod(preview_anim_time_, dur);
        if (preview_anim_time_ < 0.0f) preview_anim_time_ = 0.0f;
    }

    // Baked clip → ECS clip → sampled pose (per-node TRS overrides).
    engine::ecs::AnimationClip clip;
    clip.duration = preview_clip_.duration;
    for (const auto& ch : preview_clip_.channels) {
        engine::ecs::AnimChannel oc;
        oc.target_node = ch.node;
        oc.interp = ch.step ? engine::ecs::AnimInterp::kStep
                            : engine::ecs::AnimInterp::kLinear;
        oc.times = ch.times;
        if (ch.path == engine::helper::RwAnimPath::kRotation) {
            oc.path = engine::ecs::AnimPath::kRotation;
            for (const auto& v : ch.values)
                oc.quat.emplace_back(v.w, v.x, v.y, v.z);
        } else {
            oc.path = (ch.path == engine::helper::RwAnimPath::kScale)
                          ? engine::ecs::AnimPath::kScale
                          : engine::ecs::AnimPath::kTranslation;
            for (const auto& v : ch.values)
                oc.vec.emplace_back(v.x, v.y, v.z);
        }
        clip.channels.push_back(std::move(oc));
    }
    engine::ecs::AnimPose pose;
    engine::ecs::AnimationSystem::sample(clip, preview_anim_time_, pose);

    // Per-node local matrices: bind, overridden by the sampled channels under
    // RIGID-SKELETON constraints:
    //   • Bone lengths are preserved — only ROTATION is taken from the clip for
    //     non-root bones; their translation/scale stay at bind (a non-root
    //     translation or scale would change the bone's length).  Only the ROOT
    //     may translate (locomotion / bob).
    //   • Each bone's rotation is clamped to preview_node_rot_limit_ around its
    //     bind orientation, so a joint cannot hyper-rotate.
    std::vector<glm::mat4> local(nn);
    for (size_t i = 0; i < nn; ++i) {
        const glm::mat4& bl = preview_node_bind_local_[i];
        const glm::vec3 t_bind(bl[3]);
        const glm::vec3 c0(bl[0]), c1(bl[1]), c2(bl[2]);
        const glm::vec3 s_bind(glm::length(c0), glm::length(c1),
                               glm::length(c2));
        const glm::mat3 rm(
            s_bind.x > 1e-8f ? glm::vec3(c0 / s_bind.x) : glm::vec3(1, 0, 0),
            s_bind.y > 1e-8f ? glm::vec3(c1 / s_bind.y) : glm::vec3(0, 1, 0),
            s_bind.z > 1e-8f ? glm::vec3(c2 / s_bind.z) : glm::vec3(0, 0, 1));
        const glm::quat r_bind = glm::quat_cast(rm);

        glm::vec3 t = t_bind;          // keep bone length (bind translation)
        glm::quat r = r_bind;
        const bool is_root = preview_node_parent_[i] < 0;
        if (i < pose.nodes.size()) {
            const auto& nd = pose.nodes[i];
            if (nd.has_r) r = nd.rotation;
            if (is_root && nd.has_t) t = nd.translation;   // root only
            // scale is never taken from the clip — rigid bones
        }
        // Rotation constraint: cap the deviation from the bind orientation.
        const float lim = (i < preview_node_rot_limit_.size())
                              ? preview_node_rot_limit_[i] : 3.14159265f;
        glm::quat dev = glm::normalize(glm::inverse(r_bind) * r);
        if (dev.w < 0.0f) dev = -dev;                      // shortest arc
        const float ang = 2.0f * std::acos(glm::clamp(dev.w, -1.0f, 1.0f));
        if (ang > lim && ang > 1e-5f)
            dev = glm::slerp(glm::quat(1, 0, 0, 0), dev, lim / ang);
        r = glm::normalize(r_bind * dev);

        local[i] = glm::translate(glm::mat4(1.0f), t) *
                   glm::mat4_cast(r) *
                   glm::scale(glm::mat4(1.0f), s_bind);
    }

    // World matrices: resolve parents in repeated passes (any node order).
    std::vector<glm::mat4> world(nn, glm::mat4(1.0f));
    std::vector<char> done(nn, 0);
    bool progress = true;
    while (progress) {
        progress = false;
        for (size_t i = 0; i < nn; ++i) {
            if (done[i]) continue;
            const int par = preview_node_parent_[i];
            if (par < 0 || par >= (int)nn) {
                world[i] = local[i]; done[i] = 1; progress = true;
            } else if (done[par]) {
                world[i] = world[par] * local[i]; done[i] = 1; progress = true;
            }
        }
    }
    for (size_t i = 0; i < nn; ++i)
        if (!done[i]) world[i] = local[i];   // cycle guard

    // Joint world positions (node origins) for the skeleton debug overlay.
    // Same world space as the skinned mesh, so they project consistently.
    preview_joint_world_.resize(nn);
    for (size_t i = 0; i < nn; ++i)
        preview_joint_world_[i] = glm::vec3(world[i][3]);

    // Joint matrices.  The trailing preview_mesh_node_inv_ cancels the bake's
    // node_to_world baked into the source-world bind vertices, so the result
    // is the correct world-space skinned position (matches the engine's GPU
    // path: world = sum w * jointWorld * inverseBind * nodeLocalVertex).
    std::vector<glm::mat4> jm(nn);
    for (size_t i = 0; i < nn; ++i)
        jm[i] = world[i] * preview_node_invbind_[i] * preview_mesh_node_inv_;

    // Linear-blend skin the bind pose into a fresh payload.  The weight-debug
    // mode applies a per-bone influence multiplier (then renormalizes), so the
    // user can scrub a bone's weight up/down and watch the deformation change.
    const bool wdebug = preview_show_weights_;
    // Selected bone (for the weight render): pack its per-vertex weight into
    // uv.x and tell the preview pass to draw the flat weight colour instead of
    // the textured material.
    int wsel_node = -1;
    if (wdebug && !preview_weight_bones_.empty()) {
        int s = preview_weight_sel_;
        if (s < 0) s = 0;
        if (s >= (int)preview_weight_bones_.size())
            s = (int)preview_weight_bones_.size() - 1;
        wsel_node = preview_weight_bones_[s];
    }
    const bool sdebug = preview_show_segments_ && !wdebug;  // segment colouring
    const bool ddebug = preview_show_distance_ && !wdebug && !sdebug;
    // Weight-sum colouring: total skin weight per vertex (0=red,1=white,2=blue).
    const bool sumdebug =
        preview_show_weight_sum_ && !wdebug && !sdebug && !ddebug;
    engine::helper::MeshPreview::setWeightDebug(wdebug || ddebug);  // shared heatmap
    engine::helper::MeshPreview::setSegmentDebug(sdebug);
    engine::helper::MeshPreview::setWeightSumDebug(sumdebug);

    // Distance mode: show the auto-rig's BAKED closeness for the selected bone
    // — NO runtime geodesic recompute.  preview_skin_closeness_ holds, per
    // vertex, the closeness of its 4 influencing bones (parallel to joints); we
    // read the column matching the selected node.  Refreshed only on selection.
    if (ddebug && !preview_weight_bones_.empty() &&
        preview_skin_closeness_.size() == vc) {
        int sel = preview_weight_sel_;
        if (sel < 0) sel = 0;
        if (sel >= (int)preview_weight_bones_.size())
            sel = (int)preview_weight_bones_.size() - 1;
        if (sel != preview_dist_sel_cached_ ||
            preview_surface_close_.size() != vc) {
            preview_dist_sel_cached_ = sel;
            // Look up the SELECTED bone's baked closeness on every vertex:
            // find which of the vertex's 4 influencing joints is the selected
            // node and read its closeness (0 where the bone has no influence).
            const uint32_t selNode = (uint32_t)preview_weight_bones_[sel];
            const bool has8c =
                preview_skin_joints1_.size() == vc &&
                preview_skin_closeness1_.size() == vc;
            preview_surface_close_.assign(vc, 0.0f);
            for (size_t v = 0; v < vc; ++v) {
                const glm::uvec4 J = preview_skin_joints_[v];
                const glm::vec4  C = preview_skin_closeness_[v];
                float c = 0.0f;
                bool found = false;
                for (int k = 0; k < 4; ++k)
                    if (J[k] == selNode) { c = C[k]; found = true; break; }
                if (!found && has8c) {   // 8-bone debug: second set (4..7)
                    const glm::uvec4 J1 = preview_skin_joints1_[v];
                    const glm::vec4  C1 = preview_skin_closeness1_[v];
                    for (int k = 0; k < 4; ++k)
                        if (J1[k] == selNode) { c = C1[k]; break; }
                }
                preview_surface_close_[v] = c;
            }
        }
    }

    // 8-bone skinning debug: second influence set, parallel when present.
    const bool has8 = preview_skin_joints1_.size() == vc &&
                      preview_skin_weights1_.size() == vc;
    engine::helper::MeshPreviewPayload out = preview_anim_base_;
    for (size_t v = 0; v < vc; ++v) {
        const glm::uvec4 J = preview_skin_joints_[v];
        const glm::vec4  W = preview_skin_weights_[v];
        const glm::uvec4 J1 = has8 ? preview_skin_joints1_[v] : glm::uvec4(0u);
        const glm::vec4  W1 = has8 ? preview_skin_weights1_[v] : glm::vec4(0.0f);
        glm::mat4 m(0.0f);
        float wsum = 0.0f;
        auto blend = [&](const glm::uvec4& Jx, const glm::vec4& Wx) {
            for (int k = 0; k < 4; ++k) {
                float w = Wx[k];
                if (w <= 0.0f) continue;
                const uint32_t n = Jx[k];
                if (n >= nn) continue;
                if (n < preview_bone_weight_scale_.size())
                    w *= preview_bone_weight_scale_[n];   // tweakable influence
                if (w <= 0.0f) continue;
                m += jm[n] * w; wsum += w;
            }
        };
        blend(J, W);
        if (has8) blend(J1, W1);
        // Weight render: pack the selected bone's weight on this vertex into
        // uv.x (read by the preview shader's flat heat-map branch).
        if (wdebug && v < out.uvs.size()) {
            float wt = 0.0f;
            for (int k = 0; k < 4; ++k) {
                if ((int)J[k] == wsel_node) wt += W[k];
                if (has8 && (int)J1[k] == wsel_node) wt += W1[k];
            }
            out.uvs[v] = glm::vec2(wt, 0.0f);
        } else if (sdebug && v < out.uvs.size()) {
            // Segment render: pack the DOMINANT bone's node index into uv.x;
            // the shader maps it to a distinct hue.
            uint32_t dn = J[0]; float dw = -1.0f;
            for (int k = 0; k < 4; ++k) {
                if (W[k] > dw) { dw = W[k]; dn = J[k]; }
                if (has8 && W1[k] > dw) { dw = W1[k]; dn = J1[k]; }
            }
            out.uvs[v] = glm::vec2((float)dn, 0.0f);
        } else if (ddebug && v < out.uvs.size() &&
                   v < preview_surface_close_.size()) {
            // Distance render: closeness to the selected bone (heatmap branch).
            out.uvs[v] = glm::vec2(preview_surface_close_[v], 0.0f);
        } else if (sumdebug && v < out.uvs.size()) {
            // Weight-sum render: pack the RAW (pre-normalization) total of all
            // skin influences on this vertex into uv.x.  Read by the preview
            // shader's weight-sum branch (0=red, 1=white, 2=blue).  Uses the
            // uploaded weights directly (no per-bone scale) so it reflects the
            // baked skin data, surfacing verts that don't sum to ~1.
            float sum = W.x + W.y + W.z + W.w;
            if (has8) sum += W1.x + W1.y + W1.z + W1.w;
            out.uvs[v] = glm::vec2(sum, 0.0f);
        }
        const glm::vec3& bp = preview_anim_base_.positions[v];
        if (wsum <= 1e-6f) {
            // Unweighted vertex: leave it at the bind pose (do NOT collapse).
            out.positions[v] = bp;
            continue;
        }
        // Normalize by the weight total so vertices whose 4 weights don't sum
        // to exactly 1 aren't pulled toward the origin (the shard/spike
        // artifact).  Dividing by wsum is the correct weighted blend.
        const float inv = 1.0f / wsum;
        out.positions[v] = glm::vec3(m * glm::vec4(bp, 1.0f)) * inv;
        if (v < preview_anim_base_.normals.size()) {
            glm::vec3 n = (glm::mat3(m) * preview_anim_base_.normals[v]) * inv;
            const float l = glm::length(n);
            out.normals[v] =
                (l > 1e-6f) ? (n / l) : preview_anim_base_.normals[v];
        }
    }

    // Re-stage directly (bypass stagePreviewPayload so the ready flag stays).
    preview_payload_ = std::move(out);
    preview_pending_ = true;
    preview_active_  = true;
}

// Stage a ready payload for the GPU PBR preview pass: the application
// consumes it via takeDebugPreviewPayload() and records
// helper::MeshPreview's offscreen render before the ImGui pass samples the
// persistent preview target.
void Menu::stagePreviewPayload(engine::helper::MeshPreviewPayload&& payload) {
    if (payload.positions.empty() || payload.indices.size() < 3) {
        retireDebugPreview();
        return;
    }
    preview_payload_ = std::move(payload);
    preview_pending_ = true;
    preview_active_  = true;
    // Default to a STATIC preview; buildRwGroupPreview re-enables animation
    // after staging when the group is fully skinned.  tickPreviewAnimation
    // bypasses this path (writes preview_payload_ directly) so it doesn't
    // clear its own flag.
    preview_anim_ready_ = false;
    engine::helper::MeshPreview::setWeightDebug(false);   // back to textured
    engine::helper::MeshPreview::setSegmentDebug(false);
    // NOTE: tab focus is requested by finishAssetPreview (content previews
    // only) — scene-selection previews must NOT yank the Outliner tab away
    // while the user is clicking rows in it.
}

// TriangleMesh adapter (scene-object selections + any path that still
// produces an auto_rig mesh).  Carries UVs + the base-colour texture when
// the mesh has them; untextured meshes shade as neutral PBR.
bool Menu::stageAutoRigSkinLayer() {
    if (!plugin_manager_) return false;
    plugins::auto_rig::AutoRigPlugin* ar = nullptr;
    for (auto& p : plugin_manager_->plugins())
        if (auto* a = dynamic_cast<plugins::auto_rig::AutoRigPlugin*>(p.get())) {
            ar = a; break;
        }
    if (!ar) return false;

    const plugins::auto_rig::TriangleMesh& mesh = ar->getMesh();
    const std::vector<uint8_t>&            skin = ar->getBaseSkinVerts();
    if (mesh.positions.empty() || skin.size() != mesh.positions.size()) {
        dbg_skin_status_ = "no baked skin classification "
                           "(rebuild, then Run Auto-Rig / Bake)";
        return false;   // nothing baked yet
    }
    size_t nskin_v = 0;
    for (uint8_t s : skin) nskin_v += s;

    engine::helper::MeshPreviewPayload p;
    p.positions = mesh.positions;
    // Normals: reuse the mesh's if present, else accumulate face normals.
    if (mesh.normals.size() == mesh.positions.size()) {
        p.normals = mesh.normals;
    } else {
        p.normals.assign(mesh.positions.size(), glm::vec3(0.0f));
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            const uint32_t a = mesh.indices[t], b = mesh.indices[t+1],
                           c = mesh.indices[t+2];
            const glm::vec3 fn = glm::cross(mesh.positions[b] - mesh.positions[a],
                                            mesh.positions[c] - mesh.positions[a]);
            p.normals[a] += fn; p.normals[b] += fn; p.normals[c] += fn;
        }
        for (auto& n : p.normals) { float l = glm::length(n); if (l > 1e-12f) n /= l; }
    }
    // Keep ONLY triangles whose 3 vertices are all base skin.
    p.indices.reserve(mesh.indices.size());
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t i0 = mesh.indices[t], i1 = mesh.indices[t+1],
                       i2 = mesh.indices[t+2];
        if (skin[i0] && skin[i1] && skin[i2]) {
            p.indices.push_back(i0); p.indices.push_back(i1); p.indices.push_back(i2);
        }
    }
    const size_t total_tris = mesh.indices.size() / 3;
    const size_t kept_tris  = p.indices.size() / 3;
    if (p.indices.empty()) {
        dbg_skin_status_ = "classifier marked 0 tris as skin (nothing to show)";
        return false;
    }

    engine::helper::MeshPreviewSection sec;
    sec.first_index = 0;
    sec.index_count = (uint32_t)p.indices.size();
    p.sections.push_back(sec);

    engine::helper::MeshPreview::setWeightDebug(false);
    engine::helper::MeshPreview::setSegmentDebug(false);
    stagePreviewPayload(std::move(p));

    char cap[176];
    std::snprintf(cap, sizeof(cap),
        "base-skin layer: %zu / %zu tris kept  (%zu / %zu verts are skin)",
        kept_tris, total_tris, nskin_v, skin.size());
    dbg_skin_status_  = cap;
    dbg_disp_caption_ = std::string("Auto-Rig ") + cap;
    return true;
}

void Menu::installDebugPreview(plugins::auto_rig::TriangleMesh& mesh) {
    if (mesh.positions.empty() || mesh.indices.size() < 3) {
        retireDebugPreview();
        return;
    }
    mesh.recomputeBounds();
    if (mesh.normals.size() != mesh.positions.size()) {
        mesh.recomputeNormals();
    }

    // Scene-selection previews aren't arrow-navigable (no sibling list).
    preview_nav_ = PreviewNav::None;

    engine::helper::MeshPreviewPayload p;
    p.positions = std::move(mesh.positions);
    p.normals   = std::move(mesh.normals);
    p.indices   = std::move(mesh.indices);

    engine::helper::MeshPreviewSection sec;
    sec.first_index = 0;
    sec.index_count = (uint32_t)p.indices.size();
    if (mesh.texcoords.size() == p.positions.size() &&
        !mesh.base_color_texture.empty() &&
        (mesh.base_color_texture.channels == 3 ||
         mesh.base_color_texture.channels == 4)) {
        p.uvs = std::move(mesh.texcoords);
        const auto& t = mesh.base_color_texture;
        engine::helper::MeshPreviewTexture mt;
        mt.w = t.width;
        mt.h = t.height;
        const size_t n = (size_t)t.width * t.height;
        mt.rgba.resize(n * 4);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* sp = t.pixels.data() + i * t.channels;
            uint8_t* dp = mt.rgba.data() + i * 4;
            dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
            dp[3] = (t.channels == 4) ? sp[3] : 255;
        }
        p.textures.push_back(std::move(mt));
        sec.tex_index = 0;
    }
    p.sections.push_back(sec);
    stagePreviewPayload(std::move(p));
}

// Content Browser → Debug Display: preview an asset file (sub_index < 0) or
// one of its sub-objects.  CPU-only parse extracts geometry, UVs, the
// base-colour texture and the PBR factors; the GPU pass does the shading.
void Menu::buildAssetPreview(const std::string& path, int sub_index,
                             const std::string& caption) {
    const std::string key = path + "#" + std::to_string(sub_index);
    if (key == dbg_asset_key_) return;   // already showing

    engine::helper::ModelPreviewData data;
    if (!engine::helper::loadModelPreviewData(path, sub_index, data)) {
        EditorLog::get().push(
            "[preview] could not load geometry from: " + path);
        return;
    }
    // This path previews a raw source model FILE (.gltf/.glb/.fbx) that lives
    // in content; that file IS the asset's own data, not a separate original.
    // loadModelPreviewData only loads static geometry (no joints), so read the
    // skin flag from the same file.  (Bake-only character imports never reach
    // here — they preview via buildRwGroupPreview / buildRwGeoPreview, which
    // read the skin blobs straight out of the baked .rwgeo.)
    dbg_preview_skinned_ = engine::helper::modelHasSkin(path);
    engine::helper::MeshPreviewPayload p;
    p.positions = std::move(data.positions);
    p.normals   = std::move(data.normals);
    p.uvs       = std::move(data.uvs);
    p.indices   = std::move(data.indices);
    for (auto& t : data.textures) {
        engine::helper::MeshPreviewTexture mt;
        mt.rgba = std::move(t.rgba);
        mt.w = t.w;
        mt.h = t.h;
        p.textures.push_back(std::move(mt));
    }
    for (const auto& s : data.sections) {
        engine::helper::MeshPreviewSection ms;
        ms.first_index = s.first_index;
        ms.index_count = s.index_count;
        ms.base_color  = s.base_color;
        ms.metallic    = s.metallic;
        ms.roughness   = s.roughness;
        ms.tex_index   = s.tex_index;
        ms.nrm_index   = s.nrm_index;
        ms.mr_index    = s.mr_index;
        p.sections.push_back(ms);
    }
    stagePreviewPayload(std::move(p));
    finishAssetPreview(key, caption);

    // Arrow-key navigation: step through this model's sub-objects.
    if (sub_index >= 0) {
        preview_nav_       = PreviewNav::SubObjects;
        preview_nav_path_  = path;
        preview_nav_index_ = sub_index;
    } else {
        preview_nav_ = PreviewNav::None;
    }
}

// .rwobj object preview — prefers the render-ready bake (.rwgeo + .rwtex
// under the group folder, written at import) and only re-parses the source
// model when no baked data exists.
// ── Rendering ▸ Render Debug submenu content ───────────────────────────────
// Drives DEBUG_RENDER_MODE_* (packed into camera_info.input_features by
// application.cpp).  Both base.frag and cluster_bindless.frag read and
// dispatch on this value; mode 0 = the normal shaded path.
void Menu::drawRenderDebugMenuContent() {
    // Modes 0..12 are the original PBR / G-buffer debug overlays baked
    // into base.frag and cluster_bindless.frag.  Mode 13 is the
    // MeshCategory solid-colour overlay (AI-backed material classifier);
    // its value MUST match DEBUG_RENDER_MODE_CATEGORY in
    // global_definition.glsl.h.
    struct RenderDebugItem {
        int         mode_id;
        const char* label;
    };
    static const RenderDebugItem kRenderDebugItems[] = {
        {  0, "0: Final shaded"            },
        {  1, "1: Albedo (baseColor)"      },
        {  2, "2: Normal (perturbed)"      },
        {  3, "3: Diffuse term"            },
        {  4, "4: Specular term"           },
        {  5, "5: Shadow factor"           },
        {  6, "6: Roughness (perceptual)"  },
        {  7, "7: Metallic"                },
        {  8, "8: Geometric normal"        },
        {  9, "9: Translucent (alpha mode)"},
        { 10, "10: Velocity (NDC delta x50)"},
        { 11, "11: SSAO (raw AO factor)"   },
        { 12, "12: Hi-Z pyramid (mip)"     },
        { 13, "13: Mesh category (solid)"  },
        { 14, "14: Object ID (per mesh)"   },
        { 15, "15: Weight sum (skin)"      },
    };
    for (int i = 0; i < IM_ARRAYSIZE(kRenderDebugItems); ++i) {
        const auto& item = kRenderDebugItems[i];
        bool selected = (debug_render_mode_ == item.mode_id);
        if (ImGui::MenuItem(item.label, NULL, selected)) {
            debug_render_mode_ = item.mode_id;
        }
    }

    // ── Skeleton view selector ──────────────────────────────────────
    // Tri-state: character only (default), bones + character (alignment
    // check), or bones only.  Application reads this each frame and
    // toggles DrawableObject::setVisible(...) accordingly.
    ImGui::Separator();
    ImGui::TextDisabled("Skeleton view");
    {
        struct SkelItem {
            SkeletonDebugMode mode;
            const char*       label;
        };
        static const SkelItem kSkelItems[] = {
            { SkeletonDebugMode::CharacterOnly,
              "Character only (default)"                },
            { SkeletonDebugMode::BoneWithCharacter,
              "Bones + character (alignment check)"     },
            { SkeletonDebugMode::BoneOnly,
              "Bones only (skeleton in isolation)"      },
        };
        for (int i = 0; i < IM_ARRAYSIZE(kSkelItems); ++i) {
            const auto& it = kSkelItems[i];
            bool sel = (skeleton_debug_mode_ == it.mode);
            if (ImGui::MenuItem(it.label, NULL, sel)) {
                skeleton_debug_mode_ = it.mode;
            }
        }
    }

    // ── Colour legend for the Mesh-Category mode ────────────────────
    // Swatch RGBs MUST match categoryColor() in collision_debug.frag and
    // the DEBUG_RENDER_MODE_CATEGORY branch in cluster_bindless.frag.
    // Only shown while mode 13 is active.
    if (debug_render_mode_ == 13) {
        ImGui::Separator();
        ImGui::TextDisabled("Mesh-Category colour key");
        // The G-buffer doesn't carry per-material flag bits, so the
        // deferred resolve can't read the category — application.cpp
        // force-flips to forward rendering while this mode is active.
        ImGui::TextDisabled(
            "(forces Forward rendering — Deferred drops the "
            "category bits in the G-buffer)");
        struct CatLegend {
            const char* name;
            float       r, g, b;
        };
        static const CatLegend kLegend[] = {
            {"Floor",      0.20f, 0.75f, 0.30f},
            {"Wall",       0.80f, 0.20f, 0.20f},
            {"Door",       0.95f, 0.75f, 0.10f},
            {"Object",     0.55f, 0.25f, 0.80f},
            {"Glass",      0.40f, 0.85f, 0.95f},
            {"Ceiling",    0.25f, 0.45f, 0.85f},
            {"Stairs",     0.95f, 0.50f, 0.10f},
            {"Vegetation", 0.55f, 0.65f, 0.20f},
            {"Elevator",   0.95f, 0.20f, 0.65f},
            {"Ladder",     0.75f, 0.55f, 0.35f},
            {"Unknown",    0.55f, 0.55f, 0.55f},
        };
        for (const auto& row : kLegend) {
            ImGui::ColorButton(
                row.name,
                ImVec4(row.r, row.g, row.b, 1.0f),
                ImGuiColorEditFlags_NoTooltip |
                    ImGuiColorEditFlags_NoPicker |
                    ImGuiColorEditFlags_NoLabel,
                ImVec2(16, 16));
            ImGui::SameLine();
            ImGui::TextUnformatted(row.name);
        }
        // Per-name inspector window toggle (LLM material/object →
        // category verdicts) — enabled once the classifier snapshot
        // has been pushed from application.cpp.
        ImGui::Separator();
        if (mesh_category_snapshot_valid_) {
            if (ImGui::MenuItem(
                    "Open Category Inspector",
                    NULL,
                    show_mesh_category_inspector_)) {
                show_mesh_category_inspector_ =
                    !show_mesh_category_inspector_;
            }
        } else {
            ImGui::TextDisabled(
                "Category Inspector (waiting for LLM)");
        }
        ImGui::Separator();
    }
    // Hi-Z mip selector — only meaningful when DEBUG_RENDER_MODE_HIZ is
    // the active mode (4 bits of mip selection in input_features; the
    // application clamps against the actual pyramid mip count).
    if (debug_render_mode_ == 12) {
        ImGui::Separator();
        ImGui::SliderInt("Hi-Z mip", &hiz_debug_mip_, 0, 15);
    }
    ImGui::Separator();

    // ── Forward vs Deferred rendering toggle ────────────────────────
    // The application reads isDeferredRendering() each frame in
    // drawScene to route the cluster opaque pass through the G-buffer +
    // compute resolve (Deferred) or the legacy single-pass bindless
    // pipeline (Forward).  Non-cluster passes are unaffected.
    if (ImGui::MenuItem(
            "Pipeline: Deferred", NULL, deferred_rendering_)) {
        deferred_rendering_ = true;
    }
    if (ImGui::MenuItem(
            "Pipeline: Forward",  NULL, !deferred_rendering_)) {
        deferred_rendering_ = false;
    }
    ImGui::Separator();
    // Camera-positioned dynamic reflection cubemap face viewer.
    if (ImGui::MenuItem(
            "Dynamic Cubemap Viewer", NULL, show_dynamic_cube_debug_)) {
        show_dynamic_cube_debug_ = !show_dynamic_cube_debug_;
    }
    // Hi-Z pyramid (last-frame depth) mip-strip viewer.
    if (ImGui::MenuItem(
            "Hi-Z Pyramid Viewer", NULL, show_hiz_debug_)) {
        show_hiz_debug_ = !show_hiz_debug_;
    }
    // VT pool viewer — the four 4096² layer pool textures.
    if (ImGui::MenuItem(
            "VT Pool Viewer", NULL, show_vt_pool_debug_)) {
        show_vt_pool_debug_ = !show_vt_pool_debug_;
    }
    // In-scene probe icospheres (pending probes draw solid red).
    if (ImGui::MenuItem(
            "Show Probes (in scene)", NULL, show_probe_debug_)) {
        show_probe_debug_ = !show_probe_debug_;
    }

    // ── Glass / translucent rendering mode ──────────────────────────
    // Pure storage on the cluster renderer; the application's glass
    // dispatch reads this and calls drawTranslucentForward
    // (ALPHA_BLEND) or drawTranslucentOit (WBOIT).
    if (cluster_renderer_) {
        ImGui::Separator();
        using TMode = engine::scene_rendering::
            ClusterRenderer::TranslucentMode;
        TMode cur_mode = cluster_renderer_->getTranslucentMode();
        if (ImGui::MenuItem(
                "Glass: Alpha Blend (forward)", NULL,
                cur_mode == TMode::ALPHA_BLEND)) {
            cluster_renderer_->setTranslucentMode(
                TMode::ALPHA_BLEND);
        }
        if (ImGui::MenuItem(
                "Glass: WBOIT (order-independent)", NULL,
                cur_mode == TMode::WBOIT)) {
            cluster_renderer_->setTranslucentMode(
                TMode::WBOIT);
        }
    }
}

// ── Whole-group preview ────────────────────────────────────────────────────
// Merge every baked objects/*.rwgeo of an import group into ONE preview
// payload.  Each .rwgeo loads in source-world coordinates (its header
// matrix is re-applied by loadRwGeo), so the assembly reconstructs the
// authored layout; textures are dedup'd across objects by file path.
// (Preview-side geodesic weight de-bleed removed — the auto-rig pass now does
// geodesic surface weighting at rig time, so the asset weights are clean.)

// Synthesize a looping WALK clip for a rig that ships without animation
// (e.g. scene-skinned: a 19-bone humanoid skeleton, no clips).  Bones are
// matched by name; channels are per-bone LOCAL rotations.  The bind pose has
// identity bone rotations and is Y-up with limbs hanging down, so each bone's
// local Z is the medial-lateral axis — rotating about it swings the limb
// fore/aft (the walk swing).  Returns an empty clip for rigs we don't
// recognise (caller then falls back to a static preview).
static engine::helper::RwAnimClip makeProceduralWalk(
    const std::vector<engine::helper::RwHierNode>& hier) {
    engine::helper::RwAnimClip clip;

    std::unordered_map<std::string, int> idx;
    for (size_t i = 0; i < hier.size(); ++i) {
        std::string n = hier[i].name;
        for (auto& c : n) c = (char)std::tolower((unsigned char)c);
        idx[n] = (int)i;
    }
    auto find = [&](const char* nm) -> int {
        auto it = idx.find(nm);
        return it == idx.end() ? -1 : it->second;
    };
    const int l_thigh = find("left_upper_leg"),  r_thigh = find("right_upper_leg");
    const int l_shin  = find("left_lower_leg"),  r_shin  = find("right_lower_leg");
    const int l_arm   = find("left_upper_arm"),  r_arm   = find("right_upper_arm");
    const int l_fore  = find("left_lower_arm"),  r_fore  = find("right_lower_arm");
    const int spine   = find("spine"),           hips    = find("hips");
    if (l_thigh < 0 || r_thigh < 0) return clip;   // not a rig we know

    clip.name = "walk";
    const float kStride = 1.0f;    // one stride per second
    clip.duration = kStride;
    const int   kSteps = 24;       // samples per cycle (+1 closing key)
    const float kPI = 3.14159265f;
    const glm::vec3 axisZ(0, 0, 1), axisY(0, 1, 0);
    // Swing amplitudes (radians).
    const float A_thigh = 0.55f, A_knee = 0.9f, A_arm = 0.45f,
                A_elbow = 0.25f, A_spin = 0.08f, BOB = 0.025f;

    auto quat = [](const glm::vec3& ax, float a) -> glm::vec4 {
        const float s = std::sin(a * 0.5f);
        return glm::vec4(ax.x * s, ax.y * s, ax.z * s, std::cos(a * 0.5f));
    };
    auto addRot = [&](int node, const glm::vec3& axis, auto angFn) {
        if (node < 0) return;
        engine::helper::RwAnimChannel ch;
        ch.node = node;
        ch.path = engine::helper::RwAnimPath::kRotation;
        ch.step = 0;
        for (int k = 0; k <= kSteps; ++k) {
            const float p = 2.0f * kPI * (float)k / (float)kSteps;
            ch.times.push_back(kStride * (float)k / (float)kSteps);
            ch.values.push_back(quat(axis, angFn(p)));
        }
        clip.channels.push_back(std::move(ch));
    };

    // Thighs swing fore/aft (legs in opposite phase).
    addRot(l_thigh, axisZ, [&](float p) { return A_thigh * std::sin(p); });
    addRot(r_thigh, axisZ, [&](float p) { return A_thigh * std::sin(p + kPI); });
    // Knees flex one-directionally, peaking on the back/lift part of the step.
    addRot(l_shin, axisZ, [&](float p) {
        return -A_knee * 0.5f * (1.0f - std::cos(p + 1.2f)); });
    addRot(r_shin, axisZ, [&](float p) {
        return -A_knee * 0.5f * (1.0f - std::cos(p + kPI + 1.2f)); });
    // Arms swing opposite the same-side leg.
    addRot(l_arm, axisZ, [&](float p) { return A_arm * std::sin(p + kPI); });
    addRot(r_arm, axisZ, [&](float p) { return A_arm * std::sin(p); });
    // Slight constant elbow bend.
    addRot(l_fore, axisZ, [&](float) { return -A_elbow; });
    addRot(r_fore, axisZ, [&](float) { return -A_elbow; });
    // Subtle torso counter-twist about the up axis.
    addRot(spine, axisY, [&](float p) { return A_spin * std::sin(p); });

    // Hips vertical bob (twice per stride) via a translation channel.
    if (hips >= 0) {
        const glm::vec3 base(hier[hips].local[3]);   // bind translation
        engine::helper::RwAnimChannel ch;
        ch.node = hips;
        ch.path = engine::helper::RwAnimPath::kTranslation;
        ch.step = 0;
        for (int k = 0; k <= kSteps; ++k) {
            const float p = 2.0f * kPI * (float)k / (float)kSteps;
            glm::vec3 tr = base;
            tr.y += BOB * (-std::cos(2.0f * p));
            ch.times.push_back(kStride * (float)k / (float)kSteps);
            ch.values.push_back(glm::vec4(tr, 0.0f));
        }
        clip.channels.push_back(std::move(ch));
    }
    return clip;
}

void Menu::buildRwGroupPreview(const std::string& group_dir,
                               const std::string& display_name) {
    const std::string key = group_dir + "#group";
    if (key == dbg_asset_key_) return;   // already showing
    // Detected from the baked raw data below: a character import is a group
    // whose objects/*.rwgeo carry skin blobs (joints/skin table).  We do NOT
    // consult the source model — the imported .rwgeo is the source of truth.
    dbg_preview_skinned_ = false;

    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<std::string> geos;
    for (auto& e : fs::directory_iterator(
             fs::path(group_dir) / "objects", ec)) {
        if (e.path().extension() == ".rwgeo")
            geos.push_back(e.path().string());
    }
    std::sort(geos.begin(), geos.end());
    if (geos.empty()) {
        EditorLog::get().push(
            "[preview] no baked objects in group: " + group_dir);
        return;
    }

    // Vertex budget: a whole Bistro-sized group merges to millions of
    // vertices; cap the assembly so the preview upload stays sane.
    constexpr size_t kMaxGroupVerts = 3'000'000;

    engine::helper::MeshPreviewPayload p;
    std::unordered_map<std::string, int> texmap;   // tex path → payload idx
    size_t skipped = 0;
    // Animated-preview capture (used only when the whole group is skinned).
    std::vector<glm::uvec4>            sj;            // per vtx: 4 node indices
    std::vector<glm::vec4>             sw;            // per vtx: 4 weights
    std::vector<glm::vec4>             sc;            // per vtx: 4 baked closeness
    // 8-bone skinning debug: second influence set (4..7), parallel to sj/sw/sc.
    std::vector<glm::uvec4>            sj1;
    std::vector<glm::vec4>             sw1;
    std::vector<glm::vec4>             sc1;
    std::unordered_map<int, glm::mat4> invbind_map;   // skeleton node → inv bind
    bool all_skinned = true;
    int  first_skinned_ordinal = -1;   // owner mesh node (bind-space cancel)
    for (const auto& geo : geos) {
        if (p.positions.size() >= kMaxGroupVerts) { ++skipped; continue; }
        // Skip objects disabled in the Content Browser — they're left out of
        // the assembled Debug Display preview.  Check the persistent
        // sidecar too (the in-memory set is only seeded while browsing).
        if (content_disabled_.find(geo) != content_disabled_.end()) continue;
        {
            std::error_code dec;
            if (fs::exists(geo + ".disabled", dec)) continue;
        }
        engine::helper::ModelPreviewData md;
        std::vector<std::string> tpaths;
        if (!engine::helper::loadRwGeo(geo, md, &tpaths) ||
            md.positions.empty() || md.indices.empty()) {
            continue;
        }
        // Raw-data skin detection: any baked object carrying a skin table /
        // per-vertex joints makes the whole group a skeleton mesh.
        if (!md.skin_joint_nodes.empty() || !md.joints.empty())
            dbg_preview_skinned_ = true;
        const uint32_t vbase = (uint32_t)p.positions.size();
        const uint32_t ibase = (uint32_t)p.indices.size();
        p.positions.insert(p.positions.end(),
                           md.positions.begin(), md.positions.end());
        p.normals.insert(p.normals.end(),
                         md.normals.begin(), md.normals.end());
        // Keep the uv stream parallel to positions (zero-fill objects
        // that have none) so a textured object later in the merge still
        // maps correctly.
        for (size_t i = 0; i < md.positions.size(); ++i) {
            p.uvs.push_back(i < md.uvs.size() ? md.uvs[i]
                                              : glm::vec2(0.0f));
        }
        for (uint32_t ix : md.indices) p.indices.push_back(vbase + ix);

        // ── Animated-preview skin capture (parallel to p.positions) ──
        // Remap each vertex's object-local cluster indices to GLOBAL skeleton
        // node indices (skin_joint_nodes were baked as hierarchy.rwhier
        // indices) so one node-indexed joint table covers the whole group.
        {
            const bool obj_skinned =
                md.joints.size()  == md.positions.size() &&
                md.weights.size() == md.positions.size() &&
                !md.skin_joint_nodes.empty();
            if (!obj_skinned) all_skinned = false;
            if (obj_skinned && first_skinned_ordinal < 0) {
                // Leading digits of "NNN_name.rwgeo" == mesh_ordinal.
                const std::string fn = fs::path(geo).filename().string();
                int ord = 0; bool any = false;
                for (char c : fn) {
                    if (c >= '0' && c <= '9') { ord = ord * 10 + (c - '0');
                                                any = true; }
                    else break;
                }
                if (any) first_skinned_ordinal = ord;
            }
            const bool obj_close =
                obj_skinned && md.closeness.size() == md.positions.size();
            // 8-bone skinning debug: second influence set (4..7).
            const bool obj_skin8 = obj_skinned &&
                md.joints1.size()  == md.positions.size() &&
                md.weights1.size() == md.positions.size();
            const bool obj_close8 = obj_skin8 &&
                md.closeness1.size() == md.positions.size();
            for (size_t i = 0; i < md.positions.size(); ++i) {
                glm::uvec4 J(0u), J1(0u);
                glm::vec4  W(0.0f), W1(0.0f);
                glm::vec4  C(0.0f), C1(0.0f);
                if (obj_skinned) {
                    const glm::u16vec4 lj = md.joints[i];
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t local = lj[k];
                        J[k] = (local < md.skin_joint_nodes.size())
                            ? (uint32_t)md.skin_joint_nodes[local] : 0u;
                    }
                    W = md.weights[i];
                    if (obj_close) C = md.closeness[i];
                }
                if (obj_skin8) {
                    const glm::u16vec4 lj1 = md.joints1[i];
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t local = lj1[k];
                        J1[k] = (local < md.skin_joint_nodes.size())
                            ? (uint32_t)md.skin_joint_nodes[local] : 0u;
                    }
                    W1 = md.weights1[i];
                    if (obj_close8) C1 = md.closeness1[i];
                }
                sj.push_back(J);
                sw.push_back(W);
                sc.push_back(C);
                sj1.push_back(J1);
                sw1.push_back(W1);
                sc1.push_back(C1);
            }
            if (obj_skinned)
                for (size_t j = 0; j < md.skin_joint_nodes.size() &&
                                   j < md.skin_inverse_bind.size(); ++j)
                    invbind_map[md.skin_joint_nodes[j]] =
                        md.skin_inverse_bind[j];
        }

        // Texture dedup across objects (tpaths is parallel to
        // md.textures — covers albedo/normal/mr refs alike).
        std::vector<int> remap(md.textures.size(), -1);
        for (size_t t = 0; t < md.textures.size() &&
                           t < tpaths.size(); ++t) {
            auto it = texmap.find(tpaths[t]);
            if (it != texmap.end()) {
                remap[t] = it->second;
                continue;
            }
            engine::helper::MeshPreviewTexture mt;
            mt.rgba = std::move(md.textures[t].rgba);
            mt.w = md.textures[t].w;
            mt.h = md.textures[t].h;
            remap[t] = (int)p.textures.size();
            p.textures.push_back(std::move(mt));
            texmap.emplace(tpaths[t], remap[t]);
        }
        auto remap_idx = [&remap](int idx) -> int {
            return (idx >= 0 && idx < (int)remap.size()) ? remap[idx] : -1;
        };
        for (const auto& s : md.sections) {
            engine::helper::MeshPreviewSection ms;
            ms.first_index = ibase + s.first_index;
            ms.index_count = s.index_count;
            ms.base_color  = s.base_color;
            ms.metallic    = s.metallic;
            ms.roughness   = s.roughness;
            ms.tex_index   = remap_idx(s.tex_index);
            ms.nrm_index   = remap_idx(s.nrm_index);
            ms.mr_index    = remap_idx(s.mr_index);
            p.sections.push_back(ms);
        }
    }
    if (p.positions.empty() || p.indices.size() < 3) {
        EditorLog::get().push(
            "[preview] group has no readable baked geometry: " +
            group_dir);
        return;
    }
    if (skipped > 0) {
        EditorLog::get().push(
            "[preview] group preview capped — " +
            std::to_string(skipped) + " object(s) omitted");
    }
    // ── Animated preview setup (fully-skinned group + skeleton + clip) ──
    bool want_anim = dbg_preview_skinned_ && all_skinned &&
                     sj.size() == p.positions.size();
    std::vector<engine::helper::RwHierNode> hier;
    std::vector<engine::helper::RwAnimClip> clips;
    engine::helper::RwAnimClip chosen_clip;
    if (want_anim) {
        namespace fs = std::filesystem;
        want_anim = engine::helper::loadRwHier(
            (fs::path(group_dir) / "hierarchy.rwhier").string(), hier) &&
            !hier.empty();
    }
    if (want_anim) {
        namespace fs = std::filesystem;
        // Prefer a baked clip; otherwise synthesize a walk for rigs that
        // ship without animation (e.g. scene-skinned: bones, no clips).
        if (engine::helper::loadRwAnim(
                (fs::path(group_dir) / "animation.rwanim").string(), clips) &&
            !clips.empty()) {
            chosen_clip = clips[0];
        } else {
            chosen_clip = makeProceduralWalk(hier);
        }
        if (chosen_clip.channels.empty()) want_anim = false;
    }
    if (want_anim) {
        preview_anim_base_    = p;            // copy BIND pose before the move

        // Bind-pose joint world matrices (reused by the de-bleed below AND the
        // mesh-node-inverse further down).
        std::vector<glm::mat4> nwb(hier.size(), glm::mat4(1.0f));
        {
            std::vector<char> dn(hier.size(), 0);
            bool prog = true;
            while (prog) {
                prog = false;
                for (size_t i = 0; i < hier.size(); ++i) {
                    if (dn[i]) continue;
                    const int par = hier[i].parent;
                    if (par < 0 || par >= (int)hier.size()) {
                        nwb[i] = hier[i].local; dn[i] = 1; prog = true;
                    } else if (dn[par]) {
                        nwb[i] = nwb[par] * hier[i].local; dn[i] = 1; prog = true;
                    }
                }
            }
        }

        // Weights come straight from the asset now (the auto-rig pass does the
        // geodesic surface weighting at rig time — no preview-side de-bleed).
        preview_skin_joints_    = std::move(sj);
        preview_skin_weights_   = std::move(sw);
        preview_skin_closeness_ = std::move(sc);
        // 8-bone debug second set (all-zero weights for 4-bone assets — a
        // zero weight contributes nothing to the CPU blend).
        preview_skin_joints1_    = std::move(sj1);
        preview_skin_weights1_   = std::move(sw1);
        preview_skin_closeness1_ = std::move(sc1);
        preview_node_parent_.assign(hier.size(), -1);
        preview_node_name_.assign(hier.size(), std::string());
        preview_node_bind_local_.assign(hier.size(), glm::mat4(1.0f));
        preview_node_invbind_.assign(hier.size(), glm::mat4(1.0f));
        for (size_t i = 0; i < hier.size(); ++i) {
            preview_node_parent_[i]     = hier[i].parent;
            preview_node_name_[i]       = hier[i].name;
            preview_node_bind_local_[i] = hier[i].local;
            auto it = invbind_map.find((int)i);
            if (it != invbind_map.end())
                preview_node_invbind_[i] = it->second;
        }

        // Per-bone rotation limits (radians, around bind) so joints can't
        // hyper-rotate.  Keyed by name; generous default for anything else.
        preview_node_rot_limit_.assign(hier.size(), 3.14159265f);
        for (size_t i = 0; i < hier.size(); ++i) {
            std::string n = hier[i].name;
            for (auto& c : n) c = (char)std::tolower((unsigned char)c);
            auto has = [&](const char* s) {
                return n.find(s) != std::string::npos; };
            float lim = 3.14159265f;
            if      (has("lower_leg") || has("calf") || has("knee"))      lim = 2.50f;
            else if (has("lower_arm") || has("forearm") || has("elbow"))  lim = 1.75f;
            else if (has("upper_leg") || has("thigh") ||
                     has("upper_arm") || has("shoulder"))                 lim = 1.30f;
            else if (has("spine") || has("chest") ||
                     has("neck")  || has("head"))                         lim = 0.45f;
            else if (has("hand")  || has("foot") || has("toe"))           lim = 0.70f;
            preview_node_rot_limit_[i] = lim;
        }

        // Weight-debug: selectable joint list (the skin's bones) + per-bone
        // influence multipliers (1 = unchanged).
        preview_weight_bones_.clear();
        for (const auto& kv : invbind_map)
            preview_weight_bones_.push_back(kv.first);
        std::sort(preview_weight_bones_.begin(), preview_weight_bones_.end());
        preview_weight_sel_ = 0;
        preview_bone_weight_scale_.assign(hier.size(), 1.0f);

        preview_clip_ = chosen_clip;

        // Owner skinned-mesh node's bind world (nwb, computed above) → its
        // inverse cancels the node_to_world the bake folded into the
        // source-world vertices, so joint matrices act on node-local positions.
        preview_mesh_node_inv_ = glm::mat4(1.0f);
        for (size_t i = 0; i < hier.size(); ++i)
            if (hier[i].mesh_ordinal == first_skinned_ordinal) {
                preview_mesh_node_inv_ = glm::inverse(nwb[i]);
                break;
            }

        // Cache the welded BIND surface graph + bind joint positions for the
        // geodesic distance-to-selected-bone debug view.
        {
            const auto& P = preview_anim_base_.positions;
            const auto& I = preview_anim_base_.indices;
            const int Vp = (int)P.size();
            preview_weld_id_.assign(Vp, 0);
            preview_weld_pos_.clear();
            std::unordered_map<uint64_t, int> weld; weld.reserve(Vp * 2);
            // Tolerance weld (snap to a grid) — MUST match the auto-rig's
            // welding so the two geodesic graphs are identical and the soup-like
            // surface reconnects (see auto_rig_plugin computeSkinWeights).
            glm::vec3 pmn(1e30f), pmx(-1e30f);
            for (const auto& pp : P) { pmn = glm::min(pmn, pp); pmx = glm::max(pmx, pp); }
            const double weld_eps =
                std::max((double)glm::length(pmx - pmn) * 1e-4, 1e-9);
            auto keyOf = [weld_eps](const glm::vec3& p) -> uint64_t {
                uint64_t h = 1469598103934665603ull;
                for (int i = 0; i < 3; ++i) {
                    const int64_t qi = (int64_t)std::llround((double)p[i] / weld_eps);
                    h ^= (uint64_t)qi; h *= 1099511628211ull;
                }
                return h;
            };
            for (int i = 0; i < Vp; ++i) {
                const uint64_t k = keyOf(P[i]);
                auto it = weld.find(k);
                if (it == weld.end()) {
                    const int id = (int)preview_weld_pos_.size();
                    weld.emplace(k, id); preview_weld_pos_.push_back(P[i]);
                    preview_weld_id_[i] = id;
                } else preview_weld_id_[i] = it->second;
            }
            const int Wn = (int)preview_weld_pos_.size();
            preview_weld_adj_.assign(Wn, {});
            auto addE = [&](int a, int b) {
                if (a == b) return;
                const float w = glm::length(preview_weld_pos_[a] -
                                            preview_weld_pos_[b]);
                preview_weld_adj_[a].push_back({b, w});
                preview_weld_adj_[b].push_back({a, w});
            };
            preview_weld_tris_.clear();
            preview_weld_tris_.reserve(I.size() / 3);
            for (size_t t = 0; t + 2 < I.size(); t += 3) {
                const int a = preview_weld_id_[I[t]],
                          b = preview_weld_id_[I[t + 1]],
                          c = preview_weld_id_[I[t + 2]];
                addE(a, b); addE(b, c); addE(c, a);
                preview_weld_tris_.push_back({ a, b, c });
            }
            preview_joint_bind_world_.assign(hier.size(), glm::vec3(0.0f));
            for (size_t i = 0; i < hier.size(); ++i)
                preview_joint_bind_world_[i] = glm::vec3(nwb[i][3]);
            // DIAGNOSTIC: same numbers the auto-rig logs as [AutoRig][dist-diag],
            // so the two welded geodesic graphs can be compared directly.
            {
                size_t edges = 0;
                for (const auto& a : preview_weld_adj_) edges += a.size();
                glm::vec3 dmn(1e30f), dmx(-1e30f);
                for (const auto& p : preview_weld_pos_) {
                    dmn = glm::min(dmn, p); dmx = glm::max(dmx, p);
                }
                char b[256];
                std::snprintf(b, sizeof(b),
                    "[Preview][dist-diag] meshNv=%d weldedW=%d edges=%zu tris=%zu "
                    "diag=%.4f", Vp, Wn, edges / 2, preview_weld_tris_.size(),
                    glm::length(dmx - dmn));
                EditorLog::get().push(b);   // → on-screen Output Log window
            }
            preview_dist_sel_cached_ = -999;
            preview_surface_close_.clear();
        }
    }

    stagePreviewPayload(std::move(p));   // clears preview_anim_ready_
    finishAssetPreview(key, display_name + " (group)");
    preview_nav_ = PreviewNav::None;

    if (want_anim) {
        preview_anim_time_    = 0.0f;
        preview_anim_playing_ = true;
        preview_anim_ready_   = true;    // set AFTER stage (which cleared it)
    }
}

// Preview a baked .rwgeo directly (no .rwobj ref needed) — how the
// bake-only character imports' skeleton meshes show in the Debug Display.
void Menu::buildRwGeoPreview(const std::string& rwgeo_path,
                             const std::string& display_name) {
    const std::string key = rwgeo_path + "#geo";
    if (key == dbg_asset_key_) return;   // already showing
    engine::helper::ModelPreviewData data;
    if (!engine::helper::loadRwGeo(rwgeo_path, data)) {
        EditorLog::get().push(
            "[preview] baked geometry unreadable: " + rwgeo_path);
        return;
    }
    dbg_preview_skinned_ = !data.joints.empty();
    engine::helper::MeshPreviewPayload p;
    p.positions = std::move(data.positions);
    p.normals   = std::move(data.normals);
    p.uvs       = std::move(data.uvs);
    p.indices   = std::move(data.indices);
    for (auto& t : data.textures) {
        engine::helper::MeshPreviewTexture mt;
        mt.rgba = std::move(t.rgba);
        mt.w = t.w;
        mt.h = t.h;
        p.textures.push_back(std::move(mt));
    }
    for (const auto& s : data.sections) {
        engine::helper::MeshPreviewSection ms;
        ms.first_index = s.first_index;
        ms.index_count = s.index_count;
        ms.base_color  = s.base_color;
        ms.metallic    = s.metallic;
        ms.roughness   = s.roughness;
        ms.tex_index   = s.tex_index;
        ms.nrm_index   = s.nrm_index;
        ms.mr_index    = s.mr_index;
        p.sections.push_back(ms);
    }
    stagePreviewPayload(std::move(p));
    finishAssetPreview(key, display_name);
    preview_nav_ = PreviewNav::None;
}

void Menu::buildRwObjPreview(const std::string& rwobj_path,
                             const std::string& fallback_name) {
    std::string src, name, geo;
    int node = -1;
    if (!engine::helper::readRwObjRef(rwobj_path, src, node, name, geo)) {
        EditorLog::get().push(
            "[preview] bad .rwobj reference: " + rwobj_path);
        return;
    }
    if (name.empty()) name = fallback_name;
    dbg_preview_skinned_ = false;   // .rwobj objects are static props

    if (!geo.empty()) {
        const std::string key = rwobj_path + "#geo";
        if (key == dbg_asset_key_) return;   // already showing
        engine::helper::ModelPreviewData data;
        if (engine::helper::loadRwGeo(geo, data)) {
            engine::helper::MeshPreviewPayload p;
            p.positions = std::move(data.positions);
            p.normals   = std::move(data.normals);
            p.uvs       = std::move(data.uvs);
            p.indices   = std::move(data.indices);
            for (auto& t : data.textures) {
                engine::helper::MeshPreviewTexture mt;
                mt.rgba = std::move(t.rgba);
                mt.w = t.w;
                mt.h = t.h;
                p.textures.push_back(std::move(mt));
            }
            for (const auto& s : data.sections) {
                engine::helper::MeshPreviewSection ms;
                ms.first_index = s.first_index;
                ms.index_count = s.index_count;
                ms.base_color  = s.base_color;
                ms.metallic    = s.metallic;
                ms.roughness   = s.roughness;
                ms.tex_index   = s.tex_index;
                ms.nrm_index   = s.nrm_index;
                ms.mr_index    = s.mr_index;
                p.sections.push_back(ms);
            }
            stagePreviewPayload(std::move(p));
            finishAssetPreview(key, name);
            // Arrow-key navigation: step through sibling .rwobj files.
            preview_nav_       = PreviewNav::RwObjSiblings;
            preview_nav_path_  = rwobj_path;
            preview_nav_index_ = -1;
            return;
        }
        EditorLog::get().push(
            "[preview] baked data unreadable, re-parsing source: " + geo);
    }
    buildAssetPreview(src, node, name);
    // The fallback parsed the SOURCE model — keep navigation anchored on
    // the .rwobj siblings so arrows still walk the group folder.
    if (preview_nav_ != PreviewNav::None) {
        preview_nav_       = PreviewNav::RwObjSiblings;
        preview_nav_path_  = rwobj_path;
        preview_nav_index_ = -1;
    }
}

// Arrow-key navigation: Left/Right step one item (wrapping); Up/Down move a
// whole ROW of the Content Browser grid (clamped at the edges).  The caller
// gates on ITS OWN window focus, so this runs for whichever panel the user
// is interacting with.
void Menu::handlePreviewArrowNav() {
    const int cols = std::max(1, browser_grid_cols_);
    int  step      = 0;
    bool row_move  = false;   // row moves CLAMP (no wrap)
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) step = +1;
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))  step = -1;
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        step = +cols;
        row_move = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        step = -cols;
        row_move = true;
    }
    if (step == 0) return;

    auto next_index = [&](int cur, int n) -> int {
        if (n <= 0) return -1;
        if (row_move) {
            const int nxt = cur + step;
            return (nxt >= 0 && nxt < n) ? nxt : -1;  // clamp
        }
        return ((cur + step) % n + n) % n;            // wrap
    };

    if (preview_nav_ == PreviewNav::RwObjSiblings) {
        // Enumerate the group folder's .rwobj files (sorted — the NNN_
        // prefix keeps bake order = tile order).
        namespace fs = std::filesystem;
        std::error_code ec;
        std::vector<std::string> sibs;
        const fs::path dir = fs::path(preview_nav_path_).parent_path();
        for (auto& e : fs::directory_iterator(dir, ec)) {
            if (e.path().extension() == ".rwobj")
                sibs.push_back(e.path().string());
        }
        std::sort(sibs.begin(), sibs.end());
        if (!sibs.empty()) {
            int cur = 0;
            for (int i = 0; i < (int)sibs.size(); ++i)
                if (sibs[i] == preview_nav_path_) { cur = i; break; }
            const int nxt = next_index(cur, (int)sibs.size());
            if (nxt >= 0 && nxt != cur) {
                buildRwObjPreview(
                    sibs[nxt], fs::path(sibs[nxt]).stem().string());
                browser_scroll_to_selected_ = true;
            }
        }
    } else if (preview_nav_ == PreviewNav::SubObjects) {
        const auto& subs = assetSubObjects(preview_nav_path_);
        if (!subs.empty()) {
            const int nxt =
                next_index(preview_nav_index_, (int)subs.size());
            if (nxt >= 0 && nxt != preview_nav_index_) {
                buildAssetPreview(
                    preview_nav_path_, nxt,
                    std::filesystem::path(preview_nav_path_)
                            .filename().string() +
                        "  /  " + subs[nxt]);
                browser_scroll_to_selected_ = true;
            }
        }
    }
}

void Menu::finishAssetPreview(const std::string& key,
                              const std::string& caption) {
    dbg_asset_key_    = key;
    dbg_disp_caption_ = caption;

    // Invalidate the scene-selection key so clicking any scene object next
    // rebuilds from the selection — but record the CURRENT selection as
    // "seen" so the unchanged selection doesn't overwrite this preview on
    // the very next frame.
    dbg_disp_obj_  = nullptr;
    dbg_disp_node_ = -3;
    engine::game_object::DrawableObject* cobj = nullptr;
    int cnode = -3;
    if (!getSelectedHighlight(cobj, cnode)) { cobj = nullptr; cnode = -3; }
    dbg_sel_seen_obj_  = cobj;
    dbg_sel_seen_node_ = cnode;

    // Bring the Debug Display tab to the front: the direct by-name call
    // raises the dock tab immediately (works mid-frame), and the flag makes
    // the panel itself claim focus on its next Begin as a fallback.
    ImGui::SetWindowFocus("Debug Display");
    preview_focus_ = true;
}

void Menu::drawDebugDisplayPanel() {
    // A freshly staged preview raises this window's dock tab so the result
    // is visible immediately (it shares a dock slot with the Outliner).
    if (preview_focus_) {
        ImGui::SetNextWindowFocus();
        preview_focus_ = false;
    }
    // Keep this panel wide/tall enough that the skinned-preview toolbar
    // (Bones / Weights / Segments / Dist) never clips, while still letting the
    // user enlarge it.
    ImGui::SetNextWindowSizeConstraints(ImVec2(420.0f, 340.0f),
                                        ImVec2(100000.0f, 100000.0f));
    if (ImGui::Begin("Debug Display")) {
        engine::game_object::DrawableObject* obj = nullptr;
        int node = -3;
        if (!getSelectedHighlight(obj, node)) { obj = nullptr; node = -3; }

        // Rebuild from the scene only when the selection CHANGES, so a
        // content-asset preview (double-clicked in the Content Browser)
        // stays up until the user clicks something else.
        const bool sel_changed =
            (obj != dbg_sel_seen_obj_) || (node != dbg_sel_seen_node_);
        dbg_sel_seen_obj_  = obj;
        dbg_sel_seen_node_ = node;
        if (sel_changed) {
            updateDebugDisplay(obj, node);
            dbg_disp_caption_.clear();
            if (obj && editor_selected_ >= 0 &&
                editor_selected_ < (int)editor_objects_.size()) {
                const auto& eo = editor_objects_[editor_selected_];
                dbg_disp_caption_ =
                    eo.name.empty() ? std::string("Object") : eo.name;
                const auto& kids = outlinerChildren(eo);
                if (editor_selected_child_ >= 0 &&
                    editor_selected_child_ < (int)kids.size())
                    dbg_disp_caption_ +=
                        "  /  " + kids[editor_selected_child_].first;
            }
        }

        // Stage the auto-rig's last-baked BASE-SKIN layer (skin triangles only).
        // Persists until another object is selected (selection didn't change
        // here, so the rebuild above won't override it).
        if (ImGui::SmallButton("Auto-Rig Skin Layer")) {
            stageAutoRigSkinLayer();   // sets dbg_skin_status_ with counts/status
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(skin only, from last bake)");
        if (!dbg_skin_status_.empty())
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "%s",
                               dbg_skin_status_.c_str());

        // The image lives in helper::MeshPreview's persistent GPU target —
        // rendered by the app (real pipeline, three spot lights) whenever a
        // staged mesh is consumed, sampled here every frame.
        const ImTextureID prev_id =
            engine::helper::MeshPreview::imguiId();
        const bool show_image =
            preview_active_ &&
            engine::helper::MeshPreview::hasImage() &&
            prev_id != 0;

        if (!dbg_disp_caption_.empty() && show_image) {
            ImGui::TextDisabled("%s", dbg_disp_caption_.c_str());
            ImGui::Separator();
        }

        if (show_image) {
            // Animated skeleton preview: re-pose + re-stage each frame.
            tickPreviewAnimation();
            // Fit the preview into the panel, preserving aspect, centred.
            // Reserve a couple of lines at the bottom for the help text so it
            // is not clipped when the image fills the panel.
            ImVec2 avail = ImGui::GetContentRegionAvail();
            const float kReserveBottom =
                ImGui::GetTextLineHeightWithSpacing() * 2.0f + 6.0f;
            if (avail.y > kReserveBottom + 32.0f) avail.y -= kReserveBottom;
            // Drive the offscreen target resolution from the panel pixel size
            // so the preview renders natively (not an upscaled fixed 512^2).
            {
                ImGuiIO& io = ImGui::GetIO();
                const float dpi = io.DisplayFramebufferScale.x > 0.0f
                                      ? io.DisplayFramebufferScale.x : 1.0f;
                auto rstep = [](float v) {
                    int p = (int)v; p = (p / 32) * 32; return p < 64 ? 64 : p;
                };
                engine::helper::MeshPreview::requestSize(
                    (uint32_t)rstep(avail.x * dpi),
                    (uint32_t)rstep(avail.y * dpi));
            }
            const float pw = (float)engine::helper::MeshPreview::width();
            const float ph = (float)engine::helper::MeshPreview::height();
            float s = std::min(avail.x / pw, avail.y / ph);
            if (s <= 0.0f || avail.x < 8.0f || avail.y < 8.0f) s = 1.0f;
            const ImVec2 sz(pw * s, ph * s);
            const float pad_x = (avail.x - sz.x) * 0.5f;
            if (pad_x > 0.0f)
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad_x);
            const ImVec2 img_p0 = ImGui::GetCursorScreenPos();
            ImGui::Image(prev_id, sz);

            // ── Orbit camera: drag to rotate, wheel to zoom ───────────
            // Overlay an invisible button on the image so a drag manipulates
            // the camera WITHOUT also moving the window (a floating/popup
            // Debug Display would otherwise be dragged too). The active item
            // consumes the mouse; the offscreen pass re-records when the
            // camera moves.
            ImGui::SetCursorScreenPos(img_p0);
            ImGui::SetNextItemAllowOverlap();   // top-left toggle sits on top
            ImGui::InvisibleButton(
                "##preview_drag", sz,
                ImGuiButtonFlags_MouseButtonLeft |
                    ImGuiButtonFlags_MouseButtonRight);
            const bool prev_active  = ImGui::IsItemActive();
            const bool prev_hovered = ImGui::IsItemHovered();
            {
                ImGuiIO& io = ImGui::GetIO();
                if (prev_active &&
                    ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                    const ImVec2 d = io.MouseDelta;
                    if (d.x != 0.0f || d.y != 0.0f)
                        engine::helper::MeshPreview::orbit(
                            -d.x * 0.4f, d.y * 0.4f);
                }
                // Right-drag: move the object around the view (pans the
                // camera target in the view plane).
                if (prev_active &&
                    ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
                    const ImVec2 d = io.MouseDelta;
                    if (d.x != 0.0f || d.y != 0.0f)
                        engine::helper::MeshPreview::pan(d.x, d.y);
                }
                if (prev_hovered && io.MouseWheel != 0.0f)
                    engine::helper::MeshPreview::zoom(io.MouseWheel);
            }

            // ── Static / Skeletal badge (top-right corner of the image) ──
            {
                const char* badge =
                    dbg_preview_skinned_ ? "Skeletal" : "Static";
                const ImU32 fg = dbg_preview_skinned_
                                     ? IM_COL32(130, 190, 255, 255)
                                     : IM_COL32(200, 200, 200, 255);
                const ImVec2 tsz = ImGui::CalcTextSize(badge);
                const float padx = 6.0f, pady = 3.0f;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const ImVec2 b1(img_p0.x + sz.x - tsz.x - padx * 2.0f - 4.0f,
                                img_p0.y + 4.0f);
                const ImVec2 b2(img_p0.x + sz.x - 4.0f,
                                img_p0.y + 4.0f + tsz.y + pady * 2.0f);
                dl->AddRectFilled(b1, b2, IM_COL32(0, 0, 0, 150), 4.0f);
                dl->AddText(ImVec2(b1.x + padx, b1.y + pady), fg, badge);
            }

            // ── Skeleton debug draw: top-left toggle + bone overlay ──────
            // The toggle/overlay only make sense for a skeleton mesh that has
            // a live skeleton (preview_anim_ready_).  Save/restore the layout
            // cursor so the button (placed at the image's top-left) doesn't
            // push the controls below the image off-position.
            const ImVec2 cursor_below_img = ImGui::GetCursorScreenPos();
            if (dbg_preview_skinned_ && preview_anim_ready_) {
                ImGui::SetCursorScreenPos(ImVec2(img_p0.x + 6.0f,
                                                 img_p0.y + 6.0f));
                if (ImGui::Button(preview_show_skeleton_ ? "Bones: On"
                                                         : "Bones: Off"))
                    preview_show_skeleton_ = !preview_show_skeleton_;
                ImGui::SameLine();
                if (ImGui::Button(preview_show_weights_ ? "Weights: On"
                                                        : "Weights: Off")) {
                    preview_show_weights_ = !preview_show_weights_;
                    if (preview_show_weights_) {
                        preview_show_segments_ = false;
                        preview_show_distance_ = false;
                        preview_show_weight_sum_ = false;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button(preview_show_segments_ ? "Segments: On"
                                                         : "Segments: Off")) {
                    preview_show_segments_ = !preview_show_segments_;
                    if (preview_show_segments_) {
                        preview_show_weights_ = false;
                        preview_show_distance_ = false;
                        preview_show_weight_sum_ = false;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button(preview_show_distance_ ? "Dist: On"
                                                         : "Dist: Off")) {
                    preview_show_distance_ = !preview_show_distance_;
                    if (preview_show_distance_) {
                        preview_show_weights_  = false;
                        preview_show_segments_ = false;
                        preview_show_weight_sum_ = false;
                    }
                    preview_dist_sel_cached_ = -999;   // force recompute
                }
                ImGui::SameLine();
                if (ImGui::Button(preview_show_weight_sum_ ? "Sum: On"
                                                           : "Sum: Off")) {
                    preview_show_weight_sum_ = !preview_show_weight_sum_;
                    if (preview_show_weight_sum_) {
                        preview_show_weights_  = false;
                        preview_show_segments_ = false;
                        preview_show_distance_ = false;
                    }
                }
            }
            // Skeleton overlay — shown for Bones mode OR Weights mode (the
            // weight colour lives on the MESH now, so here we only draw the
            // bones, highlighting the selected bone white in Weights mode).
            const bool overlay_skel =
                (preview_show_skeleton_ || preview_show_weights_ ||
                 preview_show_segments_ || preview_show_distance_ ||
                 preview_show_weight_sum_) &&
                preview_anim_ready_ &&
                preview_joint_world_.size() == preview_node_parent_.size() &&
                !preview_joint_world_.empty();
            if (overlay_skel) {
                int selnode = -1;
                if ((preview_show_weights_ || preview_show_distance_) &&
                    !preview_weight_bones_.empty()) {
                    int s = preview_weight_sel_;
                    if (s < 0) s = 0;
                    if (s >= (int)preview_weight_bones_.size())
                        s = (int)preview_weight_bones_.size() - 1;
                    selnode = preview_weight_bones_[s];
                }
                const glm::mat4 vpm = engine::helper::MeshPreview::lastViewProj();
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->PushClipRect(img_p0,
                                 ImVec2(img_p0.x + sz.x, img_p0.y + sz.y), true);
                auto project = [&](const glm::vec3& w, ImVec2& out) -> bool {
                    const glm::vec4 c = vpm * glm::vec4(w, 1.0f);
                    if (c.w <= 1e-5f) return false;            // behind camera
                    const float u = (c.x / c.w) * 0.5f + 0.5f;
                    const float v = (c.y / c.w) * 0.5f + 0.5f; // Vulkan Y flip
                    out = ImVec2(img_p0.x + u * sz.x, img_p0.y + v * sz.y);
                    return true;
                };
                const ImU32 bone_col  = IM_COL32(255, 215, 70, 220);
                const ImU32 joint_col = IM_COL32(255, 90, 60, 255);
                const ImU32 sel_col   = IM_COL32(255, 255, 255, 255);
                for (size_t i = 0; i < preview_joint_world_.size(); ++i) {
                    ImVec2 a;
                    if (!project(preview_joint_world_[i], a)) continue;
                    const bool is_sel = ((int)i == selnode);
                    const int par = preview_node_parent_[i];
                    if (par >= 0 && par < (int)preview_joint_world_.size()) {
                        ImVec2 b;
                        if (project(preview_joint_world_[par], b))
                            dl->AddLine(a, b, is_sel ? sel_col : bone_col,
                                        is_sel ? 3.0f : 2.0f);
                    }
                    dl->AddCircleFilled(a, is_sel ? 5.0f : 3.0f,
                                        is_sel ? sel_col : joint_col);
                }
                dl->PopClipRect();
            }
            ImGui::SetCursorScreenPos(cursor_below_img);

            // ── Animation playback controls (skeleton previews only) ────
            if (preview_anim_ready_) {
                ImGui::Checkbox("Play##prevanim", &preview_anim_playing_);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110.0f);
                ImGui::SliderFloat("Speed##prevanim", &preview_anim_speed_,
                                   0.0f, 3.0f);
                const float dur = preview_clip_.duration > 0.0f
                                      ? preview_clip_.duration : 1.0f;
                ImGui::SetNextItemWidth(-1.0f);
                if (ImGui::SliderFloat("##prevanimtime", &preview_anim_time_,
                                       0.0f, dur, "%.2fs"))
                    preview_anim_playing_ = false;   // scrubbing pauses
            }

            // ── Weight debug: bone selector + influence tweak ───────────
            if ((preview_show_weights_ || preview_show_distance_) &&
                !preview_weight_bones_.empty()) {
                const int count = (int)preview_weight_bones_.size();
                if (ImGui::SmallButton("<##wbprev")) preview_weight_sel_--;
                ImGui::SameLine();
                if (ImGui::SmallButton(">##wbnext")) preview_weight_sel_++;
                // Wrap the selection.
                if (preview_weight_sel_ < 0)        preview_weight_sel_ = count - 1;
                if (preview_weight_sel_ >= count)   preview_weight_sel_ = 0;
                const int sn = preview_weight_bones_[preview_weight_sel_];
                ImGui::SameLine();
                const std::string nm =
                    (sn >= 0 && sn < (int)preview_node_name_.size())
                        ? preview_node_name_[sn] : std::string("?");
                ImGui::Text("bone %d/%d: %s", preview_weight_sel_ + 1, count,
                            nm.c_str());
                // Influence tweak is weights-mode only (distance is read-only).
                if (preview_show_weights_ &&
                    sn >= 0 && sn < (int)preview_bone_weight_scale_.size()) {
                    ImGui::SetNextItemWidth(-1.0f);
                    ImGui::SliderFloat("##wscale",
                                       &preview_bone_weight_scale_[sn],
                                       0.0f, 3.0f, "influence x%.2f");
                }
            }

            ImGui::TextDisabled(
                "drag: orbit    right-drag: move    wheel: zoom    "
                "arrows: navigate panel grid");

            // ── Arrow keys: navigate the Content Browser grid ───────────
            // Shared handler — also invoked from the Content Browser when
            // IT has focus, so arrows move the selection from either side.
            if (ImGui::IsWindowFocused(
                    ImGuiFocusedFlags_RootAndChildWindows) &&
                preview_nav_ != PreviewNav::None) {
                handlePreviewArrowNav();
            }
        } else {
            ImGui::TextDisabled(
                "(click an object in the scene / Outliner, or double-click\n"
                " a tile in the Content Browser, to preview it here)");
        }
    }
    ImGui::End();
}

}//namespace ui
}//namespace engine
