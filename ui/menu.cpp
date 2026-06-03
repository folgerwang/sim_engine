#include <vector>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <source_location>
#include <cstdlib>

// stbi_load / stbi_image_free are already compiled into the project via
// tinygltf (STB_IMAGE_IMPLEMENTATION lives in engine_helper.cpp).
// Pull in just the declarations so we can scan the background PNG.
extern "C" {
    unsigned char* stbi_load(const char*, int*, int*, int*, int);
    void stbi_image_free(void*);
}
#ifndef STBI_rgb_alpha
#define STBI_rgb_alpha 4
#endif

#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder API for the editor default layout
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "game_object/drawable_object.h"  // Details panel reads object transform

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"

#include "menu.h"
#include "plugins/plugin_manager.h"
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
        s.Colors[ImGuiCol_PopupBg]           = ImVec4(0.05f, 0.06f, 0.12f, 0.96f);
        s.Colors[ImGuiCol_Border]            = ImVec4(0.55f, 0.45f, 0.22f, 0.40f);
        s.Colors[ImGuiCol_BorderShadow]      = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        s.Colors[ImGuiCol_FrameBg]           = ImVec4(0.10f, 0.10f, 0.18f, 0.85f);
        s.Colors[ImGuiCol_FrameBgHovered]    = ImVec4(0.18f, 0.16f, 0.28f, 1.00f);
        s.Colors[ImGuiCol_FrameBgActive]     = ImVec4(0.25f, 0.22f, 0.38f, 1.00f);
        s.Colors[ImGuiCol_TitleBg]           = ImVec4(0.04f, 0.05f, 0.10f, 1.00f);
        s.Colors[ImGuiCol_TitleBgActive]     = ImVec4(0.08f, 0.08f, 0.15f, 1.00f);
        s.Colors[ImGuiCol_TitleBgCollapsed]  = ImVec4(0.04f, 0.05f, 0.10f, 0.75f);
        s.Colors[ImGuiCol_MenuBarBg]         = ImVec4(0.03f, 0.04f, 0.09f, 0.95f);
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
        s.Colors[ImGuiCol_Tab]               = ImVec4(0.08f, 0.08f, 0.15f, 1.00f);
        s.Colors[ImGuiCol_TabHovered]        = gold_dim;
        s.Colors[ImGuiCol_TabActive]         = ImVec4(0.14f, 0.14f, 0.22f, 1.00f);
        s.Colors[ImGuiCol_TabUnfocused]      = ImVec4(0.05f, 0.06f, 0.12f, 1.00f);
        s.Colors[ImGuiCol_TabUnfocusedActive]= ImVec4(0.10f, 0.10f, 0.18f, 1.00f);
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

    // rt_texture_id_ and main_texture_id_ are also pool-allocated and
    // would be dangling here.  They're only referenced inside `#if 0`
    // blocks today (drawViewport for the disabled raytrace preview),
    // but null them out defensively so any future re-enabling doesn't
    // accidentally bind a dead handle from the old pool.
    rt_texture_id_   = ImTextureID(0);
    main_texture_id_ = ImTextureID(0);
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
    // panels dock around it.  Only in-game; the title screen is untouched.
    if (editor_enabled_ && game_state_ == GameState::InGame) {
        drawEditorDockSpace();
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
    if ((classifier_status_ == ClassifierStatus::Pending ||
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
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImVec2 vp_pos = vp->Pos;
        const float  vp_w   = vp->Size.x > 0.0f ? vp->Size.x : float(screen_size.x);

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
        const float vp_w = vp->Size.x;
        const float vp_h = vp->Size.y;

        // Menu block: centred horizontally, lower-third vertically.
        const float item_w = 320.0f;
        const float item_h = 48.0f;
        const float spacing = 12.0f;
        const int   n_items = (int)title_config_.menu_items.size();
        const float block_h = n_items * item_h + (n_items - 1) * spacing;
        const float start_x = vp->Pos.x + (vp_w - item_w) * 0.5f;
        const float start_y = vp->Pos.y + vp_h * 0.52f;

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

    // (Title banner + version stamp now drawn inline inside the
    // twilight backdrop window earlier in this frame — see the call
    // to drawFantasyTopBackdrop() above.)

    // Async mesh-load HUD — dual counter-rotating rune ring + filename
    // list. Sits just under the fps widget and stays hidden when
    // nothing is loading. NoInputs so it can't steal clicks from the
    // menu. See drawRuneLoader() above for the ring geometry; the
    // aesthetic mirrors the loader in
    // realworld/design/fantasy_menu.html.
    if (mesh_load_task_manager_) {
        auto in_flight = mesh_load_task_manager_->inFlightFilenames();
        if (!in_flight.empty()) {
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
            ImGui::TextColored(
                ImVec4(0.95f, 0.86f, 0.60f, 1.0f),
                "Loading %zu mesh%s",
                in_flight.size(),
                in_flight.size() == 1 ? "" : "es");
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
        if (ImGui::BeginMenu("Raytracing"))
        {
            if (ImGui::MenuItem("Turn off ray tracing", NULL, turn_off_ray_tracing_)) {
                turn_off_ray_tracing_ = !turn_off_ray_tracing_;
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Terrain"))
        {
            if (ImGui::MenuItem("Turn off water pass", NULL, turn_off_water_pass_)) {
                turn_off_water_pass_ = !turn_off_water_pass_;
            }

            if (ImGui::MenuItem("Turn off grass pass", NULL, turn_off_grass_pass_)) {
                turn_off_grass_pass_ = !turn_off_grass_pass_;
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Game Objects"))
        {
            // (The legacy "Spawn player" menu item was removed: the
            // player is now eager-loaded at application startup with
            // a single asset, so an on-demand spawn would either
            // reload the same rig or substitute a different asset
            // whose joint names PlayerController doesn't know how to
            // pose.  Use "Reset player position" below to re-snap the
            // existing character to the current camera view.)

            // Recovery option for the "I can't find my character" case:
            // re-anchors the player to a point in front of the
            // current camera, on the ground.  Cheap to invoke — just
            // sets a flag the application drains and forwards to
            // PlayerController::spawnAt().  See the menu's
            // reset_player_position_requested_ field for details.
            if (ImGui::MenuItem("Reset player position", NULL)) {
                reset_player_position_requested_ = true;
            }

            if (ImGui::MenuItem("Load gltf", NULL)) {
                s_select_load_gltf = true;
            }

            ImGui::EndMenu();
        }

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

        if (ImGui::BeginMenu("Skydome")) {
            s_show_skydome = true;
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Weather System")) {
            s_show_weather = true;
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("IBL Debug")) {
            if (ImGui::MenuItem("Open IBL / Sky Debug")) {
                show_ibl_debug_ = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Shadow"))
        {
            if (ImGui::MenuItem("Turn off shadow pass", NULL, turn_off_shadow_pass_)) {
                turn_off_shadow_pass_ = !turn_off_shadow_pass_;
            }

            // CSM silhouette prepass — see comment on the member field in
            // menu.h and the rationale at csm_silhouette_prepass.mesh's
            // header.  Turning it off makes the shadow pass clear depth
            // to 1.0 (the legacy behaviour) and skip the prepass dispatch,
            // so the user can A/B-compare shadow-pass timing.
            if (ImGui::MenuItem("CSM silhouette prepass", NULL,
                                 csm_silhouette_prepass_enabled_)) {
                csm_silhouette_prepass_enabled_ =
                    !csm_silhouette_prepass_enabled_;
            }

            // ── CSM drawable-shadow draw mode ─────────────────────────
            // Three mutually-exclusive picks for how the drawable shadow
            // path amplifies geometry across the cascades.  See the enum
            // comment in menu.h for the per-mode trade-offs.  The cluster
            // shadow path is unaffected — it always uses task+mesh.
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

        // ── Forward / cluster PBR debug visualisation ────────────────────────
        // Drives DEBUG_RENDER_MODE_* (packed into camera_info.input_features
        // by application.cpp).  Both base.frag and cluster_bindless.frag read
        // and dispatch on this value; mode 0 = the normal shaded path.
        if (ImGui::BeginMenu("Render Debug"))
        {
            // Modes 0..12 are the original PBR / G-buffer debug overlays
            // baked into base.frag and cluster_bindless.frag.  Mode 13
            // is the MeshCategory solid-colour overlay added when the
            // AI-backed material classifier landed — it reads category
            // bits packed into BindlessMaterialParams.flags and paints
            // every rendered mesh with the same colour the collision
            // debug overlay uses (Floor green, Wall red, Door amber, …).
            // The enum value MUST match DEBUG_RENDER_MODE_CATEGORY in
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
            };
            for (int i = 0; i < IM_ARRAYSIZE(kRenderDebugItems); ++i) {
                const auto& item = kRenderDebugItems[i];
                bool selected = (debug_render_mode_ == item.mode_id);
                if (ImGui::MenuItem(item.label, NULL, selected)) {
                    debug_render_mode_ = item.mode_id;
                }
            }

            // ── Skeleton view selector ─────────────────────────────────
            // Tri-state: show the character only (default), show the 19
            // bone-marker cubes ON TOP of the character (alignment check),
            // or show ONLY the bones (inspect the skeleton in isolation
            // with the skinned mesh hidden).  Application reads this each
            // frame and toggles DrawableObject::setVisible(...) on the
            // player mesh and every bone marker.
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

            // ── Colour legend for the Mesh-Category mode ─────────────────
            // Inline 11-row swatch so users can read the overlay without
            // remembering the colour mapping.  Swatch RGBs MUST match
            // the categoryColor() switch in collision_debug.frag and the
            // DEBUG_RENDER_MODE_CATEGORY branch in cluster_bindless.frag
            // — drift will mis-label the legend.  Rendered as a small
            // child block so it doesn't bloat the menu when collapsed;
            // only shown while mode 13 is the active selection.
            if (debug_render_mode_ == 13) {
                ImGui::Separator();
                ImGui::TextDisabled("Mesh-Category colour key");
                // The G-buffer doesn't carry per-material flag bits,
                // so deferred_resolve.comp can't read the category.
                // application.cpp force-flips to forward rendering
                // while this mode is active; surfacing the fact here
                // so it's not surprising when the toggle "doesn't do
                // anything" in Deferred (it does -- it just also
                // bypasses Deferred).
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
                    // ImGui::ColorButton draws a non-interactive swatch
                    // when the NoTooltip+NoPicker flags are set; pair
                    // it with a label on the same line for a compact
                    // 11-row key.
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
                // Toggle to open the per-name inspector window so the
                // user can see the actual material / object → category
                // verdicts the LLM produced (511 rows would not fit
                // on the menu, so it opens as a separate scrollable
                // window).  Only enabled once the classifier snapshot
                // has been pushed from application.cpp — before then
                // the window has nothing to show.
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
            // Hi-Z mip selector — only meaningful when DEBUG_RENDER_MODE_HIZ
            // is the active mode.  Range is 0..15 (the field carries 4 bits
            // of mip selection in input_features); the application clamps
            // against the actual pyramid mip count when packing.
            if (debug_render_mode_ == 12) {
                ImGui::Separator();
                ImGui::SliderInt("Hi-Z mip", &hiz_debug_mip_, 0, 15);
            }
            ImGui::Separator();

            // ── Forward vs Deferred rendering toggle ─────────────────────
            // Two mutually-exclusive radio items — picking one un-ticks the
            // other.  The application reads isDeferredRendering() each
            // frame in drawScene to route the cluster opaque pass through
            // the G-buffer + compute resolve (Deferred) or the legacy
            // single-pass bindless pipeline (Forward).  All non-cluster
            // opaque passes (terrain / grass / hair / sky / OIT) are
            // unaffected — they always render forward over the result.
            if (ImGui::MenuItem(
                    "Pipeline: Deferred", NULL, deferred_rendering_)) {
                deferred_rendering_ = true;
            }
            if (ImGui::MenuItem(
                    "Pipeline: Forward",  NULL, !deferred_rendering_)) {
                deferred_rendering_ = false;
            }
            ImGui::Separator();
            // Toggle the popup window that displays each face of the
            // camera-positioned dynamic reflection cubemap.  Useful for
            // verifying that the per-frame face capture and the depth-
            // aware reprojection of the other 5 faces are producing
            // coherent results as the camera moves.  The window is
            // rendered later in this same draw() call (see the
            // `show_dynamic_cube_debug_` block further down).
            if (ImGui::MenuItem(
                    "Dynamic Cubemap Viewer", NULL, show_dynamic_cube_debug_)) {
                show_dynamic_cube_debug_ = !show_dynamic_cube_debug_;
            }
            // Hi-Z pyramid (last-frame depth) viewer — opens a strip of
            // mip thumbnails so we can verify the per-frame Hi-Z build
            // dispatch is actually populating the pyramid with depth
            // values, before chasing further bugs in the cluster cull
            // sample math.
            if (ImGui::MenuItem(
                    "Hi-Z Pyramid Viewer", NULL, show_hiz_debug_)) {
                show_hiz_debug_ = !show_hiz_debug_;
            }
            // VT pool viewer — opens a strip of the four 4096² layer
            // pool textures so we can verify that Runtime Virtual
            // Texture registration is actually copying texel data
            // into the pool.  Each populated 128×128 page shows up
            // as a patch within the atlas; empty slots stay black.
            // Albedo + Normal panels populate; MR-AO + Emissive stay
            // black until those layers are wired in v2.
            if (ImGui::MenuItem(
                    "VT Pool Viewer", NULL, show_vt_pool_debug_)) {
                show_vt_pool_debug_ = !show_vt_pool_debug_;
            }
            // Toggle in-scene probe icospheres.  Each sphere is colored
            // by its probe's SH-evaluated irradiance in the surface
            // normal direction — pending probes (not yet baked) draw
            // solid red so they're easy to spot.  See
            // AmbientProbeSystem::drawDebug + probe_debug.frag.
            if (ImGui::MenuItem(
                    "Show Probes (in scene)", NULL, show_probe_debug_)) {
                show_probe_debug_ = !show_probe_debug_;
            }

            // ── Glass / translucent rendering mode ───────────────────────
            // Pure storage on the cluster renderer; the application's
            // glass dispatch block in drawScene reads this and calls
            // either drawTranslucentForward (ALPHA_BLEND) or
            // drawTranslucentOit (WBOIT).  Both pipelines are kept alive
            // after init, so switching is free at runtime.  Previously
            // lived inside the VT Pool Debug window, which made it
            // invisible unless that unrelated viewer happened to be open.
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
            ImGui::EndMenu();
        }

        if (cluster_renderer_ && ImGui::MenuItem("Smart Mesh")) {
            show_smart_mesh_window_ = !show_smart_mesh_window_;
        }

        if (ImGui::BeginMenu("Tools"))
        {
            if (ImGui::MenuItem("Compile Shaders", NULL)) {
                compile_shaders = true;
            }

            if (ImGui::MenuItem("Dump noise volumetric texture", NULL)) {
                dump_volume_noise = true;
            }

            if (ImGui::MenuItem("GPU Profiler", NULL, s_show_gpu_profiler)) {
                s_show_gpu_profiler = !s_show_gpu_profiler;
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

    bool in_focus =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) ||
        ImGui::IsWindowFocused(ImGuiHoveredFlags_AnyWindow);

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
        gpu_profiler_->drawImGui();
    }
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

    // ---- Analog clock overlay (top-left, always on top) --------------------
    // Drawn directly on the foreground draw list so it is always visible
    // above every ImGui window, regardless of window z-order.
    {
        constexpr float kFaceSize = 96.0f;
        constexpr float kPad      = 4.0f;
        const float kMenuH = ImGui::GetFrameHeight();

        // Top-left anchor in screen space (below the menu bar).
        const ImVec2 vp_min = ImGui::GetMainViewport()->Pos;
        const ImVec2 origin(vp_min.x + 14.0f, vp_min.y + kMenuH + 8.0f);
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
        ImGui::DockBuilderDockWindow("Outliner", right);
        ImGui::DockBuilderDockWindow("Details",  right);   // tab with Outliner
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
    drawDetailsPanel();
    drawContentBrowserPanel();
}

void Menu::drawOutlinerPanel() {
    if (ImGui::Begin("Outliner")) {
        ImGui::TextDisabled("World  (%d objects)", (int)editor_objects_.size());
        ImGui::Separator();
        for (int i = 0; i < (int)editor_objects_.size(); ++i) {
            const std::string label =
                (editor_objects_[i].name.empty()
                    ? ("Object " + std::to_string(i))
                    : editor_objects_[i].name) + "##obj" + std::to_string(i);
            if (ImGui::Selectable(label.c_str(), editor_selected_ == i))
                editor_selected_ = i;
        }
        if (editor_objects_.empty())
            ImGui::TextDisabled("(no scene objects yet)");
    }
    ImGui::End();
}

void Menu::drawDetailsPanel() {
    if (ImGui::Begin("Details")) {
        if (editor_selected_ >= 0 &&
            editor_selected_ < (int)editor_objects_.size()) {
            EditorSceneObject& o = editor_objects_[editor_selected_];
            ImGui::Text("Name: %s", o.name.c_str());
            ImGui::Separator();
            if (o.obj) {
                ImGui::SeparatorText("Transform");
                glm::mat4 m = o.obj->getLocation();
                glm::vec3 pos(m[3].x, m[3].y, m[3].z);
                if (ImGui::DragFloat3("Location", &pos.x, 0.05f)) {
                    o.obj->setInstanceRootTransform(
                        pos, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
                }
                ImGui::SeparatorText("Rendering");
                bool vis = o.obj->isVisible();
                if (ImGui::Checkbox("Visible", &vis)) o.obj->setVisible(vis);
                ImGui::Text("Loaded: %s", o.obj->isReady() ? "yes" : "(streaming)");
            } else {
                ImGui::TextDisabled("(no drawable bound)");
            }
        } else {
            ImGui::TextDisabled("Select an object in the Outliner.");
        }
    }
    ImGui::End();
}

void Menu::drawContentBrowserPanel() {
    namespace fs = std::filesystem;
    if (ImGui::Begin("Content Browser")) {
        if (ImGui::SmallButton("Up")) {
            fs::path p(content_dir_);
            if (p.has_parent_path() && !p.parent_path().empty())
                content_dir_ = p.parent_path().string();
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(content_dir_.c_str());
        ImGui::Separator();

        std::error_code ec;
        if (fs::exists(content_dir_, ec) && fs::is_directory(content_dir_, ec)) {
            std::vector<std::string> dirs, files;
            for (auto& e : fs::directory_iterator(content_dir_, ec)) {
                if (e.is_directory()) dirs.push_back(e.path().filename().string());
                else                  files.push_back(e.path().filename().string());
            }
            std::sort(dirs.begin(), dirs.end());
            std::sort(files.begin(), files.end());
            for (const auto& d : dirs) {
                const std::string lbl = "[ " + d + " ]##d_" + d;
                if (ImGui::Selectable(lbl.c_str(), false,
                        ImGuiSelectableFlags_AllowDoubleClick) &&
                    ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    content_dir_ = (fs::path(content_dir_) / d).string();
                }
            }
            for (const auto& f : files)
                ImGui::BulletText("%s", f.c_str());
        } else {
            ImGui::TextDisabled("(folder not found: %s)", content_dir_.c_str());
        }
    }
    ImGui::End();
}

}//namespace ui
}//namespace engine
