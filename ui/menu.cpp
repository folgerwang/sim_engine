#include <vector>
#include <filesystem>
#include <cmath>
#include <algorithm>
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
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"

#include "menu.h"
#include "plugins/plugin_manager.h"
#include "game_object/mesh_load_task_manager.h"
#include "scene_rendering/ssao.h"
#include "scene_rendering/cluster_renderer.h"
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

    static bool s_spawn_player = false;
    static bool s_select_load_gltf = false;
    static bool s_show_skydome = false;
    static bool s_show_weather = false;
    static bool s_show_shader_error_message = false;
    static std::string s_shader_error_message;
    static bool s_show_gpu_profiler = false;
    bool compile_shaders = false;

    bool test_true = true;
    ImVec2 vp_pos = ImGui::GetMainViewport()->Pos;
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
    ImGui::SetWindowPos(ImVec2(vp_pos.x + float(screen_size.x) - 220.0f, vp_pos.y + menu_height));
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
            const ImVec2 mesh_win_pos = {
                vp_pos.x + float(screen_size.x) - 380.0f,
                vp_pos.y + menu_height + 40.0f };

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
            if (ImGui::MenuItem("Spawn player", NULL)) {
                s_spawn_player = true;
            }

            if (ImGui::MenuItem("Load gltf", NULL)) {
                s_select_load_gltf = true;
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
            show_ibl_debug_ = true;
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Shadow"))
        {
            if (ImGui::MenuItem("Turn off shadow pass", NULL, turn_off_shadow_pass_)) {
                turn_off_shadow_pass_ = !turn_off_shadow_pass_;
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

    if (s_spawn_player) {
        spawn_gltf_name_ = "assets/CesiumMan.gltf";
        s_spawn_player = false;
    }

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

    // ---- IBL / Sky cubemap debug window -----------------------------------
    if (show_ibl_debug_) {
        const float thumb = ibl_debug_thumb_size_;
        const float row_w = thumb * 6.0f + 60.0f;
        const float row_h = thumb + 60.0f;  // +slider line per row
        const float win_w = row_w + 40.0f;
        const float win_h = row_h * 5.0f + 80.0f;
        ImGui::SetNextWindowSize(ImVec2(win_w, win_h), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowViewport(main_vp_id);
        if (ImGui::Begin("IBL / Sky Debug", &show_ibl_debug_,
                         ImGuiWindowFlags_NoDocking)) {
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

            // IBL diffuse (Lambertian, single mip).
            draw_mip_row(
                "ibl_diffuse", "IBL diffuse (Lambertian)",
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
            const char* labels[] = { "Cascade 0\n(near)", "Cascade 1", "Cascade 2", "Cascade 3\n(far)" };
            for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
                if (k > 0) ImGui::SameLine();
                ImGui::BeginGroup();
                // Label above each image
                ImGui::Text("%s", labels[k]);
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

    // Chat box / dialogue UI only during gameplay.
    if (game_state_ == GameState::InGame) {
        chat_box_->draw(cmd_buf, render_pass, framebuffer, screen_size, skydome, dump_volume_noise, delta_t);
    }

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
}

}//namespace ui
}//namespace engine
