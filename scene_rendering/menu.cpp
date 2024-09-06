#include <vector>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "engine_helper.h"

#include "menu.h"

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
}

namespace engine {
namespace scene_rendering {

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
    const std::shared_ptr<renderer::ImageView>& image_view) {
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
    texture_id_ =
        renderer::Helper::addImTextureID(sampler, image_view);
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
}

bool Menu::draw(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::Framebuffer>& framebuffer,
    const glm::uvec2& screen_size,
    const std::shared_ptr<scene_rendering::Skydome>& skydome,
    bool& dump_volume_noise,
    const float& delta_t) {

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    std::vector<er::ClearValue> clear_values;
    clear_values.resize(2);
    clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

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
    bool compile_shaders = false;

    bool test_true = true;
    ImGui::Begin(
        "fps",
        &test_true,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoInputs);
    ImGui::SetWindowPos(ImVec2(float(screen_size.x) - 128.0f, 20.0f));
    ImGui::SetWindowSize(ImVec2((float)128, (float)12));
    ImGui::BeginChild("fps", ImVec2(0, 0), false);
    float fps = delta_t > 0.0f ? 1.0f / delta_t : 0.0f;
    ImGui::Text("fps : %8.5f", fps);
    ImGui::EndChild();
    ImGui::End();

#if 1
    ImVec2 childSize = ImVec2(1024, 768);  // Define the size of the child window
    ImGui::Begin(
        "Viewport",
        nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar);
    ImGui::SetWindowPos(ImVec2(0, 20.0f));
    auto window_size = ImGui::GetWindowSize();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::BeginChild("Viewport", ImVec2(0, 0), true);
//    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    // Render something here, for example, a colored rectangle:
    ImVec2 rectMin = ImGui::GetCursorScreenPos();
    ImVec2 rectMax = ImVec2(rectMin.x + childSize.x, rectMin.y + childSize.y);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    //drawList->AddRectFilled(rectMin, rectMax, IM_COL32(100, 100, 255, 255), 20.0f, ImDrawFlags_RoundCornersAll);
 
    drawList->AddImage(texture_id_, ImVec2(0, 0), window_size, ImVec2(0, 0), ImVec2(1, 1));

//    ImGui::PopStyleVar(1);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::End();
#endif
    if (ImGui::BeginMainMenuBar())
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

        if (ImGui::BeginMenu("Tools"))
        {
            if (ImGui::MenuItem("Compile Shaders", NULL)) {
                compile_shaders = true;
            }

            if (ImGui::MenuItem("Dump noise volumetric texture", NULL)) {
                dump_volume_noise = true;
            }

            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    bool in_focus =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) ||
        ImGui::IsWindowFocused(ImGuiHoveredFlags_AnyWindow);

    if (s_show_skydome) {
        if (ImGui::Begin("Skydome", &s_show_skydome)) {
            ImGui::SliderFloat("phase func g", &skydome->getG(), -1.0f, 2.0f);
            ImGui::SliderFloat("rayleigh scale height", &skydome->getRayleighScaleHeight(), 0.0f, 16000.0f);
            ImGui::SliderFloat("mei scale height", &skydome->getMieScaleHeight(), 0.0f, 2400.0f);
        }
        ImGui::End();
    }

    if (s_show_weather) {
        if (ImGui::Begin("Weather", &s_show_weather, ImGuiWindowFlags_NoScrollbar)) {
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
        if (ImGui::Begin("Shader Compile Error", &s_show_shader_error_message, ImGuiWindowFlags_NoScrollbar)) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%s", s_shader_error_message.c_str());
        }
        ImGui::End();
    }

    renderer::Helper::addImGuiToCommandBuffer(cmd_buf);

    cmd_buf->endRenderPass();

    return in_focus;
}

void Menu::destroy() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

}//namespace scene_rendering
}//namespace engine
