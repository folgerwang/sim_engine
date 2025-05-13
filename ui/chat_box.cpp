#include <vector>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "engine_helper.h"

#include "chat_box.h"

namespace engine {
namespace ui {

bool ChatBox::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::Framebuffer>& framebuffer,
    const glm::uvec2& screen_size,
    const std::shared_ptr<scene_rendering::Skydome>& skydome,
    bool& dump_volume_noise,
    const float& delta_t) {

    // In your per-frame UI construction section:

// Get screen dimensions for positioning
    ImGuiIO& io = ImGui::GetIO();
    float screenWidth = io.DisplaySize.x;
    float screenHeight = io.DisplaySize.y;

    // --- Speaker Name and Dialogue Text ---
    // Let's define its properties more clearly

    // Adjust these percentages to fit your layout and desired look
    float speakerWindowPosX = screenWidth * 0.15f;  // Example: Start 15% from the left
    float speakerWindowPosY = screenHeight * 0.1f; // Example: Start 10% from the top (above chat options)
    float speakerWindowWidth = screenWidth * 0.45f; // KEY: Increase this width (e.g., 45-50% of screen width)

    ImGui::SetNextWindowPos(ImVec2(speakerWindowPosX, speakerWindowPosY), ImGuiCond_Appearing);
    // Set a specific width, height will be determined by content due to AlwaysAutoResize
    ImGui::SetNextWindowSize(ImVec2(speakerWindowWidth, 0.0f), ImGuiCond_Always); // Using 0.0f for height + AlwaysAutoResize

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f)); // Add some internal padding
    ImGui::SetNextWindowBgAlpha(0.0f); // Transparent background as in the image (or a very subtle dark background)

    ImGuiWindowFlags speaker_window_flags = ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_AlwaysAutoResize; // Important for height

    ImGui::Begin("SpeakerInfo", nullptr, speaker_window_flags);

    std::string current_speaker_name = "SSGT HERRERA";

    // --- Speaker Name ---
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f)); // Light text color for speaker name
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, ImGui::GetStyle().ItemSpacing.y * 1.5f)); // More space after name
    ImGui::TextUnformatted(current_speaker_name.c_str());
    ImGui::PopStyleVar(); // Pop ItemSpacing
    ImGui::PopStyleColor(); // Pop text color

    // --- Dialogue Lines ---
    std::vector<std::string> currentDialogueLines = { 
        "You listen to what I say, I'll get your ass out in the smallest possible",
        "number of pieces.And you can have a cookie."
    };
    std::string currentQuestionLine = "You got any questions, now's the time.";

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, 1.0f)); // Slightly dimmer for dialogue
    for (const auto& line : currentDialogueLines) {
        ImGui::TextWrapped("%s", line.c_str()); // TextWrapped is crucial here with the defined width
    }
    ImGui::Spacing(); // Adds a small vertical space
    ImGui::TextWrapped("%s", currentQuestionLine.c_str());
    ImGui::PopStyleColor(); // Pop text color

    ImGui::End(); // End of "SpeakerInfo"
    ImGui::PopStyleVar(); // Pop WindowPadding



    // --- Dialogue Options Area (Chat Box) ---
    // Style similar to the screenshot:
    // - Positioned more towards the bottom-left, but not necessarily screen edge.
    // - Width determined by content, or a fixed moderate width.
    // - Height determined by content.

    float chatBoxWidth = screenWidth * 0.45f; // Estimate: 45% of screen width
    // float chatBoxMaxHeight = screenHeight * 0.4f; // Max height it can take up

    // Approximate position from screenshot (adjust these percentages based on your game's layout)
    // It seems to start about 5-10% from the left and maybe 60-70% from the top.
    ImVec2 chatBoxPos = ImVec2(screenWidth * 0.05f, screenHeight * 0.65f);

    ImGui::SetNextWindowPos(chatBoxPos, ImGuiCond_Appearing); // Set position once when it appears
    // SetNextWindowSizeConstraints can be useful to allow auto-resize by content
    // up to a certain maximum, or a fixed size.
    // For a box that grows with content like in the image:
    ImGui::SetNextWindowSize(ImVec2(chatBoxWidth, 0.0f), ImGuiCond_Appearing); // Width fixed, height auto (0.0f) on first appear

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(15.0f, 10.0f)); // Add some padding inside the window
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.08f, 0.85f)); // Darker, slightly bluish, semi-transparent

    ImGuiWindowFlags window_flags_1 = ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |       // User cannot resize
        ImGuiWindowFlags_NoMove |         // User cannot move
        ImGuiWindowFlags_NoScrollbar |    // No scrollbar (unless content overflows fixed size)
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_AlwaysAutoResize; // Key flag for height to fit content

    // If you want a fixed height and enable scrolling if content overflows:
    // ImGuiWindowFlags window_flags_1 = ImGuiWindowFlags_NoTitleBar |
    //                                 ImGuiWindowFlags_NoResize |
    //                                 ImGuiWindowFlags_NoMove |
    //                                 ImGuiWindowFlags_NoSavedSettings;
    // ImGui::SetNextWindowSize(ImVec2(chatBoxWidth, desiredFixedChatBoxHeight));

    ImGui::Begin("DialogueUI", nullptr, window_flags_1);

    // Inside the "DialogueUI" window

    // Assume:
    std::vector<std::string> dialogueOptions = {
         "You don't deploy with us?",
         "What kind of ship is this?",
         "No questions. I'm good to go.",
         "Goodbye."
     };
     int currentlySelectedOption = 3; // Example: "Goodbye." is pre-selected or focused

    // Spacing from the left edge, similar to the screenshot
    float indentAmount = screenWidth * 0.05f;
    ImGui::Indent(indentAmount);

    for (int i = 0; i < dialogueOptions.size(); ++i) {
        bool isSelected = (currentlySelectedOption == i);
        // ... (Push/Pop style colors as before) ...
        if (isSelected) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
        }
        else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
        }

        ImGui::PushID(i);
        ImGui::TextUnformatted("+"); // Or your icon
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x * 0.5f);

        if (ImGui::Selectable(dialogueOptions[i].c_str(), isSelected, ImGuiSelectableFlags_DontClosePopups)) {
            currentlySelectedOption = i;
            // ... Your game logic ...
        }
        ImGui::PopID();
        ImGui::PopStyleColor();
    }
   
    ImGui::Unindent(indentAmount);

    ImGui::End();


    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();

    return true;
}

void ChatBox::destroy() {
}

}//namespace ui
}//namespace engine
