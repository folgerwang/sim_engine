#include <vector>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "audio/tts_engine.h"   // speak NPC dialog lines

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
    const float& delta_t,
    const glm::vec2& vp_origin,
    const glm::vec2& vp_size) {

    // In your per-frame UI construction section:

    // Positioning frame: the editor Viewport rect when supplied, else the
    // full main viewport / DisplaySize.  All dialogue positions below are
    // expressed relative to (vpPos, screenWidth/Height) so passing the
    // viewport rect keeps the chat inside the 3D viewport in editor mode.
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 vpPos;
    float screenWidth, screenHeight;
    if (vp_size.x > 1.0f && vp_size.y > 1.0f) {
        vpPos = ImVec2(vp_origin.x, vp_origin.y);
        screenWidth = vp_size.x; screenHeight = vp_size.y;
    } else {
        vpPos = ImGui::GetMainViewport()->Pos;
        screenWidth = io.DisplaySize.x; screenHeight = io.DisplaySize.y;
    }

    // --- Speaker Name and Dialogue Text ---
    float speakerWindowPosX = vpPos.x + screenWidth * 0.15f;
    float speakerWindowPosY = vpPos.y + screenHeight * 0.1f;
    float speakerWindowWidth = screenWidth * 0.45f;

    ImGui::SetNextWindowPos(ImVec2(speakerWindowPosX, speakerWindowPosY), ImGuiCond_Always);
    // Set a specific width, height will be determined by content due to AlwaysAutoResize
    ImGui::SetNextWindowSize(ImVec2(speakerWindowWidth, 0.0f), ImGuiCond_Always); // Using 0.0f for height + AlwaysAutoResize

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f)); // Add some internal padding
    ImGui::SetNextWindowBgAlpha(0.0f); // Transparent background as in the image (or a very subtle dark background)

    ImGuiWindowFlags speaker_window_flags = ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_AlwaysAutoResize; // Important for height

    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
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

    // ── Text-to-voice ────────────────────────────────────────────────────
    // Speak the NPC's lines once each time they CHANGE (this UI redraws
    // every frame, so keying on the concatenated text is what gives
    // "say each new line once" semantics).  TtsEngine synthesizes on a
    // worker thread and plays on the voice bus; a no-op when the
    // sherpa-onnx backend or the voice model is absent.
    {
        std::string spoken;
        for (const auto& line : currentDialogueLines) {
            spoken += line;
            spoken += ' ';
        }
        spoken += currentQuestionLine;
        static std::string s_last_spoken;
        if (spoken != s_last_spoken) {
            s_last_spoken = spoken;
            engine::audio::TtsEngine::speak(spoken);
        }
    }

    ImGui::End(); // End of "SpeakerInfo"
    ImGui::PopStyleVar(); // Pop WindowPadding



    // --- Dialogue Options Area (Chat Box) ---
    // Style similar to the screenshot:
    // - Positioned more towards the bottom-left, but not necessarily screen edge.
    // - Width determined by content, or a fixed moderate width.
    // - Height determined by content.

    float chatBoxWidth = screenWidth * 0.45f; // Estimate: 45% of screen width

    // Anchor the dialog box from the bottom-left so it grows upward as options
    // are added, rather than overflowing off the bottom of the screen.
    float bottomMargin = screenHeight * 0.05f;  // 5% margin from screen bottom
    float leftMargin   = screenWidth * 0.05f;

    // Use pivot (0, 1) = bottom-left corner anchored at the position.
    ImVec2 chatBoxPos = ImVec2(vpPos.x + leftMargin, vpPos.y + screenHeight - bottomMargin);
    ImGui::SetNextWindowPos(chatBoxPos, ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowSize(ImVec2(chatBoxWidth, 0.0f), ImGuiCond_Always); // Width fixed, height auto

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(15.0f, 10.0f)); // Add some padding inside the window
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.08f, 0.85f)); // Darker, slightly bluish, semi-transparent

    ImGuiWindowFlags window_flags_1 = ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |       // User cannot resize
        ImGuiWindowFlags_NoMove |         // User cannot move
        ImGuiWindowFlags_NoScrollbar |    // No scrollbar (unless content overflows fixed size)
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_AlwaysAutoResize; // Key flag for height to fit content

    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
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
