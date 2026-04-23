#pragma once
// ------------------------------------------------------------------
// Lightweight title-screen configuration loaded from an XML file.
//
// The parser is intentionally minimal — it only understands the
// specific element/attribute schema used by title_screen.xml and is
// NOT a general-purpose XML parser.  This keeps the engine free of
// external XML library dependencies.
// ------------------------------------------------------------------

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>

namespace engine {
namespace ui {

// ---- Data model ---------------------------------------------------

struct TitleMenuItem {
    std::string label;   // displayed text
    std::string action;  // engine-interpreted action id
};

struct TitleScreenConfig {
    // Background
    std::string background_image;  // e.g. "assets/ui/fantasy_bg.png"

    // Top bar
    std::string title;
    std::string subtitle;
    std::string version;
    float       top_bar_height_extra = 72.0f;

    // Menu items
    std::vector<TitleMenuItem> menu_items;

    // New-game mesh list
    std::vector<std::string> new_game_meshes;

    bool loaded = false;
};

// ---- Minimal XML helpers ------------------------------------------

namespace detail {

inline std::string trim(const std::string& s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Extract the value of an attribute from an XML tag string.
// e.g. attr(R"(<item label="Foo" action="bar"/>)", "label") -> "Foo"
inline std::string attr(const std::string& tag, const char* name) {
    std::string key = std::string(name) + "=\"";
    auto pos = tag.find(key);
    if (pos == std::string::npos) return "";
    pos += key.size();
    auto end = tag.find('"', pos);
    if (end == std::string::npos) return "";
    return tag.substr(pos, end - pos);
}

// Simple float attribute with default.
inline float attr_f(const std::string& tag, const char* name, float def) {
    std::string v = attr(tag, name);
    if (v.empty()) return def;
    try { return std::stof(v); } catch (...) { return def; }
}

} // namespace detail

// ---- Loader -------------------------------------------------------

inline TitleScreenConfig loadTitleScreenConfig(const char* path) {
    TitleScreenConfig cfg;

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "[title_screen] Could not open %s\n", path);
        return cfg;
    }

    // Slurp the whole file.
    std::string xml((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());

    // --- background ---
    {
        auto pos = xml.find("<background ");
        if (pos != std::string::npos) {
            auto end = xml.find("/>", pos);
            if (end != std::string::npos) {
                std::string tag = xml.substr(pos, end - pos + 2);
                cfg.background_image = detail::attr(tag, "image");
            }
        }
    }

    // --- top_bar ---
    {
        auto pos = xml.find("<top_bar ");
        if (pos != std::string::npos) {
            auto end = xml.find(">", pos);
            if (end != std::string::npos) {
                std::string tag = xml.substr(pos, end - pos + 1);
                cfg.top_bar_height_extra =
                    detail::attr_f(tag, "height_extra", 72.0f);
            }
        }
    }

    // --- title ---
    {
        auto pos = xml.find("<title ");
        if (pos != std::string::npos) {
            auto end = xml.find("/>", pos);
            if (end == std::string::npos)
                end = xml.find(">", pos);
            if (end != std::string::npos) {
                std::string tag = xml.substr(pos, end - pos + 2);
                cfg.title    = detail::attr(tag, "text");
                cfg.subtitle = detail::attr(tag, "subtitle");
                cfg.version  = detail::attr(tag, "version");
            }
        }
    }

    // --- menu_items ---
    {
        std::string::size_type search = 0;
        while (true) {
            auto pos = xml.find("<item ", search);
            if (pos == std::string::npos) break;
            auto end = xml.find("/>", pos);
            if (end == std::string::npos) break;
            std::string tag = xml.substr(pos, end - pos + 2);
            TitleMenuItem mi;
            mi.label  = detail::attr(tag, "label");
            mi.action = detail::attr(tag, "action");
            if (!mi.label.empty())
                cfg.menu_items.push_back(mi);
            search = end + 2;
        }
    }

    // --- new_game_meshes ---
    {
        auto block_start = xml.find("<new_game_meshes>");
        auto block_end   = xml.find("</new_game_meshes>");
        if (block_start != std::string::npos &&
            block_end   != std::string::npos) {
            std::string block = xml.substr(block_start,
                                           block_end - block_start);
            std::string::size_type search = 0;
            while (true) {
                auto pos = block.find("<mesh ", search);
                if (pos == std::string::npos) break;
                auto end = block.find("/>", pos);
                if (end == std::string::npos) break;
                std::string tag = block.substr(pos, end - pos + 2);
                std::string p = detail::attr(tag, "path");
                if (!p.empty())
                    cfg.new_game_meshes.push_back(p);
                search = end + 2;
            }
        }
    }

    cfg.loaded = true;
    fprintf(stderr, "[title_screen] Loaded %s  (%zu menu items, %zu meshes)\n",
            path,
            cfg.menu_items.size(),
            cfg.new_game_meshes.size());
    return cfg;
}

} // namespace ui
} // namespace engine
