// The look and the settings pages shared by the start screen and the F12 menu (runtime/ui/settings_ui.h). From Let's Go
// Recomp 0.6.0, in Sword's colours, with the pages cut to what this port has.
#define IMGUI_DEFINE_MATH_OPERATORS
#include "../nvn/vulkan/dlss.h"  // (dlss_available: the DLAA choice)
#include <SDL3/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "settings_ui.h"
#include "imgui_internal.h"

extern "C" {
#include "../paths.h"
#include "../settings.h"
}

// ------------------------------------------------------------------ language
int g_ui_lang;
void ui_pick_language(void) {
    const char *e = getenv("LARECOMP_LANG"), *lang = e && *e ? e : "";
    SDL_Locale **l = NULL;
    int n = 0;
    if (!*lang && (l = SDL_GetPreferredLocales(&n)) && n > 0 && l[0]->language) lang = l[0]->language;
    g_ui_lang = !strncmp(lang, "pt", 2) ? 1 : !strncmp(lang, "ru", 2) ? 2 : 0;
    SDL_free(l);
}

// ------------------------------------------------------------------ look
// In Sword's colours (Let's Go Recomp's look, after its game's Options screen, recoloured): a blue background with big
// pale ball rings, slow diagonal stripes and halftone dots; each setting a white pill with dark text, the one under the
// cursor light blue with a bobbing arrow at its left and < > around its value; a description box with a cool-toned
// frame. All of it drawn here from shapes and gradients: nothing of the game's.
const ImVec4 UI_GOLD(0.05f, 0.33f, 0.75f, 1), UI_GREEN_TEXT(0.07f, 0.48f, 0.27f, 1), UI_RED_TEXT(0.80f, 0.16f, 0.12f, 1),
    UI_TEXT(0.13f, 0.16f, 0.24f, 1), UI_HEADING(0.05f, 0.24f, 0.56f, 1);

UiFonts ui_load_fonts(void) {
    ImGuiIO &io = ImGui::GetIO();
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    static const ImWchar ranges[] = {0x20, 0xFF, 0x400, 0x45F, 0};  // (Latin-1: Portuguese accents; Cyrillic)
    UiFonts f = {};
    // (Segoe UI Semibold first: dark text on light pills, as the game's)
    const char *paths[] = {"C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                           "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"};
    for (const char *p : paths) {
        if (FILE *fp = fopen(p, "rb")) fclose(fp);  // (a missing file is an assertion in ImGui, not a NULL return)
        else continue;
        f.body = io.Fonts->AddFontFromFileTTF(p, 34, &cfg, ranges);
        f.big = io.Fonts->AddFontFromFileTTF(p, 56, &cfg, ranges);
        break;
    }
    char baloo[1024];
    if (rt_resource("Baloo2-SemiBold.ttf", "third_party/fonts/Baloo2/Baloo2-SemiBold.ttf", baloo, sizeof baloo))
        f.title = io.Fonts->AddFontFromFileTTF(baloo, 50, &cfg, ranges);
    if (!f.body) f.body = io.Fonts->AddFontDefault();
    if (!f.big) f.big = f.body;
    if (!f.title || g_ui_lang == 2) f.title = f.body;
    return f;
}

void ui_style(void) {
    ImGuiStyle &s = ImGui::GetStyle();
    ImGui::StyleColorsLight();
    // (rounding past half the height: pills)
    s.WindowRounding = 30, s.ChildRounding = 12, s.PopupRounding = 18, s.FrameRounding = 40, s.GrabRounding = 40;
    s.TabRounding = 40, s.ScrollbarRounding = 40;
    s.WindowPadding = ImVec2(32, 26), s.FramePadding = ImVec2(18, 10), s.ItemSpacing = ImVec2(14, 12);
    s.ItemInnerSpacing = ImVec2(10, 8), s.ScrollbarSize = 16, s.GrabMinSize = 26;
    s.WindowBorderSize = 0, s.ChildBorderSize = 0, s.FrameBorderSize = 2, s.PopupBorderSize = 3, s.TabBarBorderSize = 0;
    ImVec4 *c = s.Colors;
    // (the names stay Let's Go's: "yellow" is the cursor's colour, here Sword's light blue)
    ImVec4 white(1, 1, 1, 0.94f), yellow(0.42f, 0.80f, 1.0f, 1), yellow_hi(0.62f, 0.88f, 1.0f, 1), yellow_dark(0.24f, 0.66f, 0.96f, 1);
    c[ImGuiCol_Text] = UI_TEXT;
    c[ImGuiCol_TextDisabled] = ImVec4(0.27f, 0.37f, 0.52f, 1);
    c[ImGuiCol_WindowBg] = ImVec4(0.70f, 0.84f, 0.99f, 0.98f);  // (under ui_backdrop)
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = ImVec4(0.97f, 0.99f, 1, 0.98f);
    c[ImGuiCol_Border] = ImVec4(0.42f, 0.80f, 1.0f, 1);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = ImVec4(0.88f, 0.94f, 1.0f, 1);
    c[ImGuiCol_FrameBgHovered] = yellow_hi;
    c[ImGuiCol_FrameBgActive] = yellow;
    c[ImGuiCol_Button] = white;
    c[ImGuiCol_ButtonHovered] = yellow_hi;
    c[ImGuiCol_ButtonActive] = yellow_dark;
    c[ImGuiCol_Header] = yellow_hi;
    c[ImGuiCol_HeaderHovered] = ImVec4(0.74f, 0.92f, 1.0f, 1);
    c[ImGuiCol_HeaderActive] = yellow;
    c[ImGuiCol_CheckMark] = ImVec4(0.05f, 0.40f, 0.85f, 1);
    c[ImGuiCol_SliderGrab] = ImVec4(0.10f, 0.50f, 0.92f, 1);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.98f, 0.76f, 0.20f, 1);
    c[ImGuiCol_Tab] = white;
    c[ImGuiCol_TabHovered] = yellow_hi;
    c[ImGuiCol_TabSelected] = yellow;
    c[ImGuiCol_TabSelectedOverline] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TabDimmed] = c[ImGuiCol_Tab];
    c[ImGuiCol_TabDimmedSelected] = c[ImGuiCol_TabSelected];
    c[ImGuiCol_Separator] = c[ImGuiCol_SeparatorHovered] = c[ImGuiCol_SeparatorActive] = ImVec4(1, 1, 1, 0.75f);
    c[ImGuiCol_ScrollbarBg] = ImVec4(1, 1, 1, 0.35f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.06f, 0.24f, 0.52f, 1);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.09f, 0.32f, 0.66f, 1);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.04f, 0.18f, 0.42f, 1);
    c[ImGuiCol_PlotHistogram] = ImVec4(0.10f, 0.50f, 0.92f, 1);
    c[ImGuiCol_TableHeaderBg] = ImVec4(1, 1, 1, 0.5f);
    c[ImGuiCol_TableBorderStrong] = c[ImGuiCol_TableBorderLight] = ImVec4(1, 1, 1, 0.6f);
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1, 1, 1, 0.2f);
    c[ImGuiCol_TextSelectedBg] = ImVec4(0.42f, 0.80f, 1.0f, 0.5f);
    c[ImGuiCol_NavCursor] = ImVec4(0, 0, 0, 0);  // (the focus: the arrow, ui_frame_end, and the yellow row or button)
}

float ui_scale(UiScaler &s, float window_h) {
    float scale = window_h / 1080.0f;
    if (scale < 0.45f) scale = 0.45f;
    ImGui::GetIO().FontGlobalScale = scale;
    if (s.scale < 0) s.base = ImGui::GetStyle();
    if (fabsf(s.scale - scale) > 0.01f) {
        ImGui::GetStyle() = s.base;
        ImGui::GetStyle().ScaleAllSizes(scale);
        s.scale = scale;
    }
    return scale;
}

// ---- drawing
static ImU32 mix(ImU32 a, ImU32 b, float t) {
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    ImU32 r = 0;
    for (int sh = 0; sh < 32; sh += 8) {
        float x = (float)((a >> sh) & 0xFF), y = (float)((b >> sh) & 0xFF);
        r |= (ImU32)(x + (y - x) * t + 0.5f) << sh;
    }
    return r;
}
// The vertices from v0 on coloured by their position (f), keeping their alpha (the anti-aliased edges) times f's
template <class F> static void shade(ImDrawList *dl, int v0, F f) {
    for (int i = v0; i < dl->VtxBuffer.Size; i++) {
        ImDrawVert &v = dl->VtxBuffer[i];
        ImU32 c = f(v.pos), a = ((v.col >> IM_COL32_A_SHIFT) & 0xFF) * ((c >> IM_COL32_A_SHIFT) & 0xFF) / 255;
        v.col = (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
    }
}
static float ui_px(void) { return ImGui::GetIO().FontGlobalScale; }  // 1 at 1080 px

// A ball's outline (Let's Go Recomp had gears here): its rim, the band across it turned by `angle` (from the rim to the
// button's ring on each side), the button
static void ring(ImDrawList *dl, ImVec2 c, float r, float angle, ImU32 col) {
    float rim = r * 0.12f;
    dl->AddCircle(c, r * 0.9f, col, 0, rim);
    dl->AddCircle(c, r * 0.30f, col, 0, rim * 0.6f);
    dl->AddCircleFilled(c, r * 0.16f, col);
    ImVec2 u(cosf(angle), sinf(angle)), n(-u.y * r * 0.055f, u.x * r * 0.055f);
    for (int side = -1; side <= 1; side += 2) {
        float r0 = r * 0.36f, r1 = r * 0.9f - rim * 0.5f;
        ImVec2 p0(c.x + u.x * r0 * side, c.y + u.y * r0 * side), p1(c.x + u.x * r1 * side, c.y + u.y * r1 * side);
        ImVec2 q[4] = {ImVec2(p0.x + n.x, p0.y + n.y), ImVec2(p1.x + n.x, p1.y + n.y), ImVec2(p1.x - n.x, p1.y - n.y), ImVec2(p0.x - n.x, p0.y - n.y)};
        dl->AddConvexPolyFilled(q, 4, col);
    }
}

void ui_backdrop(ImDrawList *dl, ImVec2 a, ImVec2 b, float rounding) {
    float s = ui_px(), t = (float)ImGui::GetTime(), w = b.x - a.x, h = b.y - a.y;
    float r = fminf(rounding, fminf(w, h) * 0.5f);
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(a, b, IM_COL32_WHITE, r);
    shade(dl, v0, [&](ImVec2 p) {
        return mix(IM_COL32(222, 240, 255, 255), IM_COL32(112, 172, 244, 255), (p.x - a.x) / w * 0.35f + (p.y - a.y) / h * 0.65f);
    });
    // the decoration, clipped to the rounded shape: three rects, the corners left plain
    const ImVec4 clips[3] = {ImVec4(a.x + r, a.y, b.x - r, b.y), ImVec4(a.x, a.y + r, a.x + r, b.y - r), ImVec4(b.x - r, a.y + r, b.x, b.y - r)};
    float m = fminf(w, h), spin = t * 0.08f;
    for (const ImVec4 &cl : clips) {
        if (cl.z <= cl.x || cl.w <= cl.y) continue;
        dl->PushClipRect(ImVec2(cl.x, cl.y), ImVec2(cl.z, cl.w), true);
        // wide stripes slanting up to the right, sliding slowly
        float sw = 70 * s, period = sw * 2.6f, off = fmodf(t * 16 * s, period), slant = h * 0.55f;
        for (float x = a.x - slant - period + off; x < b.x; x += period)
            dl->AddQuadFilled(ImVec2(x + slant, a.y), ImVec2(x + slant + sw, a.y), ImVec2(x + sw, b.y), ImVec2(x, b.y), IM_COL32(255, 255, 255, 34));
        ring(dl, ImVec2(a.x + w * 0.05f, a.y + h * 0.10f), m * 0.22f, spin, IM_COL32(255, 255, 255, 90));
        ring(dl, ImVec2(b.x - w * 0.03f, a.y + h * 0.62f), m * 0.32f, -spin * 0.8f, IM_COL32(255, 255, 255, 72));
        ring(dl, ImVec2(a.x + w * 0.13f, b.y - h * 0.04f), m * 0.25f, spin * 0.9f + 0.3f, IM_COL32(255, 255, 255, 80));
        ring(dl, ImVec2(a.x + w * 0.78f, a.y + h * 0.02f), m * 0.15f, -spin * 1.3f, IM_COL32(236, 246, 255, 110));
        // halftone dots in the bottom left corner, smaller away from it
        float step = 22 * s, dw = w * 0.30f, dh = h * 0.45f;
        for (float y = b.y - step * 0.5f; y > b.y - dh; y -= step)
            for (float x = a.x + step * 0.5f; x < a.x + dw; x += step) {
                float dx = (x - a.x) / dw, dy = (b.y - y) / dh, d = sqrtf(dx * dx + dy * dy);
                if (d < 1) dl->AddCircleFilled(ImVec2(x, y), step * 0.30f * (1 - d), IM_COL32(255, 255, 255, 120), 10);
            }
        dl->PopClipRect();
    }
}

void ui_window_backdrop(void) {
    ImGuiWindow *w = ImGui::GetCurrentWindow();
    w->DrawList->PushClipRect(w->Pos, w->Pos + w->Size, false);
    ui_backdrop(w->DrawList, w->Pos, w->Pos + w->Size, ImGui::GetStyle().WindowRounding);
    w->DrawList->PopClipRect();
}

// green, cyan, blue, purple (pastel: towards white)
static ImU32 rainbow(float u, float pastel) {
    static const ImU32 st[4] = {IM_COL32(92, 210, 244, 255), IM_COL32(80, 160, 250, 255), IM_COL32(60, 110, 230, 255), IM_COL32(120, 110, 236, 255)};
    u = (u < 0 ? 0 : u > 1 ? 1 : u) * 3;
    int i = u >= 2 ? 2 : (int)u;
    return mix(mix(st[i], st[i + 1], u - i), IM_COL32_WHITE, pastel);
}

void ui_info_box(ImDrawList *dl, ImVec2 a, ImVec2 b) {
    float s = ui_px(), r = 24 * s, bw = 6 * s;
    dl->AddRectFilled(ImVec2(a.x, a.y + 5 * s), ImVec2(b.x, b.y + 5 * s), IM_COL32(10, 40, 100, 55), r);  // (shadow)
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(a, b, IM_COL32_WHITE, r);
    shade(dl, v0, [&](ImVec2 p) { return rainbow((p.x - a.x) / (b.x - a.x) * 0.8f + (p.y - a.y) / (b.y - a.y) * 0.2f, 0); });
    ImVec2 ia(a.x + bw, a.y + bw), ib(b.x - bw, b.y - bw);
    v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(ia, ib, IM_COL32_WHITE, r - bw * 0.6f);
    shade(dl, v0, [&](ImVec2 p) {
        float u = (p.x - ia.x) / (ib.x - ia.x), v = (p.y - ia.y) / (ib.y - ia.y);
        return rainbow(u * 0.8f + v * 0.2f, 0.62f + 0.30f * (1 - fabsf(u * 2 - 1)) * (1 - v * 0.5f));
    });
    dl->AddRect(ImVec2(ia.x + 2 * s, ia.y + 2 * s), ImVec2(ib.x - 2 * s, ib.y - 2 * s), IM_COL32(255, 255, 255, 210), r - bw, 0, 2.5f * s);
}

// A white pill, or the cursor's: yellow with a light rim (with the alpha of a disabled row)
void ui_pill(ImDrawList *dl, ImVec2 a, ImVec2 b, bool hot) {
    float r = (b.y - a.y) * 0.5f, s = ui_px();
    if (!hot) { dl->AddRectFilled(a, b, ImGui::GetColorU32(IM_COL32(255, 255, 255, 240)), r); return; }
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(a, b, ImGui::GetColorU32(IM_COL32_WHITE), r);
    shade(dl, v0, [&](ImVec2 p) { return mix(IM_COL32(170, 230, 255, 255), IM_COL32(84, 186, 252, 255), (p.y - a.y) / (b.y - a.y)); });
    dl->AddRect(ImVec2(a.x + 1.5f * s, a.y + 1.5f * s), ImVec2(b.x - 1.5f * s, b.y - 1.5f * s), ImGui::GetColorU32(IM_COL32(225, 245, 255, 255)), r, 0, 3 * s);
}

// The cursor: a white, shaded arrow with a dark outline pointing at tip, bobbing left and right
static void cursor_arrow(ImDrawList *dl, ImVec2 tip, float h) {
    tip.x -= (0.5f + 0.5f * sinf((float)ImGui::GetTime() * 7)) * h * 0.20f;
    float w = h * 0.95f;
    ImVec2 p[4] = {tip, ImVec2(tip.x - w, tip.y - h * 0.5f), ImVec2(tip.x - w * 0.70f, tip.y), ImVec2(tip.x - w, tip.y + h * 0.5f)};
    int v0 = dl->VtxBuffer.Size;
    dl->AddConcavePolyFilled(p, 4, IM_COL32_WHITE);
    shade(dl, v0, [&](ImVec2 q) { return mix(IM_COL32(255, 255, 255, 255), IM_COL32(196, 198, 204, 255), (q.y - p[1].y) / h); });
    dl->AddPolyline(p, 4, IM_COL32(78, 78, 78, 255), ImDrawFlags_Closed, h * 0.09f);
}

// ---- the cursor: the row under it (keyboard / controller focus, or the mouse) and its arrow
static bool g_hot_set, g_row_hot;  // a row has the cursor this frame / the row being drawn has it
static ImRect g_hot_row;
static bool nav_rect(ImRect *out) {
    ImGuiContext &g = *GImGui;
    if (!g.NavCursorVisible || !g.NavId || !g.NavWindow) return false;
    *out = ImGui::WindowRectRelToAbs(g.NavWindow, g.NavWindow->NavRectRel[g.NavLayer]);
    return true;
}
static bool rect_hot(const ImRect &r) {
    ImGuiContext &g = *GImGui;
    ImRect n;
    if (nav_rect(&n) && g.NavWindow->RootWindowForNav == ImGui::GetCurrentWindow()->RootWindowForNav && r.Contains(n.GetCenter()))
        return true;
    return !g.NavHighlightItemUnderNav && ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(r.Min, r.Max);
}

// ---- descriptions
static const char *g_tip_now, *g_tip_last;
bool g_ui_tips_in_box;
void ui_tip(const char *text) {
    bool hov = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
    if (hov || ImGui::IsItemFocused()) g_tip_now = text;
    if (hov && !g_ui_tips_in_box) ImGui::SetTooltip("%s", text);
}
const char *ui_tip_last(void) { return g_tip_last; }

void ui_frame_end(void) {
    ImGuiContext &g = *GImGui;
    ImDrawList *dl = ImGui::GetForegroundDrawList();
    float h = ImGui::GetFrameHeight() * 0.62f;
    ImRect n;
    // (its tip on the pill's left end, before the text column)
    if (g_hot_set) cursor_arrow(dl, ImVec2(g_hot_row.Min.x + ui_inset() * 0.6f, g_hot_row.GetCenter().y), h);
    else if (nav_rect(&n) && g.NavIdIsAlive && g.NavWindow->Active) cursor_arrow(dl, ImVec2(n.Min.x - 4 * ui_px(), n.GetCenter().y), h);
    g_hot_set = false;
    g_tip_last = g_tip_now, g_tip_now = NULL;
}

void ui_title(ImFont *title, const char *text) {
    ImGui::PushFont(title);
    ImGui::TextColored(UI_TEXT, "%s", text);
    ImGui::PopFont();
}

void ui_heading(const char *text) {
    if (ImGui::GetCursorPosY() > ImGui::GetCursorStartPos().y + 1) ImGui::Spacing();  // (a group's gap above it, not at the top)
    ImGui::TextColored(UI_HEADING, "%s", text);
}

// ---- the grid (settings_ui.h)
float ui_inset(void) { return roundf(ImGui::GetFrameHeight() * 0.45f); }

void ui_page_begin(const char *id, ImVec2 size) {
    // (no side padding: the pills reach the page's edges; a little above and below for the cursor row's rim)
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, ImGui::GetStyle().ItemSpacing.y * 0.5f));
    ImGui::BeginChild(id, size, ImGuiChildFlags_NavFlattened | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar();
    ImGui::Indent(ui_inset());
}
void ui_page_end(void) {
    ImGui::Unindent(ui_inset());
    ImGui::EndChild();
}

void ui_group_gap(void) { ImGui::Spacing(); }

// The pill of the row on the current line: from the page's left edge (the text column less the inset) to its right one
static ImRect row_rect(void) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    return ImRect(ImVec2(p.x - ui_inset(), p.y), ImVec2(ImGui::GetCurrentWindow()->WorkRect.Max.x, p.y + ImGui::GetFrameHeight()));
}

bool ui_hot(ImVec2 a, ImVec2 b) {
    ImRect r(a, b);
    bool hot = rect_hot(r);
    if (hot) g_hot_set = true, g_hot_row = r;
    return hot;
}

bool ui_pill_row(void) {
    ImRect r = row_rect();
    g_row_hot = ui_hot(r.Min, r.Max);
    ui_pill(ImGui::GetWindowDrawList(), r.Min, r.Max, g_row_hot);
    return g_row_hot;
}

// The values' column: the pill's right half, so it stays put from one page to the next; after the longest label when
// that doesn't fit (long translations)
float ui_label_x(const char *const *labels, int n) {
    float w = 0;
    for (int i = 0; i < n; i++) w = fmaxf(w, ImGui::CalcTextSize(labels[i]).x);
    ImGuiWindow *win = ImGui::GetCurrentWindow();
    float in = ui_inset(), x0 = ImGui::GetCursorScreenPos().x - in;
    float half = x0 + (win->WorkRect.Max.x - x0) * 0.5f, fit = x0 + in + w + ImGui::GetStyle().ItemSpacing.x * 2;
    return fmaxf(half, fit) - win->Pos.x;
}

// The row's pill, yellow when it has the cursor; its label on the text column
void ui_row(const char *label, float x) {
    ui_pill_row();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(x);
    ImGui::SetNextItemWidth(-ui_inset() * 0.6f);
}

bool ui_button(const char *label, float width) {
    ImGuiWindow *win = ImGui::GetCurrentWindow();
    float in = ui_inset(), h = ImGui::GetFrameHeight();
    const char *end = ImGui::FindRenderedTextEnd(label);
    ImVec2 ts = ImGui::CalcTextSize(label, end), p = ImGui::GetCursorScreenPos();
    float x0 = win->DC.IsSameLine ? p.x : p.x - in;  // (beside another item: where it ends, plus the spacing)
    float w = width < 0 ? win->WorkRect.Max.x - x0 : width > 0 ? width : ts.x + in * 2;
    ImGui::SetCursorScreenPos(ImVec2(x0, p.y));
    bool hit = ImGui::InvisibleButton(label, ImVec2(w, h));
    ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    bool hot = ui_hot(a, b) || ImGui::IsItemActive();
    ui_pill(win->DrawList, a, b, hot);
    if (!hot) win->DrawList->AddRect(a, b, ImGui::GetColorU32(ImGuiCol_Border), h * 0.5f, 0, 2 * ui_px());  // (a button, not a row)
    win->DrawList->AddText(ImVec2(a.x + in, a.y + (h - ts.y) * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), label, end);
    return hit;
}

// (left / right on a focused item, not being dragged: a step of its value instead of a move to another item)
static int lr_step(void) {
    if (!ImGui::IsItemFocused() || ImGui::IsItemActive() || (GImGui->CurrentItemFlags & ImGuiItemFlags_Disabled)) return 0;
    bool l = ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) || ImGui::IsKeyPressed(ImGuiKey_GamepadLStickLeft) || ImGui::IsKeyPressed(ImGuiKey_LeftArrow);
    bool r = ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) || ImGui::IsKeyPressed(ImGuiKey_GamepadLStickRight) || ImGui::IsKeyPressed(ImGuiKey_RightArrow);
    if (l || r) ImGui::NavMoveRequestCancel();
    return l == r ? 0 : l ? -1 : 1;
}

bool ui_slider(const char *id, float *v, float lo, float hi, const char *fmt, float step) {
    ImGuiWindow *win = ImGui::GetCurrentWindow();
    float old = *v, h = ImGui::GetFrameHeight(), in = ui_inset();
    ImVec2 p = ImGui::GetCursorScreenPos();
    char txt[32], widest[32];
    snprintf(txt, sizeof txt, fmt, *v);
    // (room for the longest value, at least a "000%": the tracks of a page as long as each other)
    snprintf(widest, sizeof widest, fmt, fabsf(hi) > fabsf(lo) ? hi : lo);
    float vw = fmaxf(fmaxf(ImGui::CalcTextSize(widest).x, ImGui::CalcTextSize(txt).x), ImGui::CalcTextSize("000%").x);
    float end = win->WorkRect.Max.x - in;
    // ImGui's slider with nothing of its own drawn: it takes the clicks and drags, the track is drawn below
    ImGui::PushStyleColor(ImGuiCol_FrameBg, 0);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, 0);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, 0);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, 0);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0);
    ImGui::SetNextItemWidth(fmaxf(end - vw - in - p.x, h * 2));
    ImGui::SliderFloat(id, v, lo, hi, "", ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(5);
    if (int s = lr_step()) *v = fminf(hi, fmaxf(lo, *v + s * step));
    // the track (ImGui's own geometry: the knob where a click puts it), its filled part and the knob; the value after it
    ImDrawList *dl = win->DrawList;
    ImRect bb(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    float grab = ImGui::GetStyle().GrabMinSize, x0 = bb.Min.x + 2 + grab * 0.5f, x1 = bb.Max.x - 2 - grab * 0.5f;
    float t = hi > lo ? (*v - lo) / (hi - lo) : 0, kx = x0 + (x1 - x0) * t, cy = bb.GetCenter().y, th = h * 0.18f, kr = h * 0.27f;
    dl->AddRectFilled(ImVec2(x0 - th * 0.5f, cy - th * 0.5f), ImVec2(x1 + th * 0.5f, cy + th * 0.5f), ImGui::GetColorU32(IM_COL32(200, 220, 240, 255)), th);
    dl->AddRectFilled(ImVec2(x0 - th * 0.5f, cy - th * 0.5f), ImVec2(kx, cy + th * 0.5f), ImGui::GetColorU32(ImGuiCol_PlotHistogram), th);
    dl->AddCircleFilled(ImVec2(kx, cy), kr, ImGui::GetColorU32(IM_COL32_WHITE));
    dl->AddCircle(ImVec2(kx, cy), kr, ImGui::GetColorU32(ImGui::IsItemActive() ? ImGuiCol_SliderGrabActive : ImGuiCol_PlotHistogram), 0, h * 0.07f);
    snprintf(txt, sizeof txt, fmt, *v);
    ImVec2 ts = ImGui::CalcTextSize(txt);
    dl->AddText(ImVec2(end - ts.x, cy - ts.y * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), txt);
    return *v != old;
}

bool ui_value_button(const char *id, const char *text, ImU32 col) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::CalcItemWidth(), h = ImGui::GetFrameHeight();
    bool hit = ImGui::InvisibleButton(id, ImVec2(w, h));
    ImVec2 ts = ImGui::CalcTextSize(text);
    ImGui::GetWindowDrawList()->AddText(ImVec2(p.x + (w - ts.x) * 0.5f, p.y + (h - ts.y) * 0.5f),
                                        col ? col : ImGui::GetColorU32(ImGuiCol_Text), text);
    return hit;
}

// A choice as in the game: its value centred, < > around it when the row has the cursor; left / right (d-pad, stick,
// arrow keys) or a click on < > step through the choices, A / Enter / a click on the value opens the list
bool ui_combo(const char *id, int *v, const char *const *items, int n) {
    ImGuiContext &g = *GImGui;
    int old = *v;
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::CalcItemWidth(), h = ImGui::GetFrameHeight(), cw = h * 0.9f;
    ImRect fr(p, ImVec2(p.x + w, p.y + h)), left(p, ImVec2(p.x + cw, p.y + h)), right(ImVec2(fr.Max.x - cw, p.y), fr.Max);
    bool disabled = (g.CurrentItemFlags & ImGuiItemFlags_Disabled) != 0, hot = g_row_hot;
    int step = 0;
    if (!disabled && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
        step = ImGui::IsMouseHoveringRect(left.Min, left.Max) ? -1 : ImGui::IsMouseHoveringRect(right.Min, right.Max) ? 1 : 0;
    ImGui::SetCursorScreenPos(ImVec2(p.x + cw, p.y));
    ImGui::SetNextItemWidth(w - cw * 2);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, 0);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, 0);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0);
    bool open = ImGui::BeginCombo(id, "", ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);
    if (open) {
        for (int i = 0; i < n; i++) {
            bool sel = i == *v;
            if (ImGui::Selectable(items[i], sel)) *v = i;
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    } else if (!disabled && ImGui::IsItemFocused()) {
        bool l = ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) || ImGui::IsKeyPressed(ImGuiKey_GamepadLStickLeft) || ImGui::IsKeyPressed(ImGuiKey_LeftArrow);
        bool r = ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) || ImGui::IsKeyPressed(ImGuiKey_GamepadLStickRight) || ImGui::IsKeyPressed(ImGuiKey_RightArrow);
        if (l != r) step = l ? -1 : 1;
        if (l || r) ImGui::NavMoveRequestCancel();  // (a change of the value, not a move to another item)
    }
    if (step && n > 0) *v = (*v + step + n) % n;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const char *txt = *v >= 0 && *v < n ? items[*v] : "";
    ImVec2 ts = ImGui::CalcTextSize(txt);
    dl->AddText(ImVec2(fr.GetCenter().x - ts.x * 0.5f, fr.GetCenter().y - ts.y * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), txt);
    if (hot && !disabled) {  // < >: chunky, white with a dark outline, as the game's
        float ch = h * 0.32f;
        for (int k = 0; k < 2; k++) {
            float x = k ? right.GetCenter().x : left.GetCenter().x, y = fr.GetCenter().y, d = ch * (k ? 0.8f : -0.8f), t = ch * (k ? 0.55f : -0.55f);
            ImVec2 pts[6] = {ImVec2(x - d * 0.5f, y - ch), ImVec2(x - d * 0.5f + t, y - ch), ImVec2(x + d * 0.5f + t, y),
                             ImVec2(x - d * 0.5f + t, y + ch), ImVec2(x - d * 0.5f, y + ch), ImVec2(x + d * 0.5f, y)};
            dl->AddConcavePolyFilled(pts, 6, IM_COL32(255, 255, 255, 255));
            dl->AddPolyline(pts, 6, IM_COL32(90, 90, 90, 255), ImDrawFlags_Closed, h * 0.05f);
        }
    }
    return *v != old;
}

// An on / off setting as a row with its value, as the game's (Sí / No)
bool ui_toggle(const char *id, const char *label, float x, bool *v) {
    const char *oo[] = {UI_T("Off", "Desligado", "Выкл."), UI_T("On", "Ligado", "Вкл.")};
    int i = *v;
    ui_row(label, x);
    bool changed = ui_combo(id, &i, oo, 2);
    *v = i != 0;
    return changed;
}

// ------------------------------------------------------------------ tabs
void ui_tabs_begin(int *current, int count) {
    int step = ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) ? -1 : ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) ? 1 : 0;
    if (step && count > 0) *current = (*current + step + count) % count;
}

// ------------------------------------------------------------------ the graphics page
// (Let's Go Recomp's page also has GI, the shadow filter, TAA, FSR 3.1 and frame generation: they read that game's
// scene passes and motion vectors, not ported to Sword's renderer; its SSAO is: runtime/nvn/vulkan/exec.c ssao_run)
static const char *const *graphics_labels(int *n) {
    static const char *l[6];
    l[0] = UI_T("Frame rate", "Taxa de quadros", "Частота кадров");
    l[1] = UI_T("Anti-aliasing", "Antisserrilhado", "Сглаживание");
    l[2] = UI_T("Anisotropic filtering", "Filtro anisotrópico", "Анизотропная фильтрация");
    l[3] = UI_T("Show FPS", "Mostrar FPS", "Показывать FPS");
    l[4] = UI_T("Shadow resolution", "Resolução das sombras", "Разрешение теней");
    l[5] = UI_T("Ambient occlusion", "Oclusão de ambiente", "Затенение окружения");
    *n = 6;
    return l;
}

float ui_graphics_x(const char *const *extra, int n_extra) {
    int n;
    const char *const *l = graphics_labels(&n);
    float x = ui_label_x(l, n);
    return extra && n_extra ? fmaxf(x, ui_label_x(extra, n_extra)) : x;
}

// known: the build is one runtime/hle/swsh60.c has the frame time table of (Sword 1.0.0 and 1.3.2)
void ui_page_graphics(float x, bool known) {
    int n;
    const char *const *l = graphics_labels(&n);
    {   // the whole game at 30 (the game's own) or 60 (runtime/hle/swsh60.c, from the next start: the F12 menu's change
        // waits for it). No "unlimited": the game advances a fixed step a frame, so above 60 it runs fast.
        const char *fr[] = {UI_T("30 fps (original)", "30 fps (original)", "30 fps (оригинал)"), "60 fps"};
        int f = known && g_settings.fps == 60 ? 1 : 0;
        ui_row(l[0], x);
        ImGui::BeginDisabled(!known);
        if (ui_combo("##fps", &f, fr, 2)) g_settings.fps = f ? 60 : 30;
        ImGui::EndDisabled();
        ui_tip(!known ? UI_T("Only with Sword 1.3.2 (or 1.0.0) for now.", "Por enquanto só no Sword 1.3.2 (ou 1.0.0).",
                             "Пока только Sword 1.3.2 (или 1.0.0).")
                      : UI_T("The whole game at 60 fps, like the community's 60 fps patch. Takes effect the next time the game starts.",
                             "O jogo todo a 60 fps, como o patch de 60 fps da comunidade. Vale na próxima vez que o jogo abrir.",
                             "Вся игра в 60 к/с, как патч сообщества. Применяется при следующем запуске."));
    }
    ui_group_gap();
    {
        // (DLAA only where NGX came up at start: an NVIDIA RTX driver and nvngx_dlss.dll next to the executable)
        bool dl = dlss_available();
        const char *aa[] = {UI_T("Off", "Desligado", "Выкл."), "SMAA", "TAA", "DLAA (NVIDIA)"};
        static const int mode[] = {AA_OFF, AA_SMAA, AA_TAA, AA_DLAA};
        int v = 0;
        for (int i = 0; i < 4; i++) if (mode[i] == g_settings.antialias) v = i;
        if (v == 3 && !dl) v = 2;  // (DLAA chosen, not available here: TAA runs)
        ui_row(l[1], x);
        if (ui_combo("##aa", &v, aa, dl ? 4 : 3)) g_settings.antialias = mode[v];
        ui_tip(UI_T("Smooths jagged edges. SMAA: on the final image, light. TAA: blends the last frames along the camera's "
                    "motion (the cleanest edges; foliage and thin lines stop shimmering). DLAA: NVIDIA's AI in TAA's place, "
                    "the best (RTX cards).",
                    "Suaviza os serrilhados. SMAA: na imagem final, leve. TAA: junta os últimos quadros seguindo o movimento "
                    "da câmera (bordas bem mais limpas; folhagem e linhas finas param de cintilar). DLAA: a IA da NVIDIA no "
                    "lugar do TAA, o melhor (placas RTX).",
                    "Сглаживает зубчатые края. SMAA: на итоговом кадре. TAA: смешивает последние кадры по движению камеры. "
                    "DLAA: ИИ NVIDIA вместо TAA (карты RTX)."));
    }
    {   // (runtime/nvn/vulkan/textures.c: read once, the samplers are kept)
        const char *af[] = {UI_T("The game's", "Do jogo", "Как в игре"), "2x", "4x", "8x", "16x"};
        static const int lv[] = {0, 2, 4, 8, 16};
        int v = 0;
        for (int i = 0; i < 5; i++) if (lv[i] == g_settings.aniso) v = i;
        ui_row(l[2], x);
        if (ui_combo("##aniso", &v, af, 5)) g_settings.aniso = lv[v];
        ui_tip(UI_T("Keeps textures sharp at grazing angles (paths, grass towards the horizon). Almost free on a dedicated "
                    "GPU. Takes effect the next time the game starts.",
                    "Mantém as texturas nítidas vistas de lado (caminhos, grama no horizonte). Quase de graça numa GPU "
                    "dedicada. Vale na próxima vez que o jogo abrir.",
                    "Сохраняет чёткость текстур под острым углом. Применяется при следующем запуске."));
    }
    {   // (runtime/nvn/vulkan/textures.c shadow_k: the map is reallocated at once)
        const char *sr[] = {UI_T("The game's", "Do jogo", "Как в игре"), "2x", "4x"};
        static const int lv[] = {1, 2, 4};
        int v = g_settings.shadow_res >= 4 ? 2 : g_settings.shadow_res >= 2 ? 1 : 0;
        ui_row(l[4], x);
        if (ui_combo("##shadowres", &v, sr, 3)) g_settings.shadow_res = lv[v];
        ui_tip(UI_T("The sun's shadow map drawn 2 or 4 times as large: shadows with sharper, steadier edges (the game's "
                    "is 1024 per cascade). 4x uses ~128 MB more video memory.",
                    "O mapa de sombra do sol desenhado 2 ou 4 vezes maior: sombras com bordas mais nítidas e estáveis (o do "
                    "jogo tem 1024 por cascata). 4x usa ~128 MB a mais de memória de vídeo.",
                    "Карта теней солнца в 2 или 4 раза больше: более чёткие края теней. 4x — ещё ~128 МБ видеопамяти."));
    }
    {   // (runtime/nvn/vulkan/exec.c ssao_run, live; settings_apply_levels: the samples per level)
        const char *ao[] = {UI_T("Off", "Desligado", "Выкл."), UI_T("Low", "Baixa", "Низкое"), UI_T("Medium", "Média", "Среднее"),
                            UI_T("High", "Alta", "Высокое")};
        int v = g_settings.ssao_level < FX_OFF || g_settings.ssao_level > FX_HIGH ? FX_HIGH : g_settings.ssao_level;
        ui_row(l[5], x);
        if (ui_combo("##ssao", &v, ao, 4)) g_settings.ssao_level = v;
        ui_tip(UI_T("SSAO: soft shading where objects meet and in corners (under characters, along walls, between the "
                    "houses' details). Low / medium / high: 8 / 12 / 16 samples a pixel.",
                    "SSAO: sombreado suave onde os objetos se encontram e nos cantos (sob os personagens, junto às paredes, "
                    "nos detalhes das casas). Baixa / média / alta: 8 / 12 / 16 amostras por pixel.",
                    "SSAO: мягкое затенение в углах и местах соприкосновения объектов."));
    }
    ui_toggle("##showfps", l[3], x, &g_settings.show_fps);
    ui_tip(UI_T("The frame rate in a corner of the screen.", "A taxa de quadros num canto da tela.", "Частота кадров в углу экрана."));
    settings_apply_levels();
}

// ------------------------------------------------------------------ image and sound
// Volume and the game image's colour, applied live in game (runtime/services/audio.c apply_gain,
// runtime/nvn/vulkan/exec.c post_process, color.frag), and the TAA sharpening at the present
void ui_page_image_sound(void) {
    const char *l_vol = UI_T("Volume", "Volume", "Громкость"), *l_bri = UI_T("Brightness", "Brilho", "Яркость"),
               *l_con = UI_T("Contrast", "Contraste", "Контраст"), *l_sat = UI_T("Saturation", "Saturação", "Насыщенность");
    const char *labels[] = {l_vol, l_bri, l_con, l_sat};
    float x = ui_label_x(labels, 4);
    struct Row { const char *label; float *v, lo, hi; } rows[] = {
        {l_vol, &g_settings.volume, 0.0f, 2.0f}, {l_bri, &g_settings.brightness, 0.5f, 1.5f},
        {l_con, &g_settings.contrast, 0.5f, 1.5f}, {l_sat, &g_settings.saturation, 0.0f, 2.0f}};
    for (int i = 0; i < 4; i++) {
        if (i == 1) ui_group_gap();
        float pct = *rows[i].v * 100;
        ui_row(rows[i].label, x);
        ImGui::PushID(100 + i);
        if (ui_slider("##v", &pct, rows[i].lo * 100, rows[i].hi * 100, "%.0f%%", 5)) *rows[i].v = pct / 100;
        ImGui::PopID();
    }
    ui_group_gap();
    if (ui_button(UI_T("Reset", "Restaurar", "Сбросить"))) {
        g_settings.volume = g_settings.brightness = g_settings.contrast = g_settings.saturation = 1.0f;
    }
    settings_clamp_color();
}

// ------------------------------------------------------------------ advanced
// (Let's Go Recomp's page: its motion vectors and SSAO / GI debug views; nothing of that here)
void ui_page_advanced(void) {}
