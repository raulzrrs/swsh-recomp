// The mouse in Sword's menus. Sword's menus are Nintendo ui2d layouts (bflyt) with Game Freak's UIKit on top, made for
// a controller only (the game reads no touch screen). Every pane of every layout goes through one function each frame
// (PANE_CALC, the panes' common virtual 19, found with runtime/hle/uiprobe.c); the hook keeps the panes of each frame,
// and at the next frame they become a snapshot: the menu's buttons (parts panes "L_...", where each is on the screen)
// and the game's own cursor (the picture "P_cursor_..." of its cursor layout, the hand / arrow beside the selected
// button). A click on a button walks the selection there with the d-pad (one step, look where the cursor went, the
// next) and presses A once it is there; a click elsewhere is A. A frame with that cursor is a menu: the mouse is a
// pointer there, and turns the camera elsewhere (runtime/hle/input.c).
//
// ui2d Pane (this build): +0x00 vtable, +0x08 / +0x10 sibling links, +0x18 parent, +0x20 children, +0x30 translate,
// +0x3C rotate, +0x48 scale, +0x50 size (w, h), +0x58 flags (bit 0 visible), +0x59 alpha, +0x5A global alpha,
// +0x5B base position, +0x70 global matrix (3 x 4 rows, layout units: 1920 x 1080 around the centre, y up),
// +0xB0 name (24 bytes). Parts panes (a layout placed in another, the buttons) have the vtable PARTS_VT.
//
// Research: LARECOMP_UI_LOG="<present>[,<present>...]" logs the panes shown in those frames.
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL3/SDL_timer.h>

#include "hle.h"

uint64_t renderer_present_count(void);
void rt_register_fnhook_abs(uint64_t addr, int (*fn)(Cpu *c, uint64_t addr));

#define PANE_CALC 0x70018c01f0ull  // (Sword 1.3.2)
#define PARTS_VT 0x700260d8c8ull

#define NREC 8192
static uint64_t g_rec[2][NREC];
static _Atomic int g_nrec[2];
static int g_cur;
static uint64_t g_cur_present;

static bool guest(uint64_t p) { return p >= 0x7000000000ull && p < 0x7800000000ull; }

typedef struct { float x0, y0, x1, y1; } Rect;  // (layout pixels, 1920 x 1080, y down)
static bool pane_shown(uint64_t p) {
    for (int d = 0; d < 32 && guest(p); d++, p = *(const uint64_t *)(uintptr_t)(p + 0x18)) {
        const uint8_t *f = (const uint8_t *)(uintptr_t)(p + 0x58);
        if (!(f[0] & 1) || f[1] == 0) return false;
    }
    return true;
}
static Rect pane_rect(uint64_t p) {
    const float *m = (const float *)(uintptr_t)(p + 0x70), *sz = (const float *)(uintptr_t)(p + 0x50);
    float sx = sqrtf(m[0] * m[0] + m[4] * m[4]), sy = sqrtf(m[1] * m[1] + m[5] * m[5]);
    float w = sz[0] * sx, h = sz[1] * sy;
    uint8_t bp = *(const uint8_t *)(uintptr_t)(p + 0x5B);
    float cx = 960 + m[3], cy = 540 - m[7];
    int hx = bp % 3, vy = bp / 3;  // (0 centre, 1 left / top, 2 right / bottom)
    if (hx == 1) cx += w / 2; else if (hx == 2) cx -= w / 2;
    if (vy == 1) cy += h / 2; else if (vy == 2) cy -= h / 2;
    return (Rect){cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
}
static void pane_name(uint64_t p, char nm[25]) {
    memcpy(nm, (const void *)(uintptr_t)(p + 0xB0), 24);
    nm[24] = 0;
    for (int k = 0; k < 24 && nm[k]; k++) if (nm[k] < 32 || nm[k] > 126) nm[k] = '?';
}

// ---- the snapshot of a frame: its buttons and the game's cursor
#define NBTN 96
typedef struct { uint64_t p; Rect r; } Btn;
typedef struct {
    Btn b[NBTN];
    int nb;
    bool cursor;
    float cx, cy;  // the cursor's centre
    uint64_t present;
} Snap;
static Snap g_snap[2];
static _Atomic int g_snap_i;

static void build_snap(int buf, uint64_t present) {
    Snap *s = &g_snap[atomic_load(&g_snap_i) ^ 1];
    s->nb = 0, s->cursor = false, s->present = present;
    int n = g_nrec[buf];
    for (int i = 0; i < n; i++) {
        uint64_t p = g_rec[buf][i];
        const char *nm = (const char *)(uintptr_t)(p + 0xB0);
        bool parts = *(const uint64_t *)(uintptr_t)p == PARTS_VT && nm[0] == 'L' && nm[1] == '_';
        bool cur = !strncmp(nm, "P_cursor", 8);
        if (!parts && !cur) continue;
        if (!pane_shown(p)) continue;
        Rect r = pane_rect(p);
        float w = r.x1 - r.x0, h = r.y1 - r.y0;
        if (cur) {
            if (!s->cursor && w > 4 && h > 4) s->cursor = true, s->cx = (r.x0 + r.x1) / 2, s->cy = (r.y0 + r.y1) / 2;
            continue;
        }
        if (w < 16 || h < 16 || w > 1900 || h > 1060 || s->nb >= NBTN) continue;
        bool dup = false;
        for (int k = 0; k < s->nb && !dup; k++) dup = s->b[k].p == p;
        if (!dup) s->b[s->nb++] = (Btn){p, r};
    }
    atomic_store(&g_snap_i, atomic_load(&g_snap_i) ^ 1);
    static int was = -1;
    int now = s->cursor && s->nb > 0;
    if (now != was) {
        if (now) LOG("[uimouse] menu at present %llu: %d buttons, the game's cursor at %.0f, %.0f\n", (unsigned long long)present,
                     s->nb, s->cx, s->cy);
        else LOG("[uimouse] no menu cursor from present %llu\n", (unsigned long long)present);
        was = now;
    }
}

static void log_frame(int buf, uint64_t present) {
    int n = g_nrec[buf];
    LOG("[ui] present %llu: %d panes\n", (unsigned long long)present, n);
    for (int i = 0; i < n; i++) {
        uint64_t p = g_rec[buf][i];
        if (!pane_shown(p)) continue;
        char nm[25];
        pane_name(p, nm);
        Rect r = pane_rect(p);
        const float *m = (const float *)(uintptr_t)(p + 0x70);
        LOG("[ui]   %llx vt %llx %-24s rect %6.0f %6.0f %6.0f %6.0f scale %.2f alpha %u parent %llx\n", (unsigned long long)p,
            (unsigned long long)*(const uint64_t *)(uintptr_t)p, nm, r.x0, r.y0, r.x1, r.y1, sqrtf(m[0] * m[0] + m[4] * m[4]),
            *(const uint8_t *)(uintptr_t)(p + 0x5A), (unsigned long long)*(const uint64_t *)(uintptr_t)(p + 0x18));
    }
}

static const char *g_log;
static int pane_hook(Cpu *c, uint64_t addr) {
    (void)addr;
    uint64_t pr = renderer_present_count();
    if (pr != g_cur_present) {  // (a new frame: the last one's list is complete)
        int done = g_cur;
        if (g_cur_present) {
            build_snap(done, g_cur_present);
            static const char *click = (const char *)1;  // (tests: LARECOMP_UI_CLICK="<present>:<x>,<y>[;...]", layout pixels)
            if (click == (const char *)1) click = getenv("LARECOMP_UI_CLICK");
            for (const char *q = click; q && *q;) {
                long at = strtol(q, (char **)&q, 10);
                float x = 0, y = 0;
                if (*q == ':') x = strtof(q + 1, (char **)&q);
                if (*q == ',') y = strtof(q + 1, (char **)&q);
                if (at == (long)g_cur_present) {
                    extern bool uimouse_click(float x, float y);
                    LOG("[uimouse] test click at %.0f, %.0f: %s\n", x, y, uimouse_click(x, y) ? "a button" : "no button (A)");
                }
                if (*q == ';') q++; else break;
            }
            if (g_log) {
                char want[24];
                snprintf(want, sizeof want, "%llu", (unsigned long long)g_cur_present);
                const char *h = strstr(g_log, want);
                if (h && (h == g_log || h[-1] == ',') && (h[strlen(want)] == 0 || h[strlen(want)] == ',')) log_frame(done, g_cur_present);
            }
        }
        g_cur ^= 1;
        g_nrec[g_cur] = 0;
        g_cur_present = pr;
    }
    uint64_t p = c->x[0];
    if (guest(p)) {
        int n = atomic_fetch_add(&g_nrec[g_cur], 1);
        if (n < NREC) g_rec[g_cur][n] = p;
        else g_nrec[g_cur] = NREC;
    }
    return 0;
}

// runtime/main.c, before each boot: Sword 1.3.2 only (the addresses are its)
void uimouse_configure(bool sword132) {
    static bool hooked;
    if (!sword132 || hooked || (getenv("LARECOMP_UI_MOUSE") && !atoi(getenv("LARECOMP_UI_MOUSE")))) return;
    hooked = true;
    g_log = getenv("LARECOMP_UI_LOG");
    rt_register_fnhook_abs(PANE_CALC, pane_hook);
    LOG("[uimouse] menus: the mouse points and clicks (Sword 1.3.2)\n");
}

// ---- for runtime/hle/input.c
static const Snap *snap_now(void) {
    const Snap *s = &g_snap[atomic_load(&g_snap_i)];
    return s->present + 6 >= renderer_present_count() ? s : NULL;  // (a stale one: no menu)
}

// A menu with the game's cursor is up (the mouse is a pointer there)
bool uimouse_menu_active(void) {
    static uint64_t seen;  // (the cursor blinks out for a frame or two between menus: the pointer stays 0.4 s)
    const Snap *s = snap_now();
    uint64_t now = SDL_GetTicksNS();
    if (s && s->cursor && s->nb > 0) seen = now;
    return seen && now - seen < 400000000ull;
}

static float rect_dist(Rect r, float x, float y) {
    float dx = x < r.x0 ? r.x0 - x : x > r.x1 ? x - r.x1 : 0, dy = y < r.y0 ? r.y0 - y : y > r.y1 ? y - r.y1 : 0;
    return sqrtf(dx * dx + dy * dy);
}
static int button_at(const Snap *s, float x, float y) {  // (the smallest button containing x, y)
    int best = -1;
    float ba = 0;
    for (int i = 0; i < s->nb; i++) {
        Rect r = s->b[i].r;
        if (x < r.x0 || x > r.x1 || y < r.y0 || y > r.y1) continue;
        float a = (r.x1 - r.x0) * (r.y1 - r.y0);
        if (best < 0 || a < ba) best = i, ba = a;
    }
    return best;
}
static int selected(const Snap *s) {  // (the button the game's cursor is at: nearest to it)
    int best = -1;
    float bd = 1e9f;
    for (int i = 0; i < s->nb; i++) {
        float d = rect_dist(s->b[i].r, s->cx, s->cy);
        if (d < bd) bd = d, best = i;
    }
    return bd < 200 ? best : -1;
}

// The walk to a clicked button
static struct {
    _Atomic bool on;
    uint64_t target;
    float tx, ty;
    int steps, stuck, axis;
    uint64_t last_sel;
    uint64_t press_until, next_at;
    uint64_t bit;
} nav;
enum { NB_A = 0, NB_LEFT = 12, NB_UP = 13, NB_RIGHT = 14, NB_DOWN = 15 };  // (input.c's button bits)

// A left click at x, y (layout pixels): true when it is the menu's (a walk to a button started); false: an A press
bool uimouse_click(float x, float y) {
    const Snap *s = snap_now();
    if (!s || !s->cursor) return false;
    int t = button_at(s, x, y);
    if (t < 0) return false;
    nav.target = s->b[t].p;
    nav.tx = (s->b[t].r.x0 + s->b[t].r.x1) / 2, nav.ty = (s->b[t].r.y0 + s->b[t].r.y1) / 2;
    nav.steps = nav.stuck = 0, nav.axis = -1, nav.last_sel = 0;
    nav.press_until = 0, nav.next_at = SDL_GetTicksNS();
    nav.on = true;
    return true;
}

// The walk's buttons now (each controller read: runtime/hle/input.c fill_state)
uint64_t uimouse_pad(void) {
    uint64_t now = SDL_GetTicksNS();
    if (now < nav.press_until) return nav.bit;
    if (!nav.on || now < nav.next_at) return 0;
    const Snap *s = snap_now();
    if (!s || !s->cursor) { nav.on = false; return 0; }
    int t = -1;
    for (int i = 0; i < s->nb; i++) if (s->b[i].p == nav.target) t = i;
    int c = selected(s);
    if (t < 0 || c < 0 || nav.steps > 30) { nav.on = false; return 0; }
    if (c == t) {  // there: A
        nav.on = false;
        nav.bit = 1ull << NB_A, nav.press_until = now + 70000000ull, nav.next_at = nav.press_until;
        return nav.bit;
    }
    uint64_t sel = s->b[c].p;
    float cx = (s->b[c].r.x0 + s->b[c].r.x1) / 2, cy = (s->b[c].r.y0 + s->b[c].r.y1) / 2;
    float dx = nav.tx - cx, dy = nav.ty - cy;
    // the axis with more to go (in button sizes); the other one when a step along it did not move the selection
    float bw = s->b[c].r.x1 - s->b[c].r.x0, bh = s->b[c].r.y1 - s->b[c].r.y0;
    int axis = fabsf(dx) / (bw > 1 ? bw : 1) >= fabsf(dy) / (bh > 1 ? bh : 1) ? 0 : 1;
    if (nav.steps && sel == nav.last_sel) {
        if (++nav.stuck >= 3) { nav.on = false; return 0; }
        axis = nav.axis ^ 1;
        if ((axis == 0 && fabsf(dx) < 4) || (axis == 1 && fabsf(dy) < 4)) { nav.on = false; return 0; }
    } else {
        nav.stuck = 0;
    }
    nav.axis = axis, nav.last_sel = sel, nav.steps++;
    int b = axis == 0 ? (dx > 0 ? NB_RIGHT : NB_LEFT) : (dy > 0 ? NB_DOWN : NB_UP);
    nav.bit = 1ull << b;
    nav.press_until = now + 60000000ull, nav.next_at = nav.press_until + 110000000ull;  // (the cursor's move: a few frames)
    return nav.bit;
}
