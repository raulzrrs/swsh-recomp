// Research aid (the mouse in Sword's menus: runtime/hle/uimouse.c): LARECOMP_UI_PROBE="<vtable hex>[,<vtable hex>...]@<present>"
// hooks, once the game has presented <present> frames, every function of those vtables (their first 64 slots) and
// counts its calls for the next 120 presents; then logs each slot's count and the names of the panes it was called on
// (x0, a Nintendo ui2d pane: its name at +0xB0). The slot called once a frame per visible pane is the one to hook.
#ifdef _WIN32
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "hle.h"

uint64_t renderer_game_frame(void);
void rt_register_fnhook_abs(uint64_t addr, int (*fn)(Cpu *c, uint64_t addr));

#define NV 4
#define NS 64
static uint64_t g_vt[NV];
static int g_nvt;
static long g_at = 2150;
static struct { uint64_t fn; _Atomic uint64_t calls; char names[4][25]; _Atomic int nn; int vt, slot; } g_slot[NV * NS];
static int g_nslot;

static bool guest(uint64_t p) { return p >= 0x7000000000ull && p < 0x7800000000ull; }

static int count_hook(Cpu *c, uint64_t addr) {
    for (int i = 0; i < g_nslot; i++)
        if (g_slot[i].fn == addr) {
            g_slot[i].calls++;
            uint64_t self = c->x[0];
            if (guest(self) && atomic_load(&g_slot[i].nn) < 4) {
                uint64_t vt = *(const uint64_t *)(uintptr_t)self;
                bool pane = false;
                for (int k = 0; k < g_nvt; k++) pane |= vt == g_vt[k];
                if (pane) {
                    const char *nm = (const char *)(uintptr_t)(self + 0xB0);
                    int n = atomic_fetch_add(&g_slot[i].nn, 1);
                    if (n < 4) {
                        for (int k = 0; k < 24; k++) g_slot[i].names[n][k] = nm[k] >= 32 && nm[k] < 127 ? nm[k] : 0;
                        g_slot[i].names[n][24] = 0;
                    }
                }
            }
            break;
        }
    return 0;  // (the game's own function runs)
}

static DWORD WINAPI probe_thread(LPVOID unused) {
    (void)unused;
    while ((long)renderer_game_frame() < g_at) Sleep(20);
    for (int v = 0; v < g_nvt; v++)
        for (int s = 0; s < NS; s++) {
            uint64_t fn = ((const uint64_t *)(uintptr_t)g_vt[v])[s];
            if (fn < 0x7000000000ull || fn >= 0x7004000000ull) break;  // (main's code only: the vtable has ended)
            bool dup = false;
            for (int i = 0; i < g_nslot; i++) dup |= g_slot[i].fn == fn;
            if (dup) continue;
            g_slot[g_nslot].fn = fn, g_slot[g_nslot].vt = v, g_slot[g_nslot].slot = s;
            g_nslot++;
        }
    for (int i = 0; i < g_nslot; i++) rt_register_fnhook_abs(g_slot[i].fn, count_hook);
    long from = (long)renderer_game_frame();
    LOG("[uiprobe] %d functions hooked at present %ld\n", g_nslot, from);
    while ((long)renderer_game_frame() < from + 120) Sleep(20);
    for (int i = 0; i < g_nslot; i++)
        LOG("[uiprobe] vtable %d slot %2d fn %llx: %6llu calls (%.1f a frame) %s %s %s %s\n", g_slot[i].vt, g_slot[i].slot,
            (unsigned long long)g_slot[i].fn, (unsigned long long)g_slot[i].calls, g_slot[i].calls / 120.0, g_slot[i].names[0],
            g_slot[i].names[1], g_slot[i].names[2], g_slot[i].names[3]);
    LOG("[uiprobe] done\n");
    return 0;
}

__attribute__((constructor)) static void uiprobe_init(void) {
    const char *e = getenv("LARECOMP_UI_PROBE");
    if (!e) return;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", e);
    char *at = strchr(buf, '@');
    if (at) *at = 0, g_at = atol(at + 1);
    for (char *p = buf; *p && g_nvt < NV;) {
        g_vt[g_nvt++] = strtoull(p, &p, 16);
        if (*p == ',') p++; else break;
    }
    if (g_nvt) CreateThread(NULL, 0, probe_thread, NULL, 0, NULL);
}
#endif
