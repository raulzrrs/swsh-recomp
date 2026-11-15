// Sword / Shield at 60 fps (the launcher's 60 fps option). What the community's 60 fps patch does (Fl4sh9174's
// "60FPS v1.3.2" pchtxt), mapped to each build, done by the runtime instead of patching the executable:
//  - the game asks for a present interval of 2 (30 fps) through a small wrapper (1.0.0: main+0x178e098, 1.3.2:
//    main+0x18a2c88, `mov w1, w21` before the call); the patch passes 1. Here nvnWindowSetPresentInterval gets 1.
//  - the game keeps a table of frame durations in ns by interval (16666667, 33333334, 50000000, ...), read with the
//    measured time to scale animations (main+0x686908 in 1.0.0); the patch makes the interval-2 entry 1/60 s.
// The game measures real frame times itself (main+0x5dfee4), so the rest follows; verified against the patch's own
// addresses in 1.3.2 (docs/DIAGNOSTICO-SWSH.md).
#include <stdio.h>
#include <string.h>

#include "../hle.h"
#include "../settings.h"

typedef struct { uint8_t build[8]; uint64_t table2; const char *name; } Swsh60Build;
static const Swsh60Build k_builds[] = {
    {{0x8d, 0xaf, 0xed, 0xbb, 0x5b, 0xe8, 0x1c, 0x2c}, 0x7001f07578ull, "Sword 1.0.0"},
    {{0xa3, 0xb7, 0x5b, 0xcd, 0x33, 0x11, 0x38, 0x5a}, 0x700203eeb8ull, "Sword 1.3.2"},
};

static const Swsh60Build *g_build;  // the running build, when it is in the table
static bool g_on, g_applied;

static const Swsh60Build *build_of(const char *exefs) {
    char p[1100];
    snprintf(p, sizeof p, "%s/main", exefs);
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    uint8_t id[8] = {0};
    fseek(f, 0x40, SEEK_SET);  // (NSO header: the module id)
    size_t n = fread(id, 1, 8, f);
    fclose(f);
    for (size_t i = 0; n == 8 && i < sizeof k_builds / sizeof *k_builds; i++)
        if (!memcmp(id, k_builds[i].build, 8)) return &k_builds[i];
    return NULL;
}

bool swsh60_supported(const char *exefs) { return build_of(exefs) != NULL; }

// runtime/main.c, before each boot: the option and the build
void swsh60_configure(const char *exefs) {
    extern void settings_load(void);
    settings_load();  // (read once; with --play nothing has read larecomp.ini yet)
    g_build = build_of(exefs);
    g_on = g_build && g_settings.fps == 60;
    g_applied = false;
    if (g_build) LOG("[swsh60] %s: 60 fps %s\n", g_build->name, g_on ? "on" : "off");
    extern void uimouse_configure(bool sword132);  // (runtime/hle/uimouse.c: the mouse in the menus)
    uimouse_configure(g_build == &k_builds[1]);
}

bool swsh60_enabled(void) { return g_on; }
bool swsh60_known(void) { return g_build != NULL; }  // (the F12 menu: the 60 fps choice is offered)

// nvnWindowSetPresentInterval (runtime/nvn/objects.c): the interval the game asked for -> the one to use. The table
// entry is written here the first time: the game's main is loaded by then.
int swsh60_interval(int asked) {
    if (!g_on) return asked;
    if (!g_applied) {
        g_applied = true;
        uint64_t *e = (uint64_t *)(uintptr_t)g_build->table2;
        if (*e == 33333334ull) { *e = 16666667ull; LOG("[swsh60] frame time of interval 2: 1/60 s\n"); }
        else LOG("[swsh60] unexpected table value %llu at %llx: left alone\n", (unsigned long long)*e,
                 (unsigned long long)g_build->table2);
    }
    return asked == 2 ? 1 : asked;
}
