// nn::ngc::ProfanityFilter (the system's word filter), implemented on the host: nothing is filtered.
//
// The SDK's filter reads its word lists from system data 0100000000000823 (NgWord2), which this port doesn't have:
// Initialize mounted it, fsp-srv OpenDataStorageByDataId answered 2002-1001 (runtime/services/fs.c fsp_srv) and the
// SDK aborted (svcBreak) inside nn::fs::QueryMountSystemDataCacheSize. The game creates a filter on entering the
// Battle Tower (issue #7) and on other screens where players name things. Sword and Shield import only the
// constructor, Initialize and Finalize from the SDK - the checks run on the game's side against online names, which
// this port has none of - so Initialize and Finalize succeed without mounting anything, and the constructor stays
// the SDK's.
//
// The filter's other functions are reached through its vtable, not imported (issue #18: the Battle Tower aborted with
// 2002-6905 at 0.2.0 when the game asked GetContentVersion, which opens the word lists' version file): the first
// Initialize hooks them where the SDK has them - the version is 1, no word is profane, no text is masked.
#include "../hle.h"
#include "../rt.h"

void rt_register_fnhook_abs(uint64_t addr, int (*fn)(Cpu *c, uint64_t addr));  // runtime/fnhook.c

enum { NG_VERSION, NG_CHECK, NG_CHECK_IMPL, NG_MASK };
static const struct { const char *sym; int kind; } k_ngc[] = {
    {"_ZN2nn3ngc15ProfanityFilter17GetContentVersionEv", NG_VERSION},
    {"_ZN2nn3ngc15ProfanityFilter19CheckProfanityWordsEPjjPPKcm", NG_CHECK},
    {"_ZN2nn3ngc15ProfanityFilter19CheckProfanityWordsEPjjPPKtm", NG_CHECK},
    {"_ZN2nn3ngc15ProfanityFilter19CheckProfanityWordsEPjjPPKDsm", NG_CHECK},
    {"_ZN2nn3ngc15ProfanityFilter23CheckProfanityWordsImplEPjPPKDsmb", NG_CHECK_IMPL},
    {"_ZN2nn3ngc15ProfanityFilter24MaskProfanityWordsInTextEPiPcj", NG_MASK},
    {"_ZN2nn3ngc15ProfanityFilter24MaskProfanityWordsInTextEPiPtj", NG_MASK},
    {"_ZN2nn3ngc15ProfanityFilter24MaskProfanityWordsInTextEPiPDsj", NG_MASK},
    {"_ZN2nn3ngc15ProfanityFilter30MaskProfanityWordsInTextCommonEPiPcj", NG_MASK},
};

static int ngc_hook(Cpu *c, uint64_t addr) {
    for (size_t i = 0; i < sizeof k_ngc / sizeof *k_ngc; i++) {
        if (loader_find_symbol(k_ngc[i].sym) != addr) continue;  // (another build's SDK after a reboot: not ours)
        switch (k_ngc[i].kind) {
        case NG_VERSION: c->x[0] = 1; return 1;
        // (methods: x0 is the filter)
        case NG_CHECK:  // (u32 *results, u32 regions, words, count): one bit mask per word, all clear
            for (uint64_t k = 0; c->x[1] && k < c->x[4]; k++) ST32(c->x[1] + 4 * k, 0);
            c->x[0] = RES_OK;
            return 1;
        case NG_CHECK_IMPL:  // (u32 *results, words, count, ...)
            for (uint64_t k = 0; c->x[1] && k < c->x[3]; k++) ST32(c->x[1] + 4 * k, 0);
            c->x[0] = RES_OK;
            return 1;
        case NG_MASK:  // (int *masked, text, regions): the text stays as it is
            if (c->x[1]) ST32(c->x[1], 0);
            c->x[0] = RES_OK;
            return 1;
        }
    }
    return 0;
}

// nn::Result nn::ngc::ProfanityFilter::Initialize(void *workMemory, size_t workMemorySize)
HLE_FUNC("_ZN2nn3ngc15ProfanityFilter10InitializeEPvm", ngc_filter_initialize) {
    static int logged;
    if (!logged++) LOG("[ngc] ProfanityFilter::Initialize: no word lists on this port, nothing is filtered\n");
    static uint64_t hooked;  // (the SDK's GetContentVersion the hooks were placed for: again after a reboot into another build)
    uint64_t v = loader_find_symbol(k_ngc[0].sym);
    if (v && v != hooked) {
        hooked = v;
        for (size_t i = 0; i < sizeof k_ngc / sizeof *k_ngc; i++) rt_register_fnhook_abs(loader_find_symbol(k_ngc[i].sym), ngc_hook);
    }
    c->x[0] = RES_OK;
}

// nn::Result nn::ngc::ProfanityFilter::Finalize(): nothing was mounted
HLE_FUNC("_ZN2nn3ngc15ProfanityFilter8FinalizeEv", ngc_filter_finalize) {
    c->x[0] = RES_OK;
}
