// nn::ngc::ProfanityFilter (the system's word filter), implemented on the host: nothing is filtered.
//
// The SDK's filter reads its word lists from system data 0100000000000823 (NgWord2), which this port doesn't have:
// Initialize mounted it, fsp-srv OpenDataStorageByDataId answered 2002-1001 (runtime/services/fs.c fsp_srv) and the
// SDK aborted (svcBreak) inside nn::fs::QueryMountSystemDataCacheSize. The game creates a filter on entering the
// Battle Tower (issue #7) and on other screens where players name things. Sword and Shield import only the
// constructor, Initialize and Finalize from the SDK - the checks run on the game's side against online names, which
// this port has none of - so Initialize and Finalize succeed without mounting anything, and the constructor stays
// the SDK's.
#include "../hle.h"

// nn::Result nn::ngc::ProfanityFilter::Initialize(void *workMemory, size_t workMemorySize)
HLE_FUNC("_ZN2nn3ngc15ProfanityFilter10InitializeEPvm", ngc_filter_initialize) {
    static int logged;
    if (!logged++) LOG("[ngc] ProfanityFilter::Initialize: no word lists on this port, nothing is filtered\n");
    c->x[0] = RES_OK;
}

// nn::Result nn::ngc::ProfanityFilter::Finalize(): nothing was mounted
HLE_FUNC("_ZN2nn3ngc15ProfanityFilter8FinalizeEv", ngc_filter_finalize) {
    c->x[0] = RES_OK;
}
