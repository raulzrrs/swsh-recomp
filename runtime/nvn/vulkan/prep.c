// Shader preparation: every shader stage of the game, compiled for this GPU ahead of the draws that need it.
//
// On a new install each shader and pipeline is compiled the first time a draw meets it, and the frame waits (9.9 s
// of hitches in 68 s of play, docs/engineering/estudo-shaders-ports-switch.md). The game's shaders are all in its
// files: BNSH containers (nn::gfx ResShaderFile), loose and inside the .gfpak packs (LZ4), effects (.ptcl) and .arc.
// tools/swsh/shader_store.py measured that they hold 97.5% of the stages a real session draws with (the rest is made
// at run time). The approach is OpenSmash's (nvn_vk_store.c / nvn_vk_prep, MIT; README: no holds in its first run
// after the preparation): with VK_EXT_graphics_pipeline_library each stage becomes its own pipeline part, and the
// part a draw builds from the same SPIR-V, set layout and state is a pipeline cache hit (exec.c vk_stage_layout,
// vk_prep_part), so the draw only links (~0.01 ms) instead of waiting for the driver (20-120 ms, up to 2.5 s).
//
// When the game opens its RomFS (services/fs.c) the candidate files are listed; threads at idle priority then
//  1. read them (and the game's executables, which hold a few) and collect every distinct stage (FNV-1a of the code block, the runtime's own id: exec.c
//     stage_hash_rt) and every vertex + fragment pair of a variation;
//  2. every vertex stage: translated with the keys a draw of this game uses (below), compiled to SPIR-V (the same
//     disk cache the draws use) and its part built; then every fragment stage, keyed by the vertex outputs it may get
//     (all of them, and those of each vertex stage a variation pairs it with: most fragment stages sit in variations
//     of their own, and the game picks the vertex stage at run time);
//  3. write the parts' pipeline cache and the ids of the parts done, so the next start skips them (the ids every 30 s
//     too: a run closed early resumes where it stopped).
// Draw-time keys the files don't say (vertex attribute types, integer targets / textures) take the defaults most
// draws use (cache/pipeline_recipes: 3687 recorded pipelines); a draw with others builds its own, as before.
// Work done is keyed by the translator build, the GPU and its driver: another build or driver prepares again.
// LARECOMP_SHADER_PREP=0: off. =<n>: n threads (default: half the logical cores, at most 8). Needs the pipeline parts
// (g_vk.gpl, on by default where supported). The log reports progress ([prep]). LARECOMP_SHADER_PREP_EXIT=1: quit when it is done (tests).
#include <fcntl.h>
#include <pthread.h>
#include <SDL3/SDL_cpuinfo.h>
#include <SDL3/SDL_timer.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../../rt.h"
#include "../../paths.h"

#ifndef O_BINARY
#define O_BINARY 0
#endif
#include "shaders.h"
#include "vk.h"

VkPipelineLayout vk_stage_layout(VkDescriptorSetLayout sl, int set);       // exec.c
VkPipeline vk_prep_part(int fragment, VkShaderModule m, VkPipelineLayout layout);
bool vk_shader_jobs_waiting(void);

// ------------------------------------------------------------------ the files
typedef struct { uint64_t off, size; } PrepFile;
static PrepFile *g_files;
static int g_nfiles, g_cap_files;
static int g_fd = -1;
static uint64_t g_bytes;

static bool candidate(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return false;
    static const char *const ext[] = {".gfpak", ".ptcl", ".bnsh", ".bnsh_vsh", ".bnsh_fsh", ".arc"};
    for (size_t i = 0; i < sizeof ext / sizeof ext[0]; i++)
        if (!strcmp(dot, ext[i])) return true;
    return false;
}

static void add_file(const char *path, int fd, uint64_t off, uint64_t size, void *u) {
    (void)fd, (void)u;
    if (!candidate(path) || !size || size > (512u << 20)) return;
    if (g_nfiles == g_cap_files) {
        g_cap_files = g_cap_files ? g_cap_files * 2 : 4096;
        g_files = realloc(g_files, (size_t)g_cap_files * sizeof *g_files);
    }
    g_files[g_nfiles++] = (PrepFile){off, size};
    g_bytes += size;
}

// ------------------------------------------------------------------ containers
static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

// LZ4 block format (every .gfpak entry of this game); false on a malformed block
static bool lz4_block(const uint8_t *src, size_t n, uint8_t *dst, size_t cap) {
    size_t i = 0, o = 0;
    while (i < n) {
        uint8_t t = src[i++];
        size_t lit = t >> 4;
        if (lit == 15) for (uint8_t b = 255; b == 255 && i < n;) lit += b = src[i++];
        if (i + lit > n || o + lit > cap) return false;
        memcpy(dst + o, src + i, lit);
        i += lit, o += lit;
        if (i >= n) break;  // (the last sequence has literals only)
        if (i + 2 > n) return false;
        size_t dist = src[i] | (size_t)src[i + 1] << 8, len = (t & 15) + 4;
        i += 2;
        if ((t & 15) == 15) for (uint8_t b = 255; b == 255 && i < n;) len += b = src[i++];
        if (!dist || dist > o || o + len > cap) return false;
        for (size_t k = 0; k < len; k++, o++) dst[o] = dst[o - dist];  // (overlapping copies repeat a pattern)
    }
    return o == cap;
}

static uint64_t fnv1a(const uint8_t *b, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t k = 0; k < n; k++) h = (h ^ b[k]) * 0x100000001B3ull;
    return h ? h : 1;
}

// distinct stages, by id (the code's FNV-1a), and the pairs of a variation
typedef struct { uint64_t h; uint8_t *code; uint32_t size, const_off; uint8_t stage, nmasks; uint32_t out_mask, masks[8]; } Stage;
#define STAGE_CAP (1 << 16)
static Stage g_stages[STAGE_CAP];
static int g_nstages;
typedef struct { uint64_t vs, fs; } Pair;
static Pair *g_pairs;
static int g_npairs, g_cap_pairs;
static pthread_mutex_t g_scan_lock = PTHREAD_MUTEX_INITIALIZER;

static Stage *stage_find(uint64_t h, bool add) {  // (open addressing; under g_scan_lock while scanning)
    uint32_t i = (uint32_t)(h ^ (h >> 29)) & (STAGE_CAP - 1);
    for (int n = 0; n < STAGE_CAP; n++, i = (i + 1) & (STAGE_CAP - 1)) {
        if (g_stages[i].h == h) return &g_stages[i];
        if (!g_stages[i].h) {
            if (!add || g_nstages >= STAGE_CAP * 3 / 4) return NULL;
            g_stages[i].h = h;
            g_nstages++;
            return &g_stages[i];
        }
    }
    return NULL;
}

static uint64_t note_stage(const uint8_t *code, uint32_t size, const uint8_t *ctrl, int stage) {
    uint64_t h = fnv1a(code, size);
    pthread_mutex_lock(&g_scan_lock);
    Stage *s = stage_find(h, true);
    if (s && !s->code) {
        // (the translator may read up to the constants, or 64 KB without them, as it does from guest memory: zeros)
        uint32_t co = rd32(ctrl + 0x700), max = co ? co : 0x10000;
        s->code = calloc(1, (size_t)(size > max ? size : max) + 0x100);
        memcpy(s->code, code, size);
        s->size = size;
        s->const_off = co;
        s->stage = (uint8_t)stage;
    }
    pthread_mutex_unlock(&g_scan_lock);
    return s ? h : 0;
}

// The BNSH at d[base:]: its variations' vertex and fragment stages (the layout: SuperMarioWonderRecomp
// tools/maxwell/containers.py, MIT; every pointer is an offset from the container).
static void scan_bnsh(const uint8_t *d, size_t n, size_t base) {
    if (base + 0x20 > n) return;
    uint32_t size = rd32(d + base + 0x1C);
    uint16_t first = rd16(d + base + 0x16);
    if (size <= 0x100 || size > n - base || base + first + 0x30 > n || memcmp(d + base + first, "grsc", 4)) return;
    const uint8_t *b = d + base, *g = b + first;
    uint32_t nvar = rd32(g + 0x1C);
    uint64_t vars = rd64(g + 0x20);
    for (uint32_t k = 0; k < nvar && vars + (uint64_t)(k + 1) * 0x40 <= size; k++) {
        uint64_t prog = rd64(b + vars + k * 0x40 + 0x10);
        if (!prog || prog + 0x40 > size) continue;
        uint64_t ids[2] = {0, 0};
        for (int s = 0; s < 5; s++) {  // ShaderStage order: vertex, hull, domain, geometry, pixel
            if (s != 0 && s != 4) continue;
            uint64_t sc = rd64(b + prog + 8 + s * 8);
            if (!sc || sc + 0x20 > size) continue;
            uint64_t ctrl = rd64(b + sc + 8), code = rd64(b + sc + 0x10);
            uint32_t csz = rd32(b + sc + 0x18), ksz = rd32(b + sc + 0x1C);
            if (ksz < 0x708 || ctrl + ksz > size || code + csz > size || rd32(b + ctrl) != 0x98761234 || rd32(b + code) != 0x12345678) continue;
            uint32_t hsz = rd32(b + ctrl + 0x704);  // (what the runtime hashes: objects.c set_shaders)
            if (!hsz || hsz > csz) hsz = csz;
            ids[s == 4] = note_stage(b + code, hsz, b + ctrl, s == 4 ? STAGE_FRAGMENT : STAGE_VERTEX);
        }
        if (!ids[0]) continue;
        pthread_mutex_lock(&g_scan_lock);
        if (g_npairs == g_cap_pairs) {
            g_cap_pairs = g_cap_pairs ? g_cap_pairs * 2 : 4096;
            g_pairs = realloc(g_pairs, (size_t)g_cap_pairs * sizeof *g_pairs);
        }
        g_pairs[g_npairs++] = (Pair){ids[0], ids[1]};
        pthread_mutex_unlock(&g_scan_lock);
    }
}

static void scan_buffer(const uint8_t *d, size_t n) {  // (every "BNSH\0\0\0\0": memchr for the B, then the rest)
    if (n < 8) return;
    for (const uint8_t *p = d; (p = memchr(p, 'B', (size_t)(d + n - 7 - p))); p++)
        if (!memcmp(p, "BNSH\0\0\0\0", 8)) scan_bnsh(d, n, (size_t)(p - d));
}

static void scan_file(const PrepFile *f) {
    uint8_t *d = malloc(f->size);
    if (!d || pread(g_fd, d, f->size, (off_t)f->off) != (ssize_t)f->size) { free(d); return; }
    if (f->size >= 0x28 && !memcmp(d, "GFLXPACK", 8)) {
        uint32_t nf = rd32(d + 0x10);
        uint64_t fi = rd64(d + 0x18);
        for (uint32_t i = 0; i < nf && fi + (uint64_t)(i + 1) * 24 <= f->size; i++) {
            const uint8_t *e = d + fi + i * 24;  // level u16, compression u16, size, compressed size, pad, offset
            uint16_t ct = rd16(e + 2);
            uint32_t dsz = rd32(e + 4), csz = rd32(e + 8);
            uint64_t off = rd64(e + 16);
            if (off + csz > f->size || dsz > (256u << 20)) continue;
            if (ct == 0) { scan_buffer(d + off, csz); continue; }
            if (ct != 2) continue;  // (zlib / Oodle: not used by this game)
            uint8_t *u = malloc(dsz);
            if (u && lz4_block(d + off, csz, u, dsz)) scan_buffer(u, dsz);
            free(u);
        }
    } else {
        scan_buffer(d, f->size);
    }
    free(d);
}

// ------------------------------------------------------------------ parts done (cache/shader_prep.bin)
// One u64 per part: the stage id mixed with its key; the file starts with the id of this translator build + GPU + driver.
#define DONE_CAP (1 << 17)
static uint64_t g_done[DONE_CAP];
static int g_ndone;
static uint64_t g_epoch;
static pthread_mutex_t g_done_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t *g_new_done;
static int g_nnew, g_cap_new;

static bool done_has(uint64_t id, bool add) {  // (under g_done_lock)
    uint32_t i = (uint32_t)(id ^ (id >> 31)) & (DONE_CAP - 1);
    for (int n = 0; n < DONE_CAP; n++, i = (i + 1) & (DONE_CAP - 1)) {
        if (g_done[i] == id) return true;
        if (!g_done[i]) {
            if (add && g_ndone < DONE_CAP * 3 / 4) { g_done[i] = id; g_ndone++; }
            return false;
        }
    }
    return false;
}

static void done_path(char *out, size_t n) { snprintf(out, n, "%s/shader_prep.bin", rt_cache_dir()); }

// cache/shader_prep_complete: one u64, this build + GPU + driver (g_epoch) mixed with the files scanned. A start that
// finds it skips the preparation; without it every start read and translated all 18k stages again (38 s of CPU) to
// learn that every part was done.
#define PREP_VERSION 3  // (what the preparation covers: 2 = the executables' shaders too, 3 = float 7/8/11, integer target 1)
static uint64_t complete_id(void) { return (g_epoch ^ (uint64_t)g_nfiles * 0x9E3779B97F4A7C15ull ^ g_bytes * 31 ^ PREP_VERSION * 0x51ED27ull) | 1; }

static bool complete_check(void) {
    char path[600];
    snprintf(path, sizeof path, "%s/shader_prep_complete", rt_cache_dir());
    FILE *f = fopen(path, "rb");
    uint64_t v = 0;
    bool ok = f && fread(&v, 8, 1, f) == 1 && v == complete_id();
    if (f) fclose(f);
    return ok;
}

static void complete_write(void) {
    char path[600];
    snprintf(path, sizeof path, "%s/shader_prep_complete", rt_cache_dir());
    FILE *f = fopen(path, "wb");
    uint64_t v = complete_id();
    if (f) { fwrite(&v, 8, 1, f); fclose(f); }
}

static void done_load(void) {
    const char *stamp = maxwell_build_stamp();
    uint64_t h = fnv1a((const uint8_t *)stamp, strlen(stamp));
    h = fnv1a((const uint8_t *)&g_vk.props.pipelineCacheUUID, VK_UUID_SIZE) ^ h * 31;
    g_epoch = (h ^ g_vk.props.driverVersion * 0x9E3779B97F4A7C15ull ^ g_vk.props.deviceID) | 1;
    char path[600];
    done_path(path, sizeof path);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint64_t e = 0, id;
    if (fread(&e, 8, 1, f) == 1 && e == g_epoch)
        while (fread(&id, 8, 1, f) == 1) done_has(id, true);
    fclose(f);
}

static void done_save(void) {  // (append the new ids; a new epoch rewrites the file)
    pthread_mutex_lock(&g_done_lock);
    int n = g_nnew;
    uint64_t *ids = g_new_done;
    g_new_done = NULL, g_nnew = g_cap_new = 0;
    pthread_mutex_unlock(&g_done_lock);
    if (!n) { free(ids); return; }
    char path[600];
    done_path(path, sizeof path);
    FILE *f = fopen(path, "rb");
    uint64_t e = 0;
    bool same = f && fread(&e, 8, 1, f) == 1 && e == g_epoch;
    if (f) fclose(f);
    mkdir(rt_cache_dir(), 0755);
    f = fopen(path, same ? "ab" : "wb");
    if (f) {
        if (!same) fwrite(&g_epoch, 8, 1, f);
        fwrite(ids, 8, (size_t)n, f);
        fclose(f);
    }
    free(ids);
}

// true when this part still has to be built (and it is claimed)
static bool claim(uint64_t id) {
    pthread_mutex_lock(&g_done_lock);
    bool had = done_has(id, true);
    pthread_mutex_unlock(&g_done_lock);
    return !had;
}

static void finished(uint64_t id) {
    pthread_mutex_lock(&g_done_lock);
    if (g_nnew == g_cap_new) {
        g_cap_new = g_cap_new ? g_cap_new * 2 : 1024;
        g_new_done = realloc(g_new_done, (size_t)g_cap_new * sizeof *g_new_done);
    }
    g_new_done[g_nnew++] = id;
    pthread_mutex_unlock(&g_done_lock);
}

// ------------------------------------------------------------------ parts that stopped the driver (cache/shader_prep_inflight.bin)
// The driver's shader compiler can crash the process on one stage (NVIDIA, nvgpucomp64.dll: issues #1, #6), and the
// preparation resumed where it stopped, so it built that stage again and crashed on every start, at the same
// percentage. Each thread writes the id of the part it hands to the driver into its slot of this file and clears it
// when the driver returns; a slot still set at the next start is a part the driver didn't come back from: it is
// marked done (skipped) and logged. A draw that really uses that stage still builds it then, alone.
#define INFLIGHT_SLOTS 16
static uint64_t g_inflight[INFLIGHT_SLOTS];
static pthread_mutex_t g_inflight_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_inflight_f;

static void inflight_path(char *out, size_t n) { snprintf(out, n, "%s/shader_prep_inflight.bin", rt_cache_dir()); }

static void inflight_write(void) {  // (under g_inflight_lock: the whole table, flushed to the OS, which keeps it if we crash)
    if (!g_inflight_f) return;
    fseek(g_inflight_f, 0, SEEK_SET);
    fwrite(g_inflight, sizeof g_inflight, 1, g_inflight_f);
    fflush(g_inflight_f);
}

// At the start, after done_load: the parts of the last run that never came back are skipped from now on.
static void inflight_recover(void) {
    char path[600];
    inflight_path(path, sizeof path);
    FILE *f = fopen(path, "rb");
    uint64_t ids[INFLIGHT_SLOTS] = {0};
    if (f) {
        if (fread(ids, sizeof ids, 1, f) != 1) memset(ids, 0, sizeof ids);
        fclose(f);
    }
    int n = 0;
    for (int i = 0; i < INFLIGHT_SLOTS; i++)
        if (ids[i]) {
            pthread_mutex_lock(&g_done_lock);
            bool had = done_has(ids[i], true);
            pthread_mutex_unlock(&g_done_lock);
            if (!had) { finished(ids[i]); n++; }
            LOG("[prep] part %016llx: the last run stopped while the driver compiled it; skipped (a draw that needs it builds it)\n",
                (unsigned long long)ids[i]);
        }
    if (n) done_save();
    g_inflight_f = fopen(path, "wb");
    pthread_mutex_lock(&g_inflight_lock);
    memset(g_inflight, 0, sizeof g_inflight);
    inflight_write();
    pthread_mutex_unlock(&g_inflight_lock);
}

static int inflight_begin(uint64_t id) {  // -> slot, or -1 (all taken: the part is built unrecorded)
    pthread_mutex_lock(&g_inflight_lock);
    int s = -1;
    for (int i = 0; i < INFLIGHT_SLOTS && s < 0; i++)
        if (!g_inflight[i]) s = i;
    if (s >= 0) { g_inflight[s] = id; inflight_write(); }
    pthread_mutex_unlock(&g_inflight_lock);
    return s;
}

static void inflight_end(int s) {
    if (s < 0) return;
    pthread_mutex_lock(&g_inflight_lock);
    g_inflight[s] = 0;
    inflight_write();
    pthread_mutex_unlock(&g_inflight_lock);
}

// ------------------------------------------------------------------ building
static _Atomic int g_next_file;
static _Atomic uint64_t stat_parts, stat_skipped, stat_failed, stat_part_ns;

static void key_defaults(ShaderKey *k) {
    memset(k, 0, sizeof *k);
    k->glsl = 1;
    k->flip_y = 1;           // (Sword: lower-left window origin and NVN's [-1, 1] depth in every recorded draw,
    k->depth_minus_one = 1;  // exec.c draw_shader_keys)
}

// One stage: translated with these keys, compiled, and its part built. Returns the translation (for the vertex
// stage's outputs), or NULL. done: the part was built before (only the translation is wanted).
static ShaderInfo *build_stage(const Stage *s, const ShaderKey *key, bool want_info, uint64_t id) {
    bool todo = claim(id);
    if (!todo && !want_info) { atomic_fetch_add(&stat_skipped, 1); return NULL; }
    maxwell_gmem_base = HEAP_BASE;  // (as exec.c translate_program)
    maxwell_gmem_size = HEAP_MAX;
    ShaderInfo *si = maxwell_translate(s->code, s->const_off ? s->const_off : 0x10000, s->stage, key);
    if (!si || !si->msl) { atomic_fetch_add(&stat_failed, 1); if (todo) finished(id); maxwell_free(si); return NULL; }
    if (getenv("LARECOMP_PREP_DEBUG")) {
        uint64_t g = 0xcbf29ce484222325ull;
        for (const char *q = si->msl; *q; q++) g = (g ^ (uint8_t)*q) * 0x100000001B3ull;
        LOG("[prepdbg] prep %s %016llx glsl %016llx\n", s->stage == STAGE_FRAGMENT ? "fs" : "vs", (unsigned long long)s->h, (unsigned long long)g);
    }
    if (!todo) { atomic_fetch_add(&stat_skipped, 1); return si; }
    uint64_t t0 = SDL_GetTicksNS();
    int slot = inflight_begin(id);  // (until the driver has come back: a crash in it skips this part next start)
    size_t nw = 0;
    uint32_t *spv = vk_glsl_to_spirv(si->msl, s->stage, s->stage == STAGE_VERTEX ? "vs" : "fs", &nw);
    bool ok = false;
    if (spv) {
        VkBindingInfo binds[96];
        int nb = vk_reflect(spv, nw, binds, 96);
        int set = s->stage == STAGE_FRAGMENT;
        VkDescriptorSetLayout sl = vk_set_layout(binds, nb, (uint32_t)set, set ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineLayout pl = vk_stage_layout(sl, set);
        VkShaderModule m = vk_module(spv, nw);
        VkPipeline part = m ? vk_prep_part(set, m, pl) : VK_NULL_HANDLE;
        ok = part != VK_NULL_HANDLE;
        if (part) vkDestroyPipeline(g_vk.dev, part, NULL);
        if (m) vkDestroyShaderModule(g_vk.dev, m, NULL);
        vkDestroyPipelineLayout(g_vk.dev, pl, NULL);
        vkDestroyDescriptorSetLayout(g_vk.dev, sl, NULL);
        free(spv);
    }
    inflight_end(slot);
    atomic_fetch_add(&stat_part_ns, SDL_GetTicksNS() - t0);
    atomic_fetch_add(ok ? &stat_parts : &stat_failed, 1);
    finished(id);
    if (!want_info) { maxwell_free(si); return NULL; }
    return si;
}

static uint64_t key_id(uint64_t h, const ShaderKey *k) { return fnv1a((const uint8_t *)k, sizeof *k) ^ h * 0x100000001B3ull ^ g_epoch; }

// Vertex attributes the game feeds as unsigned integers at some draws and as floats at others: locations 7, 8 and 11
// (the bone indices and their companions on skinned models; something else, a float, on others: a new game drew 31
// vertex stages that read them as floats). A stage reading any of them gets both: all of them unsigned, all floats.
// (Every combination, 8 at most per stage, took 25141 parts and 7.4 GB for draws that don't come.)
static const int INT_LOCS[] = {7, 8, 11};

static void build_vertex(Stage *s) {
    ShaderKey k;
    key_defaults(&k);
    for (int i = 0; i < 3; i++) k.attr_kind[INT_LOCS[i]] = 2;  // (the kind of a location the stage doesn't read is unused)
    ShaderInfo *si = build_stage(s, &k, true, key_id(s->h, &k));
    if (!si) return;
    s->out_mask = si->out_mask;  // (the fragment stages it pairs with are keyed by it)
    bool reads = false;
    for (int i = 0; i < 3; i++) reads |= (si->attr_mask >> INT_LOCS[i]) & 1;
    maxwell_free(si);
    if (!reads) return;
    for (int i = 0; i < 3; i++) k.attr_kind[INT_LOCS[i]] = 0;
    maxwell_free(build_stage(s, &k, false, key_id(s->h, &k)));
}

// A fragment stage reads only some of the vertex outputs, and its translation depends on nothing else of the vertex
// stage (an input the vertex stage doesn't write reads 0): built with every output there (most programs), and once
// more for each pairing in the files' variations that leaves some of its inputs out. A stage that writes color
// target 1 is built again with that target unsigned (an integer target in some passes: 20 parts of a new game).
static void build_fragment_masks(Stage *s, ShaderKey k, bool also_rt1) {
    k.vs_out_mask = ~0u;
    ShaderInfo *si = build_stage(s, &k, true, key_id(s->h, &k));
    uint32_t in = si ? si->attr_mask : 0, seen[8];
    bool rt1 = si && (si->out_mask & 2);
    int nseen = 0;
    maxwell_free(si);
    for (int i = 0; i < s->nmasks && in; i++) {
        uint32_t m = s->masks[i] & in;
        bool dup = m == in;
        for (int j = 0; j < nseen && !dup; j++) dup = seen[j] == m;
        if (dup) continue;
        seen[nseen++] = m;
        k.vs_out_mask = m;  // (the bits it doesn't read don't change it: as a draw with that vertex stage keys it)
        maxwell_free(build_stage(s, &k, false, key_id(s->h, &k)));
    }
    if (also_rt1 && rt1) {
        k.rt_kind[1] = 2;
        build_fragment_masks(s, k, false);
    }
}

static void build_fragment(Stage *s) {
    ShaderKey k;
    key_defaults(&k);
    k.depth_minus_one = 0;  // (draw_shader_keys sets it on the vertex key only)
    build_fragment_masks(s, k, true);
}

static _Atomic int g_next_stage;
// progress, for the notice on screen (shader_prep_progress): files scanned, stages built, and when it ended
static _Atomic int g_files_done, g_stages_done, g_running;
static _Atomic uint64_t g_end_ns;
static int g_phase_stage;  // STAGE_VERTEX, then STAGE_FRAGMENT
// Started from the start screen (shader_prep_start_image): its threads run below normal priority, as many as the CPU
// has to spare, while nothing else needs it. Once the game runs (g_in_game) they drop to idle priority and only the
// first g_game_threads of them go on taking stages, so the preparation never competes with the game's frames.
static _Atomic int g_from_launcher, g_in_game, g_complete, g_game_threads = 1, g_thread_seq;
static _Atomic uint64_t g_build_t0;  // (when the stage building began: the rate since then gives the time left)

static void *prep_thread(void *a) {
    (void)a;
    int me = atomic_fetch_add(&g_thread_seq, 1);
    bool lowered = !atomic_load(&g_from_launcher) || atomic_load(&g_in_game);
    if (lowered) plat_thread_idle(); else plat_thread_background();
    if (g_phase_stage < 0) {  // (scan)
        // The executables too, as loaded: the game keeps shaders of its own in main (BNSH 2.1.11 containers, 14
        // of the stages a town and a battle drew that no file has). One thread takes them.
        static _Atomic int modules_taken;
        if (!atomic_exchange(&modules_taken, 1))
            for (int m = 0; m < g_nmodules; m++)
                if (g_modules[m].base && g_modules[m].size) scan_buffer((const uint8_t *)(uintptr_t)g_modules[m].base, g_modules[m].size);
        int nf;
        while ((nf = atomic_fetch_add(&g_next_file, 1)) < g_nfiles) { scan_file(&g_files[nf]); atomic_fetch_add(&g_files_done, 1); }
        return NULL;
    }
    int i;
    for (;;) {
        if (!lowered && atomic_load(&g_in_game)) {  // (the game started: idle priority, and fewer threads)
            plat_thread_idle();
            lowered = true;
        }
        if (lowered && atomic_load(&g_from_launcher) && me >= atomic_load(&g_game_threads)) break;
        if ((i = atomic_fetch_add(&g_next_stage, 1)) >= STAGE_CAP) break;
        Stage *s = &g_stages[i];
        if (!s->code || s->stage != g_phase_stage) continue;
        while (vk_shader_jobs_waiting()) SDL_DelayNS(2 * 1000000ull);  // (a draw's own builds go first)
        if (s->stage == STAGE_VERTEX) build_vertex(s); else build_fragment(s);
        atomic_fetch_add(&g_stages_done, 1);
    }
    return NULL;
}

static _Atomic int g_live;

static void *phase_thread(void *a) {
    prep_thread(a);
    atomic_fetch_sub(&g_live, 1);
    return NULL;
}

// Runs one phase on n threads. Every 30 s the ids of the parts built so far are written: a run closed before the end
// keeps them (the parts themselves are in the driver's own cache, and vk_parts.bin is written at exit).
static void run_phase(int phase, int n) {
    g_phase_stage = phase;
    atomic_store(&g_thread_seq, 0);
    if (phase >= 0 && !atomic_load(&g_build_t0)) atomic_store(&g_build_t0, SDL_GetTicksNS());
    atomic_store(&g_next_stage, 0);
    atomic_store(&g_live, n);
    pthread_t t[64];
    for (int k = 0; k < n; k++) pthread_create(&t[k], NULL, phase_thread, NULL);
    uint64_t last = SDL_GetTicksNS();
    while (atomic_load(&g_live)) {
        SDL_DelayNS(200 * 1000000ull);
        if (phase >= 0 && SDL_GetTicksNS() - last > 30 * 1000000000ull) { last = SDL_GetTicksNS(); done_save(); }
    }
    for (int k = 0; k < n; k++) pthread_join(t[k], NULL);
}

static void log_progress(const char *what, uint64_t t0) {
    LOG("[prep] %s: %llu parts built (%.0f ms of driver time), %llu done before, %llu failed, %.0f s\n", what,
        (unsigned long long)atomic_load(&stat_parts), atomic_load(&stat_part_ns) / 1e6, (unsigned long long)atomic_load(&stat_skipped),
        (unsigned long long)atomic_load(&stat_failed), (SDL_GetTicksNS() - t0) / 1e9);
}

static void *prep_main(void *a) {
    (void)a;
    plat_thread_idle();
    while (!g_vk.dev || !g_vk.pcache) SDL_DelayNS(50 * 1000000ull);
    if (!g_vk.gpl) { LOG("[prep] off: needs the pipeline parts (LARECOMP_GPL=0, or not supported)\n"); return NULL; }
    done_load();
    inflight_recover();
    if (complete_check()) {
        LOG("[prep] already done for this build, GPU and driver\n");
        atomic_store(&g_complete, 1);
        return NULL;
    }
    int cores = SDL_GetNumLogicalCPUCores(), n = cores / 2;
    const char *e = getenv("LARECOMP_SHADER_PREP");
    if (e && atoi(e) > 0) n = atoi(e);
    if (n < 1) n = 1;
    if (n > 8 && !(e && atoi(e) > 0)) n = 8;
    if (n > 64) n = 64;
    atomic_store(&g_game_threads, n);
    if (atomic_load(&g_from_launcher) && !(e && atoi(e) > 0)) {  // (the start screen: all but two of the CPU's threads)
        int l = cores - 2;
        n = l < n ? n : l > 32 ? 32 : l;
    }
    uint64_t t0 = SDL_GetTicksNS();
    LOG("[prep] %d files (%.0f MB) to scan for shaders, %d threads, %d parts done before\n", g_nfiles, g_bytes / 1e6, n, g_ndone);
    atomic_store(&g_running, 1);
    run_phase(-1, n);
    LOG("[prep] scanned in %.1f s: %d distinct stages, %d vertex + fragment pairs\n", (SDL_GetTicksNS() - t0) / 1e9, g_nstages, g_npairs);
    run_phase(STAGE_VERTEX, n);
    log_progress("vertex stages", t0);
    for (int i = 0; i < g_npairs; i++) {  // (each fragment stage: the outputs of the vertex stages it is paired with)
        Stage *v = stage_find(g_pairs[i].vs, false), *f = g_pairs[i].fs ? stage_find(g_pairs[i].fs, false) : NULL;
        if (!v || !f || !v->out_mask || v->out_mask == ~0u) continue;
        bool have = false;
        for (int k = 0; k < f->nmasks && !have; k++) have = f->masks[k] == v->out_mask;
        if (!have && f->nmasks < 8) f->masks[f->nmasks++] = v->out_mask;
    }
    run_phase(STAGE_FRAGMENT, n);
    log_progress("fragment stages", t0);
    // The parts' cache (once: a save blocks every pipeline created meanwhile, device.c), then the last ids.
    uint64_t ts = SDL_GetTicksNS();
    vk_parts_cache_save();
    done_save();
    complete_write();
    atomic_store(&g_end_ns, SDL_GetTicksNS());
    atomic_store(&g_complete, 1);
    atomic_store(&g_running, 0);
    LOG("[prep] done in %.0f s (cache written in %.1f s)\n", (SDL_GetTicksNS() - t0) / 1e9, (SDL_GetTicksNS() - ts) / 1e9);
    if (getenv("LARECOMP_SHADER_PREP_EXIT")) {  // (tests: a first start that only prepares, then a run that plays)
        LOG("[prep] finished: exit\n");
        fflush(stderr);
        _exit(0);
    }
    for (int k = 0; k < STAGE_CAP; k++) { free(g_stages[k].code); g_stages[k].code = NULL; }
    free(g_files), g_files = NULL;
    free(g_pairs), g_pairs = NULL;
    return NULL;
}

// For the notice on screen (ui/menu.mm): 1 while preparing (*frac: 0..1; the scan counts as the first 10%), 2 for a few
// seconds after it ended, 0 otherwise. A start with everything done before ends in about a minute (scan + translation).
int shader_prep_progress(float *frac) {
    if (atomic_load(&g_running)) {
        float scan = g_nfiles ? (float)atomic_load(&g_files_done) / (float)g_nfiles : 0;
        float build = g_nstages && scan >= 1 ? (float)atomic_load(&g_stages_done) / (float)g_nstages : 0;
        *frac = 0.1f * scan + 0.9f * (build > 1 ? 1 : build);
        return 1;
    }
    uint64_t end = atomic_load(&g_end_ns);
    if (end && SDL_GetTicksNS() - end < 6 * 1000000000ull) { *frac = 1; return 2; }
    return 0;
}

static _Atomic bool g_started;

// Seconds left, from the rate since the stage building began (-1: not known yet). The scan comes first (~30 s).
float shader_prep_eta(void) {
    uint64_t t0 = atomic_load(&g_build_t0);
    int done = atomic_load(&g_stages_done);
    if (!atomic_load(&g_running) || !t0 || done < 200 || !g_nstages) return -1;
    double rate = done / ((SDL_GetTicksNS() - t0) / 1e9);  // stages per second
    int left = g_nstages - done;
    if (atomic_load(&g_in_game) && atomic_load(&g_from_launcher)) rate /= 2;  // (fewer threads once the game runs)
    return rate > 0 ? (float)(left > 0 ? left / rate : 0) : -1;
}

// 0 not started (or off), 1 preparing, 2 done for this build, GPU and driver (now or before).
int shader_prep_state(void) {
    if (atomic_load(&g_complete)) return 2;
    return atomic_load(&g_started) ? 1 : 0;
}

// The start screen starts the preparation as soon as the game files are set up (ui/launcher_lg.mm), so it runs while
// the player looks at the options instead of during the first minutes of play. It reads the image's file table itself
// (the game's own index, services/fs.c rom_index, belongs to the game's boot, mods and all): every file's name and its
// data's place in the image are all the preparation needs. The executables' own stages (14) are taken at the game's
// boot only, where the modules are loaded; their draws build them then, as before the preparation existed.
bool shader_prep_start_image(const char *romfs_bin) {
    const char *e = getenv("LARECOMP_SHADER_PREP");
    if (atomic_load(&g_started) || (e && !strcmp(e, "0"))) return atomic_load(&g_started);
    int fd = open(romfs_bin, O_RDONLY | O_BINARY);
    if (fd < 0) return false;
    uint64_t h[10];
    if (pread(fd, h, sizeof h, 0) != (ssize_t)sizeof h || h[0] != 0x50 || !h[8] || h[8] > (64ull << 20)) { close(fd); return false; }
    uint8_t *meta = malloc(h[8]);
    if (!meta || pread(fd, meta, h[8], (off_t)h[7]) != (ssize_t)h[8]) { free(meta); close(fd); return false; }
    for (uint64_t o = 0; o + 0x20 <= h[8];) {  // (parent, sibling, data offset, size, hash, name length, name)
        uint64_t data, size;
        uint32_t namelen;
        memcpy(&data, meta + o + 8, 8);
        memcpy(&size, meta + o + 0x10, 8);
        memcpy(&namelen, meta + o + 0x1C, 4);
        char name[256];
        uint32_t nl = namelen > 255 ? 255 : namelen;
        if (o + 0x20 + nl > h[8]) break;
        memcpy(name, meta + o + 0x20, nl);
        name[nl] = 0;
        add_file(name, fd, h[9] + data, size, NULL);
        o += 0x20 + ((namelen + 3ull) & ~3ull);
    }
    free(meta);
    if (!g_nfiles) { close(fd); return false; }
    atomic_store(&g_started, true);
    atomic_store(&g_from_launcher, 1);
    g_fd = fd;
    LOG("[prep] started from the start screen: %d files\n", g_nfiles);
    pthread_t t;
    pthread_create(&t, NULL, prep_main, NULL);
    pthread_detach(t);
    return true;
}

void shader_prep_romfs(int fd, void (*each)(int, void (*)(const char *, int, uint64_t, uint64_t, void *), void *)) {
    atomic_store(&g_in_game, 1);  // (the game opened its files: a preparation the start screen began steps aside)
    const char *e = getenv("LARECOMP_SHADER_PREP");
    if (atomic_load(&g_started) || (e && !strcmp(e, "0"))) return;
    atomic_store(&g_started, true);
    each(fd, add_file, NULL);
    if (!g_nfiles) return;
    g_fd = dup(fd);  // (the fs service closes its own on a reboot)
    pthread_t t;
    pthread_create(&t, NULL, prep_main, NULL);
    pthread_detach(t);
}
