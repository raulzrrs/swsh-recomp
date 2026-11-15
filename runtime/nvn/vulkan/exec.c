// Executes recorded NVN command lists with Vulkan (port of renderer.m; same structure, same behaviour).
#include <SDL3/SDL_cpuinfo.h>
#include <immintrin.h>
#include <SDL3/SDL_timer.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../../kernel.h"
#include "shaders.h"
#include "vk.h"
#include "../renderer.h"
#include "../../settings.h"
#include "dlss.h"

static uint64_t g_draws_total;  // (never reset: LARECOMP_COUNTER_LOG)
static void watch_tex(NvnTexture *t, const char *what, const NvnTexture *from);  // (LARECOMP_WATCH_TEX, below)
static uint64_t stat_draws, stat_skipped, stat_passes, stat_dispatches, stat_wait_ns, stat_flushes, stat_feedback_copies;
static uint64_t stat_pf_walks, stat_pf_pipes, stat_pf_progs, stat_pf_ns;  // (prefetch, below)
static _Atomic uint64_t stat_gpl_links, stat_gpl_link_ns, stat_gpl_parts, stat_gpl_part_ns, stat_gpl_used, stat_gpl_failed;  // (pipeline parts, below)
static uint64_t stat_copy_flushed, stat_copy_inline;  // CopyBufferToBuffer: with a GPU flush first / without
static uint64_t stat_nvn_barriers, stat_barrier_bits[0x200];  // nvnCommandBufferBarrier: applied (nonzero), and per bit value
static uint64_t stat_holds, stat_hold_ns, stat_hold_max_ns, stat_hold_claims, stat_hold_timeouts;  // (wait_built)
static uint64_t stat_alpha_func[16];  // draws by NVN alpha test function (ColorState byte 2; 8 = ALWAYS, the default)
// Skipped draws by reason (logged with the stats): 0 no program bound, 1 program still compiling, 2 program failed,
// 3 no program object, 4 primitive mode, 5 pipeline still building, 6 pipeline failed.
static uint64_t stat_skip_why[8];
uint64_t g_skin_watch;  // (debug, LARECOMP_DUMP_SKIN: a vertex cbuf watched across frames)
// docs/letsgo/PLANO-3D.md experiments, logged with the stats: [0] palettes replaced by identity, [1] skinned-draw cbufs
// read in place (UBO_DIRECT), [2] skinned-draw cbufs UBO_DIRECT could not place (copied), [3] one-bone shaders.
static uint64_t g_exp_count[4];
#define SKIP(why) (stat_skipped++, stat_skip_why[(why) & 7]++)
static uint64_t stat_upload_bytes;
// LARECOMP_VK_PROF=1: render-thread time per category in the fps log (exec = whole command lists, draw = draw(),
// sets = descriptor sets incl. cbuf copies, vtx = index scan + vertex copies, submit = vkQueueSubmit2)
static int prof = -1;
static uint64_t prof_exec, prof_draw, prof_sets, prof_vtx, prof_submit, prof_pre, prof_key, prof_dyn, prof_tex, prof_up, prof_gpu;
static VkQueryPool prof_qp;
static uint64_t cum_exec, cum_wait, cum_draws, cum_submits;  // (profiling) never reset: hitch log
static uint32_t prof_end_line[4096];  // passes ended, by the exec.c line that ended them (0: submit / present)
uint64_t prof_views_created;  // textures.c vk_image_view cache misses
uint64_t prof_up_ns = ~0ull;  // textures.c upload checks (~0: profiling off)
#define PROF_T0() uint64_t prof_t0_ = prof ? SDL_GetTicksNS() : 0
#define PROF_ADD(v) do { if (prof) v += SDL_GetTicksNS() - prof_t0_; } while (0)
static _Atomic int stat_compiles;
static _Atomic uint64_t stat_translate_ns, stat_spirv_ns, stat_pipe_ns;  // CPU time translating / compiling programs
static uint64_t stat_shader_wait_ns;  // draws waiting for their program / pipeline (wait_built)
static _Atomic int new_pipes;  // pipelines created since the pipeline cache was last saved
static uint64_t op_counts[CMD_COUNT];
static uint64_t stat_vtx_copied, stat_vtx_direct;  // vertex / index ranges of the draws: copied to VRAM / read in place
bool g_debug_frame;
static char stats_text[2048];
void vk_collect_dead(void);

// ================================================================== submissions
// A submission = optional upload command buffer + the main one, signalling the timeline with its number.
// The ring of submissions: the render thread can be at most this many submissions ahead of the GPU (a slot reused
// waits for its previous submission). The game submits ~220 command lists a frame in Postwick, so 8 slots held the
// render thread to the GPU nearly at every list. LARECOMP_VK_SUBS=<n> (1..256, default 8).
#define SUBS_MAX 256
// Submission slots in flight (LARECOMP_VK_SUBS). A town ends ~60 submissions a frame (one per texture readback): with 8
// slots the render thread waited for the GPU to free one, 620 ms of every 2 s (town uncapped 65 fps; 128 slots: 90).
// The open world is unchanged by it. (8 was chosen when each slot's first descriptor pool was full size, ~7 MB.)
static int SUBS = 128;
typedef struct {
    VkCommandBuffer main, upload;
    bool main_begun, upload_begun, ts;  // ts: GPU timestamps written (profiling)
    uint64_t seq;
    VkDescriptorPool pools[16];
    int npools, cur_pool;
} Sub;
static Sub subs[SUBS_MAX];
static int cur_sub;
static uint64_t submit_seq;  // last submitted
static pthread_mutex_t seq_lock = PTHREAD_MUTEX_INITIALIZER;
static bool in_pass;

uint64_t vk_completed_seq(void) {
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &v);
    return v;
}

// ---- deferred texture -> buffer readbacks
// nvnCommandBufferCopyTextureToBuffer writes guest memory from the GPU. It waited for the GPU to go idle so the result
// could be copied right away: ~3 times a frame in the overworld, 0.6 s of every 2 s spent waiting. As on the console,
// the result now lands when the GPU has run it: each readback has its own host buffer and the submission number it was
// recorded in, and is copied into guest memory once that submission is done - when anyone waits for it (vk_wait_seq:
// the game's nvnSyncWait, flushes), at the start of each submitted list, or before this thread reads or writes guest
// memory it overlaps (readback_sync_range). LARECOMP_READBACK_SYNC=1: wait at the copy, as before.
typedef struct { uint64_t seq, dst, size; VkBuffer buf; VmaAllocation mem; void *ptr; VkImage img; VmaAllocation img_mem; bool superseded; } Readback;
#define RB_MAX 64
static Readback g_rb[RB_MAX];
static int g_nrb;
static pthread_mutex_t g_rb_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t stat_rb_deferred, stat_rb_sync, stat_rb_gmem_waits, stat_rb_superseded;
static void readbacks_done(uint64_t done) {
    if (!g_nrb) return;  // (a racy peek: the lock below decides)
    pthread_mutex_lock(&g_rb_lock);
    for (int i = 0; i < g_nrb;) {
        Readback *r = &g_rb[i];
        if (r->seq > done) { i++; continue; }
        if (!r->superseded) memcpy((void *)(uintptr_t)r->dst, r->ptr, r->size);
        vmaDestroyBuffer(g_vk.vma, r->buf, r->mem);
        if (r->img) vmaDestroyImage(g_vk.vma, r->img, r->img_mem);
        g_rb[i] = g_rb[--g_nrb];
    }
    pthread_mutex_unlock(&g_rb_lock);
}
void vk_wait_seq(uint64_t seq);
// A draw / dispatch whose shaders read guest memory in place (global memory) may read a readback's destination: the
// pending readbacks land first (they are all submitted; this only waits).
static void readbacks_before_gmem(void) {
    if (!g_nrb) return;
    uint64_t need = 0;
    pthread_mutex_lock(&g_rb_lock);
    for (int i = 0; i < g_nrb; i++) if (g_rb[i].seq > need) need = g_rb[i].seq;
    pthread_mutex_unlock(&g_rb_lock);
    if (need) vk_wait_seq(need > submit_seq ? submit_seq : need);
    stat_rb_gmem_waits++;
}
// Before reading or writing [a, a + n) of guest memory on the CPU: the readbacks that land there are finished first.
void readback_sync_range(uint64_t a, uint64_t n) {
    if (!g_nrb) return;
    uint64_t need = 0;
    pthread_mutex_lock(&g_rb_lock);
    for (int i = 0; i < g_nrb; i++)
        if (!g_rb[i].superseded && g_rb[i].dst < a + n && a < g_rb[i].dst + g_rb[i].size && g_rb[i].seq > need) need = g_rb[i].seq;
    pthread_mutex_unlock(&g_rb_lock);
    if (!need) return;
    vk_wait_seq(need > submit_seq ? submit_seq : need);  // (each deferred readback is submitted right after it is recorded)
}

// LARECOMP_VK_WAITPROF=1: wait time by caller (return address), the top ones logged with the stats line
#define WAITPROF_N 32
static struct { void *ra; uint64_t ns, n; } g_waitprof[WAITPROF_N];
static pthread_mutex_t g_waitprof_lock = PTHREAD_MUTEX_INITIALIZER;
void vk_wait_seq(uint64_t seq) {
    if (!seq) return;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_vk.timeline,
                              .pValues = &seq};
    uint64_t t0 = SDL_GetTicksNS();
    VkResult r = vkWaitSemaphores(g_vk.dev, &wi, 10ull * 1000000000ull);
    stat_wait_ns += SDL_GetTicksNS() - t0;
    static int wp = -1;
    if (wp < 0) wp = getenv("LARECOMP_VK_WAITPROF") != NULL;
    if (wp) {
        void *ra = __builtin_return_address(0);
        uint64_t dt = SDL_GetTicksNS() - t0;
        pthread_mutex_lock(&g_waitprof_lock);
        int i = 0;
        while (i < WAITPROF_N && g_waitprof[i].ra && g_waitprof[i].ra != ra) i++;
        if (i < WAITPROF_N) { g_waitprof[i].ra = ra; g_waitprof[i].ns += dt; g_waitprof[i].n++; }
        pthread_mutex_unlock(&g_waitprof_lock);
    }
    cum_wait += SDL_GetTicksNS() - t0;
    if (r == VK_SUCCESS) readbacks_done(seq);
    if (r == VK_TIMEOUT) LOG("[vk] GPU did not finish submission %llu within 10 s (device lost?)\n", (unsigned long long)seq);
    else if (r != VK_SUCCESS) {
        extern void vk_devlost_dump(void);
        vk_devlost_dump();
        vk_device_fault_report();
        FATAL("vkWaitSemaphores: %d (device lost)\n", r);
    }
}

uint64_t vk_pending_seq(void) { return submit_seq + 1; }

static Sub *sub(void) {
    Sub *s = &subs[cur_sub];
    if (s->seq != submit_seq + 1) {  // slot reused: its previous submission must be done
        vk_wait_seq(s->seq);
        if (s->ts) {  // (profiling) GPU time of that submission
            uint64_t t[2];
            if (vkGetQueryPoolResults(g_vk.dev, prof_qp, (uint32_t)cur_sub * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] > t[0])
                prof_gpu += (uint64_t)((double)(t[1] - t[0]) * g_vk.props.limits.timestampPeriod);
            s->ts = false;
        }
        for (int i = 0; i < s->npools; i++) vkResetDescriptorPool(g_vk.dev, s->pools[i], 0);
        s->cur_pool = 0;
        pthread_mutex_lock(&seq_lock);
        s->main_begun = s->upload_begun = false;
        s->seq = submit_seq + 1;
        pthread_mutex_unlock(&seq_lock);
    }
    return s;
}

static void begin_cb(VkCommandBuffer *cb, bool *begun) {
    if (*begun) return;
    if (!*cb) {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_vk.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VK_CHECK(vkAllocateCommandBuffers(g_vk.dev, &ai, cb));
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(*cb, &bi));
    pthread_mutex_lock(&seq_lock);
    *begun = true;
    pthread_mutex_unlock(&seq_lock);
}

static void gp_mark(VkCommandBuffer cb, const char *fmt, ...);
VkCommandBuffer vk_cmd(void) {
    Sub *s = sub();
    if (!s->main_begun) {
        begin_cb(&s->main, &s->main_begun);
        gp_mark(s->main, "(submission start; includes GPU idle before it)");
        if (prof && prof_qp) {
            vkCmdResetQueryPool(s->main, prof_qp, (uint32_t)cur_sub * 2, 2);
            vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_NONE, prof_qp, (uint32_t)cur_sub * 2);
            s->ts = true;
        }
    }
    return s->main;
}

VkCommandBuffer vk_upload_cmd(void) {
    Sub *s = sub();
    begin_cb(&s->upload, &s->upload_begun);
    return s->upload;
}

void vk_full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void ring_retire(void);

// ---- LARECOMP_GPUPROF_AT=<frame>: GPU time per pass / dispatch / copy of that frame, logged as "[gpuprof]".
// A timestamp (ALL_COMMANDS: after everything before it) at each mark; a row = the work since the previous mark.
bool g_gpuprof_frame;
static VkQueryPool gp_qp;
static int gp_n;
#define GP_MAX 4096
static char gp_label[GP_MAX][112];
static uint32_t gp_draws[GP_MAX];
// LARECOMP_GPUPROF_STATS=1 (with LARECOMP_GPUPROF_DRAWS): the pipeline statistics of every draw of the profiled frames
// (input vertices, vertex shader invocations, clipped primitives, fragment shader invocations), printed on its row.
static VkQueryPool gps_qp;
static int gps_n;
static int gp_stat[GP_MAX];  // mark -> its index in gps_qp + 1 (0: none)
// LARECOMP_VK_DEVLOST_TRACE=1: the last 64 GPU operations, dumped on device loss (with LARECOMP_SYNC_SUBMIT=1 the
// dump ends with the command list that faulted).
static char dl_ring[64][128];
static unsigned dl_n;
static int dl_on = -1;
static void dl_note(const char *fmt, ...) {
    if (dl_on < 0) dl_on = getenv("LARECOMP_VK_DEVLOST_TRACE") ? atoi(getenv("LARECOMP_VK_DEVLOST_TRACE")) : 0;
    if (!dl_on) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dl_ring[dl_n++ & 63], sizeof dl_ring[0], fmt, ap);
    va_end(ap);
}
void vk_dl_note_upload(void *img, uint32_t w, uint32_t h, uint32_t lv) { dl_note("  upload %p %ux%u lv%u", img, w, h, lv); }
void vk_devlost_dump(void) {
    if (!dl_on) return;
    LOG("[vk] last GPU operations before the device loss (oldest first):\n");
    for (unsigned i = dl_n > 64 ? dl_n - 64 : 0; i < dl_n; i++) LOG("[vk]   %s\n", dl_ring[i & 63]);
}
uint64_t vk_frame_counter(void);
static uint64_t gp_frame;  // the frame counter when the profiled frame began (parity: LARECOMP_EXP_ALT)
static void gp_mark(VkCommandBuffer cb, const char *fmt, ...) {
    if (!g_gpuprof_frame || gp_n >= GP_MAX - 1) return;
    if (!gp_qp) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = GP_MAX};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &gp_qp) != VK_SUCCESS) { g_gpuprof_frame = false; return; }
    }
    if (!gp_n) {
        gp_frame = vk_frame_counter();
        vkCmdResetQueryPool(cb, gp_qp, 0, GP_MAX);
        if (getenv("LARECOMP_GPUPROF_STATS")) {
            if (!gps_qp) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
                                            .queryCount = GP_MAX,
                                            .pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
                                                                  VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
                                                                  VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                                                                  VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT};
                if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &gps_qp) != VK_SUCCESS) gps_qp = VK_NULL_HANDLE;
            }
            if (gps_qp) vkCmdResetQueryPool(cb, gps_qp, 0, GP_MAX);
            gps_n = 0;
        }
    }
    gp_stat[gp_n] = 0;
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, gp_qp, (uint32_t)gp_n);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(gp_label[gp_n], sizeof gp_label[0], fmt, ap);
    va_end(ap);
    gp_draws[gp_n] = 0;
    gp_n++;
}
void renderer_flush(void);
void vk_gpuprof_report(void) {
    if (!gp_n) return;
    renderer_flush();
    static uint64_t t[GP_MAX];
    if (vkGetQueryPoolResults(g_vk.dev, gp_qp, 0, (uint32_t)gp_n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod / 1e6, total = (t[gp_n - 1] - t[0]) * per;
        LOG("[gpuprof] %d marks, %.2f ms first to last (frame %llu; rows: the work after each mark, until the next)\n", gp_n, total,
            (unsigned long long)gp_frame);
        static uint64_t stats[GP_MAX * 4];
        bool have_stats = gps_qp && gps_n &&
            vkGetQueryPoolResults(g_vk.dev, gps_qp, 0, (uint32_t)gps_n, sizeof stats, stats, 32, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS;
        for (int i = 0; i + 1 < gp_n; i++) {
            double ms = (t[i + 1] - t[i]) * per;
            bool st = have_stats && gp_stat[i];
            LOG("[gpuprof] %6.3f ms %5.1f%%  %s%s", ms, 100 * ms / (total > 0 ? total : 1), gp_label[i], gp_draws[i] || st ? "" : "\n");
            if (gp_draws[i]) LOG(", %u draws\n", gp_draws[i]);
            if (st) {
                const uint64_t *q = &stats[(gp_stat[i] - 1) * 4];  // (the enabled bits, in order: IA vertices, VS invocations, clip prims, FS invocations)
                LOG(" | ia %llu vsinv %llu clipprim %llu fsinv %llu\n", (unsigned long long)q[0], (unsigned long long)q[1],
                    (unsigned long long)q[2], (unsigned long long)q[3]);
            }
        }
    }
    gp_n = 0;
}

static void ring_flush_gpu(void);
// Submit the pending work (if any). wait/signal: binary semaphores for presentation (may be null).
void vk_submit_extra(VkSemaphore wait, VkSemaphore signal, VkCommandBuffer extra) {
    (void)extra;
    vk_end_pass();
    ring_flush_gpu();
    Sub *s = sub();
    if (!s->main_begun && !s->upload_begun && !wait && !signal) return;
    VkCommandBufferSubmitInfo cbs[2];
    int n = 0;
    if (s->upload_begun) {
        vk_full_barrier(s->upload);  // uploads complete before anything in the main buffer reads them
        VK_CHECK(vkEndCommandBuffer(s->upload));
        cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->upload};
    }
    vk_cmd();  // (an empty main buffer still carries the semaphores)
    if (s->ts) vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_qp, (uint32_t)cur_sub * 2 + 1);
    VK_CHECK(vkEndCommandBuffer(s->main));
    cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->main};
    VkSemaphoreSubmitInfo sig[2] = {{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_vk.timeline,
                                     .value = submit_seq + 1, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                                    {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = signal,
                                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    VkSemaphoreSubmitInfo wt = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = wait,
                                .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = (uint32_t)n, .pCommandBufferInfos = cbs,
                        .signalSemaphoreInfoCount = signal ? 2 : 1, .pSignalSemaphoreInfos = sig,
                        .waitSemaphoreInfoCount = wait ? 1 : 0, .pWaitSemaphoreInfos = &wt};
    {
        PROF_T0();
        dl_note("submit %llu", (unsigned long long)submit_seq + 1);
        VkResult sr = vkQueueSubmit2(g_vk.queue, 1, &si, VK_NULL_HANDLE);
        if (sr == VK_ERROR_DEVICE_LOST) vk_devlost_dump();
        VK_CHECK(sr);
        PROF_ADD(prof_submit);
    }
    pthread_mutex_lock(&seq_lock);
    submit_seq++;
    cur_sub = (cur_sub + 1) % SUBS;
    pthread_mutex_unlock(&seq_lock);
    stat_flushes++;
    cum_submits++;
    ring_retire();
    vk_collect_dead();
}

void renderer_submit(void) { vk_submit_extra(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE); }

void renderer_flush(void) {
    renderer_submit();
    vk_wait_seq(submit_seq);
}

void renderer_sync(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    vk_wait_seq(s);
}

uint64_t renderer_fence_seq(void) {
    pthread_mutex_lock(&seq_lock);
    Sub *s = &subs[cur_sub];
    // A reused slot keeps its old begun flags until sub() prepares it: an empty queue's fence must name the last
    // submission, not a future one that submit's empty-work fast path never signals (LAS Recomp 37b7ff6).
    bool pending = s->seq == submit_seq + 1 && (s->main_begun || s->upload_begun);
    uint64_t q = submit_seq + (pending ? 1 : 0);
    pthread_mutex_unlock(&seq_lock);
    return q;
}

void renderer_wait_seq(uint64_t seq) {
    if (!seq) return;
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    vk_wait_seq(seq > s ? s : seq);  // not submitted yet (other thread): everything submitted so far
}

VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout layout) {
    Sub *s = sub();
    for (;;) {
        if (s->cur_pool == s->npools) {
            if (s->npools == 16) FATAL("descriptor pools exhausted\n");
            // A submission's pools grow as it fills them: 256 sets, then 512... up to 8192 (16 pools: >100k sets). A
            // full-size first pool cost ~7 MB per submission slot, too much for the many short submissions of a town
            // (~60 texture readbacks a frame, each ending one); most never need more than the first.
            uint32_t sets = 256u << (s->npools < 5 ? s->npools : 5), f = 8192 / sets;
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768 / f}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 / f},
                                            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768 / f},
                                            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024 / f}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024 / f}};
            VkDescriptorPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = sets,
                                             .poolSizeCount = 5, .pPoolSizes = sizes};
            VK_CHECK(vkCreateDescriptorPool(g_vk.dev, &ci, NULL, &s->pools[s->npools++]));
        }
        VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .descriptorPool = s->pools[s->cur_pool], .descriptorSetCount = 1, .pSetLayouts = &layout};
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(g_vk.dev, &ai, &set);
        if (r == VK_SUCCESS) return set;
        s->cur_pool++;
    }
}

// ================================================================== transient memory ring
// The largest index of a draw (bounds the vertex data copied for it), ignoring the restart value: AVX2 where the CPU has it (the
// game code needs it too), a plain loop elsewhere. A draw scans a few thousand indices: ~1.4 us scalar, ~0.15 us here.
__attribute__((target("avx2"))) static uint32_t scan_max16_avx2(const uint16_t *q, uint32_t n, uint16_t restart) {
    __m256i acc = _mm256_setzero_si256(), rs = _mm256_set1_epi16((short)restart);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(q + i));
        acc = _mm256_max_epu16(acc, _mm256_andnot_si256(_mm256_cmpeq_epi16(v, rs), v));
    }
    uint16_t t[16];
    _mm256_storeu_si256((__m256i *)t, acc);
    uint32_t m = 0;
    for (int k = 0; k < 16; k++) if (t[k] > m) m = t[k];
    for (; i < n; i++) if (q[i] != restart && q[i] > m) m = q[i];
    return m;
}
__attribute__((target("avx2"))) static uint32_t scan_max32_avx2(const uint32_t *q, uint32_t n, uint32_t restart) {
    __m256i acc = _mm256_setzero_si256(), rs = _mm256_set1_epi32((int)restart);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(q + i));
        acc = _mm256_max_epu32(acc, _mm256_andnot_si256(_mm256_cmpeq_epi32(v, rs), v));
    }
    uint32_t t[8];
    _mm256_storeu_si256((__m256i *)t, acc);
    uint32_t m = 0;
    for (int k = 0; k < 8; k++) if (t[k] > m) m = t[k];
    for (; i < n; i++) if (q[i] != restart && q[i] > m) m = q[i];
    return m;
}
static uint32_t scan_max_index(const void *src, uint32_t isz, uint32_t n, uint32_t restart) {
    static int avx2 = -1;
    if (avx2 < 0) avx2 = __builtin_cpu_supports("avx2");
    uint32_t m = 0;
    if (isz == 2) {
        if (avx2) return scan_max16_avx2(src, n, (uint16_t)restart);
        const uint16_t *q = src;
        for (uint32_t i = 0; i < n; i++) if (q[i] > m && q[i] != (uint16_t)restart) m = q[i];
    } else {
        if (avx2) return scan_max32_avx2(src, n, restart);
        const uint32_t *q = src;
        for (uint32_t i = 0; i < n; i++) if (q[i] > m && q[i] != restart) m = q[i];
    }
    return m;
}

// Two pools of 64 MB chunks, a chunk reused once every submission that allocated from it has completed.
// Pool 0: host-visible chunks (VRAM through the BAR when the GPU has one) the GPU reads in place: constants and the sources of
// texture uploads. Pool 1 (ring_gpu_enabled, discrete GPUs): the vertex and index data of the draws, in VRAM. The CPU writes a
// staging chunk in system memory and each submission starts with a copy of what was written since the last one (ring_flush_gpu),
// so the GPU fetches vertices from VRAM instead of reading the game's memory over PCIe once per vertex invocation (an
// instanced draw re-read its per-instance streams for every vertex: ~0.8 ms for 71 thousand invocations, docs/PESQUISA-GPU.md).
#define CHUNK_SIZE (64ull << 20)
typedef struct { VkBuffer buf, dev; VmaAllocation mem, dmem; uint8_t *ptr; uint64_t size, used, seq, flushed; } Chunk;
typedef struct { Chunk *c; int n, active; bool gpu; } Pool;
static Pool pools[2] = {{.active = -1}, {.active = -1, .gpu = true}};

// LARECOMP_VERTEX_READ=direct|copy|auto (default auto: copies on a discrete GPU, in place on an integrated one, whose memory
// is the system's). The draws' own ranges the GPU writes stay in place either way (gpuw below).
static bool ring_gpu_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LARECOMP_VERTEX_READ");
        if (!e) e = g_settings.vertex_read == VREAD_DIRECT ? "direct" : g_settings.vertex_read == VREAD_COPY ? "copy" : NULL;
        if (e && !strcmp(e, "direct")) on = 0;
        else if (e && (!strcmp(e, "copy") || !strcmp(e, "hybrid"))) on = 1;
        else on = g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        LOG("[vk] vertex and index data: %s\n", on ? "copied to VRAM (ranges the GPU writes are read in place)" : "read in place from the game's memory");
    }
    return on;
}

static int new_chunk(Pool *pl, uint64_t size) {
    pl->c = realloc(pl->c, (pl->n + 1) * sizeof *pl->c);
    Chunk *c = &pl->c[pl->n];
    memset(c, 0, sizeof *c);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = pl->gpu ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                              : VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = pl->gpu ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO,
                                  .flags = (pl->gpu ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &c->buf, &c->mem, &info));
    c->ptr = info.pMappedData;
    c->size = size;
    if (pl->gpu) {
        VkBufferCreateInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                                 .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo da = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &di, &da, &c->dev, &c->dmem, NULL));
    }
    {   // (where the GPU reads the copies from: VRAM through the BAR, or system memory over PCIe)
        VkMemoryPropertyFlags mp = 0;
        vmaGetAllocationMemoryProperties(g_vk.vma, pl->gpu ? c->dmem : c->mem, &mp);
        LOG("[vk] ring %d chunk %d: %llu MB, %s%s\n", pl->gpu ? 1 : 0, pl->n, (unsigned long long)(size >> 20),
            mp & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "device-local" : "system memory",
            mp & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " host-visible" : "");
    }
    return pl->n++;
}

static void ring_retire(void) {}  // chunks are recycled lazily in transient_from

static VkSpan transient_from(Pool *pl, uint64_t size, uint64_t align) {
    if (align < 16) align = 16;
    uint64_t pending = vk_pending_seq();
    if (pl->active >= 0) {
        Chunk *c = &pl->c[pl->active];
        uint64_t off = (c->used + align - 1) & ~(align - 1);
        if (off + size <= c->size) {
            c->used = off + size;
            c->seq = pending;
            stat_upload_bytes += size;
            return (VkSpan){pl->gpu ? c->dev : c->buf, off, c->ptr + off};
        }
    }
    uint64_t done = vk_completed_seq();
    int pick = -1;
    for (int i = 0; i < pl->n; i++)
        if (i != pl->active && pl->c[i].seq <= done && pl->c[i].size >= size) { pick = i; break; }
    if (pick < 0) pick = new_chunk(pl, size > CHUNK_SIZE ? size : CHUNK_SIZE);
    pl->active = pick;
    Chunk *c = &pl->c[pick];
    c->used = size;
    c->flushed = 0;
    c->seq = pending;
    stat_upload_bytes += size;
    return (VkSpan){pl->gpu ? c->dev : c->buf, 0, c->ptr};
}

VkSpan vk_transient(uint64_t size, uint64_t align) { return transient_from(&pools[0], size, align); }
// Vertex / index data of a draw: in VRAM (valid once the submission's first copy has run), or as vk_transient where that is off.
static VkSpan transient_gpu(uint64_t size, uint64_t align) { return transient_from(&pools[ring_gpu_enabled() ? 1 : 0], size, align); }

// The start of a submission: the staging bytes written since the last one go to VRAM (the upload buffer runs first, then a
// barrier, vk_submit_extra).
static void ring_flush_gpu(void) {
    Pool *pl = &pools[1];
    for (int i = 0; i < pl->n; i++) {
        Chunk *c = &pl->c[i];
        if (c->used <= c->flushed) continue;
        VkBufferCopy rg = {c->flushed, c->flushed, c->used - c->flushed};
        vkCmdCopyBuffer(vk_upload_cmd(), c->buf, c->dev, 1, &rg);
        c->flushed = c->used;
    }
}

// GPU -> CPU copies: the ring is write-combined (often device-local over the BAR), where CPU reads crawl (a
// 1280x720 screenshot took 7 s). Readbacks use one host-cached buffer instead. Every user waits for the GPU and
// reads the data before the next readback (render thread), so a single grow-only buffer suffices.
VkSpan vk_readback(uint64_t size) {
    static VkBuffer buf;
    static VmaAllocation mem;
    static uint8_t *ptr;
    static uint64_t cap;
    if (size > cap) {
        if (buf) { vk_wait_seq(submit_seq); vmaDestroyBuffer(g_vk.vma, buf, mem); }
        cap = (size + (16 << 20) - 1) & ~((16ull << 20) - 1);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &buf, &mem, &info));
        ptr = info.pMappedData;
    }
    return (VkSpan){buf, 0, ptr};
}

VkSpan vk_transient_copy(const void *data, uint64_t size, uint64_t align) {
    VkSpan s = vk_transient(size ? size : 16, align);
    if (size) memcpy(s.ptr, data, size);
    return s;
}

// Per-draw constants that usually repeat the previous draw's (scale table, tint, driver constants): reuse that copy
// while it's identical, within one submission. slot: one per kind of data.
enum { DEDUPE_RSCALE_V, DEDUPE_RSCALE_F, DEDUPE_TINT, DEDUPE_DRV_V, DEDUPE_DRV_F, DEDUPE_DRV_C, DEDUPE_SLOTS };
static VkSpan copy_dedupe(int slot, const void *data, uint64_t size, uint64_t align) {
    static struct { uint64_t seq, size; uint8_t data[0x800]; VkSpan span; } last[DEDUPE_SLOTS];
    uint64_t seq = vk_pending_seq();
    if (size > sizeof last[0].data) return vk_transient_copy(data, size, align);
    if (last[slot].seq == seq && last[slot].size == size && !memcmp(last[slot].data, data, size)) return last[slot].span;
    VkSpan s = vk_transient_copy(data, size, align);
    last[slot].seq = seq; last[slot].size = size; memcpy(last[slot].data, data, size); last[slot].span = s;
    return s;
}

// Guest memory copied once per submission: draws binding the same range share the copy.
#define GCOPY_TAB 4096
static struct { uint64_t addr, size, seq; VkSpan span; bool gpu; } gcopy[GCOPY_TAB];
static uint64_t gcopy_gen;  // bumped to drop every shared copy (guest memory written mid-list)
static uint64_t stat_gcopy_drops, stat_gcopy_kept;
static void guest_copy_flush(void) { gcopy_gen++; stat_gcopy_drops++; }
static void pc_dirty(uint64_t a, uint64_t n);  // (persistent copies, below)
// The 64 KB pages (hashed into 2^16 bits) that hold the shared copies of the current submission / generation. This thread
// writes guest memory ~200 times a frame (buffer copies, uniform updates) and the game issues ~500 barriers: dropping every
// copy each time made the shared copies useless, and a heavy overworld view copied ~31 MB of vertex data a frame.
// Now a write drops them only when it lands on a page that has a copy, and a barrier only after a shader that stores
// to guest memory (the only GPU writes the copies can miss). LARECOMP_GCOPY_PRECISE=0: drop on every write / barrier.
static uint64_t gcopy_pages[1024], gcopy_pages_key;
static bool g_gpu_stored;  // a shader that stores to guest memory was recorded since the copies were last dropped
static bool gcopy_precise(void) {
    static int on = -1;
    if (on < 0) on = !getenv("LARECOMP_GCOPY_PRECISE") || atoi(getenv("LARECOMP_GCOPY_PRECISE")) != 0;
    return on;
}
static void gcopy_mark(uint64_t a, uint64_t n) {
    uint64_t key = vk_pending_seq() + (gcopy_gen << 40);
    if (key != gcopy_pages_key) { memset(gcopy_pages, 0, sizeof gcopy_pages); gcopy_pages_key = key; }
    for (uint64_t p = a >> 16, e = (a + (n ? n : 1) - 1) >> 16; p <= e; p++) gcopy_pages[(p >> 6) & 1023] |= 1ull << (p & 63);
}
// This thread wrote [a, a + n) of guest memory.
static void guest_copy_written(uint64_t a, uint64_t n) {
    pc_dirty(a, n);
    if (!gcopy_precise()) { guest_copy_flush(); return; }
    if (gcopy_pages_key != vk_pending_seq() + (gcopy_gen << 40)) return;  // (no copy shared in this submission yet)
    for (uint64_t p = a >> 16, e = (a + (n ? n : 1) - 1) >> 16; p <= e; p++)
        if (gcopy_pages[(p >> 6) & 1023] >> (p & 63) & 1) { guest_copy_flush(); return; }
    stat_gcopy_kept++;
}
// An NVN barrier: GPU writes before it are visible after it.
static void guest_copy_barrier(void) {
    if (!gcopy_precise() || g_gpu_stored) { guest_copy_flush(); g_gpu_stored = false; }
    else stat_gcopy_kept++;
}
static VkSpan transient_gpu_copy(const void *data, uint64_t size, uint64_t align) {
    VkSpan s = transient_gpu(size ? size : 16, align);
    if (size) memcpy(s.ptr, data, size);
    return s;
}
void readback_sync_range(uint64_t a, uint64_t n);
static VkSpan guest_copy_pool(uint64_t addr, uint64_t size, bool gpu) {  // (gpu: vertex / index data, in VRAM when that ring is on)
    readback_sync_range(addr, size);
    uint64_t seq = vk_pending_seq() + (gcopy_gen << 40), h = (addr * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (GCOPY_TAB - 1);
    if (gcopy[slot].seq == seq && gcopy[slot].addr == addr && gcopy[slot].size == size && gcopy[slot].gpu == gpu) return gcopy[slot].span;
    VkSpan s = gpu ? transient_gpu_copy((const void *)(uintptr_t)addr, size, 256) : vk_transient_copy((const void *)(uintptr_t)addr, size, 256);
    gcopy[slot].addr = addr; gcopy[slot].size = size; gcopy[slot].seq = seq; gcopy[slot].span = s; gcopy[slot].gpu = gpu;
    gcopy_mark(addr, size);
    return s;
}
static VkSpan guest_copy(uint64_t addr, uint64_t size) { return guest_copy_pool(addr, size, false); }

// Vertex data copied to VRAM, once per start address and submission: every draw of a mesh starts at the same address and wants a
// prefix of it (up to its largest index), so one copy serves them all. A per-vertex stream is copied whole (its bound range, up to
// 8 MB) the first time, so that later draws of other parts of the buffer don't copy it again (37 MB a frame became 16).
// ================================================================== persistent copies (static vertex / index data)
// The draws' vertex and index data was copied from guest memory into the VRAM ring once per submission: ~20 MB a frame
// in a town, ~31 MB in a heavy overworld view, most of it meshes that never change. With the guest window under write
// watch (platform: plat_written_pages), a copy is kept in VRAM across frames and redone only when a page of it was
// written (by the game, this thread's buffer copies / uniform updates, or a readback landing):
//  - each list starts with a scan of the watched 2 MB regions; a written page dirties its 64 KB chunk (pc_epoch);
//  - an entry (start address -> VRAM range, length) is valid while no chunk under it is dirtier than its copy;
//  - a new copy goes to a new VRAM range (the old one is freed once the GPU is past its last use), written by the
//    submission's upload buffer, which runs before its draws;
//  - data rewritten every few frames (CPU-animated meshes, UI) is marked dynamic and goes through the ring as before;
//  - index data also keeps its largest index, so the index scan runs once per copy instead of once per draw.
// LARECOMP_PCACHE=0 (or no write watch): the ring copies of before. Budget: LARECOMP_PCACHE_MB (default 1024) of VRAM.
#define PC_ARENA_SIZE (64ull << 20)
#define PC_ARENAS 64
#define PC_TAB (1 << 15)
#define PC_CHUNKS (1 << 17)
#define PC_REGIONS 4096
typedef struct { VkBuffer buf; VmaAllocation mem; VmaVirtualBlock vb; uint64_t size; } PcArena;
typedef struct {
    uint64_t addr, len, epoch, checked, last_seq, last_frame;
    uint64_t off;
    VmaVirtualAllocation va;
    int arena;
    uint64_t mi_len;                    // (index entries: the bytes scanned for max_index, 0: none)
    uint32_t max_index, restart, isz;   // (... the largest index for this restart value and index size)
    uint16_t dirt;                      // recent invalidations (dynamic data: decays)
    uint32_t dyn_until;                 // frame until which it is treated as dynamic
} PcEnt;
static PcArena pc_ar[PC_ARENAS];
static int pc_nar;
static PcEnt pc_tab[PC_TAB];
static struct { uint64_t chunk, epoch; } pc_chunk[PC_CHUNKS];  // dirty epoch of the 64 KB chunks under entries
static uint64_t pc_reg[PC_REGIONS];  // watched 2 MB regions, sorted
static int pc_nreg;
static uint64_t pc_epoch = 1;
static struct { int arena; VmaVirtualAllocation va; uint64_t seq; } pc_retired[8192];
static int pc_nret;
static uint64_t pc_budget, pc_used;
static uint64_t stat_pc_hits, stat_pc_copies, stat_pc_bytes, stat_pc_dyn, stat_pc_scans, stat_pc_scan_ns, stat_pc_dirty;
static int pc_on = -1;

static bool pc_enabled(void) {
    if (pc_on < 0) {
        uint64_t probe;
        pc_on = (!getenv("LARECOMP_PCACHE") || atoi(getenv("LARECOMP_PCACHE")) != 0) && ring_gpu_enabled() &&
                plat_written_pages(HEAP_BASE, 4096, &probe, 1) >= 0;
        pc_budget = (getenv("LARECOMP_PCACHE_MB") ? strtoull(getenv("LARECOMP_PCACHE_MB"), NULL, 10) : 1024) << 20;
        LOG("[vk] persistent vertex / index copies: %s\n", pc_on ? "on (write watch)" : "off");
    }
    return pc_on;
}

static uint64_t *pc_chunk_slot(uint64_t chunk, bool add) {  // dirty epoch of a 64 KB chunk (NULL: not watched)
    uint32_t h = (uint32_t)(chunk * 0x9E3779B97F4A7C15ull >> 47);
    for (int i = 0; i < 64; i++) {
        uint32_t k = (h + i) & (PC_CHUNKS - 1);
        if (pc_chunk[k].chunk == chunk + 1) return &pc_chunk[k].epoch;
        if (!pc_chunk[k].chunk) {
            if (!add) return NULL;
            pc_chunk[k].chunk = chunk + 1;
            pc_chunk[k].epoch = 0;
            return &pc_chunk[k].epoch;
        }
    }
    return NULL;  // (table crowded: the caller treats it as unknown)
}

static bool pc_watch(uint64_t a, uint64_t n) {  // register the chunks and regions under [a, a + n)
    for (uint64_t c = a >> 16, e = (a + n - 1) >> 16; c <= e; c++)
        if (!pc_chunk_slot(c, true)) return false;
    for (uint64_t r = a >> 21, e = (a + n - 1) >> 21; r <= e; r++) {
        int lo = 0, hi = pc_nreg;
        while (lo < hi) { int m = (lo + hi) / 2; if (pc_reg[m] < r) lo = m + 1; else hi = m; }
        if (lo < pc_nreg && pc_reg[lo] == r) continue;
        if (pc_nreg == PC_REGIONS) return false;
        {   // (a newly watched region: what was written before this point is in the copy about to be made)
            static uint64_t tmp[512];
            plat_written_pages(r << 21, 1ull << 21, tmp, 512);
        }
        memmove(&pc_reg[lo + 1], &pc_reg[lo], (size_t)(pc_nreg - lo) * sizeof pc_reg[0]);
        pc_reg[lo] = r;
        pc_nreg++;
    }
    return true;
}

// [a, a + n) of guest memory was written (this thread, or a scan): copies over it are stale.
static void pc_dirty(uint64_t a, uint64_t n) {
    if (pc_on <= 0 || !n) return;
    uint64_t e = ++pc_epoch;
    for (uint64_t c = a >> 16, last = (a + n - 1) >> 16; c <= last; c++) {
        uint64_t *s = pc_chunk_slot(c, false);
        if (s) *s = e;
    }
}

static void pc_free_range(int arena, VmaVirtualAllocation va, uint64_t seq) {
    if (pc_nret == (int)(sizeof pc_retired / sizeof pc_retired[0])) { vk_wait_seq(submit_seq); }
    if (pc_nret == (int)(sizeof pc_retired / sizeof pc_retired[0])) {  // (all done now)
        for (int i = 0; i < pc_nret; i++) vmaVirtualFree(pc_ar[pc_retired[i].arena].vb, pc_retired[i].va);
        pc_nret = 0;
    }
    pc_retired[pc_nret].arena = arena; pc_retired[pc_nret].va = va; pc_retired[pc_nret].seq = seq; pc_nret++;
}

// Start of a list: what was written since the last scan; finished frees.
static void pc_scan(void) {
    if (pc_on <= 0 || !pc_nreg) return;
    uint64_t t0 = SDL_GetTicksNS();
    uint64_t e = pc_epoch + 1;
    bool any = false;
    for (int i = 0; i < pc_nreg;) {  // contiguous regions in one call
        int j = i + 1;
        while (j < pc_nreg && pc_reg[j] == pc_reg[j - 1] + 1 && j - i < 64) j++;
        uint64_t a = pc_reg[i] << 21, n = (uint64_t)(j - i) << 21;
        static uint64_t *big;
        static long bigcap;
        long cap = (long)(n >> 12);
        if (cap > bigcap) { free(big); big = malloc((size_t)cap * sizeof *big); bigcap = cap; }
        long got = plat_written_pages(a, n, big, cap);
        if (got < 0 || got >= cap) {  // (unknown or too many: the whole span)
            for (uint64_t c = a >> 16; c < (a + n) >> 16; c++) { uint64_t *s = pc_chunk_slot(c, false); if (s) *s = e; }
            any = true;
        } else {
            for (long k = 0; k < got; k++) {
                uint64_t *s = pc_chunk_slot(big[k] >> 16, false);
                if (s) { *s = e; any = true; stat_pc_dirty++; }
            }
        }
        i = j;
    }
    if (any) pc_epoch = e;
    uint64_t done = vk_completed_seq();
    int m = 0;
    for (int i = 0; i < pc_nret; i++) {
        if (pc_retired[i].seq <= done) vmaVirtualFree(pc_ar[pc_retired[i].arena].vb, pc_retired[i].va);
        else pc_retired[m++] = pc_retired[i];
    }
    pc_nret = m;
    stat_pc_scans++;
    stat_pc_scan_ns += SDL_GetTicksNS() - t0;
}

static bool pc_valid(PcEnt *en) {
    if (en->checked == pc_epoch) return true;
    for (uint64_t c = en->addr >> 16, e = (en->addr + en->len - 1) >> 16; c <= e; c++) {
        uint64_t *s = pc_chunk_slot(c, false);
        if (!s || *s > en->epoch) return false;
    }
    en->checked = pc_epoch;
    return true;
}

static bool pc_alloc(uint64_t len, int *arena, VmaVirtualAllocation *va, uint64_t *off) {
    VmaVirtualAllocationCreateInfo ci = {.size = len, .alignment = 256};
    for (int i = pc_nar - 1; i >= 0; i--)
        if (vmaVirtualAllocate(pc_ar[i].vb, &ci, va, off) == VK_SUCCESS) { *arena = i; return true; }
    uint64_t size = len > PC_ARENA_SIZE ? (len + 0xFFFF) & ~0xFFFFull : PC_ARENA_SIZE;
    if (pc_nar == PC_ARENAS || pc_used + size > pc_budget) return false;
    PcArena *a = &pc_ar[pc_nar];
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &a->buf, &a->mem, NULL) != VK_SUCCESS) return false;
    VmaVirtualBlockCreateInfo vi = {.size = size};
    if (vmaCreateVirtualBlock(&vi, &a->vb) != VK_SUCCESS) { vmaDestroyBuffer(g_vk.vma, a->buf, a->mem); return false; }
    a->size = size;
    pc_used += size;
    LOG("[vk] persistent copies: arena %d, %llu MB (%llu MB in all)\n", pc_nar, (unsigned long long)(size >> 20), (unsigned long long)(pc_used >> 20));
    *arena = pc_nar++;
    return vmaVirtualAllocate(a->vb, &ci, va, off) == VK_SUCCESS;
}

// The VRAM copy of [addr, addr + len) (a longer copy of the same start serves too), or {0} for the ring path.
static VkSpan pc_get(uint64_t addr, uint64_t len, PcEnt **out) {
    if (out) *out = NULL;
    if (!pc_enabled() || len < 4096 || addr < HEAP_BASE) return (VkSpan){0};
    uint32_t h = (uint32_t)((addr >> 4) * 0x9E3779B97F4A7C15ull >> 49), at = PC_TAB;
    PcEnt *en = NULL;
    for (int i = 0; i < 16; i++) {
        uint32_t k = (h + i) & (PC_TAB - 1);
        if (pc_tab[k].addr == addr) { en = &pc_tab[k]; break; }
        if (!pc_tab[k].addr && at == PC_TAB) at = k;
    }
    uint64_t frame = vk_frame_counter();
    if (!en) {
        if (at == PC_TAB) return (VkSpan){0};  // (full here: the ring)
        en = &pc_tab[at];
        memset(en, 0, sizeof *en);
        en->addr = addr;
        en->arena = -1;
    }
    if (en->dyn_until > frame) { stat_pc_dyn++; return (VkSpan){0}; }
    bool stale = en->arena >= 0 && !pc_valid(en);
    if (en->arena >= 0 && !stale && en->len >= len) {
        en->last_seq = vk_pending_seq();
        en->last_frame = frame;
        stat_pc_hits++;
        if (out) *out = en;
        return (VkSpan){pc_ar[en->arena].buf, en->off, NULL};
    }
    // (re)copy: count how often this data changes
    if (en->arena >= 0) {
        if (stale) {
            if (en->last_frame + 8 >= frame) {
                if (++en->dirt >= 3) { en->dyn_until = (uint32_t)(frame + 600); en->dirt = 0; }
            } else en->dirt = 0;
        }
        pc_free_range(en->arena, en->va, en->last_seq);
        en->arena = -1;
        if (en->dyn_until > frame) { stat_pc_dyn++; return (VkSpan){0}; }
    }
    if (len < en->len && en->len <= (8ull << 20)) len = en->len;  // (keep the longest prefix seen)
    if (!pc_watch(addr, len)) return (VkSpan){0};
    int arena;
    VmaVirtualAllocation va;
    uint64_t off;
    if (!pc_alloc(len, &arena, &va, &off)) return (VkSpan){0};
    readback_sync_range(addr, len);
    uint64_t epoch = pc_epoch;
    VkSpan st = vk_transient_copy((const void *)(uintptr_t)addr, len, 16);
    VkBufferCopy rg = {st.off, off, len};
    vkCmdCopyBuffer(vk_upload_cmd(), st.buf, pc_ar[arena].buf, 1, &rg);
    en->len = len; en->arena = arena; en->va = va; en->off = off; en->epoch = epoch; en->checked = 0;
    en->mi_len = 0;
    en->last_seq = vk_pending_seq();
    en->last_frame = frame;
    stat_pc_copies++;
    stat_pc_bytes += len;
    if (out) *out = en;
    return (VkSpan){pc_ar[arena].buf, off, NULL};
}

static void pc_reset(void) {  // (reboot: guest memory is replaced)
    if (pc_on <= 0) return;
    vk_wait_seq(submit_seq);
    for (int i = 0; i < pc_nar; i++) {
        vmaClearVirtualBlock(pc_ar[i].vb);
        vmaDestroyVirtualBlock(pc_ar[i].vb);
        vmaDestroyBuffer(g_vk.vma, pc_ar[i].buf, pc_ar[i].mem);
    }
    pc_nar = 0; pc_used = 0; pc_nret = 0; pc_nreg = 0;
    memset(pc_tab, 0, sizeof pc_tab);
    memset(pc_chunk, 0, sizeof pc_chunk);
}

#define VCOPY_TAB 1024
static struct { uint64_t addr, len, seq; VkSpan span; } vcopy[VCOPY_TAB];
static VkSpan vertex_copy(uint64_t addr, uint64_t need, uint64_t bound, bool per_vertex) {
    readback_sync_range(addr, bound > need ? bound : need);
    uint64_t seq = vk_pending_seq() + (gcopy_gen << 40);
    uint32_t h = (uint32_t)((addr >> 4) * 0x9E3779B1u) & (VCOPY_TAB - 1), at = h;
    for (int i = 0; i < 4; i++) {
        uint32_t k = (h + i) & (VCOPY_TAB - 1);
        if (vcopy[k].addr == addr && vcopy[k].seq == seq) {
            if (vcopy[k].len >= need) return vcopy[k].span;
            at = k;  // (a longer prefix is needed now: a new copy replaces it)
            goto copy;
        }
        if (vcopy[k].seq != seq) at = k;  // (a free or stale slot)
    }
copy:;
    uint64_t len = per_vertex && bound <= (8ull << 20) && bound > need ? bound : need;
    VkSpan sp = pc_get(addr, len, NULL);
    if (sp.buf) {
        vcopy[at].addr = addr; vcopy[at].len = len; vcopy[at].seq = seq; vcopy[at].span = sp;
        return sp;
    }
    sp = transient_gpu_copy((const void *)(uintptr_t)addr, len, 256);
    vcopy[at].addr = addr; vcopy[at].len = len; vcopy[at].seq = seq; vcopy[at].span = sp;
    gcopy_mark(addr, len);
    return sp;
}

// ================================================================== worker threads (shader / pipeline builds)
// First-use compiles run off the render thread. Two queues: the jobs a draw asked for, then the prewarm's (pipeline
// recipes, below): a worker takes from the first while it has any. A draw that needs a job nobody has started yet
// takes it out of the queue and runs it itself (job_claim) instead of waiting behind the others (mk8-recomp's "claim").
// LARECOMP_SYNC_SHADERS=1 builds on first use, blocking (deterministic tests).
typedef struct Job { void (*fn)(void *); void *arg; struct Job *next; } Job;
static Job *jobs_head, *jobs_tail, *low_head, *low_tail;
static pthread_mutex_t jobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jobs_cond = PTHREAD_COND_INITIALIZER;

static void *worker(void *a) {
    (void)a;
    // Below the game and render threads: a first visit queues hundreds of builds on 16 workers, which took the CPU from
    // the threads that make the frame (an area entry with 300 builds pending ran its frames at a fraction of the speed).
    static int bg = -1;
    if (bg < 0) bg = !getenv("LARECOMP_WORKER_BG") || atoi(getenv("LARECOMP_WORKER_BG")) != 0;
    if (bg) plat_thread_background();
    for (;;) {
        pthread_mutex_lock(&jobs_lock);
        while (!jobs_head && !low_head) pthread_cond_wait(&jobs_cond, &jobs_lock);
        Job *j;
        if (jobs_head) { j = jobs_head; jobs_head = j->next; if (!jobs_head) jobs_tail = NULL; }
        else { j = low_head; low_head = j->next; if (!low_head) low_tail = NULL; }
        pthread_mutex_unlock(&jobs_lock);
        j->fn(j->arg);
        free(j);
    }
    return NULL;
}

static bool sync_shaders(void) {
    static int v = -1;
    if (v < 0) v = getenv("LARECOMP_SYNC_SHADERS") != NULL;
    return v;
}

// Shader translation runs on the workers too, then compiles on the same worker (translate_program). It had one
// dedicated thread (the translator used a static string ring, now per thread): on a first visit, ~380 programs took
// ~240 ms one after another while the workers idled. Off the game thread either way (up to ~90 ms per frame there).
static void queue_job(void (*fn)(void *), void *arg, bool low) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    if (!started) {
        started = true;
        // first entry into an area builds dozens of pipelines at once: use most of the CPU (draws wait for them)
        int n = SDL_GetNumLogicalCPUCores() - 4;
        if (n < 4) n = 4;
        if (n > 16) n = 16;
        for (int i = 0; i < n; i++) { pthread_t t; pthread_create(&t, NULL, worker, NULL); }
    }
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->next = NULL;
    pthread_mutex_lock(&jobs_lock);
    Job **head = low ? &low_head : &jobs_head, **tail = low ? &low_tail : &jobs_tail;
    if (*tail) (*tail)->next = j; else *head = j;
    *tail = j;
    pthread_cond_signal(&jobs_cond);
    pthread_mutex_unlock(&jobs_lock);
}
static void run_job(void (*fn)(void *), void *arg) { queue_job(fn, arg, false); }
static void run_job_low(void (*fn)(void *), void *arg) { queue_job(fn, arg, true); }

// Whole pipelines, when draws use the ones linked from parts (g_vk.gpl): two threads at idle priority. On the workers
// they took the CPU and the workers from the links and translations a new area needs at once (a town: 16 s of driver
// time queued in the same 2 s as 205 parts). They replace the linked pipeline whenever they are ready.
static Job *bg_head, *bg_tail;
static pthread_cond_t bg_cond = PTHREAD_COND_INITIALIZER;
static void *bg_worker(void *a) {
    (void)a;
    plat_thread_idle();
    for (;;) {
        pthread_mutex_lock(&jobs_lock);
        while (!bg_head) pthread_cond_wait(&bg_cond, &jobs_lock);
        Job *j = bg_head;
        bg_head = j->next;
        if (!bg_head) bg_tail = NULL;
        pthread_mutex_unlock(&jobs_lock);
        j->fn(j->arg);
        free(j);
    }
    return NULL;
}
static void run_job_bg(void (*fn)(void *), void *arg) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    pthread_mutex_lock(&jobs_lock);
    if (!started) {
        started = true;
        for (int i = 0; i < 2; i++) { pthread_t t; pthread_create(&t, NULL, bg_worker, NULL); }
    }
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->next = NULL;
    if (bg_tail) bg_tail->next = j; else bg_head = j;
    bg_tail = j;
    pthread_cond_signal(&bg_cond);
    pthread_mutex_unlock(&jobs_lock);
}
bool vk_shader_jobs_waiting(void) { return __atomic_load_n(&jobs_head, __ATOMIC_RELAXED) != NULL; }  // (prep.c yields to them)
static bool wait_built(_Atomic int *state, void (*fn)(void *), void *arg);  // (draws and dispatches: below)

// Takes the job (fn, arg) out of any queue if no worker has started it: the caller then runs it itself.
static bool job_claim(void (*fn)(void *), void *arg) {
    bool found = false;
    pthread_mutex_lock(&jobs_lock);
    for (int q = 0; q < 3 && !found; q++) {
        Job **head = q == 2 ? &bg_head : q ? &low_head : &jobs_head, **tail = q == 2 ? &bg_tail : q ? &low_tail : &jobs_tail, *prev = NULL;
        for (Job *j = *head; j; prev = j, j = j->next) {
            if (j->fn != fn || j->arg != arg) continue;
            if (prev) prev->next = j->next; else *head = j->next;
            if (*tail == j) *tail = prev;
            free(j);
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&jobs_lock);
    return found;
}

// ================================================================== state
typedef struct {
    NvnTexture *color[8];
    NvnTextureView cview[8];
    uint8_t has_cview[8];
    int ncolor;
    NvnTexture *depth;
    NvnTextureView dview;
    uint8_t has_dview;
} RtState;

typedef struct { uint64_t addr, size; } Range;

static struct {
    RtState rt;
    bool rt_dirty;
    NvnProgram *program;
    uint8_t blend[8][8];
    uint32_t channel_mask;
    uint8_t color_state[4], ds[8], ms[0x18], poly[4];
    float poly_offset[3];
    uint32_t attrib[16];
    int nattrib;
    uint32_t stream[16][2];
    int nstream;
    Range vbuf[16], ubuf[6][18], sbuf[6][16];
    uint64_t tex[6][32], img[6][8];
    NvnDescPool *tpool, *spool;
    float viewport[16][4];
    int32_t vswz[16][4];  // NVNviewportSwizzle x/y/z/w (0 = not set; 3 = NEGATIVE_Y: the draw comes out upside down)
    int32_t scissor[16][4];
    float depth_range[16][2];
    float blend_color[4];
    uint32_t stencil_ref[2], stencil_mask[2], stencil_vmask[2];
    bool prim_restart;
    bool depth_clamp;  // nvnCommandBufferSetDepthClamp: fragments beyond the near / far planes clamped instead of clipped
    uint32_t restart_index;
    uint64_t copy_row_stride, copy_image_stride;
    uint64_t obj;       // frame interpolation: the game object the current draws belong to (CMD_OBJECT), 0 = none
    uint32_t obj_draw;  // draws of that object so far in this run of it
} st;

// frame interpolation (prototype, LARECOMP_INTERP=1): the hooks below; the rest at the end of this file
static int ip_on = -1;
static bool ip_replaying;     // running a recorded frame again for the in-between image
static uint64_t ip_stats_tagged, ip_stats_untagged;  // draws with / without a game object (CMD_OBJECT)
static uint64_t ip_draw_sig;  // the current draw's identity (draw_)
static bool ip_draw_obj;      // ... and whether it belongs to a game object (CMD_OBJECT): only those are blended
static bool ip_draw_2d;       // ... a 2D draw (no depth test): interface, menus
static bool ip_enabled(void);
bool vk_interp_enabled(void) { return ip_enabled(); }  // (cmdbuf.c: hooks the game's object draw functions)
static bool ip_enabled(void) {
    // The 60 fps mode (runtime/hle/fps60.c): the world at 60 natively; menus and the Pokédex at the game's 30 with the
    // in-between image; battles and captures at 30 without it (tried, left out: docs/HISTORICO.md 46).
    // LARECOMP_INTERP=1 (research): always on.
    if (ip_on == -1) ip_on = getenv("LARECOMP_INTERP") ? atoi(getenv("LARECOMP_INTERP")) : 0;
    extern bool fps60_menu(void);
    return ip_on > 0 || fps60_menu();
}
static const void *ip_cbuf(int stage, int k, uint64_t addr, uint32_t size);
static void ip_note_list(uint64_t handle);
static bool ip_skip_op(uint32_t op) {  // what the replay must not do again: it writes guest memory or signals the game
    switch (op) {
    case CMD_DISPATCH: case CMD_DISPATCH_INDIRECT: case CMD_COPY_BUFFER_TO_BUFFER: case CMD_COPY_TEXTURE_TO_BUFFER:
    case CMD_UPDATE_UNIFORM_BUFFER: case CMD_FENCE_SYNC: case CMD_WAIT_SYNC: case CMD_SIGNAL_EVENT: case CMD_WAIT_EVENT:
    case CMD_REPORT_COUNTER: case CMD_CLEAR_BUFFER:
    case CMD_COPY_BUFFER_TO_TEXTURE:  // (uploads: done when the frame ran; again, they overran scaled images)
        return true;
    default:
        return false;
    }
}

uint32_t nvn_window_origin(void);
uint32_t nvn_depth_mode(void);
bool renderer_origin_lower_left(void) { return nvn_window_origin() == 0; }

// (LARECOMP_VK_PROF) called once per presented frame: frames over 50 ms are logged with where their time went.
// present_ns: time in backend_present's acquire/blit/present.
void vk_prof_frame(uint64_t frame, uint64_t present_ns) {
    if (!prof) return;
    static uint64_t last_t, l_exec, l_wait, l_draws, l_subs;
    uint64_t now = SDL_GetTicksNS();
    if (last_t && now - last_t > 50000000) {
        uint64_t dt = now - last_t, ex = cum_exec - l_exec, wt = cum_wait - l_wait;
        LOG("[hitch] frame %llu: %.0f ms = backend exec %.1f (of which GPU waits %.1f) + present %.1f + other %.1f; %llu draws, "
            "%llu submits, compiles %d pending\n", (unsigned long long)frame, dt / 1e6, ex / 1e6, wt / 1e6, present_ns / 1e6,
            ((double)dt - ex - present_ns) / 1e6, (unsigned long long)(cum_draws - l_draws), (unsigned long long)(cum_submits - l_subs),
            atomic_load(&stat_compiles));
    }
    last_t = now; l_exec = cum_exec; l_wait = cum_wait; l_draws = cum_draws; l_subs = cum_submits;
}

static void save_pipeline_cache(void *a) {
    (void)a;
    uint64_t t0 = SDL_GetTicksNS();
    vk_pipeline_cache_save();
    LOG("[vk] pipeline cache saved (%.0f ms)\n", (SDL_GetTicksNS() - t0) / 1e6);
}

const char *renderer_stats(void) {
    // The pipeline cache was written only at a clean exit: a crash or a killed process lost every pipeline built in
    // the session, and the next one popped in again. Save it on a worker once a burst of new pipelines settles
    // (no new ones for ~6 s: entering an area builds them for a few seconds).
    static int quiet, last_new;
    int np = atomic_load(&new_pipes);
    if (np && np == last_new) {
        if (++quiet >= 3) { atomic_fetch_sub(&new_pipes, np); np = 0; quiet = 0; run_job(save_pipeline_cache, NULL); }
    } else {
        quiet = 0;
    }
    last_new = np;
    static int dumps;
    if (dumps++ < 3) {
        LOG("[vk] command counts:");
        for (int i = 0; i < CMD_COUNT; i++) if (op_counts[i]) LOG(" %d:%llu", i, (unsigned long long)op_counts[i]);
        LOG("\n");
    }
    snprintf(stats_text, sizeof stats_text,
             "%llu draws, %llu skipped, %llu passes, %llu dispatches, %llu flushes, wait %.0fms, compile %d (translate %.0fms, spirv %.0fms, pipelines %.0fms, draws waited %.0fms), fbcopy %llu, upload %.1fMB",
             (unsigned long long)stat_draws, (unsigned long long)stat_skipped, (unsigned long long)stat_passes,
             (unsigned long long)stat_dispatches, (unsigned long long)stat_flushes, (double)stat_wait_ns / 1e6,
             atomic_exchange(&stat_compiles, 0), atomic_exchange(&stat_translate_ns, 0) / 1e6, atomic_exchange(&stat_spirv_ns, 0) / 1e6, atomic_exchange(&stat_pipe_ns, 0) / 1e6,
             stat_shader_wait_ns / 1e6,
             (unsigned long long)stat_feedback_copies, (double)stat_upload_bytes / 1048576.0);
    {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | vertex ranges %llu copied, %llu in place", (unsigned long long)stat_vtx_copied, (unsigned long long)stat_vtx_direct);
        stat_vtx_copied = stat_vtx_direct = 0;
    }
    stat_feedback_copies = stat_upload_bytes = stat_shader_wait_ns = 0;
    if (g_exp_count[0] | g_exp_count[1] | g_exp_count[2] | g_exp_count[3]) {
        LOG("[exp] skin identity %llu, ubo direct %llu, ubo copied %llu, one-bone shaders %llu\n",
            (unsigned long long)g_exp_count[0], (unsigned long long)g_exp_count[1], (unsigned long long)g_exp_count[2],
            (unsigned long long)g_exp_count[3]);
        g_exp_count[0] = g_exp_count[1] = g_exp_count[2] = 0;  // ([3]: a running total)
    }
    if (g_skin_watch) {
        const uint32_t *w = (const uint32_t *)(uintptr_t)g_skin_watch;
        LOG("[skin] cb3 %llx now: row8 %08x %08x row9 %08x %08x\n", (unsigned long long)g_skin_watch, w[32], w[33], w[36], w[37]);
    }
    {   // NVN barriers since the last line: how many were applied, and the bit values seen (NVNbarrierBits:
        // 1 ORDER_PRIMITIVES, 2 ORDER_FRAGMENTS, 4 ORDER_FRAGMENTS_TILED, 8 ORDER_INDIRECT_DATA, 0x10 INVALIDATE_TEXTURE,
        // 0x20 INVALIDATE_SHADER, 0x40 INVALIDATE_TEXTURE_DESCRIPTOR, 0x80 INVALIDATE_ZCULL, 0x100 INVALIDATE_L2_CACHE)
        size_t n = strlen(stats_text);
        n += snprintf(stats_text + n, sizeof stats_text - n, " | nvn barriers %llu [", (unsigned long long)stat_nvn_barriers);
        for (int b = 0; b < 0x200 && n < sizeof stats_text - 32; b++)
            if (stat_barrier_bits[b]) n += snprintf(stats_text + n, sizeof stats_text - n, " %#x:%llu", b, (unsigned long long)stat_barrier_bits[b]);
        snprintf(stats_text + n, sizeof stats_text - n, " ]");
        stat_nvn_barriers = 0;
        memset(stat_barrier_bits, 0, sizeof stat_barrier_bits);
    }
    {   // pipeline recipes (prewarm): loaded, programs and pipelines prewarmed, pipelines a draw adopted, new recipes
        extern void recipes_stats(char *out, size_t n);
        size_t n = strlen(stats_text);
        recipes_stats(stats_text + n, sizeof stats_text - n);
    }
    if (stat_rb_deferred || stat_rb_sync) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | readbacks %llu deferred, %llu waited, %llu gmem waits, %llu superseded", (unsigned long long)stat_rb_deferred,
                 (unsigned long long)stat_rb_sync, (unsigned long long)stat_rb_gmem_waits, (unsigned long long)stat_rb_superseded);
        stat_rb_deferred = stat_rb_sync = stat_rb_gmem_waits = stat_rb_superseded = 0;
    }
    if (stat_copy_flushed || stat_copy_inline) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | buffer copies %llu flushed, %llu inline", (unsigned long long)stat_copy_flushed,
                 (unsigned long long)stat_copy_inline);
        stat_copy_flushed = stat_copy_inline = 0;
    }
    if (stat_pc_hits || stat_pc_copies || stat_pc_dyn) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | persistent copies %llu hits, %llu copied (%.1f MB), %llu dynamic, %llu scans %.1f ms, %llu dirty pages, %llu MB VRAM",
                 (unsigned long long)stat_pc_hits, (unsigned long long)stat_pc_copies, stat_pc_bytes / 1048576.0, (unsigned long long)stat_pc_dyn,
                 (unsigned long long)stat_pc_scans, stat_pc_scan_ns / 1e6, (unsigned long long)stat_pc_dirty, (unsigned long long)(pc_used >> 20));
        stat_pc_hits = stat_pc_copies = stat_pc_bytes = stat_pc_dyn = stat_pc_scans = stat_pc_scan_ns = stat_pc_dirty = 0;
    }
    if (atomic_load(&stat_gpl_links) || atomic_load(&stat_gpl_used)) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pipeline parts %llu built (%.0f ms), %llu links (%.1f ms), %llu draws linked, %llu failed",
                 (unsigned long long)atomic_exchange(&stat_gpl_parts, 0), atomic_exchange(&stat_gpl_part_ns, 0) / 1e6,
                 (unsigned long long)atomic_exchange(&stat_gpl_links, 0), atomic_exchange(&stat_gpl_link_ns, 0) / 1e6,
                 (unsigned long long)atomic_exchange(&stat_gpl_used, 0), (unsigned long long)atomic_exchange(&stat_gpl_failed, 0));
    }
    if (stat_pf_walks) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | prefetch %llu walks (%.0f ms), %llu programs, %llu pipelines",
                 (unsigned long long)stat_pf_walks, stat_pf_ns / 1e6, (unsigned long long)stat_pf_progs, (unsigned long long)stat_pf_pipes);
        stat_pf_walks = stat_pf_ns = stat_pf_progs = stat_pf_pipes = 0;
    }
    if (stat_gcopy_drops || stat_gcopy_kept) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | shared copies dropped %llu, kept %llu", (unsigned long long)stat_gcopy_drops,
                 (unsigned long long)stat_gcopy_kept);
        stat_gcopy_drops = stat_gcopy_kept = 0;
    }
    if (stat_holds) {  // draws that waited for their program / pipeline (wait_built)
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | holds %llu (%.0fms, max %.1fms, claimed %llu, timeouts %llu)",
                 (unsigned long long)stat_holds, stat_hold_ns / 1e6, stat_hold_max_ns / 1e6, (unsigned long long)stat_hold_claims,
                 (unsigned long long)stat_hold_timeouts);
        stat_holds = stat_hold_ns = stat_hold_max_ns = stat_hold_claims = stat_hold_timeouts = 0;
    }
    {   // draws with the NVN alpha test on (anything but ALWAYS): measured before implementing it (backlog ALPHATEST)
        uint64_t on = 0;
        for (int f = 0; f < 16; f++) if (f != 8) on += stat_alpha_func[f];
        if (on) {
            size_t n = strlen(stats_text);
            n += snprintf(stats_text + n, sizeof stats_text - n, " | alpha test draws %llu [", (unsigned long long)on);
            for (int f = 0; f < 16 && n < sizeof stats_text - 24; f++)
                if (f != 8 && stat_alpha_func[f]) n += snprintf(stats_text + n, sizeof stats_text - n, " %d:%llu", f, (unsigned long long)stat_alpha_func[f]);
            snprintf(stats_text + n, sizeof stats_text - n, " ]");
        }
        memset(stat_alpha_func, 0, sizeof stat_alpha_func);
    }
    if (getenv("LARECOMP_VK_WAITPROF")) {  // waits for the GPU by caller since the last line (ra: lasrecomp+offset)
        size_t n = strlen(stats_text);
        n += snprintf(stats_text + n, sizeof stats_text - n, " | waits by caller:");
        pthread_mutex_lock(&g_waitprof_lock);
        extern const char *plat_symbol(uint64_t pc, uint64_t *offset);
        for (int i = 0; i < WAITPROF_N && g_waitprof[i].ra && n < sizeof stats_text - 64; i++) {
            if (!g_waitprof[i].n) continue;
            uint64_t off = 0;
            const char *mod = plat_symbol((uint64_t)(uintptr_t)g_waitprof[i].ra, &off);
            extern char __ImageBase;  // (the exe: an offset from its base, for addr2line -e lasrecomp.exe 0x140000000+off)
            if (!mod) mod = "exe", off = (uint64_t)((char *)g_waitprof[i].ra - &__ImageBase);
            n += snprintf(stats_text + n, sizeof stats_text - n, " %s+%llx %.0fms/%llu", mod, (unsigned long long)off,
                          g_waitprof[i].ns / 1e6, (unsigned long long)g_waitprof[i].n);
            g_waitprof[i].ns = g_waitprof[i].n = 0;
        }
        pthread_mutex_unlock(&g_waitprof_lock);
    }
    {   // validation layer messages so far (LARECOMP_VK_VALIDATION=1, device.c)
        extern atomic_uint g_vkval_errors, g_vkval_warnings;
        unsigned e = atomic_load(&g_vkval_errors), w = atomic_load(&g_vkval_warnings);
        if (e || w) {
            size_t n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | vkval E %u W %u", e, w);
        }
    }
    if (stat_skipped) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " [skip: noprog %llu compiling %llu failed %llu noobj %llu mode %llu pipe-building %llu pipe-failed %llu]",
                 (unsigned long long)stat_skip_why[0], (unsigned long long)stat_skip_why[1],
                 (unsigned long long)stat_skip_why[2], (unsigned long long)stat_skip_why[3],
                 (unsigned long long)stat_skip_why[4], (unsigned long long)stat_skip_why[5],
                 (unsigned long long)stat_skip_why[6]);
        memset(stat_skip_why, 0, sizeof stat_skip_why);
    }
    if (prof) {
        prof_up = prof_up_ns;
        prof_up_ns = 0;
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | gpu %.0f cpu exec %.0f draw %.0f (pre %.0f key %.0f dyn %.0f sets %.0f tex %.0f (upload %.0f) vtx %.0f) submit %.0f ms",
                 prof_gpu / 1e6, prof_exec / 1e6, prof_draw / 1e6, prof_pre / 1e6, prof_key / 1e6, prof_dyn / 1e6, prof_sets / 1e6,
                 prof_tex / 1e6, prof_up / 1e6, prof_vtx / 1e6, prof_submit / 1e6);
        extern uint64_t prof_tex_checks, prof_tex_samples, prof_tex_full, prof_tex_full_bytes, prof_tex_dyn;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | tex checks %llu samples %llu full %llu (%.1f MB, dynamic %llu)",
                 (unsigned long long)prof_tex_checks, (unsigned long long)prof_tex_samples, (unsigned long long)prof_tex_full,
                 prof_tex_full_bytes / 1048576.0, (unsigned long long)prof_tex_dyn);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pass ends:");
        for (int l = 0; l < 4096; l++)
            if (prof_end_line[l] > 20) { n = strlen(stats_text); snprintf(stats_text + n, sizeof stats_text - n, " %d:%u", l, prof_end_line[l]); }
        memset(prof_end_line, 0, sizeof prof_end_line);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | views created %llu", (unsigned long long)prof_views_created);
        prof_views_created = 0;
        prof_tex_checks = prof_tex_samples = prof_tex_full = prof_tex_full_bytes = prof_tex_dyn = 0;
        prof_exec = prof_draw = prof_sets = prof_vtx = prof_submit = prof_pre = prof_key = prof_dyn = prof_tex = prof_up = prof_gpu = 0;
    }
    stat_draws = stat_skipped = stat_passes = stat_dispatches = stat_flushes = stat_wait_ns = 0;
    return stats_text;
}

static void st_defaults(void) {
    memset(&st, 0, sizeof st);
    for (int i = 0; i < 16; i++) {
        st.viewport[i][2] = 1280; st.viewport[i][3] = 720;
        st.scissor[i][2] = 16384; st.scissor[i][3] = 16384;
        st.depth_range[i][1] = 1.0f;
    }
    st.channel_mask = 0xFFFFFFFF;
    st.stencil_mask[0] = st.stencil_mask[1] = 0xFF;
    st.stencil_vmask[0] = st.stencil_vmask[1] = 0xFF;
    st.ds[1] = 1; st.ds[2] = 2;
    st.poly[1] = 1; st.poly[2] = 2;
    for (int i = 0; i < 8; i++) { st.blend[i][1] = 2; st.blend[i][2] = 1; st.blend[i][3] = 2; st.blend[i][4] = 1; st.blend[i][5] = 1; st.blend[i][6] = 1; }
}

void vk_exec_init(void) {
    if (getenv("LARECOMP_VK_SUBS")) {
        int n = atoi(getenv("LARECOMP_VK_SUBS"));
        SUBS = n < 1 ? 1 : n > SUBS_MAX ? SUBS_MAX : n;
    }
    LOG("[vk] submission ring: %d\n", SUBS);
    prof = getenv("LARECOMP_VK_PROF") != NULL;
    if (prof) {
        prof_up_ns = 0;
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = SUBS_MAX * 2};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &prof_qp) != VK_SUCCESS) prof_qp = VK_NULL_HANDLE;
    }
    st_defaults();
}

void renderer_texture_scale(NvnTexture *t, float *sx, float *sy);
bool renderer_texture_presented(NvnTexture *t);
bool renderer_texture_first_write(NvnTexture *t);
bool renderer_texture_level_fresh(NvnTexture *t);
bool renderer_texture_frame_fresh(NvnTexture *t);
void renderer_texture_mark_presented(NvnTexture *t);
uint64_t renderer_present_count(void);
void renderer_save_texture_tex(TexBackend *b, const char *path);
static float pass_sx = 1, pass_sy = 1;
static float vp_sx = 1, vp_sy = 1, vp_ox = 0, vp_oy = 0;
static float tex_sc[2][64][2];
static uint32_t tex_wh[2][64][2];

// ================================================================== SSAO state (ssao_run, below draw_)
// Ported from Let's Go Recomp (its exec.c ssao_*), placed for Sword's frame: shadow cascades, a depth prepass, the
// opaque scene into an HDR target with its depth (and a second, multi-target opaque pass on the same two), then the
// transparent draws on another depth, the bloom chain, the tone mapping and the interface. The scene is the frame's
// first pass drawing into a scalable float colour target with a depth; its opaque draws (blending off, depth test and
// write on) are counted and give the projection (vertex constants). The occlusion is computed and multiplied into the
// HDR colour at the first draw after them that is not the scene's: a pass on other targets, or a draw in the scene
// without the depth test - after all the opaque geometry, before anything blended or the bloom reads it.
static struct {
    bool qual, pending, busy;  // the pass now is the scene's; a non-scene pass after the opaque (run at its first draw)
    NvnTexture *dtex, *ctex;   // this frame's scene depth and colour target
    VkImageView cview;
    VkFormat cfmt;
    uint32_t cw, ch, opaque;
    uint64_t frame, done_frame, proj_frame, scan_frame;
    float proj[4];             // P00, P11, A, B (GL style: z_ndc = (A z + B) / -z)
    float vp[4];               // the scene draws' viewport in the target's pixels (x, y, w, h): Sword's dynamic resolution
                               // draws the scene into part of its targets (80% of each side seen)
    int pbind;                 // where the projection was found: vertex cbuf, byte offset
    uint32_t poff;
} ao = {.pbind = -1};

// TAA / DLAA (taa_run): the scene's passes are jittered by a sub-pixel offset each frame (through their viewport: the
// whole image moves, no shader variant), and the HDR scene, complete (opaque, SSAO, blended), is resolved against its
// history along the camera's motion vectors before the first pass on another colour target (the bloom).
static struct {
    double vp[16], vp_prev[16];   // the camera's view-projection (row-major), this frame's and the previous frame's
    uint64_t cam_frame;           // the frame vp is of
    bool prev_ok, cam_warned;     // vp_prev is the frame just before
    uint64_t scene_d, scene_c;    // the scene's depth and colour (addresses, kept from frame to frame): jittered
    bool jit_pass, pending;       // this pass's draws are jittered; resolve before this pass's first draw
    float jx, jy;                 // this frame's jitter in pixels (the content moves right, down)
    uint64_t jit_frame, jit_used; // the frame jx, jy are of; the last frame a draw took them
    uint64_t done_frame, ok_frame;// resolve tried this frame; the last frame resolved (no resolve: no jitter either)
    uint64_t hist_frame;          // the history's frame, size, viewport and kind (TAA / DLAA): kept only if all match
    uint32_t hist_w, hist_h;
    float hist_vp[4];
    int hist_mode;
} taa;
static bool taa_wanted(void) { return g_settings.antialias == AA_TAA || g_settings.antialias == AA_DLAA; }
static float halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    for (; i; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}

// ================================================================== render passes
static VkFormat pass_color[8], pass_depth, pass_stencil;
static uint32_t pass_ncolor, pass_samples = 1, pass_w, pass_h;
static int debug_pass_no;

static void vk_end_pass_at(int line);
void vk_end_pass(void) { vk_end_pass_at(0); }
#define vk_end_pass() vk_end_pass_at(__LINE__)
static void vk_end_pass_at(int line) {
    if (!in_pass) return;
    if (prof && line < 4096) prof_end_line[line]++;
    vkCmdEndRendering(vk_cmd());
    in_pass = false;
    if (g_debug_frame && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->backend) {
        char path[256];
        static uint64_t debug_pass_frame = ~0ull;
        static int multi = -1;
        if (multi < 0) multi = getenv("LARECOMP_DEBUG_FRAMES") != NULL;
        if (multi && renderer_present_count() != debug_pass_frame) { debug_pass_frame = renderer_present_count(); debug_pass_no = 0; }
        if (multi) snprintf(path, sizeof path, "build/pass_%llu_%03d.png", (unsigned long long)debug_pass_frame, debug_pass_no++);
        else snprintf(path, sizeof path, "build/pass_%03d.png", debug_pass_no++);
        renderer_save_texture_tex(st.rt.color[0]->backend, path);
    }
}

static uint32_t level_extent(uint32_t v, uint32_t lv) { v >>= lv; return v ? v : 1; }


// begin_pass, TAA on: whether this pass's draws are jittered, whether the scene is resolved before it
static void taa_pass_check(bool has_d) {
    uint64_t fr = vk_frame_counter();
    if (taa.jit_frame != fr) {  // (Halton 2, 3 over 8 frames)
        taa.jit_frame = fr;
        uint32_t i = (uint32_t)(fr % 8) + 1;
        taa.jx = halton(i, 2) - 0.5f, taa.jy = halton(i, 3) - 0.5f;
    }
    uint64_t da = has_d && st.rt.depth ? st.rt.depth->addr : 0;
    uint64_t ca = pass_ncolor && st.rt.color[0] ? st.rt.color[0]->addr : 0;
    // the scene's passes (its depth: the depth-only pass first; its colour with a depth: the blended draws on their own
    // depth), until the resolve - and only while it keeps resolving (else the image would shake)
    if (taa.done_frame != fr && taa.ok_frame + 2 >= fr && taa.ok_frame && pass_samples == 1 &&
        ((da && da == taa.scene_d) || (da && ca && ca == taa.scene_c)))
        taa.jit_pass = true, taa.jit_used = fr;
    // after the scene's effects ran (the SSAO's place: ao.done_frame), the first pass on another colour target
    if (ao.dtex && ao.ctex && ao.done_frame == fr && taa.done_frame != fr && ca && ca != ao.ctex->addr) taa.pending = true;
}

static void begin_pass(void) {
    if (in_pass && !st.rt_dirty) return;
    vk_end_pass();
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each pass its own submission)
    st.rt_dirty = false;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);  // passes, copies and dispatches see each other's results
    {
        NvnTexture *t0n = st.rt.ncolor ? st.rt.color[0] : NULL;
        dl_note("pass %d color %ux%u fmt %#x depth %p", st.rt.ncolor, t0n ? t0n->width : 0, t0n ? t0n->height : 0, t0n ? t0n->format : 0,
                (void *)st.rt.depth);
    }
    if (g_gpuprof_frame) {
        NvnTexture *t0 = st.rt.ncolor ? st.rt.color[0] : NULL;
        gp_mark(cb, "pass: %d color %ux%u fmt %#x, depth %s", st.rt.ncolor, t0 ? t0->width : 0, t0 ? t0->height : 0,
                t0 ? t0->format : 0, st.rt.depth ? "yes" : "no");
    }
    {   // LARECOMP_PASS_LOG=<frame>[,<frame>]: every pass of those presented frames, its targets and the draws of the
        // one before - the map a scene effect (SSAO) is placed by
        static const char *pl = (const char *)1;
        static uint64_t last_draws;
        if (pl == (const char *)1) pl = getenv("LARECOMP_PASS_LOG");
        if (pl) {
            char want[32];
            snprintf(want, sizeof want, "%llu", (unsigned long long)vk_frame_counter());
            const char *hit = strstr(pl, want);
            if (hit && (hit == pl || hit[-1] == ',') && (hit[strlen(want)] == 0 || hit[strlen(want)] == ',')) {
                char line[700];
                int n = snprintf(line, sizeof line, "[pass] f%llu prev %llu draws |", (unsigned long long)vk_frame_counter(),
                                 (unsigned long long)(stat_draws - last_draws));
                for (int i = 0; i < st.rt.ncolor && n < (int)sizeof line - 80; i++) {
                    NvnTexture *t = st.rt.color[i];
                    if (t) n += snprintf(line + n, sizeof line - n, " c%d %ux%ux%u fmt %#x @%llx", i, t->width, t->height, t->depth,
                                         t->format, (unsigned long long)t->addr);
                }
                NvnTexture *d = st.rt.depth;
                if (d) snprintf(line + n, sizeof line - n, " | d %ux%ux%u fmt %#x @%llx", d->width, d->height, d->depth, d->format,
                                (unsigned long long)d->addr);
                LOG("%s\n", line);
            }
            last_draws = stat_draws;
        }
    }
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    uint32_t w = 0, minw = UINT32_MAX, minh = UINT32_MAX;  // w: first attachment seen
    if (st.rt.depth && renderer_format_is_depth(st.rt.depth->format)) {  // (the shadow map at its own scale: textures.c)
        extern void vk_tex_shadow_pair(NvnTexture *depth, NvnTexture *color);
        for (int i = 0; i < st.rt.ncolor; i++) vk_tex_shadow_pair(st.rt.depth, st.rt.color[i]);
    }
    pass_ncolor = (uint32_t)st.rt.ncolor;
    pass_samples = 1;
    for (int i = 0; i < st.rt.ncolor; i++) {
        NvnTexture *t = st.rt.color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        pass_color[i] = VK_FORMAT_UNDEFINED;
        if (!t) continue;
        watch_tex(t, "render target", NULL);
        renderer_mark_gpu_owned(t);
        VkImageView v = vk_view(t, &st.rt.cview[i], st.rt.has_cview[i], true, TEXTYPE_2D);
        TexBackend *b = t->backend;
        if (!v) continue;
        ca[i].imageView = v;
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        bool fresh = (int)renderer_texture_level_fresh(t) | (int)renderer_texture_frame_fresh(t);  // both: each marks its state
        if (renderer_texture_first_write(t) || fresh) {
            ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            ca[i].clearValue.color.float32[3] = fresh ? 0.0f : 1.0f;
        }
        pass_color[i] = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4) && vk_format(t->format) == b->format
                      ? vk_format(st.rt.cview[i].format) : b->format;
        uint32_t lv = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 1) ? st.rt.cview[i].base_level : 0;
        uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
        if (!w) { renderer_texture_scale(t, &pass_sx, &pass_sy); w = ew; }
        if (ew < minw) minw = ew;
        if (eh < minh) minh = eh;
        pass_samples = b->samples;
    }
    // Bound color slots without a texture / view went in as VK_NULL_HANDLE attachments (legal), but Intel's Windows
    // driver for UHD 630 crashes in vkCmdBeginRendering on them (LAS Recomp f4cb039, its issues #8 / #46). Empty slots
    // at the end are left out (pipelines and clears follow pass_ncolor); one in the middle is logged once.
    while (pass_ncolor && !ca[pass_ncolor - 1].imageView) pass_ncolor--;
    static bool logged_hole;
    for (uint32_t i = 0; i < pass_ncolor && !logged_hole; i++)
        if (!ca[i].imageView) {
            logged_hole = true;
            LOG("[vk] pass with an empty color slot %u of %u (texture %s)\n", i, pass_ncolor, st.rt.color[i] ? "without a view" : "none");
        }
    pass_depth = pass_stencil = VK_FORMAT_UNDEFINED;
    bool has_d = false, has_s = false;
    if (st.rt.depth) {
        renderer_mark_gpu_owned(st.rt.depth);
        VkImageView v = vk_view(st.rt.depth, &st.rt.dview, st.rt.has_dview, true, TEXTYPE_2D);
        TexBackend *b = st.rt.depth->backend;
        if (v && renderer_format_is_depth(st.rt.depth->format)) {
            VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v,
                                           .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                           .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            if (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da = a; has_d = true; pass_depth = b->format; }
            if (b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa = a; has_s = true; pass_stencil = b->format; }
            uint32_t lv = st.rt.has_dview && (st.rt.dview.setmask & 1) ? st.rt.dview.base_level : 0;
            uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
            if (!w) { renderer_texture_scale(st.rt.depth, &pass_sx, &pass_sy); w = ew; }
            if (ew < minw) minw = ew;
            if (eh < minh) minh = eh;
            if (!pass_ncolor) pass_samples = b->samples;
        }
    }
    if (!w) { pass_sx = pass_sy = 1; minw = minh = 16; }
    pass_w = minw; pass_h = minh;
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {minw, minh}}, .layerCount = 1,
                          .colorAttachmentCount = pass_ncolor, .pColorAttachments = ca,
                          .pDepthAttachment = has_d ? &da : NULL, .pStencilAttachment = has_s ? &sa : NULL};
    vkCmdBeginRendering(cb, &ri);
    in_pass = true;
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    stat_passes++;
    ao.qual = ao.pending = false;
    taa.pending = taa.jit_pass = false;
    if ((g_settings.ssao || taa_wanted()) && !ao.busy) {
        uint64_t fr = vk_frame_counter();
        if (ao.frame != fr) { ao.frame = fr; ao.opaque = 0; ao.dtex = ao.ctex = NULL; }
        TexBackend *c0 = pass_ncolor && ca[0].imageView && st.rt.color[0] ? st.rt.color[0]->backend : NULL;
        VkFormat f = pass_color[0];
        bool scene = c0 && c0->scalable && has_d && pass_samples == 1 && st.rt.color[0]->width >= 640 &&
                     (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT || f == VK_FORMAT_R32G32B32A32_SFLOAT);
        if (scene && ao.opaque < 16 && (ao.dtex != st.rt.depth || ao.ctex != st.rt.color[0])) {
            ao.dtex = st.rt.depth; ao.ctex = st.rt.color[0]; ao.opaque = 0;  // (the first, or a better candidate)
        }
        if (scene && ao.dtex == st.rt.depth && ao.ctex == st.rt.color[0]) {
            ao.qual = true;
            ao.cview = ca[0].imageView;
            ao.cfmt = f;
            ao.cw = minw; ao.ch = minh;
        } else if (ao.dtex && ao.opaque >= 16 && ao.done_frame != fr) {
            ao.pending = true;
        }
        if (ao.dtex && ao.ctex) taa.scene_d = ao.dtex->addr, taa.scene_c = ao.ctex->addr;
    }
    if (taa_wanted() && !ao.busy) taa_pass_check(has_d);
}

static void pass_size(int *w, int *h) {
    *w = *h = 0;
    for (int i = 0; i < st.rt.ncolor && !*w; i++)
        if (st.rt.color[i]) { *w = (int)st.rt.color[i]->width; *h = (int)st.rt.color[i]->height; }
    if (!*w && st.rt.depth) { *w = (int)st.rt.depth->width; *h = (int)st.rt.depth->height; }
    *w = (int)lroundf(*w * pass_sx);
    *h = (int)lroundf(*h * pass_sy);
}

static VkRect2D scissor_rect(void) {
    int w, h;
    pass_size(&w, &h);
    if (!w) return (VkRect2D){{0, 0}, {pass_w, pass_h}};
    int x = st.scissor[0][0], y = st.scissor[0][1], sw = st.scissor[0][2], sh = st.scissor[0][3];
    if (vp_sx != 1 || vp_sy != 1 || vp_ox != 0 || vp_oy != 0) {
        double x0 = floor(x * (double)vp_sx + vp_ox), y0 = floor(y * (double)vp_sy + vp_oy);
        double x1 = ceil(((double)x + sw) * vp_sx + vp_ox), y1 = ceil(((double)y + sh) * vp_sy + vp_oy);
        if (sw >= 16384) x1 = 1e9;
        if (sh >= 16384) y1 = 1e9;
        if (x1 > 1e9) x1 = 1e9;
        if (y1 > 1e9) y1 = 1e9;
        x = (int)x0; y = (int)y0; sw = (int)(x1 - x0); sh = (int)(y1 - y0);
    }
    if (x < 0) { sw += x; x = 0; }
    if (y < 0) { sh += y; y = 0; }
    if (w > (int)pass_w) w = (int)pass_w;
    if (h > (int)pass_h) h = (int)pass_h;
    if (x >= w || y >= h) { x = y = 0; sw = sh = 0; }
    if (x + sw > w) sw = w - x;
    if (y + sh > h) sh = h - y;
    if (sw < 0) sw = 0;
    if (sh < 0) sh = 0;
    return (VkRect2D){{x, y}, {(uint32_t)sw, (uint32_t)sh}};
}

static void apply_scissor(void) {
    VkRect2D r = scissor_rect();
    vkCmdSetScissor(vk_cmd(), 0, 1, &r);
}

// Metal convention (y up in NDC, origin top-left) through a negative-height viewport. neg_y: the game's NEGATIVE_Y
// viewport swizzle (y = -y after the vertex shader), one more flip.
static void set_viewport(float x, float y, float w, float h, float zn, float zf, bool neg_y) {
    if (w <= 0) w = 1;
    if (h == 0) h = 1;
    VkViewport vp = neg_y ? (VkViewport){x, y, w, h, fminf(fmaxf(zn, 0), 1), fminf(fmaxf(zf, 0), 1)}
                          : (VkViewport){x, y + h, w, -h, fminf(fmaxf(zn, 0), 1), fminf(fmaxf(zf, 0), 1)};
    if (ao.qual) {  // (the SSAO rebuilds positions within the scene's viewport)
        ao.vp[0] = vp.x, ao.vp[1] = vp.height < 0 ? vp.y + vp.height : vp.y, ao.vp[2] = vp.width, ao.vp[3] = fabsf(vp.height);
    }
    if (taa.jit_pass) vp.x += taa.jx, vp.y += taa.jy;  // (TAA: the content moves by the jitter, in this pass's pixels)
    vkCmdSetViewport(vk_cmd(), 0, 1, &vp);
}

// ================================================================== clears
static VkPipelineLayout pc_layout(void) {  // push constants only (clears)
    static VkPipelineLayout l;
    if (!l) {
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    }
    return l;
}

static const VkDynamicState DYN[] = {
    VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
#define NDYN (sizeof DYN / sizeof DYN[0])

// Fixed-function state that isn't part of a pipeline is set explicitly on every bind (it persists across pipelines).
static void dyn_defaults(VkCommandBuffer cb) {
    vkCmdSetDepthBias(cb, 0, 0, 0);
    vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    float bc[4] = {0, 0, 0, 0};
    vkCmdSetBlendConstants(cb, bc);
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilTestEnable(cb, VK_FALSE);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);
    vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
}

typedef struct { uint64_t key; VkPipeline pipe; } PipeEnt;
#define CLEAR_TAB 256
static PipeEnt clear_tab[CLEAR_TAB];

static VkPipeline clear_pipeline(int index, int kind, uint32_t mask, bool depth, bool stencil) {
    uint64_t key = 0xcbf29ce484222325ull;
#define MIX(v) do { uint64_t v_ = (uint64_t)(v); for (int b_ = 0; b_ < 8; b_++) { key ^= (v_ >> (b_ * 8)) & 0xFF; key *= 0x100000001B3ull; } } while (0)
    MIX(index); MIX(kind); MIX(mask); MIX(depth); MIX(stencil); MIX(pass_ncolor); MIX(pass_depth); MIX(pass_stencil); MIX(pass_samples);
    for (uint32_t i = 0; i < pass_ncolor; i++) MIX(pass_color[i]);
    uint32_t slot = (uint32_t)key & (CLEAR_TAB - 1);
    for (int p = 0; p < CLEAR_TAB; p++, slot = (slot + 1) & (CLEAR_TAB - 1)) {
        if (clear_tab[slot].key == key) return clear_tab[slot].pipe;
        if (!clear_tab[slot].key) break;
    }
    char fsname[32];
    if (index >= 0) snprintf(fsname, sizeof fsname, "clear.frag.%c%d", kind == 1 ? 'i' : kind == 2 ? 'u' : 'f', index);
    else snprintf(fsname, sizeof fsname, "clear.frag.none");
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("clear.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fsname), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)pass_samples};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < pass_ncolor; i++)
        if ((int)i == index) cba[i].colorWriteMask = mask & 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = pass_ncolor, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo rinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = pass_ncolor,
                                           .pColorAttachmentFormats = pass_color, .depthAttachmentFormat = pass_depth,
                                           .stencilAttachmentFormat = pass_stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rinfo, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = pc_layout()};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p) != VK_SUCCESS) LOG("[vk] clear pipeline failed\n");
    clear_tab[slot].key = key;
    clear_tab[slot].pipe = p;
    return p;
}

static void clear(int index, int kind, const void *color, uint32_t mask, bool depth, float dval, bool stencil, uint32_t sval,
                  uint32_t smask) {
    begin_pass();
    VkCommandBuffer cb = vk_cmd();
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    VkRect2D rect = scissor_rect();
    if (!rect.extent.width || !rect.extent.height) return;
    if (index >= 0 && (index >= (int)pass_ncolor || pass_color[index] == VK_FORMAT_UNDEFINED)) return;
    bool simple = (index < 0 || (mask & 0xF) == 0xF) && (!stencil || (smask & 0xFF) == 0xFF);
    if (simple) {
        VkClearAttachment a[2];
        uint32_t n = 0;
        if (index >= 0) {
            a[n] = (VkClearAttachment){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)index};
            memcpy(a[n].clearValue.color.float32, color, 16);
            n++;
        }
        VkImageAspectFlags dsa = (depth && pass_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (stencil && pass_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        if (dsa) a[n++] = (VkClearAttachment){.aspectMask = dsa, .clearValue.depthStencil = {dval, sval}};
        VkClearRect cr = {rect, 0, 1};
        if (n) vkCmdClearAttachments(cb, n, a, 1, &cr);
        return;
    }
    VkPipeline p = clear_pipeline(index, kind, mask, depth, stencil);
    if (!p) return;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    dyn_defaults(cb);
    if (depth) { vkCmdSetDepthTestEnable(cb, VK_TRUE); vkCmdSetDepthWriteEnable(cb, VK_TRUE); }
    if (stencil) {
        vkCmdSetStencilTestEnable(cb, VK_TRUE);
        vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                          VK_COMPARE_OP_ALWAYS);
        vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, smask);
        vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, sval);
    }
    int w, h;
    pass_size(&w, &h);
    VkViewport vp = {0, 0, (float)(w ? w : (int)pass_w), (float)(h ? h : (int)pass_h), 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &rect);
    float pc[8] = {0};
    if (color) memcpy(pc, color, 16);
    pc[4] = dval;
    vkCmdPushConstants(cb, pc_layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
}

// ================================================================== programs
#define MAX_BIND 96
typedef struct {
    ShaderInfo *vi, *fi;
    uint64_t code[2], cmax[2];  // translation inputs (copied: translated on the translator thread)
    ShaderKey vkey, fkey;
    VkShaderModule vs, fs;
    VkBindingInfo binds[MAX_BIND];
    int nbinds;
    VkDescriptorSetLayout set_layout[2];
    VkPipelineLayout layout;
    VkPipelineLayout stage_layout[2];  // with pipeline parts (g_vk.gpl): {vertex set, -} and {-, fragment set}, below
    bool vs_uses_vid;
    uint64_t chash[2];  // the stages' content hashes (NvnShaderStage.content_hash; 0 unknown): pipeline recipes
    _Atomic int state;  // 0 building, 1 ready, 2 failed
} Prog;

// A layout of one stage's set alone (pipeline parts): the other set VK_NULL_HANDLE, independent sets. Each set's
// layout comes from that stage's own reflection, so a stage's part is the same in every program that uses it, and one
// built ahead of the game (prep.c) is a pipeline cache hit at the draw.
VkPipelineLayout vk_stage_layout(VkDescriptorSetLayout sl, int set) {
    VkDescriptorSetLayout sets[2] = {set ? VK_NULL_HANDLE : sl, set ? sl : VK_NULL_HANDLE};
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT, .setLayoutCount = 2, .pSetLayouts = sets};
    VkPipelineLayout l;
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    return l;
}

static VkPipelineLayout make_layout(Prog *p, int nsets, VkShaderStageFlags s0, VkShaderStageFlags s1) {
    p->set_layout[0] = vk_set_layout(p->binds, p->nbinds, 0, s0);
    if (nsets > 1) p->set_layout[1] = vk_set_layout(p->binds, p->nbinds, 1, s1);
    bool parts = g_vk.gpl && nsets == 2;
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = (uint32_t)nsets,
                                     .pSetLayouts = p->set_layout,
                                     .flags = parts ? VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT : 0};
    VkPipelineLayout l;
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    if (parts) for (int i = 0; i < 2; i++) p->stage_layout[i] = vk_stage_layout(p->set_layout[i], i);
    return l;
}

static void save_glsl(const char *src, uint64_t addr, const char *suffix) {
    if (!getenv("LARECOMP_SAVE_MSL")) return;
    mkdir("build/glsl", 0755);
    char path[256];
    snprintf(path, sizeof path, "build/glsl/%llx_%s.glsl", (unsigned long long)addr, suffix);
    FILE *f = fopen(path, "w");
    if (f) { fputs(src, f); fclose(f); }
}

static void compile_program(void *arg) {
    Prog *p = arg;
    size_t nv = 0, nf = 0;
    uint64_t t0 = SDL_GetTicksNS();
    uint32_t *sv = vk_glsl_to_spirv(p->vi->msl, STAGE_VERTEX, "vs", &nv);
    uint32_t *sf = sv ? vk_glsl_to_spirv(p->fi->msl, STAGE_FRAGMENT, "fs", &nf) : NULL;
    if (sv && getenv("LARECOMP_RENDERDOC_AT")) {  // which guest shader each SPIR-V is: tools/rdc_skin.py (FNV-1a 64)
        static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < nv * 4; i++) { h ^= ((const uint8_t *)sv)[i]; h *= 0x100000001B3ull; }
        pthread_mutex_lock(&lock);
        FILE *f = fopen("build/rdc-shaders.txt", "a");
        if (f) { fprintf(f, "%016llx vs %llx\n", (unsigned long long)h, (unsigned long long)p->code[0]); fclose(f); }
        pthread_mutex_unlock(&lock);
    }
    if (!sv || !sf) { free(sv); free(sf); atomic_store(&p->state, 2); return; }
    p->nbinds = vk_reflect(sv, nv, p->binds, MAX_BIND);
    p->nbinds += vk_reflect(sf, nf, p->binds + p->nbinds, MAX_BIND - p->nbinds);
    p->vs = vk_module(sv, nv);
    p->fs = vk_module(sf, nf);
    free(sv);
    free(sf);
    p->layout = make_layout(p, 2, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
    atomic_fetch_add(&stat_spirv_ns, SDL_GetTicksNS() - t0);
    atomic_fetch_add(&stat_compiles, 1);
    atomic_store(&p->state, p->vs && p->fs ? 1 : 2);
}

static void translate_program(void *arg) {
    Prog *p = arg;
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    uint64_t t0 = SDL_GetTicksNS();
    p->vi = maxwell_translate((const uint8_t *)(uintptr_t)p->code[0], p->cmax[0], STAGE_VERTEX, &p->vkey);
    // Experiment (research): LARECOMP_EXP_VS_EARLY=<vertex program address, hex>: that program's vertex shader returns at
    // the start of main (every output at its default), to tell the cost of running the shader from the rest of its draw.
    // (LARECOMP_EXP_VS_EARLY=<address>:<pc, hex>: it returns just before the instruction at that pc instead)
    if (p->vi->msl && getenv("LARECOMP_EXP_VS_EARLY") && p->code[0] == strtoull(getenv("LARECOMP_EXP_VS_EARLY"), NULL, 16)) {
        const char *m = strstr(p->vi->msl, "void main() {"), *r = m ? strstr(m, "  uint r") : NULL;
        const char *colon = strchr(getenv("LARECOMP_EXP_VS_EARLY"), ':');
        if (m && colon) {
            char pat[32];
            snprintf(pat, sizeof pat, "// pc %s:", colon + 1);
            const char *at = strstr(m, pat);
            while (at && at > m && at[-1] != '\n') at--;  // (the start of that line)
            r = at ? at : NULL;
        }
        if (r) {
            size_t len = strlen(p->vi->msl), pr = (size_t)(r - p->vi->msl);
            char *s = malloc(len + 16);
            memcpy(s, p->vi->msl, pr);
            memcpy(s + pr, "  return;\n", 10);
            memcpy(s + pr + 10, p->vi->msl + pr, len - pr + 1);
            p->vi->msl = s;
            LOG("[exp] vertex program %llx returns at the start of main\n", (unsigned long long)p->code[0]);
        }
    }
    // Experiment (research): LARECOMP_GLSL_OVERRIDE_DIR=<dir>: <dir>/<code address, hex>_vs.glsl replaces the generated GLSL
    // of that vertex program (a hand-edited version of what LARECOMP_SAVE_MSL=1 wrote in build/glsl/).
    if (p->vi->msl && getenv("LARECOMP_GLSL_OVERRIDE_DIR")) {
        char path[512];
        snprintf(path, sizeof path, "%s/%llx_vs.glsl", getenv("LARECOMP_GLSL_OVERRIDE_DIR"), (unsigned long long)p->code[0]);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *s = malloc((size_t)n + 1);
            if (s && fread(s, 1, (size_t)n, f) == (size_t)n) { s[n] = 0; p->vi->msl = s; LOG("[exp] GLSL override: %s\n", path); }
            fclose(f);
        }
    }
    // Experiment X4 (docs/letsgo/PLANO-3D.md), LARECOMP_VK_SKIN_ONEBONE=1: skinned vertex shaders see 100% weight on the
    // first bone (a12 = bone weights) - rigid, but whole, if the first bone's matrix reaches the GPU correctly.
    // Experiment X6, LARECOMP_VK_SKIN_IDX_X=1: every bone slot uses the first bone index (a11.x), weights unchanged.
    if (p->vi->msl && getenv("LARECOMP_VK_SKIN_IDX_X") && strstr(p->vi->msl, "in uvec4 a11;")) {
        const char *d = strstr(p->vi->msl, "in uvec4 a11;"), *m = strstr(p->vi->msl, "void main() {");
        if (d && m && d < m) {
            static const char def[] = "#define a11 uvec4(a11_in.x, a11_in.x, a11_in.x, a11_in.x)\n";
            size_t len = strlen(p->vi->msl), pd = (size_t)(d - p->vi->msl) + strlen("in uvec4 a11"), pm = (size_t)(m - p->vi->msl);
            char *s = malloc(len + 16 + sizeof def), *o = s;
            memcpy(o, p->vi->msl, pd); o += pd;
            memcpy(o, "_in", 3); o += 3;                       // in uvec4 a11_in;
            memcpy(o, p->vi->msl + pd, pm - pd); o += pm - pd;
            memcpy(o, def, sizeof def - 1); o += sizeof def - 1;
            memcpy(o, p->vi->msl + pm, len - pm + 1);
            p->vi->msl = s;
            g_exp_count[3]++;
        }
    }
    // Experiments X8+, LARECOMP_VK_SKIN_W="<GLSL vec4 expression of a12_in>": the bone weights a skinned vertex shader
    // sees, rewritten (a12_in = the weights as the GPU fetched them).
    if (p->vi->msl && getenv("LARECOMP_VK_SKIN_W") && strstr(p->vi->msl, "in vec4 a12;")) {
        const char *d = strstr(p->vi->msl, "in vec4 a12;"), *m = strstr(p->vi->msl, "void main() {");
        if (d && m && d < m) {
            char def[512];
            snprintf(def, sizeof def, "#define a12 (%s)\n", getenv("LARECOMP_VK_SKIN_W"));
            size_t dl = strlen(def), len = strlen(p->vi->msl), pd = (size_t)(d - p->vi->msl) + strlen("in vec4 a12"),
                   pm = (size_t)(m - p->vi->msl);
            char *s = malloc(len + 16 + dl), *o = s;
            memcpy(o, p->vi->msl, pd); o += pd;
            memcpy(o, "_in", 3); o += 3;                       // in vec4 a12_in;
            memcpy(o, p->vi->msl + pd, pm - pd); o += pm - pd;
            memcpy(o, def, dl); o += dl;
            memcpy(o, p->vi->msl + pm, len - pm + 1);
            p->vi->msl = s;
            g_exp_count[3]++;
        }
    }
    // (only skinned shaders - the ones with bone indices too: a12 is something else elsewhere.)
    if (p->vi->msl && getenv("LARECOMP_VK_SKIN_ONEBONE") && strstr(p->vi->msl, "in vec4 a12;") &&
        strstr(p->vi->msl, "in uvec4 a11;")) {
        const char *m = strstr(p->vi->msl, "void main() {");
        if (m) {
            static const char def[] = "#define a12 vec4(1.0, 0.0, 0.0, 0.0)\n";
            size_t pre = (size_t)(m - p->vi->msl), len = strlen(p->vi->msl);
            char *s = malloc(len + sizeof def);
            memcpy(s, p->vi->msl, pre);
            memcpy(s + pre, def, sizeof def - 1);
            memcpy(s + pre + sizeof def - 1, m, len - pre + 1);
            p->vi->msl = s;  // (the translator's string is kept: it lives as long as the program)
            g_exp_count[3]++;
        }
    }
    if (p->vi->msl && !p->code[1]) {  // vertex-only program (see get_program): an empty fragment shader
        p->fi = calloc(1, sizeof *p->fi);
        p->fi->stage = STAGE_FRAGMENT;
        p->fi->msl = strdup("#version 460\nvoid main() {}\n");
    } else if (p->vi->msl) {
        p->fkey.vs_out_mask = p->vi->out_mask;
        p->fi = maxwell_translate((const uint8_t *)(uintptr_t)p->code[1], p->cmax[1], STAGE_FRAGMENT, &p->fkey);
    }
    atomic_fetch_add(&stat_translate_ns, SDL_GetTicksNS() - t0);
    if (!p->vi->msl) {
        LOG("[shader] VS translate failed: %s (code %llx)\n", p->vi->error, (unsigned long long)p->code[0]);
        atomic_store(&p->state, 2);
        return;
    }
    if (!p->fi->msl) {
        LOG("[shader] FS translate failed: %s\n", p->fi->error);
        atomic_store(&p->state, 2);
        return;
    }
    p->vs_uses_vid = strstr(p->vi->msl, "gl_VertexIndex") != NULL;
    save_glsl(p->vi->msl, p->code[0], "vs");
    save_glsl(p->fi->msl, p->code[1], "fs");
    if (getenv("LARECOMP_PREP_DEBUG")) {  // (prep.c logs the same line for what it built: the keys must match)
        for (int i = 0; i < 2; i++) {
            const char *m = i ? p->fi->msl : p->vi->msl;
            uint64_t h = 0xcbf29ce484222325ull;
            for (const char *q = m; q && *q; q++) h = (h ^ (uint8_t)*q) * 0x100000001B3ull;
            LOG("[prepdbg] draw %s %016llx glsl %016llx\n", i ? "fs" : "vs", (unsigned long long)p->chash[i], (unsigned long long)h);
        }
    }
    compile_program(p);  // (this worker: queued again it would wait behind the other translations)
}

// ================================================================== shader content hashes (render thread)
// A program is identified by its stages' code contents, hashed HERE, at the draw, in the GPU order of commands: the
// game may write a shader's code with a GPU copy (CopyBufferToBuffer: a memmove on this thread, in order) that runs
// after nvnProgramSetShaders, so a hash taken in SetShaders could be of the old code at that address and pair the new
// program with the old translation (G4, the black rug, 1 run in 9 after keying by the SetShaders hash). Cached by
// address; an entry is dropped when a copy writes over its code or the game hands a program over at that address.
#define SH_TAB (1 << 14)
static struct { uint64_t addr, hash; uint32_t size; } g_sh_tab[SH_TAB];
static uint64_t g_sh_lo = ~0ull, g_sh_hi;  // range of the cached blobs (a copy outside it has nothing to drop)
static pthread_mutex_t g_sh_set_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t *g_sh_set;
static int g_nsh_set, g_cap_sh_set;
static uint64_t stat_sh_hashed, stat_sh_dropped;
// 64 KB pages that hold cached code (count per page): a copy that touches none of them has nothing to drop
#define SH_PAGES (1 << 13)
static struct { uint64_t page; uint32_t n; } g_sh_pages[SH_PAGES];
static int sh_page_slot(uint64_t page, bool add) {
    uint32_t i = (uint32_t)(page ^ (page >> 13)) & (SH_PAGES - 1);
    for (int k = 0; k < SH_PAGES; k++, i = (i + 1) & (SH_PAGES - 1)) {
        if (g_sh_pages[i].page == page && g_sh_pages[i].n) return (int)i;
        if (!g_sh_pages[i].n) {
            if (!add) { if (!g_sh_pages[i].page) return -1; continue; }  // (a freed slot: keep probing)
            g_sh_pages[i].page = page;
            return (int)i;
        }
    }
    return -1;
}
static void sh_pages_note(uint64_t addr, uint32_t size, int d) {
    for (uint64_t p = addr >> 16; p <= (addr + size - 1) >> 16; p++) {
        int s = sh_page_slot(p, d > 0);
        if (s >= 0) g_sh_pages[s].n += (uint32_t)d;
    }
}

void renderer_stage_set(uint64_t addr) {
    pthread_mutex_lock(&g_sh_set_lock);
    if (g_nsh_set == g_cap_sh_set) {
        g_cap_sh_set = g_cap_sh_set ? g_cap_sh_set * 2 : 256;
        g_sh_set = realloc(g_sh_set, (size_t)g_cap_sh_set * sizeof *g_sh_set);
    }
    g_sh_set[g_nsh_set++] = addr;
    pthread_mutex_unlock(&g_sh_set_lock);
}

static int sh_slot(uint64_t addr) {
    uint32_t i = (uint32_t)((addr >> 4) ^ (addr >> 21)) & (SH_TAB - 1);
    for (int n = 0; n < SH_TAB; n++, i = (i + 1) & (SH_TAB - 1))
        if (!g_sh_tab[i].addr || g_sh_tab[i].addr == addr) return (int)i;
    return -1;
}

static void sh_drop_slot(int i) {  // (open addressing: re-insert the rest of the cluster)
    sh_pages_note(g_sh_tab[i].addr, g_sh_tab[i].size, -1);
    g_sh_tab[i].addr = 0;
    stat_sh_dropped++;
    for (int j = (i + 1) & (SH_TAB - 1); g_sh_tab[j].addr; j = (j + 1) & (SH_TAB - 1)) {
        typeof(g_sh_tab[0]) e = g_sh_tab[j];
        g_sh_tab[j].addr = 0;
        int k = sh_slot(e.addr);
        if (k >= 0) g_sh_tab[k] = e;
    }
}

// a guest-memory write on this thread (copies): drop the hashes of the code it overlaps
static void sh_written(uint64_t dst, uint64_t size) {
    if (!size || dst + size <= g_sh_lo || dst >= g_sh_hi) return;
    bool any = false;
    for (uint64_t p = dst >> 16; p <= (dst + size - 1) >> 16 && !any; p++) any = sh_page_slot(p, false) >= 0;
    if (!any) return;
    for (int i = 0; i < SH_TAB; i++)
        if (g_sh_tab[i].addr && g_sh_tab[i].addr < dst + size && dst < g_sh_tab[i].addr + g_sh_tab[i].size) { sh_drop_slot(i); i--; }
}

static void sh_pump(void) {  // (start of each submitted list: the programs handed over since)
    uint64_t a[256];
    pthread_mutex_lock(&g_sh_set_lock);
    int n = g_nsh_set < 256 ? g_nsh_set : 256;
    memcpy(a, g_sh_set, (size_t)n * sizeof *a);
    memmove(g_sh_set, g_sh_set + n, (size_t)(g_nsh_set - n) * sizeof *a);
    g_nsh_set -= n;
    pthread_mutex_unlock(&g_sh_set_lock);
    for (int i = 0; i < n; i++) {
        int s = sh_slot(a[i]);
        if (s >= 0 && g_sh_tab[s].addr) sh_drop_slot(s);
    }
}

static uint64_t stage_hash_rt(const NvnShaderStage *ss) {
    if (!ss->code_addr || !ss->code_size) return 0;
    int s = sh_slot(ss->code_addr);
    if (s >= 0 && g_sh_tab[s].addr && g_sh_tab[s].size == ss->code_size) return g_sh_tab[s].hash;
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t *b = (const uint8_t *)(uintptr_t)ss->code_addr;
    for (uint32_t k = 0; k < ss->code_size; k++) h = (h ^ b[k]) * 0x100000001B3ull;
    if (!h) h = 1;
    stat_sh_hashed++;
    static const char *dump = (const char *)1;  // LARECOMP_DUMP_STAGES=<dir>: each stage drawn, once (tools/swsh/shader_store.py)
    if (dump == (const char *)1) dump = getenv("LARECOMP_DUMP_STAGES");
    if (dump) {
        char path[700];
        snprintf(path, sizeof path, "%s/%016llx_%llx.code", dump, (unsigned long long)h, (unsigned long long)ss->code_addr);
        FILE *f = fopen(path, "rb");
        if (f) fclose(f);
        else if ((f = fopen(path, "wb"))) {
            fwrite(b, 1, ss->code_size, f);
            fclose(f);
            // (research: the container the code sits in, if it is in guest memory: the nearest "BNSH" before it, and
            // the 256 bytes in front of that, in <hash>.ctx)
            uint64_t lo = HEAP_BASE, hi = HEAP_BASE + kernel_heap_size(), a = ss->code_addr & ~7ull;
            if (a >= lo && a < hi) {
                uint64_t stop = a - lo > (32u << 20) ? a - (32u << 20) : lo;
                for (uint64_t p = a; p >= stop + 8; p -= 8)
                    if (!memcmp((const void *)(uintptr_t)p, "BNSH\0\0\0\0", 8)) {
                        snprintf(path, sizeof path, "%s/%016llx.ctx", dump, (unsigned long long)h);
                        if ((f = fopen(path, "wb"))) {
                            uint64_t from = p - lo >= 256 ? p - 256 : lo;
                            fprintf(f, "%llx %llx\n", (unsigned long long)p, (unsigned long long)(a - p));
                            fwrite((const void *)(uintptr_t)from, 1, (size_t)(p + 0x40 - from), f);
                            fclose(f);
                        }
                        break;
                    }
            }
        }
    }
    if (s >= 0) {
        g_sh_tab[s].addr = ss->code_addr, g_sh_tab[s].hash = h, g_sh_tab[s].size = ss->code_size;
        sh_pages_note(ss->code_addr, ss->code_size, +1);
        if (ss->code_addr < g_sh_lo) g_sh_lo = ss->code_addr;
        if (ss->code_addr + ss->code_size > g_sh_hi) g_sh_hi = ss->code_addr + ss->code_size;
    }
    return h;
}

// Programs and pipelines by key hash, for the whole session: tables that double at half full. (They were fixed: 4096
// programs and 8192 pipelines, probed with no limit, so the render thread spun for good once one filled. Shader
// preparation fills the pipeline table early, so a long session could reach it: the freeze after a long session of
// Let's Go Recomp issues #21 / #33 / #35.)
typedef struct { uint64_t key; void *val; } MapSlot;
typedef struct { MapSlot *s; uint32_t cap, n; const char *name; } Map;
static uint32_t map_start(uint32_t dflt) {  // (LARECOMP_VK_MAP_START=<n>: small tables, to test the growing)
    const char *e = getenv("LARECOMP_VK_MAP_START");
    uint32_t n = e && atoi(e) >= 8 ? (uint32_t)atoi(e) : dflt, c = 8;
    while (c < n) c <<= 1;
    return c;
}
static void *map_get(const Map *m, uint64_t key) {
    if (!m->s) return NULL;
    for (uint32_t i = (uint32_t)(key ^ (key >> 31)) & (m->cap - 1); m->s[i].key; i = (i + 1) & (m->cap - 1))
        if (m->s[i].key == key) return m->s[i].val;
    return NULL;
}
static void map_put(Map *m, uint64_t key, void *val, uint32_t start) {  // (key not in the table; nonzero)
    if (!m->s || (m->n + 1) * 2 > m->cap) {
        uint32_t cap = m->s ? m->cap * 2 : map_start(start);
        MapSlot *s = calloc(cap, sizeof *s);
        for (uint32_t i = 0; m->s && i < m->cap; i++) {
            if (!m->s[i].key) continue;
            uint32_t j = (uint32_t)(m->s[i].key ^ (m->s[i].key >> 31)) & (cap - 1);
            while (s[j].key) j = (j + 1) & (cap - 1);
            s[j] = m->s[i];
        }
        if (m->s) LOG("[vk] %s table: %u -> %u slots (%u entries)\n", m->name, m->cap, cap, m->n);
        free(m->s);
        m->s = s, m->cap = cap;
    }
    uint32_t i = (uint32_t)(key ^ (key >> 31)) & (m->cap - 1);
    while (m->s[i].key) i = (i + 1) & (m->cap - 1);
    m->s[i].key = key, m->s[i].val = val;
    m->n++;
}
static void map_clear(Map *m) {
    if (m->s) memset(m->s, 0, (size_t)m->cap * sizeof *m->s);
    m->n = 0;
}

static Map prog_tab = {.name = "program"};

// BindProgram(program, stages): the stages in the mask come from `program` (NULL: unbound), the others stay. A bind
// that leaves one program's stages and nothing else uses that program itself (as before); a mix of stages from
// several programs (Let's Go: vertex and fragment bound separately) uses a combined program. The shader caches are
// keyed by code address, so the combination needs no identity of its own.
static NvnProgram g_comb_prog;
static void bind_program_stages(NvnProgram *p, uint32_t mask) {
    if (!mask) mask = p ? p->stage_mask : 0x3F;  // (no mask recorded: the whole program)
    NvnProgram *cur = st.program;
    NvnProgram next = cur ? *cur : (NvnProgram){0};
    next.backend = NULL;
    for (int i = 0; i < 6; i++) {
        uint32_t bit = 1u << i;
        if (!(mask & bit)) continue;
        if (p && (p->stage_mask & bit)) next.stages[i] = p->stages[i], next.stage_mask |= bit;
        else memset(&next.stages[i], 0, sizeof next.stages[i]), next.stage_mask &= ~bit;
    }
    if (!next.stage_mask) { st.program = NULL; return; }
    if (p && next.stage_mask == p->stage_mask &&
        !memcmp(next.stages, p->stages, sizeof next.stages)) { st.program = p; return; }
    g_comb_prog = next;
    st.program = &g_comb_prog;
}

static Prog *get_program_q(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey, bool prewarm);
static Prog *get_program(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) { return get_program_q(np, vkey, fkey, false); }
static Prog *get_program_q(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey, bool prewarm) {
    // A vertex shader without a fragment shader (Let's Go's depth / shadow passes: stages 0x1) gets an empty fragment
    // shader in translate_program - no color output, only depth, as the hardware does. They were all skipped.
    if (!(np->stage_mask & 1)) {
        static uint32_t seen;
        if (np->stage_mask < 32 && !(seen & (1u << np->stage_mask))) {
            seen |= 1u << np->stage_mask;
            LOG("[shader] draw with stages %#x (needs vertex + fragment): skipped\n", np->stage_mask);
        }
        return NULL;
    }
    uint64_t h = 0xcbf29ce484222325ull;
    // The stages by content when their hashes are known (the game holds byte-identical shaders at several addresses:
    // 531 of 649 pipelines in one run were such copies, each translated and built again), else by address.
    uint64_t ch[2] = {stage_hash_rt(&np->stages[0]), (np->stage_mask & 2) ? stage_hash_rt(&np->stages[1]) : 0};
    bool by_content = ch[0] && (ch[1] || !(np->stage_mask & 2));
    for (int i = 0; i < 2; i++)  // (research: the code changed between nvnProgramSetShaders and this draw)
        if (ch[i] && np->stages[i].content_hash && ch[i] != np->stages[i].content_hash) {
            static int n;
            if (n++ < 40) LOG("[shhash] stage %d at %llx: SetShaders hash %016llx, draw hash %016llx\n", i,
                              (unsigned long long)np->stages[i].code_addr, (unsigned long long)np->stages[i].content_hash,
                              (unsigned long long)ch[i]);
        }
    for (int i = 0; i < 2; i++) {
        uint64_t a = by_content ? ch[i] : np->stages[i].code_addr;
        for (int k = 0; k < 8; k++) { h ^= (a >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
    }
    h ^= by_content;
    const uint8_t *vb = (const uint8_t *)vkey;
    for (size_t i = 0; i < sizeof *vkey; i++) { h ^= vb[i]; h *= 0x100000001B3ull; }
    for (int i = 0; i < 8; i++) { h ^= fkey->rt_kind[i]; h *= 0x100000001B3ull; }
    h = (h ^ fkey->tex_uint) * 0x100000001B3ull;
    h = (h ^ ((uint64_t)fkey->tex_sint << 32)) * 0x100000001B3ull;
    if (!h) h = 1;
    Prog *found = map_get(&prog_tab, h);
    if (found) return found;
    Prog *p = calloc(1, sizeof *p);
    map_put(&prog_tab, h, p, 4096);
    for (int i = 0; i < 2; i++) {
        p->code[i] = np->stages[i].code_addr;
        p->cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
    }
    p->vkey = *vkey;
    p->fkey = *fkey;
    p->chash[0] = by_content ? ch[0] : 0;
    p->chash[1] = by_content ? ch[1] : 0;
    if (prewarm) run_job_low(translate_program, p);
    else run_job(translate_program, p);
    return p;
}

// ================================================================== vertex formats / fixed function tables
static int attr_kind(uint32_t fmt) {
    switch (fmt) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x1B: case 0x20: case 0x23:
    case 0x27: case 0x2C: case 0x2F: case 0x3E:
        return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x1C: case 0x21: case 0x24:
    case 0x28: case 0x2D: case 0x30:
        return 1;
    default: return 0;
    }
}

static VkFormat vertex_format(uint32_t f) {
    switch (f) {
    case 0x01: return VK_FORMAT_R8_UNORM;
    case 0x02: return VK_FORMAT_R8_SNORM;
    case 0x03: return VK_FORMAT_R8_UINT;
    case 0x04: return VK_FORMAT_R8_SINT;
    case 0x05: return VK_FORMAT_R16_SFLOAT;
    case 0x06: return VK_FORMAT_R16_UNORM;
    case 0x07: return VK_FORMAT_R16_SNORM;
    case 0x08: return VK_FORMAT_R16_UINT;
    case 0x09: return VK_FORMAT_R16_SINT;
    case 0x0A: return VK_FORMAT_R32_SFLOAT;
    case 0x0B: return VK_FORMAT_R32_UINT;
    case 0x0C: return VK_FORMAT_R32_SINT;
    case 0x0D: return VK_FORMAT_R8G8_UNORM;
    case 0x0E: return VK_FORMAT_R8G8_SNORM;
    case 0x0F: return VK_FORMAT_R8G8_UINT;
    case 0x10: return VK_FORMAT_R8G8_SINT;
    case 0x11: return VK_FORMAT_R16G16_SFLOAT;
    case 0x12: return VK_FORMAT_R16G16_UNORM;
    case 0x13: return VK_FORMAT_R16G16_SNORM;
    case 0x14: return VK_FORMAT_R16G16_UINT;
    case 0x15: return VK_FORMAT_R16G16_SINT;
    case 0x16: return VK_FORMAT_R32G32_SFLOAT;
    case 0x17: return VK_FORMAT_R32G32_UINT;
    case 0x18: return VK_FORMAT_R32G32_SINT;
    case 0x19: return VK_FORMAT_R8G8B8_UNORM;
    case 0x1A: return VK_FORMAT_R8G8B8_SNORM;
    case 0x1B: return VK_FORMAT_R8G8B8_UINT;
    case 0x1C: return VK_FORMAT_R8G8B8_SINT;
    case 0x1D: return VK_FORMAT_R16G16B16_SFLOAT;
    case 0x1E: return VK_FORMAT_R16G16B16_UNORM;
    case 0x1F: return VK_FORMAT_R16G16B16_SNORM;
    case 0x20: return VK_FORMAT_R16G16B16_UINT;
    case 0x21: return VK_FORMAT_R16G16B16_SINT;
    case 0x22: return VK_FORMAT_R32G32B32_SFLOAT;
    case 0x23: return VK_FORMAT_R32G32B32_UINT;
    case 0x24: return VK_FORMAT_R32G32B32_SINT;
    case 0x25: return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x26: return VK_FORMAT_R8G8B8A8_SNORM;
    case 0x27: return VK_FORMAT_R8G8B8A8_UINT;
    case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
    case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 0x2A: return VK_FORMAT_R16G16B16A16_UNORM;
    case 0x2B: return VK_FORMAT_R16G16B16A16_SNORM;
    case 0x2C: return VK_FORMAT_R16G16B16A16_UINT;
    case 0x2D: return VK_FORMAT_R16G16B16A16_SINT;
    case 0x2E: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 0x2F: return VK_FORMAT_R32G32B32A32_UINT;
    case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
    case 0x3D: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case 0x3F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 0x40: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 0x51: return VK_FORMAT_R8_UINT;
    default: {
        static uint8_t told[256];
        if (f < 256 && !told[f]++) LOG("[vk] vertex format %#x unknown: read as RGBA32F\n", f);
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    }
}

// 3-component 8/16-bit vertex formats aren't universally supported: read 4 components instead.
static VkFormat vertex_format_supported(VkFormat f) {
    static uint8_t known[256];  // 1 ok, 2 not
    uint32_t i = f < 256 ? f : 0;
    if (!known[i]) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &p);
        known[i] = (p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : 2;
    }
    if (known[i] == 1) return f;
    { static uint8_t told[256]; if (!told[i]++) LOG("[vk] vertex format %d unsupported: read as 4 components\n", f); }
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

static VkBlendFactor blend_factor(uint8_t f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: {
        static uint8_t warned[256];
        if (!warned[f]++) LOG("[vk] unknown blend factor %#x\n", f);
        return VK_BLEND_FACTOR_ONE;
    }
    }
}

static VkBlendOp blend_op(uint8_t e) {
    switch (e) {
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

static VkCompareOp compare_fn(uint8_t f) {
    switch (f) {
    case 1: return VK_COMPARE_OP_NEVER;
    case 2: return VK_COMPARE_OP_LESS;
    case 3: return VK_COMPARE_OP_EQUAL;
    case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 5: return VK_COMPARE_OP_GREATER;
    case 6: return VK_COMPARE_OP_NOT_EQUAL;
    case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static VkStencilOp stencil_op(uint8_t o) {
    switch (o) {
    case 2: return VK_STENCIL_OP_ZERO;
    case 3: return VK_STENCIL_OP_REPLACE;
    case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 6: return VK_STENCIL_OP_INVERT;
    case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default: return VK_STENCIL_OP_KEEP;
    }
}

static int rt_kind(uint32_t f) {
    switch (f) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x27: case 0x2C: case 0x2F: case 0x3E: return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x28: case 0x2D: case 0x30: return 1;
    default: return 0;
    }
}

// ================================================================== graphics pipelines
// Everything a pipeline bakes in, captured by value so it can be built on a worker thread.
#define ZERO_BINDING 16
typedef struct {
    Prog *prog;
    VkFormat color[8], depth, stencil;
    uint32_t ncolor, samples;
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32];
    uint32_t attr_mask;
    uint32_t stride[17], divisor[17], used_streams;
    uint8_t blend[8][8];
    uint8_t blend_on, a2c, topo_class, poly_fill, depth_clamp;
    uint32_t write_mask[8];
} PipeKey;

// state: 0 building, 1 ready, 2 failed. fast: linked from parts (below); fstate: the same for it, -1 not asked for
typedef struct { PipeKey k; uint64_t hash; VkPipeline pipe, fast; _Atomic int state, fstate; bool prewarmed, drawn; } Pipe;
static void link_pipeline(void *arg);
static Map pipe_tab = {.name = "pipeline"};

// 8 bytes a step (PipeKey is ~700 bytes, hashed per draw); in-memory tables only - hmix values are compared with
// stored constants (fragment-shader hashes) and must not change.
static uint64_t hash_words(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (; n >= 8; n -= 8, b += 8) { uint64_t w; memcpy(&w, b, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; n; n--, b++) { h = (h ^ *b) * 0x100000001B3ull; }
    return h ^ (h >> 32);
}

static uint64_t hmix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}

static VkPrimitiveTopology topo_class_base(uint8_t c) {
    return c == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : c == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

// LARECOMP_PIPE_STATS=<vertex program address, hex>|all: after each pipeline is created, what the driver reports about its
// compiled shaders (VK_KHR_pipeline_executable_properties: one executable per stage; registers, spills, instructions...).
static void pipe_exec_report(VkPipeline pipe, uint64_t vs, uint64_t fs) {
    const char *want = getenv("LARECOMP_PIPE_STATS");
    if (!want || (strcmp(want, "all") && strtoull(want, NULL, 16) != vs)) return;
    static PFN_vkGetPipelineExecutablePropertiesKHR get_props;
    static PFN_vkGetPipelineExecutableStatisticsKHR get_stats;
    if (!get_props) {
        get_props = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetPipelineExecutablePropertiesKHR");
        get_stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetPipelineExecutableStatisticsKHR");
        if (!get_props || !get_stats) { LOG("[pipestats] the driver does not export the functions\n"); return; }
    }
    uint32_t n = 0;
    VkPipelineInfoKHR pinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipe};
    if (get_props(g_vk.dev, &pinfo, &n, NULL) != VK_SUCCESS || !n) return;
    VkPipelineExecutablePropertiesKHR props[8] = {0};
    if (n > 8) n = 8;
    for (uint32_t i = 0; i < n; i++) props[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    if (get_props(g_vk.dev, &pinfo, &n, props) != VK_SUCCESS) return;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t ns = 0;
        VkPipelineExecutableInfoKHR einfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipe, .executableIndex = i};
        if (get_stats(g_vk.dev, &einfo, &ns, NULL) != VK_SUCCESS) continue;
        VkPipelineExecutableStatisticKHR stats[64] = {0};
        if (ns > 64) ns = 64;
        for (uint32_t s = 0; s < ns; s++) stats[s].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        if (get_stats(g_vk.dev, &einfo, &ns, stats) != VK_SUCCESS) continue;
        char line[1024];
        int l = snprintf(line, sizeof line, "[pipestats] vs %llx fs %llx  %s (stages %#x):", (unsigned long long)vs, (unsigned long long)fs,
                         props[i].name, props[i].stages);
        for (uint32_t s = 0; s < ns && l < (int)sizeof line - 80; s++) {
            const VkPipelineExecutableStatisticValueKHR *v = &stats[s].value;
            switch (stats[s].format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: l += snprintf(line + l, sizeof line - l, " %s=%u;", stats[s].name, v->b32); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: l += snprintf(line + l, sizeof line - l, " %s=%lld;", stats[s].name, (long long)v->i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: l += snprintf(line + l, sizeof line - l, " %s=%llu;", stats[s].name, (unsigned long long)v->u64); break;
            default: l += snprintf(line + l, sizeof line - l, " %s=%g;", stats[s].name, v->f64); break;
            }
        }
        LOG("%s\n", line);
    }
}

// LARECOMP_DRIVER_CACHE_NONCE=<n> (tests): every graphics shader stage carries an unused specialization constant with
// this value, so the driver's own shader cache (kept across runs, outside cache/) misses: a first-time compile measured
// as a player meets it, without clearing anything of the driver's.
static const VkSpecializationInfo *driver_nonce(void) {
    static int on = -1;
    static uint32_t v;
    static VkSpecializationMapEntry e = {.constantID = 0x7FFFFFF0u, .offset = 0, .size = 4};
    static VkSpecializationInfo si = {.mapEntryCount = 1, .pMapEntries = &e, .dataSize = 4, .pData = &v};
    if (on < 0) { const char *n = getenv("LARECOMP_DRIVER_CACHE_NONCE"); on = n && *n; v = n ? (uint32_t)strtoul(n, NULL, 0) : 0; }
    return on ? &si : NULL;
}

static void build_pipeline(void *arg) {
    Pipe *pp = arg;
    PipeKey *k = &pp->k;
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = k->prog->vs, .pName = "main",
         .pSpecializationInfo = driver_nonce()},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = k->prog->fs, .pName = "main",
         .pSpecializationInfo = driver_nonce()}};
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    uint32_t na = 0, nb = 0, nd = 0;
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        bool inst = k->divisor[s] != 0;
        vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        if (inst && k->divisor[s] > 1 && g_vk.divisor) vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
                                                          .vertexBindingDivisorCount = nd, .pVertexBindingDivisors = vd};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nd ? &dvi : NULL,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    {   // (LARECOMP_PIPE_STATS: the vertex fetch layout of the program asked for)
        const char *w = getenv("LARECOMP_PIPE_STATS");
        if (w && (!strcmp(w, "all") || strtoull(w, NULL, 16) == k->prog->code[0])) {
            LOG("[pipestats] vs %llx inputs:", (unsigned long long)k->prog->code[0]);
            for (uint32_t i = 0; i < na; i++) LOG(" a%u(bind %u fmt %d off %u)", va[i].location, va[i].binding, (int)va[i].format, va[i].offset);
            LOG(" | bindings:");
            for (uint32_t i = 0; i < nb; i++)
                LOG(" %u(stride %u %s)", vb[i].binding, vb[i].stride, vb[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE ? "instance" : "vertex");
            for (uint32_t i = 0; i < nd; i++) LOG(" div%u=%u", vd[i].binding, vd[i].divisor);
            LOG("\n");
        }
    }
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = topo_class_base(k->topo_class)};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = k->poly_fill ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE, .lineWidth = 1,
                                                 .depthClampEnable = k->depth_clamp != 0};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < k->ncolor; i++) {
        cba[i].colorWriteMask = k->write_mask[i];
        if ((k->blend_on >> i) & 1) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = VK_TRUE;
            cba[i].srcColorBlendFactor = blend_factor(b[1]);
            cba[i].dstColorBlendFactor = blend_factor(b[2]);
            cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
            cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
            cba[i].colorBlendOp = blend_op(b[5]);
            cba[i].alphaBlendOp = blend_op(b[6]);
        }
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = k->ncolor, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->ncolor,
                                        .pColorAttachmentFormats = k->color, .depthAttachmentFormat = k->depth,
                                        .stencilAttachmentFormat = k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = k->prog->layout};
    uint64_t tp = SDL_GetTicksNS();
    extern bool g_pipe_exec_stats;
    if (g_pipe_exec_stats) pi.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &pp->pipe);
    uint64_t pdt = SDL_GetTicksNS() - tp;
    atomic_fetch_add(&stat_pipe_ns, pdt);
    if (pdt > 150000000ull) {  // (slow driver compiles: which programs, to look at their translation)
        static _Atomic int nslow;
        if (atomic_fetch_add(&nslow, 1) < 200)
            LOG("[slowpipe] %.0f ms: vs %llx fs %llx\n", pdt / 1e6, (unsigned long long)k->prog->code[0], (unsigned long long)k->prog->code[1]);
    }
    if (r == VK_SUCCESS && g_pipe_exec_stats) pipe_exec_report(pp->pipe, k->prog->code[0], k->prog->code[1]);
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline creation failed: %d\n", r);
    }
    atomic_fetch_add(&stat_compiles, 1);
    if (r == VK_SUCCESS) atomic_fetch_add(&new_pipes, 1);
    atomic_store(&pp->state, r == VK_SUCCESS ? 1 : 2);
}

// ================================================================== graphics pipeline library (fast-linked pipelines)
// A whole pipeline is one driver compile of both shaders with every baked-in state: ~20-120 ms each, up to 2.5 s, and
// a first visit to a town needed 85 new ones, built one at a time while the frame waited (17 s). With
// VK_EXT_graphics_pipeline_library (g_vk.gpl) a pipeline is linked from four parts:
//  - vertex input (vertex layout, topology class) and fragment output (formats, blend, write masks, samples): cheap;
//  - pre-rasterization (the vertex shader) and fragment shader (the fragment shader): the compiles, but one per program
//    (and a few states), shared by every combination of the cheap parts;
// and a fast link (no link-time optimization) takes well under a millisecond. A draw whose whole pipeline is not ready
// uses the linked one; the whole pipeline is still built (low priority) and replaces it once ready.
// Parts are kept for the session in a table; a part being built by another thread is waited for.
enum { GP_VI, GP_PR, GP_FS, GP_FO };
typedef struct {
    int kind;
    Prog *prog;
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32], attr_mask, stride[17], divisor[17], used_streams;  // VI
    uint8_t topo_class, poly_fill, depth_clamp, a2c;                                                     // VI / PR / FS, FO
    uint32_t samples;                                                                                    // FS, FO
    VkFormat color[8], depth, stencil;                                                                   // FO
    uint32_t ncolor, write_mask[8];
    uint8_t blend[8][8], blend_on;
} GplKey;
typedef struct GplPart { GplKey k; uint64_t hash; VkPipeline lib; _Atomic int state; struct GplPart *next; } GplPart;
#define GPL_TAB 8192
static GplPart *gpl_tab[GPL_TAB];
static pthread_mutex_t gpl_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t stat_gpl_links, stat_gpl_link_ns, stat_gpl_parts, stat_gpl_part_ns, stat_gpl_used, stat_gpl_failed;

static const VkDynamicState GPL_DYN_VI[] = {VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
static const VkDynamicState GPL_DYN_PR[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS,
                                            VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE};
static const VkDynamicState GPL_DYN_FS[] = {VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
                                            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
                                            VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
static const VkDynamicState GPL_DYN_FO[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};

static void gpl_key(int kind, const PipeKey *k, GplKey *g) {
    memset(g, 0, sizeof *g);
    g->kind = kind;
    switch (kind) {
    case GP_VI:
        memcpy(g->attr_fmt, k->attr_fmt, sizeof g->attr_fmt);
        memcpy(g->attr_off, k->attr_off, sizeof g->attr_off);
        memcpy(g->attr_bind, k->attr_bind, sizeof g->attr_bind);
        g->attr_mask = k->attr_mask;
        memcpy(g->stride, k->stride, sizeof g->stride);
        memcpy(g->divisor, k->divisor, sizeof g->divisor);
        g->used_streams = k->used_streams;
        g->topo_class = k->topo_class;
        break;
    case GP_PR:
        g->prog = k->prog;
        g->poly_fill = k->poly_fill;
        g->depth_clamp = k->depth_clamp;
        break;
    case GP_FS:
        g->prog = k->prog;
        g->samples = k->samples;
        g->a2c = k->a2c;
        break;
    case GP_FO:
        memcpy(g->color, k->color, sizeof g->color);
        g->depth = k->depth;
        g->stencil = k->stencil;
        g->ncolor = k->ncolor;
        g->samples = k->samples;
        g->a2c = k->a2c;
        memcpy(g->write_mask, k->write_mask, sizeof g->write_mask);
        memcpy(g->blend, k->blend, sizeof g->blend);
        g->blend_on = k->blend_on;
        break;
    }
}

// Builds one part from the pipeline key it was derived from (its other fields are ignored by that subset).
static VkPipeline gpl_build_part(int kind, const PipeKey *k) {
    static const VkGraphicsPipelineLibraryFlagsEXT sub[4] = {
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    VkGraphicsPipelineLibraryCreateInfoEXT li = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, .flags = sub[kind]};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .pNext = &li};
    if (kind == GP_FO) {
        ri.colorAttachmentCount = k->ncolor;
        ri.pColorAttachmentFormats = k->color;
        ri.depthAttachmentFormat = k->depth;
        ri.stencilAttachmentFormat = k->stencil;
    }
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR};  // (no link-time optimization info: only fast links are made, and it made each part ~75 KB in the pipeline cache)
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    pi.pDynamicState = &dy;
    // (vertex input)
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    // (pre-rasterization)
    VkPipelineShaderStageCreateInfo st = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pName = "main",
                                          .pSpecializationInfo = driver_nonce()};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .lineWidth = 1};
    // (fragment)
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = k->a2c};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    switch (kind) {
    case GP_VI: {
        uint32_t na = 0, nb = 0, nd = 0;
        for (int i = 0; i < 32; i++)
            if (k->attr_mask & (1u << i)) va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
        for (uint32_t s = 0; s < 17; s++) {
            if (!(k->used_streams & (1u << s))) continue;
            bool inst = k->divisor[s] != 0;
            vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
            if (inst && k->divisor[s] > 1 && g_vk.divisor) vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
        }
        dvi.vertexBindingDivisorCount = nd;
        dvi.pVertexBindingDivisors = vd;
        vi.pNext = nd ? &dvi : NULL;
        vi.vertexBindingDescriptionCount = nb;
        vi.pVertexBindingDescriptions = vb;
        vi.vertexAttributeDescriptionCount = na;
        vi.pVertexAttributeDescriptions = va;
        ia.topology = topo_class_base(k->topo_class);
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        dy.dynamicStateCount = 2; dy.pDynamicStates = GPL_DYN_VI;
        break;
    }
    case GP_PR:
        st.stage = VK_SHADER_STAGE_VERTEX_BIT;
        st.module = k->prog->vs;
        rs.polygonMode = k->poly_fill ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE;
        rs.depthClampEnable = k->depth_clamp != 0;
        pi.stageCount = 1; pi.pStages = &st;
        pi.pViewportState = &vs;
        pi.pRasterizationState = &rs;
        pi.layout = k->prog->stage_layout[0];
        dy.dynamicStateCount = sizeof GPL_DYN_PR / sizeof GPL_DYN_PR[0]; dy.pDynamicStates = GPL_DYN_PR;
        break;
    case GP_FS:
        st.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st.module = k->prog->fs;
        pi.stageCount = 1; pi.pStages = &st;
        pi.pDepthStencilState = &ds;
        pi.pMultisampleState = &ms;
        pi.layout = k->prog->stage_layout[1];
        dy.dynamicStateCount = sizeof GPL_DYN_FS / sizeof GPL_DYN_FS[0]; dy.pDynamicStates = GPL_DYN_FS;
        break;
    case GP_FO:
        for (uint32_t i = 0; i < k->ncolor; i++) {
            cba[i].colorWriteMask = k->write_mask[i];
            if ((k->blend_on >> i) & 1) {
                const uint8_t *b = k->blend[i];
                cba[i].blendEnable = VK_TRUE;
                cba[i].srcColorBlendFactor = blend_factor(b[1]);
                cba[i].dstColorBlendFactor = blend_factor(b[2]);
                cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
                cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
                cba[i].colorBlendOp = blend_op(b[5]);
                cba[i].alphaBlendOp = blend_op(b[6]);
            }
        }
        cb.attachmentCount = k->ncolor;
        cb.pAttachments = cba;
        pi.pColorBlendState = &cb;
        pi.pMultisampleState = &ms;
        dy.dynamicStateCount = 1; dy.pDynamicStates = GPL_DYN_FO;
        break;
    }
    VkPipeline p = VK_NULL_HANDLE;
    uint64_t t0 = SDL_GetTicksNS();
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache_parts, 1, &pi, NULL, &p);  // (their own cache: device.c)
    atomic_store(&g_vk.parts_dirty, 1);
    uint64_t dt = SDL_GetTicksNS() - t0;
    atomic_fetch_add(&stat_gpl_part_ns, dt);
    atomic_fetch_add(&stat_gpl_parts, 1);
    // A stage's part that took the driver more than 20 ms was not in the caches: the shader preparation (prep.c) did not
    // build it. The stage, the hash of its GLSL (prep.c logs the same with LARECOMP_PREP_DEBUG) and the keys that
    // shape it, to tell a stage the game files lack from a key the preparation guessed differently.
    const Prog *pg = k->prog;
    if ((kind == GP_PR || kind == GP_FS) && dt > 20000000 && pg->vi && pg->fi) {
        static _Atomic int nmiss;
        if (atomic_fetch_add(&nmiss, 1) < 4000) {
            const ShaderInfo *si = kind == GP_FS ? pg->fi : pg->vi;
            uint64_t g = 0xcbf29ce484222325ull;
            for (const char *q = si->msl; q && *q; q++) g = (g ^ (uint8_t)*q) * 0x100000001B3ull;
            char keys[160];
            if (kind == GP_PR) {
                int n = snprintf(keys, sizeof keys, "attrs");
                for (int i = 0; i < 32 && n < (int)sizeof keys - 8; i++)
                    if (pg->vi->attr_mask & (1u << i)) n += snprintf(keys + n, sizeof keys - n, " %d:%d", i, pg->vkey.attr_kind[i]);
            } else {
                snprintf(keys, sizeof keys, "inputs %x of %x, rt %d%d%d%d, tex u%x s%x", pg->fi->attr_mask & pg->fkey.vs_out_mask,
                         pg->fi->attr_mask, pg->fkey.rt_kind[0], pg->fkey.rt_kind[1], pg->fkey.rt_kind[2], pg->fkey.rt_kind[3],
                         pg->fkey.tex_uint, pg->fkey.tex_sint);
            }
            LOG("[partmiss] %s %016llx glsl %016llx %.0f ms (%s)\n", kind == GP_FS ? "fs" : "vs",
                (unsigned long long)pg->chash[kind == GP_FS], (unsigned long long)g, dt / 1e6, keys);
        }
    }
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline library part %d failed: %d\n", kind, r);
        return VK_NULL_HANDLE;
    }
    return p;
}

// The part for this pipeline key: built here if nobody has, waited for if another thread is building it.
static VkPipeline gpl_part(int kind, const PipeKey *k) {
    GplKey g;
    gpl_key(kind, k, &g);
    uint64_t h = hash_words(0x9E3779B97F4A7C15ull, &g, sizeof g);
    pthread_mutex_lock(&gpl_lock);
    GplPart *p = gpl_tab[h & (GPL_TAB - 1)];
    while (p && !(p->hash == h && !memcmp(&p->k, &g, sizeof g))) p = p->next;
    bool mine = false;
    if (!p) {
        p = calloc(1, sizeof *p);
        p->k = g;
        p->hash = h;
        p->next = gpl_tab[h & (GPL_TAB - 1)];
        gpl_tab[h & (GPL_TAB - 1)] = p;
        mine = true;
    }
    pthread_mutex_unlock(&gpl_lock);
    if (mine) {
        p->lib = gpl_build_part(kind, k);
        atomic_store(&p->state, p->lib ? 1 : 2);
    } else {
        while (atomic_load(&p->state) == 0) SDL_DelayNS(20000);
    }
    return atomic_load(&p->state) == 1 ? p->lib : VK_NULL_HANDLE;
}

// A pipeline linked from its four parts (no link-time optimization: fast), or VK_NULL_HANDLE.
static VkPipeline gpl_link(const PipeKey *k) {
    VkPipeline libs[4];
    for (int i = 0; i < 4; i++)
        if (!(libs[i] = gpl_part(i, k))) { atomic_fetch_add(&stat_gpl_failed, 1); return VK_NULL_HANDLE; }
    VkPipelineLibraryCreateInfoKHR lc = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4, .pLibraries = libs};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &lc, .layout = k->prog->layout};
    VkPipeline p = VK_NULL_HANDLE;
    uint64_t t0 = SDL_GetTicksNS();
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p);
    atomic_fetch_add(&stat_gpl_link_ns, SDL_GetTicksNS() - t0);
    atomic_fetch_add(&stat_gpl_links, 1);
    if (r != VK_SUCCESS) { atomic_fetch_add(&stat_gpl_failed, 1); return VK_NULL_HANDLE; }
    return p;
}

static void link_pipeline(void *arg) {  // (a worker, or the draw that needs it first)
    Pipe *pp = arg;
    pp->fast = gpl_link(&pp->k);
    atomic_store(&pp->fstate, pp->fast ? 1 : 2);
}

// The vertex (pre-rasterization) or fragment part of one stage, as a draw builds it with the most common state of this
// game (filled polygons, no depth clamp, one sample, no alpha to coverage: 3298 of 3687 recorded pipelines), for the
// shader preparation (prep.c). The caller destroys it: what it is for is the pipeline cache entry it leaves.
VkPipeline vk_prep_part(int fragment, VkShaderModule m, VkPipelineLayout layout) {
    Prog p = {0};
    PipeKey k = {.prog = &p, .samples = 1, .poly_fill = 1};
    if (fragment) p.fs = m; else p.vs = m;
    p.stage_layout[fragment] = layout;
    return gpl_build_part(fragment ? GP_FS : GP_PR, &k);
}

// ================================================================== feedback snapshots (renderer.m resolve_feedback)
static NvnTexture *feedback_tex[4];
static VkImageView feedback_view[4];
static int nfeedback;
static uint64_t g_cur_fs_hash;
static NvnTexture *snap_tex[8];
static bool snap_ok[8];
static const char *snap_why[8];
static const char *snap_reason = "?";
static void snapshots_invalidate(NvnTexture *t) {
    for (int i = 0; i < 8; i++) if ((!t || snap_tex[i] == t) && snap_ok[i]) { snap_ok[i] = false; snap_why[i] = snap_reason; }
}

typedef struct { NvnTexture *key; TexBackend copy; } Snap;
static Snap snaps[16];
static int nsnaps;

static NvnTexture *shader_texture_object(int stage, ShaderTexture *st_) {
    int slot = ((int)st_->handle_offset - 0x20) / 8;
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    if (!h || !st.tpool || tic >= (uint32_t)st.tpool->count) return NULL;
    return ((NvnTexEntry *)st.tpool->entries)[tic].tex;
}

static bool is_attachment(NvnTexture *t) {
    if (!t) return false;
    if (t == st.rt.depth) return true;
    for (int i = 0; i < st.rt.ncolor; i++) if (t == st.rt.color[i]) return true;
    return false;
}

static TexBackend *snapshot_image(NvnTexture *t, TexBackend *src) {
    Snap *s = NULL;
    for (int i = 0; i < nsnaps; i++) if (snaps[i].key == t) s = &snaps[i];
    if (!s) {
        if (nsnaps == 16) { static int rr; s = &snaps[rr++ & 15]; } else s = &snaps[nsnaps++];
        s->key = t;
    }
    TexBackend *d = &s->copy;
    if (d->image && (d->width != src->width || d->height != src->height || d->format != src->format)) {
        vk_views_forget(d);  // (resolution change)
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = src->format,
                                .extent = {src->width, src->height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = (VkSampleCountFlagBits)src->samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = src->format; d->width = src->width; d->height = src->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = src->samples; d->aspect = src->aspect; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

// (debug frames) a depth image's depth values as a greyscale PGM, stretched over its own min..max
static void debug_save_depth(TexBackend *b, const char *path) {
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || b->format != VK_FORMAT_D32_SFLOAT_S8_UINT) return;
    uint64_t n = (uint64_t)b->width * b->height;
    VkSpan out = vk_readback(n * 4);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                           .imageExtent = {b->width, b->height, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    const float *d = out.ptr;
    float lo = 1e30f, hi = -1e30f;
    for (uint64_t i = 0; i < n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i]; }
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5 %u %u 255\n", b->width, b->height);
    for (uint64_t i = 0; i < n; i++) fputc(hi > lo ? (int)((d[i] - lo) / (hi - lo) * 255.0f) : 0, f);
    fclose(f);
    LOG("[vk] saved %s (%ux%u depth %.6f..%.6f)\n", path, b->width, b->height, lo, hi);
}

static void resolve_feedback(Prog *pr) {
    nfeedback = 0;
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_FEEDBACK") != NULL;
    if (off) return;
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < infos[sg]->ntex; i++) {
            NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
            if (!is_attachment(t)) continue;
            bool have = false;
            for (int k = 0; k < nfeedback; k++) have |= feedback_tex[k] == t;
            if (have || nfeedback == 4) continue;
            static int nosnap = -1;
            if (nosnap < 0) nosnap = getenv("LARECOMP_NO_SNAPSHOT_REUSE") != NULL;
            int si = -1;
            for (int k = 0; k < 8; k++) if (snap_tex[k] == t) si = k;
            bool self_write = t == st.rt.depth && st.ds[0] && st.ds[1];
            TexBackend *src = vk_tex(t, true);
            Snap *existing = NULL;
            for (int k = 0; k < nsnaps; k++) if (snaps[k].key == t && snaps[k].copy.image) existing = &snaps[k];
            VkImageAspectFlags sa = src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : src->aspect;
            static const VkComponentMapping id = {0};
            if (!nosnap && si >= 0 && (snap_ok[si] || self_write) && existing) {
                feedback_tex[nfeedback] = t;
                feedback_view[nfeedback++] = vk_image_view(&existing->copy, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
                continue;
            }
            stat_feedback_copies++;
            vk_end_pass();
            TexBackend *dst = snapshot_image(t, src);
            VkCommandBuffer cb = vk_cmd();
            vk_full_barrier(cb);
            gp_mark(cb, "feedback snapshot %ux%u fmt %d", src->width, src->height, src->format);
            VkImageCopy rg = {.srcSubresource = {src->aspect, 0, 0, 1}, .dstSubresource = {src->aspect, 0, 0, 1},
                              .extent = {src->width, src->height, 1}};
            vkCmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
            if (g_debug_frame && (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
                static int nd;
                char path[128];
                snprintf(path, sizeof path, "build/fbdepth_%03d_src.pgm", nd);
                debug_save_depth(src, path);
                snprintf(path, sizeof path, "build/fbdepth_%03d_snap.pgm", nd++);
                debug_save_depth(dst, path);
                cb = vk_cmd();
            }
            feedback_tex[nfeedback] = t;
            feedback_view[nfeedback++] = vk_image_view(dst, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
            if (si < 0) { static int rr; si = rr++ & 7; snap_tex[si] = t; }
            snap_ok[si] = true;
            st.rt_dirty = true;
        }
}

// ================================================================== resource binding
static VkDescriptorBufferInfo ubo_span(VkSpan s, uint64_t size) {
    uint64_t max = g_vk.props.limits.maxUniformBufferRange;
    return (VkDescriptorBufferInfo){s.buf, s.off, size > max ? max : size};
}

// Constant buffer k of a stage (renderer.m bind_cbufs), copied into transient memory.
static VkDescriptorBufferInfo cbuf_info(int stage, ShaderInfo *si, int k, uint32_t gx, uint32_t gy, uint32_t gz) {
    static const uint8_t zeros[0x10000];
    if (k == 0) {
        uint32_t drv[0x800 / 4] = {0};
        if (stage == 5) {
            drv[0x200 / 4] = gx; drv[0x204 / 4] = gy; drv[0x208 / 4] = gz;
            for (int i = 0; i < 16; i++) {
                uint32_t *u = &drv[(0x210 + 16 * i) / 4];
                u[0] = (uint32_t)st.ubuf[5][i].addr; u[1] = (uint32_t)(st.ubuf[5][i].addr >> 32); u[2] = (uint32_t)st.ubuf[5][i].size;
                uint32_t *e = &drv[(0x310 + 16 * i) / 4];
                e[0] = (uint32_t)st.sbuf[5][i].addr; e[1] = (uint32_t)(st.sbuf[5][i].addr >> 32); e[2] = (uint32_t)st.sbuf[5][i].size;
            }
        } else {
            uint32_t base = 0x110 + 0x100 * (stage == 1 ? 4 : 0);
            for (int i = 0; i < 16; i++) {
                uint32_t *e = &drv[(base + 16 * i) / 4];
                e[0] = (uint32_t)st.sbuf[stage][i].addr;
                e[1] = (uint32_t)(st.sbuf[stage][i].addr >> 32);
                e[2] = (uint32_t)st.sbuf[stage][i].size;
            }
        }
        uint64_t size = si->cbuf_size[0] > sizeof drv ? sizeof drv : (si->cbuf_size[0] ? si->cbuf_size[0] : 256);
        if (size < 0x500 && stage == 5) size = 0x500;
        return ubo_span(copy_dedupe(stage == 5 ? DEDUPE_DRV_C : stage == 1 ? DEDUPE_DRV_F : DEDUPE_DRV_V, drv, size, 256), size);
    }
    if (k == 1) {  // compiler constants stored after the shader code
        NvnShaderStage *ss = &st.program->stages[stage];
        uint64_t size = ss->const_size ? ss->const_size : 16;
        if (size < si->cbuf_size[1]) size = si->cbuf_size[1];
        return ubo_span(guest_copy(ss->code_addr + ss->const_offset, size), size);  // (constant: shared per submission)
    }
    int idx = k - 3;  // cbuf_base
    Range r = (idx >= 0 && idx < 18) ? st.ubuf[stage][idx] : (Range){0, 0};
    uint64_t size = si->cbuf_size[k] ? si->cbuf_size[k] : 256;
    if (!r.addr) return ubo_span(vk_transient_copy(zeros, size, 256), size);
    if ((si->cbuf_dynamic & (1u << k)) && r.size) size = r.size > 0x10000 ? 0x10000 : r.size;
    if (size > r.size && r.size) size = r.size;
    if (size < 16) size = 16;
    bool skinned_vs = stage == 0 && ((si->attr_mask >> 11) & 1);  // (Let's Go: a11 = bone indices)
    // Experiment X2 (docs/letsgo/PLANO-3D.md), LARECOMP_VK_SKIN_IDENTITY=1: a skinned vertex shader's dynamically indexed
    // cbuf (its bone palette) reads identity matrices - models in their bind pose if the rest of the chain is right.
    static int skin_identity = -1;
    if (skin_identity < 0) skin_identity = getenv("LARECOMP_VK_SKIN_IDENTITY") != NULL;
    if (skin_identity && skinned_vs && (si->cbuf_dynamic & (1u << k))) {
        static float ident[0x10000 / 4];
        if (ident[0] != 1.0f)
            for (int m = 0; m < 0x10000 / 64; m++) ident[m * 16] = ident[m * 16 + 5] = ident[m * 16 + 10] = ident[m * 16 + 15] = 1.0f;
        g_exp_count[0]++;
        return ubo_span(vk_transient_copy(ident, size, 256), size);
    }
    // LARECOMP_VK_UBO_DIRECT=1 (experiment X3): bind the range in place in the imported guest heap, read when the GPU
    // runs - as on the console - in case compute shaders write it after this point. Counted: direct vs copied.
    static int direct = -1;
    if (direct < 0) direct = getenv("LARECOMP_VK_UBO_DIRECT") ? atoi(getenv("LARECOMP_VK_UBO_DIRECT")) : 0;
    if (direct) {
        extern bool vk_guest_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off);
        VkBuffer gb;
        VkDeviceSize goff;
        uint64_t align = g_vk.props.limits.minUniformBufferOffsetAlignment;
        if (!(r.addr % (align ? align : 256)) && vk_guest_direct(r.addr, size, &gb, &goff)) {
            if (skinned_vs) g_exp_count[1]++;
            return ubo_span((VkSpan){.buf = gb, .off = goff}, size);
        }
        if (skinned_vs) g_exp_count[2]++;
    }
    if (ip_enabled()) {  // frame interpolation: recorded, or the blended copy in the replay
        const void *d = ip_cbuf(stage, k, r.addr, (uint32_t)size);
        if (d) return ubo_span(vk_transient_copy(d, size, 256), size);
    }
    // snapshot: later CPU writes must not leak in. Shared by the draws of one submission binding the same range:
    // within a command list nothing writes guest memory (no inline UpdateUniformBuffer: that writes through below
    // and drops the shared copies).
    return ubo_span(guest_copy(r.addr, size), size);
}

// Global memory (shader LDG/STG, maxwell_gmem_base = HEAP_BASE): the guest heap imported as a GPU buffer
// (VK_EXT_external_memory_host), zero-copy like metal.m's g_guest_buffer, so compute results land in guest
// memory. The import pins its pages, so it covers only the heap the game set up (SetHeapSize), made on first use.
// LARECOMP_VK_NO_GMEM=1: no import, reads return zero.
static VkBuffer gmem_buf;
static VkDeviceMemory gmem_mem;
static uint64_t gmem_size;
#define GMEM_STEP 0x10000000ull  // (256 MB)
#define GMEM_MIN  0x20000000ull  // (512 MB: below this the import is of no use)

static bool gmem_import(uint64_t size) {
    PFN_vkGetMemoryHostPointerPropertiesEXT props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(g_vk.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    static uint64_t test_max = ~0ull;  // LARECOMP_VK_GMEM_MAX_MB=<n>: refuse larger imports, as a driver may (to test gmem_info)
    const char *tm = getenv("LARECOMP_VK_GMEM_MAX_MB");
    if (tm) test_max = (uint64_t)atoll(tm) << 20;
    if (size > test_max) return false;
    VkResult pr = props ? props(g_vk.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)(uintptr_t)HEAP_BASE, &mp) : VK_ERROR_EXTENSION_NOT_PRESENT;
    if (pr != VK_SUCCESS || !mp.memoryTypeBits) {
        LOG("[vk] guest heap import: host pointer properties: result %d, memory types 0x%x\n", pr, mp.memoryTypeBits);
        return false;
    }
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            .pHostPointer = (void *)(uintptr_t)HEAP_BASE};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = size,
                               .memoryTypeIndex = (uint32_t)__builtin_ctz(mp.memoryTypeBits)};
    VkDeviceMemory mem;
    VkResult ar = vkAllocateMemory(g_vk.dev, &ai, NULL, &mem);
    if (ar != VK_SUCCESS) {
        VkPhysicalDeviceMaintenance3Properties m3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m3};
        vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
        VkPhysicalDeviceMemoryProperties mprops;
        vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &mprops);
        LOG("[vk] guest heap import of %llu MB: result %d (memory types 0x%x, memory heap %llu MB, largest allocation %llu MB, storage buffer range %llu MB)\n",
            (unsigned long long)(size >> 20), ar, mp.memoryTypeBits, (unsigned long long)(mprops.memoryHeaps[mprops.memoryTypes[ai.memoryTypeIndex].heapIndex].size >> 20),
            (unsigned long long)(m3.maxMemoryAllocationSize >> 20), (unsigned long long)(p2.properties.limits.maxStorageBufferRange >> 20));
        return false;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};  // (uniforms read in place: cbuf_info)
    VkBuffer buf;
    if (vkCreateBuffer(g_vk.dev, &bi, NULL, &buf) != VK_SUCCESS) { vkFreeMemory(g_vk.dev, mem, NULL); return false; }
    vkBindBufferMemory(g_vk.dev, buf, mem, 0);
    if (gmem_buf) {  // the heap grew: the old import may still be in use by submitted work
        vk_wait_seq(submit_seq);
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
    }
    gmem_buf = buf;
    gmem_mem = mem;
    gmem_size = size;
    LOG("[vk] guest heap imported for shader global memory: %llu MB\n", (unsigned long long)(size >> 20));
    return true;
}

static uint64_t gmem_tried;  // heap size the import was last attempted for (gmem_info imports / re-imports)
static uint64_t gmem_cap = HEAP_MAX;  // the largest import the driver takes, once it has refused one

// A driver may refuse the whole heap (issue #49: all the game's text is drawn from storage buffers, ~1.4-1.65 GB into the heap, so
// without the import it is blank): the import then covers less, from the heap's start. Reads past its end return zero (robustBufferAccess).
static VkDescriptorBufferInfo gmem_info(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_GMEM") != NULL || !g_vk.host_import;
    uint64_t want = (kernel_heap_size() + 0xFFFF) & ~0xFFFFull;
    if (want > gmem_cap) want = gmem_cap;
    if (!off && want > gmem_size && !gmem_import(want)) {
        uint64_t heap = want;
        do want = want > GMEM_MIN + GMEM_STEP ? (want - 1) / GMEM_STEP * GMEM_STEP : 0;  // (the next multiple of the step below)
        while (want && !gmem_import(want));
        if (want) {
            gmem_cap = want;
            LOG("[vk] guest heap import: the driver refused %llu MB, took %llu MB\n", (unsigned long long)(heap >> 20), (unsigned long long)(want >> 20));
        } else {
            LOG("[vk] guest heap import failed (%llu MB): shader global memory reads return zero\n", (unsigned long long)(heap >> 20));
            off = 1;
        }
    }
    if (gmem_buf) return (VkDescriptorBufferInfo){gmem_buf, 0, gmem_size};
    static uint32_t zero[1024];
    VkSpan s = vk_transient_copy(zero, sizeof zero, 256);
    return (VkDescriptorBufferInfo){s.buf, s.off, sizeof zero};
}

// Once a present (backend_present): the guest heap is imported, or re-imported after it grew, here rather than at the
// first draw / dispatch with global memory access - the import of ~3.2 GB takes ~300-400 ms in the driver, which
// stalled the first battle / effect that used it; at a present it happens at boot (the heap is set first thing).
void vk_gmem_prepare(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_GMEM") != NULL || !g_vk.host_import || (getenv("LARECOMP_GMEM_EARLY") && !atoi(getenv("LARECOMP_GMEM_EARLY")));
    if (off) return;
    uint64_t want = (kernel_heap_size() + 0xFFFF) & ~0xFFFFull;
    if (want > HEAP_MAX) want = HEAP_MAX;
    if (want > gmem_size && want != gmem_tried) (void)gmem_info();
}

// Guest memory ranges the GPU may write: the storage buffers a compute dispatch binds (the game's compute shaders produce vertex
// data for some draws: a 16-byte stream here, a few more there). Vertex or index data inside one of them is read in place when
// the GPU runs, as the game expects, and is never copied when the draw is recorded (the copy would hold what the GPU has not
// written yet). Sorted, merged intervals; the render thread is the only user.
static struct { uint64_t a, e; } gpuw[512];
static int ngpuw;
static void gpuw_note(uint64_t a, uint64_t n) {
    pc_dirty(a, n ? n : 16);  // (the GPU writes it: a VRAM copy of it is stale; it is read in place from now on)
    uint64_t e = a + (n ? n : 16);
    int i = 0, j;
    while (i < ngpuw && gpuw[i].e < a) i++;
    for (j = i; j < ngpuw && gpuw[j].a <= e; j++) {
        if (gpuw[j].a < a) a = gpuw[j].a;
        if (gpuw[j].e > e) e = gpuw[j].e;
    }
    if (j > i) {  // merged with [i, j)
        if (i + 1 == j && gpuw[i].a == a && gpuw[i].e == e) return;  // (known)
        gpuw[i].a = a; gpuw[i].e = e;
        memmove(&gpuw[i + 1], &gpuw[j], (size_t)(ngpuw - j) * sizeof gpuw[0]);
        ngpuw -= j - i - 1;
    } else {
        if (ngpuw == 512) { if (gpuw[ngpuw - 1].e < e) gpuw[ngpuw - 1].e = e; return; }  // (full: widen the last, stays safe)
        memmove(&gpuw[i + 1], &gpuw[i], (size_t)(ngpuw - i) * sizeof gpuw[0]);
        gpuw[i].a = a; gpuw[i].e = e;
        ngpuw++;
    }
    if (ENV("LARECOMP_LOG_GPUWRITES")) LOG("[gpuw] the GPU may write %llx-%llx (%d ranges)\n", (unsigned long long)a, (unsigned long long)e, ngpuw);
}
// A shader that stores to global memory with no storage buffer bound writes through pointers we cannot see: from then on every
// vertex / index range is read in place, as before the VRAM ring (slower, never stale).
static bool gpuw_unknown;
static void gpuw_stores(const ShaderInfo *si, const Range *sbufs, int n, const char *what, uint64_t code) {
    if (!si || !si->gmem_stores) return;
    g_gpu_stored = true;
    bool any = false;
    for (int i = 0; i < n; i++) if (sbufs[i].addr) { gpuw_note(sbufs[i].addr, sbufs[i].size); any = true; }
    if (!any && !gpuw_unknown) {
        gpuw_unknown = true;
        LOG("[vk] %s %llx stores to global memory without a storage buffer: vertex and index data is read in place from now on\n", what, (unsigned long long)code);
    }
}
// The pending submission of the last draw / dispatch whose shaders read or write guest memory in place (global memory):
// until the GPU has run it, any guest range may be in use by the GPU.
static uint64_t g_gmem_seq;
// ... and the ranges it reaches: global memory is accessed through the storage buffers bound to the stage (their
// address and size are what the shader reads from the driver constants), so a pending draw / dispatch can only touch
// those. A buffer copy used to flush whenever any such work was pending: once a frame, ~4 ms of GPU wait in a heavy
// overworld view. A stage with global memory access and no storage buffer bound sets g_gmem_any_seq (any range).
// LARECOMP_GMEM_PRECISE=0: any pending global-memory work may touch any range (the old rule).
typedef struct { uint64_t a, e, seq; } GmemRange;
#define GMR_MAX 256
static GmemRange g_gmr[GMR_MAX];
static int g_ngmr;
static uint64_t g_gmem_any_seq;
static void gmem_ranges_note(const Range *sb, int n) {
    uint64_t seq = vk_pending_seq();
    bool any = false;
    for (int i = 0; i < n; i++) {
        if (!sb[i].addr) continue;
        any = true;
        uint64_t a = sb[i].addr, e = sb[i].addr + (sb[i].size ? sb[i].size : 1);
        int k = 0;
        while (k < g_ngmr && !(g_gmr[k].a == a && g_gmr[k].e == e)) k++;
        if (k == g_ngmr) {
            if (g_ngmr == GMR_MAX) {  // (full: drop the finished ones)
                uint64_t done = vk_completed_seq();
                int m = 0;
                for (int j = 0; j < g_ngmr; j++) if (g_gmr[j].seq > done) g_gmr[m++] = g_gmr[j];
                g_ngmr = m;
            }
            if (g_ngmr == GMR_MAX) { g_gmem_any_seq = seq; continue; }
            k = g_ngmr++;
            g_gmr[k].a = a; g_gmr[k].e = e;
        }
        g_gmr[k].seq = seq;
    }
    if (!any) g_gmem_any_seq = seq;
}
static bool gmem_precise(void) {
    static int on = -1;
    if (on < 0) on = !getenv("LARECOMP_GMEM_PRECISE") || atoi(getenv("LARECOMP_GMEM_PRECISE")) != 0;
    return on;
}
static bool gpuw_overlaps(uint64_t a, uint64_t n);
// The GPU may still read or write [a, a + n) of guest memory in place: a compute / shader store range (gpuw), or work
// with global memory access not finished yet. Everything else the GPU uses was copied when its command was recorded
// (constant buffers, vertex / index data outside gpuw ranges, texture uploads), in command order on this thread.
static bool gpu_may_touch(uint64_t a, uint64_t n) {
    if (gpuw_overlaps(a, n)) return true;
    if (!g_gmem_seq) return false;
    uint64_t done = 0;
    vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &done);
    if (!gmem_precise()) return g_gmem_seq > done;
    if (g_gmem_any_seq > done) return true;
    for (int i = 0; i < g_ngmr; i++)
        if (g_gmr[i].seq > done && a < g_gmr[i].e && g_gmr[i].a < a + n) return true;
    return false;
}
static bool gpuw_overlaps(uint64_t a, uint64_t n) {
    uint64_t e = a + n;
    if (gpuw_unknown) return true;
    int lo = 0, hi = ngpuw;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (gpuw[mid].e <= a) lo = mid + 1; else hi = mid;
    }
    return lo < ngpuw && gpuw[lo].a < e;
}

// Vertex / index data read in place from the imported guest heap (renderer.m bind_guest does the same with
// g_guest_buffer): no per-draw copy, no index scan for the copy size. The game fences its CPU writes as on the
// Switch. Falls back to copies outside the heap or without the import; LARECOMP_VK_COPY_VERTICES=1 forces copies.
static bool guest_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    static int copy = -1;
    if (copy < 0) copy = getenv("LARECOMP_VK_COPY_VERTICES") != NULL;
    if (copy || addr < HEAP_BASE) return false;
    if (addr + size > HEAP_BASE + gmem_size && kernel_heap_size() != gmem_tried) { gmem_tried = kernel_heap_size(); gmem_info(); }
    if (!gmem_buf || addr + size > HEAP_BASE + gmem_size) return false;
    *buf = gmem_buf;
    *off = addr - HEAP_BASE;
    return true;
}
// Vertex / index range of a draw: in place from the game's memory (true) or to be copied (false). With the VRAM ring on, copied
// unless a compute dispatch may write it.
volatile int g_heal_tex, g_heal_vtx;  // (LARECOMP_HEAL_KEYS research keys, runtime/hle/input.c)
static bool vtx_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (g_heal_vtx) gpuw_unknown = true;
    if (ring_gpu_enabled() && !gpuw_overlaps(addr, size)) { stat_vtx_copied++; return false; }
    bool d = guest_direct(addr, size, buf, off);
    if (d) stat_vtx_direct++;
    return d;
}
bool vk_guest_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {  // (cbuf_info, above)
    return guest_direct(addr, size, buf, off);
}

// Compute programs by key, for the whole session: a table that grows (map_put; compute_program below). It was a fixed
// 256 entries searched one by one, and a program past the 256th was never kept: every dispatch of it translated and
// compiled it again (~200 ms with the frame waiting on it, and a leaked program each time).
static Map cs_tab = {.name = "compute program"};

// In-place reboot (device.c backend_reset; GPU idle): drop what is keyed by guest addresses - programs by code
// address (the old Prog / Pipe objects are leaked, so pipeline keys holding their pointers can't collide), compute
// programs, shared copies - and the heap import: its pages are replaced (re-imported on first use).
void vk_exec_reset(void) {
    map_clear(&prog_tab);
    map_clear(&pipe_tab);
    map_clear(&cs_tab);
    ngpuw = 0;  // (guest memory is replaced)
    gcopy_gen++;
    pc_reset();
    if (gmem_buf) {
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
        gmem_buf = VK_NULL_HANDLE;
        gmem_mem = VK_NULL_HANDLE;
    }
    gmem_size = gmem_tried = 0;
    st_defaults();
}

static void texture_info(int stage, ShaderInfo *si, int i, VkDescriptorImageInfo *out, float sc[2], uint32_t wh[2]) {
    ShaderTexture *st_ = &si->tex[i];
    int slot = ((int)st_->handle_offset - 0x20) / 8;  // 64-bit handles from offset 0x20
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler smp = VK_NULL_HANDLE;
    NvnTexture *tx = NULL;  // (the bound texture, for the depth-compare check below)
    uint32_t tic = h & 0xFFFFF, tsc = (uint32_t)(h >> 20) & 0xFFF;
    sc[0] = sc[1] = 1;
    wh[0] = wh[1] = 0;
    if (h && st.tpool && tic < (uint32_t)st.tpool->count) {
        NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
        tx = e->tex;
        if (e->tex) {
            // Per-frame memo: ~2400 lookups per overworld frame recomputed the view (format / range math, cache
            // probe). The first lookup of a frame still runs vk_view (upload check); a view that may have been
            // destroyed (eviction, image re-created) bumps vk_view_gen. LARECOMP_VK_NO_TEX_MEMO=1 disables.
            extern uint32_t vk_view_gen;
            extern uint64_t vk_frame_counter(void);
            static int nomemo = -1;
            if (nomemo < 0) nomemo = getenv("LARECOMP_VK_NO_TEX_MEMO") != NULL;
            static struct { NvnTexture *tex; uint64_t vh, frame; uint32_t gen; uint8_t type, has_view; VkImageView view; } memo[1024];
            uint64_t vh = e->has_view ? hash_words(0, &e->view, sizeof e->view) : 0, frame = vk_frame_counter();
            uint32_t ms = (uint32_t)((((uintptr_t)e->tex >> 4) ^ vh ^ (uint64_t)st_->type * 0x9E3779B97F4A7C15ull) & 1023);
            if (!nomemo && memo[ms].tex == e->tex && memo[ms].vh == vh && memo[ms].frame == frame && memo[ms].gen == vk_view_gen &&
                memo[ms].type == st_->type && memo[ms].has_view == e->has_view) {
                view = memo[ms].view;
            } else {
                view = vk_view(e->tex, &e->view, e->has_view, false, st_->type);
                memo[ms].tex = e->tex; memo[ms].vh = vh; memo[ms].frame = frame; memo[ms].gen = vk_view_gen;
                memo[ms].type = (uint8_t)st_->type; memo[ms].has_view = e->has_view; memo[ms].view = view;
            }
            renderer_texture_scale(e->tex, &sc[0], &sc[1]);
            wh[0] = e->tex->width; wh[1] = e->tex->height;
            for (int k = 0; k < nfeedback; k++)
                if (e->tex == feedback_tex[k] && (st_->type == TEXTYPE_2D || st_->type == TEXTYPE_2D_ARRAY)) {
                    view = feedback_view[k];
                    if (st_->type == TEXTYPE_2D_ARRAY) view = vk_dummy_view(TEXTYPE_2D_ARRAY);  // (2D snapshot as array: rare)
                }
        }
    }
    if (h && st.spool && tsc < (uint32_t)st.spool->count) {
        NvnSamplerDesc *sd = &((NvnSamplerDesc *)st.spool->entries)[tsc];
        bool int_view = h && st.tpool && tic < (uint32_t)st.tpool->count && ((NvnTexEntry *)st.tpool->entries)[tic].tex &&
                        rt_kind(((NvnTexEntry *)st.tpool->entries)[tic].tex->format);
        smp = int_view ? vk_sampler_nearest(sd) : vk_sampler(sd);  // (an integer view can't be filtered)
    }
    // a depth-compare lookup without a depth texture (the game's colour stand-in when it has no shadow map): a depth one
    // at 1.0 (vk_dummy_depth_view; a compare on a colour format is undefined in Vulkan)
    if (st_->shadow && (!view || !tx || !renderer_format_is_depth(tx->format))) view = vk_dummy_depth_view(st_->type);
    if (!view) view = vk_dummy_view(st_->type);
    if (!smp) smp = vk_default_sampler(true);
    *out = (VkDescriptorImageInfo){smp, view, VK_IMAGE_LAYOUT_GENERAL};
}

// Writes the descriptor set of one stage. gx..gz: compute group counts (driver constants).
// pre_ii: texture descriptors already resolved by the caller (draw: fragment textures, needed earlier for the
// viewport decision), or NULL.
static VkDescriptorSet stage_set(Prog *p, VkDescriptorSetLayout layout, uint32_t set, int stage, ShaderInfo *si, const float *rscale_row0,
                                 const float tint[8], uint32_t gx, uint32_t gy, uint32_t gz, const VkDescriptorImageInfo *pre_ii) {
    VkDescriptorSet ds = vk_alloc_set(layout);
    VkWriteDescriptorSet w[MAX_BIND];
    VkDescriptorBufferInfo bi[MAX_BIND];
    VkDescriptorImageInfo ii[MAX_BIND];
    int n = 0;
    int sg = stage == 1 ? 1 : 0;
    float rs[65][4] = {{1, 1, 0, 0}};
    if (rscale_row0) memcpy(rs[0], rscale_row0, 16);
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && b->binding >= 32) {
            int ti = (int)b->binding - 32;
            float sc[2]; uint32_t wh[2];
            if (ti < si->ntex && pre_ii && ti < 64) { ii[n] = pre_ii[ti]; sc[0] = tex_sc[sg][ti][0]; sc[1] = tex_sc[sg][ti][1]; wh[0] = tex_wh[sg][ti][0]; wh[1] = tex_wh[sg][ti][1]; }
            else if (ti < si->ntex) texture_info(stage, si, ti, &ii[n], sc, wh);
            else { ii[n] = (VkDescriptorImageInfo){vk_default_sampler(true), vk_dummy_view(TEXTYPE_2D), VK_IMAGE_LAYOUT_GENERAL}; sc[0] = sc[1] = 1; wh[0] = wh[1] = 0; }
            if (ti < 64) {
                tex_sc[sg][ti][0] = sc[0]; tex_sc[sg][ti][1] = sc[1];
                tex_wh[sg][ti][0] = wh[0]; tex_wh[sg][ti][1] = wh[1];
            }
            w[n].pImageInfo = &ii[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && b->binding < 17) {
            bi[n] = cbuf_info(stage, si, (int)b->binding, gx, gy, gz);
            w[n].pBufferInfo = &bi[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            bi[n] = gmem_info();
            w[n].pBufferInfo = &bi[n];
        } else {
            continue;  // rscale / tint: written below, once the textures (and their scales) are known
        }
        n++;
    }
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set || b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || (b->binding != 18 && b->binding != 19)) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->binding == 18) {
            for (int t = 0; t < si->ntex && t < 64; t++) { rs[1 + t][0] = tex_sc[sg][t][0]; rs[1 + t][1] = tex_sc[sg][t][1]; }
            bi[n] = ubo_span(copy_dedupe(sg ? DEDUPE_RSCALE_F : DEDUPE_RSCALE_V, rs, sizeof rs, 256), sizeof rs);
        } else {
            float zero[8] = {0};
            bi[n] = ubo_span(copy_dedupe(DEDUPE_TINT, tint ? tint : zero, 32, 256), 32);
        }
        w[n].pBufferInfo = &bi[n];
        n++;
    }
    if (n) vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL);
    return ds;
}

// ================================================================== compute
typedef struct { Prog p; ShaderInfo *si; VkPipeline pipe; uint32_t local[3]; } CsProg;

static void build_compute(void *arg) {
    CsProg *cp = arg;
    size_t n = 0;
    uint32_t *spv = vk_glsl_to_spirv(cp->si->msl, STAGE_COMPUTE, "cs", &n);
    if (!spv) { atomic_store(&cp->p.state, 2); return; }
    cp->p.nbinds = vk_reflect(spv, n, cp->p.binds, MAX_BIND);
    VkShaderModule m = vk_module(spv, n);
    free(spv);
    cp->p.layout = make_layout(&cp->p, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0);
    VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
    VkSpecializationInfo spec = {3, me, 12, cp->local};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main",
                                                .pSpecializationInfo = &spec},
                                      .layout = cp->p.layout};
    VkResult r = vkCreateComputePipelines(g_vk.dev, g_vk.pcache, 1, &ci, NULL, &cp->pipe);
    atomic_fetch_add(&stat_compiles, 1);
    if (r == VK_SUCCESS) atomic_fetch_add(&new_pipes, 1);
    atomic_store(&cp->p.state, r == VK_SUCCESS ? 1 : 2);
    if (r != VK_SUCCESS) LOG("[compute] pipeline failed: %d\n", r);
}

static void cs_recipe_record(uint64_t ch, uint32_t co, const uint32_t local[3]);  // (compute recipes, below)

// A compute program for this key: translated here, its pipeline built on a worker - right away for a dispatch, on the
// low-priority queue for a compute recipe's prewarm (below), which a dispatch can still claim (wait_built).
static CsProg *compute_make(uint64_t key, uint64_t code, uint32_t co, const uint32_t local[3], bool prewarm) {
    CsProg *cp = calloc(1, sizeof *cp);
    map_put(&cs_tab, key, cp, 256);
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    ShaderKey k = {0};
    k.glsl = 1;
    cp->si = maxwell_translate((const uint8_t *)(uintptr_t)code, co ? co : 0x10000, STAGE_COMPUTE, &k);
    cp->p.vi = cp->si;
    if (!cp->si->msl) {
        LOG("[compute] translate failed: %s (code %llx)\n", cp->si->error, (unsigned long long)code);
        atomic_store(&cp->p.state, 2);
        return cp;
    }
    for (int i = 0; i < 3; i++) cp->local[i] = local[i] ? local[i] : 1;
    save_glsl(cp->si->msl, code, "cs");
    if (prewarm) run_job_low(build_compute, cp);
    else run_job(build_compute, cp);
    return cp;
}

static CsProg *compute_program(NvnProgram *np) {
    NvnShaderStage *ss = &np->stages[5];
    if (!ss->code_addr) return NULL;
    // keyed by content when known: the game reuses its shader memory between areas, and a key by address returned the
    // translation of the shader that was there before (G4, the black rug: the same bug in the graphics programs)
    uint64_t ch = stage_hash_rt(ss);
    uint64_t key = ch ? ch : ss->code_addr;
    CsProg *found = map_get(&cs_tab, key);
    if (found) return found;
    uint32_t local[3];
    for (int i = 0; i < 3; i++) local[i] = ss->control ? *(const uint32_t *)(ss->control + 0x748 + 4 * i) : 1;
    CsProg *cp = compute_make(key, ss->code_addr, ss->const_offset, local, false);
    if (ch && cp->si && cp->si->msl) cs_recipe_record(ch, ss->const_offset, cp->local);
    return cp;
}

static void dispatch_compute(uint32_t gx, uint32_t gy, uint32_t gz) {
    snap_reason = "compute"; snapshots_invalidate(NULL);
    if (!st.program || !gx || !gy || !gz) return;
    CsProg *cp = compute_program(st.program);
    if (cp) wait_built(&cp->p.state, build_compute, cp);  // (a skipped dispatch leaves its results missing: hold as draws do)
    if (!cp || atomic_load(&cp->p.state) != 1) { stat_skipped++; return; }
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    gp_mark(cb, "dispatch %ux%ux%u (local %ux%ux%u)", gx, gy, gz, cp->local[0], cp->local[1], cp->local[2]);
    dl_note("dispatch %ux%ux%u cs %llx gmem %d", gx, gy, gz, (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    gpuw_stores(cp->si, st.sbuf[5], 16, "compute shader", st.program->stages[5].code_addr);
    if (cp->si->uses_gmem) { readbacks_before_gmem(); g_gmem_seq = vk_pending_seq(); gmem_ranges_note(st.sbuf[5], 16); }
    for (int i = 0; i < 16; i++) if (st.sbuf[5][i].addr) gpuw_note(st.sbuf[5][i].addr, st.sbuf[5][i].size);  // (any storage buffer it has bound)
    nfeedback = 0;
    VkDescriptorSet ds = stage_set(&cp->p, cp->p.set_layout[0], 0, 5, cp->si, NULL, NULL, gx, gy, gz, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->p.layout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, gx, gy, gz);
    stat_dispatches++;
}

// ================================================================== draws
static void debug_draw(Prog *pr, CmdRec *r, bool indexed);

// Quads / fans / polygons as a 32-bit triangle index list.
static uint32_t *expand_indices(uint32_t mode, uint32_t count, const void *indices, int itype, uint32_t first, uint32_t *out_count) {
    uint32_t tris = mode == 7 ? count / 4 * 2 : count >= 3 ? count - 2 : 0;
    uint32_t *out = malloc(tris * 3 * 4 + 4);
    uint32_t n = 0;
#define IDX(i) (indices ? (itype == 0 ? ((const uint8_t *)indices)[i] : itype == 1 ? ((const uint16_t *)indices)[i] : ((const uint32_t *)indices)[i]) : first + (i))
    if (mode == 7) {
        for (uint32_t q = 0; q + 3 < count; q += 4) {
            out[n++] = IDX(q); out[n++] = IDX(q + 1); out[n++] = IDX(q + 2);
            out[n++] = IDX(q); out[n++] = IDX(q + 2); out[n++] = IDX(q + 3);
        }
    } else {
        for (uint32_t i = 1; i + 1 < count; i++) { out[n++] = IDX(0); out[n++] = IDX(i); out[n++] = IDX(i + 1); }
    }
#undef IDX
    *out_count = n;
    return out;
}

static bool topology(uint32_t mode, VkPrimitiveTopology *t, uint8_t *cls) {
    switch (mode) {
    case 0: *t = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; *cls = 0; return true;
    case 1: *t = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; *cls = 1; return true;
    case 3: *t = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; *cls = 1; return true;
    case 4: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return true;
    case 5: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; *cls = 2; return true;
    default: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return false;
    }
}

static void draw_(CmdRec *r, bool indexed);
static void draw(CmdRec *r, bool indexed) {
    PROF_T0();
    draw_(r, indexed);
    PROF_ADD(prof_draw);
    // LARECOMP_DEBUG_PASS_DRAWS=<pass number> (with F9 / LARECOMP_DEBUG_FRAME): in the debug frame, that pass's colour
    // target after each of its draws -> build/draw_<frame>_<n>.png, with the draw's programs logged (which draw paints
    // what: docs/DEPURACAO.md). The pass is ended and begun again (loading what is there) around each save.
    static int want = -2;
    if (want == -2) want = getenv("LARECOMP_DEBUG_PASS_DRAWS") ? atoi(getenv("LARECOMP_DEBUG_PASS_DRAWS")) : -1;
    if (want < 0 || !g_debug_frame || debug_pass_no != want || !in_pass || !st.rt.ncolor || !st.rt.color[0] ||
        !st.rt.color[0]->backend)
        return;
    static uint64_t frame = ~0ull;
    static int n;
    if (renderer_present_count() != frame) { frame = renderer_present_count(); n = 0; }
    vkCmdEndRendering(vk_cmd());
    in_pass = false;
    char path[256];
    snprintf(path, sizeof path, "build/draw_%llu_%03d.png", (unsigned long long)frame, n);
    LOG("[drawdump] %s: vs %llx fs %llx (fs hash %016llx)\n", path, st.program ? (unsigned long long)st.program->stages[0].code_addr : 0ull,
        st.program ? (unsigned long long)st.program->stages[1].code_addr : 0ull, (unsigned long long)g_cur_fs_hash);
    renderer_save_texture_tex(st.rt.color[0]->backend, path);
    n++;
    st.rt_dirty = true;  // (begin_pass again before the next draw; the target is loaded, not cleared)
}
// A draw whose program or pipeline is still being built HOLDS (the render thread waits) instead of being skipped: a
// skipped draw is missing from the image (objects popping in, a black rug, a shadow map left incomplete). Every
// project that measured it rejected skipping (mk8-recomp, OpenSmash, SuperMarioWonderRecomp: 5,367 draws missing in one
// window). A job nobody has started is claimed and built right here (job_claim), so a hold lasts one build, not the
// queue. The wait has a ceiling per draw, LARECOMP_SHADER_HOLD_MS (default 3000): past it the draw is skipped as before
// and counted ("hold timeouts": should stay 0). The pipeline recipes (below) build most of them ahead, so holds are
// rare with a warm cache. LARECOMP_SHADER_HOLD_MS=0: the old mode, a budget of LARECOMP_SHADER_WAIT_MS (20) per frame,
// then skips (A/B).
static bool wait_built(_Atomic int *state, void (*fn)(void *), void *arg) {
    if (atomic_load(state) != 0) return true;
    static int64_t hold = -1, budget = -1;
    if (hold < 0) {
        hold = (getenv("LARECOMP_SHADER_HOLD_MS") ? atoi(getenv("LARECOMP_SHADER_HOLD_MS")) : 3000) * 1000000ll;
        budget = (getenv("LARECOMP_SHADER_WAIT_MS") ? atoi(getenv("LARECOMP_SHADER_WAIT_MS")) : 20) * 1000000ll;
    }
    if (sync_shaders()) return true;  // (built already)
    uint64_t t0 = SDL_GetTicksNS();
    if (hold > 0) {
        if (fn && job_claim(fn, arg)) { fn(arg); stat_hold_claims++; }
        while (atomic_load(state) == 0 && (int64_t)(SDL_GetTicksNS() - t0) < hold) SDL_DelayNS(50000);
        uint64_t dt = SDL_GetTicksNS() - t0;
        stat_holds++;
        stat_hold_ns += dt;
        if (dt > stat_hold_max_ns) stat_hold_max_ns = dt;
        stat_shader_wait_ns += dt;
        if (atomic_load(state) == 0) stat_hold_timeouts++;
        if (dt > 50000000) {  // ([hold] what a long one waited for: the hitches left once the shaders are prepared)
            const char *what = fn == build_compute ? "compute pipeline" : fn == translate_program ? "program translation"
                             : fn == link_pipeline ? "pipeline link" : fn == build_pipeline ? "whole pipeline" : "?";
            LOG("[hold] %.0f ms on a %s%s\n", dt / 1e6, what, fn && arg ? "" : " (no job)");
        }
        return atomic_load(state) != 0;
    }
    extern uint64_t vk_frame_counter(void);
    static uint64_t frame;
    static int64_t spent;
    if (vk_frame_counter() != frame) { frame = vk_frame_counter(); spent = 0; }
    if (spent >= budget) return false;
    while (atomic_load(state) == 0 && spent + (int64_t)(SDL_GetTicksNS() - t0) < budget) SDL_DelayNS(100000);
    uint64_t dt = SDL_GetTicksNS() - t0;
    spent += (int64_t)dt;
    stat_shader_wait_ns += dt;
    return atomic_load(state) != 0;
}

// ================================================================== pipeline recipes (prewarm)
// <cache>/pipeline_recipes.bin: every graphics pipeline a draw created on this PC, by value - the two stages' content
// hashes, their shader keys and the PipeKey without its Prog pointer. When the game hands a program over
// (nvnProgramSetShaders -> renderer_stage_seen, game thread), the recipes whose two stages are now both in memory are
// prewarmed on the low-priority queue: the program is translated (get_program, keyed by the stages' addresses as a draw
// keys it) and then the pipeline is built from the recipe's key through pipe_get - the table and function a draw uses.
// A draw that later meets that key adopts the pipeline. A recipe that no longer matches anything (another translator,
// a stale file) only costs prewarm work: it cannot change what a draw binds. The same design was measured in mk8-recomp
// (recipes), OpenSmash (pipes.bin) and SuperMarioWonderRecomp (.keys): docs/engineering/recomp-project-comparison.md.
// The file is append-only; each record is checked (magic, version, struct sizes, checksum) and a bad one is skipped.
// LARECOMP_PIPE_RECIPES=0: no prewarm and no recording (A/B).
#define RECIPE_MAGIC 0x50434552u  // "RECP"
#define RECIPE_VERSION 2u  // 2: fragment-less programs (chash[1] = 0)
#define RECIPE_MAX 40000
typedef struct {
    uint32_t magic, version, keysize, pipesize;
    uint64_t chash[2];
    ShaderKey vkey, fkey;
    PipeKey k;  // k.prog = NULL
    uint64_t check;
} RecipeRec;
// state: 0 waiting for its stages, 1 queued, 2 program requested, 3 done (pipeline requested, or dropped)
typedef struct { RecipeRec r; uint8_t state; Prog *prog; } Recipe;
static Recipe *g_recipes;
static int g_nrecipes, g_cap_recipes;
typedef struct { uint64_t h; int idx; } RecStage;
static RecStage *g_rec_by_stage;  // (stage hash, recipe) for both stages, sorted by hash
static int g_nrec_by_stage;
#define REC_KNOWN (1 << 17)
static uint64_t g_rec_known[REC_KNOWN];  // checksums of the recipes on file or recorded this run (open addressing)
static RecipeRec *g_rec_out;             // recorded this run, not yet appended to the file
static int g_nrec_out, g_cap_rec_out;
static int *g_rec_queue;                 // recipes whose two stages are in memory, to prewarm in order
static int g_nrec_queue, g_cap_rec_queue, g_rec_queue_at;
static uint64_t stat_recipe_stale, stat_recipe_loaded, stat_recipe_progs, stat_recipe_pipes, stat_recipe_adopted, stat_recipe_recorded;
// stages the game handed over (game thread -> render thread)
typedef struct { uint64_t h, addr; uint32_t co, size; } StageSeen;
static pthread_mutex_t g_stage_lock = PTHREAD_MUTEX_INITIALIZER;
static StageSeen *g_stage_in;
static int g_nstage_in, g_cap_stage_in;
#define STAGE_TAB (1 << 15)
static StageSeen g_stage_tab[STAGE_TAB];  // content hash -> where that blob is now (render thread)

static bool recipes_on(void) {
    static int on = -1;
    if (on < 0) on = !getenv("LARECOMP_PIPE_RECIPES") || atoi(getenv("LARECOMP_PIPE_RECIPES")) != 0;
    return on;
}

static uint64_t recipe_check(const RecipeRec *r) { return hash_words(0x5245435052454350ull, r, offsetof(RecipeRec, check)) | 1; }

static bool recipe_known_add(uint64_t c) {  // false when already known
    uint32_t i = (uint32_t)(c ^ (c >> 23)) & (REC_KNOWN - 1);
    for (int n = 0; n < REC_KNOWN; n++, i = (i + 1) & (REC_KNOWN - 1)) {
        if (g_rec_known[i] == c) return false;
        if (!g_rec_known[i]) { g_rec_known[i] = c; return true; }
    }
    return false;  // (full: treated as known)
}

static void recipe_path(char *out, size_t n) {
    extern const char *rt_cache_dir(void);
    // (the record layout in the name: a build whose keys changed size starts its own file instead of misreading)
    snprintf(out, n, "%s/pipeline_recipes_v%u_%u_%u.bin", rt_cache_dir(), RECIPE_VERSION, (unsigned)sizeof(ShaderKey), (unsigned)sizeof(PipeKey));
}

static int cmp_rec_stage(const void *a, const void *b) {
    uint64_t x = ((const RecStage *)a)->h, y = ((const RecStage *)b)->h;
    return x < y ? -1 : x > y;
}

static void recipes_load(void) {
    static bool done;
    if (done) return;
    done = true;
    char path[600];
    recipe_path(path, sizeof path);
    FILE *f = fopen(path, "rb");
    int bad = 0;
    if (f) {
        RecipeRec r;
        while (g_nrecipes < RECIPE_MAX && fread(&r, sizeof r, 1, f) == 1) {
            if (r.magic != RECIPE_MAGIC || r.version != RECIPE_VERSION || r.keysize != sizeof(ShaderKey) ||
                r.pipesize != sizeof(PipeKey) || r.check != recipe_check(&r) || !r.chash[0]) { bad++; continue; }
            if (!recipe_known_add(r.check)) continue;
            r.k.depth_clamp = r.k.depth_clamp != 0;  // (older builds recorded 15: not a VkBool32, and no draw's key matched it)
            if (g_nrecipes == g_cap_recipes) {
                g_cap_recipes = g_cap_recipes ? g_cap_recipes * 2 : 1024;
                g_recipes = realloc(g_recipes, (size_t)g_cap_recipes * sizeof *g_recipes);
            }
            memset(&g_recipes[g_nrecipes], 0, sizeof *g_recipes);
            g_recipes[g_nrecipes++].r = r;
        }
        fclose(f);
    }
    g_rec_by_stage = malloc((size_t)(2 * g_nrecipes + 1) * sizeof *g_rec_by_stage);
    for (int i = 0; i < g_nrecipes; i++)
        for (int s = 0; s < 2; s++)
            if (g_recipes[i].r.chash[s]) g_rec_by_stage[g_nrec_by_stage++] = (RecStage){g_recipes[i].r.chash[s], i};
    qsort(g_rec_by_stage, (size_t)g_nrec_by_stage, sizeof *g_rec_by_stage, cmp_rec_stage);
    stat_recipe_loaded = (uint64_t)g_nrecipes;
    LOG("[recipes] %d pipeline recipes from %s%s\n", g_nrecipes, path, bad ? " (invalid records skipped)" : "");
    extern void cs_recipes_load_ext(void);
    cs_recipes_load_ext();
}

void renderer_stage_seen(uint64_t h, uint64_t addr, uint32_t co, uint32_t size) {
    if (!recipes_on()) return;
    pthread_mutex_lock(&g_stage_lock);
    if (g_nstage_in == g_cap_stage_in) {
        g_cap_stage_in = g_cap_stage_in ? g_cap_stage_in * 2 : 256;
        g_stage_in = realloc(g_stage_in, (size_t)g_cap_stage_in * sizeof *g_stage_in);
    }
    g_stage_in[g_nstage_in++] = (StageSeen){h, addr, co, size};
    pthread_mutex_unlock(&g_stage_lock);
}

// The blob at a stage's recorded address still has its content (prewarm: the memory may have been reused since)
static bool stage_still_there(const StageSeen *e) {
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t *b = (const uint8_t *)(uintptr_t)e->addr;
    for (uint32_t k = 0; k < e->size; k++) h = (h ^ b[k]) * 0x100000001B3ull;
    return (h ? h : 1) == e->h;
}

static int stage_slot(uint64_t h) {
    uint32_t i = (uint32_t)(h ^ (h >> 29)) & (STAGE_TAB - 1);
    for (int n = 0; n < STAGE_TAB; n++, i = (i + 1) & (STAGE_TAB - 1))
        if (!g_stage_tab[i].h || g_stage_tab[i].h == h) return (int)i;
    return -1;
}

static void rec_queue_push(int idx) {
    if (g_nrec_queue == g_cap_rec_queue) {
        g_cap_rec_queue = g_cap_rec_queue ? g_cap_rec_queue * 2 : 1024;
        g_rec_queue = realloc(g_rec_queue, (size_t)g_cap_rec_queue * sizeof *g_rec_queue);
    }
    g_rec_queue[g_nrec_queue++] = idx;
}

// The pipeline with this exact key, made by a draw or by the prewarm (render thread only).
static Pipe *pipe_get(const PipeKey *k, bool prewarm, bool *created) {
    uint64_t ph = hash_words(0xcbf29ce484222325ull, k, sizeof *k);
    if (!ph) ph = 1;
    Pipe *pp = map_get(&pipe_tab, ph);
    *created = !pp;
    if (!pp) {
        pp = calloc(1, sizeof *pp);
        pp->k = *k;
        pp->prewarmed = prewarm;
        map_put(&pipe_tab, ph, pp, 8192);
        // With parts the pipeline a draw uses is linked from them, on a worker as soon as the pipeline is known (a
        // draw, or the prefetch walk of its list: linked in parallel, not one after another on this thread), and the
        // whole one follows in the background. A recipe's pipeline is linked only if a draw needs it before then.
        atomic_store(&pp->fstate, g_vk.gpl && !prewarm ? 0 : -1);
        if (g_vk.gpl && !prewarm) run_job(link_pipeline, pp);
        if (g_vk.gpl) run_job_bg(build_pipeline, pp);
        else if (prewarm) run_job_low(build_pipeline, pp);
        else run_job(build_pipeline, pp);
    }
    return pp;
}

static uint64_t stat_recipe_nohash[2], stat_recipe_dup, stat_pipes_drawn;
static void recipe_record(const Prog *pr, const PipeKey *k) {
    stat_pipes_drawn++;
    if (!recipes_on()) return;
    if (!pr->chash[0] || (!pr->chash[1] && pr->code[1])) { stat_recipe_nohash[!pr->chash[0] ? 0 : 1]++; return; }
    RecipeRec r;
    memset(&r, 0, sizeof r);
    r.magic = RECIPE_MAGIC;
    r.version = RECIPE_VERSION;
    r.keysize = sizeof(ShaderKey);
    r.pipesize = sizeof(PipeKey);
    r.chash[0] = pr->chash[0];
    r.chash[1] = pr->chash[1];
    r.vkey = pr->vkey;
    r.fkey = pr->fkey;
    r.k = *k;
    r.k.prog = NULL;
    r.check = recipe_check(&r);
    if (!recipe_known_add(r.check)) { stat_recipe_dup++; return; }
    if (g_nrec_out == g_cap_rec_out) {
        g_cap_rec_out = g_cap_rec_out ? g_cap_rec_out * 2 : 256;
        g_rec_out = realloc(g_rec_out, (size_t)g_cap_rec_out * sizeof *g_rec_out);
    }
    g_rec_out[g_nrec_out++] = r;
    stat_recipe_recorded++;
}

// ---- compute recipes: <cache>/compute_recipes_v1.bin, every compute program a dispatch built on this PC - its code's
// content hash, its constants' offset and its group size (from the program's control data, which the shader
// preparation can't know: the reason compute stages were the hitches left after it, ~230 ms each, once per driver).
// When the game hands that code over again (renderer_stage_seen), the pipeline is built on the low-priority queue
// before the first dispatch needs it. Same checks as the graphics recipes; LARECOMP_PIPE_RECIPES=0 turns both off.
#define CS_RECIPE_MAGIC 0x50434352u  // "RCCP"
typedef struct { uint32_t magic, version; uint64_t chash; uint32_t co, local[3]; uint64_t check; } CsRecipe;
static CsRecipe *g_cs_rec;      // on file, sorted by hash
static int g_ncs_rec;
static CsRecipe *g_cs_rec_out;  // recorded this run, not yet appended
static int g_ncs_rec_out, g_cap_cs_rec_out;
static uint64_t stat_cs_prewarmed;

static uint64_t cs_recipe_check(const CsRecipe *r) { return hash_words(0x5243435052434350ull, r, offsetof(CsRecipe, check)) | 1; }

static void cs_recipe_path(char *out, size_t n) {
    extern const char *rt_cache_dir(void);
    snprintf(out, n, "%s/compute_recipes_v1.bin", rt_cache_dir());
}

static int cmp_cs_rec(const void *a, const void *b) {
    uint64_t x = ((const CsRecipe *)a)->chash, y = ((const CsRecipe *)b)->chash;
    return x < y ? -1 : x > y;
}

void cs_recipes_load_ext(void);  // (recipes_load, above, loads these too)
static void cs_recipes_load(void);
void cs_recipes_load_ext(void) { cs_recipes_load(); }

static void cs_recipes_load(void) {
    char path[600];
    cs_recipe_path(path, sizeof path);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    CsRecipe r;
    int cap = 0;
    while (fread(&r, sizeof r, 1, f) == 1) {
        if (r.magic != CS_RECIPE_MAGIC || r.version != 1 || r.check != cs_recipe_check(&r) || !r.chash) continue;
        if (!recipe_known_add(r.check)) continue;
        if (g_ncs_rec == cap) { cap = cap ? cap * 2 : 256; g_cs_rec = realloc(g_cs_rec, (size_t)cap * sizeof *g_cs_rec); }
        g_cs_rec[g_ncs_rec++] = r;
    }
    fclose(f);
    qsort(g_cs_rec, (size_t)g_ncs_rec, sizeof *g_cs_rec, cmp_cs_rec);
    LOG("[recipes] %d compute recipes from %s\n", g_ncs_rec, path);
}

static void cs_recipe_record(uint64_t ch, uint32_t co, const uint32_t local[3]) {
    if (!recipes_on()) return;
    CsRecipe r;
    memset(&r, 0, sizeof r);
    r.magic = CS_RECIPE_MAGIC;
    r.version = 1;
    r.chash = ch;
    r.co = co;
    memcpy(r.local, local, sizeof r.local);
    r.check = cs_recipe_check(&r);
    if (!recipe_known_add(r.check)) return;
    if (g_ncs_rec_out == g_cap_cs_rec_out) {
        g_cap_cs_rec_out = g_cap_cs_rec_out ? g_cap_cs_rec_out * 2 : 64;
        g_cs_rec_out = realloc(g_cs_rec_out, (size_t)g_cap_cs_rec_out * sizeof *g_cs_rec_out);
    }
    g_cs_rec_out[g_ncs_rec_out++] = r;
}

// A stage the game handed over: its compute recipe, if any, prewarmed (render thread, from recipes_pump)
static void cs_recipe_seen(const StageSeen *s) {
    int lo = 0, hi = g_ncs_rec;
    while (lo < hi) { int m = (lo + hi) / 2; if (g_cs_rec[m].chash < s->h) lo = m + 1; else hi = m; }
    if (lo >= g_ncs_rec || g_cs_rec[lo].chash != s->h || map_get(&cs_tab, s->h)) return;
    if (!stage_still_there(s)) return;
    compute_make(s->h, s->addr, g_cs_rec[lo].co, g_cs_rec[lo].local, true);
    stat_cs_prewarmed++;
}

static void recipes_flush(void) {  // (with the stats line, every ~2 s: append what this run recorded)
    if (g_ncs_rec_out) {
        char path[600];
        cs_recipe_path(path, sizeof path);
        FILE *f = fopen(path, "ab");
        if (f) {
            fwrite(g_cs_rec_out, sizeof *g_cs_rec_out, (size_t)g_ncs_rec_out, f);
            fclose(f);
        }
        g_ncs_rec_out = 0;
    }
    if (!g_nrec_out) return;
    char path[600];
    recipe_path(path, sizeof path);
    FILE *f = fopen(path, "ab");
    if (f) {
        fwrite(g_rec_out, sizeof *g_rec_out, (size_t)g_nrec_out, f);
        fclose(f);
    }
    g_nrec_out = 0;
}

// Render thread, once per submitted list: take the stages the game handed over, queue the recipes they complete,
// and advance a bounded number of queued recipes (the program first, the pipeline once the program is ready).
static void recipes_pump(void) {
    if (!recipes_on()) return;
    recipes_load();
    if (!g_nrecipes && !g_ncs_rec) return;
    StageSeen in[256];
    pthread_mutex_lock(&g_stage_lock);
    int n = g_nstage_in < 256 ? g_nstage_in : 256;
    memcpy(in, g_stage_in, (size_t)n * sizeof *in);
    memmove(g_stage_in, g_stage_in + n, (size_t)(g_nstage_in - n) * sizeof *in);
    g_nstage_in -= n;
    pthread_mutex_unlock(&g_stage_lock);
    for (int i = 0; i < n; i++) {
        int sl = stage_slot(in[i].h);
        if (sl < 0) continue;
        bool fresh = !g_stage_tab[sl].h;
        g_stage_tab[sl] = in[i];
        if (!fresh) continue;  // (seen before: its recipes were queued then)
        if (g_ncs_rec) cs_recipe_seen(&in[i]);  // (a compute stage: its pipeline ahead of the first dispatch)
        int lo = 0, hi = g_nrec_by_stage;  // first entry with this hash
        while (lo < hi) { int m = (lo + hi) / 2; if (g_rec_by_stage[m].h < in[i].h) lo = m + 1; else hi = m; }
        for (int e = lo; e < g_nrec_by_stage && g_rec_by_stage[e].h == in[i].h; e++) {
            Recipe *r = &g_recipes[g_rec_by_stage[e].idx];
            if (r->state) continue;
            int s0 = stage_slot(r->r.chash[0]), s1 = r->r.chash[1] ? stage_slot(r->r.chash[1]) : 0;
            if (s0 < 0 || s1 < 0 || !g_stage_tab[s0].h || (r->r.chash[1] && !g_stage_tab[s1].h)) continue;
            r->state = 1;
            rec_queue_push(g_rec_by_stage[e].idx);
        }
    }
    // advance: at most 48 steps per list (a step queues a job: microseconds)
    int steps = 0;
    bool prefix_done = true;
    for (int q = g_rec_queue_at; q < g_nrec_queue && steps < 48; q++) {
        Recipe *r = &g_recipes[g_rec_queue[q]];
        if (r->state == 1) {
            int s0 = stage_slot(r->r.chash[0]), s1 = r->r.chash[1] ? stage_slot(r->r.chash[1]) : -1;
            NvnProgram np;
            memset(&np, 0, sizeof np);
            np.stage_mask = s1 >= 0 ? 3 : 1;
            np.stages[0].stage = 0, np.stages[0].code_addr = g_stage_tab[s0].addr, np.stages[0].const_offset = g_stage_tab[s0].co;
            np.stages[0].content_hash = r->r.chash[0];
            np.stages[0].code_size = g_stage_tab[s0].size;
            if (s1 >= 0) {
                np.stages[1].stage = 1, np.stages[1].code_addr = g_stage_tab[s1].addr, np.stages[1].const_offset = g_stage_tab[s1].co;
                np.stages[1].content_hash = r->r.chash[1];
                np.stages[1].code_size = g_stage_tab[s1].size;
            }
            if (!stage_still_there(&g_stage_tab[s0]) || (s1 >= 0 && !stage_still_there(&g_stage_tab[s1]))) {
                r->state = 3;  // (moved or freed since: no prewarm; a draw that needs it translates it as before)
                stat_recipe_stale++;
                continue;
            }
            ShaderKey vk = r->r.vkey, fk = r->r.fkey;
            r->prog = get_program_q(&np, &vk, &fk, true);
            r->state = r->prog ? 2 : 3;
            stat_recipe_progs++;
            steps++;
        }
        if (r->state == 2) {
            int ps = atomic_load(&r->prog->state);
            if (ps == 1) {
                PipeKey k = r->r.k;
                k.prog = r->prog;
                bool created;
                pipe_get(&k, true, &created);
                if (created) stat_recipe_pipes++;
                steps++;
                r->state = 3;
            } else if (ps > 1) {
                r->state = 3;  // (the program failed: nothing to build)
            }
        }
        if (r->state == 3 && prefix_done) g_rec_queue_at = q + 1;  // (the done prefix is not looked at again)
        else prefix_done = false;
    }
}

void recipes_stats(char *out, size_t n) {
    recipes_flush();
    if (!recipes_on()) return;
    if (stat_cs_prewarmed) {  // (compute recipes prewarmed so far)
        LOG("[recipes] %llu compute pipelines prewarmed\n", (unsigned long long)stat_cs_prewarmed);
        stat_cs_prewarmed = 0;
    }
    snprintf(out, n, " | recipes %llu: progs %llu pipes %llu adopted %llu new %llu (draw pipes %llu, no vs hash %llu, no fs hash %llu, dup %llu, stale %llu)",
             (unsigned long long)stat_recipe_loaded, (unsigned long long)stat_recipe_progs, (unsigned long long)stat_recipe_pipes,
             (unsigned long long)stat_recipe_adopted, (unsigned long long)stat_recipe_recorded, (unsigned long long)stat_pipes_drawn,
             (unsigned long long)stat_recipe_nohash[0], (unsigned long long)stat_recipe_nohash[1], (unsigned long long)stat_recipe_dup, (unsigned long long)stat_recipe_stale);
    stat_pipes_drawn = stat_recipe_nohash[0] = stat_recipe_nohash[1] = stat_recipe_dup = stat_recipe_stale = 0;
    size_t l = strlen(out);
    snprintf(out + l, n - l, " | shader hashes %llu, dropped %llu", (unsigned long long)stat_sh_hashed, (unsigned long long)stat_sh_dropped);
    stat_sh_hashed = stat_sh_dropped = 0;
    stat_recipe_progs = stat_recipe_pipes = stat_recipe_adopted = stat_recipe_recorded = 0;
}

// The shader keys a draw's program is translated for (vertex attribute kinds, target kinds, window origin...).
static void draw_shader_keys(ShaderKey *vkey, ShaderKey *fkey) {
    memset(vkey, 0, sizeof *vkey);
    memset(fkey, 0, sizeof *fkey);
    vkey->glsl = fkey->glsl = 1;
    vkey->flip_y = fkey->flip_y = nvn_window_origin() == 0;
    vkey->depth_minus_one = nvn_depth_mode() == 0;
    for (int i = 0; i < st.nattrib; i++) vkey->attr_kind[i] = (uint8_t)attr_kind(st.attrib[i] & 0xFF);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) fkey->rt_kind[i] = (uint8_t)rt_kind(st.rt.color[i]->format);
}

// Textures bound to integer formats: the program's variant that declares them so (ShaderKey.tex_uint / tex_sint).
// True when the keys changed (the program has to be asked for again).
static bool draw_texint_keys(Prog *pr, ShaderKey *vkey, ShaderKey *fkey) {
    static int texint = -1;
    if (texint < 0) texint = !getenv("LARECOMP_TEXINT") || atoi(getenv("LARECOMP_TEXINT")) != 0;
    if (!texint) return false;
    uint32_t m[2][2] = {{0, 0}, {0, 0}};  // [stage][uint, sint]
    for (int sgi = 0; sgi < 2; sgi++) {
        ShaderInfo *si = sgi ? pr->fi : pr->vi;
        for (int i = 0; si && i < si->ntex && i < 32; i++) {
            if (si->tex[i].shadow) continue;
            NvnTexture *t = shader_texture_object(sgi, &si->tex[i]);
            int kind = t ? rt_kind(t->format) : 0;
            if (kind) m[sgi][kind == 2 ? 0 : 1] |= 1u << i;
        }
    }
    if (m[0][0] == vkey->tex_uint && m[0][1] == vkey->tex_sint && m[1][0] == fkey->tex_uint && m[1][1] == fkey->tex_sint) return false;
    vkey->tex_uint = m[0][0], vkey->tex_sint = m[0][1], fkey->tex_uint = m[1][0], fkey->tex_sint = m[1][1];
    return true;
}

// The pipeline key of a draw of program pr into a pass with these formats.
static void pipe_key_fill(PipeKey *k, Prog *pr, uint8_t cls, uint32_t ncolor, const VkFormat *color, VkFormat depth, VkFormat stencil,
                          uint32_t samples) {
    memset(k, 0, sizeof *k);
    k->prog = pr;
    k->ncolor = ncolor;
    // (only the pass's own slots: the ones past them keep what the last pass with more targets set, so a depth-only
    // pass's pipelines got several keys and were built again for each - LAS Recomp 3921201)
    for (uint32_t i = 0; i < ncolor && i < 8; i++) k->color[i] = color[i];
    k->depth = depth;
    k->stencil = stencil;
    k->samples = samples;
    for (int i = 0; i < 32; i++) {
        if (!(pr->vi->attr_mask & (1u << i))) continue;
        k->attr_mask |= 1u << i;
        uint32_t a = i < st.nattrib ? st.attrib[i] : 0x2E;
        uint32_t fmt = a & 0xFF, stream = (a >> 8) & 0x1F, off = a >> 13;
        if (i >= st.nattrib || fmt == 0 || stream >= 16) {  // unbound attribute: (0, 0, 0, 1) from a constant stream
            k->attr_fmt[i] = VK_FORMAT_R32G32B32A32_SFLOAT;
            k->attr_bind[i] = ZERO_BINDING;
            k->used_streams |= 1u << ZERO_BINDING;
            continue;
        }
        k->attr_fmt[i] = vertex_format_supported(vertex_format(fmt));
        k->attr_off[i] = off;
        k->attr_bind[i] = stream;
        k->used_streams |= 1u << stream;
    }
    for (int s = 0; s < 16; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        k->stride[s] = s < st.nstream ? st.stream[s][0] : 0;
        k->divisor[s] = s < st.nstream && k->stride[s] ? st.stream[s][1] : 0;
        if (k->divisor[s] > 1 && !g_vk.divisor) k->divisor[s] = 1;
    }
    for (uint32_t i = 0; i < k->ncolor && i < 8; i++) {
        if (!st.rt.color[i] || k->color[i] == VK_FORMAT_UNDEFINED) continue;
        // Targets the fragment shader doesn't output keep their contents on Metal, but are undefined in Vulkan
        // (NVIDIA writes black): mask them. Foliage depth pre-passes left black silhouettes (user 2026-10-08).
        k->write_mask[i] = ((pr->fi->out_mask >> (i * 4)) & 0xF) ? (st.channel_mask >> (i * 4)) & 0xF : 0;
        if (((st.color_state[0] >> i) & 1) && !rt_kind(st.rt.color[i]->format)) {
            k->blend_on |= (uint8_t)(1u << i);
            memcpy(k->blend[i], st.blend[i], 8);
        }
    }
    k->a2c = st.ms[0xC] != 0;
    k->topo_class = cls;
    k->poly_fill = st.poly[2] != 1;
    k->depth_clamp = st.depth_clamp && g_vk.depth_clamp;
}

// ================================================================== prefetch of a list's programs / pipelines
// A first visit (a town: ~350 new programs and pipelines at once) held each draw while its own program and pipeline
// were built - one at a time on this thread, the 16 workers idle, since the next draw's needs were only known after
// it (8 s at a town's entry with a cold driver cache). At the first draw of a list that has to wait, the rest of the
// list is walked once without recording anything: state commands only (no copies, clears, barriers, syncs...), and
// each draw's program is asked for and, once it is ready, its pipeline - the same keys as the draw itself computes,
// so the builds run on the workers in parallel and the draws then find them built or building. Pass formats are read
// from the render targets' existing images (a target without one yet: that draw is left to the draw itself). A
// wrong guess only builds a pipeline nobody uses; what a draw binds is decided by the draw. LARECOMP_PREFETCH=0: off.
static int g_prefetch;               // > 0: exec_list is walking a list for its needs (no recording, no side effects)
static bool g_pf_progs_pending;      // a walk met programs still being translated: walk again once they are
static Prog *g_pf_wait[1024];        // ... those programs
static int g_pf_nwait;
static uint64_t g_exec_top;          // the submitted (depth 0) list being executed
int rt_pending_lists(uint64_t *out, int max);  // renderthread.c: the lists queued after it
static uint64_t g_pf_list_epoch, g_pf_done_epoch;  // one prefetch per submitted list
static uint64_t stat_pf_walks, stat_pf_pipes, stat_pf_progs, stat_pf_ns;
static void exec_list(uint64_t handle, int depth);

static bool pf_on(void) {
    static int on = -1;
    if (on < 0) on = !getenv("LARECOMP_PREFETCH") || atoi(getenv("LARECOMP_PREFETCH")) != 0;
    return on && !sync_shaders();
}

// The commands a walk applies (the state the draws' keys come from); every other one is skipped.
static bool pf_applies(uint32_t op) {
    switch (op) {
    case CMD_END: case CMD_JUMP: case CMD_CALL: case CMD_OBJECT: case CMD_SET_DEPTH_CLAMP:
    case CMD_BIND_PROGRAM: case CMD_BIND_BLEND: case CMD_BIND_CHANNEL_MASK: case CMD_BIND_COLOR: case CMD_BIND_DEPTH_STENCIL:
    case CMD_BIND_MULTISAMPLE: case CMD_BIND_POLYGON: case CMD_BIND_VERTEX_ATTRIB: case CMD_BIND_VERTEX_STREAM:
    case CMD_BIND_VERTEX_BUFFERS: case CMD_BIND_UNIFORM_BUFFERS: case CMD_BIND_STORAGE_BUFFERS: case CMD_BIND_TEXTURES:
    case CMD_BIND_IMAGES: case CMD_SET_TEXTURE_POOL: case CMD_SET_SAMPLER_POOL: case CMD_SET_RENDER_TARGETS:
    case CMD_SET_PRIMITIVE_RESTART: case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS:
        return true;
    default:
        return false;
    }
}

// The pass formats begin_pass would use for the bound targets: from their images, or for a target without one yet
// the format its image will get (textures.c: the NVN format's Vulkan format, its sample count).
static bool pf_pass_formats(uint32_t *ncolor, VkFormat color[8], VkFormat *depth, VkFormat *stencil, uint32_t *samples) {
    uint32_t n = (uint32_t)st.rt.ncolor, last = 0;
    *samples = 1;
    for (uint32_t i = 0; i < n && i < 8; i++) {
        NvnTexture *t = st.rt.color[i];
        color[i] = VK_FORMAT_UNDEFINED;
        if (!t) continue;
        TexBackend *b = t->backend;
        VkFormat bf = b ? b->format : vk_format(t->format);
        color[i] = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4) && vk_format(t->format) == bf
                 ? vk_format(st.rt.cview[i].format) : bf;
        *samples = b ? b->samples : t->samples > 1 ? t->samples : 1;
        last = i + 1;
    }
    *ncolor = last;
    *depth = *stencil = VK_FORMAT_UNDEFINED;
    if (st.rt.depth) {
        TexBackend *b = st.rt.depth->backend;
        if (renderer_format_is_depth(st.rt.depth->format)) {
            VkFormat df = b ? b->format : vk_format(st.rt.depth->format);
            bool has_s = b ? (b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0
                           : df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_D16_UNORM_S8_UINT ||
                             df == VK_FORMAT_S8_UINT;
            bool has_d = b ? (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 : df != VK_FORMAT_S8_UINT;
            if (has_d) *depth = df;
            if (has_s) *stencil = df;
            if (!last) *samples = b ? b->samples : st.rt.depth->samples > 1 ? st.rt.depth->samples : 1;
        }
    }
    return true;
}

static void pf_draw(CmdRec *r) {
    if (!st.program) return;
    ShaderKey vkey, fkey;
    draw_shader_keys(&vkey, &fkey);
    Prog *pr = get_program(st.program, &vkey, &fkey);
    if (!pr) return;
    if (atomic_load(&pr->state) == 0) { g_pf_progs_pending = true; if (g_pf_nwait < 1024) g_pf_wait[g_pf_nwait++] = pr; return; }
    if (atomic_load(&pr->state) != 1) return;
    if (draw_texint_keys(pr, &vkey, &fkey)) {
        pr = get_program(st.program, &vkey, &fkey);
        if (!pr) return;
        if (atomic_load(&pr->state) == 0) { g_pf_progs_pending = true; if (g_pf_nwait < 1024) g_pf_wait[g_pf_nwait++] = pr; return; }
        if (atomic_load(&pr->state) != 1) return;
    }
    VkPrimitiveTopology topo;
    uint8_t cls;
    uint32_t mode = (uint32_t)r->x[0];
    if (!topology(mode, &topo, &cls) && mode != 6 && mode != 7 && mode != 9) return;
    uint32_t ncolor, samples;
    VkFormat color[8], depth, stencil;
    if (!pf_pass_formats(&ncolor, color, &depth, &stencil, &samples)) return;
    PipeKey k;
    pipe_key_fill(&k, pr, cls, ncolor, color, depth, stencil, samples);
    bool created;
    pipe_get(&k, false, &created);
    if (created) stat_pf_pipes++;
}

// Called by a draw that is about to wait for its program or pipeline: walk the rest of the list once (twice when
// programs were still translating: the second walk asks for their pipelines).
static void pf_rest_of_list(void) {
    if (g_prefetch || !pf_on() || g_pf_done_epoch == g_pf_list_epoch || !g_exec_top) return;
    g_pf_done_epoch = g_pf_list_epoch;
    uint64_t t0 = SDL_GetTicksNS();
    static __typeof__(st) saved;
    static NvnProgram saved_comb;
    saved = st;
    saved_comb = g_comb_prog;
    // this whole list (its draws sit in called lists: the part already run only finds what exists) and the lists queued
    // after it (the rest of the frame); each walk starts from the state at this point
    static uint64_t lists[64];
    lists[0] = g_exec_top;
    static uint64_t queued_frame = ~0ull;  // (the queued lists: once a frame)
    bool queued = queued_frame != vk_frame_counter() && rt_active() && rt_on_render_thread();
    if (queued) queued_frame = vk_frame_counter();
    int nl = 1 + (queued ? rt_pending_lists(lists + 1, 63) : 0);
    for (int walk = 0; walk < 3; walk++) {
        g_pf_progs_pending = false;
        g_pf_nwait = 0;
        g_prefetch++;
        for (int l = 0; l < nl; l++) {
            st = saved;
            g_comb_prog = saved_comb;
            exec_list(lists[l], 0);
        }
        g_prefetch--;
        stat_pf_walks++;
        if (!g_pf_progs_pending) break;
        // the programs this walk asked for translate on the workers in parallel: wait (bounded) for them, taking the
        // ones nobody has started (job_claim) to translate here meanwhile
        stat_pf_progs += (uint64_t)g_pf_nwait;
        uint64_t tw = SDL_GetTicksNS();
        for (int i = 0; i < g_pf_nwait; i++)
            if (atomic_load(&g_pf_wait[i]->state) == 0 && job_claim(translate_program, g_pf_wait[i])) translate_program(g_pf_wait[i]);
        for (int i = 0; i < g_pf_nwait && SDL_GetTicksNS() - tw < 1500000000ull;)
            if (atomic_load(&g_pf_wait[i]->state) == 0) SDL_DelayNS(200000);
            else i++;
    }
    st = saved;
    g_comb_prog = saved_comb;
    stat_pf_ns += SDL_GetTicksNS() - t0;
}

static uint64_t prof_tp_;
static void taa_capture_camera(const float *p, uint32_t off, uint64_t size);
// ================================================================== SSAO (Let's Go Recomp's, non-temporal)
// A GL-style perspective matrix at m (row-major: the rows of P; column-major: its columns): P00, |P11|, A, B.
static bool ssao_proj_at(const float *m, float out[4]) {
    if (m[1] == 0 && m[3] == 0 && m[4] == 0 && m[7] == 0 && m[12] == 0 && m[13] == 0 && m[14] == -1 && m[15] == 0 &&
        m[0] > 0 && m[5] != 0 && m[10] < 0 && m[11] < 0) {
        out[0] = m[0], out[1] = fabsf(m[5]), out[2] = m[10], out[3] = m[11];
        return true;
    }
    if (m[1] == 0 && m[2] == 0 && m[3] == 0 && m[4] == 0 && m[6] == 0 && m[7] == 0 && m[11] == -1 && m[12] == 0 && m[13] == 0 &&
        m[15] == 0 && m[0] > 0 && m[5] != 0 && m[10] < 0 && m[14] < 0) {
        out[0] = m[0], out[1] = fabsf(m[5]), out[2] = m[10], out[3] = m[14];
        return true;
    }
    return false;
}

// A camera whose view (the 64 bytes before the projection, row-major) does not move the world: a sky's. Not the scene's.
static bool ssao_cam_at_origin(const float *p, uint32_t off) {
    if (off < 64) return false;
    const float *v = p - 16;
    return fabsf(v[3]) < 1 && fabsf(v[7]) < 1 && fabsf(v[11]) < 1;
}

// The scene's projection, from the vertex constant buffers of its opaque draws: where it was last found, else a scan
// (once per frame).
static void ssao_find_projection(void) {
    uint64_t fr = vk_frame_counter();
    if (ao.proj_frame == fr) return;
    float proj[4];
    if (ao.pbind >= 0) {
        Range u = st.ubuf[0][ao.pbind];
        const uint8_t *blk = (const uint8_t *)(uintptr_t)u.addr;
        if (blk && ao.poff + 64 <= u.size && ssao_proj_at((const float *)(blk + ao.poff), proj) &&
            !ssao_cam_at_origin((const float *)(blk + ao.poff), ao.poff)) {
            memcpy(ao.proj, proj, sizeof ao.proj);
            ao.proj_frame = fr;
            taa_capture_camera((const float *)(blk + ao.poff), ao.poff, u.size);
            return;
        }
    }
    if (ao.scan_frame == fr) return;
    ao.scan_frame = fr;
    for (int b = 0; b < 18; b++) {
        Range u = st.ubuf[0][b];
        if (!u.addr || u.size < 64) continue;
        uint64_t n = u.size > 4096 ? 4096 : u.size;
        for (uint32_t o = 0; o + 64 <= n; o += 16)
            if (ssao_proj_at((const float *)(uintptr_t)(u.addr + o), proj) && !ssao_cam_at_origin((const float *)(uintptr_t)(u.addr + o), o)) {
                if (ao.pbind != b || ao.poff != o)
                    LOG("[ssao] projection in vertex cbuf %d at %u: near %.4g far %.4g (P00 %.3f P11 %.3f)\n", b, o, proj[3] / (proj[2] - 1),
                        proj[3] / (proj[2] + 1), proj[0], proj[1]);
                memcpy(ao.proj, proj, sizeof ao.proj);
                ao.pbind = b, ao.poff = o, ao.proj_frame = fr;
                taa_capture_camera((const float *)(uintptr_t)(u.addr + o), o, u.size);
                return;
            }
    }
}

// The SSAO's passes: a fullscreen triangle into one colour target, up to two inputs (nearest, clamp), 48 bytes of
// push constants. blend 1: the output multiplies the target's RGB (the occlusion onto the scene).
static VkPipelineLayout ssao_layout;
static VkDescriptorSetLayout ssao_set_layout;

static VkPipeline ssao_pipe(const char *fs, VkFormat fmt, int blend) {
    static struct { const char *fs; VkFormat fmt; int blend; VkPipeline p; } cache[16];
    for (int i = 0; i < 16; i++) if (cache[i].p && cache[i].fs == fs && cache[i].fmt == fmt && cache[i].blend == blend) return cache[i].p;
    if (!ssao_layout) {
        VkDescriptorSetLayoutBinding b[4];
        for (uint32_t i = 0; i < 4; i++)
            b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &ssao_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &ssao_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &ssao_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("post.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fs), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    if (blend == 1)
        cba = (VkPipelineColorBlendAttachmentState){.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                    .dstColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR, .colorBlendOp = VK_BLEND_OP_ADD,
                                                    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                    .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0x7};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = ssao_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 16; i++) if (!cache[i].p) { cache[i].fs = fs; cache[i].fmt = fmt; cache[i].blend = blend; cache[i].p = p; break; }
    return p;
}

// The effects' image i at w x h in format f, kept until the size or the format changes. 0, 1: the SSAO's occlusion and
// its blur's ping-pong; 2: a dump; 3: TAA's motion vectors (RGBA16F); 4, 5: TAA's history (RGBA16F); 6: the depth as
// R32F (DLSS); 7: DLSS's output (RGBA16F)
static TexBackend *fx_image(int i, uint32_t w, uint32_t h, VkFormat f) {
    static TexBackend img[8];
    TexBackend *d = &img[i];
    if (d->image && (d->width != w || d->height != h || d->format != f)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = f, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         (f == VK_FORMAT_R16G16B16A16_SFLOAT ? VK_IMAGE_USAGE_STORAGE_BIT : 0)};  // (DLSS writes it)
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

#define ssao_image(i, w, h) fx_image(i, w, h, VK_FORMAT_R8G8B8A8_UNORM)

static VkImageView ssao_view(TexBackend *d) {
    static const VkComponentMapping id = {0};
    return vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// One SSAO pass (outside any rendering): srcs -> the view dv (dw x dh); load: keep its contents (the apply's blending)
static void ssao_pass(VkCommandBuffer cb, VkPipeline pipe, const VkImageView *srcs, int n, VkImageView dv, uint32_t dw, uint32_t dh,
                      const float *pc, uint32_t pcsize, unsigned linear, bool load) {
    vk_full_barrier(cb);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = dv,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {dw, dh}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkDescriptorSet ds = vk_alloc_set(ssao_set_layout);
    VkDescriptorImageInfo ii[4];
    VkWriteDescriptorSet w[4];
    for (int i = 0; i < 4; i++) {  // (all bindings written: an unused one gets the first input; linear: bit i, filtered)
        ii[i] = (VkDescriptorImageInfo){vk_default_sampler(i < n && (linear >> i & 1)), srcs[i < n ? i : 0], VK_IMAGE_LAYOUT_GENERAL};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    vkUpdateDescriptorSets(g_vk.dev, 4, w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, ssao_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)dw, (float)dh, 0, 1};
    VkRect2D sc = {{0, 0}, {dw, dh}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdPushConstants(cb, ssao_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, pcsize, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    vk_full_barrier(cb);
}

static float ssao_env(const char *name, float def) {
    const char *e = getenv(name);
    return e && *e ? (float)atof(e) : def;
}

// The occlusion of this frame's scene (its depth, the projection found by its draws), blurred along x and y with
// the depth as the edges, multiplied into the HDR colour. Outside any pass (the caller ended the scene's).
static void ssao_run(void) {
    uint64_t fr = vk_frame_counter();
    ao.done_frame = fr;
    TexBackend *d = ao.dtex ? ao.dtex->backend : NULL;
    static int nolog;
    if (ao.proj_frame != fr || !d || !d->image || !(d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || d->samples > 1) {
        if (nolog < 5 && ao.proj_frame != fr) nolog++, LOG("[ssao] frame %llu: no projection found in the scene's %u opaque draws\n",
                                                           (unsigned long long)fr, ao.opaque);
        return;
    }
    ao.busy = true;
    VkCommandBuffer cb = vk_cmd();
    if (g_gpuprof_frame) gp_mark(cb, "ours: SSAO");
    static const VkComponentMapping id = {0};
    VkImageView dv = vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_DEPTH_BIT);
    uint32_t w = d->width, h = d->height;
    float mo = nvn_depth_mode() == 0 ? 1.0f : 0.0f;
    VkPipeline p_ao = ssao_pipe("ssao.frag", VK_FORMAT_R8G8B8A8_UNORM, 0), p_blur = ssao_pipe("ssao_blur.frag", VK_FORMAT_R8G8B8A8_UNORM, 0);
    VkPipeline p_apply = ssao_pipe("ssao_apply.frag", ao.cfmt, 1);
    TexBackend *a0 = ssao_image(0, w, h), *a1 = ssao_image(1, w, h);
    static float radius = -1, kint = -1;
    if (radius < 0) radius = ssao_env("LARECOMP_SSAO_RADIUS", g_settings.ssao_radius), kint = ssao_env("LARECOMP_SSAO_INTENSITY", 1.0f);
    // the viewport (target pixels) in the depth's pixels (the same size here; scaled in case they differ)
    float sx = (float)w / (float)(ao.cw ? ao.cw : w), sy = (float)h / (float)(ao.ch ? ao.ch : h);
    float vx = ao.vp[0] * sx, vy = ao.vp[1] * sy, vw = ao.vp[2] * sx, vh = ao.vp[3] * sy;
    if (vw < 16 || vh < 16) vx = vy = 0, vw = (float)w, vh = (float)h;
    float pc[16] = {ao.proj[0], ao.proj[1], ao.proj[2], ao.proj[3], radius, g_settings.ssao_intensity * kint, mo,
                    0.08f * vh, 0, (float)g_settings.ssao_samples, 0, 0, vx, vy, vw, vh};
    ssao_pass(cb, p_ao, &dv, 1, ssao_view(a0), w, h, pc, 64, 0, false);
    float pb[16] = {ao.proj[0], ao.proj[1], ao.proj[2], ao.proj[3], 1, 0, mo, 0, 0, 0, 0, 0, vx, vy, vw, vh};
    VkImageView bx[2] = {ssao_view(a0), dv};
    ssao_pass(cb, p_blur, bx, 2, ssao_view(a1), w, h, pb, 64, 0, false);
    pb[4] = 0, pb[5] = 1;
    VkImageView by[2] = {ssao_view(a1), dv};
    ssao_pass(cb, p_blur, by, 2, ssao_view(a0), w, h, pb, 64, 0, false);
    VkImageView ai[1] = {ssao_view(a0)};
    // LARECOMP_SSAO_DUMP=<frame>,...: the occlusion as applied -> build/frames/ao_<frame>.png
    static const char *dump = (const char *)1;
    if (dump == (const char *)1) dump = getenv("LARECOMP_SSAO_DUMP");
    for (const char *q = dump; q && *q;) {
        long at = strtol(q, (char **)&q, 10);
        if (at == (long)fr) {
            char path[96];
            snprintf(path, sizeof path, "build/frames/ao_%05ld.png", at);
            renderer_save_texture_tex(a0, path);
            cb = vk_cmd();  // (submitted by the save)
        }
        if (*q == ',') q++; else break;
    }
    {
        float pa[16] = {0, 0, 0, 0, 1.0f / (float)ao.cw, 1.0f / (float)ao.ch, 0, 0};
        ssao_pass(cb, p_apply, ai, 1, ao.cview, ao.cw, ao.ch, pa, 64, 0, true);
    }
    static bool logged;
    if (!logged) {
        logged = true;
        LOG("[ssao] scene %ux%u (format %d), depth %ux%u (format %d), viewport %.0f,%.0f %.0fx%.0f, %u opaque draws before it; "
            "radius %.1f intensity %.2f samples %d\n", ao.cw, ao.ch, (int)ao.cfmt, w, h, (int)d->format, vx, vy, vw, vh, ao.opaque, radius,
            g_settings.ssao_intensity * kint, g_settings.ssao_samples);
    }
    ao.busy = false;
}


// ================================================================== TAA / DLAA
static bool mat4_inv(const double *m, double *o) {  // (row-major 4x4, Gauss-Jordan with pivoting)
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++) a[r][c] = c < 4 ? m[r * 4 + c] : (c - 4 == r);
    for (int c = 0; c < 4; c++) {
        int pv = c;
        for (int r = c + 1; r < 4; r++) if (fabs(a[r][c]) > fabs(a[pv][c])) pv = r;
        if (fabs(a[pv][c]) < 1e-20) return false;
        if (pv != c) for (int k = 0; k < 8; k++) { double t = a[c][k]; a[c][k] = a[pv][k]; a[pv][k] = t; }
        double d = a[c][c];
        for (int k = 0; k < 8; k++) a[c][k] /= d;
        for (int r = 0; r < 4; r++)
            if (r != c && a[r][c] != 0) {
                double f = a[r][c];
                for (int k = 0; k < 8; k++) a[r][k] -= f * a[c][k];
            }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) o[r * 4 + c] = a[r][c + 4];
    return true;
}

// Matrix at m (16 floats; rows: row-major, else column-major) as row-major doubles
static void taa_mat_at(const float *m, bool rows, double o[16]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) o[r * 4 + c] = rows ? m[r * 4 + c] : m[c * 4 + r];
}
static bool taa_rigid(const double *V) {
    if (V[12] != 0 || V[13] != 0 || V[14] != 0 || V[15] != 1) return false;
    for (int r = 0; r < 3; r++)
        if (fabs(V[r * 4] * V[r * 4] + V[r * 4 + 1] * V[r * 4 + 1] + V[r * 4 + 2] * V[r * 4 + 2] - 1) > 1e-3) return false;
    return true;
}
static void taa_mul(const double *a, const double *b, double *o) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double sum = 0;
            for (int k = 0; k < 4; k++) sum += a[r * 4 + k] * b[k * 4 + c];
            o[r * 4 + c] = sum;
        }
}
static bool taa_close(const double *a, const double *b) {
    for (int i = 0; i < 16; i++)
        if (fabs(a[i] - b[i]) > 1e-3 * fmax(1.0, fabs(a[i]))) return false;
    return true;
}

// The camera of this frame, from the projection found at p (byte off of its constant buffer, size bytes): the view
// matrix V of the same block - a rigid transform whose P x V the block holds too (its view-projection: Sword keeps it
// at the block's start, the view 464 bytes before the projection and the camera's own matrix, also rigid, elsewhere;
// Let's Go keeps view, projection, view-projection in a row). Found by a scan of the block once, then checked each
// frame where it was. vp = P x V.
static void taa_capture_camera(const float *p, uint32_t off, uint64_t size) {
    uint64_t fr = vk_frame_counter();
    if (!taa_wanted() || taa.cam_frame == fr) return;
    static int64_t voff = -1, vpoff = -1;  // (relative to the projection, bytes)
    bool row = p[14] == -1;  // (ssao_proj_at's two forms: rows, or columns)
    const float *b0 = p - off / 4;
    uint64_t n = size > 4096 ? 4096 : size;
    double P[16], V[16], vp[16], M[16];
    taa_mat_at(p, row, P);
    bool ok = false;
    if (voff != -1 && (int64_t)off + voff >= 0 && (uint64_t)((int64_t)off + voff) + 64 <= n) {  // (where it was)
        taa_mat_at(p + voff / 4, row, V);
        ok = taa_rigid(V);
    }
    if (!ok) {
        for (uint64_t o = 0; o + 64 <= n && !ok; o += 16) {
            if (o == off) continue;
            taa_mat_at(b0 + o / 4, row, V);
            if (!taa_rigid(V) || fabs(V[3]) + fabs(V[7]) + fabs(V[11]) < 1) continue;  // (a camera that does not move: none)
            taa_mul(P, V, vp);
            for (uint64_t q = 0; q + 64 <= n; q += 16) {
                taa_mat_at(b0 + q / 4, row, M);
                if (taa_close(vp, M)) { ok = true, voff = (int64_t)o - off, vpoff = (int64_t)q - off; break; }
            }
        }
        static int64_t said = -2;
        if (ok && said != voff)
            said = voff, LOG("[taa] camera: view at %+lld bytes from the projection, view-projection at %+lld (%s)\n",
                             (long long)voff, (long long)vpoff, row ? "rows" : "columns");
    }
    if (!ok) {
        if (!taa.cam_warned) {
            taa.cam_warned = true;
            LOG("[taa] no view matrix found by the projection (at %u, %s): no motion vectors\n", off, row ? "rows" : "columns");
            if (getenv("LARECOMP_CAM_PROBE"))  // (research: the camera block's words, 16 bytes a line)
                for (uint64_t o = 0; o + 16 <= n && o < off + 256; o += 16)
                    LOG("[taa]   +%4llu: %12.5g %12.5g %12.5g %12.5g\n", (unsigned long long)o, b0[o / 4], b0[o / 4 + 1], b0[o / 4 + 2],
                        b0[o / 4 + 3]);
        }
        return;
    }
    taa_mul(P, V, vp);
    taa.prev_ok = taa.cam_frame && taa.cam_frame + 1 == fr;
    memcpy(taa.vp_prev, taa.vp, sizeof vp);
    memcpy(taa.vp, vp, sizeof vp);
    taa.cam_frame = fr;
}

static float taa_env(const char *name, float def) {
    const char *e = getenv(name);
    return e && *e ? (float)atof(e) : def;
}

// Resolve this frame's scene (outside any pass): its camera motion vectors (from the depth and the two frames'
// view-projections), then DLAA (NVIDIA, when chosen and up) or TAA against the history, back into the scene's colour.
static void taa_run(void) {
    uint64_t fr = vk_frame_counter();
    taa.done_frame = fr;
    TexBackend *d = ao.dtex ? ao.dtex->backend : NULL;
    TexBackend *cb_ = ao.ctex ? ao.ctex->backend : NULL;
    static int nolog;
    if (!d || !d->image || !(d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || d->samples > 1 || !cb_ || !cb_->image || !ao.cview) return;
    if (taa.cam_frame != fr || ao.proj_frame != fr) {
        if (nolog < 5) nolog++, LOG("[taa] frame %llu: no camera found in the scene's %u opaque draws: not resolved\n",
                                    (unsigned long long)fr, ao.opaque);
        return;
    }
    uint32_t w = d->width, h = d->height;
    if (w != ao.cw || h != ao.ch) {
        if (nolog < 5) nolog++, LOG("[taa] depth %ux%u and scene %ux%u differ: not resolved\n", w, h, ao.cw, ao.ch);
        return;
    }
    ao.busy = true;
    VkCommandBuffer cb = vk_cmd();
    if (g_gpuprof_frame) gp_mark(cb, "ours: TAA");
    static const VkComponentMapping id = {0};
    VkImageView dv = vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_DEPTH_BIT);
    float mo = nvn_depth_mode() == 0 ? 1.0f : 0.0f;
    float vx = ao.vp[0], vy = ao.vp[1], vw = ao.vp[2], vh = ao.vp[3];
    if (vw < 16 || vh < 16) vx = vy = 0, vw = (float)w, vh = (float)h;
    // the motion vectors: previous view-projection x the inverse of this one (identity without a previous frame)
    double m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, inv[16];
    bool moved = taa.prev_ok && mat4_inv(taa.vp, inv);
    if (moved)
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) {
                double sum = 0;
                for (int k = 0; k < 4; k++) sum += taa.vp_prev[r * 4 + k] * inv[k * 4 + c];
                m[r * 4 + c] = sum;
            }
    static float ysign;
    if (!ysign) ysign = taa_env("LARECOMP_MV_YSIGN", 1);
    float pm[32] = {0};
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) pm[c * 4 + r] = (float)m[r * 4 + c];  // (GLSL mat4: columns)
    pm[16] = mo, pm[17] = ysign, pm[20] = vx, pm[21] = vy, pm[22] = vw, pm[23] = vh;
    const VkFormat F16 = VK_FORMAT_R16G16B16A16_SFLOAT;
    TexBackend *mv = fx_image(3, w, h, F16);
    ssao_pass(cb, ssao_pipe("taa_mv.frag", F16, 0), &dv, 1, ssao_view(mv), w, h, pm, 96, 0, false);
    bool full = vx == 0 && vy == 0 && (uint32_t)vw == w && (uint32_t)vh == h;
    // DLAA: the scene's rectangle from the top-left (all of the target, or Sword's dynamic resolution's part of it)
    bool corner = vx == 0 && vy == 0 && vw == floorf(vw) && vh == floorf(vh);
    (void)full;
    int mode = g_settings.antialias == AA_DLAA && dlss_available() && corner ? 2 : 1;
    static uint64_t taa_prev_frame;
    static int taa_prev_mode;
    bool reset = !moved || taa.hist_frame + 1 != fr || taa.hist_w != w || taa.hist_h != h || taa.hist_mode != mode ||
                 memcmp(taa.hist_vp, (float[4]){vx, vy, vw, vh}, sizeof taa.hist_vp) != 0;
    float jx = taa.jit_used == fr ? taa.jx : 0, jy = taa.jit_used == fr ? taa.jy : 0;
    static float sharp_k = -1;
    if (sharp_k < 0) sharp_k = taa_env("LARECOMP_TAA_SHARPEN", 1.0f);
    TexBackend *res = NULL;
    if (mode == 2) {  // DLAA: the scene's own image, the depth as R32F, the vectors in pixels (previous - current)
        uint32_t rw = (uint32_t)vw, rh = (uint32_t)vh;
        TexBackend *dz = fx_image(6, w, h, VK_FORMAT_R32_SFLOAT), *out = fx_image(7, rw, rh, F16);
        float pz[16] = {0};
        ssao_pass(cb, ssao_pipe("depth_r32.frag", VK_FORMAT_R32_SFLOAT, 0), &dv, 1, ssao_view(dz), w, h, pz, 64, 0, false);
        DlssInput in = {.color = cb_->image, .depth = dz->image, .mv = mv->image, .output = out->image, .color_fmt = cb_->format,
                        .rw = rw, .rh = rh, .ow = rw, .oh = rh, .iw = w, .ih = h, .jx = jx, .jy = jy,
                        .mv_scale_x = -(float)w, .mv_scale_y = -(float)h,
                        .reset = reset};
        if (dlss_dispatch(cb, &in)) res = out;
        else mode = 1;  // (DLSS failed: TAA from now on)
        cb = vk_cmd();
    }
    if (mode == 1) {  // TAA: this frame against the history (ping-pong 4 / 5), the history by Catmull-Rom (linear)
        TexBackend *hc = fx_image(4 + (int)(fr & 1), w, h, F16), *hp = fx_image(4 + (int)((fr + 1) & 1), w, h, F16);
        static float weight = -1;
        if (weight < 0) weight = taa_env("LARECOMP_TAA_HISTORY", 0.9f);
        float pt[16] = {ao.proj[0], ao.proj[1], ao.proj[2], ao.proj[3], reset ? 1.0f : 0.0f, weight, mo, 0,
                        vx / (float)w, vy / (float)h, (vx + vw) / (float)w, (vy + vh) / (float)h};
        VkImageView src[4] = {ao.cview, ssao_view(hp), ssao_view(mv), dv};
        ssao_pass(cb, ssao_pipe("taa.frag", F16, 0), src, 4, ssao_view(hc), w, h, pt, 64, 2, false);
        res = hc;
    }
    // back into the scene, sharpened (TAA_SHARPNESS: the history's blur; DLAA sharpens nothing itself)
    float pk[8] = {0, 0, 0, 0, 1.0f / (float)w, 1.0f / (float)h, g_settings.taa_sharpness * (mode == 2 ? 0.5f : 0.8f) * sharp_k, 0};
    VkImageView ri[1] = {ssao_view(res)};
    ssao_pass(cb, ssao_pipe("taa_copy.frag", ao.cfmt, 0), ri, 1, ao.cview, w, h, pk, 32, 0, false);
    taa.hist_frame = fr, taa.hist_w = w, taa.hist_h = h, taa.hist_mode = mode;
    taa.hist_vp[0] = vx, taa.hist_vp[1] = vy, taa.hist_vp[2] = vw, taa.hist_vp[3] = vh;
    taa.ok_frame = fr;
    static int logged_mode;
    if (logged_mode != mode) {
        logged_mode = mode;
        LOG("[taa] %s: scene %ux%u (format %d), viewport %.0f,%.0f %.0fx%.0f, history %s, jitter %.3f %.3f px, %u opaque draws\n",
            mode == 2 ? "DLAA" : "TAA", w, h, (int)cb_->format, vx, vy, vw, vh, reset ? "reset" : "kept", jx, jy, ao.opaque);
    }
    {   // every 600 frames: how many were resolved, with their history, jittered (the reasons for a reset)
        static uint32_t n, kept, jit, why[4];
        n++, kept += !reset, jit += taa.jit_used == fr;
        if (reset) why[0] += !moved, why[1] += taa_prev_frame + 1 != fr, why[2] += taa_prev_mode != mode;
        taa_prev_frame = fr, taa_prev_mode = mode;
        if (n >= 600) {
            LOG("[taa] %u frames resolved: history kept %u, jittered %u; resets: no previous camera %u, a frame skipped %u, "
                "mode changed %u\n", n, kept, jit, why[0], why[1], why[2]);
            n = kept = jit = 0, memset(why, 0, sizeof why);
        }
    }
    ao.busy = false;
}

// draw_, right after begin_pass: counts the scene's opaque draws (and finds the projection with them); at the first
// draw after them that is not the scene's (another pass, or no depth test in the scene) ends the pass and runs the
// SSAO. true: the caller begins the pass again.
static bool ssao_draw_hook(void) {
    uint64_t fr = vk_frame_counter();
    static int nlog;
    if (ao.pending) {
        ao.pending = false;
        if (nlog < 3) {
            nlog++;
            NvnTexture *c = st.rt.ncolor ? st.rt.color[0] : NULL;
            LOG("[ssao] frame %llu (present %llu): run before a pass on other targets (%d colour, c0 fmt %#x %ux%u @%llx, d @%llx; "
                "scene c0 @%llx d @%llx), after %u opaque draws, %llu draws in all\n", (unsigned long long)fr,
                (unsigned long long)renderer_present_count(), st.rt.ncolor, c ? c->format : 0, c ? c->width : 0, c ? c->height : 0,
                c ? (unsigned long long)c->addr : 0ull, st.rt.depth ? (unsigned long long)st.rt.depth->addr : 0ull,
                (unsigned long long)ao.ctex->addr, (unsigned long long)ao.dtex->addr, ao.opaque, (unsigned long long)stat_draws);
        }
        if (!g_settings.ssao) { ao.done_frame = fr; return false; }  // (TAA on alone: the scene's place only)
        vk_end_pass();
        ssao_run();
        return true;
    }
    if (!(st.color_state[0] & 1) && st.ds[0] && st.ds[1]) {
        ao.opaque++;
        ssao_find_projection();
        return false;
    }
    if (st.ds[0] || ao.opaque < 16 || ao.done_frame == fr) return false;
    if (!g_settings.ssao) { ao.done_frame = fr; return false; }
    if (nlog < 3) nlog++, LOG("[ssao] frame %llu: run before a draw without depth test in the scene, after %u opaque draws\n",
                              (unsigned long long)fr, ao.opaque);
    vk_end_pass();
    ssao_run();
    return true;
}

static void draw_(CmdRec *r, bool indexed) {
    if (prof) prof_tp_ = SDL_GetTicksNS();
    if (!st.program) { SKIP(0); return; }
    if (ip_enabled()) {  // (frame interpolation: what identifies this draw from one frame to the next)
        // what the draw is (program, vertex buffer, parameters), and which game object it belongs to: the same mesh
        // drawn for two objects (bushes, Pidgeys) is told apart by the object, not by the order the game drew them in
        uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t)(uintptr_t)st.program ^ ((uint64_t)indexed << 63);
        h = (h ^ st.vbuf[0].addr) * 0x100000001B3ull;
        for (int i = 0; i < 8; i++) h = (h ^ r->x[i]) * 0x100000001B3ull;
        if (st.obj) h = (h ^ st.obj) * 0x100000001B3ull, ip_stats_tagged++;
        else ip_stats_untagged++;
        ip_draw_sig = h;
        // 3D draws (a depth target with the depth test on) are blended, with or without a game object: the ground and the
        // scenery have none, and they must take the same (blended) light matrices as the shadow map the objects are drawn
        // into, or shadows slide away from their casters. Interface and full-screen passes don't test depth: frame N.
        ip_draw_obj = st.rt.depth != NULL && st.ds[0];
        // In menus (the Pokédex, the bag...) the 2D draws are blended too: lists and panels slide there. A 2D draw takes the
        // strict jump test in ip_cbuf (a recycled row or glyph moves far: it keeps frame N).
        ip_draw_2d = !ip_draw_obj;
        extern bool fps60_menu(void);
        if (fps60_menu()) ip_draw_obj = true;
    }
    ShaderKey vkey, fkey;
    draw_shader_keys(&vkey, &fkey);
    Prog *pr = get_program(st.program, &vkey, &fkey);
    if (pr && atomic_load(&pr->state) == 0) pf_rest_of_list();  // (about to wait: find the rest of the list's needs first)
    if (pr) wait_built(&pr->state, translate_program, pr);
    if (!pr || atomic_load(&pr->state) != 1) { SKIP(pr ? 1 + (atomic_load(&pr->state) > 1) : 3); return; }
    if (draw_texint_keys(pr, &vkey, &fkey)) {
        pr = get_program(st.program, &vkey, &fkey);
        if (pr && atomic_load(&pr->state) == 0) pf_rest_of_list();
        if (pr) wait_built(&pr->state, translate_program, pr);
        if (!pr || atomic_load(&pr->state) != 1) { SKIP(pr ? 1 + (atomic_load(&pr->state) > 1) : 3); return; }
    }
    {   // fragment-shader hash: interior backdrop detection (widescreen), skip lists, item twinkle (renderer.m)
        static uint64_t last_fs, last_hash;
        uint64_t fs = st.program->stages[1].code_addr;
        if (fs != last_fs) {
            last_fs = fs;
            last_hash = fs ? hmix(1469598103934665603ull, (const void *)(uintptr_t)(fs + 0x30), 0x200) : 0;
        }
        static uint64_t skip[16];
        static int nskip = -1;
        if (nskip < 0) {
            nskip = 0;
            for (const char *p = getenv("LARECOMP_SKIP_FSHASH"); p && *p && nskip < 16;) {
                skip[nskip++] = strtoull(p, (char **)&p, 16);
                while (*p == ',') p++;
            }
        }
        for (int i = 0; i < nskip; i++) if (last_hash == skip[i]) { stat_skipped++; return; }
        if (g_debug_frame && getenv("LARECOMP_LOG_FSHASH")) LOG("[fshash] fs=%llx hash %016llx\n", (unsigned long long)fs, (unsigned long long)last_hash);
        g_cur_fs_hash = last_hash;
    }
    {   // (research, LARECOMP_COUNT_TEX=<w>x<h>: per frame, the draws and instances sampling a fragment texture of that size)
        static int tw = -1, th;
        if (tw < 0) { const char *e = getenv("LARECOMP_COUNT_TEX"); tw = 0; if (e) sscanf(e, "%dx%d", &tw, &th); }
        if (tw > 0) {
            static uint64_t frame = ~0ull;
            static unsigned nd, ni;
                        if (renderer_present_count() != frame) {
                if (frame != ~0ull) LOG("[counttex] frame %llu: %u draws %u instances\n", (unsigned long long)frame, nd, ni);
                frame = renderer_present_count(), nd = ni = 0;
            }
            for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
                NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
                if (t && (int)t->width == tw && (int)t->height == th) { nd++; ni += (unsigned)(indexed ? r->x[6] : r->x[4]); break; }
            }
        }
    }
    if (g_debug_frame) debug_draw(pr, r, indexed);
    else if ((pr->vi->attr_mask >> 11) & 1) {  // skinned draws (a11: bone indices)
        static int n;
        static int64_t dump_frame = -2;  // LARECOMP_DUMP_SKIN_FRAME: dump the first one from that frame on
        if (dump_frame == -2) dump_frame = getenv("LARECOMP_DUMP_SKIN_FRAME") ? atoll(getenv("LARECOMP_DUMP_SKIN_FRAME")) : -1;
        if (dump_frame >= 0 && renderer_present_count() >= (uint64_t)dump_frame) {
            debug_draw(pr, r, indexed);  // (debug_draw writes build/skin/ once)
            dump_frame = -1;
        } else if (getenv("LARECOMP_LOG_SKIN") && n++ % 50 == 0 && n < 1000) {
            debug_draw(pr, r, indexed);
        }
    }
    resolve_feedback(pr);
    snap_reason = "draw-depthwrite";
    if (st.rt.depth && st.ds[0] && st.ds[1]) snapshots_invalidate(st.rt.depth);
    snap_reason = "draw-colorwrite";
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i] && ((st.channel_mask >> (i * 4)) & 0xF)) snapshots_invalidate(st.rt.color[i]);
    begin_pass();
    if ((ao.qual || ao.pending) && ssao_draw_hook()) begin_pass();
    if (taa.pending) { taa.pending = false; vk_end_pass(); taa_run(); begin_pass(); }  // (TAA / DLAA: taa_run)
    VkCommandBuffer cb = vk_cmd();

    uint32_t mode = (uint32_t)r->x[0];
    VkPrimitiveTopology topo;
    uint8_t cls;
    bool native = topology(mode, &topo, &cls);
    if (!native && mode != 6 && mode != 7 && mode != 9) { SKIP(4); return; }

    uint64_t prof_tk_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_pre += prof_tk_ - prof_tp_;
    // ---- pipeline
    PipeKey k;
    pipe_key_fill(&k, pr, cls, pass_ncolor, pass_color, pass_depth, pass_stencil, pass_samples);
    stat_alpha_func[st.color_state[2] & 15]++;
    bool created;
    Pipe *pp = pipe_get(&k, false, &created);
    if (created || (!pp->drawn && !pp->prewarmed)) recipe_record(pr, &k);  // (a prefetched pipe: first drawn here)
    else if (pp->prewarmed && !pp->drawn) stat_recipe_adopted++;
    pp->drawn = true;
    if (atomic_load(&pp->state) == 0) pf_rest_of_list();  // (about to wait: find the rest of the list's needs first)
    VkPipeline use = VK_NULL_HANDLE;
    if (g_vk.gpl && atomic_load(&pp->state) != 1) {  // (whole pipeline not ready: the one linked from parts)
        int none = -1;
        if (atomic_compare_exchange_strong(&pp->fstate, &none, 0)) link_pipeline(pp);
        else wait_built(&pp->fstate, link_pipeline, pp);
        if (atomic_load(&pp->fstate) == 1) { use = pp->fast; atomic_fetch_add(&stat_gpl_used, 1); }
    }
    if (!use) {
        wait_built(&pp->state, build_pipeline, pp);
        if (atomic_load(&pp->state) != 1) { SKIP(5 + (atomic_load(&pp->state) > 1)); return; }
        use = pp->pipe;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, use);
    uint64_t prof_td_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_key += prof_td_ - prof_tk_;

    // ---- depth / stencil (dynamic)
    static int nodt = -1;
    if (nodt < 0) nodt = getenv("LARECOMP_NO_DEPTH_TEST") != NULL;
    bool dt = st.ds[0] && !nodt && pass_depth;
    vkCmdSetDepthTestEnable(cb, dt);
    vkCmdSetDepthWriteEnable(cb, dt && st.ds[1]);
    VkCompareOp dop = dt ? compare_fn(st.ds[2]) : VK_COMPARE_OP_ALWAYS;
    // EQUAL against a depth pre-pass written by a different vertex shader (foliage, trees) needs both shaders to land
    // on identical depths. They do since the translator stopped using unpackHalf2x16 (NVIDIA ran the half math after
    // it in fp16, differently per shader: black slashes in foliage). LARECOMP_VK_EQUAL_TOL=1 brings back the old
    // workaround: LESS_OR_EQUAL with a few ulps of bias towards the camera, which also let walls paint over the
    // plants stuck flat on them (beach) - so it is off.
    static int tol = -1;
    if (tol < 0) tol = getenv("LARECOMP_VK_EQUAL_TOL") != NULL;
    bool equal_tol = dop == VK_COMPARE_OP_EQUAL && tol;
    if (equal_tol) dop = VK_COMPARE_OP_LESS_OR_EQUAL;
    vkCmdSetDepthCompareOp(cb, dop);
    bool sten = st.ds[3] && pass_stencil;
    vkCmdSetStencilTestEnable(cb, sten);
    if (sten) {
        uint32_t ops = st.ds[5] | st.ds[6] << 8 | st.ds[7] << 16;
        for (int f = 0; f < 2; f++) {
            uint32_t o = f ? ops >> 12 : ops;
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkCmdSetStencilOp(cb, face, stencil_op(o & 0xF), stencil_op((o >> 8) & 0xF), stencil_op((o >> 4) & 0xF),
                              compare_fn(f ? st.ds[4] >> 4 : st.ds[4] & 0xF));
            vkCmdSetStencilCompareMask(cb, face, st.stencil_vmask[f] & 0xFF);
            vkCmdSetStencilWriteMask(cb, face, st.stencil_mask[f] & 0xFF);
            vkCmdSetStencilReference(cb, face, st.stencil_ref[f]);
        }
    }
    vkCmdSetBlendConstants(cb, st.blend_color);

    // ---- rasterizer
    uint8_t cull = st.poly[0];
    vkCmdSetCullMode(cb, cull == 1 ? VK_CULL_MODE_FRONT_BIT : cull == 2 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE);
    bool ccw = st.poly[1] == 1;
    static int flipw = -1;  // LARECOMP_VK_FLIP_WINDING: invert the front-face mapping (A/B against Metal)
    if (flipw < 0) flipw = getenv("LARECOMP_VK_FLIP_WINDING") != NULL;
    bool front_ccw = (ccw != vkey.flip_y) != (flipw != 0);
    vkCmdSetFrontFace(cb, front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, native ? topo : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetPrimitiveRestartEnable(cb, st.prim_restart && native && (mode == 3 || mode == 5));
    {
        float f[3] = {0, 0, 0};
        static int off = -1;
        if (off < 0) off = getenv("LARECOMP_NO_DEPTH_BIAS") != NULL;
        bool on = (st.poly[3] & 4) && !off;
        if (on) memcpy(f, st.poly_offset, sizeof f);
        if (equal_tol) { f[1] -= 8; on = true; }  // (see the EQUAL comment above: a few depth ulps nearer)
        vkCmdSetDepthBiasEnable(cb, on);
        vkCmdSetDepthBias(cb, f[1], f[2], f[0]);
    }

    // ---- textures first (their scales feed the widescreen decision and the rscale table), then the viewport
    float *vp = st.viewport[0];
    float zn = st.depth_range[0][0], zf = st.depth_range[0][1];
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    float tint[8] = {0};  // (the per-draw tint the translator can apply to colour target 0: none)
    // fragment textures are bound before deciding the viewport (tex_sc / tex_wh), rscale depends on the decision:
    // build the vertex set now and the fragment set after the decision (stage_set fills tex_sc for stage 1).
    if (prof) prof_dyn += SDL_GetTicksNS() - prof_td_;
    PROF_T0();
    VkDescriptorSet vset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, NULL, NULL, 0, 0, 0, NULL);
    PROF_ADD(prof_sets);
    VkDescriptorImageInfo fii[64];
    uint64_t prof_tx_ = prof ? SDL_GetTicksNS() : 0;
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
        float sc[2]; uint32_t wh[2];
        texture_info(1, pr->fi, i, &fii[i], sc, wh);
        tex_sc[1][i][0] = sc[0]; tex_sc[1][i][1] = sc[1]; tex_wh[1][i][0] = wh[0]; tex_wh[1][i][1] = wh[1];
    }
    if (prof) prof_tex += SDL_GetTicksNS() - prof_tx_;
    if (fabsf(pass_sx - pass_sy) > 0.01f * pass_sy && st.rt.ncolor && renderer_texture_presented(st.rt.color[0])) {
        // Draws into the swapchain image (see renderer.m): composites fill the window, UI goes in a centred 16:9 box.
        bool reads_scene = false;
        for (int i = 0; i < pr->fi->ntex && i < 64; i++)
            if ((tex_sc[1][i][0] != 1 || tex_sc[1][i][1] != 1) && tex_wh[1][i][0] == st.rt.color[0]->width &&
                tex_wh[1][i][1] == st.rt.color[0]->height)
                reads_scene = true;
        bool fullscreen_quad = r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 &&
                               st.viewport[0][2] >= st.rt.color[0]->width && st.viewport[0][3] >= st.rt.color[0]->height;
        bool fullscreen_fill = pr->fi->ntex == 0 && fullscreen_quad;
        if (fullscreen_quad && !reads_scene && pr->fi->ntex == 1 && (tex_sc[1][0][0] != 1 || tex_sc[1][0][1] != 1)) reads_scene = true;
        static uint64_t wide_frame = ~0ull;
        if (reads_scene) wide_frame = renderer_present_count();
        bool fade_fill = false;
        if (fullscreen_fill && st.ubuf[1][0].addr && st.ubuf[1][0].size >= 12) {
            const float *c = (const float *)(uintptr_t)st.ubuf[1][0].addr;
            bool black = c[0] < 0.01f && c[1] < 0.01f && c[2] < 0.01f, white = c[0] > 0.99f && c[1] > 0.99f && c[2] > 0.99f;
            fade_fill = black || white;
        }
        bool composite = reads_scene || fade_fill || (fullscreen_fill && wide_frame == renderer_present_count());
        if (!composite) wide_frame = ~0ull;
        if (!composite) {
            int pw, ph;
            pass_size(&pw, &ph);
            float u = fminf(pass_sx, pass_sy);
            vp_sx = vp_sy = u;
            vp_ox = (pw - st.rt.color[0]->width * u) * 0.5f;
            vp_oy = (ph - st.rt.color[0]->height * u) * 0.5f;
        }
    }
    {   // (research, LARECOMP_COUNT_FS=<fs hash>: draws with that fragment shader per frame, logged when the count changes)
        static uint64_t want = 1, frame = ~0ull;
        static unsigned n, last = ~0u;
        if (want == 1) { const char *e = getenv("LARECOMP_COUNT_FS"); want = e ? strtoull(e, NULL, 16) : 0; }
        if (want) {
                        if (renderer_present_count() != frame) {
                if (frame != ~0ull && n != last) LOG("[countfs] frame %llu: %u\n", (unsigned long long)frame, n);
                if (frame != ~0ull) last = n;
                frame = renderer_present_count(), n = 0;
            }
            if (g_cur_fs_hash == want) n++;
        }
    }
    {
        float vx = vp[0] * vp_sx + vp_ox, vw = vp[2] * vp_sx;
        bool neg_y = st.vswz[0][1] == 3;  // NVN_VIEWPORT_SWIZZLE_NEGATIVE_Y
        static bool told;
        if (neg_y && !told) { told = true; LOG("[nvn] a draw with the NEGATIVE_Y viewport swizzle (flipped)\n"); }
        set_viewport(vx, vp[1] * vp_sy + vp_oy, vw, vp[3] * vp_sy, zn, zf, neg_y);
        if (g_debug_frame) {
            VkRect2D sr = scissor_rect();
            LOG("[draw]   vp %.2f %.2f %.2f %.2f (fs hash %016llx) scissor %d %d %u %u | guest sc %d %d %d %d | pass %ux%u s %.4f %.4f\n",
                vx, vp[1] * vp_sy + vp_oy, vw, vp[3] * vp_sy, (unsigned long long)g_cur_fs_hash, sr.offset.x, sr.offset.y,
                sr.extent.width, sr.extent.height, st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3], pass_w, pass_h,
                vp_sx, vp_sy);
        }
    }
    apply_scissor();
    float rs0[4] = {vp_sx, vp_sy, vp_ox, vp_oy};
    uint64_t prof_t1_ = prof ? SDL_GetTicksNS() : 0;
    gpuw_stores(pr->vi, st.sbuf[0], 16, "vertex shader", st.program->stages[0].code_addr);
    gpuw_stores(pr->fi, st.sbuf[1], 16, "fragment shader", st.program->stages[1].code_addr);
    if (pr->vi->uses_gmem || pr->fi->uses_gmem) {
        readbacks_before_gmem();
        g_gmem_seq = vk_pending_seq();
        if (pr->vi->uses_gmem) gmem_ranges_note(st.sbuf[0], 16);
        if (pr->fi->uses_gmem) gmem_ranges_note(st.sbuf[1], 16);
    }
    VkDescriptorSet fset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, tint, 0, 0, 0, fii);
    if (prof) prof_sets += SDL_GetTicksNS() - prof_t1_;
    VkDescriptorSet sets[2] = {vset, fset};
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 0, 2, sets, 0, NULL);

    uint64_t prof_t2_ = prof ? SDL_GetTicksNS() : 0;
    // ---- indices (copied; the largest index bounds the vertex data copied below)
    uint32_t first = 0, count = (uint32_t)r->x[2], inst, base_inst, max_index = 0;
    int32_t base_vertex = 0;
    VkSpan ib = {0};
    VkIndexType itype_vk = VK_INDEX_TYPE_UINT32;
    uint32_t icount = 0;
    bool use_index = indexed || !native;
    const void *scan_src = NULL;  // deferred index scan (direct index buffer)
    uint32_t scan_isz = 0, scan_restart = 0;
    if (!indexed) {
        first = (uint32_t)r->x[1]; base_inst = (uint32_t)r->x[3]; inst = (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 1;
        max_index = first + count;
        if (!native) {
            uint32_t *e = expand_indices(mode, count, NULL, 0, first, &icount);
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
    } else {
        uint32_t itype = (uint32_t)r->x[1];
        uint64_t iaddr = r->x[3];
        base_vertex = (int32_t)r->x[4];
        base_inst = (uint32_t)r->x[5];
        inst = (uint32_t)r->x[6] ? (uint32_t)r->x[6] : 1;
        const void *src = (const void *)(uintptr_t)iaddr;
        uint32_t restart = st.prim_restart ? st.restart_index : 0xFFFFFFFFu;
        if (native && itype != 0) {
            uint32_t isz = itype == 1 ? 2 : 4;
            itype_vk = itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            icount = count;
            if (iaddr % isz == 0 && vtx_direct(iaddr, (uint64_t)count * isz, &ib.buf, &ib.off)) {
                scan_src = src; scan_isz = isz; scan_restart = restart;  // max_index only if a stream gets copied
            } else {
                PcEnt *pe;
                uint64_t ilen = (uint64_t)count * isz;
                ib = iaddr % 4 == 0 ? pc_get(iaddr, ilen + 4, &pe) : (VkSpan){0};
                if (ib.buf) {  // (persistent copy: its largest index too)
                    if (!(pe->mi_len == ilen && pe->isz == isz && pe->restart == restart)) {
                        pe->max_index = scan_max_index(src, isz, count, restart);
                        pe->mi_len = ilen; pe->isz = isz; pe->restart = restart;
                    }
                    if (pe->max_index > max_index) max_index = pe->max_index;
                } else {
                    ib = transient_gpu_copy(src, ilen + 4, 4);
                    { uint32_t m = scan_max_index(src, isz, count, restart); if (m > max_index) max_index = m; }
                }
            }
        } else if (native) {  // 8-bit indices: widen to 16
            ib = vk_transient((uint64_t)count * 2 + 4, 4);
            uint16_t *w = ib.ptr;
            const uint8_t *s = src;
            for (uint32_t i = 0; i < count; i++) { w[i] = s[i] == 0xFF && st.prim_restart ? 0xFFFF : s[i]; if (s[i] > max_index && w[i] != 0xFFFF) max_index = s[i]; }
            itype_vk = VK_INDEX_TYPE_UINT16;
            icount = count;
        } else {
            uint32_t *e = expand_indices(mode, count, src, (int)itype, 0, &icount);
            for (uint32_t i = 0; i < icount; i++) if (e[i] > max_index) max_index = e[i];
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
        if (!scan_src) {
            max_index += 1;
            if (base_vertex > 0) max_index += (uint32_t)base_vertex;
        }
    }
    if (use_index) vkCmdBindIndexBuffer(cb, ib.buf, ib.off, itype_vk);

    // ---- vertex streams: the range this draw can read, copied once per submission
    for (int s = 0; s < 17; s++) {
        if (!(k.used_streams & (1u << s))) continue;
        VkSpan sp;
        if (s == ZERO_BINDING) {
            static const float zero[4] = {0, 0, 0, 1};
            sp = vk_transient_copy(zero, 16, 16);
        } else {
            Range v = st.vbuf[s];
            VkBuffer dbuf;
            VkDeviceSize doff;
            if (v.addr && v.size && vtx_direct(v.addr, v.size, &dbuf, &doff)) {
                vkCmdBindVertexBuffers(cb, (uint32_t)s, 1, &dbuf, &doff);
                continue;
            }
            if (scan_src) {  // copying after all: the index scan bounds the copy
                { uint32_t m = scan_max_index(scan_src, scan_isz, icount, scan_restart); if (m > max_index) max_index = m; }
                max_index += 1;
                if (base_vertex > 0) max_index += (uint32_t)base_vertex;
                scan_src = NULL;
            }
            uint64_t stride = k.stride[s], need;
            if (!stride) need = 256;
            else if (k.divisor[s]) need = ((uint64_t)base_inst + inst + k.divisor[s] - 1) / k.divisor[s] * stride + 256;
            else need = (uint64_t)max_index * stride + 256;
            if (v.size && need > v.size) need = v.size;
            if (!v.addr) { static const uint8_t zero[256]; sp = vk_transient_copy(zero, 256, 16); }
            else {
                sp = vertex_copy(v.addr, need ? need : 16, v.size, k.divisor[s] == 0);
                if (ENV("LARECOMP_LOG_VCOPY") && g_gpuprof_frame)  // (research: what a profiled frame copies, per stream)
                    LOG("[vcopy] vs %llx stream %d addr %llx need %llu of bound %llu stride %llu %s max_index %u base_vertex %d base_inst %u inst %u\n",
                        (unsigned long long)st.program->stages[0].code_addr, s, (unsigned long long)v.addr, (unsigned long long)need,
                        (unsigned long long)v.size, (unsigned long long)stride, k.divisor[s] ? "instance" : "vertex", max_index, base_vertex, base_inst, inst);
            }
        }
        VkDeviceSize off = sp.off;
        vkCmdBindVertexBuffers(cb, (uint32_t)s, 1, &sp.buf, &off);
    }

    if (prof) prof_vtx += SDL_GetTicksNS() - prof_t2_;
    // ---- draw
    // LARECOMP_GPUPROF_DRAWS=1 (with LARECOMP_GPUPROF_AT): a mark before every draw, so each row is one draw's GPU time
    static int gp_draw_marks = -1;
    if (gp_draw_marks < 0) gp_draw_marks = getenv("LARECOMP_GPUPROF_DRAWS") != NULL;
    if (g_gpuprof_frame && gp_draw_marks)
        gp_mark(cb, "draw vs %llx fs %016llx fsaddr %llx %s %u inst %u", (unsigned long long)st.program->stages[0].code_addr,
                (unsigned long long)g_cur_fs_hash, (unsigned long long)st.program->stages[1].code_addr, use_index ? "elements" : "arrays", use_index ? icount : count, inst);
    // Experiment (research): LARECOMP_EXP_VS=<vertex program address, hex> with LARECOMP_EXP_MODE=skip | scissor | half
    // changes only the draws of that program: no draw, a 1x1 scissor (no pixel work, vertex work kept) or half the instances.
    static uint64_t exp_vs;
    static int exp_mode = -1;
    if (exp_mode < 0) {
        const char *v = getenv("LARECOMP_EXP_VS"), *m = getenv("LARECOMP_EXP_MODE");
        exp_vs = v ? strtoull(v, NULL, 16) : 0;
        exp_mode = !exp_vs || !m ? 0 : !strcmp(m, "skip") ? 1 : !strcmp(m, "scissor") ? 2 : !strcmp(m, "half") ? 3 : 0;
    }
    // (LARECOMP_EXP_ALT=1: only on odd frames, so one run holds both versions of the same scene)
    static int exp_alt = -1;
    if (exp_alt < 0) exp_alt = getenv("LARECOMP_EXP_ALT") != NULL;
    bool exp_hit = exp_mode && st.program->stages[0].code_addr == exp_vs && (!exp_alt || (vk_frame_counter() & 1));
    uint32_t inst_eff = exp_hit && exp_mode == 3 && inst > 1 ? inst / 2 : inst;
    if (exp_hit && exp_mode == 2) { VkRect2D one = {{0, 0}, {1, 1}}; vkCmdSetScissor(cb, 0, 1, &one); }
    bool gp_q = g_gpuprof_frame && gp_draw_marks && gps_qp && gp_n && gps_n < GP_MAX;
    if (gp_q) vkCmdBeginQuery(cb, gps_qp, (uint32_t)gps_n, 0);
    if (exp_hit && exp_mode == 1) { /* skipped */ }
    else if (!use_index) vkCmdDraw(cb, count, inst_eff, first, base_inst);
    else vkCmdDrawIndexed(cb, icount, inst_eff, 0, base_vertex, base_inst);
    if (gp_q) { vkCmdEndQuery(cb, gps_qp, (uint32_t)gps_n); gp_stat[gp_n - 1] = gps_n + 1; gps_n++; }
    if (exp_hit && exp_mode == 2) apply_scissor();
    stat_draws++;
    g_draws_total++;
    cum_draws++;
    if (g_gpuprof_frame && gp_n && !gp_draw_marks) {
        if (!gp_draws[gp_n - 1]) {  // first draw of the row: its fragment shader hash
            size_t l = strlen(gp_label[gp_n - 1]);
            snprintf(gp_label[gp_n - 1] + l, sizeof gp_label[0] - l, ", first fs %016llx", (unsigned long long)g_cur_fs_hash);
        }
        gp_draws[gp_n - 1]++;
    }
}

static void debug_draw(Prog *pr, CmdRec *r, bool indexed) {
    LOG("[draw] %s inst %llu mode %llu count %llu prog vs=%llx fs=%llx | vs cb %#x fs cb %#x | rt %d", indexed ? "e" : "a",
        (unsigned long long)(indexed ? r->x[6] : r->x[4]), (unsigned long long)r->x[0], (unsigned long long)r->x[2],
        (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr,
        pr->vi->cbuf_mask, pr->fi->cbuf_mask, st.rt.ncolor);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) LOG(" %ux%u/f%#x", st.rt.color[i]->width, st.rt.color[i]->height, st.rt.color[i]->format);
    if (st.rt.depth) LOG(" d%#x %p", st.rt.depth->format, (void *)st.rt.depth);
    LOG(" | vp %.0f %.0f %.0f %.0f | attrs %#x blend %#x ds %02x%02x%02x poly %02x%02x chmask %08x\n", st.viewport[0][0], st.viewport[0][1],
        st.viewport[0][2], st.viewport[0][3], pr->vi->attr_mask, st.color_state[0], st.ds[0], st.ds[1], st.ds[2], st.poly[0], st.poly[1],
        st.channel_mask);
    LOG("[draw]   attribs:");  // format / stream / offset per attribute, and the streams' strides
    for (int i = 0; i < st.nattrib; i++)
        if ((pr->vi->attr_mask >> i) & 1)
            LOG(" a%d=f%#x/s%u/+%u", i, st.attrib[i] & 0xFF, (st.attrib[i] >> 8) & 0x1F, st.attrib[i] >> 13);
    for (int s = 0; s < st.nstream && s < 4; s++) LOG(" | stream%d stride %u", s, st.stream[s][0]);
    LOG("\n");
    if (getenv("LARECOMP_DUMP_SKIN") && ((pr->vi->attr_mask >> 11) & 1)) {  // one skinned draw's inputs -> build/skin/
        static int dumped;
        const char *at_frame = getenv("LARECOMP_DUMP_SKIN_FRAME");  // (the first one from that frame on)
        if (at_frame ? renderer_present_count() >= (uint64_t)atoll(at_frame) && !dumped++
                     : ++dumped == (atoi(getenv("LARECOMP_DUMP_SKIN")) > 1 ? atoi(getenv("LARECOMP_DUMP_SKIN")) : 1)) {  // (or the Nth skinned draw)
            mkdir("build/skin", 0755);
            char path[256];
            FILE *f;
            snprintf(path, sizeof path, "build/skin/info.txt");
            if ((f = fopen(path, "w"))) {
                fprintf(f, "vs %llx stride %u\n", (unsigned long long)st.program->stages[0].code_addr, st.stream[0][0]);
                for (int i = 0; i < st.nattrib; i++)
                    if ((pr->vi->attr_mask >> i) & 1)
                        fprintf(f, "attr %d fmt %#x off %u\n", i, st.attrib[i] & 0xFF, st.attrib[i] >> 13);
                for (int c = 0; c < 18; c++)
                    if ((pr->vi->cbuf_mask >> (c + 3)) & 1)
                        fprintf(f, "cb %d size %u\n", c + 3, (unsigned)st.ubuf[0][c].size);
                fclose(f);
            }
            uint64_t vsz = (uint64_t)st.stream[0][0] * 64;
            if (st.vbuf[0].addr && (f = fopen("build/skin/verts.bin", "wb"))) { fwrite((void *)(uintptr_t)st.vbuf[0].addr, 1, vsz < st.vbuf[0].size ? vsz : st.vbuf[0].size, f); fclose(f); }
            for (int c = 0; c < 18; c++) {
                if (!((pr->vi->cbuf_mask >> (c + 3)) & 1) || !st.ubuf[0][c].addr) continue;
                snprintf(path, sizeof path, "build/skin/cb%d.bin", c + 3);
                if ((f = fopen(path, "wb"))) { fwrite((void *)(uintptr_t)st.ubuf[0][c].addr, 1, st.ubuf[0][c].size, f); fclose(f); }
            }
            {   // compiler constants (cb1) live after the shader code
                NvnShaderStage *ss = &st.program->stages[0];
                if (ss->const_offset && (f = fopen("build/skin/cb1.bin", "wb"))) {
                    fwrite((void *)(uintptr_t)(ss->code_addr + ss->const_offset), 1, ss->const_size ? ss->const_size : 256, f);
                    fclose(f);
                }
            }
            LOG("[skin] dumped draw inputs of vs %llx to build/skin/\n", (unsigned long long)st.program->stages[0].code_addr);
            extern uint64_t g_skin_watch;  // (renderer_stats: is this cbuf filled later, by the GPU?)
            g_skin_watch = st.ubuf[0][0].addr;
            const uint32_t *w = (const uint32_t *)(uintptr_t)g_skin_watch;
            LOG("[skin] cb3 %llx at draw: row8 %08x %08x row9 %08x %08x\n", (unsigned long long)g_skin_watch, w[32], w[33], w[36], w[37]);
        }
    }
    if (getenv("LARECOMP_DEBUG_VSCB"))
        LOG("[draw]   vbuf0 %llx+%llu vbuf1 %llx+%llu\n", (unsigned long long)st.vbuf[0].addr,
            (unsigned long long)st.vbuf[0].size, (unsigned long long)st.vbuf[1].addr, (unsigned long long)st.vbuf[1].size);
    if (getenv("LARECOMP_DEBUG_VSCB") && st.vbuf[0].addr) {  // first vertices: bytes of attrs 11 / 12
        const uint8_t *v = (const uint8_t *)(uintptr_t)st.vbuf[0].addr;
        uint32_t stride = st.stream[0][0];
        for (int i = 0; i < 4 && stride; i++) {
            const uint8_t *p = v + (uint64_t)i * stride;
            LOG("[draw]   v%d pos %.3f %.3f %.3f", i, ((const float *)p)[0], ((const float *)p)[1], ((const float *)p)[2]);
            for (int a = 11; a <= 12; a++) {
                if (!((pr->vi->attr_mask >> a) & 1)) continue;
                const uint8_t *q = p + (st.attrib[a] >> 13);
                LOG(" a%d %u %u %u %u", a, q[0], q[1], q[2], q[3]);
            }
            LOG("\n");
        }
    }
    for (int c = 0; c < 18 && getenv("LARECOMP_DEBUG_VSCB"); c++) {  // dynamically indexed vertex cbufs (bone palettes)
        if (!((pr->vi->cbuf_dynamic >> (c + 3)) & 1) || !st.ubuf[0][c].addr) continue;
        const float *u = (const float *)(uintptr_t)st.ubuf[0][c].addr;
        LOG("[draw]   vcb%d %llx size %u:", c, (unsigned long long)st.ubuf[0][c].addr, (unsigned)st.ubuf[0][c].size);
        int n = (int)(st.ubuf[0][c].size / 4), bad = 0, ident = 0;
        float mx = 0;
        for (int k = 0; k < n; k++) {
            float f = u[k];
            if (f != f || f > 1e6f || f < -1e6f) bad++;
            else if (fabsf(f) > mx) mx = fabsf(f);
        }
        for (int m = 0; m + 16 <= n; m += 16) {
            bool id = true;
            for (int k = 0; k < 16; k++) if (u[m + k] != ((k % 5) == 0 ? 1.0f : 0.0f)) id = false;
            ident += id;
        }
        LOG(" floats %d, non-finite/huge %d, max |x| %.4g, identity 4x4 %d of %d | m1:", n, bad, mx, ident, n / 16);
        for (int k = 16; k < 32 && k < n; k++) LOG(" %.4g", u[k]);
        LOG(" | m5:");
        for (int k = 80; k < 96 && k < n; k++) LOG(" %.4g", u[k]);
        LOG("\n");
    }
    if (getenv("LARECOMP_DEBUG_DATA")) {  // vertex streams 0-1 and fragment cbufs: the first floats
        for (int s = 0; s < 2; s++) {
            if (!st.vbuf[s].addr) continue;
            const float *v = (const float *)(uintptr_t)st.vbuf[s].addr;
            LOG("[draw]   vbuf%d %llx size %llu:", s, (unsigned long long)st.vbuf[s].addr, (unsigned long long)st.vbuf[s].size);
            for (int k = 0; k < 16; k++) LOG(" %.4g", v[k]);
            LOG("\n");
        }
        for (int c = 0; c < 18; c++) {
            if (!((pr->fi->cbuf_mask >> c) & 1) || !st.ubuf[1][c].addr) continue;
            const float *u = (const float *)(uintptr_t)st.ubuf[1][c].addr;
            LOG("[draw]   fcb%d %llx size %u:", c, (unsigned long long)st.ubuf[1][c].addr, (unsigned)st.ubuf[1][c].size);
            for (int k = 0; k < 96 && k * 4 < (int)st.ubuf[1][c].size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
    }
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {  // fragment textures: guest size / format, allocation, scale
        NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
        const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
        if (!t) { LOG("[draw]   tex%d: none\n", i); continue; }
        LOG("[draw]   tex%d: %p %ux%u f%#x lv %u -> %ux%u scale %.4f %.4f%s | target %u flags %#x layers %u bh %u addr %llx "
            "stride %llu%s%s '%s'\n", i, (void *)t, t->width, t->height, t->format,
            b ? b->levels : 0, b ? b->width : 0, b ? b->height : 0, b ? b->sx : 0, b ? b->sy : 0, b && b->scalable ? " scalable" : "",
            t->target, t->flags, t->layout.layers, t->layout.block_height_log2, (unsigned long long)t->addr,
            (unsigned long long)t->stride, b && b->gpu_owned ? " gpu-owned" : "", b && b->dynamic ? " dynamic" : "", t->label);
    }
}

// ================================================================== texture copies
typedef struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; } TexRefRec;

static void copy_region_dims(NvnTexture *t, const TexRefRec *ref, const FormatInfo **fi, int32_t rg[6], int *level,
                             uint64_t *row, uint64_t *img) {
    *fi = nvn_format_info(t->format);
    memcpy(rg, ref->region, sizeof ref->region);
    if (!rg[3]) rg[3] = (int32_t)t->width;
    if (!rg[4]) rg[4] = (int32_t)t->height;
    if (!rg[5]) rg[5] = 1;
    *level = ref->has_view && (ref->view.setmask & 1) ? ref->view.base_level : 0;
    uint32_t wb = ((uint32_t)rg[3] + (*fi)->bw - 1) / (*fi)->bw, hb = ((uint32_t)rg[4] + (*fi)->bh - 1) / (*fi)->bh;
    *row = st.copy_row_stride ? st.copy_row_stride : (uint64_t)wb * (*fi)->bpb;
    *img = st.copy_image_stride ? st.copy_image_stride : *row * hb;
}

static void copy_buffer_to_texture(uint64_t src, const TexRefRec *dst) {
    snap_reason = "buf2tex"; snapshots_invalidate(NULL);
    NvnTexture *t = (NvnTexture *)(uintptr_t)dst->tex;
    if (!t || !src) return;
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, dst, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb) return;
    if (getenv("LARECOMP_LOG_BUF2TEX") && (t->format == 0x1 || t->format == 0xd)) {  // (movie planes)
        static int n;
        if (n++ < 12) {
            const uint8_t *p = (const uint8_t *)(uintptr_t)src;
            unsigned sum = 0;
            for (int i = 0; i < 4096; i++) sum += p[(uint64_t)i * 97 % (row * (uint64_t)rg[4] ? row * (uint64_t)rg[4] : 1)];
            LOG("[vk] buf2tex src %llx -> %ux%u fmt %#x region %d,%d %dx%d row %llu sample-sum %u\n",
                (unsigned long long)src, t->width, t->height, t->format, rg[0], rg[1], rg[3], rg[4],
                (unsigned long long)row, sum);
        }
    }
    extern void vk_tex_unscale(NvnTexture *t);
    vk_tex_unscale(t);  // the copied data is full size: no output-resolution scaling for this texture
    TexBackend *b = vk_tex(t, false);  // prior CPU contents uploaded first
    watch_tex(t, "buffer upload", NULL);
    renderer_mark_gpu_owned(t);
    if (b->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) return;
    if (b->astc_bw) {  // (blocks would need the CPU decode: not seen in the game yet)
        static int nlog;
        if (nlog++ < 5) LOG("[vk] buffer -> ASTC texture copy skipped (%ux%u)\n", t->width, t->height);
        return;
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t hb = ((uint32_t)rg[4] + fi->bh - 1) / fi->bh, wb = ((uint32_t)rg[3] + fi->bw - 1) / fi->bw;
    uint32_t out_bpb = fi->bpb == 3 ? 4 : fi->bpb;
    uint32_t nz = (uint32_t)rg[5];
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    uint64_t out_row = (uint64_t)wb * out_bpb;
    VkSpan s = vk_transient(out_row * hb * nz, 16);
    for (uint32_t z = 0; z < nz; z++)
        for (uint32_t y = 0; y < hb; y++) {
            const uint8_t *sr = (const uint8_t *)(uintptr_t)(src + (uint64_t)z * img + (uint64_t)y * row);
            uint8_t *dr = (uint8_t *)s.ptr + ((uint64_t)z * hb + y) * out_row;
            if (out_bpb == fi->bpb) memcpy(dr, sr, out_row);
            else for (uint32_t x = 0; x < wb; x++) { memcpy(dr + x * 4, sr + x * 3, 3); dr[x * 4 + 3] = 255; }
        }
    VkBufferImageCopy c = {.bufferOffset = s.off, .bufferRowLength = wb * fi->bw, .bufferImageHeight = hb * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vkCmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

static void copy_texture_to_buffer(const TexRefRec *src, uint64_t dst) {
    NvnTexture *t = (NvnTexture *)(uintptr_t)src->tex;
    if (!t || !dst || dst < HEAP_BASE) return;
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, src, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb || fi->bpb == 3 || row % fi->bpb) return;
    TexBackend *b = vk_tex(t, false);
    float tsx, tsy;
    renderer_texture_scale(t, &tsx, &tsy);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImage srcimg = b->image;
    VmaAllocation small_mem = NULL;
    if ((tsx != 1 || tsy != 1) && level == 0 && !renderer_format_is_depth(t->format)) {
        // the CPU expects guest-sized texels: filter the scaled allocation down first
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = b->format,
                                .extent = {t->width, t->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &srcimg, &small_mem, NULL));
        vk_image_init_layout(cb, srcimg, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)b->width, (int32_t)b->height, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)t->width, (int32_t)t->height, 1}}};
        vkCmdBlitImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, srcimg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
        vk_full_barrier(cb);
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t nz = (uint32_t)rg[5];
    uint64_t total = img * nz;
    static int sync_rb = -1;
    if (sync_rb < 0) sync_rb = getenv("LARECOMP_READBACK_SYNC") ? atoi(getenv("LARECOMP_READBACK_SYNC")) : 0;
    guest_copy_written(dst, total);  // (dst gets the GPU's result: copies of it taken before are stale)
    {   // an older readback still pending for exactly this range: the game never waited for it (its wait would have
        // landed it before this list ran), so nobody reads it - it is superseded (its buffer freed once the GPU is done)
        pthread_mutex_lock(&g_rb_lock);
        for (int i = 0; i < g_nrb; i++)
            if (g_rb[i].dst == dst && g_rb[i].size == total) { g_rb[i].superseded = true; stat_rb_superseded++; }
        pthread_mutex_unlock(&g_rb_lock);
    }
    readback_sync_range(dst, total);  // (a partly overlapping one lands first)
    // deferred unless a compute store range covers dst (in-place reads there); work recorded before this copy finishes
    // before it anyway (submissions run in order), and a later shader reading guest memory in place waits for the
    // pending readbacks when it is recorded (readbacks_before_gmem)
    if (!sync_rb && !gpuw_overlaps(dst, total)) {
        pthread_mutex_lock(&g_rb_lock);
        bool room = g_nrb < RB_MAX;
        pthread_mutex_unlock(&g_rb_lock);
        if (room) {
            Readback rb = {.seq = vk_pending_seq(), .dst = dst, .size = total};
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
            VmaAllocationInfo info;
            if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &rb.buf, &rb.mem, &info) == VK_SUCCESS) {
                rb.ptr = info.pMappedData;
                VkBufferImageCopy c = {.bufferOffset = 0, .bufferRowLength = (uint32_t)(row / fi->bpb) * fi->bw,
                                       .bufferImageHeight = (uint32_t)(img / row) * fi->bh,
                                       .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                            (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                                       .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                                       .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
                vkCmdCopyImageToBuffer(cb, srcimg, VK_IMAGE_LAYOUT_GENERAL, rb.buf, 1, &c);
                vk_full_barrier(cb);
                if (small_mem) rb.img = srcimg, rb.img_mem = small_mem;
                pthread_mutex_lock(&g_rb_lock);
                g_rb[g_nrb++] = rb;
                pthread_mutex_unlock(&g_rb_lock);
                stat_rb_deferred++;
                // submitted now (not waited): whoever needs the result later only waits for a submission already sent,
                // never has to send the one being built (a flush in the middle of a draw would leave its command buffer)
                extern void renderer_submit(void);
                renderer_submit();
                return;
            }
        }
    }
    stat_rb_sync++;
    VkSpan out = vk_readback(total);
    VkBufferImageCopy c = {.bufferOffset = out.off, .bufferRowLength = (uint32_t)(row / fi->bpb) * fi->bw,
                           .bufferImageHeight = (uint32_t)(img / row) * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vkCmdCopyImageToBuffer(cb, srcimg, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();  // the CPU may read the result right after the fence
    memcpy((void *)(uintptr_t)dst, out.ptr, total);
    if (small_mem) vmaDestroyImage(g_vk.vma, srcimg, small_mem);
}

// LARECOMP_WATCH_TEX=<guest addr, hex>: every write to that texture (render target, copy destination, buffer upload)
// logged with the frame - who fills a texture the debug frame only shows being read (docs/DEPURACAO.md).
static void watch_tex(NvnTexture *t, const char *what, const NvnTexture *from) {
    static uint64_t addr = ~0ull;
    if (addr == ~0ull) { const char *e = getenv("LARECOMP_WATCH_TEX"); addr = e ? strtoull(e, NULL, 16) : 0; }
    if (!addr || !t || t->addr != addr) return;
    extern uint64_t vk_frame_counter(void);
    const NvnProgram *p = st.program;
    LOG("[watch] frame %llu: %s %ux%u f%#x (vs %llx fs %llx, from %ux%u f%#x)\n", (unsigned long long)vk_frame_counter(),
        what, t->width, t->height, t->format, p ? (unsigned long long)p->stages[0].code_addr : 0ull,
        p ? (unsigned long long)p->stages[1].code_addr : 0ull, from ? from->width : 0, from ? from->height : 0,
        from ? from->format : 0);
}

static void copy_texture(const TexRefRec *src, const TexRefRec *dst, uint64_t flags) {
    snap_reason = "tex2tex"; snapshots_invalidate(NULL);
    NvnTexture *st_ = (NvnTexture *)(uintptr_t)src->tex, *dt = (NvnTexture *)(uintptr_t)dst->tex;
    if (!st_ || !dt) return;
    watch_tex(dt, "copy destination", st_);
    vk_end_pass();
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each copy its own submission, waited on)
    gp_mark(vk_cmd(), "copy texture %ux%u -> %ux%u", st_->width, st_->height, dt->width, dt->height);
    dl_note("copy texture %ux%u f%#x -> %ux%u f%#x (%p -> %p)", st_->width, st_->height, st_->format, dt->width, dt->height,
            dt->format, (void *)st_, (void *)dt);
    TexBackend *s = vk_tex(st_, false), *d = vk_tex(dt, true);
    renderer_mark_gpu_owned(dt);
    if (renderer_texture_presented(dt) && st_->width == dt->width && st_->height == dt->height) renderer_texture_mark_presented(st_);
    renderer_texture_first_write(dt);
    renderer_texture_level_fresh(dt);
    const int32_t *sr = src->region, *dr = dst->region;
    float ssx, ssy, dsx, dsy;
    renderer_texture_scale(st_, &ssx, &ssy);
    renderer_texture_scale(dt, &dsx, &dsy);
    int sx0 = (int)lroundf(sr[0] * ssx), sy0 = (int)lroundf(sr[1] * ssy);
    int dx0 = (int)lroundf(dr[0] * dsx), dy0 = (int)lroundf(dr[1] * dsy);
    int sw = sr[3] ? (int)lroundf(sr[3] * ssx) : (int)s->width, sh = sr[4] ? (int)lroundf(sr[4] * ssy) : (int)s->height;
    int dw = dr[3] ? (int)lroundf(dr[3] * dsx) : (int)d->width, dh = dr[4] ? (int)lroundf(dr[4] * dsy) : (int)d->height;
    uint32_t slv = src->has_view && (src->view.setmask & 1) ? src->view.base_level : 0;
    uint32_t dlv = dst->has_view && (dst->view.setmask & 1) ? dst->view.base_level : 0;
    uint32_t sly = src->has_view && (src->view.setmask & 2) ? src->view.min_layer : 0;
    uint32_t dly = dst->has_view && (dst->view.setmask & 2) ? dst->view.min_layer : 0;
    if (slv >= s->levels || dlv >= d->levels || sly >= s->layers || dly >= d->layers) return;
    {   // whole-image regions are the level's extent, and every region stays inside its level: scaled allocations
        // (internal resolution below 1x) round guest regions past the edge, and out-of-bounds copies / blits are
        // undefined in Vulkan (garbage that changed every frame on NVIDIA: flicker in the depth pyramid's users)
        int slw = (int)level_extent(s->width, slv), slh = (int)level_extent(s->height, slv);
        int dlw = (int)level_extent(d->width, dlv), dlh = (int)level_extent(d->height, dlv);
        if (!sr[3]) sw = slw - sx0;
        if (!sr[4]) sh = slh - sy0;
        if (!dr[3]) dw = dlw - dx0;
        if (!dr[4]) dh = dlh - dy0;
        if (sx0 + sw > slw) sw = slw - sx0;
        if (sy0 + sh > slh) sh = slh - sy0;
        if (dx0 + dw > dlw) dw = dlw - dx0;
        if (dy0 + dh > dlh) dh = dlh - dy0;
        if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    }
    if (g_debug_frame)
        LOG("[copytex] %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] -> %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] phys (%d,%d %dx%d) -> (%d,%d %dx%d) flags %llx\n",
            (void *)st_, st_->width, st_->height, st_->format, slv, sr[0], sr[1], sr[3], sr[4], s->width, s->height, (void *)dt, dt->width,
            dt->height, dt->format, dlv, dr[0], dr[1], dr[3], dr[4], d->width, d->height, sx0, sy0, sw, sh, dx0, dy0, dw, dh,
            (unsigned long long)flags);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageAspectFlags sa = s->aspect, da = d->aspect;
    // NVN copy flags: 1 linear filter, 2 2D engine, 4 / 8 mirror X / Y (Sword's scene-change snapshot is copied with
    // MIRROR_Y and drawn back with its Y negated: ignoring the flag showed the snapshot upside down)
    bool mirror_x = flags & 4, mirror_y = flags & 8;
    if (s->samples > 1 && d->samples == 1 && sw == dw && sh == dh && s->format == d->format && !(sa & VK_IMAGE_ASPECT_DEPTH_BIT) &&
        !mirror_x && !mirror_y) {
        VkImageResolve rv = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                             .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vkCmdResolveImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
        return;
    }
    if (s->samples != d->samples) return;
    if (sw == dw && sh == dh && s->format == d->format && !mirror_x && !mirror_y) {
        VkImageCopy c = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                         .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vkCmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        return;
    }
    // scaled / converting copy: the game builds its depth pyramid this way (nearest), colour copies filter if asked
    bool depth = (sa | da) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (depth && s->format != d->format) return;
    VkImageBlit bl = {.srcSubresource = {depth ? (sa & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : sa) : sa, slv, sly, 1},
                      .srcOffsets = {{sx0, sy0, 0}, {sx0 + sw, sy0 + sh, 1}},
                      .dstSubresource = {depth ? (da & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : da) : da, dlv, dly, 1},
                      .dstOffsets = {{dx0, dy0, 0}, {dx0 + dw, dy0 + dh, 1}}};
    if (mirror_x) { bl.dstOffsets[0].x = dx0 + dw; bl.dstOffsets[1].x = dx0; }  // (reversed offsets: a mirrored blit)
    if (mirror_y) { bl.dstOffsets[0].y = dy0 + dh; bl.dstOffsets[1].y = dy0; }
    bool linear = (flags & 1) && !depth && !rt_kind(st_->format);
    dl_note("  blit %p lv%u/%u %ux%u (%d,%d %dx%d) -> %p lv%u/%u %ux%u (%d,%d %dx%d) %s", (void *)s->image, slv, s->levels, s->width,
            s->height, sx0, sy0, sw, sh, (void *)d->image, dlv, d->levels, d->width, d->height, dx0, dy0, dw, dh, linear ? "linear" : "nearest");
    vkCmdBlitImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

// ================================================================== present helpers (device.c)
static VkPipelineLayout present_layout;
static VkDescriptorSetLayout present_set_layout;

static VkPipeline present_pipeline(VkFormat fmt) {
    static struct { VkFormat f; VkPipeline p; } cache[4];
    for (int i = 0; i < 4; i++) if (cache[i].p && cache[i].f == fmt) return cache[i].p;
    if (!present_layout) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &present_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &present_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &present_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("present.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader("present.frag"), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = present_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 4; i++) if (!cache[i].p) { cache[i].f = fmt; cache[i].p = p; break; }
    return p;
}

// Scanout shows the bytes the game wrote: sRGB images are read through a UNORM view (no decode), like metal.m.
static VkImageView raw_view(TexBackend *b) {
    VkFormat f = b->format == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM
               : b->format == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : b->format;
    static const VkComponentMapping id = {0};
    return vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, f, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// ---- post passes (Video tab "Anti-aliasing" / "Upscaler"): SMAA 1x at the game image's size, then AMD FSR 1 (EASU
// upscale to the image's size in the window, RCAS sharpen) when it is smaller than its place in the window, as
// metal.m does with MetalFX. Each pass: a fullscreen triangle into an RGBA8 image (gamma-encoded, like the input);
// up to three inputs (bindings 0-2), sampled linear / clamp.
static VkPipelineLayout post_layout;
static VkDescriptorSetLayout post_set_layout;

static VkPipeline post_pipeline(const char *fs) {
    static struct { const char *fs; VkPipeline p; } cache[8];
    for (int i = 0; i < 8; i++) if (cache[i].p && cache[i].fs == fs) return cache[i].p;
    if (!post_layout) {
        VkDescriptorSetLayoutBinding b[3];
        for (uint32_t i = 0; i < 3; i++)
            b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &post_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &post_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &post_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("post.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fs), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = post_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 8; i++) if (!cache[i].p) { cache[i].fs = fs; cache[i].p = p; break; }
    return p;
}

// Post image `i` at w x h (RGBA8, kept until the size changes). 0, 1: SMAA edges / weights, 2: SMAA output,
// 3, 4: FSR EASU / RCAS.
static TexBackend *post_image(int i, uint32_t w, uint32_t h) {
    static TexBackend img[6];  // (0-2 SMAA, 3-4 FSR 1, 5 the colour adjustment)
    TexBackend *d = &img[i];
    if (d->image && (d->width != w || d->height != h)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = VK_FORMAT_R8G8B8A8_UNORM; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

static VkImageView post_view(TexBackend *d) {
    static const VkComponentMapping id = {0};
    return vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// SMAA's lookup textures (runtime/nvn/smaa_data.c): 0 = area (160 x 560 R8G8), 1 = search (64 x 16 R8), uploaded once.
static VkImageView smaa_lut(int i) {
    static TexBackend t[2];
    TexBackend *d = &t[i];
    if (!d->image) {
        extern const unsigned char *smaa_area_tex(void);
        extern const unsigned char *smaa_search_tex(void);
        uint32_t w = i ? 64 : 160, h = i ? 16 : 560, bpp = i ? 1 : 2;
        VkFormat f = i ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        VkCommandBuffer cb = vk_cmd();
        vk_image_init_layout(cb, d->image, d->aspect);
        VkSpan s = vk_transient_copy(i ? smaa_search_tex() : smaa_area_tex(), (uint64_t)w * h * bpp, 16);
        VkBufferImageCopy c = {.bufferOffset = s.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        vkCmdCopyBufferToImage(cb, s.buf, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    return post_view(d);
}

// One post pass (outside the window image's rendering): srcs (bindings 0..n-1) -> dst, fragment shader `fs`.
// clear: start from zero (SMAA's edge pass only writes edge pixels; its weights / output read them).
static void post_pass(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                      bool clear) {
    VkPipeline pipe = post_pipeline(fs);  // (before the rendering begins: created inside it, the NVIDIA driver crashed)
    vk_full_barrier(cb);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(dst),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {dst->width, dst->height}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkDescriptorSet ds = vk_alloc_set(post_set_layout);
    VkDescriptorImageInfo ii[3];
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < n; i++) {
        ii[i] = (VkDescriptorImageInfo){vk_default_sampler(true), srcs[i], VK_IMAGE_LAYOUT_GENERAL};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, post_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)dst->width, (float)dst->height, 0, 1};
    VkRect2D sc = {{0, 0}, {dst->width, dst->height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdPushConstants(cb, post_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
}

// The view the present samples: the game image, or its anti-aliased / upscaled version. ow x oh: the whole image's
// size in the window (its crop is drawn into the viewport).
static VkImageView post_process(VkCommandBuffer cb, TexBackend *b, double ow_d, double oh_d) {
    VkImageView v = raw_view(b);
    // with TAA / DLAA too (LARECOMP_TAA_SMAA=0: not): the game's depth of field, after the resolve, cuts its blur along
    // the depth, which is not anti-aliased - the far edges' stairs come back there; SMAA on the final image smooths them
    static int taa_smaa = -1;
    if (taa_smaa < 0) taa_smaa = getenv("LARECOMP_TAA_SMAA") ? atoi(getenv("LARECOMP_TAA_SMAA")) : 1;
    bool smaa = g_settings.antialias == AA_SMAA || (taa_smaa && taa_wanted() && taa.ok_frame + 2 >= vk_frame_counter());
    uint32_t ow = (uint32_t)lround(ow_d), oh = (uint32_t)lround(oh_d);
    if (ow > 16384) ow = 16384;
    if (oh > 16384) oh = 16384;
    bool fsr = g_settings.upscaler == UPSCALE_FSR1 && (ow > b->width + 1 || oh > b->height + 1);  // (an upscale only)
    float grade[3] = {g_settings.brightness, g_settings.contrast, g_settings.saturation};  // (F12 menu / start screen)
    bool color = fabsf(grade[0] - 1) > 0.005f || fabsf(grade[1] - 1) > 0.005f || fabsf(grade[2] - 1) > 0.005f;
    if (!smaa && !fsr && !color) return v;
    extern void vk_frame_suspend(void);
    extern void vk_frame_resume(void);
    vk_frame_suspend();
    if (smaa) {  // SMAA 1x: luma edges -> blending weights -> neighbourhood blending
        float w = (float)b->width, h = (float)b->height;
        float m[8] = {1.0f / w, 1.0f / h, w, h};
        TexBackend *edges = post_image(0, b->width, b->height), *weights = post_image(1, b->width, b->height);
        TexBackend *out = post_image(2, b->width, b->height);
        VkImageView in[3] = {v};
        post_pass(cb, "smaa_edge.frag", in, 1, edges, m, true);
        VkImageView wi[3] = {post_view(edges), smaa_lut(0), smaa_lut(1)};
        post_pass(cb, "smaa_weight.frag", wi, 3, weights, m, true);
        VkImageView bi[3] = {v, post_view(weights)};
        post_pass(cb, "smaa_blend.frag", bi, 2, out, m, false);
        v = post_view(out);
    }
    if (color) {  // after SMAA (its edges are found on the game's own colours), before FSR (at the smaller size)
        float pc[8] = {grade[0], grade[1], grade[2]};
        TexBackend *c = post_image(5, b->width, b->height);
        VkImageView ci[3] = {v};
        post_pass(cb, "color.frag", ci, 1, c, pc, false);
        v = post_view(c);
    }
    if (fsr) {
        float iw = (float)b->width, ih = (float)b->height;
        TexBackend *e = post_image(3, ow, oh), *r = post_image(4, ow, oh);
        float pe[8] = {iw / ow, ih / oh, 0.5f * iw / ow - 0.5f, 0.5f * ih / oh - 0.5f, iw, ih};
        VkImageView ei[3] = {v};
        post_pass(cb, "easu.frag", ei, 1, e, pe, false);
        float pr[8] = {exp2f(-0.2f), 0, 0, 0, (float)ow, (float)oh};  // (sharpness 0.2 stops: AMD's default)
        VkImageView ri_[3] = {post_view(e)};
        post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
        v = post_view(r);
    }
    vk_full_barrier(cb);
    vk_frame_resume();
    {
        static uint32_t piw, pih, pow_, poh;
        static int psm = -1, pfs = -1, pco = -1;
        if (b->width != piw || b->height != pih || ow != pow_ || oh != poh || smaa != psm || fsr != pfs || color != pco)
            LOG("[post] %ux%u%s%s%s -> %ux%u\n", b->width, b->height, smaa ? " smaa" : "", color ? " colour" : "", fsr ? " fsr1" : "", ow, oh);
        piw = b->width; pih = b->height; pow_ = ow; poh = oh; psm = smaa; pfs = fsr; pco = color;
    }
    return v;
}

// Inside the window image's rendering (device.c vk_frame_begin): the game image into (vx, vy, vw, vh).
void vk_present_draw(VkCommandBuffer cb, TexBackend *b, float vx, float vy, float vw, float vh, const float pv[8]) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    VkImageView src = post_process(cb, b, pv[6] > 0 ? vw / pv[6] : vw, pv[7] > 0 ? vh / pv[7] : vh);
    VkPipeline p = present_pipeline(g_vk.sc_format);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), src, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {vx, vy, vw, vh, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    VkViewport full = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &full);
}

// Screenshot of a game texture (presented crop applied to build/frame_* shots), via the present shader into RGBA8.
void renderer_save_texture_tex(TexBackend *b, const char *path) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    extern const float *vk_shot_crop(void);
    float crop[4] = {0, 0, 1, 1};
    if (strstr(path, "frame_")) memcpy(crop, vk_shot_crop(), sizeof crop);
    uint32_t w = (uint32_t)(b->width * crop[2]), h = (uint32_t)(b->height * crop[3]);
    if (!w || !h) return;
    vk_end_pass();
    VkImage img;
    VmaAllocation mem;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &img, &mem, NULL));
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    vk_image_init_layout(cb, img, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView tv;
    VK_CHECK(vkCreateImageView(g_vk.dev, &vci, NULL, &tv));
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = tv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkPipeline p = present_pipeline(VK_FORMAT_R8G8B8A8_UNORM);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), raw_view(b), VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &wr, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    float pv[8] = {renderer_origin_lower_left() && !strstr(path, "window") ? 1.0f : 0.0f, 0, 0, 0, crop[0], crop[1], crop[2], crop[3]};
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    vk_full_barrier(cb);
    VkSpan out = vk_readback((uint64_t)w * h * 4);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    vk_save_png(path, out.ptr, w, h);
    vkDestroyImageView(g_vk.dev, tv, NULL);
    vmaDestroyImage(g_vk.vma, img, mem);
}

// ================================================================== executor (renderer.m exec_list)
static void exec_list(uint64_t handle, int depth) {
    uint64_t p = handle;
    if (depth == 0) st.obj = 0, st.obj_draw = 0;  // (a submitted list starts with no object; its called lists inherit)
    if (depth == 0 && !g_prefetch) g_pf_list_epoch++, g_exec_top = handle;
    if (depth == 0 && !g_prefetch) {
        sh_pump();
        recipes_pump();
        pc_scan();
        if (g_nrb) {  // (deferred readbacks the GPU has finished)
            uint64_t done = 0;
            vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &done);
            readbacks_done(done);
        }
    }  // (pipeline recipes: prewarm what the programs handed over complete)
    for (int guard = 0; guard < 10000000; guard++) {
        CmdRec *r = cmd_rec_at(p);  // (p is a guest address; the record is in the shadow, nvn.h)
        void *x = r->extra;
        if (g_prefetch) {  // (a walk for the list's needs: state and draws only)
            if (!pf_applies(r->op)) { p += CMD_GSIZE(r->size); continue; }
        }
        if (r->op < CMD_COUNT) op_counts[r->op]++;
        if (ip_replaying && ip_skip_op(r->op)) { p += CMD_GSIZE(r->size); continue; }
        switch (r->op) {
        case CMD_END: return;
        case CMD_OBJECT: st.obj = r->x[0]; st.obj_draw = 0; { static int d; if (getenv("LARECOMP_OBJDBG") && vk_frame_counter() == 1450 && d++ < 80) LOG("[objdbg] depth %d object %llx\n", depth,(unsigned long long)r->x[0]); } break;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_SET_DEPTH_CLAMP: {  // (LARECOMP_DEPTH_CLAMP=0: ignored, as before)
            static int on = -1;
            if (on < 0) on = !getenv("LARECOMP_DEPTH_CLAMP") || atoi(getenv("LARECOMP_DEPTH_CLAMP")) != 0;
            st.depth_clamp = on && r->x[0] != 0;
            break;
        }
        case CMD_BARRIER:
            // nvnCommandBufferBarrier(bits): orders draws / dispatches and makes their writes visible to later reads
            // (render targets read as textures, written buffers read as constants). Recorded but ignored until 20/10
            // (falls to default): a draw that samples what an earlier draw of the same pass wrote read stale data.
            // As SkyRecomp (MIT): a nonzero barrier ends the pass (a Vulkan memory barrier can't be inside dynamic
            // rendering), barriers everything, and drops the per-submission constant copies.
            // LARECOMP_NVN_BARRIER=0: ignore them again (A/B).
            {
                static int on = -1;
                if (on < 0) on = !getenv("LARECOMP_NVN_BARRIER") || atoi(getenv("LARECOMP_NVN_BARRIER")) != 0;
                stat_barrier_bits[r->x[0] & 0x1FF]++;  // (the bits the game uses: logged with the stats line)
                if (on && r->x[0]) {
                    vk_end_pass();
                    st.rt_dirty = true;
                    vk_full_barrier(vk_cmd());
                    guest_copy_barrier();
                    stat_nvn_barriers++;
                }
            }
            break;
        case CMD_CALL: {
            const uint64_t *h = x;
            uint64_t obj = st.obj;  // (a called list names objects of its own: they end with it)
            uint32_t obj_draw = st.obj_draw;
            for (uint64_t i = 0; i < r->x[0] && depth < 8; i++) {
                st.obj = obj, st.obj_draw = obj_draw;
                exec_list(h[i], depth + 1);
            }
            st.obj = obj, st.obj_draw = obj_draw;
            break;
        }
        case CMD_BIND_PROGRAM: bind_program_stages((NvnProgram *)(uintptr_t)r->x[0], (uint32_t)r->x[1]); break;
        case CMD_BIND_BLEND: { uint8_t *b = x; memcpy(st.blend[b[0] & 7], b, 8); break; }
        case CMD_BIND_CHANNEL_MASK: memcpy(&st.channel_mask, x, 4); break;
        case CMD_BIND_COLOR: memcpy(st.color_state, x, 4); break;
        case CMD_BIND_DEPTH_STENCIL: memcpy(st.ds, x, 8); break;
        case CMD_BIND_MULTISAMPLE: memcpy(st.ms, x, 0x18); break;
        case CMD_BIND_POLYGON: memcpy(st.poly, x, 4); break;
        case CMD_BIND_VERTEX_ATTRIB:
            st.nattrib = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.attrib, x, st.nattrib * 4);
            break;
        case CMD_BIND_VERTEX_STREAM:
            st.nstream = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.stream, x, st.nstream * 8);
            break;
        case CMD_BIND_VERTEX_BUFFERS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(&st.vbuf[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_UNIFORM_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 18 && r->x[0] < 6; i++)
                memcpy(&st.ubuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_STORAGE_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 16 && r->x[0] < 6; i++)
                memcpy(&st.sbuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_TEXTURES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 32 && r->x[0] < 6; i++)
                memcpy(&st.tex[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_BIND_IMAGES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 8 && r->x[0] < 6; i++)
                memcpy(&st.img[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_TEXTURE_POOL: st.tpool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_SAMPLER_POOL: st.spool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_RENDER_TARGETS: {
            const struct { uint64_t tex[8]; NvnTextureView views[8]; uint8_t has_view[8]; uint64_t depth;
                           NvnTextureView dview; uint8_t has_dview; } *rt = x;
            RtState prev = st.rt;  // the same targets again: keep the pass (a restart = full barrier + reload)
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = (int)r->x[0] > 8 ? 8 : (int)r->x[0];
            for (int i = 0; i < st.rt.ncolor; i++) {
                st.rt.color[i] = (NvnTexture *)(uintptr_t)rt->tex[i];
                st.rt.cview[i] = rt->views[i];
                st.rt.has_cview[i] = rt->has_view[i];
            }
            st.rt.depth = (NvnTexture *)(uintptr_t)rt->depth;
            st.rt.dview = rt->dview;
            st.rt.has_dview = rt->has_dview;
            static int always = -1;
            if (always < 0) always = getenv("LARECOMP_VK_PASS_RESTART") != NULL;
            if (always || memcmp(&prev, &st.rt, sizeof prev)) st.rt_dirty = true;
            break;
        }
        case CMD_SET_VIEWPORTS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.viewport[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_SCISSORS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.scissor[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_VIEWPORT_SWIZZLES:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) {
                memcpy(st.vswz[r->x[0] + i], (uint8_t *)x + i * 16, 16);
                const int32_t *s = st.vswz[r->x[0] + i];
                static int logged;  // (only Y flips are done: X / Z / W swizzles are logged)
                if ((s[0] != 0 || (s[1] != 2 && s[1] != 3) || s[2] != 4 || s[3] != 6) && logged++ < 4)
                    LOG("[nvn] viewport %llu swizzle %d %d %d %d (only NEGATIVE_Y is applied)\n",
                        (unsigned long long)(r->x[0] + i), s[0], s[1], s[2], s[3]);
            }
            break;
        case CMD_SET_DEPTH_RANGES:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.depth_range[r->x[0] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_BLEND_COLOR: memcpy(st.blend_color, x, 16); break;
        case CMD_SET_STENCIL: {
            int face = (int)r->x[0];
            for (int f = 0; f < 2; f++) {
                if (!(face & (1 << f))) continue;
                if (r->x[6] == 0) st.stencil_ref[f] = (uint32_t)r->x[1];
                else if (r->x[6] == 1) st.stencil_mask[f] = (uint32_t)r->x[1];
                else st.stencil_vmask[f] = (uint32_t)r->x[1];
            }
            break;
        }
        case CMD_SET_POLYGON_OFFSET: memcpy(st.poly_offset, r->f, sizeof st.poly_offset); break;
        case CMD_SET_PRIMITIVE_RESTART: st.prim_restart = r->x[0] != 0; st.restart_index = (uint32_t)r->x[1]; break;
        case CMD_SET_COPY_STRIDES: st.copy_row_stride = r->x[0]; st.copy_image_stride = r->x[1]; break;
        case CMD_CLEAR_COLOR: {
            snap_reason = "clear-color"; snapshots_invalidate(NULL);
            int index = (int)r->x[0];
            if (getenv("LARECOMP_LOG_CLEARS") && index < st.rt.ncolor && st.rt.color[index])
                LOG("[cclear] frame %llu idx %d %ux%u f%#x mask %#x color %.3f %.3f %.3f %.3f\n", (unsigned long long)renderer_present_count(),
                    index, st.rt.color[index]->width, st.rt.color[index]->height, st.rt.color[index]->format, (uint32_t)r->x[2],
                    ((const float *)x)[0], ((const float *)x)[1], ((const float *)x)[2], ((const float *)x)[3]);
            if (index < st.rt.ncolor && st.rt.color[index]) clear(index, (int)r->x[5], x, (uint32_t)r->x[2], false, 0, false, 0, 0);
            break;
        }
        case CMD_CLEAR_TEXTURE: {
            // ClearTexture(cmd, texture, view, region, color, mask): a render-target clear of the texture alone (renderer.m)
            const struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; float color[4]; } *ct = x;
            NvnTexture *t = (NvnTexture *)(uintptr_t)ct->tex;
            if (!t) break;
            snap_reason = "clear-texture"; snapshots_invalidate(NULL);
            RtState saved = st.rt;
            int32_t sc[4];
            memcpy(sc, st.scissor[0], sizeof sc);
            vk_end_pass();
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = 1;
            st.rt.color[0] = t;
            if (ct->has_view) { st.rt.cview[0] = ct->view; st.rt.has_cview[0] = 1; }
            st.rt_dirty = true;
            const int32_t *rg = ct->region;
            if (rg[3] > 0 && rg[4] > 0) { st.scissor[0][0] = rg[0]; st.scissor[0][1] = rg[1]; st.scissor[0][2] = rg[3]; st.scissor[0][3] = rg[4]; }
            else { st.scissor[0][0] = st.scissor[0][1] = 0; st.scissor[0][2] = st.scissor[0][3] = 16384; }
            clear(0, (int)r->x[6], ct->color, (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 0xF, false, 0, false, 0, 0);
            vk_end_pass();
            st.rt = saved;
            memcpy(st.scissor[0], sc, sizeof sc);
            st.rt_dirty = true;
            break;
        }
        case CMD_CLEAR_DEPTH_STENCIL:
            snap_reason = "clear-depth"; snapshots_invalidate(NULL);
            if (st.rt.depth) {
                bool has_stencil = renderer_format_has_stencil(st.rt.depth->format);
                clear(-1, 0, NULL, 0, r->x[0] != 0, r->f[0], has_stencil && r->x[2] != 0, (uint32_t)r->x[1], (uint32_t)r->x[2]);
            }
            break;
        case CMD_DISPATCH: dispatch_compute((uint32_t)r->x[0], (uint32_t)r->x[1], (uint32_t)r->x[2]); break;
        case CMD_DRAW_ARRAYS: if (g_prefetch) pf_draw(r); else draw(r, false); break;
        case CMD_DRAW_ELEMENTS: if (g_prefetch) pf_draw(r); else draw(r, true); break;
        case CMD_UPDATE_UNIFORM_BUFFER:
            readback_sync_range(r->x[0] + r->x[2], r->x[3]);
            memcpy((void *)(uintptr_t)(r->x[0] + r->x[2]), x, r->x[3]);
            guest_copy_written(r->x[0] + r->x[2], r->x[3]);  // (cbuf copies shared within the submission may be stale now)
            break;
        case CMD_FENCE_SYNC: {
            static int fence_flush = -1;
            if (fence_flush < 0) fence_flush = getenv("LARECOMP_FENCE_FLUSH") ? 1 : 0;
            if (fence_flush) renderer_flush();
            extern void nvn_sync_set_seq(uint64_t guest, uint64_t seq);
            // (x[0] the sync - x[1] is the condition. With the render thread the sync took this list's ticket when the
            // game submitted it: objects.c nvn_list_submitted)
            if (r->x[0] && !rt_on_render_thread()) nvn_sync_set_seq(r->x[0], renderer_fence_seq());
            break;
        }
        case CMD_REPORT_COUNTER: {
            // The game times its GPU work with these (two per frame) and drops to 30 fps when that looks slow;
            // the value is our translation time, not GPU time. LARECOMP_VK_COUNTER=frozen: the same timestamp for
            // every counter within a presented frame (experiment).
            uint64_t *out = (uint64_t *)(uintptr_t)r->x[1];
            uint32_t type = (uint32_t)r->x[0];
            static uint32_t seen;
            if (type < 32 && !(seen & (1u << type))) { seen |= 1u << type; LOG("[nvn] ReportCounter type %u\n", type); }
            static int frozen = -1;  // 1: frozen; 2: ideal (LARECOMP_VK_COUNTER=ideal: a 33.3 ms frame, 1 ms per counter in it)
            if (frozen < 0) { const char *e = getenv("LARECOMP_VK_COUNTER"); frozen = !e ? 0 : !strcmp(e, "frozen") ? 1 : !strcmp(e, "ideal") ? 2 : 0; }
            static uint64_t frame_seen = ~0ull, frame_ts, in_frame;
            extern uint64_t vk_frame_counter(void);
            if (vk_frame_counter() != frame_seen) { frame_seen = vk_frame_counter(); frame_ts = SDL_GetTicksNS(); in_frame = 0; }
            uint64_t v = frozen == 1 ? frame_ts : frozen == 2 ? frame_seen * 33333333ull + in_frame * 1000000ull : SDL_GetTicksNS();
            in_frame++;
            if (out) { out[0] = 0; out[1] = v; }
            static int clog = -1;  // LARECOMP_COUNTER_LOG=1: each counter, with the draws recorded since the previous one
            if (clog < 0) clog = getenv("LARECOMP_COUNTER_LOG") != NULL;
            static uint64_t last_draws;
            if (clog) LOG("[counter] frame %llu #%llu value %llu draws %llu\n", (unsigned long long)frame_seen, (unsigned long long)in_frame,
                          (unsigned long long)v, (unsigned long long)(g_draws_total - last_draws));
            last_draws = g_draws_total;
            break;
        }
        case CMD_COPY_TEXTURE_TO_TEXTURE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], r->x[6]); break; }
        case CMD_DOWNSAMPLE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], 1); break; }
        case CMD_COPY_BUFFER_TO_TEXTURE: copy_buffer_to_texture(r->x[0], x); break;
        case CMD_COPY_TEXTURE_TO_BUFFER: copy_texture_to_buffer(x, r->x[3]); break;
        case CMD_COPY_BUFFER_TO_BUFFER: {
            // A buffer copy is done by this thread, in command order. It used to wait for the GPU to go idle first
            // (submit + wait), ~230 times a frame in the overworld: 0.8 s of every 2 s spent waiting, the GPU idle 85%.
            // Only a range the GPU may touch in place needs that (gpu_may_touch); the rest is already ordered.
            // LARECOMP_COPY_FLUSH=1: always flush (the old behaviour, A/B).
            static int always = -1;
            if (always < 0) always = getenv("LARECOMP_COPY_FLUSH") ? atoi(getenv("LARECOMP_COPY_FLUSH")) : 0;
            readback_sync_range(r->x[0], r->x[2]);
            readback_sync_range(r->x[1], r->x[2]);
            if (always || gpu_may_touch(r->x[0], r->x[2]) || gpu_may_touch(r->x[1], r->x[2])) {
                static int nlog;
                if (getenv("LARECOMP_COPY_LOG") && nlog < 40) {
                    nlog++;
                    LOG("[copy] flush %llx -> %llx (%llu bytes): gpuw src %d dst %d, gmem pending %d\n", (unsigned long long)r->x[0],
                        (unsigned long long)r->x[1], (unsigned long long)r->x[2], gpuw_overlaps(r->x[0], r->x[2]),
                        gpuw_overlaps(r->x[1], r->x[2]), g_gmem_seq != 0);
                }
                renderer_flush();
                stat_copy_flushed++;
            }
            else stat_copy_inline++;
            memmove((void *)(uintptr_t)r->x[1], (void *)(uintptr_t)r->x[0], r->x[2]);
            guest_copy_written(r->x[1], r->x[2]);  // (copies of guest memory shared within the submission may be stale now)
            sh_written(r->x[1], r->x[2]);  // (shader code copied in: its cached hash is stale)
            break;
        }
        default: break;
        }
        p += CMD_GSIZE(r->size);
    }
}

void backend_execute(uint64_t handle) {
    if (ENV("LARECOMP_EXP_NOEXEC")) return;  // (research: the game's chain without the backend's work; the picture stays blank)
    PROF_T0();
    if (ip_enabled()) ip_note_list(handle);
    exec_list(handle, 0);
    PROF_ADD(prof_exec);
    if (prof) cum_exec += SDL_GetTicksNS() - prof_t0_;
    static int sync_submit = -1;
    if (sync_submit < 0) sync_submit = getenv("LARECOMP_SYNC_SUBMIT") ? 1 : 0;
    if (sync_submit) renderer_flush();
    else renderer_submit();
}

// ================================================================== frame interpolation (prototype)
// LARECOMP_INTERP=1: 60 fps from the game's 30 by drawing an in-between frame (docs/letsgo/PESQUISA-60FPS.md). The game's
// logic advances one step per frame, so the frame rate can't simply be raised; instead, like RT64 (Zelda64Recomp)
// and Ship of Harkinian, frame N's draws are run a second time with their constant buffers (camera, world, bone
// matrices) blended halfway between N-1 and N, and that image is shown half a frame before N.
//  - recording: the command lists submitted in a frame (and a hash of their records), the exec state at the frame's
//    start, and a copy of every vertex / fragment constant buffer a draw binds (cbuf 3 and up), keyed by stage,
//    binding, the program, vertex buffer 0, the draw's parameters and the occurrence of that key in the frame;
//  - replay (vk_interp_render, at the present): the same lists from the same start state, skipping what writes guest
//    memory (compute, copies to buffers, inline uniform updates, syncs, counters); each constant buffer is the blend of
//    its two copies, word by word, for normal floats close enough (other words, and draws without a match, keep N);
//  - a list whose records changed since it ran (the game reused its memory) cancels that frame's in-between image.
typedef struct { uint64_t key; uint32_t off, size; uint64_t obj; } IpEntry;  // (obj: the game object of the draw)
typedef struct {
    IpEntry *e;
    uint32_t n, cap;
    uint8_t *data;
    uint64_t used, dcap;
    uint32_t *tab;  // open addressing: key -> e index + 1
    uint64_t lists[256], hashes[256];
    int nlists;
    bool overflow, valid;
    uint64_t destroyed;  // g_nvn_destroyed when the frame started
    __typeof__(st) start;
} IpFrame;
extern uint64_t g_nvn_destroyed;  // (objects.c: textures, pools, programs the game finalized)
#define IP_TAB (1u << 16)
static IpFrame ipf[2];
static int ipc;  // ipf[ipc]: the frame being recorded; ipf[ipc ^ 1]: the one before
static float ip_t = 0.5f;
static uint32_t ip_occ[IP_TAB];  // occurrences of each base key in this frame / replay (by slot)
static uint64_t ip_occ_key[IP_TAB];
static uint64_t ip_stats[4];  // blended, kept (no match), words kept, frames replayed
// LARECOMP_INTERP_DIAG=1: what the in-between image is made of (docs/letsgo/PESQUISA-60FPS.md, "Diagnóstico da interpolação")
typedef struct { int stage, k; uint32_t size; uint64_t draws, nomatch, tagged, words, changed, blended, jumps, nonfloat; } IpDiag;
static IpDiag ip_diag[96];
static int ip_ndiag;
static uint64_t ip_rej[8];  // frames without an in-between image: no previous frame, overflow, destroyed, format, list changed
static IpDiag *ip_diag_for(int stage, int k, uint32_t size) {
    for (int i = 0; i < ip_ndiag; i++)
        if (ip_diag[i].stage == stage && ip_diag[i].k == k && ip_diag[i].size == size) return &ip_diag[i];
    if (ip_ndiag >= 96) return &ip_diag[95];
    ip_diag[ip_ndiag] = (IpDiag){.stage = stage, .k = k, .size = size};
    return &ip_diag[ip_ndiag++];
}
static void ip_diag_dump(void) {
    LOG("[interp-diag] rejected frames: no-prev %llu overflow %llu destroyed %llu format %llu list-changed %llu; replayed %llu\n",
        (unsigned long long)ip_rej[0], (unsigned long long)ip_rej[1], (unsigned long long)ip_rej[2], (unsigned long long)ip_rej[3],
        (unsigned long long)ip_rej[4], (unsigned long long)ip_stats[3]);
    for (int i = 0; i < ip_ndiag; i++) {
        const IpDiag *d = &ip_diag[i];
        LOG("[interp-diag] stage %d cbuf %d size %u: binds %llu (tagged %llu, no match %llu) words %llu changed %llu blended %llu jumps %llu nonfloat %llu\n",
            d->stage, d->k, d->size, (unsigned long long)d->draws, (unsigned long long)d->tagged, (unsigned long long)d->nomatch,
            (unsigned long long)d->words, (unsigned long long)d->changed, (unsigned long long)d->blended, (unsigned long long)d->jumps,
            (unsigned long long)d->nonfloat);
    }
    memset(ip_diag, 0, sizeof ip_diag);
    ip_ndiag = 0;
    memset(ip_rej, 0, sizeof ip_rej);
}

static uint64_t ip_mix(uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); return h * 0x100000001B3ull; }

static void ip_occ_reset(void) { memset(ip_occ_key, 0, sizeof ip_occ_key); }

static uint64_t ip_key(int stage, int k) {
    uint64_t base = ip_mix(ip_mix(ip_draw_sig, (uint64_t)stage), (uint64_t)k);
    if (!base) base = 1;
    uint32_t s = (uint32_t)(base ^ (base >> 29)) & (IP_TAB - 1);
    while (ip_occ_key[s] && ip_occ_key[s] != base) s = (s + 1) & (IP_TAB - 1);
    if (!ip_occ_key[s]) { ip_occ_key[s] = base; ip_occ[s] = 0; }
    return ip_mix(base, ip_occ[s]++);
}

static const IpEntry *ip_find(const IpFrame *f, uint64_t key) {
    if (!f->tab) return NULL;
    uint32_t s = (uint32_t)(key ^ (key >> 31)) & (IP_TAB - 1);
    while (f->tab[s]) {
        const IpEntry *e = &f->e[f->tab[s] - 1];
        if (e->key == key) return e;
        s = (s + 1) & (IP_TAB - 1);
    }
    return NULL;
}

static void ip_frame_clear(IpFrame *f) {
    f->n = 0; f->used = 0; f->nlists = 0; f->overflow = false; f->valid = false;
    if (f->tab) memset(f->tab, 0, IP_TAB * sizeof *f->tab);
}

static void ip_record(uint64_t key, const void *data, uint32_t size) {
    IpFrame *f = &ipf[ipc];
    if (!f->tab) f->tab = calloc(IP_TAB, sizeof *f->tab);
    if (f->n >= IP_TAB / 2) { f->overflow = true; return; }
    if (f->n == f->cap) { f->cap = f->cap ? f->cap * 2 : 4096; f->e = realloc(f->e, f->cap * sizeof *f->e); }
    if (f->used + size > f->dcap) { f->dcap = (f->used + size) * 2 + (1 << 20); f->data = realloc(f->data, f->dcap); }
    memcpy(f->data + f->used, data, size);
    f->e[f->n] = (IpEntry){key, (uint32_t)f->used, size, st.obj};
    f->used += size;
    uint32_t s = (uint32_t)(key ^ (key >> 31)) & (IP_TAB - 1);
    while (f->tab[s]) s = (s + 1) & (IP_TAB - 1);
    f->tab[s] = ++f->n;
}

// cbuf_info, constant buffers 3+ of the vertex / fragment stage: recording copies them (returns NULL: bind as usual);
// the replay returns the blended data to bind.
// Objects whose buffers changed too much between N-1 and N to be blended (vk_interp_render's check, before the replay):
// a camera cut, a model swapped (another level of detail, another skeleton) or two draws paired wrongly. Blending them
// mixes unrelated poses - characters stretched across the screen, Pokémon torn apart. They keep frame N's pose.
#define IP_REJ 1024
static uint64_t ip_rej_obj[IP_REJ];
static bool ip_obj_rejected(uint64_t obj) {
    uint32_t s = (uint32_t)((obj >> 4) * 0x9E3779B1u) & (IP_REJ - 1);
    while (ip_rej_obj[s]) { if (ip_rej_obj[s] == obj) return true; s = (s + 1) & (IP_REJ - 1); }
    return false;
}
static const void *ip_cbuf(int stage, int k, uint64_t addr, uint32_t size) {
    if (stage > 1) return NULL;
    uint64_t key = ip_key(stage, k);
    if (!ip_replaying) { ip_record(key, (const void *)(uintptr_t)addr, size); return NULL; }
    const IpEntry *n = ip_find(&ipf[ipc], key), *o = ip_find(&ipf[ipc ^ 1], key);
    static int diag = -1;
    if (diag < 0) diag = getenv("LARECOMP_INTERP_DIAG") ? 1 : 0;
    IpDiag *dg = diag ? ip_diag_for(stage, k, size) : NULL;
    if (dg) { dg->draws++; if (ip_draw_obj) dg->tagged++; if (!o) dg->nomatch++; }
    if (!n) return NULL;
    const uint8_t *nd = ipf[ipc].data + n->off;
    if (!ip_draw_obj || ip_obj_rejected(st.obj ? st.obj : 1)) { ip_stats[1]++; return nd; }  // interface and other draws no object owns: their order and text change between frames, a blend smears them
    if (!o || o->size != n->size) { ip_stats[1]++; return nd; }
    static uint32_t buf[0x10000 / 4];
    const uint32_t *a = (const uint32_t *)(ipf[ipc ^ 1].data + o->off), *b = (const uint32_t *)nd;
    uint32_t nw = n->size / 4;
    // A jump anywhere in the buffer (a bone, a matrix row) keeps all of it at N: blending the rest would pull part of a
    // skeleton or of a matrix halfway and leave the other part, which stretches the model.
    for (uint32_t i = 0; i < nw; i++) {
        if (a[i] == b[i]) continue;
        uint32_t ea = (a[i] >> 23) & 0xFF, eb = (b[i] >> 23) & 0xFF;
        if (!ea || !eb || ea == 0xFF || eb == 0xFF) continue;
        float fa, fb;
        memcpy(&fa, &a[i], 4);
        memcpy(&fb, &b[i], 4);
        bool jump = ip_draw_2d ? fabsf(fb - fa) > 0.3f * fmaxf(fabsf(fa), fabsf(fb)) && fabsf(fb - fa) > 0.02f  // (2D: a big move is a recycled row or glyph, not motion)
                               : fabsf(fb - fa) > 0.5f * (fabsf(fa) + fabsf(fb)) + 4.0f;
        if (jump) {
            if (dg) dg->jumps += nw;
            ip_stats[1]++;
            return nd;
        }
    }
    // Blended a vec4 at a time, all or nothing: a matrix row with some components blended and others kept (one of them
    // crossing zero, or an integer in a float's place) is not a transformation any more - it shears and shrinks the model.
    // Words that are equal need no test; the others must be finite floats (0 counts) without a jump.
    for (uint32_t g = 0; g < nw; g += 4) {
        uint32_t m = nw - g < 4 ? nw - g : 4;
        int bad = 0, changed = 0, jump = 0;
        for (uint32_t i = g; i < g + m; i++) {
            if (a[i] == b[i]) continue;
            changed++;
            uint32_t ea = (a[i] >> 23) & 0xFF, eb = (b[i] >> 23) & 0xFF;
            bool fa_ok = ea != 0xFF && (ea || !(a[i] & 0x7FFFFF)), fb_ok = eb != 0xFF && (eb || !(b[i] & 0x7FFFFF));
            float fa, fb;
            memcpy(&fa, &a[i], 4);
            memcpy(&fb, &b[i], 4);
            if (!fa_ok || !fb_ok) bad++;
            else if (fabsf(fb - fa) > 0.5f * (fabsf(fa) + fabsf(fb)) + 4.0f) jump++;
        }
        if (dg) { dg->words += m; dg->changed += changed; }
        if (!bad && !jump) {
            for (uint32_t i = g; i < g + m; i++) {
                if (a[i] == b[i]) { buf[i] = b[i]; continue; }
                float fa, fb, v;
                memcpy(&fa, &a[i], 4);
                memcpy(&fb, &b[i], 4);
                v = fa + (fb - fa) * ip_t;
                memcpy(&buf[i], &v, 4);
            }
            if (dg) dg->blended += changed;
        } else {
            for (uint32_t i = g; i < g + m; i++) buf[i] = b[i];
            ip_stats[2] += changed;
            if (dg) { dg->jumps += jump ? changed : 0; dg->nonfloat += bad ? changed : 0; }
        }
    }
    if (n->size > nw * 4) memcpy((uint8_t *)buf + nw * 4, nd + nw * 4, n->size - nw * 4);
    ip_stats[0]++;
    return buf;
}

// hash of a command list's records (JUMPs and CALLs followed), to tell whether its memory still holds what ran
static uint64_t ip_list_hash(uint64_t handle, int depth) {
    uint64_t h = 0xcbf29ce484222325ull, p = handle;
    for (int guard = 0; guard < 10000000; guard++) {
        if (p < GUEST_LO || p >= GUEST_HI) return 0;
        const CmdRec *r = cmd_rec_at(p);
        if (r->op >= CMD_COUNT || r->size < sizeof(CmdRec) || r->size > (1u << 20)) return 0;
        const uint64_t *w = (const uint64_t *)r;
        for (uint32_t i = 0; i < r->size / 8; i++) h = ip_mix(h, w[i]);
        if (r->op == CMD_END) return h;
        if (r->op == CMD_JUMP) { p = r->x[0]; continue; }
        if (r->op == CMD_CALL) {
            const uint64_t *hs = (const uint64_t *)r->extra;
            for (uint64_t i = 0; i < r->x[0] && depth < 8; i++) h = ip_mix(h, ip_list_hash(hs[i], depth + 1));
        }
        p += CMD_GSIZE(r->size);
    }
    return 0;
}

static void ip_note_list(uint64_t handle) {
    IpFrame *f = &ipf[ipc];
    if (f->nlists >= 256) { f->overflow = true; return; }
    f->hashes[f->nlists] = ip_list_hash(handle, 0);
    f->lists[f->nlists++] = handle;
}

// The hold image: frame N's presented texture, kept while the replay overwrites it with the in-between image.
static TexBackend ip_hold;
static TexBackend *ip_hold_copy(TexBackend *src) {
    TexBackend *d = &ip_hold;
    if (d->image && (d->width != src->width || d->height != src->height || d->format != src->format)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                                .imageType = VK_IMAGE_TYPE_2D, .format = src->format, .extent = {src->width, src->height, 1},
                                .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = src->format; d->width = src->width; d->height = src->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    d->sx = src->sx; d->sy = src->sy; d->scalable = src->scalable;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .extent = {src->width, src->height, 1}};
    vkCmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    vk_full_barrier(cb);
    return d;
}

// At the present of frame N (presented texture b): when frame N-1 was recorded too and N's lists are intact, renders
// the in-between image into the game's textures (b then holds it) and returns frame N's own image in *hold; false:
// nothing to show before N. Either way, recording moves on to the next frame.
bool vk_interp_render(TexBackend *b, TexBackend **hold) {
    if (!ip_enabled()) {
        // Off (the world, a battle): what was recorded the last time it was on is stale - a menu opened later blended
        // with that old scene (a flash when the menu opened while walking)
        ipf[0].valid = ipf[1].valid = false;
        return false;
    }
    IpFrame *cur = &ipf[ipc], *prev = &ipf[ipc ^ 1];
    bool ok = prev->valid && cur->valid && cur->nlists > 0 && !cur->overflow && !prev->overflow && b && b->image &&
              b->aspect == VK_IMAGE_ASPECT_COLOR_BIT && b->samples <= 1 && cur->destroyed == g_nvn_destroyed;
    bool structural = ok;
    for (int i = 0; ok && i < cur->nlists; i++)
        if (!cur->hashes[i] || ip_list_hash(cur->lists[i], 0) != cur->hashes[i]) ok = false;
    if (!ok) {
        if (!prev->valid || !cur->valid || cur->nlists == 0) ip_rej[0]++;
        else if (cur->overflow || prev->overflow) ip_rej[1]++;
        else if (cur->destroyed != g_nvn_destroyed) ip_rej[2]++;
        else if (!structural) ip_rej[3]++;
        else ip_rej[4]++;
    }
    if (ok) {
        // Per object, before the replay: the share of its changed float words that changed by a large step. Above
        // LARECOMP_INTERP_OBJ_LIMIT (0.2) the object is not blended (ip_obj_rejected); when the rejected objects hold
        // most of the frame's changes (a camera cut), there is no in-between image at all.
        typedef struct { uint64_t obj, changed, big; } ObjAcc;
        static ObjAcc acc[IP_REJ];
        memset(acc, 0, sizeof acc);
        memset(ip_rej_obj, 0, sizeof ip_rej_obj);
        uint64_t changed = 0, big = 0, rej_changed = 0;
        int nrej = 0, nobj = 0;
        for (uint32_t i = 0; i < cur->n; i++) {
            const IpEntry *n = &cur->e[i], *o = ip_find(prev, n->key);
            if (!o || o->size != n->size) continue;
            uint64_t gobj = n->obj ? n->obj : 1;  // (draws without an object: one group)
            const float *fa = (const float *)(prev->data + o->off), *fb = (const float *)(cur->data + n->off);
            uint64_t c = 0, bg = 0;
            for (uint32_t w = 0; w < n->size / 4; w++) {
                float a = fa[w], v = fb[w];
                if (a == v || !isfinite(a) || !isfinite(v)) continue;
                float m = fmaxf(fabsf(a), fabsf(v));
                if (m < 1e-3f || m > 1e7f) continue;
                c++;
                if (fabsf(v - a) > 0.3f * m && fabsf(v - a) > 0.1f) bg++;
            }
            if (!c) continue;
            uint32_t sl = (uint32_t)((gobj >> 4) * 0x9E3779B1u) & (IP_REJ - 1);
            while (acc[sl].obj && acc[sl].obj != gobj) sl = (sl + 1) & (IP_REJ - 1);
            if (!acc[sl].obj) { acc[sl].obj = gobj; nobj++; }
            acc[sl].changed += c, acc[sl].big += bg;
            changed += c, big += bg;
        }
        extern bool fps60_menu(void);
        bool menu = fps60_menu();  // (menus: the cut checks are for battles; a menu screen changes by large steps all the time)
        static double obj_limit = -1;
        if (obj_limit < 0) obj_limit = getenv("LARECOMP_INTERP_OBJ_LIMIT") ? atof(getenv("LARECOMP_INTERP_OBJ_LIMIT")) : 0.2;
        for (int i = 0; i < IP_REJ; i++)
            if (!menu && acc[i].obj && acc[i].changed >= 8 && (double)acc[i].big > obj_limit * (double)acc[i].changed) {
                uint32_t sl = (uint32_t)((acc[i].obj >> 4) * 0x9E3779B1u) & (IP_REJ - 1);
                while (ip_rej_obj[sl]) sl = (sl + 1) & (IP_REJ - 1);
                ip_rej_obj[sl] = acc[i].obj;
                nrej++;
                rej_changed += acc[i].changed;
            }
        if (getenv("LARECOMP_INTERP_DIAG"))
            LOG("[interp-cut] present %llu: objects %d rejected %d (%llu of %llu changed words; big %llu)\n",
                (unsigned long long)renderer_present_count(), nobj, nrej, (unsigned long long)rej_changed, (unsigned long long)changed,
                (unsigned long long)big);
        if (!menu && changed && rej_changed * 10 > changed * 6) { ok = false; ip_rej[5]++; }
    }
    if (ok) {
        vk_end_pass();
        *hold = ip_hold_copy(b);
        __typeof__(st) end = st;
        st = cur->start;
        st.rt_dirty = true;
        ip_replaying = true;
        ip_occ_reset();
        for (int i = 0; i < cur->nlists; i++) exec_list(cur->lists[i], 0);
        vk_end_pass();
        ip_replaying = false;
        st = end;
        st.rt_dirty = true;
        ip_stats[3]++;
        if (ip_stats[3] % 300 == 1)
            LOG("[interp] frame %llu: %llu blended, %llu without a match, %llu words kept (replays %llu)\n",
                (unsigned long long)vk_frame_counter(), (unsigned long long)ip_stats[0], (unsigned long long)ip_stats[1],
                (unsigned long long)ip_stats[2], (unsigned long long)ip_stats[3]);
        if (ip_stats[3] % 300 == 0 && getenv("LARECOMP_INTERP_DIAG")) ip_diag_dump();
        if (ip_stats[3] % 300 == 1)
            LOG("[interp] draws with an object %llu, without %llu\n", (unsigned long long)ip_stats_tagged, (unsigned long long)ip_stats_untagged);
    }
    // the next frame records into the other slot, from the state this one ended with
    ipc ^= 1;
    ip_frame_clear(&ipf[ipc]);
    ipf[ipc].start = st;
    ipf[ipc].destroyed = g_nvn_destroyed;
    ipf[ipc].valid = true;
    ip_occ_reset();
    return ok;
}
