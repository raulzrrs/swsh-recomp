// NVIDIA DLSS / DLAA (anti-aliasing "DLAA" on the Graphics page): on Sword's HDR scene before its bloom (exec.c
// taa_run, where TAA resolves otherwise) - the scene's colour, its depth as R32F, the camera motion vectors, the jitter -
// through NVIDIA's NGX. NGX is loaded at run time from the driver's NGX core (_nvngx.dll, found through
// the NGXCore registry key or the display driver's DriverStore folder); NGX then loads the DLSS
// model, nvngx_dlss.dll, from the executable's folder (NVIDIA's DLSS SDK, github.com/NVIDIA/DLSS). No NVIDIA SDK code
// or header is compiled in: the few NGX types used are declared here from the SDK's documented ABI.
//
// NGX is brought up at start only on an NVIDIA driver with an NGX core and nvngx_dlss.dll next to the executable (its
// device extensions can only be asked for when the device is made); everywhere else nothing of this runs and "DLAA"
// falls back to TAA. Ported from Let's Go Recomp's runtime/nvn/vulkan/dlss.c.
//
// Adapted from TotK Recomp's runtime/vk_dlaa.c (https://github.com/valentderah/totk-pc-recomp, MIT, Copyright (c) 2026
// Valiantsin Dzerakh): the NGX declarations, the parameter object's vtable layout and its check, the core's loading,
// the feature's creation and evaluation parameters.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <SDL3/SDL_timer.h>

#include "vk.h"
#include "dlss.h"
#include "../../rt.h"

typedef int NgxResult;  // NVSDK_NGX_Result: 1 success, 0xBAD00000 | n failures
#define NGX_OK(r) (((unsigned)(r) & 0xFFF00000u) != 0xBAD00000u)
#define NGX_VERSION_API 0x15
#define NGX_FEATURE_SUPERSAMPLING 1
#define NGX_PERFQ_DLAA 5  // NVSDK_NGX_PerfQuality_Value: 0 performance, 1 balanced, 2 quality, 3 ultra performance
typedef struct NgxParam NgxParam;
typedef struct NgxHandle NgxHandle;
typedef struct { void *slot[17]; } NgxParamVtbl;
struct NgxParam { const NgxParamVtbl *vt; };
typedef struct {  // NVSDK_NGX_ImageViewInfo_VK
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned width, height;
} NgxImageView;
typedef struct {  // NVSDK_NGX_Resource_VK
    union { NgxImageView img; struct { VkBuffer buffer; unsigned bytes; } buf; } r;
    int type;  // 0 image view, 1 buffer
    _Bool rw;
} NgxResourceVK;

typedef NgxResult (*PFN_Init)(unsigned long long app, const wchar_t *path, VkInstance, VkPhysicalDevice, VkDevice,
                              PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int version, const void *info);
typedef NgxResult (*PFN_ReqExt)(unsigned *ni, const char ***ie, unsigned *nd, const char ***de);
typedef NgxResult (*PFN_Params)(NgxParam **);
typedef NgxResult (*PFN_DestroyParams)(NgxParam *);
typedef NgxResult (*PFN_Create)(VkCommandBuffer, int feature, NgxParam *, NgxHandle **);
typedef NgxResult (*PFN_Evaluate)(VkCommandBuffer, const NgxHandle *, const NgxParam *, void *callback);
typedef NgxResult (*PFN_Release)(NgxHandle *);

static struct {
    void *core;
    PFN_Init init;
    PFN_ReqExt req_ext;
    PFN_Params caps, alloc_params;
    PFN_DestroyParams destroy_params;
    PFN_Create create;
    PFN_Evaluate evaluate;
    PFN_Release release;
    int tried, loaded, available, failed;
} N;

typedef void (*SetI)(NgxParam *, const char *, int);
typedef void (*SetU)(NgxParam *, const char *, unsigned);
typedef void (*SetF)(NgxParam *, const char *, float);
typedef void (*SetP)(NgxParam *, const char *, void *);
typedef NgxResult (*GetI)(const NgxParam *, const char *, int *);
typedef NgxResult (*GetU)(const NgxParam *, const char *, unsigned *);
typedef NgxResult (*GetF)(const NgxParam *, const char *, float *);
typedef NgxResult (*GetP)(const NgxParam *, const char *, void **);
// The SDK declares Set(u64), Set(float), Set(double), Set(u32), Set(int), Set(ID3D11Resource *), Set(ID3D12Resource *),
// Set(void *), then the Get overloads in the same order; MSVC (the Windows core) lays each group out in reverse.
#ifdef _WIN32
enum { S_P = 0, S_D12, S_D11, S_I, S_U32, S_D, S_F, S_U64, G_P, G_D12, G_D11, G_I, G_U32, G_D, G_F, G_U64 };
#else
enum { S_U64 = 0, S_F, S_D, S_U32, S_I, S_D11, S_D12, S_P, G_U64, G_F, G_D, G_U32, G_I, G_D11, G_D12, G_P };
#endif
static void p_seti(NgxParam *p, const char *n, int v) { ((SetI)p->vt->slot[S_I])(p, n, v); }
static void p_setu(NgxParam *p, const char *n, unsigned v) { ((SetU)p->vt->slot[S_U32])(p, n, v); }
static void p_setf(NgxParam *p, const char *n, float v) { ((SetF)p->vt->slot[S_F])(p, n, v); }
static void p_setp(NgxParam *p, const char *n, void *v) { ((SetP)p->vt->slot[S_P])(p, n, v); }
static int p_geti(const NgxParam *p, const char *n, int *v) { return NGX_OK(((GetI)p->vt->slot[G_I])(p, n, v)); }
static int p_getu(const NgxParam *p, const char *n, unsigned *v) { return NGX_OK(((GetU)p->vt->slot[G_U32])(p, n, v)); }
static int p_getf(const NgxParam *p, const char *n, float *v) { return NGX_OK(((GetF)p->vt->slot[G_F])(p, n, v)); }
static int p_getp(const NgxParam *p, const char *n, void **v) { return NGX_OK(((GetP)p->vt->slot[G_P])(p, n, v)); }

// ---- loading
#ifdef _WIN32
static bool exe_dir_wide(wchar_t *out, size_t cap) {
    DWORD len = GetModuleFileNameW(NULL, out, (DWORD)cap);
    if (!len || len >= cap) return false;
    while (len && out[len - 1] != L'\\' && out[len - 1] != L'/') len--;
    out[len] = 0;
    return len != 0;
}
#endif

static bool core_load(void) {
    if (N.tried) return N.loaded;
    N.tried = 1;
#ifdef _WIN32
    if (getenv("LARECOMP_DLSS") && atoi(getenv("LARECOMP_DLSS")) == 0) return false;  // (off for a test)
    // the DLSS model next to the executable, or nothing is loaded (not every NVIDIA user has it)
    wchar_t model[MAX_PATH];
    if (!exe_dir_wide(model, MAX_PATH) || wcslen(model) + 20 >= MAX_PATH) return false;
    wcscat(model, L"nvngx_dlss.dll");
    if (GetFileAttributesW(model) == INVALID_FILE_ATTRIBUTES) return false;
    wchar_t dir[MAX_PATH];
    DWORD n = sizeof dir;
    bool found = RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", L"FullPath", RRF_RT_REG_SZ,
                              NULL, dir, &n) == ERROR_SUCCESS && wcslen(dir) + 12 < MAX_PATH;
    if (found) wcscat(dir, L"\\_nvngx.dll");
    if (!found) {  // newer drivers have no NGXCore key: the core sits in the display driver's DriverStore folder, the one
                   // its OpenGLDriverName names (the display adapters' class key, one subkey per adapter)
        HKEY cls;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}", 0,
                          KEY_READ, &cls) == ERROR_SUCCESS) {
            wchar_t sub[64];
            for (DWORD i = 0; !found; i++) {
                DWORD ns = 64;
                if (RegEnumKeyExW(cls, i, sub, &ns, NULL, NULL, NULL, NULL) != ERROR_SUCCESS) break;
                wchar_t gl[MAX_PATH];
                DWORD ng = sizeof gl;
                if (RegGetValueW(cls, sub, L"OpenGLDriverName", RRF_RT_REG_SZ | RRF_RT_REG_MULTI_SZ, NULL, gl, &ng) != ERROR_SUCCESS)
                    continue;
                wchar_t *slash = wcsrchr(gl, L'\\');
                if (!slash || (size_t)(slash - gl) + 12 >= MAX_PATH) continue;
                slash[1] = 0;
                wcscat(gl, L"_nvngx.dll");
                if (GetFileAttributesW(gl) != INVALID_FILE_ATTRIBUTES) wcscpy(dir, gl), found = true;
            }
            RegCloseKey(cls);
        }
    }
    if (!found) {
        LOG("[dlss] nvngx_dlss.dll found, but the driver has no NGX core (not an NVIDIA RTX driver): DLSS off\n");
        return false;
    }
    HMODULE m = LoadLibraryW(dir);
    if (!m) { LOG("[dlss] cannot load the driver's _nvngx.dll (%lu): DLSS off\n", GetLastError()); return false; }
    N.core = m;
#define FN(f, name) N.f = (void *)GetProcAddress(m, name)
    FN(init, "NVSDK_NGX_VULKAN_Init_Ext2");
    FN(req_ext, "NVSDK_NGX_VULKAN_RequiredExtensions");
    FN(caps, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    FN(alloc_params, "NVSDK_NGX_VULKAN_AllocateParameters");
    FN(destroy_params, "NVSDK_NGX_VULKAN_DestroyParameters");
    FN(create, "NVSDK_NGX_VULKAN_CreateFeature");
    FN(evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature");
    FN(release, "NVSDK_NGX_VULKAN_ReleaseFeature");
#undef FN
    if (!N.init || !N.req_ext || !N.caps || !N.alloc_params || !N.create || !N.evaluate || !N.release) {
        LOG("[dlss] the NGX core lacks Vulkan entry points: DLSS off\n");
        return false;
    }
    N.loaded = 1;
    return true;
#else
    return false;
#endif
}

static bool listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return true;
    return false;
}

void dlss_instance_exts(const char **ext, uint32_t *n, uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < ni; i++)
        if (!listed(ext, *n, ie[i]) && *n < max) ext[(*n)++] = ie[i];
}

void dlss_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < nd; i++) {
        bool ok = false;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) LOG("[dlss] the device lacks %s, which NGX asks for\n", de[i]);
        else if (!listed(ext, *n, de[i]) && *n < max) ext[(*n)++] = de[i];
    }
}

static bool vtable_ok(void) {  // (a set-then-get round trip before anything relies on the layout)
    NgxParam *p = NULL;
    if (!NGX_OK(N.alloc_params(&p)) || !p) return false;
    int i = 0;
    unsigned u = 0;
    float f = 0;
    void *v = NULL;
    p_seti(p, "LARecomp.Test.I", -12345);
    p_setu(p, "LARecomp.Test.U", 4000000000u);
    p_setf(p, "LARecomp.Test.F", 1.25f);
    p_setp(p, "LARecomp.Test.P", (void *)p);
    bool ok = p_geti(p, "LARecomp.Test.I", &i) && i == -12345 && p_getu(p, "LARecomp.Test.U", &u) && u == 4000000000u &&
              p_getf(p, "LARecomp.Test.F", &f) && f == 1.25f && p_getp(p, "LARecomp.Test.P", &v) && v == (void *)p;
    if (N.destroy_params) N.destroy_params(p);
    if (!ok) LOG("[dlss] NGX parameter layout check failed: DLSS off\n");
    return ok;
}

void dlss_init(void) {
    if (!N.loaded) return;
#ifdef _WIN32
    wchar_t wdir[MAX_PATH];
    if (!exe_dir_wide(wdir, MAX_PATH)) return;
    // 231313132: the application id NVIDIA's DLSS samples use for builds without an NVIDIA-issued id
    NgxResult r = N.init(231313132ull, wdir, g_vk.inst, g_vk.phys, g_vk.dev, vkGetInstanceProcAddr, vkGetDeviceProcAddr,
                         NGX_VERSION_API, NULL);
    if (!NGX_OK(r)) { LOG("[dlss] NGX init failed (%08x): DLSS off\n", (unsigned)r); return; }
    if (!vtable_ok()) return;
    NgxParam *caps = NULL;
    int avail = 0, driver = 0;
    if (NGX_OK(N.caps(&caps)) && caps) {
        p_geti(caps, "SuperSampling.Available", &avail);
        p_geti(caps, "SuperSampling.NeedsUpdatedDriver", &driver);
    }
    N.available = avail != 0;
    LOG("[dlss] NGX up; DLSS %s%s\n", avail ? "available" : "not available on this GPU", driver ? "; the driver needs an update" : "");
#endif
}

bool dlss_available(void) { return N.available && !N.failed; }

// ---- the feature per render / output size, and the image views it reads
// Kept per size (Sword's dynamic resolution moves between the full target and 80% of it at every area change: a
// feature made again each time held the frame 100-240 ms); the least recently used goes when all four are taken.
typedef struct {
    uint32_t w, h, ow, oh;
    int quality;
    NgxHandle *feat;
    uint64_t used;
} Feat;
static Feat FS[4];
static uint64_t feat_clock;
static NgxParam *P;
static VkImageView views[4][4];  // views of the last frames: destroyed when their slot comes round again
static int view_slot;
static uint64_t last_eval_ns;

static VkImageView make_view(VkImage img, VkFormat f, VkImageAspectFlags a) {
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = f, .subresourceRange = {a, 0, 1, 0, 1}};
    VkImageView v = VK_NULL_HANDLE;
    vkCreateImageView(g_vk.dev, &ci, NULL, &v);
    return v;
}

static NgxResourceVK res_image(VkImageView v, VkImage img, VkFormat f, uint32_t w, uint32_t h, int rw) {
    NgxResourceVK r;
    memset(&r, 0, sizeof r);
    r.r.img = (NgxImageView){v, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, f, w, h};
    r.type = 0;
    r.rw = rw;
    return r;
}

static int perf_quality(uint32_t w, uint32_t ow) {  // (the scene's share of the output width)
    if (w >= ow) return NGX_PERFQ_DLAA;
    double k = (double)w / ow;
    return k >= 0.66 ? 2 : k >= 0.57 ? 1 : k >= 0.49 ? 0 : 3;
}

static void fail(const char *what, NgxResult r) {
    LOG("[dlss] %s failed (%08x): DLSS off for this run (TAA instead)\n", what, (unsigned)r);
    N.failed = 1;
}

bool dlss_dispatch(VkCommandBuffer cb, const DlssInput *d) {
    if (!dlss_available() || !d->color || !d->depth || !d->mv || !d->output) {
        static bool said;
        if (!said && dlss_available())
            said = true, LOG("[dlss] not run: an input is missing (color %d depth %d mv %d output %d)\n", d->color != 0, d->depth != 0,
                             d->mv != 0, d->output != 0);
        return false;
    }
    if (!P && !NGX_OK(N.alloc_params(&P))) { fail("parameter allocation", 0); return false; }
    int q = perf_quality(d->rw, d->ow);
    bool reset = d->reset;
    Feat *fe = NULL, *lru = &FS[0];
    for (int i = 0; i < 4 && !fe; i++) {
        Feat *c = &FS[i];
        if (c->feat && c->w == d->rw && c->h == d->rh && c->ow == d->ow && c->oh == d->oh && c->quality == q) fe = c;
        else if (!c->feat || (lru->feat && c->used < lru->used)) lru = c;
    }
    static Feat *last;
    if (fe && fe != last) reset = true;  // (its history is of another stretch of frames)
    if (!fe) {
        fe = lru;
        if (fe->feat) {
            vkDeviceWaitIdle(g_vk.dev);  // (the old feature may still be in flight)
            N.release(fe->feat);
            fe->feat = NULL;
        }
    }
    last = fe;
    fe->used = ++feat_clock;
#define F (*fe)
    if (!F.feat) {
        p_setu(P, "CreationNodeMask", 1);
        p_setu(P, "VisibilityNodeMask", 1);
        p_setu(P, "Width", d->rw);
        p_setu(P, "Height", d->rh);
        p_setu(P, "OutWidth", d->ow);
        p_setu(P, "OutHeight", d->oh);
        p_seti(P, "PerfQualityValue", q);
        // IsHDR | MVLowRes | AutoExposure: the game's linear HDR scene (before its bloom and tone mapping), the motion
        // vectors at the render size, no exposure of our own to give
        p_seti(P, "DLSS.Feature.Create.Flags", (1 << 0) | (1 << 1) | (1 << 6));
        p_seti(P, "DLSS.Enable.Output.Subrects", 0);
        // NVIDIA's model per mode for DLSS 4.5: K (transformer) for DLAA, Quality and Balanced, M for Performance, L for
        // Ultra Performance (NVSDK_NGX_DLSS_Hint_Render_Preset: 1 A ... 11 K, 12 L, 13 M)
        p_setu(P, "DLSS.Hint.Render.Preset.DLAA", 11);
        p_setu(P, "DLSS.Hint.Render.Preset.UltraQuality", 11);
        p_setu(P, "DLSS.Hint.Render.Preset.Quality", 11);
        p_setu(P, "DLSS.Hint.Render.Preset.Balanced", 11);
        p_setu(P, "DLSS.Hint.Render.Preset.Performance", 13);
        p_setu(P, "DLSS.Hint.Render.Preset.UltraPerformance", 12);
        NgxResult r = N.create(cb, NGX_FEATURE_SUPERSAMPLING, P, &F.feat);
        if (!NGX_OK(r) || !F.feat) { fail("DLSS feature creation", r); return false; }
        F.w = d->rw, F.h = d->rh, F.ow = d->ow, F.oh = d->oh, F.quality = q;
        reset = true;
        LOG("[dlss] feature created: %ux%u -> %ux%u, quality value %d (HDR, colour format %d)\n", d->rw, d->rh, d->ow, d->oh, q,
            (int)d->color_fmt);
    }
    view_slot = (view_slot + 1) & 3;
    for (int i = 0; i < 4; i++)
        if (views[view_slot][i]) vkDestroyImageView(g_vk.dev, views[view_slot][i], NULL), views[view_slot][i] = VK_NULL_HANDLE;
    VkImageView *v = views[view_slot];
    v[0] = make_view(d->color, d->color_fmt, VK_IMAGE_ASPECT_COLOR_BIT);
    v[1] = make_view(d->depth, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    v[2] = make_view(d->mv, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    v[3] = make_view(d->output, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!v[0] || !v[1] || !v[2] || !v[3]) return false;
    uint32_t iw = d->iw ? d->iw : d->rw, ih = d->ih ? d->ih : d->rh;
    NgxResourceVK rc = res_image(v[0], d->color, d->color_fmt, iw, ih, 0);
    NgxResourceVK rz = res_image(v[1], d->depth, VK_FORMAT_R32_SFLOAT, iw, ih, 0);
    NgxResourceVK rm = res_image(v[2], d->mv, VK_FORMAT_R16G16B16A16_SFLOAT, iw, ih, 0);
    NgxResourceVK ro = res_image(v[3], d->output, VK_FORMAT_R16G16B16A16_SFLOAT, d->ow, d->oh, 1);
    // NGX's jitter: the content's move along x, against it along y (Let's Go Recomp checked +x, +y with its y-up viewport,
    // where the content moves up; here it moves down). LARECOMP_DLSS_JITTER_SX / _SY: other signs (a test: a wrong
    // axis is visibly softer - both wrong measured 315, one wrong 330 on a still crop)
    static float sx = 0, sy = 0;
    if (!sx) {
        sx = getenv("LARECOMP_DLSS_JITTER_SX") ? (float)atof(getenv("LARECOMP_DLSS_JITTER_SX")) : 1.0f;
        sy = getenv("LARECOMP_DLSS_JITTER_SY") ? (float)atof(getenv("LARECOMP_DLSS_JITTER_SY")) : -1.0f;
    }
    p_setp(P, "Color", &rc);
    p_setp(P, "Output", &ro);
    p_setp(P, "Depth", &rz);
    p_setp(P, "MotionVectors", &rm);
    p_setf(P, "Jitter.Offset.X", d->jx * sx);
    p_setf(P, "Jitter.Offset.Y", d->jy * sy);
    p_setf(P, "Sharpness", 0.0f);
    p_seti(P, "Reset", reset);
    p_setp(P, "DLSS.Input.Bias.Current.Color.Mask", NULL);
    p_setf(P, "MV.Scale.X", d->mv_scale_x);
    p_setf(P, "MV.Scale.Y", d->mv_scale_y);
    p_setu(P, "DLSS.Input.Color.Subrect.Base.X", 0);  // (the scene at the images' top-left)
    p_setu(P, "DLSS.Input.Color.Subrect.Base.Y", 0);
    p_setu(P, "DLSS.Input.Depth.Subrect.Base.X", 0);
    p_setu(P, "DLSS.Input.Depth.Subrect.Base.Y", 0);
    p_setu(P, "DLSS.Input.MV.Subrect.Base.X", 0);
    p_setu(P, "DLSS.Input.MV.Subrect.Base.Y", 0);
    p_setu(P, "DLSS.Output.Subrect.Base.X", 0);
    p_setu(P, "DLSS.Output.Subrect.Base.Y", 0);
    p_setu(P, "DLSS.Render.Subrect.Dimensions.Width", d->rw);
    p_setu(P, "DLSS.Render.Subrect.Dimensions.Height", d->rh);
    p_setf(P, "DLSS.Pre.Exposure", 1.0f);
    p_setf(P, "DLSS.Exposure.Scale", 1.0f);
    uint64_t now = SDL_GetTicksNS();
    double dt = last_eval_ns ? (now - last_eval_ns) / 1e6 : 0;
    last_eval_ns = now;
    p_setf(P, "FrameTimeDeltaInMsec", dt > 0 && dt < 100 ? (float)dt : 16.7f);
    NgxResult r = N.evaluate(cb, F.feat, P, NULL);
    if (!NGX_OK(r)) { fail("DLSS evaluation", r); return false; }
    vk_full_barrier(cb);
    return true;
}
#undef F
