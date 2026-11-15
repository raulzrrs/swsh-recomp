// NVIDIA DLSS / DLAA through the driver's NGX (runtime/nvn/vulkan/dlss.c), on Sword's HDR scene before its bloom
// (exec.c taa_run): the scene's colour, its depth as R32F, the camera motion vectors and the jitter.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    VkImage color, depth, mv, output;  // the scene (its own format), R32F depth, RGBA16F motion vectors, RGBA16F output
    VkFormat color_fmt;
    uint32_t rw, rh, ow, oh;           // render size, output size (the same for DLAA)
    uint32_t iw, ih;                   // the input images' size: the scene is their top-left rw x rh (Sword's dynamic
                                       // resolution draws into part of its targets); the output image is ow x oh
    float jx, jy;                      // this frame's jitter in render pixels: how far the content moved (right, down)
    float mv_scale_x, mv_scale_y;      // the motion vectors to pixels, previous - current
    bool reset;                        // no history (a cut, the first frame)
} DlssInput;

// The extensions NGX asks for, added when the driver has an NGX core and nvngx_dlss.dll is next to the executable.
void dlss_instance_exts(const char **ext, uint32_t *n, uint32_t max);
void dlss_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max);
void dlss_init(void);       // after the device is made
bool dlss_available(void);  // NGX up and DLSS supported on this GPU (and no failure so far)
// records DLSS into cb (outside any rendering); false: not done (TAA takes over)
bool dlss_dispatch(VkCommandBuffer cb, const DlssInput *d);

#ifdef __cplusplus
}
#endif
