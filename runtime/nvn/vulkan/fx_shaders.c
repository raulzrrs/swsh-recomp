// Built-in shaders of the scene effects added after the SSAO (exec.c taa_run): TAA's motion vectors, its resolve and
// its copy back into the scene, the depth as R32F for DLSS. Kept out of shaders.c, whose text is part of the key of
// the shaders prepared on the player's PC (Makefile TRANSLATOR_SRCS: a change there prepares them all again).
// shaders.c builtin_src asks here for the names it does not know.
#include <string.h>
#include "../maxwell/shader.h"

const char *fx_builtin_src(const char *name, int *stage) {
    // Camera motion vectors: each pixel's position, rebuilt from its depth, taken to the previous frame by m = previous
    // view-projection x inverse of this one (the game's matrices, GL style). a.x = 1: depth stored as (z_ndc + 1) / 2;
    // a.y = the sign of NDC y at the bottom of the viewport. v = the scene's viewport in the depth's pixels (Sword's
    // dynamic resolution draws into part of its targets). Out (RGBA16F): xy = this uv - the previous frame's, in uv of
    // the whole texture; z = the pixel's distance in the previous frame / its distance now (disocclusion).
    if (!strcmp(name, "taa_mv.frag")) {
        *stage = STAGE_FRAGMENT;
        return "#version 450\n"
               "layout(set = 0, binding = 0) uniform sampler2D dep;\n"
               "layout(push_constant) uniform P { mat4 m; vec4 a; vec4 v; } pc;\n"
               "layout(location = 0) out vec4 o;\n"
               "void main() {\n"
               "  ivec2 sz = textureSize(dep, 0), p = ivec2(gl_FragCoord.xy);\n"
               "  float d = texelFetch(dep, p, 0).r;\n"
               "  vec2 uv = (vec2(p) + 0.5 - pc.v.xy) / pc.v.zw;\n"
               "  vec2 n = uv * 2.0 - 1.0;\n"
               "  n.y *= pc.a.y;\n"
               "  vec4 q = pc.m * vec4(n, pc.a.x > 0.5 ? d * 2.0 - 1.0 : d, 1.0);\n"
               "  vec2 pn = q.xy / q.w;\n"
               "  pn.y *= pc.a.y;\n"
               "  vec2 pu = pn * 0.5 + 0.5;\n"
               "  o = vec4((uv - pu) * pc.v.zw / vec2(sz), q.w, 1.0);\n"
               "}\n";
    }
    // TAA (Let's Go Recomp's), on the game's HDR scene before its bloom: t = this frame (jittered), h = the history
    // (rgb, a = its distance), mv = the motion vectors, dep = depth. a = the projection (P00, P11, A, B), b = (reset,
    // history weight, depth mode, 0), c = the viewport in uv (x0, y0, x1, y1): history outside it is not taken.
    // Blended in a tone-mapped space (c / (1 + max c): no fireflies from the bright highlights); the velocity of the
    // nearest of the 3x3 (edges keep the foreground's); the history by Catmull-Rom, clipped to the 3x3's YCoCg mean
    // +- 1.25 sigma (variance clipping), dropped where its distance is not the one expected (uncovered).
    // Out: rgb, a = this distance.
    if (!strcmp(name, "taa.frag")) {
        *stage = STAGE_FRAGMENT;
        return "#version 450\n"
               "layout(set = 0, binding = 0) uniform sampler2D t;\n"
               "layout(set = 0, binding = 1) uniform sampler2D h;\n"
               "layout(set = 0, binding = 2) uniform sampler2D mv;\n"
               "layout(set = 0, binding = 3) uniform sampler2D dep;\n"
               "layout(push_constant) uniform P { vec4 a; vec4 b; vec4 c; } pc;\n"
               "layout(location = 0) out vec4 o;\n"
               "float lin(float d) { return pc.a.w / ((pc.b.z > 0.5 ? d * 2.0 - 1.0 : d) + pc.a.z); }\n"
               "vec3 tm(vec3 c) { return c / (1.0 + max(c.r, max(c.g, c.b))); }\n"
               "vec3 itm(vec3 c) { return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-4); }\n"
               "vec3 ycc(vec3 c) { return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }\n"
               "vec3 rgb(vec3 y) { return vec3(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z); }\n"
               "vec3 cat(vec2 uv, vec2 sz) {\n"  // Catmull-Rom, 5 bilinear taps (the corners dropped)
               "  vec2 sp = uv * sz, tp = floor(sp - 0.5) + 0.5, f = sp - tp;\n"
               "  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);\n"
               "  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);\n"
               "  vec2 w12 = w1 + w2, t0 = (tp - 1.0) / sz, t3 = (tp + 2.0) / sz, t12 = (tp + w2 / w12) / sz;\n"
               "  vec3 r = texture(h, vec2(t12.x, t0.y)).rgb * (w12.x * w0.y) + texture(h, vec2(t0.x, t12.y)).rgb * (w0.x * w12.y)\n"
               "         + texture(h, t12).rgb * (w12.x * w12.y) + texture(h, vec2(t3.x, t12.y)).rgb * (w3.x * w12.y)\n"
               "         + texture(h, vec2(t12.x, t3.y)).rgb * (w12.x * w3.y);\n"
               "  float ws = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;\n"
               "  return max(r / ws, vec3(0.0));\n"
               "}\n"
               "void main() {\n"
               "  ivec2 sz = textureSize(t, 0), p = ivec2(gl_FragCoord.xy);\n"
               "  ivec2 dsz = textureSize(dep, 0);\n"
               "  vec3 c = max(texelFetch(t, p, 0).rgb, vec3(0.0));\n"
               "  vec2 dsc = vec2(dsz) / vec2(sz);\n"
               "  float l = lin(texelFetch(dep, min(ivec2(vec2(p) * dsc), dsz - 1), 0).r);\n"
               "  if (pc.b.x > 0.5) { o = vec4(c, l); return; }\n"
               "  vec3 m1 = vec3(0.0), m2 = vec3(0.0);\n"
               "  float dn = 1.0;\n"
               "  ivec2 np = p;\n"
               "  for (int y = -1; y <= 1; y++)\n"
               "    for (int x = -1; x <= 1; x++) {\n"
               "      ivec2 q = clamp(p + ivec2(x, y), ivec2(0), sz - 1);\n"
               "      vec3 s = ycc(tm(max(texelFetch(t, q, 0).rgb, vec3(0.0))));\n"
               "      m1 += s; m2 += s * s;\n"
               "      float d = texelFetch(dep, min(ivec2(vec2(q) * dsc), dsz - 1), 0).r;\n"
               "      if (d < dn) { dn = d; np = q; }\n"
               "    }\n"
               "  m1 /= 9.0;\n"
               "  vec3 sg = sqrt(max(m2 / 9.0 - m1 * m1, vec3(0.0)));\n"
               "  vec4 v = texelFetch(mv, min(ivec2(vec2(np) * vec2(textureSize(mv, 0)) / vec2(sz)), textureSize(mv, 0) - 1), 0);\n"
               "  vec2 uv = (vec2(p) + 0.5) / vec2(sz), pu = uv - v.xy;\n"
               "  if (any(lessThan(pu, pc.c.xy)) || any(greaterThan(pu, pc.c.zw))) { o = vec4(c, l); return; }\n"
               "  ivec2 hs = textureSize(h, 0);\n"
               "  float lh = texelFetch(h, clamp(ivec2(pu * vec2(hs)), ivec2(0), hs - 1), 0).a, le = v.z * l;\n"
               "  float keep = clamp(1.0 - (abs(lh - le) / max(le * 0.02, 2.0) - 1.0), 0.0, 1.0);\n"
               "  vec3 hy = ycc(tm(cat(pu, vec2(hs))));\n"
               "  vec3 lo = m1 - 1.25 * sg, hi = m1 + 1.25 * sg, ce = 0.5 * (hi + lo), ex = 0.5 * (hi - lo) + 1e-4;\n"
               "  vec3 dv = (hy - ce) / ex;\n"
               "  float mx = max(abs(dv.x), max(abs(dv.y), abs(dv.z)));\n"
               "  if (mx > 1.0) hy = ce + (hy - ce) / mx;\n"
               "  vec3 cy = ycc(tm(c));\n"
               "  float w = pc.b.y * keep;\n"
               "  o = vec4(itm(rgb(mix(cy, hy, w))), l);\n"
               "}\n";
    }
    // The resolved image back into the game's scene (the same size), sharpened: b.z = strength (0: a copy). The
    // sharpening (a 4-neighbour unsharp mask, kept within the neighbours' range: no halos) gives back what the
    // resolve's history blurs.
    if (!strcmp(name, "taa_copy.frag")) {
        *stage = STAGE_FRAGMENT;
        return "#version 450\n"
               "layout(set = 0, binding = 0) uniform sampler2D t;\n"
               "layout(push_constant) uniform P { vec4 a; vec4 b; } pc;\n"
               "layout(location = 0) out vec4 o;\n"
               "void main() {\n"
               "  ivec2 sz = textureSize(t, 0), p = min(ivec2(gl_FragCoord.xy), sz - 1);\n"  // (DLAA's output: the scene's part)
               "  vec3 c = max(texelFetch(t, p, 0).rgb, vec3(0.0));\n"
               "  if (pc.b.z > 0.0) {\n"
               "    vec3 n = texelFetch(t, clamp(p + ivec2(0, -1), ivec2(0), sz - 1), 0).rgb;\n"
               "    vec3 s = texelFetch(t, clamp(p + ivec2(0, 1), ivec2(0), sz - 1), 0).rgb;\n"
               "    vec3 e = texelFetch(t, clamp(p + ivec2(1, 0), ivec2(0), sz - 1), 0).rgb;\n"
               "    vec3 w = texelFetch(t, clamp(p + ivec2(-1, 0), ivec2(0), sz - 1), 0).rgb;\n"
               "    vec3 mn = min(c, min(min(n, s), min(e, w))), mx = max(c, max(max(n, s), max(e, w)));\n"
               "    c = clamp(c + pc.b.z * (c - 0.25 * (n + s + e + w)), mn, mx);\n"
               "  }\n"
               "  o = vec4(c, 1.0);\n"
               "}\n";
    }
    // The scene's depth as R32F (DLSS reads it as a colour image)
    if (!strcmp(name, "depth_r32.frag")) {
        *stage = STAGE_FRAGMENT;
        return "#version 450\n"
               "layout(set = 0, binding = 0) uniform sampler2D dep;\n"
               "layout(location = 0) out vec4 o;\n"
               "void main() { o = vec4(texelFetch(dep, ivec2(gl_FragCoord.xy), 0).r, 0.0, 0.0, 1.0); }\n";
    }
    return NULL;
}
