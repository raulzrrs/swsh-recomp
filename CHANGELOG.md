# Changelog

## 0.3.0 (2026-11-15)

Better graphics and a real PC mouse: anti-aliasing with TAA or NVIDIA DLAA, ambient occlusion, sharper sun shadows,
the camera turned by moving the mouse, and the menus' options clicked where they are. On the reference PC (RTX 3080)
all of it holds 60 fps at 1440p and at 4K.

### Graphics
- **Anti-aliasing: TAA and NVIDIA DLAA** (Graphics → Anti-aliasing: Off / SMAA / TAA / DLAA). The scene is jittered a
  fraction of a pixel each frame and resolved against the frames before it along the camera's motion, on the game's
  HDR image before its bloom: edges, fences, windows and foliage stop stair-stepping and shimmering. DLAA is NVIDIA's
  model in TAA's place (RTX cards; NVIDIA's `nvngx_dlss.dll` comes in the zip); without an RTX card and driver DLAA
  runs as TAA. It follows the game's dynamic resolution (the scene in 80% of the image in some areas).
- **Ambient occlusion** (Graphics → Ambient occlusion: Off / Low / Medium / High): soft contact shadows under the
  characters, at the foot of walls, in windows and in the grass (Let's Go Recomp's SSAO, fitted to Sword's scene).
- **Sharper sun shadows**: the shadow map at 2x (default) or 4x (Graphics → Shadow resolution); shadows get a shape
  instead of a blur.

### Mouse
- **The camera follows the mouse**, no button held, like a PC game (Controls → Camera with the mouse: always, while
  the right button is held, or off; Mouse sensitivity sets its speed).
- **Click the menus' options**: in a menu with the game's cursor (the X menu, the bag, the party, submenus...) the
  mouse is a pointer, and a click on an option moves the game's own selection there and confirms it - done with the
  d-pad and A, as a controller would, so the game's logic is untouched. Out of menus the mouse is the camera again,
  by itself.
- Left button A (outside a menu's options: advances text), right B, middle X (the menu), the wheel up / down in lists,
  side buttons L / R (tabs, the box's pages).

### Fixes
- **Mods that replace hundreds of files no longer crash** with `2002-6063` (#17): every replaced file was kept open,
  and past Windows' 512 the next ones failed to open and were read as zeros.
- Fences recorded in the game's command lists are kept: the game waits for the GPU work it fenced (from Let's Go Recomp).
- Compute shaders are built ahead from the ones recorded on this PC: the ~230 ms hold the first time each was used is
  gone after the first session.

### For testers and developers
- `LARECOMP_LOG_FILES=1` names a mod's files too (`[fs] read (mod) <path>`); a mod file that cannot be opened is logged.
- Soft sun shadows (PCSS) are in, experimental and off (`pcss = <percent>` in larecomp.ini): not yet fitted to Sword's map.
- `tests/scenarios/xmenu*.json`, `LARECOMP_UI_CLICK` (a click in a scenario), `LARECOMP_UI_LOG` (a frame's menu panes).

## 0.2.0 (2026-11-14)

Smooth 60 fps in the open world, battles and Max Raids on the reference PC (Core i5-14600KF, RTX 3080, 2560x1440):
a Wild Area save that ran at 45 fps with a ~160 ms stall every ~2 seconds now holds 60 fps with a 17 ms 99th
percentile. Shaders are prepared on the start screen, and most of the open issues are fixed. The game code is rebuilt
by itself when this version is unzipped over 0.1.0.

### Performance
- **The log no longer stalls the game.** Every log line was written to the disk on the game thread that logged it;
  the game brings its network up and down every ~1.7 s, logging ~50 lines each time, and a slow disk, an antivirus or
  a console made each of those a stall of up to ~160 ms. Lines now go to a writer thread of their own; a crash still
  writes the last ones first.
- **Shaders are prepared on the start screen** (all but two CPU threads, below normal priority), with the share done
  and the time left above Play. Playing before it ends asks first, and the preparation then steps aside (idle
  priority, fewer threads).
- **Prepared shaders are kept** across rebuilds and updates until the shader translator itself changes (it was keyed
  by the build time: every new version prepared everything again).
- The program, pipeline and compute program tables grow instead of filling up: the render thread could spin for
  good once one was full (a freeze after a long session), and a compute program past the 256th was compiled again on
  every use.

### Crashes and freezes
- **The shader preparation no longer crashes in a loop at the same percentage** (#1, #6): a shader the NVIDIA driver's
  compiler crashes on is skipped at the next start.
- **GPU "device lost" in the starter cutscene on RTX 50 cards** (#2, #14): depth-compare lookups without a depth texture
  get a depth stand-in, and guarded forward branches in shaders become ifs (both from Let's Go Recomp, its #24).
- **The Battle Tower no longer crashes** (#7): the system's word filter (nn::ngc) is answered on the host; the game asked
  for system data the port doesn't have.
- **An incomplete game image is refused** with a clear message instead of crashing in the game (#10: a mod's romfs had
  been set up as the game's).

### Graphics
- **Invisible text on Intel UHD and other small GPUs** (#8, probably #3): when the driver refuses to import the whole
  game heap, a smaller part is imported instead of none (from Let's Go Recomp, its #49).
- The menu's font texture is made at the first frame (a driver crash on a small integrated GPU).

### Options
- **Anisotropic filtering** (the game's own, 2x to 16x) and **Show FPS** on the Graphics page.
- A **Western** controller layout: A on the bottom button and B on the right, X and Y where Nintendo has them (#15).

### For testers and developers
- Slow IPC requests, slow RomFS reads and long shader holds are logged (`[ipc] slow`, `[fs] slow romfs read`, `[hold]`).
- `tools/swsh/scenario.py --ini key=value` runs a scenario with any setting.

## 0.1.0 (2026-10-22, preview)

The first version of the port for *Pokémon Sword* / *Shield*, forked from
[Let's Go Recomp](https://github.com/raulzrrs/letsgo-recomp) 0.3.1 (its changes up to that version are in its own
changelog). Tested with *Sword* 1.3.2 and its Expansion Pass on a Core i5-14600KF with an RTX 3080 at 2560x1440;
*Shield* is not tested. The measurements behind each item are in [docs/engineering](docs/engineering/README.md) and
the cycle reports.

### Runs the game
- **Sword and Shield are accepted** by the setup (program ids), and *Sword* boots to the title, the intro and the
  overworld: the game's `setjmp` / `longjmp`, `friend:u`'s service objects, host files closed when the game closes
  them, and every shader instruction the game uses on the way (`.FMZ` multiplies, `I2I .SAT`, `SHF` modes, `HSETP2`,
  `IPA` with its own multiplier).
- **The 1.3.2 update and the Expansion Pass**: add-on content is read from `<data root>/aoc/<data id>.bin`.
- **60 fps** for the whole game, an option for 1.0.0 and 1.3.2 (`fps = 60`).

### Launcher and settings
- **Let's Go Recomp 0.6.0's start screen and F12 menu**, in Sword's colours and with a new icon: a launcher window with
  pages (Game, Graphics, Image and sound, Controls, Mods), the same graphics and image pages in game on F12. Options:
  resolution, fullscreen, 30 / 60 fps, SMAA, brightness / contrast / saturation (a new colour pass), volume, the
  controller layout (Nintendo / Xbox), the keys, and several mods at once. Let's Go's SSAO, GI, TAA, FSR 3.1 and frame
  generation read that game's own scene passes: not here.

### Graphics fixes
- The scene-change snapshot is no longer drawn upside down (NVN's mirror copy flags), and the rug at the foot of the
  stairs is no longer black (programs identified by the content of their code, hashed when they are drawn).
- Integer textures are sampled as integers; NVN barriers are honoured; depth clamp is applied.
- Draws **wait** for a shader that is still being built instead of being skipped (no missing objects).

### Performance
- **The open world went from about 33 to 60 fps** at 2560x1440: GPU copies no longer stall the render thread,
  texture readbacks land when the GPU is done with them, the game's floating-point code no longer goes through a slow
  path (flush-to-zero in hardware), static vertex and index data stays in video memory, and the render thread keeps
  only the copies that a write can actually reach.
- **Towns** went from about 65 to 100 fps without the frame cap (more submission slots; descriptor pools that grow).
- **First visits stall much less**: shaders with a computed branch are translated into code the driver compiles in
  milliseconds instead of up to a second, and at the first shader a frame has to wait for, the frame's other shaders
  are found and built in parallel (a town's first entry: 8.4 s → 1.4 s with an empty cache). Shader workers run below
  the game's threads, and the game memory the GPU reads is set up at boot instead of at the first effect that needs it.
- **Shader preparation on the first start**: every shader stage in the game's files (18,587, found in its packs and
  effect files: 97.5% of what a session draws with) is compiled in the background into its own pipeline part
  (graphics pipeline library, now on by default), so a first visit links instead of compiling. As a new install: a
  wild battle went from 50 frames over 50 ms to 9-12, a town's entry from 3.5-4 s of hitches to 0.8 s, and the first 6
  minutes of a new game from 7.75 s to about 0.6 s. About 6 minutes on 8 threads (a notice in a corner of the screen shows the progress) and 3 GB of disk;
  shaders on disk are compressed (zstd).

### Tools (for developers)
- A scenario harness with saved positions, regression checks and a supervisor (`tools/swsh/`, `tools/agent/`), a
  sampling profiler for the game and render threads (`LARECOMP_SAMPLE`), wait profiles (`LARECOMP_RT_PROF`,
  `LARECOMP_VK_WAITPROF`), and a way to measure true first-time shader compiles (`LARECOMP_DRIVER_CACHE_NONCE`).
  All in [docs/DEPURACAO.md](docs/DEPURACAO.md).
