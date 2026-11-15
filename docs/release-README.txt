SwSh Recomp
===========

A native Windows port of Pokemon Sword and Pokemon Shield (Nintendo Switch) by static recompilation.
Project page, full documentation and source code: https://github.com/raulzrrs/swsh-recomp

NO GAME FILES ARE INCLUDED. You need your own legally obtained copy of Pokemon Sword or Shield, extracted (the exefs
and romfs folders).

Supported: Pokemon Sword (title id 0100ABF008968000) version 1.3.2, the most tested, with or without the Expansion
Pass. Pokemon Shield (01008DB008C2C000) is recognized but NOT TESTED yet. Cartridge or eShop, any region; extract the
game with the 1.3.2 update applied. Packed files (.xci, .nsp, .nsz, .nca) must be extracted first.

Requirements
------------
- Windows 10 or 11, 64-bit
- A CPU with AVX2, FMA and BMI2 (Intel Haswell / 2013 or newer, AMD Excavator / Zen or newer); checked at startup
- A GPU whose driver supports Vulkan 1.3; see vulkan.gpuinfo.org for yours
- RAM: about 5 GB while playing
- About 18 GB of free disk space

How to play
-----------
1. Unzip this folder anywhere you can write to (e.g. D:\Games\SwShRecomp). Not inside Program Files.
2. Open SwShRecomp.exe. It is not code-signed: if Windows says "Windows protected your PC", click "More info" and
   then "Run anyway".
3. Click "Select game folder..." and pick the folder with your extracted game (the one that holds exefs and romfs,
   or any folder above it).
4. Wait while the game files are copied and the game code is recompiled for your PC. This happens only once and
   takes a while (about 40 minutes on a 20-thread CPU, longer with fewer threads).
5. Click Play.

On the first start the game's shaders are prepared for your graphics card in the background (about 6 minutes; the
game runs slower meanwhile). After that, new places hitch much less. They are kept on disk (cache\, about 3 GB).

Options (start screen, and F12 in the game): resolution, fullscreen, 30 / 60 fps, anti-aliasing (SMAA, TAA, NVIDIA
DLAA), ambient occlusion, shadow resolution, brightness, contrast, saturation, volume, the controller layout (Nintendo /
Xbox), the keys, the mouse, and mods (several at once). More in larecomp.ini.

NVIDIA DLAA (RTX cards) uses NVIDIA's DLSS model, nvngx_dlss.dll, included next to SwShRecomp.exe (NVIDIA DLSS,
Copyright NVIDIA Corporation, redistributed under the NVIDIA DLSS SDK license: github.com/NVIDIA/DLSS). It needs a
current NVIDIA driver; elsewhere the DLAA choice is hidden and TAA does the job.

Everything stays in this folder: settings (larecomp.ini), saves (save-sword\ or save-shield\), the game files
(extracted\), shader caches
(cache\) and the log (lasrecomp.log). To move or back up the game, move or copy the whole folder.

Expansion Pass (optional)
-------------------------
Not set up by the launcher yet: each DLC's RomFS has to be packed into extracted\aoc\<data id>.bin. See
docs/DIAGNOSTICO-SWSH.md on the project page.

Controls
--------
Keyboard: WASD move, Space/Enter = A, Q/Backspace = B, Tab = X (menu), R = Y, Z/C = L/R, 1/3 = ZL/ZR, = and - = Plus/Minus,
arrows = D-pad, numpad 8/4/5/6 = right stick (camera), Esc = close. All keys can be changed on the start screen.
Controllers: any controller SDL recognizes (Xbox, PlayStation, Switch Pro...).
Mouse: moving it turns the camera (the window holds the mouse; F12, Esc or Alt+Tab let it go). In a menu with the
game's cursor it is a pointer: click an option to choose it. Left = A, right = B, middle = X (menu), wheel = up / down,
side buttons = L / R.

Troubleshooting
---------------
- "That's not Pokemon: Let's Go or Sword / Shield": pick the folder of your own extracted game.
- Packed game files (.xci / .nsp) must be extracted first.
- "The game code builder is missing": your antivirus may have quarantined lasgen\lasgen.exe (a PyInstaller
  program, sometimes flagged by mistake). Restore it or add this folder as an exception.
- The game closes or doesn't start: see lasrecomp.log in this folder and report it on the project page.

Legal
-----
Fan project, not affiliated with Nintendo, The Pokemon Company, Game Freak or Creatures. MIT licensed (LICENSE.txt),
based on Let's Go Recomp (https://github.com/raulzrrs/letsgo-recomp) and LAS Recomp by Peiikai
(LICENSE-LASRecomp.txt, https://github.com/peiikai/LASRecomp).
Bundled: SDL3 (zlib), FFmpeg (LGPL 2.1, built with only the decoders the game needs; source: https://ffmpeg.org),
shaderc (Apache 2.0), libiconv (LGPL), Zig (MIT), the Baloo 2 font (SIL OFL 1.1, Baloo2-OFL.txt), and the
GCC / MinGW-w64 runtime libraries (GPL with the runtime library exception).
