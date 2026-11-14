// RomFS image from a folder of game files (runtime/romfs_build.c). The game reads its data through one raw RomFS
// image (romfs.bin, runtime/services/fs.c); players bring their own game files, and the start screen builds the image
// from their romfs/ folder.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Writes the RomFS image of the folder `dir` to `out`. progress (may be NULL): bytes of file data written so far and
// the total, now and then. false = failed (why: a short message).
bool romfs_build(const char *dir, const char *out, void (*progress)(uint64_t done, uint64_t total), char *why,
                 size_t why_size);

// Whether the file at path is a raw RomFS image the game can read: its 0x50-byte header, the tables and data it points
// at inside the file, and about as many files as the whole game. false: why says what's wrong (an IVFC image - a dump
// tool's other format -, a cut file, a mod's or an update's romfs alone...).
bool romfs_image_ok(const char *path, char *why, size_t why_size);

// Bytes of file data under dir (the image is about this plus its tables).
uint64_t romfs_dir_bytes(const char *dir);

// Copies the folder tree src to dst (hard links where the file system allows: no second copy of 6 GB). progress as
// above. false = failed.
bool romfs_copy_tree(const char *src, const char *dst, void (*progress)(uint64_t done, uint64_t total));
// One file the same way (a hard link where the file system allows).
bool romfs_copy_file(const char *src, const char *dst, void (*progress)(uint64_t done, uint64_t total));

#ifdef __cplusplus
}
#endif
