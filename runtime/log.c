// The log, written off the game's threads.
//
// LOG used to be fprintf(stderr) on whatever thread called it, and a release build's stderr is an unbuffered file
// (platform/windows.c log_to_file_without_console): every line was a write to the disk on a game thread. When that
// write waits - a slow SATA SSD, an antivirus scanning the growing file, a console or a pipe that drains slowly - the
// game thread waits with it. The game brings its network up and down every ~1.7 s, which logs ~50 lines at once, and
// a run whose log went through a slow pipe dropped from 60 to 45 fps with a ~160 ms stall at each of those (the same
// save and place: 60 fps, 18 ms worst frame, with the log written straight to a file).
//
// Now a line is formatted on the calling thread into a 1 MB ring and a writer thread of its own writes the ring out
// in batches. The ring full (a burst faster than the disk), a line is dropped and counted, never waited for. FATAL,
// the crash report and exit() empty the ring first (rt_log_flush), so the last lines before a crash are kept.
// LARECOMP_LOG_SYNC=1: the old way, each line written on its own thread (comparisons).
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "rt.h"

#define RING (1u << 20)

static char g_ring[RING];
static size_t g_head, g_tail;  // bytes written in / out, ever (indexes are mod RING)
static uint64_t g_dropped;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_more = PTHREAD_COND_INITIALIZER, g_done = PTHREAD_COND_INITIALIZER;
static int g_mode;  // 0 not decided, 1 asynchronous, 2 synchronous
static volatile size_t g_out;  // g_tail, readable without the lock (crash path)

static void *writer(void *arg) {
    (void)arg;
    plat_thread_background();
    char buf[64 << 10];
    for (;;) {
        pthread_mutex_lock(&g_lock);
        while (g_head == g_tail) pthread_cond_wait(&g_more, &g_lock);
        size_t n = g_head - g_tail, at = g_tail % RING;
        if (n > sizeof buf) n = sizeof buf;
        if (n > RING - at) n = RING - at;  // (up to the ring's end; the rest on the next pass)
        memcpy(buf, g_ring + at, n);
        uint64_t dropped = g_dropped;
        g_dropped = 0;
        pthread_mutex_unlock(&g_lock);
        if (dropped) fprintf(stderr, "[log] %llu line(s) dropped: the log was written more slowly than it filled\n",
                             (unsigned long long)dropped);
        fwrite(buf, 1, n, stderr);
        pthread_mutex_lock(&g_lock);
        g_tail += n;
        g_out = g_tail;
        pthread_cond_broadcast(&g_done);
        pthread_mutex_unlock(&g_lock);
    }
    return NULL;
}

static void start(void) {  // (under g_lock)
    const char *e = getenv("LARECOMP_LOG_SYNC");
    g_mode = e && atoi(e) ? 2 : 1;
    if (g_mode == 2) return;
    pthread_t t;
    if (pthread_create(&t, NULL, writer, NULL) != 0) { g_mode = 2; return; }
    pthread_detach(t);
    atexit(rt_log_flush);
}

void rt_log(const char *fmt, ...) {
    char line[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n >= sizeof line) {  // (cut: keep the line ending)
        n = (int)sizeof line - 1;
        line[n - 1] = '\n';
    }
    pthread_mutex_lock(&g_lock);
    if (!g_mode) start();
    if (g_mode == 2) {
        pthread_mutex_unlock(&g_lock);
        fwrite(line, 1, (size_t)n, stderr);
        return;
    }
    if (RING - (g_head - g_tail) < (size_t)n) {
        g_dropped++;
    } else {
        size_t at = g_head % RING, first = RING - at < (size_t)n ? RING - at : (size_t)n;
        memcpy(g_ring + at, line, first);
        memcpy(g_ring, line + first, (size_t)n - first);
        g_head += (size_t)n;
        pthread_cond_signal(&g_more);
    }
    pthread_mutex_unlock(&g_lock);
}

// Waits (at most 2 s) until the writer has written everything queued so far. Safe from a crashing thread: it doesn't
// take the lock when another thread holds it (the crashed one may), it then watches the writer's progress instead.
void rt_log_flush(void) {
    if (g_mode != 1) return;
    uint64_t end = plat_now_ns() + 2000000000ull;
    if (pthread_mutex_trylock(&g_lock) == 0) {
        size_t want = g_head;
        while (g_tail < want && plat_now_ns() < end) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 50000000;
            if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
            pthread_cond_timedwait(&g_done, &g_lock, &ts);
        }
        pthread_mutex_unlock(&g_lock);
    } else {
        size_t want = g_head;
        while (g_out < want && plat_now_ns() < end) plat_sleep_until_ns(plat_now_ns() + 1000000);
    }
    fflush(stderr);
}
