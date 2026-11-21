// time:u / time:a / time:s
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "../hle.h"
#include "../ipc.h"

static const uint8_t clock_uuid[16] = {0x4c, 0x41, 0x52, 0x65, 0x63, 0x6f, 0x6d, 0x70, 1, 2, 3, 4, 5, 6, 7, 8};

// LARECOMP_DAYS_AHEAD=<n>: the clock n days ahead (tests of daily things: the partner's hunger, daily items)
// LARECOMP_FIXED_TIME=<unix seconds>: the clock starts there and runs from boot (tests: the game lights its scenes by
// the time of day, so runs at different hours do not compare - tools/swsh/scenario.py always sets it)
//
// The steady clock: on the console it runs on across restarts with the same source id, and games measure time spent
// with it (nn::time::GetSpanBetween, ClockSnapshot: issue #25, the Poke Jobs). Here it is the host's clock at the
// first look plus the system tick since then: it goes on between sessions and reboots of the PC instead of starting
// from the PC's uptime, which a reboot set back (to the game: the clock turned back by the user).
static int64_t fixed_time(void) {
    static int64_t fixed = -1;
    if (fixed < 0) fixed = getenv("LARECOMP_FIXED_TIME") ? atoll(getenv("LARECOMP_FIXED_TIME")) : 0;
    return fixed;
}
static int64_t steady_base(void) {  // seconds: steady = base + ticks / 19.2 MHz
    static int64_t base = INT64_MIN;
    if (base == INT64_MIN) base = (fixed_time() > 0 ? fixed_time() : (int64_t)time(NULL)) - (int64_t)(rt_ticks() / 19200000ull);
    return base;
}
static int64_t steady_seconds(void) { return steady_base() + (int64_t)(rt_ticks() / 19200000ull); }
static int64_t unix_now(void) {
    static int64_t ahead = -1;
    if (ahead < 0) ahead = getenv("LARECOMP_DAYS_AHEAD") ? atoll(getenv("LARECOMP_DAYS_AHEAD")) * 86400 : 0;
    if (fixed_time() > 0) return steady_seconds() + ahead;
    return (int64_t)time(NULL) + ahead;
}

static void write_system_context(uint8_t *p) {  // SystemClockContext (0x20)
    int64_t steady = steady_seconds();
    *(int64_t *)p = unix_now() - steady;
    *(int64_t *)(p + 8) = steady;
    memcpy(p + 0x10, clock_uuid, 16);
}

static void time_shmem_map(KSharedMem *s) {
    uint8_t *m = (uint8_t *)(uintptr_t)s->guest_addr;
    LOG("[time] shared memory mapped at %llx\n", (unsigned long long)s->guest_addr);
    // 0x00: LockFreeAtomicType<SteadyClockContext{u64 internal_offset, uuid}>
    *(uint32_t *)m = 0;
    *(int64_t *)(m + 0x08) = steady_base() * 1000000000ll;  // (internal offset in ns, added to the tick's)
    memcpy(m + 0x10, clock_uuid, 16);
    // 0x38 local system clock, 0x80 network system clock
    *(uint32_t *)(m + 0x38) = 0;
    write_system_context(m + 0x40);
    *(uint32_t *)(m + 0x80) = 0;
    write_system_context(m + 0x88);
    // 0xC8 automatic correction enabled
    *(uint32_t *)(m + 0xC8) = 0;
    m[0xCC] = 0;
}

static void system_clock(Service *s, IpcReq *req, IpcResp *resp) {
    (void)s;
    switch (req->cmd) {
    case 0: OUT(int64_t, 0) = unix_now(); resp_size(resp, 8); break;  // GetCurrentTime
    case 2: write_system_context(resp->data); resp_size(resp, 0x20); break;  // GetSystemClockContext
    default: LOG("[time] ISystemClock cmd %u (ok)\n", req->cmd);
    }
}

static void steady_clock(Service *s, IpcReq *req, IpcResp *resp) {
    (void)s;
    switch (req->cmd) {
    case 0:  // GetCurrentTimePoint
        OUT(int64_t, 0) = steady_seconds();
        memcpy(resp->data + 8, clock_uuid, 16);
        resp_size(resp, 0x18);
        break;
    default: OUT(int64_t, 0) = 0; resp_size(resp, 8); LOG("[time] ISteadyClock cmd %u (ok)\n", req->cmd);
    }
}

static void to_calendar(int64_t t, IpcResp *resp, int utc) {
    time_t tt = (time_t)t;
    struct tm tm;
    if (utc) gmtime_r(&tt, &tm); else localtime_r(&tt, &tm);
    uint8_t *o = resp->data;
    *(uint16_t *)o = (uint16_t)(tm.tm_year + 1900);
    o[2] = (uint8_t)(tm.tm_mon + 1); o[3] = (uint8_t)tm.tm_mday;
    o[4] = (uint8_t)tm.tm_hour; o[5] = (uint8_t)tm.tm_min; o[6] = (uint8_t)tm.tm_sec;
    *(uint32_t *)(o + 8) = (uint32_t)tm.tm_wday;
    *(uint32_t *)(o + 12) = (uint32_t)tm.tm_yday;
#ifdef _WIN32  // no tm_zone / tm_gmtoff: offset = local broken-down time read back as UTC, minus now
    struct tm g;
    gmtime_r(&tt, &g);
    long gmtoff = (long)difftime(_mkgmtime(&tm), _mkgmtime(&g));
    memcpy(o + 16, utc ? "UTC" : "LCL", 4);
#else
    long gmtoff = tm.tm_gmtoff;
    memcpy(o + 16, utc ? "UTC" : tm.tm_zone, 4);
#endif
    *(uint32_t *)(o + 24) = (uint32_t)tm.tm_isdst;
    *(int32_t *)(o + 28) = utc ? 0 : (int32_t)gmtoff;
    resp_size(resp, 0x20);
}

// nn::time::ClockSnapshot (0xD0), IStaticService GetClockSnapshot (issue #25): the game takes one when a Poke Job
// starts and compares it with a later one (GetSpanBetween its steady time points; the user clock's difference). It was
// answered "ok" with nothing written: an empty snapshot, and no time ever passed.
static void calendar_fields(int64_t t, uint8_t *cal, uint8_t *info) {  // CalendarTime (8) + CalendarAdditionalInfo (0x18), local
    time_t tt = (time_t)t;
    struct tm tm, g;
    localtime_r(&tt, &tm);
    gmtime_r(&tt, &g);
    *(uint16_t *)cal = (uint16_t)(tm.tm_year + 1900);
    cal[2] = (uint8_t)(tm.tm_mon + 1), cal[3] = (uint8_t)tm.tm_mday;
    cal[4] = (uint8_t)tm.tm_hour, cal[5] = (uint8_t)tm.tm_min, cal[6] = (uint8_t)tm.tm_sec;
    *(uint32_t *)info = (uint32_t)tm.tm_wday;
    *(uint32_t *)(info + 4) = (uint32_t)tm.tm_yday;
    memcpy(info + 8, "LCL", 4);
    *(uint32_t *)(info + 16) = (uint32_t)(tm.tm_isdst > 0);
#ifdef _WIN32
    *(int32_t *)(info + 20) = (int32_t)difftime(_mkgmtime(&tm), _mkgmtime(&g));
#else
    *(int32_t *)(info + 20) = (int32_t)tm.tm_gmtoff;
#endif
}
static void clock_snapshot(uint8_t *p, uint8_t type) {
    memset(p, 0, 0xD0);
    write_system_context(p + 0x00);  // user clock context
    write_system_context(p + 0x20);  // network clock context
    int64_t now = unix_now();
    *(int64_t *)(p + 0x40) = now;  // user time
    *(int64_t *)(p + 0x48) = now;  // network time
    calendar_fields(now, p + 0x50, p + 0x60);
    calendar_fields(now, p + 0x58, p + 0x78);
    *(int64_t *)(p + 0x90) = steady_seconds();  // steady clock time point
    memcpy(p + 0x98, clock_uuid, 16);
    memcpy(p + 0xA8, "UTC", 4);  // location name (as GetDeviceLocationName)
    p[0xCC] = 0;  // automatic correction
    p[0xCD] = type;
}

static void timezone_service(Service *s, IpcReq *req, IpcResp *resp) {
    (void)s;
    switch (req->cmd) {
    case 0: memcpy(resp->data, "UTC", 4); resp_size(resp, 0x24); break;  // GetDeviceLocationName
    case 100: to_calendar(IN(int64_t, 0), resp, 1); break;  // ToCalendarTime(rule buffer)
    case 101: to_calendar(IN(int64_t, 0), resp, 0); break;  // ToCalendarTimeWithMyRule
    case 201: {  // ToPosixTimeWithMyRule -> count + buffer of posix times
        const uint8_t *in = req->data;
        struct tm tm = {0};
        tm.tm_year = *(uint16_t *)in - 1900; tm.tm_mon = in[2] - 1; tm.tm_mday = in[3];
        tm.tm_hour = in[4]; tm.tm_min = in[5]; tm.tm_sec = in[6]; tm.tm_isdst = -1;
        int64_t t = mktime(&tm);
        if (req->nc) *(int64_t *)(uintptr_t)req->c[0].addr = t;
        else if (req->nb) *(int64_t *)(uintptr_t)req->b[0].addr = t;
        OUT(uint32_t, 0) = 1;
        resp_size(resp, 4);
        break;
    }
    default: LOG("[time] ITimeZoneService cmd %u (ok)\n", req->cmd); resp_size(resp, 0x10);
    }
}

static uint32_t g_shmem_handle;
RT_ON_RESET(time_reset) { g_shmem_handle = 0; }

static void static_service(Service *s, IpcReq *req, IpcResp *resp) {
    (void)s;
    switch (req->cmd) {
    case 0: case 1: case 4: case 5:
        ipc_return_object(resp, service_new("ISystemClock", system_clock, NULL));
        break;
    case 2: ipc_return_object(resp, service_new("ISteadyClock", steady_clock, NULL)); break;
    case 3: ipc_return_object(resp, service_new("ITimeZoneService", timezone_service, NULL)); break;
    case 20: {  // GetSharedMemoryNativeHandle
        if (!g_shmem_handle) kernel_create_shared_memory(0x1000, time_shmem_map, &g_shmem_handle);
        resp->copy_handles[resp->ncopy++] = g_shmem_handle;
        break;
    }
    case 100: case 101: OUT(uint8_t, 0) = 0; resp_size(resp, 1); break;
    case 400: {  // GetClockSnapshot(u8 type) -> ClockSnapshot in the out buffer
        IpcBuf *b = out_buf(req, 0);
        if (b && b->size >= 0xD0) clock_snapshot((uint8_t *)(uintptr_t)b->addr, IN(uint8_t, 0));
        static bool logged;
        if (!logged) { logged = true; LOG("[time] GetClockSnapshot (type %u)\n", IN(uint8_t, 0)); }
        break;
    }
    default: LOG("[time] IStaticService cmd %u (ok)\n", req->cmd); resp_size(resp, 0x10);
    }
}

SERVICE("time:u", time_u) { static_service(s, req, resp); }
SERVICE("time:a", time_a) { static_service(s, req, resp); }
SERVICE("time:s", time_s) { static_service(s, req, resp); }

// nn::time::StandardUserSystemClock::GetCurrentTime(PosixTime*): answered directly from the host clock. The SDK's
// shared-memory path aborts (svcBreak via nn::diag::Abort, result 0xcc74) if the local clock's source id ever
// differs from the steady clock's; that happened once in the room sweep on an autosave (entering Eagle's Tower) and
// could not be reproduced, so the date stamp no longer depends on it.
HLE_FUNC("_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE", time_std_user_now) {
    static int n;
    if (n++ < 3) LOG("[time] StandardUserSystemClock::GetCurrentTime (host clock)\n");
    if (c->x[0]) *(int64_t *)(uintptr_t)c->x[0] = unix_now();
    c->x[0] = 0;  // ResultSuccess
}
