#ifndef PROFILER_H
#define PROFILER_H

#ifndef PROFILER
#define PROFILER 0
#endif

#if PROFILER

#if PLATFORM_WINDOWS
inline u64 ReadOSTimer()
{
    LARGE_INTEGER large_int;
    const BOOL retval = QueryPerformanceCounter(&large_int);
    Assert(retval != 0);
    u64 result = large_int.QuadPart;
    return result;
}

inline u64 GetOSTimerFrequency()
{
    LARGE_INTEGER large_int;
    BOOL retval = QueryPerformanceFrequency(&large_int);
    Assert(retval != 0);
    u64 result = large_int.QuadPart;
    return result;
}
#elif PLATFORM_LINUX || PLATFORM_MACOS
#include <time.h>
static inline u64 GetOSTimerFrequency()
{
    return 1000000000;
}

static inline u64 ReadOSTimer()
{
    struct timespec ts = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    u64 result = (u64)ts.tv_nsec + ts.tv_sec * 1000000000LL;
    return result;
}
#endif

#if ARCH_X64
#if PLATFORM_WINDOWS
#include <intrin.h>
#elif PLATFORM_LINUX
#include <x86intrin.h>
#endif
static inline u64 ReadCPUTimer()
{
    return __rdtsc();
}
#elif ARCH_ARM64
static inline u64 ReadCPUTimer()
{
    u64 cntvct;
    asm volatile ("mrs %0, cntvct_el0; " : "=r"(cntvct) :: "memory");
    return cntvct;
}
#endif

// NOTE(achal): There are ways to query this information directly from the CPU (through
// inline assembly) and from the OS, but this is fine for my purposes, at least for now.
static u64 EstimateCPUTimerFrequency(u64 ms_to_wait)
{
    Assert((ms_to_wait % 1000) == 0);
    u64 os_hz = GetOSTimerFrequency();
    u64 os_wait_time = (os_hz * (ms_to_wait/1000));
    
    u64 os_elapsed = 0;
    u64 os_begin = ReadOSTimer();
    
    u64 cpu_begin = ReadCPUTimer();
    while (os_elapsed < os_wait_time)
    {
        os_elapsed = ReadOSTimer() - os_begin;
    }
    u64 cpu_end = ReadCPUTimer();
    
    u64 cpu_elapsed = cpu_end - cpu_begin;
    // Use the invariant: os_elapsed/os_hz == cpu_elapsed/cpu_hz
    u64 cpu_hz = (os_hz * cpu_elapsed)/os_elapsed;
    
    return cpu_hz;
}

typedef struct ProfileAnchor ProfileAnchor;
struct ProfileAnchor
{
    u64 elapsed_inclusive;
    u64 elapsed_exclusive;
    u64 hit_count;
    const char *label;
};

typedef struct Profiler Profiler;
struct Profiler
{
    u64 elapsed;
    
    u32 active_anchor_id;
    b32 running;
    ProfileAnchor anchors[1+31];
    u64 counter_start_value;
};

static Profiler g_profiler;

static inline void BeginProfiler()
{
    g_profiler.running = 1;
    g_profiler.elapsed = ReadCPUTimer();
    // NOTE(achal): Due getting the first value of __COUNTER__ here,
    // we will always start filling up the anchors array from index 1.
    // That means we can use the first element as "invalid" if we need to.
    g_profiler.counter_start_value = __COUNTER__;
}

static inline void EndProfiler()
{
    g_profiler.elapsed = ReadCPUTimer() - g_profiler.elapsed;
    g_profiler.running = 0;
}

typedef struct ProfileScope ProfileScope;
struct ProfileScope
{
    u64 tsc_begin;
    u32 anchor_id;
    u32 parent_anchor_id;
    const char *label;
    u64 old_elapsed_inclusive;
};

inline static ProfileScope BeginProfileScope(const char *label_, u32 id)
{
    ProfileScope prof_scope = {0};
    if (!g_profiler.running)
        return prof_scope;
    
    Assert(id > g_profiler.counter_start_value);

    u32 anchor_id = id - g_profiler.counter_start_value;
    Assert(anchor_id < ArrayCount(g_profiler.anchors));
    
    prof_scope.anchor_id = anchor_id;
    prof_scope.parent_anchor_id = g_profiler.active_anchor_id;
    prof_scope.label = label_;
    
    ProfileAnchor *anchor = g_profiler.anchors + prof_scope.anchor_id;
    prof_scope.old_elapsed_inclusive = anchor->elapsed_inclusive;
    
    g_profiler.active_anchor_id = anchor_id;
    
    prof_scope.tsc_begin = ReadCPUTimer();
    
    return prof_scope;
}

inline static void EndProfileScope(ProfileScope *prof_scope)
{
    u64 elapsed = ReadCPUTimer() - prof_scope->tsc_begin;
    
    if (!g_profiler.running)
        return;
    
    ProfileAnchor *anchor = g_profiler.anchors + prof_scope->anchor_id;
    ProfileAnchor *parent_anchor = g_profiler.anchors + prof_scope->parent_anchor_id;
    
    if (anchor->hit_count == 0)
        anchor->label = prof_scope->label;
    
    ++anchor->hit_count;
    anchor->elapsed_inclusive = prof_scope->old_elapsed_inclusive + elapsed;
    anchor->elapsed_exclusive += elapsed;
    parent_anchor->elapsed_exclusive -= elapsed;
    
    g_profiler.active_anchor_id = prof_scope->parent_anchor_id;
}

static void PrintPerformanceProfile()
{
    Assert(g_profiler.running == 0);
    u64 total_time = g_profiler.elapsed;
    
    u64 cpu_hz = EstimateCPUTimerFrequency(1000);
    for (u32 i = 0; i < ArrayCount(g_profiler.anchors); ++i)
    {
        ProfileAnchor *anchor = g_profiler.anchors + i;
        if (anchor->label && anchor->hit_count > 0)
        {
            f32 ms = (anchor->elapsed_exclusive*1000.f)/cpu_hz;
            f32 percent = (anchor->elapsed_exclusive*100.f)/total_time;
            
            printf("%s[%llu]: %.3f ms (%.3f%%)\n", anchor->label, anchor->hit_count, ms, percent);
        }
    }
}

#define PROFILE_SCOPE_BEGIN(label) ProfileScope _prof_scope_ = (BeginProfileScope(label, __COUNTER__))
#define PROFILE_SCOPE_END() (EndProfileScope(&_prof_scope_))

#define PROFILER_END_OF_COMPILATION_UNIT StaticAssert(ArrayCount(g_profiler.anchors) >= __COUNTER__+1, ran_out_of_profile_anchors)

#else
#define BeginProfiler(...)
#define EndProfiler(...)
#define PrintPerformanceProfile(...)

#define PROFILE_SCOPE_BEGIN(...)
#define PROFILE_SCOPE_END(...)
#define PROFILER_END_OF_COMPILATION_UNIT
#endif // PROFILER

#endif // PROFILER_H
