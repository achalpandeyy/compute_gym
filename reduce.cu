#include <stdio.h>

#if defined(__clang__)
#define COMPILER_CLANG 1
#elif defined(_MSC_VER)
#define COMPILER_MSVC 1
#elif defined(__NVCC__)
#define COMPILER_NVCC
#else
#error "Compiler not supported"
#endif

#if defined(_M_X64) || defined(__x86_64__)
#define ARCH_X64 1
#elif defined(__aarch64__)
#define ARCH_ARM64 1
#elif defined(__wasm32__)
#define ARCH_WASM32 1
#else
#error "Architecture not supported"
#endif

#if defined(_WIN32)
#define PLATFORM_WINDOWS 1
#elif defined(__linux__)
#define PLATFORM_LINUX 1
#elif defined(__APPLE__)
#define PLATFORM_MACOS 1
#elif defined(__wasm32__)
#define PLATFORM_WASM 1
#else
#error "Platform not supported"
#endif

#define KiloBytes(x) (u64)(1024ull*(x))
#define MegaBytes(x) (u64)(1024ull*KiloBytes(x))
#define GigaBytes(x) (u64)(1024ull*MegaBytes(x))

#define Minimum(x, y) ((x) < (y) ? (x) : (y))
#define Maximum(x, y) ((x) > (y) ? (x) : (y))
#define ArrayCount(x) (sizeof(x)/sizeof((x)[0]))

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)

#if PLATFORM_WINDOWS
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <stdint.h>

#if !ARCH_WASM32
#include <assert.h>
#define Assert assert
#else
#define Assert(...)
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u8 b8;
typedef u32 b32;

typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef float f32;
typedef double f64;

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

inline static void GetCUDAErrorDetails(cudaError_t error, char const **error_name, char const **error_string)
{
    if (error_name)
        *error_name = cudaGetErrorName(error);
    
    if (error_string)
        *error_string = cudaGetErrorString(error);
}

#define CUDACheck_(fn_call, line)\
{\
    cudaError_t prev_error = cudaGetLastError();\
    while (prev_error != cudaSuccess)\
    {\
        char const *error_name = 0;\
        char const *error_string = 0;\
        GetCUDAErrorDetails(prev_error, &error_name, &error_string);\
        printf("[ERROR]: CUDA Runtime already had an error: %s %s", error_name, error_string);\
        prev_error = cudaGetLastError();\
    }\
    fn_call;\
    cudaError_t error = cudaGetLastError();\
    if (error != cudaSuccess)\
    {\
        char const *error_name = 0;\
        char const *error_string = 0;\
        GetCUDAErrorDetails(error, &error_name, &error_string);\
        printf("CUDA Error on line %u: %s %s", line, error_name, error_string);\
    }\
}
#define CUDACheck(fn_call) CUDACheck_(fn_call, __LINE__)

__global__ void ReduceKernel1(f32 *input)
{
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if ((threadIdx.x % stride) == 0)
        {
            int index = 2*threadIdx.x;
            input[index] += input[index + stride];
        }
        __syncthreads();
    }
}

int main()
{
    u64 array_count = 2048; // 100000000;
    f32 *input = (f32 *)malloc(array_count*sizeof(f32));
    for (u64 i = 0; i < array_count; ++i)
        input[i] = 1.f;

    u64 elapsed = 0;
    f32 sum = 0.f;
    int rep_count = 10;
    for (int rep = 0; rep < rep_count; ++rep)
    {
        sum = 0.f;

        u64 begin_ts = ReadCPUTimer();
        for (u64 i = 0; i < array_count; ++i)
            sum += input[i];
        u64 end_ts = ReadCPUTimer();
        
        elapsed += end_ts - begin_ts;
    }

    printf("Sum: %f\n", sum);

    f64 elapsed_avg = (f64)elapsed/(f64)rep_count;
    u64 cpu_hz = EstimateCPUTimerFrequency(1000);
    f64 cpu_elapsed = ((double)elapsed / (double)cpu_hz)*1000.0;
    printf("CPU elapsed: %f ms\n", cpu_elapsed);

    f32 *d_input = 0;
    CUDACheck(cudaMalloc(&d_input, array_count*sizeof(f32)));
    CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));

    ReduceKernel1<<<1, array_count/2>>>(d_input);
    CUDACheck(cudaDeviceSynchronize());

    f32 out = 0.f;
    CUDACheck(cudaMemcpy(&out, d_input, sizeof(f32), cudaMemcpyDeviceToHost));
    printf("GPU: %f\n", out);

    return 0;
}