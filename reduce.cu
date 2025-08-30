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

// Dummy kernel for retrieving PTX version.
__global__ void DummyKernel() {}

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

__global__ void ReduceKernel2(u32 count, f32 *input, f32 *output)
{
    int segment_start = blockIdx.x*(2*blockDim.x);
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if ((threadIdx.x % stride) == 0)
        {
            int index = segment_start + 2*threadIdx.x;
            if (index < count)
            {
                f32 temp = 0.f;
                if (index + stride < count)
                    temp = input[index + stride];
                input[index] += temp;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(output, input[segment_start]);
}

#if COMPILING_FROM_PYTORCH
void Reduce(torch::Tensor input, torch::Tensor output)
{
    int array_count = input.numel();
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (array_count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel2<<<grid_dim, block_dim>>>(array_count, input.data_ptr<f32>(), output.data_ptr<f32>());
    CUDACheck(cudaDeviceSynchronize());
}
#else

void Reduce1(int count, f32 *input)
{
    assert(count <= 2048);

    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel1<<<grid_dim, block_dim>>>(input);
}

void Reduce2(int count, f32 *input, f32 *output)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel2<<<grid_dim, block_dim>>>(count, input, output);
}

int main()
{
    if (0)
    {
        cudaFuncAttributes attr;
        CUDACheck(cudaFuncGetAttributes(&attr, DummyKernel));

        int major_ver = attr.ptxVersion/10;
        int minor_ver = attr.ptxVersion%10;
        printf("PTX version: %d.%d\n", major_ver, minor_ver);

        int device;
        CUDACheck(cudaGetDevice(&device));

        cudaDeviceProp device_prop = { 0 };
        CUDACheck(cudaGetDeviceProperties(&device_prop, device));

        printf("Device name: %s\n", device_prop.name);
        printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("SMs: %d\n", device_prop.multiProcessorCount);
        // NOTE(achal): Compute capability 7.5 has 64 CUDA cores per SM.
        printf("CUDA cores: %d\n", device_prop.multiProcessorCount*64);
        printf("Clock rate: %d KHz\n", device_prop.clockRate); // NOTE(achal): This is deprecated.
        printf("Total Global Memory: %.2f GB (%llu bytes)\n", (device_prop.totalGlobalMem/(1024.f*1024.f*1024.f)), device_prop.totalGlobalMem);
        printf("Shared Memory (per block): %.2f KB (%llu bytes)\n", (device_prop.sharedMemPerBlock/1024.f), device_prop.sharedMemPerBlock);
        printf("Total Constant Memory: %.2f KB (%llu bytes)\n", (device_prop.totalConstMem/1024.f), device_prop.totalConstMem);
        printf("Warp Size: %d threads\n", device_prop.warpSize);
        printf("Max threads per block: %d\n", device_prop.maxThreadsPerBlock);
        printf("Max Block dimension: %dx%dx%d\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("Max Grid dimension: %dx%dx%d\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]); 
        printf("32-bit registers (per block): %d\n", device_prop.regsPerBlock);
    }

    u64 array_count = 100000000;
    f32 *input = (f32 *)malloc(array_count*sizeof(f32));
    for (u64 i = 0; i < array_count; ++i)
        input[i] = 1.f;

    u64 elapsed = 0;
    f64 sum = 0.0;
    int rep_count = 20;
    for (int rep = 0; rep < rep_count; ++rep)
    {
        sum = 0.0;

        u64 begin_ts = ReadCPUTimer();
        for (u64 i = 0; i < array_count; ++i)
            sum += input[i];
        u64 end_ts = ReadCPUTimer();
        
        elapsed += end_ts - begin_ts;
    }

    printf("Result (CPU): %f\n", sum);

    u64 cpu_hz = EstimateCPUTimerFrequency(1000);
    f64 cpu_elapsed = ((f64)elapsed / ((f64)cpu_hz*rep_count))*1000.0;
    printf("Elapsed (CPU): %f ms\n", cpu_elapsed);

    f32 *d_input = 0;
    CUDACheck(cudaMalloc(&d_input, array_count*sizeof(f32)));
    CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));

    f32 *d_output = 0;
    CUDACheck(cudaMalloc(&d_output, sizeof(f32)));
    CUDACheck(cudaMemset(d_output, 0, sizeof(f32)));

    // Reduce1(array_count, d_input);

    cudaEvent_t start_event, stop_event;
    CUDACheck(cudaEventCreate(&start_event));
    CUDACheck(cudaEventCreate(&stop_event));

    // Warmups
    for (int i = 0; i < 3; ++i)
    {
        CUDACheck(cudaMemcpyAsync(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));
        CUDACheck(cudaMemsetAsync(d_output, 0, sizeof(f32)));
        Reduce2(array_count, d_input, d_output);
    }

    f64 ms = 0;
    for (int i = 0; i < rep_count; ++i)
    {
        CUDACheck(cudaMemcpyAsync(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));
        CUDACheck(cudaMemsetAsync(d_output, 0, sizeof(f32)));

        CUDACheck(cudaEventRecord(start_event));
        Reduce2(array_count, d_input, d_output);
        CUDACheck(cudaEventRecord(stop_event));
        CUDACheck(cudaEventSynchronize(stop_event));

        f32 curr_ms;
        CUDACheck(cudaEventElapsedTime(&curr_ms, start_event, stop_event));
        ms += curr_ms;
    }

    f32 out = 0.f;
    CUDACheck(cudaMemcpy(&out, d_output, sizeof(f32), cudaMemcpyDeviceToHost));
    printf("Result (GPU): %f\n", out);

    ms /= rep_count;
    printf("Elapsed (GPU): %f ms\n", ms);
    printf("Bandwidth: %f GB/s\n", (1000.0*(array_count*sizeof(f32)))/(ms*1024.0*1024.0*1024.0));
    
    return 0;
}
#endif