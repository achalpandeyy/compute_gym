#ifndef COMMON_CU
#define COMMON_CU

#include "core_types.h"

#include <stdio.h>

inline static u64 GetRandomNumber(u32 max_bits)
{
    // This value is implementation dependent. It's guaranteed that this value is at least 32767.
    // https://en.cppreference.com/w/c/numeric/random/RAND_MAX.html
    Assert(RAND_MAX >= 32767);

    u64 result = 0;
    s32 bits_left = max_bits - 1;
    while (bits_left > 0)
    {
        s32 bits_to_add = Minimum(14, bits_left);
        u32 mask = (1 << bits_to_add) - 1;
        
        u32 x = (u32)rand() & mask;
        result |= (x << (max_bits - 1 - bits_left));
        
        bits_left -= bits_to_add;
    }

    Assert(result < (1ull << max_bits));
    return result;
}

inline static void GetCUDAErrorDetails(cudaError_t error, char const **error_name, char const **error_string)
{
    if (error_name)
        *error_name = cudaGetErrorName(error);
    
    if (error_string)
        *error_string = cudaGetErrorString(error);
}

#if BUILD_DEBUG
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
    cudaDeviceSynchronize();\
    cudaError_t error = cudaGetLastError();\
    if (error != cudaSuccess)\
    {\
        char const *error_name = 0;\
        char const *error_string = 0;\
        GetCUDAErrorDetails(error, &error_name, &error_string);\
        printf("CUDA Error on line %u: %s %s", line, error_name, error_string);\
        assert(0);\
    }\
}
#else
#define CUDACheck_(fn_call, line) (fn_call);
#endif

#define CUDACheck(...) CUDACheck_(__VA_ARGS__, __LINE__)

static inline u32 GetCUDACoresPerSM(int major, int minor)
{
    if (major == 7 && minor == 5)
    {
        return 64;
    }
    else if (major == 8 && minor == 9)
    {
        return 128;
    }
    else
    {
        printf("Unsupported compute capability: %d.%d\n", major, minor);
        assert(0);
    }

    return 0;
}

static void GetPeakMeasurements(f64 *peak_gbps, f64 *peak_gflops, bool print_device_info = false)
{
    int device;
    CUDACheck(cudaGetDevice(&device));

    cudaDeviceProp device_prop = { 0 };
    CUDACheck(cudaGetDeviceProperties(&device_prop, device));
    
    // Peak GFLOPS
    {
        int sm_count;
        CUDACheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

        int cuda_cores_per_sm = GetCUDACoresPerSM(device_prop.major, device_prop.minor);

        int peak_clock_freq;
        CUDACheck(cudaDeviceGetAttribute(&peak_clock_freq, cudaDevAttrClockRate, device));

        // NOTE(achal): 1 FMA is 2 ops.
        *peak_gflops = (sm_count*cuda_cores_per_sm*2.0*peak_clock_freq*1000.0)/1.0e9;
    }

    // Peak GBPS
    {
        int peak_mem_clock_freq;
        CUDACheck(cudaDeviceGetAttribute(&peak_mem_clock_freq, cudaDevAttrMemoryClockRate, device));

        printf("Peak memory clock frequency: %d kHz\n", peak_mem_clock_freq);

        int bus_width;
        CUDACheck(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device));

        // NOTE(achal): 2.0 is for double transfer (DDR).
        *peak_gbps = (peak_mem_clock_freq*1000.0*2.0)*(bus_width/8.0)/(1000.0*1000.0*1000.0);
    }

    if (print_device_info)
    {
        printf("Device name: %s\n", device_prop.name);
        printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("Total CUDA cores: %d\n", device_prop.multiProcessorCount*GetCUDACoresPerSM(device_prop.major, device_prop.minor));

        printf("Total Global Memory: %.2f GB\n", (device_prop.totalGlobalMem/(1000.f*1000.f*1000.f)));
        printf("Shared Memory (per block): %.2f KB\n", (device_prop.sharedMemPerBlock/1000.f));
        printf("Total Constant Memory: %.2f KB\n", (device_prop.totalConstMem/1000.f));
        printf("L2 Cache Size: %.2f KB\n", (device_prop.l2CacheSize/1000.f));

        printf("Warp Size: %d threads\n", device_prop.warpSize);
        printf("Max threads per block: %d\n", device_prop.maxThreadsPerBlock);
        printf("Max Block dimension: %dx%dx%d\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("Max Grid dimension: %dx%dx%d\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]); 
        printf("32-bit registers (per block): %d\n", device_prop.regsPerBlock);
        printf("SM count: %d\n", device_prop.multiProcessorCount);
        printf("Max blocks per SM: %d\n", device_prop.maxBlocksPerMultiProcessor);
        printf("Max threads per SM: %d\n", device_prop.maxThreadsPerMultiProcessor);
    }
}

#endif // COMMON_CU