#ifndef COMMON_CUH
#define COMMON_CUH

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#define ArrayCount(array) (sizeof(array)/sizeof(array[0]))

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
#define CUDACheck(...) CUDACheck_(__VA_ARGS__, __LINE__)

// Dummy kernel for retrieving PTX version.
__global__ void DummyKernel() {}

static void GetPeakMeasurements(f64 *peak_gbps, f64 *peak_gflops, bool print_device_info = false)
{
    cudaFuncAttributes attr;
    CUDACheck(cudaFuncGetAttributes(&attr, DummyKernel));

    int major_ver = attr.ptxVersion/10;
    int minor_ver = attr.ptxVersion%10;
    
    int device;
    CUDACheck(cudaGetDevice(&device));

    cudaDeviceProp device_prop = { 0 };
    CUDACheck(cudaGetDeviceProperties(&device_prop, device));
    
    // Peak GFLOPS
    {
        int sm_count;
        CUDACheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

        // NOTE(achal): Compute capability 7.5 has 64 CUDA cores per SM.
        assert(device_prop.major == 7 && device_prop.minor == 5);
        int cuda_cores_per_sm = 64;

        int peak_clock_freq;
        CUDACheck(cudaDeviceGetAttribute(&peak_clock_freq, cudaDevAttrClockRate, device));

        // NOTE(achal): 1 FMA is 2 ops.
        *peak_gflops = (sm_count*cuda_cores_per_sm*2.0*peak_clock_freq*1000.0)/1.0e9;
    }

    // Peak GBPS
    {
        int peak_mem_clock_freq;
        CUDACheck(cudaDeviceGetAttribute(&peak_mem_clock_freq, cudaDevAttrMemoryClockRate, device));

        int bus_width;
        CUDACheck(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device));

        // NOTE(achal): 2.0 is for double transfer (DDR).
        *peak_gbps = (peak_mem_clock_freq*1000.0*2.0)*(bus_width/8.0)/(1024.0*1024.0*1024.0);
    }

    if (print_device_info)
    {
        printf("Device name: %s\n", device_prop.name);
        printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("PTX version: %d.%d\n", major_ver, minor_ver);

        printf("Total Global Memory: %.2f GB\n", (device_prop.totalGlobalMem/(1024.f*1024.f*1024.f)));
        printf("Shared Memory (per block): %.2f KB\n", (device_prop.sharedMemPerBlock/1024.f));
        printf("Total Constant Memory: %.2f KB\n", (device_prop.totalConstMem/1024.f));

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

#endif