#include <stdio.h>
#include <stdint.h>
#include <assert.h>

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
#define CUDACheck(fn_call) CUDACheck_(fn_call, __LINE__)

// Dummy kernel for retrieving PTX version.
__global__ void DummyKernel() {}

static void PrintDeviceInfo()
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

__global__ void ReduceKernel3(u32 count, f32 *input, f32 *output)
{
    int segment_start = blockIdx.x*(2*blockDim.x);
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            int index = segment_start + threadIdx.x;
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
    // ReduceKernel2<<<grid_dim, block_dim>>>(array_count, input.data_ptr<f32>(), output.data_ptr<f32>());
    ReduceKernel3<<<grid_dim, block_dim>>>(array_count, input.data_ptr<f32>(), output.data_ptr<f32>());
    CUDACheck(cudaDeviceSynchronize());
}
#else

#include <cuda_profiler_api.h>
#include <thrust/reduce.h>

void Reduce1(int count, f32 *input)
{
    assert(count <= 2048);

    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel1<<<grid_dim, block_dim>>>(input);
}

void Reduce2(int count, f32 *input, f32 *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel2<<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

void Reduce3(int count, f32 *input, f32 *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel3<<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

void Benchmark()
{
    int warmup_count = 3;
    int rep_count = 20;

    FILE *file = fopen("bench_reduce3.bin", "wb");
    assert(file);
    for (u32 exp = 1; exp <= 30; ++exp)
    {
        u64 array_count = 1 << exp;

        {
            f32 *input = (f32 *)malloc(array_count*sizeof(f32));
            for (u64 i = 0; i < array_count; ++i)
                input[i] = 1.f;

            f32 *d_input = 0;
            CUDACheck(cudaMalloc(&d_input, array_count*sizeof(f32)));
            CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));

            f32 *d_output = 0;
            CUDACheck(cudaMalloc(&d_output, sizeof(f32)));
            CUDACheck(cudaMemset(d_output, 0, sizeof(f32)));

            cudaEvent_t start_event, stop_event;
            CUDACheck(cudaEventCreate(&start_event));
            CUDACheck(cudaEventCreate(&stop_event));

            cudaStream_t stream;
            CUDACheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            // Warmups
            for (int i = 0; i < warmup_count; ++i)
            {
                CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemset(d_output, 0, sizeof(f32)));
                // Reduce2(array_count, d_input, d_output, stream);
                Reduce3(array_count, d_input, d_output, stream);
                // thrust::reduce_into(thrust::cuda::par.on(stream), d_input, d_input + array_count, d_output, 0.f);
            }

            f64 ms = 0;
            for (int i = 0; i < rep_count; ++i)
            {
                CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemset(d_output, 0, sizeof(f32)));

                CUDACheck(cudaEventRecord(start_event, stream));
                // Reduce1(array_count, d_input);
                // Reduce2(array_count, d_input, d_output, stream);
                Reduce3(array_count, d_input, d_output, stream);
                // thrust::reduce_into(thrust::cuda::par.on(stream), d_input, d_input + array_count, d_output, 0.f);
                CUDACheck(cudaEventRecord(stop_event, stream));
                CUDACheck(cudaEventSynchronize(stop_event));

                f32 curr_ms;
                CUDACheck(cudaEventElapsedTime(&curr_ms, start_event, stop_event));
                ms += curr_ms;
            }

            f32 out = 0.f;
            CUDACheck(cudaMemcpy(&out, d_output, sizeof(f32), cudaMemcpyDeviceToHost));

            bool is_correct = ((int)out == array_count);
            printf("[%s] Result (GPU): %f\n", is_correct ? "PASS" : "FAIL", out);

            ms /= rep_count;

            f64 bandwidth = (1000.0*(array_count*sizeof(f32)))/(ms*1024.0*1024.0*1024.0);

            fwrite(&array_count, sizeof(u64), 1, file);
            fwrite(&ms, sizeof(f64), 1, file);
            fwrite(&bandwidth, sizeof(f64), 1, file);

            printf("Elapsed (GPU): %f ms\n", ms);
            printf("Bandwidth: %f GB/s\n", bandwidth);

            CUDACheck(cudaStreamDestroy(stream));
            CUDACheck(cudaEventDestroy(stop_event));
            CUDACheck(cudaEventDestroy(start_event));
            CUDACheck(cudaFree(d_output));
            CUDACheck(cudaFree(d_input));
            free(input);
        }   
    }

    fclose(file);
}

int main()
{
    printf("Thrust version: %d.%d.%d (THRUST_VERSION: %d)\n", THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION, THRUST_VERSION);
    
    if (0)
    {
        PrintDeviceInfo();
    }

    if (1)
    {
        Benchmark();
    }

    return 0;
}
#endif