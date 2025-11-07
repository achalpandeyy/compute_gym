#include <stdio.h>

#include "core_types.h"
#include "core_memory.h"
#include "core.h"

#define PROFILER 1
#include "profiler.h"

#include "common.cuh"

#include "benchmarking.cu"

template <typename T>
static void GEMMCPU(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C)
{
    PROFILE_SCOPE_BEGIN("GEMMCPU");
    for (u32 y = 0; y < m; ++y)
    {
        for (u32 x = 0; x < n; ++x)
        {
            T result(0);
            for (u32 index = 0; index < k; ++index)
            {
                result += A[y*k + index]*B[index*n + x];
            }
            C[y*n + x] = alpha*result + beta*C[y*n + x];
        }
    }
    PROFILE_SCOPE_END();
}

template <typename T>
__global__ void GEMMKernel(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C)
{
    u32 x = blockIdx.x*blockDim.x + threadIdx.x;
    u32 y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < n && y < m)
    {
        T result(0);
        for (u32 index = 0; index < k; ++index)
        {
            result += A[y*k + index]*B[index*n + x];
        }
        C[y*n + x] = alpha*result + beta*C[y*n + x];
    }
}

template <typename T>
static void GEMM(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C, cudaStream_t stream)
{
    enum { tile_dim = 16 };
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((n + tile_dim - 1)/tile_dim, (m + tile_dim - 1)/tile_dim, 1);
    GEMMKernel<T><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

template <typename T, u32 tile_dim>
__global__ void GEMMKernel2(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C)
{
    __shared__ T tile_A[tile_dim][tile_dim];
    __shared__ T tile_B[tile_dim][tile_dim];

    // one block computes one output tile
    u32 tile_x = blockIdx.x;
    u32 tile_y = blockIdx.y;

    u32 tx = threadIdx.x;
    u32 ty = threadIdx.y;

    T result(0);
    u32 step_count = (k + tile_dim - 1)/tile_dim;
    for (u32 step_index = 0; step_index < step_count; ++step_index)
    {
        // (tile_x, tile_y) -> tile_A
        u32 ax = step_index*tile_dim + tx;
        u32 ay = tile_y*tile_dim + ty;

        T a(0);
        if (ax < k && ay < m)
            a = A[ay*k + ax];

        // (tile_x, tile_y) -> tile_B
        u32 bx = tile_x*tile_dim + tx;
        u32 by = step_index*tile_dim + ty;

        T b(0); // PxN
        if (bx < n && by < k)
            b = B[by*n + bx];
        
        tile_A[ty][tx] = a;
        tile_B[ty][tx] = b;
        __syncthreads();
        
        for (int index = 0; index < tile_dim; ++index)
        {
            result += tile_A[ty][index]*tile_B[index][tx];
        }
        __syncthreads();
    }

    u32 col = tile_x*tile_dim + tx;
    u32 row = tile_y*tile_dim + ty;

    if (row < m && col < n)
    {
        C[row*n + col] = alpha*result + beta*C[row*n + col];
    }
}

template <typename T>
static void GEMM2(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C, cudaStream_t stream)
{
    enum { tile_dim = 16 };
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((n + tile_dim - 1)/tile_dim, (m + tile_dim - 1)/tile_dim, 1);
    GEMMKernel2<T, tile_dim><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

template <typename T>
struct DataDescriptor
{
    DataDescriptor *next;
    u32 m, n, k;
};

template <typename T>
struct Data
{
    u32 m, n, k;

    T *d_A;
    T *d_B;
    T *d_C;

    T *h_A;
    T *h_B;
};

template <typename T>
static u64 GetDataTransferSize(DataDescriptor<T> *descriptor)
{
    return (descriptor->m*descriptor->k + descriptor->k*descriptor->n + descriptor->m*descriptor->n)*sizeof(T);
}

template <typename T>
static u64 GetFLOPS(DataDescriptor<T> *descriptor)
{
    return 2ull*descriptor->m*descriptor->n*descriptor->k;
}

template <typename T, u32 elements_per_block>
static Data<T> *CreateData(Arena *arena, DataDescriptor<T> *descriptor, cudaStream_t stream)
{
    Data<T> *data = PushStructZero(arena, Data<T>);
    u32 m = (data->m = descriptor->m);
    u32 n = (data->n = descriptor->n);
    u32 k = (data->k = descriptor->k);

    data->h_A = PushArrayZero(arena, T, m*k);
    data->h_B = PushArrayZero(arena, T, k*n);
    for (u32 i = 0; i < m*k; ++i) data->h_A[i] = T(1);
    for (u32 i = 0; i < k*n; ++i) data->h_B[i] = T(1);

    CUDACheck(cudaMalloc(&data->d_A, m*k*sizeof(*data->d_A)));
    CUDACheck(cudaMalloc(&data->d_B, k*n*sizeof(*data->d_B)));
    CUDACheck(cudaMalloc(&data->d_C, m*n*sizeof(*data->d_C)));

    CUDACheck(cudaMemcpyAsync(data->d_A, data->h_A, m*k*sizeof(*data->d_A), cudaMemcpyHostToDevice, stream));
    CUDACheck(cudaMemcpyAsync(data->d_B, data->h_B, k*n*sizeof(*data->d_B), cudaMemcpyHostToDevice, stream));

    return data;
}

template <typename T>
static void DestroyData(Data<T> *data)
{
    CUDACheck(cudaFree(data->d_C));
    CUDACheck(cudaFree(data->d_B));
    CUDACheck(cudaFree(data->d_A));
}

template <typename T>
static b32 ValidateGPUOutput(Arena *arena, Data<T> *data)
{
    Scratch scratch = ScratchBegin(arena);

    u32 m = data->m;
    u32 n = data->n;
    u32 k = data->k;

    T *C_gpu = PushArrayZero(scratch.arena, T, m*n);
    CUDACheck(cudaMemcpy(C_gpu, data->d_C, m*n*sizeof(*C_gpu), cudaMemcpyDeviceToHost));

    T *C = PushArrayZero(scratch.arena, T, m*n);
    GEMMCPU(m, n, k, T(1), data->h_A, data->h_B, T(0), C);

    b32 result = (memcmp(C_gpu, C, m*n*sizeof(*C)) == 0);

    ScratchEnd(&scratch);

    return result;    
}

template <typename T, u32 block_dim, u32 coarse_factor>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
    GEMM2(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
}

int main(int argc, char **argv)
{
    BeginProfiler();

    char *file_name = 0;
    if (argc == 2)
    {
        file_name = argv[1];
    }

    using InputType = f32;
    if (0)
    {
        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(8)));

        DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
        u32 m = desc->m = 64;
        u32 n = desc->n = 64;
        u32 k = desc->k = 64;

        Data<InputType> *data = CreateData<InputType, 0>(scratch.arena, desc, 0);

        // GEMM(m, n, k, T(1), data->d_A, data->d_B, T(0), data->d_C, 0);
        GEMM2(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);

        CUDACheck(cudaDeviceSynchronize());

        if (!ValidateGPUOutput<InputType>(scratch.arena, data))
        {
            printf("Failed\n");
            exit(1);
        }
        else
        {
            printf("Passed\n");
        }
    }

    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);

    printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
    printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
    printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    if (file_name)
    {
        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(8)));

        // benchmarking data
        DataDescriptorList<InputType> *benching_data = PushStructZero(scratch.arena, DataDescriptorList<InputType>);
        {
            u32 dims[] = {128, 256, 512, 1024, 2048, 4096};
            u32 skip_from_last = 2;
            for (u32 i = 0; i < ArrayCount(dims) - skip_from_last; ++i)
            {
                DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
                desc->m = dims[i];
                desc->n = dims[i];
                desc->k = dims[i];
                ListPush(benching_data->first, benching_data->last, next, desc);
            }
        }

        Benchmark<InputType, 0, 0>(benching_data, peak_gbps, peak_gflops, file_name);
        ScratchEnd(&scratch);
    }
    
    EndProfiler();
    PrintPerformanceProfile();
    
    return 0;
}
PROFILER_END_OF_COMPILATION_UNIT;