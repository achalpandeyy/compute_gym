#include <stdio.h>

#include "core_types.h"
#include "core_memory.h"
#include "core.h"

#define PROFILER 1
#include "profiler.h"

#include "common.cuh"

#include "benchmarking.cu"

template <typename T>
static void GEMMCPU(u32 m, u32 n, u32 p, T *A, T *B, T *C)
{
    PROFILE_SCOPE_BEGIN("GEMMCPU");
    for (u32 y = 0; y < m; ++y)
    {
        for (u32 x = 0; x < n; ++x)
        {
            T result(0);
            for (u32 index = 0; index < p; ++index)
            {
                result += A[y*p + index]*B[index*n + x];
            }
            C[y*n + x] = result;
        }
    }
    PROFILE_SCOPE_END();
}

template <typename T>
__global__ void GEMMKernel(u32 m, u32 n, u32 p, T *A, T *B, T *C)
{
    u32 x = blockIdx.x*blockDim.x + threadIdx.x;
    u32 y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < n && y < m)
    {
        T result(0);
        for (u32 index = 0; index < p; ++index)
        {
            result += A[y*p + index]*B[index*n + x];
        }
        C[y*n + x] = result;
    }
}

template <typename T>
static void GEMM(u32 m, u32 n, u32 p, T *A, T *B, T *C, cudaStream_t stream)
{
    enum { tile_dim = 16 };
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((n + tile_dim - 1)/tile_dim, (m + tile_dim - 1)/tile_dim, 1);
    GEMMKernel<T><<<grid, block, 0, stream>>>(m, n, p, A, B, C);
}

template <typename T, u32 tile_dim>
__global__ void GEMMKernel2(u32 m, u32 n, u32 p, T *A, T *B, T *C)
{
    __shared__ T tile_A[tile_dim][tile_dim];
    __shared__ T tile_B[tile_dim][tile_dim];

    // one block computes one output tile
    u32 tile_x = blockIdx.x;
    u32 tile_y = blockIdx.y;

    u32 tx = threadIdx.x;
    u32 ty = threadIdx.y;

    T result(0);
    u32 step_count = (p + tile_dim - 1)/tile_dim;
    for (u32 step_index = 0; step_index < step_count; ++step_index)
    {
        // (tile_x, tile_y) -> tile_A
        u32 ax = step_index*tile_dim + tx;
        u32 ay = tile_y*tile_dim + ty;

        T a(0);
        if (ax < p && ay < m)
            a = A[ay*p + ax];

        // (tile_x, tile_y) -> tile_B
        u32 bx = tile_x*tile_dim + tx;
        u32 by = step_index*tile_dim + ty;

        T b(0); // PxN
        if (bx < n && by < p)
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
        C[row*n + col] = (T)result;
    }
}

template <typename T>
static void GEMM2(u32 m, u32 n, u32 p, T *A, T *B, T *C, cudaStream_t stream)
{
    enum { tile_dim = 16 };
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((n + tile_dim - 1)/tile_dim, (m + tile_dim - 1)/tile_dim, 1);
    GEMMKernel2<T><<<grid, block, 0, stream>>>(m, n, p, A, B, C);
}

template <typename T>
struct DataDescriptor
{
    DataDescriptor *next;
    u32 m, n, p;
};

template <typename T>
struct Data
{
    u32 m, n, p;

    T *d_A;
    T *d_B;
    T *d_C;

    T *h_A;
    T *h_B;
};

template <typename T>
static u64 GetDataTransferSize(DataDescriptor<T> *descriptor)
{
    return (descriptor->m*descriptor->p + descriptor->p*descriptor->n + descriptor->m*descriptor->n)*sizeof(T);
}

template <typename T, u32 elements_per_block>
static Data<T> *CreateData(Arena *arena, DataDescriptor<T> *descriptor, cudaStream_t stream)
{
    Data<T> *data = PushStructZero(arena, Data<T>);
    u32 m = (data->m = descriptor->m);
    u32 n = (data->n = descriptor->n);
    u32 p = (data->p = descriptor->p);

    data->h_A = PushArrayZero(arena, T, m*p);
    data->h_B = PushArrayZero(arena, T, p*n);
    for (u32 i = 0; i < m*p; ++i) data->h_A[i] = T(1);
    for (u32 i = 0; i < p*n; ++i) data->h_B[i] = T(1);

    CUDACheck(cudaMalloc(&data->d_A, m*p*sizeof(*data->d_A)));
    CUDACheck(cudaMalloc(&data->d_B, p*n*sizeof(*data->d_B)));
    CUDACheck(cudaMalloc(&data->d_C, m*n*sizeof(*data->d_C)));

    CUDACheck(cudaMemcpyAsync(data->d_A, data->h_A, m*p*sizeof(*data->d_A), cudaMemcpyHostToDevice, stream));
    CUDACheck(cudaMemcpyAsync(data->d_B, data->h_B, p*n*sizeof(*data->d_B), cudaMemcpyHostToDevice, stream));

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
    u32 p = data->p;

    T *C_gpu = PushArrayZero(scratch.arena, T, m*n);
    CUDACheck(cudaMemcpy(C_gpu, data->d_C, m*n*sizeof(*C_gpu), cudaMemcpyDeviceToHost));

    T *C = PushArrayZero(scratch.arena, T, m*n);
    GEMMCPU(m, n, p, data->h_A, data->h_B, C);

    b32 result = (memcmp(C_gpu, C, m*n*sizeof(*C)) == 0);

    ScratchEnd(&scratch);

    return result;    
}

template <typename T, u32 block_dim, u32 coarse_factor>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
    GEMM(data->m, data->n, data->p, data->d_A, data->d_B, data->d_C, stream);
}

int main(int argc, char **argv)
{
    BeginProfiler();

    char *file_name = 0;
    if (argc == 2)
    {
        file_name = argv[1];
    }

    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);

    using InputType = f32;
    if (1)
    {
        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(8)));

        DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
        u32 m = desc->m = 64;
        u32 n = desc->n = 64;
        u32 p = desc->p = 64;

        Data<InputType> *data = CreateData<InputType, 0>(scratch.arena, desc, 0);

        GEMM(m, n, p, data->d_A, data->d_B, data->d_C, 0);

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

    {
        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(8)));
        // benchmarking data
        DataDescriptorList<InputType> *benching_data = PushStructZero(scratch.arena, DataDescriptorList<InputType>);
        {
            // 64x64
            {
                DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
                desc->m = 64;
                desc->n = 64;
                desc->p = 64;
                ListPush(benching_data->first, benching_data->last, next, desc);
            }

            // 128x128
            {
                DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
                desc->m = 128;
                desc->n = 128;
                desc->p = 128;
                ListPush(benching_data->first, benching_data->last, next, desc);
            }
        }

        Benchmark<InputType, 0, 0>(benching_data, peak_gbps, peak_gflops, file_name);
        ScratchEnd(&scratch);
    }
    

    EndProfiler();
    // PrintPerformanceProfile();
    
    return 0;
}
PROFILER_END_OF_COMPILATION_UNIT;