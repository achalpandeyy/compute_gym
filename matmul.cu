#include <stdio.h>

#include "core_types.h"
#include "core_memory.h"

#define PROFILER 1
#include "profiler.h"

#include "common.cu"
#include "benchmarking.cu"

#include <random>
#include <cublas_v2.h>

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
    enum { block_dim = 32 };
    dim3 block(block_dim, block_dim, 1);
    dim3 grid((n + block_dim - 1)/block_dim, (m + block_dim - 1)/block_dim, 1);
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
    enum { tile_dim = 32 };
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((n + tile_dim - 1)/tile_dim, (m + tile_dim - 1)/tile_dim, 1);
    GEMMKernel2<T, tile_dim><<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

#if 0
template <typename T>
__global__ void GEMMKernel3(u32 m, u32 n, u32 k, )
{
    
}

template <typename T>
static void GEMM3()
{
    
}
#endif

template <typename T>
struct DataDescriptor
{
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

// NOTE(achal): clangd can't find it from the <random> header, for some reason.
namespace std { using default_random_engine = mersenne_twister_engine<unsigned int, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18, 1812433253>; };

template <typename T, u32 elements_per_block>
static Data<T> *CreateData(Arena *arena, DataDescriptor<T> *descriptor, cudaStream_t stream)
{
    Data<T> *data = PushStructZero(arena, Data<T>);
    u32 m = (data->m = descriptor->m);
    u32 n = (data->n = descriptor->n);
    u32 k = (data->k = descriptor->k);

    data->h_A = PushArrayZero(arena, T, m*k);
    data->h_B = PushArrayZero(arena, T, k*n);
    std::normal_distribution<T> distribution(0.0, 1.0);
    std::default_random_engine generator(12345);
    for (u32 i = 0; i < m*k; ++i) data->h_A[i] = distribution(generator);
    for (u32 i = 0; i < k*n; ++i) data->h_B[i] = distribution(generator);

    CUDACheck(cudaMalloc(&data->d_A, m*k*sizeof(*data->d_A)));
    CUDACheck(cudaMalloc(&data->d_B, k*n*sizeof(*data->d_B)));
    CUDACheck(cudaMalloc(&data->d_C, m*n*sizeof(*data->d_C)));

    CUDACheck(cudaMemcpyAsync(data->d_A, data->h_A, m*k*sizeof(*data->d_A), cudaMemcpyHostToDevice, stream));
    CUDACheck(cudaMemcpyAsync(data->d_B, data->h_B, k*n*sizeof(*data->d_B), cudaMemcpyHostToDevice, stream));
    CUDACheck(cudaMemsetAsync(data->d_C, 0, m*n*sizeof(*data->d_C), stream));

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

    b32 result = 1;
    f64 base_tolerance = 1e-2;
    f64 tolerance = base_tolerance * sqrt(k/64.0); // NOTE(achal): This is a heuristic to scale the tolerance based on the matrix size.

#if 1
    T *C_ref = PushArrayZero(scratch.arena, T, m*n);
#if 1
    {
        T *d_C_ref = 0;
        CUDACheck(cudaMalloc(&d_C_ref, m*n*sizeof(*d_C_ref)));
        CUDACheck(cudaMemset(d_C_ref, 0, m*n*sizeof(*d_C_ref)));
        // Assert(data->beta == 0.f);

        // NOTE(achal): cuBLAS, for some reason, uses column-major layout whereas all the matrices are allocated in row-major,
        // so we are computing C^T = alpha*(B^T@A^T) + beta*C^T
        // A^T -> kxm
        // B^T -> nxk
        // C^T -> nxm
        // C = alpha*A@B + beta*C
        // C^T = alpha*(B^T@A^T) + beta*C^T
        cublasHandle_t handle;
        cublasCreate(&handle);
        f32 alpha = 1.f;
        f32 beta = 0.f;
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, data->d_B, n, data->d_A, k, &beta, d_C_ref, n);
        Assert(status == CUBLAS_STATUS_SUCCESS);
        CUDACheck(cudaDeviceSynchronize());
        CUDACheck(cudaMemcpy(C_ref, d_C_ref, m*n*sizeof(*C_ref), cudaMemcpyDeviceToHost));
        CUDACheck(cudaFree(d_C_ref));
        cublasDestroy(handle);
    }
#else
    // NOTE(achal): Sometimes checking against cuBLAS just fail randomly, so keep this here as a fallback
    {
        GEMMCPU(m, n, k, T(1), data->h_A, data->h_B, T(0), C_ref);
    }
#endif

    for (u32 i = 0; i < m*n; ++i)
    {
        f64 abs_error = fabsf(C_gpu[i] - C_ref[i]);
        f64 rel_error = abs_error / (fabsf(C_ref[i]) + 1e-8);
        if (abs_error > tolerance && rel_error > tolerance)
        {
            printf("C_ref[%d] = %f, C_gpu[%d] = %f, abs_error = %f, rel_error = %f\n", i, C_ref[i], i, C_gpu[i], abs_error, rel_error);
            result = 0;
            break;
        }
    }
#endif

    ScratchEnd(&scratch);

    return result;
}

template <typename T, u32 block_dim, u32 coarse_factor>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
    // GEMM(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
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
    if (1)
    {
        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

        DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
        u32 m = desc->m = 64;
        u32 n = desc->n = 64;
        u32 k = desc->k = 64;

        Data<InputType> *data = CreateData<InputType, 0>(scratch.arena, desc, 0);

        // GEMM(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);
        GEMM2(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);

        if (!ValidateGPUOutput<InputType>(scratch.arena, data))
        {
            printf("Failed\n");
            exit(1);
        }
        else
        {
            printf("Passed\n");
        }

        DestroyData<InputType>(data);
        ScratchEnd(&scratch);
    }

    if (0)
    {
        f64 peak_gbps = 0.0;
        f64 peak_gflops = 0.0;
        GetPeakMeasurements(&peak_gbps, &peak_gflops, false);

        printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
        printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
        printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

        FILE *file = 0;
        if (file_name)
            file = fopen(file_name, "wb");

        {
            
            u32 dims[] = {128, 256, 512, 1024, 2048, 4096};
            // u32 dims[] = {1024};
            u32 skip_from_last = 0;
            
            printf("%-20s %-20s %-20s %-20s\n", "Input (m x n x k)", "GBPS", "GFLOPS", "Runtime (ms)");
            for (u32 i = 0; i < ArrayCount(dims) - skip_from_last; ++i)
            {
                Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

                DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
                desc->m = dims[i];
                desc->n = dims[i];
                desc->k = dims[i];
                f64 ms = Benchmark<InputType, 0, 0>(desc);

                f64 gbps = (1000.0*GetDataTransferSize(desc))/(ms*1000.0*1000.0*1000.0);
                f64 gflops = (1000.0*GetFLOPS(desc))/(ms*1000.0*1000.0*1000.0);

                // NOTE(achal): snprintf will write the null terminator but its return value will not include it.
                u32 max_input_label_length = 1023 + 1;

                u8 *input_label = PushArrayZero(scratch.arena, u8, max_input_label_length);
                u32 label_length = (u32)snprintf((char *)input_label, max_input_label_length, "%ux%ux%u", dims[i], dims[i], dims[i]);
                Assert(label_length <= max_input_label_length-1);
                
                printf("%-20s %-20.6f %-20.6f %-20.6f\n", (char *)input_label, gbps, gflops, ms);

                if (file)
                {
                    fwrite(&label_length, sizeof(u32), 1, file);
                    fwrite(input_label, sizeof(u8), label_length, file);

                    fwrite(&gbps, sizeof(f64), 1, file);
                    fwrite(&gflops, sizeof(f64), 1, file);
                    fwrite(&ms, sizeof(f64), 1, file);
                }
                ScratchEnd(&scratch);
            }

        }

        if (file)
            fclose(file);
    }

    EndProfiler();
    PrintPerformanceProfile();
    
    return 0;
}
PROFILER_END_OF_COMPILATION_UNIT;