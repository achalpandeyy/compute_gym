#include <cuda_fp16.h>
#include <stdio.h>
#include <type_traits>

#include "core_types.h"
#include "core_memory.h"
#include "core.h"

#define PROFILER 0
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

        T b(0); // KxN
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

template <typename T, u32 tile_dim, u8 elements_per_thread>
__global__ void GEMMKernel3(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C)
{
    __shared__ T tile_A[tile_dim][tile_dim];
    __shared__ T tile_B[tile_dim][tile_dim];

    // one block computes one output tile
    u32 tile_x = blockIdx.x;
    u32 tile_y = blockIdx.y;

    u32 tx = threadIdx.x;
    u32 ty = threadIdx.y;

    T results[elements_per_thread] = {0};
    u32 step_count = (k + tile_dim - 1)/tile_dim;
    for (u32 step_index = 0; step_index < step_count; ++step_index)
    {
        u32 ax = step_index*tile_dim + tx;
        u32 bx = tile_x*tile_dim + tx;
        for (u8 i = 0; i < elements_per_thread; ++i)
        {
            u32 ay = tile_y*tile_dim + (ty*elements_per_thread + i);
            u32 by = step_index*tile_dim + (ty*elements_per_thread + i);
            
            T a(0);
            if (ax < k && ay < m)
                a = A[ay*k + ax];

            T b(0);
            if (bx < n && by < k)
                b = B[by*n + bx];

            tile_A[ty*elements_per_thread + i][tx] = a;
            tile_B[ty*elements_per_thread + i][tx] = b;
        }
        __syncthreads();

        for (u32 index = 0; index < tile_dim; ++index)
        {
            T b = tile_B[index][tx];
            for (u8 i = 0; i < elements_per_thread; ++i)
            {
                results[i] += tile_A[ty*elements_per_thread + i][index]*b;
            }
        }
        __syncthreads();
    }

    u32 col = tile_x*tile_dim + tx;
    for (u8 i = 0; i < elements_per_thread; ++i)
    {
        u32 row = tile_y*tile_dim + (ty*elements_per_thread + i);
        if (row < m && col < n)
        {
            C[row*n + col] = alpha*results[i] + beta*C[row*n + col];
        }
    }
}

template <typename T>
static void GEMM3(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C, cudaStream_t stream)
{
    enum
    {
        elements_per_thread = 8,
        tile_dim = 64,
    };
    Assert((tile_dim % elements_per_thread) == 0);

    dim3 block_dim(tile_dim, tile_dim/elements_per_thread, 1);
    dim3 elements_per_block(block_dim.x, block_dim.y*elements_per_thread, block_dim.z);
    dim3 grid_dim(IntegerCeil(n, elements_per_block.x), IntegerCeil(m, elements_per_block.y), 1);
    GEMMKernel3<T, tile_dim, elements_per_thread><<<grid_dim, block_dim, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

__global__ void GEMMKernel4(u32 M, u32 N, u32 K, half *A, half *B, f32 *C)
{
 enum {m = 16, n = 8, k = 16};

 __shared__ alignas(16) half shared_A[m][k];
 __shared__ alignas(16) half shared_B[k][n];

 int bidx = blockIdx.x;
 int bidy = blockIdx.y;
 int tid = threadIdx.x;

 f32 tile_c[4] = {0.f, 0.f, 0.f, 0.f};
 for(int step = 0; step < K/k; ++step)
 {
  if(tid < 16)
  {
   for(int i = 0; i < 16; ++i)
    shared_A[tid][i] = A[bidy*(m*k*(K/k)) + tid*(k*(K/k)) + step*(k) + i*(1)];
 
   for(int i = 0; i < 8; ++i)
    shared_B[tid][i] = B[bidx*(n*k*(K/k)) + i*(k*(K/k)) + step*(k) + tid*(1)];
  }

  half tile_a[8];
  u32 *regs_a = (u32 *)tile_a;
  u32 addr_a = __cvta_generic_to_shared(&shared_A[tid % 16][(tid / 16)*8]);
  asm volatile(
   "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
   "{%0, %1, %2, %3}, "
   "[%4];"
   : "=r"(regs_a[0]), "=r"(regs_a[1]), "=r"(regs_a[2]), "=r"(regs_a[3])
   : "r"(addr_a)
  );

  // Even though the last sixteen threads, in the warp, have the
  // same smem address as the first sixteen, the instruction works
  // in a way that all the threads end up with the correct values,
  // as expected by mma.
  half tile_b[4];
  u32 *regs_b = (u32 *)tile_b;
  u32 addr_b = __cvta_generic_to_shared(&shared_B[tid % 16][0]);
  asm volatile(
   "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
   "{%0, %1}, "
   "[%2];"
   : "=r"(regs_b[0]), "=r"(regs_b[1])
   : "r"(addr_b)
  );

  asm volatile(
   "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
   "{%0, %1, %2, %3}, " // d
   "{%4, %5, %6, %7}, " // a
   "{%8, %9}, " // b
   "{%0, %1, %2, %3};\n" // c
   : "+f"(tile_c[0]), "+f"(tile_c[1]), "+f"(tile_c[2]), "+f"(tile_c[3])
   : "r"(regs_a[0]),  "r"(regs_a[1]),  "r"(regs_a[2]),  "r"(regs_a[3]), 
     "r"(regs_b[0]),  "r"(regs_b[1])
  );
 }

 int group_id = tid >> 2;
 for(int i = 0; i < ArrayCount(tile_c); ++i)
 {
  int row = group_id + (i/2)*8;
  int col = 2*(tid % 4) + (i % 2);
  C[bidy*(m*(N/n)*n) + row*((N/n)*n) + bidx*(n) + col*(1)] = tile_c[i];
 }
}

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

#if 0
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
 // GEMM2(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
 GEMM3(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
}

template <typename T>
static void Debug_PrintMatrix(Arena *arena, T *matrix, u32 m, u32 n, bool col_major = true)
{
 CUDACheck(cudaDeviceSynchronize());
 
 Scratch scratch = ScratchBegin(arena);
 T *h_matrix = PushArrayZero(scratch.arena, T, m*n);
 CUDACheck(cudaMemcpy(h_matrix, matrix, m*n*sizeof(*h_matrix), cudaMemcpyDeviceToHost));
 for (u32 r = 0; r < m; ++r)
 {
  for (u32 c = 0; c < n; ++c)
  {
   int index = col_major ? (c*m + r) : (r*n + c);
   if constexpr (std::is_floating_point<T>::value)
    printf("%10.6f ", h_matrix[index]);
   else
    printf("%2d ", (u16)h_matrix[index]);
  }
  printf("\n");
 }
 ScratchEnd(&scratch);
}

int main(int argc, char **argv)
{
 BeginProfiler();

 char *file_name = 0;
 if (argc == 2)
 {
  file_name = argv[1];
 }

 {
  Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

  int M = 32; Assert((M%16) == 0);
  int K = 32; Assert((K%16) == 0);
  int N = 16; Assert((N% 8) == 0);
  
  enum {m = 16, n = 8, k = 16};
  static_assert(std::is_trivially_constructible<half>::value);

  half *h_A = PushArrayZero(scratch.arena, half, M*K);
  for(int r = 0; r < M; ++r)
  {
   for(int c = 0; c < K; ++c)
   {
    f32 v = (f32)((r/m)*(K/k) + (c/k));
    h_A[r*K + c] = __float2half(v); // row-major
   }
  }

  half *h_B = PushArrayZero(scratch.arena, half, K*N);
  for(int r = 0; r < K; ++r)
  {
   for(int c = 0; c < N; ++c)
   {
    f32 v = (f32)((r/k)*(N/n) + (c/n));
    h_B[c*K + r] = __float2half(v); // col-major
   }
  }
  
  half *d_A = 0, *d_B = 0;
  CUDACheck(cudaMalloc(&d_A, M*K*sizeof(half)));
  CUDACheck(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
  CUDACheck(cudaMalloc(&d_B, K*N*sizeof(half)));
  CUDACheck(cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice));

  f32 *d_C = 0;
  CUDACheck(cudaMalloc(&d_C, M*N*sizeof(f32)));
  dim3 grid_dim(N/n, M/m, 1);
  GEMMKernel4<<<grid_dim, 32>>>(M, N, K, d_A, d_B, d_C);

  // printf("A:\n");
  // Debug_PrintMatrix(scratch.arena, d_A, M, K, false);
  // printf("B:\n");
  // Debug_PrintMatrix(scratch.arena, d_B, K, N);

  CUDACheck(cudaDeviceSynchronize());
  printf("C:\n");
  Debug_PrintMatrix(scratch.arena, d_C, M, N, false);

  ScratchEnd(&scratch);

  return 0;
 }

 using InputType = f32;
 if (0)
 {
     Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

     DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
     u32 m = desc->m = 2048; // 15;
     u32 n = desc->n = 2048; // 17;
     u32 k = desc->k = 2048; // 16;

     Data<InputType> *data = CreateData<InputType, 0>(scratch.arena, desc, 0);
     // GEMM(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);
     // GEMM2(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);
     GEMM3(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);

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

 if (1)
 {
     f64 peak_gbps = 0.0;
     f64 peak_gflops = 0.0;
     GetPeakMeasurements(&peak_gbps, &peak_gflops, 1);

     printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
     printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
     printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

     FILE *file = 0;
     if (file_name)
         file = fopen(file_name, "wb");

     {
         
         // u32 dims[] = {128, 256, 512, 1024, 2048, 4096};
         u32 dims[] = {4096};
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