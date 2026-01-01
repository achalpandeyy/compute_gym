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


static void GEMMCPU(int M, int N, int K, half alpha, half *A, half *B, half beta, float *C)
{
 PROFILE_SCOPE_BEGIN("GEMMCPU");
 for(int row = 0; row < M; ++row)
 {
  for(int col = 0; col < N; ++col)
  {
   float result = 0.f;
   for(int index = 0; index < K; ++index)
   {
    result += __half2float(A[row*K + index])*__half2float(B[index*N + col]);
   }
   C[row*N + col] = __half2float(alpha)*result + __half2float(beta)*C[row*N + col];
  }
 }
 PROFILE_SCOPE_END();
}

__global__ void GEMMKernel(int M, int N, int K, half alpha, half *A, half *B, half beta, float *C)
{
 int row = blockIdx.y*blockDim.y + threadIdx.y;
 int col = blockIdx.x*blockDim.x + threadIdx.x;

 if(row < M && col < N)
 {
  float result = 0.f;
  for(int index = 0; index < K; ++index)
  {
   result += __half2float(A[row*K + index])*__half2float(B[index*N + col]);
  }
  C[row*N + col] = __half2float(alpha)*result + __half2float(beta)*C[row*N + col];
 }
}

static void GEMM(int M, int N, int K, half alpha, half *A, half *B, half beta, float *C, cudaStream_t stream)
{
 enum { block_dim = 32 };
 dim3 block(block_dim, block_dim, 1);
 dim3 grid((N + block_dim - 1)/block_dim, (M + block_dim - 1)/block_dim, 1);
 GEMMKernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}

__global__ void GEMMKernel2(int M, int N, int K, half alpha, half *A, half *B, half beta, float *C)
{
 enum { tile_dim = 32 };
 __shared__ half tile_A[tile_dim][tile_dim];
 __shared__ half tile_B[tile_dim][tile_dim];

 int bidm = blockIdx.y;
 int bidn = blockIdx.x;
 int tm = threadIdx.y;
 int tn = threadIdx.x;

 float result = 0.f;
 for(int step = 0; step < K/tile_dim; ++step)
 {
  int row = bidm*tile_dim + tm;
  int col = step*tile_dim + tn;
  tile_A[tm][tn] = A[row*K + col];

  row = step*tile_dim + tm;
  col = bidn*tile_dim + tn;
  tile_B[tm][tn] = B[row*N + col];
  __syncthreads();
     
  for(int k = 0; k < tile_dim; ++k)
   result += __half2float(tile_A[tm][k])*__half2float(tile_B[k][tn]);

  __syncthreads();
 }

 int row = bidm*tile_dim + tm;
 int col = bidn*tile_dim + tn;
 C[row*N + col] = __half2float(alpha)*result + __half2float(beta)*C[row*N + col];
}

static void GEMM2(int m, int n, int k, half alpha, half *A, half *B, half beta, float *C, cudaStream_t stream)
{
 enum { tile_dim = 32 };
 dim3 block(tile_dim, tile_dim, 1);
 dim3 grid(n/tile_dim, m/tile_dim, 1);
 assert((m % tile_dim) == 0);
 assert((n % tile_dim) == 0);
 assert((k % tile_dim) == 0);
 GEMMKernel2<<<grid, block, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
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
 for(u32 step_index = 0; step_index < step_count; ++step_index)
 {
  u32 ax = step_index*tile_dim + tx;
  u32 bx = tile_x*tile_dim + tx;
  for(u8 i = 0; i < elements_per_thread; ++i)
  {
   u32 ay = tile_y*tile_dim + (ty*elements_per_thread + i);
   u32 by = step_index*tile_dim + (ty*elements_per_thread + i);
   
   T a(0);
   if(ax < k && ay < m)
    a = A[ay*k + ax];

   T b(0);
   if(bx < n && by < k)
    b = B[by*n + bx];

   tile_A[ty*elements_per_thread + i][tx] = a;
   tile_B[ty*elements_per_thread + i][tx] = b;
  }
  __syncthreads();

  for(u32 index = 0; index < tile_dim; ++index)
  {
   T b = tile_B[index][tx];
   for(u8 i = 0; i < elements_per_thread; ++i)
   {
    results[i] += tile_A[ty*elements_per_thread + i][index]*b;
   }
  }
  __syncthreads();
 }

 u32 col = tile_x*tile_dim + tx;
 for(u8 i = 0; i < elements_per_thread; ++i)
 {
  u32 row = tile_y*tile_dim + (ty*elements_per_thread + i);
  if(row < m && col < n)
  {
   C[row*n + col] = alpha*results[i] + beta*C[row*n + col];
  }
 }
}

template <typename T>
static void GEMM3(u32 m, u32 n, u32 k, T alpha, T *A, T *B, T beta, T *C, cudaStream_t stream)
{
 enum {elements_per_thread = 8, tile_dim = 64};
 Assert((tile_dim % elements_per_thread) == 0);

 dim3 block_dim(tile_dim, tile_dim/elements_per_thread, 1);
 dim3 elements_per_block(block_dim.x, block_dim.y*elements_per_thread, block_dim.z);
 dim3 grid_dim(IntegerCeil(n, elements_per_block.x), IntegerCeil(m, elements_per_block.y), 1);
 GEMMKernel3<T, tile_dim, elements_per_thread><<<grid_dim, block_dim, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

__device__ u8 *g_debug_buffer;

__device__ void LoadMatrix_x4(void *addr, uint32_t *data)
{
 uint32_t smem_ptr = __cvta_generic_to_shared(addr);
 asm volatile(
  "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
  "{%0, %1, %2, %3}, "
  "[%4];"
  : "=r"(data[0]), "=r"(data[1]), "=r"(data[2]), "=r"(data[3])
  : "r"(smem_ptr)
 );
}

__device__ void LoadMatrix_x2(void *addr, uint32_t *data)
{
 uint32_t smem_ptr = __cvta_generic_to_shared(addr);
 asm volatile(
  "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
  "{%0, %1}, "
  "[%2];"
  : "=r"(data[0]), "=r"(data[1])
  : "r"(smem_ptr)
 );
}

__device__ void MMA_m16n8k16(uint32_t *A, uint32_t *B, float *C)
{
 asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
  "{%0, %1, %2, %3}, " // d
  "{%4, %5, %6, %7}, " // a
  "{%8, %9}, " // b
  "{%0, %1, %2, %3};" // c
  : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
  :  "r"(A[0]),  "r"(A[1]),  "r"(A[2]),  "r"(A[3]), 
     "r"(B[0]),  "r"(B[1])
 );
}

template<int M, int N, int K, int M_tile, int N_tile>
__global__ void GEMMKernel4(half *A, half *B, float *C)
{
 enum {m = 16, n = 8, k = 16};

 __shared__ uint4 shared_A[M_tile*32][2];
 __shared__ uint4 shared_B[16][(N_tile*32)/8];

 int bidn = blockIdx.x;
 int bidm = blockIdx.y;
 int tn = threadIdx.x;
 int tm = threadIdx.y;
 int tid = tm*blockDim.x + tn;
 int wid = tid/32;
 int lane_id = tid % 32;
 int wn = wid % 4;
 int wm = wid / 4;

 float tile_c[M_tile][N_tile][4] = {0.f,};
 for(int step = 0; step < K/k; ++step)
 {
  // Assert(blockDim.x==16 && blockDim.y==16);
  if(tid < (M_tile*32)*2) // 2 uint4 loads per row
  {
   int base_row = bidm*(M_tile*32);
   int base_col = step*16;
   int row = tid % (M_tile*32);
   int col = tid / (M_tile*32);
   uint4 *src = (uint4 *)&A[(base_row + row)*K + (base_col + col*8)];
   shared_A[row][col] = src[0];
  }

  if(tid < k*8) // 8 uint64 loads per row
  {
   int base_row = step*16;
   int base_col = bidn*(N_tile*32);
   int row = tid % 16;
   int col = tid / 16;
   uint4 *src = (uint4 *)&B[(base_row + row)*N + (base_col + col*8)];
   shared_B[row][col] = src[0];
  }
  __syncthreads();
  
  for(int iwm = 0; iwm < M_tile; ++iwm)
  {
   half tile_a[8];
   int row = iwm*32 + wm*16 + (lane_id % 16);
   int col = lane_id / 16;
   LoadMatrix_x4(&shared_A[row][col], (uint32_t *)tile_a);
   for(int iwn = 0; iwn < N_tile; ++iwn)
   {
    // Even though the last sixteen threads, in the warp, have the
    // same smem address as the first sixteen, the instruction works
    // in a way that all the threads end up with the correct values,
    // as expected by mma.
    half tile_b[4];
    int row = lane_id % 16;
    int col = iwn*4 + wn;
    LoadMatrix_x2(&shared_B[row][col], (uint32_t *)tile_b);
    MMA_m16n8k16((uint32_t *)tile_a, (uint32_t *)tile_b, tile_c[iwm][iwn]);
  }
 }
  __syncthreads();
 }

 for(int iwm = 0; iwm < M_tile; ++iwm)
 {
  for(int iwn = 0; iwn < N_tile; ++iwn)
  {
   int base_row = bidm*(M_tile*32) + iwm*32 + wm*16;
   int base_col = bidn*(N_tile*32) + iwn*32 + wn*8;

   C[(base_row + (lane_id/4    ))*N + (base_col + (2*(lane_id % 4)    ))] = tile_c[iwm][iwn][0];
   C[(base_row + (lane_id/4    ))*N + (base_col + (2*(lane_id % 4) + 1))] = tile_c[iwm][iwn][1];
   C[(base_row + (lane_id/4 + 8))*N + (base_col + (2*(lane_id % 4)    ))] = tile_c[iwm][iwn][2];
   C[(base_row + (lane_id/4 + 8))*N + (base_col + (2*(lane_id % 4) + 1))] = tile_c[iwm][iwn][3];
  }
 }
}

template<int M_tile, int N_tile>
static void GEMM4(int M, int N, int K, half alpha, half *A, half *B, half beta, f32 *C, cudaStream_t stream)
{
 enum {m = 16, n = 8, k = 16};
 Assert((M%m) == 0);
 Assert((N%n) == 0);
 Assert((K%k) == 0);

 dim3 block_dim(16, 16, 1);
 dim3 warps(4, 2, 1);
 Assert(block_dim.x*block_dim.y == warps.x*warps.y*32);

 dim3 tile_dim(warps.x*(n*N_tile), warps.y*(m*M_tile), 1);
 Assert((M % tile_dim.y) == 0);
 Assert((N % tile_dim.x) == 0);
 Assert((K % k) == 0);
 dim3 grid_dim(N/tile_dim.x, M/tile_dim.y, 1);

 if(M==4096 && N==4096 && K==4096)
 {
  GEMMKernel4<4096, 4096, 4096, M_tile, N_tile><<<grid_dim, block_dim, 0, stream>>>(A, B, C);
 }
 else if(M==64 && N==64 && K==64)
 {
  GEMMKernel4<64, 64, 64, M_tile, N_tile><<<grid_dim, block_dim, 0, stream>>>(A, B, C);
 }
 else if(M==128 && N==128 && K==128)
 {
  GEMMKernel4<128, 128, 128, M_tile, N_tile><<<grid_dim, block_dim, 0, stream>>>(A, B, C);
 }
 else if(M==32 && N==512 && K==32)
 {
  GEMMKernel4<32, 512, 32, M_tile, N_tile><<<grid_dim, block_dim, 0, stream>>>(A, B, C);
 }
 else if(M==2048 && N==2048 && K==2048)
 {
  GEMMKernel4<2048, 2048, 2048, M_tile, N_tile><<<grid_dim, block_dim, 0, stream>>>(A, B, C);
 }
 else if(M==64 && N==64 && K==32)
 {
  GEMMKernel4<64, 64, 32, M_tile, N_tile><<<grid_dim, block_dim, 0, stream>>>(A, B, C);
 }
 else
 {
  Assert(!"Invalid code path");
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
 f32 *d_C;

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
 std::normal_distribution<f32> distribution(0.f, 1.f);
 std::default_random_engine generator(12345);
 for(int r = 0; r < m; ++r) for(int c = 0; c < k; ++c)
 {
  f32 v = distribution(generator);
  if constexpr(std::is_same<T, half>::value)
   data->h_A[r*k + c] = __float2half(v);
  else
   data->h_A[r*k + c] = v;
 }
 for(int r = 0; r < k; ++r) for(int c = 0; c < n; ++c)
 {
  f32 v = distribution(generator);
  if constexpr(std::is_same<T, half>::value)
   data->h_B[r*n + c] = __float2half(v);
  else
   data->h_B[r*n + c] = v;
 }

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

 f32 *C_gpu = PushArrayZero(scratch.arena, f32, m*n);
 CUDACheck(cudaMemcpy(C_gpu, data->d_C, m*n*sizeof(*C_gpu), cudaMemcpyDeviceToHost));

 b32 result = 1;
 f64 base_tolerance = 1e-2;
 f64 tolerance = base_tolerance * sqrt(k/64.0); // NOTE(achal): This is a heuristic to scale the tolerance based on the matrix size.

#if 1
 f32 *C_ref = PushArrayZero(scratch.arena, f32, m*n);
#if 1
 {
  f32 *d_C_ref = 0;
  CUDACheck(cudaMalloc(&d_C_ref, m*n*sizeof(*d_C_ref)));
  CUDACheck(cudaMemset(d_C_ref, 0, m*n*sizeof(*d_C_ref)));
  // Assert(data->beta == 0.f);

  // NOTE(achal): cuBLAS uses column-major for all matrices whereas they are allocated in row-major via CUDA,
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
  cublasStatus_t status = cublasGemmEx(
   handle,
   CUBLAS_OP_N, CUBLAS_OP_N,
   n, m, k,
   &alpha,
   data->d_B, CUDA_R_16F, n,
   data->d_A, CUDA_R_16F, k,
   &beta,
   d_C_ref, CUDA_R_32F, n,
   CUDA_R_32F,
   CUBLAS_GEMM_DEFAULT);
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
#if 1
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
#endif
 ScratchEnd(&scratch);
 return result;
}

template <typename T, u32 block_dim, u32 coarse_factor>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
 // GEMM(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
 // GEMM2(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
 // GEMM3(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
 GEMM4<2, 2>(data->m, data->n, data->k, T(1), data->d_A, data->d_B, T(0), data->d_C, stream);
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
    printf("%2.2f ", h_matrix[index]);
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
 if(argc == 2)
 {
  file_name = argv[1];
 }

 using InputType = half;
 if(1)
 {
  u8 *d_debug_buffer = 0;
  {
   CUDACheck(cudaMalloc(&d_debug_buffer, MegaBytes(1)));
   CUDACheck(cudaMemcpyToSymbol(g_debug_buffer, &d_debug_buffer, sizeof(g_debug_buffer), 0, cudaMemcpyHostToDevice));
  }

  Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

  DataDescriptor<InputType> *desc = PushStructZero(scratch.arena, DataDescriptor<InputType>);
  u32 m = desc->m = 4096; // 64;
  u32 n = desc->n = 4096; // 64;
  u32 k = desc->k = 4096; // 32;

  Data<InputType> *data = CreateData<InputType, 0>(scratch.arena, desc, 0);
  // GEMM(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);
  // GEMM2(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);
  // GEMM3(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);
  // for(int _ = 0; _ < 50 + 5; ++_)
  {
   GEMM4<2, 2>(m, n, k, InputType(1), data->d_A, data->d_B, InputType(0), data->d_C, 0);
   if (!ValidateGPUOutput<InputType>(scratch.arena, data))
   {
    printf("Failed\n");
   }
   else
   {
    printf("Passed\n");
   }
  }

  CUDACheck(cudaDeviceSynchronize());
  u8 *h_debug_buffer = PushBytes(scratch.arena, MegaBytes(1));
  CUDACheck(cudaMemcpy(h_debug_buffer, d_debug_buffer, MegaBytes(1), cudaMemcpyDeviceToHost));

  FILE *debug_file = fopen("debug.bin", "wb");
  if(debug_file)
  {
   size_t size = fwrite(h_debug_buffer, 1, MegaBytes(1), debug_file);
   if(size != MegaBytes(1)) printf("Debug buffer write incomplete (%llu/%llu)\n", size, MegaBytes(1));
  }
  else
  {
   printf("Failed to write debug buffer\n");
  }
  fclose(debug_file);

  DestroyData<InputType>(data);
  ScratchEnd(&scratch);
 }

 if(0)
 {
  f64 peak_gbps = 0.0;
  f64 peak_gflops = 0.0;
  GetPeakMeasurements(&peak_gbps, &peak_gflops, 1);

  printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
  printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
  printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

  FILE *file = 0;
  if(file_name)
   file = fopen(file_name, "wb");

  {
   u32 dims[] = {128, 256, 512, 1024, 2048, 4096};
   u32 skip_from_last = 0;
   
   printf("%-20s %-20s %-20s %-20s\n", "Input (m x n x k)", "GBPS", "GFLOPS", "Runtime (ms)");
   for(u32 i = 0; i < ArrayCount(dims) - skip_from_last; ++i)
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

    if(file)
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

  if(file)
   fclose(file);
 }

 EndProfiler();
 PrintPerformanceProfile();
    
 return 0;
}
PROFILER_END_OF_COMPILATION_UNIT;