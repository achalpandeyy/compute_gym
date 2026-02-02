#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <type_traits>

#include "common.cu"

#include <random>
#include <cublas_v2.h>

static void GEMMCPU(int M, int N, int K, half alpha, half *A, half *B, half beta, float *C)
{
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

template <typename T, uint32_t tile_dim, uint8_t elements_per_thread>
__global__ void GEMMKernel3(uint32_t m, uint32_t n, uint32_t k, T alpha, T *A, T *B, T beta, T *C)
{
 __shared__ T tile_A[tile_dim][tile_dim];
 __shared__ T tile_B[tile_dim][tile_dim];

 // one block computes one output tile
 uint32_t tile_x = blockIdx.x;
 uint32_t tile_y = blockIdx.y;

 uint32_t tx = threadIdx.x;
 uint32_t ty = threadIdx.y;

 T results[elements_per_thread] = {0};
 uint32_t step_count = (k + tile_dim - 1)/tile_dim;
 for(uint32_t step_index = 0; step_index < step_count; ++step_index)
 {
  uint32_t ax = step_index*tile_dim + tx;
  uint32_t bx = tile_x*tile_dim + tx;
  for(uint8_t i = 0; i < elements_per_thread; ++i)
  {
   uint32_t ay = tile_y*tile_dim + (ty*elements_per_thread + i);
   uint32_t by = step_index*tile_dim + (ty*elements_per_thread + i);
   
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

  for(uint32_t index = 0; index < tile_dim; ++index)
  {
   T b = tile_B[index][tx];
   for(uint8_t i = 0; i < elements_per_thread; ++i)
   {
    results[i] += tile_A[ty*elements_per_thread + i][index]*b;
   }
  }
  __syncthreads();
 }

 uint32_t col = tile_x*tile_dim + tx;
 for(uint8_t i = 0; i < elements_per_thread; ++i)
 {
  uint32_t row = tile_y*tile_dim + (ty*elements_per_thread + i);
  if(row < m && col < n)
  {
   C[row*n + col] = alpha*results[i] + beta*C[row*n + col];
  }
 }
}

template <typename T>
static void GEMM3(uint32_t m, uint32_t n, uint32_t k, T alpha, T *A, T *B, T beta, T *C, cudaStream_t stream)
{
 enum {elements_per_thread = 8, tile_dim = 64};
 assert((tile_dim % elements_per_thread) == 0);

 dim3 block_dim(tile_dim, tile_dim/elements_per_thread, 1);
 dim3 elements_per_block(block_dim.x, block_dim.y*elements_per_thread, block_dim.z);
 dim3 grid_dim((n + elements_per_block.x - 1)/elements_per_block.x, (m + elements_per_block.y - 1)/elements_per_block.y, 1);
 GEMMKernel3<T, tile_dim, elements_per_thread><<<grid_dim, block_dim, 0, stream>>>(m, n, k, alpha, A, B, beta, C);
}

int main(int argc, char **argv)
{
    
 return 0;
}