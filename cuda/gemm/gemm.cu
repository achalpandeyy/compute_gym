#include <random>

#include <cuda_fp16.h>
#include <cublas_v2.h>
static_assert(std::is_trivially_constructible<half>::value);

#include "common.cu"

#include "ada/kernel1.cu"
#include "ada/kernel2.cu"
#include "ada/kernel3.cu"
// #include "ada/kernel_latest.cu"

#define BENCHMARKING 0

static void LaunchKernel(int index /*0 for latest*/, int M, int N, int K, float alpha, half *A, half *B, half *B_T, float beta, float *C)
{
 switch(index)
 {
  case 0:
  {
   // KernelLatestHost<2, 2>(M, N, K, alpha, A, B, beta, C);
  } break;

  case 1:
  {
   Kernel1Host(M, N, K, alpha, A, B, beta, C);
  } break;

  case 2:
  {
   Kernel2Host(M, N, K, alpha, A, B, beta, C);
  } break;

  case 3:
  {
   Kernel3Host(M, N, K, alpha, A, B_T, beta, C);
  } break;

  default: assert(!"Invalid kernel index");
 }
}

int main()
{
 RegisterTracing("Kernel3");

 int M = 64;
 int K = 32;
 int N = 64;
 float alpha = 0.5f;
 float beta = 1.f;
 
 std::normal_distribution<float> distribution(0.f, 1.f);
 std::default_random_engine generator(12345);
 half *h_A = new half[M*K];
 for(int r = 0; r < M; ++r) for(int c = 0; c < K; ++c) h_A[r*K + c] = __float2half(distribution(generator));
 half *h_B = new half[K*N];
 for(int r = 0; r < K; ++r) for(int c = 0; c < N; ++c) h_B[r*N + c] = __float2half(distribution(generator));
 float *h_C = new float[M*N];
 for(int r = 0; r < M; ++r) for(int c = 0; c < N; ++c) h_C[r*N + c] = distribution(generator);
 
 half *d_A = 0, *d_B = 0, *d_B_T = 0;
 CUDACheck(cudaMalloc(&d_A, M*K*sizeof(half)));
 CUDACheck(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
 CUDACheck(cudaMalloc(&d_B, K*N*sizeof(half)));
 CUDACheck(cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice));

 half *h_B_T = new half[K*N];
 for(int r = 0; r < N; ++r) for(int c = 0; c < K; ++c) h_B_T[r*K + c] = h_B[c*N + r];
 CUDACheck(cudaMalloc(&d_B_T, K*N*sizeof(half)));
 CUDACheck(cudaMemcpy(d_B_T, h_B_T, K*N*sizeof(half), cudaMemcpyHostToDevice));
 
 float *d_C = 0;
 CUDACheck(cudaMalloc(&d_C, M*N*sizeof(*d_C)));
 CUDACheck(cudaMemcpy(d_C, h_C, M*N*sizeof(*d_C), cudaMemcpyHostToDevice));

#if BENCHMARKING
 for(int _ = 0; _ < 50 + 5; _ += 1)
#endif
 {
  LaunchKernel(3, M, N, K, alpha, d_A, d_B, d_B_T, beta, d_C);
 }
 CUDACheck(cudaDeviceSynchronize());

 // compare against cuBLAS
#if !BENCHMARKING
 if(1)
 {
  bool passed = true;
  
  float *d_C_ref = 0;
  CUDACheck(cudaMalloc(&d_C_ref, M*N*sizeof(*d_C_ref)));
  CUDACheck(cudaMemcpy(d_C_ref, h_C, M*N*sizeof(*d_C_ref), cudaMemcpyHostToDevice));

  // NOTE(achal): cuBLAS uses column-major for all matrices whereas they are allocated in row-major via CUDA,
  // so we are computing C^T = alpha*(B^T@A^T) + beta*C^T
  // A^T -> KxM
  // B^T -> NxK
  // C^T -> NxM
  // C = alpha*A@B + beta*C
  // C^T = alpha*(B^T@A^T) + beta*C^T
  cublasHandle_t handle;
  cublasCreate(&handle);
  
  cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K, &beta, d_C_ref, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  CUDACheck(cudaDeviceSynchronize());
  if(status == CUBLAS_STATUS_SUCCESS)
  {
   CUDACheck(cudaMemcpy(h_C, d_C, M*N*sizeof(*h_C), cudaMemcpyDeviceToHost));

   float *h_C_ref = new float[M*N];
   CUDACheck(cudaMemcpy(h_C_ref, d_C_ref, M*N*sizeof(*h_C_ref), cudaMemcpyDeviceToHost));
   double tolerance = 1e-2 * sqrt(K/64.0); // NOTE(achal): This is a heuristic to scale the tolerance based on the matrix size.
   for(int i = 0; i < M*N; ++i)
   {
    double abs_error = fabsf(h_C[i] - h_C_ref[i]);
    double rel_error = abs_error / (fabsf(h_C_ref[i]) + 1e-8);
    if (abs_error > tolerance && rel_error > tolerance)
    {
     printf("h_C_ref[%d] = %f, h_C[%d] = %f, abs_error = %f, rel_error = %f\n", i, h_C_ref[i], i, h_C[i], abs_error, rel_error);
     passed = false;
     break;
    }
   }
  }
  else
  {
   printf("cublasGemmEx failed with status %d\n", status);
   passed = false;
  }
  CUDACheck(cudaFree(d_C_ref));
  cublasDestroy(handle);

  passed ? printf("Passed\n") : printf("Failed\n");
 }
#endif

 return 0;
}