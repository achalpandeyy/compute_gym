#include "common.cu"

#include "kernel1.cu"
#include "kernel2.cu"

static void LaunchKernel(int index, int count, int *input, int *output)
{
 switch(index)
 {
  case 1:
  {
   Kernel1Host(count, input, output);
  } break;

  case 2:
  {
   Kernel2Host(count, input, output);
  } break;

  default:
   assert(!"Invalid kernel index");
 }
}

int main()
{
 int count = 1 << 28;

 int *h_input = new int[count];
 for(int i = 0; i < count; ++i)
  h_input[i] = 1;

 int *d_input = nullptr;
 CUDACheck(cudaMalloc(&d_input, count * sizeof(int)));
 CUDACheck(cudaMemcpy(d_input, h_input, count * sizeof(int), cudaMemcpyHostToDevice));
 
 int *d_output = nullptr;
 CUDACheck(cudaMalloc(&d_output, sizeof(int)));
 CUDACheck(cudaMemset(d_output, 0, sizeof(int)));

 CUDACheck(cudaMemset(d_output, 0, sizeof(int)));
 LaunchKernel(1, count, d_input, d_output);
 CUDACheck(cudaDeviceSynchronize());

 int h_output = 0;
 CUDACheck(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

 (h_output == count) ? printf("Passed\n") : printf("Failed\n");
 return 0;
}