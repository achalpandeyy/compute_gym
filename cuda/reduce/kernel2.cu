__global__ void Kernel2(int count, int *input, int *output)
{
 int tid = threadIdx.x;
 __shared__ int segment[512];
 segment[tid] = input[blockIdx.x*blockDim.x + tid];

 for(int stride = blockDim.x/2; stride >= 1; stride /= 2)
 {
  __syncthreads();
  if(tid < stride)
   segment[tid] += segment[tid + stride];
 }

 if(tid == 0)
  atomicAdd(output, segment[0]);
}

static void Kernel2Host(int count, int *input, int *output)
{
 int block = 512;
 assert((count % block) == 0);
 int grid = count/block;
 Kernel2<<<grid, block>>>(count, input, output);
}