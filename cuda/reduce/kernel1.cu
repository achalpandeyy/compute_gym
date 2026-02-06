__global__ void Kernel1(int count, int *input, int *output)
{
 int tid = threadIdx.x;
 int *segment = input + blockIdx.x*blockDim.x;;

 for(int stride = blockDim.x/2; stride >= 1; stride /= 2)
 {
  if(tid < stride)
   segment[tid] += segment[tid + stride];
  __syncthreads();
 }
 
 if(tid == 0)
  atomicAdd(output, segment[0]);
}

static void Kernel1Host(int count, int *input, int *output)
{
 int block = 512;
 assert((count % block) == 0);
 int grid = count/block;
 Kernel1<<<grid, block>>>(count, input, output);
}