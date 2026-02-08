__global__ void Kernel3(int count, int *input, int *output)
{
 int tid = threadIdx.x;
 __shared__ int segment[512];
 
 int start = blockIdx.x*blockDim.x*8 + tid;
 
 int thread_sum = 0;
 for(int i = 0; i < 8; ++i)
  thread_sum += input[start + i];

 segment[tid] = thread_sum;
 for(int stride = blockDim.x/2; stride >= 1; stride /= 2)
 {
  __syncthreads();
  if(tid < stride)
   segment[tid] += segment[tid + stride];
 }

 if(tid == 0)
  atomicAdd(output, segment[0]);
}

static void Kernel3Host(int count, int *input, int *output)
{
 int block = 512;
 int elements_per_block = block*8;
 assert((count % elements_per_block) == 0);
 int grid = count/elements_per_block;
 Kernel3<<<grid, block>>>(count, input, output);
}