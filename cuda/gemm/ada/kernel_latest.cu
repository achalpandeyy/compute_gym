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
__global__ void KernelLatest(half *A, half *B, float *C)
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
   int row = tid / 2;
   int col = tid % 2;
   uint4 *src = (uint4 *)&A[(base_row + row)*K + (base_col + col*8)];
   // int row = tid % (M_tile*32);
   // int col = tid / (M_tile*32);
   int store_col = col ^ ((row/4) % 2);
   shared_A[row][store_col] = src[0];
  }

  if(tid < k*8) // 8 uint64 loads per row
  {
   int base_row = step*16;
   int base_col = bidn*(N_tile*32);
   int row = tid % 16;
   int col = tid / 16;
   uint4 *src = (uint4 *)&B[(base_row + row)*N + (base_col + col*8)];
   int store_col = col ^ (row % 8);
   shared_B[row][store_col] = src[0];
  }
  __syncthreads();
  
  for(int iwm = 0; iwm < M_tile; ++iwm)
  {
   half tile_a[8];
   int row = iwm*32 + wm*16 + (lane_id % 16);
   int col = lane_id / 16;
   int load_col = col ^ ((row/4) % 2);
   LoadMatrix_x4(&shared_A[row][load_col], (uint32_t *)tile_a);
   for(int iwn = 0; iwn < N_tile; ++iwn)
   {
    // Even though the last sixteen threads, in the warp, have the
    // same smem address as the first sixteen, the instruction works
    // in a way that all the threads end up with the correct values,
    // as expected by mma.
    half tile_b[4];
    int row = lane_id % 16;
    int col = iwn*4 + wn;
    load_col = col ^ (row % 8);
    LoadMatrix_x2(&shared_B[row][load_col], (uint32_t *)tile_b);
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
static void KernelLatestHost(int M, int N, int K, half alpha, half *A, half *B, half beta, float *C)
{
 enum {m = 16, n = 8, k = 16};
 assert((M%m) == 0);
 assert((N%n) == 0);
 assert((K%k) == 0);

 dim3 block_dim(16, 16, 1);
 dim3 warps(4, 2, 1);
 assert(block_dim.x*block_dim.y == warps.x*warps.y*32);

 dim3 tile_dim(warps.x*(n*N_tile), warps.y*(m*M_tile), 1);
 assert((M % tile_dim.y) == 0);
 assert((N % tile_dim.x) == 0);
 assert((K % k) == 0);
 dim3 grid_dim(N/tile_dim.x, M/tile_dim.y, 1);

 if(M==4096 && N==4096 && K==4096)
 {
  KernelLatest<4096, 4096, 4096, M_tile, N_tile><<<grid_dim, block_dim>>>(A, B, C);
 }
 else if(M==64 && N==64 && K==64)
 {
  KernelLatest<64, 64, 64, M_tile, N_tile><<<grid_dim, block_dim>>>(A, B, C);
 }
 else if(M==128 && N==128 && K==128)
 {
  KernelLatest<128, 128, 128, M_tile, N_tile><<<grid_dim, block_dim>>>(A, B, C);
 }
 else if(M==32 && N==512 && K==32)
 {
  KernelLatest<32, 512, 32, M_tile, N_tile><<<grid_dim, block_dim>>>(A, B, C);
 }
 else if(M==2048 && N==2048 && K==2048)
 {
  KernelLatest<2048, 2048, 2048, M_tile, N_tile><<<grid_dim, block_dim>>>(A, B, C);
 }
 else if(M==64 && N==64 && K==32)
 {
  KernelLatest<64, 64, 32, M_tile, N_tile><<<grid_dim, block_dim>>>(A, B, C);
 }
 else
 {
  assert(!"Invalid code path");
 }
}
