#define M_TILE 2
#define N_TILE 2

__launch_bounds__(16*16)
__global__ void Kernel2(int M, int N, int K, half alpha, half *A, half *B, float beta, float *C)
{
 __shared__ half As[32*M_TILE][16];
 __shared__ half Bs[16][32*N_TILE];

 int tm = threadIdx.y;
 int tn = threadIdx.x;
 int m_block = M_TILE*32*blockIdx.y;
 int n_block = N_TILE*32*blockIdx.x;
 
 int tid = threadIdx.y*blockDim.x + threadIdx.x;
 int warp_id = tid / 32;
 int lane_id = tid % 32;
 int m_warp = 16*(warp_id / 4);
 int n_warp =  8*(warp_id % 4);

 int group_id = lane_id / 4;
 int group_lane_id = lane_id % 4;

 half tile_a[8];
 half tile_b[4];
 float tile_c[M_TILE][N_TILE][4];

 for(int m = 0; m < M_TILE; ++m)
 {
  int m_tile = m*32;
  for(int n = 0; n < N_TILE; ++n)
  {
   int n_tile = n*32;
   tile_c[m][n][0] = beta*C[(m_block + m_tile + m_warp + group_id    )*N + (n_block + n_tile + n_warp + 2*group_lane_id    )];
   tile_c[m][n][1] = beta*C[(m_block + m_tile + m_warp + group_id    )*N + (n_block + n_tile + n_warp + 2*group_lane_id + 1)];
   tile_c[m][n][2] = beta*C[(m_block + m_tile + m_warp + group_id + 8)*N + (n_block + n_tile + n_warp + 2*group_lane_id    )];
   tile_c[m][n][3] = beta*C[(m_block + m_tile + m_warp + group_id + 8)*N + (n_block + n_tile + n_warp + 2*group_lane_id + 1)];
  }
 }

 for(int k_start = 0; k_start < K; k_start += 16)
 {
  for(int m = 0; m < M_TILE; ++m)
  {
   int m_tile = m*32;
   As[m_tile + tm     ][tn] = A[(m_block + m_tile + tm     )*K + (k_start + tn)];
   As[m_tile + tm + 16][tn] = A[(m_block + m_tile + tm + 16)*K + (k_start + tn)];
  }

  for(int n = 0; n < N_TILE; ++n)
  {
   int n_tile = n*32;
   Bs[tm][n_tile + tn     ] = alpha*B[(k_start + tm)*N + (n_block + n_tile + tn     )];
   Bs[tm][n_tile + tn + 16] = alpha*B[(k_start + tm)*N + (n_block + n_tile + tn + 16)];
  }
  __syncthreads();

  for(int m = 0; m < M_TILE; ++m)
  {
   int m_tile = m*32;
   tile_a[0] = As[m_tile + m_warp + group_id    ][group_lane_id*2    ];
   tile_a[1] = As[m_tile + m_warp + group_id    ][group_lane_id*2 + 1];
   tile_a[2] = As[m_tile + m_warp + group_id + 8][group_lane_id*2    ];
   tile_a[3] = As[m_tile + m_warp + group_id + 8][group_lane_id*2 + 1];
   tile_a[4] = As[m_tile + m_warp + group_id    ][group_lane_id*2 + 8];
   tile_a[5] = As[m_tile + m_warp + group_id    ][group_lane_id*2 + 9];
   tile_a[6] = As[m_tile + m_warp + group_id + 8][group_lane_id*2 + 8];
   tile_a[7] = As[m_tile + m_warp + group_id + 8][group_lane_id*2 + 9];
   for(int n = 0; n < N_TILE; ++n)
   {
    int n_tile = n*32;
    tile_b[0] = Bs[group_lane_id*2 + 0][n_tile + n_warp + group_id];
    tile_b[1] = Bs[group_lane_id*2 + 1][n_tile + n_warp + group_id];
    tile_b[2] = Bs[group_lane_id*2 + 8][n_tile + n_warp + group_id];
    tile_b[3] = Bs[group_lane_id*2 + 9][n_tile + n_warp + group_id];

    uint32_t *regs_a = (uint32_t *)tile_a;
    uint32_t *regs_b = (uint32_t *)tile_b;
    asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
     "{%0, %1, %2, %3}, " // d
     "{%4, %5, %6, %7}, " // a
     "{%8, %9}, " // b
     "{%0, %1, %2, %3};" // c
     : "+f"(tile_c[m][n][0]), "+f"(tile_c[m][n][1]), "+f"(tile_c[m][n][2]), "+f"(tile_c[m][n][3])
     : "r"(regs_a[0]),  "r"(regs_a[1]),  "r"(regs_a[2]),  "r"(regs_a[3]), 
       "r"(regs_b[0]),  "r"(regs_b[1])
    );
   }
  }
  __syncthreads();
 }

 for(int m = 0; m < M_TILE; ++m)
 {
  int m_tile = m*32;
  for(int n = 0; n < N_TILE; ++n)
  {
   int n_tile = n*32;
   C[(m_block + m_tile + m_warp + group_id    )*N + (n_block + n_tile + n_warp + 2*group_lane_id    )] = tile_c[m][n][0];
   C[(m_block + m_tile + m_warp + group_id    )*N + (n_block + n_tile + n_warp + 2*group_lane_id + 1)] = tile_c[m][n][1];
   C[(m_block + m_tile + m_warp + group_id + 8)*N + (n_block + n_tile + n_warp + 2*group_lane_id    )] = tile_c[m][n][2];
   C[(m_block + m_tile + m_warp + group_id + 8)*N + (n_block + n_tile + n_warp + 2*group_lane_id + 1)] = tile_c[m][n][3];
  }
 }
}

static void Kernel2Host(int M, int N, int K, float alpha, half *A, half *B, float beta, float *C)
{
 dim3 block(16, 16, 1);
 dim3 grid((N + 64 - 1)/64, (M + 64 - 1)/64, 1);
 half alpha16 = __float2half(alpha);
 Kernel2<<<grid, block>>>(M, N, K, alpha16, A, B, beta, C);
}