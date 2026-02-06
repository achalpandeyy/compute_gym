__forceinline__ __device__ void LoadMatrix_x4(unsigned int *reg, uint4 *addr)
{
 unsigned int ptx_src_addr = __cvta_generic_to_shared(addr);
 asm volatile(
  "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
  : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
  :  "r"(ptx_src_addr)
  );
}

__forceinline__ __device__ void LoadMatrix_x2(unsigned int *reg, uint4 *addr)
{
 unsigned int ptx_src_addr = __cvta_generic_to_shared(addr);
 asm volatile(
  "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
  : "=r"(reg[0]), "=r"(reg[1])
  :  "r"(ptx_src_addr)
  );
}

__forceinline__ __device__ void MMA_m16n8k16(unsigned int *A, unsigned int *B, float *C)
{
 asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
  "{%0, %1, %2, %3}, " // D
  "{%4, %5, %6, %7}, " // A
  "{%8, %9}, "         // B
  "{%0, %1, %2, %3};"  // C
  : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
  : "r"(A[0]),  "r"(A[1]),  "r"(A[2]),  "r"(A[3]), 
    "r"(B[0]),  "r"(B[1])
 );
}

__launch_bounds__(16*16)
__global__ void Kernel3(int M, int N, int K, half alpha, half *A, half *B, float beta, float *C)
{
 __shared__ uint4 As[32][8];
 __shared__ uint4 Bs[32][8];

 int m_block = 64*blockIdx.y;
 int n_block = 64*blockIdx.x;
 
 int tid = threadIdx.y*blockDim.x + threadIdx.x;
 int warp_id = tid / 32;
 int lane_id = tid % 32;
 int m_warp = 16*(warp_id / 4);
 int n_warp =  8*(warp_id % 4);

 int group_id = lane_id / 4;
 int group_lane_id = lane_id % 4;

 uint4 *global_tile_A = (uint4 *)(A + m_block*K);
 uint4 *global_tile_B = (uint4 *)(B + n_block*K);

 int store_row = warp_id*4 + (lane_id/8);
 int store_col = (lane_id % 8) ^ (lane_id/8);

 int load_row_A = (lane_id % 16) / 2;
 int load_col_A = (lane_id / 16 + 4 * (lane_id % 2)) ^ (load_row_A % 4);
 int load_row_B = (lane_id % 8) / 2;
 int load_col_B = (lane_id / 8 + 4 * (lane_id % 2)) ^ (load_row_B % 4);

 unsigned int a_reg[4];
 unsigned int b_reg[2];
 
 float c_reg[2][2][4];
 for(int m = 0; m < 2; m += 1)
 {
  for(int n = 0; n < 2; n += 1)
  {
   int m_tile = m*16;
   int n_tile = n*8;
   float2 *c0_in = (float2 *)(C + (m_block + m_tile + 2*m_warp + group_id    )*N + (n_block + n_tile + 2*n_warp + 2*group_lane_id));
   float2 *c2_in = (float2 *)(C + (m_block + m_tile + 2*m_warp + group_id + 8)*N + (n_block + n_tile + 2*n_warp + 2*group_lane_id));
   float2 *c0 = (float2 *)&c_reg[m][n][0];
   float2 *c2 = (float2 *)&c_reg[m][n][2];
   *c0 = *c0_in;
   *c2 = *c2_in;
   c0->x *= beta; c0->y *= beta;
   c2->x *= beta; c2->y *= beta;
  }
 }

 for(int k_start = 0; k_start < K/8; k_start += 4)
 {
  uint4 a = global_tile_A[(warp_id*8 + lane_id/4)*(K/8) + k_start + (lane_id % 4)];
  __half2 *a_h2 = (__half2 *)&a;
  for(int i = 0; i < 4; i += 1)
   a_h2[i] = __hmul2(a_h2[i], __half2(alpha, alpha));

  As[store_row][store_col] = a;
  Bs[store_row][store_col] = global_tile_B[(warp_id*8 + lane_id/4)*(K/8) + k_start + (lane_id % 4)];
  __syncthreads();

  for(int m = 0; m < 2; ++m)
  {
   int m_tile = m*8;
   for(int n = 0; n < 2; ++n)
   {
    int n_tile = n*4;
    LoadMatrix_x4(a_reg, As[m_warp + m_tile + load_row_A] + load_col_A);
    LoadMatrix_x2(b_reg, Bs[n_warp + n_tile + load_row_B] + load_col_B);
    MMA_m16n8k16(a_reg, b_reg, c_reg[m][n]);

    LoadMatrix_x4(a_reg, As[m_warp + m_tile + load_row_A] + (load_col_A ^ 2));
    LoadMatrix_x2(b_reg, Bs[n_warp + n_tile + load_row_B] + (load_col_B ^ 2));
    MMA_m16n8k16(a_reg, b_reg, c_reg[m][n]);
   }
  }
  __syncthreads();
 }

 for(int m = 0; m < 2; m += 1)
 {
  for(int n = 0; n < 2; n += 1)
  {
   int m_tile = m*16;
   int n_tile = n*8;
   float2 *c0_out = (float2 *)(C + (m_block + m_tile + 2*m_warp + group_id    )*N + (n_block + n_tile + 2*n_warp + 2*group_lane_id));
   float2 *c2_out = (float2 *)(C + (m_block + m_tile + 2*m_warp + group_id + 8)*N + (n_block + n_tile + 2*n_warp + 2*group_lane_id));
   float2 c0 = make_float2(c_reg[m][n][0], c_reg[m][n][1]);
   float2 c2 = make_float2(c_reg[m][n][2], c_reg[m][n][3]);
   *c0_out = c0;
   *c2_out = c2;
  }
 }
}

static void Kernel3Host(int M, int N, int K, float alpha, half *A, half *B, float beta, float *C)
{
 dim3 block(16, 16, 1);
 dim3 grid((N + 64 - 1)/64, (M + 64 - 1)/64, 1);
 half alpha16 = __float2half(alpha);
 Kernel3<<<grid, block>>>(M, N, K, alpha16, A, B, beta, C);
}