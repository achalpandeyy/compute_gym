#define TILE_DIM 16

template <typename scalar_t>
__global__ void matmul_kernel(int i, int j, int k, scalar_t *M, scalar_t *N, scalar_t *P)
{
    __shared__ float M_tile[TILE_DIM][TILE_DIM];
    __shared__ float N_tile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y*blockDim.y + ty;
    int col = blockIdx.x*blockDim.x + tx;

    float result(0);
    int step_count = (k + TILE_DIM - 1)/TILE_DIM;
    for (int step = 0; step < step_count; ++step)
    {
        int m_row = row;
        int m_col = step*TILE_DIM + tx;

        int n_row = step*TILE_DIM + ty;
        int n_col = col;

        float m(0);
        if (m_row < i && m_col < k)
            m = (float)M[m_row*k + m_col];

        float n(0);
        if (n_row < k && n_col < j)
            n = (float)N[n_row*j + n_col];
        
        M_tile[ty][tx] = m;
        N_tile[ty][tx] = n;
        __syncthreads();
        
        for (int index = 0; index < TILE_DIM; ++index)
        {
            result += M_tile[ty][index]*N_tile[index][tx];
        }
        __syncthreads();
    }

    if (row < i && col < j)
    {
        P[row*j + col] = (scalar_t)result;
    }
}

#if COMPILING_FROM_PYTORCH
void matmul(torch::Tensor M, torch::Tensor N, torch::Tensor P)
{
    int i = M.size(0);
    int j = N.size(1);
    int k = M.size(1);

    int tile_dim = TILE_DIM;
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((j + TILE_DIM - 1)/TILE_DIM, (i + TILE_DIM - 1)/TILE_DIM, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(M.scalar_type(), "matmul_kernel", [&]{
        matmul_kernel<<<grid, block>>>(i, j, k, M.data_ptr<scalar_t>(), N.data_ptr<scalar_t>(), P.data_ptr<scalar_t>());
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
}
#endif

#include <stdio.h>

inline static void GetCUDAErrorDetails(cudaError_t error, char const **error_name, char const **error_string)
{
    if (error_name)
        *error_name = cudaGetErrorName(error);
    
    if (error_string)
        *error_string = cudaGetErrorString(error);
}

#define CUDACheck_(fn_call, line)\
{\
    cudaError_t prev_error = cudaGetLastError();\
    while (prev_error != cudaSuccess)\
    {\
        char const *error_name = 0;\
        char const *error_string = 0;\
        GetCUDAErrorDetails(prev_error, &error_name, &error_string);\
        printf("[ERROR]: CUDA Runtime already had an error: %s %s", error_name, error_string);\
        prev_error = cudaGetLastError();\
    }\
    fn_call;\
    cudaError_t error = cudaGetLastError();\
    if (error != cudaSuccess)\
    {\
        char const *error_name = 0;\
        char const *error_string = 0;\
        GetCUDAErrorDetails(error, &error_name, &error_string);\
        printf("CUDA Error on line %u: %s %s", line, error_name, error_string);\
    }\
}
#define CUDACheck(fn_call) CUDACheck_(fn_call, __LINE__)

int main()
{
    int i = 64;
    int j = 64;
    int k = 64;

    float *M = (float *)malloc(i*k*sizeof(float));
    float *N = (float *)malloc(k*j*sizeof(float));
    float *P = (float *)malloc(i*j*sizeof(float));

    for (int i = 0; i < i*k; ++i)
    {
        M[i] = 1.f;
    }
    for (int i = 0; i < k*j; ++i)
    {
        N[i] = 1.f;
    }
    memset(P, 0, i*j*sizeof(float));

    for (int row = 0; row < i; ++row)
    {
        for (int col = 0; col < j; ++col)
        {
            float result = 0.f;
            for (int index = 0; index < k; ++index)
            {
                result += M[row*k + index]*N[index*j + col];
            }
            P[row*j + col] = result;
        }
    }

    float *d_M = 0;
    CUDACheck(cudaMalloc(&d_M, i*k*sizeof(float)));
    CUDACheck(cudaMemcpy(d_M, M, i*k*sizeof(float), cudaMemcpyHostToDevice));

    float *d_N = 0;
    CUDACheck(cudaMalloc(&d_N, k*j*sizeof(float)));
    CUDACheck(cudaMemcpy(d_N, N, k*j*sizeof(float), cudaMemcpyHostToDevice));

    float *d_P = 0;
    CUDACheck(cudaMalloc(&d_P, i*j*sizeof(float)));

    dim3 block(TILE_DIM, TILE_DIM, 1);
    dim3 grid((j + TILE_DIM - 1)/TILE_DIM, (i + TILE_DIM - 1)/TILE_DIM, 1);
    
    matmul_kernel<float><<<grid, block>>>(i, j, k, d_M, d_N, d_P);

    float *P_gpu = (float *)malloc(i*j*sizeof(float));
    CUDACheck(cudaMemcpy(P_gpu, d_P, i*j*sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int row = 0; row < i; ++row)
    {
        for (int col = 0; col < j; ++col)
        {
            if (P[row*j + col] != P_gpu[row*j + col])
            {
                printf("P[%d][%d] = %f, P_gpu[%d][%d] = %f\n", row, col, P[row*j + col], row, col, P_gpu[row*j + col]);
                passed = false;
            }
        }
    }

    if (passed)
    {
        printf("Passed\n");
    }
    else
    {
        printf("Failed\n");
    }

    return 0;
}