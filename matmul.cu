template <typename scalar_t>
__global__ void matmul_kernel(int M, int N, int K, scalar_t *A, scalar_t *B, scalar_t *C)
{
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float result = 0.f;
        for (int k = 0; k < K; ++k)
        {
            result += A[row*K + k]*B[k*N + col];
        }
        C[row*N + col] = result;
    }
}

#if COMPILING_FROM_PYTORCH
void matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    int tile_dim = 16;
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((M + tile_dim - 1)/tile_dim, (N + tile_dim - 1)/tile_dim, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "matmul_kernel", [&]{
        matmul_kernel<<<grid, block>>>(M, N, K, A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>());
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
    int M = 64;
    int N = 64;
    int K = 64;

    float *A = (float *)malloc(M*K*sizeof(float));
    float *B = (float *)malloc(K*N*sizeof(float));
    float *C = (float *)malloc(M*N*sizeof(float));
    
    for (int i = 0; i < M*K; ++i)
    {
        A[i] = 1.f;
    }
    for (int i = 0; i < K*N; ++i)
    {
        B[i] = 1.f;
    }

    memset(C, 0, M*N*sizeof(float));

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float result = 0.f;
            for (int k = 0; k < K; ++k)
            {
                result += A[row*K + k]*B[k*N + col];
            }
            C[row*N + col] = result;
        }
    }

    printf("C (CPU):\n");
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            printf("%f ", C[row*N + col]);
        }
        printf("\n");
    }

    float *d_A = 0;
    CUDACheck(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDACheck(cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice));

    float *d_B = 0;
    CUDACheck(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDACheck(cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    float *d_C = 0;
    CUDACheck(cudaMalloc(&d_C, M*N*sizeof(float)));

    int tile_dim = 16;
    dim3 block(tile_dim, tile_dim, 1);
    dim3 grid((M + tile_dim - 1)/tile_dim, (N + tile_dim - 1)/tile_dim, 1);
    
    matmul_kernel<float><<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    CUDACheck(cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("C (GPU):\n");
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            printf("%f ", C[row*N + col]);
        }
        printf("\n");
    }

    return 0;
}