#include <stdio.h>

__global__ void conv2d_kernel(int N, int C_in, int H_in, int W_in, int K_h, int K_w, int C_out, int H_out, int W_out, float *input_tensor, float *kernel, float *output_tensor)
{
    int tile_dim = blockDim.x;
    
    int sample = blockIdx.z;
    int out_channel = blockIdx.x;

    int grid_w = (W_out + tile_dim - 1)/tile_dim;
    int tile_h = blockIdx.y/grid_w;
    int tile_w = blockIdx.y%grid_w;

    int h = tile_h*tile_dim + threadIdx.y;
    int w = tile_w*tile_dim + threadIdx.x;

    if (h < H_out && w < W_out)
    {
        float result = 0.f;
        for (int c = 0; c < C_in; ++c)
        {
            for (int kh = 0; kh < K_h; ++kh)
            {
                for (int kw = 0; kw < K_w; ++kw)
                {
                    int ih = h + kh;
                    int iw = w + kw;
                    int in_index = sample*(C_in*H_in*W_in) + c*(H_in*W_in) + ih*W_in + iw;
                    float in = input_tensor[in_index];

                    int kernel_index = out_channel*(C_in*K_h*K_w) + c*(K_h*K_w) + kh*K_w + kw;
                    float k = kernel[kernel_index];
                    
                    result += in*k;
                }
            }
        }
        int out_index = sample*(C_out*H_out*W_out) + out_channel*(H_out*W_out) + h*W_out + w;
        output_tensor[out_index] = result;
    }
}

#if COMPILING_FROM_PYTORCH
void conv2d(torch::Tensor input_tensor, torch::Tensor kernel, torch::Tensor output_tensor)
{
    int N = input_tensor.size(0);
    int C_in = input_tensor.size(1);
    int H_in = input_tensor.size(2);
    int W_in = input_tensor.size(3);

    int K_h = kernel.size(2);
    int K_w = kernel.size(3);
    int C_out = kernel.size(0);

    int H_out = H_in - K_h + 1;
    int W_out = W_in - K_w + 1;

    int tile_dim = 16;
    dim3 block(tile_dim, tile_dim, 1);
    int tile_count = ((H_out + tile_dim - 1) / tile_dim) * ((W_out + tile_dim - 1) / tile_dim);
    dim3 grid(C_out, tile_count, N);
    conv2d_kernel<<<grid, block>>>(
        N, C_in, H_in, W_in,
        K_h, K_w,
        C_out, H_out, W_out,
        input_tensor.data_ptr<float>(), kernel.data_ptr<float>(), output_tensor.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
}
#else
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
    int N = 1;
    int C_in = 16;
    int H_in = 32;
    int W_in = 32;
    float *input_tensor = (float *)malloc(N*C_in*H_in*W_in*sizeof(float));
    {
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C_in; ++c)
            {
                for (int y = 0; y < H_in; ++y)
                {
                    for (int x = 0; x < W_in; ++x)
                    {
                        uint64_t index = n*(C_in*H_in*W_in) + c*(H_in*W_in) + y*W_in + x;
                        input_tensor[index] = 1.f;
                    }
                }
            }
        }
    }
    int C_out = 16;
    int H_k = 4;
    int W_k = 4;
    float *kernel = (float *)malloc(C_out*C_in*H_k*W_k*sizeof(float));
    {
        for (int outc = 0; outc < C_out; ++outc)
        {
            for (int inc = 0; inc < C_in; ++inc)
            {
                for (int y = 0; y < H_k; ++y)
                {
                    for (int x = 0; x < W_k; ++x)
                    {
                        uint64_t index = outc*(C_in*H_k*W_k) + inc*(H_k*W_k) + y*W_k + x;
                        kernel[index] = 1.f;
                    }
                }
            }
        }
    }

    int H_out = H_in - H_k + 1;
    int W_out = W_in - W_k + 1;
    float *output_tensor = (float *)malloc(N*C_out*H_out*W_out*sizeof(float));

    for (int n = 0; n < N; ++n)
    {
        for (int outc = 0; outc < C_out; ++outc)
        {
            for (int y = 0; y < H_out; ++y)
            {
                for (int x = 0; x < W_out; ++x)
                {
                    float result = 0.f;
                    for (int inc = 0; inc < C_in; ++inc)
                    {
                        for (int ky = 0; ky < H_k; ++ky)
                        {
                            for (int kx = 0; kx < W_k; ++kx)
                            {
                                uint64_t kindex = outc*(C_in*H_k*W_k) + inc*(H_k*W_k) + ky*W_k + kx;
                                float k = kernel[kindex];

                                int ix = x + kx;
                                int iy = y + ky;
                                uint64_t iindex = n*(C_in*H_in*W_in) + inc*(H_in*W_in) + iy*W_in + ix;
                                float in = input_tensor[iindex];

                                result += in*k; 
                            }
                        }
                    }

                    uint64_t outindex = n*(C_out*H_out*W_out) + outc*(H_out*W_out) + y*W_out + x;
                    output_tensor[outindex] = result;
                }
            }
        }
    }

    printf("Output:\n");
    for (int n = 0; n < N; ++n)
    {
        for (int outc = 0; outc < C_out; ++outc)
        {
            for (int y = 0; y < H_out; ++y)
            {
                for (int x = 0; x < W_out; ++x)
                {
                    uint64_t outindex = n*(C_out*H_out*W_out) + outc*(H_out*W_out) + y*W_out + x;
                    printf("%.2f, ", output_tensor[outindex]);
                }
                printf("\n");
            }
            printf("\n\n");
        }
        printf("~~minbatch end~~\n");
    }

    memset(output_tensor, 0, N*C_out*H_out*W_out*sizeof(float));

    float *d_input_tensor = 0;
    CUDACheck(cudaMalloc(&d_input_tensor, N*C_in*H_in*W_in*sizeof(float)));
    CUDACheck(cudaMemcpy(d_input_tensor, input_tensor, N*C_in*H_in*W_in*sizeof(float), cudaMemcpyHostToDevice));

    float *d_kernel = 0;
    CUDACheck(cudaMalloc(&d_kernel, C_out*C_in*H_k*W_k*sizeof(float)));
    CUDACheck(cudaMemcpy(d_kernel, kernel, C_out*C_in*H_k*W_k*sizeof(float), cudaMemcpyHostToDevice));

    float *d_output_tensor = 0;
    CUDACheck(cudaMalloc(&d_output_tensor, N*C_out*H_out*W_out*sizeof(float)));

    int tile_dim = 16;
    dim3 block(tile_dim, tile_dim, 1);
    int tile_count = ((H_out + tile_dim - 1) / tile_dim) * ((W_out + tile_dim - 1) / tile_dim);
    dim3 grid(C_out, tile_count, N);
    conv2d_kernel<<<grid, block>>>(
        N, C_in, H_in, W_in,
        H_k, W_k,
        C_out, H_out, W_out,
        d_input_tensor, d_kernel, d_output_tensor);
    
    CUDACheck(cudaMemcpy(output_tensor, d_output_tensor, N*C_out*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost));

    CUDACheck(cudaFree(d_input_tensor));
    CUDACheck(cudaFree(d_kernel));
    CUDACheck(cudaFree(d_output_tensor));

    printf("Output (GPU):\n");
    for (int n = 0; n < N; ++n)
    {
        for (int outc = 0; outc < C_out; ++outc)
        {
            for (int y = 0; y < H_out; ++y)
            {
                for (int x = 0; x < W_out; ++x)
                {
                    uint64_t outindex = n*(C_out*H_out*W_out) + outc*(H_out*W_out) + y*W_out + x;
                    printf("%.2f, ", output_tensor[outindex]);
                }
                printf("\n");
            }
            printf("\n\n");
        }
        printf("~~minbatch end~~\n");
    }

    return 0;
}
#endif