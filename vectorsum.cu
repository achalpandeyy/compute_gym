__global__ void vectorsum_kernel(uint64_t count, float *d_in, uint64_t stride)
{
    int lid = threadIdx.x;
    uint64_t gid = lid + blockIdx.x*blockDim.x;
    uint64_t index = gid*stride;

    // TODO(achal): Can we make this pass count calculation simpler given that we
    // will always be dealing with an integer number of threads per block?
    uint32_t pass_count = (uint32_t)ceilf(log2f((float)blockDim.x));

    // Each pass folds in the array into half
    for (int32_t pass = pass_count-1; pass >= 0; pass--)
    {
        uint32_t offset = 1 << pass;
        if ((lid < offset) && (index < count))
        {
            float a = d_in[index];
            float b = 0.f;
            if (index + offset*stride < count)
            b = d_in[index + offset*stride];
            d_in[index] = a + b;
        }
        __syncthreads();
    }
}

#if COMPILING_FROM_PYTORCH
void vectorsum(torch::Tensor a)
{
    int array_count = a.numel();
    
    int N = array_count;
    int block_dim = 1024;
    int block_count = (N + block_dim - 1)/block_dim;
    uint64_t stride = 1;
    
    do
    {
        vectorsum_kernel<<<block_count, block_dim>>>(array_count, a.data_ptr<float>(), stride);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("ERROR: %s\n", cudaGetErrorString(err));
        }
        
        N = block_count;
        if (block_count == 1)
        block_count = 0;
        else
        block_count = (N + block_dim - 1)/block_dim;
        stride *= block_dim;
    } while (block_count > 0);
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

// Dummy kernel for retrieving PTX version.
__global__ void DummyKernel() {}

static void Reduce(int array_count, float *d_in)
{
    // - One thread per input element
    // - One block is processing BLOCK_DIM elements at a time
    // - Multiple passes of the algorithm

    int N = array_count;
    int block_dim = 1024;
    int block_count = (N + block_dim - 1)/block_dim;
    uint64_t stride = 1;

    do
    {
        CUDACheck((vectorsum_kernel<<<block_count, block_dim>>>(array_count, d_in, stride)));
        
        N = block_count;
        if (block_count == 1)
            block_count = 0;
        else
            block_count = (N + block_dim - 1)/block_dim;
        stride *= block_dim;
    } while (block_count > 0);
}

static void Test_Reduce()
{
    for (int test = 0; test < 1024; test++)
    {
        int array_count = rand();
        if (array_count == 0)
            ++array_count;
        
        uint64_t array_size = array_count*sizeof(float);

        float *d_in = 0;
        CUDACheck(cudaMalloc(&d_in, array_size));

        float *h_in = (float *)malloc(array_size);
        for (int i = 0; i < array_count; ++i)
        {
            h_in[i] = 1.f;
        }

        CUDACheck(cudaMemcpy(d_in, h_in, array_size, cudaMemcpyHostToDevice));

        Reduce(array_count, d_in);

        CUDACheck(cudaDeviceSynchronize());

        float out = 0.f;
        CUDACheck(cudaMemcpy(&out, d_in, sizeof(float), cudaMemcpyDeviceToHost));

        // Compare with the host result
        {
            float ref = float(array_count);
            if (out == ref)
            {
                // printf("array_count: %llu \tPassed [out: %f, ref: %f]\n", array_count, out, ref);
            }
        }

        free(h_in);
        CUDACheck(cudaFree(d_in));
    }  
}

/**
 * Compile with:
 * nvcc.exe --use-local-env --device-debug --debug --generate-code=arch=compute_75,code=[compute_75,sm_75] vectorsum.cu -o "vectorsum"
*/

int main()
{
    cudaFuncAttributes attr;
    CUDACheck(cudaFuncGetAttributes(&attr, DummyKernel));

    int major_ver = attr.ptxVersion/10;
    int minor_ver = attr.ptxVersion%10;
    // Perhaps we can use PTX version to inform the kernel launch parameters:
    // https://github.com/moderngpu/moderngpu/blob/d8cf03d33917cb8ee792a59be54c4026913730e4/src/moderngpu/launch_box.hxx#L52
    printf("PTX version: %d.%d\n", major_ver, minor_ver);

    int device;
    CUDACheck(cudaGetDevice(&device));

    cudaDeviceProp device_prop = { 0 };
    CUDACheck(cudaGetDeviceProperties(&device_prop, device));

    if (1)
    {
        printf("Device name: %s\n", device_prop.name);
        printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("SMs: %d\n", device_prop.multiProcessorCount);
        // NOTE(achal): Compute capability 7.5 has 64 CUDA cores per SM.
        printf("CUDA cores: %d\n", device_prop.multiProcessorCount*64);
        printf("Clock rate: %d KHz\n", device_prop.clockRate); // NOTE(achal): This is deprecated.
        printf("Total Global Memory: %.2f GB (%llu bytes)\n", (device_prop.totalGlobalMem/(1024.f*1024.f*1024.f)), device_prop.totalGlobalMem);
        printf("Shared Memory (per block): %.2f KB (%llu bytes)\n", (device_prop.sharedMemPerBlock/1024.f), device_prop.sharedMemPerBlock);
        printf("Total Constant Memory: %.2f KB (%llu bytes)\n", (device_prop.totalConstMem/1024.f), device_prop.totalConstMem);
        printf("Warp Size: %d threads\n", device_prop.warpSize);
        printf("Max threads per block: %d\n", device_prop.maxThreadsPerBlock);
        printf("Max Block dimension: %dx%dx%d\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("Max Grid dimension: %dx%dx%d\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]); 
        printf("32-bit registers (per block): %d\n", device_prop.regsPerBlock);
    }

    bool run_tests = 1;
    if (run_tests)
    {
        Test_Reduce();
    }

    return 0;
}
#endif