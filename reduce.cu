#include "common.cuh"

template <typename T>
__global__ void ReduceKernel1(T *input)
{
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if ((threadIdx.x % stride) == 0)
        {
            int index = 2*threadIdx.x;
            input[index] += input[index + stride];
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void ReduceKernel2(u64 count, T *input, T *output)
{
    int segment_start = blockIdx.x*(2*blockDim.x);
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if ((threadIdx.x % stride) == 0)
        {
            int index = segment_start + 2*threadIdx.x;
            if (index < count)
            {
                T temp = T(0);
                if (index + stride < count)
                    temp = input[index + stride];
                input[index] += temp;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(output, input[segment_start]);
}

template <typename T>
__global__ void ReduceKernel3(u64 count, T *input, T *output)
{
    int segment_start = blockIdx.x*(2*blockDim.x);
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            int index = segment_start + threadIdx.x;
            if (index < count)
            {
                T temp = T(0);
                if (index + stride < count)
                    temp = input[index + stride];
                input[index] += temp;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(output, input[segment_start]);
}

#define COARSE_FACTOR 32

template <typename T>
__global__ void ReduceKernel4(u64 count, T *input, T *output)
{
    int segment_start = blockIdx.x*COARSE_FACTOR*blockDim.x;
    int index = segment_start + threadIdx.x;

    T sum = T(0);
    for (int e = 0; e < COARSE_FACTOR; ++e)
    {
        if (index + e*blockDim.x < count)
            sum += input[index + e*blockDim.x];
    }

    if (index < count)
        input[index] = sum;
    
    for (int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            if (index < count)
            {
                T temp = T(0);
                if (index + stride < count)
                    temp = input[index + stride];
                input[index] += temp;
            }
        }
    }

    if (threadIdx.x == 0)
        atomicAdd(output, input[segment_start]);
}

template <typename T>
__global__ void ReduceKernel5(u64 count, T *input, T *output)
{
    __shared__ T smem;
    T *input_s = &smem;

    int segment_start = blockIdx.x*COARSE_FACTOR*blockDim.x;
    int index = segment_start + threadIdx.x;

    T sum = T(0);
    for (int e = 0; e < COARSE_FACTOR; ++e)
    {
        if (index + e*blockDim.x < count)
            sum += input[index + e*blockDim.x];
    }

    input_s[threadIdx.x] = sum;
    
    for (int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
    }

    if (threadIdx.x == 0)
        atomicAdd(output, input_s[0]);
}

template <typename T>
void Reduce1(u64 count, T *input)
{
    assert(count <= 2048);

    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    CUDACheck((ReduceKernel1<T><<<grid_dim, block_dim>>>(input)));
}

template <typename T>
void Reduce2(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    CUDACheck((ReduceKernel2<T><<<grid_dim, block_dim, 0, stream>>>(count, input, output)));
}

template <typename T>
void Reduce3(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    CUDACheck((ReduceKernel3<T><<<grid_dim, block_dim, 0, stream>>>(count, input, output)));
}

template <typename T>
void Reduce4(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = COARSE_FACTOR*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    CUDACheck((ReduceKernel4<T><<<grid_dim, block_dim, 0, stream>>>(count, input, output)));
}

template <typename T>
void Reduce5(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = COARSE_FACTOR*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    CUDACheck((ReduceKernel5<T><<<grid_dim, block_dim, block_dim*sizeof(T), stream>>>(count, input, output)));
}

template <typename T>
using ReduceFn = void (*)(u64, T*, T*, cudaStream_t);

#if COMPILING_FROM_PYTORCH
#include <c10/cuda/CUDAStream.h>

void Reduce(torch::Tensor input, torch::Tensor output)
{
#ifndef KERNEL_VERSION
#define KERNEL_VERSION 5
#endif
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int array_count = input.numel();
    f32 *d_input = input.data_ptr<f32>();
    f32 *d_output = output.data_ptr<f32>();

    ReduceFn Reduce = g_reduce_fns[KERNEL_VERSION];
    Reduce(array_count, d_input, d_output, stream);
}
#else
#include <thrust/reduce.h>

#include "benchmarking.cu"

using InputType = f32;

static ReduceFn<InputType> g_Reduce = 0;

template <typename T>
void ThrustReduce(u64 count, T *input, T *output, cudaStream_t stream)
{
    thrust::reduce_into(thrust::cuda::par.on(stream), input, input + count, output, T(0));
}

template <typename T>
struct Data
{
    u64 count;
    T *h_in;
    T *d_in;
    T *d_out;
};

template <typename T>
static void CreateData(Data<T> *data, cudaStream_t stream)
{
    data->h_in = (T *)malloc(data->count*sizeof(*data->h_in));
    for (u64 i = 0; i < data->count; ++i)
        data->h_in[i] = T(1);

    CUDACheck(cudaMalloc(&data->d_in, data->count*sizeof(*data->d_in)));
    CUDACheck(cudaMemcpyAsync(data->d_in, data->h_in, data->count*sizeof(*data->d_in), cudaMemcpyHostToDevice, stream));

    CUDACheck(cudaMalloc(&data->d_out, sizeof(*data->d_out)));
    CUDACheck(cudaMemsetAsync(data->d_out, T(0), sizeof(*data->d_out), stream));

    CUDACheck(cudaStreamSynchronize(stream));
}

template <typename T>
static void DestroyData(Data<T> *data)
{
    CUDACheck(cudaFree(data->d_in));
    CUDACheck(cudaFree(data->d_out));

    free(data->h_in);
}

template <typename T>
static b32 ValidateGPUOutput(Data<T> *data)
{
    T out = T(0);
    CUDACheck(cudaMemcpy(&out, data->d_out, sizeof(*data->d_out), cudaMemcpyDeviceToHost));

    T out_ref = (T)data->count;
    b32 result = (out_ref == out);

    return result;
}

template <typename T>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
    g_Reduce(data->count, data->d_in, data->d_out, stream);
}

static void TestReduce(int reduce_index);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <reduce_index>\n", argv[0]);
        return 1;
    }
    
    int reduce_index = atoi(argv[1]);
    printf("Reduce index: %d\n", reduce_index);

    if (1)
    {
        TestReduce(reduce_index);
        printf("All tests passed\n");
    }

    ReduceFn<InputType> reduce_fns[] =
    {
        0,
        0,
        Reduce2<InputType>,
        Reduce3<InputType>,
        Reduce4<InputType>,
        Reduce5<InputType>,
        ThrustReduce<InputType>,
    };
    g_Reduce = reduce_fns[reduce_index];

    const char *file_names[] =
    {
        0,
        0,
        "bench_reduce2.bin",
        "bench_reduce3.bin",
        "bench_reduce4.bin",
        "bench_reduce5.bin",
        "bench_reduce_thrust.bin",
    };

    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);

    printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
    printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
    printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    const char *file_name = file_names[reduce_index];
    Benchmark<InputType>(peak_gbps, peak_gflops, file_name);

    return 0;
}

static void TestReduce(int reduce_index)
{
    ReduceFn<int> reduce_fns[] =
    {
        0,
        0,
        Reduce2<int>,
        Reduce3<int>,
        Reduce4<int>,
        Reduce5<int>,
        ThrustReduce<int>,
    };
    ReduceFn<int> Reduce = reduce_fns[reduce_index];

    srand(0);
    int test_count = 10;
    while (test_count > 0)
    {
        u64 count = GetRandomNumber(30);
        printf("Count: %llu", count);

        Data<int> data;
        data.count = count;
        CreateData(&data, 0);

        Reduce(count, data.d_in, data.d_out, 0);

        if (!ValidateGPUOutput<int>(&data))
        {
            // assert(0);
            exit(1);
        }
        printf(",\tPassed\n");

        DestroyData(&data);

        --test_count;
    }
}
#endif