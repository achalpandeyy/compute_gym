#include "core_types.h"
#include "core_memory.h"
#include "common.cuh"

template <typename T, u32 block_dim, u32 coarse_factor>
__global__ void ReduceKernel(u64 count, T *input, T *output)
{
    __shared__ T segment[block_dim];

    u64 segment_start = blockIdx.x*coarse_factor*block_dim;
    u64 index = segment_start + threadIdx.x;

    T sum = T(0);
    for (u8 i = 0; i < coarse_factor; ++i)
    {
        if (index + i*block_dim < count)
            sum += input[index + i*block_dim];
    }

    segment[threadIdx.x] = sum;
    
    for (int stride = block_dim/2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            segment[threadIdx.x] += segment[threadIdx.x + stride];
    }

    if (threadIdx.x == 0)
        atomicAdd(output, segment[0]);
}

template <typename T, u32 block_dim, u32 coarse_factor>
void Reduce(u64 count, T *input, T *output, cudaStream_t stream)
{
    int elements_per_block = coarse_factor*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    CUDACheck((ReduceKernel<T, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(count, input, output)));
}

#if COMPILING_FROM_PYTORCH
#include <c10/cuda/CUDAStream.h>

void Reduce(torch::Tensor input, torch::Tensor output)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int array_count = input.numel();
    f32 *d_input = input.data_ptr<f32>();
    f32 *d_output = output.data_ptr<f32>();

    Reduce(array_count, d_input, d_output, stream);
}
#else

#include "benchmarking.cu"

using InputType = f32;

#if 0
#include <thrust/reduce.h>

template <typename T>
void ThrustReduce(u64 count, T *input, T *output, cudaStream_t stream)
{
    thrust::reduce_into(thrust::cuda::par.on(stream), input, input + count, output, T(0));
}
#endif

template <typename T>
struct Data
{
    u64 count;
    T *h_in;
    T *d_in;
    T *d_out;
};

template <typename T, u32 elements_per_block>
static Data<T> *CreateData(Arena *arena, u64 count, cudaStream_t stream)
{
    Data<T> *data = PushStruct(arena, Data<T>);
    data->count = count;

    data->h_in = PushArray(arena, T, data->count);
    for (u64 i = 0; i < data->count; ++i)
        data->h_in[i] = T(1);

    CUDACheck(cudaMalloc(&data->d_in, data->count*sizeof(*data->d_in)));
    CUDACheck(cudaMemcpyAsync(data->d_in, data->h_in, data->count*sizeof(*data->d_in), cudaMemcpyHostToDevice, stream));

    CUDACheck(cudaMalloc(&data->d_out, sizeof(*data->d_out)));
    CUDACheck(cudaMemsetAsync(data->d_out, T(0), sizeof(*data->d_out), stream));

    CUDACheck(cudaStreamSynchronize(stream));

    return data;
}

template <typename T>
static void DestroyData(Data<T> *data)
{
    CUDACheck(cudaFree(data->d_in));
    CUDACheck(cudaFree(data->d_out));
}

template <typename T>
static b32 ValidateGPUOutput(Arena *arena, Data<T> *data)
{
    T out = T(0);
    CUDACheck(cudaMemcpy(&out, data->d_out, sizeof(*data->d_out), cudaMemcpyDeviceToHost));

    T out_ref = (T)data->count;
    b32 result = (out_ref == out);

    return result;
}

template <typename T, u32 block_dim, u32 coarse_factor>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
    Reduce<T, block_dim, coarse_factor>(data->count, data->d_in, data->d_out, stream);
    // ThrustReduce(data->count, data->d_in, data->d_out, stream);
}

static void Test_Reduce();

int main(int argc, char **argv)
{
    char *file_name = 0;
    if (argc == 2)
    {
        file_name = argv[1];
    }

    if (0)
    {
        Test_Reduce();
        printf("All tests passed\n");
    }

    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);

    printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
    printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
    printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    if (file_name)
    {
        enum { block_dim = 512, coarse_factor = 4 };
        Benchmark<InputType, block_dim, coarse_factor>(peak_gbps, peak_gflops, file_name);
    }

    return 0;
}

static void Test_Reduce()
{
    enum { block_dim = 1024, coarse_factor = 32 };

    srand(0);

    int test_count = 10;
    while (test_count > 0)
    {
        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

        u64 count = GetRandomNumber(30);
        printf("Count: %llu", count);

        Data<int> *data = CreateData<int, block_dim*coarse_factor>(scratch.arena, count, 0);
        Reduce<int, block_dim, coarse_factor>(count, data->d_in, data->d_out, 0);

        if (!ValidateGPUOutput<int>(scratch.arena, data))
        {
            // assert(0);
            exit(1);
        }
        printf(",\tPassed\n");

        DestroyData<int>(data);
        ScratchEnd(&scratch);

        --test_count;
    }
}
#endif