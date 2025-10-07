#include "core_types.h"
#include "core_memory.h"

#include "common.cuh"

template <typename T>
static void SequentialInclusiveScan(int count, T *array)
{
    for (int i = 1; i < count; ++i)
    {
        array[i] += array[i - 1];
    }
}

template <typename T>
static void SequentialExclusiveScan(int count, T *array)
{
    T accum = T(0);
    for (int i = 0; i < count; ++i)
    {
        T temp = accum;
        accum += array[i];
        array[i] = temp;
    }
}

#define SEGMENT_SIZE 1024

template <typename T>
__device__ T BlockScan1(T *block)
{   
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();

        T temp = T(0);
        if (threadIdx.x >= stride)
            temp = block[threadIdx.x - stride];
        __syncthreads();

        block[threadIdx.x] = block[threadIdx.x] + temp;
    }
    
    return block[threadIdx.x];
}

template <typename T>
__global__ void InclusiveScanUpsweep(u64 count, T *array, T *summary)
{
    __shared__ T segment[SEGMENT_SIZE];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < count)
        segment[threadIdx.x] = array[index];
    else
        segment[threadIdx.x] = T(0);
    __syncthreads();

    T scan_result = BlockScan1(segment);

    if (index < count)
        array[index] = scan_result;

    if (summary && (threadIdx.x == blockDim.x - 1))
        summary[blockIdx.x] = scan_result;
}

template <typename T>
__global__ void InclusiveScanDownsweep(u64 count, T *array, T *summary)
{
    if (blockIdx.x > 0)
    {
        int prev_block_index = blockIdx.x - 1;
        T prev_block_sum = summary[prev_block_index];
        
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < count)
            array[index] = array[index] + prev_block_sum;
    }
}

template <typename T>
__global__ void ExclusiveScanUpsweep(u64 count, T *array, T *summary)
{
    __shared__ T segment[SEGMENT_SIZE];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < count && threadIdx.x != 0)
        segment[threadIdx.x] = array[index - 1];
    else
        segment[threadIdx.x] = T(0);

    T scan_result = BlockScan1(segment);

    if (summary && (threadIdx.x == blockDim.x - 1))
        summary[blockIdx.x] = scan_result + array[index];

    if (index < count)
        array[index] = scan_result;
}

template <typename T>
__global__ void ExclusiveScanDownsweep(u64 count, T *array, T *summary)
{
    if (blockIdx.x > 0)
    {
        int block_index = blockIdx.x;
        T block_sum = summary[block_index];
        
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < count)
            array[index] = array[index] + block_sum;
    }
}

template <typename T>
struct Scan_PassInput
{
    T *d_array;
    T *d_summary;
    u64 element_count;
};

template <typename T>
struct Scan_Input
{
    T *d_scratch;
    u32 pass_input_count;
    Scan_PassInput<T> *pass_inputs;
};

template <typename T>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array, int block_dim)
{
    Scan_Input<T> *input = PushStruct(arena, Scan_Input<T>);

    // Allocate scratch
    {
        u64 size = 0ull;

        // Add the number of elements for the first pass..
        u64 out_count = (count + block_dim - 1)/block_dim;
        size += out_count*sizeof(*input->d_scratch);

        // Add the number of elements for the second pass.. and we are done because we can just ping-pong.
        out_count = (out_count + block_dim - 1)/block_dim;
        size += out_count*sizeof(*input->d_scratch);

        CUDACheck(cudaMalloc(&input->d_scratch, size));
    }

    input->pass_input_count = (u32)ceilf(logf(count)/logf(block_dim));
    input->pass_inputs = PushArrayZero(arena, Scan_PassInput<T>, input->pass_input_count);

    int out_count = (count + block_dim - 1)/block_dim; // output of the first pass
    T *d_pingpong[] = {input->d_scratch, input->d_scratch + out_count};
    u64 element_count = count;

    for (u32 i = 0; i < input->pass_input_count; ++i)
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + i;
        pass_input->d_array = d_array;
        pass_input->d_summary = d_pingpong[i % 2];
        pass_input->element_count = element_count;

        int grid_dim = (element_count + block_dim - 1)/block_dim;
        
        d_array = pass_input->d_summary;
        element_count = grid_dim;
    }

    input->pass_inputs[input->pass_input_count - 1].d_summary = 0; // the "top" pass doesn't require a summary

    return input;
}

template <typename T>
static void Scan_DestroyInput(Scan_Input<T> *input)
{
    CUDACheck(cudaFree(input->d_scratch));
}

template <typename T>
static void InclusiveScan1(Scan_Input<T> *input, cudaStream_t stream)
{
    int block_dim = SEGMENT_SIZE;
    u32 k = input->pass_input_count;

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;
        int grid_dim = (pass_input->element_count + block_dim - 1)/block_dim;
        CUDACheck((InclusiveScanUpsweep<T><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;
        int grid_dim = (pass_input->element_count + block_dim - 1)/block_dim;
        CUDACheck((InclusiveScanDownsweep<T><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }
}

template <typename T>
static void ExclusiveScan1(Scan_Input<T> *input, cudaStream_t stream)
{
    int block_dim = SEGMENT_SIZE;
    u32 k = input->pass_input_count;

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;
        int grid_dim = (pass_input->element_count + block_dim - 1)/block_dim;
        CUDACheck((ExclusiveScanUpsweep<T><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;
        int grid_dim = (pass_input->element_count + block_dim - 1)/block_dim;
        CUDACheck((ExclusiveScanDownsweep<T><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }
}

#include "benchmarking.cu"

#include <thrust/scan.h>

using InputType = int;

template <typename T>
void ThrustInclusiveScan(u64 count, T *input, T *output, cudaStream_t stream)
{
    thrust::inclusive_scan(thrust::cuda::par.on(stream), input, input + count, output);
}

template <typename T>
void ThrustExclusiveScan(u64 count, T *input, T *output, cudaStream_t stream)
{
    thrust::exclusive_scan(thrust::cuda::par.on(stream), input, input + count, output);
}

template <typename T>
struct Data
{
    u64 count;
    T *h_array;
    T *d_array;
    Scan_Input<T> *input;
};

template <typename T>
static Data<T> *CreateData(Arena *arena, u64 count, cudaStream_t stream)
{
    Data<T> *data = PushStruct(arena, Data<T>);
    data->count = count;

    data->h_array = PushArray(arena, T, data->count);
    for (u64 i = 0; i < data->count; ++i)
        data->h_array[i] = T(1);

    CUDACheck(cudaMalloc(&data->d_array, data->count*sizeof(*data->d_array)));
    CUDACheck(cudaMemcpyAsync(data->d_array, data->h_array, data->count*sizeof(*data->d_array), cudaMemcpyHostToDevice, stream));

    CUDACheck(cudaStreamSynchronize(stream));

    data->input = Scan_CreateInput(arena, data->count, data->d_array, SEGMENT_SIZE);

    return data;
}

template <typename T>
static void DestroyData(Data<T> *data)
{
    Scan_DestroyInput(data->input);
    CUDACheck(cudaFree(data->d_array));
}

template <typename T>
static b32 ValidateGPUOutput(Arena *arena, Data<T> *data)
{
    Scratch scratch = ScratchBegin(arena);

    T *gpu_out = PushArray(scratch.arena, T, data->count);
    CUDACheck(cudaMemcpy(gpu_out, data->d_array, data->count*sizeof(*gpu_out), cudaMemcpyDeviceToHost));

    SequentialInclusiveScan<T>(data->count, data->h_array);
    b32 result = (memcmp(gpu_out, data->h_array, data->count*sizeof(*gpu_out)) == 0);

    ScratchEnd(&scratch);

    return result;
}

template <typename T>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
    InclusiveScan1<T>(data->input, stream);
    // ThrustInclusiveScan<T>(data->count, data->d_array, data->d_array, stream);
}

static void TestScan();

int main(int argc, char **argv)
{
    char *file_name = 0;
    if (argc == 2)
    {
        file_name = argv[1];
    }

    if (1)
    {
        TestScan();
        printf("All tests passed\n");
    }

    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);
    
    printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
    printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
    printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    Benchmark<InputType>(peak_gbps, peak_gflops, file_name);

    return 0;
}

static void TestScan()
{
    srand(0);

    int test_count = 10;
    while (test_count > 0)
    {
        
        u64 count = GetRandomNumber(30);
        printf("Count: %llu:\t", count);
        
        // Inclusive scan
        {
            printf("Inclusive, ");
            Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));
            
            Data<int> *data = CreateData<int>(scratch.arena, count, 0);
            Scan_Input<int> *input = Scan_CreateInput(scratch.arena, count, data->d_array, SEGMENT_SIZE);
            InclusiveScan1<int>(input, 0);
            
            int *gpu_output = PushArray(scratch.arena, int, count);
            CUDACheck(cudaMemcpy(gpu_output, data->d_array, count*sizeof(int), cudaMemcpyDeviceToHost));

            SequentialInclusiveScan<int>(count, data->h_array);
            for (int i = 0; i < count; ++i)
            {
                if (data->h_array[i] != gpu_output[i])
                {
                    printf("[FAIL] Result[%d] (GPU): %d \tExpected: %d\n", i, gpu_output[i], data->h_array[i]);
                    exit(1);
                }
            }
            printf("Passed\t");

            DestroyData<int>(data);

            ScratchEnd(&scratch);
        }

        // Exclusive scan
        {
            printf("Exclusive, ");
            Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

            Data<int> *data = CreateData<int>(scratch.arena, count, 0);
            Scan_Input<int> *input = Scan_CreateInput(scratch.arena, count, data->d_array, SEGMENT_SIZE);
            ExclusiveScan1<int>(input, 0);
            
            int *gpu_output = PushArray(scratch.arena, int, count);
            CUDACheck(cudaMemcpy(gpu_output, data->d_array, count*sizeof(int), cudaMemcpyDeviceToHost));

            SequentialExclusiveScan<int>(count, data->h_array);
            for (int i = 0; i < count; ++i)
            {
                if (data->h_array[i] != gpu_output[i])
                {
                    printf("[FAIL] Result[%d] (GPU): %d \tExpected: %d\n", i, gpu_output[i], data->h_array[i]);
                    exit(1);
                }
            }
            printf("Passed\n");

            DestroyData<int>(data);
            ScratchEnd(&scratch);
        }

        --test_count;
    }
}