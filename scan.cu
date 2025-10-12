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

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__device__ T SegmentScan_KoggeStone(u64 count, T *array)
{
    __shared__ T segment[coarse_factor*block_dim];

    for (int i = 0; i < coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = blockIdx.x*coarse_factor*block_dim + smem_index;
        if constexpr (inclusive)
        {
            if (gmem_index < count)
                segment[smem_index] = array[gmem_index];
            else
                segment[smem_index] = T(0);
        }
        else
        {
            if ((gmem_index >= count) || ((threadIdx.x == 0) && (i == 0)))
                segment[smem_index] = T(0);
            else
                segment[smem_index] = array[gmem_index - 1];
        }
    }
    __syncthreads();

    // step I: thread local sequential scan
    {
        u32 start = threadIdx.x*coarse_factor;
        for (u32 j = 1; j < coarse_factor; ++j)
            segment[start + j] += segment[start + (j - 1)];
    }
    __syncthreads();

    // step II: strided scan
    for (int stride = 1; stride < block_dim; stride *= 2)
    {
        T temp = T(0);
        if (threadIdx.x >= stride)
            temp = segment[(threadIdx.x - stride)*coarse_factor + (coarse_factor - 1)];
        __syncthreads();

        segment[threadIdx.x*coarse_factor + (coarse_factor - 1)] += temp;
        __syncthreads();
    }

    // step III: fixup
    {
        u32 start = threadIdx.x*coarse_factor;
        if (start > 0)
        {
            T prev = segment[start - 1];
            for (u32 i = 0; i < coarse_factor - 1; ++i)
                segment[start + i] += prev;
        }
    }
    __syncthreads();

    T segment_result = segment[(block_dim - 1)*coarse_factor + (coarse_factor - 1)];
    if constexpr (!inclusive)
    {
        u64 index = blockIdx.x*coarse_factor*block_dim + (block_dim - 1)*coarse_factor + (coarse_factor - 1);
        if (index > count - 1)
            index = count - 1;
        segment_result += array[index];
    }

    for (int i = 0; i < coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = blockIdx.x*coarse_factor*block_dim + smem_index;
        if (gmem_index < count)
            array[gmem_index] = segment[smem_index];
    }

    return segment_result;
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__device__ T SegmentScan_WE(u64 count, T *array)
{
    __shared__ T segment[2*coarse_factor*block_dim];

    for (int i = 0; i < 2*coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = blockIdx.x*2*coarse_factor*block_dim + smem_index;
        if (gmem_index < count)
                segment[smem_index] = array[gmem_index];
            else
                segment[smem_index] = T(0);
    }
    __syncthreads();

    // step I: thread local sequential scan
    {
        u32 thread_start = (threadIdx.x*2)*coarse_factor;
        for (u32 i = 0; i < 2; ++i)
        {
            u32 start = thread_start + i*coarse_factor;
            for (u32 j = 1; j < coarse_factor; ++j)
                segment[start + j] += segment[start + (j-1)];
        }
    }
    __syncthreads();

    // step II.I: reduction tree
    for (int stride = 1*coarse_factor; stride <= block_dim*coarse_factor; stride *= 2)
    {
        int index = (threadIdx.x + 1)*2*stride - 1;
        if (index < 2*block_dim*coarse_factor)
            segment[index] += segment[index - stride];
        __syncthreads();
    }

    T segment_result = segment[(block_dim - 1)*2*coarse_factor + (2*coarse_factor - 1)];

    if constexpr (inclusive)
    {
        // step II.II: downward
        for (int stride = ((2*block_dim)/4)*coarse_factor; stride >= 1*coarse_factor; stride /= 2)
        {
            int index = (threadIdx.x + 1)*2*stride - 1;
            if (index + stride < 2*block_dim*coarse_factor)
                segment[index + stride] += segment[index];
            __syncthreads();
        }

        // step III: fixup
        {
            u32 thread_start = (threadIdx.x*2)*coarse_factor;
            for (u32 i = 0; i < 2; ++i)
            {
                u32 start = thread_start + i*coarse_factor;
                if (start > 0)
                {
                    T prev = segment[start - 1];
                    for (u32 j = 0; j < coarse_factor - 1; ++j)
                        segment[start + j] += prev;
                }
            }
        }
        __syncthreads();
    }
    else
    {
        if (threadIdx.x == (block_dim - 1))
        {
            u32 index = threadIdx.x*2*coarse_factor + (2*coarse_factor - 1);
            segment[index] = T(0);
        }
        __syncthreads();

        // step II.II: downsweep
        for (u32 stride = block_dim*coarse_factor; stride >= 1*coarse_factor; stride /= 2)
        {
            u32 index = (threadIdx.x + 1)*2*stride - 1;
            if (index < 2*block_dim*coarse_factor)
            {
                T temp = segment[index - stride];
                segment[index - stride] = segment[index];
                segment[index] += temp;
            }
            __syncthreads();
        }

        // step III: fixup
        {
            u32 thread_start = (threadIdx.x*2)*coarse_factor;
            for (u32 i = 0; i < 2; ++i)
            {
                u32 start = thread_start + i*coarse_factor;
                T last = segment[start + (coarse_factor - 1)];
                for (s32 j = coarse_factor - 1; j >= 0; --j)
                {
                    T prev = 0;
                    if (j > 0)
                        prev = segment[start + (j - 1)];
                    segment[start + j] = prev + last;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 2*coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = blockIdx.x*2*coarse_factor*block_dim + smem_index;
        if (gmem_index < count)
            array[gmem_index] = segment[smem_index];
    }

    return segment_result;
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void Scan_Upsweep(u64 count, T *array, T *summary, b32 work_efficient)
{
    T segment_result = T(0);
    if (work_efficient)
        segment_result = SegmentScan_WE<T, inclusive, block_dim, coarse_factor>(count, array);
    else
        segment_result = SegmentScan_KoggeStone<T, inclusive, block_dim, coarse_factor>(count, array);

    if (summary && (threadIdx.x == block_dim - 1))
        summary[blockIdx.x] = segment_result;
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void Scan_Downsweep(u64 count, T *array, T *summary, b32 work_efficient)
{
    u64 elements_per_thread = coarse_factor;
    if (work_efficient)
        elements_per_thread *= 2;

    if (blockIdx.x > 0)
    {
        T segment_result;
        if constexpr (inclusive)
            segment_result = summary[blockIdx.x - 1];
        else
            segment_result = summary[blockIdx.x];

        for (int i = 0; i < elements_per_thread; ++i)
        {
            int index = blockIdx.x*elements_per_thread*block_dim + i*block_dim + threadIdx.x;
            if (index < count)
                array[index] += segment_result;
        }
    }
}

template <typename T>
struct Scan_PassInput
{
    T *d_array;
    T *d_summary;
    u64 element_count;
    b32 work_efficient;    
};

template <typename T>
struct Scan_Input
{
    T *d_scratch;
    u32 pass_input_count;
    Scan_PassInput<T> *pass_inputs;
};

b32 g_work_efficient = 1;

template <typename T, u32 block_dim, u32 coarse_factor>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array)
{
    Scan_Input<T> *input = PushStruct(arena, Scan_Input<T>);

    u64 elements_per_block = g_work_efficient ? 2*coarse_factor*block_dim : coarse_factor*block_dim;

    // Allocate scratch
    {
        u64 size = 0ull;

        // Add the number of elements for the first pass..
        u64 out_count = (count + elements_per_block - 1)/elements_per_block;
        size += out_count*sizeof(*input->d_scratch);

        // Add the number of elements for the second pass.. and we are done because we can just ping-pong.
        out_count = (out_count + elements_per_block - 1)/elements_per_block;
        size += out_count*sizeof(*input->d_scratch);

        CUDACheck(cudaMalloc(&input->d_scratch, size));
    }

    input->pass_input_count = (u32)ceilf(logf(count)/logf(elements_per_block));
    input->pass_inputs = PushArrayZero(arena, Scan_PassInput<T>, input->pass_input_count);

    int out_count = (count + elements_per_block - 1)/elements_per_block; // output of the first pass
    T *d_pingpong[] = {input->d_scratch, input->d_scratch + out_count};
    u64 element_count = count;

    for (u32 i = 0; i < input->pass_input_count; ++i)
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + i;
        pass_input->d_array = d_array;
        pass_input->d_summary = d_pingpong[i % 2];
        pass_input->element_count = element_count;
        pass_input->work_efficient = g_work_efficient;

        int grid_dim = (element_count + elements_per_block - 1)/elements_per_block;
        
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

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
static void Scan(Scan_Input<T> *input, cudaStream_t stream)
{
    u32 k = input->pass_input_count;

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        u64 elements_per_block = coarse_factor*block_dim;
        if (pass_input->work_efficient)
            elements_per_block *= 2;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((Scan_Upsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary, pass_input->work_efficient)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        u64 elements_per_block = coarse_factor*block_dim;
        if (pass_input->work_efficient)
            elements_per_block *= 2;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((Scan_Downsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary, pass_input->work_efficient)));
    }
}

#include "benchmarking.cu"
using InputType = int;

#if 0
#include <thrust/scan.h>

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
#endif

template <typename T>
struct Data
{
    u64 count;
    T *h_array;
    T *d_array;
    Scan_Input<T> *input;
};

enum
{
    g_block_dim = 512,
    g_coarse_factor = 8,
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

    data->input = Scan_CreateInput<T, g_block_dim, g_coarse_factor>(arena, data->count, data->d_array);

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
    Scan<T, true, g_block_dim, g_coarse_factor>(data->input, stream);
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

    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);

    if (1)
    {
        TestScan();
        printf("All tests passed\n");
    }
    
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
            Scan<int, true, g_block_dim, g_coarse_factor>(data->input, 0);
            
            int *gpu_output = PushArray(scratch.arena, int, count);
            CUDACheck(cudaMemcpy(gpu_output, data->d_array, count*sizeof(int), cudaMemcpyDeviceToHost));

            SequentialInclusiveScan<int>(count, data->h_array);
            for (int i = 0; i < count; ++i)
            {
                if (data->h_array[i] != gpu_output[i])
                {
                    if (1)
                    {
                        for (int _i = 0; _i < 100; _i++)
                        {
                            printf("%d, ", gpu_output[_i]);
                        }
                        printf("\t...\t%d\n", gpu_output[count - 1]);
                    }
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
            Scan<int, false, g_block_dim, g_coarse_factor>(data->input, 0);
            
            int *gpu_output = PushArray(scratch.arena, int, count);
            CUDACheck(cudaMemcpy(gpu_output, data->d_array, count*sizeof(int), cudaMemcpyDeviceToHost));

            SequentialExclusiveScan<int>(count, data->h_array);
            for (int i = 0; i < count; ++i)
            {
                if (data->h_array[i] != gpu_output[i])
                {
                    if (1)
                    {
                        for (int _i = 0; _i < 100; _i++)
                        {
                            printf("%d, ", gpu_output[_i]);
                        }
                        printf("\t...\t%d\n", gpu_output[count - 1]);
                    }
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