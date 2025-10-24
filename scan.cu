#include "core_types.h"
#include "core_memory.h"

#include "common.cuh"

#define BENCHMARK_THRUST 0
#define SEGMENTED_SCAN 0

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
        // TODO(achal): It would be better to allocate all shared memory at one place.
        __shared__ T segment_result_shared;
        if (threadIdx.x == block_dim - 1)
        {
            u64 index = blockIdx.x*coarse_factor*block_dim + (block_dim - 1)*coarse_factor + (coarse_factor - 1);
            if (index > count - 1)
                index = count - 1;
            segment_result_shared = segment_result + array[index];
        }
        __syncthreads();
        segment_result = segment_result_shared;
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
__global__ void Scan1_Upsweep(u64 count, T *array, T *summary)
{
    T segment_result = SegmentScan_KoggeStone<T, inclusive, block_dim, coarse_factor>(count, array);
    if (summary && (threadIdx.x == (block_dim - 1)))
        summary[blockIdx.x] = segment_result;
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void Scan1_Downsweep(u64 count, T *array, T *summary)
{
    u8 elements_per_thread = coarse_factor;

    if (blockIdx.x > 0)
    {
        T segment_result;
        if constexpr (inclusive)
            segment_result = summary[blockIdx.x - 1];
        else
            segment_result = summary[blockIdx.x];

        for (u8 i = 0; i < elements_per_thread; ++i)
        {
            u64 index = blockIdx.x*elements_per_thread*block_dim + i*block_dim + threadIdx.x;
            if (index < count)
                array[index] += segment_result;
        }
    }
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__device__ T SegmentScan_BrentKung(u64 count, T *array)
{
    __shared__ T segment[2*coarse_factor*block_dim];

    for (int i = 0; i < 2*coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = blockIdx.x*2*coarse_factor*block_dim + smem_index;

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
    if constexpr (!inclusive)
    {
        __shared__ T segment_result_shared;
        if (threadIdx.x == block_dim - 1)
        {
            u64 index = blockIdx.x*2*coarse_factor*block_dim + (block_dim - 1)*2*coarse_factor + (2*coarse_factor - 1);
            if (index > count - 1)
                index = count - 1;
            segment_result_shared = segment_result + array[index];
        }
        __syncthreads();
        segment_result = segment_result_shared;
    }

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
__global__ void Scan2_Upsweep(u64 count, T *array, T *summary)
{
    T segment_result = SegmentScan_BrentKung<T, inclusive, block_dim, coarse_factor>(count, array);
    if (summary && (threadIdx.x == (block_dim - 1)))
        summary[blockIdx.x] = segment_result;
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void Scan2_Downsweep(u64 count, T *array, T *summary)
{
    u8 elements_per_thread = 2*coarse_factor;

    if (blockIdx.x > 0)
    {
        T segment_result;
        if constexpr (inclusive)
            segment_result = summary[blockIdx.x - 1];
        else
            segment_result = summary[blockIdx.x];

        for (u8 i = 0; i < elements_per_thread; ++i)
        {
            u64 index = blockIdx.x*elements_per_thread*block_dim + i*block_dim + threadIdx.x;
            if (index < count)
                array[index] += segment_result;
        }
    }
}

template <typename T>
struct Tile
{    
    // NOTE(achal): If we are concerned about memory usage we can make the flag u8 and
    // the atomic loading code slightly more complicated to handle the sizeof(Tile<T>)
    // equals 4 case.
    u32 flag;
    T value;
};

template <typename T>
__device__ Tile<T> LoadTile(Tile<T> *addr)
{
    Tile<T> tile = {0, 0};
    if constexpr (sizeof(tile) == 16)
    {
        u64 tile_data[2];
        tile_data[0] = atomicAdd((u64 *)addr, 0);
        __threadfence();
        tile_data[1] = atomicAdd((u64 *)addr + 1, 0);

        // flag | pad | value
        tile.flag = tile_data[0] & 0xFFFFFFFF;
        tile.value = (T)(tile_data[1]);
    }
    else if constexpr (sizeof(tile) == 8)
    {
        u64 tile_data = atomicAdd((u64 *)addr, 0);
        tile.flag = tile_data & 0xFFFFFFFF;
        tile.value = (T)((tile_data >> 32) & 0xFFFFFFFF);
    }
    else
    {
        static_assert(0);
    }
    return tile;
}

template <typename T>
__device__ void StoreTile(Tile<T> *addr, Tile<T> tile)
{
    if constexpr (sizeof(tile) == 16)
    {
        u64 tile_data[2];
        tile_data[1] = (u64)tile.value;
        tile_data[0] = tile_data[0] | tile.flag;

        atomicAdd((u64 *)addr + 1, tile_data[1]);
        __threadfence();
        atomicAdd((u64 *)addr, tile_data[0]);
    }
    else if constexpr (sizeof(tile) == 8)
    {
        u64 packed = ((u64)tile.value << 32) | tile.flag;
        atomicAdd((u64 *)addr, packed);
    }
    else
    {
        static_assert(0);
    }
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void Scan3Kernel(u64 count, T *array, Tile<T> *tiles, u64 *block_id_counter)
{
    __shared__ u64 block_id;
    if (threadIdx.x == 0)
    {
        block_id = atomicAdd(block_id_counter, 1);
    }
    __syncthreads();

    __shared__ T segment[coarse_factor*block_dim];

    for (int i = 0; i < coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = block_id*coarse_factor*block_dim + smem_index;
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
        __shared__ T segment_result_shared;
        if (threadIdx.x == block_dim - 1)
        {
            u64 index = block_id*coarse_factor*block_dim + (block_dim - 1)*coarse_factor + (coarse_factor - 1);
            if (index > count - 1)
                index = count - 1;
            segment_result_shared = segment_result + array[index];
        }
        __syncthreads();
        segment_result = segment_result_shared;
    }

    __shared__ T prev_block_sums;
    if (threadIdx.x == 0)
    {
        Tile<T> src_tile = LoadTile(&tiles[block_id]);
        while (src_tile.flag == 0)
        {
            src_tile = LoadTile(&tiles[block_id]);
        }
        
        prev_block_sums = src_tile.value;
        Tile<T> dst_tile = {1, prev_block_sums + segment_result};
        StoreTile(&tiles[block_id + 1], dst_tile);
    }
    __syncthreads();

    for (int i = 0; i < coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = block_id*coarse_factor*block_dim + smem_index;
        if (gmem_index < count)
            array[gmem_index] = prev_block_sums + segment[smem_index];
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
#if SEGMENTED_SCAN
    T *d_scratch;
    u32 pass_input_count;
    Scan_PassInput<T> *pass_inputs;
#else
    u64 count;
    T *d_array;
    Tile<T> *tiles;
    u64 *block_id_counter;
#endif
};

template <typename T, u32 elements_per_block>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array)
{
    Scan_Input<T> *input = PushStruct(arena, Scan_Input<T>);

#if SEGMENTED_SCAN
    input->pass_input_count = (u32)ceilf(logf(count)/logf(elements_per_block));
    input->pass_inputs = PushArrayZero(arena, Scan_PassInput<T>, input->pass_input_count);

    // Allocate scratch
    {
        u64 element_count = count;
        u64 size = 0ull;
        for (u32 i = 0; i < input->pass_input_count - 1; ++i)
        {
            u64 grid_dim = (element_count + elements_per_block - 1)/elements_per_block;
            size += grid_dim*sizeof(*input->d_scratch);
            
            element_count = grid_dim;
        }

        CUDACheck(cudaMalloc(&input->d_scratch, size));
    }

    u64 element_count = count;
    u64 scratch_offset = 0;
    for (u32 i = 0; i < input->pass_input_count; ++i)
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + i;
        pass_input->d_array = d_array;
        pass_input->d_summary = input->d_scratch + scratch_offset;
        pass_input->element_count = element_count;

        u64 grid_dim = (element_count + elements_per_block - 1)/elements_per_block;
        scratch_offset += grid_dim;
        
        d_array = pass_input->d_summary;
        element_count = grid_dim;
    }

    input->pass_inputs[input->pass_input_count - 1].d_summary = 0; // the "top" pass doesn't require a summary
#else
    input->count = count;
    input->d_array = d_array;

    // Allocate scratch
    {
        // TODO(achal): It will be better if we can allocate all of this in one go,
        // but we need to be congizant about alignment requirements.

        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

        u64 grid_dim = (count + elements_per_block - 1)/elements_per_block;
        
        Tile<T> *tiles = PushArrayZero(scratch.arena, Tile<T>, grid_dim);
        tiles[0].flag = 1;

        CUDACheck(cudaMalloc(&input->tiles, grid_dim*sizeof(*input->tiles)));
        CUDACheck(cudaMemcpy(input->tiles, tiles, grid_dim*sizeof(*input->tiles), cudaMemcpyHostToDevice));
        
        // block_id_counter
        CUDACheck(cudaMalloc(&input->block_id_counter, sizeof(u64)));
        CUDACheck(cudaMemset(input->block_id_counter, 0, sizeof(u64)));

        ScratchEnd(&scratch);
    }
#endif

    return input;
}

template <typename T>
static void Scan_DestroyInput(Scan_Input<T> *input)
{
#if SEGMENTED_SCAN
    CUDACheck(cudaFree(input->d_scratch));
#else
    CUDACheck(cudaFree(input->tiles));
    CUDACheck(cudaFree(input->block_id_counter));
#endif
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
static void Scan1(Scan_Input<T> *input, cudaStream_t stream)
{
    u32 elements_per_block = coarse_factor*block_dim;
    u32 k = input->pass_input_count;

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((Scan1_Upsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((Scan1_Downsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
static void Scan2(Scan_Input<T> *input, cudaStream_t stream)
{
    u32 elements_per_block = 2*coarse_factor*block_dim;
    u32 k = input->pass_input_count;

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((Scan2_Upsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((Scan2_Downsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
static void Scan3(Scan_Input<T> *input, cudaStream_t stream)
{
    u32 elements_per_block = coarse_factor*block_dim;
    u64 grid_dim = (input->count + elements_per_block - 1)/elements_per_block;
    CUDACheck((Scan3Kernel<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(input->count, input->d_array, input->tiles, input->block_id_counter)));
}

#include "benchmarking.cu"
using InputType = int;

#if BENCHMARK_THRUST
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

// StaticAssert(IsPoT(g_block_dim));

template <typename T, u32 elements_per_block>
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

    data->input = Scan_CreateInput<T, elements_per_block>(arena, data->count, data->d_array);

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
    // SequentialExclusiveScan<T>(data->count, data->h_array);
    b32 result = (memcmp(gpu_out, data->h_array, data->count*sizeof(*gpu_out)) == 0);

    ScratchEnd(&scratch);

    return result;
}

template <typename T, u32 block_dim, u32 coarse_factor>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
#if BENCHMARK_THRUST
    ThrustInclusiveScan<T>(data->count, data->d_array, data->d_array, stream);
#else
#if SEGMENTED_SCAN
    // Scan1<T, true, block_dim, coarse_factor>(data->input, stream);
    Scan2<T, true, block_dim, coarse_factor>(data->input, stream);
#else
    Scan3<T, true, block_dim, coarse_factor>(data->input, stream);
#endif
#endif
}

enum Test_ScanAlgorithm
{
    Test_ScanAlgorithm_1 = 0, // Kogge-Stone
    Test_ScanAlgorithm_2, // Brent-Kung
    Test_ScanAlgorithm_Count,
};

template <u32 block_dim, u32 coarse_factor>
static void Test_Scan(Test_ScanAlgorithm algorithm);

static void Test_ScanSuite();

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

    if (0)
    {
        // Test_ScanSuite();
        Test_Scan<512, 5>((Test_ScanAlgorithm)Test_ScanAlgorithm_1);
        printf("All tests passed\n");
    }
    
    printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
    printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
    printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    if (file_name)
    {
        enum { block_dim = 512, coarse_factor = 5 };
        Benchmark<InputType, block_dim, coarse_factor>(peak_gbps, peak_gflops, file_name);
    }

    return 0;
}

template <bool inclusive, u32 block_dim, u32 coarse_factor>
static void Test_Scan_(Test_ScanAlgorithm algorithm, u64 count)
{
    using DataType = int;

    Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));
            
    Data<DataType> *data = CreateData<DataType, block_dim*coarse_factor>(scratch.arena, count, 0);

    switch (algorithm)
    {
#if SEGMENTED_SCAN
        case Test_ScanAlgorithm_1:
            Scan1<DataType, inclusive, block_dim, coarse_factor>(data->input, 0);
            break;
        case Test_ScanAlgorithm_2:
            Scan2<DataType, inclusive, block_dim, coarse_factor>(data->input, 0);
            break;
        default:
            Assert(0);
#else
        case Test_ScanAlgorithm_1:
            Scan3<DataType, inclusive, block_dim, coarse_factor>(data->input, 0);
            break;
        default:
            Assert(0);
#endif
    }

    DataType *gpu_output = PushArray(scratch.arena, DataType, count);
    CUDACheck(cudaMemcpy(gpu_output, data->d_array, count*sizeof(DataType), cudaMemcpyDeviceToHost));

    if constexpr (inclusive)
    {
        SequentialInclusiveScan<DataType>(count, data->h_array);
    }
    else
    {
        SequentialExclusiveScan<DataType>(count, data->h_array);
    }

    for (int i = 0; i < count; ++i)
    {
        if (data->h_array[i] != gpu_output[i])
        {
            printf("[FAIL] Result[%d] (GPU): %d \tExpected: %d\n", i, gpu_output[i], data->h_array[i]);
            break;
        }
    }

    DestroyData<DataType>(data);
    ScratchEnd(&scratch);
}

template <u32 block_dim, u32 coarse_factor>
static void Test_Scan(Test_ScanAlgorithm algorithm)
{
    printf("Algorithm: %d, Block dim: %d, Coarse factor: %d\n", algorithm, block_dim, coarse_factor);

    srand(0);
    int test_count = 10;

    for (int test_index = 0; test_index < test_count; ++test_index)
    {
        u64 count = GetRandomNumber(30);
        printf("\tCount: %llu:\t", count);

        printf("Inclusive, ");
        Test_Scan_<true, block_dim, coarse_factor>(algorithm, count);
        printf("Passed, ");

        printf("Exclusive, ");
        Test_Scan_<false, block_dim, coarse_factor>(algorithm, count);
        printf("Passed\n");
    }
}

static void Test_ScanSuite()
{
    enum { block_dim = 512, };
#if SEGMENTED_SCAN
    for (int algorithm = 0; algorithm < Test_ScanAlgorithm_Count; ++algorithm)
#else
    for (int algorithm = 0; algorithm < 1; ++algorithm)
#endif
    {
        Test_Scan<block_dim, 1>((Test_ScanAlgorithm)algorithm);
        Test_Scan<block_dim, 2>((Test_ScanAlgorithm)algorithm);
        Test_Scan<block_dim, 3>((Test_ScanAlgorithm)algorithm);
        Test_Scan<block_dim, 4>((Test_ScanAlgorithm)algorithm);
        Test_Scan<block_dim, 5>((Test_ScanAlgorithm)algorithm);
        Test_Scan<block_dim, 6>((Test_ScanAlgorithm)algorithm);
        Test_Scan<block_dim, 7>((Test_ScanAlgorithm)algorithm);
        Test_Scan<block_dim, 8>((Test_ScanAlgorithm)algorithm);
    }
}