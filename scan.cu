#include "core_types.h"
#include "core_memory.h"

#include "common.cuh"

#define SEGMENTED_KOGGE_STONE_SCAN 0
#define SEGMENTED_BRENT_KUNG_SCAN 1
#define SINGLE_PASS_SCAN 0
#define CUB_SCAN 0
#define THRUST_SCAN 0

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

#if SEGMENTED_KOGGE_STONE_SCAN
enum
{
    g_block_dim = 512,
    g_coarse_factor = 5,
};

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

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__device__ T SegmentScanKernel(u64 count, T *array)
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
__global__ void ScanKernelUpsweep(u64 count, T *array, T *summary)
{
    T segment_result = SegmentScanKernel<T, inclusive, block_dim, coarse_factor>(count, array);
    if (summary && (threadIdx.x == (block_dim - 1)))
        summary[blockIdx.x] = segment_result;
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void ScanKernelDownsweep(u64 count, T *array, T *summary)
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

template <typename T, u32 elements_per_block>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array)
{
    Scan_Input<T> *input = PushStructZero(arena, Scan_Input<T>);

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
    u32 elements_per_block = coarse_factor*block_dim;
    u32 k = input->pass_input_count;

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((ScanKernelUpsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((ScanKernelDownsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }
}

#elif SEGMENTED_BRENT_KUNG_SCAN
enum
{
    g_block_dim = 512,
    g_coarse_factor = 5,
};

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

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__device__ T SegmentScanKernel(u64 count, T *array)
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
__global__ void ScanKernelUpsweep(u64 count, T *array, T *summary)
{
    T segment_result = SegmentScanKernel<T, inclusive, block_dim, coarse_factor>(count, array);
    if (summary && (threadIdx.x == (block_dim - 1)))
        summary[blockIdx.x] = segment_result;
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void ScanKernelDownsweep(u64 count, T *array, T *summary)
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

template <typename T, u32 elements_per_block>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array)
{
    Scan_Input<T> *input = PushStructZero(arena, Scan_Input<T>);

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
    u32 elements_per_block = 2*coarse_factor*block_dim;
    u32 k = input->pass_input_count;

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((ScanKernelUpsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        Scan_PassInput<T> *pass_input = input->pass_inputs + pass;

        int grid_dim = (pass_input->element_count + elements_per_block - 1)/elements_per_block;
        CUDACheck((ScanKernelDownsweep<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }
}
#elif SINGLE_PASS_SCAN
enum
{
    g_block_dim = 512,
    g_coarse_factor = 23,
};

// NOTE(achal): If we are concerned about memory usage we can make the flag u8 and
// the atomic loading code slightly more complicated to handle the sizeof(Tile<T>)
// equals 4 case.
enum TileStatus : u32
{
    TileStatus_Invalid = 0,
    TileStatus_Partial,
    TileStatus_Inclusive,
};

template <typename T>
struct Tile
{
    TileStatus status;
    T value;
};

template <typename T>
__device__ Tile<T> LoadTile(Tile<T> *addr)
{
    Tile<T> tile;
    if constexpr (sizeof(tile) == 16)
    {
        u64 tile_data[2];
        tile_data[0] = atomicAdd((u64 *)addr, 0);
        __threadfence();
        tile_data[1] = atomicAdd((u64 *)addr + 1, 0);

        // flag | pad | value
        tile.status = (TileStatus)(tile_data[0] & 0xFFFFFFFF);
        tile.value = *reinterpret_cast<T *>(&tile_data[1]);
    }
    else if constexpr (sizeof(tile) == 8)
    {
        u64 tile_data = atomicAdd((u64 *)addr, 0);
        tile.status = (TileStatus)(tile_data & 0xFFFFFFFF);
        
        u32 tile_value_u32 = (u32)((tile_data >> 32) & 0xFFFFFFFF);
        tile.value = *reinterpret_cast<T *>(&tile_value_u32);
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
        tile_data[1] = *reinterpret_cast<u64 *>(&tile.value);
        tile_data[0] = tile_data[0] | tile.status;

        atomicExch((u64 *)addr + 1, tile_data[1]);
        __threadfence();
        atomicExch((u64 *)addr, tile_data[0]);
    }
    else if constexpr (sizeof(tile) == 8)
    {
        u32 tile_value_u32 = *reinterpret_cast<u32 *>(&tile.value);
        u64 packed = ((u64)tile_value_u32 << 32) | tile.status;
        atomicExch((u64 *)addr, packed);
    }
    else
    {
        static_assert(0);
    }
}

template <typename T>
struct Scan_Input
{
    u64 count;
    T *d_array;
    Tile<T> *tiles;
    u64 *block_id_counter;
};

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
__global__ void ScanKernel(u64 count, T *array, Tile<T> *tiles, u64 *block_id_counter)
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

    __shared__ T exclusive_prefix;
    if (block_id == 0)
    {
        if (threadIdx.x == 0)
        {
            Tile<T> tile;
            tile.status = TileStatus_Inclusive;
            tile.value = segment_result;
            StoreTile(&tiles[block_id], tile);
        }

        exclusive_prefix = 0;
    }
    else
    {   
        if (threadIdx.x == 0)
        {
            {
                Tile<T> tile;
                tile.status = TileStatus_Partial;
                tile.value = segment_result;
                StoreTile(&tiles[block_id], tile);
            }

            exclusive_prefix = 0;

            // Check on the predecessor(s)
            int predecessor_index = block_id - 1;
            while (predecessor_index >= 0)
            {
                // Wait until the predecessor tile becomes valid i.e. not TileStatus_Invalid
                Tile<T> predecessor_tile = LoadTile(&tiles[predecessor_index]);
                while (predecessor_tile.status == TileStatus_Invalid)
                {
                    predecessor_tile = LoadTile(&tiles[predecessor_index]);
                }

                if (predecessor_tile.status == TileStatus_Inclusive) // early termination of lookback
                {
                    exclusive_prefix += predecessor_tile.value;
                    break;
                }
                else if (predecessor_tile.status == TileStatus_Partial)
                {
                    exclusive_prefix += predecessor_tile.value;
                }
                else
                {
                    printf("Invalid tile status: %d\n", predecessor_tile.status);
                    Assert(0); // invalid code path
                }

                predecessor_index--;
            }

            {
                Tile<T> tile;
                tile.status = TileStatus_Inclusive;
                tile.value = exclusive_prefix + segment_result;
                StoreTile(&tiles[block_id], tile);
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < coarse_factor; ++i)
    {
        int smem_index = i*block_dim + threadIdx.x;
        int gmem_index = block_id*coarse_factor*block_dim + smem_index;
        if (gmem_index < count)
            array[gmem_index] = exclusive_prefix + segment[smem_index];
    }
}

template <typename T, u32 elements_per_block>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array)
{
    Scan_Input<T> *input = PushStructZero(arena, Scan_Input<T>);

    input->count = count;
    input->d_array = d_array;

    // Allocate scratch
    {
        // TODO(achal): It will be better if we can allocate all of this in one go,
        // but we need to be congizant about alignment requirements.

        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

        u64 grid_dim = (count + elements_per_block - 1)/elements_per_block;
        
        Tile<T> *tiles = PushArrayZero(scratch.arena, Tile<T>, grid_dim);

        CUDACheck(cudaMalloc(&input->tiles, grid_dim*sizeof(*input->tiles)));
        CUDACheck(cudaMemcpy(input->tiles, tiles, grid_dim*sizeof(*input->tiles), cudaMemcpyHostToDevice));
        
        // block_id_counter
        CUDACheck(cudaMalloc(&input->block_id_counter, sizeof(u64)));
        CUDACheck(cudaMemset(input->block_id_counter, 0, sizeof(u64)));

        ScratchEnd(&scratch);
    }

    return input;
}

template <typename T>
static void Scan_DestroyInput(Scan_Input<T> *input)
{
    CUDACheck(cudaFree(input->tiles));
    CUDACheck(cudaFree(input->block_id_counter));
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
static void Scan(Scan_Input<T> *input, cudaStream_t stream)
{
    u32 elements_per_block = coarse_factor*block_dim;
    u64 grid_dim = (input->count + elements_per_block - 1)/elements_per_block;
    CUDACheck((ScanKernel<T, inclusive, block_dim, coarse_factor><<<grid_dim, block_dim, 0, stream>>>(input->count, input->d_array, input->tiles, input->block_id_counter)));
}
#elif CUB_SCAN
#include <cub/device/device_scan.cuh>

// NOTE(achal): CUB should just ignore this.
enum
{
    g_block_dim = 512,
    g_coarse_factor = 5,
};

template <typename T>
struct Scan_Input
{
    u64 count;
    T *d_array;
    void *d_temp_storage;
    size_t temp_storage_bytes;
};

template <typename T, u32 elements_per_block>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array)
{
    Scan_Input<T> *input = PushStructZero(arena, Scan_Input<T>);

    input->count = count;
    input->d_array = d_array;
    // NOTE(achal): I am assuming that the storage requirements are the same for inclusive and exclusive scans!
    cub::DeviceScan::InclusiveSum(input->d_temp_storage, input->temp_storage_bytes, input->d_array, input->count);
    CUDACheck(cudaMalloc(&input->d_temp_storage, input->temp_storage_bytes));

    return input;
}

template <typename T>
static void Scan_DestroyInput(Scan_Input<T> *input)
{
    CUDACheck(cudaFree(input->d_temp_storage));
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
void Scan(Scan_Input<T> *input, cudaStream_t stream)
{
    if constexpr (inclusive)
    {
        cub::DeviceScan::InclusiveSum(input->d_temp_storage, input->temp_storage_bytes, input->d_array, input->count, stream);
    }
    else
    {
        cub::DeviceScan::ExclusiveSum(input->d_temp_storage, input->temp_storage_bytes, input->d_array, input->count, stream);
    }
}
#elif THRUST_SCAN
#include <thrust/scan.h>

// NOTE(achal): Thrust should just ignore this.
enum
{
    g_block_dim = 512,
    g_coarse_factor = 5,
};

template <typename T>
struct Scan_Input
{
    u64 count;
    T *d_array;
};

template <typename T, u32 elements_per_block>
static Scan_Input<T> *Scan_CreateInput(Arena *arena, u64 count, T *d_array)
{
    Scan_Input<T> *input = PushStructZero(arena, Scan_Input<T>);

    input->count = count;
    input->d_array = d_array;

    return input;
}

template <typename T>
static void Scan_DestroyInput(Scan_Input<T> *input)
{
}

template <typename T, bool inclusive, u32 block_dim, u32 coarse_factor>
void Scan(Scan_Input<T> *input, cudaStream_t stream)
{
    if constexpr (inclusive)
    {
        thrust::inclusive_scan(thrust::cuda::par.on(stream), input->d_array, input->d_array + input->count, input->d_array);
    }
    else
    {
        thrust::exclusive_scan(thrust::cuda::par.on(stream), input->d_array, input->d_array + input->count, input->d_array);
    }
}
#else
#error "No scan algorithm selected"
#endif

#include "benchmarking.cu"
using InputType = int;

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
    Data<T> *data = PushStructZero(arena, Data<T>);

    data->count = count;

    data->h_array = PushArrayZero(arena, T, data->count);
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
    Scan<T, true, block_dim, coarse_factor>(data->input, stream);
}

template <u32 block_dim, u32 coarse_factor>
static void Test_Scan();

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
        Test_Scan<512, 5>();
        printf("All tests passed\n");
    }
    
    printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
    printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
    printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    if (file_name)
    {
        Benchmark<InputType, g_block_dim, g_coarse_factor>(peak_gbps, peak_gflops, file_name);
    }

    return 0;
}

template <bool inclusive, u32 block_dim, u32 coarse_factor>
static void Test_Scan_(u64 count)
{
    using DataType = int;

    Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));            
    Data<DataType> *data = CreateData<DataType, block_dim*coarse_factor>(scratch.arena, count, 0);

    Scan<DataType, inclusive, block_dim, coarse_factor>(data->input, 0);

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
static void Test_Scan()
{
    printf("Block dim: %d, Coarse factor: %d\n", block_dim, coarse_factor);

    srand(0);
    int test_count = 10;

    for (int test_index = 0; test_index < test_count; ++test_index)
    {
        u64 count = GetRandomNumber(30);
        printf("\tCount: %llu:\t", count);

        printf("Inclusive, ");
        Test_Scan_<true, block_dim, coarse_factor>(count);
        printf("Passed, ");

        printf("Exclusive, ");
        Test_Scan_<false, block_dim, coarse_factor>(count);
        printf("Passed\n");
    }
}

static void Test_ScanSuite()
{
    enum { block_dim = 512, };

    Test_Scan<block_dim, 1>();
    Test_Scan<block_dim, 2>();
    Test_Scan<block_dim, 3>();
    Test_Scan<block_dim, 4>();
    Test_Scan<block_dim, 5>();
    Test_Scan<block_dim, 6>();
    Test_Scan<block_dim, 7>();
    Test_Scan<block_dim, 8>();
}