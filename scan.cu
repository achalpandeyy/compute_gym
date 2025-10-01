#include "common.cuh"

template <typename T>
static void SequentialScan(int count, T *array)
{
    for (int i = 1; i < count; ++i)
    {
        array[i] += array[i - 1];
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
__global__ void ScanUpsweep(u64 count, T *array, T *summary)
{
    __shared__ T segment[SEGMENT_SIZE];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < count)
        segment[threadIdx.x] = array[index];
    else
        segment[threadIdx.x] = T(0);

    T scan_result = BlockScan1(segment);

    if (index < count)
        array[index] = scan_result;

    if (summary && (threadIdx.x == blockDim.x - 1))
        summary[blockIdx.x] = scan_result;
}

template <typename T>
__global__ void ScanDownsweep(u64 count, T *array, T *summary)
{
    if (blockIdx.x > 0)
    {
        int prev_block_index = blockIdx.x - 1;
        T prev_block_sum = summary[prev_block_index];
        
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index < count)
            array[index] += prev_block_sum;
    }
}

template <typename T>
static void Scan1(u64 count, T *array)
{
    assert(count <= (4ull*1024*1024*1024)/sizeof(*array));

    int block_dim = SEGMENT_SIZE;

    u32 k = (u32)ceilf(logf(count)/logf(block_dim));

    T *d_scratch = 0;
    {
        u64 scratch_memory_size = 0ull;

        // Add the memory for the first pass..
        int out_count = (count + block_dim - 1)/block_dim;
        scratch_memory_size += out_count*sizeof(*array);

        // Add the memory for the second pass.. and we are done because we can just ping-pong.
        out_count = (out_count + block_dim - 1)/block_dim;
        scratch_memory_size += out_count*sizeof(*array); 
        CUDACheck(cudaMalloc(&d_scratch, scratch_memory_size));
    }

    struct ScanPassInput
    {
        T *d_array;
        T *d_summary;
        u64 element_count;
    };

    ScanPassInput *pass_inputs = (ScanPassInput *)malloc(k*sizeof(ScanPassInput));
    {
        int out_count = (count + block_dim - 1)/block_dim; // output of the first pass

        T *d_array = array;
        T *d_pingpong[] = {d_scratch, d_scratch + out_count};
        u64 element_count = count;

        for (u32 pass = 0; pass < k; ++pass)
        {
            ScanPassInput *pass_input = pass_inputs + pass;
            pass_input->d_array = d_array;
            pass_input->d_summary = d_pingpong[pass % 2];
            pass_input->element_count = element_count;

            int grid_dim = (element_count + block_dim - 1)/block_dim;
            
            d_array = pass_input->d_summary;
            element_count = grid_dim;
        }

        pass_inputs[k - 1].d_summary = 0; // the "top" pass doesn't require a summary
    }

    for (u32 pass = 0; pass < k; ++pass) // k upsweeps
    {
        ScanPassInput *pass_input = pass_inputs + pass;
        int grid_dim = (pass_input->element_count + block_dim - 1)/block_dim;
        CUDACheck((ScanUpsweep<T><<<grid_dim, block_dim>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    for (s32 pass = k - 2; pass >= 0; --pass) // k - 1 downsweeps
    {
        ScanPassInput *pass_input = pass_inputs + pass;
        int grid_dim = (pass_input->element_count + block_dim - 1)/block_dim;
        CUDACheck((ScanDownsweep<T><<<grid_dim, block_dim>>>(pass_input->element_count, pass_input->d_array, pass_input->d_summary)));
    }

    free(pass_inputs);
    CUDACheck(cudaFree(d_scratch));
}

#include "benchmarking.cu"

using InputType = int;

template <typename T>
struct Data
{
    u64 count;
    T *h_array;
    T *d_array;
};

template <typename T>
static void CreateData(Data<T> *data, cudaStream_t stream)
{
    data->h_array = (T *)malloc(data->count*sizeof(*data->h_array));
    for (u64 i = 0; i < data->count; ++i)
        data->h_array[i] = T(1);

    CUDACheck(cudaMalloc(&data->d_array, data->count*sizeof(*data->d_array)));
    CUDACheck(cudaMemcpyAsync(data->d_array, data->h_array, data->count*sizeof(*data->d_array), cudaMemcpyHostToDevice, stream));

    CUDACheck(cudaStreamSynchronize(stream));
}

template <typename T>
static void DestroyData(Data<T> *data)
{
    CUDACheck(cudaFree(data->d_array));
    free(data->h_array);
}

template <typename T>
static b32 ValidateGPUOutput(Data<T> *data)
{
    T *gpu_out = (T *)malloc(data->count*sizeof(*gpu_out));
    CUDACheck(cudaMemcpy(gpu_out, data->d_array, data->count*sizeof(*gpu_out), cudaMemcpyDeviceToHost));

    SequentialScan<T>(data->count, data->h_array);
    b32 result = (memcmp(gpu_out, data->h_array, data->count*sizeof(*gpu_out)) == 0);
    free(gpu_out);

    return result;
}

template <typename T>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream)
{
    Scan1<T>(data->count, data->d_array);
}

static void TestScan();

int main()
{
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

    Benchmark<InputType>(peak_gbps, peak_gflops, "bench_scan.bin");

    return 0;
}

static void TestScan()
{
    srand(0);
    int test_count = 10;
    while (test_count > 0)
    {
        u64 count = GetRandomNumber(30);
        printf("Count: %llu", count);

        int *array = (int *)malloc(count*sizeof(int));
        for (int i = 0; i < count; ++i)
            array[i] = 1;

        int *d_array = 0;
        CUDACheck(cudaMalloc(&d_array, count*sizeof(int)));
        CUDACheck(cudaMemcpy(d_array, array, count*sizeof(int), cudaMemcpyHostToDevice));

        Scan1<int>(count, d_array);

        int *gpu_output = (int *)malloc(count*sizeof(int));
        CUDACheck(cudaMemcpy(gpu_output, d_array, count*sizeof(int), cudaMemcpyDeviceToHost));

        SequentialScan<int>(count, array);
        for (int i = 0; i < count; ++i)
        {
            if (array[i] != gpu_output[i])
            {
                printf("[FAIL] Result (GPU): %d \tExpected: %d\n", gpu_output[i], array[i]);
                exit(1);
            }
        }
        printf(",\tPassed\n");

        free(gpu_output);
        CUDACheck(cudaFree(d_array));
        free(array);

        --test_count;
    }
}