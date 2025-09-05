#include <stdio.h>
#include <stdint.h>
#include <assert.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u8 b8;
typedef u32 b32;

typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef float f32;
typedef double f64;

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

static void GetPeakMeasurements(f64 *peak_gbps, f64 *peak_gflops, bool print_device_info = false)
{
    cudaFuncAttributes attr;
    CUDACheck(cudaFuncGetAttributes(&attr, DummyKernel));

    int major_ver = attr.ptxVersion/10;
    int minor_ver = attr.ptxVersion%10;
    
    int device;
    CUDACheck(cudaGetDevice(&device));

    cudaDeviceProp device_prop = { 0 };
    CUDACheck(cudaGetDeviceProperties(&device_prop, device));
    
    // Peak GFLOPS
    {
        int sm_count;
        CUDACheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

        // NOTE(achal): Compute capability 7.5 has 64 CUDA cores per SM.
        assert(device_prop.major == 7 && device_prop.minor == 5);
        int cuda_cores_per_sm = 64;

        int peak_clock_freq;
        CUDACheck(cudaDeviceGetAttribute(&peak_clock_freq, cudaDevAttrClockRate, device));
        *peak_gflops = (sm_count*cuda_cores_per_sm*2.0*peak_clock_freq*1000.0)/1.0e9;
    }

    // Peak GBPS
    {
        int peak_mem_clock_freq;
        CUDACheck(cudaDeviceGetAttribute(&peak_mem_clock_freq, cudaDevAttrMemoryClockRate, device));

        int bus_width;
        CUDACheck(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device));

        printf("Peak memory clock frequency: %d kHz\n", peak_mem_clock_freq);
        printf("Bus width: %d bits\n", bus_width);

        // NOTE(achal): 2.0 is for double transfer (DDR).
        *peak_gbps = (peak_mem_clock_freq*1000.0*2.0)*(bus_width/8.0)/(1024.0*1024.0*1024.0);
    }

    if (print_device_info)
    {
        printf("Device name: %s\n", device_prop.name);
        printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("PTX version: %d.%d\n", major_ver, minor_ver);

        printf("Total Global Memory: %.2f GB\n", (device_prop.totalGlobalMem/(1024.f*1024.f*1024.f)));
        printf("Shared Memory (per block): %.2f KB\n", (device_prop.sharedMemPerBlock/1024.f));
        printf("Total Constant Memory: %.2f KB\n", (device_prop.totalConstMem/1024.f));

        printf("Warp Size: %d threads\n", device_prop.warpSize);
        printf("Max threads per block: %d\n", device_prop.maxThreadsPerBlock);
        printf("Max Block dimension: %dx%dx%d\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("Max Grid dimension: %dx%dx%d\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]); 
        printf("32-bit registers (per block): %d\n", device_prop.regsPerBlock);
        printf("SM count: %d\n", device_prop.multiProcessorCount);
        printf("Max blocks per SM: %d\n", device_prop.maxBlocksPerMultiProcessor);
        printf("Max threads per SM: %d\n", device_prop.maxThreadsPerMultiProcessor);
    }
}

__global__ void CopyKernel(f32 * __restrict__ input, f32 * __restrict__ output)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    f32 value = input[index];
    if (threadIdx.x == 0)
        output[index] = value;
}

static void BenchmarkCopy()
{
    int warmup_count = 3;
    int rep_count = 20;

    FILE *file = fopen("bench_copy.bin", "wb");
    assert(file);
    for (int exp = 1; exp <= 30; ++exp)
    {
        u64 array_count = 1 << exp;

        f32 *d_input = 0;
        CUDACheck(cudaMalloc(&d_input, array_count*sizeof(f32)));
        f32 *d_output = 0;
        CUDACheck(cudaMalloc(&d_output, array_count*sizeof(f32)));

        cudaEvent_t start_event, stop_event;
        CUDACheck(cudaEventCreate(&start_event));
        CUDACheck(cudaEventCreate(&stop_event));

        int block_dim = 1024;
        int grid_dim = (array_count + block_dim - 1)/block_dim;

        for (int i = 0; i < warmup_count; ++i)
        {
            CopyKernel<<<grid_dim, block_dim>>>(d_input, d_output);
        }

        f64 ms = 0;
        for (int i = 0; i < rep_count; ++i)
        {
            CUDACheck(cudaEventRecord(start_event));
            CopyKernel<<<grid_dim, block_dim>>>(d_input, d_output);
            CUDACheck(cudaEventRecord(stop_event));
            CUDACheck(cudaEventSynchronize(stop_event));

            f32 curr_ms;
            CUDACheck(cudaEventElapsedTime(&curr_ms, start_event, stop_event));
            ms += curr_ms;
        }

        ms /= rep_count;

        f64 bandwidth = (1000.0*(array_count*sizeof(f32)))/(ms*1024.0*1024.0*1024.0);

        fwrite(&array_count, sizeof(u64), 1, file);
        fwrite(&ms, sizeof(f64), 1, file);
        fwrite(&bandwidth, sizeof(f64), 1, file);

        printf("Elapsed (GPU): %f ms\n", ms);
        printf("Bandwidth: %f GB/s\n", bandwidth);

        CUDACheck(cudaFree(d_output));
        CUDACheck(cudaFree(d_input));
    }
    
    fclose(file);
}

__global__ void ReduceKernel1(f32 *input)
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

__global__ void ReduceKernel2(u32 count, f32 *input, f32 *output)
{
    int segment_start = blockIdx.x*(2*blockDim.x);
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if ((threadIdx.x % stride) == 0)
        {
            int index = segment_start + 2*threadIdx.x;
            if (index < count)
            {
                f32 temp = 0.f;
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

__global__ void ReduceKernel3(u32 count, f32 *input, f32 *output)
{
    int segment_start = blockIdx.x*(2*blockDim.x);
    for (int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            int index = segment_start + threadIdx.x;
            if (index < count)
            {
                f32 temp = 0.f;
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

__global__ void ReduceKernel4(u32 count, f32 *input, f32 *output)
{
    int segment_start = blockIdx.x*COARSE_FACTOR*blockDim.x;
    int index = segment_start + threadIdx.x;

    f32 sum = 0.f;
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
                f32 temp = 0.f;
                if (index + stride < count)
                    temp = input[index + stride];
                input[index] += temp;
            }
        }
    }

    if (threadIdx.x == 0)
        atomicAdd(output, input[segment_start]);
}

__global__ void ReduceKernel5(u32 count, f32 *input, f32 *output)
{
    extern __shared__ f32 input_s[];

    int segment_start = blockIdx.x*COARSE_FACTOR*blockDim.x;
    int index = segment_start + threadIdx.x;

    f32 sum = 0.f;
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

#if COMPILING_FROM_PYTORCH
void Reduce(torch::Tensor input, torch::Tensor output)
{
    int array_count = input.numel();
    int block_dim = 1024;
    // int elements_per_block = 2*block_dim;
    int elements_per_block = COARSE_FACTOR*block_dim;
    int grid_dim = (array_count + elements_per_block - 1)/(elements_per_block);
    // ReduceKernel2<<<grid_dim, block_dim>>>(array_count, input.data_ptr<f32>(), output.data_ptr<f32>());
    // ReduceKernel3<<<grid_dim, block_dim>>>(array_count, input.data_ptr<f32>(), output.data_ptr<f32>());
    // ReduceKernel4<<<grid_dim, block_dim>>>(array_count, input.data_ptr<f32>(), output.data_ptr<f32>());
    ReduceKernel5<<<grid_dim, block_dim, block_dim*sizeof(f32)>>>(array_count, input.data_ptr<f32>(), output.data_ptr<f32>());
    CUDACheck(cudaDeviceSynchronize());
}
#else

#include <cuda_profiler_api.h>
#include <thrust/reduce.h>

void Reduce1(u32 count, f32 *input)
{
    assert(count <= 2048);

    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel1<<<grid_dim, block_dim>>>(input);
}

void Reduce2(u32 count, f32 *input, f32 *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel2<<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

void Reduce3(u32 count, f32 *input, f32 *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel3<<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

void Reduce4(u32 count, f32 *input, f32 *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = COARSE_FACTOR*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel4<<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

void Reduce5(u32 count, f32 *input, f32 *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = COARSE_FACTOR*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel5<<<grid_dim, block_dim, block_dim*sizeof(f32), stream>>>(count, input, output);
}

void ThrustReduce(u32 count, f32 *input, f32 *output, cudaStream_t stream)
{
    thrust::reduce_into(thrust::cuda::par.on(stream), input, input + count, output, 0.f);
}

typedef void (*ReduceFn)(u32, f32 *, f32 *, cudaStream_t);

void Benchmark(ReduceFn Reduce, f64 peak_gbps, f64 peak_gflops, const char *file_name)
{
    int warmup_count = 3;
    int rep_count = 20;

    FILE *file = fopen(file_name, "wb");
    assert(file);

    fwrite(&peak_gbps, sizeof(f64), 1, file);
    fwrite(&peak_gflops, sizeof(f64), 1, file);

    for (u32 exp = 1; exp <= 30; ++exp)
    {
        u64 array_count = 1 << exp;

        {
            f32 *input = (f32 *)malloc(array_count*sizeof(f32));
            for (u64 i = 0; i < array_count; ++i)
                input[i] = 1.f;

            f32 *d_input = 0;
            CUDACheck(cudaMalloc(&d_input, array_count*sizeof(f32)));
            CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));

            f32 *d_output = 0;
            CUDACheck(cudaMalloc(&d_output, sizeof(f32)));
            CUDACheck(cudaMemset(d_output, 0, sizeof(f32)));

            cudaEvent_t start_event, stop_event;
            CUDACheck(cudaEventCreate(&start_event));
            CUDACheck(cudaEventCreate(&stop_event));

            cudaStream_t stream;
            CUDACheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            // Warmups
            for (int i = 0; i < warmup_count; ++i)
            {
                CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemset(d_output, 0, sizeof(f32)));
                Reduce(array_count, d_input, d_output, stream);
            }

            f64 ms = 0;
            for (int i = 0; i < rep_count; ++i)
            {
                CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(f32), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemset(d_output, 0, sizeof(f32)));

                // CUDACheck(cudaProfilerStart());
                CUDACheck(cudaEventRecord(start_event, stream));
                Reduce(array_count, d_input, d_output, stream);
                CUDACheck(cudaEventRecord(stop_event, stream));
                CUDACheck(cudaEventSynchronize(stop_event));
                // CUDACheck(cudaProfilerStop());

                f32 curr_ms;
                CUDACheck(cudaEventElapsedTime(&curr_ms, start_event, stop_event));
                ms += curr_ms;
            }

            f32 out = 0.f;
            CUDACheck(cudaMemcpy(&out, d_output, sizeof(f32), cudaMemcpyDeviceToHost));

            bool is_correct = ((int)out == array_count);
            printf("[%s] Result (GPU): %f\n", is_correct ? "PASS" : "FAIL", out);

            ms /= rep_count;

            f64 bandwidth = (1000.0*(array_count*sizeof(f32)))/(ms*1024.0*1024.0*1024.0);

            fwrite(&array_count, sizeof(u64), 1, file);
            fwrite(&ms, sizeof(f64), 1, file);
            fwrite(&bandwidth, sizeof(f64), 1, file);

            printf("Elapsed (GPU): %f ms\n", ms);
            printf("Bandwidth: %f GB/s\n", bandwidth);

            CUDACheck(cudaStreamDestroy(stream));
            CUDACheck(cudaEventDestroy(stop_event));
            CUDACheck(cudaEventDestroy(start_event));
            CUDACheck(cudaFree(d_output));
            CUDACheck(cudaFree(d_input));
            free(input);
        }   
    }

    fclose(file);
}

int main(int argc, char **argv)
{
    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);

    printf("Peak bandwidth: %.2f GBPS\n", peak_gbps);
    printf("Peak throughput: %.2f GFLOPS\n", peak_gflops);
    printf("Peak arithmetic intensity: %.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    printf("Thrust version: %d.%d.%d (THRUST_VERSION: %d)\n", THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION, THRUST_VERSION);

    if (0)
    {
        BenchmarkCopy();
        printf("Done\n");
    }

    if (1)
    {
        if (argc != 2)
        {
            printf("Usage: %s <benchmark_index>\n", argv[0]);
            return 1;
        }
        
        int benchmark_index = atoi(argv[1]);
        printf("Benchmark index: %d\n", benchmark_index);

        ReduceFn reduce_fns[] =
        {
            0,
            0,
            Reduce2,
            Reduce3,
            Reduce4,
            Reduce5,
            ThrustReduce,
        };

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

        ReduceFn Reduce = reduce_fns[benchmark_index];
        const char *file_name = file_names[benchmark_index];
        Benchmark(Reduce, peak_gbps, peak_gflops, file_name);
    }

    return 0;
}
#endif