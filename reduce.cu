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
    extern __shared__ T input_s[];

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
    ReduceKernel1<T><<<grid_dim, block_dim>>>(input);
}

template <typename T>
void Reduce2(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel2<T><<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

template <typename T>
void Reduce3(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = 2*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel3<T><<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

template <typename T>
void Reduce4(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = COARSE_FACTOR*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel4<T><<<grid_dim, block_dim, 0, stream>>>(count, input, output);
}

template <typename T>
void Reduce5(u64 count, T *input, T *output, cudaStream_t stream)
{
    int block_dim = 1024;
    int elements_per_block = COARSE_FACTOR*block_dim;
    int grid_dim = (count + elements_per_block - 1)/(elements_per_block);
    ReduceKernel5<T><<<grid_dim, block_dim, block_dim*sizeof(T), stream>>>(count, input, output);
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

template <typename T>
void ThrustReduce(u64 count, T *input, T *output, cudaStream_t stream)
{
    thrust::reduce_into(thrust::cuda::par.on(stream), input, input + count, output, T(0));
}

template <typename T>
f64 BenchmarkReduce(ReduceFn<T> Reduce, u64 array_count, int *reps)
{
    T *input = (T *)malloc(array_count*sizeof(T));
    for (u64 i = 0; i < array_count; ++i)
        input[i] = T(1);

    T *d_input = 0;
    CUDACheck(cudaMalloc(&d_input, array_count*sizeof(T)));
    CUDACheck(cudaMemcpy(d_input, input, array_count*sizeof(T), cudaMemcpyHostToDevice));

    T *d_output = 0;
    CUDACheck(cudaMalloc(&d_output, sizeof(T)));
    CUDACheck(cudaMemset(d_output, T(0), sizeof(T)));

    // NOTE(achal): Make sure we are not waiting on the default stream for anything.
    cudaStream_t stream;
    CUDACheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t start_event, stop_event;
    CUDACheck(cudaEventCreate(&start_event));
    CUDACheck(cudaEventCreate(&stop_event));

    // Correctness check
    {
        CUDACheck(cudaMemcpyAsync(d_input, input, array_count*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDACheck(cudaMemsetAsync(d_output, T(0), sizeof(T), stream));

        Reduce(array_count, d_input, d_output, stream);

        T out = T(0);
        CUDACheck(cudaMemcpyAsync(&out, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));

        CUDACheck(cudaStreamSynchronize(stream));

        bool is_correct = ((int)out == array_count);
        if (!is_correct)
        {
            assert(0);
            printf("[FAIL] Result (GPU): %f \tExpected: %f\n", out, (T)array_count);
        }
        else
        {
            // printf("[PASS] Result (GPU): %f \tExpected: %f\n", out, (T)array_count);
        }
    }

    f64 duration_ms[20];
    int max_reps = ArrayCount(duration_ms);
    int max_benchmarking_ms = 120*1000;

    f64 ms_mean = 0.0;
    int rep = 0;
    for (; rep < max_reps; ++rep)
    {
        CUDACheck(cudaMemcpyAsync(d_input, input, array_count*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDACheck(cudaMemsetAsync(d_output, T(0), sizeof(T), stream));

        CUDACheck(cudaEventRecord(start_event, stream));
        Reduce(array_count, d_input, d_output, stream);
        CUDACheck(cudaEventRecord(stop_event, stream));
        CUDACheck(cudaEventSynchronize(stop_event));
        
        f32 ms;
        CUDACheck(cudaEventElapsedTime(&ms, start_event, stop_event));
        duration_ms[rep] = (f64)ms;

        if (rep > 0)
        {
            int sample_count = rep + 1;
            for (int i = 0; i < sample_count; ++i)
                ms_mean += duration_ms[i];
            ms_mean /= sample_count;

            f64 stddev = 0.0;
            for (int i = 0; i < sample_count; ++i)
                stddev += (duration_ms[i] - ms_mean)*(duration_ms[i] - ms_mean);
            stddev /= rep - 1; // Bessel's correction
            stddev = sqrt(stddev);

            f64 sem = stddev/sqrt(sample_count);

            // NOTE(achal): Inspired by:
            // https://github.com/gpu-mode/reference-kernels/blob/750868c61cd81fdcec8826a0cfcf4cb7fea064da/problems/pmpp_v2/eval.py#L237
            // We only exit if any of these is true:
            // 1) Exceed the maximum number of repeats.
            // 2) Exceed the maximum benchmarking time.
            // 3) Sample mean is within 0.01% of population mean i.e. SEM < 0.001.
            if (sem < 0.001 || rep >= max_reps || ms_mean*sample_count > max_benchmarking_ms)
                break;
        }
    }

    if (reps)
        *reps = rep;

    CUDACheck(cudaEventDestroy(stop_event));
    CUDACheck(cudaEventDestroy(start_event));
    CUDACheck(cudaStreamDestroy(stream));
    CUDACheck(cudaFree(d_output));
    CUDACheck(cudaFree(d_input));
    free(input);

    return ms_mean;
}

template <typename T>
void Benchmark(ReduceFn<T> Reduce, f64 peak_gbps, f64 peak_gflops, const char *file_name)
{
    FILE *file = fopen(file_name, "wb");
    assert(file);

    b8 metadata_present = 1;
    fwrite(&metadata_present, sizeof(b8), 1, file);
    
    fwrite(&peak_gbps, sizeof(f64), 1, file);
    fwrite(&peak_gflops, sizeof(f64), 1, file);

    // Warmup
    (void)BenchmarkReduce<T>(Reduce, 1 << 18, 0);

    for (u32 exp = 1; exp <= 30; ++exp)
    {
        u64 array_count = 1 << exp;
        int reps = 0;
        f64 ms = BenchmarkReduce<T>(Reduce, array_count, &reps);
        f64 bandwidth = (1000.0*(array_count*sizeof(f32)))/(ms*1024.0*1024.0*1024.0);

        fwrite(&array_count, sizeof(u64), 1, file);
        fwrite(&ms, sizeof(f64), 1, file);
        fwrite(&bandwidth, sizeof(f64), 1, file);

        printf("Elapsed (GPU): %f ms [%d]\n", ms, reps);
        printf("Bandwidth: %f GB/s\n", bandwidth);  
    }

    fclose(file);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <benchmark_index>\n", argv[0]);
        return 1;
    }
    
    int benchmark_index = atoi(argv[1]);
    printf("Benchmark index: %d\n", benchmark_index);

    ReduceFn<f32> reduce_fns[] =
    {
        0,
        0,
        Reduce2<f32>,
        Reduce3<f32>,
        Reduce4<f32>,
        Reduce5<f32>,
        ThrustReduce<f32>,
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

    f64 peak_gbps = 0.0;
    f64 peak_gflops = 0.0;
    GetPeakMeasurements(&peak_gbps, &peak_gflops, true);

    printf("Bandwidth (Peak):\t%.2f GBPS\n", peak_gbps);
    printf("Throughput (Peak):\t%.2f GFLOPS\n", peak_gflops);
    printf("Arithmetic intensity (Peak):\t%.2f FLOPS/byte\n", peak_gflops/peak_gbps);

    const char *file_name = file_names[benchmark_index];
    Benchmark<f32>(reduce_fns[benchmark_index], peak_gbps, peak_gflops, file_name);

    return 0;
}
#endif