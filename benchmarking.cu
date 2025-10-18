template <typename T>
struct Data;

template <typename T, u32 elements_per_block>
static Data<T> *CreateData(Arena *arena, u64 count, cudaStream_t stream);

template <typename T>
static void DestroyData(Data<T> *data);

template <typename T>
static b32 ValidateGPUOutput(Arena *arena, Data<T> *data);

template <typename T, u32 block_dim, u32 coarse_factor>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream);

__global__ void CopyKernel(u64 count, f32 *dst, f32 *src)
{
    u64 index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < count)
        dst[index] = src[index];
}

static void FlushL2Cache()
{
    f32 *d_src, *d_dst;
    u64 element_count = (256*1024*1024)/sizeof(*d_src);
    CUDACheck(cudaMalloc(&d_src, element_count*sizeof(*d_src)));
    CUDACheck(cudaMalloc(&d_dst, element_count*sizeof(*d_dst)));
    int block_dim = 1024;
    int grid_dim = (element_count + block_dim - 1)/block_dim;
    CUDACheck((CopyKernel<<<grid_dim, block_dim>>>(element_count, d_dst, d_src)));
    CUDACheck(cudaFree(d_src));
    CUDACheck(cudaFree(d_dst));
}

// NOTE(achal): Inspired by:
// https://github.com/gpu-mode/reference-kernels/blob/750868c61cd81fdcec8826a0cfcf4cb7fea064da/problems/pmpp_v2/eval.py#L237
template <typename T, u32 block_dim, u32 coarse_factor>
void Benchmark(f64 peak_gbps, f64 peak_gflops, const char *file_name)
{
#if BUILD_DEBUG
    printf("Warning: Benchmarking in debug mode. Results will be inaccurate.\n");
#endif

    Assert(file_name);
    FILE *file = fopen(file_name, "wb");

    if (file)
    {
        b8 metadata_present = 1;
        fwrite(&metadata_present, sizeof(b8), 1, file);
        
        fwrite(&peak_gbps, sizeof(f64), 1, file);
        fwrite(&peak_gflops, sizeof(f64), 1, file);
    }
    
    // Warmup
    {
        Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

        Data<T> *data = CreateData<T, block_dim*coarse_factor>(scratch.arena, 1 << 18, 0);
        FunctionToBenchmark<T, block_dim, coarse_factor>(data, 0);
        DestroyData<T>(data);
        
        ScratchEnd(&scratch);
    }

    for (u32 exp = 1; exp <= 30; ++exp)
    {
        u64 array_count = 1 << exp;

        // Correctness check
        b32 correct = 1;
        {
            Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));
            Data<T> *data = CreateData<T, block_dim*coarse_factor>(scratch.arena, array_count, 0);

            FunctionToBenchmark<T, block_dim, coarse_factor>(data, 0);

            if (!ValidateGPUOutput<T>(scratch.arena, data))
            {
                // assert(0);
                printf("Failed for %llu elements, skipping benchmark\n", array_count);
                correct = 0;
            }

            DestroyData<T>(data);
            ScratchEnd(&scratch);
        }

        // Benchmark, if correct
        if (correct)
        {
            Scratch scratch = ScratchBegin(GetScratchArena(GigaBytes(10)));

            // NOTE(achal): Make sure we are not waiting on the default stream for anything.
            cudaStream_t stream;
            CUDACheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            cudaEvent_t start_event, stop_event;
            CUDACheck(cudaEventCreate(&start_event));
            CUDACheck(cudaEventCreate(&stop_event));

            int max_reps = 20;
            int max_benchmarking_ms = 120*1000;
            
            f64 *duration_ms = PushArray(scratch.arena, f64, max_reps);
            f64 ms_mean = 0.0;

            int rep = 0;
            for (; rep < max_reps; ++rep)
            {
                {
                    Scratch data_scratch = ScratchBegin(scratch.arena);
                    Data<T> *data = CreateData<T, block_dim*coarse_factor>(data_scratch.arena, array_count, stream);
                    
                    FlushL2Cache();
                    CUDACheck(cudaEventRecord(start_event, stream));
                    FunctionToBenchmark<T, block_dim, coarse_factor>(data, stream);
                    CUDACheck(cudaEventRecord(stop_event, stream));
                    CUDACheck(cudaEventSynchronize(stop_event));
                    
                    DestroyData<T>(data);
                    ScratchEnd(&data_scratch);                   
                }
                
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

                    // We only exit if any of these is true:
                    // 1) Exceed the maximum number of repeats.
                    // 2) Exceed the maximum benchmarking time.
                    // 3) Sample mean is within 0.01% of population mean i.e. SEM < 0.001.
                    if (sem < 0.001 || rep >= max_reps || ms_mean*sample_count > max_benchmarking_ms)
                        break;
                }
            }

            f64 bandwidth = (1000.0*(array_count*sizeof(T)))/(ms_mean*1024.0*1024.0*1024.0);

            if (file)
            {
                fwrite(&array_count, sizeof(u64), 1, file);
                fwrite(&ms_mean, sizeof(f64), 1, file);
                fwrite(&bandwidth, sizeof(f64), 1, file);
            }

            printf("Array count: %llu\n", array_count);
            printf("Elapsed (GPU): %f ms [%d]\n", ms_mean, rep);
            printf("Bandwidth: %f GB/s\n", bandwidth);

            CUDACheck(cudaEventDestroy(stop_event));
            CUDACheck(cudaEventDestroy(start_event));
            CUDACheck(cudaStreamDestroy(stream));

            ScratchEnd(&scratch);
        }          
    }

    fclose(file);
}