template <typename T>
struct Data;

template <typename T>
static void CreateData(Data<T> *data, cudaStream_t stream);

template <typename T>
static void DestroyData(Data<T> *data);

template <typename T>
static b32 ValidateGPUOutput(Data<T> *data);

template <typename T>
static void FunctionToBenchmark(Data<T> *data, cudaStream_t stream);

// NOTE(achal): Inspired by:
// https://github.com/gpu-mode/reference-kernels/blob/750868c61cd81fdcec8826a0cfcf4cb7fea064da/problems/pmpp_v2/eval.py#L237
template <typename T>
void Benchmark(f64 peak_gbps, f64 peak_gflops, const char *file_name)
{
    FILE *file = fopen(file_name, "wb");
    assert(file);

    b8 metadata_present = 1;
    fwrite(&metadata_present, sizeof(b8), 1, file);
    
    fwrite(&peak_gbps, sizeof(f64), 1, file);
    fwrite(&peak_gflops, sizeof(f64), 1, file);

    // Warmup
    {
        Data<T> data;
        data.count = 1 << 18;
        CreateData(&data, 0);
        FunctionToBenchmark(&data, 0);
        DestroyData(&data);
    }

    for (u32 exp = 1; exp <= 30; ++exp)
    {
        u64 array_count = 1 << exp;

        // Correctness check
        b32 correct = 1;
        {
            Data<T> data;
            data.count = array_count;
            CreateData(&data, 0);

            FunctionToBenchmark(&data, 0);

            if (!ValidateGPUOutput(&data))
            {
                // assert(0);
                printf("Failed for %llu elements, skipping benchmark\n", array_count);
                correct = 0;
            }

            DestroyData(&data);
        }

        // Benchmark, if correct
        if (correct)
        {
            // NOTE(achal): Make sure we are not waiting on the default stream for anything.
            cudaStream_t stream;
            CUDACheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            cudaEvent_t start_event, stop_event;
            CUDACheck(cudaEventCreate(&start_event));
            CUDACheck(cudaEventCreate(&stop_event));

            f64 duration_ms[20];
            int max_reps = ArrayCount(duration_ms);
            int max_benchmarking_ms = 120*1000;

            f64 ms_mean = 0.0;

            int rep = 0;
            for (; rep < max_reps; ++rep)
            {
                Data<T> data;
                data.count = array_count;
                CreateData(&data, stream);

                CUDACheck(cudaEventRecord(start_event, stream));
                FunctionToBenchmark(&data, stream);
                CUDACheck(cudaEventRecord(stop_event, stream));
                CUDACheck(cudaEventSynchronize(stop_event));
                
                f32 ms;
                CUDACheck(cudaEventElapsedTime(&ms, start_event, stop_event));
                duration_ms[rep] = (f64)ms;

                DestroyData(&data);

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

            fwrite(&array_count, sizeof(u64), 1, file);
            fwrite(&ms_mean, sizeof(f64), 1, file);
            fwrite(&bandwidth, sizeof(f64), 1, file);

            printf("Elapsed (GPU): %f ms [%d]\n", ms_mean, rep);
            printf("Bandwidth: %f GB/s\n", bandwidth);

            CUDACheck(cudaEventDestroy(stop_event));
            CUDACheck(cudaEventDestroy(start_event));
            CUDACheck(cudaStreamDestroy(stream));
        }          
    }

    fclose(file);
}