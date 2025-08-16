/*
    TODO(achal):
    * Make it such this file is runnable on its own i.e. without PyTorch.
*/

__global__ void vectorsum_kernel(uint64_t count, float *d_in, uint64_t stride)
{
    int lid = threadIdx.x;
    uint64_t gid = lid + blockIdx.x*blockDim.x;
    uint64_t index = gid*stride;

    // TODO(achal): Can we make this pass count calculation simpler given that we
    // will always be dealing with an integer number of threads per block?
    uint32_t pass_count = (uint32_t)ceilf(log2f((float)blockDim.x));

    // Each pass folds in the array into half
    for (int32_t pass = pass_count-1; pass >= 0; pass--)
    {
        uint32_t offset = 1 << pass;
        if ((lid < offset) && (index < count))
        {
            float a = d_in[index];
            float b = 0.f;
            if (index + offset*stride < count)
            b = d_in[index + offset*stride];
            d_in[index] = a + b;
        }
        __syncthreads();
    }
}

void vectorsum(torch::Tensor a)
{
    int array_count = a.numel();

    int N = array_count;
    int block_dim = 1024;
    int block_count = (N + block_dim - 1)/block_dim;
    uint64_t stride = 1;

    do
    {
        vectorsum_kernel<<<block_count, block_dim>>>(array_count, a.data_ptr<float>(), stride);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("ERROR: %s\n", cudaGetErrorString(err));
        }

        N = block_count;
        if (block_count == 1)
            block_count = 0;
        else
            block_count = (N + block_dim - 1)/block_dim;
        stride *= block_dim;
    } while (block_count > 0);
}

int main()
{
    return 0;
}