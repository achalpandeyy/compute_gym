import torch
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

import tinygrad

def reference_vectoradd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    result = A + B
    return result

@triton.jit
def triton_vectoradd_kernel(a_ptr: tl.tensor, b_ptr: tl.tensor, c_ptr: tl.tensor, N: int, BLOCK_DIM: tl.constexpr):
    """
    We launch one block/program for each row of N elements.
    Each block, then, processes BLOCK_DIM elements of the row at a time.
    """
    row = tl.program_id(0)
    base_offset = row*N
    for i in range((N + BLOCK_DIM - 1)//BLOCK_DIM):
        # Load
        offsets = base_offset + i*BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        mask = offsets < base_offset + N
        a = tl.load(a_ptr + offsets, mask)
        b = tl.load(b_ptr + offsets, mask)
        # Process
        c: tl.tensor = a + b
        # Sore
        tl.store(c_ptr + offsets, c, mask)

def triton_vectoradd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a, device="cuda").contiguous()
    N = c.shape[0]
    grid = lambda metaparams: (N, 1, 1)
    triton_vectoradd_kernel[grid](a, b, c, N, BLOCK_DIM=128)
    return c

# TODO(achal)
def tinygrad_vectoradd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO(achal): Add the tinygrad env vars here
    # t1 = tinygrad.Tensor([1, 1, 1, 1, 1], dtype=tinygrad.dtypes.float16, device="CUDA")
    # t2 = tinygrad.Tensor([1, 1, 1, 1, 1], dtype=tinygrad.dtypes.float16, device="CUDA")
    # t3 = t1 + t2
    # print(t3.realize().tolist())

    a_tiny = tinygrad.Tensor.from_blob(a.data_ptr(), a.shape, dtype=tinygrad.dtype._from_torch_dtype(a.dtype), device="CUDA")
    b_tiny = tinygrad.Tensor.from_blob(b.data_ptr(), b.shape, dtype=tinygrad.dtype._from_torch_dtype(b.dtype), device="CUDA")
    torch.cuda.synchronize()
    c_tiny = a_tiny + b_tiny
    torch.cuda.synchronize()
    print(c_tiny.numpy().tolist())

    # return torch.from_numpy(c_tiny.numpy()).to("cuda").to(torch.float16)

def cuda_vectoradd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cuda_source = """
    template <typename scalar_t>
    __global__ void cuda_vectoradd_kernel(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ c, int N)
    {
        int block_dim = blockDim.x;

        int row = blockIdx.x;
        int base_offset = row*N;
        for (int col_start = 0; col_start < (N + block_dim - 1)/block_dim; ++col_start)
        {
            int col = base_offset + col_start*block_dim + threadIdx.x;
            if (col < base_offset + N)
            {
                c[col] = a[col] + b[col];
            }
        }
    }

    void cuda_vectoradd(torch::Tensor a, torch::Tensor b, torch::Tensor c)
    {
        int block_dim = 128;

        int row_count = a.sizes()[0];
        int col_count = a.sizes()[1];
        assert(row_count == col_count);

        int N = row_count;
        int block_count = col_count;
        
        // NOTE(achal): This might be host code but it is still CUDA syntax;
        // you cannot put it in cpp_source!
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "cuda_vectoradd_kernel", [&]{
            cuda_vectoradd_kernel<<<block_count, block_dim>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), N);
        });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf(\"ERROR: %s\\n\", cudaGetErrorString(err));
        }
    }
    """
    cpp_source = """
    #include <torch/extension.h>
    #include <stdio.h>

    void cuda_vectoradd(torch::Tensor a, torch::Tensor b, torch::Tensor c);
    """
    vectoradd_module = load_inline(name="cuda_vectoradd", cpp_sources=cpp_source, cuda_sources=cuda_source, functions=["cuda_vectoradd"], verbose=False)
    c = torch.empty_like(a)
    vectoradd_module.cuda_vectoradd(a, b, c)
    return c

# https://github.com/gpu-mode/reference-kernels/blob/ee95b29fee216818ab497744265f2197a39b05f7/problems/pmpp_v2/vectoradd_py/task.yml#L25
test_cases: list[tuple[int, int]] = [
    # (size, seed)
    (127, 4242),
    (128, 5236),
    (129, 1001),
    (256, 5531),
    (512, 9173),
]

generator = torch.Generator(device="cuda")

for test_case in test_cases:
    size, seed = test_case
    print(f"===Test [size: {size}, seed: {seed}]===")
    generator.manual_seed(seed)

    A = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=generator).contiguous()
    B = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=generator).contiguous()

    # A = torch.ones(size, size, device="cuda", dtype=torch.float16).contiguous()
    # B = torch.ones(size, size, device="cuda", dtype=torch.float16).contiguous()

    C_reference = reference_vectoradd(A, B)
    C_triton = triton_vectoradd(A, B)
    # C = tinygrad_vectoradd(A, B)
    C_cuda = cuda_vectoradd(A, B)

    # torch.testing.assert_close(C_triton, C_reference)
    # print("Triton: Passed")

    if torch.allclose(C_triton, C_reference):
        print("Triton: Passed")
    else:
        print("Triton: Failed")

    # if torch.allclose(C_tinygrad, C_reference):
    #     print("tinygrad: Passed")
    # else:
    #     print("tinygrad: Failed")

    if torch.allclose(C_cuda, C_reference):
        print("CUDA: Passed")
    else:
        print("CUDA: Failed")
