import torch
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

import tinygrad

# - {"size": 127, "seed": 4242}
# - {"size": 128, "seed": 5236}
# - {"size": 129, "seed": 1001}
# - {"size": 256, "seed": 5531}
# - {"size": 512, "seed": 9173}

generator = torch.Generator(device="cuda")

size = 127
generator.manual_seed(4242)

A = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=generator).contiguous()
B = torch.randn(size, size, device="cuda", dtype=torch.float16, generator=generator).contiguous()

# A = torch.ones(size, size, device="cuda", dtype=torch.float16).contiguous()
# B = torch.ones(size, size, device="cuda", dtype=torch.float16).contiguous()

def reference_vectoradd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    result = A + B
    return result

@triton.jit
def triton_vectoradd_kernel(a_ptr: tl.tensor, b_ptr: tl.tensor, c_ptr: tl.tensor, element_count: int, BLOCK_DIM: tl.constexpr):
    # Load
    offsets = tl.arange(0, BLOCK_DIM)[:, None]*BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask = offsets < element_count
    a = tl.load(a_ptr + offsets, mask)
    b = tl.load(b_ptr + offsets, mask)

    # Process
    c: tl.tensor = a + b

    # Store
    tl.store(c_ptr + offsets, c, mask)

def triton_vectoradd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a, device="cuda").contiguous()

    grid = lambda metaparams: (1, 1, 1)
    # TODO(achal): Can I decide the BLOCK_DIM based on input element count?
    triton_vectoradd_kernel[grid](a, b, c, c.numel(), BLOCK_DIM=128)

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

# C = tinygrad_vectoradd(A, B)

def cuda_vectoradd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cuda_source = """
    template <typename scalar_t>
    __global__ void cuda_vectoradd_kernel(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ c, int N)
    {
        int row = threadIdx.x;
        if (row < N)
        {
            for (int col = 0; col < N; ++col)
            {
                if (col < N)
                {
                    int index = threadIdx.x*blockDim.x + col;
                    c[index] = a[index] + b[index];
                }
            }
        }
    }

    void cuda_vectoradd(torch::Tensor a, torch::Tensor b, torch::Tensor c)
    {
        int BLOCK_DIM = 128;
        int element_count = a.numel();
        int block_count = (element_count + BLOCK_DIM - 1)/element_count;
        
        // NOTE(achal): This might be host code but it is still CUDA syntax;
        // you cannot put it in cpp_source!
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "cuda_vectoradd_kernel", [&]{
            cuda_vectoradd_kernel<<<block_count, BLOCK_DIM>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), element_count);
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
    vectoradd_module = load_inline(name="cuda_vectoradd", cpp_sources=cpp_source, cuda_sources=cuda_source, functions=["cuda_vectoradd"], verbose=True)
    c = torch.empty_like(a)
    vectoradd_module.cuda_vectoradd(a, b, c)
    return c

# Match each of them to the reference output
C_reference = reference_vectoradd(A, B)
# C_triton = triton_vectoradd(A, B)
# torch.testing.assert_close(C_triton, C_reference)
# print("Triton: Passed")

C_cuda = cuda_vectoradd(A, B)
if torch.allclose(C_cuda, C_reference):
    print("CUDA: Passed")
else:
    print("CUDA: Failed")