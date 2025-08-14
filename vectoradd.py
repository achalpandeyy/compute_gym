import torch
import triton
import triton.language as tl

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
    triton_vectoradd_kernel[grid](a, b, c, c.numel(), BLOCK_DIM=128)

    return c

def tinygrad_vectoradd():
    # TODO(achal)
    pass

def cuda_vectoradd():
    # TODO(achal)
    pass

# Match each of them to the reference output
C_reference = reference_vectoradd(A, B)
C_triton = triton_vectoradd(A, B)
torch.testing.assert_close(C_triton, C_reference)
print("Triton: Passed")