import torch
from torch.utils.cpp_extension import load_inline

def reference_vectorsum(a: torch.Tensor) -> float:
    result = a.to(torch.float64).sum().to(torch.float32)
    return result;

# TODO(achal)
def triton_vectorsum():
    pass

# TODO(achal)
def tinygrad_vectorsum():
    pass

def cuda_vectorsum(a: torch.Tensor) -> float:
    # NOTE(achal): The reduction algorithm works in place so
    # first copy the Tensor.
    a_copy = a.clone()

    cpp_source = """
    #include <torch/extension.h>

    void vectorsum(torch::Tensor a);
    """
    result = None
    with open("vectorsum.cu", "r") as f:
        cuda_source = f.read()
        vectorsum_module = load_inline(
            name="vectorsum",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=["-DCOMPILING_FROM_PYTORCH"],
            functions=["vectorsum"], verbose=True)
        
        vectorsum_module.vectorsum(a_copy)
        result = a_copy[0]
    
    assert result
    return result

# https://github.com/gpu-mode/reference-kernels/blob/ee95b29fee216818ab497744265f2197a39b05f7/problems/pmpp_v2/vectorsum_py/task.yml#L24
test_cases: list[tuple[int, int]] = [
    # (size, seed)
    (1023, 4242),
    (1024, 5236),
    (1025, 1001),
    (2048, 5531),
    (4096, 9173),
]

for test_case in test_cases:
    size, seed = test_case
    print(f"===Test [size: {size}, seed: {seed}]===")

    # https://github.com/gpu-mode/reference-kernels/blob/ee95b29fee216818ab497744265f2197a39b05f7/problems/pmpp_v2/vectorsum_py/reference.py#L21
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    # Generate base random data
    data = torch.randn(
        size, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()

    # Generate random offset and scale (using different seeds to avoid correlation)
    offset_gen = torch.Generator(device="cuda")
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device="cuda")
    scale_gen.manual_seed(seed + 2)

    # Generate random offset between -100 and 100
    offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
    # Generate random scale between 0.1 and 10
    scale = (torch.rand(1, device="cuda", generator=scale_gen) * 9.9 + 0.1).item()

    # Apply scale and offset
    A = (data * scale + offset).contiguous()
    
    result_ref: float = reference_vectorsum(A)
    # result_triton = triton_vectorsum(A)
    # result_tinygrad = tinygrad_vectorsum(A)
    result_cuda: float = cuda_vectorsum(A)

    if torch.allclose(result_ref, result_cuda):
        print("CUDA: Passed")
    else:
        print("CUDA: Failed")
