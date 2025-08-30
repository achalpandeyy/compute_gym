import torch
from torch.utils.cpp_extension import load_inline

def reference_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)

def cuda_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    cpp_source = """
    #include <torch/extension.h>
    void matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C);
    """
    result: torch.Tensor | None = None
    with open("matmul.cu", "r") as f:
        cuda_source = f.read()
        matmul_module = load_inline(
            name="matmul",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=["-DCOMPILING_FROM_PYTORCH"],
            functions=["matmul"],
            verbose=True)
        
        result = torch.empty(A.shape[0], B.shape[1], device="cuda", dtype=torch.float16)
        matmul_module.matmul(A, B, result)
    
    assert result is not None
    return result


# https://github.com/gpu-mode/reference-kernels/blob/ee95b29fee216818ab497744265f2197a39b05f7/problems/pmpp_v2/matmul_py/task.yml#L23
test_cases: list[tuple[int, int, int, int]] = [
    (64, 64, 64, 53124),
    (128, 128, 128, 3321),
    (256, 256, 256, 1200),
    (32, 512, 32, 32523),
    (64, 1024, 64, 4327),
]

for test_case in test_cases:
    i, j, k, seed = test_case
    print(f"===Test [i: {i}, j: {j}, k: {k}, seed: {seed}]===")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    M = torch.empty(i, k, device="cuda", dtype=torch.float16)
    M.uniform_(0, 1, generator=gen)
    N = torch.empty(k, j, device="cuda", dtype=torch.float16)
    N.uniform_(0, 1, generator=gen)
    
    result_ref = reference_matmul(M, N)
    result_cuda = cuda_matmul(M, N)
    
    if torch.allclose(result_ref, result_cuda, rtol=1e-3, atol=1e-3):
        print("CUDA: Passed")
    else:
        print("CUDA: Failed")
    