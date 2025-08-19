import torch
from torch.utils.cpp_extension import load_inline

def reference_conv2d(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv2d(input_tensor, kernel, stride=1, padding=0)

def cuda_conv2d(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    cpp_source = """
    #include <torch/extension.h>
    void conv2d(torch::Tensor input_tensor, torch::Tensor kernel, torch::Tensor output_tensor);
    """
    result = None
    with open("conv2d.cu", "r") as f:
        cuda_source = f.read()
        conv2d_module = load_inline(
            name="conv2d",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=["-DCOMPILING_FROM_PYTORCH"],
            functions=["conv2d"],
            verbose=True)

        H_out = input_tensor.shape[2] - kernel.shape[2] + 1
        W_out = input_tensor.shape[3] - kernel.shape[3] + 1
        result = torch.empty(
            input_tensor.shape[0],
            kernel.shape[0],
            H_out,
            W_out,
            device="cuda").contiguous()
        conv2d_module.conv2d(input_tensor, kernel, result)
    
    assert result is not None
    return result

# https://github.com/gpu-mode/reference-kernels/blob/ee95b29fee216818ab497744265f2197a39b05f7/problems/pmpp_v2/conv2d_py/task.yml#L31
test_cases: list[tuple[int, int, int, int, int]] = [
    (32, 4, 16, 1, 4242),
    (32, 4, 16, 2, 5236),
    (64, 4, 32, 1, 1001),
    (64, 8, 32, 2, 5531),
    (128, 8, 64, 1, 9173),
]

for test_case in test_cases:
    size, kernelsize, channels, batch, seed = test_case
    print(f"===Test [size: {size}, kernelsize: {kernelsize}, channels: {channels}, batch: {batch}, seed: {seed}]===")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    input_tensor = torch.randn(
        batch,
        channels,
        size,
        size,
        device="cuda",
        dtype=torch.float32,
        generator=gen).contiguous()

    kernel = torch.randn(
        channels,
        channels,
        kernelsize,
        kernelsize,
        device="cuda",
        dtype=torch.float32,
        generator=gen).contiguous()
    
    result_ref: torch.Tensor = reference_conv2d(input_tensor, kernel)
    # result_triton = triton_conv2d(input_tensor, kernel)
    result_cuda: torch.Tensor = cuda_conv2d(input_tensor, kernel)

    if torch.allclose(result_ref, result_cuda, rtol=1e-3, atol=1e-3):
        print("CUDA: Passed")
    else:
        print("CUDA: Failed")