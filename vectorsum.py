import math
import struct
from typing import Callable
from types import ModuleType

import torch
from torch.utils.cpp_extension import load_inline

import triton
import triton.language as tl

def reference_vectorsum(input: torch.Tensor, output: torch.Tensor) -> None:
    result = input.sum()
    output[0] = result

def triton_vectorsum_(input: torch.Tensor, output: torch.Tensor) -> None:
    @triton.jit
    def triton_vectorsum_kernel(input_ptr: tl.tensor, output_ptr: tl.tensor, N: int, BLOCK_DIM: tl.constexpr):
        segment_start = tl.program_id(0)*BLOCK_DIM
        offsets = segment_start + tl.arange(0, BLOCK_DIM)
        mask = offsets < N

        a: tl.tensor = tl.load(input_ptr + offsets, mask)
        result: float = tl.sum(a)
        
        # How does Triton know that only one thread should do this operation?
        tl.atomic_add(output_ptr, result)
    
    N = input.numel()
    grid = lambda metaparams: ((N + metaparams["BLOCK_DIM"] - 1)//metaparams["BLOCK_DIM"], 1, 1)
    triton_vectorsum_kernel[grid](input, output, input.numel(), BLOCK_DIM=1024)

triton_vectorsum: Callable[[torch.Tensor, torch.Tensor], None] = torch.compile(triton_vectorsum_, mode="reduce-overhead")

KERNEL_VERSION = 5

def load_cuda_vectorsum() -> ModuleType:
    vectorsum_module: ModuleType = None
    cpp_source = """
    #include <torch/extension.h>

    void Reduce(torch::Tensor input, torch::Tensor output);
    """
    with open("reduce.cu", "r") as f:
        cuda_source = f.read()
        vectorsum_module = load_inline(
            name="Reduce",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_include_paths=["."],
            extra_cuda_cflags=["-DCOMPILING_FROM_PYTORCH", f"-DREDUCE_KERNEL_VERSION={KERNEL_VERSION}"],
            functions=["Reduce"], verbose=True)

    assert vectorsum_module is not None
    return vectorsum_module

vectorsum_module: ModuleType = load_cuda_vectorsum()
cuda_vectorsum: Callable[[torch.Tensor, torch.Tensor], None] = vectorsum_module.Reduce

# TODO(achal)
def tinygrad_vectorsum():
    pass

def generate_input(size: int, seed: int|None) -> torch.Tensor:
    if seed is None:
        A = torch.ones(size, device="cuda", dtype=torch.float32).contiguous()
    else:
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
    return A

def benchmark_reduce(
    reduce: Callable[[torch.Tensor, torch.Tensor], None],
    array_count: int,
    seed: int|None,
) -> tuple[float, int]:
    input = generate_input(array_count, seed)
    output = torch.zeros(1, device="cuda", dtype=torch.float32)

    # Correctness check
    torch.cuda.synchronize()
    reduce(input, output)
    torch.cuda.synchronize()

    result_ref = torch.zeros(1, device="cuda", dtype=torch.float32)
    reference_vectorsum(input, result_ref)
    
    if torch.allclose(output, result_ref):
        print(f"[PASS] Result (GPU): {output}")
    else:
        print(f"[FAIL] Result (GPU): {output}")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)

        duration_ms: list[float] = []
        max_reps = 20
        max_benchmarking_ms = 120*1000

        ms_mean = 0.0
        rep = 0
        while rep < max_reps:
            input_clone = input.clone()

            start_event.record()
            reduce(input_clone, output)
            stop_event.record()
            stop_event.synchronize()
            
            duration_ms.append(start_event.elapsed_time(stop_event))

            if rep > 0:
                sample_count = rep + 1
                ms_mean = sum(duration_ms) / sample_count
                stddev = math.sqrt(sum((x - ms_mean) ** 2 for x in duration_ms) / (sample_count - 1))
                sem = stddev / math.sqrt(sample_count)
                if sem < 0.001 or rep >= max_reps or ms_mean*sample_count > max_benchmarking_ms:
                    break
                
            rep += 1

    return ms_mean, rep

def benchmark(
    reduce: Callable[[torch.Tensor, torch.Tensor], None],
    file_name: str
) -> None:
    element_counts = [2**i for i in range(1, 31)]

    with open(file_name, "wb") as file:
        metadata_present = False
        file.write(metadata_present.to_bytes(1, "little"))

        # Warmup
        benchmark_reduce(reduce, 1 << 18, None)

        for element_count in element_counts:
            ms, reps = benchmark_reduce(reduce, element_count, None)
            bandwidth = (1000.0*(element_count*torch.float32.itemsize))/(ms*1024.0*1024.0*1024.0)

            file.write(element_count.to_bytes(8, "little"))
            file.write(struct.pack("<d", ms))
            file.write(struct.pack("<d", bandwidth))

            print(f"Elapsed (GPU): {ms} ms [{reps}]")
            print(f"Bandwidth: {bandwidth} GBPS")

# benchmark(triton_vectorsum, f"bench_vectorsum_triton.bin")
# benchmark(vectorsum_module.Reduce, f"bench_vectorsum_{KERNEL_VERSION}.bin")

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

    input = generate_input(size, seed)

    # NOTE(achal): We need to zero-initialize the result because we are doing
    # read-modify-write on it.

    result_cuda = torch.zeros(1, device="cuda", dtype=torch.float32)
    cuda_vectorsum(input.clone(), result_cuda)
    
    result_triton = torch.zeros(1, device="cuda", dtype=torch.float32)
    triton_vectorsum(input, result_triton)
    
    result_ref = torch.zeros(1, device="cuda", dtype=torch.float32)
    reference_vectorsum(input, result_ref)

    if torch.allclose(result_ref, result_triton):
        print("Triton: Passed")
    else:
        print("Trion: Failed")

    if torch.allclose(result_ref, result_cuda):
        print("CUDA: Passed")
    else:
        print("CUDA: Failed")
 
if False:
    # https://github.com/gpu-mode/reference-kernels/blob/750868c61cd81fdcec8826a0cfcf4cb7fea064da/problems/pmpp_v2/vectorsum_py/task.yml#L31
    benchmarks: list[tuple[int, int]] = [
        # (size, seed)
        (1638400, 93246),
        (3276800, 6256),
        (6553600, 8841),
        (13107200, 6252),
        (26214400, 82135),
        (52428800, 12345),
    ]

    for benchmark in benchmarks:
        size, seed = benchmark
        print(f"===Test [size: {size}, seed: {seed}]===")

        vectorsum_module: ModuleType = load_cuda_vectorsum()

        ms, reps = benchmark_reduce(vectorsum_module.Reduce, size, seed)
        print(f"Elapsed: {ms} ms [{reps}]")
        bandwidth = (1000.0*(size*torch.float32.itemsize))/(ms*1024.0*1024.0*1024.0)
        print(f"Bandwidth: {bandwidth} GBPS")
    
