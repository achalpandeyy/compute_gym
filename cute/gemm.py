import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup

import torch

@cute.kernel
def gemm_kernel(m:cutlass.Constexpr, n:cutlass.Constexpr, k:cutlass.Constexpr, l:cutlass.Constexpr, mA_mkl:cute.Tensor, mB_nkl:cute.Tensor, mC_mnl:cute.Tensor):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    gA_mkl = cute.local_tile(input=mA_mkl, tiler=(m, k, l), coord=(None, None, None))
    gB_nkl = cute.local_tile(input=mB_nkl, tiler=(n, k, l), coord=(None, None, None))
    gC_mnl = cute.local_tile(input=mC_mnl, tiler=(m, n, l), coord=(None, None, None))

    k_tile_dim:cutlass.Constexpr = 16

    c = cute.full((1), 0.0, cutlass.Float32)
    for k_tile_index in range(k//k_tile_dim):
        tA = gA_mkl[tidy, None, bidz, 0, k_tile_index, 0]
        tB = gB_nkl[tidx, None, bidz, 0, k_tile_index, 0]

        print(f"type(tA): {type(tA)}")
        
        a = tA.load()
        b = tB.load()
        for index in range(k_tile_dim):
            c += a[index]*b[index] # TODO(achal): Is there no APL API for this?
        
    gC_mnl[tidy, tidx, bidz, 0, 0, 0] = c
    # print(f"type(tC): {type(tC)}")
    # tC.store(c)

@cute.jit
def gemm(m:cutlass.Constexpr, n:cutlass.Constexpr, k:cutlass.Constexpr, l:cutlass.Constexpr, a:cute.Pointer, b:cute.Pointer, c:cute.Pointer):
    a = cute.make_tensor(a, cute.make_layout((m, k, l), stride=(k, 1, m*k)))
    b = cute.make_tensor(b, cute.make_layout((n, k, l), stride=(k, 1, n*k)))
    c = cute.make_tensor(c, cute.make_layout((m, n, l), stride=(n, 1, m*n)))
    block_dim = (n, m, 1)
    grid_dim = ((n + block_dim[0] - 1)//block_dim[0], (m + block_dim[1] - 1)//block_dim[1], l)
    gemm_kernel(m, n, k, l, a, b, c).launch(grid=grid_dim, block=block_dim)

def cute_from_torch(data_ptr:int) -> cute.Pointer:
    # TODO(achal): Put the `assumed_align=16` back in and see if it make any
    # difference to our bottom line. Assuming (at least) 16-byte memory alignment
    # is supposed to help you with vectorlized loads, at least.
    return cute.runtime.make_ptr(cutlass.Float32, data_ptr, cute.AddressSpace.gmem)

def gemm_test():
    m, n, k, l = 8, 4, 16, 4

    io_dtype = cutlass.Float32
    acc_dtype = cutlass.Float32

    a_ref = torch.arange(l*m*k, dtype=torch.float32, device="cuda").reshape(l, m, k).permute(1, 2, 0) # torch.randn(l, m, k, dtype=torch.float32, device="cuda").permute(1, 2, 0)
    # print(a_ref[:, :, 0])
    b_ref = torch.arange(l*n*k, dtype=torch.float32, device="cuda").reshape(l, n, k).permute(1, 2, 0)
    print(b_ref[:, :, 0])
    c_ref = torch.randn(l, m, n, dtype=torch.float32, device="cuda").permute(1, 2, 0)

    a:cute.Pointer = cute_from_torch(0)
    b:cute.Pointer = cute_from_torch(0)
    c:cute.Pointer = cute_from_torch(0)
    compiled_kernel = cute.compile(gemm, m, n, k, l, a, b, c, options="--gpu-arch sm_89")

    a:cute.Pointer = cute_from_torch(a_ref.data_ptr())
    b:cute.Pointer = cute_from_torch(b_ref.data_ptr())
    c:cute.Pointer = cute_from_torch(c_ref.data_ptr())

    compiled_kernel(a, b, c)

    torch.cuda.synchronize()

gemm_test()