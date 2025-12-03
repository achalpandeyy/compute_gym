import cutlass
import cutlass.cute as cute

import torch

@cute.kernel
def gemv_kernel(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, mA:cute.Tensor, mB:cute.Tensor, mC:cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    gA:cute.Tensor = cute.local_tile(input=mA, tiler=cute.slice_((M, 64, L), (None, None, 0)), coord=(None, None, None))
    gB:cute.Tensor = cute.local_tile(input=mB, tiler=cute.slice_((1, 64, L), (None, None, 0)), coord=(None, None, None))
    gC:cute.Tensor = cute.local_tile(input=mC, tiler=cute.slice_((M, 1, L), (None, None, 0)), coord=(None, None, None))

    tx = bidx*bdimx + tidx

    c = cute.full((1), 0.0, cutlass.Float32)
    for k_tile_index in range(K//64):
        tA = gA[tx, None, 0, k_tile_index, 0]
        tB = gB[0, None, 0, k_tile_index, 0]
        a = tA.load()
        b = tB.load()
        for idx in range(64):
            c += a[idx]*b[idx]

    tC = gC[tx, None, 0, 0, 0]
    tC.store(c)
    
@cute.jit
def gemv(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, a:cute.Pointer, b:cute.Pointer, c:cute.Pointer):
    a = cute.make_tensor(a, cute.make_layout((M, K, L), stride=(K, 1, M*K)))
    b = cute.make_tensor(b, cute.make_layout((1, K, L), stride=(K, 1, 1*K)))
    c = cute.make_tensor(c, cute.make_layout((M, 1, L), stride=(1, 1, M*1)))
    block_dim = 512
    grid_dim = (M + block_dim - 1)//block_dim
    gemv_kernel(M, K, L, a, b, c).launch(grid=(grid_dim, 1, 1), block=(512, 1, 1))

def gemv_test():
    M, K, L = 7168, 16384, 1
    A = torch.rand(M, K, L, dtype=torch.float32, device="cuda")
    B = torch.ones(1, K, L, dtype=torch.float32, device="cuda")
    C = torch.zeros(M, 1, L, dtype=torch.float32, device="cuda")
    
    C_expected = A[:, :, 0].sum(dim=1, keepdim=True)

    a:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    b:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    c:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    compiled_kernel = cute.compile(gemv, M, K, L, a, b, c, options="--gpu-arch sm_89")

    a:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, A.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, B.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    compiled_kernel(a, b, c)
    
    torch.testing.assert_close(C[:, :, 0], C_expected, rtol=1e-3, atol=1e-3)

gemv_test()