import cutlass
import cutlass.cute as cute

import torch

@cute.kernel
def gemv_kernel(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, A:cute.Tensor, B:cute.Tensor, C:cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, bidz = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    tx = bidx*bdimx + tidx

    K_TILE_DIM:cutlass.Constexpr = 64

    A_tiles:cute.Tensor = cute.local_tile(input=A, tiler=(M, K_TILE_DIM, L), coord=(None, None, None))
    B_tiles:cute.Tensor = cute.local_tile(input=B, tiler=(1, K_TILE_DIM, L), coord=(None, None, None))
    C_tiles:cute.Tensor = cute.local_tile(input=C, tiler=(M,  1, L), coord=(None, None, None))

    c = cute.full((1), 0.0, cutlass.Float32)
    for k_tile_index in range(K//K_TILE_DIM):
        tA = A_tiles[tx, None, bidz, 0, k_tile_index, 0]
        tB = B_tiles[0, None, bidz, 0, k_tile_index, 0]
        a = tA.load()
        b = tB.load()
        for idx in range(K_TILE_DIM):
            c += a[idx]*b[idx]

    tC = C_tiles[tx, None, bidz, 0, 0, 0]
    tC.store(c)
    
@cute.jit
def gemv(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, a:cute.Pointer, b:cute.Pointer, c:cute.Pointer):
    a = cute.make_tensor(a, cute.make_layout((M, K, L), stride=(K, 1, M*K)))
    b = cute.make_tensor(b, cute.make_layout((1, K, L), stride=(K, 1, 1*K)))
    c = cute.make_tensor(c, cute.make_layout((M, 1, L), stride=(1, 1, 1*M)))
    block_dim = (512, 1, 1)
    grid_dim = ((M + block_dim[0] - 1)//block_dim[0], 1, L)
    gemv_kernel(M, K, L, a, b, c).launch(grid=grid_dim, block=block_dim)

def gemv_test():
    M, K, L = 4096, 7168, 8
    # All of these are K-major layout
    A = torch.rand(L, M, K,  dtype=torch.float32, device="cuda").permute(1, 2, 0)
    B = torch.ones(L, 1, K,  dtype=torch.float32, device="cuda").permute(1, 2, 0)
    C = torch.zeros(L, M, 1, dtype=torch.float32, device="cuda").permute(1, 2, 0)
    
    C_expected = A.sum(dim=1, keepdim=True)

    a:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    b:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    c:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    compiled_kernel = cute.compile(gemv, M, K, L, a, b, c, options="--gpu-arch sm_89")

    a:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, A.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, B.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    compiled_kernel(a, b, c)

    torch.testing.assert_close(C, C_expected, rtol=1e-3, atol=1e-3)

gemv_test()