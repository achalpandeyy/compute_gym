import cutlass
import cutlass.cute as cute

import torch

@cute.kernel
def gemv_kernel(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, sfA:cute.Tensor, A:cute.Tensor, sfB:cute.Tensor, B:cute.Tensor, C:cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, bidz = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    tx = bidx*bdimx + tidx

    K_TILE_DIM:cutlass.Constexpr = 64

    sfA_tiles: cute.Tensor = cute.local_tile(input=sfA, tiler=(M, K_TILE_DIM//16, L), coord=(None, None, None))
    A_tiles:cute.Tensor = cute.local_tile(input=A, tiler=(M, K_TILE_DIM, L), coord=(None, None, None))

    sfB_tiles: cute.Tensor = cute.local_tile(input=sfB, tiler=(1, K_TILE_DIM//16, L), coord=(None, None, None))
    B_tiles:cute.Tensor = cute.local_tile(input=B, tiler=(1, K_TILE_DIM, L), coord=(None, None, None))

    C_tiles:cute.Tensor = cute.local_tile(input=C, tiler=(M,  1, L), coord=(None, None, None))

    c = cute.full((1), 0.0, cutlass.Float32)
    for k_tile_index in range(K//K_TILE_DIM):
        tsfA = sfA_tiles[tx, None, bidz, 0, k_tile_index, 0]
        tA = A_tiles[tx, None, bidz, 0, k_tile_index, 0]
        
        tsfB = sfB_tiles[0, None, bidz, 0, k_tile_index, 0]
        tB = B_tiles[0, None, bidz, 0, k_tile_index, 0]

        sfa = tsfA.load()
        a = tA.load()
        sfb = tsfB.load()
        b = tB.load()
        for idx in range(K_TILE_DIM):
            c += (sfa[idx//16]*a[idx])*(sfb[idx//16]*b[idx])

    tC = C_tiles[tx, None, bidz, 0, 0, 0]
    tC.store(c)
    
@cute.jit
def gemv(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, sfA:cute.Pointer, a:cute.Pointer, sfB:cute.Pointer, b:cute.Pointer, c:cute.Pointer):
    sfA = cute.make_tensor(sfA, cute.make_layout((M, K//16, L), stride=(K//16, 1, M*K//16)))
    a = cute.make_tensor(a, cute.make_layout((M, K, L), stride=(K, 1, M*K)))

    sfB = cute.make_tensor(sfB, cute.make_layout((1, K//16, L), stride=(K//16, 1, K//16)))
    b = cute.make_tensor(b, cute.make_layout((1, K, L), stride=(K, 1, 1*K)))
    
    c = cute.make_tensor(c, cute.make_layout((M, 1, L), stride=(1, 1, 1*M)))
    block_dim = (512, 1, 1)
    grid_dim = ((M + block_dim[0] - 1)//block_dim[0], 1, L)
    gemv_kernel(M, K, L, sfA, a, sfB, b, c).launch(grid=grid_dim, block=block_dim)

def cute_from_torch(data_ptr:int) -> cute.Pointer:
    # TODO(achal): Put the `assumed_align=16` back in and see if it make any
    # difference to our bottom line. Assuming (at least) 16-byte memory alignment
    # is supposed to help you with vectorlized loads, at least.
    return cute.runtime.make_ptr(cutlass.Float32, data_ptr, cute.AddressSpace.gmem)

def gemv_test():
    M, K, L = 4096, 7168, 8
    
    # All of these are K-major layout
    
    sfA_torch = torch.full((L, M, K//16), 0.2, dtype=torch.float32, device="cuda").permute(1, 2, 0)
    A = torch.rand(L, M, K,  dtype=torch.float32, device="cuda").permute(1, 2, 0)
    
    sfB_torch = torch.full((L, 1, K//16), 0.2, dtype=torch.float32, device="cuda").permute(1, 2, 0)
    B = torch.ones(L, 1, K,  dtype=torch.float32, device="cuda").permute(1, 2, 0)
    
    C = torch.zeros(L, M, 1, dtype=torch.float32, device="cuda").permute(1, 2, 0)
    
    C_expected = 0.2*0.2*A.sum(dim=1, keepdim=True)

    sfA:cute.Pointer = cute_from_torch(0)
    a:cute.Pointer = cute_from_torch(0)

    sfB:cute.Pointer = cute_from_torch(0)
    b:cute.Pointer = cute_from_torch(0)

    c:cute.Pointer = cute_from_torch(0)
    compiled_kernel = cute.compile(gemv, M, K, L, sfA, a, sfB,b, c, options="--gpu-arch sm_89")

    sfA:cute.Pointer = cute_from_torch(sfA_torch.data_ptr())
    a:cute.Pointer = cute_from_torch(A.data_ptr())

    sfB:cute.Pointer = cute_from_torch(sfB_torch.data_ptr())
    b:cute.Pointer = cute_from_torch(B.data_ptr())
    
    c:cute.Pointer = cute_from_torch(C.data_ptr())
    compiled_kernel(sfA, a, sfB, b, c)

    torch.testing.assert_close(C, C_expected, rtol=1e-3, atol=1e-3)

gemv_test()