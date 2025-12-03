import cutlass
import cutlass.cute as cute

import numpy as np
import torch

@cute.jit
def explore_local_tile_(A: cute.Tensor):
    print(f"Original layout: {A.layout}")
    tiled = cute.local_tile(A, (2, 3), (None, None))
    print(f"After local_tile: {tiled.layout}")
    
    tile00 = tiled[None, None, 0, 0]
    print(tile00.layout)
    cute.print_tensor(tile00)

def explore_local_tile():
    A = np.arange(24).reshape(4, 6)
    print(f"Original A:\n{A}")
    explore_local_tile_(cute.runtime.from_dlpack(A))
# explore_local_tile()

@cute.jit
def explore_slice():
    A = np.arange(24).reshape(4, 6)
    print(f"Original A:\n{A}")
    A = cute.runtime.from_dlpack(A)

    tile_all = cute.slice_((2, 2, 2), (None, None, None))
    print(tile_all)

    tile_some = cute.slice_((2, 2, 2), (None, 0, None))
    print(tile_some)
# explore_slice()

@cute.kernel
def kernel(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, mA:cute.Tensor, mB:cute.Tensor, mC:cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    gA:cute.Tensor = cute.local_tile(input=mA, tiler=cute.slice_((M, K, L), (None, None, 0)), coord=(None, None, None))
    gB:cute.Tensor = cute.local_tile(input=mB, tiler=cute.slice_((1, K, L), (None, None, 0)), coord=(None, None, None))
    gC:cute.Tensor = cute.local_tile(input=mC, tiler=cute.slice_((M, 1, L), (None, None, 0)), coord=(None, None, None))

    tA = gA[tidx, None, bidx, 0, 0]
    tB = gB[None, None, 0, 0, 0]
    tC = gC[tidx, None, bidx, 0, 0]

    a = tA.load()
    b = tB.load()
    c = cute.full((1), 0.0, cutlass.Float32)
    
    for idx in range(K):
        c += a[idx]*b[idx]
    tC.store(c)
    
@cute.jit
def host(M:cutlass.Constexpr, K:cutlass.Constexpr, L:cutlass.Constexpr, a:cute.Pointer, b:cute.Pointer, c:cute.Pointer):
    a = cute.make_tensor(a, cute.make_layout((M, K, L), stride=(K, 1, M*K)))
    b = cute.make_tensor(b, cute.make_layout((1, K, L), stride=(K, 1, 1*K)))
    c = cute.make_tensor(c, cute.make_layout((M, 1, L), stride=(1, 1, M*1)))
    kernel(M, K, L, a, b, c).launch(grid=(1, 1, 1), block=(M, 1, 1))

def gemv():
    # M, K, L = 256, 256, 1
    M, K, L = 8, 12, 1
    A = torch.arange(M*K*L, dtype=torch.float32, device="cuda").reshape(M, K, L)
    B = torch.ones(1, K, L, dtype=torch.float32, device="cuda")
    C = torch.zeros(M, 1, L, dtype=torch.float32, device="cuda")
    
    C_expected = A[:, :, 0].sum(dim=1, keepdim=True)

    a:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    b:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    c:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    compiled_kernel = cute.compile(host, M, K, L, a, b, c, options="--gpu-arch sm_89")

    a:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, A.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, B.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c:cute.Pointer = cute.runtime.make_ptr(cutlass.Float32, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    compiled_kernel(a, b, c)
    
    torch.testing.assert_close(C[:, :, 0], C_expected)

gemv()