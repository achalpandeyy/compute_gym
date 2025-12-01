import cutlass.cute as cute
import torch

@cute.kernel
def ewise_add_kernel(M, N, gX: cute.Tensor, gY: cute.Tensor, gZ: cute.Tensor):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    tx = bdimx*bidx + tidx
    ty = bdimy*bidy + tidy
    if tx < N and ty < M:
        gZ[ty, tx] = gX[ty, tx] + gY[ty, tx]

@cute.jit
def ewise_add(M, N, X: cute.Tensor, Y: cute.Tensor, Z: cute.Tensor):
    block_dim = (32, 32, 1)
    grid_dim = (
        (N + block_dim[0] - 1)//block_dim[0],
        (M + block_dim[1] - 1)//block_dim[1],
        1,
    )
    ewise_add_kernel(M, N, X, Y, Z).launch(grid=grid_dim, block=block_dim)

def ewise_add_test():
    M = 16*1024
    N = 16*1024
    X = torch.ones(M, N, dtype=torch.float32, device="cuda")
    Y = torch.ones(M, N, dtype=torch.float32, device="cuda")
    Z = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    ewise_add(M, N, *[cute.runtime.from_dlpack(t) for t in [X, Y, Z]])
    torch.testing.assert_close(Z, X + Y)

ewise_add_test()