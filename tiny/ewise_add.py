from tinygrad import Tensor, UOp

def ewise_add_kernel(a:UOp, b:UOp, c:UOp) -> UOp:
    a,b,c = a.flatten(), b.flatten(), c.flatten()
    i = UOp.range(c.size, 0)
    return c[i].store(a[i]+b[i]).end(i).sink()

def ewise_add_test():
    a = Tensor.ones(16*1024, 16*1024).contiguous()
    b = Tensor.ones(16*1024, 16*1024).contiguous()
    c = Tensor.empty(16*1024, 16*1024)

    res:list[UOp] = UOp.custom_kernel(a.uop, b.uop, c.uop, fxn=ewise_add_kernel)
    t = Tensor(res[2]).realize()
    assert (t == (a+b).realize()).all().item()

ewise_add_test()