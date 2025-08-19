import torch

# - {"size": 32, "kernelsize": 4, "channels": 16, "batch": 1, "seed": 4242}
# - {"size": 32, "kernelsize": 4, "channels": 16, "batch": 2, "seed": 5236}
# - {"size": 64, "kernelsize": 4, "channels": 32, "batch": 1, "seed": 1001}
# - {"size": 64, "kernelsize": 8, "channels": 32, "batch": 2, "seed": 5531}
# - {"size": 128, "kernelsize": 8, "channels": 64, "batch": 1, "seed": 9173}

size = 32
kernelsize = 4
channels = 16
batch = 1
seed = 4242

gen = torch.Generator(device="cuda")
gen.manual_seed(seed)

# A = torch.randn(batch, channels, size, size, device="cuda", dtype=torch.float32, generator=gen).contiguous()
A = torch.ones(2, 1, 3, 3, device="cuda", dtype=torch.float32).contiguous()
kernel = torch.ones(1, 1, 2, 2, device="cuda", dtype=torch.float32).contiguous()
B = torch.nn.functional.conv2d(A, kernel, stride=1, padding=0)
breakpoint()
print(B)