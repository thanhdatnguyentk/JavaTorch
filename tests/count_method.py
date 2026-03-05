import torch

# Đếm số lượng hàm công khai trong module torch
torch_functions = [f for f in dir(torch) if callable(getattr(torch, f))]
print(f"Số lượng hàm trong torch: {len(torch_functions)}")

# Đếm số lượng phương thức trong class Tensor
tensor_methods = [m for m in dir(torch.Tensor) if callable(getattr(torch.Tensor, m))]
print(f"Số lượng phương thức trong torch.Tensor: {len(tensor_methods)}")