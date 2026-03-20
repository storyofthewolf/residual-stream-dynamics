import torch
print(f"PyTorch version: {torch.version.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Quick test
if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000, device="mps")
    y = torch.randn(1000, 1000, device="mps")
    z = torch.matmul(x, y)
    print("MPS GPU acceleration working ✓")
