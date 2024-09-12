import torch
import time

def matrix_multiply(device, size):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    start = time.time()
    c = torch.matmul(a, b)
    end = time.time()
    
    return end - start

# マトリックスサイズ
sizes = [1000, 2000, 4000]

print("Device: ", "GPU" if torch.cuda.is_available() else "CPU")
print("PyTorch version: ", torch.__version__)
print("CUDA version: ", torch.version.cuda)

for size in sizes:
    print(f"\nMatrix size: {size}x{size}")
    
    # CPU計算
    cpu_time = matrix_multiply('cpu', size)
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU計算（利用可能な場合）
    if torch.cuda.is_available():
        gpu_time = matrix_multiply('cuda', size)
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"GPU is {cpu_time / gpu_time:.2f}x faster")
    else:
        print("GPU is not available")