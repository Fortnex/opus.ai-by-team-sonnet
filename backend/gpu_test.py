# Save as gpu_test.py
import torch
import os

print(torch.zeros(1).cuda())
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    # Create a test tensor on GPU
    test_tensor = torch.zeros(1).cuda()
    print(f"Test tensor device: {test_tensor.device}")
else:
    print("No GPU detected by PyTorch!")

# Check environment
print("\nEnvironment variables:")
for var in ['CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF']:
    print(f"{var}: {os.environ.get(var, 'Not set')}")