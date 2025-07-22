import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check which device a tensor would be on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Check current GPU name (if any)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Current CUDA Device:", torch.cuda.current_device())
