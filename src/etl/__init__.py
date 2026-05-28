import torch 
import os 

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device in use: {device}")

os.environ["device_torch"] = device