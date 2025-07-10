from importlib.metadata import version
import torch

print("torch version:", version("torch"))

print(torch.cuda.is_available() )

print("done")