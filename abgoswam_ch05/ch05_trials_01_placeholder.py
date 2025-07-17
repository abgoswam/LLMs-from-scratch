from importlib.metadata import version
import torch
import torch.nn as nn

print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))