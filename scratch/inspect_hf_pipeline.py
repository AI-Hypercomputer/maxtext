import torch
from diffusers import DiTPipeline
import inspect

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

print("Pipeline Call Signature:")
print(inspect.signature(pipe.__call__))

print("\nPipeline Call Source:")
print(inspect.getsource(pipe.__call__))
