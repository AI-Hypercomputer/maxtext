import inspect
from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
norm1 = pipe.transformer.transformer_blocks[0].norm1

print("norm1 type:", type(norm1))
print("norm1 source:")
print(inspect.getsource(type(norm1)))
