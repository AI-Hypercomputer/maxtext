import inspect
from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
print(pipe.transformer.transformer_blocks[0])
print(inspect.getsource(pipe.transformer.transformer_blocks[0].__class__))
