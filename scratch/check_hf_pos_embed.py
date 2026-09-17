from diffusers import DiTPipeline
import torch

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

print("pos_embed type:", type(pipe.transformer.pos_embed))
print("pos_embed structure:")
print(pipe.transformer.pos_embed)

import inspect
print(inspect.getsource(pipe.transformer.pos_embed.__init__))
