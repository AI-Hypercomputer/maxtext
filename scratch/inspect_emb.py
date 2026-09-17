import inspect
from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
emb = pipe.transformer.transformer_blocks[0].norm1.emb

print("emb type:", type(emb))
print("emb source:")
print(inspect.getsource(type(emb)))
