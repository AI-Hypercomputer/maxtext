from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
print(f"Transformer pos_embed type: {type(pipe.transformer.pos_embed)}")
print(pipe.transformer.pos_embed)
