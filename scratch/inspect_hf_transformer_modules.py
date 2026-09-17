from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
print(pipe.transformer)
