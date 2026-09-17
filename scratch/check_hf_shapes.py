import torch
from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
print("Transformer Config:")
print(pipe.transformer.config)
print("\nproj_out_2 shape:", pipe.transformer.proj_out_2.weight.shape)


print("\nVAE Config:")
print(pipe.vae.config)

print("\nScheduler Config:")
print(pipe.scheduler.config)


print("\nVAE Decoder Architecture:")
print(pipe.vae.decoder)
