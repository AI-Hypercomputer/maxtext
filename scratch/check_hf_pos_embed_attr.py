from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
print("pos_embed attribute in PatchEmbed:", pipe.transformer.pos_embed.pos_embed)
pe = pipe.transformer.pos_embed.pos_embed
print(f"HF pos_embed stats: min={pe.min():.4f}, max={pe.max():.4f}, std={pe.std():.4f}")
