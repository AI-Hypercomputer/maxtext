from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
for name, param in pipe.transformer.pos_embed.named_parameters():
    print(f"Param: {name}, Shape: {param.shape}")

print("\nAll parameters in pos_embed:")
for k, v in pipe.transformer.pos_embed.state_dict().items():
    print(f"State Dict Key: {k}, Shape: {v.shape}")
