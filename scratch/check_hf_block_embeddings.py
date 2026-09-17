from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

# Check if blocks share the same AdaLayerNormZero or embeddings
block0 = pipe.transformer.transformer_blocks[0]
block1 = pipe.transformer.transformer_blocks[1]

print("Block 0 norm1:", block0.norm1)
print("Block 1 norm1:", block1.norm1)

print("Are norm1 identical objects?", block0.norm1 is block1.norm1)
print("Are emb identical objects?", block0.norm1.emb is block1.norm1.emb)
