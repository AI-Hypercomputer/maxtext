from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

print("LabelEmbedder structure:")
print(pipe.transformer.label_embedder)

# Check number of embeddings
print(f"Number of embeddings: {pipe.transformer.label_embedder.embedding.num_embeddings}")
