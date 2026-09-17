from diffusers import DiTPipeline
import torch

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

# Check if embedding weights are identical across blocks
w0 = pipe.transformer.transformer_blocks[0].norm1.emb.class_embedder.embedding_table.weight
w1 = pipe.transformer.transformer_blocks[1].norm1.emb.class_embedder.embedding_table.weight

print("Block 0 vs Block 1 class embedding weight identical?", torch.equal(w0, w1))

# Check all blocks
weights = [block.norm1.emb.class_embedder.embedding_table.weight for block in pipe.transformer.transformer_blocks]

# Are they all identical to first one?
all_identical = all(torch.equal(w, weights[0]) for w in weights)
print(f"Are all blocks' class embedding weights identical? {all_identical}")

# Do they have different values?
# print(w0[:5, :5])
# print(w1[:5, :5])
