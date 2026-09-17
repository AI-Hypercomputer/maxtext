from diffusers.models.embeddings import PatchEmbed

# Instantiate or check config
import torch
import inspect

# Let's check init
print(inspect.getsource(PatchEmbed.__init__))
