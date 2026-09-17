import inspect
from diffusers.models.embeddings import PatchEmbed

print("Source of PatchEmbed.forward:")
print(inspect.getsource(PatchEmbed.forward))
