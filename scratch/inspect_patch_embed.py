import inspect
from diffusers.models.embeddings import PatchEmbed

print(inspect.getsource(PatchEmbed.forward))
