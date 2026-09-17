import inspect
from diffusers.models.embeddings import TimestepEmbedding

print(inspect.getsource(TimestepEmbedding.__init__))
# Wait, DiT might use custom TimestepEmbedder
from diffusers.models.embeddings import get_timestep_embedding
print(inspect.getsource(get_timestep_embedding))
