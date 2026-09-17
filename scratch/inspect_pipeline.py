import inspect
from diffusers import DiTPipeline

# Try to find unpatchify in pipeline or model
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

# Let's check how pipeline handles output
print(inspect.getsource(pipe.__call__))
