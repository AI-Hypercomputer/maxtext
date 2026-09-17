import inspect
import diffusers

# Find DiT related models
for name in diffusers.__dict__.keys():
    if "DiT" in name:
        print(f"Found {name}")

# Assuming it is DiTTransformer2DModel or similar
try:
    from diffusers import DiTTransformer2DModel
    print("Imported DiTTransformer2DModel")
    print("Source of forward:")
    print(inspect.getsource(DiTTransformer2DModel.forward))
except Exception as e:
    print(f"Error: {e}")
