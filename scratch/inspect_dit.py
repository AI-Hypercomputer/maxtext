import diffusers
import torch
import sys

def inspect_dit():
    try:
        # Load DiT model
        # We might need to know the exact class name, usually DiTTransformer2DModel
        # Let's try to find it in diffusers
        if hasattr(diffusers, 'DiTTransformer2DModel'):
            model = diffusers.DiTTransformer2DModel()
            print("DiT Model:", model)
        else:
            print("DiTTransformer2DModel not found in diffusers")
            # Try to list available models in diffusers that might be DiT
            print("Available models in diffusers:", [name for name in dir(diffusers.models) if 'DiT' in name])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_dit()
