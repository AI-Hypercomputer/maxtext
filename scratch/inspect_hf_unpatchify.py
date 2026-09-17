import inspect
from diffusers import DiTPipeline

# Try to find unpatchify method in pipeline or model
try:
    print("Inspecting DiTPipeline.__call__...")
    print(inspect.getsource(DiTPipeline.__call__))
    
    # Or look at the unpatchify function if it exists as utility
    # Search for unpatchify in diffusers
    import diffusers
    for name, obj in inspect.getmembers(diffusers):
        if "unpatchify" in name.lower():
            print(f"Found {name} in diffusers")
            
except Exception as e:
    print(f"Error: {e}")
