import torch
try:
    from diffusers import AutoencoderKL
    print("diffusers available")
except ImportError as e:
    print("diffusers NOT available:", e)
try:
    import transformers
    print("transformers available")
except ImportError as e:
    print("transformers NOT available:", e)
