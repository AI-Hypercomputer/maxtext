import diffusers
import torch
import sys

def inspect_vae():
    try:
        # Load a standard VAE or the one used by DiT if we can identify it
        # For now, let's just check the structure of diffusers.models.AutoencoderKL
        vae = diffusers.models.AutoencoderKL()
        print("VAE Decoder:", vae.decoder)
        
        # Look for the last activation
        # We can print the module list
        for name, module in vae.decoder.named_modules():
            print(f"Module: {name}, Type: {type(module)}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_vae()
