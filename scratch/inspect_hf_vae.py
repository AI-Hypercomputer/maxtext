from diffusers import AutoencoderKL
import inspect

model_id = "facebook/DiT-XL-2-256"
try:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    print("Imported AutoencoderKL")
    print("Source of vae._decode:")
    print(inspect.getsource(vae._decode))


except Exception as e:
    print(f"Error: {e}")
