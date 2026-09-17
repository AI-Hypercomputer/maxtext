import os
import sys
from typing import Sequence
import jax
import jax.numpy as jnp
from flax import nnx
from absl import app
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from PIL import Image
from maxtext.layers.vae import Decoder

# Ensure we are in path if needed
# sys.path.append('src/')

from maxtext.configs import pyconfig
from maxtext.models.dit import DiT
from maxtext.schedulers.scheduling_ddim import DDIMScheduler
from maxtext.schedulers.scheduling_ddpm import DDPMScheduler
from maxtext.utils import maxtext_utils
from maxtext.utils import max_utils

def _validate_config(config):
  assert config.load_full_state_path == "", (
      "Sampler doesn't operate on full states! Convert to parameter checkpoint first."
  )

def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    print("Initializing config...")
    # Initialize config with argv, allowing overrides
    config = pyconfig.initialize(argv)
    _validate_config(config)
    
    jax.config.update("jax_use_shardy_partitioner", config.shardy)
    max_utils.print_system_information()
    
    # Setup Mesh
    devices_array = maxtext_utils.create_device_mesh(config)

    mesh = Mesh(devices_array, config.mesh_axes)
    print(f"Created mesh with axes: {config.mesh_axes}")
    replicated_sharding = NamedSharding(mesh, P())
    
    print("Instantiating DiT model...")
    rng = jax.random.PRNGKey(0)
    rng, rng_model = jax.random.split(rng)
    nnx_rngs = nnx.Rngs(rng_model)
    
    model = DiT(config, mesh, rngs=nnx_rngs)
    print("DiT model instantiated.")
    
    if config.load_parameters_path:
        from maxtext.common import checkpointing
        print(f"Loading parameters from {config.load_parameters_path}...", flush=True)
        _, params, _ = nnx.split(model, nnx.Param, ...)
        checkpointing.load_params_from_path(
            config.load_parameters_path,
            params,
            config.checkpoint_storage_concurrent_gb,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
        )
        # Update model parameters
        params = jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), params)
        nnx.update(model, params)
        print("Parameters loaded and replicated successfully.", flush=True)
    
    # Simple lookup for testing (ImageNet classes)
    word_to_id = {
        "white shark": 2,
        "umbrella": 879,
    }
    
    # Parse prompt as comma separated words
    prompt_words = [w.strip() for w in config.prompt.split(',')]
    class_ids = []
    for w in prompt_words:
        if w in word_to_id:
            class_ids.append(word_to_id[w])
        else:
            print(f"Warning: Unknown word '{w}', using default class 0")
            class_ids.append(0)
            
    class_labels = jnp.array(class_ids, dtype=jnp.int32)
    class_labels = jax.device_put(class_labels, replicated_sharding)
    batch_size = len(class_ids)
    
    # Instantiate Scheduler
    # Using DDIM as default for faster testing
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    
    out_channels = getattr(config, 'out_channels_for_vit', config.num_channels_for_vit)
    patch_dim = config.patch_size_for_vit * config.patch_size_for_vit * out_channels
    
    seq_len = config.max_target_length
    
    # Latent dimension (input to model) is based on in_channels
    in_patch_dim = config.patch_size_for_vit * config.patch_size_for_vit * config.num_channels_for_vit
    
    print(f"Sampling parameters: batch_size={batch_size}, seq_len={seq_len}, patch_dim={patch_dim}, in_patch_dim={in_patch_dim}")
    
    # Initial Noise
    rng, rng_noise = jax.random.split(rng)
    latents = jax.random.normal(rng_noise, (batch_size, seq_len, in_patch_dim))
    latents = jax.device_put(latents, replicated_sharding)
    
    # CFG settings
    guidance_scale = 4.0
    uncond_class = config.vocab_size # Using vocab_size as uncond label
    
    # Timesteps
    num_inference_steps = 50 # Example smaller steps
    timesteps = scheduler.set_timesteps(num_inference_steps)
    
    print(f"Starting sampling loop for {len(timesteps)} steps...")
    
    for i, t in enumerate(timesteps):
        if i % 10 == 0:
            print(f"Step {i}/{len(timesteps)}, Timestep {t}")
            
        # 1. CFG duplication
        latent_model_input = jnp.concatenate([latents] * 2)
        t_batch = jnp.array([t] * (batch_size * 2))
        t_batch = jax.device_put(t_batch, replicated_sharding)
        
        # Labels for CFG
        y_cond = class_labels
        y_uncond = jnp.array([uncond_class] * batch_size, dtype=jnp.int32)
        y_batch = jnp.concatenate([y_cond, y_uncond])
        y_batch = jax.device_put(y_batch, replicated_sharding)
        
        # 2. Predict Noise
        noise_pred_batch = model(latent_model_input, t_batch, y_batch)
        
        # 3. Perform Guidance
        noise_pred_cond, noise_pred_uncond = jnp.split(noise_pred_batch, 2, axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # noise_pred shape: [B, L, P_h * P_w * C_out]
        # Assuming layout is (P_h, P_w, C_out) from model output
        p = config.patch_size_for_vit
        in_channels = config.num_channels_for_vit
        c_out = getattr(config, 'out_channels_for_vit', in_channels)
        
        noise_pred_reshaped = noise_pred.reshape(batch_size, seq_len, c_out, p, p)
        
        # DiT might output extra channels (e.g. for learned variance)
        # We only take the first in_channels_for_vit channels for the step
        noise_pred_sliced = noise_pred_reshaped[:, :, :in_channels, :, :]
        noise_pred = noise_pred_sliced.reshape(batch_size, seq_len, in_channels * p * p)
        
        # 4. Scheduler Step
        t_prev = timesteps[i+1] if i < len(timesteps) - 1 else 0
        latents = scheduler.step(noise_pred, t, latents, t_prev=t_prev)
        
    print("Sampling complete.", flush=True)
    print(f"Output latent shape: {latents.shape}", flush=True)
    
    # Latents are [B, L, C_patch]
    # Can reshape back to 2D latents if needed
    patch_size = config.patch_size_for_vit
    in_channels = config.num_channels_for_vit
    
    side = int(np.sqrt(seq_len))
    if side * side == seq_len:
        print(f"Inferred 2D latent grid side: {side}", flush=True)
        
        # Unpatchify
        p = config.patch_size_for_vit
        c = config.num_channels_for_vit
        
        latents_2d = latents.reshape(batch_size, side, side, c, p, p)
        latents_2d = jnp.transpose(latents_2d, (0, 1, 4, 2, 5, 3))
        latents_2d = latents_2d.reshape(batch_size, side * p, side * p, c)
        
        print(f"Reshaped latent shape for VAE: {latents_2d.shape}", flush=True)
        
        # VAE Decoding
        print("Decoding latents to images using bundled VAE...", flush=True)
        # Scale latents before decoding (standard for SD VAE used in DiT)
        latents_2d = latents_2d / 0.18215
        images = model.vae_decoder(latents_2d)
        print(f"Output image shape: {images.shape}", flush=True)
        
        # Save images
        # Assuming output is roughly in [-1, 1] range
        images_np = np.array((images + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        
        for i, word in enumerate(prompt_words):
            img = Image.fromarray(images_np[i])
            img_path = f"{word.replace(' ', '_')}.png"
            img.save(img_path)
            print(f"Saved image to {img_path}", flush=True)
            
    else:
        print("Sequence length is not a perfect square.", flush=True)

if __name__ == "__main__":
    app.run(main)
