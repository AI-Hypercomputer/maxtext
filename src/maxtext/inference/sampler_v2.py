# Temporary copy of sampler.py (V2)
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
    config = pyconfig.initialize(argv)
    _validate_config(config)
    
    jax.config.update("jax_use_shardy_partitioner", config.shardy)
    max_utils.print_system_information()
    
    if len(jax.devices()) == 4:
        devices = jax.devices()
    elif len(jax.devices()) >= 8:
        devices = jax.devices()[4:8]
    else:
        devices = jax.devices()[:4]
    print(f"Restricting to devices: {devices}")
    devices_array = maxtext_utils.create_device_mesh(config, devices=devices)

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
        params = jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), params)
        nnx.update(model, params)
        print("Parameters loaded and replicated successfully.", flush=True)
        
        pos_embed_val = model.pos_embed.get_value()
        print(f"Loaded pos_embed stats: min={pos_embed_val.min():.4f}, max={pos_embed_val.max():.4f}, std={pos_embed_val.std():.4f}", flush=True)
        
        vae_conv_out_kernel = model.vae_decoder.conv_out.kernel.value
        print(f"Loaded VAE conv_out kernel stats: min={vae_conv_out_kernel.min():.4f}, max={vae_conv_out_kernel.max():.4f}, std={vae_conv_out_kernel.std():.4f}", flush=True)

    
    word_to_id = {
        "white shark": 2,
        "umbrella": 879,
    }
    
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
    
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )
    
    out_channels = getattr(config, 'out_channels_for_vit', config.num_channels_for_vit)
    patch_dim = config.patch_size_for_vit * config.patch_size_for_vit * out_channels
    
    seq_len = config.max_target_length
    
    in_patch_dim = config.patch_size_for_vit * config.patch_size_for_vit * config.num_channels_for_vit
    
    print(f"Sampling parameters: batch_size={batch_size}, seq_len={seq_len}, patch_dim={patch_dim}, in_patch_dim={in_patch_dim}")
    
    rng, rng_noise = jax.random.split(rng)
    latents = jax.random.normal(rng_noise, (batch_size, seq_len, in_patch_dim))
    latents = jax.device_put(latents, replicated_sharding)
    
    guidance_scale = 4.0
    uncond_class = config.vocab_size 
    
    num_inference_steps = 50 
    timesteps = scheduler.set_timesteps(num_inference_steps)
    
    print(f"Starting sampling loop for {len(timesteps)} steps...")
    
    for i, t in enumerate(timesteps):
        if i % 10 == 0:
            print(f"Step {i}/{len(timesteps)}, Timestep {t}")
            
        latent_model_input = jnp.concatenate([latents] * 2)
        t_batch = jnp.array([t] * (batch_size * 2))
        t_batch = jax.device_put(t_batch, replicated_sharding)
        
        y_cond = class_labels
        y_uncond = jnp.array([uncond_class] * batch_size, dtype=jnp.int32)
        y_batch = jnp.concatenate([y_cond, y_uncond])
        y_batch = jax.device_put(y_batch, replicated_sharding)
        
        # Layout conversion: (P_h, P_w, C) -> (C, P_h, P_w) for patch_embed
        p = config.patch_size_for_vit
        c = config.num_channels_for_vit
        
        latent_model_input_reshaped = latent_model_input.reshape(latent_model_input.shape[0], seq_len, p, p, c)
        latent_model_input_converted = jnp.transpose(latent_model_input_reshaped, (0, 1, 4, 2, 3))
        latent_model_input_flattened = latent_model_input_converted.reshape(latent_model_input.shape[0], seq_len, p * p * c)
        
        noise_pred_batch = model(latent_model_input_flattened, t_batch, y_batch)
        
        noise_pred_cond, noise_pred_uncond = jnp.split(noise_pred_batch, 2, axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        p = config.patch_size_for_vit
        in_channels = config.num_channels_for_vit
        c_out = getattr(config, 'out_channels_for_vit', in_channels)
        
        # V2 layout assumption: (P, P, C)
        noise_pred_reshaped = noise_pred.reshape(batch_size, seq_len, p, p, c_out)
        noise_pred_sliced = noise_pred_reshaped[:, :, :, :, :in_channels]
        noise_pred = noise_pred_sliced.reshape(batch_size, seq_len, p * p * in_channels)
        
        step_ratio = 1000 // num_inference_steps
        t_prev = t - step_ratio
        latents = scheduler.step(noise_pred, t, latents, t_prev=t_prev)
        
        if i % 10 == 0:
            print(f"  Latent stats after step: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}", flush=True)
            print(f"  Noise pred stats: min={noise_pred.min():.4f}, max={noise_pred.max():.4f}, mean={noise_pred.mean():.4f}", flush=True)

        
    print("Sampling complete.", flush=True)
    print(f"Output latent shape: {latents.shape}", flush=True)
    
    patch_size = config.patch_size_for_vit
    in_channels = config.num_channels_for_vit
    
    side = int(np.sqrt(seq_len))
    if side * side == seq_len:
        print(f"Inferred 2D latent grid side: {side}", flush=True)
        
        p = config.patch_size_for_vit
        c = config.num_channels_for_vit
        
        # V2 layout assumption: (P, P, C)
        latents_2d = latents.reshape(batch_size, side, side, p, p, c)
        latents_2d = jnp.transpose(latents_2d, (0, 1, 3, 2, 4, 5))
        latents_2d = latents_2d.reshape(batch_size, side * p, side * p, c)
        
        print(f"Reshaped latent shape for VAE: {latents_2d.shape}", flush=True)
        
        print("Decoding latents to images using bundled VAE...", flush=True)
        latents_2d = latents_2d / 0.18215
        print(f"Latent stats before VAE (scaled): min={latents_2d.min():.4f}, max={latents_2d.max():.4f}, mean={latents_2d.mean():.4f}", flush=True)


        images = model.vae_decoder(latents_2d)
        images = jnp.tanh(images)

        print(f"Output image shape: {images.shape}", flush=True)
        print(f"Image stats before clipping: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}", flush=True)
        
        images_np = np.array((images + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        
        for i, word in enumerate(prompt_words):
            img = Image.fromarray(images_np[i])
            img_path = f"{word.replace(' ', '_')}_scaled.png" # Save with _scaled suffix

            img.save(img_path)
            print(f"Saved image to {img_path}", flush=True)
            
    else:
        print("Sequence length is not a perfect square.", flush=True)

if __name__ == "__main__":
    app.run(main)
