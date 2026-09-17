from diffusers import DiTPipeline
import torch

# Load in bfloat16
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.bfloat16)

# We want to intercept latents before VAE
def my_call(self, class_labels, num_inference_steps=50):
    batch_size = len(class_labels)
    latent_size = self.transformer.config.sample_size
    latent_channels = self.transformer.config.in_channels

    # Start with bfloat16 noise
    latents = torch.randn(
        batch_size, latent_channels, latent_size, latent_size, dtype=torch.bfloat16
    )
    latent_model_input = torch.cat([latents] * 2)

    class_labels = torch.tensor(class_labels).reshape(-1)
    class_null = torch.tensor([1000] * batch_size)
    class_labels_input = torch.cat([class_labels, class_null], 0)

    self.scheduler.set_timesteps(num_inference_steps)
    for t in self.scheduler.timesteps:
        half = latent_model_input[: len(latent_model_input) // 2]
        latent_model_input = torch.cat([half, half], dim=0)
        
        timesteps = torch.tensor([t], dtype=torch.int64).expand(latent_model_input.shape[0])

        noise_pred = self.transformer(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input
        ).sample

        eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + 4.0 * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        noise_pred = torch.cat([eps, rest], dim=1)

        if self.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

    latents, _ = latent_model_input.chunk(2, dim=0)
    
    print(f"Latent stats before scaling (bf16): min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")
    
    scaled_latents = 1 / self.vae.config.scaling_factor * latents
    
    print(f"Latent stats after scaling (bf16): min={scaled_latents.min():.4f}, max={scaled_latents.max():.4f}, mean={scaled_latents.mean():.4f}")
    
    # Try decoding
    samples = self.vae.decode(scaled_latents).sample
    print(f"Output image stats (bf16): min={samples.min():.4f}, max={samples.max():.4f}, mean={samples.mean():.4f}")
    
    return samples

my_call(pipe, [2, 879])
