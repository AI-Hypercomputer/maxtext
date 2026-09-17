from diffusers import DiTPipeline
import torch

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

# We want to intercept latents before VAE
# We can just run the loop manually or print from call hook if possible.
# Easiest is to copy call logic here and print.

def my_call(self, class_labels, num_inference_steps=50):
    batch_size = len(class_labels)
    latent_size = self.transformer.config.sample_size
    latent_channels = self.transformer.config.in_channels

    latents = torch.randn(
        batch_size, latent_channels, latent_size, latent_size
    )
    latent_model_input = torch.cat([latents] * 2)

    class_labels = torch.tensor(class_labels).reshape(-1)
    class_null = torch.tensor([1000] * batch_size)
    class_labels_input = torch.cat([class_labels, class_null], 0)

    self.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(self.scheduler.timesteps):
        if i % 10 == 0:
            print(f"HF Step {i}/50, Timestep {t}", flush=True)
        half = latent_model_input[: len(latent_model_input) // 2]
        latent_model_input = torch.cat([half, half], dim=0)
        
        # print("scale_model_input is identity:", torch.equal(latent_model_input, self.scheduler.scale_model_input(latent_model_input, t)))

        # broadcast to batch dimension
        timesteps = torch.tensor([t], dtype=torch.int64).expand(latent_model_input.shape[0])

        noise_pred = self.transformer(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input
        ).sample


        eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + 4.0 * (cond_eps - uncond_eps) # guidance_scale=4.0
        eps = torch.cat([half_eps, half_eps], dim=0)
        noise_pred = torch.cat([eps, rest], dim=1)

        if self.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

    latents, _ = latent_model_input.chunk(2, dim=0)
    
    print(f"Latent stats before scaling: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")
    
    scaled_latents = 1 / self.vae.config.scaling_factor * latents
    
    print(f"Latent stats after scaling: min={scaled_latents.min():.4f}, max={scaled_latents.max():.4f}, mean={scaled_latents.mean():.4f}")
    
    return scaled_latents # Stop here

my_call(pipe, [2, 879]) # white shark, umbrella
