import torch
from diffusers import DDIMScheduler

scheduler = DDIMScheduler.from_pretrained("facebook/DiT-XL-2-256", subfolder="scheduler")
sample = torch.randn(1, 4, 32, 32)
timestep = 500

scaled_sample = scheduler.scale_model_input(sample, timestep)
print(f"Is identical: {torch.allclose(sample, scaled_sample)}")
if not torch.allclose(sample, scaled_sample):
    print(f"Sample mean: {sample.mean()}, Scaled sample mean: {scaled_sample.mean()}")
