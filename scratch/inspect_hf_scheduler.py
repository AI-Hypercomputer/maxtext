import torch
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    set_alpha_to_one=True
)

print("HF alphas_cumprod[0]:", scheduler.alphas_cumprod[0])
print("HF alphas_cumprod[20]:", scheduler.alphas_cumprod[20])
print("HF alphas_cumprod[960]:", scheduler.alphas_cumprod[960])
print("HF alphas_cumprod[980]:", scheduler.alphas_cumprod[980])
