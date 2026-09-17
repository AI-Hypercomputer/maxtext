from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

print(f"final_alpha_cumprod: {scheduler.final_alpha_cumprod}")
print(f"config.set_alpha_to_one: {scheduler.config.set_alpha_to_one}")
