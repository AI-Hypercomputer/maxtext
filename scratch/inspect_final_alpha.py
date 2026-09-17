from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    set_alpha_to_one=True
)

print("final_alpha_cumprod:", scheduler.final_alpha_cumprod)
