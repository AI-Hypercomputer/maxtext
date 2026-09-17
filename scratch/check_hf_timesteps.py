from diffusers import DDIMScheduler
import numpy as np

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

scheduler.set_timesteps(50)
print("Leading timesteps:")
print(scheduler.timesteps)

try:
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        timestep_spacing="trailing"
    )
    scheduler.set_timesteps(50)
    print("Trailing timesteps:")
    print(scheduler.timesteps)
except Exception as e:
    print(f"Error with trailing: {e}")
