from diffusers import DDIMScheduler
import torch

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

# Inspect scale_model_input
import inspect
print(inspect.getsource(scheduler.scale_model_input))
