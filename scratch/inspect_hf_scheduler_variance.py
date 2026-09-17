import torch
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

# Print _get_variance source
import inspect
print(inspect.getsource(scheduler._get_variance))
