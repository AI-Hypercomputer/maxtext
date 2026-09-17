from diffusers import DDIMScheduler
import torch

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    set_alpha_to_one=True
)

x = torch.ones(1, 4, 32, 32)
y = scheduler.scale_model_input(x, 500)

print("Input equal to output?", torch.allclose(x, y))
print("Input:", x[0,0,0,0])
print("Output:", y[0,0,0,0])
