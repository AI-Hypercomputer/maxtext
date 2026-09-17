import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union

class DDIMScheduler:
  """Denoising Diffusion Implicit Models (DDIM) Scheduler.
  
  Simplified implementation for JAX.
  Supports non-Markovian sampling with skipping steps.
  """

  def __init__(
      self,
      num_train_timesteps: int = 1000,
      beta_start: float = 0.0001,
      beta_end: float = 0.02,
      beta_schedule: str = "linear",
      set_alpha_to_one: bool = True,
  ):
    self.num_train_timesteps = num_train_timesteps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.beta_schedule = beta_schedule

    if beta_schedule == "linear":
      self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
    elif beta_schedule == "scaled_linear":
      self.betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
    else:
      raise ValueError(f"Unknown beta schedule: {beta_schedule}")

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)

    # DDIM specific: set alpha_0 to 1
    self.set_alpha_to_one = set_alpha_to_one
    if set_alpha_to_one:
        self.final_alpha_cumprod = 1.0
    else:
        self.final_alpha_cumprod = self.alphas_cumprod[0]

    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

  def set_timesteps(self, num_inference_steps: int) -> jnp.ndarray:
    """Sets inference timesteps."""
    step_ratio = self.num_train_timesteps // num_inference_steps
    timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round().astype(jnp.int32)
    # Reverse order for sampling
    return timesteps[::-1]

  def add_noise(
      self,
      original_samples: jnp.ndarray,
      noise: jnp.ndarray,
      timesteps: jnp.ndarray
  ) -> jnp.ndarray:
    """Adds noise to original samples (forward diffusion)."""
    sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
    sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

    sqrt_alpha_prod = sqrt_alpha_prod.reshape(-1, *([1] * (original_samples.ndim - 1)))
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.reshape(-1, *([1] * (original_samples.ndim - 1)))

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

  def step(
      self,
      model_output: jnp.ndarray,
      timestep: Union[int, jnp.ndarray],
      sample: jnp.ndarray,
      eta: float = 0.0, # Deterministic by default (eta=0)
      key: Optional[jax.Array] = None,
      t_prev: Optional[Union[int, jnp.ndarray]] = None,
  ) -> jnp.ndarray:
    """Predicts sample at previous timestep (reverse diffusion)."""
    # Assuming model_output is predicted noise (epsilon)
    
    t = timestep
    if isinstance(t, jnp.ndarray) and t.ndim > 0:
        t = t[0]
        
    # We need t_prev
    # In DDIM, we need to know the next step in the sequence
    # This simplified version assumes we are just stepping 1 step down in standard sequence
    # For custom schedules (set_timesteps), we need to handle t_prev differently.
    # Here we assume standard step for simplicity, or we can pass t_prev.
    
    if t_prev is None:
        t_prev = t - 1
    elif isinstance(t_prev, jnp.ndarray) and t_prev.ndim > 0:
        t_prev = t_prev[0]
    
    alpha_bar_t = self.alphas_cumprod[t]
    
    clamped_t_prev = jnp.maximum(t_prev, 0)
    alpha_bar_t_prev = jnp.where(t_prev >= 0, self.alphas_cumprod[clamped_t_prev], self.final_alpha_cumprod)
    
    # 1. Predict x_0
    pred_x0 = (sample - jnp.sqrt(1 - alpha_bar_t) * model_output) / jnp.sqrt(alpha_bar_t)
    
    # 2. Compute direction pointing to x_t
    dir_xt = jnp.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t_prev / alpha_bar_t)) * model_output
    
    # 3. Add random noise (if eta > 0)
    if eta > 0.0 and key is not None:
        noise = jax.random.normal(key, sample.shape)
        variance = (1 - alpha_bar_t_prev / alpha_bar_t) * (1 - alpha_bar_t) / (1 - alpha_bar_t_prev)
        random_noise = eta * jnp.sqrt(variance) * noise
    else:
        random_noise = 0.0
        
    # 4. Compute x_{t-1}
    pred_prev_sample = jnp.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + random_noise
    
    return pred_prev_sample
