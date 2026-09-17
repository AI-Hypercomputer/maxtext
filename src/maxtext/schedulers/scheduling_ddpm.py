import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union

class DDPMScheduler:
  """Denoising Diffusion Probabilistic Models (DDPM) Scheduler.
  
  Simplified implementation for JAX.
  """

  def __init__(
      self,
      num_train_timesteps: int = 1000,
      beta_start: float = 0.0001,
      beta_end: float = 0.02,
      beta_schedule: str = "linear",
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

    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    # Use 'min' instead of 'a_min' for jnp.clip
    self.log_betas = jnp.log(jnp.clip(self.betas, min=1e-20))

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
      key: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    """Predicts sample at previous timestep (reverse diffusion)."""
    
    t = timestep
    if isinstance(t, jnp.ndarray) and t.ndim > 0:
        t = t[0]
        
    beta_t = self.betas[t]
    alpha_t = self.alphas[t]
    alpha_bar_t = self.alphas_cumprod[t]

    coef1 = (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t)
    mean = (sample - coef1 * model_output) / jnp.sqrt(alpha_t)

    if t == 0 or key is None:
        return mean
    else:
        noise = jax.random.normal(key, sample.shape)
        variance = beta_t
        return mean + jnp.sqrt(variance) * noise
