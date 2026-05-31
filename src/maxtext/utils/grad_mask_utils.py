# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-token gradient mask applied at a layer boundary.

Forward is identity; backward zeros tokens whose feature-axis RMS exceeds
the configured threshold. Healthy tokens pass through unchanged. Used at
decoder-layer boundaries to bound per-layer cotangent magnitudes
(see deepseek model usage)."""

import jax
import jax.numpy as jnp


@jax.custom_vjp
def _grad_mask(x: jax.Array, threshold: jax.Array) -> jax.Array:
  return x


def _grad_mask_fwd(x: jax.Array, threshold: jax.Array):
  return x, threshold


def _grad_mask_bwd(threshold: jax.Array, g: jax.Array):
  rms = jnp.sqrt(jnp.mean(jnp.square(g.astype(jnp.float32)), axis=-1, keepdims=True))
  mask = rms <= threshold
  return (jnp.where(mask, g, jnp.zeros_like(g)), jnp.zeros_like(threshold))


_grad_mask.defvjp(_grad_mask_fwd, _grad_mask_bwd)


def maybe_grad_mask(x: jax.Array, cfg) -> jax.Array:
  """Per-token gradient mask if cfg.grad_mask_threshold > 0; else identity."""
  if cfg.grad_mask_threshold > 0.0:
    return _grad_mask(x, jnp.asarray(cfg.grad_mask_threshold, jnp.float32))
  return x
