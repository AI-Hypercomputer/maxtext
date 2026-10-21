# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gated DeltaNet (GDN) linear attention module for Qwen3.8-Flash-Next."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
# pylint: disable=no-name-in-module,too-many-positional-arguments
from maxtext.m3.models.qwen3_8_flash_next import gdn_recurrence
from maxtext.m3.models.qwen3_8_flash_next import hyper_connection


class Qwen3_8GatedDeltaNet(nnx.Module):
  """Gated DeltaNet (GDN) linear attention layer for Qwen3.8-Flash-Next."""

  def __init__(
      self,
      emb_dim: int = 2560,
      num_k_heads: int = 16,
      num_v_heads: int = 48,
      head_k_dim: int = 128,
      head_v_dim: int = 128,
      conv_kernel_size: int = 4,
      chunk_size: int = 64,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      output_gate_activation: str,
      rngs: nnx.Rngs,
  ):
    del sharding
    self.emb_dim, self.chunk_size = emb_dim, chunk_size
    self.num_k_heads, self.num_v_heads = num_k_heads, num_v_heads
    if num_v_heads % num_k_heads != 0:
      raise ValueError(f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads}).")
    self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim
    self.output_gate_activation = output_gate_activation
    self.dtype = dtype

    self.key_dim = num_k_heads * head_k_dim
    self.value_dim = num_v_heads * head_v_dim
    conv_dim = self.key_dim * 2 + self.value_dim

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.in_proj_qkv = nnx.Linear(emb_dim, conv_dim, **kw)
    self.in_proj_z = nnx.Linear(emb_dim, self.value_dim, **kw)
    self.in_proj_b = nnx.Linear(emb_dim, num_v_heads, **kw)
    self.in_proj_a = nnx.Linear(emb_dim, num_v_heads, **kw)

    self.conv1d = nnx.Conv(
        in_features=conv_dim,
        out_features=conv_dim,
        kernel_size=(conv_kernel_size,),
        feature_group_count=conv_dim,
        padding="CAUSAL",
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )
    u = jax.random.uniform(rngs.params(), (num_v_heads,), minval=0.01, maxval=16.0)
    self.A_log = nnx.Param(jnp.log(u.astype(jnp.float32)))  # pylint: disable=invalid-name
    self.dt_bias = nnx.Param(jnp.ones((num_v_heads,), dtype=jnp.float32))
    self.norm = hyper_connection.Qwen3_8RMSNormGated(head_v_dim, eps=eps, activation=output_gate_activation, rngs=rngs)
    self.out_proj = nnx.Linear(self.value_dim, emb_dim, **kw)

  def __call__(
      self,
      hidden_states: jax.Array,
      attention_mask: jax.Array | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    if attention_mask is not None:
      hidden_states = (hidden_states * jnp.expand_dims(attention_mask, -1)).astype(hidden_states.dtype)

    b, s, _ = hidden_states.shape
    mixed_qkv = jax.nn.silu(self.conv1d(self.in_proj_qkv(hidden_states)))
    z = self.in_proj_z(hidden_states).reshape(b, s, self.num_v_heads, self.head_v_dim)

    beta = jax.nn.sigmoid(self.in_proj_b(hidden_states))
    g = -jnp.exp(self.A_log[...].astype(jnp.float32)) * jax.nn.softplus(
        self.in_proj_a(hidden_states).astype(jnp.float32) + self.dt_bias[...].astype(jnp.float32)
    )

    kd = self.key_dim
    query = mixed_qkv[..., :kd].reshape(b, s, self.num_k_heads, self.head_k_dim)
    key = mixed_qkv[..., kd : kd * 2].reshape(b, s, self.num_k_heads, self.head_k_dim)
    value = mixed_qkv[..., kd * 2 :].reshape(b, s, self.num_v_heads, self.head_v_dim)

    if attention_mask is not None:
      mask = attention_mask != 0
      key = jnp.where(mask[:, :, None, None], key, 0.0)
      value = jnp.where(mask[:, :, None, None], value, 0.0)
      g = jnp.where(mask[:, :, None], g, 0.0)

    if self.num_v_heads // self.num_k_heads > 1:
      rep = self.num_v_heads // self.num_k_heads
      query, key = jnp.repeat(query, rep, axis=2), jnp.repeat(key, rep, axis=2)

    if s == 1 and initial_state is not None:
      core_attn_out, last_state = gdn_recurrence.jax_recurrent_gated_delta_rule(
          query,
          key,
          value,
          g=g,
          beta=beta,
          initial_state=initial_state,
          output_final_state=output_final_state,
          use_qk_l2norm_in_kernel=True,
      )
    else:
      core_attn_out, last_state = gdn_recurrence.jax_chunk_gated_delta_rule(
          query,
          key,
          value,
          g=g,
          beta=beta,
          chunk_size=self.chunk_size,
          initial_state=initial_state,
          output_final_state=output_final_state,
          use_qk_l2norm_in_kernel=True,
      )
    core_normed = self.norm(core_attn_out, z).reshape(b, s, self.value_dim)
    out = self.out_proj(core_normed)
    return (out, last_state) if output_final_state else out
