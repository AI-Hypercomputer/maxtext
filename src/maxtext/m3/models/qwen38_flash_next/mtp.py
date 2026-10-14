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

"""Multi-Token Prediction (MTP) draft layer for Qwen3.8-Flash-Next."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen38_flash_next import hyper_connection  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen38_flash_next import scannable  # pylint: disable=no-name-in-module


class Qwen38MTPBlock(nnx.Module):
  """Single-layer MTP draft block fusing next-token embeddings and HC streams."""

  def __init__(self, config: Any, *, rngs: nnx.Rngs):
    self.config, self.emb_dim = config, int(config.emb_dim)
    self.hc_count, self.hc_lowrank = int(config.hc_count), int(config.hc_lowrank)
    dt, wdt, eps = jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype), float(config.normalization_layer_epsilon)
    self.pre_fc_norm_embedding = hyper_connection.Qwen38ZeroCenteredRMSNorm(self.emb_dim, eps=eps, rngs=rngs)
    self.pre_fc_norm_hidden = hyper_connection.Qwen38ZeroCenteredRMSNorm(self.hc_count * self.emb_dim, eps=eps, rngs=rngs)
    l_kw = {"use_bias": False, "dtype": dt, "param_dtype": wdt, "rngs": rngs}
    self.fc_embedding = nnx.Linear(self.emb_dim, self.emb_dim, **l_kw)
    self.fc_hidden = nnx.Linear(self.emb_dim, self.emb_dim, **l_kw)
    cyc = int(config.inhomogeneous_layer_cycle_interval)
    self.layers_0 = scannable.Qwen38DecoderLayer(config, layer_idx=cyc - 1, include_ple=False, rngs=rngs)
    self.hyper_connection_mixer = hyper_connection.Qwen38GatedResidual(
        emb_dim=self.emb_dim,
        hc_count=self.hc_count,
        hc_lowrank=self.hc_lowrank,
        use_combine=False,
        eps=eps,
        dtype=dt,
        weight_dtype=wdt,
        rngs=rngs,
    )

  def __call__(
      self,
      hc_hidden_states: jax.Array,
      next_token_embeds: jax.Array,
      *,
      positions: jax.Array | None = None,
      conv_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      return_aux: bool = False,
  ) -> jax.Array | tuple[jax.Array, dict[str, Any]]:
    """Forward pass fusing next-token embeddings with HC hidden states."""
    expected_dim = self.hc_count * self.emb_dim
    if hc_hidden_states.shape[-1] != expected_dim:
      raise ValueError(
          f"hc_hidden_states last dimension {hc_hidden_states.shape[-1]} "
          f"must equal hc_count * emb_dim ({expected_dim})."
      )
    expected_embed_shape = (*hc_hidden_states.shape[:-1], self.emb_dim)
    if next_token_embeds.shape != expected_embed_shape:
      raise ValueError(f"next_token_embeds shape {next_token_embeds.shape} must match {expected_embed_shape}.")
    e_proj = self.fc_embedding(self.pre_fc_norm_embedding(next_token_embeds))
    h_norm = self.pre_fc_norm_hidden(hc_hidden_states).reshape(*hc_hidden_states.shape[:-1], self.hc_count, self.emb_dim)
    h_proj = self.fc_hidden(h_norm)
    h_mtp_in = (h_proj + jnp.expand_dims(e_proj, axis=-2)).reshape(hc_hidden_states.shape)
    out = self.layers_0(
        h_mtp_in, positions=positions, conv_mask=conv_mask, segment_ids=segment_ids, return_aux=return_aux
    )
    h_layer, aux = out if return_aux else (out, None)
    h_out = self.hyper_connection_mixer.mix_input(h_layer)
    return (h_out, aux) if return_aux else h_out
