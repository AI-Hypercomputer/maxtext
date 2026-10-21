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

"""Qwen3.8-Flash-Next decoder backbone and model implementation."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen38_flash_next import hyper_connection, scannable  # pylint: disable=no-name-in-module


class Qwen38Decoder(nnx.Module):
  """Decoder stack with token embedding, 4-stream residual, and final mixer."""

  def __init__(self, config: Any, *, rngs: nnx.Rngs):
    c, dt, wdt = config, jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
    self.emb_dim, self.vocab_size, self.eos_token_id = int(c.emb_dim), int(c.vocab_size), int(c.ple_eos_token_id)
    self.num_layers, self.hc_count, self.scan_layers = int(c.num_decoder_layers), int(c.hc_count), bool(c.scan_layers)
    self.dtype, self.weight_dtype = dt, wdt
    self.token_embedder = nnx.Embed(self.vocab_size, self.emb_dim, dtype=dt, param_dtype=wdt, rngs=rngs)
    if self.scan_layers:
      if int(config.inhomogeneous_layer_cycle_interval) != 4:
        raise ValueError(
            f"Scanned cycles require inhomogeneous_layer_cycle_interval == 4, got "
            f"{config.inhomogeneous_layer_cycle_interval}."
        )
      if self.num_layers % 4 != 0:
        raise ValueError(f"Scanned cycles require num_decoder_layers to be a multiple of 4, got {self.num_layers}.")
      if bool(config.use_ple_layer) and int(config.ple_layer_id) != 1:
        raise ValueError(f"Scanned cycles only support ple_layer_id == 1, got {config.ple_layer_id}.")
      self.num_cycles = self.num_layers // 4
      self.cycles = scannable.create_scanned_cycles(config, self.num_cycles, rngs=rngs)
      self.ple = scannable.build_ple_layer(config, rngs=rngs)
    else:
      for i in range(self.num_layers):
        setattr(self, f"layers_{i}", scannable.Qwen38DecoderLayer(config, i, include_ple=True, rngs=rngs))
    self.hyper_connection_mixer = hyper_connection.Qwen38GatedResidual(
        emb_dim=self.emb_dim,
        hc_count=self.hc_count,
        hc_lowrank=int(config.hc_lowrank),
        use_combine=False,
        eps=float(config.normalization_layer_epsilon),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      *,
      return_aux: bool = False,
      return_pre_mixer: bool = False,
  ) -> Any:
    """Forward pass computing full decoder representations."""
    tok, (b, s) = decoder_input_tokens, decoder_input_tokens.shape
    pos = jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32), (b, s)) if decoder_positions is None else decoder_positions
    cmask = (decoder_segment_ids != 0).astype(jnp.int32) if decoder_segment_ids is not None else None
    pids = tok if cmask is None else jnp.where(cmask.astype(jnp.bool_), tok, self.eos_token_id)
    h, aux_list = hyper_connection.init_hyper_connection_streams(self.token_embedder(tok), hc_count=self.hc_count), []
    if self.scan_layers:
      res = scannable.apply_scanned_cycles(
          self.cycles,
          self.ple,
          h,
          num_cycles=self.num_cycles,
          positions=pos,
          conv_mask=cmask,
          ple_input_ids=pids,
          segment_ids=decoder_segment_ids,
          return_aux=return_aux,
      )
      h, aux_list = res if return_aux else (res, [])
    else:
      for i in range(self.num_layers):
        out = getattr(self, f"layers_{i}")(
            h, positions=pos, conv_mask=cmask, ple_input_ids=pids, segment_ids=decoder_segment_ids, return_aux=return_aux
        )
        h, _ = (out[0], aux_list.append(out[1])) if return_aux else (out, None)
    h_4stream, h_mixed = h, self.hyper_connection_mixer.mix_input(h)
    if return_pre_mixer:
      return (h_mixed, h_4stream, aux_list) if return_aux else (h_mixed, h_4stream)
    return (h_mixed, aux_list) if return_aux else h_mixed


class Qwen38FlashNextModel(nnx.Module):
  """Top-level Qwen3.8-Flash-Next model with decoder and lm_head."""

  def __init__(self, config: Any, mesh: Any = None, *, quant_mode: str = "train", rngs: nnx.Rngs):
    self.config, self.mesh, self.quant_mode, self.decoder = config, mesh, quant_mode, Qwen38Decoder(config, rngs=rngs)
    self.lm_head = nnx.Linear(
        self.decoder.emb_dim,
        self.decoder.vocab_size,
        use_bias=False,
        dtype=self.decoder.dtype,
        param_dtype=self.decoder.weight_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      *,
      enable_dropout: bool = False,
      return_hidden_states: bool = False,
      return_aux: bool = False,
      **kwargs: Any,
  ) -> Any:
    """Forward pass computing language model logits and optional aux outputs."""
    del enable_dropout, kwargs
    out = self.decoder(
        decoder_input_tokens,
        decoder_positions,
        decoder_segment_ids,
        return_aux=return_aux,
        return_pre_mixer=return_hidden_states,
    )
    return (self.lm_head(out[0]), *out[1:]) if (return_hidden_states or return_aux) else self.lm_head(out)


def create_qwen38_flash_next_model(
    config: Any, mesh: Any = None, *, rngs: nnx.Rngs | None = None, quant_mode_str: str = "train", **_: Any
) -> Qwen38FlashNextModel:
  """Factory function creating a Qwen38FlashNextModel instance."""
  return Qwen38FlashNextModel(config=config, mesh=mesh, quant_mode=quant_mode_str, rngs=rngs or nnx.Rngs(params=0))
