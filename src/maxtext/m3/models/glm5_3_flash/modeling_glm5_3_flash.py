# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.3-Flash autoregressive language model in m3 format."""

import dataclasses
from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import glm5_config
from maxtext.m3.models.glm5_3_flash import glm5_dsa_mla
from maxtext.m3.models.glm5_3_flash import glm5_mhc
from maxtext.m3.models.glm5_3_flash import glm5_moe
from maxtext.m3.models.glm5_3_flash import glm5_scan


@dataclasses.dataclass(frozen=True)
class Glm5Output:
  """Output of GLM-5.3-Flash forward pass with auxiliary data."""

  logits: jax.Array
  hidden_states: jax.Array
  mtp_logits: jax.Array | None
  aux: dict[str, Any]


class Glm5Embedding(nnx.Embed):
  """Token embedding module for GLM-5.3-Flash."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    super().__init__(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        dtype=getattr(jnp, config.dtype),
        param_dtype=getattr(jnp, config.weight_dtype),
        embedding_metadata={"sharding": sharding("embed", ("embed", None))} if sharding else {},
        rngs=rngs,
    )


class Glm5Model(nnx.Module):
  """GLM-5.3-Flash autoregressive language model with 4-stream mHC and hybrid layers."""

  def __init__(
      self,
      config: Any,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    self.config = glm5_config.Glm5Config.from_maxtext_config(config)
    if sharding is None and mesh is not None:
      sharding = glm5_config.Glm5Sharding()
    self.sharding = sharding
    self.remat_policy: Any = None

    cfg = self.config
    dt, p_dt = getattr(jnp, cfg.dtype), getattr(jnp, cfg.weight_dtype)
    self.embed_tokens = Glm5Embedding(cfg, rngs=rngs, sharding=sharding)

    num_prefix = min(cfg.num_hidden_layers, cfg.first_k_dense_replace)
    self.prefix_layers = nnx.List(
        [glm5_mhc.Glm5DecoderLayer(cfg, i, rngs=rngs, sharding=sharding) for i in range(num_prefix)]
    )
    rem = cfg.num_hidden_layers - num_prefix
    num_super_blocks, num_suffix = rem // 4, rem % 4
    self.super_block = (
        glm5_scan.create_stacked_super_block(cfg, num_super_blocks, rngs=rngs, sharding=sharding)
        if num_super_blocks > 0
        else None
    )
    suffix_start = num_prefix + 4 * num_super_blocks
    self.suffix_layers = nnx.List(
        [glm5_mhc.Glm5DecoderLayer(cfg, suffix_start + i, rngs=rngs, sharding=sharding) for i in range(num_suffix)]
    )

    self.hc_head = glm5_mhc.Glm5HyperHead()
    self.norm = glm5_dsa_mla.Glm5RMSNorm(
        cfg.hidden_size,
        eps=cfg.rms_norm_eps,
        param_dtype=p_dt,
        sharding=sharding,
    )
    lm_meta = {"sharding": sharding("lm_head", ("embed", "vocab"))} if sharding else {}
    self.lm_head = nnx.Linear(
        cfg.hidden_size,
        cfg.vocab_size,
        use_bias=False,
        dtype=dt,
        param_dtype=p_dt,
        kernel_metadata=lm_meta,
        rngs=rngs,
    )

  def forward_with_aux(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      **kwargs: Any,
  ) -> Glm5Output:
    """Executes full forward pass returning logits, hidden states, and auxiliary loss."""
    del decoder_positions, kwargs
    x = self.embed_tokens(decoder_input_tokens)
    hidden_streams = jnp.repeat(x[:, :, None, :], self.config.mhc_expansion_rate, axis=2)
    router_logits_list: list[jax.Array] = []

    layer_0_act: jax.Array | None = None

    def _run_layers(layers, streams, is_first: bool = False):
      nonlocal layer_0_act
      for idx, layer in enumerate(layers):
        if self.remat_policy is not None:
          fn = jax.checkpoint(
              lambda s, lyr=layer: lyr(s, decoder_segment_ids=decoder_segment_ids),
              policy=self.remat_policy,
          )
          streams, r = fn(streams)
        else:
          streams, r = layer(streams, decoder_segment_ids=decoder_segment_ids)
        if r is not None:
          router_logits_list.append(r)
        if is_first and idx == 0 and self.config.enable_debug_stats:
          layer_0_act = streams
      return streams

    if len(self.prefix_layers) > 0:
      hidden_streams = _run_layers(self.prefix_layers, hidden_streams, is_first=True)
    if self.super_block is not None:
      if self.config.enable_debug_stats and layer_0_act is None:
        graphdef, params, non_params = nnx.split(self.super_block, nnx.Param, ...)
        p_0 = glm5_scan._slice_state(params, 0)  # pylint: disable=protected-access
        np_0 = glm5_scan._slice_state(non_params, 0)  # pylint: disable=protected-access
        first_sb = nnx.merge(graphdef, p_0, np_0)
        layer_0_act, _ = first_sb.slot_0(hidden_streams, decoder_segment_ids=decoder_segment_ids)
      hidden_streams, sb_r = glm5_scan.scan_super_blocks(
          self.super_block,
          hidden_streams,
          decoder_segment_ids=decoder_segment_ids,
          scan_layers=self.config.scan_layers,
          remat_policy=self.remat_policy,
      )
      router_logits_list.extend(sb_r)
    is_suffix_first = layer_0_act is None
    hidden_streams = _run_layers(self.suffix_layers, hidden_streams, is_first=is_suffix_first)
    last_block_act = hidden_streams
    collapsed = self.hc_head(hidden_streams)
    normed = self.norm(collapsed)
    logits = self.lm_head(normed)
    cfg = self.config
    num_moe_layers = max(0, cfg.num_hidden_layers - cfg.first_k_dense_replace)
    if len(router_logits_list) != num_moe_layers:
      raise ValueError(f"Expected {num_moe_layers} MoE router_logits entries, got" f" {len(router_logits_list)}.")
    if num_moe_layers == 0:
      moe_lb_loss = jnp.zeros((), dtype=jnp.float32)
    else:
      flat_router = [r.reshape(-1, r.shape[-1]) for r in router_logits_list]
      moe_lb_loss = glm5_moe.compute_moe_load_balancing_loss(flat_router, cfg.n_routed_experts, cfg.num_experts_per_tok)
    aux: dict[str, Any] = {
        "router_logits": tuple(router_logits_list),
        "moe_lb_loss": moe_lb_loss,
    }
    if cfg.enable_debug_stats:
      if layer_0_act is None:
        raise ValueError("enable_debug_stats=True requires layer 0 activation, but layer_0_act was not captured.")
      l0_f32 = layer_0_act.astype(jnp.float32)
      last_f32 = last_block_act.astype(jnp.float32)
      aux["debug_stats"] = {
          "first_block_mean": jnp.mean(l0_f32),
          "first_block_var": jnp.var(l0_f32),
          "last_block_mean": jnp.mean(last_f32),
          "last_block_var": jnp.var(last_f32),
          "hidden_norm_mean": jnp.mean(jnp.linalg.norm(normed, axis=-1)),
          "logits_max": jnp.max(logits),
          "logits_min": jnp.min(logits),
      }
    return Glm5Output(logits=logits, hidden_states=normed, mtp_logits=None, aux=aux)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      **kwargs: Any,
  ) -> jax.Array:
    """Executes full forward pass returning output logits."""
    return self.forward_with_aux(decoder_input_tokens, decoder_positions, decoder_segment_ids, **kwargs).logits

  def logits_from_hidden_states_for_vocab_tiling(self, hidden_states: jax.Array, **kwargs: Any) -> jax.Array:
    """Projects hidden states to logits for vocab-tiled computation."""
    del kwargs
    return self.lm_head(hidden_states)

  @classmethod
  def from_maxtext_config(
      cls,
      config: Any,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
      quant_mode_str: str | None = None,
      **kwargs: Any,
  ) -> "Glm5Model":
    """Constructs a Glm5Model from MaxText configuration."""
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    if quant_mode_str is not None and quant_mode_str != "":
      raise ValueError("Runtime quantization is unsupported for GLM-5.3-Flash:" f" {quant_mode_str!r}.")
    if kwargs:
      raise TypeError("Unexpected keyword arguments in from_maxtext_config:" f" {sorted(kwargs)!r}.")
    return cls(config=config, mesh=mesh, rngs=rngs)


def create_glm5_3_flash_model(config: Any, mesh: Mesh | None = None, *, rngs: nnx.Rngs, **kwargs: Any) -> Glm5Model:
  """Creates a Glm5Model instance from config and mesh."""
  return Glm5Model.from_maxtext_config(config, mesh, rngs=rngs, **kwargs)
