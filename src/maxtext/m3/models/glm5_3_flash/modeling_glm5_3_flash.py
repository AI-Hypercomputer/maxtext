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


@jax.tree_util.register_dataclass
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
        embedding_metadata={"sharding": sharding("embed", ("vocab", "embed"))} if sharding else {},
        rngs=rngs,
    )


class Glm5Model(nnx.Module):
  """GLM-5.3-Flash autoregressive language model with 4-stream mHC and hybrid layers."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
      place_on_devices: bool,
  ):
    if not isinstance(place_on_devices, bool):
      raise TypeError("place_on_devices must be a bool, got" f" {type(place_on_devices).__name__}")
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    self.config = glm5_config.Glm5Config.from_maxtext_config(config)
    self.mesh = mesh
    if sharding is None and mesh is not None:
      sharding = glm5_config.Glm5Sharding.from_mesh(mesh)
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
        glm5_scan.create_stacked_super_block(
            cfg,
            num_super_blocks,
            rngs=rngs,
            sharding=sharding,
            place_on_devices=place_on_devices,
        )
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
        precision=jax.lax.Precision.HIGHEST,
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
    if self.sharding is not None:
      hidden_streams = self.sharding.constrain(hidden_streams, "hidden_streams", ("batch", "length", None, "embed"))
    cfg = self.config
    attention_mask = (decoder_segment_ids != 0) if decoder_segment_ids is not None else None
    keep_router_logits = bool(cfg.enable_debug_stats)
    router_logits_list: list[jax.Array] = []
    lb_stats = glm5_moe.init_moe_lb_stats(cfg.n_routed_experts)

    def _run_layers(layers, streams, cur_lb):
      for layer in layers:
        if self.remat_policy is not None:
          fn = jax.checkpoint(
              lambda s, lyr=layer: lyr(s, decoder_segment_ids=decoder_segment_ids),
              policy=self.remat_policy,
              prevent_cse=True,
          )
          streams, r_logits, _, _ = fn(streams)
        else:
          streams, r_logits, _, _ = layer(streams, decoder_segment_ids=decoder_segment_ids)
        if r_logits is not None:
          cur_lb = glm5_moe.accumulate_moe_lb_stats(
              cur_lb,
              r_logits,
              num_experts=cfg.n_routed_experts,
              top_k=cfg.num_experts_per_tok,
              attention_mask=attention_mask,
          )
          if keep_router_logits:
            router_logits_list.append(r_logits)
      return streams, cur_lb

    hidden_streams, lb_stats = _run_layers(self.prefix_layers, hidden_streams, lb_stats)
    if self.super_block is not None:
      hidden_streams, sb_r, lb_stats, _, _ = glm5_scan.scan_super_blocks(
          self.super_block,
          hidden_streams,
          decoder_segment_ids=decoder_segment_ids,
          scan_layers=cfg.scan_layers,
          remat_policy=self.remat_policy,
          return_router_logits=keep_router_logits,
          lb_stats=lb_stats,
      )
      if keep_router_logits:
        router_logits_list.extend(sb_r)
    hidden_streams, lb_stats = _run_layers(self.suffix_layers, hidden_streams, lb_stats)
    collapsed = self.hc_head(hidden_streams)
    normed = self.norm(collapsed)
    logits = self.lm_head(normed)
    num_moe_layers = max(0, cfg.num_hidden_layers - cfg.first_k_dense_replace)
    if keep_router_logits and len(router_logits_list) != num_moe_layers:
      raise ValueError(f"Expected {num_moe_layers} MoE router_logits entries, got" f" {len(router_logits_list)}.")
    if num_moe_layers == 0:
      moe_lb_loss = jnp.zeros((), dtype=jnp.float32)
    else:
      moe_lb_loss = glm5_moe.finalize_moe_lb_loss(lb_stats, cfg.n_routed_experts)
    debug_stats = (
        {
            "hidden_norm_mean": jnp.mean(jnp.linalg.norm(normed, axis=-1)),
            "logits_max": jnp.max(logits),
            "logits_min": jnp.min(logits),
        }
        if cfg.enable_debug_stats
        else {}
    )
    aux = {
        "router_logits": (tuple(router_logits_list) if cfg.enable_debug_stats else ()),
        "moe_lb_loss": moe_lb_loss,
        "debug_stats": debug_stats,
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

  def logits_from_hidden_states_for_vocab_tiling(
      self,
      hidden_states: jax.Array,
      deterministic: bool = True,
      **kwargs: Any,
  ) -> jax.Array:
    """Projects post-final-norm hidden states to output logits for vocab tiling.

    Args:
      hidden_states: Post-final-norm hidden states, matching
        Glm5Output.hidden_states (after collapsing the 4 mHC streams via hc_head
        and applying self.norm).
      deterministic: Unused flag for API parity.
      **kwargs: Ignored keyword arguments for API parity.

    Returns:
      Output logits projected via self.lm_head.
    """
    del deterministic, kwargs
    return self.lm_head(hidden_states)

  @classmethod
  def from_maxtext_config(
      cls,
      config: Any,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
      quant_mode_str: str | None = None,
      place_on_devices: bool,
      **kwargs: Any,
  ) -> "Glm5Model":
    """Constructs a Glm5Model from MaxText configuration."""
    if not isinstance(place_on_devices, bool):
      raise TypeError("place_on_devices must be a bool, got" f" {type(place_on_devices).__name__}")
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    if not isinstance(config, glm5_config.Glm5Config) and config.quantization not in (None, ""):
      raise ValueError("Quantization is unsupported for GLM-5.3-Flash:" f" {config.quantization!r}.")
    if quant_mode_str not in (None, "", "train", "serve", "convert"):
      raise ValueError("Runtime quantization is unsupported for GLM-5.3-Flash:" f" {quant_mode_str!r}.")
    if kwargs:
      raise TypeError("Unexpected keyword arguments in from_maxtext_config:" f" {sorted(kwargs)!r}.")
    return cls(config=config, mesh=mesh, rngs=rngs, place_on_devices=place_on_devices)


def create_glm5_3_flash_model(config: Any, mesh: Mesh | None = None, *, rngs: nnx.Rngs, **kwargs: Any) -> Glm5Model:
  """Creates a Glm5Model instance from config and mesh."""
  if "place_on_devices" in kwargs:
    raise TypeError(
        "place_on_devices is fixed to False in create_glm5_3_flash_model;"
        " pass place_on_devices to Glm5Model or Glm5Model.from_maxtext_config"
        " directly instead."
    )
  return Glm5Model.from_maxtext_config(config, mesh, rngs=rngs, place_on_devices=False, **kwargs)
