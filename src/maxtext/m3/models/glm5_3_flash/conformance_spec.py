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

"""Conformance spec for GLM-5.3-Flash (`m3/README.md` Invariants (1)-(7))."""

from collections.abc import Mapping
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models import glm5_3_flash
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from tests.unit.m3.conformance import spec

_BATCH_SIZE = 2
_SEQ_LEN = 8
_ROW1_VALID_LEN = 7
_VOCAB_SIZE = 256
_NUM_SCANNED_BLOCKS = 2
# TODO(yaoyuchen): Make threshold configurable via flag/input (default 128).
# Covers all non-APE 2-D+ weights ((64, 2) b_proj/weights_proj, (32, 1, 4)
# conv1d); APE (2, 16)=32 is unsharded.
_MIN_SHARDED_PARAM_ELEMENTS = 128
_EXPERT_WEIGHT_PATTERNS = (r"\.mlp\.experts\.(gate_up_proj|down_proj)$",)
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting
# to 5e-2.
_BF16_ATOL = 5e-2
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting
# to 5e-2.
_BF16_RTOL = 5e-2

_GLM5_MINI_CFG = glm5_3_flash.Glm5Config.mini_config(
    vocab_size=_VOCAB_SIZE,
    hidden_size=64,
    num_hidden_layers=10,
    first_k_dense_replace=1,
    intermediate_size=128,
    moe_intermediate_size=32,
    n_routed_experts=4,
    num_experts_per_tok=2,
    q_lora_rank=32,
    kv_lora_rank=16,
    num_attention_heads=2,
    num_key_value_heads=2,
    qk_nope_head_dim=16,
    v_head_dim=16,
    index_n_heads=2,
    index_head_dim=16,
    index_topk=4,
    index_kpool=2,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    mhc_expansion_rate=2,
    mhc_sinkhorn_iters=1,
    kda_implementation="xla",
    dtype="float32",
    weight_dtype="float32",
)

_GLM5_MINI_BF16_CFG = dataclasses.replace(
    _GLM5_MINI_CFG,
    dtype="bfloat16",
)

_GLM5_FULL_SCALE_CFG = glm5_3_flash.Glm5Config(
    kda_implementation="xla",
    dtype="bfloat16",
    weight_dtype="float32",
)


def _build_glm5_with_cfg(
    base_cfg: glm5_3_flash.Glm5Config,
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a 10-layer mini `Glm5Model` with `base_cfg` (`m3/README.md` Invariants (1)-(7))."""
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  if not isinstance(scan_layers, bool):
    raise TypeError(
        f"scan_layers must be bool, got {type(scan_layers).__name__}.",
    )
  if remat_policy not in ("none", "full"):
    raise ValueError(
        f"remat_policy must be 'none' or 'full', got {remat_policy!r}.",
    )
  cfg = dataclasses.replace(
      base_cfg,
      scan_layers=scan_layers,
      remat_policy=remat_policy,
  )
  with jax.sharding.set_mesh(mesh):
    model = modeling_glm5_3_flash.Glm5Model(
        cfg,
        mesh=mesh,
        rngs=rngs,
        sharding=None,
        place_on_devices=True,
    )
    if remat_policy == "full":
      modeling_glm5_3_flash.apply_glm5_remat(
          model,
          policy=jax.checkpoint_policies.nothing_saveable,
      )
    return model


def _build_glm5_model(
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a 10-layer mini `fp32` `Glm5Model` (`m3/README.md` Invariants (1)-(7))."""
  return _build_glm5_with_cfg(_GLM5_MINI_CFG, mesh, rngs, scan_layers, remat_policy)


def _build_glm5_bf16_model(
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a 10-layer mini `bfloat16` `Glm5Model` for Check 9."""
  return _build_glm5_with_cfg(_GLM5_MINI_BF16_CFG, mesh, rngs, scan_layers, remat_policy)


def _build_glm5_full_scale_abstract_model(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Evaluates abstract shapes and `PartitionSpec`s for full-scale 45-layer `Glm5Model` (0 HBM)."""
  cfg = dataclasses.replace(
      _GLM5_FULL_SCALE_CFG,
      scan_layers=scan_layers,
      remat_policy=remat_policy,
  )
  sharding = glm5_3_flash.Glm5Sharding(
      use_tp=mesh.shape.get("tensor", 1) > 1,
      ep_size=1,
  )
  return nnx.eval_shape(
      lambda: modeling_glm5_3_flash.Glm5Model(
          cfg,
          mesh=None,
          rngs=nnx.Rngs(0),
          sharding=sharding,
          place_on_devices=False,
      )
  )


def _build_glm5_batch(
    key: jax.Array,
    mesh: jax.sharding.Mesh,
) -> Mapping[str, jax.Array]:
  """Builds sharded `(2, 8)` tokens and segment IDs (`m3/README.md` (5)-(6))."""
  raw_tokens = jax.random.randint(
      key,
      (_BATCH_SIZE, _SEQ_LEN),
      1,
      _VOCAB_SIZE,
      dtype=jnp.int32,
  )
  col_idx = jnp.arange(_SEQ_LEN, dtype=jnp.int32)
  row_lengths = jnp.array(
      [_SEQ_LEN, _ROW1_VALID_LEN],
      dtype=jnp.int32,
  )[:, None]
  segment_ids = (col_idx[None, :] < row_lengths).astype(jnp.int32)
  input_ids = jnp.where(segment_ids != 0, raw_tokens, 0)
  batch_spec = glm5_3_flash.Glm5Sharding.from_mesh(mesh)(
      "batch_tokens",
      ("batch", "length"),
      m3_sharding.TensorType.ACTIVATION,
  )
  batch_sharding = jax.sharding.NamedSharding(mesh, batch_spec)
  return {
      "input_ids": jax.device_put(input_ids, batch_sharding),
      "segment_ids": jax.device_put(segment_ids, batch_sharding),
  }


def _forward_glm5(
    model: nnx.Module,
    batch: Mapping[str, jax.Array],
) -> spec.ForwardOutput:
  """Executes `Glm5Model.forward_with_aux` (`m3/README.md` (2), (4), (7))."""
  if not isinstance(model, modeling_glm5_3_flash.Glm5Model):
    raise TypeError(f"Expected Glm5Model, got {type(model).__name__}.")
  out = model.forward_with_aux(
      batch["input_ids"],
      decoder_segment_ids=batch["segment_ids"],
  )
  if out.logits is None:
    raise RuntimeError("out.logits is None in _forward_glm5.")
  return spec.ForwardOutput(
      primary=out.logits,
      aux_losses={"moe_lb_loss": out.aux["moe_lb_loss"]},
  )


def _loss_glm5(
    out: spec.ForwardOutput,
    batch: Mapping[str, jax.Array],
    aux_weights: Mapping[str, float],
) -> jax.Array:
  """Folds `moe_lb_loss` into fp32 cross-entropy (`m3/README.md` (4))."""
  targets = jnp.roll(batch["input_ids"], shift=-1, axis=-1)
  mask = (batch["segment_ids"] != 0).astype(jnp.float32)
  log_probs = jax.nn.log_softmax(out.primary.astype(jnp.float32), axis=-1)
  token_nll = -jnp.take_along_axis(
      log_probs,
      targets[..., None],
      axis=-1,
  ).squeeze(-1)
  primary_loss = jnp.sum(token_nll * mask) / jnp.maximum(jnp.sum(mask), 1.0)
  aux_loss = out.aux_losses["moe_lb_loss"].astype(jnp.float32)
  aux_term = aux_weights["moe_lb_loss"] * aux_loss
  return primary_loss + aux_term


def make_glm5_3_flash_conformance_spec() -> spec.ConformanceSpec:
  """Constructs `glm5.3-flash` `ConformanceSpec` (`m3/README.md` (1)-(7))."""
  return spec.ConformanceSpec(
      model_name="glm5.3-flash",
      output_kind=spec.OutputKind.LOGITS,
      build_model=_build_glm5_model,
      build_batch=_build_glm5_batch,
      forward_fn=_forward_glm5,
      loss_fn=_loss_glm5,
      frozen_param_patterns=(r"\.self_attn\.indexer\.",),
      aux_losses={"moe_lb_loss": (r"\.mlp\.gate\.weight$",)},
      stacked_param_path_patterns=(r"^super_block\.",),
      num_scanned_blocks=_NUM_SCANNED_BLOCKS,
      min_sharded_param_elements=_MIN_SHARDED_PARAM_ELEMENTS,
      expert_weight_param_patterns=_EXPERT_WEIGHT_PATTERNS,
      build_bf16_model=_build_glm5_bf16_model,
      bf16_atol=_BF16_ATOL,
      bf16_rtol=_BF16_RTOL,
      build_full_scale_abstract_model=_build_glm5_full_scale_abstract_model,
      skips={
          spec.CheckId.BF16_FORWARD_PARITY: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Glm5Model (modeling_glm5_3_flash.py:217-282, 309-395, 1974,"
              " 2004) initializes Glm5Embedding, Glm5KdaAttention projections,"
              " and lm_head with default lecun_normal (std=0.125) instead of"
              " config.initializer_range=0.02 and normalizes bfloat16 KDA"
              " recurrence outputs via Glm5RMSNormGated, accumulating 0.719"
              " hidden-state drift and 1.039 logit drift (> bf16_atol=0.05) at"
              " unscaled random initialization (b/585591392).",
          ),
          spec.CheckId.PACKING_AND_PADDING_ISOLATION: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Glm5Model (glm5_3_flash_config.py:36-42, 131;"
              " modeling_glm5_3_flash.py:779) supports unpacked {0, 1}"
              " decoder_segment_ids only (multi-document packing with"
              " segment_ids > 1 is not yet implemented; b/585591392).",
          ),
          spec.CheckId.EXPERT_PARALLEL_PARITY: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Glm5Experts and Glm5MoE (modeling_glm5_3_flash.py:960-971,"
              " 1065-1078) raise NotImplementedError when ep_size > 1 on the"
              " 'expert' mesh axis (b/585591392).",
          ),
      },
  )


spec.register_conformance_spec(make_glm5_3_flash_conformance_spec)
