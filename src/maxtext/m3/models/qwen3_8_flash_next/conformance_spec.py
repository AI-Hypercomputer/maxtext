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

"""Conformance spec for Qwen3.8-Flash-Next (`m3/README.md` Invariants (1)-(7))."""

from collections.abc import Mapping
import types

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next as qwen3_8
from tests.unit.m3.conformance import spec

_BATCH_SIZE = 2
_SEQ_LEN = 8
_ROW1_VALID_LEN = 7
_VOCAB_SIZE = 256
_NUM_EXPERTS = 4
_NUM_CYCLES = 3
_PLE_LAYER_ID = 1
# Covers every 2D+ weight of the mini model down to the `(64, 1)` shared-expert gate.
_MIN_SHARDED_PARAM_ELEMENTS = 64

# Every dimension a sharding rule can land on is even, so the 2x2 driver mesh divides all of them.
_QWEN3_8_MINI_CFG = {
    "vocab_size": _VOCAB_SIZE,
    "emb_dim": 64,
    "num_decoder_layers": 4 * _NUM_CYCLES,
    "inhomogeneous_layer_cycle_interval": 4,
    "normalization_layer_epsilon": 1e-6,
    "dtype": "float32",
    "weight_dtype": "float32",
    "hc_count": 2,
    "hc_lowrank": 16,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 4,
    "gdn_key_head_dim": 16,
    "gdn_value_head_dim": 16,
    "gdn_conv_kernel_dim": 4,
    "gdn_chunk_size": 16,
    "gdn_output_gate_activation": "sigmoid",
    "gdn_chunk_inverse": "block_doubling",
    "num_query_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 16,
    "partial_rotary_factor": 0.25,
    "rope_max_timescale": 1e7,
    "mrope_section": (1, 1, 0),
    "index_n_heads": 2,
    "index_kv_heads": 1,
    "index_head_dim": 16,
    "index_select_budget": 8,
    "index_select_block_size": 4,
    "qsa_indexer_mask": "gather",
    "qsa_attention_kernel": "dense",
    "mlp_dim": 32,
    "moe_mlp_dim": 32,
    "num_experts": _NUM_EXPERTS,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "sparse_matmul": True,
    # Selects the `ragged_dot` executor, matching the GLM-5.3 conformance spec: `_sharded_grouped` (`gmm_v2`)
    # enters `shard_map` with `P(experts, None, None, mlp)` (unsharded along `expert_embed`), which Check 8
    # flags on the 2x2 `fsdp=2, tensor=2` driver mesh where `expert=1`.
    "megablox": False,
    "use_tokamax_gmm": False,
    "use_ple_layer": True,
    "ple_layer_id": _PLE_LAYER_ID,
    "ple_embed_dim": 64,
    "ple_conv_kernel_size": 4,
    "ple_eos_token_id": 3,
    "ple_seed": 1234,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 500,
    "make_ngram_vocab_size_divisible_by": 4,
    "mtp_num_layers": 1,
    "mtp_loss_scaling_factor": 0.1,
    "load_balance_loss_weight": 0.001,
}


def _randomize_ple_conv_kernel(model: qwen3_8.Qwen3_8FlashNextModel) -> None:
  """Replaces the zero-initialized PLE conv kernel so `norm_conv`, which only feeds it, receives gradient."""
  ple = getattr(model.decoder.first_cycle, f"gdn_{_PLE_LAYER_ID}").ple
  kernel = ple.conv1d.kernel
  value = kernel.get_value()
  noise = 0.02 * jax.random.normal(jax.random.key(0), value.shape, value.dtype)
  kernel.set_value(jax.device_put(noise, value.sharding))


def _build_qwen3_8_model(
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a 3-cycle (`first_cycle` + 2 stacked `cycles`) mini `Qwen3_8FlashNextModel` on `mesh`."""
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  if not isinstance(scan_layers, bool):
    raise TypeError(f"scan_layers must be bool, got {type(scan_layers).__name__}.")
  if remat_policy not in ("none", "full"):
    raise ValueError(f"remat_policy must be 'none' or 'full', got {remat_policy!r}.")
  cfg = types.SimpleNamespace(**_QWEN3_8_MINI_CFG, scan_layers=scan_layers, remat_policy=remat_policy)
  with jax.sharding.set_mesh(mesh):
    model = qwen3_8.Qwen3_8FlashNextModel(cfg, mesh, rngs=rngs)
    _randomize_ple_conv_kernel(model)
    return model


def _build_qwen3_8_batch(
    key: jax.Array,
    mesh: jax.sharding.Mesh,
) -> Mapping[str, jax.Array]:
  """Builds batch-sharded `(2, 16)` tokens and segment IDs with a padded second row."""
  raw_tokens = jax.random.randint(key, (_BATCH_SIZE, _SEQ_LEN), 4, _VOCAB_SIZE, dtype=jnp.int32)
  row_lengths = jnp.array([_SEQ_LEN, _ROW1_VALID_LEN], dtype=jnp.int32)[:, None]
  segment_ids = (jnp.arange(_SEQ_LEN, dtype=jnp.int32)[None, :] < row_lengths).astype(jnp.int32)
  input_ids = jnp.where(segment_ids != 0, raw_tokens, 0)
  batch_spec = m3_sharding.DefaultSharding()("batch_tokens", ("batch", None), m3_sharding.TensorType.ACTIVATION)
  batch_sharding = jax.sharding.NamedSharding(mesh, m3_sharding.project_spec_to_mesh(batch_spec, mesh))
  return {
      "input_ids": jax.device_put(input_ids, batch_sharding),
      "segment_ids": jax.device_put(segment_ids, batch_sharding),
  }


def _forward_qwen3_8(
    model: nnx.Module,
    batch: Mapping[str, jax.Array],
) -> spec.ForwardOutput:
  """Runs the backbone and the MTP draft head, returning both auxiliary losses (`m3/README.md` (2), (4))."""
  if not isinstance(model, qwen3_8.Qwen3_8FlashNextModel):
    raise TypeError(f"Expected Qwen3_8FlashNextModel, got {type(model).__name__}.")
  tokens, segment_ids = batch["input_ids"], batch["segment_ids"]
  logits, aux = model(tokens, None, segment_ids, return_aux=True, return_mtp=True)
  targets = jnp.roll(tokens, shift=-1, axis=-1)
  # The draft head at position `t` predicts token `t + 2`, within the same document.
  mtp_mask = ((segment_ids[:, :-1] == segment_ids[:, 1:]) & (segment_ids[:, :-1] != 0)).astype(jnp.float32)
  mtp_loss = qwen3_8.cross_entropy_loss(aux["mtp_logits"][:, :-1, :], targets[:, 1:], mask=mtp_mask)
  lb_loss = qwen3_8.router_load_balancing_loss(
      aux, _NUM_EXPERTS, mtp_mask=jnp.pad(mtp_mask, ((0, 0), (0, 1))), segment_ids=segment_ids
  )
  return spec.ForwardOutput(primary=logits, aux_losses={"moe_lb_loss": lb_loss, "mtp_loss": mtp_loss})


def _loss_qwen3_8(
    out: spec.ForwardOutput,
    batch: Mapping[str, jax.Array],
    aux_weights: Mapping[str, float],
) -> jax.Array:
  """Folds the router load-balancing and MTP losses into fp32 cross-entropy (`m3/README.md` (4))."""
  targets = jnp.roll(batch["input_ids"], shift=-1, axis=-1)
  loss = qwen3_8.cross_entropy_loss(out.primary, targets, mask=batch["segment_ids"].astype(jnp.float32))
  for name, weight in aux_weights.items():
    loss = loss + weight * out.aux_losses[name].astype(jnp.float32)
  return loss


def make_qwen3_8_flash_next_conformance_spec() -> spec.ConformanceSpec:
  """Constructs the `qwen3.8-flash-next` `ConformanceSpec` (`m3/README.md` (1)-(7))."""
  return spec.ConformanceSpec(
      model_name="qwen3.8-flash-next",
      output_kind=spec.OutputKind.LOGITS,
      build_model=_build_qwen3_8_model,
      build_batch=_build_qwen3_8_batch,
      forward_fn=_forward_qwen3_8,
      loss_fn=_loss_qwen3_8,
      # Both `scan_layers=False` and `scan_layers=True` share the always-stacked (`first_cycle` + `cycles`)
      # parameter tree (`m3/README.md` Invariant (3)), so all 4 `(scan_layers, remat_policy)` variants are checked.
      scan_remat_variants=(
          (False, "none"),
          (True, "none"),
          (False, "full"),
          (True, "full"),
      ),
      # The indexer picks blocks through a non-differentiable top-k and has no training loss of its own.
      frozen_param_patterns=(r"\.self_attn\.indexer\.",),
      aux_losses={
          "moe_lb_loss": (r"\.mlp\.gate\.linear\.kernel$",),
          "mtp_loss": (r"^mtp\.(?!.*\.indexer\.)",),
      },
      stacked_param_path_patterns=(r"^decoder\.cycles\.",),
      num_scanned_blocks=_NUM_CYCLES - 1,
      min_sharded_param_elements=_MIN_SHARDED_PARAM_ELEMENTS,
      expert_weight_param_patterns=(r"\.mlp\.experts\.(gate_up_proj|down_proj)$",),
      skips={},
  )


spec.register_conformance_spec(make_qwen3_8_flash_next_conformance_spec)
