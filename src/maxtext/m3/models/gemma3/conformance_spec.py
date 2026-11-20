# Copyright 2026 Google LLC
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

"""Conformance spec for Gemma 3 text models (`m3/README.md` Invariants (1)-(7))."""

from collections.abc import Mapping
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.gemma3 import modeling_gemma3
from tests.unit.m3.conformance import spec

_BATCH_SIZE = 2
_SEQ_LEN = 8
_VOCAB_SIZE = 256
# Two 6-layer cycles (5 local sliding-window + 1 global each) so the stacked cycle axis has length 2.
_NUM_LAYERS = 12
_NUM_SCANNED_BLOCKS = 2

_GEMMA3_MINI_CFG = modeling_gemma3.Gemma3Config(
    vocab_size=_VOCAB_SIZE,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=_NUM_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    query_pre_attn_scalar=16.0,
    sliding_window=4,
    rope_linear_scaling_factor=8.0,
    final_logit_softcapping=30.0,
    dtype="float32",
    weight_dtype="float32",
    matmul_precision="highest",
    stack_cycles=True,
    param_scan_axis=0,
)

_GEMMA3_MINI_BF16_CFG = dataclasses.replace(
    _GEMMA3_MINI_CFG,
    dtype="bfloat16",
    matmul_precision="default",
)


def _build_gemma3_with_cfg(
    base_cfg: modeling_gemma3.Gemma3Config,
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a 12-layer mini `Gemma3ForCausalLM` on `mesh` with the requested scan and remat modes."""
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  cfg = modeling_gemma3.Gemma3Config.from_config(
      base_cfg,
      scan_layers=scan_layers,
      remat_policy=remat_policy,
  )
  with jax.sharding.set_mesh(mesh):
    return modeling_gemma3.Gemma3ForCausalLM(cfg, mesh, rngs=rngs)


def _build_gemma3_model(
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a 12-layer mini `fp32` `Gemma3ForCausalLM` on `mesh`."""
  return _build_gemma3_with_cfg(_GEMMA3_MINI_CFG, mesh, rngs, scan_layers, remat_policy)


def _build_gemma3_bf16_model(
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a 12-layer mini `bfloat16` `Gemma3ForCausalLM` on `mesh` for Check 9a."""
  return _build_gemma3_with_cfg(_GEMMA3_MINI_BF16_CFG, mesh, rngs, scan_layers, remat_policy)


def _build_gemma3_full_scale_abstract_model(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Evaluates abstract shapes and `PartitionSpec`s for full-scale 26-layer `gemma3-1b` (0 HBM)."""
  cfg = modeling_gemma3.Gemma3Config.preset(
      "gemma3-1b",
      dtype="bfloat16",
      weight_dtype="float32",
      scan_layers=scan_layers,
      stack_cycles=True,
      remat_policy=remat_policy,
  )
  sharding = modeling_gemma3.Gemma3Sharding.from_mesh(mesh, cfg)
  build_mesh = mesh if jax.sharding.get_abstract_mesh().empty else None
  return nnx.eval_shape(
      lambda: modeling_gemma3.Gemma3ForCausalLM(cfg, mesh=build_mesh, rngs=nnx.Rngs(0), sharding=sharding)
  )


def _build_gemma3_batch(
    key: jax.Array,
    mesh: jax.sharding.Mesh,
) -> Mapping[str, jax.Array]:
  """Builds token, position, and segment IDs; `_SEQ_LEN > sliding_window` exercises local masking."""
  del mesh
  input_ids = jax.random.randint(key, (_BATCH_SIZE, _SEQ_LEN), 1, _VOCAB_SIZE, dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(_SEQ_LEN, dtype=jnp.int32), (_BATCH_SIZE, _SEQ_LEN))
  segment_ids = jnp.ones((_BATCH_SIZE, _SEQ_LEN), dtype=jnp.int32)
  return {"input_ids": input_ids, "positions": positions, "segment_ids": segment_ids}


def _forward_gemma3(
    model: nnx.Module,
    batch: Mapping[str, jax.Array],
) -> spec.ForwardOutput:
  """Runs the unmodified `Gemma3ForCausalLM` forward pass."""
  if not isinstance(model, modeling_gemma3.Gemma3ForCausalLM):
    raise TypeError(f"Expected Gemma3ForCausalLM, got {type(model).__name__}.")
  logits = model(
      batch["input_ids"],
      batch["positions"],
      decoder_segment_ids=batch["segment_ids"],
  )
  return spec.ForwardOutput(primary=logits)


def _loss_gemma3(
    out: spec.ForwardOutput,
    batch: Mapping[str, jax.Array],
    aux_weights: Mapping[str, float],
) -> jax.Array:
  """Computes masked fp32 next-token cross-entropy (`m3/README.md` Invariant (4))."""
  del aux_weights
  targets = jnp.roll(batch["input_ids"], shift=-1, axis=-1)
  valid_mask = (batch["segment_ids"] != 0).astype(jnp.float32)
  log_probs = jax.nn.log_softmax(out.primary.astype(jnp.float32), axis=-1)
  token_nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
  return jnp.sum(token_nll * valid_mask) / jnp.maximum(jnp.sum(valid_mask), 1.0)


def make_gemma3_conformance_spec() -> spec.ConformanceSpec:
  """Constructs the `gemma3-1b` `ConformanceSpec`."""
  return spec.ConformanceSpec(
      model_name="gemma3-1b",
      output_kind=spec.OutputKind.LOGITS,
      build_model=_build_gemma3_model,
      build_batch=_build_gemma3_batch,
      forward_fn=_forward_gemma3,
      loss_fn=_loss_gemma3,
      stacked_param_path_patterns=(r"^decoder\.layers\.",),
      num_scanned_blocks=_NUM_SCANNED_BLOCKS,
      build_bf16_model=_build_gemma3_bf16_model,
      build_full_scale_abstract_model=_build_gemma3_full_scale_abstract_model,
      skips={
          spec.CheckId.AUX_LOSS_GRAD_COUPLING: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Gemma 3 text models are dense transformers with no router or auxiliary loss.",
          ),
          spec.CheckId.NO_EXPERT_WEIGHT_ALL_GATHER: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Gemma 3 text models use a dense GeGLU MLP with no expert weights.",
          ),
          spec.CheckId.PACKING_AND_PADDING_ISOLATION: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Gemma3Attention isolates packed segments and excludes segment_ids == 0, while Gemma3ForCausalLM "
              "preserves raw residual logits on pad slots to match Hugging Face Gemma3ForCausalLM (b/585591291).",
          ),
          spec.CheckId.EXPERT_PARALLEL_PARITY: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Gemma 3 text models use a dense GeGLU MLP with no MoE expert parallelism.",
          ),
      },
  )


spec.register_conformance_spec(make_gemma3_conformance_spec)
