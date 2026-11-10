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

"""Conformance spec for Qwen3-0.6B (`m3/README.md` Invariants (1)-(7))."""

from collections.abc import Mapping
import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen3 import modeling_qwen3
from tests.unit.m3.conformance import spec

_BATCH_SIZE = 2
_SEQ_LEN = 8
_VOCAB_SIZE = 256


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Qwen3MiniConfig:
  """Minimal Qwen3 config matching `modeling_qwen3.py:58-446` attributes."""

  vocab_size: int
  emb_dim: int
  mlp_dim: int
  num_decoder_layers: int
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  normalization_layer_epsilon: float
  rope_max_timescale: float
  use_qk_norm: bool
  float32_logits: bool
  logits_dot_in_fp32: bool
  normalize_embedding_logits: bool
  final_logits_soft_cap: float | None
  cast_logits_to_fp32: bool
  dtype: Any
  weight_dtype: Any


_QWEN3_MINI_CFG = _Qwen3MiniConfig(
    vocab_size=_VOCAB_SIZE,
    emb_dim=64,
    mlp_dim=128,
    num_decoder_layers=2,
    num_query_heads=4,
    num_kv_heads=2,
    head_dim=16,
    normalization_layer_epsilon=1e-6,
    rope_max_timescale=10000.0,
    use_qk_norm=True,
    float32_logits=True,
    logits_dot_in_fp32=True,
    normalize_embedding_logits=False,
    final_logits_soft_cap=None,
    cast_logits_to_fp32=True,
    dtype=jnp.float32,
    weight_dtype=jnp.float32,
)


def _build_qwen3_model(
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    scan_layers: bool,
    remat_policy: str,
) -> nnx.Module:
  """Builds a mini `Qwen3Model` on `mesh` (`m3/README.md` (1), (2), (7))."""
  del scan_layers, remat_policy
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  with jax.sharding.set_mesh(mesh):
    return modeling_qwen3.Qwen3Model(_QWEN3_MINI_CFG, mesh, rngs=rngs)


def _build_qwen3_batch(
    key: jax.Array,
    mesh: jax.sharding.Mesh,
) -> Mapping[str, jax.Array]:
  """Builds token, position, and segment IDs (`m3/README.md` Invariant (1))."""
  del mesh
  input_ids = jax.random.randint(
      key,
      (_BATCH_SIZE, _SEQ_LEN),
      0,
      _VOCAB_SIZE,
      dtype=jnp.int32,
  )
  positions = jnp.broadcast_to(
      jnp.arange(_SEQ_LEN, dtype=jnp.int32),
      (_BATCH_SIZE, _SEQ_LEN),
  )
  segment_ids = jnp.ones((_BATCH_SIZE, _SEQ_LEN), dtype=jnp.int32)
  return {
      "input_ids": input_ids,
      "positions": positions,
      "segment_ids": segment_ids,
  }


def _forward_qwen3(
    model: nnx.Module,
    batch: Mapping[str, jax.Array],
) -> spec.ForwardOutput:
  """Executes `Qwen3Model` forward pass (`m3/README.md` Invariants (2), (7))."""
  if not isinstance(model, modeling_qwen3.Qwen3Model):
    raise TypeError(f"Expected Qwen3Model, got {type(model).__name__}.")
  logits = model(
      batch["input_ids"],
      batch["positions"],
      decoder_segment_ids=batch["segment_ids"],
  )
  return spec.ForwardOutput(primary=logits)


def _loss_qwen3(
    out: spec.ForwardOutput,
    batch: Mapping[str, jax.Array],
    aux_weights: Mapping[str, float],
) -> jax.Array:
  """Computes fp32 cross-entropy loss (`m3/README.md` Invariant (4))."""
  del aux_weights
  targets = jnp.roll(batch["input_ids"], shift=-1, axis=-1)
  log_probs = jax.nn.log_softmax(out.primary.astype(jnp.float32), axis=-1)
  token_nll = -jnp.take_along_axis(
      log_probs,
      targets[..., None],
      axis=-1,
  ).squeeze(-1)
  return jnp.mean(token_nll)


def make_qwen3_conformance_spec() -> spec.ConformanceSpec:
  """Constructs the `qwen3-0.6b` `ConformanceSpec` (`m3/README.md` (1)-(7))."""
  return spec.ConformanceSpec(
      model_name="qwen3-0.6b",
      output_kind=spec.OutputKind.LOGITS,
      build_model=_build_qwen3_model,
      build_batch=_build_qwen3_batch,
      forward_fn=_forward_qwen3,
      loss_fn=_loss_qwen3,
      skips={
          spec.CheckId.SCAN_REMAT_EQUIVALENCE: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Qwen3Decoder (modeling_qwen3.py:339-345, 402-409) uses flat"
              " layers_0..N attributes without scan_layers or remat_policy.",
          ),
          spec.CheckId.AUX_LOSS_GRAD_COUPLING: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Qwen3Model (modeling_qwen3.py:37-88, 458-478) is a dense"
              " transformer without MoE routing or auxiliary losses.",
          ),
          spec.CheckId.DRIVER_MESH_AND_AXIS_NAMES: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Qwen3Model (modeling_qwen3.py:418-446) does not store"
              " model.mesh or attach NamedSharding metadata to parameters.",
          ),
          spec.CheckId.WEIGHT_SHARDING_COVERAGE: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Qwen3MLP (modeling_qwen3.py:60-80), Qwen3Attention"
              " (modeling_qwen3.py:117-159), and Qwen3Model"
              " (modeling_qwen3.py:436-446) omit parameter sharding specs.",
          ),
          spec.CheckId.NO_EXPERT_WEIGHT_ALL_GATHER: spec.SkipDeclaration(
              spec.SkipStatus.NOT_APPLICABLE,
              "Qwen3DecoderLayer (modeling_qwen3.py:244-289) uses dense"
              " Qwen3MLP (modeling_qwen3.py:37-88) with no MoE expert weights.",
          ),
      },
  )


spec.register_conformance_spec(make_qwen3_conformance_spec)
