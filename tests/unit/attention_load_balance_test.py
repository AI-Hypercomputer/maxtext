# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks attention-local causal load balancing (`context_parallel_attention_load_balance`) on a CPU mesh.

The switch runs each attention layer in DUAL_CHUNK_SWAP token order and restores natural order on its
output, so GatedDeltaNet layers keep the natural order their recurrence needs. The CPU checks run in a
subprocess so the forced device count takes effect before JAX initializes:

  - Reorder. The two-`ppermute` reorder equals `max_utils.reorder_sequence` bit for bit, inverts
    exactly, transposes to the inverse in the backward, and lowers to exactly two collective-permutes.
  - Ring kernel. The Tokamax ring kernel (Pallas interpret mode) at context=4 fed load-balanced Q/K/V,
    with the load-balanced mask, matches the plain ring in natural order: output and dQ/dK/dV.
  - Model. A tiny hybrid Qwen3.5 (GatedDeltaNet + gated full attention, MoE, MRoPE) on a
    context=4 x expert=2 cp-as-ep mesh: loss and every parameter gradient match with the switch off.
  - Dense model. A tiny dense decoder (attention and MLP only) on an fsdp=2 x context=4 mesh: logits and
    gradients match with the switch off, and the forward is bit-identical to the existing global path
    (`context_parallel_load_balance`) run on a batch permuted up front.
  - Non-causal layers. A Qwen3.5 vision-encoder attention layer (bidirectional, rotary angles from the
    image grid) is not reordered: its output is bit-identical with the switch on.

"Match" means bitwise for the reorder and a normalized max error max|a - b| / max|b| of at most 1e-5 in
float32 for the kernel. For the hybrid model it is 2x a null control measured in the same run, because
its GatedDeltaNet layers amplify float32 reduction-order noise to ~1e-5 (see `_run_model_checks`); its
logits pass on the 1e-5 floor of that gate. The dense model has no such amplifier, so it is gated at a
fixed 1e-6. Either way the paths differ only in reduction order. Each check has a positive control, a
deliberately wrong reorder or mask, that must fail the same gate; otherwise a gate that cannot fail would
pass.
"""

# pylint: disable=protected-access

import contextlib
import dataclasses
import functools
import os
import re
import subprocess
import sys
import types
import unittest.mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pydantic
import pytest

from maxtext.configs import pyconfig
from maxtext.kernels.attention import tokamax_ring_attention
from maxtext.utils import max_utils
from maxtext.utils import globals as maxtext_globals

_BASE_CONFIG_PATH = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
# Reduction-order gate: normalized max error in float32.
_TOLERANCE = 1e-5
# The same gate for the dense model, which has no GatedDeltaNet layers to amplify float32 rounding.
_DENSE_TOLERANCE = 1e-6
_CHILD_MARKER = "ATTENTION_LOAD_BALANCE_CHECKS_PASSED"


def _normalized_max_error(actual, expected) -> float:
  actual = np.asarray(actual, np.float64)
  expected = np.asarray(expected, np.float64)
  return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-30))


def _count_hlo_ops(hlo_text: str, op: str) -> int:
  return len(re.findall(rf"= \S+ {op}(?:-start)?\(", hlo_text))


def _roll_one_chunk(x, context_parallel_size: int, seq_dim: int):
  """A deliberately wrong inverse: the right one, then shifted by one DUAL_CHUNK_SWAP chunk."""
  return jnp.roll(x, x.shape[seq_dim] // (2 * context_parallel_size), axis=seq_dim)


# ---------------------------------------------------------------------------
# 1. The reorder itself.
# ---------------------------------------------------------------------------


def _run_reorder_checks(mesh, batch_axes):
  """The ppermute reorder equals `reorder_sequence`, inverts, transposes, and costs two permutes."""
  cp = mesh.shape["context"]
  seq = 16 * cp
  rng = np.random.default_rng(0)
  reorder = functools.partial(
      tokamax_ring_attention.reorder_for_load_balance, mesh=mesh, axis_name="context", batch_axes=batch_axes
  )
  batch = 4
  cases = {
      "hidden": rng.standard_normal((batch, seq, 8)).astype(np.float32),
      "positions": np.broadcast_to(np.arange(seq, dtype=np.int32), (batch, seq)).copy(),
      "mrope_positions": rng.integers(0, 1000, (batch, seq, 3)).astype(np.int32),
      "batch_of_one": rng.standard_normal((1, seq, 8)).astype(np.float32),
  }
  for name, x in cases.items():
    with jax.set_mesh(mesh):
      balanced = jax.jit(reorder)(x)
      restored = jax.jit(functools.partial(reorder, to_natural=True))(balanced)
    expected = max_utils.reorder_sequence(jnp.asarray(x), cp_size=cp, seq_dim=1)
    assert np.array_equal(np.asarray(balanced), np.asarray(expected)), f"{name}: reorder != reorder_sequence"
    assert np.array_equal(np.asarray(restored), x), f"{name}: inverse reorder is not exact"

  # Backward: the transpose of the reorder is the inverse reorder, exactly.
  x, w = cases["hidden"], rng.standard_normal(cases["hidden"].shape).astype(np.float32)
  with jax.set_mesh(mesh):
    grad = jax.jit(jax.grad(lambda x: jnp.sum(jnp.asarray(w) * reorder(x))))(x)
  expected_grad = max_utils.reorder_sequence(jnp.asarray(w), cp_size=cp, seq_dim=1, to_contiguous=True)
  assert np.array_equal(np.asarray(grad), np.asarray(expected_grad)), "reorder VJP != inverse reorder"

  # Cost: two point-to-point permutes of half a shard, no all-gather / all-to-all.
  sharded = jax.device_put(x, NamedSharding(mesh, P(batch_axes, "context", None)))
  with jax.set_mesh(mesh):
    hlo = jax.jit(reorder).lower(sharded).compile().as_text()
  permutes, gathers, all_to_all = (_count_hlo_ops(hlo, op) for op in ("collective-permute", "all-gather", "all-to-all"))
  assert (permutes, gathers, all_to_all) == (2, 0, 0), f"reorder lowers to {permutes=} {gathers=} {all_to_all=}"

  # Positive control: an inverse that is off by one chunk must fail the bitwise comparison.
  with jax.set_mesh(mesh):
    wrong = _roll_one_chunk(jax.jit(functools.partial(reorder, to_natural=True))(jax.jit(reorder)(x)), cp, 1)
  assert not np.array_equal(np.asarray(wrong), x), "positive control: an off-by-one-chunk inverse went undetected"
  print(f"reorder checks passed: mesh={dict(mesh.shape)} permutes={permutes}", flush=True)


# ---------------------------------------------------------------------------
# 2. The Tokamax ring kernel with the load-balanced mask.
# ---------------------------------------------------------------------------


def _ring_config(block: int):
  return types.SimpleNamespace(
      dq_reduction_steps=3,
      sa_block_q=block,
      sa_block_kv=block,
      sa_block_kv_compute=block,
      sa_block_q_dkv=block,
      sa_block_kv_dkv=block,
      sa_block_kv_dkv_compute=block,
      sa_q_layout="HEAD_DIM_MINOR",
      sa_k_layout="HEAD_DIM_MINOR",
      sa_v_layout="HEAD_DIM_MINOR",
      cost_estimate_flops_fwd=-1,
      cost_estimate_flops_bwd=-1,
      use_splash_scheduler=False,
      ring_scan_unroll=1,
      sa_bwd_dkv_megacore=False,
  )


def _ring_attention_fn(mesh, seq: int, block: int, load_balanced: bool):
  """Mirrors `make_sharded_ring_attention_kernel` + `call_ring_attention`, in Pallas TPU interpret mode.

  The TPU interpreter (`pltpu.InterpretParams`), not the generic `interpret=True` one: the staged-dq
  backward (`dq_reduction_steps=3`) and the GQA dK/dV path accumulate through `input_output_aliases`
  across revisited grid steps, which the generic interpreter does not reproduce. Under it both the
  plain and the balanced ring miss a dense reference by 10-90% in dQ/dK/dV, while the TPU interpreter
  matches it to ~7e-7.
  """
  # pylint: disable=import-outside-toplevel
  from jax.experimental.pallas import tpu as pltpu

  cp = mesh.shape["context"]
  splash_config = tokamax_ring_attention.build_splash_config(
      _ring_config(block), q_seq_len=seq, kv_seq_len=seq, context_parallel_size=cp, load_balanced=load_balanced
  )
  splash_config = dataclasses.replace(splash_config, interpret=pltpu.InterpretParams())
  mask = tokamax_ring_attention._make_causal_mask((seq, seq), cp, load_balanced=load_balanced)
  kernel = tokamax_ring_attention.ring_attention_kernel.make_ring_attention(
      mask, config=splash_config, is_mqa=False, ring_axis="context", q_seq_shards=cp, kv_seq_shards=cp
  )
  qkv_spec, segment_spec = P("fsdp", None, "context", None), P("fsdp", "context")

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(kernel.manual_sharding_spec(), qkv_spec, qkv_spec, qkv_spec, segment_spec),
      out_specs=qkv_spec,
      check_vma=False,
  )
  def ring(kernel, q, k, v, segment_ids):
    return tokamax_ring_attention.call_ring_attention(q, k, v, segment_ids, segment_ids, kernel)

  return lambda q, k, v, segment_ids: ring(kernel, q, k, v, segment_ids), splash_config.dq_reduction_steps


def _dense_causal_attention(q, k, v, segment_ids):
  """Reference: dense causal GQA attention with segment masking, natural token order."""
  group = q.shape[1] // k.shape[1]
  k, v = jnp.repeat(k, group, axis=1), jnp.repeat(v, group, axis=1)
  logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)
  seq = q.shape[2]
  allowed = jnp.tril(jnp.ones((seq, seq), bool))[None, None] & (
      segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
  )
  return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(jnp.where(allowed, logits, -1e30), axis=-1), v)


def _run_ring_kernel_checks(mesh, seq: int, block: int, expect_staged_dq: bool):
  """Balanced ring (reorder -> load-balanced mask -> inverse) == plain ring == dense, forward and backward."""
  cp = mesh.shape["context"]
  batch, q_heads, kv_heads, head_dim = 2, 4, 2, 128
  rng = np.random.default_rng(1)
  q = rng.standard_normal((batch, q_heads, seq, head_dim)).astype(np.float32) / np.sqrt(head_dim)
  k = rng.standard_normal((batch, kv_heads, seq, head_dim)).astype(np.float32)
  v = rng.standard_normal((batch, kv_heads, seq, head_dim)).astype(np.float32)
  w = rng.standard_normal((batch, q_heads, seq, head_dim)).astype(np.float32)
  # Two packed segments per row, with a boundary that is not chunk aligned.
  segment_ids = np.where(np.arange(seq) < (seq * 11) // 16, 1, 2).astype(np.int32)
  segment_ids = np.broadcast_to(segment_ids, (batch, seq)).copy()

  reorder = functools.partial(
      tokamax_ring_attention.reorder_for_load_balance, mesh=mesh, axis_name="context", batch_axes="fsdp"
  )
  plain_ring, plain_dq_steps = _ring_attention_fn(mesh, seq, block, load_balanced=False)
  balanced_ring, balanced_dq_steps = _ring_attention_fn(mesh, seq, block, load_balanced=True)
  # Make sure the case that expects the staged dq reduction gets it.
  assert (plain_dq_steps == 3) == (balanced_dq_steps == 3) == expect_staged_dq, (plain_dq_steps, balanced_dq_steps)

  def dense(q, k, v):
    return _dense_causal_attention(q, k, v, jnp.asarray(segment_ids))

  def plain(q, k, v):
    return plain_ring(q, k, v, segment_ids)

  def balanced(q, k, v, *, inverse_off_by_one_chunk=False, ring=balanced_ring):
    r = functools.partial(reorder, seq_dim=2)
    out = ring(r(q), r(k), r(v), reorder(jnp.asarray(segment_ids)))
    out = r(out, to_natural=True)
    return _roll_one_chunk(out, cp, 2) if inverse_off_by_one_chunk else out

  def value_and_grads(fn, with_grads=True):
    def loss(q, k, v):
      return jnp.sum(fn(q, k, v) * jnp.asarray(w))

    with jax.set_mesh(mesh):
      out = jax.jit(fn)(q, k, v)
      grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v) if with_grads else ()
    return (np.asarray(out), *(np.asarray(g) for g in grads))

  reference = value_and_grads(dense)
  plain_result = value_and_grads(plain)
  balanced_result = value_and_grads(balanced)
  errors = {
      "balanced_vs_plain": [_normalized_max_error(c, r) for c, r in zip(balanced_result, plain_result)],
      "plain_vs_dense": [_normalized_max_error(c, r) for c, r in zip(plain_result, reference)],
      "balanced_vs_dense": [_normalized_max_error(c, r) for c, r in zip(balanced_result, reference)],
  }
  print(
      f"ring kernel seq={seq} block={block} dq_reduction_steps={balanced_dq_steps}: normalized max error "
      f"out/dq/dk/dv = {errors}",
      flush=True,
  )
  for name, values in errors.items():
    assert max(values) <= _TOLERANCE, f"{name}: {values}"

  # Positive controls, forward only, each must fail the same gate:
  #  (a) the output restored with an inverse permutation that is off by one chunk;
  #  (b) load-balanced tokens run under the plain (natural-order) causal mask.
  control_errors = {
      "off_by_one_chunk_inverse": _normalized_max_error(
          value_and_grads(functools.partial(balanced, inverse_off_by_one_chunk=True), with_grads=False)[0],
          plain_result[0],
      ),
      "balanced_tokens_plain_mask": _normalized_max_error(
          value_and_grads(functools.partial(balanced, ring=plain_ring), with_grads=False)[0], plain_result[0]
      ),
  }
  print(f"ring kernel positive controls: {control_errors}", flush=True)
  for name, error in control_errors.items():
    assert error > _TOLERANCE, f"positive control {name} passed the gate ({error}); the gate cannot fail"


# ---------------------------------------------------------------------------
# 3. A tiny hybrid Qwen3.5, end to end.
# ---------------------------------------------------------------------------


def _tiny_qwen3_5_config(attention_load_balance: bool, seq: int, mesh: tuple[int, int, int] = (1, 4, 2)):
  """A tiny hybrid Qwen3.5 on 8 CPU devices; `mesh` is (fsdp, context, expert) under cp-as-ep."""
  fsdp, context, expert = mesh
  argv = [
      None,
      _BASE_CONFIG_PATH,
      "run_name=attention_load_balance_test",
      "model_name=qwen3.5-397b-a17b",
      "override_model_config=true",
      "hardware=cpu",
      "custom_mesh_and_rule=cp-as-ep",
      f"ici_fsdp_parallelism={fsdp}",
      f"ici_context_parallelism={context}",
      f"ici_expert_parallelism={expert}",
      "ici_tensor_parallelism=1",
      "context_parallel_load_balance=False",
      f"context_parallel_attention_load_balance={attention_load_balance}",
      "attention=dot_product",
      "use_gdn_kernel=false",
      "sparse_matmul=false",
      "megablox=false",
      "use_ring_of_experts=false",
      "dtype=float32",
      "weight_dtype=float32",
      "matmul_precision=highest",
      "scan_layers=True",
      "enable_checkpointing=false",
      "enable_dropout=false",
      "dataset_type=synthetic",
      "base_num_decoder_layers=8",  # two blocks of [3 GatedDeltaNet + 1 full attention]
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=64",
      "mrope_section=[3,3,2]",  # rotary_dim / 2 = 64 * 0.25 / 2
      "base_mlp_dim=32",
      "base_moe_mlp_dim=32",
      "num_experts=8",
      "num_experts_per_tok=2",
      "shared_experts=1",
      "gdn_key_head_dim=16",
      "gdn_value_head_dim=16",
      "gdn_num_key_heads=2",
      "gdn_num_value_heads=4",
      "gdn_chunk_size=16",
      "vocab_size=384",
      f"max_target_length={seq}",
      "per_device_batch_size=0.5",
      "use_multimodal=false",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
  ]
  return pyconfig.initialize(argv)


def _model_outputs_and_grads(cfg, data, cotangent, patch=None):
  """Logits, the parameter gradients of <logits, cotangent>, and the compiled HLO of one tiny model.

  A fixed random cotangent on the logits makes the backward per-token sensitive (at random init the
  cross-entropy is nearly flat: a wholesale token misalignment moves it by only ~3e-5).
  """
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.utils import maxtext_utils, maxtext_utils_nnx, model_creation_utils

  mesh = maxtext_utils.get_mesh_from_config(cfg)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=jax.random.PRNGKey(0))
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def logits_of(params):
    merged = nnx.merge(graphdef, params, rest, copy=True)
    logits = merged(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        enable_dropout=False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    return logits.astype(jnp.float32)

  def value_and_grads(params):
    logits, vjp = jax.vjp(logits_of, params)
    return logits, vjp(cotangent)[0]

  with patch or contextlib.nullcontext(), nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    compiled = jax.jit(value_and_grads).lower(params).compile()
    logits, grads = compiled(params)
  leaves = {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}
  return np.asarray(logits), leaves, compiled.as_text()


def _model_errors(logits, grads, ref_logits, ref_grads) -> dict[str, float]:
  """Forward and backward discrepancies against a reference run."""
  assert grads.keys() == ref_grads.keys()
  names = sorted(ref_grads)
  diff = np.sqrt(sum(np.sum((grads[n].astype(np.float64) - ref_grads[n].astype(np.float64)) ** 2) for n in names))
  norm = np.sqrt(sum(np.sum(ref_grads[n].astype(np.float64) ** 2) for n in names))
  embedding = next(n for n in names if "token_embedder" in n)
  return {
      "logits": _normalized_max_error(logits, ref_logits),
      "grads_rel_l2": float(diff / norm),
      "embedding_grad": _normalized_max_error(grads[embedding], ref_grads[embedding]),
  }


def _run_model_checks(seq: int):
  """Tiny hybrid Qwen3.5: attention-local balance on == off, in the logits and in every gradient.

  The gate is calibrated in the same run by a null control: the same model with the switch off on a
  context=2 mesh instead of context=4. That change is mathematically exact but reorders the attention
  and GatedDeltaNet reductions, so its discrepancy is this model's float32 reduction-order noise floor
  (measured ~1e-5 on the logits and the embedding gradient, and ~5e-5 on single weight leaves, so a
  fixed 1e-5 cannot tell exact from inexact here). Balanced-vs-natural must stay within 2x of that
  floor on every detector, and the positive control must exceed the gate by 1000x.
  """
  off_cfg = _tiny_qwen3_5_config(False, seq)
  on_cfg = _tiny_qwen3_5_config(True, seq)
  null_cfg = _tiny_qwen3_5_config(False, seq, mesh=(2, 2, 2))
  batch = off_cfg.micro_batch_size_to_train_on
  assert null_cfg.micro_batch_size_to_train_on == batch
  tokens = jax.random.randint(jax.random.PRNGKey(7), (batch, seq + 1), 0, off_cfg.vocab_size)
  # Two packed segments per row so segment ids and position resets travel through the reorder.
  boundary = (seq * 5) // 8
  segmentation = jnp.where(jnp.arange(seq) < boundary, 1, 2).astype(jnp.int32)
  positions = jnp.where(jnp.arange(seq) < boundary, jnp.arange(seq), jnp.arange(seq) - boundary).astype(jnp.int32)
  data = {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "inputs_position": jnp.broadcast_to(positions, (batch, seq)),
      "inputs_segmentation": jnp.broadcast_to(segmentation, (batch, seq)),
      "targets_segmentation": jnp.broadcast_to(segmentation, (batch, seq)),
  }
  cotangent = jax.random.normal(jax.random.PRNGKey(3), (batch, seq, off_cfg.vocab_size), jnp.float32)
  off_logits, off_grads, off_hlo = _model_outputs_and_grads(off_cfg, data, cotangent)
  on_logits, on_grads, on_hlo = _model_outputs_and_grads(on_cfg, data, cotangent)
  null_logits, null_grads, _ = _model_outputs_and_grads(null_cfg, data, cotangent)

  # The switch must actually be live: the balanced program carries the extra reorder permutes.
  off_permutes, on_permutes = (_count_hlo_ops(h, "collective-permute") for h in (off_hlo, on_hlo))
  assert on_permutes > off_permutes, f"switch had no effect on the program ({off_permutes=} {on_permutes=})"

  null_errors = _model_errors(null_logits, null_grads, off_logits, off_grads)
  gates = {name: max(_TOLERANCE, 2.0 * error) for name, error in null_errors.items()}
  errors = _model_errors(on_logits, on_grads, off_logits, off_grads)
  print(
      f"model seq={seq} batch={batch}: balanced vs natural {errors}; null control (context=2 mesh) {null_errors}; "
      f"gates {gates}; collective-permutes off={off_permutes} on={on_permutes}",
      flush=True,
  )
  for name, error in errors.items():
    assert error <= gates[name], f"{name} differs beyond reduction order: {error} > {gates[name]}"

  # Positive control: restore natural order with an inverse that is off by one chunk. The GatedDeltaNet
  # and MoE layers after each attention layer then see permuted tokens, and every gate must fail.
  bad_logits, bad_grads, _ = _model_outputs_and_grads(on_cfg, data, cotangent, patch=_off_by_one_chunk_inverse())
  bad_errors = _model_errors(bad_logits, bad_grads, off_logits, off_grads)
  print(f"model positive control: {bad_errors}", flush=True)
  for name, error in bad_errors.items():
    assert error > 1000 * gates[name], f"positive control {name}={error} is not 1000x its gate {gates[name]}"


def _off_by_one_chunk_inverse():
  """Patches the reorder so that restoring natural order is off by one chunk."""
  real_reorder = tokamax_ring_attention.reorder_for_load_balance

  def reorder(x, *, mesh, axis_name, batch_axes, to_natural=False, seq_dim=1):
    out = real_reorder(x, mesh=mesh, axis_name=axis_name, batch_axes=batch_axes, to_natural=to_natural, seq_dim=seq_dim)
    return _roll_one_chunk(out, mesh.shape[axis_name], seq_dim) if to_natural and out is not None else out

  return unittest.mock.patch.object(tokamax_ring_attention, "reorder_for_load_balance", reorder)


# ---------------------------------------------------------------------------
# 4. A tiny dense decoder, against the switch off and against the global path.
# ---------------------------------------------------------------------------


def _tiny_dense_config(attention_load_balance: bool, seq: int, global_load_balance: bool = False):
  """A tiny dense decoder (attention and MLP only) on an fsdp=2 x context=4 mesh of 8 CPU devices."""
  argv = [
      None,
      _BASE_CONFIG_PATH,
      "run_name=attention_load_balance_dense_test",
      "hardware=cpu",
      "ici_fsdp_parallelism=2",
      "ici_context_parallelism=4",
      "ici_tensor_parallelism=1",
      f"context_parallel_load_balance={global_load_balance}",
      f"context_parallel_attention_load_balance={attention_load_balance}",
      "attention=dot_product",
      "dtype=float32",
      "weight_dtype=float32",
      "matmul_precision=highest",
      "scan_layers=True",
      "enable_checkpointing=false",
      "enable_dropout=false",
      "dataset_type=synthetic",
      "base_num_decoder_layers=2",
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=16",
      "base_mlp_dim=64",
      "vocab_size=384",
      f"max_target_length={seq}",
      "per_device_batch_size=0.25",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
  ]
  return pyconfig.initialize(argv)


def _natural_order_mutant(field: str):
  """Patches the layer-input reorder to leave one of positions, segment ids or K/V in natural order."""
  # pylint: disable=import-outside-toplevel
  from maxtext.layers import attentions

  real_order = attentions.Attention._to_load_balanced_order

  def to_load_balanced_order(self, inputs_q, inputs_kv, inputs_positions, decoder_segment_ids, *, use_shared_kv):
    q, kv, positions, segment_ids = real_order(
        self, inputs_q, inputs_kv, inputs_positions, decoder_segment_ids, use_shared_kv=use_shared_kv
    )
    if field == "positions":
      positions = inputs_positions
    elif field == "segment_ids":
      segment_ids = decoder_segment_ids
    else:
      kv = inputs_kv
    return q, kv, positions, segment_ids

  return unittest.mock.patch.object(attentions.Attention, "_to_load_balanced_order", to_load_balanced_order)


def _run_dense_model_checks(seq: int):
  """Tiny dense decoder: balanced == natural at a fixed 1e-6, and == the existing global path bit for bit.

  Without GatedDeltaNet layers to amplify float32 rounding, a fixed tolerance separates exact from inexact.
  The sharper comparison is against `context_parallel_load_balance`, which permutes the whole batch up
  front. Both paths then run the same attention on the same permuted tokens with the same mask, and every
  other op acts token by token, so the forward must be bit-identical and the gradients may differ only in
  the order of the weight-gradient sums over tokens. Five deliberately wrong plumbings of the switch must
  each miss the gate by 1e4x.
  """
  # pylint: disable=import-outside-toplevel
  from maxtext.layers import attention_op as attention_op_lib

  off_cfg = _tiny_dense_config(False, seq)
  on_cfg = _tiny_dense_config(True, seq)
  global_cfg = _tiny_dense_config(False, seq, global_load_balance=True)
  cp = on_cfg.ici_context_parallelism
  batch = off_cfg.micro_batch_size_to_train_on
  tokens = jax.random.randint(jax.random.PRNGKey(7), (batch, seq + 1), 0, off_cfg.vocab_size)
  # Two packed segments per row, with a boundary that is not chunk aligned.
  boundary = (seq * 11) // 16
  segmentation = jnp.where(jnp.arange(seq) < boundary, 1, 2).astype(jnp.int32)
  positions = jnp.where(jnp.arange(seq) < boundary, jnp.arange(seq), jnp.arange(seq) - boundary).astype(jnp.int32)
  data = {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "inputs_position": jnp.broadcast_to(positions, (batch, seq)),
      "inputs_segmentation": jnp.broadcast_to(segmentation, (batch, seq)),
      "targets_segmentation": jnp.broadcast_to(segmentation, (batch, seq)),
  }
  cotangent = jax.random.normal(jax.random.PRNGKey(3), (batch, seq, off_cfg.vocab_size), jnp.float32)
  off_logits, off_grads, off_hlo = _model_outputs_and_grads(off_cfg, data, cotangent)
  on_logits, on_grads, on_hlo = _model_outputs_and_grads(on_cfg, data, cotangent)
  off_permutes, on_permutes = (_count_hlo_ops(h, "collective-permute") for h in (off_hlo, on_hlo))
  assert on_permutes > off_permutes, f"switch had no effect on the program ({off_permutes=} {on_permutes=})"
  errors = _model_errors(on_logits, on_grads, off_logits, off_grads)

  # The existing global path, on a batch this test permutes the way the input pipeline would.
  def balanced(x):
    return max_utils.reorder_sequence(x, cp_size=cp, seq_dim=1)

  global_logits, global_grads, _ = _model_outputs_and_grads(
      global_cfg, {name: balanced(x) for name, x in data.items()}, balanced(cotangent)
  )
  global_logits = np.asarray(
      max_utils.reorder_sequence(jnp.asarray(global_logits), cp_size=cp, seq_dim=1, to_contiguous=True)
  )
  global_errors = _model_errors(on_logits, on_grads, global_logits, global_grads)
  forward_bitwise = bool(np.array_equal(on_logits, global_logits))
  print(
      f"dense model seq={seq} batch={batch}: balanced vs natural {errors}; balanced vs global path {global_errors} "
      f"(forward bit-identical: {forward_bitwise}); collective-permutes off={off_permutes} on={on_permutes}",
      flush=True,
  )
  for name, error in errors.items():
    assert error <= _DENSE_TOLERANCE, f"{name} differs beyond reduction order: {error} > {_DENSE_TOLERANCE}"
  assert forward_bitwise, "the balanced forward is not bit-identical to the global path on the same permuted batch"
  for name, error in global_errors.items():
    assert error <= _DENSE_TOLERANCE, f"{name} differs from the global path: {error} > {_DENSE_TOLERANCE}"

  real_predicate = attention_op_lib.AttentionOp._load_balanced_context_parallel
  mutants = {
      "positions_left_natural": _natural_order_mutant("positions"),
      "segment_ids_left_natural": _natural_order_mutant("segment_ids"),
      "kv_left_natural": _natural_order_mutant("kv"),
      "mask_ignores_the_reorder": unittest.mock.patch.object(
          attention_op_lib.AttentionOp,
          "_load_balanced_context_parallel",
          lambda self, sequence_load_balanced=False: real_predicate(self, False),
      ),
      "off_by_one_chunk_inverse": _off_by_one_chunk_inverse(),
  }
  for name, patch in mutants.items():
    bad_logits, bad_grads, _ = _model_outputs_and_grads(on_cfg, data, cotangent, patch=patch)
    bad_errors = _model_errors(bad_logits, bad_grads, off_logits, off_grads)
    print(f"dense model positive control {name}: {bad_errors}", flush=True)
    for detector, error in bad_errors.items():
      assert error > 1e4 * _DENSE_TOLERANCE, f"positive control {name}: {detector}={error} is not 1e4x the gate"


# ---------------------------------------------------------------------------
# 5. Layers that are not causal keep natural order.
# ---------------------------------------------------------------------------


def _vision_attention_config(attention_load_balance: bool):
  """Qwen3.5 with a tiny vision encoder on a context=4 x expert=2 cp-as-ep mesh of 8 CPU devices.

  The switch rejects `use_multimodal=True`, so the vision attention layer is built directly.
  """
  argv = [
      None,
      _BASE_CONFIG_PATH,
      "run_name=attention_load_balance_vision_test",
      "model_name=qwen3.5-397b-a17b",
      "override_model_config=true",
      "hardware=cpu",
      "custom_mesh_and_rule=cp-as-ep",
      "ici_fsdp_parallelism=1",
      "ici_context_parallelism=4",
      "ici_expert_parallelism=2",
      "ici_tensor_parallelism=1",
      "context_parallel_load_balance=False",
      f"context_parallel_attention_load_balance={attention_load_balance}",
      "attention=dot_product",
      "dtype=float32",
      "weight_dtype=float32",
      "dtype_mm=float32",
      "matmul_precision=highest",
      "use_multimodal=false",
      "hidden_size_for_vit=64",
      "num_attention_heads_for_vit=4",
      "intermediate_size_for_vit=64",
      "num_hidden_layers_for_vit=1",
      "base_num_decoder_layers=4",
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=64",
      "mrope_section=[3,3,2]",
      "base_mlp_dim=32",
      "base_moe_mlp_dim=32",
      "num_experts=8",
      "num_experts_per_tok=2",
      "vocab_size=384",
      "max_target_length=256",
      "per_device_batch_size=0.25",
      "enable_checkpointing=false",
      "dataset_type=synthetic",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
  ]
  return pyconfig.initialize(argv)


def _vision_attention_output(cfg, x, patch=None):
  """One Qwen3.5 vision-encoder attention layer on a 1 x 4 x 8 patch grid.

  Returns how many shards its attention op load-balances over, how many it would if the layer were
  causal, its output, and the compiled HLO.
  """
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common.common_types import AttentionType, MODEL_MODE_TRAIN
  from maxtext.models import qwen3
  from maxtext.utils import maxtext_utils

  mesh = maxtext_utils.get_mesh_from_config(cfg)
  with patch or contextlib.nullcontext(), nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    layer = qwen3.Qwen3OmniMoeVisionAttention(config=cfg, mesh=mesh, rngs=nnx.Rngs(0))
    op = layer.attn.attention_op
    size = op.attention_load_balance_context_size(MODEL_MODE_TRAIN)
    attention_type = op.attention_type
    op.attention_type = AttentionType.GLOBAL
    try:
      size_if_causal = op.attention_load_balance_context_size(MODEL_MODE_TRAIN)
    finally:
      op.attention_type = attention_type
    graphdef, state = nnx.split(layer)

    def forward(state, x):
      return nnx.merge(graphdef, state)(x, num_frames=1, height=4, width=8, deterministic=True)

    compiled = jax.jit(forward).lower(state, x).compile()
    out = compiled(state, x)
  return size, size_if_causal, np.asarray(out), compiled.as_text()


def _run_vision_layer_checks():
  """A bidirectional vision-encoder attention layer keeps natural order with the switch on.

  Its rotary angles come from the (frames, height, width) patch grid in natural token order, not from
  positions that could travel with permuted tokens, so reordering the layer would silently change its
  output. Two guards keep it in natural order, each sufficient on its own: the layer is not causal
  (`AttentionType.FULL`), and it passes `rope_kwargs`. The positive control removes both.
  """
  # pylint: disable=import-outside-toplevel
  from maxtext.layers import attention_op as attention_op_lib
  from maxtext.layers import attentions

  off_cfg, on_cfg = _vision_attention_config(False), _vision_attention_config(True)
  cp = on_cfg.ici_context_parallelism
  x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 64), jnp.float32)
  _, _, off_out, off_hlo = _vision_attention_output(off_cfg, x)
  size, size_if_causal, on_out, on_hlo = _vision_attention_output(on_cfg, x)
  permutes = tuple(_count_hlo_ops(h, "collective-permute") for h in (off_hlo, on_hlo))
  error = _normalized_max_error(on_out, off_out)
  print(
      f"vision attention layer: load-balanced over {size} shards ({size_if_causal} if it were causal); "
      f"switch on vs off {error:.3e}; collective-permutes off/on {permutes}",
      flush=True,
  )
  assert size_if_causal == cp, f"the switch is not live in this config ({size_if_causal=}), so the check is vacuous"
  assert size == 1, f"a FULL vision attention layer is load-balanced over {size} shards"
  assert np.array_equal(on_out, off_out), f"the switch changed the vision attention output by {error:.3e}"
  assert permutes[0] == permutes[1], f"the switch added collective-permutes to the vision attention layer {permutes}"

  # The rope_kwargs guard alone: with the layer-type check bypassed the layer still keeps natural order.
  every_layer_balanced = unittest.mock.patch.object(
      attention_op_lib.AttentionOp, "attention_load_balance_context_size", lambda self, model_mode: cp
  )
  _, _, guarded_out, _ = _vision_attention_output(on_cfg, x, patch=every_layer_balanced)
  assert np.array_equal(guarded_out, off_out), "rope_kwargs alone did not keep the vision layer in natural order"

  # Positive control: remove both guards; the layer is reordered and its output must change.
  no_guards = unittest.mock.patch.object(attentions.Attention, "_attention_load_balance_size", lambda self, *args: cp)
  _, _, bad_out, _ = _vision_attention_output(on_cfg, x, patch=no_guards)
  bad_error = _normalized_max_error(bad_out, off_out)
  print(f"vision attention positive control (reordered anyway): {bad_error:.3e}", flush=True)
  assert bad_error > 1e-2, f"positive control: reordering the vision layer changed its output by only {bad_error:.3e}"


def _run_child_checks():
  """Every check that needs 8 CPU devices; run in the child process."""
  devices = np.array(jax.devices()[:8])
  _run_reorder_checks(Mesh(devices.reshape(2, 4), ("fsdp", "context")), "fsdp")
  _run_reorder_checks(Mesh(devices.reshape(4, 2), ("fsdp", "context")), "fsdp")
  _run_reorder_checks(Mesh(devices.reshape(1, 8), ("fsdp", "context")), None)
  _run_vision_layer_checks()
  _run_dense_model_checks(seq=256)
  ring_mesh = Mesh(devices.reshape(2, 4), ("fsdp", "context"))
  # Blocks aligned to the DUAL_CHUNK_SWAP chunks, with the staged dq reduction, and
  # blocks that straddle two chunks, so the in-kernel mask reads a non-monotone q_sequence.
  _run_ring_kernel_checks(ring_mesh, seq=2048, block=128, expect_staged_dq=True)
  _run_ring_kernel_checks(ring_mesh, seq=1024, block=256, expect_staged_dq=False)
  _run_model_checks(seq=256)
  print(_CHILD_MARKER, flush=True)


@pytest.mark.cpu_only
def test_attention_load_balance_matches_natural_order_on_cpu_mesh():
  if len(jax.devices()) >= 8 and jax.devices()[0].platform == "cpu":
    _run_child_checks()
    return
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _CHILD_MARKER in result.stdout


class LoadBalancePermutationTest(absltest.TestCase):
  """The permutation arithmetic, with no devices."""

  def test_permutations_at_context_4(self):
    first, second = tokamax_ring_attention.load_balance_permutations(4)
    self.assertEqual(first, ((0, 0), (1, 2), (2, 3), (3, 1)))
    self.assertEqual(second, ((0, 1), (1, 3), (2, 2), (3, 0)))

  def test_permutations_reproduce_reorder_sequence(self):
    for cp in (2, 4, 6, 8):
      first, second = tokamax_ring_attention.load_balance_permutations(cp)
      self.assertEqual(sorted(dst for _, dst in first), list(range(cp)))
      self.assertEqual(sorted(dst for _, dst in second), list(range(cp)))
      # Simulate: rank d holds chunks (2d, 2d+1); even ranks keep (from first, from second).
      held = {}
      for src, dst in first:
        held.setdefault(dst, {})["first"] = 2 * src
      for src, dst in second:
        held.setdefault(dst, {})["second"] = 2 * src + 1
      order = []
      for rank in range(cp):
        lead, tail = ("first", "second") if rank % 2 == 0 else ("second", "first")
        order += [held[rank][lead], held[rank][tail]]
      expected = max_utils.reorder_mask_load_balancing(np.arange(2 * cp), cp, 0).tolist()
      self.assertEqual(order, expected, f"cp={cp}")

  def test_rejects_odd_context(self):
    with self.assertRaisesRegex(ValueError, "even"):
      tokamax_ring_attention.load_balance_permutations(3)


class AttentionLoadBalanceConfigTest(absltest.TestCase):
  """The validator: legal with GatedDeltaNet, exclusive with the global reorder."""

  def _initialize(self, *overrides):
    """Builds a Qwen3.5 (GatedDeltaNet) ring config on 8 mocked TPU devices with the switch on."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "model_name=qwen3.5-397b-a17b",
        "override_model_config=true",
        "base_num_decoder_layers=4",
        "use_multimodal=false",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "context_parallel_attention_load_balance=True",
        "ici_context_parallelism=4",
        "max_target_length=4096",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
        *overrides,
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      return pyconfig.initialize(argv)

  def test_accepts_gated_delta_net_with_ring(self):
    config = self._initialize()
    self.assertTrue(config.context_parallel_attention_load_balance)
    self.assertGreater(config.ici_context_parallelism, 1)

  def test_global_load_balance_still_rejected_for_gated_delta_net(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "GatedDeltaNet"):
      self._initialize("context_parallel_attention_load_balance=False", "context_parallel_load_balance=True")

  def test_rejects_both_reorders(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "context_parallel_load_balance=False"):
      self._initialize("context_parallel_load_balance=True")

  def test_rejects_ulysses(self):
    with self.assertRaisesRegex(
        (ValueError, pydantic.ValidationError), "supports context_parallel_strategy='ring' or 'all_gather'"
    ):
      self._initialize("context_parallel_strategy=ulysses", "base_num_query_heads=4", "base_num_kv_heads=4")

  def test_rejects_odd_context(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "even context parallelism"):
      self._initialize("ici_context_parallelism=3", "max_target_length=4608")

  def test_rejects_multimodal(self):
    # The ring strategy rejects multimodal attention on its own, so this uses all_gather, where only the switch does.
    multimodal = ("use_multimodal=true", "context_parallel_strategy=all_gather")
    self.assertTrue(self._initialize(*multimodal, "context_parallel_attention_load_balance=False").use_multimodal)
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "does not support use_multimodal=True"):
      self._initialize(*multimodal)

  def test_rejects_gpu_kernels(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "supports attention='flash'"):
      self._initialize("hardware=cpu", "context_parallel_strategy=all_gather", "attention=autoselected")


if __name__ == "__main__":
  if len(sys.argv) == 1:
    _run_child_checks()
  else:
    absltest.main()
