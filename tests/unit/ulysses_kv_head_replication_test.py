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
"""Checks Ulysses attention with fewer KV heads than context ranks on a forced 4-device CPU mesh.

With 2 KV heads at context 4, each KV head is repeated twice before the all-to-all so that every rank receives
the KV head its contiguous block of query heads uses. The checks run in a subprocess so the forced device count
takes effect before JAX initializes:

  - Collectives. The real repeat and all-to-all helpers around a dense GQA kernel match a dense single-device
    reference in the output and in dQ/dK/dV, on 8 query heads and on 32 (the Qwen3.5 head count).
  - Attention op. `AttentionOp.tpu_flash_attention`, on a packed row with a long segment, short segments and a
    padded tail: the Ulysses path matches the all-gather path and a dense reference, output and dQ/dK/dV.

Each check has positive controls that must fail the same gate: tiling the KV heads (`[kv0, kv1, kv0, kv1]`), which
pairs half of the query heads with the wrong KV head, and a repeat whose backward keeps only the first copy's
gradient instead of summing the copies.

The attention op runs with a dense stand-in for the Tokamax Splash kernel (same call, mask and grouped query/KV
head pairing), not with the kernel in Pallas TPU interpret mode: every interpreted kernel call waits on a barrier
across all devices, and XLA:CPU may run a collective that does not depend on the kernel before it on one device and
after it on another, so the devices can wait on each other forever. The kernel itself is covered on TPU by the
Ulysses cases in attention_test.py.
"""

import functools
import os
import subprocess
import sys
from unittest import mock

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from maxtext.kernels.attention import ulysses_attention

_CHILD_MARKER = "ULYSSES_KV_HEAD_REPLICATION_CHECKS_PASSED"
# Normalized max error max|a - b| / max|b| in float32; the paths differ only in reduction order.
_TOLERANCE = 1e-5
_NUM_DEVICES = 4


@pytest.mark.cpu_only
def test_ulysses_kv_head_replication_on_cpu_mesh():
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _CHILD_MARKER in result.stdout


def _normalized_max_error(actual, expected) -> float:
  actual = np.asarray(actual, np.float64)
  expected = np.asarray(expected, np.float64)
  return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-30))


def _tile_kv_heads(tensor, replication):
  """Positive control: tiled copies, which give rank 1 of 4 the second KV head for first-group query heads."""
  return tensor if replication == 1 else jnp.tile(tensor, (1, replication, 1, 1))


def _repeat_without_group_sum(tensor, replication):
  """Positive control: correct copies, but a backward that drops every copy's gradient except the first."""
  if replication == 1:
    return tensor

  @jax.custom_vjp
  def repeat(x):
    return jnp.repeat(x, replication, axis=1)

  def repeat_fwd(x):
    return repeat(x), None

  def repeat_bwd(_, cotangent):
    return (cotangent[:, ::replication],)

  repeat.defvjp(repeat_fwd, repeat_bwd)
  return repeat(tensor)


@jax.tree_util.register_pytree_node_class
class _DenseSplashKernel:
  """Dense stand-in for a Tokamax Splash kernel: the same call, mask and grouped query/KV head pairing.

  Splash pairs query head h with KV head h // (H_q / H_kv). The mask is materialized, so it is sharded over the
  query sequence exactly as the real kernel's mask metadata is, and each shard holds the rows of its queries.
  """

  def __init__(self, mask):
    self.mask = mask

  def tree_flatten(self):
    return (self.mask,), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  def __call__(self, query, key, value, segment_ids, sinks=None, max_logit_value=None):
    del sinks, max_logit_value
    group = query.shape[0] // key.shape[0]
    key, value = jnp.repeat(key, group, axis=0), jnp.repeat(value, group, axis=0)
    logits = jnp.einsum("hqd,hkd->hqk", query, key, precision="highest")
    allowed = self.mask & (segment_ids.q[:, None] == segment_ids.kv[None, :])
    probs = jax.nn.softmax(jnp.where(allowed[None], logits, -1e30), axis=-1)
    return jnp.einsum("hqk,hkd->hqd", probs, value, precision="highest").astype(query.dtype)


def _make_dense_splash_kernel(mask, config, q_seq_shards):
  del config, q_seq_shards
  return _DenseSplashKernel(jnp.asarray(mask[:, :]))


def _segment_ids(seq_len: int, row: int = 0) -> np.ndarray:
  """A packed row. Row 0: a long segment over several context shards, five short segments, then padding (id 0).

  Row 1: sixteen short segments whose boundaries are not aligned to the context shards.
  """
  positions = np.arange(seq_len)
  if row == 1:
    boundaries = np.linspace(0, seq_len, 17)[1:-1].astype(int) + 3
    return (np.searchsorted(boundaries, positions, side="right") + 1).astype(np.int32)
  long_end = (seq_len * 9) // 16
  pad_start = (seq_len * 15) // 16
  short = np.searchsorted(np.linspace(long_end, pad_start, 6)[1:-1], positions, side="right") + 2
  return np.where(positions < long_end, 1, np.where(positions < pad_start, short, 0)).astype(np.int32)


def _dense_attention(query, key, value, segment_ids):
  """Causal, same-segment GQA attention over `[B, H, S, D]` (padding attends to padding, as in Splash)."""
  seq_len = query.shape[2]
  group = query.shape[1] // key.shape[1]
  key, value = jnp.repeat(key, group, axis=1), jnp.repeat(value, group, axis=1)
  logits = jnp.einsum("bhqd,bhkd->bhqk", query, key, precision="highest")
  allowed = jnp.tril(jnp.ones((seq_len, seq_len), bool))[None, None] & (
      segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
  )
  probs = jax.nn.softmax(jnp.where(allowed, logits, -1e30), axis=-1)
  return jnp.einsum("bhqk,bhkd->bhqd", probs, value, precision="highest")


def _value_and_grads(fn, args, cotangent, mesh, with_grads=True):
  def loss(*xs):
    return jnp.sum(fn(*xs).astype(jnp.float32) * cotangent)

  with jax.set_mesh(mesh):
    out = jax.jit(fn)(*args)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*args) if with_grads else ()
  return [np.asarray(out, np.float32)] + [np.asarray(g, np.float32) for g in grads]


def _check(name, result, reference, controls):
  """`result` must match `reference`; every `controls` entry (label, values, field index) must miss it."""
  errors = [_normalized_max_error(r, e) for r, e in zip(result, reference)]
  print(f"{name}: normalized max error out/dq/dk/dv = {errors}", flush=True)
  assert max(errors) <= _TOLERANCE, f"{name}: {errors}"
  for label, values, index in controls:
    error = _normalized_max_error(values[index], reference[index])
    print(f"{name} positive control {label}: {error}", flush=True)
    assert error > 1000 * _TOLERANCE, f"{name}: positive control {label} passed the gate ({error})"


def _run_collective_checks(num_query_heads: int):
  """The repeat and all-to-all helpers around a dense kernel vs a dense single-device reference."""
  mesh = Mesh(np.array(jax.devices()), ("context",))
  ulysses_axis = "context"
  batch, num_kv_heads, seq_len, head_dim = 1, 2, 64, 8
  rng = np.random.default_rng(0)
  query = jnp.asarray(rng.standard_normal((batch, num_query_heads, seq_len, head_dim)), jnp.float32)
  key = jnp.asarray(rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)), jnp.float32)
  value = jnp.asarray(rng.standard_normal((batch, num_kv_heads, seq_len, head_dim)), jnp.float32)
  cotangent = jnp.asarray(rng.standard_normal((batch, num_query_heads, seq_len, head_dim)), jnp.float32)
  segment_ids = jnp.asarray(_segment_ids(seq_len)[None])
  data_spec, segment_spec = P(None, None, "context", None), P(None, "context")

  def ulysses(repeat_fn):
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(data_spec, data_spec, data_spec, segment_spec),
        out_specs=data_spec,
        check_vma=False,
    )
    def attention(query, key, value, segment_ids):
      replication = ulysses_attention.kv_head_replication(key.shape[1], mesh.shape[ulysses_axis])
      key, value = repeat_fn(key, replication), repeat_fn(value, replication)
      query, key, value = (ulysses_attention.ulysses_all_to_all(x, ulysses_axis) for x in (query, key, value))
      segment_ids = jax.lax.all_gather(segment_ids, ulysses_axis, axis=1, tiled=True)
      output = _dense_attention(query, key, value, segment_ids)
      return ulysses_attention.inverse_ulysses_all_to_all(output, ulysses_axis)

    return lambda q, k, v: attention(q, k, v, segment_ids)

  args = (query, key, value)
  reference = _value_and_grads(lambda q, k, v: _dense_attention(q, k, v, segment_ids), args, cotangent, mesh)
  result = _value_and_grads(ulysses(ulysses_attention.repeat_kv_heads_for_ulysses), args, cotangent, mesh)
  tiled = _value_and_grads(ulysses(_tile_kv_heads), args, cotangent, mesh, with_grads=False)
  no_sum = _value_and_grads(ulysses(_repeat_without_group_sum), args, cotangent, mesh)
  _check(
      f"collectives {num_query_heads}q/{num_kv_heads}kv",
      result,
      reference,
      [("tiled KV heads (out)", tiled, 0), ("no group sum (dk)", no_sum, 2), ("no group sum (dv)", no_sum, 3)],
  )


def _run_attention_op_checks():
  """`AttentionOp.tpu_flash_attention`, Ulysses vs all-gather vs dense, with the dense stand-in for the kernel."""
  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning

  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.configs import pyconfig
  from maxtext.layers import attention_op
  from maxtext.utils import globals as maxtext_globals
  from maxtext.utils import maxtext_utils

  seq_len, num_query_heads, num_kv_heads, head_dim, block = 512, 8, 2, 128, 128
  cfg = pyconfig.initialize(
      [
          None,
          os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml"),
          "run_name=ulysses_kv_head_replication_test",
          "hardware=cpu",
          "custom_mesh_and_rule=cp-as-ep",
          "ici_fsdp_parallelism=1",
          f"ici_context_parallelism={_NUM_DEVICES}",
          "ici_expert_parallelism=1",
          "attention=flash",
          "use_tokamax_splash=True",
          "use_jax_splash=False",
          # The layer is built for all-gather; the Ulysses case switches the strategy at call time (below).
          "context_parallel_strategy=all_gather",
          "context_parallel_load_balance=False",
          f"max_target_length={seq_len}",
          f"base_num_query_heads={num_query_heads}",
          f"base_num_kv_heads={num_kv_heads}",
          f"head_dim={head_dim}",
          *(f"sa_{name}={block}" for name in ("block_q", "block_kv", "block_kv_compute", "block_q_dkv")),
          *(f"sa_{name}={block}" for name in ("block_kv_dkv", "block_kv_dkv_compute")),
          "dq_reduction_steps=3",
          f"per_device_batch_size={2 / _NUM_DEVICES}",
          "packing=True",
          "dataset_type=synthetic",
          "enable_checkpointing=false",
          "skip_jax_distributed_system=True",
          "enable_tensorboard=False",
      ]
  )
  mesh = maxtext_utils.get_mesh_from_config(cfg)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    op = attention_op.AttentionOp(
        config=cfg,
        mesh=mesh,
        attention_kernel="flash",
        max_target_length=seq_len,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        dtype=jnp.float32,
    )

  rng = np.random.default_rng(1)
  batch = cfg.micro_batch_size_to_train_on
  sharding = NamedSharding(mesh, P(None, "context"))
  query = rng.standard_normal((batch, seq_len, num_query_heads, head_dim)).astype(np.float32) / np.sqrt(head_dim)
  key = rng.standard_normal((batch, seq_len, num_kv_heads, head_dim)).astype(np.float32)
  value = rng.standard_normal((batch, seq_len, num_kv_heads, head_dim)).astype(np.float32)
  cotangent = rng.standard_normal((batch, seq_len, num_query_heads, head_dim)).astype(np.float32)
  segment_ids = np.stack([_segment_ids(seq_len, row % 2) for row in range(batch)])
  query, key, value, cotangent, segment_ids = (
      jax.device_put(x, sharding) for x in (query, key, value, cotangent, segment_ids)
  )

  def flash(strategy, repeat_fn=ulysses_attention.repeat_kv_heads_for_ulysses):
    def attention(q, k, v):
      with (
          mock.patch.dict(cfg.get_keys(), {"context_parallel_strategy": strategy}),
          mock.patch.object(attention_op.tokamax_splash_kernel, "make_splash_mha", _make_dense_splash_kernel),
          mock.patch.object(ulysses_attention, "repeat_kv_heads_for_ulysses", repeat_fn),
          nn_partitioning.axis_rules(cfg.logical_axis_rules),
      ):
        return op.tpu_flash_attention(q, k, v, segment_ids, model_mode=MODEL_MODE_TRAIN)[0]

    return attention

  def dense(q, k, v):
    q, k, v = (jnp.transpose(x, (0, 2, 1, 3)) for x in (q, k, v))
    return jnp.transpose(_dense_attention(q, k, v, segment_ids), (0, 2, 1, 3))

  args = (query, key, value)
  reference = _value_and_grads(dense, args, cotangent, mesh)
  all_gather = _value_and_grads(flash("all_gather"), args, cotangent, mesh)
  result = _value_and_grads(flash("ulysses"), args, cotangent, mesh)
  tiled = _value_and_grads(flash("ulysses", _tile_kv_heads), args, cotangent, mesh, with_grads=False)
  no_sum = _value_and_grads(flash("ulysses", _repeat_without_group_sum), args, cotangent, mesh)
  controls = [("tiled KV heads (out)", tiled, 0), ("no group sum (dk)", no_sum, 2), ("no group sum (dv)", no_sum, 3)]
  _check("attention op, ulysses vs all_gather", result, all_gather, controls)
  _check("attention op, ulysses vs dense", result, reference, controls)

  with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    hlo = jax.jit(flash("ulysses")).lower(*args).compile().as_text()
  # q, k, v in and the output back; no K/V all-gather (the one all-gather is the int32 segment IDs).
  assert hlo.count(" all-to-all(") == 4, hlo.count(" all-to-all(")
  assert hlo.count(" all-gather(") + hlo.count(" all-gather-start(") == 1


if __name__ == "__main__":
  assert len(jax.devices()) == _NUM_DEVICES, jax.devices()
  _run_collective_checks(num_query_heads=8)
  _run_collective_checks(num_query_heads=32)
  _run_attention_op_checks()
  print(_CHILD_MARKER)
