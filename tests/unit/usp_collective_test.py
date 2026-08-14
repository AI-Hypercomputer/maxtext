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
"""Executes the USP collectives on a forced multi-device CPU mesh.

Runs as a subprocess so the forced device count takes effect before JAX
initializes; the parent pytest process has already initialized JAX with the
default device count. The child checks, against a dense single-device
reference, the Ulysses exchange and round trip over the pair sharding, the
segment-ID gather into the ring layout, per-ring-chunk attention equivalence,
and independent Q, K, and V gradients, on 2x2, 2x4, and 4x2 ring x Ulysses
meshes and on a 3-D fsdp x ring x Ulysses mesh with the batch sharded over
fsdp. The ring dimension is modeled by gathering K/V over the ring axis; the
rotation itself belongs to the TPU ring kernel tests. The asymmetric
factorizations matter: a symmetric mesh cannot expose an accidental swap of
the ring and Ulysses dimensions.
"""

import os
import subprocess
import sys
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

from maxtext.kernels.attention import ulysses_attention


@pytest.mark.cpu_only
def test_usp_collectives_match_dense_reference_on_cpu_mesh():
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "USP_COLLECTIVE_CHECKS_PASSED" in result.stdout


def _dense_reference_attention(query, key, value, segment_ids):
  """Causal segment-masked GQA attention computed on one device."""
  _, num_query_heads, seq_len, _ = query.shape
  num_kv_heads = key.shape[1]
  group_size = num_query_heads // num_kv_heads
  key = jnp.repeat(key, group_size, axis=1)
  value = jnp.repeat(value, group_size, axis=1)

  logits = jnp.einsum("bhqd,bhkd->bhqk", query, key)
  causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
  same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
  not_padding = segment_ids != 0
  mask = causal[None, None, :, :] & same_segment[:, None, :, :] & not_padding[:, None, None, :]
  logits = jnp.where(mask, logits, -1e30)
  weights = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
  weights = weights * mask
  weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-30)
  return jnp.einsum("bhqk,bhkd->bhqd", weights, value)


def _run_collective_checks(mesh, batch_axis):
  """Runs the exchange, segment-ID layout, attention, and gradient checks on one mesh."""
  batch, num_query_heads, num_kv_heads, seq_len, head_dim = 2, 8, 4, 32, 4
  ulysses_axis = "context_usp_ulysses"
  ring_axis = "context"
  ring_size = mesh.shape[ring_axis]
  data_spec = P(batch_axis, None, ("context", "context_usp_ulysses"), None)
  segment_spec = P(batch_axis, ("context", "context_usp_ulysses"))

  # Rank-coded values make any head or sequence misordering visible exactly.
  def coded(num_heads, offset):
    values = np.arange(batch * num_heads * seq_len * head_dim, dtype=np.float32)
    return jnp.asarray(values.reshape(batch, num_heads, seq_len, head_dim) / 100.0 + offset)

  query = coded(num_query_heads, 1.0)
  key = coded(num_kv_heads, 2.0)
  value = coded(num_kv_heads, 3.0)
  # Segments begin and end inside different shards of the pair sharding, with
  # trailing padding zeros.
  segment_ids = jnp.broadcast_to(jnp.asarray([1] * 10 + [2] * 12 + [0] * 10, dtype=jnp.int32)[None, :], (batch, seq_len))

  # Round trip through the real helpers is exact on the 2-D mesh.
  @partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=data_spec,
      out_specs=data_spec,
      check_vma=False,
  )
  def round_trip(tensor):
    return ulysses_attention.inverse_ulysses_all_to_all(
        ulysses_attention.ulysses_all_to_all(tensor, ulysses_axis), ulysses_axis
    )

  np.testing.assert_array_equal(jax.device_get(round_trip(query)), jax.device_get(query))

  # The forward exchange produces each rank's head subset over its ring chunk.
  @partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=data_spec,
      out_specs=P(batch_axis, "context_usp_ulysses", "context", None),
      check_vma=False,
  )
  def exchange(tensor):
    return ulysses_attention.ulysses_all_to_all(tensor, ulysses_axis)

  np.testing.assert_array_equal(jax.device_get(exchange(query)), jax.device_get(query))

  @partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(data_spec, data_spec, data_spec, segment_spec),
      out_specs=data_spec,
      check_vma=False,
  )
  def usp_attention_fn(query, key, value, segment_ids):
    query = ulysses_attention.ulysses_all_to_all(query, ulysses_axis)
    key = ulysses_attention.ulysses_all_to_all(key, ulysses_axis)
    value = ulysses_attention.ulysses_all_to_all(value, ulysses_axis)
    ring_segment_ids = jax.lax.all_gather(segment_ids, ulysses_axis, axis=1, tiled=True)
    # Model the ring dimension by gathering K/V over the ring axis: each device
    # computes its own ring chunk of query rows against the full sequence.
    full_key = jax.lax.all_gather(key, ring_axis, axis=2, tiled=True)
    full_value = jax.lax.all_gather(value, ring_axis, axis=2, tiled=True)
    full_segment_ids = jax.lax.all_gather(ring_segment_ids, ring_axis, axis=1, tiled=True)
    full_query = jax.lax.all_gather(query, ring_axis, axis=2, tiled=True)
    output = _dense_reference_attention(full_query, full_key, full_value, full_segment_ids)
    chunk = output.shape[2] // ring_size
    output = jax.lax.dynamic_slice_in_dim(output, jax.lax.axis_index(ring_axis) * chunk, chunk, axis=2)
    return ulysses_attention.inverse_ulysses_all_to_all(output, ulysses_axis)

  def dense_loss(query, key, value):
    output = _dense_reference_attention(query, key, value, segment_ids)
    return jnp.sum(output * jnp.cos(output))

  def usp_loss(query, key, value):
    output = usp_attention_fn(query, key, value, segment_ids)
    return jnp.sum(output * jnp.cos(output))

  dense_output = _dense_reference_attention(query, key, value, segment_ids)
  usp_output = usp_attention_fn(query, key, value, segment_ids)
  np.testing.assert_allclose(jax.device_get(usp_output), jax.device_get(dense_output), atol=1e-5)

  dense_grads = jax.grad(dense_loss, argnums=(0, 1, 2))(query, key, value)
  usp_grads = jax.grad(usp_loss, argnums=(0, 1, 2))(query, key, value)
  for name, dense_grad, usp_grad in zip(("dQ", "dK", "dV"), dense_grads, usp_grads):
    np.testing.assert_allclose(jax.device_get(usp_grad), jax.device_get(dense_grad), atol=1e-5, err_msg=name)


if __name__ == "__main__":
  _devices = np.array(jax.devices())
  assert len(_devices) == 8, jax.devices()
  _run_collective_checks(Mesh(_devices[:4].reshape(2, 2), ("context", "context_usp_ulysses")), batch_axis=None)
  _run_collective_checks(Mesh(_devices.reshape(2, 4), ("context", "context_usp_ulysses")), batch_axis=None)
  _run_collective_checks(Mesh(_devices.reshape(4, 2), ("context", "context_usp_ulysses")), batch_axis=None)
  _run_collective_checks(Mesh(_devices.reshape((2, 2, 2)), ("fsdp", "context", "context_usp_ulysses")), batch_axis="fsdp")
  print("USP_COLLECTIVE_CHECKS_PASSED")
