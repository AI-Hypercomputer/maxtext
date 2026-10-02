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

"""Correctness tests and microbenchmarks for the SparseCore ragged kernels.

Covers ``ragged_gather`` and ``ragged_gather_reduce`` (v2) against their JAX
reference implementations, using routing that mimics ``ring_ragged_sort`` /
``ring_ragged_unsort`` (ring-of-experts EP with a bounded local buffer).

The production-shape benchmark is skipped unless RAGGED_KERNELS_BENCH=1:
  RAGGED_KERNELS_BENCH=1 python3 -m pytest tests/unit/ragged_kernels_test.py -k Benchmark -s
Set RAGGED_KERNELS_PROFILE_DIR=/path to also capture a jax.profiler trace.
"""

import contextlib
import dataclasses
import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.ragged import ragged_gather as rg
from maxtext.kernels.ragged import ragged_gather_reduce_v2 as rgr
import numpy as np
import pytest

_RUN_BENCH = os.environ.get("RAGGED_KERNELS_BENCH", "0") == "1"
_PROFILE_DIR = os.environ.get("RAGGED_KERNELS_PROFILE_DIR", "")
_BENCH_ITERS = int(os.environ.get("RAGGED_KERNELS_BENCH_ITERS", "20"))


@contextlib.contextmanager
def _maybe_profile():
  if _PROFILE_DIR:
    with jax.profiler.trace(_PROFILE_DIR):
      yield _PROFILE_DIR
  else:
    yield None


@dataclasses.dataclass(frozen=True)
class _Routing:
  """Host-side routing arrays mimicking ``ring_ragged_sort`` (buffered path)."""

  num_tokens: int
  topk: int
  buffer_size: int
  start: int  # shard_output_start
  end: int  # shard_output_end
  token_indices_sorted: np.ndarray  # [n]
  topk_argsort_indices: np.ndarray  # [n]
  topk_argsort_revert_indices: np.ndarray  # [n]

  @property
  def n(self) -> int:
    return self.num_tokens * self.topk

  @property
  def gather_end(self) -> int:
    return min(self.end - self.start, self.buffer_size)

  def fwd_gather_indices(self) -> np.ndarray:
    """``sliced_indices`` passed to ragged_gather in ring_ragged_sort fwd."""
    padded = np.pad(self.token_indices_sorted, (0, self.buffer_size))
    return padded[self.start : self.start + self.buffer_size].astype(np.int32)

  def bwd_gather_indices(self) -> np.ndarray:
    """``sliced_idx_inv // topk`` passed to ragged_gather in ring_ragged_unsort bwd."""
    padded = np.pad(self.topk_argsort_indices, (0, self.buffer_size))
    return (padded[self.start : self.start + self.buffer_size] // self.topk).astype(np.int32)

  def gather_reduce_args(self) -> tuple[np.ndarray, np.ndarray]:
    """``(safe_indices, valid_rows_mask)`` used by ring_ragged_unsort fwd."""
    shifted = self.topk_argsort_revert_indices - self.start
    valid = (shifted >= 0) & (shifted < self.gather_end)
    return np.where(valid, shifted, 0).astype(np.int32), valid


def _make_routing(num_tokens, topk, num_experts, ep_size, shard_idx, buffer_factor, seed=0) -> _Routing:
  rng = np.random.RandomState(seed)
  topk_indices = np.argsort(rng.random_sample((num_tokens, num_experts)), axis=-1)[:, :topk]
  flat = topk_indices.reshape(-1)
  argsort_idx = np.argsort(flat, kind="stable")
  token_indices_sorted = np.repeat(np.arange(num_tokens, dtype=np.int32), topk)[argsort_idx]
  group_sizes = np.bincount(flat, minlength=num_experts)
  offsets = np.concatenate([[0], np.cumsum(group_sizes)])
  local = num_experts // ep_size
  start = int(offsets[shard_idx * local])
  end = int(offsets[(shard_idx + 1) * local])
  buffer_size = int(num_tokens * topk // ep_size * buffer_factor)
  return _Routing(
      num_tokens=num_tokens,
      topk=topk,
      buffer_size=buffer_size,
      start=start,
      end=end,
      token_indices_sorted=token_indices_sorted.astype(np.int32),
      topk_argsort_indices=argsort_idx.astype(np.int32),
      topk_argsort_revert_indices=np.argsort(argsort_idx).astype(np.int32),
  )


def _random_rows(shape, dtype, seed):
  x = jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float32)
  return x.astype(dtype)


def _as_bits(x):
  """Bit-exact view used to compare gathered rows (works for fp8 too)."""
  bits = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}[jnp.dtype(x.dtype).itemsize]
  return np.asarray(jax.lax.bitcast_convert_type(x, bits))


def _has_sparsecore() -> bool:
  try:
    if not any(d.platform == "tpu" for d in jax.devices()):
      return False
    return pltpu.get_tpu_info().sparse_core is not None
  except Exception:  # pylint: disable=broad-exception-caught
    return False


class _SparseCoreTestCase(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not _has_sparsecore():
      self.skipTest("SparseCore ragged kernels need a TPU with SparseCore.")


@pytest.mark.tpu_only
class RaggedGatherTest(_SparseCoreTestCase):

  @parameterized.product(
      dtype=[jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn],
      hidden=[1024, 7168],
      bounds=["full", "unaligned", "empty"],
      pipelined=[True, False],
  )
  def test_matches_reference(self, dtype, hidden, bounds, pipelined):
    num_in, num_out = 2048, 3000
    x = _random_rows((num_in, hidden), dtype, seed=1)
    indices = jax.random.randint(jax.random.PRNGKey(2), (num_out,), 0, num_in, dtype=jnp.int32)
    start, end = {"full": (0, num_out), "unaligned": (37, 2021), "empty": (100, 100)}[bounds]

    out = rg.ragged_gather(x, indices, jnp.int32(start)[None], jnp.int32(end)[None], use_pipelined_kernel=pipelined)
    self.assertGreaterEqual(out.shape[0], num_out)
    self.assertGreaterEqual(out.shape[1], hidden)
    expected = x[indices]
    # Only rows in [start, end) are defined by the kernel contract.
    np.testing.assert_array_equal(_as_bits(out[start:end, :hidden]), _as_bits(expected[start:end]))

  @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], pipelined=[True, False])
  def test_weights(self, dtype, pipelined):
    num_in, num_out, hidden = 1024, 1536, 1024
    x = _random_rows((num_in, hidden), dtype, seed=3)
    indices = jax.random.randint(jax.random.PRNGKey(4), (num_out,), 0, num_in, dtype=jnp.int32)
    weights = jax.random.uniform(jax.random.PRNGKey(5), (num_out,), dtype=jnp.float32)
    out = rg.ragged_gather(
        x,
        indices,
        jnp.int32(0)[None],
        jnp.int32(num_out)[None],
        weights=weights,
        has_weights=True,
        use_pipelined_kernel=pipelined,
    )
    expected = (x[indices].astype(jnp.float32) * weights[:, None]).astype(dtype)
    np.testing.assert_allclose(
        np.asarray(out[:num_out, :hidden], np.float32), np.asarray(expected, np.float32), rtol=1e-2, atol=1e-2
    )

  def test_fp8_weights_rejected(self):
    x = _random_rows((256, 1024), jnp.float8_e4m3fn, seed=3)
    indices = jnp.zeros((512,), jnp.int32)
    with self.assertRaises(NotImplementedError):
      rg.ragged_gather(x, indices, jnp.int32(0)[None], jnp.int32(512)[None], weights=jnp.ones((512,)), has_weights=True)

  @parameterized.parameters(jnp.float8_e4m3fn, jnp.bfloat16)
  def test_ring_routing(self, dtype):
    r = _make_routing(num_tokens=4096, topk=8, num_experts=64, ep_size=8, shard_idx=3, buffer_factor=1.375)
    x = _random_rows((r.num_tokens, 7168), dtype, seed=6)
    idx = jnp.asarray(r.fwd_gather_indices())
    out = rg.ragged_gather(x, idx, jnp.int32(0)[None], jnp.int32(r.gather_end)[None])
    np.testing.assert_array_equal(_as_bits(out[: r.gather_end]), _as_bits(x[idx[: r.gather_end]]))


@pytest.mark.tpu_only
class RaggedGatherReduceTest(_SparseCoreTestCase):

  @parameterized.product(
      reduce_group_size=[1, 8],
      hidden=[1024, 7168],
      valid_fraction=[0.0, 0.17, 1.0],
  )
  def test_matches_reference(self, reduce_group_size, hidden, valid_fraction):
    num_out = 1024
    n = num_out * reduce_group_size
    num_src = 3000
    x = _random_rows((num_src, hidden), jnp.bfloat16, seed=7)
    indices = jax.random.randint(jax.random.PRNGKey(8), (n,), 0, num_src, dtype=jnp.int32)
    weights = jax.random.uniform(jax.random.PRNGKey(9), (n,), dtype=jnp.float32)
    valid = jax.random.uniform(jax.random.PRNGKey(10), (n,)) < valid_fraction

    out = rgr.ragged_gather_reduce(x, indices, weights, valid, reduce_group_size=reduce_group_size)
    expected = rgr.ragged_gather_reduce(
        x, indices, weights, valid, reduce_group_size=reduce_group_size, enforce_fallback=True
    )
    self.assertEqual(out.shape, expected.shape)
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(expected, np.float32), rtol=2e-2, atol=2e-2)

  def test_ring_routing(self):
    r = _make_routing(num_tokens=4096, topk=8, num_experts=64, ep_size=8, shard_idx=3, buffer_factor=1.375)
    x = _random_rows((r.buffer_size, 7168), jnp.bfloat16, seed=11)
    safe, valid = r.gather_reduce_args()
    weights = jax.random.uniform(jax.random.PRNGKey(12), (r.n,), dtype=jnp.float32)
    out = rgr.ragged_gather_reduce(x, jnp.asarray(safe), weights, jnp.asarray(valid), reduce_group_size=r.topk)
    expected = rgr.ragged_gather_reduce(
        x, jnp.asarray(safe), weights, jnp.asarray(valid), reduce_group_size=r.topk, enforce_fallback=True
    )
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(expected, np.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.tpu_only
class RaggedKernelsBenchmark(_SparseCoreTestCase):
  """Production-shape microbenchmarks (DeepSeek-style MoE, ring EP=8)."""

  def setUp(self):
    super().setUp()
    if not _RUN_BENCH:
      self.skipTest("Set RAGGED_KERNELS_BENCH=1 to run the benchmark.")

  NUM_TOKENS = 32768
  HIDDEN = 7168
  TOPK = 8
  NUM_EXPERTS = 256
  EP_SIZE = 8
  BUFFER_FACTOR = 1.375  # -> buffer_size = 45056, as in the profiled run.

  def _time(self, name, fn, *args, useful_bytes):
    """Prints wall-clock per call; a profile gives exact device time per op."""
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))  # compile + warmup
    out = jax.block_until_ready(jitted(*args))
    iters = _BENCH_ITERS
    t0 = time.perf_counter()
    for _ in range(iters):
      out = jitted(*args)
    jax.block_until_ready(out)
    us = (time.perf_counter() - t0) / iters * 1e6
    print(
        f"[ragged-bench] {name:40s} {us:9.1f} us/call  useful={useful_bytes / 1e6:8.1f} MB"
        f"  eff_bw={useful_bytes / us / 1e3:7.1f} GB/s",
        flush=True,
    )
    return us

  def test_benchmark(self):
    r = _make_routing(
        self.NUM_TOKENS, self.TOPK, self.NUM_EXPERTS, self.EP_SIZE, shard_idx=3, buffer_factor=self.BUFFER_FACTOR
    )
    print(f"[ragged-bench] buffer_size={r.buffer_size} gather_end={r.gather_end} n={r.n}", flush=True)
    h = self.HIDDEN
    start0 = jnp.int32(0)[None]
    end = jnp.int32(r.gather_end)[None]

    x_fp8 = _random_rows((self.NUM_TOKENS, h), jnp.float8_e4m3fn, seed=20)
    g_bf16 = _random_rows((self.NUM_TOKENS, h), jnp.bfloat16, seed=21)
    y_bf16 = _random_rows((r.buffer_size, h), jnp.bfloat16, seed=22)
    fwd_idx = jnp.asarray(r.fwd_gather_indices())
    bwd_idx = jnp.asarray(r.bwd_gather_indices())
    safe, valid = (jnp.asarray(a) for a in r.gather_reduce_args())
    weights = jax.random.uniform(jax.random.PRNGKey(23), (r.n,), dtype=jnp.float32)

    gather_fp8_bytes = 2 * r.gather_end * h * 1
    gather_bf16_bytes = 2 * r.gather_end * h * 2
    num_valid = int(np.sum(np.asarray(valid)))
    gr_bytes = num_valid * h * 2 + self.NUM_TOKENS * h * 2

    def sc_gather(**kw):
      return lambda x, i: rg.ragged_gather(x, i, start0, end, **kw)

    cases = []
    for tag, x, idx, nbytes in (
        ("fwd_gather_fp8", x_fp8, fwd_idx, gather_fp8_bytes),
        ("bwd_gather_bf16", g_bf16, bwd_idx, gather_bf16_bytes),
    ):
      cases += [
          (f"{tag}/sc_v1", sc_gather(use_pipelined_kernel=False), (x, idx), nbytes),
          (f"{tag}/sc_pipelined", sc_gather(), (x, idx), nbytes),
          (f"{tag}/sc_pipelined_dma_only", sc_gather(debug_dma_only=True), (x, idx), nbytes),
          (f"{tag}/xla", lambda x, i: x[i], (x, idx), nbytes),
      ]
    cases += [
        (
            "gather_reduce_bf16/sc",
            lambda x, i, w, v: rgr.ragged_gather_reduce(x, i, w, v, reduce_group_size=self.TOPK),
            (y_bf16, safe, weights, valid),
            gr_bytes,
        ),
    ]
    with _maybe_profile() as profile_dir:
      for name, fn, args, nbytes in cases:
        self._time(name, fn, *args, useful_bytes=nbytes)
    if profile_dir:
      print(f"[ragged-bench] Profile written to: {profile_dir}", flush=True)


if __name__ == "__main__":
  absltest.main()
