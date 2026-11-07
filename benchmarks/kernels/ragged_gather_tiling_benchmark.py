# Copyright 2023–2026 Google LLC
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

"""Compares the SparseCore ragged gather kernels with the old and new VMEM budget.

The kernels used to size their column tiles from a hard-coded per-generation VMEM table; they now read
`SparseCoreInfo.vmem_capacity_bytes`. This benchmark compiles each kernel twice on the same chip and the same
inputs, once with the old tile-size functions patched in and once with the current ones, checks that both
outputs are bit-identical and reports the median step time of each. It exits non-zero if the new tiling is
slower than the old one by more than `--tolerance`.

Needs a TPU with SparseCore (v5p, v6e, 7x, ...):

  python benchmarks/kernels/ragged_gather_tiling_benchmark.py --hidden-sizes 2048,4096,7168,8192
"""

import argparse
import statistics
import sys
import time
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxtext.kernels.ragged import ragged_gather
from maxtext.kernels.ragged import ragged_gather_reduce_v2


def _old_calculate_col_size(hidden_size: int) -> int:
  """`ragged_gather.calculate_col_size` before it used the SparseCore VMEM capacity."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_lanes = tpu_info.num_lanes
  num_simd_lanes = sc_info.num_lanes

  match tpu_info.chip_version:
    case 6:
      target_bytes = (256 * 1024) * 0.9
    case 7:
      target_bytes = (512 * 1024) * 0.9
    case _:
      target_bytes = (128 * 1024) * 0.9

  base_bytes = num_simd_lanes * hidden_size * (32 // 8)
  num_cols = 1

  while pl.cdiv(base_bytes, num_cols * num_lanes) * num_lanes > target_bytes:
    num_cols += 1
  return pl.cdiv(hidden_size, (num_cols * num_lanes)) * num_lanes


def _old_calculate_col_chunk_size(col_size: int, num_simd_lanes: int) -> int:
  """`ragged_gather_reduce_v2._calculate_col_chunk_size` before it used the SparseCore VMEM capacity."""
  generation = pltpu.get_tpu_info().generation
  match generation:
    case 6:
      target_bytes = int(256 * 1024 * 0.95)
    case 7:
      target_bytes = int(512 * 1024 * 0.95)
    case _:
      target_bytes = int(128 * 1024 * 0.95)

  bytes_per_col = num_simd_lanes * 4 * 2
  max_safe_col = (target_bytes // bytes_per_col // 128) * 128
  max_safe_col = min(max_safe_col, ragged_gather_reduce_v2._CostModelConstants.MAX_COL_CHUNK_SIZE)  # pylint: disable=protected-access

  start_col = (min(col_size, max_safe_col) // 128) * 128
  for chunk in range(start_col, 127, -128):
    if col_size % chunk == 0:
      return chunk
  return 128


_GATHER_PATCH = (ragged_gather, "calculate_col_size", _old_calculate_col_size)
_REDUCE_PATCH = (ragged_gather_reduce_v2, "_calculate_col_chunk_size", _old_calculate_col_chunk_size)


def _compile(fn, args, static_argnames, patch, use_old):
  """Compiles `fn` with the old or the current tile-size function; returns it and the tile size it picked."""
  module, name, old_fn = patch
  tile_fn = old_fn if use_old else getattr(module, name)
  tiles = []

  def spy(*a):
    tiles.append(tile_fn(*a))
    return tiles[-1]

  jitted = jax.jit(fn.__wrapped__, static_argnames=static_argnames)
  # The trace cache is keyed on the function, not the jit wrapper; clear it so this variant re-traces.
  jax.clear_caches()
  with mock.patch.object(module, name, spy):
    compiled = jitted.lower(*args).compile()
  return compiled, tiles[0]


def _gather_case(key, num_tokens, topk, hidden_size, dtype):
  """Forward dispatch of `ragged_sort`: every token copied to `topk` expert-sorted slots."""
  kx, ki = jax.random.split(key)
  x = jax.random.normal(kx, (num_tokens, hidden_size), dtype)
  indices = jax.random.permutation(ki, jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), topk))
  n = num_tokens * topk
  args = (x, indices, jnp.zeros((1,), jnp.int32), jnp.full((1,), n, jnp.int32))
  return ragged_gather.ragged_gather, args, (), _GATHER_PATCH


def _reduce_case(key, num_tokens, topk, hidden_size, dtype):
  """Backward combine of `ragged_sort`: `topk` sorted slots summed back into each token."""
  kx, ki = jax.random.split(key)
  n = num_tokens * topk
  x = jax.random.normal(kx, (n, hidden_size), dtype)
  indices = jax.random.permutation(ki, jnp.arange(n, dtype=jnp.int32))
  args = (x, indices, jnp.ones((n,), jnp.float32), jnp.ones((n,), jnp.bool_), topk)
  return ragged_gather_reduce_v2.ragged_gather_reduce, args, ("reduce_group_size",), _REDUCE_PATCH


def _time_us(compiled, args, iters):
  """Mean wall time of one call over `iters` back-to-back calls, in microseconds."""
  out = None
  start = time.perf_counter()
  for _ in range(iters):
    out = compiled(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - start) / iters * 1e6


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--hidden-sizes", default="2048,4096,7168,8192")
  parser.add_argument("--num-tokens", type=int, default=4096)
  parser.add_argument("--topk", type=int, default=8)
  parser.add_argument("--dtype", default="bfloat16")
  parser.add_argument("--iters", type=int, default=20, help="Calls per timed trial.")
  parser.add_argument("--trials", type=int, default=30, help="Timed trials per variant; old and new alternate.")
  parser.add_argument("--tolerance", type=float, default=0.03, help="Allowed slowdown of new vs old, as a fraction.")
  args = parser.parse_args(argv)

  info = pltpu.get_tpu_info()
  if info.sparse_core is None:
    print(f"{jax.devices()[0].device_kind} has no SparseCore; the kernels would use the JAX fallback.")
    return 2
  print(f"device={jax.devices()[0].device_kind} sc_vmem={info.sparse_core.vmem_capacity_bytes // 1024} KiB")
  print(f"num_tokens={args.num_tokens} topk={args.topk} dtype={args.dtype} iters={args.iters} trials={args.trials}")
  print()
  print("| kernel | hidden | tile old | tile new | old us (median) | new us (median) | new/old | identical |")
  print("|---|---|---|---|---|---|---|---|")

  dtype = jnp.dtype(args.dtype)
  regressions = []
  for hidden_size in (int(h) for h in args.hidden_sizes.split(",")):
    for kernel, make_case in (("gather", _gather_case), ("gather_reduce", _reduce_case)):
      fn, case_args, static_argnames, patch = make_case(jax.random.key(0), args.num_tokens, args.topk, hidden_size, dtype)
      old, tile_old = _compile(fn, case_args, static_argnames, patch, use_old=True)
      new, tile_new = _compile(fn, case_args, static_argnames, patch, use_old=False)
      # Static arguments are baked into the executable.
      dyn_args = tuple(a for a in case_args if isinstance(a, jax.Array))

      identical = bool(np.array_equal(np.asarray(old(*dyn_args)), np.asarray(new(*dyn_args))))
      for compiled in (old, new):
        _time_us(compiled, dyn_args, args.iters)  # Warm up.

      old_us, new_us = [], []
      for _ in range(args.trials):
        old_us.append(_time_us(old, dyn_args, args.iters))
        new_us.append(_time_us(new, dyn_args, args.iters))
      old_med, new_med = statistics.median(old_us), statistics.median(new_us)
      ratio = new_med / old_med

      print(
          f"| {kernel} | {hidden_size} | {tile_old} | {tile_new} | {old_med:.1f} | {new_med:.1f} | {ratio:.3f} |"
          f" {identical} |"
      )
      if not identical or ratio > 1 + args.tolerance:
        regressions.append((kernel, hidden_size, ratio, identical))

  print()
  if regressions:
    for kernel, hidden_size, ratio, identical in regressions:
      print(f"FAIL {kernel} hidden={hidden_size}: new/old={ratio:.3f} identical={identical}")
    return 1
  print(f"PASS: no kernel is slower than {1 + args.tolerance:.2f}x the old tiling, and all outputs are identical.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
