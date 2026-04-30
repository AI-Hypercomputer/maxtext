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

"""Tests for sparse core gather reduce kernel."""

import functools
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels import gather_reduce_pallas
from maxtext.kernels import gather_reduce_sc
import numpy as np


def _snr(signal, grnd_truth):
  error = signal - grnd_truth
  return jnp.log(1 + jnp.sum(signal**2) / (jnp.sum(error**2) + 1e-6))


class GatherReduceScTest(parameterized.TestCase):

  def setUp(self):
    """Skips tests if the TPU version is not supported."""
    if jax.default_backend() == "gpu":
      self.skipTest("gather_reduce_sc kernels are not supported on GPU")
    tpu_info = pltpu.get_tpu_info()
    if tpu_info is None or tpu_info.chip_version not in (pltpu.ChipVersion.TPU_7X,):
      self.skipTest("Expect TPUv7+")

    super().setUp()

  @parameterized.product(
      shape_idx_size=[
          ((128 * 1024, 7 * 1024), 128 * 1024, 8),
      ],
      data_type=[
          "random_int",
          # "debug",
      ],
      use_pallas=[False, True],
  )
  def test_column(self, shape_idx_size, data_type, use_pallas):
    rows, cols = shape_idx_size[0]

    if data_type == "random_int":
      # Use numpy for faster data generation on CPU, then transfer to device.
      rng_inputs = np.random.RandomState(42)
      inputs = rng_inputs.randint(0, 1024, shape_idx_size[0]).astype(jnp.bfloat16)

      rng_others = np.random.RandomState(43)
      idx = rng_others.randint(0, shape_idx_size[0][0], (shape_idx_size[1],), dtype=np.int32)
    elif data_type == "debug":
      # Create inputs with structured, semi-unique small numbers.
      # Each row ascends from 0 to cols-1, with a decimal part indicating the
      # row index.
      base = jnp.tile(jnp.arange(cols, dtype=jnp.bfloat16), (rows, 1))
      offset = jnp.arange(rows, dtype=jnp.bfloat16)[:, None] * 0.01
      inputs = base + offset

      # Structure indices to be ascending from 0.
      idx = jnp.arange(shape_idx_size[1], dtype=jnp.int32)

    else:
      raise ValueError("Unsupported data type: " + data_type)

    def _run_nojit(op, idx):
      group_size = shape_idx_size[2]
      gathered = op[idx, :]
      gathered = jnp.reshape(gathered, (-1, group_size, op.shape[1]))
      return jnp.sum(gathered.astype(jnp.float32), axis=1).astype(jnp.bfloat16)

    module = gather_reduce_pallas if use_pallas else gather_reduce_sc
    kernel = functools.partial(
        module.sc_gather_reduce,
        reduce_group_size=shape_idx_size[2],
        single_sc=True,
    )
    _run_sc = jax.jit(kernel)

    maybe_compile_sc = _run_sc.lower(inputs, idx).compile()
    maybe_compile_sc_og = None
    if not use_pallas:
      maybe_compile_sc_og = _run_sc.lower(
          inputs.astype(jnp.float32),
          idx,
      ).compile()

    out = _run_nojit(inputs, idx)
    out_og = _run_nojit(inputs.astype(jnp.float32), idx)

    out_sc_og = None
    for _ in range(5):
      out_sc = jax.block_until_ready(maybe_compile_sc(inputs, idx))
      if maybe_compile_sc_og is not None:
        out_sc_og = jax.block_until_ready(maybe_compile_sc_og(inputs.astype(jnp.float32), idx))

    np.testing.assert_array_equal(out_sc, out)
    if out_sc_og is not None:
      np.testing.assert_array_equal(out_sc_og, out_og)

  @parameterized.product(
      shape_idx_size=[
          ((128 * 1024, 7 * 1024), 128 * 1024, 8),
      ],
      use_pallas=[False, True],
  )
  def test_topk_mult(self, shape_idx_size, use_pallas):
    timings = {}
    start_time = time.time()

    # Use numpy for faster data generation on CPU, then transfer to device.
    rng_inputs = np.random.RandomState(42)
    inputs = rng_inputs.randint(0, 1024, shape_idx_size[0]).astype(jnp.bfloat16)

    rng_others = np.random.RandomState(43)
    topk_wgt = rng_others.standard_normal(shape_idx_size[1]).astype(jnp.bfloat16)

    idx = rng_others.randint(0, shape_idx_size[0][0], (shape_idx_size[1],), dtype=np.int32)

    for ci, rnd_idx in enumerate([89, 129, 257, 7, 90]):
      inputs[rnd_idx, :] = np.nan
      inds_nan = np.where(idx == rnd_idx)
      topk_wgt[inds_nan] = 0.0
      if len(inds_nan) == 0:
        idx[ci] = rnd_idx

    topk_wgt = topk_wgt.reshape(-1, 128).astype(jnp.bfloat16)

    timings["data_creation"] = time.time() - start_time

    def _run_nojit(op, idx, topk_wgt_local, acc_dtype):
      group_size = 8
      gathered = op[idx, :]
      if topk_wgt_local is not None:
        flat_weights = topk_wgt_local.flatten()
        weights_for_mul = flat_weights[:, None].astype(jnp.float32)
        gathered_f32 = gathered.astype(jnp.float32)
        gathered = jnp.where(weights_for_mul == 0, 0.0, gathered_f32 * weights_for_mul)
      gathered = jnp.reshape(gathered, (-1, group_size, op.shape[1]))
      return jnp.sum(gathered.astype(jnp.float32), axis=1).astype(jnp.bfloat16)

    module = gather_reduce_pallas if use_pallas else gather_reduce_sc
    kernel = functools.partial(
        module.sc_gather_reduce,
        reduce_group_size=shape_idx_size[2],
        single_sc=True,
        topk_wgt_zero_nan=True,
    )

    @jax.jit
    def _run_sc(
        op,
        idx,
        topk_wgt_local,
    ):
      return kernel(op, idx, topk_weights=topk_wgt_local)

    start_time = time.time()
    maybe_compile_sc = _run_sc.lower(
        inputs,
        idx,
        topk_wgt,
    ).compile()
    timings["compilation"] = time.time() - start_time

    start_time = time.time()
    # out = _run_nojit(inputs, idx, topk_wgt, jnp.float32)
    out_bf16 = _run_nojit(inputs, idx, topk_wgt, jnp.bfloat16)
    timings["baseline"] = time.time() - start_time

    start_time = time.time()
    for _ in range(5):
      out_sc = jax.block_until_ready(
          maybe_compile_sc(
              inputs,
              idx,
              topk_wgt,
          )
      )
    timings["kernel_execution"] = time.time() - start_time

    for k, v in timings.items():
      print(f"{k} took: {v:.4f}s")

    # on SC we do accm in fp32 so we need some tolerance.
    np.testing.assert_allclose(out_sc, out_bf16, atol=22, rtol=0.08)
    # np.testing.assert_allclose(out_sc, out, rtol=0.08)


if __name__ == "__main__":
  absltest.main()
