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

"""Column tiling of the SparseCore ragged gather, checked on a CPU host.

`ragged_gather.calculate_col_size` sizes the kernel's column tile from the target chip's SparseCore VMEM. It
must switch on `TpuInfo.generation`, an int. `TpuInfo.chip_version` is a `ChipVersion` enum (on a TPU v7x,
`ChipVersion.TPU_7X`, whose value is "7x"), which matches neither `case 6` nor `case 7`, so switching on it gave
every chip the smallest, 128 KiB, budget. On a TPU v7x at hidden 4096 that meant three 1408-wide column tiles:
the kernel wrote a 4224-wide output that the wrapper sliced back to 4096, a copy XLA materialised before every
expert GEMM, and its last tile read input columns 4096-4223, past the end of every row.

The tests choose the target chip with an `AbstractMesh` whose device is a TPU, which is what
`pltpu.get_tpu_info()` reads, so they need no TPU.

`CalculateColSizeTest` checks the tile at hidden 4096 on v5p, v6e and v7x; that on every chip with a SparseCore
the tiles cover the row and fit the SparseCore VMEM; and that on v5p, v6e and v7x the tile divides the hidden
size of the MoE models that use this kernel, so the kernel's output needs no slice.

`KernelOutputTest` runs the unmodified `main_kernel` in Pallas's TPU interpret mode on a v7x target at hidden
4096, and requires its output to be bit-identical to `x[indices]` for the tile `calculate_col_size` picks and for
other tiles, the old 1408 among them. jax's TPU interpreter (0.11.1) does not model two things the kernel does:
  - A DMA through a bitcast ref. The interpreter computes every access range with
    `jax._src.pallas.mosaic.interpret.utils.to_range`, which only understands indexers, and `main_kernel`
    bitcasts its HBM refs to uint32. The data is therefore uint32, for which that bitcast is the identity, and
    the test leaves identity bitcasts out of the range computation. The kernel's bf16 unpack/repack works
    within one 128-lane chunk and never reads the column tile's offset, so it does not depend on the tile.
  - SparseCore subcores. The interpreter answers `lax.axis_index` of a mesh axis with its own core id, 0, on
    every subcore. The kernel therefore runs on a 1x1 `VectorSubcoreMesh`, where 0 is the right answer, and
    one subcore walks every row block, 16 rows (the v7x SIMD width) at a time.
The launch mirrors the wrapper, `ragged_gather.ragged_gather`, which cannot run the kernel here: it takes its
jnp fallback whenever `jax.devices()` is not a TPU.

The checks are two-sided:
  - The interpreter's bounds check passes the new v7x tile and must reject the old 1408 tile, which reads past
    the row. With out-of-bounds reads allowed, the old tile's output, once sliced, is still exact: the fix
    changes the layout, not the values.
  - Wrong indices, and the old tile's output sliced one lane chunk off, must fail the same comparison.
"""

import functools
import unittest
from unittest import mock

from absl.testing import parameterized
import jax
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.ragged import ragged_gather

_HIDDEN = 4096
_V5P, _V6E, _V7X = "TPU v5", "TPU v6 lite", "TPU7x"
# jax's device kinds for the chips with a SparseCore (jax/_src/tpu_info.py, chip_version_from_device_kind).
_SPARSECORE_CHIPS = (_V5P, _V6E, _V7X, "TPU8i", "TPU8t")
# Hidden sizes of MoE models that use the ragged sort: Qwen3-30B-A3B and Qwen3-Next (2048), Qwen3-235B-A22B and
# Qwen3.5-397B-A17B (4096), Llama 4 (5120), Mixtral 8x22B (6144), DeepSeek-V3 and Kimi K2 (7168).
_MOE_HIDDEN_SIZES = (2048, 4096, 5120, 6144, 7168)

# The kernel test gathers 32 rows of a 40-row table and checks the rows in [_START, _END), the range the kernel
# is asked for; callers mask the rest.
_ROWS, _GATHERED, _START, _END = 40, 32, 3, 29
_OLD_V7X_COL_SIZE = 1408  # What the chip_version switch gave a v7x at hidden 4096: three tiles, 4224 wide.


def _tpu_target(device_kind):
  """Returns a context in which `pltpu.get_tpu_info()` describes a TPU of `device_kind`."""
  device = jax.sharding.AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  return jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=device))


def _col_size(device_kind, hidden_size):
  with _tpu_target(device_kind):
    return ragged_gather.calculate_col_size(hidden_size)


@pytest.mark.cpu_only
class CalculateColSizeTest(parameterized.TestCase):
  """`calculate_col_size` budgets by TPU generation."""

  @parameterized.named_parameters(
      # 128 KiB for an 8-lane row: two tiles.
      ("v5p", _V5P, 2048),
      # 256 KiB for an 8-lane row, and 512 KiB for a 16-lane row: one tile each.
      ("v6e", _V6E, 4096),
      ("v7x", _V7X, 4096),
  )
  def test_col_size_at_hidden_4096(self, device_kind, expected):
    self.assertEqual(_col_size(device_kind, _HIDDEN), expected)

  @parameterized.named_parameters(*((kind.replace(" ", "_"), kind) for kind in _SPARSECORE_CHIPS))
  def test_tiles_cover_the_row_and_fit_vmem(self, device_kind):
    with _tpu_target(device_kind):
      tpu_info = pltpu.get_tpu_info()
      sparse_core = tpu_info.sparse_core
      for hidden_size in range(tpu_info.num_lanes, 16384 + 1, tpu_info.num_lanes):
        col_size = ragged_gather.calculate_col_size(hidden_size)
        num_cols = pl.cdiv(hidden_size, col_size)
        msg = f"{device_kind} hidden {hidden_size}: col {col_size}"
        self.assertEqual(col_size % tpu_info.num_lanes, 0, msg)
        # Every tile holds part of the row.
        self.assertLess((num_cols - 1) * col_size, hidden_size, msg)
        # The kernel's VMEM tile is (SIMD lanes, col_size) uint32.
        self.assertLessEqual(sparse_core.num_lanes * col_size * 4, sparse_core.vmem_capacity_bytes, msg)

  @parameterized.named_parameters(
      *(
          (f"{name}_{hidden_size}", kind, hidden_size)
          for name, kind in (("v5p", _V5P), ("v6e", _V6E), ("v7x", _V7X))
          for hidden_size in _MOE_HIDDEN_SIZES
      )
  )
  def test_tile_divides_the_hidden_size(self, device_kind, hidden_size):
    self.assertEqual(hidden_size % _col_size(device_kind, hidden_size), 0)


def _to_range_without_identity_bitcasts(transforms, *, to_range):
  """`to_range` for accesses whose only bitcasts are to uint32 on uint32 data, i.e. the identity."""
  kept = []
  for transform in transforms:
    if isinstance(transform, state_types.BitcastTransform):
      if transform.dtype != jnp.uint32:
        raise NotImplementedError(f"only identity bitcasts of uint32 data are modelled, got {transform.dtype}")
      continue
    kept.append(transform)
  return to_range(kept)


@functools.partial(jax.jit, static_argnames=("col_size", "out_of_bounds_reads"))
def _interpreted_gather(x, indices, start, end, *, col_size, out_of_bounds_reads):
  """The wrapper's kernel launch (`ragged_gather.ragged_gather`) on a 1x1 SparseCore mesh, in interpret mode.

  Returns every gathered row at the kernel's full, `col_size`-aligned width, before the wrapper's slice.
  """
  sparse_core = pltpu.get_tpu_info().sparse_core
  num_lanes = sparse_core.num_lanes
  hidden_size, out_size = x.shape[-1], indices.size
  # One core with one subcore, so a row block is one row tile of `num_lanes` rows.
  out_pad_size = pl.cdiv(out_size, num_lanes) * num_lanes - out_size
  mesh = plsc.VectorSubcoreMesh(num_cores=1, num_subcores=1, core_axis_name="core", subcore_axis_name="subcore")
  out = pl.kernel(  # pylint: disable=too-many-function-args
      functools.partial(ragged_gather.main_kernel, core_axis_name="core", subcore_axis_name="subcore", has_weights=False),
      out_type=jax.ShapeDtypeStruct((out_size + out_pad_size, pl.cdiv(hidden_size, col_size) * col_size), x.dtype),
      compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True, disable_bounds_checks=True),
      scratch_types=[
          pltpu.VMEM((num_lanes,), jnp.int32),
          pltpu.VMEM((num_lanes,), jnp.int32),
          pltpu.VMEM((num_lanes, col_size), jnp.uint32),
          pltpu.VMEM((num_lanes,), jnp.int32),
          pltpu.VMEM((num_lanes,), jnp.float32),
          pltpu.SemaphoreType.DMA((2,)),
      ],
      mesh=mesh,
      name="sc_ragged_gather",
      interpret=pltpu.InterpretParams(out_of_bounds_reads=out_of_bounds_reads),
  )(start, end, x, jnp.pad(indices, (0, out_pad_size)), jnp.ones((out_size + out_pad_size,), jnp.float32))
  return out[:out_size]


@pytest.mark.cpu_only
class KernelOutputTest(parameterized.TestCase):
  """At hidden 4096 on a v7x target, the kernel's output is `x[indices]` whatever the column tile."""

  def setUp(self):
    """Lets the interpreter model the kernel's identity bitcasts, and draws the rows and indices to gather."""
    super().setUp()
    self.enterContext(
        mock.patch.object(
            interpret_utils,
            "to_range",
            functools.partial(_to_range_without_identity_bitcasts, to_range=interpret_utils.to_range),
        )
    )
    rng = np.random.default_rng(0)
    self.x = rng.integers(0, 2**32, (_ROWS, _HIDDEN), dtype=np.uint64).astype(np.uint32)
    self.indices = rng.integers(0, _ROWS, _GATHERED).astype(np.int32)
    self.expected = self.x[self.indices][_START:_END]

  def _gather(self, col_size, indices=None, out_of_bounds_reads="raise"):
    """The kernel's rows in [_START, _END), at its full width."""
    indices = self.indices if indices is None else indices
    with _tpu_target(_V7X):
      out = _interpreted_gather(
          jnp.asarray(self.x),
          jnp.asarray(indices),
          jnp.array([_START, 0], jnp.int32),
          jnp.array([_END, 0], jnp.int32),
          col_size=col_size,
          out_of_bounds_reads=out_of_bounds_reads,
      )
    return np.asarray(out)[_START:_END]

  def test_v7x_tile_is_one_exact_tile(self):
    out = self._gather(_col_size(_V7X, _HIDDEN))
    self.assertEqual(out.shape, self.expected.shape)
    np.testing.assert_array_equal(out, self.expected)

  @parameterized.named_parameters(("two_tiles", 2048), ("four_tiles", 1024))
  def test_other_tiles_are_exact(self, col_size):
    np.testing.assert_array_equal(self._gather(col_size), self.expected)

  def test_old_v7x_tile_reads_past_the_row(self):
    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, r"Out-of-bounds read .*slice\(4096, 4224"):
      self._gather(_OLD_V7X_COL_SIZE)

  def test_old_v7x_tile_is_exact_once_sliced(self):
    out = self._gather(_OLD_V7X_COL_SIZE, out_of_bounds_reads="uninitialized")
    self.assertEqual(out.shape[1], 3 * _OLD_V7X_COL_SIZE)
    np.testing.assert_array_equal(out[:, :_HIDDEN], self.expected)
    # Positive control: sliced one lane chunk off, the same output must not match.
    self.assertFalse(np.array_equal(out[:, 128 : 128 + _HIDDEN], self.expected))

  def test_wrong_indices_fail(self):
    out = self._gather(_HIDDEN, indices=(self.indices + 1) % _ROWS)
    self.assertFalse(np.array_equal(out, self.expected))


if __name__ == "__main__":
  unittest.main()
