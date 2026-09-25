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

"""Correctness, layout and benchmark tests for the spatial-minor TGMM kernel."""

import os
import re
import time
from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental import layout
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2_lib
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_lib
import numpy as np
import pytest

pytestmark = pytest.mark.tpu_only

Format = layout.Format
Layout = layout.Layout


def assert_running_on_tpu() -> None:
  """Fails loudly if the test landed on CPU instead of a real TPU.

  `--test_output=streamed` forces local execution, so Forge never allocates the
  Ghostfish requested by the target's `requires-ghostfish` tag and JAX silently
  falls back to CPU. Kernel tests then fail with a confusing `get_tpu_info`
  error, but pure-JAX benchmark arms would happily report meaningless timings.
  """
  device = jax.devices()[0]
  logging.info(
      "jax devices=%s platform=%s device_kind=%s",
      jax.devices(),
      device.platform,
      device.device_kind,
  )
  if device.platform != "tpu":
    raise AssertionError(
        f"Expected to run on TPU, got platform={device.platform!r}"
        f" device_kind={device.device_kind!r}. Do not pass"
        " --test_output=streamed; it forces local (CPU) execution."
    )


def _fmt(major_to_minor: tuple[int, ...]) -> Any:
  """Builds a concrete single-device Format for the given major_to_minor."""
  return Format(
      Layout(major_to_minor=major_to_minor),
      jax.sharding.SingleDeviceSharding(jax.devices()[0]),
  )


# Target logical shape: [g=16, ?=2048, ?=7168] bf16.
G, D1, D2 = 16, 2048, 7168


def _hlo_text(compiled: Any) -> str:
  text = compiled.as_text()
  assert text is not None
  return text


def _entry_result_layout(compiled: Any) -> str:
  """Extracts the ENTRY computation's result layout string from HLO text."""
  for line in _hlo_text(compiled).splitlines():
    if line.startswith("ENTRY "):
      return line.strip()
  return "<no ENTRY line>"


def _output_bytes(compiled: Any) -> int:
  mem = compiled.memory_analysis()
  assert mem is not None
  return mem.output_size_in_bytes


def _peak_bytes(compiled: Any) -> int:
  """Peak HBM: live temporaries + arguments + output."""
  mem = compiled.memory_analysis()
  assert mem is not None
  return mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes


def entry_computation_layout(compiled: Any) -> str:
  """Extracts the `entry_computation_layout={...}` clause from the HLO header.

  The `ENTRY` line does not carry layouts; the layout string with its tiling
  (e.g. `bf16[16,2048,7168]{0,1,2:T(8,128)(2,1)}`) only appears in the module
  header. This is the authoritative record of what the compiler actually
  materialises, as opposed to what we asked for.

  Args:
    compiled: A compiled JAX executable to read the HLO module header from.

  Returns:
    The full `entry_computation_layout={...}` clause, including the balanced
    braces.
  """
  text = _hlo_text(compiled)
  marker = "entry_computation_layout="
  start = text.find(marker)
  if start < 0:
    raise AssertionError("no entry_computation_layout in HLO module header")
  # The clause runs to the end of the balanced {...} group that follows.
  brace_start = text.index("{", start)
  depth = 0
  for i in range(brace_start, len(text)):
    if text[i] == "{":
      depth += 1
    elif text[i] == "}":
      depth -= 1
      if depth == 0:
        return text[start : i + 1]
  raise AssertionError("unterminated entry_computation_layout clause")


class LayoutProbeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # minor_to_major {1,0,2}  ->  major_to_minor (2,0,1)
      ("m2m_201__minor_to_major_102", (2, 0, 1)),
      # minor_to_major {0,1,2}  ->  major_to_minor (2,1,0)
      ("m2m_210__minor_to_major_012", (2, 1, 0)),
  )
  def test_layout_and_padding(self, major_to_minor):
    x = jnp.zeros((G, D1, D2), dtype=jnp.bfloat16)

    f = jax.jit(
        lambda a: a + jnp.bfloat16(1),
        out_shardings=_fmt(major_to_minor),
    )
    compiled = f.lower(x).compile()

    logging.info("=== major_to_minor=%s ===", major_to_minor)
    logging.info("ENTRY line: %s", _entry_result_layout(compiled))
    logging.info("output_size_in_bytes: %s", _output_bytes(compiled))
    logging.info("logical bytes: %s", G * D1 * D2 * 2)
    logging.info("HLO head:\n%s", "\n".join(_hlo_text(compiled).splitlines()[:40]))
    logging.info("result format: %r", f(x).format)

  def test_default_layout(self):
    x = jnp.zeros((G, D1, D2), dtype=jnp.bfloat16)
    f = jax.jit(lambda a: a + jnp.bfloat16(1))
    compiled = f.lower(x).compile()
    logging.info("DEFAULT ENTRY line: %s", _entry_result_layout(compiled))
    logging.info("DEFAULT output_size_in_bytes: %s", _output_bytes(compiled))
    logging.info("DEFAULT result format: %r", f(x).format)

  def test_transpose_is_bitcast(self):
    """Does a logical transpose of a [d2,g,d1] array fold into a bitcast?"""
    y = jnp.zeros((D2, G, D1), dtype=jnp.bfloat16)

    # transpose (1, 2, 0): [d2, g, d1] -> [g, d1, d2]
    f = jax.jit(
        lambda a: jnp.transpose(a, (1, 2, 0)),
        out_shardings=_fmt((2, 0, 1)),
    )
    compiled = f.lower(y).compile()
    logging.info("TRANSPOSE ENTRY: %s", _entry_result_layout(compiled))
    logging.info("TRANSPOSE HLO:\n%s", _hlo_text(compiled))


def reference_tgmm_spatial_minor(
    lhs: np.ndarray,  # [m, k]
    rhs: np.ndarray,  # [m, n]
    group_sizes: np.ndarray,  # [size_lhs_group]
    num_groups: int,
    group_offset: int = 0,
) -> np.ndarray:
  """Loop reference producing `[g, n, k]` with `out[g, n, k] == dW_g[k, n]`.

  `group_sizes` partitions the rows of `lhs`/`rhs` in order and is indexed
  globally. `group_offset` selects which window of it is computed: groups
  `group_sizes[group_offset : group_offset + num_groups]`. The groups before
  the offset are not computed but *do* consume rows, so the row cursor starts
  at their cumulative size rather than at zero. The result is written locally,
  i.e. `out[i]` holds the product for global group `group_offset + i`.

  Args:
    lhs: The left-hand side array, `[m, k]`.
    rhs: The right-hand side array, `[m, n]`.
    group_sizes: Per-group row counts, `[size_lhs_group]`.
    num_groups: Number of groups to compute, i.e. `num_actual_groups`.
    group_offset: Index of the first group to compute.

  Returns:
    A `[num_groups, n, k]` float32 array.
  """
  size_k = lhs.shape[1]
  size_n = rhs.shape[1]
  out = np.zeros((num_groups, size_n, size_k), dtype=np.float32)
  # Rows owned by the groups before the offset are skipped, not re-based.
  start = int(np.sum(group_sizes[:group_offset]))
  for i in range(num_groups):
    end = start + int(group_sizes[group_offset + i])
    if end > start:
      # dW_g = lhs[g].T @ rhs[g], shape [k, n]. We store its transpose.
      dw = lhs[start:end].astype(np.float32).T @ rhs[start:end].astype(np.float32)
      out[i] = dw.T
    start = end
  return out


class CorrectnessTest(parameterized.TestCase):
  """Numerical correctness of `tgmm_spatial_minor_v2` vs a loop reference."""

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  @parameterized.named_parameters(
      ("even_2g", 512, 256, 256, [256, 256]),
      ("uneven_2g", 512, 256, 256, [128, 384]),
      ("first_empty", 512, 256, 256, [0, 512]),
      ("last_empty", 512, 256, 256, [512, 0]),
      ("one_row", 512, 256, 256, [1, 511]),
      ("one_row_last", 512, 256, 256, [511, 1]),
      ("all_empty", 512, 256, 256, [0, 0]),
      ("g4_mixed", 512, 256, 384, [0, 200, 0, 312]),
      ("g16_even", 1024, 512, 256, [64] * 16),
      ("g16_sparse", 1024, 512, 256, [0, 0, 512, 0, 0, 0, 512] + [0] * 9),
      ("unaligned_m", 500, 256, 256, [123, 377]),
      # Non-power-of-2 and lane/sublane-unaligned k and n. The kernel pads both
      # up to a whole tile and slices the padding back off, so these exercise
      # that the slice boundaries are right.
      ("unaligned_k", 512, 250, 256, [200, 312]),
      ("unaligned_n", 512, 256, 200, [200, 312]),
      ("odd_k_and_n", 512, 257, 129, [256, 256]),
      ("prime_k_and_n", 512, 199, 61, [100, 412]),
      ("all_unaligned", 500, 250, 200, [123, 377]),
      # Larger g: 64 exceeds the f32 `bg` word alignment of 8 and stays within
      # one 128 lane group chunk; 128 is a full lane's worth of groups.
      ("g64_even", 1024, 256, 256, [16] * 64),
      ("g128_sparse", 1024, 128, 128, [8] * 128),
  )
  def test_matches_reference(self, m, k, n, group_sizes_list):
    num_groups = len(group_sizes_list)
    self.assertLessEqual(sum(group_sizes_list), m)

    rng = np.random.default_rng(0)
    lhs_np = rng.normal(size=(m, k)).astype(np.float32) / 8.0
    rhs_np = rng.normal(size=(m, n)).astype(np.float32) / 8.0
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)

    lhs = jnp.asarray(lhs_np, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rhs_np, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    got = tgmm_lib.tgmm_spatial_minor_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        out_major_to_minor=tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
    )
    self.assertEqual(got.shape, (num_groups, n, k))

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-2, atol=2e-2)

  @parameterized.named_parameters(
      ("even_2g", 512, 256, 256, [256, 256]),
      ("uneven_4g", 512, 256, 384, [0, 200, 0, 312]),
  )
  def test_matches_existing_tgmm_v2(self, m, k, n, group_sizes_list):
    """The new kernel must agree with the shipped `tgmm_v2`, modulo transpose."""
    num_groups = len(group_sizes_list)
    rng = np.random.default_rng(1)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray(group_sizes_list, dtype=np.int32))

    baseline = tgmm_lib.tgmm_v2(  # [g, k, n]
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.float32,
    )
    got = tgmm_lib.tgmm_spatial_minor_v2(  # [g, n, k]
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        out_major_to_minor=tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(jnp.transpose(baseline, (0, 2, 1))),
        rtol=2e-2,
        atol=2e-2,
    )


class FallbackEquivalenceTest(parameterized.TestCase):
  """Both dispatch arms must produce the same `[g, n, k]` values.

  `tgmm_spatial_minor_v2` routes to the native spatial-minor Pallas kernel only
  when `out_major_to_minor == SPATIAL_MINOR_MAJOR_TO_MINOR`; every other layout
  falls back to `tgmm_v2` plus a logical transpose. Those are two completely
  different kernels, so the fact that they agree is a property that has to be
  measured on hardware, not asserted by construction.
  """

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  @parameterized.named_parameters(
      ("even_2g", 512, 256, 256, [256, 256]),
      ("uneven_2g", 512, 256, 256, [128, 384]),
      ("g4_mixed", 512, 256, 384, [0, 200, 0, 312]),
      ("g16_even", 1024, 512, 256, [64] * 16),
      ("g16_sparse", 1024, 512, 256, [0, 0, 512, 0, 0, 0, 512] + [0] * 9),
      ("unaligned_m", 500, 256, 256, [123, 377]),
      ("unaligned_k", 512, 250, 256, [200, 312]),
      ("unaligned_n", 512, 256, 200, [200, 312]),
      ("all_unaligned", 500, 250, 200, [123, 377]),
      ("g64_even", 1024, 256, 256, [16] * 64),
  )
  def test_fallback_matches_spatial_minor(self, m, k, n, group_sizes_list):
    num_groups = len(group_sizes_list)
    self.assertLessEqual(sum(group_sizes_list), m)

    rng = np.random.default_rng(7)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    def run(out_major_to_minor):
      return tgmm_lib.tgmm_spatial_minor_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=num_groups,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
      )

    spatial = run(tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR)
    # The default, and a third explicit non-spatial-minor permutation, must
    # both take the fallback and land on the same values.
    default = run(tgmm_lib.DEFAULT_MAJOR_TO_MINOR)
    other = run(M2M_102)

    for name, arr in (
        ("spatial", spatial),
        ("default", default),
        ("other", other),
    ):
      self.assertEqual(arr.shape, (num_groups, n, k), msg=name)
      self.assertEqual(arr.dtype, jnp.float32, msg=name)

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    # Both arms must match the NumPy loop reference ...
    np.testing.assert_allclose(np.asarray(spatial), want, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(default), want, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(other), want, rtol=2e-2, atol=2e-2)
    # ... and, more tightly, each other. Both accumulate in f32 on the same
    # MXU, so the only permitted difference is reduction order.
    np.testing.assert_allclose(np.asarray(default), np.asarray(spatial), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.asarray(other), np.asarray(default), rtol=1e-2, atol=1e-2)

  def test_omitting_the_layout_takes_the_fallback(self):
    """`tgmm_gnk_v2` with no layout argument must equal the explicit default.

    This is the behaviour that makes the fast path the one you get by
    accident: `tgmm_gnk_v2` runs `tgmm_v2` plus a transpose unless asked
    otherwise. The values are the same either way, which is what this pins
    down. (`tgmm_spatial_minor_v2` defaults the other way; see
    `EntryPointNamingTest`.)
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [200, 312]
    num_groups = len(group_sizes_list)
    rng = np.random.default_rng(11)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    implicit = tgmm_lib.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.float32,
    )
    explicit = tgmm_lib.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        out_major_to_minor=tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_array_equal(np.asarray(implicit), np.asarray(explicit))

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    np.testing.assert_allclose(np.asarray(implicit), want, rtol=2e-2, atol=2e-2)


class GroupOffsetTest(parameterized.TestCase):
  """Non-zero `group_offset` must be honoured identically by both arms.

  `group_offset` selects a window of `group_sizes`: the kernel computes groups
  `group_sizes[q : q + num_actual_groups]` and writes them to `out[0:...]`,
  i.e. the output is indexed *locally* while `group_sizes` and the row cursor
  are indexed *globally*. The groups before `q` are not computed but still own
  rows, so they advance the row cursor.

  Both arms reach this through completely separate group metadata. The
  `tgmm_v2` fallback fills it with `gmm_v2.fill_metadata`. The spatial minor
  wrapper precomputes `m_bounds`, `g_first` and `g_last` from `group_sizes`
  and `group_offset`, and prefetches them into SMEM. Agreement is therefore a
  property that has to be measured, not assumed. Every case below checks both
  arms against the NumPy reference *and* against each other.
  """

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  @parameterized.named_parameters(
      # (name, m, k, n, full group_sizes, num_actual_groups, group_offset)
      ("skip_first_of_4", 512, 256, 256, [128, 96, 160, 128], 2, 1),
      ("skip_two_of_4", 512, 256, 256, [128, 96, 160, 128], 2, 2),
      ("last_group_only", 512, 256, 256, [128, 96, 160, 128], 1, 3),
      ("whole_window_offset_0", 512, 256, 256, [128, 96, 160, 128], 4, 0),
      # The skipped prefix is empty, so the row cursor must still start at 0.
      ("skipped_prefix_empty", 512, 256, 256, [0, 200, 0, 312], 2, 1),
      # The computed window contains empty groups.
      ("empty_inside_window", 512, 256, 256, [100, 0, 412, 0], 3, 1),
      # A window that starts unaligned to the sublane tiling of m.
      ("unaligned_row_cursor", 512, 256, 256, [100, 150, 162, 100], 2, 1),
      # m is not a whole number of sublane rows, so the dispatcher pads it.
      ("unaligned_m", 500, 256, 256, [123, 77, 200, 100], 2, 1),
      # Non-tile-aligned k and n on top of an offset.
      ("unaligned_k_and_n", 512, 250, 200, [128, 96, 160, 128], 2, 1),
      # A wider group axis: window of 4 taken from the middle of 16.
      ("g16_window_4", 1024, 256, 256, [64] * 16, 4, 6),
      # Offset past a long run of empty groups.
      (
          "g16_sparse_offset",
          1024,
          256,
          256,
          [0, 0, 512, 0, 0, 0, 512] + [0] * 9,
          8,
          4,
      ),
  )
  def test_group_offset_matches_reference_on_both_arms(self, m, k, n, group_sizes_list, num_groups, offset):
    self.assertLessEqual(sum(group_sizes_list), m)
    self.assertLessEqual(offset + num_groups, len(group_sizes_list))

    rng = np.random.default_rng(23)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)
    group_offset = jnp.asarray(np.asarray([offset], dtype=np.int32))

    def run(out_major_to_minor):
      return tgmm_lib.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_groups,
          group_offset,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
      )

    spatial = run(tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR)
    fallback = run(tgmm_lib.DEFAULT_MAJOR_TO_MINOR)

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
        group_offset=offset,
    )
    self.assertEqual(want.shape, (num_groups, n, k))

    for name, arr in (("spatial", spatial), ("fallback", fallback)):
      self.assertEqual(arr.shape, (num_groups, n, k), msg=name)
      np.testing.assert_allclose(np.asarray(arr), want, rtol=2e-2, atol=2e-2, err_msg=name)
    # The two arms must also agree with each other, more tightly than either
    # agrees with the f32 reference.
    np.testing.assert_allclose(np.asarray(fallback), np.asarray(spatial), rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      ("spatial_minor", tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR),
      ("fallback", tgmm_lib.DEFAULT_MAJOR_TO_MINOR),
  )
  def test_offset_actually_shifts_the_window(self, out_major_to_minor):
    """A different offset must give a different answer.

    Without this, every assertion in
    `test_group_offset_matches_reference_on_both_arms` would still pass if
    `group_offset` were quietly ignored by *both* the kernel and the
    reference, since they would then be consistently wrong together. Here the
    group sizes and contents differ per group, so shifting the window by one
    must change the result, and the result at offset `q` must equal the
    reference's window at offset `q`.

    Args:
      out_major_to_minor: Layout hint selecting the dispatch arm.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = 2
    rng = np.random.default_rng(29)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    def run(offset):
      return np.asarray(
          tgmm_lib.tgmm_gnk_v2(
              lhs,
              rhs,
              group_sizes,
              num_groups,
              jnp.asarray(np.asarray([offset], dtype=np.int32)),
              out_major_to_minor=out_major_to_minor,
              preferred_element_type=jnp.float32,
          )
      )

    at0, at1, at2 = run(0), run(1), run(2)

    # Sanity: the windows genuinely differ, so the comparisons below have
    # something to catch.
    self.assertFalse(np.allclose(at0, at1, rtol=1e-3, atol=1e-3))
    self.assertFalse(np.allclose(at1, at2, rtol=1e-3, atol=1e-3))

    lhs_np = np.asarray(lhs, dtype=np.float32)
    rhs_np = np.asarray(rhs, dtype=np.float32)
    for offset, got in ((0, at0), (1, at1), (2, at2)):
      want = reference_tgmm_spatial_minor(lhs_np, rhs_np, group_sizes_np, num_groups, group_offset=offset)
      np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2, err_msg=f"offset={offset}")

    # An explicit zero offset must be identical to omitting the argument.
    implicit = np.asarray(
        tgmm_lib.tgmm_gnk_v2(
            lhs,
            rhs,
            group_sizes,
            num_groups,
            out_major_to_minor=out_major_to_minor,
            preferred_element_type=jnp.float32,
        )
    )
    np.testing.assert_array_equal(implicit, at0)

  @parameterized.named_parameters(
      ("spatial_minor", tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR),
      ("fallback", tgmm_lib.DEFAULT_MAJOR_TO_MINOR),
  )
  def test_offset_accepts_scalar_and_rank1_spellings(self, out_major_to_minor):
    """A Python int, a 0-d array and a `(1,)` array must all mean the same.

    Args:
      out_major_to_minor: Layout hint selecting the dispatch arm.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = 2
    rng = np.random.default_rng(31)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray(group_sizes_list, dtype=np.int32))

    def run(offset):
      return np.asarray(
          tgmm_lib.tgmm_gnk_v2(
              lhs,
              rhs,
              group_sizes,
              num_groups,
              offset,
              out_major_to_minor=out_major_to_minor,
              preferred_element_type=jnp.float32,
          )
      )

    rank1 = run(jnp.asarray(np.asarray([1], dtype=np.int32)))
    scalar_0d = run(jnp.asarray(np.int32(1)))
    python_int = run(1)

    np.testing.assert_array_equal(scalar_0d, rank1)
    np.testing.assert_array_equal(python_int, rank1)


class InputValidationTest(parameterized.TestCase):
  """Malformed operands must be rejected up front, not deep inside Pallas."""

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  def _call(self, lhs, rhs, group_sizes, num_groups=2, **kwargs):
    return tgmm_lib.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.float32,
        **kwargs,
    )

  @parameterized.named_parameters(
      ("lhs_3d", (2, 256, 128), (512, 256), "lhs", 3),
      ("lhs_1d", (512,), (512, 256), "lhs", 1),
      ("rhs_3d", (512, 128), (2, 256, 256), "rhs", 3),
      ("rhs_1d", (512, 128), (512,), "rhs", 1),
      ("lhs_4d", (1, 1, 512, 128), (512, 256), "lhs", 4),
  )
  def test_rejects_wrong_rank(self, lhs_shape, rhs_shape, operand, rank):
    lhs = jnp.zeros(lhs_shape, dtype=jnp.bfloat16)
    rhs = jnp.zeros(rhs_shape, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes)

    message = str(ctx.exception)
    self.assertIn(f"tgmm {operand} must be a rank-2", message)
    self.assertIn(f"got rank {rank}", message)
    self.assertIn("Batched (>2D) operands are not supported", message)

  def test_rejects_mismatched_m(self):
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((256, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes)

    message = str(ctx.exception)
    self.assertIn("must agree on the contracted size_m dimension", message)
    self.assertIn("lhs.shape[0]=512", message)
    self.assertIn("rhs.shape[0]=256", message)

  def test_rejects_non_1d_group_sizes(self):
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.zeros((2, 1), dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes)

    message = str(ctx.exception)
    self.assertIn("tgmm group_sizes must be a rank-1", message)
    self.assertIn("got rank 2", message)

  @parameterized.named_parameters(
      ("not_a_permutation", (0, 1, 1)),
      ("wrong_length", (0, 1)),
      ("out_of_range", (0, 1, 3)),
  )
  def test_rejects_bad_layout(self, out_major_to_minor):
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes, out_major_to_minor=out_major_to_minor)

    message = str(ctx.exception)
    self.assertIn("out_major_to_minor must be a permutation of (0, 1, 2)", message)
    self.assertIn(str(out_major_to_minor), message)

  def test_accepts_valid_shapes(self):
    """The happy path must survive all of the above checks."""
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    out = self._call(lhs, rhs, group_sizes)
    self.assertEqual(out.shape, (2, 256, 128))

  def test_validate_tgmm_inputs_without_operands_is_unchanged(self):
    """Existing `tgmm_v2` callers pass no operands and must be unaffected."""
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    tgmm_lib.validate_tgmm_inputs(group_sizes, 2)

    with self.assertRaises(ValueError) as ctx:
      tgmm_lib.validate_tgmm_inputs(group_sizes, 4)
    self.assertIn("must be >= group_offset", str(ctx.exception))

  def test_validate_tgmm_inputs_with_operands(self):
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    good_lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    good_rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    tgmm_lib.validate_tgmm_inputs(group_sizes, 2, lhs=good_lhs, rhs=good_rhs)

    bad_lhs = jnp.zeros((2, 256, 128), dtype=jnp.bfloat16)
    with self.assertRaises(ValueError) as ctx:
      tgmm_lib.validate_tgmm_inputs(group_sizes, 2, lhs=bad_lhs, rhs=good_rhs)
    self.assertIn("tgmm lhs must be a rank-2", str(ctx.exception))

    with self.assertRaises(ValueError) as ctx:
      tgmm_lib.validate_tgmm_inputs(group_sizes, 2, lhs=good_lhs)
    self.assertIn("takes lhs and rhs together or not at all", str(ctx.exception))


class DtypeCoverageTest(parameterized.TestCase):
  """Value correctness across input, output and accumulator dtypes.

  Two things vary independently and both matter:

    * The *input* dtype sets the sublane tiling of `m` (8 for f32, 16 for
      bf16), so it changes how the dispatcher pads `m` and how both kernels
      reshape their operands.
    * The *output* dtype is also the stage dtype on the spatial minor path, so
      it selects the epilogue: bf16 packs pairs of `n` rows into uint32 words
      before transposing, while f32 transposes directly.

  `num_groups = 4` pads the group axis to `Gp = 8` on the spatial minor path,
  so every case also covers the zeroed padding groups and the final slice
  that drops them.
  """

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  @parameterized.named_parameters(
      # (name, in_dtype, preferred_element_type, acc_dtype, tol)
      ("bf16_in_f32_out", jnp.bfloat16, jnp.float32, None, 2e-2),
      ("bf16_in_bf16_out", jnp.bfloat16, jnp.bfloat16, None, 4e-2),
      ("bf16_in_default_out", jnp.bfloat16, None, None, 4e-2),
      ("f32_in_f32_out", jnp.float32, jnp.float32, None, 1e-2),
      ("f32_in_bf16_out", jnp.float32, jnp.bfloat16, None, 4e-2),
      ("f32_in_default_out", jnp.float32, None, None, 1e-2),
      # Explicit accumulator alongside an explicit output dtype.
      ("f32_in_f32_out_f32_acc", jnp.float32, jnp.float32, jnp.float32, 1e-2),
      (
          "bf16_in_bf16_out_f32_acc",
          jnp.bfloat16,
          jnp.bfloat16,
          jnp.float32,
          4e-2,
      ),
      ("f32_in_bf16_out_f32_acc", jnp.float32, jnp.bfloat16, jnp.float32, 4e-2),
  )
  def test_dtypes_match_reference_on_both_arms(self, in_dtype, out_dtype, acc_dtype, tol):
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = len(group_sizes_list)

    rng = np.random.default_rng(37)
    lhs_np = rng.normal(size=(m, k)).astype(np.float32) / 8.0
    rhs_np = rng.normal(size=(m, n)).astype(np.float32) / 8.0
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)

    lhs = jnp.asarray(lhs_np, dtype=in_dtype)
    rhs = jnp.asarray(rhs_np, dtype=in_dtype)
    group_sizes = jnp.asarray(group_sizes_np)

    # `preferred_element_type=None` means "use lhs.dtype".
    want_dtype = jnp.dtype(out_dtype if out_dtype is not None else in_dtype)

    def run(out_major_to_minor):
      return tgmm_lib.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=num_groups,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=out_dtype,
          acc_dtype=acc_dtype,
      )

    spatial = run(tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR)
    fallback = run(tgmm_lib.DEFAULT_MAJOR_TO_MINOR)

    # Round the reference through the input dtype so the comparison measures
    # the kernel, not the operand quantisation.
    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )

    for name, arr in (("spatial", spatial), ("fallback", fallback)):
      self.assertEqual(arr.shape, (num_groups, n, k), msg=name)
      self.assertEqual(arr.dtype, want_dtype, msg=name)
      np.testing.assert_allclose(
          np.asarray(arr, dtype=np.float32),
          want,
          rtol=tol,
          atol=tol,
          err_msg=name,
      )
    np.testing.assert_allclose(
        np.asarray(fallback, dtype=np.float32),
        np.asarray(spatial, dtype=np.float32),
        rtol=tol,
        atol=tol,
    )

  @parameterized.named_parameters(
      ("f32_out_align_g_8", jnp.float32),
      ("bf16_out_align_g_16", jnp.bfloat16),
  )
  def test_explicit_bg_respects_output_dtype_alignment(self, out_dtype):
    """`bg` must be a multiple of `32 // out_dtype.itemsize`.

    `align_g` is 8 for an f32 output and 16 for a bf16 one, so the same `bg`
    is legal for one and rejected for the other. This pins that the rule is
    driven by the output dtype rather than the input dtype.

    Args:
      out_dtype: The `preferred_element_type` under test.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = len(group_sizes_list)
    align_g = 32 // jnp.dtype(out_dtype).itemsize

    rng = np.random.default_rng(41)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    group_sizes = jnp.asarray(group_sizes_np)

    def run(bg):
      return tgmm_lib.tgmm_spatial_minor_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=num_groups,
          preferred_element_type=out_dtype,
          bg=bg,
      )

    # An aligned bg computes the right answer.
    got = run(align_g)
    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    np.testing.assert_allclose(np.asarray(got, dtype=np.float32), want, rtol=4e-2, atol=4e-2)

    # A misaligned one is rejected, naming the required multiple.
    with self.assertRaises(ValueError) as ctx:
      run(align_g + 1)
    self.assertIn(f"must be a multiple of {align_g}", str(ctx.exception))


class SpatialMinorOnlyArgumentTest(parameterized.TestCase):
  """Arguments only the spatial-minor kernel can honour must not be dropped.

  `tile_info` and `bg` belong to the spatial minor Pallas kernel: `tile_info`
  sets its tiling and `bg` is validated against its output alignment. On the
  `tgmm_v2` fallback there is nowhere to forward them, so supplying them
  alongside a non-spatial-minor `out_major_to_minor` is always a caller
  mistake: the request would otherwise vanish with no signal. `None` means
  "not supplied" and must keep working on both arms.
  """

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  def _operands(self, m=512, k=256, n=256):
    rng = np.random.default_rng(43)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    return lhs, rhs, group_sizes

  @parameterized.named_parameters(
      ("default", tgmm_lib.DEFAULT_MAJOR_TO_MINOR),
      # `M2M_102` is defined further down the module, so spell it out here.
      ("other_permutation", (2, 0, 1)),
  )
  def test_bg_on_fallback_is_rejected(self, out_major_to_minor):
    """An explicit `bg` on a fallback layout is refused, naming the argument.

    Args:
      out_major_to_minor: A non-spatial-minor layout hint.
    """
    lhs, rhs, group_sizes = self._operands()
    with self.assertRaises(ValueError) as ctx:
      tgmm_lib.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
          bg=16,
      )
    message = str(ctx.exception)
    self.assertIn("bg", message)
    self.assertIn("spatial-minor kernel only", message)
    self.assertIn(str(out_major_to_minor), message)

  def test_tile_info_on_fallback_is_rejected(self):
    lhs, rhs, group_sizes = self._operands()
    tiles = gmm_v2_lib.TileSizes(tile_m=128, tile_k=128, tile_n=128)
    with self.assertRaises(ValueError) as ctx:
      tgmm_lib.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
          preferred_element_type=jnp.float32,
          tile_info=tiles,
      )
    message = str(ctx.exception)
    self.assertIn("tile_info", message)
    self.assertIn("tgmm_v2 directly", message)

  def test_both_are_named_in_one_message(self):
    lhs, rhs, group_sizes = self._operands()
    tiles = gmm_v2_lib.TileSizes(tile_m=128, tile_k=128, tile_n=128)
    with self.assertRaises(ValueError) as ctx:
      tgmm_lib.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
          preferred_element_type=jnp.float32,
          tile_info=tiles,
          bg=16,
      )
    message = str(ctx.exception)
    self.assertIn("bg", message)
    self.assertIn("tile_info", message)

  def test_accepted_on_the_spatial_minor_arm(self):
    """The same arguments are accepted when the spatial minor kernel runs."""
    lhs, rhs, group_sizes = self._operands()
    out = tgmm_lib.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=2,
        out_major_to_minor=tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
        tile_info=gmm_v2_lib.TileSizes(tile_m=128, tile_k=128, tile_n=128),
        bg=8,
    )
    self.assertEqual(out.shape, (2, 256, 256))

  def test_omitting_them_still_works_on_both_arms(self):
    """`None` must continue to mean "not supplied" on the fallback."""
    lhs, rhs, group_sizes = self._operands()
    for out_major_to_minor in (
        tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
        tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR,
    ):
      out = tgmm_lib.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
          tile_info=None,
          bg=None,
      )
      self.assertEqual(out.shape, (2, 256, 256), msg=str(out_major_to_minor))

  def test_rejects_group_sizes_shorter_than_num_actual_groups(self):
    """The offset-independent half of the `validate_tgmm_inputs` bound."""
    lhs, rhs, _ = self._operands()
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    with self.assertRaises(ValueError) as ctx:
      tgmm_lib.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=4,
          preferred_element_type=jnp.float32,
      )
    message = str(ctx.exception)
    self.assertIn("group_sizes has 2 entries", message)
    self.assertIn("num_actual_groups=4", message)


class EntryPointNamingTest(parameterized.TestCase):
  """Each entry point's *default* must match what its name promises.

  `tgmm_gnk_v2` is named for its result shape, `[g, n, k]`, and says nothing
  about the implementation, so it defaults to the fast `tgmm_v2` fallback.
  `tgmm_spatial_minor_v2` is named for the kernel, so it defaults to running
  it. Passing `out_major_to_minor` explicitly makes the two identical.

  The claim is invisible in the returned values (both arms compute the same
  thing), so it is read out of the Pallas scope name in the compiled HLO.
  """

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  def _compile(self, fn):
    m, k, n = 512, 256, 256
    rng = np.random.default_rng(47)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    return jax.jit(fn).lower(lhs, rhs, group_sizes).compile()

  def test_gnk_default_selects_the_fallback(self):
    compiled = self._compile(
        lambda l, r, g: tgmm_lib.tgmm_gnk_v2(l, r, g, num_actual_groups=2, preferred_element_type=jnp.float32)
    )
    scope = kernel_scope_from_hlo(compiled)
    logging.info("tgmm_gnk_v2 default scope: %s", scope)
    self.assertStartsWith(scope, "tgmm_v2-")
    self.assertNotIn("tgmm_spatial_minor_v2", scope)

  def test_spatial_minor_default_selects_the_spatial_minor_kernel(self):
    compiled = self._compile(
        lambda l, r, g: tgmm_lib.tgmm_spatial_minor_v2(l, r, g, num_actual_groups=2, preferred_element_type=jnp.float32)
    )
    scope = kernel_scope_from_hlo(compiled)
    logging.info("tgmm_spatial_minor_v2 default scope: %s", scope)
    self.assertStartsWith(scope, "tgmm_spatial_minor_v2-")

  def test_alias_forwards_an_explicit_layout(self):
    """With an explicit layout the alias must behave exactly like `gnk`."""
    compiled = self._compile(
        lambda l, r, g: tgmm_lib.tgmm_spatial_minor_v2(
            l,
            r,
            g,
            num_actual_groups=2,
            out_major_to_minor=tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
            preferred_element_type=jnp.float32,
        )
    )
    scope = kernel_scope_from_hlo(compiled)
    self.assertStartsWith(scope, "tgmm_v2-")

  @parameterized.named_parameters(
      ("spatial_minor", tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR),
      ("fallback", tgmm_lib.DEFAULT_MAJOR_TO_MINOR),
  )
  def test_alias_and_gnk_agree_value_for_value(self, out_major_to_minor):
    """Given the same explicit layout, both names must return the same array.

    Args:
      out_major_to_minor: The layout hint passed explicitly to both.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = len(group_sizes_list)
    rng = np.random.default_rng(53)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray(group_sizes_list, dtype=np.int32))
    group_offset = jnp.asarray(np.asarray([1], dtype=np.int32))

    kwargs = {
        "num_actual_groups": num_groups - 1,
        "out_major_to_minor": out_major_to_minor,
        "preferred_element_type": jnp.float32,
    }
    via_gnk = tgmm_lib.tgmm_gnk_v2(lhs, rhs, group_sizes, group_offset=group_offset, **kwargs)
    via_alias = tgmm_lib.tgmm_spatial_minor_v2(lhs, rhs, group_sizes, group_offset=group_offset, **kwargs)
    np.testing.assert_array_equal(np.asarray(via_gnk), np.asarray(via_alias))


# ---------------------------------------------------------------------------
# Target configuration, benchmark arms.
# ---------------------------------------------------------------------------

# The deliverable tensor is bf16[16, 2048, 7168] with layout {0,1,2}. The
# kernel's logical result is [g, n, k], so n = 2048 and k = 7168.
TARGET_N = 2048
TARGET_K = 7168
TARGET_M = 4096

# `major_to_minor` tuples, and the layout string each one prints as. XLA's
# layout string is `minor_to_major`, i.e. the reverse of `major_to_minor`
# (xla_data.proto LayoutProto: "from minor (fastest varying index) to major").
M2M_012 = (2, 1, 0)  # prints {0,1,2}: k major, n sublane, g minormost (lanes)
M2M_102 = (2, 0, 1)  # prints {1,0,2}: k major, g sublane, n minormost (lanes)

_MIB = 1024 * 1024


def even_group_sizes(m: int, g: int) -> np.ndarray:
  sizes = np.full((g,), m // g, dtype=np.int32)
  sizes[: m % g] += 1
  return sizes


def make_inputs(m: int, k: int, n: int, g: int, seed: int = 0):
  rng = np.random.default_rng(seed)
  lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
  rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
  return lhs, rhs, jnp.asarray(even_group_sizes(m, g))


def arm_tgmm_v2_plus_relayout(g: int):
  """Shipped `tgmm_v2` -> `[g, k, n]`, then a logical transpose to `[g, n, k]`.

  Whether that transpose is a real relayout copy or a free bitcast depends
  entirely on the output layout it is pinned to, which is what arms 1 and 3
  vary.

  Args:
    g: Number of groups (experts) to pass as `num_actual_groups`.

  Returns:
    A callable `(lhs, rhs, group_sizes) -> [g, n, k]` array.
  """

  def f(lhs, rhs, gs):
    out = tgmm_lib.tgmm_v2(lhs, rhs, gs, num_actual_groups=g, preferred_element_type=jnp.bfloat16)
    return jnp.transpose(out, (0, 2, 1))

  return f


def arm_spatial_minor(g: int, vmem_limit_bytes: int | None = None):
  """The new kernel, which emits the `[k, n, g]` byte order natively.

  Args:
    g: Number of groups (experts) to pass as `num_actual_groups`.
    vmem_limit_bytes: Optional scoped-VMEM reservation override.

  Returns:
    A callable `(lhs, rhs, group_sizes) -> [g, n, k]` array.
  """

  def f(lhs, rhs, gs):
    return tgmm_lib.tgmm_spatial_minor_v2(
        lhs,
        rhs,
        gs,
        num_actual_groups=g,
        out_major_to_minor=tgmm_lib.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.bfloat16,
        vmem_limit_bytes=vmem_limit_bytes,
    )

  return f


def arm_dispatch(g: int, out_major_to_minor: tuple[int, ...]):
  """`tgmm_spatial_minor_v2` invoked through its layout dispatch.

  Args:
    g: Number of groups (experts) to pass as `num_actual_groups`.
    out_major_to_minor: The layout hint that selects the implementation.

  Returns:
    A callable `(lhs, rhs, group_sizes) -> [g, n, k]` array.
  """

  def f(lhs, rhs, gs):
    return tgmm_lib.tgmm_spatial_minor_v2(
        lhs,
        rhs,
        gs,
        num_actual_groups=g,
        out_major_to_minor=out_major_to_minor,
        preferred_element_type=jnp.bfloat16,
    )

  return f


def jit_to_layout(f, major_to_minor):
  return jax.jit(f, out_shardings=_fmt(major_to_minor))


def kernel_scope_from_hlo(compiled: Any) -> str:
  """Recovers the tiling the kernel actually selected, from the HLO.

  The two kernels name their scopes differently: `tgmm_spatial_minor_v2`
  encodes `tm/tk/tn/bg`, while the shipped `tgmm_v2` encodes `act` and no tile
  sizes. Match both.

  Args:
    compiled: A compiled JAX executable to search for a Pallas scope name.

  Returns:
    The lexicographically first matching scope name, or a placeholder string
    if the HLO contains no Pallas scope.
  """
  pattern = r"tgmm(?:_spatial_minor)?_v2-g_\d+-m_\d+-k_\d+-[A-Za-z0-9_\-]+"
  found = re.findall(pattern, _hlo_text(compiled))
  return sorted(set(found))[0] if found else "<no pallas scope in HLO>"


def bench_ms(fn, args, warmup: int = 3, iters: int = 20) -> float:
  """Median-free mean wall time per call, in milliseconds."""
  out = fn(*args)
  jax.block_until_ready(out)
  for _ in range(warmup - 1):
    out = fn(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(iters):
    out = fn(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / iters * 1e3


def tflops(m: int, k: int, n: int, ms: float) -> float:
  return (2.0 * m * k * n) / (ms * 1e-3) / 1e12


def trace_dir(name: str) -> str:
  base = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
  path = os.path.join(base, "xprof", name)
  os.makedirs(path, exist_ok=True)
  return path


ARMS = (
    ("arm1_tgmm_v2_plus_copy", arm_tgmm_v2_plus_relayout, M2M_012, "{0,1,2}"),
    ("arm2_spatial_minor", arm_spatial_minor, M2M_012, "{0,1,2}"),
    ("arm3_tgmm_v2_to_102", arm_tgmm_v2_plus_relayout, M2M_102, "{1,0,2}"),
)


class LayoutAssertionTest(parameterized.TestCase):
  """The new kernel must materialise exactly the requested layout."""

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  def test_spatial_minor_produces_target_layout(self):
    g = 16
    lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
    f = jit_to_layout(arm_spatial_minor(g), M2M_012)
    compiled = f.lower(lhs, rhs, gs).compile()

    clause = entry_computation_layout(compiled)
    logging.info("arm2 g=16 entry_computation_layout:\n%s", clause)
    logging.info("arm2 g=16 kernel scope: %s", kernel_scope_from_hlo(compiled))
    logging.info("arm2 g=16 output bytes: %d", _output_bytes(compiled))

    self.assertIn("bf16[16,2048,7168]{0,1,2:T(8,128)(2,1)}", clause)

  def test_all_arms_report_their_layouts(self):
    """Records what each arm actually materialises, for the report."""
    for g in (16, 128):
      lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
      for name, arm, m2m, expect in ARMS:
        compiled = jit_to_layout(arm(g), m2m).lower(lhs, rhs, gs).compile()
        clause = entry_computation_layout(compiled)
        logging.info(
            "g=%d %s expect=%s\n  scope=%s\n  out_bytes=%d peak_bytes=%d\n  %s",
            g,
            name,
            expect,
            kernel_scope_from_hlo(compiled),
            _output_bytes(compiled),
            _peak_bytes(compiled),
            clause,
        )
        # The layout string always carries its tiling, e.g.
        # `bf16[16,2048,7168]{0,1,2:T(8,128)(2,1)}`, so match up to the colon.
        self.assertIn(f"bf16[{g},2048,7168]{expect[:-1]}:", clause.replace(" ", ""))


class FallbackLayoutTest(parameterized.TestCase):
  """The layout dispatch must pick the right kernel and honour the layout.

  Two independent claims are checked from the compiled HLO, at the real target
  shape:

    1. The requested layout genuinely materialises (`entry_computation_layout`).
    2. The non-spatial-minor requests genuinely avoid the slow kernel. That is
       the whole point of the fallback, and it is invisible in the values, so
       it has to be read out of the Pallas scope name in the HLO.
  """

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  @parameterized.named_parameters(
      # (name, out_major_to_minor hint, layout to pin, expected layout string,
      #  expected Pallas scope prefix)
      (
          "spatial_minor_012",
          M2M_012,
          M2M_012,
          "{0,1,2}",
          "tgmm_spatial_minor_v2-",
      ),
      ("fallback_102", M2M_102, M2M_102, "{1,0,2}", "tgmm_v2-"),
      (
          "fallback_default_210",
          tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
          tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
          "{2,1,0}",
          "tgmm_v2-",
      ),
      # The honest worst case: the caller wants `{0,1,2}` but did not ask for
      # the spatial-minor kernel, so they get `tgmm_v2` plus a real relayout
      # copy. At this shape that is still faster than the kernel it declines
      # to use; see `BenchmarkTest`.
      (
          "fallback_hint_mismatch_012",
          tgmm_lib.DEFAULT_MAJOR_TO_MINOR,
          M2M_012,
          "{0,1,2}",
          "tgmm_v2-",
      ),
  )
  def test_dispatch_selects_kernel_and_layout(self, hint, pinned, expect_layout, expect_scope_prefix):
    g = 16
    lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
    compiled = jit_to_layout(arm_dispatch(g, hint), pinned).lower(lhs, rhs, gs).compile()

    clause = entry_computation_layout(compiled)
    scope = kernel_scope_from_hlo(compiled)
    logging.info("hint=%s pinned=%s -> scope=%s\n  %s", hint, pinned, scope, clause)

    self.assertIn(
        f"bf16[{g},{TARGET_N},{TARGET_K}]{expect_layout[:-1]}:",
        clause.replace(" ", ""),
    )
    self.assertStartsWith(scope, expect_scope_prefix)
    if expect_scope_prefix == "tgmm_v2-":
      self.assertNotIn("tgmm_spatial_minor_v2", scope)


class HbmPreflightTest(absltest.TestCase):
  """Measures each arm's HBM footprint before anything large is launched."""

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  def test_hbm_budget(self):
    stats = jax.local_devices()[0].memory_stats()
    assert stats is not None
    limit = stats.get("bytes_limit")
    logging.info("device memory_stats: %s", stats)
    logging.info("HBM bytes_limit: %s", limit)

    rows = []
    for g in (16, 128):
      lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
      for name, arm, m2m, expect in ARMS:
        compiled = jit_to_layout(arm(g), m2m).lower(lhs, rhs, gs).compile()
        rows.append((g, name, expect, _output_bytes(compiled), _peak_bytes(compiled)))

    logging.info(
        "%-4s %-24s %-9s %14s %14s",
        "g",
        "arm",
        "layout",
        "out_bytes",
        "peak_bytes",
    )
    for g, name, expect, out_b, peak_b in rows:
      logging.info("%-4d %-24s %-9s %14d %14d", g, name, expect, out_b, peak_b)
      if limit is not None:
        self.assertLess(peak_b, limit, f"arm {name} at g={g} does not fit in HBM")


class VmemSweepTest(parameterized.TestCase):
  """Sweeps the scoped VMEM reservation for the spatial minor kernel.

  The reservation is not a neutral knob: it feeds the tiling heuristic, so it
  directly sets the kernel's tile size and hence its arithmetic intensity. For
  each reservation this logs the tiling selected (read from the kernel scope
  name) and the measured latency.
  """

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  @parameterized.named_parameters(("g16", 16), ("g128", 128))
  def test_arm2_vmem_sweep(self, g):
    lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
    results = []
    for mib in (16, 24, 32, 40, 48, 56):
      try:
        f = jit_to_layout(arm_spatial_minor(g, vmem_limit_bytes=mib * _MIB), M2M_012)
        compiled = f.lower(lhs, rhs, gs).compile()
        scope = kernel_scope_from_hlo(compiled)
        ms = bench_ms(f, (lhs, rhs, gs))
        results.append((mib, scope, ms, tflops(TARGET_M, TARGET_K, TARGET_N, ms)))
        logging.info("g=%d vmem=%dMiB %s -> %.4f ms", g, mib, scope, ms)
      except Exception as e:  # pylint: disable=broad-except
        logging.info("g=%d vmem=%dMiB FAILED: %s", g, mib, str(e)[:400])
        results.append((mib, "<failed>", float("nan"), float("nan")))

    logging.info("=== VMEM sweep, arm2, g=%d ===", g)
    logging.info("%-8s %-58s %10s %10s", "vmem", "scope", "ms", "TFLOP/s")
    for mib, scope, ms, tf in results:
      logging.info("%-8s %-58s %10.4f %10.3f", f"{mib}MiB", scope, ms, tf)


class BenchmarkTest(parameterized.TestCase):
  """3 arms x 2 group configs, with an XProf trace per arm."""

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  @parameterized.named_parameters(("g16", 16), ("g128", 128))
  def test_benchmark_arms(self, g):
    lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
    logging.info(
        "shapes: lhs=%s rhs=%s g=%d (m=%d k=%d n=%d)",
        lhs.shape,
        rhs.shape,
        g,
        TARGET_M,
        TARGET_K,
        TARGET_N,
    )

    results = []
    for name, arm, m2m, expect in ARMS:
      f = jit_to_layout(arm(g), m2m)
      compiled = f.lower(lhs, rhs, gs).compile()
      scope = kernel_scope_from_hlo(compiled)
      out_b = _output_bytes(compiled)
      peak_b = _peak_bytes(compiled)

      ms = bench_ms(f, (lhs, rhs, gs))

      tag = f"{name}_g{g}"
      with jax.profiler.trace(trace_dir(tag)):
        for _ in range(5):
          out = f(lhs, rhs, gs)
        jax.block_until_ready(out)

      results.append(
          (
              name,
              expect,
              scope,
              out_b,
              peak_b,
              ms,
              tflops(TARGET_M, TARGET_K, TARGET_N, ms),
          )
      )
      logging.info("%s g=%d -> %.4f ms (trace tag %s)", name, g, ms, tag)

    logging.info(
        "=== BENCHMARK g=%d (m=%d k=%d n=%d bf16) ===",
        g,
        TARGET_M,
        TARGET_K,
        TARGET_N,
    )
    logging.info(
        "%-24s %-9s %10s %10s %14s %14s  %s",
        "arm",
        "layout",
        "ms",
        "TFLOP/s",
        "out_bytes",
        "peak_bytes",
        "scope",
    )
    for name, expect, scope, out_b, peak_b, ms, tf in results:
      logging.info(
          "%-24s %-9s %10.4f %10.3f %14d %14d  %s",
          name,
          expect,
          ms,
          tf,
          out_b,
          peak_b,
          scope,
      )


class EpilogueTest(parameterized.TestCase):
  # pylint: disable=invalid-name,cell-var-from-loop

  def setUp(self):
    super().setUp()
    assert_running_on_tpu()

  def test_gate2_dynamic_start_rejected_by_mosaic(self):
    """Shows the strided per pair epilogue fails Gate 2 with a dynamic start.

    That epilogue handles one uint32 row `i` of the stage at a time, where each
    row packs a pair of bf16 `n` rows. It loads row `i` of every group with a
    strided load (stride `bn // 2`), transposes the `[Gp, 128]` result, and
    stores it into the output with the same stride.

    Mosaic's `apply_vector_layout` strictly requires dynamic indices along
    tiled dimensions to be divisible by the tile size (4). A dynamic loop
    variable `i = base + s` (where s in [1..7]) cannot be proven divisible
    by 4 and is rejected by the compiler.
    """
    bk = 128
    bn = 128
    Gp = 128
    half = bn // 2

    def epilogue_kernel(stage_ref, out_ref):
      stage_u32 = stage_ref.bitcast(jnp.uint32)
      out_u32 = out_ref.bitcast(jnp.uint32)
      out2d = out_u32.reshape(bk * half, Gp)
      for kc in range(bk // 128):
        src2d = stage_u32.at[kc].reshape(Gp * half, 128)

        def pair_block(ib, carry):
          base = pl.multiple_of(ib * 8, 8)
          for s in range(8):
            i = base + s
            t = src2d[pl.ds(i, Gp, stride=half), :]
            out2d[pl.ds(kc * 128 * half + i, 128, stride=half), :] = t.T
          return carry

        lax.fori_loop(0, half // 8, pair_block, None)

    stage = jax.random.normal(jax.random.PRNGKey(0), (bk // 128, Gp, bn, 128), dtype=jnp.bfloat16)

    @jax.jit
    def run_epilogue(s):
      return pl.pallas_call(
          epilogue_kernel,
          out_shape=jax.ShapeDtypeStruct((bk, bn, Gp), jnp.bfloat16),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(1,),
              in_specs=[pl.BlockSpec((bk // 128, Gp, bn, 128), lambda *_: (0, 0, 0, 0))],
              out_specs=pl.BlockSpec((bk, bn, Gp), lambda *_: (0, 0, 0)),
          ),
      )(s)

    with self.assertRaises(Exception) as ctx:
      out = run_epilogue(stage)
      out.block_until_ready()
    self.assertIn("divisible by the tiling", str(ctx.exception))
    logging.info(
        "Confirmed: dynamic base+s is rejected by Mosaic as predicted by" " Gate 2: %s",
        ctx.exception,
    )

  def test_gate0_section6_2_epilogue_proposal(self):
    bk = 128
    bn = 128
    Gp = 128
    half = bn // 2

    def epilogue_kernel(stage_ref, out_ref):
      stage_u32 = stage_ref.bitcast(jnp.uint32)
      out_u32 = out_ref.bitcast(jnp.uint32)
      out2d = out_u32.reshape(bk * half, Gp)
      for kc in range(bk // 128):
        src2d = stage_u32.at[kc].reshape(Gp * half, 128)

        for i in range(half):
          t = src2d[pl.ds(i, Gp, stride=half), :]
          out2d[pl.ds(kc * 128 * half + i, 128, stride=half), :] = t.T

    stage = jax.random.normal(jax.random.PRNGKey(0), (bk // 128, Gp, bn, 128), dtype=jnp.bfloat16)

    @jax.jit
    def run_epilogue(s):
      return pl.pallas_call(
          epilogue_kernel,
          out_shape=jax.ShapeDtypeStruct((bk, bn, Gp), jnp.bfloat16),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(1,),
              in_specs=[pl.BlockSpec((bk // 128, Gp, bn, 128), lambda *_: (0, 0, 0, 0))],
              out_specs=pl.BlockSpec((bk, bn, Gp), lambda *_: (0, 0, 0)),
          ),
      )(s)

    run_epilogue_jit = jax.jit(run_epilogue)
    compiled = run_epilogue_jit.lower(stage).compile()
    logging.info("HLO text:\n%s", compiled.as_text())

    out = compiled(stage)
    out.block_until_ready()
    logging.info("Epilogue proposal succeeded! Output shape: %s", out.shape)

    # Expected: out[k, n, g] == stage[k // 128, g, n, k % 128]
    # stage: (bk // 128, Gp, bn, 128) -> (0: kc, 1: g, 2: n, 3: kk)
    # transpose (0, 3, 2, 1) -> (kc, kk, n, g) -> reshape (bk, bn, Gp)
    expected = jnp.transpose(stage, (0, 3, 2, 1)).reshape(bk, bn, Gp)
    np.testing.assert_array_equal(np.array(out), np.array(expected))

  def test_candidate_bn_gp_bk(self):
    bk = 128
    bn = 128
    Gp = 128

    def kernel(stage_ref, out_ref):
      for kc in range(bk // 128):
        for n in range(bn):
          tile = stage_ref[n, :, kc * 128 : (kc + 1) * 128]  # (Gp, 128)
          t = jnp.transpose(tile, (1, 0))  # (128, Gp)
          out_ref[kc * 128 : (kc + 1) * 128, n, :] = t

    stage = jax.random.normal(jax.random.PRNGKey(0), (bn, Gp, bk), dtype=jnp.bfloat16)

    @jax.jit
    def run_kernel(s):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((bk, bn, Gp), jnp.bfloat16),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(1,),
              in_specs=[pl.BlockSpec((bn, Gp, bk), lambda *_: (0, 0, 0))],
              out_specs=pl.BlockSpec((bk, bn, Gp), lambda *_: (0, 0, 0)),
          ),
      )(s)

    out = run_kernel(stage)
    out.block_until_ready()
    logging.info("Candidate bn_gp_bk succeeded! Output shape: %s", out.shape)

    # stage: (0: n, 1: g, 2: k) -> transpose (2, 0, 1) -> (k, n, g)
    expected = jnp.transpose(stage, (2, 0, 1))
    np.testing.assert_array_equal(np.array(out), np.array(expected))

  @parameterized.named_parameters(
      ("bk128_bn128", 128, 128),
      ("bk256_bn128", 256, 128),
      ("bk256_bn64", 256, 64),
  )
  def test_gate0_scratch_epilogue_multichunk(self, bk, bn):
    Gp = 128
    half = bn // 2

    def kernel(inp_ref, out_ref, stage_ref):
      # Phase 1 simulation: write inp_ref into scratch stage_ref
      for kc in range(bk // 128):
        for g in range(Gp):
          stage_ref[kc, g, :, :] = inp_ref[kc, g, :, :]

      # Phase 2: Transpose epilogue from scratch stage_ref to out_ref
      stage_u32 = stage_ref.bitcast(jnp.uint32)
      out_u32 = out_ref.bitcast(jnp.uint32)
      out2d = out_u32.reshape(bk * half, Gp)
      for kc in range(bk // 128):
        src2d = stage_u32.at[kc].reshape(Gp * half, 128)
        for i in range(half):
          t = src2d[pl.ds(i, Gp, stride=half), :]
          out2d[pl.ds(kc * 128 * half + i, 128, stride=half), :] = t.T

    stage_shape = (bk // 128, Gp, bn, 128)
    out_shape = (bk, bn, Gp)
    inp = jax.random.normal(jax.random.PRNGKey(1), stage_shape, dtype=jnp.bfloat16)

    @jax.jit
    def run_kernel(x):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(1,),
              in_specs=[pl.BlockSpec(stage_shape, lambda *_: (0, 0, 0, 0))],
              out_specs=pl.BlockSpec(out_shape, lambda *_: (0, 0, 0)),
              scratch_shapes=[pltpu.VMEM(stage_shape, jnp.bfloat16)],
          ),
      )(x)

    out = run_kernel(inp)
    out.block_until_ready()
    logging.info(
        "Scratch epilogue (bk=%d, bn=%d) succeeded! Output shape: %s",
        bk,
        bn,
        out.shape,
    )

    # Expected: out[k, n, g] == inp[k // 128, g, n, k % 128]
    expected = jnp.transpose(inp, (0, 3, 2, 1)).reshape(bk, bn, Gp)
    np.testing.assert_array_equal(np.array(out), np.array(expected))

  def test_gate5_epilogue_throughput(self):
    Gp = 128

    for bk, bn in [(128, 128), (256, 128)]:
      half = bn // 2
      stage_shape = (bk // 128, Gp, bn, 128)
      out_shape = (bk, bn, Gp)

      # Approach A: Proposed Bitcast uint32 + strided load/store
      def kernel_a(stage_ref, out_ref):
        stage_u32 = stage_ref.bitcast(jnp.uint32)
        out_u32 = out_ref.bitcast(jnp.uint32)
        out2d = out_u32.reshape(bk * half, Gp)
        for kc in range(bk // 128):
          src2d = stage_u32.at[kc].reshape(Gp * half, 128)
          for i in range(half):
            t = src2d[pl.ds(i, Gp, stride=half), :]
            out2d[pl.ds(kc * 128 * half + i, 128, stride=half), :] = t.T

      @jax.jit
      def run_a(s):
        return pl.pallas_call(
            kernel_a,
            out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(stage_shape, lambda *_: (0, 0, 0, 0))],
                out_specs=pl.BlockSpec(out_shape, lambda *_: (0, 0, 0)),
            ),
        )(s)

      stage_a = jax.random.normal(jax.random.PRNGKey(0), stage_shape, dtype=jnp.bfloat16)
      ms_a = bench_ms(run_a, (stage_a,), warmup=5, iters=50)

      tile_bytes = bk * bn * Gp * 2  # bf16 output bytes
      gb = tile_bytes / 1e9
      sec = ms_a * 1e-3
      bw_gb_s = gb / sec

      logging.info(
          "=== GATE 5 TIMING: bk=%d bn=%d Gp=128 (tile=%.2f MB) ===",
          bk,
          bn,
          tile_bytes / 1e6,
      )
      logging.info(
          "Approach A (u32 bitcast + strided vld/vst): %.4f ms (%.2f GB/s)",
          ms_a,
          bw_gb_s,
      )

      expected = jnp.transpose(stage_a, (0, 3, 2, 1)).reshape(bk, bn, Gp)

      # Approach C: u32 (128, 8, 128) with (1,0,2) -> (0,2,1) -> (1,0,2)
      def kernel_c(stage_ref, out_ref):
        stage_u32 = stage_ref.bitcast(jnp.uint32)
        out_u32 = out_ref.bitcast(jnp.uint32)
        for kc in range(bk // 128):
          for p in range(half // 8):
            v = stage_u32[kc, :, p * 8 : (p + 1) * 8, :]
            v1 = jnp.transpose(v, (1, 0, 2))
            v2 = jnp.transpose(v1, (0, 2, 1))
            v3 = jnp.transpose(v2, (1, 0, 2))
            out_u32[kc * 128 : (kc + 1) * 128, p * 8 : (p + 1) * 8, :] = v3

      @jax.jit
      def run_c(s):
        return pl.pallas_call(
            kernel_c,
            out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(stage_shape, lambda *_: (0, 0, 0, 0))],
                out_specs=pl.BlockSpec(out_shape, lambda *_: (0, 0, 0)),
            ),
        )(s)

      out_c = run_c(stage_a)
      np.testing.assert_array_equal(np.array(out_c), np.array(expected))
      ms_c = bench_ms(run_c, (stage_a,), warmup=5, iters=50)
      logging.info(
          "Approach C (u32 128x8x128 3-step transpose): %.4f ms (%.2f GB/s) ->" " %.2fx vs A",
          ms_c,
          gb / (ms_c * 1e-3),
          ms_a / ms_c,
      )

      # Approach D: bf16 (128, 16, 128) with (1,0,2) -> (0,2,1) -> (1,0,2)
      def kernel_d(stage_ref, out_ref):
        for kc in range(bk // 128):
          for p in range(bn // 16):
            v = stage_ref[kc, :, p * 16 : (p + 1) * 16, :]
            v1 = jnp.transpose(v, (1, 0, 2))
            v2 = jnp.transpose(v1, (0, 2, 1))
            v3 = jnp.transpose(v2, (1, 0, 2))
            out_ref[kc * 128 : (kc + 1) * 128, p * 16 : (p + 1) * 16, :] = v3

      @jax.jit
      def run_d(s):
        return pl.pallas_call(
            kernel_d,
            out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(stage_shape, lambda *_: (0, 0, 0, 0))],
                out_specs=pl.BlockSpec(out_shape, lambda *_: (0, 0, 0)),
            ),
        )(s)

      out_d = run_d(stage_a)
      np.testing.assert_array_equal(np.array(out_d), np.array(expected))
      ms_d = bench_ms(run_d, (stage_a,), warmup=5, iters=50)
      logging.info(
          "Approach D (bf16 128x16x128 3-step transpose): %.4f ms (%.2f GB/s)" " -> %.2fx vs A",
          ms_d,
          gb / (ms_d * 1e-3),
          ms_a / ms_d,
      )

      # Approach E: test direct transpose(v, (2, 1, 0)) on u32 (128, 8, 128)
      try:

        def kernel_e(stage_ref, out_ref):
          stage_u32 = stage_ref.bitcast(jnp.uint32)
          out_u32 = out_ref.bitcast(jnp.uint32)
          for kc in range(bk // 128):
            for p in range(half // 8):
              v = stage_u32[kc, :, p * 8 : (p + 1) * 8, :]
              out_u32[kc * 128 : (kc + 1) * 128, p * 8 : (p + 1) * 8, :] = jnp.transpose(v, (2, 1, 0))

        @jax.jit
        def run_e(s):
          return pl.pallas_call(
              kernel_e,
              out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
              grid_spec=pltpu.PrefetchScalarGridSpec(
                  num_scalar_prefetch=0,
                  grid=(1,),
                  in_specs=[pl.BlockSpec(stage_shape, lambda *_: (0, 0, 0, 0))],
                  out_specs=pl.BlockSpec(out_shape, lambda *_: (0, 0, 0)),
              ),
          )(s)

        out_e = run_e(stage_a)
        np.testing.assert_array_equal(np.array(out_e), np.array(expected))
        ms_e = bench_ms(run_e, (stage_a,), warmup=5, iters=50)
        logging.info(
            "Approach E (u32 direct transpose(2,1,0)): %.4f ms (%.2f GB/s)",
            ms_e,
            gb / (ms_e * 1e-3),
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.info("Approach E (direct transpose(2,1,0)) failed: %s", str(e)[:300])

      # For bk=128, bn=128, also measure Approach B to compare
      if bk == 128 and bn == 128:

        def kernel_b(stage_ref, out_ref):
          for kc in range(bk // 128):
            for n in range(bn):
              tile = stage_ref[n, :, kc * 128 : (kc + 1) * 128]
              t = jnp.transpose(tile, (1, 0))
              out_ref[kc * 128 : (kc + 1) * 128, n, :] = t

        @jax.jit
        def run_b(s):
          return pl.pallas_call(
              kernel_b,
              out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
              grid_spec=pltpu.PrefetchScalarGridSpec(
                  num_scalar_prefetch=0,
                  grid=(1,),
                  in_specs=[pl.BlockSpec((bn, Gp, bk), lambda *_: (0, 0, 0))],
                  out_specs=pl.BlockSpec(out_shape, lambda *_: (0, 0, 0)),
              ),
          )(s)

        stage_b = jax.random.normal(jax.random.PRNGKey(0), (bn, Gp, bk), dtype=jnp.bfloat16)
        ms_b = bench_ms(run_b, (stage_b,), warmup=5, iters=50)
        logging.info(
            "Approach B (bn_gp_bk slice-by-1): %.4f ms (%.2f GB/s) ->" " Approach A is %.1fx FASTER",
            ms_b,
            gb / (ms_b * 1e-3),
            ms_b / ms_a,
        )


if __name__ == "__main__":
  absltest.main()
