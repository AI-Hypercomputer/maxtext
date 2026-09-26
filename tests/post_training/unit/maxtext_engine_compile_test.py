# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `maxtext_engine_compile`, the engine's AOT compile tool, and its memory report.

A kernel's `memory_analysis()` sees only its own arguments, and counts a donated buffer in both
its arguments and its outputs. `maxtext_engine_compile.memory_report` corrects both. These tests
check the arithmetic on stand-in executables and abstract-mesh avals, then check the report on a
real CPU compile against XLA's own byte counts.

The Zero-1 case needs four devices, and the CPU backend reads
`--xla_force_host_platform_device_count` only at initialization -- already past by the time
pytest imports this file. `test_zero1_report_on_four_cpu_devices` re-execs this module with the
flag set, as `maxtext_engine_xaot_test.py` does, and fails unless the child reports that tests
actually ran.
"""

# Some tests compare the report with the engine's own (private) train state.
# pylint: disable=protected-access

import contextlib
import io
import os
import re
import subprocess
import sys
import types
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train_compile as pre_train_compile
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import maxtext_engine_compile
from maxtext.utils import maxtext_utils
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]

_MIB = 2**20
_GIB = 2**30
_FLOAT32_BYTES = 4

# Two ways, so that a sharded leaf's per-device bytes differ from its global ones. Abstract, so no
# second device is needed: `shard_shape` and `memory_kind` are all the report reads off a sharding.
_MESH = AbstractMesh((2,), ("data",), axis_types=(AxisType.Explicit,))

_ZERO1_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_COMPILE_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")


def _aval(num_bytes: int, spec: P = P(), memory_kind: str | None = None) -> jax.ShapeDtypeStruct:
  """A float32 aval of `num_bytes` global bytes on `_MESH`. Nothing is allocated, so size is free."""
  sharding = NamedSharding(_MESH, spec)
  if memory_kind is not None:
    sharding = sharding.with_memory_kind(memory_kind)
  return jax.ShapeDtypeStruct((num_bytes // _FLOAT32_BYTES,), jnp.float32, sharding=sharding)


def _param(shape: tuple[int, ...], dtype: Any, spec: P = P()) -> nnx.Param:
  """An `nnx.Param` holding an aval on `_MESH`, as a traced train state carries one."""
  return nnx.Param(jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(_MESH, spec)))


def _state(optimizer: dict) -> nnx.State:
  """A train state shaped as `nnx.split(TrainStateNNX(...))` shapes it: a model and an optimizer."""
  return nnx.State({"model": {"kernel": _aval(64 * _MIB)}, "optimizer": optimizer})


class _Compiled:
  """Stands in for `jax.stages.Compiled`, of which the report reads only `memory_analysis()`."""

  def __init__(self, **sizes: int):
    stats = {
        f"{prefix}{field}_size_in_bytes": 0
        for prefix in ("", "host_")
        for field in ("argument", "output", "alias", "temp", "generated_code")
    }
    unknown = set(sizes) - set(stats)
    if unknown:
      raise TypeError(f"not CompiledMemoryStats fields: {sorted(unknown)}")
    stats.update(sizes)
    self._stats = types.SimpleNamespace(**stats)

  def memory_analysis(self) -> types.SimpleNamespace:
    return self._stats


def _kernels(**per_kernel: dict[str, int]) -> dict[str, _Compiled]:
  """One stand-in per kernel, in `KERNEL_NAMES` order; a kernel not given reports all zeros."""
  return {name: _Compiled(**per_kernel.get(name, {})) for name in maxtext_engine_compile.KERNEL_NAMES}


def _by_kernel(rows) -> dict:
  return {row.kernel: row for row in rows}


def _header(lines: list[str]) -> str:
  """The table's column-header line; matched on its first columns, since legend prose can start with "kernel"."""
  return next(line for line in lines if re.match(r"kernel\s+arg\s+out\b", line))


class MemoryReportTest(absltest.TestCase):
  """The arithmetic, on stand-in executables whose numbers are chosen to tell the cases apart."""

  def test_optimizer_added_to_fwd_bwd_not_update(self):
    """Neither forward/backward kernel is passed the optimizer, yet it is in HBM while they run."""
    optimizer = {"mu": _aval(8 * _MIB), "nu": _aval(8 * _MIB)}
    compiled = _kernels(**{name: {"argument_size_in_bytes": 100 * _MIB} for name in maxtext_engine_compile.KERNEL_NAMES})

    rows = _by_kernel(maxtext_engine_compile.memory_report(compiled, _state(optimizer)))

    for name in ("fwd_bwd", "fwd_bwd_accum"):
      with self.subTest(kernel=name):
        self.assertEqual(rows[name].not_passed, ("optimizer",))
        self.assertEqual(rows[name].device_not_passed, 16 * _MIB)
        self.assertEqual(rows[name].device_total, 116 * _MIB)
    # The update's own arguments already include the optimizer; adding it again double-counts it.
    self.assertEqual(rows["update"].not_passed, ())
    self.assertEqual(rows["update"].device_not_passed, 0)
    self.assertEqual(rows["update"].device_total, 100 * _MIB)

  def test_donated_buffer_counted_once(self):
    """`update` donates its 40 MiB state: that buffer is in the 100 MiB of arguments and the 60 of outputs."""
    compiled = _kernels(
        update={
            "argument_size_in_bytes": 100 * _MIB,
            "output_size_in_bytes": 60 * _MIB,
            "alias_size_in_bytes": 40 * _MIB,
            "temp_size_in_bytes": 10 * _MIB,
        }
    )

    row = _by_kernel(maxtext_engine_compile.memory_report(compiled, _state({})))["update"]

    self.assertEqual(row.resident, 130 * _MIB)

  def test_host_bytes_excluded_from_device_peak(self):
    """`update` under optimizer offload, with byte counts taken from a TPU compile.

    The offloaded optimizer state is donated on the host, so it is in the host arguments, the host
    outputs and the host alias alike. None of it counts toward the device peak.
    """
    offloaded = 1_192_568_320
    compiled = _kernels(
        update={
            "argument_size_in_bytes": 1_192_578_048,
            "output_size_in_bytes": 596_295_392,
            "alias_size_in_bytes": 596_294_144,
            "temp_size_in_bytes": 1_529_469_760,
            "host_argument_size_in_bytes": offloaded,
            "host_output_size_in_bytes": offloaded,
            "host_alias_size_in_bytes": offloaded,
        }
    )

    row = _by_kernel(maxtext_engine_compile.memory_report(compiled, _state({})))["update"]

    self.assertEqual((row.host_argument, row.host_output, row.host_alias), (offloaded, offloaded, offloaded))
    self.assertEqual(row.resident, 1_192_578_048 + 596_295_392 - 596_294_144 + 1_529_469_760)
    self.assertEqual(row.device_total, row.resident)

  def test_host_temp_excluded_from_device_peak(self):
    """Host temp, e.g. activations offloaded by `decoder_layer_input=offload`, has its own column.

    It is host RAM, so it is not part of `resident`.
    """
    offloaded = 58_720_256
    forward = {
        "argument_size_in_bytes": 100 * _MIB,
        "temp_size_in_bytes": 10 * _MIB,
        "host_temp_size_in_bytes": offloaded,
    }
    compiled = _kernels(fwd_bwd=forward, fwd_bwd_accum=forward)

    rows = maxtext_engine_compile.memory_report(compiled, _state({}))
    report = maxtext_engine_compile.format_memory_report(rows).splitlines()

    self.assertIn("host temp", _header(report))
    for name in ("fwd_bwd", "fwd_bwd_accum"):
      with self.subTest(kernel=name):
        row = _by_kernel(rows)[name]
        self.assertEqual(row.host_temp, offloaded)
        self.assertEqual(row.resident, 110 * _MIB)
        self.assertEqual(row.device_total, 110 * _MIB)
        cells = next(line for line in report if line.startswith(f"{name} ")).split()
        # kernel, 7 device columns, host arg, host out, host alias, then host temp.
        self.assertEqual(cells[11], f"{offloaded / _GIB:.2f}")
    self.assertEqual(_by_kernel(rows)["update"].host_temp, 0)

  def test_xla_counts_donated_buffer_twice(self):
    """On a real executable, a donated buffer is in `argument` and `output`, and `resident` counts it once."""
    buffer = jnp.zeros((256 * 1024,), jnp.float32)
    update = jax.jit(lambda state, grads: state - grads, donate_argnums=(0,))
    compiled = {"update": update.lower(buffer, buffer).compile()}

    row = maxtext_engine_compile.memory_report(compiled, _state({}))[0]

    self.assertEqual(row.alias, buffer.nbytes)
    self.assertEqual(row.argument + row.output, 3 * buffer.nbytes, "the raw sum counts the donated state twice")
    self.assertEqual(row.resident - row.temp, 2 * buffer.nbytes)

  def test_host_optimizer_not_counted_as_device(self):
    """An optimizer in host memory moves from the forward/backward kernels' `+state` to `host state`."""
    for memory_kind in ("pinned_host", "unpinned_host"):
      with self.subTest(memory_kind=memory_kind):
        optimizer = {"mu": _aval(8 * _MIB, memory_kind=memory_kind), "nu": _aval(8 * _MIB, memory_kind=memory_kind)}

        rows = _by_kernel(maxtext_engine_compile.memory_report(_kernels(), _state(optimizer)))

        self.assertEqual(rows["fwd_bwd"].device_not_passed, 0)
        self.assertEqual(rows["fwd_bwd"].host_not_passed, 16 * _MIB)

  def test_device_and_unset_memory_kind_count_as_device(self):
    """A concrete mesh reports `device`; an abstract one, or a leaf not yet placed, reports None."""
    optimizer = {
        "mu": _aval(8 * _MIB, memory_kind="device"),
        "nu": _aval(8 * _MIB),
        "count": _aval(8 * _MIB, memory_kind="pinned_host"),
    }

    row = _by_kernel(maxtext_engine_compile.memory_report(_kernels(), _state(optimizer)))["fwd_bwd"]

    self.assertEqual(row.device_not_passed, 16 * _MIB)
    self.assertEqual(row.host_not_passed, 8 * _MIB)

  def test_per_device_bytes_follow_sharding(self):
    """What fits is per device: a leaf split two ways costs each device half of it."""
    self.assertEqual(maxtext_engine_compile.per_device_bytes(_aval(8 * _MIB, P("data"))), (4 * _MIB, 0))
    self.assertEqual(maxtext_engine_compile.per_device_bytes(_aval(8 * _MIB)), (8 * _MIB, 0))
    unsharded = jax.ShapeDtypeStruct((2 * _MIB,), jnp.float32)
    self.assertEqual(maxtext_engine_compile.per_device_bytes(unsharded), (8 * _MIB, 0))

    optimizer = {"mu": _aval(8 * _MIB, P("data")), "nu": _aval(8 * _MIB)}
    row = _by_kernel(maxtext_engine_compile.memory_report(_kernels(), _state(optimizer)))["fwd_bwd"]
    self.assertEqual(row.device_not_passed, 12 * _MIB)

  def test_prng_key_leaf_bytes(self):
    """The model's RNG state carries key dtypes, which `np.dtype` refuses; threefry keys are 8 bytes."""
    keys = jax.ShapeDtypeStruct((3,), jax.random.key(0).dtype)

    self.assertEqual(maxtext_engine_compile.per_device_bytes(keys), (24, 0))

  def test_peak_includes_state_not_passed(self):
    """The peak is the largest `device_total`, even when another kernel has the largest `resident`."""
    optimizer = {"mu": _aval(8 * _MIB), "nu": _aval(8 * _MIB)}
    compiled = _kernels(
        fwd_bwd_accum={"argument_size_in_bytes": 60 * _MIB, "temp_size_in_bytes": 40 * _MIB},
        update={"argument_size_in_bytes": 100 * _MIB, "temp_size_in_bytes": 10 * _MIB},
    )

    rows = maxtext_engine_compile.memory_report(compiled, _state(optimizer))
    by_kernel = _by_kernel(rows)
    peak = maxtext_engine_compile.device_peak(rows)

    self.assertGreater(by_kernel["update"].resident, by_kernel["fwd_bwd_accum"].resident)
    self.assertEqual(peak.kernel, "fwd_bwd_accum")
    self.assertEqual(peak.device_total, 116 * _MIB)

  def test_verdict_names_peak_and_parts(self):
    # Sized in whole hundredths of a GiB so the expected strings are exact.
    compiled = _kernels(
        fwd_bwd={"argument_size_in_bytes": 40 * _GIB, "temp_size_in_bytes": 20 * _GIB},
        fwd_bwd_accum={"argument_size_in_bytes": 42 * _GIB, "temp_size_in_bytes": int(20.5 * _GIB)},
        update={"argument_size_in_bytes": 50 * _GIB},
    )
    on_device = {"mu": _aval(int(4.375 * _GIB)), "nu": _aval(int(4.375 * _GIB))}
    on_host = {key: _aval(int(4.375 * _GIB), memory_kind="pinned_host") for key in ("mu", "nu")}
    cases = (
        (on_device, "TRAIN-KERNEL DEVICE PEAK: 71.25 GiB (fwd_bwd_accum 62.50 + optimizer state 8.75 not passed to it)"),
        (
            on_host,
            "TRAIN-KERNEL DEVICE PEAK: 62.50 GiB (fwd_bwd_accum 62.50 + optimizer state 0.00 not passed to it; "
            "8.75 GiB more of it is on host)",
        ),
    )
    for optimizer, verdict in cases:
      with self.subTest(verdict=verdict):
        report = maxtext_engine_compile.format_memory_report(
            maxtext_engine_compile.memory_report(compiled, _state(optimizer))
        )
        lines = report.splitlines()

        self.assertEqual(lines[-1], verdict)
        for name in maxtext_engine_compile.KERNEL_NAMES:
          self.assertLen([line for line in lines if line.startswith(f"{name} ")], 1, name)

  def test_verdict_when_update_is_peak(self):
    compiled = _kernels(update={"argument_size_in_bytes": 50 * _GIB})

    report = maxtext_engine_compile.format_memory_report(maxtext_engine_compile.memory_report(compiled, _state({})))

    self.assertEqual(
        report.splitlines()[-1],
        "TRAIN-KERNEL DEVICE PEAK: 50.00 GiB (update, which is passed all the train state it runs with)",
    )

  def test_table_cells_align_with_headers(self):
    """The table body, cell by cell: every value distinct, so a swapped column or a wrong field shows.

    Sized in quarter GiBs so every cell prints exactly.
    """
    compiled = _kernels(
        fwd_bwd={
            "argument_size_in_bytes": 3 * _GIB,
            "output_size_in_bytes": 2 * _GIB,
            "alias_size_in_bytes": 1 * _GIB,
            "temp_size_in_bytes": 4 * _GIB,
            "host_argument_size_in_bytes": _GIB // 2,
            "host_output_size_in_bytes": 3 * _GIB // 4,
            "host_alias_size_in_bytes": _GIB // 4,
            "host_temp_size_in_bytes": 7 * _GIB // 4,
        }
    )
    optimizer = {"mu": _aval(9 * _GIB // 4), "nu": _aval(11 * _GIB // 4, memory_kind="pinned_host")}
    expected = {
        "arg": "3.00",
        "out": "2.00",
        "alias": "1.00",
        "temp": "4.00",
        "resident": "8.00",
        "+state": "2.25",
        "total": "10.25",
        "host arg": "0.50",
        "host out": "0.75",
        "host alias": "0.25",
        "host temp": "1.75",
        "host state": "2.75",
    }

    lines = maxtext_engine_compile.format_memory_report(
        maxtext_engine_compile.memory_report(compiled, _state(optimizer))
    ).splitlines()
    header = _header(lines)
    row = next(line for line in lines if line.startswith("fwd_bwd "))

    self.assertEqual(row.split(), ["fwd_bwd", *expected.values(), "optimizer"])
    # Right-aligned: each cell ends where its header ends, in this order left to right.
    position = len("kernel")
    for name, cell in expected.items():
      with self.subTest(column=name):
        end = header.index(name, position) + len(name)
        self.assertEqual(row[end - len(cell) : end], cell)
        self.assertEqual(row[end - len(cell) - 1], " ")
        position = end
    self.assertGreater(header.index("state not passed", position), position)

  def test_legend_lists_exclusions(self):
    """The legend lists what the train-kernel peak excludes, so it is not read as the whole step's peak."""
    report = maxtext_engine_compile.format_memory_report(maxtext_engine_compile.memory_report(_kernels(), _state({})))
    lines = report.splitlines()
    legend = " ".join(lines[: lines.index(_header(lines))])

    self.assertNotIn("ENGINE DEVICE PEAK", report)
    for excluded in ("weight-sync staging", "release_weight_sync", "fwd_only", "eval", "generated code", "the caller"):
      with self.subTest(excluded=excluded):
        self.assertIn(excluded, legend)

  def test_rejects_unaccounted_state(self):
    """Each of these would otherwise report a peak lower than the real one."""
    with self.assertRaisesRegex(ValueError, "no entry for kernel 'fwd_only'"):
      maxtext_engine_compile.memory_report({"fwd_only": _Compiled()}, _state({}))
    with self.assertRaisesRegex(ValueError, r"no \['optimizer'\] subtree"):
      maxtext_engine_compile.memory_report(_kernels(), nnx.State({"model": {"kernel": _aval(_MIB)}}))
    # A third subtree, e.g. an EMA of the weights: resident, and in no kernel's arguments or `+state`.
    third = nnx.State({"model": {"kernel": _aval(_MIB)}, "optimizer": {}, "ema": {"kernel": _aval(_MIB)}})
    with self.assertRaisesRegex(ValueError, r"subtrees \['ema'\] that STATE_NOT_PASSED does not classify"):
      maxtext_engine_compile.memory_report(_kernels(), third)

    no_analysis = _Compiled()
    no_analysis.memory_analysis = lambda: None
    with self.assertRaisesRegex(ValueError, "has no memory analysis"):
      maxtext_engine_compile.memory_report({"update": no_analysis}, _state({}))

  def test_weight_sync_staging_counts_new_buffers(self):
    """A cast to another dtype and a per-layer slice each make a buffer; a cast to a leaf's own dtype does not.

    One leaf per case, each a different size so that a case counted wrongly cannot be hidden by
    another counted wrongly the other way.
    """
    model = {
        "decoder": {
            "layers": {
                "mlp": _param((256, 2, 1024), jnp.float32),  # 2 MiB: cast and sliced, 1 MiB staged
                "scale": _param((64, 2, 1024), jnp.bfloat16),  # 256 KiB: sliced only
                "flat": _param((4096,), jnp.bfloat16),  # 8 KiB: no axis to slice, already bf16
            },
            "embedding": _param((2048, 1024), jnp.float32, P("data")),  # 4 MiB per device: cast, 2 MiB staged
            "norm": _param((32, 1024), jnp.bfloat16),  # 64 KiB: neither
        },
        "stats": nnx.BatchStat(jax.ShapeDtypeStruct((1024, 1024), jnp.float32)),  # 4 MiB: not a Param
    }
    state = nnx.State({"model": model, "optimizer": {"mu": _aval(16 * _MIB)}})
    model_bytes = 2 * _MIB + 256 * 1024 + 8 * 1024 + 4 * _MIB + 64 * 1024 + 4 * _MIB

    for scan_layers, held_copy in ((True, 1 * _MIB + 256 * 1024 + 2 * _MIB), (False, 1 * _MIB + 2 * _MIB)):
      with self.subTest(scan_layers=scan_layers):
        staging = maxtext_engine_compile.weight_sync_staging(
            state, scan_layers=scan_layers, scan_axis=1, use_weight_converter=False
        )
        self.assertEqual(staging.held_copy, held_copy)
        self.assertEqual((staging.model, staging.optimizer), (model_bytes, 16 * _MIB))
        self.assertEqual(staging.device_total, model_bytes + 16 * _MIB + held_copy)

    converted = maxtext_engine_compile.weight_sync_staging(
        state, scan_layers=True, scan_axis=1, use_weight_converter=True
    )
    self.assertIsNone(converted.held_copy, "the converter's output layout is not known here")
    self.assertIsNone(converted.device_total)

  def test_weight_sync_line_outside_peak(self):
    compiled = _kernels(fwd_bwd={"argument_size_in_bytes": 10 * _GIB})
    rows = maxtext_engine_compile.memory_report(compiled, _state({}))
    prefix = "WEIGHT-SYNC STAGING, outside the peak below: "
    cases = (
        (
            maxtext_engine_compile.WeightSyncStaging(
                held_copy=int(1.25 * _GIB), model=int(2.5 * _GIB), optimizer=5 * _GIB
            ),
            prefix + "prepare_weight_sync holds a 1.25 GiB copy of the parameters until release_weight_sync, on top "
            "of 7.50 GiB of train state (model 2.50 + optimizer 5.00): 8.75 GiB before the conversion's temporaries",
        ),
        (
            maxtext_engine_compile.WeightSyncStaging(held_copy=4 * _GIB, model=int(2.5 * _GIB), optimizer=5 * _GIB),
            prefix + "prepare_weight_sync holds a 4.00 GiB copy of the parameters until release_weight_sync, on top "
            "of 7.50 GiB of train state (model 2.50 + optimizer 5.00): 11.50 GiB before the conversion's "
            "temporaries; ABOVE the train-kernel device peak",
        ),
        (
            maxtext_engine_compile.WeightSyncStaging(held_copy=None, model=int(2.5 * _GIB), optimizer=5 * _GIB),
            prefix + "not estimated -- use_weight_converter stages the rollout's layout, which this tool does not "
            "see. It is held until release_weight_sync on top of 7.50 GiB of train state (model 2.50 + optimizer "
            "5.00).",
        ),
    )
    for staging, line in cases:
      with self.subTest(line=line):
        lines = maxtext_engine_compile.format_memory_report(rows, staging).splitlines()

        self.assertEqual(lines[-2], line)
        self.assertTrue(lines[-1].startswith("TRAIN-KERNEL DEVICE PEAK: 10.00 GiB"), lines[-1])

  def test_every_kernel_has_state_entry(self):
    """A kernel added to the engine must be classified here, or the report refuses to run."""
    self.assertEqual(sorted(maxtext_engine_compile.STATE_NOT_PASSED), sorted(maxtext_engine_compile.KERNEL_NAMES))


def _tiny_overrides(data_parallelism: int) -> list[str]:
  """A decoder small enough to compile three kernels inside a test, with adamw's two moments."""
  return [
      "model_name=default",
      "enable_checkpointing=False",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
      "enable_dropout=False",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "remat_policy=none",
      "scan_layers=False",
      "attention=dot_product",
      "shard_mode=explicit",
      f"ici_data_parallelism={data_parallelism}",
      "ici_fsdp_parallelism=1",
      "ici_tensor_parallelism=1",
      "per_device_batch_size=1",
      "vocab_size=128",
      "base_emb_dim=64",
      "base_mlp_dim=128",
      "base_num_decoder_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "head_dim=16",
      "max_target_length=32",
      "opt_type=adamw",
      "gradient_accumulation_steps=1",
      "profiler_steps=0",
  ]


def _config(data_parallelism: int, *extra: str) -> pyconfig.HyperParameters:
  """The tiny decoder, with each of `extra` replacing the default of the same key rather than repeating it."""
  replaced = {override.split("=", 1)[0] for override in extra}
  defaults = [override for override in _tiny_overrides(data_parallelism) if override.split("=", 1)[0] not in replaced]
  return pyconfig.initialize(
      ["", get_test_config_path("base.yml"), "run_name=engine_compile_test"] + defaults + list(extra)
  )


def _subtree_bytes(state, key: str, memory: int = 0) -> int:
  """Bytes per device of one top-level subtree, by the report's own per-leaf rule.

  `memory` indexes `per_device_bytes`'s `(device, host)`: device bytes by default, 1 for host.
  """
  return sum(maxtext_engine_compile.per_device_bytes(leaf)[memory] for leaf in jax.tree.leaves(state[key]))


class CompiledMemoryReportTest(absltest.TestCase):
  """The report on the engine's real kernels, checked against what XLA says it allocated."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cfg = _config(1)
    cls._engine, cls._compiled = maxtext_engine_compile.compile_engine(cfg, maxtext_utils.get_mesh_from_config(cfg))

  # CPU only: the report counts logical shard bytes, while a TPU executable's buffers also carry
  # tiled-layout padding, so the byte counts are exactly equal only on XLA:CPU.
  @pytest.mark.cpu_only
  def test_report_matches_xla_allocation(self):
    """The report's train-state bytes match XLA's, and the forward/backward kernels are not passed the optimizer.

    `update` donates the whole train state, so its alias is XLA's own count of that state.
    """
    state = self._engine.train_state_avals()
    model, optimizer = _subtree_bytes(state, "model"), _subtree_bytes(state, "optimizer")
    rows = _by_kernel(maxtext_engine_compile.memory_report(self._compiled, state))

    self.assertGreater(optimizer, 0)
    self.assertEqual(rows["update"].alias, model + optimizer)
    self.assertEqual(rows["update"].device_not_passed, 0)
    for name in ("fwd_bwd", "fwd_bwd_accum"):
      with self.subTest(kernel=name):
        # The accumulating kernel also takes the donated gradient sum; set it aside.
        own_arguments = rows[name].argument - rows[name].alias
        self.assertGreaterEqual(own_arguments, model)
        self.assertLess(own_arguments, model + optimizer, "the optimizer is among this kernel's arguments")
        self.assertEqual(rows[name].device_not_passed, optimizer)

  def test_train_state_avals_returns_copy(self):
    """The engine's cached state is re-placed in place on a recompile, so a caller must not share it."""
    cfg = _config(1)
    engine, _ = maxtext_engine_compile.compile_engine(cfg, maxtext_utils.get_mesh_from_config(cfg))

    handed_out = engine.train_state_avals()
    self.assertIsNot(handed_out, engine._read_state_pure())
    handed_out["optimizer"] = nnx.State({})

    self.assertGreater(_subtree_bytes(engine._read_state_pure(), "optimizer"), 0, "the caller's edit reached the engine")
    self.assertEqual(
        _subtree_bytes(engine.train_state_avals(), "optimizer"), _subtree_bytes(engine._read_state_pure(), "optimizer")
    )

  def test_train_state_avals_requires_compile(self):
    """Before compiling, the optimizer is not yet on the layout it runs on."""
    cfg = _config(1)
    engine = maxtext_engine_compile.AbstractMaxTextEngine(cfg, maxtext_utils.get_mesh_from_config(cfg))

    with self.assertRaisesRegex(RuntimeError, "only meaningful after compile_kernels"):
      engine.train_state_avals()

  def _run_main(self, *extra: str) -> str:
    """Runs `main` itself on this host's CPU and returns what it printed.

    Only the topology lookup, which needs libtpu, is replaced. `main` sets the PRNG implementation
    process-wide, so it is put back.
    """
    previous = jax.config.jax_default_prng_impl
    self.addCleanup(jax.config.update, "jax_default_prng_impl", previous)
    argv = (
        ["", get_test_config_path("base.yml"), "run_name=engine_compile_main_test"]
        + ["compile_topology=v6e-1", "compile_topology_num_slices=1"]
        + [override for override in _tiny_overrides(1) if not override.startswith("ici_")]
        + list(extra)
    )
    stdout = io.StringIO()
    with (
        mock.patch.dict(os.environ),
        mock.patch.object(pre_train_compile, "get_topology_mesh", maxtext_utils.get_mesh_from_config),
        contextlib.redirect_stdout(stdout),
    ):
      maxtext_engine_compile.main(argv)
    return stdout.getvalue()

  def test_main_prints_report_after_analyses(self):
    lines = self._run_main().splitlines()
    raw = [index for index, line in enumerate(lines) if line.startswith("Memory analysis: ")]
    verdict = [index for index, line in enumerate(lines) if line.startswith("TRAIN-KERNEL DEVICE PEAK: ")]
    self.assertLen(raw, len(maxtext_engine_compile.KERNEL_NAMES), "the raw analyses must still be printed")
    self.assertLen(verdict, 1)
    self.assertGreater(verdict[0], raw[-1])
    self.assertRegex(lines[verdict[0]], r"^TRAIN-KERNEL DEVICE PEAK: \d+\.\d\d GiB \((fwd_bwd|fwd_bwd_accum|update)\b")
    for name in maxtext_engine_compile.KERNEL_NAMES:
      self.assertLen([line for line in lines[raw[-1] : verdict[0]] if line.startswith(f"{name} ")], 1, name)

  def test_main_saves_executables_before_report(self):
    """When `memory_report` raises (here: no memory analysis), the executables are already saved."""
    output = os.path.join(self.create_tempdir().full_path, "engine.pickle")

    with mock.patch.object(jax.stages.Compiled, "memory_analysis", return_value=None):
      with self.assertRaisesRegex(ValueError, "has no memory analysis on this backend"):
        self._run_main(f"compiled_trainstep_file={output}")

    for name in maxtext_engine_compile.KERNEL_NAMES:
      written = maxtext_engine_compile.kernel_save_path(output, name)
      self.assertTrue(os.path.exists(written), f"{name} was not written to {written}")
      self.assertGreater(os.path.getsize(written), 0, written)


class OptimizerOffloadReportTest(absltest.TestCase):
  """`optimizer_memory_host_offload` on a real compile, where the raw analyses cannot show it."""

  def test_offload_moves_optimizer_out_of_peak(self):
    """Offload leaves the forward/backward kernels' own analyses unchanged; the report's `+state` shows it.

    Those kernels are not passed the optimizer, so their `memory_analysis()` is the same whether
    it is in device or host memory.
    """
    reports = {}
    for offload in (False, True):
      cfg = _config(1, f"optimizer_memory_host_offload={offload}")
      engine, compiled = maxtext_engine_compile.compile_engine(cfg, maxtext_utils.get_mesh_from_config(cfg))
      state = engine.train_state_avals()
      reports[offload] = (_by_kernel(maxtext_engine_compile.memory_report(compiled, state)), state)
    (off, off_state), (on, on_state) = reports[False], reports[True]

    optimizer = _subtree_bytes(off_state, "optimizer")
    self.assertGreater(optimizer, 0)
    # All of it moves, scalars included, and nothing of it is left on the device.
    self.assertEqual(_subtree_bytes(on_state, "optimizer", memory=1), optimizer)
    self.assertEqual(_subtree_bytes(on_state, "optimizer"), 0)
    for name in ("fwd_bwd", "fwd_bwd_accum"):
      with self.subTest(kernel=name):
        self.assertEqual(on[name].resident, off[name].resident, "offload changed a kernel it is not passed to")
        self.assertEqual((off[name].device_not_passed, off[name].host_not_passed), (optimizer, 0))
        self.assertEqual((on[name].device_not_passed, on[name].host_not_passed), (0, optimizer))
        self.assertEqual(off[name].device_total - on[name].device_total, optimizer)
    self.assertEqual((off["update"].host_argument, off["update"].host_output), (0, 0))
    # `update` is passed the optimizer either way, so offload can only move its bytes between the
    # device and host columns. Which one is up to the backend: XLA:CPU has a single memory space
    # and counts a pinned_host argument in `argument`.
    self.assertEqual(
        on["update"].argument + on["update"].host_argument, off["update"].argument + off["update"].host_argument
    )


def _fake_weight_sync_modules(bound: list) -> dict[str, types.ModuleType]:
  """Stands in for `tunix.experimental.weight_sync`, whose synchronizer only records what it is bound to.

  Inactive, so `prepare_weight_sync` stops after `bind` without a transfer: everything it does to
  the parameters before that is the engine's own code, unchanged.
  """
  synchronizer = types.SimpleNamespace(
      bind=bound.append,
      active=False,
      work_unit_metadata_all=lambda: [],
      release_buffers=lambda: 0,
      close=lambda: None,
  )
  module = types.ModuleType("tunix.experimental.weight_sync.weight_sync")
  module.create_weight_synchronizer = lambda **kwargs: synchronizer
  package = types.ModuleType("tunix.experimental.weight_sync")
  package.weight_sync = module
  return {package.__name__: package, module.__name__: module}


def _new_buffer_bytes(tree, existing) -> int:
  """Bytes of the distinct buffers in `tree` that none of `existing`'s arrays uses. One device only."""
  seen = {leaf.unsafe_buffer_pointer() for leaf in jax.tree.leaves(existing) if isinstance(leaf, jax.Array)}
  new = {}
  for leaf in jax.tree.leaves(tree):
    pointer = leaf.unsafe_buffer_pointer()
    if pointer not in seen:
      new[pointer] = leaf.nbytes
  return sum(new.values())


class WeightSyncStagingTest(parameterized.TestCase):
  """`weight_sync_staging` against the buffers the engine's own `prepare_weight_sync` makes and binds."""

  @parameterized.named_parameters(
      # Cast to bf16: a copy of every parameter.
      ("float32_unscanned", "float32", False, True),
      # Nothing to cast and nothing to slice: staged in place, so the estimate must be zero.
      ("bfloat16_unscanned", "bfloat16", False, False),
      # Sliced per layer: a copy of the scanned parameters only, not of the embedding or final norm.
      ("bfloat16_scanned", "bfloat16", True, True),
      ("float32_scanned", "float32", True, True),
  )
  def test_estimate_matches_prepare_weight_sync(self, weight_dtype, scan_layers, copies):
    # The path the estimate mirrors. With the converter (the config default) it returns None instead.
    cfg = _config(1, f"weight_dtype={weight_dtype}", f"scan_layers={scan_layers}", "use_weight_converter=False")
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=maxtext_utils.get_mesh_from_config(cfg))
    bound = []

    with mock.patch.dict(sys.modules, _fake_weight_sync_modules(bound)):
      engine.prepare_weight_sync(staging_transport="raiden")
    measured = _new_buffer_bytes(bound, nnx.state(engine.model))
    estimate = maxtext_engine_compile.weight_sync_staging(
        nnx.split(engine.state)[1],
        scan_layers=cfg.scan_layers,
        scan_axis=cfg.param_scan_axis,
        use_weight_converter=cfg.use_weight_converter,
    )

    self.assertLen(bound, 1, "prepare_weight_sync never bound a staged tree")
    self.assertEqual(estimate.held_copy, measured)
    # Tells the in-place case apart from a conversion that stopped making copies.
    self.assertEqual(measured > 0, copies)
    params = sum(leaf.nbytes for leaf in jax.tree.leaves(nnx.state(engine.model, nnx.Param)))
    if scan_layers and weight_dtype == "bfloat16":
      self.assertLess(measured, params, "the unscanned leaves are staged in place, so less than all of them")


@pytest.mark.cpu_only
def test_zero1_report_on_four_cpu_devices():
  """Runs `Zero1MemoryReportTest` in a child process with four CPU devices; see the module docstring."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_ZERO1_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  # An exit status of 0 is also what a run that skipped everything produces.
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


class Zero1MemoryReportTest(absltest.TestCase):
  """Zero-1 shards the optimizer at compile time, so the report must read the state after it."""

  __test__ = False  # collected only via the subprocess entry point above.

  def test_report_counts_sharded_optimizer(self):
    cfg = _config(_ZERO1_DEVICES, "shard_optimizer_over_data=True")
    engine = maxtext_engine_compile.AbstractMaxTextEngine(cfg, maxtext_utils.get_mesh_from_config(cfg))
    # Counted before compiling: compiling re-places this tree in place, so counting it afterwards
    # would read the sharded layout.
    unsharded = _subtree_bytes(engine._read_state_pure(), "optimizer")
    compiled = engine.compile_kernels(maxtext_engine_compile.get_shaped_micro_batch(cfg))
    state = engine.train_state_avals()
    rows = _by_kernel(maxtext_engine_compile.memory_report(compiled, state))

    self.assertIsNotNone(engine._zero1_params_shardings, "Zero-1 never engaged")
    model = _subtree_bytes(state, "model")
    optimizer = rows["fwd_bwd"].device_not_passed
    self.assertEqual(rows["update"].alias, model + optimizer)
    # The pre-compile, unsharded optimizer is ~4x larger and does not match XLA's count.
    self.assertNotEqual(rows["update"].alias, model + unsharded)
    self.assertGreater(unsharded, 3 * optimizer)


if __name__ == "__main__":
  if jax.device_count() < _ZERO1_DEVICES:
    raise SystemExit(
        f"needs {_ZERO1_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_ZERO1_DEVICES}"
    )
  _result = unittest.TextTestRunner(verbosity=2).run(
      unittest.defaultTestLoader.loadTestsFromTestCase(Zero1MemoryReportTest)
  )
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")
