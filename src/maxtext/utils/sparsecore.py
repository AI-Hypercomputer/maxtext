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

"""TPU SparseCore capability detection and compute offloading.

Recent TPUs pair each TensorCore with one or more SparseCores. XLA:TPU can run
some ops there instead of on the TensorCore when they carry the
``tpu_sparsecore`` compute type, which frees TensorCore cycles for the matmuls
and lets the offloaded work overlap with them.

This module owns two things:

  * :func:`sparse_core_info` / :func:`has_sparse_core` -- whether the *target*
    chip has a SparseCore at all. The answer is taken from JAX's own chip table
    (``pltpu.get_tpu_info_for_chip``) rather than pattern-matching device
    strings, so it stays correct as new chips are added.
  * :func:`offload` -- the single context manager used to tag ops.

Offloading is opt-in per target; see ``moe_sparse_core_offload_targets``.

The annotation never changes results: it picks which core runs an op, not what
it computes. It is not, however, purely advisory for *collectives*. When the
matching ``--xla_tpu_enable_sparse_core_collective_offload_*`` flag is on, XLA's
offload pass CHECK-fails on an annotated collective the target chip cannot
lower ("instruction has compute type annotation sparseoffload but the operation
is currently not supported on SC") rather than falling back to the TensorCore.
Measured on v5p: annotated all-reduce and reduce-scatter are accepted, and
all-gather is rejected at every rank. Those flags are off by default outside
Ironwood, which is why the default flag set compiles either way. See the
``moe_sparse_core_offload_targets`` comment in ``base.yml``.
"""

from __future__ import annotations

import contextlib

import jax
from jax.experimental import compute_on

from maxtext.utils import accelerator_to_spec_map

# XLA compute type that moves an op onto the SparseCore.
SPARSE_CORE_COMPUTE_TYPE = "tpu_sparsecore"

# Offload targets accepted by ``moe_sparse_core_offload_targets``.
FSDP_ALL_GATHER = "fsdp_all_gather"
EP_COLLECTIVES = "ep_collectives"
RAGGED_SORT = "ragged_sort"
OFFLOAD_TARGETS = (FSDP_ALL_GATHER, EP_COLLECTIVES, RAGGED_SORT)
_ALL_TARGETS = "all"

# ``hardware`` values that can never have a SparseCore.
_NON_TPU_HARDWARE = ("cpu", "gpu", "gpu_multiprocess")


def _chip_version(accelerator_name: str):
  """Maps a user-facing accelerator name (e.g. ``tpu7x-256``) to a ChipVersion.

  ``accelerator_to_spec_map`` keys are ``<family>-<num_chips>`` and the family
  matches ``pltpu.ChipVersion``'s value modulo a ``tpu`` prefix (``tpu7x`` vs
  ``7x``), so no per-chip table is needed here.
  """
  import jax.experimental.pallas.tpu as pltpu  # pylint: disable=import-outside-toplevel

  family = accelerator_name.split("-", maxsplit=1)[0].lower()
  for candidate in (family, family.removeprefix("tpu")):
    try:
      return pltpu.ChipVersion(candidate)
    except ValueError:
      continue
  return None


def sparse_core_info(compile_topology: str = "", hardware: str = ""):
  """Returns the target chip's ``SparseCoreInfo``, or ``None`` if it has none.

  Args:
    compile_topology: AOT target topology (e.g. ``tpu7x-256``). When set, the
      answer describes that target rather than the local devices.
    hardware: the ``hardware`` config value, used to rule out CPU/GPU runs.

  Returns:
    JAX's ``SparseCoreInfo`` for the target chip, or ``None`` when the target
    has no SparseCore or cannot be determined.
  """
  if hardware in _NON_TPU_HARDWARE:
    return None

  import jax.experimental.pallas.tpu as pltpu  # pylint: disable=import-outside-toplevel

  if compile_topology:
    try:
      spec = accelerator_to_spec_map.get_system_characteristics(compile_topology)
    except ValueError:
      return None
    if spec.platform != "tpu":
      return None
    chip_version = _chip_version(compile_topology)
    if chip_version is None:
      return None
    # SparseCore presence does not depend on the Megacore split, so 1 core per
    # logical device is a valid probe for every chip version.
    return pltpu.get_tpu_info_for_chip(chip_version, 1).sparse_core

  if not is_tpu_runtime():
    return None
  try:
    return pltpu.get_tpu_info().sparse_core
  except (RuntimeError, ValueError, AttributeError, TypeError, IndexError):
    return None


def has_sparse_core(compile_topology: str = "", hardware: str = "") -> bool:
  """Whether the target chip has a SparseCore. See :func:`sparse_core_info`."""
  return sparse_core_info(compile_topology, hardware) is not None


def is_tpu_runtime() -> bool:
  """Whether the local devices are TPUs. Mirrors the kernel fallback guards."""
  try:
    return jax.devices()[0].platform == "tpu"
  except (RuntimeError, IndexError):
    return False


def parse_offload_targets(targets: str) -> frozenset[str]:
  """Parses ``moe_sparse_core_offload_targets`` into a set of target names.

  Args:
    targets: empty, ``"all"``, or a comma-separated subset of
      :data:`OFFLOAD_TARGETS`.

  Returns:
    The requested targets.

  Raises:
    ValueError: if an unrecognized target is requested.
  """
  if not targets:
    return frozenset()
  requested = [t.strip() for t in targets.split(",") if t.strip()]
  if _ALL_TARGETS in requested:
    return frozenset(OFFLOAD_TARGETS)
  unknown = sorted(set(requested) - set(OFFLOAD_TARGETS))
  if unknown:
    raise ValueError(
        f"Unknown SparseCore offload target(s) {unknown} in "
        f"moe_sparse_core_offload_targets={targets!r}. "
        f"Supported targets: {list(OFFLOAD_TARGETS)} or '{_ALL_TARGETS}'."
    )
  return frozenset(requested)


@contextlib.contextmanager
def offload(enabled: bool):
  """Runs ops traced inside this block on the SparseCore when ``enabled``.

  The annotation is a hint: XLA offloads the ops it can lower to SparseCore and
  leaves the rest on the TensorCore, so this is numerically transparent. It is a
  no-op off TPU, which keeps CPU/GPU unit tests and simulations working with a
  TPU config.

  Args:
    enabled: whether this offload target is turned on.

  Yields:
    None.
  """
  if not enabled or not is_tpu_runtime():
    yield
    return
  with compute_on.compute_on(SPARSE_CORE_COMPUTE_TYPE):
    yield
