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

This module owns three things:

  * :func:`sparse_core_info` / :func:`has_sparse_core` -- whether the *target*
    chip has a SparseCore at all. The answer is taken from JAX's own chip table
    (``pltpu.get_tpu_info_for_chip``) rather than pattern-matching device
    strings, so it stays correct as new chips are added.
  * :func:`supports_collective_offload` / :func:`supported_offload_targets` --
    which collectives that SparseCore can actually run.
  * :func:`offload` -- the single context manager used to tag ops.

Offloading is opt-in per target; see ``moe_sparse_core_offload_targets``, which
the config validator refuses to enable unless the *target* chip has a
SparseCore. That check is what lets :func:`offload` annotate unconditionally,
so an ahead-of-time compile on a TPU-less host produces the same HLO the real
run will.

The annotation never changes results: it picks which core runs an op, not what
it computes. For ordinary compute it is also purely advisory -- XLA offloads the
ops it can lower and silently leaves the rest on the TensorCore.

*Collectives* are the exception, and the reason
:func:`supports_collective_offload` exists. XLA's SparseCore collective-offload
pass treats the annotation as "force this one", so when it selects an annotated
collective the chip cannot lower it CHECK-fails -- aborting the compiler, not
falling back::

    F sparse_core_collective_offload.cc:586] Candidate rejected: instruction has
    compute type annotation sparseoffload but the operation is currently not
    supported on SC. %all-gather-start = ...

Measured on v5p with libtpu 0.0.46, where the pass runs under its default flags:
an annotated ragged all-to-all and reduce-scatter are offloaded, an annotated
all-reduce is silently ignored, and an annotated all-gather aborts the compile
at every rank. Per ``benchmarks/xla_flags_library.py`` and the original
SparseCore MoE work, all-gather offload arrives with Ironwood, so
:data:`_MIN_GENERATION` only lets an all-gather carry the annotation from chip
generation 7 on -- and, via :data:`_TRANSPOSE_OF`, neither does a reduce-scatter,
whose backward pass *is* an all-gather.
"""

from __future__ import annotations

import contextlib
import functools
import inspect

import jax
from jax.experimental import compute_on

from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import max_logging

# XLA compute type that moves an op onto the SparseCore.
SPARSE_CORE_COMPUTE_TYPE = "tpu_sparsecore"

# Offload targets accepted by ``moe_sparse_core_offload_targets``.
FSDP_ALL_GATHER = "fsdp_all_gather"
EP_COLLECTIVES = "ep_collectives"
RAGGED_SORT = "ragged_sort"
OFFLOAD_TARGETS = (FSDP_ALL_GATHER, EP_COLLECTIVES, RAGGED_SORT)
_ALL_TARGETS = "all"

# Collective kinds that get annotated, for :func:`supports_collective_offload`.
ALL_GATHER = "all_gather"
RAGGED_ALL_TO_ALL = "ragged_all_to_all"
REDUCE_SCATTER = "reduce_scatter"

# Lowest chip generation whose SparseCore can run each collective. Anything
# absent is assumed offloadable on every chip that has a SparseCore, which is
# the safe assumption because an annotation XLA does not act on costs nothing --
# see the module docstring for what happens when it does act on one it cannot
# lower.
_MIN_GENERATION = {ALL_GATHER: 7}

# What each collective becomes under transposition. The compute type is
# snapshotted onto the jaxpr equation, and AD carries it onto the transposed
# equation, so annotating one end annotates the other: an annotated
# reduce-scatter puts an annotated all-gather in the backward pass, which is
# just as fatal as annotating the all-gather directly. Both ends therefore have
# to be offloadable before either is annotated.
_TRANSPOSE_OF = {
    ALL_GATHER: REDUCE_SCATTER,
    REDUCE_SCATTER: ALL_GATHER,
    RAGGED_ALL_TO_ALL: RAGGED_ALL_TO_ALL,
}

# The one collective a target exists to annotate. Requesting such a target on a
# chip that cannot offload that collective is pointless, so the target is
# dropped whole rather than left to restructure the graph for nothing. Targets
# that annotate a mix of collectives are absent here and gate per call site.
_TARGET_REQUIRED_COLLECTIVE = {FSDP_ALL_GATHER: ALL_GATHER}

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


@functools.cache
def _tpu_info(compile_topology: str = "", hardware: str = ""):
  """Returns JAX's ``TpuInfo`` for the target chip, or ``None`` if it is not a TPU.

  Args:
    compile_topology: AOT target topology (e.g. ``tpu7x-256``). When set, the
      answer describes that target rather than the local devices.
    hardware: the ``hardware`` config value, used to rule out CPU/GPU runs.

  Returns:
    The target chip's ``TpuInfo``, or ``None`` when the target is not a TPU or
    cannot be determined.
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
    # Neither SparseCore presence nor the chip generation depends on the
    # Megacore split, so 1 core per logical device is a valid probe here.
    return pltpu.get_tpu_info_for_chip(chip_version, 1)

  if not is_tpu_runtime():
    return None
  try:
    return pltpu.get_tpu_info()
  except (RuntimeError, ValueError, AttributeError, TypeError, IndexError):
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
  info = _tpu_info(compile_topology, hardware)
  return None if info is None else info.sparse_core


def has_sparse_core(compile_topology: str = "", hardware: str = "") -> bool:
  """Whether the target chip has a SparseCore. See :func:`sparse_core_info`."""
  return sparse_core_info(compile_topology, hardware) is not None


def supports_collective_offload(collective: str, compile_topology: str = "", hardware: str = "") -> bool:
  """Whether the target's SparseCore can run `collective` when it is annotated.

  Getting this wrong in the permissive direction aborts the compiler rather than
  costing performance, so the answer is conservative: a collective is offloadable
  only from the chip generation both it and its transpose are known to work on.
  See the module docstring.

  Args:
    collective: one of :data:`ALL_GATHER`, :data:`RAGGED_ALL_TO_ALL`,
      :data:`REDUCE_SCATTER`.
    compile_topology: AOT target topology, as in :func:`sparse_core_info`.
    hardware: the ``hardware`` config value.

  Returns:
    Whether it is safe to annotate that collective for the target chip.
  """
  info = _tpu_info(compile_topology, hardware)
  if info is None or info.sparse_core is None:
    return False
  both_ends = {collective, _TRANSPOSE_OF.get(collective, collective)}
  return all(info.generation >= _MIN_GENERATION.get(kind, 0) for kind in both_ends)


def is_tpu_runtime() -> bool:
  """Whether the local devices are TPUs. Mirrors the kernel fallback guards."""
  try:
    return jax.devices()[0].platform == "tpu"
  except (RuntimeError, IndexError):
    return False


@functools.cache
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


@functools.cache
def _warn_target_unsupported(target: str, collective: str, chip: str) -> None:
  """Logs once that a requested offload target cannot run on the target chip."""
  max_logging.log(
      f"moe_sparse_core_offload_targets requests {target!r}, but the SparseCore on {chip} cannot run an annotated "
      f"{collective} (XLA would abort the compile rather than fall back). Ignoring that target and keeping those ops "
      "on the TensorCore; results are unaffected. Other requested targets are unaffected."
  )


@functools.cache
def supported_offload_targets(targets: str, compile_topology: str = "", hardware: str = "") -> frozenset[str]:
  """Parses ``moe_sparse_core_offload_targets`` and drops what the chip cannot serve.

  A target whose whole purpose is annotating one collective (see
  :data:`_TARGET_REQUIRED_COLLECTIVE`) is dropped on a chip that cannot offload
  it, with a warning, rather than failing the run: the offload is an
  optimization, and the same config should stay usable across chip generations.
  Targets that annotate a mix of collectives survive and gate per call site.

  Args:
    targets: the ``moe_sparse_core_offload_targets`` config value.
    compile_topology: AOT target topology, as in :func:`sparse_core_info`.
    hardware: the ``hardware`` config value.

  Returns:
    The requested targets this chip can actually serve.

  Raises:
    ValueError: if an unrecognized target is requested.
  """
  requested = parse_offload_targets(targets)
  if not requested:
    return requested
  supported = set(requested)
  for target, collective in _TARGET_REQUIRED_COLLECTIVE.items():
    if target in supported and not supports_collective_offload(collective, compile_topology, hardware):
      info = _tpu_info(compile_topology, hardware)
      _warn_target_unsupported(target, collective, compile_topology or str(info.chip_version if info else "this chip"))
      supported.discard(target)
  return frozenset(supported)


def _resolve_compute_type_context():
  """Finds the JAX API that stamps a compute type onto every op traced in a block.

  JAX has moved this around. Through 0.11.0,
  ``jax.experimental.compute_on.compute_on`` was a context manager taking the
  compute type. In 0.11.1 that name became a function transform,
  ``compute_on(f, *, compute_type, out_memory_spaces)``, which traces ``f`` into
  its own computation and puts the attribute on the call instead.

  The transform cannot express what the MoE needs. It refuses to nest at all --
  ``_compute_on_lowering`` raises "Nesting `compute_on` with different compute
  types is not allowed" for *any* nesting, and
  ``moe_sparse_core_offload_targets=all`` nests the ragged-sort blocks inside the
  expert-parallel ones. It would also turn every annotated block into a separate
  computation, changing what XLA is free to fuse and forcing the intermediates
  across a call boundary, which is not what per-op offloading is supposed to do.

  The per-op mechanism itself is unchanged in every version: the compute type
  lives in a config context that ``jax._src.core.JaxprEqnContext`` snapshots onto
  each equation, and ``mlir.wrap_compute_type_in_place`` turns that into the
  ``_xla_compute_type`` frontend attribute. So prefer the public context manager
  while it still is one, and otherwise fall back to the private helper that both
  public spellings are built on.

  Returns:
    A callable mapping a compute type to a context manager, or ``None`` if this
    JAX exposes neither spelling.
  """
  public = getattr(compute_on, "compute_on", None)
  if public is not None:
    try:
      parameters = list(inspect.signature(public).parameters)
    except (TypeError, ValueError):
      parameters = []
    if parameters == ["compute_type"]:
      return public
  try:
    # pylint: disable-next=import-outside-toplevel
    from jax._src.compute_on import extend_compute_type
  except ImportError:
    return None
  return extend_compute_type


_COMPUTE_TYPE_CONTEXT = _resolve_compute_type_context()


@functools.cache
def _warn_offload_unavailable() -> None:
  """Logs once that this JAX offers no way to annotate a block of ops."""
  max_logging.log(
      "moe_sparse_core_offload_targets is set, but this JAX exposes no "
      "compute-type context manager (looked for a context-manager "
      "jax.experimental.compute_on.compute_on and for "
      "jax._src.compute_on.extend_compute_type). Running everything on the "
      "TensorCore instead; results are unaffected."
  )


@contextlib.contextmanager
def offload(enabled: bool, collective: str | None = None, compile_topology: str = "", hardware: str = ""):
  """Runs ops traced inside this block on the SparseCore when ``enabled``.

  The annotation picks which core runs an op, not what it computes, so this is
  numerically transparent. Non-TPU backends ignore the frontend attribute, and
  the config validator already refuses to enable a target whose hardware has no
  SparseCore, so this deliberately does *not* re-check the local devices:
  ahead-of-time compilation for a SparseCore topology usually runs on a host
  that has no TPU at all, and its HLO has to match what the real run compiles.

  Args:
    enabled: whether this offload target is turned on.
    collective: the collective kind traced inside the block, when it contains
      one. The block is left unannotated if the target chip's SparseCore cannot
      run it, because XLA aborts the compile on such an annotation instead of
      falling back. Blocks of ordinary compute pass ``None``.
    compile_topology: AOT target topology, as in :func:`sparse_core_info`.
    hardware: the ``hardware`` config value.

  Yields:
    None.
  """
  if not enabled:
    yield
    return
  if collective is not None and not supports_collective_offload(collective, compile_topology, hardware):
    yield
    return
  if _COMPUTE_TYPE_CONTEXT is None:
    _warn_offload_unavailable()
    yield
    return
  with _COMPUTE_TYPE_CONTEXT(SPARSE_CORE_COMPUTE_TYPE):
    yield
