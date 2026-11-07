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

"""Per-device memory analysis and reporting for `MaxTextTrainingEngine`'s compiled kernels."""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import jax
from maxtext.training_engine import maxtext_engine

_GIB = 2**30
PEAK_LABEL = "TRAIN-KERNEL DEVICE PEAK"
_HOST_MEMORY_KINDS = frozenset({"pinned_host", "unpinned_host"})
_MODEL_KEY = maxtext_engine._MODEL_STATE_KEY  # pylint: disable=protected-access
_OPTIMIZER_KEY = maxtext_engine._OPTIMIZER_STATE_KEY  # pylint: disable=protected-access

# Top-level train-state subtrees that remain resident in HBM while each kernel runs without being
# passed as arguments.
STATE_NOT_PASSED = {
    "fwd_bwd": (_OPTIMIZER_KEY,),
    "fwd_bwd_accum": (_OPTIMIZER_KEY,),
    "update": (),
}


@dataclasses.dataclass(frozen=True)
class KernelMemory:
  """Per-device memory breakdown (in bytes) for a single compiled kernel.

  Attributes:
    kernel: Kernel name (one of `KERNEL_NAMES`).
    argument: Device bytes of kernel arguments.
    output: Device bytes of kernel outputs.
    alias: Device bytes shared between donated arguments and outputs.
    temp: Temporary device scratch bytes allocated while the kernel runs.
    host_argument: Host bytes of kernel arguments (e.g. `pinned_host` leaves).
    host_output: Host bytes of kernel outputs.
    host_alias: Host bytes shared between donated arguments and outputs.
    host_temp: Temporary host scratch bytes allocated while the kernel runs.
    not_passed: Names of train-state subtrees resident in HBM while this kernel
      runs but not passed as arguments.
    device_not_passed: Device bytes of `not_passed`.
    host_not_passed: Host bytes of `not_passed`.
  """

  kernel: str
  argument: int
  output: int
  alias: int
  temp: int
  host_argument: int
  host_output: int
  host_alias: int
  host_temp: int
  not_passed: tuple[str, ...]
  device_not_passed: int
  host_not_passed: int

  @property
  def resident(self) -> int:
    """Device bytes held by the kernel itself: `argument + output - alias + temp`."""
    return self.argument + self.output - self.alias + self.temp

  @property
  def device_total(self) -> int:
    """Total device bytes in use while this kernel runs (`resident + device_not_passed`)."""
    return self.resident + self.device_not_passed


def per_device_bytes(leaf: Any) -> tuple[int, int]:
  """Returns `(device_bytes, host_bytes)` held per device for `leaf`."""
  if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
    return 0, 0
  sharding = getattr(leaf, "sharding", None)
  shard_shape = sharding.shard_shape(leaf.shape) if sharding is not None else leaf.shape
  nbytes = math.prod(shard_shape) * leaf.dtype.itemsize
  if getattr(sharding, "memory_kind", None) in _HOST_MEMORY_KINDS:
    return 0, nbytes
  return nbytes, 0


def _tree_bytes_per_device(tree: Any) -> tuple[int, int]:
  """Returns total `(device_bytes, host_bytes)` per device across all leaves in `tree`."""
  device, host = 0, 0
  for leaf in jax.tree.leaves(tree):
    leaf_device, leaf_host = per_device_bytes(leaf)
    device += leaf_device
    host += leaf_host
  return device, host


def memory_report(compiled: dict[str, Any], state: Any) -> list[KernelMemory]:
  """Computes per-device `KernelMemory` for each compiled kernel, including unpassed resident state.

  Args:
    compiled: Mapping from kernel name to `jax.stages.Compiled` executable.
    state: Pure train state dict (`model` and `optimizer` avals on their
      compiled shardings).

  Returns:
    A list of `KernelMemory` rows in the iteration order of `compiled`.
  """
  # Every subtree must be passed to a kernel or classified for it, or it would be resident while
  # that kernel runs yet counted in neither its arguments nor `+state`.
  known = {_MODEL_KEY}.union(*STATE_NOT_PASSED.values())
  unknown = sorted(str(key) for key in state if key not in known)
  if unknown:
    raise ValueError(
        f"The train state has top-level subtrees {unknown} that"
        " STATE_NOT_PASSED does not classify. Add them, based on the kernel"
        " signatures in `_compile_for_batch`."
    )
  rows = []
  for name, executable in compiled.items():
    if name not in STATE_NOT_PASSED:
      raise ValueError(
          f"STATE_NOT_PASSED has no entry for kernel {name!r}. Add one, based"
          " on its signature in `_compile_for_batch`."
      )
    stats = executable.memory_analysis()
    if stats is None:
      raise ValueError(f"{name} has no memory analysis on this backend, so there is nothing" " to report.")
    not_passed = STATE_NOT_PASSED[name]
    missing = [key for key in not_passed if key not in state]
    if missing:
      raise ValueError(f"The train state has no {missing} subtree, which {name} is expected" " to run alongside.")
    device_not_passed, host_not_passed = _tree_bytes_per_device([state[key] for key in not_passed])
    rows.append(
        KernelMemory(
            kernel=name,
            argument=stats.argument_size_in_bytes,
            output=stats.output_size_in_bytes,
            alias=stats.alias_size_in_bytes,
            temp=stats.temp_size_in_bytes,
            host_argument=stats.host_argument_size_in_bytes,
            host_output=stats.host_output_size_in_bytes,
            host_alias=stats.host_alias_size_in_bytes,
            host_temp=stats.host_temp_size_in_bytes,
            not_passed=not_passed,
            device_not_passed=device_not_passed,
            host_not_passed=host_not_passed,
        )
    )
  return rows


def device_peak(rows: Sequence[KernelMemory]) -> KernelMemory:
  """Returns the `KernelMemory` row with the largest `device_total`."""
  if not rows:
    raise ValueError("There are no kernels to take a peak over.")
  return max(rows, key=lambda row: row.device_total)


# The table's columns between the kernel name and the subtrees not passed to it: (header, width,
# bytes). The header and the cell come from one entry, so they cannot drift apart.
_TABLE_COLUMNS = (
    ("arg", 9, lambda row: row.argument),
    ("out", 9, lambda row: row.output),
    ("alias", 9, lambda row: row.alias),
    ("temp", 9, lambda row: row.temp),
    ("resident", 10, lambda row: row.resident),
    ("+state", 9, lambda row: row.device_not_passed),
    ("total", 9, lambda row: row.device_total),
    ("host arg", 12, lambda row: row.host_argument),
    ("host out", 9, lambda row: row.host_output),
    ("host alias", 11, lambda row: row.host_alias),
    ("host temp", 10, lambda row: row.host_temp),
    ("host state", 11, lambda row: row.host_not_passed),
)


def _table_line(first: str, cells: Any, last: str) -> str:
  """Returns one formatted table row."""
  return f"{first:<15}" + "".join(f"{text:>{width}}" for _, width, text in cells) + f"   {last}"


def format_memory_report(rows: Sequence[KernelMemory]) -> str:
  """Formats the per-device memory table and device peak."""

  def gib(num_bytes: int) -> str:
    return f"{num_bytes / _GIB:.2f}"

  lines = [
      ("Engine memory per device, GiB (2^30). resident = arg + out - alias +" " temp: a donated buffer is in"),
      ("both arg and out, and alias is what they share. +state = what is in" " device memory while the kernel"),
      ("runs without being one of its arguments, so its memory analysis" " cannot see it: the optimizer state"),
      ("during fwd_bwd and fwd_bwd_accum. The host columns are the same" " quantities in host memory"),
      ("(pinned_host), which XLA:CPU counts as device. The peak below is" " these three kernels' only. NOT"),
      ("included: weight-sync staging (a copy of the parameters in the" " rollout's layout, held until"),
      ("release_weight_sync), fwd_only scoring, the eval kernel, generated" " code, and anything the caller"),
      ("holds. It includes fwd_bwd_accum even when" " gradient_accumulation_steps=1 (where only fwd_bwd runs)."),
      _table_line(
          "kernel",
          ((name, width, name) for name, width, _ in _TABLE_COLUMNS),
          "state not passed",
      ),
  ]
  for row in rows:
    cells = ((name, width, gib(cell(row))) for name, width, cell in _TABLE_COLUMNS)
    lines.append(_table_line(row.kernel, cells, ", ".join(row.not_passed) or "-"))

  peak = device_peak(rows)
  if peak.not_passed:
    held = " + ".join(f"{key} state" for key in peak.not_passed)
    because = f"{peak.kernel} {gib(peak.resident)} + {held}" f" {gib(peak.device_not_passed)} not passed to it"
    if peak.host_not_passed:
      because += f"; {gib(peak.host_not_passed)} GiB more of it is on host"
  else:
    because = f"{peak.kernel}, which is passed all the train state it runs with"

  lines.append(f"{PEAK_LABEL}: {gib(peak.device_total)} GiB ({because})")
  return "\n".join(lines)
