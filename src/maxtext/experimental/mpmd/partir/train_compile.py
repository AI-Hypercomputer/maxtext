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
r"""Lowers the MPMD training step without TPUs, and compiles it on Pathways.

Lets a pipeline configuration be checked before spending an allocation.
Because the program is assembled by :func:`runner.build` -- the same function
``train.py`` uses -- a successful lowering here is evidence about the program
that will actually run, not about a similar-looking one.

OSS has no AOT cross-compilation against a TPU topology on a CPU host and no
serialization of the compiled program, and Shardy MPMD's IFRT IR program
compiles only through a Pathways IFRT client (plain PJRT fails with
"PjRtCompiler requires an HloProgram").
This binary therefore:

  * lowers (StableHLO -> MPMD -> IFRT IR) on fake CPU devices (default), on an
    OSS AOT topology's devices (``--compile_topology=v6e:4x4``), or on the
    attached backend's devices (``--compile_topology=local``);
  * pins the transfers as ``train.py`` does and logs the program's shape
    (fragments and transfers per mesh), optionally dumping the IR;
  * compiles and logs the memory / cost report when the backend is Pathways
    (``JAX_PLATFORMS=proxy``, e.g. a TPU VM's Pathways in docker,
    ``local_pathways.sh``).

It also checks what only the TPUs would otherwise reject: that each microbatch
splits over a stage's batch-sharding devices.

Example (8 fake CPU devices by default, ``--num_cpu_devices``)::

    JAX_PLATFORMS=cpu python3 -m maxtext.experimental.mpmd.partir.train_compile \
      --pp=4 --fsdp=2 --num_microbatches=4 --per_device_batch_size=4 \
      --pipeline_schedule=1F1B --model_name=deepseek3-tiny --ir_dump_path=/tmp/mpmd_pp4
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import collections
import dataclasses
import os
import re
from typing import Any

from absl import app
from absl import flags
import jax
import pathwaysutils

from maxtext.utils import max_logging

from maxtext.experimental.mpmd.partir import hparams as hparams_lib
from maxtext.experimental.mpmd.partir import runner as runner_lib
from maxtext.experimental.mpmd.partir import train_step as train_step_lib


def define_flags(flag_values: flags.FlagValues = flags.FLAGS) -> None:
  """Registers the lowering check's flags, the hyperparameters' included, on `flag_values`."""
  hparams_lib.define_flags(flag_values)
  fv = {"flag_values": flag_values}
  flags.DEFINE_string(
      "compile_topology",
      "cpu",
      "Devices to lower against: 'cpu' (fake CPU devices, the default), 'local' (the attached backend, e.g."
      " Pathways), or an OSS AOT topology such as 'v6e:4x4' or 'v5p:2x2x2' (needs libtpu, which only one process per"
      " TPU VM can load).",
      **fv,
  )
  flags.DEFINE_integer(
      "num_cpu_devices",
      8,
      "Fake CPU devices for --compile_topology=cpu, unless XLA_FLAGS sets --xla_force_host_platform_device_count.",
      **fv,
  )
  flags.DEFINE_integer("compile_topology_num_slices", 1, "Number of slices in the topology.", **fv)
  flags.DEFINE_string("chip_config", "default", "chip_config_name for an AOT topology.", **fv)
  flags.DEFINE_string(
      "ir_dump_path",
      "",
      "If set, write <path>.{stablehlo,mpmd,ifrt}.mlir (the IFRT IR after transfer pinning).",
      **fv,
  )
  flags.DEFINE_enum(
      "skip_compile",
      "auto",
      ["auto", "true", "false"],
      "auto: compile only on a Pathways backend (nothing else can compile an IFRT IR program). false: always try,"
      " and fail off Pathways.",
      **fv,
  )


_CALL_RE = re.compile(r"ifrt\.Call @([A-Za-z0-9_.$]+)")
_MESH_RE = re.compile(r"ifrt\.mesh_name\s*=\s*\"([^\"]+)\"")


@dataclasses.dataclass(frozen=True)
class CompilationReport:
  """Memory and cost analysis of a compiled whole-program MPMD executable.

  Attributes:
    peak_memory_bytes: Peak HBM bytes required on the most-loaded device across
      the entire MPMD program schedule.
    argument_size_bytes: Whole-program input buffer footprint in bytes.
    output_size_bytes: Whole-program output buffer footprint in bytes.
    device_to_peak_bytes_used: Per-device peak HBM usage map from the IFRT IR
      program memory stats.
    fragment_memory_stats: Mapping from outlined stage fragment name to its XLA
      ``CompiledMemoryStats`` (temp, argument, output, alias, generated code
      bytes).
    fragment_cost_analysis: Mapping from outlined stage fragment name to its XLA
      cost analysis dictionary (FLOPs, bytes accessed, etc.).
    unavailable: Statistics the executable could not provide, with the first
      line of the error each one raised.
  """

  peak_memory_bytes: int
  argument_size_bytes: int
  output_size_bytes: int
  device_to_peak_bytes_used: dict[int, int]
  fragment_memory_stats: dict[str, Any]
  fragment_cost_analysis: dict[str, dict[str, Any]]
  unavailable: dict[str, str] = dataclasses.field(default_factory=dict)

  @property
  def peak_memory_gib(self) -> float:
    """Returns the peak per-device HBM usage in GiB."""
    return self.peak_memory_bytes / (1024.0**3)

  def format_summary(self) -> str:
    """Returns a human-readable summary of whole-program and fragment stats."""
    lines = ["MPMD Whole-Program Compilation Report:"]
    if "ifrt_ir_program_memory_stats" not in self.unavailable:
      lines += [
          (f"  peak HBM per device   : {self.peak_memory_gib:.3f} GiB" f" ({self.peak_memory_bytes:,} B)"),
          ("  program inputs        :" f" {self.argument_size_bytes / (1024**2):.2f} MiB"),
          ("  program outputs       :" f" {self.output_size_bytes / (1024**2):.2f} MiB"),
      ]
    lines.append("  outlined XLA fragments:" f" {len(self.fragment_memory_stats or self.fragment_cost_analysis)}")
    lines += [f"  unavailable           : {name} ({error})" for name, error in self.unavailable.items()]
    names = sorted(set(self.fragment_memory_stats) | set(self.fragment_cost_analysis))
    for name in names:
      mem = self.fragment_memory_stats.get(name)
      temp_mib = getattr(mem, "temp_size_in_bytes", 0) / (1024.0**2)
      arg_mib = getattr(mem, "argument_size_in_bytes", 0) / (1024.0**2)
      out_mib = getattr(mem, "output_size_in_bytes", 0) / (1024.0**2)
      alias_mib = getattr(mem, "alias_size_in_bytes", 0) / (1024.0**2)
      cost = self.fragment_cost_analysis.get(name) or {}
      flops = float(cost.get("flops", 0.0)) if isinstance(cost, Mapping) else 0.0
      lines.append(
          f"    {name:32s}: temp={temp_mib:8.2f} MiB, args={arg_mib:8.2f} MiB,"
          f" out={out_mib:7.2f} MiB, alias={alias_mib:8.2f} MiB,"
          f" GFLOPs={flops / 1e9:8.2f}"
      )
    return "\n".join(lines)


def build_compilation_report(compiled: Any) -> CompilationReport:
  """Extracts a :class:`CompilationReport` from a compiled ``MpmdCompiled``.

  Each statistic is read on its own: through the IFRT proxy (any GKE Pathways
  run) the program-level memory stats raise ``UNIMPLEMENTED:
  GetIfrtIrProgramMemoryStats only supported on MpmdLoadedExecutable``, and
  that must not hide the per-fragment analyses. What could not be read is
  listed in :attr:`CompilationReport.unavailable`.
  """
  unavailable: dict[str, str] = {}

  def _read(name: str, fn: Any, default: Any) -> Any:
    try:
      return fn()
    except Exception as error:  # pylint: disable=broad-exception-caught
      unavailable[name] = (str(error).splitlines() or [repr(error)])[0][:200]
      return default

  prog_stats = _read(
      "ifrt_ir_program_memory_stats",
      lambda: compiled.ifrt_ir_program_memory_stats,
      None,
  )
  frag_mem = dict(_read("memory_analysis", compiled.memory_analysis, None) or {})
  raw_cost = _read("cost_analysis", compiled.cost_analysis, None) or {}
  frag_cost: dict[str, dict[str, Any]] = {
      str(name): dict(stats) if isinstance(stats, Mapping) else {} for name, stats in raw_cost.items()
  }
  device_peaks = dict(getattr(prog_stats, "device_to_peak_bytes_used", {}) or {})
  return CompilationReport(
      peak_memory_bytes=max(device_peaks.values(), default=0),
      argument_size_bytes=int(getattr(prog_stats, "argument_size_in_bytes", 0)),
      output_size_bytes=int(getattr(prog_stats, "output_size_in_bytes", 0)),
      device_to_peak_bytes_used=device_peaks,
      fragment_memory_stats=frag_mem,
      fragment_cost_analysis=frag_cost,
      unavailable=unavailable,
  )


@dataclasses.dataclass(frozen=True)
class ProgramSummary:
  """The shape of a lowered IFRT IR program.

  Attributes:
    calls: Number of ``ifrt.Call`` ops (fragment executions) in ``@main``.
    copies: Number of ``ifrt.CopyArrays`` ops (cross-mesh transfers).
    pinned_copies: Copies with an ``after`` control dependency.
    programs: Number of distinct fragment programs called.
    calls_per_mesh: Fragment executions per mesh name.
  """

  calls: int
  copies: int
  pinned_copies: int
  programs: int
  calls_per_mesh: dict[str, int]

  def format_summary(self) -> str:
    """Returns the summary as aligned lines."""
    per_mesh = ", ".join(f"{mesh}:{count}" for mesh, count in sorted(self.calls_per_mesh.items()))
    return "\n".join(
        [
            "Lowered IFRT IR program:",
            f"  fragment executions : {self.calls} ({self.programs} programs)",
            f"  per mesh            : {per_mesh}",
            (f"  cross-mesh copies   : {self.copies}" f" ({self.pinned_copies} pinned to their producer)"),
        ]
    )


def summarize_ifrt(ifrt_text: str) -> ProgramSummary:
  """Counts the fragment executions and transfers of an IFRT IR program."""
  calls = 0
  copies = 0
  pinned = 0
  programs: set[str] = set()
  per_mesh: dict[str, int] = collections.Counter()
  for line in ifrt_text.splitlines():
    if "ifrt.Call " in line or "ifrt.Call @" in line:
      match = _CALL_RE.search(line)
      if match:
        calls += 1
        programs.add(match.group(1))
        mesh = _MESH_RE.search(line)
        per_mesh[mesh.group(1) if mesh else "?"] += 1
    elif "ifrt.CopyArrays(" in line:
      copies += 1
      pinned += bool(re.search(r"\)\s*after\s+%", line))
  return ProgramSummary(
      calls=calls,
      copies=copies,
      pinned_copies=pinned,
      programs=len(programs),
      calls_per_mesh=dict(per_mesh),
  )


def resolve_compile_devices(compile_topology: str, num_slices: int = 1, chip_config: str = "default") -> Sequence[Any]:
  """Returns the devices to lower against; see ``--compile_topology``."""
  if compile_topology in ("", "cpu"):
    return jax.devices("cpu")
  if compile_topology == "local":
    return jax.devices()
  # Imported lazily: the AOT topology machinery loads libtpu.
  from jax.experimental import topologies  # pylint: disable=import-outside-toplevel

  return topologies.get_topology_desc(
      platform="tpu",
      topology_name=compile_topology,
      chip_config_name=chip_config,
      chips_per_host_bounds=(2, 2, 1),
      num_slices=num_slices,
  ).devices


def run_compile(
    hparams: hparams_lib.Hparams,
    *,
    compile_topology: str = "cpu",
    num_slices: int = 1,
    chip_config: str = "default",
    ir_dump_path: str = "",
    skip_compile: str = "auto",
    config_overrides: dict[str, Any] | None = None,
) -> tuple[Any, Any, ProgramSummary]:
  """Lowers, and on Pathways compiles, the whole MPMD training step.

  Args:
    hparams: Pipeline hyperparameters.
    compile_topology: See ``--compile_topology``.
    num_slices: Number of slices in an AOT topology.
    chip_config: ``chip_config_name`` for an AOT topology.
    ir_dump_path: Optional path prefix for the lowered IR dumps.
    skip_compile: ``auto``, ``true`` or ``false``; see ``--skip_compile``.
    config_overrides: Extra MaxText settings, for tests.

  Returns:
    A tuple ``(lowered, compiled, summary)``; `compiled` is None when
    compilation was skipped.
  """
  devices = resolve_compile_devices(compile_topology, num_slices, chip_config)
  max_logging.log(
      f"Targeting {len(devices)} devices ({compile_topology}," f" {devices[0].platform}, {num_slices} slice(s))."
  )

  runnable = runner_lib.build(
      hparams,
      devices=devices,
      mode=train_step_lib.ExecutionMode.MPMD,
      abstract=True,
      config_overrides=config_overrides,
      check_microbatches=True,
  )
  note = hparams_lib.ragged_buffer_note(hparams)
  if note:
    max_logging.log(f"Note: {note}")
  max_logging.log(runnable.describe())

  max_logging.log("Lowering MPMD training step to IFRT IR...")
  lowered = runnable.lower()
  max_logging.log("Lowering succeeded.")

  ifrt_text = lowered.as_text(dialect="ifrt")
  summary = summarize_ifrt(ifrt_text)
  max_logging.log(summary.format_summary())

  if ir_dump_path:
    for dialect in ("stablehlo", "mpmd", "ifrt"):
      path = f"{ir_dump_path}.{dialect}.mlir"
      with open(path, "w", encoding="utf-8") as handle:
        handle.write(ifrt_text if dialect == "ifrt" else lowered.as_text(dialect=dialect))
      max_logging.log(f"Wrote {path}.")

  on_pathways = bool(pathwaysutils.is_pathways_backend_used())
  if skip_compile == "true" or (skip_compile == "auto" and not on_pathways):
    if not on_pathways:
      max_logging.log(
          "Not compiling: Shardy MPMD's IFRT IR program compiles only through a"
          " Pathways backend (JAX_PLATFORMS=proxy)."
      )
    return lowered, None, summary

  max_logging.log("Compiling whole MPMD pipeline program...")
  compiled = lowered.compile()
  max_logging.log("Compilation succeeded.")

  try:
    report = build_compilation_report(compiled)
  except Exception as error:  # pylint: disable=broad-exception-caught
    max_logging.warning(f"Compilation report unavailable: {error}")
  else:
    max_logging.log("=" * 72)
    max_logging.log(report.format_summary())
    max_logging.log("=" * 72)

  return lowered, compiled, summary


def main(argv: Sequence[str], flag_values: flags.FlagValues = flags.FLAGS) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected command line arguments: {argv[1:]}")
  # Registers the Pathways proxy backend when JAX_PLATFORMS contains "proxy";
  # it must run before the backend is first used.
  pathwaysutils.initialize()

  fv = flag_values
  if fv.compile_topology in ("", "cpu"):
    if not jax.config.jax_platforms:
      # Fake CPU devices only: on a TPU VM, JAX would otherwise take the TPU.
      jax.config.update("jax_platforms", "cpu")
    if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
      jax.config.update("jax_num_cpu_devices", fv.num_cpu_devices)
  run_compile(
      hparams_lib.from_flags(fv),
      compile_topology=fv.compile_topology,
      num_slices=fv.compile_topology_num_slices,
      chip_config=fv.chip_config,
      ir_dump_path=fv.ir_dump_path,
      skip_compile=fv.skip_compile,
  )


if __name__ == "__main__":
  define_flags()
  jax.config.config_with_absl()
  app.run(main)
