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

r"""Generate a small MaxText model that fits a per-device HBM budget.

Given a model already onboarded in MaxText, a target topology, and a per-device
HBM budget, greedily shrink the model (depth -> expert -> width by default)
until the real MaxText train step compiles within budget, and stop at the first
fit. Outputs:

  <output-dir>/checkpoint/           randomly initialized MaxText checkpoint
  <output-dir>/model_overrides.json  config overrides that turn `model_name` into the slice
  <output-dir>/slice_report.json     every compile attempt and the selected model

Example:

  python -m maxtext.checkpoint_conversion.slicing.slice_model \
      model_name=qwen3.5-35b-a3b compile_topology=v6e-8 per_device_batch_size=1 \
      --hbm-budget-per-device=24GiB --output-dir=/tmp/qwen35_small

Every `key=value` argument is an ordinary MaxText config override and defines the
workload that is measured (batch size, sequence length, remat, parallelism, ...).
An optional leading `*.yml` argument replaces the default `configs/base.yml`.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import absl
from absl import app
import jax

from maxtext.checkpoint_conversion.slicing import reducer
from maxtext.checkpoint_conversion.slicing import slice_utils
from maxtext.checkpoint_conversion.slicing.slice_utils import GiB
from maxtext.configs import pyconfig
from maxtext.utils import max_logging
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

# Target-topology / mesh keys, dropped when initializing the checkpoint on local devices.
_TOPOLOGY_KEY_PREFIXES = ("compile_topology", "ici_", "dcn_", "num_slices")
_SIZE_UNITS = {
    "": 1, "b": 1,
    "k": 2**10, "kib": 2**10, "m": 2**20, "mib": 2**20, "g": 2**30, "gib": 2**30, "t": 2**40, "tib": 2**40,
    "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
}  # fmt: skip


class SliceError(RuntimeError):
  """A user-facing failure, recorded as `status` in the report."""

  def __init__(self, status: str, message: str):
    super().__init__(message)
    self.status = status


# ----------------------------------------------------------------------------- CLI + config


def parse_size(text: str) -> int:
  """Parse `24GiB`, `24G` (binary), `24GB` (decimal), or a plain byte count."""
  match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", str(text))
  if not match or match.group(2).lower() not in _SIZE_UNITS:
    raise argparse.ArgumentTypeError(f"Invalid size {text!r}; expected e.g. 24GiB, 24GB, or a byte count.")
  return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()])


def parse_strategy(text: str) -> tuple[str, ...]:
  axes = tuple(a.strip() for a in text.split(",") if a.strip())
  if not axes or len(set(axes)) != len(axes) or any(a not in reducer.AXES for a in axes):
    raise argparse.ArgumentTypeError(f"--strategy must be a comma list of distinct axes from {reducer.AXES}.")
  return axes


def parse_flags(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
  """Split `argv` into slicer flags and the MaxText argv (`[prog, (*.yml), key=value, ...]`)."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--hbm-budget-per-device", type=parse_size, required=True, help="e.g. 24GiB")
  parser.add_argument("--hbm-safety-margin", type=parse_size, default=0, help="Subtracted from the budget.")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--strategy", type=parse_strategy, default=reducer.AXES, help="default: depth,expert,width")
  parser.add_argument("--min-experts", type=int, default=None, help="Lower bound for expert reduction.")
  parser.add_argument("--min-width-ratio", type=float, default=min(reducer.DEFAULT_WIDTH_RATIOS))
  parser.add_argument("--skip-checkpoint", action="store_true", help="Only write the overrides and report.")
  parser.add_argument("--seed", type=int, default=0, help="Seed for the random checkpoint initialization.")
  flags, rest = parser.parse_known_args(list(argv[1:]))

  kv_args = rest[1:] if rest and rest[0].endswith(".yml") else rest
  bad = [a for a in kv_args if "=" not in a or a.startswith("-")]
  if bad:
    parser.error(f"Unrecognized arguments {bad}; MaxText overrides must be key=value.")
  if not 0.0 < flags.min_width_ratio <= 1.0:
    parser.error("--min-width-ratio must be in (0, 1].")
  if flags.hbm_safety_margin >= flags.hbm_budget_per_device:
    parser.error("--hbm-safety-margin must be smaller than --hbm-budget-per-device.")
  return flags, [argv[0], *rest]


def split_maxtext_argv(argv: Sequence[str]) -> tuple[list[str], dict[str, str]]:
  """`[prog, (*.yml), k=v, ...]` -> (`[prog, (*.yml)]`, `{k: v}`)."""
  head = list(argv[:2]) if len(argv) > 1 and argv[1].endswith(".yml") else list(argv[:1])
  return head, dict(a.split("=", 1) for a in argv[len(head) :])


def check_onboarded(model_name: str | None) -> None:
  """Reject models without a MaxText model config (MaxText is the architecture source of truth)."""
  name = (model_name or "").replace("-Instruct", "")
  if not name or name == "default" or not (Path(MAXTEXT_CONFIGS_DIR) / "models" / f"{name}.yml").is_file():
    raise SliceError(
        "unsupported_model",
        f"Model {model_name!r} is not supported by MaxText slicing. Please onboard the model into MaxText first.",
    )


def load_config(maxtext_argv: Sequence[str], overrides: Mapping[str, Any] | None = None, drop_topology=False) -> Any:
  """`pyconfig.initialize` the user's MaxText argv with the slice overrides on top."""
  overrides = dict(overrides or {})
  head, kv = split_maxtext_argv(maxtext_argv)
  kv = {
      k: v for k, v in kv.items() if k not in overrides and not (drop_topology and k.startswith(_TOPOLOGY_KEY_PREFIXES))
  }
  defaults = {"enable_checkpointing": "false"}
  if kv.get("compile_topology"):
    defaults["compile_topology_num_slices"] = "1"
  # override_model_config lets the slice overrides replace values from the model YAML.
  merged = {**defaults, **kv, "override_model_config": "true"}
  return pyconfig.initialize([*head, *(f"{k}={v}" for k, v in merged.items())], **overrides)


def format_override(key: str, value: Any) -> str:
  if isinstance(value, (list, tuple)):
    return f"{key}={json.dumps(list(value), separators=(',', ':'))}"
  return f"{key}={value}"


# ----------------------------------------------------------------------------- greedy search


@dataclasses.dataclass
class Attempt:
  """One compiled candidate: the overrides tried and how its peak HBM compared to the limit."""

  axis: str
  overrides: dict[str, Any]
  result: str  # "fits" | "too_large" | "oom" | "error"
  peak_hbm_bytes: int | None = None

  def to_dict(self, config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"axis": self.axis, **reducer.describe(config, self.overrides), "result": self.result}
    if self.peak_hbm_bytes is not None:
      out["peak_hbm_gib"] = round(self.peak_hbm_bytes / GiB, 2)
    return out


@dataclasses.dataclass
class SearchLog:
  """Attempts and skipped-axis notes, owned by the caller so they survive a failed search."""

  attempts: list[Attempt] = dataclasses.field(default_factory=list)
  notes: list[str] = dataclasses.field(default_factory=list)


def search(
    config: Any,
    strategy: Sequence[str],
    measure: Callable[[dict[str, Any]], int | None],
    limit_bytes: int,
    steps_for_axis: Callable[[str], list[dict[str, Any]]],
    *,
    log: SearchLog,
) -> dict[str, Any] | None:
  """Greedy: try the full model, then walk each axis in order; return the first overrides that fit.

  `measure(overrides)` returns per-device peak HBM bytes, or None for a compile-time OOM.
  Any other exception from `measure` stops the search (after being recorded in `log`).
  """

  def fits(axis: str, overrides: dict[str, Any]) -> bool:
    try:
      peak = measure(overrides)
    except Exception:
      log.attempts.append(Attempt(axis, dict(overrides), "error"))
      raise
    result = "oom" if peak is None else ("fits" if peak <= limit_bytes else "too_large")
    log.attempts.append(Attempt(axis, dict(overrides), result, peak))
    peak_text = "OOM" if peak is None else f"{peak / GiB:.2f} GiB"
    max_logging.log(f"slice_model {axis:>6}: {reducer.describe(config, overrides)} -> {peak_text} ({result})")
    return result == "fits"

  current: dict[str, Any] = {}
  if fits("full", current):
    return current
  for axis in strategy:
    try:
      steps = steps_for_axis(axis)
    except reducer.UnsupportedSliceError as exc:
      log.notes.append(f"{axis}: skipped ({exc})")
      max_logging.log(f"slice_model {axis}: skipped ({exc})")
      continue
    for step in steps:
      candidate = {**current, **step}
      if fits(axis, candidate):
        return candidate
      current = candidate
  return None


# ----------------------------------------------------------------------------- main


def write_json(path: Path, payload: Any) -> None:
  path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str], flags: argparse.Namespace) -> int:
  """Run the slicer. `argv` is the MaxText argv; `flags` are the slicer flags from `parse_flags`."""
  _, maxtext_args = split_maxtext_argv(argv)
  out_dir = Path(flags.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  limit = flags.hbm_budget_per_device - flags.hbm_safety_margin
  report: dict[str, Any] = {
      "status": "running",
      "model_name": maxtext_args.get("model_name"),
      "target": {
          "topology": maxtext_args.get("compile_topology"),
          "hbm_budget_gib": round(flags.hbm_budget_per_device / GiB, 3),
          "hbm_safety_margin_gib": round(flags.hbm_safety_margin / GiB, 3),
          "hbm_limit_gib": round(limit / GiB, 3),
      },
      "maxtext_args": maxtext_args,
      "strategy": list(flags.strategy),
  }
  config = None
  log = SearchLog()
  try:
    check_onboarded(maxtext_args.get("model_name"))
    if not maxtext_args.get("compile_topology"):
      raise SliceError("invalid_request", "compile_topology=<target, e.g. v6e-8> is required.")
    config = load_config(argv)
    try:
      plan = reducer.get_depth_plan(config)
    except reducer.UnsupportedSliceError as exc:
      raise SliceError("unsupported_slicing_rule", str(exc)) from exc
    report["depth_plan"] = dataclasses.asdict(plan)
    mesh_shape = slice_utils.target_mesh_shape(config)
    report["target"]["mesh"] = mesh_shape

    def measure(overrides: dict[str, Any]) -> int | None:
      try:
        candidate_config = load_config(argv, overrides)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        raise SliceError("invalid_candidate_config", f"MaxText rejected {overrides}: {exc}") from exc
      return slice_utils.measure_peak_hbm(candidate_config)

    def steps_for_axis(axis: str) -> list[dict[str, Any]]:
      if axis == "depth":
        return reducer.reduce_depth(config, plan)
      if axis == "expert":
        return reducer.reduce_experts(config, mesh_shape, flags.min_experts)
      return reducer.reduce_width(config, mesh_shape, flags.min_width_ratio)

    selected = search(config, flags.strategy, measure, limit, steps_for_axis, log=log)
    if selected is None:
      raise SliceError("no_model_fits", "No legal slice fits the HBM limit; see attempts.")

    write_json(out_dir / "model_overrides.json", selected)
    report["selected"] = {
        **reducer.describe(config, selected),
        "peak_hbm_gib": round(log.attempts[-1].peak_hbm_bytes / GiB, 2),
        "overrides": selected,
    }
    launch_args = [f"model_name={maxtext_args['model_name']}", "override_model_config=true"]
    launch_args += [format_override(k, v) for k, v in selected.items()]
    report["launch_args"] = launch_args

    if flags.skip_checkpoint:
      report["checkpoint"] = {"type": "skipped"}
    else:
      try:
        ckpt_config = load_config(argv, {**selected, "dataset_type": "synthetic"}, drop_topology=True)
        load_path = slice_utils.write_random_checkpoint(ckpt_config, str(out_dir / "checkpoint"), flags.seed)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        raise SliceError("checkpoint_failed", f"{type(exc).__name__}: {exc}") from exc
      report["checkpoint"] = {
          "type": "random_initialized",
          "seed": flags.seed,
          "load_parameters_path": load_path,
          "note": "Freshly initialized weights for bring-up/verification; does not preserve pretrained behavior.",
      }
      launch_args.append(f"load_parameters_path={load_path}")
    report["status"] = "selected"
  except SliceError as exc:
    report.update(status=exc.status, error=str(exc))
  except slice_utils.CompileFailedError as exc:
    report.update(status="compile_failed", error=str(exc))
  finally:
    if report["status"] == "running":
      report["status"] = "crashed"
    report["notes"] = log.notes
    if config is not None:
      report["attempts"] = [a.to_dict(config) for a in log.attempts]
    write_json(out_dir / "slice_report.json", report)

  max_logging.log(f"slice_model status={report['status']} report={out_dir / 'slice_report.json'}")
  if report["status"] != "selected":
    max_logging.error(f"slice_model {report.get('error', '')}")
    return 1
  return 0


if __name__ == "__main__":
  # Same compile-affecting settings as train_compile.main.
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log
  slicer_flags, model_argv = parse_flags(sys.argv)
  app.run(functools.partial(main, flags=slicer_flags), argv=model_argv)
