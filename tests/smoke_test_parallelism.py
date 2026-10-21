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

"""Hermetic smoke test for MaxText sweep configs and parallelism without TPUs.

Performs multi-stage pre-flight validation on any MaxText sweep or configuration:
  1. Configuration Parsing & Pydantic Validation (types, rules, schema constraints).
  2. Parallelism Arithmetic & Hardware Device Matching (ICI/DCN product rules).
  3. Compile-Only Device Mesh Creation (using compile-only topology devices).
  4. Abstract Model & Parameter Sharding Verification (eval_shape / get_shaped_inputs).
  5. Memory & Remat Policy Sanity Check (flags potential OOM hazards).
"""

from collections.abc import Sequence
import itertools
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np
import yaml

import jax
from jax.experimental import topologies
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from maxtext.configs import pyconfig
from maxtext.common.common_types import ShardMode
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_SWEEP_FILE = flags.DEFINE_string(
    "sweep_file",
    "",
    "Path to YAML sweep configuration file to smoke test.",
)
_SCRIPT_FILE = flags.DEFINE_string(
    "script_file",
    "",
    "Path to reference shell script to smoke test.",
)
_SCRIPTS_DIR = flags.DEFINE_string(
    "scripts_dir",
    "",
    "Path to directory containing reference shell scripts to smoke test.",
)
_PLATFORM = flags.DEFINE_string(
    "platform",
    "gf_4x4x4",
    "Target accelerator platform, e.g. gf_4x4x4, v5e-256, v6e-16.",
)
_NUM_SLICES = flags.DEFINE_integer(
    "num_slices",
    1,
    "Number of slices for the target topology.",
)
_CHECK_MODEL_SHARDING = flags.DEFINE_boolean(
    "check_model_sharding",
    True,
    "Whether to build abstract model and verify parameter shardings.",
)
_FAIL_FAST = flags.DEFINE_boolean(
    "fail_fast",
    False,
    "Whether to exit immediately on first trial failure.",
)


class TrialResult:
  """Structured result of a single trial smoke test."""

  def __init__(self, trial_name: str, model_name: str):
    self.trial_name = trial_name
    self.model_name = model_name
    self.passed = False
    self.failed_stage: Optional[str] = None
    self.error_message: Optional[str] = None
    self.warning_message: Optional[str] = None
    self.ici_parallelism: Optional[List[int]] = None
    self.dcn_parallelism: Optional[List[int]] = None
    self.remat_policy: Optional[str] = None
    self.param_count: Optional[int] = None


def resolve_platform_spec(platform: str, num_slices: int) -> Tuple[str, int, bool]:
  """Resolves platform string to (topology_name, devices_per_slice, is_internal)."""
  normalized = platform.strip().lower()

  # 1. Ghostfish patterns: gf_4x4x4, gf=4x4x4, etc.
  if normalized.startswith("gf_") or normalized.startswith("gf="):
    topo_core = normalized.replace("gf_", "").replace("gf=", "")
    dims = [int(x) for x in topo_core.split("x")]
    # Ghostfish has 64 chips = 128 devices (2 TensorCores per chip)
    chips = int(np.prod(dims))
    devices_per_slice = chips * 2
    return f"gf={topo_core}", devices_per_slice, True

  # 2. Open-source or standardized TPU names: e.g. v5e-256, v4-128, tpu7x-128
  if normalized in accelerator_to_spec_map.UserFacingNameToSystemCharacteristics:
    spec = accelerator_to_spec_map.get_system_characteristics(normalized)
    topo_name = spec.topology_name if spec.topology_name else normalized
    return topo_name, spec.devices_per_slice, False

  # 3. Internal formats: pf=..., vf=..., df=...
  for prefix in ["pf", "vf", "df", "gfc", "glp"]:
    if normalized.startswith(f"{prefix}_") or normalized.startswith(f"{prefix}="):
      topo_core = normalized.replace(f"{prefix}_", "").replace(f"{prefix}=", "")
      dims = [int(x) for x in topo_core.split("x")]
      chips = int(np.prod(dims))
      devices_per_slice = chips * 2 if prefix in ["pf", "gfc"] else chips
      return f"{prefix}={topo_core}", devices_per_slice, True

  # Fallback: assume chips x 2
  if "_" in normalized and all(p.isdigit() for p in normalized.split("_")[-1].split("x")):
    dims = [int(x) for x in normalized.split("_")[-1].split("x")]
    chips = int(np.prod(dims))
    return normalized.replace("_", "="), chips * 2, True

  raise ValueError(f"Unable to parse platform: {platform}")


def expand_macros(
    maxtext_flags: Dict[str, Any],
    extra_flags: Dict[str, Any],
    macros: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Expands macros defined in sweep configs."""
  result_mt = dict(maxtext_flags)
  result_extra = dict(extra_flags)

  macro_names = result_mt.pop("macros", [])
  if isinstance(macro_names, str):
    macro_names = [macro_names]

  for m_name in macro_names:
    if m_name in macros:
      macro_def = macros[m_name]
      if "maxtext_flags" in macro_def:
        nested_mt, nested_extra = expand_macros(
            macro_def["maxtext_flags"],
            macro_def.get("extra_flags", {}),
            macros,
        )
        result_mt.update(nested_mt)
        result_extra.update(nested_extra)
      else:
        result_mt.update(macro_def)

  return result_mt, result_extra


def resolve_file_path(path: str) -> str:
  """Resolves file path against cwd, BUILD_WORKING_DIRECTORY, or workspace."""
  if os.path.exists(path):
    return path
  candidates = [
      os.path.join(os.environ.get("BUILD_WORKING_DIRECTORY", ""), path),
      os.path.join(os.environ.get("BUILD_WORKSPACE_DIRECTORY", ""), path),
      os.path.join(os.getcwd(), path),
  ]
  for c in candidates:
    if c and os.path.exists(c):
      return c
  raise FileNotFoundError(f"Sweep file not found: {path} (checked {candidates})")


def load_sweep_trials(sweep_file_path: str) -> List[Dict[str, Any]]:
  """Loads trials from YAML sweep file."""
  sweep_file_path = resolve_file_path(sweep_file_path)

  with open(sweep_file_path, "r") as f:
    config = yaml.safe_load(f) or {}

  macros = config.get("macros", {})
  defaults = config.get("defaults", {})
  base_maxtext = dict(defaults.get("maxtext_flags", {}))
  base_extra = dict(defaults.get("extra_flags", {}))

  generators = config.get("generators", [])
  explicit_trials = config.get("explicit_trials", [])

  trials: List[Dict[str, Any]] = []

  if not generators and not explicit_trials:
    exp_mt, exp_ex = expand_macros(base_maxtext, base_extra, macros)
    trials.append({"name": "default", "flags": exp_mt, "extra": exp_ex})
    return trials

  for gen in generators:
    gen_type = gen.get("type")
    name_template = gen.get("name_template")
    targets = gen.get("targets", [])
    values = gen.get("values", [])
    grid = gen.get("grid", {})

    grid_keys = list(grid.keys())
    grid_val_combos = (
        list(itertools.product(*[grid[k] for k in grid_keys]))
        if grid_keys
        else [()]
    )

    if gen_type == "uniform":
      tmpl = name_template or "all_{value}"
      for val in values:
        for combo in grid_val_combos:
          grid_dict = dict(zip(grid_keys, combo))
          trial_mt = dict(base_maxtext)
          for t in targets:
            trial_mt[t] = val
          trial_mt.update(grid_dict)
          name = tmpl.replace("{value}", str(val))
          exp_mt, exp_ex = expand_macros(trial_mt, base_extra, macros)
          trials.append({"name": name, "flags": exp_mt, "extra": exp_ex})

    elif gen_type == "one_at_a_time":
      tmpl = name_template or "{target}_{value}"
      for target in targets:
        for val in values:
          for combo in grid_val_combos:
            grid_dict = dict(zip(grid_keys, combo))
            trial_mt = dict(base_maxtext)
            trial_mt[target] = val
            trial_mt.update(grid_dict)
            name = tmpl.replace("{target}", str(target)).replace("{value}", str(val))
            exp_mt, exp_ex = expand_macros(trial_mt, base_extra, macros)
            trials.append({"name": name, "flags": exp_mt, "extra": exp_ex})

  for exp_entry in explicit_trials:
    name = exp_entry.get("name", "explicit_trial")
    trial_mt = dict(base_maxtext)
    trial_mt.update(exp_entry.get("maxtext_flags", {}))
    trial_ex = dict(base_extra)
    trial_ex.update(exp_entry.get("extra_flags", {}))
    exp_mt, exp_ex = expand_macros(trial_mt, trial_ex, macros)
    trials.append({"name": name, "flags": exp_mt, "extra": exp_ex})

  return trials


def _coerce_value(val: str) -> Any:
  """Coerces string flag values to boolean, integer, float, or string."""
  if val.lower() == "true":
    return True
  if val.lower() == "false":
    return False
  try:
    return int(val)
  except ValueError:
    pass
  try:
    return float(val)
  except ValueError:
    pass
  return val


def load_script_trial(script_path: str) -> List[Dict[str, Any]]:
  """Parses a reference bash launch script into trial dictionary/dictionaries."""
  script_path = resolve_file_path(script_path)
  with open(script_path, "r") as f:
    content = f.read()

  variables: Dict[str, Any] = {}
  raw_maxtext_flags: Dict[str, str] = {}
  raw_extra_flags: Dict[str, str] = {}

  for line in content.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    # Array variables: e.g. MODELS=("gemma4-26b")
    m_arr = re.match(r"^(?:export\s+)?([A-Za-z_0-9]+)=\s*\(\s*([^)]*)\s*\)", line)
    if m_arr:
      k = m_arr.group(1)
      raw_items = m_arr.group(2).split()
      items = [item.strip("\"'") for item in raw_items if item.strip("\"'")]
      variables[k] = items
      continue
    # Scalar variable: export FOO="bar" or FOO=bar
    m_var = re.match(r"^(?:export\s+)?([A-Za-z_0-9]+)=([\"']?)([^\"'#\n]*)\2", line)
    if m_var:
      variables[m_var.group(1)] = m_var.group(3).strip()

  for line in content.splitlines():
    line = line.strip().rstrip("\\").strip()
    m_mt = re.match(r"--maxtext_flag=[\"']?([A-Za-z_0-9]+)=(.*)", line)
    if m_mt:
      raw_maxtext_flags[m_mt.group(1)] = m_mt.group(2).strip().strip("\"'")
    m_ex = re.match(r"--extra_flags=[\"']?([A-Za-z_0-9]+)=(.*)", line)
    if m_ex:
      raw_extra_flags[m_ex.group(1)] = m_ex.group(2).strip().strip("\"'")

  def expand_vars(text: str, current_model: str = "") -> str:
    user = os.environ.get("USER", "maxtext_user")
    for _ in range(5):
      if current_model:
        text = text.replace("${MODEL}", current_model).replace("$MODEL", current_model)
      for k, v in variables.items():
        if isinstance(v, str):
          text = text.replace(f"${{{k}}}", v).replace(f"${k}", v)
      text = re.sub(r"\$\(whoami\)", user, text)
      text = re.sub(r"\$\(date[^\)]*\)", "20260912", text)
    # Strip any remaining unexpanded ${...} to prevent OmegaConf interpolation errors
    text = re.sub(r"\$\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\$([A-Za-z_0-9]+)", r"\1", text)
    return text

  models = variables.get("MODELS", [])
  if not models and "model_name" in raw_maxtext_flags:
    models = [expand_vars(raw_maxtext_flags["model_name"])]
  if not models:
    models = ["unknown"]

  base_name = os.path.splitext(os.path.basename(script_path))[0]
  trials: List[Dict[str, Any]] = []
  for m in models:
    t_flags = {
        k: _coerce_value(expand_vars(v, current_model=m))
        for k, v in raw_maxtext_flags.items()
    }
    t_extra = {
        k: _coerce_value(expand_vars(v, current_model=m))
        for k, v in raw_extra_flags.items()
    }
    t_flags["model_name"] = m
    platform = expand_vars(variables.get("PLATFORM", "gf_4x4x4"), current_model=m)
    # Provide a dummy checkpoint path for dry run if empty
    if "load_parameters_path" in t_flags and not t_flags["load_parameters_path"]:
      t_flags["load_parameters_path"] = "dummy_checkpoint_path"
    trial_name = f"{base_name}_{m}" if len(models) > 1 else base_name
    trials.append({
        "name": trial_name,
        "platform": platform,
        "flags": t_flags,
        "extra": t_extra,
    })
  return trials


def load_scripts_from_dir(directory: str) -> List[Dict[str, Any]]:
  """Recursively finds and loads trials from all bash scripts in directory."""
  dir_path = resolve_file_path(directory)
  all_trials: List[Dict[str, Any]] = []
  for root, _, files in os.walk(dir_path):
    for f in sorted(files):
      if f.endswith(".sh") and not f.startswith("run_all_smoke_tests"):
        script_full_path = os.path.join(root, f)
        rel_path = os.path.relpath(script_full_path, dir_path)
        trials = load_script_trial(script_full_path)
        for t in trials:
          t["name"] = f"{os.path.dirname(rel_path)}/{t['name']}" if os.path.dirname(rel_path) else t["name"]
          all_trials.append(t)
  return all_trials


def smoke_test_trial(
    trial: Dict[str, Any],
    topology_name: str,
    devices_per_slice: int,
    num_slices: int,
    is_internal: bool,
    check_model_sharding: bool,
) -> TrialResult:
  """Runs the 5-stage smoke test on a single trial."""
  trial_name = trial["name"]
  flags_dict = trial["flags"]
  model_name = str(flags_dict.get("model_name", "unknown"))
  result = TrialResult(trial_name, model_name)

  # --------------------------------------------------------------------------
  # Stage 1: Build argv and run Pydantic Config Validation
  # --------------------------------------------------------------------------
  cli_args = [
      "",
      get_test_config_path(),
      f"compile_topology={topology_name}",
      f"compile_topology_num_slices={num_slices}",
      f"internal_compile={'true' if is_internal else 'false'}",
  ]
  if is_internal:
    cli_args.append(f"internal_compile_num_devices={devices_per_slice * num_slices}")
  cli_args.extend([
      "skip_jax_distributed_system=true",
      "enable_checkpointing=false",
  ])
  for k, v in flags_dict.items():
    if isinstance(v, bool):
      cli_args.append(f"{k}={'true' if v else 'false'}")
    elif isinstance(v, (list, tuple)):
      cli_args.append(f"{k}={','.join(str(x) for x in v)}")
    else:
      cli_args.append(f"{k}={v}")

  try:
    config = pyconfig.initialize(cli_args)
    train_compile.validate_config(config)
  except Exception as e:
    result.failed_stage = "1. Config Validation"
    result.error_message = f"{type(e).__name__}: {str(e)}"
    return result

  result.remat_policy = getattr(config, "remat_policy", "default")

  # --------------------------------------------------------------------------
  # Stage 2: Parallelism Arithmetic Validation
  # --------------------------------------------------------------------------
  try:
    ici_parallelism = config.ici_parallelism.copy()
    filled_ici = max_utils.fill_unspecified_mesh_axes(
        ici_parallelism, devices_per_slice, "ICI"
    )
    result.ici_parallelism = list(filled_ici)

    ici_product = int(np.prod(filled_ici))
    if ici_product != devices_per_slice:
      result.failed_stage = "2. ICI Parallelism Arithmetic"
      result.error_message = (
          f"Number of devices per slice {devices_per_slice} does not match "
          f"the product of the ICI parallelism {ici_product} (axes: {filled_ici})"
      )
      return result

    dcn_parallelism = config.dcn_parallelism.copy()
    filled_dcn = max_utils.fill_unspecified_mesh_axes(
        dcn_parallelism, num_slices, "DCN"
    )
    result.dcn_parallelism = list(filled_dcn)
    dcn_product = int(np.prod(filled_dcn))
    if dcn_product != num_slices:
      result.failed_stage = "2. DCN Parallelism Arithmetic"
      result.error_message = (
          f"Number of slices {num_slices} does not match the product of "
          f"the DCN parallelism {dcn_product} (axes: {filled_dcn})"
      )
      return result

  except Exception as e:
    result.failed_stage = "2. Parallelism Arithmetic"
    result.error_message = f"{type(e).__name__}: {str(e)}"
    return result

  # --------------------------------------------------------------------------
  # Stage 3: Compile-Only Device Mesh Creation
  # --------------------------------------------------------------------------
  try:
    topo_desc = topologies.get_topology_desc(
        topology_name=topology_name,
        platform="tpu",
        num_slices=num_slices,
    )
    topology_devices = topo_desc.devices
    if len(topology_devices) != devices_per_slice * num_slices:
      result.failed_stage = "3. Topology Devices"
      result.error_message = (
          f"Expected {devices_per_slice * num_slices} devices from topology "
          f"{topology_name}, but got {len(topology_devices)}"
      )
      return result

    topology_device_mesh = maxtext_utils.create_device_mesh(config, topology_devices)
    mesh_axis_type = (
        AxisType.Explicit
        if config.shard_mode == ShardMode.EXPLICIT
        else AxisType.Auto
    )
    topology_mesh = Mesh(
        topology_device_mesh,
        config.mesh_axes,
        axis_types=(mesh_axis_type,) * len(config.mesh_axes),
    )
  except Exception as e:
    result.failed_stage = "3. Mesh Creation"
    result.error_message = f"{type(e).__name__}: {str(e)}"
    return result

  # --------------------------------------------------------------------------
  # Stage 4: Abstract Model Creation & Sharding Check (eval_shape)
  # --------------------------------------------------------------------------
  if check_model_sharding:
    try:
      (
          shaped_train_args,
          shaped_train_kwargs,
          state_mesh_shardings,
          logical_annotations,
          model,
      ) = train_compile.get_shaped_inputs(topology_mesh, config)

      abstract_state = shaped_train_args[0]
      params = getattr(abstract_state, "params", None)
      if params:
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        result.param_count = total_params

    except Exception as e:
      result.failed_stage = "4. Model & Sharding"
      result.error_message = f"{type(e).__name__}: {str(e)}"
      return result

  # --------------------------------------------------------------------------
  # Stage 5: Memory & Remat Policy Sanity Check
  # --------------------------------------------------------------------------
  if result.param_count and result.param_count > 10_000_000_000:
    if result.remat_policy == "none":
      result.warning_message = (
          f"HIGH OOM RISK: {model_name} (~{result.param_count / 1e9:.1f}B params) "
          f"has remat_policy='none'. On 96GB HBM devices this will likely OOM. "
          f"Consider setting remat_policy='full'."
      )

  result.passed = True
  return result


def main(argv: Sequence[str]) -> None:
  trials: List[Dict[str, Any]] = []
  target_description = ""

  if _SCRIPTS_DIR.value:
    target_description = f"Scripts Directory: {_SCRIPTS_DIR.value}"
    trials = load_scripts_from_dir(_SCRIPTS_DIR.value)
  elif _SCRIPT_FILE.value:
    target_description = f"Script File:       {_SCRIPT_FILE.value}"
    trials = load_script_trial(_SCRIPT_FILE.value)
  elif _SWEEP_FILE.value:
    target_description = f"Sweep File:        {_SWEEP_FILE.value}"
    trials = load_sweep_trials(_SWEEP_FILE.value)
  elif len(argv) > 1:
    target = argv[1]
    if os.path.isdir(target):
      target_description = f"Scripts Directory: {target}"
      trials = load_scripts_from_dir(target)
    elif target.endswith(".sh"):
      target_description = f"Script File:       {target}"
      trials = load_script_trial(target)
    else:
      target_description = f"Sweep File:        {target}"
      trials = load_sweep_trials(target)
  else:
    print(
        "Error: Must specify --scripts_dir, --script_file, or --sweep_file",
        file=sys.stderr,
    )
    sys.exit(2)

  print("=" * 80)
  print("MaxText Parallelism & Configuration Smoke Test")
  print("=" * 80)
  print(f"  Target:            {target_description}")
  print(f"  Default Platform:  {_PLATFORM.value}")
  print(f"  Number of Slices:  {_NUM_SLICES.value}")
  print(f"  Model Sharding:    {_CHECK_MODEL_SHARDING.value}")
  print("-" * 80)
  print(f"Found {len(trials)} trial(s) to test.\n")

  results: List[TrialResult] = []
  any_failed = False

  for i, trial in enumerate(trials):
    trial_name = trial["name"]
    trial_platform = trial.get("platform", _PLATFORM.value)
    topology_name, devices_per_slice, is_internal = resolve_platform_spec(
        trial_platform, _NUM_SLICES.value
    )
    print(
        f"[{i+1:02d}/{len(trials):02d}] Testing: {trial_name} ({trial_platform}) ... ",
        end="",
        flush=True,
    )
    res = smoke_test_trial(
        trial,
        topology_name=topology_name,
        devices_per_slice=devices_per_slice,
        num_slices=_NUM_SLICES.value,
        is_internal=is_internal,
        check_model_sharding=_CHECK_MODEL_SHARDING.value,
    )
    results.append(res)

    if res.passed:
      param_str = f" (~{res.param_count / 1e9:.1f}B params)" if res.param_count else ""
      print(f"PASS{param_str}")
      if res.warning_message:
        print(f"       WARNING: {res.warning_message}")
    else:
      any_failed = True
      print(f"FAIL [{res.failed_stage}]")
      print(f"       ERROR: {res.error_message}")
      if _FAIL_FAST.value:
        break

  print("\n" + "=" * 80)
  print("SMOKE TEST SUMMARY REPORT")
  print("=" * 80)
  print(f"{'#':<3} {'Trial Name':<32} {'Model':<16} {'ICI Shape':<14} {'Remat':<6} {'Status':<8}")
  print("-" * 80)
  for i, r in enumerate(results):
    status_str = "PASS" if r.passed else "FAIL"
    ici_str = str(r.ici_parallelism) if r.ici_parallelism else "N/A"
    remat_str = str(r.remat_policy) if r.remat_policy else "N/A"
    print(f"{i+1:<3} {r.trial_name[:32]:<32} {r.model_name[:16]:<16} {ici_str[:14]:<14} {remat_str:<6} {status_str:<8}")
    if not r.passed:
      print(f"    >>> Failed at {r.failed_stage}: {r.error_message}")
    elif r.warning_message:
      print(f"    >>> Warning: {r.warning_message}")

  print("-" * 80)
  pass_count = sum(1 for r in results if r.passed)
  print(f"Total: {len(results)} | Passed: {pass_count} | Failed: {len(results) - pass_count}")
  print("=" * 80)

  if any_failed:
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
