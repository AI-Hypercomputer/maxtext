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

"""Runs one phase of a CPU-mesh test on a forced number of CPU devices.

`--xla_force_host_platform_device_count` only takes effect before JAX initializes its backends, so tests that need
more CPU devices than the test process has run their device checks in a fresh child process. Splitting such a test
into phases, one test case and one child per phase, lets pytest-xdist (or sharded test runners) run the phases in
parallel and reports each phase's duration separately.

Usage, in a test module:

  _PHASES = {"reorder": _run_reorder_checks, "model": _run_model_checks}

  class CpuMeshTest(parameterized.TestCase):

    @parameterized.named_parameters(*((name, name) for name in _PHASES))
    def test_phase(self, phase):
      cpu_mesh_child.run_phase(__file__, phase, _PHASES, num_devices=8)

  if __name__ == "__main__":
    cpu_mesh_child.main(_PHASES)

A phase runs in the test process itself when that already has exactly `num_devices` CPU devices (a test target
whose env forces the device count, or pytest run with that XLA_FLAGS); otherwise in a child that re-executes the test
module with the phase named in the environment. A phase fails by raising.
"""

from collections.abc import Callable, Mapping
import os
import subprocess
import sys

from absl import app
from absl.testing import absltest
import jax

PHASE_ENV = "MAXTEXT_CPU_MESH_PHASE"
_PASS_TOKEN = "CPU_MESH_PHASE_PASSED"
# Test runners' XML / sharding outputs belong to the parent, so child processes must not write them.
_PARENT_RUNNER_ENV_VARS = (
    "XML_OUTPUT_FILE",
    "TEST_SHARD_INDEX",
    "TEST_TOTAL_SHARDS",
    "TEST_SHARD_STATUS_FILE",
)
# tests/utils/cpu_mesh_child.py -> the repository root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _pass_line(phase: str) -> str:
  return f"{_PASS_TOKEN}: {phase}"


def has_cpu_devices(num_devices: int) -> bool:
  """Whether this process runs on exactly `num_devices` CPU devices."""
  devices = jax.devices()
  return len(devices) == num_devices and devices[0].platform == "cpu"


def child_env(
    num_devices: int,
    phase: str,
    extra_env: Mapping[str, str] | None = None,
    *,
    extra_xla_flags: str = "",
) -> dict[str, str]:
  """The environment of a child that runs `phase` on `num_devices` forced CPU devices."""
  env = {k: v for k, v in os.environ.items() if k not in _PARENT_RUNNER_ENV_VARS}
  xla_flags = [
      env.get("XLA_FLAGS", ""),
      f"--xla_force_host_platform_device_count={num_devices}",
      extra_xla_flags,
  ]
  env["XLA_FLAGS"] = " ".join(flag for flag in xla_flags if flag).strip()
  env["JAX_PLATFORMS"] = "cpu"
  env[PHASE_ENV] = phase
  paths = [os.path.join(_REPO_ROOT, "src"), _REPO_ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
  env["PYTHONPATH"] = os.pathsep.join(paths)
  env.update(extra_env or {})
  return env


def run_phase(
    test_file: str,
    phase: str,
    phases: Mapping[str, Callable[[], object]],
    num_devices: int,
    *,
    in_process: bool = True,
    extra_env: Mapping[str, str] | None = None,
    extra_xla_flags: str = "",
) -> None:
  """Runs `phases[phase]` on `num_devices` CPU devices: here if this process has them, else in a child of `test_file`.

  `in_process=False` always uses a child (e.g. when the phase must set up a backend or XLA_FLAGS before JAX
  initializes).
  """
  assert phase in phases, f"unknown phase {phase!r}; known: {sorted(phases)}"
  if in_process and has_cpu_devices(num_devices):
    phases[phase]()
    return
  # Hermetic test binaries may have no `sys.executable`; re-run the binary itself in that case.
  cmd = [sys.executable, test_file] if sys.executable else [sys.argv[0]]
  result = subprocess.run(
      cmd,
      env=child_env(num_devices, phase, extra_env, extra_xla_flags=extra_xla_flags),
      capture_output=True,
      text=True,
      check=False,
  )
  print(result.stdout, flush=True)
  detail = f"phase {phase!r} of {test_file}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, detail
  assert _pass_line(phase) in result.stdout, detail


def main(phases: Mapping[str, Callable[[], object]]) -> None:
  """The `__main__` of a test module: in a child, runs the phase named in the environment; else the absltest suite."""
  phase = os.environ.get(PHASE_ENV)
  if not phase:
    absltest.main()
    return

  def run(argv):
    del argv
    phases[phase]()
    print(_pass_line(phase), flush=True)

  # Under app.run, as absltest.main runs the tests: some JAX builds refuse to initialize before absl has started.
  app.run(run)
