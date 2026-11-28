# Copyright 2025 Google LLC
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

"""Pytest configuration helpers for decoupled test selection.

Automatically apply the `decoupled` marker (when DECOUPLE_GCLOUD=TRUE) to
tests that remain collected. Tests that are explicitly skipped because they
require external integrations or specific hardware (for example `tpu_only`)
are not marked.
"""

import contextlib
import gc
import os
import sys
import unittest
import warnings
import pytest

# For explicit CPU-only test runs (e.g., local `JAX_PLATFORMS=cpu pytest ...`), skip post-HLO
# LLVM backend optimization passes and 32-way parallel codegen splitting unless the caller already
# configured those flags. Never applied when JAX_PLATFORMS is unset or targets TPU/GPU, and only
# active inside pytest (never production runs).
_CPU_TEST_XLA_DEFAULTS = (
    ("--xla_backend_optimization_level", "0"),
    ("--xla_llvm_disable_expensive_passes", "true"),
    ("--xla_cpu_parallel_codegen_split_count", "1"),
)
if os.getenv("JAX_PLATFORMS", "").lower() == "cpu":
  _existing_xla_flags = os.environ.get("XLA_FLAGS", "")
  _missing_xla_flags = [f"{flag}={value}" for flag, value in _CPU_TEST_XLA_DEFAULTS if flag not in _existing_xla_flags]
  if _missing_xla_flags:
    os.environ["XLA_FLAGS"] = f"{_existing_xla_flags} {' '.join(_missing_xla_flags)}".strip()

warnings.filterwarnings(
    "ignore", message="builtin type swigvarlink has no __module__ attribute", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message="builtin type SwigPyPacked has no __module__ attribute", category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", message="builtin type SwigPyObject has no __module__ attribute", category=DeprecationWarning
)

# Prevent libraries that use absl flags (e.g. tokamax) from lazily parsing sys.argv,
# which would pick up pytest flags like `-v -m` and fail to parse them as integers.
from absl import flags as _absl_flags

if not _absl_flags.FLAGS.is_parsed():
  _absl_flags.FLAGS(sys.argv[:1])

import jax
import importlib.util

# Force early JAX initialization on GPU to prevent CUDA context conflicts with TensorFlow/PyTorch.
# If JAX initialization is deferred, TensorFlow/PyTorch (imported during test collection)
# might initialize CUDA first, causing JAX's subsequent NCCL communicator creation to fail
# with 'corrupted comm object detected'.
# Detect GPU environment using standard JAX env vars, GHA runner device types,
# and nvidia-docker visible device markers.
_jax_platforms = os.getenv("JAX_PLATFORMS", "").lower()
_device_type = os.getenv("INPUTS_DEVICE_TYPE", "").lower()
_has_gpu = (
    "cuda" in _jax_platforms
    or "gpu" in _jax_platforms
    or "cuda" in _device_type
    or "gpu" in _device_type
    or os.getenv("CUDA_VISIBLE_DEVICES") is not None
    or os.getenv("NVIDIA_VISIBLE_DEVICES") is not None
)
if _has_gpu:
  try:
    _ = jax.devices()
  except Exception:  # pylint: disable=broad-exception-caught
    pass

# --- Monkeypatch for absl.testing.parameterized ---
# Context: Decorating a test method with @parameterized.named_parameters returns a custom
# iterable container (_ParameterizedTestIter) instead of a standard function object.
# Problem: When pytest markers are applied above @parameterized in the decorator stack:
#
#   @pytest.mark.cpu_only
#   @parameterized.named_parameters(...)
#   def test_foo(self, ...):
#
# pytest attaches the marker attributes exclusively to the outer iterable container object.
# During class initialization, the test metaclass unwraps the base function to generate
# individual test methods, omitting the outer container entirely. Consequently, marker
# attributes attached to the outer container are dropped and lost before pytest collection.
# Solution: Intercept _ParameterizedTestIter.__iter__ to dynamically propagate any discovered
# pytestmark attributes from the outer container object down to all generated test methods.
from absl.testing import parameterized

try:
  # pylint: disable=protected-access
  _orig_iter = parameterized._ParameterizedTestIter.__iter__

  def _custom_iter(self):
    """Custom iterator propagating outer pytestmark attributes to generated test methods."""
    outer_marks = getattr(self, "pytestmark", None)
    if outer_marks is None:
      yield from _orig_iter(self)
    else:
      if not isinstance(outer_marks, list):
        outer_marks = [outer_marks]

      for func in _orig_iter(self):
        existing_marks = getattr(func, "pytestmark", [])
        if not isinstance(existing_marks, list):
          existing_marks = [existing_marks]
        func.pytestmark = existing_marks + outer_marks
        yield func

  parameterized._ParameterizedTestIter.__iter__ = _custom_iter
  # pylint: enable=protected-access
except AttributeError:
  pass


if os.getenv("JAX_PLATFORMS") == "proxy":
  # Import maxtext early to register the pathways proxy backend before JAX is queried.
  import maxtext  # pylint: disable=unused-import

from maxtext.common.gcloud_stub import is_decoupled
from tests.utils import class_split
from tests.utils.newly_added_detection import get_changed_tests
from tests.utils.newly_added_detection import is_changed_item

# Configure JAX to use unsafe_rbg PRNG implementation to match main scripts.
if is_decoupled():
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")


GCP_MARKERS = {"external_serving", "external_training"}
HARDWARE_MARKERS = {"tpu_only", "gpu_only", "cpu_only"}


def _has_tpu_backend_support() -> bool:
  """Whether JAX has TPU backend support installed (PJRT TPU plugin).

  This is intentionally *not* the same as having TPU hardware available.
  """
  try:
    if importlib.util.find_spec("jaxlib") is None:
      return False
  except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
    return False

  # Heuristic: TPU backend support is provided via the `libtpu` package.
  try:
    return importlib.util.find_spec("libtpu") is not None
  except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
    return False


_FRAMEWORK_TESTCASE_MODULES = frozenset({"unittest.case", "absl.testing.absltest", "absl.testing.parameterized"})


def _has_shared_class_fixture(item) -> bool:
  """Returns whether `item` belongs to a class with shared class-level setup/teardown or fixtures.

  Excludes framework-default `setUpClass`/`tearDownClass` hooks (such as `absltest.TestCase.setUpClass`
  in `absl-py`) and autouse class-scoped fixtures defined in this conftest (such as
  `restore_jax_prng_and_libtpu_env_for_class`), so stateless test classes like `TrainCompile` remain
  ungrouped and fan out freely across workers and shards.
  """
  cls = getattr(item, "cls", None)
  if cls is None:
    return False
  if issubclass(cls, unittest.TestCase):
    for base in cls.__mro__:
      if base is unittest.TestCase:
        break
      if (getattr(base, "__module__", None) or "") in _FRAMEWORK_TESTCASE_MODULES:
        continue
      if base.__dict__.get("setUpClass") is not None or base.__dict__.get("tearDownClass") is not None:
        return True
  elif callable(getattr(cls, "setup_class", None)) or callable(getattr(cls, "teardown_class", None)):
    return True
  fixtureinfo = getattr(item, "_fixtureinfo", None)
  name2fixturedefs = getattr(fixtureinfo, "name2fixturedefs", None) or {}
  for fixturedefs in name2fixturedefs.values():
    for fd in fixturedefs:
      mod = getattr(fd.func, "__module__", None) or ""
      # Ignore conftest's own autouse class fixtures and pytest's internal xunit wrappers
      # (e.g., `_pytest.unittest._make_xunit_fixture`, which pytest attaches to every
      # `unittest.TestCase` even when it has no custom `setUpClass`/`tearDownClass`).
      if fd.scope == "class" and mod != __name__ and not mod.startswith("_pytest."):
        return True
  return False


# tryfirst: run before plugins registered after conftest (notably pytest-xdist's WorkerInteractor,
# which reads `xdist_group` inside its own `pytest_collection_modifyitems` hook on xdist 3.x) and
# before pytest's `-m` marker deselection.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
  """Customize pytest collection behavior.

  - Skip hardware-specific tests when hardware is missing.
  - Deselect tests marked as external_serving/training in decoupled mode.
  - Mark remaining tests with the `decoupled` marker when running decoupled.
  - Mark tests without explicit hardware markers as `cpu_only`.
  - Auto-assign `xdist_group` to classes with `setUpClass` or test-defined class fixtures.
  """

  changed_tests = get_changed_tests()  # set[(file_path, test_name)]
  if changed_tests:
    for item in items:
      # Matches the def name too, so absl parameterized tests (collected as test_x_bf16 or test_x0) are marked.
      if is_changed_item(item, changed_tests):
        item.add_marker(pytest.mark.newly_added)

  decoupled = is_decoupled()
  remaining = []
  deselected = []

  skip_no_tpu_backend = None
  if not _has_tpu_backend_support():
    skip_no_tpu_backend = pytest.mark.skip(
        reason=(
            "Skipped: requires a TPU-enabled JAX install (TPU PJRT plugin). "
            "Install a TPU-enabled jax/jaxlib build to run this test."
        )
    )

  for item in items:
    cur_test_markers = {m.name for m in item.iter_markers()}

    if skip_no_tpu_backend and "tpu_backend" in cur_test_markers:
      item.add_marker(skip_no_tpu_backend)
      remaining.append(item)
      continue

    if decoupled and (cur_test_markers & GCP_MARKERS):
      # Deselect tests marked as external_serving/training entirely.
      deselected.append(item)
      continue

    remaining.append(item)

  # Update items in-place to only keep remaining tests.
  items[:] = remaining
  if deselected:
    config.hook.pytest_deselected(items=deselected)

  # Add decoupled marker to all remaining tests when running decoupled.
  if decoupled:
    for item in remaining:
      item.add_marker(pytest.mark.decoupled)

  # Auto-mark remaining tests as cpu_only unless explicitly marked with hardware target markers,
  # and keep classes with custom setUpClass / class-scoped fixtures on one xdist worker and shard.
  for item in remaining:
    cur_test_markers = {m.name for m in item.iter_markers()}
    if cur_test_markers.isdisjoint(HARDWARE_MARKERS):
      item.add_marker(pytest.mark.cpu_only)
    if "xdist_group" not in cur_test_markers and _has_shared_class_fixture(item):
      group_name = f"{item.cls.__module__}.{item.cls.__qualname__}".replace(".", "_")
      item.add_marker(pytest.mark.xdist_group(group_name))


def pytest_addoption(parser):
  """Registers the class-level shard split options; pass them as --opt=value (see tests/utils/class_split.py)."""
  class_split.pytest_addoption(parser)


def pytest_configure(config):
  """Registers custom pytest markers dynamically."""
  for m in [
      "gpu_only: tests that require GPU hardware",
      "tpu_only: tests that require TPU hardware",
      "cpu_only: tests that require CPU-only environment (skipped on active accelerator hardware)",
      "tpu_backend: tests that require a TPU-enabled JAX install (TPU PJRT plugin), but not TPU hardware",
      "external_serving: JetStream / serving / decode server components",
      "external_training: goodput integrations",
      "decoupled: marked on tests that are not skipped due to GCP deps, when DECOUPLE_GCLOUD=TRUE",
      "skip_on_tpu7x: skip test if running on TPU7x platform",
      "newly_added: newly introduced or modified tests in PRs, executed even if scheduled_only",
      "xdist_group: keep tests sharing class/module fixtures on one xdist worker and CI shard",
  ]:
    config.addinivalue_line("markers", m)
  class_split.pytest_configure(config)


@pytest.fixture(autouse=True, scope="module")
def clear_jax_compilation_cache():
  """Clears JAX in-memory compilation caches and forces GC when a worker leaves a test module.

  Runs on every backend. xdist workers are long-lived processes that interleave many modules, so
  this bounds how much jit/tracing cache a worker carries forward, at the cost of re-compiles when
  it returns to a module. Intra-module cache reuse is preserved; the persistent on-disk cache is
  not touched. This caps cache growth; it does not reclaim memory a worker holds for other reasons.
  """
  yield
  try:
    jax.clear_caches()
    gc.collect()
  except Exception:  # pylint: disable=broad-exception-caught
    pass


@pytest.fixture(autouse=True)
def restore_jax_remove_size_one_mesh_axis_from_type():
  """Saves and restores jax_remove_size_one_mesh_axis_from_type around each test item."""
  previous_flag = getattr(jax.config, "jax_remove_size_one_mesh_axis_from_type", None)
  yield
  if previous_flag is not None:
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", previous_flag)


@contextlib.contextmanager
def _restore_jax_prng_and_libtpu_env():
  """Saves and restores global JAX PRNG/Shardy/effort config and LIBTPU_INIT_ARGS across a scope."""
  previous_prng = getattr(jax.config, "jax_default_prng_impl", None)
  previous_shardy = getattr(jax.config, "jax_use_shardy_partitioner", None)
  previous_opt_level = getattr(jax.config, "jax_optimization_level", None)
  previous_mem_level = getattr(jax.config, "jax_memory_fitting_level", None)
  previous_libtpu = os.environ.get("LIBTPU_INIT_ARGS")
  try:
    yield
  finally:
    if previous_prng is not None and getattr(jax.config, "jax_default_prng_impl", None) != previous_prng:
      jax.config.update("jax_default_prng_impl", previous_prng)
    if previous_shardy is not None and getattr(jax.config, "jax_use_shardy_partitioner", None) != previous_shardy:
      jax.config.update("jax_use_shardy_partitioner", previous_shardy)
    if previous_opt_level is not None and getattr(jax.config, "jax_optimization_level", None) != previous_opt_level:
      jax.config.update("jax_optimization_level", previous_opt_level)
    if previous_mem_level is not None and getattr(jax.config, "jax_memory_fitting_level", None) != previous_mem_level:
      jax.config.update("jax_memory_fitting_level", previous_mem_level)
    if previous_libtpu is None:
      os.environ.pop("LIBTPU_INIT_ARGS", None)
    else:
      os.environ["LIBTPU_INIT_ARGS"] = previous_libtpu


@pytest.fixture(autouse=True, scope="class")
def restore_jax_prng_and_libtpu_env_for_class():
  """Restores JAX PRNG config and LIBTPU_INIT_ARGS after each test class, covering setUpClass()."""
  with _restore_jax_prng_and_libtpu_env():
    yield


@pytest.fixture(autouse=True)
def restore_jax_prng_and_libtpu_env():
  """Restores JAX PRNG config and LIBTPU_INIT_ARGS after each test item, covering setUp() and tests."""
  with _restore_jax_prng_and_libtpu_env():
    yield


@pytest.fixture(autouse=True)
def handle_skip_on_tpu7x(request):
  """Dynamically skip tests marked with skip_on_tpu7x if running on TPU7x."""
  if request.node.get_closest_marker("skip_on_tpu7x"):
    try:
      is_tpu7x = any("TPU7x" in d.device_kind for d in jax.devices())
    except Exception:  # pylint: disable=broad-exception-caught
      is_tpu7x = False
    if is_tpu7x:
      pytest.skip("Skipped: test is marked skip_on_tpu7x (unsupported or under investigation on TPU7x)")


@pytest.fixture(autouse=True)
def handle_cpu_only(request):
  """Dynamically skip cpu_only tests on TPU or GPU hardware."""
  if request.node.get_closest_marker("cpu_only"):
    try:
      has_accelerator = any(d.platform in ("tpu", "gpu") for d in jax.devices())
    except Exception:  # pylint: disable=broad-exception-caught
      has_accelerator = False
    if has_accelerator:
      pytest.skip("Skipped: cpu_only test bypassed on hardware accelerator testbeds")


@pytest.fixture(autouse=True)
def handle_tpu_only(request):
  """Dynamically skip tpu_only tests if running on non-TPU hardware."""
  if request.node.get_closest_marker("tpu_only"):
    try:
      has_tpu = any(d.platform == "tpu" for d in jax.devices())
    except Exception:  # pylint: disable=broad-exception-caught
      has_tpu = False
    if not has_tpu:
      pytest.skip("Skipped: requires TPU hardware, none detected")


@pytest.fixture(autouse=True)
def handle_gpu_only(request):
  """Dynamically skip gpu_only tests if running on non-GPU hardware."""
  if request.node.get_closest_marker("gpu_only"):
    try:
      has_gpu = any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # pylint: disable=broad-exception-caught
      has_gpu = False
    if not has_gpu:
      pytest.skip("Skipped: requires GPU hardware, none detected")


@contextlib.contextmanager
def _tpu_test_compiler_defaults():
  """Applies fast XLA:TPU compilation effort defaults for TPU test sessions and restores on exit.

  `O0` optimization level skips latency-hiding/async-op scheduling, MSA, collective pipelining,
  and narrows HLO fixpoint iterations and lowering VMEM budget, cutting TPU test compile time
  without changing numerical semantics or synchronous collective presence in HLO.
  `O1` memory fitting keeps post-fusion rematerialization enabled so larger TPU tests do not OOM.

  Only active inside pytest on TPU hardware (never production runs or CPU/GPU tests), skipped if
  the caller already set `JAX_OPTIMIZATION_LEVEL` / `JAX_MEMORY_FITTING_LEVEL`, and overridable
  per test via `jax.config.update("jax_optimization_level", "O2")` (auto-restored on teardown).
  """
  previous_opt_level = None
  previous_mem_level = None
  if os.getenv("JAX_PLATFORMS", "").lower() != "cpu" and os.getenv("INPUTS_DEVICE_TYPE", "").lower() != "cpu":
    try:
      has_tpu = any(d.platform == "tpu" for d in jax.devices())
    except Exception:  # pylint: disable=broad-exception-caught
      has_tpu = False
    if has_tpu:
      if not os.getenv("JAX_OPTIMIZATION_LEVEL") and hasattr(jax.config, "jax_optimization_level"):
        previous_opt_level = jax.config.jax_optimization_level
        jax.config.update("jax_optimization_level", "O0")
      if not os.getenv("JAX_MEMORY_FITTING_LEVEL") and hasattr(jax.config, "jax_memory_fitting_level"):
        previous_mem_level = jax.config.jax_memory_fitting_level
        jax.config.update("jax_memory_fitting_level", "O1")
  try:
    yield
  finally:
    if previous_opt_level is not None:
      jax.config.update("jax_optimization_level", previous_opt_level)
    if previous_mem_level is not None:
      jax.config.update("jax_memory_fitting_level", previous_mem_level)


@pytest.fixture(autouse=True, scope="session")
def _configure_tpu_test_compiler_defaults():
  """Session-scoped fixture enabling fast XLA:TPU effort levels during pytest runs on TPU."""
  with _tpu_test_compiler_defaults():
    yield


def pytest_runtest_setup(item):
  """Hook to inject markers as properties into the test item."""
  for marker in item.iter_markers():
    item.user_properties.append(("marker", marker.name))
