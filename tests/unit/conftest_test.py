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

"""Unit tests for TPU compiler defaults and class-fixture grouping in `tests/conftest.py`."""

import contextlib
import io
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import jax
import pytest

from tests.conftest import (
    _has_shared_class_fixture,
    _restore_jax_prng_and_libtpu_env,
    _tpu_test_compiler_defaults,
)


class ConftestTpuCompilerDefaultsTest(unittest.TestCase):
  """Tests `_tpu_test_compiler_defaults` and effort-level restoration in `tests/conftest.py`."""

  def _make_fake_config(self, opt_level="O2", mem_level="O2"):
    """Returns a minimal stand-in for `jax.config` with update tracking."""
    cfg = types.SimpleNamespace(
        jax_default_prng_impl="threefry2x32",
        jax_use_shardy_partitioner=True,
        jax_optimization_level=opt_level,
        jax_memory_fitting_level=mem_level,
    )
    cfg.update = lambda key, val: setattr(cfg, key, val)
    return cfg

  def test_tpu_sets_o0_and_o1_and_restores_on_teardown(self):
    fake_cfg = self._make_fake_config(opt_level="O2", mem_level="O2")
    tpu_dev = types.SimpleNamespace(platform="tpu")
    env = {"JAX_PLATFORMS": "", "INPUTS_DEVICE_TYPE": ""}
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(jax, "config", fake_cfg),
        mock.patch.object(jax, "devices", return_value=[tpu_dev]),
    ):
      os.environ.pop("JAX_OPTIMIZATION_LEVEL", None)
      os.environ.pop("JAX_MEMORY_FITTING_LEVEL", None)
      with _tpu_test_compiler_defaults():
        self.assertEqual(fake_cfg.jax_optimization_level, "O0")
        self.assertEqual(fake_cfg.jax_memory_fitting_level, "O1")
    self.assertEqual(fake_cfg.jax_optimization_level, "O2")
    self.assertEqual(fake_cfg.jax_memory_fitting_level, "O2")

  def test_skips_cpu_and_gpu_and_still_yields(self):
    cases = [
        (
            "jax_platforms_cpu",
            {"JAX_PLATFORMS": "cpu", "INPUTS_DEVICE_TYPE": ""},
            [types.SimpleNamespace(platform="tpu")],
        ),
        ("device_type_cpu", {"JAX_PLATFORMS": "", "INPUTS_DEVICE_TYPE": "cpu"}, [types.SimpleNamespace(platform="tpu")]),
        ("gpu_only", {"JAX_PLATFORMS": "", "INPUTS_DEVICE_TYPE": ""}, [types.SimpleNamespace(platform="gpu")]),
    ]
    for name, env, devices in cases:
      with self.subTest(case=name):
        fake_cfg = self._make_fake_config(opt_level="O2", mem_level="O2")
        yielded = False
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(jax, "config", fake_cfg),
            mock.patch.object(jax, "devices", return_value=devices),
        ):
          os.environ.pop("JAX_OPTIMIZATION_LEVEL", None)
          os.environ.pop("JAX_MEMORY_FITTING_LEVEL", None)
          with _tpu_test_compiler_defaults():
            yielded = True
            self.assertEqual(fake_cfg.jax_optimization_level, "O2")
            self.assertEqual(fake_cfg.jax_memory_fitting_level, "O2")
        self.assertTrue(yielded)

  def test_respects_caller_env_overrides(self):
    fake_cfg = self._make_fake_config(opt_level="O2", mem_level="O2")
    tpu_dev = types.SimpleNamespace(platform="tpu")
    env = {
        "JAX_PLATFORMS": "",
        "INPUTS_DEVICE_TYPE": "",
        "JAX_OPTIMIZATION_LEVEL": "O2",
        "JAX_MEMORY_FITTING_LEVEL": "O2",
    }
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(jax, "config", fake_cfg),
        mock.patch.object(jax, "devices", return_value=[tpu_dev]),
    ):
      with _tpu_test_compiler_defaults():
        self.assertEqual(fake_cfg.jax_optimization_level, "O2")
        self.assertEqual(fake_cfg.jax_memory_fitting_level, "O2")

  def test_restore_context_restores_per_test_effort_overrides(self):
    fake_cfg = self._make_fake_config(opt_level="O0", mem_level="O1")
    with mock.patch.object(jax, "config", fake_cfg):
      with _restore_jax_prng_and_libtpu_env():
        fake_cfg.update("jax_optimization_level", "O2")
        fake_cfg.update("jax_memory_fitting_level", "O2")
    self.assertEqual(fake_cfg.jax_optimization_level, "O0")
    self.assertEqual(fake_cfg.jax_memory_fitting_level, "O1")

  def test_pytest_session_runs_conftest_fixtures_cleanly_on_cpu(self):
    """End-to-end check that the autouse session fixture yields cleanly under pytest on CPU."""
    conftest_mod = sys.modules[_has_shared_class_fixture.__module__]
    with tempfile.TemporaryDirectory() as tmpdir:
      workdir = pathlib.Path(tmpdir).resolve()
      (workdir / "smoke_test.py").write_text("def test_ok():\n  assert True\n", encoding="utf-8")
      out = io.StringIO()
      err = io.StringIO()
      sys.modules.pop("smoke_test", None)
      try:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "JAX_PLATFORMS": "cpu",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTEST_ADDOPTS": "",
                },
            ),
            contextlib.chdir(workdir),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
          returncode = pytest.main(
              ["-q", "-p", "no:cacheprovider", f"--rootdir={workdir}", "smoke_test.py"],
              plugins=[conftest_mod],
          )
      finally:
        sys.modules.pop("smoke_test", None)
    self.assertEqual(int(returncode), 0, out.getvalue() + err.getvalue())


class ConftestSharedClassFixtureTest(unittest.TestCase):
  """Tests `_has_shared_class_fixture` detection for `setUpClass` and `tearDownClass`."""

  def test_detects_setup_and_teardown_class_hooks(self):
    class PlainCase(unittest.TestCase):
      pass

    class SetUpClassCase(unittest.TestCase):

      @classmethod
      def setUpClass(cls):
        super().setUpClass()
        cls.initialized = True

    class TearDownClassCase(unittest.TestCase):

      @classmethod
      def tearDownClass(cls):
        cls.cleaned = True
        super().tearDownClass()

    class PlainPytestClass:
      pass

    class TeardownPytestClass:

      @classmethod
      def teardown_class(cls):
        pass

    self.assertFalse(_has_shared_class_fixture(types.SimpleNamespace(cls=PlainCase)))
    self.assertTrue(_has_shared_class_fixture(types.SimpleNamespace(cls=SetUpClassCase)))
    self.assertTrue(_has_shared_class_fixture(types.SimpleNamespace(cls=TearDownClassCase)))
    self.assertFalse(_has_shared_class_fixture(types.SimpleNamespace(cls=PlainPytestClass)))
    self.assertTrue(_has_shared_class_fixture(types.SimpleNamespace(cls=TeardownPytestClass)))

  def test_auto_xdist_group_runs_before_late_registered_xdist_worker_hook(self):
    """Verifies conftest marks `xdist_group` before plugins registered after conftest (like xdist 3.x)."""
    conftest_mod = sys.modules[_has_shared_class_fixture.__module__]
    observed_groups = {}

    class _FakeXdistWorkerPlugin:
      """Mimics pytest-xdist 3.x WorkerInteractor registered after conftest loading."""

      def pytest_collection_modifyitems(self, config, items):
        del config
        for item in items:
          mark = item.get_closest_marker("xdist_group")
          observed_groups[item.nodeid] = mark.args[0] if (mark and mark.args) else None

    sample = (
        "import unittest\n"
        "import pytest\n"
        "class GroupedCase(unittest.TestCase):\n"
        "  @classmethod\n"
        "  def setUpClass(cls):\n"
        "    super().setUpClass()\n"
        "  def test_a(self): pass\n"
        "  def test_b(self): pass\n"
        "class StatelessCase(unittest.TestCase):\n"
        "  def test_c(self): pass\n"
        "class TestFixtureGroupedPytestCase:\n"
        "  @pytest.fixture(scope='class', autouse=True)\n"
        "  def shared(self): yield\n"
        "  def test_d(self): pass\n"
        "class TestStatelessPytestCase:\n"
        "  def test_e(self): pass\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      workdir = pathlib.Path(tmpdir).resolve()
      (workdir / "group_order_test.py").write_text(sample, encoding="utf-8")
      out = io.StringIO()
      err = io.StringIO()
      sys.modules.pop("group_order_test", None)
      try:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "JAX_PLATFORMS": "cpu",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTEST_ADDOPTS": "",
                },
            ),
            contextlib.chdir(workdir),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
          # Register conftest_mod first and _FakeXdistWorkerPlugin second (the exact order in
          # pytest-xdist 3.x `remote.py`, where `_prepareconfig` loads conftest before registering
          # `WorkerInteractor`). Without `tryfirst=True`, pluggy's LIFO order runs the worker
          # plugin first and misses every auto-added `xdist_group` marker.
          returncode = pytest.main(
              ["--collect-only", "-q", "-p", "no:cacheprovider", f"--rootdir={workdir}", "group_order_test.py"],
              plugins=[conftest_mod, _FakeXdistWorkerPlugin()],
          )
      finally:
        sys.modules.pop("group_order_test", None)
    self.assertEqual(int(returncode), 0, out.getvalue() + err.getvalue())
    self.assertEqual(
        observed_groups["group_order_test.py::GroupedCase::test_a"],
        "group_order_test_GroupedCase",
    )
    self.assertEqual(
        observed_groups["group_order_test.py::GroupedCase::test_b"],
        "group_order_test_GroupedCase",
    )
    self.assertIsNone(observed_groups["group_order_test.py::StatelessCase::test_c"])
    self.assertEqual(
        observed_groups["group_order_test.py::TestFixtureGroupedPytestCase::test_d"],
        "group_order_test_TestFixtureGroupedPytestCase",
    )
    self.assertIsNone(observed_groups["group_order_test.py::TestStatelessPytestCase::test_e"])


if __name__ == "__main__":
  unittest.main()
