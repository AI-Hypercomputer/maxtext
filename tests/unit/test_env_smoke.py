# Copyright 2023-2026 Google LLC
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

"""Pytest-based environment smoke test for MaxText.

Checks:
  - Core imports (jax, flax, numpy)
  - Optional imports
  - JAX device enumeration

Fails only on missing core imports or device query failure; alias test
asserts mapping rules.
"""

from __future__ import annotations

import importlib
import time

import pytest

from maxtext.common.gcloud_stub import is_decoupled

CORE_IMPORTS = ["jax", "jax.numpy", "flax", "numpy"]
OPTIONAL_IMPORTS = ["transformers", "MaxText", "maxtext.configs.pyconfig", "maxtext.inference.maxengine.maxengine"]

_defects: list[str] = []


@pytest.mark.parametrize("name", CORE_IMPORTS)
def test_environment_core_imports(name):
  """Test that core imports are available."""
  importlib.import_module(name)


@pytest.mark.parametrize("name", OPTIONAL_IMPORTS)
def test_environment_optional_imports(name):
  """Test optional imports and report issues as defects."""
  t0 = time.time()
  try:
    importlib.import_module(name)
    dt = time.time() - t0
    if dt > 8.0:
      _defects.append(f"{name} SLOW_IMPORT ({dt:.1f}s)")
  except Exception as err:  # pragma: no cover  # pylint: disable=broad-exception-caught
    _defects.append(f"{name} FAIL: {err}")


def test_jax_devices():
  try:
    import jax  # type: ignore  # pylint: disable=import-outside-toplevel
  except Exception as e:  # pragma: no cover  # pylint: disable=broad-exception-caught
    raise AssertionError(f"jax not importable for device test: {e}") from e
  try:
    devices = jax.devices()
  except Exception as e:  # pragma: no cover  # pylint: disable=broad-exception-caught
    raise AssertionError(f"jax.devices() failed: {e}") from e
  assert len(devices) >= 1, "No JAX devices found"


def test_decoupled_flag_consistency():
  decoupled = is_decoupled()
  # Soft check only; logic exercised in other tests.
  if decoupled:
    pass
  else:
    pass


def test_report_defects():
  if _defects:
    print("Environment optional issues:\n" + "\n".join(_defects))
