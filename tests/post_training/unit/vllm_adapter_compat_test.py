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

"""Unit tests for the JAX compatibility patches applied by the MaxText vLLM adapter."""

import unittest
from unittest import mock

import pytest

pytest.importorskip("vllm")
pytest.importorskip("tpu_inference")

pytestmark = pytest.mark.post_training

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np
from maxtext.integration.vllm import maxtext_vllm_adapter
from maxtext.integration.vllm.maxtext_vllm_adapter import adapter

_MISSING = object()


def _fake_reciprocal(x, *, approx=False):
  del approx
  return 1.0 / x


class PatchPallasTpuReciprocalTest(unittest.TestCase):
  """Tests for `patch_pallas_tpu_reciprocal`."""

  def setUp(self):
    super().setUp()
    # The patch mutates the shared `jax.experimental.pallas.tpu` module, so restore it after each test.
    original = vars(pltpu).get("reciprocal", _MISSING)

    def restore():
      if original is _MISSING:
        vars(pltpu).pop("reciprocal", None)
      else:
        pltpu.reciprocal = original

    self.addCleanup(restore)

  def test_aliases_pl_reciprocal_when_pltpu_lacks_it(self):
    """JAX < 0.11.2 only exports `reciprocal` from `jax.experimental.pallas`."""
    vars(pltpu).pop("reciprocal", None)
    with mock.patch.object(pl, "reciprocal", _fake_reciprocal, create=True):
      adapter.patch_pallas_tpu_reciprocal()
    self.assertIs(pltpu.reciprocal, _fake_reciprocal)

  def test_keeps_existing_pltpu_reciprocal(self):
    """JAX >= 0.11.2 already exports `pltpu.reciprocal`, which must be left untouched."""
    existing = object()
    pltpu.reciprocal = existing
    adapter.patch_pallas_tpu_reciprocal()
    self.assertIs(pltpu.reciprocal, existing)

  def test_reciprocal_runs_in_pallas_kernel(self):
    """`pltpu.reciprocal(x, approx=True)`, as called by tpu-inference kernels, traces and runs."""
    adapter.patch_pallas_tpu_reciprocal()

    def kernel(x_ref, o_ref):
      o_ref[...] = pltpu.reciprocal(x_ref[...], approx=True)

    x = jnp.full((8, 128), 4.0, dtype=jnp.float32)
    out = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), interpret=True)(x)
    np.testing.assert_allclose(np.asarray(out), 0.25, rtol=1e-2)


class RegisterTest(unittest.TestCase):
  """Tests for `register`."""

  def test_register_applies_reciprocal_patch(self):
    with (
        mock.patch.object(maxtext_vllm_adapter, "get_multimodal_handler", return_value=None),
        mock.patch.object(maxtext_vllm_adapter, "register_model"),
        mock.patch.object(maxtext_vllm_adapter.MaxTextForCausalLM, "supports_multimodal", False),
        mock.patch.object(adapter, "patch_kv_cache_manager"),
        mock.patch.object(adapter, "patch_pallas_tpu_reciprocal") as mock_patch_reciprocal,
    ):
      maxtext_vllm_adapter.register()
    mock_patch_reciprocal.assert_called_once_with()


if __name__ == "__main__":
  unittest.main()
