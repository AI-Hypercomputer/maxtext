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

"""Tests for moe_accumulate_chunk_wgrad config validation and the gmm return_rhs backend guard."""

import unittest

import jax.numpy as jnp

from maxtext.configs import pyconfig
from maxtext.kernels.megablox import ops as mblx
from tests.utils.test_helpers import get_test_config_path


class AccumulateChunkWgradConfigTest(unittest.TestCase):
  """The flag's default and its validation."""

  def _init(self, **kw):
    return pyconfig.initialize(
        [None, get_test_config_path()], run_name="accumulate_chunk_wgrad_test", enable_checkpointing=False, **kw
    )

  def test_default_off(self):
    self.assertFalse(self._init().moe_accumulate_chunk_wgrad)

  def test_accepts_tokamax_gmm_v2(self):
    self.assertTrue(
        self._init(moe_accumulate_chunk_wgrad=True, use_tokamax_gmm=True, use_gmm_v2=True).moe_accumulate_chunk_wgrad
    )

  def test_rejects_unsupported_configs(self):
    for kw in (
        {"use_tokamax_gmm": False, "use_gmm_v2": False},
        {"use_tokamax_gmm": True, "use_gmm_v2": False},
        {"use_tokamax_gmm": True, "use_gmm_v2": True, "prefuse_moe_weights": True},
    ):
      with self.subTest(**kw):
        with self.assertRaisesRegex(ValueError, "moe_accumulate_chunk_wgrad=True requires"):
          self._init(moe_accumulate_chunk_wgrad=True, **kw)


class GmmReturnRhsTest(unittest.TestCase):
  """return_rhs is only supported on the tokamax gmm_v2 backend."""

  def test_rejects_non_gmm_v2_backend(self):
    lhs = jnp.ones((8, 4), jnp.float32)
    rhs = jnp.ones((2, 4, 4), jnp.float32)
    group_sizes = jnp.array([4, 4], jnp.int32)
    with self.assertRaisesRegex(ValueError, "return_rhs=True requires"):
      mblx.gmm(lhs, rhs, group_sizes, use_tokamax_backend=True, use_gmm_v2=False, return_rhs=True)


if __name__ == "__main__":
  unittest.main()
