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

"""Drift guard tests comparing vendored functions across maxtext and tunix."""

import inspect
import os
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

try:
  from maxtext.integration.vllm import convert_utils as maxtext_convert_utils
  from maxtext.integration.vllm.moe_padding import TPU_V5P_SUBCORE_LANE_SIZE as MAXTEXT_LANE_SIZE
except ModuleNotFoundError:
  from maxtext.src.maxtext.integration.vllm import convert_utils as maxtext_convert_utils
  from maxtext.src.maxtext.integration.vllm.moe_padding import TPU_V5P_SUBCORE_LANE_SIZE as MAXTEXT_LANE_SIZE
from tunix.experimental.weight_sync import raiden_synchronizer
import tunix.generate.utils as tunix_gen_utils

pytestmark = [pytest.mark.post_training]


class CrossRepoDriftTest(unittest.TestCase):
  """Verify vendored MoE weight interleaving functions match across repositories."""

  @pytest.mark.cpu_only
  def test_constants_match(self):
    self.assertEqual(MAXTEXT_LANE_SIZE, 128)
    self.assertTrue(hasattr(tunix_gen_utils, "TPU_V5P_SUBCORE_LANE_SIZE"))
    self.assertEqual(
        tunix_gen_utils.TPU_V5P_SUBCORE_LANE_SIZE,
        MAXTEXT_LANE_SIZE,
    )

  @pytest.mark.cpu_only
  def test_interleave_moe_weights_signatures_match(self):
    maxtext_fn = maxtext_convert_utils._interleave_moe_weights
    tunix_fn = tunix_gen_utils._interleave_moe_weights

    # Inspect the underlying functions (unwrap any jax.jit decorators)
    maxtext_unwrapped = getattr(maxtext_fn, "__wrapped__", maxtext_fn)
    tunix_unwrapped = getattr(tunix_fn, "__wrapped__", tunix_fn)

    maxtext_sig = inspect.signature(maxtext_unwrapped)
    tunix_sig = inspect.signature(tunix_unwrapped)

    self.assertEqual(
        [(p.name, p.default) for p in maxtext_sig.parameters.values()],
        [(p.name, p.default) for p in tunix_sig.parameters.values()],
    )

  @pytest.mark.cpu_only
  def test_interleave_moe_weights_outputs_match_interleaved(self):
    wi_0 = jnp.arange(256, dtype=jnp.float32).reshape(1, 1, 256)
    wi_1 = jnp.arange(256, 512, dtype=jnp.float32).reshape(1, 1, 256)
    tgt_shape = (1, 1, 512)

    maxtext_out = maxtext_convert_utils._interleave_moe_weights(wi_0, wi_1, tgt_shape=tgt_shape, n_shards=2, axis=2)
    tunix_out = tunix_gen_utils._interleave_moe_weights(wi_0, wi_1, tgt_shape=tgt_shape, n_shards=2, axis=2)

    np.testing.assert_array_equal(np.array(maxtext_out), np.array(tunix_out))

  @pytest.mark.cpu_only
  def test_interleave_moe_weights_outputs_match_fallback(self):
    wi_0 = jnp.arange(200, dtype=jnp.float32).reshape(1, 1, 200)
    wi_1 = jnp.arange(200, 400, dtype=jnp.float32).reshape(1, 1, 200)
    tgt_shape = (1, 1, 400)

    maxtext_out = maxtext_convert_utils._interleave_moe_weights(wi_0, wi_1, tgt_shape=tgt_shape, n_shards=4, axis=2)
    tunix_out = tunix_gen_utils._interleave_moe_weights(wi_0, wi_1, tgt_shape=tgt_shape, n_shards=4, axis=2)

    np.testing.assert_array_equal(np.array(maxtext_out), np.array(tunix_out))

  @pytest.mark.cpu_only
  def test_raiden_ffi_resolution_matrix(self):
    # Table-driven over:
    # (is_proxy, env_val, wheel_present, host_stage_in, exp_ffi, exp_hs)
    cases = [
        # Under proxy with wheel present:
        (True, None, True, None, True, False),
        (True, "1", True, None, True, False),
        (True, "true", True, False, True, False),
        (True, "yes", True, True, True, False),
        (True, "0", True, None, False, True),
        (True, "false", True, False, False, True),  # explicit host_stage=False overridden to True
        (True, "no", True, True, False, True),
        # Under proxy without wheel:
        (True, None, False, None, False, True),
        (True, "1", False, False, False, True),
        (True, "true", False, None, False, True),
        (True, "0", False, None, False, True),
        # Non-proxy with wheel present:
        (False, None, True, None, False, False),
        (False, None, True, True, False, True),
        (False, "1", True, None, True, False),
        (False, "true", True, False, True, False),
        (False, "0", True, None, False, False),
        # Non-proxy without wheel:
        (False, "1", False, None, False, False),
    ]

    for is_proxy, env_val, wheel_present, hs_in, exp_ffi, exp_hs in cases:
      env_dict = {}
      if env_val is not None:
        env_dict["RAIDEN_USE_FFI"] = env_val
      fake_wheel = mock.MagicMock() if wheel_present else None

      with (
          mock.patch.dict("os.environ", env_dict),
          mock.patch.object(raiden_synchronizer, "_get_raiden_ffi", return_value=fake_wheel),
      ):
        if env_val is None:
          os.environ.pop("RAIDEN_USE_FFI", None)
        tunix_ffi = raiden_synchronizer.resolve_use_ffi(None, is_proxy=is_proxy)
        tunix_hs = raiden_synchronizer.normalize_host_stage(hs_in, use_ffi=tunix_ffi, is_proxy=is_proxy)

        self.assertEqual(
            tunix_ffi,
            exp_ffi,
            f"use_ffi mismatch for is_proxy={is_proxy}, env={env_val}, wheel={wheel_present}",
        )
        self.assertEqual(
            tunix_hs,
            exp_hs,
            f"host_stage mismatch for is_proxy={is_proxy}, env={env_val}, wheel={wheel_present}, hs_in={hs_in}",
        )

        if is_proxy:
          self.assertFalse(
              tunix_ffi is False and tunix_hs is False,
              "Under proxy, both use_ffi and host_stage cannot be False!",
          )


if __name__ == "__main__":
  unittest.main()
