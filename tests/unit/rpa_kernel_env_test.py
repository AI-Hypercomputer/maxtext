# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that `attention` drives tpu-inference's RPA kernel even after it was imported."""

import os
import sys
import types
import unittest
from unittest import mock

from maxtext.integration.vllm import rpa_kernel_env

_AI = "tpu_inference.layers.common.attention_interface"
_LONG_CTX = "tpu_inference.kernels.experimental.batched_rpa_long_ctx.wrapper"
_BATCHED = "tpu_inference.kernels.experimental.batched_rpa.wrapper"


def _fake_kernel(name):
  module = types.ModuleType(name)
  module.ragged_paged_attention = lambda *a, **k: name
  module.get_kv_cache_shape = lambda *a, **k: name
  return module


class SelectRpaKernelTest(unittest.TestCase):
  """select_rpa_kernel sets the env var and repairs an import-time binding."""

  def setUp(self):
    super().setUp()
    self.v3 = _fake_kernel("v3")
    self.long_ctx = _fake_kernel(_LONG_CTX)
    self.batched = _fake_kernel(_BATCHED)
    # Mimic an older tpu-inference: attention_interface already imported and
    # bound to the v3 kernel because the env var was unset at import time.
    self.attention_interface = types.ModuleType(_AI)
    self.attention_interface.ragged_paged_attention = self.v3.ragged_paged_attention
    self.attention_interface.get_kv_cache_shape = self.v3.get_kv_cache_shape
    self.enterContext(
        mock.patch.dict(
            sys.modules,
            {_AI: self.attention_interface, _LONG_CTX: self.long_ctx, _BATCHED: self.batched},
        )
    )
    self.enterContext(mock.patch.dict(os.environ, {}, clear=False))
    os.environ.pop("USE_BATCHED_RPA_LONG_CTX_KERNEL", None)
    os.environ.pop("USE_BATCHED_RPA_KERNEL", None)

  def test_long_ctx_sets_env_and_rebinds_imported_module(self):
    self.assertEqual(rpa_kernel_env.select_rpa_kernel("vllm_batched_rpa_long_ctx"), "USE_BATCHED_RPA_LONG_CTX_KERNEL")
    self.assertEqual(os.environ["USE_BATCHED_RPA_LONG_CTX_KERNEL"], "1")
    self.assertIs(self.attention_interface.ragged_paged_attention, self.long_ctx.ragged_paged_attention)
    self.assertIs(self.attention_interface.get_kv_cache_shape, self.long_ctx.get_kv_cache_shape)

  def test_batched_rpa_and_legacy_override(self):
    self.assertEqual(rpa_kernel_env.select_rpa_kernel("vllm_batched_rpa"), "USE_BATCHED_RPA_KERNEL")
    self.assertIs(self.attention_interface.ragged_paged_attention, self.batched.ragged_paged_attention)
    self.assertEqual(rpa_kernel_env.select_rpa_kernel("vllm_rpa", use_batched_rpa=True), "USE_BATCHED_RPA_KERNEL")

  def test_vllm_rpa_leaves_environment_and_binding_alone(self):
    self.assertIsNone(rpa_kernel_env.select_rpa_kernel("vllm_rpa"))
    self.assertIsNone(rpa_kernel_env.select_rpa_kernel(None))
    self.assertNotIn("USE_BATCHED_RPA_LONG_CTX_KERNEL", os.environ)
    self.assertNotIn("USE_BATCHED_RPA_KERNEL", os.environ)
    self.assertIs(self.attention_interface.ragged_paged_attention, self.v3.ragged_paged_attention)

  def test_call_time_dispatching_tpu_inference_is_not_touched(self):
    # Newer tpu-inference exposes rpa_kernel_module and resolves the env var on
    # every call; its dispatchers must not be replaced.
    self.attention_interface.rpa_kernel_module = lambda: self.v3
    dispatcher = self.attention_interface.ragged_paged_attention
    rpa_kernel_env.select_rpa_kernel("vllm_batched_rpa_long_ctx")
    self.assertEqual(os.environ["USE_BATCHED_RPA_LONG_CTX_KERNEL"], "1")
    self.assertIs(self.attention_interface.ragged_paged_attention, dispatcher)

  def test_not_yet_imported_module_is_left_to_import(self):
    del sys.modules[_AI]
    rpa_kernel_env.select_rpa_kernel("vllm_batched_rpa_long_ctx")
    self.assertEqual(os.environ["USE_BATCHED_RPA_LONG_CTX_KERNEL"], "1")
    self.assertNotIn(_AI, sys.modules)


if __name__ == "__main__":
  unittest.main()
