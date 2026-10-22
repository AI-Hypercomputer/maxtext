# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the diffusion pipeline helpers (`pipeline_utils.py`, `wan_pipeline.py`)."""

import unittest

import pytest

pytest.importorskip("diffusers")  # the pipelines need diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.inference.diffuser_pipelines import pipeline_utils
from maxtext.inference.diffuser_pipelines import wan_pipeline
from maxtext.m3.models.wan import WanModel

# pylint: enable=wrong-import-position

_WAN_CONFIG = {"num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 8}


class PipelineUtilsTest(unittest.TestCase):

  def test_to_int_path(self):
    self.assertEqual(pipeline_utils._to_int_path(("blocks", "0", "attn", "1")), ("blocks", 0, "attn", 1))  # pylint: disable=protected-access

  def test_restored_component(self):
    restored = {
        "transformer_state": {"w": 1},
        "transformer_config": dict(_WAN_CONFIG, _class_name="WanTransformer3DModel"),
        "vae_state": {"w": 2},
    }
    state, config = pipeline_utils._restored_component(restored, "transformer", WanModel)  # pylint: disable=protected-access
    self.assertEqual(state, {"w": 1})
    self.assertEqual(config, _WAN_CONFIG)  # filtered to WanModel.__init__ arguments

  def test_restored_component_legacy_and_missing(self):
    legacy = {"wan_state": {"w": 1}, "wan_config": _WAN_CONFIG}
    state, config = pipeline_utils._restored_component(legacy, "transformer", WanModel)  # pylint: disable=protected-access
    self.assertEqual((state, config), ({"w": 1}, _WAN_CONFIG))
    with self.assertRaisesRegex(ValueError, "no `vae_state`"):
      pipeline_utils._restored_component(legacy, "vae", WanModel)  # pylint: disable=protected-access

  def test_numpy_to_pil(self):
    images = pipeline_utils.DiffusionPipeline.numpy_to_pil(np.zeros((2, 8, 6, 3), np.float32))
    self.assertEqual(len(images), 2)
    self.assertEqual(images[0].size, (6, 8))


class WanPipelineHelpersTest(unittest.TestCase):

  def test_prompt_clean(self):
    self.assertEqual(wan_pipeline.prompt_clean("  a  cat &amp;\n a dog  "), "a cat & a dog")

  def test_cast_with_exclusion_keeps_norms_in_fp32(self):
    params = {"blocks": {"norm2": {"scale": jnp.ones(2)}, "attn1": {"kernel": jnp.ones(2)}}}
    cast = jax.tree_util.tree_map_with_path(lambda p, x: wan_pipeline.cast_with_exclusion(p, x, jnp.bfloat16), params)
    self.assertEqual(cast["blocks"]["norm2"]["scale"].dtype, jnp.float32)
    self.assertEqual(cast["blocks"]["attn1"]["kernel"].dtype, jnp.bfloat16)

  def test_model_init_config_for_wan(self):
    raw = dict(_WAN_CONFIG, scan_layers=True, _diffusers_version="0.33.0", foo=1)
    self.assertEqual(wan_pipeline.model_init_config_for_wan(raw), _WAN_CONFIG)

  def test_nearest_interp(self):
    src = np.arange(10.0)
    np.testing.assert_array_equal(wan_pipeline.nearest_interp(src, 1), [9.0])
    np.testing.assert_array_equal(wan_pipeline.nearest_interp(src, 3), [0.0, 4.0, 9.0])

  def test_init_magcache_interpolates_ratios(self):
    base = np.linspace(1.0, 0.5, 20)  # (cond, uncond) pairs for 10 steps
    *_, skip_warmup, mag_ratios = wan_pipeline.init_magcache(5, 0.2, base)
    self.assertEqual(skip_warmup, 1)
    self.assertEqual(mag_ratios.shape, (10,))  # resampled to 5 (cond, uncond) pairs
    *_, mag_ratios = wan_pipeline.init_magcache(10, 0.2, base)
    np.testing.assert_array_equal(mag_ratios, base)  # already the right length


if __name__ == "__main__":
  unittest.main()
