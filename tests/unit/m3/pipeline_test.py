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

import os
import types
import unittest
from unittest import mock

import pytest

pytest.importorskip("diffusers")  # the pipelines need diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.inference import sample
from maxtext.m3.configs import config as config_lib
from maxtext.m3.diffuser_pipelines import pipeline_utils
from maxtext.m3.diffuser_pipelines import wan_pipeline
from maxtext.m3.models.wan import WanModel

# pylint: enable=wrong-import-position

_WAN_CONFIG = {"num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 8}
_WAN_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "maxtext", "m3", "models", "wan", "configs"
)


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

  def test_get_pipeline_class(self):
    class _Cfg:

      def __init__(self, model_name, model_type="T2V", supported_types=()):
        self.model_name = model_name
        self.model_type = model_type
        self.supported_types = list(supported_types)

    self.assertIs(sample.get_pipeline_class(_Cfg("wan2.1")), wan_pipeline.WanPipeline2_1)
    self.assertIs(sample.get_pipeline_class(_Cfg("wan2.2", "T2V", ["T2V"])), wan_pipeline.WanPipeline2_1)
    self.assertIs(sample.get_pipeline_class(_Cfg("wan2.1", "I2V", ["T2V", "I2V"])), wan_pipeline.WanPipeline2_1)
    with self.assertRaises(NotImplementedError):  # TI2V image conditioning is not implemented for wan2.2
      sample.get_pipeline_class(_Cfg("wan2.2", "I2V", ["T2V"]))

  def test_align_video_size(self):
    def _pipe(scale_factor_spatial):
      pipe = object.__new__(wan_pipeline.WanPipeline2_1)  # no components needed
      pipe.vae_scale_factor_spatial = scale_factor_spatial
      pipe.transformer = types.SimpleNamespace(patch_size=(1, 2, 2))
      return pipe

    wan22, wan21 = _pipe(16), _pipe(8)
    self.assertEqual(wan22._align_video_size(720, 1280), (704, 1280))  # pylint: disable=protected-access
    self.assertEqual(wan22._align_video_size(704, 1280), (704, 1280))  # pylint: disable=protected-access
    self.assertEqual(wan21._align_video_size(720, 1280), (720, 1280))  # pylint: disable=protected-access
    self.assertEqual(wan21._align_video_size(721, 1290), (720, 1280))  # pylint: disable=protected-access

  def test_wan_yml_supported_types(self):
    """Each Wan model YAML declares the `model_type`s its checkpoint can run; others fail in `get_pipeline_class`."""
    expected = {
        "wan2.1-t2v-14b.yml": ("T2V", ["T2V"]),
        "wan2.1-i2v-14b-480p.yml": ("I2V", ["I2V"]),
        "wan2.2-t2v-5b.yml": ("T2V", ["T2V"]),
    }
    for name, (model_type, supported_types) in expected.items():
      with self.subTest(name):
        path = os.path.join(_WAN_CONFIG_DIR, name)
        cfg = config_lib.initialize(["prog", path, "run_name=t"], unittest=True)
        self.assertEqual(cfg.model_type, model_type)
        self.assertEqual(list(cfg.supported_types), supported_types)
        self.assertIs(sample.get_pipeline_class(cfg), wan_pipeline.WanPipeline2_1)
        other = "T2V" if model_type == "I2V" else "I2V"
        with self.assertRaises(NotImplementedError):
          sample.get_pipeline_class(
              config_lib.initialize(["prog", path, "run_name=t", f"model_type={other}"], unittest=True)
          )

  def test_call_pipeline_image_only_for_i2v(self):
    """`call_pipeline` loads `image_url` and passes `image=` for I2V only; T2V ignores a stray `image_url`."""
    calls = []

    def _pipeline(**kwargs):
      calls.append(kwargs)

    def _cfg(model_type, image_url):
      return types.SimpleNamespace(
          model_type=model_type,
          image_url=image_url,
          num_inference_steps=2,
          height=480,
          width=832,
          num_frames=5,
          guidance_scale=5.0,
          use_cfg_cache=False,
          use_magcache=False,
          magcache_thresh=0.12,
          magcache_K=2,
          retention_ratio=0.2,
          use_kv_cache=False,
      )

    with mock.patch.object(sample, "_load_conditioning_image", return_value="IMG") as load:
      sample.call_pipeline(_cfg("I2V", "http://x/img.jpg"), _pipeline, "p", "n")
      sample.call_pipeline(_cfg("T2V", "http://x/img.jpg"), _pipeline, "p", "n")
      load.assert_called_once_with("http://x/img.jpg")
      with self.assertRaises(ValueError):  # I2V needs an image
        sample.call_pipeline(_cfg("I2V", ""), _pipeline, "p", "n")
    self.assertEqual(calls[0]["image"], "IMG")
    self.assertNotIn("image", calls[1])


if __name__ == "__main__":
  unittest.main()
