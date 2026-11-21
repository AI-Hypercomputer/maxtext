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

"""Tests for the Ming-Image pipeline helpers (sampling schedule, config, dispatch)."""

import os
import tempfile
import unittest

import pytest

diffusers = pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from absl.testing import parameterized
import numpy as np

from maxtext.inference import aot_cache
from maxtext.inference import sample
from maxtext.m3.configs import config as config_lib
from maxtext.m3.diffuser_pipelines import ming_image_pipeline

# pylint: enable=wrong-import-position

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "../../../../src/maxtext/m3/models/ming_image/configs/ming-image-0.1-design.yml"
)


def _reference_sigmas(num_inference_steps, image_seq_len):
  """Sigmas as set up by the reference `ImageGenerator` / `ImageGenerationPipeline.__call__`."""
  scheduler = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, use_dynamic_shifting=False, shift=6.0)
  scheduler.register_to_config(use_dynamic_shifting=True)  # reference: `config['use_dynamic_shifting'] = True`
  if image_seq_len >= 4096:
    max_seq, max_shift = image_seq_len, 1.35
  else:
    max_seq, max_shift = 4096, 1.15
  m = (max_shift - 0.5) / (max_seq - 256)
  mu = image_seq_len * m + 0.5 - m * 256
  scheduler.sigma_min = 0.0
  scheduler.set_timesteps(num_inference_steps, mu=mu)
  return scheduler.sigmas.numpy()


class FlowMatchScheduleTest(parameterized.TestCase):

  @parameterized.parameters((12, 64 * 64), (12, 128 * 128), (12, 32 * 32), (50, 64 * 64), (2, 64 * 64))
  def test_matches_reference_scheduler(self, steps, image_seq_len):
    np.testing.assert_allclose(
        ming_image_pipeline.flow_match_sigmas(steps, image_seq_len),
        _reference_sigmas(steps, image_seq_len),
        atol=1e-6,
    )

  def test_shift(self):
    self.assertAlmostEqual(ming_image_pipeline.calculate_shift(4096), 1.35)
    self.assertAlmostEqual(ming_image_pipeline.calculate_shift(16384), 1.35)
    self.assertAlmostEqual(ming_image_pipeline.calculate_shift(256), 0.5)

  def test_reference_euler_step(self):
    """`x + (s_next - s) * (-out)` == diffusers `step(-out)` (the reference negates the transformer output)."""
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0)
    scheduler.register_to_config(use_dynamic_shifting=True)
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(4, mu=1.35)
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x, out = rng.standard_normal((2, 4, 8, 8)).astype(np.float32), rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    expected = scheduler.step(torch.from_numpy(-out), scheduler.timesteps[1], torch.from_numpy(x)).prev_sample.numpy()
    sigmas = ming_image_pipeline.flow_match_sigmas(4, 64 * 64)
    np.testing.assert_allclose(x + (sigmas[2] - sigmas[1]) * (-out), expected, atol=1e-5)


class MingImageConfigTest(unittest.TestCase):

  def test_config_and_dispatch(self):
    config = config_lib.initialize([None, _CONFIG_PATH])
    self.assertEqual(config.model_name, "ming-image")
    self.assertEqual(config.model_type, "T2I")
    self.assertEqual((config.height, config.width, config.num_inference_steps), (1024, 1024, 12))
    self.assertEqual(config.guidance_scale, 1.0)
    self.assertIs(sample.get_pipeline_class(config), ming_image_pipeline.MingImagePipeline)
    self.assertTrue(sample._get_output_filename(config, "", 0).endswith(".png"))  # pylint: disable=protected-access


class _StubTokenizer:
  """Character-level stand-in for the Bailing tokenizer (ids < the tiny vocab)."""

  eos_token = "<|role_end|>"

  def encode(self, text, add_special_tokens=False):
    del add_special_tokens
    return [3 + (ord(c) % 150) for c in text]

  def __call__(self, text, add_special_tokens=False, **unused):
    return {"input_ids": self.encode(text)}


# Tiny but structurally complete components; heads / MLP dims divide 8 so the sharding rules are exercised.
_TINY_LLM = {
    "vocab_size": 200,
    "hidden_size": 64,
    "head_dim": 128,  # video_rope's fixed mrope section [8, 12, 12] needs rope_dim 64 = 0.5 * 128
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "num_shared_experts": 1,
    "n_group": 4,
    "topk_group": 2,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "partial_rotary_factor": 0.5,
    "rope_theta": 600000,
    "rope_scaling": {"factor": None, "type": "video_rope"},
    "rms_norm_eps": 1e-6,
    "use_qkv_bias": False,
    "use_bias": False,
    "use_expert_bias": True,
    "image_patch_token": 190,
    "image_start_token": 191,
    "image_end_token": 192,
    "video_patch_token": 196,
    "video_start_token": 193,
    "video_end_token": 194,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "spatial_merge_size": 2,
    "tokens_per_second": 2,
}
_TINY_CONNECTOR = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1e6,
}
_TINY_CONDITIONER = {
    "img_gen_scales": [16],
    "diffusion_c_input_dim": 24,
    "diffusion_inner_dim": 64,
    "selected_hidden_states_layers": [1, 2, 4],
    "llm_hidden_size": 64,
    "connector_hidden_size": 32,
}
_TINY_TRANSFORMER = {
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "axes_dims": [4, 6, 6],
    "axes_lens": [1024, 64, 64],
    "cap_feat_dim": 24,
    "dim": 64,
    "in_channels": 4,
    "n_heads": 4,
    "n_kv_heads": 4,
    "n_layers": 2,
    "n_refiner_layers": 1,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "alignment_padding_mode": "zero_masked",
    "multi_frame_output": False,
}
_TINY_VAE = {
    "attn_scales": [],
    "base_dim": 8,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "input_channels": 4,
    "num_res_blocks": 1,
    "temperal_downsample": [False, True, True],
    "z_dim": 4,
}


def _random_restored_checkpoint():
  """An in-memory `restore_diffusers_checkpoint`-style dict with random tiny Ming-Image components."""
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.traverse_util import flatten_dict, unflatten_dict
  from maxtext.m3.models.ming_image import MingConditioner, MingConnector, MingImageTransformer, MingTextEncoder
  from maxtext.m3.models.wan.autoencoder_kl_wan import AutoencoderKLWan

  components = {
      "mllm": (MingTextEncoder, {"llm_config": _TINY_LLM}),
      "connector": (MingConnector, _TINY_CONNECTOR),
      "mlp": (MingConditioner, _TINY_CONDITIONER),
      "transformer": (MingImageTransformer, _TINY_TRANSFORMER),
      "vae": (AutoencoderKLWan, _TINY_VAE),
  }
  restored = {"scheduler_config": {"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 6.0}}
  rng = np.random.default_rng(0)
  for name, (cls, cfg) in components.items():
    model = cls(**cfg, rngs=nnx.Rngs(0))
    params = flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())
    # Random (not init-default) weights so a dropped / mis-placed param changes the output.
    params = {
        tuple(str(p) for p in path): (0.05 * rng.standard_normal(v.shape)).astype(np.float32)
        for path, v in params.items()
    }
    restored[f"{name}_state"] = unflatten_dict(params)
    restored[f"{name}_config"] = cfg
  # The real connector config.json carries `"dtype": "float32"` (kept by `model_init_config`).
  restored["connector_config"] = dict(restored["connector_config"], dtype="float32")
  return restored


class MingImagePipelineSmokeTest(unittest.TestCase):
  """End-to-end tiny pipeline: checkpoint placement on the mesh, text encoding, sampling, RGBA decode."""

  def test_tiny_pipeline_generates_rgba(self):
    config = config_lib.initialize(
        [
            None,
            _CONFIG_PATH,
            "attention=dot_product",
            "height=64",
            "width=64",
            "num_inference_steps=2",
            "text_length_buckets=[512]",
            "max_sequence_length=512",
            "vae_spatial=1",
        ]
    )
    pipeline_cls = ming_image_pipeline.MingImagePipeline
    original = ming_image_pipeline.text_encoder_ming.load_tokenizer
    ming_image_pipeline.text_encoder_ming.load_tokenizer = lambda _: _StubTokenizer()
    original_snapshot = pipeline_cls._snapshot_dir  # pylint: disable=protected-access
    pipeline_cls._snapshot_dir = classmethod(lambda cls, cfg: "/nonexistent")  # pylint: disable=protected-access
    try:
      pipeline = pipeline_cls.from_checkpoint(config, restored_checkpoint=_random_restored_checkpoint())
    finally:
      ming_image_pipeline.text_encoder_ming.load_tokenizer = original
      pipeline_cls._snapshot_dir = original_snapshot  # pylint: disable=protected-access

    images, trace = pipeline(prompt=["a red cube", "a blue poster with the word HELLO"], height=64, width=64)
    self.assertEqual(images.shape, (2, 64, 64, 4))
    self.assertEqual(images.dtype, np.uint8)
    self.assertGreater(np.ptp(images), 0)
    for stage in ("conditioning", "denoise_total", "vae_decode"):
      self.assertIn(stage, trace)
    # Same prompt -> cached conditioning, deterministic sampling.
    images_again, _ = pipeline(prompt=["a red cube", "a blue poster with the word HELLO"], height=64, width=64)
    np.testing.assert_array_equal(images, images_again)
    # CFG path (guidance > 1) compiles and runs.
    images_cfg, _ = pipeline(prompt=["a red cube", "a red cube"], height=64, width=64, guidance_scale=2.0)
    self.assertEqual(images_cfg.shape, (2, 64, 64, 4))

    # sample.py flow: AOT cache enabled, zero-execution warmup, then the real call. The warmup must not poison the
    # prompt-embedding cache (it once cached all-zero conditioning, making every image prompt-independent). Compare
    # denoised latents: the tiny random VAE quantizes small differences away in the uint8 images.
    pipeline._decode = lambda latents, trace: np.asarray(latents)  # pylint: disable=protected-access
    with tempfile.TemporaryDirectory() as aot_dir:
      aot_cache.install(aot_dir, meta={"test": "ming"}, mesh=pipeline.mesh)
      try:
        prompts = ["a green tree on a hill", "a poster with the word SALE"]
        with aot_cache.warmup_mode():
          pipeline(prompt=prompts, height=64, width=64, num_inference_steps=1)
        latents, _ = pipeline(prompt=prompts, height=64, width=64)
        cap_feats = np.asarray(pipeline.encode_prompt(prompts)[0])
        self.assertGreater(np.abs(cap_feats).max(), 0.0)  # Real (non-zero) conditioning is cached.
        swapped, _ = pipeline(prompt=prompts[::-1], height=64, width=64)
        # The output depends on the prompt: the same batch row (same noise) changes when its prompt changes.
        self.assertGreater(np.abs(latents[0] - swapped[0]).max(), 1e-4)
      finally:
        aot_cache.install("", meta={}, mesh=None)


if __name__ == "__main__":
  unittest.main()
