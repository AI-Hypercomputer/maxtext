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

"""Tests for LingBot-World-Fast (`maxtext.m3.models.lingbot`): mapping coverage and parity with the reference.

The parity tests compare against the official LingBot-World implementation (https://github.com/Robbyant/lingbot-world,
the reference of the Hugging Face model card) on tiny random models, through the checkpoint converter. They need torch
and a checkout of that repo in `LINGBOT_WORLD_REPO`, and are skipped otherwise (local check only):

  LINGBOT_WORLD_REPO=/path/to/lingbot-world JAX_PLATFORMS=cpu python -m pytest tests/unit/m3/lingbot_test.py
"""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan needs diffusers (TPU pre-training requirements).

# pylint: disable=wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.checkpoint_conversion.to_maxtext_diffusers import convert_diffusers_component
from maxtext.checkpoint_conversion.utils.param_mapping import DIFFUSERS_PARAM_MAPPING, LINGBOT_WORLD_FAST_HF_TO_MAXTEXT
from maxtext.checkpoint_conversion.utils.utils import validate_and_filter_param_map_keys
from maxtext.m3.models.lingbot import camera_lingbot
from maxtext.m3.models.lingbot.checkpoint_lingbot import convert_wan_t5_to_hf
from maxtext.m3.models.lingbot.pipeline_lingbot import LingBotFastScheduler, first_frame_mask, latent_geometry, resize_bicubic
from maxtext.m3.models.lingbot.transformer_lingbot import LingBotWorldModel, rope_cos_sin
from maxtext.m3.models.wan import AutoencoderKLWan, UMT5EncoderModel
from maxtext.m3.models.wan.autoencoder_kl_wan import AutoencoderKLWanCache

try:
  import torch
except ImportError:
  torch = None
# pylint: enable=wrong-import-position

_REPO = os.environ.get("LINGBOT_WORLD_REPO", "")
_HAS_REFERENCE = torch is not None and bool(_REPO) and os.path.isdir(os.path.join(_REPO, "wan"))

# Tiny LingBot-World-Fast transformer (real model: dim 5120, 40 heads, 40 layers, ffn 13824, text_dim 4096).
_TRANSFORMER_CONFIG = {
    "model_type": "i2v",
    "in_dim": 36,
    "dim": 128,
    "ffn_dim": 256,
    "freq_dim": 32,
    "text_dim": 48,
    "out_dim": 16,
    "num_heads": 2,
    "num_layers": 2,
    "text_len": 8,
    "eps": 1e-6,
}


def _reference(module_name):
  """Imports a module of the reference repo without running `wan/__init__.py` (it pulls in every pipeline)."""
  for package in ("wan", "wan.modules", "wan.utils"):
    if package not in sys.modules:
      module = types.ModuleType(package)
      module.__path__ = [os.path.join(_REPO, *package.split("."))]
      sys.modules[package] = module
  if "wan.modules.tokenizers" not in sys.modules:  # needs ftfy; unused by the tested modules
    stub = types.ModuleType("wan.modules.tokenizers")
    stub.HuggingfaceTokenizer = None
    sys.modules["wan.modules.tokenizers"] = stub
  return importlib.import_module(module_name)


def _load_converted(model, state):
  flat = {tuple(int(p) if p.isdigit() else p for p in path): v for path, v in flatten_dict(state).items()}
  params = nnx.state(model, nnx.Param)
  nnx.replace_by_pure_dict(params, unflatten_dict(flat))
  nnx.update(model, params)
  return model


def _randomize(model, scale=0.05):
  with torch.no_grad():
    for name, param in model.named_parameters():
      if "norm" in name:
        param.copy_(1.0 + 0.1 * torch.randn_like(param))
      else:
        param.copy_(scale * torch.randn_like(param))


class LingBotParamMappingTest(parameterized.TestCase):

  def test_registered(self):
    self.assertIs(DIFFUSERS_PARAM_MAPPING["lingbot-world-fast"], LINGBOT_WORLD_FAST_HF_TO_MAXTEXT)

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_mapping_covers_transformer(self, scan_layers):
    model = nnx.eval_shape(lambda: LingBotWorldModel(**_TRANSFORMER_CONFIG, scan_layers=scan_layers, rngs=nnx.Rngs(0)))
    params = nnx.state(model, nnx.Param).to_pure_dict()
    target = {"-".join(map(str, path)): leaf.shape for path, leaf in flatten_dict(params).items()}
    mapping = LINGBOT_WORLD_FAST_HF_TO_MAXTEXT.param_mapping("transformer", _TRANSFORMER_CONFIG, scan_layers)
    keys = validate_and_filter_param_map_keys(mapping.keys(), set(target))  # raises on an unmapped model param
    self.assertLen(keys, len(target))
    hf_keys = [k for key in keys for k in (mapping[key] if isinstance(mapping[key], list) else [mapping[key]])]
    self.assertLen(set(hf_keys), len(hf_keys), "an HF tensor is mapped twice")
    for name in (
        "blocks.0.self_attn.q.weight",
        "blocks.1.cross_attn.norm_k.weight",
        "blocks.0.norm3.bias",
        "blocks.1.modulation",
        "blocks.0.cam_scale_layer.weight",
        "head.modulation",
        "head.head.weight",
        "time_projection.1.weight",
        "text_embedding.2.bias",
        "patch_embedding_wancamctrl.weight",
    ):
      self.assertIn(name, hf_keys)

  def test_t5_key_renaming(self):
    renamed = convert_wan_t5_to_hf(
        {
            "token_embedding.weight": 0,
            "norm.weight": 1,
            "blocks.3.ffn.gate.0.weight": 2,
            "blocks.3.pos_embedding.embedding.weight": 3,
        }
    )
    self.assertEqual(
        renamed,
        {
            "shared.weight": 0,
            "encoder.final_layer_norm.weight": 1,
            "encoder.block.3.layer.1.DenseReluDense.wi_0.weight": 2,
            "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight": 3,
        },
    )

  def test_latent_geometry_and_mask(self):
    # The example image (5145 x 2894) at 480*832 gives 464 x 832 output, as the reference.
    self.assertEqual(latent_geometry((2894, 5145), 480 * 832, 81, 3), (464, 832, 58, 104, 21, 81))
    # sqrt(480 * 832 * 480 / 832) rounds just below 480 in float64, so the reference also gives 464 rows here.
    self.assertEqual(latent_geometry((480, 832), 480 * 832, 21, 3), (464, 832, 58, 104, 6, 21))
    mask = first_frame_mask(21, 6, 2, 3)
    self.assertEqual(mask.shape, (4, 6, 2, 3))
    np.testing.assert_array_equal(mask[:, 0], 1.0)
    np.testing.assert_array_equal(mask[:, 1:], 0.0)


@unittest.skipUnless(_HAS_REFERENCE, "needs torch and LINGBOT_WORLD_REPO (the reference implementation)")
class LingBotReferenceParityTest(parameterized.TestCase):
  """Tiny random reference models converted with to_maxtext_diffusers.py, compared in float32 on CPU."""

  def _reference_transformer(self):
    """Tiny random reference `WanModelFast`, with its CUDA attention kernels replaced by float32 SDPA."""
    model_fast = _reference("wan.modules.model_fast")

    def sdpa(q, k, v, *args, **kwargs):  # [B, L, N, D]; the reference kernels are CUDA / bf16 only
      del args, kwargs
      out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
      return out.transpose(1, 2)

    model_fast.attention = sdpa
    model_fast.flash_attention = sdpa
    torch.manual_seed(0)
    model = model_fast.WanModelFast(control_type="cam", **_TRANSFORMER_CONFIG).eval()
    _randomize(model)
    return model

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_transformer_chunks_match_reference(self, scan_layers):
    ref = self._reference_transformer()
    cfg = _TRANSFORMER_CONFIG
    with tempfile.TemporaryDirectory() as repo_dir:  # laid out like robbyant/lingbot-world-fast
      from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel

      save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, os.path.join(repo_dir, "model.safetensors"))
      with open(os.path.join(repo_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cfg, _class_name="WanModel"), f)
      state, model_config = convert_diffusers_component(
          repo_dir, "transformer", LINGBOT_WORLD_FAST_HF_TO_MAXTEXT, scan_layers=scan_layers, weights_dtype=np.float32
      )
    self.assertEqual(model_config["scan_layers"], scan_layers)
    model = LingBotWorldModel(
        **model_config, dtype=jnp.float32, weights_dtype=jnp.float32, attention="dot_product", rngs=nnx.Rngs(1)
    )
    _load_converted(model, state)

    # Two chunks of 3 latent frames at 4 x 6 latents: tokens / frame 2 * 3 = 6, chunk 18, cache 36.
    chunk, frames, lat_h, lat_w = 3, 6, 4, 6
    frame_seqlen = (lat_h // 2) * (lat_w // 2)
    kv_size = frames * frame_seqlen
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((16, frames, lat_h, lat_w), dtype=np.float32)
    cond = rng.standard_normal((20, frames, lat_h, lat_w), dtype=np.float32)
    cam = rng.standard_normal((1, 6 * 64, frames, lat_h, lat_w), dtype=np.float32)
    context = np.zeros((cfg["text_len"], cfg["text_dim"]), np.float32)
    context[:5] = rng.standard_normal((5, cfg["text_dim"]), dtype=np.float32)

    # Reference: chunk 0 at t=900, commit chunk 0 (t=0), chunk 1 at t=500.
    head_dim = cfg["dim"] // cfg["num_heads"]
    ref_kv = [
        {
            "k": torch.zeros(1, kv_size, cfg["num_heads"], head_dim),
            "v": torch.zeros(1, kv_size, cfg["num_heads"], head_dim),
            "global_end_index": torch.tensor([0]),
            "local_end_index": torch.tensor([0]),
        }
        for _ in range(cfg["num_layers"])
    ]
    ref_cross = [
        {
            "k": torch.zeros(1, cfg["text_len"], cfg["num_heads"], head_dim),
            "v": torch.zeros(1, cfg["text_len"], cfg["num_heads"], head_dim),
            "is_init": torch.tensor(0),
        }
        for _ in range(cfg["num_layers"])
    ]

    def ref_forward(x, t, chunk_id, first):
      sl = slice(chunk_id * chunk, (chunk_id + 1) * chunk)
      with torch.no_grad():
        return ref(
            x=[torch.from_numpy(x)],
            t=torch.tensor([t]),
            context=[torch.from_numpy(context[:5])],
            seq_len=chunk * frame_seqlen,
            y=[torch.from_numpy(cond[:, sl])],
            dit_cond_dict={"c2ws_plucker_emb": (torch.from_numpy(cam[:, :, sl]),)},
            kv_cache=ref_kv,
            crossattn_cache=ref_cross,
            current_start=chunk_id * chunk * frame_seqlen,
            max_attention_size=kv_size,
            frame_seqlen=frame_seqlen,
            cross_attn_first_call=first,
        )[0].numpy()

    x0_chunk0 = noise[:, :chunk] * 0.5
    expected = [ref_forward(noise[:, :chunk], 900.0, 0, True)]
    ref_forward(x0_chunk0, 0.0, 0, False)
    expected.append(ref_forward(noise[:, chunk:], 500.0, 1, False))

    with jax.default_matmul_precision("float32"):
      cross_kv = model.compute_cross_kv(jnp.asarray(context)[None])
      kv_cache = tuple(jnp.zeros(model.kv_cache_shape(1, kv_size), jnp.float32) for _ in range(2))

      def forward(x, t, chunk_id, kv_cache):
        sl = slice(chunk_id * chunk, (chunk_id + 1) * chunk)
        cos, sin = rope_cos_sin(head_dim, chunk_id * chunk, chunk, lat_h // 2, lat_w // 2)
        return model(
            jnp.asarray(x)[None],
            jnp.asarray([t], jnp.float32),
            jnp.asarray(cond[:, sl])[None],
            jnp.asarray(cam[:, :, sl]),
            cross_kv,
            cos,
            sin,
            kv_cache,
            jnp.int32(chunk_id * chunk * frame_seqlen),
        )

      out0, _ = forward(noise[:, :chunk], 900.0, 0, kv_cache)
      _, (new_k, new_v) = forward(x0_chunk0, 0.0, 0, kv_cache)
      kv_cache = tuple(jax.lax.dynamic_update_slice(c, n, (0, 0, 0, 0, 0)) for c, n in zip(kv_cache, (new_k, new_v)))
      out1, _ = forward(noise[:, chunk:], 500.0, 1, kv_cache)
    np.testing.assert_allclose(np.asarray(out0)[0], expected[0], atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(out1)[0], expected[1], atol=2e-4, rtol=2e-4)

  def test_text_encoder_matches_reference(self):
    current_device = torch.cuda.current_device
    torch.cuda.current_device = lambda: "cpu"  # `T5EncoderModel` evaluates it as a default argument at import
    try:
      t5 = _reference("wan.modules.t5")
    finally:
      torch.cuda.current_device = current_device
    torch.manual_seed(0)
    ref = t5.T5Encoder(
        vocab=64, dim=32, dim_attn=32, dim_ffn=48, num_heads=4, num_layers=3, num_buckets=8, shared_pos=False, dropout=0.0
    ).eval()
    _randomize(ref, scale=0.2)
    config = {
        "vocab_size": 64,
        "d_model": 32,
        "d_kv": 8,
        "d_ff": 48,
        "num_layers": 3,
        "num_heads": 4,
        "relative_attention_num_buckets": 8,
        "relative_attention_max_distance": 128,  # T5RelativeEmbedding default
        "layer_norm_epsilon": 1e-6,
        "dense_act_fn": "gelu_new",
        "is_gated_act": True,
    }

    class TinyMapping(LINGBOT_WORLD_FAST_HF_TO_MAXTEXT):
      COMPONENT_SOURCES = {
          "text_encoder": dict(LINGBOT_WORLD_FAST_HF_TO_MAXTEXT.COMPONENT_SOURCES["text_encoder"], config=config)
      }

    with tempfile.TemporaryDirectory() as repo_dir:
      torch.save(ref.state_dict(), os.path.join(repo_dir, "models_t5_umt5-xxl-enc-bf16.pth"))
      state, model_config = convert_diffusers_component(
          repo_dir, "text_encoder", TinyMapping, scan_layers=False, weights_dtype=np.float32
      )
    model = _load_converted(
        UMT5EncoderModel(**model_config, precision=jax.lax.Precision.HIGHEST, rngs=nnx.Rngs(0)), state
    )
    batch, length = 2, 40
    ids = np.random.default_rng(0).integers(1, 64, size=(batch, length))
    mask = (np.arange(length)[None, :] < np.array([length, 17])[:, None]).astype(np.int64)
    ids = np.where(mask > 0, ids, 0)
    with torch.no_grad():
      expected = ref(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
    actual = np.asarray(model(ids, mask))
    valid = mask.astype(bool)
    np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-4, rtol=1e-4)

  def test_vae_matches_reference(self):
    vae = _reference("wan.modules.vae2_1")
    torch.manual_seed(0)
    ref = vae.WanVAE_(dim=8, z_dim=4, temperal_downsample=[False, True, True]).eval()
    _randomize(ref)
    config = dict(LINGBOT_WORLD_FAST_HF_TO_MAXTEXT.COMPONENT_SOURCES["vae"])
    config["config"] = {
        "base_dim": 8,
        "z_dim": 4,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
    }

    class TinyMapping(LINGBOT_WORLD_FAST_HF_TO_MAXTEXT):
      COMPONENT_SOURCES = {"vae": config}

    with tempfile.TemporaryDirectory() as repo_dir:
      torch.save(ref.state_dict(), os.path.join(repo_dir, "Wan2.1_VAE.pth"))
      state, model_config = convert_diffusers_component(
          repo_dir, "vae", TinyMapping, scan_layers=False, weights_dtype=np.float32
      )
    model = _load_converted(AutoencoderKLWan(**model_config, rngs=nnx.Rngs(0)), state)
    video = np.random.default_rng(0).uniform(-1.0, 1.0, (1, 3, 5, 16, 16)).astype(np.float32)
    scale = [torch.zeros(4), torch.ones(4)]
    with torch.no_grad():
      expected_mu = ref.encode(torch.from_numpy(video), scale)
      expected_video = ref.decode(expected_mu, scale).clamp(-1, 1).numpy()
    expected_mu = expected_mu.numpy()
    with jax.default_matmul_precision("float32"):
      mu = model.encode(video, AutoencoderKLWanCache(model)).latent_dist.mode()
      decoded = model.decode(np.transpose(expected_mu, (0, 2, 3, 4, 1)), AutoencoderKLWanCache(model)).sample
    np.testing.assert_allclose(np.transpose(np.asarray(mu), (0, 4, 1, 2, 3)), expected_mu, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.transpose(np.asarray(decoded), (0, 4, 1, 2, 3)), expected_video, atol=1e-4, rtol=1e-4)

  def test_camera_condition_matches_reference(self):
    cam_utils = _reference("wan.utils.cam_utils")
    rng = np.random.default_rng(0)
    num_frames, height, width, lat_h, lat_w, lat_f, chunk = 21, 64, 96, 8, 12, 6, 3
    angles = np.cumsum(rng.normal(0, 0.05, (num_frames, 3)), axis=0)
    rot = np.stack([camera_lingbot.Rotation.from_rotvec(a).as_matrix() for a in angles])
    c2ws = np.tile(np.eye(4), (num_frames, 1, 1))
    c2ws[:, :3, :3] = rot
    c2ws[:, :3, 3] = np.cumsum(rng.normal(0, 0.1, (num_frames, 3)), axis=0)
    c2ws = c2ws.astype(np.float32)
    intrinsics = np.tile(np.array([[500.0, 480.0, 416.0, 240.0]], np.float32), (num_frames, 1))
    actual = camera_lingbot.camera_condition(c2ws, intrinsics, height, width, lat_h, lat_w, lat_f, chunk)

    # Reference (WanI2VFast.generate, on CPU).
    ks = cam_utils.get_Ks_transformed(torch.from_numpy(intrinsics), 480, 832, height, width, height, width)[0]
    c2ws_infer = cam_utils.interpolate_camera_poses(
        np.linspace(0, num_frames - 1, num_frames), c2ws[:, :3, :3], c2ws[:, :3, 3], np.linspace(0, num_frames - 1, lat_f)
    )
    c2ws_infer = cam_utils.compute_relative_poses(c2ws_infer, framewise=True)
    plucker = cam_utils.get_plucker_embeddings(c2ws_infer, ks.repeat(lat_f, 1), height, width)
    plucker = plucker.reshape(lat_f, lat_h, height // lat_h, lat_w, width // lat_w, 6)
    expected = plucker.permute(5, 2, 4, 0, 1, 3).reshape(1, -1, lat_f, lat_h, lat_w).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

  def test_schedule_matches_reference(self):
    solver = _reference("wan.utils.fm_solvers_unipc")
    ref = solver.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    ref.set_timesteps(1000, shift=10.0)
    ours = LingBotFastScheduler(1000, 10.0, (0, 179, 358, 679))
    np.testing.assert_array_equal(ours.timesteps, ref.timesteps[[0, 179, 358, 679]].numpy())
    for t in ours.timesteps:
      idx = torch.argmin((ref.timesteps.double() - float(t)).abs())
      self.assertEqual(ours.x0_sigma(t), ref.sigmas.double()[idx].item())
      x0 = torch.zeros(1)
      noise = torch.ones(1)
      self.assertEqual(float(ours.noise_sigma(t)), ref.add_noise(x0, noise, torch.tensor(t)).item())

  def test_bicubic_resize_matches_torch(self):
    image = np.random.default_rng(0).uniform(-1, 1, (3, 37, 53)).astype(np.float32)
    for size in ((16, 24), (37, 53), (60, 90)):
      expected = torch.nn.functional.interpolate(torch.from_numpy(image)[None], size=size, mode="bicubic")[0].numpy()
      np.testing.assert_allclose(resize_bicubic(image, *size), expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
