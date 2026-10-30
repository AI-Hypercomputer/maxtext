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

"""Layer-wise tests: MaxText Ming talker heads vs the publisher's `talker/` PyTorch modules.

Compares the CFM DiT (with classifier-free guidance), the full Euler sampler, the aggregator and the
AudioVAE decoder (sliding-window Qwen2 backbone + ISTFT head) with small random weights on CPU in float32.
The talker LLM is MaxText's regular Qwen2 decoder and is covered by the end-to-end parity check
(`tests/utils/ming_reference/talker_parity.py`).

The reference modules are imported from a local copy of the HF snapshot's `talker/` directory
(`MING_TALKER_HF_PATH`); they need `x-transformers` and `diffusers` (`MING_TALKER_EXTRA_PATH` may point at a
side directory with extra packages). Frontend tests run without the reference.

Run:
  MING_TALKER_HF_PATH=<snapshot>/talker MING_TALKER_EXTRA_PATH=<extra site dir> \
    JAX_PLATFORMS=cpu python3 -m pytest tests/unit/ming_talker_layers_test.py
"""

import os
import sys
import unittest

import numpy as np
import pytest

from flax import nnx
import jax.numpy as jnp

from maxtext.configs import pyconfig
from maxtext.models import ming_talker
from maxtext.multimodal import processor_ming_talker
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

BASE_CONFIG = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
HF_PATH = os.environ.get("MING_TALKER_HF_PATH", "")
EXTRA_PATH = os.environ.get("MING_TALKER_EXTRA_PATH", "")

# Small but structurally faithful dimensions.
EMB, LATENT = 32, 8
DIT_HIDDEN, DIT_HEADS, DIT_LAYERS = 64, 4, 2
VAE_HIDDEN, VAE_HEADS, VAE_KV_HEADS, VAE_MLP, VAE_LAYERS, VAE_WINDOW, VAE_HOP = 64, 4, 2, 128, 2, 8, 16
ATOL, RTOL = 2e-4, 2e-4


def make_config(**kwargs):
  """Small, structurally faithful Ming talker config (CPU, float32)."""
  overrides = dict(  # pylint: disable=use-dict-literal
      model_name="ming-3.0-audio-tiny-talker",
      override_model_config=True,
      base_emb_dim=EMB,
      base_num_query_heads=4,
      base_num_kv_heads=2,
      head_dim=8,
      base_mlp_dim=64,
      base_num_decoder_layers=1,
      vocab_size=128,
      ming_talker_latent_dim=LATENT,
      ming_talker_speaker_dim=16,
      ming_talker_dit_hidden_size=DIT_HIDDEN,
      ming_talker_dit_num_layers=DIT_LAYERS,
      ming_talker_dit_num_heads=DIT_HEADS,
      ming_talker_aggregator_hidden_size=DIT_HIDDEN,
      ming_talker_aggregator_num_layers=DIT_LAYERS,
      ming_talker_aggregator_num_heads=DIT_HEADS,
      ming_vae_hidden_size=VAE_HIDDEN,
      ming_vae_num_layers=VAE_LAYERS,
      ming_vae_num_heads=VAE_HEADS,
      ming_vae_num_kv_heads=VAE_KV_HEADS,
      ming_vae_mlp_dim=VAE_MLP,
      ming_vae_sliding_window=VAE_WINDOW,
      ming_vae_hop_length=VAE_HOP,
      per_device_batch_size=1,
      max_target_length=32,
      max_prefill_predict_length=16,
      scan_layers=False,
      matmul_precision="highest",
      dtype="float32",
      weight_dtype="float32",
      skip_jax_distributed_system=True,
      hardware="cpu",
  )
  overrides.update(kwargs)
  return pyconfig.initialize(["", BASE_CONFIG], **overrides)


class MingTalkerFrontendTest(unittest.TestCase):
  """Reference-free checks of the text frontend."""

  def test_ip_instruction(self):
    instruction = processor_ming_talker.build_ip_instruction("灵小柔")
    self.assertEqual(
        instruction,
        '{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, "风格": null, "语速": null, "基频": null,'
        ' "音量": null, "情感": null, "BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null,'
        ' "ENV": null, "SNR": null}, "IP": "灵小柔"}]}',
    )

  def test_prompt_language(self):
    default = processor_ming_talker.DEFAULT_TTS_PROMPT
    self.assertEqual(processor_ming_talker.build_tts_prompt(None), default)
    self.assertEqual(processor_ming_talker.build_tts_prompt("en"), default)
    self.assertEqual(processor_ming_talker.build_tts_prompt("ja"), default + "Target language: Japanese.\n")

  def test_split_text(self):
    self.assertEqual(processor_ming_talker.split_text("Hello world"), ["Hello world."])
    frags = processor_ming_talker.split_text("One two three. Four five six! " * 4, max_length=40)
    self.assertTrue(all(len(f) <= 40 for f in frags))
    self.assertEqual(" ".join(frags), ("One two three. Four five six! " * 4).strip())

  def test_trim_trailing_silence(self):
    sr = 1000
    speech = np.concatenate([0.5 * np.ones(250), np.zeros(1000), 0.5 * np.ones(30)])
    # Frames of 100: last loud frame is frame 2 (samples 200-299); keep 0.3 s after it; remainder 80 samples dropped.
    out = processor_ming_talker.trim_trailing_silence(speech[:1250], sr)
    self.assertEqual(out.shape[0], 300 + 300)
    self.assertEqual(processor_ming_talker.trim_trailing_silence(np.zeros(50), sr).shape[0], 50)

  def test_timesteps(self):
    t = np.asarray([0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], np.float32) / 32
    np.testing.assert_allclose(ming_talker.epss_timesteps(10), 1 - np.cos(np.pi / 2 * t), atol=1e-6)


@pytest.mark.skipif(not HF_PATH, reason="MING_TALKER_HF_PATH (HF snapshot talker/ directory) is not set")
class MingTalkerLayersTest(unittest.TestCase):
  """MaxText talker heads vs the publisher's PyTorch modules with identical random weights."""

  @classmethod
  def setUpClass(cls):
    for path in (EXTRA_PATH, HF_PATH):
      if path and path not in sys.path:
        sys.path.insert(0, path)
    import torch  # pylint: disable=import-outside-toplevel
    from talker_module import aggregator, cfm, dit  # pylint: disable=import-outside-toplevel,import-error
    from AudioVAE import vae_modules  # pylint: disable=import-outside-toplevel,import-error

    cls.torch, cls.hf_dit, cls.hf_cfm, cls.hf_agg, cls.hf_vae = torch, dit, cfm, aggregator, vae_modules
    torch.manual_seed(0)
    cls.config = make_config()
    cls.heads = ming_talker.MingTalkerHeads(cls.config, rngs=nnx.Rngs(0))

  # ---------------------------------------------------------------- helpers
  def t2j(self, t):
    return jnp.asarray(t.detach().float().cpu().numpy())

  def randomize(self, module, seed, std=0.1):
    """Fills HF parameters with random values; RoPE buffers are bf16-rounded like the checkpoint."""
    torch = self.torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
      for name, p in list(module.named_parameters()) + list(module.named_buffers()):
        if "inv_freq" in name:
          # The publisher's checkpoint stores bf16-rounded RoPE frequencies.
          p.copy_(p.to(torch.bfloat16).float())
        elif name.endswith("window"):
          continue
        elif "norm" in name:
          p.copy_(1.0 + std * torch.randn(p.shape, generator=g))
        else:
          p.copy_(std * torch.randn(p.shape, generator=g))

  def set_dense(self, mt, hf):
    mt.kernel.value = self.t2j(hf.weight).T
    if getattr(hf, "bias", None) is not None:
      mt.bias.value = self.t2j(hf.bias)

  def set_norm(self, mt, hf):
    mt.scale.value = self.t2j(hf.weight)

  def copy_blocks(self, mt, hf_blocks):
    for i, blk in enumerate(hf_blocks):
      m = getattr(mt, f"blocks_{i}")
      self.set_norm(m.norm1, blk.norm1)
      self.set_norm(m.norm2, blk.norm2)
      for n in ("to_q", "to_k", "to_v"):
        self.set_dense(getattr(m.attn, n), getattr(blk.attn, n))
      self.set_dense(m.attn.to_out, blk.attn.to_out[0])
      self.set_dense(m.mlp_in, blk.mlp.ff[0][0])
      self.set_dense(m.mlp_out, blk.mlp.ff[2])

  def make_dit(self):
    """Builds a random HF DiT and copies its weights into the MaxText CFM DiT."""
    hf = self.hf_dit.DiT(
        in_channels=LATENT,
        hidden_size=DIT_HIDDEN,
        depth=DIT_LAYERS,
        num_heads=DIT_HEADS,
        mlp_ratio=4,
        llm_cond_dim=EMB,
        attn_backend="torch",
        attn_mask_enabled=False,
        dropout=0.0,
    ).eval()
    self.randomize(hf, 1)
    mt = self.heads.cfm
    self.set_dense(mt.x_embedder, hf.x_embedder)
    self.set_dense(mt.c_embedder, hf.c_embedder.cond_embedder)
    self.set_dense(mt.time_fc1, hf.t_embedder.time_mlp[0])
    self.set_dense(mt.time_fc2, hf.t_embedder.time_mlp[2])
    self.copy_blocks(mt, hf.blocks)
    self.set_norm(mt.final_norm, hf.final_layer.norm_final)
    self.set_dense(mt.final_linear, hf.final_layer.linear)
    return hf, mt

  def assert_close(self, mt_out, hf_out, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(np.asarray(mt_out, np.float32), hf_out.detach().float().numpy(), atol=atol, rtol=rtol)

  # ---------------------------------------------------------------- tests
  def test_dit_rope_matches_checkpoint_buffers(self):
    torch = self.torch
    ref = (1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))).to(torch.bfloat16).float()
    np.testing.assert_array_equal(ming_talker.dit_rope_freqs(2, 64)[1, ::2], ref.numpy())
    hann = torch.hann_window(3528).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(ming_talker.bf16_round(ming_talker.periodic_hann_window(3528)), hann)

  def test_dit_forward_with_cfg(self):
    torch = self.torch
    hf, mt = self.make_dit()
    g = torch.Generator().manual_seed(2)
    x, hist = torch.randn(2, 1, LATENT, generator=g), torch.randn(2, 1, LATENT, generator=g)
    cond = torch.randn(2, 1, EMB, generator=g)
    t = torch.tensor(0.3)
    with torch.no_grad():
      pred, null_pred = torch.chunk(hf.forward_with_cfg(x, t, cond, hist), 2, dim=0)
      expected = pred + (pred - null_pred) * 2.0
    out = mt.forward_with_cfg(self.t2j(x), jnp.float32(0.3), self.t2j(cond), self.t2j(hist), 2.0)
    self.assert_close(out, expected)

  def test_cfm_sample(self):
    torch = self.torch
    hf_dit, mt = self.make_dit()
    hf = self.hf_cfm.CFM(hf_dit, steps=10)
    g = torch.Generator().manual_seed(3)
    noise, hist = torch.randn(1, 1, LATENT, generator=g), torch.randn(1, 1, LATENT, generator=g)
    cond = torch.randn(1, 1, EMB, generator=g)
    sde = torch.randn(10, 1, 1, LATENT, generator=g)
    timesteps = self.hf_cfm.get_epss_timesteps(10, "cpu", torch.float32)
    for sigma, temperature in ((0.25, 0.0), (0.25, 0.5)):
      with torch.no_grad():
        expected = hf.sample(cond, hist, noise, timesteps, torch.tensor([2.0, sigma, temperature]), sde)
      out = ming_talker.cfm_sample(
          mt, self.t2j(cond), self.t2j(hist), self.t2j(noise), 10, 2.0, sigma, temperature, self.t2j(sde)
      )
      self.assert_close(out, expected, atol=1e-3, rtol=1e-3)

  def test_aggregator(self):
    torch = self.torch
    hf = self.hf_agg.Aggregator(
        in_channels=LATENT,
        hidden_size=DIT_HIDDEN,
        depth=DIT_LAYERS,
        num_heads=DIT_HEADS,
        mlp_ratio=4,
        llm_input_dim=EMB,
        attn_backend="torch",
        attn_mask_enabled=False,
        dropout=0.0,
    ).eval()
    self.randomize(hf, 4)
    mt = self.heads.aggregator
    mt.word_embedder.value = self.t2j(hf.word_embedder.weight)
    self.set_dense(mt.x_embedder, hf.x_embedder)
    self.copy_blocks(mt, hf.blocks)
    self.set_norm(mt.final_norm, hf.final_layer.norm_final)
    self.set_dense(mt.final_linear, hf.final_layer.linear)
    latent = torch.randn(3, 1, LATENT, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
      expected = hf(latent)
    self.assert_close(mt(self.t2j(latent)), expected)

  def test_vae_decoder(self):
    torch = self.torch
    backbone = {
        "hidden_size": VAE_HIDDEN,
        "intermediate_size": VAE_MLP,
        "num_hidden_layers": VAE_LAYERS,
        "num_attention_heads": VAE_HEADS,
        "num_key_value_heads": VAE_KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "sliding_window": VAE_WINDOW,
        "use_sliding_window": True,
        "max_window_layers": 0,
        "max_position_embeddings": 4096,
        "vocab_size": 1,
        "is_causal": True,
        "use_cache": False,
        "_attn_implementation": "eager",
        "attn_implementation": "eager",
        "hidden_act": "silu",
        "model_type": "qwen2",
    }
    hf = self.hf_vae.Decoder(decoder_args=backbone, output_dim=VAE_HOP, latent_dim=LATENT, patch_size=4).eval()
    self.randomize(hf, 6, std=0.05)
    mt = self.heads.audio_vae
    self.set_dense(mt.fc1, hf.fc1)
    for i, layer in enumerate(hf.decoder.layers):
      m = getattr(mt, f"layers_{i}")
      self.set_norm(m.input_layernorm, layer.input_layernorm)
      self.set_norm(m.post_attention_layernorm, layer.post_attention_layernorm)
      for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        self.set_dense(getattr(m, n), getattr(layer.self_attn, n))
      for n in ("gate_proj", "up_proj", "down_proj"):
        self.set_dense(getattr(m, n), getattr(layer.mlp, n))
    self.set_norm(mt.norm, hf.decoder.norm)
    self.set_dense(mt.head_out, hf.head.out)
    # 6 latents -> 24 frames > sliding window 8, so the window mask is exercised.
    latents = torch.randn(1, 6, LATENT, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
      expected, _, _ = hf.low_level_reconstruct(
          latents, use_cache=False, stream_state=(None, None, None), last_chunk=True
      )
    out = mt(self.t2j(latents))
    self.assertEqual(out.shape, (1, 6 * 4 * VAE_HOP))
    self.assert_close(out, expected[:, 0], atol=5e-4, rtol=5e-3)
    # Edge-padding the latents (shape bucketing in ming_talker_decode) leaves the real samples unchanged.
    padded = jnp.pad(self.t2j(latents), ((0, 0), (0, 5), (0, 0)), mode="edge")
    padded_out = mt(padded, jnp.int32(6))
    np.testing.assert_allclose(np.asarray(padded_out[:, : out.shape[1]]), np.asarray(out), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
