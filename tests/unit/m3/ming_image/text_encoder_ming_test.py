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

"""Tests for the Ming-Image MLLM text encoder (`text_encoder_ming.py`).

Parity tests run the Ming-Image-Ant PyTorch reference (`modeling_bailing_moe_v2.py`, `modeling_bailingmm2.py`) when
its repo is available (`MING_IMAGE_REPO`, default `~/workspace/diffusion/ming/Ming-Image-Ant`); skipped otherwise.
Checkpoint-index coverage tests need the HF snapshot (`MING_IMAGE_SNAPSHOT`, default: the maxdiffusion HF cache).
"""

import glob
import json
import os
import sys
import types
import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan (shared sharding utils) needs diffusers.

# pylint: disable=wrong-import-position
from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.models.ming_image import text_encoder_ming
from maxtext.m3.models.ming_image.text_encoder_ming import (
    MingTextEncoder,
    build_t2i_inputs,
    get_t_scale_rope_index,
    mllm_param_mapping,
    video_rope_axes,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ming_reference_shims

try:
  import torch
except ImportError:
  torch = None
# pylint: enable=wrong-import-position

# The released Ming-Image-0.1-Design mllm/config.json "llm_config" (architecture keys).
FULL_LLM_CONFIG = {
    "vocab_size": 157184,
    "hidden_size": 2048,
    "intermediate_size": 5120,
    "num_hidden_layers": 20,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "hidden_act": "silu",
    "use_qkv_bias": False,
    "use_bias": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 600000,
    "rope_scaling": {"factor": None, "type": "video_rope"},
    "partial_rotary_factor": 0.5,
    "num_experts": 256,
    "num_shared_experts": 1,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "routed_scaling_factor": 2.5,
    "moe_intermediate_size": 512,
    "first_k_dense_replace": 1,
    "norm_topk_prob": True,
    "router_type": "MultiRouter",
    "use_expert_bias": True,
    "image_patch_token": 157157,
    "image_start_token": 157158,
    "image_end_token": 157159,
    "video_patch_token": 157175,
    "video_start_token": 157160,
    "video_end_token": 157161,
    "spatial_merge_size": 2,
    "tokens_per_second": 2,
    "pad_token_id": 156892,
    "max_position_embeddings": 32768,
}

# Tiny BailingMoeV2 (head_dim 128 x partial 0.5 = 64 rotary channels, as the hard-coded mrope_section needs).
TINY_LLM_CONFIG = {
    **FULL_LLM_CONFIG,
    "vocab_size": 200,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "n_group": 4,
    "topk_group": 2,
    "moe_intermediate_size": 32,
    "image_patch_token": 190,
    "image_start_token": 191,
    "image_end_token": 192,
    "video_patch_token": 196,
    "video_start_token": 193,
    "video_end_token": 194,
    "pad_token_id": 0,
}
TINY_LAYERS = (1, 2, 4)  # Plays the role of [5, 12, 20]: two intermediate layers and the final-normed output.
SCALES = (16,)


def snapshot_dir():
  """The Ming-Image-0.1-Design HF snapshot, or None."""
  path = os.environ.get("MING_IMAGE_SNAPSHOT")
  if path:
    return path if os.path.isdir(path) else None
  hits = glob.glob(
      os.path.join(
          os.environ.get("HF_HOME", "/dev/shm/hf_cache"),
          "maxdiffusion/models--inclusionAI--Ming-Image-0.1-Design/snapshots/*",
      )
  )
  return hits[0] if hits else None


def _param_shapes(model) -> dict:
  params = nnx.state(model, nnx.Param).to_pure_dict()
  return {"-".join(map(str, p)): tuple(v.shape) for p, v in flatten_dict(params).items()}


def make_reference_llm(mbm, cfg: dict, seed: int = 0):
  """A randomly initialized reference `BailingMoeV2ForCausalLM` (eager attention) for `cfg`."""
  config = mbm.BailingMoeV2Config(**cfg, _attn_implementation="eager")
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  torch.manual_seed(seed)
  model = mbm.BailingMoeV2ForCausalLM(config).eval()
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith("norm.weight"):
        p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
      elif name.endswith("expert_bias"):
        p.copy_(0.05 * torch.randn(p.shape, generator=gen))
      elif "word_embeddings" in name:
        p.copy_(torch.randn(p.shape, generator=gen))
      else:
        p.copy_(torch.randn(p.shape, generator=gen) / np.sqrt(p.shape[-1]))
  return config, model


def reference_t2i_llm_inputs(mm, config, prompts, pad_id, query_tokens):
  """The reference `get_condition_embeds_for_image_gen` preprocessing for right-padded `prompts`."""
  length = max(len(p) for p in prompts)
  ids = torch.tensor([p + [pad_id] * (length - len(p)) for p in prompts])
  attn = torch.tensor([[1] * len(p) + [0] * (length - len(p)) for p in prompts])
  cls = mm.BailingMM2NativeForConditionalGeneration
  patch = config.image_patch_token
  ids, labels, attn, gen_mask = cls.append_input_ids_with_multiscale_learnable_tokens(
      None, ids, attn, list(SCALES), patch + 1, patch + 2, patch
  )
  fake = types.SimpleNamespace(img_gen_scales=list(SCALES), query_tokens_dict={f"{s}x{s}": query_tokens for s in SCALES})
  grid, image_embeds = cls.appand_learnable_tokens(fake, ids, gen_mask, None, None, patch)
  return ids, labels, attn, gen_mask, grid, image_embeds


class BuildT2IInputsTest(unittest.TestCase):
  """`build_t2i_inputs` / `get_t_scale_rope_index` vs the reference layout and positions."""

  def test_layout_without_reference(self):
    out = build_t2i_inputs([5, 6, 7], 270, FULL_LLM_CONFIG)
    patch = FULL_LLM_CONFIG["image_patch_token"]
    self.assertEqual(out["input_ids"][:3].tolist(), [5, 6, 7])
    self.assertEqual(out["input_ids"][3], patch + 1)
    self.assertTrue(np.all(out["input_ids"][4:260] == patch))
    self.assertEqual(out["input_ids"][260], patch + 2)
    self.assertEqual(int(out["attention_mask"].sum()), 261)
    self.assertEqual(out["gen_mask"].tolist(), out["image_mask"].tolist())
    self.assertEqual(int(out["gen_mask"].sum()), 256)
    self.assertEqual(out["prompt_mask"].tolist(), [True] * 3 + [False] * 267)
    pos = out["position_ids"]
    # Text 0..3 (incl. <image_start>), image t = h = 4, w = 4 - 127 .. 4 + 128, <image_end> at t_max + 1 = 5.
    np.testing.assert_array_equal(pos[:, :4], np.tile(np.arange(4), (3, 1)))
    np.testing.assert_array_equal(pos[0, 4:260], 4)
    np.testing.assert_array_equal(pos[1, 4:260], 4)
    np.testing.assert_array_equal(pos[2, 4:260], np.arange(256) - 127 + 4)
    np.testing.assert_array_equal(pos[:, 260], 5)
    np.testing.assert_array_equal(pos[:, 261:], 1)
    with self.assertRaises(ValueError):
      build_t2i_inputs([1] * 20, 270, FULL_LLM_CONFIG)

  def test_video_rope_axes(self):
    axes = video_rope_axes()
    self.assertEqual(axes.shape, (64,))
    np.testing.assert_array_equal(axes[:24], [1, 2] * 12)
    np.testing.assert_array_equal(axes[24:32], 0)
    np.testing.assert_array_equal(axes[32:56], [1, 2] * 12)
    np.testing.assert_array_equal(axes[56:], 0)

  @unittest.skipIf(torch is None, "PyTorch is not installed")
  def test_matches_reference(self):
    ref = ming_reference_shims.import_reference("modeling_bailing_moe_v2", "modeling_bailingmm2")
    if ref is None:
      self.skipTest("Ming-Image-Ant reference repo not available")
    mbm, mm = ref
    config = mbm.BailingMoeV2Config(**FULL_LLM_CONFIG)
    prompts = [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
    query = torch.zeros(256, 8)
    ids, labels, attn, gen_mask, grid, _ = reference_t2i_llm_inputs(mm, config, prompts, 156895, query)
    pos, _ = mbm.get_t_scale_rope_index(config, ids, grid, None, attn, scale_factor=2.0)
    ref_len = ids.shape[1]
    ours = [build_t2i_inputs(p, ref_len + 7, FULL_LLM_CONFIG) for p in prompts]
    for b, out in enumerate(ours):
      valid = attn[b].numpy() == 1
      np.testing.assert_array_equal(out["input_ids"][:ref_len][valid], ids[b].numpy()[valid])
      np.testing.assert_array_equal(out["attention_mask"][:ref_len], attn[b].numpy())
      np.testing.assert_array_equal(out["attention_mask"][ref_len:], 0)
      np.testing.assert_array_equal(out["position_ids"][:, :ref_len], pos[:, b].numpy())
      np.testing.assert_array_equal(out["gen_mask"][:ref_len], gen_mask[b].numpy().astype(bool))
      np.testing.assert_array_equal(out["image_mask"][:ref_len], (ids[b] == config.image_patch_token).numpy())
      prompt_mask = ((labels[b] < 0) & (attn[b] == 1)).numpy()
      np.testing.assert_array_equal(out["prompt_mask"][:ref_len], prompt_mask)
      self.assertFalse(out["prompt_mask"][ref_len:].any() or out["gen_mask"][ref_len:].any())
    # Batched numpy port == reference on the reference's own (right-padded) batch.
    ours_batched = get_t_scale_rope_index(
        ids.numpy(), attn.numpy(), grid.tolist(), image_start_token=157158, image_patch_token=157157
    )
    np.testing.assert_array_equal(ours_batched, pos.numpy())

  @unittest.skipIf(torch is None, "PyTorch is not installed")
  def test_t2i_prompt_tokens(self):
    snap = snapshot_dir()
    if snap is None or not os.path.isfile(os.path.join(snap, "mllm", "tokenizer.json")):
      self.skipTest("Ming-Image-0.1-Design snapshot not available")
    text = text_encoder_ming.t2i_prompt_text("A red fox in the snow.")
    self.assertEqual(
        text,
        "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>A red fox in the"
        " snow.<|role_end|><role>ASSISTANT</role>",
    )
    tok = text_encoder_ming.load_tokenizer(os.path.join(snap, "mllm"))
    ids = text_encoder_ming.format_t2i_prompt("A red fox in the snow.", tok)
    self.assertEqual(ids.count(156895), 2)  # Two <|role_end|> (eos), no BOS.
    self.assertNotEqual(ids[0], 156891)
    ref = ming_reference_shims.import_reference("tokenization_bailing")
    if ref is not None:
      ref[0].BailingTokenizer.verbose = False  # Attribute read before PreTrainedTokenizerBase sets it (transformers 5).
      try:
        ref_tok = ref[0].BailingTokenizer.from_pretrained(os.path.join(snap, "mllm"))
      except Exception as e:  # pylint: disable=broad-exception-caught
        self.skipTest(f"reference tokenizer does not load under this transformers: {e}")
      self.assertEqual(ref_tok(text)["input_ids"], ids)


class MingTextEncoderTest(unittest.TestCase):

  @unittest.skipIf(torch is None, "PyTorch is not installed")
  def test_parity_with_reference(self):
    ref = ming_reference_shims.import_reference("modeling_bailing_moe_v2", "modeling_bailingmm2")
    if ref is None:
      self.skipTest("Ming-Image-Ant reference repo not available")
    mbm, mm = ref
    config, ref_model = make_reference_llm(mbm, TINY_LLM_CONFIG)
    gen = torch.Generator().manual_seed(1)
    query = torch.randn(256, TINY_LLM_CONFIG["hidden_size"], generator=gen)
    prompts = [list(range(3, 12)), list(range(40, 54))]
    ids, _, attn, _, grid, image_embeds = reference_t2i_llm_inputs(mm, config, prompts, 0, query)
    with torch.no_grad():
      embeds, image_mask, _ = ref_model.model.prompt_wrap_navit(
          ids.clip(0, config.vocab_size - 1), config, query_embeds_image=image_embeds
      )
      out = ref_model.forward(
          input_ids=ids,
          attention_mask=attn,
          position_ids=None,
          past_key_values=None,
          inputs_embeds=embeds,
          image_grid_thw=grid,
          use_cache=False,
          image_mask=image_mask,
          audio_mask=None,
          output_hidden_states=True,
      )
    ref_hidden = [out.hidden_states[i].numpy() for i in TINY_LAYERS]
    # hidden_states[-1] is after the final norm (= hidden_states[num_layers]).
    np.testing.assert_array_equal(out.hidden_states[-1].numpy(), ref_hidden[-1])

    model = MingTextEncoder(llm_config={"llm_config": TINY_LLM_CONFIG}, rngs=nnx.Rngs(0))
    ming_reference_shims.load_with_mapping(
        model, mllm_param_mapping({"llm_config": TINY_LLM_CONFIG}), ming_reference_shims.torch_state(ref_model, "model.")
    )
    static_len = ids.shape[1] + 5
    batch = [build_t2i_inputs(p, static_len, TINY_LLM_CONFIG) for p in prompts]
    inputs = {k: np.stack([b[k] for b in batch], axis=-2 if k == "position_ids" else 0) for k in batch[0]}
    self.assertEqual(inputs["position_ids"].shape, (3, 2, static_len))

    @jax.jit
    def run(model, inputs):
      return model(
          inputs["input_ids"],
          inputs["position_ids"],
          inputs["attention_mask"],
          inputs["image_mask"],
          query_embeds=jnp.asarray(query.numpy()),
          output_layers=TINY_LAYERS,
      )

    selected, last = run(model, inputs)
    np.testing.assert_array_equal(np.asarray(last), np.asarray(selected[-1]))
    valid = attn.numpy() == 1
    for layer, ours, theirs in zip(TINY_LAYERS, selected, ref_hidden):
      ours = np.asarray(ours)[:, : ids.shape[1]][valid]
      theirs = theirs[valid]
      diff = np.max(np.abs(ours - theirs))
      print(f"hidden_states[{layer}] max abs diff {diff:.3e} (max |ref| {np.max(np.abs(theirs)):.2f})")
      np.testing.assert_allclose(ours, theirs, atol=1e-4, rtol=1e-4, err_msg=f"hidden_states[{layer}]")

  def test_image_gate_routing(self):
    """Query-token positions use `image_gate`: zeroing it changes only those positions' outputs."""
    model = MingTextEncoder(llm_config=TINY_LLM_CONFIG, rngs=nnx.Rngs(0))
    inputs = build_t2i_inputs(list(range(3, 12)), 280, TINY_LLM_CONFIG)
    inputs = {k: v[:, None] if k == "position_ids" else v[None] for k, v in inputs.items()}
    query = jnp.ones((256, TINY_LLM_CONFIG["hidden_size"])) * 0.1

    def run():
      return model(
          inputs["input_ids"],
          inputs["position_ids"],
          inputs["attention_mask"],
          inputs["image_mask"],
          query_embeds=query,
          output_layers=(2,),
      )[0][0]

    before = np.asarray(run())
    for layer in model.layers:
      if layer.is_moe:
        gate = layer.mlp.image_gate.weight
        gate[...] = jax.random.normal(jax.random.key(3), gate.shape)
    after = np.asarray(run())
    text = ~inputs["gen_mask"][0] & (np.arange(280) < 9)  # Prompt tokens precede the query tokens (causal).
    np.testing.assert_array_equal(before[0][text], after[0][text])
    self.assertGreater(np.max(np.abs(before[0][inputs["gen_mask"][0]] - after[0][inputs["gen_mask"][0]])), 1e-3)


class FullSizeCoverageTest(unittest.TestCase):
  """Abstract full-size model: the mapping covers exactly the params; HF keys exist in the checkpoint index."""

  def test_mapping_covers_params(self):
    cfg = {"llm_config": FULL_LLM_CONFIG}
    model = nnx.eval_shape(lambda: MingTextEncoder(rngs=nnx.Rngs(0), **cfg))
    shapes = _param_shapes(model)
    mapping, hooks = mllm_param_mapping(cfg)
    self.assertEqual(set(mapping), set(shapes))
    self.assertTrue(set(hooks) <= set(mapping))
    self.assertEqual(shapes["layers-1-mlp-experts-gate_proj"], (256, 2048, 512))
    self.assertEqual(shapes["layers-1-attention-query-kernel"], (2048, 2048))
    self.assertEqual(shapes["layers-1-attention-key-kernel"], (2048, 512))
    self.assertEqual(shapes["layers-0-mlp-gate_proj-kernel"], (2048, 5120))
    self.assertEqual(len(mapping["layers-19-mlp-experts-down_proj"]), 256)
    # Split hooks on the fused [(16 + 2 * 4) * 128, 2048] query_key_value weight.
    fused = np.arange(3072 * 4, dtype=np.float32).reshape(3072, 4)
    q = hooks["layers-0-attention-query-kernel"](fused, (4, 2048))
    k = hooks["layers-0-attention-key-kernel"](fused, (4, 512))
    v = hooks["layers-0-attention-value-kernel"](fused, (4, 512))
    np.testing.assert_array_equal(np.concatenate([q, k, v], axis=1), fused.T)

  def test_hf_keys_in_index(self):
    snap = snapshot_dir()
    index = os.path.join(snap, "mllm", "model.safetensors.index.json") if snap else ""
    if not os.path.isfile(index):
      self.skipTest("Ming-Image-0.1-Design snapshot not available")
    with open(index, "r", encoding="utf-8") as f:
      weight_map = json.load(f)["weight_map"]
    with open(os.path.join(snap, "mllm", "config.json"), "r", encoding="utf-8") as f:
      hf_config = json.load(f)
    mapping, _ = mllm_param_mapping({"llm_config": hf_config["llm_config"]})
    hf_keys = set()
    for v in mapping.values():
      hf_keys.update(v if isinstance(v, list) else [v])
    self.assertEqual(hf_keys - set(weight_map), set())
    model = nnx.eval_shape(lambda: MingTextEncoder(rngs=nnx.Rngs(0), llm_config=hf_config["llm_config"]))
    self.assertEqual(set(mapping), set(_param_shapes(model)))


if __name__ == "__main__":
  unittest.main()
