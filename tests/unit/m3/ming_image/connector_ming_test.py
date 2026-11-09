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

"""Tests for the Ming-Image connector and condition head (`connector_ming.py`).

Parity tests run transformers' Qwen2 and the Ming-Image-Ant reference `get_condition_embeds_for_image_gen` when the
repo is available (`MING_IMAGE_REPO`); skipped otherwise.
"""

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
import numpy as np

from maxtext.m3.models.ming_image.connector_ming import (
    MingConditioner,
    MingConnector,
    conditioner_param_mapping,
    connector_param_mapping,
    get_condition_embeds,
)
from maxtext.m3.models.ming_image.text_encoder_ming import MingTextEncoder, build_t2i_inputs, mllm_param_mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ming_reference_shims
import text_encoder_ming_test as te_test

try:
  import torch
except ImportError:
  torch = None
# pylint: enable=wrong-import-position

# The released connector/config.json (Qwen2) and mlp/config.json.
FULL_CONNECTOR_CONFIG = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
}
FULL_MLP_CONFIG = {
    "use_identity_mlp": True,
    "use_vlm_directvlm_condition": True,
    "use_learnable_token_condition": True,
    "diffusion_c_input_dim": 2560,
    "text_encoder_norm": False,
    "connector_norm": False,
    "img_gen_scales": [16],
    "selected_hidden_states_layers": [5, 12, 20],
    "diffusion_inner_dim": 3840,
}
TINY_CONNECTOR_CONFIG = {
    **FULL_CONNECTOR_CONFIG,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}
TINY_MLP_CONFIG = {
    **FULL_MLP_CONFIG,
    "diffusion_c_input_dim": 40,
    "diffusion_inner_dim": 24,
    "selected_hidden_states_layers": list(te_test.TINY_LAYERS),
    "llm_hidden_size": te_test.TINY_LLM_CONFIG["hidden_size"],
    "connector_hidden_size": TINY_CONNECTOR_CONFIG["hidden_size"],
}


def _param_shapes(model) -> dict:
  params = nnx.state(model, nnx.Param).to_pure_dict()
  return {"-".join(map(str, p)): tuple(v.shape) for p, v in flatten_dict(params).items()}


def make_reference_connector(cfg: dict, seed: int = 0):
  """A random transformers `Qwen2ForCausalLM` set up like `load_image_gen_modules` (non-causal layers)."""
  import transformers  # pylint: disable=import-outside-toplevel

  config = transformers.Qwen2Config(
      **{k: v for k, v in cfg.items() if k != "hidden_act"}, vocab_size=64, tie_word_embeddings=True
  )
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  torch.manual_seed(seed)
  model = transformers.Qwen2ForCausalLM(config).eval()
  gen = torch.Generator().manual_seed(seed + 1)
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith("norm.weight"):
        p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
      else:
        p.copy_(torch.randn(p.shape, generator=gen) / np.sqrt(p.shape[-1]))
  for layer in model.model.layers:
    layer.self_attn.is_causal = False
  return model


class MingConnectorTest(unittest.TestCase):

  @unittest.skipIf(torch is None, "PyTorch is not installed")
  def test_parity_with_qwen2_bidirectional(self):
    ref = make_reference_connector(TINY_CONNECTOR_CONFIG)
    x = torch.randn(2, 19, TINY_CONNECTOR_CONFIG["hidden_size"], generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
      # Exactly the reference call: all-ones 4D mask, `hidden_states[-1]`.
      out = ref(inputs_embeds=x, attention_mask=torch.ones(2, 1, 19, 19), output_hidden_states=True)
      expected = out.hidden_states[-1].numpy()
      # hidden_states[-1] is after the final norm (= Qwen2Model.last_hidden_state).
      last = ref.model(inputs_embeds=x, attention_mask=torch.ones(2, 1, 19, 19)).last_hidden_state.numpy()
    np.testing.assert_allclose(expected, last, atol=1e-6)
    model = MingConnector(**TINY_CONNECTOR_CONFIG, rngs=nnx.Rngs(0))
    ming_reference_shims.load_with_mapping(
        model, connector_param_mapping(TINY_CONNECTOR_CONFIG), ming_reference_shims.torch_state(ref)
    )
    ours = np.asarray(jax.jit(lambda m, v: m(v))(model, x.numpy()))
    diff = np.max(np.abs(ours - expected))
    print(f"connector max abs diff {diff:.3e}")
    np.testing.assert_allclose(ours, expected, atol=1e-4, rtol=1e-4)
    # Bidirectional: the first token depends on the last one.
    x2 = x.numpy().copy()
    x2[:, -1] += 1.0
    self.assertGreater(np.max(np.abs(np.asarray(model(x2))[:, 0] - ours[:, 0])), 1e-4)


class ConditionEmbedsTest(unittest.TestCase):
  """End-to-end `get_condition_embeds_for_image_gen` tail parity (cap_feats, cap_feats_2) on tiny configs."""

  @unittest.skipIf(torch is None, "PyTorch is not installed")
  def test_parity_with_reference(self):
    ref = ming_reference_shims.import_reference("modeling_bailing_moe_v2", "modeling_bailingmm2")
    if ref is None:
      self.skipTest("Ming-Image-Ant reference repo not available")
    mbm, mm = ref
    from diffusers.models.normalization import RMSNorm  # pylint: disable=import-outside-toplevel

    hidden, conn_hidden = te_test.TINY_LLM_CONFIG["hidden_size"], TINY_CONNECTOR_CONFIG["hidden_size"]
    llm_config, llm = te_test.make_reference_llm(mbm, te_test.TINY_LLM_CONFIG)
    connector = make_reference_connector(TINY_CONNECTOR_CONFIG, seed=3)
    gen = torch.Generator().manual_seed(7)
    torch.manual_seed(7)
    directvlm_dim = hidden * len(te_test.TINY_LAYERS)
    proj_directvlm = torch.nn.Sequential(
        RMSNorm(directvlm_dim, eps=1e-5), torch.nn.Linear(directvlm_dim, TINY_MLP_CONFIG["diffusion_inner_dim"])
    )
    with torch.no_grad():
      proj_directvlm[0].weight.copy_(1.0 + 0.2 * torch.randn(directvlm_dim, generator=gen))
    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(llm_config=llm_config),
        model=llm,
        img_gen_scales=[16],
        scale_indices=[256],
        query_tokens_dict=torch.nn.ParameterDict({"16x16": torch.nn.Parameter(torch.randn(256, hidden, generator=gen))}),
        use_vlm_directvlm_condition=True,
        selected_hidden_states_layers=list(te_test.TINY_LAYERS),
        proj_directvlm=proj_directvlm,
        use_learnable_token_condition=True,
        proj_in=torch.nn.Linear(hidden, conn_hidden),
        connector=connector,
        proj_out=torch.nn.Linear(conn_hidden, TINY_MLP_CONFIG["diffusion_c_input_dim"]),
        connector_norm=False,
    )
    cls = mm.BailingMM2NativeForConditionalGeneration
    for name in ("append_input_ids_with_multiscale_learnable_tokens", "appand_learnable_tokens"):
      setattr(fake, name, types.MethodType(getattr(cls, name), fake))
    prompts = [list(range(3, 12)), list(range(40, 54))]
    n_max = max(len(p) for p in prompts)
    ids = torch.tensor([p + [0] * (n_max - len(p)) for p in prompts])
    attn = torch.tensor([[1] * len(p) + [0] * (n_max - len(p)) for p in prompts])
    with torch.no_grad():
      ref_cap, ref_cap2 = cls.get_condition_embeds_for_image_gen(fake, ids, attn, None, None, False, None, None)
    ref_cap, ref_cap2 = ref_cap.numpy(), ref_cap2.numpy()
    self.assertEqual(ref_cap2.shape[1], n_max)  # The reference truncates to the longest prompt.

    text_encoder = MingTextEncoder(llm_config=te_test.TINY_LLM_CONFIG, rngs=nnx.Rngs(0))
    ming_reference_shims.load_with_mapping(
        text_encoder, mllm_param_mapping(te_test.TINY_LLM_CONFIG), ming_reference_shims.torch_state(llm, "model.")
    )
    our_connector = MingConnector(**TINY_CONNECTOR_CONFIG, rngs=nnx.Rngs(0))
    ming_reference_shims.load_with_mapping(
        our_connector, connector_param_mapping(TINY_CONNECTOR_CONFIG), ming_reference_shims.torch_state(connector)
    )
    conditioner = MingConditioner(**TINY_MLP_CONFIG, rngs=nnx.Rngs(0))
    mlp_state = {"query_tokens_dict.16x16": fake.query_tokens_dict["16x16"].detach().numpy()}
    for name in ("proj_in", "proj_out", "proj_directvlm"):
      mlp_state.update(ming_reference_shims.torch_state(getattr(fake, name), f"{name}."))
    ming_reference_shims.load_with_mapping(conditioner, conditioner_param_mapping(TINY_MLP_CONFIG), mlp_state)

    static_len = 300
    batch = [build_t2i_inputs(p, static_len, te_test.TINY_LLM_CONFIG) for p in prompts]
    inputs = {k: np.stack([b[k] for b in batch], axis=-2 if k == "position_ids" else 0) for k in batch[0]}
    cap, cap2 = jax.jit(get_condition_embeds)(text_encoder, our_connector, conditioner, inputs)
    cap, cap2 = np.asarray(cap), np.asarray(cap2)
    self.assertEqual(cap.shape, ref_cap.shape)
    self.assertEqual(cap2.shape, (2, static_len, TINY_MLP_CONFIG["diffusion_inner_dim"]))
    print(f"cap_feats max abs diff {np.max(np.abs(cap - ref_cap)):.3e} (max |ref| {np.max(np.abs(ref_cap)):.2f})")
    print(f"cap_feats_2 max abs diff {np.max(np.abs(cap2[:, :n_max] - ref_cap2)):.3e}")
    np.testing.assert_allclose(cap, ref_cap, atol=1e-4, rtol=1e-4)
    # First max(prompt length) rows == the reference (incl. bias rows past the shorter prompt); the rest are the bias.
    np.testing.assert_allclose(cap2[:, :n_max], ref_cap2, atol=1e-4, rtol=1e-4)
    bias = proj_directvlm[1].bias.detach().numpy()
    np.testing.assert_allclose(cap2[:, n_max:], np.broadcast_to(bias, cap2[:, n_max:].shape), atol=1e-6)


class FullSizeCoverageTest(unittest.TestCase):
  """Abstract full-size models: mappings cover exactly the params; HF keys exist in the checkpoint."""

  def test_connector_mapping(self):
    model = nnx.eval_shape(lambda: MingConnector(rngs=nnx.Rngs(0), **FULL_CONNECTOR_CONFIG))
    shapes = _param_shapes(model)
    mapping, hooks = connector_param_mapping(FULL_CONNECTOR_CONFIG)
    self.assertEqual(set(mapping), set(shapes))
    self.assertTrue(set(hooks) <= set(mapping))
    self.assertEqual(shapes["layers-0-self_attn-k_proj-kernel"], (1536, 256))

  def test_conditioner_mapping(self):
    model = nnx.eval_shape(lambda: MingConditioner(rngs=nnx.Rngs(0), **FULL_MLP_CONFIG))
    shapes = _param_shapes(model)
    mapping, _ = conditioner_param_mapping(FULL_MLP_CONFIG)
    self.assertEqual(set(mapping), set(shapes))
    self.assertEqual(shapes["query_tokens"], (256, 2048))
    self.assertEqual(shapes["proj_directvlm-1-kernel"], (6144, 3840))
    self.assertEqual(shapes["proj_out-kernel"], (1536, 2560))

  def test_hf_keys_exist(self):
    snap = te_test.snapshot_dir()
    if snap is None:
      self.skipTest("Ming-Image-0.1-Design snapshot not available")
    with open(os.path.join(snap, "connector", "model.safetensors.index.json"), "r", encoding="utf-8") as f:
      connector_keys = set(json.load(f)["weight_map"])
    with open(os.path.join(snap, "connector", "config.json"), "r", encoding="utf-8") as f:
      connector_cfg = json.load(f)
    self.assertEqual(set(connector_param_mapping(connector_cfg)[0].values()) - connector_keys, set())
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel

    with safe_open(os.path.join(snap, "mlp", "model.safetensors"), "np") as f:
      mlp_shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    with open(os.path.join(snap, "mlp", "config.json"), "r", encoding="utf-8") as f:
      mlp_cfg = json.load(f)
    mapping, hooks = conditioner_param_mapping(mlp_cfg)
    model = nnx.eval_shape(lambda: MingConditioner(rngs=nnx.Rngs(0), **mlp_cfg))
    for key, shape in _param_shapes(model).items():
      hf_shape = mlp_shapes[mapping[key]]
      self.assertEqual(hf_shape[::-1] if key in hooks else hf_shape, shape, key)


if __name__ == "__main__":
  unittest.main()
