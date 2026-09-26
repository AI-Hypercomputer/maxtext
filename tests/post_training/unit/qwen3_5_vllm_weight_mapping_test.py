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

"""CPU tests for the MaxText Qwen3.5 -> vLLM (torchax) weight mapping.

The canonical re-layouts are cross-checked against MaxText's own HF export hooks
(`QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(saving_to_hf=True)`), and the mapping is checked
to cover every produced tensor and to resolve real vLLM parameter names uniquely.
"""

import re
import unittest
from types import SimpleNamespace

import pytest

import numpy as np

from maxtext.checkpoint_conversion.utils.param_mapping import QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN
from maxtext.integration.tunix.weight_mapping import StandaloneVllmWeightMapping
from maxtext.integration.tunix.weight_mapping.qwen3 import QWEN3_VLLM_MAPPING
from maxtext.integration.tunix.weight_mapping.qwen3_5 import (
    QWEN3_5_VLLM_MAPPING,
    qwen3_5_maxtext_to_vllm_canonical,
)


pytestmark = [pytest.mark.post_training]

# Tiny geometry: 8 layers in cycles of 4 (layers 3 and 7 are full attention).
D, H, HD, HKV = 64, 4, 16, 2
HK, HV, DK, DV = 4, 8, 8, 8
E, F, V, K = 6, 24, 100, 4
N_LAYERS = 8
HF_CFG = {
    "text_config": {
        "linear_num_key_heads": HK,
        "linear_num_value_heads": HV,
        "linear_key_head_dim": DK,
        "linear_value_head_dim": DV,
        "num_hidden_layers": N_LAYERS,
    }
}


def _make_state(rng):
  def r(*shape):
    return rng.standard_normal(shape).astype(np.float32)

  state = {
      "params.params.token_embedder.embedding": r(V, D),
      "params.params.decoder.decoder_norm.scale": r(D),
      "params.params.decoder.logits_dense.kernel": r(D, V),
  }
  for i in range(N_LAYERS):
    p = f"params.params.decoder.layers_{i}."
    state[p + "input_layernorm.scale"] = r(D)
    state[p + "post_attention_layernorm.scale"] = r(D)
    if i % 4 == 3:
      state[p + "attention.attention.query.kernel"] = r(D, H, 2 * HD)
      state[p + "attention.attention.key.kernel"] = r(D, HKV, HD)
      state[p + "attention.attention.value.kernel"] = r(D, HKV, HD)
      state[p + "attention.attention.out.kernel"] = r(H * HD, D)
      state[p + "attention.attention.query_norm.scale"] = r(HD)
      state[p + "attention.attention.key_norm.scale"] = r(HD)
    else:
      state[p + "attention.in_proj_qkvz.kernel"] = r(D, HK * (2 * DK + 2 * (HV // HK) * DV))
      state[p + "attention.in_proj_ba.kernel"] = r(D, 2 * HV)
      state[p + "attention.conv1d.kernel"] = r(K, 1, 2 * HK * DK + HV * DV)
      state[p + "attention.A_log"] = r(HV)
      state[p + "attention.dt_bias"] = r(HV)
      state[p + "attention.norm.rms_norm.scale"] = r(DV)
      state[p + "attention.out_proj.kernel"] = r(HV * DV, D)
    state[p + "mlp.routed_experts.gate.kernel"] = r(D, E)
    state[p + "mlp.routed_experts.wi_0"] = r(E, D, F)
    state[p + "mlp.routed_experts.wi_1"] = r(E, D, F)
    state[p + "mlp.routed_experts.wo"] = r(E, F, D)
    state[p + "mlp.shared_expert.wi_0.kernel"] = r(D, F)
    state[p + "mlp.shared_expert.wi_1.kernel"] = r(D, F)
    state[p + "mlp.shared_expert.wo.kernel"] = r(F, D)
    state[p + "mlp.shared_expert_gate.kernel"] = r(D, 1)
  return state


class Qwen35VllmWeightMappingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.state = _make_state(np.random.default_rng(0))
    self.out = qwen3_5_maxtext_to_vllm_canonical(self.state, HF_CFG)

  def test_registry_prefers_qwen3_5_over_qwen3(self):
    self.assertIs(StandaloneVllmWeightMapping()["qwen3.5-35b-a3b"], QWEN3_5_VLLM_MAPPING)
    self.assertIs(StandaloneVllmWeightMapping()["qwen3-8b"], QWEN3_VLLM_MAPPING)

  def test_geometry_inferred_from_shapes_matches_config(self):
    inferred = qwen3_5_maxtext_to_vllm_canonical(self.state, None)
    self.assertEqual(set(inferred), set(self.out))
    for k in self.out:
      np.testing.assert_array_equal(np.asarray(inferred[k]), np.asarray(self.out[k]), err_msg=k)

  def test_relayouts_match_maxtext_hf_export_hooks(self):
    hooks = QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        HF_CFG,
        SimpleNamespace(inhomogeneous_layer_cycle_interval=4, use_multimodal=False),
        scan_layers=False,
        saving_to_hf=True,
    )
    s = self.state
    p0, p3 = "params-decoder-layers_0-", "params-decoder-layers_3-"
    l0, l3 = "params.params.decoder.layers_0.", "params.params.decoder.layers_3."

    qkv, z = hooks[p0 + "attention-in_proj_qkvz-kernel"](s[l0 + "attention.in_proj_qkvz.kernel"])
    np.testing.assert_array_equal(np.asarray(self.out["base.decoder.layers.0.attention.in_proj_qkvz"]), np.concatenate([qkv, z], 0))
    b, a = hooks[p0 + "attention-in_proj_ba-kernel"](s[l0 + "attention.in_proj_ba.kernel"])
    np.testing.assert_array_equal(np.asarray(self.out["base.decoder.layers.0.attention.in_proj_ba"]), np.concatenate([b, a], 0))
    conv = hooks[p0 + "attention-conv1d-kernel"](s[l0 + "attention.conv1d.kernel"])
    np.testing.assert_array_equal(np.asarray(self.out["base.decoder.layers.0.attention.conv1d"]), conv)

    q = hooks[p3 + "attention-attention-query-kernel"](s[l3 + "attention.attention.query.kernel"], (H * 2 * HD, D))
    k = hooks[p3 + "attention-attention-key-kernel"](s[l3 + "attention.attention.key.kernel"], (HKV * HD, D))
    v = hooks[p3 + "attention-attention-value-kernel"](s[l3 + "attention.attention.value.kernel"], (HKV * HD, D))
    np.testing.assert_array_equal(np.asarray(self.out["base.decoder.layers.3.attention.qkv_proj"]), np.concatenate([q, k, v], 0))
    o = hooks[p3 + "attention-attention-out-kernel"](s[l3 + "attention.attention.out.kernel"], (D, H * HD))
    np.testing.assert_array_equal(np.asarray(self.out["base.decoder.layers.3.attention.o_proj"]), o)

    gu = hooks[(p0 + "mlp-routed_experts-wi_0", p0 + "mlp-routed_experts-wi_1")](
        (s[l0 + "mlp.routed_experts.wi_0"], s[l0 + "mlp.routed_experts.wi_1"])
    )
    self.assertEqual(gu.shape, (E, 2 * F, D))
    np.testing.assert_array_equal(np.asarray(self.out["base.decoder.layers.0.mlp.experts.w13"]), gu)
    dn = hooks[p0 + "mlp-routed_experts-wo"](s[l0 + "mlp.routed_experts.wo"])
    self.assertEqual(dn.shape, (E, D, F))
    np.testing.assert_array_equal(np.asarray(self.out["base.decoder.layers.0.mlp.experts.w2"]), dn)

  def test_canonical_vllm_layouts(self):
    out = self.out
    self.assertEqual(out["base.decoder.logits_dense.kernel"].shape, (V, D))
    self.assertEqual(out["base.decoder.layers.0.mlp.gate"].shape, (E, D))
    self.assertEqual(out["base.decoder.layers.0.mlp.shared_expert.gate_up_proj"].shape, (2 * F, D))
    self.assertEqual(out["base.decoder.layers.0.mlp.shared_expert.down_proj"].shape, (D, F))
    self.assertEqual(out["base.decoder.layers.0.mlp.shared_expert_gate"].shape, (1, D))
    self.assertEqual(out["base.decoder.layers.0.attention.gdn_out_proj"].shape, (D, HV * DV))
    # shared expert gate_up is [gate | up] rows, i.e. wi_0 first.
    np.testing.assert_array_equal(
        np.asarray(out["base.decoder.layers.0.mlp.shared_expert.gate_up_proj"])[:F],
        self.state["params.params.decoder.layers_0.mlp.shared_expert.wi_0.kernel"].T,
    )

  def test_mapping_covers_all_produced_keys_and_resolves_vllm_names(self):
    mapping = QWEN3_5_VLLM_MAPPING.to_hf_mapping()
    src_patterns = [re.compile("^" + re.escape(s).replace(r"\*", r"(\d+)") + "$") for s in mapping]
    for key in self.out:
      self.assertTrue(any(p.match(key) for p in src_patterns), f"produced key without mapping: {key}")
    for s in mapping:
      gdn = any(t in s for t in ("in_proj", "gdn", "A_log", "dt_bias", "attention.norm", "conv1d"))
      concrete = s.replace("*", "0" if gdn else "3")
      self.assertIn(concrete, self.out, f"mapping source never produced: {s}")

    vllm_names = [
        "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.3.self_attn.qkv_proj.weight",
        "language_model.model.layers.0.mlp.experts.routed_experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
    ]
    for n in vllm_names:
      hits = [(s, m) for s, (t, _) in mapping.items() if (m := re.compile("^" + t + "$").match(n))]
      self.assertEqual(len(hits), 1, (n, hits))
      self.assertEqual(len(hits[0][1].groups()), 1 if ".layers." in n else 0, n)

  def test_scanned_blocks_give_same_result(self):
    scanned = {k: v for k, v in self.state.items() if "decoder.layers_" not in k}
    for b in range(4):
      for name in [k for k in self.state if f"decoder.layers_{b}." in k]:
        rest = name.split(f"decoder.layers_{b}.")[1]
        stacked = np.stack([self.state[f"params.params.decoder.layers_{b + 4 * j}.{rest}"] for j in range(2)], axis=1)
        scanned[f"params.params.decoder.layers.layer_{b}.{rest}"] = stacked
    out_scanned = qwen3_5_maxtext_to_vllm_canonical(scanned, HF_CFG)
    self.assertEqual(set(out_scanned), set(self.out))
    for k in self.out:
      np.testing.assert_array_equal(np.asarray(out_scanned[k]), np.asarray(self.out[k]), err_msg=k)


if __name__ == "__main__":
  unittest.main()
