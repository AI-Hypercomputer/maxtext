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

"""Config audit: production nemotron3-nano-omni-30b-a3b.yml against the pinned HF config.json.

Every MaxText value is either copied from a config.json key, derived from config.json keys by a
stated formula, or hardcoded with a cited modeling_nemotron_h.py line. Every llm_config key is
either consumed or listed as dead with a reason, so a new HF key cannot be silently ignored.
"""

import copy
import os
import tempfile
import types
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np

from maxtext.checkpoint_conversion.utils.hf_model_configs import nemotron3_nano_omni_30b_a3b_dict
from maxtext.checkpoint_conversion.utils.hf_shape import NEMOTRON3_HF_WEIGHTS_TO_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import (
    NEMOTRON3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from maxtext.checkpoint_conversion.utils.utils import process_maxtext_param
from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path

MODELING = "modeling_nemotron_h.py"
MODEL_NAME = "nemotron3-nano-omni-30b-a3b"

# (MaxText key, expected value as a function of llm_config, source citation)
AUDIT = [
    (
        "decoder_block",
        lambda c: "nemotron3",
        "llm_config.architectures=[NemotronHForCausalLM], model_type=nemotron_h",
    ),
    ("vocab_size", lambda c: c["vocab_size"], "llm_config.vocab_size"),
    ("base_emb_dim", lambda c: c["hidden_size"], "llm_config.hidden_size"),
    ("base_num_decoder_layers", lambda c: c["num_hidden_layers"], "llm_config.num_hidden_layers"),
    ("hybrid_override_pattern", lambda c: c["hybrid_override_pattern"], "llm_config.hybrid_override_pattern"),
    (
        "normalization_layer_epsilon",
        lambda c: c["layer_norm_epsilon"],
        "llm_config.layer_norm_epsilon (= norm_eps)",
    ),
    ("logits_via_embedding", lambda c: c["tie_word_embeddings"], "llm_config.tie_word_embeddings"),
    ("base_num_query_heads", lambda c: c["num_attention_heads"], "llm_config.num_attention_heads"),
    ("base_num_kv_heads", lambda c: c["num_key_value_heads"], "llm_config.num_key_value_heads"),
    ("head_dim", lambda c: c["head_dim"], "llm_config.head_dim"),
    ("attention_bias", lambda c: c["attention_bias"], "llm_config.attention_bias"),
    ("use_qk_norm", lambda c: False, f"{MODELING}:938-941 (q/k/v/o projections only, no q_norm/k_norm)"),
    ("float32_logits", lambda c: True, f"{MODELING}:917 softmax(..., dtype=torch.float32)"),
    ("float32_qk_product", lambda c: False, f"{MODELING}:913 q @ k^T in the model dtype"),
    ("mamba_num_heads", lambda c: c["mamba_num_heads"], "llm_config.mamba_num_heads"),
    ("mamba_head_dim", lambda c: c["mamba_head_dim"], "llm_config.mamba_head_dim"),
    ("mamba_ssm_state_size", lambda c: c["ssm_state_size"], "llm_config.ssm_state_size"),
    ("mamba_n_groups", lambda c: c["n_groups"], "llm_config.n_groups"),
    ("mamba_conv_kernel", lambda c: c["conv_kernel"], "llm_config.conv_kernel"),
    ("mamba_conv_bias", lambda c: c["use_conv_bias"], "llm_config.use_conv_bias"),
    ("mamba_chunk_size", lambda c: c["chunk_size"], "llm_config.chunk_size"),
    (
        "mamba_time_step_limit",
        lambda c: list(c["time_step_limit"]),
        "llm_config.time_step_limit",
    ),
    ("num_experts", lambda c: c["n_routed_experts"], "llm_config.n_routed_experts"),
    ("num_experts_per_tok", lambda c: c["num_experts_per_tok"], "llm_config.num_experts_per_tok"),
    ("base_moe_mlp_dim", lambda c: c["moe_intermediate_size"], "llm_config.moe_intermediate_size"),
    ("base_mlp_dim", lambda c: c["intermediate_size"], "llm_config.intermediate_size"),
    (
        "shared_experts",
        lambda c: c["moe_shared_expert_intermediate_size"] // c["moe_intermediate_size"],
        "derived: moe_shared_expert_intermediate_size / moe_intermediate_size",
    ),
    (
        "mlp_activations",
        lambda c: [c["mlp_hidden_act"]],
        "llm_config.mlp_hidden_act (one activation = no gate matrix)",
    ),
    ("mlp_bias", lambda c: c["mlp_bias"], "llm_config.mlp_bias"),
    ("routed_score_func", lambda c: "sigmoid", f"{MODELING}:782 router_logits.sigmoid()"),
    ("routed_bias", lambda c: True, f"{MODELING}:783,847 e_score_correction_bias added for the choice"),
    ("routed_scaling_factor", lambda c: c["routed_scaling_factor"], "llm_config.routed_scaling_factor"),
    ("norm_topk_prob", lambda c: c["norm_topk_prob"], "llm_config.norm_topk_prob"),
    ("n_routing_groups", lambda c: c["n_group"], "llm_config.n_group"),
    ("topk_routing_group", lambda c: c["topk_group"], "llm_config.topk_group"),
    ("float32_gate_logits", lambda c: True, f"{MODELING}:780 gate logits cast to float32 before routing"),
    ("float32_weight_sum", lambda c: True, f"{MODELING}:853 weights = weights / weight_sum in float32"),
]

HARDCODED_HF = {
    "architectures": (["NemotronHForCausalLM"], "exact architecture string"),
    "model_type": ("nemotron_h", "model family slug"),
    "mamba_hidden_act": ("silu", "SSM gate activation is silu"),
    "torch_dtype": ("bfloat16", "HF weights distributed as bf16"),
    "rescale_prenorm_residual": (True, "pre-norm scaling enabled"),
    "hidden_dropout": (0.0, "zero dropout"),
    "attention_dropout": (0.0, "zero dropout"),
    "partial_rotary_factor": (1.0, "full rotary embedding"),
}

DEAD = {
    "bos_token_id": "tokenizer artifact; MaxText reads the tokenizer",
    "eos_token_id": "tokenizer artifact",
    "pad_token_id": "tokenizer artifact",
    "initializer_range": "checkpoint conversion loads weights; not used for init",
    "max_position_embeddings": "sequence length limit, controlled by max_target_length",
    "num_logits_to_keep": "generation option; MaxText has its own decode loop",
    "residual_in_fp32": "Mamba-specific training option; upstream PyTorch kept false",
    "rope_theta": "NoPE attention has no rotary embeddings",
    "sliding_window": "null, no sliding window",
    "time_step_floor": "Mamba-2 dt floor, subsumed by time_step_limit",
    "time_step_max": "Mamba-2 dt clamp, subsumed by time_step_limit",
    "time_step_min": "Mamba-2 dt clamp, subsumed by time_step_limit",
    "use_bias": "global bias flag; layer-specific flags are used",
    "use_cache": "inference flag",
    "use_mamba_kernels": "PyTorch kernel dispatch flag",
    "expand": "mamba expansion ratio, superseded by mamba_num_heads * mamba_head_dim",
    "n_shared_experts": "always 1",
    "mamba_proj_bias": "unused by modeling",
    "norm_eps": "duplicate of layer_norm_epsilon",
}


def _production_config(**overrides):
  config_args = [None, get_test_config_path(), "run_name=test", "skip_jax_distributed_system=True"]
  return pyconfig.initialize(
      config_args, model_name=MODEL_NAME, enable_checkpointing=False, override_model_config=True, **overrides
  )


class NemotronConfigAuditTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.hf = dict(nemotron3_nano_omni_30b_a3b_dict)
    config_args = [
        None,
        get_test_config_path(f"models/{MODEL_NAME}.yml"),
        "run_name=test",
        "skip_jax_distributed_system=True",
    ]
    cls.cfg = pyconfig.initialize(config_args, enable_checkpointing=False)

  def test_audit_rows(self):
    failures = []
    for key, fn, source in AUDIT:
      expected, actual = fn(self.hf), getattr(self.cfg, key)
      if hasattr(actual, "value"):
        actual = actual.value
      if isinstance(actual, tuple):
        actual = list(actual)
      if actual != expected:
        failures.append((key, expected, actual, source))
    self.assertEqual(failures, [])

  def test_hardcoded_hf_values(self):
    for key, (value, source) in HARDCODED_HF.items():
      self.assertEqual(self.hf[key], value, f"{key}: {source}")

  def test_every_hf_key_accounted_for(self):
    consumed = {
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "hybrid_override_pattern",
        "layer_norm_epsilon",
        "tie_word_embeddings",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "attention_bias",
        "mamba_num_heads",
        "mamba_head_dim",
        "ssm_state_size",
        "n_groups",
        "conv_kernel",
        "use_conv_bias",
        "chunk_size",
        "time_step_limit",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "intermediate_size",
        "moe_shared_expert_intermediate_size",
        "mlp_hidden_act",
        "mlp_bias",
        "routed_scaling_factor",
        "norm_topk_prob",
        "n_group",
        "topk_group",
    } | set(HARDCODED_HF)
    unaccounted = sorted(set(self.hf) - consumed - set(DEAD))
    self.assertEqual(unaccounted, [])

  def test_derived_values(self):
    c = self.hf  # pylint: disable=redefined-outer-name
    width = c["mamba_num_heads"] * c["mamba_head_dim"]
    self.assertEqual(width, 4096)
    self.assertNotEqual(width, c["expand"] * c["hidden_size"])
    self.assertEqual(width + 2 * c["n_groups"] * c["ssm_state_size"], 6144)
    self.assertEqual(2 * width + 2 * c["n_groups"] * c["ssm_state_size"] + c["mamba_num_heads"], 14400 - 4096)
    self.assertEqual(width // c["n_groups"], 512)
    self.assertEqual(c["moe_shared_expert_intermediate_size"], 2 * c["moe_intermediate_size"])

  def test_pattern_counts(self):
    p = self.hf["hybrid_override_pattern"]
    self.assertEqual((p.count("M"), p.count("E"), p.count("*"), p.count("-")), (23, 23, 6, 0))
    self.assertEqual([i for i, ch in enumerate(p) if ch == "*"], [5, 12, 19, 26, 33, 42])

  def test_scan_layers_true_rejected(self):
    """Decision 3 guard: the Nemotron3 decoder is unscanned only."""
    with self.assertRaises(Exception):
      config_args = [
          None,
          get_test_config_path(f"models/{MODEL_NAME}.yml"),
          "run_name=test",
          "skip_jax_distributed_system=True",
      ]
      pyconfig.initialize(config_args, scan_layers=True, enable_checkpointing=False)

  def test_model_registered(self):
    # pylint: disable=import-outside-toplevel
    from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
    from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
    from maxtext.utils.globals import HF_IDS

    self.assertIn(MODEL_NAME, HF_IDS)
    self.assertIn(MODEL_NAME, PARAM_MAPPING)
    self.assertIn(MODEL_NAME, HOOK_FNS)
    hf_cfg = HF_MODEL_CONFIGS[MODEL_NAME]
    for key in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "hybrid_override_pattern",
        "n_routed_experts",
        "moe_intermediate_size",
        "moe_shared_expert_intermediate_size",
        "mamba_num_heads",
    ):
      self.assertEqual(getattr(hf_cfg, key), self.hf[key], key)


TINY_HF = {
    "vocab_size": 64,
    "hidden_size": 32,
    "hybrid_override_pattern": "ME*",
    "mamba_num_heads": 4,
    "mamba_head_dim": 8,
    "ssm_state_size": 8,
    "n_groups": 2,
    "conv_kernel": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "n_routed_experts": 3,
    "moe_intermediate_size": 16,
    "moe_shared_expert_intermediate_size": 32,
}
TINY_SHAPES = {
    "language_model.backbone.embeddings.weight": [64, 32],
    "language_model.backbone.norm_f.weight": [32],
    "language_model.lm_head.weight": [64, 32],
    "language_model.backbone.layers.0.norm.weight": [32],
    "language_model.backbone.layers.0.mixer.norm.weight": [32],
    "language_model.backbone.layers.0.mixer.in_proj.weight": [2 * 32 + 2 * 2 * 8 + 4, 32],
    "language_model.backbone.layers.0.mixer.conv1d.weight": [32 + 2 * 2 * 8, 1, 4],
    "language_model.backbone.layers.0.mixer.conv1d.bias": [64],
    "language_model.backbone.layers.0.mixer.dt_bias": [4],
    "language_model.backbone.layers.0.mixer.A_log": [4],
    "language_model.backbone.layers.0.mixer.D": [4],
    "language_model.backbone.layers.0.mixer.out_proj.weight": [32, 32],
    "language_model.backbone.layers.1.norm.weight": [32],
    "language_model.backbone.layers.1.mixer.gate.weight": [3, 32],
    "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias": [3],
    "language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight": [32, 32],
    "language_model.backbone.layers.1.mixer.shared_experts.down_proj.weight": [32, 32],
    "language_model.backbone.layers.2.norm.weight": [32],
    "language_model.backbone.layers.2.mixer.q_proj.weight": [32, 32],
    "language_model.backbone.layers.2.mixer.k_proj.weight": [16, 32],
    "language_model.backbone.layers.2.mixer.v_proj.weight": [16, 32],
    "language_model.backbone.layers.2.mixer.o_proj.weight": [32, 32],
}
for _e in range(3):
  TINY_SHAPES[f"language_model.backbone.layers.1.mixer.experts.{_e}.up_proj.weight"] = [16, 32]
  TINY_SHAPES[f"language_model.backbone.layers.1.mixer.experts.{_e}.down_proj.weight"] = [32, 16]


def _tiny_maxtext_config(**overrides):
  """The attributes the Nemotron3 mapping and hooks read from a MaxText config."""
  values = {
      "hybrid_override_pattern": "ME*",
      "shared_experts": 2,
      "num_experts": 3,
      "mamba_conv_bias": True,
      "weight_dtype": "bfloat16",
  }
  values.update(overrides)
  return types.SimpleNamespace(**values)


class NemotronCheckpointMappingTest(unittest.TestCase):
  """Shape table, parameter mapping, hooks and full-precision router state of the checkpoint converter."""

  def test_hf_weights_to_shape(self):
    self.assertEqual(NEMOTRON3_HF_WEIGHTS_TO_SHAPE(TINY_HF), TINY_SHAPES)
    # transformers>=5.17 configs carry layers_block_type instead of the pattern string.
    by_type = {k: v for k, v in TINY_HF.items() if k != "hybrid_override_pattern"}
    by_type["layers_block_type"] = ["mamba", "moe", "attention"]
    self.assertEqual(NEMOTRON3_HF_WEIGHTS_TO_SHAPE(by_type), TINY_SHAPES)

  def test_hf_weights_to_shape_production_count(self):
    # 3 + 52 block norms + 23 Mamba x 8 + 6 attention x 4 + 23 MoE x (2 router + 2 x 128 experts + 2 shared).
    shapes = NEMOTRON3_HF_WEIGHTS_TO_SHAPE(nemotron3_nano_omni_30b_a3b_dict)
    self.assertEqual(len(shapes), 6243)
    self.assertEqual(shapes["language_model.backbone.layers.0.mixer.in_proj.weight"], [10304, 2688])
    self.assertEqual(shapes["language_model.backbone.layers.1.mixer.experts.127.up_proj.weight"], [1856, 2688])

  def test_param_mapping_covers_every_hf_weight(self):
    mapping = NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING(TINY_HF, _tiny_maxtext_config())
    hf_keys = []
    for value in mapping.values():
      hf_keys.extend(value if isinstance(value, list) else [value])
    self.assertEqual(sorted(hf_keys), sorted(TINY_SHAPES))
    self.assertEqual(
        mapping["MoEBiasVar-decoder-layers_1-mixer-MoeBlock_0-gate-bias"],
        "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias",
    )
    self.assertEqual(
        mapping["params-decoder-layers_1-mixer-MoeBlock_0-wi_0"],
        [f"language_model.backbone.layers.1.mixer.experts.{e}.up_proj.weight" for e in range(3)],
    )
    no_bias = NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING(TINY_HF, _tiny_maxtext_config(mamba_conv_bias=False))
    self.assertNotIn("params-decoder-layers_0-mixer-conv_bias", no_bias)

  def test_param_mapping_production(self):
    cfg = _production_config()
    mapping = NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING(nemotron3_nano_omni_30b_a3b_dict, cfg)
    hf_keys = []
    for value in mapping.values():
      hf_keys.extend(value if isinstance(value, list) else [value])
    self.assertEqual(sorted(hf_keys), sorted(NEMOTRON3_HF_WEIGHTS_TO_SHAPE(nemotron3_nano_omni_30b_a3b_dict)))

  def test_param_mapping_rejects_bad_configs(self):
    with self.assertRaisesRegex(ValueError, "scan_layers"):
      NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING(TINY_HF, _tiny_maxtext_config(), scan_layers=True)
    with self.assertRaisesRegex(ValueError, "moe_shared_expert_intermediate_size"):
      NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING(TINY_HF, _tiny_maxtext_config(shared_experts=1))
    with self.assertRaisesRegex(ValueError, "Unsupported Nemotron-H block type"):
      NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING(TINY_HF, _tiny_maxtext_config(hybrid_override_pattern="M-"))
    with self.assertRaisesRegex(ValueError, "scan_layers"):
      NEMOTRON3_MAXTEXT_TO_HF_PARAM_HOOK_FN(TINY_HF, _tiny_maxtext_config(), scan_layers=True)

  def test_hooks_load_and_round_trip(self):
    cfg = _tiny_maxtext_config()
    load = NEMOTRON3_MAXTEXT_TO_HF_PARAM_HOOK_FN(TINY_HF, cfg, saving_to_hf=False)
    save = NEMOTRON3_MAXTEXT_TO_HF_PARAM_HOOK_FN(TINY_HF, cfg, saving_to_hf=True)
    mapping = NEMOTRON3_MAXTEXT_TO_HF_PARAM_MAPPING(TINY_HF, cfg)
    self.assertEqual(
        set(load),
        {
            "params-decoder-logits_dense-kernel",
            "params-decoder-layers_0-mixer-in_proj-kernel",
            "params-decoder-layers_0-mixer-out_proj-kernel",
            "params-decoder-layers_0-mixer-conv_weight",
            "params-decoder-layers_1-mixer-MoeBlock_0-gate-kernel",
            "params-decoder-layers_1-mixer-MoeBlock_0-wi_0",
            "params-decoder-layers_1-mixer-MoeBlock_0-wo",
            "params-decoder-layers_1-mixer-shared_experts-wi-kernel",
            "params-decoder-layers_1-mixer-shared_experts-wo-kernel",
            "params-decoder-layers_2-mixer-query-kernel",
            "params-decoder-layers_2-mixer-key-kernel",
            "params-decoder-layers_2-mixer-value-kernel",
            "params-decoder-layers_2-mixer-out-kernel",
        },
    )
    maxtext_shapes = {
        "params-decoder-logits_dense-kernel": (32, 64),
        "params-decoder-layers_0-mixer-in_proj-kernel": (32, 100),
        "params-decoder-layers_0-mixer-out_proj-kernel": (32, 32),
        "params-decoder-layers_0-mixer-conv_weight": (4, 64),
        "params-decoder-layers_1-mixer-MoeBlock_0-gate-kernel": (32, 3),
        "params-decoder-layers_1-mixer-shared_experts-wi-kernel": (32, 32),
        "params-decoder-layers_1-mixer-shared_experts-wo-kernel": (32, 32),
        "params-decoder-layers_2-mixer-query-kernel": (32, 4, 8),
        "params-decoder-layers_2-mixer-key-kernel": (32, 2, 8),
        "params-decoder-layers_2-mixer-value-kernel": (32, 2, 8),
        "params-decoder-layers_2-mixer-out-kernel": (4, 8, 32),
    }
    rng = np.random.default_rng(0)
    for key, mt_shape in maxtext_shapes.items():
      with self.subTest(key):
        hf_shape = tuple(TINY_SHAPES[mapping[key]])
        hf_value = rng.normal(size=hf_shape).astype(np.float32)
        mt_value = load[key](hf_value, mt_shape)
        self.assertEqual(mt_value.shape, mt_shape)
        np.testing.assert_array_equal(save[key](mt_value, hf_shape), hf_value)
    # Depthwise conv: HF [C, 1, K] -> MaxText [K, C], tap k of channel c preserved.
    conv = rng.normal(size=(64, 1, 4)).astype(np.float32)
    mt_conv = load["params-decoder-layers_0-mixer-conv_weight"](conv)
    np.testing.assert_array_equal(mt_conv[2, 5], conv[5, 0, 2])
    # Attention q: HF row h * head_dim + d -> MaxText [:, h, d].
    q = rng.normal(size=(32, 32)).astype(np.float32)
    mt_q = load["params-decoder-layers_2-mixer-query-kernel"](q, (32, 4, 8))
    np.testing.assert_array_equal(mt_q[:, 3, 1], q[3 * 8 + 1])
    # Stacked experts: the hook transposes each [out, in] expert matrix.
    up = rng.normal(size=(16, 32)).astype(np.float32)
    np.testing.assert_array_equal(load["params-decoder-layers_1-mixer-MoeBlock_0-wi_0"](up), up.T)

  def test_router_bias_saved_to_hf_in_float32(self):
    key = "MoEBiasVar-decoder-layers_1-mixer-MoeBlock_0-gate-bias"
    hf_key = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias"
    bias = np.array([55.6875, 55.6884, 55.6891], np.float32)  # Closer than one bf16 ulp at this magnitude.
    out = process_maxtext_param(key, bias, {key: hf_key}, {}, {hf_key: [3]}, _tiny_maxtext_config())
    self.assertEqual(out[0][0], hf_key)
    self.assertEqual(out[0][1].dtype, np.float32)
    np.testing.assert_array_equal(out[0][1], bias)


class NemotronHfArchitectureUpdateTest(unittest.TestCase):
  """to_huggingface._validate_or_update_architecture on a Nemotron-H config."""

  def test_shared_expert_width_and_pattern(self):
    # pylint: disable=import-outside-toplevel
    from maxtext.checkpoint_conversion import to_huggingface
    from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS

    cfg = _production_config()
    hf_cfg = copy.deepcopy(HF_MODEL_CONFIGS[MODEL_NAME])
    # n_shared_experts=1 (width 3712) vs shared_experts=2 (2 x 1856) is not a mismatch.
    to_huggingface._validate_or_update_architecture(hf_cfg, cfg, override=False)  # pylint: disable=protected-access

    sliced = _production_config(base_num_decoder_layers=3, hybrid_override_pattern="ME*")
    to_huggingface._validate_or_update_architecture(hf_cfg, sliced, override=True)  # pylint: disable=protected-access
    self.assertEqual(list(hf_cfg.layers_block_type), ["linear_attention", "moe", "full_attention"])
    self.assertEqual(hf_cfg.hybrid_override_pattern, "ME*")

    # Config classes that store the pattern as a plain attribute get it written directly.
    plain = types.SimpleNamespace(hybrid_override_pattern=cfg.hybrid_override_pattern)
    to_huggingface._validate_or_update_architecture(plain, sliced, override=True)  # pylint: disable=protected-access
    self.assertEqual(plain.hybrid_override_pattern, "ME*")


class LazyTensorLeafHandlerTest(unittest.TestCase):
  """The Orbax v1 LazyTensor handler saves in bounded batches; MoEBiasVar restores in float32."""

  def test_save_in_batches_and_restore_router_bias_in_float32(self):
    # pylint: disable=import-outside-toplevel
    from maxtext.checkpoint_conversion import to_maxtext
    from maxtext.checkpoint_conversion.utils import utils as conversion_utils

    rng = np.random.default_rng(0)
    # 130 leaves > the 128-leaf batch cap, so the handler writes two batches.
    weights = {f"w{i}": rng.normal(size=(2, 3)).astype(np.float32) for i in range(129)}
    bias = np.array([55.6875, 55.6884, 55.6891], np.float32)
    tree = {
        "params": {k: to_maxtext.LazyTensor(lambda v=v: v, v.shape, "bfloat16", name=k) for k, v in weights.items()},
        "MoEBiasVar": {"bias": to_maxtext.LazyTensor(lambda: bias, bias.shape, "float32", name="bias")},
    }
    logs = []
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(to_maxtext.max_logging, "log", logs.append):
      conversion_utils.save_weights_to_checkpoint(tmp, tree, device_count=1, use_ocdbt=True, use_zarr3=True)
      config = types.SimpleNamespace(
          checkpoint_storage_concurrent_gb=1,
          checkpoint_storage_use_ocdbt=True,
          checkpoint_storage_use_zarr3=True,
          lora=types.SimpleNamespace(lora_restore_path=""),
          load_parameters_path=os.path.join(tmp, "0", "items"),
          weight_dtype="bfloat16",
      )
      restored = conversion_utils.load_orbax_checkpoint(config)["params"]
    self.assertTrue(any("[LazyTensorLeafHandler] Saved batch 2/2" in str(m) for m in logs), logs[-5:])
    self.assertEqual(restored["MoEBiasVar"]["bias"].dtype, np.float32)
    np.testing.assert_array_equal(np.asarray(restored["MoEBiasVar"]["bias"]), bias)
    self.assertEqual(restored["params"]["w0"].dtype, jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(restored["params"]["w7"], np.float32), weights["w7"].astype(jnp.bfloat16).astype(np.float32)
    )


if __name__ == "__main__":
  unittest.main()
