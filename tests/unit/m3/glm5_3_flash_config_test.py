# Copyright 2023-2026 Google LLC
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

"""Unit tests for GLM-5.3-Flash configuration, sharding, and validation guards."""

import importlib
import json
import pathlib
import sys
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
import jax
from jax.sharding import PartitionSpec as P
from maxtext.configs import pyconfig
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Config
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Sharding
from maxtext.m3.models.glm5_3_flash.glm5_config import validate_no_packing
import numpy as np


class Glm5ConfigTest(absltest.TestCase):
  """Tests for Glm5Config invariants, loud guards, and pyconfig integration."""

  def test_default_config_invariants(self):
    c = Glm5Config()
    self.assertEqual((c.num_hidden_layers, c.first_k_dense_replace), (45, 3))
    self.assertEqual((c.vocab_size, c.intermediate_size), (154880, 12288))
    self.assertEqual((c.hidden_size, c.moe_intermediate_size), (4096, 2048))
    self.assertEqual((c.routed_scaling_factor, c.n_routed_experts), (2.5, 288))
    self.assertEqual((c.num_experts_per_tok, c.n_shared_experts), (8, 1))
    self.assertEqual((c.linear_num_key_heads, c.linear_key_head_dim), (64, 128))
    self.assertEqual((c.linear_num_heads, c.linear_head_dim), (64, 128))
    self.assertEqual((c.linear_conv_kernel_dim, c.linear_lower_bound), (4, -5.0))
    self.assertEqual((c.num_attention_heads, c.num_key_value_heads), (64, 64))
    self.assertEqual((c.q_lora_rank, c.kv_lora_rank), (1536, 512))
    self.assertEqual((c.v_head_dim, c.qk_head_dim), (256, 256))
    self.assertEqual((c.qk_nope_head_dim, c.qk_rope_head_dim), (256, 0))
    self.assertEqual((c.mhc_sinkhorn_iters, c.mhc_expansion_rate), (20, 4))
    self.assertEqual((c.index_topk, c.index_kpool, c.mhc_eps), (2048, 4, 1e-6))
    self.assertTrue(c.index_kpool_always_select_tail)
    self.assertEqual((c.indexer_loss_scaling_factor, c.packing), (0.0, False))
    self.assertEqual((c.rms_norm_eps, c.rope_parameters), (1e-5, None))
    self.assertEqual((c.scan_layers, c.kda_implementation), (True, "mosaic"))
    self.assertEqual((c.moe_lb_loss_weight, c.mlp_clamp_limit), (0.001, 10.0))
    self.assertEqual(c.remat_policy, "full")
    self.assertEqual(c.dsa_attention_implementation, "dense_mask")
    self.assertEqual(c.router_param_dtype, "float32")
    self.assertEqual(c.weight_dtype, "float32")
    assert c.layer_types is not None and c.indexer_types is not None
    self.assertEqual((len(c.layer_types), set(c.indexer_types)), (45, {"full"}))
    self.assertEqual(c.layer_types.count("linear_attention"), 34)
    self.assertEqual(c.layer_types.count("full_attention"), 11)

  def test_import_glm5_config_does_not_import_tokamax(self):
    saved_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k.startswith(("maxtext.src.maxtext.m3.models.glm5_3_flash", "tokamax"))
    }
    try:
      importlib.import_module("maxtext.src.maxtext.m3.models.glm5_3_flash.glm5_config")
      importlib.import_module("maxtext.src.maxtext.m3.models.glm5_3_flash")
      self.assertFalse(
          any(m == "tokamax" or m.startswith("tokamax.") for m in sys.modules),
          [m for m in sys.modules if "tokamax" in m],
      )
    finally:
      sys.modules.update(saved_modules)

  def test_matches_published_hf_text_config(self):
    # Provenance: checked-in copy of published zai-org/GLM-5.3-Flash/config.json
    config_path = pathlib.Path(__file__).parent / "testdata" / "glm5_3_flash_config.json"
    with config_path.open("r", encoding="utf-8") as f:
      hf_root = json.load(f)
    hf_text_config = hf_root["text_config"]
    c = Glm5Config()
    self.assertEqual(c.vocab_size, hf_text_config["vocab_size"])
    self.assertEqual(c.hidden_size, hf_text_config["hidden_size"])
    self.assertEqual(c.num_hidden_layers, hf_text_config["num_hidden_layers"])
    expected_mlp = ["dense"] * c.first_k_dense_replace + ["sparse"] * (c.num_hidden_layers - c.first_k_dense_replace)
    self.assertEqual(expected_mlp, hf_text_config["mlp_layer_types"])
    self.assertEqual(c.mhc_eps, hf_text_config["hc_eps"])
    self.assertEqual(c.mhc_expansion_rate, hf_text_config["hc_mult"])
    self.assertEqual(c.mhc_sinkhorn_iters, hf_text_config["hc_sinkhorn_iters"])
    self.assertEqual(c.index_head_dim, hf_text_config["index_head_dim"])
    self.assertEqual(c.index_kpool, hf_text_config["index_kpool"])
    self.assertEqual(
        c.index_kpool_always_select_tail,
        hf_text_config["index_kpool_always_select_tail"],
    )
    self.assertEqual(c.index_n_heads, hf_text_config["index_n_heads"])
    self.assertEqual(c.index_topk, hf_text_config["index_topk"])
    self.assertEqual(c.intermediate_size, hf_text_config["intermediate_size"])
    self.assertEqual(c.kv_lora_rank, hf_text_config["kv_lora_rank"])
    self.assertEqual(c.moe_intermediate_size, hf_text_config["moe_intermediate_size"])
    self.assertEqual(c.n_group, hf_text_config["n_group"])
    self.assertEqual(c.n_routed_experts, hf_text_config["n_routed_experts"])
    self.assertEqual(c.n_shared_experts, hf_text_config["n_shared_experts"])
    self.assertEqual(c.norm_topk_prob, hf_text_config["norm_topk_prob"])
    self.assertEqual(c.num_attention_heads, hf_text_config["num_attention_heads"])
    self.assertEqual(c.num_key_value_heads, hf_text_config["num_key_value_heads"])
    self.assertEqual(c.num_experts_per_tok, hf_text_config["num_experts_per_tok"])
    self.assertEqual(c.num_nextn_predict_layers, hf_text_config["num_nextn_predict_layers"])
    self.assertEqual(c.q_lora_rank, hf_text_config["q_lora_rank"])
    self.assertEqual(c.qk_head_dim, hf_text_config["qk_head_dim"])
    self.assertEqual(c.qk_nope_head_dim, hf_text_config["qk_nope_head_dim"])
    self.assertEqual(c.qk_rope_head_dim, hf_text_config["qk_rope_head_dim"])
    self.assertEqual(c.rms_norm_eps, hf_text_config["rms_norm_eps"])
    self.assertEqual(c.routed_scaling_factor, hf_text_config["routed_scaling_factor"])
    self.assertEqual(c.moe_lb_loss_weight, hf_text_config["router_aux_loss_coef"])
    self.assertEqual(c.scoring_func, hf_text_config["scoring_func"])
    self.assertEqual(c.mlp_clamp_limit, hf_text_config["swiglu_limit"])
    self.assertEqual(c.topk_group, hf_text_config["topk_group"])
    self.assertEqual(c.v_head_dim, hf_text_config["v_head_dim"])
    self.assertEqual(c.indexer_types, tuple(hf_text_config["indexer_types"]))
    assert c.layer_types is not None
    expected_hf_layer_types = [
        "linear_attention" if t == "linear_attention" else "deepseek_sparse_attention" for t in c.layer_types
    ]
    self.assertEqual(expected_hf_layer_types, hf_text_config["layer_types"])

    linear_cfg = hf_text_config["linear_attn_config"]
    self.assertEqual(c.linear_num_heads, linear_cfg["num_heads"])
    self.assertEqual(c.linear_num_key_heads, linear_cfg["num_heads"])
    self.assertEqual(c.linear_num_value_heads, linear_cfg["num_heads"])
    self.assertEqual(c.linear_head_dim, linear_cfg["head_dim"])
    self.assertEqual(c.linear_key_head_dim, linear_cfg["head_dim"])
    self.assertEqual(c.linear_value_head_dim, linear_cfg["head_dim"])
    self.assertEqual(c.linear_conv_kernel_dim, linear_cfg["short_conv_kernel_size"])
    self.assertEqual(c.linear_lower_bound, linear_cfg["gate_lower_bound"])
    kda_layers = [i for i, t in enumerate(c.layer_types) if t == "linear_attention"]
    full_layers = [i for i, t in enumerate(c.layer_types) if t == "full_attention"]
    self.assertEqual(kda_layers, linear_cfg["kda_layers"])
    self.assertEqual(full_layers, linear_cfg["full_attn_layers"])

  def test_mini_config_invariants(self):
    exp = ("linear_attention",) * 3 + ("full_attention", "linear_attention")
    for cfg in (Glm5Config.mini_config(), Glm5Config(num_hidden_layers=5)):
      self.assertEqual((cfg.num_hidden_layers, cfg.layer_types), (5, exp))
      self.assertEqual(cfg.indexer_types, ("full",) * 5)

  def test_from_maxtext_config_identity_bypass_with_overrides(self):
    base_cfg = Glm5Config()
    new_lt = ("linear_attention",) * 5
    new_it = ("full",) * 5
    overridden = Glm5Config.from_maxtext_config(
        base_cfg,
        num_hidden_layers=5,
        layer_types=new_lt,
        indexer_types=new_it,
    )
    self.assertEqual(overridden.num_hidden_layers, 5)
    self.assertEqual(overridden.layer_types, new_lt)
    self.assertEqual(overridden.indexer_types, new_it)
    overridden_auto = Glm5Config.from_maxtext_config(base_cfg, num_hidden_layers=5)
    self.assertEqual(len(overridden_auto.layer_types), 5)
    self.assertEqual(len(overridden_auto.indexer_types), 5)
    same = Glm5Config.from_maxtext_config(base_cfg)
    self.assertIs(same, base_cfg)

  def test_from_maxtext_config_qk_head_dim_derived_on_override(self):
    cfg = Glm5Config(qk_nope_head_dim=128)
    self.assertEqual(cfg.qk_head_dim, 128)
    overridden_qk = Glm5Config.from_maxtext_config(Glm5Config(), qk_nope_head_dim=128)
    self.assertEqual(overridden_qk.qk_head_dim, 128)

  def test_dsa_attention_implementation(self):
    c_default = Glm5Config()
    self.assertEqual(c_default.dsa_attention_implementation, "dense_mask")
    c_dense = Glm5Config(dsa_attention_implementation="dense_mask")
    self.assertEqual(c_dense.dsa_attention_implementation, "dense_mask")
    c_mini = Glm5Config.mini_config(dsa_attention_implementation="dense_mask")
    self.assertEqual(c_mini.dsa_attention_implementation, "dense_mask")

    dummy_cfg = pyconfig.initialize(["", "src/maxtext/configs/base.yml", "model_name=glm5.3-flash"])
    c_from = Glm5Config.from_maxtext_config(dummy_cfg, dsa_attention_implementation="dense_mask")
    self.assertEqual(c_from.dsa_attention_implementation, "dense_mask")

  def test_loud_guards(self):
    for kwargs, pat in (
        ({"packing": True}, "packing=True"),
        ({"indexer_types": ("shared",) * 45}, "Shared indexer"),
        ({"indexer_types": ("full",) * 44}, r"len\(indexer_types\)"),
        ({"layer_types": ("linear_attention",) * 44}, r"len\(layer_types\)"),
        ({"layer_types": ("bad",) * 45}, "invalid"),
        ({"routed_bias_update_rate": 0.01}, "routed_bias_update_rate"),
        ({"indexer_loss_scaling_factor": 0.1}, "indexer_loss_scaling_factor"),
        ({"kda_implementation": cast(Any, ["mosaic"])}, "kda_implementation"),
        ({"kda_implementation": "bad"}, "kda_implementation"),
        (
            {"kda_implementation": "mosaic", "linear_lower_bound": None},
            "linear_lower_bound",
        ),
        (
            {"kda_implementation": "xla_chunked", "linear_lower_bound": None},
            "linear_lower_bound",
        ),
        ({"index_kpool": 0}, "index_kpool must be >= 1"),
        (
            {"index_topk": 2048, "index_kpool": 3},
            r"index_topk.*divisible by index_kpool",
        ),
        ({"mlp_clamp_limit": 0.0}, "mlp_clamp_limit must be > 0.0"),
        ({"mlp_clamp_limit": -1.0}, "mlp_clamp_limit must be > 0.0"),
        (
            {"dsa_attention_implementation": cast(Any, ["dense_mask"])},
            "dsa_attention_implementation must be a str",
        ),
        (
            {"dsa_attention_implementation": cast(Any, None)},
            "dsa_attention_implementation must be a str",
        ),
        (
            {"dsa_attention_implementation": cast(Any, 123)},
            "dsa_attention_implementation must be a str",
        ),
        (
            {"dsa_attention_implementation": "tokamax_splash"},
            "dsa_attention_implementation must be one of",
        ),
        (
            {"dsa_attention_implementation": "gather_scan"},
            "dsa_attention_implementation must be one of",
        ),
        (
            {"dsa_attention_implementation": "flash"},
            "dsa_attention_implementation must be one of",
        ),
        (
            {"dsa_attention_implementation": "bad"},
            "dsa_attention_implementation must be one of",
        ),
        ({"qk_rope_head_dim": 64}, "NoPE"),
        ({"rope_parameters": {"type": "yarn"}}, "NoPE"),
        ({"num_key_value_heads": 32}, r"num_key_value_heads.*must equal"),
        ({"linear_num_value_heads": 32}, r"linear_num_key_heads.*must equal"),
        ({"linear_value_head_dim": 64}, r"linear_key_head_dim.*must equal"),
        ({"remat_policy": "bogus"}, "remat_policy must be 'none' or 'full'"),
        (
            {"router_param_dtype": "bfloat16"},
            "router_param_dtype must be 'float32'",
        ),
        (
            {"router_param_dtype": cast(Any, 123)},
            "router_param_dtype must be 'float32'",
        ),
        (
            {"router_param_dtype": cast(Any, None)},
            "router_param_dtype must be 'float32'",
        ),
        ({"weight_dtype": "int8"}, "weight_dtype must be one of"),
        ({"weight_dtype": cast(Any, 123)}, "weight_dtype must be one of"),
    ):
      with self.assertRaisesRegex(ValueError, pat):
        Glm5Config(**cast(dict[str, Any], kwargs))
    with self.assertRaisesRegex(TypeError, "remat_policy must be a str"):
      Glm5Config(remat_policy=cast(Any, 123))
    # Intentional negative test: qk_head_dim is a derived property, not an init field.
    with self.assertRaises(TypeError):
      Glm5Config(**cast(Any, {"qk_head_dim": 128}))  # pylint: disable=unexpected-keyword-arg

  def test_validate_no_packing_helper(self):
    validate_no_packing(None)
    validate_no_packing(np.array([[0, 1, 1], [0, 0, 1]]))
    with self.assertRaisesRegex(ValueError, "Packed sequences unsupported"):
      validate_no_packing(np.array([[0, 1, 2]]))

  def test_sharding_resolution(self):
    s = Glm5Sharding()
    s_tp = Glm5Sharding(use_tp=True)
    s_ep = Glm5Sharding(ep_in_batch=True)
    act = m3_sharding.TensorType.ACTIVATION
    self.assertEqual(s("expert_w", ("expert",)), P("expert"))
    self.assertEqual(s("embed_w", ("embed",)), P("fsdp"))
    self.assertEqual(s("q_lora_w", ("q_lora",)), P("fsdp"))
    self.assertEqual(s("kv_lora_w", ("kv_lora",)), P("fsdp"))
    self.assertEqual(s("embed_table", ("vocab", "embed")), P(None, "fsdp"))
    self.assertEqual(s("heads_w", ("heads",)), P(None))
    self.assertEqual(s("b", ("batch",), act), P(("data", "fsdp")))
    self.assertEqual(s("h", ("batch", "embed"), act), P(("data", "fsdp"), None))
    self.assertEqual(s("q_act", ("batch", "q_lora"), act), P(("data", "fsdp"), None))
    self.assertEqual(s("kv_act", ("batch", "kv_lora"), act), P(("data", "fsdp"), None))
    self.assertEqual(s_tp("heads_w", ("heads",)), P("tensor"))
    self.assertEqual(s_tp("embed_table", ("vocab", "embed")), P("tensor", "fsdp"))
    self.assertEqual(s_tp("q_lora_w", ("q_lora",)), P("fsdp"))
    self.assertEqual(s_tp("kv_lora_w", ("kv_lora",)), P("fsdp"))
    self.assertEqual(s_ep("b", ("batch",), act), P(("data", "fsdp", "expert")))
    self.assertEqual(s_ep("h", ("batch", "embed"), act), P(("data", "fsdp", "expert"), None))
    with self.assertRaises(KeyError):
      s("bad", ("unknown_axis",))

  def test_from_mesh(self):
    # Mesh with tp=1 (use_tp=False)
    devices = np.array(jax.devices()[:1]).reshape((1, 1, 1, 1))
    mesh_m3_tp1 = jax.sharding.Mesh(devices, ("data", "fsdp", "tensor", "expert"))
    s1 = Glm5Sharding.from_mesh(mesh_m3_tp1)
    self.assertFalse(s1.use_tp)
    self.assertEqual(s1.ep_size, 1)
    self.assertEqual(
        (s1.dp_axis, s1.fsdp_axis, s1.tp_axis, s1.ep_axis),
        ("data", "fsdp", "tensor", "expert"),
    )

    # Mesh with tp=2 (use_tp=True)
    mesh_m3_tp2 = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mesh_m3_tp2.axis_names = ("data", "fsdp", "tensor", "expert")
    mesh_m3_tp2.shape = {"data": 1, "fsdp": 1, "tensor": 2, "expert": 1}
    s2 = Glm5Sharding.from_mesh(mesh_m3_tp2)
    self.assertTrue(s2.use_tp)

    # MaxText mesh ("data", "fsdp", "tensor", "expert")
    mesh_maxtext = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mesh_maxtext.axis_names = ("data", "fsdp", "tensor", "expert")
    mesh_maxtext.shape = {"data": 1, "fsdp": 1, "tensor": 2, "expert": 1}
    s_mt = Glm5Sharding.from_mesh(mesh_maxtext)
    self.assertTrue(s_mt.use_tp)
    self.assertEqual((s_mt.dp_axis, s_mt.fsdp_axis, s_mt.tp_axis, s_mt.ep_axis), ("data", "fsdp", "tensor", "expert"))
    self.assertEqual(
        s_mt(
            "batch_act",
            ("batch", "heads", "expert"),
            m3_sharding.TensorType.ACTIVATION,
        ),
        P(("data", "fsdp"), "tensor", None),
    )
    self.assertEqual(
        s_mt("w", ("embed", "heads", "expert")),
        P("fsdp", "tensor", "expert"),
    )

    # Context axis test (P2-5)
    mesh_ctx_1 = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mesh_ctx_1.axis_names = ("data", "fsdp", "tensor", "expert", "context")
    mesh_ctx_1.shape = {
        "data": 1,
        "fsdp": 1,
        "tensor": 1,
        "expert": 1,
        "context": 1,
    }
    s_ctx_1 = Glm5Sharding.from_mesh(mesh_ctx_1)
    self.assertEqual(s_ctx_1.ep_size, 1)

    mesh_ctx_2 = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mesh_ctx_2.axis_names = ("data", "fsdp", "tensor", "expert", "context")
    mesh_ctx_2.shape = {
        "data": 1,
        "fsdp": 1,
        "tensor": 1,
        "expert": 1,
        "context": 2,
    }
    with self.assertRaisesRegex(NotImplementedError, "not yet supported in Glm5Sharding"):
      Glm5Sharding.from_mesh(mesh_ctx_2)

    # Unsupported mesh axes raising ValueError
    mesh_bad = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mesh_bad.axis_names = ("x", "y")
    mesh_bad.shape = {"x": 1, "y": 1}
    with self.assertRaisesRegex(ValueError, "Unsupported mesh axis_names"):
      Glm5Sharding.from_mesh(mesh_bad)

  def test_pyconfig_integration(self):
    argv = ["", "src/maxtext/configs/base.yml", "model_name=glm5.3-flash"]
    raw_cfg = pyconfig.initialize(argv)
    self.assertTrue(raw_cfg.use_m3_model)
    self.assertEqual(raw_cfg.tokenizer_path, "zai-org/GLM-5.3-Flash")
    self.assertEqual(raw_cfg.tokenizer_type, "huggingface")

    c = Glm5Config.from_maxtext_config(raw_cfg)
    self.assertEqual((c.vocab_size, c.hidden_size), (154880, 4096))
    self.assertEqual((c.num_hidden_layers, c.num_attention_heads), (45, 64))
    self.assertEqual((c.intermediate_size, c.n_routed_experts), (12288, 288))
    self.assertEqual((c.moe_intermediate_size, c.qk_head_dim), (2048, 256))
    self.assertEqual((c.first_k_dense_replace, c.n_group), (3, 1))
    self.assertEqual((c.num_nextn_predict_layers, c.packing), (1, False))
    self.assertEqual((c.index_n_heads, c.index_head_dim), (32, 128))
    self.assertEqual((c.index_topk, c.mhc_expansion_rate), (2048, 4))
    self.assertEqual((c.mhc_sinkhorn_iters, c.topk_group), (20, 1))
    self.assertEqual((c.mtp_loss_scaling_factor, c.rms_norm_eps), (0.1, 1e-5))
    self.assertEqual((c.norm_topk_prob, c.scoring_func), (True, "sigmoid"))
    self.assertEqual(c.routed_bias_update_rate, 0.0)
    self.assertEqual(c.indexer_loss_scaling_factor, 0.0)
    self.assertEqual(c.moe_lb_loss_weight, 0.001)
    self.assertEqual(c.mlp_clamp_limit, 10.0)

    self.assertEqual(
        Glm5Config.from_maxtext_config(pyconfig.initialize(argv + ["remat_policy=none"])).remat_policy,
        "none",
    )

    cfg_custom = pyconfig.initialize(
        argv + ["override_model_config=true", "load_balance_loss_weight=0.05", "mlp_activations_limit=8.0"]
    )
    c_custom = Glm5Config.from_maxtext_config(cfg_custom)
    self.assertEqual(c_custom.moe_lb_loss_weight, 0.05)
    self.assertEqual(c_custom.mlp_clamp_limit, 8.0)
    self.assertEqual(c.dsa_attention_implementation, "dense_mask")
    c_dense = Glm5Config.from_maxtext_config(pyconfig.initialize(argv), dsa_attention_implementation="dense_mask")
    self.assertEqual(c_dense.dsa_attention_implementation, "dense_mask")
    with self.assertRaisesRegex(ValueError, "dsa_attention_implementation must be one of"):
      Glm5Config.from_maxtext_config(
          pyconfig.initialize(argv),
          dsa_attention_implementation="tokamax_splash",
      )

    d4 = ["routed_bias_update_rate=0.01", "decoder_block=deepseek"]
    d3 = ["indexer_loss_scaling_factor=0.1", "max_target_length=4096"]
    argv_override = argv + ["override_model_config=true"]
    for extra, pat in (
        (["packing=true"], "FU-D5"),
        (d4, "FU-D4"),
        (d3, "FU-D3"),
        (["mlp_activations_limit=-1.0"], "mlp_clamp_limit must be > 0.0"),
        (["use_m3_model=false"], "requires use_m3_model=True"),
    ):
      cfg_err = pyconfig.initialize(argv_override + extra)
      with self.assertRaisesRegex(ValueError, pat):
        Glm5Config.from_maxtext_config(cfg_err)


if __name__ == "__main__":
  absltest.main()
