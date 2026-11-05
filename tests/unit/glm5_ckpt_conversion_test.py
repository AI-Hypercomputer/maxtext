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

"""Unit tests for GLM-5.1 and GLM-5.2 HuggingFace <-> MaxText checkpoint conversion."""

import contextlib
import copy
import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

from absl.testing import parameterized
import jax
from maxtext.checkpoint_conversion import to_huggingface
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import utils as ckpt_utils
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.utils import detect_and_extract_checkpoint, load_orbax_checkpoint
from maxtext.common import checkpointing
from maxtext.configs import pyconfig
import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import load_file as load_safetensors_np
from safetensors.numpy import save_file as save_safetensors_np
from tests.utils.test_helpers import get_test_config_path
import transformers


@contextlib.contextmanager
def _orbax_checkpointer_compat():
  """Compatibility shim when running locally against orbax-checkpoint < 0.12.8 (no `logger` kwarg)."""
  orig_cls = checkpointing.ocp.training.Checkpointer
  if "logger" in inspect.signature(orig_cls.__init__).parameters:
    yield
    return

  def _compat_checkpointer(*args, **kwargs):
    if kwargs.get("logger") is None:
      kwargs.pop("logger", None)
    return orig_cls(*args, **kwargs)

  with mock.patch.object(checkpointing.ocp.training, "Checkpointer", side_effect=_compat_checkpointer):
    yield


def _make_mini_glm_hf_config(use_index_share: bool):
  """Creates a mini 6-layer GlmMoeDsaConfig matching GLM-5.1 or GLM-5.2 (`FFSSFS`)."""
  indexer_types = ["full", "full", "shared", "shared", "full", "shared"] if use_index_share else ["full"] * 6
  cfg_dict = {
      "architectures": ["GlmMoeDsaForCausalLM"],
      "attention_bias": False,
      "attention_dropout": 0.0,
      "dtype": "float32",
      "first_k_dense_replace": 2,
      "hidden_act": "silu",
      "hidden_size": 32,
      "index_head_dim": 8,
      "index_n_heads": 4,
      "index_topk": 4,
      "index_topk_freq": 4 if use_index_share else 1,
      "indexer_rope_interleave": True,
      "indexer_types": indexer_types,
      "initializer_range": 0.02,
      "intermediate_size": 64,
      "kv_lora_rank": 8,
      "max_position_embeddings": 128,
      "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
      "model_type": "glm_moe_dsa",
      "moe_intermediate_size": 16,
      "n_group": 1,
      "n_routed_experts": 4,
      "n_shared_experts": 1,
      "norm_topk_prob": True,
      "num_attention_heads": 4,
      "num_experts_per_tok": 2,
      "num_hidden_layers": 6,
      "num_key_value_heads": 4,
      "q_lora_rank": 16,
      "qk_nope_head_dim": 8,
      "qk_rope_head_dim": 4,
      "rms_norm_eps": 1e-5,
      "rope_interleave": True,
      "rope_parameters": {
          "rope_theta": 8000000.0 if use_index_share else 1000000.0,
          "rope_type": "default",
      },
      "routed_scaling_factor": 2.5,
      "scoring_func": "sigmoid",
      "tie_word_embeddings": False,
      "topk_group": 1,
      "use_cache": True,
      "v_head_dim": 8,
      "vocab_size": 64,
  }
  if hasattr(transformers, "GlmMoeDsaConfig"):
    return transformers.GlmMoeDsaConfig(**cfg_dict)
  return transformers.PreTrainedConfig(**cfg_dict)


def _mini_maxtext_cli_overrides(model_name: str, scan_layers: bool, use_index_share: bool) -> list[str]:
  """Returns CLI overrides matching `_make_mini_glm_hf_config`."""
  overrides = [
      f"model_name={model_name}",
      "override_model_config=True",
      "skip_jax_distributed_system=True",
      f"scan_layers={scan_layers}",
      "dtype=float32",
      "weight_dtype=float32",
      "base_emb_dim=32",
      "base_mlp_dim=64",
      "base_moe_mlp_dim=16",
      "base_num_decoder_layers=6",
      "first_num_dense_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "q_lora_rank=16",
      "kv_lora_rank=8",
      "qk_nope_head_dim=8",
      "qk_rope_head_dim=4",
      "v_head_dim=8",
      "indexer_n_heads=4",
      "indexer_head_dim=8",
      "indexer_topk=4",
      "num_experts=4",
      "num_experts_per_tok=2",
      "shared_experts=1",
      "vocab_size=64",
      "max_position_embeddings=128",
      "original_max_position_embeddings=128",
      "max_target_length=16",
      "per_device_batch_size=1",
  ]
  if use_index_share:
    overrides.append("index_share_pattern=FFSSFS")
  return overrides


def _build_synthetic_hf_tensors(model_name: str, hf_cfg) -> dict[str, np.ndarray]:
  """Builds deterministic, uniquely distinguishable float32 tensors for every expected HF key."""
  shape_map = HF_SHAPE[model_name](hf_cfg.to_dict())
  rng = np.random.default_rng(12345)
  tensors: dict[str, np.ndarray] = {}
  for idx, (key, shape) in enumerate(sorted(shape_map.items())):
    # Offset each key by (idx + 1) * 0.01 so accidental cross-layer / cross-expert aliasing is caught.
    arr = rng.standard_normal(size=shape).astype(np.float32) * 0.1 + np.float32((idx + 1) * 0.01)
    tensors[key] = np.ascontiguousarray(arr)
  return tensors


class Glm5CheckpointConversionTest(parameterized.TestCase):
  """End-to-end HF <-> MaxText round-trip and fast-fail tests for GLM-5.1 and GLM-5.2."""

  def setUp(self):
    super().setUp()
    self.enterContext(_orbax_checkpointer_compat())
    self.enterContext(mock.patch.object(ckpt_utils.gc, "collect", return_value=0))
    self.enterContext(mock.patch.object(to_huggingface.gc, "collect", return_value=0))

  def _resolve_orbax_items_dir(self, mt_out_dir: str) -> str:
    items_path = os.path.join(mt_out_dir, "0", "items")
    if os.path.isdir(items_path):
      return items_path
    return os.path.join(mt_out_dir, "0")

  def test_hf_shape_glm51_vs_glm52_744b(self):
    """Verifies 78 vs 21 indexer layers in HF_SHAPE and canonical GlmMoeDsaConfig metadata."""
    cfg_51 = HF_MODEL_CONFIGS["glm5.1-744b"]
    cfg_52 = HF_MODEL_CONFIGS["glm5.2-744b"]

    for cfg in (cfg_51, cfg_52):
      cfg_dict = cfg.to_dict()
      self.assertEqual(cfg_dict["architectures"], ["GlmMoeDsaForCausalLM"])
      self.assertEqual(cfg_dict["model_type"], "glm_moe_dsa")
      self.assertEqual(cfg_dict["rope_parameters"]["rope_type"], "default")

    shapes_51 = HF_SHAPE["glm5.1-744b"](cfg_51.to_dict())
    shapes_52 = HF_SHAPE["glm5.2-744b"](cfg_52.to_dict())

    indexer_keys_51 = [k for k in shapes_51 if ".self_attn.indexer." in k]
    indexer_keys_52 = [k for k in shapes_52 if ".self_attn.indexer." in k]

    # 5 indexer tensors per Full layer: wq_b, wk, k_norm.weight, k_norm.bias, weights_proj
    self.assertEqual(len(indexer_keys_51), 78 * 5)
    self.assertEqual(len(indexer_keys_52), 21 * 5)

    expected_full_layers = {0, 1, 2} | set(range(6, 78, 4))
    actual_indexer_layers_52 = {int(k.split(".")[2]) for k in indexer_keys_52}
    self.assertEqual(actual_indexer_layers_52, expected_full_layers)

    # Non-indexer keys are identical between 5.1 and 5.2
    non_indexer_51 = {k: v for k, v in shapes_51.items() if ".self_attn.indexer." not in k}
    non_indexer_52 = {k: v for k, v in shapes_52.items() if ".self_attn.indexer." not in k}
    self.assertEqual(non_indexer_51, non_indexer_52)

  def test_param_save_dtype_preserves_fp32_router_bias_and_indexer_norms(self):
    """Verifies MoEBiasVar and sensitive indexer norms stay float32 even when save_dtype='bfloat16'."""
    # pylint: disable=protected-access
    self.assertEqual(
        to_maxtext._param_save_dtype("MoEBiasVar-decoder-moe_layers-moe_block-gate-bias", "bfloat16"),
        "float32",
    )
    self.assertEqual(
        to_maxtext._param_save_dtype("params-decoder-layers_0-self_attention-indexer-k_norm-scale", "bfloat16"),
        "float32",
    )
    self.assertEqual(
        to_maxtext._param_save_dtype("params-decoder-layers_0-self_attention-indexer-weights_proj-kernel", "bfloat16"),
        "float32",
    )
    self.assertEqual(
        to_maxtext._param_save_dtype("params-decoder-layers_0-self_attention-out-kernel", "bfloat16"),
        "bfloat16",
    )
    # pylint: enable=protected-access

  @parameterized.named_parameters(
      {"testcase_name": "scanned", "scan_layers": True},
      {"testcase_name": "unscanned_pruned", "scan_layers": False},
  )
  def test_glm52_roundtrip_hf_to_maxtext_to_hf(self, scan_layers: bool):
    """Tests GLM-5.2 (`FFSSFS`) HF -> MaxText -> HF round-trip for scanned and unscanned pruned modes."""
    model_name = "glm5.2-744b"
    mini_hf_cfg = _make_mini_glm_hf_config(use_index_share=True)
    input_tensors = _build_synthetic_hf_tensors(model_name, mini_hf_cfg)

    # Input checkpoint must have indexer keys ONLY on Full layers {0, 1, 4}.
    full_layers = {0, 1, 4}
    shared_layers = {2, 3, 5}
    for layer_idx in full_layers:
      self.assertIn(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", input_tensors)
    for layer_idx in shared_layers:
      self.assertNotIn(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", input_tensors)

    with tempfile.TemporaryDirectory() as tmpdir:
      hf_in_dir = os.path.join(tmpdir, "hf_in")
      mt_ckpt_dir = os.path.join(tmpdir, "mt_ckpt")
      hf_out_dir = os.path.join(tmpdir, "hf_out")
      os.makedirs(hf_in_dir, exist_ok=True)
      save_safetensors_np(input_tensors, os.path.join(hf_in_dir, "model.safetensors"))
      mini_hf_cfg.to_json_file(os.path.join(hf_in_dir, "config.json"))

      cli_overrides = _mini_maxtext_cli_overrides(model_name, scan_layers=scan_layers, use_index_share=True)
      dev_count = max(2, len(jax.devices()))

      with (
          mock.patch.dict(to_maxtext.HF_MODEL_CONFIGS, {model_name: copy.deepcopy(mini_hf_cfg)}),
          mock.patch.dict(to_huggingface.HF_MODEL_CONFIGS, {model_name: copy.deepcopy(mini_hf_cfg)}),
          mock.patch.object(to_huggingface.AutoTokenizer, "from_pretrained", return_value=None),
      ):
        to_maxtext.main(
            [
                "",
                get_test_config_path(),
                f"base_output_directory={mt_ckpt_dir}",
                "run_name=glm52_to_mt",
                *cli_overrides,
            ],
            lazy_load_tensors=True,
            hf_model_path=hf_in_dir,
            save_dtype="float32",
            simulated_cpu_devices_count=dev_count,
        )

        items_dir = self._resolve_orbax_items_dir(mt_ckpt_dir)

        # Inspect converted MaxText checkpoint structure before exporting back to HF.
        mt_cfg = pyconfig.initialize(
            [
                "",
                get_test_config_path(),
                f"load_parameters_path={items_dir}",
                f"base_output_directory={mt_ckpt_dir}",
                "run_name=glm52_inspect",
                *cli_overrides,
            ]
        )
        raw_ckpt = load_orbax_checkpoint(mt_cfg)
        mt_flat = detect_and_extract_checkpoint(raw_ckpt)
        if not scan_layers:
          # Unscanned with prune_shared_indexers=True omits Shared-layer indexer keys in MaxText state.
          def _mt_layer_token(global_idx: int) -> str:
            return f"dense_layers_{global_idx}" if global_idx < 2 else f"moe_layers_{global_idx - 2}"

          for s_idx in shared_layers:
            mt_tok = _mt_layer_token(s_idx)
            self.assertFalse(
                any(f"-{mt_tok}-self_attention-indexer-" in k for k in mt_flat),
                f"Expected no indexer keys for pruned shared layer {s_idx} ({mt_tok})",
            )
          for f_idx in full_layers:
            mt_tok = _mt_layer_token(f_idx)
            self.assertTrue(
                any(f"-{mt_tok}-self_attention-indexer-" in k for k in mt_flat),
                f"Expected indexer keys for full layer {f_idx} ({mt_tok})",
            )

        to_huggingface.main(
            [
                "",
                get_test_config_path(),
                f"load_parameters_path={items_dir}",
                f"base_output_directory={hf_out_dir}",
                "run_name=glm52_to_hf",
                *cli_overrides,
            ]
        )

      exported_tensors = load_safetensors_np(os.path.join(hf_out_dir, "model.safetensors"))
      with open(os.path.join(hf_out_dir, "config.json"), "r", encoding="utf-8") as f:
        exported_cfg = json.load(f)

      # Verify config.json metadata and canonical rope_type="default".
      self.assertEqual(exported_cfg["architectures"], ["GlmMoeDsaForCausalLM"])
      self.assertEqual(exported_cfg["model_type"], "glm_moe_dsa")
      self.assertEqual(exported_cfg["rope_parameters"]["rope_type"], "default")

      # Verify exported HF checkpoint contains indexer keys ONLY on Full layers {0, 1, 4}.
      self.assertEqual(set(exported_tensors.keys()), set(input_tensors.keys()))
      for s_idx in shared_layers:
        self.assertFalse(any(k.startswith(f"model.layers.{s_idx}.self_attn.indexer.") for k in exported_tensors))

      # Explicitly assert exact equality on key representative parameters and across all keys.
      for layer_idx in range(6):
        o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        np.testing.assert_array_equal(exported_tensors[o_key], input_tensors[o_key])
      for layer_idx in full_layers:
        wq_b_key = f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight"
        np.testing.assert_array_equal(exported_tensors[wq_b_key], input_tensors[wq_b_key])
      for layer_idx in range(2, 6):
        bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        self.assertEqual(exported_tensors[bias_key].dtype, np.float32)
        np.testing.assert_array_equal(exported_tensors[bias_key], input_tensors[bias_key])

      for key, expected_arr in input_tensors.items():
        np.testing.assert_array_equal(exported_tensors[key], expected_arr, err_msg=f"Mismatch on key {key}")

  @parameterized.named_parameters(
      {"testcase_name": "scanned", "scan_layers": True},
      {"testcase_name": "unscanned", "scan_layers": False},
  )
  def test_glm51_roundtrip_hf_to_maxtext_to_hf(self, scan_layers: bool):
    """Tests GLM-5.1 (`use_index_share=False`) HF -> MaxText -> HF round-trip with all 6 indexer layers."""
    model_name = "glm5.1-744b"
    mini_hf_cfg = _make_mini_glm_hf_config(use_index_share=False)
    input_tensors = _build_synthetic_hf_tensors(model_name, mini_hf_cfg)

    for layer_idx in range(6):
      self.assertIn(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", input_tensors)

    with tempfile.TemporaryDirectory() as tmpdir:
      hf_in_dir = os.path.join(tmpdir, "hf_in")
      mt_ckpt_dir = os.path.join(tmpdir, "mt_ckpt")
      hf_out_dir = os.path.join(tmpdir, "hf_out")
      os.makedirs(hf_in_dir, exist_ok=True)
      save_safetensors_np(input_tensors, os.path.join(hf_in_dir, "model.safetensors"))
      mini_hf_cfg.to_json_file(os.path.join(hf_in_dir, "config.json"))

      cli_overrides = _mini_maxtext_cli_overrides(model_name, scan_layers=scan_layers, use_index_share=False)
      dev_count = max(2, len(jax.devices()))

      with (
          mock.patch.dict(to_maxtext.HF_MODEL_CONFIGS, {model_name: copy.deepcopy(mini_hf_cfg)}),
          mock.patch.dict(to_huggingface.HF_MODEL_CONFIGS, {model_name: copy.deepcopy(mini_hf_cfg)}),
          mock.patch.object(to_huggingface.AutoTokenizer, "from_pretrained", return_value=None),
      ):
        to_maxtext.main(
            [
                "",
                get_test_config_path(),
                f"base_output_directory={mt_ckpt_dir}",
                "run_name=glm51_to_mt",
                *cli_overrides,
            ],
            lazy_load_tensors=True,
            hf_model_path=hf_in_dir,
            save_dtype="float32",
            simulated_cpu_devices_count=dev_count,
        )

        items_dir = self._resolve_orbax_items_dir(mt_ckpt_dir)
        to_huggingface.main(
            [
                "",
                get_test_config_path(),
                f"load_parameters_path={items_dir}",
                f"base_output_directory={hf_out_dir}",
                "run_name=glm51_to_hf",
                *cli_overrides,
            ]
        )

      exported_tensors = load_safetensors_np(os.path.join(hf_out_dir, "model.safetensors"))
      with open(os.path.join(hf_out_dir, "config.json"), "r", encoding="utf-8") as f:
        exported_cfg = json.load(f)

      self.assertEqual(exported_cfg["architectures"], ["GlmMoeDsaForCausalLM"])
      self.assertEqual(exported_cfg["model_type"], "glm_moe_dsa")
      self.assertEqual(exported_cfg["rope_parameters"]["rope_type"], "default")
      self.assertEqual(set(exported_tensors.keys()), set(input_tensors.keys()))

      for layer_idx in range(6):
        np.testing.assert_array_equal(
            exported_tensors[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            input_tensors[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
        )
        np.testing.assert_array_equal(
            exported_tensors[f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight"],
            input_tensors[f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight"],
        )
      for layer_idx in range(2, 6):
        bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        np.testing.assert_array_equal(exported_tensors[bias_key], input_tensors[bias_key])

      for key, expected_arr in input_tensors.items():
        np.testing.assert_array_equal(exported_tensors[key], expected_arr, err_msg=f"Mismatch on key {key}")

  def test_missing_full_or_donor_indexer_keys_fast_fail(self):
    """Verifies missing donor-layer or Full-layer indexer keys raise ValueError/KeyError instead of zero-filling."""
    dev_count = max(2, len(jax.devices()))

    # 1. GLM-5.2 (`FFSSFS`, scanned): missing donor layer 1 indexer key (even though layer 0 and layer 4 exist)
    # must fail fast both when reading layer 1 (Full) and when resolving donor 1 for Shared layers 2 & 3.
    mini_52_cfg = _make_mini_glm_hf_config(use_index_share=True)
    base_52_tensors = _build_synthetic_hf_tensors("glm5.2-744b", mini_52_cfg)

    for scan_layers, missing_key in (
        (True, "model.layers.1.self_attn.indexer.wq_b.weight"),
        (False, "model.layers.4.self_attn.indexer.wk.weight"),
    ):
      with self.subTest(model="glm5.2-744b", scan_layers=scan_layers, missing_key=missing_key):
        bad_tensors = dict(base_52_tensors)
        del bad_tensors[missing_key]
        with tempfile.TemporaryDirectory() as tmpdir:
          hf_in_dir = os.path.join(tmpdir, "hf_in")
          mt_ckpt_dir = os.path.join(tmpdir, "mt_ckpt")
          os.makedirs(hf_in_dir, exist_ok=True)
          save_safetensors_np(bad_tensors, os.path.join(hf_in_dir, "model.safetensors"))

          cli_overrides = _mini_maxtext_cli_overrides("glm5.2-744b", scan_layers=scan_layers, use_index_share=True)
          with mock.patch.dict(to_maxtext.HF_MODEL_CONFIGS, {"glm5.2-744b": copy.deepcopy(mini_52_cfg)}):
            with self.assertRaises((ValueError, KeyError, SafetensorError)):
              to_maxtext.main(
                  [
                      "",
                      get_test_config_path(),
                      f"base_output_directory={mt_ckpt_dir}",
                      "run_name=glm52_missing_key",
                      *cli_overrides,
                  ],
                  lazy_load_tensors=True,
                  hf_model_path=hf_in_dir,
                  save_dtype="float32",
                  simulated_cpu_devices_count=dev_count,
              )

          # Sharded checkpoint index path: missing key in model.safetensors.index.json raises ValueError immediately.
          hf_sharded_dir = os.path.join(tmpdir, "hf_sharded")
          mt_sharded_ckpt_dir = os.path.join(tmpdir, "mt_sharded_ckpt")
          os.makedirs(hf_sharded_dir, exist_ok=True)
          shard_name = "model-00001-of-00001.safetensors"
          save_safetensors_np(bad_tensors, os.path.join(hf_sharded_dir, shard_name))
          with open(os.path.join(hf_sharded_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as idx_f:
            json.dump({"metadata": {}, "weight_map": {k: shard_name for k in bad_tensors}}, idx_f)

          with mock.patch.dict(to_maxtext.HF_MODEL_CONFIGS, {"glm5.2-744b": copy.deepcopy(mini_52_cfg)}):
            with self.assertRaisesRegex(ValueError, "not found in HF checkpoint index"):
              to_maxtext.main(
                  [
                      "",
                      get_test_config_path(),
                      f"base_output_directory={mt_sharded_ckpt_dir}",
                      "run_name=glm52_missing_index_key",
                      *cli_overrides,
                  ],
                  lazy_load_tensors=True,
                  hf_model_path=hf_sharded_dir,
                  save_dtype="float32",
                  simulated_cpu_devices_count=dev_count,
              )

    # 2. Direct donor-resolution check: Shared layer 2 (`FFSSFS`, donor=1) must NOT fall back to layer 0
    # when donor layer 1's key is missing.
    with tempfile.TemporaryDirectory() as donor_tmpdir:
      cfg_52_obj = pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              f"base_output_directory={donor_tmpdir}",
              "run_name=donor_fast_fail",
              *_mini_maxtext_cli_overrides("glm5.2-744b", scan_layers=True, use_index_share=True),
          ]
      )
    only_layer0_dict = {
        "model.layers.0.self_attn.indexer.wq_b.weight": np.ones((16, 32), dtype=np.float32),
    }

    def strict_getter(k):
      if k not in only_layer0_dict:
        raise KeyError(k)
      return only_layer0_dict[k]

    with self.assertRaisesRegex(
        ValueError, "Missing donor indexer weight 'model.layers.1.self_attn.indexer.wq_b.weight'"
    ):
      # pylint: disable=protected-access
      to_maxtext._resolve_shared_indexer_tensor(
          "model.layers.2.self_attn.indexer.wq_b.weight",
          strict_getter,
          config=cfg_52_obj,
      )
      # pylint: enable=protected-access

    # 3. GLM-5.1 (`use_index_share=False`): missing layer 2 indexer key must NOT borrow from layer 1.
    mini_51_cfg = _make_mini_glm_hf_config(use_index_share=False)
    bad_51_tensors = _build_synthetic_hf_tensors("glm5.1-744b", mini_51_cfg)
    del bad_51_tensors["model.layers.2.self_attn.indexer.wq_b.weight"]
    with tempfile.TemporaryDirectory() as tmpdir:
      hf_in_dir = os.path.join(tmpdir, "hf_in")
      mt_ckpt_dir = os.path.join(tmpdir, "mt_ckpt")
      os.makedirs(hf_in_dir, exist_ok=True)
      save_safetensors_np(bad_51_tensors, os.path.join(hf_in_dir, "model.safetensors"))
      cli_overrides = _mini_maxtext_cli_overrides("glm5.1-744b", scan_layers=True, use_index_share=False)
      with mock.patch.dict(to_maxtext.HF_MODEL_CONFIGS, {"glm5.1-744b": copy.deepcopy(mini_51_cfg)}):
        with self.assertRaises((ValueError, KeyError, SafetensorError)):
          to_maxtext.main(
              [
                  "",
                  get_test_config_path(),
                  f"base_output_directory={mt_ckpt_dir}",
                  "run_name=glm51_missing_key",
                  *cli_overrides,
              ],
              lazy_load_tensors=True,
              hf_model_path=hf_in_dir,
              save_dtype="float32",
              simulated_cpu_devices_count=dev_count,
          )


if __name__ == "__main__":
  unittest.main()
