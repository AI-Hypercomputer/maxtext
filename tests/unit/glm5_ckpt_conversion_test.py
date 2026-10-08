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
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from absl.testing import parameterized
from flax import nnx
import jax
from maxtext.checkpoint_conversion import to_huggingface
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import utils as ckpt_utils
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.utils import detect_and_extract_checkpoint, load_orbax_checkpoint
from maxtext.common import checkpointing
from maxtext.common.common_types import DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.configs.types import DType
from maxtext.layers import moe
from maxtext.utils import checkpoint_dtype_utils
import ml_dtypes
import numpy as np
from safetensors import SafetensorError
from safetensors import safe_open
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
  base_cfg = HF_MODEL_CONFIGS["glm5.2-744b" if use_index_share else "glm5.1-744b"]
  indexer_types = ["full", "full", "shared", "shared", "full", "shared"] if use_index_share else ["full"] * 6
  cfg_dict = {
      "architectures": ["GlmMoeDsaForCausalLM"],
      "attention_bias": False,
      "attention_dropout": 0.0,
      "dtype": "float32",
      "eos_token_id": copy.deepcopy(base_cfg.eos_token_id),
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
      "pad_token_id": base_cfg.pad_token_id,
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
    return transformers.GlmMoeDsaConfig(**cfg_dict)  # pylint: disable=unexpected-keyword-arg
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
      self.assertEqual(cfg.eos_token_id, [154820, 154827, 154829])
      self.assertEqual(cfg.pad_token_id, 154820)
      cfg_dict = cfg.to_dict()
      self.assertEqual(cfg_dict["architectures"], ["GlmMoeDsaForCausalLM"])
      self.assertEqual(cfg_dict["model_type"], "glm_moe_dsa")
      self.assertEqual(cfg_dict["rope_parameters"]["rope_type"], "default")
      self.assertEqual(cfg_dict["eos_token_id"], [154820, 154827, 154829])
      self.assertEqual(cfg_dict["pad_token_id"], 154820)

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

  def test_glm5_hf_configs_export_eos_and_pad_token_ids(self):
    """Verifies GLM-5.1 and GLM-5.2 HF configs preserve list eos_token_id and pad_token_id on export."""
    expected_eos = [154820, 154827, 154829]
    expected_pad = 154820

    for model_name, raw_dict in (
        ("glm5.1-744b", hf_model_configs.glm5_1_744b_dict),
        ("glm5.2-744b", hf_model_configs.glm5_2_744b_dict),
    ):
      with self.subTest(model=model_name):
        self.assertEqual(raw_dict["eos_token_id"], expected_eos)
        self.assertEqual(raw_dict["pad_token_id"], expected_pad)

        cfg = HF_MODEL_CONFIGS[model_name]
        self.assertEqual(cfg.eos_token_id, expected_eos)
        self.assertEqual(cfg.pad_token_id, expected_pad)

        fallback_cfg = hf_model_configs.PTConfig(**raw_dict)
        self.assertEqual(fallback_cfg.eos_token_id, expected_eos)
        self.assertEqual(fallback_cfg.pad_token_id, expected_pad)

        if hasattr(transformers, "GlmMoeDsaConfig"):
          # Without explicit overrides, GlmMoeDsaConfig falls back to eos_token_id=1 and pad_token_id=None.
          default_glm_cfg = transformers.GlmMoeDsaConfig()
          self.assertEqual(default_glm_cfg.eos_token_id, 1)
          self.assertIsNone(default_glm_cfg.pad_token_id)

          explicit_glm_cfg = transformers.GlmMoeDsaConfig(**raw_dict)  # pylint: disable=unexpected-keyword-arg
          self.assertEqual(explicit_glm_cfg.eos_token_id, expected_eos)
          self.assertEqual(explicit_glm_cfg.pad_token_id, expected_pad)

        with tempfile.TemporaryDirectory() as tmpdir:
          ckpt_utils.save_config_file(copy.deepcopy(cfg), tmpdir, tmpdir, "config.json")
          with open(os.path.join(tmpdir, "config.json"), "r", encoding="utf-8") as f:
            saved_json = json.load(f)
          self.assertEqual(saved_json["eos_token_id"], expected_eos)
          self.assertEqual(saved_json["pad_token_id"], expected_pad)

          if hasattr(transformers, "GlmMoeDsaConfig"):
            reloaded_cfg = transformers.AutoConfig.from_pretrained(tmpdir)
            self.assertEqual(reloaded_cfg.eos_token_id, expected_eos)
            self.assertEqual(reloaded_cfg.pad_token_id, expected_pad)
            gen_cfg = transformers.GenerationConfig.from_model_config(reloaded_cfg)
            self.assertEqual(gen_cfg.eos_token_id, expected_eos)
            self.assertEqual(gen_cfg.pad_token_id, expected_pad)

  def test_param_save_dtype_preserves_fp32_router_bias_and_indexer_norms(self):
    """Verifies shared checkpoint_dtype_utils and to_maxtext._param_save_dtype preserve fp32 state."""
    # pylint: disable=protected-access
    for fn in (to_maxtext._param_save_dtype, checkpoint_dtype_utils.param_save_dtype):
      self.assertEqual(
          fn("MoEBiasVar-decoder-moe_layers-moe_block-gate-bias", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("Tid2EidVar-decoder-moe_layers-moe_block-tid2eid", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("params-decoder-layers_0-attn_hc-fn", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("params-decoder-layers_0-ffn_hc-base", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("params-decoder-hc_head-scale", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("params-decoder-layers_0-self_attention-sinks", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("params-decoder-layers_0-self_attention-indexer-k_norm-scale", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("params-decoder-layers_0-self_attention-indexer-weights_proj-kernel", "bfloat16"),
          "float32",
      )
      self.assertEqual(
          fn("params-decoder-layers_0-self_attention-out-kernel", "bfloat16"),
          "bfloat16",
      )
    # pylint: enable=protected-access

    # Verify NNX (path, var) matching aligns with `-`-delimited keys across all canonical rules.
    dummy_param = nnx.Param(jax.numpy.ones((2,)))
    dummy_bias = moe.MoEBiasVar(jax.numpy.ones((2,)))
    dummy_tid = moe.Tid2EidVar(jax.numpy.ones((2,)))
    self.assertTrue(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(
            ("decoder", "moe_layers_0", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias"), dummy_bias
        )
    )
    self.assertTrue(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(("decoder", "moe_layers_0", "tid2eid"), dummy_tid)
    )
    self.assertTrue(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(("decoder", "layers", 0, "attn_hc", "fn"), dummy_param)
    )
    self.assertTrue(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(("decoder", "layers_0", "ffn_hc", "base"), dummy_param)
    )
    self.assertTrue(checkpoint_dtype_utils.should_keep_full_precision_nnx(("decoder", "hc_head", "scale"), dummy_param))
    self.assertTrue(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(
            ("decoder", "layers", 0, "self_attention", "sinks"), dummy_param
        )
    )
    self.assertTrue(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(
            ("decoder", "moe_layers_0", "self_attention", "indexer", "k_norm", "scale"), dummy_param
        )
    )
    self.assertTrue(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(
            ("decoder", "moe_layers_0", "self_attention", "indexer", "weights_proj", "kernel"), dummy_param
        )
    )
    self.assertFalse(
        checkpoint_dtype_utils.should_keep_full_precision_nnx(
            ("decoder", "moe_layers_0", "self_attention", "indexer", "wq_b", "kernel"), dummy_param
        )
    )

    # Verify importing checkpoint_dtype_utils in a clean process pulls in neither torch nor transformers.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import maxtext.utils.checkpoint_dtype_utils; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    self.assertEqual(proc.returncode, 0, msg=proc.stderr)

  def test_get_model_mappings_dispatches_saving_to_hf_via_signature(self):
    """Verifies _get_model_mappings passes saving_to_hf=True iff the mapping signature accepts it."""
    calls: list[tuple[str, dict]] = []

    def mapping_with_kwarg(hf_cfg, mt_cfg, scan_layers=False, saving_to_hf=False):
      del hf_cfg, mt_cfg
      calls.append(("explicit", {"scan_layers": scan_layers, "saving_to_hf": saving_to_hf}))
      return {"params-token_embedder-embedding": "model.embed_tokens.weight"}

    def mapping_with_var_kwargs(hf_cfg, mt_cfg, scan_layers=False, **kwargs):
      del hf_cfg, mt_cfg
      calls.append(("var_kw", {"scan_layers": scan_layers, **kwargs}))
      return {"params-token_embedder-embedding": "model.embed_tokens.weight"}

    def mapping_without_kwarg(hf_cfg, mt_cfg, scan_layers=False):
      del hf_cfg, mt_cfg
      calls.append(("legacy", {"scan_layers": scan_layers}))
      return {"params-token_embedder-embedding": "model.embed_tokens.weight"}

    def dummy_shape_fn(hf_cfg):
      del hf_cfg
      return {"model.embed_tokens.weight": (8, 4)}

    def dummy_hook_fn(hf_cfg, mt_cfg, scan_layers=False, saving_to_hf=False):
      del hf_cfg, mt_cfg, scan_layers, saving_to_hf
      return {}

    mt_cfg = SimpleNamespace()

    # pylint: disable=protected-access
    with (
        mock.patch.dict(
            to_huggingface.PARAM_MAPPING,
            {
                "custom-explicit": mapping_with_kwarg,
                "custom-varkw": mapping_with_var_kwargs,
                "custom-legacy": mapping_without_kwarg,
            },
        ),
        mock.patch.dict(
            to_huggingface.HF_SHAPE,
            {
                "custom-explicit": dummy_shape_fn,
                "custom-varkw": dummy_shape_fn,
                "custom-legacy": dummy_shape_fn,
            },
        ),
        mock.patch.dict(
            to_huggingface.HOOK_FNS,
            {
                "custom-explicit": dummy_hook_fn,
                "custom-varkw": dummy_hook_fn,
                "custom-legacy": dummy_hook_fn,
            },
        ),
    ):
      to_huggingface._get_model_mappings("custom-explicit", True, {}, mt_cfg)
      to_huggingface._get_model_mappings("custom-varkw", False, {}, mt_cfg)
      to_huggingface._get_model_mappings("custom-legacy", True, {}, mt_cfg)
    # pylint: enable=protected-access

    self.assertEqual(
        calls,
        [
            ("explicit", {"scan_layers": True, "saving_to_hf": True}),
            ("var_kw", {"scan_layers": False, "saving_to_hf": True}),
            ("legacy", {"scan_layers": True}),
        ],
    )

  def test_yarn_rope_export_exemption_keyed_on_glm5_decoder_block(self):
    """Verifies GLM5 + rope_factor==1.0 skips _apply_yarn_rope_config while non-GLM yarn factor==1.0 does not."""
    # pylint: disable=protected-access
    for glm_block in (DecoderBlockType.GLM5, "glm5"):
      for factor in (1.0, 1):
        mt_glm = SimpleNamespace(decoder_block=glm_block, rope_type="yarn", rope_factor=factor)
        self.assertFalse(to_huggingface._should_apply_yarn_rope_config(mt_glm))

      # Non-unit factor on GLM5 IS NOT exempted.
      mt_glm_scaled = SimpleNamespace(
          decoder_block=glm_block,
          rope_type="yarn",
          rope_factor=2.0,
          rope_max_timescale=8000000.0,
          original_max_position_embeddings=4096,
          beta_fast=32.0,
          beta_slow=1.0,
          rope_truncate=True,
      )
      self.assertTrue(to_huggingface._should_apply_yarn_rope_config(mt_glm_scaled))

      # Canonical GLM HF config keeps rope_type="default" and has no rope_scaling when exempted.
      glm_hf_cfg = _make_mini_glm_hf_config(use_index_share=True)
      if to_huggingface._should_apply_yarn_rope_config(
          SimpleNamespace(decoder_block=glm_block, rope_type="yarn", rope_factor=1.0)
      ):
        to_huggingface._apply_yarn_rope_config(glm_hf_cfg, mt_glm_scaled)
      glm_dict = glm_hf_cfg.to_dict()
      self.assertNotIn("rope_scaling", glm_dict)
      self.assertEqual(glm_dict["rope_parameters"]["rope_type"], "default")

    # Non-GLM models with rope_type="yarn" and rope_factor=1.0 (e.g. deepseek-custom, ming-3.0-audio-tiny)
    # MUST NOT be exempted.
    for non_glm_block in (
        DecoderBlockType.DEEPSEEK,
        "deepseek",
        DecoderBlockType.MING,
        "ming",
    ):
      mt_non_glm = SimpleNamespace(
          decoder_block=non_glm_block,
          rope_type="yarn",
          rope_factor=1.0,
          rope_max_timescale=10000.0,
          original_max_position_embeddings=4096,
          beta_fast=32.0,
          beta_slow=1.0,
          rope_truncate=True,
      )
      self.assertTrue(to_huggingface._should_apply_yarn_rope_config(mt_non_glm))
      non_glm_hf_cfg = transformers.PreTrainedConfig()
      to_huggingface._apply_yarn_rope_config(non_glm_hf_cfg, mt_non_glm)
      self.assertIsNotNone(non_glm_hf_cfg.rope_scaling)
      self.assertEqual(non_glm_hf_cfg.rope_scaling["rope_type"], "yarn")
      self.assertEqual(non_glm_hf_cfg.rope_scaling["factor"], 1.0)
      non_glm_dict = non_glm_hf_cfg.to_dict()
      serialized_rope = non_glm_dict.get("rope_scaling") or non_glm_dict.get("rope_parameters")
      self.assertIsNotNone(serialized_rope)
      self.assertEqual(serialized_rope["rope_type"], "yarn")
      self.assertEqual(serialized_rope["factor"], 1.0)
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

      # Verify config.json metadata and canonical rope_type="default" with no rope_scaling.
      self.assertEqual(exported_cfg["architectures"], ["GlmMoeDsaForCausalLM"])
      self.assertEqual(exported_cfg["model_type"], "glm_moe_dsa")
      self.assertNotIn("rope_scaling", exported_cfg)
      self.assertEqual(exported_cfg["rope_parameters"]["rope_type"], "default")
      self.assertEqual(exported_cfg["eos_token_id"], [154820, 154827, 154829])
      self.assertEqual(exported_cfg["pad_token_id"], 154820)

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
      self.assertNotIn("rope_scaling", exported_cfg)
      self.assertEqual(exported_cfg["rope_parameters"]["rope_type"], "default")
      self.assertEqual(exported_cfg["eos_token_id"], [154820, 154827, 154829])
      self.assertEqual(exported_cfg["pad_token_id"], 154820)
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

  def test_process_maxtext_param_preserves_fp32_leaves_across_all_stack_cases(self):
    """Verifies process_maxtext_param keeps fp32 router/indexer state and casts regular weights to bf16 in all 4 cases."""
    fp32_fine = np.array([34.62297058, 1.0 + 2**-16, -3.0 - 2**-15, 16.0600128], dtype=np.float32)
    # Sanity-check that bf16 rounding would corrupt every element of fp32_fine.
    self.assertFalse(np.any(fp32_fine.astype(ml_dtypes.bfloat16).astype(np.float32) == fp32_fine))

    for weight_dtype in ("bfloat16", DType.BFLOAT16):
      # Case 1: Unscanned 1-to-1 mappings.
      cfg_unscan = SimpleNamespace(scan_layers=False, param_scan_axis=1, weight_dtype=weight_dtype)
      unscan_map = {
          "MoEBiasVar-decoder-moe_layers_0-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias": (
              "model.layers.2.mlp.gate.e_score_correction_bias"
          ),
          "params-decoder-moe_layers_0-self_attention-indexer-k_norm-scale": (
              "model.layers.2.self_attn.indexer.k_norm.weight"
          ),
          "params-decoder-moe_layers_0-self_attention-indexer-k_norm-bias": (
              "model.layers.2.self_attn.indexer.k_norm.bias"
          ),
          "params-decoder-moe_layers_0-self_attention-indexer-weights_proj-kernel": (
              "model.layers.2.self_attn.indexer.weights_proj.weight"
          ),
          "params-decoder-moe_layers_0-self_attention-indexer-wq_b-kernel": (
              "model.layers.2.self_attn.indexer.wq_b.weight"
          ),
          "params-token_embedder-embedding": "model.embed_tokens.weight",
      }
      unscan_shapes = {hf_k: (4,) for hf_k in unscan_map.values()}
      for mt_key, hf_key in unscan_map.items():
        out = dict(
            ckpt_utils.process_maxtext_param(
                mt_key, jax.numpy.asarray(fp32_fine), unscan_map, {}, unscan_shapes, cfg_unscan
            )
        )
        if checkpoint_dtype_utils.is_full_precision_param_key(mt_key):
          self.assertEqual(out[hf_key].dtype, np.float32)
          np.testing.assert_array_equal(out[hf_key], fp32_fine)
        else:
          self.assertEqual(out[hf_key].dtype, ml_dtypes.bfloat16)
          np.testing.assert_array_equal(out[hf_key], fp32_fine.astype(ml_dtypes.bfloat16))

      # Case 2 & Case 3: 1D stacked mappings (scanned layers & unscanned MoE experts).
      cfg_scan = SimpleNamespace(scan_layers=True, param_scan_axis=1, weight_dtype=weight_dtype)
      stacked_axis0 = jax.numpy.stack([fp32_fine, fp32_fine + np.float32(2.0)], axis=0)  # [2, 4] for MoEBiasVar
      stacked_axis1 = jax.numpy.stack([fp32_fine, fp32_fine + np.float32(2.0)], axis=1)  # [4, 2] for param_scan_axis=1
      scan_1d_cases = [
          (
              "MoEBiasVar-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias",
              stacked_axis0,
              ["l2.bias", "l3.bias"],
              np.float32,
          ),
          (
              "params-decoder-moe_layers-self_attention-indexer-k_norm-scale",
              stacked_axis1,
              ["l2.knorm_w", "l3.knorm_w"],
              np.float32,
          ),
          (
              "params-decoder-moe_layers-self_attention-indexer-k_norm-bias",
              stacked_axis1,
              ["l2.knorm_b", "l3.knorm_b"],
              np.float32,
          ),
          (
              "params-decoder-moe_layers-self_attention-indexer-weights_proj-kernel",
              stacked_axis1,
              ["l2.wproj", "l3.wproj"],
              np.float32,
          ),
          (
              "params-decoder-moe_layers-self_attention-indexer-wq_b-kernel",
              stacked_axis1,
              ["l2.wqb", "l3.wqb"],
              ml_dtypes.bfloat16,
          ),
      ]
      for mt_key, weight, hf_targets, expected_dtype in scan_1d_cases:
        out = dict(
            ckpt_utils.process_maxtext_param(
                mt_key, weight, {mt_key: hf_targets}, {}, {k: (4,) for k in hf_targets}, cfg_scan
            )
        )
        for i, hf_k in enumerate(hf_targets):
          expected_slice = fp32_fine + np.float32(2.0 * i)
          self.assertEqual(out[hf_k].dtype, expected_dtype)
          if expected_dtype == np.float32:
            np.testing.assert_array_equal(out[hf_k], expected_slice)
          else:
            np.testing.assert_array_equal(out[hf_k], expected_slice.astype(ml_dtypes.bfloat16))

      # Case 3: Unscanned MoE 1D expert stack.
      unscan_moe_key = "params-decoder-moe_layers_0-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"
      unscan_moe_targets = ["e0.gate", "e1.gate"]
      out_unscan_moe = dict(
          ckpt_utils.process_maxtext_param(
              unscan_moe_key,
              stacked_axis0,
              {unscan_moe_key: unscan_moe_targets},
              {},
              {k: (4,) for k in unscan_moe_targets},
              cfg_unscan,
          )
      )
      for i, hf_k in enumerate(unscan_moe_targets):
        self.assertEqual(out_unscan_moe[hf_k].dtype, ml_dtypes.bfloat16)
        np.testing.assert_array_equal(out_unscan_moe[hf_k], (fp32_fine + np.float32(2.0 * i)).astype(ml_dtypes.bfloat16))

      # Case 4: Multi-axis stacked mappings (_emit_slices) for both regular MoE experts and nested-scan fp32 leaves.
      multi_regular_key = "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"
      multi_regular_targets = [["e0_l0", "e0_l1"], ["e1_l0", "e1_l1"]]
      multi_regular_weight = jax.numpy.stack([stacked_axis0, stacked_axis0 + np.float32(4.0)], axis=0)  # [2, 2, 4]
      out_multi_reg = dict(
          ckpt_utils.process_maxtext_param(
              multi_regular_key,
              multi_regular_weight,
              {multi_regular_key: multi_regular_targets},
              {},
              {"e0_l0": (4,), "e0_l1": (4,), "e1_l0": (4,), "e1_l1": (4,)},
              cfg_scan,
          )
      )
      for hf_k in ("e0_l0", "e0_l1", "e1_l0", "e1_l1"):
        self.assertEqual(out_multi_reg[hf_k].dtype, ml_dtypes.bfloat16)

      multi_fp32_key = "params-decoder-layers-local_layers-self_attention-indexer-weights_proj-kernel"
      multi_fp32_targets = [["b0_l0", "b0_l1"], ["b1_l0", "b1_l1"]]
      # Nested local scan places (blocks, local) at (param_scan_axis, param_scan_axis + 1) = (1, 2): shape [4, 2, 2].
      multi_fp32_weight = jax.numpy.stack([stacked_axis1, stacked_axis1 + np.float32(4.0)], axis=1)
      out_multi_fp32 = dict(
          ckpt_utils.process_maxtext_param(
              multi_fp32_key,
              multi_fp32_weight,
              {multi_fp32_key: multi_fp32_targets},
              {},
              {"b0_l0": (4,), "b0_l1": (4,), "b1_l0": (4,), "b1_l1": (4,)},
              cfg_scan,
          )
      )
      self.assertEqual(out_multi_fp32["b0_l0"].dtype, np.float32)
      np.testing.assert_array_equal(out_multi_fp32["b0_l0"], fp32_fine)
      self.assertEqual(out_multi_fp32["b1_l1"].dtype, np.float32)
      np.testing.assert_array_equal(out_multi_fp32["b1_l1"], fp32_fine + np.float32(6.0))

  @parameterized.named_parameters(
      {"testcase_name": "scanned", "scan_layers": True},
      {"testcase_name": "unscanned_pruned", "scan_layers": False},
  )
  def test_glm52_bfloat16_export_preserves_fp32_router_bias_and_indexer_params(self, scan_layers: bool):
    """Tests GLM-5.2 HF export with weight_dtype=bfloat16 keeps fp32 router bias & indexer k_norm/weights_proj."""
    model_name = "glm5.2-744b"
    mini_hf_cfg = _make_mini_glm_hf_config(use_index_share=True)
    input_tensors = _build_synthetic_hf_tensors(model_name, mini_hf_cfg)

    fp32_suffixes = (
        ".mlp.gate.e_score_correction_bias",
        ".self_attn.indexer.k_norm.weight",
        ".self_attn.indexer.k_norm.bias",
        ".self_attn.indexer.weights_proj.weight",
    )
    # Perturb full-precision tensors with values that bfloat16 cannot represent exactly, so any premature
    # bf16 truncation in to_maxtext, load_orbax_checkpoint, or process_maxtext_param is caught immediately.
    for key in list(input_tensors.keys()):
      if key.endswith(fp32_suffixes):
        base = np.float32(34.62297058) + np.abs(input_tensors[key]).astype(np.float32) + np.float32(2**-16)
        # Ensure every element differs from its bf16 round-trip.
        rounded = base.astype(ml_dtypes.bfloat16).astype(np.float32)
        base = np.where(base == rounded, base + np.float32(2**-6), base).astype(np.float32)
        self.assertFalse(np.any(base.astype(ml_dtypes.bfloat16).astype(np.float32) == base), msg=key)
        input_tensors[key] = np.ascontiguousarray(base)

    full_layers = {0, 1, 4}
    shared_layers = {2, 3, 5}

    with tempfile.TemporaryDirectory() as tmpdir:
      hf_in_dir = os.path.join(tmpdir, "hf_in")
      mt_ckpt_dir = os.path.join(tmpdir, "mt_ckpt")
      hf_out_dir = os.path.join(tmpdir, "hf_out")
      os.makedirs(hf_in_dir, exist_ok=True)
      save_safetensors_np(input_tensors, os.path.join(hf_in_dir, "model.safetensors"))
      mini_hf_cfg.to_json_file(os.path.join(hf_in_dir, "config.json"))

      base_overrides = [
          o
          for o in _mini_maxtext_cli_overrides(model_name, scan_layers=scan_layers, use_index_share=True)
          if not o.startswith("weight_dtype=")
      ]
      cli_overrides = [*base_overrides, "weight_dtype=bfloat16"]
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
                "run_name=glm52_bf16_to_mt",
                *cli_overrides,
            ],
            lazy_load_tensors=True,
            hf_model_path=hf_in_dir,
            save_dtype="bfloat16",
            simulated_cpu_devices_count=dev_count,
        )

        items_dir = self._resolve_orbax_items_dir(mt_ckpt_dir)
        to_huggingface.main(
            [
                "",
                get_test_config_path(),
                f"load_parameters_path={items_dir}",
                f"base_output_directory={hf_out_dir}",
                "run_name=glm52_bf16_to_hf",
                *cli_overrides,
            ]
        )

      out_safetensors_path = os.path.join(hf_out_dir, "model.safetensors")
      exported_tensors = load_safetensors_np(out_safetensors_path)
      with safe_open(out_safetensors_path, framework="np") as st_file:
        safetensor_header_dtypes = {k: st_file.get_slice(k).get_dtype() for k in st_file.keys()}

      self.assertEqual(set(exported_tensors.keys()), set(input_tensors.keys()))
      for s_idx in shared_layers:
        self.assertFalse(any(k.startswith(f"model.layers.{s_idx}.self_attn.indexer.") for k in exported_tensors))

      # Explicitly verify full-precision routing bias and indexer k_norm / weights_proj on Full layers.
      for layer_idx in range(2, 6):
        bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        self.assertEqual(safetensor_header_dtypes[bias_key], "F32")
        self.assertEqual(exported_tensors[bias_key].dtype, np.float32)
        np.testing.assert_array_equal(exported_tensors[bias_key], input_tensors[bias_key])

      for layer_idx in full_layers:
        for idx_fp32_suffix in (
            "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.weights_proj.weight",
        ):
          idx_key = f"model.layers.{layer_idx}.{idx_fp32_suffix}"
          self.assertEqual(safetensor_header_dtypes[idx_key], "F32")
          self.assertEqual(exported_tensors[idx_key].dtype, np.float32)
          np.testing.assert_array_equal(exported_tensors[idx_key], input_tensors[idx_key])

        for idx_bf16_suffix in ("self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight"):
          idx_bf16_key = f"model.layers.{layer_idx}.{idx_bf16_suffix}"
          self.assertEqual(safetensor_header_dtypes[idx_bf16_key], "BF16")
          self.assertEqual(exported_tensors[idx_bf16_key].dtype, ml_dtypes.bfloat16)

      # Verify representative regular weights (embed_tokens, wq_b, experts, lm_head) and exhaustive check over all keys.
      for bf16_rep_key in (
          "model.embed_tokens.weight",
          "lm_head.weight",
          "model.layers.0.self_attn.q_b_proj.weight",
          "model.layers.2.mlp.experts.0.gate_proj.weight",
          "model.layers.2.mlp.experts.0.down_proj.weight",
      ):
        self.assertEqual(safetensor_header_dtypes[bf16_rep_key], "BF16")
        self.assertEqual(exported_tensors[bf16_rep_key].dtype, ml_dtypes.bfloat16)

      for key, expected_arr in input_tensors.items():
        if key.endswith(fp32_suffixes):
          self.assertEqual(safetensor_header_dtypes[key], "F32", msg=key)
          self.assertEqual(exported_tensors[key].dtype, np.float32, msg=key)
          np.testing.assert_array_equal(exported_tensors[key], expected_arr, err_msg=f"FP32 mismatch on {key}")
        else:
          self.assertEqual(safetensor_header_dtypes[key], "BF16", msg=key)
          self.assertEqual(exported_tensors[key].dtype, ml_dtypes.bfloat16, msg=key)
          np.testing.assert_array_equal(
              exported_tensors[key],
              expected_arr.astype(ml_dtypes.bfloat16),
              err_msg=f"BF16 mismatch on {key}",
          )

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
