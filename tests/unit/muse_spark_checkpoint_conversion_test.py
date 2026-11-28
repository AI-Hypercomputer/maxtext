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

"""Bidirectional pure-JAX/NumPy checkpoint conversion tests for Muse Spark (`muse-spark-816b-a42b`).

Runs in default CPU CI without requiring `torch` or `transformers.models.muse_spark`.
Covers `HF -> MaxText` (`saving_to_hf=False`) and `MaxText -> HF` (`saving_to_hf=True` via
`process_maxtext_param`), scanned and unscanned layouts, composite `(wi_0, wi_1)` expert hooks,
`MoEBiasVar` side-collection keys, float32 preservation under bfloat16 export, and forward logit
agreement between scanned and unscanned models loaded from the converted weights.
"""

from __future__ import annotations

import copy
import os
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.nnx import variablelib
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import ml_dtypes
import numpy as np
import pytest

from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.checkpoint_conversion.utils.hf_model_configs import (
    HF_MODEL_CONFIGS,
    PTConfig,
    muse_spark_816b_a42b_dict,
)
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.tensor_handling import apply_hook_fns, single_stack_axis
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import muse_spark
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import maxtext_utils, model_creation_utils

pytestmark = [pytest.mark.cpu_only]

MODEL_NAME = "muse-spark-816b-a42b"
BASE_CONFIG = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")

_NUM_LAYERS = 6  # layer 0 dense, layer 1 unrolled MoE prefix, layers 2..5 one scanned 4-layer cycle
_FIRST_DENSE = 1
_CYCLE = 4
_EMB = 32
_Q_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16
_MLP = 64
_MOE_IN = 16
_MOE_MLP = 24
_EXPERTS = 4
_TOPK = 2
_VOCAB = 64
_SEQ_LEN = 8


def _toy_hf_config() -> dict:
  """Builds a miniature HuggingFace Muse Spark config dict for fast CPU tests."""
  cfg = copy.deepcopy(muse_spark_816b_a42b_dict)
  tcfg = cfg["text_config"]
  tcfg.update(
      vocab_size=_VOCAB,
      hidden_size=_EMB,
      intermediate_size=_MLP,
      moe_hidden_size=_MOE_IN,
      moe_intermediate_size=_MOE_MLP,
      num_hidden_layers=_NUM_LAYERS,
      num_attention_heads=_Q_HEADS,
      num_key_value_heads=_KV_HEADS,
      head_dim=_HEAD_DIM,
      num_local_experts=_EXPERTS,
      num_experts_per_tok=_TOPK,
      sliding_window=4,
      layer_rope_theta=[0 if (_NUM_LAYERS - 1 - i) % _CYCLE == 0 else 500000.0 for i in range(_NUM_LAYERS)],
      layer_types=[
          "full_attention" if (_NUM_LAYERS - 1 - i) % _CYCLE == 0 else "sliding_attention" for i in range(_NUM_LAYERS)
      ],
      mlp_layer_types=["dense" if i < _FIRST_DENSE else "sparse" for i in range(_NUM_LAYERS)],
  )
  return cfg


def _toy_maxtext_config(scan_layers: bool, **overrides):
  """Builds a miniature MaxText Muse Spark config matching `_toy_hf_config()`."""
  kw = {
      "model_name": MODEL_NAME,
      "override_model_config": True,
      "base_emb_dim": _EMB,
      "base_num_query_heads": _Q_HEADS,
      "base_num_kv_heads": _KV_HEADS,
      "head_dim": _HEAD_DIM,
      "base_mlp_dim": _MLP,
      "moe_expert_input_dim": _MOE_IN,
      "base_moe_mlp_dim": _MOE_MLP,
      "num_experts": _EXPERTS,
      "num_experts_per_tok": _TOPK,
      "shared_experts": 0,
      "routed_scaling_factor": 1.0,
      "routed_score_func": "sigmoid",
      "routed_bias": True,
      "norm_topk_prob": True,
      "first_num_dense_layers": _FIRST_DENSE,
      "inhomogeneous_layer_cycle_interval": _CYCLE,
      "sliding_window_size": 4,
      "vocab_size": _VOCAB,
      "base_num_decoder_layers": _NUM_LAYERS,
      "per_device_batch_size": 1,
      "max_target_length": _SEQ_LEN,
      "max_prefill_predict_length": _SEQ_LEN // 2,
      "max_position_embeddings": 64,
      "original_max_position_embeddings": 64,
      "scan_layers": scan_layers,
      "attention": "dot_product",
      "sparse_matmul": False,
      "capacity_factor": -1,
      "matmul_precision": "highest",
      "dtype": "float32",
      "weight_dtype": "float32",
      "float32_logits": True,
      "float32_qk_product": True,
      "dropout_rate": 0.0,
      "skip_jax_distributed_system": True,
      "hardware": "cpu",
  }
  kw.update(overrides)
  return pyconfig.initialize(["", BASE_CONFIG], **kw)


def _param_key_from_nnx_path(collection: str, path: tuple) -> str:
  """Converts an NNX collection name and variable path tuple to a hyphenated MaxText key."""
  parts = [collection]
  for entry in path:
    key = entry.key if hasattr(entry, "key") else str(entry)
    parts.append(str(key))
  return "-".join(parts)


def _abstract_model_shapes(config, mesh) -> dict[str, tuple[int, ...]]:
  """Returns abstract parameter shapes keyed by hyphenated MaxText parameter key."""
  _, abstract_model = model_creation_utils.create_nnx_abstract_model(config, mesh)
  shapes = {}
  for path, var in nnx.to_flat_state(nnx.state(abstract_model)):
    if isinstance(var, (nnx.RngState, nnx.Intermediate, nnx.Cache)):
      continue
    col = variablelib.variable_name_from_type(type(var), allow_register=True)
    key = _param_key_from_nnx_path(col, path)
    shapes[key] = tuple(var.get_value().shape)
  return shapes


def _hf_leaf_keys(hf_val) -> list[str]:
  """Flattens nested HF key lists into a flat list of safetensors leaf keys."""
  if isinstance(hf_val, list):
    return [k for item in hf_val for k in _hf_leaf_keys(item)]
  return [hf_val]


def _convert_hf_to_maxtext(hf_state: dict[str, np.ndarray], hf_cfg: dict, mt_cfg, mt_shapes: dict[str, tuple[int, ...]]):
  """Converts a synthetic HF state dict into a MaxText parameter dict using `PARAM_MAPPING` and `HOOK_FNS`."""
  mapping = PARAM_MAPPING[MODEL_NAME](hf_cfg, mt_cfg, scan_layers=mt_cfg.scan_layers)
  hooks = HOOK_FNS[MODEL_NAME](hf_cfg, mt_cfg, scan_layers=mt_cfg.scan_layers, saving_to_hf=False)
  out: dict[str, np.ndarray] = {}
  for mt_key, hf_src in mapping.items():
    hook = hooks.get(mt_key)
    if isinstance(mt_key, tuple):
      target_shape = [mt_shapes[k] for k in mt_key]
    else:
      target_shape = mt_shapes[mt_key]

    if isinstance(hf_src, list):
      sample_shape = target_shape[0] if isinstance(target_shape, list) else target_shape
      axis = single_stack_axis(mt_key, mt_cfg, sample_shape)
      slice_shape = (
          [s[:axis] + s[axis + 1 :] for s in target_shape]
          if isinstance(target_shape, list)
          else target_shape[:axis] + target_shape[axis + 1 :]
      )
      converted = np.stack([apply_hook_fns(hf_state[k], slice_shape, hook) for k in hf_src], axis=axis)
    else:
      converted = apply_hook_fns(hf_state[hf_src], target_shape, hook)

    if isinstance(mt_key, tuple):
      for idx, sub_key in enumerate(mt_key):
        out[sub_key] = converted[..., idx]
    else:
      out[mt_key] = converted
  return out


def _load_converted_into_nnx(model, converted: dict[str, np.ndarray]) -> int:
  """Loads converted NumPy parameter arrays into a live NNX Muse Spark model."""
  flat = nnx.to_flat_state(nnx.state(model))
  updated = []
  loaded = 0
  for path, var in flat:
    if isinstance(var, (nnx.RngState, nnx.Intermediate, nnx.Cache)):
      updated.append((path, var))
      continue
    col = variablelib.variable_name_from_type(type(var), allow_register=True)
    mt_key = _param_key_from_nnx_path(col, path)
    arr = jnp.asarray(converted[mt_key], dtype=var.get_value().dtype)
    updated.append((path, var.replace(arr)))
    loaded += 1
  nnx.update(model, nnx.from_flat_state(updated))
  return loaded


class MuseSparkCheckpointConversionTest(parameterized.TestCase):
  """Pure-JAX/NumPy bidirectional checkpoint conversion tests for `muse-spark-816b-a42b`."""

  @classmethod
  def setUpClass(cls):
    """Initializes miniature scanned/unscanned Muse Spark configs and synthetic HF weights."""
    super().setUpClass()
    cls.hf_cfg = _toy_hf_config()
    cls.cfg = {scan: _toy_maxtext_config(scan) for scan in (False, True)}
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cls.cfg[False]), cls.cfg[False].mesh_axes)
    cls.shapes = {scan: _abstract_model_shapes(cls.cfg[scan], cls.mesh) for scan in (False, True)}
    cls.mapping = {scan: PARAM_MAPPING[MODEL_NAME](cls.hf_cfg, cls.cfg[scan], scan_layers=scan) for scan in (False, True)}
    cls.hf_shapes = HF_SHAPE[MODEL_NAME](cls.hf_cfg)

    rng = np.random.default_rng(42)
    cls.hf_state = {}
    for hf_key, shape in sorted(cls.hf_shapes.items()):
      vals = (rng.standard_normal(tuple(shape)) * 0.1).astype(np.float32)
      cls.hf_state[hf_key] = vals

  def test_registries_contain_muse_spark(self):
    """Verifies `muse-spark-816b-a42b` is registered and its `PTConfig` fallback exposes `text_config` attributes."""
    self.assertEqual(maxtext_globals.HF_IDS[MODEL_NAME], "meta-models/Muse-Spark-Open-816B-A42B")
    self.assertIn(MODEL_NAME, HF_MODEL_CONFIGS)
    self.assertIn(MODEL_NAME, HF_SHAPE)
    self.assertIn(MODEL_NAME, PARAM_MAPPING)
    self.assertIn(MODEL_NAME, HOOK_FNS)
    hf_cfg_obj = HF_MODEL_CONFIGS[MODEL_NAME]
    self.assertEqual(hf_cfg_obj.text_config.hidden_size, 8192)
    fallback_cfg = PTConfig(
        **{**self.hf_cfg, "text_config": PTConfig(**self.hf_cfg["text_config"])}  # pytype: disable=wrong-arg-types
    )
    self.assertEqual(fallback_cfg.text_config.hidden_size, _EMB)
    self.assertEqual(HF_SHAPE[MODEL_NAME](fallback_cfg), self.hf_shapes)

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_mapping_and_shapes_bijection(self, scan_layers: bool):
    """Verifies 1-to-1 bijection between HF safetensors keys and MaxText NNX parameter keys."""
    mapping = self.mapping[scan_layers]
    hooks = HOOK_FNS[MODEL_NAME](self.hf_cfg, self.cfg[scan_layers], scan_layers=scan_layers, saving_to_hf=False)
    flat_mt_keys = set()
    for k in mapping:
      if isinstance(k, tuple):
        flat_mt_keys.update(k)
      else:
        flat_mt_keys.add(k)
    self.assertEqual(flat_mt_keys, set(self.shapes[scan_layers]))
    self.assertTrue(set(hooks).issubset(set(mapping)))
    hf_targets = [k for v in mapping.values() for k in _hf_leaf_keys(v)]
    self.assertEqual(len(hf_targets), len(set(hf_targets)))
    self.assertEqual(set(hf_targets), set(self.hf_shapes))
    if scan_layers:
      self.assertIn("MoEBiasVar-decoder-layers_1-moe_block-gate-bias", flat_mt_keys)
      self.assertIn("MoEBiasVar-decoder-scanned_blocks-layers_0-moe_block-gate-bias", flat_mt_keys)

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_bidirectional_round_trip_fp32_and_bf16_export(self, scan_layers: bool):
    """Verifies exact fp32 bidirectional round-trip and fp32 preservation under bfloat16 export."""
    mt_cfg = self.cfg[scan_layers]
    converted = _convert_hf_to_maxtext(self.hf_state, self.hf_cfg, mt_cfg, self.shapes[scan_layers])
    for k, arr in converted.items():
      self.assertEqual(arr.shape, self.shapes[scan_layers][k], k)

    mapping = self.mapping[scan_layers]
    to_hf_hooks = HOOK_FNS[MODEL_NAME](self.hf_cfg, mt_cfg, scan_layers=scan_layers, saving_to_hf=True)

    # 1. Exact float32 MaxText -> HF round trip (`saving_to_hf=True`).
    exported_fp32 = {}
    for mt_key in mapping:
      weight = (
          [jnp.asarray(converted[k]) for k in mt_key] if isinstance(mt_key, tuple) else jnp.asarray(converted[mt_key])
      )
      exported_fp32.update(
          conversion_utils.process_maxtext_param(mt_key, weight, mapping, to_hf_hooks, self.hf_shapes, mt_cfg)
      )
    self.assertEqual(set(exported_fp32), set(self.hf_state))
    for hf_key, want in self.hf_state.items():
      np.testing.assert_allclose(exported_fp32[hf_key], want, atol=1e-6, rtol=1e-6, err_msg=hf_key)

    # 2. `weight_dtype=bfloat16` export downcasts expert/attention matrices in JAX without NumPy `TypeError`,
    # while preserving exact float32 for router weights/biases and residual gates.
    bf16_cfg = _toy_maxtext_config(scan_layers, weight_dtype="bfloat16")
    exported_bf16 = {}
    for mt_key in mapping:
      weight = (
          [jnp.asarray(converted[k]) for k in mt_key] if isinstance(mt_key, tuple) else jnp.asarray(converted[mt_key])
      )
      exported_bf16.update(
          conversion_utils.process_maxtext_param(mt_key, weight, mapping, to_hf_hooks, self.hf_shapes, bf16_cfg)
      )
    self.assertEqual(
        exported_bf16["model.language_model.layers.1.mlp.gate.weight"].dtype,
        np.float32,
    )
    self.assertEqual(
        exported_bf16["model.language_model.layers.1.mlp.gate.e_score_correction_bias"].dtype,
        np.float32,
    )
    self.assertEqual(
        exported_bf16["model.language_model.layers.1.post_attention_residual_gate.gate"].dtype,
        np.float32,
    )
    self.assertEqual(
        exported_bf16["model.language_model.layers.2.mlp.experts.gate_up_proj"].dtype,
        ml_dtypes.bfloat16,
    )

  def test_scanned_and_unscanned_converted_models_produce_matching_logits(self):
    """Verifies scanned and unscanned NNX models produce identical logits from converted HF weights."""
    conv_u = _convert_hf_to_maxtext(self.hf_state, self.hf_cfg, self.cfg[False], self.shapes[False])
    conv_s = _convert_hf_to_maxtext(self.hf_state, self.hf_cfg, self.cfg[True], self.shapes[True])
    tokens = jnp.asarray([[1, 5, 9, 13, 17, 21, 25, 29]], dtype=jnp.int32)
    positions = jnp.arange(_SEQ_LEN, dtype=jnp.int32)[None, :]
    seg_ids = jnp.ones_like(tokens)

    logits = {}
    for scan_layers, conv in ((False, conv_u), (True, conv_s)):
      cfg = self.cfg[scan_layers]
      with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        model = model_creation_utils.from_config(cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
        loaded = _load_converted_into_nnx(model, conv)
        self.assertEqual(loaded, len(self.shapes[scan_layers]))
        out = model(
            decoder_input_tokens=tokens,
            decoder_positions=positions,
            decoder_segment_ids=seg_ids,
            model_mode=MODEL_MODE_TRAIN,
            enable_dropout=False,
        )
      logits[scan_layers] = np.asarray(out, dtype=np.float32)
      self.assertTrue(np.all(np.isfinite(logits[scan_layers])))

    num_prefix, num_blocks = muse_spark.scan_layout(self.cfg[True])
    self.assertEqual((num_prefix, num_blocks), (2, 1))
    self.assertEqual(param_mapping._muse_spark_scan_layout(62, 4, 0, _EXPERTS, True), (2, 15))  # pylint: disable=protected-access
    self.assertEqual(param_mapping._muse_spark_scan_layout(6, 4, _FIRST_DENSE, _EXPERTS, True), (2, 1))  # pylint: disable=protected-access
    np.testing.assert_allclose(logits[True], logits[False], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
