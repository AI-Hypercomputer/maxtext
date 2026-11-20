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

"""Kimi-K3 checkpoint conversion for the scanned layout (`scan_layers=True`).

CPU-only, no HF reference checkout or torch needed. On a 13-layer toy (prefix 0..3 with the
dense layer 0, scanned cycles [KDA, KDA, KDA, MLA] at 4..7 and 8..11, suffix [12]):

  1. The scanned mapping's keys are exactly the scanned model's checkpoint keys (as
     `to_maxtext.get_maxtext_model_info` lays them out, incl. the `MoEBiasVar` collection),
     and its hooks only name mapped keys.
  2. HF tensors -> scanned MaxText through `to_maxtext`'s own loaders (eager and lazy)
     equals the unscanned conversion re-stacked on the scan axis, incl. the nested
     (expert, block) routed-expert stacks and the router bias.
  3. scanned MaxText -> HF (`process_maxtext_param`, the `to_huggingface` path) is lossless.
  4. The scanned model loaded from the scanned conversion produces the same logits as the
     unscanned model loaded from the unscanned conversion.
"""

import copy
import re
import sys
import unittest

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.checkpoint_conversion.utils.hf_model_configs import kimi_k3_dict
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from tests.utils.kimi_k3_conversion_utils import convert_hf_state_dict_via_to_maxtext, load_checkpoint_params_into_nnx
from tests.utils.test_helpers import get_test_config_path

_NUM_LAYERS = 13
_FULL_ATTN_LAYERS = (3, 7, 11, 12)  # 0-indexed
_NUM_EXPERTS = 4
_SEQ_LEN = 8
_LAYER_KEY = re.compile(r"^(params|MoEBiasVar)-decoder-(scanned_blocks-)?layers_(\d+)-(.+)$")


def _hf_config() -> dict:
  """The released HF text config shrunk to the toy layout (HF layer lists are 1-indexed)."""
  cfg = copy.deepcopy(kimi_k3_dict)
  cfg.update(
      vocab_size=128,
      hidden_size=64,
      intermediate_size=64,
      num_hidden_layers=_NUM_LAYERS,
      num_attention_heads=4,
      num_key_value_heads=4,
      head_dim=16,
      q_lora_rank=32,
      kv_lora_rank=16,
      qk_nope_head_dim=16,
      qk_rope_head_dim=8,
      v_head_dim=16,
      num_experts=_NUM_EXPERTS,
      num_experts_per_token=2,
      routed_expert_hidden_size=32,
      moe_intermediate_size=32,
      num_shared_experts=1,
      first_k_dense_replace=1,
  )
  cfg["linear_attn_config"] = {
      **cfg["linear_attn_config"],
      "num_heads": 4,
      "head_dim": 16,
      "short_conv_kernel_size": 4,
      "full_attn_layers": [i + 1 for i in _FULL_ATTN_LAYERS],
      "kda_layers": [i + 1 for i in range(_NUM_LAYERS) if i not in _FULL_ATTN_LAYERS],
  }
  return cfg


def _maxtext_config(scan_layers: bool, **overrides):
  """The matching toy MaxText config (fp32, CPU-friendly)."""
  kwargs = {
      "override_model_config": True,
      "base_emb_dim": 64,
      "base_num_decoder_layers": _NUM_LAYERS,
      "full_attn_layers": list(_FULL_ATTN_LAYERS),
      "first_num_dense_layers": 1,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 16,
      "vocab_size": 128,
      "q_lora_rank": 32,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 8,
      "v_head_dim": 16,
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 32,
      "moe_intermediate_size": 32,
      "shared_intermediate_size": 32,
      "shared_experts": 1,
      "num_shared_experts": 1,
      "routed_expert_hidden_size": 32,
      "num_experts": _NUM_EXPERTS,
      "num_experts_per_tok": 2,
      "moe_expert_input_dim": 32,
      "moe_router_input_dim": 64,
      "megablox": False,
      "attn_res_block_size": 5,  # commits at 0, 5, 10: inside the scanned cycles
      "max_target_length": _SEQ_LEN,
      "max_position_embeddings": _SEQ_LEN,
      "original_max_position_embeddings": _SEQ_LEN,
      "max_prefill_predict_length": _SEQ_LEN // 2,
      "per_device_batch_size": 1,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "packing": False,
      "enable_checkpointing": False,
      "scan_layers": scan_layers,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], model_name="kimi-k3", **kwargs)


def _hf_targets(hf_value) -> list[str]:
  if isinstance(hf_value, list):
    return [k for v in hf_value for k in _hf_targets(v)]
  return [hf_value]


def _random_hf_state_dict(hf_config: dict, hf_keys, seed: int = 0) -> dict[str, np.ndarray]:
  """Well-conditioned random HF weights: unit norm scales, small projections."""
  shapes = HF_SHAPE["kimi-k3"](hf_config)
  rng = np.random.default_rng(seed)
  state = {}
  for key in sorted(hf_keys):
    shape = tuple(shapes[key])
    if key.endswith(("norm.weight",)) and len(shape) == 1:
      value = 1.0 + 0.1 * rng.standard_normal(shape)
    elif len(shape) >= 2:
      value = rng.standard_normal(shape) * (0.5 / np.sqrt(shape[-1]))
    else:
      value = 0.1 * rng.standard_normal(shape)
    if key.endswith(".A_log"):
      # The release stores A_log as [head_dim]: num_heads log-decays, then zero padding.
      value[hf_config["linear_attn_config"]["num_heads"] :] = 0.0
    state[key] = value.astype(np.float32)
  return state


def _scan_axis(mt_key: str, ndim: int, config) -> int:
  """Where `nnx_scan.create_scanned_layers` puts the block axis of a scanned leaf."""
  if not mt_key.startswith("params-"):
    return 0  # non-Param collections (MoEBiasVar) stay stacked on axis 0
  return config.param_scan_axis if ndim > config.param_scan_axis else 0


class KimiK3ScannedConversionTest(parameterized.TestCase):
  """Scanned-layout `PARAM_MAPPING` / `HOOK_FNS` against the scanned model and the unscanned conversion."""

  @classmethod
  def setUpClass(cls):
    """Builds the HF config, both MaxText configs and the scan layout once for the class."""
    super().setUpClass()
    cls.hf_config = _hf_config()
    cls.cfg = {scan: _maxtext_config(scan) for scan in (False, True)}
    cls.layout = kimi_k3.kimi_scan_layout(cls.cfg[True])
    cls.model_info = {scan: dict(to_maxtext.get_maxtext_model_info(cls.cfg[scan])[0]) for scan in (False, True)}
    cls.shapes = {scan: {k: tuple(v[1]) for k, v in cls.model_info[scan].items()} for scan in (False, True)}
    cls.mapping = {
        scan: PARAM_MAPPING["kimi-k3"](cls.hf_config, cls.cfg[scan], scan_layers=scan) for scan in (False, True)
    }
    hf_keys = {k for v in cls.mapping[False].values() for k in _hf_targets(v)}
    cls.hf_state = _random_hf_state_dict(cls.hf_config, hf_keys)
    cls._converted = {}

  def _convert(self, scan_layers: bool, lazy: bool = False):
    if (scan_layers, lazy) not in self._converted:
      self._converted[(scan_layers, lazy)] = convert_hf_state_dict_via_to_maxtext(
          self.hf_state, self.hf_config, self.cfg[scan_layers], self.shapes[scan_layers], lazy=lazy
      )
    return self._converted[(scan_layers, lazy)]

  def test_layout(self):
    self.assertEqual(tuple(self.layout), (4, 2, 12, ((True, True),) * 3 + ((False, True),)))

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_mapping_keys_match_model(self, scan_layers):
    """Mapping keys == the model's checkpoint keys; hooks only for mapped keys; HF targets all exist."""
    mapping = self.mapping[scan_layers]
    hooks = HOOK_FNS["kimi-k3"](self.hf_config, self.cfg[scan_layers], scan_layers=scan_layers, saving_to_hf=False)
    model_keys = set(self.shapes[scan_layers])
    self.assertEqual(sorted(model_keys - set(mapping)), [], "model params missing from the mapping")
    self.assertEqual(sorted(set(mapping) - model_keys), [], "mapping keys not in the model")
    self.assertEqual(sorted(set(hooks) - set(mapping)), [], "hooks for keys not in the mapping")
    all_targets = [k for v in mapping.values() for k in _hf_targets(v)]
    self.assertEqual(len(all_targets), len(set(all_targets)), "an HF tensor is mapped twice")
    self.assertEqual(set(all_targets), set(self.hf_state), "scanned and unscanned mappings cover different HF tensors")
    if scan_layers:
      self.assertIn("MoEBiasVar-decoder-scanned_blocks-layers_3-mlp-routed_experts-gate-bias", mapping)
      self.assertIn("MoEBiasVar-decoder-layers_12-mlp-routed_experts-gate-bias", mapping)
      self.assertNotIn("params-decoder-layers_4-pre_self_attention_layer_norm-scale", mapping)

  @parameterized.named_parameters(("eager", False), ("lazy", True))
  def test_scanned_conversion_equals_restacked_unscanned(self, lazy):
    """HF -> scanned params (to_maxtext loaders) == HF -> unscanned params, re-stacked per scanned block."""
    unscanned = self._convert(False)
    scanned = self._convert(True, lazy=lazy)
    cfg = self.cfg[True]
    checked = set()
    for key, value in scanned.items():
      self.assertEqual(value.shape, self.shapes[True][key], key)
      match = _LAYER_KEY.match(key)
      if not match or not match.group(2):
        np.testing.assert_array_equal(value, unscanned[key], err_msg=key)
        checked.add(key)
        continue
      collection, _, sub_layer, rest = match.groups()
      axis = _scan_axis(key, value.ndim, cfg)
      self.assertEqual(value.shape[axis], self.layout.num_blocks, key)
      for block, layer in enumerate(self.layout.scanned_layer_indices(int(sub_layer))):
        ref_key = f"{collection}-decoder-layers_{layer}-{rest}"
        np.testing.assert_array_equal(np.take(value, block, axis=axis), unscanned[ref_key], err_msg=f"{key}[{block}]")
        checked.add(ref_key)
    self.assertEqual(checked, set(unscanned))

  def test_router_bias_and_experts_layout(self):
    """MoEBiasVar is (blocks, experts); routed experts are (experts, blocks, ...) with HF per-layer values."""
    scanned = self._convert(True)
    hf = "language_model.model.layers"
    bias = scanned["MoEBiasVar-decoder-scanned_blocks-layers_2-mlp-routed_experts-gate-bias"]
    self.assertEqual(bias.shape, (self.layout.num_blocks, _NUM_EXPERTS))
    for block, layer in enumerate(self.layout.scanned_layer_indices(2)):
      np.testing.assert_array_equal(
          bias[block], self.hf_state[f"{hf}.{layer}.block_sparse_moe.gate.e_score_correction_bias"]
      )
    wo = scanned["params-decoder-scanned_blocks-layers_3-mlp-routed_experts-wo"]
    self.assertEqual(wo.shape[:2], (_NUM_EXPERTS, self.layout.num_blocks))
    layer = self.layout.scanned_layer_indices(3)[1]
    np.testing.assert_array_equal(wo[2, 1], self.hf_state[f"{hf}.{layer}.block_sparse_moe.experts.2.w2.weight"].T)

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_export_round_trip(self, scan_layers):
    """MaxText -> HF via `process_maxtext_param` (the to_huggingface path) restores every HF tensor exactly."""
    converted = self._convert(scan_layers)
    mapping = self.mapping[scan_layers]
    to_hf_hooks = HOOK_FNS["kimi-k3"](self.hf_config, self.cfg[scan_layers], scan_layers=scan_layers, saving_to_hf=True)
    hf_shapes = HF_SHAPE["kimi-k3"](self.hf_config)
    exported = {}
    for mt_key, hf_value in mapping.items():
      del hf_value
      exported.update(
          conversion_utils.process_maxtext_param(
              mt_key, jnp.asarray(converted[mt_key]), mapping, to_hf_hooks, hf_shapes, self.cfg[scan_layers]
          )
      )
    self.assertEqual(sorted(exported), sorted(self.hf_state))
    for hf_key, want in self.hf_state.items():
      np.testing.assert_array_equal(np.asarray(exported[hf_key], np.float32), want, err_msg=hf_key)

  def test_scanned_model_logits_match_unscanned(self):
    """Scanned model on the scanned conversion == unscanned model on the unscanned conversion."""
    tokens = (jnp.arange(_SEQ_LEN, dtype=jnp.int32)[None] * 5 % 97 + 1).astype(jnp.int32)
    positions = jnp.arange(_SEQ_LEN, dtype=jnp.int32)[None]
    logits = {}
    for scan_layers in (False, True):
      model = model_creation_utils.from_config(
          self.cfg[scan_layers], devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
      )
      loaded = load_checkpoint_params_into_nnx(model, self._convert(scan_layers))
      self.assertLen(loaded, len(self.shapes[scan_layers]))
      self.assertEqual(hasattr(model.decoder, "scanned_blocks"), scan_layers)
      logits[scan_layers] = np.asarray(model(tokens, positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False))
      self.assertTrue(np.isfinite(logits[scan_layers]).all())
    np.testing.assert_allclose(logits[True], logits[False], rtol=2e-5, atol=2e-5)

  def test_scan_layout_mismatch_is_rejected(self):
    """The mapping refuses a MaxText config whose layer schedule differs from the HF config's."""
    other = _maxtext_config(True, full_attn_layers=[3, 7, 11], base_num_decoder_layers=12)
    with self.assertRaisesRegex(ValueError, "does not match"):
      PARAM_MAPPING["kimi-k3"](self.hf_config, other, scan_layers=True)


if __name__ == "__main__":
  unittest.main()
