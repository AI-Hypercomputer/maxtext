# Copyright 2026 Google LLC
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

"""Parity and verification tests for GLM-5.3-Flash checkpoint masked-prefill and debug-stats."""

# pylint: disable=protected-access

from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import inspect
import json
import logging
import math
import os
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from tests.utils import glm5_test_utils
from tests.utils import zero_gen_logits_checker
import numpy as np
import pytest

try:
  from maxtext.m3.models.glm5_3_flash import create_mini_slice
  from maxtext.m3.models.glm5_3_flash import glm5_ckpt
  import safetensors.torch
  import torch
  import transformers
  from transformers.integrations import moe as hf_moe_integ
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model
  from transformers.quantizers import quantizer_finegrained_fp8

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]

_RUN_REAL_SLICE = flags.DEFINE_bool(
    "run_real_slice",
    False,
    "If true, runs the non-hermetic real-weight mini-slice parity suite.",
)
_GENERATE_GOLDEN = flags.DEFINE_bool(
    "generate_golden",
    False,
    "If true, generates HF golden reference files and exits.",
)
_WORK_DIR = flags.DEFINE_string(
    "work_dir",
    None,
    "Required directory containing the staged 7-layer mini-slice files.",
)
_GOLDEN_PATH = flags.DEFINE_string(
    "golden_path",
    None,
    "Required path to explicit HF golden artifact directory.",
)
_REAL_SLICE_SEQ_LEN = flags.DEFINE_string(
    "real_slice_seq_len",
    "all",
    "Sequence length to run for real mini-slice prefill parity tests. " "Must be one of 'all', '512', or '4096'.",
)

_HF_MLA_LAYER_TYPES = (
    "full_attention",
    "deepseek_sparse_attention",
    "indexed_attention",
)

# Verified against zai-org/GLM-5.3-Flash commit 46a050bec13002ad29ee795c16e61cf4fa3c4a9c.
_GLM5_PINNED_COMMIT_SHA = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
_EXPECTED_SHARD_SHA256_PAIRS = (
    (1, "9ff3c9397e75fc8883f24d8474960f53e993b0c0e4a8a4687e81a1cbec39965b"),
    (2, "328225cabdc9b1595bc8bd0062cdc99fb23872eb5bbf7ec5db62f2be8abeaeeb"),
    (3, "e0fc42c270a8f8a2c1a38c91b06c28ca0646721f8d4768fa09ed97faca918300"),
    (17, "530a4034da64dc4a94aae6ef2bafd40d4b498b324f6143a1fc60d548c356e407"),
    (31, "e2f31cb37dcf8ca7000e82990f3af061a847f1086f79f207c7223fa090f04502"),
    (32, "f33b53853ea5576b5aabd259bd9d933e6dbbaf65c511a8cdb328caab4349d8a9"),
    (46, "9e748de0fd06f845a5d70f623ee720f55c1ce1954ef8a33331738d9e244493dc"),
    (47, "ddf0691ca8aa43ecdcc6cc988946d43c82ad9a673bcb3b933223bf5b33be0ae1"),
    (54, "1ae7351f071a6a99e4b7eff39eb598e1ab02c086a7863c37c9f69e3dcde5b3d1"),
    (55, "8d64c91ff600a93311c15cde18b62b9b3de239727295ad6bc4b5f6d6654cef8e"),
    (56, "748dda91c9b697e596ebd7ef56244450c9f17c7de2b463d9de1d28b2572f3f51"),
    (57, "e128c2da3d3e9d100662504703d545ada43783bb4c6f2870bd3a17169b0b8141"),
    (62, "d3087816db95f962a3a74c057b8398e1492458ecd99608f706a400f112a825c6"),
)
_EXPECTED_SHARD_SHA256S: dict[str, str] = {
    f"model-{idx:05d}-of-00062.safetensors": sha for idx, sha in _EXPECTED_SHARD_SHA256_PAIRS
}


def _verify_shard_sha256(shard_name: str, computed_sha: str) -> None:
  """Verifies shard SHA256 matches expected authoritative HF digest."""
  if shard_name not in _EXPECTED_SHARD_SHA256S:
    raise ValueError(f"Unknown shard name {shard_name!r}; not in _EXPECTED_SHARD_SHA256S.")
  expected_sha = _EXPECTED_SHARD_SHA256S[shard_name]
  if computed_sha != expected_sha:
    raise ValueError(f"Shard {shard_name!r} SHA256 mismatch: expected {expected_sha}, got" f" {computed_sha}.")


def _compute_inputs_sha256(
    tokens_512: np.ndarray,
    seg_ids_512: np.ndarray,
    tokens_4096: np.ndarray,
    seg_ids_4096: np.ndarray,
) -> str:
  """Computes combined SHA256 of token and segment ID arrays."""
  hasher = hashlib.sha256()
  hasher.update(np.ascontiguousarray(tokens_512).tobytes())
  hasher.update(np.ascontiguousarray(seg_ids_512).tobytes())
  hasher.update(np.ascontiguousarray(tokens_4096).tobytes())
  hasher.update(np.ascontiguousarray(seg_ids_4096).tobytes())
  return hasher.hexdigest()


def _validate_golden_path_isolation(golden_path: str, raw_work_dir: str, slice_dir: str) -> None:
  """Validates that golden_path does not equal or reside within raw_work_dir or slice_dir."""
  if not golden_path or not golden_path.strip():
    raise ValueError("--golden_path must be specified.")
  real_golden = os.path.realpath(golden_path).rstrip(os.sep)
  real_raw = os.path.realpath(raw_work_dir).rstrip(os.sep)
  real_slice = os.path.realpath(slice_dir).rstrip(os.sep)
  if real_golden == real_raw or real_golden.startswith(real_raw + os.sep):
    raise ValueError(
        f"Golden path {golden_path!r} (realpath: {real_golden!r}) must not"
        f" equal or reside within raw_work_dir {raw_work_dir!r} (realpath:"
        f" {real_raw!r})."
    )
  if real_golden == real_slice or real_golden.startswith(real_slice + os.sep):
    raise ValueError(
        f"Golden path {golden_path!r} (realpath: {real_golden!r}) must not"
        f" equal or reside within slice_dir {slice_dir!r} (realpath:"
        f" {real_slice!r})."
    )


def _atomic_save_npy(target_path: str, arr: np.ndarray) -> None:
  """Saves numpy array atomically to target_path using PID-tagged temporary file."""
  target = epath.Path(target_path)
  tmp_path = target.parent / f"{target.name}.tmp.{os.getpid()}"
  with tmp_path.open("wb") as f:
    np.save(f, arr)
  tmp_path.replace(target)


def _atomic_save_json(target_path: str, data: dict[str, Any]) -> None:
  """Saves json dict atomically to target_path using PID-tagged temporary file."""
  target = epath.Path(target_path)
  tmp_path = target.parent / f"{target.name}.tmp.{os.getpid()}"
  with tmp_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
  tmp_path.replace(target)


def _validate_golden_manifest(
    golden_path: str,
    *,
    expected_num_layers: int,
    expected_slice_sha256: str,
    expected_seq_lens: list[int],
    expected_inputs_sha256: str,
    expected_hf_loader_code_sha256: str,
    num_moe: int,
    num_full: int,
) -> dict[str, Any]:
  """Validates golden manifest and verifies all required arrays exist and match metadata."""
  if not golden_path or not golden_path.strip():
    raise ValueError("golden_path must be specified.")
  manifest_path = epath.Path(golden_path) / "manifest.json"
  if not manifest_path.exists():
    raise FileNotFoundError(f"Golden manifest not found at {manifest_path!s}.")
  with manifest_path.open("r", encoding="utf-8") as f:
    manifest_data = json.load(f)

  required_keys = (
      "schema_version",
      "num_hidden_layers",
      "slice_sha256",
      "seq_lens",
      "inputs_sha256",
      "hf_loader_code_sha256",
      "torch_numerics",
      "arrays",
  )
  for k in required_keys:
    if k not in manifest_data:
      raise ValueError(f"Golden manifest at {manifest_path!r} missing required key: {k!r}")

  if manifest_data["schema_version"] != 2:
    raise ValueError("Unsupported golden manifest schema_version:" f" {manifest_data['schema_version']!r}; expected 2.")
  if manifest_data["num_hidden_layers"] != expected_num_layers:
    raise ValueError(
        "Golden manifest num_hidden_layers mismatch: expected"
        f" {expected_num_layers}, got {manifest_data['num_hidden_layers']}."
    )
  if manifest_data["slice_sha256"] != expected_slice_sha256:
    raise ValueError(
        "Golden manifest slice_sha256 mismatch: expected"
        f" {expected_slice_sha256!r}, got {manifest_data['slice_sha256']!r}."
    )
  if manifest_data["seq_lens"] != expected_seq_lens:
    raise ValueError(
        f"Golden manifest seq_lens mismatch: expected {expected_seq_lens!r}, " f"got {manifest_data['seq_lens']!r}."
    )
  if manifest_data["inputs_sha256"] != expected_inputs_sha256:
    raise ValueError(
        "Golden manifest inputs_sha256 mismatch: expected"
        f" {expected_inputs_sha256!r}, got {manifest_data['inputs_sha256']!r}."
    )
  if manifest_data["hf_loader_code_sha256"] != expected_hf_loader_code_sha256:
    raise ValueError(
        "Golden manifest hf_loader_code_sha256 mismatch: expected"
        f" {expected_hf_loader_code_sha256!r}, got"
        f" {manifest_data['hf_loader_code_sha256']!r}."
    )

  arrays_meta = manifest_data["arrays"]
  expected_files = []
  for sl in expected_seq_lens:
    expected_files.append(f"hf_logits_{sl}.npy")
    for i in range(num_moe):
      expected_files.append(f"hf_exp_{sl}_{i}.npy")
      expected_files.append(f"hf_rlog_{sl}_{i}.npy")
    for i in range(num_full):
      expected_files.append(f"hf_dsa_{sl}_{i}.npy")

  for fname in expected_files:
    if fname not in arrays_meta:
      raise ValueError(f"Required array {fname!r} missing from manifest arrays metadata.")
    meta = arrays_meta[fname]
    fpath = epath.Path(golden_path) / fname
    if not fpath.exists():
      raise FileNotFoundError(f"Required golden array missing at {fpath!s}")
    with fpath.open("rb") as f:
      arr = np.load(f)
    if list(arr.shape) != meta["shape"]:
      raise ValueError(f"Array shape mismatch for {fname}: expected {meta['shape']}, got {list(arr.shape)}")
    if str(arr.dtype) != meta["dtype"]:
      raise ValueError(f"Array dtype mismatch for {fname}: expected {meta['dtype']}, got {str(arr.dtype)}")
    actual_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    if actual_sha != meta["sha256"]:
      raise ValueError(f"Array sha256 mismatch for {fname}: expected {meta['sha256']}, got {actual_sha}")

  return manifest_data


def _verify_hf_safetensors_key_bijection(
    safetensors_keys: set[str],
    consumed_keys: set[str],
    num_layers: int,
) -> None:
  """Verifies that every language-model weight and scale key in safetensors for layers 0..num_layers-1 was consumed."""
  expected_keys: set[str] = set()
  for k in safetensors_keys:
    norm_k = k
    for prefix in ("model.language_model.", "model."):
      if norm_k.startswith(prefix):
        norm_k = norm_k[len(prefix) :]
        break
    if norm_k in (
        "embed_tokens.weight",
        "embed_tokens.weight_scale_inv",
        "norm.weight",
        "norm.weight_scale_inv",
    ) or k in ("lm_head.weight", "lm_head.weight_scale_inv"):
      expected_keys.add(k)
    elif norm_k.startswith("layers."):
      parts = norm_k.split(".")
      if len(parts) > 1 and parts[1].isdigit():
        l_idx = int(parts[1])
        if l_idx < num_layers:
          expected_keys.add(k)

  unconsumed = expected_keys - consumed_keys
  unexpected = consumed_keys - expected_keys
  if unconsumed or unexpected:
    raise ValueError(
        "Safetensors key bijection check failed:" f" unconsumed={sorted(unconsumed)}, unexpected={sorted(unexpected)}"
    )


_make_cfg = glm5_test_utils._make_cfg
_init_hf = glm5_test_utils._init_hf
_hf_forward = glm5_test_utils._hf_forward
_fp64_mode = glm5_test_utils._fp64_mode
_assert_e3 = glm5_test_utils._assert_e3
_hf_module_state_to_hub = glm5_test_utils._hf_module_state_to_hub
_LazySafetensorsMapping = glm5_test_utils._LazySafetensorsMapping
ensure_real_mini_slice = glm5_test_utils.ensure_real_mini_slice


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5CkptParityTest(parameterized.TestCase):
  """Comprehensive test suite for GLM-5.3-Flash checkpoint masked-prefill and debug-stats."""

  def setUp(self):
    super().setUp()
    if _GENERATE_GOLDEN.value:
      raise RuntimeError("Test methods must not run under --generate_golden=true.")
    if _RUN_REAL_SLICE.value:
      self.skipTest("Skipping synthetic test when --run_real_slice=true.")
    self.enter_context(torch.enable_grad())
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.manual_seed(42)
    np.random.seed(42)

  def tearDown(self):
    jax.config.update("jax_default_matmul_precision", "default")
    super().tearDown()

  def _assert_e3(self, a: Any, b: Any, name: str = "") -> None:
    """Asserts parameter gradients satisfy E3 magnitude-scaled atol, relative frob, and cosine."""
    glm5_test_utils.assert_e3(a, b, name=name)

  def _prepare_model_and_hf_state(
      self,
      num_layers: int,
      scan_layers: bool,
      enable_debug_stats: bool = False,
      dtype: str = "float32",
      weight_dtype: str = "float32",
  ) -> tuple[
      modeling_glm5_3_flash.Glm5Model,
      hf_model.Glm5NextTextModel,
      torch.nn.Linear,
      dict[str, Any],
  ]:
    """Initializes and loads matching weights between JAX and HuggingFace models in Hub format."""
    c, hf_c = _make_cfg(
        num_hidden_layers=num_layers,
        scan_layers=scan_layers,
        dtype=dtype,
        weight_dtype=weight_dtype,
        enable_debug_stats=enable_debug_stats,
    )
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch.float32 if dtype == "float32" else torch.float64)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(
        torch.float32 if dtype == "float32" else torch.float64
    )
    _init_hf(hf_m, hf_head)
    hf_state = dict(hf_m.state_dict(), **{"lm_head.weight": hf_head.weight.detach().clone()})

    for l_idx in range(c.first_k_dense_replace, num_layers):
      b_val = torch.linspace(
          -0.05,
          0.05,
          c.n_routed_experts,
          dtype=torch.float32 if dtype == "float32" else torch.float64,
      )
      hf_m.layers[l_idx].mlp.gate.e_score_correction_bias.copy_(b_val)
      hf_state[f"layers.{l_idx}.mlp.gate.e_score_correction_bias"] = b_val

    hub_state = _hf_module_state_to_hub(
        hf_state,
        num_experts=c.n_routed_experts,
        moe_intermediate_size=c.moe_intermediate_size,
    )
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_m, hub_state, ignore_mtp_layer=True)
    return jax_m, hf_m, hf_head, hub_state

  @parameterized.named_parameters(
      ("seq512_5layers_unscanned_left", 512, 5, False, "left"),
      ("seq512_5layers_unscanned_right", 512, 5, False, "right"),
      ("seq512_13layers_scanned_left", 512, 13, True, "left"),
      ("seq512_13layers_scanned_right", 512, 13, True, "right"),
      ("seq4096_5layers_unscanned_left", 4096, 5, False, "left"),
      ("seq4096_13layers_scanned_left", 4096, 13, True, "left"),
  )
  def test_masked_prefill_parity_and_tier_1ab(
      self,
      seq_len: int,
      num_layers: int,
      scan_layers: bool,
      padding_side: str = "left",
  ):
    """Verifies Tier 1A routing index and Tier 1B masked-token logit parity at 512 and 4096 tokens."""
    jax_m, hf_m, hf_head, _ = self._prepare_model_and_hf_state(num_layers, scan_layers, enable_debug_stats=True)

    texts = [
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["prose"],
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["code_json_math"],
    ]
    tokens_np, seg_ids_np, mask_np = zero_gen_logits_checker.synthetic_golden_batch(
        texts,
        vocab_size=jax_m.config.vocab_size,
        seq_len=seq_len,
        pad_token_id=0,
        batch_size=2,
        target_start_fraction=0.5,
        valid_lengths=[seq_len, seq_len - 16],
        padding_side=padding_side,
    )
    tokens_jax = jnp.array(tokens_np)
    seg_ids_jax = jnp.array(seg_ids_np)
    ids_torch = torch.from_numpy(tokens_np).long()
    attn_mask_torch = torch.from_numpy(seg_ids_np > 0).bool()

    hf_expert_indices: list[np.ndarray] = []
    hf_dsa_indices: list[np.ndarray] = []

    def _make_gate_hook():
      def _hook(module, inputs, output):
        del module, inputs
        hf_expert_indices.append(output[2].detach().cpu().numpy())

      return _hook

    def _make_dsa_hook():
      def _dsa_hook(module, inputs, output):
        del module, inputs
        hf_dsa_indices.append(output.detach().cpu().numpy())

      return _dsa_hook

    gate_hooks = [
        hf_m.layers[i].mlp.gate.register_forward_hook(_make_gate_hook())
        for i, lt in enumerate(hf_m.config.mlp_layer_types[:num_layers])
        if lt == "sparse"
    ]
    dsa_hooks = [
        hf_m.layers[i].self_attn.indexer.register_forward_hook(_make_dsa_hook())
        for i, lt in enumerate(hf_m.config.layer_types[:num_layers])
        if lt in _HF_MLA_LAYER_TYPES
    ]

    with torch.no_grad():
      try:
        hf_logits = _hf_forward(hf_m, hf_head, ids_torch, attention_mask=attn_mask_torch)
      finally:
        for h in gate_hooks + dsa_hooks:
          h.remove()

    jax_out = jax_m.forward_with_aux(tokens_jax, decoder_segment_ids=seg_ids_jax)

    num_moe = max(0, num_layers - jax_m.config.first_k_dense_replace)
    self.assertLen(hf_expert_indices, num_moe)

    # Read live router_topk_indices and dsa_topk_indices directly from model aux debug_stats per Row 84
    jax_expert_indices = list(jax_out.aux["debug_stats"]["router_topk_indices"])
    jax_dsa_indices = list(jax_out.aux["debug_stats"]["dsa_topk_indices"])

    # Single-row runs for each row in batch per Row 81
    single_logits = [
        jax_m(tokens_jax[b : b + 1], decoder_segment_ids=seg_ids_jax[b : b + 1]) for b in range(tokens_jax.shape[0])
    ]

    ref_max = max(1.0, float(np.max(np.abs(hf_logits.detach().cpu().numpy()))))
    eval_result = zero_gen_logits_checker.evaluate_masked_prefill_parity(
        jax_logits=jax_out.logits,
        ref_logits=hf_logits,
        target_mask=mask_np,
        jax_expert_indices=jax_expert_indices,
        ref_expert_indices=hf_expert_indices,
        jax_dsa_indices=jax_dsa_indices,
        ref_dsa_indices=hf_dsa_indices,
        single_row_logits=single_logits,
        decoder_segment_ids=seg_ids_np,
        index_topk=jax_m.config.index_topk,
        index_kpool=jax_m.config.index_kpool,
        cos_sim_threshold=1.0 - 1e-6,
        l_inf_threshold=1e-5 * ref_max,
        batch_diff_threshold=1e-5 * ref_max,
    )

    self.assertTrue(
        eval_result["tier_1a_passed"],
        msg=f"Tier 1A failed at seq_len={seq_len}: {eval_result}",
    )
    self.assertEqual(eval_result["expert_index_match_rate"], 1.0)
    self.assertGreater(eval_result["dsa_eval_token_count"], 0)
    self.assertEqual(eval_result["dsa_index_match_rate"], 1.0)

    self.assertTrue(
        eval_result["tier_1b_passed"],
        msg=f"Tier 1B failed at seq_len={seq_len}: {eval_result}",
    )
    self.assertGreater(eval_result["cos_sim"], 1.0 - 1e-6)
    self.assertLess(eval_result["l_inf"], 1e-5 * ref_max)
    self.assertLess(eval_result["max_kl_div"], 1e-3)
    self.assertGreater(eval_result["logit_std"], 1e-4)
    self.assertEqual(eval_result["zero_norm_token_count"], 0)
    self.assertTrue(eval_result["all_passed"])
    self._assert_e3(
        np.asarray(jax_out.logits)[seg_ids_np > 0],
        hf_logits.detach().cpu().numpy()[seg_ids_np > 0],
        "hermetic_prefill_logits",
    )

  @parameterized.parameters(
      (5, False, "right"),
      (5, False, "left"),
      (13, True, "right"),
      (13, True, "left"),
  )
  def test_batch_determinism(self, num_layers: int, scan_layers: bool, padding_side: str):
    """Verifies single-sample vs batched forward pass agreement under left/right-padding."""
    jax_m, _, _, _ = self._prepare_model_and_hf_state(num_layers, scan_layers)
    seq_len = 64
    texts = [
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["prose"],
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["code_json_math"],
    ]
    tokens_np, seg_ids_np, _ = zero_gen_logits_checker.synthetic_golden_batch(
        texts,
        vocab_size=jax_m.config.vocab_size,
        seq_len=seq_len,
        pad_token_id=0,
        batch_size=2,
        target_start_fraction=0.5,
        valid_lengths=[seq_len, seq_len - 12],
        padding_side=padding_side,
    )
    tokens_jax = jnp.array(tokens_np)
    seg_ids_jax = jnp.array(seg_ids_np)

    batched_logits = jax_m(tokens_jax, decoder_segment_ids=seg_ids_jax)
    for b in range(tokens_jax.shape[0]):
      single_logits = jax_m(tokens_jax[b : b + 1], decoder_segment_ids=seg_ids_jax[b : b + 1])
      valid_b = seg_ids_np[b] > 0
      diff = float(np.mean(np.abs(np.array(batched_logits[b][valid_b]) - np.array(single_logits[0][valid_b]))))
      self.assertLessEqual(
          diff,
          1e-4,
          msg=(f"Batch determinism mismatch on row {b} with" f" padding_side={padding_side}: diff={diff}"),
      )

  @parameterized.parameters((5, False), (13, True))
  def test_debug_stats_verification(self, num_layers: int, scan_layers: bool):
    """Verifies Appendix B first/last-block stats match HF, and omit debug_stats when disabled."""
    jax_m, hf_m, _, _ = self._prepare_model_and_hf_state(num_layers, scan_layers, enable_debug_stats=False)
    seq_len = 32
    tokens_np, seg_ids_np, _ = zero_gen_logits_checker.synthetic_golden_batch(
        zero_gen_logits_checker.GOLDEN_BATCH_TEXT,
        vocab_size=jax_m.config.vocab_size,
        seq_len=seq_len,
        pad_token_id=0,
        batch_size=1,
        target_start_fraction=0.5,
    )
    tokens_jax = jnp.array(tokens_np)
    seg_ids_jax = jnp.array(seg_ids_np)
    ids_torch = torch.from_numpy(tokens_np).long()

    out_no_debug = jax_m.forward_with_aux(tokens_jax, decoder_segment_ids=seg_ids_jax)
    self.assertNotIn("debug_stats", out_no_debug.aux)

    jax_m.config = dataclasses.replace(jax_m.config, enable_debug_stats=True)
    out_with_debug = jax_m.forward_with_aux(tokens_jax, decoder_segment_ids=seg_ids_jax)
    self.assertIn("debug_stats", out_with_debug.aux)
    stats = out_with_debug.aux["debug_stats"]

    hf_layer_outputs: dict[int, torch.Tensor] = {}

    def _make_layer_hook(idx: int):
      def _hook(module, inputs, output):
        del module, inputs
        hf_layer_outputs[idx] = output[0].detach()

      return _hook

    hooks = [
        hf_m.layers[0].register_forward_hook(_make_layer_hook(0)),
        hf_m.layers[num_layers - 1].register_forward_hook(_make_layer_hook(num_layers - 1)),
    ]
    try:
      orig = hf_model.chunk_kimi_delta_attention
      hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
      try:
        with torch.no_grad():
          _ = hf_m(input_ids=ids_torch)
      finally:
        hf_model.chunk_kimi_delta_attention = orig
    finally:
      for h in hooks:
        h.remove()

    hf_l0 = hf_layer_outputs[0].float().cpu().numpy()
    hf_llast = hf_layer_outputs[num_layers - 1].float().cpu().numpy()

    exp_l0_mean, exp_l0_var = float(np.mean(hf_l0)), float(np.var(hf_l0))
    exp_ll_mean, exp_ll_var = float(np.mean(hf_llast)), float(np.var(hf_llast))

    np.testing.assert_allclose(float(stats["first_block_mean"]), exp_l0_mean, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(float(stats["first_block_var"]), exp_l0_var, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(float(stats["last_block_mean"]), exp_ll_mean, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(float(stats["last_block_var"]), exp_ll_var, atol=1e-5, rtol=1e-5)
    self.assertIn("hidden_norm_mean", stats)
    self.assertIn("logits_max", stats)
    self.assertIn("logits_min", stats)
    self.assertNotIn("layer_0_mean", stats)
    self.assertNotIn("layer_0_var", stats)
    self.assertNotIn(f"layer_{num_layers - 1}_mean", stats)
    self.assertNotIn(f"layer_{num_layers - 1}_var", stats)
    expected_num_dsa = sum(1 for lt in jax_m.config.layer_types[:num_layers] if lt == "full_attention")
    expected_num_moe = max(0, num_layers - jax_m.config.first_k_dense_replace)
    self.assertLen(stats["dsa_topk_indices"], expected_num_dsa)
    self.assertLen(stats["router_topk_indices"], expected_num_moe)

  def _run_ce_and_grad_parity(self, num_layers: int, scan_layers: bool, is_fp64: bool) -> None:
    """Executes CE loss and parameter gradient parity against HuggingFace for fp32 or fp64."""
    dtype_str = "float64" if is_fp64 else "float32"
    jax_m, hf_m, hf_head, _ = self._prepare_model_and_hf_state(
        num_layers, scan_layers, dtype=dtype_str, weight_dtype=dtype_str
    )
    seq_len = 32
    texts = [
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["prose"],
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["code_json_math"],
    ]
    tokens_np, seg_ids_np, mask_np = zero_gen_logits_checker.synthetic_golden_batch(
        texts,
        vocab_size=jax_m.config.vocab_size,
        seq_len=seq_len,
        pad_token_id=0,
        batch_size=2,
        target_start_fraction=0.5,
        valid_lengths=[seq_len, seq_len - 8],
        padding_side="left",
    )
    tokens_jax = jnp.array(tokens_np)
    seg_ids_jax = jnp.array(seg_ids_np)
    loss_mask_np = mask_np[:, 1:] & (seg_ids_np[:, 1:] > 0)
    loss_mask_jax = jnp.array(loss_mask_np)

    def _jax_ce(m):
      logits = m(tokens_jax, decoder_segment_ids=seg_ids_jax)
      pred_logits = logits[:, :-1, :]
      target_tokens = tokens_jax[:, 1:]
      log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
      target_lp = jnp.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)
      return -jnp.sum(target_lp * loss_mask_jax) / jnp.sum(loss_mask_jax)

    jax_loss, jax_grads = nnx.value_and_grad(_jax_ce)(jax_m)

    ids_torch = torch.from_numpy(tokens_np).long()
    attn_mask_torch = torch.from_numpy(seg_ids_np > 0).bool()
    hf_logits = _hf_forward(hf_m, hf_head, ids_torch, attention_mask=attn_mask_torch)
    hf_pred = hf_logits[:, :-1, :].reshape(-1, jax_m.config.vocab_size)
    hf_tgt = ids_torch[:, 1:].reshape(-1)
    hf_ce_all = torch.nn.functional.cross_entropy(hf_pred, hf_tgt, reduction="none").reshape(ids_torch.shape[0], -1)
    hf_mask_t = torch.from_numpy(loss_mask_np)
    hf_loss = (hf_ce_all * hf_mask_t).sum() / hf_mask_t.sum()
    hf_loss.backward()

    hf_loss_val = hf_loss.detach().cpu().numpy()
    if is_fp64:
      l_frob = float(np.abs(float(jax_loss) - float(hf_loss_val))) / float(np.abs(float(hf_loss_val)))
      self.assertLessEqual(l_frob, 1e-10, f"fp64 CE loss frob_rel: {l_frob}")
    else:
      np.testing.assert_allclose(float(jax_loss), hf_loss_val, atol=1e-5, rtol=1e-5)

    hf_grad_state: dict[str, torch.Tensor] = {}
    for n, p in hf_m.named_parameters():
      hf_grad_state[n] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
    for n, b in hf_m.named_buffers():
      hf_grad_state[n] = torch.zeros_like(b)
    hf_grad_state["lm_head.weight"] = (
        hf_head.weight.grad.detach().clone() if hf_head.weight.grad is not None else torch.zeros_like(hf_head.weight)
    )

    hub_grad_state = _hf_module_state_to_hub(
        hf_grad_state,
        num_experts=jax_m.config.n_routed_experts,
        moe_intermediate_size=jax_m.config.moe_intermediate_size,
    )
    jax_grad_m = modeling_glm5_3_flash.Glm5Model(jax_m.config, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_grad_m, hub_grad_state, ignore_mtp_layer=True)

    leaves_jax = jax.tree.leaves(nnx.state(jax_grads, nnx.Param))
    leaves_hf = jax.tree.leaves(nnx.state(jax_grad_m, nnx.Param))
    self.assertEqual(len(leaves_jax), len(leaves_hf))
    for gj, gh in zip(leaves_jax, leaves_hf, strict=True):
      if is_fp64:
        aj = np.asarray(gj, dtype=np.float64)
        ah = np.asarray(gh, dtype=np.float64)
        norm_h = float(np.linalg.norm(ah))
        if norm_h == 0.0:
          self.assertEqual(float(np.linalg.norm(aj)), 0.0)
        else:
          self.assertGreater(float(np.linalg.norm(aj)), 0.0)
          diff = float(np.linalg.norm(aj - ah)) / norm_h
          self.assertLessEqual(diff, 1e-10)
      else:
        self._assert_e3(gj, gh, "golden_batch_param_grad")

    if not is_fp64:
      self.assertEmpty(dict(nnx.state(jax_grads, modeling_glm5_3_flash.Glm5RouterBias)))

  @parameterized.parameters((5, False), (13, True))
  def test_golden_batch_ce_loss_and_all_param_gradient_parity(self, num_layers: int, scan_layers: bool):
    """Verifies golden-batch masked CE loss and all-param gradients match HF under fp32 and fp64."""
    self._run_ce_and_grad_parity(num_layers, scan_layers, is_fp64=False)
    with _fp64_mode():
      self._run_ce_and_grad_parity(num_layers, scan_layers, is_fp64=True)

  def test_tokenize_golden_batch_with_real_or_callable_tokenizer(self):
    """Verifies tokenize_golden_batch with callable/object tokenizers and error paths."""
    with self.assertRaisesRegex(ValueError, "tokenizer must not be None"):
      zero_gen_logits_checker.tokenize_golden_batch(
          "test",
          tokenizer=None,
          seq_len=16,
          pad_token_id=0,
          batch_size=1,
          target_start_fraction=0.5,
      )

    def _dummy_tok(s: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
      del add_special_tokens
      return {"input_ids": [ord(c) % 50 + 1 for c in s]}

    with self.assertRaisesRegex(ValueError, "seq_len must be positive"):
      zero_gen_logits_checker.tokenize_golden_batch(
          "test",
          tokenizer=_dummy_tok,
          seq_len=0,
          pad_token_id=0,
          batch_size=1,
          target_start_fraction=0.5,
      )
    with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
      zero_gen_logits_checker.tokenize_golden_batch(
          "test",
          tokenizer=_dummy_tok,
          seq_len=16,
          pad_token_id=0,
          batch_size=0,
          target_start_fraction=0.5,
      )
    with self.assertRaisesRegex(ValueError, "pad_token_id must be non-negative"):
      zero_gen_logits_checker.tokenize_golden_batch(
          "test",
          tokenizer=_dummy_tok,
          seq_len=16,
          pad_token_id=-1,
          batch_size=1,
          target_start_fraction=0.5,
      )
    with self.assertRaisesRegex(ValueError, "target_start_fraction must be in"):
      zero_gen_logits_checker.tokenize_golden_batch(
          "test",
          tokenizer=_dummy_tok,
          seq_len=16,
          pad_token_id=0,
          batch_size=1,
          target_start_fraction=1.5,
      )

    tokens, seg_ids, mask = zero_gen_logits_checker.tokenize_golden_batch(
        ["hello world", "test sequence"],
        tokenizer=_dummy_tok,
        seq_len=16,
        pad_token_id=0,
        batch_size=2,
        target_start_fraction=0.5,
        valid_lengths=[11, 13],
    )
    self.assertEqual(tokens.shape, (2, 16))
    self.assertEqual(seg_ids.shape, (2, 16))
    self.assertEqual(mask.shape, (2, 16))
    self.assertTrue(np.all(tokens[0, 11:] == 0))
    self.assertTrue(np.all(seg_ids[0, :11] == 1))
    self.assertTrue(np.all(seg_ids[0, 11:] == 0))

    class MockTokenizer:

      def __call__(self, s: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(c) % 50 + 1 for c in s]}

    tokens2, _, mask2 = zero_gen_logits_checker.tokenize_golden_batch(
        "single string batch",
        tokenizer=MockTokenizer(),
        seq_len=32,
        pad_token_id=0,
        batch_size=1,
        target_start_fraction=0.5,
    )
    self.assertEqual(tokens2.shape, (1, 32))
    self.assertTrue(np.any(mask2[0]))

  def test_expected_shard_sha256s_verification(self):
    """Verifies _verify_shard_sha256 succeeds on expected digest and raises on mismatch."""
    self.assertLen(_EXPECTED_SHARD_SHA256S, 13)
    with self.assertRaises(ValueError):
      ensure_real_mini_slice("")
    with self.assertRaises(FileNotFoundError):
      ensure_real_mini_slice(self.create_tempdir().full_path)

    shard = "model-00001-of-00062.safetensors"
    valid_sha = _EXPECTED_SHARD_SHA256S[shard]
    # 1. Matching shard and digest succeeds
    _verify_shard_sha256(shard, valid_sha)

    # 2. Unknown shard raises ValueError
    with self.assertRaisesRegex(ValueError, "Unknown shard name"):
      _verify_shard_sha256("model-99999-of-00062.safetensors", valid_sha)

    # 3. Digest mismatch raises ValueError
    with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
      _verify_shard_sha256(shard, "0" * 64)

  def test_golden_manifest_validation_and_isolation(self):
    """Verifies _validate_golden_manifest and _validate_golden_path_isolation without silent fallbacks."""
    raw_dir = "/path/to/raw_work_dir"
    slice_dir = "/path/to/slice_dir"

    # Empty or whitespace golden_path raises ValueError
    with self.assertRaisesRegex(ValueError, "--golden_path must be specified"):
      _validate_golden_path_isolation("", raw_dir, slice_dir)
    with self.assertRaisesRegex(ValueError, "--golden_path must be specified"):
      _validate_golden_path_isolation("   ", raw_dir, slice_dir)

    # Valid isolated path passes
    _validate_golden_path_isolation("/path/to/golden_dir", raw_dir, slice_dir)

    # Identical to raw raises ValueError
    with self.assertRaisesRegex(ValueError, "must not equal or reside within"):
      _validate_golden_path_isolation(raw_dir, raw_dir, slice_dir)

    # Subdirectory of raw raises ValueError
    with self.assertRaisesRegex(ValueError, "must not equal or reside within"):
      _validate_golden_path_isolation(f"{raw_dir}/nested_golden", raw_dir, slice_dir)

    # Identical to slice raises ValueError
    with self.assertRaisesRegex(ValueError, "must not equal or reside within"):
      _validate_golden_path_isolation(slice_dir, raw_dir, slice_dir)

    # Subdirectory of slice raises ValueError
    with self.assertRaisesRegex(ValueError, "must not equal or reside within"):
      _validate_golden_path_isolation(f"{slice_dir}/hf_ref_cache_v1", raw_dir, slice_dir)

    temp_dir = self.create_tempdir().full_path
    manifest_path = os.path.join(temp_dir, "manifest.json")

    code_sha = _compute_hf_loader_code_sha256()
    self.assertEqual(len(code_sha), 64)

    kwargs = {
        "expected_num_layers": 7,
        "expected_slice_sha256": "abc123slice",
        "expected_seq_lens": [512, 4096],
        "expected_inputs_sha256": "def456inputs",
        "expected_hf_loader_code_sha256": code_sha,
        "num_moe": 1,
        "num_full": 1,
    }

    # Empty or whitespace golden_path raises ValueError
    with self.assertRaisesRegex(ValueError, "golden_path must be specified"):
      _validate_golden_manifest("", **kwargs)
    with self.assertRaisesRegex(ValueError, "golden_path must be specified"):
      _validate_golden_manifest("   ", **kwargs)

    # Missing manifest.json raises FileNotFoundError
    with self.assertRaises(FileNotFoundError):
      _validate_golden_manifest(temp_dir, **kwargs)

    dummy_arr = np.zeros((1, 1), dtype=np.float32)
    dummy_sha = hashlib.sha256(dummy_arr.tobytes()).hexdigest()
    arrays_meta = {}
    for sl in (512, 4096):
      for fname in (
          f"hf_logits_{sl}.npy",
          f"hf_exp_{sl}_0.npy",
          f"hf_rlog_{sl}_0.npy",
          f"hf_dsa_{sl}_0.npy",
      ):
        _atomic_save_npy(os.path.join(temp_dir, fname), dummy_arr)
        arrays_meta[fname] = {
            "sha256": dummy_sha,
            "shape": [1, 1],
            "dtype": "float32",
        }

    valid_manifest = {
        "schema_version": 2,
        "num_hidden_layers": 7,
        "slice_sha256": "abc123slice",
        "seq_lens": [512, 4096],
        "inputs_sha256": "def456inputs",
        "hf_loader_code_sha256": code_sha,
        "torch_numerics": {
            "torch_version": "2.4.0",
            "float32_matmul_precision": "high",
            "param_dtype": "float32",
        },
        "arrays": arrays_meta,
    }
    _atomic_save_json(manifest_path, valid_manifest)

    self.assertFalse(any(".tmp." in fname for fname in os.listdir(temp_dir)))

    # Exact match validates successfully (even without transformers_version)
    validated = _validate_golden_manifest(temp_dir, **kwargs)
    self.assertEqual(validated["schema_version"], 2)

    # Unknown schema_version raises ValueError
    bad_schema = dict(valid_manifest, schema_version=99)
    _atomic_save_json(manifest_path, bad_schema)
    with self.assertRaisesRegex(ValueError, "Unsupported golden manifest schema_version"):
      _validate_golden_manifest(temp_dir, **kwargs)

    # Missing key raises ValueError
    for k in valid_manifest:
      bad = dict(valid_manifest)
      del bad[k]
      _atomic_save_json(manifest_path, bad)
      with self.assertRaisesRegex(ValueError, "missing required key"):
        _validate_golden_manifest(temp_dir, **kwargs)

    # Mismatched slice_sha256 raises ValueError
    bad_slice = dict(valid_manifest, slice_sha256="mismatched_slice")
    _atomic_save_json(manifest_path, bad_slice)
    with self.assertRaisesRegex(ValueError, "slice_sha256 mismatch"):
      _validate_golden_manifest(temp_dir, **kwargs)

    # Mismatched seq_lens raises ValueError
    bad_seq = dict(valid_manifest, seq_lens=[512])
    _atomic_save_json(manifest_path, bad_seq)
    with self.assertRaisesRegex(ValueError, "seq_lens mismatch"):
      _validate_golden_manifest(temp_dir, **kwargs)

    # Mismatched inputs_sha256 raises ValueError
    bad_inputs = dict(valid_manifest, inputs_sha256="mismatched_inputs")
    _atomic_save_json(manifest_path, bad_inputs)
    with self.assertRaisesRegex(ValueError, "inputs_sha256 mismatch"):
      _validate_golden_manifest(temp_dir, **kwargs)

    # Mismatched hf_loader_code_sha256 raises ValueError
    bad_code = dict(valid_manifest, hf_loader_code_sha256="mismatched_code")
    _atomic_save_json(manifest_path, bad_code)
    with self.assertRaisesRegex(ValueError, "hf_loader_code_sha256 mismatch"):
      _validate_golden_manifest(temp_dir, **kwargs)

    # Missing array file raises FileNotFoundError
    _atomic_save_json(manifest_path, valid_manifest)
    os.remove(os.path.join(temp_dir, "hf_logits_512.npy"))
    with self.assertRaises(FileNotFoundError):
      _validate_golden_manifest(temp_dir, **kwargs)

    # Array shape mismatch raises ValueError
    _atomic_save_npy(os.path.join(temp_dir, "hf_logits_512.npy"), np.zeros((2, 2), dtype=np.float32))
    with self.assertRaisesRegex(ValueError, "Array shape mismatch"):
      _validate_golden_manifest(temp_dir, **kwargs)

    # Array sha256 mismatch raises ValueError
    _atomic_save_npy(os.path.join(temp_dir, "hf_logits_512.npy"), np.ones((1, 1), dtype=np.float32))
    with self.assertRaisesRegex(ValueError, "Array sha256 mismatch"):
      _validate_golden_manifest(temp_dir, **kwargs)

  def test_real_slice_seq_len_flag_validation(self):
    """Verifies --real_slice_seq_len flag validation on supported and unsupported values."""
    for val in ("all", "512", "4096"):
      with flagsaver.flagsaver((_REAL_SLICE_SEQ_LEN, val)):
        self.assertIn(_REAL_SLICE_SEQ_LEN.value, ("all", "512", "4096"))
    for invalid in ("invalid", "1024", "", "ALL"):
      with flagsaver.flagsaver((_REAL_SLICE_SEQ_LEN, invalid), (_RUN_REAL_SLICE, True)):
        inst = Glm5RealSliceParityTest()
        with self.assertRaisesRegex(ValueError, "Unsupported --real_slice_seq_len"):
          inst.setUp()

  def test_stage_or_resolve_golden_dir_validation_and_gcs_staging(self):
    """Verifies _stage_or_resolve_golden_dir handles validation, local paths, and GCS staging."""
    with self.assertRaisesRegex(ValueError, "--golden_path must be explicitly specified"):
      _stage_or_resolve_golden_dir("")
    with self.assertRaisesRegex(ValueError, "--golden_path must be explicitly specified"):
      _stage_or_resolve_golden_dir("   ")

    temp_dir = self.create_tempdir().full_path
    non_existent = os.path.join(temp_dir, "does_not_exist")
    with self.assertRaisesRegex(FileNotFoundError, "Golden path directory does not exist"):
      _stage_or_resolve_golden_dir(non_existent)

    # Local path returns as is when it exists
    self.assertEqual(_stage_or_resolve_golden_dir(temp_dir), temp_dir)

    # Simulated GCS path staging
    gcs_path = "gs://test-bucket/glm5/golden"
    orig_epath_path = epath.Path

    # 1. Missing manifest test
    gcs_empty_dir = self.create_tempdir("gcs_empty").full_path

    def fake_empty_path(p, *args, **kwargs):
      p_str = str(p)
      if p_str.startswith("gs://"):
        rel = os.path.relpath(p_str, gcs_path)
        return orig_epath_path(gcs_empty_dir) if rel == "." else orig_epath_path(gcs_empty_dir) / rel
      return orig_epath_path(p, *args, **kwargs)

    with mock.patch.object(epath, "Path", side_effect=fake_empty_path):
      with self.assertRaisesRegex(FileNotFoundError, "Golden manifest not found in GCS golden_path"):
        _stage_or_resolve_golden_dir(gcs_path)

    # 2. Populated manifest and arrays staging test
    gcs_populated_dir = self.create_tempdir("gcs_populated").full_path
    manifest_path = os.path.join(gcs_populated_dir, "manifest.json")
    dummy_manifest = {
        "schema_version": 2,
        "arrays": {"hf_logits_512.npy": {"shape": [2, 512, 10], "dtype": "float32"}},
    }
    manifest_bytes = json.dumps(dummy_manifest).encode("utf-8")
    arr_bytes = b"dummy_array_content"
    with open(manifest_path, "wb") as f:
      f.write(manifest_bytes)
    with open(os.path.join(gcs_populated_dir, "hf_logits_512.npy"), "wb") as f:
      f.write(arr_bytes)

    def fake_populated_path(p, *args, **kwargs):
      p_str = str(p)
      if p_str.startswith("gs://"):
        rel = os.path.relpath(p_str, gcs_path)
        return orig_epath_path(gcs_populated_dir) if rel == "." else orig_epath_path(gcs_populated_dir) / rel
      return orig_epath_path(p, *args, **kwargs)

    with mock.patch.object(epath, "Path", side_effect=fake_populated_path):
      staged_dir = _stage_or_resolve_golden_dir(gcs_path)
      self.assertTrue(os.path.isdir(staged_dir))
      staged_manifest = os.path.join(staged_dir, "manifest.json")
      self.assertTrue(os.path.exists(staged_manifest))
      with open(staged_manifest, "r", encoding="utf-8") as f:
        self.assertEqual(json.load(f), dummy_manifest)
      staged_arr = os.path.join(staged_dir, "hf_logits_512.npy")
      self.assertTrue(os.path.exists(staged_arr))
      with open(staged_arr, "rb") as f:
        self.assertEqual(f.read(), arr_bytes)

  def test_hf_safetensors_key_bijection_check(self):
    """Verifies _verify_hf_safetensors_key_bijection catches unconsumed and unexpected keys."""
    keys = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.input_layernorm.weight_scale_inv",
        "model.language_model.layers.1.input_layernorm.weight",
    }
    # 1. Exact match for num_layers=2 succeeds
    _verify_hf_safetensors_key_bijection(keys, set(keys), num_layers=2)

    # 2. Keys for layers >= num_layers (e.g. layer 5 or MTP layer 45) are ignored
    keys_with_extra_layer = set(keys) | {"model.language_model.layers.45.input_layernorm.weight"}
    _verify_hf_safetensors_key_bijection(keys_with_extra_layer, set(keys), num_layers=2)

    # 3. Unconsumed key in scope raises ValueError
    missing_one = set(keys) - {"model.language_model.layers.0.input_layernorm.weight"}
    with self.assertRaisesRegex(ValueError, "Safetensors key bijection check failed: unconsumed="):
      _verify_hf_safetensors_key_bijection(keys, missing_one, num_layers=2)

    # 4. Unexpected consumed key raises ValueError
    unexpected_one = set(keys) | {"unexpected.key.weight"}
    with self.assertRaisesRegex(ValueError, "Safetensors key bijection check failed:.*unexpected="):
      _verify_hf_safetensors_key_bijection(keys, unexpected_one, num_layers=2)

  def test_glm5_config_from_hf_text_config_dict_matches_testdata(self):
    """Verifies complete field mapping and parity check against testdata config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "testdata", "glm5_3_flash_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    text_cfg_dict = data["text_config"]
    num_layers = 5
    layer_types = tuple(text_cfg_dict["layer_types"][:num_layers])
    indexer_types = tuple(text_cfg_dict["indexer_types"][:num_layers])
    with self.assertRaises(TypeError):
      _glm5_config_from_hf_text_config_dict(  # pylint: disable=missing-kwoa
          text_cfg_dict,
          num_hidden_layers=num_layers,
          layer_types=layer_types,
          indexer_types=indexer_types,
      )

    c, hf_c = _glm5_config_from_hf_text_config_dict(
        text_cfg_dict,
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        indexer_types=indexer_types,
        dtype="float32",
        weight_dtype="float32",
        kda_implementation="xla",
    )
    self.assertEqual(c.num_hidden_layers, 5)
    self.assertEqual(c.hidden_size, 4096)
    self.assertEqual(c.mlp_clamp_limit, 10.0)
    self.assertEqual(c.mlp_clamp_limit, hf_c.swiglu_limit)
    self.assertEqual(c.routed_scaling_factor, 2.5)
    self.assertEqual(c.routed_scaling_factor, hf_c.routed_scaling_factor)
    self.assertEqual(c.rms_norm_eps, 1e-5)
    self.assertEqual(c.rms_norm_eps, hf_c.rms_norm_eps)
    self.assertEqual(c.linear_lower_bound, -5.0)
    self.assertEqual(c.linear_lower_bound, hf_c.linear_lower_bound)

    py_cfg = pyconfig.initialize(["", "src/maxtext/configs/base.yml", "model_name=glm5.3-flash"])
    prod_yml_c = glm5_config.Glm5Config.from_maxtext_config(
        py_cfg,
        num_hidden_layers=num_layers,
        dtype="float32",
        weight_dtype="float32",
        kda_implementation="xla",
    )
    prod_default_c = glm5_config.Glm5Config(
        num_hidden_layers=num_layers,
        dtype="float32",
        weight_dtype="float32",
        kda_implementation="xla",
    )
    arch_fields = (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "first_k_dense_replace",
        "layer_types",
        "indexer_types",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "linear_lower_bound",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "rope_parameters",
        "use_indexer",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "index_kpool",
        "index_kpool_always_select_tail",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
        "n_shared_experts",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "norm_topk_prob",
        "scoring_func",
        "mlp_clamp_limit",
        "moe_lb_loss_weight",
        "mhc_expansion_rate",
        "mhc_sinkhorn_iters",
        "mhc_eps",
        "rms_norm_eps",
        "num_nextn_predict_layers",
        "mtp_loss_scaling_factor",
    )
    for field_name in arch_fields:
      self.assertEqual(
          getattr(c, field_name),
          getattr(prod_yml_c, field_name),
          f"Mismatch on {field_name} vs from_maxtext_config(glm5.3-flash.yml)",
      )
      self.assertEqual(
          getattr(c, field_name),
          getattr(prod_default_c, field_name),
          f"Mismatch on {field_name} vs Glm5Config()",
      )

  def test_load_hf_model_from_fp8_slice_parity(self):
    """Verifies hermetic loading of FP8 slice via native FineGrainedFP8Config and parity with JAX."""
    c, hf_c = _make_cfg(num_hidden_layers=4, scan_layers=False)
    _, hf_m, _, hub_state = self._prepare_model_and_hf_state(num_layers=4, scan_layers=False)
    text_cfg_dict = dict(hf_m.config.to_dict())
    vision_cfg_dict = {
        "depth": 1,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 4,
        "out_hidden_size": 64,
        "projection_intermediate_size": 128,
    }
    hf_vis = hf_model.Glm5NextVisionModel(hf_cfg.Glm5NextVisionConfig(**vision_cfg_dict)).to(torch.float32)

    hub_fp8 = {f"model.visual.{k}": v.detach().clone() for k, v in hf_vis.state_dict().items()}
    quantized_keys: set[str] = set()
    for k, v in hub_state.items():
      self.assertEqual(
          _is_real_ckpt_fp8_quantized_key(k, c.layer_types),
          _is_real_ckpt_fp8_quantized_key(k, tuple(hf_c.layer_types)),
      )
      if _is_real_ckpt_fp8_quantized_key(k, c.layer_types):
        m, n = v.shape
        s = torch.full((math.ceil(m / 128), math.ceil(n / 128)), 1.25, dtype=torch.float32)
        exp_s = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :n]
        w_fp8 = (v / exp_s).to(torch.float8_e4m3fn)
        hub_fp8[k] = w_fp8
        scale_k = k[:-7] + ".weight_scale_inv"
        hub_fp8[scale_k] = s
        quantized_keys.add(k)
      else:
        hub_fp8[k] = v.to(torch.bfloat16) if v.ndim == 2 and k.endswith(".weight") else v

    expected_quantized = {
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.up_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.language_model.layers.3.self_attn.q_a_proj.weight",
        "model.language_model.layers.3.self_attn.q_b_proj.weight",
        "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight",
    }
    for k in expected_quantized:
      self.assertIn(k, quantized_keys)

    forbidden_unquantized = {
        "lm_head.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.kv_b_proj.weight",
        "model.language_model.layers.3.self_attn.indexer.wq_b.weight",
        "model.language_model.layers.3.mlp.gate.weight",
    }
    for k in forbidden_unquantized:
      self.assertNotIn(k, quantized_keys)

    with tempfile.TemporaryDirectory() as slice_dir:
      safetensors.torch.save_file(hub_fp8, os.path.join(slice_dir, "model.safetensors"))
      with open(os.path.join(slice_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "glm5_next",
                "text_config": text_cfg_dict,
                "vision_config": vision_cfg_dict,
            },
            f,
        )
      hf_loaded, loading_info = _load_hf_model_from_fp8_slice(slice_dir, text_cfg_dict, vision_cfg_dict=vision_cfg_dict)
      self.assertEmpty(loading_info["missing_keys"])
      self.assertEmpty(loading_info["unexpected_keys"])
      self.assertEmpty(loading_info["mismatched_keys"])
      self.assertEmpty(loading_info["error_msgs"])
      for p_name, p in hf_loaded.named_parameters():
        if p.is_floating_point():
          self.assertEqual(
              p.dtype,
              torch.float32,
              msg=f"Parameter {p_name} has dtype {p.dtype}",
          )

      jax_loaded = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
      glm5_ckpt.load_hf_state_into_model(jax_loaded, hub_fp8, ignore_mtp_layer=True)

      np.testing.assert_allclose(
          np.array(jax_loaded.lm_head.kernel.value),
          hf_loaded.lm_head.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.prefix_layers[0].self_attn.o_proj.kernel.value),
          hf_loaded.model.language_model.layers[0].self_attn.o_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.prefix_layers[0].self_attn.f_a_proj.kernel.value),
          hf_loaded.model.language_model.layers[0].self_attn.forget_gate.f_a_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.prefix_layers[0].self_attn.f_b_proj.kernel.value),
          hf_loaded.model.language_model.layers[0].self_attn.forget_gate.f_b_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.prefix_layers[0].mlp.gate_proj.kernel.value),
          hf_loaded.model.language_model.layers[0].mlp.gate_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.suffix_layers[0].self_attn.q_a_proj.kernel.value),
          hf_loaded.model.language_model.layers[3].self_attn.q_a_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.suffix_layers[0].self_attn.o_proj.kernel.value),
          hf_loaded.model.language_model.layers[3].self_attn.o_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.suffix_layers[0].self_attn.kv_b_proj.kernel.value),
          hf_loaded.model.language_model.layers[3].self_attn.kv_b_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.suffix_layers[0].mlp.shared_experts.gate_proj.kernel.value),
          hf_loaded.model.language_model.layers[3].mlp.shared_experts.gate_proj.weight.detach().cpu().numpy().T,
          atol=1e-5,
          rtol=1e-5,
      )
      np.testing.assert_allclose(
          np.array(jax_loaded.suffix_layers[0].mlp.experts.down_proj.value),
          hf_loaded.model.language_model.layers[3].mlp.experts.down_proj.detach().cpu().numpy().swapaxes(1, 2),
          atol=1e-5,
          rtol=1e-5,
      )

      tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
      jax_logits = np.array(jax_loaded(jnp.array(tokens)))
      with torch.no_grad():
        with mock.patch.object(hf_moe_integ, "_can_use_grouped_mm", return_value=False):
          hf_out = hf_loaded(input_ids=torch.from_numpy(tokens).long())
          hf_logits = hf_out.logits.detach().cpu().numpy()
      np.testing.assert_allclose(jax_logits, hf_logits, atol=1e-4, rtol=1e-4)

  def test_dsa_index_match_rate_when_tokens_zero(self):
    """When s <= index_topk or total_dsa_tokens == 0, dsa_eval_token_count is 0 and dsa_index_match_rate is NaN."""
    b, s, v = 2, 32, 64
    jax_logits = np.random.randn(b, s, v).astype(np.float32)
    ref_logits = jax_logits.copy()
    target_mask = np.ones((b, s), dtype=bool)

    # index_topk=64 with index_kpool=4 means topk_pools=16.
    # With s=32, vis_pools <= 32 // 4 = 8 < 16, so dsa_token_mask is all False.
    jax_dsa = [np.random.randint(0, 16, (b, s, 4))]
    ref_dsa = [jax_dsa[0].copy()]

    res = zero_gen_logits_checker.evaluate_masked_prefill_parity(
        jax_logits=jax_logits,
        ref_logits=ref_logits,
        target_mask=target_mask,
        jax_expert_indices=[np.zeros((b, s, 2), dtype=np.int32)],
        ref_expert_indices=[np.zeros((b, s, 2), dtype=np.int32)],
        jax_dsa_indices=jax_dsa,
        ref_dsa_indices=ref_dsa,
        single_row_logits=[jax_logits[i] for i in range(b)],
        index_topk=64,
        index_kpool=4,
    )
    self.assertEqual(res["dsa_eval_token_count"], 0)
    self.assertTrue(np.isnan(res["dsa_index_match_rate"]))
    self.assertTrue(res["tier_1a_passed"])

  def test_dsa_index_match_rate_when_tokens_positive(self):
    """When s > index_topk, dsa_eval_token_count > 0 and dsa_index_match_rate is 1.0 on matching indices."""
    b, s, v = 2, 64, 64
    jax_logits = np.random.randn(b, s, v).astype(np.float32)
    ref_logits = jax_logits.copy()
    target_mask = np.ones((b, s), dtype=bool)

    # index_topk=16 with index_kpool=4 means topk_pools=4.
    # With s=64, vis_pools reaches 16 > 4, so dsa_token_mask has True entries.
    jax_dsa = [np.random.randint(0, 16, (b, s, 4))]
    ref_dsa = [jax_dsa[0].copy()]

    res = zero_gen_logits_checker.evaluate_masked_prefill_parity(
        jax_logits=jax_logits,
        ref_logits=ref_logits,
        target_mask=target_mask,
        jax_expert_indices=[np.zeros((b, s, 2), dtype=np.int32)],
        ref_expert_indices=[np.zeros((b, s, 2), dtype=np.int32)],
        jax_dsa_indices=jax_dsa,
        ref_dsa_indices=ref_dsa,
        single_row_logits=[jax_logits[i] for i in range(b)],
        index_topk=16,
        index_kpool=4,
    )
    self.assertGreater(res["dsa_eval_token_count"], 0)
    self.assertEqual(res["dsa_index_match_rate"], 1.0)

  def test_evaluate_masked_prefill_parity_mismatched_single_row_length_raises(
      self,
  ):
    """Verifies evaluate_masked_prefill_parity raises ValueError on mismatched single-row length (P2-48)."""
    batch_size = 2
    seq_len = 16
    vocab_size = 32
    jax_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
    ref_logits = jax_logits.copy()
    target_mask = np.ones((batch_size, seq_len), dtype=bool)
    seg_ids = np.ones((batch_size, seq_len), dtype=np.int32)
    dummy_indices = [np.zeros((batch_size, seq_len, 2), dtype=np.int32)]
    # Mismatched length: s_row has length 10 while jb_valid has length 16
    bad_single_row_logits = [
        np.random.randn(10, vocab_size).astype(np.float32),
        np.random.randn(seq_len, vocab_size).astype(np.float32),
    ]
    with self.assertRaisesRegex(
        ValueError,
        r"Single-row logits length \(10\) does not match batched valid length" r" \(16\)",
    ):
      zero_gen_logits_checker.evaluate_masked_prefill_parity(
          jax_logits=jax_logits,
          ref_logits=ref_logits,
          target_mask=target_mask,
          jax_expert_indices=dummy_indices,
          ref_expert_indices=dummy_indices,
          jax_dsa_indices=dummy_indices,
          ref_dsa_indices=dummy_indices,
          single_row_logits=bad_single_row_logits,
          decoder_segment_ids=seg_ids,
      )

  def test_evaluate_masked_prefill_parity_mismatched_batch_size_raises(self):
    """Verifies evaluate_masked_prefill_parity raises ValueError on mismatched batch size (T43)."""
    batch_size = 2
    seq_len = 16
    vocab_size = 32
    jax_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
    ref_logits = jax_logits.copy()
    target_mask = np.ones((batch_size, seq_len), dtype=bool)
    dummy_indices = [np.zeros((batch_size, seq_len, 2), dtype=np.int32)]
    # Mismatched length: bad_single_row_logits has 1 element, batch_size is 2
    bad_single_row_logits = [
        np.random.randn(seq_len, vocab_size).astype(np.float32),
    ]
    with self.assertRaisesRegex(
        ValueError,
        r"Single-row logits list length \(1\) does not match batch size \(2\)",
    ):
      zero_gen_logits_checker.evaluate_masked_prefill_parity(
          jax_logits=jax_logits,
          ref_logits=ref_logits,
          target_mask=target_mask,
          jax_expert_indices=dummy_indices,
          ref_expert_indices=dummy_indices,
          jax_dsa_indices=dummy_indices,
          ref_dsa_indices=dummy_indices,
          single_row_logits=bad_single_row_logits,
      )


# Authoritative FP8-quantized weight suffix allowlist from the real checkpoint index
# (`zai-org/GLM-5.3-Flash/model.safetensors.index.json` /
# `/dev/shm/hf_mini/glm5_3_flash_5layers/model.safetensors.index.json` and
# `config.json` `quantization_config.modules_to_not_convert`):
# Only these 13 per-layer `.weight` patterns carry `.weight_scale_inv` in the real
# checkpoint (`self_attn.o_proj.weight` is quantized ONLY in MLA layers where
# `layer_types[layer_idx] == "deepseek_sparse_attention"`, and stays unquantized on KDA layers).
# Unquantized 2D `.weight` tensors in the real checkpoint include `lm_head.weight`,
# `embed_tokens.weight`, `mlp.gate.weight` (router), `self_attn.kv_b_proj.weight`,
# all KDA projections (`q_proj`, `k_proj`, `v_proj`, `f_a_proj`, `f_b_proj`,
# `g_a_proj`, `g_b_proj`, `b_proj`, KDA `o_proj`),
# `self_attn.indexer.{wq_b,wk,weights_proj}.weight`, and MTP `eh_proj.weight`.
_REAL_CKPT_FP8_WEIGHT_PATTERNS: tuple[str, ...] = (
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp.experts.*.gate_proj.weight",
    "mlp.experts.*.up_proj.weight",
    "mlp.experts.*.down_proj.weight",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.o_proj.weight",  # MLA layers only ("deepseek_sparse_attention" / "full_attention")
)


def _is_real_ckpt_fp8_quantized_key(
    hub_key: str,
    layer_types: tuple[str, ...] | list[str],
) -> bool:
  """Returns True iff `hub_key` carries `.weight_scale_inv` in the real GLM-5.3-Flash checkpoint."""
  prefix = "model.language_model.layers."
  if not hub_key.startswith(prefix):
    return False
  rest = hub_key[len(prefix) :]
  layer_str, sep, suffix = rest.partition(".")
  if not sep or not layer_str.isdigit():
    return False
  layer_idx = int(layer_str)
  if suffix == "self_attn.o_proj.weight":
    return layer_idx < len(layer_types) and layer_types[layer_idx] in _HF_MLA_LAYER_TYPES
  if suffix.startswith("mlp.experts."):
    parts = suffix.split(".")
    if len(parts) == 5 and parts[2].isdigit():
      pattern = f"mlp.experts.*.{parts[3]}.{parts[4]}"
      return pattern in _REAL_CKPT_FP8_WEIGHT_PATTERNS
    return False
  return suffix in _REAL_CKPT_FP8_WEIGHT_PATTERNS


def _load_hf_model_from_fp8_slice(
    slice_dir: str,
    text_cfg_dict: dict[str, Any],
    *,
    vision_cfg_dict: dict[str, Any] | None = None,
) -> tuple[hf_model.Glm5NextForConditionalGeneration, dict[str, Any]]:
  """Loads HuggingFace GLM-5 model from an FP8 mini-slice directory."""
  quant_cfg = transformers.FineGrainedFP8Config(dequantize=True)
  quant_dict = quant_cfg.to_dict()
  full_cfg_kwargs: dict[str, Any] = {
      "text_config": text_cfg_dict,
      "quantization_config": quant_dict,
  }
  if vision_cfg_dict is not None:
    full_cfg_kwargs["vision_config"] = vision_cfg_dict
  hf_full_cfg = hf_cfg.Glm5NextConfig(**full_cfg_kwargs)
  # quantizer_finegrained_fp8.py:31-35 checks is_accelerate_available() before
  # returning early on quantization_config.dequantize=True. accelerate is
  # intentionally not a dep; any real use raises ImportError.
  with mock.patch(
      "transformers.quantizers.quantizer_finegrained_fp8.is_accelerate_available",
      return_value=True,
  ):
    hf_m, loading_info = hf_model.Glm5NextForConditionalGeneration.from_pretrained(
        str(slice_dir),
        config=hf_full_cfg,
        dtype=torch.float32,
        quantization_config=quant_cfg,
        output_loading_info=True,
    )
  missing_lm = sorted(k for k in loading_info["missing_keys"] if not k.startswith("model.visual."))
  unexpected_lm = sorted(
      k
      for k in loading_info["unexpected_keys"]
      if not k.startswith(f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}.")
  )
  mismatched = list(loading_info["mismatched_keys"])
  error_msgs = list(loading_info["error_msgs"])
  if missing_lm or unexpected_lm or mismatched or error_msgs:
    raise ValueError(
        "HF from_pretrained loading report had issues:"
        f" missing_lm={missing_lm}, unexpected_lm={unexpected_lm},"
        f" mismatched={mismatched}, error_msgs={error_msgs}"
    )
  # transformers core_model_loading.py:1694-1708 skips the
  # dtype=torch.float32 cast on renamed keys under a pre-quantized quantizer
  # (such as self_attn.f_a_proj.weight -> self_attn.forget_gate.f_a_proj.weight
  # in conversion_mapping.py:536-543), and finegrained_fp8.py:1074-1078 returns
  # unquantized weights without casting to output_dtype. Promote all floating
  # parameters/buffers to float32 so PyTorch matches glm5_ckpt.py:440-443.
  hf_m = hf_m.to(torch.float32)
  for p_name, p in hf_m.named_parameters():
    if p.is_floating_point() and p.dtype != torch.float32:
      raise ValueError(f"Expected float32 parameter after cast, got {p_name}: {p.dtype}")
  hf_m.eval()
  return hf_m, loading_info


def _compute_hf_loader_code_sha256() -> str:
  """Computes SHA-256 digest of HF model loading and golden reference generation source code."""
  hasher = hashlib.sha256()
  for mod in (hf_model, hf_cfg, quantizer_finegrained_fp8, hf_moe_integ):
    hasher.update(inspect.getsource(mod).encode("utf-8"))
  for fn in (_load_hf_model_from_fp8_slice, _generate_hf_golden_reference):
    tree = ast.parse(inspect.getsource(fn))
    hasher.update(ast.dump(tree).encode("utf-8"))
  return hasher.hexdigest()


def _generate_hf_golden_reference(slice_dir: str, golden_path: str) -> None:
  """Generates HF golden reference arrays and manifest into golden_path with atomic writes."""
  with open(os.path.join(slice_dir, "config.json"), "r", encoding="utf-8") as f:
    slice_cfg_data = json.load(f)
  text_cfg_dict = dict(slice_cfg_data["text_config"])
  num_layers = 7
  text_cfg_dict["num_hidden_layers"] = num_layers
  text_cfg_dict["_attn_implementation"] = "eager"
  for k in ("layer_types", "indexer_types", "mlp_layer_types"):
    if k in text_cfg_dict:
      text_cfg_dict[k] = text_cfg_dict[k][:num_layers]

  tokenizer = transformers.AutoTokenizer.from_pretrained(slice_dir, trust_remote_code=True)
  real_sections = [
      zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["prose"],
      zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["code_json_math"],
  ]
  tokens_512, seg_ids_512, _ = zero_gen_logits_checker.tokenize_golden_batch(
      real_sections,
      tokenizer=tokenizer,
      seq_len=512,
      pad_token_id=154820,
      batch_size=2,
      target_start_fraction=0.5,
      valid_lengths=[512, 512 - 16],
      padding_side="left",
  )
  tokens_4096, seg_ids_4096, _ = zero_gen_logits_checker.tokenize_golden_batch(
      real_sections,
      tokenizer=tokenizer,
      seq_len=4096,
      pad_token_id=154820,
      batch_size=2,
      target_start_fraction=0.5,
      valid_lengths=[4096, 4096 - 16],
      padding_side="left",
  )

  slice_st_path = os.path.join(slice_dir, "model.safetensors")
  hasher = hashlib.sha256()
  with open(slice_st_path, "rb") as f:
    while chunk := f.read(65536):
      hasher.update(chunk)
  slice_sha256 = hasher.hexdigest()
  inputs_sha256 = _compute_inputs_sha256(tokens_512, seg_ids_512, tokens_4096, seg_ids_4096)

  hf_m, _ = _load_hf_model_from_fp8_slice(slice_dir, text_cfg_dict)

  hf_expert_indices_512 = []
  hf_dsa_indices_512 = []
  hf_router_logits_512 = []
  hf_expert_indices_4096 = []
  hf_dsa_indices_4096 = []
  hf_router_logits_4096 = []

  def _make_hf_hooks(exp_list, dsa_list, rlog_list):
    def _g_hook(module, inputs, output):
      del module, inputs
      rlog_list.append(output[0].detach().cpu().numpy())
      exp_list.append(output[2].detach().cpu().numpy())

    def _d_hook(module, inputs, output):
      del module, inputs
      dsa_list.append(output.detach().cpu().numpy())

    gh = [
        hf_m.model.language_model.layers[i].mlp.gate.register_forward_hook(_g_hook)
        for i, lt in enumerate(text_cfg_dict["mlp_layer_types"][:num_layers])
        if lt == "sparse"
    ]
    dh = [
        hf_m.model.language_model.layers[i].self_attn.indexer.register_forward_hook(_d_hook)
        for i, lt in enumerate(text_cfg_dict["layer_types"][:num_layers])
        if lt in ("full_attention", "deepseek_sparse_attention")
    ]
    return gh + dh

  with mock.patch.object(hf_moe_integ, "_can_use_grouped_mm", return_value=False):
    hooks_512 = _make_hf_hooks(hf_expert_indices_512, hf_dsa_indices_512, hf_router_logits_512)
    with torch.no_grad():
      try:
        hf_logits_512 = (
            hf_m(
                input_ids=torch.from_numpy(tokens_512).long(),
                attention_mask=torch.from_numpy(seg_ids_512 > 0).bool(),
            )
            .logits.detach()
            .cpu()
            .numpy()
        )
      finally:
        for h in hooks_512:
          h.remove()

    hooks_4096 = _make_hf_hooks(hf_expert_indices_4096, hf_dsa_indices_4096, hf_router_logits_4096)
    with torch.no_grad():
      try:
        hf_logits_4096 = (
            hf_m(
                input_ids=torch.from_numpy(tokens_4096).long(),
                attention_mask=torch.from_numpy(seg_ids_4096 > 0).bool(),
            )
            .logits.detach()
            .cpu()
            .numpy()
        )
      finally:
        for h in hooks_4096:
          h.remove()

  del hf_m
  gc.collect()

  golden_dir = epath.Path(golden_path)
  golden_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = golden_dir / "manifest.json"
  if manifest_path.exists():
    manifest_path.unlink()

  arrays_meta: dict[str, dict[str, Any]] = {}

  def _save_array(fname: str, arr: np.ndarray) -> None:
    fpath = os.path.join(golden_path, fname)
    _atomic_save_npy(fpath, arr)
    arrays_meta[fname] = {
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }

  _save_array("hf_logits_512.npy", hf_logits_512)
  for i, arr in enumerate(hf_expert_indices_512):
    _save_array(f"hf_exp_512_{i}.npy", arr)
  for i, arr in enumerate(hf_dsa_indices_512):
    _save_array(f"hf_dsa_512_{i}.npy", arr)
  for i, arr in enumerate(hf_router_logits_512):
    _save_array(f"hf_rlog_512_{i}.npy", arr)

  _save_array("hf_logits_4096.npy", hf_logits_4096)
  for i, arr in enumerate(hf_expert_indices_4096):
    _save_array(f"hf_exp_4096_{i}.npy", arr)
  for i, arr in enumerate(hf_dsa_indices_4096):
    _save_array(f"hf_dsa_4096_{i}.npy", arr)
  for i, arr in enumerate(hf_router_logits_4096):
    _save_array(f"hf_rlog_4096_{i}.npy", arr)

  manifest_data = {
      "schema_version": 2,
      "num_hidden_layers": num_layers,
      "slice_sha256": slice_sha256,
      "seq_lens": [512, 4096],
      "inputs_sha256": inputs_sha256,
      "hf_loader_code_sha256": _compute_hf_loader_code_sha256(),
      "torch_numerics": {
          "torch_version": str(torch.__version__),
          "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
          "param_dtype": "float32",
      },
      "arrays": arrays_meta,
  }
  _atomic_save_json(str(manifest_path), manifest_data)


def _stage_or_resolve_slice_dir(raw_work_dir: str) -> str:
  """Validates mini-slice files in raw_work_dir, returning the local directory path."""
  required_slice_files = (
      "model.safetensors",
      "model.safetensors.index.json",
      "config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "shard_sha256s.json",
  )

  slice_dir = raw_work_dir
  for fname in required_slice_files:
    fpath = os.path.join(slice_dir, fname)
    if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
      raise FileNotFoundError(f"Required file missing or empty in work_dir {slice_dir!r}: {fpath}")
  return slice_dir


def _stage_or_resolve_golden_dir(raw_golden_path: str) -> str:
  """Validates and stages (if GCS) golden reference files, returning the local directory path."""
  if not raw_golden_path or not raw_golden_path.strip():
    raise ValueError("--golden_path must be explicitly specified.")
  if raw_golden_path.startswith("gs://"):
    local_dir = epath.Path(absltest.get_default_test_tmpdir()) / "glm5_3_flash_golden"
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_src = epath.Path(raw_golden_path) / "manifest.json"
    if not manifest_src.exists():
      raise FileNotFoundError(f"Golden manifest not found in GCS golden_path {raw_golden_path!r}: {manifest_src}")
    manifest_dst = local_dir / "manifest.json"
    manifest_dst.write_bytes(manifest_src.read_bytes())
    with manifest_dst.open("r", encoding="utf-8") as f:
      manifest_data = json.load(f)
    if "arrays" not in manifest_data or not manifest_data["arrays"]:
      raise ValueError(f"Golden manifest in GCS golden_path {raw_golden_path!r} contains no arrays.")
    arrays = manifest_data["arrays"]
    for fname in arrays.keys():
      src_path = epath.Path(raw_golden_path) / fname
      if not src_path.exists():
        raise FileNotFoundError(f"Required golden file {fname} not found in GCS golden_path {raw_golden_path!r}")
      dst_path = local_dir / fname
      src_sz = src_path.stat().length
      if not dst_path.exists() or dst_path.stat().length != src_sz:
        tmp_dst = local_dir / f"{fname}.tmp.{os.getpid()}"
        with src_path.open("rb") as sf, tmp_dst.open("wb") as df:
          while chunk := sf.read(64 * 1024 * 1024):
            df.write(chunk)
        tmp_dst.replace(dst_path)
    return str(local_dir)
  if not os.path.exists(raw_golden_path):
    raise FileNotFoundError(f"Golden path directory does not exist: {raw_golden_path!r}")
  return raw_golden_path


def _glm5_config_from_hf_text_config_dict(
    text_cfg_dict: dict[str, Any],
    *,
    num_hidden_layers: int,
    layer_types: tuple[str, ...],
    indexer_types: tuple[str, ...],
    dtype: str,
    weight_dtype: str,
    kda_implementation: str,
) -> tuple[glm5_config.Glm5Config, hf_cfg.Glm5NextTextConfig]:
  """Constructs aligned Glm5Config and Glm5NextTextConfig with exhaustive field-by-field verification."""
  hf_kwargs: dict[str, Any] = {
      **text_cfg_dict,
      "num_hidden_layers": num_hidden_layers,
      "layer_types": list(layer_types),
      "indexer_types": list(indexer_types),
  }
  if "mlp_layer_types" in text_cfg_dict:
    hf_kwargs["mlp_layer_types"] = list(text_cfg_dict["mlp_layer_types"][:num_hidden_layers])
  hf_c = hf_cfg.Glm5NextTextConfig(**hf_kwargs)

  # Production Glm5Config.__post_init__ (in glm5_3_flash_config.py) validates
  # lt in {"linear_attention", "full_attention"}; HF Glm5NextTextConfig
  # names MLA layers "deepseek_sparse_attention" (or "indexed_attention" in
  # newer transformers). No production helper translates HF layer_types
  # strings, so map once here.
  maxtext_layer_types = tuple("full_attention" if lt in _HF_MLA_LAYER_TYPES else lt for lt in hf_c.layer_types)

  # Note: rope_parameters, use_indexer, and mtp_loss_scaling_factor
  # are MaxText-only fields in Glm5Config and do not exist in Glm5NextTextConfig or HF config.json.
  c = glm5_config.Glm5Config(
      vocab_size=hf_c.vocab_size,
      hidden_size=hf_c.hidden_size,
      num_hidden_layers=hf_c.num_hidden_layers,
      first_k_dense_replace=hf_c.first_k_dense_replace,
      layer_types=maxtext_layer_types,
      indexer_types=tuple(hf_c.indexer_types),
      linear_num_key_heads=hf_c.linear_num_heads,
      linear_num_value_heads=hf_c.linear_num_heads,
      linear_key_head_dim=hf_c.linear_head_dim,
      linear_value_head_dim=hf_c.linear_head_dim,
      linear_conv_kernel_dim=hf_c.linear_conv_kernel_dim,
      linear_lower_bound=hf_c.linear_lower_bound,
      num_attention_heads=hf_c.num_attention_heads,
      num_key_value_heads=hf_c.num_key_value_heads,
      q_lora_rank=hf_c.q_lora_rank,
      kv_lora_rank=hf_c.kv_lora_rank,
      qk_nope_head_dim=hf_c.qk_nope_head_dim,
      qk_rope_head_dim=hf_c.qk_rope_head_dim,
      v_head_dim=hf_c.v_head_dim,
      index_n_heads=hf_c.index_n_heads,
      index_head_dim=hf_c.index_head_dim,
      index_topk=hf_c.index_topk,
      index_kpool=hf_c.index_kpool,
      index_kpool_always_select_tail=hf_c.index_kpool_always_select_tail,
      intermediate_size=hf_c.intermediate_size,
      moe_intermediate_size=hf_c.moe_intermediate_size,
      n_routed_experts=hf_c.n_routed_experts,
      num_experts_per_tok=hf_c.num_experts_per_tok,
      n_shared_experts=hf_c.n_shared_experts,
      n_group=hf_c.n_group,
      topk_group=hf_c.topk_group,
      routed_scaling_factor=hf_c.routed_scaling_factor,
      norm_topk_prob=hf_c.norm_topk_prob,
      scoring_func=hf_c.scoring_func,
      mlp_clamp_limit=hf_c.swiglu_limit,
      moe_lb_loss_weight=hf_c.router_aux_loss_coef,
      mhc_expansion_rate=hf_c.hc_mult,
      mhc_sinkhorn_iters=hf_c.hc_sinkhorn_iters,
      mhc_eps=hf_c.hc_eps,
      rms_norm_eps=hf_c.rms_norm_eps,
      num_nextn_predict_layers=hf_c.num_nextn_predict_layers,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kda_implementation=kda_implementation,
  )

  # Exhaustive shared architecture attribute equality check
  assert c.mlp_clamp_limit == hf_c.swiglu_limit, f"{c.mlp_clamp_limit} != {hf_c.swiglu_limit}"
  assert hf_c.mla_use_nope is True and c.rope_parameters is None
  assert (
      c.routed_scaling_factor == hf_c.routed_scaling_factor
  ), f"{c.routed_scaling_factor} != {hf_c.routed_scaling_factor}"
  assert c.rms_norm_eps == hf_c.rms_norm_eps, f"{c.rms_norm_eps} != {hf_c.rms_norm_eps}"
  assert c.linear_lower_bound == hf_c.linear_lower_bound, f"{c.linear_lower_bound} != {hf_c.linear_lower_bound}"
  assert c.linear_key_head_dim == hf_c.linear_head_dim, f"{c.linear_key_head_dim} != {hf_c.linear_head_dim}"
  assert c.linear_num_key_heads == hf_c.linear_num_heads, f"{c.linear_num_key_heads} != {hf_c.linear_num_heads}"
  assert (
      c.linear_conv_kernel_dim == hf_c.linear_conv_kernel_dim
  ), f"{c.linear_conv_kernel_dim} != {hf_c.linear_conv_kernel_dim}"
  assert c.mhc_expansion_rate == hf_c.hc_mult, f"{c.mhc_expansion_rate} != {hf_c.hc_mult}"
  assert c.mhc_sinkhorn_iters == hf_c.hc_sinkhorn_iters, f"{c.mhc_sinkhorn_iters} != {hf_c.hc_sinkhorn_iters}"
  assert c.mhc_eps == hf_c.hc_eps, f"{c.mhc_eps} != {hf_c.hc_eps}"
  assert c.vocab_size == hf_c.vocab_size
  assert c.hidden_size == hf_c.hidden_size
  assert c.num_hidden_layers == hf_c.num_hidden_layers
  assert c.first_k_dense_replace == hf_c.first_k_dense_replace
  assert ["full_attention" if lt in _HF_MLA_LAYER_TYPES else lt for lt in hf_c.layer_types] == list(c.layer_types)
  expected_mlp_layer_types = ["dense" if i < c.first_k_dense_replace else "sparse" for i in range(num_hidden_layers)]
  assert list(hf_c.mlp_layer_types) == expected_mlp_layer_types, f"{hf_c.mlp_layer_types} != {expected_mlp_layer_types}"
  assert list(c.indexer_types) == list(hf_c.indexer_types)
  assert c.num_attention_heads == hf_c.num_attention_heads
  assert c.num_key_value_heads == hf_c.num_key_value_heads
  assert c.q_lora_rank == hf_c.q_lora_rank
  assert c.kv_lora_rank == hf_c.kv_lora_rank
  assert c.qk_nope_head_dim == hf_c.qk_nope_head_dim
  assert c.qk_rope_head_dim == hf_c.qk_rope_head_dim
  assert c.v_head_dim == hf_c.v_head_dim
  assert c.index_n_heads == hf_c.index_n_heads
  assert c.index_head_dim == hf_c.index_head_dim
  assert c.index_topk == hf_c.index_topk
  assert c.index_kpool == hf_c.index_kpool
  assert c.index_kpool_always_select_tail == hf_c.index_kpool_always_select_tail
  assert c.intermediate_size == hf_c.intermediate_size
  assert c.moe_intermediate_size == hf_c.moe_intermediate_size
  assert c.n_routed_experts == hf_c.n_routed_experts
  assert c.num_experts_per_tok == hf_c.num_experts_per_tok
  assert c.n_shared_experts == hf_c.n_shared_experts
  assert c.n_group == hf_c.n_group
  assert c.topk_group == hf_c.topk_group
  assert c.norm_topk_prob == hf_c.norm_topk_prob
  assert c.scoring_func == hf_c.scoring_func
  assert c.num_nextn_predict_layers == hf_c.num_nextn_predict_layers
  return c, hf_c


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5RealSliceParityTest(absltest.TestCase):
  """Non-hermetic 7-layer real-weight mini-slice parity suite against HuggingFace."""

  _real_slice_tests_executed: int = 0

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if _RUN_REAL_SLICE.value and cls._real_slice_tests_executed == 0:
      raise AssertionError(
          "No real-slice parity tests were executed despite --run_real_slice=true! "
          f"Check --real_slice_seq_len={_REAL_SLICE_SEQ_LEN.value!r}."
      )

  def setUp(self):
    """Sets up test environment and verifies execution flags."""
    super().setUp()
    if _GENERATE_GOLDEN.value:
      raise RuntimeError("Test methods must not run under --generate_golden=true.")
    if not _RUN_REAL_SLICE.value:
      self.skipTest("Real-slice test requires --run_real_slice=true.")
    if _REAL_SLICE_SEQ_LEN.value not in ("all", "512", "4096"):
      raise ValueError(
          f"Unsupported --real_slice_seq_len={_REAL_SLICE_SEQ_LEN.value!r}; " "expected 'all', '512', or '4096'."
      )
    self._prev_matmul_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")

  def tearDown(self):
    jax.config.update("jax_default_matmul_precision", self._prev_matmul_precision)
    super().tearDown()

  def _run_real_7layer_mini_slice_prefill_parity_for_seq_len(self, *, seq_len: int) -> None:
    """Executes 7-layer mini-slice prefill parity for a single sequence length against HuggingFace golden reference."""
    if seq_len not in (512, 4096):
      raise ValueError(f"Unsupported seq_len: {seq_len}; expected 512 or 4096.")
    raw_work_dir = _WORK_DIR.value
    if not raw_work_dir or not raw_work_dir.strip():
      raise ValueError("--work_dir must be explicitly specified when --run_real_slice=true.")
    if not _GOLDEN_PATH.value or not _GOLDEN_PATH.value.strip():
      raise ValueError("--golden_path must be explicitly specified when --run_real_slice=true.")

    slice_dir = _stage_or_resolve_slice_dir(raw_work_dir)
    golden_dir = _stage_or_resolve_golden_dir(_GOLDEN_PATH.value)
    _validate_golden_path_isolation(_GOLDEN_PATH.value, raw_work_dir, slice_dir)
    _validate_golden_path_isolation(golden_dir, raw_work_dir, slice_dir)

    with open(
        os.path.join(slice_dir, "model.safetensors.index.json"),
        "r",
        encoding="utf-8",
    ) as f:
      weight_map = json.load(f)["weight_map"]

    num_layers = 7
    sel_keys = create_mini_slice.extract_mini_slice_keys(weight_map, num_layers, include_mtp_layer=45)
    self.assertNotEmpty(sel_keys)

    sha256_path = os.path.join(slice_dir, "shard_sha256s.json")
    with open(sha256_path, "r", encoding="utf-8") as f:
      shard_sha256s = json.load(f)
    self.assertEqual(
        set(shard_sha256s.keys()),
        set(_EXPECTED_SHARD_SHA256S.keys()),
        msg=f"Shard keys mismatch: {sorted(shard_sha256s.keys())}",
    )
    for s, computed_sha in sorted(shard_sha256s.items()):
      _verify_shard_sha256(s, computed_sha)

    slice_st_path = os.path.join(slice_dir, "model.safetensors")
    logging.info(
        "Real mini-slice shards (%d) and SHA256: %s",
        len(shard_sha256s),
        shard_sha256s,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(slice_dir, trust_remote_code=True)
    raw_enc = tokenizer.encode(
        zero_gen_logits_checker.GOLDEN_BATCH_TEXT,
        add_special_tokens=False,
    )
    tok_check, _, _ = zero_gen_logits_checker.tokenize_golden_batch(
        zero_gen_logits_checker.GOLDEN_BATCH_TEXT,
        tokenizer=tokenizer,
        seq_len=512,
        pad_token_id=154820,
        batch_size=2,
        target_start_fraction=0.5,
    )
    v_raw = min(512, len(raw_enc))
    np.testing.assert_array_equal(tok_check[0, :v_raw], np.asarray(raw_enc[:v_raw], dtype=np.int32))

    real_sections = [
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["prose"],
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["code_json_math"],
    ]
    tokens_512, seg_ids_512, mask_512 = zero_gen_logits_checker.tokenize_golden_batch(
        real_sections,
        tokenizer=tokenizer,
        seq_len=512,
        pad_token_id=154820,
        batch_size=2,
        target_start_fraction=0.5,
        valid_lengths=[512, 512 - 16],
        padding_side="left",
    )
    tokens_4096, seg_ids_4096, mask_4096 = zero_gen_logits_checker.tokenize_golden_batch(
        real_sections,
        tokenizer=tokenizer,
        seq_len=4096,
        pad_token_id=154820,
        batch_size=2,
        target_start_fraction=0.5,
        valid_lengths=[4096, 4096 - 16],
        padding_side="left",
    )

    hasher = hashlib.sha256()
    with open(slice_st_path, "rb") as f:
      while chunk := f.read(65536):
        hasher.update(chunk)
    slice_sha256 = hasher.hexdigest()

    inputs_sha256 = _compute_inputs_sha256(tokens_512, seg_ids_512, tokens_4096, seg_ids_4096)
    hf_loader_code_sha256 = _compute_hf_loader_code_sha256()

    self.assertNotEmpty(slice_sha256)
    self.assertLen(slice_sha256, 64)
    self.assertNotEmpty(inputs_sha256)
    logging.info(
        "Real mini-slice provenance: slice_sha256=%s inputs_sha256=%s",
        slice_sha256,
        inputs_sha256,
    )
    lazy_st = _LazySafetensorsMapping(slice_st_path)
    self.assertEqual(set(sel_keys.keys()), set(lazy_st.keys()))
    del lazy_st

    with open(os.path.join(slice_dir, "config.json"), "r", encoding="utf-8") as f:
      slice_cfg_data = json.load(f)
    text_cfg_dict = dict(slice_cfg_data["text_config"])
    c, _ = _glm5_config_from_hf_text_config_dict(
        text_cfg_dict,
        num_hidden_layers=num_layers,
        layer_types=tuple(text_cfg_dict["layer_types"][:num_layers]),
        indexer_types=tuple(text_cfg_dict["indexer_types"][:num_layers]),
        dtype="float32",
        weight_dtype="float32",
        kda_implementation="xla",
    )
    num_moe = sum(1 for i in range(num_layers) if i >= text_cfg_dict["first_k_dense_replace"])
    num_full = sum(1 for t in text_cfg_dict["layer_types"][:num_layers] if t in _HF_MLA_LAYER_TYPES)

    _validate_golden_manifest(
        golden_dir,
        expected_num_layers=num_layers,
        expected_slice_sha256=slice_sha256,
        expected_seq_lens=[512, 4096],
        expected_inputs_sha256=inputs_sha256,
        expected_hf_loader_code_sha256=hf_loader_code_sha256,
        num_moe=num_moe,
        num_full=num_full,
    )

    hf_logits = np.load(os.path.join(golden_dir, f"hf_logits_{seq_len}.npy"), mmap_mode="r")
    hf_expert_indices = [np.load(os.path.join(golden_dir, f"hf_exp_{seq_len}_{i}.npy")) for i in range(num_moe)]
    hf_dsa_indices = [np.load(os.path.join(golden_dir, f"hf_dsa_{seq_len}_{i}.npy")) for i in range(num_full)]
    hf_router_logits = [np.load(os.path.join(golden_dir, f"hf_rlog_{seq_len}_{i}.npy")) for i in range(num_moe)]

    if seq_len == 512:
      tokens = tokens_512
      seg_ids = seg_ids_512
      mask = mask_512
    else:
      tokens = tokens_4096
      seg_ids = seg_ids_4096
      mask = mask_4096

    c = dataclasses.replace(c, num_nextn_predict_layers=0, scan_layers=True, enable_debug_stats=True)
    self.assertEqual(c.hidden_size, text_cfg_dict["hidden_size"])
    self.assertEqual(c.intermediate_size, text_cfg_dict["intermediate_size"])
    self.assertEqual(c.num_attention_heads, text_cfg_dict["num_attention_heads"])
    self.assertEqual(c.num_key_value_heads, text_cfg_dict["num_key_value_heads"])
    self.assertEqual(c.n_routed_experts, text_cfg_dict["n_routed_experts"])
    self.assertEqual(c.num_experts_per_tok, text_cfg_dict["num_experts_per_tok"])
    self.assertEqual(c.moe_intermediate_size, text_cfg_dict["moe_intermediate_size"])
    self.assertEqual(c.first_k_dense_replace, text_cfg_dict["first_k_dense_replace"])
    self.assertEqual(c.index_topk, text_cfg_dict["index_topk"])
    self.assertEqual(c.index_kpool, text_cfg_dict["index_kpool"])

    lazy_st = _LazySafetensorsMapping(slice_st_path)
    jax_m = modeling_glm5_3_flash.Glm5Model(c, rngs=nnx.Rngs(0), place_on_devices=False)
    glm5_ckpt.load_hf_state_into_model(jax_m, lazy_st, ignore_mtp_layer=True)
    del lazy_st
    gc.collect()

    def _check_expert_mismatches(
        *,
        scan_layers_val: bool,
        seq_len_val: int,
        moe_idx: int,
        j_rlog_valid: np.ndarray,
        j_exp_valid: np.ndarray,
        h_exp_valid: np.ndarray,
    ) -> None:
      matched_valid = np.all(
          np.sort(j_exp_valid, axis=-1) == np.sort(h_exp_valid, axis=-1),
          axis=-1,
      )
      if not np.all(matched_valid):
        l = c.first_k_dense_replace + moe_idx
        sec, b_idx, s_idx = modeling_glm5_3_flash.layer_to_block_slot(l, c.first_k_dense_replace, c.num_hidden_layers)
        if sec == "super_block":
          target_slot = getattr(jax_m.super_block, f"slot_{s_idx}")
          bias_np = np.asarray(
              target_slot.mlp.gate.e_score_correction_bias[b_idx],
              dtype=np.float32,
          )
        else:
          target_layer = getattr(jax_m, f"{sec}_layers")[b_idx]
          bias_np = np.asarray(
              target_layer.mlp.gate.e_score_correction_bias[...],
              dtype=np.float32,
          )

        exp_per_group = c.n_routed_experts // c.n_group
        for tok_i in np.where(~matched_valid)[0]:
          j_tok_rlog = j_rlog_valid[tok_i].astype(np.float32)
          j_scores = 1.0 / (1.0 + np.exp(-np.clip(j_tok_rlog, -50.0, 50.0))) + bias_np
          grouped = j_scores.reshape(c.n_group, exp_per_group)
          group_scores = np.sum(np.sort(grouped, axis=-1)[:, -2:], axis=-1)
          topk_groups = np.argsort(group_scores)[-c.topk_group :]
          group_mask = np.zeros(c.n_group, dtype=bool)
          group_mask[topk_groups] = True
          score_mask = np.repeat(group_mask, exp_per_group)
          masked_scores = np.where(score_mask, j_scores, -1e30)
          sorted_scores = np.sort(masked_scores)[::-1]
          score_gap = float(abs(sorted_scores[c.num_experts_per_tok - 1] - sorted_scores[c.num_experts_per_tok]))
          print(
              f"[REAL_SLICE] scan_layers={scan_layers_val}"
              f" seq_len={seq_len_val} moe_layer={moe_idx} mismatch"
              f" tok_i={tok_i} jax_exp={j_exp_valid[tok_i].tolist()}"
              f" hf_exp={h_exp_valid[tok_i].tolist()} |score_8 -"
              f" score_9|={score_gap:.6e}",
              flush=True,
          )

    results: dict[bool, dict[str, Any]] = {}
    for scan_layers in (True, False):
      jax_m.config = dataclasses.replace(jax_m.config, scan_layers=scan_layers)

      out = jax_m.forward_with_aux(jnp.array(tokens), decoder_segment_ids=jnp.array(seg_ids))
      single = [
          jax_m(
              jnp.array(tokens[b : b + 1]),
              decoder_segment_ids=jnp.array(seg_ids[b : b + 1]),
          )
          for b in range(tokens.shape[0])
      ]
      jax_exp = list(out.aux["debug_stats"]["router_topk_indices"])
      jax_dsa = list(out.aux["debug_stats"]["dsa_topk_indices"])

      valid_mask = seg_ids > 0
      valid_flat = valid_mask.reshape(-1)
      for moe_idx, (j_rlog, h_rlog) in enumerate(zip(out.aux["router_logits"], hf_router_logits)):
        j_rlog_np = np.asarray(j_rlog).reshape(-1, j_rlog.shape[-1])[valid_flat]
        h_rlog_np = np.asarray(h_rlog).reshape(-1, h_rlog.shape[-1])[valid_flat]
        max_diff = float(np.max(np.abs(j_rlog_np - h_rlog_np)))
        print(
            f"[REAL_SLICE] scan_layers={scan_layers} moe_layer={moe_idx}" f" max_router_logit_diff={max_diff:.6e}",
            flush=True,
        )
        j_exp_flat = np.asarray(jax_exp[moe_idx]).reshape(-1, jax_exp[moe_idx].shape[-1])[valid_flat]
        h_exp_flat = np.asarray(hf_expert_indices[moe_idx]).reshape(-1, hf_expert_indices[moe_idx].shape[-1])[valid_flat]
        _check_expert_mismatches(
            scan_layers_val=scan_layers,
            seq_len_val=seq_len,
            moe_idx=moe_idx,
            j_rlog_valid=j_rlog_np,
            j_exp_valid=j_exp_flat,
            h_exp_valid=h_exp_flat,
        )

      res = zero_gen_logits_checker.evaluate_masked_prefill_parity(
          jax_logits=out.logits,
          ref_logits=hf_logits,
          target_mask=mask,
          decoder_segment_ids=seg_ids,
          jax_expert_indices=jax_exp,
          ref_expert_indices=hf_expert_indices,
          jax_dsa_indices=jax_dsa,
          ref_dsa_indices=hf_dsa_indices,
          single_row_logits=single,
          index_topk=c.index_topk,
          index_kpool=c.index_kpool,
      )
      print(
          f"[REAL_SLICE] scan_layers={scan_layers} seq_len={seq_len}"
          f" cos_sim={res['cos_sim']:.6f} l_inf={res['l_inf']:.6e}"
          f" expert_match={res['expert_index_match_rate']:.4f}"
          f" dsa_match={res['dsa_index_match_rate']:.4f}",
          flush=True,
      )
      results[scan_layers] = res

      del out, single, jax_exp, jax_dsa
      gc.collect()

    for sl in (True, False):
      res = results[sl]
      self.assertTrue(
          res["tier_1a_passed"],
          msg=f"Real-slice Tier 1A failed at {seq_len} (scan_layers={sl}): {res}",
      )
      self.assertEqual(res["expert_index_match_rate"], 1.0)
      if seq_len == 512:
        self.assertEqual(res["dsa_eval_token_count"], 0)
        self.assertTrue(np.isnan(res["dsa_index_match_rate"]))
      else:
        self.assertGreater(res["dsa_eval_token_count"], 0)
        self.assertEqual(res["dsa_index_match_rate"], 1.0)
      self.assertTrue(
          res["tier_1b_passed"],
          msg=f"Real-slice Tier 1B failed at {seq_len} (scan_layers={sl}): {res}",
      )
      self.assertGreater(res["cos_sim"], 0.999)
      self.assertLess(res["l_inf"], 1e-3)
      self.assertLess(res["max_kl_div"], 1e-3)
      self.assertGreater(res["logit_std"], 1e-4)
      self.assertLessEqual(res["mean_batch_diff"], 1e-4)
      self.assertTrue(res["all_passed"])

    del (
        results,
        hf_logits,
        hf_expert_indices,
        hf_dsa_indices,
        hf_router_logits,
        jax_m,
        _check_expert_mismatches,
    )
    jax.clear_caches()
    gc.collect()

  def test_real_7layer_mini_slice_prefill_parity_seq512(self):
    """Executes 7-layer mini-slice prefill parity for seq_len=512."""
    if _REAL_SLICE_SEQ_LEN.value not in ("all", "512", "4096"):
      raise ValueError(
          f"Unsupported --real_slice_seq_len={_REAL_SLICE_SEQ_LEN.value!r}; " "expected 'all', '512', or '4096'."
      )
    if _REAL_SLICE_SEQ_LEN.value not in ("all", "512"):
      return
    self.__class__._real_slice_tests_executed += 1
    self._run_real_7layer_mini_slice_prefill_parity_for_seq_len(seq_len=512)

  def test_real_7layer_mini_slice_prefill_parity_seq4096(self):
    """Executes 7-layer mini-slice prefill parity for seq_len=4096."""
    if _REAL_SLICE_SEQ_LEN.value not in ("all", "512", "4096"):
      raise ValueError(
          f"Unsupported --real_slice_seq_len={_REAL_SLICE_SEQ_LEN.value!r}; " "expected 'all', '512', or '4096'."
      )
    if _REAL_SLICE_SEQ_LEN.value not in ("all", "4096"):
      return
    self.__class__._real_slice_tests_executed += 1
    self._run_real_7layer_mini_slice_prefill_parity_for_seq_len(seq_len=4096)


def main(argv: list[str]) -> None:
  if _GENERATE_GOLDEN.value:
    if not _WORK_DIR.value or not _WORK_DIR.value.strip():
      raise ValueError("--work_dir must be specified when --generate_golden is set.")
    if not _GOLDEN_PATH.value or not _GOLDEN_PATH.value.strip():
      raise ValueError("--golden_path must be specified when --generate_golden is set.")
    slice_dir = _stage_or_resolve_slice_dir(_WORK_DIR.value)
    _validate_golden_path_isolation(_GOLDEN_PATH.value, _WORK_DIR.value, slice_dir)
    _generate_hf_golden_reference(slice_dir, _GOLDEN_PATH.value)
    print(f"[GOLDEN_GENERATION] Complete at {_GOLDEN_PATH.value}", flush=True)
    sys.exit(0)
  absltest.main(argv=argv)


if __name__ == "__main__":
  remaining_argv = flags.FLAGS(sys.argv)
  main(remaining_argv)
