# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Zero-Gen masked prefill driver and golden generator CLI."""

# pylint: disable=no-name-in-module,missing-class-docstring,missing-function-docstring
import os
import types
from typing import Any
import unittest
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen38_flash_next import modeling_qwen38_flash_next
from tests.utils import masked_prefill_driver as driver
from tests.utils import zero_gen_logits_checker as checker
import numpy as np
import pytest

configuration_qwen4_exp: Any
modeling_qwen4_exp: Any

try:
  from transformers.models.qwen4_exp import configuration_qwen4_exp  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import modeling_qwen4_exp  # pylint: disable=g-import-not-at-top

  _HAS_QWEN4_EXP = True
except ImportError:
  configuration_qwen4_exp = None
  modeling_qwen4_exp = None
  _HAS_QWEN4_EXP = False


def _create_2cycle_mini_config(seq_len: int = 16) -> types.SimpleNamespace:
  c = types.SimpleNamespace(max_target_length=seq_len, scan_layers=False, dtype=jnp.float32, weight_dtype=jnp.float32)
  u = c.__dict__.update
  u(emb_dim=64, num_decoder_layers=8, vocab_size=128, eos_token_id=127, ple_eos_token_id=127, hc_count=4)
  u(hc_lowrank=16, normalization_layer_epsilon=1e-6, inhomogeneous_layer_cycle_interval=4, gdn_chunk_size=16)
  u(gdn_num_key_heads=2, gdn_num_value_heads=4, gdn_key_head_dim=16, gdn_value_head_dim=16, gdn_conv_kernel_dim=4)
  u(gdn_output_gate_activation="sigmoid", num_query_heads=4, num_kv_heads=1, head_dim=16, partial_rotary_factor=0.25)
  u(rope_max_timescale=1e4, mrope_section=(1, 1, 0), index_n_heads=2, index_kv_heads=1, index_head_dim=16)
  u(index_select_budget=8, index_select_block_size=4, moe_mlp_dim=32, mlp_dim=32, num_experts=4)
  u(num_experts_per_tok=2, norm_topk_prob=True, routed_scaling_factor=1.0, shared_experts=1, use_ple_layer=False)
  u(heads_per_ngram=2, ngram_size=3, ple_embed_dim=64, ple_kernel_size=3, ple_conv_kernel_size=4, ple_seed=0)
  u(split_ngram_parts=2, ple_hash_seed=0, ngram_vocab_size_base=500, make_ngram_vocab_size_divisible_by=4)
  u(
      mtp_num_layers=1,
      mtp_num_streams=4,
      mtp_loss_scaling_factor=0.1,
      use_qk_norm_in_gdn=True,
      load_balance_loss_weight=0.001,
  )
  return c


def _make_matching_hf_config(c: Any) -> Any:
  assert configuration_qwen4_exp is not None
  l = int(c.num_decoder_layers)
  cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(num_hidden_layers=l, attention_bias=False, attention_dropout=0.0)
  m = (
      "index_select_block_size:indexer_compress_ratio gdn_output_gate_activation:output_gate_type ple_conv_kernel_size "
      "gdn_num_value_heads:linear_num_value_heads gdn_conv_kernel_dim:linear_conv_kernel_dim ngram_vocab_size_base "
      "gdn_value_head_dim:linear_value_head_dim normalization_layer_epsilon:rms_norm_eps index_n_heads:indexer_n_heads "
      "mlp_dim:shared_expert_intermediate_size gdn_num_key_heads:linear_num_key_heads num_kv_heads:num_key_value_heads "
      "gdn_key_head_dim:linear_key_head_dim num_query_heads:num_attention_heads make_ngram_vocab_size_divisible_by "
      "index_select_budget:indexer_budget moe_mlp_dim:moe_intermediate_size index_kv_heads:indexer_kv_heads vocab_size "
      "index_head_dim:indexer_head_dim ple_eos_token_id:eos_token_id num_experts_per_tok emb_dim:hidden_size head_dim "
      "heads_per_ngram norm_topk_prob ple_seed:seed ple_embed_dim num_experts ngram_size hc_lowrank hc_count"
  )
  for pair in m.split():
    src, dst = pair.split(":") if ":" in pair else (pair, pair)
    setattr(cfg, dst, getattr(c, src))
  cfg.ple_layer_ids, cfg.number_of_conv_states = [2], 3
  cfg.rope_parameters = {
      "rope_type": "default",
      "rope_theta": float(c.rope_max_timescale),
      "mrope_interleaved": True,
      "partial_rotary_factor": float(c.partial_rotary_factor),
      "mrope_section": list(c.mrope_section),
  }
  return cfg


class _OneHotPredictor:
  """Deterministic one-hot predictor for testing shift=True next-token vs current-token loss."""

  def __init__(self, vocab_size: int = 128, shift: int = 1):
    self.vocab_size = vocab_size
    self.shift = shift
    self.config = types.SimpleNamespace(num_experts=4, load_balance_loss_weight=0.001)
    self.mtp = None

  def __call__(self, tokens: jax.Array, *args: Any, return_aux: bool = False, **kwargs: Any) -> Any:
    del args, kwargs
    b, s = tokens.shape
    logits = np.full((b, s, self.vocab_size), -20.0, dtype=np.float32)
    toks = (np.asarray(tokens) + (10 if self.shift else 0)) % self.vocab_size
    np.put_along_axis(logits, toks[..., None], 20.0, axis=-1)
    aux = [{"router_probs": jnp.full((b, s, 4), 0.25), "router_indices": jnp.zeros((b, s, 2), dtype=jnp.int32)}]
    return (jnp.asarray(logits), aux) if return_aux else jnp.asarray(logits)


@pytest.mark.cpu_only
class MaskedPrefillDriverTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.cfg = _create_2cycle_mini_config(seq_len=16)
    self.model = modeling_qwen38_flash_next.Qwen38FlashNextModel(self.cfg, rngs=nnx.Rngs(params=0))
    if _HAS_QWEN4_EXP and modeling_qwen4_exp is not None and configuration_qwen4_exp is not None:
      self.pt_model = modeling_qwen4_exp.Qwen4ExpForCausalLM(_make_matching_hf_config(self.cfg)).eval()
    else:
      self.pt_model = None

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_generate_goldens_pytorch_reference_and_hooks(self):
    assert self.pt_model is not None and modeling_qwen4_exp is not None and configuration_qwen4_exp is not None
    g_path, toks = os.path.join(self.temp_dir, "g16.npz"), driver.build_byte_mod_test_tokens(16, 128)
    kw = {"seq_len": 16, "pt_model": self.pt_model, "tokens": toks}
    res = driver.run_generate_goldens(g_path, **kw, index_budget_blocks=2)
    self.assertTrue(res["passed"])
    self.assertEqual(res["routing_indices"].shape, (8, 1, 16, 2))
    self.assertEqual(res["indexer_selected_blocks"].shape, (2, 1, 16, 2))
    c_pt = self.pt_model.config
    cfg_ns = types.SimpleNamespace(text_config=c_pt, **vars(c_pt))
    delattr(cfg_ns, "indexer_compress_ratio")
    delattr(cfg_ns, "indexer_budget")
    with mock.patch.object(self.pt_model, "config", cfg_ns):
      res_comp = driver.run_generate_goldens(g_path, **kw)
    self.assertEqual(res_comp["indexer_selected_blocks"].shape, (2, 1, 16, 2))
    loaded = checker.load_golden_npz(g_path)
    np.testing.assert_array_equal(loaded["tokens"], toks)
    self.assertEqual(np.asarray(loaded["logits"]).shape, (1, 16, 128))
    for m, t in ((self.pt_model, None), (None, toks)):
      with self.assertRaises(ValueError):
        driver.run_generate_goldens(g_path, seq_len=16, pt_model=m, tokens=t)

  def test_token_encoding_and_parity_assertions(self):
    mock_tok = mock.MagicMock(encode=mock.MagicMock(return_value=[10, 20, 30, 40]))
    toks_hf, toks_max = driver.encode_golden_batch_hf(mock_tok, 8), driver.encode_golden_batch_maxtext(mock_tok, 8)
    self.assertEqual(toks_hf.shape, (1, 8))
    driver.assert_token_id_parity(toks_hf, toks_max)
    with self.assertRaisesRegex(ValueError, "Token shape mismatch"):
      driver.assert_token_id_parity(toks_hf, toks_hf[:, :4])
    with self.assertRaisesRegex(ValueError, "pytorch_token_ids != jax_token_ids"):
      driver.assert_token_id_parity(toks_hf, toks_hf + 1)

  def test_prefill_and_tinymmlu_and_tier1b_modes(self):
    g_path, tokens = os.path.join(self.temp_dir, "tiny.npz"), driver.build_byte_mod_test_tokens(16, 128)
    checker.save_golden_npz(g_path, **driver.extract_model_prefill_outputs(self.model, tokens))
    c1 = driver.run_prefill_logits_check(self.model, g_path, tokens, seq_len=16)
    c2 = driver.run_prefill_logits_check(self.model, g_path, tokens, seq_len=16, require_aux=False)
    c3 = driver.run_tinymmlu_check(self.model, g_path, tokens)
    self.assertTrue(c1["passed"] and c2["passed"] and c2["routing"].get("skipped"))
    self.assertTrue(c3["passed"] and abs(float(c3["loss_delta"])) < 1e-4)
    pert = modeling_qwen38_flash_next.Qwen38FlashNextModel(self.cfg, rngs=nnx.Rngs(params=42))
    self.assertFalse(driver.run_tinymmlu_check(pert, g_path, tokens, tol_nats=1e-5)["passed"])

  def test_holdout_ce_one_hot_next_token_shift_contract(self):
    tokens = np.arange(10, 90, 10, dtype=np.int64)[None, :]
    p_next, p_curr = _OneHotPredictor(128, 1), _OneHotPredictor(128, 0)
    for p_post, p_pre, s in [(p_next, p_curr, True), (p_curr, p_next, False)]:
      res = driver.run_holdout_ce_check(p_post, p_pre, tokens=tokens, shift=s)
      self.assertTrue(res["passed"] and float(res["ce_post_sft"]) < 1e-4 and float(res["ce_pre_sft"]) > 10.0)

  def test_run_cli_contract_and_checker_main_delegation(self):
    g_path, toks = os.path.join(self.temp_dir, "cli.npz"), driver.build_byte_mod_test_tokens(16, 128)
    checker.save_golden_npz(g_path, **driver.extract_model_prefill_outputs(self.model, toks))
    with self.assertRaisesRegex(ValueError, "--mode"):
      driver.run_cli([f"golden_tensor_path={g_path}"])
    for m in (
        "prefill_logits",
        "tier1b_logits",
        "tinymmlu",
        "holdout_ce",
    ):
      res = driver.run_cli([f"mode={m}", f"golden_tensor_path={g_path}"], lambda _: self.model, lambda p, s: toks)
      self.assertTrue(res.get("passed", True) or np.isfinite(float(res.get("total_loss", 0.0))))
    with mock.patch.object(driver, "run_cli", side_effect=[{"passed": True}, {"passed": False}]):
      self.assertEqual([checker.main(["mode=prefill_logits", f"golden_tensor_path={g_path}"]) for _ in (0, 1)], [0, 1])
    with self.assertRaisesRegex(ValueError, "Unsupported mode"):
      checker.main(["mode=unsupported_mode", f"golden_tensor_path={g_path}"])
    with self.assertRaisesRegex(ValueError, "golden_tensor_path is required"):
      checker.main(["mode=prefill_logits"])


if __name__ == "__main__":
  absltest.main()
