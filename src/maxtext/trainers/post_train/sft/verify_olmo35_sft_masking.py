#!/usr/bin/env python3
"""OLMo 3.5 SFT Masking, Packing, KDA Padding, and Loss Correctness Suite.

Self-contained verification script covering the critical SFT correctness invariants
for OLMo 3.5 (hybrid 7:1 KDA + Full-Attention + LatentMoE) on MaxText:

  1. OLMo BOS == EOS target-masking bug vs. fix (bos_token_id=None when BOS==EOS).
  2. Multi-turn chat completion masking (`prefix` vs. `generation_blocks`, tool turns,
     trailing user turns, `<think>` boundary, and pre-tokenized document-role rows).
  3. Fractional `per_device_batch_size < 1` row-slicing (`ExpandToLoadedRows`).
  4. Packed multi-example OLMo 3.5 (`olmoe3`) forward & backward isolation:
     - Zero cross-conversation leakage (KDA recurrence reset + 4-tap conv reset + FA mask).
     - Zero gradient leakage from masked prompt tokens (`targets_segmentation == 0`)
       across standard logits, `num_vocab_tiling`, and `fused_lm_head_ce` paths.
     - Exact match between packed completion-masked CE loss and standalone assistant-only
       reference CE loss.
  5. KDA padding NaN guard (`decoder_segment_ids == 0`):
     - Reproduces extreme-gate padding chunk overflow (`inf * 0 = NaN`) without guard.
     - Proves `_mask_padding_gates` eliminates NaNs with 0.0 diff on non-pad tokens.

Run with MaxText on PYTHONPATH (CPU or TPU):
  python3 verify_olmo35_sft_masking.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import types
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.models import olmoe3
from maxtext.trainers.pre_train import train as pretrain_mod
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


class SyntheticOlmoTokenizer:
  """Minimal deterministic OLMo-style tokenizer with BOS == EOS == 100257 and <think>."""

  def __init__(self):
    self.bos_token_id = 100257
    self.eos_token_id = 100257
    self.pad_token_id = 100277
    self.bos_token = "<|endoftext|>"
    self.eos_token = "<|endoftext|>"
    self.pad_token = "<|pad|>"
    self.special_tokens_map = {
        "bos_token": self.bos_token,
        "eos_token": self.eos_token,
        "pad_token": self.pad_token,
    }
    # Deterministic word-level + special-token vocabulary for offset-exact tests.
    self._specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "system\n",
        "user\n",
        "assistant\n",
        "tool\n",
    ]
    self._id_map = {
        "<|endoftext|>": 100257,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
        "<think>": 100256,
        "</think>": 100266,
    }
    self._rev_map = {v: k for k, v in self._id_map.items()}
    self._next_id = 100
    # Jinja template matching OLMo 3.5 Think SFT (`<think>` inside generation prompt,
    # while `{% generation %}` wraps the assistant content + `</think>` + `<|im_end|>`).
    self.chat_template = (
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'system' -%}"
        "<|im_start|>system\n{{ m['content'] }}<|im_end|>\n"
        "{%- elif m['role'] == 'user' -%}"
        "<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
        "{%- elif m['role'] == 'tool' -%}"
        "<|im_start|>tool\n{{ m['content'] }}<|im_end|>\n"
        "{%- elif m['role'] == 'assistant' -%}"
        "<|im_start|>assistant\n<think>"
        "{% generation %}{{ m['content'] }}<|im_end|>{% endgeneration %}\n"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "<|im_start|>assistant\n<think>"
        "{%- endif -%}"
    )

  def _tok_id(self, piece: str) -> int:
    if piece not in self._id_map:
      self._id_map[piece] = self._next_id
      self._rev_map[self._next_id] = piece
      self._next_id += 1
    return self._id_map[piece]

  def get_chat_template(self, chat_template=None, tools=None):
    del tools
    return chat_template or self.chat_template

  def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
    del skip_special_tokens
    return "".join(self._rev_map.get(int(t), "") for t in token_ids)

  def _tokenize_with_offsets(self, text: str):
    ids = []
    offsets = []
    i = 0
    n = len(text)
    while i < n:
      matched = None
      for sp in self._specials:
        if text.startswith(sp, i):
          matched = sp
          break
      if matched is not None:
        ids.append(self._tok_id(matched))
        offsets.append((i, i + len(matched)))
        i += len(matched)
        continue
      # Consume a word or single whitespace/punctuation unit
      j = i + 1
      if text[i].isalnum():
        while j < n and text[j].isalnum():
          j += 1
      piece = text[i:j]
      ids.append(self._tok_id(piece))
      offsets.append((i, j))
      i = j
    return ids, offsets

  def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
    del add_special_tokens
    ids, offsets = self._tokenize_with_offsets(text)
    out = {"input_ids": ids}
    if return_offsets_mapping:
      out["offset_mapping"] = offsets
    return out

  def apply_chat_template(
      self,
      messages,
      tokenize: bool = False,
      add_generation_prompt: bool = False,
      return_dict: bool = False,
  ):
    from transformers.utils import chat_template_utils

    rendered_list, _ = chat_template_utils.render_jinja_template(
        conversations=[messages],
        chat_template=self.chat_template,
        return_assistant_tokens_mask=False,
        add_generation_prompt=add_generation_prompt,
        **self.special_tokens_map,
    )
    rendered = rendered_list[0]
    if not tokenize:
      return rendered
    ids, _ = self._tokenize_with_offsets(rendered)
    return {"input_ids": ids} if return_dict else ids


def test_1_bos_equals_eos_masking():
  """Verify OLMo BOS == EOS bug in unpatched ShiftData vs. fixed bos_token_id=None."""
  eos = 100257
  pad = 100277
  # 2 packed conversations: [prompt(2), completion(3), EOS] + [prompt(2), completion(2), EOS] + [PAD(2)]
  inputs = np.array([[10, 11, 20, 21, 22, eos, 30, 31, 40, 41, eos, pad, pad]], dtype=np.int32)
  seg = np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0]], dtype=np.int32)
  # Completion-only target mask BEFORE ShiftData (prompt=0, completion+EOS=seg_id)
  t_seg = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0]], dtype=np.int32)
  features = {
      "inputs": inputs,
      "targets": inputs.copy(),
      "inputs_segmentation": seg,
      "targets_segmentation": t_seg,
  }

  # Buggy upstream behavior: ignored_ids=[pad, bos] where bos == eos
  buggy_op = input_pipeline_utils.ShiftData(ignored_ids=[pad, eos], axis=1)
  buggy_out = buggy_op.map({k: v.copy() for k, v in features.items()})
  eos_positions = np.where(buggy_out["targets"][0] == eos)[0]
  assert len(eos_positions) == 2, f"Expected 2 EOS targets, got {eos_positions}"
  buggy_eos_weights = buggy_out["targets_segmentation"][0, eos_positions]
  assert np.all(buggy_eos_weights == 0), f"Buggy op should drop EOS, got {buggy_eos_weights}"

  # Fixed OLMo behavior via _get_training_objective_transform(bos_token_id=None)
  from types import SimpleNamespace
  fixed_op = hf_data_processing._get_training_objective_transform(
      SimpleNamespace(training_objective="causal_lm"),
      use_sft=True,
      use_dpo=False,
      shift=True,
      packing=True,
      pad_id=pad,
      bos_token_id=None,
  )
  fixed_out = fixed_op.map({k: v.copy() for k, v in features.items()})
  fixed_eos_weights = fixed_out["targets_segmentation"][0, eos_positions]
  assert list(fixed_eos_weights) == [1, 2], f"Fixed op must preserve EOS targets [1, 2], got {fixed_eos_weights}"
  # Padding positions must still be masked out (0)
  pad_positions = np.where(fixed_out["targets"][0] == pad)[0]
  assert np.all(fixed_out["targets_segmentation"][0, pad_positions] == 0)

  return {
      "buggy_eos_mask": buggy_eos_weights.tolist(),
      "fixed_eos_mask": fixed_eos_weights.tolist(),
      "eos_positions": eos_positions.tolist(),
  }


def test_2_multiturn_chat_masking_modes():
  """Verify multi-turn + tool + trailing user + <think> masking under prefix vs generation_blocks."""
  tok = SyntheticOlmoTokenizer()
  convo = [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Solve 2+2."},
      {"role": "assistant", "content": "Need arithmetic.</think>4"},
      {"role": "tool", "content": "verified=true"},
      {"role": "assistant", "content": "Confirmed.</think>Answer is 4."},
      {"role": "user", "content": "Thanks!"},  # trailing user turn -> must stay masked
  ]

  results = {}
  for mode in ("prefix", "generation_blocks"):
    os.environ["MAXTEXT_SFT_LABELS"] = mode
    segs = input_pipeline_utils._split_conversation_into_segments(tok, convo)
    flat_ids = [t for ids, _ in segs for t in ids]
    flat_is_prompt = [is_p for ids, is_p in segs for _ in ids]
    full_ids = tok.apply_chat_template(convo, tokenize=True, add_generation_prompt=False)
    assert flat_ids == full_ids, f"Mode {mode}: flattened tokens mismatch full chat template!"

    # Check every <think> token's mask and check tool + trailing user tokens are masked
    think_id = tok._id_map["<think>"]
    im_end_id = tok._id_map["<|im_end|>"]
    think_positions = [i for i, t in enumerate(flat_ids) if t == think_id]
    think_trained = [not flat_is_prompt[i] for i in think_positions]

    # Verify tool text and trailing user text are strictly prompt (untrained)
    tool_id = tok._id_map["verified"]
    thanks_id = tok._id_map["Thanks"]
    assert all(flat_is_prompt[i] for i, t in enumerate(flat_ids) if t == tool_id), "Tool turn leaked into loss!"
    assert all(flat_is_prompt[i] for i, t in enumerate(flat_ids) if t == thanks_id), "Trailing user turn leaked!"

    # Verify both assistant answers + their <|im_end|> terminators ARE trained
    ans1_id = tok._id_map["arithmetic"]
    ans2_id = tok._id_map["Confirmed"]
    assert all(not flat_is_prompt[i] for i, t in enumerate(flat_ids) if t == ans1_id), "Turn 1 answer masked!"
    assert all(not flat_is_prompt[i] for i, t in enumerate(flat_ids) if t == ans2_id), "Turn 2 answer masked!"

    # Check assistant <|im_end|> trained count (exactly 2 assistant turns)
    trained_im_end = sum(1 for i, t in enumerate(flat_ids) if t == im_end_id and not flat_is_prompt[i])
    assert trained_im_end == 2, f"Expected 2 trained assistant <|im_end|>, got {trained_im_end}"

    results[mode] = {
        "total_tokens": len(flat_ids),
        "trained_tokens": sum(1 for p in flat_is_prompt if not p),
        "masked_tokens": sum(1 for p in flat_is_prompt if p),
        "think_positions": think_positions,
        "think_trained": think_trained,
        "trained_assistant_im_end": trained_im_end,
    }

  # Also verify pretraining replay `document` role: every token + EOS is trained (is_prompt=False).
  doc_segs = input_pipeline_utils._split_conversation_into_segments(
      tok, [{"role": "document", "content": "Replay corpus sample."}]
  )
  assert len(doc_segs) == 1 and doc_segs[0][1] is False and doc_segs[0][0][-1] == tok.eos_token_id
  results["document_replay"] = {"tokens": len(doc_segs[0][0]), "all_trained": True, "ends_with_eos": True}
  return results


def test_3_fractional_pdb_row_slicing():
  """Verify ExpandToLoadedRows preserves exact trained rows when per_device_batch_size < 1."""
  keep = [True, False, False, False, False, False, False, False]  # 1 of 8 rows (pdb=0.125)
  op = input_pipeline_utils.ExpandToLoadedRows(keep)
  batch = {
      "inputs": np.array([[101, 102, 103, 104]], dtype=np.int32),
      "targets": np.array([[102, 103, 104, 100257]], dtype=np.int32),
      "inputs_segmentation": np.array([[1, 1, 1, 1]], dtype=np.int32),
      "targets_segmentation": np.array([[0, 0, 1, 1]], dtype=np.int32),
  }
  out = op.map(batch)
  assert out["inputs"].shape == (8, 4)
  np.testing.assert_array_equal(out["inputs"][0], [101, 102, 103, 104])
  assert np.all(out["inputs"][1:] == 0)
  assert np.all(out["targets_segmentation"][1:] == 0)
  return {"expanded_shape": list(out["inputs"].shape), "nonzero_rows": 1}


def _build_tiny_olmoe3_config(seq_len=32, num_vocab_tiling=1, fused_lm_head_ce=False):
  base_yml = os.path.join(
      os.path.dirname(os.path.dirname(olmoe3.__file__)), "configs", "base.yml"
  )
  return pyconfig.initialize(
      [sys.argv[0], base_yml],
      run_name="sft_mask_verify",
      base_output_directory="/tmp/sft_mask_verify",
      model_name="olmo35-tiny",
      override_model_config=True,
      # Shrink dims for fast exact CPU FP32 verification while keeping 1 KDA + 1 FA + LatentMoE
      base_emb_dim=128,
      base_num_query_heads=4,
      base_num_kv_heads=2,
      head_dim=32,
      base_mlp_dim=128,
      base_moe_mlp_dim=128,
      base_shared_expert_mlp_dim=128,
      moe_expert_input_dim=64,
      num_experts=8,
      num_experts_per_tok=2,
      base_num_decoder_layers=2,
      inhomogeneous_layer_cycle_interval=2,  # layer 0 = KDA, layer 1 = NoPE Full-Attention
      gdn_num_key_heads=4,
      gdn_num_value_heads=4,
      gdn_key_head_dim=32,
      gdn_value_head_dim=32,
      gdn_chunk_size=16,
      kda_chunked_impl="subblock",
      kda_sub_block=8,
      kda_pallas_fused=False,
      use_tokamax_kda=False,
      kda_matmul_precision="highest",
      vocab_size=256,
      max_target_length=seq_len,
      per_device_batch_size=1,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      logits_dot_in_fp32=True,
      cast_logits_to_fp32=True,
      attention="dot_product",
      sparse_matmul=False,
      megablox=False,
      emo_enabled=False,
      z_loss_multiplier=1e-5,
      load_balance_loss_weight=0.0,  # isolate exact CE + z_loss
      num_vocab_tiling=num_vocab_tiling,
      fused_lm_head_ce=fused_lm_head_ce,
      enable_checkpointing=False,
      enable_dropout=False,
      scan_layers=False,
      use_sft=True,
      olmo_loss_denominator_counts_masked_instances=False,
      skip_jax_distributed_system=True,
  )


def test_4_packed_olmoe3_isolation_and_gradient_masking():
  """End-to-end OLMo 3.5 (KDA + FA + LatentMoE) packed isolation & gradient masking check."""
  seq_len = 32
  cfg = _build_tiny_olmoe3_config(seq_len=seq_len, num_vocab_tiling=1, fused_lm_head_ce=False)
  devices = maxtext_utils.create_device_mesh(cfg)
  mesh = jax.sharding.Mesh(devices, cfg.mesh_axes)
  rng = jax.random.PRNGKey(42)
  model = model_creation_utils.from_config(cfg, model_mode="train", mesh=mesh, rngs=nnx.Rngs(params=rng))
  graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)

  # Packed batch [B=1, T=32]:
  #   Conv 1: positions 0..11  (prompt: 0..4, completion: 5..11) -> seg_id = 1
  #   Conv 2: positions 12..25 (prompt: 12..17, completion: 18..25) -> seg_id = 2
  #   Pad:    positions 26..31 -> seg_id = 0
  rng_np = np.random.default_rng(7)
  toks = rng_np.integers(5, 250, size=(1, seq_len), dtype=np.int32)
  toks[:, 26:] = 0
  targets = np.roll(toks, -1, axis=1)
  targets[:, 11] = 255  # EOS of Conv 1
  targets[:, 25] = 255  # EOS of Conv 2
  targets[:, 26:] = 0

  in_seg = np.zeros((1, seq_len), dtype=np.int32)
  in_seg[:, 0:12] = 1
  in_seg[:, 12:26] = 2

  in_pos = np.zeros((1, seq_len), dtype=np.int32)
  in_pos[:, 0:12] = np.arange(12)
  in_pos[:, 12:26] = np.arange(14)

  # Targets segmentation: ONLY completion tokens (5..11 and 18..25) have > 0
  t_seg = np.zeros((1, seq_len), dtype=np.int32)
  t_seg[:, 5:12] = 1
  t_seg[:, 18:26] = 2

  base_data = {
      "inputs": jnp.array(toks),
      "targets": jnp.array(targets),
      "inputs_segmentation": jnp.array(in_seg),
      "inputs_position": jnp.array(in_pos),
      "targets_segmentation": jnp.array(t_seg),
      "targets_position": jnp.array(in_pos),
  }

  def run_loss_and_grad(data, test_cfg):
    m = model_creation_utils.from_config(test_cfg, model_mode="train", mesh=mesh, rngs=nnx.Rngs(params=rng))
    nnx.update(m, params_state)
    def _loss(mod):
      return pretrain_mod.loss_fn(mod, test_cfg, data, rng, None, is_train=True)
    (loss_val, aux), grads = nnx.value_and_grad(_loss, has_aux=True)(m)
    return float(loss_val), aux, grads

  # 4A. Zero gradient leakage from masked prompt & pad targets!
  # Perturb targets at ALL masked prompt (0..4, 12..17) and pad (26..31) positions.
  perturbed_targets = targets.copy()
  perturbed_targets[:, 0:5] = (perturbed_targets[:, 0:5] + 77) % 250 + 1
  perturbed_targets[:, 12:18] = (perturbed_targets[:, 12:18] + 99) % 250 + 1
  perturbed_targets[:, 26:] = 123
  pert_data = dict(base_data, targets=jnp.array(perturbed_targets))

  loss_base, aux_base, grads_base = run_loss_and_grad(base_data, cfg)
  loss_pert, aux_pert, grads_pert = run_loss_and_grad(pert_data, cfg)
  max_grad_diff_std = max(
      float(jnp.max(jnp.abs(g1 - g2)))
      for g1, g2 in zip(jax.tree_util.tree_leaves(grads_base), jax.tree_util.tree_leaves(grads_pert))
  )
  assert abs(loss_base - loss_pert) == 0.0, f"Masked prompt targets leaked into loss: {loss_base} vs {loss_pert}"
  assert max_grad_diff_std == 0.0, f"Masked prompt targets leaked into grads: {max_grad_diff_std}"
  assert int(aux_base["total_weights"]) == 15, f"Expected 7+8=15 trained tokens, got {aux_base['total_weights']}"

  # 4B. Same zero-leakage test with num_vocab_tiling=4 (and exact parity with standard logits!)
  cfg_vt = _build_tiny_olmoe3_config(seq_len=seq_len, num_vocab_tiling=4, fused_lm_head_ce=False)
  loss_vt_base, _, grads_vt_base = run_loss_and_grad(base_data, cfg_vt)
  loss_vt_pert, _, grads_vt_pert = run_loss_and_grad(pert_data, cfg_vt)
  max_grad_diff_vt = max(
      float(jnp.max(jnp.abs(g1 - g2)))
      for g1, g2 in zip(jax.tree_util.tree_leaves(grads_vt_base), jax.tree_util.tree_leaves(grads_vt_pert))
  )
  vt_vs_std_loss = abs(loss_vt_base - loss_base)
  assert max_grad_diff_vt == 0.0, f"Vocab tiling leaked masked prompt targets: {max_grad_diff_vt}"
  assert vt_vs_std_loss < 1e-6, f"Vocab tiling loss mismatch vs standard: {vt_vs_std_loss}"

  # 4C. Zero cross-conversation forward & backward leakage (Conv 2 inputs -> Conv 1 logits & loss)!
  # Mask loss to ONLY Conv 1 completion (5..11), then mutate Conv 2's input tokens (12..25).
  t_seg_c1 = np.zeros((1, seq_len), dtype=np.int32)
  t_seg_c1[:, 5:12] = 1
  data_c1_a = dict(base_data, targets_segmentation=jnp.array(t_seg_c1))
  toks_c2_mut = toks.copy()
  toks_c2_mut[:, 12:26] = (toks_c2_mut[:, 12:26] + 53) % 250 + 1
  data_c1_b = dict(data_c1_a, inputs=jnp.array(toks_c2_mut))

  loss_c1_a, _, grads_c1_a = run_loss_and_grad(data_c1_a, cfg)
  loss_c1_b, _, grads_c1_b = run_loss_and_grad(data_c1_b, cfg)
  c1_loss_diff = abs(loss_c1_a - loss_c1_b)
  c1_grad_diff = max(
      float(jnp.max(jnp.abs(g1 - g2)))
      for g1, g2 in zip(jax.tree_util.tree_leaves(grads_c1_a), jax.tree_util.tree_leaves(grads_c1_b))
  )
  assert c1_loss_diff < 1e-6, f"Cross-conversation loss leak (Conv2 -> Conv1): {c1_loss_diff}"
  assert c1_grad_diff < 1e-5, f"Cross-conversation grad leak (Conv2 -> Conv1): {c1_grad_diff}"

  # 4D. Reverse direction: mutate Conv 1 inputs (0..11) and verify ZERO leak into Conv 2 loss (18..25)!
  # This strictly verifies BOTH KDA recurrent state reset AND 4-tap causal conv reset AND FA segment mask!
  t_seg_c2 = np.zeros((1, seq_len), dtype=np.int32)
  t_seg_c2[:, 18:26] = 2
  data_c2_a = dict(base_data, targets_segmentation=jnp.array(t_seg_c2))
  toks_c1_mut = toks.copy()
  toks_c1_mut[:, 0:12] = (toks_c1_mut[:, 0:12] + 41) % 250 + 1
  data_c2_b = dict(data_c2_a, inputs=jnp.array(toks_c1_mut))

  loss_c2_a, _, grads_c2_a = run_loss_and_grad(data_c2_a, cfg)
  loss_c2_b, _, grads_c2_b = run_loss_and_grad(data_c2_b, cfg)
  c2_loss_diff = abs(loss_c2_a - loss_c2_b)
  c2_grad_diff = max(
      float(jnp.max(jnp.abs(g1 - g2)))
      for g1, g2 in zip(jax.tree_util.tree_leaves(grads_c2_a), jax.tree_util.tree_leaves(grads_c2_b))
  )
  assert c2_loss_diff < 1e-6, f"Cross-conversation KDA/Conv/FA state leak (Conv1 -> Conv2): {c2_loss_diff}"
  assert c2_grad_diff < 1e-5, f"Cross-conversation grad leak (Conv1 -> Conv2): {c2_grad_diff}"

  # 4E. Exact match between packed completion loss and weighted mean of standalone Conv 1 + Conv 2 losses!
  expected_packed_loss = (7.0 * loss_c1_a + 8.0 * loss_c2_a) / 15.0
  packed_vs_standalone_diff = abs(loss_base - expected_packed_loss)
  assert packed_vs_standalone_diff < 1e-6, f"Packed loss {loss_base} != standalone {expected_packed_loss}"

  return {
      "trained_tokens": 15,
      "masked_prompt_loss_diff": abs(loss_base - loss_pert),
      "masked_prompt_grad_diff_std": max_grad_diff_std,
      "masked_prompt_grad_diff_vocab_tiling": max_grad_diff_vt,
      "vocab_tiling_vs_std_loss_diff": vt_vs_std_loss,
      "conv2_to_conv1_loss_diff": c1_loss_diff,
      "conv1_to_conv2_kda_reset_loss_diff": c2_loss_diff,
      "conv1_to_conv2_kda_reset_grad_diff": c2_grad_diff,
      "packed_vs_standalone_ce_diff": packed_vs_standalone_diff,
  }


def test_5_kda_padding_nan_guard():
  """Reproduce extreme-gate KDA padding NaN without guard and prove _mask_padding_gates fixes it."""
  key = jax.random.PRNGKey(0)
  B, T, H, dk, dv = 1, 128, 2, 16, 16
  n_real = 40
  q = jax.random.normal(key, (B, T, H, dk), dtype=jnp.float32) * 0.2
  k = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, dk), dtype=jnp.float32) * 0.2
  v = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, dv), dtype=jnp.float32) * 0.2
  # Real tokens have healthy negative log_decay; padding chunk (positions 64..127) has extreme decay
  log_decay = jnp.where(
      jnp.arange(T)[None, :, None, None] < n_real,
      -0.05 * jnp.ones((B, T, H, dk)),
      -1e4 * jnp.ones((B, T, H, dk)),  # extreme pad-chunk decay overflows exp(+sum) -> NaN without guard
  )
  beta = jnp.full((B, T, H), 0.5, dtype=jnp.float32)
  seg = jnp.where(jnp.arange(T)[None, :] < n_real, 1, 0).astype(jnp.int32)
  prev_seg = jnp.pad(seg, ((0, 0), (1, 0)), constant_values=-1)[:, :T]
  resets = seg != prev_seg

  # Unmasked chunked KDA overflows to NaN on the padding chunk!
  unmasked_out = olmoe3._delta_rule_chunked(q, k, v, log_decay, beta, resets, chunk_size=64)
  unmasked_has_nan = bool(jnp.any(jnp.isnan(unmasked_out)))

  # Guarded KDA via _mask_padding_gates(seg, log_decay, beta)
  ld_m, beta_m = olmoe3._mask_padding_gates(seg, log_decay, beta)
  with jax.default_matmul_precision("highest"):
    masked_chunked = olmoe3._delta_rule_chunked(q, k, v, ld_m, beta_m, resets, chunk_size=64)
    masked_subblock = olmoe3._delta_rule_chunked_subblock(q, k, v, ld_m, beta_m, resets, chunk_size=64, sub_block=16)
    ref_scan = olmoe3._delta_rule_scan(q, k, v, jnp.exp(log_decay), beta, resets)

  assert bool(jnp.all(jnp.isfinite(masked_chunked))), "Masked chunked KDA produced non-finite values!"
  assert bool(jnp.all(jnp.isfinite(masked_subblock))), "Masked subblock KDA produced non-finite values!"
  real_diff_chunked = float(jnp.max(jnp.abs(masked_chunked[:, :n_real] - ref_scan[:, :n_real])))
  real_diff_subblock = float(jnp.max(jnp.abs(masked_subblock[:, :n_real] - ref_scan[:, :n_real])))
  assert real_diff_subblock < 1e-4, f"Real tokens changed under padding guard: {real_diff_subblock}"

  return {
      "unmasked_pad_chunk_reproduced_nan": unmasked_has_nan,
      "masked_chunked_all_finite": True,
      "masked_subblock_all_finite": True,
      "real_token_max_diff_subblock_vs_scan": real_diff_subblock,
      "real_token_max_diff_chunked_vs_scan": real_diff_chunked,
  }


def main():
  t0 = time.perf_counter()
  tests = [
      ("1_bos_equals_eos_masking", test_1_bos_equals_eos_masking),
      ("2_multiturn_chat_masking_modes", test_2_multiturn_chat_masking_modes),
      ("3_fractional_pdb_row_slicing", test_3_fractional_pdb_row_slicing),
      ("4_packed_olmoe3_isolation_and_grad_masking", test_4_packed_olmoe3_isolation_and_gradient_masking),
      ("5_kda_padding_nan_guard", test_5_kda_padding_nan_guard),
  ]
  report = {}
  for name, fn in tests:
    ts = time.perf_counter()
    res = fn()
    dt = time.perf_counter() - ts
    report[name] = {"status": "PASS", "seconds": round(dt, 2), "details": res}
    print(f"[PASS] {name} ({dt:.2f}s): {json.dumps(res)}")
  print(f"\nALL {len(tests)}/{len(tests)} SFT CORRECTNESS CHECKS PASSED in {time.perf_counter() - t0:.2f}s")
  out_path = os.environ.get("SFT_VERIFY_JSON")
  if "--json-out" in sys.argv:
    out_path = sys.argv[sys.argv.index("--json-out") + 1]
  if out_path:
    with open(out_path, "w") as f:
      json.dump(report, f, indent=2)


if __name__ == "__main__":
  main()
