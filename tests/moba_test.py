"""
A unified test suite for MoBA correctness.

This test verifies the correctness of the JAX-based MoBA (Mixture of KV-Head
Bottlenecks) implementation by comparing both its final output and its
intermediate tensors against a reference PyTorch implementation.
"""
import unittest
import torch
import jax
import jax.numpy as jnp
import numpy as np
import math
import os
import sys

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers.attention_op import AttentionOp
from jax.sharding import Mesh
from MaxText import maxtext_utils

# pylint: disable=missing-function-docstring
class MobaTest(unittest.TestCase):
  """
  A unified test class for MoBA correctness, covering both end-to-end output
  and intermediate tensor comparisons.
  """

  def setUp(self):
    """
    Initializes common parameters, generates inputs, and pre-computes
    both JAX and PyTorch results for all subsequent tests.
    """
    self.batch = 1
    self.num_q_heads = 4
    self.num_kv_heads = 4
    self.seq_len = 2048
    self.head_dim = 256
    self.moba_chunk_size = 256
    self.moba_topk = 5
    self.dtype_torch = torch.bfloat16
    self.dtype_jax = jnp.bfloat16
    self.rtol = 1e-2
    self.atol = 1e-2

    # Generate deterministic, non-repeating inputs to prevent ties in gate
    # values, which can cause non-deterministic top-k behavior.
    total_q_elements = self.batch * self.seq_len * self.num_q_heads * self.head_dim
    total_kv_elements = self.batch * self.seq_len * self.num_kv_heads * self.head_dim

    q_flat = (np.arange(total_q_elements, dtype=np.float32) / total_q_elements) * 0.5
    k_flat = (np.arange(total_kv_elements, dtype=np.float32) / total_kv_elements) * 0.5

    self.q_np = q_flat.reshape(self.batch, self.seq_len, self.num_q_heads, self.head_dim)
    self.k_np = k_flat.reshape(self.batch, self.seq_len, self.num_kv_heads, self.head_dim)

    # The V tensor does not affect gate selection, so it can remain random.
    np.random.seed(42)
    self.v_np = np.random.randn(self.batch, self.seq_len, self.num_kv_heads, self.head_dim).astype(np.float32)

    # Pre-compute all results to avoid redundancy in test methods.
    self.torch_results = self._get_torch_results()
    self.jax_results = self._get_jax_results()

  def _get_torch_results(self):
    """
    Computes and returns both the final output and a dictionary of
    intermediate tensors from the reference PyTorch implementation.
    """
    q = torch.from_numpy(self.q_np).to(self.dtype_torch)
    k = torch.from_numpy(self.k_np).to(self.dtype_torch)
    v = torch.from_numpy(self.v_np).to(self.dtype_torch)

    q_ = q.squeeze(0)
    k_ = k.squeeze(0)
    v_ = v.squeeze(0)

    # Logic adapted from moba_naive.py to capture intermediates
    key_gate_weight_list = []
    batch_size = self.seq_len
    num_block = math.ceil(batch_size / self.moba_chunk_size)
    for block_idx in range(0, num_block):
      block_start = block_idx * self.moba_chunk_size
      block_end = min(batch_size, block_start + self.moba_chunk_size)
      key_gate_weight_list.append(k_[block_start:block_end].mean(dim=0, keepdim=True))
    key_gate_weight = torch.cat(key_gate_weight_list, dim=0)

    q_f32 = q_.type(torch.float32)
    key_gate_weight_f32 = key_gate_weight.type(torch.float32)
    gate = torch.einsum("shd,nhd->hsn", q_f32, key_gate_weight_f32)
    gate_before_masking = gate.clone()

    for i in range(num_block):
      gate[:, : (i + 1) * self.moba_chunk_size, i] = float("-inf")
      gate[:, i * self.moba_chunk_size : (i + 1) * self.moba_chunk_size, i] = float("inf")
    gate_after_masking = gate.clone()

    k_for_topk = min(self.moba_topk, num_block)
    gate_top_k_val, gate_top_k_idx = torch.topk(gate, k=k_for_topk, dim=-1, largest=True, sorted=False)
    gate_top_k_val_min_thresh, _ = gate_top_k_val.min(dim=-1)
    need_attend_threshold_mask = gate >= gate_top_k_val_min_thresh.unsqueeze(-1)

    gate_idx_mask = torch.zeros(need_attend_threshold_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=True)
    need_attend = torch.logical_and(need_attend_threshold_mask, gate_idx_mask)

    final_gate = gate_after_masking.clone()
    final_gate[need_attend] = 0
    final_gate[~need_attend] = -float("inf")
    final_gate = final_gate.repeat_interleave(self.moba_chunk_size, dim=-1)[:, :, :batch_size]
    final_gate.masked_fill_(torch.ones_like(final_gate, dtype=torch.bool).tril().logical_not(), -float("inf"))

    qk = torch.einsum("xhd,yhd->hxy", q_f32, k_.type(torch.float32))
    qk += final_gate
    softmax_scale = q.shape[-1] ** (-0.5)
    qk *= softmax_scale

    p = qk.softmax(dim=-1)
    output = torch.einsum("hxy,yhd->xhd", p, v_.type(torch.float32))
    output = output.unsqueeze(0).type_as(q)

    return {
        "output": output.to(torch.float32).numpy(),
        "key_gate_weight": key_gate_weight.to(torch.float32).numpy(),
        "gate_before_masking": gate_before_masking.to(torch.float32).numpy(),
        "gate_after_masking": gate_after_masking.to(torch.float32).numpy(),
        "gate_top_k_val": gate_top_k_val.to(torch.float32).numpy(),
        "gate_top_k_idx": gate_top_k_idx.numpy(),
        "gate_top_k_val_min": gate_top_k_val_min_thresh.unsqueeze(-1).to(torch.float32).numpy(),
        "need_attend_threshold_mask": need_attend_threshold_mask.numpy(),
        "gate_idx_mask": gate_idx_mask.numpy(),
        "need_attend": need_attend.numpy(),
    }

  def _get_jax_results(self):
    """
    Computes and returns both the final output and a dictionary of
    intermediate tensors from the JAX implementation.
    """
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        run_name="moba_test",
        enable_checkpointing=False,
        model_name="default",
        dtype="bfloat16",
        moba_naive=True,
        moba_chunk_size=self.moba_chunk_size,
        moba_topk=self.moba_topk,
        matmul_precision="highest",
    )

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    attention_op = AttentionOp(
        config=config,
        mesh=mesh,
        attention_kernel="dot_product",
        max_target_length=self.seq_len,
        num_query_heads=self.num_q_heads,
        num_kv_heads=self.num_kv_heads,
        float32_qk_product=True,
        float32_logits=True,
    )

    q_jax = jnp.asarray(self.q_np, dtype=self.dtype_jax)
    k_jax = jnp.asarray(self.k_np, dtype=self.dtype_jax)
    v_jax = jnp.asarray(self.v_np, dtype=self.dtype_jax)

    # Get intermediates by calling the internal logic directly for testing.
    q_positions = jnp.arange(self.seq_len)
    intermediates = attention_op._calculate_moba_gate_logic(q_jax[0], k_jax[0], q_positions)
    (
        key_gate_weight,
        gate_before,
        gate_after,
        gate_top_k_val,
        gate_top_k_idx,
        gate_top_k_val_min,
        need_attend_threshold_mask,
        gate_idx_mask,
        need_attend,
    ) = intermediates

    # Get final output
    scaling = self.head_dim ** (-0.5)
    q_jax_scaled = q_jax * scaling
    unnormalized_output, _, exponentials_sum = attention_op.apply_attention_dot(
        query=q_jax_scaled,
        key=k_jax,
        value=v_jax,
        decoder_segment_ids=None,
        model_mode="train",
        qk_product_einsum=jnp.einsum,
        wv_product_einsum=jnp.einsum,
    )
    output = unnormalized_output / exponentials_sum

    # Reshape intermediates to match PyTorch's [h, s, n] format
    k, g, s, n = gate_before.shape
    h = k * g
    _, _, _, n_k = gate_top_k_idx.shape

    return {
        "output": np.array(output),
        "key_gate_weight": np.array(key_gate_weight),
        "gate_before_masking": np.array(gate_before).reshape(h, s, n),
        "gate_after_masking": np.array(gate_after).reshape(h, s, n),
        "gate_top_k_val": np.array(gate_top_k_val).reshape(h, s, n_k),
        "gate_top_k_idx": np.array(gate_top_k_idx).reshape(h, s, n_k),
        "gate_top_k_val_min": np.array(gate_top_k_val_min).reshape(h, s, 1),
        "need_attend_threshold_mask": np.array(need_attend_threshold_mask).reshape(h, s, n),
        "gate_idx_mask": np.array(gate_idx_mask).reshape(h, s, n),
        "need_attend": np.array(need_attend).reshape(h, s, n),
    }

  def test_final_output_correctness(self):
    """Tests that the final output of JAX and PyTorch match."""
    np.testing.assert_allclose(
        self.torch_results["output"],
        self.jax_results["output"],
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Final outputs from PyTorch and JAX implementations do not match.",
    )

  def test_key_gate_weight(self):
    """Tests the 'key_gate_weight' intermediate tensor."""
    np.testing.assert_allclose(
        self.torch_results["key_gate_weight"],
        self.jax_results["key_gate_weight"],
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'key_gate_weight' does not match.",
    )

  def test_gate_before_masking(self):
    """Tests the 'gate' tensor before block-causal masking."""
    np.testing.assert_allclose(
        self.torch_results["gate_before_masking"],
        self.jax_results["gate_before_masking"],
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'gate' (before masking) does not match.",
    )

  def test_gate_after_masking(self):
    """Tests the 'gate' tensor after block-causal masking."""
    np.testing.assert_allclose(
        self.torch_results["gate_after_masking"],
        self.jax_results["gate_after_masking"],
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'gate' (after masking) does not match.",
    )

  def test_gate_top_k_val(self):
    """Tests the 'gate_top_k_val' intermediate tensor."""
    # Sort values for comparison as top-k order is not guaranteed.
    torch_val_sorted = np.sort(self.torch_results["gate_top_k_val"], axis=-1)
    jax_val_sorted = np.sort(self.jax_results["gate_top_k_val"], axis=-1)
    np.testing.assert_allclose(
        torch_val_sorted,
        jax_val_sorted,
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'gate_top_k_val' does not match.",
    )

  def _assert_masks_are_equivalent_despite_ties(self, torch_mask, jax_mask, err_msg):
    """
    Asserts that two boolean masks are equivalent, accounting for valid
    discrepancies that arise from different tie-breaking in top-k.
    A mismatch is only valid if the gate value at that position is equal
    to the threshold (the k-th smallest value in the top-k).
    """
    mismatched_indices = np.where(torch_mask != jax_mask)
    if mismatched_indices[0].size == 0:
      return  # Masks are identical

    torch_gate = self.torch_results["gate_after_masking"]
    torch_min_val = self.torch_results["gate_top_k_val_min"]

    h_coords, s_coords, n_coords = mismatched_indices
    mismatched_gate_values = torch_gate[mismatched_indices]
    mismatched_thresholds = torch_min_val[h_coords, s_coords, 0]

    is_tie = np.isclose(mismatched_gate_values, mismatched_thresholds, atol=self.atol, rtol=self.rtol)
    are_both_inf = (mismatched_gate_values == -np.inf) & (mismatched_thresholds == -np.inf)
    is_valid_mismatch = np.logical_or(is_tie, are_both_inf)

    if not np.all(is_valid_mismatch):
      first_bad_idx = np.where(~is_valid_mismatch)[0][0]
      bad_index = (h_coords[first_bad_idx], s_coords[first_bad_idx], n_coords[first_bad_idx])
      bad_gate_val = mismatched_gate_values[first_bad_idx]
      bad_threshold = mismatched_thresholds[first_bad_idx]
      self.fail(
          f"{err_msg}\n"
          f"Mismatch at index {bad_index} is not due to a tie at the threshold.\n"
          f"Gate value was {bad_gate_val}, but threshold was {bad_threshold}."
      )

  def test_need_attend_mask(self):
    """Tests the final 'need_attend' boolean mask, accounting for tie-breaks."""
    self._assert_masks_are_equivalent_despite_ties(
        self.torch_results["need_attend"],
        self.jax_results["need_attend"],
        "Final 'need_attend' mask has a critical mismatch.",
    )

if __name__ == "__main__":
  unittest.main()
