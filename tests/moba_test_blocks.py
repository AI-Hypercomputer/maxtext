"""
A test for MoBA correctness that compares intermediate tensors block by block.
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
class MobaBlockTest(unittest.TestCase):
  """
  Tests the correctness of the MoBA implementation by comparing intermediate
  tensors from JAX against a reference PyTorch implementation.
  """

  def setUp(self):
    self.batch = 1
    self.num_q_heads = 4
    self.num_kv_heads = 4
    self.seq_len = 2048
    self.head_dim = 128
    self.moba_chunk_size = 256
    self.moba_topk = 2
    self.dtype_torch = torch.bfloat16
    self.dtype_jax = jnp.bfloat16
    self.rtol = 1e-2
    self.atol = 1e-2

    # Generate random inputs using NumPy to ensure they are identical for both frameworks
    np.random.seed(1234)
    self.q_np = np.random.randn(self.batch, self.seq_len, self.num_q_heads, self.head_dim).astype(np.float32)
    self.k_np = np.random.randn(self.batch, self.seq_len, self.num_kv_heads, self.head_dim).astype(np.float32)

    # JAX setup
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
    self.attention_op = AttentionOp(
        config=config,
        mesh=mesh,
        attention_kernel="dot_product",
        max_target_length=self.seq_len,
        num_query_heads=self.num_q_heads,
        num_kv_heads=self.num_kv_heads,
        float32_qk_product=True,
        float32_logits=True,
    )

  def get_torch_intermediates(self):
    """Computes and returns intermediate tensors from the PyTorch implementation."""
    q = torch.from_numpy(self.q_np).to(self.dtype_torch)
    k = torch.from_numpy(self.k_np).to(self.dtype_torch)

    q_ = q.squeeze(0)
    k_ = k.squeeze(0)

    key_gate_weight = []
    batch_size = self.seq_len
    num_block = math.ceil(batch_size / self.moba_chunk_size)
    for block_idx in range(0, num_block):
      block_start = block_idx * self.moba_chunk_size
      block_end = min(batch_size, block_start + self.moba_chunk_size)
      key_gate_weight.append(k_[block_start:block_end].mean(dim=0, keepdim=True))
    key_gate_weight = torch.cat(key_gate_weight, dim=0)

    q_f32 = q_.type(torch.float32)
    key_gate_weight_f32 = key_gate_weight.type(torch.float32)
    gate = torch.einsum("shd,nhd->hsn", q_f32, key_gate_weight_f32)
    gate_before_masking = gate.clone()

    for i in range(num_block):
      gate[:, : (i + 1) * self.moba_chunk_size, i] = float("-inf")
      gate[:, i * self.moba_chunk_size : (i + 1) * self.moba_chunk_size, i] = float("inf")
    gate_after_masking = gate.clone()

    k_for_topk = min(self.moba_topk, num_block)
    if k_for_topk <= 0:
        need_attend = torch.zeros_like(gate, dtype=torch.bool)
        gate_top_k_val = torch.zeros_like(gate[..., :k_for_topk])
        gate_top_k_idx = torch.zeros_like(gate[..., :k_for_topk], dtype=torch.int64)
        gate_top_k_val_min = torch.zeros_like(gate[..., :1])
        need_attend_threshold_mask = torch.zeros_like(gate, dtype=torch.bool)
        gate_idx_mask = torch.zeros_like(gate, dtype=torch.bool)
    else:
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=k_for_topk, dim=-1, largest=True, sorted=False
        )
        gate_top_k_val_min, _ = gate_top_k_val.min(dim=-1, keepdim=True)
        need_attend_threshold_mask = gate >= gate_top_k_val_min

        gate_idx_mask = torch.zeros_like(gate, dtype=torch.bool).scatter_(
            -1, gate_top_k_idx, True
        )
        need_attend = torch.logical_and(need_attend_threshold_mask, gate_idx_mask)

    return (
        key_gate_weight,
        gate_before_masking,
        gate_after_masking,
        gate_top_k_val,
        gate_top_k_idx,
        gate_top_k_val_min,
        need_attend_threshold_mask,
        gate_idx_mask,
        need_attend,
    )

  def get_jax_intermediates(self):
    """Computes and returns intermediate tensors from the JAX implementation."""
    q_jax = jnp.asarray(self.q_np, dtype=self.dtype_jax)
    k_jax = jnp.asarray(self.k_np, dtype=self.dtype_jax)

    # The gate calculation should use the unscaled query
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
    ) = self.attention_op._debug_moba_intermediates(q_jax, k_jax)
    return (
        key_gate_weight,
        gate_before,
        gate_after,
        gate_top_k_val,
        gate_top_k_idx,
        gate_top_k_val_min,
        need_attend_threshold_mask,
        gate_idx_mask,
        need_attend,
    )

  def test_key_gate_weight(self):
    """Tests the 'key_gate_weight' intermediate tensor."""
    (
        torch_kgw,
        *_,
    ) = self.get_torch_intermediates()
    (
        jax_kgw,
        *_,
    ) = self.get_jax_intermediates()

    np.testing.assert_allclose(
        torch_kgw.to(torch.float32).numpy(),
        np.array(jax_kgw),
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'key_gate_weight' does not match.",
    )

  def test_gate_before_masking(self):
    """Tests the 'gate' tensor before block-causal masking."""
    (
        _,
        torch_gate,
        *_,
    ) = self.get_torch_intermediates()
    (
        _,
        jax_gate,
        *_,
    ) = self.get_jax_intermediates()

    # JAX output is (k,g,s,N), Torch is (h,s,n). Reshape JAX to match.
    # k=n_kv_heads, g=groups, s=seq_len, N=num_blocks
    # h=n_q_heads, s=seq_len, n=num_blocks
    k, g, s, n = jax_gate.shape
    h = k * g
    jax_gate_reshaped = jax_gate.reshape(h, s, n)

    np.testing.assert_allclose(
        torch_gate.to(torch.float32).numpy(),
        np.array(jax_gate_reshaped),
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'gate' (before masking) does not match.",
    )

  def test_gate_after_masking(self):
    """Tests the 'gate' tensor after block-causal masking."""
    (
        _,
        _,
        torch_gate,
        *_,
    ) = self.get_torch_intermediates()
    (
        _,
        _,
        jax_gate,
        *_,
    ) = self.get_jax_intermediates()

    k, g, s, n = jax_gate.shape
    h = k * g
    jax_gate_reshaped = jax_gate.reshape(h, s, n)

    np.testing.assert_allclose(
        torch_gate.to(torch.float32).numpy(),
        np.array(jax_gate_reshaped),
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'gate' (after masking) does not match.",
    )

  def test_gate_top_k_val(self):
    """Tests the 'gate_top_k_val' intermediate tensor."""
    (
        *_,
        torch_val,
        _,
        _,
        _,
        _,
        _,
    ) = self.get_torch_intermediates()
    (
        *_,
        jax_val,
        _,
        _,
        _,
        _,
        _,
    ) = self.get_jax_intermediates()

    k, g, s, n = jax_val.shape
    h = k * g
    jax_val_reshaped = jax_val.reshape(h, s, n)

    # Sort values for comparison as order is not guaranteed
    torch_val_sorted, _ = torch.sort(torch_val, dim=-1)
    jax_val_sorted = np.sort(np.array(jax_val_reshaped), axis=-1)

    np.testing.assert_allclose(
        torch_val_sorted.to(torch.float32).numpy(),
        jax_val_sorted,
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'gate_top_k_val' does not match.",
    )

  def test_gate_top_k_idx(self):
    """
    Tests the 'gate_top_k_idx' intermediate tensor, allowing for
    differences in indices if the corresponding gate value is -inf.
    """
    _, _, torch_gate, _, torch_idx, *_ = self.get_torch_intermediates()
    _, _, _, _, jax_idx, *_ = self.get_jax_intermediates()

    k, g, s, n_k = jax_idx.shape
    h = k * g
    jax_idx_reshaped = jax_idx.reshape(h, s, n_k)

    # Sort both index arrays to make them comparable
    torch_idx_sorted = torch.sort(torch_idx, dim=-1)[0].numpy()
    jax_idx_sorted = np.sort(np.array(jax_idx_reshaped), axis=-1)

    # Find where the sorted indices still don't match
    mismatched_indices = np.where(torch_idx_sorted != jax_idx_sorted)

    if mismatched_indices[0].size > 0:
      # Check the gate values at the first point of divergence
      h_idx, s_idx, _ = (idx[0] for idx in mismatched_indices)
      
      # Get the top-k indices for this specific query from both frameworks
      torch_indices_for_query = set(torch_idx[h_idx, s_idx, :].numpy())
      jax_indices_for_query = set(np.array(jax_idx_reshaped[h_idx, s_idx, :]).tolist())
      
      # Find the specific block indices that differ
      differing_indices = torch_indices_for_query.symmetric_difference(jax_indices_for_query)

      for block_idx in differing_indices:
        gate_val = torch_gate[h_idx, s_idx, block_idx].item()
        if gate_val != -np.inf:
          self.fail(
              "Intermediate tensor 'gate_top_k_idx' has a critical mismatch.\n" 
              f"Mismatch for query at (head={h_idx}, seq_pos={s_idx}). " 
              f"Differing block index was {block_idx}, but its gate value was {gate_val}, not -inf."
          )

  def test_gate_top_k_val_min(self):
    """Tests the 'gate_top_k_val_min' intermediate tensor."""
    (
        *_, 
        _, 
        _, 
        torch_min, 
        _, 
        _, 
        _,
    ) = self.get_torch_intermediates()
    (
        *_, 
        _, 
        _, 
        jax_min, 
        _, 
        _, 
        _,
    ) = self.get_jax_intermediates()

    k, g, s, n = jax_min.shape
    h = k * g
    jax_min_reshaped = jax_min.reshape(h, s, n)

    np.testing.assert_allclose(
        torch_min.to(torch.float32).numpy(),
        np.array(jax_min_reshaped),
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'gate_top_k_val_min' does not match.",
    )

  def test_need_attend_threshold_mask(self):
    """Tests the 'need_attend_threshold_mask' intermediate tensor."""
    (
        *_, 
        _, 
        _, 
        _, 
        torch_mask, 
        _, 
        _,
    ) = self.get_torch_intermediates()
    (
        *_, 
        _, 
        _, 
        _, 
        jax_mask, 
        _, 
        _,
    ) = self.get_jax_intermediates()

    k, g, s, n = jax_mask.shape
    h = k * g
    jax_mask_reshaped = jax_mask.reshape(h, s, n)

    np.testing.assert_array_equal(
        torch_mask.numpy(),
        np.array(jax_mask_reshaped),
        err_msg="Intermediate tensor 'need_attend_threshold_mask' does not match.",
    )

  def test_gate_idx_mask(self):
    """
    Tests the 'gate_idx_mask' intermediate tensor, ignoring mismatches
    where the gate value is -inf.
    """
    _, _, torch_gate, *_, torch_mask, _ = self.get_torch_intermediates()
    _, _, _, *_, jax_mask, _ = self.get_jax_intermediates()

    k, g, s, n = jax_mask.shape
    h = k * g
    jax_mask_reshaped = jax_mask.reshape(h, s, n)

    mismatched_indices = np.where(torch_mask.numpy() != np.array(jax_mask_reshaped))

    if mismatched_indices[0].size > 0:
      mismatched_gate_values = torch_gate[mismatched_indices].numpy()
      non_inf_mismatches = np.where(mismatched_gate_values != -np.inf)[0]

      if non_inf_mismatches.size > 0:
        first_bad_index = tuple(idx[non_inf_mismatches[0]] for idx in mismatched_indices)
        first_bad_gate_val = mismatched_gate_values[non_inf_mismatches[0]]
        self.fail(
            "Intermediate tensor 'gate_idx_mask' has a critical mismatch.\n" 
            f"Mismatch at index {first_bad_index} where gate value was {first_bad_gate_val}, not -inf."
        )

  def test_need_attend_mask(self):
    """
    Tests the final 'need_attend' boolean mask, ignoring mismatches
    where the gate value is -inf due to top-k implementation differences.
    """
    _, _, torch_gate, *_, torch_mask = self.get_torch_intermediates()
    _, _, _, *_, jax_mask = self.get_jax_intermediates()

    k, g, s, n = jax_mask.shape
    h = k * g
    jax_mask_reshaped = jax_mask.reshape(h, s, n)

    # Find where the masks disagree
    mismatched_indices = np.where(torch_mask.numpy() != np.array(jax_mask_reshaped))

    if mismatched_indices[0].size > 0:
      # For every mismatch, check if the gate value was -inf.
      # A mismatch is acceptable in this case because the choice between
      # different -inf blocks is arbitrary and doesn't affect the outcome.
      mismatched_gate_values = torch_gate[mismatched_indices].numpy()
      non_inf_mismatches = np.where(mismatched_gate_values != -np.inf)[0]

      if non_inf_mismatches.size > 0:
        first_bad_index = tuple(idx[non_inf_mismatches[0]] for idx in mismatched_indices)
        first_bad_gate_val = mismatched_gate_values[non_inf_mismatches[0]]
        self.fail(
            "Intermediate tensor 'need_attend' mask has a critical mismatch.\n" 
            f"Mismatch at index {first_bad_index} where gate value was {first_bad_gate_val}, not -inf."
        )


if __name__ == "__main__":
  unittest.main()
