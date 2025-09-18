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
    self.head_dim = 256
    self.moba_chunk_size = 256
    self.moba_topk = 4
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

    # The following logic is adapted from moba_naive.py to be nearly identical,
    # while capturing intermediate values for testing.

    # calc key gate weight
    key_gate_weight = []
    batch_size = self.seq_len
    num_block = math.ceil(batch_size / self.moba_chunk_size)
    for block_idx in range(0, num_block):
      block_start = block_idx * self.moba_chunk_size
      block_end = min(batch_size, block_start + self.moba_chunk_size)
      key_gate_weight.append(k_[block_start:block_end].mean(dim=0, keepdim=True))
    key_gate_weight = torch.cat(key_gate_weight, dim=0)  # [ N, H, D ]

    # calc & mask gate
    # use fp32 to avoid precision issue in bf16
    q_f32 = q_.type(torch.float32)
    key_gate_weight_f32 = key_gate_weight.type(torch.float32)
    gate = torch.einsum("shd,nhd->hsn", q_f32, key_gate_weight_f32)  # [ H, S, N ]
    gate_before_masking = gate.clone()

    for i in range(num_block):
      # select the future Qs that can attend to KV chunk i
      gate[:, : (i + 1) * self.moba_chunk_size, i] = float("-inf")
      gate[:, i * self.moba_chunk_size : (i + 1) * self.moba_chunk_size, i] = float("inf")
    gate_after_masking = gate.clone()

    k_for_topk = min(self.moba_topk, num_block)
    if k_for_topk <= 0:
        # Handle case where topk is 0 or negative
        need_attend = torch.zeros_like(gate, dtype=torch.bool)
        gate_top_k_val_for_test = torch.zeros_like(gate[..., :k_for_topk])
        gate_top_k_idx = torch.zeros_like(gate[..., :k_for_topk], dtype=torch.int64)
        gate_top_k_val_min_for_test = torch.zeros_like(gate[..., :1])
        need_attend_threshold_mask = torch.zeros_like(gate, dtype=torch.bool)
        gate_idx_mask = torch.zeros_like(gate, dtype=torch.bool)
    else:
        # gate_top_k_idx = gate_top_k_val = [ H S K ]
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=k_for_topk, dim=-1, largest=True, sorted=False
        )
        # Preserve the original top-k values for the test case
        gate_top_k_val_for_test = gate_top_k_val.clone()

        gate_top_k_val, _ = gate_top_k_val.min(dim=-1)  # [ H, S ]

        need_attend = gate >= gate_top_k_val.unsqueeze(-1)
        # Preserve the threshold mask for the test case
        need_attend_threshold_mask = need_attend.clone()

        # add gate_idx_mask in case of there is cornercases of same topk val been selected
        gate_idx_mask = torch.zeros(need_attend.shape, dtype=torch.bool, device=q.device)
        gate_idx_mask = gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=True)
        need_attend = torch.logical_and(need_attend, gate_idx_mask)

        # This is the threshold value used for the >= comparison
        gate_top_k_val_min_for_test = gate_top_k_val.unsqueeze(-1)

    return (
        key_gate_weight,
        gate_before_masking,
        gate_after_masking,
        gate_top_k_val_for_test,
        gate_top_k_idx,
        gate_top_k_val_min_for_test,
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
    Tests the 'gate_top_k_idx' intermediate tensor.
    This test accounts for differences in tie-breaking between JAX and PyTorch's
    top_k implementations. A mismatch in indices is considered acceptable only if
    the corresponding gate values are tied at the threshold (the k-th value).
    """
    (
        _,
        _,  # gate_before_masking
        torch_gate,  # gate_after_masking
        _,
        torch_idx,
        torch_min_val,  # The threshold value
        *_,
    ) = self.get_torch_intermediates()
    (
        _,
        _,
        _,
        _,
        jax_idx,
        jax_min_val,  # The threshold value
        *_,
    ) = self.get_jax_intermediates()

    k, g, s, n_k = jax_idx.shape
    h = k * g
    jax_idx_reshaped = jax_idx.reshape(h, s, n_k)

    # Reshape threshold tensors
    torch_min_val_reshaped = torch_min_val.numpy().reshape(h, s)

    # Iterate over every query's top-k result
    for h_idx in range(h):
      for s_idx in range(s):
        torch_indices_for_query = set(torch_idx[h_idx, s_idx, :].numpy())
        jax_indices_for_query = set(np.array(jax_idx_reshaped[h_idx, s_idx, :]).tolist())

        if torch_indices_for_query == jax_indices_for_query:
          continue

        # If the sets of indices differ, find the indices that are not in common
        differing_indices = torch_indices_for_query.symmetric_difference(jax_indices_for_query)

        threshold = torch_min_val_reshaped[h_idx, s_idx]

        # For each differing index, the gate value must be equal to the threshold
        for block_idx in differing_indices:
          gate_val = torch_gate[h_idx, s_idx, block_idx].item()

          is_close = np.isclose(gate_val, threshold, atol=self.atol, rtol=self.rtol)
          is_inf = gate_val == -np.inf and threshold == -np.inf

          if not (is_close or is_inf):
            self.fail(
                "Intermediate tensor 'gate_top_k_idx' has a critical mismatch.\n"
                f"Mismatch for query at (head={h_idx}, seq_pos={s_idx}).\n"
                f"Index sets were {torch_indices_for_query} (torch) and {jax_indices_for_query} (jax).\n"
                f"Differing block index {block_idx} has gate value {gate_val}, "
                f"which is not equal to the threshold {threshold}."
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
    Tests the 'gate_idx_mask' intermediate tensor.
    This test accounts for differences in tie-breaking between JAX and PyTorch's
    top_k implementations. A mismatch in the mask is considered acceptable only if
    the corresponding gate value is tied at the threshold (the k-th value).
    """
    (
        _,
        _,  # gate_before_masking
        torch_gate,  # gate_after_masking
        _,
        _,
        torch_min_val,  # The threshold value
        _,
        torch_mask,
        _,
    ) = self.get_torch_intermediates()
    (
        _,
        _,
        _,
        _,
        _,
        jax_min_val,  # The threshold value
        _,
        jax_mask,
        _,
    ) = self.get_jax_intermediates()

    k, g, s, n = jax_mask.shape
    h = k * g
    jax_mask_reshaped = jax_mask.reshape(h, s, n)

    mismatched_indices = np.where(torch_mask.numpy() != np.array(jax_mask_reshaped))

    if mismatched_indices[0].size > 0:
      h_coords, s_coords, _ = mismatched_indices
      mismatched_gate_values = torch_gate[mismatched_indices].numpy()

      # Get the thresholds for the mismatched positions
      mismatched_thresholds = torch_min_val.numpy()[h_coords, s_coords, 0]

      # A mismatch is only valid if the gate value is equal to the threshold
      is_tie = np.isclose(mismatched_gate_values, mismatched_thresholds, atol=self.atol, rtol=self.rtol)

      # Or if both are -inf (a tie between non-selectable blocks)
      are_both_inf = (mismatched_gate_values == -np.inf) & (mismatched_thresholds == -np.inf)

      is_valid_mismatch = np.logical_or(is_tie, are_both_inf)

      if not np.all(is_valid_mismatch):
        first_bad_mismatch_idx = np.where(~is_valid_mismatch)[0][0]

        bad_h = h_coords[first_bad_mismatch_idx]
        bad_s = s_coords[first_bad_mismatch_idx]
        bad_n = mismatched_indices[2][first_bad_mismatch_idx]

        bad_index = (bad_h, bad_s, bad_n)
        bad_gate_val = mismatched_gate_values[first_bad_mismatch_idx]
        bad_threshold = mismatched_thresholds[first_bad_mismatch_idx]

        self.fail(
            "Intermediate tensor 'gate_idx_mask' has a critical mismatch.\n"
            f"Mismatch at index {bad_index} is not due to a tie at the threshold.\n"
            f"Gate value was {bad_gate_val}, but threshold was {bad_threshold}."
        )

  def test_need_attend_mask(self):
    """
    Tests the final 'need_attend' boolean mask.
    This test accounts for differences in tie-breaking between JAX and PyTorch's
    top_k implementations. A mismatch in the mask is considered acceptable only if
    the corresponding gate value is tied at the threshold (the k-th value).
    """
    (
        _,
        _,  # gate_before_masking
        torch_gate,  # gate_after_masking
        _,
        _,
        torch_min_val,  # The threshold value
        _,
        _,
        torch_mask,
    ) = self.get_torch_intermediates()
    (
        _,
        _,
        _,
        _,
        _,
        jax_min_val,  # The threshold value
        _,
        _,
        jax_mask,
    ) = self.get_jax_intermediates()

    k, g, s, n = jax_mask.shape
    h = k * g
    jax_mask_reshaped = jax_mask.reshape(h, s, n)

    mismatched_indices = np.where(torch_mask.numpy() != np.array(jax_mask_reshaped))

    if mismatched_indices[0].size > 0:
      h_coords, s_coords, _ = mismatched_indices
      mismatched_gate_values = torch_gate[mismatched_indices].numpy()

      # Get the thresholds for the mismatched positions
      mismatched_thresholds = torch_min_val.numpy()[h_coords, s_coords, 0]

      # A mismatch is only valid if the gate value is equal to the threshold
      is_tie = np.isclose(mismatched_gate_values, mismatched_thresholds, atol=self.atol, rtol=self.rtol)

      # Or if both are -inf (a tie between non-selectable blocks)
      are_both_inf = (mismatched_gate_values == -np.inf) & (mismatched_thresholds == -np.inf)

      is_valid_mismatch = np.logical_or(is_tie, are_both_inf)

      if not np.all(is_valid_mismatch):
        first_bad_mismatch_idx = np.where(~is_valid_mismatch)[0][0]

        bad_h = h_coords[first_bad_mismatch_idx]
        bad_s = s_coords[first_bad_mismatch_idx]
        bad_n = mismatched_indices[2][first_bad_mismatch_idx]

        bad_index = (bad_h, bad_s, bad_n)
        bad_gate_val = mismatched_gate_values[first_bad_mismatch_idx]
        bad_threshold = mismatched_thresholds[first_bad_mismatch_idx]

        self.fail(
            "Intermediate tensor 'need_attend' mask has a critical mismatch.\n"
            f"Mismatch at index {bad_index} is not due to a tie at the threshold.\n"
            f"Gate value was {bad_gate_val}, but threshold was {bad_threshold}."
        )


if __name__ == "__main__":
  unittest.main()
