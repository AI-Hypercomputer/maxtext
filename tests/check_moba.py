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

# Define test configurations
TEST_CONFIGS = [
    {
        "name": "Standard",
        "seq_len": 2048,
        "moba_chunk_size": 256,
        "moba_topk": 4,
    },
    {
        "name": "Non-divisible_SeqLen",
        "seq_len": 2000,
        "moba_chunk_size": 256,
        "moba_topk": 4,
    },
    {
        "name": "TopK_greater_than_blocks",
        "seq_len": 1024,
        "moba_chunk_size": 256,
        "moba_topk": 8,  # 8 > 1024/256=4
    },
    {
        "name": "Small_and_quick",
        "seq_len": 512,
        "moba_chunk_size": 128,
        "moba_topk": 2,
    },
]


class MobaTest(unittest.TestCase):
  """
  A unified, parameterized test class for MoBA correctness.
  """

  def _run_and_get_results(self, config):
    """
    Generates inputs and computes results for both JAX and PyTorch
    based on a given configuration.
    """
    # Unpack config
    seq_len = config["seq_len"]
    moba_chunk_size = config["moba_chunk_size"]
    moba_topk = config["moba_topk"]

    # Common parameters
    batch = 1
    num_q_heads = 4
    num_kv_heads = 4
    head_dim = 256
    dtype_torch = torch.bfloat16
    dtype_jax = jnp.bfloat16

    # Generate deterministic inputs
    total_q_elements = batch * seq_len * num_q_heads * head_dim
    total_kv_elements = batch * seq_len * num_kv_heads * head_dim
    q_flat = (np.arange(total_q_elements, dtype=np.float32) / total_q_elements) * 0.5
    k_flat = (np.arange(total_kv_elements, dtype=np.float32) / total_kv_elements) * 0.5
    q_np = q_flat.reshape(batch, seq_len, num_q_heads, head_dim)
    k_np = k_flat.reshape(batch, seq_len, num_kv_heads, head_dim)
    np.random.seed(42)
    v_np = np.random.randn(batch, seq_len, num_kv_heads, head_dim).astype(np.float32)

    # Get PyTorch results
    torch_results = self._get_torch_results(
        q_np, k_np, v_np, seq_len, moba_chunk_size, moba_topk, dtype_torch
    )

    # Get JAX results
    jax_results = self._get_jax_results(
        q_np, k_np, v_np, seq_len, moba_chunk_size, moba_topk, num_q_heads, num_kv_heads, head_dim, dtype_jax
    )

    return torch_results, jax_results

  def _get_torch_results(self, q_np, k_np, v_np, seq_len, moba_chunk_size, moba_topk, dtype_torch):
    """Computes results from the reference PyTorch implementation."""
    q = torch.from_numpy(q_np).to(dtype_torch)
    k = torch.from_numpy(k_np).to(dtype_torch)
    v = torch.from_numpy(v_np).to(dtype_torch)
    q_, k_, v_ = q.squeeze(0), k.squeeze(0), v.squeeze(0)

    key_gate_weight_list = []
    num_block = math.ceil(seq_len / moba_chunk_size)
    for block_idx in range(num_block):
      start, end = block_idx * moba_chunk_size, min(seq_len, (block_idx + 1) * moba_chunk_size)
      key_gate_weight_list.append(k_[start:end].mean(dim=0, keepdim=True))
    key_gate_weight = torch.cat(key_gate_weight_list, dim=0)

    q_f32 = q_.type(torch.float32)
    gate = torch.einsum("shd,nhd->hsn", q_f32, key_gate_weight.type(torch.float32))
    gate_before_masking = gate.clone()

    for i in range(num_block):
      gate[:, : (i + 1) * moba_chunk_size, i] = float("-inf")
      gate[:, i * moba_chunk_size : (i + 1) * moba_chunk_size, i] = float("inf")
    gate_after_masking = gate.clone()

    k_for_topk = min(moba_topk, num_block)
    gate_top_k_val, gate_top_k_idx = torch.topk(gate, k=k_for_topk, dim=-1, largest=True, sorted=False)
    gate_top_k_val_min_thresh, _ = gate_top_k_val.min(dim=-1)
    need_attend_threshold_mask = gate >= gate_top_k_val_min_thresh.unsqueeze(-1)

    gate_idx_mask = torch.zeros_like(need_attend_threshold_mask, dtype=torch.bool, device=q.device)
    gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=True)
    need_attend = torch.logical_and(need_attend_threshold_mask, gate_idx_mask)

    final_gate = gate_after_masking.clone()
    final_gate[need_attend] = 0
    final_gate[~need_attend] = -float("inf")
    final_gate = final_gate.repeat_interleave(moba_chunk_size, dim=-1)[:, :, :seq_len]
    final_gate.masked_fill_(torch.ones_like(final_gate, dtype=torch.bool).tril().logical_not(), -float("inf"))

    qk = torch.einsum("xhd,yhd->hxy", q_f32, k_.type(torch.float32)) + final_gate
    qk *= q.shape[-1] ** -0.5
    p = qk.softmax(dim=-1)
    output = torch.einsum("hxy,yhd->xhd", p, v_.type(torch.float32)).unsqueeze(0).type_as(q)

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

  def _get_jax_results(
      self,
      q_np,
      k_np,
      v_np,
      seq_len,
      moba_chunk_size,
      moba_topk,
      num_q_heads,
      num_kv_heads,
      head_dim,
      dtype_jax
  ):
    """Computes results from the JAX implementation."""
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        run_name="moba_test",
        enable_checkpointing=False,
        model_name="default",
        dtype="bfloat16",
        moba_naive=True,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
        matmul_precision="highest",
    )
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    attention_op = AttentionOp(
        config, mesh, "dot_product", seq_len, num_q_heads, num_kv_heads, float32_qk_product=True, float32_logits=True
    )

    q_jax, k_jax, v_jax = map(lambda x: jnp.asarray(x, dtype=dtype_jax), (q_np, k_np, v_np))

    q_positions = jnp.arange(seq_len)
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

    unnormalized_output, _, exponentials_sum = attention_op.apply_attention_dot(
        q_jax * (head_dim**-0.5), k_jax, v_jax, None, "train", qk_product_einsum=jnp.einsum, wv_product_einsum=jnp.einsum
    )
    output = unnormalized_output / exponentials_sum

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

  def test_all_configurations(self):
    """Iterates through all test configs and runs all assertion types."""
    for config in TEST_CONFIGS:
      with self.subTest(config=config["name"]):
        torch_res, jax_res = self._run_and_get_results(config)

        # Test final output
        np.testing.assert_allclose(
            torch_res["output"], jax_res["output"], atol=1e-2, rtol=1e-2, err_msg="Final outputs do not match."
        )

        # Test intermediate numerical tensors
        for key in ["key_gate_weight", "gate_before_masking", "gate_after_masking", "gate_top_k_val_min"]:
          np.testing.assert_allclose(
              torch_res[key], jax_res[key], atol=1e-2, rtol=1e-2, err_msg=f"Intermediate tensor '{key}' does not match."
          )

        # Test top_k values (sorted, as order isn't guaranteed)
        np.testing.assert_allclose(
            np.sort(torch_res["gate_top_k_val"], axis=-1),
            np.sort(jax_res["gate_top_k_val"], axis=-1),
            atol=1e-2,
            rtol=1e-2,
            err_msg="Intermediate tensor 'gate_top_k_val' does not match.",
        )

        # Test boolean masks with tie-breaking logic
        for key in ["need_attend_threshold_mask", "gate_idx_mask", "need_attend"]:
          self._assert_masks_equivalent(
              torch_res[key], jax_res[key], torch_res["gate_after_masking"], jax_res["gate_top_k_val_min"], key
          )

        # Test top_k indices with tie-breaking logic
        self._assert_indices_equivalent(
            torch_res["gate_top_k_idx"],
            jax_res["gate_top_k_idx"],
            torch_res["gate_after_masking"],
            jax_res["gate_top_k_val_min"],
        )

  def _assert_masks_equivalent(self, torch_mask, jax_mask, torch_gate, jax_min_val, mask_name):
    """Asserts boolean masks are equivalent, accounting for tie-breaking."""
    mismatched_indices = np.where(torch_mask != jax_mask)
    if mismatched_indices[0].size == 0:
      return

    h_coords, s_coords, n_coords = mismatched_indices
    mismatched_gate_values = torch_gate[mismatched_indices]
    mismatched_thresholds = jax_min_val[h_coords, s_coords, 0]

    is_tie = np.isclose(mismatched_gate_values, mismatched_thresholds, atol=1e-2, rtol=1e-2)
    are_both_inf = (mismatched_gate_values == -np.inf) & (mismatched_thresholds == -np.inf)
    is_valid_mismatch = np.logical_or(is_tie, are_both_inf)

    if not np.all(is_valid_mismatch):
      bad_idx = np.where(~is_valid_mismatch)[0][0]
      bad_index = (h_coords[bad_idx], s_coords[bad_idx], n_coords[bad_idx])
      self.fail(
          f"Mask '{mask_name}' mismatch at index {bad_index} is not due to a tie at the threshold. "
          f"Gate value was {mismatched_gate_values[bad_idx]}, but threshold was {mismatched_thresholds[bad_idx]}."
      )

  def _assert_indices_equivalent(self, torch_idx, jax_idx, torch_gate, jax_min_val):
    """Asserts top-k indices are equivalent, accounting for tie-breaking."""
    h, s, _ = torch_idx.shape
    for h_idx in range(h):
      for s_idx in range(s):
        torch_indices_set = set(torch_idx[h_idx, s_idx, :])
        jax_indices_set = set(jax_idx[h_idx, s_idx, :])

        if torch_indices_set == jax_indices_set:
          continue

        differing_indices = torch_indices_set.symmetric_difference(jax_indices_set)
        threshold = jax_min_val[h_idx, s_idx, 0]

        for block_idx in differing_indices:
          gate_val = torch_gate[h_idx, s_idx, block_idx]
          is_close = np.isclose(gate_val, threshold, atol=1e-2, rtol=1e-2)
          is_inf = gate_val == -np.inf and threshold == -np.inf
          if not (is_close or is_inf):
            self.fail(
                f"Intermediate tensor 'gate_top_k_idx' has a critical mismatch.\n"
                f"Mismatch for query at (head={h_idx}, seq_pos={s_idx}).\n"
                f"Index sets were {torch_indices_set} (torch) and {jax_indices_set} (jax).\n"
                f"Differing block index {block_idx} has gate value {gate_val}, "
                f"which is not equal to the threshold {threshold}."
            )


if __name__ == "__main__":
  unittest.main()
  