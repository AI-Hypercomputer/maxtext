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
    np.random.seed(42)
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

    gate_top_k_val, gate_top_k_idx = torch.topk(
        gate, k=min(self.moba_topk, num_block), dim=-1, largest=True, sorted=False
    )
    gate_top_k_val, _ = gate_top_k_val.min(dim=-1)
    need_attend = gate >= gate_top_k_val.unsqueeze(-1)

    gate_idx_mask = torch.zeros(need_attend.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=True)
    need_attend = torch.logical_and(need_attend, gate_idx_mask)

    return key_gate_weight, gate_before_masking, gate_after_masking, need_attend

  def get_jax_intermediates(self):
    """Computes and returns intermediate tensors from the JAX implementation."""
    q_jax = jnp.asarray(self.q_np, dtype=self.dtype_jax)
    k_jax = jnp.asarray(self.k_np, dtype=self.dtype_jax)

    # The gate calculation should use the unscaled query
    key_gate_weight, gate_before, gate_after, need_attend = self.attention_op._debug_moba_intermediates(q_jax, k_jax)
    return key_gate_weight, gate_before, gate_after, need_attend

  def test_key_gate_weight(self):
    """Tests the 'key_gate_weight' intermediate tensor."""
    torch_kgw, _, _, _ = self.get_torch_intermediates()
    jax_kgw, _, _, _ = self.get_jax_intermediates()

    np.testing.assert_allclose(
        torch_kgw.to(torch.float32).numpy(),
        np.array(jax_kgw),
        atol=self.atol,
        rtol=self.rtol,
        err_msg="Intermediate tensor 'key_gate_weight' does not match.",
    )

  def test_gate_before_masking(self):
    """Tests the 'gate' tensor before block-causal masking."""
    _, torch_gate, _, _ = self.get_torch_intermediates()
    _, jax_gate, _, _ = self.get_jax_intermediates()

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
    _, _, torch_gate, _ = self.get_torch_intermediates()
    _, _, jax_gate, _ = self.get_jax_intermediates()

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

  def test_need_attend_mask(self):
    """Tests the final 'need_attend' boolean mask."""
    _, _, _, torch_mask = self.get_torch_intermediates()
    _, _, _, jax_mask = self.get_jax_intermediates()

    k, g, s, n = jax_mask.shape
    h = k * g
    jax_mask_reshaped = jax_mask.reshape(h, s, n)

    np.testing.assert_array_equal(
        torch_mask.numpy(),
        np.array(jax_mask_reshaped),
        err_msg="Intermediate tensor 'need_attend' mask does not match.",
    )


if __name__ == "__main__":
  unittest.main()
