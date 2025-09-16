"""
A test for MoBA correctness using a live PyTorch comparison.
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
  Tests the correctness of the MoBA implementation by comparing its output
  with a reference PyTorch implementation running live on CPU.
  """

  def setUp(self):
    self.batch = 1
    self.num_q_heads = 4
    self.num_kv_heads = 4
    self.seq_len = 1024
    self.head_dim = 128
    self.moba_chunk_size = 128
    self.moba_topk = 2
    self.dtype_torch = torch.bfloat16
    self.dtype_jax = jnp.bfloat16

    # Generate random inputs using NumPy to ensure they are identical for both frameworks
    np.random.seed(42)
    self.q_np = np.random.randn(self.batch, self.seq_len, self.num_q_heads, self.head_dim).astype(np.float32)
    self.k_np = np.random.randn(self.batch, self.seq_len, self.num_kv_heads, self.head_dim).astype(np.float32)
    self.v_np = np.random.randn(self.batch, self.seq_len, self.num_kv_heads, self.head_dim).astype(np.float32)

  def moba_attn_varlen_naive_torch(self, q, k, v, moba_chunk_size, moba_topk):
    """
    Reference MoBA implementation in PyTorch, adapted to run on CPU.
    """
    # This is a simplified version for a single batch item
    q_ = q.squeeze(0)
    k_ = k.squeeze(0)
    v_ = v.squeeze(0)

    key_gate_weight = []
    batch_size = self.seq_len
    num_block = math.ceil(batch_size / moba_chunk_size)
    for block_idx in range(0, num_block):
        block_start = block_idx * moba_chunk_size
        block_end = min(batch_size, block_start + moba_chunk_size)
        key_gate_weight.append(k_[block_start:block_end].mean(dim=0, keepdim=True))
    key_gate_weight = torch.cat(key_gate_weight, dim=0)

    q_ = q_.type(torch.float32)
    key_gate_weight = key_gate_weight.type(torch.float32)
    gate = torch.einsum("shd,nhd->hsn", q_, key_gate_weight)
    key_gate_weight = key_gate_weight.type_as(k)
    q_ = q_.type_as(k)

    for i in range(num_block):
        gate[:, : (i + 1) * moba_chunk_size, i] = float("-inf")
        gate[:, i * moba_chunk_size : (i + 1) * moba_chunk_size, i] = float("inf")

    gate_top_k_val, gate_top_k_idx = torch.topk(
        gate, k=min(moba_topk, num_block), dim=-1, largest=True, sorted=False
    )
    gate_top_k_val, _ = gate_top_k_val.min(dim=-1)
    need_attend = gate >= gate_top_k_val.unsqueeze(-1)

    gate_idx_mask = torch.zeros(need_attend.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=True)
    need_attend = torch.logical_and(need_attend, gate_idx_mask)

    gate[need_attend] = 0
    gate[~need_attend] = -float("inf")
    gate = gate.repeat_interleave(moba_chunk_size, dim=-1)[:, :, :batch_size]
    gate.masked_fill_(
        torch.ones_like(gate, dtype=torch.bool).tril().logical_not(), -float("inf")
    )

    q_ = q_.type(torch.float32)
    k_ = k_.type(torch.float32)
    v_ = v_.type(torch.float32)
    qk = torch.einsum("xhd,yhd->hxy", q_, k_)
    qk += gate
    softmax_scale = q.shape[-1] ** (-0.5)
    qk *= softmax_scale

    p = qk.softmax(dim=-1)
    o_ = torch.einsum("hxy,yhd->xhd", p, v_)
    return o_.unsqueeze(0).type_as(q)

  def test_moba_correctness(self):
    # 1. PyTorch execution (CPU)
    q_torch = torch.from_numpy(self.q_np).to(self.dtype_torch)
    k_torch = torch.from_numpy(self.k_np).to(self.dtype_torch)
    v_torch = torch.from_numpy(self.v_np).to(self.dtype_torch)

    output_torch = self.moba_attn_varlen_naive_torch(
        q_torch, k_torch, v_torch, self.moba_chunk_size, self.moba_topk
    )
    output_torch_np = output_torch.to(torch.float32).numpy()

    # 2. JAX execution
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

    # Manually apply scaling to the query to simulate the behavior of the parent Attention module.
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
    
    output_jax = unnormalized_output / exponentials_sum
    output_jax_np = np.array(output_jax)

    # 3. Comparison
    np.testing.assert_allclose(
        output_torch_np,
        output_jax_np,
        atol=1e-2,
        rtol=1e-2,
        err_msg="Outputs from PyTorch and JAX implementations do not match."
    )

if __name__ == '__main__':
  unittest.main()