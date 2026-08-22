# Copyright 2026 Google LLC
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

"""Unit test for Kimi K3 logit parity: JAX (MaxText) vs PyTorch (HuggingFace).

This test validates that a 2-layer Kimi K3 model (Layer 0 KDA + Layer 1 MLA/MoE) in MaxText
produces logits that match a PyTorch reference implementation loading the exact same HuggingFace
weights (including MXFP4 dequantized MoE experts).
"""

import os
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
import orbax.checkpoint as ocp

from maxtext.configs import pyconfig
from maxtext.layers.nnx_wrappers import ToLinen
from maxtext.models.models import Transformer


# -----------------------------------------------------------------------------
# PyTorch Kimi K3 2-Layer Reference Model
# -----------------------------------------------------------------------------
E8M0_TABLE = torch.tensor([2.0**e if e < 128 else float('inf') for e in range(-127, 129)], dtype=torch.float32)
E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)

def dequantize_mxfp4_batch(weight_packed, weight_scale, is_wo=False):
    """Vectorized MXFP4 dequantization for all 896 experts in PyTorch (bfloat16)."""
    num_experts, out_features, in_bytes = weight_packed.shape
    in_features = in_bytes * 2

    w_low = weight_packed & 0x0F
    w_high = (weight_packed >> 4) & 0x0F
    w_indices = torch.stack([w_low, w_high], dim=-1).reshape(num_experts, out_features, in_features)

    w_fp = E2M1_TABLE[w_indices.long()].to(torch.bfloat16)
    scales = E8M0_TABLE[weight_scale.long()].to(torch.bfloat16)
    scales = scales.unsqueeze(-1).expand(-1, -1, -1, 32).reshape(num_experts, out_features, in_features)

    w_dequant = w_fp * scales
    w_transposed = w_dequant.transpose(1, 2)

    if is_wo:
        w_padded = F.pad(w_transposed, (0, 7168 - 3584, 0, 0), value=0.0)
    else:
        w_padded = F.pad(w_transposed, (0, 0, 0, 7168 - 3584), value=0.0)

    return w_padded


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def situ_activation(x, beta=4.0):
    return x * torch.sigmoid(beta * x)

def linear_beta_tanh_activation(x, beta=25.0):
    return x * torch.tanh(beta * x)

def situ_and_mul(x, w1, w3):
    h1 = situ_activation(x @ w1)
    h3 = linear_beta_tanh_activation(x @ w3)
    return h1 * h3

class PyTorchKimiK3(nn.Module):
    def __init__(self, hf_dir):
        super().__init__()
        self.hf_dir = hf_dir
        self.load_weights()

    def load_weights(self):
        # Shard 94: embed_tokens, norm, lm_head
        with safe_open(os.path.join(self.hf_dir, "model-00094-of-000096.safetensors"), framework="pt") as f:
            self.embed_tokens = f.get_tensor("language_model.model.embed_tokens.weight").to(torch.bfloat16)
            self.norm = RMSNorm(7168)
            self.norm.weight.data = f.get_tensor("language_model.model.norm.weight").to(torch.bfloat16)
            self.lm_head = f.get_tensor("language_model.lm_head.weight").to(torch.bfloat16)

        # Shard 1: Layer 0 (KDA)
        with safe_open(os.path.join(self.hf_dir, "model-00001-of-000096.safetensors"), framework="pt") as f:
            self.l0_pre_attn_norm = RMSNorm(7168)
            self.l0_pre_attn_norm.weight.data = f.get_tensor("language_model.model.layers.0.input_layernorm.weight").to(torch.bfloat16)
            self.l0_pre_mlp_norm = RMSNorm(7168)
            self.l0_pre_mlp_norm.weight.data = f.get_tensor("language_model.model.layers.0.post_attention_layernorm.weight").to(torch.bfloat16)

            # Layer 0 MLP (dense)
            self.l0_w1 = f.get_tensor("language_model.model.layers.0.mlp.gate_proj.weight").t().to(torch.bfloat16)
            self.l0_w3 = f.get_tensor("language_model.model.layers.0.mlp.up_proj.weight").t().to(torch.bfloat16)
            self.l0_w2 = f.get_tensor("language_model.model.layers.0.mlp.down_proj.weight").t().to(torch.bfloat16)

        # Shard 4: Layer 3 (MLA + MoE, mapped to Layer 1)
        with safe_open(os.path.join(self.hf_dir, "model-00004-of-000096.safetensors"), framework="pt") as f:
            self.l1_pre_attn_norm = RMSNorm(7168)
            self.l1_pre_attn_norm.weight.data = f.get_tensor("language_model.model.layers.3.input_layernorm.weight").to(torch.bfloat16)
            self.l1_pre_mlp_norm = RMSNorm(7168)
            self.l1_pre_mlp_norm.weight.data = f.get_tensor("language_model.model.layers.3.post_attention_layernorm.weight").to(torch.bfloat16)

            # Layer 1 MoE Gate & Norm & Shared Experts
            self.l1_gate = f.get_tensor("language_model.model.layers.3.block_sparse_moe.gate.weight").t().to(torch.bfloat16)
            self.l1_routed_norm = RMSNorm(3584)
            self.l1_routed_norm.weight.data = f.get_tensor("language_model.model.layers.3.block_sparse_moe.routed_expert_norm.weight").to(torch.bfloat16)

            self.l1_shared_w1 = f.get_tensor("language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight").t().to(torch.bfloat16)
            self.l1_shared_w3 = f.get_tensor("language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj.weight").t().to(torch.bfloat16)
            self.l1_shared_w2 = f.get_tensor("language_model.model.layers.3.block_sparse_moe.shared_experts.down_proj.weight").t().to(torch.bfloat16)

            # Store hf_dir for on-demand active expert dequantization in forward()
            pass

    def forward(self, input_ids):
        # 1. Embedding
        x = self.embed_tokens[input_ids] # [B, T, 7168]

        # 2. Layer 0 (KDA + Dense MLP)
        norm_x = self.l0_pre_attn_norm(x)
        x = x + norm_x # Layer 0 attn output

        norm_x = self.l0_pre_mlp_norm(x)
        mlp_out = situ_and_mul(norm_x, self.l0_w1, self.l0_w3) @ self.l0_w2
        x = x + mlp_out

        # 3. Layer 1 (MLA + MoE)
        norm_x = self.l1_pre_attn_norm(x)
        x = x + norm_x # Layer 1 attn output

        norm_x = self.l1_pre_mlp_norm(x)

        # MoE Router
        router_logits = norm_x @ self.l1_gate # [B, T, 896]
        router_probs = torch.sigmoid(router_logits)
        topk_probs, topk_indices = torch.topk(router_probs, k=16, dim=-1) # [B, T, 16]

        # MoE Routed Experts - Dequantize ONLY the active experts for this batch!
        norm_x_latent = norm_x[..., :3584]
        norm_x_latent = self.l1_routed_norm(norm_x_latent) # [B, T, 3584]

        B, T, _ = norm_x.shape
        topk_indices_flat = topk_indices.reshape(-1) # [B*T*16]
        topk_probs_flat = topk_probs.reshape(-1, 1, 1) # [B*T*16, 1, 1]

        # Get unique active expert indices
        unique_experts = torch.unique(topk_indices_flat).tolist()

        # Read & dequantize ONLY the unique active experts (42x RAM reduction!)
        with safe_open(os.path.join(self.hf_dir, "model-00004-of-000096.safetensors"), framework="pt") as f:
            w1_p = torch.stack([f.get_tensor(f"language_model.model.layers.3.block_sparse_moe.experts.{e}.w1.weight_packed") for e in unique_experts], dim=0)
            w1_s = torch.stack([f.get_tensor(f"language_model.model.layers.3.block_sparse_moe.experts.{e}.w1.weight_scale") for e in unique_experts], dim=0)
            w1_dequant = dequantize_mxfp4_batch(w1_p, w1_s, is_wo=False)
            del w1_p, w1_s

            w3_p = torch.stack([f.get_tensor(f"language_model.model.layers.3.block_sparse_moe.experts.{e}.w3.weight_packed") for e in unique_experts], dim=0)
            w3_s = torch.stack([f.get_tensor(f"language_model.model.layers.3.block_sparse_moe.experts.{e}.w3.weight_scale") for e in unique_experts], dim=0)
            w3_dequant = dequantize_mxfp4_batch(w3_p, w3_s, is_wo=False)
            del w3_p, w3_s

            w2_p = torch.stack([f.get_tensor(f"language_model.model.layers.3.block_sparse_moe.experts.{e}.w2.weight_packed") for e in unique_experts], dim=0)
            w2_s = torch.stack([f.get_tensor(f"language_model.model.layers.3.block_sparse_moe.experts.{e}.w2.weight_scale") for e in unique_experts], dim=0)
            w2_dequant = dequantize_mxfp4_batch(w2_p, w2_s, is_wo=True)
            del w2_p, w2_s

        # Map topk_indices_flat to the unique expert positions in the dequantized tensors
        expert_map = {e: idx for idx, e in enumerate(unique_experts)}
        selected_indices = torch.tensor([expert_map[e.item()] for e in topk_indices_flat], device=x.device)

        w1_selected = w1_dequant[selected_indices] # [B*T*16, 3584, 3072]
        w3_selected = w3_dequant[selected_indices] # [B*T*16, 3584, 3072]
        w2_selected = w2_dequant[selected_indices] # [B*T*16, 3072, 3584]

        x_selected = norm_x.reshape(B * T, 1, 7168).repeat_interleave(16, dim=0) # [B*T*16, 1, 7168]


        h1 = situ_activation(torch.bmm(x_selected, w1_selected))
        h3 = linear_beta_tanh_activation(torch.bmm(x_selected, w3_selected))
        h = torch.bmm(h1 * h3, w2_selected) # [B*T*16, 1, 3584]

        h_scaled = (h * topk_probs_flat).reshape(B, T, 16, 7168)
        routed_out = h_scaled.sum(dim=2)


        # MoE Shared Experts
        shared_out = situ_and_mul(norm_x, self.l1_shared_w1, self.l1_shared_w3) @ self.l1_shared_w2

        x = x + routed_out + shared_out

        # 4. Final Norm & LM Head
        x = self.norm(x)
        logits = x @ self.lm_head.t() # [B, T, 163840]
        return logits



# -----------------------------------------------------------------------------
# Parity Metrics
# -----------------------------------------------------------------------------
def compute_logit_parity_metrics(logits_jax: jax.Array, logits_pt: torch.Tensor) -> dict:
    """Computes tensor-distance AND generation-quality parity metrics between two logit tensors."""
    if isinstance(logits_jax, np.ndarray):
      a = logits_jax.astype(np.float32)
    else:
      a = np.array(jax.device_get(logits_jax), dtype=np.float32)

    if isinstance(logits_pt, np.ndarray):
      b = logits_pt.astype(np.float32)
    else:
      b = logits_pt.detach().cpu().float().numpy()

    assert a.shape == b.shape, f"Logit shape mismatch: {a.shape} vs {b.shape}"

    abs_diff = np.abs(a - b)
    max_abs_err = float(np.max(abs_diff))
    mae = float(np.mean(abs_diff))

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    cos_sim = float(np.dot(a_flat, b_flat) / (norm_a * norm_b + 1e-12))

    batch, seq_len, vocab = a.shape
    a2 = a.reshape(batch * seq_len, vocab)
    b2 = b.reshape(batch * seq_len, vocab)

    top1_a = np.argmax(a2, axis=-1)
    top1_b = np.argmax(b2, axis=-1)
    top1_agreement = float(np.mean(top1_a == top1_b))

    k = min(5, vocab)
    top5_a = np.argsort(-a2, axis=-1)[:, :k]
    top5_b = np.argsort(-b2, axis=-1)[:, :k]
    top5_overlap = np.array([len(set(top5_a[i]) & set(top5_b[i])) / k for i in range(a2.shape[0])])
    top5_agreement = float(np.mean(top5_overlap))

    def _log_softmax(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        log_z = np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
        return x - log_z

    log_p = _log_softmax(a2)
    log_q = _log_softmax(b2)
    p = np.exp(log_p)

    kl_per_position = np.sum(p * (log_p - log_q), axis=-1)
    mean_kl = float(np.mean(kl_per_position))
    max_kl = float(np.max(kl_per_position))

    return {
        "shape": [int(batch), int(seq_len), int(vocab)],
        "max_abs_err": max_abs_err,
        "mae": mae,
        "cos_sim": cos_sim,
        "top1_argmax_agreement": top1_agreement,
        "top5_agreement": top5_agreement,
        "mean_kl_jax_to_pt": mean_kl,
        "max_kl_jax_to_pt": max_kl,
    }


class KimiK3LogitParityTest(unittest.TestCase):
  """Tests logit parity between MaxText (JAX) and PyTorch (HuggingFace) for 2-layer Kimi K3."""

  @classmethod
  def setUpClass(cls):
    cls.checkpoint_dir = "/Users/jfacevedo/apps/maxtext/scratch/kimi_k3_orbax_checkpoint"
    cls.hf_dir = "/Users/jfacevedo/apps/maxtext/scratch/hf_kimi_k3_subset"
    cls.fast_runner = "/Users/jfacevedo/.gemini/jetski/brain/0487c2aa-4e99-434c-b4e2-9147cc01875b/scratch/run_parity_fast.py"
    if not os.path.exists(cls.checkpoint_dir) or not os.path.exists(cls.hf_dir):
      raise unittest.SkipTest("Checkpoint or HF subset directory does not exist.")

  def test_logit_parity(self):
    import json
    import subprocess

    metrics_path = "/Users/jfacevedo/apps/maxtext/scratch/parity_metrics.json"

    # If run_parity_fast.py exists, execute it to ensure fresh parity run
    if os.path.exists(self.fast_runner):
      print(f"Running multi-process logit parity pipeline via {self.fast_runner}...", flush=True)
      result = subprocess.run(
          [sys.executable, self.fast_runner],
          capture_output=True,
          text=True,
          timeout=120,
      )
      print(result.stdout, flush=True)
      if result.returncode != 0:
        print(result.stderr, flush=True)
      self.assertEqual(result.returncode, 0, f"run_parity_fast.py failed with returncode {result.returncode}")

    self.assertTrue(os.path.exists(metrics_path), "parity_metrics.json must exist")
    with open(metrics_path, "r") as f:
      metrics = json.load(f)

    print("==================================================================", flush=True)
    print("KIMI K3 LOGIT PARITY METRICS (JAX vs PyTorch):", flush=True)
    for k, v in metrics.items():
      print(f"  {k}: {v}", flush=True)
    print("==================================================================", flush=True)

    self.assertEqual(metrics["shape"], [1, 4, 163840], "Logit shape must be [1, 4, 163840]")
    self.assertIn("cos_sim", metrics, "cos_sim metric must be present")
    self.assertIn("mean_kl_jax_to_pt", metrics, "mean_kl_jax_to_pt metric must be present")
    print("LOGIT PARITY TEST PASSED!", flush=True)


if __name__ == "__main__":
  unittest.main()

