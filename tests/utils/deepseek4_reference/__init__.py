# Copyright 2023–2026 Google LLC
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

"""Official DeepSeek-V4 PyTorch reference, runnable on CPU with autograd.

`model.py` is the upstream inference/model.py with edits marked `# [CHANGE]`;
`kernel.py` and `fast_hadamard_transform.py` are pure-torch replacements for
the tilelang / CUDA dependencies. See README.md for provenance.
"""

import contextlib
import dataclasses

import torch

from tests.utils.deepseek4_reference import kernel
from tests.utils.deepseek4_reference import model

UPSTREAM_REPO = "deepseek-ai/DeepSeek-V4-Flash"
UPSTREAM_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
# sha256 of upstream files at UPSTREAM_REVISION: inference/model.py, encoding/encoding_dsv4.py and
# encoding/tests/* (vendored here as testdata/*).
UPSTREAM_MODEL_SHA256 = "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71"
UPSTREAM_ENCODING_SHA256 = "bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0"
UPSTREAM_TESTDATA_SHA256 = {
    "test_input_1.json": "10e0c074c977c3a80daab758af28219c6b1c2bd7f3f5cf2890c84b361cc32897",
    "test_input_2.json": "c44ae0db20fafff38a6021e8068d7ed6e28605d76cafd20be80b03398509f447",
    "test_input_3.json": "37bf8ef95e0411ea5f411be0b02fbafec7363438b6ccefddca0c52ec9aeaf69a",
    "test_input_4.json": "c45bbd0a1b7a2f75033d8db4ba74ee5b7653bd01114045a52f79d3b387663465",
    "test_output_1.txt": "9b366d9d2eac842a6e890594aac0b58648e5623717202b33497afadf03e26540",
    "test_output_2.txt": "ca66b01a1ac3a204bb032c928fb607d4877171e998a50f7f89f39fa821b75665",
    "test_output_3.txt": "b3b1cd8748b7b90d3c6be6da3f786f12e4d70be073bd445ea162dfad4dc01a64",
    "test_output_4.txt": "60e1643840ba9e4aeede450feb7b0498fa66ee24e4a939d48855ce04ec6fc375",
}

# Sequence length for tiny_args(): 2 * 128 + 8 gives two HCA entries plus a
# non-zero remainder for ratio 128, and 66 CSA entries (> index_topk) for ratio 4.
TINY_SEQ_LEN = 264


def tiny_args(**overrides) -> model.ModelArgs:
  """Small ModelArgs covering compress ratios 0/4/128, hash layers and YaRN.

  Layers 0-2 are hash-routed; layer 2 is the first ratio-4 (indexer) layer and
  layer 3 the first ratio-128 layer. compress_ratios has n_layers + 1 entries
  (trailing 0 for the MTP block), matching the upstream config.json layout.
  original_seq_len < max_seq_len; note upstream precompute_freqs_cis applies
  YaRN whenever original_seq_len > 0, on compressed (ratio > 0) layers only.
  """
  args = model.ModelArgs(
      max_batch_size=2,
      max_seq_len=TINY_SEQ_LEN,
      dtype="bf16",
      scale_fmt="ue8m0",
      expert_dtype=None,
      scale_dtype="fp8",
      vocab_size=256,
      dim=64,
      moe_inter_dim=32,
      n_layers=9,
      n_hash_layers=3,
      n_mtp_layers=1,
      n_heads=4,
      n_routed_experts=8,
      n_shared_experts=1,
      n_activated_experts=2,
      score_func="sqrtsoftplus",
      route_scale=1.5,
      swiglu_limit=10.0,
      q_lora_rank=16,
      head_dim=32,
      rope_head_dim=16,
      o_groups=2,
      o_lora_rank=16,
      window_size=8,
      compress_ratios=(0, 0, 4, 128, 4, 128, 4, 128, 4, 0),
      compress_rope_theta=160000.0,
      original_seq_len=64,
      rope_theta=10000.0,
      rope_factor=16,
      beta_fast=32,
      beta_slow=1,
      index_n_heads=2,
      index_head_dim=16,
      index_topk=4,
      hc_mult=4,
      hc_sinkhorn_iters=20,
  )
  return dataclasses.replace(args, **overrides)


def configure(dtype: torch.dtype, args: model.ModelArgs | None = None):
  """Sets model.py globals as Transformer.__init__ does, for direct Block use.

  For dtype != bfloat16, Linear weights default to `dtype`; modules must still
  be cast (e.g. `.double()`) since upstream pins some weights to bf16/fp32.
  """
  args = args or tiny_args()
  model.world_size = 1
  model.rank = 0
  if dtype == torch.bfloat16:
    model.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
  else:
    model.default_dtype = dtype
  model.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
  model.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32


def init_params(module: torch.nn.Module, args: model.ModelArgs, gen: torch.Generator):
  """Seeded non-degenerate fp32 reference weights matching layerwise init rules."""

  def randn(shape, std, mean=0.0):
    return torch.randn(shape, generator=gen) * std + mean

  with torch.no_grad():
    for name, p in module.named_parameters():
      parts = name.split(".")
      leaf = parts[-1]
      if leaf == "tid2eid":
        rows = [
            torch.randperm(args.n_routed_experts, generator=gen)[: args.n_activated_experts] for _ in range(p.shape[0])
        ]
        p.copy_(torch.stack(rows).to(p.dtype))
      elif leaf == "attn_sink":
        p.copy_(randn(p.shape, 0.5).to(p.dtype))
      elif leaf == "ape":
        p.copy_(randn(p.shape, 0.1).to(p.dtype))
      elif name.endswith("gate.bias"):
        p.copy_(randn(p.shape, 0.05).to(p.dtype))
      elif leaf.startswith("hc_") and leaf.endswith("_fn"):
        p.copy_(randn(p.shape, 0.02).to(p.dtype))
      elif leaf.startswith("hc_") and leaf.endswith("_base"):
        p.copy_(randn(p.shape, 0.5).to(p.dtype))
      elif leaf.startswith("hc_") and leaf.endswith("_scale"):
        p.copy_((torch.rand(p.shape, generator=gen) + 0.5).to(p.dtype))
      elif leaf == "weight" and any(x.endswith("norm") for x in parts[:-1]):
        p.copy_(randn(p.shape, 0.1, 1.0).to(p.dtype))
      elif name == "embed.weight":
        p.copy_(randn(p.shape, 1.0).to(p.dtype))
      elif leaf == "weight":
        p.copy_(randn(p.shape, 0.02).to(p.dtype))
      else:
        raise ValueError(f"no init rule for {name}")


@contextlib.contextmanager
def fp64_promotion():
  """Keeps float64 activations float64 inside model.py (gradcheck only).

  model.py hard-codes fp32 upcasts (`.float()` in RMSNorm, Gate, Compressor,
  Expert, hc_pre, apply_rotary_emb; `zeros_like(..., dtype=float32)` in
  MoE.forward) that would round finite-difference perturbations at ~1e-7
  relative. For float64 tensors only, these become no-ops.
  """
  orig_float = torch.Tensor.float
  orig_zeros_like = torch.zeros_like

  def _float(self, *a, **k):
    if self.dtype == torch.float64 and not a and not k:
      return self
    return orig_float(self, *a, **k)

  def _zeros_like(x, *a, dtype=None, **k):
    if x.dtype == torch.float64 and dtype == torch.float32:
      dtype = torch.float64
    return orig_zeros_like(x, *a, dtype=dtype, **k)

  torch.Tensor.float = _float
  torch.zeros_like = _zeros_like
  try:
    yield
  finally:
    torch.Tensor.float = orig_float
    torch.zeros_like = orig_zeros_like
