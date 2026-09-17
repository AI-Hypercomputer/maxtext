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

"""Run the publisher's text decoder on CPU without its CUDA-only FLA kernels.

The decoder, MLA, router, MLP and norms are loaded verbatim from pinned upstream
source. Only ShortConvolution, gated RMSNorm and the fused KDA kernel receive
plain PyTorch fallbacks. This is a validation adapter, not a checkpoint converter.
"""
import ast
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.activations import ACT2FN


class ShortConvolution(nn.Conv1d):

  def __init__(self, hidden_size, kernel_size, activation="silu", **kwargs):
    super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)

  def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
    if cu_seqlens is not None and len(cu_seqlens) > 2:
      raise ValueError("CPU reference supports one unpadded sequence per batch.")
    x = x.transpose(1, 2)
    width = self.kernel_size[0]
    prefix = (
        torch.zeros((*x.shape[:2], width - 1), dtype=x.dtype, device=x.device)
        if cache is None
        else cache[..., -(width - 1) :]
    )
    joined = torch.cat((prefix, x), -1)
    output = F.silu(F.conv1d(joined, self.weight, groups=self.groups)).transpose(1, 2)
    return output, joined[..., -width:].clone() if output_final_state else None


class FusedRMSNormGated(nn.Module):

  def __init__(self, hidden_size, eps=1e-6, activation="sigmoid"):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.eps = eps

  def forward(self, x, gate):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
    return y.to(x.dtype) * self.weight * torch.sigmoid(gate.float()).to(x.dtype)


def recurrent_kda(q, k, v, g, beta, A_log, dt_bias, initial_state=None, output_final_state=True, **kwargs):
  # FLA's published naive recurrence; fused-kernel normalization epsilon is 1e-6.
  q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
  k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
  g = kwargs["lower_bound"] * torch.sigmoid(
      A_log.float().exp().reshape(1, 1, -1, 1) * (g.float() + dt_bias.float().reshape(1, 1, q.shape[2], -1))
  )
  state = (
      torch.zeros((q.shape[0], q.shape[2], q.shape[3], v.shape[3])) if initial_state is None else initial_state.clone()
  )
  outputs = []
  for i in range(q.shape[1]):
    state = state * g[:, i].exp().unsqueeze(-1)
    residual = v[:, i].float() - (k[:, i, :, :, None] * state).sum(-2)
    state = state + torch.einsum("bhk,bhv->bhkv", beta[:, i, :, None] * k[:, i], residual)
    outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * q.shape[-1] ** -0.5, state))
  return torch.stack(outputs, 1).to(v.dtype), state if output_final_state else None


def load_reference(source_path, config_path):
  """Extract unchanged decoder definitions to avoid Transformers version imports."""
  namespace = dict(globals())
  config_tree = ast.parse(Path(config_path).read_text())
  cfg_class = next(n for n in config_tree.body if isinstance(n, ast.ClassDef))

  class ConfigBase:

    def __init__(self, **kwargs):
      self.__dict__.update(kwargs)
      self._attn_implementation = "eager"

    def to_dict(self):
      return dict(self.__dict__)

  namespace["PretrainedConfig"] = ConfigBase
  exec(compile(ast.Module(body=[cfg_class], type_ignores=[]), str(config_path), "exec"), namespace)
  names = {
      "index_first_axis",
      "index_put_first_axis",
      "pad_input",
      "_get_unpad_data",
      "BailingMoeV3RMSNorm",
      "BailingMoeV3MLP",
      "BailingMoeV3Gate",
      "BailingMoeV3SparseMoeBlock",
      "rotate_half",
      "repeat_kv2",
      "eager_attention_forward",
      "apply_rotary_pos_emb_interleave",
      "yarn_get_mscale",
      "BailingMoeV3MultiLatentAttention",
      "BailingMoeV3KimiDeltaAttention",
      "BailingMoeV3DecoderLayer",
  }
  tree = ast.parse(Path(source_path).read_text())
  nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
  # Only remove version/deprecation decorators and runtime-evaluated annotations.
  for node in nodes:
    for child in ast.walk(node):
      if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
        child.decorator_list = [
            d
            for d in child.decorator_list
            if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) and d.value.id == "torch"
        ]
  namespace.update(chunk_kda=recurrent_kda, fused_recurrent_kda=recurrent_kda)
  future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
  tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
  exec(compile(tree, str(source_path), "exec"), namespace)
  return SimpleNamespace(**namespace)


def position_embeddings(config, positions):
  # Upstream BailingMoeV3RotaryEmbedding uses the explicit rotary width, ignoring
  # partial_rotary_factor; avoid the changed Transformers ROPE_INIT_FUNCTIONS API.
  inv = 1.0 / (config.rope_theta ** (torch.arange(0, config.qk_rope_head_dim, 2).float() / config.qk_rope_head_dim))
  phase = positions.float()[..., None] * inv
  phase = torch.cat((phase, phase), -1)
  return phase.cos(), phase.sin()
