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

"""Self-contained PyTorch reference for upstream HuggingFace RecursiveSI/flint."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types


class TorchRouterOutput(NamedTuple):
  """Per-layer PyTorch MoE router tensors (`[B, S, E]` and `[B, S, K]`)."""

  logits: torch.Tensor
  scores: torch.Tensor
  topk_ids: torch.Tensor
  topk_weights: torch.Tensor


class TorchFlintRMSNorm(nn.Module):
  """Flint RMSNorm with FP32 variance and FP32 weight multiplication."""

  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    """Initializes RMSNorm with unit scale."""
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Normalizes `x` along the last dimension."""
    in_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    out = x_f32 * torch.rsqrt(variance + self.variance_epsilon) * self.weight.to(torch.float32)
    return out.to(in_dtype)


def torch_rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates the split-half last dimension `[-x2, x1]`."""
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return torch.cat((-x2, x1), dim=-1)


def torch_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Applies split-half RoPE to `q, k` (`[B, S, H, head_dim]`)."""
  exponents = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / float(head_dim)
  inv_freq = 1.0 / (rope_theta**exponents)
  freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
  emb = torch.cat((freqs, freqs), dim=-1)
  cos = emb.cos().to(dtype=q.dtype).unsqueeze(-2)
  sin = emb.sin().to(dtype=q.dtype).unsqueeze(-2)
  q_embed = (q * cos) + (torch_rotate_half(q) * sin)
  k_embed = (k * cos.to(k.dtype)) + (torch_rotate_half(k) * sin.to(k.dtype))
  return q_embed, k_embed


class TorchFlintMLP(nn.Module):
  """Biasless SwiGLU feed-forward block."""

  def __init__(self, hidden_size: int, intermediate_size: int) -> None:
    """Initializes `gate_proj`, `up_proj`, and `down_proj`."""
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Computes `down_proj(silu(gate_proj(x)) * up_proj(x))`."""
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TorchFlintAttention(nn.Module):
  """Hybrid GQA attention with QK-RMSNorm, optional RoPE, and FP32 sinks."""

  def __init__(
      self,
      model_cfg: flint_config.FlintModelConfig,
      is_global: bool,
  ) -> None:
    """Initializes projections, QK norms, and `softmax_offset` sinks."""
    super().__init__()
    self.is_global = is_global
    self.num_heads = model_cfg.num_attention_heads
    self.num_kv_heads = model_cfg.num_key_value_heads
    self.num_kv_groups = self.num_heads // self.num_kv_heads
    self.head_dim = model_cfg.head_dim
    self.scaling = self.head_dim**-0.5
    self.sliding_window = model_cfg.sliding_window
    self.rope_theta = model_cfg.rope_theta

    q_dim = self.num_heads * self.head_dim
    kv_dim = self.num_kv_heads * self.head_dim
    self.q_proj = nn.Linear(model_cfg.hidden_size, q_dim, bias=False)
    self.k_proj = nn.Linear(model_cfg.hidden_size, kv_dim, bias=False)
    self.v_proj = nn.Linear(model_cfg.hidden_size, kv_dim, bias=False)
    self.o_proj = nn.Linear(q_dim, model_cfg.hidden_size, bias=False)
    self.q_norm = TorchFlintRMSNorm(self.head_dim, eps=model_cfg.rms_norm_eps)
    self.k_norm = TorchFlintRMSNorm(self.head_dim, eps=model_cfg.rms_norm_eps)
    self.softmax_offset = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))

  def forward(
      self,
      hidden_states: torch.Tensor,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Computes GQA attention with optional sliding window and `segment_ids`."""
    bsz, seq_len, _ = hidden_states.shape
    if positions is None:
      positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.int64).unsqueeze(0).expand(bsz, seq_len)

    q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)
    if not self.is_global:
      q, k = torch_apply_rotary_pos_emb(q, k, positions, self.head_dim, rope_theta=self.rope_theta)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2).repeat_interleave(self.num_kv_groups, dim=1)
    v = v.transpose(1, 2).repeat_interleave(self.num_kv_groups, dim=1)

    attn_logits = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * self.scaling

    idx = torch.arange(seq_len, device=hidden_states.device)
    q_idx = idx[:, None]
    k_idx = idx[None, :]
    allow = q_idx >= k_idx
    if not self.is_global:
      allow = allow & ((q_idx - k_idx) < self.sliding_window)
    allow = allow.unsqueeze(0).unsqueeze(0)
    if segment_ids is not None:
      seg_match = segment_ids[:, :, None] == segment_ids[:, None, :]
      allow = allow & seg_match.unsqueeze(1)

    attn_logits = attn_logits.masked_fill(~allow, float("-inf"))
    sink_col = self.softmax_offset.to(torch.float32).view(1, self.num_heads, 1, 1).expand(bsz, self.num_heads, seq_len, 1)
    logits_with_sink = torch.cat((attn_logits, sink_col), dim=-1)
    probs_with_sink = torch.softmax(logits_with_sink, dim=-1)
    attn_probs = probs_with_sink[..., :-1].to(v.dtype)

    attn_out = torch.matmul(attn_probs, v)
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
    return self.o_proj(attn_out)


class TorchFlintSparseMoeBlock(nn.Module):
  """Flint sparse MoE block with FP32 sigmoid top-k router + shared expert."""

  def __init__(self, model_cfg: flint_config.FlintModelConfig) -> None:
    """Initializes `gate`, singular `shared_expert`, and `experts` list."""
    super().__init__()
    self.num_experts = model_cfg.num_experts
    self.top_k = model_cfg.num_experts_per_tok
    self.routed_scaling_factor = model_cfg.routed_scaling_factor
    self.gate = nn.Linear(model_cfg.hidden_size, model_cfg.num_experts, bias=False)
    self.shared_expert = TorchFlintMLP(model_cfg.hidden_size, model_cfg.shared_expert_intermediate_size)
    self.experts = nn.ModuleList(
        [TorchFlintMLP(model_cfg.hidden_size, model_cfg.moe_intermediate_size) for _ in range(model_cfg.num_experts)]
    )

  def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, TorchRouterOutput]:
    """Computes `routed_out + shared_expert(x)` and returns `TorchRouterOutput`."""
    orig_shape = hidden_states.shape
    lead_shape = orig_shape[:-1]
    flat_x = hidden_states.reshape(-1, orig_shape[-1])

    router_logits = torch.matmul(
        flat_x.to(torch.float32),
        self.gate.weight.to(torch.float32).transpose(0, 1),
    )
    scores = torch.sigmoid(router_logits)
    topk_scores, topk_ids = torch.topk(scores, k=self.top_k, dim=-1)
    topk_weights = (topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)) * self.routed_scaling_factor

    routed_out = torch.zeros_like(flat_x)
    topk_weights_cast = topk_weights.to(flat_x.dtype)
    for expert_idx, expert_mlp in enumerate(self.experts):
      tok_idx, slot_idx = torch.where(topk_ids == expert_idx)
      if tok_idx.numel() == 0:
        continue
      expert_out = expert_mlp(flat_x[tok_idx])
      weights = topk_weights_cast[tok_idx, slot_idx].unsqueeze(-1)
      routed_out.index_add_(0, tok_idx, (expert_out * weights).to(routed_out.dtype))

    shared_out = self.shared_expert(flat_x)
    combined = (routed_out + shared_out).view(orig_shape)
    router_out = TorchRouterOutput(
        logits=router_logits.view(*lead_shape, self.num_experts),
        scores=scores.view(*lead_shape, self.num_experts),
        topk_ids=topk_ids.view(*lead_shape, self.top_k),
        topk_weights=topk_weights.view(*lead_shape, self.top_k),
    )
    return combined, router_out


class TorchFlintDecoderLayer(nn.Module):
  """Single Flint decoder layer (dense prefix or sparse MoE)."""

  def __init__(
      self,
      model_cfg: flint_config.FlintModelConfig,
      layer_idx: int,
  ) -> None:
    """Initializes pre/post RMSNorms, attention, and dense or MoE FFN."""
    super().__init__()
    self.layer_idx = layer_idx
    self.is_moe = flint_config.is_moe_layer(layer_idx, model_cfg)
    self.is_global = flint_config.is_global_layer(layer_idx, model_cfg)
    self.input_layernorm = TorchFlintRMSNorm(model_cfg.hidden_size, eps=model_cfg.rms_norm_eps)
    self.self_attn = TorchFlintAttention(model_cfg, is_global=self.is_global)
    self.post_attention_layernorm = TorchFlintRMSNorm(model_cfg.hidden_size, eps=model_cfg.rms_norm_eps)
    self.mlp: TorchFlintMLP | TorchFlintSparseMoeBlock
    if self.is_moe:
      self.mlp = TorchFlintSparseMoeBlock(model_cfg)
    else:
      self.mlp = TorchFlintMLP(model_cfg.hidden_size, model_cfg.intermediate_size)

  def forward(
      self,
      hidden_states: torch.Tensor,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, TorchRouterOutput | None]:
    """Runs one Flint decoder layer and returns `(hidden, router_out)`."""
    h = hidden_states + self.self_attn(
        self.input_layernorm(hidden_states),
        positions=positions,
        segment_ids=segment_ids,
    )
    normed = self.post_attention_layernorm(h)
    if isinstance(self.mlp, TorchFlintSparseMoeBlock):
      ffn_out, router_out = self.mlp(normed)
    else:
      ffn_out = self.mlp(normed)
      router_out = None
    return h + ffn_out, router_out


class TorchFlintModel(nn.Module):
  """Flint transformer backbone (`embed_tokens`, `layers`, `norm`)."""

  def __init__(self, model_cfg: flint_config.FlintModelConfig) -> None:
    """Initializes embedding table, decoder layers, and final RMSNorm."""
    super().__init__()
    self.config = model_cfg
    self.embed_tokens = nn.Embedding(model_cfg.vocab_size, model_cfg.hidden_size)
    self.layers = nn.ModuleList([TorchFlintDecoderLayer(model_cfg, idx) for idx in range(model_cfg.num_hidden_layers)])
    self.norm = TorchFlintRMSNorm(model_cfg.hidden_size, eps=model_cfg.rms_norm_eps)

  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> tuple[
      torch.Tensor,
      tuple[torch.Tensor, ...],
      tuple[TorchRouterOutput | None, ...],
  ]:
    """Returns `(final_normed_hidden, per_layer_hidden_states, router_outs)`."""
    h = self.embed_tokens(input_ids)
    layer_hiddens: list[torch.Tensor] = []
    router_outs: list[TorchRouterOutput | None] = []
    for layer in self.layers:
      h, r_out = layer(h, positions=positions, segment_ids=segment_ids)
      layer_hiddens.append(h)
      router_outs.append(r_out)
    return self.norm(h), tuple(layer_hiddens), tuple(router_outs)


class TorchFlintForCausalLM(nn.Module):
  """Upstream-compatible `FlintForCausalLM` reference with exact HF keys."""

  def __init__(self, model_cfg: flint_config.FlintModelConfig) -> None:
    """Initializes `model` and untied `lm_head`."""
    super().__init__()
    self.config = model_cfg
    self.model = TorchFlintModel(model_cfg)
    self.lm_head = nn.Linear(model_cfg.hidden_size, model_cfg.vocab_size, bias=False)

  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> tuple[
      torch.Tensor,
      tuple[torch.Tensor, ...],
      tuple[TorchRouterOutput | None, ...],
  ]:
    """Returns `(fp32_logits, per_layer_hidden_states, router_outputs)`."""
    normed_h, layer_hiddens, router_outs = self.model(input_ids, positions=positions, segment_ids=segment_ids)
    logits = F.linear(normed_h.to(torch.float32), self.lm_head.weight.to(torch.float32))
    return logits, layer_hiddens, router_outs


def compute_torch_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Computes FP32 next-token cross-entropy loss normalized by valid mask."""
  logits_f32 = logits.to(torch.float32)
  log_probs = F.log_softmax(logits_f32, dim=-1)
  safe_targets = targets.to(torch.int64).clamp(min=0, max=logits.shape[-1] - 1)
  token_nll = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
  if loss_mask is None:
    return token_nll.mean()
  mask_f32 = loss_mask.to(torch.float32)
  denom = mask_f32.sum().clamp(min=1.0)
  return (token_nll * mask_f32).sum() / denom


def randomize_torch_flint(module: nn.Module, seed: int = 0) -> None:
  """Populates `module` parameters with deterministic non-trivial values.

  Args:
    module: PyTorch Flint module to randomize in-place.
    seed: Deterministic CPU generator seed.
  """
  gen = torch.Generator(device="cpu")
  gen.manual_seed(seed)
  with torch.no_grad():
    for name, param in module.named_parameters():
      if name.endswith("softmax_offset"):
        vals = torch.empty(param.shape, dtype=torch.float32).uniform_(-1.5, 1.5, generator=gen)
      elif name.endswith("norm.weight"):
        vals = torch.empty(param.shape, dtype=torch.float32).uniform_(0.5, 2.5, generator=gen)
      else:
        std = 1.0 / math.sqrt(float(param.shape[-1]))
        vals = torch.empty(param.shape, dtype=torch.float32).normal_(mean=0.0, std=std, generator=gen)
      param.copy_(vals.to(device=param.device, dtype=param.dtype))


def expected_hf_state_dict_keys(
    model_cfg: flint_config.FlintModelConfig,
) -> set[str]:
  """Returns the exact set of HuggingFace safetensors keys for `model_cfg`."""
  keys = {
      "model.embed_tokens.weight",
      "model.norm.weight",
      "lm_head.weight",
  }
  for idx in range(model_cfg.num_hidden_layers):
    pfx = f"model.layers.{idx}"
    keys.add(f"{pfx}.input_layernorm.weight")
    keys.add(f"{pfx}.post_attention_layernorm.weight")
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
      keys.add(f"{pfx}.self_attn.{proj}.weight")
    keys.add(f"{pfx}.self_attn.softmax_offset")
    if flint_config.is_moe_layer(idx, model_cfg):
      keys.add(f"{pfx}.mlp.gate.weight")
      for proj in ("gate_proj", "up_proj", "down_proj"):
        keys.add(f"{pfx}.mlp.shared_expert.{proj}.weight")
      for exp_idx in range(model_cfg.num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
          keys.add(f"{pfx}.mlp.experts.{exp_idx}.{proj}.weight")
    else:
      for proj in ("gate_proj", "up_proj", "down_proj"):
        keys.add(f"{pfx}.mlp.{proj}.weight")
  return keys


def _to_np(tensor: torch.Tensor, dtype: jnp.dtype) -> np.ndarray:
  return tensor.detach().cpu().to(torch.float32).numpy().astype(dtype)


def _extract_attn_np(
    state_dict: Mapping[str, torch.Tensor],
    layer_idx: int,
    param_dt: jnp.dtype,
) -> flint_types.FlintAttnWeights[np.ndarray]:
  pfx = f"model.layers.{layer_idx}.self_attn"
  return flint_types.FlintAttnWeights(
      q_proj=_to_np(state_dict[f"{pfx}.q_proj.weight"].T, param_dt),
      k_proj=_to_np(state_dict[f"{pfx}.k_proj.weight"].T, param_dt),
      v_proj=_to_np(state_dict[f"{pfx}.v_proj.weight"].T, param_dt),
      o_proj=_to_np(state_dict[f"{pfx}.o_proj.weight"].T, param_dt),
      q_norm=_to_np(state_dict[f"{pfx}.q_norm.weight"], param_dt),
      k_norm=_to_np(state_dict[f"{pfx}.k_norm.weight"], param_dt),
      sinks=_to_np(state_dict[f"{pfx}.softmax_offset"], jnp.dtype(jnp.float32)),
  )


def _extract_mlp_np(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
    param_dt: jnp.dtype,
) -> flint_types.FlintMLPWeights[np.ndarray]:
  return flint_types.FlintMLPWeights(
      gate_proj=_to_np(state_dict[f"{prefix}.gate_proj.weight"].T, param_dt),
      up_proj=_to_np(state_dict[f"{prefix}.up_proj.weight"].T, param_dt),
      down_proj=_to_np(state_dict[f"{prefix}.down_proj.weight"].T, param_dt),
  )


def _extract_sparse_layer_np(
    state_dict: Mapping[str, torch.Tensor],
    layer_idx: int,
    num_experts: int,
    param_dt: jnp.dtype,
) -> flint_types.FlintSparseLayerWeights[np.ndarray]:
  """Extracts a single sparse MoE layer's weights from an HF `state_dict`."""
  pfx = f"model.layers.{layer_idx}"
  gu_list = []
  down_list = []
  for exp_idx in range(num_experts):
    epfx = f"{pfx}.mlp.experts.{exp_idx}"
    gate_w = _to_np(state_dict[f"{epfx}.gate_proj.weight"].T, param_dt)
    up_w = _to_np(state_dict[f"{epfx}.up_proj.weight"].T, param_dt)
    gu_list.append(np.concatenate([gate_w, up_w], axis=-1))
    down_list.append(_to_np(state_dict[f"{epfx}.down_proj.weight"].T, param_dt))

  return flint_types.FlintSparseLayerWeights(
      input_layernorm=_to_np(state_dict[f"{pfx}.input_layernorm.weight"], param_dt),
      self_attn=_extract_attn_np(state_dict, layer_idx, param_dt),
      post_attention_layernorm=_to_np(state_dict[f"{pfx}.post_attention_layernorm.weight"], param_dt),
      moe=flint_types.FlintMoEWeights(
          router=flint_types.FlintRouterWeights(
              gate=_to_np(state_dict[f"{pfx}.mlp.gate.weight"].T, param_dt),
          ),
          shared_expert=_extract_mlp_np(state_dict, f"{pfx}.mlp.shared_expert", param_dt),
          routed_experts=flint_types.FlintRoutedExpertsWeights(
              gate_up_proj=np.stack(gu_list, axis=0),
              down_proj=np.stack(down_list, axis=0),
          ),
      ),
  )


def torch_state_dict_to_flint_params(
    state_dict: Mapping[str, torch.Tensor],
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
) -> flint_types.FlintParams[jax.Array]:
  """Converts a HuggingFace `state_dict` into sharded `FlintParams` on `mesh`."""
  m = cfg.model
  expected_keys = expected_hf_state_dict_keys(m)
  actual_keys = set(state_dict.keys())
  if actual_keys != expected_keys:
    missing = sorted(expected_keys - actual_keys)[:5]
    extra = sorted(actual_keys - expected_keys)[:5]
    raise ValueError(f"state_dict key mismatch: missing={missing}, unexpected={extra}.")

  param_dt = cfg.train.param_jnp_dtype
  prefix_dense = []
  for idx in range(m.num_dense_prefix_layers):
    pfx = f"model.layers.{idx}"
    prefix_dense.append(
        flint_types.FlintDenseLayerWeights(
            input_layernorm=_to_np(state_dict[f"{pfx}.input_layernorm.weight"], param_dt),
            self_attn=_extract_attn_np(state_dict, idx, param_dt),
            post_attention_layernorm=_to_np(state_dict[f"{pfx}.post_attention_layernorm.weight"], param_dt),
            mlp=_extract_mlp_np(state_dict, f"{pfx}.mlp", param_dt),
        )
    )

  sparse_layers = [
      _extract_sparse_layer_np(state_dict, idx, m.num_experts, param_dt)
      for idx in range(m.num_dense_prefix_layers, m.num_hidden_layers)
  ]

  scanned_blocks: flint_types.FlintSuperblockWeights[np.ndarray] | None
  unscanned_moe: tuple[flint_types.FlintSparseLayerWeights[np.ndarray], ...]
  tail_moe: tuple[flint_types.FlintSparseLayerWeights[np.ndarray], ...]
  if m.scan_layers:
    prefix, num_sb, tail_start = flint_config.flint_scan_layout(m)
    if num_sb > 0:
      slots: list[flint_types.FlintSparseLayerWeights[np.ndarray]] = []
      for slot_idx in range(4):
        slot_layers = [sparse_layers[4 * sb_idx + slot_idx] for sb_idx in range(num_sb)]
        slots.append(jax.tree.map(lambda *xs: np.stack(xs, axis=0), *slot_layers))
      scanned_blocks = flint_types.FlintSuperblockWeights(
          sliding_0=slots[0],
          sliding_1=slots[1],
          sliding_2=slots[2],
          global_3=slots[3],
      )
    else:
      scanned_blocks = None
    unscanned_moe = ()
    tail_moe = tuple(sparse_layers[tail_start - prefix :])
  else:
    scanned_blocks = None
    unscanned_moe = tuple(sparse_layers)
    tail_moe = ()

  np_params = flint_types.FlintParams(
      embed=_to_np(state_dict["model.embed_tokens.weight"], param_dt),
      decoder=flint_types.FlintDecoderWeights(
          prefix_dense=tuple(prefix_dense),
          scanned_blocks=scanned_blocks,
          unscanned_moe=unscanned_moe,
          tail_moe=tail_moe,
      ),
      final_norm=_to_np(state_dict["model.norm.weight"], param_dt),
      lm_head=_to_np(state_dict["lm_head.weight"].T, param_dt),
  )

  shardings = flint_sharding.named_param_shardings(mesh, cfg)
  return jax.tree.map(
      lambda arr, sh: jax.make_array_from_callback(arr.shape, sh, lambda idx: arr[idx]),
      np_params,
      shardings,
  )
