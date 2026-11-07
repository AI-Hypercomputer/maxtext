# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Consolidated test utilities and HuggingFace integration helpers for GLM-5.3-Flash."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import os
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
import numpy as np

try:
  import safetensors
  import torch
  from transformers.integrations import moe as hf_moe_integ
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

MINI_SLICE_DIR = "/dev/shm/hf_mini/glm5_3_flash_5layers"


def _make_cfg(
    num_hidden_layers: int = 7,
    scan_layers: bool = True,
    dtype: str = "float32",
    weight_dtype: str = "float32",
    index_topk: int = 8,
    enable_debug_stats: bool = False,
) -> tuple[glm5_config.Glm5Config, hf_cfg.Glm5NextTextConfig | None]:
  """Constructs aligned JAX and HuggingFace model configurations with mini dimensions."""
  kws = (
      "hidden_size=64 intermediate_size=128 moe_intermediate_size=32"
      " n_routed_experts=8 num_experts_per_tok=2 q_lora_rank=32 kv_lora_rank=16"
      " num_attention_heads=4 num_key_value_heads=4 qk_nope_head_dim=16"
      " v_head_dim=16 index_n_heads=8 index_head_dim=16 linear_num_key_heads=4"
      " linear_num_value_heads=4 linear_key_head_dim=16"
      " linear_value_head_dim=16"
  )
  args = {k: int(v) for k, v in (s.split("=") for s in kws.split())}
  args.update(
      num_hidden_layers=num_hidden_layers,
      scan_layers=scan_layers,
      index_topk=index_topk,
      kda_implementation="xla",
      vocab_size=256,
      dtype=dtype,
      weight_dtype=weight_dtype,
      enable_debug_stats=enable_debug_stats,
  )
  c = glm5_config.Glm5Config.mini_config(**args)
  hf_args = {x: args[x] for x in args if not x.startswith("linear_") and x != "enable_debug_stats"}
  hf_k = (
      "vocab_size pad_token_id num_hidden_layers _attn_implementation "
      "index_kpool linear_num_heads linear_head_dim linear_conv_kernel_dim"
  ).split()
  hf_args.update(zip(hf_k, (256, 0, num_hidden_layers, "eager", 4, 4, 16, 4)))
  hf_args["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * (num_hidden_layers - 3)
  hf_args["layer_types"] = list(c.layer_types)
  hf_args["indexer_types"] = list(c.indexer_types)
  return c, (hf_cfg.Glm5NextTextConfig(**hf_args) if HAS_TORCH else None)


def _init_hf(hf_m: torch.nn.Module, hf_head: torch.nn.Module) -> None:
  """Initializes HuggingFace model weights with deterministic random values."""
  with torch.no_grad():
    for n, p in list(hf_m.named_parameters()) + list(hf_m.named_buffers()):
      if p.is_floating_point():
        if "layernorm" in n or "norm" in n:
          if n.endswith(".bias"):
            p.zero_()
          else:
            p.fill_(1.0)
        elif "embed_tokens.weight" in n:
          p.normal_(0.0, 0.02)
          p[:, 0] = 0.06
        elif "weights_proj.weight" in n:
          p.normal_(0.0, 0.002)
          p[:, 0] = 0.25
        elif "e_score_correction_bias" in n:
          p.copy_(torch.linspace(-0.05, 0.05, p.numel(), dtype=p.dtype).reshape(p.shape))
        elif "A_log" in n:
          p.uniform_(-3.5, -1.5)
        elif "dt_bias" in n:
          p.uniform_(-2.0, -0.5)
        else:
          p.normal_(0.0, 0.02)
    hf_head.weight.normal_(0.0, 0.02)


def _hf_forward(
    hf_m: torch.nn.Module,
    hf_head: torch.nn.Module,
    ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Executes HuggingFace reference forward pass with recurrent kimi delta attention."""
  orig = hf_model.chunk_kimi_delta_attention
  hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
  try:
    if attention_mask is not None:
      return hf_head(hf_m(input_ids=ids, attention_mask=attention_mask).last_hidden_state)
    return hf_head(hf_m(input_ids=ids).last_hidden_state)
  finally:
    hf_model.chunk_kimi_delta_attention = orig


@contextlib.contextmanager
def _fp64_mode():
  """Context manager promoting float32 casts in JAX and Torch/HF to float64."""
  orig_x64 = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(jnp, "float32", jnp.float64))
      if HAS_TORCH:
        stack.enter_context(mock.patch.object(torch, "float32", torch.float64))
        stack.enter_context(mock.patch.object(torch.Tensor, "float", torch.Tensor.double))
        stack.enter_context(
            mock.patch.object(
                hf_moe_integ,
                "_can_use_grouped_mm",
                lambda *args, **kwargs: False,
            )
        )
      yield
  finally:
    jax.config.update("jax_enable_x64", orig_x64)


def assert_e3(a: Any, b: Any, name: str = "", *, is_fp64: bool = False) -> tuple[float, float, float]:
  """Asserts parameter parity using E3 relative Frobenius and cosine metrics."""
  dt = np.float64 if is_fp64 else np.float32
  a_f, b_f = np.asarray(a, dtype=dt), np.asarray(b, dtype=dt)
  norm_a = float(np.linalg.norm(a_f))
  norm_b = float(np.linalg.norm(b_f))
  if norm_b == 0.0:
    if norm_a != 0.0:
      raise AssertionError(f"{name}: expected zero norm, got {norm_a}")
    return (0.0, 0.0, 1.0)
  if norm_a == 0.0:
    raise AssertionError(f"{name}: actual norm is zero, expected non-zero")
  diff = float(np.linalg.norm(a_f - b_f))
  frob_rel = diff / norm_b
  floored_frob_rel = diff / max(1.0, norm_b)
  cos = float(np.sum(a_f * b_f) / (norm_a * norm_b))
  ref_inf = float(np.max(np.abs(b_f)))
  max_rel_inf = float(np.max(np.abs(a_f - b_f))) / ref_inf if ref_inf > 0.0 else 0.0
  if is_fp64:
    if frob_rel > 1e-10:
      raise AssertionError(f"FP64 Frobenius mismatch in {name}: frob_rel={frob_rel:.3e} > 1e-10")
    return (frob_rel, max_rel_inf, cos)
  np.testing.assert_allclose(a_f, b_f, atol=1e-5 * max(1.0, ref_inf), rtol=1e-5, err_msg=name)
  if floored_frob_rel > 1e-5:
    raise AssertionError(f"{name} frob_rel={floored_frob_rel:.3e} exceeds tolerance 1e-5")
  if cos <= 0.99999:
    raise AssertionError(f"{name} cosine similarity={cos} <= 0.99999")
  return (floored_frob_rel, max_rel_inf, cos)


_assert_e3 = assert_e3


def _hf_module_state_to_hub(
    state_dict: Mapping[str, torch.Tensor],
    *,
    num_experts: int,
    moe_intermediate_size: int,
) -> dict[str, torch.Tensor]:
  """Inverts transformers glm5_next conversion mapping to Hub safetensors format.

  Authority: transformers conversion_mapping.py:535-580.
  Converts PyTorch Glm5NextTextModel state_dict keys and fused tensors into
  the exact flat Hub safetensors layout under `model.language_model.` and
  `lm_head.`.

  Args:
    state_dict: PyTorch Glm5NextTextModel state dictionary.
    num_experts: Number of routed experts in the model.
    moe_intermediate_size: Intermediate dimension of MoE experts.

  Returns:
    Flat dictionary mapping Hub safetensors tensor names to tensors.
  """
  hub_dict: dict[str, torch.Tensor] = {}
  for orig_k, tensor in state_dict.items():
    if orig_k == "lm_head.weight":
      hub_dict["lm_head.weight"] = tensor
      continue
    k = orig_k[len("model.language_model.") :] if orig_k.startswith("model.language_model.") else orig_k

    if k.endswith(".self_attn.conv1d.weight"):
      prefix = k[: -len("self_attn.conv1d.weight")]
      q_t, k_t, v_t = torch.chunk(tensor, 3, dim=0)
      hub_dict[f"model.language_model.{prefix}self_attn.q_conv1d.weight"] = q_t
      hub_dict[f"model.language_model.{prefix}self_attn.k_conv1d.weight"] = k_t
      hub_dict[f"model.language_model.{prefix}self_attn.v_conv1d.weight"] = v_t
      continue

    if k.endswith(".mlp.experts.gate_up_proj"):
      prefix = k[: -len("mlp.experts.gate_up_proj")]
      for e in range(num_experts):
        ge, up_e = torch.split(tensor[e], moe_intermediate_size, dim=0)
        hub_dict[f"model.language_model.{prefix}mlp.experts.{e}.gate_proj.weight"] = ge
        hub_dict[f"model.language_model.{prefix}mlp.experts.{e}.up_proj.weight"] = up_e
      continue

    if k.endswith(".mlp.experts.down_proj"):
      prefix = k[: -len("mlp.experts.down_proj")]
      for e in range(num_experts):
        hub_dict[f"model.language_model.{prefix}mlp.experts.{e}.down_proj.weight"] = tensor[e]
      continue

    k = k.replace("self_attn.forget_gate.f_a_proj.", "self_attn.f_a_proj.")
    k = k.replace("self_attn.forget_gate.f_b_proj.", "self_attn.f_b_proj.")
    k = k.replace("self_attn.forget_gate.dt_bias", "self_attn.dt_bias")
    k = k.replace("self_attn.forget_gate.A_log", "self_attn.A_log")

    k = k.replace("attn_hc.fn", "hc_attn_fn")
    k = k.replace("attn_hc.base", "hc_attn_base")
    k = k.replace("attn_hc.scale", "hc_attn_scale")
    k = k.replace("ffn_hc.fn", "hc_ffn_fn")
    k = k.replace("ffn_hc.base", "hc_ffn_base")
    k = k.replace("ffn_hc.scale", "hc_ffn_scale")

    hub_dict[f"model.language_model.{k}"] = tensor
  return hub_dict


_ORIG_TORCH_FLOAT32 = torch.float32 if HAS_TORCH else None


class _LazySafetensorsMapping(Mapping[str, Any]):
  """Memory-safe lazy safetensors mapping fetching tensors on-demand from mmap."""

  def __init__(self, safetensors_path: str):
    self._path = safetensors_path
    self._safe_open = safetensors.safe_open(safetensors_path, framework="pt")
    self._keys = set(self._safe_open.keys())

  def __getitem__(self, key: str) -> Any:
    with mock.patch.object(torch, "float32", _ORIG_TORCH_FLOAT32):
      return self._safe_open.get_tensor(key)

  def get_slice(self, key: str, row_slice: slice) -> Any:
    with mock.patch.object(torch, "float32", _ORIG_TORCH_FLOAT32):
      return self._safe_open.get_slice(key)[row_slice]

  def close(self) -> None:
    self._safe_open = None

  def __len__(self) -> int:
    return len(self._keys)

  def __iter__(self) -> Iterator[str]:
    return iter(self._keys)

  def keys(self) -> set[str]:
    return self._keys


def ensure_real_mini_slice(slice_dir: str, num_hidden_layers: int = 5) -> None:
  """Verifies that the required real-weight mini-slice files exist on disk."""
  del num_hidden_layers
  if not slice_dir or not slice_dir.strip():
    raise ValueError("slice_dir must be explicitly specified and non-empty.")
  for fname in ("model.safetensors", "config.json"):
    fpath = os.path.join(slice_dir, fname)
    if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
      raise FileNotFoundError(f"Required mini-slice file missing or empty in {slice_dir!r}: {fpath}")


_ensure_real_mini_slice = ensure_real_mini_slice
