# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for MaxText-vs-PyTorch Kimi-K3 parity tests.

This module is the single place that knows how to:

  1. Load the local Hugging Face reference (`kimi-k3-hf-reference/`) and make its
     KDA path faithful on CPU (see `chunk_kda_reference`).
  2. Build a *paired* tiny configuration: one `KimiLinearConfig` for PyTorch and one
     MaxText `HyperParameters` (through the real `pyconfig` pipeline with
     `model_name="kimi-k3"`) from a single `TinyKimiK3Spec`.
  3. Convert a PyTorch `state_dict` through the production
     `PARAM_MAPPING["kimi-k3"]` / `HOOK_FNS["kimi-k3"]` and load the result into an
     NNX module by walking its parameter tree.

Why the KDA path needs patching
-------------------------------
The reference runs KDA through `fla` CUDA kernels with `use_qk_l2norm_in_kernel`,
`use_gate_in_kernel`, `use_beta_sigmoid_in_kernel` and `lower_bound` set. Without
`fla` installed the reference falls back to `_fallback_kda`, which silently ignores
the L2-norm, the 1/sqrt(d) query scale and the lower-bound gate. `chunk_kda_reference`
reproduces the kernel semantics on CPU (`tests/utils/kimi_k3_kda_reference.py`, written
to the spec of fla's pure-torch reference functions and checked against them) and is
monkeypatched over both `chunk_kda` and
`fused_recurrent_kda` by `load_hf_reference`. Note that with `lower_bound` set, `fla`'s
gate is `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, not a clamped softplus.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils import kimi_k3_kda_reference
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR
from tests.utils.test_helpers import get_test_config_path


# =============================================================================
# 1. Faithful pure-PyTorch KDA kernel + HF reference loader
# =============================================================================
def chunk_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: Optional[float] = -5.0,
    transpose_state_layout: bool = True,
    cu_seqlens: Any = None,
    **unused_kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """CPU stand-in for `fla`'s `chunk_kda` / `fused_recurrent_kda`
  (`tests/utils/kimi_k3_kda_reference.py`).

  Shapes: q, k, v, g are [B, T, H, D]; beta is [B, T, H]; A_log is [H]; dt_bias is
  [H * D]. Returns (o [B, T, H, D] in v's dtype, final_state or None).

  NOTE: with `lower_bound` set the decay gate is `lower_bound * sigmoid(exp(A_log) * (g +
  dt_bias))`, not a clamp of the softplus gate (an earlier version of this oracle had it
  wrong, and so did MaxText).
  """
  del unused_kwargs
  return kimi_k3_kda_reference.chunk_kda(
      q,
      k,
      v,
      g,
      beta,
      A_log=A_log,
      dt_bias=dt_bias,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      use_gate_in_kernel=use_gate_in_kernel,
      use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      transpose_state_layout=transpose_state_layout,
      cu_seqlens=cu_seqlens,
  )


_HF_KDA_ORIGINALS: dict[str, Any] = {}


def load_hf_reference(patch_kda: bool = True):
  """Imports the local HF reference modules, optionally patching the KDA kernel.

  Returns `(configuration_kimi_k3, modeling_kimi_linear)`.

  The patch mutates a module object that is shared for the lifetime of the process, so
  callers must pair this with `unpatch_hf_reference()` (typically in `tearDownClass`).
  Otherwise every test file that later imports `modeling_kimi_linear` in the same pytest
  process silently gets the patched kernel, and whether a suite passes depends on the
  order pytest happens to collect files in.
  """
  if KIMI_K3_REFERENCE_DIR not in sys.path:
    sys.path.insert(0, KIMI_K3_REFERENCE_DIR)
  import configuration_kimi_k3 as hf_config  # pylint: disable=import-outside-toplevel
  import modeling_kimi_linear as hf_model  # pylint: disable=import-outside-toplevel

  if patch_kda:
    if not _HF_KDA_ORIGINALS:
      _HF_KDA_ORIGINALS["module"] = hf_model
      _HF_KDA_ORIGINALS["chunk_kda"] = hf_model.chunk_kda
      _HF_KDA_ORIGINALS["fused_recurrent_kda"] = hf_model.fused_recurrent_kda
    hf_model.chunk_kda = chunk_kda_reference
    hf_model.fused_recurrent_kda = chunk_kda_reference
  return hf_config, hf_model


def unpatch_hf_reference() -> None:
  """Restores the kernels replaced by `load_hf_reference(patch_kda=True)`."""
  if not _HF_KDA_ORIGINALS:
    return
  module = _HF_KDA_ORIGINALS.pop("module")
  for name, fn in _HF_KDA_ORIGINALS.items():
    setattr(module, name, fn)
  _HF_KDA_ORIGINALS.clear()


# The HF reference allocates these with `torch.empty` and never initializes them (neither
# the module `__init__` nor `KimiPreTrainedModel._init_weights` touches them): in production
# they come from the checkpoint. A freshly constructed reference therefore holds whatever
# the allocator hands back -- usually benign on fresh pages, but garbage / NaN after a
# memory-heavy test, which made parity suites fail depending on pytest collection order.
def init_hf_uninitialized_params(model: "torch.nn.Module", seed: int = 0) -> "torch.nn.Module":
  """Deterministically fills the HF reference params that are left as `torch.empty`.

  Uses a private generator so the global torch RNG (and hence every other weight a test
  draws) is unaffected. `dt_bias` gets small uniform values so the KDA decay path is
  exercised; the router correction bias is zeroed, which keeps top-k selection driven by
  the (random) router weights. Call right after constructing any HF Kimi module.
  """
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      if name.endswith("dt_bias"):
        p.copy_(torch.rand(p.shape, generator=gen, dtype=torch.float32).mul_(2.0).sub_(1.0).mul_(0.5).to(p.dtype))
      elif name.endswith("e_score_correction_bias"):
        p.zero_()
  return model


# =============================================================================
# 2. Paired tiny configuration
# =============================================================================
# MaxText hardcodes these in `linears.situ_gate` / `linears.situ_linear`; the real
# checkpoint uses the same values (`activation_situ_beta: 4.0`,
# `activation_situ_linear_beta: 25.0` in config.json).
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


@dataclasses.dataclass(frozen=True)
class TinyKimiK3Spec:
  """One source of truth for a toy Kimi-K3 shared by PyTorch and MaxText.

  Defaults: 8 layers in two AttnRes blocks of 4, with MLA at 0-indexed layers 3 and 7
  (mirroring the real 3xKDA+1xMLA cycle), dense MLP on layer 0, latent MoE elsewhere.
  """

  vocab_size: int = 256
  hidden_size: int = 64
  num_layers: int = 8
  num_heads: int = 4
  head_dim: int = 16
  # MLA
  q_lora_rank: int = 32
  kv_lora_rank: int = 16
  qk_nope_head_dim: int = 16
  qk_rope_head_dim: int = 8
  v_head_dim: int = 16
  # MLP / MoE
  intermediate_size: int = 128  # layer-0 dense MLP
  num_experts: int = 4
  top_k: int = 2
  routed_expert_hidden_size: int = 32
  moe_intermediate_size: int = 32
  num_shared_experts: int = 1
  first_k_dense_replace: int = 1
  # Hybrid / highway layout (0-indexed)
  full_attn_layers: tuple[int, ...] = (3, 7)
  attn_res_block_size: int = 4
  # Misc
  short_conv_kernel_size: int = 4
  gate_lower_bound: float = -5.0
  rms_norm_eps: float = 1e-5
  seq_len: int = 8
  batch_size: int = 2

  @property
  def shared_intermediate_size(self) -> int:
    # HF derives the shared expert width as moe_intermediate_size * num_shared_experts.
    return self.moe_intermediate_size * self.num_shared_experts

  def is_full_attn(self, layer_idx: int) -> bool:
    return layer_idx in self.full_attn_layers

  def is_moe(self, layer_idx: int) -> bool:
    return layer_idx >= self.first_k_dense_replace

  # --- PyTorch side -----------------------------------------------------------
  def hf_config_kwargs(self) -> dict[str, Any]:
    """kwargs for `KimiLinearConfig` (HF layer lists are 1-indexed)."""
    full_1idx = [i + 1 for i in self.full_attn_layers]
    kda_1idx = [i + 1 for i in range(self.num_layers) if not self.is_full_attn(i)]
    return {
        "vocab_size": self.vocab_size,
        "hidden_size": self.hidden_size,
        "intermediate_size": self.intermediate_size,
        "num_hidden_layers": self.num_layers,
        "num_attention_heads": self.num_heads,
        "num_key_value_heads": self.num_heads,
        "head_dim": self.head_dim,
        "q_lora_rank": self.q_lora_rank,
        "kv_lora_rank": self.kv_lora_rank,
        "qk_nope_head_dim": self.qk_nope_head_dim,
        "qk_rope_head_dim": self.qk_rope_head_dim,
        "v_head_dim": self.v_head_dim,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "num_experts": self.num_experts,
        "num_experts_per_token": self.top_k,
        "routed_expert_hidden_size": self.routed_expert_hidden_size,
        "moe_intermediate_size": self.moe_intermediate_size,
        "num_shared_experts": self.num_shared_experts,
        "routed_scaling_factor": 1.0,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "latent_moe_use_norm": True,
        "topk_method": "noaux_tc",
        "use_grouped_topk": False,
        "rms_norm_eps": self.rms_norm_eps,
        "first_k_dense_replace": self.first_k_dense_replace,
        "attn_res_block_size": self.attn_res_block_size,
        "hidden_act": "situ",
        "activation_situ_beta": SITU_BETA,
        "activation_situ_linear_beta": SITU_LINEAR_BETA,
        "max_position_embeddings": self.seq_len,
        "linear_attn_config": {
            "short_conv_kernel_size": self.short_conv_kernel_size,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "use_full_rank_gate": True,
            "gate_lower_bound": self.gate_lower_bound,
            "kda_layers": kda_1idx,
            "full_attn_layers": full_1idx,
        },
        # FlashAttention-2 is CUDA-only; eager is the CPU path.
        "_attn_implementation": "eager",
    }

  # --- Checkpoint-mapping side -----------------------------------------------
  def mapping_config(self) -> dict[str, Any]:
    """Flat dict consumed by `PARAM_MAPPING["kimi-k3"]` / `HOOK_FNS["kimi-k3"]`.

    `full_attn_layers` is given 0-indexed at the top level on purpose: the mapping's
    1-indexed auto-detection heuristic (`4 in list and 3 not in list`) is not reliable
    for arbitrary toy layouts.
    """
    return {
        "hidden_size": self.hidden_size,
        "num_hidden_layers": self.num_layers,
        "num_attention_heads": self.num_heads,
        "num_key_value_heads": self.num_heads,
        "q_lora_rank": self.q_lora_rank,
        "kv_lora_rank": self.kv_lora_rank,
        "qk_nope_head_dim": self.qk_nope_head_dim,
        "qk_rope_head_dim": self.qk_rope_head_dim,
        "v_head_dim": self.v_head_dim,
        "num_experts": self.num_experts,
        "first_k_dense_replace": self.first_k_dense_replace,
        "full_attn_layers": list(self.full_attn_layers),
        "rms_norm_eps": self.rms_norm_eps,
    }

  # --- MaxText side -----------------------------------------------------------
  def maxtext_overrides(self) -> dict[str, Any]:
    """Overrides for `pyconfig.initialize(..., model_name="kimi-k3", ...)`."""
    return {
        "override_model_config": True,
        # Core dims.
        "base_emb_dim": self.hidden_size,
        "base_num_decoder_layers": self.num_layers,
        "base_num_query_heads": self.num_heads,
        "base_num_kv_heads": self.num_heads,
        "head_dim": self.head_dim,
        "vocab_size": self.vocab_size,
        # MLA.
        "q_lora_rank": self.q_lora_rank,
        "kv_lora_rank": self.kv_lora_rank,
        "qk_nope_head_dim": self.qk_nope_head_dim,
        "qk_rope_head_dim": self.qk_rope_head_dim,
        "v_head_dim": self.v_head_dim,
        "mla_use_output_gate": True,
        "mla_naive_kvcache": True,
        "attention": "dot_product",
        # MLP / MoE. `moe_intermediate_size` / `num_shared_experts` are what the Kimi
        # layer code reads; the `base_moe_mlp_dim` / `shared_experts` twins keep the
        # generic MaxText validators happy.
        "base_mlp_dim": self.intermediate_size,
        "base_moe_mlp_dim": self.moe_intermediate_size,
        "moe_intermediate_size": self.moe_intermediate_size,
        "shared_intermediate_size": self.shared_intermediate_size,
        "routed_expert_hidden_size": self.routed_expert_hidden_size,
        "num_experts": self.num_experts,
        "num_experts_per_tok": self.top_k,
        "shared_experts": self.num_shared_experts,
        "num_shared_experts": self.num_shared_experts,
        "routed_scaling_factor": 1.0,
        "first_num_dense_layers": self.first_k_dense_replace,
        "first_k_dense_replace": self.first_k_dense_replace,
        # Hybrid / highway.
        "full_attn_layers": list(self.full_attn_layers),
        "attn_res_block_size": self.attn_res_block_size,
        "short_conv_kernel_size": self.short_conv_kernel_size,
        "normalization_layer_epsilon": self.rms_norm_eps,
        # Sequence. Keep original == max so MLA's YaRN mscale rescaling stays off.
        "max_target_length": self.seq_len,
        "max_position_embeddings": self.seq_len,
        "original_max_position_embeddings": self.seq_len,
        "max_prefill_predict_length": self.seq_len,
        "per_device_batch_size": self.batch_size,
        "global_batch_size_to_train_on": self.batch_size,
        # Numerics: parity tests run in fp32 at highest precision.
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        # Kimi-K3 is unscanned only for now.
        "scan_layers": False,
        "enable_checkpointing": False,
        "enable_dropout": False,
    }


def make_hf_config(spec: TinyKimiK3Spec, hf_config_module=None):
  """Builds a `KimiLinearConfig` for `spec`."""
  if hf_config_module is None:
    hf_config_module, _ = load_hf_reference()
  return hf_config_module.KimiLinearConfig(**spec.hf_config_kwargs())


def make_maxtext_config(spec: TinyKimiK3Spec, **extra_overrides):
  """Builds a MaxText `HyperParameters` for `spec` via the real kimi-k3 yml."""
  overrides = spec.maxtext_overrides()
  overrides.update(extra_overrides)
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      model_name="kimi-k3",
      **overrides,
  )


def make_mesh(cfg) -> Mesh:
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


# =============================================================================
# 3. Weight conversion + NNX loading
# =============================================================================
def convert_hf_layer_state_dict(
    pt_layer: torch.nn.Module,
    layer_idx: int,
    spec: TinyKimiK3Spec,
) -> dict[str, np.ndarray]:
  """Converts a single `KimiDecoderLayer` state_dict via the production mapping.

  The bare layer's keys (e.g. `self_attn.q_proj.weight`) are re-rooted under
  `model.layers.{layer_idx}.` so the converter matches exactly rather than by suffix.
  """
  rooted = {f"model.layers.{layer_idx}.{k}": v for k, v in pt_layer.state_dict().items()}
  return convert_pytorch_module_to_maxtext_params(
      rooted,
      spec.mapping_config(),
      None,
      prefix_filter=f"params-decoder-layers_{layer_idx}-",
  )


def convert_hf_model_state_dict(pt_model: torch.nn.Module, spec: TinyKimiK3Spec) -> dict[str, np.ndarray]:
  """Converts a full `KimiLinearForCausalLM` state_dict via the production mapping."""
  return convert_pytorch_module_to_maxtext_params(pt_model.state_dict(), spec.mapping_config(), None)


def _path_to_key(path) -> str:
  """Flattens a PyTree path tuple into a hyphen-delimited parameter key."""
  parts = []
  for p in path:
    if hasattr(p, "key"):
      parts.append(str(p.key))
    elif hasattr(p, "name"):
      parts.append(str(p.name))
    elif hasattr(p, "idx"):
      parts.append(str(p.idx))
    else:
      parts.append(str(p))
  return "-".join(parts)


def load_params_into_nnx(
    module: nnx.Module,
    converted: dict[str, np.ndarray],
    prefix: str,
    *,
    strict: bool = True,
) -> list[str]:
  """Copies converted weights into `module` by walking its `nnx.Param` tree.

  Each `nnx.Param` reachable from `module` at path `a.b.c` is matched against
  `f"{prefix}-a-b-c"` in `converted`. Returns the list of keys loaded. With
  `strict=True`, raises if any param under `prefix` has no converted weight or if any
  converted key under `prefix` was never consumed.
  """
  loaded: list[str] = []
  missing: list[str] = []
  for path, node in nnx.iter_graph(module):
    if not isinstance(node, nnx.Param):
      continue
    key = f"{prefix}-{_path_to_key(path)}"
    if key not in converted:
      missing.append(key)
      continue
    src = np.asarray(converted[key])
    dst_shape = tuple(node.shape)
    if tuple(src.shape) != dst_shape:
      raise ValueError(f"shape mismatch for {key}: converted {src.shape} vs module {dst_shape}")
    node[...] = jnp.asarray(src, dtype=node.dtype)
    loaded.append(key)

  if strict:
    if missing:
      raise KeyError(f"no converted weight for module params: {missing}")
    unconsumed = sorted(k for k in converted if k.startswith(prefix + "-") and k not in loaded)
    if unconsumed:
      raise KeyError(f"converted keys never loaded into module: {unconsumed}")
  return loaded


# =============================================================================
# 4. Small numerics helpers
# =============================================================================
def hf_block_residual_to_maxtext(b_hf: torch.Tensor | np.ndarray, batch: int, seq: int) -> np.ndarray:
  """HF carries `[B*S, num_blocks, D]`; MaxText carries `[B, S, num_blocks, D]`."""
  arr = b_hf.detach().cpu().numpy() if isinstance(b_hf, torch.Tensor) else np.asarray(b_hf)
  return arr.reshape(batch, seq, arr.shape[-2], arr.shape[-1])


def causal_mask_pt(seq_len: int) -> torch.Tensor:
  """Additive `[1, 1, S, S]` causal mask for eager HF attention."""
  return torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)


def positions_and_segments(batch: int, seq: int) -> tuple[jax.Array, jax.Array]:
  positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (batch, seq))
  segment_ids = jnp.ones((batch, seq), dtype=jnp.int32)
  return positions, segment_ids


# -----------------------------------------------------------------------------
# logit parity assertion
# -----------------------------------------------------------------------------
# Explicit (stored) mantissa bits. The implicit leading 1 is not counted, so the
# spacing of representable values inside binade [2^e, 2^(e+1)) is 2^(e - bits).
_EXPLICIT_MANTISSA_BITS = {"bfloat16": 7, "float16": 10, "float32": 23}


def logit_ulp(value, dtype: str = "bfloat16"):
  """Spacing of representable `dtype` values at magnitude `value`.

  Computed per element from the *local* magnitude rather than from a global max:
  spacing doubles at every binade boundary, so a single global ULP would be wrong
  by up to 2x for any logit outside the top binade.
  """
  bits = _EXPLICIT_MANTISSA_BITS[dtype]
  mag = np.abs(np.asarray(value, dtype=np.float64))
  safe = np.maximum(mag, np.finfo(np.float64).tiny)
  return np.exp2(np.floor(np.log2(safe)) - bits)


def assert_logit_parity(
    logits_mt: np.ndarray,
    logits_pt: np.ndarray,
    dtype: str = "bfloat16",
    margin_ulps: float = 4.0,
    min_cosine: float = 0.999,
    min_pearson: float = 0.999,
    max_rank: int = 1,
    verbose: bool = True,
) -> dict[str, Any]:
  """Asserts MaxText logits match the PyTorch reference to within `dtype` precision.

  Why not simply require exact top-1 agreement
  --------------------------------------------
  The reference is itself computed in bf16. Where its own top-1/top-2 margin is only
  a couple of ULPs, its ordering is below its own quantization noise, so *no*
  independent implementation can be required to reproduce it -- the "correct" answer
  is not determined by the math, only by accumulation order. Worse, an exact tie in
  the candidate implementation is resolved by `argmax` index order, which is pure
  array-layout happenstance. Asserting a raw top-1 count would therefore encode an
  unsatisfiable requirement and would flake.

  Instead we assert the properties a correct implementation must have:

    1. Top-1 agreement at every position whose *reference* margin exceeds
       `margin_ulps` ULPs at that position's magnitude (a genuine, resolvable
       decision).
    2. The reference top-1 token ranks within `max_rank` in the candidate at
       *every* position, including the unresolvable ones.
    3. Global cosine and Pearson correlation above threshold.

  A real bug scatters disagreement across all margins; precision noise cannot
  produce a disagreement where the margin is comfortably resolvable.

  Returns a dict of computed statistics. Raises AssertionError listing every
  violation (not just the first) so a single run reports the full picture.
  """
  pt = np.asarray(logits_pt, dtype=np.float64)
  mt = np.asarray(logits_mt, dtype=np.float64)
  if pt.shape != mt.shape:
    raise ValueError(f"shape mismatch: maxtext {mt.shape} vs reference {pt.shape}")
  if pt.ndim == 3:
    if pt.shape[0] != 1:
      raise ValueError(f"expected batch 1, got {pt.shape[0]}")
    pt, mt = pt[0], mt[0]

  if not np.isfinite(mt).all():
    raise AssertionError("non-finite values in maxtext logits")

  seq = pt.shape[0]
  rows = []
  for i in range(seq):
    pt_sorted = np.sort(pt[i])[::-1]
    mt_sorted = np.sort(mt[i])[::-1]
    pt_top1 = int(pt[i].argmax())
    mt_top1 = int(mt[i].argmax())
    # Rank of the reference's chosen token within the candidate's ordering.
    rank = int(np.where(np.argsort(mt[i])[::-1] == pt_top1)[0][0])
    pt_gap = float(pt_sorted[0] - pt_sorted[1])
    ulp = float(logit_ulp(pt_sorted[0], dtype))
    rows.append(
        {
            "pos": i,
            "pt_top1": pt_top1,
            "mt_top1": mt_top1,
            "agree": pt_top1 == mt_top1,
            "pt_gap": pt_gap,
            "mt_gap": float(mt_sorted[0] - mt_sorted[1]),
            "ulp": ulp,
            "gap_ulps": pt_gap / ulp,
            "resolvable": pt_gap > margin_ulps * ulp,
            "rank": rank,
        }
    )

  flat_pt, flat_mt = pt.ravel(), mt.ravel()
  pearson = float(np.corrcoef(flat_pt, flat_mt)[0, 1])
  cosine = float(flat_pt @ flat_mt / (np.linalg.norm(flat_pt) * np.linalg.norm(flat_mt)))
  n_resolvable = sum(r["resolvable"] for r in rows)
  n_agree = sum(r["agree"] for r in rows)
  n_agree_resolvable = sum(r["agree"] for r in rows if r["resolvable"])

  if verbose:
    print(
        f"{'pos':>4} {'pt_top1':>8} {'mt_top1':>8} {'agree':>6} "
        f"{'pt_gap':>9} {'ulp':>9} {'gap/ulp':>8} {'resolv':>7} {'rank':>5}",
        flush=True,
    )
    for r in rows:
      print(
          f"{r['pos']:>4} {r['pt_top1']:>8} {r['mt_top1']:>8} {str(r['agree']):>6} "
          f"{r['pt_gap']:>9.4f} {r['ulp']:>9.5f} {r['gap_ulps']:>8.2f} "
          f"{str(r['resolvable']):>7} {r['rank']:>5}",
          flush=True,
      )
    print(
        f"top-1 agreement {n_agree}/{seq} overall, "
        f"{n_agree_resolvable}/{n_resolvable} on resolvable positions; "
        f"pearson {pearson:.6f} cosine {cosine:.6f}",
        flush=True,
    )

  failures = []
  for r in rows:
    if r["resolvable"] and not r["agree"]:
      failures.append(
          f"position {r['pos']}: top-1 mismatch at a resolvable margin "
          f"({r['gap_ulps']:.1f} ULPs > {margin_ulps}); "
          f"reference {r['pt_top1']} vs maxtext {r['mt_top1']}"
      )
    if r["rank"] > max_rank:
      failures.append(
          f"position {r['pos']}: reference top-1 token {r['pt_top1']} ranks "
          f"{r['rank']} in maxtext (allowed <= {max_rank})"
      )
  if pearson < min_pearson:
    failures.append(f"pearson {pearson:.6f} < {min_pearson}")
  if cosine < min_cosine:
    failures.append(f"cosine {cosine:.6f} < {min_cosine}")
  if n_resolvable == 0:
    failures.append(
        "no position had a resolvable margin, so top-1 agreement was never actually "
        "tested; use a prompt or model depth that produces confident predictions"
    )

  if failures:
    raise AssertionError("logit parity failed:\n  " + "\n  ".join(failures))

  return {
      "rows": rows,
      "pearson": pearson,
      "cosine": cosine,
      "n_agree": n_agree,
      "n_agree_resolvable": n_agree_resolvable,
      "n_resolvable": n_resolvable,
      "seq": seq,
  }
