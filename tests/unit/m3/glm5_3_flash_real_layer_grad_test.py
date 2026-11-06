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

"""Real-weight per-layer-type gradient parity test suite for GLM-5.3-Flash."""

# pylint: disable=protected-access

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import contextlib
import gc
import json
import os
from typing import Any
import unittest
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
from tests.utils import glm5_test_utils
from tests.utils import zero_gen_logits_checker
import numpy as np
import pytest

try:
  from maxtext.m3.models.glm5_3_flash import glm5_ckpt
  import safetensors
  import torch
  import transformers
  from transformers.integrations import moe as hf_moe_integ
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]

_assert_e3 = glm5_test_utils._assert_e3
_ensure_real_mini_slice = glm5_test_utils._ensure_real_mini_slice
_fp64_mode = glm5_test_utils._fp64_mode
_hf_module_state_to_hub = glm5_test_utils._hf_module_state_to_hub
_init_hf = glm5_test_utils._init_hf
_LazySafetensorsMapping = glm5_test_utils._LazySafetensorsMapping


def _lookup_key(keys: set[str], norm_k: str) -> str | None:
  for cand in (norm_k, f"model.language_model.{norm_k}", f"model.{norm_k}"):
    if cand in keys:
      return cand
  return None


def _get_st_tensor(
    st_mapping: Mapping[str, Any],
    norm_k: str,
    target_dtype: torch.dtype,
    row_slice: slice | None = None,
) -> torch.Tensor:
  """Extracts tensor from safetensors mapping, applying block-128 dequant if FP8."""
  keys = set(st_mapping.keys())
  actual_k = _lookup_key(keys, norm_k)
  if actual_k is None:
    raise KeyError(f"Missing parameter in safetensors: {norm_k!r}")
  scale_k = _lookup_key(
      keys,
      norm_k[:-7] + ".weight_scale_inv" if norm_k.endswith(".weight") else norm_k + "_scale_inv",
  )
  if row_slice is not None and hasattr(st_mapping, "get_slice") and scale_k is None:
    val = st_mapping.get_slice(actual_k, row_slice)
  else:
    val = st_mapping[actual_k]
    if row_slice is not None:
      val = val[row_slice]
  if scale_k is not None:
    scale = st_mapping[scale_k]
    if row_slice is not None:
      scale = scale[row_slice]
    jnp_dt = jnp.float64 if target_dtype == torch.float64 else jnp.float32
    deq = glm5_ckpt.dequantize_fp8_block128(val, scale, target_dtype=jnp_dt)
    return torch.from_numpy(np.asarray(deq)).to(target_dtype)
  if isinstance(val, torch.Tensor):
    return val.detach().to(target_dtype)
  return torch.from_numpy(np.asarray(val)).to(target_dtype)


_RUN_REAL_LAYER_GRAD = flags.DEFINE_bool(
    "run_real_layer_grad",
    False,
    "Whether to run real-weight per-layer gradient parity test.",
)
_REAL_LAYER_CASE = flags.DEFINE_enum(
    "real_layer_case",
    "all",
    ("all", "l0_kda_dense", "l3_mla_moe", "l4_kda_moe"),
    "Which layer case to run.",
)
_RUN_REAL_LAYER_FP64 = flags.DEFINE_bool(
    "run_real_layer_fp64",
    False,
    "Whether to also run float64 parity pass asserting frob_rel <= 1e-10.",
)
_MINI_SLICE_DIR = flags.DEFINE_string(
    "mini_slice_dir",
    glm5_test_utils.MINI_SLICE_DIR,
    "Path to 5-layer mini-slice directory containing model.safetensors and config.json.",
    allow_override=True,
)


class _LazyTensorDict(Mapping[str, Any]):
  """Lazy mapping executing tensor getters on demand without caching in memory."""

  def __init__(self, getters: dict[str, Callable[[], torch.Tensor]]):
    self._getters = getters

  def __getitem__(self, key: str) -> torch.Tensor:
    return self._getters[key]()

  def __contains__(self, key: object) -> bool:
    return key in self._getters

  def __len__(self) -> int:
    return len(self._getters)

  def __iter__(self) -> Iterator[str]:
    return iter(self._getters)

  def keys(self):
    return self._getters.keys()


def _build_1layer_tensors(
    st_mapping: Mapping[str, Any],
    case: str,
    target_dtype: torch.dtype,
    n_routed_experts: int = 32,
    hub_prefix: bool = False,
) -> _LazyTensorDict:
  """Constructs a lazy 1-layer tensor state dict from checkpoint mapping."""
  st_keys = set(st_mapping.keys())
  if case == "l0_kda_dense":
    src_l = 0
    is_moe = False
    is_linear = True
  elif case == "l3_mla_moe":
    src_l = 3
    is_moe = True
    is_linear = False
  elif case == "l4_kda_moe":
    src_l = 4
    is_moe = True
    is_linear = True
  else:
    raise ValueError(f"Unknown real layer case: {case!r}")

  def _map_key(k: str) -> str:
    if not hub_prefix:
      return k
    if k.startswith(("model.language_model.", "lm_head.")):
      return k
    return f"model.language_model.{k}"

  getters: dict[str, Callable[[], torch.Tensor]] = {}

  getters[_map_key("embed_tokens.weight")] = lambda: _get_st_tensor(st_mapping, "embed_tokens.weight", target_dtype)
  getters[_map_key("norm.weight")] = lambda: _get_st_tensor(st_mapping, "norm.weight", target_dtype)
  getters[_map_key("lm_head.weight")] = lambda: _get_st_tensor(st_mapping, "lm_head.weight", target_dtype)

  getters[_map_key("layers.0.input_layernorm.weight")] = lambda: _get_st_tensor(
      st_mapping, f"layers.{src_l}.input_layernorm.weight", target_dtype
  )
  getters[_map_key("layers.0.post_attention_layernorm.weight")] = lambda: _get_st_tensor(
      st_mapping,
      f"layers.{src_l}.post_attention_layernorm.weight",
      target_dtype,
  )

  for pre in ("attn", "ffn"):
    for s in ("fn", "base", "scale"):
      getters[_map_key(f"layers.0.hc_{pre}_{s}")] = lambda pre=pre, s=s: _get_st_tensor(
          st_mapping, f"layers.{src_l}.hc_{pre}_{s}", target_dtype
      )

  if is_linear:
    for p in ("q", "k", "v", "b", "o", "g_a", "g_b"):
      getters[_map_key(f"layers.0.self_attn.{p}_proj.weight")] = lambda p=p: _get_st_tensor(
          st_mapping,
          f"layers.{src_l}.self_attn.{p}_proj.weight",
          target_dtype,
      )
    fa_k = f"layers.{src_l}.self_attn.forget_gate.f_a_proj.weight"
    if _lookup_key(st_keys, fa_k) is None:
      fa_k = f"layers.{src_l}.self_attn.f_a_proj.weight"
    getters[_map_key("layers.0.self_attn.f_a_proj.weight")] = lambda k=fa_k: _get_st_tensor(st_mapping, k, target_dtype)

    fb_k = f"layers.{src_l}.self_attn.forget_gate.f_b_proj.weight"
    if _lookup_key(st_keys, fb_k) is None:
      fb_k = f"layers.{src_l}.self_attn.f_b_proj.weight"
    getters[_map_key("layers.0.self_attn.f_b_proj.weight")] = lambda k=fb_k: _get_st_tensor(st_mapping, k, target_dtype)

    a_log_k = f"layers.{src_l}.self_attn.forget_gate.A_log"
    if _lookup_key(st_keys, a_log_k) is None:
      a_log_k = f"layers.{src_l}.self_attn.A_log"
    getters[_map_key("layers.0.self_attn.A_log")] = lambda k=a_log_k: _get_st_tensor(st_mapping, k, target_dtype)

    dt_k = f"layers.{src_l}.self_attn.forget_gate.dt_bias"
    if _lookup_key(st_keys, dt_k) is None:
      dt_k = f"layers.{src_l}.self_attn.dt_bias"
    getters[_map_key("layers.0.self_attn.dt_bias")] = lambda k=dt_k: _get_st_tensor(st_mapping, k, target_dtype)

    getters[_map_key("layers.0.self_attn.o_norm.weight")] = lambda: _get_st_tensor(
        st_mapping, f"layers.{src_l}.self_attn.o_norm.weight", target_dtype
    )

    qc_k = f"layers.{src_l}.self_attn.q_conv1d.weight"
    if _lookup_key(st_keys, qc_k) is not None:
      for c in ("q", "k", "v"):
        getters[_map_key(f"layers.0.self_attn.{c}_conv1d.weight")] = lambda c=c: _get_st_tensor(
            st_mapping,
            f"layers.{src_l}.self_attn.{c}_conv1d.weight",
            target_dtype,
        )
    else:
      cw_k = f"layers.{src_l}.self_attn.conv1d.weight"

      def _get_conv1d_third(idx: int) -> torch.Tensor:
        cw = _get_st_tensor(st_mapping, cw_k, target_dtype)
        k_dim = cw.shape[0] // 3
        if idx == 0:
          return cw[:k_dim]
        elif idx == 1:
          return cw[k_dim : 2 * k_dim]
        else:
          return cw[2 * k_dim :]

      getters[_map_key("layers.0.self_attn.q_conv1d.weight")] = lambda: _get_conv1d_third(0)
      getters[_map_key("layers.0.self_attn.k_conv1d.weight")] = lambda: _get_conv1d_third(1)
      getters[_map_key("layers.0.self_attn.v_conv1d.weight")] = lambda: _get_conv1d_third(2)
  else:
    for p in ("q_a", "q_b", "kv_b", "o"):
      getters[_map_key(f"layers.0.self_attn.{p}_proj.weight")] = lambda p=p: _get_st_tensor(
          st_mapping,
          f"layers.{src_l}.self_attn.{p}_proj.weight",
          target_dtype,
      )
    getters[_map_key("layers.0.self_attn.kv_a_proj_with_mqa.weight")] = lambda: _get_st_tensor(
        st_mapping,
        f"layers.{src_l}.self_attn.kv_a_proj_with_mqa.weight",
        target_dtype,
    )
    for p in ("q_a", "kv_a"):
      getters[_map_key(f"layers.0.self_attn.{p}_layernorm.weight")] = lambda p=p: _get_st_tensor(
          st_mapping,
          f"layers.{src_l}.self_attn.{p}_layernorm.weight",
          target_dtype,
      )
    getters[_map_key("layers.0.self_attn.indexer.wq_b.weight")] = lambda: _get_st_tensor(
        st_mapping,
        f"layers.{src_l}.self_attn.indexer.wq_b.weight",
        target_dtype,
    )
    for p in ("wk", "weights_proj"):
      getters[_map_key(f"layers.0.self_attn.indexer.{p}.weight")] = lambda p=p: _get_st_tensor(
          st_mapping,
          f"layers.{src_l}.self_attn.indexer.{p}.weight",
          target_dtype,
      )
    for p in ("weight", "bias"):
      getters[_map_key(f"layers.0.self_attn.indexer.k_norm.{p}")] = lambda p=p: _get_st_tensor(
          st_mapping,
          f"layers.{src_l}.self_attn.indexer.k_norm.{p}",
          target_dtype,
      )
    for p in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
      getters[_map_key(f"layers.0.self_attn.indexer.{p}")] = lambda p=p: _get_st_tensor(
          st_mapping, f"layers.{src_l}.self_attn.indexer.{p}", target_dtype
      )

  if not is_moe:
    for p in ("gate", "up", "down"):
      getters[_map_key(f"layers.0.mlp.{p}_proj.weight")] = lambda p=p: _get_st_tensor(
          st_mapping, f"layers.{src_l}.mlp.{p}_proj.weight", target_dtype
      )
  else:
    for p in ("gate", "up", "down"):
      getters[_map_key(f"layers.0.mlp.shared_experts.{p}_proj.weight")] = lambda p=p: _get_st_tensor(
          st_mapping,
          f"layers.{src_l}.mlp.shared_experts.{p}_proj.weight",
          target_dtype,
      )
    gw_k = f"layers.{src_l}.mlp.gate.weight"
    getters[_map_key("layers.0.mlp.gate.weight")] = lambda k=gw_k: _get_st_tensor(
        st_mapping, k, target_dtype, row_slice=slice(0, n_routed_experts)
    )
    gb_k = f"layers.{src_l}.mlp.gate.e_score_correction_bias"
    getters[_map_key("layers.0.mlp.gate.e_score_correction_bias")] = lambda k=gb_k: _get_st_tensor(
        st_mapping, k, target_dtype, row_slice=slice(0, n_routed_experts)
    )
    for e in range(n_routed_experts):
      for p in ("gate", "up", "down"):
        ep_k = f"layers.{src_l}.mlp.experts.{e}.{p}_proj.weight"
        getters[_map_key(f"layers.0.mlp.experts.{e}.{p}_proj.weight")] = lambda k=ep_k: _get_st_tensor(
            st_mapping, k, target_dtype
        )

  return _LazyTensorDict(getters)


def _load_1layer_into_hf(
    hf_m: torch.nn.Module,
    hf_head: torch.nn.Module,
    tensors: Mapping[str, torch.Tensor],
) -> None:
  """Loads extracted 1-layer tensor dictionary into HuggingFace model and head."""
  with torch.no_grad():
    if "lm_head.weight" in tensors:
      hf_head.weight.copy_(tensors["lm_head.weight"])
    for n, p in list(hf_m.named_parameters()) + list(hf_m.named_buffers()):
      norm_n = n
      for prefix in ("model.language_model.", "model."):
        if norm_n.startswith(prefix):
          norm_n = norm_n[len(prefix) :]
          break
      if norm_n in ("embed_tokens.weight", "norm.weight"):
        p.copy_(tensors[norm_n])
        continue
      if norm_n.startswith("layers.0."):
        rest = norm_n[len("layers.0.") :]
        if rest.startswith("attn_hc."):
          attr = rest[len("attn_hc.") :]
          p.copy_(tensors[f"layers.0.hc_attn_{attr}"])
          continue
        if rest.startswith("ffn_hc."):
          attr = rest[len("ffn_hc.") :]
          p.copy_(tensors[f"layers.0.hc_ffn_{attr}"])
          continue
        if rest.startswith("self_attn.forget_gate."):
          attr = rest[len("self_attn.forget_gate.") :]
          p.copy_(tensors[f"layers.0.self_attn.{attr}"])
          continue
        if rest == "self_attn.conv1d.weight":
          q = tensors["layers.0.self_attn.q_conv1d.weight"]
          k = tensors["layers.0.self_attn.k_conv1d.weight"]
          v = tensors["layers.0.self_attn.v_conv1d.weight"]
          p.copy_(torch.cat([q, k, v], dim=0))
          del q, k, v
          continue
        if rest == "mlp.experts.gate_up_proj":
          ne = p.shape[0]
          gu_list = []
          for e in range(ne):
            ge = tensors[f"layers.0.mlp.experts.{e}.gate_proj.weight"]
            up_e = tensors[f"layers.0.mlp.experts.{e}.up_proj.weight"]
            gu_list.append(torch.cat([ge, up_e], dim=0))
          p.copy_(torch.stack(gu_list, dim=0))
          del gu_list
          continue
        if rest == "mlp.experts.down_proj":
          ne = p.shape[0]
          dp_list = [tensors[f"layers.0.mlp.experts.{e}.down_proj.weight"] for e in range(ne)]
          p.copy_(torch.stack(dp_list, dim=0))
          del dp_list
          continue
        p.copy_(tensors[f"layers.0.{rest}"])
      else:
        p.copy_(tensors[norm_n])


def _construct_mini_configs(
    num_hidden_layers: int = 5,
    scan_layers: bool = False,
    dtype: str = "float32",
    weight_dtype: str = "float32",
    index_topk: int = 16,
    enable_debug_stats: bool = False,
) -> tuple[glm5_config.Glm5Config, hf_cfg.Glm5NextTextConfig]:
  """Constructs aligned JAX and HuggingFace model configurations with mini dimensions."""
  kws = (
      "hidden_size=64 intermediate_size=128 moe_intermediate_size=32"
      " n_routed_experts=48 num_experts_per_tok=8 q_lora_rank=32"
      " kv_lora_rank=16 num_attention_heads=4 num_key_value_heads=4"
      " qk_nope_head_dim=16 v_head_dim=16 index_n_heads=32 index_head_dim=16"
      " linear_num_key_heads=4 linear_num_value_heads=4"
      " linear_key_head_dim=16 linear_value_head_dim=16"
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
  hf_args = {x: args[x] for x in args if not x.startswith("linear_") and x not in ("qk_head_dim", "enable_debug_stats")}
  hf_k = (
      "vocab_size pad_token_id num_hidden_layers _attn_implementation "
      "index_kpool linear_num_heads linear_head_dim linear_conv_kernel_dim"
  ).split()
  hf_args.update(zip(hf_k, (256, 0, num_hidden_layers, "eager", 4, 4, 16, 4)))
  hf_args["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * (num_hidden_layers - 3)
  hf_args["layer_types"] = list(c.layer_types)
  hf_args["indexer_types"] = list(c.indexer_types)
  return c, hf_cfg.Glm5NextTextConfig(**hf_args)


def _create_synthetic_checkpoint(tempdir: str) -> str:
  """Builds a scaled synthetic safetensors checkpoint and config for hermetic testing."""
  _, hf_c = _construct_mini_configs(num_hidden_layers=5)
  hf_m = hf_model.Glm5NextTextModel(hf_c)
  hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False)
  torch.manual_seed(42)
  _init_hf(hf_m, hf_head)

  state = dict(hf_m.state_dict())
  state["lm_head.weight"] = hf_head.weight
  hub_state = _hf_module_state_to_hub(
      state,
      num_experts=hf_c.n_routed_experts,
      moe_intermediate_size=hf_c.moe_intermediate_size,
  )

  st_path = os.path.join(tempdir, "model.safetensors")
  safetensors.torch.save_file(hub_state, st_path)

  cfg_dict = {"text_config": hf_c.to_dict()}
  with open(os.path.join(tempdir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg_dict, f, indent=2)

  return tempdir


def _run_hf_stage(
    st_path: str,
    hf_c: hf_cfg.Glm5NextTextConfig,
    case: str,
    torch_dt: torch.dtype,
    is_fp64: bool,
    ids_t: torch.Tensor,
    mask_t: torch.Tensor,
    loss_mask_t: torch.Tensor,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
  """Runs Stage 1 (HF) forward and backward, returning loss, logits, and grads while freeing memory."""
  lazy_st = _LazySafetensorsMapping(st_path)
  lazy_tensors = _build_1layer_tensors(lazy_st, case, torch_dt, n_routed_experts=32, hub_prefix=False)

  with _fp64_mode() if is_fp64 else contextlib.nullcontext():
    hf_m = hf_model.Glm5NextTextModel(hf_c).to(torch_dt)
    hf_head = torch.nn.Linear(hf_c.hidden_size, hf_c.vocab_size, bias=False).to(torch_dt)
    _load_1layer_into_hf(hf_m, hf_head, lazy_tensors)

    lazy_st.close()
    del lazy_tensors, lazy_st
    gc.collect()

    orig_kda = hf_model.chunk_kimi_delta_attention
    hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
    with mock.patch.object(hf_moe_integ, "_can_use_grouped_mm", return_value=False):
      try:
        hf_logits = hf_head(hf_m(input_ids=ids_t, attention_mask=mask_t, use_cache=False).last_hidden_state)
      finally:
        hf_model.chunk_kimi_delta_attention = orig_kda

    hf_pred = hf_logits[:, :-1, :].reshape(-1, hf_c.vocab_size)
    hf_tgt = ids_t[:, 1:].reshape(-1)
    hf_ce_all = torch.nn.functional.cross_entropy(hf_pred, hf_tgt, reduction="none").reshape(ids_t.shape[0], -1)
    hf_loss = (hf_ce_all * loss_mask_t).sum() / loss_mask_t.sum()
    hf_loss_val = float(hf_loss.detach().cpu().numpy())
    hf_loss.backward()

    hf_logits_np = hf_logits.detach().cpu().numpy().copy()
    hf_grads: dict[str, np.ndarray] = {}
    if hf_head.weight.grad is not None:
      hf_grads["lm_head.weight"] = hf_head.weight.grad.detach().cpu().numpy().copy()
      hf_head.weight.grad = None
    for n, p in list(hf_m.named_parameters()):
      norm_n = n
      for prefix in ("model.language_model.", "model."):
        if norm_n.startswith(prefix):
          norm_n = norm_n[len(prefix) :]
          break
      if p.grad is not None:
        hf_grads[norm_n] = p.grad.detach().cpu().numpy().copy()
        p.grad = None

    del hf_m, hf_head, hf_logits, hf_pred, hf_tgt, hf_ce_all, hf_loss
    gc.collect()

  return hf_loss_val, hf_logits_np, hf_grads


def _run_single_layer_grad_pass(
    test_case: absltest.TestCase,
    slice_dir: str,
    case: str,
    is_fp64: bool,
    is_synthetic: bool = False,
) -> None:
  """Executes sequential HF vs JAX forward and gradient parity for a single layer case."""
  with open(os.path.join(slice_dir, "config.json"), "r", encoding="utf-8") as f:
    cfg_data = json.load(f)
  base_text_cfg = dict(cfg_data["text_config"])

  torch_dt = torch.float64 if is_fp64 else torch.float32
  jnp_dt_str = "float64" if is_fp64 else "float32"

  hf_text_cfg = dict(base_text_cfg)
  hf_text_cfg["num_hidden_layers"] = 1
  hf_text_cfg["_attn_implementation"] = "eager"
  hf_text_cfg["use_cache"] = False
  if case == "l0_kda_dense":
    hf_text_cfg["first_k_dense_replace"] = 1
    hf_text_cfg["layer_types"] = ["linear_attention"]
    hf_text_cfg["indexer_types"] = ["full"]
    hf_text_cfg["mlp_layer_types"] = ["dense"]
  elif case == "l3_mla_moe":
    hf_text_cfg["first_k_dense_replace"] = 0
    hf_text_cfg["layer_types"] = ["full_attention"]
    hf_text_cfg["indexer_types"] = ["full"]
    hf_text_cfg["mlp_layer_types"] = ["sparse"]
    hf_text_cfg["n_routed_experts"] = 32
    hf_text_cfg["num_experts_per_tok"] = 8
    hf_text_cfg["index_topk"] = 4
  elif case == "l4_kda_moe":
    hf_text_cfg["first_k_dense_replace"] = 0
    hf_text_cfg["layer_types"] = ["linear_attention"]
    hf_text_cfg["indexer_types"] = ["full"]
    hf_text_cfg["mlp_layer_types"] = ["sparse"]
    hf_text_cfg["n_routed_experts"] = 32
    hf_text_cfg["num_experts_per_tok"] = 8
  else:
    raise ValueError(f"Unknown real layer case: {case!r}")

  hf_c = hf_cfg.Glm5NextTextConfig(**hf_text_cfg)

  st_path = os.path.join(slice_dir, "model.safetensors")

  vocab_size = hf_c.vocab_size
  if is_synthetic:
    seq_len = 64 if case == "l3_mla_moe" else 16
    batch_size = 2
    tokens = np.random.RandomState(42).randint(1, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    test_case.assertTrue(np.all(tokens != 0), "Synthetic tokens must not contain pad token id 0.")
    seg_ids = np.ones((batch_size, seq_len), dtype=np.int32)
    loss_mask_np = np.ones((batch_size, seq_len - 1), dtype=bool)
  else:
    seq_len = 64 if case == "l3_mla_moe" else 32
    batch_size = 2
    tokenizer = transformers.AutoTokenizer.from_pretrained(slice_dir, trust_remote_code=True)
    tokens, seg_ids, mask = zero_gen_logits_checker.tokenize_golden_batch(
        zero_gen_logits_checker.GOLDEN_BATCH_TEXT,
        tokenizer=tokenizer,
        seq_len=seq_len,
        pad_token_id=154820,
        batch_size=batch_size,
        target_start_fraction=0.5,
        valid_lengths=[seq_len, seq_len - 4],
    )
    loss_mask_np = mask[:, 1:] & (seg_ids[:, 1:] > 0)

  ids_t = torch.from_numpy(tokens).long()
  mask_t = torch.from_numpy(seg_ids > 0).bool()
  loss_mask_t = torch.from_numpy(loss_mask_np)

  # Stage 1: Build and evaluate HuggingFace model
  hf_loss_val, hf_logits_np, hf_grads = _run_hf_stage(
      st_path=st_path,
      hf_c=hf_c,
      case=case,
      torch_dt=torch_dt,
      is_fp64=is_fp64,
      ids_t=ids_t,
      mask_t=mask_t,
      loss_mask_t=loss_mask_t,
  )
  del ids_t, mask_t, loss_mask_t
  gc.collect()

  # Stage 2: Build and evaluate JAX model
  jax_c = glm5_config.Glm5Config(
      num_hidden_layers=1,
      first_k_dense_replace=1 if case == "l0_kda_dense" else 0,
      layer_types=("linear_attention",) if case != "l3_mla_moe" else ("full_attention",),
      indexer_types=("full",),
      n_routed_experts=32 if case != "l0_kda_dense" else hf_c.n_routed_experts,
      num_experts_per_tok=hf_c.num_experts_per_tok,
      num_nextn_predict_layers=0,
      scan_layers=False,
      dtype=jnp_dt_str,
      weight_dtype=jnp_dt_str,
      kda_implementation="xla",
      hidden_size=hf_c.hidden_size,
      intermediate_size=hf_c.intermediate_size,
      moe_intermediate_size=hf_c.moe_intermediate_size,
      vocab_size=hf_c.vocab_size,
      num_attention_heads=hf_c.num_attention_heads,
      num_key_value_heads=hf_c.num_key_value_heads,
      linear_num_value_heads=hf_c.linear_num_heads,
      linear_value_head_dim=hf_c.linear_head_dim,
      linear_num_key_heads=hf_c.linear_num_heads,
      linear_key_head_dim=hf_c.linear_head_dim,
      linear_conv_kernel_dim=hf_c.linear_conv_kernel_dim,
      q_lora_rank=hf_c.q_lora_rank,
      kv_lora_rank=hf_c.kv_lora_rank,
      qk_nope_head_dim=hf_c.qk_nope_head_dim,
      v_head_dim=hf_c.v_head_dim,
      index_n_heads=hf_c.index_n_heads,
      index_head_dim=hf_c.index_head_dim,
      index_topk=hf_c.index_topk,
      index_kpool=hf_c.index_kpool,
      mhc_expansion_rate=hf_c.hc_mult,
      mhc_sinkhorn_iters=hf_c.hc_sinkhorn_iters,
      mhc_eps=hf_c.hc_eps,
      rms_norm_eps=hf_c.rms_norm_eps,
      routed_scaling_factor=hf_c.routed_scaling_factor,
      norm_topk_prob=hf_c.norm_topk_prob,
      mlp_clamp_limit=hf_c.swiglu_limit,
  )

  try:
    with _fp64_mode() if is_fp64 else contextlib.nullcontext():
      jax_m = modeling_glm5_3_flash.Glm5Model(jax_c, rngs=nnx.Rngs(0), place_on_devices=False)
      lazy_st = _LazySafetensorsMapping(st_path)
      lazy_hub_tensors = _build_1layer_tensors(lazy_st, case, torch_dt, n_routed_experts=32, hub_prefix=True)
      glm5_ckpt.load_hf_state_into_model(jax_m, lazy_hub_tensors, ignore_mtp_layer=True)

      lazy_st.close()
      del lazy_hub_tensors, lazy_st
      gc.collect()

      tokens_jax = jnp.array(tokens)
      seg_ids_jax = jnp.array(seg_ids)
      loss_mask_jax = jnp.array(loss_mask_np)

      def _loss_fn(model):
        logits = model(tokens_jax, decoder_segment_ids=seg_ids_jax)
        pred_logits = logits[:, :-1, :]
        target_tokens = tokens_jax[:, 1:]
        log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
        target_lp = jnp.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)
        loss = -jnp.sum(target_lp * loss_mask_jax) / jnp.sum(loss_mask_jax)
        return loss, logits

      (jax_loss, jax_logits), jax_grads = nnx.value_and_grad(_loss_fn, has_aux=True)(jax_m)
      del jax_m
      gc.collect()

      # Compare forward logits
      valid_tokens = seg_ids > 0
      l_frob_rel, _, _ = _assert_e3(
          np.asarray(jax_logits)[valid_tokens],
          hf_logits_np[valid_tokens],
          name=f"forward_logits_{case}",
          is_fp64=is_fp64,
      )

      del jax_logits, hf_logits_np
      gc.collect()

      # Compare masked CE loss
      if is_fp64:
        ce_diff = abs(float(jax_loss) - hf_loss_val) / max(1.0, abs(hf_loss_val))
        test_case.assertLessEqual(ce_diff, 1e-10, f"CE loss mismatch in fp64: {ce_diff:.3e}")
      else:
        np.testing.assert_allclose(float(jax_loss), hf_loss_val, atol=1e-5, rtol=1e-5)

      # Compare gradients
      num_params_checked = 0
      active_experts_set: set[int] = set()
      min_grad_cos = 1.0
      max_grad_frob_rel = 0.0

      def _check_param(
          name: str,
          jax_arr: Any,
          hf_arr: Any,
          is_expert: bool = False,
          expert_idx: int = -1,
      ) -> None:
        nonlocal num_params_checked, min_grad_cos, max_grad_frob_rel
        aj = np.asarray(jax_arr)
        ah = np.asarray(hf_arr)
        test_case.assertEqual(
            aj.shape,
            ah.shape,
            f"Shape mismatch for {name}: {aj.shape} vs {ah.shape}",
        )
        norm_j = float(np.linalg.norm(aj))
        norm_h = float(np.linalg.norm(ah))

        if is_expert:
          if norm_h < 1e-8 and norm_j < 1e-8:
            test_case.assertLess(norm_j, 1e-8)
            test_case.assertLess(norm_h, 1e-8)
            return
          active_experts_set.add(expert_idx)

        test_case.assertGreater(norm_j, 0.0, f"JAX grad is zero for {name}")
        test_case.assertGreater(norm_h, 0.0, f"HF grad is zero for {name}")
        frob, _, cos = _assert_e3(aj, ah, name=name, is_fp64=is_fp64)
        num_params_checked += 1
        min_grad_cos = min(min_grad_cos, cos)
        max_grad_frob_rel = max(max_grad_frob_rel, frob)

      layer_prefix = "layers.0."
      checked_conv1d: set[str] = set()
      flat_grads = nnx.to_flat_state(jax_grads)
      for path, val in flat_grads:  # pylint: disable=too-many-nested-blocks
        g_jax = val.value
        try:
          if path == ("embed_tokens", "embedding"):
            _check_param(
                "embed_tokens.weight",
                g_jax,
                hf_grads.pop("embed_tokens.weight"),
            )
          elif path == ("norm", "weight"):
            _check_param("norm.weight", g_jax, hf_grads.pop("norm.weight"))
          elif path == ("lm_head", "kernel"):
            _check_param("lm_head.weight", g_jax.T, hf_grads.pop("lm_head.weight"))
          elif path[0] in ("prefix_layers", "suffix_layers"):
            rest = path[2:]
            lp = layer_prefix
            if rest == ("attn_hc", "fn"):
              _check_param(f"{lp}attn_hc.fn", g_jax.T, hf_grads.pop(f"{lp}attn_hc.fn"))
            elif rest == ("attn_hc", "base"):
              _check_param(f"{lp}attn_hc.base", g_jax, hf_grads.pop(f"{lp}attn_hc.base"))
            elif rest == ("attn_hc", "scale"):
              _check_param(
                  f"{lp}attn_hc.scale",
                  g_jax,
                  hf_grads.pop(f"{lp}attn_hc.scale"),
              )
            elif rest == ("ffn_hc", "fn"):
              _check_param(f"{lp}ffn_hc.fn", g_jax.T, hf_grads.pop(f"{lp}ffn_hc.fn"))
            elif rest == ("ffn_hc", "base"):
              _check_param(f"{lp}ffn_hc.base", g_jax, hf_grads.pop(f"{lp}ffn_hc.base"))
            elif rest == ("ffn_hc", "scale"):
              _check_param(f"{lp}ffn_hc.scale", g_jax, hf_grads.pop(f"{lp}ffn_hc.scale"))
            elif rest == ("input_layernorm", "weight"):
              _check_param(
                  f"{lp}input_layernorm.weight",
                  g_jax,
                  hf_grads.pop(f"{lp}input_layernorm.weight"),
              )
            elif rest == ("post_attention_layernorm", "weight"):
              _check_param(
                  f"{lp}post_attention_layernorm.weight",
                  g_jax,
                  hf_grads.pop(f"{lp}post_attention_layernorm.weight"),
              )
            elif rest[0] == "self_attn":
              if rest[1] == "indexer":
                norm_idx = float(np.linalg.norm(np.asarray(g_jax)))
                test_case.assertLess(
                    norm_idx,
                    1e-6,
                    f"Indexer grad non-zero for {lp}indexer.{rest}: {norm_idx}",
                )
              elif rest[1] in ("q_conv1d", "k_conv1d", "v_conv1d"):
                c_char = rest[1][0]
                checked_conv1d.add(c_char)
                c_idx = {"q": 0, "k": 1, "v": 2}[c_char]
                dim = g_jax.shape[0]
                hf_conv = hf_grads[f"{layer_prefix}self_attn.conv1d.weight"][c_idx * dim : (c_idx + 1) * dim]
                _check_param(
                    f"{layer_prefix}self_attn.{c_char}_conv1d.weight",
                    g_jax,
                    hf_conv,
                )
              elif rest[1] in ("f_a_proj", "f_b_proj"):
                fg_k = f"{lp}self_attn.forget_gate.{rest[1]}.weight"
                if fg_k not in hf_grads:
                  fg_k = f"{lp}self_attn.{rest[1]}.weight"
                _check_param(f"{lp}self_attn.{rest[1]}", g_jax.T, hf_grads.pop(fg_k))
              elif rest[1] == "A_log":
                a_log_k = f"{lp}self_attn.forget_gate.A_log"
                if a_log_k not in hf_grads:
                  a_log_k = f"{lp}self_attn.A_log"
                _check_param(f"{lp}self_attn.A_log", g_jax, hf_grads.pop(a_log_k))
              elif rest[1] == "dt_bias":
                dt_k = f"{lp}self_attn.forget_gate.dt_bias"
                if dt_k not in hf_grads:
                  dt_k = f"{lp}self_attn.dt_bias"
                _check_param(f"{lp}self_attn.dt_bias", g_jax, hf_grads.pop(dt_k))
              elif rest[1] == "o_norm":
                _check_param(
                    f"{lp}self_attn.o_norm.weight",
                    g_jax,
                    hf_grads.pop(f"{lp}self_attn.o_norm.weight"),
                )
              elif rest[1] in ("q_a_layernorm", "kv_a_layernorm"):
                _check_param(
                    f"{lp}self_attn.{rest[1]}.weight",
                    g_jax,
                    hf_grads.pop(f"{lp}self_attn.{rest[1]}.weight"),
                )
              elif rest[1] in (
                  "q_proj",
                  "k_proj",
                  "v_proj",
                  "b_proj",
                  "o_proj",
                  "g_a_proj",
                  "g_b_proj",
                  "q_a_proj",
                  "q_b_proj",
                  "kv_b_proj",
                  "kv_a_proj_with_mqa",
              ):
                _check_param(
                    f"{lp}self_attn.{rest[1]}.kernel",
                    g_jax.T,
                    hf_grads.pop(f"{lp}self_attn.{rest[1]}.weight"),
                )
              else:
                raise ValueError(f"Unhandled self_attn parameter: {path!r}")
            elif rest[0] == "mlp":
              if rest[1] in ("gate_proj", "up_proj", "down_proj"):
                _check_param(
                    f"{lp}mlp.{rest[1]}.kernel",
                    g_jax.T,
                    hf_grads.pop(f"{lp}mlp.{rest[1]}.weight"),
                )
              elif rest[1] == "shared_experts":
                p_name = rest[2]
                _check_param(
                    f"{lp}mlp.shared_experts.{p_name}.kernel",
                    g_jax.T,
                    hf_grads.pop(f"{lp}mlp.shared_experts.{p_name}.weight"),
                )
              elif rest[1] == "gate":
                _check_param(
                    f"{lp}mlp.gate.weight",
                    g_jax.T,
                    hf_grads.pop(f"{lp}mlp.gate.weight"),
                )
              elif rest[1] == "experts":
                if rest[2] == "gate_up_proj":
                  hf_gu = hf_grads.pop(f"{lp}mlp.experts.gate_up_proj")
                  ne, h, _, inter = g_jax.shape
                  g_gu_reshaped = g_jax.reshape(ne, h, 2 * inter)
                  for e in range(32):
                    _check_param(
                        f"{lp}mlp.experts.{e}.gate_up_proj",
                        g_gu_reshaped[e].T,
                        hf_gu[e],
                        is_expert=True,
                        expert_idx=e,
                    )
                  del hf_gu
                elif rest[2] == "down_proj":
                  hf_dp = hf_grads.pop(f"{lp}mlp.experts.down_proj")
                  for e in range(32):
                    _check_param(
                        f"{lp}mlp.experts.{e}.down_proj",
                        g_jax[e].T,
                        hf_dp[e],
                        is_expert=True,
                        expert_idx=e,
                    )
                  del hf_dp
                else:
                  raise ValueError(f"Unhandled mlp.experts parameter: {path!r}")
              else:
                raise ValueError(f"Unhandled mlp parameter: {path!r}")
            else:
              raise ValueError(f"Unhandled layer parameter: {path!r}")
          else:
            raise ValueError(f"Unhandled parameter path: {path!r}")
        finally:
          val.value = None

      if checked_conv1d == {"q", "k", "v"}:
        conv1d_key = f"{layer_prefix}self_attn.conv1d.weight"
        hf_grads.pop(conv1d_key)
      test_case.assertEmpty(
          hf_grads,
          f"Unchecked HF gradient keys remaining: {list(hf_grads.keys())}",
      )

      del flat_grads, jax_grads, hf_grads
      gc.collect()

      num_active_experts = len(active_experts_set)
      if case in ("l3_mla_moe", "l4_kda_moe"):
        test_case.assertGreaterEqual(
            num_active_experts,
            8,
            f"Expected at least 8 active routed experts for {case}, got" f" {num_active_experts}",
        )

      print(
          f"[REAL_LAYER_GRAD] case={case} dtype={jnp_dt_str}"
          f" ce_jax={float(jax_loss):.8e} ce_hf={hf_loss_val:.8e}"
          f" logits_frob_rel={l_frob_rel:.3e}"
          f" num_params_checked={num_params_checked}"
          f" num_active_experts={num_active_experts}"
          f" min_grad_cos={min_grad_cos:.8f}"
          f" max_grad_frob_rel={max_grad_frob_rel:.3e}",
          flush=True,
      )
  finally:
    jax.clear_caches()


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5RealLayerGradHermeticTest(parameterized.TestCase):
  """Hermetic unit tests executing 3 layer-type gradient parity passes on synthetic weights."""

  def setUp(self):
    super().setUp()
    if _RUN_REAL_LAYER_GRAD.value:
      self.skipTest("Skipping hermetic test when --run_real_layer_grad=true.")
    self.enter_context(torch.enable_grad())
    self.tempdir = self.create_tempdir().full_path
    _create_synthetic_checkpoint(self.tempdir)

  @parameterized.named_parameters(
      ("l0_kda_dense_fp32", "l0_kda_dense", False),
      ("l0_kda_dense_fp64", "l0_kda_dense", True),
      ("l3_mla_moe_fp32", "l3_mla_moe", False),
      ("l3_mla_moe_fp64", "l3_mla_moe", True),
      ("l4_kda_moe_fp32", "l4_kda_moe", False),
      ("l4_kda_moe_fp64", "l4_kda_moe", True),
  )
  def test_hermetic_layer_grad_parity(self, case: str, is_fp64: bool):
    if _RUN_REAL_LAYER_GRAD.value:
      self.skipTest("Skipping hermetic test when --run_real_layer_grad=true.")
    _run_single_layer_grad_pass(self, self.tempdir, case, is_fp64=is_fp64, is_synthetic=True)


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5RealLayerGradTest(parameterized.TestCase):
  """Real-weight per-layer-type gradient parity tests on GLM-5.3-Flash checkpoint slice."""

  def setUp(self):
    super().setUp()
    if not _RUN_REAL_LAYER_GRAD.value:
      self.skipTest("Real-weight layer gradient test requires --run_real_layer_grad=true.")
    self.enter_context(torch.enable_grad())
    _ensure_real_mini_slice(_MINI_SLICE_DIR.value)

  @parameterized.named_parameters(
      ("l0_kda_dense", "l0_kda_dense"),
      ("l3_mla_moe", "l3_mla_moe"),
      ("l4_kda_moe", "l4_kda_moe"),
  )
  def test_real_layer_grad_parity(self, case: str):
    if _REAL_LAYER_CASE.value not in ("all", case):
      self.skipTest(f"Skipping {case} because --real_layer_case={_REAL_LAYER_CASE.value}")

    # Pass 1: float32
    _run_single_layer_grad_pass(self, _MINI_SLICE_DIR.value, case, is_fp64=False, is_synthetic=False)

    # Pass 2: float64
    if _RUN_REAL_LAYER_FP64.value:
      _run_single_layer_grad_pass(self, _MINI_SLICE_DIR.value, case, is_fp64=True, is_synthetic=False)


if __name__ == "__main__":
  absltest.main()
