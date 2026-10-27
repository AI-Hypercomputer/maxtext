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

"""Shared mini-twin builders and parameter mappings for DeepSeek-V4.1 multimodal parity tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import contextlib
import json
import os
import re
import sys
import tempfile
from typing import Any
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file as save_numpy_file
import transformers

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import encoders
from maxtext.layers import engram_deepseek41
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.models import deepseek41_vision
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.parity import layer_parity

save_file: Any
torch: Any
nn: Any
ref_text_model: Any
multimodal: Any
ref_vision: Any
try:
  from safetensors.torch import save_file  # pylint: disable=g-import-not-at-top
  import torch  # pylint: disable=g-import-not-at-top
  from torch import nn  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import model as ref_text_model  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import multimodal  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import vision as ref_vision  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  save_file = None
  torch = None
  nn = None
  ref_text_model = None
  multimodal = None
  ref_vision = None
  HAS_TORCH = False

TOKENIZER_PATH = (
    os.environ["DEEPSEEK_V41_TOKENIZER"] if "DEEPSEEK_V41_TOKENIZER" in os.environ else "deepseek-ai/DeepSeek-V4.1-Flash"
)
OFFLINE_TOKENIZER_PATH = "src/maxtext/assets/tokenizers/qwen3-tokenizer"
MINI_VOCAB_SIZE = 4096
MINI_IMAGE_TOKEN_ID = 4000
MINI_SEQ_LEN = 40
MINI_BATCH_SIZE = 2
MINI_COMPRESS_RATIOS = (0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1)
MINI_KV_SOURCE_LAYERS = (2, 4, 6, 8)
MINI_INDEX_SOURCE_LAYERS = (2, 4, 6, 8, 10, 12)
MINI_CANDIDATE_SOURCE_LAYER = 8
MINI_ENGRAM_LAYER_IDS = (1, 6)
MINI_GRID_PADDED = (7, 5)
MINI_GRID_EXACT = (6, 9)
MINI_DEFAULT_SPAN_STARTS = ((0, 16), (4, 22))

_TOKENIZER_CACHE: dict[str, Any] = {}


def get_reference_tokenizer() -> Any:
  """Returns a cached HuggingFace tokenizer loaded from `TOKENIZER_PATH`."""
  if "tokenizer" not in _TOKENIZER_CACHE:
    _TOKENIZER_CACHE["tokenizer"] = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)
  return _TOKENIZER_CACHE["tokenizer"]


def mini_vision_maxtext_config(
    *,
    grid: tuple[int, int] = MINI_GRID_PADDED,
    **overrides: Any,
) -> Any:
  """Builds a MaxText config matching `multimodal.mini_model_args()` for vision parity tests."""
  args: dict[str, Any] = {
      "run_name": "deepseek41_mm_vision_test",
      "base_output_directory": os.path.join(tempfile.gettempdir(), "maxtext_mm_test_out"),
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "model_name": "deepseek4.1-flash-mini",
      "override_model_config": True,
      "use_multimodal": True,
      "vision_encoder_block": "deepseek41",
      "freeze_vision_encoder_params": False,
      "per_device_batch_size": 1.0,
      "max_target_length": MINI_SEQ_LEN,
      "dtype": "float32",
      "weight_dtype": "float32",
      "dtype_mm": "float32",
      "matmul_precision": "highest",
      "base_emb_dim": 64,
      "base_num_query_heads": 4,
      "head_dim": 32,
      "qk_rope_head_dim": 8,
      "q_lora_rank": 16,
      "o_groups": 2,
      "o_lora_rank": 8,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "sparse_matmul": False,
      "capacity_factor": -1,
      "sliding_window_size": 8,
      "base_num_decoder_layers": len(MINI_COMPRESS_RATIOS),
      "compress_ratios": list(MINI_COMPRESS_RATIOS),
      "kv_source_layers": list(MINI_KV_SOURCE_LAYERS),
      "index_source_layers": list(MINI_INDEX_SOURCE_LAYERS),
      "indexer_n_heads": 16,
      "indexer_head_dim": 16,
      "indexer_topk": 3,
      "candidate_source_layer": MINI_CANDIDATE_SOURCE_LAYER,
      "candidate_topk_blocks": 2,
      "candidate_block_size": 4,
      "ds41_engram_layers": list(MINI_ENGRAM_LAYER_IDS),
      "ds41_engram_vocab_size": 50,
      "ds41_engram_max_ngram_size": 3,
      "ds41_engram_n_heads": 2,
      "ds41_engram_head_dim": 8,
      "num_hidden_layers_for_vit": 2,
      "hidden_size_for_vit": 128,
      "num_attention_heads_for_vit": 2,
      "intermediate_size_for_vit": 352,
      "patch_size_for_vit": 14,
      "rope_theta_for_vit": 10000.0,
      "num_channels_for_vit": 3,
      "ds41_vision_downsample_ratio": 3,
      "ds41_vision_grid": list(grid),
      "tokenizer_path": TOKENIZER_PATH,
  }
  args.update(overrides)
  return pyconfig.initialize([sys.argv[0], "src/maxtext/configs/base.yml"], **args)


def single_device_mesh() -> jax.sharding.Mesh:
  """Returns a 1-device CPU mesh for unit-level parity checks."""
  return jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))


@contextlib.contextmanager
def fp64_maxtext_context(config: Any) -> Iterator[None]:
  """Enables x64 in JAX and patches `config` compute/weight dtypes to float64."""
  with jax.enable_x64(True):
    with mock.patch.dict(
        config._flat_config,  # pylint: disable=protected-access
        {
            "dtype": jnp.float64,
            "weight_dtype": jnp.float64,
            "dtype_mm": jnp.float64,
        },
    ):
      yield


def torch_to_jax(tensor: torch.Tensor, dtype: Any) -> jax.Array:
  """Converts a CPU torch tensor to a JAX array of `dtype` without downcasting float64."""
  np_dtype = np.float64 if dtype == jnp.float64 else np.float32
  return jnp.asarray(tensor.detach().cpu().to(torch.float64).numpy().astype(np_dtype), dtype=dtype)


def copy_linear(mt_dense: Any, pt_linear: nn.Linear, dtype: Any) -> None:
  """Copies `nn.Linear` `[out, in]` weights (and optional bias) into a 2D `DenseGeneral`."""
  w = torch_to_jax(pt_linear.weight, dtype).T
  if mt_dense.kernel[...].shape != w.shape:
    raise ValueError(f"Kernel shape mismatch: {mt_dense.kernel[...].shape} vs {w.shape}")
  mt_dense.kernel[...] = w
  if pt_linear.bias is not None:
    b = torch_to_jax(pt_linear.bias, dtype)
    if mt_dense.bias[...].shape != b.shape:
      raise ValueError(f"Bias shape mismatch: {mt_dense.bias[...].shape} vs {b.shape}")
    mt_dense.bias[...] = b


def copy_vision_block(
    mt_block: deepseek41_vision.DeepSeek41VisionBlock,
    ref_block: ref_vision.Block,
    dtype: Any,
) -> None:
  """Copies one reference `vision.Block` into `DeepSeek41VisionBlock`."""
  mt_block.norm1.scale[...] = torch_to_jax(ref_block.norm1.weight, dtype)
  copy_linear(mt_block.attn.wqkv, ref_block.attn.wqkv, dtype)
  copy_linear(mt_block.attn.wo, ref_block.attn.wo, dtype)
  mt_block.norm2.scale[...] = torch_to_jax(ref_block.norm2.weight, dtype)
  copy_linear(mt_block.mlp.w1, ref_block.mlp.w1, dtype)
  copy_linear(mt_block.mlp.w2, ref_block.mlp.w2, dtype)


def copy_vision_encoder(
    mt_encoder: deepseek41_vision.DeepSeek41VisionEncoder,
    ref_vit: ref_vision.ViT,
    dtype: Any,
) -> None:
  """Copies reference `vision.ViT` weights into `DeepSeek41VisionEncoder`."""
  copy_linear(mt_encoder.patch_embed, ref_vit.patch_embed.proj, dtype)
  for i, ref_block in enumerate(ref_vit.blocks):
    copy_vision_block(getattr(mt_encoder, f"blocks_{i}"), ref_block, dtype)
  mt_encoder.norm.scale[...] = torch_to_jax(ref_vit.norm.weight, dtype)


def copy_vision_projector(
    mt_projector: deepseek41_vision.DeepSeek41VisionProjector,
    ref_aligner: ref_vision.Aligner,
    image_start: torch.Tensor,
    image_end: torch.Tensor,
    image_newline: torch.Tensor,
    dtype: Any,
) -> None:
  """Copies reference `Aligner` and image delimiter embeddings into `DeepSeek41VisionProjector`."""
  copy_linear(mt_projector.w1, ref_aligner.w1, dtype)
  copy_linear(mt_projector.w2, ref_aligner.w2, dtype)
  mt_projector.image_start[...] = torch_to_jax(image_start, dtype)
  mt_projector.image_end[...] = torch_to_jax(image_end, dtype)
  mt_projector.image_newline[...] = torch_to_jax(image_newline, dtype)


def copy_full_vision_encoder(
    mt_wrapper: encoders.VisionEncoder,
    ref_model: multimodal.MultimodalTransformer,
    dtype: Any,
) -> None:
  """Copies `ref_model.vision`, `aligner`, and `image_*` into `encoders.VisionEncoder`."""
  if ref_model.vision is None or ref_model.aligner is None:
    raise ValueError("Reference model has no vision tower.")
  copy_vision_encoder(getattr(mt_wrapper, mt_wrapper.encoder_name), ref_model.vision, dtype)
  copy_vision_projector(
      getattr(mt_wrapper, mt_wrapper.projector_name),
      ref_model.aligner,
      ref_model.image_start,
      ref_model.image_end,
      ref_model.image_newline,
      dtype,
  )


def extract_pure_params(module: nnx.Module) -> tuple[nnx.GraphDef, nnx.State, dict[str, Any], nnx.State]:
  """Splits an NNX module into `(graphdef, param_state, pure_param_dict, other_state)`."""
  graphdef, param_state, other_state = nnx.split(module, nnx.Param, ...)
  return graphdef, param_state, nnx.to_pure_dict(param_state), other_state


def apply_with_pure_params(
    graphdef: nnx.GraphDef,
    param_state: nnx.State,
    pure_params: dict[str, Any],
    other_state: nnx.State,
) -> Any:
  """Rebuilds an NNX module from `extract_pure_params` outputs inside a functional transform."""
  state_copy = jax.tree.map(lambda arr: arr, param_state)
  other_copy = jax.tree.map(lambda arr: arr, other_state)
  nnx.replace_by_pure_dict(state_copy, pure_params)
  return nnx.merge(graphdef, state_copy, other_copy)


def check_input_gradient(
    *,
    ref_fn: Callable[[torch.Tensor], torch.Tensor],
    ref_input: torch.Tensor,
    cand_fn: Callable[[jax.Array], jax.Array],
    cand_input: jax.Array,
    grad_tol: layer_parity.Tolerance,
    cotangent_seed: int,
) -> tuple[bool, layer_parity.Metrics]:
  """Compares input gradients of `cand_fn` and `ref_fn` through a shared seeded projection."""
  ref_x = ref_input.detach().clone().requires_grad_(True)
  ref_out = ref_fn(ref_x)
  ref_out_np = ref_out.detach().cpu().to(torch.float64).numpy()
  cotangent = np.random.default_rng(cotangent_seed).standard_normal(ref_out_np.shape)
  ref_loss = (ref_out * torch.as_tensor(cotangent, dtype=ref_out.dtype)).sum()
  (ref_dx,) = torch.autograd.grad(ref_loss, ref_x)

  cand_out = cand_fn(cand_input)
  cand_ct = jnp.asarray(cotangent, dtype=cand_out.dtype)
  cand_dx = jax.grad(lambda x: jnp.sum(cand_fn(x) * cand_ct))(cand_input)
  return layer_parity.compare(
      np.asarray(cand_dx),
      ref_dx.detach().cpu().to(torch.float64).numpy(),
      grad_tol,
  )


def block_weight_map(prefix: str = "") -> dict[str, str]:
  """Returns reference-to-candidate leaf mapping for one `vision.Block`."""
  cand_prefix = f"{prefix}/" if prefix else ""
  return {
      "norm1.weight": f"{cand_prefix}norm1/scale",
      "attn.wqkv.weight": f"{cand_prefix}attn/wqkv/kernel:T",
      "attn.wqkv.bias": f"{cand_prefix}attn/wqkv/bias",
      "attn.wo.weight": f"{cand_prefix}attn/wo/kernel:T",
      "attn.wo.bias": f"{cand_prefix}attn/wo/bias",
      "norm2.weight": f"{cand_prefix}norm2/scale",
      "mlp.w1.weight": f"{cand_prefix}mlp/w1/kernel:T",
      "mlp.w2.weight": f"{cand_prefix}mlp/w2/kernel:T",
  }


def vit_weight_map(num_layers: int, cand_root: str = "") -> dict[str, str]:
  """Returns reference-to-candidate leaf mapping for `vision.ViT`."""
  prefix = f"{cand_root}/" if cand_root else ""
  mapping: dict[str, str] = {
      "patch_embed.proj.weight": f"{prefix}patch_embed/kernel:T",
      "patch_embed.proj.bias": f"{prefix}patch_embed/bias",
      "norm.weight": f"{prefix}norm/scale",
  }
  for i in range(num_layers):
    for ref_key, cand_key in block_weight_map(f"{prefix}blocks_{i}").items():
      mapping[f"blocks.{i}.{ref_key}"] = cand_key
  return mapping


def projector_weight_map(cand_root: str = "projector", include_delimiters: bool = True) -> dict[str, str]:
  """Returns reference-to-candidate leaf mapping for `Aligner` (+ optional delimiters)."""
  prefix = f"{cand_root}/" if cand_root else ""
  mapping: dict[str, str] = {
      "aligner.w1.weight": f"{prefix}w1/kernel:T",
      "aligner.w1.bias": f"{prefix}w1/bias",
      "aligner.w2.weight": f"{prefix}w2/kernel:T",
      "aligner.w2.bias": f"{prefix}w2/bias",
  }
  if include_delimiters:
    mapping["image_start"] = f"{prefix}image_start"
    mapping["image_end"] = f"{prefix}image_end"
    mapping["image_newline"] = f"{prefix}image_newline"
  return mapping


def vision_wrapper_weight_map(num_layers: int) -> dict[str, str]:
  """Returns reference-to-candidate leaf mapping for `encoders.VisionEncoder`."""
  mapping = {f"vision.{k}": v for k, v in vit_weight_map(num_layers, cand_root="DeepSeek41VisionEncoder_0").items()}
  mapping.update(projector_weight_map(cand_root="DeepSeek41VisionProjector_0", include_delimiters=True))
  return mapping


def copy_norm(mt_norm: Any, pt_norm: Any, dtype: Any) -> None:
  """Copies a reference RMSNorm scale into a MaxText RMSNorm."""
  mt_norm.scale[...] = torch_to_jax(pt_norm.weight, dtype)


def copy_linear_general(mt_dense: Any, pt_linear: Any, dtype: Any, out_shape: tuple[int, ...] | None = None) -> None:
  """Copies a reference Linear `[out, in]` weight into a MaxText DenseGeneral `[in, *out_shape]`."""
  w = torch_to_jax(pt_linear.weight, dtype).T
  if out_shape is not None:
    w = w.reshape((w.shape[0],) + tuple(out_shape))
  if mt_dense.kernel[...].shape != w.shape:
    raise ValueError(f"Kernel shape mismatch: {mt_dense.kernel[...].shape} vs {w.shape}")
  mt_dense.kernel[...] = w


def copy_engram(mt_engram: engram_deepseek41.DeepSeek41Engram, pt_engram: ref_text_model.Engram, dtype: Any) -> None:
  """Copies reference `Engram` weights (with zero-padded rows) into `DeepSeek41Engram`."""
  table = torch_to_jax(pt_engram.embed.weight, dtype)
  padded_rows = mt_engram.embed[...].shape[0]
  mt_engram.embed[...] = jnp.pad(table, ((0, padded_rows - table.shape[0]), (0, 0)))
  copy_linear_general(mt_engram.wkv, pt_engram.wkv, dtype)
  mt_engram.q_weight[...] = torch_to_jax(pt_engram.q_weight, dtype)
  mt_engram.k_weight[...] = torch_to_jax(pt_engram.k_weight, dtype)


def engram_weight_map(cand_root: str = "") -> dict[str, str]:
  """Returns reference-to-candidate leaf mapping for `DeepSeek41Engram`."""
  prefix = f"{cand_root}/" if cand_root else ""
  return {
      "embed.weight": f"{prefix}embed",
      "wkv.weight": f"{prefix}wkv/kernel:T",
      "q_weight": f"{prefix}q_weight",
      "k_weight": f"{prefix}k_weight",
  }


def copy_moe(mt_mlp: moe.RoutedAndSharedMoE, pt_ffn: ref_text_model.MoE, dtype: Any) -> None:
  """Copies reference `MoE` weights (including `gate.bias` and optional `gate.bias_vl`) into `RoutedAndSharedMoE`."""
  moe_block = mt_mlp.MoeBlock_0
  moe_block.gate.kernel[...] = torch_to_jax(pt_ffn.gate.weight, dtype).T
  moe_block.gate.bias[...] = torch_to_jax(pt_ffn.gate.bias, jnp.float32)
  if moe_block.gate.bias_vl is not None:
    if pt_ffn.gate.bias_vl is None:
      raise ValueError("MaxText GateLogit has bias_vl, but reference Gate has bias_vl=None.")
    moe_block.gate.bias_vl[...] = torch_to_jax(pt_ffn.gate.bias_vl, jnp.float32)
  moe_block.wi_0[...] = jnp.stack([torch_to_jax(e.w1.weight, dtype).T for e in pt_ffn.experts])
  moe_block.wi_1[...] = jnp.stack([torch_to_jax(e.w3.weight, dtype).T for e in pt_ffn.experts])
  moe_block.wo[...] = jnp.stack([torch_to_jax(e.w2.weight, dtype).T for e in pt_ffn.experts])
  copy_linear_general(mt_mlp.shared_experts.wi_0, pt_ffn.shared_experts.w1, dtype)
  copy_linear_general(mt_mlp.shared_experts.wi_1, pt_ffn.shared_experts.w3, dtype)
  copy_linear_general(mt_mlp.shared_experts.wo, pt_ffn.shared_experts.w2, dtype)


def copy_decoder_layer(
    mt_layer: deepseek41.DeepSeek41DecoderLayer,
    pt_layer: ref_text_model.Block,
    dtype: Any,
) -> None:
  """Copies one reference `Block` into a MaxText `DeepSeek41DecoderLayer`."""
  copy_norm(mt_layer.pre_self_attention_layer_norm, pt_layer.attn_norm, dtype)
  copy_norm(mt_layer.post_self_attention_layer_norm, pt_layer.ffn_norm, dtype)
  for name in ("hc_attn", "hc_ffn"):
    getattr(mt_layer, f"{name}_fn")[...] = torch_to_jax(getattr(pt_layer, f"{name}_fn"), dtype).T
    getattr(mt_layer, f"{name}_base")[...] = torch_to_jax(getattr(pt_layer, f"{name}_base"), dtype)
    getattr(mt_layer, f"{name}_scale")[...] = torch_to_jax(getattr(pt_layer, f"{name}_scale"), dtype)

  attn, ref_attn = mt_layer.self_attention, pt_layer.attn
  copy_linear_general(attn.wq_a, ref_attn.wq_a, dtype)
  copy_norm(attn.q_norm, ref_attn.q_norm, dtype)
  copy_linear_general(attn.wq_b, ref_attn.wq_b, dtype, (ref_attn.n_heads, ref_attn.head_dim))
  copy_linear_general(attn.wkv, ref_attn.wkv, dtype)
  copy_norm(attn.kv_norm, ref_attn.kv_norm, dtype)
  groups = ref_attn.n_groups
  attn.o_a_proj.kernel[...] = (
      torch_to_jax(ref_attn.wo_a.weight, dtype).reshape(groups, ref_attn.o_lora_rank, -1).transpose(0, 2, 1)
  )
  copy_linear_general(attn.o_b_proj, ref_attn.wo_b, dtype)
  attn.sinks[...] = torch_to_jax(ref_attn.attn_sink, dtype)
  if ref_attn.compressor is not None:
    copy_linear_general(attn.compressor.wkv, ref_attn.compressor.wkv, dtype)
    if ref_attn.compressor.compress_ratio > 1:
      copy_linear_general(attn.compressor.wgate, ref_attn.compressor.wgate, dtype)
    copy_norm(attn.compressor.norm, ref_attn.compressor.norm, dtype)
  if ref_attn.indexer is not None:
    copy_linear_general(
        attn.indexer.wq_b,
        ref_attn.indexer.wq_b,
        dtype,
        (ref_attn.indexer.n_heads, ref_attn.indexer.index_head_dim),
    )
    copy_linear_general(attn.indexer.weights_proj, ref_attn.indexer.weights_proj, dtype)
    if ref_attn.indexer.owns_k:
      copy_linear_general(attn.indexer.wk, ref_attn.indexer.wk, dtype)
      copy_norm(attn.indexer.k_norm, ref_attn.indexer.k_norm, dtype)

  copy_moe(mt_layer.mlp, pt_layer.ffn, dtype)
  if pt_layer.engram is not None:
    copy_engram(mt_layer.engram, pt_layer.engram, dtype)


def copy_full_multimodal_model(
    mt_model: Any,
    pt_model: ref_text_model.Transformer,
    config: Any,
    dtype: Any,
) -> None:
  """Copies all vision (when multimodal) and decoder weights into `mt_model` (scanned or unscanned)."""
  if config.use_multimodal:
    if mt_model.vision_encoder is None:
      raise ValueError("config.use_multimodal=True requires mt_model.vision_encoder to be initialized.")
    if not isinstance(pt_model, multimodal.MultimodalTransformer):
      raise TypeError("Multimodal MaxText model requires a MultimodalTransformer reference.")
    copy_full_vision_encoder(mt_model.vision_encoder, pt_model, dtype)

  mt_model.token_embedder.embedding[...] = torch_to_jax(pt_model.embed.weight, dtype)
  copy_norm(mt_model.decoder.decoder_norm, pt_model.norm, dtype)
  copy_linear_general(mt_model.decoder.logits_dense, pt_model.head, dtype)

  if not config.scan_layers:
    for i, ref_layer in enumerate(pt_model.layers):
      copy_decoder_layer(getattr(mt_model.decoder, f"layers_{i}"), ref_layer, dtype)
    return

  specs = deepseek41.build_layer_specs(config)
  scan_axis = config.param_scan_axis
  for k, segment in enumerate(deepseek41.build_segments(config)):
    group_params = []
    group_biases = []
    for group in segment.groups:
      tmp_block = deepseek41.DeepSeek41Block(
          config=config,
          mesh=mt_model.mesh,
          model_mode=mt_model.decoder.model_mode,
          specs=tuple(specs[idx] for idx in group),
          quant=mt_model.quant,
          rngs=nnx.Rngs(params=0),
      )
      for j, layer_idx in enumerate(group):
        copy_decoder_layer(getattr(tmp_block, f"layers_{j}"), pt_model.layers[layer_idx], dtype)
      _, p_state, b_state, _ = nnx.split(tmp_block, nnx.Param, moe.MoEBiasVar, ...)
      group_params.append(nnx.to_pure_dict(p_state))
      group_biases.append(nnx.to_pure_dict(b_state))

    def _stack_param_leaves(*leaves: jax.Array) -> jax.Array:
      stacked = jnp.stack(leaves, axis=0)
      return jnp.moveaxis(stacked, 0, scan_axis) if stacked.ndim > scan_axis else stacked

    stacked_params = jax.tree.map(_stack_param_leaves, *group_params)
    stacked_biases = jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *group_biases)
    stack = getattr(mt_model.decoder, f"segments_{k}")
    graphdef, s_params, s_biases, s_other = nnx.split(stack, nnx.Param, moe.MoEBiasVar, ...)
    nnx.replace_by_pure_dict(s_params, stacked_params)
    nnx.replace_by_pure_dict(s_biases, stacked_biases)
    setattr(mt_model.decoder, f"segments_{k}", nnx.merge(graphdef, s_params, s_biases, s_other))


def flatten_pure_dict(tree: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
  """Flattens a nested pure-parameter or pure-gradient dictionary to `/`-joined NumPy arrays."""
  out: dict[str, np.ndarray] = {}
  for key, val in tree.items():
    path = f"{prefix}/{key}" if prefix else str(key)
    if isinstance(val, Mapping):
      out.update(flatten_pure_dict(val, path))
    else:
      out[path] = np.asarray(val)
  return out


def unstack_model_param_grads(grad_pure: Mapping[str, Any], config: Any) -> dict[str, np.ndarray]:
  """Flattens `grad_pure` and unstacks `decoder/segments_*` leaves to `decoder/layers_{i}/*`."""
  flat = flatten_pure_dict(grad_pure)
  if not config.scan_layers:
    return flat
  out: dict[str, np.ndarray] = {}
  segments = deepseek41.build_segments(config)
  scan_axis = config.param_scan_axis
  seg_pat = re.compile(r"^decoder/segments_(\d+)/layers_(\d+)/(.+)$")
  for path, arr in flat.items():
    match = seg_pat.match(path)
    if match is None:
      out[path] = arr
      continue
    seg_idx = int(match.group(1))
    slot_idx = int(match.group(2))
    subpath = match.group(3)
    groups = segments[seg_idx].groups
    moved = np.moveaxis(arr, scan_axis, 0) if arr.ndim > scan_axis else arr
    if moved.shape[0] != len(groups):
      raise ValueError(f"Scan repeat mismatch on {path}: got {moved.shape}, expected {len(groups)} repeats.")
    for rep_idx, group in enumerate(groups):
      out[f"decoder/layers_{group[slot_idx]}/{subpath}"] = moved[rep_idx]
  return out


_TOP_GRAD_MAP: dict[str, tuple[str, bool]] = {
    "embed.weight": ("token_embedder/embedding", False),
    "norm.weight": ("decoder/decoder_norm/scale", False),
    "head.weight": ("decoder/logits_dense/kernel", True),
    "image_start": ("vision_encoder/DeepSeek41VisionProjector_0/image_start", False),
    "image_end": ("vision_encoder/DeepSeek41VisionProjector_0/image_end", False),
    "image_newline": ("vision_encoder/DeepSeek41VisionProjector_0/image_newline", False),
    "aligner.w1.weight": ("vision_encoder/DeepSeek41VisionProjector_0/w1/kernel", True),
    "aligner.w1.bias": ("vision_encoder/DeepSeek41VisionProjector_0/w1/bias", False),
    "aligner.w2.weight": ("vision_encoder/DeepSeek41VisionProjector_0/w2/kernel", True),
    "aligner.w2.bias": ("vision_encoder/DeepSeek41VisionProjector_0/w2/bias", False),
    "vision.patch_embed.proj.weight": ("vision_encoder/DeepSeek41VisionEncoder_0/patch_embed/kernel", True),
    "vision.patch_embed.proj.bias": ("vision_encoder/DeepSeek41VisionEncoder_0/patch_embed/bias", False),
    "vision.norm.weight": ("vision_encoder/DeepSeek41VisionEncoder_0/norm/scale", False),
}

_VISION_BLOCK_GRAD_MAP: dict[str, tuple[str, bool]] = {
    "norm1.weight": ("norm1/scale", False),
    "norm2.weight": ("norm2/scale", False),
    "attn.wqkv.weight": ("attn/wqkv/kernel", True),
    "attn.wqkv.bias": ("attn/wqkv/bias", False),
    "attn.wo.weight": ("attn/wo/kernel", True),
    "attn.wo.bias": ("attn/wo/bias", False),
    "mlp.w1.weight": ("mlp/w1/kernel", True),
    "mlp.w2.weight": ("mlp/w2/kernel", True),
}

_DECODER_LAYER_GRAD_MAP: dict[str, tuple[str, bool]] = {
    "attn_norm.weight": ("pre_self_attention_layer_norm/scale", False),
    "ffn_norm.weight": ("post_self_attention_layer_norm/scale", False),
    "hc_attn_fn": ("hc_attn_fn", True),
    "hc_ffn_fn": ("hc_ffn_fn", True),
    "hc_attn_base": ("hc_attn_base", False),
    "hc_ffn_base": ("hc_ffn_base", False),
    "hc_attn_scale": ("hc_attn_scale", False),
    "hc_ffn_scale": ("hc_ffn_scale", False),
    "attn.attn_sink": ("self_attention/sinks", False),
    "attn.wq_a.weight": ("self_attention/wq_a/kernel", True),
    "attn.q_norm.weight": ("self_attention/q_norm/scale", False),
    "attn.wkv.weight": ("self_attention/wkv/kernel", True),
    "attn.kv_norm.weight": ("self_attention/kv_norm/scale", False),
    "attn.wo_b.weight": ("self_attention/o_b_proj/kernel", True),
    "attn.compressor.norm.weight": ("self_attention/compressor/norm/scale", False),
    "attn.compressor.wkv.weight": ("self_attention/compressor/wkv/kernel", True),
    "attn.compressor.wgate.weight": ("self_attention/compressor/wgate/kernel", True),
    "attn.indexer.weights_proj.weight": ("self_attention/indexer/weights_proj/kernel", True),
    "attn.indexer.wk.weight": ("self_attention/indexer/wk/kernel", True),
    "attn.indexer.k_norm.weight": ("self_attention/indexer/k_norm/scale", False),
    "ffn.gate.weight": ("mlp/MoeBlock_0/gate/kernel", True),
    "ffn.shared_experts.w1.weight": ("mlp/shared_experts/wi_0/kernel", True),
    "ffn.shared_experts.w3.weight": ("mlp/shared_experts/wi_1/kernel", True),
    "ffn.shared_experts.w2.weight": ("mlp/shared_experts/wo/kernel", True),
    "engram.q_weight": ("engram/q_weight", False),
    "engram.k_weight": ("engram/k_weight", False),
    "engram.wkv.weight": ("engram/wkv/kernel", True),
    "engram.embed.weight": ("engram/embed", False),
}


def map_ref_param_to_cand_grad(
    ref_name: str,
    unstacked_grads: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, str]:
  """Maps a reference parameter name onto `(cand_grad_in_ref_shape, unstacked_leaf_key)`."""
  if ref_name in _TOP_GRAD_MAP:
    cand_key, transpose = _TOP_GRAD_MAP[ref_name]
    arr = unstacked_grads[cand_key]
    return (arr.T if transpose else arr), cand_key

  vis_match = re.match(r"^vision\.blocks\.(\d+)\.(.+)$", ref_name)
  if vis_match is not None:
    sub, transpose = _VISION_BLOCK_GRAD_MAP[vis_match.group(2)]
    cand_key = f"vision_encoder/DeepSeek41VisionEncoder_0/blocks_{vis_match.group(1)}/{sub}"
    arr = unstacked_grads[cand_key]
    return (arr.T if transpose else arr), cand_key

  lay_match = re.match(r"^layers\.(\d+)\.(.+)$", ref_name)
  if lay_match is None:
    raise KeyError(f"Unmapped reference parameter: {ref_name!r}")
  pfx = f"decoder/layers_{lay_match.group(1)}"
  rest = lay_match.group(2)

  exp_match = re.match(r"^ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$", rest)
  if exp_match is not None:
    expert_id = int(exp_match.group(1))
    mt_w = {"w1": "wi_0", "w3": "wi_1", "w2": "wo"}[exp_match.group(2)]
    cand_key = f"{pfx}/mlp/MoeBlock_0/{mt_w}"
    return unstacked_grads[cand_key][expert_id].T, cand_key

  if rest in ("attn.wq_b.weight", "attn.indexer.wq_b.weight"):
    sub = "self_attention/wq_b/kernel" if rest == "attn.wq_b.weight" else "self_attention/indexer/wq_b/kernel"
    cand_key = f"{pfx}/{sub}"
    arr = unstacked_grads[cand_key]
    return arr.reshape(arr.shape[0], -1).T, cand_key

  if rest == "attn.wo_a.weight":
    cand_key = f"{pfx}/self_attention/o_a_proj/kernel"
    arr = unstacked_grads[cand_key]
    return arr.transpose(0, 2, 1).reshape(-1, arr.shape[1]), cand_key

  sub, transpose = _DECODER_LAYER_GRAD_MAP[rest]
  cand_key = f"{pfx}/{sub}"
  arr = unstacked_grads[cand_key]
  return (arr.T if transpose else arr), cand_key


def check_full_model_gradients(
    *,
    ref_params: Mapping[str, torch.Tensor],
    ref_grads: Mapping[str, torch.Tensor | None],
    cand_grad_pure: Mapping[str, Any],
    config: Any,
    grad_classes: layer_parity.GradientClasses,
    grad_tol: layer_parity.Tolerance,
) -> tuple[dict[str, layer_parity.Metrics], list[str], set[str], dict[str, int]]:
  """Checks all full-model parameter leaf gradients across all five `GradientClasses`.

  Validates `grad_classes` via `layer_parity.validate_gradient_classes`, unstacks
  scanned decoder segments when `config.scan_layers` is `True`, checks every
  reference parameter in `compared`, `padded`, `discrete_only`, `frozen`, and
  `non_gradient`, and asserts 100% consumption of all candidate `nnx.Param`
  leaves.
  """
  layer_parity.validate_gradient_classes(grad_classes, set(ref_params), ref_grads)
  unstacked = unstack_model_param_grads(cand_grad_pure, config)
  grad_metrics: dict[str, layer_parity.Metrics] = {}
  failures: list[str] = []
  consumed: set[str] = set()
  counts = {
      "compared": 0,
      "padded": 0,
      "discrete_only": 0,
      "frozen": 0,
      "non_gradient": 0,
  }
  for name, ref_param in ref_params.items():
    ref_g = ref_grads[name]
    if name in grad_classes.non_gradient:
      counts["non_gradient"] += 1
      continue
    cand_g, cand_key = map_ref_param_to_cand_grad(name, unstacked)
    consumed.add(cand_key)
    if name in grad_classes.compared:
      if ref_g is None:
        raise ValueError(f"Compared parameter {name!r} has ref_grad=None")
      ref_g64 = ref_g.detach().cpu().to(torch.float64).numpy()
      ok, metrics = layer_parity.compare(cand_g, ref_g64, grad_tol)
      grad_metrics[name] = metrics
      counts["compared"] += 1
      if not ok:
        failures.append(f"grad {name}: {metrics}")
    elif name in grad_classes.padded:
      if ref_g is None:
        raise ValueError(f"Padded parameter {name!r} has ref_grad=None")
      ref_g64 = ref_g.detach().cpu().to(torch.float64).numpy()
      ok, metrics = layer_parity.compare_padded_rows(cand_g, ref_g64, ref_param.shape[0], grad_tol)
      grad_metrics[name] = metrics
      counts["padded"] += 1
      if not ok:
        failures.append(f"padded grad {name}: {metrics}")
    elif name in grad_classes.discrete_only or name in grad_classes.frozen:
      cls_key = "discrete_only" if name in grad_classes.discrete_only else "frozen"
      counts[cls_key] += 1
      if cand_g.shape != tuple(ref_param.shape):
        raise ValueError(f"Shape mismatch on {cls_key} {name!r}: {cand_g.shape} vs {tuple(ref_param.shape)}")
      cand_g64 = np.asarray(cand_g, dtype=np.float64)
      finite = bool(np.all(np.isfinite(cand_g64)))
      max_abs = float(np.max(np.abs(cand_g64))) if finite else float("nan")
      frob = float(np.linalg.norm(cand_g64)) if finite else float("nan")
      metrics = layer_parity.Metrics(max_abs=max_abs, ref_inf=0.0, frob_rel=frob, finite=finite)
      grad_metrics[name] = metrics
      if not (finite and max_abs == 0.0):
        failures.append(f"{cls_key} grad {name} not identically zero: {metrics}")
    else:
      failures.append(f"unclassified parameter {name!r}")

  unconsumed = sorted(set(unstacked) - consumed)
  if unconsumed:
    raise ValueError(f"Unconsumed candidate nnx.Param gradient leaves: {unconsumed}")
  return grad_metrics, failures, consumed, counts


F32_CHECKPOINT_SUFFIXES: tuple[str, ...] = (
    "attn.attn_sink",
    "ffn.gate.bias",
    "ffn.gate.bias_vl",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)

_RELEASE_QUANTIZED_SUFFIXES: tuple[str, ...] = (
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.indexer.wq_b.weight",
    "w1.weight",
    "w2.weight",
    "w3.weight",
    "engram.wkv.weight",
    "engram.embed.weight",
)


def write_mini_safetensors_checkpoint(
    ckpt_dir: str,
    ref_model: nn.Module,
    *,
    param_dtype: Any = None,
    include_mtp: bool = False,
) -> dict[str, torch.Tensor]:
  """Writes `ref_model` in release HF SafeTensors layout and returns the saved tensor dict.

  Tensors matching `F32_CHECKPOINT_SUFFIXES` (routing biases, hyper-connection
  mixes, attention sinks) are saved as `torch.float32`, matching the release
  checkpoint dtypes; all other floating-point tensors are saved as `param_dtype`.
  When `include_mtp=True`, representative `mtp.{0,1,2}.*` tensors (including
  `ffn.gate.bias_vl`) are added so loader key-accounting tests can verify they
  remain unloaded.
  """
  if param_dtype is None:
    param_dtype = torch.bfloat16
  os.makedirs(ckpt_dir, exist_ok=True)
  saved: dict[str, torch.Tensor] = {}
  for key, tensor in ref_model.state_dict().items():
    t = tensor.detach().cpu()
    if t.is_floating_point():
      target_dtype = torch.float32 if key.endswith(F32_CHECKPOINT_SUFFIXES) else param_dtype
      t = t.to(target_dtype)
    saved[key] = t.contiguous()

  if include_mtp:
    g = torch.Generator().manual_seed(901)
    saved["mtp.0.main_norm.weight"] = torch.randn((8,), generator=g, dtype=torch.float32).to(param_dtype).contiguous()
    saved["mtp.0.ffn.gate.bias_vl"] = torch.randn((4,), generator=g, dtype=torch.float32).contiguous()
    saved["mtp.1.attn_norm.weight"] = torch.randn((8,), generator=g, dtype=torch.float32).to(param_dtype).contiguous()
    saved["mtp.1.ffn.gate.bias_vl"] = torch.randn((4,), generator=g, dtype=torch.float32).contiguous()
    saved["mtp.2.norm.weight"] = torch.randn((8,), generator=g, dtype=torch.float32).to(param_dtype).contiguous()
    saved["mtp.2.ffn.gate.bias_vl"] = torch.randn((4,), generator=g, dtype=torch.float32).contiguous()

  shard_name = "model-00001-of-00001.safetensors"
  save_file(saved, os.path.join(ckpt_dir, shard_name), metadata={"format": "pt"})
  index_payload = {
      "metadata": {},
      "weight_map": {key: shard_name for key in sorted(saved)},
  }
  with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump(index_payload, f, indent=2, sort_keys=True)
  with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({"model_type": "deepseek_v41"}, f, indent=2, sort_keys=True)
  return saved


def write_numpy_mini_mm_checkpoint(
    ckpt_dir: str,
    *,
    seed: int = 1,
    use_multimodal: bool = True,
    include_mtp: bool = False,
) -> tuple[Any, dict[str, np.ndarray]]:
  """Writes a pure-NumPy 14-layer mini multimodal SafeTensors checkpoint without PyTorch."""
  os.makedirs(ckpt_dir, exist_ok=True)
  cfg = mini_vision_maxtext_config(
      use_multimodal=use_multimodal,
      vocab_size=MINI_VOCAB_SIZE,
      ds41_image_token_id=MINI_IMAGE_TOKEN_ID,
      ds41_engram_compressed_vocab_size=0,
      tokenizer_path=OFFLINE_TOKENIZER_PATH,
      scan_layers=False,
  )
  mesh = single_device_mesh()
  with mesh:
    model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  rng = np.random.default_rng(seed)
  engram_unpadded = {
      lid: sum(sum(per_ngram) for per_ngram in layer)
      for lid, layer in zip(
          MINI_ENGRAM_LAYER_IDS,
          engram_deepseek41.engram_primes(MINI_ENGRAM_LAYER_IDS, 3, 2, 50),
      )
  }
  hf_tensors: dict[str, np.ndarray] = {
      "embed.weight": np.asarray(model.token_embedder.embedding[...], np.float32)[:MINI_VOCAB_SIZE].copy(),
      "norm.weight": np.asarray(model.decoder.decoder_norm.scale[...], np.float32).copy(),
      "head.weight": np.asarray(model.decoder.logits_dense.kernel[...], np.float32).T[:MINI_VOCAB_SIZE].copy(),
  }
  if use_multimodal:
    enc = model.vision_encoder.DeepSeek41VisionEncoder_0
    proj = model.vision_encoder.DeepSeek41VisionProjector_0
    hf_tensors["vision.patch_embed.proj.weight"] = np.asarray(enc.patch_embed.kernel[...], np.float32).T.copy()
    hf_tensors["vision.patch_embed.proj.bias"] = rng.normal(0.0, 0.02, size=enc.patch_embed.bias[...].shape).astype(
        np.float32
    )
    for bi in range(cfg.num_hidden_layers_for_vit):
      blk = getattr(enc, f"blocks_{bi}")
      vblk = f"vision.blocks.{bi}"
      hf_tensors[f"{vblk}.norm1.weight"] = np.asarray(blk.norm1.scale[...], np.float32).copy()
      hf_tensors[f"{vblk}.norm2.weight"] = np.asarray(blk.norm2.scale[...], np.float32).copy()
      hf_tensors[f"{vblk}.attn.wqkv.weight"] = np.asarray(blk.attn.wqkv.kernel[...], np.float32).T.copy()
      hf_tensors[f"{vblk}.attn.wqkv.bias"] = rng.normal(0.0, 0.02, size=blk.attn.wqkv.bias[...].shape).astype(np.float32)
      hf_tensors[f"{vblk}.attn.wo.weight"] = np.asarray(blk.attn.wo.kernel[...], np.float32).T.copy()
      hf_tensors[f"{vblk}.attn.wo.bias"] = rng.normal(0.0, 0.02, size=blk.attn.wo.bias[...].shape).astype(np.float32)
      hf_tensors[f"{vblk}.mlp.w1.weight"] = np.asarray(blk.mlp.w1.kernel[...], np.float32).T.copy()
      hf_tensors[f"{vblk}.mlp.w2.weight"] = np.asarray(blk.mlp.w2.kernel[...], np.float32).T.copy()
    hf_tensors["vision.norm.weight"] = np.asarray(enc.norm.scale[...], np.float32).copy()
    hf_tensors["aligner.w1.weight"] = np.asarray(proj.w1.kernel[...], np.float32).T.copy()
    hf_tensors["aligner.w1.bias"] = rng.normal(0.0, 0.02, size=proj.w1.bias[...].shape).astype(np.float32)
    hf_tensors["aligner.w2.weight"] = np.asarray(proj.w2.kernel[...], np.float32).T.copy()
    hf_tensors["aligner.w2.bias"] = rng.normal(0.0, 0.02, size=proj.w2.bias[...].shape).astype(np.float32)
    hf_tensors["image_start"] = rng.normal(0.0, 0.2, size=proj.image_start[...].shape).astype(np.float32)
    hf_tensors["image_end"] = rng.normal(0.0, 0.2, size=proj.image_end[...].shape).astype(np.float32)
    hf_tensors["image_newline"] = rng.normal(0.0, 0.2, size=proj.image_newline[...].shape).astype(np.float32)

  specs = deepseek41.build_layer_specs(cfg)
  for i, spec in enumerate(specs):
    layer = getattr(model.decoder, f"layers_{i}")
    hf_tensors[f"layers.{i}.attn_norm.weight"] = np.asarray(layer.pre_self_attention_layer_norm.scale[...], np.float32)
    hf_tensors[f"layers.{i}.ffn_norm.weight"] = np.asarray(layer.post_self_attention_layer_norm.scale[...], np.float32)
    for name in ("hc_attn", "hc_ffn"):
      fn_val = np.asarray(getattr(layer, f"{name}_fn")[...], np.float32)
      hf_tensors[f"layers.{i}.{name}_fn"] = fn_val.reshape(fn_val.shape[0], -1).T.copy()
      hf_tensors[f"layers.{i}.{name}_base"] = np.asarray(getattr(layer, f"{name}_base")[...], np.float32)
      hf_tensors[f"layers.{i}.{name}_scale"] = np.asarray(getattr(layer, f"{name}_scale")[...], np.float32)
    attn = layer.self_attention
    for mt_attr, hf_sub in (
        ("wq_a", "wq_a.weight"),
        ("wq_b", "wq_b.weight"),
        ("wkv", "wkv.weight"),
        ("o_b_proj", "wo_b.weight"),
    ):
      k_val = np.asarray(getattr(attn, mt_attr).kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.{hf_sub}"] = k_val.reshape(k_val.shape[0], -1).T.copy()
    oa_val = np.asarray(attn.o_a_proj.kernel[...], np.float32)
    hf_tensors[f"layers.{i}.attn.wo_a.weight"] = oa_val.transpose(0, 2, 1).reshape(-1, oa_val.shape[1]).copy()
    hf_tensors[f"layers.{i}.attn.q_norm.weight"] = np.asarray(attn.q_norm.scale[...], np.float32)
    hf_tensors[f"layers.{i}.attn.kv_norm.weight"] = np.asarray(attn.kv_norm.scale[...], np.float32)
    hf_tensors[f"layers.{i}.attn.attn_sink"] = rng.normal(0.0, 0.2, size=attn.sinks[...].shape).astype(np.float32)
    if spec.is_kv_source:
      cw = np.asarray(attn.compressor.wkv.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.compressor.wkv.weight"] = cw.reshape(cw.shape[0], -1).T.copy()
      if spec.compress_ratio > 1:
        cg = np.asarray(attn.compressor.wgate.kernel[...], np.float32)
        hf_tensors[f"layers.{i}.attn.compressor.wgate.weight"] = cg.reshape(cg.shape[0], -1).T.copy()
      hf_tensors[f"layers.{i}.attn.compressor.norm.weight"] = np.asarray(attn.compressor.norm.scale[...], np.float32)
    if spec.is_index_source:
      iwq = np.asarray(attn.indexer.wq_b.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.indexer.wq_b.weight"] = iwq.reshape(iwq.shape[0], -1).T.copy()
      iwp = np.asarray(attn.indexer.weights_proj.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.indexer.weights_proj.weight"] = iwp.reshape(iwp.shape[0], -1).T.copy()
      if spec.is_kv_source:
        iwk = np.asarray(attn.indexer.wk.kernel[...], np.float32)
        hf_tensors[f"layers.{i}.attn.indexer.wk.weight"] = iwk.reshape(iwk.shape[0], -1).T.copy()
        hf_tensors[f"layers.{i}.attn.indexer.k_norm.weight"] = np.asarray(attn.indexer.k_norm.scale[...], np.float32)
    moe_block = layer.mlp.MoeBlock_0
    gk = np.asarray(moe_block.gate.kernel[...], np.float32)
    hf_tensors[f"layers.{i}.ffn.gate.weight"] = gk.reshape(gk.shape[0], -1).T.copy()
    hf_tensors[f"layers.{i}.ffn.gate.bias"] = rng.normal(0.0, 0.05, size=moe_block.gate.bias[...].shape).astype(
        np.float32
    )
    if use_multimodal:
      hf_tensors[f"layers.{i}.ffn.gate.bias_vl"] = rng.normal(0.0, 0.05, size=moe_block.gate.bias[...].shape).astype(
          np.float32
      )
    wi0 = np.asarray(moe_block.wi_0[...], np.float32)
    wi1 = np.asarray(moe_block.wi_1[...], np.float32)
    wo = np.asarray(moe_block.wo[...], np.float32)
    for e in range(cfg.num_experts):
      hf_tensors[f"layers.{i}.ffn.experts.{e}.w1.weight"] = wi0[e].T.copy()
      hf_tensors[f"layers.{i}.ffn.experts.{e}.w3.weight"] = wi1[e].T.copy()
      hf_tensors[f"layers.{i}.ffn.experts.{e}.w2.weight"] = wo[e].T.copy()
    for mt_attr, hf_sub in (("wi_0", "w1"), ("wi_1", "w3"), ("wo", "w2")):
      sk = np.asarray(getattr(layer.mlp.shared_experts, mt_attr).kernel[...], np.float32)
      hf_tensors[f"layers.{i}.ffn.shared_experts.{hf_sub}.weight"] = sk.reshape(sk.shape[0], -1).T.copy()
    if spec.has_engram:
      rows = engram_unpadded[i]
      hf_tensors[f"layers.{i}.engram.embed.weight"] = rng.normal(0.0, 0.2, size=(rows, 8)).astype(np.float32)
      ew = np.asarray(layer.engram.wkv.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.engram.wkv.weight"] = ew.reshape(ew.shape[0], -1).T.copy()
      hf_tensors[f"layers.{i}.engram.q_weight"] = np.asarray(layer.engram.q_weight[...], np.float32)
      hf_tensors[f"layers.{i}.engram.k_weight"] = np.asarray(layer.engram.k_weight[...], np.float32)

  if include_mtp:
    hf_tensors["mtp.0.main_norm.weight"] = rng.normal(size=(8,)).astype(np.float32)
    hf_tensors["mtp.0.ffn.gate.bias_vl"] = rng.normal(size=(4,)).astype(np.float32)

  shard_name = "model-00001-of-00001.safetensors"
  save_numpy_file(hf_tensors, os.path.join(ckpt_dir, shard_name))
  index_payload = {
      "metadata": {},
      "weight_map": {key: shard_name for key in sorted(hf_tensors)},
  }
  with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump(index_payload, f, indent=2, sort_keys=True)
  with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({"model_type": "deepseek_v41", "quantization_config": None}, f, indent=2, sort_keys=True)
  return cfg, hf_tensors


def load_mini_maxtext_from_safetensors(
    ckpt_dir: str,
    *,
    grid: tuple[int, int] = MINI_GRID_PADDED,
    scan_layers: bool = False,
    mesh: jax.sharding.Mesh | None = None,
    **config_overrides: Any,
) -> tuple[Any, Any, jax.sharding.Mesh]:
  """Loads a mini MaxText model from `ckpt_dir` via real `safetensors_dynamic` streaming."""
  cfg = mini_vision_maxtext_config(
      grid=grid,
      scan_layers=scan_layers,
      load_parameters_path=ckpt_dir,
      source_checkpoint_layout="safetensors_dynamic",
      **config_overrides,
  )
  target_mesh = mesh if mesh is not None else jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with target_mesh:
    model = model_creation_utils.from_pretrained(cfg, mesh=target_mesh, model_mode=MODEL_MODE_TRAIN)
  return model, cfg, target_mesh


def flatten_mapped_hf_keys(param_map: Mapping[str, Any]) -> set[str]:
  """Returns the flat set of all HF tensor names referenced in `param_map`."""
  keys: set[str] = set()

  def _walk(node: Any) -> None:
    if isinstance(node, str):
      keys.add(node)
    elif isinstance(node, (list, tuple)):
      for child in node:
        _walk(child)
    else:
      raise TypeError(f"Unsupported HF source entry in param_map: {node!r} ({type(node).__name__})")

  for value in param_map.values():
    _walk(value)
  return keys


def build_release_40l_hf_keys(*, include_mtp: bool = True) -> set[str]:
  """Constructs the exact 96,085-key release `DeepSeek-V4.1-Flash` SafeTensors key set."""
  keys: set[str] = {"embed.weight", "norm.weight", "head.weight"}
  keys.update(
      {
          "vision.patch_embed.proj.weight",
          "vision.patch_embed.proj.bias",
          "vision.norm.weight",
          "aligner.w1.weight",
          "aligner.w1.bias",
          "aligner.w2.weight",
          "aligner.w2.bias",
          "image_start",
          "image_end",
          "image_newline",
      }
  )
  for blk_id in range(32):
    vblk = f"vision.blocks.{blk_id}"
    keys.update(
        {
            f"{vblk}.norm1.weight",
            f"{vblk}.norm2.weight",
            f"{vblk}.attn.wqkv.weight",
            f"{vblk}.attn.wqkv.bias",
            f"{vblk}.attn.wo.weight",
            f"{vblk}.attn.wo.bias",
            f"{vblk}.mlp.w1.weight",
            f"{vblk}.mlp.w2.weight",
        }
    )

  def _add_layer_key(name: str) -> None:
    keys.add(name)
    if name.endswith(_RELEASE_QUANTIZED_SUFFIXES):
      keys.add(f"{name[:-len('.weight')]}.scale")

  compress_ratios = (0, 0) + (2,) * 18 + (1,) * 20
  kv_source_layers = {2, 8, 14, 20}
  index_source_layers = {2, 8, 14, 20, 24, 28, 32, 36}
  engram_layers = {1, 14}
  for layer_id in range(40):
    pfx = f"layers.{layer_id}"
    for suffix in (
        "attn_norm.weight",
        "ffn_norm.weight",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "attn.attn_sink",
        "attn.wq_a.weight",
        "attn.q_norm.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.kv_norm.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "ffn.gate.bias_vl",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.weight",
    ):
      _add_layer_key(f"{pfx}.{suffix}")
    for expert_id in range(384):
      for w_name in ("w1", "w2", "w3"):
        _add_layer_key(f"{pfx}.ffn.experts.{expert_id}.{w_name}.weight")
    if layer_id in kv_source_layers:
      _add_layer_key(f"{pfx}.attn.compressor.norm.weight")
      _add_layer_key(f"{pfx}.attn.compressor.wkv.weight")
      if compress_ratios[layer_id] > 1:
        _add_layer_key(f"{pfx}.attn.compressor.wgate.weight")
    if layer_id in index_source_layers:
      _add_layer_key(f"{pfx}.attn.indexer.wq_b.weight")
      _add_layer_key(f"{pfx}.attn.indexer.weights_proj.weight")
      if layer_id in kv_source_layers:
        _add_layer_key(f"{pfx}.attn.indexer.wk.weight")
        _add_layer_key(f"{pfx}.attn.indexer.k_norm.weight")
    if layer_id in engram_layers:
      _add_layer_key(f"{pfx}.engram.embed.weight")
      _add_layer_key(f"{pfx}.engram.wkv.weight")
      _add_layer_key(f"{pfx}.engram.q_weight")
      _add_layer_key(f"{pfx}.engram.k_weight")

  if include_mtp:
    for mtp_id in range(3):
      mpfx = f"mtp.{mtp_id}"
      for suffix in (
          "attn_norm.weight",
          "ffn_norm.weight",
          "hc_attn_fn",
          "hc_attn_base",
          "hc_attn_scale",
          "hc_ffn_fn",
          "hc_ffn_base",
          "hc_ffn_scale",
          "attn.attn_sink",
          "attn.wq_a.weight",
          "attn.q_norm.weight",
          "attn.wq_b.weight",
          "attn.wkv.weight",
          "attn.kv_norm.weight",
          "attn.wo_a.weight",
          "attn.wo_b.weight",
          "ffn.gate.weight",
          "ffn.gate.bias",
          "ffn.gate.bias_vl",
          "ffn.shared_experts.w1.weight",
          "ffn.shared_experts.w2.weight",
          "ffn.shared_experts.w3.weight",
      ):
        _add_layer_key(f"{mpfx}.{suffix}")
      for expert_id in range(128):
        for w_name in ("w1", "w2", "w3"):
          _add_layer_key(f"{mpfx}.ffn.experts.{expert_id}.{w_name}.weight")
    keys.update({"mtp.0.main_norm.weight", "mtp.0.main_proj.weight", "mtp.0.main_proj.scale"})
    keys.update(
        {
            "mtp.2.confidence_head.proj.weight",
            "mtp.2.markov_head.embed.weight",
            "mtp.2.markov_head.head.weight",
            "mtp.2.norm.weight",
        }
    )
  return keys
