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

# pylint: disable=too-many-positional-arguments,not-callable

"""Layer-subgroup harness: MaxText DeepSeek-V4 vs the vendored official reference.

Subgroups (model-size agnostic; the tiny CPU test and the TPU script share them):
  S1: unrolled prefix hash layers `layers_0 .. layers_{n_hash-1}`.
  S2: scanned `[HCA, CSA]` blocks through `NNXDecoder._apply_layers_sequentially`.
  S3: one scanned block + `hc_head` + `apply_output_head` + masked cross entropy.

Weights move between the native (official) key space and MaxText only through the
production converter tables `param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_*`, applied
with the converters' own loader (`to_maxtext`) and saver (`process_maxtext_param`)
helpers.
"""

import contextlib
import dataclasses
import functools
import os
import re
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_compressed
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from tests.utils import deepseek4_reference as R
from tests.utils.deepseek4_reference import kernel as ref_kernel
from tests.utils.deepseek4_reference import model as ref_model

# MaxText hard-codes the hyper-connection epsilon (mhc.sinkhorn, mhc mapping, DeepSeek4HyperHead).
MAXTEXT_HC_EPS = 1e-6

# ModelArgs field -> MaxText config field, compared by equality.
ARCH_FIELD_MAP = (
    ("dim", "emb_dim"),
    ("n_heads", "num_query_heads"),
    ("head_dim", "head_dim"),
    ("rope_head_dim", "qk_rope_head_dim"),
    ("q_lora_rank", "q_lora_rank"),
    ("o_lora_rank", "o_lora_rank"),
    ("o_groups", "o_groups"),
    ("n_routed_experts", "num_experts"),
    ("n_activated_experts", "num_experts_per_tok"),
    ("n_shared_experts", "shared_experts"),
    ("moe_inter_dim", "moe_mlp_dim"),
    ("index_n_heads", "indexer_n_heads"),
    ("index_head_dim", "indexer_head_dim"),
    ("index_topk", "indexer_topk"),
    ("window_size", "sliding_window_size"),
    ("hc_mult", "mhc_expansion_rate"),
    ("hc_sinkhorn_iters", "sinkhorn_iterations"),
    ("vocab_size", "vocab_size"),
    ("n_hash_layers", "first_num_hash_layers"),
    ("score_func", "routed_score_func"),
    ("route_scale", "routed_scaling_factor"),
    ("swiglu_limit", "mlp_activations_limit"),
    ("norm_eps", "normalization_layer_epsilon"),
    ("rope_theta", "rope_max_timescale"),
    ("compress_rope_theta", "compressed_rope_max_timescale"),
    ("rope_factor", "rope_factor"),
    ("beta_fast", "beta_fast"),
    ("beta_slow", "beta_slow"),
    ("original_seq_len", "original_max_position_embeddings"),
)

# MaxText fields with a single value valid for DeepSeek-V4.
MAXTEXT_FIXED_FIELDS = (
    ("num_kv_heads", 1),
    ("routed_bias", True),
    ("mlp_activations", ["silu", "linear"]),
    ("n_routing_groups", -1),
    ("use_indexer", True),
    ("mtp_num_layers", 0),
)

LAYER_TYPES = frozenset({"SWA+hash", "CSA+indexer+hash", "HCA+topk", "CSA+indexer+topk"})


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------


def config_kwargs_from_ref_args(args: ref_model.ModelArgs, num_layers: int) -> dict[str, Any]:
  """MaxText architecture overrides mirroring `args` for the first `num_layers` layers."""
  if num_layers > args.n_layers:
    raise ValueError(f"num_layers={num_layers} > n_layers={args.n_layers}")
  kwargs = {
      "base_emb_dim": args.dim,
      "base_num_query_heads": args.n_heads,
      "base_num_kv_heads": 1,
      "head_dim": args.head_dim,
      "qk_rope_head_dim": args.rope_head_dim,
      "q_lora_rank": args.q_lora_rank,
      "o_lora_rank": args.o_lora_rank,
      "o_groups": args.o_groups,
      "num_experts": args.n_routed_experts,
      "num_experts_per_tok": args.n_activated_experts,
      "shared_experts": args.n_shared_experts,
      "base_moe_mlp_dim": args.moe_inter_dim,
      "base_mlp_dim": args.moe_inter_dim,
      "indexer_n_heads": args.index_n_heads,
      "indexer_head_dim": args.index_head_dim,
      "indexer_topk": args.index_topk,
      "sliding_window_size": args.window_size,
      "mhc_expansion_rate": args.hc_mult,
      "sinkhorn_iterations": args.hc_sinkhorn_iters,
      "vocab_size": args.vocab_size,
      "first_num_hash_layers": args.n_hash_layers,
      "base_num_decoder_layers": num_layers,
      "compress_ratios": list(args.compress_ratios[:num_layers]),
      "routed_scaling_factor": args.route_scale,
      "mlp_activations_limit": args.swiglu_limit,
      "routed_score_func": args.score_func,
      "norm_topk_prob": True,
      "routed_bias": True,
      "mlp_activations": ["silu", "linear"],
      "normalization_layer_epsilon": args.norm_eps,
      "rope_max_timescale": args.rope_theta,
      "compressed_rope_max_timescale": args.compress_rope_theta,
      "rope_type": "yarn" if args.original_seq_len > 0 else "default",
      "rope_factor": args.rope_factor,
      "beta_fast": args.beta_fast,
      "beta_slow": args.beta_slow,
  }
  if args.original_seq_len > 0:
    kwargs["original_max_position_embeddings"] = args.original_seq_len
  return kwargs


def make_config(
    args: ref_model.ModelArgs,
    seq: int,
    dtype: str = "float32",
    num_layers: int | None = None,
    scan_layers: bool = True,
    model_name: str = "deepseek4-tiny",
    **overrides,
):
  """Pyconfig for DeepSeek-V4 matching `args` (CPU/TPU agnostic; mesh from config)."""
  kwargs = {
      "model_name": model_name,
      "override_model_config": True,
      "run_name": "ds4_layerwise",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "dtype": dtype,
      "weight_dtype": dtype,
      "matmul_precision": "highest",
      "attention": "dot_product",
      "attention_type": "compressed",
      "use_indexer": True,
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 1.0,
      "sparse_matmul": True,
      "megablox": False,
      "per_device_batch_size": args.max_batch_size,
      "max_target_length": seq,
      "scan_layers": scan_layers,
  }
  kwargs.update(config_kwargs_from_ref_args(args, num_layers or args.n_layers))
  kwargs.update(overrides)
  base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
  return pyconfig.initialize(["", base_yml], **kwargs)


def make_tiny_config(seq: int, dtype: str = "float32", num_layers: int = 9, scan_layers: bool = True, **overrides):
  """Pyconfig matching `R.tiny_args()` exactly."""
  return make_config(R.tiny_args(), seq, dtype, num_layers, scan_layers, **overrides)


def hf_config_from_ref_args(args: ref_model.ModelArgs, num_layers: int | None = None) -> dict[str, Any]:
  """Converter config dict (deepseek4_284b_dict key names) for `args`."""
  return {
      "hidden_size": args.dim,
      "num_attention_heads": args.n_heads,
      "num_key_value_heads": 1,
      "head_dim": args.head_dim,
      "qk_rope_head_dim": args.rope_head_dim,
      "q_lora_rank": args.q_lora_rank,
      "o_lora_rank": args.o_lora_rank,
      "o_groups": args.o_groups,
      "n_routed_experts": args.n_routed_experts,
      "num_experts_per_tok": args.n_activated_experts,
      "n_shared_experts": args.n_shared_experts,
      "moe_intermediate_size": args.moe_inter_dim,
      "index_n_heads": args.index_n_heads,
      "index_head_dim": args.index_head_dim,
      "index_topk": args.index_topk,
      "sliding_window": args.window_size,
      "hc_mult": args.hc_mult,
      "hc_eps": args.hc_eps,
      "hc_sinkhorn_iters": args.hc_sinkhorn_iters,
      "vocab_size": args.vocab_size,
      "num_hash_layers": args.n_hash_layers,
      "num_hidden_layers": num_layers or args.n_layers,
      "num_nextn_predict_layers": args.n_mtp_layers,
      "compress_ratios": list(args.compress_ratios),
      "rms_norm_eps": args.norm_eps,
      "rope_theta": args.rope_theta,
      "compress_rope_theta": args.compress_rope_theta,
      "rope_scaling": {
          "type": "yarn",
          "factor": args.rope_factor,
          "beta_fast": args.beta_fast,
          "beta_slow": args.beta_slow,
          "original_max_position_embeddings": args.original_seq_len,
      },
      "routed_scaling_factor": args.route_scale,
      "scoring_func": args.score_func,
      "swiglu_limit": args.swiglu_limit,
      "norm_topk_prob": True,
      "max_position_embeddings": args.max_seq_len,
  }


def static_config_diff(mt_config, ref_args: ref_model.ModelArgs) -> list[str]:
  """Every architecture mismatch between a MaxText config and ModelArgs (empty == match)."""
  diffs = []
  for ref_field, mt_field in ARCH_FIELD_MAP:
    ref_val, mt_val = getattr(ref_args, ref_field), getattr(mt_config, mt_field)
    if ref_val != mt_val:
      diffs.append(f"{ref_field}={ref_val!r} != {mt_field}={mt_val!r}")
  n = mt_config.num_decoder_layers
  if n > ref_args.n_layers:
    diffs.append(f"num_decoder_layers={n} > n_layers={ref_args.n_layers}")
  if list(mt_config.compress_ratios[:n]) != list(ref_args.compress_ratios[:n]):
    diffs.append(f"compress_ratios[:{n}] {list(mt_config.compress_ratios[:n])} != {list(ref_args.compress_ratios[:n])}")
  want_rope = "yarn" if ref_args.original_seq_len > 0 else "default"
  got_rope = getattr(mt_config.rope_type, "value", mt_config.rope_type)
  if got_rope != want_rope:
    diffs.append(f"rope_type={got_rope!r} != {want_rope!r} (original_seq_len={ref_args.original_seq_len})")
  if getattr(mt_config.decoder_block, "value", mt_config.decoder_block) != "deepseek4":
    diffs.append(f"decoder_block={mt_config.decoder_block!r}")
  if getattr(mt_config.attention_type, "value", mt_config.attention_type) != "compressed":
    diffs.append(f"attention_type={mt_config.attention_type!r}")
  for field, want in MAXTEXT_FIXED_FIELDS:
    if getattr(mt_config, field) != want:
      diffs.append(f"{field}={getattr(mt_config, field)!r} != {want!r}")
  if ref_args.hc_eps != MAXTEXT_HC_EPS:
    diffs.append(f"hc_eps={ref_args.hc_eps!r} != MaxText hard-coded {MAXTEXT_HC_EPS!r}")
  return diffs


# --------------------------------------------------------------------------------------
# MaxText model construction and flat state
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def maxtext_context(mt_config, mesh):
  with nn_partitioning.axis_rules(mt_config.logical_axis_rules), jax.set_mesh(mesh):
    yield


def build_maxtext_model(mt_config, seed: int = 0):
  """Production `models.Transformer` (token embedder + NNXDecoder) and its mesh."""
  mesh = maxtext_utils.get_mesh_from_config(mt_config)
  with maxtext_context(mt_config, mesh):
    model = models.Transformer(
        mt_config,
        mesh,
        quantizations.configure_quantization(mt_config),
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=seed, dropout=seed),
    )
  return model, mesh


def _flat_key(collection: str, path) -> str:
  parts = conversion_utils.param_key_parts_from_path(tuple(jax.tree_util.DictKey(p) for p in path))
  return "-".join([collection] + parts)


def _collection(var) -> str:
  return "params" if isinstance(var, nnx.Param) else type(var).__name__


def flat_state(state) -> dict[str, Any]:
  """{converter key: Variable} for every non-RNG leaf of an nnx State."""
  out = {}
  for path, var in nnx.to_flat_state(state):
    if isinstance(var, nnx.RngState):
      continue
    out[_flat_key(_collection(var), path)] = var
  return out


def flat_arrays(state, collection: str) -> dict[str, np.ndarray]:
  """{converter key: np.ndarray} for a State whose leaves are arrays of one collection."""
  out = {}
  for path, leaf in nnx.to_flat_state(state):
    val = leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf
    out[_flat_key(collection, path)] = np.asarray(val)
  return out


# --------------------------------------------------------------------------------------
# Weight transfer through the production converter
# --------------------------------------------------------------------------------------

_LAYER_KEY = re.compile(r"^layers\.(\d+)\.(.*)$")


def _remap_layer(key: str, layer_map: dict[int, int] | None) -> str:
  if not layer_map:
    return key
  m = _LAYER_KEY.match(key)
  if not m:
    return key
  return f"layers.{layer_map.get(int(m.group(1)), int(m.group(1)))}.{m.group(2)}"


def _to_numpy(t) -> np.ndarray:
  if isinstance(t, torch.Tensor):
    return t.detach().cpu().numpy()
  return np.asarray(t)


@dataclasses.dataclass
class TransferReport:
  """Accounting for one native -> MaxText transfer."""

  filled: list[str]
  dummy: list[str]
  dummy_all_ones: bool
  left_at_init: list[str]
  consumed: set[str]
  unconsumed: list[str]

  def summary(self) -> str:
    return (
        f"MaxText leaves filled={len(self.filled)} dummy(None-mapped)={len(self.dummy)} "
        f"(all == 1.0: {self.dummy_all_ones}) left_at_init={len(self.left_at_init)}; "
        f"native keys consumed={len(self.consumed)} unconsumed={len(self.unconsumed)}"
    )


def load_reference_into_maxtext(
    ref_state_dict: dict[str, Any],
    mt_module: nnx.Module,
    mt_config,
    hf_cfg_dict: dict[str, Any],
    scan_layers: bool | None = None,
    layer_map: dict[int, int] | None = None,
) -> TransferReport:
  """Fills every MaxText leaf (params, Tid2EidVar, MoEBiasVar) from native keys.

  Args:
    ref_state_dict: native-key tensors (e.g. reference `Transformer.state_dict()`).
    mt_module: `models.Transformer`; its state keys follow the converter naming.
    mt_config: MaxText config of `mt_module`.
    hf_cfg_dict: converter config dict sized like `mt_module` (see hf_config_from_ref_args).
    scan_layers: defaults to `mt_config.scan_layers`.
    layer_map: MaxText-side layer index -> native layer index (e.g. {3: 7, 4: 8}).

  Returns:
    TransferReport.
  """
  scan_layers = mt_config.scan_layers if scan_layers is None else scan_layers
  mapping = param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, mt_config, scan_layers)
  hooks = param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg_dict, mt_config, scan_layers, saving_to_hf=False)
  state = nnx.state(mt_module)
  entries = flat_state(state)
  abstract = {k: (i, tuple(v.get_value().shape)) for i, (k, v) in enumerate(entries.items())}
  keys = conversion_utils.validate_and_filter_param_map_keys(mapping.keys(), abstract.keys())

  consumed = set()

  def getter(key):
    native = _remap_layer(key, layer_map)
    consumed.add(native)
    return _to_numpy(ref_state_dict[native])

  final = [None] * len(abstract)
  for mt_key in keys:
    idx, shape = to_maxtext._get_maxtext_indices_and_shapes(mt_key, abstract)  # pylint: disable=protected-access
    load_fn = to_maxtext._get_hf_loading_function(  # pylint: disable=protected-access
        mapping[mt_key], getter, hooks.get(mt_key), shape, mt_config, mt_key
    )
    to_maxtext._get_maxtext_weight(  # pylint: disable=protected-access
        load_fn, idx, shape, mt_key, final, save_dtype=None, use_lazy_load=False
    )

  filled, dummy, left = [], [], []
  dummy_all_ones = True
  for (key, var), arr in zip(entries.items(), final):
    if arr is None:
      left.append(key)
      continue
    var.set_value(jnp.asarray(arr, dtype=var.get_value().dtype))
    if mapping.get(key, "") is None:
      dummy.append(key)
      dummy_all_ones &= bool(np.all(np.asarray(arr) == 1.0))
    else:
      filled.append(key)
  nnx.update(mt_module, state)
  return TransferReport(
      filled=filled,
      dummy=dummy,
      dummy_all_ones=dummy_all_ones,
      left_at_init=left,
      consumed=consumed,
      unconsumed=sorted(set(ref_state_dict) - consumed),
  )


class _ConfigView:
  """Read-only config with some fields overridden."""

  def __init__(self, cfg, **overrides):
    self._cfg = cfg
    self._overrides = overrides

  def __getattr__(self, name):
    return self._overrides[name] if name in self._overrides else getattr(self._cfg, name)


def _saver_config(key, mt_config, fix_stack_axis: bool):
  """Config handed to process_maxtext_param for `key`.

  With scan_layers=True, process_maxtext_param slices every 1-D target list on
  param_scan_axis. That is wrong for (a) the unscanned prefix layers' expert-stacked
  MoE weights (expert axis 0) and (b) scanned MoEBiasVar [blocks, experts] (layer
  axis 0). `fix_stack_axis` supplies the actual stacking axis for those keys.
  """
  first = key[0] if isinstance(key, tuple) else key
  if not fix_stack_axis or not mt_config.scan_layers:
    return mt_config
  if "scanned_blocks" not in first:
    return _ConfigView(mt_config, scan_layers=False)
  if first.startswith("MoEBiasVar-"):
    return _ConfigView(mt_config, param_scan_axis=0)
  return mt_config


def native_from_maxtext(
    mt_flat: dict[str, np.ndarray],
    mt_config,
    hf_cfg_dict: dict[str, Any],
    scan_layers: bool | None = None,
    layer_map: dict[int, int] | None = None,
    fix_stack_axis: bool = True,
) -> dict[str, np.ndarray]:
  """Expresses MaxText-keyed tensors (weights or dW) in native keys via the saver.

  Valid for gradients because every DeepSeek-V4 hook is a pure layout map
  (transpose / reshape / slice / concat), i.e. a permutation of entries.
  None-mapped leaves (mhc_norm dummies) have no native counterpart and are dropped.
  `fix_stack_axis=False` runs the production saver unmodified (see _saver_config).
  """
  scan_layers = mt_config.scan_layers if scan_layers is None else scan_layers
  mapping = param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, mt_config, scan_layers)
  hooks = param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg_dict, mt_config, scan_layers, saving_to_hf=True)
  # Composite-key folding identical to to_huggingface._get_model_mappings.
  for hook_key in list(hooks.keys()):
    if isinstance(hook_key, tuple):
      hf_path = mapping.get(hook_key[0])
      if hf_path is not None:
        mapping[hook_key] = hf_path
        for k in hook_key:
          mapping.pop(k, None)
  shapes = hf_shape.DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE(hf_cfg_dict)
  keys = conversion_utils.validate_and_filter_param_map_keys(mapping.keys(), mt_flat.keys())
  out = {}
  for key in keys:
    weight = [jnp.asarray(mt_flat[k]) for k in key] if isinstance(key, tuple) else jnp.asarray(mt_flat[key])
    cfg = _saver_config(key, mt_config, fix_stack_axis)
    for hf_path, arr in conversion_utils.process_maxtext_param(key, weight, mapping, hooks, shapes, cfg):
      out[_remap_layer(hf_path, layer_map)] = np.asarray(arr)
  return out


# --------------------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------------------

_REF_DTYPE_NAME = {torch.float32: "fp32", torch.float64: "fp64"}


def build_reference(ref_state_dict: dict[str, torch.Tensor] | None, args: ref_model.ModelArgs, dtype=torch.float32):
  """Fresh reference Transformer in `dtype` (cast before loading, so pinned bf16 params stay exact).

  Uses Module.float()/double(), which cast floating-point tensors only; Module.to(dtype)
  would also cast the complex `freqs_cis` buffers to real and drop the RoPE sine.
  """
  ref_kernel.set_mode("train")
  model = ref_model.Transformer(dataclasses.replace(args, dtype=_REF_DTYPE_NAME[dtype]))
  model = {torch.float32: model.float, torch.float64: model.double}[dtype]()
  if ref_state_dict is not None:
    model.load_state_dict(ref_state_dict, strict=True)
  return model


def init_reference_state(args: ref_model.ModelArgs, seed: int) -> dict[str, torch.Tensor]:
  """Seeded non-degenerate fp32 reference weights (native keys, incl. mtp.*)."""
  model = build_reference(None, args, torch.float32)
  R.init_params(model, args, torch.Generator().manual_seed(seed))
  return {k: v.detach().clone() for k, v in model.state_dict().items()}


def ref_embed(model, tokens: torch.Tensor) -> torch.Tensor:
  """Transformer.forward embedding + hc expansion (model.py:846-848)."""
  h = model.embed(tokens)
  return h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)


def ref_layers(model, layer_ids, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
  for i in layer_ids:
    h = model.layers[i](h, 0, tokens)
  return h


def ref_head_logits(model, h: torch.Tensor) -> torch.Tensor:
  """Full-sequence logits (harness re-expression of ParallelHead.forward, model.py:761-778).

  ParallelHead.get_logits keeps only the last token; here the same hc_head -> norm ->
  linear chain is applied to every position.
  """
  x = model.head.hc_head(h, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
  x = model.norm(x)
  return F.linear(x.float(), model.head.weight)


def ref_masked_xent(logits: torch.Tensor, targets: torch.Tensor, target_seg: torch.Tensor):
  xent = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view(targets.shape)
  mask = (target_seg != 0).to(xent.dtype)
  return (xent * mask).sum(), mask.sum()


# --------------------------------------------------------------------------------------
# MaxText subgroup functions (production modules and entry points only)
# --------------------------------------------------------------------------------------


class MaxTextSubgroups:
  """Functional views of a `models.Transformer`; each fn is jit/vjp-able in (h, params, bias)."""

  def __init__(self, model, mt_config, mesh):
    self.cfg = mt_config
    self.mesh = mesh
    self.graphdef, self.params, self.bias, self.rest = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
    self.num_blocks = (mt_config.num_decoder_layers - mt_config.first_num_hash_layers) // 2

  def _merge(self, params, bias):
    # Fresh non-Param Variables owned by the current trace (flax 0.12 TraceContextError otherwise).
    bias = jax.tree.map(lambda t: t, bias)
    rest = jax.tree.map(lambda t: t, self.rest)
    return nnx.merge(self.graphdef, params, bias, rest)

  def embed(self, params, bias, tokens, positions):
    """Token embedding + mhc expansion exactly as NNXDecoder.__call__ (nnx_decoders.py:1785-1800)."""
    m = self._merge(params, bias)
    y = m.decoder._apply_embedding(m.token_embedder, tokens, positions, True, MODEL_MODE_TRAIN)  # pylint: disable=protected-access
    expand, _ = mhc.get_functions(self.cfg.mhc_expansion_rate)
    return expand(y)

  def s1(self, h, params, bias, tokens, segs, positions):
    """Unrolled prefix layers, as NNXDecoder._apply_deepseek4_scanned_blocks step 1."""
    d = self._merge(params, bias).decoder
    for i in range(self.cfg.first_num_hash_layers):
      h, _ = getattr(d, f"layers_{i}")(
          h, segs, positions, True, MODEL_MODE_TRAIN, previous_chunk=None, slot=None, decoder_input_tokens=tokens
      )
    return h

  def blocks(self, h, params, bias, tokens, segs, positions):
    """All scanned blocks, as NNXDecoder._apply_deepseek4_scanned_blocks step 2."""
    d = self._merge(params, bias).decoder
    y, _, _ = d._apply_layers_sequentially(  # pylint: disable=protected-access
        d.scanned_blocks,
        h,
        segs,
        positions,
        True,
        MODEL_MODE_TRAIN,
        length=self.num_blocks,
        metadata_axis_name="scanned_blocks",
        previous_chunk=None,
        slot=None,
        decoder_input_tokens=tokens,
    )
    return y

  def head(self, h, params, bias, targets, target_segs):
    """hc_head + apply_output_head + train.py masked cross entropy -> (logits, xent_sum)."""
    m = self._merge(params, bias)
    hidden = m.decoder.hc_head(h)
    logits = m.decoder.apply_output_head(m.token_embedder, hidden, True, MODEL_MODE_TRAIN)
    one_hot = jax.nn.one_hot(targets, self.cfg.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=0.0)
    xent = xent * (target_segs != 0)
    return logits, jnp.sum(xent)

  def s3(self, h, params, bias, tokens, segs, positions, targets, target_segs):
    return self.head(self.blocks(h, params, bias, tokens, segs, positions), params, bias, targets, target_segs)

  def full(self, params, bias, tokens, segs, positions):
    """Production Transformer.__call__ -> logits."""
    m = self._merge(params, bias)
    return m(tokens, positions, decoder_segment_ids=segs, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)


def vjp_runner(fn: Callable) -> Callable:
  """jit(h, params, bias, ct) -> (out, dh, dparams, dbias) for fn(h, params, bias)."""

  def run(h, params, bias, ct):
    out, pull = jax.vjp(fn, h, params, bias)
    dh, dparams, dbias = pull(ct)
    return out, dh, dparams, dbias

  return jax.jit(run)


# --------------------------------------------------------------------------------------
# Subgroup suite (shared by the tiny CPU test and the noise-floor script)
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Subgroup:
  """A MaxText model slice and the native layers it reproduces."""

  name: str
  num_layers: int  # MaxText model depth (prefix + scanned blocks).
  ref_layers: tuple[int, ...]  # Native layers applied by this subgroup.
  layer_map: dict[int, int] | None = None
  head: bool = False


def default_subgroups(args: ref_model.ModelArgs) -> tuple[Subgroup, ...]:
  """S1 prefix, S2 first two blocks, S3 last block + head (layer_map onto the last block)."""
  n, p = args.n_layers, args.n_hash_layers
  if (n - p) % 2 or n - p < 4:
    raise ValueError(f"need an even number (>= 4) of post-prefix layers, got {n - p}")
  last = (n - 2, n - 1)
  if tuple(args.compress_ratios[i] for i in last) != tuple(args.compress_ratios[p : p + 2]):
    raise ValueError("last block pattern differs from the first block; S3 layer_map would be invalid")
  return (
      Subgroup("S1", p + 2, tuple(range(p))),
      Subgroup("S2", p + 4, tuple(range(p, p + 4))),
      Subgroup("S3", p + 2, last, layer_map={p: last[0], p + 1: last[1]}, head=True),
  )


def head_native_keys(ref_state_dict) -> list[str]:
  return sorted(k for k in ref_state_dict if not k.startswith(("layers.", "mtp.", "embed.")))


def subgroup_native_keys(sg: Subgroup, ref_state_dict) -> list[str]:
  keys = [k for k in ref_state_dict if (m := _LAYER_KEY.match(k)) and int(m.group(1)) in sg.ref_layers]
  return sorted(keys + (head_native_keys(ref_state_dict) if sg.head else []))


@dataclasses.dataclass
class LoadedModel:
  cfg: Any
  model: Any
  mesh: Any
  report: TransferReport
  views: "MaxTextSubgroups"
  hf_cfg: dict[str, Any]
  layer_map: dict[int, int] | None


def build_loaded_model(args, seq, num_layers, ref_state_dict, layer_map=None, dtype="float32") -> LoadedModel:
  cfg = make_config(args, seq, dtype, num_layers)
  model, mesh = build_maxtext_model(cfg)
  hf_cfg = hf_config_from_ref_args(args, num_layers)
  report = load_reference_into_maxtext(ref_state_dict, model, cfg, hf_cfg, layer_map=layer_map)
  return LoadedModel(cfg, model, mesh, report, MaxTextSubgroups(model, cfg, mesh), hf_cfg, layer_map)


def make_inputs(args, seq: int, seed: int, n_masked: int = 8) -> dict[str, np.ndarray]:
  """Random tokens/targets; the last `n_masked` targets per row are loss-masked."""
  g = np.random.default_rng(seed)
  b = args.max_batch_size
  tseg = np.ones((b, seq), np.int32)
  tseg[:, seq - n_masked :] = 0
  return {
      "tokens": g.integers(0, args.vocab_size, (b, seq)),
      "targets": g.integers(0, args.vocab_size, (b, seq)),
      "target_segs": tseg,
      "segs": np.ones((b, seq), np.int32),
      "pos": np.broadcast_to(np.arange(seq), (b, seq)).copy(),
  }


def reference_boundaries(ref_state_dict, args, inputs) -> list[np.ndarray]:
  """Reference fp32 hidden states [embed, after layer 0, ..., after layer n-1] (Schema A inputs)."""
  model = build_reference(ref_state_dict, args, torch.float32)
  tt = torch.tensor(inputs["tokens"])
  with torch.no_grad():
    h = ref_embed(model, tt)
    out = [h.numpy()]
    for i in range(args.n_layers):
      h = model.layers[i](h, 0, tt)
      out.append(h.numpy())
  return out


class Suite:
  """Schema A forward + K-cotangent backward for each subgroup, on both implementations."""

  def __init__(self, args, seq: int, weight_seed: int, data_seed: int, num_cotangents: int = 4):
    self.args = args
    self.seq = seq
    self.k = num_cotangents
    self.ref_sd = init_reference_state(args, weight_seed)
    self.inputs = make_inputs(args, seq, data_seed)
    self.bounds = reference_boundaries(self.ref_sd, args, self.inputs)
    self.subgroups = default_subgroups(args)
    self.models = {
        sg.name: build_loaded_model(args, seq, sg.num_layers, self.ref_sd, sg.layer_map) for sg in self.subgroups
    }
    g = np.random.default_rng(data_seed + 1)
    b, d, v = args.max_batch_size, args.dim, args.vocab_size
    self.cotangents = {}
    for sg in self.subgroups:
      if sg.head:
        self.cotangents[sg.name] = [
            (g.standard_normal((b, seq, v)).astype(np.float32), np.float32(g.standard_normal())) for _ in range(self.k)
        ]
      else:
        self.cotangents[sg.name] = [
            g.standard_normal((b, seq, args.hc_mult, d)).astype(np.float32) for _ in range(self.k)
        ]

  def h_in(self, sg: Subgroup) -> np.ndarray:
    return self.bounds[sg.ref_layers[0]]

  def maxtext_fn(self, sg: Subgroup):
    v, x = self.models[sg.name].views, self.inputs
    if sg.head:
      return lambda h, p, b: v.s3(h, p, b, x["tokens"], x["segs"], x["pos"], x["targets"], x["target_segs"])
    if sg.name == "S1":
      return lambda h, p, b: v.s1(h, p, b, x["tokens"], x["segs"], x["pos"])
    return lambda h, p, b: v.blocks(h, p, b, x["tokens"], x["segs"], x["pos"])

  def maxtext_results(self) -> dict[str, dict[str, Any]]:
    """Schema A MaxText forward + VJP backward across all subgroups."""
    res = {}
    for sg in self.subgroups:
      lm = self.models[sg.name]
      v = lm.views
      mapping = param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(lm.hf_cfg, lm.cfg, lm.cfg.scan_layers)
      run = vjp_runner(self.maxtext_fn(sg))
      r = {"dh": [], "dW": [], "dbias_max": [], "unmapped_grad_norm": []}
      with maxtext_context(lm.cfg, lm.mesh):
        for ct in self.cotangents[sg.name]:
          ct_j = jax.tree.map(jnp.asarray, ct)
          out, dh, dp, db = run(jnp.asarray(self.h_in(sg)), v.params, v.bias, ct_j)
          flat = flat_arrays(dp, "params")
          r["dh"].append(np.asarray(dh))
          r["dW"].append(native_from_maxtext(flat, lm.cfg, lm.hf_cfg, layer_map=lm.layer_map))
          r["dbias_max"].append(
              max((float(np.abs(x).max()) for x in flat_arrays(db, "MoEBiasVar").values()), default=0.0)
          )
          r["unmapped_grad_norm"].append(
              {k: float(np.linalg.norm(a)) for k, a in flat.items() if mapping.get(k, "") is None}
          )
        if sg.head:
          r["logits"], r["xent_sum"] = np.asarray(out[0]), float(out[1])
          r["total_weights"] = float(jnp.sum(jnp.asarray(self.inputs["target_segs"]) != 0))
        else:
          r["out"] = np.asarray(out)
      res[sg.name] = r
    return res

  def reference_results(self, dtype=torch.float32) -> dict[str, dict[str, Any]]:
    """Schema A PyTorch reference forward + autograd backward across all subgroups."""
    res = {}
    tt = torch.tensor(self.inputs["tokens"])
    promo = R.fp64_promotion() if dtype == torch.float64 else contextlib.nullcontext()
    with promo:
      for sg in self.subgroups:
        keys = subgroup_native_keys(sg, self.ref_sd)
        r = {"dh": [], "dW": [], "no_grad": None}
        for ct in self.cotangents[sg.name]:
          model = build_reference(self.ref_sd, self.args, dtype)
          params = dict(model.named_parameters())
          train = [k for k in keys if params[k].requires_grad]
          h = torch.tensor(self.h_in(sg), dtype=dtype, requires_grad=True)
          y = ref_layers(model, sg.ref_layers, h, tt)
          if sg.head:
            logits = ref_head_logits(model, y)
            xs, tw = ref_masked_xent(
                logits, torch.tensor(self.inputs["targets"]), torch.tensor(self.inputs["target_segs"])
            )
            outs, cts = (logits, xs), (torch.tensor(ct[0], dtype=dtype), torch.tensor(float(ct[1]), dtype=dtype))
          else:
            outs, cts = (y,), (torch.tensor(ct, dtype=dtype),)
          grads = torch.autograd.grad(outs, [h] + [params[k] for k in train], cts, allow_unused=True)
          r["dh"].append(grads[0].detach().numpy())
          r["dW"].append({k: g.detach().numpy() for k, g in zip(train, grads[1:]) if g is not None})
          r["no_grad"] = sorted(set(keys) - set(r["dW"][-1]))
        if sg.head:
          r["logits"], r["xent_sum"], r["total_weights"] = logits.detach().numpy(), float(xs.detach()), float(tw.detach())
        else:
          r["out"] = y.detach().numpy()
        res[sg.name] = r
    return res

  def maxtext_chain(self) -> tuple[np.ndarray, float]:
    """Schema B: MaxText embed -> S1 -> S2 -> S3 chained on MaxText activations."""
    x = self.inputs
    s1, s2, s3 = (self.models[sg.name] for sg in self.subgroups)
    with maxtext_context(s1.cfg, s1.mesh):
      h = jax.jit(s1.views.embed)(s1.views.params, s1.views.bias, x["tokens"], x["pos"])
      h = jax.jit(self.maxtext_fn(self.subgroups[0]))(h, s1.views.params, s1.views.bias)
    with maxtext_context(s2.cfg, s2.mesh):
      h = jax.jit(self.maxtext_fn(self.subgroups[1]))(h, s2.views.params, s2.views.bias)
    with maxtext_context(s3.cfg, s3.mesh):
      logits, xent = jax.jit(self.maxtext_fn(self.subgroups[2]))(h, s3.views.params, s3.views.bias)
    return np.asarray(logits), float(xent)

  def reference_full(self, dtype=torch.float32) -> tuple[np.ndarray, float]:
    """Reference embed -> all layers -> full-sequence head -> masked xent sum."""
    model = build_reference(self.ref_sd, self.args, dtype)
    x = self.inputs
    tt = torch.tensor(x["tokens"])
    promo = R.fp64_promotion() if dtype == torch.float64 else contextlib.nullcontext()
    with promo, torch.no_grad():
      logits = ref_head_logits(model, ref_layers(model, range(self.args.n_layers), ref_embed(model, tt), tt))
      xs, _ = ref_masked_xent(logits, torch.tensor(x["targets"]), torch.tensor(x["target_segs"]))
    return logits.numpy(), float(xs)

  def maxtext_full_logits(self, full: LoadedModel) -> np.ndarray:
    """Production Transformer.__call__ of an n_layers model loaded from the same weights."""
    x = self.inputs
    with maxtext_context(full.cfg, full.mesh):
      return np.asarray(jax.jit(full.views.full)(full.views.params, full.views.bias, x["tokens"], x["segs"], x["pos"]))


def packed_isolation_run(args, ref_state_dict, doc_len: int, data_seed: int, ref_dtypes=(torch.float32,)):
  """MaxText S1+S2 on two packed docs of `doc_len` (segment ids 1, 2; positions restart).

  Returns MaxText outputs and d(h_in) for the original tokens and for doc 1 perturbed
  (same cotangent), plus the reference S1+S2 output on doc 2 alone per dtype.
  """
  p, seq, b = args.n_hash_layers, 2 * doc_len, args.max_batch_size
  lm = build_loaded_model(args, seq, p + 4, ref_state_dict)
  v = lm.views
  g = np.random.default_rng(data_seed)
  tokens = g.integers(0, args.vocab_size, (b, seq))
  perturbed = tokens.copy()
  perturbed[:, :doc_len] = (tokens[:, :doc_len] + g.integers(1, args.vocab_size, (b, doc_len))) % args.vocab_size
  segs = np.concatenate([np.ones((b, doc_len), np.int32), np.full((b, doc_len), 2, np.int32)], axis=1)
  pos = np.tile(np.arange(doc_len), 2)[None].repeat(b, 0)
  ct = jnp.asarray(g.standard_normal((b, seq, args.hc_mult, args.dim)), jnp.float32)

  @jax.jit
  def run(toks):
    h0 = v.embed(v.params, v.bias, toks, pos)

    def fn(h):
      return v.blocks(v.s1(h, v.params, v.bias, toks, segs, pos), v.params, v.bias, toks, segs, pos)

    out, pull = jax.vjp(fn, h0)
    return out, pull(ct)[0]

  with maxtext_context(lm.cfg, lm.mesh):
    out, dh = (np.asarray(t) for t in run(tokens))
    out_p, dh_p = (np.asarray(t) for t in run(perturbed))
  doc2 = torch.tensor(tokens[:, doc_len:])
  ref_args = dataclasses.replace(args, max_seq_len=doc_len)
  ref_out = {}
  for dtype in ref_dtypes:
    model = build_reference(ref_state_dict, ref_args, dtype)
    promo = R.fp64_promotion() if dtype == torch.float64 else contextlib.nullcontext()
    with promo, torch.no_grad():
      ref_out[dtype] = ref_layers(model, range(p + 4), ref_embed(model, doc2), doc2).numpy()
  return {"out": out, "dh": dh, "out_p": out_p, "dh_p": dh_p, "ref_out": ref_out, "doc_len": doc_len}


def worst(metrics: list[tuple[str, dict[str, float]]]) -> dict[str, Any]:
  """Max rel_l2 / max_abs and min cos over (tag, metrics) pairs, with the worst tag."""
  tag, m = max(metrics, key=lambda t: t[1]["rel_l2"])
  return {
      "rel_l2": m["rel_l2"],
      "cos": min(x["cos"] for _, x in metrics),
      "max_abs": max(x["max_abs"] for _, x in metrics),
      "worst": tag,
  }


def compare_results(actual, reference, subgroups, ref_state_dict) -> dict[str, dict[str, Any]]:
  """Per-quantity worst-case metrics; dW over every subgroup key with a reference grad."""
  out = {}
  for sg in subgroups:
    a, r = actual[sg.name], reference[sg.name]
    for q in ("logits",) if sg.head else ("out",):
      out[f"{sg.name}/{q}"] = worst([(q, compare(a[q], r[q]))])
    if sg.head:
      out[f"{sg.name}/xent_sum"] = worst([("xent_sum", compare([a["xent_sum"]], [r["xent_sum"]]))])
    out[f"{sg.name}/dh"] = worst([(f"ct{i}", compare(x, y)) for i, (x, y) in enumerate(zip(a["dh"], r["dh"]))])
    keys = subgroup_native_keys(sg, ref_state_dict)
    dw = []
    for i, (x, y) in enumerate(zip(a["dW"], r["dW"])):
      dw += [(f"ct{i}:{k}", compare(x[k], y[k])) for k in keys if k in y]
    out[f"{sg.name}/dW"] = worst(dw)
  return out


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def compare(actual, reference) -> dict[str, float]:
  """max_abs, rel_l2 (||a-r|| / ||r||) and cosine of `actual` vs `reference`."""
  a = np.asarray(actual, np.float64).ravel()
  r = np.asarray(reference, np.float64).ravel()
  if a.shape != r.shape:
    raise ValueError(f"shape mismatch {a.shape} vs {r.shape}")
  d = a - r
  na, nr = np.linalg.norm(a), np.linalg.norm(r)
  if nr > 0:
    rel = float(np.linalg.norm(d) / nr)
  else:
    rel = 0.0 if na == 0 else float("inf")
  if na > 0 and nr > 0:
    cos = float(np.dot(a, r) / (na * nr))
  else:
    cos = 1.0 if na == nr else 0.0
  return {"max_abs": float(np.max(np.abs(d))) if d.size else 0.0, "rel_l2": rel, "cos": cos}


def topk_agreement(actual: np.ndarray, reference: np.ndarray) -> float:
  """Fraction of rows whose top-k index sets (incl. -1 padding) are identical."""
  k = reference.shape[-1]
  a = np.sort(np.asarray(actual).reshape(-1, k), axis=-1)
  r = np.sort(np.asarray(reference).reshape(-1, k), axis=-1)
  return float(np.mean(np.all(a == r, axis=-1)))


# --------------------------------------------------------------------------------------
# Routing / indexer capture
# --------------------------------------------------------------------------------------


def _record(records, kind, tag, *arrays):
  records.append((kind, tag) + tuple(np.asarray(x) for x in arrays))


@contextlib.contextmanager
def capture_maxtext_topk(records: list):
  """TEST-ONLY patch: records MoE top-k (RoutedMoE.get_topk) and CSA indexer top-k.

  Appends ("moe", is_hash, indices[B,S,k], gate_logits[B,S,E]) and ("indexer", None,
  indices[B,S,k]) in program order via ordered host callbacks (works inside scan).
  """
  orig_topk = moe.RoutedMoE.get_topk
  orig_indexer = attention_compressed.DeepseekV4Indexer.__call__

  def get_topk(self, gate_logits, *args, **kwargs):
    weights, indices = orig_topk(self, gate_logits, *args, **kwargs)
    cb = functools.partial(_record, records, "moe", bool(self.is_hash_routing))
    jax.debug.callback(cb, indices, gate_logits, ordered=True)
    return weights, indices

  def indexer_call(self, *args, **kwargs):
    out = orig_indexer(self, *args, **kwargs)
    jax.debug.callback(functools.partial(_record, records, "indexer", None), out[0], ordered=True)
    return out

  moe.RoutedMoE.get_topk = get_topk
  attention_compressed.DeepseekV4Indexer.__call__ = indexer_call
  try:
    yield records
  finally:
    moe.RoutedMoE.get_topk = orig_topk
    attention_compressed.DeepseekV4Indexer.__call__ = orig_indexer


@contextlib.contextmanager
def capture_reference_topk(model, records: list):
  """Forward hooks recording (kind, is_hash, indices, selection scores) in call order.

  Gate scores are the biased sqrtsoftplus scores used by topk; indexer scores are
  recomputed from the hooked module exactly as Indexer.forward (model.py:454-469).
  """

  def gate_hook(module, inputs, output):
    x = inputs[0]
    scores = F.softplus(F.linear(x.float(), module.weight.float())).sqrt()
    if module.bias is not None:
      scores = scores + module.bias
    _record(records, "moe", bool(module.hash), output[1].detach(), scores.detach())

  def indexer_hook(module, inputs, output):
    x, qr, start_pos = inputs[0], inputs[1], inputs[2]
    seqlen, ratio = x.shape[1], module.compress_ratio
    with torch.no_grad():
      q = module.wq_b(qr).unflatten(-1, (module.n_local_heads, module.head_dim))
      ref_model.apply_rotary_emb(q[..., -module.rope_head_dim :], module.freqs_cis[start_pos : start_pos + seqlen])
      q = ref_model.rotate_activation(q)
      w = module.weights_proj(x) * (module.softmax_scale * module.n_heads**-0.5)
      kv = module.kv_cache[: x.shape[0], : (start_pos + seqlen) // ratio]
      score = (torch.einsum("bshd,btd->bsht", q, kv).relu() * w.unsqueeze(-1)).sum(dim=2)
      mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
      score = score + torch.where(mask, float("-inf"), 0.0)
    idx = output.detach()
    # Prefill offsets valid entries by seqlen (model.py:552, :473); undo for block ids.
    _record(records, "indexer", None, torch.where(idx >= 0, idx - seqlen, idx), score)

  handles = []
  for mod in model.modules():
    if isinstance(mod, ref_model.Gate):
      handles.append(mod.register_forward_hook(gate_hook))
    elif isinstance(mod, ref_model.Indexer):
      handles.append(mod.register_forward_hook(indexer_hook))
  try:
    yield records
  finally:
    for h in handles:
      h.remove()


def valid_topk_fraction(actual: np.ndarray, reference: np.ndarray, scores: np.ndarray, atol: float) -> float:
  """Fraction of rows where `actual` is a valid top-k under the reference `scores`.

  A row is valid if the sorted reference scores of the actual picks equal those of the
  reference picks within `atol` (i.e. differences are tie-breaks only); -1 = masked.
  """
  k = reference.shape[-1]
  a = np.asarray(actual).reshape(-1, k)
  r = np.asarray(reference).reshape(-1, k)
  s = np.asarray(scores, np.float64).reshape(a.shape[0], -1)

  def picked(idx):
    vals = np.where(idx >= 0, np.take_along_axis(s, np.maximum(idx, 0), axis=-1), -np.inf)
    return np.sort(vals, axis=-1)

  pa, pr = picked(a), picked(r)
  with np.errstate(invalid="ignore"):
    same = np.where(pa == pr, True, np.abs(pa - pr) <= atol)
  return float(np.mean(np.all(same, axis=-1)))


# --------------------------------------------------------------------------------------
# Coverage tracing
# --------------------------------------------------------------------------------------


def layer_type(compress_ratio: int, has_indexer: bool, is_hash: bool) -> str:
  kind = "SWA" if compress_ratio == 0 else ("CSA" if compress_ratio == 4 else "HCA")
  return kind + ("+indexer" if has_indexer else "") + ("+hash" if is_hash else "+topk")


_STATIC_ATTRS = ("layer_idx", "compress_ratio", "compress_rate", "is_hash_routing")


def _maxtext_signature(module) -> tuple:
  """Extracts structural configuration signature of a MaxText layer module."""
  attrs = []
  for name in _STATIC_ATTRS:
    val = getattr(module, name, None)
    if isinstance(val, (bool, int)) and not isinstance(val, jax.Array):
      attrs.append((name, val))
  attn = getattr(module, "self_attention", None)
  mlp = getattr(module, "mlp", None)
  if attn is not None and mlp is not None and hasattr(attn, "compress_ratio"):
    csa = getattr(attn, "csa_compressor", None)
    has_indexer = csa is not None and getattr(csa, "indexer", None) is not None
    attrs.append(("layer_type", layer_type(attn.compress_ratio, has_indexer, bool(mlp.MoeBlock_0.is_hash_routing))))
  return (type(module).__name__,) + tuple(attrs)


@contextlib.contextmanager
def trace_maxtext_modules(root: nnx.Module, records: set):
  """Temporarily wraps __call__ of every nnx.Module class in `root`'s graph."""
  classes = {}
  for _, node in nnx.iter_graph(root):
    if isinstance(node, nnx.Module):
      for c in type(node).__mro__:
        if c is not nnx.Module and issubclass(c, nnx.Module) and "__call__" in c.__dict__:
          classes.setdefault(c, c.__dict__["__call__"])

  def make_wrapper(cls, orig):
    @functools.wraps(orig)
    def wrapper(self_, *args, **kwargs):
      if next(c for c in type(self_).__mro__ if c in classes) is cls:
        records.add(_maxtext_signature(self_))
      return orig(self_, *args, **kwargs)

    return wrapper

  for cls, orig in classes.items():
    setattr(cls, "__call__", make_wrapper(cls, orig))
  try:
    yield records
  finally:
    for cls, orig in classes.items():
      setattr(cls, "__call__", orig)


def _torch_signature(module) -> tuple:
  attrs = []
  for name in ("compress_ratio", "hash", "rotate", "overlap"):
    val = getattr(module, name, None)
    if isinstance(val, (bool, int)):
      attrs.append((name, val))
  if isinstance(module, ref_model.Block):
    has_indexer = bool(module.attn.compress_ratio) and module.attn.indexer is not None
    attrs.append(("layer_type", layer_type(module.attn.compress_ratio, has_indexer, bool(module.ffn.gate.hash))))
  return (type(module).__name__,) + tuple(attrs)


@contextlib.contextmanager
def trace_torch_modules(model: torch.nn.Module, records: set):
  handles = [m.register_forward_hook(lambda mod, i, o: records.add(_torch_signature(mod))) for m in model.modules()]
  try:
    yield records
  finally:
    for h in handles:
      h.remove()


def layer_types_of(records: set) -> set[str]:
  return {dict(r[1:])["layer_type"] for r in records if "layer_type" in dict(r[1:])}
