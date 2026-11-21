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

r"""Layer-wise and block-wise forward verification utility for Qwen3.5 MoE models.

Streams HuggingFace Safetensors weights for Qwen3.5 MoE models (including
60-layer `qwen3.5-397b-a17b-fp8`) one decoder layer (or one 4-layer
`Qwen3_5ScannableBlock`) at a time onto a small TPU or CPU mesh, chains the
boundary activations across all layers (`embed -> layer_0 -> ... -> layer_{L-1}
-> decoder_norm -> logits_dense -> cross_entropy_loss`), and records:
  1. Per-layer residual RMS (overall, `doc1` where `segment_id == 1`, and
     `doc2plus` where `segment_id >= 2`), final-norm RMS, logit RMS/std, and
     cross-entropy losses (overall, `doc1`, `doc2plus`, position buckets, and
     temperature-scaled);
  2. Optional cross-run activation comparisons (`save_mb0_act_dir` /
     `ref_mb0_act_dir`): per-layer `chained_diff` (`h_A[l]` vs `h_B[l]`) and
     `teacher_forced_diff` (`f_A(h_B[l-1])` vs `f_B(h_B[l-1])`) relative L2 and
     cosine similarity on microbatch 0.

Related work:
-MaxText multi-layer debugger utility (`maxtext#5996`).

Usage (single-arm forward verification):
```bash
python3 -m maxtext.src.maxtext.utils.qwen3_5_layerwise \
    maxtext/src/maxtext/configs/base.yml \
    model_name=qwen3.5-397b-a17b-fp8 \
    ici_fsdp_parallelism=4 \
    per_device_batch_size=1.0 \
    max_target_length=4096 \
    --checkpoint_dir=/path/to/checkpoint \
    --output_json=/tmp/layerwise_arm_a.json \
    --save_mb0_act_dir=/tmp/layerwise_acts_arm_a
```

Usage (cross-run comparison against saved reference activations):
```bash
python3 -m maxtext.src.maxtext.utils.qwen3_5_layerwise \
    maxtext/src/maxtext/configs/base.yml \
    model_name=qwen3.5-397b-a17b-fp8 \
    ici_fsdp_parallelism=4 \
    per_device_batch_size=1.0 \
    max_target_length=4096 \
    --checkpoint_dir=/path/to/checkpoint \
    --ref_mb0_act_dir=/tmp/layerwise_acts_arm_a \
    --ref_json=/tmp/layerwise_arm_a.json \
    --include_teacher_forced \
    --output_json=/tmp/layerwise_comparison.json
```
"""

import collections
from collections.abc import Sequence
import concurrent.futures
import copy
import hashlib
import json
import os
import struct
import tempfile
import time
from typing import Any

from absl import app
from absl import flags
from flax import nnx
from flax.core.spmd import logical_axis_rules as axis_rules
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.common import common_types
from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline import tfds_data_processing
from maxtext.layers import embeddings
from maxtext.layers import linears
from maxtext.layers import quantizations
from maxtext.models import qwen3
from maxtext.models import qwen3_5
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
import ml_dtypes
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
MODEL_MODE_TRAIN = common_types.MODEL_MODE_TRAIN

flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Directory containing HuggingFace safetensors shards and index JSON.",
)
flags.DEFINE_string(
    "safetensors_index_json",
    None,
    "Optional explicit path to model.safetensors.index.json.",
)
flags.DEFINE_string(
    "staged_tokenizer_dir",
    None,
    "Optional local directory containing staged tokenizer assets.",
)
flags.DEFINE_string(
    "output_json",
    None,
    "Optional path to write JSON verification metrics.",
)
flags.DEFINE_string(
    "save_mb0_act_dir",
    None,
    "Optional directory to save/resume microbatch-0 activations and"
    " checkpoints.",
)
flags.DEFINE_string(
    "ref_mb0_act_dir",
    None,
    "Optional directory of reference microbatch-0 activations for cross-run"
    " diff.",
)
flags.DEFINE_string(
    "ref_json",
    None,
    "Optional reference run JSON path to compare via compare_layerwise_runs.",
)
flags.DEFINE_integer(
    "ref_hosts",
    None,
    "Number of dataloader hosts in the reference multi-host run.",
)
flags.DEFINE_integer(
    "step_idx",
    None,
    "Zero-based step index to advance the data iterator to before"
    " verification.",
)
flags.DEFINE_integer(
    "num_steps",
    None,
    "Number of steps of microbatches to verify starting from step_idx.",
)
flags.DEFINE_integer(
    "max_layers",
    None,
    "Optional cap on the number of layers to run (0 = all layers).",
)
flags.DEFINE_bool(
    "run_clean_single_doc",
    None,
    "Whether to also run a clean single-document (segment_id == 1) control"
    " batch.",
)
flags.DEFINE_bool(
    "include_teacher_forced",
    None,
    "When ref_mb0_act_dir is set, whether to also evaluate each layer on"
    " reference inputs.",
)
flags.DEFINE_bool(
    "run_blockwise",
    None,
    "If True, also run 4-layer Qwen3_5ScannableBlock streaming.",
)


_SAFETENSORS_DTYPE_MAP = {
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "BF16": ml_dtypes.bfloat16,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "U8": np.uint8,
    "I16": np.int16,
    "U16": np.uint16,
    "I32": np.int32,
    "U32": np.uint32,
    "I64": np.int64,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


class Qwen35SafetensorsReader:
  """Selective per-layer Safetensors reader with coalesced ranged reads."""

  def __init__(self, checkpoint_dir: str, index_json_path: str | None = None):
    self.checkpoint_dir = checkpoint_dir
    self._header_cache: dict[str, tuple[int, dict[str, Any]]] = {}
    self._cached_mappings: dict[
        int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = {}
    self.weight_map: dict[str, str] = {}
    self.total_bytes_read: int = 0
    self.total_read_seconds: float = 0.0
    self.total_range_reads: int = 0
    self.last_hf_tensors: dict[str, np.ndarray] = {}

    staged_index = index_json_path or os.environ.get(
        "QWEN_SAFETENSORS_INDEX_JSON", ""
    )
    index_path = (
        staged_index
        if (staged_index and tf.io.gfile.exists(staged_index))
        else os.path.join(checkpoint_dir, "model.safetensors.index.json")
    )
    single_path = os.path.join(checkpoint_dir, "model.safetensors")
    if tf.io.gfile.exists(index_path):
      with tf.io.gfile.GFile(index_path, "r") as f:
        idx_data = json.load(f)
      self.weight_map = dict(idx_data["weight_map"])
    elif tf.io.gfile.exists(single_path):
      _, hdr = self._get_shard_header("model.safetensors")
      for k in hdr:
        if k != "__metadata__":
          self.weight_map[k] = "model.safetensors"
    else:
      raise FileNotFoundError(
          "Neither model.safetensors.index.json nor model.safetensors found in"
          f" {checkpoint_dir}"
      )

  def _get_shard_header(self, shard_file: str) -> tuple[int, dict[str, Any]]:
    """Reads and caches the Safetensors JSON header for `shard_file`."""
    if shard_file in self._header_cache:
      return self._header_cache[shard_file]
    shard_path = os.path.join(self.checkpoint_dir, shard_file)
    t0 = time.perf_counter()
    with tf.io.gfile.GFile(shard_path, "rb") as f:
      raw_len = f.read(8)
      header_len = struct.unpack("<Q", raw_len)[0]
      header_bytes = f.read(header_len)
    self.total_read_seconds += time.perf_counter() - t0
    self.total_bytes_read += 8 + header_len
    self.total_range_reads += 1
    header = json.loads(header_bytes.decode("utf-8"))
    res = (8 + header_len, header)
    self._header_cache[shard_file] = res
    return res

  def read_tensors_batch(
      self,
      hf_keys: Sequence[str],
      max_gap_bytes: int = 2 * 1024 * 1024,
      max_span_bytes: int = 16 * 1024 * 1024,
      max_workers: int = 256,
  ) -> dict[str, np.ndarray]:
    """Reads a batch of HF tensor keys using parallel coalesced contiguous file reads."""
    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for k in dict.fromkeys(hf_keys):
      if k not in self.weight_map:
        raise KeyError(
            f"HF key {k!r} not found in checkpoint weight_map"
            f" ({len(self.weight_map)} keys)"
        )
      by_shard[self.weight_map[k]].append(k)

    uncached_shards = [sf for sf in by_shard if sf not in self._header_cache]
    if len(uncached_shards) == 1:
      self._get_shard_header(uncached_shards[0])
    elif len(uncached_shards) > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(max_workers, len(uncached_shards))
      ) as pool:
        list(pool.map(self._get_shard_header, uncached_shards))

    result: dict[str, np.ndarray] = {}
    tasks: list[
        tuple[
            str, int, int, int, list[tuple[int, int, str, str, tuple[int, ...]]]
        ]
    ] = []
    chunk_tasks: list[tuple[str, int, int, np.ndarray, int]] = []
    large_chunk_bytes = min(max_span_bytes, 16 * 1024 * 1024)

    for shard_file, keys in by_shard.items():
      data_offset, header = self._get_shard_header(shard_file)
      shard_path = os.path.join(self.checkpoint_dir, shard_file)
      small_items = []
      for k in keys:
        meta = header[k]
        start, end = meta["data_offsets"]
        nbytes = end - start
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        if nbytes > max_span_bytes and large_chunk_bytes > 0:
          np_dtype = _SAFETENSORS_DTYPE_MAP[dtype_str]
          arr = np.empty(shape, dtype=np_dtype)
          result[k] = arr
          arr_u8 = arr.view(np.uint8).reshape(-1)
          for c_off in range(0, nbytes, large_chunk_bytes):
            c_len = min(large_chunk_bytes, nbytes - c_off)
            chunk_tasks.append(
                (shard_path, data_offset + start + c_off, c_len, arr_u8, c_off)
            )
        else:
          small_items.append((start, end, k, dtype_str, shape))

      if small_items:
        small_items.sort(key=lambda x: x[0])
        cur_start, cur_end, cur_list = (
            small_items[0][0],
            small_items[0][1],
            [small_items[0]],
        )
        for item in small_items[1:]:
          s, e = item[0], item[1]
          if (s - cur_end) <= max_gap_bytes and (
              e - cur_start
          ) <= max_span_bytes:
            cur_end = max(cur_end, e)
            cur_list.append(item)
          else:
            tasks.append(
                (shard_path, data_offset, cur_start, cur_end, cur_list)
            )
            cur_start, cur_end, cur_list = s, e, [item]
        tasks.append((shard_path, data_offset, cur_start, cur_end, cur_list))

    def _read_span(task):
      shard_path, data_offset, span_start, span_end, span_items = task
      with tf.io.gfile.GFile(shard_path, "rb") as f:
        f.seek(data_offset + span_start)
        buf = f.read(span_end - span_start)
      local_res: dict[str, np.ndarray] = {}
      for s, _, k, dtype_str, shape in span_items:
        np_dtype = _SAFETENSORS_DTYPE_MAP[dtype_str]
        count = int(np.prod(shape)) if shape else 1
        arr = np.frombuffer(
            buf, dtype=np_dtype, count=count, offset=s - span_start
        ).reshape(shape)
        local_res[k] = arr.copy()
      return len(buf), local_res

    def _read_chunk(ctask):
      shard_path, abs_offset, c_len, arr_u8, c_off = ctask
      with tf.io.gfile.GFile(shard_path, "rb") as f:
        f.seek(abs_offset)
        buf = f.read(c_len)
      arr_u8[c_off : c_off + c_len] = np.frombuffer(
          buf, dtype=np.uint8, count=c_len
      )
      return len(buf)

    t0 = time.perf_counter()
    total_work = len(tasks) + len(chunk_tasks)
    num_workers = min(max_workers, max(1, total_work))
    if num_workers <= 1:
      for ctask in chunk_tasks:
        nbytes = _read_chunk(ctask)
        self.total_bytes_read += nbytes
        self.total_range_reads += 1
      for t in tasks:
        nbytes, sub = _read_span(t)
        self.total_bytes_read += nbytes
        self.total_range_reads += 1
        result.update(sub)
    else:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=num_workers
      ) as pool:
        chunk_futs = [pool.submit(_read_chunk, ct) for ct in chunk_tasks]
        span_futs = [pool.submit(_read_span, t) for t in tasks]
        for cf in chunk_futs:
          nbytes = cf.result()
          self.total_bytes_read += nbytes
          self.total_range_reads += 1
        for sf in span_futs:
          nbytes, sub = sf.result()
          self.total_bytes_read += nbytes
          self.total_range_reads += 1
          result.update(sub)
    self.total_read_seconds += time.perf_counter() - t0
    return result

  def get_hf_config_and_unscanned_mappings(
      self, config: Any, total_layers: int | None = None
  ):
    """Returns `(hf_config_dict, param_map_mt_to_hf, hook_fn_map_mt)` for unscanned layers."""
    model_key = config.model_name.replace("-Instruct", "")
    hf_cfg_obj = copy.deepcopy(hf_model_configs.HF_MODEL_CONFIGS[model_key])
    hf_cfg_dict = hf_cfg_obj.to_dict()
    num_layers = int(
        total_layers
        or getattr(config, "num_decoder_layers", 0)
        or hf_cfg_dict.get("num_hidden_layers", 60)
    )
    if num_layers in self._cached_mappings:
      return self._cached_mappings[num_layers]
    num_experts = int(
        getattr(config, "num_experts", 0) or hf_cfg_dict.get("num_experts", 512)
    )
    if "text_config" in hf_cfg_dict:
      hf_cfg_dict["text_config"]["num_hidden_layers"] = num_layers
      hf_cfg_dict["text_config"]["num_experts"] = num_experts
      hf_cfg_dict["num_hidden_layers"] = num_layers
      hf_cfg_dict["num_experts"] = num_experts
      for k, v in hf_cfg_dict["text_config"].items():
        hf_cfg_dict.setdefault(k, v)
    else:
      hf_cfg_dict["num_hidden_layers"] = num_layers
      hf_cfg_dict["num_experts"] = num_experts

    class _CfgProxy:

      def __getattr__(self, name):
        if name in ("num_decoder_layers", "base_num_decoder_layers"):
          return num_layers
        if name == "scan_layers":
          return False
        return getattr(config, name)

    proxy = _CfgProxy()
    mapping = param_mapping.QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(
        hf_cfg_dict, proxy, scan_layers=False
    )
    hooks = param_mapping.QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        hf_cfg_dict, proxy, scan_layers=False, saving_to_hf=False
    )
    res = (hf_cfg_dict, mapping, hooks)
    self._cached_mappings[num_layers] = res
    return res

  @staticmethod
  def _fast_ascontiguous(arr: np.ndarray) -> np.ndarray:
    """Fast contiguous copy for custom `ml_dtypes` (`float8_e4m3fn`, `bfloat16`) via uint views."""
    if arr.flags.c_contiguous:
      return arr
    itemsize = arr.dtype.itemsize
    u_dt = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(itemsize)
    if u_dt is not None and arr.dtype != u_dt:
      return np.ascontiguousarray(arr.view(u_dt)).view(arr.dtype)
    return np.ascontiguousarray(arr)

  def _transform_leaf_numpy(
      self,
      hf_source: str | list[str] | tuple[str, ...],
      hf_tensors: dict[str, np.ndarray],
      hook_fn: Any,
      target_shape: tuple[int, ...],
      target_dtype: Any,
  ) -> np.ndarray:
    """Applies MaxText HF-to-MaxText hook and stacking in host NumPy."""
    np_target_dtype = np.dtype(target_dtype)
    if isinstance(hf_source, (list, tuple)):
      if len(hf_source) == target_shape[0] and len(target_shape) >= 3:
        num_e = len(hf_source)
        out = np.empty(target_shape, dtype=np_target_dtype)
        u_dt = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(
            np_target_dtype.itemsize
        )

        def _fill_expert(idx: int):
          e_arr = hf_tensors[hf_source[idx]]
          if e_arr.dtype != np_target_dtype:
            e_arr = e_arr.astype(np_target_dtype)
          if u_dt is not None and e_arr.ndim == 2:
            out.view(u_dt)[idx] = (
                e_arr.view(u_dt).T if hook_fn is not None else e_arr.view(u_dt)
            )
          else:
            out[idx] = (
                np.swapaxes(e_arr, -1, -2) if hook_fn is not None else e_arr
            )

        if num_e >= 16 and int(np.prod(target_shape)) >= 4 * 1024 * 1024:
          with concurrent.futures.ThreadPoolExecutor(
              max_workers=min(32, num_e)
          ) as pool:
            list(pool.map(_fill_expert, range(num_e)))
        else:
          for i in range(num_e):
            _fill_expert(i)
        return out
      else:
        parts = [hf_tensors[k] for k in hf_source]
        arr = (
            hook_fn(parts, target_shape)
            if hook_fn is not None
            else np.concatenate(parts, axis=0)
        )
    else:
      raw = hf_tensors[hf_source]
      arr = hook_fn(raw, target_shape) if hook_fn is not None else raw

    if arr.dtype != np_target_dtype:
      arr = arr.astype(np_target_dtype)
    if (
        arr.ndim == 2
        and len(target_shape) == 2
        and arr.shape[0] == target_shape[0]
        and arr.shape[1] > target_shape[1]
    ):
      arr = arr[:, : target_shape[1]]
    if arr.shape != tuple(target_shape):
      raise ValueError(
          f"Transformed shape {arr.shape} != target_shape {target_shape} for"
          f" source {hf_source}"
      )
    return self._fast_ascontiguous(arr)

  @staticmethod
  def _flatten_abstract_state(
      abstract_state: Any,
  ) -> tuple[list[tuple[tuple[str, ...], Any]], Any]:
    """Flattens an NNX State or dict into `((path_tuple, leaf), ...)` and its `PyTreeDef`."""
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(
        abstract_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
    )
    items = []
    for key_path, leaf in path_leaves:
      path_tuple = tuple(
          str(getattr(k, "key", getattr(k, "name", k))) for k in key_path
      )
      items.append((path_tuple, leaf))
    return items, treedef

  @staticmethod
  def _leaf_meta_and_replace(abs_leaf: Any, new_val: Any = None):
    if isinstance(abs_leaf, nnx.Variable):
      meta = (
          abs_leaf.get_value()
          if hasattr(abs_leaf, "get_value")
          else abs_leaf.value
      )
      return meta, (abs_leaf.replace(new_val) if new_val is not None else None)
    return abs_leaf, new_val

  @staticmethod
  def host_to_sharded_array(
      np_arr: np.ndarray, sharding_obj: Any = None
  ) -> jax.Array:
    """Places a global host NumPy array onto a single- or multi-host JAX NamedSharding."""
    if sharding_obj is None:
      return jnp.asarray(np_arr)
    return jax.make_array_from_callback(
        np_arr.shape, sharding_obj, lambda idx: np_arr[idx]
    )

  def load_non_layer_params(
      self,
      abstract_non_layer: Any,
      config: Any,
      total_layers: int | None = None,
  ) -> Any:
    """Loads `token_embedder`, `decoder_norm`, and `logits_dense` parameters onto device."""
    _, mapping, hooks = self.get_hf_config_and_unscanned_mappings(
        config, total_layers=total_layers
    )
    non_layer_mt_keys = [
        "params-token_embedder-embedding",
        "params-decoder-decoder_norm-scale",
        "params-decoder-logits_dense-kernel",
    ]
    needed_hf = [mapping[k] for k in non_layer_mt_keys if k in mapping]
    hf_tensors = self.read_tensors_batch(needed_hf)

    items, treedef = self._flatten_abstract_state(abstract_non_layer)
    out_leaves = []
    for path_tuple, abs_leaf in items:
      val_meta, _ = self._leaf_meta_and_replace(abs_leaf)
      flat_k = "-".join(path_tuple)
      mt_key = flat_k if flat_k.startswith("params-") else f"params-{flat_k}"
      hf_src = mapping[mt_key]
      hook_fn = hooks.get(mt_key)
      np_arr = self._transform_leaf_numpy(
          hf_src, hf_tensors, hook_fn, val_meta.shape, val_meta.dtype
      )
      dev_arr = self.host_to_sharded_array(
          np_arr, getattr(val_meta, "sharding", None)
      )
      _, rep_leaf = self._leaf_meta_and_replace(abs_leaf, dev_arr)
      out_leaves.append(rep_leaf)
    return treedef.unflatten(out_leaves)

  def load_single_layer_numpy(
      self,
      layer_idx: int,
      abstract_layer_params: Any,
      config: Any,
      total_layers: int | None = None,
  ) -> dict[tuple[str, ...], np.ndarray]:
    """Loads a single `Qwen3_5DecoderLayer`'s leaves into host NumPy."""
    _, mapping, hooks = self.get_hf_config_and_unscanned_mappings(
        config, total_layers=total_layers
    )
    prefix = f"params-decoder-layers_{layer_idx}-"
    items, _ = self._flatten_abstract_state(abstract_layer_params)

    needed_hf: list[str] = []
    for path_tuple, _ in items:
      mt_key = prefix + "-".join(path_tuple)
      if mt_key not in mapping:
        raise KeyError(
            f"Layer {layer_idx} leaf {mt_key!r} missing from"
            " QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING"
        )
      src = mapping[mt_key]
      if isinstance(src, (list, tuple)):
        needed_hf.extend(src)
      else:
        needed_hf.append(src)

    hf_tensors = self.read_tensors_batch(needed_hf)
    self.last_hf_tensors = hf_tensors
    flat_np: dict[tuple[str, ...], np.ndarray] = {}
    for path_tuple, abs_leaf in items:
      val_meta, _ = self._leaf_meta_and_replace(abs_leaf)
      mt_key = prefix + "-".join(path_tuple)
      hf_src = mapping[mt_key]
      hook_fn = hooks.get(mt_key)
      flat_np[path_tuple] = self._transform_leaf_numpy(
          hf_src, hf_tensors, hook_fn, val_meta.shape, val_meta.dtype
      )
    return flat_np

  def load_single_layer_params_from_numpy(
      self,
      flat_np: dict[tuple[str, ...], np.ndarray],
      abstract_layer_params: Any,
  ) -> Any:
    """Places pre-loaded host NumPy layer parameters onto the target mesh."""
    items, treedef = self._flatten_abstract_state(abstract_layer_params)
    out_leaves = []
    for path_tuple, abs_leaf in items:
      val_meta, _ = self._leaf_meta_and_replace(abs_leaf)
      np_arr = flat_np[path_tuple]
      dev_arr = self.host_to_sharded_array(
          np_arr, getattr(val_meta, "sharding", None)
      )
      _, rep_leaf = self._leaf_meta_and_replace(abs_leaf, dev_arr)
      out_leaves.append(rep_leaf)
    return treedef.unflatten(out_leaves)

  def load_single_layer_params(
      self,
      layer_idx: int,
      abstract_layer_params: Any,
      config: Any,
      total_layers: int | None = None,
  ) -> Any:
    """Loads a single `Qwen3_5DecoderLayer`'s parameters onto the target mesh."""
    flat_np = self.load_single_layer_numpy(
        layer_idx, abstract_layer_params, config, total_layers=total_layers
    )
    return self.load_single_layer_params_from_numpy(
        flat_np, abstract_layer_params
    )

  @staticmethod
  def _expand_singleton_axes(
      arr: np.ndarray, target_shape: tuple[int, ...]
  ) -> np.ndarray:
    """Expands size-1 axes on `arr` as needed to match `target_shape`."""
    if arr.shape != tuple(target_shape):
      for ax, dim in enumerate(target_shape):
        if dim == 1 and (ax >= arr.ndim or arr.shape[ax] != 1):
          arr = np.expand_dims(arr, axis=ax)
          if arr.shape == tuple(target_shape):
            break
    return np.ascontiguousarray(arr)

  def load_single_block_params(
      self,
      block_idx: int,
      abstract_block_params: Any,
      config: Any,
      single_gdn_abstract: Any,
      single_attn_abstract: Any,
      total_layers: int | None = None,
  ) -> Any:
    """Loads a 4-layer `Qwen3_5ScannableBlock` (3 scanned GDN + 1 FullAttention) onto the mesh."""
    cycle = int(config.inhomogeneous_layer_cycle_interval)
    num_local = cycle - 1
    base_layer = block_idx * cycle
    scan_axis = int(config.param_scan_axis)

    local_np_list = [
        self.load_single_layer_numpy(
            base_layer + pos,
            single_gdn_abstract,
            config,
            total_layers=total_layers,
        )
        for pos in range(num_local)
    ]
    global_np = self.load_single_layer_numpy(
        base_layer + num_local,
        single_attn_abstract,
        config,
        total_layers=total_layers,
    )

    items, treedef = self._flatten_abstract_state(abstract_block_params)
    out_leaves = []
    for path_tuple, abs_leaf in items:
      val_meta, _ = self._leaf_meta_and_replace(abs_leaf)
      head = path_tuple[0]
      sub_path = path_tuple[1:]
      if head == "local_layers":
        stacked = np.stack([d[sub_path] for d in local_np_list], axis=scan_axis)
        np_arr = self._expand_singleton_axes(stacked, tuple(val_meta.shape))
      elif head == "global_layer":
        np_arr = self._expand_singleton_axes(
            global_np[sub_path], tuple(val_meta.shape)
        )
      else:
        raise KeyError(f"Unexpected block parameter prefix: {path_tuple}")

      if np_arr.shape != tuple(val_meta.shape):
        raise ValueError(
            f"Block {block_idx} leaf {path_tuple}: got {np_arr.shape}, expected"
            f" {val_meta.shape}"
        )
      dev_arr = self.host_to_sharded_array(
          np_arr, getattr(val_meta, "sharding", None)
      )
      _, rep_leaf = self._leaf_meta_and_replace(abs_leaf, dev_arr)
      out_leaves.append(rep_leaf)
    return treedef.unflatten(out_leaves)


def make_clean_single_doc_batch(batch: dict[str, Any]) -> dict[str, Any]:
  """Creates a clean 1-document-per-row batch by keeping only `doc1` (`segment == 1`)."""
  # For rows that already have a single document, this is a no-op. For packed
  # rows, tokens with segment >= 2 are masked out (segmentation = 0) so each row
  # contains strictly one contiguous document starting at position 0.
  in_seg = batch["inputs_segmentation"]
  tgt_seg = batch["targets_segmentation"]
  clean_in_seg = jnp.where(in_seg == 1, 1, 0).astype(in_seg.dtype)
  clean_tgt_seg = jnp.where(tgt_seg == 1, 1, 0).astype(tgt_seg.dtype)
  seq_len = batch["inputs"].shape[1]
  pos = jnp.broadcast_to(
      jnp.arange(seq_len, dtype=batch["inputs_position"].dtype)[None, :],
      batch["inputs"].shape,
  )
  return {
      "inputs": jnp.where(clean_in_seg == 1, batch["inputs"], 0),
      "inputs_position": jnp.where(clean_in_seg == 1, pos, 0),
      "inputs_segmentation": clean_in_seg,
      "targets": jnp.where(clean_tgt_seg == 1, batch["targets"], 0),
      "targets_position": jnp.where(clean_tgt_seg == 1, pos, 0),
      "targets_segmentation": clean_tgt_seg,
  }


class _NonLayerShell(nnx.Module):
  """Holds only `token_embedder`, `decoder_norm`, and `logits_dense` for streaming."""

  def __init__(self, config: Any, mesh: jax.sharding.Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.token_embedder = embeddings.Embed(
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        dtype=config.dtype,
        attend_dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
        embedding_init=jax.nn.initializers.normal(stddev=1.0),
        config=config,
        mesh=mesh,
        rngs=rngs,
    )
    self.positional_embedding = embeddings.PositionalEmbedding(
        embedding_dims=config.base_emb_dim
    )

    class _DecoderTail(nnx.Module):
      """Holds `decoder_norm` and `logits_dense` matching `Decoder` parameter paths."""

      def __init__(self, cfg: Any, m: jax.sharding.Mesh, *, r: nnx.Rngs):
        del m
        self.decoder_norm = qwen3.Qwen3NextRMSNorm(
            num_features=cfg.emb_dim,
            epsilon=cfg.normalization_layer_epsilon,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            shard_mode=cfg.shard_mode,
            rngs=r,
        )
        lm_head_axes = sharding.lm_head_logical_axes(
            cfg.lm_head_fsdp_as_vocab_tp
        )
        self.logits_dense = linears.DenseGeneral(
            in_features_shape=cfg.emb_dim,
            out_features_shape=cfg.vocab_size,
            weight_dtype=common_types.get_weight_dtype(cfg, "logits_dense"),
            dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
            kernel_axes=lm_head_axes[1],
            shard_mode=cfg.shard_mode,
            matmul_precision=cfg.matmul_precision,
            parameter_memory_host_offload=cfg.parameter_memory_host_offload,
            weight_quant=quantizations.get_weight_quant_config(
                cfg, "logits_dense"
            ),
            weight_grad_in_kernel_order=cfg.lm_head_weight_grad_in_kernel_order,
            rngs=r,
        )

    self.decoder = _DecoderTail(config, mesh, r=rngs)


class _RecordingQwen35ScannableBlock(qwen3_5.Qwen3_5ScannableBlock):
  """`Qwen3_5ScannableBlock` that records each sub-layer's valid-token squared-mean sum."""

  def _run_layer(
      self, layer, y, layer_kwargs, kv_cache=None, forced_routed_experts=None
  ):
    out, updated_kv = super()._run_layer(
        layer,
        y,
        layer_kwargs,
        kv_cache=kv_cache,
        forced_routed_experts=forced_routed_experts,
    )
    seg_ids = layer_kwargs.get("decoder_segment_ids")
    valid_f32 = (
        (seg_ids != 0).astype(jnp.float32)
        if seg_ids is not None
        else jnp.ones(out.shape[:2], dtype=jnp.float32)
    )
    tok_ms = jnp.mean(jnp.square(out.astype(jnp.float32)), axis=-1)
    layer.sow(nnx.Intermediate, "resid_ms_sum", jnp.sum(tok_ms * valid_f32))
    return out, updated_kv


def _attach_named_shardings(
    abstract_state: Any, mesh: jax.sharding.Mesh, rules: Any
) -> Any:
  """Attaches `NamedSharding` to each `VariableState` leaf of an `eval_shape` state."""
  with axis_rules(rules):
    named_sharding_state = sharding.nnx_construct_named_sharding(
        abstract_state, mesh
    )
    return jax.tree.map(
        lambda a, s: a.replace(
            jax.ShapeDtypeStruct(
                a.value.shape,
                a.value.dtype,
                sharding=s.get_value() if isinstance(s, nnx.Variable) else s,
            )
        ),
        abstract_state,
        named_sharding_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )


def _materialize_template(state: Any) -> Any:
  """Replaces `ShapeDtypeStruct` leaves in an NNX template state with concrete zeros."""

  def _to_zeros(v):
    if isinstance(v, nnx.Variable):
      meta = v.get_value() if hasattr(v, "get_value") else v.value
      if hasattr(meta, "shape") and hasattr(meta, "dtype"):
        return v.replace(jnp.zeros(meta.shape, meta.dtype))
      return v
    if hasattr(v, "shape") and hasattr(v, "dtype"):
      return jnp.zeros(v.shape, v.dtype)
    return v

  return jax.tree.map(
      _to_zeros, state, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )


def _clone_at_current_trace_level(state: Any) -> Any:
  """Re-instantiates NNX `Variable` wrappers at the active JAX trace level."""

  def _clone_var(v):
    if isinstance(v, nnx.Variable):
      val = v.get_value() if hasattr(v, "get_value") else v.value
      return v.replace(val)
    return v

  return jax.tree.map(
      _clone_var, state, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )


class Qwen35LayerwiseVerifier:
  """Streams Qwen3.5 weights layer-by-layer or block-by-block and verifies forward numerics."""

  def __init__(
      self,
      config: Any,
      mesh: jax.sharding.Mesh,
      checkpoint_dir: str,
      total_layers: int | None = None,
      safetensors_index_json: str | None = None,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quantizations.configure_quantization(config)
    self.reader = Qwen35SafetensorsReader(
        checkpoint_dir, index_json_path=safetensors_index_json
    )
    self.total_layers = total_layers or int(config.num_decoder_layers)
    self.cycle = int(config.inhomogeneous_layer_cycle_interval)

    # Do NOT enter jax.set_mesh(mesh) during nnx.eval_shape, because Flax's
    # _to_variable constructs NamedSharding(mesh, raw_pspec) before
    # remove_size_one_mesh_axis strips size-1 axes, raising DuplicateSpecError.
    with axis_rules(config.logical_axis_rules):

      def _init_shell():
        m = _NonLayerShell(config, mesh, rngs=nnx.Rngs(params=0))
        return nnx.split(m, nnx.Param, ...)

      self.shell_graphdef, abs_shell_params, shell_other = nnx.eval_shape(
          _init_shell
      )
      self.shell_other = _materialize_template(shell_other)
      self.abs_shell_params = _attach_named_shardings(
          abs_shell_params, mesh, config.logical_axis_rules
      )

      def _init_gdn():
        m = qwen3_5.Qwen3_5DecoderLayer(
            config=config,
            mesh=mesh,
            model_mode=common_types.MODEL_MODE_TRAIN,
            layer_idx=0,
            quant=self.quant,
            is_full_attention_layer=False,
            rngs=nnx.Rngs(params=0),
        )
        return nnx.split(m, nnx.Param, nnx.Intermediate, ...)

      self.gdn_graphdef, abs_gdn_params, gdn_inter, gdn_other = nnx.eval_shape(
          _init_gdn
      )
      self.gdn_inter = _materialize_template(gdn_inter)
      self.gdn_other = _materialize_template(gdn_other)
      self.abs_gdn_params = _attach_named_shardings(
          abs_gdn_params, mesh, config.logical_axis_rules
      )

      def _init_attn():
        m = qwen3_5.Qwen3_5DecoderLayer(
            config=config,
            mesh=mesh,
            model_mode=common_types.MODEL_MODE_TRAIN,
            layer_idx=self.cycle - 1,
            quant=self.quant,
            is_full_attention_layer=True,
            rngs=nnx.Rngs(params=0),
        )
        return nnx.split(m, nnx.Param, nnx.Intermediate, ...)

      self.attn_graphdef, abs_attn_params, attn_inter, attn_other = (
          nnx.eval_shape(_init_attn)
      )
      self.attn_inter = _materialize_template(attn_inter)
      self.attn_other = _materialize_template(attn_other)
      self.abs_attn_params = _attach_named_shardings(
          abs_attn_params, mesh, config.logical_axis_rules
      )

      def _init_block():
        m = _RecordingQwen35ScannableBlock(
            config=config,
            mesh=mesh,
            model_mode=common_types.MODEL_MODE_TRAIN,
            quant=self.quant,
            num_of_layers=self.cycle,
            layer_idx_offset=0,
            apply_internal_remat=False,
            rngs=nnx.Rngs(params=0),
        )
        return nnx.split(m, nnx.Param, nnx.Intermediate, ...)

      self.block_graphdef, abs_block_params, block_inter, block_other = (
          nnx.eval_shape(_init_block)
      )
      self.block_inter = _materialize_template(block_inter)
      self.block_other = _materialize_template(block_other)
      self.abs_block_params = _attach_named_shardings(
          abs_block_params, mesh, config.logical_axis_rules
      )

    self._compile_callables()

  def _make_layer_callable(
      self, graphdef: Any, inter_template: Any, other_template: Any
  ):
    """Builds a jitted single-layer forward callable."""

    def _layer_fn(
        params, hidden_states, decoder_segment_ids, decoder_positions
    ):
      layer = nnx.merge(
          graphdef,
          _clone_at_current_trace_level(params),
          _clone_at_current_trace_level(inter_template),
          _clone_at_current_trace_level(other_template),
      )
      out = layer(
          hidden_states,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_TRAIN,
      )
      y = out[0] if isinstance(out, tuple) else out
      lo_f32 = y.astype(jnp.float32)
      tok_ms = jnp.mean(jnp.square(lo_f32), axis=-1)
      seg_all = (decoder_segment_ids != 0).astype(jnp.float32)
      seg_d1 = (decoder_segment_ids == 1).astype(jnp.float32)
      seg_d2p = (decoder_segment_ids >= 2).astype(jnp.float32)
      resid_stats = jnp.stack([
          jnp.sum(tok_ms * seg_all),
          jnp.sum(seg_all),
          jnp.sum(tok_ms * seg_d1),
          jnp.sum(seg_d1),
          jnp.sum(tok_ms * seg_d2p),
          jnp.sum(seg_d2p),
      ])
      return y, resid_stats

    return jax.jit(_layer_fn)

  def _make_block_callable(self):
    """Builds a jitted 4-layer `_RecordingQwen35ScannableBlock` forward callable."""
    graphdef = self.block_graphdef
    inter_template = self.block_inter
    other_template = self.block_other

    def _block_fn(
        params, hidden_states, decoder_segment_ids, decoder_positions
    ):
      block = nnx.merge(
          graphdef,
          _clone_at_current_trace_level(params),
          _clone_at_current_trace_level(inter_template),
          _clone_at_current_trace_level(other_template),
      )
      out = block(
          hidden_states,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_TRAIN,
      )
      y = out[0] if isinstance(out, tuple) else out
      _, _, new_inter, _ = nnx.split(block, nnx.Param, nnx.Intermediate, ...)
      inter_dict = nnx.to_pure_dict(new_inter)
      loc_inter = inter_dict.get("local_layers", {})
      glb_inter = inter_dict.get("global_layer", {})
      loc_raw = loc_inter.get(
          "resid_ms_sum", jnp.zeros((self.cycle - 1,), jnp.float32)
      )
      glb_raw = glb_inter.get("resid_ms_sum", jnp.float32(0.0))
      loc_ms = jnp.asarray(
          loc_raw[-1] if isinstance(loc_raw, (list, tuple)) else loc_raw,
          jnp.float32,
      )
      glb_ms = jnp.asarray(
          glb_raw[-1] if isinstance(glb_raw, (list, tuple)) else glb_raw,
          jnp.float32,
      )
      all_ms = jnp.concatenate([loc_ms.reshape(-1), glb_ms.reshape(1)], axis=0)
      return y, all_ms

    return jax.jit(_block_fn)

  def _compile_callables(self):
    """Creates jitted callables for embed, layers, blocks, and final head."""
    shell_graphdef = self.shell_graphdef
    shell_other = self.shell_other
    cfg = self.config

    @jax.jit
    def _embed_fn(shell_params, inputs, inputs_position, inputs_segmentation):
      del inputs_position
      shell = nnx.merge(
          shell_graphdef,
          _clone_at_current_trace_level(shell_params),
          _clone_at_current_trace_level(shell_other),
      )
      y = shell.token_embedder(inputs.astype(jnp.int32))
      y = y.astype(cfg.dtype)
      seg_all = (inputs_segmentation != 0).astype(jnp.float32)
      tok_ms = jnp.mean(jnp.square(y.astype(jnp.float32)), axis=-1)
      embed_ms_sum = jnp.sum(tok_ms * seg_all)
      return y, embed_ms_sum, jnp.sum(seg_all)

    @jax.jit
    def _head_fn(
        shell_params,
        hidden_states,
        targets,
        targets_segmentation,
        targets_position,
    ):
      shell = nnx.merge(
          shell_graphdef,
          _clone_at_current_trace_level(shell_params),
          _clone_at_current_trace_level(shell_other),
      )
      y_norm = shell.decoder.decoder_norm(hidden_states)
      fn_f32 = y_norm.astype(jnp.float32)
      valid_f32 = (targets_segmentation != 0).astype(jnp.float32)
      final_norm_ms_sum = jnp.sum(
          jnp.mean(jnp.square(fn_f32), axis=-1) * valid_f32
      )
      logits = shell.decoder.logits_dense(y_norm)
      if cfg.cast_logits_to_fp32:
        logits = logits.astype(jnp.float32)
      logits_f32 = logits.astype(jnp.float32)
      one_hot = jax.nn.one_hot(targets, cfg.vocab_size, dtype=logits_f32.dtype)
      xent, z_loss = max_utils.cross_entropy_with_logits(
          logits_f32, one_hot, z_loss=cfg.z_loss_multiplier
      )
      xent_f32 = xent.astype(jnp.float32) * valid_f32
      z_f32 = z_loss.astype(jnp.float32) * valid_f32

      doc1_m = (targets_segmentation == 1).astype(jnp.float32)
      doc2p_m = (targets_segmentation >= 2).astype(jnp.float32)
      d2_h = ((targets_segmentation >= 2) & (targets_position < 4)).astype(
          jnp.float32
      )
      d2_b = ((targets_segmentation >= 2) & (targets_position >= 4)).astype(
          jnp.float32
      )

      logits_mean = jnp.mean(logits_f32, axis=-1, keepdims=True)
      logits_ms_sum = jnp.sum(
          jnp.mean(jnp.square(logits_f32), axis=-1) * valid_f32
      )
      logits_var_sum = jnp.sum(
          jnp.mean(jnp.square(logits_f32 - logits_mean), axis=-1) * valid_f32
      )

      stats = {
          "xent_sum": jnp.sum(xent_f32),
          "z_loss_sum": jnp.sum(z_f32),
          "valid_tok": jnp.sum(valid_f32),
          "doc1_loss_sum": jnp.sum(xent.astype(jnp.float32) * doc1_m),
          "doc1_tok": jnp.sum(doc1_m),
          "doc2p_loss_sum": jnp.sum(xent.astype(jnp.float32) * doc2p_m),
          "doc2p_tok": jnp.sum(doc2p_m),
          "doc2p_pos0to3_loss_sum": jnp.sum(xent.astype(jnp.float32) * d2_h),
          "doc2p_pos0to3_tok": jnp.sum(d2_h),
          "doc2p_pos4p_loss_sum": jnp.sum(xent.astype(jnp.float32) * d2_b),
          "doc2p_pos4p_tok": jnp.sum(d2_b),
          "final_norm_ms_sum": final_norm_ms_sum,
          "logits_ms_sum": logits_ms_sum,
          "logits_var_sum": logits_var_sum,
      }
      seq_idx = jnp.arange(targets.shape[1], dtype=jnp.int32)[None, :]
      for b_idx in range(8):
        lo = b_idx * 512
        hi = (b_idx + 1) * 512
        bm = (
            (targets_segmentation != 0) & (seq_idx >= lo) & (seq_idx < hi)
        ).astype(jnp.float32)
        stats[f"pos_loss_sum_b{b_idx}"] = jnp.sum(xent.astype(jnp.float32) * bm)
        stats[f"pos_tok_b{b_idx}"] = jnp.sum(bm)
      for t_lbl, t_val in (
          ("T0p5", 0.5),
          ("T1p0", 1.0),
          ("T1p5", 1.5),
          ("T2p0", 2.0),
          ("T3p0", 3.0),
          ("T4p0", 4.0),
      ):
        t_ce, _ = max_utils.cross_entropy_with_logits(
            logits_f32 / jnp.float32(t_val), one_hot, z_loss=0.0
        )
        stats[f"temp_loss_sum_{t_lbl}"] = jnp.sum(
            t_ce.astype(jnp.float32) * valid_f32
        )
      return logits_f32, xent_f32, stats

    @jax.jit
    def _diff_fn(y_a, y_b, seg_ids):
      a_f32 = y_a.astype(jnp.float32)
      b_f32 = y_b.astype(jnp.float32)
      diff = a_f32 - b_f32
      out = []
      for m in ((seg_ids != 0), (seg_ids == 1), (seg_ids >= 2)):
        m_f = m.astype(jnp.float32)[..., None]
        diff_sq = jnp.sum(jnp.square(diff) * m_f)
        ref_sq = jnp.sum(jnp.square(b_f32) * m_f)
        cur_sq = jnp.sum(jnp.square(a_f32) * m_f)
        dot_val = jnp.sum((a_f32 * b_f32) * m_f)
        max_abs = jnp.max(jnp.where(m[..., None], jnp.abs(diff), 0.0))
        out.extend([diff_sq, ref_sq, cur_sq, dot_val, max_abs])
      return jnp.stack(out)

    self.embed_fn = _embed_fn
    self.head_fn = _head_fn
    self.diff_fn = _diff_fn
    self.gdn_fwd = self._make_layer_callable(
        self.gdn_graphdef, self.gdn_inter, self.gdn_other
    )
    self.attn_fwd = self._make_layer_callable(
        self.attn_graphdef, self.attn_inter, self.attn_other
    )
    self.block_fwd = self._make_block_callable()

  @staticmethod
  def _summarize_diff_stats(diff_15_list: list[np.ndarray]) -> dict[str, float]:
    """Aggregates `[diff_sq, ref_sq, cur_sq, dot, max_abs]` across microbatches for `all`/`doc1`/`doc2plus`."""
    arr = np.stack(
        [np.asarray(x, dtype=np.float64) for x in diff_15_list], axis=0
    )
    res = {}
    for idx, prefix in enumerate(("all", "doc1", "doc2plus")):
      base = idx * 5
      diff_sq = float(np.sum(arr[:, base + 0]))
      ref_sq = float(np.sum(arr[:, base + 1]))
      cur_sq = float(np.sum(arr[:, base + 2]))
      dot_val = float(np.sum(arr[:, base + 3]))
      max_abs = float(np.max(arr[:, base + 4]))
      res[f"{prefix}_rel_l2"] = (
          float(np.sqrt(diff_sq / max(ref_sq, 1e-30))) if ref_sq > 0 else 0.0
      )
      denom_cos = float(np.sqrt(max(ref_sq * cur_sq, 1e-30)))
      res[f"{prefix}_cosine"] = (
          float(dot_val / denom_cos) if (ref_sq > 0 and cur_sq > 0) else 1.0
      )
      res[f"{prefix}_max_abs"] = max_abs
    return res

  @staticmethod
  def _summarize_head_stats(
      stats_list: list[dict[str, Any]],
  ) -> dict[str, float]:
    """Aggregates head loss and RMS stats across microbatches."""
    sums: dict[str, float] = collections.defaultdict(float)
    for st in stats_list:
      for k, v in st.items():
        sums[k] += float(np.asarray(v))
    valid_tok = max(sums["valid_tok"], 1.0)
    d1_tok = max(sums["doc1_tok"], 1.0)
    d2_tok = max(sums["doc2p_tok"], 1.0)
    d2h_tok = max(sums["doc2p_pos0to3_tok"], 1.0)
    d2b_tok = max(sums["doc2p_pos4p_tok"], 1.0)
    out = {
        "eval_loss": sums["xent_sum"] / valid_tok,
        "total_loss": sums["xent_sum"],
        "z_loss": sums["z_loss_sum"] / valid_tok,
        "valid_tokens": sums["valid_tok"],
        "doc1_loss": (
            sums["doc1_loss_sum"] / d1_tok if sums["doc1_tok"] > 0 else 0.0
        ),
        "doc1_tokens": sums["doc1_tok"],
        "doc2plus_loss": (
            sums["doc2p_loss_sum"] / d2_tok if sums["doc2p_tok"] > 0 else 0.0
        ),
        "doc2plus_tokens": sums["doc2p_tok"],
        "doc2plus_pos0to3_loss": (
            sums["doc2p_pos0to3_loss_sum"] / d2h_tok
            if sums["doc2p_pos0to3_tok"] > 0
            else 0.0
        ),
        "doc2plus_pos0to3_tokens": sums["doc2p_pos0to3_tok"],
        "doc2plus_pos4plus_loss": (
            sums["doc2p_pos4p_loss_sum"] / d2b_tok
            if sums["doc2p_pos4p_tok"] > 0
            else 0.0
        ),
        "doc2plus_pos4plus_tokens": sums["doc2p_pos4p_tok"],
        "final_norm_rms": float(np.sqrt(sums["final_norm_ms_sum"] / valid_tok)),
        "logits_rms": float(np.sqrt(sums["logits_ms_sum"] / valid_tok)),
        "logits_std": float(np.sqrt(sums["logits_var_sum"] / valid_tok)),
    }
    for b_idx in range(8):
      btok = sums.get(f"pos_tok_b{b_idx}", 0.0)
      out[f"pos_loss_b{b_idx}"] = (
          sums.get(f"pos_loss_sum_b{b_idx}", 0.0) / max(btok, 1.0)
          if btok > 0
          else 0.0
      )
      out[f"pos_tokens_b{b_idx}"] = btok
    for t_lbl in ("T0p5", "T1p0", "T1p5", "T2p0", "T3p0", "T4p0"):
      out[f"temp_loss_{t_lbl}"] = sums[f"temp_loss_sum_{t_lbl}"] / valid_tok
    return out

  @staticmethod
  def _shards_to_u16_bytes(arr: jax.Array) -> tuple[bytes, list[int]]:
    """Serializes local addressable shards of a bfloat16/int32 array in `addressable_shards` order."""
    parts = []
    starts = []
    for s in arr.addressable_shards:
      b_start = int(s.index[0].start or 0) if s.index else 0
      starts.append(b_start)
      a_np = np.asarray(s.data)
      if a_np.dtype == jnp.bfloat16 or str(a_np.dtype) == "bfloat16":
        parts.append(a_np.view(np.uint16).tobytes())
      else:
        parts.append(a_np.tobytes())
    return b"".join(parts), starts

  @staticmethod
  def _u16_bytes_to_sharded_bf16(
      raw_bytes: bytes, template_arr: jax.Array
  ) -> jax.Array:
    """Reconstructs a sharded bfloat16 `jax.Array` on the same mesh/devices from `_shards_to_u16_bytes`."""
    shards = template_arr.addressable_shards
    shard_shape = shards[0].data.shape
    shard_nbytes = int(np.prod(shard_shape)) * 2
    dev_bufs = []
    for i, s in enumerate(shards):
      chunk = raw_bytes[i * shard_nbytes : (i + 1) * shard_nbytes]
      arr_np = (
          np.frombuffer(chunk, dtype=np.uint16)
          .reshape(shard_shape)
          .view(jnp.bfloat16)
      )
      dev_bufs.append(jax.device_put(arr_np, s.device))
    return jax.make_array_from_single_device_arrays(
        template_arr.shape, template_arr.sharding, dev_bufs
    )

  def _save_resume_checkpoint(
      self,
      save_mb0_act_dir: str,
      proc_idx: int,
      completed_layers: int,
      h_packed: Sequence[jax.Array],
      h_clean: Sequence[jax.Array],
      meta_payload: dict[str, Any],
  ) -> None:
    """Saves a multi-host preemption-resume checkpoint of hidden states and metrics."""
    if not save_mb0_act_dir:
      return
    ckpt_dir = os.path.join(save_mb0_act_dir, f"resume_L{completed_layers:02d}")
    tf.io.gfile.makedirs(ckpt_dir)
    chunk_size = 5

    p_bytes = [self._shards_to_u16_bytes(arr)[0] for arr in h_packed]
    c_bytes = [self._shards_to_u16_bytes(arr)[0] for arr in h_clean]

    def _write_chunk(prefix: str, start_i: int, items: list[bytes]) -> None:
      path = os.path.join(
          ckpt_dir, f"{prefix}_host{proc_idx}_b{start_i:03d}.bin"
      )
      with tf.io.gfile.GFile(path, "wb") as f:
        f.write(b"".join(items))

    tasks = []
    for i in range(0, len(p_bytes), chunk_size):
      tasks.append(("packed", i, p_bytes[i : i + chunk_size]))
    for i in range(0, len(c_bytes), chunk_size):
      tasks.append(("clean", i, c_bytes[i : i + chunk_size]))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(16, max(1, len(tasks)))
    ) as pool:
      futs = [pool.submit(_write_chunk, pfx, si, it) for pfx, si, it in tasks]
      for fut in futs:
        fut.result()

    meta_path = os.path.join(ckpt_dir, f"meta_host{proc_idx}.json")
    with tf.io.gfile.GFile(meta_path, "w") as fm:
      json.dump(meta_payload, fm, indent=2, sort_keys=True)

  def _try_load_resume_checkpoint(
      self,
      save_mb0_act_dir: str,
      proc_idx: int,
      proc_count: int,
      num_layers: int,
      template_arr: jax.Array,
      num_packed: int,
      num_clean: int,
      rec_nbytes: int,
  ) -> tuple[int, list[jax.Array], list[jax.Array], dict[str, Any]] | None:
    """Finds and loads the latest complete multi-host resume checkpoint."""
    if not save_mb0_act_dir or not tf.io.gfile.exists(save_mb0_act_dir):
      return None
    chunk_size = 5
    for cand_layer in range(num_layers, 0, -1):
      ckpt_dir = os.path.join(save_mb0_act_dir, f"resume_L{cand_layer:02d}")
      if not tf.io.gfile.exists(ckpt_dir):
        continue
      all_hosts_ready = True
      for p in range(proc_count):
        m_path = os.path.join(ckpt_dir, f"meta_host{p}.json")
        p_act = os.path.join(save_mb0_act_dir, f"packed_mb0_host{p}.bin")
        if not tf.io.gfile.exists(m_path) or not tf.io.gfile.exists(p_act):
          all_hosts_ready = False
          break
        try:
          if tf.io.gfile.stat(p_act).length < (cand_layer + 1) * rec_nbytes:
            all_hosts_ready = False
            break
        except Exception:  # pylint: disable=broad-exception-caught
          all_hosts_ready = False
          break
      if not all_hosts_ready:
        continue

      try:
        m_path = os.path.join(ckpt_dir, f"meta_host{proc_idx}.json")
        with tf.io.gfile.GFile(m_path, "r") as fm:
          meta = json.load(fm)
        if (
            meta.get("num_layers") != num_layers
            or meta.get("num_microbatches") != num_packed
        ):
          continue

        def _read_chunk(
            prefix: str, start_i: int, count: int, cur_ckpt_dir: str = ckpt_dir
        ) -> list[bytes]:
          path = os.path.join(
              cur_ckpt_dir, f"{prefix}_host{proc_idx}_b{start_i:03d}.bin"
          )
          with tf.io.gfile.GFile(path, "rb") as f:
            buf = f.read()
          return [
              buf[k * rec_nbytes : (k + 1) * rec_nbytes] for k in range(count)
          ]

        p_ranges = [
            (i, min(chunk_size, num_packed - i))
            for i in range(0, num_packed, chunk_size)
        ]
        c_ranges = [
            (i, min(chunk_size, num_clean - i))
            for i in range(0, num_clean, chunk_size)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
          p_futs = [
              pool.submit(_read_chunk, "packed", i, cnt) for i, cnt in p_ranges
          ]
          c_futs = [
              pool.submit(_read_chunk, "clean", i, cnt) for i, cnt in c_ranges
          ]
          p_raw_list: list[bytes] = []
          for fut in p_futs:
            p_raw_list.extend(fut.result())
          c_raw_list: list[bytes] = []
          for fut in c_futs:
            c_raw_list.extend(fut.result())

        h_packed = [
            self._u16_bytes_to_sharded_bf16(b, template_arr) for b in p_raw_list
        ]
        h_clean = [
            self._u16_bytes_to_sharded_bf16(b, template_arr) for b in c_raw_list
        ]
        return cand_layer, h_packed, h_clean, meta
      except Exception as e:  # pylint: disable=broad-exception-caught
        max_logging.log(
            "[Layerwise] Warning: failed loading checkpoint"
            f" L{cand_layer:02d}: {e}"
        )
        continue
    return None

  def _put_batches_on_mesh(
      self, batches: Sequence[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    """Places input batches onto the target mesh data sharding."""
    data_sharding = sharding.get_input_data_sharding(self.config, self.mesh)

    def _put_leaf(x):
      if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return x
      return self.reader.host_to_sharded_array(np.asarray(x), data_sharding)

    return [jax.tree.map(_put_leaf, b) for b in batches]

  def _embed_batches(
      self,
      shell_params: Any,
      batches: Sequence[dict[str, Any]],
      first_embed: tuple[jax.Array, jax.Array, jax.Array] | None = None,
  ) -> tuple[list[jax.Array], float, float]:
    """Runs `embed_fn` across `batches` and returns `(h_list, ms_sum, tok_sum)`."""
    if not batches:
      return [], 0.0, 0.0
    h_list: list[jax.Array] = []
    ms_list: list[jax.Array] = []
    tok_list: list[jax.Array] = []
    start_idx = 0
    if first_embed is not None:
      y0, ms0, tok0 = first_embed
      h_list.append(y0)
      ms_list.append(ms0)
      tok_list.append(tok0)
      start_idx = 1
    for b in batches[start_idx:]:
      y0, ms_s, tok_s = self.embed_fn(
          shell_params,
          b["inputs"],
          b["inputs_position"],
          b["inputs_segmentation"],
      )
      h_list.append(y0)
      ms_list.append(ms_s)
      tok_list.append(tok_s)
    ms_host, tok_host = jax.device_get((ms_list, tok_list))
    ms_sum = float(np.sum([np.asarray(x, dtype=np.float64) for x in ms_host]))
    tok_sum = float(np.sum([np.asarray(x, dtype=np.float64) for x in tok_host]))
    return h_list, ms_sum, tok_sum

  def _open_mb0_act_writer(
      self,
      act_path: str,
      start_layer: int,
      rec_nbytes: int,
      h0: jax.Array,
  ) -> Any:
    """Opens a microbatch-0 activation file for writing or resumption."""
    if start_layer > 0 and tf.io.gfile.exists(act_path):
      with tf.io.gfile.GFile(act_path, "rb") as fr:
        prefix = fr.read((start_layer + 1) * rec_nbytes)
      fh = tf.io.gfile.GFile(act_path, "wb")
      fh.write(prefix)
      return fh
    fh = tf.io.gfile.GFile(act_path, "wb")
    raw_0, _ = self._shards_to_u16_bytes(h0)
    fh.write(raw_0)
    return fh

  def _run_layer_on_split(
      self,
      fwd_fn: Any,
      layer_params: Any,
      batches: Sequence[dict[str, Any]],
      h_list: list[jax.Array],
      layer_idx: int,
      is_full_attn: bool,
      include_doc_split: bool,
      load_sec: float | None,
      f_act: Any,
      f_ref: Any,
      h_ref_prev: jax.Array | None,
      rec_nbytes: int,
      include_teacher_forced: bool,
  ) -> tuple[dict[str, Any], jax.Array | None]:
    """Runs one layer across `batches`, updates `h_list` in place, and computes diffs."""
    r_list = []
    for mb_idx, b in enumerate(batches):
      seg = b["inputs_segmentation"]
      pos = b["inputs_position"]
      y_out, r_out = fwd_fn(layer_params, h_list[mb_idx], seg, pos)
      r_list.append(r_out)
      h_list[mb_idx] = y_out

    r_host = jax.device_get(r_list)
    resid = np.sum([np.asarray(x, dtype=np.float64) for x in r_host], axis=0)
    if f_act is not None:
      raw_act, _ = self._shards_to_u16_bytes(h_list[0])
      f_act.write(raw_act)

    rec: dict[str, Any] = {
        "layer_idx": layer_idx,
        "layer_type": "full_attention" if is_full_attn else "gdn",
        "resid_rms": float(np.sqrt(resid[0] / max(resid[1], 1.0))),
    }
    if load_sec is not None:
      rec["load_seconds"] = load_sec
    if include_doc_split:
      rec["doc1_resid_rms"] = float(np.sqrt(resid[2] / max(resid[3], 1.0)))
      rec["doc2plus_resid_rms"] = (
          float(np.sqrt(resid[4] / max(resid[5], 1.0))) if resid[5] > 0 else 0.0
      )

    if f_ref is not None and h_ref_prev is not None:
      h_ref_curr = self._u16_bytes_to_sharded_bf16(
          f_ref.read(rec_nbytes), h_list[0]
      )
      seg0 = batches[0]["inputs_segmentation"]
      pos0 = batches[0]["inputs_position"]
      ch_d = self.diff_fn(h_list[0], h_ref_curr, seg0)
      rec["chained_diff"] = self._summarize_diff_stats([np.asarray(ch_d)])
      if include_teacher_forced:
        y_tf, _ = fwd_fn(layer_params, h_ref_prev, seg0, pos0)
        tf_d = self.diff_fn(y_tf, h_ref_curr, seg0)
        rec["teacher_forced_diff"] = self._summarize_diff_stats(
            [np.asarray(tf_d)]
        )
      h_ref_prev = h_ref_curr
    return rec, h_ref_prev

  def _eval_head_on_split(
      self,
      shell_params: Any,
      batches: Sequence[dict[str, Any]],
      h_list: Sequence[jax.Array],
      h_ref_prev: jax.Array | None,
  ) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluates `head_fn` across `batches` and computes optional microbatch-0 logits diff."""
    head_list = []
    logits_diff: dict[str, float] = {}
    for mb_idx, b in enumerate(batches):
      seg = b["targets_segmentation"]
      pos = b["targets_position"]
      tgt = b["targets"]
      lg, _, st = self.head_fn(shell_params, h_list[mb_idx], tgt, seg, pos)
      head_list.append(st)
      if mb_idx == 0 and h_ref_prev is not None:
        lg_ref, _, _ = self.head_fn(shell_params, h_ref_prev, tgt, seg, pos)
        ld = self.diff_fn(lg, lg_ref, seg)
        logits_diff = self._summarize_diff_stats([np.asarray(ld)])
    return self._summarize_head_stats(head_list), logits_diff

  def run_layerwise_verification(
      self,
      batches: Sequence[dict[str, Any]],
      run_clean_single_doc: bool = True,
      include_teacher_forced: bool = True,
      max_layers: int | None = None,
      save_mb0_act_dir: str = "",
      ref_mb0_act_dir: str = "",
      progress_json_path: str = "",
      checkpoint_interval: int = 5,
  ) -> dict[str, Any]:
    """Streams all layers one at a time and chains boundary activations across microbatches."""
    t_start = time.perf_counter()
    num_layers = max_layers or self.total_layers
    packed_batches = self._put_batches_on_mesh(batches)
    with jax.set_mesh(self.mesh), axis_rules(self.config.logical_axis_rules):
      jit_clean_fn = jax.jit(make_clean_single_doc_batch)
      clean_batches = (
          [jit_clean_fn(b) for b in packed_batches]
          if run_clean_single_doc
          else []
      )

    proc_idx = jax.process_index()
    proc_count = jax.process_count()
    n_local_shards = len(
        packed_batches[0]["inputs_segmentation"].addressable_shards
    )
    shard_b = int(
        packed_batches[0]["inputs_segmentation"]
        .addressable_shards[0]
        .data.shape[0]
    )
    seq_l = int(packed_batches[0]["inputs_segmentation"].shape[1])
    emb_dim = int(self.config.emb_dim)
    rec_nbytes = n_local_shards * shard_b * seq_l * emb_dim * 2

    t0 = time.perf_counter()
    shell_params = self.reader.load_non_layer_params(
        self.abs_shell_params, self.config, total_layers=self.total_layers
    )
    non_layer_load_sec = time.perf_counter() - t0
    max_logging.log(
        f"[Layerwise] Loaded non-layer params in {non_layer_load_sec:.2f}s"
    )

    with jax.set_mesh(self.mesh), axis_rules(self.config.logical_axis_rules):
      y0_template, ms0, tok0 = self.embed_fn(
          shell_params,
          packed_batches[0]["inputs"],
          packed_batches[0]["inputs_position"],
          packed_batches[0]["inputs_segmentation"],
      )
      resumed = None
      if checkpoint_interval > 0:
        resumed = self._try_load_resume_checkpoint(
            save_mb0_act_dir=save_mb0_act_dir,
            proc_idx=proc_idx,
            proc_count=proc_count,
            num_layers=num_layers,
            template_arr=y0_template,
            num_packed=len(packed_batches),
            num_clean=len(clean_batches),
            rec_nbytes=rec_nbytes,
        )

      if resumed is not None:
        start_layer, h_packed, h_clean, resume_meta = resumed
        embed_ms_packed = float(resume_meta["embed_ms_packed"])
        embed_tok_packed = float(resume_meta["embed_tok_packed"])
        embed_ms_clean = float(resume_meta.get("embed_ms_clean", 0.0))
        embed_tok_clean = float(resume_meta.get("embed_tok_clean", 0.0))
        per_layer_packed = list(resume_meta["packed"]["layers"])
        per_layer_clean = list(
            resume_meta.get("clean_single_doc", {}).get("layers", [])
        )
        layer_load_times = list(resume_meta.get("layer_load_times", []))
        layer_exec_times = list(resume_meta.get("layer_exec_times", []))
        max_logging.log(
            "[Layerwise] Resumed from preemption checkpoint at layer"
            f" {start_layer:02d}/{num_layers:02d}"
        )
      else:
        start_layer = 0
        h_packed, embed_ms_packed, embed_tok_packed = self._embed_batches(
            shell_params,
            packed_batches,
            first_embed=(y0_template, ms0, tok0),
        )
        h_clean, embed_ms_clean, embed_tok_clean = self._embed_batches(
            shell_params, clean_batches
        )
        per_layer_packed = []
        per_layer_clean = []
        layer_load_times = []
        layer_exec_times = []

    f_packed_act = None
    f_clean_act = None
    if save_mb0_act_dir:
      tf.io.gfile.makedirs(save_mb0_act_dir)
      _, p_starts = self._shards_to_u16_bytes(
          packed_batches[0]["inputs_segmentation"]
      )
      meta = {
          "process_index": proc_idx,
          "process_count": proc_count,
          "num_layers": num_layers,
          "num_records": num_layers + 1,
          "batch_shard_starts": p_starts,
          "local_mb0_shape": [n_local_shards * shard_b, seq_l, emb_dim],
      }
      with tf.io.gfile.GFile(
          os.path.join(save_mb0_act_dir, f"mb0_meta_host{proc_idx}.json"), "w"
      ) as fm:
        json.dump(meta, fm, indent=2)
      f_packed_act = self._open_mb0_act_writer(
          os.path.join(save_mb0_act_dir, f"packed_mb0_host{proc_idx}.bin"),
          start_layer,
          rec_nbytes,
          h_packed[0],
      )
      if clean_batches:
        f_clean_act = self._open_mb0_act_writer(
            os.path.join(save_mb0_act_dir, f"clean_mb0_host{proc_idx}.bin"),
            start_layer,
            rec_nbytes,
            h_clean[0],
        )

    f_ref_packed = None
    f_ref_clean = None
    if ref_mb0_act_dir:
      ref_p_path = os.path.join(
          ref_mb0_act_dir, f"packed_mb0_host{proc_idx}.bin"
      )
      if tf.io.gfile.exists(ref_p_path):
        f_ref_packed = tf.io.gfile.GFile(ref_p_path, "rb")
        if start_layer > 0:
          f_ref_packed.seek(start_layer * rec_nbytes)
      if clean_batches:
        ref_c_path = os.path.join(
            ref_mb0_act_dir, f"clean_mb0_host{proc_idx}.bin"
        )
        if tf.io.gfile.exists(ref_c_path):
          f_ref_clean = tf.io.gfile.GFile(ref_c_path, "rb")
          if start_layer > 0:
            f_ref_clean.seek(start_layer * rec_nbytes)

    with jax.set_mesh(self.mesh), axis_rules(self.config.logical_axis_rules):
      h_ref_packed_prev = None
      if f_ref_packed is not None:
        h_ref_packed_prev = self._u16_bytes_to_sharded_bf16(
            f_ref_packed.read(rec_nbytes), h_packed[0]
        )
      h_ref_clean_prev = None
      if f_ref_clean is not None and h_clean:
        h_ref_clean_prev = self._u16_bytes_to_sharded_bf16(
            f_ref_clean.read(rec_nbytes), h_clean[0]
        )

    def _load_np_for_layer(
        l_idx: int,
    ) -> tuple[dict[tuple[str, ...], np.ndarray], float]:
      t_l0 = time.perf_counter()
      l_full = (l_idx + 1) % self.cycle == 0
      l_abs = self.abs_attn_params if l_full else self.abs_gdn_params
      f_np = self.reader.load_single_layer_numpy(
          l_idx, l_abs, self.config, total_layers=self.total_layers
      )
      return f_np, (time.perf_counter() - t_l0)

    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    prefetch_fut = (
        prefetch_pool.submit(_load_np_for_layer, start_layer)
        if start_layer < num_layers
        else None
    )

    try:
      for layer_idx in range(start_layer, num_layers):
        is_full_attn = (layer_idx + 1) % self.cycle == 0
        abs_params = (
            self.abs_attn_params if is_full_attn else self.abs_gdn_params
        )
        fwd_fn = self.attn_fwd if is_full_attn else self.gdn_fwd

        t_wait0 = time.perf_counter()
        assert prefetch_fut is not None
        flat_np, bg_load_sec = prefetch_fut.result()
        layer_params = self.reader.load_single_layer_params_from_numpy(
            flat_np, abs_params
        )
        del flat_np
        load_sec = max(time.perf_counter() - t_wait0, bg_load_sec)
        layer_load_times.append(load_sec)

        if layer_idx + 1 < num_layers:
          prefetch_fut = prefetch_pool.submit(_load_np_for_layer, layer_idx + 1)
        else:
          prefetch_fut = None

        t_exec0 = time.perf_counter()
        with jax.set_mesh(self.mesh), axis_rules(
            self.config.logical_axis_rules
        ):
          p_layer_rec, h_ref_packed_prev = self._run_layer_on_split(
              fwd_fn=fwd_fn,
              layer_params=layer_params,
              batches=packed_batches,
              h_list=h_packed,
              layer_idx=layer_idx,
              is_full_attn=is_full_attn,
              include_doc_split=True,
              load_sec=load_sec,
              f_act=f_packed_act,
              f_ref=f_ref_packed,
              h_ref_prev=h_ref_packed_prev,
              rec_nbytes=rec_nbytes,
              include_teacher_forced=include_teacher_forced,
          )
          per_layer_packed.append(p_layer_rec)

          if clean_batches:
            c_layer_rec, h_ref_clean_prev = self._run_layer_on_split(
                fwd_fn=fwd_fn,
                layer_params=layer_params,
                batches=clean_batches,
                h_list=h_clean,
                layer_idx=layer_idx,
                is_full_attn=is_full_attn,
                include_doc_split=False,
                load_sec=None,
                f_act=f_clean_act,
                f_ref=f_ref_clean,
                h_ref_prev=h_ref_clean_prev,
                rec_nbytes=rec_nbytes,
                include_teacher_forced=include_teacher_forced,
            )
            per_layer_clean.append(c_layer_rec)

        exec_sec = time.perf_counter() - t_exec0
        layer_exec_times.append(exec_sec)
        del layer_params
        if f_packed_act is not None:
          f_packed_act.flush()
        if f_clean_act is not None:
          f_clean_act.flush()

        clean_partial: dict[str, Any] = {}
        if clean_batches:
          clean_partial = {
              "embed_rms": float(
                  np.sqrt(embed_ms_clean / max(embed_tok_clean, 1.0))
              ),
              "layers": per_layer_clean,
          }
        partial_payload = {
            "model_name": self.config.model_name,
            "num_layers": num_layers,
            "layers_completed": layer_idx + 1,
            "num_microbatches": len(packed_batches),
            "elapsed_wall_seconds": time.perf_counter() - t_start,
            "non_layer_load_seconds": non_layer_load_sec,
            "total_layer_load_seconds": float(np.sum(layer_load_times)),
            "total_layer_exec_seconds": float(np.sum(layer_exec_times)),
            "layer_load_times": layer_load_times,
            "layer_exec_times": layer_exec_times,
            "embed_ms_packed": embed_ms_packed,
            "embed_tok_packed": embed_tok_packed,
            "embed_ms_clean": embed_ms_clean,
            "embed_tok_clean": embed_tok_clean,
            "packed": {
                "embed_rms": float(
                    np.sqrt(embed_ms_packed / max(embed_tok_packed, 1.0))
                ),
                "layers": per_layer_packed,
            },
            "clean_single_doc": clean_partial,
        }
        if progress_json_path and proc_idx == 0:
          try:
            progress_dir = os.path.dirname(progress_json_path)
            if progress_dir:
              tf.io.gfile.makedirs(progress_dir)
            with tf.io.gfile.GFile(progress_json_path, "w") as fp:
              json.dump(partial_payload, fp, indent=2, sort_keys=True)
          except Exception as e:  # pylint: disable=broad-exception-caught
            max_logging.log(
                f"[Layerwise] Warning: could not write progress JSON: {e}"
            )

        if (
            save_mb0_act_dir
            and checkpoint_interval > 0
            and (
                (layer_idx + 1) % checkpoint_interval == 0
                or (layer_idx + 1) == num_layers
            )
        ):
          try:
            t_ckpt0 = time.perf_counter()
            self._save_resume_checkpoint(
                save_mb0_act_dir=save_mb0_act_dir,
                proc_idx=proc_idx,
                completed_layers=layer_idx + 1,
                h_packed=h_packed,
                h_clean=h_clean,
                meta_payload=partial_payload,
            )
            max_logging.log(
                "[Layerwise] Saved preemption checkpoint at layer"
                f" {layer_idx + 1:02d}/{num_layers:02d} in"
                f" {time.perf_counter() - t_ckpt0:.2f}s"
            )
          except Exception as e:  # pylint: disable=broad-exception-caught
            max_logging.log(
                "[Layerwise] Warning: could not save preemption"
                f" checkpoint: {e}"
            )

        tf_msg = (
            f" ch_rel_l2={p_layer_rec['chained_diff']['all_rel_l2']:.6f}"
            if "chained_diff" in p_layer_rec
            else ""
        )
        max_logging.log(
            f"[Layerwise] Layer {layer_idx:02d}/{num_layers:02d}"
            f" ({'ATTN' if is_full_attn else 'GDN '}) load={load_sec:.2f}s"
            f" exec={exec_sec:.2f}s | packed_rms={p_layer_rec['resid_rms']:.6f}"
            f" doc1_rms={p_layer_rec['doc1_resid_rms']:.6f}"
            f" doc2p_rms={p_layer_rec['doc2plus_resid_rms']:.6f}{tf_msg}"
        )
    finally:
      prefetch_pool.shutdown(wait=False)

    for fh in (f_packed_act, f_clean_act, f_ref_packed, f_ref_clean):
      if fh is not None:
        fh.close()

    with jax.set_mesh(self.mesh), axis_rules(self.config.logical_axis_rules):
      packed_head, packed_logits_diff = self._eval_head_on_split(
          shell_params, packed_batches, h_packed, h_ref_packed_prev
      )
      clean_head: dict[str, float] = {}
      clean_logits_diff: dict[str, float] = {}
      if clean_batches:
        clean_head, clean_logits_diff = self._eval_head_on_split(
            shell_params, clean_batches, h_clean, h_ref_clean_prev
        )

    total_wall_sec = time.perf_counter() - t_start
    clean_summary: dict[str, Any] = {}
    if clean_batches:
      clean_summary = {
          "embed_rms": float(
              np.sqrt(embed_ms_clean / max(embed_tok_clean, 1.0))
          ),
          "head": clean_head,
          "logits_diff_mb0": clean_logits_diff,
          "layers": per_layer_clean,
      }
    return {
        "model_name": self.config.model_name,
        "num_layers": num_layers,
        "num_microbatches": len(packed_batches),
        "total_wall_seconds": total_wall_sec,
        "non_layer_load_seconds": non_layer_load_sec,
        "total_layer_load_seconds": float(np.sum(layer_load_times)),
        "total_layer_exec_seconds": float(np.sum(layer_exec_times)),
        "total_bytes_read": self.reader.total_bytes_read,
        "total_range_reads": self.reader.total_range_reads,
        "packed": {
            "embed_rms": float(
                np.sqrt(embed_ms_packed / max(embed_tok_packed, 1.0))
            ),
            "head": packed_head,
            "logits_diff_mb0": packed_logits_diff,
            "layers": per_layer_packed,
        },
        "clean_single_doc": clean_summary,
    }

  def run_blockwise_forward(
      self,
      batches: Sequence[dict[str, Any]],
      max_blocks: int | None = None,
  ) -> dict[str, Any]:
    """Streams 4-layer `Qwen3_5ScannableBlock`s one at a time."""
    num_blocks = max_blocks or (self.total_layers // self.cycle)
    dev_batches = self._put_batches_on_mesh(batches)
    shell_params = self.reader.load_non_layer_params(
        self.abs_shell_params, self.config, total_layers=self.total_layers
    )
    fwd_block = self.block_fwd

    with jax.set_mesh(self.mesh), axis_rules(self.config.logical_axis_rules):
      h_list, embed_ms_sum, embed_tok_sum = self._embed_batches(
          shell_params, dev_batches
      )
      layer_ms_sums = np.zeros(num_blocks * self.cycle, dtype=np.float64)
      for blk_idx in range(num_blocks):
        blk_params = self.reader.load_single_block_params(
            blk_idx,
            self.abs_block_params,
            self.config,
            self.abs_gdn_params,
            self.abs_attn_params,
            total_layers=self.total_layers,
        )
        for mb_idx, b in enumerate(dev_batches):
          y_next, blk_ms = fwd_block(
              blk_params,
              h_list[mb_idx],
              b["inputs_segmentation"],
              b["inputs_position"],
          )
          h_list[mb_idx] = y_next
          layer_ms_sums[
              blk_idx * self.cycle : (blk_idx + 1) * self.cycle
          ] += np.asarray(blk_ms, dtype=np.float64)
        del blk_params

      head_summary, _ = self._eval_head_on_split(
          shell_params, dev_batches, h_list, None
      )

    layer_rms = [
        float(np.sqrt(ms / max(embed_tok_sum, 1.0))) for ms in layer_ms_sums
    ]
    return {
        "embed_rms": float(np.sqrt(embed_ms_sum / max(embed_tok_sum, 1.0))),
        "layer_rms": layer_rms,
        "head": head_summary,
    }


def _maybe_stage_remote_tokenizer(
    tok_path: str, staged_tok_dir: str = ""
) -> str:
  """Caches remote `tf.io.gfile` tokenizer files in a local temp directory if needed."""
  src_tok_dir = (
      staged_tok_dir
      if (staged_tok_dir and tf.io.gfile.exists(staged_tok_dir))
      else tok_path
  )
  if (
      not isinstance(src_tok_dir, str)
      or not src_tok_dir
      or os.path.exists(src_tok_dir)
  ):
    return src_tok_dir
  if not tf.io.gfile.exists(src_tok_dir):
    return src_tok_dir

  path_hash = hashlib.sha256(src_tok_dir.encode("utf-8")).hexdigest()[:16]
  local_tok_dir = os.path.join(
      tempfile.gettempdir(), "qwen_layerwise_tokenizer_cache", path_hash
  )
  os.makedirs(local_tok_dir, exist_ok=True)
  tok_files = (
      "config.json",
      "generation_config.json",
      "merges.txt",
      "preprocessor_config.json",
      "special_tokens_map.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "vocab.json",
      "chat_template.jinja",
  )

  def _stage_tok_file(fname: str):
    dst_f = os.path.join(local_tok_dir, fname)
    if os.path.exists(dst_f) and os.path.getsize(dst_f) > 0:
      return
    src_f = os.path.join(src_tok_dir, fname)
    if tf.io.gfile.exists(src_f):
      with tf.io.gfile.GFile(src_f, "rb") as fr, open(dst_f, "wb") as fw:
        fw.write(fr.read())

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(tok_files)
  ) as pool:
    list(pool.map(_stage_tok_file, tok_files))
  if os.path.exists(os.path.join(local_tok_dir, "tokenizer_config.json")):
    return local_tok_dir
  return src_tok_dir


def load_reference_step_microbatches(
    config: Any,
    mesh: jax.sharding.Mesh,
    reference_num_hosts: int = 16,
    step_idx: int = 0,
    num_steps: int = 1,
    staged_tokenizer_dir: str = "",
) -> list[dict[str, Any]]:
  """Loads the `reference_num_hosts` microbatches corresponding to a multi-host run."""
  proc_count = jax.process_count()
  proc_idx = jax.process_index()

  if (
      getattr(config, "dataset_type", "") == "tfds"
      and reference_num_hosts >= proc_count
      and reference_num_hosts % proc_count == 0
  ):
    num_groups = reference_num_hosts // proc_count
    use_eval = getattr(config, "eval_steps", 0) > 0 or bool(
        getattr(config, "eval_dataset_name", "")
    )
    ds_name = config.eval_dataset_name if use_eval else config.dataset_name
    ds_split = config.eval_split if use_eval else config.train_split
    data_cols = (
        config.eval_data_columns if use_eval else config.train_data_columns
    )
    tokenize_flag = (
        config.tokenize_eval_data if use_eval else config.tokenize_train_data
    )
    gbs = int(
        config.global_batch_size_to_load_eval
        if use_eval
        else config.global_batch_size_to_load
    )

    tok_path = config.tokenizer_path
    if num_groups > 1:
      tok_path = _maybe_stage_remote_tokenizer(
          tok_path,
          staged_tokenizer_dir
          or os.environ.get("QWEN_STAGED_TOKENIZER_DIR", ""),
      )

    group_iters = []
    for g in range(num_groups):
      host_idx = g * proc_count + proc_idx
      ds = tfds_data_processing.get_datasets(
          dataset_name=ds_name,
          dataset_path=config.dataset_path,
          data_split=ds_split,
          shuffle_files=False if use_eval else config.enable_data_shuffling,
          shuffle_seed=config.data_shuffle_seed,
          dataloading_host_index=host_idx,
          dataloading_host_count=reference_num_hosts,
      )
      dataloader = tfds_data_processing.preprocessing_pipeline(
          dataset=ds,
          tokenizer_path=tok_path,
          tokenizer_type=config.tokenizer_type,
          global_batch_size=gbs,
          max_target_length=config.max_target_length,
          data_column_names=data_cols,
          shuffle=False if use_eval else config.enable_data_shuffling,
          data_shuffle_seed=config.data_shuffle_seed,
          tokenize=tokenize_flag,
          add_bos=config.add_bos,
          add_eos=config.add_eos,
          pack_examples=config.packing,
          hf_access_token=config.hf_access_token,
      )
      group_iters.append(
          multihost_dataloading.MultiHostDataLoadIterator(
              dataloader,
              mesh,
              config.generate_padding_batch_eval if use_eval else False,
          )
      )

    for _ in range(step_idx):
      for it in group_iters:
        next(it)

    batches = []
    for _ in range(num_steps):
      for it in group_iters:
        batches.append(next(it))
    return batches

  if (
      getattr(config, "dataset_type", "") == "grain"
      and proc_count == 1
      and reference_num_hosts > 1
  ):

    class _GrainShardProxy:

      def __getattr__(self, name):
        if name == "expansion_factor_real_data":
          return 1.0 / float(reference_num_hosts)
        if name == "grain_worker_count":
          return 0
        return getattr(config, name)

    iter_list = grain_data_processing.make_grain_train_iterator(
        _GrainShardProxy(), mesh, [0]
    )
    for _ in range(step_idx):
      for it in iter_list:
        next(it)
    batches = []
    for _ in range(num_steps):
      for it in iter_list:
        batches.append(next(it))
    return batches

  train_iter, eval_iter = input_pipeline_interface.create_data_iterator(
      config, mesh
  )
  active_iter = (
      eval_iter
      if (eval_iter is not None and getattr(config, "eval_steps", 0) > 0)
      else train_iter
  )
  if isinstance(active_iter, list):
    for _ in range(step_idx):
      for it in active_iter:
        next(it)
    batches = []
    for _ in range(num_steps):
      for it in active_iter:
        batches.append(next(it))
    return batches

  return [
      next(active_iter)
      for _ in range(max(1, reference_num_hosts // proc_count) * num_steps)
  ]


def compare_layerwise_runs(
    res_a: dict[str, Any], res_b: dict[str, Any]
) -> dict[str, Any]:
  """Merges two layer-wise verification JSON results (`res_a` vs `res_b`) and computes deltas."""
  merged: dict[str, Any] = {
      "model_name": res_a.get("model_name", res_b.get("model_name", "")),
      "num_layers": res_a.get("num_layers", res_b.get("num_layers", 0)),
      "num_microbatches": res_a.get(
          "num_microbatches", res_b.get("num_microbatches", 0)
      ),
      "arm_a_wall_seconds": res_a.get("total_wall_seconds", 0.0),
      "arm_b_wall_seconds": res_b.get("total_wall_seconds", 0.0),
  }
  for mode in ("packed", "clean_single_doc"):
    if not res_a.get(mode) or not res_b.get(mode):
      continue
    pa = res_a[mode]
    pb = res_b[mode]
    ha = pa["head"]
    hb = pb["head"]
    layers_merged = []
    for la, lb in zip(pa["layers"], pb["layers"]):
      l_rec = {
          "layer_idx": la["layer_idx"],
          "layer_type": la["layer_type"],
          "a_resid_rms": la["resid_rms"],
          "b_resid_rms": lb["resid_rms"],
          "resid_rms_delta_a_minus_b": la["resid_rms"] - lb["resid_rms"],
          "a_doc1_resid_rms": la.get("doc1_resid_rms", la["resid_rms"]),
          "b_doc1_resid_rms": lb.get("doc1_resid_rms", lb["resid_rms"]),
          "a_doc2plus_resid_rms": la.get("doc2plus_resid_rms", 0.0),
          "b_doc2plus_resid_rms": lb.get("doc2plus_resid_rms", 0.0),
          "chained_diff": la.get("chained_diff", {}),
          "teacher_forced_diff": la.get("teacher_forced_diff", {}),
      }
      layers_merged.append(l_rec)
    merged[mode] = {
        "embed_rms_a": pa["embed_rms"],
        "embed_rms_b": pb["embed_rms"],
        "head_a": ha,
        "head_b": hb,
        "loss_delta_a_minus_b": ha["eval_loss"] - hb["eval_loss"],
        "doc1_loss_delta_a_minus_b": (
            ha.get("doc1_loss", 0.0) - hb.get("doc1_loss", 0.0)
        ),
        "doc2plus_loss_delta_a_minus_b": (
            ha.get("doc2plus_loss", 0.0) - hb.get("doc2plus_loss", 0.0)
        ),
        "doc2plus_pos0to3_loss_delta_a_minus_b": (
            ha.get("doc2plus_pos0to3_loss", 0.0)
            - hb.get("doc2plus_pos0to3_loss", 0.0)
        ),
        "doc2plus_pos4plus_loss_delta_a_minus_b": (
            ha.get("doc2plus_pos4plus_loss", 0.0)
            - hb.get("doc2plus_pos4plus_loss", 0.0)
        ),
        "logits_diff_mb0": pa.get("logits_diff_mb0", {}),
        "layers": layers_merged,
    }
  return merged


def _resolve_opt(
    flag_val: Any, env_key: str, default: Any, cast_fn: Any = str
) -> Any:
  if flag_val is not None:
    return flag_val
  env_val = os.environ.get(env_key, "")
  if env_val:
    if cast_fn is bool:
      return env_val not in ("0", "false", "False", "no")
    return cast_fn(env_val)
  return default


def run_layerwise_from_config(config: Any) -> dict[str, Any]:
  """Runs layer-wise (and optional block-wise) verification from a MaxText `config`."""
  parsed = FLAGS.is_parsed()
  checkpoint_dir = _resolve_opt(
      FLAGS.checkpoint_dir if parsed else None,
      "QWEN_SAFETENSORS_DIR",
      getattr(config, "hf_checkpoint_path", "")
      or getattr(config, "load_parameters_path", ""),
  )
  if not checkpoint_dir:
    raise ValueError(
        "A checkpoint directory must be provided via --checkpoint_dir, "
        "QWEN_SAFETENSORS_DIR, or load_parameters_path."
    )
  index_json = _resolve_opt(
      FLAGS.safetensors_index_json if parsed else None,
      "QWEN_SAFETENSORS_INDEX_JSON",
      None,
  )
  staged_tok_dir = _resolve_opt(
      FLAGS.staged_tokenizer_dir if parsed else None,
      "QWEN_STAGED_TOKENIZER_DIR",
      "",
  )
  ref_hosts = _resolve_opt(
      FLAGS.ref_hosts if parsed else None,
      "QWEN_LAYERWISE_REF_HOSTS",
      16,
      int,
  )
  step_idx = _resolve_opt(
      FLAGS.step_idx if parsed else None,
      "QWEN_LAYERWISE_STEP_IDX",
      0,
      int,
  )
  num_steps = _resolve_opt(
      FLAGS.num_steps if parsed else None,
      "QWEN_LAYERWISE_NUM_STEPS",
      max(1, int(getattr(config, "eval_steps", 1))),
      int,
  )
  max_layers_raw = _resolve_opt(
      FLAGS.max_layers if parsed else None,
      "QWEN_LAYERWISE_MAX_LAYERS",
      0,
      int,
  )
  max_layers = max_layers_raw if max_layers_raw > 0 else None
  run_clean = _resolve_opt(
      FLAGS.run_clean_single_doc if parsed else None,
      "QWEN_LAYERWISE_RUN_CLEAN",
      True,
      bool,
  )
  run_tf = _resolve_opt(
      FLAGS.include_teacher_forced if parsed else None,
      "QWEN_LAYERWISE_RUN_TEACHER_FORCED",
      True,
      bool,
  )
  run_blockwise = _resolve_opt(
      FLAGS.run_blockwise if parsed else None,
      "QWEN_LAYERWISE_RUN_BLOCKWISE",
      False,
      bool,
  )
  save_mb0_act_dir = _resolve_opt(
      FLAGS.save_mb0_act_dir if parsed else None,
      "QWEN_LAYERWISE_MB0_ACT_DIR",
      "",
  )
  ref_mb0_act_dir = _resolve_opt(
      FLAGS.ref_mb0_act_dir if parsed else None,
      "QWEN_LAYERWISE_REF_MB0_ACT_DIR",
      "",
  )
  ref_json_path = _resolve_opt(
      FLAGS.ref_json if parsed else None,
      "QWEN_LAYERWISE_REF_JSON",
      "",
  )
  out_json_path = _resolve_opt(
      FLAGS.output_json if parsed else None,
      "QWEN_LAYERWISE_OUTPUT_JSON",
      "",
  )

  devices = jax.devices()
  mesh = maxtext_utils.get_mesh_from_config(config, devices)
  max_logging.log(
      "[Layerwise] Starting Qwen3.5 layer-wise verification on"
      f" mesh={mesh.shape} checkpoint_dir={checkpoint_dir}"
      f" ref_hosts={ref_hosts} step_idx={step_idx} num_steps={num_steps}"
  )

  t_data0 = time.perf_counter()
  batches = load_reference_step_microbatches(
      config,
      mesh,
      reference_num_hosts=ref_hosts,
      step_idx=step_idx,
      num_steps=num_steps,
      staged_tokenizer_dir=staged_tok_dir,
  )
  data_sec = time.perf_counter() - t_data0
  max_logging.log(
      f"[Layerwise] Loaded {len(batches)} microbatches in {data_sec:.2f}s"
  )

  verifier = Qwen35LayerwiseVerifier(
      config=config,
      mesh=mesh,
      checkpoint_dir=checkpoint_dir,
      total_layers=int(config.num_decoder_layers),
      safetensors_index_json=index_json,
  )
  progress_json_path = f"{out_json_path}.partial.json" if out_json_path else ""
  result = verifier.run_layerwise_verification(
      batches=batches,
      run_clean_single_doc=run_clean,
      include_teacher_forced=run_tf,
      max_layers=max_layers,
      save_mb0_act_dir=save_mb0_act_dir,
      ref_mb0_act_dir=ref_mb0_act_dir,
      progress_json_path=progress_json_path,
  )
  result["data_load_seconds"] = data_sec

  if run_blockwise and (max_layers is None or max_layers % verifier.cycle == 0):
    num_blk = (max_layers // verifier.cycle) if max_layers is not None else None
    t_blk0 = time.perf_counter()
    blk_res = verifier.run_blockwise_forward(batches, max_blocks=num_blk)
    result["blockwise_packed"] = {
        "wall_seconds": time.perf_counter() - t_blk0,
        "result": blk_res,
    }

  p_head = result["packed"]["head"]
  max_logging.log(
      f"[Layerwise Summary Packed] eval_loss={p_head['eval_loss']:.8f}"
      f" valid_tok={p_head['valid_tokens']:.0f} |"
      f" doc1_loss={p_head['doc1_loss']:.8f}"
      f" doc1_tok={p_head['doc1_tokens']:.0f} |"
      f" doc2plus_loss={p_head['doc2plus_loss']:.8f}"
      f" doc2plus_tok={p_head['doc2plus_tokens']:.0f} |"
      f" final_norm_rms={p_head['final_norm_rms']:.8f}"
      f" logits_rms={p_head['logits_rms']:.8f}"
  )
  if result.get("clean_single_doc"):
    c_head = result["clean_single_doc"]["head"]
    max_logging.log(
        f"[Layerwise Summary Clean1Doc] eval_loss={c_head['eval_loss']:.8f}"
        f" valid_tok={c_head['valid_tokens']:.0f} |"
        f" final_norm_rms={c_head['final_norm_rms']:.8f}"
        f" logits_rms={c_head['logits_rms']:.8f}"
    )

  if ref_json_path and tf.io.gfile.exists(ref_json_path):
    with tf.io.gfile.GFile(ref_json_path, "r") as fr:
      ref_res = json.load(fr)
    comparison = compare_layerwise_runs(result, ref_res)
    result["comparison_vs_ref"] = comparison
    cp = comparison["packed"]
    max_logging.log(
        f"[Layerwise A-vs-B Packed] eval_loss_A={cp['head_a']['eval_loss']:.8f}"
        f" eval_loss_B={cp['head_b']['eval_loss']:.8f}"
        f" delta={cp['loss_delta_a_minus_b']:+.8f} |"
        f" doc1_delta={cp['doc1_loss_delta_a_minus_b']:+.8f}"
        f" doc2plus_delta={cp['doc2plus_loss_delta_a_minus_b']:+.8f}"
    )

  for line in json.dumps(result, indent=2, sort_keys=True).splitlines():
    print(f"QWEN_LAYERWISE_JSON: {line}", flush=True)

  if out_json_path and jax.process_index() == 0:
    out_dir = os.path.dirname(out_json_path)
    if out_dir:
      tf.io.gfile.makedirs(out_dir)
    with tf.io.gfile.GFile(out_json_path, "w") as f:
      json.dump(result, f, indent=2, sort_keys=True)
    max_logging.log(f"[Layerwise] Wrote full JSON report to {out_json_path}")

  return result


def main(argv: Sequence[str]) -> None:
  argv_list = list(argv)
  for idx, arg in enumerate(argv_list[1:], start=1):
    if arg.startswith("tokenizer_path="):
      raw_tok = arg.split("=", 1)[1]
      staged = _maybe_stage_remote_tokenizer(
          raw_tok, FLAGS.staged_tokenizer_dir or ""
      )
      argv_list[idx] = f"tokenizer_path={staged}"
  config = pyconfig.initialize(argv_list)
  run_layerwise_from_config(config)


if __name__ == "__main__":
  app.run(main)
