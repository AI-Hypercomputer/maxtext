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

"""CPU unit tests for parallel streaming multi-shard HF weight audit in `deepseek_v41_mm_audit.py`."""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import types
import unittest
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import transformers
from etils import epath
from flax import nnx

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils import deepseek_v41_frozen_topk
from tests.utils import deepseek_v41_mm_audit

try:
  import torch
  from safetensors.torch import save_file

  from tests.utils.deepseek_v41_reference import multimodal

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  save_file = None  # type: ignore[assignment]
  multimodal = None  # type: ignore[assignment]
  HAS_TORCH = False

from tests.utils import deepseek_v41_mm_testing as mm_testing
import safetensors.numpy

_EXPERT_WEIGHT_KEY_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")


def _save_two_shard_safetensors_dir(
    out_dir: str | os.PathLike[str],
    tensors: Mapping[str, Any],
) -> None:
  """Writes a 2-shard safetensors directory with `model.safetensors.index.json`."""
  out_path = Path(out_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  sorted_keys = sorted(tensors.keys())
  shard_tensors: dict[str, dict[str, Any]] = {
      "model-00001-of-00002.safetensors": {},
      "model-00002-of-00002.safetensors": {},
  }
  weight_map: dict[str, str] = {}
  for idx, key in enumerate(sorted_keys):
    shard_name = "model-00001-of-00002.safetensors" if idx % 2 == 0 else "model-00002-of-00002.safetensors"
    shard_tensors[shard_name][key] = tensors[key]
    weight_map[key] = shard_name
  for shard_name, sub_dict in shard_tensors.items():
    if HAS_TORCH and save_file is not None:
      save_file(sub_dict, (out_path / shard_name).as_posix(), metadata={"format": "pt"})
    else:
      safetensors.numpy.save_file(sub_dict, (out_path / shard_name).as_posix(), metadata={"format": "np"})

  def _tensor_size(t: Any) -> int:
    if hasattr(t, "nbytes"):
      return int(t.nbytes)
    return int(t.numel() * t.element_size())

  index_payload = {
      "metadata": {"total_size": sum(_tensor_size(t) for t in tensors.values())},
      "weight_map": weight_map,
  }
  with open(out_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
    json.dump(index_payload, f, indent=2)
  with open(out_path / "config.json", "w", encoding="utf-8") as f:
    json.dump({"model_type": "deepseek_v41"}, f, indent=2)


def _build_mini_hf_tensors_from_ref(ref_mm: Any) -> dict[str, Any]:
  """Extracts a complete HF safetensors dictionary (`BF16` weights + `F32` `bias`/`bias_vl` + `mtp.*`) from `ref_mm`."""
  out: dict[str, torch.Tensor] = {}
  for name, tensor in ref_mm.named_parameters():
    t_cpu = tensor.detach().cpu().contiguous()
    if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
      out[name] = t_cpu.to(torch.float32).contiguous()
    else:
      out[name] = t_cpu.to(torch.bfloat16).contiguous()
  for name, tensor in ref_mm.named_buffers():
    if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
      out[name] = tensor.detach().cpu().to(torch.float32).contiguous()
  out["mtp.0.norm.weight"] = torch.ones((16,), dtype=torch.bfloat16)
  out["mtp.0.ffn.gate.bias_vl"] = torch.zeros((4,), dtype=torch.float32)
  return out


def _copy_unscanned_to_scanned(mt_scanned: Any, mt_unscanned: Any, config: Any) -> None:
  """Copies weights from an unscanned mini model to a scanned mini model."""
  if config.use_multimodal:
    _, v_p, v_b, _ = nnx.split(mt_unscanned.vision_encoder, nnx.Param, moe.MoEBiasVar, ...)
    nnx.update(mt_scanned.vision_encoder, v_p, v_b)
  mt_scanned.token_embedder.embedding[...] = mt_unscanned.token_embedder.embedding[...]
  mt_scanned.decoder.decoder_norm.scale[...] = mt_unscanned.decoder.decoder_norm.scale[...]
  mt_scanned.decoder.logits_dense.kernel[...] = mt_unscanned.decoder.logits_dense.kernel[...]

  specs = deepseek41.build_layer_specs(config)
  scan_axis = config.param_scan_axis
  for k, segment in enumerate(deepseek41.build_segments(config)):
    group_params = []
    group_biases = []
    for group in segment.groups:
      tmp_block = deepseek41.DeepSeek41Block(
          config=config,
          mesh=mt_scanned.mesh,
          model_mode=mt_scanned.decoder.model_mode,
          specs=tuple(specs[idx] for idx in group),
          quant=mt_scanned.quant,
          rngs=nnx.Rngs(params=0),
      )
      for j, layer_idx in enumerate(group):
        unscanned_layer = getattr(mt_unscanned.decoder, f"layers_{layer_idx}")
        _, p_state, b_state, _ = nnx.split(unscanned_layer, nnx.Param, moe.MoEBiasVar, ...)
        nnx.update(getattr(tmp_block, f"layers_{j}"), p_state, b_state)
      _, p_state, b_state, _ = nnx.split(tmp_block, nnx.Param, moe.MoEBiasVar, ...)
      group_params.append(nnx.to_pure_dict(p_state))
      group_biases.append(nnx.to_pure_dict(b_state))

    def _stack_param_leaves(*leaves: jax.Array) -> jax.Array:
      stacked = jnp.stack(leaves, axis=0)
      return jnp.moveaxis(stacked, 0, scan_axis) if stacked.ndim > scan_axis else stacked

    def _stack_bias_leaves(*leaves: jax.Array) -> jax.Array:
      return jnp.stack(leaves, axis=0)

    stacked_params = jax.tree.map(_stack_param_leaves, group_params[0], *group_params[1:])
    stacked_biases = jax.tree.map(_stack_bias_leaves, group_biases[0], *group_biases[1:])
    stack = getattr(mt_scanned.decoder, f"segments_{k}")
    graphdef, s_params, s_biases, s_other = nnx.split(stack, nnx.Param, moe.MoEBiasVar, ...)
    nnx.replace_by_pure_dict(s_params, stacked_params)
    nnx.replace_by_pure_dict(s_biases, stacked_biases)
    setattr(mt_scanned.decoder, f"segments_{k}", nnx.merge(graphdef, s_params, s_biases, s_other))


def _inverse_transform(mt_arr: np.ndarray, transform: str, config: Any, unpadded_rows: int | None = None) -> np.ndarray:
  """Inverts MaxText parameter transforms to reconstruct exact HF safetensors layout."""
  if transform == "identity":
    return np.ascontiguousarray(mt_arr)
  if transform == "transpose_2d":
    return np.ascontiguousarray(mt_arr.T)
  if transform in ("wq_b_reshape", "indexer_wq_b_reshape"):
    return np.ascontiguousarray(mt_arr.reshape(int(config.q_lora_rank), -1).T)
  if transform == "wo_a_reshape":
    reshaped = np.transpose(mt_arr, (0, 2, 1))
    return np.ascontiguousarray(reshaped.reshape(int(config.o_groups) * int(config.o_lora_rank), -1))
  if transform == "engram_pad":
    if unpadded_rows is not None:
      return np.ascontiguousarray(mt_arr[:unpadded_rows])
    return np.ascontiguousarray(mt_arr)
  raise ValueError(f"Unknown transform: {transform}")


def _build_mini_hf_tensors_from_maxtext(model_un: nnx.Module, config: Any) -> dict[str, np.ndarray]:
  """Constructs bit-exact 2-shard mini HF safetensors dict directly from an initialized MaxText model."""
  _, params_state, bias_state, _ = nnx.split(model_un, nnx.Param, moe.MoEBiasVar, ...)
  merged_state = nnx.merge_state(params_state, bias_state)
  pure_weights = nnx.to_pure_dict(merged_state)
  leaves = deepseek_v41_mm_audit.flatten_state_arrays(pure_weights)
  rules = deepseek_v41_mm_audit.build_independent_audit_rules(config)

  engram_unpadded_map = {}
  for lid in config.ds41_engram_layers:
    layer = getattr(model_un.decoder, f"layers_{lid}")
    engram_unpadded_map[lid] = int(layer.engram_hasher.val.num_embeddings)

  hf_tensors: dict[str, np.ndarray] = {}
  for r in rules:
    arr = np.asarray(leaves[r.maxtext_path])
    if r.kind == "single":
      hf_k = r.hf_keys[0]
      unpadded = None
      if r.transform == "engram_pad":
        layer_idx = int(r.maxtext_path.split("/")[1].replace("layers_", ""))
        unpadded = engram_unpadded_map[layer_idx]
      hf_tensors[hf_k] = _inverse_transform(arr, r.transform, config, unpadded)
    elif r.kind == "expert_stack":
      for exp_idx, hf_k in enumerate(r.hf_keys):
        sub_arr = arr[exp_idx]
        hf_tensors[hf_k] = _inverse_transform(sub_arr, r.transform, config)

  hf_tensors["mtp.0.norm.weight"] = np.asarray(jnp.ones((16,), dtype=jnp.bfloat16))
  hf_tensors["mtp.0.ffn.gate.bias_vl"] = np.zeros((4,), dtype=np.float32)
  return hf_tensors


def _reshard_model_to_abstract_mesh(
    model: nnx.Module,
    config: Any,
    mesh: jax.sharding.Mesh,
) -> None:
  """Places every `nnx.Param` and `MoEBiasVar` leaf of `model` onto its expected `NamedSharding` on `mesh`."""
  _, abs_model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh)
  _, p_state, b_state, rest_state = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
  _, abs_p_state, abs_b_state, _ = nnx.split(abs_model, nnx.Param, moe.MoEBiasVar, ...)

  def _put_leaf(arr: Any, aval: Any) -> Any:
    return jax.device_put(arr, aval.sharding)

  p_sharded = jax.tree.map(_put_leaf, p_state, abs_p_state)
  b_sharded = jax.tree.map(_put_leaf, b_state, abs_b_state)
  nnx.update(model, p_sharded, b_sharded, rest_state)


def _flip_single_bf16_element(
    arr: jax.Array,
    coord: tuple[int, ...],
) -> jax.Array:
  """Returns a copy of `arr` on the exact same `NamedSharding` with 1 ULP flipped at `coord`."""
  np_arr = np.array(arr, copy=True)
  u16 = np_arr.view(np.uint16)
  u16[coord] = np.uint16(int(u16[coord]) ^ 1)
  return jax.device_put(jnp.asarray(np_arr), arr.sharding)


def _replace_model_leaf(model: nnx.Module, target_path: str, new_leaf: jax.Array) -> jax.Array:
  """Replaces `target_path` inside `model` (`Param` or `MoEBiasVar`) and returns the original `jax.Array` leaf."""
  _, p_state, b_state, rest_state = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
  p_pure = nnx.to_pure_dict(p_state)
  b_pure = nnx.to_pure_dict(b_state)
  parts = target_path.split("/")

  def _lookup_and_set(pure_tree: dict[str, Any]) -> tuple[bool, jax.Array | None]:
    cursor: Any = pure_tree
    for part in parts[:-1]:
      if not isinstance(cursor, Mapping) or part not in cursor:
        return False, None
      cursor = cursor[part]
    if not isinstance(cursor, dict) or parts[-1] not in cursor:
      return False, None
    orig = cursor[parts[-1]]
    cursor[parts[-1]] = new_leaf
    return True, orig

  found_p, orig_leaf = _lookup_and_set(p_pure)
  if found_p:
    nnx.replace_by_pure_dict(p_state, p_pure)
  else:
    found_b, orig_leaf = _lookup_and_set(b_pure)
    if not found_b:
      raise KeyError(f"Leaf {target_path!r} not found in Param or MoEBiasVar state")
    nnx.replace_by_pure_dict(b_state, b_pure)
  assert orig_leaf is not None
  nnx.update(model, p_state, b_state, rest_state)
  return orig_leaf


def _run_tracked_audit(
    *,
    model: nnx.Module,
    config: Any,
    hf_checkpoint_dir: Path,
    mesh: jax.sharding.Mesh,
) -> tuple[dict[str, Any], collections.Counter[str], int, list[Any]]:
  """Runs `audit_loaded_model_against_hf` while recording `load_raw_tensor` call counts and peak `_tensor_cache` size."""
  raw_counts: collections.Counter[str] = collections.Counter()
  count_lock = threading.Lock()
  captured_readers: list[Any] = []
  max_cache_len = 0

  orig_init = deepseek_v41_mm_audit._SafetensorsShardReader.__init__  # pylint: disable=protected-access
  orig_load_raw = deepseek_v41_mm_audit._SafetensorsShardReader.load_raw_tensor  # pylint: disable=protected-access

  def _tracked_init(reader_self: Any, checkpoint_dir: str | os.PathLike[str]) -> None:
    orig_init(reader_self, checkpoint_dir)
    captured_readers.append(reader_self)

  def _tracked_load_raw(reader_self: Any, key: str) -> Any:
    nonlocal max_cache_len
    out = orig_load_raw(reader_self, key)
    with count_lock:
      raw_counts[key] += 1
      cur_cache_len = len(reader_self._tensor_cache)  # pylint: disable=protected-access
      max_cache_len = max(max_cache_len, cur_cache_len)
    return out

  with (
      mock.patch.object(
          deepseek_v41_mm_audit._SafetensorsShardReader,  # pylint: disable=protected-access
          "__init__",
          _tracked_init,
      ),
      mock.patch.object(
          deepseek_v41_mm_audit._SafetensorsShardReader,  # pylint: disable=protected-access
          "load_raw_tensor",
          _tracked_load_raw,
      ),
  ):
    res = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
        model=model,
        config=config,
        hf_checkpoint_dir=hf_checkpoint_dir,
        mesh=mesh,
        verify_all_text_leaves=True,
    )
  return res, raw_counts, max_cache_len, captured_readers


class _ChunkedReadStream:
  """Wraps raw bytes to return at most `max_chunk` bytes per `.read()` and optionally raise `OSError`."""

  def __init__(
      self,
      payload: bytes,
      *,
      max_chunk: int,
      fail_on_read: bool = False,
      always_fail: bool = False,
  ) -> None:
    self._buf = io.BytesIO(payload)
    self._max_chunk = max_chunk
    self._fail_on_read = fail_on_read
    self._always_fail = always_fail
    self.closed = False

  def seek(self, offset: int) -> int:
    if self.closed:
      raise ValueError("I/O operation on closed stream")
    return self._buf.seek(offset)

  def read(self, size: int = -1) -> bytes:
    if self.closed:
      raise ValueError("I/O operation on closed stream")
    if self._always_fail:
      raise OSError("Simulated persistent GCS stream failure")
    if self._fail_on_read:
      self._fail_on_read = False
      raise OSError("Simulated stale GCS HTTP stream reset")
    limit = self._max_chunk if size < 0 else min(size, self._max_chunk)
    return self._buf.read(limit)

  def close(self) -> None:
    self.closed = True
    self._buf.close()

  def __enter__(self) -> "_ChunkedReadStream":
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self.close()


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class DeepSeekV41MmAuditStreamingTest(unittest.TestCase):
  """Verifies single-pass multi-shard streaming audit, zero tensor caching, thread-local handles, and negative controls."""

  @classmethod
  def setUpClass(cls) -> None:
    """Initializes the 2-shard mini multimodal safetensors fixture."""
    if len(jax.devices()) < 4:
      raise RuntimeError(
          f"DeepSeekV41MmAuditStreamingTest requires >= 4 CPU devices (got {len(jax.devices())}); "
          "set XLA_FLAGS=--xla_force_host_platform_device_count=4"
      )
    cls._orig_matmul_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    cls.addClassCleanup(jax.config.update, "jax_default_matmul_precision", cls._orig_matmul_precision)
    cls.tmp_root = Path(tempfile.mkdtemp(prefix="ds41_mm_audit_unit_"))
    cls.addClassCleanup(shutil.rmtree, cls.tmp_root, ignore_errors=True)
    if HAS_TORCH and multimodal is not None:
      cls.tokenizer = transformers.AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)
      cls.ref_args = multimodal.mini_model_args()
      cls.ref_mm = multimodal.build_multimodal_model(cls.ref_args, tokenizer=cls.tokenizer)
      multimodal.init_mini_reference(cls.ref_mm, seed=42)
      for layer in cls.ref_mm.layers:
        with torch.no_grad():
          layer.ffn.gate.bias.copy_(torch.linspace(-0.05, 0.05, cls.ref_args.n_routed_experts, dtype=torch.float32))
          if layer.ffn.gate.bias_vl is not None:
            layer.ffn.gate.bias_vl.copy_(torch.linspace(0.1, -0.1, cls.ref_args.n_routed_experts, dtype=torch.float32))
      with torch.no_grad():
        for name, param in cls.ref_mm.named_parameters():
          if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
            param.copy_(param.to(torch.float32))
          else:
            param.copy_(param.to(torch.bfloat16).to(torch.float32))
      cls.hf_tensors = _build_mini_hf_tensors_from_ref(cls.ref_mm)
      cls.mt_unscanned_source = None
    else:
      cls.tokenizer = None
      cls.ref_args = types.SimpleNamespace(dim=64, moe_inter_dim=32, n_routed_experts=4)
      cls.ref_mm = None
      cfg_un = mm_testing.mini_vision_maxtext_config(
          scan_layers=False,
          dtype="float32",
          weight_dtype="bfloat16",
          dtype_mm="float32",
          matmul_precision="highest",
          remat_policy="none",
          ds41_activation_quant=False,
          per_device_batch_size=1.0,
          ici_fsdp_parallelism=2,
          ici_tensor_parallelism=2,
          ici_expert_parallelism=1,
          max_target_length=40,
          tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
          ds41_engram_compressed_vocab_size=0,
      )
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_un), cfg_un.mesh_axes)
      cls.mt_unscanned_source = model_creation_utils.from_config(
          cfg_un,
          mesh=mesh,
          rngs=nnx.Rngs(params=0),
          model_mode=MODEL_MODE_TRAIN,
      )
      _reshard_model_to_abstract_mesh(cls.mt_unscanned_source, cfg_un, mesh)
      cls.hf_tensors = _build_mini_hf_tensors_from_maxtext(cls.mt_unscanned_source, cfg_un)
    cls.hf_ckpt_dir = cls.tmp_root / "hf_two_shard"
    _save_two_shard_safetensors_dir(cls.hf_ckpt_dir, cls.hf_tensors)
    cls.expert_weight_keys = sorted(k for k in cls.hf_tensors if _EXPERT_WEIGHT_KEY_RE.match(k))

  @classmethod
  def tearDownClass(cls) -> None:
    """Removes the temporary safetensors directory and restores matmul precision."""
    jax.config.update("jax_default_matmul_precision", cls._orig_matmul_precision)
    shutil.rmtree(cls.tmp_root, ignore_errors=True)

  def _build_trailing_sharded_model(
      self,
      *,
      scan_layers: bool,
      param_scan_axis: int,
  ) -> tuple[Any, jax.sharding.Mesh, nnx.Module]:
    """Builds a 4-device `(fsdp=2, tensor=2)` mini model sharding trailing `(embed, mlp)` axes."""
    mini_seq_len = 40 if multimodal is None else multimodal.MINI_SEQ_LEN
    tok_overrides = (
        {}
        if HAS_TORCH
        else {
            "tokenizer_path": mm_testing.OFFLINE_TOKENIZER_PATH,
            "ds41_engram_compressed_vocab_size": 0,
        }
    )
    cfg = mm_testing.mini_vision_maxtext_config(
        scan_layers=scan_layers,
        param_scan_axis=param_scan_axis,
        dtype="float32",
        weight_dtype="bfloat16",
        dtype_mm="float32",
        matmul_precision="highest",
        remat_policy="none",
        ds41_activation_quant=False,
        per_device_batch_size=1.0,
        ici_fsdp_parallelism=2,
        ici_tensor_parallelism=2,
        ici_expert_parallelism=1,
        max_target_length=mini_seq_len,
        **tok_overrides,
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = model_creation_utils.from_config(
        cfg,
        mesh=mesh,
        rngs=nnx.Rngs(params=0),
        model_mode=MODEL_MODE_TRAIN,
    )
    if HAS_TORCH and self.ref_mm is not None:
      mm_testing.copy_full_multimodal_model(model, self.ref_mm, cfg, jnp.bfloat16)
    else:
      if scan_layers:
        _copy_unscanned_to_scanned(model, self.mt_unscanned_source, cfg)
      else:
        _, p_state, b_state, _ = nnx.split(self.mt_unscanned_source, nnx.Param, moe.MoEBiasVar, ...)
        nnx.update(model, p_state, b_state)
    _reshard_model_to_abstract_mesh(model, cfg, mesh)
    return cfg, mesh, model

  def test_trailing_sharded_expert_audit_single_pass_no_cache_and_corruption_control(self) -> None:
    """Verifies single-pass expert reads (`count==1` on 4 shards), empty cache, and 1-ULP corruption control."""
    self.assertGreater(len(self.expert_weight_keys), 0)

    for label, scan_layers, scan_axis in (
        ("scanned_axis1", True, 1),
        ("scanned_axis0", True, 0),
        ("unscanned", False, 0),
    ):
      with self.subTest(layout=label):
        cfg, mesh, model = self._build_trailing_sharded_model(
            scan_layers=scan_layers,
            param_scan_axis=scan_axis,
        )
        self.assertEqual(mesh.devices.size, 4)

        _, p_state, b_state, _ = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
        leaves = deepseek_v41_mm_audit.flatten_state_arrays(nnx.to_pure_dict(nnx.merge_state(p_state, b_state)))

        if scan_layers:
          rep_axis = scan_axis
          exp_axis = 1 if rep_axis == 0 else 0
          candidate_paths = [
              k for k, v in sorted(leaves.items()) if k.endswith("/mlp/MoeBlock_0/wi_0") and v.shape[rep_axis] >= 2
          ]
          self.assertTrue(candidate_paths, msg=f"Expected scanned segment with L >= 2 in {sorted(leaves.keys())}")
          target_leaf_path = candidate_paths[0]
          target_arr = leaves[target_leaf_path]
          num_reps = target_arr.shape[rep_axis]
          num_exps = target_arr.shape[exp_axis]
          self.assertGreaterEqual(num_reps, 2)
          self.assertGreaterEqual(num_exps, 4)
          rep0_shards = [s for s in target_arr.addressable_shards if s.replica_id == 0]
          self.assertEqual(len(rep0_shards), 4)
          unique_idx_keys = {tuple((sl.start, sl.stop, sl.step) for sl in s.index) for s in rep0_shards}
          self.assertEqual(len(unique_idx_keys), 4)
          for s in rep0_shards:
            self.assertEqual(s.index[rep_axis].indices(num_reps), (0, num_reps, 1))
            self.assertEqual(s.index[exp_axis].indices(num_exps), (0, num_exps, 1))
          nonzero_shards = [
              s
              for s in rep0_shards
              if s.index[2].indices(target_arr.shape[2])[0] > 0 and s.index[3].indices(target_arr.shape[3])[0] > 0
          ]
          self.assertEqual(len(nonzero_shards), 1)
          nz_shard = nonzero_shards[0]
          r_start = nz_shard.index[2].indices(target_arr.shape[2])[0]
          c_start = nz_shard.index[3].indices(target_arr.shape[3])[0]
          corrupt_coord = (1, 2, r_start + 1, c_start + 1) if rep_axis == 0 else (2, 1, r_start + 1, c_start + 1)
        else:
          target_leaf_path = "decoder/layers_1/mlp/MoeBlock_0/wi_0"
          target_arr = leaves[target_leaf_path]
          num_exps = target_arr.shape[0]
          self.assertGreaterEqual(num_exps, 4)
          rep0_shards = [s for s in target_arr.addressable_shards if s.replica_id == 0]
          self.assertEqual(len(rep0_shards), 4)
          unique_idx_keys = {tuple((sl.start, sl.stop, sl.step) for sl in s.index) for s in rep0_shards}
          self.assertEqual(len(unique_idx_keys), 4)
          for s in rep0_shards:
            self.assertEqual(s.index[0].indices(num_exps), (0, num_exps, 1))
          nonzero_shards = [
              s
              for s in rep0_shards
              if s.index[1].indices(target_arr.shape[1])[0] > 0 and s.index[2].indices(target_arr.shape[2])[0] > 0
          ]
          self.assertEqual(len(nonzero_shards), 1)
          nz_shard = nonzero_shards[0]
          r_start = nz_shard.index[1].indices(target_arr.shape[1])[0]
          c_start = nz_shard.index[2].indices(target_arr.shape[2])[0]
          corrupt_coord = (2, r_start + 1, c_start + 1)

        res_clean, raw_counts, max_cache_len, captured_readers = _run_tracked_audit(
            model=model,
            config=cfg,
            hf_checkpoint_dir=self.hf_ckpt_dir,
            mesh=mesh,
        )

        self.assertTrue(res_clean["passed"], msg=f"Clean audit failed for {label}: {res_clean}")
        self.assertEqual(res_clean["total_bit_mismatches"], 0)
        self.assertEqual(res_clean["total_leaf_failures"], 0)
        self.assertEqual(len(captured_readers), 1)
        self.assertEqual(max_cache_len, 0)
        self.assertEqual(len(captured_readers[0]._tensor_cache), 0)  # pylint: disable=protected-access
        self.assertEqual(len(captured_readers[0]._open_handles), 0)  # pylint: disable=protected-access
        for exp_key in self.expert_weight_keys:
          self.assertEqual(
              raw_counts[exp_key],
              1,
              msg=f"Expert key {exp_key} read {raw_counts[exp_key]} times (expected 1 across 4 shards) in {label}",
          )

        # Negative control: corrupt 1 element in the non-zero (fsdp=1, tensor=1) shard at (r=1, e=2, ...).
        corrupted_leaf = _flip_single_bf16_element(target_arr, corrupt_coord)
        orig_leaf = _replace_model_leaf(model, target_leaf_path, corrupted_leaf)
        try:
          res_bad = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
              model=model,
              config=cfg,
              hf_checkpoint_dir=self.hf_ckpt_dir,
              mesh=mesh,
              verify_all_text_leaves=True,
          )
        finally:
          _replace_model_leaf(model, target_leaf_path, orig_leaf)

        self.assertFalse(res_bad["passed"])
        self.assertEqual(res_bad["total_bit_mismatches"], 1)
        self.assertEqual(res_bad["total_leaf_failures"], 1)
        self.assertEqual(len(res_bad["leaf_failures"]), 1)
        self.assertIn(
            f"{target_leaf_path} shard {nz_shard.index}: 1 bitwise element mismatches",
            res_bad["leaf_failures"][0],
        )

  def _verify_shard_reader_stream_lifecycle(self, *, use_generator_cm: bool) -> None:
    """Runs partial-read, handle-reuse, stale-`OSError` reopen, and `close()` checks for direct or `@contextmanager` open."""
    shard_file = self.hf_ckpt_dir / "model-00001-of-00002.safetensors"
    shard_bytes = shard_file.read_bytes()

    with deepseek_v41_mm_audit._SafetensorsShardReader(self.hf_ckpt_dir) as baseline_reader:  # pylint: disable=protected-access
      sample_keys = [k for k, s in sorted(baseline_reader.weight_map.items()) if s == shard_file.name][:4]
      self.assertEqual(len(sample_keys), 4)
      two_d_key = next(
          k
          for k, s in sorted(baseline_reader.weight_map.items())
          if s == shard_file.name
          and len(baseline_reader.get_header_entry(k)["shape"]) == 2
          and int(baseline_reader.get_header_entry(k)["shape"][0]) >= 6
      )
      baseline_raw_first = baseline_reader.load_raw_tensor(sample_keys[0])
      baseline_raw_slice = baseline_reader.load_raw_row_slice(two_d_key, 1, 5)
      baseline_arrays = {k: baseline_reader.load_dequantized_tensor(k, "bfloat16", use_cache=False) for k in sample_keys}
      self.assertEqual(len(baseline_reader._tensor_cache), 0)  # pylint: disable=protected-access

    @contextlib.contextmanager
    def _make_gcs_cm(stream_obj: _ChunkedReadStream) -> Iterator[_ChunkedReadStream]:
      try:
        yield stream_obj
      finally:
        stream_obj.close()

    if use_generator_cm:
      probe_stream = _ChunkedReadStream(b"probe", max_chunk=2)
      probe_cm = _make_gcs_cm(probe_stream)
      self.assertNotIn("seek", dir(probe_cm))
      self.assertNotIn("close", dir(probe_cm))
      with probe_cm as entered_probe:
        self.assertEqual(entered_probe.read(2), b"pr")
      self.assertTrue(probe_stream.closed)

    open_calls = 0
    created_streams: list[_ChunkedReadStream] = []
    persistent_fail = False
    concrete_epath_cls = type(epath.Path(self.hf_ckpt_dir))
    orig_epath_open = concrete_epath_cls.open

    def _chunked_open(path_self: epath.Path, *args: Any, **kwargs: Any) -> Any:
      nonlocal open_calls
      mode = args[0] if args else (kwargs["mode"] if "mode" in kwargs else "r")
      if mode == "rb" and path_self.name == shard_file.name:
        open_calls += 1
        stream = _ChunkedReadStream(
            shard_bytes,
            max_chunk=7,
            fail_on_read=False,
            always_fail=persistent_fail,
        )
        created_streams.append(stream)
        return _make_gcs_cm(stream) if use_generator_cm else stream
      return orig_epath_open(path_self, *args, **kwargs)

    with mock.patch.object(concrete_epath_cls, "open", _chunked_open):
      # Exercises _read_safetensors_header across 7-byte partial stream chunks as well.
      reader = deepseek_v41_mm_audit._SafetensorsShardReader(self.hf_ckpt_dir)  # pylint: disable=protected-access
      self.assertEqual(open_calls, 1)
      self.assertTrue(created_streams[0].closed)
      open_calls = 0
      created_streams.clear()

      with reader:
        got_raw = reader.load_raw_tensor(sample_keys[0])
        if HAS_TORCH and torch is not None:
          self.assertTrue(torch.equal(got_raw.view(torch.uint8), baseline_raw_first.view(torch.uint8)))
        else:
          self.assertTrue(
              np.array_equal(np.asarray(got_raw).view(np.uint8), np.asarray(baseline_raw_first).view(np.uint8))
          )
        self.assertEqual(open_calls, 1)
        self.assertEqual(len(reader._open_handles), 1)  # pylint: disable=protected-access

        got_row_slice = reader.load_raw_row_slice(two_d_key, 1, 5)
        if HAS_TORCH and torch is not None:
          self.assertTrue(torch.equal(got_row_slice.view(torch.uint8), baseline_raw_slice.view(torch.uint8)))
        else:
          self.assertTrue(
              np.array_equal(np.asarray(got_row_slice).view(np.uint8), np.asarray(baseline_raw_slice).view(np.uint8))
          )
        self.assertEqual(open_calls, 1)

        got_first = reader.load_dequantized_tensor(sample_keys[0], "bfloat16", use_cache=False)
        self.assertEqual(open_calls, 1)

        # Inject one-shot stale-stream OSError on the cached thread-local stream; next read must reopen cleanly.
        created_streams[0]._fail_on_read = True  # pylint: disable=protected-access
        got_second = reader.load_dequantized_tensor(sample_keys[1], "bfloat16", use_cache=False)
        self.assertEqual(open_calls, 2)
        self.assertTrue(created_streams[0].closed)
        self.assertEqual(len(reader._open_handles), 1)  # pylint: disable=protected-access

        # Subsequent reads on the same thread reuse the reopened handle without opening a 3rd stream.
        got_third = reader.load_dequantized_tensor(sample_keys[2], "bfloat16", use_cache=False)
        got_fourth = reader.load_dequantized_tensor(sample_keys[3], "bfloat16", use_cache=False)
        self.assertEqual(open_calls, 2)
        self.assertEqual(len(reader._tensor_cache), 0)  # pylint: disable=protected-access

        # Negative control: if the reopened stream ALSO fails with OSError, the error propagates loudly.
        created_streams[1]._fail_on_read = True  # pylint: disable=protected-access
        persistent_fail = True
        with self.assertRaisesRegex(OSError, "Simulated persistent GCS stream failure"):
          reader.load_dequantized_tensor(sample_keys[0], "bfloat16", use_cache=False)

    # Context exit must close all tracked open context managers / streams.
    self.assertTrue(all(s.closed for s in created_streams))
    self.assertEqual(len(reader._open_handles), 0)  # pylint: disable=protected-access

    for k, got_arr in zip(sample_keys, (got_first, got_second, got_third, got_fourth), strict=True):
      ok, n_bad = deepseek_v41_mm_audit._bitwise_equal(got_arr, baseline_arrays[k])  # pylint: disable=protected-access
      self.assertTrue(ok, msg=f"Chunked/reopened read mismatch on {k}: {n_bad} elements")

  def test_shard_reader_partial_stream_reads_stale_oserror_reopen_and_context_close(self) -> None:
    """Verifies partial-chunk read looping, per-thread handle reuse, `OSError` reopen, and `close()` for Posix and GCS."""
    for label, use_gen_cm in (("posix_direct_stream", False), ("gcs_generator_contextmanager", True)):
      with self.subTest(open_mode=label):
        self._verify_shard_reader_stream_lifecycle(use_generator_cm=use_gen_cm)

  def test_build_expected_shard_slice_parallel_pool_matches_direct_without_caching(self) -> None:
    """Verifies `_build_expected_shard_slice(..., pool=pool)` slices trailing axes before stacking with 0 cache entries."""
    cfg_sc = mm_testing.mini_vision_maxtext_config(
        scan_layers=True,
        param_scan_axis=1,
        weight_dtype="bfloat16",
        tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        ds41_engram_compressed_vocab_size=0,
    )
    rules_sc = {r.maxtext_path: r for r in deepseek_v41_mm_audit.build_independent_audit_rules(cfg_sc)}
    exp_rule = next(r for r in rules_sc.values() if r.kind == "scanned_expert_stack" and len(r.hf_keys) >= 2)
    single_sc_rule = next(
        r
        for r in rules_sc.values()
        if r.kind == "scanned_single" and r.transform == "transpose_2d" and len(r.hf_keys) >= 2
    )
    bias_sc_rule = next(
        r for r in rules_sc.values() if r.kind == "scanned_single" and r.maxtext_path.endswith("/gate/bias_vl")
    )

    with (
        deepseek_v41_mm_audit._SafetensorsShardReader(self.hf_ckpt_dir) as reader,  # pylint: disable=protected-access
        concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,
    ):
      num_reps = len(exp_rule.hf_keys)
      num_exps = len(exp_rule.hf_keys[0])
      global_exp_shape = (num_exps, num_reps, self.ref_args.dim, self.ref_args.moe_inter_dim)
      shard_idx = (slice(0, num_exps), slice(0, num_reps), slice(16, 48), slice(8, 24))
      got_pool = deepseek_v41_mm_audit._build_expected_shard_slice(  # pylint: disable=protected-access
          rule=exp_rule,
          global_shape=global_exp_shape,
          shard_index=shard_idx,
          scan_axis=1,
          reader=reader,
          config=cfg_sc,
          pool=pool,
      )
      got_serial = deepseek_v41_mm_audit._build_expected_shard_slice(  # pylint: disable=protected-access
          rule=exp_rule,
          global_shape=global_exp_shape,
          shard_index=shard_idx,
          scan_axis=1,
          reader=reader,
          config=cfg_sc,
          pool=None,
      )
      ok_exp, _ = deepseek_v41_mm_audit._bitwise_equal(got_pool, got_serial)  # pylint: disable=protected-access
      self.assertTrue(ok_exp)
      self.assertEqual(got_pool.shape, (num_exps, num_reps, 32, 16))
      self.assertEqual(len(reader._tensor_cache), 0)  # pylint: disable=protected-access

      sample_plane = deepseek_v41_mm_audit._transform_single_hf_tensor(  # pylint: disable=protected-access
          reader.load_dequantized_tensor(
              str(single_sc_rule.hf_keys[0]), single_sc_rule.expected_mt_dtype, use_cache=False
          ),
          single_sc_rule.transform,
          cfg_sc,
      )
      n_layers_sc = len(single_sc_rule.hf_keys)
      global_sc_shape = (sample_plane.shape[0], n_layers_sc, sample_plane.shape[1])
      sc_idx = (slice(4, 20), slice(0, n_layers_sc), slice(2, 14))
      got_sc_pool = deepseek_v41_mm_audit._build_expected_shard_slice(  # pylint: disable=protected-access
          rule=single_sc_rule,
          global_shape=global_sc_shape,
          shard_index=sc_idx,
          scan_axis=1,
          reader=reader,
          config=cfg_sc,
          pool=pool,
      )
      got_sc_serial = deepseek_v41_mm_audit._build_expected_shard_slice(  # pylint: disable=protected-access
          rule=single_sc_rule,
          global_shape=global_sc_shape,
          shard_index=sc_idx,
          scan_axis=1,
          reader=reader,
          config=cfg_sc,
          pool=None,
      )
      ok_sc, _ = deepseek_v41_mm_audit._bitwise_equal(got_sc_pool, got_sc_serial)  # pylint: disable=protected-access
      self.assertTrue(ok_sc)
      self.assertEqual(got_sc_pool.shape, (16, n_layers_sc, 12))

      n_bias_layers = len(bias_sc_rule.hf_keys)
      got_bias = deepseek_v41_mm_audit._build_expected_shard_slice(  # pylint: disable=protected-access
          rule=bias_sc_rule,
          global_shape=(n_bias_layers, num_exps),
          shard_index=(slice(0, n_bias_layers), slice(1, 3)),
          scan_axis=1,
          reader=reader,
          config=cfg_sc,
          pool=pool,
      )
      self.assertEqual(got_bias.shape, (n_bias_layers, 2))
      self.assertEqual(got_bias.dtype, np.float32)
      self.assertEqual(len(reader._tensor_cache), 0)  # pylint: disable=protected-access

  def test_safeshard_reader_torch_free_float8_and_scaled_importerror(self):
    """Verifies explicit ImportError on scaled tensors without PyTorch and float8 non-torch dtype entries."""
    with tempfile.TemporaryDirectory(prefix="ds41_audit_notorch_") as tmp_dir:
      ckpt_dir = Path(tmp_dir)
      raw = np.zeros((4, 4), dtype=np.int8)
      scale = np.ones((4, 1), dtype=np.float32)
      fp8_raw = np.ones((2, 2), dtype=deepseek_v41_mm_audit.ml_dtypes.float8_e4m3fn)
      # Write a minimal safetensors shard using numpy bytes (no PyTorch dependency).
      hdr_obj = {
          "w.weight": {"dtype": "I8", "shape": [4, 4], "data_offsets": [0, 16]},
          "w.scale": {"dtype": "F32", "shape": [4, 1], "data_offsets": [16, 32]},
          "fp8.weight": {"dtype": "F8_E4M3", "shape": [2, 2], "data_offsets": [32, 36]},
      }
      hdr_bytes = json.dumps(hdr_obj).encode("utf-8")
      payload = len(hdr_bytes).to_bytes(8, "little") + hdr_bytes + raw.tobytes() + scale.tobytes() + fp8_raw.tobytes()
      (ckpt_dir / "model.safetensors").write_bytes(payload)
      idx = {
          "weight_map": {
              "w.weight": "model.safetensors",
              "w.scale": "model.safetensors",
              "fp8.weight": "model.safetensors",
          }
      }
      (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(idx), encoding="utf-8")

      with (
          mock.patch.object(deepseek_v41_mm_audit, "HAS_TORCH", False),
          mock.patch.object(deepseek_v41_mm_audit, "torch", None),
          mock.patch.object(deepseek_v41_mm_audit, "ref_runner", None),
          mock.patch.dict(
              deepseek_v41_mm_audit._SAFETENSORS_DTYPES,  # pylint: disable=protected-access
              {
                  "I8": (np.int8, 1),
                  "F32": (np.float32, 4),
                  "F8_E4M3": (deepseek_v41_mm_audit.ml_dtypes.float8_e4m3fn, 1),
              },
              clear=True,
          ),
      ):
        reader = deepseek_v41_mm_audit._SafetensorsShardReader(ckpt_dir)  # pylint: disable=protected-access
        try:
          fp8_arr = reader.load_dequantized_tensor("fp8.weight", "bfloat16")
          self.assertEqual(fp8_arr.shape, (2, 2))
          with self.assertRaisesRegex(ImportError, "PyTorch is required to dequantize"):
            reader.load_dequantized_tensor("w.weight", "bfloat16")
          with self.assertRaisesRegex(ImportError, "PyTorch is required to dequantize"):
            reader.load_dequantized_row_slice("w.weight", "bfloat16", 0, 2)
        finally:
          reader.close()

  def test_frozen_topk_and_natural_trace_preserve_moe_bias_var_sharding_and_audit(self) -> None:
    """Verifies trace/frozen passes preserve MoEBiasVar sharding and audit accepts equivalent P() vs P(None, None)."""
    cfg, mesh, model = self._build_trailing_sharded_model(scan_layers=True, param_scan_axis=0)
    seq_len = int(cfg.max_target_length)
    n_vit_h, n_vit_w = cfg.ds41_vision_grid
    patch_dim = int(cfg.num_channels_for_vit) * (int(cfg.patch_size_for_vit) ** 2)

    def _collect_2d_bias_leaves() -> dict[str, jax.Array]:
      _, bias_state, _ = nnx.split(model, moe.MoEBiasVar, ...)
      all_bias = deepseek_v41_mm_audit.flatten_state_arrays(nnx.to_pure_dict(bias_state))
      return {k: v for k, v in all_bias.items() if v.ndim == 2}

    initial_2d_biases = _collect_2d_bias_leaves()
    self.assertTrue(any(k.endswith("/gate/bias") for k in initial_2d_biases))
    self.assertTrue(any(k.endswith("/gate/bias_vl") for k in initial_2d_biases))
    expected_2d_spec = jax.sharding.PartitionSpec(None, None)
    for path_b, arr_b in initial_2d_biases.items():
      self.assertIsInstance(arr_b.sharding, jax.sharding.NamedSharding, msg=path_b)
      self.assertEqual(arr_b.sharding.spec, expected_2d_spec, msg=path_b)

    ids = np.arange(seq_len, dtype=np.int32)[None, :] % int(cfg.vocab_size)
    pos = np.arange(seq_len, dtype=np.int32)[None, :]
    seg = np.ones((1, seq_len), dtype=np.int32)
    imgs = np.zeros((1, 1, int(n_vit_h) * int(n_vit_w), patch_dim), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.int32)

    deepseek_v41_frozen_topk.clear_frozen_topk_step_caches()
    nat_res = deepseek_v41_frozen_topk.run_natural_routing_forward_with_trace_nnx(
        model,
        cfg,
        mesh,
        sample_name="audit_bias_sharding_nat",
        ids=ids,
        decoder_positions=pos,
        decoder_segment_ids=seg,
        images=imgs,
        image_masks=masks,
    )
    for path_b, arr_b in _collect_2d_bias_leaves().items():
      self.assertIsInstance(arr_b.sharding, jax.sharding.NamedSharding, msg=f"Post-natural {path_b}")
      self.assertEqual(arr_b.sharding.spec, expected_2d_spec, msg=f"Post-natural {path_b}")

    res_post_nat = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
        model=model,
        config=cfg,
        hf_checkpoint_dir=self.hf_ckpt_dir,
        mesh=mesh,
        verify_all_text_leaves=True,
    )
    self.assertTrue(res_post_nat["passed"], msg=f"Post-natural audit failed: {res_post_nat}")
    self.assertEqual(res_post_nat["sharding_mismatches"], [])

    sorted_idx_layers = sorted(nat_res.tpu_indexer_topk.keys())
    raw_idx_layer_ids = np.asarray(sorted_idx_layers, dtype=np.int32)
    raw_idx_0based = np.stack(
        [np.asarray(nat_res.tpu_indexer_topk[lid][0, :seq_len, :], dtype=np.int32) for lid in sorted_idx_layers],
        axis=0,
    )
    raw_idx_offset = np.where(raw_idx_0based >= 0, raw_idx_0based + seq_len, -1).astype(np.int32)
    deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx(
        model,
        cfg,
        mesh,
        np.asarray(nat_res.tpu_moe_topk[:, 0, :seq_len, :], dtype=np.int32),
        sample_name="audit_bias_sharding_frz",
        ids=ids,
        decoder_positions=pos,
        decoder_segment_ids=seg,
        images=imgs,
        image_masks=masks,
        seq_len=seq_len,
        raw_indexer_topk=raw_idx_offset,
        raw_indexer_layer_ids=raw_idx_layer_ids,
    )
    for path_b, arr_b in _collect_2d_bias_leaves().items():
      self.assertIsInstance(arr_b.sharding, jax.sharding.NamedSharding, msg=f"Post-frozen {path_b}")
      self.assertEqual(arr_b.sharding.spec, expected_2d_spec, msg=f"Post-frozen {path_b}")

    res_post_frz = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
        model=model,
        config=cfg,
        hf_checkpoint_dir=self.hf_ckpt_dir,
        mesh=mesh,
        verify_all_text_leaves=True,
    )
    self.assertTrue(res_post_frz["passed"], msg=f"Post-frozen audit failed: {res_post_frz}")
    self.assertEqual(res_post_frz["sharding_mismatches"], [])

    # Verify audit accepts equivalent PartitionSpec() on a 2D MoEBiasVar whose abstract spec is PartitionSpec(None, None),
    # while rejecting non-equivalent PartitionSpec("fsdp", None) on the 4-device (fsdp=2, tensor=2) mesh.
    target_bias_path = next(
        p for p, v in sorted(initial_2d_biases.items()) if p.endswith("/gate/bias") and v.shape[0] >= 2
    )
    equiv_arr = jax.device_put(
        initial_2d_biases[target_bias_path],
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    orig_arr = _replace_model_leaf(model, target_bias_path, equiv_arr)
    try:
      self.assertEqual(_collect_2d_bias_leaves()[target_bias_path].sharding.spec, jax.sharding.PartitionSpec())
      res_equiv = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
          model=model,
          config=cfg,
          hf_checkpoint_dir=self.hf_ckpt_dir,
          mesh=mesh,
          verify_all_text_leaves=True,
      )
      self.assertTrue(res_equiv["passed"], msg=f"Equivalent P() vs P(None, None) audit failed: {res_equiv}")
      self.assertEqual(res_equiv["sharding_mismatches"], [])

      bad_arr = jax.device_put(
          orig_arr,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", None)),
      )
      _replace_model_leaf(model, target_bias_path, bad_arr)
      res_bad = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
          model=model,
          config=cfg,
          hf_checkpoint_dir=self.hf_ckpt_dir,
          mesh=mesh,
          verify_all_text_leaves=True,
      )
      self.assertFalse(res_bad["passed"])
      self.assertTrue(any(target_bias_path in msg for msg in res_bad["sharding_mismatches"]))
    finally:
      _replace_model_leaf(model, target_bias_path, orig_arr)


if __name__ == "__main__":
  unittest.main()
