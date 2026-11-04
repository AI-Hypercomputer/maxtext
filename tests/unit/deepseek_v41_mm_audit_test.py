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
  """Replaces `target_path` inside `model` and returns the original `jax.Array` leaf."""
  _, p_state, b_state, rest_state = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
  p_pure = nnx.to_pure_dict(p_state)
  parts = target_path.split("/")
  cursor: Any = p_pure
  for part in parts[:-1]:
    cursor = cursor[part]
  orig_leaf = cursor[parts[-1]]
  cursor[parts[-1]] = new_leaf
  nnx.replace_by_pure_dict(p_state, p_pure)
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
    cls.tmp_root = Path(tempfile.mkdtemp(prefix="ds41_mm_audit_unit_"))
    cls.tokenizer = transformers.AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)
    if HAS_TORCH and multimodal is not None:
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
      cls.ref_args = types.SimpleNamespace(dim=16, moe_inter_dim=16, n_routed_experts=4)
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
    cfg_sc = mm_testing.mini_vision_maxtext_config(scan_layers=True, param_scan_axis=1, weight_dtype="bfloat16")
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


def _make_40layer_r4_fixture(
    root_dir: Path,
    *,
    num_steps: int = 20,
    peak_hbm_gb: float = 76.25,
    bytes_in_use_gib: float = 72.5,
) -> tuple[Path, Path, Path, Path, dict[int, np.ndarray], dict[str, np.ndarray]]:
  """Creates a 40-layer float32[384] HF safetensors fixture, 20-step R4 log/metrics, and clean `bias_vl` npz."""
  hf_dir = root_dir / "hf_40layer_bias_vl"
  hf_dir.mkdir(parents=True, exist_ok=True)
  hf_tensors_np: dict[int, np.ndarray] = {}
  hf_tensors_save: dict[str, Any] = {}
  bvl_hex_list: list[str] = []
  for layer_idx in range(40):
    arr = np.linspace(0.1 + layer_idx * 0.01, -0.1 - layer_idx * 0.01, 384, dtype=np.float32)
    hf_tensors_np[layer_idx] = arr
    if HAS_TORCH:
      hf_tensors_save[f"layers.{layer_idx}.ffn.gate.bias_vl"] = torch.from_numpy(arr.copy())
    else:
      hf_tensors_save[f"layers.{layer_idx}.ffn.gate.bias_vl"] = arr.copy()
    # Perturb the logged scalar norm by 1 ULP from CPU sqrt(sum(arr^2)) while keeping it constant across steps 0..19
    # to prove TT6 Option 1 gates on bit-exact 40x384 tensors rather than cross-platform scalar L2 norm reduction.
    l2_val = np.float32(np.sqrt(np.sum(np.square(arr), dtype=np.float32), dtype=np.float32))
    l2_u32 = np.array([l2_val], dtype=np.float32).view(np.uint32)
    l2_u32[0] ^= np.uint32(1)
    bvl_hex_list.append(float(l2_u32.view(np.float32)[0]).hex())
  _save_two_shard_safetensors_dir(hf_dir, hf_tensors_save)

  vit_hex = float(np.float32(12.5)).hex()
  delim_hex = float(np.float32(1.75)).hex()
  bvl_joined = ",".join(bvl_hex_list)
  log_lines: list[str] = []
  metric_records: list[dict[str, Any]] = []
  for step_idx in range(num_steps):
    loss_val = 12.5 - 0.30 * step_idx
    lb_val = 0.00015
    align_hex = float(np.float32(4.0 + 0.02 * step_idx)).hex()
    delim_hex = float(np.float32(1.75 + 0.015 * step_idx)).hex()
    log_lines.append(
        f"completed step: {step_idx}, seconds: 1.0, TFLOP/s/device: 100.0, loss: {loss_val:.6f}, moe_lb_loss: {lb_val:.6f}"
    )
    log_lines.append(
        f"DS41_MM_NORMS step={step_idx} vit_norm={vit_hex} aligner_norm={align_hex} "
        f"delim_norm={delim_hex} bias_vl_norm=[{bvl_joined}] "
        f"peak_hbm_gb={peak_hbm_gb:.6f} live_peak_hbm_gib={peak_hbm_gb:.6f} "
        f"bytes_in_use_gib={bytes_in_use_gib:.6f}"
    )
    metric_records.append({"step": step_idx, "learning/raw_grad_norm": 2.5 + 0.05 * step_idx})

  log_path = root_dir / "r4_train.log"
  metrics_path = root_dir / "r4_metrics.jsonl"
  log_path.write_text("\n".join(log_lines), encoding="utf-8")
  metrics_path.write_text("\n".join(json.dumps(r) for r in metric_records), encoding="utf-8")

  last_step = num_steps - 1
  npz_payload: dict[str, np.ndarray] = {
      "steps": np.asarray([0, last_step], dtype=np.int32),
      "layer_order": np.arange(40, dtype=np.int32),
  }
  for layer_idx in range(40):
    npz_payload[f"step_0_layer_{layer_idx}"] = hf_tensors_np[layer_idx].copy()
    npz_payload[f"step_{last_step}_layer_{layer_idx}"] = hf_tensors_np[layer_idx].copy()
  npz_path = root_dir / "r4_bias_vl_steps.npz"
  np.savez(npz_path, **npz_payload)
  return hf_dir, log_path, metrics_path, npz_path, hf_tensors_np, npz_payload


@pytest.mark.cpu_only
class DeepSeekV41MmR4Tt6Option1Test(unittest.TestCase):
  """Unit tests and negative controls for User Ruling TT6 = Option 1 (`uint32` bit-exact 40x384 `bias_vl`)."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.tmp_root = Path(tempfile.mkdtemp(prefix="ds41_tt6_opt1_"))
    cls.thresholds_path = Path(__file__).resolve().parents[2] / "tests/end_to_end/tpu/deepseek/v4.1-flash/thresholds.json"
    (
        cls.hf_dir,
        cls.log_path,
        cls.metrics_path,
        cls.clean_npz_path,
        cls.hf_tensors_np,
        cls.clean_npz_payload,
    ) = _make_40layer_r4_fixture(cls.tmp_root)

  @classmethod
  def tearDownClass(cls) -> None:
    shutil.rmtree(cls.tmp_root, ignore_errors=True)

  def test_1_clean_pass_40_layers_bit_exact_step0_and_step19(self) -> None:
    """1. Clean PASS: all 40 layers match HF bit-for-bit at step 0 and step 19."""
    loaded_hf = deepseek_v41_mm_audit.load_hf_bias_vl_tensors(self.hf_dir)
    self.assertEqual(sorted(loaded_hf.keys()), list(range(40)))
    for i in range(40):
      self.assertEqual(loaded_hf[i].dtype, np.float32)
      self.assertEqual(loaded_hf[i].shape, (384,))
      self.assertTrue(np.array_equal(loaded_hf[i].view(np.uint32), self.hf_tensors_np[i].view(np.uint32)))

    res = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=self.clean_npz_path,
    )
    self.assertTrue(res["passed"], msg=f"Unexpected failures: {res['failures']}")
    self.assertTrue(res["bias_vl_step0_bit_exact_hf"])
    self.assertTrue(res["bias_vl_last_step_bit_exact_hf"])
    table = res["bias_vl_bit_equality_table"]
    self.assertEqual(len(table), 40)
    for idx, row in enumerate(table):
      self.assertEqual(row["layer_idx"], idx)
      self.assertEqual(row["shape"], [384])
      self.assertEqual(row["dtype"], "float32")
      self.assertEqual(len(row["hf_sha256_16"]), 16)
      self.assertEqual(row["step0_sha256_16"], row["hf_sha256_16"])
      self.assertEqual(row["last_step_sha256_16"], row["hf_sha256_16"])
      self.assertTrue(row["step0_uint32_equal"])
      self.assertTrue(row["last_step_uint32_equal"])
      self.assertEqual(row["step0_diff_elements"], 0)
      self.assertEqual(row["last_step_diff_elements"], 0)
      self.assertEqual(row["step0_max_ulp_diff"], 0)
      self.assertEqual(row["last_step_max_ulp_diff"], 0)

  def test_2_negative_control_flip_1_ulp_layer13_step0(self) -> None:
    """2. Negative Control 1: flip 1 ULP in element 17 of layer 13 at step 0 -> MUST FAIL."""
    bad_payload = {k: v.copy() for k, v in self.clean_npz_payload.items()}
    bad_payload["step_0_layer_13"].view(np.uint32)[17] ^= np.uint32(1)
    bad_npz = self.tmp_root / "nc1_step0_layer13_1ulp.npz"
    np.savez(bad_npz, **bad_payload)

    res = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=bad_npz,
    )
    self.assertFalse(res["passed"])
    self.assertFalse(res["bias_vl_step0_bit_exact_hf"])
    self.assertTrue(res["bias_vl_last_step_bit_exact_hf"])
    row13 = res["bias_vl_bit_equality_table"][13]
    self.assertFalse(row13["step0_uint32_equal"])
    self.assertTrue(row13["last_step_uint32_equal"])
    self.assertEqual(row13["step0_diff_elements"], 1)
    self.assertEqual(row13["step0_max_ulp_diff"], 1)
    self.assertTrue(any("step 0 layer 13 bias_vl is not bit-exact" in f for f in res["failures"]))

  def test_3_negative_control_flip_1_ulp_layer39_step19(self) -> None:
    """3. Negative Control 2: flip 1 ULP in element 300 of layer 39 at step 19 -> MUST FAIL."""
    bad_payload = {k: v.copy() for k, v in self.clean_npz_payload.items()}
    bad_payload["step_19_layer_39"].view(np.uint32)[300] ^= np.uint32(1)
    bad_npz = self.tmp_root / "nc2_step19_layer39_1ulp.npz"
    np.savez(bad_npz, **bad_payload)

    res = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=bad_npz,
    )
    self.assertFalse(res["passed"])
    self.assertTrue(res["bias_vl_step0_bit_exact_hf"])
    self.assertFalse(res["bias_vl_last_step_bit_exact_hf"])
    row39 = res["bias_vl_bit_equality_table"][39]
    self.assertTrue(row39["step0_uint32_equal"])
    self.assertFalse(row39["last_step_uint32_equal"])
    self.assertEqual(row39["last_step_diff_elements"], 1)
    self.assertEqual(row39["last_step_max_ulp_diff"], 1)
    self.assertTrue(any("step 19 layer 39 bias_vl is not bit-exact" in f for f in res["failures"]))

  def test_4_negative_control_wrong_layer_order_and_swapped_layers(self) -> None:
    """4. Negative Control 3: swapped layers 5 and 6 or corrupted `layer_order` -> MUST FAIL."""
    swapped_payload = {k: v.copy() for k, v in self.clean_npz_payload.items()}
    swapped_payload["step_0_layer_5"], swapped_payload["step_0_layer_6"] = (
        swapped_payload["step_0_layer_6"].copy(),
        swapped_payload["step_0_layer_5"].copy(),
    )
    swapped_npz = self.tmp_root / "nc3_swapped_5_6.npz"
    np.savez(swapped_npz, **swapped_payload)

    res_swap = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=swapped_npz,
    )
    self.assertFalse(res_swap["passed"])
    self.assertFalse(res_swap["bias_vl_step0_bit_exact_hf"])

    bad_order_payload = {k: v.copy() for k, v in self.clean_npz_payload.items()}
    bad_order_payload["layer_order"][5], bad_order_payload["layer_order"][6] = 6, 5
    bad_order_npz = self.tmp_root / "nc3_bad_layer_order.npz"
    np.savez(bad_order_npz, **bad_order_payload)

    res_order = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=bad_order_npz,
    )
    self.assertFalse(res_order["passed"])
    self.assertFalse(res_order["bias_vl_step0_bit_exact_hf"])
    self.assertFalse(res_order["bias_vl_last_step_bit_exact_hf"])
    self.assertTrue(any("layer_order" in f for f in res_order["failures"]))

  def test_5_negative_control_missing_layer_or_wrong_shape_dtype(self) -> None:
    """5. Negative Control 4: missing npz, missing layer 39, float16 dtype, or shape (383,) -> MUST FAIL."""
    res_missing_file = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=self.tmp_root / "does_not_exist.npz",
    )
    self.assertFalse(res_missing_file["passed"])
    self.assertFalse(res_missing_file["bias_vl_step0_bit_exact_hf"])
    self.assertFalse(res_missing_file["bias_vl_last_step_bit_exact_hf"])

    missing_l39_payload = {k: v.copy() for k, v in self.clean_npz_payload.items() if k != "step_19_layer_39"}
    missing_l39_npz = self.tmp_root / "nc4_missing_l39.npz"
    np.savez(missing_l39_npz, **missing_l39_payload)
    res_missing_l39 = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=missing_l39_npz,
    )
    self.assertFalse(res_missing_l39["passed"])
    self.assertFalse(res_missing_l39["bias_vl_last_step_bit_exact_hf"])

    bad_dtype_payload = {k: v.copy() for k, v in self.clean_npz_payload.items()}
    bad_dtype_payload["step_0_layer_0"] = bad_dtype_payload["step_0_layer_0"].astype(np.float16)
    bad_dtype_npz = self.tmp_root / "nc4_float16.npz"
    np.savez(bad_dtype_npz, **bad_dtype_payload)
    res_dtype = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=bad_dtype_npz,
    )
    self.assertFalse(res_dtype["passed"])
    self.assertFalse(res_dtype["bias_vl_step0_bit_exact_hf"])
    self.assertEqual(res_dtype["bias_vl_bit_equality_table"][0]["dtype"], "float16")
    self.assertEqual(res_dtype["bias_vl_bit_equality_table"][0]["step0_dtype"], "float16")
    self.assertEqual(res_dtype["bias_vl_bit_equality_table"][0]["last_step_dtype"], "float32")

    bad_shape_payload = {k: v.copy() for k, v in self.clean_npz_payload.items()}
    bad_shape_payload["step_19_layer_7"] = bad_shape_payload["step_19_layer_7"][:383]
    bad_shape_npz = self.tmp_root / "nc4_shape383.npz"
    np.savez(bad_shape_npz, **bad_shape_payload)
    res_shape = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=bad_shape_npz,
    )
    self.assertFalse(res_shape["passed"])
    self.assertFalse(res_shape["bias_vl_last_step_bit_exact_hf"])
    self.assertEqual(res_shape["bias_vl_bit_equality_table"][7]["shape"], [383])
    self.assertEqual(res_shape["bias_vl_bit_equality_table"][7]["step0_shape"], [384])
    self.assertEqual(res_shape["bias_vl_bit_equality_table"][7]["last_step_shape"], [383])

  def test_6_extract_model_bias_vl_tensors_scanned_unscanned_and_wrapper(self) -> None:
    """6. Extraction helper test on 40-layer scanned (`segments_0..`) and unscanned (`layers_0..39`) structures."""
    base_cfg_attrs = {
        "num_decoder_layers": 40,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
        "kv_source_layers": [2, 8, 14, 20],
        "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer": 20,
        "ds41_engram_layers": [1, 14],
    }
    cfg_unscanned = types.SimpleNamespace(**base_cfg_attrs, scan_layers=False)
    unscanned_state = {
        "decoder": {
            f"layers_{i}": {
                "mlp": {"MoeBlock_0": {"gate": {"bias_vl": moe.MoEBiasVar(jnp.asarray(self.hf_tensors_np[i]))}}}
            }
            for i in range(40)
        }
    }
    got_unscanned = deepseek_v41_mm_audit.extract_model_bias_vl_tensors(cfg_unscanned, unscanned_state)
    self.assertEqual(sorted(got_unscanned.keys()), list(range(40)))
    for i in range(40):
      self.assertTrue(np.array_equal(got_unscanned[i].view(np.uint32), self.hf_tensors_np[i].view(np.uint32)))

    cfg_scanned = types.SimpleNamespace(**base_cfg_attrs, scan_layers=True)
    segments = deepseek41.build_segments(cfg_scanned)
    scanned_dec: dict[str, Any] = {}
    for seg_idx, segment in enumerate(segments):
      seg_dict: dict[str, Any] = {}
      for slot_idx in range(len(segment.groups[0])):
        stacked = np.stack([self.hf_tensors_np[group[slot_idx]] for group in segment.groups], axis=0)
        seg_dict[f"layers_{slot_idx}"] = {
            "mlp": {"MoeBlock_0": {"gate": {"bias_vl": moe.MoEBiasVar(jnp.asarray(stacked))}}}
        }
      scanned_dec[f"segments_{seg_idx}"] = seg_dict

    got_scanned = deepseek_v41_mm_audit.extract_model_bias_vl_tensors(cfg_scanned, {"decoder": scanned_dec})
    self.assertEqual(sorted(got_scanned.keys()), list(range(40)))
    for i in range(40):
      self.assertTrue(np.array_equal(got_scanned[i].view(np.uint32), self.hf_tensors_np[i].view(np.uint32)))

    # Also verify run_train_with_bias_vl_dump wrapping maxtext_train.training_loop_iteration across 20 steps.
    from maxtext.trainers.pre_train import train as maxtext_train  # pylint: disable=import-outside-toplevel

    fake_model = mock.MagicMock()
    dumped_npz = self.tmp_root / "wrapped_r4_dump.npz"

    def _fake_iter(
        jax_device_state: dict[str, Any], python_vars: dict[str, Any], immutable_data: dict[str, Any]
    ) -> dict[str, Any]:
      del jax_device_state, python_vars, immutable_data
      return {}

    def _fake_main(argv: list[str]) -> None:
      self.assertGreaterEqual(len(argv), 2)
      jax_device_state = {"state": types.SimpleNamespace(model=fake_model)}
      python_vars = {"step": 0}
      immutable_data = {"steps": 20, "config": cfg_scanned}
      while python_vars["step"] < immutable_data["steps"]:
        maxtext_train.training_loop_iteration(jax_device_state, python_vars, immutable_data)
        python_vars["step"] += 1

    with (
        mock.patch.object(nnx, "state", return_value={"decoder": scanned_dec}),
        mock.patch.object(maxtext_train, "training_loop_iteration", side_effect=_fake_iter),
        mock.patch.object(maxtext_train, "main", side_effect=_fake_main),
    ):
      deepseek_v41_mm_audit._cli_main(  # pylint: disable=protected-access
          ["--mode=run_r4_train", f"--bias_vl_npz={dumped_npz}", "--", "train.py", "base.yml"]
      )

    res_wrapped = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=dumped_npz,
    )
    self.assertTrue(res_wrapped["passed"], msg=f"Failures: {res_wrapped['failures']}")

    # Non-zero worker process (e.g. jax.process_index() == 3) also writes local .npz so all 16 TPU pods pass verify_r4_gates.
    dumped_worker3_npz = self.tmp_root / "wrapped_r4_dump_worker3.npz"
    with (
        mock.patch.object(jax, "process_index", return_value=3),
        mock.patch.object(nnx, "state", return_value={"decoder": scanned_dec}),
        mock.patch.object(maxtext_train, "training_loop_iteration", side_effect=_fake_iter),
        mock.patch.object(maxtext_train, "main", side_effect=_fake_main),
    ):
      deepseek_v41_mm_audit._cli_main(  # pylint: disable=protected-access
          ["--mode=run_r4_train", f"--bias_vl_npz={dumped_worker3_npz}", "--", "train.py", "base.yml"]
      )
    self.assertTrue(dumped_worker3_npz.exists())
    res_worker3 = deepseek_v41_mm_audit.verify_r4_gates(
        self.log_path,
        self.metrics_path,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=20,
        bias_vl_npz=dumped_worker3_npz,
    )
    self.assertTrue(res_worker3["passed"], msg=f"Failures: {res_worker3['failures']}")

  def test_7_run_train_with_bias_vl_dump_emits_info_logs_and_passes_gate_with_negative_control(self) -> None:
    """7. CPU proof: fresh subprocess hook emits `max_logging.log` INFO lines, passes gate, and fails when stripped."""
    import subprocess  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    repo_root = Path(__file__).resolve().parents[2]
    hook_dir = self.tmp_root / "r4_hook_info_proof"
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook_log = hook_dir / "r4_train.log"
    hook_metrics = hook_dir / "r4_metrics.jsonl"
    hook_npz = hook_dir / "r4_bias_vl_steps.npz"

    child_script = f"""
import json
import logging as std_logging
from pathlib import Path
import types
from unittest import mock

from absl import logging as absl_logging
from flax import nnx
import jax.numpy as jnp
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.trainers.pre_train import train as maxtext_train
from maxtext.utils import max_logging
import numpy as np

from tests.utils import deepseek_v41_mm_audit

# Force default fresh-process WARNING level before calling run_train_with_bias_vl_dump:
std_logging.getLogger().setLevel(std_logging.WARNING)
absl_logging.set_verbosity(absl_logging.WARNING)

hf_tensors = deepseek_v41_mm_audit.load_hf_bias_vl_tensors({str(self.hf_dir)!r})
base_cfg_attrs = {{
    "num_decoder_layers": 40,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
    "kv_source_layers": [2, 8, 14, 20],
    "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer": 20,
    "ds41_engram_layers": [1, 14],
    "scan_layers": True,
}}
cfg_scanned = types.SimpleNamespace(**base_cfg_attrs)
segments = deepseek41.build_segments(cfg_scanned)
scanned_dec = {{}}
for seg_idx, segment in enumerate(segments):
  seg_dict = {{}}
  for slot_idx in range(len(segment.groups[0])):
    stacked = np.stack([hf_tensors[group[slot_idx]] for group in segment.groups], axis=0)
    seg_dict[f"layers_{{slot_idx}}"] = {{
        "mlp": {{"MoeBlock_0": {{"gate": {{"bias_vl": moe.MoEBiasVar(jnp.asarray(stacked))}}}}}}
    }}
  scanned_dec[f"segments_{{seg_idx}}"] = seg_dict

vit_hex = float(np.float32(12.5)).hex()
delim_hex = float(np.float32(1.75)).hex()
bvl_hex = ",".join(
    float(np.float32(np.sqrt(np.sum(np.square(hf_tensors[i]), dtype=np.float32), dtype=np.float32))).hex()
    for i in range(40)
)
records = []

def _fake_iter(jax_device_state, python_vars, immutable_data, *args, **kwargs):
  del jax_device_state, immutable_data, args
  assert kwargs.get("elastic_retry_cache") == {{"dummy": 1}}
  step_idx = int(python_vars["step"])
  loss_val = 12.5 - 0.4 * step_idx
  align_hex = float(np.float32(4.0 + 0.05 * step_idx)).hex()
  delim_hex = float(np.float32(1.75 + 0.02 * step_idx)).hex()
  # Use max_logging.log (logging.info), NEVER print(), to prove INFO logging is active:
  max_logging.log(
      f"completed step: {{step_idx}}, seconds: 1.000, TFLOP/s/device: 100.000, "
      f"loss: {{loss_val:.6f}}, moe_lb_loss: 0.000150"
  )
  max_logging.log(
      f"DS41_MM_NORMS step={{step_idx}} vit_norm={{vit_hex}} aligner_norm={{align_hex}} "
      f"delim_norm={{delim_hex}} bias_vl_norm=[{{bvl_hex}}] "
      "peak_hbm_gb=76.250000 live_peak_hbm_gib=76.250000 bytes_in_use_gib=72.500000"
  )
  records.append({{"step": step_idx, "learning/raw_grad_norm": 2.5 + 0.1 * step_idx}})
  return {{}}

def _fake_main(argv):
  assert len(argv) >= 2
  jax_device_state = {{"state": types.SimpleNamespace(model=mock.MagicMock())}}
  python_vars = {{"step": 0}}
  immutable_data = {{"steps": 2, "config": cfg_scanned}}
  while python_vars["step"] < immutable_data["steps"]:
    maxtext_train.training_loop_iteration(
        jax_device_state,
        python_vars,
        immutable_data,
        elastic_retry_cache={{"dummy": 1}},
    )
    python_vars["step"] += 1
  Path({str(hook_metrics)!r}).write_text(chr(10).join(json.dumps(r) for r in records), encoding="utf-8")

with (
    mock.patch.object(nnx, "state", return_value={{"decoder": scanned_dec}}),
    mock.patch.object(maxtext_train, "training_loop_iteration", side_effect=_fake_iter),
    mock.patch.object(maxtext_train, "main", side_effect=_fake_main),
):
  deepseek_v41_mm_audit._cli_main(
      ["--mode=run_r4_train", f"--bias_vl_npz={hook_npz}", "--", "train.py", "base.yml"]
  )
"""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = f"{repo_root.as_posix()}:{(repo_root / 'src').as_posix()}"
    proc = subprocess.run(
        [sys.executable, "-c", child_script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr} stdout={proc.stdout}")
    combined_out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    hook_log.write_text(combined_out, encoding="utf-8")

    self.assertIn("completed step: 0,", combined_out)
    self.assertIn("completed step: 1,", combined_out)
    self.assertIn("DS41_MM_NORMS step=0", combined_out)
    self.assertIn("DS41_MM_NORMS step=1", combined_out)
    self.assertTrue(hook_npz.exists())

    res_pass = deepseek_v41_mm_audit.verify_r4_gates(
        hook_log,
        hook_metrics,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=2,
        bias_vl_npz=hook_npz,
    )
    self.assertTrue(res_pass["passed"], msg=f"Unexpected failures: {res_pass['failures']}")
    self.assertTrue(res_pass["bias_vl_step0_bit_exact_hf"])
    self.assertTrue(res_pass["bias_vl_last_step_bit_exact_hf"])
    self.assertEqual(len(res_pass["bias_vl_bit_equality_table"]), 40)

    # Negative Control A: strip `completed step:` lines -> gate MUST fail, while still reporting bias_vl bit-exactness.
    stripped_step_log = hook_dir / "r4_train_no_completed_step.log"
    stripped_step_log.write_text(
        "\n".join(line for line in combined_out.splitlines() if "completed step:" not in line),
        encoding="utf-8",
    )
    res_no_step = deepseek_v41_mm_audit.verify_r4_gates(
        stripped_step_log,
        hook_metrics,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=2,
        bias_vl_npz=hook_npz,
    )
    self.assertFalse(res_no_step["passed"])
    self.assertTrue(any("Missing completed step lines for steps: [0, 1]" in f for f in res_no_step["failures"]))
    self.assertTrue(res_no_step["bias_vl_step0_bit_exact_hf"])
    self.assertTrue(res_no_step["bias_vl_last_step_bit_exact_hf"])
    self.assertEqual(len(res_no_step["bias_vl_bit_equality_table"]), 40)

    # Negative Control B: strip `DS41_MM_NORMS` lines -> gate MUST fail, while still reporting bias_vl bit-exactness.
    stripped_norms_log = hook_dir / "r4_train_no_mm_norms.log"
    stripped_norms_log.write_text(
        "\n".join(line for line in combined_out.splitlines() if "DS41_MM_NORMS" not in line),
        encoding="utf-8",
    )
    res_no_norms = deepseek_v41_mm_audit.verify_r4_gates(
        stripped_norms_log,
        hook_metrics,
        self.thresholds_path,
        hf_checkpoint_dir=self.hf_dir,
        aot_hbm_gb=74.3736,
        expected_steps=2,
        bias_vl_npz=hook_npz,
    )
    self.assertFalse(res_no_norms["passed"])
    self.assertTrue(any("Missing DS41_MM_NORMS lines for steps: [0, 1]" in f for f in res_no_norms["failures"]))
    self.assertTrue(res_no_norms["bias_vl_step0_bit_exact_hf"])
    self.assertTrue(res_no_norms["bias_vl_last_step_bit_exact_hf"])
    self.assertEqual(len(res_no_norms["bias_vl_bit_equality_table"]), 40)

  def test_8_run_train_with_bias_vl_dump_forwards_elastic_retry_cache_and_negative_control(self) -> None:
    """8. CPU proof: run_train_with_bias_vl_dump forwards elastic_retry_cache to orig_iteration."""
    from maxtext.trainers.pre_train import train as maxtext_train  # pylint: disable=import-outside-toplevel

    orig_iter_calls = []

    def _fake_orig_iteration(jax_device_state, python_vars, immutable_data, *args, **kwargs):
      del jax_device_state, python_vars, immutable_data
      orig_iter_calls.append({"args": args, "kwargs": kwargs})
      return {}

    def _fake_train_main(argv):
      del argv
      jax_device_state = {"state": types.SimpleNamespace(model=mock.MagicMock())}
      python_vars = {"step": 0}
      immutable_data = {"steps": 1, "config": types.SimpleNamespace(num_decoder_layers=0, scan_layers=False)}
      # Call training_loop_iteration matching src/maxtext/trainers/pre_train/train.py:1697
      maxtext_train.training_loop_iteration(
          jax_device_state,
          python_vars,
          immutable_data,
          elastic_retry_cache={"dummy": 1},
      )

    bias_vl_npz = self.tmp_root / "test_8_bias_vl.npz"
    with (
        mock.patch.object(nnx, "state", return_value={"decoder": {}}),
        mock.patch.object(maxtext_train, "training_loop_iteration", side_effect=_fake_orig_iteration),
        mock.patch.object(maxtext_train, "main", side_effect=_fake_train_main),
    ):
      deepseek_v41_mm_audit.run_train_with_bias_vl_dump(
          ["train.py", "base.yml"],
          bias_vl_npz,
      )

    self.assertEqual(len(orig_iter_calls), 1)
    self.assertEqual(orig_iter_calls[0]["kwargs"].get("elastic_retry_cache"), {"dummy": 1})

    # Negative control: with old 3-arg signature, passing elastic_retry_cache raises TypeError
    def _old_wrapped_iteration(jax_device_state, python_vars, immutable_data):
      del jax_device_state, python_vars, immutable_data
      return {}

    with mock.patch.object(maxtext_train, "training_loop_iteration", side_effect=_old_wrapped_iteration):
      with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'elastic_retry_cache'"):
        _fake_train_main(["train.py"])


if __name__ == "__main__":
  unittest.main()
