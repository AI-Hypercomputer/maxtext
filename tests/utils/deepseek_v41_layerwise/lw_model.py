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

"""Builds standalone DeepSeek-V4.1 modules abstractly and fills them from the HF checkpoint.

Nothing is initialized at full size: the module skeleton comes from `nnx.eval_shape` (as in
`model_creation_utils.create_nnx_abstract_model`), the weights are streamed by `hf_streaming_load` straight into
the abstract, sharded targets, and `checkpointing._raise_on_weight_mismatch` rejects any weight the mapping did not
fill. Only the non-weight state (the MoE's RNG key and counter) is materialized by running the constructor under
`jax.jit`, where XLA drops every weight initializer because nothing uses it.

An Engram table can instead be loaded as a row slice (`EngramSlice`): the active hash rows plus control rows read
directly from the safetensors shard, dequantized with the canonical `param_mapping.dequantize_block_scaled`, and a
NaN poison row that every hash id outside the slice is sent to, so a missing row fails loudly under `jit` too.
"""

import dataclasses
import gc
import json
import os
import pathlib
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from flax.core.spmd import logical_axis_rules
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.checkpoint_conversion.utils import hf_streaming_load, param_mapping
from maxtext.common import checkpointing
from maxtext.layers import engram, moe
from maxtext.utils import sharding
from tests.utils.parity import negctl

WEIGHT_FILTER = (nnx.Param, moe.MoEBiasVar)


@dataclasses.dataclass
class ModuleParts:
  """A module split into its graph, its weights (`nnx.Param` + `moe.MoEBiasVar`) and the rest of its state."""

  graphdef: Any
  weights: nnx.State
  rest: nnx.State

  def merge(self) -> Any:
    return merge_fresh(self.graphdef, self.weights, self.rest)


def merge_fresh(graphdef: Any, *states: nnx.State) -> Any:
  """`nnx.merge` on copies of `states`, so the module's Variables belong to the current trace.

  A State closed over from an outer `jit` holds Variables of that trace; production code that updates layer state
  inside an inner transform (e.g. `NNXDecoder._apply_layer_with_remat`) cannot mutate them.
  """
  return nnx.merge(graphdef, *[jax.tree.map(lambda x: x, s) for s in states])


def perturb_leaf(weights: nnx.State, leaf_path: str, rel_scale: float, seed: int) -> nnx.State:
  """`weights` with the leaf at `leaf_path` (pure-dict path, e.g. "mlp/shared_experts/wo/kernel") moved by
  `rel_scale` relative norm with the canonical `negctl.perturb_one_leaf`, keeping every leaf's sharding."""
  pure = nnx.to_pure_dict(weights)
  moved = negctl.perturb_one_leaf(pure, target_path=leaf_path, rel_scale=rel_scale, seed=seed)
  out = jax.tree.map(lambda x: x, weights)
  nnx.replace_by_pure_dict(out, moved)
  return jax.tree.map(lambda x, ref: jax.device_put(x, ref.sharding), out, weights)


def abstract_parts(build_fn: Callable[[], Any], config: Any, mesh: Mesh) -> ModuleParts:
  """`build_fn()`'s graph with abstract weights carrying their NamedShardings, and abstract non-weight state."""
  with logical_axis_rules(config.logical_axis_rules):
    abstract_module = nnx.eval_shape(build_fn)
    graphdef, weights, rest = nnx.split(abstract_module, WEIGHT_FILTER, ...)
    shardings = sharding.nnx_construct_named_sharding(weights, mesh)
  weights = jax.tree.map(lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), weights, shardings)
  return ModuleParts(graphdef=graphdef, weights=weights, rest=rest)


def materialize_rest(build_fn: Callable[[], Any], config: Any, mesh: Mesh) -> nnx.State:
  """The non-weight state of `build_fn()` (RNG keys / counters), replicated on `mesh`.

  The constructor runs under `jax.jit` and only the non-weight state is returned, so XLA eliminates the (unused)
  weight initializers and nothing of full size is allocated.
  """
  replicated = NamedSharding(mesh, PartitionSpec())

  def _rest() -> nnx.State:
    with logical_axis_rules(config.logical_axis_rules):
      return nnx.split(build_fn(), WEIGHT_FILTER, ...)[2]

  return jax.jit(_rest, out_shardings=replicated)()


def hf_is_quantized(hf_dir: pathlib.Path) -> bool:
  """Whether the HF checkpoint stores block-quantized weights (its config.json has a `quantization_config`)."""
  with open(hf_dir / "config.json", "r", encoding="utf-8") as f:
    return "quantization_config" in json.load(f)


def load_weights(target: Mapping[str, Any], config: Any, hf_dir: pathlib.Path) -> dict[str, Any]:
  """Streams the HF weights named by the DeepSeek-V4.1 param mapping into the abstract `target` tree.

  Args:
    target: Nested dict of `jax.ShapeDtypeStruct`s with shardings, keyed like the full model's NNX pure dict
      (e.g. `{"decoder": {"layers_14": ...}}` or `{"vision_encoder": ..., "token_embedder": ...}`).
    config: An unscanned MaxText config (`scan_layers=False`).
    hf_dir: Directory of the HF safetensors checkpoint.

  Returns:
    `target` with every leaf loaded.

  Raises:
    ValueError: If `config.scan_layers` is set, or (from `checkpointing._raise_on_weight_mismatch`) any target leaf
      was not filled or came back with the wrong shape / dtype.
  """
  if config.scan_layers:
    raise ValueError("load_weights maps unscanned layers_<i> keys; pass a config with scan_layers=False")
  quant = {"quantization_config": hf_is_quantized(hf_dir)}
  param_map = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING(quant, config, scan_layers=False)
  hook_map = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN(quant, config, scan_layers=False, saving_to_hf=False)
  loaded = hf_streaming_load.load_hf_params_streaming(str(hf_dir), dict(target), param_map, hook_map, config)
  bare = checkpointing._bare_weights(loaded)  # pylint: disable=protected-access
  checkpointing._raise_on_weight_mismatch(dict(target), bare, config=config)  # pylint: disable=protected-access
  return bare


def fill_weights(parts: ModuleParts, pure_weights: Mapping[str, Any]) -> ModuleParts:
  """`parts` with its abstract weights replaced by `pure_weights` (same structure)."""
  weights = jax.tree.map(lambda x: x, parts.weights)
  nnx.replace_by_pure_dict(weights, dict(pure_weights))
  return ModuleParts(graphdef=parts.graphdef, weights=weights, rest=parts.rest)


def _buffer_pointers(x: jax.Array) -> set[int]:
  """Device buffer addresses of `x`'s shards (a `device_put` to an equivalent sharding can alias them)."""
  return {shard.data.unsafe_buffer_pointer() for shard in x.addressable_shards}


def release(tree: Any, *, keep: Any = None) -> dict[str, float]:
  """Deletes every device buffer in `tree` that is not also in `keep`, and confirms none of them is still live.

  `keep` matters when `tree` shares device buffers with live state (a perturbed copy of weights, or weights
  `device_put` to an equivalent sharding): arrays sharing any buffer with `keep` are left alone.

  Returns:
    {"released_gib", "live_gib_after"}.

  Raises:
    AssertionError: If a released buffer still shows up in `jax.live_arrays()`.
  """
  kept = {ptr for x in jax.tree.leaves(keep) if isinstance(x, jax.Array) for ptr in _buffer_pointers(x)}
  arrays = [
      x
      for x in jax.tree.leaves(tree)
      if isinstance(x, jax.Array) and not x.is_deleted() and not kept.intersection(_buffer_pointers(x))
  ]
  released = sum(x.nbytes for x in arrays)
  ids = {id(x) for x in arrays}
  for x in arrays:
    x.delete()
  del arrays
  gc.collect()
  live = jax.live_arrays()
  still_live = [x for x in live if id(x) in ids]
  if still_live:
    raise AssertionError(f"{len(still_live)} released buffers are still live")
  return {"released_gib": released / 2**30, "live_gib_after": sum(x.nbytes for x in live) / 2**30}


def device_memory(label: str) -> dict[str, Any]:
  """Per-device `memory_stats()` after `label`: `peak_bytes_in_use` is the process-lifetime peak, so a record taken
  after a load or a gate bounds everything that ran before it. CPU has no device memory stats; on TPU or any non-CPU
  backend, missing stats or keys raise `RuntimeError`."""
  backend = jax.default_backend()
  if backend == "cpu":
    return {"label": label, "available": False, "backend": backend}
  stats = [d.memory_stats() for d in jax.local_devices()]
  if any(s is None or "peak_bytes_in_use" not in s or "bytes_in_use" not in s for s in stats):
    raise RuntimeError(f"Device memory_stats() unavailable or incomplete on non-CPU backend {backend!r}: {stats!r}")
  peaks = [int(s["peak_bytes_in_use"]) for s in stats]
  return {
      "label": label,
      "available": True,
      "peak_bytes_in_use": peaks,
      "bytes_in_use": [int(s["bytes_in_use"]) for s in stats],
      "max_peak_gib": max(peaks) / 2**30,
  }


def _read_safetensors_header(shard: pathlib.Path | str) -> tuple[int, dict[str, Any]]:
  """Reads safetensors header length and JSON dict without torch or safe_open."""
  with open(shard, "rb") as f:
    hdr_len_bytes = f.read(8)
    if len(hdr_len_bytes) < 8:
      raise ValueError(f"Truncated safetensors file: {shard}")
    hdr_len = struct.unpack("<Q", hdr_len_bytes)[0]
    hdr_bytes = f.read(hdr_len)
    if len(hdr_bytes) < hdr_len:
      raise ValueError(f"Truncated safetensors header: {shard}")
    hdr = json.loads(hdr_bytes.decode("utf-8"))
    return hdr_len, hdr


@dataclasses.dataclass(frozen=True)
class EngramSlice:
  """Rows of one Engram table kept on device.

  row_ids: sorted global row ids held in slice rows [0, len(row_ids)).
  active: bool per row id, True for rows the hasher reaches on the verified inputs.
  poison_row: slice row filled with NaN; hash ids outside `row_ids` are mapped to it.
  masked_pad_row: slice row filled with 0.0; pad/masked decoder segment positions map to it.
  num_slice_rows: total slice rows (`row_ids`, the poison row, masked_pad_row, then zero padding to the shard multiple).
  """

  row_ids: np.ndarray
  active: np.ndarray
  poison_row: int
  masked_pad_row: int
  num_slice_rows: int


def plan_engram_slice(
    active_row_ids: np.ndarray, control_row_ids: np.ndarray, *, num_table_rows: int, row_multiple: int
) -> EngramSlice:
  """Lays out the slice of `active_row_ids` + `control_row_ids` (disjoint, inside [0, num_table_rows))."""
  active = np.unique(np.asarray(active_row_ids, np.int64))
  controls = np.unique(np.asarray(control_row_ids, np.int64))
  if np.intersect1d(active, controls).size:
    raise ValueError(f"Engram control rows overlap active rows: {np.intersect1d(active, controls)[:8]}")
  row_ids = np.concatenate([active, controls])
  if row_ids.size == 0 or row_ids.min() < 0 or row_ids.max() >= num_table_rows:
    raise ValueError(f"Engram slice rows must lie in [0, {num_table_rows}), got {row_ids.min()}..{row_ids.max()}")
  order = np.argsort(row_ids)
  poison_row = int(row_ids.size)
  masked_pad_row = int(row_ids.size) + 1
  return EngramSlice(
      row_ids=row_ids[order],
      active=np.concatenate([np.ones(active.size, bool), np.zeros(controls.size, bool)])[order],
      poison_row=poison_row,
      masked_pad_row=masked_pad_row,
      num_slice_rows=-(-(masked_pad_row + 1) // row_multiple) * row_multiple,
  )


def read_engram_rows(hf_dir: pathlib.Path, layer_id: int, row_ids: np.ndarray) -> np.ndarray:
  """Reads rows `row_ids` of `layers.<layer_id>.engram.embed.weight` as [len, head_dim].

  A quantized checkpoint's FP8 rows are dequantized with `param_mapping.dequantize_block_scaled` (exact in bf16);
  an unquantized checkpoint's rows are returned in float32, so no precision is dropped.
  """
  with open(hf_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
  name = f"layers.{layer_id}.engram.embed.weight"
  shard = hf_dir / weight_map[name]
  quantized = hf_is_quantized(hf_dir)
  scale_name = name[: -len("weight")] + "scale"
  if quantized and weight_map.get(scale_name) != weight_map[name]:
    raise ValueError(f"{scale_name} is in {weight_map[scale_name]}, not beside {name} in {weight_map[name]}")

  hdr_len, hdr = _read_safetensors_header(shard)
  data_base = 8 + hdr_len
  w_info = hdr[name]
  w_offset = data_base + w_info["data_offsets"][0]
  w_shape = w_info["shape"]
  head_dim = w_shape[1]

  fd = os.open(shard, os.O_RDONLY)
  try:
    if not quantized:
      elem_size = 4 if w_info["dtype"] in ("F32", "float32") else 2
      dt = np.float32 if elem_size == 4 else ml_dtypes.bfloat16
      w_bytes_per_row = head_dim * elem_size
      rows = np.empty((len(row_ids), head_dim), dtype=dt)
      for i, r in enumerate(row_ids):
        buf = os.pread(fd, w_bytes_per_row, w_offset + int(r) * w_bytes_per_row)
        rows[i] = np.frombuffer(buf, dtype=dt)
      return rows.astype(np.float32)

    s_info = hdr[scale_name]
    s_offset = data_base + s_info["data_offsets"][0]
    scale_dim = s_info["shape"][1]
    w_rows = np.empty((len(row_ids), head_dim), dtype=np.uint8)
    s_rows = np.empty((len(row_ids), scale_dim), dtype=np.uint8)
    for i, r in enumerate(row_ids):
      w_rows[i] = np.frombuffer(os.pread(fd, head_dim, w_offset + int(r) * head_dim), dtype=np.uint8)
      s_rows[i] = np.frombuffer(os.pread(fd, scale_dim, s_offset + int(r) * scale_dim), dtype=np.uint8)
    w_fp8 = w_rows.view(ml_dtypes.float8_e4m3fn)
    s_fp8 = s_rows.view(ml_dtypes.float8_e8m0fnu)
    return np.asarray(param_mapping.dequantize_block_scaled((w_fp8, s_fp8)))
  finally:
    os.close(fd)


def build_engram_slice_table(rows: np.ndarray, plan: EngramSlice, dtype: Any, table_sharding: Any) -> jax.Array:
  """The device slice table: the rows in `plan.row_ids` order, the NaN poison row, masked_pad_row, then zero padding."""
  if rows.shape[0] != plan.row_ids.size:
    raise ValueError(f"Got {rows.shape[0]} Engram rows for {plan.row_ids.size} row ids")
  table = np.zeros((plan.num_slice_rows, rows.shape[1]), np.float32)
  table[: plan.row_ids.size] = np.asarray(rows, np.float32)
  table[plan.poison_row] = np.nan
  table[plan.masked_pad_row] = 0.0
  return jax.device_put(jnp.asarray(table, dtype), table_sharding)


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
  """Reads exactly `size` bytes from `fd` starting at `offset`, handling Linux pread buffer limits."""
  buf = bytearray()
  while len(buf) < size:
    chunk = os.pread(fd, min(size - len(buf), 0x40000000), offset + len(buf))
    if not chunk:
      raise EOFError(f"Unexpected EOF reading {size} bytes at offset {offset}, read {len(buf)} bytes")
    buf.extend(chunk)
  return bytes(buf)


_ENGRAM_LOAD_CHUNK_ROWS = 4_000_088


def load_full_engram_table(hf_dir: pathlib.Path, layer_id: int, table_abstract: jax.ShapeDtypeStruct) -> jax.Array:
  """Streams `layers.<layer_id>.engram.embed.weight` into `table_abstract` one shard/chunk at a time.

  Avoids materializing or dequantizing the full 98.3 GiB table on device 0 at once and replaces thousands of small
  ranged reads with contiguous chunk reads (`_ENGRAM_LOAD_CHUNK_ROWS`, a multiple of the 128 block size). Rows beyond
  the stored safetensors row count up to `table_abstract.shape[0]` are zero-padded.
  """
  with open(hf_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
  name = f"layers.{layer_id}.engram.embed.weight"
  shard = hf_dir / weight_map[name]
  quantized = hf_is_quantized(hf_dir)
  scale_name = name[: -len("weight")] + "scale"
  if quantized and weight_map.get(scale_name) != weight_map[name]:
    raise ValueError(f"{scale_name} is in {weight_map.get(scale_name)}, not beside {name} in {weight_map[name]}")

  total_rows, head_dim = table_abstract.shape
  target_dtype = np.dtype(table_abstract.dtype)
  idx_map = table_abstract.sharding.addressable_devices_indices_map(table_abstract.shape)
  dev_ranges: list[tuple[jax.Device, int, int]] = []
  for dev, slices in idx_map.items():
    r0, r1, rstep = slices[0].indices(total_rows)
    c0, c1, cstep = slices[1].indices(head_dim)
    if rstep != 1 or (c0, c1, cstep) != (0, head_dim, 1):
      raise ValueError(f"Unsupported Engram table shard slice {slices} on device {dev}")
    dev_ranges.append((dev, r0, r1))

  is_replicated = len(dev_ranges) > 1 and all((r0, r1) == (0, total_rows) for _, r0, r1 in dev_ranges)

  hdr_len, hdr = _read_safetensors_header(shard)
  data_base = 8 + hdr_len
  w_info = hdr[name]
  w_shape = tuple(w_info["shape"])
  if len(w_shape) != 2 or w_shape[1] != head_dim or w_shape[0] > total_rows:
    raise ValueError(f"{name} shape {w_shape} incompatible with target {table_abstract.shape}")
  stored_rows = int(w_shape[0])
  w_offset = data_base + w_info["data_offsets"][0]

  s_info = hdr[scale_name] if quantized else None
  s_offset = (data_base + s_info["data_offsets"][0]) if quantized else 0
  scale_dim = s_info["shape"][1] if quantized else 0

  fd = os.open(shard, os.O_RDONLY)
  try:

    def _fill_host_range(r0: int, r1: int) -> np.ndarray:
      host_buf = np.zeros((r1 - r0, head_dim), dtype=target_dtype)
      valid_end = min(r1, stored_rows)
      for c0 in range(r0, valid_end, _ENGRAM_LOAD_CHUNK_ROWS):
        c1 = min(c0 + _ENGRAM_LOAD_CHUNK_ROWS, valid_end)
        num_rows = c1 - c0
        if not quantized:
          elem_size = 4 if w_info["dtype"] in ("F32", "float32") else 2
          dt = np.float32 if elem_size == 4 else ml_dtypes.bfloat16
          bytes_per_row = head_dim * elem_size
          w_data = _pread_exact(fd, num_rows * bytes_per_row, w_offset + c0 * bytes_per_row)
          w_np = np.frombuffer(w_data, dtype=dt).reshape(num_rows, head_dim)
          host_buf[c0 - r0 : c1 - r0] = np.asarray(w_np, dtype=target_dtype)
          del w_data, w_np
        else:
          w_data = _pread_exact(fd, num_rows * head_dim, w_offset + c0 * head_dim)
          s_data = _pread_exact(fd, num_rows * scale_dim, s_offset + c0 * scale_dim)
          w_np = np.frombuffer(w_data, dtype=np.uint8).reshape(num_rows, head_dim).view(ml_dtypes.float8_e4m3fn)
          s_np = np.frombuffer(s_data, dtype=np.uint8).reshape(num_rows, scale_dim).view(ml_dtypes.float8_e8m0fnu)
          deq = param_mapping.dequantize_block_scaled((w_np, s_np))
          host_buf[c0 - r0 : c1 - r0] = np.asarray(jax.device_get(deq), dtype=target_dtype)
          if isinstance(deq, jax.Array) and not deq.is_deleted():
            deq.delete()
          del deq, w_np, s_np, w_data, s_data
        gc.collect()
      return host_buf

    if is_replicated:
      host_full = _fill_host_range(0, total_rows)
      out = jax.device_put(jnp.asarray(host_full, table_abstract.dtype), table_abstract.sharding)
      del host_full
      gc.collect()
      return out

    dev_arrays: list[jax.Array] = []
    for dev, r0, r1 in dev_ranges:
      host_shard = _fill_host_range(r0, r1)
      dev_arrays.append(jax.device_put(host_shard, dev))
      del host_shard
      gc.collect()
    return jax.make_array_from_single_device_arrays(table_abstract.shape, table_abstract.sharding, dev_arrays)
  finally:
    os.close(fd)


class SlicedEngramHasher:
  """The layer's hasher with its global row ids remapped to `EngramSlice` rows (missing ids -> poison row)."""

  def __init__(self, base_hasher: Any, plan: EngramSlice):
    self.base = base_hasher
    self.plan = plan
    self.num_embeddings = plan.num_slice_rows

  def __call__(
      self, input_ids: jax.Array, decoder_segment_ids: jax.Array | None = None, image_mask: jax.Array | None = None
  ):
    raw = self.base(input_ids, decoder_segment_ids=decoder_segment_ids, image_mask=image_mask)
    row_ids = jnp.asarray(self.plan.row_ids, jnp.int32)
    pos = jnp.clip(jnp.searchsorted(row_ids, raw), 0, row_ids.shape[0] - 1)
    mapped = jnp.where(row_ids[pos] == raw, pos, self.plan.poison_row).astype(jnp.int32)
    if decoder_segment_ids is not None:
      pad_mask = (decoder_segment_ids == 0)[..., None]
      mapped = jnp.where(pad_mask, self.plan.masked_pad_row, mapped)
    return mapped


def install_sliced_hasher(layer: Any, plan: EngramSlice) -> None:
  """Points `layer.engram_hasher` at the slice (call after merging a layer whose engram table is the slice)."""
  layer.engram_hasher = engram.StaticWrapper(SlicedEngramHasher(layer.engram_hasher.val, plan))


def verify_slice_covers(base_hasher: Any, plan: EngramSlice, inputs: Sequence[Mapping[str, Any]]) -> None:
  """Raises unless every row the base hasher reaches on each of `inputs` (hasher kwargs) is an active slice row."""
  active = set(plan.row_ids[plan.active].tolist())
  for kwargs in inputs:
    reached = set(np.unique(np.asarray(base_hasher(**kwargs))).tolist())
    missing = sorted(reached - active)
    if missing:
      raise ValueError(f"Engram slice misses {len(missing)} reached rows, e.g. {missing[:8]}")


def packing_control_bounds(seq_len: int) -> tuple[int, int]:
  """Returns (s_a, s_b) bounds for synthetic 2-document packing negative control."""
  s_a = min(seq_len // 2, 64) | 1
  s_b = max(1, min(seq_len // 4, 256))
  return s_a, s_b


def packing_control_segment_ids(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
  """Constructs the synthetic 2-doc segment IDs and image mask used by _packing_control."""
  b, s = tokens.shape[0], tokens.shape[1]
  s_a, s_b = packing_control_bounds(s)
  s_pad = s - s_a - s_b
  seg_proper = np.concatenate(
      [
          np.ones((b, s_a), dtype=np.int32),
          2 * np.ones((b, s_b), dtype=np.int32),
          np.zeros((b, s_pad), dtype=np.int32),
      ],
      axis=1,
  )
  seg_ones = np.ones((b, s), dtype=np.int32)
  img_mask = np.zeros((b, s), dtype=bool)
  img_mask[:, : min(2, s_a)] = True
  return seg_proper, seg_ones, img_mask, s_a, s_b


def engram_packing_control_rows(base_hasher: Any, tokens: np.ndarray) -> np.ndarray:
  """Row IDs reached during the 2-doc packing control."""
  if tokens.shape[1] < 16:
    return np.array([], dtype=np.int64)
  seg_proper, seg_ones, img_mask, _, _ = packing_control_segment_ids(tokens)
  h_prop = np.asarray(
      base_hasher(jnp.asarray(tokens), decoder_segment_ids=jnp.asarray(seg_proper), image_mask=jnp.asarray(img_mask))
  )
  h_ones = np.asarray(
      base_hasher(jnp.asarray(tokens), decoder_segment_ids=jnp.asarray(seg_ones), image_mask=jnp.asarray(img_mask))
  )
  pack_rows = np.unique(np.concatenate([h_prop.reshape(-1), h_ones.reshape(-1)]))
  return pack_rows[pack_rows >= 0].astype(np.int64)
