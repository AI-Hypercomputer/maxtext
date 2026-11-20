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

"""Tests for hf_streaming_load, the streaming HF SafeTensors -> MaxText loader."""

import os

# Simulate 4 devices on CPU so both the load split and the target shardings are real.
# This only takes effect if JAX is not initialized yet; the tests pass with any device count.
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

# pylint: disable=wrong-import-position
import collections
import functools
import math
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax.traverse_util
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import safetensors_reader
from maxtext.checkpoint_conversion.utils import tensor_handling
import numpy as np
import safetensors.flax

# Tiny model dims. The vocab (66) splits only 2 ways across 4 devices (gcd 2), and 3 experts
# can't be split at all (gcd 1), so every load-split case is exercised.
_LAYERS, _EMBED, _HEADS, _KV_HEADS, _HEAD_DIM, _MLP, _VOCAB, _EXPERTS = 3, 16, 4, 2, 8, 24, 66, 3
_HF_CONFIG = {"num_hidden_layers": _LAYERS, "hidden_size": _EMBED, "head_dim": _HEAD_DIM}
_MAPPINGS = {
    "llama": (param_mapping.LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING, param_mapping.LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN),
    "mixtral": (param_mapping.MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING, param_mapping.MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN),
}


def _config(scan_layers):
  return types.SimpleNamespace(scan_layers=scan_layers, param_scan_axis=1, num_experts=_EXPERTS, head_dim=_HEAD_DIM)


def _maps(model, config):
  param_fn, hook_fn = _MAPPINGS[model]
  param_map = param_fn(_HF_CONFIG, config, scan_layers=config.scan_layers)
  hook_map = hook_fn(_HF_CONFIG, config, scan_layers=config.scan_layers, saving_to_hf=False)
  return param_map, hook_map


def _hf_shapes(model):
  """HF tensor name -> shape for a tiny Llama or Mixtral checkpoint."""
  shapes = {
      "model.embed_tokens.weight": (_VOCAB, _EMBED),
      "lm_head.weight": (_VOCAB, _EMBED),
      "model.norm.weight": (_EMBED,),
  }
  for i in range(_LAYERS):
    p = f"model.layers.{i}."
    shapes |= {
        p + "self_attn.q_proj.weight": (_HEADS * _HEAD_DIM, _EMBED),
        p + "self_attn.k_proj.weight": (_KV_HEADS * _HEAD_DIM, _EMBED),
        p + "self_attn.v_proj.weight": (_KV_HEADS * _HEAD_DIM, _EMBED),
        p + "self_attn.o_proj.weight": (_EMBED, _HEADS * _HEAD_DIM),
        p + "input_layernorm.weight": (_EMBED,),
        p + "post_attention_layernorm.weight": (_EMBED,),
    }
    if model == "llama":
      shapes |= {
          p + "mlp.gate_proj.weight": (_MLP, _EMBED),
          p + "mlp.up_proj.weight": (_MLP, _EMBED),
          p + "mlp.down_proj.weight": (_EMBED, _MLP),
      }
    else:
      shapes[p + "block_sparse_moe.gate.weight"] = (_EXPERTS, _EMBED)
      for j in range(_EXPERTS):
        e = f"{p}block_sparse_moe.experts.{j}."
        shapes |= {e + "w1.weight": (_MLP, _EMBED), e + "w3.weight": (_MLP, _EMBED), e + "w2.weight": (_EMBED, _MLP)}
  return shapes


def _mt_shapes(model, scan_layers):
  """Flattened MaxText (Linen) weight name -> shape, laid out as MaxText does."""
  shapes = {
      "params.token_embedder.embedding": (_VOCAB, _EMBED),
      "params.decoder.logits_dense.kernel": (_EMBED, _VOCAB),
      "params.decoder.decoder_norm.scale": (_EMBED,),
  }
  per_layer = {
      "self_attention.query.kernel": (_EMBED, _HEADS, _HEAD_DIM),
      "self_attention.key.kernel": (_EMBED, _KV_HEADS, _HEAD_DIM),
      "self_attention.value.kernel": (_EMBED, _KV_HEADS, _HEAD_DIM),
      "self_attention.out.kernel": (_HEADS, _HEAD_DIM, _EMBED),
      "pre_self_attention_layer_norm.scale": (_EMBED,),
      "post_self_attention_layer_norm.scale": (_EMBED,),
  }
  per_expert = {}
  if model == "llama":
    per_layer |= {"mlp.wi_0.kernel": (_EMBED, _MLP), "mlp.wi_1.kernel": (_EMBED, _MLP), "mlp.wo.kernel": (_MLP, _EMBED)}
  else:
    per_layer["MoeBlock_0.gate.kernel"] = (_EMBED, _EXPERTS)
    per_expert = {"MoeBlock_0.wi_0": (_EMBED, _MLP), "MoeBlock_0.wi_1": (_EMBED, _MLP), "MoeBlock_0.wo": (_MLP, _EMBED)}
  if scan_layers:
    # Layers stack along param_scan_axis=1; expert weights are (experts, layers, ...).
    shapes |= {f"params.decoder.layers.{k}": s[:1] + (_LAYERS,) + s[1:] for k, s in per_layer.items()}
    shapes |= {f"params.decoder.layers.{k}": (_EXPERTS, _LAYERS) + s for k, s in per_expert.items()}
  else:
    for i in range(_LAYERS):
      shapes |= {f"params.decoder.layers_{i}.{k}": s for k, s in per_layer.items()}
      shapes |= {f"params.decoder.layers_{i}.{k}": (_EXPERTS,) + s for k, s in per_expert.items()}
  return shapes


def _mesh():
  devices = jax.devices()[:4]
  grid = (2, len(devices) // 2) if len(devices) % 2 == 0 else (1, len(devices))
  return Mesh(np.array(devices).reshape(grid), ("fsdp", "tensor"))


def _spec(shape, mesh):
  """Puts "fsdp" on the first dim it divides and "tensor" on a later one, like a sharded model."""
  spec, dim = [None] * len(shape), 0
  for axis in ("fsdp", "tensor"):
    while dim < len(shape) and shape[dim] % mesh.shape[axis]:
      dim += 1
    if dim < len(shape):
      spec[dim] = axis
      dim += 1
  return PartitionSpec(*spec)


def _target_tree(model, scan_layers, dtype, mesh):
  flat = {
      name: jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, _spec(shape, mesh)))
      for name, shape in _mt_shapes(model, scan_layers).items()
  }
  return flax.traverse_util.unflatten_dict(flat, sep=".")


def _write_checkpoint(directory, shapes, skip=()):
  """Writes random bf16 tensors round-robin into 2 files, so every layer spans both files."""
  rng = np.random.default_rng(0)
  tensors = {
      k: jnp.asarray(rng.standard_normal(s, dtype=np.float32)).astype(jnp.bfloat16)
      for k, s in shapes.items()
      if k not in skip
  }
  keys = sorted(tensors)
  for i in range(2):
    path = os.path.join(directory, f"model-0000{i + 1}-of-00002.safetensors")
    safetensors.flax.save_file({k: tensors[k] for k in keys[i::2]}, path)
  return tensors


class LoadShardingTest(parameterized.TestCase):
  """The rule for how each HF tensor is split across the TPU chips while it is read."""

  @parameterized.named_parameters(
      ("divisible", (8192, 8192), 4),
      ("gcd_split", (6, 1 << 20), 2),
      ("coprime_dim0_full_copy", (7, 1 << 20), 1),
      ("tiny_full_copy", (64, 64), 1),
      ("one_dim_large", (1 << 20,), 4),
      ("one_dim_tiny", (8192,), 1),
      ("three_dim_splits_dim0", (8, 1024, 1024), 4),
      ("scalar", (), 1),
  )
  def test_split_factor(self, shape, expected):
    self.assertEqual(hf_streaming_load.load_split_factor(shape, jnp.bfloat16, num_devices=4), expected)

  def test_min_bytes_to_split(self):
    self.assertEqual(hf_streaming_load.load_split_factor((64, 64), jnp.bfloat16, 4, min_bytes_to_split=0), 4)

  def test_pieces_are_whole_rows(self):
    """Every TPU chip gets whole rows, i.e. one unbroken byte range of the file."""
    devices = jax.devices()[:4]
    shape = (6, 1 << 18)
    sharding = hf_streaming_load.choose_load_sharding(shape, jnp.bfloat16, devices)
    pieces = set()
    for rows, cols in sharding.devices_indices_map(shape).values():
      self.assertEqual(cols, slice(None))
      pieces.add((rows.start, rows.stop))
    self.assertLen(pieces, hf_streaming_load.load_split_factor(shape, jnp.bfloat16, len(devices)))

  @parameterized.named_parameters(
      ("gcd_of_split_factors", [(8, 1024), (4, 1024), (6, 1024)], 2),
      ("unsplit_tensors_ignored", [(8, 1024), (7, 1024), (4, 4)], 8),
      ("none_split", [(7, 1024), (4, 4)], 1),
  )
  def test_shared_split_factor(self, shapes, expected):
    metas = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes]
    self.assertEqual(hf_streaming_load.shared_split_factor(metas, num_devices=8, min_bytes_to_split=1024), expected)

  def test_shared_mesh(self):
    """Tensors loaded with a shared factor all sit on one mesh, split by it or not at all."""
    devices = jax.devices()[:4]
    shapes = [(8, 1024), (2, 1024), (7, 1024)]
    factor = hf_streaming_load.shared_split_factor(
        [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes], len(devices)
    )
    shardings = [hf_streaming_load.choose_load_sharding(s, jnp.float32, devices, mesh_factor=factor) for s in shapes]
    self.assertLen({sharding.mesh for sharding in shardings}, 1)
    for shape, sharding in zip(shapes, shardings):
      own = hf_streaming_load.load_split_factor(shape, jnp.float32, len(devices))
      pieces = {rows.start for rows, _ in sharding.devices_indices_map(shape).values()}
      self.assertLen(pieces, factor if own > 1 else 1)

  def test_group_bytes_per_host(self):
    """A split tensor is read in parts across hosts; one that isn't split is read in full on every host."""
    target = types.SimpleNamespace(sharding=NamedSharding(_mesh(), PartitionSpec()))
    metadata = {
        "split": jax.ShapeDtypeStruct((8, 256), jnp.float32),
        "whole": jax.ShapeDtypeStruct((7, 256), jnp.float32),
    }
    group = [hf_streaming_load._Write((target,), (), key) for key in metadata]  # pylint: disable=protected-access
    num_devices = len(_mesh().devices.flat)
    with mock.patch.object(jax, "process_count", return_value=2):
      got = hf_streaming_load._group_bytes_per_host(group, metadata, 0)  # pylint: disable=protected-access
    want = sum(
        math.prod(m.shape) * 4 // min(hf_streaming_load.load_split_factor(m.shape, m.dtype, num_devices, 0), 2)
        for m in metadata.values()
    )
    self.assertEqual(got, want)
    if num_devices == 4:
      self.assertEqual(got, 8 * 256 * 4 // 2 + 7 * 256 * 4)


def _group_sizes(model):
  """HF bytes of each load group of the bf16 test checkpoint: the non-layer tensors, then each layer."""
  sizes = collections.Counter()
  for key, shape in _hf_shapes(model).items():
    sizes[hf_streaming_load._layer_group(key)] += 2 * math.prod(shape)  # pylint: disable=protected-access
  return [sizes[group] for group in sorted(sizes)]


class PackGroupsTest(parameterized.TestCase):
  """How consecutive load groups (whole layers) are packed into read calls by size."""

  @parameterized.named_parameters(
      ("fills_up_to_budget", [5, 3, 3, 3, 9, 1], 6, [[0], [1, 2], [3], [4], [5]]),
      ("everything_fits", [1, 2, 3], 100, [[0, 1, 2]]),
      ("no_budget_one_call", [5, 3, 3], None, [[0, 1, 2]]),
      ("zero_budget_one_group_each", [5, 3, 3], 0, [[0], [1], [2]]),
      ("no_groups", [], 6, []),
  )
  def test_pack_groups(self, sizes, max_bytes, expected):
    self.assertEqual([list(call) for call in hf_streaming_load.pack_groups(sizes, max_bytes)], expected)


class StreamingLoadTest(parameterized.TestCase):
  """The streamed weights must equal the in-memory reference transform, bit for bit."""

  def _load(self, model, scan_layers, dtype=jnp.float32, skip_hf=(), edit_maps=None, extra_hf=None, **load_kwargs):
    """Writes a tiny HF checkpoint and streams it into MaxText weights of `dtype`.

    `extra_hf(hf_tensors)` may return more HF tensors, saved in a third file.
    """
    ckpt_dir = self.create_tempdir().full_path
    hf_tensors = _write_checkpoint(ckpt_dir, _hf_shapes(model), skip=skip_hf)
    if extra_hf:
      safetensors.flax.save_file(extra_hf(hf_tensors), os.path.join(ckpt_dir, "model-extra.safetensors"))
    config = _config(scan_layers)
    param_map, hook_map = _maps(model, config)
    if edit_maps:
      edit_maps(param_map, hook_map)
    target_tree = _target_tree(model, scan_layers, dtype, _mesh())
    restored = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir, target_tree, param_map, hook_map, config, min_bytes_to_split=0, **load_kwargs
    )
    return restored, target_tree, hf_tensors, param_map, hook_map, config

  @parameterized.product(
      model=("llama", "mixtral"),
      scan_layers=(True, False),
      dtype=(jnp.float32, jnp.bfloat16),
      read_bytes_per_host=(0, None),  # One read call per layer, or one call for everything.
      prefetch=(True, False),
  )
  def test_matches_in_memory_transform(self, model, scan_layers, dtype, read_bytes_per_host, prefetch):
    got, target_tree, hf_tensors, param_map, hook_map, config = self._load(
        model, scan_layers, dtype, read_bytes_per_host=read_bytes_per_host, prefetch=prefetch
    )
    want = load_dynamic.transform_hf_state_to_mt_state(dict(hf_tensors), target_tree, param_map, hook_map, config)

    flat_got = flax.traverse_util.flatten_dict(got, sep=".")
    flat_want = flax.traverse_util.flatten_dict(want, sep=".")
    flat_target = flax.traverse_util.flatten_dict(target_tree, sep=".")
    self.assertEqual(set(flat_got), set(flat_target))
    for name, target in flat_target.items():
      with self.subTest(name):
        self.assertIsInstance(flat_got[name], jax.Array)
        self.assertEqual(flat_got[name].dtype, target.dtype)
        self.assertEqual(flat_got[name].sharding, target.sharding)
        # The reference skips the cast for unstacked weights; compare in float32 (exact for bf16).
        np.testing.assert_array_equal(
            np.asarray(flat_got[name], np.float32), np.asarray(flat_want[name].astype(target.dtype), np.float32)
        )

  @parameterized.parameters(True, False)
  def test_tuple_mt_key_splits_one_hf_tensor(self, scan_layers):
    """One fused HF tensor feeds two MaxText weights, as Qwen3.5's gate_up_proj -> (wi_0, wi_1) does."""

    def fuse(hf_tensors):
      return {
          f"model.layers.{i}.mlp.gate_up_proj.weight": jnp.concatenate(
              [hf_tensors[f"model.layers.{i}.mlp.gate_proj.weight"], hf_tensors[f"model.layers.{i}.mlp.up_proj.weight"]]
          )
          for i in range(_LAYERS)
      }

    def split_gate_up(x, target_shape):
      # One shape per MaxText weight, as `to_maxtext` passes them (Qwen3-VL's vision hooks index them).
      gate, up = jnp.split(x, 2, axis=-2)
      return jnp.stack([gate.T.reshape(target_shape[0]), up.T.reshape(target_shape[1])], axis=-1)

    def use_fused(param_map, hook_map):
      for wi_0 in [k for k in param_map if isinstance(k, str) and k.endswith("mlp-wi_0-kernel")]:
        wi_1 = wi_0.replace("wi_0", "wi_1")
        source = param_map.pop(wi_0)
        del param_map[wi_1]
        hook_map.pop(wi_0, None)
        hook_map.pop(wi_1, None)
        fused = [s.replace("gate_proj", "gate_up_proj") for s in source] if isinstance(source, list) else None
        param_map[(wi_0, wi_1)] = fused or source.replace("gate_proj", "gate_up_proj")
        hook_map[(wi_0, wi_1)] = split_gate_up

    got, target_tree, hf_tensors, *_ = self._load("llama", scan_layers, edit_maps=use_fused, extra_hf=fuse)
    config = _config(scan_layers)
    param_map, hook_map = _maps("llama", config)
    want = load_dynamic.transform_hf_state_to_mt_state(dict(hf_tensors), target_tree, param_map, hook_map, config)
    flat_got = flax.traverse_util.flatten_dict(got, sep=".")
    flat_want = flax.traverse_util.flatten_dict(want, sep=".")
    mlp = [name for name in flat_got if ".mlp.wi_" in name]
    self.assertLen(mlp, 2 if scan_layers else 2 * _LAYERS)
    for name, array in flat_got.items():
      with self.subTest(name):
        self.assertIsInstance(array, jax.Array)
        np.testing.assert_array_equal(np.asarray(array), np.asarray(flat_want[name], np.float32))

  def test_fused_hf_tensors_with_different_split_factors(self):
    """Qwen3.5's in_proj_qkv and in_proj_z split differently (4 and 2 ways on 4 chips), yet one hook joins them."""
    text_config = {
        "num_hidden_layers": 4,
        "num_experts": 1,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 1,
        "linear_value_head_dim": 2,
    }
    maxtext_config = mock.Mock(
        num_decoder_layers=4, num_experts=1, inhomogeneous_layer_cycle_interval=4, use_multimodal=False
    )
    key = "params-decoder-layers-local_layers-attention-in_proj_qkvz-kernel"
    hook = param_mapping.QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        {"text_config": text_config}, maxtext_config, scan_layers=True
    )[key]
    rng = np.random.default_rng(0)
    hf = {"qkv": rng.standard_normal((4, 8), dtype=np.float32), "z": rng.standard_normal((2, 8), dtype=np.float32)}
    ckpt_dir = self.create_tempdir().full_path
    safetensors.flax.save_file({k: jnp.asarray(v) for k, v in hf.items()}, os.path.join(ckpt_dir, "model.safetensors"))
    want = hook((hf["qkv"], hf["z"]), None)
    mesh = _mesh()
    target = {"w": jax.ShapeDtypeStruct(want.shape, jnp.float32, sharding=NamedSharding(mesh, _spec(want.shape, mesh)))}

    got = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir, target, {"params-w": ("qkv", "z")}, {"params-w": hook}, _config(True), min_bytes_to_split=0
    )
    np.testing.assert_array_equal(np.asarray(got["params"]["w"]), want)

  def test_numpy_hook_falls_back_to_whole_tensors_on_every_host(self):
    """A hook using host numpy fails on a tensor split across hosts, then gets the whole tensor, as in load_dynamic."""
    kinds = []
    keys = ("params-decoder-logits_dense-kernel", "params-decoder-layers-mlp-wo-kernel")  # Unstacked and stacked.

    def numpy_only(hook):
      def wrapped(x, target_shape):
        kinds.append(type(x).__name__)
        if isinstance(x, jax.Array):
          raise RuntimeError("Fetching value for `jax.Array` that spans non-addressable (non process local) devices")
        return tensor_handling.apply_hook_fns(np.asarray(x), target_shape, hook)

      return wrapped

    def use_numpy_hooks(param_map, hook_map):
      del param_map
      for key in keys:
        hook_map[key] = numpy_only(hook_map.get(key))

    with (
        mock.patch.object(tensor_handling, "_spans_other_hosts", return_value=True),
        mock.patch.object(tensor_handling, "_gather_to_host", side_effect=lambda w: jax.tree.map(np.asarray, w)),
    ):
      got, target_tree, hf_tensors, *_ = self._load("llama", scan_layers=True, edit_maps=use_numpy_hooks)
    self.assertIn("ndarray", kinds)
    config = _config(True)
    param_map, hook_map = _maps("llama", config)
    want = load_dynamic.transform_hf_state_to_mt_state(dict(hf_tensors), target_tree, param_map, hook_map, config)
    flat_got = flax.traverse_util.flatten_dict(got, sep=".")
    flat_want = flax.traverse_util.flatten_dict(want, sep=".")
    flat_target = flax.traverse_util.flatten_dict(target_tree, sep=".")
    for name in ("params.decoder.logits_dense.kernel", "params.decoder.layers.mlp.wo.kernel"):
      with self.subTest(name):
        self.assertEqual(flat_got[name].sharding, flat_target[name].sharding)
        np.testing.assert_array_equal(np.asarray(flat_got[name]), np.asarray(flat_want[name]))

  def test_legacy_hf_names_are_renamed(self):
    """A checkpoint saved under older names (e.g. gemma-3's `language_model.model.*`) loads with `rename_hf_keys`."""

    def legacy(key):
      return key.replace("model.", "language_model.model.", 1) if key.startswith("model.") else key

    def current(source):
      if isinstance(source, list):
        return [current(s) for s in source]
      return source.replace("model.", "model.language_model.", 1) if source.startswith("model.") else source

    ckpt_dir = self.create_tempdir().full_path
    hf_tensors = _write_checkpoint(ckpt_dir, {legacy(k): s for k, s in _hf_shapes("llama").items()})
    config = _config(True)
    param_map, hook_map = _maps("llama", config)
    param_map = {k: current(v) for k, v in param_map.items()}
    target_tree = _target_tree("llama", True, jnp.float32, _mesh())
    rename = functools.partial(load_dynamic.rename_legacy_hf_keys, param_map_mt_to_hf=param_map)

    with self.assertRaisesRegex(ValueError, "not in the checkpoint"):
      hf_streaming_load.load_hf_params_streaming(ckpt_dir, target_tree, param_map, hook_map, config)
    got = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir, target_tree, param_map, hook_map, config, min_bytes_to_split=0, rename_hf_keys=rename
    )
    want = load_dynamic.transform_hf_state_to_mt_state(rename(dict(hf_tensors)), target_tree, param_map, hook_map, config)
    flat_got = flax.traverse_util.flatten_dict(got, sep=".")
    for name, array in flax.traverse_util.flatten_dict(want, sep=".").items():
      with self.subTest(name):
        np.testing.assert_array_equal(np.asarray(flat_got[name]), np.asarray(array, np.float32))

  def test_collection_key_wins_over_params_key_for_one_weight(self):
    """DeepSeek-V3 maps its gate bias under `params-` and `MoEBiasVar-`; in an NNX tree both name one weight."""
    norm_key = "params-decoder-decoder_norm-scale"
    collection_key = "MoEBiasVar-decoder-decoder_norm-scale"
    collection_source = "model.layers.0.input_layernorm.weight"

    def add_collection_key(param_map, hook_map):
      del hook_map
      entries = [(collection_key, collection_source)] + list(param_map.items())  # The collection key first.
      param_map.clear()
      param_map.update(entries)

    ckpt_dir = self.create_tempdir().full_path
    hf_tensors = _write_checkpoint(ckpt_dir, _hf_shapes("llama"))
    config = _config(True)
    param_map, hook_map = _maps("llama", config)
    add_collection_key(param_map, hook_map)
    # An NNX tree: bare weights, without the `params` collection.
    flat = flax.traverse_util.flatten_dict(_target_tree("llama", True, jnp.float32, _mesh()), sep=".")
    target_tree = flax.traverse_util.unflatten_dict({k.removeprefix("params."): v for k, v in flat.items()}, sep=".")

    got = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir, target_tree, param_map, hook_map, config, min_bytes_to_split=0
    )
    self.assertIn(norm_key, param_map)
    np.testing.assert_array_equal(
        np.asarray(got["params"]["decoder"]["decoder_norm"]["scale"]),
        np.asarray(hf_tensors[collection_source], np.float32),
    )

  def test_peak_hbm_is_none_where_not_reported(self):
    device = mock.Mock(memory_stats=mock.Mock(side_effect=NotImplementedError))
    with mock.patch.object(jax, "local_devices", return_value=[device]):
      self.assertIsNone(hf_streaming_load._peak_hbm_gb())  # pylint: disable=protected-access

  def _read_calls(self, **load_kwargs):
    """Streams the tiny Mixtral checkpoint and returns the request of each read call, in order."""
    real_fetch = safetensors_reader.SafetensorsReader.fetch
    requests = []

    def spy(reader, request, *args, **kwargs):
      requests.append(dict(request))
      return real_fetch(reader, request, *args, **kwargs)

    with mock.patch.object(safetensors_reader.SafetensorsReader, "fetch", autospec=True, side_effect=spy):
      self._load("mixtral", scan_layers=True, **load_kwargs)
    return requests

  def _layers_per_call(self, requests):
    return [sorted({hf_streaming_load._layer_group(k) for k in r}) for r in requests]  # pylint: disable=protected-access

  def test_zero_budget_reads_one_layer_per_call(self):
    """One read per decoder layer (plus one for the rest), each tensor read once, split by rows."""
    requests = self._read_calls(read_bytes_per_host=0)

    self.assertEqual(self._layers_per_call(requests), [[("", -1)]] + [[("model.", i)] for i in range(_LAYERS)])
    devices = tuple(_mesh().devices.flat)
    read = []
    for request in requests:
      for sds in request.values():
        self.assertEqual(sds.dtype, jnp.bfloat16)  # The file's dtype: no cast on the host.
        self.assertEqual(sds.sharding, hf_streaming_load.choose_load_sharding(sds.shape, sds.dtype, devices, 0))
      read.extend(request)
    self.assertCountEqual(read, _hf_shapes("mixtral"))

  def test_budget_packs_consecutive_whole_layers(self):
    """A budget of two layers' bytes reads [non-layer tensors + layer 0], then [layers 1 and 2]."""
    non_layer_bytes, layer_bytes, *_ = _group_sizes("mixtral")
    self.assertLess(non_layer_bytes, layer_bytes)  # The expected packing relies on this.
    requests = self._read_calls(read_bytes_per_host=2 * layer_bytes)

    self.assertEqual(self._layers_per_call(requests), [[("", -1), ("model.", 0)], [("model.", 1), ("model.", 2)]])
    for request in requests:
      call_bytes = sum(hf_streaming_load._nbytes(sds) for sds in request.values())  # pylint: disable=protected-access
      self.assertLessEqual(call_bytes, 2 * layer_bytes)

  def test_no_budget_reads_everything_in_one_call(self):
    requests = self._read_calls(read_bytes_per_host=None)
    self.assertLen(requests, 1)
    self.assertCountEqual(requests[0], _hf_shapes("mixtral"))

  def test_unmapped_weight_stays_abstract(self):
    """A weight no mapping covers is returned unloaded, so load_state_if_possible reports it."""

    def drop_norm(param_map, hook_map):
      del param_map["params-decoder-decoder_norm-scale"], hook_map

    restored, *_ = self._load("llama", scan_layers=True, edit_maps=drop_norm)
    self.assertIsInstance(restored["params"]["decoder"]["decoder_norm"]["scale"], jax.ShapeDtypeStruct)
    self.assertIsInstance(restored["params"]["decoder"]["logits_dense"]["kernel"], jax.Array)

  def test_missing_hf_tensor_raises(self):
    with self.assertRaisesRegex(ValueError, r"not in the checkpoint.*model\.layers\.1\.self_attn\.k_proj\.weight"):
      self._load("llama", scan_layers=True, skip_hf=("model.layers.1.self_attn.k_proj.weight",))

  def test_short_layer_list_raises(self):
    """A mapping that doesn't fill every layer would leave part of a preallocated weight zero."""

    def drop_last_layer(param_map, hook_map):
      del hook_map
      key = "params-decoder-layers-self_attention-query-kernel"
      param_map[key] = param_map[key][:-1]

    with self.assertRaisesRegex(ValueError, rf"gives {_LAYERS - 1} entries"):
      self._load("llama", scan_layers=True, edit_maps=drop_last_layer)

  def test_wrong_hook_shape_raises(self):
    """A hook whose output doesn't fit the MaxText weight fails with the weight's name."""

    def identity_hook(param_map, hook_map):
      del param_map
      hook_map["params-decoder-logits_dense-kernel"] = lambda x, shape: x

    with self.assertRaisesRegex(ValueError, "Hooks for params-decoder-logits_dense-kernel"):
      self._load("llama", scan_layers=True, edit_maps=identity_hook)

  @parameterized.parameters(True, False)
  def test_none_hf_key_uses_hook_alone(self, scan_layers):
    """A mapping with no HF tensor gets its value from the hook alone, as DeepSeek-V4's mhc_norm scales do."""

    def no_hf_norm(param_map, hook_map):
      param_map["params-decoder-decoder_norm-scale"] = None
      hook_map["params-decoder-decoder_norm-scale"] = lambda x, shape: np.full(shape, 2.0, np.float32)

    restored, target_tree, *_ = self._load("llama", scan_layers, dtype=jnp.bfloat16, edit_maps=no_hf_norm)
    got = restored["params"]["decoder"]["decoder_norm"]["scale"]
    want = target_tree["params"]["decoder"]["decoder_norm"]["scale"]
    self.assertEqual((got.dtype, got.sharding), (want.dtype, want.sharding))
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.full(want.shape, 2.0, np.float32))

  def test_none_hf_key_without_hook_raises(self):
    def no_hf_norm(param_map, hook_map):
      param_map["params-decoder-decoder_norm-scale"] = None
      hook_map.pop("params-decoder-decoder_norm-scale", None)

    with self.assertRaisesRegex(ValueError, "no HF tensor for params-decoder-decoder_norm-scale"):
      self._load("llama", scan_layers=True, edit_maps=no_hf_norm)


class Float8SourceTest(parameterized.TestCase):
  """An FP8 HF tensor must not be cast into a wider MaxText weight without its scales."""

  def _load(self, target_dtype, param_map, hook_map=None):
    ckpt_dir = self.create_tempdir().full_path
    tensors = {
        "w": jnp.asarray(np.arange(32, dtype=np.float32).reshape(8, 4)).astype(jnp.float8_e4m3fn),
        "w_scale": jnp.full((8, 1), 2.0, jnp.float32),
    }
    safetensors.flax.save_file(tensors, os.path.join(ckpt_dir, "model.safetensors"))
    mesh = _mesh()
    target = {"w": jax.ShapeDtypeStruct((8, 4), target_dtype, sharding=NamedSharding(mesh, PartitionSpec()))}
    return hf_streaming_load.load_hf_params_streaming(
        ckpt_dir, target, param_map, hook_map or {}, _config(True), min_bytes_to_split=0
    )

  def test_lone_fp8_tensor_into_wider_weight_raises(self):
    with self.assertRaisesRegex(ValueError, "FP8 HF tensors would be cast into wider MaxText weights"):
      self._load(jnp.bfloat16, {"params-w": "w"})

  def test_fp8_tensor_into_fp8_weight_loads(self):
    got = self._load(jnp.float8_e4m3fn, {"params-w": "w"})
    self.assertEqual(got["params"]["w"].dtype, jnp.float8_e4m3fn)

  def test_fp8_tensor_with_its_scale_loads(self):
    def dequantize(x, target_shape):
      del target_shape
      weight, scale = x
      return weight.astype(jnp.float32) * scale

    got = self._load(jnp.float32, {"params-w": ("w", "w_scale")}, {"params-w": dequantize})
    fp8 = jnp.asarray(np.arange(32, dtype=np.float32).reshape(8, 4)).astype(jnp.float8_e4m3fn)
    np.testing.assert_array_equal(np.asarray(got["params"]["w"]), 2 * np.asarray(fp8.astype(jnp.float32)))


class DeepSeekV4LayoutTest(parameterized.TestCase):
  """DeepSeek-V4 layouts with scanned layers that stack a flat HF key list along axis 0, as `load_dynamic` does."""

  @parameterized.named_parameters(
      ("routing_bias_layers_first", "MoEBiasVar-decoder-scanned_blocks-layers_0-mlp-MoeBlock_0-gate-bias", (20, 8), 0),
      ("hash_table_layers_first", "Tid2EidVar-decoder-layers_0-mlp-MoeBlock_0-tid2eid", (4, 6), 0),
      ("experts_of_unscanned_layer", "params-decoder-layers_1-mlp-MoeBlock_0-wi_0", (8, 16, 24), 0),
      (
          "experts_of_mtp_layer",
          "params-mtp_block-mtp_layer_1-mtp_1_transformer_layer-mlp-MoeBlock_0-wi_0",
          (8, 16, 24),
          0,
      ),
      ("scanned_moe_gate", "params-decoder-layers-MoeBlock_0-gate-kernel", (16, 3, 8), 1),  # Mixtral.
      ("scanned_block_weight", "params-decoder-scanned_blocks-layers_0-mlp-MoeBlock_0-gate-kernel", (16, 20, 8), 1),
      ("scanned_layer_weight", "params-decoder-layers-self_attention-query-kernel", (16, 3, 4, 8), 1),
      ("rank_1_weight", "params-decoder-layers-pre_self_attention_layer_norm-scale", (3,), 0),
  )
  def test_flat_list_axis(self, mt_key, shape, axis):
    config = _config(scan_layers=True)
    self.assertEqual(hf_streaming_load._stacked_axes(mt_key, shape, 1, config), (axis,))  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ("linen_collection", "MoEBiasVar.decoder.layers_3.mlp.MoeBlock_0.gate.bias"),
      ("nnx_without_collection", "decoder.layers_3.mlp.MoeBlock_0.gate.bias"),
  )
  def test_resolves_other_collections(self, flat_key):
    """A Linen tree keeps the `MoEBiasVar` collection name; an NNX tree holds the weight beside the rest."""
    flat_target = {flat_key: None, "decoder.layers_3.mlp.MoeBlock_0.gate.kernel": None}
    mt_key = "MoEBiasVar-decoder-layers_3-mlp-MoeBlock_0-gate-bias"
    self.assertEqual(hf_streaming_load.resolve_target_name(mt_key, flat_target), flat_key)

  def test_collections_beside_params_are_loaded_and_kept(self):
    """A routing bias in `MoEBiasVar` and the experts of an unscanned layer, with scanned layers."""
    ckpt_dir = self.create_tempdir().full_path
    rng = np.random.default_rng(0)
    hf = {f"layers.{i}.ffn.gate.bias": rng.standard_normal((_EXPERTS,), np.float32) for i in range(_LAYERS)}
    hf |= {
        f"layers.0.ffn.experts.{e}.w1.weight": rng.standard_normal((_MLP, _EMBED), np.float32) for e in range(_EXPERTS)
    }
    safetensors.flax.save_file({k: jnp.asarray(v) for k, v in hf.items()}, os.path.join(ckpt_dir, "model.safetensors"))
    param_map = {
        "MoEBiasVar-decoder-layers-MoeBlock_0-gate-bias": [f"layers.{i}.ffn.gate.bias" for i in range(_LAYERS)],
        "params-decoder-layers_0-MoeBlock_0-wi_0": [f"layers.0.ffn.experts.{e}.w1.weight" for e in range(_EXPERTS)],
    }
    hook_map = {"params-decoder-layers_0-MoeBlock_0-wi_0": lambda x, shape: x.T}
    mesh = _mesh()
    shapes = {
        "MoEBiasVar.decoder.layers.MoeBlock_0.gate.bias": (_LAYERS, _EXPERTS),
        "params.decoder.layers_0.MoeBlock_0.wi_0": (_EXPERTS, _EMBED, _MLP),
    }
    target_tree = flax.traverse_util.unflatten_dict(
        {
            k: jax.ShapeDtypeStruct(s, jnp.float32, sharding=NamedSharding(mesh, _spec(s, mesh)))
            for k, s in shapes.items()
        },
        sep=".",
    )
    restored = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir, target_tree, param_map, hook_map, _config(scan_layers=True), min_bytes_to_split=0
    )
    self.assertEqual(set(restored), {"params", "MoEBiasVar"})
    np.testing.assert_array_equal(
        np.asarray(restored["MoEBiasVar"]["decoder"]["layers"]["MoeBlock_0"]["gate"]["bias"]),
        np.stack([hf[f"layers.{i}.ffn.gate.bias"] for i in range(_LAYERS)]),
    )
    np.testing.assert_array_equal(
        np.asarray(restored["params"]["decoder"]["layers_0"]["MoeBlock_0"]["wi_0"]),
        np.stack([hf[f"layers.0.ffn.experts.{e}.w1.weight"].T for e in range(_EXPERTS)]),
    )


if __name__ == "__main__":
  absltest.main()
