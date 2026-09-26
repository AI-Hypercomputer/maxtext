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

  def _load(self, model, scan_layers, dtype=jnp.float32, skip_hf=(), edit_maps=None, **load_kwargs):
    """Writes a tiny HF checkpoint and streams it into MaxText weights of `dtype`."""
    ckpt_dir = self.create_tempdir().full_path
    hf_tensors = _write_checkpoint(ckpt_dir, _hf_shapes(model), skip=skip_hf)
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
  )
  def test_matches_in_memory_transform(self, model, scan_layers, dtype, read_bytes_per_host):
    got, target_tree, hf_tensors, param_map, hook_map, config = self._load(
        model, scan_layers, dtype, read_bytes_per_host=read_bytes_per_host
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

  def _read_calls(self, **load_kwargs):
    """Streams the tiny Mixtral checkpoint and returns the request of each Orbax read call, in order."""
    real_load = hf_streaming_load.ocp_v1.load
    requests = []

    def spy(path, request, *args, **kwargs):
      requests.append(dict(request))
      return real_load(path, request, *args, **kwargs)

    with mock.patch.object(hf_streaming_load.ocp_v1, "load", side_effect=spy):
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


if __name__ == "__main__":
  absltest.main()
