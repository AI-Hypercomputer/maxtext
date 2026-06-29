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

"""Tests for the streaming/low-memory unscanned DeepSeek-family converter (#4071).

Uses a tiny synthetic kimi-k2.6-style checkpoint (compressed int4 routed experts,
`language_model.` key prefix, a vision key that must be dropped) to check that:
  - no tensor data is read while indexing the shards (the converter used to buffer
    every dequantized tensor for all shards up front, needing ~2.5 TB RAM for K2.6)
  - each source tensor is read at most once
  - low-memory (disk-spilled) conversion is bit-identical to the in-memory one
  - a checkpoint saved from disk-spilled leaves restores to the expected values

Not run in GitHub runners (depends on torch).
"""

import collections
import os
import pathlib
import shutil
import tempfile
import unittest
import zlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import psutil
import pytest
import torch
from safetensors.torch import save_file

from maxtext.checkpoint_conversion.standalone_scripts import convert_deepseek_family_ckpt as ds_ckpt
from maxtext.checkpoint_conversion.standalone_scripts import convert_deepseek_family_unscanned_ckpt as unscanned_ckpt
from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from maxtext.utils import max_logging

_TINY_MODEL_SIZE = "tiny-kimi-test"
_TINY_PARAMS = {
    "num_layers": 3,  # 1 dense + 2 MoE
    "first_num_dense_layers": 1,
    "base_num_query_heads": 2,
    "base_emb_dim": 64,
    "num_experts": 4,
    "q_lora_rank": 16,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 4,
    "v_head_dim": 8,
    "has_mtp": False,
    "compressed_int4": True,
    "hf_key_prefix": "language_model.",
}
_VOCAB = 128
_DENSE_INTER = 96
_MOE_INTER = 32


def _rng_for(key):
  """Deterministic per-key RNG so expected values can be recomputed independently."""
  return np.random.default_rng(zlib.crc32(key.encode()))


def _bf16(key, shape):
  return torch.from_numpy(_rng_for(key).standard_normal(shape).astype(np.float32)).to(torch.bfloat16)


def _packed_int4(key, out_features, in_features):
  """compressed-tensors pack-quantized layout: int32 [out, in/8], bf16 scale [out, in/32], shape [2]."""
  rng = _rng_for(key)
  packed = torch.from_numpy(
      rng.integers(-(2**31), 2**31 - 1, size=(out_features, in_features // 8), dtype=np.int64).astype(np.int32)
  )
  scale = torch.from_numpy((rng.standard_normal((out_features, in_features // 32)).astype(np.float32) * 0.01)).to(
      torch.bfloat16
  )
  shape = torch.tensor([out_features, in_features], dtype=torch.int64)
  return packed, scale, shape


def _attention_tensors(params, layer_idx):
  """All bf16 tensors for one layer's attention + norms (HF [out, in] layout)."""
  emb = params["base_emb_dim"]
  num_heads = params["base_num_query_heads"]
  prefix = f"model.layers.{layer_idx}"
  return {
      f"{prefix}.input_layernorm.weight": _bf16(f"{prefix}.iln", (emb,)),
      f"{prefix}.post_attention_layernorm.weight": _bf16(f"{prefix}.pln", (emb,)),
      f"{prefix}.self_attn.q_a_proj.weight": _bf16(f"{prefix}.qa", (params["q_lora_rank"], emb)),
      f"{prefix}.self_attn.q_a_layernorm.weight": _bf16(f"{prefix}.qan", (params["q_lora_rank"],)),
      f"{prefix}.self_attn.q_b_proj.weight": _bf16(
          f"{prefix}.qb", (num_heads * (params["qk_nope_head_dim"] + params["qk_rope_head_dim"]), params["q_lora_rank"])
      ),
      f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": _bf16(
          f"{prefix}.kva", (params["kv_lora_rank"] + params["qk_rope_head_dim"], emb)
      ),
      f"{prefix}.self_attn.kv_a_layernorm.weight": _bf16(f"{prefix}.kvn", (params["kv_lora_rank"],)),
      f"{prefix}.self_attn.kv_b_proj.weight": _bf16(
          f"{prefix}.kvb", (num_heads * (params["qk_nope_head_dim"] + params["v_head_dim"]), params["kv_lora_rank"])
      ),
      f"{prefix}.self_attn.o_proj.weight": _bf16(f"{prefix}.o", (emb, num_heads * params["v_head_dim"])),
  }


def _write_tiny_hf_checkpoint(params, out_dir):
  """Synthesizes a tiny multi-shard kimi-k2.6-style HF checkpoint."""
  emb = params["base_emb_dim"]
  pfx = params["hf_key_prefix"]
  shards = []

  # shard 0: embeddings + the dense layer + a vision key that must be dropped
  shard0 = {pfx + "model.embed_tokens.weight": _bf16("embed", (_VOCAB, emb))}
  shard0.update({pfx + k: v for k, v in _attention_tensors(params, 0).items()})
  shard0[pfx + "model.layers.0.mlp.gate_proj.weight"] = _bf16("d0.gate", (_DENSE_INTER, emb))
  shard0[pfx + "model.layers.0.mlp.up_proj.weight"] = _bf16("d0.up", (_DENSE_INTER, emb))
  shard0[pfx + "model.layers.0.mlp.down_proj.weight"] = _bf16("d0.down", (emb, _DENSE_INTER))
  shard0["vision_tower.patch_embed.proj.weight"] = _bf16("vision", (8, 8))
  shards.append(shard0)

  # one shard per MoE layer
  for layer_idx in range(params["first_num_dense_layers"], params["num_layers"]):
    shard = {pfx + k: v for k, v in _attention_tensors(params, layer_idx).items()}
    prefix = f"model.layers.{layer_idx}"
    shard[pfx + f"{prefix}.mlp.gate.weight"] = _bf16(f"{prefix}.rgate", (params["num_experts"], emb))
    shard[pfx + f"{prefix}.mlp.gate.e_score_correction_bias"] = _bf16(f"{prefix}.rbias", (params["num_experts"],))
    shard[pfx + f"{prefix}.mlp.shared_experts.gate_proj.weight"] = _bf16(f"{prefix}.sg", (_MOE_INTER, emb))
    shard[pfx + f"{prefix}.mlp.shared_experts.up_proj.weight"] = _bf16(f"{prefix}.su", (_MOE_INTER, emb))
    shard[pfx + f"{prefix}.mlp.shared_experts.down_proj.weight"] = _bf16(f"{prefix}.sd", (emb, _MOE_INTER))
    for expert_idx in range(params["num_experts"]):
      for proj, (out_features, in_features) in (
          ("gate_proj", (_MOE_INTER, emb)),
          ("up_proj", (_MOE_INTER, emb)),
          ("down_proj", (emb, _MOE_INTER)),
      ):
        base = f"{prefix}.mlp.experts.{expert_idx}.{proj}"
        packed, scale, shape = _packed_int4(base, out_features, in_features)
        shard[pfx + base + ".weight_packed"] = packed
        shard[pfx + base + ".weight_scale"] = scale
        shard[pfx + base + ".weight_shape"] = shape
    shards.append(shard)

  # last shard: final norm + lm_head
  shards.append(
      {
          pfx + "model.norm.weight": _bf16("norm", (emb,)),
          pfx + "lm_head.weight": _bf16("lmhead", (_VOCAB, emb)),
      }
  )

  for i, shard in enumerate(shards):
    save_file(shard, pathlib.Path(out_dir) / f"model-{i+1:05d}-of-{len(shards):05d}.safetensors")


def _flatten(tree, prefix=""):
  out = {}
  if isinstance(tree, dict):
    for key, value in tree.items():
      out.update(_flatten(value, f"{prefix}/{key}"))
  elif tree is not None:
    out[prefix] = np.asarray(tree)
  return out


class _CountingHandle:
  """Wraps a safetensors handle, recording every get_tensor call."""

  def __init__(self, inner, path, read_log, assembly_started):
    self._inner = inner
    self._path = path
    self._read_log = read_log
    self._assembly_started = assembly_started

  def keys(self):
    return self._inner.keys()

  def get_tensor(self, name):
    self._read_log.append((os.path.basename(str(self._path)), name, bool(self._assembly_started)))
    return self._inner.get_tensor(name)

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return self._inner.__exit__(*exc)


@pytest.mark.cpu_only
class ConvertDeepseekUnscannedLowMemoryTest(unittest.TestCase):
  """Streaming + disk-spill behavior of convert_deepseek_family_unscanned_ckpt."""

  @classmethod
  def setUpClass(cls):
    cls.tmp_dir = tempfile.mkdtemp(prefix="unscanned_low_memory_test_")
    cls.hf_dir = os.path.join(cls.tmp_dir, "hf")
    os.makedirs(cls.hf_dir)
    _write_tiny_hf_checkpoint(_TINY_PARAMS, cls.hf_dir)
    ds_ckpt.MODEL_PARAMS_DICT[_TINY_MODEL_SIZE] = _TINY_PARAMS

  @classmethod
  def tearDownClass(cls):
    ds_ckpt.MODEL_PARAMS_DICT.pop(_TINY_MODEL_SIZE, None)
    shutil.rmtree(cls.tmp_dir, ignore_errors=True)

  def _convert(self, spill_dir=None):
    # pylint: disable=protected-access
    if spill_dir is None:
      return unscanned_ckpt._convert_to_jax_weights(self.hf_dir, _TINY_MODEL_SIZE, psutil.Process())
    return unscanned_ckpt._convert_to_jax_weights(self.hf_dir, _TINY_MODEL_SIZE, psutil.Process(), spill_dir)

  def test_no_tensor_reads_before_assembly_and_no_rereads(self):
    """Tensor data must be streamed during assembly, not buffered while scanning shards.

    The converter used to load (and dequantize) every tensor of every shard into one
    dict before assembling the output pytree, so peak RSS was ~the whole dequantized
    model (~2.3 TB for kimi-k2.6); see #4071.
    """
    read_log = []
    assembly_started = []
    original_safe_open = unscanned_ckpt.safe_open
    original_log = max_logging.log

    def counting_safe_open(path, *args, **kwargs):
      return _CountingHandle(original_safe_open(path, *args, **kwargs), path, read_log, assembly_started)

    def phase_marking_log(message, *args, **kwargs):
      if isinstance(message, str) and message.startswith("Processing decoder norm scale"):
        assembly_started.append(True)
      return original_log(message, *args, **kwargs)

    unscanned_ckpt.safe_open = counting_safe_open
    max_logging.log = phase_marking_log
    try:
      self._convert()
    finally:
      unscanned_ckpt.safe_open = original_safe_open
      max_logging.log = original_log

    self.assertTrue(read_log, "expected the converter to read tensors")
    reads_before_assembly = [(path, name) for path, name, in_assembly in read_log if not in_assembly]
    self.assertEqual(
        reads_before_assembly, [], f"{len(reads_before_assembly)} tensors were buffered before assembly began"
    )
    read_counts = collections.Counter((path, name) for path, name, _ in read_log)
    rereads = {key: count for key, count in read_counts.items() if count > 1}
    self.assertEqual(rereads, {}, "each source tensor should be read at most once")

  def test_low_memory_pytree_is_bit_identical(self):
    """Disk-spilled conversion must produce the same pytree as the in-memory one."""
    reference = _flatten(self._convert())
    spill_dir = os.path.join(self.tmp_dir, "spill")
    os.makedirs(spill_dir, exist_ok=True)
    spilled = self._convert(spill_dir=spill_dir)
    for leaf in (
        spilled["token_embedder"]["embedding"],
        spilled["decoder"]["moe_layers_0"]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"],
    ):
      self.assertIsInstance(leaf, np.memmap)
      self.assertFalse(leaf.flags.writeable)
    spilled = _flatten(spilled)

    self.assertEqual(set(reference), set(spilled))
    for key, want in reference.items():
      self.assertEqual(want.dtype, spilled[key].dtype, key)
      np.testing.assert_array_equal(want, spilled[key], err_msg=key)

  def test_low_memory_checkpoint_restores_expected_values(self):
    """An Orbax checkpoint saved from disk-spilled leaves restores the expected weights."""
    import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

    spill_dir = os.path.join(self.tmp_dir, "spill_save")
    os.makedirs(spill_dir, exist_ok=True)
    ckpt_dir = os.path.join(self.tmp_dir, "ckpt")
    save_weights_to_checkpoint(ckpt_dir, self._convert(spill_dir=spill_dir), 1, True, True)
    restored = _flatten(ocp.PyTreeCheckpointer().restore(os.path.join(ckpt_dir, "0", "items")))

    emb = _TINY_PARAMS["base_emb_dim"]

    def as_np_f16(tensor):
      return tensor.to(torch.float16).numpy()

    np.testing.assert_array_equal(
        restored["/params/params/token_embedder/embedding"], as_np_f16(_bf16("embed", (_VOCAB, emb)))
    )
    np.testing.assert_array_equal(
        restored["/params/params/decoder/logits_dense/kernel"], as_np_f16(_bf16("lmhead", (_VOCAB, emb))).transpose()
    )
    np.testing.assert_array_equal(restored["/params/params/decoder/decoder_norm/scale"], as_np_f16(_bf16("norm", (emb,))))
    np.testing.assert_array_equal(
        restored["/params/params/decoder/dense_layers_0/mlp/wi_0/kernel"],
        as_np_f16(_bf16("d0.gate", (_DENSE_INTER, emb))).transpose(),
    )
    # routed-expert int4 path: HF layer 1 -> moe_layers_0, expert 2, up_proj -> wi_1
    packed, scale, shape = _packed_int4("model.layers.1.mlp.experts.2.up_proj", _MOE_INTER, emb)
    expected_expert = as_np_f16(ds_ckpt.dequantize_pack_quantized_int4(packed, scale, shape.tolist())).transpose()
    np.testing.assert_array_equal(
        restored["/params/params/decoder/moe_layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_1"][2], expected_expert
    )
    # the vision tower key must not leak into the text-only checkpoint
    self.assertFalse([k for k in restored if "vision" in k.lower()])


if __name__ == "__main__":
  unittest.main()
