# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for grouped attention metadata routing across decoders and layers."""

import sys
import types
import unittest
from unittest.mock import MagicMock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.models import qwen3, qwen3_5
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "grouped_attention_metadata_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 2,
    "attention": "dot_product",
    "max_target_length": 16,
    "base_emb_dim": 256,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 512,
    "max_prefill_predict_length": 4,
    "scan_layers": False,
}

_QWEN3_CONFIG = {
    "enable_checkpointing": False,
    "run_name": "qwen3_grouped_metadata_test",
    "model_name": "qwen3-next-80b-a3b",
    "max_target_length": 8,
    "base_emb_dim": 64,
    "base_num_decoder_layers": 2,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "head_dim": 32,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 32,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 4,
    "gdn_key_head_dim": 16,
    "gdn_value_head_dim": 16,
    "gdn_chunk_size": 4,
    "sparse_matmul": True,
    "megablox": False,
    "dtype": "float32",
    "weight_dtype": "float32",
}


def _make_config(base_dict=None, **overrides):
  """Return a pyconfig Config object suitable for unit tests."""
  if base_dict is None:
    base_dict = _BASE_CONFIG
  merged = {**base_dict, **overrides}
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], override_model_config=True, **merged)


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


class GroupedAttentionMetadataRoutingTest(unittest.TestCase):
  """Tests that GroupedAttentionMetadata dictionaries are correctly unpacked per layer."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()[:1]
    self.single_device_mesh = Mesh(np.array(devices), ("data",))

  def test_nnx_decoder_unscanned_routes_grouped_metadata_per_layer(self):
    cfg = _make_config(base_num_decoder_layers=2, scan_layers=False)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(
        config=cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        config=cfg,
        mesh=mesh,
        rngs=nnx.Rngs(params=0),
    )

    meta0 = {"is_meta0": jnp.zeros((1,))}
    meta1 = {"is_meta1": jnp.zeros((1,))}

    captured = []
    orig_call = LlamaDecoderLayer.__call__

    def wrap_call(layer_self, y, *args, **kwargs):
      del layer_self, args
      meta = kwargs.get("attention_metadata")
      if isinstance(meta, dict):
        if "is_meta0" in meta:
          captured.append("meta0")
        elif "is_meta1" in meta:
          captured.append("meta1")
        elif "layer.0" in meta:
          captured.append("original_dict")
        else:
          captured.append("unknown_dict")
      elif meta is None:
        captured.append(None)
      else:
        captured.append("other")
      return y, None

    LlamaDecoderLayer.__call__ = wrap_call

    ids = jnp.zeros((1, 4), dtype=jnp.int32)
    positions = jnp.zeros((1, 4), dtype=jnp.int32)
    seg_ids = jnp.full((1, 4), DECODING_ACTIVE_SEQUENCE_INDICATOR)

    try:
      # 1. Test routing with string key ("layer.0") and integer key (1)
      decoder(
          shared_embedding,
          ids,
          positions,
          decoder_segment_ids=seg_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          attention_metadata={"layer.0": meta0, 1: meta1},
      )
      self.assertEqual(captured, ["meta0", "meta1"])

      # 2. Test fallback when a layer key is missing in attention_metadata
      captured.clear()
      decoder(
          shared_embedding,
          ids,
          positions,
          decoder_segment_ids=seg_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          attention_metadata={"layer.0": meta0},
      )
      self.assertEqual(captured, ["meta0", None])

      # 3. Test non-dict attention_metadata (e.g. None)
      captured.clear()
      decoder(
          shared_embedding,
          ids,
          positions,
          decoder_segment_ids=seg_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          attention_metadata=None,
      )
      self.assertEqual(captured, [None, None])
    finally:
      LlamaDecoderLayer.__call__ = orig_call

  def test_qwen3_next_decoder_layer_unpacks_grouped_metadata(self):
    cfg = _make_config(base_dict=_QWEN3_CONFIG, base_num_decoder_layers=2)
    mesh = _make_mesh(cfg)
    layer = qwen3.Qwen3NextDecoderLayer(
        config=cfg,
        mesh=mesh,
        layer_idx=0,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(0),
    )
    meta = types.SimpleNamespace(id="qwen3_meta")
    layer.attention = MagicMock(return_value=(jnp.zeros((1, 4, 64)), None))
    layer.mlp = MagicMock(return_value=(jnp.zeros((1, 4, 64)), None))

    inputs = jnp.zeros((1, 4, 64), dtype=jnp.float32)

    # 1. String key "layer.0"
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata={"layer.0": meta},
    )
    self.assertIs(layer.attention.call_args.kwargs["attention_metadata"], meta)

    # 2. Integer key 0
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata={0: meta},
    )
    self.assertIs(layer.attention.call_args.kwargs["attention_metadata"], meta)

    # 3. Missing key
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata={"layer.1": meta},
    )
    self.assertIsNone(layer.attention.call_args.kwargs["attention_metadata"])

    # 4. Non-dict metadata
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata=meta,
    )
    self.assertIs(layer.attention.call_args.kwargs["attention_metadata"], meta)

  def test_qwen3_5_decoder_layer_unpacks_grouped_metadata(self):
    cfg = _make_config(base_dict=_QWEN3_CONFIG, decoder_block="qwen3_5", base_num_decoder_layers=2)
    mesh = _make_mesh(cfg)
    layer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg,
        mesh=mesh,
        layer_idx=0,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(0),
    )
    meta = types.SimpleNamespace(id="qwen3_5_meta")
    layer.attention = MagicMock(return_value=(jnp.zeros((1, 4, 64)), None))
    layer.mlp = MagicMock(return_value=(jnp.zeros((1, 4, 64)), None))

    inputs = jnp.zeros((1, 4, 64), dtype=jnp.float32)

    # 1. String key "layer.0"
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata={"layer.0": meta},
    )
    self.assertIs(layer.attention.call_args.kwargs["attention_metadata"], meta)

    # 2. Integer key 0
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata={0: meta},
    )
    self.assertIs(layer.attention.call_args.kwargs["attention_metadata"], meta)

    # 3. Missing key
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata={"layer.1": meta},
    )
    self.assertIsNone(layer.attention.call_args.kwargs["attention_metadata"])

    # 4. Non-dict metadata
    layer(
        inputs,
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        attention_metadata=meta,
    )
    self.assertIs(layer.attention.call_args.kwargs["attention_metadata"], meta)


if __name__ == "__main__":
  unittest.main()
