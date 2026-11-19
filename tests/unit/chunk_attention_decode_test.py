# Copyright 2023-2026 Google LLC
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

"""Decode with CHUNK attention (Llama 4) must match a train-mode forward over the same tokens.

Single-token decode attends over the prefill cache and the AR cache separately. The chunk of the
query and of every key has to come from true token positions: the prefill cache always has
`max_prefill_predict_length` slots, and the AR cache `max_target_length - max_prefill_predict_length`,
so the last slot of either part is not the query's position once a part is longer than the chunk or
decoding crosses a chunk boundary.
"""

import sys
import types
import unittest

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    AttentionType,
)
from maxtext.configs import pyconfig
from maxtext.layers.attention_op import AttentionOp
from maxtext.utils import maxtext_utils, model_creation_utils

from maxtext.inference.maxengine import maxengine
from tests.utils.test_helpers import get_test_config_path

# In cpu-unit where jetstream is not installed, gcloud_stub's ResultTokens is a plain class;
# register it as a JAX pytree so MaxEngine._prefill_jit and _generate_jit can return it.
if "_jetstream_stubs" in getattr(maxengine.engine_api.ResultTokens, "__qualname__", ""):
  try:
    jax.tree_util.register_pytree_node(
        maxengine.engine_api.ResultTokens,
        lambda r: ((r.data, r.log_prob), (r.tokens_idx, r.valid_idx, r.length_idx, r.samples_per_slot)),
        lambda aux, children: maxengine.engine_api.ResultTokens(
            data=children[0],
            log_prob=children[1],
            tokens_idx=aux[0],
            valid_idx=aux[1],
            length_idx=aux[2],
            samples_per_slot=aux[3],
        ),
    )
  except ValueError:
    pass

PROMPT = [3, 17, 42, 5, 9]
STEPS = 30
REL_TOL = 1e-3

_COMMON = {
    "override_model_config": True,
    # Layers 0-2 use RoPE and CHUNK attention, layer 3 is the NoPE (global) layer.
    "base_num_decoder_layers": 4,
    "base_emb_dim": 128,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 2,
    "head_dim": 32,
    "vocab_size": 128,
    "base_mlp_dim": 256,
    "base_moe_mlp_dim": 128,
    "num_experts": 4,
    "sparse_matmul": False,
    "megablox": False,
    "use_multimodal": False,
    # A prefill cache of 16 slots is longer than the chunk of 8, and the 30 decode steps cross
    # four chunk boundaries.
    "chunk_attn_window_size": 8,
    "max_prefill_predict_length": 16,
    "max_target_length": 64,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "attention": "dot_product",
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
}


class ChunkAttentionDecodeTest(parameterized.TestCase):
  """Per-step decode logits must equal the forward-pass logits at the same position."""

  @parameterized.named_parameters(
      ("llama4_scout", {"model_name": "llama4-17b-16e"}),
      ("llama4_maverick", {"model_name": "llama4-17b-128e"}),
  )
  def test_decode_matches_forward_across_chunks(self, overrides):
    cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **_COMMON, **overrides)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0))
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    # Default init is close to degenerate for attention; random weights make the chunks visible.
    leaves, treedef = jax.tree.flatten(params_state)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    leaves = [jax.random.normal(k, x.shape, x.dtype) * 0.2 for k, x in zip(keys, leaves)]
    params_state = jax.tree.unflatten(treedef, leaves)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    padded = jnp.asarray(PROMPT + [0] * (cfg.max_prefill_predict_length - len(PROMPT)), dtype=jnp.int32)
    prefix, first_token = engine.prefill(params=params, padded_tokens=padded, true_length=len(PROMPT))
    tokens = list(PROMPT) + [int(first_token.data[0, 0])]
    decode_logits = [np.asarray(prefix["logits"], np.float64).reshape(-1)[-cfg.vocab_size :]]
    state = engine.insert(prefix, engine.init_decode_state(), slot=0)
    for _ in range(STEPS):
      state, result = engine.generate(params, state)
      decode_logits.append(np.asarray(state["logits"][0], np.float64).reshape(-1)[-cfg.vocab_size :])
      tokens.append(int(result.data[0, 0]))

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      train_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0))
    nnx.update(train_model, params_state)
    seq, batch = cfg.max_target_length, cfg.micro_batch_size_to_train_on
    ids = jnp.tile(jnp.asarray([tokens + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    positions = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    segment_ids = jnp.tile(jnp.asarray([[1] * len(tokens) + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      ref = train_model(
          ids, positions, decoder_segment_ids=segment_ids, enable_dropout=False, model_mode=MODEL_MODE_TRAIN
      )
    ref = np.asarray(ref[0] if isinstance(ref, tuple) else ref, np.float64)[0]

    rels = [
        float(np.max(np.abs(d - ref[len(PROMPT) - 1 + k])) / np.max(np.abs(ref[len(PROMPT) - 1 + k])))
        for k, d in enumerate(decode_logits)
    ]
    worst = int(np.argmax(rels))
    self.assertLessEqual(rels[worst], REL_TOL, f"decode step {worst} rel max-abs {rels[worst]:.3e}")


class ChunkDecodeMaskTest(parameterized.TestCase):
  """The decode-mode CHUNK mask allows exactly the keys of the query's chunk, by true position."""

  CHUNK = 8192

  def _make_op(self, max_target_length):
    config = types.SimpleNamespace(
        context_parallel_load_balance=False,
        context_sharding="context",
        shard_mode="auto",
        debug_sharding=False,
        eval_interval=-1,
    )
    return AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=max_target_length,
        mesh=Mesh(jax.devices()[:1], ["context"]),
        attention_kernel="dot_product",
        attention_type=AttentionType.CHUNK,
        chunk_attn_window_size=self.CHUNK,
    )

  def _allowed(self, op, cache_slots, valid_slots, query_position, kv_cache_part):
    """Returns the slot indices allowed by the decode attention mask."""
    segment_ids = np.zeros((1, cache_slots), np.int32)
    segment_ids[0, valid_slots] = DECODING_ACTIVE_SEQUENCE_INDICATOR
    mask = op.generate_attention_mask(
        jnp.zeros((1, 1, 1, 8)),
        jnp.zeros((1, cache_slots, 1, 8)),
        jnp.asarray(segment_ids),
        MODEL_MODE_AUTOREGRESSIVE,
        segment_positions=jnp.full((1, 1), query_position, jnp.int32),
        kv_cache_part=kv_cache_part,
    )
    return np.flatnonzero(np.asarray(mask == 0.0).reshape(-1))

  @parameterized.named_parameters(
      # (prefill cache slots, AR cache slots, prompt length, decode step): the query is the token
      # at position prompt + step, its key in AR slot `step`.
      ("prefill_cache_longer_than_chunk", 16384, 1024, 1000, 0),
      ("decode_crosses_chunk_boundary", 8192, 1024, 8180, 12),
      ("ar_cache_longer_than_chunk", 1024, 16384, 1000, 7191),
      ("short_buffers", 64, 1984, 50, 100),
  )
  def test_decode_mask_uses_token_positions(self, prefill_slots, ar_slots, prompt, step):
    op = self._make_op(prefill_slots + ar_slots)
    query_position = prompt + step
    chunk = query_position // self.CHUNK
    allowed_prefill = self._allowed(op, prefill_slots, np.arange(prompt), query_position, "prefill")
    allowed_ar = self._allowed(op, ar_slots, np.arange(step + 1), query_position, "ar")
    expected_prefill = [k for k in range(prompt) if k // self.CHUNK == chunk]
    expected_ar = [j for j in range(step + 1) if (prompt + j) // self.CHUNK == chunk]
    np.testing.assert_array_equal(allowed_prefill, expected_prefill)
    np.testing.assert_array_equal(allowed_ar, expected_ar)


if __name__ == "__main__":
  unittest.main()
