# Copyright 2023-2026 Google LLC
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

"""DeepSeek-V4 decode must match the train-mode forward pass.

MaxEngine pads the prompt to `max_prefill_predict_length`. The CSA, indexer and HCA
compressors prime their decode state (leftover buffer, accumulator index, entry count,
overlap prior) at prefill. If that priming counts pad tokens, the first decode-time
compressed blocks hold pad content at shifted positions, and decode drifts from the
forward pass once the first compressed block is emitted.

Decode also splits keys into a prefill cache and an AR cache. The sliding window must
be measured between true positions: the current token and recent decode keys stay
visible when the AR cache is longer than the window, and prompt keys leave the window
once the query is `sliding_window_size` tokens past them.

A small deepseek4 model with one CSA (ratio 4) and one HCA (ratio 8) layer is decoded
greedily; each step's logits are compared with a train-mode forward pass over the
same token sequence.
"""

import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
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
STEPS = 24
REL_TOL = 1e-3

_V4_SMALL = {
    "model_name": "deepseek4-tiny",
    "override_model_config": True,
    "base_num_decoder_layers": 4,
    "compress_ratios": [0, 0, 4, 8],
    "first_num_hash_layers": 1,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    # Dense indexer, so top-k selection cannot hide or add compressed blocks.
    "indexer_topk": 64,
    # Prompt of 5 padded to 16: both compressors see pad-only windows at prefill.
    "max_prefill_predict_length": 16,
    "max_target_length": 48,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "attention": "dot_product",
    "sparse_matmul": True,
    "use_tokamax_gmm": False,
    "megablox": False,
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "mla_naive_kvcache": False,
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
}


class DeepseekV4PaddedPrefillDecodeTest(unittest.TestCase):
  """Per-step decode logits must equal the forward-pass logits at the same position."""

  _rels_cache = {}

  @classmethod
  def tearDownClass(cls):
    cls._rels_cache.clear()
    super().tearDownClass()

  def test_decode_logits_match_forward_after_padded_prefill(self):
    self._check_decode_matches_forward(_V4_SMALL, STEPS)

  def test_decode_logits_match_forward_past_sliding_window(self):
    # AR cache (64 - 16 = 48 slots) longer than the window, and decode runs past
    # prompt + window, so both cache parts need true-position windowing.
    self._check_decode_matches_forward({**_V4_SMALL, "sliding_window_size": 8, "max_target_length": 64}, 30)

  def test_decode_logits_match_forward_when_compressor_cache_fills(self):
    # The HCA (ratio 8) emits decode blocks at positions 7, 15, 23 and 31; decode
    # continues to position 34, so the AR region must hold all of them.
    self._check_decode_matches_forward(_V4_SMALL, 30)

  @classmethod
  def _compute_decode_rels(cls, overrides, max_steps):
    """Runs `max_steps` greedy decode steps and returns per-step relative max-abs diffs vs forward pass."""
    cache_key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in overrides.items()))
    cached = cls._rels_cache.get(cache_key)
    if cached is not None and len(cached) >= max_steps + 1:
      return cached

    cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **overrides)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(
          cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0, dropout=0)
      )
    _, params_state, _ = nnx.split(model, nnx.Param, ...)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    padded = jnp.asarray(PROMPT + [0] * (cfg.max_prefill_predict_length - len(PROMPT)), dtype=jnp.int32)
    prefix, first_token = engine.prefill(params=params, padded_tokens=padded, true_length=len(PROMPT))
    decode_logits_dev = [jnp.reshape(prefix["logits"], (-1,))[-cfg.vocab_size :]]
    step_tokens_dev = [first_token.data[0, 0]]
    decode_state = engine.init_decode_state()
    decode_state = engine.insert(prefix, decode_state, slot=0)
    for _ in range(max_steps):
      decode_state, result = engine.generate(params, decode_state)
      decode_logits_dev.append(jnp.reshape(decode_state["logits"][0], (-1,))[-cfg.vocab_size :])
      step_tokens_dev.append(result.data[0, 0])

    decode_logits_np, step_tokens_np = jax.device_get((jnp.stack(decode_logits_dev), jnp.stack(step_tokens_dev)))
    decode_logits = [np.asarray(row, dtype=np.float64) for row in decode_logits_np]
    tokens = list(PROMPT) + [int(t) for t in step_tokens_np]

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      train_model = model_creation_utils.create_model(
          cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=0)
      )
    nnx.update(train_model, params_state)
    train_graphdef, train_state = nnx.split(train_model)
    seq = cfg.max_target_length
    batch = cfg.micro_batch_size_to_train_on
    ids = jnp.tile(jnp.asarray([tokens + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    positions = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    segment_ids = jnp.tile(jnp.asarray([[1] * len(tokens) + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))

    @jax.jit
    def _forward_jit(state, ids_in, pos_in, seg_in):
      m = nnx.merge(train_graphdef, state)
      return m(ids_in, pos_in, decoder_segment_ids=seg_in, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      ref = _forward_jit(train_state, ids, positions, segment_ids)
    ref = np.asarray(jax.device_get(ref[0]), dtype=np.float64)

    rels = []
    for k, dec in enumerate(decode_logits):
      row = ref[len(PROMPT) - 1 + k]
      rels.append(float(np.max(np.abs(dec - row)) / np.max(np.abs(row))))
    cls._rels_cache[cache_key] = rels
    return rels

  def _check_decode_matches_forward(self, overrides, steps):
    """Greedy-decodes `steps` tokens after a padded prefill and compares each step with the forward pass."""
    max_steps = max(steps, 30) if overrides == _V4_SMALL else steps
    rels = self._compute_decode_rels(overrides, max_steps)[: steps + 1]
    worst = int(np.argmax(rels))
    self.assertLessEqual(
        rels[worst],
        REL_TOL,
        f"decode step {worst} (position {len(PROMPT) - 1 + worst}) rel max-abs {rels[worst]:.3e}; "
        f"per-step rel: {[f'{r:.1e}' for r in rels]}",
    )


if __name__ == "__main__":
  unittest.main()
