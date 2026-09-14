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

"""The NNX decoder logits guards must read the model_mode passed to __call__.

The vocab-tiling (and indexer warm-up) guards skip the output head only in TRAIN.
They must key off the model_mode argument, not the model_mode fixed at construction:
the same model invoked in TRAIN must skip the head (logits None) while invoked in a
serving mode must run it (real logits).
"""

import sys
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path


class DecoderLogitsGuardModelModeTest(unittest.TestCase):
  """A tiny pure-NNX model with vocab tiling, built once and called in two modes.

  Built in PREFILL so the attention KV cache is allocated (TRAIN construction leaves
  it None, which a serving call can't populate). Whether the output head runs must
  then depend solely on the call-arg model_mode.
  """

  def _model(self):
    """Build a tiny pure-NNX model with vocab tiling, constructed in PREFILL mode."""
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        base_emb_dim=32,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        base_mlp_dim=64,
        base_num_decoder_layers=2,
        head_dim=16,
        max_target_length=16,
        max_prefill_predict_length=8,
        per_device_batch_size=1,
        enable_checkpointing=False,
        scan_layers=False,
        num_vocab_tiling=2,
        pure_nnx=True,
        enable_nnx=True,
        pure_nnx_decoder=True,
    )
    model = model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))
    return cfg, model

  def test_guard_follows_call_arg_not_construction_mode(self):
    cfg, model = self._model()
    seq = cfg.max_prefill_predict_length
    toks = jnp.ones((1, seq), dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(seq), toks.shape)

    # Called in a serving mode, the output head runs -> real logits.
    logits_serving = model(toks, pos, model_mode=MODEL_MODE_PREFILL, enable_dropout=False)
    self.assertIsNotNone(logits_serving)
    self.assertEqual(logits_serving.shape[-1], cfg.vocab_size)

    # Called in TRAIN with vocab tiling on, the head is skipped -> logits None. Keying
    # off the construction mode (PREFILL) instead of the call-arg would run the head.
    logits_train = model(toks, pos, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)
    self.assertIsNone(logits_train)


if __name__ == "__main__":
  unittest.main()
