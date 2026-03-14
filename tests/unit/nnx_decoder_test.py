# Copyright 2026 Google LLC
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

"""Unit tests for nnx_decoders module.

Tests cover:
  - deepstack_process: pure-JAX helper for injecting visual embeddings
  - NNXDecoderLayer: single transformer decoder layer (init + forward)
  - NNXDecoder: decoder stack utilities (get_decoder_layers, get_norm_layer,
                get_remat_policy, minimal_policy, and full forward pass)
"""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, EP_AS_CONTEXT
import sys
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN, DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.layers import linears
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder, NNXDecoderLayer, deepstack_process
from maxtext.layers.normalizations import RMSNorm
from maxtext.models.gpt3 import Gpt3LayerNorm
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_decoupled_parallelism_overrides, get_test_config_path
# Assuming the following imports are already at the top of your test file:
# from maxtext.layers.nnx_decoders import decoder_as_linen
# from maxtext.common.common_types import MODEL_MODE_TRAIN

class TestNNXDecoderLayerLogicalAxesUnmocked(unittest.TestCase):
    """
    Executes pure, unmocked forward passes through NNXDecoderLayer to 
    guarantee coverage of the logical_axis_names assignment block.
    """

    def setUp(self):
        super().setUp()
        self.rngs = nnx.Rngs(params=0, dropout=1)
        self.base_cfg = _make_config()
        self.mesh = _make_mesh(self.base_cfg)

    def _make_dummy_inputs(self, cfg):
        batch = cfg.global_batch_size_to_train_on
        seq_len = cfg.max_target_length
        emb_dim = cfg.emb_dim
        
        # Use jnp.ones to ensure stable, non-stochastic arrays for the forward pass
        inputs = jnp.ones((batch, seq_len, emb_dim), dtype=cfg.dtype)
        segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
        
        return inputs, segment_ids, positions

    def test_forward_pass_prefill_mode(self):
        """Forces execution of: if self.model_mode == MODEL_MODE_PREFILL"""
        cfg = _make_config()
        layer = NNXDecoderLayer(
            config=cfg, mesh=self.mesh, model_mode=MODEL_MODE_PREFILL, rngs=self.rngs
        )
        inputs, segment_ids, positions = self._make_dummy_inputs(cfg)

        # A real forward pass ensures all sharding and normalization lines are executed
        out, _ = layer(
            inputs, segment_ids, positions, deterministic=True, model_mode=MODEL_MODE_PREFILL
        )
        self.assertEqual(out.shape, inputs.shape)

    def test_forward_pass_ep_as_context(self):
        """Forces execution of: elif self.config.expert_shard_attention_option == EP_AS_CONTEXT..."""
        cfg = _make_config(expert_shard_attention_option=EP_AS_CONTEXT)
        layer = NNXDecoderLayer(
            config=cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs
        )
        inputs, segment_ids, positions = self._make_dummy_inputs(cfg)

        out, _ = layer(
            inputs, segment_ids, positions, deterministic=True, model_mode=MODEL_MODE_TRAIN
        )
        self.assertEqual(out.shape, inputs.shape)

    def test_forward_pass_default_axes(self):
        """Forces execution of the default 'else' fallback."""
        cfg = _make_config(expert_shard_attention_option="none")
        layer = NNXDecoderLayer(
            config=cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs
        )
        inputs, segment_ids, positions = self._make_dummy_inputs(cfg)

        out, _ = layer(
            inputs, segment_ids, positions, deterministic=True, model_mode=MODEL_MODE_TRAIN
        )
        self.assertEqual(out.shape, inputs.shape)

if __name__ == "__main__":
  unittest.main()