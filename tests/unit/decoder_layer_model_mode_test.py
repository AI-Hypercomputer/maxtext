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

"""A decoder layer must attend in the mode it is called with, not the one it was built in.

Under NNX a model object is built once and reused across modes — MaxEngine builds it
in PREFILL and then passes MODEL_MODE_AUTOREGRESSIVE per call while decoding. A layer
that forwards its construction-time `self.model_mode` to attention instead of the
`model_mode` argument therefore runs the prefill KV-cache path during decode and
produces wrong tokens from the first generated one, with no error anywhere.

The check stops at the attention boundary and asserts on the mode that arrives there,
so it stays cheap and does not depend on any particular attention implementation.
"""

import sys
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.layers import attention_mla, attentions
from maxtext.models import deepseek, gemma, gemma2, llama2, mistral, qwen2, qwen3
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# Config knobs shared by every family; small enough that construction is the only cost.
_COMMON = {
    "base_emb_dim": 64,
    "base_mlp_dim": 64,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 4,
    "base_num_decoder_layers": 3,
    "vocab_size": 64,
    "max_prefill_predict_length": 8,
    "max_target_length": 16,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "attention": "dot_product",
    "sparse_matmul": False,
    "dtype": "float32",
    "weight_dtype": "float32",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
    "pure_nnx": True,
}

_DEEPSEEK = {
    "model_name": "deepseek2-16b",
    "override_model_config": True,
    "mla_naive_kvcache": False,
    "first_num_dense_layers": 1,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_experts": 1,
}

# Each entry is (case name, layer class, extra config, extra constructor kwargs).
_LAYERS = [
    ("deepseek_dense", deepseek.DeepSeekDenseLayer, _DEEPSEEK, {}),
    ("deepseek_moe", deepseek.DeepSeekMoELayer, _DEEPSEEK, {}),
    ("llama2", llama2.LlamaDecoderLayer, {}, {}),
    ("mistral", mistral.MistralDecoderLayer, {"decoder_block": "mistral"}, {}),
    ("gemma", gemma.GemmaDecoderLayer, {"decoder_block": "gemma"}, {}),
    ("gemma2", gemma2.Gemma2DecoderLayer, {"decoder_block": "gemma2"}, {}),
    ("qwen2", qwen2.Qwen2DecoderLayer, {"decoder_block": "qwen2"}, {"quant": None}),
    ("qwen3", qwen3.Qwen3DecoderLayer, {"decoder_block": "qwen3"}, {"quant": None}),
]


def _mesh(cfg):
  """Builds the mesh over every available device.

  ici_fsdp_parallelism keeps its base.yml default of -1, so the fsdp axis absorbs
  however many devices the host has. Pinning the ICI axes instead would fix the mesh
  at one device and fail everywhere except a single-device runner.

  Args:
    cfg: Model config.

  Returns:
    The device mesh.
  """
  return jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)


class _StopAtAttention(Exception):
  """Raised by the spy so the forward pass ends once the mode has been observed."""


class DecoderLayerModelModeTest(parameterized.TestCase):
  """Every decoder layer must forward the call-time model_mode to its attention."""

  def _mode_seen_by_attention(self, layer_cls, extra_config, ctor_kwargs, build_mode, call_mode):
    """Builds a layer in one mode and calls it in another.

    Args:
      layer_cls: Decoder layer class to instantiate.
      extra_config: Config overrides this family needs, on top of _COMMON.
      ctor_kwargs: Extra constructor arguments this family requires.
      build_mode: Model mode the layer is constructed with.
      call_mode: Model mode the layer is called with.

    Returns:
      The model mode that reached the attention module.
    """
    cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **(_COMMON | extra_config))
    mesh = _mesh(cfg)
    seen = []

    def spy(_self, *args, **kwargs):
      # model_mode is the 5th positional parameter of Attention.__call__, but every
      # in-tree caller passes it by keyword. Accept both so the spy is not brittle.
      seen.append(kwargs.get("model_mode", args[4] if len(args) > 4 else None))
      raise _StopAtAttention()

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      layer = layer_cls(
          config=cfg,
          model_mode=build_mode,
          mesh=mesh,
          rngs=nnx.Rngs(params=0, dropout=0),
          **ctor_kwargs,
      )
      batch = cfg.micro_batch_size_to_train_on
      inputs = jnp.zeros((batch, 1, cfg.emb_dim), dtype=jnp.float32)
      positions = jnp.zeros((batch, 1), dtype=jnp.int32)
      segment_ids = jnp.ones((batch, 1), dtype=jnp.int32)
      with (
          mock.patch.object(attentions.Attention, "__call__", spy),
          mock.patch.object(attention_mla.MLA, "__call__", spy),
      ):
        try:
          layer(inputs, segment_ids, positions, True, call_mode)
        except _StopAtAttention:
          pass

    self.assertEqual(len(seen), 1, f"expected exactly one attention call, saw {len(seen)}")
    return seen[0]

  @parameterized.named_parameters(*_LAYERS)
  def test_decode_call_overrides_prefill_construction(self, layer_cls, extra_config, ctor_kwargs):
    """Checks a layer built for prefill and called for decode, as MaxEngine does."""
    seen = self._mode_seen_by_attention(
        layer_cls, extra_config, ctor_kwargs, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
    )
    self.assertEqual(
        seen,
        MODEL_MODE_AUTOREGRESSIVE,
        "attention ran in the layer's construction-time mode; decode would use the prefill KV-cache path",
    )

  @parameterized.named_parameters(*_LAYERS)
  def test_prefill_call_overrides_decode_construction(self, layer_cls, extra_config, ctor_kwargs):
    """Checks the reverse direction, so a layer cannot pass by hardcoding either mode."""
    seen = self._mode_seen_by_attention(
        layer_cls, extra_config, ctor_kwargs, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
    )
    self.assertEqual(seen, MODEL_MODE_PREFILL, "attention ran in the layer's construction-time mode")


if __name__ == "__main__":
  unittest.main()
