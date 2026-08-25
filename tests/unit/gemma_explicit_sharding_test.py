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

"""The Gemma decoder layers must produce the same activations under both shard modes.

Under `shard_mode=auto` a missing sharding annotation costs nothing — GSPMD infers a
layout. Under `shard_mode=explicit` the same omission is a hard `ShardingTypeError`,
because JAX type-checks the sharding of every operation. Each Gemma layer therefore has
to hand its sub-modules an `out_sharding` (and, for the MLP, an `intermediate_sharding`)
rather than rely on inference.

These tests run one forward pass per layer family in each mode and compare the results.
A regression shows up either as an exception from the explicit run or as a numerical
difference, which is what a wrong — as opposed to merely absent — annotation produces.
"""

import os
import sys
import unittest

# A single-device mesh shards nothing, so every op is trivially well-typed and the test
# would pass even with the annotations removed. Ask XLA for eight host devices before
# JAX initializes its backend so the fsdp axis actually splits the activations.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from absl.testing import parameterized  # pylint: disable=wrong-import-position
from flax import nnx  # pylint: disable=wrong-import-position
from flax.linen import partitioning as nn_partitioning  # pylint: disable=wrong-import-position

# pylint: disable=wrong-import-position
from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.configs import pyconfig
from maxtext.models import gemma, gemma2, gemma3
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# Small enough that a forward pass is cheap, but wide enough that emb_dim and mlp_dim
# stay divisible by the mesh axes on a multi-device runner.
_COMMON = {
    "base_emb_dim": 64,
    "base_mlp_dim": 128,
    "base_num_query_heads": 8,
    "base_num_kv_heads": 8,
    "base_num_decoder_layers": 2,
    "head_dim": 16,
    "vocab_size": 64,
    "max_target_length": 16,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "attention": "dot_product",
    "dtype": "float32",
    "weight_dtype": "float32",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
    "pure_nnx": True,
}

# Gemma 3 reads per-layer attention settings (the local/global pattern, rope scaling)
# off the named model config, so it cannot run under the placeholder "default" name.
_GEMMA3 = {"model_name": "gemma3-4b", "override_model_config": True, "decoder_block": "gemma3"}

# (case name, layer class, extra config)
_LAYERS = [
    ("gemma", gemma.GemmaDecoderLayer, {"decoder_block": "gemma"}),
    ("gemma2", gemma2.Gemma2DecoderLayer, {"decoder_block": "gemma2"}),
    ("gemma3", gemma3.Gemma3DecoderLayer, _GEMMA3),
    ("gemma3_scannable_block", gemma3.Gemma3ScannableBlock, _GEMMA3),
]


class GemmaExplicitShardingTest(parameterized.TestCase):
  """Auto and explicit shard modes must agree on every Gemma decoder layer."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest(
          "needs at least 2 devices for the mesh to shard anything; "
          "set XLA_FLAGS=--xla_force_host_platform_device_count=8"
      )

  def _forward(self, layer_cls, extra_config, shard_mode):
    """Builds one decoder layer in the given shard mode and runs a single forward pass.

    The rng seeds are fixed so both modes initialize identical weights; any difference
    in the returned activations is then attributable to the sharding annotations.

    Args:
      layer_cls: Decoder layer class to instantiate.
      extra_config: Config overrides this family needs, on top of _COMMON.
      shard_mode: Either "auto" or "explicit".

    Returns:
      The layer output as a numpy array.
    """
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **(_COMMON | extra_config | {"shard_mode": shard_mode}),
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
      layer = layer_cls(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      batch = cfg.micro_batch_size_to_train_on
      length = cfg.max_target_length
      inputs = jnp.reshape(
          jnp.arange(batch * length * cfg.emb_dim, dtype=jnp.float32) / (batch * length * cfg.emb_dim),
          (batch, length, cfg.emb_dim),
      )
      positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
      segment_ids = jnp.ones((batch, length), dtype=jnp.int32)
      # Gemma3ScannableBlock returns the bare activations outside the scan path, while
      # the single-layer classes always return an (output, kv_cache) pair.
      output = layer(inputs, segment_ids, positions, True, MODEL_MODE_TRAIN)
      if isinstance(output, tuple):
        output = output[0]

    return np.asarray(jax.device_get(output))

  @parameterized.named_parameters(*_LAYERS)
  def test_explicit_matches_auto(self, layer_cls, extra_config):
    """The explicit run must both succeed and reproduce the auto-mode activations."""
    auto_out = self._forward(layer_cls, extra_config, "auto")
    explicit_out = self._forward(layer_cls, extra_config, "explicit")

    self.assertEqual(auto_out.shape, explicit_out.shape)
    np.testing.assert_allclose(
        explicit_out,
        auto_out,
        rtol=1e-5,
        atol=1e-5,
        err_msg="explicit shard mode changed the layer output; an out_sharding is wrong, not just missing",
    )

  @parameterized.named_parameters(*_LAYERS)
  def test_explicit_shard_mode_is_accepted_by_config(self, layer_cls, extra_config):
    """Every Gemma decoder must be on the explicit-sharding allowlist in configs/types.py."""
    del layer_cls
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **(_COMMON | extra_config | {"shard_mode": "explicit"}),
    )
    self.assertEqual(cfg.shard_mode, ShardMode.EXPLICIT)


if __name__ == "__main__":
  unittest.main()
