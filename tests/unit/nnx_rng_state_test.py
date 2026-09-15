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

"""Regression tests for NNX RNG state surviving jax.lax.scan / jax.checkpoint.

`nnx.Rngs` derives each draw as ``fold_in(key, count)`` and then increments
``count``. The key never changes, so ``count`` is the only thing that makes two
successive draws differ. Any code that splits a module, runs it through a JAX
transform and writes the result back has to write ``nnx.RngState`` back too --
otherwise the counter resets to its pre-transform value on every call and the
layer replays the exact same dropout mask on every training step.

That failure is silent: no shape changes, no error, and neither a linter nor
``jax_debug_key_reuse`` can see it (the latter only inspects typed keys and
treats ``fold_in`` as a pure source). Asserting on the counter is the only way
to catch it, hence these tests.

Only the ``dropout`` stream is asserted on. The ``params`` stream is drawn from
at construction time and is expected to sit still during a forward pass.
"""

import sys

from absl.testing import absltest
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "nnx_rng_state_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 2,
    "attention": "dot_product",
    "max_target_length": 16,
    "base_emb_dim": 256,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 512,
    "max_prefill_predict_length": 4,
    "dropout_rate": 0.1,
}

# Dropout counters that live inside the decoder layer stack. Under
# ``scan_layers=False`` these are plain per-layer modules; under
# ``scan_layers=True`` they are a single stacked variable behind jax.lax.scan.
_SCANNED_DROPOUT_COUNTS = (
    "layers/dropout/rngs/dropout/count",
    "layers/mlp/dropout/rngs/dropout/count",
)
_UNSCANNED_DROPOUT_COUNTS = (
    "layers_0/dropout/rngs/dropout/count",
    "layers_0/mlp/dropout/rngs/dropout/count",
    "layers_1/dropout/rngs/dropout/count",
    "layers_1/mlp/dropout/rngs/dropout/count",
)


def _make_config(**overrides):
  merged = {**_BASE_CONFIG, **overrides}
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], override_model_config=True, **merged)


def _rng_counts(module):
  """Maps every nnx.RngCount path in `module` to its current value."""
  counts = {}
  for path, leaf in nnx.to_flat_state(nnx.state(module, nnx.RngCount)):
    value = leaf.value if isinstance(leaf, nnx.Variable) else leaf
    counts["/".join(str(p) for p in path)] = jnp.asarray(value)
  return counts


class DecoderRngStateTest(absltest.TestCase):
  """A forward pass must leave the decoder's dropout counters advanced."""

  def _build(self, **overrides):
    """Builds a decoder plus its shared embedding from an overridden base config."""
    cfg = _make_config(**overrides)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=rngs,
    )
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))

    def run():
      # Decoder.__call__ returns (logits, hidden_state, kv_caches), plus an
      # optional fourth element (expert_indices) for MoE models, so index
      # instead of unpacking.
      outputs = decoder(
          shared_embedding,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=False,
          model_mode=MODEL_MODE_TRAIN,
      )
      return outputs[0]

    return decoder, run

  def _assert_counts_advance(self, decoder, run, paths):
    """Asserts every dropout RNG counter in `paths` strictly advances across one forward pass."""
    before = _rng_counts(decoder)
    for path in paths:
      self.assertIn(path, before, f"expected {path} in the decoder state; available: {sorted(before)}")

    run()
    after = _rng_counts(decoder)

    # `<=` rather than `==`: a counter that went backwards or reset is just as
    # broken as one that stood still, and both mean a replayed dropout mask.
    stalled = [path for path in paths if bool(jnp.any(after[path] <= before[path]))]
    self.assertEqual(
        stalled,
        [],
        f"Dropout RNG counters did not strictly advance across a forward pass: {stalled}. The updated "
        "nnx.RngState was dropped on write-back, so every step replays the same dropout mask.",
    )

  def test_scanned_layers_dropout_counts_advance(self):
    decoder, run = self._build(scan_layers=True)
    self._assert_counts_advance(decoder, run, _SCANNED_DROPOUT_COUNTS)

  def test_unscanned_layers_dropout_counts_advance(self):
    """Control: the non-scanned path writes state back through plain attributes."""
    decoder, run = self._build(scan_layers=False)
    self._assert_counts_advance(decoder, run, _UNSCANNED_DROPOUT_COUNTS)

  def test_remainder_block_dropout_counts_advance(self):
    """The rematerialized remainder block must carry its RNG state out of jax.checkpoint.

    Gemma4 repeats a 6-layer pattern; with 3 decoder layers there is no full
    block to scan, so the whole stack goes through ``_apply_remainder_block``.
    """
    decoder, run = self._build(
        decoder_block="gemma4",
        scan_layers=True,
        # Must be `base_num_decoder_layers`, not `num_decoder_layers`: config
        # post-init unconditionally recomputes the latter as
        # `int((2 ** layer_scale) * base_num_decoder_layers)`
        # (configs/types.py), so a direct `num_decoder_layers` override is
        # silently discarded.
        base_num_decoder_layers=3,
        vocab_size=256,
    )
    # Match a Dropout module's own stream ("<...>/dropout/rngs/dropout/count").
    # A module that merely stores the nnx.Rngs it was constructed with also has a
    # "<...>/rngs/dropout/count", but that stream is only drawn from at
    # construction time and is expected to sit still during a forward pass.
    paths = [path for path in _rng_counts(decoder) if "/dropout/rngs/dropout/count" in path]
    self.assertNotEmpty(paths, "expected the remainder block to hold dropout RNG state at dropout_rate>0")

    self._assert_counts_advance(decoder, run, paths)


if __name__ == "__main__":
  absltest.main()
