# Copyright 2026 Ant Group. All Rights Reserved.
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

"""Decoder integration tests for KDA (attention_type='kda').

These exercise the *real* MaxText decoder path — `NNXDecoderLayer` selecting
`KimiDeltaAttention` via the `attention_type` dispatch — rather than the
hand-rolled block used in the standalone layer smoke test.

CPU-runnable tests cover config acceptance, the scan_layers guard, and the
layer-branch dispatch (no kernel needed at construction). TPU-only tests run
an actual forward/backward training loop through the full model.

Run with: python -m pytest tests/unit/kda_decoder_integration_test.py -v
"""

import sys

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.attentions import Attention
from maxtext.layers.attention_kda import KimiDeltaAttention
from maxtext.layers import nnx_decoders
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

try:
  from tokamax._src.ops.experimental.kda import api  # noqa: F401  # pylint: disable=unused-import

  KDA_API_AVAILABLE = True
except ImportError:
  KDA_API_AVAILABLE = False

# Small shared hyper-parameters for the integration model.
_KDA_TEST_VOCAB = 128
_KDA_TEST_SEQ = 64  # multiple of the KDA chunk size


def _kda_pyconfig(**kwargs):
  """Build a tiny train config that selects the KDA attention variant."""
  defaults = {
      "per_device_batch_size": 4.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "decoder_block": "default",  # generic NNXDecoderLayer, where the KDA branch lives
      "base_num_decoder_layers": 2,
      "attention": "dot_product",
      "attention_type": "kda",
      "scan_layers": False,
      "max_target_length": _KDA_TEST_SEQ,
      "base_emb_dim": 128,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 128,
      "vocab_size": _KDA_TEST_VOCAB,
      "max_prefill_predict_length": 4,
      # Bounded-decay (safe) gate keeps the Delta-Rule recurrence stable,
      # matching the standalone layer smoke. fp32 for the same reason; bf16
      # hyperparameter tuning for KDA models belongs to the model-landing
      # follow-up, not the integration validation.
      "use_kda_safe_gate": True,
      "kda_lower_bound": -5.0,
      "dtype": "float32",
      "weight_dtype": "float32",
  }
  defaults.update(kwargs)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **defaults)


class TestKdaDecoderConfig:
  """Config-level acceptance and guards (CPU-runnable, no kernel)."""

  def test_kda_attention_type_accepted(self):
    """attention_type='kda' with scan_layers=false builds a valid config."""
    cfg = _kda_pyconfig()
    assert cfg.attention_type == "kda"
    assert cfg.scan_layers is False

  def test_kda_requires_scan_layers_false(self):
    """KDA layers are not validated in a scanned stack; the config must reject it."""
    with pytest.raises(ValueError, match="scan_layers"):
      _kda_pyconfig(scan_layers=True)

  def test_kda_decoder_layer_dispatches_to_kimi_delta(self):
    """NNXDecoderLayer builds KimiDeltaAttention when attention_type='kda'."""
    cfg = _kda_pyconfig(per_device_batch_size=1.0)
    mesh = Mesh(np.array(jax.devices()), (cfg.mesh_axes[0],))
    layer = nnx_decoders.NNXDecoderLayer(
        config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, attention_type="kda", rngs=nnx.Rngs(0)
    )
    assert isinstance(layer.self_attention, KimiDeltaAttention)

  def test_default_decoder_layer_keeps_attention(self):
    """Non-KDA attention types still build the regular Attention module."""
    cfg = _kda_pyconfig(per_device_batch_size=1.0, attention_type="global")
    mesh = Mesh(np.array(jax.devices()), (cfg.mesh_axes[0],))
    layer = nnx_decoders.NNXDecoderLayer(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    assert isinstance(layer.self_attention, Attention)


@pytest.mark.tpu_only
@pytest.mark.skipif(not KDA_API_AVAILABLE, reason="KDA API not available in the installed tokamax")
class TestKdaDecoderTraining:
  """End-to-end training through the real decoder with the KDA variant.

  Uses the same history-dependent delayed-copy task as the standalone smoke:
  the loss collapses only if the KDA recurrent state carries history through
  the full forward/backward/optimizer chain inside the actual decoder.
  """

  @staticmethod
  def _delayed_copy_dataset(seed, num_seqs, seq_len, delay):
    rng = np.random.default_rng(seed)
    total = seq_len + 1
    seqs = rng.integers(0, _KDA_TEST_VOCAB, size=(num_seqs, total), dtype=np.int32)
    for i in range(delay, total):
      seqs[:, i] = seqs[:, i - delay]
    return seqs

  def test_kda_decoder_train_loss_decreases(self):
    cfg = _kda_pyconfig()
    delay = 5
    assert delay > cfg.linear_conv_kernel_dim

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    # Every decoder layer instance must carry a KDA attention module.
    for i in range(cfg.num_decoder_layers):
      layer = getattr(model.decoder, f"layers_{i}")
      assert isinstance(layer.self_attention, KimiDeltaAttention)

    data = self._delayed_copy_dataset(seed=42, num_seqs=512, seq_len=_KDA_TEST_SEQ, delay=delay)
    # Position j predicts token j+1 = t[j+1-delay]; mask the irreducible prefix.
    loss_mask = jnp.asarray(np.arange(_KDA_TEST_SEQ) >= delay - 1, dtype=jnp.float32)[None, :]
    optimizer = nnx.Optimizer(model, optax.adamw(2e-3), wrt=nnx.Param)

    def masked_ce(logits, labels):
      ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits.astype(jnp.float32), labels=labels)
      return (ce * loss_mask).sum() / loss_mask.sum() / labels.shape[0]

    @nnx.jit
    def train_step(model, optimizer, tokens):
      def loss_fn(model):
        positions = jnp.broadcast_to(jnp.arange(_KDA_TEST_SEQ, dtype=jnp.int32)[None, :], tokens[:, :-1].shape)
        logits = model(tokens[:, :-1], positions, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
        return masked_ce(logits, tokens[:, 1:]), logits

      (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      optimizer.update(model, grads)
      return loss

    perm_rng = np.random.default_rng(1)
    steps, batch = 400, int(cfg.global_batch_size_to_train_on)
    losses = []
    for step in range(steps):
      idx = perm_rng.integers(0, data.shape[0], size=batch)
      loss_val = float(train_step(model, optimizer, jnp.asarray(data[idx])))
      assert np.isfinite(loss_val), f"non-finite loss {loss_val} at step {step}"
      losses.append(loss_val)

    init_loss, final_loss = losses[0], float(np.mean(losses[-20:]))
    assert final_loss < 0.5 * init_loss and final_loss < 1.0, (
        f"kda decoder loss did not collapse: init={init_loss:.4f} final={final_loss:.4f} "
        "(the KDA recurrent state is not carrying history through the real decoder)"
    )
