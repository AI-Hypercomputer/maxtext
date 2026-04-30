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


"""Unit tests for distillation metrics.

Covers:
  - KL direction (forward KL = teacher || student)
  - KL self-equivalence (KL(p || p) == 0 at any temperature)
  - Cross-entropy / perplexity numerics
  - (sum, count) aggregator is unbiased under uneven masks
  - Sharded (multi-device) verification of the same property
  - Label mask correctness for pad_id == 0 and pad_id != 0
  - T^2 scaling of soft loss
  - Feature-mapping loss (beta > 0 path) and layer_indices slicing
"""

import pytest

pytest.importorskip("optax")
pytest.importorskip("tunix")

pytestmark = [pytest.mark.cpu_only, pytest.mark.post_training]

import os
import pickle
import tempfile
import unittest
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from array_record.python import array_record_module
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.trainers.post_train.distillation import distillation_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    vocab_size: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    alpha: float = 0.5,
    beta_feature: float = 0.0,
    layer_indices: Optional[List[int]] = None,
):
  """Builds a CombinedDistillationStrategy with no-op forward fns (we feed outputs directly)."""

  def noop(*_args, **_kwargs):
    return None

  return distillation_utils.CombinedDistillationStrategy(
      student_forward_fn=noop,
      teacher_forward_fn=noop,
      pad_id=pad_id,
      temperature=temperature,
      alpha=alpha,
      beta_feature=beta_feature,
      layer_indices=layer_indices,
      vocab_size=vocab_size,
      max_steps=1,
  )


def _one_hot_labels(targets: jnp.ndarray, vocab_size: int, pad_id: int = 0) -> jnp.ndarray:
  one_hot = jax.nn.one_hot(targets, vocab_size)
  mask = jnp.not_equal(targets, pad_id).astype(one_hot.dtype)[..., None]
  return one_hot * mask


def _mean(pair):
  """Unpacks a (sum, count) metric tuple into a mean value."""
  s, c = pair
  c_val = float(c)
  return float(s) / c_val if c_val > 0 else float(s)


def _has_4_devices() -> bool:
  return len(jax.devices()) >= 4


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class DistillationMetricsTest(unittest.TestCase):
  """Unit tests for CombinedDistillationStrategy metrics."""

  # --- 1. KL direction --------------------------------------------------

  def test_kl_direction_matches_forward_kl_teacher_student(self):
    """optax.kl_divergence(log_pred, target) == KL(target || pred). We use it
    with log_pred=log_student, target=teacher_probs => KL(teacher || student)."""
    rng = np.random.default_rng(0)
    vocab_size = 8
    s_logits = jnp.asarray(rng.normal(size=(1, 1, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(1, 1, vocab_size)).astype(np.float32))

    log_s = jax.nn.log_softmax(s_logits, axis=-1)
    t_p = jax.nn.softmax(t_logits, axis=-1)
    optax_kl = float(optax.kl_divergence(log_s, t_p)[0, 0])

    # Reference: forward KL(teacher || student) = sum_i t_i (log t_i - log s_i).
    log_t = np.asarray(jax.nn.log_softmax(t_logits, axis=-1)[0, 0])
    ref_fwd = float(np.sum(np.asarray(t_p[0, 0]) * (log_t - np.asarray(log_s[0, 0]))))
    np.testing.assert_allclose(optax_kl, ref_fwd, rtol=1e-5, atol=1e-6)

    # Reverse KL(student || teacher) should NOT match — assert we didn't
    # accidentally compute it in the wrong direction.
    s_p = jax.nn.softmax(s_logits, axis=-1)
    log_t_jax = jax.nn.log_softmax(t_logits, axis=-1)
    ref_rev = float(np.sum(np.asarray(s_p[0, 0]) * (np.asarray(log_s[0, 0]) - np.asarray(log_t_jax[0, 0]))))
    self.assertFalse(np.isclose(ref_fwd, ref_rev, atol=1e-3), "test logits chosen badly: fwd == rev KL")

  def test_kl_self_distillation_is_zero_at_any_temperature(self):
    """KL(p || p) == 0; with student==teacher logits, soft_loss must be 0 at any T."""
    vocab_size, batch_size, seq_len = 8, 2, 3
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    out = distillation_utils.DistillationForwardOutput(logits=logits, out_projection_activations=None)
    targets = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size)

    for temperature in [0.5, 1.0, 2.0, 8.0]:
      strategy = _make_strategy(vocab_size, temperature=temperature, alpha=1.0)  # alpha=1 -> total = soft only
      total_loss, metrics = strategy.compute_loss(out, out, labels, step=None)
      # soft_loss is scaled by T^2, so per-position float32 noise gets amplified
      # by T^2. Scale tolerances with T^2 so the assertion is meaningful across T.
      t_sq = max(1.0, float(temperature)) ** 2
      np.testing.assert_allclose(float(total_loss), 0.0, atol=1e-4 * t_sq)
      soft_sum, _ = metrics["distill/soft_loss"]
      np.testing.assert_allclose(float(soft_sum), 0.0, atol=1e-3 * t_sq)

  # --- 2. Perplexity / CE numerics --------------------------------------

  def test_hard_loss_matches_manual_cross_entropy_and_perplexity_relation(self):
    vocab_size = 4
    s_logits = jnp.array([[[2.0, 1.0, 0.5, 0.1], [0.0, 3.0, 1.0, 0.5]]], dtype=jnp.float32)
    # Target token 0 is the pad_id so it would be masked. Use non-zero targets.
    targets = jnp.array([[3, 1]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)
    out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)

    strategy = _make_strategy(vocab_size, pad_id=0, alpha=0.0)  # alpha=0 -> total = hard only
    total_loss, metrics = strategy.compute_loss(out, out, labels, step=None)

    log_p = jax.nn.log_softmax(s_logits, axis=-1)
    manual_ce = -float(log_p[0, 0, 3] + log_p[0, 1, 1]) / 2.0
    np.testing.assert_allclose(float(total_loss), manual_ce, rtol=1e-5)

    # Perplexity sanity: exp(hard_loss) finite and >= 1 for non-degenerate logits.
    ppl = float(np.exp(_mean(metrics["distill/hard_loss"])))
    self.assertGreaterEqual(ppl, 1.0)
    self.assertTrue(np.isfinite(ppl))

  # --- 3. Label mask: pad_id == 0 and pad_id != 0 -----------------------

  def test_label_mask_excludes_pad_tokens_pad0(self):
    self._run_label_mask_excludes_pad(pad_id=0)

  def test_label_mask_excludes_pad_tokens_pad7(self):
    self._run_label_mask_excludes_pad(pad_id=7)

  def _run_label_mask_excludes_pad(self, pad_id):
    """Asserts pad positions are zeroed in labels and excluded from hard loss."""
    vocab_size = 8
    strategy = _make_strategy(vocab_size, pad_id=pad_id, alpha=0.0)
    # second token is padding
    targets = jnp.array([[1, pad_id]], dtype=jnp.int32)
    labels = strategy.create_labels(targets)
    # the padded row must be all zeros
    np.testing.assert_array_equal(np.asarray(labels[0, 1]), np.zeros(vocab_size))

    s_logits = jnp.array([[[5.0, -5.0, 0, 0, 0, 0, 0, 0], [-5.0, 5.0, 0, 0, 0, 0, 0, 0]]], dtype=jnp.float32)
    out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    total_loss, metrics = strategy.compute_loss(out, out, labels, step=None)

    _, hard_cnt = metrics["distill/hard_loss"]
    np.testing.assert_allclose(float(hard_cnt), 1.0, atol=1e-6)  # only 1 valid token
    # Total loss must equal CE on token 0 only, ignoring padded position.
    log_p = jax.nn.log_softmax(s_logits, axis=-1)
    expected = -float(log_p[0, 0, 1])  # target = 1
    np.testing.assert_allclose(float(total_loss), expected, rtol=1e-5)

  def test_create_labels_masks_packed_segmentation(self):
    """Positions where targets_segmentation == 0 must be zeroed even when the target token is non-pad."""
    vocab_size = 8
    strategy = _make_strategy(vocab_size, pad_id=0, alpha=0.0)
    # Bin layout: doc1 at [0,1], doc2 at [2], in-bin pad at [3]. All targets non-pad.
    targets = jnp.array([[1, 2, 3, 1]], dtype=jnp.int32)
    targets_segmentation = jnp.array([[1, 1, 2, 0]], dtype=jnp.int32)

    labels_packed = strategy.create_labels(targets, targets_segmentation=targets_segmentation)
    labels_unpacked = strategy.create_labels(targets)

    np.testing.assert_array_equal(np.asarray(labels_packed[0, 3]), np.zeros(vocab_size))
    self.assertGreater(float(np.sum(labels_unpacked[0, 3])), 0.0)
    for pos in (0, 1, 2):
      np.testing.assert_array_equal(np.asarray(labels_packed[0, pos]), np.asarray(labels_unpacked[0, pos]))

  def test_offline_iterator_preserves_packing_fields(self):
    """Packed segmentation fields survive write -> ArrayRecord -> OfflineArrayRecordIterator -> Tunix adapter."""
    record = {
        "tokens": np.array([[10, 11, 12, 13]], dtype=np.int32),
        "top_k_logits": np.zeros((1, 4, 8), dtype=np.float32),
        "top_k_indices": np.zeros((1, 4, 8), dtype=np.int32),
        "inputs_position": np.array([[0, 1, 0, 0]], dtype=np.int32),
        "inputs_segmentation": np.array([[1, 1, 2, 0]], dtype=np.int32),
        "targets": np.array([[11, 12, 13, 0]], dtype=np.int32),
        "targets_segmentation": np.array([[1, 1, 2, 0]], dtype=np.int32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.array_record")
      writer = array_record_module.ArrayRecordWriter(path, "group_size:1")
      writer.write(pickle.dumps(record))
      writer.close()

      it = distillation_utils.OfflineArrayRecordIterator(path, epochs=1)
      batch = next(it)

    np.testing.assert_array_equal(batch["inputs"], record["tokens"])
    np.testing.assert_array_equal(batch["inputs_segmentation"], record["inputs_segmentation"])
    np.testing.assert_array_equal(batch["targets_segmentation"], record["targets_segmentation"])
    np.testing.assert_array_equal(batch["targets"], record["targets"])

    adapter = distillation_utils.MaxTextToTunixIterator(iter([batch]))
    tunix_input = next(adapter)
    np.testing.assert_array_equal(np.asarray(tunix_input.decoder_segment_ids), record["inputs_segmentation"])
    np.testing.assert_array_equal(np.asarray(tunix_input.targets_segmentation), record["targets_segmentation"])

  # --- 4. Temperature^2 scaling of soft loss ----------------------------

  def test_soft_loss_scales_with_temperature_squared_in_high_T_limit(self):
    """As T grows, soft_loss = T^2 * KL(softmax(t/T) || softmax(s/T)) should
    approach a finite non-zero value (Hinton scaling). Verify it doesn't collapse to 0."""
    vocab_size, batch_size, seq_len = 16, 2, 4
    rng = np.random.default_rng(2)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_logits, out_projection_activations=None)
    targets = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size)

    losses = []
    for temperature in [1.0, 4.0, 16.0]:
      strategy = _make_strategy(vocab_size, temperature=temperature, alpha=1.0)
      _, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)
      losses.append(_mean(metrics["distill/soft_loss"]))

    # All positive and bounded; high-T value should not be drastically smaller than T=1.
    for value in losses:
      self.assertGreater(value, 0)
      self.assertTrue(np.isfinite(value))
    # T^2 scaling keeps high-T loss within an order of magnitude of T=1 in this regime.
    self.assertGreater(losses[-1], 0.1 * losses[0])

  def test_kl_T1_metric_is_temperature_invariant_when_logged(self):
    """distill/kl_div_T1 should be the same regardless of self.temperature."""
    vocab_size, batch_size, seq_len = 8, 1, 2
    rng = np.random.default_rng(3)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_logits, out_projection_activations=None)
    targets = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size)

    values = []
    for temperature in [1.0, 4.0, 16.0]:
      strategy = _make_strategy(vocab_size, temperature=temperature, alpha=1.0)
      _, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)
      values.append(_mean(metrics["distill/kl_div_T1"]))
    np.testing.assert_allclose(values[0], values[1], rtol=1e-5)
    np.testing.assert_allclose(values[0], values[2], rtol=1e-5)

  # --- 5. (sum, count) aggregator: unbiased under uneven masks ----------

  def test_weighted_mean_aggregator_token_weighted(self):
    """The aggregator should compute sum(sums) / sum(counts), not mean of per-step means."""
    pairs = [
        (jnp.array(8.0), jnp.array(8.0)),  # 8 valid tokens, mean 1.0
        (jnp.array(5.0), jnp.array(1.0)),  # 1 valid token, value 5.0
        (jnp.array(8.0), jnp.array(4.0)),  # 4 valid tokens, mean 2.0
        (jnp.array(8.0), jnp.array(4.0)),  # 4 valid tokens, mean 2.0
    ]
    # mean-of-means would be (1+5+2+2)/4 = 2.5 (biased)
    # token-weighted: 29/17 ≈ 1.7058 (correct)
    np.testing.assert_allclose(distillation_utils.weighted_mean(pairs), 29.0 / 17.0, rtol=1e-6)

  def test_weighted_mean_aggregator_handles_empty_and_zero_count(self):
    self.assertEqual(distillation_utils.weighted_mean([]), 0.0)
    self.assertEqual(distillation_utils.weighted_mean([(jnp.array(0.0), jnp.array(0.0))]), 0.0)

  # --- 6. Multi-device sharded version ----------------------------------

  @unittest.skipIf(not _has_4_devices(), "Requires ≥4 devices.")
  def test_sum_count_unbiased_under_sharded_uneven_masks(self):
    """Across 4 devices with wildly different valid-token counts,
    mean-of-per-device-means is biased; token-weighted (sum/count) is correct."""
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ("data",))

    # Per-device (B=2, T=4). Layout: 4 shards along the data axis.
    # Device 0: 8 valid tokens, value 1.0  -> per-device mean 1.0
    # Device 1: 1 valid token,  value 5.0  -> per-device mean 5.0
    # Device 2: 4 valid tokens, value 2.0  -> per-device mean 2.0
    # Device 3: 4 valid tokens, value 2.0  -> per-device mean 2.0
    losses = jnp.array(
        [
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    masks = jnp.array(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=jnp.float32,
    )

    sharding = NamedSharding(mesh, P("data"))
    losses = jax.device_put(losses, sharding)
    masks = jax.device_put(masks, sharding)

    @jax.jit
    def biased_mean_of_means(losses, masks):
      per_dev_sum = jnp.sum(losses * masks, axis=(1, 2))
      per_dev_cnt = jnp.maximum(jnp.sum(masks, axis=(1, 2)), 1.0)
      return jnp.mean(per_dev_sum / per_dev_cnt)

    @jax.jit
    def unbiased_token_weighted(losses, masks):
      return jnp.sum(losses * masks) / jnp.sum(masks)

    np.testing.assert_allclose(float(biased_mean_of_means(losses, masks)), 2.5, rtol=1e-5)
    np.testing.assert_allclose(float(unbiased_token_weighted(losses, masks)), 29.0 / 17.0, rtol=1e-5)

  @unittest.skipIf(not _has_4_devices(), "Requires ≥4 devices.")
  def test_compute_loss_sums_aggregate_correctly_across_shards(self):
    """Run compute_loss on a sharded batch; verify weighted_mean over per-shard
    (sum, count) outputs matches the global mean computed on the full batch."""
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ("data",))

    vocab_size, batch_size, seq_len = 8, 4, 4  # B=4 so each shard gets one example
    rng = np.random.default_rng(7)
    s_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    t_logits = jnp.asarray(rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32))
    # Different valid-token counts per row (shard) to exercise the bias path.
    targets = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 0, 0, 0],
            [1, 2, 0, 0],
            [1, 2, 3, 0],
        ],
        dtype=jnp.int32,
    )
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)

    strategy = _make_strategy(vocab_size, alpha=0.0)  # hard-loss only for clarity
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    t_out = distillation_utils.DistillationForwardOutput(logits=t_logits, out_projection_activations=None)

    # Global reference (single-host).
    _, global_metrics = strategy.compute_loss(s_out, t_out, labels, step=None)
    global_mean = _mean(global_metrics["distill/hard_loss"])

    # Sharded: compute per-shard (sum, count), aggregate via weighted_mean.
    sharding = NamedSharding(mesh, P("data"))
    s_sharded = jax.device_put(s_logits, sharding)
    t_sharded = jax.device_put(t_logits, sharding)
    l_sharded = jax.device_put(labels, sharding)

    @jax.jit
    def per_shard(s_l, t_l, lab):
      s_o = distillation_utils.DistillationForwardOutput(logits=s_l, out_projection_activations=None)
      t_o = distillation_utils.DistillationForwardOutput(logits=t_l, out_projection_activations=None)
      _, m = strategy.compute_loss(s_o, t_o, lab, step=None)
      return m["distill/hard_loss"]  # (sum, count) — sharded

    shard_sum, shard_cnt = per_shard(s_sharded, t_sharded, l_sharded)
    # Pull each shard's contribution as if it came from a different host.
    per_shard_pairs = [
        (jax.device_get(shard_sum.addressable_shards[i].data), jax.device_get(shard_cnt.addressable_shards[i].data))
        for i in range(len(devices))
    ]
    np.testing.assert_allclose(distillation_utils.weighted_mean(per_shard_pairs), global_mean, rtol=1e-5)

  # --- 7. Feature-mapping loss (beta > 0 path) --------------------------

  def test_feature_loss_zero_when_student_features_match_teacher(self):
    """With beta>0 and s_features == t_features, distill/out_proj_feature_loss
    must be 0 (cosine distance of identical vectors). Total loss then equals the
    base logit loss — verifies the feature path doesn't contribute spurious mass."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 2, 8
    num_layers = 4
    s_logits = jnp.array([[[2.0, 1.0, 0.5, 0.1], [0.0, 3.0, 1.0, 0.5]]], dtype=jnp.float32)
    rng = np.random.default_rng(11)
    features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=features)

    targets = jnp.array([[3, 1]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)

    strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 2])
    total_loss, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    feat_sum, _ = metrics["distill/out_proj_feature_loss"]
    np.testing.assert_allclose(float(feat_sum), 0.0, atol=1e-5)
    # alpha=0 + identical logits + zero feature loss -> total_loss is just hard CE.
    np.testing.assert_allclose(float(total_loss), _mean(metrics["distill/hard_loss"]), rtol=1e-5)

  def test_feature_loss_positive_when_student_features_differ_and_respects_layer_indices(self):
    """With beta>0 and s_features != t_features on the sliced layers, feature
    loss must be > 0; and restricting layer_indices to a subset must yield a
    different value than using all layers (slice is actually applied)."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 2, 8
    num_layers = 4
    s_logits = jnp.array([[[2.0, 1.0, 0.5, 0.1], [0.0, 3.0, 1.0, 0.5]]], dtype=jnp.float32)
    rng = np.random.default_rng(12)
    s_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    # Teacher features differ on layers 1 and 3; match on layers 0 and 2.
    t_features = s_features.at[1].set(s_features[1] + 1.0).at[3].set(s_features[3] - 1.0)
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s_features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t_features)

    targets = jnp.array([[3, 1]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)

    # Case A: include only layers where features match -> feature loss == 0.
    strategy_match = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 2])
    _, m_match = strategy_match.compute_loss(s_out, t_out, labels, step=None)
    feat_match, _ = m_match["distill/out_proj_feature_loss"]
    np.testing.assert_allclose(float(feat_match), 0.0, atol=1e-5)

    # Case B: include the differing layers -> feature loss > 0.
    strategy_diff = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[1, 3])
    _, m_diff = strategy_diff.compute_loss(s_out, t_out, labels, step=None)
    feat_diff, _ = m_diff["distill/out_proj_feature_loss"]
    self.assertGreater(float(feat_diff), 0.0)

  # --- 7b. Feature loss NaN-safety (pad mask + epsilon) -----------------

  def _features_with_zero_pad_positions(self, num_layers, batch_size, seq_len, hidden_dim, rng):
    """Returns [L, B, T, D] features where the last column of T is all zeros
    (matches what typically sits at padded positions after masked attention)."""
    feats = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    feats = feats.at[:, :, -1, :].set(0.0)
    return feats

  def test_feature_loss_stays_finite_when_padded_positions_are_zero_norm(self):
    """Before the patch, cosine distance at padded positions gave 0/0 = NaN
    and the mean leaked into the aggregated loss. The mask must zero those
    positions out before any reduction."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 2, 4, 8
    num_layers = 4
    rng = np.random.default_rng(21)
    # Padded positions (last token) have zero features for BOTH student and teacher.
    s_features = self._features_with_zero_pad_positions(num_layers, batch_size, seq_len, hidden_dim, rng)
    t_features = self._features_with_zero_pad_positions(num_layers, batch_size, seq_len, hidden_dim, rng)

    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)
    # Targets: last position is pad (pad_id=0); others are valid tokens 1..3.
    targets = jnp.array([[1, 2, 3, 0], [3, 2, 1, 0]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)

    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s_features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t_features)

    strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 1, 2, 3])
    total_loss, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    feat_sum, _ = metrics["distill/out_proj_feature_loss"]
    self.assertTrue(np.isfinite(float(feat_sum)), f"feature loss not finite: {float(feat_sum)!r}")
    self.assertTrue(np.isfinite(float(total_loss)), f"total loss not finite: {float(total_loss)!r}")

  def test_feature_loss_is_invariant_to_padded_position_values(self):
    """The mask must make the feature loss independent of what sits at
    padded positions — i.e. rewriting pad features to garbage or zeros must
    not change the loss value."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 3, 8
    num_layers = 2
    rng = np.random.default_rng(22)
    s_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    t_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))

    # Mark the last position as padding in labels.
    targets = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)
    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)

    def run(pad_filler):
      s = s_features.at[:, :, -1, :].set(pad_filler)
      t = t_features.at[:, :, -1, :].set(pad_filler)
      s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s)
      t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t)
      strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 1])
      _, m = strategy.compute_loss(s_out, t_out, labels, step=None)
      return float(m["distill/out_proj_feature_loss"][0])

    # Replace pad positions with zeros, then with huge noise; both must give the same loss.
    loss_zero_pad = run(0.0)
    loss_huge_pad = run(1e6)
    np.testing.assert_allclose(loss_zero_pad, loss_huge_pad, rtol=1e-5, atol=1e-6)

  def test_feature_loss_safe_norm_handles_zero_valid_activations(self):
    """Even on valid (non-padded) positions, a zero-norm row must not blow up
    cosine distance — the optax epsilon in the patched fn floors the
    denominator. This guards against late-training activation collapse."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 2, 8
    num_layers = 2
    # BOTH student and teacher have zero-norm vectors on ALL positions
    # (including the valid ones). No mask can rescue us here — epsilon must.
    s_features = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    t_features = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)

    targets = jnp.array([[1, 2]], dtype=jnp.int32)  # both valid
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)
    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)

    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s_features)
    t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t_features)

    strategy = _make_strategy(vocab_size, alpha=0.0, beta_feature=1.0, layer_indices=[0, 1])
    total_loss, metrics = strategy.compute_loss(s_out, t_out, labels, step=None)

    feat_sum, _ = metrics["distill/out_proj_feature_loss"]
    self.assertTrue(np.isfinite(float(feat_sum)))
    self.assertTrue(np.isfinite(float(total_loss)))

  def test_feature_loss_gradient_finite_under_degenerate_inputs(self):
    """The backward pass is where 0/0 actually bites in training: jax.grad
    of cosine on a zero-norm row would produce NaN without epsilon. Assert
    gradients flow cleanly through the patched feature loss."""
    num_layers, batch_size, seq_len, hidden_dim = 2, 1, 2, 4

    def _loss(s, t, mask):
      # Exercise the patched default feature_loss_fn in isolation.
      strategy = _make_strategy(vocab_size=4, alpha=0.0, beta_feature=1.0)
      return strategy.feature_loss_fn(s, t, mask)

    # Degenerate case: zero-norm rows everywhere.
    s = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    t = jnp.zeros((num_layers, batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

    grad_s = jax.grad(_loss, argnums=0)(s, t, mask)
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad_s))), f"grad not finite: {np.asarray(grad_s)!r}")

  def test_feature_loss_l2_respects_mask(self):
    """Same masking contract applies to the L2 branch — padded positions
    must not contribute to the loss."""
    vocab_size, batch_size, seq_len, hidden_dim = 4, 1, 3, 8
    num_layers = 2
    rng = np.random.default_rng(23)
    s_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))
    t_features = jnp.asarray(rng.normal(size=(num_layers, batch_size, seq_len, hidden_dim)).astype(np.float32))

    targets = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size, pad_id=0)
    s_logits = jnp.zeros((batch_size, seq_len, vocab_size), dtype=jnp.float32)

    def run(pad_filler):
      s = s_features.at[:, :, -1, :].set(pad_filler)
      t = t_features.at[:, :, -1, :].set(pad_filler)
      # Build an L2 strategy directly (constructor override).
      strategy = distillation_utils.CombinedDistillationStrategy(
          student_forward_fn=lambda *_a, **_k: None,
          teacher_forward_fn=lambda *_a, **_k: None,
          pad_id=0,
          temperature=1.0,
          alpha=0.0,
          beta_feature=1.0,
          layer_indices=[0, 1],
          feature_loss_type="l2",
          vocab_size=vocab_size,
          max_steps=1,
      )
      s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=s)
      t_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=t)
      _, m = strategy.compute_loss(s_out, t_out, labels, step=None)
      return float(m["distill/out_proj_feature_loss"][0])

    loss_zero_pad = run(0.0)
    loss_huge_pad = run(1e6)
    np.testing.assert_allclose(loss_zero_pad, loss_huge_pad, rtol=1e-5, atol=1e-6)

  # --- 8. compute_eval_loss returns (sum, count) ------------------------

  def test_compute_eval_loss_returns_sum_count_pair(self):
    vocab_size = 4
    s_logits = jnp.array([[[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]], dtype=jnp.float32)
    targets = jnp.array([[0, 1]], dtype=jnp.int32)
    labels = _one_hot_labels(targets, vocab_size)
    s_out = distillation_utils.DistillationForwardOutput(logits=s_logits, out_projection_activations=None)
    strategy = _make_strategy(vocab_size)
    task_loss, metrics = strategy.compute_eval_loss(s_out, labels)
    self.assertIn("eval/hard_loss", metrics)
    np.testing.assert_allclose(_mean(metrics["eval/hard_loss"]), float(task_loss), rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
