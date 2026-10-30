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

"""CPU-only tests of bench_driver.py's payload builder and metric helpers (numpy only; no Tunix, no JAX).

The payload is a small stand-in for Tunix's packed RLTrainerPayload with the fields `realistic_payload` reads; it
checks the same invariants as check_payload_routing.py does at the production shapes with Tunix's own builder.
"""

import dataclasses
import os
import sys
import types
import unittest

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench_driver as d  # pylint: disable=g-import-not-at-top,wrong-import-position

pytestmark = pytest.mark.cpu_only

ROWS, BUDGET, LAYERS, TOP_K, EXPERTS, EP = 4, 4096, 4, 2, 64, 8
RATIO_KW = {"ep_rows": 2, "context": 4, "chunks": 4}


@dataclasses.dataclass(frozen=True)
class FakePayload:
  """The fields of a packed RLTrainerPayload that realistic_payload reads or writes."""

  completion_ids: np.ndarray
  segment_ids: np.ndarray
  segment_positions: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  old_per_token_logps: np.ndarray | None
  rollout_per_token_logps: np.ndarray | None
  routed_experts: np.ndarray | None
  num_segments: int = 3


def warmup_like(routed=True):
  """A warm-up-style payload: one short item per row at the start (3 segments), the rest padding, all routing
  UNSET and every token the same id, as Tunix's TRAINER_WARMUP_COMPILE builder writes it."""
  seg = np.zeros((ROWS, BUDGET), np.int32)
  cm = np.zeros((ROWS, BUDGET), np.int32)
  adv = np.zeros((ROWS, BUDGET), np.float32)
  for s in range(3):
    seg[:, s * 100 : (s + 1) * 100] = s + 1
    cm[:, s * 100 + 40 : (s + 1) * 100] = 1
    adv[:, s * 100 + 40 : (s + 1) * 100] = 1.0 if s % 2 else -1.0
  return FakePayload(
      completion_ids=np.where(seg > 0, 7, 0).astype(np.int32),
      segment_ids=seg,
      segment_positions=np.tile(np.arange(BUDGET, dtype=np.int32), (ROWS, 1)),
      completion_mask=cm,
      advantages=adv,
      old_per_token_logps=None,
      rollout_per_token_logps=np.zeros((ROWS, BUDGET), np.float32),
      routed_experts=(np.full((ROWS, BUDGET, LAYERS, TOP_K), d.UNSET_ROUTED_EXPERT, np.int16) if routed else None),
  )


def build(**kw):
  """Builds a small realistic payload with test defaults, overridden by `kw`."""
  args = {
      "num_experts": EXPERTS,
      "ep": EP,
      "pad_id": 0,
      "vocab_size": 5000,
      "seed": 0,
      "pool_size": 4096,
      "layer_targets_override": np.array([1.0, 1.3, 1.6, 2.0]),
  }
  args.update(kw)
  return d.realistic_payload(warmup_like(), **args)


class PayloadTest(unittest.TestCase):

  def test_shapes_dtypes_and_padding(self):
    base = warmup_like()
    p = build()
    for f in dataclasses.fields(base):
      a, b = getattr(p, f.name), getattr(base, f.name)
      if isinstance(b, np.ndarray):
        self.assertEqual((a.shape, a.dtype), (b.shape, b.dtype), f.name)
    self.assertIsNone(p.old_per_token_logps)
    self.assertEqual(p.num_segments, base.num_segments)
    real = p.segment_ids > 0
    self.assertAlmostEqual(1.0 - real.mean(), 0.07, delta=0.001)
    self.assertTrue((p.completion_ids[~real] == 0).all())
    self.assertTrue((p.routed_experts[~real] == d.UNSET_ROUTED_EXPERT).all())
    self.assertTrue((p.completion_mask[~real] == 0).all() and (p.advantages[~real] == 0).all())
    # Same number of segments per row as the base, contiguous ids starting at 1.
    for r in range(ROWS):
      self.assertEqual(sorted(np.unique(p.segment_ids[r][real[r]])), [1, 2, 3])
    # Varied ids in range and replayed routing: top-k distinct, in range, on every real token.
    ids = p.completion_ids[real]
    self.assertTrue((ids >= 256).all() and (ids < 5000).all() and len(np.unique(ids)) > 1000)
    r = p.routed_experts[real]
    self.assertTrue((r >= 0).all() and (r < EXPERTS).all())
    srt = np.sort(r, axis=-1)
    self.assertFalse((srt[..., 1:] == srt[..., :-1]).any())
    # Rollout logps: the synthetic constant on scored tokens, 0 elsewhere.
    self.assertTrue((p.rollout_per_token_logps[p.completion_mask > 0] == np.float32(-0.7)).all())
    self.assertTrue((p.rollout_per_token_logps[p.completion_mask == 0] == 0).all())

  def test_routing_follows_targets(self):
    targets = np.array([1.0, 1.3, 1.6, 2.0])
    ratio, pad = d.payload_shard_load_ratio(build(), num_experts=EXPERTS, ep=EP, **RATIO_KW)
    self.assertAlmostEqual(pad, 0.07, delta=0.001)
    # The replayed share counts padding in the balanced load, so scale by the real fraction before comparing.
    np.testing.assert_allclose(ratio / 0.93, targets, atol=0.25)
    self.assertEqual(int(np.argmax(ratio)), 3)
    uniform, _ = d.payload_shard_load_ratio(build(routing="uniform"), num_experts=EXPERTS, ep=EP, **RATIO_KW)
    # Small-shape sampling noise lifts the max over chunks above 1; the hot layer still stands well clear of it.
    self.assertLess(uniform.max() / 0.93, 1.35)
    self.assertGreater(ratio[3], uniform.max() + 0.4)

  def test_routing_pool_bisection(self):
    rng = np.random.default_rng(0)
    for target in (1.25, 1.75, 2.5):
      pool = d.routing_pool(target, EXPERTS, EP, TOP_K, 8192, rng, hot_shard=3)
      self.assertAlmostEqual(d._pool_ratio(pool, EXPERTS, EP), target, delta=0.1)  # pylint: disable=protected-access
      counts = np.bincount((pool // (EXPERTS // EP)).ravel(), minlength=EP)
      self.assertEqual(int(np.argmax(counts)), 3)

  def test_final_micro_batch_pad_rows(self):
    p = build(pad_rows_fraction=0.5, seed=1)
    real = p.segment_ids > 0
    self.assertFalse(real[2:].any())  # the last half of the rows are entirely padding
    self.assertAlmostEqual(real[:2].mean(), 0.93, delta=0.001)
    self.assertTrue((p.routed_experts[2:] == d.UNSET_ROUTED_EXPERT).all())

  def test_advantages_random_keeps_everything_else(self):
    a, b = build(), build(advantages="random")
    for f in (
        "completion_ids",
        "segment_ids",
        "segment_positions",
        "completion_mask",
        "routed_experts",
        "rollout_per_token_logps",
    ):
      np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
    scored = a.completion_mask > 0
    self.assertEqual(set(np.unique(a.advantages[scored])), {-1.0, 1.0})  # inherited from the base
    self.assertGreater(len(np.unique(b.advantages[scored])), 3)
    # One advantage per segment.
    for r in range(ROWS):
      for s in (1, 2, 3):
        vals = np.unique(b.advantages[r][(b.segment_ids[r] == s) & scored[r]])
        self.assertEqual(len(vals), 1)
    with self.assertRaises(ValueError):
      build(advantages="bogus")

  def test_no_routed_experts(self):
    p = d.realistic_payload(warmup_like(routed=False), num_experts=EXPERTS, ep=EP, seed=0, vocab_size=5000)
    self.assertIsNone(p.routed_experts)

  def test_seeded(self):
    np.testing.assert_array_equal(build().routed_experts, build().routed_experts)
    self.assertFalse(np.array_equal(build().completion_ids, build(seed=5).completion_ids))


class CalibrationTest(unittest.TestCase):

  def test_probe_index_to_layer_is_a_permutation(self):
    layers = [d.probe_index_to_layer(k) for k in range(60)]
    self.assertEqual(sorted(layers), list(range(60)))
    self.assertEqual(d.probe_index_to_layer(59), 59)
    self.assertEqual(d.probe_index_to_layer(1), 4)

  def test_layer_targets(self):
    t = d.layer_targets(60)
    self.assertEqual(t.shape, (60,))
    self.assertAlmostEqual(t.max(), max(d.PROBE_REGULAR_MEAN_RATIO))
    np.testing.assert_allclose(d.layer_targets(60, scale=2.0) - 1.0, 2.0 * (t - 1.0))
    self.assertEqual(d.layer_targets(8).shape, (8,))  # another depth cycles the profile
    own = tuple(1.0 + 0.01 * k for k in range(60))
    t_own = d.layer_targets(60, probe=own)
    self.assertAlmostEqual(t_own[d.probe_index_to_layer(10)], 1.10)

  def test_parse_routing_targets(self):
    self.assertEqual(d.parse_routing_targets("1.5, 2.0,1.0", 3), (1.5, 2.0, 1.0))
    with self.assertRaises(ValueError):
      d.parse_routing_targets("1.5,2.0", 3)
    with self.assertRaises(ValueError):
      d.parse_routing_targets("0.5,2.0", 2)


class HelpersTest(unittest.TestCase):

  def test_strip_logp_gates(self):
    argv = [
        "--a=1",
        "--seq_logprob_error_threshold=2.0",
        "--truncated_importance_sampling_type",
        "seq-mask-tis",
        "--truncated_importance_sampling_ratio_min=0.999",
        "--truncated_importance_sampling_ratio=1.002",
        "--b=2",
    ]
    self.assertEqual(d.strip_logp_gates(argv), ["--a=1", "--b=2"])

  def test_summarize_metrics(self):
    wm = lambda s, den, min_denom=None: types.SimpleNamespace(  # pylint: disable=unnecessary-lambda-assignment
        unreduced_sum=np.array(s), denominator=np.array(den), eps=None, min_denom=min_denom
    )
    buf = types.SimpleNamespace(
        weighted_metrics={"loss": wm([2.0, 0.0, -3.0], [4.0, 0.0, 6.0]), "kept": wm([0.0], [0.0], min_denom=1.0)},
        scalar_metrics={"gradient_norm": np.array([0.125]), "is_ratio/mean": np.array([1.0, 1.0, 1.0])},
    )
    s = d.summarize_metrics(buf)
    self.assertAlmostEqual(s["loss"], -0.1)
    self.assertEqual(s["micro_denominator"], [4.0, 0.0, 6.0])
    self.assertTrue(np.isnan(s["micro_loss"][1]))
    self.assertEqual(s["weighted"]["kept"], 0.0)  # min_denom keeps an empty metric finite
    self.assertEqual(s["scalar"], {"gradient_norm": 0.125, "is_ratio/mean": [1.0, 1.0, 1.0]})
    loss_line, metrics_line = d.metric_log_lines(3, s)
    self.assertTrue(loss_line.startswith("STEP_LOSS step=3 loss=-0.1 grad_norm=0.125 micro_loss=[0.5,nan,-0.5]"))
    self.assertIn("is_ratio/mean=[1,1,1]", metrics_line)


if __name__ == "__main__":
  unittest.main()
