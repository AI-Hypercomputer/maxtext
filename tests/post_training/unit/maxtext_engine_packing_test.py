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
"""Sequence-packing support in the MaxText -> Tunix adapter (CPU-only).

Tunix decides whether a model can consume packed batches by inspecting its
``__call__`` signature for a parameter named exactly ``segment_ids``, with a
``**kwargs`` escape hatch (``tunix/rl/common.py::_call_contains_by_type``).
When that check fails, ``compute_per_token_logps`` takes its fallback branch and
passes ``attention_mask`` instead -- which for a *packed* batch is ``None``,
because ``process_ids`` returns no mask when ``segment_ids`` is supplied.

The adapter accepts ``attention_mask`` and never reads it, so the failure is
silent: MaxText falls back to causal-only masking over the whole packed row,
every sequence attends to every sequence before it, and the loss stays finite.

These tests pin that contract layer by layer, innermost first:

* ``AdapterSegmentIdsGateTest`` -- the signature gate itself, on the class.
* ``PackedVersusUnpackedLogpsTest`` -- the numerical consequence, through a real
  (tiny, randomly initialized) MaxText Transformer.
* ``PackedVersusUnpackedGradientsTest`` -- the same property one layer out,
  through ``MaxTextTrainingEngine.fwd_bwd`` driving tunix's real
  ``grpo_loss_fn``. The forward test above cannot see anything the loss
  aggregation, the accumulator denominator or the static/dynamic batch split get
  wrong, and those are the parts a packed batch reaches only via the engine.
* ``PackedBatchStaticsTest`` -- that ``num_segments`` crosses the jit boundary as
  a *static* value and that changing it forces a recompile.
* ``PackedDenominatorPartitionsTest`` -- the same accumulation identity over four
  *arbitrary* partitions into packed micro-batches, where the fixed slot count is
  unequally filled. An off-by-one divisor is invisible at two micro-batches.
* ``PackedUpdateCadenceTest`` -- one layer further still: that ``fwd_bwd``
  accumulates without applying and ``update`` applies exactly once. The only
  tests here that call ``update()`` at all.
* ``PackedCompiledPathTest`` -- the same work on the *compiled* branch of
  ``fwd_bwd``. Every class above it runs the eager one, and the orchestrator
  never does.
* ``ClosedFormPackedLossTest`` -- the packed loss against a hand-computed number
  rather than against an unpacked run. Everything above compares the two layouts,
  which cannot catch a fault they share; aggregating by row instead of by segment
  is exactly such a fault, since by-row is what the unpacked layout should do.

Lives under ``tests/post_training/`` because it imports tunix, which is only
installed for post-training test environments.
"""

import dataclasses
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.integration.tunix import tunix_adapter as tunix_adapter_module
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.models import models
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

from tunix.rl import algo_core
from tunix.rl import common as rl_common


# Without this the file is deselected by every CI job: tests/conftest.py
# auto-marks it cpu_only, and the cpu/tpu post-training jobs additionally
# filter on post_training while the unit jobs --ignore tests/post_training.
pytestmark = [pytest.mark.post_training]


# A real key in HF_MODEL_CONFIGS, so the adapter constructor's
# `HF_MODEL_CONFIGS[model_name].to_dict()` lookup resolves. Every dimension is
# overridden below via `override_model_config`, so nothing 0.6b-sized is built.
_MODEL_NAME = "qwen3-0.6b"

_NUM_SEQ = 3
_SEQ_LEN = 4
_PACKED_LEN = _NUM_SEQ * _SEQ_LEN

# The prompt/completion split within each sequence, for the engine-level tests.
# Only the completion half is scored, which is what makes the per-segment
# denominator smaller than the segment and therefore able to disagree.
_PROMPT_LEN = 2
_COMPLETION_LEN = _SEQ_LEN - _PROMPT_LEN


def _scored_lens(n):
  """Scored-token counts for `n` consecutive sequences: 2, 1, 2, 1, ...

  Deliberately unequal. `sequence-mean-token-mean`, what `_GrpoConfig` uses,
  averages per-segment means; with equal counts that is algebraically the mean
  over their union, i.e. what the unsegmented row branch computes. A uniform
  fixture therefore *executes* `_aggregate_loss_segmented` every step without
  ever distinguishing it from the code it replaces, so a packed run that reduced
  per row would pass. `ClosedFormPackedLossTest` pins that at the loss level;
  this carries it into the gradient and accumulation tests.

  A function of position, not of token ids, so packed and unpacked builders
  score the same tokens without agreeing on anything but the slice they were
  handed. Callers partitioning a block across micro-batches must pass the
  matching slice of the *block's* counts rather than recomputing from 0 -- see
  `PackedDenominatorPartitionsTest._accumulate`.

  Per *sequence*, so denominators here (which count scored sequences, not
  tokens) are unaffected: 1 <= count <= `_COMPLETION_LEN` keeps every live
  segment scored, and an unscored segment is the padding bucket's case.
  """
  assert _COMPLETION_LEN >= 2, "unequal segment lengths need at least two scored slots"
  return np.array([_COMPLETION_LEN - (i % 2) for i in range(n)], np.int32)


# `pad_id=0` matches the token draws below, which start at 1. `eos_id=-1` can
# never equal a real token, so no sequence is truncated at an id that happened to
# land in a random draw.
_PAD_ID = 0
_EOS_ID = -1

# The shared config warms the learning rate up from zero over 15000 steps, so it
# is *exactly* 0.0 at step 0 and 2e-09 at step 1. An `update()` there is a
# correctly-behaving no-op, which means "the weights did not change" holds
# whether or not the engine works -- so only the cadence test, which needs
# weights that visibly move, overrides it. Everything else keeps the default.
_LIVE_LR = {"learning_rate": 0.1, "warmup_steps_fraction": 0.0}


def _tiny_cfg(seq_len, run_name, **overrides):
  """A CPU-sized, single-layer Qwen3 with a real (random) parameter tree.

  `overrides` go straight to `pyconfig.initialize`, for the one test that needs a
  learning rate that actually moves a weight -- see `_LIVE_LR`.
  """
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path(), "attention=dot_product"],
      override_model_config=True,
      **overrides,
      model_name=_MODEL_NAME,
      base_emb_dim=64,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=32,
      base_mlp_dim=64,
      base_num_decoder_layers=1,
      num_decoder_layers=1,
      scan_layers=False,
      vocab_size=64,
      max_target_length=seq_len,
      max_prefill_predict_length=seq_len,
      per_device_batch_size=1.0,
      # float32 throughout: the packed-vs-unpacked comparison is an exact
      # algebraic identity, so bf16 rounding would only blur the signal.
      dtype="float32",
      weight_dtype="float32",
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      run_name=run_name,
  )


class AdapterSegmentIdsGateTest(unittest.TestCase):
  """The signature gate, checked against the adapter class itself.

  This needs no model: `model_call_contains` only inspects
  `type(model).__call__`. Uses the stub-base pattern from
  `tunix_adapter_test.py` so the constructor runs without building a MaxText model.
  """

  _STUB_MODEL_NAME = "_stub_model"

  def setUp(self):
    super().setUp()
    weight_mapping_patcher = mock.patch.object(tunix_adapter_module, "VllmWeightMapping")
    weight_mapping_patcher.start()
    self.addCleanup(weight_mapping_patcher.stop)

    hf_configs_patcher = mock.patch.dict(
        tunix_adapter_module.HF_MODEL_CONFIGS,
        {self._STUB_MODEL_NAME: SimpleNamespace(to_dict=lambda: {})},
    )
    hf_configs_patcher.start()
    self.addCleanup(hf_configs_patcher.stop)

    base = SimpleNamespace(config=SimpleNamespace(model_name=self._STUB_MODEL_NAME))
    self.adapter = TunixMaxTextAdapter(base_model=base, pad_id=None)

  def test_adapter_passes_tunix_segment_ids_gate(self):
    """Tunix must recognise the adapter as segment-aware.

    This is the single check that decides whether every packed batch reaches
    MaxText with its sequence boundaries intact or silently loses them.
    """
    self.assertTrue(
        rl_common.model_call_contains(self.adapter, "segment_ids"),
        "Tunix looks for a parameter named exactly 'segment_ids' (or **kwargs) "
        "on TunixMaxTextAdapter.__call__. Without it, compute_per_token_logps "
        "drops the packing boundaries and MaxText attends across sequences.",
    )


class PackedVersusUnpackedLogpsTest(unittest.TestCase):
  """Packed log-probs must equal unpacked log-probs, sequence for sequence.

  The core correctness property of packing: concatenating N sequences into one
  row must not let any of them see the others. Runs a real MaxText Transformer
  (tiny and randomly initialized -- the weights are irrelevant, only that both
  sides use the *same* ones) rather than a segment-aware toy. That distinction
  is the whole point: tunix's own packing suite uses a toy whose `__call__`
  names its parameter `segment_ids`, so the toy passes the gate by construction
  and the suite stays green against an adapter that does not.
  """

  def setUp(self):
    super().setUp()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

    cfg = _tiny_cfg(_PACKED_LEN, "maxtext_engine_packing_test")
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    base = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))

    # pad_id=None so the adapter never synthesizes segment ids of its own; the
    # only boundaries in play are the ones the caller passes, which is what
    # makes a failure unambiguous.
    adapter = TunixMaxTextAdapter(base_model=base, pad_id=None)
    self.graphdef, self.state = nnx.split(adapter)

  def _logps(self, tokens, segment_ids, segment_positions):
    """Per-token log-probs for an already-packed buffer.

    `prompt_tokens` is `[B, 0]` on both paths: under packing tunix collapses the
    prompt/completion split and hands everything over as one stream. Passing
    `segment_ids` on the unpacked path too (all ones, one segment per row) keeps
    both sides on the same code path in `compute_per_token_logps`, so the
    comparison isolates attention rather than the packed/unpacked plumbing.
    """
    tokens = jnp.asarray(tokens, jnp.int32)
    return np.asarray(
        rl_common.compute_per_token_logps(
            self.graphdef,
            self.state,
            jnp.zeros((tokens.shape[0], 0), jnp.int32),
            tokens,
            pad_id=0,
            eos_id=-1,
            segment_ids=jnp.asarray(segment_ids, jnp.int32),
            segment_positions=jnp.asarray(segment_positions, jnp.int32),
        )
    )

  def test_packed_logps_match_unpacked_per_segment(self):
    # Token ids start at 1: 0 is pad_id, and a pad token in the middle of a
    # packed row would confound "boundaries were honoured" with "padding was
    # masked".
    tokens = np.random.default_rng(7).integers(1, 64, size=(_NUM_SEQ, _SEQ_LEN)).astype(np.int32)

    unpacked = self._logps(
        tokens,
        np.ones((_NUM_SEQ, _SEQ_LEN), np.int32),
        np.broadcast_to(np.arange(_SEQ_LEN, dtype=np.int32), (_NUM_SEQ, _SEQ_LEN)),
    )

    # One row, _NUM_SEQ segments. Segment ids are 1-based; 0 is the padding
    # bucket. Positions restart at every boundary.
    packed = self._logps(
        tokens.reshape(1, -1),
        np.concatenate([np.full(_SEQ_LEN, i + 1, np.int32) for i in range(_NUM_SEQ)])[None, :],
        np.concatenate([np.arange(_SEQ_LEN, dtype=np.int32) for _ in range(_NUM_SEQ)])[None, :],
    ).reshape(_NUM_SEQ, _SEQ_LEN)

    # Column 0 is skipped on both sides. Under packing `compute_per_token_logps`
    # drops the first prediction of the row and front-pads the result with 0.0
    # to restore the full width, so position 0 of every segment is either that
    # pad or a cross-boundary prediction that `completion_mask` discards
    # downstream. Comparing it would assert on a value neither path defines.
    np.testing.assert_allclose(packed[:, 1:], unpacked[:, 1:], atol=1e-4, rtol=1e-4)

  def test_segment_positions_matter_only_up_to_a_per_segment_offset(self):
    """What a packer must guarantee about `segment_positions`, and what it need not.

    Measured, not assumed. RoPE encodes position *relatively*: an attention score
    depends on `pos_q - pos_k`, so adding a constant to every position in a
    segment cancels. Because `segment_ids` already stops any query attending
    outside its own segment, a packer that numbers positions continuously across
    the row (0..11) instead of restarting them (0..3, 0..3, 0..3) produces the
    same logits to fp32 noise -- each segment is merely shifted by a constant.

    Worth pinning for two opposite reasons. It stops someone "fixing" a packer
    that emits running positions, which is not broken. And it marks the limit of
    that licence: the invariance is a property of *relative* position encoding
    holding together with correct segmentation, so it would not survive a learned
    absolute embedding, and it says nothing about ordering *within* a segment --
    asserted below, so this test cannot pass by the model ignoring positions
    altogether.
    """
    tokens = np.random.default_rng(7).integers(1, 64, size=(1, _PACKED_LEN)).astype(np.int32)
    seg_ids = np.repeat(np.arange(1, _NUM_SEQ + 1), _SEQ_LEN)[None, :].astype(np.int32)

    restart = np.tile(np.arange(_SEQ_LEN), _NUM_SEQ)[None, :].astype(np.int32)
    running = np.arange(_PACKED_LEN, dtype=np.int32)[None, :]
    # A within-segment permutation, which no constant offset can produce.
    scrambled = np.tile(np.array([0, 2, 1, 3]), _NUM_SEQ)[None, :].astype(np.int32)

    base = self._logps(tokens, seg_ids, restart)
    np.testing.assert_allclose(self._logps(tokens, seg_ids, running), base, atol=1e-4, rtol=1e-4)

    # The control. Without it, a model that ignored `positions` entirely
    # would satisfy the assertion above and this test would prove nothing.
    self.assertGreater(
        float(np.max(np.abs(self._logps(tokens, seg_ids, scrambled) - base))),
        1e-2,
        "reordering positions within a segment changed nothing, so `positions` is being ignored "
        "and the invariance asserted above is vacuous",
    )


@dataclasses.dataclass(frozen=True)
class _GrpoConfig:
  """The fields `algo_core.grpo_loss_fn` reads off its algo_config.

  Two of these settings are required by the tests below rather than arbitrary.

  `loss_agg_mode="sequence-mean-token-mean"`. `token-mean` -- the default
  everywhere else -- divides one global sum by one global token count and never
  groups by row or by segment. A packed run and an unpacked run therefore agree
  under it *even with segmentation completely broken*, so a packing test written
  against `token-mean` is green by construction. This is the cheapest mode whose
  arithmetic can tell the two layouts apart: it means over tokens within a
  sequence, then over sequences, and "sequence" is exactly the thing packing
  redefines.

  `beta=0.0`, paired with `ref_per_token_logps=None` below, drops the KL term.
  That is not just a simplification. The obvious way to build a reference is to
  run the model being tested, but then a model whose attention leaks across
  segment boundaries produces a *contaminated reference too*, the two cancel in
  the ratio, and the loss comes out clean while the gradient is wrong. Measured
  on this path: with the pre-`segment_ids` adapter, packed and unpacked losses
  agreed to seven decimals while their gradients differed by 70% in relative L2.
  Removing the term removes the cancellation channel entirely.
  """

  beta: float = 0.0
  epsilon: float = 0.2
  epsilon_high: float = 0.2
  loss_algo: str = "grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  temperature: float = 1.0
  kl_loss_mode: str = "low_var_kl"
  kl_clamp_value: float | None = None


def _sequences(seed=11, n=_NUM_SEQ):
  """An `n` x `_SEQ_LEN` token block both layouts are built from.

  Drawn once and shared, so the packed and unpacked examples differ in layout
  and in nothing else. Ids start at 1 because 0 is `_PAD_ID`, and a pad token
  inside a packed row would confound "boundaries were honoured" with "padding
  was masked".
  """
  rng = np.random.default_rng(seed)
  tokens = rng.integers(1, 64, size=(n, _SEQ_LEN)).astype(np.int32)
  advantages = rng.normal(size=(n,)).astype(np.float32)
  return tokens, advantages


def _unpacked_example(tokens, advantages, n=_NUM_SEQ, scored_lens=None):
  """`n` rows of one sequence each -- the layout packing has to reproduce."""
  tokens, advantages = tokens[:n], advantages[:n]
  # Row i scores its own number of completion slots, matching segment i of the
  # packed row built from the same sequences. Unequal by design; see
  # `_scored_lens`.
  lens = _scored_lens(n) if scored_lens is None else np.asarray(scored_lens)[:n]
  scored = np.arange(_COMPLETION_LEN)[None, :] < lens[:, None]
  return rl_common.TrainExample(
      prompt_ids=jnp.asarray(tokens[:, :_PROMPT_LEN]),
      prompt_mask=jnp.ones((n, _PROMPT_LEN), jnp.int32),
      completion_ids=jnp.asarray(tokens[:, _PROMPT_LEN:]),
      completion_mask=jnp.asarray(scored.astype(np.int32)),
      advantages=jnp.asarray(advantages),
      # See _GrpoConfig: no reference, so no cancellation channel.
      ref_per_token_logps=None,
      # None makes the importance ratio exactly 1, which is its value at the
      # first inner step of a real GRPO update anyway.
      old_per_token_logps=None,
  )


def _packed_example(tokens, advantages, n=_NUM_SEQ):
  """One `_NUM_SEQ`-slot row holding the first `n` sequences."""
  return _packed_row(tokens[:n], advantages[:n], _NUM_SEQ)


def _packed_row(tokens, advantages, slots, scored_lens=None):
  """One row of `slots` segment-sized slots, holding `len(tokens)` sequences.

  Any slots beyond the supplied sequences become the *padding bucket*: token ids
  set to `_PAD_ID`, segment id 0, zero mask and zero advantage. That is what a
  real FFD packer emits for a row it could not fill, and it is the case where a
  segment-blind denominator goes wrong -- `num_segments` stays at `slots + 1`
  (it is a static bucket count, not a count of what is occupied) while the number
  of *active* segments drops, so anything reading the former as the latter
  divides by the wrong number.

  Three fields change shape under packing, all of them forced by tunix's packed
  contract rather than chosen here:

  * `prompt_ids` is `[1, 0]`. Packing collapses the prompt/completion split --
    a row is one undivided stream -- so every token lives in `completion_ids`.
  * `advantages` becomes per-token `[1, packed_len]`. One scalar per row could
    not distinguish the sequences sharing that row, so each segment's scalar is
    broadcast across its own tokens.
  * `completion_mask` masks each segment's prompt span too, leaving the tokens
    the unpacked layout scores -- `_scored_lens`, unequal across a row's
    segments on purpose. Without it the two would disagree over the
    denominator: a real disagreement, but not the one under test.
  """
  n = len(tokens)
  live = np.arange(slots) < n
  within = np.arange(_SEQ_LEN)
  # Scored iff the slot holds a real sequence and the position is past that
  # sequence's prompt and within its own scored length. Segment i matches row i
  # of the unpacked layout, so the denominators agree while the segments differ
  # from each other -- which is what makes the segmented reduction observable.
  lens = _scored_lens(n) if scored_lens is None else np.asarray(scored_lens)[:n]
  per_seg_mask = np.zeros((slots, _SEQ_LEN), np.int32)
  per_seg_mask[:n] = (within >= _PROMPT_LEN) & (within < _PROMPT_LEN + lens[:, None])
  per_seg_mask = per_seg_mask.reshape(-1)

  # Dead slots are overwritten below, so what they are padded with here is
  # arbitrary; zeros keep the intent obvious.
  filled = np.concatenate([np.asarray(tokens), np.zeros((slots - n, _SEQ_LEN), np.int32)])
  filled_adv = np.concatenate([np.asarray(advantages), np.zeros(slots - n, np.float32)])

  ids = np.where(np.repeat(live, _SEQ_LEN), filled.reshape(-1), _PAD_ID).astype(np.int32)
  # Real segment ids are 1-based; dead slots take 0, the padding bucket.
  seg_ids = np.where(np.repeat(live, _SEQ_LEN), np.repeat(np.arange(1, slots + 1), _SEQ_LEN), 0).astype(np.int32)
  adv = np.where(np.repeat(live, _SEQ_LEN), np.repeat(filled_adv, _SEQ_LEN), 0.0).astype(np.float32)

  return rl_common.TrainExample(
      prompt_ids=jnp.zeros((1, 0), jnp.int32),
      prompt_mask=jnp.zeros((1, 0), jnp.int32),
      completion_ids=jnp.asarray(ids[None, :]),
      completion_mask=jnp.asarray(per_seg_mask[None, :]),
      advantages=jnp.asarray(adv[None, :]),
      ref_per_token_logps=None,
      old_per_token_logps=None,
      # 0 is the padding bucket, which `num_segments` counts -- hence
      # `slots + 1` and not `slots`. It is a function of `slots`, never of how
      # many are occupied: it is the static bucket count the segmented reductions
      # are shaped by, so letting it track the live count would recompile on
      # every unequally filled micro-batch, which is most of them.
      segment_ids=jnp.asarray(seg_ids[None, :]),
      # Positions restart at every boundary. Letting them run on would place
      # segment 2 as though it continued segment 1 under RoPE, which is a
      # distinct bug from attention crossing the boundary and would survive a
      # test that only checked the mask.
      segment_positions=jnp.asarray(np.tile(np.arange(_SEQ_LEN), slots)[None, :].astype(np.int32)),
      num_segments=slots + 1,
  )


def _grpo_model_input(algo_config):
  """`grpo_loss_fn`'s keyword arguments, exactly as the orchestrator builds them.

  Mirrors `tunix/experimental/orchestrator/algorithm_adapter.py::_algo_model_input`,
  which passes the payload straight through. Using tunix's loss unadapted is the
  point: an adapter written for the test could paper over a mismatch the real
  pipeline would hit.
  """
  return lambda payload: {
      "train_example": payload,
      "algo_config": algo_config,
      "pad_id": _PAD_ID,
      "eos_id": _EOS_ID,
  }


def _engine(cfg, mesh, algo_config):
  engine = maxtext_engine.MaxTextTrainingEngine(
      cfg,
      mesh=mesh,
      # grpo_loss_fn calls the model with tunix's signature, so the adapter is
      # needed for the loss itself, not only for weight sync.
      wrap_with_tunix_adapter=True,
      tokenizer_pad_id=_PAD_ID,
  )
  return engine.with_loss_fn(algo_core.grpo_loss_fn, has_aux=True).with_gen_model_input_fn(_grpo_model_input(algo_config))


def _live_engine(run_name):
  """An engine whose optimizer actually moves weights. See `_LIVE_LR`."""
  cfg = _tiny_cfg(_PACKED_LEN, run_name, **_LIVE_LR)
  return _engine(cfg, Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes), _GrpoConfig())


def _params(engine):
  """Every parameter leaf as numpy, in a stable order."""
  return [np.asarray(leaf) for leaf in jax.tree.leaves(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param)))]


def _grads_after(engine, payload):
  """The whole accumulated gradient tree from one micro-batch, as a flat list."""
  engine.fwd_bwd(payload)
  grads = engine._reduced_accumulated_grads()  # pylint: disable=protected-access
  return [np.asarray(leaf, np.float64) for leaf in jax.tree.leaves(grads)]


def _rel_l2(a, b):
  """Relative L2 over a whole flattened tree, as one number."""
  num = sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b))
  den = sum(float(np.sum(y**2)) for y in b)
  return float(np.sqrt(num / den)) if den else float(np.sqrt(num))


class PackedVersusUnpackedGradientsTest(unittest.TestCase):
  """Packed and unpacked gradients must agree, through the engine.

  `PackedVersusUnpackedLogpsTest` proves the model attends correctly. It cannot
  prove anything about what the engine then does with the result: segmented loss
  aggregation, the summed-denominator accumulation, and the static/dynamic split
  that decides whether `num_segments` is traced or closed over all sit between
  that forward pass and a weight update, and all three are packing-sensitive.

  Compares gradients rather than loss. That is not caution -- it is the finding
  that motivated this test. See `_GrpoConfig` for the measurement.
  """

  def setUp(self):
    super().setUp()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    self.algo_config = _GrpoConfig()

  def _fresh_engine(self):
    cfg = _tiny_cfg(_PACKED_LEN, "maxtext_engine_packed_grads_test")
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    return _engine(cfg, mesh, self.algo_config)

  def test_packed_gradients_match_unpacked_over_the_whole_tree(self):
    tokens, advantages = _sequences()

    # Two engines rather than one reset between runs: nothing in the engine's
    # public surface clears an accumulator without also applying it, and reaching
    # into `_accumulated_grads` to zero it would be testing a state this code
    # path never actually occupies.
    unpacked_engine = self._fresh_engine()
    packed_engine = self._fresh_engine()

    # The comparison is only meaningful if both engines started from identical
    # weights. `init_weights_seed` should guarantee it; asserted rather than
    # assumed, because if it ever stops holding this test would fail in a way
    # that looks exactly like a packing bug.
    before_unpacked = jax.tree.leaves(nnx.to_pure_dict(nnx.state(unpacked_engine.model, nnx.Param)))
    before_packed = jax.tree.leaves(nnx.to_pure_dict(nnx.state(packed_engine.model, nnx.Param)))
    for a, b in zip(before_unpacked, before_packed):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    unpacked = _grads_after(unpacked_engine, _unpacked_example(tokens, advantages))
    packed = _grads_after(packed_engine, _packed_example(tokens, advantages))

    self.assertEqual(len(packed), len(unpacked))
    self.assertTrue(all(np.all(np.isfinite(g)) for g in packed), "packed gradients are not finite")
    # A tree of zeros would satisfy every comparison below while meaning that no
    # gradient was computed at all.
    self.assertGreater(sum(float(np.sum(np.abs(g))) for g in unpacked), 0.0, "unpacked gradients are all zero")

    # Whole tree, not a sample. Recorded from a prior investigation on this same
    # stack: three sampled tensors matched their source exactly and the aggregate
    # over all 310 showed the transfer delivering under one percent. Sampling did
    # not merely miss the defect, it pointed the work the wrong way for three runs.
    for i, (p, u) in enumerate(zip(packed, unpacked)):
      np.testing.assert_allclose(p, u, atol=2e-4, rtol=2e-4, err_msg=f"gradient leaf {i} of {len(packed)} differs")

    # The per-leaf check above is tolerant near zero, where most leaves live.
    # This one is not: a uniform small bias across the tree passes elementwise and
    # fails here.
    self.assertLess(_rel_l2(packed, unpacked), 1e-3)

  def test_packed_denominator_counts_segments_not_rows(self):
    """The accumulator's denominator must see three sequences, not one row.

    Under `sequence-mean-token-mean` the denominator is the number of scored
    *sequences*. The unpacked layout has three rows and the packed layout has one
    row holding three segments, so a denominator that counted rows would read 1
    against 3 -- scaling every packed gradient by three while leaving the loss
    itself plausible. This is the specific arithmetic `_aggregate_loss_segmented`
    exists to get right, and the engine divides by it once in `_update_kernel`.
    """
    tokens, advantages = _sequences()

    unpacked_engine = self._fresh_engine()
    unpacked_engine.fwd_bwd(_unpacked_example(tokens, advantages))
    packed_engine = self._fresh_engine()
    packed_engine.fwd_bwd(_packed_example(tokens, advantages))

    unpacked_denom = float(np.asarray(unpacked_engine._accumulated_denominator))  # pylint: disable=protected-access
    packed_denom = float(np.asarray(packed_engine._accumulated_denominator))  # pylint: disable=protected-access

    self.assertEqual(packed_denom, unpacked_denom)
    self.assertEqual(packed_denom, float(_NUM_SEQ))

  def test_packed_accumulation_over_uneven_micro_batches(self):
    """Two packed micro-batches with different active-segment counts.

    Micro-batch A carries three live segments and B carries two plus a padding
    bucket, so the correct total weights A's contribution 3/5 and B's 2/5, while
    a mean of per-micro-batch means would weight them 1/2 each. Measured on this
    split, that is a 10% error in the resulting gradient tree -- with no NaN, no
    warning and a perfectly plausible loss. Packing is the regime where it
    actually bites: an FFD packer deliberately emits unequally filled rows, so
    unequal denominators are the norm rather than an edge case.

    Two assertions doing two jobs. The denominators pin the *policy*: 5, not 4
    (which is what counting `num_segments` buckets rather than active segments
    gives) and not 2.5. The gradient comparison pins packed/unpacked
    *equivalence* across an accumulation window -- and it is not vacuous, since
    the two micro-batches' normalized gradients differ by 96% in relative L2, so
    how they are weighted genuinely changes the answer.
    """
    tokens, advantages = _sequences()

    unpacked_engine = self._fresh_engine()
    unpacked_engine.fwd_bwd(_unpacked_example(tokens, advantages, n=_NUM_SEQ))
    unpacked = _grads_after(unpacked_engine, _unpacked_example(tokens, advantages, n=2))

    packed_engine = self._fresh_engine()
    packed_engine.fwd_bwd(_packed_example(tokens, advantages, n=_NUM_SEQ))
    packed = _grads_after(packed_engine, _packed_example(tokens, advantages, n=2))

    # 3 live segments then 2, summed -- not averaged to 2.5, and not 4 (which is
    # what counting `num_segments` buckets instead of active ones would give).
    self.assertEqual(
        float(np.asarray(packed_engine._accumulated_denominator)),  # pylint: disable=protected-access
        float(_NUM_SEQ + 2),
    )
    self.assertEqual(
        float(np.asarray(unpacked_engine._accumulated_denominator)),  # pylint: disable=protected-access
        float(_NUM_SEQ + 2),
    )

    for i, (p, u) in enumerate(zip(packed, unpacked)):
      np.testing.assert_allclose(p, u, atol=2e-4, rtol=2e-4, err_msg=f"accumulated gradient leaf {i} differs")
    self.assertLess(_rel_l2(packed, unpacked), 1e-3)


class PackedBatchStaticsTest(unittest.TestCase):
  """`num_segments` must cross the jit boundary as a static value.

  `_aggregate_loss_segmented` passes it to `jax.ops.segment_sum` as
  `num_segments`, which is a shape and so cannot be a tracer. Tunix keeps it
  static by declaring it `flax.struct.field(pytree_node=False)`, which puts it in
  the *treedef* rather than among the leaves. Nothing in the engine asserts that,
  and the failure mode if it changed upstream is not a wrong number but a
  `ConcretizationTypeError` from deep inside the loss.

  The second test covers the other half: a value in the treedef changes the batch
  signature, so a differently packed batch recompiles rather than silently reusing
  a kernel built for the wrong segment count.
  """

  def setUp(self):
    super().setUp()
    tokens, advantages = _sequences()
    self.batch = _grpo_model_input(_GrpoConfig())(_packed_example(tokens, advantages))

  def test_num_segments_is_not_a_traced_leaf(self):
    dynamic, static = maxtext_engine._split_static_and_dynamic(self.batch)  # pylint: disable=protected-access

    # The example itself is traced: it is mostly arrays.
    self.assertIn("train_example", dynamic)
    self.assertNotIn("train_example", static)
    # ... but num_segments is not one of the arrays that gets traced.
    self.assertNotIn(_NUM_SEQ + 1, [leaf for leaf in jax.tree.leaves(dynamic) if isinstance(leaf, int)])
    self.assertEqual(dynamic["train_example"].num_segments, _NUM_SEQ + 1)
    # pad_id/eos_id are Python ints and belong on the closed-over side.
    self.assertEqual(static["pad_id"], _PAD_ID)
    self.assertEqual(static["eos_id"], _EOS_ID)

  def test_changing_num_segments_changes_the_batch_signature(self):
    dynamic, static = maxtext_engine._split_static_and_dynamic(self.batch)  # pylint: disable=protected-access
    signature = maxtext_engine._batch_signature(dynamic, static)  # pylint: disable=protected-access

    other = dataclasses.replace(dynamic["train_example"], num_segments=_NUM_SEQ + 2)
    other_dynamic = dict(dynamic, train_example=other)
    other_signature = maxtext_engine._batch_signature(other_dynamic, static)  # pylint: disable=protected-access

    self.assertNotEqual(signature, other_signature)


class PackedDenominatorPartitionsTest(unittest.TestCase):
  """`Σ grad(sum_i)` over *any* partition into packed micro-batches, and `Σ denom_i`.

  Ported from `tunix/tests/sft/peft_trainer_test.py:1487`, which parameterizes
  the same identity over four partitions of 8-32 examples. T6 above is only the
  two-micro-batch case; the degenerate shapes are what tunix's docstring says the
  parameterization exists for -- "to catch regressions where the divisor drifts
  off-by-one" -- and an off-by-one is invisible at K=2 in a way it is not at K=8.

  The engine version runs one layer further out than tunix's. Tunix drives
  `GradientAccumulator.add(g, denom=...)` with a hand-supplied denom, so its test
  pins the accumulator's arithmetic and nothing about where the denom came from.
  Here it is *derived*, by counting live segments in a packed row, and it is the
  derivation that packing put at risk. So each micro-batch is a real packed row of
  `max(sizes)` slots with `size` occupied, and the reference is unpacked.

  Fixed slot count, unequally filled, is not an artifact: it is what an FFD packer
  emits, and it is the shape that makes `num_segments` (static, `slots + 1`)
  differ from the live count (dynamic, `size`). Anything reading the first as the
  second fails here and passes on a uniform partition.

  **Two references, deliberately, because they carry different tolerances.**
  Measuring first and setting tolerances after, rather than the reverse:

  | partition              | vs matched-partition unpacked | vs one full-batch pass |
  | ---------------------- | ----------------------------- | ---------------------- |
  | `(3, 5, 1, 7)`         | 3.2e-07                       | 1.414e-04              |
  | `(1, 1, 28, 2)`        | 2.7e-07                       | 8.256e-05              |
  | `(8,)`                 | 2.8e-07                       | 2.796e-07              |
  | `(1,) x 8`             | 0.0                           | 1.350e-04              |

  (relative L2 over the whole tree). The right-hand column is *not* a packing
  error. Running the identical partition through **unpacked** micro-batches
  reproduces it to four significant figures -- 1.414e-04, 8.256e-05, 0.0,
  1.350e-04 -- so it is what splitting one batch into K pieces costs in float32,
  and it vanishes at K=1. Packing's own contribution is the left-hand column, at
  the noise floor.

  Hence: the elementwise comparison goes against the matched-partition unpacked
  run, where the tolerance can be tight enough to mean something, and the
  full-batch identity is checked in aggregate at a tolerance the float32 floor
  actually justifies. Asserting both is what justifies the loose tolerance: without
  the tight one there would be no evidence the 1.4e-04 is arithmetic rather than a
  small real defect.
  """

  # tunix's four, unchanged, so a failure here reads against its test directly.
  # Names are tunix's too.
  _PARTITIONS = (
      ("small_pack", (3, 5, 1, 7)),
      ("single_dominant_pack", (1, 1, 28, 2)),
      ("single_pack", (8,)),
      ("many_small_packs", (1,) * 8),
  )

  # Packing's own error, 30x the worst measured above.
  _PACKING_TOL = 1e-5
  # The K-way float32 split, ~7x the worst measured above.
  _SPLIT_TOL = 1e-3

  def setUp(self):
    super().setUp()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    self.algo_config = _GrpoConfig()

  def _mesh_and_cfg(self, slots, name):
    """One config for every engine in a case, so all of them are the same model.

    The unpacked references run their `_SEQ_LEN`-wide rows through the same
    `max_target_length` as the packed row. Identical weights by construction then,
    rather than merely identically seeded.
    """
    cfg = _tiny_cfg(slots * _SEQ_LEN, f"maxtext_engine_denom_{name}")
    return cfg, Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)

  def _packed_builder(self, slots):
    """A `build` for `_accumulate` that puts each micro-batch in one `slots`-slot row.

    A closure over a parameter rather than over a loop variable, so the binding is
    the one the caller meant even when several of these are alive at once.
    """
    return lambda tokens, advantages, _size, scored_lens: _packed_row(tokens, advantages, slots, scored_lens)

  def _accumulate(self, cfg, mesh, build, tokens, advantages, sizes, *, offset=0):
    """Feeds one micro-batch per entry of `sizes` and returns `(Σg, Σdenom)`.

    `Σg` is undivided. `_reduced_accumulated_grads` reduces over *devices*, not
    over the denominator -- the single division by `Σdenom` happens later, in
    `_update_kernel`. So the identity under test here is between sums, and the
    denominator is checked as its own number rather than folded in.

    Scored lengths are cut from one block-wide array and sliced alongside the
    tokens, so sequence `i` scores the same tokens under every partition and
    under the K=1 full pass. Recomputing them per micro-batch would make
    `(3, 5, 1, 7)` score a different set than `(16,)`, breaking the identity for
    a reason unrelated to packing -- as it did on the first attempt.

    `offset` is where `tokens` starts within that block, for callers re-feeding
    one slice alone: without it the slice scores from phase 0 while the
    whole-block pass scores it from phase `offset`. Invisible while the counts'
    period divides the slice width -- which a future edit to a `sizes` tuple
    would silently break.
    """
    engine = _engine(cfg, mesh, self.algo_config)
    block_lens = _scored_lens(offset + sum(sizes))[offset:]
    start = 0
    for size in sizes:
      stop = start + size
      engine.fwd_bwd(build(tokens[start:stop], advantages[start:stop], size, block_lens[start:stop]))
      start = stop
    grads = [np.asarray(leaf, np.float64) for leaf in jax.tree.leaves(engine._reduced_accumulated_grads())]  # pylint: disable=protected-access
    return grads, float(np.asarray(engine._accumulated_denominator))  # pylint: disable=protected-access

  def test_any_partition_into_packed_micro_batches_gives_the_full_batch_gradient(self):
    for name, sizes in self._PARTITIONS:
      with self.subTest(name, sizes=sizes):
        total, slots = sum(sizes), max(sizes)
        tokens, advantages = _sequences(seed=13, n=total)
        cfg, mesh = self._mesh_and_cfg(slots, name)

        packed, denom = self._accumulate(cfg, mesh, self._packed_builder(slots), tokens, advantages, sizes)
        # Same partition, unpacked rows. Isolates packing from the K-way split.
        control, control_denom = self._accumulate(cfg, mesh, _unpacked_example, tokens, advantages, sizes)
        # One pass over everything, K=1. The tunix identity's right-hand side.
        full, full_denom = self._accumulate(cfg, mesh, _unpacked_example, tokens, advantages, (total,))

        # The divisor, as an exact integer. A drift of one shows up here directly
        # instead of having to be reasoned back out of a scale factor.
        self.assertEqual(denom, float(total), f"partition {sizes} summed to {denom}, expected {total}")
        self.assertEqual(control_denom, float(total))
        self.assertEqual(full_denom, float(total))

        # Not vacuous: a tree of zeros would satisfy every comparison below.
        self.assertGreater(sum(float(np.sum(np.abs(g))) for g in full), 0.0, "reference gradients are all zero")

        for i, (p, c) in enumerate(zip(packed, control)):
          np.testing.assert_allclose(
              p, c, atol=self._PACKING_TOL, rtol=self._PACKING_TOL, err_msg=f"leaf {i} of {len(packed)}, {sizes}"
          )
        self.assertLess(_rel_l2(packed, control), self._PACKING_TOL)
        # Aggregate only. Elementwise here would need an atol ~100x looser to
        # absorb the float32 split, which is a tolerance that no longer excludes
        # anything; the tight comparison above is where an elementwise check is informative.
        self.assertLess(_rel_l2(packed, full), self._SPLIT_TOL)

  def test_uniform_micro_batches_agree_with_the_mean_of_their_means(self):
    """The consistency check from `peft_trainer_test.py:1563`, adapted.

    When every micro-batch holds the same number of sequences, `Σg / Σd` and the
    plain mean of the per-micro-batch means must coincide. Tunix compares its two
    accumulator modes; the engine has only the summed-denominator one, so the
    mean-of-means side is built here out of one single-micro-batch engine each.

    That makes it an independent computation rather than a restatement of the test
    above, and it is what catches a divisor summing K+1 instead of K: with uniform
    sizes both routes reach the same number, so they must agree exactly. Note this
    is the one place in the file that divides -- everything else compares undivided
    sums, which cannot see the denominator at all.
    """
    sizes = (4, 4, 4, 4)
    total, slots = sum(sizes), max(sizes)
    tokens, advantages = _sequences(seed=99, n=total)
    cfg, mesh = self._mesh_and_cfg(slots, "uniform")
    build = self._packed_builder(slots)

    summed, denom = self._accumulate(cfg, mesh, build, tokens, advantages, sizes)
    self.assertEqual(denom, float(total))
    summed = [g / denom for g in summed]

    per_micro_batch = []
    for i, size in enumerate(sizes):
      lo, hi = i * size, (i + 1) * size
      grads, d = self._accumulate(cfg, mesh, build, tokens[lo:hi], advantages[lo:hi], (size,), offset=lo)
      self.assertEqual(d, float(size), "a uniform partition's micro-batches must have equal denominators")
      per_micro_batch.append([g / d for g in grads])
    mean_of_means = [np.mean(leaves, axis=0) for leaves in zip(*per_micro_batch)]

    for i, (s, m) in enumerate(zip(summed, mean_of_means)):
      np.testing.assert_allclose(s, m, atol=self._PACKING_TOL, rtol=self._PACKING_TOL, err_msg=f"leaf {i}")
    self.assertLess(_rel_l2(summed, mean_of_means), self._PACKING_TOL)


class PackedUpdateCadenceTest(unittest.TestCase):
  """Accumulate-then-apply, with a packed row at every micro-step.

  Ported from `peft_trainer_test.py:1886,1891` (`test_packing_config_keeps_cond_path`,
  `test_packing_config_respects_skip_step`) and the `:1925` `test_depth2_cadence`
  those lean on.

  Not a literal port, because the two trainers keep the cadence in different
  places. Tunix passes `is_update_step` into a jitted `_train_step` and branches
  on it with `lax.cond`, so its tests can hand in `jnp.array(False)` and assert
  on the traced jaxpr. The engine has no such flag anywhere: `fwd_bwd` and
  `update` are separate Python calls and the cadence belongs to the caller.
  `test_packing_config_keeps_cond_path` therefore has **no analogue and is
  deliberately dropped** -- there is no cond to keep. (Cadence on the engine's
  side of the integration is orchestrator-side; tunix item 14.)

  What carries over is the behaviour under the flag, which is the half that
  matters: a micro-step that is not an update must leave every weight alone and
  the accumulator intact, and the update must then apply exactly once. Nothing
  else in this file calls `update()`, so the whole apply half of the engine is
  unexercised on packed input -- these two tests are the first to reach it.
  """

  def setUp(self):
    super().setUp()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    self.algo_config = _GrpoConfig()

  def _assert_params_equal(self, expected, actual, msg):
    for i, (e, a) in enumerate(zip(expected, actual)):
      np.testing.assert_array_equal(e, a, err_msg=f"{msg}: parameter leaf {i} of {len(expected)}")

  def test_packed_micro_steps_accumulate_and_only_the_update_applies(self):
    engine = _live_engine("maxtext_engine_cadence")
    payload = _packed_example(*_sequences())
    before = _params(engine)

    for micro_step in (1, 2):
      engine.fwd_bwd(payload)
      self._assert_params_equal(before, _params(engine), f"fwd_bwd {micro_step} moved a weight")
      self.assertEqual(engine.train_step, 0, "fwd_bwd must not advance the train step")
      self.assertEqual(engine.micro_step_count, micro_step)
      self.assertTrue(engine.has_accumulated_grads)
      # Intact, not merely non-None: each packed row contributes its own live
      # segments, so an accumulator that was being reset or overwritten every
      # micro-step would still leave `has_accumulated_grads` True here.
      denominator = float(np.asarray(engine._accumulated_denominator))  # pylint: disable=protected-access
      self.assertEqual(denominator, float(micro_step * _NUM_SEQ))

    self.assertEqual(engine.update(), 1, "update must return the step count after applying")
    after = _params(engine)

    self.assertEqual(engine.train_step, 1)
    self.assertEqual(engine.micro_step_count, 0)
    self.assertFalse(engine.has_accumulated_grads)
    self.assertIsNone(engine._accumulated_denominator)  # pylint: disable=protected-access

    # The guard that makes every "unchanged" assertion above mean something. At
    # this config's default learning rate -- 0.0 at step 0, see `_LIVE_LR` -- the
    # loop passes unchanged on an engine that never computed a gradient at all.
    moved = sum(not np.array_equal(b, a) for b, a in zip(before, after))
    self.assertEqual(moved, len(before), f"update moved only {moved} of {len(before)} parameter leaves")

  def test_update_without_accumulated_grads_does_not_advance_the_step(self):
    """The early return at `maxtext_engine.py:1624`, before and after a real step.

    An orchestrator that reaches `update()` with nothing accumulated -- which
    packing makes likelier, since a token budget can leave a rank with no row to
    pack -- must get a no-op, not an optimizer step against an empty or stale
    gradient tree. The second half is the case that can actually regress: the
    reset to `None` happens at the *end* of `update()`, so a repeated `update()`
    with no `fwd_bwd` between it and the last one takes the same branch by way of
    state the first call left behind.
    """
    engine = _live_engine("maxtext_engine_cadence_noop")
    before = _params(engine)

    self.assertEqual(engine.update(), 0)
    self.assertEqual(engine.train_step, 0)
    self._assert_params_equal(before, _params(engine), "a no-op update moved a weight")

    engine.fwd_bwd(_packed_example(*_sequences()))
    self.assertEqual(engine.update(), 1)
    stepped = _params(engine)

    self.assertEqual(engine.update(), 1, "a second update with nothing accumulated must not step")
    self.assertEqual(engine.train_step, 1)
    self._assert_params_equal(stepped, _params(engine), "a repeated update applied twice")


class PackedCompiledPathTest(unittest.TestCase):
  """The same work on the *compiled* branch of `fwd_bwd` -- the branch production takes.

  Every class above runs the eager one. `fwd_bwd` only takes the compiled branch
  when `compile()` has been called (`maxtext_engine.py:1565`), and nothing else in
  this file calls it -- while the orchestrator always does, unconditionally, on
  every worker: `LifecycleDriver.bring_up` -> `TrainerWorker.compile(dummy_data)`,
  and `compile()` sets `_compile_requested` *before* its own `dummy_data is None`
  early return. So until this class, packing was proven only on a branch no real
  run reaches.

  Both ways into that branch are covered, and both are live: which one a demo
  takes is a `dummy_data` argument that is expected to change, so neither is
  pinned here as "the" configuration.

  * `compile(None)`. Nothing is compiled ahead of time and the flag is set
    anyway, so the first `fwd_bwd` compiles against the packed payload it is
    handed.
  * `compile(packed_payload)`, real ahead-of-time compilation. The kernels become
    `jax.stages.Compiled` executables, and the question is no longer whether they
    run but whether the packed run *keeps* them -- an executable built against the
    wrong batch structure is silently discarded and recompiled, which costs the
    whole point of compiling early and says so nowhere.

  What this class deliberately does not re-prove is that packing itself is right.
  Its first two tests compare the compiled branch against the eager one, so a
  fault the two share is invisible here by construction; that is T3's job above
  and `ClosedFormPackedLossTest`'s below.
  """

  # Compiled-vs-eager over the whole gradient tree. Measured: 0.0 at one
  # micro-batch and 4.8e-09 at two, the difference being XLA's freedom to fuse
  # the accumulate. 200x the worse of those.
  _COMPILED_TOL = 1e-6

  @classmethod
  def setUpClass(cls):
    """Runs the eager reference once; two of the three tests compare against it.

    Class-level because it is a second engine plus two forward-backward passes and
    an update, and it is identical for both -- deterministic init, same payload,
    same config apart from the `compile()` call under test.
    """
    super().setUpClass()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    cls.payload = _packed_example(*_sequences())

    engine = _live_engine("maxtext_engine_compiled_reference")
    # The reference has to be the *other* branch, or every comparison below is
    # between two compiled runs.
    assert not engine._compile_requested, "the eager reference was built on the compiled path"  # pylint: disable=protected-access
    _grads_after(engine, cls.payload)
    cls.eager_grads = _grads_after(engine, cls.payload)
    engine.update()
    cls.eager_params = _params(engine)

  def setUp(self):
    super().setUp()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def test_lazily_compiled_packed_micro_steps_match_the_eager_branch(self):
    """`compile(None)`: the flag alone reroutes a packed run through XLA.

    The demos pass `dummy_data=None` as of writing (`run_gsm8k_dist_grpo.py:371`,
    `run_deepswe_dist.py:324`), so this is the shape of a packed GRPO run today.
    It stays worth pinning once they pass a real payload: `compile()` keeps the
    deferral for any caller that cannot supply one, and a worker that is brought
    up before its first batch exists is exactly such a caller.
    """
    engine = _live_engine("maxtext_engine_compiled_lazy")
    engine.compile(None)

    # Nothing has been compiled, but the branch is already chosen.
    self.assertTrue(engine._compile_requested)  # pylint: disable=protected-access
    self.assertFalse(engine._compiled)  # pylint: disable=protected-access

    _grads_after(engine, self.payload)
    self.assertTrue(engine._compiled, "the first packed fwd_bwd must compile")  # pylint: disable=protected-access
    grads = _grads_after(engine, self.payload)

    # Two micro-batches, because the branch has two kernels and only the second
    # one reaches `_compiled_fwd_bwd_accum` -- the one that donates the live
    # accumulator, which is where packing's extra static (`num_segments`) has to
    # agree between the kernel and the batch or the donation is rejected.
    self.assertEqual(float(np.asarray(engine._accumulated_denominator)), 2 * _NUM_SEQ)  # pylint: disable=protected-access
    self.assertLess(_rel_l2(grads, self.eager_grads), self._COMPILED_TOL)

  def test_ahead_of_time_kernels_built_on_a_packed_batch_are_reused(self):
    """`compile(packed_payload)`: compiled early, and still used when the batch arrives.

    `_compile_for_batch` is the signal to count. It is the only thing that replaces
    the kernels, so counting its calls separates "the ahead-of-time executable ran"
    from "it was discarded and rebuilt on the first batch, identically and
    silently". Nothing else observes the difference: the numbers come out the same
    either way, and only compile time changes.
    """
    engine = _live_engine("maxtext_engine_compiled_aot")
    before = _params(engine)
    engine.compile(self.payload)

    # `jax.jit` wrappers before, XLA executables after.
    self.assertIsInstance(engine._compiled_fwd_bwd, jax.stages.Compiled)  # pylint: disable=protected-access
    self.assertIsInstance(engine._compiled_fwd_bwd_accum, jax.stages.Compiled)  # pylint: disable=protected-access
    self.assertIsInstance(engine._compiled_update, jax.stages.Compiled)  # pylint: disable=protected-access

    with (
        mock.patch.object(
            engine, "_compile_for_batch", wraps=engine._compile_for_batch  # pylint: disable=protected-access
        ) as recompile,
        mock.patch.object(
            engine, "_compiled_fwd_bwd", wraps=engine._compiled_fwd_bwd  # pylint: disable=protected-access
        ) as executable,
    ):
      _grads_after(engine, self.payload)
      grads = _grads_after(engine, self.payload)
      step = engine.update()
    recompile.assert_not_called()
    # Both halves, because either alone is satisfied by the wrong thing: a run
    # that quietly stayed eager also never recompiles, and a run that rebuilt the
    # kernel also ends up calling one.
    executable.assert_called_once()

    self.assertEqual(step, 1)
    self.assertLess(_rel_l2(grads, self.eager_grads), self._COMPILED_TOL)
    # The whole round trip, not just the gradients: the update kernel is compiled
    # too, and it is handed a donated accumulator pair that packing sized.
    for i, (compiled, eager) in enumerate(zip(_params(engine), self.eager_params)):
      np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-6, err_msg=f"parameter leaf {i}")
    # ... and the update has to have done something, or the loop above passes on
    # an engine that never applied a gradient. See `_LIVE_LR`.
    moved = sum(not np.array_equal(b, a) for b, a in zip(before, _params(engine)))
    self.assertEqual(moved, len(before), f"update moved only {moved} of {len(before)} parameter leaves")

  def test_a_changed_segment_count_recompiles_rather_than_reusing_the_kernel(self):
    """What `PackedBatchStaticsTest` proves about the signature, acted on by the engine.

    That test shows two segment counts produce different signatures. This one
    shows the live path *does something* with the difference, which is a separate
    claim: `_needs_recompile` compares the structural half and the static half
    separately, and a version that dropped the structural comparison would keep
    both assertions there green while reusing a kernel built for the wrong bucket
    count here.

    Not hypothetical under packing. `num_segments` is a per-batch property -- the
    FFD packer's slot count for that batch -- so it moves whenever the budget or
    the sequence-length mix does, which for a rollout is most steps.
    """
    engine = _live_engine("maxtext_engine_compiled_recompile")
    engine.compile(self.payload)

    with mock.patch.object(
        engine, "_compile_for_batch", wraps=engine._compile_for_batch  # pylint: disable=protected-access
    ) as recompile:
      engine.fwd_bwd(self.payload)
      self.assertEqual(recompile.call_count, 0, "the ahead-of-time kernel must survive its own batch")
      engine.fwd_bwd(self.payload)
      self.assertEqual(recompile.call_count, 0, "an identical second batch must not recompile")

      # Same arrays, same shapes, one more bucket: the only thing that differs is
      # the static value the segmented reductions are shaped by.
      engine.fwd_bwd(dataclasses.replace(self.payload, num_segments=self.payload.num_segments + 1))
      self.assertEqual(recompile.call_count, 1, "a new segment count must not reuse the kernel")


class ClosedFormPackedLossTest(unittest.TestCase):
  """The packed loss against a hand-computed number, not against itself.

  Every other test in this file compares packed against unpacked, and that whole
  family shares one blind spot: it stays green when both sides are wrong in the
  same way. Aggregating by row rather than by segment does exactly that, because
  it is the *unpacked* layout's correct behaviour -- one row is one sequence
  there -- so a packed run that copies it agrees with its own reference.

  Ported from `tunix/tests/rl/algo_core_test.py:49`, which pins the absolute
  value. The number is model-independent, which is what makes it assertable
  through a real MaxText Transformer rather than tunix's segment-aware toy: with
  `old_per_token_logps=None` the importance ratio is exactly 1, so the per-token
  loss is `-advantage` whatever the weights happen to be.

  Under `sequence-mean-token-mean` -- mean over tokens within a segment, then
  over segments -- segment A (advantage 1.5, 3 tokens) and segment B (advantage
  3.0, 1 token) give `(-1.5 + -3.0) / 2 = -2.25`.

  `-1.875` is the value to watch for. That is `(-1.5*3 + -3.0*1) / 4`, what
  per-row aggregation gives, and it is what the two-segment row produces when
  scored as a single sequence. The margin between the two is 17%, comfortably
  outside any tolerance, but a *uniform* segment length would collapse them: the
  two aggregations coincide when every segment has the same token count, which is
  why the fixture's segments are deliberately 3 tokens and 1.
  """

  # Both layouts are four tokens wide: packed is one row of 4, unpacked is two
  # rows of one prompt token plus three completion slots.
  _WIDTH = 4

  def setUp(self):
    super().setUp()
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

    cfg = _tiny_cfg(self._WIDTH, "maxtext_engine_closed_form_test")
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    base = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
    # pad_id=None for the same reason as in `PackedVersusUnpackedLogpsTest`: the
    # only segment boundaries in play should be the ones passed in.
    self.model = TunixMaxTextAdapter(base_model=base, pad_id=None)

  def _packed(self):
    """One row, two segments of 3 and 1 tokens, plus the padding bucket."""
    return rl_common.TrainExample(
        prompt_ids=jnp.zeros((1, 0), jnp.int32),
        prompt_mask=jnp.zeros((1, 0), jnp.int32),
        completion_ids=jnp.array([[3, 4, 5, 6]], jnp.int32),
        completion_mask=jnp.array([[1, 1, 1, 1]], jnp.float32),
        # Per-token, so each segment's scalar advantage is broadcast over its
        # own tokens -- one scalar per row could not tell the two apart.
        advantages=jnp.array([[1.5, 1.5, 1.5, 3.0]], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=jnp.array([[1, 1, 1, 2]], jnp.int32),
        segment_positions=jnp.array([[0, 1, 2, 0]], jnp.int32),
        # 2 real segments + the 0 bucket.
        num_segments=3,
    )

  def _unpacked(self):
    """The same two sequences, one per row. Row B is a third the length."""
    return rl_common.TrainExample(
        prompt_ids=jnp.array([[7], [7]], jnp.int32),
        prompt_mask=jnp.array([[1], [1]], jnp.int32),
        completion_ids=jnp.array([[3, 4, 5], [6, 0, 0]], jnp.int32),
        completion_mask=jnp.array([[1, 1, 1], [1, 0, 0]], jnp.float32),
        advantages=jnp.array([1.5, 3.0], jnp.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
        segment_ids=None,
        segment_positions=None,
        num_segments=None,
    )

  def _loss(self, example, loss_algo):
    algo_config = dataclasses.replace(_GrpoConfig(), loss_algo=loss_algo)
    out = algo_core.grpo_loss_fn(self.model, example, algo_config, pad_id=_PAD_ID, eos_id=_EOS_ID)
    return float(out.primary_loss.compute())

  def test_packed_loss_equals_the_hand_computed_value(self):
    # `gspo-token` as well as `grpo`: it pools the importance ratio per sequence
    # before applying it, so "sequence" means something to it that it does not
    # mean to `grpo`, and it reads `segment_ids` on a second code path.
    for loss_algo in ("grpo", "gspo-token"):
      with self.subTest(loss_algo=loss_algo):
        packed = self._loss(self._packed(), loss_algo)
        unpacked = self._loss(self._unpacked(), loss_algo)

        np.testing.assert_allclose(packed, unpacked, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(packed, -2.25, rtol=1e-4, atol=1e-4)

  def test_the_wrong_aggregation_is_a_value_this_test_can_see(self):
    """The margin behind the constant: `-2.25` and `-1.875` must be distinguishable.

    Asserting a constant is only meaningful if the failure mode it is aimed
    at produces a different constant. Recomputing the per-row value from the same
    fixture keeps that margin visible in the file rather than in a commit
    message, and fails if someone later evens out the segment lengths -- which
    would leave the test above passing while it had stopped testing anything.
    """
    advantages, per_segment = np.array([1.5, 1.5, 1.5, 3.0]), np.array([3, 1])

    by_segment = -float(np.mean([1.5, 3.0]))
    by_row = -float(np.sum(advantages) / np.sum(per_segment))

    self.assertAlmostEqual(by_segment, -2.25, places=6)
    self.assertAlmostEqual(by_row, -1.875, places=6)
    self.assertGreater(abs(by_segment - by_row), 0.1)


if __name__ == "__main__":
  unittest.main()
