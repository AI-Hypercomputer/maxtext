# Copyright 2025-2026 Google LLC
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

"""Restoring weights whose Flax collection is not the `params` collection.

A weight's Flax collection name is derived from its NNX Variable *class*, so promoting
a weight from `nnx.Param` to a custom `nnx.Variable` subclass moves it out of `params`
into a collection of its own. Two checkpoint shapes then exist for the same model:

  * "legacy" -- written before the promotion, so every weight sits in `params`.
  * "split"  -- written after, so the custom collection is materialised on disk
                alongside `params`. This is what the DeepSeek-V4 converter emits.

DeepSeek-V4's routed gate bias became a `MoEBiasVar`, and restore skipped the unknown
collection outright: all 40 bias tensors stayed at their zero initialiser with no error
raised, silently corrupting expert selection in every MoE layer. `test_split_*` below is
that bug in miniature -- it fails without the collection-aware restore and passes with it.

Drives the real save_params_to_path -> load_params_from_path stack on CPU, no mocks.
"""
# This suite unit-tests the private restore helpers directly, by design.
# pylint: disable=protected-access

import shutil
import tempfile
import unittest
from unittest import mock

from flax import nnx
from flax.nnx import variablelib
import jax.numpy as jnp
import numpy as np

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
import optax


class _MoEBiasVar(nnx.Variable):
  """A custom collection, mirroring DeepSeek-V4's routed gate bias."""


# Flax derives the on-disk collection name from the Variable class, so ask it rather
# than hardcoding a string the class could drift away from.
_MOE_BIAS_COLLECTION = variablelib.variable_name_from_type(_MoEBiasVar, allow_register=True)


class _LegacyModel(nnx.Module):
  """`gate_bias` as a plain Param: how older checkpoints were written."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.gate_bias = nnx.Param(jnp.arange(3, dtype=jnp.float32))


class _PromotedModel(nnx.Module):
  """`gate_bias` promoted to a custom Variable: how the model reads it today."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.gate_bias = _MoEBiasVar(jnp.zeros((3,), jnp.float32))


class _Tid2EidVar(nnx.Variable):
  """A SECOND custom collection, mirroring DeepSeek-V4's tid2eid routing table."""


_TID2EID_COLLECTION = variablelib.variable_name_from_type(_Tid2EidVar, allow_register=True)


class _LegacyTwoExtraModel(nnx.Module):
  """Both extra weights as plain Params: the checkpoint-writing side."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.gate_bias = nnx.Param(jnp.arange(3, dtype=jnp.float32))
    self.tid2eid = nnx.Param(jnp.arange(4, dtype=jnp.float32))


class _TwoCustomModel(nnx.Module):
  """Both extra weights promoted, so the request spans THREE collections."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.gate_bias = _MoEBiasVar(jnp.zeros((3,), jnp.float32))
    self.tid2eid = _Tid2EidVar(jnp.zeros((4,), jnp.float32))


class _TransientModel(nnx.Module):
  """Carries one of every variable kind, to pin down what `_abstract_params` keeps."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.gate_bias = _MoEBiasVar(jnp.zeros((3,), jnp.float32))
    self.cache = nnx.Cache(jnp.zeros((2,), jnp.float32))
    self.batch_stat = nnx.BatchStat(jnp.zeros((2,), jnp.float32))
    self.intermediate = nnx.Intermediate(jnp.zeros((2,), jnp.float32))
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)  # contributes RngState


_TX = optax.adam(1e-3)


def _abstract_for(model_cls):
  """The abstract params blueprint the production caller hands to the loader.

  Deliberately routed through the production `_abstract_params` rather than
  re-implementing its variable filter: duplicating the filter here would let the test
  keep passing after the production predicate changed, which is the exact drift that
  let the DeepSeek-V4 bias tensors go unnoticed.
  """
  state = nnx.eval_shape(lambda: nnx.state(model_cls(nnx.Rngs(0))))
  return checkpointing._abstract_params(state)


def _legacy_weights():
  return nnx.state(_LegacyModel(nnx.Rngs(0))).to_pure_dict()


class TestRestoreAcrossCollections(unittest.TestCase):
  """Real save -> restore over both on-disk collection shapes."""

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

  def _save(self, payload):
    """Writes a params-only checkpoint and returns the documented `.../items` path."""
    root = self._dir + "/ckpt"
    checkpointing.save_params_to_path(root, payload)
    return root + "/items"

  def _legacy_checkpoint(self):
    """Every weight in the single `params` collection."""
    return self._save({"params": _legacy_weights()})

  def _split_checkpoint(self):
    """`gate_bias` in its own on-disk collection, as the DeepSeek-V4 converter writes it."""
    weights = _legacy_weights()
    gate = weights.pop("gate_bias")
    return self._save({"params": weights, _MOE_BIAS_COLLECTION: {"gate_bias": gate}})

  def _checkpoint_missing_the_custom_weight(self):
    """Neither a `params` copy nor the custom collection: the weight simply isn't there."""
    weights = _legacy_weights()
    weights.pop("gate_bias")
    return self._save({"params": weights})

  def test_split_collection_restores_promoted_weight(self):
    """THE DeepSeek-V4 case: a weight stored in its own collection must reach the model.

    Without collection-aware restore the `MoEBiasVar` collection is skipped and
    `gate_bias` keeps its zero initialiser -- the silent corruption this guards against.
    """
    path = self._split_checkpoint()
    restored = checkpointing.load_params_from_path(path, _abstract_for(_PromotedModel), 8).to_pure_dict()

    # Array equality already subsumes "is not the zero initialiser", so assert it once.
    np.testing.assert_array_equal(np.asarray(restored["gate_bias"]), np.arange(3, dtype=np.float32))
    # The ordinary Param must survive the collection split untouched -- compare VALUES,
    # since a shape check would pass even if the restore returned the wrong tensor.
    np.testing.assert_array_equal(
        np.asarray(restored["linear"]["kernel"]), np.asarray(_legacy_weights()["linear"]["kernel"])
    )

  def test_legacy_collection_restores_promoted_weight(self):
    """A checkpoint predating the promotion still loads: the weight is found in `params`."""
    path = self._legacy_checkpoint()
    restored = checkpointing.load_params_from_path(path, _abstract_for(_PromotedModel), 8).to_pure_dict()

    np.testing.assert_array_equal(np.asarray(restored["gate_bias"]), np.arange(3, dtype=np.float32))

  def test_legacy_collection_restores_unpromoted_model(self):
    """Negative control: the ordinary all-Param path is untouched."""
    path = self._legacy_checkpoint()
    restored = checkpointing.load_params_from_path(path, _abstract_for(_LegacyModel), 8).to_pure_dict()

    np.testing.assert_array_equal(np.asarray(restored["gate_bias"]), np.arange(3, dtype=np.float32))
    self.assertEqual(tuple(restored["linear"]["kernel"].shape), (2, 3))

  def test_split_collection_into_unpromoted_model_raises(self):
    """A genuine mismatch must still be reported, not papered over by the rescue."""
    path = self._split_checkpoint()
    with self.assertRaises(ValueError) as ctx:
      checkpointing.load_params_from_path(path, _abstract_for(_LegacyModel), 8)
    self.assertIn("gate_bias", str(ctx.exception))

  def test_absent_custom_collection_is_reported_not_silently_skipped(self):
    """A custom-collection weight the checkpoint lacks must raise, not come back unmaterialized.

    This is the failure mode the whole change exists to prevent: dropping a collection
    without complaint leaves the model running on its initialiser. Orbax's partial_load
    hands back a ShapeDtypeStruct placeholder for an unrequested weight, which reaches
    the model and fails much later -- or worse, silently trains on an untrained value.
    """
    path = self._checkpoint_missing_the_custom_weight()
    with self.assertRaises(ValueError) as ctx:
      restored = checkpointing.load_params_from_path(path, _abstract_for(_PromotedModel), 8)
      # If it did not raise, say precisely what came back so the failure is diagnosable.
      self.fail(f"expected a missing-weight error, got gate_bias={restored.to_pure_dict().get('gate_bias')!r}")
    self.assertIn("gate_bias", str(ctx.exception))

  def test_unreadable_metadata_still_restores_custom_collections(self):
    """A metadata read failure must not silently narrow the request to `params` only.

    The collection set is derived from checkpoint metadata. If that read fails we know
    NOTHING about which collections exist -- which is not the same as knowing only
    `params` exists. Treating it as the latter drops every custom collection and turns a
    transient blip into a hard failure on a perfectly valid checkpoint.
    """
    path = self._split_checkpoint()

    def _unreadable(*_args, **_kwargs):
      raise RuntimeError("simulated metadata read failure")

    with mock.patch.object(checkpointing.ocp, "metadata", _unreadable):
      restored = checkpointing.load_params_from_path(path, _abstract_for(_PromotedModel), 8).to_pure_dict()

    np.testing.assert_array_equal(np.asarray(restored["gate_bias"]), np.arange(3, dtype=np.float32))

  def test_three_collections_all_restore(self):
    """More than one custom collection: every one must be requested and merged back."""
    weights = nnx.state(_LegacyTwoExtraModel(nnx.Rngs(0))).to_pure_dict()
    gate = weights.pop("gate_bias")
    tid = weights.pop("tid2eid")
    path = self._save(
        {"params": weights, _MOE_BIAS_COLLECTION: {"gate_bias": gate}, _TID2EID_COLLECTION: {"tid2eid": tid}}
    )

    restored = checkpointing.load_params_from_path(path, _abstract_for(_TwoCustomModel), 8).to_pure_dict()

    np.testing.assert_array_equal(np.asarray(restored["gate_bias"]), np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(restored["tid2eid"]), np.arange(4, dtype=np.float32))

  def test_stale_duplicate_in_params_does_not_win(self):
    """A weight present in BOTH `params` and its own collection must resolve to the latter.

    Checkpoints written across a promotion boundary can carry a stale `params` copy
    alongside the authoritative one. Merging collections by path would let iteration
    order decide the winner; the live value must always come from the real collection.
    """
    weights = _legacy_weights()  # keeps a STALE gate_bias inside `params`
    weights["gate_bias"] = jnp.full((3,), -99.0, jnp.float32)
    path = self._save({"params": weights, _MOE_BIAS_COLLECTION: {"gate_bias": jnp.arange(3, dtype=jnp.float32)}})

    restored = checkpointing.load_params_from_path(path, _abstract_for(_PromotedModel), 8).to_pure_dict()

    np.testing.assert_array_equal(np.asarray(restored["gate_bias"]), np.arange(3, dtype=np.float32))


class TestAliasLegacyCollections(unittest.TestCase):
  """Unit coverage of the rescue itself, where failures name the exact branch."""

  def test_moves_leaf_that_exists_in_stored_params(self):
    params_collection = {"params": {"w": 1}, "MoEBiasVar": {"bias": 2}}
    stored = {"params": {"w": object(), "bias": object()}}

    patched, aliased = checkpointing._alias_legacy_collections(params_collection, stored)

    self.assertEqual(aliased, {"MoEBiasVar": [("bias",)]})
    self.assertEqual(patched["params"], {"w": 1, "bias": 2})
    self.assertNotIn("MoEBiasVar", patched)

  def test_leaves_genuinely_absent_weight_alone(self):
    """A weight in neither place stays put, so the mismatch check can still report it."""
    params_collection = {"params": {"w": 1}, "MoEBiasVar": {"bias": 2}}
    stored = {"params": {"w": object()}}  # no `bias` anywhere

    patched, aliased = checkpointing._alias_legacy_collections(params_collection, stored)

    self.assertEqual(aliased, {})
    self.assertEqual(patched["MoEBiasVar"], {"bias": 2})

  def test_skips_collection_the_checkpoint_already_has(self):
    """When the checkpoint has the collection, read it from there -- no aliasing."""
    params_collection = {"params": {"w": 1}, "MoEBiasVar": {"bias": 2}}
    stored = {"params": {"w": object(), "bias": object()}, "MoEBiasVar": {"bias": object()}}

    patched, aliased = checkpointing._alias_legacy_collections(params_collection, stored)

    self.assertEqual(aliased, {})
    self.assertEqual(patched["MoEBiasVar"], {"bias": 2})

  def test_partial_collection_keeps_the_remainder(self):
    """Only the leaves found in `params` move; the rest stay in their own collection."""
    params_collection = {"params": {"w": 1}, "MoEBiasVar": {"found": 2, "absent": 3}}
    stored = {"params": {"w": object(), "found": object()}}

    patched, aliased = checkpointing._alias_legacy_collections(params_collection, stored)

    self.assertEqual(aliased, {"MoEBiasVar": [("found",)]})
    self.assertEqual(patched["params"], {"w": 1, "found": 2})
    self.assertEqual(patched["MoEBiasVar"], {"absent": 3})

  def test_noop_when_metadata_unreadable(self):
    """Metadata read failures fall through untouched rather than guessing."""
    params_collection = {"params": {"w": 1}, "MoEBiasVar": {"bias": 2}}

    for stored in (None, {}, {"params": "not-a-dict"}):
      patched, aliased = checkpointing._alias_legacy_collections(params_collection, stored)
      self.assertEqual(aliased, {})
      self.assertEqual(patched, params_collection)

  def test_unalias_restores_the_original_shape(self):
    """Round trip: the Linen path must put rescued weights back where they belong."""
    params_collection = {"params": {"w": 1}, "MoEBiasVar": {"bias": 2}}
    stored = {"params": {"w": object(), "bias": object()}}

    patched, aliased = checkpointing._alias_legacy_collections(params_collection, stored)
    self.assertEqual(checkpointing._unalias_legacy_collections(patched, aliased), params_collection)

  def test_unalias_without_aliases_is_identity(self):
    restored = {"params": {"w": 1}}
    self.assertEqual(checkpointing._unalias_legacy_collections(restored, {}), restored)


class TestDeepMergeDicts(unittest.TestCase):
  """Merging collection subtrees into one flat weight tree."""

  def test_merges_disjoint_trees(self):
    self.assertEqual(checkpointing._deep_merge_dicts({"a": 1}, {"b": 2}), {"a": 1, "b": 2})

  def test_merges_nested_without_clobbering_siblings(self):
    merged = checkpointing._deep_merge_dicts({"a": {"x": 1}}, {"a": {"y": 2}})
    self.assertEqual(merged, {"a": {"x": 1, "y": 2}})

  def test_source_wins_on_leaf_conflict(self):
    self.assertEqual(checkpointing._deep_merge_dicts({"a": 1}, {"a": 2}), {"a": 2})

  def test_does_not_mutate_its_inputs(self):
    target = {"a": {"x": 1}}
    checkpointing._deep_merge_dicts(target, {"a": {"y": 2}})
    self.assertEqual(target, {"a": {"x": 1}})


class TestLookupPath(unittest.TestCase):

  def test_finds_nested_leaf(self):
    self.assertEqual(checkpointing._lookup_path({"a": {"b": 7}}, ("a", "b")), 7)

  def test_returns_none_when_absent(self):
    self.assertIsNone(checkpointing._lookup_path({"a": {}}, ("a", "b")))

  def test_returns_none_through_a_non_dict(self):
    self.assertIsNone(checkpointing._lookup_path({"a": 5}, ("a", "b")))


class TestAbstractParams(unittest.TestCase):
  """The allowlist -> denylist widening, which is what lets custom collections be requested."""

  def _state(self):
    def make():
      model = _TransientModel(nnx.Rngs(9))
      return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param)))

    return nnx.eval_shape(make)

  def test_keeps_custom_variables_and_drops_transient_state(self):
    """A custom Variable is a persistent weight and must be requested from the checkpoint.

    The previous `nnx.split_state(..., nnx.Param, ...)` allowlist dropped it, which is
    why the DeepSeek-V4 bias tensors were never asked for in the first place.
    """
    params = checkpointing._abstract_params(self._state())
    paths = {"/".join(str(p) for p in path) for path, _ in nnx.to_flat_state(params)}

    self.assertIn("gate_bias", paths)  # the custom collection
    self.assertIn("linear/kernel", paths)  # an ordinary Param
    for transient in ("cache", "batch_stat", "intermediate"):
      self.assertNotIn(transient, paths, f"{transient} is transient and must not be restored")
    self.assertFalse([p for p in paths if "rngs" in p], f"rng state must not be restored, got {paths}")


class TestUnconsumedCheckpointWeights(unittest.TestCase):
  """The other half of the question: did the model consume everything the checkpoint had?"""

  def test_warns_naming_the_ignored_weight(self):
    want = {"kept": 1}
    stored = {"kept": 1, "dropped": 2}

    with mock.patch.object(checkpointing.max_logging, "log") as log:
      checkpointing._log_unconsumed_checkpoint_weights(want, stored)

    messages = " ".join(str(c) for c in log.call_args_list)
    self.assertIn("dropped", messages)
    self.assertIn("WARNING", messages)

  def test_silent_when_everything_was_consumed(self):
    with mock.patch.object(checkpointing.max_logging, "log") as log:
      checkpointing._log_unconsumed_checkpoint_weights({"a": 1}, {"a": 1})
    log.assert_not_called()

  def test_silent_on_non_dict_input(self):
    with mock.patch.object(checkpointing.max_logging, "log") as log:
      checkpointing._log_unconsumed_checkpoint_weights(None, {"a": 1})
      checkpointing._log_unconsumed_checkpoint_weights({"a": 1}, None)
    log.assert_not_called()


if __name__ == "__main__":
  unittest.main()
