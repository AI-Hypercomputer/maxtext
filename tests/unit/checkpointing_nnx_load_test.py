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

"""Unit tests for the NNX branches of load_state_if_possible."""

import os
import tempfile
import unittest
from unittest import mock

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
import optax
import orbax.checkpoint as ocp


class _Model(nnx.Module):
  """Tiny single-linear NNX model for restore tests."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)


class _ModelDropout(nnx.Module):
  """Linear + dropout, so the state carries rngs that split out into nnx_aux."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


def _abstract_nnx_state():
  """Build an nnx.State from a TrainStateNNX — same shape that pre_train passes in."""
  model = _Model(rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.scale_by_adam(), wrt=nnx.Param)
  return nnx.state(train_state_nnx.TrainStateNNX(model, optimizer))


def _dropout_state(seed):
  """A concrete TrainStateNNX state for `_ModelDropout` with its dropout stream advanced."""
  model = _ModelDropout(nnx.Rngs(seed))
  state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.scale_by_adam(), wrt=nnx.Param))
  grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 2)), deterministic=False) ** 2))(state.model)
  state.apply_gradients(grads)  # advances step + dropout rng count off the base 0
  return nnx.state(state)


def _abstract_dropout_state():
  def make():
    model = _ModelDropout(nnx.Rngs(9))
    return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.scale_by_adam(), wrt=nnx.Param)))

  return nnx.eval_shape(make)


class TestLoadStateIfPossibleNNX(unittest.TestCase):
  """Cover the NNX branches in load_state_if_possible."""

  def test_emergency_restore_recovers_nnx_aux(self):
    """Emergency restore reshapes back to NNX and recovers rngs/dropout from items/nnx_aux."""
    concrete = _dropout_state(0)
    orig_count = int(concrete.to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self.assertGreater(orig_count, 0)
    # The single state tree an emergency manager writes (weights + opt_state + step + nnx_aux).
    saved_linen = train_state_nnx.to_checkpoint_dict(concrete)
    self.assertIn("nnx_aux", saved_linen)

    checkpoint_manager = mock.Mock()
    checkpoint_manager.restore.return_value = mock.Mock(state=saved_linen)
    restored = checkpointing._restore_emergency_linen_checkpoint_into_nnx(  # pylint: disable=protected-access
        checkpoint_manager,
        14,
        _abstract_dropout_state(),
        lambda leaf: ocp.type_handlers.ArrayRestoreArgs(global_shape=leaf.shape, dtype=leaf.dtype),
    )
    restored = restored.to_pure_dict()

    # The restore target the manager was handed carries nnx_aux, so emergency checkpoints persist it.
    restore_target = checkpoint_manager.restore.call_args.kwargs["args"].state.item
    self.assertIn("nnx_aux", restore_target)
    # Reshaped back to NNX, with the dropout stream recovered (not reset to 0).
    self.assertIn("model", restored)
    self.assertIn("optimizer", restored)
    self.assertNotIn("params", restored)
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), orig_count)

  def test_load_parameters_from_path_splits_nnx_state_for_param_view(self):
    """When abstract_unboxed_pre_state is an nnx.State, the function must call
    nnx.split(model, nnx.Param, ...) to get the params and forward them to load_params_from_path."""
    abstract = _abstract_nnx_state()
    sentinel_restored = {"linear": {"kernel": jnp.ones((2, 1)), "bias": jnp.zeros((1,))}}

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel_restored) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=abstract,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel_restored)
    m.assert_called_once()
    forwarded_params = m.call_args[0][1]  # second positional arg = abstract_unboxed_params
    # The forwarded params come from nnx.split(..., nnx.Param, ...) — same key shape as the model.
    leaves = jax.tree.leaves(forwarded_params)
    self.assertEqual(len(leaves), 2)  # linear.kernel + linear.bias

  def test_load_parameters_from_path_uses_state_params_for_linen(self):
    """For Linen TrainState, the function must use state.params (not nnx.split)."""
    fake_state = mock.Mock(spec=["params"])
    fake_state.params = {"layer": {"kernel": jnp.ones((2, 2))}}
    sentinel = object()

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=fake_state,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel)
    forwarded_params = m.call_args[0][1]
    self.assertIs(forwarded_params, fake_state.params)

  def test_no_paths_returns_none_none(self):
    """Sanity: with no checkpoint manager and no load paths, the function returns (None, None)."""
    full, params = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=_abstract_nnx_state(),
    )
    self.assertIsNone(full)
    self.assertIsNone(params)


class TestLoadParamsIntoNNX(unittest.TestCase):
  """Weight-only load (load_parameters_path) of a Linen-layout checkpoint into NNX."""

  def test_linen_layout_params_restore_into_nnx_state(self):
    """load_params_from_path reshapes an on-disk Linen-layout checkpoint into the NNX params state."""
    model = _Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)
    weights = {
        "linear": {
            "kernel": jnp.arange(2, dtype=jnp.float32).reshape(2, 1),
            "bias": jnp.array([5.0]),
        }
    }

    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      path = os.path.join(d, "ckpt")
      # On-disk Linen layout: params/params/<weights> plus an unrelated `step`.
      ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
          epath.Path(path),
          {"params": {"params": weights}, "step": jnp.array(3)},
          force=True,
      )
      restored = checkpointing.load_params_from_path(path, params_abstract, 8)

    self.assertIsInstance(restored, nnx.State)
    pure = restored.to_pure_dict()
    self.assertTrue(jnp.array_equal(pure["linear"]["kernel"], weights["linear"]["kernel"]))
    self.assertTrue(jnp.array_equal(pure["linear"]["bias"], weights["linear"]["bias"]))


class TestToCheckpointDict(unittest.TestCase):
  """train_state_nnx.to_checkpoint_dict splits the NNX state by Variable type."""

  def test_splits_params_optimizer_and_nnx_aux(self):
    model = _ModelDropout(nnx.Rngs(0))
    state = nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.scale_by_adam(), wrt=nnx.Param)))
    ckpt = train_state_nnx.to_checkpoint_dict(state)
    # Learnable weights land in the Linen params collection; optimizer + step alongside.
    self.assertEqual(set(ckpt["params"]["params"].keys()), {"linear"})
    self.assertIn("opt_state", ckpt)
    self.assertIn("step", ckpt)
    # rngs/dropout land in nnx_aux, not in params.
    self.assertIn("dropout", ckpt["nnx_aux"]["model"])
    self.assertNotIn("dropout", ckpt["params"]["params"])

  def test_batch_stats_split_out_from_weights(self):
    """BatchNorm scale/bias are weights; its running mean/var are batch stats -> nnx_aux."""

    class _BN(nnx.Module):

      def __init__(self, rngs):
        self.bn = nnx.BatchNorm(2, rngs=rngs)

    state = nnx.state(train_state_nnx.TrainStateNNX(_BN(nnx.Rngs(0)), None))
    ckpt = train_state_nnx.to_checkpoint_dict(state)
    self.assertEqual(set(ckpt["params"]["params"]["bn"].keys()), {"scale", "bias"})
    self.assertEqual(set(ckpt["nnx_aux"]["model"]["bn"].keys()), {"mean", "var"})

  def test_flat_opt_state_round_trips_without_chain_wrapper(self):
    """A flat (un-chained) opt_state survives to_linen -> from_linen unchanged.

    Regression guard: from_linen used to re-wrap a flat opt_state under a `0` chain index,
    which then failed to overlay onto the model's flat opt_state on resume.
    """
    model = _Model(nnx.Rngs(0))
    pure = nnx.state(
        train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.scale_by_adam(), wrt=nnx.Param))
    ).to_pure_dict()
    linen = train_state_nnx.to_linen_checkpoint_dict(pure)
    back = train_state_nnx.from_linen_checkpoint_dict(linen)
    self.assertEqual(set(back["optimizer"]["opt_state"].keys()), set(pure["optimizer"]["opt_state"].keys()))
    self.assertIn("count", back["optimizer"]["opt_state"])  # flat, not nested under a `0` index

  def test_chained_opt_state_restores_onto_model(self):
    """A chained optimizer's (optax.adamw) opt_state must round-trip back onto the state.

    Regression guard: to_linen used to flatten a single-entry chain `{0: ...}`, so the overlay
    didn't line up with the model's int-keyed opt_state and adamw restore crashed. Other tests use
    scale_by_adam (flat), which never hits a chain.
    """

    def make():
      model = _Model(nnx.Rngs(0))
      return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)))

    saved = make()
    self.assertIn(0, saved.to_pure_dict()["optimizer"]["opt_state"])  # chained -> int-keyed, not flat
    overlay = train_state_nnx.from_linen_checkpoint_dict(train_state_nnx.to_checkpoint_dict(saved))
    nnx.replace_by_pure_dict(make(), overlay)  # raised before the fix

  def test_cache_and_intermediates_excluded(self):
    """Non-weight model variables (caches/intermediates) are not checkpointed."""

    class _Cached(nnx.Module):

      def __init__(self, rngs):
        self.linear = nnx.Linear(2, 1, rngs=rngs)
        self.cache = nnx.Cache(jnp.zeros((2,)))

    state = nnx.state(train_state_nnx.TrainStateNNX(_Cached(nnx.Rngs(0)), None))
    ckpt = train_state_nnx.to_checkpoint_dict(state)
    self.assertEqual(set(ckpt["params"]["params"].keys()), {"linear"})  # cache dropped
    self.assertNotIn("nnx_aux", ckpt)

  def test_no_nnx_aux_when_state_has_none(self):
    state = _abstract_nnx_state()  # plain linear, no rngs/batch stats
    self.assertNotIn("nnx_aux", train_state_nnx.to_checkpoint_dict(state))


class TestLinenItemsToNnx(unittest.TestCase):
  """checkpointing._linen_items_to_nnx reshapes restored items into the NNX-layout overlay."""

  def _to_nnx(self, restored):
    """Reshape `restored` against the `_ModelDropout` abstract, as the restore paths do."""
    state = checkpointing._linen_items_to_nnx(restored, _abstract_dropout_state())  # pylint: disable=protected-access
    return state.to_pure_dict()

  def test_materialized_aux_is_kept(self):
    restored = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(3, jnp.int32),
        "nnx_aux": {"model": {"dropout": {"rngs": {"count": jnp.asarray(42, jnp.uint32)}}}},
    }
    out = self._to_nnx(restored)
    self.assertEqual(int(out["model"]["dropout"]["rngs"]["count"]), 42)  # from the checkpoint
    self.assertTrue(jnp.array_equal(out["model"]["linear"]["kernel"], jnp.ones((2, 1))))

  def test_weights_and_aux_are_unioned_not_clobbered(self):
    """Weights and nnx_aux both nest under `model/` -- merging must keep both, at the leaf level."""
    restored = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(3, jnp.int32),
        "nnx_aux": {"model": {"dropout": {"rngs": {"count": jnp.asarray(7, jnp.uint32)}}}},
    }
    out = self._to_nnx(restored)
    self.assertIn("linear", out["model"])  # weights survived the aux merge
    self.assertIn("dropout", out["model"])  # aux survived the weights merge
    self.assertTrue(jnp.array_equal(out["model"]["linear"]["kernel"], jnp.ones((2, 1))))
    self.assertEqual(int(out["model"]["dropout"]["rngs"]["count"]), 7)

  def test_unmaterialized_leaf_kept_as_placeholder(self):
    """A leaf the checkpoint didn't carry stays a ShapeDtypeStruct; the caller fills it from init."""
    restored = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(3, jnp.int32),
        "nnx_aux": {"model": {"dropout": {"rngs": {"count": jax.ShapeDtypeStruct((), jnp.uint32)}}}},
    }
    out = self._to_nnx(restored)
    self.assertIsInstance(out["model"]["dropout"]["rngs"]["count"], jax.ShapeDtypeStruct)  # placeholder, not defaulted
    self.assertIsInstance(out["model"]["linear"]["bias"], jax.ShapeDtypeStruct)  # absent weight -> placeholder
    self.assertTrue(jnp.array_equal(out["model"]["linear"]["kernel"], jnp.ones((2, 1))))

  def test_no_aux_key_leaves_rng_as_placeholder(self):
    """No nnx_aux on disk -> the rng leaves stay placeholders (init supplies them later)."""
    restored = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(3, jnp.int32),
    }
    out = self._to_nnx(restored)
    self.assertIsInstance(out["model"]["dropout"]["rngs"]["count"], jax.ShapeDtypeStruct)


class TestMissingWeightPolicy(unittest.TestCase):
  """The Param-type weight check errors/warns on weights the checkpoint didn't carry."""

  def _want(self):
    """The weights the model expects, as a pure dict (what `_expected_and_restored_params` returns)."""
    return {
        "linear": {"kernel": jax.ShapeDtypeStruct((2, 1), jnp.float32), "bias": jax.ShapeDtypeStruct((1,), jnp.float32)}
    }

  def _have(self, with_bias=True):
    """The weights the checkpoint carried, as a pure dict; drop the bias to simulate a missing weight."""
    have = {"linear": {"kernel": jnp.ones((2, 1))}}
    if with_bias:
      have["linear"]["bias"] = jnp.ones((1,))
    return have

  def test_all_weights_present_passes(self):
    checkpointing._enforce_missing_weight_policy(self._want(), self._have(), "error")  # pylint: disable=protected-access

  def test_missing_weight_raises_naming_path(self):
    with self.assertRaises(ValueError) as ctx:
      checkpointing._enforce_missing_weight_policy(self._want(), self._have(with_bias=False), "error")  # pylint: disable=protected-access
    self.assertIn("model/linear/bias", str(ctx.exception))
    self.assertIn("missing model weights", str(ctx.exception))

  def test_surviving_shape_dtype_struct_counts_as_missing(self):
    have = self._have()
    have["linear"]["bias"] = jax.ShapeDtypeStruct((1,), jnp.float32)
    with self.assertRaises(ValueError) as ctx:
      checkpointing._enforce_missing_weight_policy(self._want(), have, "error")  # pylint: disable=protected-access
    self.assertIn("model/linear/bias", str(ctx.exception))

  def test_warn_logs_and_returns(self):
    with mock.patch.object(checkpointing.max_logging, "log") as logmock:
      checkpointing._enforce_missing_weight_policy(self._want(), self._have(with_bias=False), "warn")  # pylint: disable=protected-access
    self.assertTrue(any("missing model weights" in str(c) for c in logmock.call_args_list))

  def test_missing_weight_paths_finds_absent_and_sds(self):
    want = {"a": {"k": jax.ShapeDtypeStruct((2,), jnp.float32), "b": jax.ShapeDtypeStruct((1,), jnp.float32)}}
    have = {"a": {"k": jnp.ones((2,))}}  # b absent
    missing = checkpointing._missing_weight_paths(want, have)  # pylint: disable=protected-access
    self.assertEqual([p for p, _, _ in missing], ["a/b"])

  def test_expected_and_restored_params_splits_by_param_type(self):
    """Only nnx.Param weights land in `want`; rngs/dropout (nnx.RngState) are excluded from the check."""
    model = _ModelDropout(nnx.Rngs(0))
    abstract = nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)))
    restored = {"params": {"params": {"linear": {"kernel": jnp.ones((2, 1)), "bias": jnp.ones((1,))}}}}
    want, have = checkpointing._expected_and_restored_params(abstract, restored)  # pylint: disable=protected-access
    self.assertEqual(set(want.keys()), {"linear"})  # only the Param weights, no dropout rng
    self.assertNotIn("dropout", want)
    self.assertEqual(have, restored["params"]["params"])
    checkpointing._enforce_missing_weight_policy(want, have, "error")  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
