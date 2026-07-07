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
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
  return nnx.state(train_state_nnx.TrainStateNNX(model, optimizer))


def _dropout_state(seed):
  """A concrete TrainStateNNX state for `_ModelDropout` with its dropout stream advanced."""
  model = _ModelDropout(nnx.Rngs(seed))
  state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param))
  grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 2)), deterministic=False) ** 2))(state.model)
  state.apply_gradients(grads)  # advances step + dropout rng count off the base 0
  return nnx.state(state)


def _abstract_dropout_state():
  def make():
    model = _ModelDropout(nnx.Rngs(9))
    return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)))

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
    state = nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)))
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

  def test_no_nnx_aux_when_state_has_none(self):
    state = _abstract_nnx_state()  # plain linear, no rngs/batch stats
    self.assertNotIn("nnx_aux", train_state_nnx.to_checkpoint_dict(state))


class TestDeepMerge(unittest.TestCase):
  """checkpointing._deep_merge overlays leaves onto a base dict."""

  def test_overlay_wins_and_bases_survive(self):
    base = {"model": {"linear": {"kernel": 1}, "rngs": {"count": 0}}}
    overlay = {"model": {"rngs": {"count": 99}}}
    merged = checkpointing._deep_merge(base, overlay)  # pylint: disable=protected-access
    self.assertEqual(merged["model"]["linear"]["kernel"], 1)  # untouched
    self.assertEqual(merged["model"]["rngs"]["count"], 99)  # overlaid

  def test_does_not_mutate_inputs(self):
    base = {"a": {"b": 1}}
    checkpointing._deep_merge(base, {"a": {"c": 2}})  # pylint: disable=protected-access
    self.assertEqual(base, {"a": {"b": 1}})

  def test_non_dict_leaves_prefer_overlay(self):
    """Leaf vs leaf: overlay wins, but a None overlay keeps the base."""
    self.assertEqual(checkpointing._deep_merge(1, 2), 2)  # pylint: disable=protected-access
    self.assertEqual(checkpointing._deep_merge(1, None), 1)  # pylint: disable=protected-access

  def test_adds_keys_absent_from_base(self):
    merged = checkpointing._deep_merge({"a": 1}, {"b": 2})  # pylint: disable=protected-access
    self.assertEqual(merged, {"a": 1, "b": 2})


class TestLinenItemsToNnx(unittest.TestCase):
  """checkpointing._linen_items_to_nnx reshapes restored items back to NNX, handling nnx_aux."""

  def _abstract_pure(self):
    return {
        "model": {
            "linear": {"kernel": jax.ShapeDtypeStruct((2, 1), jnp.float32)},
            "dropout": {"rngs": {"count": jax.ShapeDtypeStruct((), jnp.uint32)}},
        },
        "optimizer": {"step": jax.ShapeDtypeStruct((), jnp.uint32)},
    }

  def test_all_materialized_detects_surviving_sds(self):
    # pylint: disable=protected-access
    self.assertTrue(checkpointing._all_materialized({"a": jnp.ones((2,))}))
    self.assertFalse(checkpointing._all_materialized({"a": jax.ShapeDtypeStruct((2,), jnp.float32)}))

  def test_materialized_aux_is_merged(self):
    restored = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(3, jnp.int32),
        "nnx_aux": {"model": {"dropout": {"rngs": {"count": jnp.asarray(42, jnp.uint32)}}}},
    }
    out = checkpointing._linen_items_to_nnx(restored, self._abstract_pure())  # pylint: disable=protected-access
    self.assertEqual(int(out["model"]["dropout"]["rngs"]["count"]), 42)  # restored, not defaulted
    self.assertTrue(jnp.array_equal(out["model"]["linear"]["kernel"], jnp.ones((2, 1))))

  def test_unmaterialized_aux_falls_back_to_default(self):
    """An old checkpoint returns nnx_aux as ShapeDtypeStruct -> treated as absent -> default."""
    restored = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(3, jnp.int32),
        "nnx_aux": {"model": {"dropout": {"rngs": {"count": jax.ShapeDtypeStruct((), jnp.uint32)}}}},
    }
    out = checkpointing._linen_items_to_nnx(restored, self._abstract_pure())  # pylint: disable=protected-access
    self.assertEqual(int(out["model"]["dropout"]["rngs"]["count"]), 0)  # _default_for_sds

  def test_no_aux_key_falls_back_to_default(self):
    restored = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(3, jnp.int32),
    }
    out = checkpointing._linen_items_to_nnx(restored, self._abstract_pure())  # pylint: disable=protected-access
    self.assertEqual(int(out["model"]["dropout"]["rngs"]["count"]), 0)


class TestMissingWeightPolicy(unittest.TestCase):
  """The Param-type weight check errors/warns on weights the checkpoint didn't materialize."""

  def _abstract(self):
    """A typed nnx.State (linear weights + dropout rng) so nnx.split_state(_, nnx.Param) works."""
    model = _ModelDropout(nnx.Rngs(0))
    return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)))

  def _restored(self, with_bias=True):
    """A restored Linen-layout `items` dict; drop the bias to simulate a missing weight."""
    params = {"linear": {"kernel": jnp.ones((2, 1))}}
    if with_bias:
      params["linear"]["bias"] = jnp.ones((1,))
    return {"params": {"params": params}}

  def test_all_weights_present_passes(self):
    checkpointing._enforce_missing_weight_policy(self._abstract(), self._restored(), "error")  # pylint: disable=protected-access

  def test_missing_weight_raises_naming_path(self):
    with self.assertRaises(ValueError) as ctx:
      checkpointing._enforce_missing_weight_policy(  # pylint: disable=protected-access
          self._abstract(), self._restored(with_bias=False), "error"
      )
    self.assertIn("model/linear/bias", str(ctx.exception))
    self.assertIn("missing model weights", str(ctx.exception))

  def test_surviving_shape_dtype_struct_counts_as_missing(self):
    restored = self._restored()
    restored["params"]["params"]["linear"]["bias"] = jax.ShapeDtypeStruct((1,), jnp.float32)
    with self.assertRaises(ValueError) as ctx:
      checkpointing._enforce_missing_weight_policy(self._abstract(), restored, "error")  # pylint: disable=protected-access
    self.assertIn("model/linear/bias", str(ctx.exception))

  def test_warn_logs_and_returns(self):
    with mock.patch.object(checkpointing.max_logging, "log") as logmock:
      checkpointing._enforce_missing_weight_policy(  # pylint: disable=protected-access
          self._abstract(), self._restored(with_bias=False), "warn"
      )
    self.assertTrue(any("missing model weights" in str(c) for c in logmock.call_args_list))

  def test_rng_state_is_never_flagged(self):
    """rngs/dropout are nnx.RngState, not nnx.Param, so their absence from params never trips the check."""
    checkpointing._enforce_missing_weight_policy(self._abstract(), self._restored(), "error")  # pylint: disable=protected-access

  def test_missing_weight_paths_finds_absent_and_sds(self):
    want = {"a": {"k": jax.ShapeDtypeStruct((2,), jnp.float32), "b": jax.ShapeDtypeStruct((1,), jnp.float32)}}
    have = {"a": {"k": jnp.ones((2,))}}  # b absent
    missing = checkpointing._missing_weight_paths(want, have)  # pylint: disable=protected-access
    self.assertEqual([p for p, _, _ in missing], ["a/b"])


if __name__ == "__main__":
  unittest.main()
