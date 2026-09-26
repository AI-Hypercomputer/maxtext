# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MaxText training engine."""
# pylint: disable=protected-access

import dataclasses
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import metrics as metrics_module
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import optax
import orbax.checkpoint as ocp
import pydantic
import pytest
from tunix.experimental.common import datatypes
from tunix.experimental.train import abstract_trainer
from tunix.sft import utils as sft_utils

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]


class DummyNNXModel(nnx.Module):

  def __init__(self):
    self.weights = nnx.Param(jnp.array([1.0, 2.0]))


class DummyStatefulNNXModel(nnx.Module):
  """A model whose state is not all `nnx.Param`, so `rest` is non-empty.

  `DummyNNXModel` is parameter-only, which makes `nnx.split(model, nnx.Param, ...)` return an
  empty `rest` and every publish of non-parameter state a no-op. Real models carry RNG
  counters and batch statistics, so one model here has to as well.
  """

  def __init__(self):
    self.weights = nnx.Param(jnp.array([1.0, 2.0]))
    self.calls = nnx.BatchStat(jnp.array(0.0))


@struct.dataclass(frozen=True, kw_only=True)
class DummyPayload(abstract_engine.TrainerPayload):
  token_ids: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))
  token_mask: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))
  metadata: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)


def _nested_tuples(value):
  """A logical axis rule set with every list turned into a tuple, so two spellings of it compare equal."""
  return tuple(_nested_tuples(v) for v in value) if isinstance(value, (list, tuple)) else value


class MaxTextTrainingEngineTest(absltest.TestCase):

  def setUp(self):
    """Sets up test dependencies and mocks."""
    super().setUp()
    self.mock_config = self.setup_config()
    dummy_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(self.mock_config), self.mock_config.mesh_axes)
    dummy_model = DummyNNXModel()
    # `create_training_optimizer` returns `(schedule, tx)` where `tx` is a raw optax
    # GradientTransformation; the engine wraps it in an nnx.Optimizer itself. Returning
    # an already-wrapped nnx.Optimizer here would make the engine wrap it twice.
    patcher = mock.patch.object(
        maxtext_engine.train_utils,
        "create_training_optimizer",
        return_value=(lambda step: jnp.array(0.001), optax.sgd(0.01)),
    )
    self.addCleanup(patcher.stop)
    patcher.start()

    # These tests always construct the engine without a mesh, and `from_pretrained`
    # returns `(model, model.mesh)` in that case -- it enters `with mesh:` before
    # returning, so the mesh it hands back is never None. (It returns a bare model
    # only when the caller supplies a mesh, which no test here does.)
    from_pretrained_patcher = mock.patch.object(
        maxtext_engine.model_creation_utils,
        "from_pretrained",
        return_value=(dummy_model, dummy_mesh),
    )
    self.addCleanup(from_pretrained_patcher.stop)
    self.mock_from_pretrained = from_pretrained_patcher.start()

  def setup_config(self, enable_checkpointing: bool = False, **kwargs):
    """Sets up a MaxText config via pyconfig.initialize."""
    overrides = {
        "model_name": "llama3.1-8b",
        "run_name": "test_run",
        "base_output_directory": self.create_tempdir().full_path,
        "init_weights_seed": 42,
        "micro_batch_size_to_train_on": 2,
        "gradient_accumulation_steps": 1,
        "enable_dropout": False,
        "record_internal_nn_metrics": False,
        "enable_tensorboard": False,
        "tensorboard_dir": self.create_tempdir().full_path,
        "skip_jax_distributed_system": True,
        "enable_checkpointing": enable_checkpointing,
        "profiler_steps": 0,
    }
    if enable_checkpointing:
      overrides.update(
          {
              "checkpoint_dir": self.create_tempdir().full_path,
              "checkpoint_period": 1,
              "max_num_checkpoints_to_keep": 10,
              "async_checkpointing": False,
          }
      )
    overrides.update(kwargs)
    return pyconfig.initialize([None, get_test_config_path()], **overrides)

  def _mock_orbax_manager(self, engine, latest_step=None):
    """Installs a mock Orbax manager and returns it."""
    mock_orbax_mgr = mock.MagicMock()
    mock_orbax_mgr.latest_step.return_value = latest_step
    mock_orbax_mgr.save.return_value = True
    engine._checkpoint_manager._checkpoint_manager = mock_orbax_mgr
    return mock_orbax_mgr

  def _mock_saved_micro_step_count(self, mock_orbax_mgr, micro_step_count):
    """Makes the mocked Orbax manager report how far into its step a saved checkpoint got."""
    saved_metadata = mock.MagicMock()
    saved_metadata.custom_metadata = {"micro_step_count": micro_step_count}
    mock_orbax_mgr.metadata.return_value = saved_metadata

  def test_raises_type_error_for_non_pyconfig(self):
    invalid_config = abstract_engine.TrainingConfig()
    with self.assertRaises(TypeError):
      maxtext_engine.MaxTextTrainingEngine(invalid_config)  # pytype: disable=wrong-arg-types

  def test_raises_value_error_for_missing_model_name(self):
    with self.assertRaises(ValueError):
      self.setup_config(model_name="")

  def test_max_text_trainer_instantiation_with_pyconfig(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )
    self.assertIsInstance(t, abstract_engine.AbstractTrainingEngine)
    self.mock_from_pretrained.assert_called_once()

    for step in range(2):
      self.assertEqual(t.train_step, step)
      payload = DummyPayload(
          token_ids=jnp.ones((2, 2)),
          token_mask=jnp.ones((2, 2)),
      )
      t.compile(payload)
      self.assertTrue(t._compiled)
      t.fwd_bwd(payload)
      self.assertEqual(t._micro_step_count, 1)
      t.update()
      self.assertEqual(t._micro_step_count, 0)
      self.assertIsNone(t._accumulated_grads)
    self.assertEqual(t.train_step, 2)

  def test_compiled_steps_publish_weights_and_non_param_state(self):
    """Exercises the cached pure-state path end to end, which nothing else on CPU does.

    Three conditions have to hold at once for the cache to be involved, and no other test
    here meets all three: the model must have non-`Param` state (otherwise `_publish_model_rest`
    is vacuous), `compile()` must be called (the cache is seeded by `_compile_for_batch`, so
    the eager path never touches it), and `update()` must run (only that reaches
    `_publish_state`). Two steps rather than one, because the interesting failure is the
    second step reading a stale or wrongly-partitioned cache written by the first.
    """
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(self.mock_config), self.mock_config.mesh_axes)
    self.mock_from_pretrained.return_value = (DummyStatefulNNXModel(), mesh)
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    def loss_fn(model, *_args, **_kwargs):
      # Mutating a non-`Param` makes `new_rest` differ from the cached `rest`; scaling the
      # loss by it turns a stale publish into a wrong gradient, not just a wrong counter.
      model.calls.value = model.calls.value + 1.0
      return (
          abstract_engine.WeightedMetric(
              unreduced_sum=jnp.sum(model.weights.value) * model.calls.value,
              denominator=jnp.array(1.0),
          ),
          {},
      )

    t.with_loss_fn(loss_fn)
    payload = DummyPayload()
    before = np.asarray(t.model.weights.value)

    t.compile(payload)
    self.assertIsNotNone(t._params_pure, "compile() did not seed the pure-state cache")
    for _ in range(2):
      t.fwd_bwd(payload)
      t.update()

    self.assertIsNotNone(t._params_pure, "the pure-state cache fell back to re-splitting the graph")
    # `nnx.update` is the publish barrier: the live module must track the cache.
    self.assertEqual(float(t.model.calls.value), 2.0)
    self.assertGreater(float(np.abs(np.asarray(t.model.weights.value) - before).max()), 0.0)
    self.assertEqual(t.train_step, 2)

  def test_gradient_norm_is_over_the_accumulated_normalized_gradient(self):
    """Pins down *which* gradient is normed once accumulation no longer means-of-means.

    `test_gradient_norm_is_recorded_every_step` already covers the norm existing with
    `skip_step_on_spikes` off, on a single micro-batch. Two micro-batches here, so the value
    can only come out right if the norm is taken over the accumulated sum after its single
    division by the accumulated denominator: a norm over either micro-batch's own gradient
    reads sqrt(2), and one taken before the division reads 4x this. The micro-batches are
    uniform, so this does not separate sum/sum from mean-of-means -- §3 of the parity write-up
    is where that distinction is measured.
    """
    self.assertFalse(self.mock_config.skip_step_on_spikes)
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    # d(unreduced_sum)/dw is 1.0 per element, so two micro-batches accumulate [2.0, 2.0]
    # against a denominator of 8.0 -> [0.25, 0.25], whose l2 norm is sqrt(2 * 0.25**2).
    t.with_loss_fn(
        lambda model, *_args, **_kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights.value), denominator=jnp.array(4.0)),
            {},
        )
    )
    payload = DummyPayload()
    t.fwd_bwd(payload)
    t.fwd_bwd(payload)
    t.update()

    metrics = t.get_metrics(clear_cache=True)
    self.assertIn("gradient_norm", metrics.scalar_metrics)
    recorded = np.asarray(metrics.scalar_metrics["gradient_norm"]).reshape(-1)
    self.assertEqual(recorded.shape, (1,), "one norm per update, not one per micro-batch")
    np.testing.assert_allclose(recorded[0], np.sqrt(2 * 0.25**2), rtol=1e-5)

  def test_trainable_parameters_mask_freezes_updates_and_isolates_grad_norm(self):
    """trainable_parameters_mask freezes specified parameters and isolates grad_norm."""

    class MoEModel(nnx.Module):

      def __init__(self):
        self.router_gate = nnx.Param(jnp.array([10.0, 10.0]))
        self.weights = nnx.Param(jnp.array([1.0, 2.0]))

    moe_model = MoEModel()
    dummy_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(self.mock_config), self.mock_config.mesh_axes)
    self.mock_from_pretrained.return_value = (moe_model, dummy_mesh)

    config = self.setup_config(trainable_parameters_mask=["^(?!.*router_gate).*"])
    base_opt = optax.sgd(0.1)
    tx = optimizers.apply_trainable_parameters_mask(base_opt, config)

    with mock.patch.object(maxtext_engine.train_utils, "create_training_optimizer", return_value=(lambda s: 0.1, tx)):
      t = maxtext_engine.MaxTextTrainingEngine(config)

      t.with_loss_fn(
          lambda model, *_args, **_kwargs: (
              abstract_engine.WeightedMetric(
                  unreduced_sum=jnp.sum(model.weights.get_value()) + 5.0 * jnp.sum(model.router_gate.get_value()),
                  denominator=jnp.array(1.0),
              ),
              {},
          )
      )
      payload = DummyPayload()
      t.fwd_bwd(payload)
      t.update()

      np.testing.assert_allclose(np.asarray(t.model.router_gate.get_value()), [10.0, 10.0])
      self.assertFalse(np.allclose(np.asarray(t.model.weights.get_value()), [1.0, 2.0]))

      metrics = t.get_metrics(clear_cache=True)
      recorded_norm = float(np.asarray(metrics.scalar_metrics["gradient_norm"]).reshape(-1)[0])
      # Without masking, the norm would include router_gate: sqrt(1^2 + 1^2 + 5^2 + 5^2) = sqrt(52) = 7.211.
      # With masking, it reflects only the weights (around sqrt(2) = 1.414).
      self.assertIsNone(t._freeze_mask)
      self.assertLess(recorded_norm, 2.0)
      self.assertGreater(recorded_norm, 0.0)

  def test_trainable_parameters_mask_compiled(self):
    """trainable_parameters_mask works in compiled mode."""

    class MoEModel(nnx.Module):

      def __init__(self):
        self.router_gate = nnx.Param(jnp.array([10.0, 10.0]))
        self.weights = nnx.Param(jnp.array([1.0, 2.0]))

    moe_model = MoEModel()
    dummy_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(self.mock_config), self.mock_config.mesh_axes)
    self.mock_from_pretrained.return_value = (moe_model, dummy_mesh)

    config = self.setup_config(trainable_parameters_mask=["^(?!.*router_gate).*"])
    base_opt = optax.sgd(0.1)
    tx = optimizers.apply_trainable_parameters_mask(base_opt, config)

    with mock.patch.object(maxtext_engine.train_utils, "create_training_optimizer", return_value=(lambda s: 0.1, tx)):
      t = maxtext_engine.MaxTextTrainingEngine(config)

      t.with_loss_fn(
          lambda model, *_args, **_kwargs: (
              abstract_engine.WeightedMetric(
                  unreduced_sum=jnp.sum(model.weights.get_value()) + 5.0 * jnp.sum(model.router_gate.get_value()),
                  denominator=jnp.array(1.0),
              ),
              {},
          )
      )
      payload = DummyPayload()
      t.compile(payload)
      t.fwd_bwd(payload)
      t.update()

      np.testing.assert_allclose(np.asarray(t.model.router_gate.get_value()), [10.0, 10.0])
      self.assertFalse(np.allclose(np.asarray(t.model.weights.get_value()), [1.0, 2.0]))

      metrics = t.get_metrics(clear_cache=True)
      recorded_norm = float(np.asarray(metrics.scalar_metrics["gradient_norm"]).reshape(-1)[0])
      self.assertIsNotNone(t._freeze_mask)
      self.assertLess(recorded_norm, 2.0)
      self.assertGreater(recorded_norm, 0.0)

  def test_trainable_parameters_mask_logs_only_once(self):
    """apply_trainable_parameters_mask logs parameter summary only once."""
    config = self.setup_config(trainable_parameters_mask=["^(?!.*router_gate).*"])
    base_opt = optax.sgd(0.1)
    with mock.patch.object(optimizers.max_logging, "log") as mock_log:
      tx = optimizers.apply_trainable_parameters_mask(base_opt, config)
      params = {"router_gate": jnp.zeros(2), "weights": jnp.zeros(2)}
      state = tx.init(params)
      initial_log_count = mock_log.call_count
      self.assertGreater(initial_log_count, 0)

      # Subsequent update calls should not log again.
      tx.update(params, state, params)
      tx.update(params, state, params)
      self.assertEqual(mock_log.call_count, initial_log_count)

  @mock.patch("orbax.checkpoint.PyTreeCheckpointHandler")
  @mock.patch("orbax.checkpoint.CheckpointManager")
  def test_max_text_trainer_checkpoint_manager_init(self, mock_create_mgr, mock_handler):
    mock_config = self.setup_config(
        enable_checkpointing=True,
        checkpoint_storage_use_ocdbt=False,
        checkpoint_storage_use_zarr3=False,
    )

    _ = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_create_mgr.assert_called_once_with(
        directory=mock_config.checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=mock_config.checkpoint_period,
            max_to_keep=mock_config.max_num_checkpoints_to_keep,
            enable_async_checkpointing=mock_config.async_checkpointing,
        ),
        item_handlers={
            "model_params": mock_handler.return_value,
            "optimizer_state": mock_handler.return_value,
            "accumulated_metrics": mock_handler.return_value,
            "accumulated_grads": mock_handler.return_value,
        },
    )
    self.assertEqual(mock_handler.call_count, 4)
    mock_handler.assert_has_calls(
        [
            mock.call(
                use_ocdbt=False,
                use_zarr3=False,
                save_device_host_concurrent_gb=mock_config.checkpoint_storage_device_host_concurrent_gb,
            )
        ]
        * 4,
    )

  @mock.patch("orbax.checkpoint.PyTreeCheckpointHandler")
  @mock.patch("orbax.checkpoint.CheckpointManager")
  def test_max_text_trainer_checkpoint_manager_init_custom_device_host_concurrent_gb(self, mock_create_mgr, mock_handler):
    mock_config = self.setup_config(
        enable_checkpointing=True,
        checkpoint_storage_use_ocdbt=True,
        checkpoint_storage_use_zarr3=True,
        checkpoint_storage_device_host_concurrent_gb=16,
    )

    _ = maxtext_engine.MaxTextTrainingEngine(mock_config)
    self.assertEqual(mock_handler.call_count, 4)
    mock_handler.assert_has_calls(
        [
            mock.call(
                use_ocdbt=True,
                use_zarr3=True,
                save_device_host_concurrent_gb=16,
            )
        ]
        * 4,
    )

  @mock.patch.dict("os.environ", {"ENABLE_PATHWAYS_PERSISTENCE": "1"})
  @mock.patch("orbax.checkpoint.pathways.register_type_handlers")
  def test_maybe_register_pathways_persistence(self, mock_register_type_handlers):
    import orbax.checkpoint.pathways as ocp_pathways  # pylint: disable=import-outside-toplevel
    from maxtext.training_engine import checkpointing as checkpointing_module  # pylint: disable=import-outside-toplevel

    checkpointing_module._PATHWAYS_PERSISTENCE_REGISTERED = False
    checkpointing_module._maybe_register_pathways_persistence()

    mock_register_type_handlers.assert_called_once()
    self.assertEqual(
        mock_register_type_handlers.call_args.kwargs["checkpointing_impl"],
        ocp_pathways.CheckpointingImpl.PERSISTENCE,
    )

  def test_save_checkpoint_called_after_update(self):
    mock_config = self.setup_config(enable_checkpointing=True)

    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = self._mock_orbax_manager(t)

    dummy_metadata = mock.MagicMock()
    t.save_checkpoint(metadata=dummy_metadata)

    # Verify orbax save was called
    mock_orbax_mgr.save.assert_called_once()
    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    self.assertEqual(call_kwargs["custom_metadata"]["micro_step_count"], 0)
    self.assertEqual(call_kwargs["custom_metadata"]["additional_metadata"], dummy_metadata)
    args_dict = (
        dict(call_kwargs["args"].items())
        if hasattr(call_kwargs["args"], "items") and callable(call_kwargs["args"].items)
        else call_kwargs["args"].__dict__
    )
    self.assertIn("model_params", args_dict)
    self.assertNotIn("accumulated_metrics", args_dict)
    self.assertNotIn("accumulated_grads", args_dict)

  def test_save_checkpoint_omits_items_with_no_leaves(self):
    mock_config = self.setup_config(enable_checkpointing=True)
    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = self._mock_orbax_manager(t)

    # When no metrics have been recorded, accumulated_metrics is empty list and not saved
    t.save_checkpoint(metadata=None)
    mock_orbax_mgr.save.assert_called_once()
    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    args_dict = (
        dict(call_kwargs["args"].items())
        if hasattr(call_kwargs["args"], "items") and callable(call_kwargs["args"].items)
        else call_kwargs["args"].__dict__
    )
    self.assertNotIn("accumulated_metrics", args_dict)

    # When accumulated_metrics is None in CheckpointState, it is not saved
    mock_orbax_mgr.reset_mock()
    ckpt_state_none = maxtext_engine.checkpointing.CheckpointState(
        model=t.model,
        accumulated_metrics=None,
    )
    t._checkpoint_manager.save_checkpoint(
        step=1,
        checkpoint_state=ckpt_state_none,
        force=True,
    )
    mock_orbax_mgr.save.assert_called_once()
    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    args_dict = (
        dict(call_kwargs["args"].items())
        if hasattr(call_kwargs["args"], "items") and callable(call_kwargs["args"].items)
        else call_kwargs["args"].__dict__
    )
    self.assertNotIn("accumulated_metrics", args_dict)

    # When accumulated_metrics is empty list in CheckpointState, it is not saved
    mock_orbax_mgr.reset_mock()
    ckpt_state_empty = maxtext_engine.checkpointing.CheckpointState(
        model=t.model,
        accumulated_metrics=[],
    )
    t._checkpoint_manager.save_checkpoint(
        step=2,
        checkpoint_state=ckpt_state_empty,
        force=True,
    )
    mock_orbax_mgr.save.assert_called_once()
    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    args_dict = (
        dict(call_kwargs["args"].items())
        if hasattr(call_kwargs["args"], "items") and callable(call_kwargs["args"].items)
        else call_kwargs["args"].__dict__
    )
    self.assertNotIn("accumulated_metrics", args_dict)

  def test_save_checkpoint_skips_if_already_saved(self):
    mock_config = self.setup_config(enable_checkpointing=True)

    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=10)

    t.save_checkpoint(metadata={"step": 10})
    mock_orbax_mgr.save.assert_not_called()
    mock_orbax_mgr.delete.assert_not_called()

  def test_save_checkpoint_overwrites_intra_step_checkpoint_at_same_step(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.setup_config(enable_checkpointing=True))
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=10)
    self._mock_saved_micro_step_count(mock_orbax_mgr, 2)

    # The step that intra-step checkpoint belongs to has since run to completion.
    t.train_step = 10
    t._micro_step_count = 0

    t.save_checkpoint(metadata=None)

    mock_orbax_mgr.wait_until_finished.assert_called()
    mock_orbax_mgr.delete.assert_called_once_with(10)
    mock_orbax_mgr.save.assert_called_once()
    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    self.assertEqual(call_kwargs["step"], 10)
    # Orbax's save-interval policy would otherwise decline a step it has already saved.
    self.assertTrue(call_kwargs["force"])
    self.assertEqual(call_kwargs["custom_metadata"]["micro_step_count"], 0)

  def test_save_checkpoint_overwrites_less_complete_intra_step_checkpoint(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.setup_config(enable_checkpointing=True))
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=10)
    self._mock_saved_micro_step_count(mock_orbax_mgr, 1)

    t.train_step = 9
    t._micro_step_count = 3
    t._accumulated_grads = {"params": {"w": jnp.array([0.5, 0.5])}}

    t.save_checkpoint(metadata=None)

    mock_orbax_mgr.delete.assert_called_once_with(10)
    self.assertEqual(mock_orbax_mgr.save.call_args.kwargs["custom_metadata"]["micro_step_count"], 3)

  def test_save_checkpoint_never_overwrites_a_complete_step(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.setup_config(enable_checkpointing=True))
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=10)
    self._mock_saved_micro_step_count(mock_orbax_mgr, 0)

    t.train_step = 10
    t._micro_step_count = 0

    t.save_checkpoint(metadata=None)

    mock_orbax_mgr.save.assert_not_called()
    mock_orbax_mgr.delete.assert_not_called()

  def test_update_supersedes_intra_step_checkpoint_it_resumed_from(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.setup_config(enable_checkpointing=True))
    t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=5)
    self._mock_saved_micro_step_count(mock_orbax_mgr, 2)

    # State as `restore_checkpoint` leaves it after loading an intra-step checkpoint at 5.
    t.train_step = 4
    t._resumed_mid_step = True

    payload = DummyPayload(token_ids=jnp.ones((2, 2)), token_mask=jnp.ones((2, 2)))
    t.compile(payload)
    t.fwd_bwd(payload)
    self.assertEqual(t.update(), 5)

    # The completed step must land on disk now, not a checkpoint period later.
    mock_orbax_mgr.delete.assert_called_once_with(5)
    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    self.assertEqual(call_kwargs["step"], 5)
    self.assertEqual(call_kwargs["custom_metadata"]["micro_step_count"], 0)
    self.assertFalse(t._resumed_mid_step)

  def test_update_does_not_checkpoint_when_not_resumed_mid_step(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.setup_config(enable_checkpointing=True))
    t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )
    mock_orbax_mgr = self._mock_orbax_manager(t)

    payload = DummyPayload(token_ids=jnp.ones((2, 2)), token_mask=jnp.ones((2, 2)))
    t.compile(payload)
    t.fwd_bwd(payload)
    t.update()

    mock_orbax_mgr.save.assert_not_called()

  def test_save_checkpoint_drains_inflight_throttler(self):
    mock_config = self.setup_config(enable_checkpointing=True)
    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = self._mock_orbax_manager(t)

    # Add a dummy item to the throttler queue.
    dummy_computation = jnp.array(1.0)
    t._throttler.add_computation(computation=dummy_computation, metrics=None)
    self.assertEqual(t._throttler._inflight_queue.qsize(), 1)

    t.save_checkpoint(metadata={"step": 10})

    # Checkpoint should be saved and throttler queue should be drained.
    mock_orbax_mgr.save.assert_called_once()
    self.assertTrue(t._throttler._inflight_queue.empty())

  def test_save_checkpoint_called_after_fwd_bwd_before_update(self):
    mock_config = self.setup_config(enable_checkpointing=True)
    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = self._mock_orbax_manager(t)

    t._micro_step_count = 1
    t.train_step = 10
    t._accumulated_grads = {"params": {"w": jnp.array([0.5, 0.5])}}
    t._accumulated_denominator = jnp.float32(6.0)

    dummy_metadata = mock.MagicMock()
    t.save_checkpoint(metadata=dummy_metadata)

    # Verify orbax save was called
    mock_orbax_mgr.save.assert_called_once()
    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    self.assertEqual(call_kwargs["custom_metadata"]["micro_step_count"], 1)
    # The gradients are saved unreduced, so their divisor has to ride along with them.
    self.assertEqual(call_kwargs["custom_metadata"]["accumulated_denominator"], 6.0)
    self.assertEqual(call_kwargs["custom_metadata"]["additional_metadata"], dummy_metadata)
    args_dict = (
        dict(call_kwargs["args"].items())
        if hasattr(call_kwargs["args"], "items") and callable(call_kwargs["args"].items)
        else call_kwargs["args"].__dict__
    )
    self.assertIn("model_params", args_dict)
    self.assertNotIn("accumulated_metrics", args_dict)
    self.assertIn("accumulated_grads", args_dict)

  def test_close_writes_final_checkpoint(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.setup_config(enable_checkpointing=True))
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=3)
    t.train_step = 4
    t._micro_step_count = 0

    t.close()

    call_kwargs = mock_orbax_mgr.save.call_args.kwargs
    self.assertEqual(call_kwargs["step"], 4)
    self.assertTrue(call_kwargs["force"])

  def test_close_saves_an_incomplete_step(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.setup_config(enable_checkpointing=True))
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=4)
    mock_logger = mock.MagicMock()
    t._throttler._metrics_logger = mock_logger
    t.train_step = 4
    t.record_metrics("loss", jnp.array(1.5))
    t._micro_step_count = 2
    t._accumulated_grads = {"params": {"w": jnp.array([0.5, 0.5])}}

    t.close()

    self.assertEqual(mock_orbax_mgr.save.call_args.kwargs["step"], 5)

  def test_restore_checkpoint_no_checkpoint_returns_defaults(self):
    mock_config = self.setup_config(enable_checkpointing=True)

    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    _ = self._mock_orbax_manager(t)

    restored_metadata = t.restore_checkpoint()
    self.assertIsNone(restored_metadata)

  def test_restore_checkpoint_restores_ckpt_metadata(self):
    mock_config = self.setup_config(enable_checkpointing=True)
    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=10)

    # Mock metadata with item_metadata and custom_metadata attributes
    dummy_metadata = mock.MagicMock()
    mock_metadata = mock.MagicMock()
    mock_metadata.item_metadata = {"model_params": {}, "optimizer_state": {}}
    mock_metadata.custom_metadata = {"additional_metadata": dummy_metadata}
    mock_orbax_mgr.metadata.return_value = mock_metadata

    # Return dummy model and optimizer state from orbax restore
    dummy_model = DummyNNXModel()
    dummy_opt = nnx.Optimizer(dummy_model, optax.sgd(0.01), wrt=nnx.Param)
    dummy_opt_state = nnx.state(dummy_opt, nnx.optimizer.OptState)
    mock_orbax_mgr.restore.return_value = {
        "model_params": nnx.state(dummy_model),
        "optimizer_state": dummy_opt_state,
    }
    t._checkpoint_manager._checkpoint_manager = mock_orbax_mgr

    restored_metadata = t.restore_checkpoint(step=10)
    self.assertEqual(t.train_step, 10)
    self.assertEqual(restored_metadata, dummy_metadata)
    mock_orbax_mgr.restore.assert_called_once()

  def test_restore_intra_step_checkpoint(self):
    mock_config = self.setup_config(enable_checkpointing=True)
    t = maxtext_engine.MaxTextTrainingEngine(mock_config)
    mock_orbax_mgr = self._mock_orbax_manager(t, latest_step=5)

    # Mock metadata with item_metadata and custom_metadata attributes
    dummy_metadata = mock.MagicMock()
    mock_metadata = mock.MagicMock()
    mock_metadata.item_metadata = {"model_params": {}, "optimizer_state": {}}
    mock_metadata.custom_metadata = {"micro_step_count": 2, "additional_metadata": dummy_metadata}
    mock_orbax_mgr.metadata.return_value = mock_metadata

    metrics_buf = abstract_engine.MetricsBuffer(id=5, mode="train")
    # `weighted_metrics` is a plain dict; pylint cannot resolve that through
    # flax.struct.dataclass and wrongly reports it as unsubscriptable.
    # pylint: disable-next=unsupported-assignment-operation
    metrics_buf.weighted_metrics["loss"] = abstract_engine.WeightedMetric(
        unreduced_sum=jnp.array([4.0, 6.0]),
        denominator=jnp.array([2.0, 2.0]),
    )
    dummy_grads = {"params": {"w": jnp.array([0.5, 0.5])}}
    dummy_model = DummyNNXModel()
    dummy_opt = nnx.Optimizer(dummy_model, optax.sgd(0.01), wrt=nnx.Param)
    dummy_opt_state = nnx.state(dummy_opt, nnx.optimizer.OptState)
    mock_orbax_mgr.restore.return_value = {
        "model_params": nnx.state(dummy_model),
        "optimizer_state": dummy_opt_state,
        "accumulated_metrics": [metrics_buf],
        "accumulated_grads": dummy_grads,
    }
    t._checkpoint_manager._checkpoint_manager = mock_orbax_mgr

    _ = t.restore_checkpoint(step=5)
    self.assertEqual(t._micro_step_count, 2)
    # Flags the partial checkpoint at step 5 for replacement once that step completes.
    self.assertTrue(t._resumed_mid_step)
    self.assertEqual(t._accumulated_grads, dummy_grads)
    self.assertEqual(len(t._cached_losses), 2)
    self.assertTrue(isinstance(t._cached_losses[0], abstract_engine.WeightedMetric))
    self.assertAlmostEqual(float(t._cached_losses[0].unreduced_sum), 4.0)
    self.assertAlmostEqual(float(t._cached_losses[1].unreduced_sum), 6.0)

  def test_record_and_get_metrics(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    # Record WeightedMetric
    t.record_metrics(
        name="loss",
        metric=abstract_engine.WeightedMetric(unreduced_sum=jnp.array(20.0), denominator=jnp.array(4.0)),
    )
    t.record_metrics(
        name="loss",
        metric=abstract_engine.WeightedMetric(unreduced_sum=jnp.array(30.0), denominator=jnp.array(6.0)),
    )

    # Record scalar
    t.record_metrics(
        name="lr",
        metric=0.002,
        aggregation_fn=lambda x: np.round(np.asarray(x), 4),
    )

    step0_metrics: Any = t.get_metrics(clear_cache=True)
    self.assertIsInstance(step0_metrics, abstract_engine.MetricsBuffer)
    self.assertIn("loss", step0_metrics.weighted_metrics)
    np.testing.assert_array_equal(
        step0_metrics.weighted_metrics["loss"].unreduced_sum,
        jnp.array([20.0, 30.0]),
    )
    np.testing.assert_array_equal(
        step0_metrics.weighted_metrics["loss"].denominator,
        jnp.array([4.0, 6.0]),
    )
    self.assertIn("lr", step0_metrics.scalar_metrics)
    np.testing.assert_array_equal(step0_metrics.scalar_metrics["lr"], jnp.array([0.002]))
    self.assertIn("lr", step0_metrics.aggregation_fns)
    self.assertEqual(step0_metrics.aggregation_fns["lr"](jnp.array([0.002])), 0.002)

  def test_update_with_inflight_throttling(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )

    payload = DummyPayload()
    t.compile(payload)

    # train_step=0: fwd_bwd + fwd_bwd + update
    t.fwd_bwd(payload)
    # Loss for micro_step_count=0 is queued. qsize=1.
    self.assertEqual(t._throttler._inflight_queue.qsize(), 1)
    t.fwd_bwd(payload)
    # Loss for micro_step_count=1 is also queued. qsize=2 (full).
    self.assertEqual(t._throttler._inflight_queue.qsize(), 2)
    t.update()
    self.assertEqual(t.train_step, 1)
    # wait_for_next() in update() sees qsize=2 (full), so it pops
    # index 0 (loss for micro_step_count=0), leaving qsize=1.
    # Then add_computation() queues the update's gradient norm and step 0 metrics.
    # Since we removed the trailing wait_for_next() from update(), qsize
    # remains 2.
    self.assertEqual(t._throttler._inflight_queue.qsize(), 2)
    for idx, (computation, metrics) in enumerate(t._throttler._inflight_queue.queue):
      if idx == 0:
        # Loss for micro_step_count=0.
        self.assertIsNone(metrics)
      if idx == 1:
        # Metrics for train_step=0, waited on through the update's gradient norm -- one
        # scalar out of the same executable, not the state, whose buffers get donated away.
        self.assertIsNotNone(metrics)
        self.assertLen(computation, 1)
        self.assertEqual(jnp.shape(computation[0]), ())

    # train_step=1: fwd_bwd + update
    # Calling fwd_bwd() while queue is full (qsize=2) triggers wait_for_next(),
    # popping index 0 (loss from micro_step_count=1) before adding the new loss.
    t.fwd_bwd(payload)
    self.assertEqual(t._throttler._inflight_queue.qsize(), 2)
    # When update() runs for train_step=1, wait_for_next() pops the metrics
    # for train_step=0. This blocks on expected_state_leaves and logs
    # train_step=0 metrics.
    t.update()
    self.assertEqual(t.train_step, 2)
    self.assertEqual(t._throttler._inflight_queue.qsize(), 2)

    # Closing trainer drains remaining inflight items.
    t.close()
    self.assertTrue(t._throttler._inflight_queue.empty())

  def test_fwd_bwd_with_loss_output_and_aux_metrics(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    payload = DummyPayload()

    def _loss_fn(model, *args, **kwargs):
      return abstract_engine.LossOutput(
          primary_loss=abstract_engine.WeightedMetric(
              unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0)
          ),
          aux_metrics={
              "metric_a": abstract_engine.WeightedMetric(unreduced_sum=jnp.array(12.0), denominator=jnp.array(3.0)),
              "metric_b": jnp.array(0.42),
          },
      )

    t.with_loss_fn(_loss_fn)
    t.compile(payload)
    t.fwd_bwd(payload)

    self.assertEqual(t._micro_step_count, 1)
    self.assertIsNotNone(t._accumulated_grads)

    # Gradients accumulate unreduced: d(unreduced_sum)/dw is 8.0 per element and the 1/4.0
    # from the denominator is applied once, in update().
    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)
    np.testing.assert_allclose(t._accumulated_denominator, 4.0, rtol=1e-5)

    metrics = t.get_metrics(clear_cache=True)
    self.assertIn("loss", metrics.weighted_metrics)
    self.assertIn("metric_a", metrics.weighted_metrics)
    self.assertIn("metric_b", metrics.scalar_metrics)
    self.assertAlmostEqual(
        float(metrics.weighted_metrics["loss"].compute().item()),
        6.0,
        places=4,
    )
    self.assertAlmostEqual(
        float(metrics.weighted_metrics["metric_a"].compute().item()),
        4.0,
        places=4,
    )

  def test_fwd_bwd_with_loss_and_aux_dict_tuple(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    payload = DummyPayload()

    def custom_loss(model, *args, **kwargs):
      unreduced_sum = jnp.sum(model.weights[...]) * 8.0
      denominator = jnp.array(4.0)
      return unreduced_sum / denominator, {
          "aux_stat": jnp.array(1.23),
          "xent_sum": unreduced_sum,
          "total_weights": denominator,
      }

    t.with_loss_fn(custom_loss, has_aux=True)
    t.fwd_bwd(payload)

    # Unreduced, with the 1/4.0 deferred to update().
    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)
    np.testing.assert_allclose(t._accumulated_denominator, 4.0, rtol=1e-5)
    metrics = t.get_metrics(clear_cache=True)
    self.assertIn("loss", metrics.weighted_metrics)
    self.assertIn("aux_stat", metrics.scalar_metrics)

  def test_gen_model_input_fn_selects_the_tunix_call_convention(self):
    """With an adapter set, the loss is called `loss_fn(model, **inputs)`.

    That is Tunix's convention and what `with_gen_model_input_fn` has always documented
    its return value to be, so a Tunix loss needs no adapter closure.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    seen = {}

    def _loss_fn(model, **kwargs):
      seen.update(kwargs)
      return abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0))

    t.with_loss_fn(_loss_fn)
    t.with_gen_model_input_fn(lambda payload: {"alpha": jnp.array(1.0), "beta": jnp.array(2.0)})
    t.fwd_bwd(DummyPayload())

    # Arrived by keyword, under the names the adapter chose.
    self.assertEqual(sorted(seen), ["alpha", "beta"])
    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)

  def test_without_gen_model_input_fn_the_maxtext_convention_is_kept(self):
    """With no adapter, the loss still gets MaxText's positional signature."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    seen = {}

    def _loss_fn(model, config, data, dropout_rng, params, is_train=True):
      seen.update(config=config, data=data, dropout_rng=dropout_rng, params=params, is_train=is_train)
      return abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0))

    t.with_loss_fn(_loss_fn)
    t.fwd_bwd(DummyPayload())

    self.assertIs(seen["config"], self.mock_config)
    self.assertIsNone(seen["dropout_rng"])
    self.assertIsNone(seen["params"])
    self.assertTrue(seen["is_train"])
    # The payload's fields are auto-extracted into the positional `data`.
    self.assertIn("token_ids", seen["data"])
    self.assertNotIn("metadata", seen["data"])

  def test_prepare_batch_clears_metadata_on_dataclass(self):
    """Payload metadata is cleared to {} on dataclasses so gen_model_input_fn receives empty metadata."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    payload = DummyPayload(metadata={"request_id": 12345, "step": 1})

    # With gen_model_input_fn: receives payload with metadata replaced by {}
    received_payload = []
    t.with_gen_model_input_fn(lambda p: received_payload.append(p) or {"tokens": p.token_ids})
    t._prepare_batch(payload)
    self.assertEqual(len(received_payload), 1)
    self.assertEqual(received_payload[0].metadata, {})

    # Without gen_model_input_fn: returns dict without metadata
    t_no_gen = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    prepared = t_no_gen._prepare_batch(payload)
    self.assertIsInstance(prepared, dict)
    self.assertNotIn("metadata", prepared)

    # RLTrainerPayload from Tunix with non-empty metadata is also cleared
    rl_payload = datatypes.RLTrainerPayload(
        prompt_ids=jnp.zeros((1, 4)),
        prompt_mask=jnp.ones((1, 4)),
        completion_ids=jnp.zeros((1, 4)),
        completion_mask=jnp.ones((1, 4)),
        advantages=jnp.zeros((1,)),
        metadata={"client_id": "worker-0"},
    )
    received_rl_payload = []
    t.with_gen_model_input_fn(lambda p: received_rl_payload.append(p) or {"tokens": p.completion_ids})
    t._prepare_batch(rl_payload)
    self.assertEqual(len(received_rl_payload), 1)
    self.assertEqual(received_rl_payload[0].metadata, {})

    # Dataclass without a dataclass field for metadata (e.g. property) is not replaced
    @dataclasses.dataclass
    class _PayloadWithPropertyMetadata:
      tokens: jax.Array

      @property
      def metadata(self):
        return {"property": True}

    prop_payload = _PayloadWithPropertyMetadata(tokens=jnp.ones((2, 2)))
    prepared_prop = t_no_gen._prepare_batch(prop_payload)
    self.assertIn("tokens", prepared_prop)

  def test_prepare_batch_prevents_recompilation_on_metadata_change(self):
    """Payload metadata changes must not alter the Treedef or trigger recompilation."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(lambda model, **kwargs: (abstract_engine.WeightedMetric(jnp.array(1.0), jnp.array(1.0)), {}))
    # Mirrors tunix algorithm_adapter._algo_model_input which passes the payload through.
    t.with_gen_model_input_fn(lambda payload: {"train_example": payload})

    payload1 = datatypes.RLTrainerPayload(
        prompt_ids=jnp.zeros((1, 4)),
        prompt_mask=jnp.ones((1, 4)),
        completion_ids=jnp.zeros((1, 4)),
        completion_mask=jnp.ones((1, 4)),
        advantages=jnp.zeros((1,)),
        metadata={"step": 0, "request_id": 100},
    )
    payload2 = datatypes.RLTrainerPayload(
        prompt_ids=jnp.zeros((1, 4)),
        prompt_mask=jnp.ones((1, 4)),
        completion_ids=jnp.zeros((1, 4)),
        completion_mask=jnp.ones((1, 4)),
        advantages=jnp.zeros((1,)),
        metadata={"step": 1, "request_id": 101},
    )

    with mock.patch.object(t, "_compile_for_batch", wraps=t._compile_for_batch) as mock_compile:
      t.compile(payload1)
      self.assertEqual(mock_compile.call_count, 1)
      t.fwd_bwd(payload2)
      self.assertEqual(mock_compile.call_count, 1)

    dummy1 = DummyPayload(token_ids=jnp.ones((2, 2)), metadata={"step": 0})
    dummy2 = DummyPayload(token_ids=jnp.ones((2, 2)), metadata={"step": 1})
    t_dummy = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t_dummy.with_loss_fn(lambda model, **kwargs: (abstract_engine.WeightedMetric(jnp.array(1.0), jnp.array(1.0)), {}))
    t_dummy.with_gen_model_input_fn(lambda payload: {"train_example": payload})
    with mock.patch.object(t_dummy, "_compile_for_batch", wraps=t_dummy._compile_for_batch) as mock_dummy_compile:
      t_dummy.compile(dummy1)
      self.assertEqual(mock_dummy_compile.call_count, 1)
      t_dummy.fwd_bwd(dummy2)
      self.assertEqual(mock_dummy_compile.call_count, 1)

  def test_gen_model_input_fn_returning_a_non_dict_raises(self):
    """The adapter's contract is a dict of kwargs; anything else fails clearly."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(lambda model, **kwargs: abstract_engine.WeightedMetric(jnp.array(1.0), jnp.array(1.0)))
    t.with_gen_model_input_fn(lambda payload: payload)
    with self.assertRaisesRegex(TypeError, "must return a dict of loss-fn keyword arguments"):
      t.fwd_bwd(DummyPayload())

  def _engine_with_mixed_batch(self, algo_config):
    """An engine whose adapter returns arrays alongside non-array loss arguments.

    This is the shape Tunix's GRPO adapter produces: a `TrainExample` next to an
    `algo_config` object and integer `pad_id`/`eos_id`.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    def _loss_fn(model, tokens, algo_config, pad_id, eos_id):
      del tokens, algo_config, pad_id, eos_id
      return abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0))

    t.with_loss_fn(_loss_fn)
    t.with_gen_model_input_fn(
        lambda payload: {
            "tokens": payload.token_ids,
            "algo_config": algo_config,
            "pad_id": 151643,
            "eos_id": 151645,
        }
    )
    return t

  def test_compiled_path_closes_over_non_array_loss_arguments(self):
    """A batch mixing arrays with plain objects still compiles, and matches eager.

    `algo_config` is not a JAX type, so passing it as a jit argument fails outright. It
    has to be closed over instead. Comparing gradients against the eager path is what
    proves the closed-over values actually reached the loss rather than being dropped.
    """
    algo_config = types.SimpleNamespace(beta=0.0, epsilon=0.2)

    eager = self._engine_with_mixed_batch(algo_config)
    eager.fwd_bwd(DummyPayload())
    self.assertFalse(eager._compiled, "an engine that never called compile() must stay eager")

    compiled = self._engine_with_mixed_batch(algo_config)
    compiled.compile(DummyPayload())
    self.assertTrue(compiled._compiled)
    compiled.fwd_bwd(DummyPayload())

    np.testing.assert_allclose(compiled._accumulated_grads["weights"], eager._accumulated_grads["weights"], rtol=1e-5)
    np.testing.assert_allclose(compiled._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)

  def test_compile_without_dummy_data_defers_to_first_fwd_bwd(self):
    """`compile(None)` cannot know input shapes, so it defers instead of failing.

    Tunix's `TrainerWorker.compile` passes nothing, because `PeftTrainer.compile` is a
    no-op that never needed a payload. Compiling eagerly there would jit against shapes
    the engine has not seen; refusing outright would break the worker lifecycle.
    """
    t = self._engine_with_mixed_batch(types.SimpleNamespace(beta=0.0))

    with self.assertLogs(level="INFO") as logs:
      t.compile(None)
    self.assertTrue(any("without dummy_data" in line for line in logs.output))
    self.assertFalse(t._compiled, "compile(None) has no shapes to compile against")

    # Deferred, not abandoned: the first real batch supplies the shapes.
    t.fwd_bwd(DummyPayload())
    self.assertTrue(t._compiled)
    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)

  def test_compiled_kernel_is_rebuilt_when_a_static_loss_argument_changes(self):
    """A changed non-traced loss argument must reach the loss, not the stale closure.

    Static arguments are closed over by the compiled kernel rather than passed to it, so
    keying recompilation on the traced half alone would leave a caller that swaps its
    `algo_config` (or schedules `pad_id`) silently computing against the value captured at
    compile time -- a wrong answer with no error and no log line.
    """
    algo_config = types.SimpleNamespace(scale=1.0)
    holder = [algo_config]
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    def _loss_fn(model, tokens, algo_config):
      del tokens
      return abstract_engine.WeightedMetric(
          unreduced_sum=jnp.sum(model.weights[...]) * 8.0 * algo_config.scale,
          denominator=jnp.array(4.0),
      )

    t.with_loss_fn(_loss_fn)
    t.with_gen_model_input_fn(lambda payload: {"tokens": payload.token_ids, "algo_config": holder[0]})
    t.compile(DummyPayload())

    t.fwd_bwd(DummyPayload())
    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)

    # Same payload shapes, different static value: only the static half of the signature
    # can catch this.
    holder[0] = types.SimpleNamespace(scale=3.0)
    t._accumulated_grads = None
    t.fwd_bwd(DummyPayload())
    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([24.0, 24.0]), rtol=1e-5)

  def test_uncomparable_static_arguments_warn_once(self):
    """An uncomparable static argument recompiles every step, and says so once.

    Failing towards recompilation is right for correctness, but its cost is unbounded: if
    the comparison always raises, every fwd_bwd recompiles. XLA's cache can disguise that
    as merely a slow run, so it must be diagnosable -- and warning per step would itself
    be the noise.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    def _loss_fn(model, tokens, algo_config):
      del tokens, algo_config
      return abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0))

    t.with_loss_fn(_loss_fn)
    # A fresh object per call holding an array: equality compares elementwise, so bool()
    # is ambiguous. Reusing one instance would short-circuit on identity and stay
    # comparable, which is why today's functools.partial adapter is unaffected.
    t.with_gen_model_input_fn(
        lambda payload: {
            "tokens": payload.token_ids,
            "algo_config": types.SimpleNamespace(w=jnp.ones((3,))),
        }
    )
    t.compile(DummyPayload())

    with self.assertLogs(level="WARNING") as logs:
      for _ in range(3):
        t.fwd_bwd(DummyPayload())

    matching = [line for line in logs.output if "Could not compare" in line]
    self.assertLen(matching, 1, f"expected exactly one warning, got {len(matching)}")
    self.assertIn("recompile on EVERY fwd_bwd", matching[0])
    # Names the culprit. The halves are compared separately precisely so that a
    # badly-behaved treedef cannot be reported as the caller's static arguments.
    self.assertIn("static loss arguments", matching[0])
    # Correctness is unaffected: it still recompiles rather than reusing a stale kernel.
    self.assertTrue(t._compiled)

  def test_uncomparable_batch_structure_is_reported_as_structure(self):
    """A structural comparison failure must not be blamed on static loss arguments.

    The signature is one tuple, so comparing it whole would funnel any misbehaving
    treedef or shape entry into the static-argument message and send a reader looking at
    their `gen_model_input_fn` for a fault that is not there.
    """

    class _Hostile:
      # `__eq__`, not `__ne__`: tuple comparison probes its elements with `==`.

      def __eq__(self, other):
        raise ValueError("hostile structural comparison")

      __hash__ = None

    t = self._engine_with_mixed_batch(types.SimpleNamespace(beta=0.0))
    t.compile(DummyPayload())
    # Corrupt only the structural half; the static half stays perfectly comparable.
    t._compiled_signature = (_Hostile(), (), t._compiled_signature[2])

    with self.assertLogs(level="WARNING") as logs:
      t.fwd_bwd(DummyPayload())

    matching = [line for line in logs.output if "Could not compare" in line]
    self.assertLen(matching, 1)
    self.assertIn("batch structure", matching[0])
    self.assertNotIn("static loss arguments", matching[0])

  def test_compiled_kernel_is_rebuilt_when_the_batch_shape_changes(self):
    """A differently-shaped batch recompiles rather than raising a sharding mismatch.

    `in_shardings` is baked into the compiled callable, so reusing it across a shape
    change reports an in_shardings prefix error that says nothing about the real cause.
    """
    t = self._engine_with_mixed_batch(types.SimpleNamespace(beta=0.0))
    t.compile(DummyPayload())
    first_signature = t._compiled_signature

    t.fwd_bwd(DummyPayload(token_ids=jnp.ones((4, 8)), token_mask=jnp.ones((4, 8))))

    self.assertNotEqual(first_signature, t._compiled_signature)
    self.assertTrue(t._compiled)

  def test_conforms_to_tunix_abstract_trainer(self):
    """Every method Tunix's AbstractTrainer requires exists on the engine.

    MaxText deliberately does not inherit that ABC: adding an abstractmethod upstream
    would then break construction at runtime, in production, on a version bump. This test
    buys the same drift detection and reports it as a failing test instead.

    It must iterate `__abstractmethods__` rather than list today's names -- a hard-coded
    list would never notice the additions this exists to catch.
    """
    required = abstract_trainer.AbstractTrainer.__abstractmethods__
    self.assertNotEmpty(required)
    missing = [name for name in required if not hasattr(maxtext_engine.MaxTextTrainingEngine, name)]
    self.assertEmpty(
        missing,
        f"MaxTextTrainingEngine is missing {missing}, required by tunix's AbstractTrainer. "
        "Implement them, or record why the divergence is intended.",
    )

  def test_shared_types_are_tunix_classes(self):
    """The engine's data types are Tunix's own, which is what makes a Tunix loss work.

    A same-named local copy would make every isinstance check in diff_wrapper miss and
    surface as "Unsupported return type from loss function".
    """
    self.assertIs(abstract_engine.LossOutput, sft_utils.LossOutput)
    self.assertIs(abstract_engine.WeightedMetric, sft_utils.WeightedMetric)
    self.assertIs(abstract_engine.TrainerPayload, datatypes.TrainerPayload)
    self.assertIs(abstract_engine.RLTrainerPayload, datatypes.RLTrainerPayload)

    tunix_metric = sft_utils.WeightedMetric(unreduced_sum=jnp.array(4.0), denominator=jnp.array(2.0))
    self.assertIsInstance(tunix_metric, abstract_engine.WeightedMetric)
    self.assertIsInstance(sft_utils.LossOutput(primary_loss=tunix_metric, aux_metrics={}), abstract_engine.LossOutput)

    # Must be the sft.utils class, not the same-named one in tunix.experimental.metrics
    # whose compute()/compute_scale() raise NotImplementedError -- importing that one
    # would break gradient scaling at runtime rather than at import.
    self.assertEqual(float(tunix_metric.compute()), 2.0)

    # What actually arrives at fwd_bwd from GRPOAdapter.create_trainer_payloads.
    rl_payload = datatypes.RLTrainerPayload(
        prompt_ids=jnp.zeros((1, 4)),
        prompt_mask=jnp.ones((1, 4)),
        completion_ids=jnp.zeros((1, 4)),
        completion_mask=jnp.ones((1, 4)),
        advantages=jnp.zeros((1,)),
    )
    self.assertIsInstance(rl_payload, abstract_engine.TrainerPayload)
    self.assertIsInstance(rl_payload, abstract_engine.RLTrainerPayload)

  def test_unsupported_loss_return_raises_naming_the_type(self):
    """An unrecognised return fails loudly and says what it received."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(lambda *args, **kwargs: "not a loss")
    with self.assertRaisesRegex(TypeError, "Unsupported return type.*str"):
      t.fwd_bwd(DummyPayload())

    # A 2-tuple is recognised in shape but not constructible into a WeightedMetric.
    t2 = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t2.with_loss_fn(lambda *args, **kwargs: (jnp.array(1.0), {"unrelated": jnp.array(2.0)}))
    with self.assertRaisesRegex(TypeError, "Cannot construct WeightedMetric"):
      t2.fwd_bwd(DummyPayload())

  def test_mixed_aux_dict_buckets_by_type(self):
    """WeightedMetric aux lands in weighted_metrics, plain arrays in scalar_metrics.

    Regression guard on MetricsRecorder._record_metric: reading a weighted loss out of
    scalar_metrics is what produced a fabricated 0.0 in the parity harness.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    def _loss_fn(model, *args, **kwargs):
      return sft_utils.LossOutput(
          primary_loss=sft_utils.WeightedMetric(
              unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0)
          ),
          aux_metrics={
              "kl": sft_utils.WeightedMetric(unreduced_sum=jnp.array(6.0), denominator=jnp.array(3.0)),
              "entropy": jnp.array(1.5),
          },
      )

    t.with_loss_fn(_loss_fn)
    t.fwd_bwd(DummyPayload())
    buf = t.get_metrics(clear_cache=True)

    self.assertIn("kl", buf.weighted_metrics)
    self.assertIn("entropy", buf.scalar_metrics)
    self.assertNotIn("kl", buf.scalar_metrics)
    self.assertNotIn("entropy", buf.weighted_metrics)
    self.assertAlmostEqual(float(buf.weighted_metrics["kl"].compute().item()), 2.0, places=4)
    # The primary loss is weighted, not scalar -- the specific confusion behind that 0.0.
    self.assertIn("loss", buf.weighted_metrics)
    self.assertNotIn("loss", buf.scalar_metrics)

  def test_eval_step_records_eval_metrics_and_mutates_no_training_state(self):
    """eval_step scores a batch without disturbing training, and its metrics stay separate."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )
    t.fwd_bwd(DummyPayload())

    grads_before = jax.tree.map(jnp.copy, t._accumulated_grads)
    micro_steps_before = t._micro_step_count
    train_step_before = t.train_step
    self.assertEqual(micro_steps_before, 1)
    train_entries_before = t._metrics_recorder.get_step_metrics(train_step_before).weighted_metrics["loss"]
    train_entries_before = train_entries_before.unreduced_sum.size

    with mock.patch.object(t._metrics_logger, "write_metrics") as write_metrics:
      with t.eval_context():
        t.eval_step(DummyPayload())
        t.eval_step(DummyPayload())
        t.eval_step(DummyPayload())

    # Nothing about the in-flight training step moved.
    self.assertEqual(t._micro_step_count, micro_steps_before)
    self.assertEqual(t.train_step, train_step_before)
    jax.tree.map(np.testing.assert_array_equal, grads_before, t._accumulated_grads)

    # The train buffer still holds exactly the one fwd_bwd loss: no eval leaked into it.
    train_buf = t._metrics_recorder.get_step_metrics(train_step_before)
    self.assertEqual(train_buf.weighted_metrics["loss"].unreduced_sum.size, train_entries_before)
    self.assertEqual(train_buf.mode, metrics_module.Mode.TRAIN)

    # Leaving the context writes the pass once, tagged eval, against the step it ran at --
    # not once per micro-batch, which would put three points on the curve at one x.
    self.assertEqual(write_metrics.call_count, 1)
    eval_buf = write_metrics.call_args.args[0]
    self.assertEqual(write_metrics.call_args.kwargs["mode"], metrics_module.Mode.EVAL)
    self.assertEqual(eval_buf.mode, metrics_module.Mode.EVAL)
    self.assertEqual(eval_buf.id, train_step_before)

    # All three micro-batches accumulated into that one buffer. `compute()` is elementwise,
    # so it stays per-micro-batch here; `process_metrics` is what averages it down.
    eval_loss = eval_buf.weighted_metrics["loss"]
    self.assertEqual(eval_loss.unreduced_sum.size, 3)
    self.assertAlmostEqual(float(np.mean(np.asarray(eval_loss.compute()))), 0.5, places=4)

    # The recorder is drained, so a later pass cannot re-write this one's numbers.
    self.assertEmpty(t._eval_metrics_recorder.get_metrics_history(clear_cache=False))

  def test_get_metrics_returns_one_buffer_and_a_sentinel_when_empty(self):
    """`get_metrics` returns a single buffer, matching both ABCs.

    When nothing has been recorded it returns an empty buffer identified by
    EMPTY_METRICS_BUFFER_ID rather than None, which is what Tunix's PeftTrainer does.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    empty = t.get_metrics(clear_cache=True)
    self.assertIsInstance(empty, abstract_engine.MetricsBuffer)
    self.assertEqual(empty.id, maxtext_engine.EMPTY_METRICS_BUFFER_ID)
    self.assertEmpty(empty.weighted_metrics)

    t.record_metrics(
        name="loss",
        metric=abstract_engine.WeightedMetric(unreduced_sum=jnp.array(4.0), denominator=jnp.array(2.0)),
    )
    buf = t.get_metrics(clear_cache=True)
    self.assertIsInstance(buf, abstract_engine.MetricsBuffer)
    self.assertNotIsInstance(buf, list)
    self.assertIn("loss", buf.weighted_metrics)
    # A real buffer is identified by its train step, so it never collides with the sentinel.
    self.assertNotEqual(buf.id, maxtext_engine.EMPTY_METRICS_BUFFER_ID)

    # Draining leaves nothing behind, so the sentinel comes back.
    self.assertEqual(t.get_metrics(clear_cache=True).id, maxtext_engine.EMPTY_METRICS_BUFFER_ID)

  def test_get_metrics_returns_newest_and_history_stays_reachable(self):
    """Older step buffers are still readable through the recorder, and never dropped silently."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    for step in range(3):
      t.train_step = step
      t.record_metrics(name="loss", metric=jnp.array(float(step)))

    history = t._metrics_recorder.get_metrics_history(clear_cache=False)
    self.assertLen(history, 3)
    self.assertEqual([b.id for b in history], [0, 1, 2])

    # Dropping the older buffers must be audible rather than silent.
    with self.assertLogs(level="WARNING") as logs:
      newest = t.get_metrics(clear_cache=True)
    self.assertEqual(newest.id, 2)
    self.assertIn("dropping 2 older buffer", "".join(logs.output))

  def test_metrics_history_is_bounded(self):
    """The step history is a window, so a driver that never drains it cannot grow forever.

    Each retained buffer pins live device arrays and `save_checkpoint` serializes the whole
    history, so an unbounded list would cost HBM and checkpoint latency linear in step count.
    """
    recorder = metrics_module.MetricsRecorder(max_buffered_steps=4)
    for step in range(10):
      recorder.buffer_metrics(train_step=step, name="loss", metric=jnp.array(float(step)))

    history = recorder.get_metrics_history(clear_cache=False)
    self.assertLen(history, 4)
    # The newest steps are the ones kept, and the step currently being written is never evicted.
    self.assertEqual([b.id for b in history], [6, 7, 8, 9])
    self.assertEqual(recorder.get_step_metrics(9).id, 9)
    self.assertEqual(recorder._dropped_buffer_count, 6)

    # Opting out is possible for drivers that drain the history themselves.
    unbounded = metrics_module.MetricsRecorder(max_buffered_steps=0)
    for step in range(10):
      unbounded.buffer_metrics(train_step=step, name="loss", metric=jnp.array(float(step)))
    self.assertLen(unbounded.get_metrics_history(clear_cache=False), 10)

    # The engine's own recorder is bounded by default.
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    self.assertGreater(t._metrics_recorder._max_buffered_steps, 0)

  def test_has_aux_false_drops_tuple_aux(self):
    """`has_aux=False` suppresses aux recording; `has_aux=True` keeps it.

    Both directions are asserted deliberately: checking only the `True` case would let
    an implementation that accepts the flag and ignores it pass.
    """

    def custom_loss(model, *args, **kwargs):
      unreduced_sum = jnp.sum(model.weights[...]) * 8.0
      denominator = jnp.array(4.0)
      return unreduced_sum / denominator, {
          "aux_stat": jnp.array(1.23),
          "xent_sum": unreduced_sum,
          "total_weights": denominator,
      }

    def _run(has_aux):
      t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
      t.with_loss_fn(custom_loss, has_aux=has_aux)
      t.fwd_bwd(DummyPayload())
      return t.get_metrics(clear_cache=True)

    recorded = _run(has_aux=True)
    self.assertIn("aux_stat", recorded.scalar_metrics)

    dropped = _run(has_aux=False)
    self.assertNotIn("aux_stat", dropped.scalar_metrics)
    # The primary loss is still derived from xent_sum/total_weights in the aux, so
    # suppressing the aux must not suppress the loss itself.
    self.assertIn("loss", dropped.weighted_metrics)
    self.assertAlmostEqual(float(dropped.weighted_metrics["loss"].compute().item()), 6.0, places=4)

  def test_default_loss_fn_records_aux_without_with_loss_fn(self):
    """An engine that never calls `with_loss_fn` still records the built-in loss's aux.

    The default `maxtext_train.loss_fn` returns `(loss, aux)`, and the parity harness
    relies on those aux metrics. If `_has_aux` defaulted to `with_loss_fn`'s own default
    of False, they would vanish silently.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    self.assertTrue(t._has_aux)

  def test_signatures_match_the_tunix_trainer_contract(self):
    """`with_loss_fn` returns self, `fwd_bwd` takes kwargs, `update` returns train_step."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    returned = t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )
    self.assertIs(returned, t)

    # Unknown kwargs are accepted and ignored rather than raising.
    t.fwd_bwd(DummyPayload(), skip_jit=False)
    step = t.update(skip_jit=False)
    self.assertIsInstance(step, int)
    self.assertEqual(step, t.train_step)
    self.assertEqual(step, 1)

    # With nothing accumulated, update() is a no-op that still reports the current step.
    self.assertEqual(t.update(), 1)

  def test_fwd_bwd_with_bare_weighted_metric(self):
    """A loss may return a bare WeightedMetric, carrying no aux metrics."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    payload = DummyPayload()

    def _loss_fn(model, *args, **kwargs):
      return abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0))

    t.with_loss_fn(_loss_fn)
    t.fwd_bwd(payload)

    # d(unreduced_sum)/dw is 8.0 per element; the 1/4.0 is applied once, in update().
    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)
    metrics = t.get_metrics(clear_cache=True)
    self.assertIn("loss", metrics.weighted_metrics)
    self.assertAlmostEqual(float(metrics.weighted_metrics["loss"].compute().item()), 6.0, places=4)
    # This form carries no aux, so nothing beyond the loss is recorded.
    self.assertEmpty(metrics.scalar_metrics)

  def test_fwd_bwd_with_tunix_spelled_loss_output(self):
    """A loss written against Tunix's API behaves identically to the MaxText spelling.

    `abstract_engine.LossOutput` re-exports `tunix.sft.utils.LossOutput`, so a Tunix
    loss function such as `algo_core.grpo_loss_fn` is accepted by the same branch. The
    expected values mirror `test_fwd_bwd_with_loss_output_and_aux_metrics`.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    payload = DummyPayload()

    def _tunix_loss_fn(model, *args, **kwargs):
      return sft_utils.LossOutput(
          primary_loss=sft_utils.WeightedMetric(
              unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0)
          ),
          aux_metrics={
              "metric_a": sft_utils.WeightedMetric(unreduced_sum=jnp.array(12.0), denominator=jnp.array(3.0)),
              "metric_b": jnp.array(0.42),
          },
      )

    t.with_loss_fn(_tunix_loss_fn)
    t.fwd_bwd(payload)

    np.testing.assert_allclose(t._accumulated_grads["weights"], jnp.array([8.0, 8.0]), rtol=1e-5)
    metrics = t.get_metrics(clear_cache=True)
    self.assertIn("loss", metrics.weighted_metrics)
    self.assertIn("metric_a", metrics.weighted_metrics)
    self.assertIn("metric_b", metrics.scalar_metrics)
    self.assertAlmostEqual(float(metrics.weighted_metrics["loss"].compute().item()), 6.0, places=4)
    self.assertAlmostEqual(float(metrics.weighted_metrics["metric_a"].compute().item()), 4.0, places=4)

  def _weighted_loss_fn(self, model, *_args, **_kwargs):
    """Loss whose gradient wrt the dummy model's weights is exactly [2.0, 2.0].

    `unreduced_sum` is 8 * sum(w) = 24.0 over a denominator of 4.0, so the reported loss
    is 6.0 and each gradient element is 8.0 * compute_scale() = 2.0.
    """
    return abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights[...]) * 8.0, denominator=jnp.array(4.0))

  def test_enable_lora_is_rejected_instead_of_silently_full_finetuning(self):
    """This engine has no LoRA path, so the flag must fail loudly."""
    self.assertFalse(self.mock_config.lora.enable_lora, "the default config must not enable LoRA")
    maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    self.mock_from_pretrained.assert_called_once()
    self.mock_from_pretrained.reset_mock()

    lora_config = self.setup_config(lora={"enable_lora": True, "lora_rank": 8, "lora_alpha": 16.0})
    with self.assertRaisesRegex(NotImplementedError, "does not support LoRA"):
      maxtext_engine.MaxTextTrainingEngine(lora_config)
    self.mock_from_pretrained.assert_not_called()

  def test_gradient_norm_is_recorded_every_step(self):
    cfg = self.setup_config(gradient_clipping_threshold=0.0, grad_dtype="bfloat16")
    self.assertFalse(cfg.skip_step_on_spikes, "this test covers the default, spike-detection-off path")

    t = maxtext_engine.MaxTextTrainingEngine(cfg)
    t.with_loss_fn(self._weighted_loss_fn)
    t.fwd_bwd(DummyPayload())
    t.update()

    buffer = t.get_metrics(clear_cache=True)
    self.assertIn("gradient_norm", buffer.scalar_metrics)
    grad_norm = buffer.scalar_metrics["gradient_norm"]
    np.testing.assert_allclose(np.asarray(grad_norm), [np.sqrt(8.0)], rtol=1e-3)
    # `skip_step_on_nan` defaults to True, so the default path records whether the step was
    # skipped; finite gradients mean it was not.
    self.assertTrue(cfg.skip_step_on_nan)
    np.testing.assert_array_equal(np.asarray(buffer.scalar_metrics["step_skipped"]), [0.0])

  def test_bf16_grads_divided_by_float32_denominator(self):
    """bf16 accumulated gradients are divided by the float32 denominator, not by a bf16 cast of it.

    bf16 holds integers exactly only up to 256, so a denominator of 257 casts to 256. The
    expected gradient is 256/257 rounded once to bf16 (0.99609375), not 1.0. Checked on both the
    eager and the compiled path.
    """
    cfg = self.setup_config(gradient_clipping_threshold=0.0, grad_dtype="bfloat16")
    expected = float(jnp.asarray(np.float32(256.0) / np.float32(257.0)).astype(jnp.bfloat16))
    self.assertEqual(float(jnp.asarray(257.0, jnp.bfloat16)), 256.0, "257 must be unrepresentable in bf16")
    self.assertNotEqual(expected, 1.0)
    for compiled in (False, True):
      with self.subTest(compiled=compiled):
        self.mock_from_pretrained.return_value = (DummyNNXModel(), self.mock_from_pretrained.return_value[1])
        t = maxtext_engine.MaxTextTrainingEngine(cfg)
        t.with_loss_fn(
            lambda model, *_a, **_k: abstract_engine.WeightedMetric(
                unreduced_sum=jnp.sum(model.weights[...]) * 256.0, denominator=jnp.array(257.0)
            )
        )
        if compiled:
          t.compile(DummyPayload())
        t.fwd_bwd(DummyPayload())
        t.update()
        grad_norm = t.get_metrics(clear_cache=True).scalar_metrics["gradient_norm"]
        # Both elements equal, so the norm is sqrt(2) times the normalized gradient.
        np.testing.assert_allclose(np.asarray(grad_norm), [np.sqrt(2.0) * expected], rtol=1e-6)

  def test_grads_accumulate_in_grad_dtype(self):
    """By default micro-batch gradients are accumulated in `grad_dtype`, on the eager and the compiled path.

    Under bfloat16 each micro-batch is rounded to bf16 and summed in bf16, which saves device
    memory over a float32 sum; `grad_accumulation_dtype=float32` asks for the float32 sum.
    """
    cfg = self.setup_config(gradient_clipping_threshold=0.0, grad_dtype="bfloat16")
    third = np.float32(1.0) / np.float32(3.0)
    num_micro_batches = 16
    expected_sum = jnp.zeros((), jnp.bfloat16)
    for _ in range(num_micro_batches):
      expected_sum = expected_sum + jnp.asarray(third).astype(jnp.bfloat16)
    for compiled in (False, True):
      with self.subTest(compiled=compiled):
        self.mock_from_pretrained.return_value = (DummyNNXModel(), self.mock_from_pretrained.return_value[1])
        t = maxtext_engine.MaxTextTrainingEngine(cfg)
        t.with_loss_fn(
            lambda model, *_a, **_k: abstract_engine.WeightedMetric(
                unreduced_sum=jnp.sum(model.weights[...]) * third, denominator=jnp.array(1.0)
            )
        )
        if compiled:
          t.compile(DummyPayload())
        for _ in range(num_micro_batches):
          t.fwd_bwd(DummyPayload())

        for leaf in jax.tree.leaves(t._accumulated_grads):  # pylint: disable=protected-access
          self.assertEqual(leaf.dtype, jnp.bfloat16)
          np.testing.assert_array_equal(np.asarray(leaf, np.float32), np.full(leaf.shape, float(expected_sum)))

  # A per-micro-batch gradient and denominator for which each order of sum, divide and cast gives a
  # different bf16 result: 16 micro-batches of 1/13 over 3 tokens each give 0.025634766 summed in
  # float32, divided once and cast once; 0.025756836 cast to bf16 before the division; 0.025878906
  # summed in bf16. A total denominator that is a power of two could not show where the cast
  # happens, and a denominator of 1 could not tell a sum of sums from a mean of means.
  _SCALE = np.float32(1.0) / np.float32(13.0)
  _DENOMINATOR = 3.0
  _MICRO_BATCHES = 16

  def _run_scaled_micro_batches(self, cfg, compiled):
    """Folds `_MICRO_BATCHES` micro-batches of gradient `_SCALE` over `_DENOMINATOR` tokens into a fresh engine.

    On the eager path, or through the compiled kernels if `compiled` is set.
    """
    self.mock_from_pretrained.return_value = (DummyNNXModel(), self.mock_from_pretrained.return_value[1])
    t = maxtext_engine.MaxTextTrainingEngine(cfg)
    t.with_loss_fn(
        lambda model, *_a, **_k: abstract_engine.WeightedMetric(
            unreduced_sum=jnp.sum(model.weights[...]) * self._SCALE, denominator=jnp.array(self._DENOMINATOR)
        )
    )
    if compiled:
      t.compile(DummyPayload())
    for _ in range(self._MICRO_BATCHES):
      t.fwd_bwd(DummyPayload())
    expected_sum = np.float32(0.0)
    for _ in range(self._MICRO_BATCHES):
      expected_sum = np.float32(expected_sum + self._SCALE)
    return t, expected_sum, np.float32(expected_sum / np.float32(self._DENOMINATOR * self._MICRO_BATCHES))

  def test_float32_accumulation_casts_once(self):
    """grad_accumulation_dtype=float32 under grad_dtype=bfloat16: sum in float32, divide once, cast once.

    The accumulator must hold the undivided float32 sum, and the optimizer the quotient rounded
    to bf16 once. Casting before the division, dividing each micro-batch by its own denominator,
    or dropping the final cast each change the recorded norm; see `_SCALE`.
    """
    cfg = self.setup_config(gradient_clipping_threshold=0.0, grad_dtype="bfloat16", grad_accumulation_dtype="float32")
    for compiled in (False, True):
      with self.subTest(compiled=compiled):
        t, expected_sum, expected_mean = self._run_scaled_micro_batches(cfg, compiled)
        for leaf in jax.tree.leaves(t._accumulated_grads):  # pylint: disable=protected-access
          self.assertEqual(leaf.dtype, jnp.float32)
          np.testing.assert_array_equal(np.asarray(leaf), np.full(leaf.shape, expected_sum))
        np.testing.assert_array_equal(
            np.asarray(t._accumulated_denominator),  # pylint: disable=protected-access
            np.float32(self._DENOMINATOR * self._MICRO_BATCHES),
        )

        t.update()
        mean_in_grad_dtype = float(jnp.asarray(expected_mean).astype(jnp.bfloat16))
        self.assertEqual(mean_in_grad_dtype, 210 * 2.0**-13, "`_SCALE` no longer separates the orders")
        grad_norm = t.get_metrics(clear_cache=True).scalar_metrics["gradient_norm"]
        np.testing.assert_allclose(np.asarray(grad_norm), [np.sqrt(2.0) * mean_in_grad_dtype], rtol=1e-6)

  def test_float32_grad_dtype_stays_float32(self):
    """Under grad_dtype=float32 the default is float32 end to end: nothing is rounded through bf16.

    Checks the values, not just the dtype, so a bf16 rounding anywhere between the sum and the
    optimizer fails here.
    """
    cfg = self.setup_config(gradient_clipping_threshold=0.0, grad_dtype="float32")
    for compiled in (False, True):
      with self.subTest(compiled=compiled):
        t, expected_sum, expected_mean = self._run_scaled_micro_batches(cfg, compiled)
        for leaf in jax.tree.leaves(t._accumulated_grads):  # pylint: disable=protected-access
          self.assertEqual(leaf.dtype, jnp.float32)
          np.testing.assert_array_equal(np.asarray(leaf), np.full(leaf.shape, expected_sum))

        t.update()
        grad_norm = t.get_metrics(clear_cache=True).scalar_metrics["gradient_norm"]
        np.testing.assert_allclose(np.asarray(grad_norm), [np.sqrt(2.0) * float(expected_mean)], rtol=1e-6)

  def test_bf16_weights_accumulate_in_float32(self):
    """The default accumulates in `grad_dtype`, not the weights' dtype, so bf16 weights keep a float32 sum.

    `train.py` sums these in bf16, the parameters' dtype -- a known difference from it.
    """
    cfg = self.setup_config(gradient_clipping_threshold=0.0, grad_dtype="float32")
    for compiled in (False, True):
      with self.subTest(compiled=compiled):
        model = DummyNNXModel()
        model.weights = nnx.Param(jnp.array([1.0, 2.0], jnp.bfloat16))
        self.mock_from_pretrained.return_value = (model, self.mock_from_pretrained.return_value[1])
        t = maxtext_engine.MaxTextTrainingEngine(cfg)
        t.with_loss_fn(self._weighted_loss_fn)
        if compiled:
          t.compile(DummyPayload())
        t.fwd_bwd(DummyPayload())
        t.fwd_bwd(DummyPayload())
        for leaf in jax.tree.leaves(t._accumulated_grads):  # pylint: disable=protected-access
          self.assertEqual(leaf.dtype, jnp.float32)
          # 8.0 per element per micro-batch; see `_weighted_loss_fn`.
          np.testing.assert_array_equal(np.asarray(leaf), np.full(leaf.shape, 16.0))

  def test_compiled_micro_batches_share_fwd_bwd_and_accumulate_in_place(self):
    """Every compiled micro-batch runs the one `fwd_bwd`, and each after the first adds into the donated sum.

    The running sum is never an argument of `fwd_bwd` (see `_accumulate_kernel`), and `accumulate`
    writes it back in place rather than allocating a second parameter-sized tree.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(self._weighted_loss_fn)
    t.compile(DummyPayload())
    with (
        mock.patch.object(t, "_compiled_fwd_bwd", wraps=t._compiled_fwd_bwd) as fwd_bwd,
        mock.patch.object(t, "_compiled_accumulate", wraps=t._compiled_accumulate) as accumulate,
    ):
      t.fwd_bwd(DummyPayload())
      accumulate.assert_not_called()
      first_sum = jax.tree.leaves(t._accumulated_grads)
      t.fwd_bwd(DummyPayload())
      t.fwd_bwd(DummyPayload())

    self.assertEqual(fwd_bwd.call_count, 3)
    self.assertEqual(accumulate.call_count, 2)
    self.assertTrue(all(leaf.is_deleted() for leaf in first_sum), "the running sum was copied, not donated")
    # 8.0 per element and a denominator of 4.0 per micro-batch, summed unreduced; see `_weighted_loss_fn`.
    np.testing.assert_array_equal(np.asarray(t._accumulated_grads["weights"]), [24.0, 24.0])
    np.testing.assert_array_equal(np.asarray(t._accumulated_denominator), np.float32(12.0))

  def test_empty_grad_accumulation_dtype_is_default(self):
    """`grad_accumulation_dtype=` on the command line, which reaches pydantic as None, selects the default."""
    common = {"model_name": "llama3.1-8b", "run_name": "test_run", "skip_jax_distributed_system": True}
    from_argv = pyconfig.initialize([None, get_test_config_path(), "grad_accumulation_dtype="], **common)
    self.assertEqual(from_argv.grad_accumulation_dtype, "")
    from_kwargs = pyconfig.initialize([None, get_test_config_path()], grad_accumulation_dtype=None, **common)
    self.assertEqual(from_kwargs.grad_accumulation_dtype, "")
    # Still a closed set: None is the only new spelling of the default.
    with self.assertRaisesRegex(pydantic.ValidationError, "grad_accumulation_dtype"):
      pyconfig.initialize([None, get_test_config_path(), "grad_accumulation_dtype=fp32"], **common)

  def _adam_engine(self, **config_overrides):
    """Returns an engine over a fresh dummy model with a real Adam optimizer (it has moments)."""
    mesh = self.mock_from_pretrained.return_value[1]
    self.mock_from_pretrained.return_value = (DummyNNXModel(), mesh)
    cfg = self.setup_config(gradient_clipping_threshold=0.0, **config_overrides)
    with mock.patch.object(
        maxtext_engine.train_utils,
        "create_training_optimizer",
        return_value=(lambda step: jnp.array(0.1), optax.adam(0.1)),
    ):
      t = maxtext_engine.MaxTextTrainingEngine(cfg)
    t.with_loss_fn(self._weighted_loss_fn)
    return t

  @staticmethod
  def _memory_kinds(tree):
    return {leaf.sharding.memory_kind for leaf in jax.tree.leaves(tree) if hasattr(leaf, "sharding")}

  def test_optimizer_offload_keeps_state_on_host(self):
    """With the flag set, the optimizer state stays in host memory across compiled steps."""
    t = self._adam_engine(optimizer_memory_host_offload=True)
    t.compile(DummyPayload())
    self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), {"pinned_host"})
    self.assertEqual(self._memory_kinds(nnx.state(t.state.model, nnx.Param)), {"device"})

    for _ in range(2):
      t.fwd_bwd(DummyPayload())
      t.fwd_bwd(DummyPayload())
      t.update()
      # The update kernel hands the new state straight back to the host.
      self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), {"pinned_host"})
      self.assertEqual(self._memory_kinds(nnx.state(t.state.model, nnx.Param)), {"device"})

  def test_optimizer_offload_preserves_numerics(self):
    """Offload moves bytes only: two steps give bit-identical weights and moments."""
    trained = []
    for offload in (False, True):
      t = self._adam_engine(optimizer_memory_host_offload=offload)
      t.compile(DummyPayload())
      for _ in range(2):
        t.fwd_bwd(DummyPayload())
        t.fwd_bwd(DummyPayload())
        t.update()
      trained.append(jax.device_get((nnx.state(t.state.model, nnx.Param), nnx.state(t.state.optimizer))))
    jax.tree.map(np.testing.assert_array_equal, trained[0], trained[1])
    self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), {"pinned_host"})

  def test_optimizer_state_on_device_without_offload(self):
    """Without the flag, the optimizer state stays in device memory."""
    t = self._adam_engine()
    t.compile(DummyPayload())
    t.fwd_bwd(DummyPayload())
    t.update()
    self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), {"device"})

  def test_update_reoffloads_restored_optimizer_state(self):
    """A restore can hand the moments back on the device after the kernels were compiled."""
    t = self._adam_engine(optimizer_memory_host_offload=True)
    t.compile(DummyPayload())
    t.fwd_bwd(DummyPayload())
    t.update()
    # What a restore that lands on the device leaves behind.
    on_device = jax.tree.map(
        lambda x: jax.device_put(x, x.sharding.with_memory_kind("device")), nnx.state(t.state.optimizer)
    )
    nnx.update(t.state.optimizer, on_device)
    t._invalidate_pure_state()  # pylint: disable=protected-access
    self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), {"device"})

    t.fwd_bwd(DummyPayload())
    t.update()  # Would raise an in_shardings mismatch without the re-offload.
    self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), {"pinned_host"})

  def test_optimizer_offload_on_eager_path(self):
    """`optimizer_memory_host_offload` also applies on the eager path, where `compile()` is never called.

    It changes placement only: two eager steps give the same weights and moments with it off.
    """
    trained = []
    for offload in (False, True):
      t = self._adam_engine(optimizer_memory_host_offload=offload)
      expected = {"pinned_host"} if offload else {"device"}
      for _ in range(2):
        t.fwd_bwd(DummyPayload())
        # Off HBM from the first forward pass, as a compile would have moved it.
        self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), expected)
        t.fwd_bwd(DummyPayload())
        t.update()
        self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), expected)
        self.assertEqual(self._memory_kinds(nnx.state(t.state.model, nnx.Param)), {"device"})
      self.assertFalse(t._compiled)  # pylint: disable=protected-access
      trained.append(jax.device_get((nnx.state(t.state.model, nnx.Param), nnx.state(t.state.optimizer))))
    jax.tree.map(np.testing.assert_array_equal, trained[0], trained[1])

  def test_update_after_state_setter_swaps_optimizer(self):
    """The `state` setter drops the previous compile's optimizer shardings.

    Between the setter and the next `fwd_bwd` nothing is compiled, so `update()` runs eagerly and
    must place the new optimizer state (SGD's, replacing Adam's) by its own tree structure, with
    and without offload.
    """
    for offload in (False, True):
      with self.subTest(offload=offload):
        t = self._adam_engine(optimizer_memory_host_offload=offload)
        t.compile(DummyPayload())
        t.fwd_bwd(DummyPayload())
        t.update()
        t.fwd_bwd(DummyPayload())
        before = np.asarray(t.state.model.weights[...])

        t.state = train_state_nnx.TrainStateNNX(t.model, nnx.Optimizer(t.model, optax.sgd(0.1), wrt=nnx.Param))
        self.assertIsNone(t._optimizer_device_shardings)  # pylint: disable=protected-access
        t.update()

        # `_weighted_loss_fn`'s gradient is 2.0 per element, so SGD at 0.1 moves each weight by 0.2.
        np.testing.assert_allclose(before - np.asarray(t.state.model.weights[...]), [0.2, 0.2], atol=1e-6)
        expected = {"pinned_host"} if offload else {"device"}
        self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), expected)
        # And the next step recompiles against the new tree.
        t.fwd_bwd(DummyPayload())
        t.update()
        self.assertEqual(self._memory_kinds(nnx.state(t.state.optimizer)), expected)

  def test_parameter_offload_is_rejected(self):
    """The engine has no parameter-offload path, so the flag must fail loudly."""
    with self.assertRaisesRegex(NotImplementedError, "parameter_memory_host_offload"):
      maxtext_engine.MaxTextTrainingEngine(self.setup_config(parameter_memory_host_offload=True))

  def test_differing_eval_rules_are_rejected(self):
    """`train.py` evaluates under `logical_axis_rules_for_eval`; the engine only has the training rules."""
    differing = self.setup_config(custom_mesh_and_rule_for_eval="pure-fsdp")
    self.assertNotEqual(
        _nested_tuples(differing.logical_axis_rules_for_eval), _nested_tuples(differing.logical_axis_rules)
    )
    with self.assertRaisesRegex(NotImplementedError, "pure-fsdp"):
      maxtext_engine.MaxTextTrainingEngine(differing)
    self.mock_from_pretrained.assert_not_called()

    # Both of these are accepted: the default, where the eval rules are the training rules, and an
    # eval rule set named explicitly and equal to the training one. The latter reaches the engine
    # as lists while the training rules are tuples, so it must be compared by value.
    maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    same = self.setup_config(custom_mesh_and_rule="pure-fsdp", custom_mesh_and_rule_for_eval="pure-fsdp")
    self.assertNotEqual(same.logical_axis_rules_for_eval, same.logical_axis_rules)
    maxtext_engine.MaxTextTrainingEngine(same)

  def _rules_and_mesh(self):
    """The logical axis rules and the abstract mesh in force right now."""
    return maxtext_engine.sharding.get_logical_axis_rules(), jax.sharding.get_abstract_mesh()

  def test_build_model_uses_rules_without_mesh(self):
    """What `from_pretrained` sees when Tunix constructs the engine with a mesh.

    The config's own logical axis rules must be in force, because some layers (e.g. Tokamax ring
    attention) validate their sharding at `__init__`. A `jax.set_mesh` must not be: under one,
    flax's `nnx.eval_shape` in `create_nnx_abstract_model` re-derives every sharding from its
    logical names and raises for a name the rules leave unmapped, such as `norm` under `cp-as-ep`.
    """
    model, mesh = self.mock_from_pretrained.return_value
    seen = {}

    def record(*_args, **_kwargs):
      seen["rules"], abstract_mesh = self._rules_and_mesh()
      seen["mesh_set"] = not abstract_mesh.empty
      return model  # A caller-supplied mesh means `from_pretrained` returns the bare model.

    self.mock_from_pretrained.side_effect = record
    maxtext_engine.MaxTextTrainingEngine(self.mock_config, mesh=mesh)

    self.assertEqual(seen["rules"], self.mock_config.logical_axis_rules)
    self.assertFalse(seen["mesh_set"])

  def test_build_model_clears_caller_mesh(self):
    """`with jax.set_mesh(mesh): MaxTextTrainingEngine(...)` must not reach `from_pretrained`.

    See the previous test for why; `maxtext_engine_model_build_test.py` covers it on a real
    model. The caller's mesh is restored afterwards.
    """
    model, mesh = self.mock_from_pretrained.return_value
    seen = {}

    def record(*_args, **_kwargs):
      seen["mesh_set"] = not jax.sharding.get_abstract_mesh().empty
      return model

    self.mock_from_pretrained.side_effect = record
    with jax.set_mesh(mesh):
      maxtext_engine.MaxTextTrainingEngine(self.mock_config, mesh=mesh)
      restored = jax.sharding.get_abstract_mesh()

    self.assertFalse(seen["mesh_set"])
    self.assertEqual(restored, mesh.abstract_mesh)

  def test_eval_step_traces_under_engine_rules(self):
    """The eval kernel traces under the config's rules and the engine mesh, as `fwd_bwd`'s does.

    Tunix's `TrainerWorker.compile()` passes no dummy data, so `eval_step` is where the eval
    kernel is first traced on that path. Covers both the deferred-compile path and the eager one,
    and compares by value, so a wrong rule set or a missing mesh fails too.
    """
    seen = []

    def loss_fn(model, *_args, **_kwargs):
      seen.append(self._rules_and_mesh())
      return abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(model.weights[...]), denominator=jnp.array(1.0))

    deferred = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    deferred.with_loss_fn(loss_fn)
    deferred.compile(None)  # What Tunix's worker does: no dummy payload, so nothing is built yet.
    deferred.eval_step(DummyPayload())

    self.mock_from_pretrained.return_value = (DummyNNXModel(), self.mock_from_pretrained.return_value[1])
    eager = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    eager.with_loss_fn(loss_fn)
    eager.eval_step(DummyPayload())

    self.assertLen(seen, 2)
    for engine, (rules, abstract_mesh) in zip((deferred, eager), seen):
      self.assertEqual(rules, self.mock_config.logical_axis_rules)
      self.assertEqual(abstract_mesh, engine._mesh.abstract_mesh)  # pylint: disable=protected-access

  def test_zero1_sharding_keeps_memory_kind(self):
    """`add_data_to_sharding` builds a fresh sharding; the memory kind must survive it."""
    mesh = self.mock_from_pretrained.return_value[1]
    base = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)).with_memory_kind("pinned_host")
    target = maxtext_engine._zero1_sharding(  # pylint: disable=protected-access
        mesh, jax.ShapeDtypeStruct((4,), jnp.float32), base
    )
    self.assertIsNotNone(target)
    self.assertIn("data", jax.tree.leaves(tuple(target.spec)))
    self.assertEqual(target.memory_kind, "pinned_host")

  def test_perplexity_is_emitted_alongside_the_loss(self):
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(self._weighted_loss_fn)
    t.fwd_bwd(DummyPayload())
    t.update()

    processed = metrics_module.MetricsLogger(self.mock_config).process_metrics(t.get_metrics(clear_cache=True))

    self.assertAlmostEqual(processed["loss"], 6.0, places=4)
    self.assertIn("perplexity", processed)
    self.assertAlmostEqual(processed["perplexity"], float(np.exp(6.0)), places=3)

  def test_prepare_weight_sync_rejects_an_unknown_transport(self):
    """An unrecognised transport must name itself rather than return empty metadata."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    with self.assertRaisesRegex(ValueError, "raidan"):
      t.prepare_weight_sync(staging_transport="raidan")

  def _sharded_batch_spec(self, engine, axis="data"):
    """A data sharding whose batch dim is actually sharded.

    The single-device test mesh makes `get_input_data_sharding` return a spec with `None`
    in the batch position, so the replication branch is unreachable as configured -- an
    earlier version of these tests asserted `spec[0] is None` and passed without ever
    running the code under test. Stub a spec that shards the batch dim instead.
    """
    return jax.sharding.NamedSharding(engine._mesh, jax.sharding.PartitionSpec(axis, None))  # pylint: disable=protected-access

  def test_indivisible_batch_dim_replicates_and_warns_once(self):
    """Replicating the batch dim is an N-fold compute cliff, so it must be audible."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    batch = {"a": jnp.zeros((1, 4)), "b": jnp.zeros((1, 4))}

    # Batch dim 1 against a 2-wide axis: indivisible, so the dim must be replicated.
    with mock.patch.object(maxtext_engine.sharding, "get_input_data_sharding", return_value=self._sharded_batch_spec(t)):
      with mock.patch.object(type(t), "_batch_axis_size", return_value=2):
        with self.assertLogs(level="WARNING") as logs:
          shardings = t._batch_data_shardings(batch)  # pylint: disable=protected-access
          t._batch_data_shardings(batch)  # pylint: disable=protected-access

    for name, leaf_sharding in shardings.items():
      self.assertIsNone(leaf_sharding.spec[0], f"{name} should have its batch dim replicated")

    # Once per instance, not per leaf and not per call: two leaves over two calls is four
    # chances to warn.
    warnings = [line for line in logs.output if "does not divide mesh axis" in line]
    self.assertLen(warnings, 1)
    self.assertIn("2x the work", warnings[0])

  def test_divisible_batch_dim_stays_sharded_and_is_silent(self):
    """The normal case must neither replicate nor warn."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    with mock.patch.object(maxtext_engine.sharding, "get_input_data_sharding", return_value=self._sharded_batch_spec(t)):
      with mock.patch.object(type(t), "_batch_axis_size", return_value=2):
        shardings = t._batch_data_shardings({"a": jnp.zeros((4, 4))})  # pylint: disable=protected-access

    self.assertEqual(shardings["a"].spec[0], "data")
    self.assertFalse(t._replicated_batch_warned)  # pylint: disable=protected-access

  def test_fwd_only_passes_the_live_model_and_mutates_no_training_state(self):
    """A read-only pass gets the engine's live model and leaves the in-flight step alone."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )
    t.fwd_bwd(DummyPayload())
    grads_before = jax.tree.map(jnp.copy, t._accumulated_grads)  # pylint: disable=protected-access
    micro_steps_before = t._micro_step_count  # pylint: disable=protected-access
    train_step_before = t.train_step

    def fn(model, x, *, pad_id):
      return model, pad_id, jnp.sum(x)

    model, pad_id, out = t.fwd_only(fn, np.ones((2, 4), np.int32), pad_id=0)

    # The model comes off the train state, which is what `fwd_bwd` differentiates -- not
    # `self._model`, which a restore or the `state` setter can leave behind.
    self.assertIs(model, t.state.model)
    # Python scalars must survive as scalars: callees take them as `static_argnames`, and
    # a device array there is unhashable.
    self.assertIsInstance(pad_id, int)
    self.assertEqual(float(out), 8.0)

    # Nothing about the in-flight training step moved.
    self.assertEqual(t._micro_step_count, micro_steps_before)  # pylint: disable=protected-access
    self.assertEqual(t.train_step, train_step_before)
    jax.tree.map(np.testing.assert_array_equal, grads_before, t._accumulated_grads)  # pylint: disable=protected-access

  def test_fwd_only_places_arrays_where_a_compiled_step_would(self):
    """Inputs land on the engine's own batch shardings, per leaf rank."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    # Zero-width: the empty prompt half of a sequence-packed row. There is nothing to
    # split, so it must be passed through rather than made into a degenerate global array.
    empty_prompt = np.zeros((2, 0), np.int32)

    def fn(_, tokens, prompt, *, mask):
      return tokens, prompt, mask

    with mock.patch.object(maxtext_engine.sharding, "get_input_data_sharding", return_value=self._sharded_batch_spec(t)):
      tokens, prompt, mask = t.fwd_only(fn, np.ones((2, 4), np.int32), empty_prompt, mask=np.ones((2,), np.int32))

    self.assertEqual(tokens.sharding.spec, jax.sharding.PartitionSpec("data", None))
    # A rank-1 leaf absorbs only the leading entry of the `[batch, sequence]` spec.
    self.assertEqual(mask.sharding.spec, jax.sharding.PartitionSpec("data"))
    self.assertIs(prompt, empty_prompt)

  def test_fwd_only_holds_the_logical_axis_rules_open_for_fn(self):
    """`fn` traces inside `fwd_only`, so the rules must still be in force there."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    inside = t.fwd_only(lambda _: maxtext_engine.sharding.get_logical_axis_rules())

    self.assertTrue(inside)
    # And released on exit: the scope is a loan, not a global mode switch.
    self.assertFalse(maxtext_engine.sharding.get_logical_axis_rules())

  def test_model_scope_yields_model_and_placed_inputs(self):
    """`model_scope` is what Tunix's `TrainerWorker.per_token_logps` enters to score log-probs.

    It must hand back the model a step trains, the inputs on the step's shardings with Python
    scalars left alone, and keep the logical axis rules in force for the whole block, because
    the caller's own `jax.jit` traces inside it.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)

    with mock.patch.object(maxtext_engine.sharding, "get_input_data_sharding", return_value=self._sharded_batch_spec(t)):
      with t.model_scope(np.ones((2, 4), np.int32), pad_id=0, temperature=1.0) as (model, args, kwargs):
        rules_inside = maxtext_engine.sharding.get_logical_axis_rules()
        (tokens,) = args

    self.assertIs(model, t.state.model)
    self.assertEqual(tokens.sharding.spec, jax.sharding.PartitionSpec("data", None))
    self.assertEqual(kwargs, {"pad_id": 0, "temperature": 1.0})
    self.assertIsInstance(kwargs["pad_id"], int)
    self.assertTrue(rules_inside)
    self.assertFalse(maxtext_engine.sharding.get_logical_axis_rules())

  def test_model_scope_is_context_manager(self):
    """Tunix's worker calls `model_scope` on the trainer; the engine must answer it as a context manager."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    scope = t.model_scope()
    self.assertTrue(hasattr(scope, "__enter__") and hasattr(scope, "__exit__"))
    with scope as (model, args, kwargs):
      self.assertIs(model, t.state.model)
      self.assertEqual((args, kwargs), ((), {}))

  def test_model_scope_holds_mesh_and_rules(self):
    """The caller's jit traces inside the block, so it needs what `fwd_bwd` traces under: both, by value."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    with t.model_scope():
      rules, abstract_mesh = self._rules_and_mesh()
    self.assertEqual(rules, self.mock_config.logical_axis_rules)
    self.assertEqual(abstract_mesh, t._mesh.abstract_mesh)  # pylint: disable=protected-access

  def test_model_scope_releases_on_exception(self):
    """An exception in the caller's block must not leave the engine's rules or mesh bound on the thread."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    with self.assertRaises(KeyError):
      with t.model_scope(np.ones((2, 4), np.int32)):
        raise KeyError("boom")
    rules, abstract_mesh = self._rules_and_mesh()
    self.assertFalse(rules)
    self.assertTrue(abstract_mesh.empty)

  def test_model_scope_yields_rebound_model(self):
    """After the `state` setter the model a step trains is the new one, not `self._model`.

    In the default fixture `state.model is self._model`, so the other `model_scope` tests cannot
    tell the two apart; rebinding the state here makes them differ.
    """
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    new_model = DummyNNXModel()
    t.state = train_state_nnx.TrainStateNNX(new_model, nnx.Optimizer(new_model, optax.sgd(0.1), wrt=nnx.Param))
    self.assertIsNot(new_model, t._model)  # pylint: disable=protected-access
    with t.model_scope() as (model, _, _):
      self.assertIs(model, new_model)

  def test_fwd_only_throttles_consecutive_calls_and_flushes_pending_metrics(self):
    """Consecutive `fwd_only` calls stay bounded by the throttler and flush stashed metrics."""
    t = maxtext_engine.MaxTextTrainingEngine(self.mock_config)
    t.with_loss_fn(
        lambda *args, **kwargs: (
            abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.5), denominator=jnp.array(1.0)),
            {},
        )
    )
    payload = DummyPayload()
    t.fwd_bwd(payload)
    t.update()
    # Queue now holds [fwd_bwd loss, update grad_norm + step 0 metrics] (qsize=2, full).
    self.assertTrue(t._throttler._inflight_queue.full())

    with (
        mock.patch.object(t._metrics_logger, "write_metrics") as write_metrics,
        mock.patch("jax.block_until_ready", wraps=jax.block_until_ready) as mock_block,
    ):
      for _ in range(4):
        _ = t.fwd_only(lambda _, x: jnp.sum(x), np.ones((2, 4), np.int32))
        self.assertTrue(t._throttler._inflight_queue.full())

      # Every iteration entered with a full queue, so all 4 popped and blocked.
      self.assertEqual(mock_block.call_count, 4)
      # The second iteration popped the `update` entry and flushed its stashed step-0 metrics
      # when `fwd_only` queued its computation.
      write_metrics.assert_called_once()
      self.assertIsNone(t._throttler._pending_metrics)


if __name__ == "__main__":
  absltest.main()
