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

"""Integration test for setup_train_loop with pure_nnx=True.

setup_train_loop wires together create_nnx_abstract_model, the training
optimizer,
the checkpoint manager, the data iterator, and finally nnx.split / nnx.merge to
return a fully-formed TrainStateNNX. This test exercises that wiring end-to-end
on a tiny synthetic config — the goal is to cover the integration glue that the
unit tests in tests/unit/train_utils_nnx_test.py cannot reach.
"""

import os
import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.utils.train_utils import setup_train_loop
from tests.utils.test_helpers import get_test_config_path
import pytest


def _tiny_nnx_pyconfig(**overrides):
  """Build a tiny pyconfig suitable for a single-host setup_train_loop run."""
  init_kwargs = {
      "run_name": "setup_train_loop_nnx_test",
      "enable_checkpointing": False,
      "dataset_type": "synthetic",
      "model_name": "default",
      "pure_nnx": True,
      "per_device_batch_size": 1.0,
      "base_emb_dim": 8,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "base_mlp_dim": 32,
      "base_num_decoder_layers": 2,
      "head_dim": 128,
      "max_target_length": 128,
      "vocab_size": 256,
      "steps": 1,
      "tokenizer_path": os.path.join(
          MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"
      ),
      "enable_goodput_recording": False,
      "enable_checkpoint_cloud_logger": False,
      "monitor_goodput": False,
  }
  init_kwargs.update(overrides)
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()], **init_kwargs
  )


@pytest.mark.integration_test
@pytest.mark.tpu_only
class SetupTrainLoopNNXIntegrationTest(unittest.TestCase):
  """End-to-end check that setup_train_loop returns a usable TrainStateNNX."""

  def test_pure_nnx_setup_returns_train_state_nnx(self):
    config = _tiny_nnx_pyconfig()

    (
        init_rng,
        checkpoint_manager,
        state_mesh_shardings,
        model,
        mesh,
        learning_rate_schedule,
        data_iterator,
        data_loader,
        rampup_manager,
        eval_data_iterator,
        train_state,
    ) = setup_train_loop(config, recorder=None)

    # The NNX path returns a fully-merged TrainStateNNX (lines 352-354 in train_utils.py).
    self.assertIsInstance(train_state, train_state_nnx.TrainStateNNX)
    # Optimizer.step starts at 0 for a fresh init.
    self.assertEqual(int(train_state.optimizer.step.get_value()), 0)
    # The returned model is train_state.model, an NNX module.
    self.assertIsInstance(model, nnx.Module)
    self.assertIs(model, train_state.model)

    # Sanity for sibling outputs:
    self.assertIsNotNone(init_rng)
    self.assertIsNotNone(mesh)
    self.assertTrue(callable(learning_rate_schedule))
    # data_loader is mandatory; data_iterator may be wrapped/unwrapped.
    self.assertIsNotNone(data_loader)
    self.assertIsNotNone(data_iterator)

    # state_mesh_shardings (NNX) is an nnx.State and contains a 'model' branch.
    self.assertIsInstance(state_mesh_shardings, nnx.State)
    self.assertIn("model", state_mesh_shardings)

    # Cleanup: the rest are not asserted on but referenced so linters don't
    # flag them as unused — they're part of the public return contract.
    del checkpoint_manager, rampup_manager, eval_data_iterator

  def test_pure_nnx_setup_param_only_split_matches_model(self):
    """nnx.split(state.model, nnx.Param, ...) must yield a non-empty Param tree

    whose structure matches state_mesh_shardings.model after the same split.
    """
    config = _tiny_nnx_pyconfig()
    *_, state_mesh_shardings, model, _, _, _, _, _, _, train_state = (
        setup_train_loop(config, recorder=None)
    )

    _, params, _ = nnx.split(train_state.model, nnx.Param, ...)
    _, params_shardings, _ = nnx.split(
        state_mesh_shardings.model, nnx.Param, ...
    )

    # Same key-set after nnx.split — this is what setup_train_loop relies on at
    # train_utils.py:281-282 to pair state_params with state_mesh_shardings_params.
    self.assertEqual(
        jax.tree_util.tree_structure(params),
        jax.tree_util.tree_structure(params_shardings),
    )
    self.assertGreater(len(jax.tree.leaves(params)), 0)

    del model

  def test_pure_nnx_dpo_setup_materializes_reference_model(self):
    """With use_dpo=True the NNX init_state_fn materializes a frozen reference

    model alongside the policy (train_utils.py:233-237). Both come from
    _create_model_partial() with the same init_weights_seed, so absent a step-0
    checkpoint the reference starts bit-identical to the policy.

    Positive replacement for the removed
    test_pure_nnx_dpo_raises_not_implemented:
    NNX DPO is supported now, so setup_train_loop builds the reference instead
    of
    raising.
    """
    config = _tiny_nnx_pyconfig(use_dpo=True, packing=False)
    *_, train_state = setup_train_loop(config, recorder=None)

    self.assertIsInstance(train_state, train_state_nnx.TrainStateNNX)
    # The reference is a sibling NNX module, distinct from the policy.
    self.assertTrue(hasattr(train_state, "reference_model"))
    self.assertIsInstance(train_state.reference_model, nnx.Module)
    self.assertIsNot(train_state.reference_model, train_state.model)

    # Same param tree, identical values at init (same seed, no step-0 override).
    policy_leaves = jax.tree.leaves(nnx.state(train_state.model, nnx.Param))
    reference_leaves = jax.tree.leaves(
        nnx.state(train_state.reference_model, nnx.Param)
    )
    self.assertGreater(len(policy_leaves), 0)
    self.assertEqual(len(policy_leaves), len(reference_leaves))
    for p, r in zip(policy_leaves, reference_leaves):
      self.assertTrue(jnp.array_equal(p, r))


if __name__ == "__main__":
  unittest.main()
