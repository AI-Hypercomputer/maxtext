"""Tests for rewriting Tunix-layout post-training checkpoints into MaxText's."""

import os
import sys
import tempfile
import unittest

from etils import epath
from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest

from maxtext.checkpoint_conversion import tunix_to_maxtext
from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path


def _tiny_config(**overrides):
  """Returns a model small enough to build on CPU in a test."""
  merged = {
      "per_device_batch_size": 1.0,
      "run_name": "tunix_to_maxtext_test",
      "enable_checkpointing": False,
      "base_num_decoder_layers": 2,
      "base_emb_dim": 16,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "base_mlp_dim": 32,
      "max_target_length": 8,
      "vocab_size": 32,
      "scan_layers": False,
      **overrides,
  }
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], override_model_config=True, **merged)


def _write_old_layout(step_dir, model_state, opt_state, step):
  """Writes a step the way Tunix used to: the whole model state beside the optimizer's."""
  checkpointer = ocp.PyTreeCheckpointer()
  checkpointer.save(step_dir / "model_params", model_state)
  checkpointer.save(step_dir / "optimizer_state", {"opt_state": opt_state, "step": np.asarray(step, np.uint32)})


class PostTrainLayoutConversionTest(unittest.TestCase):
  """The conversion has to preserve every leaf and put each one where pre-training looks."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self.addCleanup(self.tmp.cleanup)
    self.config = _tiny_config()
    _, model = model_creation_utils.create_nnx_abstract_model(self.config)
    self.model_state = jax.tree.map(
        lambda x: np.ones(x.shape, np.float32) if hasattr(x, "shape") else x,
        nnx.state(model).to_pure_dict(),
    )

  def _convert(self, step=3, opt_state=None):
    root = epath.Path(self.tmp.name) / "checkpoints"
    step_dir = root / str(step)
    _write_old_layout(step_dir, self.model_state, opt_state if opt_state is not None else {}, step)
    aux, expected, shapes = tunix_to_maxtext.model_layout(self.config)
    return tunix_to_maxtext.convert_step(step_dir, aux, expected, shapes), aux

  @pytest.mark.cpu_only
  def test_finds_only_steps_still_in_the_old_layout(self):
    root = epath.Path(self.tmp.name) / "checkpoints"
    _write_old_layout(root / "1", self.model_state, {}, 1)
    _write_old_layout(root / "2", self.model_state, {}, 2)
    (root / "2" / "items").mkdir(parents=True, exist_ok=True)
    self.assertEqual([s for s, _ in tunix_to_maxtext.find_old_checkpoints(root)], [1])

  @pytest.mark.cpu_only
  def test_finds_the_actor_steps_rl_writes(self):
    root = epath.Path(self.tmp.name) / "checkpoints"
    _write_old_layout(root / "actor" / "4", self.model_state, {}, 4)
    found = tunix_to_maxtext.find_old_checkpoints(root)
    self.assertEqual([s for s, _ in found], [4])
    self.assertEqual(found[0][1].parent.name, "actor")

  @pytest.mark.cpu_only
  def test_weights_land_in_the_linen_params_collection(self):
    items, _ = self._convert()
    self.assertIn("params", items["params"])
    self.assertIn("decoder", items["params"]["params"])

  @pytest.mark.cpu_only
  def test_every_leaf_survives_the_split(self):
    """Nothing may be dropped: params and nnx_aux together have to be the model state."""
    items, _ = self._convert()
    before = tunix_to_maxtext._leaf_paths(self.model_state)  # pylint: disable=protected-access
    after = tunix_to_maxtext._leaf_paths(items["params"]["params"])  # pylint: disable=protected-access
    after |= tunix_to_maxtext._leaf_paths(items["nnx_aux"]["model"])  # pylint: disable=protected-access
    self.assertEqual(before, after)

  @pytest.mark.cpu_only
  def test_rng_state_is_kept_out_of_the_params_collection(self):
    items, aux = self._convert()
    self.assertTrue(aux, "the tiny model should carry rng state")
    in_params = tunix_to_maxtext._leaf_paths(items["params"]["params"])  # pylint: disable=protected-access
    self.assertEqual(in_params & aux, set())

  @pytest.mark.cpu_only
  def test_step_comes_from_the_optimizer_item(self):
    items, _ = self._convert(step=7)
    self.assertEqual(int(items["step"]), 7)

  @pytest.mark.cpu_only
  def test_the_inject_hyperparams_shell_is_stripped(self):
    """The trainers that schedule a learning rate wrap the optimizer; pre-training reads inside."""
    inner = {"0": {"count": np.asarray(1, np.int32), "mu": {"decoder": np.ones((2,), np.float32)}}}
    wrapped = {
        "count": np.asarray(1, np.int32),
        "hyperparams": {"learning_rate": np.asarray(0.1, np.float32)},
        "hyperparams_states": {},
        "inner_state": inner,
    }
    items, _ = self._convert(opt_state=wrapped)
    # opt_state_to_linen turns the chain's numbered keys back into the list optax uses.
    self.assertNotIn("inner_state", items["opt_state"])
    self.assertIn("params", items["opt_state"][0]["mu"])

  @pytest.mark.cpu_only
  def test_pre_training_loads_what_was_converted(self):
    """The point of the conversion: pre-training reads the result through its own loader."""
    root = epath.Path(self.tmp.name) / "checkpoints"
    step_dir = root / "5"
    _write_old_layout(step_dir, self.model_state, {}, 5)
    aux, expected, shapes = tunix_to_maxtext.model_layout(self.config)
    items = tunix_to_maxtext.convert_step(step_dir, aux, expected, shapes)
    out = epath.Path(self.tmp.name) / "converted" / "5"
    tunix_to_maxtext.write_step(items, out)

    _, model = model_creation_utils.create_nnx_abstract_model(self.config)
    abstract_params = nnx.state(model, nnx.Param)
    restored = checkpointing.load_params_from_path(
        str(out / "items"),
        abstract_params,
        checkpoint_storage_concurrent_gb=None,
        use_ocdbt=True,
        use_zarr3=False,
    )
    restored_leaves = jax.tree.leaves(restored)
    self.assertTrue(restored_leaves, "pre-training restored nothing")
    self.assertTrue(all(np.allclose(np.asarray(leaf), 1.0) for leaf in restored_leaves))

  @pytest.mark.cpu_only
  def test_a_mismatched_model_is_refused(self):
    """Splitting by the wrong model would look like it worked and quietly misplace weights."""
    root = epath.Path(self.tmp.name) / "checkpoints"
    step_dir = root / "1"
    stray = dict(self.model_state)
    stray["not_in_this_model"] = np.ones((2,), np.float32)
    _write_old_layout(step_dir, stray, {}, 1)
    aux, expected, shapes = tunix_to_maxtext.model_layout(self.config)
    with self.assertRaisesRegex(ValueError, "does not match the model"):
      tunix_to_maxtext.convert_step(step_dir, aux, expected, shapes)


if __name__ == "__main__":
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  unittest.main()
