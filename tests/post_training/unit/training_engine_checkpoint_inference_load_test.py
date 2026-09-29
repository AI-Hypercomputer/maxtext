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

"""A trainer checkpoint's `model_params` loads through the standalone-inference path.

The RL trainer saves `nnx.state(TunixMaxTextAdapter)`: every key carries the adapter's `base.`
prefix and the tree includes RNG leaves, in the Pathways-persistence on-disk format (no OCDBT,
zarr v2). `decode.py`, `vllm_decode.py` and a new run's `MAXTEXT_CKPT` all load it through
`model_creation_utils.from_pretrained` with default settings and no adapter. This proves that
path yields bit-identical weights and matching logits.

Lives under tests/post_training/ because it imports maxtext_engine, which pulls in tunix.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import orbax.checkpoint as ocp
import pytest

# A CPU Tier-C proof (fp32, dot_product attention); keep it out of the TPU jobs.
pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_SEQ = 8


def _cfg(base_output_directory, **overrides):
  """Tiny Qwen3.5 that keeps the real layer cycle (3 GDN + 1 full attention) and head_dim."""
  kwargs = {
      "override_model_config": True,
      "model_name": "qwen3.5-35b-a3b",
      "run_name": "ckpt_inference_load",
      "base_output_directory": base_output_directory,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "base_emb_dim": 256,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 256,  # rotary_dim / 2 == 32 == sum(mrope_section); must not shrink
      "partial_rotary_factor": 0.25,
      "base_mlp_dim": 256,
      "base_moe_mlp_dim": 256,
      "vocab_size": 256,
      "base_num_decoder_layers": 8,  # 2 per cycle offset, so each scanned stack has a real layer axis
      "inhomogeneous_layer_cycle_interval": 4,
      "scan_layers": True,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      "per_device_batch_size": 1.0,
      "dtype": "float32",
      "weight_dtype": "float32",
      "enable_dropout": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path(), "attention=dot_product"], **kwargs)


def _params(model):
  flat = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(nnx.state(model, nnx.Param)))[0]
  return {jax.tree_util.keystr(p): np.asarray(x) for p, x in flat}


def _logits(model):
  tokens = (jnp.arange(_SEQ, dtype=jnp.int32)[None, :] * 37 + 11) % 256
  positions = jnp.arange(_SEQ, dtype=jnp.int32)[None, :]
  segment_ids = jnp.ones_like(tokens)
  return np.asarray(model(tokens, positions, decoder_segment_ids=segment_ids, enable_dropout=False))


class TrainerCheckpointInferenceLoadTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.out_dir = self.enterContext(tempfile.TemporaryDirectory())  # pylint: disable=consider-using-with
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)

  def test_trainer_checkpoint_loads_via_from_pretrained_with_identical_logits(self):
    train_cfg = _cfg(
        self.out_dir,
        enable_checkpointing=True,
        async_checkpointing=True,
        checkpoint_storage_use_ocdbt=False,  # the ENABLE_PATHWAYS_PERSISTENCE on-disk format
        checkpoint_storage_use_zarr3=False,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(train_cfg), train_cfg.mesh_axes)
    engine = maxtext_engine.MaxTextTrainingEngine(train_cfg, mesh=mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=0)
    trainer = engine.model.base
    init = _params(trainer)
    # Stand-in for an optimizer update: moves every weight, salted per leaf so same-shape leaves
    # with constant init (norm scales, `dt_bias`) end up distinct and a swap between them shows.
    leaves, treedef = jax.tree.flatten(nnx.state(trainer, nnx.Param))
    moved = [x + 0.01 * jax.random.normal(jax.random.key(i), x.shape, x.dtype) for i, x in enumerate(leaves)]
    nnx.update(trainer, jax.tree.unflatten(treedef, moved))
    want = _params(trainer)
    want_logits = _logits(trainer)
    engine.save_checkpoint(metadata={"step": 1})
    engine._checkpoint_manager.close()  # pylint: disable=protected-access  # waits for the async save
    ckpt = os.path.join(train_cfg.checkpoint_dir, "1", "model_params")

    # The on-disk tree really has the trainer's shape: `base.` prefix plus RNG leaves.
    tree = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).metadata(ckpt).item_metadata.tree
    self.assertEqual(list(tree), ["base"])
    self.assertIn("rngs", str(tree))

    # Standalone consumer: no adapter, default (OCDBT/zarr3) storage flags, same geometry.
    serve_cfg = _cfg(self.out_dir, run_name="ckpt_inference_load_serve", load_parameters_path=ckpt)
    self.assertTrue(serve_cfg.checkpoint_storage_use_ocdbt and serve_cfg.checkpoint_storage_use_zarr3)
    served = model_creation_utils.from_pretrained(serve_cfg, mesh=mesh)
    got = _params(served)

    # Load-bearing: every model param was restored, and every one moved away from init.
    self.assertEqual(got.keys(), want.keys())
    # Same 70 weight leaves as the real 397B `checkpoints/2/model_params` (probe_397b_ckpt.py).
    self.assertEqual(len(got), 70)
    for name, value in want.items():
      self.assertFalse(np.array_equal(value, init[name]), msg=f"perturbation missed {name}")
      np.testing.assert_array_equal(got[name], value, err_msg=name, strict=True)
    np.testing.assert_allclose(_logits(served), want_logits, atol=1e-5, rtol=0)

  def test_restore_targets_meet_cloud_pathways_array_handler_contract(self):
    """G1 CPU pre-check: what `restore_checkpoint` hands Orbax is what Pathways can deserialize.

    `CloudPathwaysArrayHandler.deserialize` rejects anything but `ArrayRestoreArgs` with a
    `NamedSharding`, and without `global_shape`/`dtype` it opens on-disk metadata and restores the
    saved dtype with no cast. After Slice 1 any such leaf crashes the resume instead of starting
    fresh, so audit the exact targets an engine that has never compiled -- Tunix resumes right
    after `bring_up_workers(dummy_data=None)` -- hands `CheckpointManager.restore_checkpoint`.
    """
    cfg = _cfg(
        self.out_dir, enable_checkpointing=True, checkpoint_storage_use_ocdbt=False, checkpoint_storage_use_zarr3=False
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=0)
    targets = {}

    def spy(checkpoint_state, step):  # reads the targets exactly as CheckpointManager.restore_checkpoint does
      del step
      targets["model"] = nnx.state(checkpoint_state.model)
      targets["opt"] = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      return None, None, None

    with mock.patch.object(engine._checkpoint_manager, "restore_checkpoint", side_effect=spy):  # pylint: disable=protected-access
      self.assertIsNone(engine.restore_checkpoint())
    model_state, opt_state = targets["model"], targets["opt"]

    self.assertEqual(_pathways_restore_violations(model_state), [])
    self.assertEqual(_pathways_restore_violations(opt_state), [])
    # Non-vacuous: every weight, RNG and optimizer leaf was audited, typed keys included.
    counts = (_num_leaves(model_state), _num_typed_keys(model_state), _num_leaves(opt_state))
    self.assertEqual(counts, (_MODEL_LEAVES, _MODEL_TYPED_KEYS, _OPT_LEAVES))

    # Self-check of the audit: one optimizer leaf placed on a single device is exactly 1 violation.
    leaves, treedef = jax.tree.flatten(opt_state)
    idx = next(i for i, x in enumerate(leaves) if x.ndim)  # first moment/velocity leaf
    leaves[idx] = jax.device_put(leaves[idx], jax.devices()[0])
    self.assertEqual(len(_pathways_restore_violations(jax.tree.unflatten(treedef, leaves))), 1)


# Tiny-engine counts: `nnx.state(model)` leaves (weights + RNG), its typed-key leaves, `OptState` leaves.
# 106 and 143 equal the array counts of the real 397B `checkpoints/2/{model_params,optimizer_state}`.
_MODEL_LEAVES = 106
_MODEL_TYPED_KEYS = 18
_OPT_LEAVES = 143


def _num_leaves(state):
  return len(jax.tree.leaves(state))


def _is_typed_key(x):
  return jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)


def _num_typed_keys(state):
  return sum(_is_typed_key(x) for x in jax.tree.leaves(state))


def _pathways_restore_violations(state) -> list[str]:
  """Leaves whose restore args, as `CheckpointManager.restore_checkpoint` builds them, Pathways rejects or casts."""
  violations = []
  restore_args = ocp.checkpoint_utils.construct_restore_args(target=state)
  flat = jax.tree_util.tree_flatten_with_path(state)[0]
  args = jax.tree.leaves(restore_args, is_leaf=lambda x: isinstance(x, ocp.RestoreArgs))
  assert len(args) == len(flat), (len(args), len(flat))
  for (path, leaf), arg in zip(flat, args):
    name = jax.tree_util.keystr(path)
    # Typed keys are targeted as their uint32 key_data by design; the handler re-wraps them from the
    # array metadata store (RANDOM_KEY_IMPL) that the persistence registration installs.
    want = jax.eval_shape(jax.random.key_data, leaf) if _is_typed_key(leaf) else leaf
    if not isinstance(arg, ocp.ArrayRestoreArgs):
      violations.append(f"{name}: {type(arg).__name__}, not ArrayRestoreArgs")
    elif not isinstance(arg.sharding, jax.sharding.NamedSharding):
      violations.append(f"{name}: sharding {type(arg.sharding).__name__}, not NamedSharding")
    elif arg.global_shape != want.shape or arg.dtype != want.dtype:
      violations.append(f"{name}: global_shape/dtype {arg.global_shape}/{arg.dtype} != {want.shape}/{want.dtype}")
  return violations


if __name__ == "__main__":
  unittest.main()
