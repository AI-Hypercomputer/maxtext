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

"""The engine's deferred data-parallel gradient all-reduce.

Under gradient accumulation the engine used to pay one cross-replica all-reduce of the
whole parameter tree per *micro*-batch. Tagging the differentiated parameters `reduced`
over the data axis makes their cotangents `unreduced`, so the accumulation stays
replica-local and the all-reduce happens once, in `update()`.

That is a pure performance change, which makes it exactly the kind that can rot into a
no-op without anything failing. So the tests here assert on the compiled HLO -- where the
collectives actually are -- and every such assertion carries a vacuity guard proving the
same probe finds the all-reduce it is looking for when the deferral is off.
"""

import os

# Must precede the first JAX import: a data-parallel mesh needs more than one device, and
# the CPU backend reads this only at initialization.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import re  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position
from unittest import mock  # pylint: disable=wrong-import-position

from absl.testing import absltest  # pylint: disable=wrong-import-position
from flax import nnx  # pylint: disable=wrong-import-position
import jax  # pylint: disable=wrong-import-position
from maxtext.configs import pyconfig  # pylint: disable=wrong-import-position
from maxtext.training_engine import maxtext_engine  # pylint: disable=wrong-import-position
from maxtext.utils import maxtext_utils  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position
from tests.utils.test_helpers import get_test_config_path  # pylint: disable=wrong-import-position

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]

_REQUIRED_DEVICES = 4


def _config(**overrides) -> pyconfig.HyperParameters:
  """A tiny real model on an explicit, purely data-parallel mesh.

  Small enough to compile in seconds, but a *real* MaxText decoder rather than a stub: the
  reduced/unreduced tags have to survive every layer that touches a parameter, and it was a
  norm layer indexing `spec[...]` directly that broke first.
  """
  argv = [
      "maxtext_engine_deferred_all_reduce_test.py",
      get_test_config_path("base.yml"),
      "model_name=default",
      "run_name=engine_deferred_all_reduce_test",
      "enable_checkpointing=False",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "enable_dropout=False",
      "init_weights_seed=0",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "remat_policy=none",
      "scan_layers=False",
      "attention=dot_product",
      # The tag only goes on when "data" is the sole batch axis of size > 1.
      "shard_mode=explicit",
      f"ici_data_parallelism={_REQUIRED_DEVICES}",
      "ici_fsdp_parallelism=1",
      "ici_tensor_parallelism=1",
      "per_device_batch_size=1",
      # Tiny model. These must be the `base_*` names: emb_dim and mlp_dim are derived.
      "vocab_size=128",
      "base_emb_dim=64",
      "base_mlp_dim=128",
      "base_num_decoder_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "head_dim=16",
      "max_target_length=32",
      # A constant schedule with no clipping, so `update()` is the optimizer and nothing else.
      "opt_type=sgd",
      "learning_rate=1e-2",
      "gradient_clipping_threshold=0.0",
      "warmup_steps_fraction=0.0",
      "learning_rate_final_fraction=1.0",
      "gradient_accumulation_steps=1",
  ]
  argv.extend(f"{k}={v}" for k, v in overrides.items())
  return pyconfig.initialize(argv)


def _batch(cfg: pyconfig.HyperParameters, seed: int) -> dict[str, np.ndarray]:
  """Minimal batch shaped for `maxtext.trainers.pre_train.train.loss_fn`.

  NumPy, not `jnp`: the compiled kernel takes its batch on `P("data", None)`, and a
  committed device array built here would arrive on `P()` and be rejected by `jax.jit`'s
  exact `in_shardings` match. Host arrays are placed by the kernel itself.
  """
  batch, seq = int(cfg.micro_batch_size_to_train_on), cfg.max_target_length
  rng = np.random.default_rng(seed)
  tokens = rng.integers(1, cfg.vocab_size, size=(batch, seq)).astype(np.int32)
  positions = np.tile(np.arange(seq, dtype=np.int32), (batch, 1))
  segmentation = np.ones((batch, seq), dtype=np.int32)
  return {
      "inputs": tokens,
      "targets": np.roll(tokens, -1, axis=-1),
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets_segmentation": segmentation,
  }


def _no_deferral():
  """Patches the gate shut, leaving everything else about the engine identical."""
  return mock.patch.object(maxtext_engine, "_deferred_all_reduce_shardings", return_value=(None, None))


# Matches the result shape of an all-reduce instruction in optimized HLO, covering both the
# fused `%x = f32[64]{0} all-reduce(...)` form and the tupled `ROOT %y = (f32[], f32[64]{0})
# all-reduce-start(...)` one.
_ALL_REDUCE = re.compile(r"=\s*(.+?)\s+all-reduce(?:-start|-done)?\(")
# A shape with any dimension at all, i.e. not the `f32[]` of a scalar loss term.
_NON_SCALAR = re.compile(r"\[\s*\d")


def _array_all_reduces(hlo: str) -> list[str]:
  """The result shapes of every all-reduce in `hlo` that moves more than a scalar.

  Scalars are ignored on purpose: the loss and its denominator are reduced across replicas
  every micro-batch and always will be. What the deferral is about is the parameter-sized
  traffic, which is four orders of magnitude larger even on the toy model here.
  """
  shapes = [m.group(1) for line in hlo.splitlines() if (m := _ALL_REDUCE.search(line))]
  return [s for s in shapes if _NON_SCALAR.search(s)]


class _KernelHlo:
  """Captures the arguments the engine passes its jitted kernels, to re-lower them.

  `jax.jit` keeps no handle on the executable it cached, so the only way to read a kernel's
  optimized HLO is to lower it again. Recording `ShapeDtypeStruct`s rather than the arrays
  themselves keeps that independent of donation -- `_compiled_update` donates its state, so
  by the time a test asks for the HLO those buffers are gone.
  """

  def __init__(self, engine, attr: str):
    self._jitted = getattr(engine, attr)
    self._avals = None
    setattr(engine, attr, self._spy)

  def _spy(self, *args):
    if self._avals is None:
      # `sharding=None` for the host-side batch arrays, which have none; `jax.jit` places
      # those from its own `in_shardings` either way.
      self._avals = jax.tree.map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None)),
          args,
      )
    return self._jitted(*args)

  def text(self) -> str:
    if self._avals is None:
      raise AssertionError("kernel was never called, so there is no HLO to read")
    return self._jitted.lower(*self._avals).compile().as_text()


@unittest.skipIf(
    jax.device_count() < _REQUIRED_DEVICES,
    f"needs {_REQUIRED_DEVICES} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}",
)
class DeferredAllReduceGateTest(absltest.TestCase):
  """`_deferred_all_reduce_shardings` decides when the tag is legal. It must decline widely."""

  def _params_shardings(self, mesh):
    return {"w": jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())}

  def test_tags_on_a_purely_data_parallel_explicit_mesh(self):
    cfg = _config()
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    reduced, unreduced = maxtext_engine._deferred_all_reduce_shardings(  # pylint: disable=protected-access
        cfg, mesh, self._params_shardings(mesh)
    )

    self.assertEqual(reduced["w"].spec.reduced, {"data"})
    self.assertEqual(unreduced["w"].spec.unreduced, {"data"})

  def test_declines_under_auto_shard_mode(self):
    cfg = _config(shard_mode="auto")
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    self.assertEqual(
        (None, None),
        maxtext_engine._deferred_all_reduce_shardings(cfg, mesh, self._params_shardings(mesh)),  # pylint: disable=protected-access
    )

  def test_declines_on_an_auto_axis_mesh_even_in_explicit_mode(self):
    """A caller can hand the engine a mesh built by bare `jax.sharding.Mesh(...)`.

    `shard_mode=explicit` then says one thing and the mesh another, and the reduced/unreduced
    specs are rejected on Auto axes. The mesh wins.
    """
    cfg = _config()
    auto_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)

    self.assertEqual(
        (None, None),
        maxtext_engine._deferred_all_reduce_shardings(cfg, auto_mesh, self._params_shardings(auto_mesh)),  # pylint: disable=protected-access
    )

  def test_declines_when_there_are_no_data_replicas(self):
    cfg = _config(ici_data_parallelism=1, ici_fsdp_parallelism=_REQUIRED_DEVICES)
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    self.assertEqual(
        (None, None),
        maxtext_engine._deferred_all_reduce_shardings(cfg, mesh, self._params_shardings(mesh)),  # pylint: disable=protected-access
    )

  def test_declines_when_fsdp_also_shards_the_batch(self):
    """Not conservatism -- JAX rejects the backward pass outright.

    A gradient contracts over the batch, and the unreduced set has to be exactly the
    contracted axes: "unreduced axes should be equal to the contracting specs. Got unreduced
    axes=frozenset({'data'}) and contracting spec=(('data', 'fsdp'), None)". Widening the tag
    to `fsdp` is not available either, since the parameters are sharded over it.
    """
    cfg = _config(ici_data_parallelism=2, ici_fsdp_parallelism=2)
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    self.assertEqual(
        (None, None),
        maxtext_engine._deferred_all_reduce_shardings(cfg, mesh, self._params_shardings(mesh)),  # pylint: disable=protected-access
    )

  def test_a_parameter_already_sharded_over_data_is_left_alone(self):
    """`reduced` and a shard over the same axis are contradictory, and JAX says so."""
    cfg = _config()
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    shardings = {"w": jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))}

    reduced, unreduced = maxtext_engine._deferred_all_reduce_shardings(cfg, mesh, shardings)  # pylint: disable=protected-access

    self.assertEmpty(reduced["w"].spec.reduced)
    self.assertEmpty(unreduced["w"].spec.unreduced)


@pytest.mark.integration_test
@unittest.skipIf(
    jax.device_count() < _REQUIRED_DEVICES,
    f"needs {_REQUIRED_DEVICES} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}",
)
class DeferredAllReduceTest(absltest.TestCase):
  """End to end on a real decoder: where the collectives land, and what the weights do."""

  def _run(self, micro_batches: int, steps: int = 2):
    """Runs `steps` optimizer steps of `micro_batches` each; returns (engine, hlo probes)."""
    cfg = _config()
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      probes = {
          "fwd_bwd": _KernelHlo(engine, "_compiled_fwd_bwd"),
          "accum": _KernelHlo(engine, "_compiled_fwd_bwd_accum"),
          "update": _KernelHlo(engine, "_compiled_update"),
      }
      for step in range(steps):
        for micro in range(micro_batches):
          engine.fwd_bwd(_batch(cfg, step * micro_batches + micro))
        engine.update()
      return engine, probes

  def test_the_gate_opens_on_this_configuration(self):
    """Guards every other test in this class: without this they would all pass vacuously."""
    engine, _ = self._run(micro_batches=1, steps=1)

    self.assertIsNotNone(
        engine._plain_grad_shardings,  # pylint: disable=protected-access
        "the deferral never engaged, so nothing below is testing it",
    )

  def test_micro_batch_kernels_move_only_scalars_across_replicas(self):
    """The whole point: no parameter-sized all-reduce per micro-batch, only per step."""
    with jax.set_mesh(maxtext_utils.get_mesh_from_config(_config())):
      _, probes = self._run(micro_batches=2, steps=1)
      first = _array_all_reduces(probes["fwd_bwd"].text())
      accum = _array_all_reduces(probes["accum"].text())
      update = _array_all_reduces(probes["update"].text())

    self.assertEmpty(first, f"first micro-batch still all-reduces arrays: {first}")
    self.assertEmpty(accum, f"accumulating micro-batches still all-reduce arrays: {accum}")
    # Vacuity guard from the other side: the traffic did not vanish, it moved.
    self.assertNotEmpty(update, "no array all-reduce in update() either -- the gradients are never reduced")

  def test_without_the_deferral_every_micro_batch_pays(self):
    """Proves the probe above can fail. Same model, same probe, tag withheld."""
    with _no_deferral(), jax.set_mesh(maxtext_utils.get_mesh_from_config(_config())):
      engine, probes = self._run(micro_batches=2, steps=1)
      first = _array_all_reduces(probes["fwd_bwd"].text())
      accum = _array_all_reduces(probes["accum"].text())
      update = _array_all_reduces(probes["update"].text())

    self.assertIsNone(engine._plain_grad_shardings)  # pylint: disable=protected-access
    self.assertNotEmpty(first, "baseline should all-reduce the gradients in the first micro-batch")
    self.assertNotEmpty(accum, "baseline should all-reduce the gradients in every micro-batch")
    self.assertEmpty(update, f"baseline should have nothing left to reduce in update(): {update}")

  def test_deferring_does_not_change_the_weights(self):
    """Same sum, different association order.

    Deferring does not drop or duplicate a term -- it moves the cross-replica addition from
    before the micro-batch sum to after it. In exact arithmetic the two agree identically,
    and on this CPU mesh they do; in float32 on a real accelerator the reassociation shows up
    around 1e-6 relative, which is what the tolerance here allows for.
    """
    with jax.set_mesh(maxtext_utils.get_mesh_from_config(_config())):
      deferred, _ = self._run(micro_batches=3, steps=2)
    with _no_deferral(), jax.set_mesh(maxtext_utils.get_mesh_from_config(_config())):
      baseline, _ = self._run(micro_batches=3, steps=2)

    want_tree = nnx.to_pure_dict(nnx.state(baseline.model, nnx.Param))
    got_tree = nnx.to_pure_dict(nnx.state(deferred.model, nnx.Param))
    for (path, want), got in zip(jax.tree.flatten_with_path(want_tree)[0], jax.tree.leaves(got_tree)):
      np.testing.assert_allclose(
          np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6, err_msg=f"parameter {jax.tree_util.keystr(path)}"
      )

  def test_an_unreduced_accumulator_survives_a_checkpoint_round_trip(self):
    """Saving mid-step must write the gradient total, not one replica's partial.

    Orbax cannot serialize an unreduced array at all -- `device_indices_map` is undefined for
    one -- so `save_checkpoint` reduces first. This pins that the value written is the same
    total `update()` would have applied, by finishing the step from the restored state and
    checking the weights match an uninterrupted run.
    """
    output_dir = self.create_tempdir().full_path
    cfg = _config(
        enable_checkpointing=True,
        base_output_directory=output_dir,
        async_checkpointing=False,
        checkpoint_period=1,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      engine.fwd_bwd(_batch(cfg, 0))
      engine.fwd_bwd(_batch(cfg, 1))
      # Mid-step: two micro-batches in, no update() yet, so the accumulator is unreduced.
      self.assertTrue(
          jax.tree.leaves(engine._accumulated_grads)[0].sharding.spec.unreduced,  # pylint: disable=protected-access
          "the accumulator is not unreduced, so this test is not exercising the reduce-on-save",
      )
      engine.save_checkpoint(metadata={"marker": 1}, force=True)
      engine._checkpoint_manager.wait_until_finished()  # pylint: disable=protected-access

      engine.restore_checkpoint()
      engine.update()
      resumed = jax.tree.leaves(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param)))

    uninterrupted_cfg = _config()
    uninterrupted_mesh = maxtext_utils.get_mesh_from_config(uninterrupted_cfg)
    with jax.set_mesh(uninterrupted_mesh):
      straight = maxtext_engine.MaxTextTrainingEngine(uninterrupted_cfg, mesh=uninterrupted_mesh)
      straight.compile(_batch(uninterrupted_cfg, 0))
      straight.fwd_bwd(_batch(uninterrupted_cfg, 0))
      straight.fwd_bwd(_batch(uninterrupted_cfg, 1))
      straight.update()
      expected = jax.tree.leaves(nnx.to_pure_dict(nnx.state(straight.model, nnx.Param)))

    for i, (got, want) in enumerate(zip(resumed, expected)):
      np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6, err_msg=f"leaf {i}")


if __name__ == "__main__":
  absltest.main()
