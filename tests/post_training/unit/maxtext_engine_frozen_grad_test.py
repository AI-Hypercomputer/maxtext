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

"""`trainable_parameters_mask`: `fwd_bwd` computes no gradient for a frozen parameter, and no number changes.

`update()` zeroes the gradient of every parameter the mask freezes before the norm, clipping and the
optimizer see it. `fwd_bwd` also stops that gradient inside the function it differentiates
(`_stop_frozen_gradients`), so it is never computed. Each test compares the engine with the same engine
whose `_stop_frozen_gradients` is the identity: the program that computes the frozen gradient and zeroes
it in `update()` alone. Both rigs clip every update, so a change to any gradient or to the norm would
reach the trained weights.

On a tiny real decoder with one MLP kernel frozen, trained by AdamW:
- the lowered `fwd_bwd` has one `dot_general` fewer, shaped like the frozen kernel, and none added;
- `fwd_bwd` returns exactly zero for the frozen kernel, where the reference computes a non-zero gradient;
- every other gradient, the running sums, each update's gradient norm and the trained weights are
  bitwise equal: under fsdp, under the deferred data-parallel all-reduce, and on the eager path, which
  does not cache the mask;
- control: a stop that also hits one trainable kernel is caught, and the comparison names that kernel.
These run in a subprocess with four CPU devices, for the reason given in
`maxtext_engine_data_parallel_test.py`: a sharded mesh needs more than one device, and the device count is
fixed before this file is imported.

On a one-layer Qwen3.5 MoE, with the MLPerf RL recipes' own mask, which freezes the router: the same
comparison through Tunix's router replay payload and loss. It runs on one device, in the pytest process.
"""

import collections
import contextlib
import functools
import os
import re
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import numpy as np
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_FROZEN_GRAD_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")

# The kernel the mask freezes, and the trainable kernel the control stops by mistake. Paths are written
# the way `trainable_parameters_mask` matches them (`optimizers.get_path_mask_fn`).
_FROZEN = "layers_0/mlp/wi_0/kernel"
_WRONGLY_STOPPED = "layers_1/mlp/wo/kernel"
# The mask lists what trains, so freezing one kernel takes a negative lookahead.
_MASK = f"^(?!.*{_FROZEN}).*"
# The MLPerf RL recipes' mask, verbatim (`TRAINABLE_PARAMETERS_MASK` in Tunix's `mlperf_base.sh`).
_ROUTER = "routed_experts/gate/kernel"
_ROUTER_MASK = "^(?!.*routed_experts/gate/kernel).*"
# Below every update's gradient norm (checked), so clipping scales each update by the norm and a change
# to the norm would change the trained weights.
_CLIP = 0.5

# fsdp is the kind of mesh the RL recipes train on. The deferred all-reduce returns the gradients
# `unreduced`, so a stopped leaf's zero has to come out with that tag too.
_FSDP = {"ici_fsdp_parallelism": _REQUIRED_DEVICES, "ici_data_parallelism": 1}
_DEFERRED = {"ici_fsdp_parallelism": 1, "ici_data_parallelism": _REQUIRED_DEVICES, "shard_mode": "explicit"}

# Every engine's `base_output_directory`, removed when the process exits.
_OUTPUT_DIR = tempfile.TemporaryDirectory(prefix="engine_frozen_grad_")  # pylint: disable=consider-using-with


def _moe_rig_cfg(**kwargs) -> pyconfig.HyperParameters:
  """A one-layer Qwen3.5 MoE with four experts, small enough to train on one CPU device."""
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path(), "attention=dot_product"],
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      override_model_config=True,
      model_name="qwen3.5-35b-a3b",
      num_experts=4,
      num_experts_per_tok=2,
      base_emb_dim=256,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=256,
      partial_rotary_factor=0.25,
      base_mlp_dim=256,
      base_moe_mlp_dim=256,
      vocab_size=1000,
      max_target_length=8,
      max_prefill_predict_length=8,
      per_device_batch_size=1.0,
      weight_dtype="bfloat16",
      inhomogeneous_layer_cycle_interval=1,
      base_num_decoder_layers=1,
      num_decoder_layers=1,
      scan_layers=False,
      run_name="engine_frozen_router_test",
      **kwargs,
  )


def _moe_rig_payload(first_token: int = 10) -> abstract_engine.RLTrainerPayload:
  """One 8-token row in the completion half, as the sequence-packed batch assembler emits it."""
  token_ids = jnp.arange(8, dtype=jnp.int32)[None, :] + first_token
  return abstract_engine.RLTrainerPayload(
      prompt_ids=jnp.zeros((1, 0), dtype=jnp.int32),
      prompt_mask=jnp.zeros((1, 0), dtype=jnp.int32),
      completion_ids=token_ids,
      completion_mask=jnp.ones_like(token_ids),
      advantages=jnp.zeros((1,), dtype=jnp.float32),
  )


@pytest.mark.post_training
@pytest.mark.cpu_only
def test_frozen_gradients_on_a_four_device_cpu_mesh():
  """Runs `FrozenGradientTest` in a child process with four CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


def _config(**overrides) -> pyconfig.HyperParameters:
  """A tiny real decoder, float32 throughout, trained by AdamW with clipping, with `_FROZEN` frozen."""
  fields = {
      "model_name": "default",
      "run_name": "engine_frozen_grad_test",
      "base_output_directory": _OUTPUT_DIR.name,
      "enable_checkpointing": False,
      "convert_checkpoint_if_possible": False,
      "skip_jax_distributed_system": True,
      "enable_tensorboard": False,
      "record_internal_nn_metrics": False,
      "enable_dropout": False,
      "init_weights_seed": 0,
      "dtype": "float32",
      "weight_dtype": "float32",
      "grad_dtype": "float32",
      "remat_policy": "none",
      "scan_layers": False,
      "attention": "dot_product",
      "ici_tensor_parallelism": 1,
      "per_device_batch_size": 1,
      "vocab_size": 128,
      "base_emb_dim": 64,
      "base_mlp_dim": 128,
      "base_num_decoder_layers": 2,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 16,
      "max_target_length": 32,
      "opt_type": "adamw",
      "learning_rate": 0.01,
      "gradient_clipping_threshold": _CLIP,
      "warmup_steps_fraction": 0.0,
      "learning_rate_final_fraction": 1.0,
      "profiler_steps": 0,
      "trainable_parameters_mask": _MASK,
      **_FSDP,
  }
  fields.update(overrides)
  argv = ["maxtext_engine_frozen_grad_test.py", get_test_config_path("base.yml")]
  return pyconfig.initialize(argv + [f"{k}={v}" for k, v in fields.items()])


def _batch(cfg: pyconfig.HyperParameters, seed: int) -> dict[str, np.ndarray]:
  """A `train.py` loss batch as host arrays, which the engine places itself."""
  rows, seq = int(cfg.micro_batch_size_to_train_on), cfg.max_target_length
  tokens = np.random.default_rng(seed).integers(1, cfg.vocab_size, size=(rows, seq)).astype(np.int32)
  ones = np.ones((rows, seq), dtype=np.int32)
  return {
      "inputs": tokens,
      "targets": np.roll(tokens, -1, axis=-1),
      "inputs_position": np.tile(np.arange(seq, dtype=np.int32), (rows, 1)),
      "inputs_segmentation": ones,
      "targets_segmentation": ones,
  }


def _path(path) -> str:
  """A leaf's path the way `trainable_parameters_mask` matches it, e.g. `decoder/layers_0/mlp/wi_0/kernel/value`."""
  return jax.tree_util.keystr(path, simple=True, separator="/")


def _bits(tree) -> dict[str, np.ndarray]:
  """Every leaf's raw bits by path, so comparing two trees is bitwise."""
  out = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
    value = np.asarray(jax.device_get(leaf))
    out[_path(path)] = value.view(np.uint16 if value.dtype.itemsize == 2 else np.uint32)
  return out


def _no_stop(engine, params):
  """The reference `_stop_frozen_gradients`: stops nothing, so `fwd_bwd` computes the frozen gradient too."""
  del engine
  return params


_REAL_STOP = maxtext_engine.MaxTextTrainingEngine._stop_frozen_gradients  # pylint: disable=protected-access


def _also_stopping(pattern: str):
  """A deliberately wrong `_stop_frozen_gradients`: the real one, plus a stop on every leaf whose path has `pattern`."""

  def stop(engine, params):
    return jax.tree_util.tree_map_with_path(
        lambda path, x: jax.lax.stop_gradient(x) if pattern in _path(path) else x, _REAL_STOP(engine, params)
    )

  return stop


def _with_stop(stop):
  """Replaces `MaxTextTrainingEngine._stop_frozen_gradients` by `stop` while active; None leaves the engine as it is."""
  if stop is None:
    return contextlib.nullcontext()
  return mock.patch.object(maxtext_engine.MaxTextTrainingEngine, "_stop_frozen_gradients", stop)


def _run(engine: maxtext_engine.MaxTextTrainingEngine, steps: list[list[Any]]) -> dict[str, Any]:
  """Trains one update per entry of `steps`, over that entry's micro-batches, and closes the engine.

  Returns, as raw bits by leaf path: `first`, the first micro-batch's gradients as `fwd_bwd` returned
  them; `sums`, the running sum before each update; `norms`, each update's gradient norm; `weights`, the
  trained parameters; `initial`, the parameters before training. Also `frozen`, the paths the mask froze;
  `clip`, the clipping threshold; `deferred`, whether the all-reduce was deferred; and `mask_cached`,
  whether the engine had cached the freeze mask.
  """
  params = nnx.state(engine.state.model, nnx.Param)
  initial = _bits(params)
  mask = engine._freeze_mask_for(params)  # pylint: disable=protected-access
  frozen = sorted(_path(path) for path, is_frozen in jax.tree_util.tree_leaves_with_path(mask) if is_frozen)
  first, sums, norms = None, [], []
  for micro_batches in steps:
    for micro_batch in micro_batches:
      engine.fwd_bwd(micro_batch)
      if first is None:
        first = _bits(engine._reduced_accumulated_grads())  # pylint: disable=protected-access
    sums.append(_bits(engine._reduced_accumulated_grads()))  # pylint: disable=protected-access
    engine.update()
    norms.append(np.asarray(engine.get_metrics(clear_cache=True).scalar_metrics["gradient_norm"]).reshape(-1))
  run = {
      "first": first,
      "sums": sums,
      "norms": norms,
      "weights": _bits(nnx.state(engine.state.model, nnx.Param)),
      "initial": initial,
      "frozen": frozen,
      "clip": engine._config.gradient_clipping_threshold,  # pylint: disable=protected-access
      "deferred": engine._unreduced_grad_shardings is not None,  # pylint: disable=protected-access
      "mask_cached": engine._freeze_mask is not None,  # pylint: disable=protected-access
  }
  engine.close()
  return run


def _train(stop=None, *, eager: bool = False, steps: int = 2, micro_batches: int = 2, **overrides) -> dict[str, Any]:
  """`_run` on the tiny decoder, with `_stop_frozen_gradients` replaced by `stop` for the whole run, tracing included."""
  with _with_stop(stop):
    cfg = _config(**overrides)
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=maxtext_utils.get_mesh_from_config(cfg))
    if not eager:
      engine.compile(_batch(cfg, 0))
    seeds = np.arange(steps * micro_batches).reshape(steps, micro_batches) + 1
    return _run(engine, [[_batch(cfg, int(seed)) for seed in row] for row in seeds])


@functools.cache
def _reference(eager: bool = False, steps: int = 2, mesh: str = "fsdp") -> dict[str, Any]:
  """`_train` with nothing stopped, the program before `_stop_frozen_gradients`; cached, as several tests share it."""
  return _train(_no_stop, eager=eager, steps=steps, **(_DEFERRED if mesh == "deferred" else _FSDP))


def _fwd_bwd_dots(stop=None) -> collections.Counter[str]:
  """The `dot_general` ops in `fwd_bwd` as lowered, before XLA could remove any, counted by result type."""
  with _with_stop(stop):
    cfg = _config()
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=maxtext_utils.get_mesh_from_config(cfg))
    text = engine._lower_kernels(_batch(cfg, 0))["fwd_bwd"].as_text()  # pylint: disable=protected-access
    engine.close()
  dots = [line for line in text.splitlines() if "stablehlo.dot_general" in line]
  result_types = collections.Counter(re.search(r"->\s*(tensor<[^>]*>)", line).group(1) for line in dots)
  assert sum(result_types.values()) == len(dots), "a dot_general line without a result type"
  return result_types


def _train_moe_rig(stop=None) -> dict[str, Any]:
  """`_run` on the one-layer Qwen3.5 MoE rig (`_moe_rig_cfg`), with the recipes' router mask.

  Through Tunix's router replay payload and loss, and `compile(None)` as Tunix's `TrainerWorker` calls
  it, so the kernels compile on the first `fwd_bwd`. Two updates of two micro-batches.
  """
  with _with_stop(stop):
    cfg = _moe_rig_cfg(
        trainable_parameters_mask=_ROUTER_MASK,
        gradient_clipping_threshold=_CLIP,
        base_output_directory=_OUTPUT_DIR.name,
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(0))
    with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=model):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    engine.with_gen_model_input_fn(maxtext_engine.router_replay_gen_model_input_fn)
    engine.with_loss_fn(maxtext_engine.make_router_replay_loss_fn(cfg), has_aux=True)
    engine.compile(None)
    return _run(engine, [[_moe_rig_payload(10), _moe_rig_payload(20)], [_moe_rig_payload(30), _moe_rig_payload(40)]])


class _FrozenGradientChecks:
  """The comparison every test here makes, for a `unittest.TestCase`."""

  def _differing(self, got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> list[str]:
    """The paths whose bits differ between two `_bits` dicts over the same leaves."""
    self.assertEqual(set(got), set(want))
    return sorted(path for path in want if not np.array_equal(got[path], want[path]))

  def _assert_frozen_gradient_skipped_and_nothing_else_changed(
      self, run: dict[str, Any], reference: dict[str, Any], frozen_path: str
  ):
    """`run` returned zero for the gradient of the leaf at `frozen_path`, and matches `reference` bitwise elsewhere."""
    # The mask froze exactly that one leaf, the same in both runs.
    self.assertLen(run["frozen"], 1)
    (frozen,) = run["frozen"]
    self.assertIn(frozen_path, frozen)
    self.assertEqual(reference["frozen"], run["frozen"])
    # Not vacuous: the reference computes a non-zero gradient for the frozen leaf ...
    self.assertTrue(np.any(reference["first"][frozen]), "the frozen leaf has no gradient, so this checks nothing")
    # ... where the engine returns exactly zero (every bit 0, i.e. +0.0).
    self.assertFalse(np.any(run["first"][frozen]), "fwd_bwd computed a gradient for the frozen leaf")
    # Every other gradient is bitwise unchanged, in the first micro-batch and in the running sums.
    self.assertEqual(self._differing(run["first"], reference["first"]), [frozen], "first micro-batch's gradients")
    for step, (got, want) in enumerate(zip(run["sums"], reference["sums"], strict=True)):
      self.assertEqual(self._differing(got, want), [frozen], f"running sum before update {step}")
    # So is everything the update computes from them: the norm, which clips every update, ...
    for step, (got, want) in enumerate(zip(run["norms"], reference["norms"], strict=True)):
      self.assertGreater(float(want[0]), run["clip"], f"update {step} was not clipped, so its norm reached no weight")
      np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32), err_msg=f"gradient norm, update {step}")
    # ... and every trained weight. The frozen leaf did not move.
    self.assertEqual(self._differing(run["weights"], reference["weights"]), [], "trained weights")
    np.testing.assert_array_equal(run["weights"][frozen], run["initial"][frozen], err_msg="the frozen leaf moved")


class FrozenGradientTest(_FrozenGradientChecks, parameterized.TestCase):
  """The tiny decoder with one kernel frozen, against the same engine computing its gradient."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def test_fwd_bwd_drops_the_frozen_kernels_weight_gradient(self):
    """One `dot_general` fewer, shaped like the frozen kernel, and none added: its weight gradient is never computed.

    Its gradient through to the activations is still computed, since layer 0's input depends on trainable
    parameters, so of the kernel's three matmuls (forward, input gradient, weight gradient) one goes.
    """
    reference, engine = _fwd_bwd_dots(_no_stop), _fwd_bwd_dots()
    removed, added = reference - engine, engine - reference
    self.assertEqual(dict(added), {}, f"dot_general ops the stop added, by result type; reference {dict(reference)}")
    self.assertEqual(sum(removed.values()), 1, f"dot_general ops the stop removed, by result type: {dict(removed)}")
    # The [emb, mlp] kernel's weight gradient, in either layout (JAX's transpose rule may emit it [mlp, emb]).
    cfg = _config()
    kernel_shaped = {f"tensor<{cfg.emb_dim}x{cfg.mlp_dim}xf32>", f"tensor<{cfg.mlp_dim}x{cfg.emb_dim}xf32>"}
    self.assertIn(next(iter(removed)), kernel_shaped, "the dot_general the stop removed is not shaped like the kernel")

  @parameterized.named_parameters(("fsdp", "fsdp"), ("deferred_all_reduce", "deferred"))
  def test_frozen_gradient_is_skipped_and_nothing_else_changes(self, mesh):
    run = _train(**(_DEFERRED if mesh == "deferred" else _FSDP))
    self.assertTrue(run["mask_cached"], "the compiled path did not cache the mask")
    self.assertEqual(run["deferred"], mesh == "deferred", "the deferral is not what this case is about")
    self._assert_frozen_gradient_skipped_and_nothing_else_changed(run, _reference(mesh=mesh), _FROZEN)

  def test_eager_path_computes_the_mask_and_stops_the_same_gradient(self):
    """Without `compile()` the mask is never cached; `_freeze_mask_for` computes it from the parameters."""
    run = _train(eager=True, steps=1)
    self.assertFalse(run["mask_cached"], "the mask was cached, so the eager fallback did not run")
    self._assert_frozen_gradient_skipped_and_nothing_else_changed(run, _reference(eager=True, steps=1), _FROZEN)

  def test_control_a_stop_on_a_trainable_kernel_is_caught(self):
    """Control: the same comparison, against a stop that also hits one trainable kernel, fails and names it."""
    reference = _reference(mesh="fsdp")
    wrong = _train(_also_stopping(_WRONGLY_STOPPED))
    (frozen,) = wrong["frozen"]
    (wrongly_stopped,) = [path for path in reference["first"] if _WRONGLY_STOPPED in path]
    # The wrongly stopped kernel's gradient is zero, and it is the only other one that changed ...
    self.assertFalse(np.any(wrong["first"][wrongly_stopped]))
    self.assertEqual(self._differing(wrong["first"], reference["first"]), sorted([frozen, wrongly_stopped]))
    # ... so the norm differs, and with it every update.
    self.assertFalse(np.array_equal(wrong["norms"][0], reference["norms"][0]))
    with self.assertRaisesRegex(AssertionError, re.escape(wrongly_stopped)):
      self._assert_frozen_gradient_skipped_and_nothing_else_changed(wrong, reference, _FROZEN)


class FrozenRouterTest(_FrozenGradientChecks, absltest.TestCase):
  """The MLPerf RL recipes' router freeze on a one-layer Qwen3.5 MoE, against the same engine computing it."""

  def test_the_router_gradient_is_skipped_and_nothing_else_changes(self):
    run, reference = _train_moe_rig(), _train_moe_rig(_no_stop)
    self.assertTrue(run["mask_cached"], "the compiled path did not cache the mask")
    self._assert_frozen_gradient_skipped_and_nothing_else_changed(run, reference, _ROUTER)


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FrozenGradientTest))
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")
