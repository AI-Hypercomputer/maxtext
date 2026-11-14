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

"""End-to-end CPU numeric parity tests of `MaxTextTrainingEngine` against `pre_train/train.py`.

Each suite trains a tiny real MaxText model end to end:

- `TrainPyParityTest` trains one tiny dense model on the same data twice, through
  `pre_train/train.py`'s jitted `train_step` and through the engine, and compares the loss at
  every step and the final parameters.
- `MoeLoadBalanceTest` verifies how the two paths weight the MoE load-balancing loss under
  gradient accumulation.

The suites need four CPU devices. The CPU backend reads `--xla_force_host_platform_device_count`
only when it initializes, which has already happened by the time pytest imports this module, so
pytest collects `TrainSubprocessTest`, which re-runs this file in a child process with the flag set.
"""

import math
import os
import re
import subprocess
import sys
import unittest
from typing import Any

from absl import app
from absl.testing import absltest
from flax import nnx
from flax.core import spmd as flax_spmd
import jax
import numpy as np
import pytest

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training, pytest.mark.integration_test, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_TRAIN_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")
# Measurements the suites print for a human to read; the parent echoes them.
_MEASURE = "ENGINE_TRAIN_MEASURE"
_CHILD_ENV = "_MAXTEXT_TRAIN_CHILD"
_IN_CHILD = os.environ.get(_CHILD_ENV) == "1"
if _IN_CHILD:
  os.environ["XLA_FLAGS"] = (
      f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
  ).strip()
  os.environ["JAX_PLATFORMS"] = "cpu"
_MULTI_DEVICE_BASE = absltest.TestCase if _IN_CHILD else object

_SUITES = ("TrainPyParityTest", "MoeLoadBalanceTest")


class TrainSubprocessTest(absltest.TestCase):
  """Runs the four-device training suites in a single child process."""

  def test_all_suites_in_subprocess(self):
    """Runs `_SUITES` in a child process with four CPU devices."""
    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    env["XLA_FLAGS"] = (f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}").strip()
    env["JAX_PLATFORMS"] = "cpu"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

    cmd = [sys.executable, __file__] if sys.executable else [sys.argv[0]]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

    for line in result.stdout.splitlines():
      if line.startswith(_MEASURE):
        print(line)
    report = f"stdout:\n{result.stdout[-20000:]}\nstderr:\n{result.stderr[-20000:]}"
    self.assertEqual(result.returncode, 0, report)
    ran = _RAN.search(result.stdout)
    self.assertIsNotNone(ran, f"the child did not report a completed run\n{report}")
    self.assertGreater(int(ran.group(1)), 0, "every test in the child skipped")


# ---------------------------------------------------------------------------------------
# Shared helpers. MaxText and Tunix modules are imported inside the functions that use
# them, so collecting this file in the parent pytest process imports neither.
# ---------------------------------------------------------------------------------------


def _measure(**fields: Any) -> None:
  """Prints one measurement line the parent echoes, `key=value` separated by spaces."""
  print(_MEASURE, " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


def _dense_overrides(**overrides) -> dict[str, Any]:
  """A tiny real dense decoder: two layers, 32 wide, 16 tokens. Fast to compile on CPU."""
  base = {
      "model_name": "default",
      "vocab_size": 32,
      "base_emb_dim": 32,
      "base_mlp_dim": 64,
      "base_num_decoder_layers": 2,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 16,
      "max_target_length": 16,
      "per_device_batch_size": 1,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "enable_dropout": False,
      "enable_checkpointing": False,
      "enable_tensorboard": False,
      "record_internal_nn_metrics": False,
      "skip_jax_distributed_system": True,
      "profiler_steps": 0,
      "log_config": False,
      "init_weights_seed": 0,
  }
  base.update(overrides)
  return base


def _argv(run_name: str, output_dir: str, overrides: dict[str, Any]) -> list[str]:
  """A MaxText command line over the test `base.yml` with `overrides` as `key=value` arguments."""
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  return [
      sys.argv[0],
      get_test_config_path("base.yml"),
      f"run_name={run_name}",
      f"base_output_directory={output_dir}",
  ] + [f"{k}={v}" for k, v in overrides.items()]


def _lm_batch(rows: int, seq: int, vocab: int, seed: int) -> dict[str, np.ndarray]:
  """A `train.py` loss batch with ragged sequence lengths, as host arrays.

  With every position real, a micro-batch's `total_weights` is a power of two and dividing by
  it is exact in any float format, so a gradient cast to bf16 before the division could not be
  told apart from one cast after it.
  """
  rng = np.random.default_rng(seed)
  lengths = seq - (np.arange(rows) * 3 + seed) % 7
  real = (np.arange(seq)[None, :] < lengths[:, None]).astype(np.int32)
  tokens = rng.integers(1, vocab, size=(rows, seq)).astype(np.int32) * real
  return {
      "inputs": tokens,
      "targets": np.roll(tokens, -1, axis=-1) * np.roll(real, -1, axis=-1),
      "inputs_position": np.tile(np.arange(seq, dtype=np.int32), (rows, 1)),
      "inputs_segmentation": real,
      "targets_segmentation": real * np.roll(real, -1, axis=-1),
  }


def _flat_params(model) -> dict[str, np.ndarray]:
  """`{path: float64 array}` over a model's `nnx.Param` leaves."""
  flat = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.to_pure_dict(nnx.state(model, nnx.Param))):
    flat[jax.tree_util.keystr(path)] = np.asarray(leaf, dtype=np.float64)
  return flat


def _update_rel(p0: dict, pa: dict, pb: dict) -> float:
  """`||(pa - p0) - (pb - p0)|| / ||pb - p0||` over the whole tree: how far apart two updates are."""
  num = sum(float(np.sum((pa[k] - pb[k]) ** 2)) for k in pb)
  den = sum(float(np.sum((pb[k] - p0[k]) ** 2)) for k in pb)
  if not den:
    raise AssertionError("The reference update left every parameter unchanged, so there is nothing to compare.")
  return math.sqrt(num / den)


def _max_diff(pa: dict, pb: dict) -> tuple[float, float]:
  """Largest absolute and largest relative elementwise difference between two param trees."""
  max_abs = max(float(np.max(np.abs(pa[k] - pb[k]))) for k in pb)
  max_rel = max(float(np.max(np.abs(pa[k] - pb[k]) / np.maximum(np.abs(pb[k]), 1e-12))) for k in pb)
  return max_abs, max_rel


def _copy_arrays(tree):
  """`tree` with every `jax.Array` leaf copied into a fresh buffer."""
  return jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, tree)


def _state_shardings(state, mesh):
  """Each leaf's own sharding if it is a `NamedSharding` on `mesh`, else replicated over `mesh`."""
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def home(leaf):
    leaf_sharding = getattr(leaf, "sharding", None)
    if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == mesh:
      return leaf_sharding
    return replicated

  return jax.tree.map(home, state)


def _jit_train_py_step(cfg, mesh, graphdef, state, state_mesh_shardings):
  """`pre_train/train.py`'s `train_step`, jitted the way train.py jits it; returns it and the data sharding."""
  # pylint: disable=import-outside-toplevel
  from maxtext.trainers.pre_train import train as pre_train
  from maxtext.utils import sharding
  from maxtext.utils import train_utils

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(cfg, state_mesh_shardings)
  data_sharding = sharding.get_input_data_sharding(cfg, mesh)
  train_step = train_utils.jit_train_step(
      cfg, graphdef, state, state_mesh_shardings, data_sharding, pre_train.train_step, params_shardings, mesh
  )
  return train_step, data_sharding


# ---------------------------------------------------------------------------------------
# 1. Parity with pre_train/train.py's train_step.
# ---------------------------------------------------------------------------------------


class TrainPyParityTest(_MULTI_DEVICE_BASE):
  """The engine against `pre_train/train.py`'s jitted `train_step`: same model, data and initial state.

  Both paths use SGD with a constant learning rate and no clipping, so the parameters after N
  steps are `p0 - lr * sum(grads)` and a gradient difference reaches them linearly. (Adam
  normalises each element by its own magnitude, which hides a relative gradient error at step
  one, so it gets a single case.) train.py's state is a copy of the engine's, taken before
  either trains. Micro-batch k of a G-step update is rows `k::G` of the global batch, the split
  `gradient_accumulation.py` makes, and sequence lengths are ragged (see `_lm_batch`).

  Every case bounds the largest per-step relative loss difference by 1e-5, and
  `update_rel = ||dp_engine - dp_train_py|| / ||dp_train_py||` over the whole tree after
  `_STEPS` steps (`dp = p_final - p0`) by 1e-4 with float32 gradients and 3e-4 with bfloat16
  ones. The forward pass is float32 and the parameters differ by a fraction of one update, so
  for bfloat16 gradients it is the update gate that checks the order of the cast.

  train.py sums micro-batch gradients in the parameters' dtype (`gradient_accumulation.py`
  accumulates into `zeros_like` of them), divides by the token count, and then casts float32
  leaves to `grad_dtype`. With float32 weights and gradients, or bfloat16 weights and bfloat16
  gradients, both sides accumulate and update in the same dtype.
  """

  _STEPS = 2
  _LR = 0.1

  def _cfg(self, name: str, **overrides):
    """The tiny dense model with the SGD setup described in the class docstring."""
    # pylint: disable-next=import-outside-toplevel
    from maxtext.configs import pyconfig

    fields = {
        "opt_type": "sgd",
        "learning_rate": self._LR,
        "warmup_steps_fraction": 0.0,
        "learning_rate_final_fraction": 1.0,
        "gradient_clipping_threshold": 0.0,
        "steps": self._STEPS,
        "remat_policy": "none",
    }
    fields.update(overrides)
    return pyconfig.initialize(_argv(name, self.create_tempdir().full_path, _dense_overrides(**fields)))

  def _run(self, name: str, **overrides) -> dict[str, float]:
    """Trains the same model on the same data through train.py and the engine; returns the gaps."""
    # pylint: disable=import-outside-toplevel
    from maxtext.training_engine import maxtext_engine
    from maxtext.utils import maxtext_utils

    cfg = self._cfg(name, **overrides)
    num_micro = cfg.gradient_accumulation_steps
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)

    # train.py's state: a copy of the engine's, taken before either side trains.
    graphdef, state = nnx.split(engine.state)
    state_mesh_shardings = _state_shardings(state, mesh)
    state = jax.device_put(_copy_arrays(state), state_mesh_shardings)
    p_train_step, data_sharding = _jit_train_py_step(cfg, mesh, graphdef, state, state_mesh_shardings)
    p0 = _flat_params(engine.state.model)

    micro_rows = int(cfg.micro_batch_size_to_train_on)
    engine_losses, train_losses, denominators = [], [], []
    for step in range(self._STEPS):
      global_batch = _lm_batch(micro_rows * num_micro, cfg.max_target_length, cfg.vocab_size, seed=step)
      micro_batches = [{k: v[m::num_micro] for k, v in global_batch.items()} for m in range(num_micro)]
      denominators.extend(int(np.sum(mb["targets_segmentation"] != 0)) for mb in micro_batches)

      # pre_train/train.py, in the context its training loop calls it under (train.py `train_loop`).
      with jax.set_mesh(mesh), flax_spmd.logical_axis_rules(cfg.logical_axis_rules):
        state, metrics = p_train_step(state, jax.device_put(global_batch, data_sharding))
      train_losses.append(float(metrics["scalar"]["learning/loss"]))

      # The engine, as its callers drive it: no context at all.
      if step == 0:
        engine.compile(micro_batches[0])
      for micro_batch in micro_batches:
        engine.fwd_bwd(micro_batch)
      engine.update()
      loss = engine.get_metrics().weighted_metrics["loss"]
      engine_losses.append(float(np.sum(loss.unreduced_sum) / np.sum(loss.denominator)))

    # Precondition: with every denominator a power of two the bf16 cases could not see the cast
    # order (see `_lm_batch`).
    self.assertTrue(any(d & (d - 1) for d in denominators), f"every denominator is a power of two: {denominators}")
    p_engine = _flat_params(engine.state.model)
    p_train = _flat_params(nnx.merge(graphdef, state).model)
    self.assertEqual(set(p_engine), set(p_train))
    loss_rel = max(abs(e - t) / abs(t) for e, t in zip(engine_losses, train_losses))
    max_abs, max_rel = _max_diff(p_engine, p_train)
    result = {
        "loss_rel": loss_rel,
        "update_rel": _update_rel(p0, p_engine, p_train),
        "param_max_abs": max_abs,
        "param_max_rel": max_rel,
    }
    _measure(
        suite="parity",
        case=name,
        engine_losses=",".join(f"{l:.7f}" for l in engine_losses),
        train_losses=",".join(f"{l:.7f}" for l in train_losses),
        **{k: f"{v:.3e}" for k, v in result.items()},
    )
    engine.close()
    return result

  def _check(self, result, update_gate):
    """The loss at every step within 1e-5 relative, and the update within `update_gate`."""
    self.assertLessEqual(result["loss_rel"], 1e-5, result)
    self.assertLessEqual(result["update_rel"], update_gate, result)

  def test_float32_grads_one_micro_batch(self):
    self._check(self._run("parity_fp32_g1", grad_dtype="float32", gradient_accumulation_steps=1), 1e-4)

  def test_float32_grads_two_micro_batches(self):
    self._check(self._run("parity_fp32_g2", grad_dtype="float32", gradient_accumulation_steps=2), 1e-4)

  def test_float32_grads_two_micro_batches_adamw(self):
    """Within the gate but not exact, unlike the two-micro-batch SGD cases.

    XLA can fold train.py's accumulating add into the token embedding's gradient scatter-add, while
    the engine's `accumulate` adds in source order, and Adam's per-element normalization magnifies
    that rounding difference on elements whose gradients are near zero. It is not a reason to loosen
    the gate.
    """
    self._check(
        self._run("parity_fp32_g2_adamw", grad_dtype="float32", gradient_accumulation_steps=2, opt_type="adamw"), 1e-4
    )

  def test_bfloat16_weights_default_accumulation(self):
    self._check(
        self._run("parity_bf16w_g2", weight_dtype="bfloat16", grad_dtype="bfloat16", gradient_accumulation_steps=2),
        3e-4,
    )


# ---------------------------------------------------------------------------------------
# 2. MoE load-balancing loss under gradient accumulation.
# ---------------------------------------------------------------------------------------


class MoeLoadBalanceTest(_MULTI_DEVICE_BASE):
  """How train.py's accumulation path and the engine weight the MoE load-balancing loss.

  `loss_fn` returns `xent_sum + moe_lb_loss * total_weights` (`_add_aux_loss`), and gradient
  accumulation divides the summed gradient by the total token count, so each micro-batch's lb is
  weighted by its token share. Both G=2 paths are expected to agree with the lb loss on as well
  as off.

  Each path takes one SGD step at lb weight 0 and `_LB_WEIGHT`, with `g = (p0 - p1) / lr`. The
  test asserts that the gradients are finite, that the lb loss reaches the router, and that the
  two G=2 paths agree with it off and on.
  """

  _LR = 0.1
  _LB_WEIGHT = 0.01

  def _cfg(self, name, **overrides):
    """A tiny qwen3 MoE (4 experts, top-2) on the dense-matmul path, trained with plain SGD."""
    # pylint: disable-next=import-outside-toplevel
    from maxtext.configs import pyconfig

    fields = _dense_overrides(
        model_name="qwen3-30b-a3b",
        override_model_config=True,
        base_emb_dim=32,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=16,
        base_mlp_dim=64,
        base_moe_mlp_dim=32,
        base_num_decoder_layers=1,
        vocab_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        sparse_matmul=False,
        megablox=False,
        opt_type="sgd",
        learning_rate=self._LR,
        warmup_steps_fraction=0.0,
        learning_rate_final_fraction=1.0,
        gradient_clipping_threshold=0.0,
        grad_dtype="float32",
        steps=1,
        remat_policy="none",
    )
    fields.update(overrides)
    return pyconfig.initialize(_argv(name, self.create_tempdir().full_path, fields))

  def _router_grads(self, lb_weight: float) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Router gradients of the engine (G=2) and train.py (G=2), and the batch's token count.

    The gradients are `{path name: {router param: gradient}}`.
    """
    # pylint: disable=import-outside-toplevel
    from maxtext.training_engine import maxtext_engine
    from maxtext.utils import maxtext_utils

    cfg = self._cfg("moe_lb_engine", gradient_accumulation_steps=2, load_balance_loss_weight=lb_weight)
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    graphdef, state0 = nnx.split(engine.state)
    state_mesh_shardings = _state_shardings(state0, mesh)
    state0 = _copy_arrays(state0)
    p0 = _flat_params(engine.state.model)
    rows = int(cfg.micro_batch_size_to_train_on)
    global_batch = _lm_batch(2 * rows, cfg.max_target_length, cfg.vocab_size, seed=0)

    grads = {}
    state = jax.device_put(_copy_arrays(state0), state_mesh_shardings)
    step, data_sharding = _jit_train_py_step(cfg, mesh, graphdef, state, state_mesh_shardings)
    with jax.set_mesh(mesh), flax_spmd.logical_axis_rules(cfg.logical_axis_rules):
      state, _ = step(state, jax.device_put(global_batch, data_sharding))
    p1 = _flat_params(nnx.merge(graphdef, state).model)
    grads["train_py_g2"] = {k: (v - p1[k]) / self._LR for k, v in p0.items()}
    del step, state, p1
    jax.clear_caches()

    micro_batches = [{k: v[m::2] for k, v in global_batch.items()} for m in range(2)]
    shares = [int(np.sum(mb["targets_segmentation"] != 0)) for mb in micro_batches]
    self.assertNotEqual(shares[0], shares[1], f"equal token shares cannot tell token-share from 1/G weighting: {shares}")
    engine.compile(micro_batches[0])
    for micro_batch in micro_batches:
      engine.fwd_bwd(micro_batch)
    engine.update()
    p1 = _flat_params(engine.state.model)
    grads["engine_g2"] = {k: (v - p1[k]) / self._LR for k, v in p0.items()}
    engine.close()
    total_weights = int(np.sum(global_batch["targets_segmentation"] != 0))
    router = [k for k in p0 if "gate" in k and "moe" in k.lower()] or [k for k in p0 if "gate" in k]
    return {label: {k: g[k] for k in router} for label, g in grads.items()}, total_weights

  @staticmethod
  def _norm(tree) -> float:
    return math.sqrt(sum(float(np.sum(v**2)) for v in tree.values()))

  @staticmethod
  def _diff(a, b) -> dict:
    return {k: a[k] - b[k] for k in a}

  def test_load_balance_loss_weighting(self):
    off, _ = self._router_grads(0.0)
    on, total_weights = self._router_grads(self._LB_WEIGHT)
    self.assertTrue(on["engine_g2"], "found no router parameters to measure")
    for state_name, grads in (("off", off), ("on", on)):
      for label, tree in grads.items():
        self.assertTrue(all(np.all(np.isfinite(v)) for v in tree.values()), f"{state_name}/{label}")

    off_rel = self._norm(self._diff(off["engine_g2"], off["train_py_g2"])) / self._norm(off["train_py_g2"])
    self.assertLessEqual(off_rel, 1e-4, f"engine and train.py disagree with lb loss off: {off_rel}")

    lb_delta_train_py = self._diff(on["train_py_g2"], off["train_py_g2"])
    lb_delta_norm = self._norm(lb_delta_train_py)
    self.assertGreater(lb_delta_norm, 0.0, "the lb loss does not reach the router, so this checks nothing")

    lb_engine = self._diff(on["engine_g2"], off["engine_g2"])
    lb_rel = self._norm(self._diff(lb_engine, lb_delta_train_py)) / lb_delta_norm
    self.assertLessEqual(lb_rel, 1e-3, f"engine and train.py weight the lb loss differently: {lb_rel}")

    norms = {label: self._norm(tree) for label, tree in on.items()}
    total_rel = self._norm(self._diff(on["engine_g2"], on["train_py_g2"])) / self._norm(on["train_py_g2"])
    # With the lb loss on, the engine weights it as train.py's accumulation path does.
    self.assertLessEqual(total_rel, 1e-4, f"engine and train.py weight the lb loss differently: {total_rel}")
    _measure(
        suite="moe_lb",
        router_params=len(on["engine_g2"]),
        total_weights=total_weights,
        grad_norm_engine_g2=f"{norms['engine_g2']:.4e}",
        grad_norm_train_py_g2=f"{norms['train_py_g2']:.4e}",
        lb_delta_norm_train_py_g2=f"{lb_delta_norm:.4e}",
        ratio_engine_over_train_py_g2=f"{norms['engine_g2'] / max(norms['train_py_g2'], 1e-30):.3f}",
        router_grad_rel_diff_off=f"{off_rel:.3e}",
        router_grad_rel_diff_engine_vs_train_py_g2=f"{total_rel:.3e}",
    )


def _main(argv):
  """Runs the standalone training engine test suites inside the child subprocess."""
  _names = argv[1:] or list(_SUITES)
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _loader = unittest.defaultTestLoader
  _runner = unittest.TextTestRunner(verbosity=2)
  ran = 0
  for name in _names:
    for case in _loader.loadTestsFromTestCase(globals()[name]):
      res = _runner.run(case)
      jax.clear_caches()
      if not res.wasSuccessful():
        sys.exit(1)
      ran += res.testsRun - len(res.skipped)
  print(f"{_SENTINEL} ran={ran}")


if __name__ == "__main__":
  if _IN_CHILD:
    app.run(_main, argv=sys.argv)
  else:
    absltest.main()
