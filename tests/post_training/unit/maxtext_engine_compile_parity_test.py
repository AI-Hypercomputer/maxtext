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

"""The abstract engine compiles Qwen3.5 behind the Tunix adapter exactly as the live engine runs it.

`maxtext_engine_xaot_test.py` compares `AbstractMaxTextEngine` with a live engine on MaxText's
default decoder, which does not exercise two behaviours needed for Qwen3.5 and Tunix:

1. `_trace_train_state` runs its graph-only `nnx.eval_shape` with no mesh in context. Under a
   mesh, flax re-derives every variable's sharding from its logical names, which raises on Qwen3.5
   under both rule sets used here: `DuplicateSpecError` under the default rules, where the scanned
   MLP weights put `fsdp_transpose` under two dimensions, and `Resource axis: norm ... is not
   found` under `cp-as-ep`, whose rules leave `norm` unmapped. The default decoder hits neither.
2. `wrap_with_tunix_adapter` / `tokenizer_pad_id` wrap the abstract model the way
   `model_creation_utils.from_pretrained` wraps the live one, so a Tunix-signature loss is compiled
   against the module it will run against. A wrong wrap would be silent: the tool would report
   memory for a different program.

So a tiny Qwen3.5 is built both ways, with the adapter and a Tunix-signature loss, under both rule
sets, and all three kernels are compared as StableHLO, as optimized HLO and by memory analysis.
Two more tests keep that comparison from passing vacuously: tracing the graph under the mesh must
still raise on these configurations, and an abstract pad id that differs from the live one must
change the HLO.

Four CPU devices, so the tests re-exec this module in a child with
`--xla_force_host_platform_device_count` set, as `maxtext_engine_xaot_test.py` does and for the same
reason: the CPU backend reads that flag only at initialization, which is past by the time pytest
imports this file.
"""

# The tests reach into the class under test: `_trace_under_mesh` uses its private helpers.
# pylint: disable=protected-access

import hashlib
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import maxtext_engine_compile
from maxtext.utils import maxtext_utils
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_COMPILE_PARITY_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")

# Not 0, so an abstract wrap that dropped the pad id or fell back to a default would be caught.
_PAD_ID = 3

# Each rule set, with the error that tracing the train-state graph under the mesh raises for it.
_RULE_SETS = (
    ("default_rules", {"ici_fsdp_parallelism": _REQUIRED_DEVICES}, r"duplicate entries for `fsdp_transpose`"),
    (
        "cp_as_ep",
        {"custom_mesh_and_rule": "cp-as-ep", "ici_fsdp_parallelism": 2, "ici_expert_parallelism": 2},
        r"Resource axis: norm of .* is not found in mesh",
    ),
)


@pytest.mark.cpu_only
def test_compile_parity_on_four_cpu_devices():
  """Runs `Qwen35CompileParityTest` in a child process with four CPU devices; see the module docstring."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  # An exit status of 0 is also what a run that skipped everything produces.
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


def _config(**overrides) -> pyconfig.HyperParameters:
  """A Qwen3.5 small enough to compile on CPU that keeps the variables a mesh-scoped trace fails on.

  Four layers is one `inhomogeneous_layer_cycle_interval`, so both the gated-delta-net and the
  full-attention layers are present; `scan_layers` is what puts `fsdp_transpose` under two
  dimensions of the scanned MLP weights. `sparse_matmul=False` keeps the MoE off megablox, which
  XLA:CPU can only interpret; it changes neither the variables nor the wrap under test.
  """
  argv = [
      "maxtext_engine_compile_parity_test.py",
      get_test_config_path("base.yml"),
      "model_name=qwen3.5-35b-a3b",
      "override_model_config=True",
      "run_name=engine_compile_parity_test",
      "enable_checkpointing=False",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
      "log_config=False",
      "init_weights_seed=0",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "scan_layers=True",
      "attention=dot_product",
      "sparse_matmul=False",
      "base_num_decoder_layers=4",
      "base_emb_dim=64",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      # The model's `mrope_section` needs `head_dim * partial_rotary_factor / 2 == 32`.
      "head_dim=256",
      "base_mlp_dim=64",
      "base_moe_mlp_dim=64",
      "num_experts=4",
      "num_experts_per_tok=2",
      "gdn_key_head_dim=16",
      "gdn_value_head_dim=16",
      "gdn_num_key_heads=2",
      "gdn_num_value_heads=4",
      "gdn_chunk_size=16",
      "vocab_size=128",
      "max_target_length=32",
      "per_device_batch_size=1",
      "profiler_steps=0",
  ]
  argv.extend(f"{key}={value}" for key, value in overrides.items())
  return pyconfig.initialize(argv)


def _tunix_loss(model, input_tokens, positions, targets, weights):
  """A loss with Tunix's model signature, which only the adapter-wrapped model accepts."""
  logits, _ = model(input_tokens, positions, None, None)
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
  weights = weights.astype(jnp.float32)
  return abstract_engine.WeightedMetric(unreduced_sum=-(target_log_probs * weights).sum(), denominator=weights.sum())


def _tunix_inputs(payload):
  return {
      "input_tokens": payload["inputs"],
      "positions": payload["inputs_position"],
      "targets": payload["targets"],
      "weights": payload["targets_segmentation"],
  }


def _trace_under_mesh(self, tx):
  """`AbstractMaxTextEngine._trace_train_state`, but with the graph trace inside `_sharding_ctx()`.

  The one difference from the real method is where `nnx.eval_shape(build, ...)` runs.
  """
  model_graphdef, model_pure = nnx.split(self._model)

  def build(model_state):
    model = nnx.merge(model_graphdef, model_state)
    return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

  propagation_mesh = maxtext_engine_compile._propagation_mesh(self._mesh)
  with self._sharding_ctx():
    state_graphdef, _ = nnx.split(nnx.eval_shape(build, model_pure))
    with jax.set_mesh(propagation_mesh):
      state_pure = jax.eval_shape(
          lambda model_state: nnx.split(build(model_state))[1],
          jax.tree.map(lambda aval: maxtext_engine_compile._rehome_aval(aval, propagation_mesh), model_pure),
      )
  return nnx.merge(
      state_graphdef, jax.tree.map(lambda aval: maxtext_engine_compile._rehome_aval(aval, self._mesh), state_pure)
  )


# StableHLO carries `loc(...)` source locations; optimized HLO carries a source-location index and
# `metadata={...}` / `stack_frame_id=N` pointing into it. Both name where the Python was, not what the
# program does. The optimized-HLO normalization matches `maxtext_engine_xaot_test._normalize`.
_LOCATION = re.compile(r"loc\([^\n]*?\)")
_SOURCE_ID_LINE = re.compile(r'^\s*\d+\s+(?:"[^"]*"|\{[^}]*\})\s*$')
_METADATA = re.compile(r"metadata=\{[^}]*\}")
_STACK_FRAME = re.compile(r"stack_frame_id=\d+")


def _stablehlo(lowered: jax.stages.Lowered) -> str:
  return _LOCATION.sub("", lowered.as_text())


def _optimized_hlo(compiled: jax.stages.Compiled) -> str:
  lines = []
  for line in compiled.as_text().splitlines():
    if _SOURCE_ID_LINE.match(line):
      continue
    lines.append(_STACK_FRAME.sub("stack_frame_id=0", _METADATA.sub("metadata={}", line)))
  return "\n".join(lines)


def _memory(compiled: jax.stages.Compiled) -> tuple[int, ...]:
  stats = compiled.memory_analysis()
  return tuple(
      getattr(stats, f"{prefix}{field}_size_in_bytes")
      for prefix in ("", "host_")
      for field in ("argument", "output", "alias", "temp")
  )


def _digest(text: str) -> str:
  return hashlib.sha256(text.encode()).hexdigest()


def _lower(engine: maxtext_engine.MaxTextTrainingEngine, cfg: pyconfig.HyperParameters) -> dict:
  """Every kernel lowered for the shaped micro-batch, through the Tunix-signature loss."""
  engine.with_loss_fn(_tunix_loss).with_gen_model_input_fn(_tunix_inputs)
  return engine._lower_kernels(maxtext_engine_compile.get_shaped_micro_batch(cfg))


class Qwen35CompileParityTest(parameterized.TestCase):
  """Qwen3.5 behind the Tunix adapter: the abstract engine against a live, randomly initialized one."""

  __test__ = False  # collected only via the subprocess entry point above.

  @parameterized.named_parameters(*((name, overrides) for name, overrides, _ in _RULE_SETS))
  def test_abstract_engine_matches_live_engine(self, overrides):
    cfg = _config(**overrides)
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    wrap = {"wrap_with_tunix_adapter": True, "tokenizer_pad_id": _PAD_ID}

    abstract = maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh, **wrap)
    live = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh, **wrap)

    # The wrap itself, stated directly: `from_pretrained` sets exactly these.
    for engine, label in ((abstract, "abstract"), (live, "live")):
      self.assertIsInstance(engine.model, TunixMaxTextAdapter, label)
      self.assertEqual(engine.model._pad_id, _PAD_ID, label)
      self.assertIsNone(engine.model.config, label)
    self.assertEqual(abstract.model.use_no_op_mappings, live.model.use_no_op_mappings)

    lowered = {"abstract": _lower(abstract, cfg), "live": _lower(live, cfg)}
    compiled = {}
    for label, engine in (("abstract", abstract), ("live", live)):
      with engine._sharding_ctx():
        compiled[label] = {name: lowered[label][name].compile() for name in maxtext_engine_compile.KERNEL_NAMES}

    for name in maxtext_engine_compile.KERNEL_NAMES:
      with self.subTest(kernel=name):
        self.assertEqual(
            _digest(_stablehlo(lowered["abstract"][name])),
            _digest(_stablehlo(lowered["live"][name])),
            f"{name}: the abstract engine lowers a different program",
        )
        self.assertEqual(
            _optimized_hlo(compiled["abstract"][name]),
            _optimized_hlo(compiled["live"][name]),
            f"{name}: the abstract engine compiles a different program",
        )
        self.assertEqual(_memory(compiled["abstract"][name]), _memory(compiled["live"][name]), name)

  @parameterized.named_parameters(*_RULE_SETS)
  def test_graph_trace_under_mesh_fails(self, overrides, error):
    """These configurations do not build if the graph is traced under the mesh, so they exercise the mesh-free trace."""
    cfg = _config(**overrides)
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    with mock.patch.object(maxtext_engine_compile.AbstractMaxTextEngine, "_trace_train_state", _trace_under_mesh):
      with self.assertRaisesRegex(Exception, error):
        maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=_PAD_ID)

  def test_pad_id_changes_hlo(self):
    """A pad id that differs between the engines changes the HLO, so the parity test can see the wrap.

    The adapter synthesizes segment ids from `input_tokens != pad_id` when Tunix passes none, so the
    pad id is a constant in both forward/backward kernels; the update never sees a token.
    """
    cfg = _config(**_RULE_SETS[0][1])
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    abstract = maxtext_engine_compile.AbstractMaxTextEngine(
        cfg, mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=_PAD_ID + 4
    )
    live = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=_PAD_ID)
    lowered_abstract, lowered_live = _lower(abstract, cfg), _lower(live, cfg)

    for name in ("fwd_bwd", "fwd_bwd_accum"):
      with self.subTest(kernel=name):
        self.assertNotEqual(
            _digest(_stablehlo(lowered_abstract[name])),
            _digest(_stablehlo(lowered_live[name])),
            f"{name}: a different pad id left the program unchanged, so the parity test cannot see the wrap",
        )


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _result = unittest.TextTestRunner(verbosity=2).run(
      unittest.defaultTestLoader.loadTestsFromTestCase(Qwen35CompileParityTest)
  )
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")
