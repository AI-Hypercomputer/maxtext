# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""query_proj_kernel_gather_once: one backward all-gather of the attention query kernel per layer.

The attention query (+ Qwen3.5 output gate) kernel is sharded on its embed dimension, so the partitioner all-gathers
it for every dot that reads it. Under remat the backward of a scanned layer reads it twice, in the replay of the
projection and in its input gradient, and the compiler may gather it twice. With the option on, the projection is a
custom VJP (layers/linears.py _dot_general_reusing_gathered_kernel) whose residual is the explicitly gathered kernel,
so the backward gathers it once.

  - Programs and values, in a child process with 8 CPU devices (see main): a stack of Qwen3NextFullAttention layers
    scanned under jax.checkpoint with the decoder's own remat policy, as nnx_decoders scans them, each followed by
    an RMS norm of the residual stream as in the decoder layer, on the 397B trainer's cp-as-ep rules (embed on fsdp x
    context x expert) and on the default rules. With the option: every dot that reads the query kernel (forward,
    replay, input gradient) reads the explicitly gathered kernel; the loss and every gradient are bit-identical to the
    option off; the backward loop body has exactly one all-gather of the query kernel; the output kernel's gathers are
    untouched; and no other collective changes (so the kernel gradient is still reduce-scattered, not all-reduced).
    The same program checks run for a TPU target (tpu7x-8, compiled on this host) when libtpu is installed; there the
    backward gathers the query kernel twice without the option, as in the 397B trainer step, and once with it.
  - In this process: the custom VJP gives the same bits as lax.dot_general; the attention layer passes the gathered
    axes to the query projection, for training only; DenseGeneral rejects what it cannot serve; the config validator
    rejects the option wherever it would not remove an all-gather, in MaxTextConfig and RLConfig.
"""

import collections
import functools
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import types

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import topologies
from jax.extend import core as jax_core
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, ShardMode
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import linears
from maxtext.layers import nnx_decoders
from maxtext.models import qwen3
from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path

_PASS_TOKEN = "QUERY_PROJ_KERNEL_GATHER_ONCE_CHECKS_PASSED"
_QUERY_KEY = "query_proj_kernel_gather_once"
_NUM_DEVICES = 8
_LAYERS = 3
# Distinct sizes, so each kernel's gathered shape is recognisable among the program's collectives: query kernel
# [256, 4, 2 * 32] (query + gate), output kernel [4 * 32, 256], activations [4, 64, 256] (batch 8 x 0.5 per device).
_EMB, _HEADS, _KV_HEADS, _HEAD_DIM, _SEQ = 256, 4, 2, 32, 64


def _bits(x):
  """The array's bytes, so equality means bit-identical (NaN-safe, -0.0 != 0.0)."""
  return np.asarray(jax.device_get(x)).tobytes()


def _cfg(**overrides):
  kwargs = {
      "run_name": "query_proj_kernel_gather_once",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


# A small Qwen3.5 attention layer in the trainer's setting: bf16 compute on fp32 weights, scanned layers,
# remat_policy=custom with the projections replayed and the layer input saved.
_SMALL_QWEN35 = {
    "decoder_block": "qwen3_5",
    "base_emb_dim": _EMB,
    "base_num_query_heads": _HEADS,
    "base_num_kv_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "base_mlp_dim": 256,
    "base_moe_mlp_dim": 256,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "base_num_decoder_layers": 4,
    "max_target_length": _SEQ,
    "per_device_batch_size": 0.5,
    "attention": "dot_product",
    "dtype": "bfloat16",
    "weight_dtype": "float32",
    "scan_layers": True,
    "remat_policy": "custom",
    "query_proj": "remat",
    "out_proj": "remat",
    "decoder_layer_input": "device",
    "context_parallel_load_balance": False,
}

# The 397B trainer's rules (embed_attn -> fsdp x context x expert), and the default rules (embed_attn -> fsdp x
# fsdp_transpose x context x ..., here fsdp 4 x context 2).
_MESHES = {
    "cp_as_ep": {
        "custom_mesh_and_rule": "cp-as-ep",
        "ici_fsdp_parallelism": 2,
        "ici_context_parallelism": 2,
        "ici_expert_parallelism": 2,
    },
    "default": {"ici_fsdp_parallelism": 4, "ici_context_parallelism": 2},
}


class CustomVjpTest(absltest.TestCase):
  """_dot_general_reusing_gathered_kernel is lax.dot_general, value and VJP, bit for bit."""

  def test_matches_dot_general(self):
    rng = np.random.default_rng(0)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
    gathered = NamedSharding(mesh, P())
    precision = jax.lax.Precision.DEFAULT
    cases = (
        ((2, 8, 16), (16, 4, 6), (((2,), (0,)), ((), ()))),  # query: [b, s, e] x [e, h, d]
        ((2, 8, 4, 6), (4, 6, 16), (((2, 3), (0, 1)), ((), ()))),  # output: [b, s, h, d] x [h, d, e]
    )
    for x_shape, w_shape, dims in cases:
      x = jnp.asarray(rng.standard_normal(x_shape), jnp.bfloat16)
      w = jnp.asarray(rng.standard_normal(w_shape), jnp.bfloat16)

      def ref(x, w, dims=dims):
        return jax.lax.dot_general(x, w, dims, precision=precision)

      def new(x, w, dims=dims):
        return linears._dot_general_reusing_gathered_kernel(x, w, dims, precision, gathered)  # pylint: disable=protected-access

      ref_out, ref_vjp = jax.vjp(ref, x, w)
      new_out, new_vjp = jax.vjp(new, x, w)
      g = jnp.asarray(rng.standard_normal(ref_out.shape), jnp.bfloat16)
      self.assertEqual(_bits(ref_out), _bits(new_out))
      for a, b in zip(ref_vjp(g), new_vjp(g)):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(_bits(a), _bits(b))
      self.assertEqual(_bits(jax.jit(ref)(x, w)), _bits(jax.jit(new)(x, w)))


class WiringTest(parameterized.TestCase):
  """The attention layer hands the query DenseGeneral the gathered axes exactly when the option is on, in training."""

  def _projections(self, model_mode, **overrides):
    cfg = _cfg(**{**_SMALL_QWEN35, "per_device_batch_size": 1.0, "max_prefill_predict_length": _SEQ // 2, **overrides})
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      layer = qwen3.Qwen3NextFullAttention(cfg, mesh, model_mode, layer_idx=3, rngs=nnx.Rngs(0))
    return {"query": layer.attention.query, "out": layer.attention.out}, mesh

  def test_train_on(self):
    projections, mesh = self._projections(MODEL_MODE_TRAIN, **{_QUERY_KEY: True})
    # Exactly the embed dimension is gathered: the query kernel's axes are (embed_attn, q_heads, kv).
    self.assertEqual(projections["query"].gathered_kernel_axes, (None, "q_heads", "kv"))
    self.assertIs(projections["query"].mesh, mesh)
    self.assertIsNone(projections["out"].gathered_kernel_axes)
    self.assertIsNone(projections["out"].mesh)

  @parameterized.named_parameters(("train_off", MODEL_MODE_TRAIN, False), ("prefill_on", MODEL_MODE_PREFILL, True))
  def test_off(self, model_mode, on):
    projections, _ = self._projections(model_mode, **{_QUERY_KEY: on})
    for projection in projections.values():
      self.assertIsNone(projection.gathered_kernel_axes)
      self.assertIsNone(projection.mesh)

  def test_dense_general_guard(self):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
    x = jnp.ones((2, 16), jnp.float32)
    common = {"kernel_axes": ("embed", "mlp"), "gathered_kernel_axes": (None, "mlp"), "rngs": nnx.Rngs(0)}
    for kwargs in ({"mesh": None}, {"mesh": mesh, "shard_mode": ShardMode.EXPLICIT}):
      dense = linears.DenseGeneral(16, 8, **common, **kwargs)
      with self.assertRaisesRegex(ValueError, "gathered_kernel_axes supports only"):
        dense(x)
    dense = linears.DenseGeneral(16, 8, mesh=mesh, **common)
    with self.assertRaisesRegex(ValueError, "gathered_kernel_axes supports only"):
      dense(x, slice_bounds=(0, 4))
    self.assertEqual(dense(x).shape, (2, 8))


def _validator_cfg(**overrides):
  """The validator's view of the 397B trainer cell (attribute names as in MaxTextConfig and RLConfig)."""
  rules = config_types.MaxTextConfig._load_mesh_config_from_yaml("cp-as-ep")["logical_axis_rules"]  # pylint: disable=protected-access
  fields = {
      _QUERY_KEY: False,
      "attention_type": "global",
      "fused_qkv": False,
      "quantization": config_types.QuantizationType.NONE,
      "shard_mode": ShardMode.AUTO,
      "ici_context_autoregressive_parallelism": 1,
      "logical_axis_rules": rules,
      "ici_fsdp_parallelism": 32,
      "ici_context_parallelism": 4,
      "ici_expert_parallelism": 2,
      "dcn_fsdp_parallelism": 1,
      "dcn_context_parallelism": 1,
      "dcn_expert_parallelism": 1,
      "remat_policy": "custom",
      "query_proj": "remat",
      "out_proj": "remat",
  }
  fields.update(overrides)
  return types.SimpleNamespace(**fields)


_UNSHARDED = {"ici_fsdp_parallelism": 1, "ici_context_parallelism": 1, "ici_expert_parallelism": 1}


def _tensor_parallel(*logical):
  """The trainer cell with `logical` axes mapped to 'tensor' and tensor parallelism 2."""
  rules = [r for r in _validator_cfg().logical_axis_rules if r[0] not in logical] + [[n, ["tensor"]] for n in logical]
  return {"logical_axis_rules": rules, "ici_tensor_parallelism": 2}


class ValidatorTest(parameterized.TestCase):
  """The validator rejects the option wherever it would not remove an all-gather from the backward."""

  def _validate(self, **overrides):
    config_types.MaxTextConfig.validate_query_proj_kernel_gather_once(_validator_cfg(**overrides))

  @parameterized.named_parameters(
      ("trainer_cell", {}),
      ("full", {"remat_policy": "full"}),
      ("save_out_proj", {"remat_policy": "save_out_proj"}),
      ("out_saved", {"out_proj": "device"}),
      ("fsdp_auto", {**_UNSHARDED, "ici_fsdp_parallelism": -1}),
      ("dcn_fsdp", {**_UNSHARDED, "dcn_fsdp_parallelism": 2}),
      ("tensor_on_mlp_only", _tensor_parallel("mlp")),
  )
  def test_accepted(self, overrides):
    self._validate(**{_QUERY_KEY: True, **overrides})

  @parameterized.named_parameters(
      ("mla", {"attention_type": "mla"}, "builds its own query projections"),
      ("compressed", {"attention_type": "compressed"}, "builds its own query projections"),
      ("fused_qkv", {"fused_qkv": True}, "requires fused_qkv=False"),
      ("int8", {"quantization": config_types.QuantizationType.INT8}, "requires quantization=''"),
      ("fp8", {"quantization": config_types.QuantizationType.FP8}, "requires quantization=''"),
      ("explicit", {"shard_mode": ShardMode.EXPLICIT}, "requires shard_mode='auto'"),
      ("autoregressive", {"ici_context_autoregressive_parallelism": 2}, "the query kernel is replicated"),
      ("unsharded", _UNSHARDED, "none of which has a parallelism other than 1"),
      ("tensor_on_q_heads", _tensor_parallel("q_heads"), "requires the kernel's other dimensions to be replicated"),
      ("tensor_on_kv", _tensor_parallel("kv"), "requires the kernel's other dimensions to be replicated"),
      ("minimal", {"remat_policy": "minimal"}, "requires a remat policy that replays the query projection"),
      ("save_qkv_proj", {"remat_policy": "save_qkv_proj"}, "requires a remat policy that replays"),
      ("qkv_proj_offloaded", {"remat_policy": "qkv_proj_offloaded"}, "requires a remat policy that replays"),
      ("save_dot_except_mlp", {"remat_policy": "save_dot_except_mlp"}, "requires a remat policy that replays"),
      ("none", {"remat_policy": "none"}, "requires a remat policy that replays"),
      ("custom_saved", {"query_proj": "device"}, "requires a remat policy that replays"),
      ("custom_offloaded", {"query_proj": "offload"}, "requires a remat policy that replays"),
  )
  def test_rejected(self, overrides, message):
    with self.assertRaisesRegex(ValueError, f"{_QUERY_KEY}=True .*{message}"):
      self._validate(**{_QUERY_KEY: True, **overrides})
    # The same config with the option off passes.
    self._validate(**overrides)

  def test_pyconfig(self):
    self.assertFalse(_cfg().query_proj_kernel_gather_once)
    self.assertFalse(config_types.Attention().query_proj_kernel_gather_once)
    small = {**_SMALL_QWEN35, **_MESHES["cp_as_ep"]}
    self.assertTrue(_cfg(**small, **{_QUERY_KEY: True}).query_proj_kernel_gather_once)
    with self.assertRaisesRegex(ValueError, "requires a remat policy that replays the query projection"):
      _cfg(**{**small, _QUERY_KEY: True, "query_proj": "device"})
    with self.assertRaisesRegex(ValueError, "requires fused_qkv=False"):
      _cfg(**{**small, _QUERY_KEY: True, "fused_qkv": True})
    tensor_parallel = {
        **_SMALL_QWEN35,
        "ici_fsdp_parallelism": 2,
        "ici_context_parallelism": 2,
        "ici_tensor_parallelism": 2,
    }
    with self.assertRaisesRegex(ValueError, "requires the kernel's other dimensions to be replicated"):
      _cfg(**{**tensor_parallel, _QUERY_KEY: True})

  def test_rl_config_runs_the_validator(self):
    # RLConfig does not inherit MaxTextConfig's validators; its own validator must call the shared one.
    source = inspect.getsource(config_types.RLConfig.set_derived_values_and_validate)
    self.assertIn("MaxTextConfig.validate_query_proj_kernel_gather_once(self)", source)
    source = inspect.getsource(config_types.MaxTextConfig.set_derived_and_validate_values)
    self.assertIn("self.validate_query_proj_kernel_gather_once(self)", source)


def _child_env(targets):
  """The environment of a `main` child process compiling for `targets` (comma-separated: cpu, tpu7x-8)."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_NUM_DEVICES}".strip()
  # A TPU target needs the TPU plugin registered (and skipped as a backend, as there is no TPU here), possibly while
  # this process also has libtpu loaded.
  env["JAX_PLATFORMS"] = "cpu" if targets == "cpu" else ""
  if targets != "cpu":
    env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
  env["QPGO_TARGETS"] = targets
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  return env


def _run_child(targets):
  result = subprocess.run(
      [sys.executable, __file__], env=_child_env(targets), capture_output=True, text=True, check=False
  )
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.cpu_only
def test_programs_in_a_child_process():
  """Runs `main` below in a fresh process with 8 CPU devices: values and the CPU compiler's programs."""
  _run_child("cpu")


@pytest.mark.cpu_only
@pytest.mark.skipif(importlib.util.find_spec("libtpu") is None, reason="needs libtpu to compile for a TPU target")
def test_tpu_programs_in_a_child_process():
  """The same program checks, compiled for a tpu7x-8 target on this host (no TPU used)."""
  _run_child("tpu7x-8")


# ----------------------------------------------------------------------------------------------------------------------
# Program checks (main).

_COMPUTATION_RE = re.compile(r"^(?:ENTRY )?%?([^\s(]+) .*\{\s*$")
_CALLEE_RE = re.compile(r"(?:body|condition|to_apply|calls|branch_computations)=\{?([^,}\s]+(?:, *%[^,}\s]+)*)\}?")
_COLLECTIVE_RE = re.compile(
    r"=\s*(\([^=]*?\)|\S+)\s+(all-gather|all-gather-start|reduce-scatter|all-reduce|all-reduce-start|all-to-all|"
    r"collective-permute|collective-permute-start)\("
)


def _group_size(line):
  """Devices per replica group, from any of the HLO printer's three notations (iota, mesh, explicit)."""
  iota = re.search(r"replica_groups=\[(\d+),(\d+)\]<=", line)
  if iota:
    return int(iota.group(2))
  mesh = re.search(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}", line)
  if mesh:
    sizes = dict(re.findall(r"'([^']+)'=(\d+)", mesh.group(1)))
    return int(np.prod([int(sizes[a]) for a in re.findall(r"'([^']+)'", mesh.group(2))]))
  explicit = re.search(r"replica_groups=\{\{([^}]*)\}", line)
  return len(explicit.group(1).split(",")) if explicit else 0


def _computations(hlo):
  """{computation name: [instruction lines]} of an HLO module's text."""
  comps, name = {}, None
  for line in hlo.splitlines():
    if not line.startswith(" "):
      m = _COMPUTATION_RE.match(line)
      name = m.group(1).lstrip("%") if m else None
      if name:
        comps[name] = []
    elif name:
      comps[name].append(line)
  return comps


def _while_bodies(comps):
  """{while body name: set of computations it reaches (itself included)}."""
  callees = {
      name: {c.strip().lstrip("%") for line in lines for group in _CALLEE_RE.findall(line) for c in group.split(",")}
      for name, lines in comps.items()
  }
  bodies = {
      m.group(1).lstrip("%")
      for lines in comps.values()
      for line in lines
      if " while(" in line
      for m in [re.search(r"body=(%?[^,\s]+)", line)]
      if m
  }

  def reach(name, seen):
    if name in seen or name not in comps:
      return seen
    seen.add(name)
    for c in callees[name]:
      reach(c, seen)
    return seen

  return {b: reach(b, set()) for b in bodies}


def _collectives(lines):
  """Counter of (opcode, result dimensions, devices per group) of the collectives in `lines`.

  Element types and layouts are left out: the CPU compiler moves bfloat16 collectives to float32, and the same
  replica groups print in different notations depending on how the sharding was introduced.
  """
  out = collections.Counter()
  for line in lines:
    m = _COLLECTIVE_RE.search(line)
    if not m:
      continue
    dims = tuple(re.findall(r"[a-z0-9]+\[([0-9,]*)\]", m.group(1)))
    out[(m.group(2).replace("-start", ""), dims, _group_size(line))] += 1
  return out


@functools.cache
def _target_devices(target):
  """The devices to compile for: this process's CPU devices, or a TPU topology (compiled for, not run, on this host).

  A TPU topology must be built before anything initializes the local backends (main builds them first).
  """
  if target == "cpu":
    return jax.devices("cpu")
  spec = accelerator_to_spec_map.get_system_characteristics(target)
  return topologies.get_topology_desc(
      platform=spec.platform,
      topology_name=spec.topology_name,
      chip_config_name=spec.chip_config_name,
      chips_per_host_bounds=spec.chips_per_host_bounds,
      num_slices=1,
      wrap=spec.wrap,
  ).devices


def _run(mesh_name, options, target):
  """A scanned, rematerialized stack of Qwen3.5 attention layers: (compiled HLO text, values or None, cfg, mesh, jaxpr).

  The layers are scanned as nnx_decoders scans them (jax.checkpoint with the decoder's remat policy around each
  layer, the layer input saved under its checkpoint name). On CPU the step also runs: values = (loss, parameter
  gradients, input gradient). For a TPU target it is only compiled.
  """
  cfg = _cfg(**{**_SMALL_QWEN35, **_MESHES[mesh_name], **options})
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg, _target_devices(target)), cfg.mesh_axes)
  rules = cfg.logical_axis_rules
  with nn_partitioning.axis_rules(rules):
    layers = [
        qwen3.Qwen3NextFullAttention(cfg, mesh, MODEL_MODE_TRAIN, layer_idx=3, rngs=nnx.Rngs(params=i, dropout=i))
        for i in range(_LAYERS)
    ]
  graphdef, template, rest = nnx.split(layers[0], nnx.Param, ...)
  stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[nnx.to_pure_dict(nnx.split(l, nnx.Param, ...)[1]) for l in layers])
  logical = nnx.to_pure_dict(nnx.get_partition_spec(template))
  param_shardings = jax.tree.map(
      lambda spec: NamedSharding(mesh, P(None, *sharding.logical_to_mesh_axes(spec, mesh, rules))),
      logical,
      is_leaf=lambda s: isinstance(s, P),
  )
  batch, seq = int(cfg.global_batch_size_to_train_on), cfg.max_target_length
  x_sharding = sharding.create_sharding(mesh, ("activation_batch", "activation_length", "activation_embed"), rules)
  x = jnp.asarray(np.random.default_rng(7).standard_normal((batch, seq, _EMB)), jnp.bfloat16)
  segment_ids = jnp.ones((batch, seq), jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (batch, seq))
  policy = nnx_decoders.NNXDecoder.get_remat_policy(types.SimpleNamespace(config=cfg))

  def loss_fn(params, x):
    def layer_fn(carry, layer_params):
      carry = checkpoint_name(carry, "decoder_layer_input")
      state = jax.tree.map(lambda v: v, template)
      nnx.replace_by_pure_dict(state, layer_params)
      out, _ = nnx.merge(graphdef, state, rest)(carry, segment_ids, positions, True, MODEL_MODE_TRAIN)
      # A parameter-free RMS norm of the residual stream, as the decoder layer's post-attention norm: its backward needs
      # the attention output, so the backward replays the output projection too.
      y = (carry + out).astype(jnp.float32)
      y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + 1e-6)
      return y.astype(carry.dtype), None

    y, _ = jax.lax.scan(jax.checkpoint(layer_fn, policy=policy, prevent_cse=False), x, params)
    return jnp.mean(jnp.square(y.astype(jnp.float32)))

  step = jax.jit(
      jax.value_and_grad(loss_fn, argnums=(0, 1)),
      in_shardings=(param_shardings, x_sharding),
      out_shardings=(NamedSharding(mesh, P()), (param_shardings, x_sharding)),
  )
  abstract = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), (stacked, x), (param_shardings, x_sharding)
  )
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
    jaxpr = jax.make_jaxpr(jax.value_and_grad(loss_fn, argnums=(0, 1)))(*abstract).jaxpr
    compiled = step.lower(*abstract).compile()
  values = None
  if target == "cpu":
    loss, (grads, x_grad) = compiled(jax.device_put(stacked, param_shardings), jax.device_put(x, x_sharding))
    values = (loss, grads, x_grad)
  return compiled.as_text(), values, cfg, mesh, jaxpr


def _sub_jaxprs(eqn):
  for value in eqn.params.values():
    for item in value if isinstance(value, (tuple, list)) else (value,):
      if isinstance(item, jax_core.ClosedJaxpr):
        yield item.jaxpr
      elif isinstance(item, jax_core.Jaxpr):
        yield item


def _kernel_reads(jaxpr, kernel_shape):
  """[(dot output shape, source)] for every dot_general anywhere in the program that reads a `kernel_shape` operand.

  source is ("gathered", spec) when the operand is the output of a sharding constraint, else the name of whatever
  produced it ("input" for a program input). Values are followed through scan, remat and call boundaries: into a
  sub-jaxpr whose inputs are the eqn's operands one to one, and out of one whose results are the eqn's results.
  """
  producers = {}

  def producer(j, var):
    if id(j) not in producers:
      producers[id(j)] = {v: (eqn, i) for eqn in j.eqns for i, v in enumerate(eqn.outvars)}
    return producers[id(j)].get(var)

  def source(j, var, env):  # env: {input of j: (outer jaxpr, outer var, outer env)}, resolved lazily.
    if not isinstance(var, jax_core.Var):
      return "literal"
    found = producer(j, var)
    if found is None:
      return source(*env[var]) if var in env else "input"
    eqn, index = found
    if eqn.primitive.name == "sharding_constraint":
      return ("gathered", tuple(eqn.params["sharding"].spec))
    subs = [k for k in _sub_jaxprs(eqn) if len(k.invars) == len(eqn.invars) and len(k.outvars) == len(eqn.outvars)]
    if len(subs) == 1:
      return source(subs[0], subs[0].outvars[index], {iv: (j, ov, env) for iv, ov in zip(subs[0].invars, eqn.invars)})
    return eqn.primitive.name

  reads = []

  def walk(j, env):
    for eqn in j.eqns:
      if eqn.primitive.name == "dot_general":
        for operand in eqn.invars:
          if tuple(getattr(operand.aval, "shape", ())) == kernel_shape:
            reads.append((tuple(eqn.outvars[0].aval.shape), source(j, operand, env)))
      for sub in _sub_jaxprs(eqn):
        walk(
            sub,
            {iv: (j, ov, env) for iv, ov in zip(sub.invars, eqn.invars)} if len(sub.invars) == len(eqn.invars) else {},
        )

  walk(jaxpr, {})
  return reads


def _strip_spec(spec):
  spec = list(spec)
  while spec and spec[-1] is None:
    spec.pop()
  return tuple(spec)


def _check(name, good, detail=""):
  print(f"{'PASS' if good else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
  return good


def _gathered_dims(cfg, mesh):
  """{kernel: its gathered per-device dimensions}: the embed dimension whole, the rest sharded as the rules say."""

  def shards(logical):
    axes = sharding.logical_to_mesh_axes((logical,), mesh, cfg.logical_axis_rules)[0]
    return 1 if axes is None else int(np.prod([mesh.shape[a] for a in np.atleast_1d(axes)]))

  return {
      "query": f"{_EMB},{_HEADS // shards('q_heads')},{2 * _HEAD_DIM}",
      "out": f"{_HEADS * _HEAD_DIM // shards('mlp')},{_EMB}",
  }


def _is_gather_of(op, dims, gathered):
  """Whether a collective produces a gathered kernel (the TPU compiler may keep the scan's leading 1)."""
  return op == "all-gather" and any(d in (gathered, "1," + gathered) for d in dims)


def _program(hlo, gathered):
  """({kernel: sorted per-loop-body gather counts}, Counter of every collective in the module)."""
  comps = _computations(hlo)
  bodies = _while_bodies(comps)
  per_body = [_collectives([l for c in reach for l in comps[c]]) for reach in bodies.values()]
  counts = {
      kernel: sorted(sum(n for (op, d, _), n in body.items() if _is_gather_of(op, d, dims)) for body in per_body)
      for kernel, dims in gathered.items()
  }
  return counts, _collectives([l for lines in comps.values() for l in lines])


# The kernels' global shapes in the jaxpr, and the logical axes the attention layer gathers them to.
_KERNEL_SHAPES = {"query": (_EMB, _HEADS, 2 * _HEAD_DIM), "out": (_HEADS * _HEAD_DIM, _EMB)}
_GATHERED_AXES = {"query": (None, "q_heads", "kv"), "out": ("mlp", None)}
# The projections' input shapes, the result shape of their input-gradient dots (batch 8 devices x 0.5 = 4).
_INPUT_SHAPES = {"query": (4, _SEQ, _EMB), "out": (4, _SEQ, _HEADS * _HEAD_DIM)}

# Per (target, mesh): the query kernel's all-gathers in the [forward, backward] loop bodies (sorted), option off and on.
# Whether the compiler already shares one gather between the two backward dots depends on the compiler and the program;
# with the option the backward has exactly one gather of the kernel and no other collective changes.
_EXPECTED = {
    ("cpu", "cp_as_ep"): ([1, 1], [1, 1]),
    ("cpu", "default"): ([1, 1], [1, 1]),
    ("tpu7x-8", "cp_as_ep"): ([1, 2], [1, 1]),
    ("tpu7x-8", "default"): ([1, 2], [1, 1]),
}


def _check_mesh(mesh_name, target):
  """Compiles (and on CPU runs) the stack with the option off and on, and checks values and programs."""
  ok = True
  runs = {"off": _run(mesh_name, {}, target), "on": _run(mesh_name, {_QUERY_KEY: True}, target)}
  _, off_values, cfg, mesh, _ = runs["off"]
  gathered = _gathered_dims(cfg, mesh)
  programs = {arm: _program(run[0], gathered) for arm, run in runs.items()}
  if os.environ.get("QPGO_DUMP_DIR"):
    for arm, run in runs.items():
      path = os.path.join(os.environ["QPGO_DUMP_DIR"], f"{target}.{mesh_name}.{arm}.hlo.txt")
      with open(path, "w", encoding="utf-8") as f:
        f.write(run[0])
  name = f"{target} {mesh_name}"
  on_values = runs["on"][1]
  if off_values is not None:
    # Values: the loss and every gradient, bit for bit.
    (off_loss, off_grads, off_xg), (on_loss, on_grads, on_xg) = off_values, on_values
    flat_off = jax.tree_util.tree_leaves_with_path(off_grads)
    flat_on = jax.tree_util.tree_leaves_with_path(on_grads)
    differing = [jax.tree_util.keystr(p) for (p, a), (_, b) in zip(flat_off, flat_on) if _bits(a) != _bits(b)]
    same = _bits(off_loss) == _bits(on_loss) and _bits(off_xg) == _bits(on_xg) and len(flat_off) == len(flat_on)
    ok &= _check(
        f"{name}: loss, input gradient and {len(flat_off)} parameter gradients bit-identical",
        same and not differing,
        f"loss {float(off_loss)!r} vs {float(on_loss)!r}, differing {differing}",
    )
  # The program handed to the compiler: with the option every dot that reads the query kernel (forward, replay and
  # input gradient) reads the explicitly gathered kernel; without it none does. This holds whichever compiler runs it.
  shape = _KERNEL_SHAPES["query"]
  want_spec = _strip_spec(sharding.create_sharding(mesh, _GATHERED_AXES["query"], cfg.logical_axis_rules).spec)
  on_reads, off_reads = _kernel_reads(runs["on"][4], shape), _kernel_reads(runs["off"][4], shape)
  good_on = [r for r in on_reads if isinstance(r[1], tuple) and _strip_spec(r[1][1]) == want_spec]
  gathered_off = [r for r in off_reads if isinstance(r[1], tuple)]
  ok &= _check(
      f"{name}: all {len(on_reads)} dots reading the query kernel {list(shape)} read it gathered to {want_spec};"
      f" off: {len(gathered_off)} of {len(off_reads)}",
      len(on_reads) == len(off_reads) >= 3
      and len(good_on) == len(on_reads)
      and _INPUT_SHAPES["query"] in {r[0] for r in good_on}
      and not gathered_off,
      f"on {on_reads}, off {off_reads}",
  )
  (off_counts, off_every), (on_counts, on_every) = programs["off"], programs["on"]
  want_off, want_on = _EXPECTED[(target, mesh_name)]
  ok &= _check(
      f"{name}: query kernel [{gathered['query']}] all-gathers per loop body, off {off_counts['query']} -> on"
      f" {on_counts['query']}",
      off_counts["query"] == want_off and on_counts["query"] == want_on,
      f"want {want_off} -> {want_on}",
  )
  ok &= _check(
      f"{name}: output kernel [{gathered['out']}] all-gathers unchanged",
      on_counts["out"] == off_counts["out"],
      f"{off_counts['out']} -> {on_counts['out']}",
  )
  # No other collective changes (the kernel gradient is still reduce-scattered, not all-reduced).
  removed, added = off_every - on_every, on_every - off_every
  want_removed = sum(want_off) - sum(want_on)
  ok &= _check(
      f"{name}: removes {want_removed} query kernel all-gather(s) and changes no other collective",
      sum(removed.values()) == want_removed
      and all(_is_gather_of(op, d, gathered["query"]) for op, d, _ in removed)
      and not added,
      f"removed {dict(removed)}, added {dict(added)}",
  )
  return ok


def main():
  targets = os.environ.get("QPGO_TARGETS", "cpu").split(",")
  # TPU topologies first: building one after the local backends are initialized fails.
  for target in sorted(targets, key=lambda t: t == "cpu"):
    assert len(_target_devices(target)) == _NUM_DEVICES, (target, _target_devices(target))
  ok = True
  for target in targets:
    for mesh_name in _MESHES:
      ok &= _check_mesh(mesh_name, target)
  if not ok:
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
