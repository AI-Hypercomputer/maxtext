# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks `context_parallel_attention_load_balance_order` on a CPU mesh.

`dual_chunk_swap_local_self` keeps the DUAL_CHUNK_SWAP reorder of `context_parallel_attention_load_balance`
but drops the (r, r) pairs from its two collective-permutes; the rank that keeps its own half takes it with
a select. The device checks run in a subprocess, so the forced CPU device count takes effect before JAX
initializes (as in attention_load_balance_test.py):

  - Reorder. At context 2, 4 and 8, both directions and their VJPs are bit-identical to the
    dual_chunk_swap order, for float32, bfloat16 and int32 inputs.
  - Program. The compiled reorder keeps the dual_chunk_swap program's source-target pairs minus every
    self pair: no collective-permute of it has a self pair, and at context 2, whose first-half permutation
    is all self pairs, that permute is gone. The dual_chunk_swap program has exactly the self pairs
    `load_balance_self_ranks` names, so the pair parser is checked against a program that keeps them.
  - Model. A tiny hybrid Qwen3.5 with the attention load balance on: the key reaches the reorder (the
    compiled program has fewer self pairs) and the logits and every parameter gradient are bit-identical.

Each device check has a positive control, a reorder whose self rank keeps the zeros the permute sends it,
which must fail the same comparison; otherwise a comparison that cannot fail would pass.
"""

import contextlib
import functools
import os
import re
import subprocess
import sys
import unittest.mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pydantic
import pytest

from maxtext.configs import pyconfig
from maxtext.kernels.attention import tokamax_ring_attention
from maxtext.utils import globals as maxtext_globals

_BASE_CONFIG_PATH = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
_KEY = "context_parallel_attention_load_balance_order"
_DCS = tokamax_ring_attention.LOAD_BALANCE_ORDER_DUAL_CHUNK_SWAP
_LOCAL_SELF = tokamax_ring_attention.LOAD_BALANCE_ORDER_DUAL_CHUNK_SWAP_LOCAL_SELF
_CHILD_MARKER = "ATTENTION_LOAD_BALANCE_ORDER_CHECKS_PASSED"


def _bits(x) -> tuple:
  """Shape, dtype and raw bytes of an array: equal tuples mean bit-identical arrays."""
  x = np.ascontiguousarray(np.asarray(x))
  return x.shape, str(x.dtype), x.tobytes()


def _permute_pairs(hlo_text: str) -> list[list[tuple[int, int]]]:
  """The source-target pairs of every collective-permute in a compiled HLO module, one sorted list per op."""
  ops = []
  for line in hlo_text.splitlines():
    if not re.search(r"=.*\bcollective-permute(?:-start)?\(", line):
      continue
    match = re.search(r"source_target_pairs=\{((?:\{\d+,\d+\},?)*)\}", line)
    assert match, f"collective-permute without source_target_pairs: {line[:200]}"
    ops.append(sorted((int(s), int(d)) for s, d in re.findall(r"\{(\d+),(\d+)\}", match.group(1))))
  return ops


def _self_pairs(ops: list[list[tuple[int, int]]]) -> int:
  return sum(1 for op in ops for src, dst in op if src == dst)


@contextlib.contextmanager
def _self_rank_keeps_zeros():
  """A deliberately wrong local_self reorder: the self rank keeps what the permute sends it (zeros)."""

  def broken(x, axis_name, perm):
    remote = tuple((src, dst) for src, dst in perm if src != dst)
    return lax.ppermute(x, axis_name, perm=remote) if remote else jnp.zeros_like(x)

  with unittest.mock.patch.object(tokamax_ring_attention, "_ppermute_local_self", broken):
    yield


# ---------------------------------------------------------------------------
# 1. The reorder and its program.
# ---------------------------------------------------------------------------


def _reorder_fn(mesh, batch_axes, order, to_natural=False):
  return functools.partial(
      tokamax_ring_attention.reorder_for_load_balance,
      mesh=mesh,
      axis_name="context",
      batch_axes=batch_axes,
      to_natural=to_natural,
      order=order,
  )


def _reorder_outputs(mesh, batch_axes, order, cases, cotangent):
  """Both directions of the reorder and both VJPs, for every input."""
  out = {}
  with jax.set_mesh(mesh):
    for name, x in cases.items():
      for to_natural in (False, True):
        out[(name, to_natural)] = _bits(jax.jit(_reorder_fn(mesh, batch_axes, order, to_natural))(x))
    for to_natural in (False, True):
      fn = _reorder_fn(mesh, batch_axes, order, to_natural)
      grad = jax.jit(jax.grad(lambda x, fn=fn: jnp.sum(jnp.asarray(cotangent) * fn(x))))(cases["float32"])
      out[("vjp", to_natural)] = _bits(grad)
  return out


def _run_reorder_checks(mesh, batch_axes):
  """local_self == dual_chunk_swap bit for bit, without self pairs in its permutes."""
  cp = mesh.shape["context"]
  seq = 8 * cp
  rng = np.random.default_rng(0)
  cases = {
      "float32": rng.standard_normal((4, seq, 8)).astype(np.float32),
      "bfloat16": jnp.asarray(rng.standard_normal((4, seq, 8)), jnp.bfloat16),
      "positions": np.broadcast_to(np.arange(seq, dtype=np.int32), (4, seq)).copy(),
      "batch_of_one": rng.standard_normal((1, seq, 8)).astype(np.float32),
  }
  cotangent = rng.standard_normal(cases["float32"].shape).astype(np.float32)
  dcs = _reorder_outputs(mesh, batch_axes, _DCS, cases, cotangent)
  local_self = _reorder_outputs(mesh, batch_axes, _LOCAL_SELF, cases, cotangent)
  differ = sorted(str(k) for k, v in dcs.items() if v != local_self[k])
  assert not differ, f"context={cp}: local_self differs from dual_chunk_swap in {differ}"

  # Program: the dual_chunk_swap pairs minus the self pairs.
  first_self, second_self = tokamax_ring_attention.load_balance_self_ranks(cp)
  groups = mesh.devices.size // cp
  sharded = jax.device_put(cases["float32"], NamedSharding(mesh, P(batch_axes, "context", None)))
  for to_natural in (False, True):
    with jax.set_mesh(mesh):
      dcs_ops, ls_ops = (
          _permute_pairs(jax.jit(_reorder_fn(mesh, batch_axes, order, to_natural)).lower(sharded).compile().as_text())
          for order in (_DCS, _LOCAL_SELF)
      )
    assert len(dcs_ops) == 2, f"context={cp}: dual_chunk_swap lowers to {len(dcs_ops)} collective-permutes"
    assert _self_pairs(dcs_ops) == groups * (len(first_self) + len(second_self)), (
        f"context={cp}: dual_chunk_swap has {_self_pairs(dcs_ops)} self pairs, expected"
        f" {groups} x ({first_self}, {second_self})"
    )
    expected = sorted(kept for kept in ([p for p in op if p[0] != p[1]] for op in dcs_ops) if kept)
    assert sorted(ls_ops) == expected, f"context={cp} to_natural={to_natural}: local_self pairs {ls_ops} != {expected}"
    assert _self_pairs(ls_ops) == 0
  print(
      f"reorder checks passed: mesh={dict(mesh.shape)} self ranks {first_self} {second_self},"
      f" local_self permutes {len(ls_ops)}",
      flush=True,
  )

  # Positive control: the self rank keeping the zeros it receives must change the output.
  if first_self or second_self:
    with _self_rank_keeps_zeros():
      broken = _reorder_outputs(mesh, batch_axes, _LOCAL_SELF, cases, cotangent)
    assert any(broken[k] != v for k, v in dcs.items()), f"context={cp}: positive control went undetected"


# ---------------------------------------------------------------------------
# 2. A tiny hybrid Qwen3.5, end to end.
# ---------------------------------------------------------------------------


def _tiny_qwen3_5_config(order: str, seq: int):
  """A tiny hybrid Qwen3.5 on 8 CPU devices: context=4 x expert=2 under cp-as-ep, attention load balance on."""
  argv = [
      None,
      _BASE_CONFIG_PATH,
      "run_name=attention_load_balance_order_test",
      "model_name=qwen3.5-397b-a17b",
      "override_model_config=true",
      "hardware=cpu",
      "custom_mesh_and_rule=cp-as-ep",
      "ici_fsdp_parallelism=1",
      "ici_context_parallelism=4",
      "ici_expert_parallelism=2",
      "ici_tensor_parallelism=1",
      "context_parallel_load_balance=False",
      "context_parallel_attention_load_balance=True",
      f"{_KEY}={order}",
      "attention=dot_product",
      "use_gdn_kernel=false",
      "sparse_matmul=false",
      "megablox=false",
      "use_ring_of_experts=false",
      "dtype=float32",
      "weight_dtype=float32",
      "matmul_precision=highest",
      "scan_layers=True",
      "enable_checkpointing=false",
      "enable_dropout=false",
      "dataset_type=synthetic",
      "base_num_decoder_layers=4",  # one block of [3 GatedDeltaNet + 1 full attention]
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=64",
      "mrope_section=[3,3,2]",
      "base_mlp_dim=32",
      "base_moe_mlp_dim=32",
      "num_experts=8",
      "num_experts_per_tok=2",
      "shared_experts=1",
      "gdn_key_head_dim=16",
      "gdn_value_head_dim=16",
      "gdn_num_key_heads=2",
      "gdn_num_value_heads=4",
      "gdn_chunk_size=16",
      "vocab_size=384",
      f"max_target_length={seq}",
      "per_device_batch_size=0.5",
      "use_multimodal=false",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
  ]
  return pyconfig.initialize(argv)


def _model_outputs_and_grads(cfg, data, cotangent, patch=None):
  """Logits, the parameter gradients of <logits, cotangent>, and the compiled HLO of one tiny model."""
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.utils import maxtext_utils, maxtext_utils_nnx, model_creation_utils

  mesh = maxtext_utils.get_mesh_from_config(cfg)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=jax.random.PRNGKey(0))
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def logits_of(params):
    merged = nnx.merge(graphdef, params, rest, copy=True)
    logits = merged(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        enable_dropout=False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    return logits.astype(jnp.float32)

  def value_and_grads(params):
    logits, vjp = jax.vjp(logits_of, params)
    return logits, vjp(cotangent)[0]

  with patch or contextlib.nullcontext(), nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    compiled = jax.jit(value_and_grads).lower(params).compile()
    logits, grads = compiled(params)
  leaves = {jax.tree_util.keystr(path): _bits(leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}
  return _bits(logits), leaves, compiled.as_text()


def _run_model_checks(seq: int):
  """Tiny hybrid Qwen3.5: local_self == dual_chunk_swap bit for bit in the logits and every gradient."""
  dcs_cfg = _tiny_qwen3_5_config(_DCS, seq)
  ls_cfg = _tiny_qwen3_5_config(_LOCAL_SELF, seq)
  assert getattr(ls_cfg, _KEY) == _LOCAL_SELF and getattr(dcs_cfg, _KEY) == _DCS
  batch = dcs_cfg.micro_batch_size_to_train_on
  tokens = jax.random.randint(jax.random.PRNGKey(7), (batch, seq + 1), 0, dcs_cfg.vocab_size)
  # Two packed segments per row so segment ids and position resets travel through the reorder.
  boundary = (seq * 5) // 8
  segmentation = jnp.where(jnp.arange(seq) < boundary, 1, 2).astype(jnp.int32)
  positions = jnp.where(jnp.arange(seq) < boundary, jnp.arange(seq), jnp.arange(seq) - boundary).astype(jnp.int32)
  data = {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "inputs_position": jnp.broadcast_to(positions, (batch, seq)),
      "inputs_segmentation": jnp.broadcast_to(segmentation, (batch, seq)),
      "targets_segmentation": jnp.broadcast_to(segmentation, (batch, seq)),
  }
  cotangent = jax.random.normal(jax.random.PRNGKey(3), (batch, seq, dcs_cfg.vocab_size), jnp.float32)
  dcs_logits, dcs_grads, dcs_hlo = _model_outputs_and_grads(dcs_cfg, data, cotangent)
  ls_logits, ls_grads, ls_hlo = _model_outputs_and_grads(ls_cfg, data, cotangent)

  # The key must reach the reorder: the local_self program has fewer self pairs.
  dcs_self, ls_self = (_self_pairs(_permute_pairs(h)) for h in (dcs_hlo, ls_hlo))
  assert ls_self < dcs_self, f"the key had no effect on the program (self pairs {dcs_self=} {ls_self=})"
  assert dcs_grads.keys() == ls_grads.keys()
  differ = sorted(name for name in dcs_grads if dcs_grads[name] != ls_grads[name])
  assert ls_logits == dcs_logits, "local_self logits differ from dual_chunk_swap"
  assert not differ, f"local_self gradients differ from dual_chunk_swap in {len(differ)} leaves: {differ[:4]}"
  print(
      f"model seq={seq} batch={batch}: logits and {len(dcs_grads)} gradient leaves bit-identical;"
      f" self pairs dual_chunk_swap={dcs_self} local_self={ls_self}",
      flush=True,
  )

  # Positive control: the self rank keeping the zeros it receives must change the logits.
  bad_logits, _, _ = _model_outputs_and_grads(ls_cfg, data, cotangent, patch=_self_rank_keeps_zeros())
  assert bad_logits != dcs_logits, "positive control: a self rank keeping zeros went undetected"
  print("model positive control detected", flush=True)


def _run_child_checks():
  """Every check that needs 8 CPU devices; run in the child process."""
  devices = np.array(jax.devices()[:8])
  _run_reorder_checks(Mesh(devices.reshape(4, 2), ("fsdp", "context")), "fsdp")
  _run_reorder_checks(Mesh(devices.reshape(2, 4), ("fsdp", "context")), "fsdp")
  _run_reorder_checks(Mesh(devices.reshape(1, 8), ("fsdp", "context")), None)
  _run_model_checks(seq=128)
  print(_CHILD_MARKER, flush=True)


@pytest.mark.cpu_only
def test_load_balance_order_is_bit_identical_on_cpu_mesh():
  if len(jax.devices()) >= 8 and jax.devices()[0].platform == "cpu":
    _run_child_checks()
    return
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _CHILD_MARKER in result.stdout


class LoadBalanceSelfRanksTest(absltest.TestCase):
  """The self ranks of the two permutations and the order names, with no devices."""

  def test_self_ranks(self):
    self.assertEqual(tokamax_ring_attention.load_balance_self_ranks(2), ((0, 1), ()))
    self.assertEqual(tokamax_ring_attention.load_balance_self_ranks(4), ((0,), (2,)))
    self.assertEqual(tokamax_ring_attention.load_balance_self_ranks(8), ((0, 5), ()))

  def test_self_ranks_are_the_fixed_points(self):
    for cp in (2, 4, 6, 8, 16):
      first, second = tokamax_ring_attention.load_balance_permutations(cp)
      self.assertEqual(
          tokamax_ring_attention.load_balance_self_ranks(cp),
          (tuple(s for s, d in first if s == d), tuple(s for s, d in second if s == d)),
          f"cp={cp}",
      )

  def test_rejects_an_unknown_order(self):
    with self.assertRaisesRegex(ValueError, "Unknown load-balance order 'half'"):
      tokamax_ring_attention.reorder_for_load_balance(None, mesh=None, axis_name="context", batch_axes=None, order="half")


class LoadBalanceOrderConfigTest(absltest.TestCase):
  """The key: its default, its values and its validator."""

  def _initialize(self, *overrides):
    """A Qwen3.5 (GatedDeltaNet) ring config on 8 mocked TPU devices with the attention load balance on."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "model_name=qwen3.5-397b-a17b",
        "override_model_config=true",
        "base_num_decoder_layers=4",
        "use_multimodal=false",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "context_parallel_attention_load_balance=True",
        "ici_context_parallelism=4",
        "max_target_length=4096",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
        *overrides,
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      return pyconfig.initialize(argv)

  def test_default_is_dual_chunk_swap(self):
    self.assertEqual(getattr(self._initialize(), _KEY), _DCS)
    self.assertEqual(getattr(self._initialize("context_parallel_attention_load_balance=False"), _KEY), _DCS)

  def test_accepts_local_self_with_the_load_balance(self):
    self.assertEqual(getattr(self._initialize(f"{_KEY}={_LOCAL_SELF}"), _KEY), _LOCAL_SELF)

  def test_rejects_an_unknown_value(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), _KEY):
      self._initialize(f"{_KEY}=dual_chunk_swap_local")

  def test_rejects_local_self_without_the_load_balance(self):
    with self.assertRaisesRegex(
        (ValueError, pydantic.ValidationError),
        rf"{_KEY}='{_LOCAL_SELF}' requires: context_parallel_attention_load_balance=True",
    ):
      self._initialize("context_parallel_attention_load_balance=False", f"{_KEY}={_LOCAL_SELF}")


if __name__ == "__main__":
  if len(sys.argv) == 1:
    _run_child_checks()
  else:
    absltest.main()
