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

"""gdn_conv1d_dense_token_stack: the Pallas GDN forward kernel's Conv1D output built from dense per-token views.

The option changes how compute_conv1d.causal_conv1d assembles a value the kernel computes either way: each token's
float32 row is viewed as [dim / 1024, 8, 128] and the views are concatenated, instead of jnp.stack(axis=1) of the
rows. Every element lands where jnp.stack puts it; only the layout the TPU compiler gives the value changes.

The forward kernel cannot run on CPU (neither Pallas interpreter supports its emit_pipeline DMAs), so:
  - Values (in this process): causal_conv1d, the one function whose code changes, gives the same bits with the
    option as without it at the kernel's in-VMEM shapes (widths 1024 and 12288, packed segment ids on/off, conv
    bias on/off, full and partial chunks), and so does the SiLU the kernel applies next. A token-minor regrouping
    is caught, so the comparison can fail. Without the option the rows go through jnp.stack itself (the same
    jaxpr), so the default program is the base program.
  - Programs, traced for a TPU target on this CPU host with no TPU or libtpu (runtime_utils.target_platform follows
    the abstract mesh). In this process: fused_conv1d_gdn hands the option to its per-sequence kernel only, and its
    batched (decode) kernel is the same program with and without it. In a child process (see main): every
    forward-kernel call carries the dense regrouping exactly when the option is on: wrapper.fused_conv1d_gdn;
    gdn_decoupled_conv1d's primal, its forward rule and the rule's replay under remat; both sequence-sharded
    context-parallel passes, under the forward rule and (forward only) under the primal; head-sharded context
    parallelism; and model_runner.run_gdn_kernel_layer reading the config key. The option composes with
    gdn_conv1d_input_in_qkv_dtype, and the backward rule accepts every positional form. Deliberately wrong
    versions must FAIL every check they break: an option the conv ignores, one the forward rule drops, and one the
    primal drops.
  - Guards: the per-call width check and the config validator reject the option where it would do nothing, and the
    validator's copy of model_runner's head-vs-sequence context-parallel rule agrees with model_runner.
"""

# pylint: disable=protected-access

import collections
import functools
import itertools
import os
import re
import subprocess
import sys
import types

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import ShardMode
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.gdn import compute_conv1d
from maxtext.kernels.gdn import config as gdn_config
from maxtext.kernels.gdn import model_runner
from maxtext.kernels.gdn import wrapper
from maxtext.kernels.gdn.gdn_bwd import api as gdn_api
from maxtext.kernels.gdn.gdn_bwd import cp_gdn
from tests.utils.test_helpers import get_test_config_path

_PASS_TOKEN = "GDN_CONV1D_DENSE_TOKEN_STACK_CHECKS_PASSED"
_CHUNK = 64
_KERNEL = 4
_HEAD_DIM = 128


def _bits(x):
  """The array's bytes, so equality means bit-identical (NaN-safe, -0.0 != 0.0)."""
  return np.asarray(jax.device_get(x)).tobytes()


def _gdn_config(dim, num_v_heads, has_seg_ids, dense):
  return gdn_config.GDNConfig(
      mode=gdn_config.GDNMode.PER_SEQ,
      dtypes=gdn_config.Dtypes(jnp.bfloat16, jnp.bfloat16, jnp.float32, jnp.float32, jnp.float32),
      batch_size=_CHUNK,
      dim_size=dim,
      kernel_size=_KERNEL,
      tile_size=_CHUNK,
      num_kq_heads=num_v_heads // 2,
      num_v_heads=num_v_heads,
      kq_head_dim=_HEAD_DIM,
      v_head_dim=_HEAD_DIM,
      has_seg_ids=has_seg_ids,
      conv_dense_token_stack=dense,
  )


def _conv_inputs(dim, num_v_heads, seed):
  """causal_conv1d's in-VMEM operands for one PER_SEQ tile: lhs = [halo | chunk tokens], weights, bias, b lanes."""
  rng = np.random.default_rng(seed)
  lhs = jnp.asarray(rng.standard_normal((1, _CHUNK + _KERNEL - 1, 1, dim)).astype(np.float32))
  weight = jnp.asarray((rng.standard_normal((_KERNEL, 1, dim)) * 0.5).astype(np.float32))
  bias = jnp.asarray(rng.standard_normal((dim,)).astype(np.float32))
  # b's lanes num_v_heads (signed active segment of each token: two documents, then padding) and num_v_heads + 1
  # (chunk header: the halo's segment ids, then the previous chunk's segment).
  b_vreg = np.zeros((1, _CHUNK, 1, 128), np.float32)
  b_vreg[0, :, 0, num_v_heads] = np.concatenate([np.full(21, 1.0), np.full(30, 2.0), np.full(_CHUNK - 51, -2.0)])
  b_vreg[0, :_KERNEL, 0, num_v_heads + 1] = 1.0
  return lhs, weight, bias, jnp.asarray(b_vreg)


def _stack_counts(jaxpr, dim):
  """(dense, stacked): how many times `jaxpr` (sub-jaxprs included) assembles a [1, chunk, 1, dim] Conv1D output from
  dense views (a concatenate to [chunk * dim / 1024, 8, 128]) or by stacking (a stack, or a concatenate, whose result
  is [1, chunk, 1, dim]; jax 0.11 traces jnp.stack as its own primitive)."""
  dense_shape, stack_shape = (_CHUNK * dim // 1024, 8, 128), (1, _CHUNK, 1, dim)
  dense = stacked = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in ("concatenate", "stack"):
      shape = tuple(eqn.outvars[0].aval.shape)
      dense += eqn.primitive.name == "concatenate" and shape == dense_shape
      stacked += shape == stack_shape
    for sub in jax_core.jaxprs_in_params(eqn.params):
      sub_dense, sub_stacked = _stack_counts(sub, dim)
      dense, stacked = dense + sub_dense, stacked + sub_stacked
  return dense, stacked


def _forward_kernels(closed_jaxpr):
  """(name, kernel jaxpr) of every forward-kernel call (a pallas_call named fused_conv1d_gdn*), in trace order."""
  found = []

  def visit(jaxpr):
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "pallas_call" and str(eqn.params.get("name", "")).startswith("fused_conv1d_gdn"):
        found.append((str(eqn.params["name"]), eqn.params["jaxpr"]))
        continue
      for sub in jax_core.jaxprs_in_params(eqn.params):
        visit(sub)

  visit(closed_jaxpr.jaxpr)
  return found


def _token_minor_stack(out_list):
  """Deliberately wrong regrouping: interleaves the tokens inside each group of 8 x 128 registers."""
  row = out_list[0]
  views = jnp.stack([o.reshape(-1, 8, 128) for o in out_list], axis=1)
  return views.reshape(row.shape[0], len(out_list), *row.shape[1:])


class StackConv1dTokensTest(parameterized.TestCase):
  """stack_conv1d_tokens(dense=True) equals jnp.stack(axis=1) bit for bit, and rejects rows it cannot regroup."""

  @parameterized.named_parameters(
      ("chunk64_width1024", 64, 1024),
      ("chunk64_width12288", 64, 12288),
      ("chunk3_width2048", 3, 2048),
  )
  def test_dense_equals_stack(self, num_tokens, dim):
    rng = np.random.default_rng(num_tokens + dim)
    rows = [jnp.asarray(rng.standard_normal((1, 1, dim)).astype(np.float32)) for _ in range(num_tokens)]
    ref = jnp.stack(rows, axis=1)
    for got in (
        compute_conv1d.stack_conv1d_tokens(rows, True),
        jax.jit(compute_conv1d.stack_conv1d_tokens, static_argnums=1)(rows, True),
    ):
      self.assertEqual(got.shape, ref.shape)
      self.assertEqual(got.dtype, ref.dtype)
      self.assertEqual(_bits(got), _bits(ref))
    self.assertEqual(_bits(compute_conv1d.stack_conv1d_tokens(rows, False)), _bits(ref))
    # Positive control: a regrouping that moves elements is caught by the same comparison. (With one register per
    # token, width 1024, the token-minor interleave is the same order, so it only moves elements at wider rows.)
    if dim > 1024:
      self.assertNotEqual(_bits(_token_minor_stack(rows)), _bits(ref))

  def test_default_is_jnp_stack(self):
    """dense=False traces to jnp.stack(axis=1) itself, so with the option off causal_conv1d is the base program.

    Equal values or shapes would not show this: a rewrite of the default path that keeps the values (a concatenate of
    expanded rows, say) is a different program.
    """
    rows = [jnp.zeros((1, 1, 2048), jnp.float32) for _ in range(_CHUNK)]
    ref = str(jax.make_jaxpr(lambda r: jnp.stack(r, axis=1))(rows))
    self.assertEqual(str(jax.make_jaxpr(lambda r: compute_conv1d.stack_conv1d_tokens(r, False))(rows)), ref)
    # Positive control: the comparison sees a program change that keeps the values.
    self.assertNotEqual(str(jax.make_jaxpr(lambda r: compute_conv1d.stack_conv1d_tokens(r, True))(rows)), ref)

  @parameterized.named_parameters(
      ("two_sequences", (2, 1, 1024), jnp.float32),
      ("partial_register", (1, 1, 1536), jnp.float32),
      ("bfloat16_rows", (1, 1, 2048), jnp.bfloat16),
  )
  def test_rejects_rows_it_cannot_regroup(self, shape, dtype):
    rows = [jnp.zeros(shape, dtype)] * 4
    with self.assertRaisesRegex(ValueError, "one sequence per tile and float32 rows"):
      compute_conv1d.stack_conv1d_tokens(rows, True)
    self.assertEqual(compute_conv1d.stack_conv1d_tokens(rows, False).shape, (shape[0], 4) + shape[1:])


_CONV_CASES = [
    {
        "testcase_name": f"width{dim}_seg{int(seg)}_bias{int(bias)}_real{real}",
        "dim": dim,
        "num_v_heads": nv,
        "has_seg_ids": seg,
        "with_bias": bias,
        "real": real,
    }
    for dim, nv in ((1024, 4), (12288, 64))
    for seg in (False, True)
    for bias in (False, True)
    for real in (_CHUNK, 37)
]


class CausalConv1dTest(parameterized.TestCase):
  """causal_conv1d with the option gives the same bits; and the option changes the traced conv."""

  @parameterized.named_parameters(*_CONV_CASES)
  def test_bit_identical(self, dim, num_v_heads, has_seg_ids, with_bias, real):
    lhs, weight, bias, b_vreg = _conv_inputs(dim, num_v_heads, seed=dim + 2 * has_seg_ids + with_bias)

    def run(dense):
      out, state = compute_conv1d.causal_conv1d(
          real_sizes=jnp.array([real], jnp.int32),
          lhs=lhs,
          conv_weight=weight,
          conv_bias=bias if with_bias else None,
          cfg=_gdn_config(dim, num_v_heads, has_seg_ids, dense),
          b_vreg=b_vreg if has_seg_ids else None,
      )
      return out, state, jax.nn.silu(out)  # inner_kernel applies the SiLU to the conv output next

    for ref, got in zip(jax.jit(lambda: run(False))(), jax.jit(lambda: run(True))()):
      self.assertEqual(ref.shape, got.shape)
      self.assertEqual(ref.dtype, got.dtype)
      self.assertEqual(_bits(ref), _bits(got))

  def test_option_changes_the_traced_conv(self):
    """Positive control for the value test: the option must change the program, not only keep the values."""
    dim, num_v_heads = 2048, 8
    lhs, weight, bias, _ = _conv_inputs(dim, num_v_heads, seed=0)

    def stacks(dense):
      cfg = _gdn_config(dim, num_v_heads, False, dense)
      jaxpr = jax.make_jaxpr(lambda: compute_conv1d.causal_conv1d(jnp.array([_CHUNK]), lhs, weight, bias, cfg))()
      return _stack_counts(jaxpr.jaxpr, dim)

    self.assertEqual(stacks(True), (1, 0))
    self.assertEqual(stacks(False), (0, 1))


class KernelIdentityTest(absltest.TestCase):
  """The option is left out of the kernel name and metadata, so the program outside the kernel body is unchanged."""

  def test_name_and_metadata_unchanged(self):
    off, on = _gdn_config(1024, 4, True, False), _gdn_config(1024, 4, True, True)
    self.assertEqual(off.get_kernel_name(), on.get_kernel_name())
    self.assertEqual(off.get_metadata(), on.get_metadata())
    self.assertNotIn("conv_dense_token_stack", off.get_metadata())
    self.assertFalse(gdn_config.GDNConfig.__dataclass_fields__["conv_dense_token_stack"].default)


def _tpu_target(shape=(), axes=()):
  """A context in which traces target TPU7x as in an AOT compile: no TPU and no libtpu needed."""
  device = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
  mesh = jax.sharding.AbstractMesh(
      shape, axes, axis_types=(jax.sharding.AxisType.Auto,) * len(axes), abstract_device=device
  )
  return jax.sharding.use_abstract_mesh(mesh), mesh


def _fused_args(dim, num_v_heads, seq=128):
  a = jax.ShapeDtypeStruct
  return (
      a((seq, dim), jnp.bfloat16),
      a((seq, num_v_heads), jnp.bfloat16),
      a((seq, num_v_heads), jnp.bfloat16),
      a((2, _KERNEL - 1, dim), jnp.bfloat16),
      a((2, num_v_heads, _HEAD_DIM, _HEAD_DIM), jnp.float32),
      a((dim, 1, _KERNEL), jnp.float32),
      None,
      a((num_v_heads,), jnp.float32),
      a((num_v_heads,), jnp.float32),
      a((2,), jnp.int32),
      a((1,), jnp.int32),
      a((3,), jnp.int32),
      a((1,), jnp.int32),
  )


def _trace_fused(num_k_heads, num_v_heads, dense, *, prefill_only=True):
  """wrapper.fused_conv1d_gdn traced for a TPU7x target.

  Prefill-only (as gdn_decoupled_conv1d calls it) runs the per-sequence kernel alone; prefill_only=False (mixed
  mode) runs the batched (decode) kernel first, at a decode tile of 1 because _fused_args has one sequence.
  """
  dim = (2 * num_k_heads + num_v_heads) * _HEAD_DIM
  ctx, _ = _tpu_target()
  fn = functools.partial(
      wrapper.fused_conv1d_gdn,
      n_kq=num_k_heads,
      n_v=num_v_heads,
      d_k=_HEAD_DIM,
      d_v=_HEAD_DIM,
      kernel_size=_KERNEL,
      compute_precision=jnp.dtype(jnp.float32),
      mixed_tile_size=_CHUNK,
      is_prefill_only=prefill_only,
      conv_dense_token_stack=dense,
  )
  with ctx:
    return jax.make_jaxpr(fn)(*_fused_args(dim, num_v_heads))


class WidthCheckTest(absltest.TestCase):
  """fused_conv1d_gdn rejects the option for a Conv1D width that is not a whole number of float32 registers."""

  def test_rejects_width_512(self):
    _trace_fused(1, 2, False)  # width 512; the default path is unchanged
    with self.assertRaisesRegex(ValueError, r"requires the Conv1D width \(512\) to be a multiple of 1024"):
      _trace_fused(1, 2, True)

  def test_accepts_width_1024(self):
    _trace_fused(2, 4, True)


class BatchedKernelTest(absltest.TestCase):
  """fused_conv1d_gdn hands the option to its per-sequence kernel only; the batched (decode) kernel keeps jnp.stack."""

  def test_batched_kernel_does_not_take_the_option(self):
    # Mixed mode runs the batched kernel, then the per-sequence kernel. With one sequence the batched kernel's tile
    # holds one sequence, so its one-token rows are the only batched rows the dense regrouping could take.
    kernels = {dense: _forward_kernels(_trace_fused(2, 4, dense, prefill_only=False)) for dense in (False, True)}
    names = [name for name, _ in kernels[True]]
    self.assertEqual(names, [name for name, _ in kernels[False]])
    self.assertLen(names, 2)
    self.assertTrue(names[0].startswith("fused_conv1d_gdn_batched_b1_"), names)
    self.assertTrue(names[1].startswith("fused_conv1d_gdn_per_seq_"), names)
    (_, batched_off), (_, per_seq_off) = kernels[False]
    (_, batched_on), (_, per_seq_on) = kernels[True]
    self.assertEqual(str(batched_on), str(batched_off))
    self.assertEqual(_stack_counts(per_seq_on, 1024), (1, 0))
    self.assertEqual(_stack_counts(per_seq_off, 1024), (0, 1))
    # Positive control: the same comparison tells the per-sequence kernel's two programs apart.
    self.assertNotEqual(str(per_seq_on), str(per_seq_off))


def _validator_cfg(**overrides):
  """The fields validate_gdn_config reads, for a Qwen3.5-397B-shaped GDN kernel path that passes it."""
  fields = {
      "decoder_block": config_types.DecoderBlockType.QWEN3_5,
      "ici_context_parallelism": 1,
      "dcn_context_parallelism": 1,
      "ici_context_usp_ulysses_parallelism": 1,
      "dcn_context_usp_ulysses_parallelism": 1,
      "use_gdn_kernel": True,
      "ici_tensor_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "gdn_chunk_size": 64,
      "gdn_key_head_dim": 128,
      "gdn_value_head_dim": 128,
      "gdn_num_key_heads": 16,
      "gdn_num_value_heads": 64,
      "use_qk_norm_in_gdn": True,
      "max_target_length": 4096,
      "gdn_conv1d_input_in_qkv_dtype": False,
      "gdn_conv1d_dense_token_stack": False,
      "gdn_gated_norm_layout": "auto",
      "gdn_cp_mode": "auto",
      "packing": True,
  }
  fields.update(overrides)
  return types.SimpleNamespace(**fields)


class ValidatorTest(parameterized.TestCase):
  """The validator rejects the key where it would do nothing; the default config is untouched."""

  @parameterized.named_parameters(
      ("no_cp", {}),
      ("head_cp4", {"ici_context_parallelism": 4, "gdn_cp_mode": "head"}),  # 3072 per rank
      ("auto_cp2_head_sharded", {"ici_context_parallelism": 2}),  # 6144 per rank
      ("auto_cp4_seq_sharded", {"ici_context_parallelism": 4, "gdn_num_key_heads": 2, "gdn_num_value_heads": 4}),
      # 2048: model_runner runs auto CP 4 sequence-sharded even with 4 key heads (head-sharded would be 512 per rank).
      (
          "auto_cp4_seq_sharded_four_key_heads",
          {"ici_context_parallelism": 4, "gdn_num_key_heads": 4, "gdn_num_value_heads": 8},
      ),
  )
  def test_accepted(self, overrides):
    cfg = _validator_cfg(gdn_conv1d_dense_token_stack=True, **overrides)
    config_types.MaxTextConfig.validate_gdn_config(cfg, is_rl_config=False)

  def test_head_sharding_agrees_with_model_runner(self):
    """The validator's copy of model_runner._resolve_gdn_cp_mode's head-vs-sequence rule agrees with it.

    configs/types.py cannot import the kernel module, so the validator repeats the rule that decides whether one
    forward-kernel call convolves every head or one rank's heads. Over every GDN context-parallel mode, degrees 1-8 on
    each context axis, and head counts and head dims that model_runner runs, the validator must accept the key exactly
    when the width one forward-kernel call convolves, as model_runner shards it, is a multiple of 1024.
    """
    seen = collections.Counter()
    for cp_mode, cp, axis, (nk, nv), head_dim in itertools.product(
        ("auto", "head", "seq"),
        range(1, 9),
        ("ici_context_parallelism", "ici_context_usp_ulysses_parallelism", "dcn_context_parallelism"),
        ((1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16), (16, 32), (16, 64), (32, 64)),
        (128, 256),
    ):
      fields = {
          axis: cp,
          "gdn_cp_mode": cp_mode,
          "gdn_num_key_heads": nk,
          "gdn_num_value_heads": nv,
          "gdn_key_head_dim": head_dim,
          "gdn_value_head_dim": head_dim,
          "max_target_length": 840 * _CHUNK,  # a whole number of chunks per rank at every degree 1-8
      }
      try:
        head_sharded, _ = model_runner._resolve_gdn_cp_mode(_validator_cfg(**fields), cp, nk)
      except ValueError:
        continue  # model_runner refuses this config itself
      ranks = cp if head_sharded else 1
      width = 2 * (nk // ranks) * head_dim + (nv // ranks) * head_dim
      try:
        config_types.MaxTextConfig.validate_gdn_config(
            _validator_cfg(gdn_conv1d_dense_token_stack=True, **fields), is_rl_config=False
        )
        accepted = True
      except ValueError as e:
        self.assertIn("gdn_conv1d_dense_token_stack=True requires the width", str(e))
        accepted = False
      self.assertEqual(accepted, width % 1024 == 0, f"{fields}: model_runner head_sharded={head_sharded}")
      seen[(head_sharded, accepted)] += 1
    # Head- and sequence-sharded configs, each both accepted and rejected: the comparison can fail either way.
    self.assertLen(seen, 4, seen)

  @parameterized.named_parameters(
      ("no_gdn_layers", {"decoder_block": config_types.DecoderBlockType.LLAMA2}, "a decoder with Gated DeltaNet"),
      ("no_gdn_kernel", {"use_gdn_kernel": False}, "use_gdn_kernel=True"),
      ("width_512", {"gdn_num_key_heads": 1, "gdn_num_value_heads": 2}, r"the width .* \(the whole Conv1D: .* = 512\)"),
      (
          "head_cp8_rank_width_1536",
          {"ici_context_parallelism": 8, "gdn_cp_mode": "head"},
          r"the width .* \(one rank's heads under head-sharded GDN context parallelism of degree 8: .* = 1536\)",
      ),
      (
          "auto_cp2_rank_width_512",
          {"ici_context_parallelism": 2, "gdn_num_key_heads": 2, "gdn_num_value_heads": 4},
          r"the width .* \(one rank's heads under head-sharded GDN context parallelism of degree 2: .* = 512\)",
      ),
  )
  def test_rejected(self, overrides, message):
    cfg = _validator_cfg(gdn_conv1d_dense_token_stack=True, **overrides)
    with self.assertRaisesRegex(ValueError, f"gdn_conv1d_dense_token_stack=True requires {message}"):
      config_types.MaxTextConfig.validate_gdn_config(cfg, is_rl_config=False)
    # The same config with the key off passes.
    config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(**overrides), is_rl_config=False)

  def test_pyconfig(self):
    def cfg(**kwargs):
      return pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="gdn_conv1d_dense_token_stack",
          enable_checkpointing=False,
          log_config=False,
          skip_jax_distributed_system=True,
          **kwargs,
      )

    self.assertFalse(cfg().gdn_conv1d_dense_token_stack)
    self.assertFalse(config_types.Qwen3Next().gdn_conv1d_dense_token_stack)
    on = cfg(model_name="qwen3.5-397b-a17b", use_gdn_kernel=True, gdn_conv1d_dense_token_stack=True)
    self.assertTrue(on.gdn_conv1d_dense_token_stack)
    with self.assertRaisesRegex(ValueError, "requires a decoder with Gated DeltaNet layers"):
      cfg(gdn_conv1d_dense_token_stack=True)


@pytest.mark.cpu_only
def test_programs_in_a_child_process():
  """Runs `main` below in a fresh CPU process (the program checks trace TPU kernels on this host)."""
  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


# ----------------------------------------------------------------------------------------------------------------------
# Program checks (main). Each forward-kernel call is a pallas_call named fused_conv1d_gdn*; _stack_counts reads which
# way its kernel jaxpr assembles the Conv1D output.


def _forward_kernel_stacks(closed_jaxpr, dim):
  """[_stack_counts] of every fused_conv1d_gdn kernel in the jaxpr, in trace order."""
  return [_stack_counts(kernel, dim) for _, kernel in _forward_kernels(closed_jaxpr)]


def _bf16_conv_inputs(closed_jaxpr, shape):
  """bf16 pad / concatenate results of `shape` outside the Pallas kernels (gdn_conv1d_input_in_qkv_dtype's marker)."""
  found = 0

  def visit(jaxpr):
    nonlocal found
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "pallas_call":
        continue
      if eqn.primitive.name in ("pad", "concatenate"):
        aval = eqn.outvars[0].aval
        found += tuple(aval.shape) == tuple(shape) and aval.dtype == jnp.bfloat16
      for sub in jax_core.jaxprs_in_params(eqn.params):
        visit(sub)

  visit(closed_jaxpr.jaxpr)
  return found


class _Value:
  """A parameter stand-in: .value and [...] like an nnx Param."""

  def __init__(self, value):
    self.value = value

  def __getitem__(self, idx):
    return self.value[idx]


def _trace_model_runner(mode, nk, nv, *, dense, qkv_dtype=False, remat=True, seg=True, seq=512, grad=True):
  """model_runner.run_gdn_kernel_layer on a stand-in GDN layer, traced for a TPU7x target.

  grad=True traces value_and_grad, so the custom VJP's forward rule makes the forward-kernel calls; grad=False traces
  the forward pass alone (as a forward-only RL pass runs it), so its primal makes them.
  """
  cp = {"none": 1, "seq4": 4, "head2": 2}[mode]
  if mode == "none":
    ctx, mesh = _tpu_target()
    layer_mesh = None
  else:
    ctx, mesh = _tpu_target((cp,), ("context",))
    layer_mesh = mesh
  cfg = types.SimpleNamespace(
      ici_context_parallelism=cp,
      dcn_context_parallelism=1,
      ici_context_usp_ulysses_parallelism=1,
      dcn_context_usp_ulysses_parallelism=1,
      gdn_cp_mode={"head2": "head", "seq4": "seq"}.get(mode, "auto"),
      gdn_cp_matmul_precision="highest",
      shard_mode=ShardMode.AUTO,
      logical_axis_rules=[],
      gdn_conv_kernel_dim=_KERNEL,
      gdn_chunk_size=_CHUNK,
      use_qk_norm_in_gdn=True,
      context_parallel_load_balance=False,
      gdn_conv1d_input_in_qkv_dtype=qkv_dtype,
      gdn_conv1d_dense_token_stack=dense,
      gdn_bwd_natural_layout="off",
  )
  key_dim, conv_dim = nk * _HEAD_DIM, (2 * nk + nv) * _HEAD_DIM

  def loss(params, x, aux):
    def body(params, x):
      layer = types.SimpleNamespace(
          config=cfg,
          mesh=layer_mesh,
          num_k_heads=nk,
          num_v_heads=nv,
          head_k_dim=_HEAD_DIM,
          head_v_dim=_HEAD_DIM,
          key_dim=key_dim,
          v_heads_per_k_head=nv // nk,
          conv1d=types.SimpleNamespace(kernel=_Value(params["conv_w"]), bias=None),
          A_log=_Value(params["a_log"]),
          dt_bias=_Value(params["dt_bias"]),
      )
      out, next_cs, next_rs = model_runner.run_gdn_kernel_layer(
          layer=layer,
          query=x["query"],
          key=x["key"],
          value=x["value_raw"].reshape(1, seq, nv, _HEAD_DIM),
          value_raw=x["value_raw"],
          b=x["b"],
          a=x["a"],
          active_cache=None,
          decoder_segment_ids=aux["seg"] if seg else None,
          model_mode="train",
          state_dtype=jnp.float32,
      )
      return (
          jnp.sum(out.astype(jnp.float32) * aux["do"])
          + 0.5 * jnp.sum(next_cs.astype(jnp.float32))
          + 0.25 * jnp.sum(next_rs.astype(jnp.float32))
      )

    if remat:
      body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
    return body(params, x)

  a = jax.ShapeDtypeStruct
  params = {
      "conv_w": a((_KERNEL, 1, conv_dim), jnp.float32),
      "a_log": a((nv,), jnp.float32),
      "dt_bias": a((nv,), jnp.float32),
  }
  x = {
      "query": a((1, seq, nk, _HEAD_DIM), jnp.bfloat16),
      "key": a((1, seq, nk, _HEAD_DIM), jnp.bfloat16),
      "value_raw": a((1, seq, nk, (nv // nk) * _HEAD_DIM), jnp.bfloat16),
      "b": a((1, seq, nv), jnp.bfloat16),
      "a": a((1, seq, nv), jnp.bfloat16),
  }
  aux = {"seg": a((1, seq), jnp.int32), "do": a((1, seq, nv, _HEAD_DIM), jnp.float32)}
  fn = jax.value_and_grad(loss, argnums=(0, 1)) if grad else loss
  with ctx:
    return jax.make_jaxpr(fn)(params, x, aux)


def _trace_decoupled(dense, *, grad, remat, qkv_dtype=False, nk=2, nv=4, seq=128):
  """gdn_decoupled_conv1d for a TPU7x target: the primal alone, or value_and_grad (forward rule), optionally remat."""
  dim = (2 * nk + nv) * _HEAD_DIM
  op = functools.partial(
      gdn_api.gdn_decoupled_conv1d,
      num_k_heads=nk,
      num_v_heads=nv,
      head_k_dim=_HEAD_DIM,
      head_v_dim=_HEAD_DIM,
      conv_kernel_size=_KERNEL,
      chunk_size=_CHUNK,
      use_qk_norm_in_gdn=True,
      compute_dtype=jnp.float32,
      conv_input_in_qkv_dtype=qkv_dtype,
      conv_dense_token_stack=dense,
  )

  def loss(qkv, b, a, cw, a_log, dtb, do):
    def body(qkv, b, a, cw, a_log, dtb):
      out, _ = op(
          qkv=qkv,
          b=b,
          a=a,
          conv_weight=cw,
          conv_bias=None,
          a_log=a_log,
          dt_bias=dtb,
          conv_state=None,
          recurrent_state=None,
      )
      return jnp.sum(out.astype(jnp.float32) * do)

    if remat:
      body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
    return body(qkv, b, a, cw, a_log, dtb)

  a = jax.ShapeDtypeStruct
  args = (
      a((1, seq, dim), jnp.bfloat16),
      a((1, seq, nv), jnp.bfloat16),
      a((1, seq, nv), jnp.bfloat16),
      a((_KERNEL, 1, dim), jnp.float32),
      a((nv,), jnp.float32),
      a((nv,), jnp.float32),
      a((1, seq, nv, _HEAD_DIM), jnp.float32),
  )
  fn = jax.value_and_grad(loss, argnums=tuple(range(6))) if grad else loss
  ctx, _ = _tpu_target()
  with ctx:
    return jax.make_jaxpr(fn)(*args), dim


def _check(label, ok, detail=""):
  print(f"{label}: [{'ok' if ok else 'FAIL'}] {detail}", flush=True)
  return ok


def _all_dense(stacks, expected_calls):
  return len(stacks) == expected_calls and all(s == (1, 0) for s in stacks)


def _all_stacked(stacks, expected_calls):
  return len(stacks) == expected_calls and all(s == (0, 1) for s in stacks)


# The cases of _option_reaches_every_forward_kernel whose forward-kernel calls go through the custom VJP's forward
# rule (traced under value_and_grad) or through its primal (traced forward only). A deliberately wrong version in main
# that drops the option in one of the two must break every case in its group.
_FORWARD_RULE_CASES = (
    "decoupled_forward_rule",
    "decoupled_forward_rule_remat",
    "runner_no_cp",
    "runner_seq_cp4",
    "runner_head_cp2",
    "runner_seq_cp4_qwen35_397b_heads",
)
_PRIMAL_CASES = ("decoupled_primal", "runner_seq_cp4_forward_only")


def _option_reaches_every_forward_kernel(names=None):
  """Every forward-kernel call carries the dense regrouping iff the option is on; returns (ok, {case: (ok, detail)}).

  `names` runs only those cases (the deliberately wrong versions in main use it).
  """
  results = {}

  def want(name):
    return names is None or name in names

  # wrapper.fused_conv1d_gdn: the per-sequence kernel (the only one a prefill-only call runs).
  for nk, nv in ((2, 4), (16, 64)):
    dim = (2 * nk + nv) * _HEAD_DIM
    name = f"fused_conv1d_gdn_width{dim}"
    if want(name):
      on = _forward_kernel_stacks(_trace_fused(nk, nv, True), dim)
      off = _forward_kernel_stacks(_trace_fused(nk, nv, False), dim)
      results[name] = (_all_dense(on, 1) and _all_stacked(off, 1), f"on={on} off={off}")
  # gdn_decoupled_conv1d: the primal (1 call), the forward rule (1), the forward rule and its remat replay (2).
  for name, grad, remat, calls in (
      ("decoupled_primal", False, False, 1),
      ("decoupled_forward_rule", True, False, 1),
      ("decoupled_forward_rule_remat", True, True, 2),
  ):
    if want(name):
      jaxpr_on, dim = _trace_decoupled(True, grad=grad, remat=remat)
      jaxpr_off, _ = _trace_decoupled(False, grad=grad, remat=remat)
      on, off = _forward_kernel_stacks(jaxpr_on, dim), _forward_kernel_stacks(jaxpr_off, dim)
      results[name] = (_all_dense(on, calls) and _all_stacked(off, calls), f"on={on} off={off}")
  # model_runner reading the config key, under remat. With value_and_grad (the custom VJP's forward rule): no context
  # parallelism (1 call + replay), sequence-sharded CP 4 (2 passes + their replays), head-sharded CP 2 (1 call per
  # rank + replay), and the production head counts (16 key, 64 value heads) under sequence-sharded CP 4. Forward
  # only (the custom VJP's primal): sequence-sharded CP 4 (2 passes, no replay).
  for name, mode, nk, nv, seq, grad, calls in (
      ("runner_no_cp", "none", 4, 8, 512, True, 2),
      ("runner_seq_cp4", "seq4", 4, 8, 512, True, 4),
      ("runner_head_cp2", "head2", 4, 8, 512, True, 2),
      ("runner_seq_cp4_qwen35_397b_heads", "seq4", 16, 64, 256, True, 4),
      ("runner_seq_cp4_forward_only", "seq4", 4, 8, 512, False, 2),
  ):
    if want(name):
      dim = (2 * nk + nv) * _HEAD_DIM // (2 if mode == "head2" else 1)
      on = _forward_kernel_stacks(_trace_model_runner(mode, nk, nv, dense=True, seq=seq, grad=grad), dim)
      off = _forward_kernel_stacks(_trace_model_runner(mode, nk, nv, dense=False, seq=seq, grad=grad), dim)
      results[name] = (_all_dense(on, calls) and _all_stacked(off, calls), f"on={on} off={off}")
  return all(ok for ok, _ in results.values()), results


def _composes_with_input_in_qkv_dtype():
  """All four combinations: each key's own marker is present iff that key is on."""
  ok = True
  for qkv_dtype in (False, True):
    for dense in (False, True):
      jaxpr, dim = _trace_decoupled(dense, grad=True, remat=False, qkv_dtype=qkv_dtype)
      stacks = _forward_kernel_stacks(jaxpr, dim)
      narrow = _bf16_conv_inputs(jaxpr, (1, 128 + _KERNEL - 1, dim))
      good = (_all_dense(stacks, 1) if dense else _all_stacked(stacks, 1)) and (narrow > 0) == qkv_dtype
      ok &= _check(
          f"compose qkv_dtype={qkv_dtype} dense={dense}", good, f"forward kernels={stacks} bf16 conv inputs={narrow}"
      )
  return ok


def _backward_rule_positional_forms():
  """The backward rule traces the same program from every positional form custom_vjp or a caller may use."""
  nk, nv, seq = 2, 4, 128
  dim = (2 * nk + nv) * _HEAD_DIM
  nondiff = (nk, nv, _HEAD_DIM, _HEAD_DIM, _KERNEL, _CHUNK, True, jnp.float32)
  prec = cp_gdn.DEFAULT_PRECISION
  forms = {
      10: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, res, cot),
      11: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, res, cot),
      12: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, res, cot),
      13: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, False, res, cot),
      14: lambda res, cot: gdn_api._gdn_decoupled_conv1d_bwd(*nondiff, None, prec, False, True, res, cot),
  }

  def program(form):
    def f(qkv, b, a, cw, a_log, dtb, do):
      _, res = gdn_api._gdn_decoupled_conv1d_fwd(qkv, b, a, cw, None, a_log, dtb, None, None, *nondiff)
      return forms[form](res, (do, (None, None)))

    a = jax.ShapeDtypeStruct
    args = (
        a((1, seq, dim), jnp.bfloat16),
        a((1, seq, nv), jnp.bfloat16),
        a((1, seq, nv), jnp.bfloat16),
        a((_KERNEL, 1, dim), jnp.float32),
        a((nv,), jnp.float32),
        a((nv,), jnp.float32),
        a((1, seq, nv, _HEAD_DIM), jnp.float32),
    )
    ctx, _ = _tpu_target()
    with ctx:
      return str(jax.make_jaxpr(f)(*args))

  ref = program(14)
  ok = True
  for form in (10, 11, 12, 13):
    try:
      same = program(form) == ref
    except Exception as e:  # pylint: disable=broad-exception-caught
      same = False
      print(f"  form {form} raised {type(e).__name__}: {e}", flush=True)
    ok &= _check(f"backward rule, {form} positional arguments", same)
  return ok


_BODY = re.compile(r"\\22body\\22: \\22[A-Za-z0-9+/=]+\\22")


def _lowers_to_mosaic():
  """Lowered for a TPU7x target (Pallas -> Mosaic, as an AOT compile does), the option changes the kernel body only."""
  ok = True
  for nk, nv in ((2, 4), (16, 64)):
    texts = {}
    for dense in (False, True):
      ctx, _ = _tpu_target()
      fn = functools.partial(
          wrapper.fused_conv1d_gdn,
          n_kq=nk,
          n_v=nv,
          d_k=_HEAD_DIM,
          d_v=_HEAD_DIM,
          kernel_size=_KERNEL,
          compute_precision=jnp.dtype(jnp.float32),
          mixed_tile_size=_CHUNK,
          is_prefill_only=True,
          conv_dense_token_stack=dense,
      )
      with ctx:
        lowered = jax.jit(fn).trace(*_fused_args((2 * nk + nv) * _HEAD_DIM, nv)).lower(lowering_platforms=("tpu",))
      texts[dense] = lowered.as_text(debug_info=False)
    names = {d: re.findall(r'kernel_name = "([^"]+)"', t) for d, t in texts.items()}
    good = (
        len(names[False]) == 1
        and names[False] == names[True]
        and names[False][0].startswith("fused_conv1d_gdn")
        and len(_BODY.findall(texts[True])) == 1
        and texts[False] != texts[True]
        and _BODY.sub("BODY", texts[False]) == _BODY.sub("BODY", texts[True])
    )
    ok &= _check(f"mosaic lowering width {(2 * nk + nv) * _HEAD_DIM}", good, f"kernels={names[True]}")
  return ok


def _with_wrong_version(obj, attr, wrong, names):
  """The results of the forward-kernel cases `names` (None: all) with `obj.attr` replaced by a wrong version."""
  real = getattr(obj, attr)
  # jax.jit caches traces (fused_conv1d_gdn is jitted): clear them so the patched function is traced, and again
  # afterwards so no later check reuses a trace of it.
  jax.clear_caches()
  setattr(obj, attr, wrong)
  try:
    return _option_reaches_every_forward_kernel(names)[1]
  finally:
    setattr(obj, attr, real)
    jax.clear_caches()


def _ignored_option(names):
  """Deliberately wrong: the conv ignores the option (always jnp.stack)."""
  real = compute_conv1d.stack_conv1d_tokens

  def ignore(out_list, dense):
    del dense
    return real(out_list, False)

  return _with_wrong_version(compute_conv1d, "stack_conv1d_tokens", ignore, names)


def _option_dropped_by_forward_rule(names):
  """Deliberately wrong: the custom VJP's forward rule (and so its remat replay) drops the option."""
  real_fwd = gdn_api.gdn_decoupled_conv1d.fwd

  def drop(*args):
    args = list(args)
    args[21] = False  # conv_dense_token_stack
    return real_fwd(*args)

  return _with_wrong_version(gdn_api.gdn_decoupled_conv1d, "fwd", drop, names)


def _option_dropped_by_primal(names):
  """Deliberately wrong: the custom VJP's primal (every forward-only call, with or without CP) drops the option."""
  real_fun = gdn_api.gdn_decoupled_conv1d.fun

  @functools.wraps(real_fun)  # custom_vjp binds keyword arguments against the primal's signature
  def drop(*args):
    args = list(args)
    args[21] = False  # conv_dense_token_stack
    return real_fun(*args)

  return _with_wrong_version(gdn_api.gdn_decoupled_conv1d, "fun", drop, names)


def main():
  ok = True
  reached, details = _option_reaches_every_forward_kernel()
  for name, (good, detail) in details.items():
    ok &= _check(name, good, detail)
  ok &= _check("option reaches every forward kernel", reached)
  ok &= _lowers_to_mosaic()
  ok &= _composes_with_input_in_qkv_dtype()
  ok &= _backward_rule_positional_forms()
  # Each deliberately wrong version must make every case it breaks FAIL. A case that passes, or that does not exist
  # (a misspelt name), counts as a miss.
  controls = {
      "control_ignored_option": (_ignored_option, tuple(details)),
      "control_option_dropped_by_forward_rule": (_option_dropped_by_forward_rule, _FORWARD_RULE_CASES),
      "control_option_dropped_by_primal": (_option_dropped_by_primal, _PRIMAL_CASES),
  }
  for name, (wrong_version, must_break) in controls.items():
    results = wrong_version(must_break)
    missed = [case for case in must_break if results.get(case, (True, ""))[0]]
    detail = f"detected by all {len(must_break)} cases it breaks" if not missed else f"NOT detected by {missed}"
    ok &= _check(name, not missed, detail)
  if not ok:
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
