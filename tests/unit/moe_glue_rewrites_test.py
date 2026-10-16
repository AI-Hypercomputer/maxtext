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
"""Three exact rewrites of MoE glue ops, each against the plain program, on CPU.

  - moe_unsplit_prefused_wi: the prefused wi = [w0 | w1] stays one tensor through the fp8 fake-quant, the FSDP
    all-gather and the prefetch barrier into the up-projection GEMM (no split at the layer entry, no concatenate).
  - moe_ragged_sort_unique_pair_argsort: the ring-of-experts ragged sort's argsorts as unstable two-operand sorts of
    unique (key, index) pairs (`argsort_unique_pairs`) instead of stable argsorts.
  - moe_prefused_swiglu_vjp: the activation on the whole [gate | up] output as one custom-VJP op
    (`moe._prefused_ffn_activation`) whose backward writes d[gate | up] as one array.

None changes a value, so everything must be bit-identical.

What a CPU run can and cannot reach: the SparseCore gather-reduce (and with it the validity sort that
moe_ragged_sort_unique_pair_argsort also rewrites) runs only on a TPU target; on CPU `ragged_gather_reduce` takes its
jnp reference fallback, which does not sort. `PreprocessTest` therefore calls `_preprocess` directly, the pure-JAX
part of that path. The ragged sort's own argsorts, the weight layout and the activation are plain JAX and run in the
layer below as on TPU (the GMM is megablox in interpret mode; tokamax gmm v2 is TPU-only).

In-process checks: `argsort_unique_pairs` equals the stable argsort (ties, booleans, every axis, the index dtype);
`_preprocess` with and without it agree; `_prefused_ffn_activation` gives the split activation's value bit for bit and
its VJP value for value (SwiGLU and the clipped GPT-OSS form, bfloat16 and float32). A -0.0 cotangent is where the
two can differ in bits: XLA:CPU turns it into +0.0 on the split path (an add of zero pads) and the fused concatenate
keeps it, so the gradient bits may differ only in the sign of exact zeros, which the test pins down. The validators
reject each key for each requirement it does not meet, naming that requirement; that includes moe_unsplit_prefused_wi
with tensor parallelism, which shards the expert MLP dimension.

The layer check runs as a subprocess with 8 CPU devices: the shrunk Qwen3.5 routed MoE of
`moe_roe_local_routing_test.py` with prefuse_moe_weights (16 experts, top 4, cp-as-ep, ragged sort, local routing,
two token chunks). Pre-registered: each key alone and all three together give an output, a loss and every gradient
bit-identical to all three off, in the plain and the pipelined chunk loop, with fp8 fake-quant and padding excluded
from routing, with forced ids and a dropless buffer, and with moe_routing_index_math="count". Two-sided:
  - Structure, so a switch that silently does nothing, or does only part of its job, cannot pass:
    moe_unsplit_prefused_wi traces fewer concatenates (and with fake-quant fewer barriers);
    moe_ragged_sort_unique_pair_argsort traces exactly two two-key sorts per token chunk (the permutation and its
    inverse) and the plain program none, and every ragged_gather_reduce call gets unique_pair_argsort=True (a spy:
    CPU runs the reference fallback, which does not sort); moe_prefused_swiglu_vjp traces one more custom_vjp call
    per token chunk.
  - The trace-time guard behind the validator: on a mesh with tensor parallelism 2 (default axis rules, expert 4 x
    tensor 2), a config that claims moe_unsplit_prefused_wi past the validator must make the layer raise rather
    than compute with a sharded [w0 | w1] (it would give the wrong output), while the plain layer runs.
  - The bench's MoE settings (expert-weight prefetch, row-major reduce-scatter, routing maps saved on device,
    ragged_buffer_factor 2.75, fp8 fake-quant, padding excluded) with all three keys in one case.
  - Deliberately wrong versions must FAIL: a fused activation whose backward swaps d(gate) and d(up) (the output
    is unchanged, the gradients are not), and a pair argsort that breaks ties by decreasing index, under a buffer
    that drops tokens (the tie order decides which tokens a full buffer drops, so the output changes).
"""

import contextlib
import os
import re
import subprocess
import sys
from unittest import mock

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.kernels.ragged import ragged_gather_reduce_v2
from maxtext.kernels.ragged import ragged_sort
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_SEQ = 64
_PASS_TOKEN = "MOE_GLUE_REWRITES_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
_KEYS = ("moe_unsplit_prefused_wi", "moe_ragged_sort_unique_pair_argsort", "moe_prefused_swiglu_vjp")


@pytest.mark.cpu_only
class ArgsortUniquePairsTest(parameterized.TestCase):
  """`argsort_unique_pairs` is the stable argsort."""

  @parameterized.named_parameters(
      ("int_ties_1d", (257,), jnp.int32, -1),
      ("int_ties_2d_last", (5, 96), jnp.int32, -1),
      ("int_ties_2d_first", (96, 5), jnp.int32, 0),
      ("bool_2d", (8, 100), jnp.bool_, 1),
  )
  def test_matches_stable_argsort(self, shape, dtype, axis):
    keys = jax.random.randint(jax.random.PRNGKey(0), shape, 0, 7).astype(dtype)  # many ties
    want = jnp.argsort(keys, axis=axis, stable=True)
    got = ragged_gather_reduce_v2.argsort_unique_pairs(keys, axis=axis)
    self.assertEqual(got.dtype, want.dtype)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_wrong_tie_order_is_detected(self):
    """Two-sided: the same comparison fails for a pair sort that breaks ties by decreasing index."""
    keys = jnp.array([1, 0, 1, 0, 1], jnp.int32)
    wrong = _argsort_ties_reversed(keys)
    self.assertFalse(np.array_equal(np.asarray(wrong), np.asarray(jnp.argsort(keys, stable=True))))


@pytest.mark.cpu_only
class PreprocessTest(parameterized.TestCase):
  """The SparseCore gather-reduce's validity sort with and without the pair argsort.

  Only the SparseCore path runs `_preprocess`; on a CPU target `ragged_gather_reduce` takes its reference fallback,
  so this pure-JAX function is called directly.
  """

  @parameterized.named_parameters(("one_partition", 96, 4, 1, 16, 16), ("v7x_like", 1024, 8, 16, 16, 48))
  def test_unique_pair_argsort_matches_argsort(self, rows, group, partitions, lanes, chunk):
    for density in (0.0, 0.4, 1.0):
      mask = jax.random.uniform(jax.random.PRNGKey(rows), (rows,)) < density
      want = ragged_gather_reduce_v2._preprocess(mask, group, partitions, lanes, chunk)  # pylint: disable=protected-access
      got = ragged_gather_reduce_v2._preprocess(  # pylint: disable=protected-access
          mask, group, partitions, lanes, chunk, unique_pair_argsort=True
      )
      for a, b in zip(want, got):
        self.assertEqual(a.dtype, b.dtype)
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


def _swiglu(gate, up):
  return (jax.nn.silu(gate) * up).astype(gate.dtype)


def _gpt_oss(gate, up, limit=7.0):
  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, -limit, limit)
  return (gate * jax.nn.sigmoid(gate * 1.702) * (up + 1)).astype(gate.dtype)


@pytest.mark.cpu_only
class PrefusedFfnActivationTest(parameterized.TestCase):
  """`_prefused_ffn_activation` gives the split activation's value bit for bit and its VJP value for value."""

  @parameterized.product(act=("swiglu", "gpt_oss"), dtype=("bfloat16", "float32"))
  def test_value_and_vjp_match_split(self, act, dtype):
    act_fn = {"swiglu": _swiglu, "gpt_oss": _gpt_oss}[act]
    h01 = (jax.random.normal(jax.random.PRNGKey(1), (64, 2 * 48), jnp.float32) * 4).astype(dtype)
    cot = jax.random.normal(jax.random.PRNGKey(2), (64, 48), jnp.float32).astype(dtype)
    cot = cot.at[0].set(-0.0)  # -0.0 cotangents must come back with the sign the split path gives them

    def split(h):
      n = h.shape[-1] // 2
      return act_fn(h[:, :n], h[:, n:])

    @jax.jit
    def both(h, c):
      y_split, vjp_split = jax.vjp(split, h)
      y_fused, vjp_fused = jax.vjp(lambda h_: moe._prefused_ffn_activation(act_fn, h_), h)  # pylint: disable=protected-access
      return y_split, vjp_split(c)[0], y_fused, vjp_fused(c)[0]

    y_split, d_split, y_fused, d_fused = both(h01, cot)
    np.testing.assert_array_equal(np.asarray(y_fused).view(np.uint8), np.asarray(y_split).view(np.uint8))
    d_split, d_fused = np.asarray(d_split), np.asarray(d_fused)
    # Equal values everywhere. Bits may differ only in the sign of an exact zero: the split path's gradient is the
    # transpose of two slices, add(pad(d_gate), pad(d_up)), and XLA:CPU computes -0.0 + 0.0 = +0.0 there, while the
    # fused backward's concatenate keeps -0.0.
    np.testing.assert_array_equal(d_fused, d_split)
    bits_differ = d_fused.view(np.uint16 if d_fused.itemsize == 2 else np.uint32) != d_split.view(
        np.uint16 if d_split.itemsize == 2 else np.uint32
    )
    # Zeros come from the -0.0 cotangent row and, in the clipped GPT-OSS form, from clipped entries (derivative 0).
    self.assertTrue(np.all(d_fused[bits_differ] == 0) and np.all(d_split[bits_differ] == 0))


# A mesh with tensor parallelism: the default logical axis rules (cp-as-ep has no tensor axis), expert 4 x tensor 2.
_TP_MESH = {
    "custom_mesh_and_rule": "",
    "ici_context_parallelism": 1,
    "ici_expert_parallelism": 4,
    "ici_tensor_parallelism": 2,
    "per_device_batch_size": 0.5,
}
_OFF_PREFUSED_PATH = (
    ("prefuse", {"prefuse_moe_weights": False}, "prefuse_moe_weights=True"),
    ("dense", {"sparse_matmul": False}, "sparse_matmul=True"),
    ("te_moe_block", {"te_moe_block": True}, "te_moe_block=False"),
    ("vllm_rpa", {"attention": "vllm_rpa"}, "an attention other than vllm_rpa*"),
)
_OFF_RING_RAGGED_SORT = (
    ("dense", {"sparse_matmul": False}, "sparse_matmul=True"),
    ("te_moe_block", {"te_moe_block": True}, "te_moe_block=False"),
    (
        "no_ring",
        {"use_ring_of_experts": False, "num_moe_token_chunks": 1, "ring_of_experts_local_routing": False},
        "use_ring_of_experts=True",
    ),
    ("no_ragged_sort", {"use_ragged_sort": False}, "use_ragged_sort=True"),
)
_REJECTED = (
    [(f"pair_argsort_{name}", "moe_ragged_sort_unique_pair_argsort", o, m) for name, o, m in _OFF_RING_RAGGED_SORT]
    + [
        (
            "pair_argsort_with_count",
            "moe_ragged_sort_unique_pair_argsort",
            {"moe_routing_index_math": "count"},
            "moe_routing_index_math='sort'",
        )
    ]
    + [(f"unsplit_wi_{name}", "moe_unsplit_prefused_wi", o, m) for name, o, m in _OFF_PREFUSED_PATH]
    + [
        (f"unsplit_wi_{name}", "moe_unsplit_prefused_wi", o, m)
        for name, o, m in (
            ("quantization", {"quantization": "int8"}, "no weight quantization"),
            ("qwix", {"use_qwix_quantization": True}, "no weight quantization"),
            (
                "two_stage_ag",
                {"moe_fsdp_use_two_stage_all_gather": True},
                "moe_fsdp_use_two_stage_all_gather=False",
            ),
            ("flat_fsdp", {"moe_flat_fsdp_weights": True}, "moe_flat_fsdp_weights=False"),
            ("shard_exp_on_fsdp", {"shard_exp_on_fsdp": True}, "shard_exp_on_fsdp=False"),
            ("2d_fsdp", {"use_2d_fsdp_sharding": True}, "use_2d_fsdp_sharding=False"),
            ("shard_embed", {"shard_embed_moe_on_fsdp": True}, "shard_embed_moe_on_fsdp=False"),
            ("chunk_wgrad", {"moe_accumulate_chunk_wgrad": True}, "moe_accumulate_chunk_wgrad=False"),
            ("emb_chunks", {"num_moe_emb_chunks": 2}, "num_moe_emb_chunks=0"),
            ("tensor_parallel", _TP_MESH, "an unsharded expert MLP dimension"),
            (
                "tensor_sequence_parallel",
                {**_TP_MESH, "ici_tensor_parallelism": 1, "ici_tensor_sequence_parallelism": 2},
                "an unsharded expert MLP dimension",
            ),
            (
                "autoregressive_parallel",
                {**_TP_MESH, "ici_tensor_parallelism": 1, "ici_autoregressive_parallelism": 2},
                "an unsharded expert MLP dimension",
            ),
        )
    ]
    + [(f"swiglu_vjp_{name}", "moe_prefused_swiglu_vjp", o, m) for name, o, m in _OFF_PREFUSED_PATH]
    + [
        (f"swiglu_vjp_{name}", "moe_prefused_swiglu_vjp", o, m)
        for name, o, m in (
            ("mlp_bias", {"mlp_bias": True}, "mlp_bias=False"),
            ("emb_chunks", {"num_moe_emb_chunks": 2}, "num_moe_emb_chunks=0"),
            ("minimal_remat", {"remat_policy": "minimal"}, "remat_policy=minimal saves it as mlpwi_0"),
            ("offloaded_remat", {"remat_policy": "minimal_offloaded"}, "remat_policy=minimal_offloaded saves it as"),
            ("context_remat", {"remat_policy": "minimal_with_context"}, "remat_policy=minimal_with_context saves it as"),
            ("flash_remat", {"remat_policy": "minimal_flash"}, "remat_policy=minimal_flash saves it as"),
            (
                "quantization_remat",
                {"remat_policy": "minimal_with_quantization"},
                "remat_policy=minimal_with_quantization saves it as",
            ),
            (
                "context_quantization_remat",
                {"remat_policy": "minimal_with_context_and_quantization"},
                "remat_policy=minimal_with_context_and_quantization saves it as",
            ),
            ("saved_gate", {"remat_policy": "custom", "mlpwi_0": "device"}, "remat_policy=custom saves it as mlpwi_0"),
            ("saved_moe_gate", {"remat_policy": "custom", "moe_mlpwi_0": "device"}, "saves it as moe_mlpwi_0"),
            ("offloaded_moe_up", {"remat_policy": "custom", "moe_mlpwi_1": "offload"}, "saves it as moe_mlpwi_1"),
        )
    ]
)


class GlueRewritesConfigTest(parameterized.TestCase):
  """Each key is rejected for each requirement it does not meet, and accepted on the path it rewrites."""

  @parameterized.named_parameters(*_REJECTED)
  def test_rejected(self, key, overrides, requirement):
    overrides = {"prefuse_moe_weights": True, **overrides}
    with self.assertRaisesRegex(ValueError, f"{key}=True requires: .*{re.escape(requirement)}"):
      _cfg(**{key: True}, **overrides)

  def test_accepted(self):
    cfg = _cfg(prefuse_moe_weights=True, remat_policy="custom", **{k: True for k in _KEYS})
    self.assertTrue(all(getattr(cfg, k) for k in _KEYS))

  def test_accepted_where_the_policy_saves_only_other_tensors(self):
    """Policies that save no routed-expert gate/up tensor: full, a dense-only name, the shared expert's mlpwi_1."""
    for overrides in ({"remat_policy": "full"}, {"remat_policy": "custom", "mlpwi_1": "device", "mlpwo": "device"}):
      self.assertTrue(_cfg(prefuse_moe_weights=True, moe_prefused_swiglu_vjp=True, **overrides).moe_prefused_swiglu_vjp)


def _cfg(**overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh (moe_roe_local_routing_test.py's, local routing)."""
  kwargs = {
      "run_name": "moe_glue_rewrites",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
      "model_name": "qwen3.5-397b-a17b",
      "override_model_config": True,
      "num_experts": _NUM_EXPERTS,
      "num_experts_per_tok": _TOP_K,
      "base_emb_dim": _EMB,
      "base_mlp_dim": 256,
      "base_moe_mlp_dim": 256,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      "per_device_batch_size": 0.25,
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "float32_gate_logits": True,
      "megablox": True,
      "sparse_matmul": True,
      # Tokamax gmm v2 is TPU-only; both programs share whichever GMM runs, so CPU uses megablox.
      "use_tokamax_gmm": False,
      "use_gmm_v2": False,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "use_custom_sort_vjp": False,
      "ragged_buffer_factor": 2.0,
      "num_moe_token_chunks": 2,
      "custom_mesh_and_rule": "cp-as-ep",
      "ici_fsdp_parallelism": 1,
      "ici_context_parallelism": 4,
      "ici_expert_parallelism": 2,
      "context_parallel_load_balance": False,
      "ring_of_experts_local_routing": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def _argsort_ties_reversed(keys, axis=-1):
  """Deliberately wrong: a two-operand pair sort that breaks ties by DEcreasing index."""
  dim = axis % keys.ndim
  iota = jax.lax.broadcasted_iota(jnp.int32, keys.shape, dim)
  _, rev = jax.lax.sort((keys, -iota), dimension=dim, is_stable=False, num_keys=2)
  return -rev


@pytest.mark.cpu_only
def test_glue_rewrites_match_plain_program_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {_XLA_FLAGS}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _build(cfg, mesh):
  return moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=cfg.moe_mlp_dim,
      weight_dtype=cfg.weight_dtype,
      dtype=cfg.dtype,
      rngs=nnx.Rngs(params=0),
  )


def _inputs(cfg, mode):
  """The layer input, forced expert ids (padded and duplicate slots) and a token mask, per `mode`."""
  batch = int(cfg.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  forced = token_mask = None
  if "forced" in mode:
    keys = jax.random.split(jax.random.PRNGKey(11), batch * _SEQ)
    ids = jax.vmap(lambda k: jax.random.permutation(k, _NUM_EXPERTS)[:_TOP_K])(keys).reshape(batch, _SEQ, _TOP_K)
    pos = jnp.arange(_SEQ)[None, :, None]
    ids = jnp.where((pos % 5 == 0) & (jnp.arange(_TOP_K) == _TOP_K - 1), -1, ids)
    ids = jnp.where((pos % 7 == 3) & (jnp.arange(_TOP_K) == 1), ids[..., :1], ids)
    forced = ids.astype(jnp.int32)
  if "padmask" in mode:
    lens = jnp.array([_SEQ - 5 * (b + 1) for b in range(batch)])
    token_mask = (jnp.arange(_SEQ)[None, :] < lens[:, None]).astype(jnp.int32)
  return x, forced, token_mask


def _eqns(jaxpr):
  """Every equation of a jaxpr, including those of nested jaxprs."""
  for eqn in jaxpr.eqns:
    yield eqn
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          yield from _eqns(sub)


def _structure(jaxpr):
  eqns = list(_eqns(jaxpr))
  return {
      "concatenate": sum(e.primitive.name == "concatenate" for e in eqns),
      "barrier": sum(e.primitive.name == "optimization_barrier" for e in eqns),
      "pair_sort": sum(e.primitive.name == "sort" and e.params.get("num_keys") == 2 for e in eqns),
      "custom_vjp": sum(e.primitive.name.startswith("custom_vjp_call") for e in eqns),
  }


def _run(cfg, mesh, params, x, forced, token_mask):
  """Loss, output, gradients, the structure of the traced forward, and the unique_pair_argsort flag of every
  ragged_gather_reduce call traced."""
  pair_flags = []
  real_gather_reduce = ragged_sort.ragged_gather_reduce

  def spy(*args, **kwargs):
    pair_flags.append(kwargs.get("unique_pair_argsort", False))
    return real_gather_reduce(*args, **kwargs)

  with (
      mock.patch.object(ragged_sort, "ragged_gather_reduce", spy),
      jax.set_mesh(mesh),
      nn_partitioning.axis_rules(cfg.logical_axis_rules),
  ):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(params, x):
      out, lb_loss, _ = nnx.merge(graphdef, params, rest)(x, forced_routed_experts=forced, token_mask=token_mask)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    structure = _structure(jax.make_jaxpr(loss_fn)(params, x).jaxpr)
    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
  structure["gather_reduce_pair_flags"] = sorted(set(pair_flags))
  return jax.device_get((loss, out, grads)), structure


def _bit_identical(label, old, new):
  """Prints the per-leaf report; True iff the output, the loss and every gradient are bit-identical."""
  (loss_old, out_old, grads_old), (loss_new, out_new, grads_new) = old, new
  named = [("output", out_old, out_new), ("loss", loss_old, loss_new)] + [
      (jax.tree_util.keystr(path), a, b)
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(grads_old)[0], jax.tree_util.tree_leaves(grads_new))
  ]
  passed, lines = True, []
  for name, a, b in named:
    a, b = np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)
    same = bool(np.array_equal(a, b)) and bool(np.isfinite(b).all()) and float(np.max(np.abs(a))) > 0
    passed &= same
    lines.append(f"  [{'ok' if same else 'FAIL'}] {name}: bit-identical={same} max|d|={np.max(np.abs(a - b)):.3e}")
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _case(label, keys, overrides, mode="real", control=None):
  """Runs the plain program and the one with `keys` on, on the same parameters; `control` wraps the second run."""
  overrides = {"prefuse_moe_weights": True, **overrides}
  cfg_old = _cfg(**overrides)
  cfg_new = _cfg(**overrides, **{k: True for k in keys})
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x, forced, token_mask = _inputs(cfg_old, mode)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, s_old = _run(cfg_old, mesh, params, x, forced, token_mask)
  with control if control is not None else contextlib.nullcontext():
    new, s_new = _run(cfg_new, mesh, params, x, forced, token_mask)
  n_chunks = cfg_new.num_moe_token_chunks
  want = {}
  if "moe_unsplit_prefused_wi" in keys:
    want["concatenate"] = s_new["concatenate"] < s_old["concatenate"]
    if cfg_new.fp8_moe_fake_quant:
      want["barrier"] = s_new["barrier"] < s_old["barrier"]
  if "moe_ragged_sort_unique_pair_argsort" in keys:
    # Both argsorts of every chunk's ragged sort (the forward traced here), and the backward's gather-reduce flag.
    want["pair_sort"] = s_new["pair_sort"] == 2 * n_chunks and s_old["pair_sort"] == 0
    flags = (s_new["gather_reduce_pair_flags"], s_old["gather_reduce_pair_flags"])
    want["gather_reduce_pair_flags"] = flags == ([True], [False])
  if "moe_prefused_swiglu_vjp" in keys:
    want["custom_vjp"] = s_new["custom_vjp"] == s_old["custom_vjp"] + n_chunks
  structure = all(want.values())
  print(f"{label}: [{'ok' if structure else 'FAIL'}] structure plain={s_old} new={s_new} checks={want}", flush=True)
  return _bit_identical(label, old, new) and structure


def _swapped_activation_cotangents():
  """Deliberately wrong: the fused activation's backward returns [d(up) | d(gate)]."""
  real = moe._prefused_ffn_activation  # pylint: disable=protected-access

  def wrong(act_fn, h01):
    n = h01.shape[-1] // 2
    swapped = jnp.concatenate([h01[..., n:], h01[..., :n]], axis=-1)
    # Same forward value (act_fn reads the halves in the right order), gradient of the halves exchanged.
    return real(lambda up, gate: act_fn(gate, up), jax.lax.stop_gradient(swapped) + (h01 - jax.lax.stop_gradient(h01)))

  return mock.patch.object(moe, "_prefused_ffn_activation", wrong)


def _ties_reversed_pair_argsort():
  return mock.patch.object(ragged_sort, "argsort_unique_pairs", _argsort_ties_reversed)


class _ConfigOverride:
  """`config` with some attributes replaced: reaches the layer's trace-time checks past the config validator."""

  def __init__(self, config, **overrides):
    self._config = config
    self._overrides = overrides

  def __getattr__(self, name):  # only called for names that are not set on the instance
    if name.startswith("__") or "_config" not in self.__dict__:
      raise AttributeError(name)
    return self._overrides[name] if name in self._overrides else getattr(self._config, name)


def _unsplit_wi_tensor_parallel_guard():
  """On a tensor-parallel mesh the layer refuses moe_unsplit_prefused_wi at trace time; the plain layer runs."""
  cfg = _cfg(prefuse_moe_weights=True, **_TP_MESH)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  x, forced, token_mask = _inputs(cfg, "real")
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg, mesh), nnx.Param, ...)
  (loss, _, _), _ = _run(cfg, mesh, params, x, forced, token_mask)
  plain_runs = bool(np.isfinite(loss))
  try:
    _run(_ConfigOverride(cfg, moe_unsplit_prefused_wi=True), mesh, params, x, forced, token_mask)
  except ValueError as e:
    raised = "the MLP dimension sharded 2-way in the expert shard_map" in str(e)
  else:
    raised = False
  ok = plain_runs and raised
  print(
      f"unsplit_wi_tensor_parallel_guard: [{'ok' if ok else 'FAIL'}] mesh {dict(mesh.shape)}: plain layer runs:"
      f" {plain_runs}; moe_unsplit_prefused_wi past the validator raises at trace time: {raised}",
      flush=True,
  )
  return ok


# The subset of the Qwen3.5-397B RL trainer benchmark's MoE settings that this file needs (the cases above do not
# all set them).
_BENCH_FLAGS = {
    "moe_chunk_pipeline": True,
    "moe_expert_weight_prefetch": "forward",
    "ring_of_experts_row_major_reduce_scatter": True,
    "moe_routing_maps": "device",
    "ragged_buffer_factor": 2.75,
    "fp8_moe_fake_quant": True,
    "moe_exclude_padding_from_routing": True,
}


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  wi, pairs, swiglu = _KEYS
  pipelined = {"moe_chunk_pipeline": True}
  fake_quant = {**pipelined, "fp8_moe_fake_quant": True, "moe_exclude_padding_from_routing": True}
  results = {
      "unsplit_wi_plain_loop": _case("unsplit_wi_plain_loop", (wi,), {}),
      "unsplit_wi_fake_quant": _case("unsplit_wi_fake_quant", (wi,), fake_quant, mode="padmask"),
      "pair_argsort_pipelined": _case("pair_argsort_pipelined", (pairs,), pipelined),
      "pair_argsort_dropping": _case(
          "pair_argsort_dropping", (pairs,), {**pipelined, "ragged_buffer_factor": 1.0}, mode="forced"
      ),
      "swiglu_vjp_pipelined": _case("swiglu_vjp_pipelined", (swiglu,), pipelined),
      "all_three_plain_loop": _case("all_three_plain_loop", _KEYS, {}),
      "all_three_fake_quant": _case("all_three_fake_quant", _KEYS, fake_quant, mode="padmask"),
      "all_three_dropless": _case("all_three_dropless", _KEYS, {"ragged_buffer_factor": -1.0}, mode="forced"),
      "wi_and_swiglu_with_count": _case(
          "wi_and_swiglu_with_count", (wi, swiglu), {**pipelined, "moe_routing_index_math": "count"}
      ),
      "all_three_bench_flags": _case("all_three_bench_flags", _KEYS, _BENCH_FLAGS, mode="padmask"),
      "unsplit_wi_tensor_parallel_guard": _unsplit_wi_tensor_parallel_guard(),
  }
  controls = {
      "control_swapped_activation_cotangents": _case(
          "control_swapped_activation_cotangents", (swiglu,), pipelined, control=_swapped_activation_cotangents()
      ),
      "control_ties_reversed_under_drops": _case(
          "control_ties_reversed_under_drops",
          (pairs,),
          {**pipelined, "ragged_buffer_factor": 1.0},
          mode="forced",
          control=_ties_reversed_pair_argsort(),
      ),
  }
  print("SUMMARY", {**results, **controls}, flush=True)
  failed = [k for k, v in results.items() if not v]
  undetected = [k for k, v in controls.items() if v]
  if failed or undetected:
    print(f"cases that failed: {failed}; positive controls that were NOT detected: {undetected}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
