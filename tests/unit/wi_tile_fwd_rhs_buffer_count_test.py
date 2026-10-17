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

"""wi_tile_fwd_rhs_buffer_count: the weight (rhs) pipeline depth of the fused wi forward GMM (gmm_v2 rhs_buffer_count).

Buffering only: the kernel computes the same values at any depth. A depth of 2 frees one [gate | up] weight tile of
VMEM, which lets wi_tile_fwd_mlp_dim=2048 fit at wi_tile_fwd_batch_seq=512 on TPU v7x.

On CPU:
- The config key: 0 by default (the kernel default); accepted at 2 and 3 next to moe_wi_gmm_fused_swiglu with explicit
  tiles; rejected, naming the key, at 1 and below, without moe_wi_gmm_fused_swiglu, and with heuristic tiling.
- make_gmm_configs under a TPU v7x target (no kernel runs): None leaves the config field, the metadata and the kernel
  name as they are without the option, and the weight BlockSpec at 3 buffers; a depth of N sets the field, the
  metadata, the name suffix -rhsbuf_N and Buffered(buffer_count=N); a depth below 2 raises.
- ops.gmm_silu lowers for a TPU v7x on this host (a lowering, not a compile), with and without fuse_bwd_dswiglu:
  fwd_rhs_buffer_count reaches the forward kernel only; the backward kernels are the same as without it.
- The MoE layer, in a child process with 8 CPU devices, with moe_wo_gmm_fused_dswiglu_test.py's jnp stand-ins for the
  two Pallas kernels (so moe.py and ops.py are the real code): with the key at 2 and one K tile in the wi forward GMM,
  every fused wi forward call gets rhs_buffer_count=2 and no other gmm_v2 call gets it; with a split K tile, or the
  key at 0, no call gets it; the output, loss and gradients are bit-identical to the key at 0 (the stand-in ignores
  buffering, so this checks that the layer computes the same thing, not the kernel).
On TPU only (Pallas TPU kernels): gmm_v2's fused forward writes the same act and [gate | up] bits at depths 2 and 4 as
at the default, with one and two n tiles per half and short groups.
"""

import collections
import functools
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

from maxtext.kernels.megablox import ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.utils import maxtext_utils
from tests.unit import moe_wo_gmm_fused_dswiglu_test as layer_setup

_KEY = "wi_tile_fwd_rhs_buffer_count"
_PASS_TOKEN = "WI_TILE_FWD_RHS_BUFFER_COUNT_CHECKS_PASSED"
_ABSENT = "absent"  # gmm_v2 called without rhs_buffer_count: the default kernel, unchanged
_GROUP_SIZES = np.array([60, 0, 100, 40], np.int32)  # sum 200 < 256: padding rows; boundaries inside sublanes
_M, _K, _NH = 256, 256, 256  # the CPU cases' fused forward: lhs [M, K], rhs [groups, K, 2 * NH] = [gate | up]


def _tpu_target(device_kind="TPU7x"):
  """A context in which traces target a TPU of `device_kind`, as in an AOT compile on a CPU host."""
  device = jax.sharding.AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  return jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=device))


def _buffering(rhs_buffer_count):
  """gmm_v2's keyword for a depth: none at all for the default, so that call is exactly the one without the option."""
  return {} if rhs_buffer_count is None else {"rhs_buffer_count": rhs_buffer_count}


class ConfigKeyTest(parameterized.TestCase):
  """The key is accepted only where the fused wi forward GMM reads it (shrunk Qwen3.5 routed-MoE config)."""

  def test_default_is_the_kernel_default(self):
    self.assertEqual(getattr(layer_setup._cfg(), _KEY), 0)  # pylint: disable=protected-access

  @parameterized.named_parameters(("two", 2), ("three", 3))
  def test_accepted_next_to_the_fused_wi_gmm(self, depth):
    cfg = layer_setup._cfg(moe_wi_gmm_fused_swiglu=True, **{_KEY: depth})  # pylint: disable=protected-access
    self.assertEqual(getattr(cfg, _KEY), depth)

  @parameterized.named_parameters(
      ("one", {_KEY: 1}, "a value of 0 (the kernel default) or at least 2"),
      ("negative", {_KEY: -1}, "a value of 0 (the kernel default) or at least 2"),
      ("unfused_wi_gmm", {_KEY: 2, "moe_wi_gmm_fused_swiglu": False}, "moe_wi_gmm_fused_swiglu=True"),
      ("heuristic_tiling", {_KEY: 2, "use_gmm_v2_heuristic_tiling": True}, "use_gmm_v2_heuristic_tiling=False"),
  )
  def test_rejected(self, overrides, requirement):
    kwargs = {"moe_wi_gmm_fused_swiglu": True, **overrides}
    with self.assertRaisesRegex(ValueError, f"{_KEY}={overrides[_KEY]} requires: .*{re.escape(requirement)}"):
      layer_setup._cfg(**kwargs)  # pylint: disable=protected-access


def _configs(rhs_buffer_count):
  """make_gmm_configs for the fused SwiGLU forward with [gate | up] (moe_wo_gmm_fused_dswiglu's), TPU v7x target."""
  lhs = jax.ShapeDtypeStruct((_M, _K), jnp.bfloat16)
  rhs = jax.ShapeDtypeStruct((len(_GROUP_SIZES), _K, 2 * _NH), jnp.bfloat16)
  with _tpu_target():
    return gmm_v2.make_gmm_configs(
        lhs,
        rhs,
        None,
        None,
        None,
        jnp.asarray(_GROUP_SIZES),
        jnp.array([0], jnp.int32),
        tile_info=gmm_v2.TileSizes(tile_m=128, tile_k=_K, tile_n=_NH),
        vmem_limit_bytes=None,
        out_dtype=jnp.bfloat16,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        fuse_act="silu",
        return_pre_act="concat",
        **_buffering(rhs_buffer_count),
    )


def _weight_buffer_count(cfgs):
  """The pipeline depth of the weight BlockSpec the kernel is built from (no index map is called)."""
  with _tpu_target():
    (_, weights, _), _ = gmm_v2.generate_block_specs(None, cfgs)
  return weights.weight.pipeline_mode.buffer_count


@pytest.mark.cpu_only
class KernelConfigTest(parameterized.TestCase):
  """gmm_v2's rhs_buffer_count from make_gmm_configs to the weight BlockSpec, the metadata and the kernel name."""

  def test_default_is_the_kernel_without_the_option(self):
    default, two = _configs(None), _configs(2)
    self.assertIsNone(default.rhs_buffer_count)
    self.assertNotIn("rhs_buffer_count", gmm_v2.get_metadata(default))
    self.assertNotIn("rhsbuf", gmm_v2.get_scope_name(default))
    self.assertEqual(_weight_buffer_count(default), 3)  # the kernel's default for a 2D lhs
    # The option changes exactly these three, and the name only by its suffix.
    self.assertEqual(gmm_v2.get_scope_name(two), gmm_v2.get_scope_name(default) + "-rhsbuf_2")
    metadata = gmm_v2.get_metadata(two)
    self.assertEqual(metadata.pop("rhs_buffer_count"), 2)
    self.assertEqual(metadata, gmm_v2.get_metadata(default))

  @parameterized.parameters(2, 3, 4)
  def test_depth_reaches_the_weight_block_spec(self, depth):
    cfgs = _configs(depth)
    self.assertEqual(cfgs.rhs_buffer_count, depth)
    self.assertEqual(_weight_buffer_count(cfgs), depth)
    self.assertTrue(gmm_v2.get_scope_name(cfgs).endswith(f"-rhsbuf_{depth}"))
    self.assertEqual(gmm_v2.get_metadata(cfgs)["rhs_buffer_count"], depth)

  @parameterized.parameters(1, 0, -1, 2.0, True)
  def test_below_two_or_not_an_int_rejected(self, depth):
    # One buffer would serialize every weight DMA with the matmul that reads it.
    with self.assertRaisesRegex(ValueError, "rhs_buffer_count must be None or an int >= 2"):
      _configs(depth)


def _kernel_names(fn, *args):
  with _tpu_target():
    lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
  return re.findall(r'kernel_name = "([^"]+)"', lowered.as_text())


@pytest.mark.cpu_only
class FusedForwardLoweringTest(parameterized.TestCase):
  """ops.gmm_silu lowers for a TPU v7x on this host (a lowering, not a compile): only its forward takes the depth."""

  _LHS = jax.ShapeDtypeStruct((_M, _K), jnp.bfloat16)
  _RHS = jax.ShapeDtypeStruct((len(_GROUP_SIZES), _K, 2 * _NH), jnp.bfloat16)
  _GS = jax.ShapeDtypeStruct((len(_GROUP_SIZES),), jnp.int32)
  # gmm's 9-tuple: the forward (gmm_silu halves its tile_n), then the DLHS and the TGMM, as the layer passes it.
  _TILING = (128, _K, 2 * _NH, 128, _K, _K, 128, _K, 2 * _NH)

  def _fwd_and_bwd_names(self, fuse_bwd_dswiglu, fwd_rhs_buffer_count):
    def fwd_and_bwd(lhs, rhs, gs):
      def f(l, r):
        return ops.gmm_silu(
            l,
            r,
            gs,
            jnp.bfloat16,
            self._TILING,
            fuse_bwd_dswiglu=fuse_bwd_dswiglu,
            fwd_rhs_buffer_count=fwd_rhs_buffer_count,
        )

      out, vjp = jax.vjp(f, lhs, rhs)
      return out, vjp(jax.tree.map(jnp.ones_like, out))

    return _kernel_names(fwd_and_bwd, self._LHS, self._RHS, self._GS)

  @parameterized.named_parameters(("unfused_backward", False), ("fused_dswiglu_backward", True))
  def test_only_the_forward_kernel_takes_the_depth(self, fuse_bwd_dswiglu):
    default = self._fwd_and_bwd_names(fuse_bwd_dswiglu, None)
    names = self._fwd_and_bwd_names(fuse_bwd_dswiglu, 2)
    forward = [n for n in names if "-act_silu-" in n]
    self.assertLen(forward, 1, names)
    self.assertTrue(forward[0].endswith("-rhsbuf_2"), forward)
    self.assertLen(names, 3, names)  # the forward, gmm's DLHS and the TGMM
    self.assertEqual([n for n in names if "rhsbuf" in n], forward)
    # Without the depth: the same kernels, without the suffix.
    self.assertNotIn("rhsbuf", " ".join(default))
    self.assertEqual([n.replace("-rhsbuf_2", "") for n in names], default)

  def test_act_only_forward(self):
    """gmm_silu when not differentiated: one kernel, the act-only forward, with the depth."""

    def act_only(lhs, rhs, gs, depth):
      return ops.gmm_silu(lhs, rhs, gs, jnp.bfloat16, self._TILING, fwd_rhs_buffer_count=depth)

    names = _kernel_names(functools.partial(act_only, depth=2), self._LHS, self._RHS, self._GS)
    default = _kernel_names(functools.partial(act_only, depth=None), self._LHS, self._RHS, self._GS)
    self.assertLen(names, 1)
    self.assertEqual(names[0], default[0] + "-rhsbuf_2")


@pytest.mark.cpu_only
def test_layer_passes_the_depth_to_the_fused_wi_forward_only():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {layer_setup._XLA_FLAGS}"  # pylint: disable=protected-access
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _same_bits(a, b):
  """Bit-identical (so -0.0 != +0.0), same shape and dtype."""
  a, b = np.asarray(a), np.asarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
  return bool(np.array_equal(a.view(unsigned), b.view(unsigned)))


def _layer_case(label, overrides, want_fused_forward, mesh, params, x, token_mask, seen):
  """The layer with `overrides`; every fused wi forward call must get `want_fused_forward`, every other call nothing."""
  seen.clear()
  cfg = layer_setup._cfg(**overrides)  # pylint: disable=protected-access
  result, traced = layer_setup._run(  # pylint: disable=protected-access
      cfg, mesh, params, x, token_mask, layer_setup._unfused_rounding_epilogue  # pylint: disable=protected-access
  )
  calls = dict(seen)
  fused_forward = {depth for (kind, depth) in calls if kind == "fused_wi_forward"}
  others = {depth for (kind, depth) in calls if kind != "fused_wi_forward"}
  n_fused_forward = sum(n for (kind, _), n in calls.items() if kind == "fused_wi_forward")
  loss = float(np.asarray(result[0]))
  ok = (
      fused_forward == {want_fused_forward}
      and others == {_ABSENT}
      and n_fused_forward >= cfg.num_moe_token_chunks
      and np.isfinite(loss)
      and loss > 0
  )
  print(f"{label}: [{'ok' if ok else 'FAIL'}] gmm_v2 calls (kind, rhs_buffer_count) -> count: {calls}; traced {traced}",
        flush=True)
  return ok, result


def main():
  assert jax.device_count() == layer_setup._NUM_DEVICES, jax.devices()  # pylint: disable=protected-access
  seen = collections.Counter()
  stand_in = layer_setup._gmm_v2_stand_in  # pylint: disable=protected-access

  def recording_stand_in(*args, **kwargs):
    if kwargs.get("fuse_act") is not None:
      kind = "fused_wi_forward"
    elif kwargs.get("fuse_dact") is not None:
      kind = "fused_dswiglu_dlhs"
    else:
      kind = "other"
    seen[(kind, kwargs.get("rhs_buffer_count", _ABSENT))] += 1
    return stand_in(*args, **kwargs)

  # The bench cell's MoE keys that run on CPU, with both fused SwiGLU GMMs.
  measured = {
      "moe_chunk_pipeline": True,
      "moe_router_grad_from_wo_input": True,
      "moe_exclude_padding_from_routing": True,
      "moe_routing_index_math": "count",
      "moe_unsplit_prefused_wi": True,
      "moe_roe_packed_routing_gather": True,
      "moe_gmm_v2_dlhs_transpose_rhs": True,
      "moe_wi_gmm_fused_swiglu": True,
      "moe_wo_gmm_fused_dswiglu": True,
  }
  emb = layer_setup._EMB  # pylint: disable=protected-access
  cfg0 = layer_setup._cfg(**measured)  # pylint: disable=protected-access
  # One K tile in the wi forward GMM: its embed-dim tile covers the embed dim.
  assert cfg0.wi_tile_fwd_embed_dim >= emb, (cfg0.wi_tile_fwd_embed_dim, emb)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg0), cfg0.mesh_axes)
  x, token_mask = layer_setup._inputs(cfg0, "padmask")  # pylint: disable=protected-access
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg0.logical_axis_rules):
    _, params, _ = nnx.split(layer_setup._build(cfg0, mesh), nnx.Param, ...)  # pylint: disable=protected-access
  cases = {
      # label: (overrides, what every fused wi forward call must get)
      "key_2_one_k_tile": ({**measured, _KEY: 2}, 2),
      "key_3_one_k_tile": ({**measured, _KEY: 3}, 3),
      # Without the fused SwiGLU backward (gmm_silu's other custom VJP, which also saves gate and up).
      "key_2_one_k_tile_unfused_backward": ({**measured, "moe_wo_gmm_fused_dswiglu": False, _KEY: 2}, 2),
      # A split K tile: the weight block changes every grid step, so the layer leaves the kernel's default.
      "key_2_split_k": ({**measured, _KEY: 2, "wi_tile_fwd_embed_dim": emb // 2}, _ABSENT),
      "key_0": (measured, _ABSENT),
  }
  results, values = {}, {}
  with mock.patch.object(layer_setup, "_gmm_v2_stand_in", recording_stand_in):
    for label, (overrides, want) in cases.items():
      results[label], values[label] = _layer_case(label, overrides, want, mesh, params, x, token_mask, seen)
  # Buffering only: the same output, loss and gradients (the stand-in ignores buffering, so this checks the layer).
  for label in ("key_2_one_k_tile", "key_2_split_k"):
    leaves, ref = jax.tree_util.tree_leaves(values[label]), jax.tree_util.tree_leaves(values["key_0"])
    same = len(leaves) == len(ref) and all(_same_bits(a, b) for a, b in zip(leaves, ref))
    print(f"{label} vs key_0: output, loss and gradients bit-identical: {same}", flush=True)
    results[f"{label}_same_values"] = same
  print("SUMMARY", results, flush=True)
  if not all(results.values()):
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


# ------------------------------------------------------------------------------------------------------------------
# TPU only.
# ------------------------------------------------------------------------------------------------------------------
_TPU_M, _TPU_K, _TPU_NH = 1024, 512, 256


def _fused_forward(lhs, rhs, group_sizes, tile_m, tile_n, rhs_buffer_count):
  """gmm_v2's fused SwiGLU forward with one K tile: (act, [gate | up]), as moe_wo_gmm_fused_dswiglu's forward."""
  fn = functools.partial(
      gmm_v2.gmm_v2,
      tile_info=gmm_v2.TileSizes(tile_m, lhs.shape[1], tile_n),
      preferred_element_type=jnp.bfloat16,
      fuse_act="silu",
      return_pre_act="concat",
      **_buffering(rhs_buffer_count),
  )
  return jax.jit(fn)(lhs, rhs, jnp.asarray(group_sizes, jnp.int32))


def _tpu_operands(num_groups, seed=7):
  k0, k1 = jax.random.split(jax.random.key(seed))
  lhs = jax.random.normal(k0, (_TPU_M, _TPU_K), jnp.float32).astype(jnp.bfloat16)
  rhs = (jax.random.normal(k1, (num_groups, _TPU_K, 2 * _TPU_NH), jnp.float32) * 0.05).astype(jnp.bfloat16)
  return lhs, rhs


def _bits(x):
  return np.asarray(jax.device_get(x)).view(np.uint16)


@pytest.mark.tpu_only
class SameValuesTpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")

  @parameterized.named_parameters(
      # The bench cell's shape class: one n tile per half; padding rows; group ends inside m tiles.
      ("one_n_tile_per_half", [300, 0, 517, 133], 256, _TPU_NH),
      # The weight block changes on most grid steps: two n tiles per half, a 1-row group, short groups.
      ("two_n_tiles_per_half_short_groups", [40, 0, 300, 17, 1, 200, 0, 395], 128, _TPU_NH // 2),
  )
  def test_fused_forward_writes_the_same_bits(self, group_sizes, tile_m, tile_n):
    lhs, rhs = _tpu_operands(len(group_sizes))
    act, gate_up = _fused_forward(lhs, rhs, group_sizes, tile_m, tile_n, None)
    self.assertTrue(np.isfinite(np.asarray(act, np.float32)).all())
    self.assertGreater(float(np.max(np.abs(np.asarray(gate_up, np.float32)))), 0.0)
    for depth in (2, 4):
      act_d, gate_up_d = _fused_forward(lhs, rhs, group_sizes, tile_m, tile_n, depth)
      np.testing.assert_array_equal(_bits(act_d), _bits(act), err_msg=f"act, rhs_buffer_count={depth}")
      np.testing.assert_array_equal(_bits(gate_up_d), _bits(gate_up), err_msg=f"[gate | up], rhs_buffer_count={depth}")


if __name__ == "__main__":
  main()
