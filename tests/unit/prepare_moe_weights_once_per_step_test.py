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
"""CPU mesh verification for `prepare_moe_weights_once_per_step` (W3a).

Verifies on an 8-device CPU mesh (`fsdp=2, context=2, expert=2`):
1. `prepare_moe_weights_once_per_step` defaults to `False`.
2. With `prefuse_moe_weights=True`, `fp8_moe_fake_quant=True`, `moe_expert_weight_prefetch="forward_backward"`,
   and `jax.checkpoint` remat, `prepare_moe_weights_once_per_step=True` produces bit-identical forward
   outputs, loss, activation gradients, and parameter gradients (`wi`, `wo`, `gate`) compared to
   `prepare_moe_weights_once_per_step=False`.
3. Hoisting weight preparation via `moe.prepare_moe_weights_for_step` before remat eliminates the
   rematerialized backward `f8e4m3fn` fake-quantization conversions and passes `wi` through FSDP
   all-gather / `barrier_tie` / `shard_map` without splitting into `(w0, w1)` and re-concatenating.
"""

import os
import subprocess
import sys

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
from tests.unit import moe_chunk_pipeline_test as layer_setup

_PASS_TOKEN = "PREPARE_MOE_WEIGHTS_ONCE_PER_STEP_CHECKS_PASSED"


@pytest.mark.cpu_only
def test_prepare_moe_weights_once_per_step_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {layer_setup._XLA_FLAGS}"  # pylint: disable=protected-access
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  src_root = os.path.join(repo_root, "src")
  env["PYTHONPATH"] = (
      os.pathsep.join([src_root, repo_root, env["PYTHONPATH"]])
      if env.get("PYTHONPATH")
      else os.pathsep.join([src_root, repo_root])
  )
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _make_cfg(**extra):
  return layer_setup._cfg(  # pylint: disable=protected-access
      False,
      ici_fsdp_parallelism=2,
      ici_context_parallelism=2,
      ici_expert_parallelism=2,
      **extra,
  )


def _build_routed_moe(cfg, mesh):
  return moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=linears.nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=cfg.moe_mlp_dim,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      quant=None,
      rngs=nnx.Rngs(params=0, dropout=0),
  )


def _run_step(cfg, mesh, params, x, forced):
  """Runs one jitted forward/backward step; returns loss, output, param/input grads and the lowered HLO."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    layer = _build_routed_moe(cfg, mesh)
    graphdef, _, rest = nnx.split(layer, nnx.Param, ...)

    def loss_fn(p, inp):
      mdl = nnx.merge(graphdef, p, rest, copy=True)
      if getattr(cfg, "prepare_moe_weights_once_per_step", False):
        moe.prepare_moe_weights_for_step(mdl, cfg)
      gdef_inner, p_inner, r_inner = nnx.split(mdl, nnx.Param, ...)

      @jax.checkpoint
      def remat_layer(p_l, r_l, h):
        m_l = nnx.merge(gdef_inner, p_l, r_l)
        out_l, _, _ = m_l(h, forced_routed_experts=forced)
        return out_l

      out = remat_layer(p_inner, r_inner, inp)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      return loss, out

    jit_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))
    hlo = jit_fn.lower(params, x).as_text(dialect="hlo")
    (loss, out), (dparams, dx) = jit_fn(params, x)
    return (
        float(loss),
        np.asarray(out, dtype=np.float32),
        jax.tree.map(lambda g: np.asarray(g, dtype=np.float32), dparams),
        np.asarray(dx, dtype=np.float32),
        hlo,
    )


def main():
  cfg_default = _make_cfg()
  assert cfg_default.prepare_moe_weights_once_per_step is False

  cfg_base = _make_cfg(
      prefuse_moe_weights=True,
      fp8_moe_fake_quant=True,
      moe_expert_weight_prefetch="forward_backward",
      prepare_moe_weights_once_per_step=False,
  )
  cfg_w3a = _make_cfg(
      prefuse_moe_weights=True,
      fp8_moe_fake_quant=True,
      moe_expert_weight_prefetch="forward_backward",
      prepare_moe_weights_once_per_step=True,
  )
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_base), cfg_base.mesh_axes)
  x, forced = layer_setup._inputs(cfg_base, forced_mode=True)  # pylint: disable=protected-access
  x = jnp.concatenate([x, x * jnp.bfloat16(0.5)], axis=0)
  forced = jnp.concatenate([forced, forced], axis=0)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_base.logical_axis_rules):
    _, params, _ = nnx.split(_build_routed_moe(cfg_base, mesh), nnx.Param, ...)

  l_b, out_b, dp_b, dx_b, hlo_b = _run_step(cfg_base, mesh, params, x, forced)
  l_w, out_w, dp_w, dx_w, hlo_w = _run_step(cfg_w3a, mesh, params, x, forced)

  assert l_b == l_w, f"loss mismatch: {l_b} vs {l_w}"
  np.testing.assert_array_equal(out_b, out_w)
  np.testing.assert_array_equal(dx_b, dx_w)
  for (path_b, g_b), (path_w, g_w) in zip(
      jax.tree_util.tree_leaves_with_path(dp_b), jax.tree_util.tree_leaves_with_path(dp_w)
  ):
    np.testing.assert_array_equal(g_b, g_w, err_msg=f"grad mismatch at {path_b} vs {path_w}")

  b_fp8_barriers = [l for l in hlo_b.splitlines() if "opt-barrier(" in l and "f8e4m3fn[" in l]
  w_fp8_barriers = [l for l in hlo_w.splitlines() if "opt-barrier(" in l and "f8e4m3fn[" in l]
  b_wi_concats = [l for l in hlo_b.splitlines() if "concatenate(" in l and "dimensions={2}" in l]
  w_wi_concats = [l for l in hlo_w.splitlines() if "concatenate(" in l and "dimensions={2}" in l]
  print(
      f"FP8 fake-quant opt-barriers: base={len(b_fp8_barriers)}, w3a={len(w_fp8_barriers)}; "
      f"wi dim-2 concats: base={len(b_wi_concats)}, w3a={len(w_wi_concats)}",
      flush=True,
  )
  assert (
      len(b_fp8_barriers) == 6 and len(w_fp8_barriers) == 2
  ), f"expected 6->2 FP8 fake-quant opt-barriers with W3a, got {len(b_fp8_barriers)}->{len(w_fp8_barriers)}"
  assert len(w_wi_concats) < len(
      b_wi_concats
  ), f"expected fewer wi dim-2 concatenations with pre-fused wi in W3a ({len(w_wi_concats)} vs {len(b_wi_concats)})"
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
