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
"""Tests for `gdn_cp_matmul_precision`, the precision of the GDN sequence-sharded CP state composition.

The multi-device checks run in a subprocess on a forced 4-device CPU mesh, because the parent pytest
process has already initialized JAX. XLA:CPU computes every f32 dot in plain f32 whatever its precision,
so these tests check where the precision reaches (the lowered dots) rather than TPU numerics.
"""

import functools
import importlib.util
import os
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pydantic
import pytest

from maxtext.configs import types

_DOT_PRECISION = re.compile(r"stablehlo\.dot_general.*?precision = \[(DEFAULT|HIGH|HIGHEST), \1\]")


def test_config_key_default_and_values():
  assert types.Qwen3Next().gdn_cp_matmul_precision == "highest"
  assert types.Qwen3Next(gdn_cp_matmul_precision="high").gdn_cp_matmul_precision == "high"
  with pytest.raises(pydantic.ValidationError):
    types.Qwen3Next(gdn_cp_matmul_precision="default")


@pytest.mark.cpu_only
def test_cp_precision_reaches_only_the_cp_composition_on_cpu_mesh():
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "GDN_CP_PRECISION_CHECKS_PASSED" in result.stdout


def _dot_precisions(lowered):
  """Returns the precision of every dot_general in the lowered module, in program order."""
  return _DOT_PRECISION.findall(lowered.as_text())


class _MarkCpGdnDots:
  """Stands in for `jnp` inside cp_gdn and gives every matmul it issues Precision.DEFAULT, as a marker."""

  def __init__(self, real):
    self._real = real

  def __getattr__(self, name):
    return getattr(self._real, name)

  def matmul(self, a, b, precision=None, **kw):
    del precision
    return self._real.matmul(a, b, precision=jax.lax.Precision.DEFAULT, **kw)

  def einsum(self, *operands, precision=None, **kw):
    del precision
    return self._real.einsum(*operands, precision=jax.lax.Precision.DEFAULT, **kw)


def _kernel_step(mesh, precision):
  """jit(value_and_grad) of gdn_decoupled_conv1d under sequence-sharded CP, as model_runner calls it."""
  from maxtext.kernels.gdn.gdn_bwd import api as gdn_api  # pylint: disable=import-outside-toplevel

  hk, hv, d, kconv, chunk = 1, 2, 128, 4, 64
  dqkv = 2 * hk * d + hv * d

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(P(None, "context", None),) * 3 + (P(),) * 3,
      out_specs=P(None, "context", None, None),
      check_vma=False,
  )
  def layer(qkv, b, a, conv_w, a_log, dt_bias):
    zeros_cs = jnp.zeros((qkv.shape[0], kconv - 1, dqkv), qkv.dtype)
    zeros_rs = jnp.zeros((qkv.shape[0], hv, d, d), jnp.float32)
    kwargs = {} if precision is None else {"cp_matmul_precision": precision}
    out, _ = gdn_api.gdn_decoupled_conv1d(
        qkv,
        b,
        a,
        conv_w,
        None,
        a_log,
        dt_bias,
        zeros_cs,
        zeros_rs,
        hk,
        hv,
        d,
        d,
        kconv,
        chunk,
        True,
        jnp.float32,
        "context",
        None,
        **kwargs,
    )
    return out

  def loss(*args):
    return jnp.sum(layer(*args).astype(jnp.float32))

  s = 4 * 2 * chunk
  keys = jax.random.split(jax.random.PRNGKey(0), 3)
  seq = NamedSharding(mesh, P(None, "context", None))
  args = (
      jax.device_put(jax.random.normal(keys[0], (1, s, dqkv), jnp.bfloat16), seq),
      jax.device_put(jax.random.normal(keys[1], (1, s, hv), jnp.bfloat16), seq),
      jax.device_put(jax.random.normal(keys[2], (1, s, hv), jnp.bfloat16), seq),
      jnp.full((kconv, 1, dqkv), 0.1, jnp.float32),
      jnp.zeros((hv,), jnp.float32),
      jnp.full((hv,), -4.0, jnp.float32),
  )
  return jax.jit(jax.value_and_grad(loss, argnums=tuple(range(6)))), args


def _check_kernel_path(mesh):
  """Default = HIGHEST; HIGH flips exactly the cp_gdn dots, each from HIGHEST, and nothing else."""
  from maxtext.kernels.gdn.gdn_bwd import cp_gdn  # pylint: disable=import-outside-toplevel

  lowered = {}
  for name, precision in (("omitted", None), ("highest", jax.lax.Precision.HIGHEST), ("high", jax.lax.Precision.HIGH)):
    step, args = _kernel_step(mesh, precision)
    lowered[name] = _dot_precisions(step.lower(*args))
  real_jnp = cp_gdn.jnp
  cp_gdn.jnp = _MarkCpGdnDots(real_jnp)
  try:
    step, args = _kernel_step(mesh, jax.lax.Precision.HIGHEST)
    marked = _dot_precisions(step.lower(*args))
  finally:
    cp_gdn.jnp = real_jnp
  assert lowered["omitted"] == lowered["highest"], "the default must be HIGHEST"
  assert len(lowered["high"]) == len(lowered["highest"]) == len(marked)
  cp_dots = [i for i, (m, h) in enumerate(zip(marked, lowered["highest"])) if m != h]
  flipped = [i for i, (h, l) in enumerate(zip(lowered["highest"], lowered["high"])) if h != l]
  assert cp_dots, "the marker found no cp_gdn dot: the check cannot see the composition"
  assert all(marked[i] == "DEFAULT" and lowered["highest"][i] == "HIGHEST" for i in cp_dots)
  assert flipped == cp_dots, f"flipped {flipped} != cp_gdn dots {cp_dots}"
  assert all(lowered["high"][i] == "HIGH" for i in flipped)
  print(f"kernel path: {len(flipped)} of {len(marked)} dots flipped HIGHEST -> HIGH, exactly the cp_gdn ones")


def _check_config_path(devices):
  """cfg.gdn_cp_matmul_precision reaches the kernel through Qwen3NextGatedDeltaNet / model_runner."""
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from maxtext.models import qwen3  # pylint: disable=import-outside-toplevel

  spec = importlib.util.spec_from_file_location(
      "gdn_cp_head_sharded_test", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_cp_head_sharded_test.py")
  )
  helper = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(helper)
  mesh = Mesh(np.array(devices[:4]), axis_names=("context",))
  counts = {}
  for value in ("highest", "high"):
    cfg = helper.create_gdn_config(
        hidden_size=128,
        num_key_heads=1,
        num_value_heads=2,
        head_dim=128,
        chunk_size=64,
        cp_size=4,
        dtype=jnp.bfloat16,
        gdn_cp_mode="seq",
    )
    cfg.gdn_cp_matmul_precision = value
    with mesh:
      model = qwen3.Qwen3NextGatedDeltaNet(config=cfg, mesh=mesh, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    x = jax.device_put(jnp.ones((1, 512, 128), jnp.bfloat16), NamedSharding(mesh, P(None, "context", None)))

    def loss(state, x, graphdef=graphdef):
      return jnp.sum(nnx.merge(graphdef, state)(x)[0].astype(jnp.float32))

    with mesh:
      text = jax.jit(jax.grad(loss)).lower(state, x).as_text()
    counts[value] = len(re.findall(r"precision = \[HIGH, HIGH\]", text))
  assert counts["high"] > counts["highest"], counts
  print(f"config path: HIGH dots {counts}")


def _main():
  devices = jax.devices()
  assert len(devices) >= 4, devices
  _check_kernel_path(Mesh(np.array(devices[:4]), ("context",)))
  _check_config_path(devices)
  print("GDN_CP_PRECISION_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
  _main()
