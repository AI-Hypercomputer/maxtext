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
"""The training engine's dropless replay program drops no token, on a CPU mesh.

The layer is `moe_chunk_pipeline_test.py`'s: a shrunk Qwen3.5 routed MoE on 8 CPU devices under cp-as-ep
(an expert-parallel group of 8), local routing, ragged sort, two pipelined token chunks and
ragged_buffer_factor=2.0. Every token is routed to experts 0..top_k-1, which live on two of the eight
shards: each receives 4x the balanced load, twice what the buffer holds. The capped layer must report an
overflow and differ from the ragged_buffer_factor=-1 reference (the positive control); the same layer after
`apply_dropless_overrides` (8 scanned token chunks) must report none and match it.
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

from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.unit import moe_chunk_pipeline_test as layer_setup

_PASS_TOKEN = "MOE_DROPLESS_REPLAY_CHECKS_PASSED"
_REL_TOL = 1e-5


@pytest.mark.cpu_only
def test_dropless_replay_program_drops_no_token_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {layer_setup._XLA_FLAGS}"  # pylint: disable=protected-access
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _run(cfg, mesh, params, x, forced, dropless_replay=False):
  """The layer's output and whether it reported an overflow."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = layer_setup._build(cfg, mesh)  # pylint: disable=protected-access
    if dropless_replay:
      maxtext_engine.apply_dropless_overrides(cfg, model)
    graphdef, _, rest = nnx.split(model, nnx.Param, ...)

    @jax.jit
    def forward(params, x):
      layer = nnx.merge(graphdef, params, rest)
      out, _, _ = layer(x, forced_routed_experts=forced)
      sown = nnx.state(layer, nnx.Intermediate).to_pure_dict()
      flags = maxtext_utils.collect_intermediates_by_suffix(sown, "moe_has_overflow")
      return out, jnp.any(jnp.concatenate(flags)) if flags else jnp.bool_(False)

    out, overflow = forward(params, x)
  return np.asarray(out, dtype=np.float32), bool(overflow)


def main():
  cfg_ref = layer_setup._cfg(True, ragged_buffer_factor=-1.0)  # pylint: disable=protected-access
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
  x, _ = layer_setup._inputs(cfg_ref, forced_mode=False)  # pylint: disable=protected-access
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
    _, params, _ = nnx.split(layer_setup._build(cfg_ref, mesh), nnx.Param, ...)  # pylint: disable=protected-access
  top_k = cfg_ref.num_experts_per_tok
  forced = jnp.broadcast_to(jnp.arange(top_k, dtype=jnp.int32), (x.shape[0], x.shape[1], top_k))

  ref, _ = _run(cfg_ref, mesh, params, x, forced)
  cfg = layer_setup._cfg(  # pylint: disable=protected-access
      True, retry_num_moe_token_chunks=8
  )
  capped, capped_overflow = _run(cfg, mesh, params, x, forced)
  replay, replay_overflow = _run(cfg, mesh, params, x, forced, dropless_replay=True)

  def rel(a):
    return float(np.max(np.abs(a - ref))) / max(float(np.max(np.abs(ref))), 1e-30)

  print(f"capped: overflow={capped_overflow}, rel diff {rel(capped):.1e}", flush=True)
  print(f"replay: overflow={replay_overflow}, rel diff {rel(replay):.1e}", flush=True)
  assert capped_overflow and rel(capped) > _REL_TOL, "positive control: the capped layer must drop tokens"
  assert not replay_overflow and rel(replay) <= _REL_TOL, "the dropless replay program must drop no token"
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
