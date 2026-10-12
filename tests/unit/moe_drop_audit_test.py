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
"""The MoE drop audit (log_moe_drops) and the final micro-batch's dropless program, on a CPU mesh.

The layer is `moe_chunk_pipeline_test.py`'s: a shrunk Qwen3.5 routed MoE on 8 CPU devices under cp-as-ep (an
expert-parallel group of 8), local routing, ragged sort and two pipelined token chunks, here at
ragged_buffer_factor=2.5. Every token is routed to experts 0..top_k-1, which live on two of the eight shards: each
receives 4x the balanced load, above the 2.5x buffer.

Checks:
- the capped layer overflows and its audit counts exactly the assignments past the buffer
  (sum over shards and chunks of max(0, shard load - int(balanced * 2.5))), with max load ratio 4.0;
- `apply_final_dropless_overrides` (force_dropless only: same chunks and pipeline) drops nothing, counts 0 and
  matches the ragged_buffer_factor=-1 reference;
- `apply_dropless_overrides` (the #5570 replay program) counts 0 and matches the reference too;
- with log_moe_drops off nothing is sown, so the production program is unchanged.
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

_PASS_TOKEN = "MOE_DROP_AUDIT_CHECKS_PASSED"
_REL_TOL = 1e-5
_RBF = 2.5


@pytest.mark.cpu_only
def test_drop_audit_and_final_dropless_program_on_cpu_mesh():
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


def _run(cfg, mesh, params, x, forced, override=None):
  """The layer's output, its overflow flag, and its audit `(dropped, max_ratio)` (None if nothing was sown)."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = layer_setup._build(cfg, mesh)  # pylint: disable=protected-access
    if override == "final":
      maxtext_engine.apply_final_dropless_overrides(model)
    elif override == "replay":
      maxtext_engine.apply_dropless_overrides(cfg, model)
    graphdef, _, rest = nnx.split(model, nnx.Param, ...)

    @jax.jit
    def forward(params, x):
      layer = nnx.merge(graphdef, params, rest)
      out, _, _ = layer(x, forced_routed_experts=forced)
      sown = nnx.state(layer, nnx.Intermediate).to_pure_dict()
      flags = maxtext_utils.collect_intermediates_by_suffix(sown, "moe_has_overflow")
      audit = maxtext_engine.collect_moe_audit(layer)
      return out, jnp.any(jnp.concatenate(flags)) if flags else jnp.bool_(False), audit

    out, overflow, audit = forward(params, x)
  if audit is not None:
    audit = (int(np.sum(np.asarray(audit["dropped"]))), float(np.max(np.asarray(audit["max_ratio"]))))
  return np.asarray(out, dtype=np.float32), bool(overflow), audit


def main():
  cfg_ref = layer_setup._cfg(True, ragged_buffer_factor=-1.0)  # pylint: disable=protected-access
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
  x, _ = layer_setup._inputs(cfg_ref, forced_mode=False)  # pylint: disable=protected-access
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
    _, params, _ = nnx.split(layer_setup._build(cfg_ref, mesh), nnx.Param, ...)  # pylint: disable=protected-access
  top_k = cfg_ref.num_experts_per_tok
  forced = jnp.broadcast_to(jnp.arange(top_k, dtype=jnp.int32), (x.shape[0], x.shape[1], top_k))

  ref, _, _ = _run(cfg_ref, mesh, params, x, forced)
  cfg_off = layer_setup._cfg(True, ragged_buffer_factor=_RBF)  # pylint: disable=protected-access
  _, off_overflow, off_audit = _run(cfg_off, mesh, params, x, forced)
  cfg = layer_setup._cfg(True, ragged_buffer_factor=_RBF, log_moe_drops=True)  # pylint: disable=protected-access
  capped, capped_overflow, capped_audit = _run(cfg, mesh, params, x, forced)
  final, final_overflow, final_audit = _run(cfg, mesh, params, x, forced, override="final")
  cfg_replay = layer_setup._cfg(  # pylint: disable=protected-access
      True, ragged_buffer_factor=_RBF, log_moe_drops=True, retry_num_moe_token_chunks=8
  )
  replay, replay_overflow, replay_audit = _run(cfg_replay, mesh, params, x, forced, override="replay")

  # The expected count, from the layer's own sizing: per token chunk the EP group (the whole mesh here, fsdp=1)
  # all-gathers `tokens` tokens; balanced = tokens // ep * top_k; buffer = int(balanced * rbf); the forced experts
  # 0..top_k-1 sit on the first top_k // experts_per_shard shards, each receiving tokens * experts_per_shard.
  ep = 8
  n_chunks = cfg.num_moe_token_chunks
  tokens = x.shape[0] * x.shape[1] // n_chunks
  balanced = tokens // ep * top_k
  buffer = int(balanced * _RBF)
  experts_per_shard = cfg.num_experts // ep
  hot_shards = top_k // experts_per_shard
  hot_load = tokens * experts_per_shard
  expected = n_chunks * hot_shards * max(0, hot_load - buffer)
  expected_ratio = hot_load / balanced

  def rel(a):
    return float(np.max(np.abs(a - ref))) / max(float(np.max(np.abs(ref))), 1e-30)

  print(f"expected: dropped={expected} ratio={expected_ratio} (tokens/chunk {tokens}, balanced {balanced}, "
        f"buffer {buffer}, hot shard load {hot_load} x {hot_shards} shards x {n_chunks} chunks)", flush=True)
  print(f"audit off: overflow={off_overflow} audit={off_audit}", flush=True)
  print(f"capped rbf {_RBF}: overflow={capped_overflow} audit={capped_audit} rel diff {rel(capped):.1e}", flush=True)
  print(f"final dropless: overflow={final_overflow} audit={final_audit} rel diff {rel(final):.1e}", flush=True)
  print(f"replay dropless: overflow={replay_overflow} audit={replay_audit} rel diff {rel(replay):.1e}", flush=True)
  assert off_overflow and off_audit is None, "with log_moe_drops off nothing is sown"
  assert expected > 0, "the setup must overflow"
  assert capped_overflow and rel(capped) > _REL_TOL, "positive control: the capped layer must drop tokens"
  assert capped_audit is not None and capped_audit[0] == expected, (capped_audit, expected)
  assert abs(capped_audit[1] - expected_ratio) < 1e-6, (capped_audit, expected_ratio)
  assert not final_overflow and rel(final) <= _REL_TOL, "the final micro-batch's program must drop no token"
  assert final_audit is not None and final_audit[0] == 0, final_audit
  assert abs(final_audit[1] - expected_ratio) < 1e-6, final_audit
  assert not replay_overflow and rel(replay) <= _REL_TOL, "the replay program must drop no token"
  assert replay_audit is not None and replay_audit[0] == 0, replay_audit
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
