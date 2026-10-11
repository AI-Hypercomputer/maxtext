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

"""The log_required_ragged_buffer_factor probe through the real ring-of-experts + ragged-sort path, on a CPU mesh.

`RaggedBufferProbeTest` (moe_test.py) calls `RoutedMoE.ragged_buffer_probe` with `mesh=None`, so the mesh
collectives (a pmax and a psum over every mesh axis, the psum divided by the token replication) and the plumbing
out of the shard_map are not covered there. Here a `RoutedMoE` runs on 8 CPU devices with forced routing, and the
probe vector it sows ([required factor, max rows dropped, total rows dropped]) is compared with a numpy reference
computed from the same global routing:

  per EP group (the devices that share one EP-gathered batch), the global per-expert row counts; per expert shard,
  local = rows routed to its experts, required = local / balanced_size, and, with a buffer of
  int(balanced_size * factor) rows, dropped = max(0, local - buffer). Expected: [max required, max dropped,
  sum of dropped over all groups and shards], each real token counted once however many devices replicate it.

Layouts: FSDP x EP, FSDP x TP x EP (tokens replicated over tensor) and cp-as-ep (EP group = context x expert).
Also checked: the dropless buffer and the `force_dropless` replay drop nothing but still report the required
factor, `moe_has_overflow` agrees with the dropped total, and the flag changes no output. A positive control (a
wrong token replication) must fail the same comparison.

Runs as a subprocess so the forced device count takes effect before JAX initializes (as moe_roe_local_routing_test.py).
"""

import contextlib
import os
import subprocess
import sys
from unittest import mock

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 8
_TOP_K = 2
_EMB = 64
_SEQ = 32
_PASS_TOKEN = "MOE_BUFFER_PROBE_MESH_CHECKS_PASSED"

# name -> (mesh overrides, per_device_batch_size). The batch is sharded over data x fsdp x expert; a mesh axis outside
# those (tensor) replicates each token, which the probe's total must not count. Under cp-as-ep the EP group is
# (context, expert) and the sequence is sharded over context.
_LAYOUTS = {
    "fsdp4_ep2": ({"ici_expert_parallelism": 2}, 1),
    "fsdp2_tp2_ep2": ({"ici_expert_parallelism": 2, "ici_tensor_parallelism": 2}, 2),
    "fsdp2_cp2_ep2_cp_as_ep": (
        {
            "custom_mesh_and_rule": "cp-as-ep",
            "ici_fsdp_parallelism": 2,
            "ici_context_parallelism": 2,
            "ici_expert_parallelism": 2,
            "context_parallel_load_balance": False,
        },
        1,
    ),
}


@pytest.mark.cpu_only
def test_probe_matches_reference_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  # The child imports `tests.utils`, so it needs the repository root on its path, not only `src`.
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _cfg(layout, factor, probe=True):
  overrides, per_device_batch_size = _LAYOUTS[layout]
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name=f"moe_buffer_probe_{layout}",
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      model_name="mixtral-8x7b",
      override_model_config=True,
      base_emb_dim=_EMB,
      base_mlp_dim=64,
      base_moe_mlp_dim=64,
      dtype="float32",
      weight_dtype="float32",
      megablox=True,  # In interpret mode on CPU.
      sparse_matmul=True,
      per_device_batch_size=per_device_batch_size,
      use_ring_of_experts=True,
      use_ragged_sort=True,
      ragged_gather_fallback=True,
      ragged_gather_reduce_fallback=True,
      max_target_length=_SEQ,
      float32_gate_logits=True,
      ragged_buffer_factor=factor,
      log_required_ragged_buffer_factor=probe,
      **overrides,
  )


def _build(cfg, mesh, force_dropless=False):
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
      force_dropless=force_dropless,
      rngs=nnx.Rngs(params=0),
  )


def _routing(pattern, batch):
  """Forced [batch, seq, top_k] expert ids: distinct per token, uniform, skewed to the low experts, or all on 0 and 1."""
  if pattern == "hot":
    return np.broadcast_to(np.arange(_TOP_K, dtype=np.int32), (batch, _SEQ, _TOP_K)).copy()
  rng = np.random.default_rng(0)
  p = np.ones(_NUM_EXPERTS) if pattern == "uniform" else np.array([8.0, 4, 2, 1, 1, 1, 1, 1])
  scores = np.log(p / p.sum())[None, None, :] + rng.gumbel(size=(batch, _SEQ, _NUM_EXPERTS))  # Gumbel top-k.
  return np.argsort(-scores, axis=-1)[..., :_TOP_K].astype(np.int32)


def _ep_size(cfg, mesh_shape):
  """Devices in one EP group: the expert axis, times the context axis under cp-as-ep (it routes experts there too)."""
  return mesh_shape["expert"] * (mesh_shape["context"] if cfg.custom_mesh_and_rule == ctypes.CustomRule.CP_AS_EP else 1)


def _reference(forced, mesh_shape, ep_size, factor, force_dropless):
  """[max required factor, max rows dropped, total rows dropped] from the global routing, in numpy."""
  groups = mesh_shape["data"] * mesh_shape["fsdp"]  # An EP group gathers the batch rows of one (data, fsdp) slice.
  batch, seq, top_k = forced.shape
  per_group = batch // groups
  local_experts = _NUM_EXPERTS // ep_size
  required, dropped = [], []
  for group in range(groups):
    counts = np.bincount(forced[group * per_group : (group + 1) * per_group].reshape(-1), minlength=_NUM_EXPERTS)
    balanced = (per_group * seq // ep_size) * top_k
    for shard in range(ep_size):
      local = int(counts[shard * local_experts : (shard + 1) * local_experts].sum())
      required.append(local / balanced)
      truncated = factor > 0 and not force_dropless
      dropped.append(max(0, local - int(balanced * factor)) if truncated else 0)
  return np.array([max(required), max(dropped), sum(dropped)])


def _run(cfg, mesh, x, forced, force_dropless=False):
  """Output, the probe reduced over layers (None when not sown) and the overflow flag of one forward pass."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = _build(cfg, mesh, force_dropless)
    out, _, _ = model(x, forced_routed_experts=forced)
    intermediates = nnx.pop(model, nnx.Intermediate).to_pure_dict()
  probes = maxtext_utils.collect_intermediates_by_suffix(intermediates, "moe_buffer_probe")
  overflow = maxtext_utils.collect_intermediates_by_suffix(intermediates, "moe_has_overflow")
  probe = np.asarray(moe.reduce_moe_buffer_probe(probes)) if probes else None
  return np.asarray(out), probe, bool(np.any([np.any(np.asarray(v)) for v in overflow]))


def _case(label, layout, factor, pattern, *, force_dropless=False, control=None):
  """Compares the probe with the reference; `control` is a context manager applied to the probe run only."""
  cfg = _cfg(layout, factor)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  batch = int(_LAYOUTS[layout][1]) * _NUM_DEVICES
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  forced_np = _routing(pattern, batch)
  with control if control is not None else contextlib.nullcontext():
    _, probe, overflow = _run(cfg, mesh, x, jnp.asarray(forced_np), force_dropless)
  expected = _reference(forced_np, dict(mesh.shape), _ep_size(cfg, dict(mesh.shape)), factor, force_dropless)
  ok = probe is not None and np.allclose(probe, expected) and overflow == (expected[2] > 0)
  got = None if probe is None else probe.tolist()
  print(f"{label}: [{'ok' if ok else 'FAIL'}] probe={got} expected={expected.tolist()} overflow={overflow}", flush=True)
  return bool(ok)


def _flag_changes_nothing(layout, factor, pattern):
  """The probe is read-only: outputs are bit-identical with the flag off, and nothing is sown."""
  cfg_on, cfg_off = _cfg(layout, factor), _cfg(layout, factor, probe=False)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_on), cfg_on.mesh_axes)
  batch = int(_LAYOUTS[layout][1]) * _NUM_DEVICES
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  forced = jnp.asarray(_routing(pattern, batch))
  out_on, probe_on, _ = _run(cfg_on, mesh, x, forced)
  out_off, probe_off, _ = _run(cfg_off, mesh, x, forced)
  same_output = np.array_equal(out_on, out_off)
  ok = same_output and probe_on is not None and probe_off is None
  sown = f"{probe_on is not None}/{probe_off is not None}"
  print(f"flag_off_identical: [{'ok' if ok else 'FAIL'}] output equal={same_output} sown on/off={sown}", flush=True)
  return bool(ok)


def _wrong_token_replication():
  """Counts every replica of a token: what the probe would report if the psum were not divided."""
  return mock.patch.object(moe, "token_replication_of", lambda pspec, mesh: 1)


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  results = {
      "ep2_uniform": _case("ep2_uniform", "fsdp4_ep2", 0.5, "uniform"),
      "ep2_skew_truncated": _case("ep2_skew_truncated", "fsdp4_ep2", 1.25, "skew"),
      "ep2_hot_truncated": _case("ep2_hot_truncated", "fsdp4_ep2", 0.5, "hot"),
      "ep2_dropless": _case("ep2_dropless", "fsdp4_ep2", -1.0, "skew"),
      "ep2_force_dropless": _case("ep2_force_dropless", "fsdp4_ep2", 0.5, "skew", force_dropless=True),
      "tp2_replicated_tokens": _case("tp2_replicated_tokens", "fsdp2_tp2_ep2", 0.5, "skew"),
      "cp_as_ep": _case("cp_as_ep", "fsdp2_cp2_ep2_cp_as_ep", 1.25, "skew"),
      "flag_off": _flag_changes_nothing("fsdp4_ep2", 1.25, "skew"),
  }
  controls = {
      "control_wrong_token_replication": _case(
          "control_wrong_token_replication", "fsdp2_tp2_ep2", 0.5, "skew", control=_wrong_token_replication()
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
