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
"""The `gdn_cp_state` and `moe_route` remat keys on a CPU mesh.

Each key saves a few small values where they are produced, so the backward replay of a rematerialized layer can skip
the work that exists only to produce them:
  - `gdn_cp_state`: the sequence-sharded CP GatedDeltaNet carries (conv halo, local transition, incoming state). The
    replay then runs pass 2 of the CP forward alone, without the halo exchange, pass 1 and the cross-rank composition.
  - `moe_route`: the ring-of-experts ragged sort's integer results (sorted and revert indices, group sizes). The replay
    then gathers the tokens again without the two argsorts.

The model is a tiny hybrid Qwen3.5 (GatedDeltaNet kernel path, gated full attention, routed and shared-expert MoE) on
8 CPU devices, fsdp1 x context4 x expert2 under cp-as-ep, as in the Qwen3.5-397B configuration: GatedDeltaNet runs
sequence-sharded context parallelism over 4 ranks, and ring of experts routes over an expert-parallel group of 8 with
the ragged sort, local top-k routing and two token chunks. remat_policy=custom with scanned layers and
decoder_layer_input/context on device, also as in that configuration. On CPU the GatedDeltaNet forward kernel runs as its
pure-JAX reference and the GMM as megablox in interpret mode. The checks run in a subprocess so the forced device count
takes effect before JAX initializes, with JAX's persistent compilation cache off: pyconfig points it at `jax_cache_dir`,
which every process on the machine shares, and a hit returns an executable another process compiled.

  - Inert by default: with both keys at 'remat', the compiled program is identical to the one compiled with the new
    checkpoint names stripped out. Programs are compared without op metadata and the stack-frame tables, which record
    where each op was traced, down to the calling line.
  - Numerics: with each key on 'device', and with both, the loss is bit-identical to the default and every parameter
    gradient agrees to reduction order, max|new - old| <= 1e-5 * max|old|.
  - The recompute is gone, counted over the compiled HLO instructions under `rematted_computation`:
      gdn_cp_state: the GatedDeltaNet kernel replays no collective-permute (halo exchange, state composition), and its
        forward loops halve (pass 1's go, pass 2's stay);
      moe_route: the ring ragged sort replays no sort.
    The default must replay all of these, so the counts cannot pass vacuously.
  - Positive controls, each of which must fail the gate it targets:
      inert by default: the program with both keys on must compare unequal to the default;
      numerics: the incoming state tagged `gdn_cp_s_in` replaced by pass 1's zero initial state; the values tagged
        `moe_route` shifted by one; and a wrong carry only the gradients can see, the backward handed the zero conv
        halo only rank 0 should see while pass 2 consumed the exchanged one, so the loss stays bit-identical;
      recompute: `gdn_cp_state` expanded to the existing residual names (`gdn_conv_state`, `gdn_recurrent_state`,
        `gdn_m_local`), which wrap the carries only after pass 2 consumed them; and the `moe_route` names attached to
        copies nothing downstream reads. The replayed work must stay.
"""

# pylint: disable=protected-access

import collections
import contextlib
import dataclasses
import os
import re
import subprocess
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.kernels.gdn.gdn_bwd import api as gdn_api
from maxtext.kernels.ragged import ragged_sort
from maxtext.utils import maxtext_utils
from maxtext.utils import globals as maxtext_globals

_BASE_CONFIG_PATH = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
_NUM_DEVICES = 8
# Four context ranks of two 64-token GatedDeltaNet kernel chunks each.
_SEQ = 512
# Reduction-order gate on every gradient leaf: max|new - old| <= _REL_TOL * max|old|.
_REL_TOL = 1e-5
_PASS_TOKEN = "REMAT_CP_STATE_AND_ROUTE_CHECKS_PASSED"
_GDN_CP_NAMES = ("gdn_cp_conv_halo", "gdn_cp_m_local", "gdn_cp_s_in")
_GDN_RESIDUAL_NAMES = ("gdn_conv_state", "gdn_recurrent_state", "gdn_m_local")


@pytest.mark.cpu_only
def test_remat_cp_state_and_route_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _config(**remat_keys):
  """The tiny hybrid Qwen3.5 on the cp-as-ep mesh, with `remat_keys` over the 397B model's custom remat policy."""
  argv = [
      None,
      _BASE_CONFIG_PATH,
      "run_name=remat_cp_state_and_route_test",
      "model_name=qwen3.5-397b-a17b",
      "override_model_config=true",
      "hardware=cpu",
      "custom_mesh_and_rule=cp-as-ep",
      "ici_fsdp_parallelism=1",
      "ici_context_parallelism=4",
      "ici_expert_parallelism=2",
      "ici_tensor_parallelism=1",
      "context_parallel_load_balance=False",
      "attention=dot_product",
      "use_gdn_kernel=true",
      "gdn_cp_mode=seq",
      "sparse_matmul=true",
      "megablox=true",
      # Tokamax gmm v2 is TPU-only; every arm runs the same GMM, so CPU uses megablox.
      "use_tokamax_gmm=false",
      "use_gmm_v2=false",
      "use_ring_of_experts=true",
      "use_ragged_sort=true",
      "use_custom_sort_vjp=false",
      "ragged_buffer_factor=2.0",
      "num_moe_token_chunks=2",
      "ring_of_experts_local_routing=true",
      # The ragged sort only supports bfloat16 activations.
      "dtype=bfloat16",
      "weight_dtype=float32",
      "scan_layers=True",
      "remat_policy=custom",
      "decoder_layer_input=device",
      "context=device",
      "packing=false",
      "enable_checkpointing=false",
      "enable_dropout=false",
      "dataset_type=synthetic",
      "base_num_decoder_layers=8",  # two blocks of [3 GatedDeltaNet + 1 full attention]
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=64",
      "mrope_section=[3,3,2]",  # rotary_dim / 2 = 64 * 0.25 / 2
      "base_mlp_dim=32",
      "base_moe_mlp_dim=32",
      "num_experts=8",
      "num_experts_per_tok=2",
      "shared_experts=1",
      # The GatedDeltaNet kernel path needs head dims that are multiples of 128 and gdn_chunk_size=64.
      "gdn_key_head_dim=128",
      "gdn_value_head_dim=128",
      "gdn_num_key_heads=2",
      "gdn_num_value_heads=4",
      "gdn_chunk_size=64",
      "vocab_size=384",
      f"max_target_length={_SEQ}",
      "per_device_batch_size=0.5",
      "use_multimodal=false",
      "skip_jax_distributed_system=True",
      "enable_tensorboard=False",
      *(f"{key}={value}" for key, value in remat_keys.items()),
  ]
  return pyconfig.initialize(argv)


def _inputs(cfg):
  """Random tokens in one segment per row, and a fixed random cotangent on the logits.

  The cotangent makes the backward per-token sensitive: at random init the cross-entropy is nearly flat.
  """
  batch = cfg.micro_batch_size_to_train_on
  tokens = jax.random.randint(jax.random.PRNGKey(7), (batch, _SEQ + 1), 0, cfg.vocab_size)
  ones = jnp.ones((batch, _SEQ), jnp.int32)
  data = {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "inputs_position": jnp.broadcast_to(jnp.arange(_SEQ, dtype=jnp.int32), (batch, _SEQ)),
      "inputs_segmentation": ones,
      "targets_segmentation": ones,
  }
  cotangent = jax.random.normal(jax.random.PRNGKey(3), (batch, _SEQ, cfg.vocab_size), jnp.float32)
  return data, cotangent


@dataclasses.dataclass
class _Result:
  loss: np.ndarray
  grads: dict[str, np.ndarray]
  hlo: str
  program: str  # `hlo` without op metadata and the stack-frame tables, for program identity


def _run(cfg, data, cotangent, control=None) -> _Result:
  """Loss <logits, cotangent>, its parameter gradients, and the compiled HLO; `control` wraps trace and compile."""
  # pylint: disable=import-outside-toplevel
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.utils import maxtext_utils_nnx, model_creation_utils

  mesh = maxtext_utils.get_mesh_from_config(cfg)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=jax.random.PRNGKey(0))
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  def loss_fn(params):
    merged = nnx.merge(graphdef, params, rest, copy=True)
    logits = merged(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        enable_dropout=False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    return jnp.sum(logits.astype(jnp.float32) * cotangent)

  with control or contextlib.nullcontext(), nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    compiled = jax.jit(jax.value_and_grad(loss_fn)).lower(params).compile()
    loss, grads = compiled(params)
  leaves = {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}
  hlo = compiled.as_text()
  return _Result(np.asarray(loss), leaves, hlo, _program(hlo))


# ---------------------------------------------------------------------------
# Gates.
# ---------------------------------------------------------------------------


def _numerics(label, result, reference) -> bool:
  """Bit-identical loss and every gradient leaf within reduction order of the reference; prints the report."""
  assert result.grads.keys() == reference.grads.keys()
  loss_equal = bool(np.array_equal(result.loss, reference.loss))
  worst, worst_name, not_bit_equal, grads_ok = 0.0, "", 0, True
  for name, ref in reference.grads.items():
    new, ref = result.grads[name].astype(np.float64), ref.astype(np.float64)
    diff, scale = float(np.max(np.abs(new - ref))), float(np.max(np.abs(ref)))
    # Every comparison is False on NaN, so a NaN fails.
    grads_ok &= bool(diff <= _REL_TOL * scale) and bool(np.isfinite(new).all())
    rel = diff / scale if scale > 0 else diff
    if rel > worst or not worst_name:
      worst, worst_name = rel, name
    not_bit_equal += int(not np.array_equal(new, ref))
  passed = loss_equal and grads_ok
  print(
      f"{label}: numerics {'PASS' if passed else 'FAIL'}: loss bit-identical={loss_equal}"
      f" (d={float(result.loss) - float(reference.loss):.6e}); gradients within {_REL_TOL:g} x max|ref|={grads_ok},"
      f" worst {worst:.3e} at {worst_name}, {not_bit_equal} of {len(reference.grads)} leaves not bit-identical",
      flush=True,
  )
  return passed


_INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%[\w.\-]+\s*=\s*(.*)$")
_OPCODE = re.compile(r"(?:^|[\s)}])([a-z][\w\-]*)\(")


_STACK_FRAME_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
_METADATA = re.compile(r', metadata=\{(?:[^{}"]|"(?:[^"\\]|\\.)*")*\}')


def _program(hlo_text: str) -> str:
  """The compiled HLO without op metadata and the stack-frame tables.

  Both record where JAX traced each op, down to the line that called `_run`, so the same program traced from two call
  sites prints different text.
  """
  lines, in_table = [], False
  for line in hlo_text.splitlines():
    text = line.strip()
    if text in _STACK_FRAME_TABLES:
      in_table = True
      continue
    if in_table and (not text or text[:1].isdigit()):
      continue
    in_table = False
    lines.append(_METADATA.sub("", line))
  return "\n".join(lines)


def _innermost_files(hlo_text: str) -> dict[int, str]:
  """stack_frame_id -> basename of the frame's source file, from the module's stack-frame tables."""
  files, locations, frames, section = {}, {}, {}, None
  for line in hlo_text.splitlines():
    text = line.strip()
    if text in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
      section = text
      continue
    if section is None or not text[:1].isdigit():
      section = None
      continue
    index, rest = text.split(" ", 1)
    if section == "FileNames":
      files[int(index)] = os.path.basename(rest.strip('"'))
    elif section == "FileLocations":
      locations[int(index)] = int(re.search(r"file_name_id=(\d+)", rest).group(1))
    elif section == "StackFrames":
      frames[int(index)] = int(re.search(r"file_location_id=(\d+)", rest).group(1))
  return {frame: files.get(locations.get(location), "?") for frame, location in frames.items()}


def _replayed(hlo_text: str) -> collections.Counter:
  """Counts compiled instructions under `rematted_computation` by (opcode, source file of their stack frame)."""
  files = _innermost_files(hlo_text)
  counts = collections.Counter()
  for line in hlo_text.splitlines():
    match = _INSTRUCTION.match(line)
    op_name = re.search(r'op_name="([^"]*)"', line)
    if not match or not op_name or "rematted_computation" not in op_name.group(1):
      continue
    opcode = _OPCODE.search(match.group(1))
    frame = re.search(r"stack_frame_id=(\d+)", line)
    if opcode:
      counts[(opcode.group(1), files.get(int(frame.group(1)), "?") if frame else "?")] += 1
  return counts


@dataclasses.dataclass(frozen=True)
class _Replay:
  """The replayed work each key targets.

  Everything traced inside a custom_vjp rule carries the stack frame of the custom_vjp's call site, so the
  GatedDeltaNet kernel's instructions resolve to its call in model_runner.py and the ring sort's to ragged_sort.py.
  """

  cp_permutes: int  # the GatedDeltaNet kernel's collective-permutes: conv halo exchange and cross-rank state composition
  gdn_loops: int  # the GatedDeltaNet kernel's while loops: its chunked forward (pure-JAX on CPU), once per CP pass
  sorts: int  # the ring ragged sort's argsorts

  @classmethod
  def of(cls, hlo_text: str) -> "_Replay":
    counts = _replayed(hlo_text)
    return cls(
        cp_permutes=counts[("collective-permute", "model_runner.py")],
        gdn_loops=counts[("while", "model_runner.py")],
        sorts=counts[("sort", "ragged_sort.py")],
    )


def _cp_state_replay_gone(replay: _Replay, default: _Replay) -> bool:
  """No halo exchange or state composition is replayed, and pass 1's loops are gone while pass 2's stay."""
  return replay.cp_permutes == 0 and 0 < replay.gdn_loops and 2 * replay.gdn_loops == default.gdn_loops


def _route_replay_gone(replay: _Replay) -> bool:
  return replay.sorts == 0


# ---------------------------------------------------------------------------
# Controls: the program before this change, and deliberately wrong checkpoints.
# ---------------------------------------------------------------------------


def _patched(target, attribute, make):
  """mock.patch of `target.attribute` with `make(real)`."""
  return mock.patch.object(target, attribute, make(getattr(target, attribute)))


@contextlib.contextmanager
def _strip_new_names():
  """The program before this change: the new checkpoint names are absent."""
  with (
      _patched(gdn_api, "checkpoint_name", lambda real: lambda x, name: x if name in _GDN_CP_NAMES else real(x, name)),
      _patched(ragged_sort, "checkpoint_name", lambda real: lambda x, name: x if name == "moe_route" else real(x, name)),
  ):
    yield


def _tag_zero_incoming_state():
  """The value tagged `gdn_cp_s_in`, which pass 2 consumes, is pass 1's zero initial state."""
  return _patched(
      gdn_api,
      "checkpoint_name",
      lambda real: lambda x, name: real(jnp.zeros_like(x) if name == "gdn_cp_s_in" else x, name),
  )


def _tag_shifted_route():
  """Every value tagged `moe_route` is shifted by one position."""
  return _patched(
      ragged_sort,
      "checkpoint_name",
      lambda real: lambda x, name: real(jnp.roll(x, 1) if name == "moe_route" else x, name),
  )


def _backward_gets_zero_conv_halo():
  """Pass 2 consumes the exchanged conv halo, but the backward's residual is the zero halo only rank 0 should see.

  The CP backward reads the halo (to recompute the conv) and m_local; it does not read the incoming state when the
  chunk states are saved, so a wrong incoming state there would be a dead mutation rather than a control.
  """

  def make(real):
    def wrong(*args, **kwargs):
      primal, t_inv, chunk_states, conv_halo, s_in_r, m_local = real(*args, **kwargs)
      return primal, t_inv, chunk_states, jnp.zeros_like(conv_halo), s_in_r, m_local

    return wrong

  return _patched(gdn_api, "_run_cp_gdn_decoupled_fwd_impl", make)


def _residual_names_for_cp_state():
  """`gdn_cp_state` expanded to the residual names, which wrap the carries after pass 2 consumed them."""

  def make(real):
    def late(names):
      expanded = real([name for name in names if name != "gdn_cp_state"])
      return expanded + list(_GDN_RESIDUAL_NAMES) if "gdn_cp_state" in names else expanded

    return late

  return _patched(maxtext_utils, "_expand_gdn_remat_names", make)


def _detached_route_names():
  """The `moe_route` names are attached to copies that nothing downstream reads."""

  def make(real):
    def detached(x, name):
      named = real(x, name)
      return x if name == "moe_route" else named

    return detached

  return _patched(ragged_sort, "checkpoint_name", make)


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  jax.config.update("jax_enable_compilation_cache", False)
  default_cfg = _config()
  data, cotangent = _inputs(default_cfg)
  default = _run(default_cfg, data, cotangent)
  default_replay = _Replay.of(default.hlo)
  print(f"default: replayed {default_replay}", flush=True)
  checks = {
      "the default replays all of the targeted work": min(dataclasses.astuple(default_replay)) > 0,
      "the default program is identical without the new names": default.program
      == _run(default_cfg, data, cotangent, control=_strip_new_names()).program,
  }

  arms = {
      "gdn_cp_state": {"gdn_cp_state": "device"},
      "moe_route": {"moe_route": "device"},
      "both": {"gdn_cp_state": "device", "moe_route": "device"},
  }
  results = {}
  for label, keys in arms.items():
    cfg = _config(**keys)
    assert set(keys) <= set(cfg.tensors_on_device), cfg.tensors_on_device
    result = results[label] = _run(cfg, data, cotangent)
    replay = _Replay.of(result.hlo)
    print(f"{label}: replayed {replay}", flush=True)
    checks[f"{label}: numerics"] = _numerics(label, result, default)
    if "gdn_cp_state" in keys:
      checks[f"{label}: no CP composition in the replay, pass 1 gone"] = _cp_state_replay_gone(replay, default_replay)
    else:
      checks[f"{label}: replayed GatedDeltaNet work unchanged"] = (replay.cp_permutes, replay.gdn_loops) == (
          default_replay.cp_permutes,
          default_replay.gdn_loops,
      )
    if "moe_route" in keys:
      checks[f"{label}: no argsort in the replay"] = _route_replay_gone(replay)
    else:
      checks[f"{label}: replayed sorts unchanged"] = replay.sorts == default_replay.sorts

  # Positive controls. Each must fail its gate; `controls` records whether it was caught.
  cp_state_cfg, route_cfg = _config(**arms["gdn_cp_state"]), _config(**arms["moe_route"])
  controls = {
      "inert by default: a program that saves the carries compares unequal": results["both"].program != default.program
  }
  wrong = _run(cp_state_cfg, data, cotangent, control=_tag_zero_incoming_state())
  controls["numerics: zero incoming state tagged gdn_cp_s_in"] = not _numerics("control zero s_in", wrong, default)
  wrong = _run(route_cfg, data, cotangent, control=_tag_shifted_route())
  controls["numerics: moe_route values shifted by one"] = not _numerics("control shifted route", wrong, default)
  wrong = _run(cp_state_cfg, data, cotangent, control=_backward_gets_zero_conv_halo())
  controls["numerics: backward-only zero conv halo (loss unchanged)"] = bool(
      np.array_equal(wrong.loss, default.loss)
  ) and not _numerics("control backward-only zero halo", wrong, default)
  replay = _Replay.of(_run(cp_state_cfg, data, cotangent, control=_residual_names_for_cp_state()).hlo)
  print(f"control residual names: replayed {replay}", flush=True)
  controls["recompute: gdn_cp_state as the residual names"] = not _cp_state_replay_gone(replay, default_replay)
  replay = _Replay.of(_run(route_cfg, data, cotangent, control=_detached_route_names()).hlo)
  print(f"control detached route names: replayed {replay}", flush=True)
  controls["recompute: detached moe_route names"] = not _route_replay_gone(replay)

  for name, ok in {**checks, **{f"positive control caught: {k}": v for k, v in controls.items()}}.items():
    print(f"  [{'ok' if ok else 'FAIL'}] {name}", flush=True)
  failed = [name for name, ok in checks.items() if not ok]
  undetected = [name for name, caught in controls.items() if not caught]
  if failed or undetected:
    print(f"checks that failed: {failed}; positive controls that were NOT detected: {undetected}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
