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

"""ds_ep's batch-split chunks on Lineage's code: the adapter to MaxText's Lineage DeepSeek-V3 layers.

Lineage (MaxText src/maxtext/experimental/lineage, the MLPerf DeepSeek-V3 batch-split stack) runs a decoder layer
as two half-microbatches, mb0 and mb1, in eight program_order phases whose EP all-gathers / reduce-scatters run on
SparseCore beside the other half's attention or experts.  This module runs a ds_ep chunk's decoder layers on those
layer functions -- the router, dispatch and combine (dsv3_router), the gmmv2 expert kernels, Lineage's MLA and
splash attention, ops.rms_norm -- chained the way Lineage's driver chains them, but without its per-layer weight
collect / gradient reduce hooks: ds_ep gathers the weights once per step (or visit) and reduce-scatters the
accumulated gradients once per step.  ds_ep's storage, optimizer, init and checkpoints stay in MaxText's layout;
`Region` converts the gathered compute copies to Lineage's collected layout inside the program and the gradients
back (`sparse_from_maxtext` / `sparse_to_maxtext`), shard-locally.  Besides those hooks and its DeepSeek-V3 recipe's
other XLA flags (`COMPILER_OPTIONS`), two parts of MaxText's Lineage setup do not run here: its post-scheduler XLA
pass, which its use_lineage path always registers and which moves every async start as early and every done as late
as the schedule allows (`import_lineage`: XLA's own schedule stands), and the fp8 quantization that recipe sets
(lineage_quantization fp8_full; MaxText's default, none, is bf16 as here) -- the layers run unquantized, in bf16
(`layer_kwargs`).

Execution context.  Lineage's functions are global-view code over an all-Explicit mesh that open their own
shard_maps and read layouts from the arrays' sharding types: they cannot run inside ds_ep's per-program shard_map,
only in a jit at global view over the stage mesh (shard_mode 'explicit').  They run on a VIEW of ds_ep's stage mesh
m2 ('fsdp', 'expert') with a trailing size-1 axis, m3 = ('fsdp', 'expert', 'attn') over the same devices
(`a3_mesh`), under the logical-axis mapping `AXIS_MAPPING` ("A3"): every device runs all heads of its own rows --
ds_ep's own compute layout, MLA's collectives over the size-1 'attn' axis are no-ops -- experts over 'expert', FSDP
over 'fsdp'.  The activation P(('fsdp','expert'),'attn',None) on m3 is ds_ep's P(FE,None,None) on m2 device for
device, and every m2 <-> m3 hand-off is a jax.reshard that moves nothing.  The gathered weights are typed `reduced`
over the axes Lineage collects them along (dsv3_sparse_layer._w_ici_axes), so Lineage's own AD gives `unreduced`
gradients with no collective in the layer -- ds_ep's explicit accumulator types (ds_ep.chunk_gacc_specs).

The driver.  `Region` is a jax.custom_vjp: its forward chains dsv3_sparse_layer's prologue(w0) ->
scan_body(w_l, w_l+1) for l < n-1 -> epilogue(w_n-1) for ANY n >= 1 MoE layers (Lineage's own driver,
ops.w_collect_pipelined_scan_prologue_epilogue, asserts n >= 3), its backward Lineage's explicit backward functions
in reverse; its primal, what a program without AD runs (ds_ep's F), is that forward without residuals and with
Lineage's activation bank sized for no layer -- the bank only spares the backward recomputing the gate GMM, so a
forward-only program would fill it and never read it (`_chain_fwd`).  All three run under
jax.sharding.use_abstract_mesh(m3.abstract_mesh) themselves, so ds_ep can take a jax.vjp of the region from the m2
context (an AD transpose of Lineage's code outside that context raises ShardingTypeError).  Each is a nested jax.jit
with a stable identity, so the F / B / A / FB programs of one chunk structure trace each once: on the TPU7x a second
B program of one region lowered in 17.0 s against the first's 45.8 s, retracing nothing (one-host TPU7x 2x2x1,
2026-11; the region's lowering, not its tracing, is per program).

jax 0.11.1 trap: jnp.stack (lax.stack_p) of `unreduced`-typed values multiplies them by the product of the
unreduced axes' sizes (the type is right, the value is not; also in 0.11.2 and 0.12.0.dev).  Lineage's own driver
stacks its scanned layers' gradients that way (ops._interleave_arrays): fed gathered weights typed reduced, its
hooks leave them unreduced, and at n=5 on the local TPU7x its layers 3 and 4 come out 4x (the routed experts' 2x).
This module never stacks an unreduced value at jit level: the per-layer gradients are concatenated inside a
shard_map, on device-local arrays.

Lineage is imported lazily (`import_lineage`): this module imports where Lineage does not (its modules import
typeguard and jaxtyping, which MaxText's TPU requirements do not bring), and its tests that need Lineage skip there
-- MAXTEXT_MPMD_REQUIRE_LINEAGE=1 makes that a failure instead (ds_ep_lineage_test).
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import importlib
import os
import sys
import threading
import types
import weakref

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import ds_ep

FE = ds_ep.FE
ATTN = "attn"
# Lineage's logical axes (dsv3_sharding.LOGICAL_AXES) on the A3 view m3.  Lineage's typing rule (traced on CPU at
# DeepSeek-V3 sizes, run here on TPU): the token axes fsdp_attention + attention must equal expert + fsdp_moe, with
# expert and fsdp_moe disjoint -- transposing a dot against a weight `reduced` over R needs the contraction (token)
# dims sharded over exactly R, and the router / shared experts are collected over (expert, fsdp_moe), the pre- and
# post-attention norms and the q / kv down-projections over (fsdp_attention, attention), the q / kv norms over
# fsdp_attention alone (dsv3_sparse_layer._w_ici_axes).  'attention' must name a real mesh axis (MLA all-gathers its
# latents over it, dsv3_mla.q_down_projection: `()` fails with "unbound axis name"), so the size-1 'attn'; and a
# physical axis may carry a Lineage logical name only where the mapping is the identity (ops.physical_pspec re-maps
# known names, init_activation_bank's spec collides otherwise) -- 'expert' is.
AXIS_MAPPING = {"dcn": (), "attention": ATTN, "fsdp_attention": FE, "expert": "expert", "fsdp_moe": ("fsdp", ATTN)}
ACT2 = P(FE, None, None)  # ds_ep's activation on m2
ACT3 = P(FE, ATTN, None)  # Lineage's (dsv3_sharding.activation_sharding) on m3: the same blocks
_NO_MESH = jax.sharding.AbstractMesh((), ())  # the empty abstract mesh (Region's trace context)
# Lineage's MoE chunk size relative to a device's balanced load of routed slots (dropless whatever it is; the
# number of chunk-loop trips depends on it, and with more than one chunk the partial sums add in bf16,
# dsv3_sparse_layer._routed_experts_chunked_impl).  MaxText's DeepSeek-V3 Lineage recipe runs 3.0
# (deepseek3-671b-lineage.yml's capacity_factor).
CAPACITY_FACTOR = 2.0
# Lineage's cross-layer activation bank: rows for this many times the balanced gating activations of a call's
# layers (dsv3_sparse_layer.init_activation_bank); a chunk that overflows them goes to scratch rows and the
# backward recomputes its gate GMM -- a memory-for-recompute knob, bitwise neutral.  Not ds_ep's
# ragged_buffer_factor (the ring's token-dropping dispatch buffer), which Lineage has no counterpart of.  A
# forward-only program sizes the bank for no layer (Region's primal: `_chain_fwd`), whatever this is.
BANK_FACTOR = 1.2
# Per-program compiler options of a Lineage program on TPU (`compiler_options`).  The scoped-VMEM limit is
# deepseek3-671b-lineage.yml's (compile_xla_flags): at the default 32 MiB the splash dkv kernel at 2048 blocks fails
# to compile (CompileTimeScopedVmemOom, 33.46M > 32.00M); ds_ep's TPU7x production LIBTPU flags carry it, runs without
# them do not.  Lineage's other recipe flags (e.g. --xla_tpu_enable_offloading_copy_to_sparsecore=false) are not
# adopted, nor is its post-scheduler overlap pass, which is no flag (`import_lineage`).
COMPILER_OPTIONS = {"xla_tpu_scoped_vmem_limit_kib": "65536"}


# --------------------------------------------------------------- Lineage, lazily
_PKG = "maxtext.experimental.lineage"
_LIN = None
_LIN_LOCK = threading.Lock()


def import_lineage():
  """Lineage's modules this adapter calls, as a namespace (imported once).  ImportError where Lineage does not
  import: a module it imports is missing (typeguard or jaxtyping, which MaxText's TPU requirements do not bring).

  The sparse / dense layer chain imports no tensorflow: only lineage/dsv3.py, dsv3_model.py and
  interop/maxtext_layout.py reach xla_transforms' hlo_pb2 import, and none of them is loaded here.  So Lineage's
  post-scheduler XLA pass (xla_transforms/max_async_overlap_transform.py: every async start as early, every done as
  late as the schedule allows) is not registered either: only Lineage's whole-model entry point registers it
  (dsv3.dsv3; MaxText's use_lineage path always has it do so, lineage_adapter.dsv3_config_from_maxtext), and this
  adapter calls the layer functions instead.  Registered as dsv3.dsv3 registers it -- jax.extend.xla's
  register_hlo_module_transformation, POST_SCHEDULER, every platform -- it would rewrite every program the process
  compiles after it, ds_ep's own too.  Not tried: no time measured with this adapter has it."""
  global _LIN
  with _LIN_LOCK:
    if _LIN is None:
      try:
        mods = {
            name: importlib.import_module(f"{_PKG}.{name}")
            for name in (
                "ops",
                "dsv3_config",
                "dsv3_mla",
                "dsv3_router",
                "dsv3_types",
                "dsv3_sparse_layer",
                "dsv3_dense_layer",
            )
        }
        mods["expert_mlp"] = importlib.import_module(f"{_PKG}.gmmv2.expert_mlp")
      except ImportError as e:
        raise ImportError(
            f"ds_ep_lineage could not import MaxText's Lineage stack ({_PKG}): {e} -- a module it imports is "
            "missing (typeguard and jaxtyping are not among MaxText's TPU requirements)"
        ) from e
      _install_sparsecore_switch(mods)
      _LIN = types.SimpleNamespace(**mods)
    return _LIN


def available():
  """Is Lineage importable here?"""
  try:
    import_lineage()
  except ImportError:
    return False
  return True


def provenance():
  """Where Lineage comes from (path and git state) -> dict; {'lineage': None, 'error': ...} without it."""
  from maxtext.experimental.mpmd.jax import ds_ep_pipeline

  try:
    lin = import_lineage()
  except ImportError as e:
    return {"lineage": None, "error": str(e)}
  path = os.path.dirname(os.path.abspath(lin.ops.__file__))
  return {"lineage": {"path": path, **ds_ep_pipeline._git(path)}}


# --------------------------------------------------------------- SparseCore switch
# libtpu offloads collectives to SparseCore only when a program's chips are the whole slice or a solid rectangle
# away from chip (0,0,0) (ds_ep.sparsecore_offload_available); on any other stage Lineage's forced
# compute_on('tpu_sparsecore') regions abort libtpu 0.0.46's compiler (`Check failed: collective_offload_config !=
# std::nullopt`, SIGABRT).  `no_sparsecore()` traces those regions as plain (TensorCore) code: Lineage's modules
# call compute_on at trace time -- dsv3_router / dsv3_mla through their module aliases (dsv3_router.compute_on,
# dsv3_mla.compute_on), ops.py through jax.experimental.compute_on.compute_on -- so wrappers installed there (and on any
# other alias of jax's compute_on a Lineage module imported by then keeps, whatever its name: a later Lineage's) return
# an identity decorator for compute_type 'tpu_sparsecore' while the switch is on in this thread.  compute_on('device')
# regions (MLA's TensorCore collectives) are untouched, and nothing changes outside the context.  The flag is
# thread-local: ds_ep traces one program at a time on one thread (ds_ep_pipeline.Pipeline.precompile), and a trace
# on another thread keeps its own setting.  The wrappers hold the flag object itself, which a reloaded copy of this
# module adopts (its own new one would switch nothing).
_SC = threading.local()


def _sc_compute_on(orig, flag):
  def compute_on(f=None, *, compute_type, **kwargs):  # jax's signature: (f=None, *, compute_type, ...)
    if compute_type == "tpu_sparsecore" and getattr(flag, "off", False):
      return (lambda g: g) if f is None else f
    return orig(f, compute_type=compute_type, **kwargs)

  compute_on.mpmd_sparsecore_switch = orig
  compute_on.mpmd_sparsecore_flag = flag
  return compute_on


def _jax_compute_on(v):
  """`v` with the switch taken off (its original), or `v`."""
  try:
    return getattr(v, "mpmd_sparsecore_switch", v)
  except Exception:  # (an attribute that answers getattr oddly is no alias)
    return v


def _install_sparsecore_switch(mods):
  """Wrap jax.experimental.compute_on.compute_on and every alias of it the Lineage modules hold -- `mods`' and any
  other module of the package imported by now, under any attribute name -- in the switch, once: where a wrapper is
  there already, adopt its flag."""
  global _SC
  import jax.experimental.compute_on as jco

  own = _jax_compute_on(jco.compute_on)
  modules = {
      id(m): m
      for m in (
          *mods.values(),
          *(m for n, m in sorted(sys.modules.items()) if m is not None and (n == _PKG or n.startswith(_PKG + "."))),
      )
  }
  owners = [(jco, "compute_on")] + [
      (m, a) for m in modules.values() for a, v in sorted(vars(m).items()) if _jax_compute_on(v) is own
  ]
  for owner, attr in owners:
    f = getattr(owner, attr)
    if hasattr(f, "mpmd_sparsecore_switch"):
      _SC = f.mpmd_sparsecore_flag
    else:
      setattr(owner, attr, _sc_compute_on(f, _SC))


@contextlib.contextmanager
def no_sparsecore(off=True):
  """While active (and `off`), Lineage's compute_on('tpu_sparsecore') regions trace as plain code in this thread.
  Takes effect at trace time: wrap the tracing / lowering of a program, not its execution."""
  import_lineage()
  prev = getattr(_SC, "off", False)
  _SC.off = bool(off)
  try:
    yield
  finally:
    _SC.off = prev


def compiler_options(mesh):
  """The compiler options of a program running Lineage's layers on `mesh` -> dict on TPU (a compile-only topology's
  devices too), None elsewhere (other backends refuse libtpu options): COMPILER_OPTIONS plus ds_ep's
  barrier-keeping option (ds_ep.PIN_COMPILER_OPTIONS) -- Lineage orders its phases with program_order's
  optimization barriers and its backward depends on them, which the v5p recipe's
  --xla_tpu_aggressive_opt_barrier_removal=ENABLED would remove (deepseek3-671b-lineage.yml says so above its
  compile_xla_flags)."""
  pin = ds_ep.pin_compiler_options(mesh)
  return None if pin is None else {**pin, **COMPILER_OPTIONS}


# --------------------------------------------------------------- config
def check(cfg: DeepSeekConfig | None = None, scfg: ds_ep.StageConfig | None = None):
  """Raise ValueError naming every setting Lineage's layers cannot run (all of them, at once).

  Lineage routes DeepSeek-V3's way only: top-k (no RNG anywhere: ds_ep's default --routing random has no counterpart),
  sigmoid scores, among each token's best groups when n_routing_groups is set (a group scored by its two best
  post-bias scores, as ds_ep's: `routing`), dropless (ds_ep's ragged_buffer_factor > 0 drops tokens), one shared
  expert, num_kv_heads == num_query_heads (dsv3_config.DSv3Config.validate).  Its expert kernels take bf16 tokens as
  [N, D / 128, 128] tiles with D / 128 even (gmmv2.relayout.check_tile_d0, gmmv2.expert_mlp._check_3d), the routed
  weights in [rows / 8, 8, cols] tiles; experts split evenly over EP.  Its rope is split-half with always-truncated
  YaRN frequencies, its activations SiLU-gated.  No context parallelism (splash shards heads, never the sequence).  It
  returns no expert counts or aux loss through this adapter (aux.group_sizes / logits would carry them: a follow-up),
  and runs no MTP layer.  The stage needs Explicit mesh axes (shard_mode 'explicit', so grad_reduce 'step').  What
  only picks among ds_ep's own kernels is not refused but ignored (IGNORED_MODEL_FIELDS).  A Qwen3.5-family model
  (ds_ep.is_qwen) is refused alone: Lineage has none of its layers (its batch split is ds_ep.split_layers,
  batch_split_impl 'native').
  """
  cfg = scfg.model if cfg is None else cfg
  if ds_ep.is_qwen(cfg):
    # (none of the fields below is a Qwen35Config's: not a list of problems but a model Lineage does not build)
    raise ValueError(
        "Lineage's batch-split layers cannot run this: a Qwen3.5-family model -- Lineage builds "
        "DeepSeek-V3's layers only (MLA, a sigmoid-scored router, one ungated shared expert), and no "
        "Gated DeltaNet, gated attention or sigmoid-gated shared expert"
    )
  bad = []
  if jnp.dtype(cfg.dtype) != jnp.bfloat16:
    bad.append(f"dtype {jnp.dtype(cfg.dtype).name}: Lineage's expert kernels take bf16 activations only")
  if cfg.routed_score_func != "sigmoid":
    bad.append(f"routed_score_func {cfg.routed_score_func!r}: Lineage scores with a sigmoid only")
  g, tg, E, k = cfg.n_routing_groups, cfg.topk_routing_group, cfg.num_experts, cfg.num_experts_per_tok
  if (g > 0) != (tg > 0):
    bad.append(f"n_routing_groups {g} / topk_routing_group {tg}: set both or neither")
  elif g > 0:
    if E % g:
      bad.append(f"num_experts {E} is not a multiple of n_routing_groups {g}")
    elif E // g < 2:
      bad.append(f"{E // g} expert(s) a routing group: a group is scored by its two best experts")
    if tg > g:
      bad.append(f"topk_routing_group {tg} > n_routing_groups {g}")
    elif E % g == 0 and k > tg * (E // g):
      bad.append(f"num_experts_per_tok {k} > the {tg} x {E // g} experts of the chosen groups")
  if cfg.shared_experts != 1:
    bad.append(f"shared_experts {cfg.shared_experts}: want 1")
  if cfg.num_kv_heads != cfg.num_query_heads:
    bad.append(f"num_kv_heads {cfg.num_kv_heads} != num_query_heads {cfg.num_query_heads}")
  if cfg.emb_dim % 256:
    bad.append(f"emb_dim {cfg.emb_dim}: Lineage's expert kernels want a multiple of 256 (D / 128 even)")
  if cfg.moe_mlp_dim % 128:
    bad.append(
        f"moe_mlp_dim {cfg.moe_mlp_dim}: want a multiple of 128 (the routed weights' 8-row tiles, the " "gmmv2 lanes)"
    )
  if cfg.q_lora_rank <= 0:
    bad.append("q_lora_rank 0: Lineage's MLA always has the query LoRA")
  if tuple(cfg.mlp_activations) != ("silu", "linear"):
    bad.append(f"mlp_activations {tuple(cfg.mlp_activations)}: Lineage hard-codes silu x linear")
  if not cfg.rope_interleave:
    bad.append("rope_interleave False: the weight conversion de-interleaves MaxText's interleaved rope channels")
  if not cfg.rope_truncate or cfg.rope_attention_scaling:
    bad.append("rope_truncate False / rope_attention_scaling True: Lineage's YaRN always truncates, never scales")
  for name in ("rope_factor", "rope_max_timescale", "beta_fast", "beta_slow"):
    v = getattr(cfg, name)
    if float(v) != int(v) or int(v) <= 0:
      bad.append(f"{name} {v}: Lineage takes a positive integer")
  if cfg.routed_bias_update_rate:
    bad.append(
        f"routed_bias_update_rate {cfg.routed_bias_update_rate}: the expert counts (Lineage's "
        "aux.group_sizes) are not wired up"
    )
  if cfg.load_balance_loss_weight:
    bad.append(
        f"load_balance_loss_weight {cfg.load_balance_loss_weight}: the aux loss (Lineage's aux.logits) is " "not wired up"
    )
  if cfg.mtp_num_layers:
    bad.append(f"mtp_num_layers {cfg.mtp_num_layers}: no MTP layer on Lineage here")
  if scfg is not None:
    if scfg.random_routing:
      bad.append("random routing: Lineage routes top-k only (--routing topk)")
    if scfg.ragged_buffer_factor > 0:
      bad.append(
          f"ragged_buffer_factor {scfg.ragged_buffer_factor}: Lineage is dropless (its bank factor is "
          f"fixed at {BANK_FACTOR})"
      )
    if scfg.cp > 1:
      bad.append(f"cp {scfg.cp}: no context parallelism")
    if scfg.shard_mode != "explicit":
      bad.append(f"shard_mode {scfg.shard_mode!r}: Lineage reads layouts from sharding types: want 'explicit'")
    if cfg.num_experts % scfg.ep:
      bad.append(f"num_experts {cfg.num_experts} is not a multiple of ep {scfg.ep}")
  if bad:
    raise ValueError("Lineage's batch-split layers cannot run this: " + "; ".join(bad))


# What ds_ep's knobs of its own layers mean nothing to Lineage's (for help texts and run summaries): the experts'
# kernels and dispatch (gmmv2 with Lineage's tiles, its own SparseCore all-gather / reduce-scatter, dW accumulated in
# place across the halves and chunks: dsv3_experts.dsv3_routed_experts_bwd_impl), where the routing weights apply (its
# combine weighs the expert outputs), ds_ep's attention kernel and its softmax, and MoE numerics.  Lineage's splash is
# base-2 always (its kernel config's use_base2_exp default, splash_attention_kernel.SplashConfig, which
# dsv3_mla.init_splash_kernel never sets; DeepSeekConfig.sa_use_base2_exp says what that does), whichever softmax
# ds_ep's own kernel would compute -- the exact one under attention_kernel 'splash' or sa_use_base2_exp False: a known
# numerics difference, like the bf16 rope tables.
# (matmul_precision is not among them: the same programs' head -- ds_ep's own, head_loss -- reads it.)
IGNORED_STAGE_KNOBS = (
    "gmm_impl",
    "wi_tiling",
    "wo_tiling",
    "use_ragged_sort",
    "ep_collectives",
    "split_dw",
    "expert_weights",
    "route_weights",
)
IGNORED_MODEL_FIELDS = (
    "attention_kernel",
    "sa_use_base2_exp",
    "moe_impl",
    "gmm_fp32_accum",
    "gmm_exact_drhs",
    "float32_weight_sum",
    "moe_use_direct_token_gather",
    "sa_cost_estimate_flops_fwd",
    "sa_cost_estimate_flops_bwd",
)


def routing(cfg: DeepSeekConfig):
  """(n_routing_groups, topk_routing_group, topk_in_group) for Lineage's router.  It scores a group by the sum of
  its `topk_in_group` best scores, an argument (dsv3_router.expert_group_mask; only Lineage's own config derives
  it, as k / topk_routing_group, dsv3_config.DSv3ModelConfig.topk_in_group), and ds_ep by its two best post-bias
  scores, summed in fp32 (deepseek.moe._top_2_in_group_sum, MaxText's): 2, for any k (DeepSeek-V3's k / g is 8 / 4 =
  2 too).  Groups off (ds_ep's n_routing_groups -1) is one group of every expert, topk_in_group k: exactly plain
  top-k (Lineage has no off switch, dsv3_router.expert_indices_and_coeffs)."""
  if cfg.n_routing_groups <= 0:
    return 1, 1, cfg.num_experts_per_tok
  return cfg.n_routing_groups, cfg.topk_routing_group, 2


def sa_blocks(cfg: DeepSeekConfig):
  """Splash's six blocks from ds_ep's config: q, kv, kv compute, then the fused backward's q, kv, kv compute, where 0
  means the forward's (as deepseek.attention.tokamax_splash_attention reads them)."""
  return (
      cfg.sa_block_q,
      cfg.sa_block_kv,
      cfg.sa_block_kv_compute,
      cfg.sa_block_q_dkv or cfg.sa_block_q,
      cfg.sa_block_kv_dkv or cfg.sa_block_kv,
      cfg.sa_block_kv_dkv_compute or cfg.sa_block_kv_compute,
  )


# --------------------------------------------------------------- the A3 view
def a3_mesh(mesh: Mesh):
  """ds_ep's stage mesh ('fsdp', 'expert'), Explicit -> the same devices, same order, with a trailing size-1 'attn'
  axis: the mesh Lineage's code runs on (AXIS_MAPPING)."""
  if tuple(mesh.axis_names) != FE or any(t != AxisType.Explicit for t in mesh.axis_types):
    raise ValueError(
        f"Lineage's layers run on an Explicit ('fsdp', 'expert') stage mesh (shard_mode 'explicit'), "
        f"got axes {mesh.axis_names} of types {mesh.axis_types}"
    )
  if jax.config.jax_remove_size_one_mesh_axis_from_type:
    # Lineage's recipe pins it False too (deepseek3-671b-lineage.yml's remove_size_one_mesh_axis_from_type): MLA
    # all-gathers over the attention axis it reads from the types, which this flag would strip of 'attn'.
    raise ValueError(
        "the A3 view's size-1 'attn' axis needs jax_remove_size_one_mesh_axis_from_type False (the " "default)"
    )
  return Mesh(mesh.devices[..., None], (*FE, ATTN), axis_types=(AxisType.Explicit,) * 3)


def _physical(axes):
  """Lineage logical axes -> the physical m3 axes AXIS_MAPPING maps them to."""
  out = set()
  for a in axes:
    v = AXIS_MAPPING[a]
    out.update((v,) if isinstance(v, str) else v)
  return frozenset(out)


# The leaves whose head dim Lineage shards over 'attention' (per layer: q_up / k_up / v_up [rank, heads, dim],
# out [heads, v, emb]; dsv3_sharding._MLA_8X8X8), and the routed experts' expert dim (dim 0) over 'expert'.
_HEAD_DIM = {"q_up": 1, "k_up": 1, "v_up": 1, "out": 0}


def lineage_types(axes_tree, shapes_tree, kind, view="m3"):
  """One layer's collected Lineage weights typed `kind` ('reduced' for the weights, 'unreduced' for their
  gradients) over the axes Lineage collects each along (`axes_tree`: dsv3_sparse_layer._w_ici_axes() /
  dsv3_dense_layer._w_ici_axes()), head dims on 'attn' and the routed experts on 'expert' -> a tree of
  PartitionSpecs on m3; `view` 'm2' is the same blocks on ds_ep's mesh (no 'attn')."""

  def one(path, axes, shape):
    name = getattr(path[-1], "name", None)
    dims = [None] * len(shape.shape)
    if name in _HEAD_DIM:
      dims[_HEAD_DIM[name]] = ATTN
    if len(path) > 1 and getattr(path[-2], "name", None) == "routed":
      dims[0] = "expert"
    red = _physical(axes)
    if view == "m2":
      dims, red = [None if d == ATTN else d for d in dims], red - {ATTN}
    return P(*dims, **({kind: red} if red else {}))

  return jax.tree_util.tree_map_with_path(one, axes_tree, shapes_tree, is_leaf=lambda x: isinstance(x, tuple))


def layer_kwargs(cfg: DeepSeekConfig, mesh3: Mesh, capacity_factor=CAPACITY_FACTOR):
  """The keyword arguments of dsv3_sparse_layer's scan-body / epilogue functions (dsv3_model.dsv3_layer_kwargs's
  values from ds_ep's config).  Traps: Lineage's qk_head_dim is the NOPE dim (DeepSeekConfig.qk_head_dim is nope +
  rope); rope_factor an int; the router kernel stays bf16, router_dtype fp32 under float32_gate_logits; rope_fn must
  be dsv3_mla.yarn itself (an identity check enables its fused tail, dsv3_mla.q_up_from_q_down).  quant None:
  Lineage's layers run unquantized, in bf16, where MaxText's DeepSeek-V3 Lineage recipe runs fp8 (lineage_quantization
  fp8_full in deepseek3-671b-lineage.yml: the routed experts' GMMs with the token all-gather feeding them, and the MLA
  projections but k up -- quantization.FP8_FULL); ds_ep's model has no fp8 path."""
  lin = import_lineage()
  groups, topk_groups, in_group = routing(cfg)
  return dict(
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=float(cfg.routed_scaling_factor),
      n_routing_groups=groups,
      topk_routing_group=topk_groups,
      topk_in_group=in_group,
      qk_head_dim=cfg.qk_nope_head_dim,
      rope_head_dim=cfg.qk_rope_head_dim,
      num_query_heads=cfg.num_query_heads,
      mscale=float(cfg.mscale),
      kv_lora_rank=cfg.kv_lora_rank,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      rope_factor=int(cfg.rope_factor),
      mesh=mesh3,
      norm_fn=functools.partial(lin.ops.rms_norm, epsilon=cfg.normalization_layer_epsilon),
      rope_fn=lin.dsv3_mla.yarn,
      gmm_fn=lin.expert_mlp.make_gmm_fn(emb_dim=cfg.emb_dim),
      axis_mapping=AXIS_MAPPING,
      expert_axis_name="expert",
      capacity_factor=float(capacity_factor),
      router_dtype=jnp.float32 if cfg.float32_gate_logits else None,
      quant=None,
  )


# The subset the prologue functions and the dense layer take.
MLA_KWARGS = (
    "qk_head_dim",
    "rope_head_dim",
    "num_query_heads",
    "mscale",
    "kv_lora_rank",
    "max_position_embeddings",
    "original_max_position_embeddings",
    "rope_factor",
    "mesh",
    "norm_fn",
    "rope_fn",
    "axis_mapping",
    "quant",
)


def kernels(cfg: DeepSeekConfig, mesh3: Mesh, batch, seq):
  """dsv3_model.init_kernels from ds_ep's config, traced under m3's abstract mesh -> (yarn_freqs, splash_kernel):
  YaRN cos / sin [batch, seq, 1, rope] for positions 0..seq-1 every row (ds_ep's segment ids are all ones, Lineage's
  None, which is the same mask), and the causal splash kernel on m3 with ds_ep's blocks (`sa_blocks`) and Lineage's
  defaults for the rest (dsv3_config.DSv3KernelConfig)."""
  lin = import_lineage()
  pos = jax.reshard(jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (batch, seq)), NamedSharding(mesh3, P(FE, None)))
  yarn = lin.dsv3_mla.get_yarn_freqs(
      pos,
      rope_head_dim=cfg.qk_rope_head_dim,
      rope_theta=int(cfg.rope_max_timescale),
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      beta_fast=int(cfg.beta_fast),
      beta_slow=int(cfg.beta_slow),
      rope_factor=int(cfg.rope_factor),
      out_pspec=P("fsdp_attention", None, None),
      mesh=mesh3,
      axis_mapping=AXIS_MAPPING,
      dtype=cfg.dtype,
  )
  k = lin.dsv3_config.DSv3KernelConfig(max_target_length=seq)
  splash = lin.dsv3_mla.init_splash_kernel(
      *sa_blocks(cfg),
      k.sa_q_layout,
      k.sa_k_layout,
      k.sa_v_layout,
      max_target_length=seq,
      num_query_heads=cfg.num_query_heads,
      kernel_out_spec=P("attention", None),
      mesh=mesh3,
      axis_mapping=AXIS_MAPPING,
      qk_diag_skip=k.qk_diag_skip,
      qk_diag_grid=k.qk_diag_grid,
      sv_diag_skip=k.sv_diag_skip,
      diag_band_skip=k.diag_band_skip,
      dq_reduction_steps=k.dq_reduction_steps,
      q_prescaled_log2e=k.q_prescaled_log2e,
  )
  return yarn, splash


# --------------------------------------------------------------- weights
# MaxText's layout (ds_ep's, with ds_ep's leaf names: ds_ep.layer_shapes) vs Lineage's collected one (dsv3_types), per
# Lineage's interop.maxtext_layout (_deinterleave_rope and the layer converters): the trailing rope channels of wq_b
# and wkv_a interleaved (2i, 2i+1) there, split-half (i, i + R/2) here; wkv_b fused there, k_up / v_up here; wi_0 /
# wi_1 separate there, one routed gate [.., E, D, 2F] here (SiLU on the first half, ops._ragged_silu_mul_kernel),
# viewed as [.., E, D/8, 8, 2F] tiles (dsv3_sparse_layer.routed_w_to_tiles; linear too); the router bias a weight leaf
# there, Lineage's moe.router.bias here.  Every transform acts on a trailing axis ds_ep's compute layout leaves
# unsharded, so all run per shard, on any number of leading (layer) axes, and the gradient transforms are their exact
# inverses.  No rope channels (qk_rope_head_dim 0) leave w as it is, as Lineage's converter does: w[..., -0:] would be
# all of it.
def _deinterleave_rope(w, rope):
  if rope <= 0:
    return w
  r = w[..., -rope:]
  return jnp.concatenate([w[..., :-rope], r[..., 0::2], r[..., 1::2]], axis=-1)


def _interleave_rope(w, rope):
  if rope <= 0:
    return w
  r, half = w[..., -rope:], rope // 2
  # concatenate, not jnp.stack: right on unreduced values outside a shard_map too (module doc)
  pairs = jnp.concatenate([r[..., :half, None], r[..., half:, None]], axis=-1)
  return jnp.concatenate([w[..., :-rope], pairs.reshape(*r.shape[:-1], rope)], axis=-1)


def _mla_from_maxtext(att, cfg):
  lin = import_lineage()
  kv = att["wkv_b"]["kernel"]
  R, N = cfg.qk_rope_head_dim, cfg.qk_nope_head_dim
  return lin.dsv3_types.DSv3MLAWeightsPytree(
      q_down=att["wq_a"]["kernel"],
      q_up=_deinterleave_rope(att["wq_b"]["kernel"], R),
      q_norm_scale=att["q_norm"]["scale"],
      kv_down=_deinterleave_rope(att["wkv_a"]["kernel"], R),
      k_up=kv[..., :N],
      v_up=kv[..., N:],
      kv_norm_scale=att["kv_norm"]["scale"],
      out=att["out"]["kernel"],
  )


def _mla_to_maxtext(g, cfg):
  R = cfg.qk_rope_head_dim
  return {
      "wq_a": {"kernel": g.q_down},
      "q_norm": {"scale": g.q_norm_scale},
      "wq_b": {"kernel": _interleave_rope(g.q_up, R)},
      "wkv_a": {"kernel": _interleave_rope(g.kv_down, R)},
      "kv_norm": {"scale": g.kv_norm_scale},
      "wkv_b": {"kernel": jnp.concatenate([g.k_up, g.v_up], axis=-1)},
      "out": {"kernel": g.out},
  }


def router_bias_dtype(cfg: DeepSeekConfig):
  """The dtype of the router bias Lineage gets: ds_ep's compute dtype of the leaf (ds_ep.leaf_dtypes: fp32 under
  float32_gate_logits, as MaxText keeps it).  A model without a routed bias gets zeros of it -- in bf16 unless
  float32_gate_logits, since an fp32 bias would promote the whole selection to fp32
  (dsv3_router._expert_selection_impl)."""
  return jnp.dtype(jnp.float32 if cfg.float32_gate_logits else cfg.dtype)


def sparse_from_maxtext(lw, cfg: DeepSeekConfig, tiles=True):
  """One MoE layer's (or a [n, ...] stack's) ds_ep compute weights -> Lineage's DSv3SparseLayerWeightsPytree in
  its collected layout, routed experts in the tiles view unless `tiles` False.  The router kernel is cast to bf16
  (Lineage computes in fp32 on a bf16 kernel under float32_gate_logits, dsv3_router.expert_selection); the router bias
  is ds_ep's `moe.gate.bias` where the model has one, else zeros (`router_bias_dtype`)."""
  lin = import_lineage()
  T = lin.dsv3_types
  moe, sh = lw["moe"], lw["moe"]["shared_experts"]
  kernel = moe["gate"]["kernel"]
  bias = moe["gate"].get("bias")
  if bias is None:
    bias = jnp.zeros(kernel.shape[:-2] + kernel.shape[-1:], router_bias_dtype(cfg))
  routed = T.DSv3MoERoutedExpertWeightsPytree(gate=jnp.concatenate([moe["wi_0"], moe["wi_1"]], axis=-1), linear=moe["wo"])
  return T.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=lw["pre_self_attention_layer_norm"]["scale"],
      mla=_mla_from_maxtext(lw["self_attention"], cfg),
      post_attn_norm_scale=lw["post_self_attention_layer_norm"]["scale"],
      moe=T.DSv3MoEWeightsPytree(
          router=T.DSv3MoERouterWeightsPytree(kernel=kernel.astype(cfg.dtype), bias=bias),
          shared=T.DSv3MoESharedExpertWeightsPytree(
              gate_0=sh["wi_0"]["kernel"], gate_1=sh["wi_1"]["kernel"], linear=sh["wo"]["kernel"]
          ),
          routed=lin.dsv3_sparse_layer.routed_w_to_tiles(routed) if tiles else routed,
      ),
  )


def sparse_to_maxtext(g, cfg: DeepSeekConfig, bias=False, tiles=True):
  """The inverse of `sparse_from_maxtext` for gradients: a DSv3SparseLayerWeightsPytree of gradients -> ds_ep's
  tree.  The router kernel's gradient leaves in ds_ep's compute dtype of the kernel (fp32 under
  float32_gate_logits, ds_ep.leaf_dtypes); `bias`: the tree has ds_ep's routed bias leaf, whose gradient is zero (it
  only picks the experts: stop_gradient, dsv3_router._expert_selection_impl) -- the slot ds_ep fills with expert
  counts when it updates the bias, not wired up here (`check`)."""
  lin = import_lineage()
  routed = lin.dsv3_sparse_layer.routed_w_from_tiles(g.moe.routed) if tiles else g.moe.routed
  f = routed.gate.shape[-1] // 2
  kernel = g.moe.router.kernel.astype(jnp.float32 if cfg.float32_gate_logits else cfg.dtype)
  gate = {"kernel": kernel}
  if bias:
    gate["bias"] = jnp.zeros(kernel.shape[:-2] + kernel.shape[-1:], router_bias_dtype(cfg))
  return {
      "pre_self_attention_layer_norm": {"scale": g.pre_attn_norm_scale},
      "post_self_attention_layer_norm": {"scale": g.post_attn_norm_scale},
      "self_attention": _mla_to_maxtext(g.mla, cfg),
      "moe": {
          "gate": gate,
          "wi_0": routed.gate[..., :f],
          "wi_1": routed.gate[..., f:],
          "wo": routed.linear,
          "shared_experts": {
              "wi_0": {"kernel": g.moe.shared.gate_0},
              "wi_1": {"kernel": g.moe.shared.gate_1},
              "wo": {"kernel": g.moe.shared.linear},
          },
      },
  }


def dense_from_maxtext(lw, cfg: DeepSeekConfig):
  """One dense layer's (or a stack's) ds_ep compute weights (ds_ep.dense_layer_shapes) -> Lineage's
  DSv3DenseLayerWeightsPytree: the MLA as the MoE layer's, the dense MLP's wi_0 / wi_1 / wo as gate_0 / gate_1 /
  linear (not fused, unlike the routed experts)."""
  lin = import_lineage()
  T = lin.dsv3_types
  mlp = lw["mlp"]
  return T.DSv3DenseLayerWeightsPytree(
      pre_attn_norm_scale=lw["pre_self_attention_layer_norm"]["scale"],
      mla=_mla_from_maxtext(lw["self_attention"], cfg),
      post_attn_norm_scale=lw["post_self_attention_layer_norm"]["scale"],
      mlp=T.DSv3MLPWeightsPytree(gate_0=mlp["wi_0"]["kernel"], gate_1=mlp["wi_1"]["kernel"], linear=mlp["wo"]["kernel"]),
  )


def dense_to_maxtext(g, cfg: DeepSeekConfig):
  """The inverse of `dense_from_maxtext` (for gradients)."""
  return {
      "pre_self_attention_layer_norm": {"scale": g.pre_attn_norm_scale},
      "post_self_attention_layer_norm": {"scale": g.post_attn_norm_scale},
      "self_attention": _mla_to_maxtext(g.mla, cfg),
      "mlp": {"wi_0": {"kernel": g.mlp.gate_0}, "wi_1": {"kernel": g.mlp.gate_1}, "wo": {"kernel": g.mlp.linear}},
  }


# --------------------------------------------------------------- the driver
def _chain_fwd(wl, h3, yarn, splash, kw, bank_layers=None):
  """Lineage's batch-split forward of len(wl) >= 1 MoE layers on `h3` (m3, ACT3) -> (y3, aux, residuals).

  The halves are each device's first and second half of its rows (ops.split_microbatches: ds_ep's h[:half] /
  h[half:]).  Each half's activation bank holds `BANK_FACTOR` x the balanced gating activations of `bank_layers`
  layers (init_activation_bank's num_layers; None: all the call's), then one MoE chunk of scratch rows
  (dsv3_sparse_layer.init_activation_bank).  Each expert chunk's gate GMM writes its output at the chunk's bank row
  and the SiLU reads it right back (dsv3_experts.dsv3_routed_experts_impl); the backward reads it there again, or -- a
  chunk that overflowed the checkpoint rows into the scratch ones, which the next chunk overwrites -- recomputes the
  gate GMM (dsv3_experts._retrieve_or_remat_gate_out, dsv3_routed_experts_bwd_impl).  `bank_layers` 0 is the
  forward-only bank (Region's primal): no checkpoint row, so every chunk overflows (dsv3_experts.bank_chunk_row) and
  the scratch chunk is all the bank -- the same y bitwise, 0.60-0.79 GiB less temp per F program at the TPU7x bench
  chunk of `Region`'s docstring; residuals from it would make the backward recompute every gate GMM.  prologue(w0)
  runs mb0's attention of layer 0; scan_body(w_l, w_l+1) mb0's MoE(l) + attention(l+1) and mb1's attention(l) +
  MoE(l); epilogue(w_n-1) the rest of the last layer -- the calls Lineage's driver makes
  (ops.w_collect_pipelined_scan_prologue_epilogue_fwd), at any n.  As dsv3_sparse_layers does (its scan body,
  dsv3_sparse_layers.dsv3_sparse_layers), each scan-body residual keeps mb0's MoE input as one sum (out_proj_out +
  orig_x, orig_x None): the backward takes either form (dsv3_sparse_layer.dsv3_sparse_layer_scan_body_bwd), the fold
  saves an activation per layer.  aux[l] is layer l's routing (each scan body returns its layer's; the epilogue the
  last's).  Consecutive scan bodies are ordered by Lineage's step barriers (its scan unrolls steps in pairs: the
  optimization barriers in the scan bodies of ops.w_collect_pipelined_scan_prologue_epilogue and of its backward --
  here the next layer's weights wait for the previous step, in the backward each step's weights for the later step's
  results): the same math, bit for bit in a program that also sums its gradients in fp32 over the devices, not in
  ds_ep's raw B program (dh and the summed dW differ there); on the TPU7x (D 2048, 4 layers) a B program holds 0.1 GiB
  less temp with them, and the summing one 9.7 GiB instead of 14.3 (step times equal)."""
  lin = import_lineage()
  SL, ops, mesh = lin.dsv3_sparse_layer, lin.ops, kw["mesh"]
  mla_kw = {k: kw[k] for k in MLA_KWARGS}
  n = len(wl)
  mb0, mb1 = ops.split_microbatches(h3, mesh=mesh)
  y01 = ops.split_microbatches(yarn, mesh=mesh)
  (b0, o0), (b1, o1) = SL.init_activation_bank(
      mb0,
      wl[0],
      num_layers=n if bank_layers is None else bank_layers,
      num_experts_per_tok=kw["num_experts_per_tok"],
      ragged_buffer_factor=BANK_FACTOR,
      capacity_factor=kw["capacity_factor"],
      expert_axis_name=kw["expert_axis_name"],
      axis_mapping=AXIS_MAPPING,
      mesh=mesh,
  )
  carry, res_pro = SL.dsv3_sparse_layer_prologue_fwd(
      (mb0, mb1), wl[0], y01, splash, banks=(b0, b1), bank_offsets=(o0, o1), **mla_kw
  )
  res_body, auxs, w_curr = [], [], wl[0]
  for l in range(n - 1):
    w_next = wl[l + 1]
    if l:  # Lineage's step barrier (its forward's): the next layer's weights wait for the previous step
      (carry, auxs[-1]), w_next = jax.lax.optimization_barrier(((carry, auxs[-1]), w_next))
    (carry, aux), res = SL.dsv3_sparse_layer_scan_body_fwd(carry, w_curr, w_next, y01, splash, **kw)
    moe0 = res.mb0.moe
    res = res._replace(mb0=res.mb0._replace(moe=moe0._replace(out_proj_out=moe0.out_proj_out + moe0.orig_x, orig_x=None)))
    res_body.append(res)
    auxs.append(aux)
    w_curr = w_next
  ((y0, y1), aux), res_epi = SL.dsv3_sparse_layer_epilogue_fwd(carry, w_curr, y01, splash, **kw)
  auxs.append(aux)
  # The aux leaves are plain (not unreduced) values: concatenating [None] slices is jnp.stack's result anyway.
  aux = jax.tree.map(lambda *a: jnp.concatenate([x[None] for x in a], axis=0), *auxs)
  return ops.merge_microbatches(y0, y1, mesh=mesh), aux, (res_pro, res_body, res_epi)


def _chain_bwd(wl, res, dy3, yarn, splash, kw):
  """The backward of `_chain_fwd` -> (per-layer weight gradients, dh3): Lineage's backward functions in reverse
  with identity reductions (ops.w_collect_pipelined_scan_prologue_epilogue_bwd with every hook the identity).
  Gradients come out half a layer skewed: mb0's attention gradient of layer l is computed by scan_body_bwd(l-1), which
  returns layer l's full gradient (`w_next_grad_partial`, through reduce_ici_first_fn = identity); layer 0's is the
  prologue's attention part plus scan_body_bwd(0)'s rest (combine_w_grads, at the end of that backward); with nothing
  pending (w_pending_grad_partial None) reduce_ici_second_fn is never called (dsv3_sparse_layer_scan_body_bwd's
  phases 5 and 7).  No aux loss: the aux cotangent carries no logits."""
  lin = import_lineage()
  SL, ops, mesh = lin.dsv3_sparse_layer, lin.ops, kw["mesh"]
  no_aux = lin.dsv3_types.DSv3RouterAux(group_sizes=None, selected_experts=None, logits=None)
  res_pro, res_body, res_epi = res
  n = len(wl)
  y01 = ops.split_microbatches(yarn, mesh=mesh)
  bkw = dict(kw, yarn_freqs=y01, splash_kernel=splash)
  gcarry, w_next = SL.dsv3_sparse_layer_epilogue_bwd(
      res_epi, (ops.split_microbatches(dy3, mesh=mesh), no_aux), wl[-1], **bkw
  )
  grads = [None] * n
  for l in reversed(range(n - 1)):
    w_l = wl[l]
    if l < n - 2:  # Lineage's step barrier (its backward's): this step's weights wait for the later step's results
      (gcarry, grads[l + 2], w_next), w_l = jax.lax.optimization_barrier(((gcarry, grads[l + 2], w_next), w_l))
    gcarry, w_curr, grads[l + 1], pending = SL.dsv3_sparse_layer_scan_body_bwd(
        res_body[l], (gcarry, no_aux), w_l, wl[l + 1], w_next, lambda g: g, None, None, **bkw
    )
    assert pending is None
    w_next = w_curr
  (gx0, gx1), g0 = SL.dsv3_sparse_layer_prologue_bwd(
      res_pro, gcarry, wl[0], yarn_freqs=y01, splash_kernel=splash, **{k: kw[k] for k in MLA_KWARGS}
  )
  grads[0] = ops.combine_w_grads(g0, w_next)
  return grads, ops.merge_microbatches(gx0, gx1, mesh=mesh)


# --------------------------------------------------------------- the region
def _drop_bias(w):
  return dataclasses.replace(w, moe=dataclasses.replace(w.moe, router=dataclasses.replace(w.moe.router, bias=None)))


def _same_type(name, x, spec):
  got = jax.typeof(x).sharding.spec
  if got != spec:
    raise ValueError(f"{name}: typed {got}, want {spec}")


class Region:
  """A chunk's decoder layers on Lineage's code: `region(w_layers, w_dense, h) -> (y, aux)`, differentiable.

  Global view on ds_ep's stage mesh `mesh` (Explicit ('fsdp', 'expert')): `w_layers` / `w_dense` are the chunk's
  'layers' / 'dense_layers' compute copies as ds_ep's explicit programs take them (MaxText layout, typed
  ds_ep.chunk_wc_specs: reduced over each leaf's partial axes; `w_dense` None without dense layers), `h` [B, S, D]
  bf16 P(FE, None, None) with an even number of rows a device.  The dense layers run first, on the whole
  microbatch (Lineage's dsv3_dense_layer, as ds_ep's split chunk runs its dense layers unsplit), then the MoE
  layers batch-split (`_chain_fwd`).  `y` is P(FE, None, None); `aux` Lineage's routing per MoE layer
  (DSv3RouterAux: group_sizes [n, devices, E] -- each device's tokens' expert counts -- and selected_experts
  [n, B, S, k], on m2; logits None), not differentiated.  Under jax.vjp the cotangents come out in ds_ep's layout
  typed ds_ep.chunk_gacc_specs(explicit=True) -- unreduced over the partial axes, each device's partial sum, ready
  for ds_ep's accumulators -- and dh P(FE, None, None); the router bias' cotangent is zero.

  A jax.custom_vjp: the forward rule converts the weights, reshards everything onto the A3 view and runs the chain under
  m3's abstract mesh; the backward rule runs Lineage's explicit backward functions under it and returns the gradients
  converted back.  The dense layers' backward is a jax.vjp of each layer, recomputed from its input: the forward saves
  only the dense layers' inputs, so none of their activations lives across the MoE chain's forward and backward -- one
  extra dense-layer forward per backward program for that memory.  (Their pullbacks could be residuals instead:
  jax.vjp's VJP object crosses from one jit into another.)  The primal -- what a program runs that only calls the region
  (ds_ep's F programs; jax.vjp and jax.checkpoint differentiate through the forward rule) -- is the forward rule's
  forward without residuals and with Lineage's activation bank sized for no layer (`_chain_fwd`): the bank holds each
  half's gate-GMM outputs for the backward to read instead of recomputing them, so a forward-only program would fill it
  and never read it.  On the TPU7x at a bench chunk (D 2048, F 1024, 32 experts, S 4096, 2 rows a device, 2 MoE
  layers) an F program's temp drops 2.434 -> 1.834 GiB on stage 1, exactly the two halves' banks, and 2.478 -> 1.691 on
  stage 0 (no SparseCore), y bitwise equal and its time the same (medians of 50: 12.05 vs 12.05 ms, 14.56 vs 14.61).
  The primal and both rules are nested jits held by this object, so every program that calls it traces each once per
  input shape; construct one per chunk structure and stage (`region` caches them) -- and in one context (`_fwd`), so a
  program with a mesh context, or a jax.checkpoint around the region, reuses them too.  The SparseCore switch is part of
  the identity: `sparsecore` False (default: whether libtpu offloads this mesh's collectives,
  ds_ep.sparsecore_offload_available, asked once a mesh) traces under `no_sparsecore()`.  Compile the programs with
  `compiler_options(mesh)`.

  Numerics.  Lineage's math, which is not ds_ep's (`IGNORED_STAGE_KNOBS`): against ds_ep's unsplit layers at top-k
  routing on the TPU7x, with the routing pinned by a routed bias, the gradients agree to cosine >= 0.99994 and 1.6e-2
  a leaf (1-5 layers, dense ones too); routed freely they do not -- the two route a few near-tie tokens differently
  (0.2-0.5% of a layer's tokens on fp32 router scores, 2.7% on bf16 ones, more in each later layer), which moves the
  router kernel's gradient by 0.06-0.14 and the cosine to 0.996-0.9995 (1-5 layers): compare pinned.  A MoE-only
  region's forward is bitwise the same in every program; with dense layers it is not (3e-3 between a forward-only and
  a backward program, about one bf16 ulp): Lineage's dense layer has no optimization barriers, so XLA rounds it by
  what else the program computes, as it does Lineage's own dense layers.
  """

  def __init__(
      self,
      scfg: ds_ep.StageConfig,
      mesh: Mesh,
      n_moe: int,
      n_dense: int = 0,
      routed_bias: bool = False,
      capacity_factor: float = CAPACITY_FACTOR,
      sparsecore: bool | None = None,
  ):
    check(scfg.model, scfg)
    if n_moe < 1:
      raise ValueError(
          f"a Lineage region runs at least one MoE layer (n_moe {n_moe}): Lineage's batch split is a " "MoE layer's"
      )
    if not capacity_factor > 0:
      raise ValueError(f"capacity_factor must be > 0, got {capacity_factor}")
    if tuple(mesh.shape.values()) != (scfg.fsdp, scfg.ep):
      raise ValueError(f"mesh {dict(mesh.shape)} is not fsdp {scfg.fsdp} x ep {scfg.ep}")
    self.lin = lin = import_lineage()
    self.scfg, self.cfg = scfg, scfg.model
    self.mesh, self.mesh3 = mesh, a3_mesh(mesh)
    self.n_moe, self.n_dense, self.routed_bias = n_moe, n_dense, routed_bias
    self.capacity_factor = float(capacity_factor)
    self.sparsecore = _sparsecore_offload(mesh) if sparsecore is None else bool(sparsecore)
    ch = ds_ep.ChunkSpec(
        index=0, layers=tuple(range(n_moe)), first=False, last=False, stage=0, dense=n_dense, routed_bias=routed_bias
    )
    self.wc_specs = ds_ep.chunk_wc_specs(scfg, ch)
    self.gacc_specs = ds_ep.chunk_gacc_specs(ch, explicit=True)
    self.kw = layer_kwargs(self.cfg, self.mesh3, self.capacity_factor)
    # one layer's Lineage shapes, for its types
    sds = lambda shapes, dt: jax.tree.map(lambda s: jax.ShapeDtypeStruct(s, dt), shapes, is_leaf=ds_ep._is_shape)
    dt = self.cfg.dtype
    sparse = jax.eval_shape(
        functools.partial(sparse_from_maxtext, cfg=self.cfg), sds(ds_ep.layer_shapes(self.cfg, routed_bias), dt)
    )
    dense = jax.eval_shape(
        functools.partial(dense_from_maxtext, cfg=self.cfg), sds(ds_ep.dense_layer_shapes(self.cfg), dt)
    )
    s_axes, d_axes = lin.dsv3_sparse_layer._w_ici_axes(), lin.dsv3_dense_layer._w_ici_axes()
    self.types = {
        (kind, view, kd): lineage_types(axes, shapes, kind, view)
        for kd, axes, shapes in (("sparse", s_axes, sparse), ("dense", d_axes, dense))
        for kind in ("reduced", "unreduced")
        for view in ("m2", "m3")
    }
    self._prim_jit = jax.jit(self._prim_impl)
    self._fwd_jit = jax.jit(self._fwd_impl)
    self._bwd_jit = jax.jit(self._bwd_impl)
    self._call = jax.custom_vjp(self._primal)
    self._call.defvjp(self._vjp_fwd, self._vjp_bwd)

  def __call__(self, w_layers, w_dense, h):
    return self._call(w_layers, w_dense, h)

  # ------------------------------------------------------------- the custom_vjp's primal and rules
  # Every call of the three jits runs under the empty abstract mesh, whatever the caller's context: the mesh context is
  # part of a jit's trace key -- no context and an explicitly set empty mesh are two keys -- and jax 0.11.1 calls a
  # custom_vjp's primal and rules in different ones: at ds_ep's top level the primal and the forward rule with no
  # context, the backward rule and jax.checkpoint's forward rule under a set empty mesh; in a program under a mesh
  # (use_abstract_mesh / set_mesh) all three under that mesh.  Each new context would trace Lineage again (Phase 1
  # measured a program under m2's mesh tracing both rules once more, a jax.checkpoint'ed region its forward twice).  For
  # Lineage's code the contexts are the same: get_abstract_mesh() reads none as the empty mesh.  Which programs trace
  # what: one without AD (ds_ep's F) the primal only; jax.vjp the two rules; a jax.checkpoint'ed region the primal too
  # -- jax.checkpoint stages the region out, as a jit does -- though it differentiates through the forward rule alone
  # (its program runs `_fwd_impl` twice, DCE'd to (y, aux) for the forward and whole for the recompute, and `_prim_impl`
  # never).  So a structure with an F and a B program traces its forward twice, at the two bank sizes (CPU, 2 MoE
  # layers: the primal 8.5-11.5 s beside the rules' 28-32 s; the pipeline test's V=1 split precompile took 180.6 s,
  # 179.8-197 s before the forward-only bank), and one under jax.checkpoint without an F program (V=1's last chunk with
  # remat_fused_last) traces a primal it never lowers (that test's --remat-last split precompile: 196.6 s, 180.6 s
  # before).
  def _fwd(self, w_layers, w_dense, h):
    with jax.sharding.use_abstract_mesh(_NO_MESH):
      return self._fwd_jit(w_layers, w_dense, h)

  def _bwd(self, res, dy):
    with jax.sharding.use_abstract_mesh(_NO_MESH):
      return self._bwd_jit(res, dy)

  def _primal(self, w_layers, w_dense, h):
    with jax.sharding.use_abstract_mesh(_NO_MESH):
      return self._prim_jit(w_layers, w_dense, h)

  def _vjp_fwd(self, w_layers, w_dense, h):
    return self._fwd(w_layers, w_dense, h)

  def _vjp_bwd(self, res, cts):
    return self._bwd(res, cts[0])

  # ------------------------------------------------------------- conversions on m2, reshards to / from m3
  def _to_lineage(self, w, dense):
    """ds_ep's stacked compute tree (m2, typed reduced) -> one Lineage tree per layer on m3 (A3 types): converted
    in a shard_map on m2 (shard-local; typed in and out like ds_ep's explicit programs), then resharded."""
    kd = "dense" if dense else "sparse"
    n = self.n_dense if dense else self.n_moe
    conv = dense_from_maxtext if dense else sparse_from_maxtext
    t2, t3 = self.types[("reduced", "m2", kd)], self.types[("reduced", "m3", kd)]
    wl = jax.shard_map(
        lambda w: [conv(ds_ep._take(w, l), self.cfg) for l in range(n)],
        mesh=self.mesh,
        in_specs=(self.wc_specs["dense_layers" if dense else "layers"],),
        out_specs=[t2] * n,
        check_vma=False,
    )(w)
    return [jax.tree.map(lambda a, s: jax.reshard(a, NamedSharding(self.mesh3, s)), x, t3) for x in wl]

  def _from_lineage(self, grads, dense):
    """Per-layer Lineage gradients on m3 (typed unreduced, checked) -> ds_ep's stacked gradient tree on m2 typed
    ds_ep.chunk_gacc_specs(explicit=True): each leaf resharded to m2 (moves nothing: 'attn' has size 1), then
    converted and concatenated in a shard_map, on device-local arrays (never jnp.stack an unreduced value)."""
    kd = "dense" if dense else "sparse"
    t2, t3 = self.types[("unreduced", "m2", kd)], self.types[("unreduced", "m3", kd)]
    if not dense:
      grads, t2, t3 = [_drop_bias(g) for g in grads], _drop_bias(t2), _drop_bias(t3)
    for l, g in enumerate(grads):
      jax.tree_util.tree_map_with_path(
          lambda p, x, s, l=l: _same_type(f"Lineage gradient of layer {l}{jax.tree_util.keystr(p)}", x, s), g, t3
      )
    g2 = [jax.tree.map(lambda a, s: jax.reshard(a, NamedSharding(self.mesh, s)), g, t2) for g in grads]

    def body(gs):
      trees = [dense_to_maxtext(g, self.cfg) if dense else sparse_to_maxtext(g, self.cfg, self.routed_bias) for g in gs]
      return jax.tree.map(lambda *ls: jnp.concatenate([a[None] for a in ls], axis=0), *trees)

    return jax.shard_map(
        body,
        mesh=self.mesh,
        in_specs=([t2] * len(grads),),
        out_specs=self.gacc_specs["dense_layers" if dense else "layers"],
        check_vma=False,
    )(g2)

  def _check_inputs(self, w_layers, w_dense, h):
    cfg, n_dev = self.cfg, self.scfg.fsdp * self.scfg.ep
    if h.dtype != jnp.bfloat16 or h.ndim != 3 or h.shape[2] != cfg.emb_dim:
      raise ValueError(f"h: want [B, S, {cfg.emb_dim}] bf16, got {h.shape} {h.dtype}")
    if h.shape[0] % (2 * n_dev):
      raise ValueError(f"h has {h.shape[0]} rows over {n_dev} devices: the batch split wants an even number a " "device")
    _same_type("h", h, ACT2)
    trees = [("w_layers", w_layers, self.wc_specs["layers"])]
    if self.n_dense:
      trees.append(("w_dense", w_dense, self.wc_specs["dense_layers"]))
    elif w_dense is not None:
      raise ValueError("w_dense given to a region without dense layers")
    for name, tree, specs in trees:
      if jax.tree.structure(tree) != jax.tree.structure(specs, is_leaf=ds_ep._is_spec):
        raise ValueError(
            f"{name}: tree {jax.tree.structure(tree)} is not ds_ep's "
            f"{jax.tree.structure(specs, is_leaf=ds_ep._is_spec)}"
        )
      jax.tree_util.tree_map_with_path(
          lambda p, x, s, name=name: _same_type(name + jax.tree_util.keystr(p), x, s), tree, specs
      )

  # ------------------------------------------------------------- the primal's and the two rules' bodies (nested jits)
  def _prim_impl(self, w_layers, w_dense, h):
    """The primal: the forward alone, the activation bank sized for no layer -> (y, aux)."""
    return self._forward(w_layers, w_dense, h, bank_layers=0)[0]

  def _fwd_impl(self, w_layers, w_dense, h):
    """The forward rule: the forward, the bank sized for every MoE layer -> ((y, aux), the backward's residuals)."""
    return self._forward(w_layers, w_dense, h, bank_layers=None)

  def _forward(self, w_layers, w_dense, h, bank_layers):
    self._check_inputs(w_layers, w_dense, h)
    lin, m3, kw = self.lin, self.mesh3, self.kw
    with no_sparsecore(not self.sparsecore):
      wl = self._to_lineage(w_layers, dense=False)
      wd = self._to_lineage(w_dense, dense=True) if self.n_dense else []
      x = jax.reshard(h, NamedSharding(m3, ACT3))
      with jax.sharding.use_abstract_mesh(m3.abstract_mesh):
        yarn, splash = kernels(self.cfg, m3, *h.shape[:2])
        dense_in = []
        for w in wd:
          dense_in.append(x)
          x = lin.dsv3_dense_layer.dsv3_dense_layer(x, w, yarn, splash, **{k: kw[k] for k in MLA_KWARGS})
        y3, aux, res = _chain_fwd(wl, x, yarn, splash, kw, bank_layers=bank_layers)
      y = jax.reshard(y3, NamedSharding(self.mesh, ACT2))
      aux = lin.dsv3_types.DSv3RouterAux(
          group_sizes=jax.reshard(aux.group_sizes, NamedSharding(self.mesh, P(None, FE, None))),
          selected_experts=jax.reshard(aux.selected_experts, NamedSharding(self.mesh, P(None, FE, None, None))),
          logits=None,
      )
    return (y, aux), (wl, wd, dense_in, res, yarn)

  def _bwd_impl(self, res, dy):
    lin, m3, kw = self.lin, self.mesh3, self.kw
    wl, wd, dense_in, chain_res, yarn = res
    with no_sparsecore(not self.sparsecore):
      dx = jax.reshard(dy, NamedSharding(m3, ACT3))
      with jax.sharding.use_abstract_mesh(m3.abstract_mesh):
        _, splash = kernels(self.cfg, m3, *dy.shape[:2])
        grads, dx = _chain_bwd(wl, chain_res, dx, yarn, splash, kw)
        dgrads = [None] * len(wd)
        mla_kw = {k: kw[k] for k in MLA_KWARGS}
        for l in reversed(range(len(wd))):
          _, vjp = jax.vjp(
              lambda w, x: lin.dsv3_dense_layer.dsv3_dense_layer(x, w, yarn, splash, **mla_kw), wd[l], dense_in[l]
          )
          dgrads[l], dx = vjp(dx)
      dh = jax.reshard(dx, NamedSharding(self.mesh, ACT2))
      dw_layers = self._from_lineage(grads, dense=False)
      dw_dense = self._from_lineage(dgrads, dense=True) if wd else None
    return dw_layers, dw_dense, dh


# The Regions `region` hands out: the REGION_CACHE last asked for, the least recently used first, and any other still
# alive (`_LIVE`, weak).  A Region holds its three nested jits' traces -- about 0.5 GiB of host memory once a
# 2-MoE-layer chunk's F and B programs have traced it (CPU, tracemalloc: evicting three freed 1.6 GiB) -- for as long as
# this cache or a live program holds it: an F program's trace does, and a jax.checkpoint'ed region's (the custom_vjp's
# call in their jaxprs holds its rules), a jax.vjp program's does not.  So a weak cache alone would drop the Region of a
# structure whose programs are all jax.vjp's (V=1's last chunk: its fused programs) between two of them, each then
# tracing Lineage anew; nor can the stage mesh, the natural owner, hold it weakly (its traces reference the mesh: such a
# tie never breaks).  A pipeline has at most one structure (a chunk structure on a stage mesh) a chunk, so 32 holds
# every structure of a pipeline of up to 32 chunks -- PP=16 x V=2 compiled in one process; at DeepSeek-V3's 58 MoE
# layers no power-of-two PP makes more than 32 structures -- each traced once.  Past 32 structures (PP >= 33, or a
# deeper model at V >= 3) the least recently used is evicted, and of the evicted only a structure whose programs are all
# jax.vjp's can come back as a new Region and trace again (a program holds the others': `_LIVE` hands them out again):
# compile time, not correctness.  The pipelines of a process share Regions while among the 32 last used or held by a
# program.
REGION_CACHE = 32
_REGIONS = collections.OrderedDict()
_LIVE = weakref.WeakValueDictionary()
_REGIONS_LOCK = threading.Lock()
_OFFLOAD = {}  # the SparseCore answer, a bool a stage mesh
_OFFLOAD_LOCK = threading.Lock()


def _sparsecore_offload(mesh: Mesh):
  """ds_ep.sparsecore_offload_available(mesh), asked once a mesh (it compiles a probe program)."""
  with _OFFLOAD_LOCK:
    if mesh not in _OFFLOAD:
      _OFFLOAD[mesh] = ds_ep.sparsecore_offload_available(mesh)
    return _OFFLOAD[mesh]


def region(
    scfg: ds_ep.StageConfig,
    ch: ds_ep.ChunkSpec,
    mesh: Mesh,
    capacity_factor: float = CAPACITY_FACTOR,
    sparsecore: bool | None = None,
):
  """The `Region` of chunk `ch`'s structure (its MoE / dense layer counts, routed bias) on stage mesh `mesh`, one
  per such structure: chunks alike share it, and with it its traces -- in this process, while among the REGION_CACHE
  structures last asked for or otherwise alive (`_REGIONS`, `_LIVE`)."""
  if sparsecore is None:
    sparsecore = _sparsecore_offload(mesh)
  key = (scfg, ch.n_moe, ch.dense, ch.routed_bias, mesh, float(capacity_factor), bool(sparsecore))
  with _REGIONS_LOCK:
    rg = _REGIONS.pop(key, None)
    if rg is None:
      rg = _LIVE.get(key)
    if rg is None:
      rg = _LIVE[key] = Region(scfg, mesh, ch.n_moe, ch.dense, ch.routed_bias, capacity_factor, sparsecore)
    _REGIONS[key] = rg  # the most recently used last
    while len(_REGIONS) > REGION_CACHE:
      _REGIONS.popitem(last=False)
    return rg
