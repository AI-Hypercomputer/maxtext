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

"""Pure-JAX TPU v7x AOT compiler, HLO memory analyzer, and scan-residual auditor."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import glob
import json
import math
import os
import re
import shutil
import time
from typing import Any

from absl import app
from absl import flags
import jax
from jax._src import xla_bridge
import jax.numpy as jnp
from jax.sharding import NamedSharding

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_train
from maxtext.experimental.flint40b import flint_types

BASE_LIBTPU_ARGS: str = "--xla_tpu_scoped_vmem_limit_kib=65536"

_SHAPE_RE = re.compile(r"\b([a-z0-9]+)\[([0-9,]+)\]")
_SCALAR_OR_TENSOR_SHAPE_RE = re.compile(r"\b([a-z0-9]+)\[([0-9,]*)\]")
_CALLEE_RE = re.compile(
    r"(?:calls|to_apply|async_computation)=%([A-Za-z0-9_.-]+)" r"|called_computations=\{%([A-Za-z0-9_.-]+)\}"
)
_CORE_IDS_RE = re.compile(r'"core_ids"\s*:\s*\[\s*"?(\d+)"?(?:\s*,\s*"?(\d+)"?)?\s*\]')
_DTYPE_ITEMSIZE: Mapping[str, int] = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "f8e4m3fn": 1,
    "f8e5m2": 1,
    "bf16": 2,
    "f16": 2,
    "s16": 2,
    "u16": 2,
    "f32": 4,
    "s32": 4,
    "u32": 4,
    "f64": 8,
    "s64": 8,
    "u64": 8,
    "c64": 8,
    "c128": 16,
}


def _shape_bytes(dtype_str: str, dims_str: str) -> int:
  """Computes byte size of an HLO tensor type `<dtype>[<dims>]`."""
  itemsize = _DTYPE_ITEMSIZE.get(dtype_str, 0)
  if itemsize == 0:
    return 0
  if not dims_str:
    return itemsize
  elems = math.prod(int(x) for x in dims_str.split(",") if x)
  return elems * itemsize


def _find_callees(line_str: str) -> list[str]:
  """Extracts HLO subcomputation names referenced on `line_str`."""
  return [a or b for a, b in _CALLEE_RE.findall(line_str)]


def _extract_sc_pin(line_str: str) -> str | None:
  """Parses SparseCore `core_ids` annotation on `line_str` if present."""
  m = _CORE_IDS_RE.search(line_str)
  if not m:
    return None
  first, second = m.group(1), m.group(2)
  if second is not None:
    return "unpinned"
  if first == "0":
    return "sc0"
  if first == "1":
    return "sc1"
  return "unpinned"


def parse_xla_compiler_options(libtpu_args: str) -> dict[str, str]:
  """Parses `--key=val` tokens into a PJRT `compiler_options` dict."""
  opts: dict[str, str] = {}
  for tok in libtpu_args.split():
    s = tok.strip()
    if not s:
      continue
    s = s.lstrip("-")
    if "=" in s:
      k, v = s.split("=", 1)
    else:
      k, v = s, "true"
    if k:
      opts[k] = v
  return opts


def resolve_libtpu_args(
    cfg: flint_config.FlintConfig,
    *,
    optimized_libtpu_args: bool,
    extra_libtpu_args: str,
) -> str:
  """Returns the `LIBTPU_INIT_ARGS` string for `cfg`.

  Order: `BASE_LIBTPU_ARGS`, then the tuned TPU v7x flags for the recipe shape
  (when `optimized_libtpu_args`), then user `extra_libtpu_args`.
  """
  tokens = BASE_LIBTPU_ARGS.split()
  if optimized_libtpu_args:
    tokens.extend(
        flint_config.optimized_v7x_libtpu_args(
            global_batch_size=cfg.train.global_batch_size,
            grad_accum_steps=cfg.train.grad_accum_steps,
            fsdp_size=cfg.sharding.fsdp_size,
            cp_size=cfg.sharding.cp_size,
        ).split()
    )
  tokens.extend(extra_libtpu_args.split())
  return " ".join(tokens)


def sync_compiler_and_dump_flags(
    libtpu_args: str,
    xla_dump_dir: str | None = None,
    *,
    hlo_module_re: str = ".*train_step.*",
) -> tuple[dict[str, str], int]:
  """Exports compiler and XLA dump flags before the TPU backend initializes.

  Sets `LIBTPU_INIT_ARGS` (and `XLA_FLAGS` when `xla_dump_dir` is set),
  registers PJRT plugins, and syncs matching `absl.flags`. Returns the PJRT
  `compiler_options` parsed from `libtpu_args` and the absl-synced flag count.
  Dump flags are never passed as PJRT compiler options.
  """
  os.environ["LIBTPU_INIT_ARGS"] = libtpu_args
  compiler_opts = parse_xla_compiler_options(libtpu_args)
  dump_opts: dict[str, str] = {}
  if xla_dump_dir is not None:
    xla_flags = " ".join(
        [
            f"--xla_dump_to={xla_dump_dir}",
            f"--xla_dump_hlo_module_re={hlo_module_re}",
            "--xla_dump_hlo_as_text=true",
            "--xla_flags_reset=true",
        ]
    )
    os.environ["XLA_FLAGS"] = xla_flags
    dump_opts = parse_xla_compiler_options(xla_flags)

  xla_bridge._discover_and_register_pjrt_plugins()  # pylint: disable=protected-access

  absl_synced = 0
  for k, v in {**compiler_opts, **dump_opts}.items():
    if k in flags.FLAGS:
      flags.FLAGS[k].parse(v)
      absl_synced += 1
  return compiler_opts, absl_synced


def _forbidden_gathered_weight_shape_keys(
    cfg: flint_config.FlintConfig,
) -> set[tuple[int, ...]]:
  """Builds sorted dimension multisets for unsharded gathered weights stacked over superblocks."""
  _, num_superblocks, _ = flint_config.flint_scan_layout(cfg.model)
  if not cfg.model.scan_layers or num_superblocks <= 0:
    return set()

  m = cfg.model
  s = cfg.sharding
  t = cfg.train
  d = m.hidden_size
  q_dim = m.num_attention_heads * m.head_dim
  kv_dim = m.num_key_value_heads * m.head_dim
  shared_inter = m.shared_expert_intermediate_size
  dense_inter = m.intermediate_size
  moe_inter = m.moe_intermediate_size
  num_experts = m.num_experts
  e_loc = num_experts // s.ep_size

  micro_gbs = t.global_batch_size // t.grad_accum_steps
  b_local = micro_gbs // s.dp_ways
  s_local = t.seq_len // s.cp_size
  tokens_per_tc = b_local * s_local

  stack_counts = {
      num_superblocks,
      num_superblocks * max(1, m.moe_cycle - 1),
      num_superblocks * m.moe_cycle,
  }
  forbidden: set[tuple[int, ...]] = set()

  if s.num_devices > 1:
    dense_other_dims: list[int] = [
        q_dim,
        kv_dim,
        q_dim + 2 * kv_dim,
        shared_inter,
        2 * shared_inter,
        dense_inter,
        2 * dense_inter,
    ]
    if not cfg.kernels.replicate_router_gate:
      dense_other_dims.append(num_experts)
    for n_stack in stack_counts:
      for other in dense_other_dims:
        if other == tokens_per_tc:
          continue
        forbidden.add(tuple(sorted((n_stack, d, other))))

  if s.fsdp_size * s.cp_size > 1:
    for n_stack in stack_counts:
      forbidden.add(tuple(sorted((n_stack, e_loc, d, 2 * moe_inter))))
      forbidden.add(tuple(sorted((n_stack, e_loc, moe_inter, d))))
      forbidden.add(tuple(sorted((n_stack, e_loc * d, 2 * moe_inter))))
      forbidden.add(tuple(sorted((n_stack, e_loc * moe_inter, d))))
      if d % 8 == 0:
        forbidden.add(tuple(sorted((n_stack, e_loc, d // 8, 8, 2 * moe_inter))))
        forbidden.add(tuple(sorted((n_stack, e_loc, d // 8, 8, moe_inter))))
      if moe_inter % 8 == 0:
        forbidden.add(tuple(sorted((n_stack, e_loc, moe_inter // 8, 8, d))))

  return forbidden


def audit_flint_hlo(
    xla_dir: str,
    cfg: flint_config.FlintConfig,
) -> dict[str, Any]:
  """Audits optimized `train_step` HLO for Splash, ragged-dot, VMEM spills, and scan residuals."""
  opt_cands = sorted(
      glob.glob(
          os.path.join(
              xla_dir,
              "*train_step*after_optimizations_before_buffer_assignment*.txt",
          )
      )
  )
  if not opt_cands:
    opt_cands = sorted(glob.glob(os.path.join(xla_dir, "*train_step*after_optimizations*.txt")))
  if not opt_cands:
    return {
        "hlo_found": False,
        "hlo_file": None,
        "has_splash": False,
        "has_dkv_head_stitch": False,
        "has_ragged_dot": False,
        "has_gmm_v2": False,
        "has_sc_gather_reduce": False,
        "has_sc_counting_sort": False,
        "has_router_topk": False,
        "has_qk_prologue": False,
        "has_lb_kv_reorder": False,
        "copy_gib_step": 0.0,
        "scatter_ops_step": 0,
        "sc_collectives_by_pin_step": {"sc0": 0, "sc1": 0, "unpinned": 0},
        "sc_all_reduces_step": 0,
        "fp8_all_gathers_step": 0,
        "gathered_weight_scan_residuals": [],
        "vmem_spill_lines": 0,
        "while_bodies": {},
    }

  hlo_path = opt_cands[0]
  comp_blocks: dict[str, list[str]] = {}
  cur_comp: str | None = None
  cur_lines: list[str] = []
  has_splash = False
  has_dkv_head_stitch = False
  has_ragged_dot = False
  has_gmm_v2 = False
  has_sc_gather_reduce = False
  has_sc_counting_sort = False
  has_router_topk = False
  has_qk_prologue = False
  has_lb_kv_reorder = False

  with open(hlo_path, "r", encoding="utf-8", errors="replace") as f:
    for raw_line in f:
      line = raw_line.rstrip("\n")
      lower = line.lower()
      if not has_splash and "splash" in lower:
        has_splash = True
      if not has_dkv_head_stitch and ("_head_stitch" in lower and "dkv_head_stitch: false" not in lower):
        has_dkv_head_stitch = True
      if not has_ragged_dot and ("ragged-dot" in lower or "ragged_dot" in lower):
        has_ragged_dot = True
      if not has_gmm_v2 and ("gmm_v2" in lower or "tgmm_v2" in lower):
        has_gmm_v2 = True
      if not has_sc_gather_reduce and "dense_gather_reduce" in lower:
        has_sc_gather_reduce = True
      if not has_sc_counting_sort and "sc_counting_sort" in lower:
        has_sc_counting_sort = True
      if not has_router_topk and "router_topk" in lower:
        has_router_topk = True
      if not has_qk_prologue and "pallas_qk_prologue" in lower:
        has_qk_prologue = True
      if not has_lb_kv_reorder and ("lb_kv_vmem_copy" in lower or "lb_kv_hbm_dma" in lower):
        has_lb_kv_reorder = True

      if line.startswith("ENTRY ") or (
          not line.startswith(" ") and not line.startswith("HloModule") and "{" in line and "%" in line
      ):
        if cur_comp is not None:
          comp_blocks[cur_comp] = cur_lines
        idx_pct = line.find("%")
        idx_par = line.find("(", idx_pct)
        cur_comp = line[idx_pct + 1 : idx_par].strip() if idx_par > idx_pct else "ENTRY"
        cur_lines = [line]
      elif line == "}" and cur_comp is not None:
        cur_lines.append(line)
        comp_blocks[cur_comp] = cur_lines
        cur_comp = None
        cur_lines = []
      elif cur_comp is not None:
        cur_lines.append(line)
  if cur_comp is not None and cur_lines:
    comp_blocks[cur_comp] = cur_lines

  need_pre_scan = not (
      has_dkv_head_stitch
      and has_ragged_dot
      and has_gmm_v2
      and has_sc_gather_reduce
      and has_sc_counting_sort
      and has_router_topk
      and has_qk_prologue
      and has_lb_kv_reorder
  )
  if need_pre_scan:
    pre_cands = sorted(glob.glob(os.path.join(xla_dir, "*train_step*before_optimizations*.txt")))
    for pre_path in pre_cands:
      with open(pre_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
          lower = raw_line.lower()
          if not has_dkv_head_stitch and "_head_stitch" in lower:
            has_dkv_head_stitch = True
          if not has_ragged_dot and ("ragged-dot" in lower or "ragged_dot" in lower):
            has_ragged_dot = True
          if not has_gmm_v2 and ("gmm_v2" in lower or "tgmm_v2" in lower):
            has_gmm_v2 = True
          if not has_sc_gather_reduce and "dense_gather_reduce" in lower:
            has_sc_gather_reduce = True
          if not has_sc_counting_sort and "sc_counting_sort" in lower:
            has_sc_counting_sort = True
          if not has_router_topk and "router_topk" in lower:
            has_router_topk = True
          if not has_qk_prologue and "pallas_qk_prologue" in lower:
            has_qk_prologue = True
          if not has_lb_kv_reorder and "lb_kv_vmem_copy" in lower:
            has_lb_kv_reorder = True
          if (
              has_dkv_head_stitch
              and has_ragged_dot
              and has_gmm_v2
              and has_sc_gather_reduce
              and has_sc_counting_sort
              and has_router_topk
              and has_qk_prologue
              and has_lb_kv_reorder
          ):
            break

  _, default_trip, _ = flint_config.flint_scan_layout(cfg.model)
  entry_name = next(
      (k for k, v in comp_blocks.items() if v and v[0].startswith("ENTRY ")),
      None,
  )
  comp_mult: dict[str, int] = collections.defaultdict(lambda: 1)
  while_bodies: dict[str, int] = {}
  boundary_lines: list[str] = []

  if entry_name is not None:
    for line in comp_blocks[entry_name]:
      if " while(" in line:
        boundary_lines.append(line)
        mb = re.search(r"body=%([A-Za-z0-9_.-]+)", line)
        mc = re.search(r"condition=%([A-Za-z0-9_.-]+)", line)
        if mb:
          bname = mb.group(1)
          trip = max(1, default_trip)
          if mc and mc.group(1) in comp_blocks:
            cond_lines = list(comp_blocks[mc.group(1)])
            for cline in comp_blocks[mc.group(1)]:
              for fcall in re.findall(r"calls=%([A-Za-z0-9_.-]+)", cline):
                if fcall in comp_blocks:
                  cond_lines.extend(comp_blocks[fcall])
            for cline in cond_lines:
              mconst = re.search(r"s32\[\]\s+constant\((\d+)\)", cline)
              if mconst:
                val = int(mconst.group(1))
                if 1 <= val <= 500:
                  trip = val
          comp_mult[bname] = trip
          while_bodies[bname] = trip

  for _ in range(5):
    changed = False
    for cname, mult in list(comp_mult.items()):
      if mult > 1 and cname in comp_blocks:
        for line in comp_blocks[cname]:
          if "fused_computation" in line:
            continue
          for mcall in _find_callees(line):
            if comp_mult[mcall] < mult:
              comp_mult[mcall] = mult
              changed = True
    if not changed:
      break

  comp_sc_pin: dict[str, str] = {}
  wrapper_callees: set[str] = set()
  ar_wrapper_callees: set[str] = set()
  for cname, lines in comp_blocks.items():
    for line in lines:
      mcalls = _find_callees(line)
      if "all-gather-start(" in line or "reduce-scatter-start(" in line:
        wrapper_callees.update(mcalls)
      if "all-reduce-start(" in line:
        ar_wrapper_callees.update(mcalls)
      pin = _extract_sc_pin(line)
      if pin is not None:
        if mcalls:
          for mcall in mcalls:
            comp_sc_pin[mcall] = pin
        else:
          comp_sc_pin[cname] = pin

  for _ in range(4):
    for cname, lines in comp_blocks.items():
      if cname in comp_sc_pin:
        pin = comp_sc_pin[cname]
        for line in lines:
          for mcall in _find_callees(line):
            comp_sc_pin.setdefault(mcall, pin)

  gib = float(1024**3)
  copy_gib = 0.0
  scatter_ops_step = 0
  sc_all_reduces_step = 0
  fp8_all_gathers_step = 0
  sc_collectives_by_pin_step: dict[str, int] = {
      "sc0": 0,
      "sc1": 0,
      "unpinned": 0,
  }
  for cname, lines in comp_blocks.items():
    if cname.startswith("fused_computation"):
      continue
    if lines and re.search(r"->\s*[a-z0-9]+\[\]\s*\{", lines[0]):
      continue
    mult = comp_mult[cname]
    default_pin = comp_sc_pin.get(cname, "unpinned")
    for line in lines:
      s = line.strip()
      if "=" not in s:
        continue
      _, rhs = s.split("=", 1)
      rhs = rhs.strip()
      if rhs.startswith("scatter(") or " scatter(" in rhs:
        scatter_ops_step += mult
      elif "fusion(" in rhs:
        for fcallee in _find_callees(s):
          for fline in comp_blocks.get(fcallee, ()):
            fs = fline.strip()
            if "=" in fs:
              _, frhs = fs.split("=", 1)
              frhs = frhs.strip()
              if frhs.startswith("scatter(") or " scatter(" in frhs:
                scatter_ops_step += mult
      if "copy(" in rhs and not rhs.startswith("copy-start") and not rhs.startswith("copy-done"):
        m_copy = _SCALAR_OR_TENSOR_SHAPE_RE.search(rhs)
        if m_copy:
          nbytes = _shape_bytes(m_copy.group(1), m_copy.group(2))
          copy_gib += (nbytes * mult) / gib
      if (
          cname not in ar_wrapper_callees
          and "S(7)" in s
          and (
              rhs.startswith("all-reduce(")
              or " all-reduce(" in rhs
              or rhs.startswith("all-reduce-start(")
              or " all-reduce-start(" in rhs
          )
      ):
        sc_all_reduces_step += mult
      if cname in wrapper_callees:
        continue
      for coll in ("all-gather", "reduce-scatter"):
        if (
            rhs.startswith(f"{coll}(")
            or f" {coll}(" in rhs
            or rhs.startswith(f"{coll}-start(")
            or f" {coll}-start(" in rhs
        ):
          line_pin = _extract_sc_pin(s) or default_pin
          bucket = line_pin if line_pin in ("sc0", "sc1") else "unpinned"
          sc_collectives_by_pin_step[bucket] += mult
          if coll == "all-gather" and "f8e4m3fn[" in s:
            fp8_all_gathers_step += mult

  for bname in while_bodies:
    for bline in comp_blocks.get(bname, ()):
      stripped = bline.strip()
      if "parameter(" in stripped or stripped.startswith("ROOT "):
        boundary_lines.append(bline)

  forbidden_keys = _forbidden_gathered_weight_shape_keys(cfg)
  leaked_shapes: dict[str, None] = collections.OrderedDict()
  if forbidden_keys:
    for bline in boundary_lines:
      for m in _SHAPE_RE.finditer(bline):
        dtype_str, dims_str = m.group(1), m.group(2)
        dims = tuple(int(x) for x in dims_str.split(",") if x)
        if len(dims) >= 3 and tuple(sorted(dims)) in forbidden_keys:
          leaked_shapes[f"{dtype_str}[{dims_str}]"] = None

  vmem_spill_lines = 0
  codegen_cands = sorted(glob.glob(os.path.join(xla_dir, "*train_step*after_codegen*.txt")))
  if codegen_cands:
    with open(codegen_cands[0], "r", encoding="utf-8", errors="replace") as f:
      for raw_line in f:
        if raw_line.startswith(" ") and "spill" in raw_line.lower():
          vmem_spill_lines += 1

  return {
      "hlo_found": True,
      "hlo_file": os.path.basename(hlo_path),
      "has_splash": has_splash,
      "has_dkv_head_stitch": has_dkv_head_stitch,
      "has_ragged_dot": has_ragged_dot,
      "has_gmm_v2": has_gmm_v2,
      "has_sc_gather_reduce": has_sc_gather_reduce,
      "has_sc_counting_sort": has_sc_counting_sort,
      "has_router_topk": has_router_topk,
      "has_qk_prologue": has_qk_prologue,
      "has_lb_kv_reorder": has_lb_kv_reorder,
      "copy_gib_step": round(copy_gib, 4),
      "scatter_ops_step": scatter_ops_step,
      "sc_collectives_by_pin_step": sc_collectives_by_pin_step,
      "sc_all_reduces_step": sc_all_reduces_step,
      "fp8_all_gathers_step": fp8_all_gathers_step,
      "gathered_weight_scan_residuals": list(leaked_shapes.keys()),
      "vmem_spill_lines": vmem_spill_lines,
      "while_bodies": while_bodies,
  }


def make_abstract_sharded_inputs(
    cfg: flint_config.FlintConfig,
    mesh: jax.sharding.Mesh,
) -> tuple[
    flint_train.FlintTrainState[jax.ShapeDtypeStruct],
    flint_types.FlintBatch,
]:
  """Builds abstract sharded `(FlintTrainState, FlintBatch)` for AOT lowering."""
  abs_state = jax.tree.map(
      lambda s, sh: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=sh),
      flint_train.abstract_train_state(cfg, mesh),
      flint_train.state_shardings(mesh, cfg),
  )
  batch_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
  tok_spec = jax.ShapeDtypeStruct(
      (cfg.train.global_batch_size, cfg.train.seq_len),
      jnp.int32,
      sharding=batch_sharding,
  )
  batch_cls: Any = flint_types.FlintBatch
  abs_batch: flint_types.FlintBatch = batch_cls(
      tokens=tok_spec,
      targets=tok_spec,
      positions=tok_spec,
      segment_ids=tok_spec,
      loss_mask=tok_spec,
  )
  return abs_state, abs_batch


def compile_single_spec(
    exp_name: str,
    out_dir: str,
    *,
    topology: str = "tpu7x:4x4x4",
    config_overrides: Mapping[str, Any] | None = None,
    optimized_libtpu_args: bool = True,
    extra_libtpu_args: str = "",
    max_hbm_gib: float = 94.0,
    keep_hlo_dumps: bool = False,
) -> dict[str, Any]:
  """Lowers, AOT-compiles, and audits a single Flint recipe on `topology`.

  `config_overrides=None` selects the default optimized 256K recipe
  (`flint_config.default_recipe_overrides()`); `{}` selects the dataclass
  defaults.
  """
  if not out_dir.strip():
    raise ValueError("out_dir must be a non-empty directory path.")
  clean_out_dir = os.path.abspath(out_dir)
  xla_dir = os.path.join(clean_out_dir, "xla")
  if os.path.isdir(xla_dir):
    shutil.rmtree(xla_dir)
  os.makedirs(xla_dir, exist_ok=True)

  jax.config.update("jax_enable_compilation_cache", False)
  overrides = flint_config.default_recipe_overrides() if config_overrides is None else dict(config_overrides)
  cfg = flint_config.flint_40b_config(**overrides)
  libtpu_args = resolve_libtpu_args(
      cfg,
      optimized_libtpu_args=optimized_libtpu_args,
      extra_libtpu_args=extra_libtpu_args,
  )
  pjrt_compiler_opts, absl_synced = sync_compiler_and_dump_flags(
      libtpu_args,
      xla_dump_dir=xla_dir,
      hlo_module_re=".*train_step.*",
  )

  t0 = time.perf_counter()
  mesh = flint_sharding.create_mesh(cfg.sharding, aot_topology=topology)
  abs_state, abs_batch = make_abstract_sharded_inputs(cfg, mesh)
  train_step_fn: Any = flint_train.make_train_step(cfg, mesh, donate_state=True)

  lowered = train_step_fn.lower(abs_state, abs_batch)
  t_lower = round(time.perf_counter() - t0, 2)
  compiled = lowered.compile(compiler_options=pjrt_compiler_opts)
  t_compile = round(time.perf_counter() - t0, 2)

  mem: Any = compiled.memory_analysis()
  if mem is None:
    raise RuntimeError("compiled.memory_analysis() returned None.")
  cost: Any = compiled.cost_analysis()

  gib = float(1024**3)
  total_hbm_bytes = (
      int(mem.generated_code_size_in_bytes)
      + int(mem.argument_size_in_bytes)
      + max(0, int(mem.output_size_in_bytes) - int(mem.alias_size_in_bytes))
      + int(mem.temp_size_in_bytes)
  )
  total_aot_hbm_gib = total_hbm_bytes / gib

  param_bytes_per_dev, adam_total_bytes_per_dev = flint_sharding.estimate_persistent_state_bytes_per_device(cfg)
  total_params, active_params, _, _ = flint_config.count_flint_params(cfg.model)
  total_tflops_per_tc, weight_tflops_per_tc, attn_tflops_per_tc = flint_config.calculate_tflops_per_device(cfg)

  comp_env_files = sorted(glob.glob(os.path.join(xla_dir, "*.tpu_comp_env.txt")))
  comp_env_txt = ""
  if comp_env_files:
    with open(comp_env_files[0], "r", encoding="utf-8", errors="replace") as f:
      comp_env_txt = f.read()
  tpu_comp_env_verified = "xla_tpu_scoped_vmem_limit_kib: 65536" in comp_env_txt

  hlo_audit = audit_flint_hlo(xla_dir, cfg)
  if not keep_hlo_dumps and os.path.isdir(xla_dir):
    shutil.rmtree(xla_dir)

  cost_info: dict[str, Any] = {}
  if cost:
    c0 = cost[0] if isinstance(cost, list) else cost
    if isinstance(c0, dict):
      cost_info = {
          "flops": c0.get("flops"),
          "bytes_accessed": c0.get("bytes accessed"),
      }

  summary: dict[str, Any] = {
      "exp_name": exp_name,
      "topology": topology,
      "seq_len": cfg.train.seq_len,
      "global_batch_size": cfg.train.global_batch_size,
      "grad_accum_steps": cfg.train.grad_accum_steps,
      "fsdp_size": cfg.sharding.fsdp_size,
      "cp_size": cfg.sharding.cp_size,
      "ep_size": cfg.sharding.ep_size,
      "param_dtype": cfg.train.param_dtype,
      "compute_dtype": cfg.train.compute_dtype,
      "use_fp8": cfg.kernels.use_fp8,
      "total_params": total_params,
      "active_params": active_params,
      "lower_sec": t_lower,
      "compile_sec": t_compile,
      "code_gib": round(int(mem.generated_code_size_in_bytes) / gib, 3),
      "arg_gib": round(int(mem.argument_size_in_bytes) / gib, 3),
      "output_gib": round(int(mem.output_size_in_bytes) / gib, 3),
      "alias_gib": round(int(mem.alias_size_in_bytes) / gib, 3),
      "temp_gib": round(int(mem.temp_size_in_bytes) / gib, 3),
      "host_temp_gib": round(int(mem.host_temp_size_in_bytes) / gib, 3),
      "total_aot_hbm_gib": round(total_aot_hbm_gib, 3),
      "max_hbm_gib": max_hbm_gib,
      "passes_hbm_gate": total_aot_hbm_gib <= max_hbm_gib,
      "estimated_param_gib_per_tc": round(param_bytes_per_dev / gib, 3),
      "estimated_persistent_state_gib_per_tc": round(adam_total_bytes_per_dev / gib, 3),
      "total_tflops_per_tc": round(total_tflops_per_tc, 3),
      "weight_tflops_per_tc": round(weight_tflops_per_tc, 3),
      "attn_tflops_per_tc": round(attn_tflops_per_tc, 3),
      "tpu_comp_env_verified": tpu_comp_env_verified,
      "optimized_libtpu_args": optimized_libtpu_args,
      "libtpu_init_args": libtpu_args,
      "absl_flags_synced_count": absl_synced,
      "pjrt_compiler_options_count": len(pjrt_compiler_opts),
      **cost_info,
      **hlo_audit,
  }

  summary_path = os.path.join(clean_out_dir, "aot_summary.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  has_moe_layers = any(flint_config.is_moe_layer(i, cfg.model) for i in range(cfg.model.num_hidden_layers))
  if not summary["hlo_found"]:
    raise AssertionError(f"[{exp_name}] Optimized train_step HLO dump was not generated.")
  if cfg.kernels.use_splash and not summary["has_splash"]:
    raise AssertionError(f"[{exp_name}] Expected Tokamax Splash kernel in HLO, but" " has_splash=False.")
  if has_moe_layers:
    if cfg.kernels.moe_impl == "pallas_gmm_v2":
      if not summary["has_gmm_v2"]:
        raise AssertionError(f"[{exp_name}] Expected Pallas gmm_v2/tgmm_v2 in HLO, but" " has_gmm_v2=False.")
    elif not summary["has_ragged_dot"]:
      raise AssertionError(f"[{exp_name}] Expected ragged-dot in HLO, but has_ragged_dot=False.")
  if summary["gathered_weight_scan_residuals"]:
    raise AssertionError(
        f"[{exp_name}] Gathered weights leaked into scan residuals: " f"{summary['gathered_weight_scan_residuals']}"
    )
  if total_aot_hbm_gib > max_hbm_gib:
    raise AssertionError(
        f"[{exp_name}] total_aot_hbm_gib={total_aot_hbm_gib:.3f} exceeds " f"max_hbm_gib={max_hbm_gib:.3f} GiB/TC."
    )
  return summary


def _load_batch_specs(batch_specs_arg: str) -> list[dict[str, Any]]:
  """Loads batch specs from either an inline JSON array or a `.json` file path."""
  raw = batch_specs_arg.strip()
  if raw.startswith("["):
    parsed = json.loads(raw)
  else:
    with open(raw, "r", encoding="utf-8") as f:
      parsed = json.load(f)
  if not isinstance(parsed, list) or not parsed:
    raise ValueError("--batch_specs_json must decode to a non-empty JSON list.")
  return parsed


def run_batch_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    default_topology: str = "tpu7x:4x4x4",
    default_extra_libtpu_args: str = "",
    default_optimized_libtpu_args: bool = True,
    default_max_hbm_gib: float = 94.0,
    keep_hlo_dumps: bool = False,
) -> list[dict[str, Any]]:
  """Compiles multiple recipe specs sequentially in-process.

  A missing or `None` `config_overrides` selects the default 256K recipe.
  """
  summaries: list[dict[str, Any]] = []
  for spec in specs:
    exp_name = str(spec["exp_name"])
    out_dir = str(spec["out_dir"])
    overrides = spec.get("config_overrides")
    if overrides is not None and not isinstance(overrides, Mapping):
      raise ValueError(f"config_overrides for {exp_name!r} must be a dict, got" f" {overrides!r}.")
    topology = str(spec.get("topology", default_topology))
    extra_args = str(spec.get("extra_libtpu_args", default_extra_libtpu_args))
    optimized = bool(spec.get("optimized_libtpu_args", default_optimized_libtpu_args))
    max_hbm = float(spec.get("max_hbm_gib", default_max_hbm_gib))
    keep_dumps = bool(spec.get("keep_hlo_dumps", keep_hlo_dumps))
    summary = compile_single_spec(
        exp_name=exp_name,
        out_dir=out_dir,
        topology=topology,
        config_overrides=overrides,
        optimized_libtpu_args=optimized,
        extra_libtpu_args=extra_args,
        max_hbm_gib=max_hbm,
        keep_hlo_dumps=keep_dumps,
    )
    print(json.dumps(summary, indent=2), flush=True)
    summaries.append(summary)
  return summaries


def _define_cli_flags() -> dict[str, flags.FlagHolder[Any]]:
  """Registers CLI flags only when `aot_compile` is executed as a binary."""
  return {
      "exp_name": flags.DEFINE_string(
          "exp_name",
          flint_config.DEFAULT_RECIPE_NAME,
          "Experiment/recipe identifier.",
      ),
      "out_dir": flags.DEFINE_string(
          "out_dir",
          f"/tmp/flint40b/{flint_config.DEFAULT_RECIPE_NAME}_aot",
          "Directory for `aot_summary.json` and temporary XLA dumps.",
      ),
      "topology": flags.DEFINE_string("topology", "tpu7x:4x4x4", "Offline TPU topology (e.g. 'tpu7x:4x4x4')."),
      "config_overrides_json": flags.DEFINE_string(
          "config_overrides_json",
          "",
          "JSON dict of `flint_40b_config` overrides. Empty (default) selects"
          " the optimized 256K recipe"
          " (`flint_config.default_recipe_overrides()`); `{}` selects the"
          " unoptimized dataclass defaults.",
      ),
      "extra_libtpu_args": flags.DEFINE_string(
          "extra_libtpu_args",
          "",
          "Optional extra space-separated `--xla_tpu_...` compiler flags.",
      ),
      "optimized_libtpu_args": flags.DEFINE_boolean(
          "optimized_libtpu_args",
          True,
          "Append the tuned TPU v7x compiler flags" " (`flint_config.optimized_v7x_libtpu_args`) for the recipe shape.",
      ),
      "max_hbm_gib": flags.DEFINE_float(
          "max_hbm_gib",
          94.0,
          "Maximum allowed AOT HBM per TensorCore in GiB.",
      ),
      "batch_specs_json": flags.DEFINE_string(
          "batch_specs_json",
          "",
          "Optional JSON list (or path to JSON file) of batch AOT specs.",
      ),
      "keep_hlo_dumps": flags.DEFINE_boolean(
          "keep_hlo_dumps",
          False,
          "Keep raw dumped HLO `.txt` files under `<out_dir>/xla` after audit.",
      ),
  }


def main() -> None:
  """CLI entrypoint. With no flags, AOT-compiles the default optimized 256K recipe."""
  cli = _define_cli_flags()

  def _run(argv: Sequence[str]) -> None:
    if len(argv) > 1:
      raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
    batch_arg = str(cli["batch_specs_json"].value).strip()
    if batch_arg:
      specs = _load_batch_specs(batch_arg)
      run_batch_specs(
          specs,
          default_topology=str(cli["topology"].value),
          default_extra_libtpu_args=str(cli["extra_libtpu_args"].value),
          default_optimized_libtpu_args=bool(cli["optimized_libtpu_args"].value),
          default_max_hbm_gib=float(cli["max_hbm_gib"].value),
          keep_hlo_dumps=bool(cli["keep_hlo_dumps"].value),
      )
      return

    exp_name = str(cli["exp_name"].value).strip()
    out_dir = str(cli["out_dir"].value).strip()
    if not exp_name or not out_dir:
      raise app.UsageError("--exp_name and --out_dir must be non-empty.")
    raw = str(cli["config_overrides_json"].value).strip()
    overrides = json.loads(raw) if raw else None
    if overrides is not None and not isinstance(overrides, Mapping):
      raise app.UsageError("--config_overrides_json must decode to a JSON dict.")
    summary = compile_single_spec(
        exp_name=exp_name,
        out_dir=out_dir,
        topology=str(cli["topology"].value),
        config_overrides=overrides,
        optimized_libtpu_args=bool(cli["optimized_libtpu_args"].value),
        extra_libtpu_args=str(cli["extra_libtpu_args"].value),
        max_hbm_gib=float(cli["max_hbm_gib"].value),
        keep_hlo_dumps=bool(cli["keep_hlo_dumps"].value),
    )
    print(json.dumps(summary, indent=2), flush=True)

  app.run(_run)


if __name__ == "__main__":
  main()
