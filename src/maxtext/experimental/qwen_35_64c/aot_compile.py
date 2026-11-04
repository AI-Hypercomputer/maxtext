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

"""Pure-JAX TPU v7x AOT compiler and Dual-SparseCore HLO auditor."""

from __future__ import annotations

import argparse
import collections
from collections.abc import Sequence
import glob
import json
import os
import re
import time
from typing import Any

# TPU v7x compiler flags (single-space separated).
BASE_LIBTPU_ARGS = " ".join(
    [
        "--xla_tpu_use_tc_device_shape_on_sc=true",
        "--xla_sc_disable_megacore_partitioning=true",
        "--xla_tpu_enable_offloading_gather_to_sparsecore=false",
        "--xla_tpu_num_sparse_cores_for_gather_offloading=1",
        "--xla_tpu_enable_offloading_scatter_to_sparsecore=false",
        "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true",
        "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true",
        "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true",
        "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=2000000",
        "--xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=2000000",
        "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=2000000",
        "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true",
        "--xla_tpu_enable_all_experimental_scheduler_features=true",
        "--xla_tpu_nd_short_transfer_max_chunks=2048",
        "--xla_tpu_use_single_sparse_core_for_all_gather_offload=false",
        "--xla_tpu_enable_concurrent_sparse_core_offloading=true",
        "--xla_tpu_aggressive_opt_barrier_removal=false",
        "--xla_tpu_scoped_vmem_limit_kib=65472",
        "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false",
        "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
        "--xla_tpu_dvfs_p_state=7",
        "--xla_tpu_enable_offloading_copy_to_sparsecore=false",
        "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
        "--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=6",
        "--xla_tpu_pcie_bandwidth_multiplier=0.05",
        "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true",
        "--xla_tpu_enable_sparse_core_reduce_scatter_padding=true",
        "--xla_tpu_scheduler_percent_shared_memory_limit=112",
        "--xla_jf_rematerialization_percent_shared_memory_limit=90",
        "--xla_tpu_autofdo=false",
    ]
)


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


_DTYPE_BYTES = {
    "pred": 1,
    "s4": 0.5,
    "u4": 0.5,
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
}
_SHAPE_RE = re.compile(r"\b([a-z0-9]+)\[([0-9,]*)\]")


def _shape_bytes(dtype: str, dims_str: str) -> int:
  b = _DTYPE_BYTES.get(dtype, 0)
  if not b:
    return 0
  if not dims_str:
    return int(b)
  elems = 1
  for d in dims_str.split(","):
    if d:
      elems *= int(d)
  return int(elems * b)


def audit_hlo_dual_sc(xla_dir: str, default_trip: int = 23) -> dict:
  """Streams HLO line-by-line to audit SC0/SC1/unpinned collectives, copies, and kernels."""
  cands = sorted(
      glob.glob(
          os.path.join(
              xla_dir,
              "*jit_train_step*after_optimizations_before_buffer_assignment.txt",
          )
      )
  )
  if not cands:
    return {"hlo_found": False}
  hlo_path = cands[0]

  comp_blocks: dict[str, list[str]] = {}
  cur_comp = None
  cur_lines: list[str] = []
  has_lossy_const = False

  with open(hlo_path, "r", errors="replace") as f:
    for raw_line in f:
      line = raw_line.rstrip("\n")
      if "constant({...})" in line:
        has_lossy_const = True
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

  comp_mult: dict[str, int] = collections.defaultdict(lambda: 1)
  entry_name = next((k for k, v in comp_blocks.items() if v[0].startswith("ENTRY ")), None)
  while_bodies: dict[str, int] = {}
  nested_whiles: list[str] = []

  if entry_name:
    for line in comp_blocks[entry_name]:
      if " while(" in line:
        mb = re.search(r"body=%([A-Za-z0-9_.-]+)", line)
        mc = re.search(r"condition=%([A-Za-z0-9_.-]+)", line)
        if mb:
          bname = mb.group(1)
          trip = default_trip
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
                if 1 <= val <= 200:
                  trip = val
          comp_mult[bname] = trip
          while_bodies[bname] = trip

  _callee_re = re.compile(
      r"(?:calls|to_apply|async_computation)=%([A-Za-z0-9_.-]+)" r"|called_computations=\{%([A-Za-z0-9_.-]+)\}"
  )

  def _find_callees(line_str: str) -> list[str]:
    return [a or b for a, b in _callee_re.findall(line_str)]

  # Propagate loop trip counts & detect any nested while inside outer scan
  for _ in range(5):
    changed = False
    for cname, mult in list(comp_mult.items()):
      if mult > 1 and cname in comp_blocks:
        for line in comp_blocks[cname]:
          if "fused_computation" in line:
            continue
          if " while(" in line and cname in while_bodies and cname not in nested_whiles:
            nested_whiles.append(cname)
          for mcall in _find_callees(line):
            if comp_mult[mcall] < mult:
              comp_mult[mcall] = mult
              changed = True
    if not changed:
      break

  # Map async computations to SparseCore core_ids annotation if present
  comp_sc_pin: dict[str, str] = {}
  for cname, lines in comp_blocks.items():
    for line in lines:
      mcalls = _find_callees(line)
      if "core_ids" in line:
        pin = "other"
        if '"0","1"' in line or "0,1" in line or '"0", "1"' in line:
          pin = "sc01_both"
        elif '"0"' in line or "[0]" in line:
          pin = "sc0"
        elif '"1"' in line or "[1]" in line:
          pin = "sc1"
        if mcalls:
          for mcall in mcalls:
            comp_sc_pin[mcall] = pin
        else:
          comp_sc_pin[cname] = pin
      elif 'async_execution_thread="sparsecore"' in line and mcalls:
        for mcall in mcalls:
          comp_sc_pin.setdefault(mcall, "sc_auto")

  for _ in range(4):
    for cname, lines in comp_blocks.items():
      if cname in comp_sc_pin:
        pin = comp_sc_pin[cname]
        for line in lines:
          for mcall in _find_callees(line):
            comp_sc_pin.setdefault(mcall, pin)

  gib = 1024**3
  copy_gib = 0.0
  copy_cnt = 0
  coll_by_pin_cnt: dict[str, int] = collections.defaultdict(int)
  coll_by_pin_gib: dict[str, float] = collections.defaultdict(float)
  kernels_cnt: dict[str, int] = collections.defaultdict(int)

  for cname, lines in comp_blocks.items():
    if cname.startswith("fused_computation"):
      continue
    if len(lines) <= 8 and any("ROOT " in l and "[]" in l for l in lines):
      continue
    mult = comp_mult[cname]
    default_pin = comp_sc_pin.get(cname, "tc_or_unpinned")

    for line in lines:
      s = line.strip()
      if "=" not in s:
        continue
      lhs, rhs = s.split("=", 1)
      rhs = rhs.strip()

      if "copy(" in rhs and not rhs.startswith("copy-start") and not rhs.startswith("copy-done"):
        m = _SHAPE_RE.search(rhs)
        if m:
          nbytes = _shape_bytes(m.group(1), m.group(2))
          copy_cnt += mult
          copy_gib += (nbytes * mult) / gib

      for coll in ("all-gather", "reduce-scatter", "all-reduce", "all-to-all"):
        if (
            rhs.startswith(f"{coll}(")
            or f" {coll}(" in rhs
            or rhs.startswith(f"{coll}-start(")
            or f" {coll}-start(" in rhs
        ):
          shapes = [_shape_bytes(m.group(1), m.group(2)) for m in _SHAPE_RE.finditer(rhs)]
          max_b = max(shapes, default=0)
          dt_m = _SHAPE_RE.search(rhs)
          dt = dt_m.group(1) if dt_m else "unk"
          pin = default_pin
          if "core_ids" in s:
            if '"0","1"' in s or "0,1" in s or '"0", "1"' in s:
              pin = "sc01_both"
            elif '"0"' in s or "[0]" in s:
              pin = "sc0"
            elif '"1"' in s or "[1]" in s:
              pin = "sc1"
          elif pin == "tc_or_unpinned" and "S(7)" in s:
            pin = "sc_auto"
          key = f"{pin}:{coll}:{dt}"
          coll_by_pin_cnt[key] += mult
          coll_by_pin_gib[key] += (max_b * mult) / gib

      if "custom-call(" in rhs:
        mk = re.search(r'"kernel_name"\s*:\s*"([^"]+)"', s)
        kname = mk.group(1) if mk else None
        if not kname:
          if "sc_ragged_gather_reduce" in s:
            kname = "sc_ragged_gather_reduce_v2"
          elif "sc_ragged_gather" in s:
            kname = "sc_ragged_gather"
          else:
            m_op = re.search(r"%([A-Za-z0-9_.-]+)", lhs)
            op_id = m_op.group(1) if m_op else ""
            for cand in (
                "linear_combine_fwd",
                "linear_bwd_dgrad",
                "linear_bwd_wgrad",
                "gate_bwd_fused",
                "gmm_v2",
                "tgmm_v2",
                "ragged_silu_mul",
                "ragged_absmax_quantize",
                "ragged_quantize",
                "ragged_gather_reduce_tc",
                "ragged_gather_tc",
                "fused_conv1d_gdn",
                "gdn_bwd",
                "splash",
            ):
              if cand in op_id or cand in s:
                kname = re.sub(r"\.\d+$", "", op_id) if cand in op_id else cand
                break
        if kname:
          kernels_cnt[kname[:48]] += mult

  sc_async_by_pin: dict[str, int] = collections.defaultdict(int)
  for cname, lines in comp_blocks.items():
    if cname.startswith("fused_computation"):
      continue
    mult = comp_mult[cname]
    for line in lines:
      if 'async_execution_thread="sparsecore"' not in line:
        continue
      mc = re.search(r"calls=%([A-Za-z0-9_.-]+)", line)
      mb = re.search(r"backend_config=(\{.*\})\s*$", line)
      pin = "sc_none"
      if mb:
        try:
          cids = json.loads(mb.group(1)).get("sparse_core_config", {}).get("core_ids", [])
          pin = "sc_" + "_".join(str(x) for x in cids) if cids else "sc_empty"
        except ValueError:
          pin = "sc_parse_err"
      call_name = mc.group(1) if mc else ""
      body_lines = list(comp_blocks.get(call_name, []))
      for l2 in list(body_lines):
        for sub in _find_callees(l2):
          body_lines.extend(comp_blocks.get(sub, []))
      body_txt = "\n".join(body_lines)
      op_kind = "other"
      for k in (
          "all-gather",
          "reduce-scatter",
          "all-reduce",
          "sc_ragged_gather_reduce",
          "sc_ragged_gather",
          "gather_offload",
          "scatter_offload",
      ):
        if k in body_txt:
          op_kind = k
          break
      sc_async_by_pin[f"{pin}:{op_kind}"] += mult

  return {
      "hlo_found": True,
      "hlo_file": os.path.basename(hlo_path),
      "has_lossy_const": has_lossy_const,
      "while_bodies": while_bodies,
      "nested_whiles": nested_whiles,
      "copy_count_step": copy_cnt,
      "copy_gib_step": round(copy_gib, 2),
      "sc_async_by_pin_step": dict(sorted(sc_async_by_pin.items())),
      "collectives_count_by_pin": dict(sorted(coll_by_pin_cnt.items())),
      "collectives_gib_by_pin": {k: round(v, 2) for k, v in sorted(coll_by_pin_gib.items())},
      "kernels_count_step": dict(sorted(kernels_cnt.items())),
  }


def run_aot_compile(argv: Sequence[str] | None = None) -> dict[str, Any]:
  parser = argparse.ArgumentParser(description="Compile Qwen3.5-397B-A17B pure-JAX train_step AOT for TPU v7x.")
  parser.add_argument("--exp_name", required=True)
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--topology", default="tpu7x:4x4x4")
  parser.add_argument("--config_overrides_json", default="{}")
  parser.add_argument("--extra_libtpu_args", default="")
  args = parser.parse_args(argv)

  xla_dir = os.path.join(args.out_dir, "xla")
  os.makedirs(xla_dir, exist_ok=True)

  xla_flags = " ".join(
      [
          f"--xla_dump_to={xla_dir}",
          "--xla_dump_hlo_module_re=jit_train_step",
          "--xla_dump_large_constants=true",
          "--xla_dump_hlo_as_text=true",
          "--xla_dump_hlo_as_proto=true",
      ]
  )
  libtpu_args = BASE_LIBTPU_ARGS
  if args.extra_libtpu_args.strip():
    libtpu_args = f"{libtpu_args} {' '.join(args.extra_libtpu_args.strip().split())}"

  os.environ["XLA_FLAGS"] = xla_flags
  os.environ["LIBTPU_INIT_ARGS"] = libtpu_args
  os.environ.setdefault("MAXTEXT_GMM_V2_IN_KERNEL_FP8_DEQUANT", "1")

  raw_compiler_opts = parse_xla_compiler_options(libtpu_args)
  dump_opts = parse_xla_compiler_options(xla_flags)

  # Import JAX only AFTER setting XLA_FLAGS and LIBTPU_INIT_ARGS.
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top

  try:
    from jax._src import xla_bridge as _xb  # pylint: disable=g-import-not-at-top

    _xb._discover_and_register_pjrt_plugins()  # pylint: disable=protected-access
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  absl_synced = 0
  try:
    from absl import flags as absl_flags  # pylint: disable=g-import-not-at-top

    for k, v in {**raw_compiler_opts, **dump_opts}.items():
      if k in absl_flags.FLAGS:
        absl_flags.FLAGS[k].parse(v)
        absl_synced += 1
  except ImportError:
    pass

  pjrt_compiler_opts = dict(raw_compiler_opts)

  from maxtext.experimental import qwen_35_64c as qwen35_jax  # pylint: disable=g-import-not-at-top
  from maxtext.experimental.qwen_35_64c import max_async_overlap_transform  # pylint: disable=g-import-not-at-top

  overrides = json.loads(args.config_overrides_json)
  for k, v in list(overrides.items()):
    if isinstance(v, list) and ("tile" in k or "sublayers" in k):
      overrides[k] = tuple(v)
  cfg = qwen35_jax.full_config(**overrides)
  t0 = time.time()
  mesh = qwen35_jax.create_mesh(cfg, aot_topology=args.topology)
  max_async_overlap_transform.reset_transform_stats()
  if cfg.use_max_async_overlap_transform:
    registered_platforms = max_async_overlap_transform.register_transform(
        bundle_quant_scales=cfg.max_overlap_bundle_quant,
        bundle_scalar_all_reduce=cfg.max_overlap_bundle_scalar_ar,
        queue_filter=cfg.max_overlap_queues,
        order_mode=cfg.max_overlap_order_mode,
    )
  else:
    registered_platforms = []
    max_async_overlap_transform.unregister_transform()

  abstract_params = qwen35_jax.get_abstract_params(cfg, mesh)
  abstract_opt_state = qwen35_jax.get_abstract_opt_state(cfg, mesh, abstract_params)

  global_b = cfg.global_batch_size
  data_sharding = NamedSharding(mesh, qwen35_jax.data_pspec(cfg))
  tok_spec = jax.ShapeDtypeStruct((global_b, cfg.seq_len), jnp.int32, sharding=data_sharding)

  train_step_fn = qwen35_jax.make_train_step(cfg, mesh, donate_state=True)
  lowered = train_step_fn.lower(
      abstract_params,
      abstract_opt_state,
      tok_spec,
      tok_spec,
      tok_spec,
      tok_spec,
  )
  t_lower = round(time.time() - t0, 2)
  compiled = lowered.compile(compiler_options=pjrt_compiler_opts)
  t_total = round(time.time() - t0, 2)

  mem = compiled.memory_analysis()
  cost = compiled.cost_analysis()
  gib = 1024**3
  total_hbm_b = (
      mem.generated_code_size_in_bytes
      + mem.argument_size_in_bytes
      + max(0, mem.output_size_in_bytes - mem.alias_size_in_bytes)
      + mem.temp_size_in_bytes
  )

  comp_env_files = sorted(glob.glob(os.path.join(xla_dir, "*.tpu_comp_env.txt")))
  comp_env_txt = ""
  if comp_env_files:
    with open(comp_env_files[0], "r", encoding="utf-8", errors="replace") as f:
      comp_env_txt = f.read()
  tpu_comp_env_verified = (
      "xla_tpu_enable_all_experimental_scheduler_features: true" in comp_env_txt
      and "xla_tpu_scoped_vmem_limit_kib: 65472" in comp_env_txt
  )
  hlo_audit = audit_hlo_dual_sc(xla_dir, default_trip=cfg.num_superblocks)

  cost_info = {}
  if cost:
    c0 = cost[0] if isinstance(cost, list) else cost
    if isinstance(c0, dict):
      cost_info = {
          "flops": c0.get("flops"),
          "bytes_accessed": c0.get("bytes accessed"),
      }

  max_overlap_stats = max_async_overlap_transform.get_last_transform_stats()
  if cfg.use_max_async_overlap_transform and max_overlap_stats.get("modules_transformed", 0) == 0:
    raise RuntimeError(
        "cfg.use_max_async_overlap_transform=True, but"
        " max_async_overlap_transform ran on 0 HLO modules"
        f" (registered_platforms={registered_platforms})!"
    )

  summary = {
      "exp_name": args.exp_name,
      "lower_sec": t_lower,
      "compile_sec": t_total,
      "code_gib": round(mem.generated_code_size_in_bytes / gib, 3),
      "arg_gib": round(mem.argument_size_in_bytes / gib, 3),
      "output_gib": round(mem.output_size_in_bytes / gib, 3),
      "alias_gib": round(mem.alias_size_in_bytes / gib, 3),
      "temp_gib": round(mem.temp_size_in_bytes / gib, 3),
      "host_temp_gib": round(mem.host_temp_size_in_bytes / gib, 3),
      "total_aot_hbm_gib": round(total_hbm_b / gib, 3),
      "passes_86_5_gib_gate": (total_hbm_b / gib) <= 86.5,
      "tpu_comp_env_verified": tpu_comp_env_verified,
      "absl_flags_synced_count": absl_synced,
      "pjrt_compiler_options_count": len(pjrt_compiler_opts),
      "max_overlap_registered_platforms": registered_platforms,
      "max_overlap_stats": max_overlap_stats,
      **cost_info,
      **hlo_audit,
  }

  summary_path = os.path.join(args.out_dir, "aot_summary.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2), flush=True)
  return summary


def main() -> None:
  from absl import app  # pylint: disable=g-import-not-at-top
  from absl import flags as absl_flags  # pylint: disable=g-import-not-at-top

  def _parse_only_program_name(argv: list[str]) -> list[str]:
    absl_flags.FLAGS(argv[:1])
    return argv

  def _main(argv: list[str]) -> None:
    run_aot_compile(argv[1:])

  app.run(_main, flags_parser=_parse_only_program_name)


if __name__ == "__main__":
  main()
