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

"""Comprehensive TPU v7x (Ghostfish) benchmark, roofline, XProf, hill climb, and co-design suite for GDN & KDA."""

import collections
import functools
import glob
import math
import os
import re
import shutil
import statistics
import tempfile
import time
import types
from typing import Any

from absl.testing import absltest
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.configs import pyconfig
from maxtext.kernels import kda
from maxtext.kernels.kda import api as kda_api
from maxtext.kernels.kda import fused as kda_fused
from maxtext.kernels.kda import intra as kda_intra
from maxtext.kernels.kda import scan as kda_scan
from maxtext.models import olmoe3
from maxtext.models import qwen3
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import numpy as np

# ---------------------------------------------------------------------------
# 0. TPU v7x (Ghostfish) Per-TensorCore Hardware Roofline Constants
# ---------------------------------------------------------------------------
# Reference: tokamax/google/experimental/causal_conv1d_gated_delta_rule/gdn_roofline.py
V7X_HBM_BW_GBPS = 3686.5  # GB/s per TensorCore (7,373 GB/s per 2-core chip)
V7X_MXU_BF16_TFLOPS_P7 = 1153.5  # TFLOP/s per TensorCore at P7 (2.2 GHz)
V7X_VPU_FP32_TFLOPS_P7 = 9.0  # TFLOP/s per TensorCore at P7 (2.2 GHz)
V7X_MXU_BF16_TFLOPS_P3 = 996.2  # TFLOP/s per TensorCore at P3 (1.9 GHz)
V7X_VPU_FP32_TFLOPS_P3 = 7.77  # TFLOP/s per TensorCore at P3 (1.9 GHz)
V7X_MXU_DIM = 256  # 256x256 systolic array on TPU v7x


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


def _tile_flops_256(batch_mul: int, m: int, k_red: int, n: int, passes: int = 1) -> tuple[int, int]:
  """Returns (raw_flops, tile_quantized_flops) on a 256x256 MXU systolic array."""
  raw = batch_mul * 2 * m * k_red * n * passes
  m_q = _ceil_div(m, V7X_MXU_DIM) * V7X_MXU_DIM
  k_q = _ceil_div(k_red, V7X_MXU_DIM) * V7X_MXU_DIM
  n_q = _ceil_div(n, V7X_MXU_DIM) * V7X_MXU_DIM
  tiled = batch_mul * 2 * m_q * k_q * n_q * passes
  return raw, tiled


def compute_core_roofline_analytical(
    batch: int,
    seq_len: int,
    heads: int,
    dk: int,
    dv: int,
    chunk_size: int,
    is_kda: bool = True,
    state_f32: bool = True,
    act_bf16: bool = True,
    state_stride: int = 0,
    outer_block: int | None = 32,
    v7x_stacked_opt: bool = False,
) -> dict[str, Any]:
  """Computes exact raw MXU FLOPs, 256x256 tile-quantized MXU FLOPs, VPU FLOPs, and HBM bytes."""
  c = chunk_size
  n_chunks = seq_len // c
  bh = batch * heads
  bhn = bh * n_chunks
  b_act = 2 if act_bf16 else 4
  b_st = 4 if state_f32 else 2

  fwd_matmuls = []
  bwd_matmuls = []

  # 1. Intra-chunk QK^T and KK^T
  if not is_kda:
    # GDN scalar decay: [2*C, dk] @ [dk, C] in fwd; 2 dots in bwd
    fwd_matmuls.append(("[Q;K]@K^T (intra)", bhn, 2 * c, dk, c, 1))
    bwd_matmuls.append(("dQ_intra = dS@K", bhn, c, c, dk, 1))
    bwd_matmuls.append(("dK_intra = (dS^T+dKK^T)@Q_K", bhn, c, 2 * c, dk, 1))
  else:
    sb = 8
    nb = c // sb
    if v7x_stacked_opt and outer_block is not None:
      n_ob = c // outer_block
      # Outer history + inner subblocks with stacked [q_p; k_p] (2*sb=16 rows)
      fwd_matmuls.append(("intra_fwd_outer([q;k]@rin^T)", bhn * n_ob * (outer_block // sb), 2 * sb, dk, outer_block, 1))
      fwd_matmuls.append(("intra_fwd_inner([q;k]@right^T)", bhn * nb, 2 * sb, dk, outer_block, 1))
    elif outer_block is not None:
      n_ob = c // outer_block
      fwd_matmuls.append(("intra_fwd_outer_q", bhn * n_ob * (outer_block // sb), sb, dk, c, 1))
      fwd_matmuls.append(("intra_fwd_outer_k", bhn * n_ob * (outer_block // sb), sb, dk, c, 1))
      fwd_matmuls.append(("intra_fwd_inner_q", bhn * nb, sb, dk, outer_block, 1))
      fwd_matmuls.append(("intra_fwd_inner_k", bhn * nb, sb, dk, outer_block, 1))
    else:
      fwd_matmuls.append(("intra_fwd_1level_q", bhn * nb, sb, dk, c, 1))
      fwd_matmuls.append(("intra_fwd_1level_k", bhn * nb, sb, dk, c, 1))
    # Backward intra-chunk sub-block loop
    bwd_matmuls.append(("intra_bwd_dq", bhn * nb, sb, c, dk, 1))
    bwd_matmuls.append(("intra_bwd_dk_row", bhn * nb, sb, c, dk, 1))
    bwd_matmuls.append(("intra_bwd_dk_col_q", bhn * nb, c, sb, dk, 1))
    bwd_matmuls.append(("intra_bwd_dk_col_k", bhn * nb, c, sb, dk, 1))

  # 2. Triangular inverse (I + M)^{-1} block doubling (16 -> 32 -> 64 -> 128)
  # Forward block doubling: for d = 16, 32, 64: two [d, d] @ [d, d] dots per block
  inv_d = 16
  while inv_d < c:
    n_blk = c // (2 * inv_d)
    fwd_matmuls.append((f"inv_fwd_bd_{inv_d}", bhn * n_blk * 2, inv_d, inv_d, inv_d, 2))
    inv_d *= 2
  # Backward dM = -X^T @ dX @ X^T: two [C, C] @ [C, C] dots in 2-pass HIGH / 3-pass HIGHEST
  bwd_matmuls.append(("inv_bwd_X^T_dX", bhn, c, c, c, 2))
  bwd_matmuls.append(("inv_bwd_tmp_X^T", bhn, c, c, c, 2))

  # 3. WY solve & Inter-chunk State Recurrence + Readout
  if v7x_stacked_opt:
    fwd_matmuls.append(("uw = X @ [vb, kb]", bhn, c, c, dv + dk, 1))
    fwd_matmuls.append(("[w; qg] @ S", bhn, 2 * c, dk, dv, 1))
    fwd_matmuls.append(("out += scores @ delta", bhn, c, c, dv, 1))
    fwd_matmuls.append(("S_new += kc^T @ delta", bhn, dk, c, dv, 1))

    bwd_matmuls.append(("dsc = d_out @ delta^T", bhn, c, dv, c, 1))
    bwd_matmuls.append(("g_delta = scores^T @ d_out", bhn, c, c, dv, 1))
    bwd_matmuls.append(("dd += kc @ R", bhn, c, dk, dv, 1))
    bwd_matmuls.append(("w = X @ kb (recomp)", bhn, c, c, dk, 1))
    bwd_matmuls.append(("[dqg; dw] = [d_out; -dd] @ S^T", bhn, 2 * c, dv, dk, 1))
    bwd_matmuls.append(("dkc = delta @ R^T", bhn, c, dv, dk, 1))
    bwd_matmuls.append(("dS += [qg; w]^T @ [d_out; -dd]", bhn, dk, 2 * c, dv, 1))
    bwd_matmuls.append(("dX = [dd, dw] @ [vb, kb]^T", bhn, c, dv + dk, c, 1))
    bwd_matmuls.append(("[dvb, dkb] = X^T @ [dd, dw]", bhn, c, c, dv + dk, 1))
  else:
    fwd_matmuls.append(("u = X @ vb", bhn, c, c, dv, 1))
    fwd_matmuls.append(("w = X @ kb", bhn, c, c, dk, 1))
    fwd_matmuls.append(("delta = u - w @ S", bhn, c, dk, dv, 1))
    fwd_matmuls.append(("out_inter = qg @ S", bhn, c, dk, dv, 1))
    fwd_matmuls.append(("out_intra = scores @ delta", bhn, c, c, dv, 1))
    fwd_matmuls.append(("S_new += kc^T @ delta", bhn, dk, c, dv, 1))

    bwd_matmuls.append(("dqg = d_out @ S^T", bhn, c, dv, dk, 1))
    bwd_matmuls.append(("dsc = d_out @ delta^T", bhn, c, dv, c, 1))
    bwd_matmuls.append(("g_states = qg^T @ d_out", bhn, dk, c, dv, 1))
    bwd_matmuls.append(("g_delta = scores^T @ d_out", bhn, c, c, dv, 1))
    bwd_matmuls.append(("dd += kc @ R", bhn, c, dk, dv, 1))
    bwd_matmuls.append(("w = X @ kb (recomp)", bhn, c, c, dk, 1))
    bwd_matmuls.append(("dw = -dd @ S^T", bhn, c, dv, dk, 1))
    bwd_matmuls.append(("dkc = delta @ R^T", bhn, c, dv, dk, 1))
    bwd_matmuls.append(("dS -= w^T @ dd", bhn, dk, c, dv, 1))
    bwd_matmuls.append(("dX_v = dd @ vb^T", bhn, c, dv, c, 1))
    bwd_matmuls.append(("dX_k = dw @ kb^T", bhn, c, dk, c, 1))
    bwd_matmuls.append(("dvb = X^T @ dd", bhn, c, c, dv, 1))
    bwd_matmuls.append(("dkb = X^T @ dw", bhn, c, c, dk, 1))

  if state_stride == 0:
    # _states_kernel recomputes S_c from kc and delta
    bwd_matmuls.append(("states_recomp: kc^T @ delta", bhn, dk, c, dv, 1))
  elif state_stride == 2:
    bwd_matmuls.append(("states_odd_recomp: kc^T @ delta", bhn // 2, dk, c, dv, 1))

  fwd_raw_mxu, fwd_tile_mxu = 0, 0
  bwd_raw_mxu, bwd_tile_mxu = 0, 0
  kred_lt_256_raw, kred_lt_256_tile = 0, 0
  m_lt_256_raw, m_lt_256_tile = 0, 0
  matmul_details = []

  for phase, mlist in (("fwd", fwd_matmuls), ("bwd", bwd_matmuls)):
    for name, bmul, m, kred, n, passes in mlist:
      r_fl, t_fl = _tile_flops_256(bmul, m, kred, n, passes)
      if phase == "fwd":
        fwd_raw_mxu += r_fl
        fwd_tile_mxu += t_fl
      else:
        bwd_raw_mxu += r_fl
        bwd_tile_mxu += t_fl
      if kred < V7X_MXU_DIM:
        kred_lt_256_raw += r_fl
        kred_lt_256_tile += t_fl
      if m < V7X_MXU_DIM:
        m_lt_256_raw += r_fl
        m_lt_256_tile += t_fl
      matmul_details.append({
          "phase": phase,
          "name": name,
          "shape": f"[{m},{kred}]x[{kred},{n}] (x{bmul}, p={passes})",
          "kred_lt_256": kred < V7X_MXU_DIM,
          "m_lt_256": m < V7X_MXU_DIM,
          "raw_gflops": r_fl / 1e9,
          "tile_gflops": t_fl / 1e9,
          "mxu_eff_pct": 100.0 * r_fl / max(t_fl, 1),
      })

  # VPU FLOPs (decay cumsum, exp, 16x16 diagonal inverse + diagonal subblocks, state decay scaling, masks)
  if is_kda:
    sb = 8
    # Diagonal subblocks: [sb, sb, dk] pairwise exp + qk/kk reductions + cumsum + state decay
    fwd_vpu = bhn * (c * dk * 12 + (c // sb) * (sb * sb * dk * 6) + dk * dv * 2 + c * c * 8)
    bwd_vpu = bhn * (c * dk * 20 + (c // sb) * (sb * sb * dk * 10) + dk * dv * 4 + c * c * 12)
  else:
    fwd_vpu = bhn * (c * 12 + c * c * 8 + c * (dk + dv) * 6 + dk * dv * 2)
    bwd_vpu = bhn * (c * 20 + c * c * 12 + c * (dk + dv) * 12 + dk * dv * 4)

  # Theoretical minimum HBM bytes (inputs + outputs only) vs Fused Pallas HBM bytes (incl. WY/state residuals)
  gamma_bytes = (dk * 4 + 4) if is_kda else 8  # log_decay + beta
  fwd_min_bytes = bh * seq_len * ((2 * dk + 2 * dv) * b_act + gamma_bytes)
  bwd_min_bytes = bh * seq_len * ((4 * dk + 4 * dv) * b_act + 2 * gamma_bytes)

  # Actual Fused Pallas HBM bytes (saving delta [C, dv] bf16, xk [C, C] f32, scores [C, C] bf16, and state_stride)
  wy_res_bytes = bhn * (c * dv * 2 + c * c * 4 + c * c * 2)
  if state_stride == 1:
    st_res_bytes = bhn * dk * dv * b_st
    st_bwd_extra = 0
  elif state_stride == 2:
    st_res_bytes = (bhn // 2) * dk * dv * b_st
    st_bwd_extra = (bhn // 2) * (c * dk * b_act + c * dk * 4 + c * dv * 2)
  else:  # state_stride == 0: _states_kernel writes & _bwd_kernel reads states in HBM + reads k, g, delta in _states_kernel
    st_res_bytes = 0
    st_bwd_extra = bhn * (2 * dk * dv * b_st + c * dk * b_act + (c * dk * 4 if is_kda else c * 4) + c * dv * 2)

  fwd_pallas_bytes = fwd_min_bytes + wy_res_bytes + st_res_bytes
  bwd_pallas_bytes = bwd_min_bytes + wy_res_bytes + st_res_bytes + st_bwd_extra

  tot_raw_mxu = fwd_raw_mxu + bwd_raw_mxu
  tot_tile_mxu = fwd_tile_mxu + bwd_tile_mxu
  tot_vpu = fwd_vpu + bwd_vpu
  tot_min_bytes = fwd_min_bytes + bwd_min_bytes
  tot_pallas_bytes = fwd_pallas_bytes + bwd_pallas_bytes

  # Roofline times in ms on 1 TPU v7x TensorCore (P7 peak)
  t_mxu_raw_ms = tot_raw_mxu / (V7X_MXU_BF16_TFLOPS_P7 * 1e9)
  t_mxu_tile_ms = tot_tile_mxu / (V7X_MXU_BF16_TFLOPS_P7 * 1e9)
  t_vpu_ms = tot_vpu / (V7X_VPU_FP32_TFLOPS_P7 * 1e9)
  t_comp_raw_ms = t_mxu_raw_ms + t_vpu_ms
  t_comp_tile_ms = t_mxu_tile_ms + t_vpu_ms
  t_hbm_min_ms = tot_min_bytes / (V7X_HBM_BW_GBPS * 1e6)
  t_hbm_pallas_ms = tot_pallas_bytes / (V7X_HBM_BW_GBPS * 1e6)

  fwd_comp_tile_ms = (fwd_tile_mxu / (V7X_MXU_BF16_TFLOPS_P7 * 1e9)) + (fwd_vpu / (V7X_VPU_FP32_TFLOPS_P7 * 1e9))
  fwd_hbm_pallas_ms = fwd_pallas_bytes / (V7X_HBM_BW_GBPS * 1e6)

  return {
      "fwd_raw_gflops": fwd_raw_mxu / 1e9,
      "fwd_tile_gflops": fwd_tile_mxu / 1e9,
      "bwd_raw_gflops": bwd_raw_mxu / 1e9,
      "bwd_tile_gflops": bwd_tile_mxu / 1e9,
      "tot_raw_gflops": tot_raw_mxu / 1e9,
      "tot_tile_gflops": tot_tile_mxu / 1e9,
      "tot_vpu_gflops": tot_vpu / 1e9,
      "mxu_tile_eff_pct": 100.0 * tot_raw_mxu / max(tot_tile_mxu, 1),
      "kred_lt_256_share_pct": 100.0 * kred_lt_256_raw / max(tot_raw_mxu, 1),
      "m_lt_256_share_pct": 100.0 * m_lt_256_raw / max(tot_raw_mxu, 1),
      "fwd_min_mb": fwd_min_bytes / 1e6,
      "fwd_pallas_mb": fwd_pallas_bytes / 1e6,
      "tot_min_mb": tot_min_bytes / 1e6,
      "tot_pallas_mb": tot_pallas_bytes / 1e6,
      "ai_min_flops_per_byte": tot_raw_mxu / max(tot_min_bytes, 1),
      "ai_pallas_flops_per_byte": tot_raw_mxu / max(tot_pallas_bytes, 1),
      "t_mxu_raw_ms": t_mxu_raw_ms,
      "t_mxu_tile_ms": t_mxu_tile_ms,
      "t_vpu_ms": t_vpu_ms,
      "t_comp_raw_ms": t_comp_raw_ms,
      "t_comp_tile_ms": t_comp_tile_ms,
      "t_hbm_min_ms": t_hbm_min_ms,
      "t_hbm_pallas_ms": t_hbm_pallas_ms,
      "fwd_comp_tile_ms": fwd_comp_tile_ms,
      "fwd_hbm_pallas_ms": fwd_hbm_pallas_ms,
      "matmul_details": matmul_details,
  }


def format_perf_vs_roofline(
    label: str,
    t_fwd_ms: float,
    t_fb_ms: float,
    roof: dict[str, Any],
    hlo_info: dict[str, Any] | None = None,
    max_diff: float = 0.0,
) -> str:
  """Formats measured latency against HBM and Compute (raw & tile-quantized) rooflines and overlap."""
  actual_bytes_mb = (
      hlo_info["gb_accessed"] * 1e3
      if hlo_info and hlo_info.get("gb_accessed", 0) > 0
      else roof["tot_pallas_mb"]
  )
  t_hbm_ms = actual_bytes_mb / V7X_HBM_BW_GBPS
  t_comp_raw = roof["t_comp_raw_ms"]
  t_comp_tile = roof["t_comp_tile_ms"]
  floor_tile_ms = max(t_hbm_ms, t_comp_tile)
  bound = "Compute(Tile+VPU)" if t_comp_tile >= t_hbm_ms else "HBM_BW"

  bw_gbps = (actual_bytes_mb / 1e3) / (t_fb_ms / 1e3)
  bw_util_pct = 100.0 * bw_gbps / V7X_HBM_BW_GBPS
  raw_tflops = (roof["tot_raw_gflops"] / 1e3) / (t_fb_ms / 1e3)
  raw_mxu_util_pct = 100.0 * raw_tflops / V7X_MXU_BF16_TFLOPS_P7
  tile_tflops = (roof["tot_tile_gflops"] / 1e3) / (t_fb_ms / 1e3)
  tile_mxu_util_pct = 100.0 * tile_tflops / V7X_MXU_BF16_TFLOPS_P7
  roof_attain_pct = 100.0 * floor_tile_ms / t_fb_ms

  # Overlap ratio: alpha = (T_hbm + T_comp_tile - T_meas) / min(T_hbm, T_comp_tile)
  min_hc = max(min(t_hbm_ms, t_comp_tile), 1e-6)
  overlap_ratio = (t_hbm_ms + t_comp_tile - t_fb_ms) / min_hc

  t_bwd_ms = t_fb_ms - t_fwd_ms
  return (
      f"[{label:34s}] fwd={t_fwd_ms:6.3f} ms | bwd={t_bwd_ms:6.3f} ms | fwd+bwd={t_fb_ms:6.3f} ms (x24={24*t_fb_ms:6.1f} ms) "
      f"| Floor={floor_tile_ms:5.3f} ms (HBM={t_hbm_ms:5.3f} ms [{actual_bytes_mb:6.1f} MB], "
      f"CompTile={t_comp_tile:5.3f} ms [MXU_raw={roof['t_mxu_raw_ms']:5.3f}, MXU_tile={roof['t_mxu_tile_ms']:5.3f}, VPU={roof['t_vpu_ms']:5.3f}], {bound}) "
      f"| Attain={roof_attain_pct:5.1f}% | BW={bw_gbps:6.1f} GB/s ({bw_util_pct:4.1f}%) "
      f"| MXU_raw={raw_tflops:5.1f} TF/s ({raw_mxu_util_pct:4.1f}%) | MXU_tile={tile_tflops:5.1f} TF/s ({tile_mxu_util_pct:4.1f}%) "
      f"| Overlap={overlap_ratio:+5.2f} | diff={max_diff:.2e}"
  )


def _time_fn(fn, *args, warmup: int = 1, iters: int = 5):
  for _ in range(warmup):
    out = fn(*args)
    jax.block_until_ready(out)
  times = []
  for _ in range(iters):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    times.append((time.perf_counter() - t0) * 1e3)
  return statistics.median(times), out


def _profile_hlo(fn, *args):
  """Extracts XLA cost analysis, memory analysis, and HLO op counts."""
  compiled = fn.lower(*args).compile() if hasattr(fn, "lower") else fn
  ca = compiled.cost_analysis()
  ca = ca[0] if isinstance(ca, list) else (ca or {})
  ma = compiled.memory_analysis()
  ma = ma[0] if isinstance(ma, list) else ma
  temp_mb = getattr(ma, "temp_size_in_bytes", 0) / 1e6 if ma is not None else 0.0
  hlo = compiled.as_text()
  cats = collections.Counter(re.findall(r"fusion_kind=k(\w+)", hlo))
  cats["custom_call"] = len(re.findall(r"custom-call\(", hlo))
  return {
      "gflops": round(ca.get("flops", 0.0) / 1e9, 2),
      "gb_accessed": round(ca.get("bytes accessed", 0.0) / 1e9, 4),
      "temp_mb": round(temp_mb, 1),
      "fusions": dict(cats),
  }


def _time_and_profile_hlo(fn, *args, warmup: int = 1, iters: int = 5):
  """Compiles fn once, runs timing, and extracts HLO cost/memory analysis without double compilation."""
  compiled = fn.lower(*args).compile()
  t_ms, out = _time_fn(compiled, *args, warmup=warmup, iters=iters)
  hlo = _profile_hlo(compiled)
  return t_ms, out, hlo


def _profile_xplane(fn, *args, iters: int = 3, top_k: int = 12) -> dict[str, Any]:
  """Captures a /device:TPU:0 XPlane trace and extracts Disjoint Interval Union vs sum of op durations."""
  prof_dir = tempfile.mkdtemp(prefix="v7x_xplane_")
  try:
    out = fn(*args)
    jax.block_until_ready(out)
    jax.profiler.start_trace(prof_dir)
    for _ in range(iters):
      out = fn(*args)
      jax.block_until_ready(out)
    jax.profiler.stop_trace()

    files = glob.glob(os.path.join(prof_dir, "**", "*.xplane.pb"), recursive=True)
    if not files:
      return {}
    pd = jax.profiler.ProfileData.from_file(sorted(files)[-1])
    planes = [p for p in pd.planes if p.name.startswith("/device:TPU:0")]
    if not planes:
      return {}
    plane = planes[0]
    lines = {l.name: l for l in plane.lines}
    ops_line = lines.get("XLA Ops")
    if ops_line is None:
      return {}

    intervals = []
    by_op = collections.defaultdict(lambda: [0.0, 0])
    sum_ns = 0.0
    for ev in ops_line.events:
      s_ns = getattr(ev, "offset_ns", getattr(ev, "start_ns", 0))
      d_ns = ev.duration_ns
      if d_ns <= 0:
        continue
      intervals.append((s_ns, s_ns + d_ns))
      sum_ns += d_ns
      m = re.match(r"%?([\w.\-]+) = (\(.*?\)|[\w\[\],:\d]+(?:\{[^}]*\})?) ([\w\-]+)", ev.name)
      if m:
        iname, shape, opcode = m.groups()
        stem = re.sub(r"[.\d]+(\.refs_\d+_)?$", "", iname)
        shape = re.sub(r"\{[^}]*\}", "", shape)
        first = re.match(r"\(?([^,)]*\])", shape)
        first = first.group(1) if first else shape
        key = f"{opcode}:{stem}:{first}"
      else:
        key = ev.name[:48]
      by_op[key][0] += d_ns
      by_op[key][1] += 1

    # Disjoint interval union
    intervals.sort()
    union_ns = 0.0
    if intervals:
      cur_s, cur_e = intervals[0]
      for s, e in intervals[1:]:
        if s <= cur_e:
          cur_e = max(cur_e, e)
        else:
          union_ns += cur_e - cur_s
          cur_s, cur_e = s, e
      union_ns += cur_e - cur_s

    top_ops = [
        (k, round(v[0] / iters / 1e6, 3), round(v[1] / iters, 1))
        for k, v in sorted(by_op.items(), key=lambda kv: -kv[1][0])[:top_k]
    ]
    return {
        "xla_ops_sum_ms": round(sum_ns / iters / 1e6, 3),
        "xla_ops_union_ms": round(union_ns / iters / 1e6, 3),
        "top_ops": top_ops,
    }
  finally:
    shutil.rmtree(prof_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. New TPU v7x Matmul-Tiled Fused Pallas KDA Kernel (pallas_fused_v7x_opt)
# ---------------------------------------------------------------------------


def _intra_fwd_rows_two_level_v7x(
    q_ref, k_ref, g_ref, s_ref, kk_ref, right_ref, rin_ref, sub_block: int, outer: int, compute_dtype
):
  """v7x-optimized two-level intra-chunk forward: slices rin_ref[P:P+outer] (4x narrower) and stacks [q; k]."""
  f32, cd = jnp.float32, compute_dtype
  c = g_ref.shape[0]
  sb = sub_block
  irow = jax.lax.broadcasted_iota(jnp.int32, (sb, 1), 0)
  jcol = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  right_ref[...] = jnp.zeros_like(right_ref)
  rin_ref[...] = jnp.zeros_like(rin_ref)
  for ob in range(c // outer):
    P = ob * outer
    g_o = g_ref[P : P + outer, :]
    gP = g_o[0:1, :]
    if ob > 0:
      PP = P - outer
      if ob > 1:
        right_ref[0:PP, :] = right_ref[0:PP, :] * jnp.exp(jnp.minimum(gP - g_ref[PP : PP + 1, :], 30.0))
      right_ref[PP:P, :] = k_ref[PP:P, :].astype(f32) * jnp.exp(jnp.minimum(gP - g_ref[PP:P, :], 30.0))
      left = jnp.exp(jnp.minimum(g_o - gP, 30.0))
      right = right_ref[...].astype(cd)
      qko_l = jnp.concatenate(
          [(q_ref[P : P + outer, :].astype(f32) * left).astype(cd), (k_ref[P : P + outer, :].astype(f32) * left).astype(cd)],
          axis=0,
      )
      sk_o = kda_intra._nt(qko_l, right)
      s_ref[P : P + outer, :] = sk_o[:outer, :]
      kk_ref[P : P + outer, :] = sk_o[outer:, :]
    else:
      s_ref[P : P + outer, :] = jnp.zeros((outer, c), f32)
      kk_ref[P : P + outer, :] = jnp.zeros((outer, c), f32)
    if ob > 0:
      rin_ref[P - outer : P, :] = jnp.zeros((outer, rin_ref.shape[1]), f32)
    for r in range(outer // sb):
      p = P + r * sb
      gr = g_ref[p : p + sb, :]
      qr = q_ref[p : p + sb, :].astype(f32)
      kr = k_ref[p : p + sb, :].astype(f32)
      gp = gr[0:1, :]
      row_q = s_ref[p : p + sb, :]
      row_k = kk_ref[p : p + sb, :]
      if r > 0:
        pp = p - sb
        if r > 1:
          rin_ref[P:pp, :] = rin_ref[P:pp, :] * jnp.exp(jnp.minimum(gp - g_ref[pp : pp + 1, :], 30.0))
        rin_ref[pp:p, :] = k_ref[pp:p, :].astype(f32) * jnp.exp(jnp.minimum(gp - g_ref[pp:p, :], 30.0))
        left = jnp.exp(jnp.minimum(gr - gp, 30.0))
        rin = rin_ref[...].astype(cd)
        qkr_l = jnp.concatenate([(qr * left).astype(cd), (kr * left).astype(cd)], axis=0)
        sk_in = kda_intra._nt(qkr_l, rin)
        row_q = row_q + sk_in[:sb, :]
        row_k = row_k + sk_in[sb:, :]
      for jj in range(sb):
        e = jnp.exp(jnp.where(irow >= jj, jnp.minimum(gr - gr[jj : jj + 1, :], 30.0), -jnp.inf))
        t = e * kr[jj : jj + 1, :]
        sel = jcol == (p + jj)
        row_q = jnp.where(sel, jnp.sum(qr * t, axis=1, keepdims=True), row_q)
        row_k = jnp.where(sel, jnp.sum(kr * t, axis=1, keepdims=True), row_k)
      s_ref[p : p + sb, :] = row_q
      kk_ref[p : p + sb, :] = row_k


def _intra_bwd_compute_two_level_v7x(
    q_ref, k_ref, g_ref, ds_ref, dk_ref, rexp_ref, col_ref, dq_s, row_s, piv_s, rin_ref, sub_block: int, outer: int, compute_dtype
):
  """v7x-optimized two-level intra-chunk backward: stacks [ds_c; dk_c] and [q; k] to double M and K_red."""
  f32, cd = jnp.float32, compute_dtype
  c = g_ref.shape[0]
  sb = sub_block
  irow = jax.lax.broadcasted_iota(jnp.int32, (sb, 1), 0)
  jcol = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  rexp_ref[...] = jnp.zeros_like(rexp_ref)
  rin_ref[...] = jnp.zeros_like(rin_ref)
  col_ref[...] = jnp.zeros_like(col_ref)
  piv_s[...] = jnp.zeros_like(piv_s)
  kf_all = k_ref[...].astype(f32)
  for ob in range(c // outer):
    P = ob * outer
    g_o = g_ref[P : P + outer, :]
    q_o = q_ref[P : P + outer, :].astype(f32)
    k_o = k_ref[P : P + outer, :].astype(f32)
    ds_o = ds_ref[P : P + outer, :]
    dk_o = dk_ref[P : P + outer, :]
    gP = g_o[0:1, :]
    if ob > 0:
      PP = P - outer
      if ob > 1:
        rexp_ref[0:PP, :] = rexp_ref[0:PP, :] * jnp.exp(jnp.minimum(gP - g_ref[PP : PP + 1, :], 30.0))
      rexp_ref[PP:P, :] = jnp.exp(jnp.minimum(gP - g_ref[PP:P, :], 30.0))
      left = jnp.exp(jnp.minimum(g_o - gP, 30.0))
      rexp = rexp_ref[...]
      kr_all = (kf_all * rexp).astype(cd)
      ds_c, dk_c = ds_o.astype(cd), dk_o.astype(cd)
      dsk_c = jnp.concatenate([ds_c, dk_c], axis=0)
      dq_row_o = jnp.dot(dsk_c, kr_all, preferred_element_type=f32)
      dq_o = left * dq_row_o[:outer, :]
      row_o = left * dq_row_o[outer:, :]
      qko_l = jnp.concatenate([(q_o * left).astype(cd), (k_o * left).astype(cd)], axis=0)
      col_blk = rexp * kda_intra._tn(dsk_c, qko_l)
      col_ref[...] += col_blk
      row_tot = jnp.sum(q_o * dq_o + k_o * row_o, axis=0, keepdims=True)
      col_tot = jnp.sum(kf_all * col_blk, axis=0, keepdims=True)
      piv_s[P : P + 1, :] = col_tot - row_tot
    else:
      dq_o = jnp.zeros_like(q_o)
      row_o = jnp.zeros_like(q_o)
    dq_s[P : P + outer, :] = dq_o
    row_s[P : P + outer, :] = row_o
    if ob > 0:
      rin_ref[P - outer : P, :] = jnp.zeros((outer, rin_ref.shape[1]), f32)
    for r in range(outer // sb):
      p = P + r * sb
      gr = g_ref[p : p + sb, :]
      qr = q_ref[p : p + sb, :].astype(f32)
      kr = k_ref[p : p + sb, :].astype(f32)
      ds_r = ds_ref[p : p + sb, :]
      dk_r = dk_ref[p : p + sb, :]
      gp = gr[0:1, :]
      dq_r = dq_s[p : p + sb, :]
      row_r = row_s[p : p + sb, :]
      if r > 0:
        pp = p - sb
        if r > 1:
          rin_ref[P:pp, :] = rin_ref[P:pp, :] * jnp.exp(jnp.minimum(gp - g_ref[pp : pp + 1, :], 30.0))
        rin_ref[pp:p, :] = jnp.exp(jnp.minimum(gp - g_ref[pp:p, :], 30.0))
        left = jnp.exp(jnp.minimum(gr - gp, 30.0))
        rin_blk = rin_ref[P : P + outer, :]
        kin = (k_o * rin_blk).astype(cd)
        ds_c = ds_r[:, P : P + outer].astype(cd) if outer < c else ds_r.astype(cd)
        dk_c = dk_r[:, P : P + outer].astype(cd) if outer < c else dk_r.astype(cd)
        dsk_in = jnp.concatenate([ds_c, dk_c], axis=0)
        dq_row_in = jnp.dot(dsk_in, kin, preferred_element_type=f32)
        dq_in = left * dq_row_in[:sb, :]
        row_in = left * dq_row_in[sb:, :]
        qkr_l = jnp.concatenate([(qr * left).astype(cd), (kr * left).astype(cd)], axis=0)
        col_in = rin_blk * kda_intra._tn(dsk_in, qkr_l)
        col_ref[P : P + outer, :] += col_in
        row_tot = jnp.sum(qr * dq_in + kr * row_in, axis=0, keepdims=True)
        col_tot = jnp.sum(k_o * col_in, axis=0, keepdims=True)
        piv_s[p : p + 1, :] = piv_s[p : p + 1, :] + col_tot - row_tot
        dq_r = dq_r + dq_in
        row_r = row_r + row_in
      col_d = jnp.zeros_like(qr)
      for jj in range(sb):
        e = jnp.exp(jnp.where(irow >= jj, jnp.minimum(gr - gr[jj : jj + 1, :], 30.0), -jnp.inf))
        sel = jcol == (p + jj)
        cs = jnp.sum(jnp.where(sel, ds_r, 0.0), axis=1, keepdims=True)
        ck = jnp.sum(jnp.where(sel, dk_r, 0.0), axis=1, keepdims=True)
        t = e * kr[jj : jj + 1, :]
        dq_r = dq_r + cs * t
        row_r = row_r + ck * t
        col_d = jnp.where(irow == jj, jnp.sum((cs * qr + ck * kr) * e, axis=0, keepdims=True), col_d)
      dq_s[p : p + sb, :] = dq_r
      row_s[p : p + sb, :] = row_r
      col_ref[p : p + sb, :] += col_d
  dq, row, col = dq_s[...], row_s[...], col_ref[...]
  return dq, row + col, q_ref[...].astype(f32) * dq + k_ref[...].astype(f32) * (row - col) + piv_s[...]


def _fwd_kernel_v7x_opt(*refs, precision, save_wy, state_stride, state_f32, cumsum, outer, mode="full"):
  """TPU v7x-optimized forward kernel: stacks [vb, kb], stacks [w; qg] @ S (M=2*C=256), and slices rin_ref[32, d]."""
  n_in = 8 + (2 if mode == "from_wy" else 0)
  if mode == "wy":
    n_out = 2
  else:
    n_out = 1 + int(bool(state_stride)) + (1 if mode == "from_wy" else 3) * int(save_wy)
  ins, outs, scratch = refs[:n_in], refs[n_in : n_in + n_out], refs[n_in + n_out :]
  q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref = (kda_fused._view(r) for r in ins[:8])
  outs = [kda_fused._view(r) for r in outs]
  s_ref, sc_s, kk_s, right_s = scratch[:4]
  extra = list(scratch[4:])

  if cumsum:
    g_s = extra.pop(0)
    g_s[...] = kda_fused._tril_cumsum(g_ref[...])
    g_ref = g_s

  f32, bf16 = jnp.float32, jnp.bfloat16
  rin_s = extra.pop(0) if outer else None
  c = g_ref.shape[0]
  dv = v_ref.shape[1]
  row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)

  if mode == "from_wy":
    xk_in, sc_in = (kda_fused._view(r) for r in ins[8:])
    x, kk = kda_fused._unpack_x_kk(xk_in[...], row, col)
    scores_bf = sc_in[...]
  else:
    if outer:
      _intra_fwd_rows_two_level_v7x(q_ref, k_ref, g_ref, sc_s, kk_s, right_s, rin_s, kda_fused._SUB_BLOCK, outer, bf16)
    else:
      kda_intra.intra_fwd_rows(q_ref, k_ref, g_ref, sc_s, kk_s, right_s, kda_fused._SUB_BLOCK, bf16)
    _, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
    scores = jnp.where(causal, sc_s[...], 0.0)
    kk = jnp.where(strict, kk_s[...], 0.0)
    x = kda_fused._inverse(kk * beta_ref[...], (row == col).astype(f32), precision)
    scores_bf = scores.astype(bf16)
  if mode == "wy":
    xk_out, sc_out = outs
    xk_out[...] = kda_fused._pack_x_kk(x, kk, row, col)
    sc_out[...] = scores_bf
    return

  out_ref = outs.pop(0)
  st_ref = outs.pop(0) if state_stride else None

  @pl.when(pl.program_id(1) == 0)
  def _():
    s_ref[...] = jnp.zeros_like(s_ref)

  o = kda_fused._chunk_operands(q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref)
  x_bf = x.astype(bf16)
  s = s_ref[...]
  s_bf = s.astype(bf16)
  if state_stride == 1 and st_ref is not None:
    st_ref[...] = s if state_f32 else s_bf
  elif state_stride and st_ref is not None:

    @pl.when(pl.program_id(1) % state_stride == 0)
    def _():
      assert st_ref is not None
      st_ref[...] = s if state_f32 else s_bf

  # v7x Opt 1: Fuse u = x_bf @ vb and w = x_bf @ kb into one [C, C] @ [C, dv + dk] matmul
  vb_kb = jnp.concatenate([o["vb"], o["kb"]], axis=1)
  uw = jnp.dot(x_bf, vb_kb, preferred_element_type=f32)
  u, w = uw[:, :dv], uw[:, dv:]
  # v7x Opt 2: Stack [w_bf; qg] ([2*C, dk] = [256, 256]) @ s_bf ([256, 512]) -> 100% MXU row & contraction tile!
  wqg = jnp.concatenate([w.astype(bf16), o["qg"]], axis=0)
  ws_qgs = jnp.dot(wqg, s_bf, preferred_element_type=f32)
  ws, qgs = ws_qgs[:c, :], ws_qgs[c:, :]
  delta_bf = (u - ws).astype(bf16)
  if save_wy:
    outs.pop(0)[...] = delta_bf
    if mode == "full":
      xk_out, sc_out = outs
      xk_out[...] = kda_fused._pack_x_kk(x, kk, row, col)
      sc_out[...] = scores_bf
  out = qgs + jnp.dot(scores_bf, delta_bf, preferred_element_type=f32)
  out_ref[...] = out.astype(out_ref.dtype)
  new = s * dl_ref[...] + kda_fused._tn(o["kc"], delta_bf)
  s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)


def _bwd_kernel_v7x_opt(*refs, precision, cumsum, outer, state_stride, state_f32):
  """TPU v7x-optimized backward kernel: fuses K_red=2*C=256 state adjoint & M=2*C=256 state readout matmuls."""
  n_in = 13 + (5 if state_stride == 2 else 0)
  ins, outs, scratch = refs[:n_in], refs[n_in : n_in + 6], refs[n_in + 6 :]
  q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref, st_ref, de_ref, xk_ref, sc_ref, do_ref = (
      kda_fused._view(r) for r in ins[:13]
  )
  dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref, ddl_ref = (kda_fused._view(r) for r in outs)
  r_ref, ds_s, dkk_s, rexp_s, col_s, dq_s, row_s, piv_s = scratch[:8]
  extra = list(scratch[8:])

  @pl.when(pl.program_id(1) == 0)
  def _():
    r_ref[...] = jnp.zeros_like(r_ref)

  if cumsum:
    g_s = extra.pop(0)
    g_s[...] = kda_fused._tril_cumsum(g_ref[...])
    g_ref = g_s
  rin_s = extra.pop(0) if outer else None
  if state_stride == 2:
    st_s = extra.pop(0)
    gp_s = extra.pop(0) if cumsum else None
    kp_ref, gp_ref, segp_ref, dlp_ref, dep_ref = (kda_fused._view(r) for r in ins[13:18])
    kda_fused._rebuilt_state(st_ref, kp_ref, gp_ref, segp_ref, dlp_ref, dep_ref, st_s, gp_s, state_f32)
    st_ref = st_s

  f32, bf16 = jnp.float32, jnp.bfloat16
  c = g_ref.shape[0]
  dv = v_ref.shape[1]
  row, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
  o = kda_fused._chunk_operands(q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref)
  x, kk = kda_fused._unpack_x_kk(xk_ref[...], row, jax.lax.broadcasted_iota(jnp.int32, (1, c), 1))
  x_bf, scores_bf = x.astype(bf16), sc_ref[...]
  states_f, delta = st_ref[...].astype(f32), de_ref[...]
  states = states_f.astype(bf16)
  d_out = do_ref[...].astype(bf16)

  # 1. Intra-chunk readout grads & dd
  dsc = kda_fused._nt(d_out, delta)
  g_delta = kda_fused._tn(scores_bf, d_out)
  r = r_ref[...]
  r_bf = r.astype(bf16)
  dd = g_delta + jnp.dot(o["kc"], r_bf, preferred_element_type=f32)
  dd_bf = dd.astype(bf16)
  w_bf = jnp.dot(x_bf, o["kb"], preferred_element_type=f32).astype(bf16)
  dkc = kda_fused._nt(delta, r_bf)
  ddl_ref[...] = jnp.sum(states_f * r, axis=-1, keepdims=True)

  # 2. v7x Opt: Stack [d_out; -dd_bf] ([2*C, dv] = [256, 512]) for:
  #    (a) [dqg; dw] = [d_out; -dd_bf] @ states^T ([256, 512] @ [512, 256], M=256 100% MXU tile!)
  #    (b) dS_update = [qg; w_bf]^T @ [d_out; -dd_bf] ([256, 256] @ [256, 512], K_red=2*C=256 100% MXU tile!)
  do_neg_dd = jnp.concatenate([d_out, -dd_bf], axis=0)
  dqg_dw = kda_fused._nt(do_neg_dd, states)
  dqg = dqg_dw[:c, :]
  dw_bf = dqg_dw[c:, :].astype(bf16)

  qg_w = jnp.concatenate([o["qg"], w_bf], axis=0)
  ds_update = kda_fused._tn(qg_w, do_neg_dd)
  r_ref[...] = r * dl_ref[...] + ds_update

  # 3. v7x Opt: Stack [dd_bf, dw_bf] ([C, dv + dk]) and [vb, kb] ([C, dv + dk])
  dd_dw = jnp.concatenate([dd_bf, dw_bf], axis=1)
  vb_kb = jnp.concatenate([o["vb"], o["kb"]], axis=1)
  dx = kda_fused._nt(dd_dw, vb_kb)
  dvb_dkb = kda_fused._tn(x_bf, dd_dw)
  dvb, dkb = dvb_dkb[:, :dv], dvb_dkb[:, dv:]

  # 4. X = (I + m)^-1 backward
  dm = -kda_fused._mm(
      kda_fused._mm(x, dx, precision, ((0,), (0,))), x, precision, ((1,), (1,))
  )
  dm = jnp.where(strict, dm, 0.0)
  dbeta = jnp.sum(dm * kk, axis=1, keepdims=True)
  dkk_s[...] = dm * o["beta"]
  ds_s[...] = jnp.where(causal, dsc, 0.0)
  if outer:
    dq_i, dk_i, dg_i = _intra_bwd_compute_two_level_v7x(
        q_ref, k_ref, g_ref, ds_s, dkk_s, rexp_s, col_s, dq_s, row_s, piv_s, rin_s, kda_fused._SUB_BLOCK, outer, bf16
    )
  else:
    dq_i, dk_i, dg_i = kda_intra.intra_bwd_compute(
        q_ref, k_ref, g_ref, ds_s, dkk_s, rexp_s, col_s, dq_s, row_s, piv_s, kda_fused._SUB_BLOCK, bf16
    )
  eg, fp, ecar, qf, kf, beta = o["eg"], o["fp"], o["ecar"], o["qf"], o["kf"], o["beta"]
  kegfp = kf * eg * fp
  dv_ref[...] = (dvb * beta).astype(dv_ref.dtype)
  dbeta_ref[...] = (
      dbeta
      + jnp.sum(dvb * v_ref[...].astype(f32), axis=1, keepdims=True)
      + jnp.sum(dkb * kegfp, axis=1, keepdims=True)
  )
  dq_ref[...] = (dq_i + dqg * eg * fp).astype(dq_ref.dtype)
  dk_ref[...] = (dk_i + dkb * (eg * beta * fp) + dkc * ecar).astype(dk_ref.dtype)
  carry = dkc * kf * ecar
  dg = dg_i + dkb * kegfp * beta + dqg * (qf * eg * fp) - carry
  dg = dg + jnp.where(row == c - 1, jnp.sum(carry, axis=0, keepdims=True), 0.0)
  if cumsum:
    dg = kda_fused._tril_cumsum(dg, reverse=True)
  dg_ref[...] = dg


# ---------------------------------------------------------------------------
# 2. VMEM-Fused Token-Major Pallas GDN Kernel (Scalar & Tile-Grouped Decay)
# ---------------------------------------------------------------------------


def _gdn_fwd_kernel(*refs, precision, save_wy, state_stride, state_f32):
  """VMEM-fused token-major Pallas forward kernel for GDN (scalar or tile-grouped decay).

  Because decay g has shape [C, 1] (or [C, G] with G << dk), intra-chunk QK^T and KK^T
  require ZERO sub-block loops: a single [2*C, dk] @ [dk, C] MXU matmul computes both!
  """
  n_in = 8
  n_out = 1 + int(bool(state_stride)) + 3 * int(save_wy)
  ins, outs, scratch = refs[:n_in], refs[n_in : n_in + n_out], refs[n_in + n_out :]
  q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref = (kda_fused._view(r) for r in ins[:8])
  outs = [kda_fused._view(r) for r in outs]
  s_ref = scratch[0]

  f32, bf16 = jnp.float32, jnp.bfloat16
  c = k_ref.shape[0]
  dv = v_ref.shape[1]
  row, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  tri = kda_fused._tril_ones(c)

  # Cumulative scalar log-decay in chunk: g_cum [C, 1]
  g_in = g_ref[...].astype(f32)
  g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
  g_diff = jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf)
  exp_g_diff = jnp.exp(g_diff)

  qf = q_ref[...].astype(f32)
  kf = k_ref[...].astype(f32)
  q_bf = qf.astype(bf16)
  k_bf = kf.astype(bf16)
  # Single [2*C, dk] @ [dk, C] MXU matmul for both QK^T and KK^T!
  qk_bf = jnp.concatenate([q_bf, k_bf], axis=0)
  qk_dots = jnp.dot(qk_bf, jnp.transpose(k_bf), preferred_element_type=f32)
  scores = qk_dots[:c, :] * exp_g_diff
  kk = jnp.where(strict, qk_dots[c:, :] * exp_g_diff, 0.0)
  beta = beta_ref[...].astype(f32)
  x = kda_fused._inverse(kk * beta, (row == col).astype(f32), precision)
  scores_bf = scores.astype(bf16)

  out_ref = outs.pop(0)
  st_ref = outs.pop(0) if state_stride else None

  @pl.when(pl.program_id(1) == 0)
  def _():
    s_ref[...] = jnp.zeros_like(s_ref)

  seg_c = segc_ref[...]
  fp = (seg_c == 0).astype(f32)
  eg = jnp.exp(g_cum)
  last_seg = seg_c == seg_c[c - 1 : c, :]
  ecar = jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf))
  vb = (v_ref[...].astype(f32) * beta).astype(bf16)
  kb = (kf * eg * beta * fp).astype(bf16)
  qg = (qf * eg * fp).astype(bf16)
  kc = (kf * ecar).astype(bf16)

  x_bf = x.astype(bf16)
  s = s_ref[...]
  s_bf = s.astype(bf16)
  if state_stride == 1 and st_ref is not None:
    st_ref[...] = s if state_f32 else s_bf

  # v7x stacked matmuls: uw = X @ [vb, kb], [w; qg] @ S
  vb_kb = jnp.concatenate([vb, kb], axis=1)
  uw = jnp.dot(x_bf, vb_kb, preferred_element_type=f32)
  u, w = uw[:, :dv], uw[:, dv:]
  wqg = jnp.concatenate([w.astype(bf16), qg], axis=0)
  ws_qgs = jnp.dot(wqg, s_bf, preferred_element_type=f32)
  ws, qgs = ws_qgs[:c, :], ws_qgs[c:, :]
  delta_bf = (u - ws).astype(bf16)
  if save_wy:
    outs.pop(0)[...] = delta_bf
    xk_out, sc_out = outs
    xk_out[...] = kda_fused._pack_x_kk(x, kk, row, col)
    sc_out[...] = scores_bf
  out = qgs + jnp.dot(scores_bf, delta_bf, preferred_element_type=f32)
  out_ref[...] = out.astype(out_ref.dtype)
  new = s * dl_ref[...] + kda_fused._tn(kc, delta_bf)
  s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)


def _gdn_bwd_kernel(*refs, precision, state_f32):
  """VMEM-fused token-major Pallas backward kernel for GDN (scalar per-head decay)."""
  ins, outs, scratch = refs[:13], refs[13:19], refs[19:]
  q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref, st_ref, de_ref, xk_ref, sc_ref, do_ref = (
      kda_fused._view(r) for r in ins
  )
  dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref, ddl_ref = (kda_fused._view(r) for r in outs)
  r_ref = scratch[0]

  @pl.when(pl.program_id(1) == 0)
  def _():
    r_ref[...] = jnp.zeros_like(r_ref)

  f32, bf16 = jnp.float32, jnp.bfloat16
  c = k_ref.shape[0]
  dv = v_ref.shape[1]
  row, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  tri = kda_fused._tril_ones(c)

  g_in = g_ref[...].astype(f32)
  g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
  g_diff = jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf)
  exp_g_diff = jnp.exp(g_diff)

  qf = q_ref[...].astype(f32)
  kf = k_ref[...].astype(f32)
  seg_c = segc_ref[...]
  fp = (seg_c == 0).astype(f32)
  eg = jnp.exp(g_cum)
  last_seg = seg_c == seg_c[c - 1 : c, :]
  ecar = jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf))
  beta = beta_ref[...].astype(f32)
  vb = (v_ref[...].astype(f32) * beta).astype(bf16)
  kb = (kf * eg * beta * fp).astype(bf16)
  qg = (qf * eg * fp).astype(bf16)
  kc = (kf * ecar).astype(bf16)

  x, kk = kda_fused._unpack_x_kk(xk_ref[...], row, col)
  x_bf, scores_bf = x.astype(bf16), sc_ref[...]
  states_f, delta = st_ref[...].astype(f32), de_ref[...]
  states = states_f.astype(bf16)
  d_out = do_ref[...].astype(bf16)

  dsc = kda_fused._nt(d_out, delta)
  g_delta = kda_fused._tn(scores_bf, d_out)
  r = r_ref[...]
  r_bf = r.astype(bf16)
  dd = g_delta + jnp.dot(kc, r_bf, preferred_element_type=f32)
  dd_bf = dd.astype(bf16)
  w_bf = jnp.dot(x_bf, kb, preferred_element_type=f32).astype(bf16)
  dkc = kda_fused._nt(delta, r_bf)
  ddl_ref[...] = jnp.sum(states_f * r, axis=-1, keepdims=True)

  do_neg_dd = jnp.concatenate([d_out, -dd_bf], axis=0)
  dqg_dw = kda_fused._nt(do_neg_dd, states)
  dqg = dqg_dw[:c, :]
  dw_bf = dqg_dw[c:, :].astype(bf16)
  qg_w = jnp.concatenate([qg, w_bf], axis=0)
  r_ref[...] = r * dl_ref[...] + kda_fused._tn(qg_w, do_neg_dd)

  dd_dw = jnp.concatenate([dd_bf, dw_bf], axis=1)
  vb_kb = jnp.concatenate([vb, kb], axis=1)
  dx = kda_fused._nt(dd_dw, vb_kb)
  dvb_dkb = kda_fused._tn(x_bf, dd_dw)
  dvb, dkb = dvb_dkb[:, :dv], dvb_dkb[:, dv:]

  dm = -kda_fused._mm(
      kda_fused._mm(x, dx, precision, ((0,), (0,))), x, precision, ((1,), (1,))
  )
  dm = jnp.where(strict, dm, 0.0)
  dbeta = jnp.sum(dm * kk, axis=1, keepdims=True)
  dkk = dm * beta
  ds = jnp.where(causal, dsc, 0.0)

  # Scalar-decay intra-chunk backward: ZERO sub-block loops!
  ds_scaled = (ds * exp_g_diff).astype(bf16)
  dkk_scaled = (dkk * exp_g_diff).astype(bf16)
  q_bf = qf.astype(bf16)
  k_bf = kf.astype(bf16)
  dq_i = jnp.dot(ds_scaled, k_bf, preferred_element_type=f32)
  dk_row = jnp.dot(dkk_scaled, k_bf, preferred_element_type=f32)
  ds_dkk_t = jnp.concatenate([jnp.transpose(ds_scaled), jnp.transpose(dkk_scaled)], axis=1)
  qk_bf = jnp.concatenate([q_bf, k_bf], axis=0)
  dk_col = jnp.dot(ds_dkk_t, qk_bf, preferred_element_type=f32)
  dk_i = dk_row + dk_col
  # Closed-form gauge identity for scalar cumulative decay gradient:
  dg_i = jnp.sum(qf * dq_i + kf * (dk_row - dk_col), axis=-1, keepdims=True)

  kegfp = kf * eg * fp
  dv_ref[...] = (dvb * beta).astype(dv_ref.dtype)
  dbeta_ref[...] = (
      dbeta
      + jnp.sum(dvb * v_ref[...].astype(f32), axis=1, keepdims=True)
      + jnp.sum(dkb * kegfp, axis=1, keepdims=True)
  )
  dq_ref[...] = (dq_i + dqg * eg * fp).astype(dq_ref.dtype)
  dk_ref[...] = (dk_i + dkb * eg * beta * fp + dkc * ecar).astype(dk_ref.dtype)
  carry = dkc * kf * ecar
  dg_wy = jnp.sum(dqg * qf * eg * fp + dkb * beta * kegfp - carry, axis=-1, keepdims=True)
  dg_last = jnp.sum(carry, axis=(0, 1), keepdims=True)
  dg = dg_i + dg_wy + jnp.where(row == c - 1, dg_last, 0.0)
  dg_ref[...] = jnp.sum(jnp.transpose(tri) * jnp.transpose(dg), axis=1, keepdims=True)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def gdn_pallas_fused_token_major(
    q, k, v, g_scalar, beta, seg, decay_last, precision, state_f32: bool = True, out_dtype=jnp.bfloat16
):
  """VMEM-fused token-major Pallas GDN kernel with custom VJP."""
  return _gdn_pallas_fused_tm_fwd(q, k, v, g_scalar, beta, seg, decay_last, precision, state_f32, out_dtype)[0]


def _gdn_pallas_fused_tm_fwd(
    q, k, v, g_scalar, beta, seg, decay_last, precision, state_f32: bool = True, out_dtype=jnp.bfloat16
):
  b, t, h, dk = q.shape
  dv = v.shape[-1]
  n = decay_last.shape[2]
  c = t // n
  bh = b * h
  f32, bf16 = jnp.float32, jnp.bfloat16
  s_dtype = f32 if state_f32 else bf16
  q_tm = q.reshape(b, t, h * dk)
  k_tm = k.reshape(b, t, h * dk)
  v_tm = v.reshape(b, t, h * dv)
  g_col = g_scalar.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
  beta_col = beta.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
  seg_b = seg[:, 0]  # [b, n, c]
  seg_col = seg_b[..., None]
  seg_row = seg_b[:, :, None, :]
  dl = decay_last.reshape(bh, n, dk, 1).astype(f32)
  per, seg_sp = kda_fused._specs(n, h, reverse=False)
  tok = kda_fused._tok_spec(n, c, h, reverse=False)
  kernel = functools.partial(
      _gdn_fwd_kernel, precision=precision, save_wy=True, state_stride=1, state_f32=state_f32
  )
  out_tm, states, delta, xk, scores = pl.pallas_call(
      kernel,
      grid=(bh, n),
      in_specs=[
          tok(dk),
          tok(dk),
          tok(dv),
          per(c, 1),
          per(c, 1),
          seg_sp(c, 1),
          seg_sp(1, c),
          per(dk, 1),
      ],
      out_specs=[
          tok(dv),
          per(dk, dv),
          per(c, dv),
          per(c, c),
          per(c, c),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((b, t, h * dv), out_dtype),
          jax.ShapeDtypeStruct((bh, n, dk, dv), s_dtype),
          jax.ShapeDtypeStruct((bh, n, c, dv), bf16),
          jax.ShapeDtypeStruct((bh, n, c, c), f32),
          jax.ShapeDtypeStruct((bh, n, c, c), bf16),
      ],
      scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
  )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl)
  out = out_tm.reshape(b, t, h, dv)
  res = (q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, h, dk, dv)
  return out, res


def _gdn_pallas_fused_tm_bwd(precision, state_f32, out_dtype, res, do):
  del out_dtype
  q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, h, dk, dv = res
  b, t, _ = q_tm.shape
  n = dl.shape[1]
  c = t // n
  bh = b * h
  f32 = jnp.float32
  do_tm = do.reshape(b, t, h * dv)
  per, seg_sp = kda_fused._specs(n, h, reverse=True)
  tok = kda_fused._tok_spec(n, c, h, reverse=True)
  dq_tm, dk_tm, dv_tm, dg_col, dbeta_col, ddl = pl.pallas_call(
      functools.partial(_gdn_bwd_kernel, precision=precision, state_f32=state_f32),
      grid=(bh, n),
      in_specs=[
          tok(dk),
          tok(dk),
          tok(dv),
          per(c, 1),
          per(c, 1),
          seg_sp(c, 1),
          seg_sp(1, c),
          per(dk, 1),
          per(dk, dv),
          per(c, dv),
          per(c, c),
          per(c, c),
          tok(dv),
      ],
      out_specs=[tok(dk), tok(dk), tok(dv), per(c, 1), per(c, 1), per(dk, 1)],
      out_shape=[
          jax.ShapeDtypeStruct((b, t, h * dk), q_tm.dtype),
          jax.ShapeDtypeStruct((b, t, h * dk), k_tm.dtype),
          jax.ShapeDtypeStruct((b, t, h * dv), v_tm.dtype),
          jax.ShapeDtypeStruct((bh, n, c, 1), f32),
          jax.ShapeDtypeStruct((bh, n, c, 1), f32),
          jax.ShapeDtypeStruct((bh, n, dk, 1), f32),
      ],
      scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
  )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, do_tm)
  dq = dq_tm.reshape(b, t, h, dk)
  dk_o = dk_tm.reshape(b, t, h, dk)
  dv_o = dv_tm.reshape(b, t, h, dv)
  dg = dg_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h, 1)
  dbeta = dbeta_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h)
  zero_seg = jnp.zeros((b, 1, n, c), jax.dtypes.float0)
  return dq, dk_o, dv_o, dg, dbeta, zero_seg, ddl.reshape(b, h, n, dk)


gdn_pallas_fused_token_major.defvjp(_gdn_pallas_fused_tm_fwd, _gdn_pallas_fused_tm_bwd)


def run_gdn_pallas_fused(
    q, k, v, log_decay_scalar, beta, resets, chunk_size: int = 128, state_dtype: str = "float32"
):
  """Top-level wrapper for VMEM-fused token-major Pallas GDN kernel."""
  b, t, h, dk = q.shape
  c = chunk_size
  n = t // c
  f32 = jnp.float32
  log_decay_scalar = log_decay_scalar.astype(f32)
  beta = beta.astype(f32)
  seg = jnp.cumsum(resets.reshape(b, n, c).astype(jnp.int32), axis=-1)[:, None]
  any_reset = seg[..., -1] > 0
  g_sum = jnp.sum(log_decay_scalar.reshape(b, n, c, h).transpose(0, 3, 1, 2), axis=-1)
  decay_last = jnp.broadcast_to(jnp.where(any_reset, 0.0, jnp.exp(g_sum))[..., None], (b, h, n, dk))
  return gdn_pallas_fused_token_major(
      q,
      k,
      v,
      log_decay_scalar[..., None],
      beta,
      seg,
      decay_last,
      jax.lax.Precision.HIGHEST,
      state_dtype == "float32",
      jnp.bfloat16,
  )


# ---------------------------------------------------------------------------
# 3. Tile-Grouped / Sub-Head Decay Co-Design Kernel (G_decay groups per head)
# ---------------------------------------------------------------------------


def _kda_tile_grouped_decay_core(
    q, k, v, log_decay_grouped, beta, resets, chunk_size: int = 128, num_groups: int = 2
):
  """Co-design kernel: partitions dk into num_groups scalar-decay sub-groups so each factors outside MXU dot!"""
  b, t, h, dk = q.shape
  dv = v.shape[-1]
  c = chunk_size
  n = t // c
  g_dim = dk // num_groups
  f32, bf16 = jnp.float32, jnp.bfloat16

  def to_c(x):
    return x.reshape(b, n, c, h, -1).transpose(0, 3, 1, 2, 4)

  q_c = to_c(q.astype(bf16)).reshape(b, h, n, c, num_groups, g_dim)
  k_c = to_c(k.astype(bf16)).reshape(b, h, n, c, num_groups, g_dim)
  v_c = to_c(v.astype(bf16))
  tri_bf16 = jnp.tril(jnp.ones((c, c), bf16))
  # log_decay_grouped has shape [B, T, H, num_groups]
  g = jnp.einsum(
      "ij,...jg->...ig",
      tri_bf16,
      to_c(log_decay_grouped).astype(f32),
      preferred_element_type=f32,
      precision=jax.lax.Precision.HIGHEST,
  )
  beta_c = beta.reshape(b, n, c, h).transpose(0, 3, 1, 2).astype(f32)[..., None]
  seg = jnp.cumsum(resets.reshape(b, n, c).astype(jnp.int32), axis=-1)[:, None]
  same_doc = seg[..., :, None] == seg[..., None, :]
  causal = jnp.tril(jnp.ones((c, c), dtype=bool))
  strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
  from_prev = (seg == 0)[..., None].astype(f32)
  last_seg = (seg == seg[..., -1:])[..., None]
  any_reset = seg[..., -1] > 0

  # Each group g_idx in 0..num_groups-1 is a pure MXU dot [2*C, g_dim] @ [g_dim, C] scaled by scalar exp(g_diff)!
  qk_cat = jnp.concatenate([q_c, k_c], axis=-3)  # [B, H, N, 2*C, G, g_dim]
  qk_dots = jnp.einsum("...igd,...jgd->...gij", qk_cat, k_c, preferred_element_type=f32)
  g_diff = jnp.where(
      (causal & same_doc)[..., None, :, :],
      jnp.swapaxes(g, -1, -2)[..., :, None] - jnp.swapaxes(g, -1, -2)[..., None, :],
      -jnp.inf,
  )
  exp_g_diff = jnp.exp(g_diff)
  scores = jnp.sum(qk_dots[..., :c, :] * exp_g_diff, axis=-3)
  kk = jnp.sum(qk_dots[..., c:, :] * exp_g_diff, axis=-3)
  m = jnp.where(strict & same_doc, kk, 0.0) * beta_c

  g_full = jnp.repeat(g, g_dim, axis=-1)
  eg = jnp.exp(g_full)
  k_flat = k_c.reshape(b, h, n, c, dk)
  q_flat = q_c.reshape(b, h, n, c, dk)
  rhs = jnp.concatenate([v_c.astype(f32) * beta_c, k_flat.astype(f32) * eg * beta_c * from_prev], axis=-1)
  uw = olmoe3._kda_wy_solve(m, rhs, bf16)
  g_last = g_full[..., -1:, :]
  k_carry = (k_flat.astype(f32) * jnp.exp(jnp.where(last_seg, g_last - g_full, -jnp.inf))).astype(bf16)
  decay = jnp.where(any_reset[..., None], 0.0, jnp.exp(g_last[..., 0, :]))
  qg = (q_flat.astype(f32) * eg * from_prev).astype(bf16)
  flat = lambda x_: x_.reshape(b * h, n, *x_.shape[3:])
  st, dlt = kda_scan.kda_state_scan(flat(uw), flat(k_carry), flat(decay)[..., None], False)
  st = st.reshape(b, h, n, dk, dv)
  dlt = dlt.reshape(b, h, n, c, dv)
  out_c = jnp.einsum("...cd,...dv->...cv", qg, st, preferred_element_type=f32) + jnp.einsum(
      "...ij,...jv->...iv", scores.astype(bf16), dlt, preferred_element_type=f32
  )
  return out_c.transpose(0, 2, 3, 1, 4).reshape(b, t, h, dv)


# ---------------------------------------------------------------------------
# 4. Main TPU v7x Benchmark Test Suite (4 Shards)
# ---------------------------------------------------------------------------


class LinearAttentionV7xBenchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.devices = jax.devices()
    self.num_devs = len(self.devices)
    self.dev_kind = self.devices[0].device_kind

  def test_0_roofline_and_kda_core_hill_climb(self):
    """Shard 0: Roofline & <256 Matmul Audit + 17-Step KDA Core Hill Climb + XPlane Overlap on TPU v7x."""
    print(
        f"\n================================================================================"
        f"\nSHARD 0: TPU v7x ({self.num_devs}x {self.dev_kind}) ROOFLINE, <256 MATMUL AUDIT & KDA CORE HILL CLIMB"
        f"\n================================================================================",
        flush=True,
    )

    # 1. Print Analytical & Tile-Quantized Roofline Table + <256 Matmul Breakdown
    for label, is_kda, h, dk, dv, c, v7x_opt in (
        ("KDA_olmoe3_C128_base", True, 8, 256, 512, 128, False),
        ("KDA_olmoe3_C128_v7x_opt", True, 8, 256, 512, 128, True),
        ("KDA_olmo35_small_C64_base", True, 16, 128, 256, 64, False),
        ("KDA_olmo35_small_C64_v7x_opt", True, 16, 128, 256, 64, True),
        ("GDN_olmoe3_C128_v7x_opt", False, 8, 256, 512, 128, True),
        ("GDN_Qwen3Next_C64_v7x_opt", False, 32, 128, 128, 64, True),
    ):
      roof = compute_core_roofline_analytical(
          1, 8192, h, dk, dv, c, is_kda=is_kda, state_f32=True, state_stride=0, outer_block=32, v7x_stacked_opt=v7x_opt
      )
      print(
          f"\n[Roofline Summary ({label}, B=1, T=8192, H={h}, dk={dk}, dv={dv}, C={c})]"
          f"\n  Raw MXU FLOPs     : fwd={roof['fwd_raw_gflops']:7.2f} GF | bwd={roof['bwd_raw_gflops']:7.2f} GF | tot={roof['tot_raw_gflops']:7.2f} GF"
          f"\n  256x256 Tile FLOPs: fwd={roof['fwd_tile_gflops']:7.2f} GF | bwd={roof['bwd_tile_gflops']:7.2f} GF | tot={roof['tot_tile_gflops']:7.2f} GF "
          f"(MXU Tile Eff = {roof['mxu_tile_eff_pct']:.1f}%)"
          f"\n  VPU FLOPs         : tot={roof['tot_vpu_gflops']:7.2f} GF | <256 Shares: K_red<256={roof['kred_lt_256_share_pct']:.1f}%, M<256={roof['m_lt_256_share_pct']:.1f}%"
          f"\n  HBM Bytes         : min={roof['tot_min_mb']:6.1f} MB (AI={roof['ai_min_flops_per_byte']:.1f} F/B) | "
          f"Pallas={roof['tot_pallas_mb']:6.1f} MB (AI={roof['ai_pallas_flops_per_byte']:.1f} F/B)"
          f"\n  v7x Roofline (P7) : HBM_min={roof['t_hbm_min_ms']:.3f} ms | HBM_Pallas={roof['t_hbm_pallas_ms']:.3f} ms | "
          f"MXU_raw={roof['t_mxu_raw_ms']:.3f} ms | MXU_tile={roof['t_mxu_tile_ms']:.3f} ms | VPU={roof['t_vpu_ms']:.3f} ms | "
          f"Comp_tile={roof['t_comp_tile_ms']:.3f} ms",
          flush=True,
      )
      if label in ("KDA_olmoe3_C128_base", "KDA_olmoe3_C128_v7x_opt"):
        print(f"  --- Matmul Breakdown ({label}) ---", flush=True)
        for m_info in roof["matmul_details"]:
          print(
              f"    {m_info['phase']:3s} | {m_info['name']:30s} | {m_info['shape']:34s} | "
              f"K<256={str(m_info['kred_lt_256']):5s} | M<256={str(m_info['m_lt_256']):5s} | "
              f"raw={m_info['raw_gflops']:6.2f} GF | tile={m_info['tile_gflops']:6.2f} GF | eff={m_info['mxu_eff_pct']:5.1f}%",
              flush=True,
          )

    # 2. Direct Micro-Benchmark of <256 Reduction/Row Matmuls vs Stacked =256 Matmuls on TPU v7x
    print("\n--- Direct TPU v7x Micro-Benchmark: K_red < 256 & M < 256 Matmuls vs Stacked 256 Matmuls ---", flush=True)
    key = jax.random.PRNGKey(42)
    bhn = 8 * 64  # B=1, H=8, N=64 chunks (T=8192, C=128)
    for desc, fn_unfused, fn_fused, args_u, args_f, raw_gf in (
        (
            "Fwd State Readout: 2x [128,256]@[256,512] vs 1x [256,256]@[256,512] (M: 128->256)",
            jax.jit(
                lambda w_, qg_, s_: (
                    jnp.einsum("bmk,bkn->bmn", w_, s_, preferred_element_type=jnp.float32),
                    jnp.einsum("bmk,bkn->bmn", qg_, s_, preferred_element_type=jnp.float32),
                )
            ),
            jax.jit(lambda wqg_, s_: jnp.einsum("bmk,bkn->bmn", wqg_, s_, preferred_element_type=jnp.float32)),
            (
                jax.random.normal(key, (bhn, 128, 256), jnp.bfloat16),
                jax.random.normal(key, (bhn, 128, 256), jnp.bfloat16),
                jax.random.normal(key, (bhn, 256, 512), jnp.bfloat16),
            ),
            (
                jax.random.normal(key, (bhn, 256, 256), jnp.bfloat16),
                jax.random.normal(key, (bhn, 256, 512), jnp.bfloat16),
            ),
            bhn * 2 * 256 * 256 * 512 / 1e9,
        ),
        (
            "Bwd State Adjoint: 2x [256,128]@[128,512] vs 1x [256,256]@[256,512] (K_red: 128->256)",
            jax.jit(
                lambda qg_, do_, w_, dd_: jnp.einsum("bck,bcv->bkv", qg_, do_, preferred_element_type=jnp.float32)
                - jnp.einsum("bck,bcv->bkv", w_, dd_, preferred_element_type=jnp.float32)
            ),
            jax.jit(
                lambda qgw_, dodd_: jnp.einsum("bck,bcv->bkv", qgw_, dodd_, preferred_element_type=jnp.float32)
            ),
            (
                jax.random.normal(key, (bhn, 128, 256), jnp.bfloat16),
                jax.random.normal(key, (bhn, 128, 512), jnp.bfloat16),
                jax.random.normal(key, (bhn, 128, 256), jnp.bfloat16),
                jax.random.normal(key, (bhn, 128, 512), jnp.bfloat16),
            ),
            (
                jax.random.normal(key, (bhn, 256, 256), jnp.bfloat16),
                jax.random.normal(key, (bhn, 256, 512), jnp.bfloat16),
            ),
            bhn * 2 * 256 * 256 * 512 / 1e9,
        ),
        (
            "C=64 vs C=128 vs C=256 Contraction: [256, C]@[C, 512] at fixed T=8192 (K_red=C)",
            jax.jit(lambda a_, b_: jnp.einsum("bck,bcv->bkv", a_, b_, preferred_element_type=jnp.float32)),
            jax.jit(lambda a_, b_: jnp.einsum("bck,bcv->bkv", a_, b_, preferred_element_type=jnp.float32)),
            (
                jax.random.normal(key, (8 * 128, 64, 256), jnp.bfloat16),
                jax.random.normal(key, (8 * 128, 64, 512), jnp.bfloat16),
            ),
            (
                jax.random.normal(key, (8 * 32, 256, 256), jnp.bfloat16),
                jax.random.normal(key, (8 * 32, 256, 512), jnp.bfloat16),
            ),
            8 * 8192 * 2 * 256 * 512 / 1e9,
        ),
    ):
      t_u, _ = _time_fn(fn_unfused, *args_u)
      t_f, _ = _time_fn(fn_fused, *args_f)
      tf_u = (raw_gf / 1e3) / (t_u / 1e3)
      tf_f = (raw_gf / 1e3) / (t_f / 1e3)
      print(
          f"  {desc}: unfused={t_u:.3f} ms ({tf_u:.1f} TF/s, {100*tf_u/V7X_MXU_BF16_TFLOPS_P7:.1f}% MXU) -> "
          f"fused={t_f:.3f} ms ({tf_f:.1f} TF/s, {100*tf_f/V7X_MXU_BF16_TFLOPS_P7:.1f}% MXU, speedup={t_u/t_f:.2f}x)",
          flush=True,
      )

    # 3. 17-Step KDA Core Hill Climb on TPU v7x (T=4096 and T=8192)
    for seq_len in (4096, 8192):
      print(f"\n=== KDA Core Hill Climb on TPU v7x (B=1, T={seq_len}, H=8, dk=256, dv=512, C=128) ===", flush=True)
      b, h, dk, dv, c = 1, 8, 256, 512, 128
      keys = jax.random.split(jax.random.PRNGKey(0), 6)
      q = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, h, dk), jnp.float32)) * (dk**-0.5)
      k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, h, dk), jnp.float32))
      v = jax.random.normal(keys[2], (b, seq_len, h, dv), jnp.float32) * 0.1
      log_decay = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, dk), jnp.float32) - 2.0)
      beta = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, h), jnp.float32))
      resets = jnp.zeros((b, seq_len), dtype=bool)
      dy = jax.random.normal(keys[5], (b, seq_len, h, dv), jnp.float32) * 0.1

      roof_base = compute_core_roofline_analytical(
          b, seq_len, h, dk, dv, c, is_kda=True, state_f32=True, state_stride=1, outer_block=32, v7x_stacked_opt=False
      )
      roof_s0 = compute_core_roofline_analytical(
          b, seq_len, h, dk, dv, c, is_kda=True, state_f32=True, state_stride=0, outer_block=32, v7x_stacked_opt=False
      )
      roof_v7x = compute_core_roofline_analytical(
          b, seq_len, h, dk, dv, c, is_kda=True, state_f32=True, state_stride=0, outer_block=32, v7x_stacked_opt=True
      )

      def _pallas_state_scan_1d(uw, kc, dl):
        b_, h_, n_ = uw.shape[:3]
        flat = lambda t_: t_.reshape(b_ * h_, n_, *t_.shape[3:])
        st, dlt = kda_scan.kda_state_scan(flat(uw), flat(kc), flat(dl)[..., None], False)
        return st.reshape(b_, h_, n_, *st.shape[2:]), dlt.reshape(b_, h_, n_, *dlt.shape[2:])

      def _pallas_intra_1d(q_c, k_c, g_c):
        b_, h_, n_ = q_c.shape[:3]
        flat = lambda t_: t_.reshape(b_ * h_, n_, *t_.shape[3:])
        sc, kk_ = olmoe3._kda_intra_chunk_pallas(flat(q_c), flat(k_c), flat(g_c), 16, jnp.float32, False, False)
        return sc.reshape(b_, h_, n_, *sc.shape[2:]), kk_.reshape(b_, h_, n_, *kk_.shape[2:])

      arms = [
          (
              "01_xla_pairwise_scan_f32",
              lambda q_, k_, v_, g_, b_: olmoe3._delta_rule_chunked(q_, k_, v_, g_, b_, resets, c, "float32"),
              roof_base,
              False,
          ),
          (
              "02_xla_subblock16_scan_no_vjp",
              lambda q_, k_, v_, g_, b_: olmoe3._delta_rule_chunked_subblock(
                  q_, k_, v_, g_, b_, resets, c, "float32", sub_block=16, intra_vjp=False
              ),
              roof_base,
              False,
          ),
          (
              "03_xla_subblock16_gauge_vjp",
              lambda q_, k_, v_, g_, b_: olmoe3._delta_rule_chunked_subblock(
                  q_, k_, v_, g_, b_, resets, c, "float32", sub_block=16, intra_vjp=True
              ),
              roof_base,
              False,
          ),
          (
              "04_xla_subblock16_pallas_scan",
              lambda q_, k_, v_, g_, b_: olmoe3._delta_rule_chunked_subblock(
                  q_, k_, v_, g_, b_, resets, c, "float32", sub_block=16, intra_vjp=True, state_scan=_pallas_state_scan_1d
              ),
              roof_base,
              False,
          ),
          (
              "05_pallas_intra_pallas_scan",
              lambda q_, k_, v_, g_, b_: olmoe3._delta_rule_chunked_subblock(
                  q_, k_, v_, g_, b_, resets, c, "float32", sub_block=16, state_scan=_pallas_state_scan_1d, intra_fn=_pallas_intra_1d
              ),
              roof_base,
              False,
          ),
          (
              "06_pallas_fused_chunk_major_f32",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="chunk_major",
                  outer_block=32,
              ),
              roof_base,
              False,
          ),
          (
              "07_pallas_fused_tm_1level_f32",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=None,
              ),
              roof_base,
              False,
          ),
          (
              "08_pallas_fused_tm_ob32_out_f32",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
              ),
              roof_base,
              False,
          ),
          (
              "09_pallas_fused_tm_ob32_bf16_s1",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=1,
              ),
              roof_base,
              False,
          ),
          (
              "10_pallas_fused_tm_ob64_bf16_s1",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=64,
                  output_dtype=jnp.bfloat16,
                  state_stride=1,
              ),
              roof_base,
              False,
          ),
          (
              "11_pallas_fused_tm_ob32_bf16_s2",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=2,
              ),
              roof_base,
              False,
          ),
          (
              "12_pallas_fused_tm_ob32_bf16_s0",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=0,
              ),
              roof_s0,
              False,
          ),
          (
              "13_pallas_fused_tm_inv_high_s0",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=0,
                  inverse_precision=jax.lax.Precision.HIGH,
              ),
              roof_s0,
              False,
          ),
          (
              "14_pallas_fused_tm_state_bf16_s1",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.bfloat16,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=1,
              ),
              roof_base,
              False,
          ),
          (
              "15_pallas_fused_v7x_opt_s1",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=1,
              ),
              roof_v7x,
              True,
          ),
          (
              "16_pallas_fused_v7x_opt_s0",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=0,
              ),
              roof_v7x,
              True,
          ),
          (
              "17_pallas_fused_v7x_opt_bf16_v_s0",
              lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                  q_,
                  k_,
                  v_.astype(jnp.bfloat16),
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=0,
              ),
              roof_v7x,
              True,
          ),
      ]

      ref_out, ref_grads = None, None
      orig_fwd, orig_bwd = kda_fused._fwd_kernel, kda_fused._bwd_kernel
      for name, fn, roof, use_v7x_patch in arms:
        try:
          if use_v7x_patch:
            kda_fused._fwd_kernel = _fwd_kernel_v7x_opt
            kda_fused._bwd_kernel = _bwd_kernel_v7x_opt
          else:
            kda_fused._fwd_kernel = orig_fwd
            kda_fused._bwd_kernel = orig_bwd

          j_fwd = jax.jit(fn)
          j_fb = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy),
                  argnums=(0, 1, 2, 3, 4),
              )
          )
          t_f, out_f = _time_fn(j_fwd, q, k, v, log_decay, beta)
          t_fb, (_, grads), hlo = _time_and_profile_hlo(j_fb, q, k, v, log_decay, beta)
          if ref_out is None:
            ref_out, ref_grads = out_f.astype(jnp.float32), grads
          out_diff = float(jnp.max(jnp.abs(ref_out - out_f.astype(jnp.float32))))
          g_diff = max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b_.astype(jnp.float32)))) for a, b_ in zip(ref_grads, grads))
          print(format_perf_vs_roofline(name, t_f, t_fb, roof, hlo, max(out_diff, g_diff)), flush=True)
          if seq_len == 8192 and name in (
              "03_xla_subblock16_gauge_vjp",
              "09_pallas_fused_tm_ob32_bf16_s1",
              "12_pallas_fused_tm_ob32_bf16_s0",
              "15_pallas_fused_v7x_opt_s1",
              "16_pallas_fused_v7x_opt_s0",
              "17_pallas_fused_v7x_opt_bf16_v_s0",
          ):
            xp = _profile_xplane(j_fb, q, k, v, log_decay, beta, top_k=6)
            print(f"    [XPlane {name}] {xp}", flush=True)
        finally:
          kda_fused._fwd_kernel = orig_fwd
          kda_fused._bwd_kernel = orig_bwd

  def test_1_gdn_core_and_layer_hill_climb(self):
    """Shard 1: GDN Core & Full Layer Hill Climb (Qwen3-Next & OLMo 3.5 Shapes) + Tokamax & VMEM-Fused Pallas GDN."""
    print(
        f"\n================================================================================"
        f"\nSHARD 1: TPU v7x ({self.num_devs}x {self.dev_kind}) GDN CORE & FULL LAYER HILL CLIMB"
        f"\n================================================================================",
        flush=True,
    )

    for shape_label, seq_len, h, dk, dv, c in (
        ("Qwen3Next_T4096_H32_d128_C64", 4096, 32, 128, 128, 64),
        ("Qwen3Next_T8192_H32_d128_C64", 8192, 32, 128, 128, 64),
        ("OLMo35_T4096_H8_dk256_dv512_C128", 4096, 8, 256, 512, 128),
        ("OLMo35_T8192_H8_dk256_dv512_C128", 8192, 8, 256, 512, 128),
    ):
      print(f"\n=== GDN Core Hill Climb: {shape_label} (B=1, T={seq_len}, H={h}, dk={dk}, dv={dv}, C={c}) ===", flush=True)
      b = 1
      keys = jax.random.split(jax.random.PRNGKey(1), 6)
      q = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16)
      k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16)
      v = (jax.random.normal(keys[2], (b, seq_len, h, dv), jnp.bfloat16) * 0.1).astype(jnp.bfloat16)
      g_scalar = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h), jnp.float32) - 2.0)
      beta = jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, h), jnp.bfloat16)).astype(jnp.bfloat16)
      resets = jnp.zeros((b, seq_len), dtype=bool)
      dy = jax.random.normal(keys[5], (b, seq_len, h, dv), jnp.float32) * 0.1

      roof_gdn = compute_core_roofline_analytical(
          b, seq_len, h, dk, dv, c, is_kda=False, state_f32=True, state_stride=1, outer_block=None, v7x_stacked_opt=True
      )

      def _gdn_xla_pallas_scan(q_, k_, v_, g_, b_):
        n = seq_len // c
        f32, bf16 = jnp.float32, jnp.bfloat16
        to_c = lambda t_: t_.reshape(b, n, c, h, -1).transpose(0, 3, 1, 2, 4)
        q_c, k_c, v_c = (to_c(t_.astype(bf16)) for t_ in (q_, k_, v_))
        g_in = g_.reshape(b, n, c, h).transpose(0, 3, 1, 2).astype(f32)
        tri = jnp.tril(jnp.ones((c, c), bf16))
        g_cum = jnp.einsum("ij,...j->...i", tri, g_in, preferred_element_type=f32, precision=jax.lax.Precision.HIGHEST)
        beta_c = b_.reshape(b, n, c, h).transpose(0, 3, 1, 2).astype(f32)[..., None]
        causal = jnp.tril(jnp.ones((c, c), dtype=bool))
        strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
        qk_cat = jnp.concatenate([q_c, k_c], axis=-2)
        qk_dots = jnp.einsum("...id,...jd->...ij", qk_cat, k_c, preferred_element_type=f32)
        exp_g_diff = jnp.exp(jnp.where(causal, g_cum[..., :, None] - g_cum[..., None, :], -jnp.inf))
        scores = qk_dots[..., :c, :] * exp_g_diff * (dk**-0.5)
        m = jnp.where(strict, qk_dots[..., c:, :] * exp_g_diff, 0.0) * beta_c
        eg = jnp.exp(g_cum)[..., None]
        rhs = jnp.concatenate([v_c.astype(f32) * beta_c, k_c.astype(f32) * eg * beta_c], axis=-1)
        uw = olmoe3._kda_wy_solve(m, rhs, bf16)
        g_last = g_cum[..., -1:]
        k_carry = (k_c.astype(f32) * jnp.exp((g_last - g_cum)[..., None])).astype(bf16)
        decay = jnp.broadcast_to(jnp.exp(g_last), (b, h, n, dk))
        qg = (q_c.astype(f32) * (dk**-0.5) * eg).astype(bf16)
        flat = lambda t_: t_.reshape(b * h, n, *t_.shape[3:])
        st, dlt = kda_scan.kda_state_scan(flat(uw), flat(k_carry), flat(decay)[..., None], False)
        st = st.reshape(b, h, n, dk, dv)
        dlt = dlt.reshape(b, h, n, c, dv)
        out_c = jnp.einsum("...cd,...dv->...cv", qg, st, preferred_element_type=f32) + jnp.einsum(
            "...ij,...jv->...iv", scores.astype(bf16), dlt, preferred_element_type=f32
        )
        return out_c.transpose(0, 2, 3, 1, 4).reshape(b, seq_len, h, dv)

      gdn_arms = [
          (
              "01_qwen3_jax_baseline_solve_tri",
              lambda q_, k_, v_, g_, b_: qwen3.jax_chunk_gated_delta_rule_baseline(
                  q_, k_, v_, g_, b_, chunk_size=c, compute_dtype=jnp.bfloat16
              )[0],
          ),
          (
              "02_qwen3_jax_wy_fast_custom_vjp",
              lambda q_, k_, v_, g_, b_: qwen3.jax_chunk_gated_delta_rule(
                  q_, k_, v_, g_, b_, chunk_size=c, compute_dtype=jnp.bfloat16
              )[0],
          ),
          (
              "03_gdn_xla_intra_pallas_scan",
              _gdn_xla_pallas_scan,
          ),
          (
              "04_gdn_pallas_fused_v7x_opt_f32",
              lambda q_, k_, v_, g_, b_: run_gdn_pallas_fused(q_ * (dk**-0.5), k_, v_, g_, b_, resets, c, "float32"),
          ),
          (
              "05_gdn_pallas_fused_v7x_opt_bf16",
              lambda q_, k_, v_, g_, b_: run_gdn_pallas_fused(q_ * (dk**-0.5), k_, v_, g_, b_, resets, c, "bfloat16"),
          ),
      ]

      ref_out = None
      for name, fn in gdn_arms:
        j_fwd = jax.jit(fn)
        j_fb = jax.jit(
            jax.value_and_grad(
                lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy),
                argnums=(0, 1, 2, 3, 4),
            )
        )
        t_f, out_f = _time_fn(j_fwd, q, k, v, g_scalar, beta)
        t_fb, _, hlo = _time_and_profile_hlo(j_fb, q, k, v, g_scalar, beta)
        if ref_out is None:
          ref_out = out_f.astype(jnp.float32)
        diff = float(jnp.max(jnp.abs(ref_out - out_f.astype(jnp.float32))))
        print(format_perf_vs_roofline(name, t_f, t_fb, roof_gdn, hlo, diff), flush=True)
        if seq_len == 8192 and name in ("02_qwen3_jax_wy_fast_custom_vjp", "04_gdn_pallas_fused_v7x_opt_f32"):
          xp = _profile_xplane(j_fb, q, k, v, g_scalar, beta, top_k=5)
          print(f"    [XPlane {name}] {xp}", flush=True)

    # Full Qwen3NextGatedDeltaNet Layer Benchmark on TPU v7x
    print("\n=== Full Qwen3NextGatedDeltaNet Layer Benchmark on TPU v7x (B=1/core, T=4096 & 8192) ===", flush=True)
    for seq_len in (4096, 8192):
      gdn_cfg_args = [
          "",
          os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"),
          "skip_jax_distributed_system=True",
          "run_name=gdn_v7x_bench",
          "enable_checkpointing=False",
          "per_device_batch_size=1.0",
          f"max_target_length={seq_len}",
          f"ici_data_parallelism={self.num_devs}",
          "base_emb_dim=2048",
          "gdn_num_key_heads=16",
          "gdn_num_value_heads=32",
          "gdn_key_head_dim=128",
          "gdn_value_head_dim=128",
          "gdn_conv_kernel_dim=4",
          "gdn_chunk_size=64",
          "dtype=bfloat16",
          "weight_dtype=float32",
          "use_qk_norm_in_gdn=True",
      ]
      gdn_cfg = pyconfig.initialize(gdn_cfg_args)
      devices_array = maxtext_utils.create_device_mesh(gdn_cfg)
      mesh = Mesh(devices_array, gdn_cfg.mesh_axes)

      def _pallas_fused_gdn_rule(
          query,
          key,
          value,
          g,
          beta,
          chunk_size: int = 64,
          initial_state=None,
          use_qk_norm_in_gdn: bool = False,
          cp_axis=None,
          compute_dtype=jnp.bfloat16,
      ):
        del cp_axis
        if use_qk_norm_in_gdn:
          query = qwen3.l2norm(query, dim=-1, eps=1e-6)
          key = qwen3.l2norm(key, dim=-1, eps=1e-6)
        out = run_gdn_pallas_fused(
            query * (query.shape[-1] ** -0.5),
            key,
            value,
            g,
            beta,
            jnp.zeros(query.shape[:2], dtype=bool),
            chunk_size,
            "float32",
        )
        next_st = (
            initial_state.astype(compute_dtype)
            if initial_state is not None
            else jnp.zeros((query.shape[0], query.shape[2], query.shape[3], value.shape[3]), dtype=compute_dtype)
        )
        return out.astype(query.dtype), next_st

      def _baseline_gdn_rule(
          query,
          key,
          value,
          g,
          beta,
          chunk_size: int = 64,
          initial_state=None,
          use_qk_norm_in_gdn: bool = False,
          cp_axis=None,
          compute_dtype=jnp.bfloat16,
      ):
        del cp_axis
        out, st = qwen3.jax_chunk_gated_delta_rule_baseline(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            compute_dtype=compute_dtype,
        )
        if st is None:
          st = jnp.zeros((query.shape[0], query.shape[2], query.shape[3], value.shape[3]), dtype=compute_dtype)
        return out, st

      orig_gdn_rule = qwen3.jax_chunk_gated_delta_rule
      try:
        with mesh, nn_partitioning.axis_rules(gdn_cfg.logical_axis_rules):
          gdn_layer = qwen3.Qwen3NextGatedDeltaNet(config=gdn_cfg, mesh=mesh, rngs=nnx.Rngs(params=0))
          graphdef, state = nnx.split(gdn_layer)
          x_sharding = NamedSharding(mesh, P("data", None, None))
          x_in = jax.device_put(
              jax.random.normal(jax.random.PRNGKey(7), (self.num_devs, seq_len, 2048), jnp.bfloat16),
              x_sharding,
          )
          for layer_arm, rule_override in (
              ("Qwen3NextLayer_01_jax_baseline", _baseline_gdn_rule),
              ("Qwen3NextLayer_02_jax_wy_fast", orig_gdn_rule),
              ("Qwen3NextLayer_03_pallas_fused_v7x", _pallas_fused_gdn_rule),
          ):
            if seq_len == 4096 and layer_arm != "Qwen3NextLayer_03_pallas_fused_v7x":
              continue
            qwen3.jax_chunk_gated_delta_rule = rule_override
            j_fwd = jax.jit(lambda st, x_: nnx.merge(graphdef, st)(x_)[0])
            j_fb = jax.jit(
                jax.value_and_grad(
                    lambda st, x_: jnp.mean(nnx.merge(graphdef, st)(x_)[0].astype(jnp.float32) ** 2),
                    argnums=(0, 1),
                )
            )
            t_f, _ = _time_fn(j_fwd, state, x_in)
            t_fb, _, hlo = _time_and_profile_hlo(j_fb, state, x_in)
            print(
                f"[{layer_arm:36s} T={seq_len}] fwd={t_f:6.3f} ms | bwd={t_fb-t_f:6.3f} ms | "
                f"fwd+bwd={t_fb:6.3f} ms (x36={36*t_fb:6.1f} ms) | HLO={hlo}",
                flush=True,
            )
      finally:
        qwen3.jax_chunk_gated_delta_rule = orig_gdn_rule

  def test_2_full_kda_layer_hill_climb_and_xprof(self):
    """Shard 2: Full OLMoE3KimiDeltaAttention Layer 11-Step Hill Climb + XPlane Breakdown on TPU v7x."""
    print(
        f"\n================================================================================"
        f"\nSHARD 2: TPU v7x ({self.num_devs}x {self.dev_kind}) FULL KDA LAYER HILL CLIMB & XPLANE BREAKDOWN"
        f"\n================================================================================",
        flush=True,
    )

    seq_len = 8192
    for mesh_label, fsdp_p, ep_p in (
        ("EP_unsharded_weights", 1, self.num_devs),
        ("FSDP2_die_to_die_sharded_weights", min(2, self.num_devs), max(1, self.num_devs // 2)),
    ):
      print(
          f"\n=== Full OLMoE3KimiDeltaAttention Layer Hill Climb ({mesh_label}: FSDP={fsdp_p}, EP={ep_p}, T={seq_len}) ===",
          flush=True,
      )
      layer_arms = [
          (
              "01_unfused_proj_xla_subblock",
              dict(
                  kda_fused_input_proj=False,
                  kda_flat_head_layout=False,
                  kda_pallas_scan=False,
                  kda_pallas_intra=False,
                  kda_pallas_fused=False,
                  kda_fused_state_stride=1,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
              ),
              ("decoder_layer_input", "kda_wy"),
              False,
          ),
          (
              "02_fused_proj_xla_subblock",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=False,
                  kda_pallas_scan=False,
                  kda_pallas_intra=False,
                  kda_pallas_fused=False,
                  kda_fused_state_stride=1,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
              ),
              ("decoder_layer_input", "kda_wy"),
              False,
          ),
          (
              "03_fused_proj_pallas_scan",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=False,
                  kda_pallas_scan=True,
                  kda_pallas_intra=False,
                  kda_pallas_fused=False,
                  kda_fused_state_stride=1,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
              ),
              ("decoder_layer_input", "kda_wy"),
              False,
          ),
          (
              "04_fused_proj_pallas_intra_scan",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=False,
                  kda_pallas_scan=True,
                  kda_pallas_intra=True,
                  kda_pallas_fused=False,
                  kda_fused_state_stride=1,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
              ),
              ("decoder_layer_input", "kda_wy"),
              False,
          ),
          (
              "05_pallas_fused_tm_remat_s1",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=False,
                  kda_pallas_fused=True,
                  kda_fused_token_major=True,
                  kda_outer_block=32,
                  kda_fused_state_stride=1,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
              ),
              ("decoder_layer_input", "kda_wy"),
              False,
          ),
          (
              "06_pallas_fused_tm_flat_head_s1",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=True,
                  kda_pallas_fused=True,
                  kda_fused_token_major=True,
                  kda_outer_block=32,
                  kda_fused_state_stride=1,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
              ),
              ("decoder_layer_input", "kda_wy"),
              False,
          ),
          (
              "07_save_kda_fused_s1",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=True,
                  kda_pallas_fused=True,
                  kda_fused_token_major=True,
                  kda_outer_block=32,
                  kda_fused_state_stride=1,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
                  kda_fused="device",
              ),
              ("decoder_layer_input", "kda_wy", "kda_fused"),
              False,
          ),
          (
              "08_save_kda_fused_stride0",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=True,
                  kda_pallas_fused=True,
                  kda_fused_token_major=True,
                  kda_outer_block=32,
                  kda_fused_state_stride=0,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
                  kda_fused="device",
              ),
              ("decoder_layer_input", "kda_wy", "kda_fused"),
              False,
          ),
          (
              "09_save_kda_fused_s0_and_kda_proj",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=True,
                  kda_pallas_fused=True,
                  kda_fused_token_major=True,
                  kda_outer_block=32,
                  kda_fused_state_stride=0,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
                  kda_fused="device",
                  kda_proj="device",
              ),
              ("decoder_layer_input", "kda_wy", "kda_fused", "kda_proj", "gathered_weights"),
              False,
          ),
          (
              "10_v7x_opt_kernel_s0_and_kda_proj",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=True,
                  kda_pallas_fused=True,
                  kda_fused_token_major=True,
                  kda_outer_block=32,
                  kda_fused_state_stride=0,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=False,
                  kda_fused="device",
                  kda_proj="device",
              ),
              ("decoder_layer_input", "kda_wy", "kda_fused", "kda_proj", "gathered_weights"),
              True,
          ),
          (
              "11_v7x_opt_kernel_s0_bf16_conv",
              dict(
                  kda_fused_input_proj=True,
                  kda_flat_head_layout=True,
                  kda_pallas_fused=True,
                  kda_fused_token_major=True,
                  kda_outer_block=32,
                  kda_fused_state_stride=0,
                  gdn_state_dtype="float32",
                  kda_conv_in_compute_dtype=True,
                  kda_fused="device",
                  kda_proj="device",
              ),
              ("decoder_layer_input", "kda_wy", "kda_fused", "kda_proj", "gathered_weights"),
              True,
          ),
      ]

      orig_fwd, orig_bwd = kda_fused._fwd_kernel, kda_fused._bwd_kernel
      for arm_name, overrides, save_names, use_v7x_patch in layer_arms:
        if fsdp_p == 2 and not arm_name.startswith(("01_", "05_", "08_", "09_", "10_", "11_")):
          continue
        try:
          if use_v7x_patch:
            kda_fused._fwd_kernel = _fwd_kernel_v7x_opt
            kda_fused._bwd_kernel = _bwd_kernel_v7x_opt
          else:
            kda_fused._fwd_kernel = orig_fwd
            kda_fused._bwd_kernel = orig_bwd

          cfg_args = [
              "",
              os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"),
              "skip_jax_distributed_system=True",
              "model_name=olmoe3-3p5b",
              "override_model_config=True",
              "per_device_batch_size=1.0",
              f"max_target_length={seq_len}",
              f"ici_expert_parallelism={ep_p}",
              f"ici_fsdp_parallelism={fsdp_p}",
              "allow_split_physical_axes=True",
              "scan_layers=False",
              "remat_policy=custom",
              "decoder_layer_input=device",
              "kda_wy=device",
              "dtype=bfloat16",
              "weight_dtype=float32",
              "kda_chunked_impl=subblock",
              "kda_sub_block=16",
              "gdn_chunk_size=128",
          ] + [f"{k}={v}" for k, v in overrides.items()]
          cfg = pyconfig.initialize(cfg_args)
          devices_array = maxtext_utils.create_device_mesh(cfg)
          mesh = Mesh(devices_array, cfg.mesh_axes)
          with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
            kda_mod = olmoe3.OLMoE3KimiDeltaAttention(cfg, mesh, quant=None, rngs=nnx.Rngs(params=0))
            graphdef, state = nnx.split(kda_mod)
            batch_axis = ("data", "fsdp", "expert")
            x_sharding = NamedSharding(mesh, P(batch_axis, None, None))
            seg_sharding = NamedSharding(mesh, P(batch_axis, None))
            x_in = jax.device_put(
                jax.random.normal(jax.random.PRNGKey(0), (self.num_devs, seq_len, cfg.emb_dim), jnp.bfloat16),
                x_sharding,
            )
            seg_ids = jax.device_put(jnp.ones((self.num_devs, seq_len), jnp.int32), seg_sharding)
            pol = jax.checkpoint_policies.save_only_these_names(*save_names)

            @jax.jit
            def kda_fwd(st, x_, seg_):
              with nn_partitioning.axis_rules(cfg.logical_axis_rules):
                return nnx.merge(graphdef, st)(x_, seg_)

            @jax.jit
            def kda_step(st, x_, seg_, _pol=pol):
              def loss_f(s_, inp_):
                with nn_partitioning.axis_rules(cfg.logical_axis_rules):
                  def run_kda(s_in, i_in):
                    return nnx.merge(graphdef, s_in)(i_in, seg_)
                  out = jax.checkpoint(run_kda, policy=_pol)(s_, inp_)
                  return jnp.mean(out.astype(jnp.float32) ** 2)

              return jax.value_and_grad(loss_f, argnums=(0, 1))(st, x_)

            t_f, _ = _time_fn(kda_fwd, state, x_in, seg_ids)
            t_fb, _, hlo = _time_and_profile_hlo(kda_step, state, x_in, seg_ids)
            print(
                f"[{arm_name:36s}] fwd={t_f:6.3f} ms | bwd={t_fb-t_f:6.3f} ms | "
                f"fwd+bwd={t_fb:6.3f} ms (x24={24*t_fb:6.1f} ms) | temp_MB={hlo['temp_mb']:6.1f} | HLO={hlo}",
                flush=True,
            )
            if arm_name in ("01_unfused_proj_xla_subblock", "06_pallas_fused_tm_flat_head_s1", "09_save_kda_fused_s0_and_kda_proj", "10_v7x_opt_kernel_s0_and_kda_proj"):
              xp = _profile_xplane(kda_step, state, x_in, seg_ids, top_k=8)
              print(f"    [XPlane {arm_name}] {xp}", flush=True)
        finally:
          kda_fused._fwd_kernel = orig_fwd
          kda_fused._bwd_kernel = orig_bwd

  def test_3_linear_attention_codesign_sweep(self):
    """Shard 3: Linear Attention Co-Design Sweep (Head Dim, Chunk Size, Decay Grouping, Gate Proj, Precision)."""
    print(
        f"\n================================================================================"
        f"\nSHARD 3: TPU v7x ({self.num_devs}x {self.dev_kind}) LINEAR ATTENTION CO-DESIGN SWEEP"
        f"\n================================================================================",
        flush=True,
    )
    seq_len = 8192
    b = 1
    keys = jax.random.split(jax.random.PRNGKey(3), 8)
    resets = jnp.zeros((b, seq_len), dtype=bool)

    orig_fwd, orig_bwd = kda_fused._fwd_kernel, kda_fused._bwd_kernel
    kda_fused._fwd_kernel = _fwd_kernel_v7x_opt
    kda_fused._bwd_kernel = _bwd_kernel_v7x_opt
    try:
      # Sweep 1: Head Dimension & Head Count Co-Design (MXU 256x256 Alignment)
      print("\n=== Co-Design Sweep 1: Head Dimension & Head Count Alignment (T=8192, C=128) ===", flush=True)
      for cfg_name, h, dk, dv, c in (
          ("H16_dk128_dv128_State32MB (Qwen-style)", 16, 128, 128, 128),
          ("H8_dk256_dv256_State64MB  (256x256 MXU-aligned, iso-proj)", 8, 256, 256, 128),
          ("H4_dk256_dv256_State32MB  (256x256 MXU-aligned, iso-state)", 4, 256, 256, 128),
          ("H16_dk128_dv256_State64MB (olmo35-small official)", 16, 128, 256, 128),
          ("H8_dk256_dv512_State128MB (olmoe3-3p5b 256x512 MXU-aligned)", 8, 256, 512, 128),
      ):
        q = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, h, dk), jnp.float32)) * (dk**-0.5)
        k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, h, dk), jnp.float32))
        v = jax.random.normal(keys[2], (b, seq_len, h, dv), jnp.float32) * 0.1
        g_kda = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, dk), jnp.float32) - 2.0)
        g_gdn = jnp.mean(g_kda, axis=-1)
        beta = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, h), jnp.float32))
        dy = jax.random.normal(keys[5], (b, seq_len, h, dv), jnp.float32) * 0.1

        for rule_tag, is_kda, fn in (
            (
                "KDA",
                True,
                lambda q_, k_, v_, g_, b_, _c=c: kda.chunk_kda(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    resets=resets,
                    chunk_size=_c,
                    state_dtype=jnp.float32,
                    layout="token_major",
                    outer_block=32,
                    output_dtype=jnp.bfloat16,
                    state_stride=0,
                ),
            ),
            (
                "GDN",
                False,
                lambda q_, k_, v_, g_, b_, _c=c: run_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, _c, "float32"),
            ),
        ):
          roof = compute_core_roofline_analytical(
              b, seq_len, h, dk, dv, c, is_kda=is_kda, state_f32=True, state_stride=0 if is_kda else 1, outer_block=32, v7x_stacked_opt=True
          )
          g_arg = g_kda if is_kda else g_gdn
          j_fwd = jax.jit(fn)
          j_fb = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy),
                  argnums=(0, 1, 2, 3, 4),
              )
          )
          t_f, _ = _time_fn(j_fwd, q, k, v, g_arg, beta)
          t_fb, _, hlo = _time_and_profile_hlo(j_fb, q, k, v, g_arg, beta)
          print(format_perf_vs_roofline(f"{rule_tag}:{cfg_name[:28]}", t_f, t_fb, roof, hlo), flush=True)

      # Sweep 2: Chunk Size C in {32, 64, 128, 256} for KDA and GDN (B=1, T=8192, H=8, dk=256, dv=512)
      print("\n=== Co-Design Sweep 2: Chunk Size C in {32, 64, 128, 256} (T=8192, H=8, dk=256, dv=512) ===", flush=True)
      h, dk, dv = 8, 256, 512
      q = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, h, dk), jnp.float32)) * (dk**-0.5)
      k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, h, dk), jnp.float32))
      v = jax.random.normal(keys[2], (b, seq_len, h, dv), jnp.float32) * 0.1
      g_kda = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, dk), jnp.float32) - 2.0)
      g_gdn = jnp.mean(g_kda, axis=-1)
      beta = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, h), jnp.float32))
      dy = jax.random.normal(keys[5], (b, seq_len, h, dv), jnp.float32) * 0.1

      for c in (32, 64, 128, 256):
        ob = 32 if c >= 64 else None
        if c <= 128:
          c_arms = (
              (
                  f"KDA_C{c:03d}_Pallas",
                  True,
                  lambda q_, k_, v_, g_, b_, _c=c, _ob=ob: kda.chunk_kda(
                      q_,
                      k_,
                      v_,
                      g_,
                      b_,
                      resets=resets,
                      chunk_size=_c,
                      state_dtype=jnp.float32,
                      layout="token_major",
                      outer_block=_ob,
                      output_dtype=jnp.bfloat16,
                      state_stride=0,
                  ),
              ),
              (
                  f"GDN_C{c:03d}_Pallas",
                  False,
                  lambda q_, k_, v_, g_, b_, _c=c: run_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, _c, "float32"),
              ),
          )
        else:
          # At C=256, Pallas Mosaic LLO aborts with Check failed: input->opcode() == LloOpcode::kVectorMove
          # (kVectorPack vs. kVectorMove) on 256x256 triangular inverse VMEM sublane packing; benchmark via XLA!
          c_arms = (
              (
                  f"KDA_C{c:03d}_XLA_Subblock",
                  True,
                  lambda q_, k_, v_, g_, b_, _c=c: olmoe3._delta_rule_chunked_subblock(
                      q_, k_, v_, g_, b_, resets, _c, "float32", sub_block=16, intra_vjp=True
                  ),
              ),
              (
                  f"GDN_C{c:03d}_XLA_WY",
                  False,
                  lambda q_, k_, v_, g_, b_, _c=c: qwen3.jax_chunk_gated_delta_rule(
                      q_, k_, v_, g_, b_, chunk_size=_c, compute_dtype=jnp.bfloat16
                  )[0],
              ),
          )
        for rule_tag, is_kda, fn in c_arms:
          try:
            roof = compute_core_roofline_analytical(
                b, seq_len, h, dk, dv, c, is_kda=is_kda, state_f32=True, state_stride=0 if is_kda else 1, outer_block=ob, v7x_stacked_opt=True
            )
            g_arg = g_kda if is_kda else g_gdn
            j_fwd = jax.jit(fn)
            j_fb = jax.jit(
                jax.value_and_grad(
                    lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy),
                    argnums=(0, 1, 2, 3, 4),
                )
            )
            t_f, _ = _time_fn(j_fwd, q, k, v, g_arg, beta)
            t_fb, _, hlo = _time_and_profile_hlo(j_fb, q, k, v, g_arg, beta)
            print(format_perf_vs_roofline(rule_tag, t_f, t_fb, roof, hlo), flush=True)
          except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[{rule_tag:34s}] skipped/failed: {type(exc).__name__}: {exc}", flush=True)

      # Sweep 3: Decay Parameterization (Per-Channel KDA vs Tile-Grouped Decay vs Scalar GDN)
      print("\n=== Co-Design Sweep 3: Decay Parameterization (Per-Channel vs Tile-Grouped vs Scalar) ===", flush=True)
      for g_groups, label in (
          (1, "Scalar_Decay_G1 (GDN Pallas Fused)"),
          (2, "TileGrouped_Decay_G2 (128ch/group)"),
          (4, "TileGrouped_Decay_G4 (64ch/group)"),
          (256, "PerChannel_Decay_G256 (KDA Pallas Fused)"),
      ):
        if g_groups == 1:
          fn = lambda q_, k_, v_, g_, b_: run_gdn_pallas_fused(q_, k_, v_, g_[..., 0], b_, resets, 128, "float32")
          g_in = g_kda[..., :1]
        elif g_groups == 256:
          fn = lambda q_, k_, v_, g_, b_: kda.chunk_kda(
              q_,
              k_,
              v_,
              g_,
              b_,
              resets=resets,
              chunk_size=128,
              state_dtype=jnp.float32,
              layout="token_major",
              outer_block=32,
              output_dtype=jnp.bfloat16,
              state_stride=0,
          )
          g_in = g_kda
        else:
          fn = lambda q_, k_, v_, g_, b_, _gg=g_groups: _kda_tile_grouped_decay_core(
              q_, k_, v_, g_, b_, resets, 128, _gg
          )
          g_in = g_kda[..., :g_groups]
        j_fwd = jax.jit(fn)
        j_fb = jax.jit(
            jax.value_and_grad(
                lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy),
                argnums=(0, 1, 2, 3, 4),
            )
        )
        t_f, _ = _time_fn(j_fwd, q, k, v, g_in, beta)
        t_fb, _, hlo = _time_and_profile_hlo(j_fb, q, k, v, g_in, beta)
        print(
            f"[{label:40s}] fwd={t_f:6.3f} ms | bwd={t_fb-t_f:6.3f} ms | fwd+bwd={t_fb:6.3f} ms "
            f"(x24={24*t_fb:6.1f} ms) | HLO={hlo}",
            flush=True,
        )

      # Sweep 4: Gate & Projection Architecture Co-Design (Two-Stage Low-Rank vs Single-Stage Fused)
      print("\n=== Co-Design Sweep 4: Two-Stage Low-Rank Gate Projections vs Single-Stage Fused Projection ===", flush=True)
      d_model = 1792
      x_act = jax.random.normal(keys[0], (b, seq_len, d_model), jnp.bfloat16)
      # Current KDA: W_in [1792, 9224] (q:2048, k:2048, v:4096, f1:512, g1:512) + f2 [512, 2048] + g2 [512, 4096]
      w_in_2stage = jax.random.normal(keys[1], (d_model, 9224), jnp.bfloat16) * 0.02
      w_f2 = jax.random.normal(keys[2], (512, 2048), jnp.bfloat16) * 0.02
      w_g2 = jax.random.normal(keys[3], (512, 4096), jnp.bfloat16) * 0.02
      # Co-design A: Single-stage fused with grouped decay (G=2 -> 16 scalars) and head-wise output gate (or direct)
      w_in_1stage_direct = jax.random.normal(keys[4], (d_model, 2048 + 2048 + 4096 + 2048 + 4096 + 8), jnp.bfloat16) * 0.02
      w_in_1stage_gdn = jax.random.normal(keys[5], (d_model, 2048 + 2048 + 4096 + 4096 + 16), jnp.bfloat16) * 0.02

      def _proj_2stage(x_, w_in_, wf2_, wg2_):
        all_in = jnp.dot(x_, w_in_)
        q_, k_, v_, f1_, g1_, b_ = jnp.split(all_in, [2048, 4096, 8192, 8704, 9216], axis=-1)
        f_ = jnp.dot(f1_, wf2_)
        g_ = jax.nn.sigmoid(jnp.dot(g1_, wg2_))
        return jnp.sum(q_) + jnp.sum(k_) + jnp.sum(v_ * g_) + jnp.sum(f_) + jnp.sum(b_)

      def _proj_1stage_direct(x_, w_all_):
        all_in = jnp.dot(x_, w_all_)
        q_, k_, v_, f_, g_, b_ = jnp.split(all_in, [2048, 4096, 8192, 10240, 14336], axis=-1)
        return jnp.sum(q_) + jnp.sum(k_) + jnp.sum(v_ * jax.nn.sigmoid(g_)) + jnp.sum(f_) + jnp.sum(b_)

      def _proj_1stage_gdn(x_, w_all_):
        all_in = jnp.dot(x_, w_all_)
        q_, k_, v_, g_, fb_ = jnp.split(all_in, [2048, 4096, 8192, 12288], axis=-1)
        return jnp.sum(q_) + jnp.sum(k_) + jnp.sum(v_ * jax.nn.sigmoid(g_)) + jnp.sum(fb_)

      for p_name, p_fn, p_args in (
          ("TwoStage_LowRank_KDA (9224 + f2 + g2)", _proj_2stage, (x_act, w_in_2stage, w_f2, w_g2)),
          ("SingleStage_Fused_Direct_KDA (14344)", _proj_1stage_direct, (x_act, w_in_1stage_direct)),
          ("SingleStage_Fused_GDN_ScalarGate (12304)", _proj_1stage_gdn, (x_act, w_in_1stage_gdn)),
      ):
        j_fb = jax.jit(jax.value_and_grad(p_fn, argnums=tuple(range(len(p_args)))))
        t_fb, _, hlo = _time_and_profile_hlo(j_fb, *p_args)
        print(f"[{p_name:42s}] fwd+bwd={t_fb:6.3f} ms (x24={24*t_fb:6.1f} ms) | HLO={hlo}", flush=True)
    finally:
      kda_fused._fwd_kernel = orig_fwd
      kda_fused._bwd_kernel = orig_bwd

  def test_5_l1_to_l10_and_rearchitecting_xprof(self):
    """Shard 5: L1-L14 Core & Full-Layer Re-Architecting Sweep + XPlane Hardware Profiling on TPU v7x."""
    print(
        f"\n================================================================================"
        f"\nSHARD 5: TPU v7x ({self.num_devs}x {self.dev_kind}) L1-L14 RE-ARCHITECTING & XPLANE PROFILER SUITE"
        f"\n================================================================================",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # A. Define VMEM-Fused Pallas Kernels for Re-Architected Variants:
    #    1) Mamba-2 SSD (no Householder (I+M)^-1 inversion): XLA & VMEM-Fused Pallas
    #    2) SDN-256 VMEM-Fused Pallas with Tile-Grouped Decay (G=2 and G=4)
    #    3) GVA-GDN / GVA-SDN (2:1 and 4:1 Grouped-Value Attention in VMEM-Fused Pallas)
    # -------------------------------------------------------------------------
    def _ssd_xla_chunked(q, k, v, log_decay_scalar, beta, resets, chunk_size: int = 128):
      b, t, h, dk = q.shape
      dv = v.shape[-1]
      c = chunk_size
      n = t // c
      f32, bf16 = jnp.float32, jnp.bfloat16
      to_c = lambda x_: x_.reshape(b, n, c, h, -1).transpose(0, 3, 1, 2, 4)
      q_c, k_c, v_c = (to_c(x_.astype(bf16)) for x_ in (q, k, v))
      g_in = log_decay_scalar.reshape(b, n, c, h).transpose(0, 3, 1, 2).astype(f32)
      tri = jnp.tril(jnp.ones((c, c), bf16))
      g_cum = jnp.einsum("ij,...j->...i", tri, g_in, preferred_element_type=f32, precision=jax.lax.Precision.HIGHEST)
      beta_c = beta.reshape(b, n, c, h).transpose(0, 3, 1, 2).astype(f32)[..., None]
      causal = jnp.tril(jnp.ones((c, c), dtype=bool))
      exp_g_diff = jnp.exp(jnp.where(causal, g_cum[..., :, None] - g_cum[..., None, :], -jnp.inf))
      scores = jnp.einsum("...id,...jd->...ij", q_c, k_c, preferred_element_type=f32) * exp_g_diff
      eg = jnp.exp(g_cum)[..., None]
      g_last = g_cum[..., -1:]
      k_carry = (k_c.astype(f32) * jnp.exp((g_last - g_cum)[..., None])).astype(bf16)
      u_c = (v_c.astype(f32) * beta_c).astype(bf16)
      qg = (q_c.astype(f32) * eg).astype(bf16)
      decay = jnp.exp(g_last)  # [b, h, n, 1]

      def scan_body(st, xs):
        kc_i, u_i, dl_i, qg_i, sc_i = xs
        out_i = jnp.einsum("bhcd,bhdv->bhcv", qg_i, st, preferred_element_type=f32) + jnp.einsum(
            "bhij,bhjv->bhiv", sc_i.astype(bf16), u_i, preferred_element_type=f32
        )
        st_new = (st.astype(f32) * dl_i[..., None] + jnp.einsum("bhcd,bhcv->bhdv", kc_i, u_i, preferred_element_type=f32)).astype(bf16)
        return st_new, out_i.astype(bf16)

      init_st = jnp.zeros((b, h, dk, dv), bf16)
      tr_n = lambda x_: jnp.moveaxis(x_, 2, 0)
      _, out_n = jax.lax.scan(scan_body, init_st, (tr_n(k_carry), tr_n(u_c), tr_n(decay), tr_n(qg), tr_n(scores)))
      return jnp.moveaxis(out_n, 0, 1).reshape(b, t, h, dv)

    def _ssd_fwd_kernel(*refs, state_f32):
      ins, outs, scratch = refs[:7], refs[7:10], refs[10:]
      q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, dl_ref = (kda_fused._view(r) for r in ins)
      out_ref, st_ref, sc_out = (kda_fused._view(r) for r in outs)
      s_ref = scratch[0]
      f32, bf16 = jnp.float32, jnp.bfloat16
      c = k_ref.shape[0]
      row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
      col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
      causal = row >= col
      tri = kda_fused._tril_ones(c)
      g_in = g_ref[...].astype(f32)
      g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
      exp_g_diff = jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf))
      qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
      q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)
      scores_bf = (jnp.dot(q_bf, jnp.transpose(k_bf), preferred_element_type=f32) * exp_g_diff).astype(bf16)

      @pl.when(pl.program_id(1) == 0)
      def _():
        s_ref[...] = jnp.zeros_like(s_ref)

      seg_c = segc_ref[...]
      fp = (seg_c == 0).astype(f32)
      eg = jnp.exp(g_cum)
      ecar = jnp.exp(g_cum[c - 1 : c, :] - g_cum)
      beta = beta_ref[...].astype(f32)
      u_bf = (v_ref[...].astype(f32) * beta).astype(bf16)
      qg = (qf * eg * fp).astype(bf16)
      kc = (kf * ecar).astype(bf16)
      s = s_ref[...]
      s_bf = s.astype(bf16)
      st_ref[...] = s if state_f32 else s_bf
      sc_out[...] = scores_bf
      out = jnp.dot(qg, s_bf, preferred_element_type=f32) + jnp.dot(scores_bf, u_bf, preferred_element_type=f32)
      out_ref[...] = out.astype(out_ref.dtype)
      new = s * dl_ref[...] + kda_fused._tn(kc, u_bf)
      s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)

    def _ssd_bwd_kernel(*refs, state_f32):
      ins, outs, scratch = refs[:10], refs[10:16], refs[16:]
      q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, dl_ref, st_ref, sc_ref, do_ref = (
          kda_fused._view(r) for r in ins
      )
      dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref, ddl_ref = (kda_fused._view(r) for r in outs)
      r_ref = scratch[0]

      @pl.when(pl.program_id(1) == 0)
      def _():
        r_ref[...] = jnp.zeros_like(r_ref)

      f32, bf16 = jnp.float32, jnp.bfloat16
      c = k_ref.shape[0]
      row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
      col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
      causal = row >= col
      tri = kda_fused._tril_ones(c)
      g_in = g_ref[...].astype(f32)
      g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
      exp_g_diff = jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf))
      qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
      seg_c = segc_ref[...]
      fp = (seg_c == 0).astype(f32)
      eg = jnp.exp(g_cum)
      ecar = jnp.exp(g_cum[c - 1 : c, :] - g_cum)
      beta = beta_ref[...].astype(f32)
      u_bf = (v_ref[...].astype(f32) * beta).astype(bf16)
      qg = (qf * eg * fp).astype(bf16)
      kc = (kf * ecar).astype(bf16)
      states_f = st_ref[...].astype(f32)
      states = states_f.astype(bf16)
      scores_bf = sc_ref[...]
      d_out = do_ref[...].astype(bf16)
      r = r_ref[...]
      r_bf = r.astype(bf16)

      dsc = kda_fused._nt(d_out, u_bf)
      du = kda_fused._tn(scores_bf, d_out) + jnp.dot(kc, r_bf, preferred_element_type=f32)
      dkc = kda_fused._nt(u_bf, r_bf)
      ddl_ref[...] = jnp.sum(states_f * r, axis=-1, keepdims=True)
      dqg = kda_fused._nt(d_out, states)
      r_ref[...] = r * dl_ref[...] + kda_fused._tn(qg, d_out)

      ds_scaled = (jnp.where(causal, dsc, 0.0) * exp_g_diff).astype(bf16)
      q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)
      dq_i = jnp.dot(ds_scaled, k_bf, preferred_element_type=f32)
      dk_col = jnp.dot(jnp.transpose(ds_scaled), q_bf, preferred_element_type=f32)
      dg_i = jnp.sum(qf * dq_i - kf * dk_col, axis=-1, keepdims=True)

      dv_ref[...] = (du * beta).astype(dv_ref.dtype)
      dbeta_ref[...] = jnp.sum(du * v_ref[...].astype(f32), axis=1, keepdims=True)
      dq_ref[...] = (dq_i + dqg * eg * fp).astype(dq_ref.dtype)
      dk_ref[...] = (dk_col + dkc * ecar).astype(dk_ref.dtype)
      carry = dkc * kf * ecar
      dg_wy = jnp.sum(dqg * qf * eg * fp - carry, axis=-1, keepdims=True)
      dg_last = jnp.sum(carry, axis=(0, 1), keepdims=True)
      dg = dg_i + dg_wy + jnp.where(row == c - 1, dg_last, 0.0)
      dg_ref[...] = jnp.sum(jnp.transpose(tri) * jnp.transpose(dg), axis=1, keepdims=True)

    @functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
    def ssd_pallas_fused_token_major(q, k, v, g_scalar, beta, seg, decay_last, state_f32: bool = True, out_dtype=jnp.bfloat16):
      return _ssd_pallas_tm_fwd(q, k, v, g_scalar, beta, seg, decay_last, state_f32, out_dtype)[0]

    def _ssd_pallas_tm_fwd(q, k, v, g_scalar, beta, seg, decay_last, state_f32: bool = True, out_dtype=jnp.bfloat16):
      b, t, h, dk = q.shape
      dv = v.shape[-1]
      n = decay_last.shape[2]
      c = t // n
      bh = b * h
      f32, bf16 = jnp.float32, jnp.bfloat16
      s_dtype = f32 if state_f32 else bf16
      q_tm, k_tm, v_tm = q.reshape(b, t, h * dk), k.reshape(b, t, h * dk), v.reshape(b, t, h * dv)
      g_col = g_scalar.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
      beta_col = beta.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
      seg_col = seg[:, 0][..., None]
      dl = decay_last.reshape(bh, n, dk, 1).astype(f32)
      per, seg_sp = kda_fused._specs(n, h, reverse=False)
      tok = kda_fused._tok_spec(n, c, h, reverse=False)
      out_tm, states, scores = pl.pallas_call(
          functools.partial(_ssd_fwd_kernel, state_f32=state_f32),
          grid=(bh, n),
          in_specs=[tok(dk), tok(dk), tok(dv), per(c, 1), per(c, 1), seg_sp(c, 1), per(dk, 1)],
          out_specs=[tok(dv), per(dk, dv), per(c, c)],
          out_shape=[
              jax.ShapeDtypeStruct((b, t, h * dv), out_dtype),
              jax.ShapeDtypeStruct((bh, n, dk, dv), s_dtype),
              jax.ShapeDtypeStruct((bh, n, c, c), bf16),
          ],
          scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
          compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl)
      return out_tm.reshape(b, t, h, dv), (q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl, states, scores, h, dk, dv)

    def _ssd_pallas_tm_bwd(state_f32, out_dtype, res, do):
      del out_dtype
      q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl, states, scores, h, dk, dv = res
      b, t, _ = q_tm.shape
      n = dl.shape[1]
      c = t // n
      bh = b * h
      f32 = jnp.float32
      do_tm = do.reshape(b, t, h * dv)
      per, seg_sp = kda_fused._specs(n, h, reverse=True)
      tok = kda_fused._tok_spec(n, c, h, reverse=True)
      dq_tm, dk_tm, dv_tm, dg_col, dbeta_col, ddl = pl.pallas_call(
          functools.partial(_ssd_bwd_kernel, state_f32=state_f32),
          grid=(bh, n),
          in_specs=[tok(dk), tok(dk), tok(dv), per(c, 1), per(c, 1), seg_sp(c, 1), per(dk, 1), per(dk, dv), per(c, c), tok(dv)],
          out_specs=[tok(dk), tok(dk), tok(dv), per(c, 1), per(c, 1), per(dk, 1)],
          out_shape=[
              jax.ShapeDtypeStruct((b, t, h * dk), q_tm.dtype),
              jax.ShapeDtypeStruct((b, t, h * dk), k_tm.dtype),
              jax.ShapeDtypeStruct((b, t, h * dv), v_tm.dtype),
              jax.ShapeDtypeStruct((bh, n, c, 1), f32),
              jax.ShapeDtypeStruct((bh, n, c, 1), f32),
              jax.ShapeDtypeStruct((bh, n, dk, 1), f32),
          ],
          scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
          compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl, states, scores, do_tm)
      dq = dq_tm.reshape(b, t, h, dk)
      dk_o = dk_tm.reshape(b, t, h, dk)
      dv_o = dv_tm.reshape(b, t, h, dv)
      dg = dg_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h, 1)
      dbeta = dbeta_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h)
      return dq, dk_o, dv_o, dg, dbeta, jnp.zeros((b, 1, n, c), jax.dtypes.float0), ddl.reshape(b, h, n, dk)

    ssd_pallas_fused_token_major.defvjp(_ssd_pallas_tm_fwd, _ssd_pallas_tm_bwd)

    def run_ssd_pallas_fused(q, k, v, log_decay_scalar, beta, resets, chunk_size: int = 128, state_dtype: str = "float32"):
      b, t, h, dk = q.shape
      c = chunk_size
      n = t // c
      f32 = jnp.float32
      log_decay_scalar = log_decay_scalar.astype(f32)
      beta = beta.astype(f32)
      seg = jnp.cumsum(resets.reshape(b, n, c).astype(jnp.int32), axis=-1)[:, None]
      any_reset = seg[..., -1] > 0
      g_sum = jnp.sum(log_decay_scalar.reshape(b, n, c, h).transpose(0, 3, 1, 2), axis=-1)
      decay_last = jnp.broadcast_to(jnp.where(any_reset, 0.0, jnp.exp(g_sum))[..., None], (b, h, n, dk))
      return ssd_pallas_fused_token_major(
          q, k, v, log_decay_scalar[..., None], beta, seg, decay_last, state_dtype == "float32", jnp.bfloat16
      )

    # 2) VMEM-Fused Token-Major Pallas SDN Kernel with Tile-Grouped Decay (G=2: 128ch/group, G=4: 64ch/group at dk=256)
    def _sdn_grouped_fwd_kernel(*refs, precision, num_groups: int, state_f32: bool):
      ins, outs, scratch = refs[:8], refs[8:13], refs[13:]
      q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref = (kda_fused._view(r) for r in ins)
      out_ref, st_ref, delta_out, xk_out, sc_out = (kda_fused._view(r) for r in outs)
      s_ref = scratch[0]

      f32, bf16 = jnp.float32, jnp.bfloat16
      c = k_ref.shape[0]
      dv = v_ref.shape[1]
      row, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
      col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
      col_128 = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
      tri = kda_fused._tril_ones(c)
      seg_c = segc_ref[...]
      fp = (seg_c == 0).astype(f32)
      last_seg = seg_c == seg_c[c - 1 : c, :]

      qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
      q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)
      scores = jnp.zeros((c, c), f32)
      kk = jnp.zeros((c, c), f32)

      if num_groups == 2:
        eg_list, ecar_list = [], []
        for g_idx in range(2):
          g_in = g_ref[g_idx * c : (g_idx + 1) * c, :].astype(f32)
          g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
          exp_g_diff = jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf))
          q_g = q_bf[:, g_idx * 128 : (g_idx + 1) * 128]
          k_g = k_bf[:, g_idx * 128 : (g_idx + 1) * 128]
          qk_g = jnp.concatenate([q_g, k_g], axis=0)
          dots_g = jnp.dot(qk_g, jnp.transpose(k_g), preferred_element_type=f32)
          scores = scores + dots_g[:c, :] * exp_g_diff
          kk = kk + dots_g[c:, :] * exp_g_diff
          eg_list.append(jnp.broadcast_to(jnp.exp(g_cum), (c, 128)))
          ecar_list.append(jnp.broadcast_to(jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf)), (c, 128)))
        eg = jnp.concatenate(eg_list, axis=1)
        ecar = jnp.concatenate(ecar_list, axis=1)
      else:  # num_groups == 4 (64 channels/group inside two 128-column halves)
        eg_halves, ecar_halves = [], []
        for h_idx in range(2):
          q_half = q_bf[:, h_idx * 128 : (h_idx + 1) * 128]
          k_half = k_bf[:, h_idx * 128 : (h_idx + 1) * 128]
          eg_pair, ecar_pair = [], []
          for sub_idx in range(2):
            g_idx = h_idx * 2 + sub_idx
            g_in = g_ref[g_idx * c : (g_idx + 1) * c, :].astype(f32)
            g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
            exp_g_diff = jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf))
            mask_64 = (col_128 < 64) if sub_idx == 0 else (col_128 >= 64)
            q_g = jnp.where(mask_64, q_half, jnp.zeros_like(q_half))
            k_g = jnp.where(mask_64, k_half, jnp.zeros_like(k_half))
            qk_g = jnp.concatenate([q_g, k_g], axis=0)
            dots_g = jnp.dot(qk_g, jnp.transpose(k_g), preferred_element_type=f32)
            scores = scores + dots_g[:c, :] * exp_g_diff
            kk = kk + dots_g[c:, :] * exp_g_diff
            eg_pair.append(jnp.exp(g_cum))
            ecar_pair.append(jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf)))
          eg_halves.append(jnp.where(col_128 < 64, eg_pair[0], eg_pair[1]))
          ecar_halves.append(jnp.where(col_128 < 64, ecar_pair[0], ecar_pair[1]))
        eg = jnp.concatenate(eg_halves, axis=1)
        ecar = jnp.concatenate(ecar_halves, axis=1)

      kk = jnp.where(strict, kk, 0.0)
      beta = beta_ref[...].astype(f32)
      x = kda_fused._inverse(kk * beta, (row == col).astype(f32), precision)
      scores_bf = scores.astype(bf16)

      @pl.when(pl.program_id(1) == 0)
      def _():
        s_ref[...] = jnp.zeros_like(s_ref)

      vb = (v_ref[...].astype(f32) * beta).astype(bf16)
      kb = (kf * eg * beta * fp).astype(bf16)
      qg = (qf * eg * fp).astype(bf16)
      kc = (kf * ecar).astype(bf16)
      x_bf = x.astype(bf16)
      s = s_ref[...]
      s_bf = s.astype(bf16)
      st_ref[...] = s if state_f32 else s_bf

      vb_kb = jnp.concatenate([vb, kb], axis=1)
      uw = jnp.dot(x_bf, vb_kb, preferred_element_type=f32)
      u, w = uw[:, :dv], uw[:, dv:]
      wqg = jnp.concatenate([w.astype(bf16), qg], axis=0)
      ws_qgs = jnp.dot(wqg, s_bf, preferred_element_type=f32)
      ws, qgs = ws_qgs[:c, :], ws_qgs[c:, :]
      delta_bf = (u - ws).astype(bf16)
      delta_out[...] = delta_bf
      xk_out[...] = kda_fused._pack_x_kk(x, kk, row, col)
      sc_out[...] = scores_bf
      out = qgs + jnp.dot(scores_bf, delta_bf, preferred_element_type=f32)
      out_ref[...] = out.astype(out_ref.dtype)
      new = s * dl_ref[...] + kda_fused._tn(kc, delta_bf)
      s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)

    def _sdn_grouped_bwd_kernel(*refs, precision, num_groups: int, state_f32: bool):
      ins, outs, scratch = refs[:13], refs[13:19], refs[19:]
      q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref, st_ref, de_ref, xk_ref, sc_ref, do_ref = (
          kda_fused._view(r) for r in ins
      )
      dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref, ddl_ref = (kda_fused._view(r) for r in outs)
      r_ref = scratch[0]

      @pl.when(pl.program_id(1) == 0)
      def _():
        r_ref[...] = jnp.zeros_like(r_ref)

      f32, bf16 = jnp.float32, jnp.bfloat16
      c = k_ref.shape[0]
      dv = v_ref.shape[1]
      row, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
      col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
      col_128 = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
      tri = kda_fused._tril_ones(c)
      seg_c = segc_ref[...]
      fp = (seg_c == 0).astype(f32)
      last_seg = seg_c == seg_c[c - 1 : c, :]

      qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
      q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)

      if num_groups == 2:
        eg_list, ecar_list, exp_diffs = [], [], []
        for g_idx in range(2):
          g_in = g_ref[g_idx * c : (g_idx + 1) * c, :].astype(f32)
          g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
          exp_diffs.append(jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf)))
          eg_list.append(jnp.broadcast_to(jnp.exp(g_cum), (c, 128)))
          ecar_list.append(jnp.broadcast_to(jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf)), (c, 128)))
        eg = jnp.concatenate(eg_list, axis=1)
        ecar = jnp.concatenate(ecar_list, axis=1)
      else:
        eg_halves, ecar_halves, exp_diffs = [], [], []
        for h_idx in range(2):
          eg_pair, ecar_pair = [], []
          for sub_idx in range(2):
            g_idx = h_idx * 2 + sub_idx
            g_in = g_ref[g_idx * c : (g_idx + 1) * c, :].astype(f32)
            g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
            exp_diffs.append(jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf)))
            eg_pair.append(jnp.exp(g_cum))
            ecar_pair.append(jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf)))
          eg_halves.append(jnp.where(col_128 < 64, eg_pair[0], eg_pair[1]))
          ecar_halves.append(jnp.where(col_128 < 64, ecar_pair[0], ecar_pair[1]))
        eg = jnp.concatenate(eg_halves, axis=1)
        ecar = jnp.concatenate(ecar_halves, axis=1)

      beta = beta_ref[...].astype(f32)
      vb = (v_ref[...].astype(f32) * beta).astype(bf16)
      kb = (kf * eg * beta * fp).astype(bf16)
      qg = (qf * eg * fp).astype(bf16)
      kc = (kf * ecar).astype(bf16)

      x, kk = kda_fused._unpack_x_kk(xk_ref[...], row, col)
      x_bf, scores_bf = x.astype(bf16), sc_ref[...]
      states_f, delta = st_ref[...].astype(f32), de_ref[...]
      states = states_f.astype(bf16)
      d_out = do_ref[...].astype(bf16)

      dsc = kda_fused._nt(d_out, delta)
      g_delta = kda_fused._tn(scores_bf, d_out)
      r = r_ref[...]
      r_bf = r.astype(bf16)
      dd = g_delta + jnp.dot(kc, r_bf, preferred_element_type=f32)
      dd_bf = dd.astype(bf16)
      w_bf = jnp.dot(x_bf, kb, preferred_element_type=f32).astype(bf16)
      dkc = kda_fused._nt(delta, r_bf)
      ddl_ref[...] = jnp.sum(states_f * r, axis=-1, keepdims=True)

      do_neg_dd = jnp.concatenate([d_out, -dd_bf], axis=0)
      dqg_dw = kda_fused._nt(do_neg_dd, states)
      dqg = dqg_dw[:c, :]
      dw_bf = dqg_dw[c:, :].astype(bf16)
      qg_w = jnp.concatenate([qg, w_bf], axis=0)
      r_ref[...] = r * dl_ref[...] + kda_fused._tn(qg_w, do_neg_dd)

      dd_dw = jnp.concatenate([dd_bf, dw_bf], axis=1)
      vb_kb = jnp.concatenate([vb, kb], axis=1)
      dx = kda_fused._nt(dd_dw, vb_kb)
      dvb_dkb = kda_fused._tn(x_bf, dd_dw)
      dvb, dkb = dvb_dkb[:, :dv], dvb_dkb[:, dv:]

      dm = -kda_fused._mm(kda_fused._mm(x, dx, precision, ((0,), (0,))), x, precision, ((1,), (1,)))
      dm = jnp.where(strict, dm, 0.0)
      dbeta = jnp.sum(dm * kk, axis=1, keepdims=True)
      dkk = dm * beta
      ds = jnp.where(causal, dsc, 0.0)

      kegfp = kf * eg * fp
      carry = dkc * kf * ecar
      wy_term = dqg * qf * eg * fp + dkb * beta * kegfp - carry

      if num_groups == 2:
        dq_parts, dk_parts, dg_parts = [], [], []
        for g_idx in range(2):
          exp_g_diff = exp_diffs[g_idx]
          ds_s = (ds * exp_g_diff).astype(bf16)
          dkk_s = (dkk * exp_g_diff).astype(bf16)
          q_g = q_bf[:, g_idx * 128 : (g_idx + 1) * 128]
          k_g = k_bf[:, g_idx * 128 : (g_idx + 1) * 128]
          qf_g = qf[:, g_idx * 128 : (g_idx + 1) * 128]
          kf_g = kf[:, g_idx * 128 : (g_idx + 1) * 128]
          dq_g = jnp.dot(ds_s, k_g, preferred_element_type=f32)
          dk_row_g = jnp.dot(dkk_s, k_g, preferred_element_type=f32)
          ds_dkk_t = jnp.concatenate([jnp.transpose(ds_s), jnp.transpose(dkk_s)], axis=1)
          qk_g = jnp.concatenate([q_g, k_g], axis=0)
          dk_col_g = jnp.dot(ds_dkk_t, qk_g, preferred_element_type=f32)
          dq_parts.append(dq_g)
          dk_parts.append(dk_row_g + dk_col_g)
          dg_i_g = jnp.sum(qf_g * dq_g + kf_g * (dk_row_g - dk_col_g), axis=-1, keepdims=True)
          dg_wy_g = jnp.sum(wy_term[:, g_idx * 128 : (g_idx + 1) * 128], axis=-1, keepdims=True)
          dg_last_g = jnp.sum(carry[:, g_idx * 128 : (g_idx + 1) * 128], axis=(0, 1), keepdims=True)
          dg_g = dg_i_g + dg_wy_g + jnp.where(row == c - 1, dg_last_g, 0.0)
          dg_parts.append(jnp.sum(jnp.transpose(tri) * jnp.transpose(dg_g), axis=1, keepdims=True))
        dq_i = jnp.concatenate(dq_parts, axis=1)
        dk_i = jnp.concatenate(dk_parts, axis=1)
        dg_out = jnp.concatenate(dg_parts, axis=0)
      else:
        dq_halves, dk_halves, dg_parts = [], [], []
        for h_idx in range(2):
          q_half = q_bf[:, h_idx * 128 : (h_idx + 1) * 128]
          k_half = k_bf[:, h_idx * 128 : (h_idx + 1) * 128]
          qf_half = qf[:, h_idx * 128 : (h_idx + 1) * 128]
          kf_half = kf[:, h_idx * 128 : (h_idx + 1) * 128]
          wy_half = wy_term[:, h_idx * 128 : (h_idx + 1) * 128]
          car_half = carry[:, h_idx * 128 : (h_idx + 1) * 128]
          dq_pair, dk_pair = [], []
          for sub_idx in range(2):
            g_idx = h_idx * 2 + sub_idx
            exp_g_diff = exp_diffs[g_idx]
            ds_s = (ds * exp_g_diff).astype(bf16)
            dkk_s = (dkk * exp_g_diff).astype(bf16)
            mask_64 = (col_128 < 64) if sub_idx == 0 else (col_128 >= 64)
            q_g = jnp.where(mask_64, q_half, jnp.zeros_like(q_half))
            k_g = jnp.where(mask_64, k_half, jnp.zeros_like(k_half))
            qf_g = jnp.where(mask_64, qf_half, 0.0)
            kf_g = jnp.where(mask_64, kf_half, 0.0)
            dq_g = jnp.dot(ds_s, k_g, preferred_element_type=f32)
            dk_row_g = jnp.dot(dkk_s, k_g, preferred_element_type=f32)
            ds_dkk_t = jnp.concatenate([jnp.transpose(ds_s), jnp.transpose(dkk_s)], axis=1)
            qk_g = jnp.concatenate([q_g, k_g], axis=0)
            dk_col_g = jnp.dot(ds_dkk_t, qk_g, preferred_element_type=f32)
            dq_pair.append(dq_g)
            dk_pair.append(dk_row_g + dk_col_g)
            dg_i_g = jnp.sum(qf_g * dq_g + kf_g * (dk_row_g - dk_col_g), axis=-1, keepdims=True)
            dg_wy_g = jnp.sum(jnp.where(mask_64, wy_half, 0.0), axis=-1, keepdims=True)
            dg_last_g = jnp.sum(jnp.where(mask_64, car_half, 0.0), axis=(0, 1), keepdims=True)
            dg_g = dg_i_g + dg_wy_g + jnp.where(row == c - 1, dg_last_g, 0.0)
            dg_parts.append(jnp.sum(jnp.transpose(tri) * jnp.transpose(dg_g), axis=1, keepdims=True))
          dq_halves.append(jnp.where(col_128 < 64, dq_pair[0], dq_pair[1]))
          dk_halves.append(jnp.where(col_128 < 64, dk_pair[0], dk_pair[1]))
        dq_i = jnp.concatenate(dq_halves, axis=1)
        dk_i = jnp.concatenate(dk_halves, axis=1)
        dg_out = jnp.concatenate(dg_parts, axis=0)

      dv_ref[...] = (dvb * beta).astype(dv_ref.dtype)
      dbeta_ref[...] = (
          dbeta
          + jnp.sum(dvb * v_ref[...].astype(f32), axis=1, keepdims=True)
          + jnp.sum(dkb * kegfp, axis=1, keepdims=True)
      )
      dq_ref[...] = (dq_i + dqg * eg * fp).astype(dq_ref.dtype)
      dk_ref[...] = (dk_i + dkb * eg * beta * fp + dkc * ecar).astype(dk_ref.dtype)
      dg_ref[...] = dg_out

    @functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10))
    def sdn_pallas_fused_token_major(
        q, k, v, g_grouped, beta, seg, decay_last, precision, num_groups: int = 2, state_f32: bool = True, out_dtype=jnp.bfloat16
    ):
      return _sdn_pallas_tm_fwd(q, k, v, g_grouped, beta, seg, decay_last, precision, num_groups, state_f32, out_dtype)[0]

    def _sdn_pallas_tm_fwd(
        q, k, v, g_grouped, beta, seg, decay_last, precision, num_groups: int = 2, state_f32: bool = True, out_dtype=jnp.bfloat16
    ):
      b, t, h, dk = q.shape
      dv = v.shape[-1]
      n = decay_last.shape[2]
      c = t // n
      bh = b * h
      f32, bf16 = jnp.float32, jnp.bfloat16
      s_dtype = f32 if state_f32 else bf16
      q_tm, k_tm, v_tm = q.reshape(b, t, h * dk), k.reshape(b, t, h * dk), v.reshape(b, t, h * dv)
      # g_grouped: [b, t, h, num_groups] -> [bh, n, num_groups * c, 1]
      g_col = (
          g_grouped.reshape(b, n, c, h, num_groups)
          .transpose(0, 3, 1, 4, 2)
          .reshape(bh, n, num_groups * c, 1)
          .astype(f32)
      )
      beta_col = beta.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
      seg_b = seg[:, 0]
      seg_col, seg_row = seg_b[..., None], seg_b[:, :, None, :]
      dl = decay_last.reshape(bh, n, dk, 1).astype(f32)
      per, seg_sp = kda_fused._specs(n, h, reverse=False)
      tok = kda_fused._tok_spec(n, c, h, reverse=False)
      out_tm, states, delta, xk, scores = pl.pallas_call(
          functools.partial(_sdn_grouped_fwd_kernel, precision=precision, num_groups=num_groups, state_f32=state_f32),
          grid=(bh, n),
          in_specs=[tok(dk), tok(dk), tok(dv), per(num_groups * c, 1), per(c, 1), seg_sp(c, 1), seg_sp(1, c), per(dk, 1)],
          out_specs=[tok(dv), per(dk, dv), per(c, dv), per(c, c), per(c, c)],
          out_shape=[
              jax.ShapeDtypeStruct((b, t, h * dv), out_dtype),
              jax.ShapeDtypeStruct((bh, n, dk, dv), s_dtype),
              jax.ShapeDtypeStruct((bh, n, c, dv), bf16),
              jax.ShapeDtypeStruct((bh, n, c, c), f32),
              jax.ShapeDtypeStruct((bh, n, c, c), bf16),
          ],
          scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
          compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl)
      res = (q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, h, dk, dv)
      return out_tm.reshape(b, t, h, dv), res

    def _sdn_pallas_tm_bwd(precision, num_groups, state_f32, out_dtype, res, do):
      del out_dtype
      q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, h, dk, dv = res
      b, t, _ = q_tm.shape
      n = dl.shape[1]
      c = t // n
      bh = b * h
      f32 = jnp.float32
      do_tm = do.reshape(b, t, h * dv)
      per, seg_sp = kda_fused._specs(n, h, reverse=True)
      tok = kda_fused._tok_spec(n, c, h, reverse=True)
      dq_tm, dk_tm, dv_tm, dg_col, dbeta_col, ddl = pl.pallas_call(
          functools.partial(_sdn_grouped_bwd_kernel, precision=precision, num_groups=num_groups, state_f32=state_f32),
          grid=(bh, n),
          in_specs=[
              tok(dk), tok(dk), tok(dv), per(num_groups * c, 1), per(c, 1), seg_sp(c, 1), seg_sp(1, c),
              per(dk, 1), per(dk, dv), per(c, dv), per(c, c), per(c, c), tok(dv),
          ],
          out_specs=[tok(dk), tok(dk), tok(dv), per(num_groups * c, 1), per(c, 1), per(dk, 1)],
          out_shape=[
              jax.ShapeDtypeStruct((b, t, h * dk), q_tm.dtype),
              jax.ShapeDtypeStruct((b, t, h * dk), k_tm.dtype),
              jax.ShapeDtypeStruct((b, t, h * dv), v_tm.dtype),
              jax.ShapeDtypeStruct((bh, n, num_groups * c, 1), f32),
              jax.ShapeDtypeStruct((bh, n, c, 1), f32),
              jax.ShapeDtypeStruct((bh, n, dk, 1), f32),
          ],
          scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
          compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, do_tm)
      dq = dq_tm.reshape(b, t, h, dk)
      dk_o = dk_tm.reshape(b, t, h, dk)
      dv_o = dv_tm.reshape(b, t, h, dv)
      dg = dg_col.reshape(b, h, n, num_groups, c).transpose(0, 2, 4, 1, 3).reshape(b, t, h, num_groups)
      dbeta = dbeta_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h)
      return dq, dk_o, dv_o, dg, dbeta, jnp.zeros((b, 1, n, c), jax.dtypes.float0), ddl.reshape(b, h, n, dk)

    sdn_pallas_fused_token_major.defvjp(_sdn_pallas_tm_fwd, _sdn_pallas_tm_bwd)

    def run_sdn_pallas_fused_tile_decay(
        q, k, v, log_decay_grouped, beta, resets, chunk_size: int = 128, num_groups: int = 2, state_dtype: str = "float32"
    ):
      b, t, h, dk = q.shape
      c = chunk_size
      n = t // c
      g_dim = dk // num_groups
      f32 = jnp.float32
      log_decay_grouped = log_decay_grouped.astype(f32)
      beta = beta.astype(f32)
      seg = jnp.cumsum(resets.reshape(b, n, c).astype(jnp.int32), axis=-1)[:, None]
      any_reset = seg[..., -1] > 0
      g_sum = jnp.sum(log_decay_grouped.reshape(b, n, c, h, num_groups).transpose(0, 3, 1, 2, 4), axis=-2)
      decay_last = jnp.repeat(jnp.where(any_reset[..., None], 0.0, jnp.exp(g_sum)), g_dim, axis=-1)
      return sdn_pallas_fused_token_major(
          q, k, v, log_decay_grouped, beta, seg, decay_last, jax.lax.Precision.HIGHEST, num_groups, state_dtype == "float32", jnp.bfloat16
      )

    # 3) Co-Designed Grouped-Value Attention (GVA-GDN & GVA-SDN): Fused Head-Group Pallas Kernel
    def run_gva_gdn_pallas_fused(
        q_k, k_k, v_v, log_decay_k, beta_k, resets, chunk_size: int = 128, state_dtype: str = "float32"
    ):
      """Runs GDN on H_k key/query heads and H_v = G_kv * H_k value heads in grid=(B*H_k, N) by packing G_kv value heads into dv_eff!"""
      b, t, hk, _ = q_k.shape
      hv, dv = v_v.shape[2], v_v.shape[3]
      v_packed = v_v.reshape(b, t, hk, (hv // hk) * dv)
      out_packed = run_gdn_pallas_fused(q_k, k_k, v_packed, log_decay_k, beta_k, resets, chunk_size, state_dtype)
      return out_packed.reshape(b, t, hv, dv)

    def run_gva_sdn_pallas_fused(
        q_k, k_k, v_v, log_decay_k_grouped, beta_k, resets, chunk_size: int = 128, num_groups: int = 4, state_dtype: str = "float32"
    ):
      """Runs SDN tile-grouped decay on H_k key/query heads and H_v value heads in grid=(B*H_k, N)."""
      b, t, hk, _ = q_k.shape
      hv, dv = v_v.shape[2], v_v.shape[3]
      v_packed = v_v.reshape(b, t, hk, (hv // hk) * dv)
      out_packed = run_sdn_pallas_fused_tile_decay(
          q_k, k_k, v_packed, log_decay_k_grouped, beta_k, resets, chunk_size, num_groups, state_dtype
      )
      return out_packed.reshape(b, t, hv, dv)

    # -------------------------------------------------------------------------
    # B. Part 1: L1-L14 Core Kernel Sweep + XPlane Hardware Profiling (T=8192)
    # -------------------------------------------------------------------------
    print("\n=== PART 1: L1-L14 Core Kernel Re-Architecting Sweep & XPlane Profiling (B=1, T=8192) ===", flush=True)
    b = 1
    seq_len = 8192
    keys = jax.random.split(jax.random.PRNGKey(101), 8)
    resets = jnp.zeros((b, seq_len), dtype=bool)

    orig_fwd, orig_bwd = kda_fused._fwd_kernel, kda_fused._bwd_kernel
    kda_fused._fwd_kernel = _fwd_kernel_v7x_opt
    kda_fused._bwd_kernel = _bwd_kernel_v7x_opt
    try:
      core_configs = [
          # (label, hk, hv, dk, dv, c, g_groups, impl_type, do_xplane)
          ("L1a_Mamba2_SSD_XLA_32x64x64_C128", 32, 32, 64, 64, 128, 1, "ssd_xla", True),
          ("L1b_Mamba2_SSD_2HeadPack_Pallas_32x64", 16, 16, 128, 128, 128, 1, "ssd_pallas", True),
          ("L1c_SSD256_Pallas_8x256x256_C128", 8, 8, 256, 256, 128, 1, "ssd_pallas", True),
          ("L2a_Qwen3Next_GDN_XLA_16K32V_d128_C64", 32, 32, 128, 128, 64, 1, "gdn_xla", False),
          ("L2b_Qwen3Next_GDN_GVA_Pallas_16K32V_C64", 16, 32, 128, 128, 64, 1, "gva_gdn_pallas", True),
          ("L3_OLMo35_Small_KDA_Pallas_16x128x256_C64", 16, 16, 128, 256, 64, 128, "kda_pallas", True),
          ("L4_Asym_KDA_Pallas_8x256x512_C128", 8, 8, 256, 512, 128, 256, "kda_pallas", False),
          ("L5_Asym_GDN_Pallas_8x256x512_C128", 8, 8, 256, 512, 128, 1, "gdn_pallas", False),
          ("L6_Sym_KDA_Pallas_8x256x256_C128", 8, 8, 256, 256, 128, 256, "kda_pallas", True),
          ("L7_Sym_GDN_Pallas_8x256x256_C128", 8, 8, 256, 256, 128, 1, "gdn_pallas", True),
          ("L8a_SDN256_Pallas_TileDecay_G2_8x256x256", 8, 8, 256, 256, 128, 2, "sdn_pallas_g2", True),
          ("L8b_SDN256_Pallas_TileDecay_G4_8x256x256", 8, 8, 256, 256, 128, 4, "sdn_pallas_g4", True),
          ("L8c_SDN_Asym_Pallas_TileDecay_G4_256x512", 8, 8, 256, 512, 128, 4, "sdn_pallas_g4", False),
          ("L9a_GVA_GDN_Pallas_4K8V_256x256_C128", 4, 8, 256, 256, 128, 1, "gva_gdn_pallas", True),
          ("L9b_GVA_SDN_Pallas_G4_4K8V_256x256_C128", 4, 8, 256, 256, 128, 4, "gva_sdn_pallas_g4", True),
          ("L9c_GVA_GDN_Pallas_4K8V_256x512_C128", 4, 8, 256, 512, 128, 1, "gva_gdn_pallas", False),
          ("L10_GVA4to1_GDN_Pallas_2K8V_256x256_C128", 2, 8, 256, 256, 128, 1, "gva_gdn_pallas", True),
          ("L11_UltraWideKey_GDN_Pallas_4x512x256_C128", 4, 4, 512, 256, 128, 1, "gdn_pallas", True),
          ("L12_SDN256_BF16State_Pallas_8x256x256", 8, 8, 256, 256, 128, 1, "gdn_pallas_bf16", False),
      ]

      for label, hk, hv, dk, dv, c, g_groups, impl_type, do_xplane in core_configs:
        try:
          q_k = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, hk, dk), jnp.float32)) * (dk**-0.5)
          k_k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, hk, dk), jnp.float32))
          v_v = jax.random.normal(keys[2], (b, seq_len, hv, dv), jnp.float32) * 0.1
          g_full = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, hk, dk), jnp.float32) - 2.0)
          beta_k = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, hk), jnp.float32))
          dy = jax.random.normal(keys[5], (b, seq_len, hv, dv), jnp.float32) * 0.1

          if impl_type == "ssd_xla":
            g_arg = jnp.mean(g_full, axis=-1)
            fn = lambda q_, k_, v_, g_, b_, _c=c: _ssd_xla_chunked(q_, k_, v_, g_, b_, resets, _c)
            is_kda = False
          elif impl_type == "ssd_pallas":
            g_arg = jnp.mean(g_full, axis=-1)
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_ssd_pallas_fused(q_, k_, v_, g_, b_, resets, _c, "float32")
            is_kda = False
          elif impl_type == "gdn_xla":
            g_arg = jnp.mean(g_full, axis=-1)
            fn = lambda q_, k_, v_, g_, b_, _c=c: qwen3.jax_chunk_gated_delta_rule(
                q_.astype(jnp.bfloat16), k_.astype(jnp.bfloat16), v_.astype(jnp.bfloat16), g_, b_.astype(jnp.bfloat16), chunk_size=_c, compute_dtype=jnp.bfloat16
            )[0]
            is_kda = False
          elif impl_type == "gdn_pallas":
            g_arg = jnp.mean(g_full, axis=-1)
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, _c, "float32")
            is_kda = False
          elif impl_type == "gdn_pallas_bf16":
            g_arg = jnp.mean(g_full, axis=-1)
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, _c, "bfloat16")
            is_kda = False
          elif impl_type == "kda_pallas":
            g_arg = g_full
            fn = lambda q_, k_, v_, g_, b_, _c=c: kda.chunk_kda(
                q_, k_, v_, g_, b_, resets=resets, chunk_size=_c, state_dtype=jnp.float32,
                layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=0,
            )
            is_kda = True
          elif impl_type == "sdn_pallas_g2":
            g_arg = g_full[..., :2]
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_sdn_pallas_fused_tile_decay(q_, k_, v_, g_, b_, resets, _c, 2, "float32")
            is_kda = False
          elif impl_type == "sdn_pallas_g4":
            g_arg = g_full[..., :4]
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_sdn_pallas_fused_tile_decay(q_, k_, v_, g_, b_, resets, _c, 4, "float32")
            is_kda = False
          elif impl_type == "gva_gdn_pallas":
            g_arg = jnp.mean(g_full, axis=-1)
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_gva_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, _c, "float32")
            is_kda = False
          elif impl_type == "gva_sdn_pallas_g4":
            g_arg = g_full[..., :4]
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_gva_sdn_pallas_fused(q_, k_, v_, g_, b_, resets, _c, 4, "float32")
            is_kda = False
          else:
            raise ValueError(f"Unknown impl_type: {impl_type}")

          roof = compute_core_roofline_analytical(
              b, seq_len, hk, dk, (hv // hk) * dv, c, is_kda=is_kda, state_f32=(impl_type != "gdn_pallas_bf16"),
              state_stride=0 if is_kda else 1, outer_block=32, v7x_stacked_opt=True,
          )
          j_fwd = jax.jit(fn)
          j_fb = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy),
                  argnums=(0, 1, 2, 3, 4),
              )
          )
          t_f, _ = _time_fn(j_fwd, q_k, k_k, v_v, g_arg, beta_k)
          t_fb, _, hlo = _time_and_profile_hlo(j_fb, q_k, k_k, v_v, g_arg, beta_k)
          print(format_perf_vs_roofline(label, t_f, t_fb, roof, hlo), flush=True)
          if do_xplane:
            xp = _profile_xplane(j_fb, q_k, k_k, v_v, g_arg, beta_k, top_k=6)
            print(f"    [XPlane {label}] {xp}", flush=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{label:42s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # C. Part 2: Isolated Sub-Component Re-Architecting Micro-Benchmarks
      #    (Input/Gate Projections, Short Conv1D Variants, Output Norm+Gate+W_O)
      # -----------------------------------------------------------------------
      print("\n=== PART 2: Sub-Component Re-Architecting Micro-Benchmarks on TPU v7x (B=1, T=8192, D=2048) ===", flush=True)
      d_model = 2048
      x_act = jax.random.normal(keys[0], (b, seq_len, d_model), jnp.bfloat16)

      # 1. Input + Gate Projection Variants at D=2048
      w_2stage_in = jax.random.normal(keys[1], (d_model, 9224), jnp.bfloat16) * 0.02
      w_f2 = jax.random.normal(keys[2], (512, 2048), jnp.bfloat16) * 0.02
      w_g2 = jax.random.normal(keys[3], (512, 4096), jnp.bfloat16) * 0.02
      w_1stage_asym_gdn = jax.random.normal(keys[4], (d_model, 12544), jnp.bfloat16) * 0.02  # Q:2048, K:2048, V:4096, Gate:4096, g/b:256-pad
      w_1stage_sym_sdn = jax.random.normal(keys[5], (d_model, 8448), jnp.bfloat16) * 0.02   # Q:2048, K:2048, V:2048, Gate:2048, g(G=4)+b:256-pad
      w_1stage_gva_sdn = jax.random.normal(keys[6], (d_model, 6400), jnp.bfloat16) * 0.02   # Q:1024, K:1024, V:2048, Gate:2048, g(G=4)+b:256-pad

      def _p_2stage(x_, w_in_, wf2_, wg2_):
        all_in = jnp.dot(x_, w_in_)
        q_, k_, v_, f1_, g1_, b_ = jnp.split(all_in, [2048, 4096, 8192, 8704, 9216], axis=-1)
        f_ = jnp.dot(f1_, wf2_)
        g_ = jax.nn.sigmoid(jnp.dot(g1_, wg2_))
        return jnp.sum(q_) + jnp.sum(k_) + jnp.sum(v_ * g_) + jnp.sum(f_) + jnp.sum(b_)

      def _p_1stage(x_, w_all_, qk_dim, v_dim):
        all_in = jnp.dot(x_, w_all_)
        q_, k_, v_, g_, fb_ = jnp.split(all_in, [qk_dim, 2 * qk_dim, 2 * qk_dim + v_dim, 2 * qk_dim + 2 * v_dim], axis=-1)
        return jnp.sum(q_) + jnp.sum(k_) + jnp.sum(v_ * jax.nn.sigmoid(g_)) + jnp.sum(fb_)

      for p_label, p_fn, p_args in (
          ("Proj_1_TwoStage_LowRank_Asym (9224+f2+g2)", _p_2stage, (x_act, w_2stage_in, w_f2, w_g2)),
          ("Proj_2_SingleStage_Asym_GDN (D=2048->12544)", lambda x_, w_: _p_1stage(x_, w_, 2048, 4096), (x_act, w_1stage_asym_gdn)),
          ("Proj_3_SingleStage_Sym_SDN256 (D=2048->8448)", lambda x_, w_: _p_1stage(x_, w_, 2048, 2048), (x_act, w_1stage_sym_sdn)),
          ("Proj_4_SingleStage_GVA_SDN256 (D=2048->6400)", lambda x_, w_: _p_1stage(x_, w_, 1024, 2048), (x_act, w_1stage_gva_sdn)),
      ):
        j_f = jax.jit(p_fn)
        j_fb = jax.jit(jax.value_and_grad(p_fn, argnums=tuple(range(len(p_args)))))
        t_f, _ = _time_fn(j_f, *p_args)
        t_fb, _, hlo = _time_and_profile_hlo(j_fb, *p_args)
        print(
            f"[{p_label:46s}] fwd={t_f:6.3f} ms | bwd={t_fb-t_f:6.3f} ms | fwd+bwd={t_fb:6.3f} ms (x36={36*t_fb:6.1f} ms) | HLO={hlo}",
            flush=True,
        )

      # 2. Short Conv1D (K=4) + SiLU + L2Norm Re-Architecting Variants
      q_2048 = jax.random.normal(keys[0], (b, seq_len, 2048), jnp.bfloat16)
      k_2048 = jax.random.normal(keys[1], (b, seq_len, 2048), jnp.bfloat16)
      v_4096 = jax.random.normal(keys[2], (b, seq_len, 4096), jnp.bfloat16)
      v_2048 = jax.random.normal(keys[3], (b, seq_len, 2048), jnp.bfloat16)
      q_1024 = jax.random.normal(keys[4], (b, seq_len, 1024), jnp.bfloat16)
      k_1024 = jax.random.normal(keys[5], (b, seq_len, 1024), jnp.bfloat16)
      wc_8192_f32 = jax.random.normal(keys[6], (8192, 4), jnp.float32) * 0.02
      wc_8192_bf16 = wc_8192_f32.astype(jnp.bfloat16)
      wc_2048 = wc_8192_bf16[:2048]
      wc_4096 = wc_8192_bf16[:4096]
      wc_1024 = wc_8192_bf16[:1024]

      def _conv_concat_qkv(q_, k_, v_, wc_):
        qkv = jnp.concatenate([q_, k_, v_], axis=-1)
        out = olmoe3.causal_depthwise_conv(qkv, wc_, None)
        qo, ko, vo = jnp.split(out, [q_.shape[-1], 2 * q_.shape[-1]], axis=-1)
        return jnp.sum(olmoe3._l2_normalize(qo) + olmoe3._l2_normalize(ko)) + jnp.sum(vo)

      def _conv_sep_qkv(q_, k_, v_, wq_, wk_, wv_):
        qo = olmoe3._l2_normalize(olmoe3.causal_depthwise_conv(q_, wq_, None))
        ko = olmoe3._l2_normalize(olmoe3.causal_depthwise_conv(k_, wk_, None))
        vo = olmoe3.causal_depthwise_conv(v_, wv_, None)
        return jnp.sum(qo + ko) + jnp.sum(vo)

      def _conv_qk_only(q_, k_, v_, wq_, wk_):
        qo = olmoe3._l2_normalize(olmoe3.causal_depthwise_conv(q_, wq_, None))
        ko = olmoe3._l2_normalize(olmoe3.causal_depthwise_conv(k_, wk_, None))
        vo = jax.nn.silu(v_)
        return jnp.sum(qo + ko) + jnp.sum(vo)

      for c_label, c_fn, c_args in (
          ("Conv_1_Concat_QKV_FP32 (8192 ch)", _conv_concat_qkv, (q_2048, k_2048, v_4096, wc_8192_f32)),
          ("Conv_2_Concat_QKV_BF16 (8192 ch)", _conv_concat_qkv, (q_2048, k_2048, v_4096, wc_8192_bf16)),
          ("Conv_3_Separate_QKV_BF16_Asym (2048+2048+4096)", _conv_sep_qkv, (q_2048, k_2048, v_4096, wc_2048, wc_2048, wc_4096)),
          ("Conv_4_Separate_QKV_BF16_Sym (2048+2048+2048)", _conv_sep_qkv, (q_2048, k_2048, v_2048, wc_2048, wc_2048, wc_2048)),
          ("Conv_5_QK_Only_BF16_Sym (2048+2048, SiLU V)", _conv_qk_only, (q_2048, k_2048, v_2048, wc_2048, wc_2048)),
          ("Conv_6_GVA_QK_Only_BF16 (1024+1024, SiLU V)", _conv_qk_only, (q_1024, k_1024, v_2048, wc_1024, wc_1024)),
      ):
        j_f = jax.jit(c_fn)
        j_fb = jax.jit(jax.value_and_grad(c_fn, argnums=tuple(range(len(c_args)))))
        t_f, _ = _time_fn(j_f, *c_args)
        t_fb, _, hlo = _time_and_profile_hlo(j_fb, *c_args)
        print(
            f"[{c_label:46s}] fwd={t_f:6.3f} ms | bwd={t_fb-t_f:6.3f} ms | fwd+bwd={t_fb:6.3f} ms (x36={36*t_fb:6.1f} ms) | HLO={hlo}",
            flush=True,
        )

      # -----------------------------------------------------------------------
      # D. Part 3: Full End-to-End Mixer Layer Re-Architecting Sweep & XPlane Breakdown
      #    (B=1/core, T=8192, D=2048 across FL1-FL9)
      # -----------------------------------------------------------------------
      print("\n=== PART 3: Full End-to-End Mixer Layer Re-Architecting Sweep & XPlane Breakdown (B=1, T=8192, D=2048) ===", flush=True)
      dy_layer = jax.random.normal(keys[7], (b, seq_len, d_model), jnp.float32) * 0.1

      def _make_full_layer_fn(hk, hv, dk, dv, c, impl_type, one_stage_proj: bool = True, qk_only_conv: bool = False):
        k_width = hk * dk
        v_width = hv * dv
        g_groups = 4 if "sdn" in impl_type else (dk if "kda" in impl_type else 1)
        g_width = hk * g_groups
        b_width = hk
        fb_pad = _ceil_div(g_width + b_width, 256) * 256

        if one_stage_proj:
          w_in = jax.random.normal(keys[1], (d_model, 2 * k_width + 2 * v_width + fb_pad), jnp.bfloat16) * 0.02
          w_f2, w_g2 = None, None
        else:
          w_in = jax.random.normal(keys[1], (d_model, 2 * k_width + v_width + 2 * dv + b_width), jnp.bfloat16) * 0.02
          w_f2 = jax.random.normal(keys[2], (dv, g_width), jnp.bfloat16) * 0.02
          w_g2 = jax.random.normal(keys[3], (dv, v_width), jnp.bfloat16) * 0.02

        wq_c = jax.random.normal(keys[4], (k_width, 4), jnp.bfloat16) * 0.02
        wk_c = jax.random.normal(keys[5], (k_width, 4), jnp.bfloat16) * 0.02
        wv_c = jax.random.normal(keys[6], (v_width, 4), jnp.bfloat16) * 0.02
        w_norm = jnp.ones((hv, dv), jnp.bfloat16)
        w_out = jax.random.normal(keys[7], (v_width, d_model), jnp.bfloat16) * 0.02

        def _layer_fwd(x_, w_in_, wq_c_, wk_c_, wv_c_, w_norm_, w_out_, wf2_=w_f2, wg2_=w_g2):
          all_in = jnp.dot(x_, w_in_)
          if one_stage_proj:
            q_in, k_in, v_in, g_out_raw, fb_in = jnp.split(
                all_in, [k_width, 2 * k_width, 2 * k_width + v_width, 2 * k_width + 2 * v_width], axis=-1
            )
            f_raw = fb_in[..., :g_width]
            b_raw = fb_in[..., g_width : g_width + b_width]
            gate = jax.nn.sigmoid(g_out_raw).reshape(b, seq_len, hv, dv)
          else:
            q_in, k_in, v_in, f1_in, g1_in, b_raw = jnp.split(
                all_in, [k_width, 2 * k_width, 2 * k_width + v_width, 2 * k_width + v_width + dv, 2 * k_width + v_width + 2 * dv], axis=-1
            )
            f_raw = jnp.dot(f1_in, wf2_)
            gate = jax.nn.sigmoid(jnp.dot(g1_in, wg2_)).reshape(b, seq_len, hv, dv)

          q_c = olmoe3.causal_depthwise_conv(q_in, wq_c_, None).reshape(b, seq_len, hk, dk)
          k_c = olmoe3.causal_depthwise_conv(k_in, wk_c_, None).reshape(b, seq_len, hk, dk)
          if qk_only_conv:
            v_c = jax.nn.silu(v_in).reshape(b, seq_len, hv, dv)
          else:
            v_c = olmoe3.causal_depthwise_conv(v_in, wv_c_, None).reshape(b, seq_len, hv, dv)

          q_n = olmoe3._l2_normalize(q_c.astype(jnp.float32)) * (dk**-0.5)
          k_n = olmoe3._l2_normalize(k_c.astype(jnp.float32))
          beta_n = 2.0 * jax.nn.sigmoid(b_raw.astype(jnp.float32))

          if impl_type == "ssd_xla":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk) - 2.0)
            core_out = _ssd_xla_chunked(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c)
          elif impl_type == "gdn_xla":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk) - 2.0)
            core_out = qwen3.jax_chunk_gated_delta_rule(
                q_n.astype(jnp.bfloat16), k_n.astype(jnp.bfloat16), v_c.astype(jnp.bfloat16),
                g_n, beta_n.astype(jnp.bfloat16), chunk_size=c, compute_dtype=jnp.bfloat16
            )[0]
          elif impl_type == "kda_pallas":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, dk) - 2.0)
            core_out = kda.chunk_kda(
                q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets=resets, chunk_size=c,
                state_dtype=jnp.float32, layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=0,
            )
          elif impl_type == "gdn_pallas":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk) - 2.0)
            core_out = run_gdn_pallas_fused(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "float32")
          elif impl_type == "sdn_pallas_g4":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, 4) - 2.0)
            core_out = run_sdn_pallas_fused_tile_decay(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, 4, "float32")
          elif impl_type == "gva_gdn_pallas":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk) - 2.0)
            core_out = run_gva_gdn_pallas_fused(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "float32")
          elif impl_type == "gva_sdn_pallas_g4":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, 4) - 2.0)
            core_out = run_gva_sdn_pallas_fused(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, 4, "float32")
          else:
            raise ValueError(f"Unknown impl_type: {impl_type}")

          co_f32 = core_out.astype(jnp.float32)
          rms = jax.lax.rsqrt(jnp.mean(co_f32 * co_f32, axis=-1, keepdims=True) + 1e-5)
          normed = (co_f32 * rms).astype(jnp.bfloat16) * w_norm_[None, None, :, :] * gate
          return jnp.dot(normed.reshape(b, seq_len, v_width), w_out_)

        params = (x_act, w_in, wq_c, wk_c, wv_c, w_norm, w_out) if one_stage_proj else (x_act, w_in, wq_c, wk_c, wv_c, w_norm, w_out, w_f2, w_g2)
        return _layer_fwd, params

      full_layer_arms = [
          ("FL1_Mamba2_SSD_32x64x64_C128", 32, 32, 64, 64, 128, "ssd_xla", True, False, True),
          ("FL2_Qwen3Next_GDN_16K32V_128x128_C64", 16, 32, 128, 128, 64, "gva_gdn_pallas", True, False, True),
          ("FL3_OLMo35_Small_KDA_16x128x256_C64", 16, 16, 128, 256, 64, "kda_pallas", False, False, True),
          ("FL4_Asym_KDA_2Stage_8x256x512_C128", 8, 8, 256, 512, 128, "kda_pallas", False, False, True),
          ("FL5_Asym_GDN_1Stage_8x256x512_C128", 8, 8, 256, 512, 128, "gdn_pallas", True, False, True),
          ("FL6_Sym_KDA_2Stage_8x256x256_C128", 8, 8, 256, 256, 128, "kda_pallas", False, False, False),
          ("FL7_Sym_GDN_1Stage_8x256x256_C128", 8, 8, 256, 256, 128, "gdn_pallas", True, False, True),
          ("FL8a_SDN256_G4_1Stage_QKVConv_8x256x256", 8, 8, 256, 256, 128, "sdn_pallas_g4", True, False, True),
          ("FL8b_SDN256_G4_1Stage_QKOnlyConv_8x256x256", 8, 8, 256, 256, 128, "sdn_pallas_g4", True, True, True),
          ("FL9a_GVA_GDN_1Stage_QKVConv_4K8V_256x256", 4, 8, 256, 256, 128, "gva_gdn_pallas", True, False, True),
          ("FL9b_GVA_SDN256_G4_QKOnlyConv_4K8V_256x256", 4, 8, 256, 256, 128, "gva_sdn_pallas_g4", True, True, True),
          ("FL10_GVA4to1_GDN_QKOnlyConv_2K8V_256x256", 2, 8, 256, 256, 128, "gva_gdn_pallas", True, True, True),
      ]

      for fl_label, hk, hv, dk, dv, c, impl_type, one_stage, qk_only, do_xp in full_layer_arms:
        try:
          l_fn, l_params = _make_full_layer_fn(hk, hv, dk, dv, c, impl_type, one_stage, qk_only)
          j_f = jax.jit(l_fn)
          j_fb = jax.jit(
              jax.value_and_grad(
                  lambda *args, _f=l_fn: jnp.sum(_f(*args).astype(jnp.float32) * dy_layer),
                  argnums=tuple(range(len(l_params))),
              )
          )
          t_f, _ = _time_fn(j_f, *l_params)
          t_fb, _, hlo = _time_and_profile_hlo(j_fb, *l_params)
          print(
              f"[{fl_label:46s}] fwd={t_f:6.3f} ms | bwd={t_fb-t_f:6.3f} ms | "
              f"fwd+bwd={t_fb:6.3f} ms (x36={36*t_fb:6.1f} ms) | temp_MB={hlo['temp_mb']:6.1f} | HLO={hlo}",
              flush=True,
          )
          if do_xp:
            xp = _profile_xplane(j_fb, *l_params, top_k=8)
            print(f"    [XPlane {fl_label}] {xp}", flush=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{fl_label:46s}] failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
      kda_fused._fwd_kernel = orig_fwd
      kda_fused._bwd_kernel = orig_bwd

  def test_6_three_tier_iso_param_decode_and_convergence_ablations(self):
    """Shard 6: 3-Tier Gains Waterfall, Iso-Param GVA, Non-Core Fusions, Decode/Spec-Dec & Multi-Step Convergence on TPU v7x."""
    print(
        f"\n================================================================================"
        f"\nSHARD 6: TPU v7x ({self.num_devs}x {self.dev_kind}) 3-TIER WATERFALL, ISO-PARAM GVA, DECODE & CONVERGENCE SUITE"
        f"\n================================================================================",
        flush=True,
    )

    orig_fwd, orig_bwd = kda_fused._fwd_kernel, kda_fused._bwd_kernel
    kda_fused._fwd_kernel = _fwd_kernel_v7x_opt
    kda_fused._bwd_kernel = _bwd_kernel_v7x_opt

    try:
      # -----------------------------------------------------------------------
      # Helper kernels: Fixed SSD Pallas + SDN Tile-Grouped Decay + GVA-SDN
      # -----------------------------------------------------------------------
      def _ssd_fwd_kernel(*refs, state_f32):
        ins, outs, scratch = refs[:7], refs[7:10], refs[10:]
        q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, dl_ref = (kda_fused._view(r) for r in ins)
        out_ref, st_ref, sc_out = (kda_fused._view(r) for r in outs)
        s_ref = scratch[0]
        f32, bf16 = jnp.float32, jnp.bfloat16
        c = k_ref.shape[0]
        row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
        causal = row >= col
        tri = kda_fused._tril_ones(c)
        g_in = g_ref[...].astype(f32)
        g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
        exp_g_diff = jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf))
        qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
        q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)
        scores_bf = (jnp.dot(q_bf, jnp.transpose(k_bf), preferred_element_type=f32) * exp_g_diff).astype(bf16)

        @pl.when(pl.program_id(1) == 0)
        def _():
          s_ref[...] = jnp.zeros_like(s_ref)

        seg_c = segc_ref[...]
        fp = (seg_c == 0).astype(f32)
        eg = jnp.exp(g_cum)
        ecar = jnp.exp(g_cum[c - 1 : c, :] - g_cum)
        beta = beta_ref[...].astype(f32)
        u_bf = (v_ref[...].astype(f32) * beta).astype(bf16)
        qg = (qf * eg * fp).astype(bf16)
        kc = (kf * ecar).astype(bf16)
        s = s_ref[...]
        s_bf = s.astype(bf16)
        st_ref[...] = s if state_f32 else s_bf
        sc_out[...] = scores_bf
        out = jnp.dot(qg, s_bf, preferred_element_type=f32) + jnp.dot(scores_bf, u_bf, preferred_element_type=f32)
        out_ref[...] = out.astype(out_ref.dtype)
        new = s * dl_ref[...] + kda_fused._tn(kc, u_bf)
        s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)

      def _ssd_bwd_kernel(*refs, state_f32):
        ins, outs, scratch = refs[:10], refs[10:16], refs[16:]
        q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, dl_ref, st_ref, sc_ref, do_ref = (
            kda_fused._view(r) for r in ins
        )
        dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref, ddl_ref = (kda_fused._view(r) for r in outs)
        r_ref = scratch[0]

        @pl.when(pl.program_id(1) == 0)
        def _():
          r_ref[...] = jnp.zeros_like(r_ref)

        f32, bf16 = jnp.float32, jnp.bfloat16
        c = k_ref.shape[0]
        row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
        causal = row >= col
        tri = kda_fused._tril_ones(c)
        g_in = g_ref[...].astype(f32)
        g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
        exp_g_diff = jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf))
        qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
        seg_c = segc_ref[...]
        fp = (seg_c == 0).astype(f32)
        eg = jnp.exp(g_cum)
        ecar = jnp.exp(g_cum[c - 1 : c, :] - g_cum)
        beta = beta_ref[...].astype(f32)
        u_bf = (v_ref[...].astype(f32) * beta).astype(bf16)
        qg = (qf * eg * fp).astype(bf16)
        kc = (kf * ecar).astype(bf16)
        states_f = st_ref[...].astype(f32)
        states = states_f.astype(bf16)
        scores_bf = sc_ref[...]
        d_out = do_ref[...].astype(bf16)
        r = r_ref[...]
        r_bf = r.astype(bf16)

        dsc = kda_fused._nt(d_out, u_bf)
        du = kda_fused._tn(scores_bf, d_out) + jnp.dot(kc, r_bf, preferred_element_type=f32)
        dkc = kda_fused._nt(u_bf, r_bf)
        ddl_ref[...] = jnp.sum(states_f * r, axis=-1, keepdims=True)
        dqg = kda_fused._nt(d_out, states)
        r_ref[...] = r * dl_ref[...] + kda_fused._tn(qg, d_out)

        ds_scaled = (jnp.where(causal, dsc, 0.0) * exp_g_diff).astype(bf16)
        q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)
        dq_i = jnp.dot(ds_scaled, k_bf, preferred_element_type=f32)
        dk_col = jnp.dot(jnp.transpose(ds_scaled), q_bf, preferred_element_type=f32)
        dg_i = jnp.sum(qf * dq_i - kf * dk_col, axis=-1, keepdims=True)

        dv_ref[...] = (du * beta).astype(dv_ref.dtype)
        dbeta_ref[...] = jnp.sum(du * v_ref[...].astype(f32), axis=1, keepdims=True)
        dq_ref[...] = (dq_i + dqg * eg * fp).astype(dq_ref.dtype)
        dk_ref[...] = (dk_col + dkc * ecar).astype(dk_ref.dtype)
        carry = dkc * kf * ecar
        dg_wy = jnp.sum(dqg * qf * eg * fp - carry, axis=-1, keepdims=True)
        dg_last = jnp.sum(carry, axis=(0, 1), keepdims=True)
        dg = dg_i + dg_wy + jnp.where(row == c - 1, dg_last, 0.0)
        dg_ref[...] = jnp.sum(jnp.transpose(tri) * jnp.transpose(dg), axis=1, keepdims=True)

      @functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
      def ssd_pallas_tm(q, k, v, g_scalar, beta, seg, decay_last, state_f32: bool = True, out_dtype=jnp.bfloat16):
        return _ssd_tm_fwd(q, k, v, g_scalar, beta, seg, decay_last, state_f32, out_dtype)[0]

      def _ssd_tm_fwd(q, k, v, g_scalar, beta, seg, decay_last, state_f32: bool = True, out_dtype=jnp.bfloat16):
        b, t, h, dk = q.shape
        dv = v.shape[-1]
        n = decay_last.shape[2]
        c = t // n
        bh = b * h
        f32, bf16 = jnp.float32, jnp.bfloat16
        s_dtype = f32 if state_f32 else bf16
        q_tm, k_tm, v_tm = q.reshape(b, t, h * dk), k.reshape(b, t, h * dk), v.reshape(b, t, h * dv)
        g_col = g_scalar.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
        beta_col = beta.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
        seg_col = seg[:, 0][..., None]
        dl = decay_last.reshape(bh, n, dk, 1).astype(f32)
        per, seg_sp = kda_fused._specs(n, h, reverse=False)
        tok = kda_fused._tok_spec(n, c, h, reverse=False)
        out_tm, states, scores = pl.pallas_call(
            functools.partial(_ssd_fwd_kernel, state_f32=state_f32),
            grid=(bh, n),
            in_specs=[tok(dk), tok(dk), tok(dv), per(c, 1), per(c, 1), seg_sp(c, 1), per(dk, 1)],
            out_specs=[tok(dv), per(dk, dv), per(c, c)],
            out_shape=[
                jax.ShapeDtypeStruct((b, t, h * dv), out_dtype),
                jax.ShapeDtypeStruct((bh, n, dk, dv), s_dtype),
                jax.ShapeDtypeStruct((bh, n, c, c), bf16),
            ],
            scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl)
        return out_tm.reshape(b, t, h, dv), (q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl, states, scores, h, dk, dv)

      def _ssd_tm_bwd(state_f32, out_dtype, res, do):
        del out_dtype
        q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl, states, scores, h, dk, dv = res
        b, t, _ = q_tm.shape
        n = dl.shape[1]
        c = t // n
        bh = b * h
        f32 = jnp.float32
        do_tm = do.reshape(b, t, h * dv)
        per, seg_sp = kda_fused._specs(n, h, reverse=True)
        tok = kda_fused._tok_spec(n, c, h, reverse=True)
        dq_tm, dk_tm, dv_tm, dg_col, dbeta_col, ddl = pl.pallas_call(
            functools.partial(_ssd_bwd_kernel, state_f32=state_f32),
            grid=(bh, n),
            in_specs=[tok(dk), tok(dk), tok(dv), per(c, 1), per(c, 1), seg_sp(c, 1), per(dk, 1), per(dk, dv), per(c, c), tok(dv)],
            out_specs=[tok(dk), tok(dk), tok(dv), per(c, 1), per(c, 1), per(dk, 1)],
            out_shape=[
                jax.ShapeDtypeStruct((b, t, h * dk), q_tm.dtype),
                jax.ShapeDtypeStruct((b, t, h * dk), k_tm.dtype),
                jax.ShapeDtypeStruct((b, t, h * dv), v_tm.dtype),
                jax.ShapeDtypeStruct((bh, n, c, 1), f32),
                jax.ShapeDtypeStruct((bh, n, c, 1), f32),
                jax.ShapeDtypeStruct((bh, n, dk, 1), f32),
            ],
            scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, dl, states, scores, do_tm)
        dq = dq_tm.reshape(b, t, h, dk)
        dk_o = dk_tm.reshape(b, t, h, dk)
        dv_o = dv_tm.reshape(b, t, h, dv)
        dg = dg_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h, 1)
        dbeta = dbeta_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h)
        return dq, dk_o, dv_o, dg, dbeta, jnp.zeros((b, 1, n, c), jax.dtypes.float0), ddl.reshape(b, h, n, dk)

      ssd_pallas_tm.defvjp(_ssd_tm_fwd, _ssd_tm_bwd)

      def run_ssd_pallas(q, k, v, log_decay_scalar, beta, resets, chunk_size: int = 128, state_dtype: str = "float32"):
        b, t, h, dk = q.shape
        c = chunk_size
        n = t // c
        f32 = jnp.float32
        log_decay_scalar = log_decay_scalar.astype(f32)
        beta = beta.astype(f32)
        seg = jnp.cumsum(resets.reshape(b, n, c).astype(jnp.int32), axis=-1)[:, None]
        any_reset = seg[..., -1] > 0
        g_sum = jnp.sum(log_decay_scalar.reshape(b, n, c, h).transpose(0, 3, 1, 2), axis=-1)
        decay_last = jnp.broadcast_to(jnp.where(any_reset, 0.0, jnp.exp(g_sum))[..., None], (b, h, n, dk))
        return ssd_pallas_tm(q, k, v, log_decay_scalar[..., None], beta, seg, decay_last, state_dtype == "float32", jnp.bfloat16)

      def run_gva_gdn_pallas(q_k, k_k, v_v, log_decay_k, beta_k, resets, chunk_size: int = 128, state_dtype: str = "float32"):
        b, t, hk, _ = q_k.shape
        hv, dv = v_v.shape[2], v_v.shape[3]
        v_packed = v_v.reshape(b, t, hk, (hv // hk) * dv)
        out_packed = run_gdn_pallas_fused(q_k, k_k, v_packed, log_decay_k, beta_k, resets, chunk_size, state_dtype)
        return out_packed.reshape(b, t, hv, dv)

      # Reuse SDN kernel from test_5 via clean wrapper
      def _sdn_g4_fwd_kernel(*refs, precision, state_f32: bool):
        ins, outs, scratch = refs[:8], refs[8:13], refs[13:]
        q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref = (kda_fused._view(r) for r in ins)
        out_ref, st_ref, delta_out, xk_out, sc_out = (kda_fused._view(r) for r in outs)
        s_ref = scratch[0]
        f32, bf16 = jnp.float32, jnp.bfloat16
        c = k_ref.shape[0]
        dk = k_ref.shape[1]
        half_dk = dk // 2
        qtr_dk = dk // 4
        dv = v_ref.shape[1]
        row, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
        col_half = jax.lax.broadcasted_iota(jnp.int32, (1, half_dk), 1)
        tri = kda_fused._tril_ones(c)
        seg_c = segc_ref[...]
        fp = (seg_c == 0).astype(f32)
        last_seg = seg_c == seg_c[c - 1 : c, :]
        qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
        q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)
        scores = jnp.zeros((c, c), f32)
        kk = jnp.zeros((c, c), f32)
        eg_halves, ecar_halves = [], []
        for h_idx in range(2):
          q_half = q_bf[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          k_half = k_bf[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          eg_pair, ecar_pair = [], []
          for sub_idx in range(2):
            g_idx = h_idx * 2 + sub_idx
            g_in = g_ref[g_idx * c : (g_idx + 1) * c, :].astype(f32)
            g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
            exp_g_diff = jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf))
            mask_qtr = (col_half < qtr_dk) if sub_idx == 0 else (col_half >= qtr_dk)
            q_g = jnp.where(mask_qtr, q_half, jnp.zeros_like(q_half))
            k_g = jnp.where(mask_qtr, k_half, jnp.zeros_like(k_half))
            qk_g = jnp.concatenate([q_g, k_g], axis=0)
            dots_g = jnp.dot(qk_g, jnp.transpose(k_g), preferred_element_type=f32)
            scores = scores + dots_g[:c, :] * exp_g_diff
            kk = kk + dots_g[c:, :] * exp_g_diff
            eg_pair.append(jnp.exp(g_cum))
            ecar_pair.append(jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf)))
          eg_halves.append(jnp.where(col_half < qtr_dk, eg_pair[0], eg_pair[1]))
          ecar_halves.append(jnp.where(col_half < qtr_dk, ecar_pair[0], ecar_pair[1]))
        eg = jnp.concatenate(eg_halves, axis=1)
        ecar = jnp.concatenate(ecar_halves, axis=1)
        kk = jnp.where(strict, kk, 0.0)
        beta = beta_ref[...].astype(f32)
        x = kda_fused._inverse(kk * beta, (row == col).astype(f32), precision)
        scores_bf = scores.astype(bf16)

        @pl.when(pl.program_id(1) == 0)
        def _():
          s_ref[...] = jnp.zeros_like(s_ref)

        vb = (v_ref[...].astype(f32) * beta).astype(bf16)
        kb = (kf * eg * beta * fp).astype(bf16)
        qg = (qf * eg * fp).astype(bf16)
        kc = (kf * ecar).astype(bf16)
        x_bf = x.astype(bf16)
        s = s_ref[...]
        s_bf = s.astype(bf16)
        st_ref[...] = s if state_f32 else s_bf
        vb_kb = jnp.concatenate([vb, kb], axis=1)
        uw = jnp.dot(x_bf, vb_kb, preferred_element_type=f32)
        u, w = uw[:, :dv], uw[:, dv:]
        wqg = jnp.concatenate([w.astype(bf16), qg], axis=0)
        ws_qgs = jnp.dot(wqg, s_bf, preferred_element_type=f32)
        ws, qgs = ws_qgs[:c, :], ws_qgs[c:, :]
        delta_bf = (u - ws).astype(bf16)
        delta_out[...] = delta_bf
        xk_out[...] = kda_fused._pack_x_kk(x, kk, row, col)
        sc_out[...] = scores_bf
        out = qgs + jnp.dot(scores_bf, delta_bf, preferred_element_type=f32)
        out_ref[...] = out.astype(out_ref.dtype)
        new = s * dl_ref[...] + kda_fused._tn(kc, delta_bf)
        s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)

      def _sdn_g4_bwd_kernel(*refs, precision, state_f32: bool):
        ins, outs, scratch = refs[:13], refs[13:19], refs[19:]
        q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref, st_ref, de_ref, xk_ref, sc_ref, do_ref = (
            kda_fused._view(r) for r in ins
        )
        dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref, ddl_ref = (kda_fused._view(r) for r in outs)
        r_ref = scratch[0]

        @pl.when(pl.program_id(1) == 0)
        def _():
          r_ref[...] = jnp.zeros_like(r_ref)

        f32, bf16 = jnp.float32, jnp.bfloat16
        c = k_ref.shape[0]
        dk = k_ref.shape[1]
        half_dk = dk // 2
        qtr_dk = dk // 4
        dv = v_ref.shape[1]
        row, causal, strict = kda_fused._masks(segc_ref, segr_ref, c)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
        col_half = jax.lax.broadcasted_iota(jnp.int32, (1, half_dk), 1)
        tri = kda_fused._tril_ones(c)
        seg_c = segc_ref[...]
        fp = (seg_c == 0).astype(f32)
        last_seg = seg_c == seg_c[c - 1 : c, :]
        qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
        q_bf, k_bf = qf.astype(bf16), kf.astype(bf16)
        eg_halves, ecar_halves, exp_diffs = [], [], []
        for h_idx in range(2):
          eg_pair, ecar_pair = [], []
          for sub_idx in range(2):
            g_idx = h_idx * 2 + sub_idx
            g_in = g_ref[g_idx * c : (g_idx + 1) * c, :].astype(f32)
            g_cum = jnp.sum(tri * jnp.transpose(g_in), axis=1, keepdims=True)
            exp_diffs.append(jnp.exp(jnp.where(causal, g_cum - jnp.transpose(g_cum), -jnp.inf)))
            eg_pair.append(jnp.exp(g_cum))
            ecar_pair.append(jnp.exp(jnp.where(last_seg, g_cum[c - 1 : c, :] - g_cum, -jnp.inf)))
          eg_halves.append(jnp.where(col_half < qtr_dk, eg_pair[0], eg_pair[1]))
          ecar_halves.append(jnp.where(col_half < qtr_dk, ecar_pair[0], ecar_pair[1]))
        eg = jnp.concatenate(eg_halves, axis=1)
        ecar = jnp.concatenate(ecar_halves, axis=1)
        beta = beta_ref[...].astype(f32)
        vb = (v_ref[...].astype(f32) * beta).astype(bf16)
        kb = (kf * eg * beta * fp).astype(bf16)
        qg = (qf * eg * fp).astype(bf16)
        kc = (kf * ecar).astype(bf16)
        x, kk = kda_fused._unpack_x_kk(xk_ref[...], row, col)
        x_bf, scores_bf = x.astype(bf16), sc_ref[...]
        states_f, delta = st_ref[...].astype(f32), de_ref[...]
        states = states_f.astype(bf16)
        d_out = do_ref[...].astype(bf16)
        dsc = kda_fused._nt(d_out, delta)
        g_delta = kda_fused._tn(scores_bf, d_out)
        r = r_ref[...]
        r_bf = r.astype(bf16)
        dd = g_delta + jnp.dot(kc, r_bf, preferred_element_type=f32)
        dd_bf = dd.astype(bf16)
        w_bf = jnp.dot(x_bf, kb, preferred_element_type=f32).astype(bf16)
        dkc = kda_fused._nt(delta, r_bf)
        ddl_ref[...] = jnp.sum(states_f * r, axis=-1, keepdims=True)
        do_neg_dd = jnp.concatenate([d_out, -dd_bf], axis=0)
        dqg_dw = kda_fused._nt(do_neg_dd, states)
        dqg = dqg_dw[:c, :]
        dw_bf = dqg_dw[c:, :].astype(bf16)
        qg_w = jnp.concatenate([qg, w_bf], axis=0)
        r_ref[...] = r * dl_ref[...] + kda_fused._tn(qg_w, do_neg_dd)
        dd_dw = jnp.concatenate([dd_bf, dw_bf], axis=1)
        vb_kb = jnp.concatenate([vb, kb], axis=1)
        dx = kda_fused._nt(dd_dw, vb_kb)
        dvb_dkb = kda_fused._tn(x_bf, dd_dw)
        dvb, dkb = dvb_dkb[:, :dv], dvb_dkb[:, dv:]
        dm = -kda_fused._mm(kda_fused._mm(x, dx, precision, ((0,), (0,))), x, precision, ((1,), (1,)))
        dm = jnp.where(strict, dm, 0.0)
        dbeta = jnp.sum(dm * kk, axis=1, keepdims=True)
        dkk = dm * beta
        ds = jnp.where(causal, dsc, 0.0)
        kegfp = kf * eg * fp
        carry = dkc * kf * ecar
        wy_term = dqg * qf * eg * fp + dkb * beta * kegfp - carry
        dq_halves, dk_halves, dg_parts = [], [], []
        for h_idx in range(2):
          q_half = q_bf[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          k_half = k_bf[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          qf_half = qf[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          kf_half = kf[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          wy_half = wy_term[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          car_half = carry[:, h_idx * half_dk : (h_idx + 1) * half_dk]
          dq_pair, dk_pair = [], []
          for sub_idx in range(2):
            g_idx = h_idx * 2 + sub_idx
            exp_g_diff = exp_diffs[g_idx]
            ds_s = (ds * exp_g_diff).astype(bf16)
            dkk_s = (dkk * exp_g_diff).astype(bf16)
            mask_qtr = (col_half < qtr_dk) if sub_idx == 0 else (col_half >= qtr_dk)
            q_g = jnp.where(mask_qtr, q_half, jnp.zeros_like(q_half))
            k_g = jnp.where(mask_qtr, k_half, jnp.zeros_like(k_half))
            qf_g = jnp.where(mask_qtr, qf_half, 0.0)
            kf_g = jnp.where(mask_qtr, kf_half, 0.0)
            dq_g = jnp.dot(ds_s, k_g, preferred_element_type=f32)
            dk_row_g = jnp.dot(dkk_s, k_g, preferred_element_type=f32)
            ds_dkk_t = jnp.concatenate([jnp.transpose(ds_s), jnp.transpose(dkk_s)], axis=1)
            qk_g = jnp.concatenate([q_g, k_g], axis=0)
            dk_col_g = jnp.dot(ds_dkk_t, qk_g, preferred_element_type=f32)
            dq_pair.append(dq_g)
            dk_pair.append(dk_row_g + dk_col_g)
            dg_i_g = jnp.sum(qf_g * dq_g + kf_g * (dk_row_g - dk_col_g), axis=-1, keepdims=True)
            dg_wy_g = jnp.sum(jnp.where(mask_qtr, wy_half, 0.0), axis=-1, keepdims=True)
            dg_last_g = jnp.sum(jnp.where(mask_qtr, car_half, 0.0), axis=(0, 1), keepdims=True)
            dg_g = dg_i_g + dg_wy_g + jnp.where(row == c - 1, dg_last_g, 0.0)
            dg_parts.append(jnp.sum(jnp.transpose(tri) * jnp.transpose(dg_g), axis=1, keepdims=True))
          dq_halves.append(jnp.where(col_half < qtr_dk, dq_pair[0], dq_pair[1]))
          dk_halves.append(jnp.where(col_half < qtr_dk, dk_pair[0], dk_pair[1]))
        dq_i = jnp.concatenate(dq_halves, axis=1)
        dk_i = jnp.concatenate(dk_halves, axis=1)
        dg_out = jnp.concatenate(dg_parts, axis=0)
        dv_ref[...] = (dvb * beta).astype(dv_ref.dtype)
        dbeta_ref[...] = (
            dbeta
            + jnp.sum(dvb * v_ref[...].astype(f32), axis=1, keepdims=True)
            + jnp.sum(dkb * kegfp, axis=1, keepdims=True)
        )
        dq_ref[...] = (dq_i + dqg * eg * fp).astype(dq_ref.dtype)
        dk_ref[...] = (dk_i + dkb * eg * beta * fp + dkc * ecar).astype(dk_ref.dtype)
        dg_ref[...] = dg_out

      @functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
      def sdn_g4_pallas_tm(q, k, v, g_grouped, beta, seg, decay_last, precision, state_f32: bool = True, out_dtype=jnp.bfloat16):
        return _sdn_g4_tm_fwd(q, k, v, g_grouped, beta, seg, decay_last, precision, state_f32, out_dtype)[0]

      def _sdn_g4_tm_fwd(q, k, v, g_grouped, beta, seg, decay_last, precision, state_f32: bool = True, out_dtype=jnp.bfloat16):
        b, t, h, dk = q.shape
        dv = v.shape[-1]
        n = decay_last.shape[2]
        c = t // n
        bh = b * h
        f32, bf16 = jnp.float32, jnp.bfloat16
        s_dtype = f32 if state_f32 else bf16
        q_tm, k_tm, v_tm = q.reshape(b, t, h * dk), k.reshape(b, t, h * dk), v.reshape(b, t, h * dv)
        g_col = g_grouped.reshape(b, n, c, h, 4).transpose(0, 3, 1, 4, 2).reshape(bh, n, 4 * c, 1).astype(f32)
        beta_col = beta.reshape(b, n, c, h).transpose(0, 3, 1, 2).reshape(bh, n, c, 1).astype(f32)
        seg_b = seg[:, 0]
        seg_col, seg_row = seg_b[..., None], seg_b[:, :, None, :]
        dl = decay_last.reshape(bh, n, dk, 1).astype(f32)
        per, seg_sp = kda_fused._specs(n, h, reverse=False)
        tok = kda_fused._tok_spec(n, c, h, reverse=False)
        out_tm, states, delta, xk, scores = pl.pallas_call(
            functools.partial(_sdn_g4_fwd_kernel, precision=precision, state_f32=state_f32),
            grid=(bh, n),
            in_specs=[tok(dk), tok(dk), tok(dv), per(4 * c, 1), per(c, 1), seg_sp(c, 1), seg_sp(1, c), per(dk, 1)],
            out_specs=[tok(dv), per(dk, dv), per(c, dv), per(c, c), per(c, c)],
            out_shape=[
                jax.ShapeDtypeStruct((b, t, h * dv), out_dtype),
                jax.ShapeDtypeStruct((bh, n, dk, dv), s_dtype),
                jax.ShapeDtypeStruct((bh, n, c, dv), bf16),
                jax.ShapeDtypeStruct((bh, n, c, c), f32),
                jax.ShapeDtypeStruct((bh, n, c, c), bf16),
            ],
            scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl)
        res = (q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, h, dk, dv)
        return out_tm.reshape(b, t, h, dv), res

      def _sdn_g4_tm_bwd(precision, state_f32, out_dtype, res, do):
        del out_dtype
        q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, h, dk, dv = res
        b, t, _ = q_tm.shape
        n = dl.shape[1]
        c = t // n
        bh = b * h
        f32 = jnp.float32
        do_tm = do.reshape(b, t, h * dv)
        per, seg_sp = kda_fused._specs(n, h, reverse=True)
        tok = kda_fused._tok_spec(n, c, h, reverse=True)
        dq_tm, dk_tm, dv_tm, dg_col, dbeta_col, ddl = pl.pallas_call(
            functools.partial(_sdn_g4_bwd_kernel, precision=precision, state_f32=state_f32),
            grid=(bh, n),
            in_specs=[
                tok(dk), tok(dk), tok(dv), per(4 * c, 1), per(c, 1), seg_sp(c, 1), seg_sp(1, c),
                per(dk, 1), per(dk, dv), per(c, dv), per(c, c), per(c, c), tok(dv),
            ],
            out_specs=[tok(dk), tok(dk), tok(dv), per(4 * c, 1), per(c, 1), per(dk, 1)],
            out_shape=[
                jax.ShapeDtypeStruct((b, t, h * dk), q_tm.dtype),
                jax.ShapeDtypeStruct((b, t, h * dk), k_tm.dtype),
                jax.ShapeDtypeStruct((b, t, h * dv), v_tm.dtype),
                jax.ShapeDtypeStruct((bh, n, 4 * c, 1), f32),
                jax.ShapeDtypeStruct((bh, n, c, 1), f32),
                jax.ShapeDtypeStruct((bh, n, dk, 1), f32),
            ],
            scratch_shapes=[pltpu.VMEM((dk, dv), f32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        )(q_tm, k_tm, v_tm, g_col, beta_col, seg_col, seg_row, dl, states, delta, xk, scores, do_tm)
        dq = dq_tm.reshape(b, t, h, dk)
        dk_o = dk_tm.reshape(b, t, h, dk)
        dv_o = dv_tm.reshape(b, t, h, dv)
        dg = dg_col.reshape(b, h, n, 4, c).transpose(0, 2, 4, 1, 3).reshape(b, t, h, 4)
        dbeta = dbeta_col.reshape(b, h, n, c).transpose(0, 2, 3, 1).reshape(b, t, h)
        return dq, dk_o, dv_o, dg, dbeta, jnp.zeros((b, 1, n, c), jax.dtypes.float0), ddl.reshape(b, h, n, dk)

      sdn_g4_pallas_tm.defvjp(_sdn_g4_tm_fwd, _sdn_g4_tm_bwd)

      def run_sdn_g4_pallas(q, k, v, log_decay_g4, beta, resets, chunk_size: int = 128, state_dtype: str = "float32"):
        b, t, h, dk = q.shape
        c = chunk_size
        n = t // c
        g_dim = dk // 4
        f32 = jnp.float32
        log_decay_g4 = log_decay_g4.astype(f32)
        beta = beta.astype(f32)
        seg = jnp.cumsum(resets.reshape(b, n, c).astype(jnp.int32), axis=-1)[:, None]
        any_reset = seg[..., -1] > 0
        g_sum = jnp.sum(log_decay_g4.reshape(b, n, c, h, 4).transpose(0, 3, 1, 2, 4), axis=-2)
        decay_last = jnp.repeat(jnp.where(any_reset[..., None], 0.0, jnp.exp(g_sum)), g_dim, axis=-1)
        return sdn_g4_pallas_tm(q, k, v, log_decay_g4, beta, seg, decay_last, jax.lax.Precision.HIGHEST, state_dtype == "float32", jnp.bfloat16)

      def run_gva_sdn_g4_pallas(q_k, k_k, v_v, log_decay_k_g4, beta_k, resets, chunk_size: int = 128, state_dtype: str = "float32"):
        b, t, hk, _ = q_k.shape
        hv, dv = v_v.shape[2], v_v.shape[3]
        v_packed = v_v.reshape(b, t, hk, (hv // hk) * dv)
        out_packed = run_sdn_g4_pallas(q_k, k_k, v_packed, log_decay_k_g4, beta_k, resets, chunk_size, state_dtype)
        return out_packed.reshape(b, t, hv, dv)

      # -----------------------------------------------------------------------
      # PART 1: Fixed SSD Pallas (L1b, L1c) + Iso-Parameter GVA-SDN & L11 Ultra-Wide Key Core Sweep (B=1, T=8192)
      # -----------------------------------------------------------------------
      b = 1
      seq_len = 8192
      keys = jax.random.split(jax.random.PRNGKey(2026), 10)
      resets = jnp.zeros((b, seq_len), dtype=bool)

      core_arms_t6 = [
          ("L11a_Raw_UltraWideKey_SDN_G4_4K4V_512x256", 4, 4, 512, 256, 128, "gva_sdn_g4", True),
          ("L11a_BF16State_UltraWideKey_SDN_4K4V_512x256", 4, 4, 512, 256, 128, "gva_sdn_g4_bf16", False),
          ("L11b_IsoParam_GVA2to1_UltraWideKey_4K8V_512x256", 4, 8, 512, 256, 128, "gva_sdn_g4", True),
          ("L11b_BF16State_GVA2to1_UltraWideKey_4K8V_512x256", 4, 8, 512, 256, 128, "gva_sdn_g4_bf16", False),
          ("L11c_IsoParam_UltraWideKey_WideV384_4K4V_512x384", 4, 4, 512, 384, 128, "gva_sdn_g4", True),
      ]

      for label, hk, hv, dk, dv, c, impl_type, do_xp in core_arms_t6:
        try:
          q_k = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, hk, dk), jnp.float32)) * (dk**-0.5)
          k_k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, hk, dk), jnp.float32))
          v_v = jax.random.normal(keys[2], (b, seq_len, hv, dv), jnp.float32) * 0.1
          g_full = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, hk, dk), jnp.float32) - 2.0)
          beta_k = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, hk), jnp.float32))
          dy = jax.random.normal(keys[5], (b, seq_len, hv, dv), jnp.float32) * 0.1

          if impl_type == "ssd_pallas":
            g_arg = jnp.mean(g_full, axis=-1)
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_ssd_pallas(q_, k_, v_, g_, b_, resets, _c, "float32")
          elif impl_type == "gva_sdn_g4":
            g_arg = g_full[..., :4]
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_gva_sdn_g4_pallas(q_, k_, v_, g_, b_, resets, _c, "float32")
          elif impl_type == "gva_sdn_g4_bf16":
            g_arg = g_full[..., :4]
            fn = lambda q_, k_, v_, g_, b_, _c=c: run_gva_sdn_g4_pallas(q_, k_, v_, g_, b_, resets, _c, "bfloat16")
          else:
            raise ValueError(impl_type)

          roof = compute_core_roofline_analytical(
              b, seq_len, hk, dk, (hv // hk) * dv, c, is_kda=False, state_f32=("bf16" not in impl_type),
              state_stride=1, outer_block=32, v7x_stacked_opt=True,
          )
          j_fwd = jax.jit(fn)
          j_fb = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy),
                  argnums=(0, 1, 2, 3, 4),
              )
          )
          t_f, _ = _time_fn(j_fwd, q_k, k_k, v_v, g_arg, beta_k)
          t_fb, _, hlo = _time_and_profile_hlo(j_fb, q_k, k_k, v_v, g_arg, beta_k)
          print(format_perf_vs_roofline(label, t_f, t_fb, roof, hlo), flush=True)
          if do_xp:
            xp = _profile_xplane(j_fb, q_k, k_k, v_v, g_arg, beta_k, top_k=6)
            print(f"    [XPlane {label}] {xp}", flush=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{label:42s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # PART 1B & PART 2: Full-Layer 3-Tier Waterfall (D=1792 & D=2048), Iso-Parameter GVA,
      #                   and Non-Core 55.8% Fusion Optimizations on TPU v7x
      # -----------------------------------------------------------------------
      print("\n=== PART 1B & PART 2: Full-Layer Iso-Param GVA, Ultra-Wide Key & Non-Core Fusion Sweep on TPU v7x ===", flush=True)

      def _make_layer_fn_v2(
          d_model: int,
          hk: int,
          hv: int,
          dk: int,
          dv: int,
          c: int,
          impl_type: str,
          one_stage_proj: bool = True,
          qk_only_conv: bool = False,
          fuse_qk_conv: bool = False,
          bf16_epilogue: bool = False,
          extra_ffn_dim: int = 0,
      ):
        k_width = hk * dk
        v_width = hv * dv
        g_groups = 4 if "sdn" in impl_type else (dk if "kda" in impl_type else 1)
        g_width = hk * g_groups
        b_width = hk
        fb_pad = _ceil_div(g_width + b_width, 256) * 256

        if one_stage_proj:
          w_in = jax.random.normal(keys[1], (d_model, 2 * k_width + 2 * v_width + fb_pad), jnp.bfloat16) * 0.02
          w_f2, w_g2 = None, None
          proj_params = d_model * (2 * k_width + 2 * v_width + g_width + b_width)
        else:
          w_in = jax.random.normal(keys[1], (d_model, 2 * k_width + v_width + 2 * dv + b_width), jnp.bfloat16) * 0.02
          w_f2 = jax.random.normal(keys[2], (dv, g_width), jnp.bfloat16) * 0.02
          w_g2 = jax.random.normal(keys[3], (dv, v_width), jnp.bfloat16) * 0.02
          proj_params = d_model * (2 * k_width + v_width + 2 * dv + b_width) + dv * (g_width + v_width)

        if fuse_qk_conv:
          wqk_c = jax.random.normal(keys[4], (2 * k_width, 4), jnp.bfloat16) * 0.02
          wq_c, wk_c = None, None
        else:
          wqk_c = None
          wq_c = jax.random.normal(keys[4], (k_width, 4), jnp.bfloat16) * 0.02
          wk_c = jax.random.normal(keys[5], (k_width, 4), jnp.bfloat16) * 0.02
        wv_c = jax.random.normal(keys[6], (v_width, 4), jnp.bfloat16) * 0.02
        w_norm = jnp.ones((hv, dv), jnp.bfloat16)
        w_out = jax.random.normal(keys[7], (v_width, d_model), jnp.bfloat16) * 0.02
        w_ffn_gu = jax.random.normal(keys[8], (d_model, 2 * extra_ffn_dim), jnp.bfloat16) * 0.02 if extra_ffn_dim > 0 else None
        w_ffn_dn = jax.random.normal(keys[9], (extra_ffn_dim, d_model), jnp.bfloat16) * 0.02 if extra_ffn_dim > 0 else None

        total_params = proj_params + v_width * d_model + (2 * k_width * 4) + (0 if qk_only_conv else v_width * 4) + (3 * d_model * extra_ffn_dim)
        state_bytes_seq = hk * dk * (hv // hk) * dv * (2 if "bf16" in impl_type else 4)

        x_in = jax.random.normal(keys[0], (b, seq_len, d_model), jnp.bfloat16)

        def _fwd(x_, w_in_, w_c1_, w_c2_, wv_c_, w_norm_, w_out_, wf2_=w_f2, wg2_=w_g2, wf_gu_=w_ffn_gu, wf_dn_=w_ffn_dn):
          all_in = jnp.dot(x_, w_in_)
          if one_stage_proj:
            if fuse_qk_conv:
              qk_in, v_in, g_out_raw, fb_in = jnp.split(
                  all_in, [2 * k_width, 2 * k_width + v_width, 2 * k_width + 2 * v_width], axis=-1
              )
            else:
              q_in, k_in, v_in, g_out_raw, fb_in = jnp.split(
                  all_in, [k_width, 2 * k_width, 2 * k_width + v_width, 2 * k_width + 2 * v_width], axis=-1
              )
            f_raw = fb_in[..., :g_width]
            b_raw = fb_in[..., g_width : g_width + b_width]
            gate = jax.nn.sigmoid(g_out_raw).reshape(b, seq_len, hv, dv)
          else:
            if fuse_qk_conv:
              qk_in, v_in, f1_in, g1_in, b_raw = jnp.split(
                  all_in, [2 * k_width, 2 * k_width + v_width, 2 * k_width + v_width + dv, 2 * k_width + v_width + 2 * dv], axis=-1
              )
            else:
              q_in, k_in, v_in, f1_in, g1_in, b_raw = jnp.split(
                  all_in, [k_width, 2 * k_width, 2 * k_width + v_width, 2 * k_width + v_width + dv, 2 * k_width + v_width + 2 * dv], axis=-1
              )
            f_raw = jnp.dot(f1_in, wf2_)
            gate = jax.nn.sigmoid(jnp.dot(g1_in, wg2_)).reshape(b, seq_len, hv, dv)

          if fuse_qk_conv:
            qk_c = olmoe3.causal_depthwise_conv(qk_in, w_c1_, None).reshape(b, seq_len, 2 * hk, dk)
            qk_n = olmoe3._l2_normalize(qk_c.astype(jnp.float32))
            q_n = qk_n[:, :, :hk, :] * (dk**-0.5)
            k_n = qk_n[:, :, hk:, :]
          else:
            q_c = olmoe3.causal_depthwise_conv(q_in, w_c1_, None).reshape(b, seq_len, hk, dk)
            k_c = olmoe3.causal_depthwise_conv(k_in, w_c2_, None).reshape(b, seq_len, hk, dk)
            q_n = olmoe3._l2_normalize(q_c.astype(jnp.float32)) * (dk**-0.5)
            k_n = olmoe3._l2_normalize(k_c.astype(jnp.float32))

          if qk_only_conv:
            v_c = jax.nn.silu(v_in).reshape(b, seq_len, hv, dv)
          else:
            v_c = olmoe3.causal_depthwise_conv(v_in, wv_c_, None).reshape(b, seq_len, hv, dv)

          beta_n = 2.0 * jax.nn.sigmoid(b_raw.astype(jnp.float32))

          if impl_type == "ssd_pallas":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk) - 2.0)
            core_out = run_ssd_pallas(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "float32")
          elif impl_type == "kda_pallas":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, dk) - 2.0)
            core_out = kda.chunk_kda(
                q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets=resets, chunk_size=c,
                state_dtype=jnp.float32, layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=0,
            )
          elif impl_type == "kda_pallas_s1":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, dk) - 2.0)
            core_out = kda.chunk_kda(
                q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets=resets, chunk_size=c,
                state_dtype=jnp.float32, layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=1,
            )
          elif impl_type == "gdn_pallas":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk) - 2.0)
            core_out = run_gdn_pallas_fused(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "float32")
          elif impl_type == "sdn_pallas_g4":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, 4) - 2.0)
            core_out = run_sdn_g4_pallas(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "float32")
          elif impl_type == "sdn_pallas_g4_bf16":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, 4) - 2.0)
            core_out = run_sdn_g4_pallas(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "bfloat16")
          elif impl_type == "gva_gdn_pallas":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk) - 2.0)
            core_out = run_gva_gdn_pallas(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "float32")
          elif impl_type == "gva_sdn_pallas_g4":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, 4) - 2.0)
            core_out = run_gva_sdn_g4_pallas(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "float32")
          elif impl_type == "gva_sdn_pallas_g4_bf16":
            g_n = -jax.nn.softplus(f_raw.astype(jnp.float32).reshape(b, seq_len, hk, 4) - 2.0)
            core_out = run_gva_sdn_g4_pallas(q_n, k_n, v_c.astype(jnp.float32), g_n, beta_n, resets, c, "bfloat16")
          else:
            raise ValueError(impl_type)

          if bf16_epilogue:
            co_bf = core_out.astype(jnp.bfloat16)
            rms = jax.lax.rsqrt(jnp.mean(co_bf.astype(jnp.float32) * co_bf.astype(jnp.float32), axis=-1, keepdims=True) + 1e-5).astype(jnp.bfloat16)
            normed = co_bf * rms * (w_norm_[None, None, :, :] * gate)
          else:
            co_f32 = core_out.astype(jnp.float32)
            rms = jax.lax.rsqrt(jnp.mean(co_f32 * co_f32, axis=-1, keepdims=True) + 1e-5)
            normed = (co_f32 * rms).astype(jnp.bfloat16) * w_norm_[None, None, :, :] * gate
          out = jnp.dot(normed.reshape(b, seq_len, v_width), w_out_)
          if extra_ffn_dim > 0:
            gu = jnp.dot(x_, wf_gu_)
            g_ff, u_ff = jnp.split(gu, 2, axis=-1)
            out = out + jnp.dot(jax.nn.silu(g_ff) * u_ff, wf_dn_)
          return out

        c1 = wqk_c if fuse_qk_conv else wq_c
        c2 = wqk_c if fuse_qk_conv else wk_c
        p_list = [x_in, w_in, c1, c2, wv_c, w_norm, w_out]
        if not one_stage_proj:
          p_list.extend([w_f2, w_g2])
        if extra_ffn_dim > 0:
          p_list.extend([w_ffn_gu, w_ffn_dn])
        return _fwd, tuple(p_list), total_params, state_bytes_seq

      layer_arms_t6 = [
          # D=1792 (OLMo 3.5 Small) L11 Ultra-Wide Key SDN Variants
          ("FL_D1792_Tier2d_Raw_UltraWideKey_SDN_4K4V_512x256", 1792, 4, 4, 512, 256, 128, "gva_sdn_pallas_g4", True, True, True, True, 0, False),
          ("FL_D1792_Tier2e_IsoParam_GVA_WideKey_4K8V_512x256", 1792, 4, 8, 512, 256, 128, "gva_sdn_pallas_g4", True, True, True, True, 0, False),
          # D=2048 L11 Ultra-Wide Key SDN Family (Raw, Opt1/Opt2, 2:1 GVA Iso-Param, Wide-V384 Iso-Param, +FFN1024 Iso-Param/Iso-State)
          ("FL11a_Raw_UltraWideKey_SDN_G4_4K4V_512x256", 2048, 4, 4, 512, 256, 128, "gva_sdn_pallas_g4", True, True, False, False, 0, False),
          ("FL11a_Opt1_UltraWideKey_SDN_FusedQK_4K4V_512x256", 2048, 4, 4, 512, 256, 128, "gva_sdn_pallas_g4", True, True, True, True, 0, True),
          ("FL11a_Opt2_UltraWideKey_SDN_BF16St_4K4V_512x256", 2048, 4, 4, 512, 256, 128, "gva_sdn_pallas_g4_bf16", True, True, True, True, 0, False),
          ("FL11b_Opt1_IsoParam_GVA2to1_WideKey_4K8V_512x256", 2048, 4, 8, 512, 256, 128, "gva_sdn_pallas_g4", True, True, True, True, 0, True),
          ("FL11b_Opt2_IsoParam_GVA2to1_WideKey_BF16St_4K8V", 2048, 4, 8, 512, 256, 128, "gva_sdn_pallas_g4_bf16", True, True, True, True, 0, False),
          ("FL11c_Opt1_IsoParam_WideKey_WideV384_4K4V_512x384", 2048, 4, 4, 512, 384, 128, "gva_sdn_pallas_g4", True, True, True, True, 0, True),
          ("FL11d_Opt1_IsoParam_IsoState_4K4V_512x256_FFN1024", 2048, 4, 4, 512, 256, 128, "gva_sdn_pallas_g4", True, True, True, True, 1024, False),
      ]

      for fl_label, d_mod, hk, hv, dk, dv, c, impl_type, one_st, qk_only, f_qk, bf_epi, ex_ffn, do_xp in layer_arms_t6:
        try:
          dy_l = jax.random.normal(keys[7], (b, seq_len, d_mod), jnp.float32) * 0.1
          l_fn, l_params, n_params, st_bytes = _make_layer_fn_v2(
              d_mod, hk, hv, dk, dv, c, impl_type, one_st, qk_only, f_qk, bf_epi, ex_ffn
          )
          j_f = jax.jit(l_fn)
          j_fb = jax.jit(
              jax.value_and_grad(
                  lambda *args, _f=l_fn, _dy=dy_l: jnp.sum(_f(*args).astype(jnp.float32) * _dy),
                  argnums=tuple(range(len(l_params))),
              )
          )
          t_f, _ = _time_fn(j_f, *l_params)
          t_fb, _, hlo = _time_and_profile_hlo(j_fb, *l_params)
          print(
              f"[{fl_label:46s}] fwd={t_f:6.3f} ms | bwd={t_fb-t_f:6.3f} ms | fwd+bwd={t_fb:6.3f} ms "
              f"| params={n_params/1e6:5.2f}M | state/seq={st_bytes/1024:6.0f}KB | temp_MB={hlo['temp_mb']:6.1f} | HLO={hlo}",
              flush=True,
          )
          if do_xp:
            xp = _profile_xplane(j_fb, *l_params, top_k=7)
            print(f"    [XPlane {fl_label}] {xp}", flush=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{fl_label:46s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # PART 3: Real TPU v7x Inference Prefill (T=8K, 32K) & Batched Decode / Speculative
      #         Verification (T_verify in {1, 5, 7, 9}, B in {1, 8, 32, 64}) Silicon Benchmark
      # -----------------------------------------------------------------------
      print("\n=== PART 3A: Real TPU v7x Inference Prefill Silicon Benchmark (B=1, T=8192 & T=32768, D=2048) ===", flush=True)
      prefill_arms = [
          ("Prefill_FL11a_Raw_UltraWideKey_4K4V_512x256", 4, 4, 512, 256, 128, "gva_sdn_pallas_g4", True, True, True),
          ("Prefill_FL11b_IsoParam_GVA_WideKey_4K8V_512x256", 4, 8, 512, 256, 128, "gva_sdn_pallas_g4", True, True, True),
          ("Prefill_FL11c_IsoParam_WideKey_WideV384_4K4V", 4, 4, 512, 384, 128, "gva_sdn_pallas_g4", True, True, True),
      ]
      for pf_label, hk, hv, dk, dv, c, impl_type, one_st, qk_only, f_qk in prefill_arms:
        for t_pf in (8192, 32768):
          try:
            seq_len_saved = seq_len
            resets_saved = resets
            seq_len = t_pf
            resets = jnp.zeros((b, t_pf), dtype=bool)
            l_fn, l_params, _, _ = _make_layer_fn_v2(2048, hk, hv, dk, dv, c, impl_type, one_st, qk_only, f_qk, True, 0)
            j_pf = jax.jit(l_fn)
            t_pf_ms, _, hlo_pf = _time_and_profile_hlo(j_pf, *l_params)
            seq_len = seq_len_saved
            resets = resets_saved
            print(
                f"[{pf_label:42s} T={t_pf:5d}] fwd={t_pf_ms:6.3f} ms (x36={36*t_pf_ms:6.1f} ms) "
                f"| HBM={hlo_pf['gb_accessed']:5.2f} GB | temp={hlo_pf['temp_mb']:5.1f} MB",
                flush=True,
            )
          except Exception as exc:  # pylint: disable=broad-exception-caught
            seq_len = 8192
            resets = jnp.zeros((b, 8192), dtype=bool)
            print(f"[{pf_label:42s} T={t_pf:5d}] failed: {type(exc).__name__}: {exc}", flush=True)

      print(
          "\n=== PART 3B: Real TPU v7x Decode & Speculative Verification (T_ver in {1, 5, 7, 9}, B in {1, 8, 32, 64}) ===",
          flush=True,
      )

      def _make_decode_verify_step(
          b_dec: int,
          t_ver: int,
          d_model: int,
          hk: int,
          hv: int,
          dk: int,
          dv: int,
          g_groups: int,
          is_ssd: bool = False,
          qk_only_conv: bool = True,
          state_f32: bool = True,
      ):
        """Builds a single-layer decode / speculative verification step for t_ver tokens (t_ver=1 AR, t_ver=5/7/9 spec verify)."""
        k_width = hk * dk
        v_width = hv * dv
        dv_eff = (hv // hk) * dv
        g_width = hk * g_groups
        b_width = hk
        fb_pad = _ceil_div(g_width + b_width, 256) * 256
        s_dtype = jnp.float32 if state_f32 else jnp.bfloat16

        x_dec = jax.random.normal(keys[0], (b_dec, t_ver, d_model), jnp.bfloat16)
        state_in = jax.random.normal(keys[1], (b_dec, hk, dk, dv_eff), s_dtype) * 0.1
        conv_state_qk = jax.random.normal(keys[2], (b_dec, 3, 2 * k_width), jnp.bfloat16) * 0.1
        conv_state_v = jax.random.normal(keys[3], (b_dec, 3, v_width), jnp.bfloat16) * 0.1
        w_in = jax.random.normal(keys[4], (d_model, 2 * k_width + 2 * v_width + fb_pad), jnp.bfloat16) * 0.02
        w_conv_qk = jax.random.normal(keys[5], (4, 2 * k_width), jnp.bfloat16) * 0.02
        w_conv_v = jax.random.normal(keys[6], (4, v_width), jnp.bfloat16) * 0.02
        w_norm = jnp.ones((hv, dv), jnp.bfloat16)
        w_out = jax.random.normal(keys[7], (v_width, d_model), jnp.bfloat16) * 0.02

        def _decode_step(x_, st_, c_qk_, c_v_, w_in_, wc_qk_, wc_v_, wn_, wo_):
          all_in = jnp.dot(x_, w_in_)
          qk_in, v_in, g_out_raw, fb_in = jnp.split(
              all_in, [2 * k_width, 2 * k_width + v_width, 2 * k_width + 2 * v_width], axis=-1
          )
          f_raw = fb_in[..., :g_width].reshape(b_dec, t_ver, hk, g_groups)
          b_raw = fb_in[..., g_width : g_width + b_width].reshape(b_dec, t_ver, hk)
          gate = jax.nn.sigmoid(g_out_raw).reshape(b_dec, t_ver, hv, dv)

          # Causal Conv1D state update over t_ver tokens (K=4)
          qk_cat = jnp.concatenate([c_qk_, qk_in], axis=1)
          c_qk_new = qk_cat[:, -3:, :]
          qk_conv = (
              qk_cat[:, 0:t_ver, :] * wc_qk_[0]
              + qk_cat[:, 1 : t_ver + 1, :] * wc_qk_[1]
              + qk_cat[:, 2 : t_ver + 2, :] * wc_qk_[2]
              + qk_cat[:, 3 : t_ver + 3, :] * wc_qk_[3]
          )
          qk_act = jax.nn.silu(qk_conv).reshape(b_dec, t_ver, 2 * hk, dk)
          qk_norm = olmoe3._l2_normalize(qk_act.astype(jnp.float32)).astype(jnp.bfloat16)
          q_t = qk_norm[:, :, :hk, :] * jnp.bfloat16(dk**-0.5)
          k_t = qk_norm[:, :, hk:, :]

          if qk_only_conv:
            c_v_new = c_v_
            v_t = jax.nn.silu(v_in).reshape(b_dec, t_ver, hk, dv_eff)
          else:
            v_cat = jnp.concatenate([c_v_, v_in], axis=1)
            c_v_new = v_cat[:, -3:, :]
            v_conv = (
                v_cat[:, 0:t_ver, :] * wc_v_[0]
                + v_cat[:, 1 : t_ver + 1, :] * wc_v_[1]
                + v_cat[:, 2 : t_ver + 2, :] * wc_v_[2]
                + v_cat[:, 3 : t_ver + 3, :] * wc_v_[3]
            )
            v_t = jax.nn.silu(v_conv).reshape(b_dec, t_ver, hk, dv_eff)

          log_g = -jax.nn.softplus(f_raw.astype(jnp.float32) - 2.0)
          if g_groups == 1:
            decay_t = jnp.exp(log_g)  # [B, T, hk, 1]
          elif g_groups < dk:
            decay_t = jnp.repeat(jnp.exp(log_g), dk // g_groups, axis=-1)  # [B, T, hk, dk]
          else:
            decay_t = jnp.exp(log_g)  # [B, T, hk, dk]
          beta_t = (2.0 * jax.nn.sigmoid(b_raw.astype(jnp.float32))).astype(jnp.bfloat16)

          # Recurrent state update over t_ver tokens (reads state_in from HBM ONCE!)
          st_cur = st_.astype(jnp.float32)
          outs = []
          for tok_i in range(t_ver):
            q_i = q_t[:, tok_i, :, :]
            k_i = k_t[:, tok_i, :, :]
            v_i = v_t[:, tok_i, :, :]
            d_i = decay_t[:, tok_i, :, :][..., None]
            b_i = beta_t[:, tok_i, :][..., None]
            st_dec = st_cur * d_i
            if is_ssd:
              delta_i = (v_i * b_i).astype(jnp.bfloat16)
            else:
              ws_i = jnp.einsum("bhk,bhkv->bhv", k_i, st_dec.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
              delta_i = ((v_i.astype(jnp.float32) - ws_i) * b_i.astype(jnp.float32)).astype(jnp.bfloat16)
            st_cur = st_dec + jnp.einsum("bhk,bhv->bhkv", k_i, delta_i, preferred_element_type=jnp.float32)
            o_i = jnp.einsum("bhk,bhkv->bhv", q_i, st_cur.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
            outs.append(o_i.astype(jnp.bfloat16))

          core_out = jnp.stack(outs, axis=1).reshape(b_dec, t_ver, hv, dv)
          rms = jax.lax.rsqrt(jnp.mean(core_out.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-5).astype(jnp.bfloat16)
          normed = core_out * rms * (wn_[None, None, :, :] * gate)
          y_out = jnp.dot(normed.reshape(b_dec, t_ver, v_width), wo_)
          return y_out, st_cur.astype(s_dtype), c_qk_new, c_v_new

        args = (x_dec, state_in, conv_state_qk, conv_state_v, w_in, w_conv_qk, w_conv_v, w_norm, w_out)
        return _decode_step, args

      decode_arms = [
          # (label, hk, hv, dk, dv, g_groups, is_ssd, qk_only, state_f32)
          ("D11a_Raw_UltraWideKey_4K4V_512x256_FP32St", 4, 4, 512, 256, 4, False, True, True),
          ("D11a_Raw_UltraWideKey_4K4V_512x256_BF16St", 4, 4, 512, 256, 4, False, True, False),
          ("D11b_IsoParam_GVA_WideKey_4K8V_512x256_FP32St", 4, 8, 512, 256, 4, False, True, True),
          ("D11b_IsoParam_GVA_WideKey_4K8V_512x256_BF16St", 4, 8, 512, 256, 4, False, True, False),
          ("D11c_IsoParam_WideKey_WideV384_4K4V_FP32St", 4, 4, 512, 384, 4, False, True, True),
      ]

      for d_label, hk, hv, dk, dv, g_grp, is_ssd, qk_only, st_f32 in decode_arms:
        for b_dec, t_ver in ((1, 1), (32, 1), (32, 5), (32, 9)):
          try:
            d_fn, d_args = _make_decode_verify_step(b_dec, t_ver, 2048, hk, hv, dk, dv, g_grp, is_ssd, qk_only, st_f32)
            j_dec = jax.jit(d_fn)
            t_ms, _, hlo_d = _time_and_profile_hlo(j_dec, *d_args, warmup=2, iters=10)
            t_us = t_ms * 1000.0
            hbm_mb = hlo_d["gb_accessed"] * 1024.0
            bw_gbps = (hlo_d["gb_accessed"] / (t_ms / 1000.0)) if t_ms > 0 else 0.0
            print(
                f"[{d_label:36s} B={b_dec:2d} T_ver={t_ver}] step={t_us:6.1f} us/layer (x36={36*t_ms:5.2f} ms) "
                f"| HBM={hbm_mb:6.2f} MB/layer | BW={bw_gbps:6.1f} GB/s",
                flush=True,
            )
          except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[{d_label:36s} B={b_dec:2d} T_ver={t_ver}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # PART 4: Empirical Multi-Step JAX Training Convergence & Recall Ablation on TPU v7x
      # -----------------------------------------------------------------------
      print(
          "\n=== PART 4: Multi-Step Training Convergence & Associative/Induction Loss Ablation on TPU v7x (250 Steps) ===",
          flush=True,
      )

      def _generate_hybrid_recall_induction_batch(seed_val: int, batch_sz: int, seq_l: int = 96, vocab_sz: int = 64):
        """Generates structured sequences combining:
        (1) Multi-timescale associative key-value overwrites (keys 1..14 -> values 17..48, with 50% key overwrites),
        (2) Local token-shift induction n-grams (prefix marker 49..56 + context token 57..62 -> target value 17..48).
        Returns inputs [B, T], targets [B, T], assoc_mask [B, T], ind_mask [B, T].
        """
        rng = np.random.RandomState(seed_val)
        num_keys = 12
        tokens = np.zeros((batch_sz, seq_l + 1), dtype=np.int32)
        assoc_mask = np.zeros((batch_sz, seq_l), dtype=np.float32)
        ind_mask = np.zeros((batch_sz, seq_l), dtype=np.float32)

        for bi in range(batch_sz):
          kv_table = {}
          ngram_hist = {}
          pos = 0
          while pos < seq_l - 3:
            mode = rng.rand()
            if mode < 0.42 or len(kv_table) < 4:
              # Emit key-value write or overwrite: 60% chance of overwriting an existing key if >=4 keys exist!
              if len(kv_table) >= 4 and rng.rand() < 0.60:
                active_k = list(kv_table.keys())
                k_tok = active_k[rng.randint(0, len(active_k))]
              else:
                k_tok = rng.randint(1, num_keys + 1)
              v_tok = rng.randint(17, 49)
              tokens[bi, pos] = k_tok
              tokens[bi, pos + 1] = v_tok
              kv_table[k_tok] = v_tok
              pos += 2
            elif mode < 0.72 and len(kv_table) > 0:
              active_keys = list(kv_table.keys())
              k_tok = active_keys[rng.randint(0, len(active_keys))]
              v_tok = kv_table[k_tok]
              tokens[bi, pos] = 0  # query marker
              tokens[bi, pos + 1] = k_tok
              tokens[bi, pos + 2] = v_tok
              assoc_mask[bi, pos + 1] = 1.0
              pos += 3
            else:
              if len(ngram_hist) >= 3 and rng.rand() < 0.65:
                active_ng = list(ngram_hist.keys())
                m_tok, a_tok = active_ng[rng.randint(0, len(active_ng))]
                b_tok = ngram_hist[(m_tok, a_tok)]
                tokens[bi, pos] = m_tok
                tokens[bi, pos + 1] = a_tok
                tokens[bi, pos + 2] = b_tok
                ind_mask[bi, pos + 1] = 1.0
              else:
                m_tok = rng.randint(49, 57)
                a_tok = rng.randint(57, vocab_sz)
                b_tok = rng.randint(17, 49)
                ngram_hist[(m_tok, a_tok)] = b_tok
                tokens[bi, pos] = m_tok
                tokens[bi, pos + 1] = a_tok
                tokens[bi, pos + 2] = b_tok
              pos += 3
        inp = jnp.asarray(tokens[:, :-1], dtype=jnp.int32)
        tgt = jnp.asarray(tokens[:, 1:], dtype=jnp.int32)
        return inp, tgt, jnp.asarray(assoc_mask), jnp.asarray(ind_mask)

      vocab_c = 64
      seq_c = 96
      b_train = 64
      b_val = 256
      n_chunks = 5
      steps_per_chunk = 50
      n_steps = n_chunks * steps_per_chunk
      d_c = 128

      val_inp, val_tgt, val_assoc_m, val_ind_m = _generate_hybrid_recall_induction_batch(9999, b_val, seq_c, vocab_c)
      # Generate 50 diverse training batches and cycle over 5 epochs (250 steps total) with fast host generation
      train_batches = [
          _generate_hybrid_recall_induction_batch(1000 + st_i, b_train, seq_c, vocab_c)
          for st_i in range(steps_per_chunk)
      ]
      tr_inp_chunk = jnp.stack([tb[0] for tb in train_batches], axis=0)
      tr_tgt_chunk = jnp.stack([tb[1] for tb in train_batches], axis=0)
      tr_assoc_chunk = jnp.stack([tb[2] for tb in train_batches], axis=0)
      tr_ind_chunk = jnp.stack([tb[3] for tb in train_batches], axis=0)

      def _run_convergence_arm(
          arm_name: str,
          hk: int,
          hv: int,
          dk: int,
          dv: int,
          g_groups: int,
          use_delta_rule: bool = True,
          qk_only_conv: bool = False,
          ffn_mult: float = 2.0,
          no_conv_at_all: bool = False,
      ):
        k_w = hk * dk
        v_w = hv * dv
        dv_eff = (hv // hk) * dv
        g_w = hk * g_groups
        ffn_w = int(round(d_c * ffn_mult))
        k_init = jax.random.split(jax.random.PRNGKey(42), 20)

        def _init_layer(kl):
          ks = jax.random.split(kl, 8)
          # Causal Conv1D init: tap -1 (index 2) and tap 0 (index 3) active so K_t immediately sees token_{t-1} & token_t
          c_prior_q = jnp.array([[0.0], [0.15], [0.45], [0.85]], dtype=jnp.float32)
          c_prior_k = jnp.array([[0.0], [0.20], [0.85], [0.35]], dtype=jnp.float32)
          c_prior_v = jnp.array([[0.0], [0.0], [0.10], [0.95]], dtype=jnp.float32)
          return {
              "w_in": jax.random.normal(ks[0], (d_c, 2 * k_w + 2 * v_w + g_w + hk), jnp.float32) * (d_c**-0.5),
              "w_cq": c_prior_q + jax.random.normal(ks[1], (4, k_w), jnp.float32) * 0.08,
              "w_ck": c_prior_k + jax.random.normal(ks[2], (4, k_w), jnp.float32) * 0.08,
              "w_cv": c_prior_v + jax.random.normal(ks[3], (4, v_w), jnp.float32) * 0.08,
              "w_out": jax.random.normal(ks[4], (v_w, d_c), jnp.float32) * (0.25 * (v_w**-0.5)),
              "w_f1": jax.random.normal(ks[5], (d_c, 2 * ffn_w), jnp.float32) * (d_c**-0.5),
              "w_f2": jax.random.normal(ks[6], (ffn_w, d_c), jnp.float32) * (0.25 * (ffn_w**-0.5)),
          }

        params = {
            "emb": jax.random.normal(k_init[0], (vocab_c, d_c), jnp.float32) * 1.0,
            "l0": _init_layer(k_init[1]),
            "l1": _init_layer(k_init[2]),
            "head": jax.random.normal(k_init[3], (d_c, vocab_c), jnp.float32) * (d_c**-0.5),
        }
        n_params_total = sum(x.size for x in jax.tree_util.tree_leaves(params))

        def _causal_conv4(x_seq, w_c):
          x_pad = jnp.pad(x_seq, ((0, 0), (3, 0), (0, 0)))
          return (
              x_pad[:, 0:seq_c, :] * w_c[0]
              + x_pad[:, 1 : seq_c + 1, :] * w_c[1]
              + x_pad[:, 2 : seq_c + 2, :] * w_c[2]
              + x_pad[:, 3 : seq_c + 3, :] * w_c[3]
          )

        def _apply_layer(h_in, lp):
          rms_in = jax.lax.rsqrt(jnp.mean(h_in * h_in, axis=-1, keepdims=True) + 1e-5)
          x_n = h_in * rms_in
          proj = jnp.dot(x_n, lp["w_in"])
          q_r, k_r, v_r, g_out_r, f_r, b_r = jnp.split(
              proj, [k_w, 2 * k_w, 2 * k_w + v_w, 2 * k_w + 2 * v_w, 2 * k_w + 2 * v_w + g_w], axis=-1
          )
          if no_conv_at_all:
            q_c = jax.nn.silu(q_r).reshape(-1, seq_c, hk, dk)
            k_c = jax.nn.silu(k_r).reshape(-1, seq_c, hk, dk)
            v_c = jax.nn.silu(v_r).reshape(-1, seq_c, hk, dv_eff)
          else:
            q_c = jax.nn.silu(_causal_conv4(q_r, lp["w_cq"])).reshape(-1, seq_c, hk, dk)
            k_c = jax.nn.silu(_causal_conv4(k_r, lp["w_ck"])).reshape(-1, seq_c, hk, dk)
            if qk_only_conv:
              v_c = jax.nn.silu(v_r).reshape(-1, seq_c, hk, dv_eff)
            else:
              v_c = jax.nn.silu(_causal_conv4(v_r, lp["w_cv"])).reshape(-1, seq_c, hk, dv_eff)

          q_n = olmoe3._l2_normalize(q_c)
          k_n = olmoe3._l2_normalize(k_c)
          log_g = -jax.nn.softplus(f_r.reshape(-1, seq_c, hk, g_groups) - 3.5)
          if g_groups == 1:
            decay = jnp.exp(log_g)
          elif g_groups < dk:
            decay = jnp.repeat(jnp.exp(log_g), dk // g_groups, axis=-1)
          else:
            decay = jnp.exp(log_g)
          beta = jax.nn.sigmoid(b_r + 0.5)

          bsz = h_in.shape[0]
          init_s = jnp.zeros((bsz, hk, dk, dv_eff), jnp.float32)

          def _step(st, xs):
            qi, ki, vi, di, bi = xs
            st_d = st * di[..., None]
            if use_delta_rule:
              ws = jnp.einsum("bhk,bhkv->bhv", ki, st_d)
              delta = (vi - ws) * bi[..., None]
            else:
              delta = vi * bi[..., None]
            st_new = st_d + jnp.einsum("bhk,bhv->bhkv", ki, delta)
            oi = jnp.einsum("bhk,bhkv->bhv", qi, st_new)
            return st_new, oi

          tr_t = lambda z: jnp.moveaxis(z, 1, 0)
          _, out_t = jax.lax.scan(_step, init_s, (tr_t(q_n), tr_t(k_n), tr_t(v_c), tr_t(decay), tr_t(beta)))
          core_y = jnp.moveaxis(out_t, 0, 1).reshape(bsz, seq_c, v_w)
          rms_y = jax.lax.rsqrt(jnp.mean(core_y * core_y, axis=-1, keepdims=True) + 1e-5)
          gated_y = (core_y * rms_y) * jax.nn.sigmoid(g_out_r)
          h_mid = h_in + jnp.dot(gated_y, lp["w_out"])

          rms_f = jax.lax.rsqrt(jnp.mean(h_mid * h_mid, axis=-1, keepdims=True) + 1e-5)
          gu = jnp.dot(h_mid * rms_f, lp["w_f1"])
          gf, uf = jnp.split(gu, 2, axis=-1)
          return h_mid + jnp.dot(jax.nn.silu(gf) * uf, lp["w_f2"])

        def _forward_logits(p, x_tok):
          h = p["emb"][x_tok]
          h = _apply_layer(h, p["l0"])
          h = _apply_layer(h, p["l1"])
          rms_f = jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + 1e-5)
          return jnp.dot(h * rms_f, p["head"])

        def _loss_fn(p, x_tok, y_tok, a_m, i_m):
          logits = _forward_logits(p, x_tok)
          log_probs = jax.nn.log_softmax(logits, axis=-1)
          nll = -jnp.take_along_axis(log_probs, y_tok[..., None], axis=-1)[..., 0]
          # Weight associative recall & induction query positions 4x so optimizer focuses on recall + induction
          w_mask = 1.0 + 4.0 * (a_m + i_m)
          return jnp.sum(nll * w_mask) / jnp.sum(w_mask)

        @jax.jit
        def _train_chunk(p_in, m_in, v_in, step_base):
          def _body(carry, local_idx):
            p, m, v = carry
            xb = tr_inp_chunk[local_idx]
            yb = tr_tgt_chunk[local_idx]
            am = tr_assoc_chunk[local_idx]
            im = tr_ind_chunk[local_idx]
            loss_val, grads = jax.value_and_grad(_loss_fn)(p, xb, yb, am, im)
            g_step = step_base + local_idx + 1
            lr = 6e-3 * 0.5 * (1.0 + jnp.cos(jnp.pi * (step_base + local_idx) / n_steps))
            m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, grads)
            v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * (g_ * g_), v, grads)
            m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - 0.9**g_step), m)
            v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - 0.999**g_step), v)
            p = jax.tree_util.tree_map(
                lambda p_, mh_, vh_: p_ * (1.0 - lr * 1e-3) - lr * mh_ / (jnp.sqrt(vh_) + 1e-8), p, m_hat, v_hat
            )
            return (p, m, v), loss_val

          return jax.lax.scan(_body, (p_in, m_in, v_in), jnp.arange(steps_per_chunk))

        @jax.jit
        def _eval_fn(p_final):
          logits_v = _forward_logits(p_final, val_inp)
          log_p = jax.nn.log_softmax(logits_v, axis=-1)
          tok_nll = -jnp.take_along_axis(log_p, val_tgt[..., None], axis=-1)[..., 0]
          pred_tok = jnp.argmax(logits_v, axis=-1)
          tok_acc = (pred_tok == val_tgt).astype(jnp.float32)

          val_nll = jnp.mean(tok_nll)
          assoc_nll = jnp.sum(tok_nll * val_assoc_m) / jnp.maximum(jnp.sum(val_assoc_m), 1.0)
          assoc_acc = jnp.sum(tok_acc * val_assoc_m) / jnp.maximum(jnp.sum(val_assoc_m), 1.0)
          ind_nll = jnp.sum(tok_nll * val_ind_m) / jnp.maximum(jnp.sum(val_ind_m), 1.0)
          ind_acc = jnp.sum(tok_acc * val_ind_m) / jnp.maximum(jnp.sum(val_ind_m), 1.0)
          return val_nll, assoc_nll, assoc_acc, ind_nll, ind_acc

        m_cur = jax.tree_util.tree_map(jnp.zeros_like, params)
        v_cur = jax.tree_util.tree_map(jnp.zeros_like, params)
        p_cur = params
        all_losses = []
        for ch_i in range(n_chunks):
          (p_cur, m_cur, v_cur), ch_losses = _train_chunk(p_cur, m_cur, v_cur, jnp.int32(ch_i * steps_per_chunk))
          all_losses.append(ch_losses)
        losses = jnp.concatenate(all_losses, axis=0)
        v_nll, a_nll, a_acc, i_nll, i_acc = _eval_fn(p_cur)
        jax.block_until_ready(v_nll)
        print(
            f"[{arm_name:42s}] params={n_params_total/1e3:6.1f}K | train_nll: s0={float(losses[0]):.4f} -> s125={float(losses[124]):.4f} -> s250={float(losses[-1]):.4f} "
            f"| val_nll={float(v_nll):.4f} | assoc_nll={float(a_nll):.4f} (acc={100*float(a_acc):5.1f}%) "
            f"| ind_nll={float(i_nll):.4f} (acc={100*float(i_acc):5.1f}%)",
            flush=True,
        )

      conv_arms = [
          # (arm_name, hk, hv, dk, dv, g_groups, use_delta, qk_only, ffn_mult, no_conv)
          ("C8_SDN_Sym_4x32x32_G4_QKOnlyConv", 4, 4, 32, 32, 4, True, True, 2.0, False),
          ("C11a_Raw_UltraWideKey_SDN_G4_2K2V_64x32", 2, 2, 64, 32, 4, True, True, 2.0, False),
          ("C11b_IsoParam_GVA2to1_WideKey_2K4V_64x32", 2, 4, 64, 32, 4, True, True, 2.0, False),
          ("C11c_IsoParam_WideKey_WideV48_2K2V_64x48", 2, 2, 64, 48, 4, True, True, 2.0, False),
          ("C11d_IsoParam_IsoState_WideKey_2K2V_WideFFN", 2, 2, 64, 32, 4, True, True, 2.50, False),
      ]
      for c_args in conv_arms:
        try:
          _run_convergence_arm(*c_args)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{c_args[0]:42s}] failed: {type(exc).__name__}: {exc}", flush=True)

    finally:
      kda_fused._fwd_kernel = orig_fwd
      kda_fused._bwd_kernel = orig_bwd

  def test_07_sparse_attention_pcsa_vs_dsa_v7x(self):
    """Priority 5 (Area 4): Empirical TPU v7x Benchmark & Convergence for Page-Aligned P-CSA (m=128) vs. Token-Level DSA (m=1, 4) & Sub-Page NSA (m=64)."""
    print(
        "\n=== PRIORITY 5 (AREA 4): Sparse Attention Co-Design (Page-Aligned P-CSA m=128 vs. Token-Level DSA m=1,4) on TPU v7x ===",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # PART 1: Training / Prefill Forward + Backward Latency & HBM Bandwidth (S=8192, H_q=8, H_kv=2, hd=256, k_sel=2048)
    # -------------------------------------------------------------------------
    print(
        "\n--- PART 1: Training Forward+Backward Step Time & HBM Bandwidth on TPU v7x (B=1, S=8192, H_q=8, H_kv=2, hd=256, k_sel=2048) ---",
        flush=True,
    )
    b_sz = 1
    seq_len = 8192
    q_chunk = 128
    n_q_chunks = seq_len // q_chunk  # 64 query blocks of 128 tokens
    hq = 8
    hkv = 2
    gqa_rep = hq // hkv  # 4 query heads per KV head
    hd = 256
    k_sel = 2048
    d_idx = 64  # Indexer head dim

    rng_k = jax.random.split(jax.random.PRNGKey(701), 8)
    q_in = jax.random.normal(rng_k[0], (b_sz, seq_len, hq, hd), dtype=jnp.bfloat16)
    k_in = jax.random.normal(rng_k[1], (b_sz, seq_len, hkv, hd), dtype=jnp.bfloat16)
    v_in = jax.random.normal(rng_k[2], (b_sz, seq_len, hkv, hd), dtype=jnp.bfloat16)
    q_idx_in = jax.random.normal(rng_k[3], (b_sz, seq_len, d_idx), dtype=jnp.bfloat16)
    k_idx_in = jax.random.normal(rng_k[4], (b_sz, seq_len, d_idx), dtype=jnp.bfloat16)
    do_in = jax.random.normal(rng_k[5], (b_sz, seq_len, hq, hd), dtype=jnp.bfloat16)

    def _make_sparse_attn_fwd_bwd(mode: str, block_m: int = 128, cross_layer_share: bool = False):
      def _fwd(q, k, v, q_idx, k_idx):
        if mode == "dense":
          # Full S x S causal attention per KV head group: [B, H_kv, G_rep, S, hd]
          q_g = q.reshape(b_sz, seq_len, hkv, gqa_rep, hd).transpose(0, 2, 3, 1, 4)
          k_g = k.transpose(0, 2, 1, 3)[:, :, None, :, :]
          v_g = v.transpose(0, 2, 1, 3)[:, :, None, :, :]
          scores = jnp.einsum("bhgqd,bhgkd->bhgqk", q_g, k_g, preferred_element_type=jnp.float32) * (hd**-0.5)
          q_pos = jax.lax.broadcasted_iota(jnp.int32, (seq_len, 1), 0)
          k_pos = jax.lax.broadcasted_iota(jnp.int32, (1, seq_len), 1)
          scores = jnp.where(q_pos >= k_pos, scores, -1e9)
          probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
          out = jnp.einsum("bhgqk,bhgkd->bhgqd", probs, v_g, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
          return out.transpose(0, 3, 1, 2, 4).reshape(b_sz, seq_len, hq, hd)

        # Coarse or fine block selection per query block of size q_chunk=128
        n_kv_blocks = seq_len // block_m
        num_sel_blocks = k_sel // block_m
        q_blk = q.reshape(b_sz, n_q_chunks, q_chunk, hkv, gqa_rep, hd)

        if block_m == 1:
          # Token-level DSA (m=1): full [B, n_q_chunks, q_chunk, S] token indexer + top-k=2048 per query block
          q_idx_c = q_idx.reshape(b_sz, n_q_chunks, q_chunk, d_idx).mean(axis=2)  # [B, n_q_chunks, d_idx]
          idx_scores = jnp.einsum("bqd,bkd->bqk", q_idx_c, k_idx, preferred_element_type=jnp.float32)
          _, top_idx = jax.lax.top_k(idx_scores, num_sel_blocks)  # [B, n_q_chunks, 2048] scattered token indices
          # Gather 2048 individual tokens [B, n_q_chunks, 2048, hkv, hd] -> 0.5 KiB per gather!
          k_gath = jnp.take_along_axis(k[:, None, :, :, :], top_idx[:, :, :, None, None], axis=2)
          v_gath = jnp.take_along_axis(v[:, None, :, :, :], top_idx[:, :, :, None, None], axis=2)
        else:
          # Block/Page-level selection (m in {4, 64, 128}): pool k_idx into block centroids first
          q_idx_c = q_idx.reshape(b_sz, n_q_chunks, q_chunk, d_idx).mean(axis=2)
          k_idx_c = k_idx.reshape(b_sz, n_kv_blocks, block_m, d_idx).mean(axis=2)
          idx_scores = jnp.einsum("bqd,bkd->bqk", q_idx_c, k_idx_c, preferred_element_type=jnp.float32)
          _, top_blk = jax.lax.top_k(idx_scores, num_sel_blocks)  # [B, n_q_chunks, num_sel_blocks]
          k_paged = k.reshape(b_sz, n_kv_blocks, block_m, hkv, hd)
          v_paged = v.reshape(b_sz, n_kv_blocks, block_m, hkv, hd)
          # Gather contiguous pages [B, n_q_chunks, num_sel_blocks, block_m, hkv, hd]
          k_gath = jnp.take_along_axis(
              k_paged[:, None, :, :, :, :], top_blk[:, :, :, None, None, None], axis=2
          ).reshape(b_sz, n_q_chunks, k_sel, hkv, hd)
          v_gath = jnp.take_along_axis(
              v_paged[:, None, :, :, :, :], top_blk[:, :, :, None, None, None], axis=2
          ).reshape(b_sz, n_q_chunks, k_sel, hkv, hd)

        # Compute block attention on [B, n_q_chunks, hkv, gqa_rep, q_chunk=128, hd=256] x [B, n_q_chunks, hkv, 1, k_sel=2048, hd=256]
        q_t = q_blk.transpose(0, 1, 3, 4, 2, 5)
        k_t = k_gath.transpose(0, 1, 3, 2, 4)[:, :, :, None, :, :]
        v_t = v_gath.transpose(0, 1, 3, 2, 4)[:, :, :, None, :, :]
        att_sc = jnp.einsum("bnhgqd,bnhgkd->bnhgqk", q_t, k_t, preferred_element_type=jnp.float32) * (hd**-0.5)
        att_p = jax.nn.softmax(att_sc, axis=-1).astype(jnp.bfloat16)
        out_blk = jnp.einsum("bnhgqk,bnhgkd->bnhgqd", att_p, v_t, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        out_1 = out_blk.transpose(0, 1, 4, 2, 3, 5).reshape(b_sz, seq_len, hq, hd)

        if cross_layer_share:
          # 2:1 Cross-Layer KV & Page-Index Sharing: Layer l+1 reuses already-gathered (k_t, v_t) without re-indexing or re-gathering!
          q2_t = (q_t * jnp.bfloat16(0.95)) + out_blk
          att_sc2 = jnp.einsum("bnhgqd,bnhgkd->bnhgqk", q2_t, k_t, preferred_element_type=jnp.float32) * (hd**-0.5)
          att_p2 = jax.nn.softmax(att_sc2, axis=-1).astype(jnp.bfloat16)
          out_blk2 = jnp.einsum("bnhgqk,bnhgkd->bnhgqd", att_p2, v_t, preferred_element_type=jnp.float32).astype(
              jnp.bfloat16
          )
          out_2 = out_blk2.transpose(0, 1, 4, 2, 3, 5).reshape(b_sz, seq_len, hq, hd)
          return 0.5 * (out_1 + out_2)
        return out_1

      def _fwd_bwd(q, k, v, q_idx, k_idx, do):
        out, vjp_fn = jax.vjp(_fwd, q, k, v, q_idx, k_idx)
        dq, dk, dv, _, _ = vjp_fn(do)
        return out, dq, dk, dv

      return _fwd_bwd

    sparse_train_arms = [
        ("S1_Dense_Causal_Attention_S8K_hd256", "dense", 128, False, 1.0),
        ("S2_Token_Level_DSA_m1_k2048_S8K", "sparse", 1, False, 1.0),
        ("S3_SubTile_DSA_m4_k2048_S8K", "sparse", 4, False, 1.0),
        ("S4_SubPage_NSA_m64_k2048_S8K", "sparse", 64, False, 1.0),
        ("S5_PageAligned_PCSA_m128_p16_S8K", "sparse", 128, False, 1.0),
        ("S6_PageAligned_PCSA_m128_CrossLayer2to1", "sparse", 128, True, 0.5),
    ]

    base_train_ms = None
    for s_label, s_mode, blk_m, x_share, layer_scale in sparse_train_arms:
      try:
        fn = jax.jit(_make_sparse_attn_fwd_bwd(s_mode, blk_m, x_share))
        raw_ms, _, hlo_s = _time_and_profile_hlo(fn, q_in, k_in, v_in, q_idx_in, k_idx_in, do_in, warmup=2, iters=5)
        eff_ms = raw_ms * layer_scale
        eff_hbm_mb = hlo_s["gb_accessed"] * 1024.0 * layer_scale
        bw_gbps = (hlo_s["gb_accessed"] / (raw_ms / 1000.0)) if raw_ms > 0 else 0.0
        if base_train_ms is None:
          base_train_ms = eff_ms
        sp = base_train_ms / eff_ms if eff_ms > 0 else 0.0
        tflops_s = hlo_s.get("tflops", 0.0) / (raw_ms / 1000.0) if raw_ms > 0 else 0.0
        print(
            f"[{s_label:40s}] fwd+bwd={eff_ms:6.3f} ms/layer ({sp:5.2f}x vs Dense) "
            f"| HBM={eff_hbm_mb:6.1f} MB/layer | BW={bw_gbps:6.1f} GB/s | TFLOP/s={tflops_s:6.1f}",
            flush=True,
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[{s_label:40s}] failed: {type(exc).__name__}: {exc}", flush=True)

    # -------------------------------------------------------------------------
    # PART 2: Long-Context Decode / Spec-Verify Latency & Effective HBM BW at S_ctx = 32,768 (B in {1, 8})
    # -------------------------------------------------------------------------
    print(
        "\n--- PART 2: Long-Context Decode Latency & Effective HBM Bandwidth at S_ctx=32,768 on TPU v7x (H_q=16, H_kv=2, hd=256, k_sel=2048) ---",
        flush=True,
    )
    s_ctx = 32768
    hq_d = 16
    hkv_d = 2
    gqa_d = hq_d // hkv_d

    def _make_decode_sparse_step(b_dec: int, mode: str, block_m: int = 128, cross_layer: bool = False):
      rk = jax.random.split(jax.random.PRNGKey(800 + b_dec), 6)
      q_d = jax.random.normal(rk[0], (b_dec, hkv_d, gqa_d, hd), dtype=jnp.bfloat16)
      k_cache = jax.random.normal(rk[1], (b_dec, s_ctx, hkv_d, hd), dtype=jnp.bfloat16)
      v_cache = jax.random.normal(rk[2], (b_dec, s_ctx, hkv_d, hd), dtype=jnp.bfloat16)
      q_idx_d = jax.random.normal(rk[3], (b_dec, d_idx), dtype=jnp.bfloat16)
      n_blks = s_ctx // block_m
      num_sel = k_sel // block_m
      k_idx_cache = jax.random.normal(rk[4], (b_dec, n_blks, d_idx), dtype=jnp.bfloat16)

      def _decode_fn(q_step, kc, vc, qi, ki_c):
        if mode == "dense":
          sc = jnp.einsum("bhgd,bshd->bhgs", q_step, kc, preferred_element_type=jnp.float32) * (hd**-0.5)
          pr = jax.nn.softmax(sc, axis=-1).astype(jnp.bfloat16)
          return jnp.einsum("bhgs,bshd->bhgd", pr, vc, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

        idx_sc = jnp.einsum("bd,bnd->bn", qi, ki_c, preferred_element_type=jnp.float32)
        _, top_b = jax.lax.top_k(idx_sc, num_sel)  # [B, num_sel]
        if block_m == 1:
          kg = jnp.take_along_axis(kc, top_b[:, :, None, None], axis=1)  # [B, 2048, hkv, hd]
          vg = jnp.take_along_axis(vc, top_b[:, :, None, None], axis=1)
        else:
          kc_p = kc.reshape(b_dec, n_blks, block_m, hkv_d, hd)
          vc_p = vc.reshape(b_dec, n_blks, block_m, hkv_d, hd)
          kg = jnp.take_along_axis(kc_p, top_b[:, :, None, None, None], axis=1).reshape(b_dec, k_sel, hkv_d, hd)
          vg = jnp.take_along_axis(vc_p, top_b[:, :, None, None, None], axis=1).reshape(b_dec, k_sel, hkv_d, hd)

        sc1 = jnp.einsum("bhgd,bshd->bhgs", q_step, kg, preferred_element_type=jnp.float32) * (hd**-0.5)
        pr1 = jax.nn.softmax(sc1, axis=-1).astype(jnp.bfloat16)
        o1 = jnp.einsum("bhgs,bshd->bhgd", pr1, vg, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        if cross_layer:
          q2 = q_step * jnp.bfloat16(0.95) + o1
          sc2 = jnp.einsum("bhgd,bshd->bhgs", q2, kg, preferred_element_type=jnp.float32) * (hd**-0.5)
          pr2 = jax.nn.softmax(sc2, axis=-1).astype(jnp.bfloat16)
          o2 = jnp.einsum("bhgs,bshd->bhgd", pr2, vg, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
          return 0.5 * (o1 + o2)
        return o1

      return _decode_fn, (q_d, k_cache, v_cache, q_idx_d, k_idx_cache)

    decode_sparse_arms = [
        ("D_S1_Dense_Decode_S32K_hd256", "dense", 128, False, 1.0),
        ("D_S2_TokenGather_DSA_m1_k2048_S32K", "sparse", 1, False, 1.0),
        ("D_S3_SubTile_DSA_m4_k2048_S32K", "sparse", 4, False, 1.0),
        ("D_S4_SubPage_NSA_m64_k2048_S32K", "sparse", 64, False, 1.0),
        ("D_S5_PageAligned_PCSA_m128_p16_S32K", "sparse", 128, False, 1.0),
        ("D_S6_PageAligned_PCSA_m128_CrossLayer2to1", "sparse", 128, True, 0.5),
    ]

    for b_dec in (1, 8):
      dense_us_ref = None
      dsa_m1_us_ref = None
      for d_label, d_mode, blk_m, x_share, l_scale in decode_sparse_arms:
        try:
          d_fn, d_args = _make_decode_sparse_step(b_dec, d_mode, blk_m, x_share)
          j_fn = jax.jit(d_fn)
          raw_ms, _, hlo_d = _time_and_profile_hlo(j_fn, *d_args, warmup=2, iters=10)
          eff_us = raw_ms * 1000.0 * l_scale
          eff_mb = hlo_d["gb_accessed"] * 1024.0 * l_scale
          bw_gbps = (hlo_d["gb_accessed"] / (raw_ms / 1000.0)) if raw_ms > 0 else 0.0
          if dense_us_ref is None:
            dense_us_ref = eff_us
          if d_label == "D_S2_TokenGather_DSA_m1_k2048_S32K":
            dsa_m1_us_ref = eff_us
          sp_dense = (dense_us_ref / eff_us) if eff_us > 0 else 0.0
          sp_dsa = (dsa_m1_us_ref / eff_us) if (dsa_m1_us_ref and eff_us > 0) else 1.0
          print(
              f"[{d_label:42s} B={b_dec:2d}] step={eff_us:7.1f} us/layer ({sp_dense:5.2f}x vs Dense, {sp_dsa:4.2f}x vs DSA-m1) "
              f"| HBM={eff_mb:6.2f} MB/layer | BW={bw_gbps:6.1f} GB/s",
              flush=True,
          )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{d_label:42s} B={b_dec:2d}] failed: {type(exc).__name__}: {exc}", flush=True)

    # -------------------------------------------------------------------------
    # PART 3: 250-Step JAX Convergence & Multi-Token Span Recall Ablation on TPU v7x
    # -------------------------------------------------------------------------
    print(
        "\n--- PART 3: 250-Step JAX Convergence & Multi-Token Span Recall (Dense vs. Token-DSA m=1 vs. Page-Aligned P-CSA m=8) ---",
        flush=True,
    )

    def _generate_page_span_recall_batch(seed_val: int, batch_sz: int, seq_l: int = 96, page_sz: int = 8, vocab_sz: int = 64):
      """Generates sequences of 12 pages x 8 tokens = 96 tokens where each page starts with a page topic key (1..12)
      followed by a 3-token compound entity span (k1, k2 -> target v in 17..48) and distractor tokens (49..63).
      At the end of the sequence (last 2 pages), queries present (page_topic_key, k1, k2 -> target v).
      Page-Aligned P-CSA (m=8) retrieves the entire 8-token page containing (topic, k1, k2, v) intact, whereas
      Token-Level DSA (m=1) with the same k_sel token budget fragments the multi-token span!
      """
      rng = np.random.RandomState(seed_val)
      n_pages = seq_l // page_sz  # 12 pages of 8 tokens
      tokens = np.zeros((batch_sz, seq_l + 1), dtype=np.int32)
      span_mask = np.zeros((batch_sz, seq_l), dtype=np.float32)

      for bi in range(batch_sz):
        facts = []
        # First 9 pages store 9 distinct multi-token facts
        for pi in range(n_pages - 3):
          base = pi * page_sz
          topic = (pi % 12) + 1
          k1 = rng.randint(13, 25)
          k2 = rng.randint(25, 37)
          v_tok = rng.randint(37, 53)
          facts.append((topic, k1, k2, v_tok))
          tokens[bi, base + 0] = topic
          tokens[bi, base + 1] = rng.randint(53, vocab_sz)
          tokens[bi, base + 2] = k1
          tokens[bi, base + 3] = k2
          tokens[bi, base + 4] = v_tok
          for rem in range(5, page_sz):
            tokens[bi, base + rem] = rng.randint(53, vocab_sz)
        # Last 3 pages issue 6 multi-token queries (each 4 tokens: topic, k1, k2, v_tok)
        q_base = (n_pages - 3) * page_sz
        for qi in range(6):
          pos = q_base + qi * 4
          if pos + 4 <= seq_l:
            topic, k1, k2, v_tok = facts[rng.randint(0, len(facts))]
            tokens[bi, pos + 0] = topic
            tokens[bi, pos + 1] = k1
            tokens[bi, pos + 2] = k2
            tokens[bi, pos + 3] = v_tok
            span_mask[bi, pos + 2] = 1.0  # Predict v_tok after seeing (topic, k1, k2)
      return (
          jnp.asarray(tokens[:, :-1], dtype=jnp.int32),
          jnp.asarray(tokens[:, 1:], dtype=jnp.int32),
          jnp.asarray(span_mask, dtype=jnp.float32),
      )

    seq_sp = 64
    page_sp = 8
    k_budget = 16  # Attend to exactly 16 tokens (2 pages of 8 tokens, or 16 individual tokens) out of 64 (8 pages = power of 2)
    vocab_sp = 64
    d_sp = 128
    b_tr_sp = 64
    b_val_sp = 256
    steps_per_ch = 50
    n_ch = 5
    tot_steps = steps_per_ch * n_ch

    val_x_sp, val_y_sp, val_m_sp = _generate_page_span_recall_batch(9991, b_val_sp, seq_sp, page_sp, vocab_sp)
    tr_sp_batches = [_generate_page_span_recall_batch(2000 + i, b_tr_sp, seq_sp, page_sp, vocab_sp) for i in range(steps_per_ch)]
    tr_x_sp = jnp.stack([b[0] for b in tr_sp_batches], axis=0)
    tr_y_sp = jnp.stack([b[1] for b in tr_sp_batches], axis=0)
    tr_m_sp = jnp.stack([b[2] for b in tr_sp_batches], axis=0)

    def _run_sparse_convergence_arm(arm_name: str, mode: str, cross_layer_share: bool = False, ffn_mult: float = 2.0):
      n_heads_sp = 4
      hd_sp = d_sp // n_heads_sp
      ffn_w = int(round(d_sp * ffn_mult))
      ks = jax.random.split(jax.random.PRNGKey(77), 16)

      def _init_sp_layer(k_l):
        kl = jax.random.split(k_l, 7)
        c_p = jnp.array([[0.0], [0.25], [0.65], [0.90]], dtype=jnp.float32)
        return {
            "w_q": jax.random.normal(kl[0], (d_sp, d_sp), jnp.float32) * (d_sp**-0.5),
            "w_kv": jax.random.normal(kl[1], (d_sp, 2 * d_sp), jnp.float32) * (d_sp**-0.5),
            "w_cq": c_p + jax.random.normal(kl[2], (4, d_sp), jnp.float32) * 0.05,
            "w_ck": c_p + jax.random.normal(kl[3], (4, d_sp), jnp.float32) * 0.05,
            "w_out": jax.random.normal(kl[4], (d_sp, d_sp), jnp.float32) * (0.25 * (d_sp**-0.5)),
            "w_f1": jax.random.normal(kl[5], (d_sp, 2 * ffn_w), jnp.float32) * (d_sp**-0.5),
            "w_f2": jax.random.normal(kl[6], (ffn_w, d_sp), jnp.float32) * (0.25 * (ffn_w**-0.5)),
        }

      params_sp = {
          "emb": jax.random.normal(ks[0], (vocab_sp, d_sp), jnp.float32) * 1.0,
          "l0": _init_sp_layer(ks[1]),
          "l1": _init_sp_layer(ks[2]),
          "head": jax.random.normal(ks[3], (d_sp, vocab_sp), jnp.float32) * (d_sp**-0.5),
      }
      n_params = sum(x.size for x in jax.tree_util.tree_leaves(params_sp))
      if cross_layer_share:
        # 2:1 Cross-Layer KV Sharing: l1 reuses l0's KV projection weights (saving l1["w_kv"] + l1["w_ck"])
        n_params -= params_sp["l1"]["w_kv"].size + params_sp["l1"]["w_ck"].size

      def _conv4_sp(x_s, wc):
        xp = jnp.pad(x_s, ((0, 0), (3, 0), (0, 0)))
        return xp[:, 0:seq_sp, :] * wc[0] + xp[:, 1 : seq_sp + 1, :] * wc[1] + xp[:, 2 : seq_sp + 2, :] * wc[2] + xp[:, 3 : seq_sp + 3, :] * wc[3]

      def _attn_block(h_in, lp, shared_kv=None):
        bsz = h_in.shape[0]
        rms = jax.lax.rsqrt(jnp.mean(h_in * h_in, axis=-1, keepdims=True) + 1e-5)
        xn = h_in * rms
        q_raw = _conv4_sp(jnp.dot(xn, lp["w_q"]), lp["w_cq"])
        if shared_kv is None:
          kv_raw = jnp.dot(xn, lp["w_kv"])
          k_r, v_r = jnp.split(kv_raw, 2, axis=-1)
          k_raw = _conv4_sp(k_r, lp["w_ck"])
          v_raw = v_r
        else:
          k_raw, v_raw = shared_kv

        q_h = q_raw.reshape(bsz, seq_sp, n_heads_sp, hd_sp).transpose(0, 2, 1, 3)
        k_h = k_raw.reshape(bsz, seq_sp, n_heads_sp, hd_sp).transpose(0, 2, 1, 3)
        v_h = v_raw.reshape(bsz, seq_sp, n_heads_sp, hd_sp).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q_h, k_h) * (hd_sp**-0.5)
        q_idx = jax.lax.broadcasted_iota(jnp.int32, (seq_sp, 1), 0)
        k_idx = jax.lax.broadcasted_iota(jnp.int32, (1, seq_sp), 1)
        causal_m = q_idx >= k_idx

        if mode == "dense":
          masked_scores = jnp.where(causal_m, scores, -1e9)
        elif mode == "dsa_m1":
          # Token-level top-k selection (k_budget=24 individual tokens): scores each token independently
          idx_s = jnp.where(causal_m, jnp.mean(scores, axis=1), -1e9)  # [B, Q, K]
          thresh = jax.lax.top_k(idx_s, k_budget)[0][..., -1:]
          sel_m = causal_m & (idx_s >= thresh)
          masked_scores = jnp.where(sel_m[:, None, :, :], scores, -1e9)
        else:
          # Page-Aligned P-CSA (m=8, 3 pages = 24 tokens): page-centroid router scores whole 8-token pages!
          n_pg = seq_sp // page_sp
          q_pg = q_raw.reshape(bsz, n_pg, page_sp, d_sp).mean(axis=2)  # [B, n_pg, d_sp]
          k_pg = k_raw.reshape(bsz, n_pg, page_sp, d_sp).mean(axis=2)  # [B, n_pg, d_sp]
          pg_scores = jnp.einsum("bqd,bkd->bqk", q_pg, k_pg) * (d_sp**-0.5)
          pg_q_i = jax.lax.broadcasted_iota(jnp.int32, (n_pg, 1), 0)
          pg_k_i = jax.lax.broadcasted_iota(jnp.int32, (1, n_pg), 1)
          pg_causal = pg_q_i >= pg_k_i
          pg_scores = jnp.where(pg_causal, pg_scores, -1e9)
          n_sel_pg = k_budget // page_sp  # 3 pages = 24 tokens
          pg_thresh = jax.lax.top_k(pg_scores, n_sel_pg)[0][..., -1:]
          pg_mask = pg_causal & (pg_scores >= pg_thresh)  # [B, n_pg, n_pg]
          tok_pg_mask = jnp.repeat(jnp.repeat(pg_mask, page_sp, axis=1), page_sp, axis=2)  # [B, S, S]
          # Soft differentiable page-router gate during training so page centroids receive direct gradient
          pg_gate = jax.nn.sigmoid((pg_scores - pg_thresh + 0.5) * 2.0)
          tok_pg_gate = jnp.repeat(jnp.repeat(pg_gate, page_sp, axis=1), page_sp, axis=2)[:, None, :, :]
          masked_scores = jnp.where(causal_m & tok_pg_mask[:, None, :, :], scores + jnp.log(tok_pg_gate + 1e-4), -1e9)

        probs = jax.nn.softmax(masked_scores, axis=-1)
        out_h = jnp.einsum("bhqk,bhkd->bhqd", probs, v_h).transpose(0, 2, 1, 3).reshape(bsz, seq_sp, d_sp)
        h_mid = h_in + jnp.dot(out_h, lp["w_out"])
        rms_f = jax.lax.rsqrt(jnp.mean(h_mid * h_mid, axis=-1, keepdims=True) + 1e-5)
        gu = jnp.dot(h_mid * rms_f, lp["w_f1"])
        gf, uf = jnp.split(gu, 2, axis=-1)
        h_out = h_mid + jnp.dot(jax.nn.silu(gf) * uf, lp["w_f2"])
        return h_out, (k_raw, v_raw)

      def _fwd_logits(p, x_tok):
        h = p["emb"][x_tok]
        h, kv0 = _attn_block(h, p["l0"], shared_kv=None)
        h, _ = _attn_block(h, p["l1"], shared_kv=kv0 if cross_layer_share else None)
        rms = jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + 1e-5)
        return jnp.dot(h * rms, p["head"])

      def _loss_sp(p, xt, yt, sm):
        lg = _fwd_logits(p, xt)
        lp = jax.nn.log_softmax(lg, axis=-1)
        nll = -jnp.take_along_axis(lp, yt[..., None], axis=-1)[..., 0]
        w = 1.0 + 6.0 * sm
        return jnp.sum(nll * w) / jnp.sum(w)

      @jax.jit
      def _train_step_sp(p, m, v, xt, yt, sm, gs):
        l_val, grads = jax.value_and_grad(_loss_sp)(p, xt, yt, sm)
        lr = 6e-3 * 0.5 * (1.0 + jnp.cos(jnp.pi * (gs - 1) / tot_steps))
        m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, grads)
        v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * (g_ * g_), v, grads)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - 0.9**gs), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - 0.999**gs), v)
        p = jax.tree_util.tree_map(lambda p_, mh_, vh_: p_ * (1.0 - lr * 1e-3) - lr * mh_ / (jnp.sqrt(vh_) + 1e-8), p, mh, vh)
        return p, m, v, l_val

      @jax.jit
      def _eval_sp(pf):
        lg = _fwd_logits(pf, val_x_sp)
        lp = jax.nn.log_softmax(lg, axis=-1)
        tnll = -jnp.take_along_axis(lp, val_y_sp[..., None], axis=-1)[..., 0]
        tacc = (jnp.argmax(lg, axis=-1) == val_y_sp).astype(jnp.float32)
        vnll = jnp.mean(tnll)
        snll = jnp.sum(tnll * val_m_sp) / jnp.maximum(jnp.sum(val_m_sp), 1.0)
        sacc = jnp.sum(tacc * val_m_sp) / jnp.maximum(jnp.sum(val_m_sp), 1.0)
        return vnll, snll, sacc

      m_c = jax.tree_util.tree_map(jnp.zeros_like, params_sp)
      v_c = jax.tree_util.tree_map(jnp.zeros_like, params_sp)
      p_c = params_sp
      losses_all = []
      for step_i in range(tot_steps):
        idx = step_i % steps_per_ch
        p_c, m_c, v_c, cl = _train_step_sp(
            p_c, m_c, v_c, tr_x_sp[idx], tr_y_sp[idx], tr_m_sp[idx], jnp.int32(step_i + 1)
        )
        losses_all.append(cl)
      losses_v = jnp.stack(losses_all, axis=0)
      vnll, snll, sacc = _eval_sp(p_c)
      jax.block_until_ready(vnll)
      print(
          f"[{arm_name:44s}] params={n_params/1e3:6.1f}K | train_nll: s0={float(losses_v[0]):.4f} -> s250={float(losses_v[-1]):.4f} "
          f"| val_nll={float(vnll):.4f} | span_recall_nll={float(snll):.4f} (acc={100*float(sacc):5.1f}%)",
          flush=True,
      )

    for sp_arm in [
        ("SP1_Dense_Full_Attention_Reference", "dense", False, 2.0),
        ("SP2_Token_Level_DSA_m1_k24", "dsa_m1", False, 2.0),
        ("SP3_PageAligned_PCSA_m8_p3_k24", "pcsa_m8", False, 2.0),
        ("SP4_PageAligned_PCSA_m8_CrossLayer2to1_IsoParam", "pcsa_m8", True, 2.5),
    ]:
      try:
        _run_sparse_convergence_arm(*sp_arm)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[{sp_arm[0]:44s}] failed: {type(exc).__name__}: {exc}", flush=True)

  def test_08_hyper_connections_tpu_hc_vs_mhc_v7x(self):
    """Priority 6 (Area 5): Empirical TPU v7x Benchmark & Deep Signal-Gain/Convergence for Fused TPU-HC + LatentMoE vs. DHC & mHC Sinkhorn."""
    print(
        "\n=== PRIORITY 6 (AREA 5): Multi-Stream Hyper-Connections (Fused TPU-HC + LatentMoE vs. DHC & mHC Sinkhorn) on TPU v7x ===",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # PART 1: Training Forward + Backward Latency & HBM Traffic on TPU v7x (B=1, T=8192, D=2048, Latent L=512, K=4)
    # -------------------------------------------------------------------------
    print(
        "\n--- PART 1: Forward+Backward Step Latency & HBM Traffic on TPU v7x (B=1, T=8192, D=2048, Latent L=512, K in {1, 2, 4}) ---",
        flush=True,
    )
    b_hc = 1
    t_hc = 8192
    d_hc = 2048
    l_lat = 512
    ffn_lat = 2048

    rk = jax.random.split(jax.random.PRNGKey(901), 12)
    w_rms = jax.random.normal(rk[0], (d_hc,), dtype=jnp.bfloat16)
    w_down_lat = jax.random.normal(rk[1], (d_hc, l_lat), dtype=jnp.bfloat16) * (d_hc**-0.5)
    w_f1_lat = jax.random.normal(rk[2], (l_lat, 2 * ffn_lat), dtype=jnp.bfloat16) * (l_lat**-0.5)
    w_f2_lat = jax.random.normal(rk[3], (ffn_lat, l_lat), dtype=jnp.bfloat16) * (ffn_lat**-0.5)
    w_up_lat = jax.random.normal(rk[4], (l_lat, d_hc), dtype=jnp.bfloat16) * (l_lat**-0.5)

    def _latent_moe_core(x_norm):
      z = jnp.dot(x_norm, w_down_lat)
      gu = jnp.dot(z, w_f1_lat)
      g, u = jnp.split(gu, 2, axis=-1)
      z_out = jnp.dot(jax.nn.silu(g) * u, w_f2_lat)
      return jnp.dot(z_out, w_up_lat)

    def _make_hc_fwd_bwd(mode: str, k_streams: int = 4):
      x_s = jax.random.normal(rk[5], (b_hc, t_hc, k_streams, d_hc) if k_streams > 1 else (b_hc, t_hc, d_hc), dtype=jnp.bfloat16)
      do_s = jax.random.normal(rk[6], x_s.shape, dtype=jnp.bfloat16)
      # Dynamic routing weights for DHC / mHC: [D, K*K + 2*K]
      w_dyn = jax.random.normal(rk[7], (d_hc, k_streams * k_streams + 2 * k_streams), dtype=jnp.bfloat16) * (d_hc**-0.5)
      # Static P2S sphere weights for Fused TPU-HC: [K, K] stream mix + [K, D] fused RMSNorm read + [K, D] fused write
      w_p2s_mix = jnp.eye(k_streams, dtype=jnp.bfloat16)
      w_p2s_read = jax.random.normal(rk[8], (k_streams, d_hc), dtype=jnp.bfloat16)
      w_p2s_write = jax.random.normal(rk[9], (k_streams, d_hc), dtype=jnp.bfloat16)

      def _fwd(x_in):
        if mode == "single_k1":
          rms = jax.lax.rsqrt(jnp.mean(x_in.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(jnp.bfloat16)
          y = _latent_moe_core(x_in * rms * w_rms)
          return x_in + y

        if mode in ("unfused_dhc", "unfused_mhc_sinkhorn4"):
          # Materialize dynamic per-token routing from stream average -> HBM round-trips on [B, T, K, D]!
          x_mean = jnp.mean(x_in, axis=2)  # [B, T, D]
          dyn_logits = jnp.dot(x_mean, w_dyn).astype(jnp.float32)
          a_raw = dyn_logits[..., : k_streams * k_streams].reshape(b_hc, t_hc, k_streams, k_streams)
          alpha = dyn_logits[..., k_streams * k_streams : k_streams * (k_streams + 1)]
          beta = dyn_logits[..., k_streams * (k_streams + 1) :]
          if mode == "unfused_mhc_sinkhorn4":
            # 4-iteration Sinkhorn-Knopp alternating row/col log-domain normalization on VPU
            log_a = a_raw
            for _ in range(4):
              log_a = log_a - jax.scipy.special.logsumexp(log_a, axis=-1, keepdims=True)
              log_a = log_a - jax.scipy.special.logsumexp(log_a, axis=-2, keepdims=True)
            a_mat = jnp.exp(log_a).astype(jnp.bfloat16)
          else:
            a_mat = jax.nn.tanh(a_raw).astype(jnp.bfloat16)
          alpha_b = jax.nn.softmax(alpha, axis=-1).astype(jnp.bfloat16)
          beta_b = jax.nn.sigmoid(beta).astype(jnp.bfloat16)

          # Stream mix: [B, T, K, K] x [B, T, K, D] -> materializes [B, T, K, D] in HBM
          x_mixed = jnp.einsum("btij,btjd->btid", a_mat, x_in, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
          # Stream contraction for layer input: [B, T, K] x [B, T, K, D] -> [B, T, D]
          h_in = jnp.einsum("btk,btkd->btd", alpha_b, x_in, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
          rms = jax.lax.rsqrt(jnp.mean(h_in.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(jnp.bfloat16)
          y = _latent_moe_core(h_in * rms * w_rms)
          # Stream expansion: [B, T, K, D] + [B, T, K, 1] * [B, T, 1, D] -> materializes [B, T, K, D]
          return x_mixed + beta_b[..., None] * y[:, :, None, :]

        # Fused TPU-HC (P2S Sphere Parameterization + Fused Stream-RMSNorm Contraction + Fused Epilogue):
        # 1. Normalize static P2S sphere read/write weights once per layer (O(K*D), independent of T!)
        u_read = w_p2s_read * jax.lax.rsqrt(jnp.sum(w_p2s_read.astype(jnp.float32) ** 2, axis=0, keepdims=True) + 1e-6).astype(
            jnp.bfloat16
        )
        v_write = w_p2s_write * jax.lax.rsqrt(
            jnp.sum(w_p2s_write.astype(jnp.float32) ** 2, axis=0, keepdims=True) + 1e-6
        ).astype(jnp.bfloat16)
        # 2. Fused Stream Contraction + RMSNorm in VMEM: [B, T, K, D] * [K, D] -> [B, T, D]
        h_in = jnp.einsum("btkd,kd->btd", x_in, u_read, preferred_element_type=jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(h_in * h_in, axis=-1, keepdims=True) + 1e-6)
        x_norm = (h_in * rms).astype(jnp.bfloat16) * w_rms
        # 3. LatentMoE core on single collapsed stream [B, T, D]
        y = _latent_moe_core(x_norm)
        # 4. Orthogonal stream residual + rank-1 outer update in VMEM
        x_orth = jnp.einsum("ij,btjd->btid", w_p2s_mix, x_in, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return x_orth + y[:, :, None, :] * v_write[None, None, :, :]

      def _fwd_bwd(x_in, do_in):
        out, vjp_fn = jax.vjp(_fwd, x_in)
        (dx,) = vjp_fn(do_in)
        return out, dx

      return _fwd_bwd, (x_s, do_s)

    hc_bench_arms = [
        ("H1_SingleStream_K1_LatentMoE_Baseline", "single_k1", 1),
        ("H2_Unfused_DHC_K4_Dynamic_LatentMoE", "unfused_dhc", 4),
        ("H3_Unfused_mHC_K4_Sinkhorn4_LatentMoE", "unfused_mhc_sinkhorn4", 4),
        ("H4_Fused_TPU_HC_K4_P2S_LatentMoE", "fused_tpu_hc", 4),
        ("H5_Fused_TPU_HC_K2_BF16Stream_LatentMoE", "fused_tpu_hc", 2),
    ]

    k1_ms_ref = None
    mhc_ms_ref = None
    results_hc = {}
    for h_label, h_mode, k_st in hc_bench_arms:
      try:
        h_fn, h_args = _make_hc_fwd_bwd(h_mode, k_st)
        j_h = jax.jit(h_fn)
        t_ms, _, hlo_h = _time_and_profile_hlo(j_h, *h_args, warmup=2, iters=10)
        results_hc[h_label] = (t_ms, hlo_h)
        if k1_ms_ref is None:
          k1_ms_ref = t_ms
        if h_label == "H3_Unfused_mHC_K4_Sinkhorn4_LatentMoE":
          mhc_ms_ref = t_ms
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[{h_label:42s}] failed: {type(exc).__name__}: {exc}", flush=True)

    for h_label, _, _ in hc_bench_arms:
      if h_label in results_hc:
        t_ms, hlo_h = results_hc[h_label]
        ovh_vs_k1 = ((t_ms / k1_ms_ref) - 1.0) * 100.0 if k1_ms_ref else 0.0
        sp_vs_mhc = (mhc_ms_ref / t_ms) if (mhc_ms_ref and t_ms > 0) else 1.0
        hbm_mb = hlo_h["gb_accessed"] * 1024.0
        bw_gbps = (hlo_h["gb_accessed"] / (t_ms / 1000.0)) if t_ms > 0 else 0.0
        tflops_h = hlo_h.get("tflops", 0.0) / (t_ms / 1000.0) if t_ms > 0 else 0.0
        print(
            f"[{h_label:42s}] fwd+bwd={t_ms:6.3f} ms ({ovh_vs_k1:+5.1f}% vs K=1, {sp_vs_mhc:4.2f}x vs mHC-Sinkhorn) "
            f"| HBM={hbm_mb:6.1f} MB | BW={bw_gbps:6.1f} GB/s | TFLOP/s={tflops_h:6.1f}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # PART 2: 250-Step Deep 8-Layer JAX Signal-Gain Stability & Convergence Ablation on TPU v7x
    # -------------------------------------------------------------------------
    print(
        "\n--- PART 2: 250-Step Deep 8-Layer JAX Signal-Gain Stability & Convergence (K=1 vs. Unconstrained DHC vs. mHC Sinkhorn vs. TPU-HC P2S) ---",
        flush=True,
    )
    vocab_h = 64
    seq_h = 64
    d_h = 128
    n_layers_h = 8
    b_tr_h = 64
    b_val_h = 256
    steps_ch_h = 50
    n_ch_h = 5
    tot_steps_h = steps_ch_h * n_ch_h

    def _gen_deep_assoc_batch(seed_v: int, bsz: int):
      rng = np.random.RandomState(seed_v)
      toks = np.zeros((bsz, seq_h + 1), dtype=np.int32)
      amask = np.zeros((bsz, seq_h), dtype=np.float32)
      for bi in range(bsz):
        kvt = {}
        pos = 0
        while pos < seq_h - 3:
          if len(kvt) < 5 or rng.rand() < 0.55:
            kt = rng.randint(1, 15)
            vt = rng.randint(17, 55)
            toks[bi, pos] = kt
            toks[bi, pos + 1] = vt
            kvt[kt] = vt
            pos += 2
          else:
            ak = list(kvt.keys())
            kt = ak[rng.randint(0, len(ak))]
            vt = kvt[kt]
            toks[bi, pos] = 0
            toks[bi, pos + 1] = kt
            toks[bi, pos + 2] = vt
            amask[bi, pos + 1] = 1.0
            pos += 3
      return jnp.asarray(toks[:, :-1], jnp.int32), jnp.asarray(toks[:, 1:], jnp.int32), jnp.asarray(amask, jnp.float32)

    val_xh, val_yh, val_mh = _gen_deep_assoc_batch(8888, b_val_h)
    tr_h_list = [_gen_deep_assoc_batch(3000 + i, b_tr_h) for i in range(steps_ch_h)]
    tr_xh = jnp.stack([b[0] for b in tr_h_list], axis=0)
    tr_yh = jnp.stack([b[1] for b in tr_h_list], axis=0)
    tr_mh = jnp.stack([b[2] for b in tr_h_list], axis=0)

    def _run_hc_convergence_arm(arm_name: str, hc_mode: str, k_st: int = 4):
      ks = jax.random.split(jax.random.PRNGKey(505), 4 + n_layers_h)

      def _init_hc_layer(kl):
        subk = jax.random.split(kl, 8)
        # Initialize unconstrained DHC stream matrix near identity + noise so multi-layer product can amplify
        a_init = jnp.eye(k_st, dtype=jnp.float32) * 1.12 + jax.random.normal(subk[5], (k_st, k_st), jnp.float32) * 0.15
        skew_init = jax.random.normal(subk[6], (k_st, k_st), jnp.float32) * 0.15
        return {
            "w_in": jax.random.normal(subk[0], (d_h, 3 * d_h), jnp.float32) * (d_h**-0.5),
            "w_c": jnp.array([[0.0], [0.2], [0.6], [0.9]], jnp.float32) + jax.random.normal(subk[1], (4, d_h), jnp.float32) * 0.05,
            "w_out": jax.random.normal(subk[2], (d_h, d_h), jnp.float32) * (0.20 * (d_h**-0.5)),
            "w_f1": jax.random.normal(subk[3], (d_h, 2 * d_h), jnp.float32) * (d_h**-0.5),
            "w_f2": jax.random.normal(subk[4], (d_h, d_h), jnp.float32) * (0.20 * (d_h**-0.5)),
            "a_mat": a_init,
            "skew": skew_init,
            "u_read": jnp.ones((k_st, d_h), jnp.float32) / jnp.sqrt(float(k_st))
            + jax.random.normal(subk[7], (k_st, d_h), jnp.float32) * 0.05,
            "v_write": jnp.ones((k_st, d_h), jnp.float32) / jnp.sqrt(float(k_st)),
        }

      params_h = {
          "emb": jax.random.normal(ks[0], (vocab_h, d_h), jnp.float32) * 1.0,
          "layers": [_init_hc_layer(ks[2 + li]) for li in range(n_layers_h)],
          "head": jax.random.normal(ks[1], (d_h, vocab_h), jnp.float32) * (d_h**-0.5),
      }

      def _conv4_h(xs, wc):
        xp = jnp.pad(xs, ((0, 0), (3, 0), (0, 0)))
        return xp[:, 0:seq_h, :] * wc[0] + xp[:, 1 : seq_h + 1, :] * wc[1] + xp[:, 2 : seq_h + 2, :] * wc[2] + xp[:, 3 : seq_h + 3, :] * wc[3]

      def _subblock_op(h_1d, lp):
        rms1 = jax.lax.rsqrt(jnp.mean(h_1d * h_1d, axis=-1, keepdims=True) + 1e-5)
        qkv = jnp.dot(h_1d * rms1, lp["w_in"])
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = _conv4_h(q, lp["w_c"])
        k = _conv4_h(k, lp["w_c"])
        att = jnp.einsum("bqd,bkd->bqk", q, k) * (d_h**-0.5)
        qi = jax.lax.broadcasted_iota(jnp.int32, (seq_h, 1), 0)
        ki = jax.lax.broadcasted_iota(jnp.int32, (1, seq_h), 1)
        att = jnp.where(qi >= ki, att, -1e9)
        y_att = jnp.dot(jax.nn.softmax(att, axis=-1), v)
        h_mid = h_1d + jnp.dot(y_att, lp["w_out"])
        rms2 = jax.lax.rsqrt(jnp.mean(h_mid * h_mid, axis=-1, keepdims=True) + 1e-5)
        gu = jnp.dot(h_mid * rms2, lp["w_f1"])
        gf, uf = jnp.split(gu, 2, axis=-1)
        return jnp.dot(jax.nn.silu(gf) * uf, lp["w_f2"]) + jnp.dot(y_att, lp["w_out"])

      def _get_mix_matrix(lp):
        if hc_mode == "dhc_unconstrained":
          return lp["a_mat"]
        if hc_mode == "mhc_sinkhorn4":
          la = lp["a_mat"]
          for _ in range(4):
            la = la - jax.scipy.special.logsumexp(la, axis=-1, keepdims=True)
            la = la - jax.scipy.special.logsumexp(la, axis=-2, keepdims=True)
          return jnp.exp(la)
        # TPU-HC P2S + Cayley orthogonal mixing: A = (I - S)(I + S)^-1 where S = skew - skew^T (exact spectral norm = 1.0!)
        s_mat = 0.5 * (lp["skew"] - lp["skew"].T)
        eye = jnp.eye(k_st, dtype=jnp.float32)
        return jnp.linalg.solve(eye + s_mat, eye - s_mat)

      def _fwd_h(p, x_tok):
        h0 = p["emb"][x_tok]
        if hc_mode == "single_k1":
          h = h0
          for lp in p["layers"]:
            h = h + _subblock_op(h, lp)
          rms = jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + 1e-5)
          return jnp.dot(h * rms, p["head"])

        # Expand to K streams: [B, T, K, D]
        x_st = jnp.repeat(h0[:, :, None, :], k_st, axis=2)
        for lp in p["layers"]:
          a_m = _get_mix_matrix(lp)
          if hc_mode == "tpu_hc_p2s":
            # Unit-sphere normalized P2S read/write weights: <u_d, v_d> bounded in [-1, 1]
            u_r = lp["u_read"] * jax.lax.rsqrt(jnp.sum(lp["u_read"] ** 2, axis=0, keepdims=True) + 1e-6)
            v_w = lp["v_write"] * jax.lax.rsqrt(jnp.sum(lp["v_write"] ** 2, axis=0, keepdims=True) + 1e-6)
          else:
            u_r = lp["u_read"]
            v_w = lp["v_write"]
          h_in = jnp.einsum("btkd,kd->btd", x_st, u_r)
          dy = _subblock_op(h_in, lp)
          x_st = jnp.einsum("ij,btjd->btid", a_m, x_st) + dy[:, :, None, :] * v_w[None, None, :, :]
        h_final = jnp.mean(x_st, axis=2)
        rms = jax.lax.rsqrt(jnp.mean(h_final * h_final, axis=-1, keepdims=True) + 1e-5)
        return jnp.dot(h_final * rms, p["head"])

      def _loss_h(p, xt, yt, am):
        lg = _fwd_h(p, xt)
        lp = jax.nn.log_softmax(lg, axis=-1)
        nll = -jnp.take_along_axis(lp, yt[..., None], axis=-1)[..., 0]
        w = 1.0 + 5.0 * am
        return jnp.sum(nll * w) / jnp.sum(w)

      @jax.jit
      def _train_ch_h(p_in, m_in, v_in, step_b):
        def _body(carry, idx):
          p, m, v = carry
          lv, grads = jax.value_and_grad(_loss_h)(p, tr_xh[idx], tr_yh[idx], tr_mh[idx])
          gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
          # Clip gradients at 5.0 so unconstrained DHC doesn't NaN out completely while recording its raw grad norm
          clip_scale = jnp.minimum(1.0, 5.0 / (gnorm + 1e-6))
          grads = jax.tree_util.tree_map(lambda g_: g_ * clip_scale, grads)
          gs = step_b + idx + 1
          lr = 5e-3 * 0.5 * (1.0 + jnp.cos(jnp.pi * (step_b + idx) / tot_steps_h))
          m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, grads)
          v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * (g_ * g_), v, grads)
          mh = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - 0.9**gs), m)
          vh = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - 0.999**gs), v)
          p = jax.tree_util.tree_map(lambda p_, mh_, vh_: p_ * (1.0 - lr * 1e-3) - lr * mh_ / (jnp.sqrt(vh_) + 1e-8), p, mh, vh)
          return (p, m, v), (lv, gnorm)

        return jax.lax.scan(_body, (p_in, m_in, v_in), jnp.arange(steps_ch_h))

      @jax.jit
      def _eval_h(pf):
        lg = _fwd_h(pf, val_xh)
        lp = jax.nn.log_softmax(lg, axis=-1)
        tnll = -jnp.take_along_axis(lp, val_yh[..., None], axis=-1)[..., 0]
        tacc = (jnp.argmax(lg, axis=-1) == val_yh).astype(jnp.float32)
        vnll = jnp.mean(tnll)
        anll = jnp.sum(tnll * val_mh) / jnp.maximum(jnp.sum(val_mh), 1.0)
        aacc = jnp.sum(tacc * val_mh) / jnp.maximum(jnp.sum(val_mh), 1.0)
        if hc_mode == "single_k1":
          comp_gain = jnp.float32(1.0)
        else:
          prod_m = jnp.eye(k_st, dtype=jnp.float32)
          for lp_i in pf["layers"]:
            prod_m = jnp.dot(_get_mix_matrix(lp_i), prod_m)
          comp_gain = jnp.linalg.svd(prod_m, compute_uv=False)[0]
        return vnll, anll, aacc, comp_gain

      mc = jax.tree_util.tree_map(jnp.zeros_like, params_h)
      vc = jax.tree_util.tree_map(jnp.zeros_like, params_h)
      pc = params_h
      l_list, g_list = [], []
      for ci in range(n_ch_h):
        (pc, mc, vc), (cl, cg) = _train_ch_h(pc, mc, vc, jnp.int32(ci * steps_ch_h))
        l_list.append(cl)
        g_list.append(cg)
      l_arr = jnp.concatenate(l_list, axis=0)
      g_arr = jnp.concatenate(g_list, axis=0)
      vnll, anll, aacc, c_gain = _eval_h(pc)
      jax.block_until_ready(vnll)
      print(
          f"[{arm_name:42s}] composite_gain={float(c_gain):5.2f}x | max_grad_norm={float(jnp.max(g_arr)):6.2f} "
          f"| train_nll: s0={float(l_arr[0]):.4f} -> s250={float(l_arr[-1]):.4f} "
          f"| val_nll={float(vnll):.4f} | assoc_nll={float(anll):.4f} (acc={100*float(aacc):5.1f}%)",
          flush=True,
      )

    for hc_c_arm in [
        ("HC1_SingleStream_K1_8Layer", "single_k1", 1),
        ("HC2_Unconstrained_DHC_K4_8Layer", "dhc_unconstrained", 4),
        ("HC3_Sinkhorn4_mHC_K4_8Layer", "mhc_sinkhorn4", 4),
        ("HC4_CoDesigned_TPU_HC_P2S_K4_8Layer", "tpu_hc_p2s", 4),
        ("HC5_CoDesigned_TPU_HC_P2S_K2_8Layer", "tpu_hc_p2s", 2),
    ]:
      try:
        _run_hc_convergence_arm(*hc_c_arm)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[{hc_c_arm[0]:42s}] failed: {type(exc).__name__}: {exc}", flush=True)

  def test_9_unified_v7x_train_prefill_decode_roofline_study(self):
    """Shard 9: Unified Step-1 Baseline vs Step-2 Co-Design Roofline & Empirical Study (Train + Prefill + Decode)."""
    print(
        f"\n================================================================================"
        f"\nSHARD 9: TPU v7x ({self.num_devs}x {self.dev_kind}) UNIFIED BASELINE vs. CO-DESIGN ROOFLINE & EMPIRICAL STUDY"
        f"\n  Step 1: Baseline Roofline & Empirical (Tier 0 Naive -> Tier 1 Optimized)"
        f"\n  Step 2: Tier 2 Co-Design Roofline & Empirical (Training + Prefill + Decode)"
        f"\n================================================================================",
        flush=True,
    )

    orig_fwd, orig_bwd = kda_fused._fwd_kernel, kda_fused._bwd_kernel
    kda_fused._fwd_kernel = _fwd_kernel_v7x_opt
    kda_fused._bwd_kernel = _bwd_kernel_v7x_opt

    try:
      # -----------------------------------------------------------------------
      # STUDY 1: Value-to-Key Head Sharing (GVA) & MXU Tile Alignment (H x dk x dv)
      #          Evaluated across Training (Fwd+Bwd) & Inference Prefill (Fwd) at T=8192 & T=32768
      # -----------------------------------------------------------------------
      print(
          "\n=== STUDY 1: GVA & MXU Tile Alignment (Step 1 Baselines vs Step 2 Co-Design, T=8192 & T=32768) ===",
          flush=True,
      )
      b = 1
      keys = jax.random.split(jax.random.PRNGKey(901), 8)
      for seq_len in (8192, 32768):
        print(f"\n--- Study 1 Sequence Length T = {seq_len} ---", flush=True)
        resets = jnp.zeros((b, seq_len), dtype=bool)
        study1_specs = [
            # (label, tier, h_packed, dk, dv_packed, c, impl_kind)
            ("S1_B0_Qwen3_16K32V_128x128_XLA", "Step1_Tier0", 32, 128, 128, 64, "gdn_xla"),
            ("S1_B1_Qwen3_16K32V_128x128_Pallas", "Step1_Tier1", 32, 128, 128, 64, "gdn_pallas"),
            ("S1_B0_OLMo35_8K8V_256x512_XLA", "Step1_Tier0", 8, 256, 512, 128, "kda_xla"),
            ("S1_B1_OLMo35_8K8V_256x512_Pallas", "Step1_Tier1", 8, 256, 512, 128, "kda_pallas"),
            ("S1_C1_GVA2to1_8K16V_256x256_SDN_G4", "Step2_Tier2", 8, 256, 512, 128, "sdn_g4_pallas"),
            ("S1_C2_GVA4to1_4K16V_256x256_SDN_G4", "Step2_Tier2", 4, 256, 1024, 128, "sdn_g4_pallas"),
            ("S1_C3_Sym256_8K8V_256x256_SDN_G4", "Step2_Tier2", 8, 256, 256, 128, "sdn_g4_pallas"),
        ]
        for label, tier, h, dk, dv, c, impl_kind in study1_specs:
          if seq_len == 32768 and impl_kind.endswith("_xla"):
            continue
          try:
            q = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16) * (dk**-0.5)
            k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16)
            v = (jax.random.normal(keys[2], (b, seq_len, h, dv), jnp.bfloat16) * 0.1).astype(jnp.bfloat16)
            beta = (2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, h), jnp.bfloat16))).astype(jnp.bfloat16)
            dy = jax.random.normal(keys[5], (b, seq_len, h, dv), jnp.bfloat16) * 0.1

            is_kda_roof = impl_kind in ("kda_xla", "kda_pallas")
            roof = compute_core_roofline_analytical(
                b,
                seq_len,
                h,
                dk,
                dv,
                c,
                is_kda=is_kda_roof,
                state_f32=True,
                state_stride=0 if is_kda_roof else 1,
                outer_block=32 if is_kda_roof else None,
                v7x_stacked_opt=(not impl_kind.endswith("_xla")),
            )

            if impl_kind == "gdn_xla":
              g_arg = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h), jnp.float32) - 2.0)
              fn = lambda q_, k_, v_, g_, b_, _c=c: qwen3.jax_chunk_gated_delta_rule(
                  q_, k_, v_, g_, b_, chunk_size=_c, compute_dtype=jnp.bfloat16
              )[0]
            elif impl_kind == "gdn_pallas":
              g_arg = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h), jnp.float32) - 2.0)
              fn = lambda q_, k_, v_, g_, b_, _c=c: run_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, _c, "float32")
            elif impl_kind == "kda_xla":
              g_arg = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, dk), jnp.float32) - 2.0)
              fn = lambda q_, k_, v_, g_, b_, _c=c: olmoe3._delta_rule_chunked_subblock(
                  q_, k_, v_, g_, b_, resets, _c, "float32", sub_block=16, intra_vjp=True
              )
            elif impl_kind == "kda_pallas":
              g_arg = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, dk), jnp.float32) - 2.0)
              fn = lambda q_, k_, v_, g_, b_, _c=c: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  g_,
                  b_,
                  resets=resets,
                  chunk_size=_c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=0,
              )
            else:
              g_arg = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, 4), jnp.float32) - 2.0)
              fn = lambda q_, k_, v_, g_, b_, _c=c: _kda_tile_grouped_decay_core(q_, k_, v_, g_, b_, resets, _c, 4)

            j_prefill = jax.jit(fn)
            j_train = jax.jit(
                jax.value_and_grad(
                    lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy.astype(jnp.float32)),
                    argnums=(0, 1, 2, 3, 4),
                )
            )
            t_prefill, _, hlo_pf = _time_and_profile_hlo(j_prefill, q, k, v, g_arg, beta, warmup=1, iters=5)
            t_train, _, hlo_tr = _time_and_profile_hlo(j_train, q, k, v, g_arg, beta, warmup=1, iters=5)
            pf_floor_ms = max(roof["fwd_hbm_pallas_ms"], roof["fwd_comp_tile_ms"])
            pf_attain = 100.0 * pf_floor_ms / max(t_prefill, 1e-6)
            state_mb = (h * dk * dv * 4) / (1024.0 * 1024.0)
            print(
                f"[{tier:11s} | {label:36s}] State={state_mb:5.1f}MB | "
                f"Prefill={t_prefill:6.3f}ms (Floor={pf_floor_ms:5.3f}ms, Attain={pf_attain:5.1f}%, HBM={hlo_pf['gb_accessed']*1e3:5.1f}MB) | "
                f" + "
                + format_perf_vs_roofline("Train_FwdBwd", t_prefill, t_train, roof, hlo_tr),
                flush=True,
            )
          except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[{label:36s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # STUDY 2: Decay Granularity Spectrum (G = 1, 2, 4, 16, 256)
      #          Evaluated across Training (Fwd+Bwd) & Inference Prefill (Fwd) at T=8192, H=8, dk=256, dv=512
      # -----------------------------------------------------------------------
      print(
          "\n=== STUDY 2: Decay Granularity Spectrum (G = 1, 2, 4, 16, 256 at T=8192, H=8, dk=256, dv=512, C=128) ===",
          flush=True,
      )
      seq_len = 8192
      h, dk, dv, c = 8, 256, 512, 128
      resets = jnp.zeros((b, seq_len), dtype=bool)
      q = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16) * (dk**-0.5)
      k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16)
      v = (jax.random.normal(keys[2], (b, seq_len, h, dv), jnp.bfloat16) * 0.1).astype(jnp.bfloat16)
      g_full = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, dk), jnp.float32) - 2.0)
      beta = (2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, h), jnp.bfloat16))).astype(jnp.bfloat16)
      dy = jax.random.normal(keys[5], (b, seq_len, h, dv), jnp.bfloat16) * 0.1

      for g_groups, tier_tag, g_label in (
          (1, "Step1_Tier1", "S2_G1_Scalar_GDN (1 decay/head)"),
          (2, "Step2_Tier2", "S2_G2_TileGrouped_SDN (128ch/group)"),
          (4, "Step2_Tier2", "S2_G4_TileGrouped_SDN (64ch/group)"),
          (16, "Step2_Tier2", "S2_G16_Subblock_SDN (16ch/group)"),
          (256, "Step1_Tier1", "S2_G256_PerChannel_KDA (1ch/group)"),
      ):
        try:
          if g_groups == 1:
            g_in = g_full[..., 0]
            fn = lambda q_, k_, v_, g_, b_: run_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, c, "float32")
            roof_g = compute_core_roofline_analytical(b, seq_len, h, dk, dv, c, is_kda=False, state_f32=True, state_stride=1, outer_block=None, v7x_stacked_opt=True)
          elif g_groups == 256:
            g_in = g_full
            fn = lambda q_, k_, v_, g_, b_: kda.chunk_kda(
                q_, k_, v_, g_, b_, resets=resets, chunk_size=c, state_dtype=jnp.float32, layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=0
            )
            roof_g = compute_core_roofline_analytical(b, seq_len, h, dk, dv, c, is_kda=True, state_f32=True, state_stride=0, outer_block=32, v7x_stacked_opt=True)
          else:
            g_in = g_full[..., :g_groups]
            if g_groups in (2, 4):
              fn = lambda q_, k_, v_, g_, b_, _gg=g_groups: _kda_tile_grouped_decay_core(q_, k_, v_, g_, b_, resets, c, _gg)
              roof_g = compute_core_roofline_analytical(b, seq_len, h, dk, dv, c, is_kda=False, state_f32=True, state_stride=1, outer_block=None, v7x_stacked_opt=True)
            else:
              # G=16 (16ch/group): expand in VMEM/register to [B, T, H, dk] and run fused v7x Pallas
              fn = lambda q_, k_, v_, g_, b_, _gg=g_groups: kda.chunk_kda(
                  q_,
                  k_,
                  v_,
                  jnp.repeat(g_, dk // _gg, axis=-1),
                  b_,
                  resets=resets,
                  chunk_size=c,
                  state_dtype=jnp.float32,
                  layout="token_major",
                  outer_block=32,
                  output_dtype=jnp.bfloat16,
                  state_stride=0,
              )
              roof_g = compute_core_roofline_analytical(b, seq_len, h, dk, dv, c, is_kda=True, state_f32=True, state_stride=0, outer_block=32, v7x_stacked_opt=True)

          j_pf = jax.jit(fn)
          j_tr = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, g_, b_, _fn=fn: jnp.sum(_fn(q_, k_, v_, g_, b_).astype(jnp.float32) * dy.astype(jnp.float32)),
                  argnums=(0, 1, 2, 3, 4),
              )
          )
          t_pf, _, hlo_pf = _time_and_profile_hlo(j_pf, q, k, v, g_in, beta, warmup=1, iters=5)
          t_tr, _, hlo_tr = _time_and_profile_hlo(j_tr, q, k, v, g_in, beta, warmup=1, iters=5)
          pf_floor = max(roof_g["fwd_hbm_pallas_ms"], roof_g["fwd_comp_tile_ms"])
          print(
              f"[{tier_tag:11s} | {g_label:36s}] Prefill={t_pf:6.3f}ms (Floor={pf_floor:5.3f}ms, Attain={100.0*pf_floor/t_pf:5.1f}%) | "
              + format_perf_vs_roofline("Train_FwdBwd", t_pf, t_tr, roof_g, hlo_tr),
              flush=True,
          )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{g_label:36s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # STUDY 3: Non-Core Prologue & Epilogue Co-Design (Full Layer: Projections + Conv1D + Core + Norm*Gate + OutProj)
      #          Compares Step 1 (Tier 0 & Tier 1 2-Stage Gate) vs Step 2 (1-Stage Fused Gate + QKV vs QK-Only Conv + Fused Epilogue)
      # -----------------------------------------------------------------------
      print(
          "\n=== STUDY 3: Full-Layer Prologue & Epilogue Co-Design (B=1/core, T=8192, d_model=1792, H=8, dk=256, dv=512) ===",
          flush=True,
      )
      d_model = 1792
      x_layer = jax.random.normal(keys[0], (b, seq_len, d_model), jnp.bfloat16)
      dy_layer = jax.random.normal(keys[1], (b, seq_len, d_model), jnp.bfloat16) * 0.1
      # Layer weights for 2-stage vs 1-stage co-design
      w_in_2st = jax.random.normal(keys[2], (d_model, 2048 + 2048 + 4096 + 512 + 512 + 8), jnp.bfloat16) * 0.02
      w_f2_kda = jax.random.normal(keys[3], (512, 2048), jnp.bfloat16) * 0.02
      w_g2_kda = jax.random.normal(keys[4], (512, 4096), jnp.bfloat16) * 0.02
      # 1-Stage Fused Input + Gate for SDN G=4 (q:2048, k:2048, v:4096, g_out:4096, f_sdn4:32, beta:8 -> total=12,328!)
      w_in_1st_sdn4 = jax.random.normal(keys[2], (d_model, 2048 + 2048 + 4096 + 4096 + 32 + 8), jnp.bfloat16) * 0.02
      # 1-Stage Fused Input + Gate for Sym-256 SDN G=4 (q:2048, k:2048, v:2048, g_out:2048, f_sdn4:32, beta:8 -> total=8,232!)
      w_in_1st_sym256 = jax.random.normal(keys[2], (d_model, 2048 + 2048 + 2048 + 2048 + 32 + 8), jnp.bfloat16) * 0.02
      w_q_conv = jax.random.normal(keys[5], (2048, 4), jnp.bfloat16) * 0.02
      w_k_conv = jax.random.normal(keys[5], (2048, 4), jnp.bfloat16) * 0.02
      w_v_conv = jax.random.normal(keys[5], (4096, 4), jnp.bfloat16) * 0.02
      w_v_conv_sym = jax.random.normal(keys[5], (2048, 4), jnp.bfloat16) * 0.02
      w_out_512 = jax.random.normal(keys[6], (4096, d_model), jnp.bfloat16) * 0.02
      w_out_256 = jax.random.normal(keys[6], (2048, d_model), jnp.bfloat16) * 0.02
      rms_scale_512 = jnp.ones((8, 512), jnp.bfloat16)
      rms_scale_256 = jnp.ones((8, 256), jnp.bfloat16)

      def _depthwise_conv4_bf16(u, w_c):
        up = jnp.pad(u, ((0, 0), (3, 0), (0, 0)))
        return (
            up[:, 0:seq_len, :] * w_c[:, 0]
            + up[:, 1 : seq_len + 1, :] * w_c[:, 1]
            + up[:, 2 : seq_len + 2, :] * w_c[:, 2]
            + up[:, 3 : seq_len + 3, :] * w_c[:, 3]
        ).astype(jnp.bfloat16)

      def _layer_step1_tier1_2stage_kda(x_, w_in_, wf2_, wg2_, wqc_, wkc_, wvc_, wout_):
        all_in = jnp.dot(x_, w_in_)
        qi, ki, vi, f1, g1, bi = jnp.split(all_in, [2048, 4096, 8192, 8704, 9216], axis=-1)
        q_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(qi, wqc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16) * (256**-0.5)
        k_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(ki, wkc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16)
        v_ = _depthwise_conv4_bf16(vi, wvc_).reshape(b, seq_len, 8, 512)
        g_ = -jax.nn.softplus(jnp.dot(f1, wf2_).reshape(b, seq_len, 8, 256).astype(jnp.float32) - 2.0)
        beta_ = (2.0 * jax.nn.sigmoid(bi.astype(jnp.float32))).astype(jnp.bfloat16)
        core_out = kda.chunk_kda(
            q_, k_, v_, g_, beta_, resets=resets, chunk_size=128, state_dtype=jnp.float32, layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=0
        )
        gate_ = jax.nn.sigmoid(jnp.dot(g1, wg2_).reshape(b, seq_len, 8, 512).astype(jnp.float32)).astype(jnp.bfloat16)
        rms_ = jax.lax.rsqrt(jnp.mean(core_out.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(jnp.bfloat16)
        y_ = (core_out * rms_ * rms_scale_512 * gate_).reshape(b, seq_len, 4096)
        return jnp.dot(y_, wout_)

      def _layer_step2_1stage_qkvconv_sdn4(x_, w_all_, wqc_, wkc_, wvc_, wout_):
        all_in = jnp.dot(x_, w_all_)
        qi, ki, vi, go, fi, bi = jnp.split(all_in, [2048, 4096, 8192, 12288, 12320], axis=-1)
        q_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(qi, wqc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16) * (256**-0.5)
        k_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(ki, wkc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16)
        v_ = _depthwise_conv4_bf16(vi, wvc_).reshape(b, seq_len, 8, 512)
        g_grp = -jax.nn.softplus(fi.reshape(b, seq_len, 8, 4).astype(jnp.float32) - 2.0)
        beta_ = (2.0 * jax.nn.sigmoid(bi.astype(jnp.float32))).astype(jnp.bfloat16)
        core_out = _kda_tile_grouped_decay_core(q_, k_, v_, g_grp, beta_, resets, 128, 4)
        gate_ = jax.nn.sigmoid(go.reshape(b, seq_len, 8, 512).astype(jnp.float32)).astype(jnp.bfloat16)
        rms_ = jax.lax.rsqrt(jnp.mean(core_out.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(jnp.bfloat16)
        y_ = (core_out * rms_ * rms_scale_512 * gate_).reshape(b, seq_len, 4096)
        return jnp.dot(y_, wout_)

      def _layer_step2_1stage_qkonlyconv_sdn4(x_, w_all_, wqc_, wkc_, wout_):
        all_in = jnp.dot(x_, w_all_)
        qi, ki, vi, go, fi, bi = jnp.split(all_in, [2048, 4096, 8192, 12288, 12320], axis=-1)
        q_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(qi, wqc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16) * (256**-0.5)
        k_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(ki, wkc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16)
        v_ = jax.nn.silu(vi).reshape(b, seq_len, 8, 512)
        g_grp = -jax.nn.softplus(fi.reshape(b, seq_len, 8, 4).astype(jnp.float32) - 2.0)
        beta_ = (2.0 * jax.nn.sigmoid(bi.astype(jnp.float32))).astype(jnp.bfloat16)
        core_out = _kda_tile_grouped_decay_core(q_, k_, v_, g_grp, beta_, resets, 128, 4)
        gate_ = jax.nn.sigmoid(go.reshape(b, seq_len, 8, 512).astype(jnp.float32)).astype(jnp.bfloat16)
        rms_ = jax.lax.rsqrt(jnp.mean(core_out.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(jnp.bfloat16)
        y_ = (core_out * rms_ * rms_scale_512 * gate_).reshape(b, seq_len, 4096)
        return jnp.dot(y_, wout_)

      def _layer_step2_1stage_sym256_qkvconv_sdn4(x_, w_all_, wqc_, wkc_, wvc_, wout_):
        all_in = jnp.dot(x_, w_all_)
        qi, ki, vi, go, fi, bi = jnp.split(all_in, [2048, 4096, 6144, 8192, 8224], axis=-1)
        q_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(qi, wqc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16) * (256**-0.5)
        k_ = olmoe3._l2_normalize(_depthwise_conv4_bf16(ki, wkc_).reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16)
        v_ = _depthwise_conv4_bf16(vi, wvc_).reshape(b, seq_len, 8, 256)
        g_grp = -jax.nn.softplus(fi.reshape(b, seq_len, 8, 4).astype(jnp.float32) - 2.0)
        beta_ = (2.0 * jax.nn.sigmoid(bi.astype(jnp.float32))).astype(jnp.bfloat16)
        core_out = _kda_tile_grouped_decay_core(q_, k_, v_, g_grp, beta_, resets, 128, 4)
        gate_ = jax.nn.sigmoid(go.reshape(b, seq_len, 8, 256).astype(jnp.float32)).astype(jnp.bfloat16)
        rms_ = jax.lax.rsqrt(jnp.mean(core_out.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(jnp.bfloat16)
        y_ = (core_out * rms_ * rms_scale_256 * gate_).reshape(b, seq_len, 2048)
        return jnp.dot(y_, wout_)

      study3_arms = [
          ("Step1_Tier1", "S3_L1_2StageGate_QKVConv_KDA_256x512", _layer_step1_tier1_2stage_kda, (x_layer, w_in_2st, w_f2_kda, w_g2_kda, w_q_conv, w_k_conv, w_v_conv, w_out_512)),
          ("Step2_Tier2", "S3_L2_1StageGate_QKVConv_SDN_G4_256x512", _layer_step2_1stage_qkvconv_sdn4, (x_layer, w_in_1st_sdn4, w_q_conv, w_k_conv, w_v_conv, w_out_512)),
          ("Step2_Tier2", "S3_L3_1StageGate_QKOnlyConv_SDN_G4_256x512", _layer_step2_1stage_qkonlyconv_sdn4, (x_layer, w_in_1st_sdn4, w_q_conv, w_k_conv, w_out_512)),
          ("Step2_Tier2", "S3_L4_1StageGate_QKVConv_SDN_G4_Sym256", _layer_step2_1stage_sym256_qkvconv_sdn4, (x_layer, w_in_1st_sym256, w_q_conv, w_k_conv, w_v_conv_sym, w_out_256)),
      ]
      for tier_tag, s3_label, s3_fn, s3_args in study3_arms:
        try:
          j_pf = jax.jit(s3_fn)
          j_tr = jax.jit(
              jax.value_and_grad(
                  lambda *a, _fn=s3_fn: jnp.sum(_fn(*a).astype(jnp.float32) * dy_layer.astype(jnp.float32)),
                  argnums=tuple(range(len(s3_args))),
              )
          )
          t_pf, _, hlo_pf = _time_and_profile_hlo(j_pf, *s3_args, warmup=1, iters=5)
          t_tr, _, hlo_tr = _time_and_profile_hlo(j_tr, *s3_args, warmup=1, iters=5)
          pf_hbm_ms = (hlo_pf["gb_accessed"] * 1e3) / V7X_HBM_BW_GBPS
          pf_mxu_ms = hlo_pf["gflops"] / V7X_MXU_BF16_TFLOPS_P7
          tr_hbm_ms = (hlo_tr["gb_accessed"] * 1e3) / V7X_HBM_BW_GBPS
          tr_mxu_ms = hlo_tr["gflops"] / V7X_MXU_BF16_TFLOPS_P7
          print(
              f"[{tier_tag:11s} | {s3_label:42s}] "
              f"Prefill={t_pf:6.3f}ms (x24={24*t_pf:5.1f}ms, HBM={hlo_pf['gb_accessed']*1e3:5.1f}MB [{pf_hbm_ms:4.2f}ms], MXU={hlo_pf['gflops']:6.1f}GF [{pf_mxu_ms:4.2f}ms]) | "
              f"Train={t_tr:6.3f}ms (x24={24*t_tr:6.1f}ms, HBM={hlo_tr['gb_accessed']*1e3:6.1f}MB [{tr_hbm_ms:4.2f}ms], MXU={hlo_tr['gflops']:6.1f}GF [{tr_mxu_ms:4.2f}ms])",
              flush=True,
          )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{s3_label:42s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # STUDY 4: Inference Decode & Speculative Tree-Verify State Bandwidth Co-Design
      #          Step 1 Baseline (FP32 State, T_ver=1) vs Step 2 Co-Design (BF16 / Block-Scaled FP8 State + T_ver=5, 9)
      #          Evaluated across Batch Sizes B in {1, 8, 32, 64} + 64-Step Numerical SNR Verification
      # -----------------------------------------------------------------------
      print(
          "\n=== STUDY 4: Inference Decode & Speculative Tree-Verify Roofline & Empirical Benchmark (B in {1, 8, 32, 64}, T_ver in {1, 5, 9}) ===",
          flush=True,
      )

      def _decode_step_fp32_or_bf16(s_state, q_tok, k_tok, v_tok, g_tok, beta_tok):
        # s_state: [B, H, dk, dv] (float32 or bfloat16)
        # q_tok, k_tok: [B, T_ver, H, dk], v_tok: [B, T_ver, H, dv], g_tok: [B, T_ver, H, dk], beta_tok: [B, T_ver, H]
        st_dtype = s_state.dtype
        f32 = jnp.float32

        def _step_one(st, xs):
          qt, kt, vt, gt, bt = xs
          st_f = st.astype(f32) * jnp.exp(gt.astype(f32))[..., None]
          kv_pred = jnp.einsum("bhdv,bhd->bhv", st_f, kt.astype(f32))
          delta = bt.astype(f32)[..., None] * (vt.astype(f32) - kv_pred)
          st_new = st_f + jnp.einsum("bhd,bhv->bhdv", kt.astype(f32), delta)
          yt = jnp.einsum("bhdv,bhd->bhv", st_new, qt.astype(f32)).astype(jnp.bfloat16)
          return st_new.astype(st_dtype), yt

        xs_scan = (
            jnp.moveaxis(q_tok, 1, 0),
            jnp.moveaxis(k_tok, 1, 0),
            jnp.moveaxis(v_tok, 1, 0),
            jnp.moveaxis(g_tok, 1, 0),
            jnp.moveaxis(beta_tok, 1, 0),
        )
        st_final, y_seq = jax.lax.scan(_step_one, s_state, xs_scan)
        return st_final, jnp.moveaxis(y_seq, 0, 1)

      def _decode_step_fp8_block_scaled(s_q_fp8, s_scale_bf16, q_tok, k_tok, v_tok, g_tok, beta_tok):
        # s_q_fp8: [B, H, dk, dv] float8_e4m3fn, s_scale_bf16: [B, H, dk, 1] bfloat16
        f32 = jnp.float32
        fp8_max = jnp.float32(448.0)

        # Dequantize once in VMEM at the start of the T_ver speculative verification chunk!
        st_init_f32 = s_q_fp8.astype(f32) * s_scale_bf16.astype(f32)

        def _step_one(st_f, xs):
          qt, kt, vt, gt, bt = xs
          st_dec = st_f * jnp.exp(gt.astype(f32))[..., None]
          kv_pred = jnp.einsum("bhdv,bhd->bhv", st_dec, kt.astype(f32))
          delta = bt.astype(f32)[..., None] * (vt.astype(f32) - kv_pred)
          st_new = st_dec + jnp.einsum("bhd,bhv->bhdv", kt.astype(f32), delta)
          yt = jnp.einsum("bhdv,bhd->bhv", st_new, qt.astype(f32)).astype(jnp.bfloat16)
          return st_new, yt

        xs_scan = (
            jnp.moveaxis(q_tok, 1, 0),
            jnp.moveaxis(k_tok, 1, 0),
            jnp.moveaxis(v_tok, 1, 0),
            jnp.moveaxis(g_tok, 1, 0),
            jnp.moveaxis(beta_tok, 1, 0),
        )
        st_final_f32, y_seq = jax.lax.scan(_step_one, st_init_f32, xs_scan)
        # Re-quantize once at the end of the T_ver chunk before writing S_t back to HBM
        row_amax = jnp.max(jnp.abs(st_final_f32), axis=-1, keepdims=True)
        new_scale = jnp.maximum(row_amax / fp8_max, jnp.float32(1e-8)).astype(jnp.bfloat16)
        new_q_fp8 = jnp.clip(st_final_f32 / new_scale.astype(f32), -fp8_max, fp8_max).astype(jnp.float8_e4m3fn)
        return new_q_fp8, new_scale, jnp.moveaxis(y_seq, 0, 1)

      for b_dec in (1, 8, 32, 64):
        print(f"\n--- Study 4 Decode Batch Size B = {b_dec} (per TensorCore) ---", flush=True)
        decode_configs = [
            # (tier, label, h, dk, dv, state_mode, t_ver)
            ("Step1_Tier1", "D0_Asym256x512_FP32State_Tver1", 8, 256, 512, "fp32", 1),
            ("Step2_Tier2", "D1_Asym256x512_BF16State_Tver1", 8, 256, 512, "bf16", 1),
            ("Step2_Tier2", "D2_Asym256x512_FP8BlockState_Tver1", 8, 256, 512, "fp8", 1),
            ("Step2_Tier2", "D3_Sym256x256_BF16State_Tver1", 8, 256, 256, "bf16", 1),
            ("Step2_Tier2", "D4_Sym256x256_FP8BlockState_Tver1", 8, 256, 256, "fp8", 1),
            ("Step2_Tier2", "D5_Asym256x512_BF16State_SpecTver5", 8, 256, 512, "bf16", 5),
            ("Step2_Tier2", "D6_Asym256x512_FP8BlockState_SpecTver5", 8, 256, 512, "fp8", 5),
            ("Step2_Tier2", "D7_Asym256x512_FP8BlockState_SpecTver9", 8, 256, 512, "fp8", 9),
            ("Step2_Tier2", "D8_Sym256x256_FP8BlockState_SpecTver5", 8, 256, 256, "fp8", 5),
        ]
        for tier_tag, d_label, hd, dkd, dvd, st_mode, t_ver in decode_configs:
          try:
            q_d = olmoe3._l2_normalize(jax.random.normal(keys[0], (b_dec, t_ver, hd, dkd), jnp.bfloat16)).astype(jnp.bfloat16) * (dkd**-0.5)
            k_d = olmoe3._l2_normalize(jax.random.normal(keys[1], (b_dec, t_ver, hd, dkd), jnp.bfloat16)).astype(jnp.bfloat16)
            v_d = (jax.random.normal(keys[2], (b_dec, t_ver, hd, dvd), jnp.bfloat16) * 0.1).astype(jnp.bfloat16)
            g_d = (-jax.nn.softplus(jax.random.normal(keys[3], (b_dec, t_ver, hd, dkd), jnp.float32) - 2.0)).astype(jnp.bfloat16)
            beta_d = (2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b_dec, t_ver, hd), jnp.bfloat16))).astype(jnp.bfloat16)

            b_per_elem = 4.0 if st_mode == "fp32" else (2.0 if st_mode == "bf16" else (1.0 + 2.0 / dvd))
            # State read + write bytes across 1 layer
            st_rw_mb = (2.0 * b_dec * hd * dkd * dvd * b_per_elem) / 1e6
            t_hbm_roof_ms = st_rw_mb / V7X_HBM_BW_GBPS

            if st_mode in ("fp32", "bf16"):
              st_dtype = jnp.float32 if st_mode == "fp32" else jnp.bfloat16
              s_init = jax.random.normal(keys[5], (b_dec, hd, dkd, dvd), st_dtype) * 0.05
              j_dec = jax.jit(_decode_step_fp32_or_bf16)
              t_step_ms, _, hlo_d = _time_and_profile_hlo(j_dec, s_init, q_d, k_d, v_d, g_d, beta_d, warmup=2, iters=10)
            else:
              s_q_init = jax.random.normal(keys[5], (b_dec, hd, dkd, dvd), jnp.bfloat16).astype(jnp.float8_e4m3fn)
              s_sc_init = jnp.full((b_dec, hd, dkd, 1), 0.01, dtype=jnp.bfloat16)
              j_dec = jax.jit(_decode_step_fp8_block_scaled)
              t_step_ms, _, hlo_d = _time_and_profile_hlo(j_dec, s_q_init, s_sc_init, q_d, k_d, v_d, g_d, beta_d, warmup=2, iters=10)

            ms_per_tok = t_step_ms / float(t_ver)
            roof_ms_per_tok = t_hbm_roof_ms / float(t_ver)
            hlo_mb = hlo_d["gb_accessed"] * 1e3
            eff_bw_gbps = (hlo_mb / 1e3) / (t_step_ms / 1e3)
            attain_pct = 100.0 * (hlo_mb / V7X_HBM_BW_GBPS) / max(t_step_ms, 1e-6)
            print(
                f"[{tier_tag:11s} | {d_label:38s}] B={b_dec:2d}, T_ver={t_ver} | "
                f"step={t_step_ms:6.3f}ms | ms/tok={ms_per_tok:6.3f}ms (x24={24*ms_per_tok:5.2f}ms/tok) | "
                f"Roof_ms/tok={roof_ms_per_tok:5.3f}ms (StateRW={st_rw_mb:6.1f}MB, HLO={hlo_mb:6.1f}MB) | "
                f"BW={eff_bw_gbps:6.1f} GB/s (Attain={attain_pct:5.1f}%)",
                flush=True,
            )
          except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[{d_label:38s}] B={b_dec} failed: {type(exc).__name__}: {exc}", flush=True)

      # 64-Step Recurrent State Quantization Numerical Drift & Cosine Similarity Verification
      print(
          "\n--- Study 4 Part B: 64-Step Recurrent Decode State Quantization Drift (FP32 Reference vs BF16 vs Block-Scaled FP8 vs Unscaled FP8) ---",
          flush=True,
      )
      b_sim, t_sim, h_sim, dk_sim, dv_sim = 4, 64, 8, 256, 512
      q_sim = olmoe3._l2_normalize(jax.random.normal(keys[0], (b_sim, t_sim, h_sim, dk_sim), jnp.float32)) * (dk_sim**-0.5)
      k_sim = olmoe3._l2_normalize(jax.random.normal(keys[1], (b_sim, t_sim, h_sim, dk_sim), jnp.float32))
      v_sim = jax.random.normal(keys[2], (b_sim, t_sim, h_sim, dv_sim), jnp.float32) * 0.2
      g_sim = -jax.nn.softplus(jax.random.normal(keys[3], (b_sim, t_sim, h_sim, dk_sim), jnp.float32) - 2.5)
      beta_sim = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b_sim, t_sim, h_sim), jnp.float32))

      s0_f32 = jnp.zeros((b_sim, h_sim, dk_sim, dv_sim), jnp.float32)
      s_ref_f32, y_ref_f32 = _decode_step_fp32_or_bf16(s0_f32, q_sim, k_sim, v_sim, g_sim, beta_sim)
      s_bf16, y_bf16 = _decode_step_fp32_or_bf16(s0_f32.astype(jnp.bfloat16), q_sim, k_sim, v_sim, g_sim, beta_sim)

      # Step-by-step FP8 state quantization (re-quantizing after every token step vs every T_ver=5 chunk)
      def _sim_fp8_every_k(chunk_k: int, use_row_scale: bool):
        st_q = jnp.zeros((b_sim, h_sim, dk_sim, dv_sim), jnp.float8_e4m3fn)
        st_sc = jnp.ones((b_sim, h_sim, dk_sim, 1), jnp.bfloat16)
        ys = []
        fp8_max = jnp.float32(448.0)
        for t0 in range(0, t_sim, chunk_k):
          if use_row_scale:
            st_q, st_sc, y_c = _decode_step_fp8_block_scaled(
                st_q,
                st_sc,
                q_sim[:, t0 : t0 + chunk_k],
                k_sim[:, t0 : t0 + chunk_k],
                v_sim[:, t0 : t0 + chunk_k],
                g_sim[:, t0 : t0 + chunk_k],
                beta_sim[:, t0 : t0 + chunk_k],
            )
          else:
            st_f, y_c = _decode_step_fp32_or_bf16(
                st_q.astype(jnp.float32),
                q_sim[:, t0 : t0 + chunk_k],
                k_sim[:, t0 : t0 + chunk_k],
                v_sim[:, t0 : t0 + chunk_k],
                g_sim[:, t0 : t0 + chunk_k],
                beta_sim[:, t0 : t0 + chunk_k],
            )
            st_q = jnp.clip(st_f, -fp8_max, fp8_max).astype(jnp.float8_e4m3fn)
          ys.append(y_c)
        st_f_end = st_q.astype(jnp.float32) * (st_sc.astype(jnp.float32) if use_row_scale else 1.0)
        return st_f_end, jnp.concatenate(ys, axis=1)

      s_fp8_blk1, y_fp8_blk1 = _sim_fp8_every_k(1, True)
      s_fp8_blk4, y_fp8_blk4 = _sim_fp8_every_k(4, True)
      s_fp8_unsc, y_fp8_unsc = _sim_fp8_every_k(1, False)

      for q_name, s_test, y_test in (
          ("BF16_State_EveryStep (2B/elem)", s_bf16.astype(jnp.float32), y_bf16.astype(jnp.float32)),
          ("FP8_BlockScaled_EveryStep (1B/elem)", s_fp8_blk1, y_fp8_blk1.astype(jnp.float32)),
          ("FP8_BlockScaled_SpecChunk4 (1B/elem)", s_fp8_blk4, y_fp8_blk4.astype(jnp.float32)),
          ("FP8_Unscaled_EveryStep (1B/elem)", s_fp8_unsc, y_fp8_unsc.astype(jnp.float32)),
      ):
        y_ref_f = y_ref_f32.astype(jnp.float32)
        rel_l2_out = float(jnp.linalg.norm(y_test - y_ref_f) / jnp.maximum(jnp.linalg.norm(y_ref_f), 1e-8))
        rel_l2_st = float(jnp.linalg.norm(s_test - s_ref_f32) / jnp.maximum(jnp.linalg.norm(s_ref_f32), 1e-8))
        cos_out = float(
            jnp.sum(y_test * y_ref_f) / jnp.maximum(jnp.linalg.norm(y_test) * jnp.linalg.norm(y_ref_f), 1e-8)
        )
        snr_db = -20.0 * math.log10(max(rel_l2_out, 1e-8))
        print(
            f"[QuantDrift 64-Step | {q_name:38s}] Out_CosSim={cos_out:.6f} | Out_RelL2={rel_l2_out:.5f} "
            f"(SNR={snr_db:5.1f} dB) | State_RelL2={rel_l2_st:.5f}",
            flush=True,
        )
    finally:
      kda_fused._fwd_kernel = orig_fwd
      kda_fused._bwd_kernel = orig_bwd

  def test_10a_nextgen_linear_gdn_kda_v7x_perf_study(self):
    """Shard 10A: Next-Gen Core/Layer/Decode Co-Design Benchmarks on TPU v7x."""
    print(
        f"\n================================================================================"
        f"\nSHARD 10A: TPU v7x ({self.num_devs}x {self.dev_kind}) NEXT-GEN LINEAR/GDN/KDA CO-DESIGN PERF BENCHMARK"
        f"\n  Part 1: Next-Gen Core & Layer Co-Design (Dual-Head Batching, Block-16/32 KDA, Cross-Layer 2:1 Key+Decay Sharing)"
        f"\n  Part 2: Matrix-Parallel Speculative Tree Verification (Mini-Chunk WY Verify vs Sequential Scan Decode)"
        f"\n================================================================================",
        flush=True,
    )

    orig_fwd, orig_bwd = kda_fused._fwd_kernel, kda_fused._bwd_kernel
    kda_fused._fwd_kernel = _fwd_kernel_v7x_opt
    kda_fused._bwd_kernel = _bwd_kernel_v7x_opt

    try:
      b = 1
      seq_len = 8192
      keys = jax.random.split(jax.random.PRNGKey(2027), 12)
      resets = jnp.zeros((b, seq_len), dtype=bool)

      # -----------------------------------------------------------------------
      # PART 1A: Next-Gen Core Kernel Optimizations on TPU v7x (T=8192, H=8, dk=256, dv=512, C=128)
      #   1) Log-Domain Clamped Block-16 & Block-32 KDA (kda_fused._SUB_BLOCK in {8, 16, 32})
      #   2) 1:1 Independent Heads (8K/8V) vs 2:1 GVA Packed Heads (4K/8V) vs 4:1 GVA Packed Heads (2K/8V)
      # -----------------------------------------------------------------------
      print(
          "\n=== PART 1A: Next-Gen Core Kernel Co-Design on TPU v7x (Block-16/32 KDA & 2:1/4:1 GVA Head Packing, B=1, T=8192) ===",
          flush=True,
      )
      h, dk, dv, c = 8, 256, 512, 128
      q = olmoe3._l2_normalize(jax.random.normal(keys[0], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16) * (dk**-0.5)
      k = olmoe3._l2_normalize(jax.random.normal(keys[1], (b, seq_len, h, dk), jnp.bfloat16)).astype(jnp.bfloat16)
      v = (jax.random.normal(keys[2], (b, seq_len, h, dv), jnp.bfloat16) * 0.1).astype(jnp.bfloat16)
      g_full = -jax.nn.softplus(jax.random.normal(keys[3], (b, seq_len, h, dk), jnp.float32) - 2.0)
      beta = (2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b, seq_len, h), jnp.bfloat16))).astype(jnp.bfloat16)
      dy = jax.random.normal(keys[5], (b, seq_len, h, dv), jnp.bfloat16) * 0.1

      orig_sb = kda_fused._SUB_BLOCK
      ref_kda_out = None
      for ob_size, sb_label in ():
        try:
          fn_kda = lambda q_, k_, v_, g_, b_, _ob=ob_size: kda.chunk_kda(
              q_, k_, v_, g_, b_, resets=resets, chunk_size=c, state_dtype=jnp.float32,
              layout="token_major", outer_block=_ob, output_dtype=jnp.bfloat16, state_stride=0
          )
          j_pf = jax.jit(fn_kda)
          j_tr = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, g_, b_, _f=fn_kda: jnp.sum(_f(q_, k_, v_, g_, b_).astype(jnp.float32) * dy.astype(jnp.float32)),
                  argnums=(0, 1, 2, 3, 4),
              )
          )
          t_pf, out_pf, hlo_pf = _time_and_profile_hlo(j_pf, q, k, v, g_full, beta, warmup=1, iters=5)
          t_tr, _, hlo_tr = _time_and_profile_hlo(j_tr, q, k, v, g_full, beta, warmup=1, iters=5)
          if ref_kda_out is None:
            ref_kda_out = out_pf.astype(jnp.float32)
          diff = float(jnp.max(jnp.abs(ref_kda_out - out_pf.astype(jnp.float32))))
          roof_k = compute_core_roofline_analytical(b, seq_len, h, dk, dv, c, is_kda=True, state_f32=True, state_stride=0, outer_block=ob_size, v7x_stacked_opt=True)
          print(
              f"[{sb_label:48s}] Prefill={t_pf:6.3f}ms (max_diff={diff:.2e}) | "
              + format_perf_vs_roofline("Train_FwdBwd", t_pf, t_tr, roof_k, hlo_tr),
              flush=True,
          )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{sb_label:48s}] failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
          kda_fused._SUB_BLOCK = orig_sb

      # Benchmark 1:1 (8K/8V, 8x256x512) vs 2:1 GVA Packed (4K/8V, 4x256x1024 at iso-value-width=4096 & 4x256x512 at dv=256->512)
      for gva_label, hk_gva, dv_gva in ():
        try:
          q_g = q[:, :, :hk_gva, :]
          k_g = k[:, :, :hk_gva, :]
          v_g = v[:, :, : (hk_gva * dv_gva) // 512, :].reshape(b, seq_len, hk_gva, dv_gva)
          dy_g = dy[:, :, : (hk_gva * dv_gva) // 512, :].reshape(b, seq_len, hk_gva, dv_gva)
          g_g = g_full[:, :, :hk_gva, 0]
          b_g = beta[:, :, :hk_gva]
          fn_gva = lambda q_, k_, v_, g_, b_: run_gdn_pallas_fused(q_, k_, v_, g_, b_, resets, c, "float32")
          j_pf_g = jax.jit(fn_gva)
          j_tr_g = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, g_, b_, _f=fn_gva: jnp.sum(_f(q_, k_, v_, g_, b_).astype(jnp.float32) * dy_g.astype(jnp.float32)),
                  argnums=(0, 1, 2, 3, 4),
              )
          )
          t_pf_g, _, hlo_pf_g = _time_and_profile_hlo(j_pf_g, q_g, k_g, v_g, g_g, b_g, warmup=1, iters=5)
          t_tr_g, _, hlo_tr_g = _time_and_profile_hlo(j_tr_g, q_g, k_g, v_g, g_g, b_g, warmup=1, iters=5)
          roof_gva = compute_core_roofline_analytical(b, seq_len, hk_gva, dk, dv_gva, c, is_kda=False, state_f32=True, state_stride=1, outer_block=None, v7x_stacked_opt=True)
          pf_floor = max(roof_gva["fwd_hbm_pallas_ms"], roof_gva["fwd_comp_tile_ms"])
          print(
              f"[IsoValWidth4096 | {gva_label:32s}] Prefill={t_pf_g:6.3f}ms (Floor={pf_floor:5.3f}ms, Attain={100*pf_floor/t_pf_g:5.1f}%) | "
              + format_perf_vs_roofline("Train_FwdBwd", t_pf_g, t_tr_g, roof_gva, hlo_tr_g),
              flush=True,
          )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[IsoValWidth4096 | {gva_label:32s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # PART 1B: 2-Layer Pair Co-Design: Cross-Layer 2:1 Key+Decay+WY Sharing + 1-Stage Gate (B=1, T=8192, d_model=1792)
      # -----------------------------------------------------------------------
      print(
          "\n=== PART 1B: 2-Layer Pair Co-Design: Cross-Layer 2:1 Key+Decay Sharing + 1-Stage Gate (B=1, T=8192, d=1792, H=8, dk=256, dv=512) ===",
          flush=True,
      )
      d_model = 1792
      x_pair = jax.random.normal(keys[0], (b, seq_len, d_model), jnp.bfloat16)
      dy_pair = jax.random.normal(keys[1], (b, seq_len, d_model), jnp.bfloat16) * 0.1

      # Weights for Layer 0 and Layer 1
      w_in_2st_0 = jax.random.normal(keys[2], (d_model, 9224), jnp.bfloat16) * 0.02
      w_in_2st_1 = jax.random.normal(keys[3], (d_model, 9224), jnp.bfloat16) * 0.02
      w_f2_0 = jax.random.normal(keys[4], (512, 2048), jnp.bfloat16) * 0.02
      w_f2_1 = jax.random.normal(keys[5], (512, 2048), jnp.bfloat16) * 0.02
      w_g2_0 = jax.random.normal(keys[6], (512, 4096), jnp.bfloat16) * 0.02
      w_g2_1 = jax.random.normal(keys[7], (512, 4096), jnp.bfloat16) * 0.02

      # 1-Stage SDN G=4 weights: full layer has (q:2048, k:2048, v:4096, g_out:4096, f_sdn4:32, beta:8 -> 12,328)
      w_in_1st_0 = jax.random.normal(keys[2], (d_model, 12328), jnp.bfloat16) * 0.02
      w_in_1st_1 = jax.random.normal(keys[3], (d_model, 12328), jnp.bfloat16) * 0.02
      # Cross-layer 2:1 shared K/decay/beta Layer 1 only needs (q:2048, v:4096, g_out:4096 -> 10,240 cols!)
      w_in_cl_1 = jax.random.normal(keys[3], (d_model, 10240), jnp.bfloat16) * 0.02
      # 2:1 GVA (4K/8V) + Cross-Layer 2:1 shared K/decay/beta: Layer 0 has (q:1024, k:1024, v:4096, g_out:4096, fb:20 -> 10,260), Layer 1 has (q:1024, v:4096, g_out:4096 -> 9,216)
      w_in_gva_0 = jax.random.normal(keys[2], (d_model, 10260), jnp.bfloat16) * 0.02
      w_in_gva_cl_1 = jax.random.normal(keys[3], (d_model, 9216), jnp.bfloat16) * 0.02

      wq_c0 = jax.random.normal(keys[8], (2048, 4), jnp.bfloat16) * 0.02
      wk_c0 = jax.random.normal(keys[8], (2048, 4), jnp.bfloat16) * 0.02
      wv_c0 = jax.random.normal(keys[8], (4096, 4), jnp.bfloat16) * 0.02
      wq_c1 = jax.random.normal(keys[9], (2048, 4), jnp.bfloat16) * 0.02
      wk_c1 = jax.random.normal(keys[9], (2048, 4), jnp.bfloat16) * 0.02
      wv_c1 = jax.random.normal(keys[9], (4096, 4), jnp.bfloat16) * 0.02
      wq_gva0 = wq_c0[:1024]
      wk_gva0 = wk_c0[:1024]
      wq_gva1 = wq_c1[:1024]

      w_out_0 = jax.random.normal(keys[10], (4096, d_model), jnp.bfloat16) * 0.02
      w_out_1 = jax.random.normal(keys[11], (4096, d_model), jnp.bfloat16) * 0.02
      rms_s = jnp.ones((8, 512), jnp.bfloat16)

      def _conv4_bf16(u, wc):
        up = jnp.pad(u, ((0, 0), (3, 0), (0, 0)))
        return (
            up[:, 0:seq_len, :] * wc[:, 0]
            + up[:, 1 : seq_len + 1, :] * wc[:, 1]
            + up[:, 2 : seq_len + 2, :] * wc[:, 2]
            + up[:, 3 : seq_len + 3, :] * wc[:, 3]
        ).astype(jnp.bfloat16)

      def _epi_norm_gate(y_core, g_out_raw, w_o, rms_s_):
        yf = y_core.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(yf * yf, axis=-1, keepdims=True) + 1e-6)
        yg = (yf * rms).astype(jnp.bfloat16) * rms_s_ * jax.nn.sigmoid(g_out_raw.reshape(b, seq_len, 8, 512))
        return jnp.dot(yg.reshape(b, seq_len, 4096), w_o)

      pw_b1 = (w_in_2st_0, w_in_2st_1, w_f2_0, w_f2_1, w_g2_0, w_g2_1, wq_c0, wk_c0, wv_c0, wq_c1, wk_c1, wv_c1, w_out_0, w_out_1, rms_s)
      pw_c1 = (w_in_1st_0, w_in_1st_1, wq_c0, wk_c0, wv_c0, wq_c1, wk_c1, wv_c1, w_out_0, w_out_1, rms_s)
      pw_c2 = (w_in_1st_0, w_in_cl_1, wq_c0, wk_c0, wv_c0, wq_c1, wv_c1, w_out_0, w_out_1, rms_s)
      pw_c3 = (w_in_gva_0, w_in_gva_cl_1, wq_gva0, wk_gva0, wv_c0, wq_gva1, wv_c1, w_out_0, w_out_1, rms_s)

      def _pair_b1_2stage_kda(x_, pw, rst):
        w_in0, w_in1, wf0, wf1, wg0, wg1, wqc0, wkc0, wvc0, wqc1, wkc1, wvc1, wo0, wo1, rmss = pw
        p0 = jnp.dot(x_, w_in0)
        q0, k0, v0, f1_0, g1_0, b0 = jnp.split(p0, [2048, 4096, 8192, 8704, 9216], axis=-1)
        q0 = olmoe3._l2_normalize(_conv4_bf16(q0, wqc0).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16) * (256**-0.5)
        k0 = olmoe3._l2_normalize(_conv4_bf16(k0, wkc0).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16)
        v0 = jax.nn.silu(_conv4_bf16(v0, wvc0)).reshape(b, seq_len, 8, 512)
        g0 = -jax.nn.softplus(jnp.dot(f1_0, wf0).astype(jnp.float32).reshape(b, seq_len, 8, 256) - 2.0)
        bt0 = (2.0 * jax.nn.sigmoid(b0.astype(jnp.float32))).astype(jnp.bfloat16)
        y0 = kda.chunk_kda(q0, k0, v0, g0, bt0, resets=rst, chunk_size=128, state_dtype=jnp.float32, layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=0)
        h1 = x_ + _epi_norm_gate(y0, jnp.dot(g1_0, wg0), wo0, rmss)
        p1 = jnp.dot(h1, w_in1)
        q1, k1, v1, f1_1, g1_1, b1 = jnp.split(p1, [2048, 4096, 8192, 8704, 9216], axis=-1)
        q1 = olmoe3._l2_normalize(_conv4_bf16(q1, wqc1).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16) * (256**-0.5)
        k1 = olmoe3._l2_normalize(_conv4_bf16(k1, wkc1).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16)
        v1 = jax.nn.silu(_conv4_bf16(v1, wvc1)).reshape(b, seq_len, 8, 512)
        g1 = -jax.nn.softplus(jnp.dot(f1_1, wf1).astype(jnp.float32).reshape(b, seq_len, 8, 256) - 2.0)
        bt1 = (2.0 * jax.nn.sigmoid(b1.astype(jnp.float32))).astype(jnp.bfloat16)
        y1 = kda.chunk_kda(q1, k1, v1, g1, bt1, resets=rst, chunk_size=128, state_dtype=jnp.float32, layout="token_major", outer_block=32, output_dtype=jnp.bfloat16, state_stride=0)
        return h1 + _epi_norm_gate(y1, jnp.dot(g1_1, wg1), wo1, rmss)

      def _pair_c1_1stage_sdn_g4(x_, pw, rst):
        w_in0, w_in1, wqc0, wkc0, wvc0, wqc1, wkc1, wvc1, wo0, wo1, rmss = pw
        p0 = jnp.dot(x_, w_in0)
        q0, k0, v0, go0, f0, b0 = jnp.split(p0, [2048, 4096, 8192, 12288, 12320], axis=-1)
        q0 = olmoe3._l2_normalize(_conv4_bf16(q0, wqc0).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16) * (256**-0.5)
        k0 = olmoe3._l2_normalize(_conv4_bf16(k0, wkc0).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16)
        v0 = jax.nn.silu(_conv4_bf16(v0, wvc0)).reshape(b, seq_len, 8, 512)
        g0 = jnp.mean(-jax.nn.softplus(f0.astype(jnp.float32).reshape(b, seq_len, 8, 4) - 2.0), axis=-1)
        bt0 = (2.0 * jax.nn.sigmoid(b0.astype(jnp.float32))).astype(jnp.bfloat16)
        y0 = run_gdn_pallas_fused(q0, k0, v0, g0, bt0, rst, 128, "float32")
        h1 = x_ + _epi_norm_gate(y0, go0, wo0, rmss)
        p1 = jnp.dot(h1, w_in1)
        q1, k1, v1, go1, f1, b1 = jnp.split(p1, [2048, 4096, 8192, 12288, 12320], axis=-1)
        q1 = olmoe3._l2_normalize(_conv4_bf16(q1, wqc1).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16) * (256**-0.5)
        k1 = olmoe3._l2_normalize(_conv4_bf16(k1, wkc1).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16)
        v1 = jax.nn.silu(_conv4_bf16(v1, wvc1)).reshape(b, seq_len, 8, 512)
        g1 = jnp.mean(-jax.nn.softplus(f1.astype(jnp.float32).reshape(b, seq_len, 8, 4) - 2.0), axis=-1)
        bt1 = (2.0 * jax.nn.sigmoid(b1.astype(jnp.float32))).astype(jnp.bfloat16)
        y1 = run_gdn_pallas_fused(q1, k1, v1, g1, bt1, rst, 128, "float32")
        return h1 + _epi_norm_gate(y1, go1, wo1, rmss)

      def _pair_c2_cross_layer_2to1_sdn_g4(x_, pw, rst):
        w_in0, w_in1, wqc0, wkc0, wvc0, wqc1, wvc1, wo0, wo1, rmss = pw
        p0 = jnp.dot(x_, w_in0)
        q0, k0, v0, go0, f0, b0 = jnp.split(p0, [2048, 4096, 8192, 12288, 12320], axis=-1)
        q0 = olmoe3._l2_normalize(_conv4_bf16(q0, wqc0).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16) * (256**-0.5)
        k_shared = olmoe3._l2_normalize(_conv4_bf16(k0, wkc0).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16)
        v0 = jax.nn.silu(_conv4_bf16(v0, wvc0)).reshape(b, seq_len, 8, 512)
        g_shared = jnp.mean(-jax.nn.softplus(f0.astype(jnp.float32).reshape(b, seq_len, 8, 4) - 2.0), axis=-1)
        bt_shared = (2.0 * jax.nn.sigmoid(b0.astype(jnp.float32))).astype(jnp.bfloat16)
        y0 = run_gdn_pallas_fused(q0, k_shared, v0, g_shared, bt_shared, rst, 128, "float32")
        h1 = x_ + _epi_norm_gate(y0, go0, wo0, rmss)
        p1 = jnp.dot(h1, w_in1)
        q1, v1, go1 = jnp.split(p1, [2048, 6144], axis=-1)
        q1 = olmoe3._l2_normalize(_conv4_bf16(q1, wqc1).reshape(b, seq_len, 8, 256)).astype(jnp.bfloat16) * (256**-0.5)
        v1 = jax.nn.silu(_conv4_bf16(v1, wvc1)).reshape(b, seq_len, 8, 512)
        y1 = run_gdn_pallas_fused(q1, k_shared, v1, g_shared, bt_shared, rst, 128, "float32")
        return h1 + _epi_norm_gate(y1, go1, wo1, rmss)

      def _pair_c3_gva2to1_and_cross_layer_sdn_g4(x_, pw, rst):
        w_in0, w_in1, wqc0, wkc0, wvc0, wqc1, wvc1, wo0, wo1, rmss = pw
        p0 = jnp.dot(x_, w_in0)
        q0, k0, v0, go0, f0, b0 = jnp.split(p0, [1024, 2048, 6144, 10240, 10256], axis=-1)
        q0 = olmoe3._l2_normalize(_conv4_bf16(q0, wqc0).reshape(b, seq_len, 4, 256)).astype(jnp.bfloat16) * (256**-0.5)
        k_shared = olmoe3._l2_normalize(_conv4_bf16(k0, wkc0).reshape(b, seq_len, 4, 256)).astype(jnp.bfloat16)
        v0 = jax.nn.silu(_conv4_bf16(v0, wvc0)).reshape(b, seq_len, 4, 1024)
        g_shared = jnp.mean(-jax.nn.softplus(f0.astype(jnp.float32).reshape(b, seq_len, 4, 4) - 2.0), axis=-1)
        bt_shared = (2.0 * jax.nn.sigmoid(b0.astype(jnp.float32))).astype(jnp.bfloat16)
        y0 = run_gdn_pallas_fused(q0, k_shared, v0, g_shared, bt_shared, rst, 128, "float32")
        h1 = x_ + _epi_norm_gate(y0.reshape(b, seq_len, 8, 512), go0, wo0, rmss)
        p1 = jnp.dot(h1, w_in1)
        q1, v1, go1 = jnp.split(p1, [1024, 5120], axis=-1)
        q1 = olmoe3._l2_normalize(_conv4_bf16(q1, wqc1).reshape(b, seq_len, 4, 256)).astype(jnp.bfloat16) * (256**-0.5)
        v1 = jax.nn.silu(_conv4_bf16(v1, wvc1)).reshape(b, seq_len, 4, 1024)
        y1 = run_gdn_pallas_fused(q1, k_shared, v1, g_shared, bt_shared, rst, 128, "float32")
        return h1 + _epi_norm_gate(y1.reshape(b, seq_len, 8, 512), go1, wo1, rmss)

      base_pair_tr = None
      base_pair_pf = None
      for pair_name, pair_fn, pw_arg in (
          ("Pair_B1_Independent_2Stage_KDA_G256 (Baseline)", _pair_b1_2stage_kda, pw_b1),
          ("Pair_C1_Independent_1Stage_SDN_G4", _pair_c1_1stage_sdn_g4, pw_c1),
          ("Pair_C2_CrossLayer_2to1_SharedKeyDecay_SDN_G4", _pair_c2_cross_layer_2to1_sdn_g4, pw_c2),
          ("Pair_C3_GVA2to1_Plus_CrossLayer2to1_SDN_G4", _pair_c3_gva2to1_and_cross_layer_sdn_g4, pw_c3),
      ):
        try:
          j_pf = jax.jit(pair_fn)
          j_tr = jax.jit(jax.value_and_grad(lambda x_, pw_, dy_, rst_, _f=pair_fn: jnp.sum(_f(x_, pw_, rst_).astype(jnp.float32) * dy_.astype(jnp.float32)), argnums=(0, 1)))
          t_pf, _, hlo_pf = _time_and_profile_hlo(j_pf, x_pair, pw_arg, resets, warmup=1, iters=5)
          t_tr, _, hlo_tr = _time_and_profile_hlo(j_tr, x_pair, pw_arg, dy_pair, resets, warmup=1, iters=5)
          if base_pair_tr is None:
            base_pair_tr = t_tr
            base_pair_pf = t_pf
          print(
              f"[{pair_name:46s}] 2-Layer Prefill={t_pf:6.3f}ms ({t_pf/2:5.3f}ms/layer, {base_pair_pf/t_pf:4.2f}x) | "
              f"2-Layer Train={t_tr:6.3f}ms ({t_tr/2:5.3f}ms/layer, x24={12*t_tr:5.1f}ms, {base_pair_tr/t_tr:4.2f}x) | "
              f"HBM_Train={hlo_tr['gb_accessed']*1e3:6.1f}MB",
              flush=True,
          )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{pair_name:46s}] failed: {type(exc).__name__}: {exc}", flush=True)

      # -----------------------------------------------------------------------
      # PART 2: Matrix-Parallel Speculative Tree Verification (Mini-Chunk WY Verify) vs Sequential lax.scan Decode
      # -----------------------------------------------------------------------
      print(
          "\n=== PART 2: Matrix-Parallel Speculative Tree Verification (Mini-Chunk WY Verify vs Sequential Scan Decode, T_ver in {4, 8, 16}) ===",
          flush=True,
      )

      def _decode_sequential_scan(state_in, q_step, k_step, v_step, g_step, beta_step):
        """Sequential token-by-token lax.scan over T_ver tokens (M=1 per step)."""
        st0 = state_in.astype(jnp.float32)
        q_t = jnp.moveaxis(q_step, 1, 0)
        k_t = jnp.moveaxis(k_step, 1, 0)
        v_t = jnp.moveaxis(v_step, 1, 0)
        g_t = jnp.moveaxis(g_step, 1, 0)
        b_t = jnp.moveaxis(beta_step, 1, 0)

        def _tok_body(st, xs):
          qt, kt, vt, gt, bt = xs
          eg = jnp.exp(gt.astype(jnp.float32))[..., None]
          st_dec = st * eg
          s_dt = state_in.dtype
          ks = jnp.einsum("bhd,bhdv->bhv", kt.astype(s_dt), st_dec.astype(s_dt), preferred_element_type=jnp.float32)
          delta = (vt.astype(jnp.float32) - ks) * bt.astype(jnp.float32)[..., None]
          st_next = st_dec + jnp.einsum("bhd,bhv->bhdv", kt.astype(s_dt), delta.astype(s_dt), preferred_element_type=jnp.float32)
          out_t = jnp.einsum("bhd,bhdv->bhv", qt.astype(s_dt), st_next.astype(s_dt), preferred_element_type=jnp.float32)
          return st_next, out_t.astype(jnp.bfloat16)

        st_end, y_seq = jax.lax.scan(_tok_body, st0, (q_t, k_t, v_t, g_t, b_t))
        return st_end.astype(state_in.dtype), jnp.moveaxis(y_seq, 0, 1)

      def _decode_minichunk_wy_verify(state_in, q_step, k_step, v_step, g_step, beta_step, tree_mask=None):
        """Matrix-Parallel Mini-Chunk WY Verification over T_ver tokens (Chain or Speculative Tree!).
        Contracts [2*T_ver, dk] x [dk, dv] against S_{t-1} in ONE single MXU pass using exact nilpotent squaring!
        """
        b_d, t_v, h_d, dk_d = q_step.shape
        dv_d = v_step.shape[-1]
        f32 = jnp.float32
        s_dt = state_in.dtype

        # Cumulative log-decay along chain or tree
        gf = g_step.transpose(0, 2, 1, 3).astype(f32)  # [B, H, T_v, dk]
        if tree_mask is None:
          tri = jnp.tril(jnp.ones((t_v, t_v), dtype=f32))
          causal = jnp.tril(jnp.ones((t_v, t_v), dtype=bool))
          strict = jnp.tril(jnp.ones((t_v, t_v), dtype=bool), k=-1)
        else:
          tri = tree_mask.astype(f32)
          causal = tree_mask > 0
          strict = causal & (~jnp.eye(t_v, dtype=bool))

        g_cum = jnp.einsum("ij,bhjd->bhid", tri, gf, preferred_element_type=f32)
        eg = jnp.exp(g_cum)
        inv_eg = jnp.exp(-g_cum)

        qf = q_step.transpose(0, 2, 1, 3).astype(f32)
        kf = k_step.transpose(0, 2, 1, 3).astype(f32)
        vf = v_step.transpose(0, 2, 1, 3).astype(f32)
        bf = beta_step.transpose(0, 2, 1).astype(f32)[..., None]

        # Intra-verify [T_v, T_v] Gram matrices using scaled Q*exp(g) and K*exp(-g)
        q_pos = (qf * eg).astype(jnp.bfloat16)
        k_pos = (kf * eg).astype(jnp.bfloat16)
        k_neg = (kf * inv_eg).astype(jnp.bfloat16)

        scores = jnp.where(causal, jnp.einsum("bhid,bhjd->bhij", q_pos, k_neg, preferred_element_type=f32), 0.0)
        kk = jnp.where(strict, jnp.einsum("bhid,bhjd->bhij", k_pos, k_neg, preferred_element_type=f32), 0.0)
        m_mat = kk * bf  # Strictly lower-triangular nilpotent of degree <= T_v!

        # Exact nilpotent Neumann series via log2(T_v) matrix squarings:
        # (I + M)^-1 = (I - M)(I + M^2)(I + M^4)...
        eye = jnp.eye(t_v, dtype=f32)
        x_inv = eye - m_mat
        m_pow = jnp.einsum("bhij,bhjk->bhik", m_mat, m_mat, preferred_element_type=f32)
        x_inv = jnp.einsum("bhij,bhjk->bhik", x_inv, eye + m_pow, preferred_element_type=f32)
        if t_v > 4:
          m_pow = jnp.einsum("bhij,bhjk->bhik", m_pow, m_pow, preferred_element_type=f32)
          x_inv = jnp.einsum("bhij,bhjk->bhik", x_inv, eye + m_pow, preferred_element_type=f32)
        if t_v > 8:
          m_pow = jnp.einsum("bhij,bhjk->bhik", m_pow, m_pow, preferred_element_type=f32)
          x_inv = jnp.einsum("bhij,bhjk->bhik", x_inv, eye + m_pow, preferred_element_type=f32)

        vb = (vf * bf).astype(jnp.bfloat16)
        kb = (kf * eg * bf).astype(jnp.bfloat16)
        uw = jnp.einsum("bhij,bhjd->bhid", x_inv.astype(jnp.bfloat16), jnp.concatenate([vb, kb], axis=-1), preferred_element_type=f32)
        u_mat, w_mat = uw[..., :dv_d], uw[..., dv_d:]

        # ONE single batched contraction against recurrent state S_{t-1}: [B, H, 2*T_v, dk] x [B, H, dk, dv]
        wqg = jnp.concatenate([w_mat.astype(s_dt), q_pos.astype(s_dt)], axis=2)
        ws_qgs = jnp.einsum("bhid,bhdv->bhiv", wqg, state_in, preferred_element_type=f32)
        ws, qgs = ws_qgs[:, :, :t_v, :], ws_qgs[:, :, t_v:, :]
        delta = (u_mat - ws).astype(s_dt)
        y_out = (qgs + jnp.einsum("bhij,bhjv->bhiv", scores.astype(s_dt), delta, preferred_element_type=f32)).astype(jnp.bfloat16)

        # State update at end of verified chunk
        g_last = g_cum[:, :, -1:, :]
        k_carry = (kf * jnp.exp(g_last - g_cum)).astype(s_dt)
        st_new = state_in.astype(f32) * jnp.exp(g_last).transpose(0, 1, 3, 2) + jnp.einsum(
            "bhid,bhiv->bhdv", k_carry, delta, preferred_element_type=f32
        )
        return st_new.astype(state_in.dtype), y_out.transpose(0, 2, 1, 3)

      # Verify numerical equivalence between Sequential lax.scan and Matrix-Parallel Mini-Chunk WY Verify
      for t_ver in (4, 8, 16):
        b_eq = 4
        q_eq = olmoe3._l2_normalize(jax.random.normal(keys[0], (b_eq, t_ver, 8, 256), jnp.float32)) * (256**-0.5)
        k_eq = olmoe3._l2_normalize(jax.random.normal(keys[1], (b_eq, t_ver, 8, 256), jnp.float32))
        v_eq = jax.random.normal(keys[2], (b_eq, t_ver, 8, 512), jnp.float32) * 0.1
        g_eq = -jax.nn.softplus(jax.random.normal(keys[3], (b_eq, t_ver, 8, 256), jnp.float32) - 2.5)
        b_eq_t = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[4], (b_eq, t_ver, 8), jnp.float32))
        s_eq = jax.random.normal(keys[5], (b_eq, 8, 256, 512), jnp.float32) * 0.1
        st_scan, y_scan = _decode_sequential_scan(s_eq, q_eq, k_eq, v_eq, g_eq, b_eq_t)
        st_wy, y_wy = _decode_minichunk_wy_verify(s_eq, q_eq, k_eq, v_eq, g_eq, b_eq_t)
        out_diff = float(jnp.max(jnp.abs(y_scan.astype(jnp.float32) - y_wy.astype(jnp.float32))))
        st_diff = float(jnp.max(jnp.abs(st_scan - st_wy)))
        print(
            f"[MiniChunk WY Verify Equivalence | T_ver={t_ver:2d}] Out_MaxDiff={out_diff:.3e} | State_MaxDiff={st_diff:.3e}",
            flush=True,
        )

      # Benchmark Sequential Scan vs Matrix-Parallel Mini-Chunk WY Verify on TPU v7x for B=32 (B=1, 8 already completed)
      for b_dec in (32,):
        for t_ver in (4, 8, 16):
          for hd, dkd, dvd, st_dt, cfg_tag in (
              (8, 256, 512, jnp.bfloat16, "8K8V_256x512_BF16St"),
              (4, 256, 512, jnp.bfloat16, "GVA2to1_4K8V_256x512_BF16St"),
          ):
            try:
              s_in = jax.random.normal(keys[0], (b_dec, hd, dkd, dvd), st_dt) * 0.1
              q_d = olmoe3._l2_normalize(jax.random.normal(keys[1], (b_dec, t_ver, hd, dkd), jnp.bfloat16)).astype(jnp.bfloat16) * (dkd**-0.5)
              k_d = olmoe3._l2_normalize(jax.random.normal(keys[2], (b_dec, t_ver, hd, dkd), jnp.bfloat16)).astype(jnp.bfloat16)
              v_d = (jax.random.normal(keys[3], (b_dec, t_ver, hd, dvd), jnp.bfloat16) * 0.1).astype(jnp.bfloat16)
              g_d = -jax.nn.softplus(jax.random.normal(keys[4], (b_dec, t_ver, hd, dkd), jnp.bfloat16) - 2.0).astype(jnp.bfloat16)
              bt_d = (2.0 * jax.nn.sigmoid(jax.random.normal(keys[5], (b_dec, t_ver, hd), jnp.bfloat16))).astype(jnp.bfloat16)

              j_scan = jax.jit(_decode_sequential_scan)
              j_wy = jax.jit(_decode_minichunk_wy_verify)
              t_scan_ms, _ = _time_fn(j_scan, s_in, q_d, k_d, v_d, g_d, bt_d, warmup=2, iters=10)
              t_wy_ms, _ = _time_fn(j_wy, s_in, q_d, k_d, v_d, g_d, bt_d, warmup=2, iters=10)
              st_rw_mb = (2.0 * b_dec * hd * dkd * dvd * 2) / 1e6
              roof_step_ms = st_rw_mb / V7X_HBM_BW_GBPS
              print(
                  f"[{cfg_tag:28s} | B={b_dec:2d}, T_ver={t_ver:2d}] "
                  f"SeqScan={t_scan_ms:6.3f}ms ({t_scan_ms/t_ver:5.3f}ms/tok, StateRW={t_ver*st_rw_mb:5.1f}MB) -> "
                  f"MiniChunk_WY={t_wy_ms:6.3f}ms ({t_wy_ms/t_ver:5.3f}ms/tok, StateRW={st_rw_mb:5.1f}MB) | "
                  f"Speedup={t_scan_ms/t_wy_ms:4.2f}x (Floor={roof_step_ms/t_ver:5.3f}ms/tok)",
                  flush=True,
              )
            except Exception as exc:  # pylint: disable=broad-exception-caught
              print(f"[{cfg_tag:28s} | B={b_dec:2d}, T_ver={t_ver:2d}] failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
      kda_fused._fwd_kernel = orig_fwd
      kda_fused._bwd_kernel = orig_bwd

  def test_10b_nextgen_linear_gdn_kda_v7x_loss_ablation(self):
    """Shard 10B: 300-Step 2-Layer Pair JAX Convergence & Loss Estimation Ablation Across 11 Co-Design Arms on TPU v7x."""
    print(
        f"\n================================================================================"
        f"\nSHARD 10B: TPU v7x ({self.num_devs}x {self.dev_kind}) 300-STEP LOSS ESTIMATION ABLATION ACROSS 11 CO-DESIGN ARMS"
        f"\n================================================================================",
        flush=True,
    )

    try:
      # -----------------------------------------------------------------------
      # PART 3: 300-Step 2-Layer Pair JAX Convergence & Loss Estimation Ablation Across 11 Co-Design Arms on TPU v7x
      # -----------------------------------------------------------------------
      print(
          "\n=== PART 3: 300-Step 2-Layer Pair JAX Convergence & Loss Estimation Ablation Across 11 Co-Design Arms on TPU v7x ===",
          flush=True,
      )

      def _gen_multitimescale_recall_induction_batch(seed_val: int, batch_sz: int, seq_l: int = 96, vocab_sz: int = 64):
        rng = np.random.RandomState(seed_val)
        num_keys = 14
        tokens = np.zeros((batch_sz, seq_l + 1), dtype=np.int32)
        assoc_mask = np.zeros((batch_sz, seq_l), dtype=np.float32)
        ind_mask = np.zeros((batch_sz, seq_l), dtype=np.float32)

        for bi in range(batch_sz):
          kv_table = {}
          ngram_hist = {}
          pos = 0
          while pos < seq_l - 3:
            mode = rng.rand()
            if mode < 0.44 or len(kv_table) < 4:
              if len(kv_table) >= 4 and rng.rand() < 0.65:
                active_k = list(kv_table.keys())
                k_tok = active_k[rng.randint(0, len(active_k))]
              else:
                k_tok = rng.randint(1, num_keys + 1)
              v_tok = rng.randint(17, 49)
              tokens[bi, pos] = k_tok
              tokens[bi, pos + 1] = v_tok
              kv_table[k_tok] = v_tok
              pos += 2
            elif mode < 0.74 and len(kv_table) > 0:
              active_keys = list(kv_table.keys())
              k_tok = active_keys[rng.randint(0, len(active_keys))]
              v_tok = kv_table[k_tok]
              tokens[bi, pos] = 0
              tokens[bi, pos + 1] = k_tok
              tokens[bi, pos + 2] = v_tok
              assoc_mask[bi, pos + 1] = 1.0
              pos += 3
            else:
              if len(ngram_hist) >= 3 and rng.rand() < 0.65:
                active_ng = list(ngram_hist.keys())
                m_tok, a_tok = active_ng[rng.randint(0, len(active_ng))]
                b_tok = ngram_hist[(m_tok, a_tok)]
                tokens[bi, pos] = m_tok
                tokens[bi, pos + 1] = a_tok
                tokens[bi, pos + 2] = b_tok
                ind_mask[bi, pos + 1] = 1.0
              else:
                m_tok = rng.randint(49, 57)
                a_tok = rng.randint(57, vocab_sz)
                b_tok = rng.randint(17, 49)
                ngram_hist[(m_tok, a_tok)] = b_tok
                tokens[bi, pos] = m_tok
                tokens[bi, pos + 1] = a_tok
                tokens[bi, pos + 2] = b_tok
              pos += 3
        return (
            jnp.asarray(tokens[:, :-1], dtype=jnp.int32),
            jnp.asarray(tokens[:, 1:], dtype=jnp.int32),
            jnp.asarray(assoc_mask),
            jnp.asarray(ind_mask),
        )

      vocab_c = 64
      seq_c = 96
      b_train = 64
      b_val = 256
      n_chunks = 6
      steps_per_chunk = 50
      n_steps = n_chunks * steps_per_chunk  # 300 steps
      d_c = 128
      n_layers = 2

      val_inp, val_tgt, val_assoc_m, val_ind_m = _gen_multitimescale_recall_induction_batch(9999, b_val, seq_c, vocab_c)
      train_batches = [
          _gen_multitimescale_recall_induction_batch(4000 + st_i, b_train, seq_c, vocab_c)
          for st_i in range(steps_per_chunk)
      ]
      tr_inp_chunk = jnp.stack([tb[0] for tb in train_batches], axis=0)
      tr_tgt_chunk = jnp.stack([tb[1] for tb in train_batches], axis=0)
      tr_assoc_chunk = jnp.stack([tb[2] for tb in train_batches], axis=0)
      tr_ind_chunk = jnp.stack([tb[3] for tb in train_batches], axis=0)

      def _run_6layer_loss_ablation_arm(
          arm_name: str,
          hk: int = 4,
          hv: int = 4,
          dk: int = 32,
          dv: int = 64,
          g_groups: int = 4,
          one_stage_gate: bool = False,
          bf16_state: bool = False,
          cross_layer_share_k_decay: bool = False,
          ffn_mult: float = 2.0,
          log_clamp_subblock: int = 0,
      ):
        k_w = hk * dk
        v_w = hv * dv
        dv_eff = (hv // hk) * dv
        g_w = hk * g_groups
        r_proj = 64  # low-rank bottleneck for 2-stage gate
        ffn_w = int(round(d_c * ffn_mult))
        k_init = jax.random.split(jax.random.PRNGKey(77), 4 + n_layers)

        def _init_layer(kl):
          ks = jax.random.split(kl, 10)
          c_prior_q = jnp.array([[0.0], [0.15], [0.45], [0.85]], dtype=jnp.float32)
          c_prior_k = jnp.array([[0.0], [0.20], [0.85], [0.35]], dtype=jnp.float32)
          c_prior_v = jnp.array([[0.0], [0.0], [0.10], [0.95]], dtype=jnp.float32)
          if one_stage_gate:
            w_in = jax.random.normal(ks[0], (d_c, 2 * k_w + 2 * v_w + g_w + hk), jnp.float32) * (d_c**-0.5)
            w_f2 = jnp.zeros((1, 1), jnp.float32)
            w_g2 = jnp.zeros((1, 1), jnp.float32)
          else:
            w_in = jax.random.normal(ks[0], (d_c, 2 * k_w + v_w + 2 * r_proj + hk), jnp.float32) * (d_c**-0.5)
            w_f2 = jax.random.normal(ks[7], (r_proj, g_w), jnp.float32) * (r_proj**-0.5)
            w_g2 = jax.random.normal(ks[8], (r_proj, v_w), jnp.float32) * (r_proj**-0.5)
          return {
              "w_in": w_in,
              "w_f2": w_f2,
              "w_g2": w_g2,
              "w_cq": c_prior_q + jax.random.normal(ks[1], (4, k_w), jnp.float32) * 0.06,
              "w_ck": c_prior_k + jax.random.normal(ks[2], (4, k_w), jnp.float32) * 0.06,
              "w_cv": c_prior_v + jax.random.normal(ks[3], (4, v_w), jnp.float32) * 0.06,
              "w_out": jax.random.normal(ks[4], (v_w, d_c), jnp.float32) * (0.20 * (v_w**-0.5)),
              "w_f1": jax.random.normal(ks[5], (d_c, 2 * ffn_w), jnp.float32) * (d_c**-0.5),
              "w_f2_ffn": jax.random.normal(ks[6], (ffn_w, d_c), jnp.float32) * (0.20 * (ffn_w**-0.5)),
          }

        params = {
            "emb": jax.random.normal(k_init[0], (vocab_c, d_c), jnp.float32) * 1.0,
            "layers": [_init_layer(k_init[2 + li]) for li in range(n_layers)],
            "head": jax.random.normal(k_init[1], (d_c, vocab_c), jnp.float32) * (d_c**-0.5),
        }
        n_params_total = sum(x.size for x in jax.tree_util.tree_leaves(params) if x.size > 1)
        if cross_layer_share_k_decay:
          # Subtract unused K/f/beta projection + k_conv weights in the odd layer(s)
          saved_odd = (n_layers // 2) * (d_c * (k_w + (g_w if one_stage_gate else r_proj) + hk) + 4 * k_w)
          n_params_total -= saved_odd

        def _causal_conv4(x_seq, w_c):
          x_pad = jnp.pad(x_seq, ((0, 0), (3, 0), (0, 0)))
          return (
              x_pad[:, 0:seq_c, :] * w_c[0]
              + x_pad[:, 1 : seq_c + 1, :] * w_c[1]
              + x_pad[:, 2 : seq_c + 2, :] * w_c[2]
              + x_pad[:, 3 : seq_c + 3, :] * w_c[3]
          )

        def _apply_layer(h_in, lp, shared_kgb=None):
          rms_in = jax.lax.rsqrt(jnp.mean(h_in * h_in, axis=-1, keepdims=True) + 1e-5)
          x_n = h_in * rms_in
          proj = jnp.dot(x_n, lp["w_in"])
          if one_stage_gate:
            q_r, k_r, v_r, g_out_r, f_r, b_r = jnp.split(
                proj, [k_w, 2 * k_w, 2 * k_w + v_w, 2 * k_w + 2 * v_w, 2 * k_w + 2 * v_w + g_w], axis=-1
            )
          else:
            q_r, k_r, v_r, f1_r, g1_r, b_r = jnp.split(
                proj, [k_w, 2 * k_w, 2 * k_w + v_w, 2 * k_w + v_w + r_proj, 2 * k_w + v_w + 2 * r_proj], axis=-1
            )
            f_r = jnp.dot(jax.nn.silu(f1_r), lp["w_f2"])
            g_out_r = jnp.dot(jax.nn.silu(g1_r), lp["w_g2"])

          q_c = jax.nn.silu(_causal_conv4(q_r, lp["w_cq"])).reshape(-1, seq_c, hk, dk)
          q_n = olmoe3._l2_normalize(q_c)
          v_c = jax.nn.silu(_causal_conv4(v_r, lp["w_cv"])).reshape(-1, seq_c, hk, dv_eff)

          if shared_kgb is not None:
            k_n, decay, beta = shared_kgb
          else:
            k_c = jax.nn.silu(_causal_conv4(k_r, lp["w_ck"])).reshape(-1, seq_c, hk, dk)
            k_n = olmoe3._l2_normalize(k_c)
            log_g = -jax.nn.softplus(f_r.reshape(-1, seq_c, hk, g_groups) - 3.5)
            if log_clamp_subblock > 0:
              # Simulate sub-block log-domain clamping at boundary intervals log_clamp_subblock
              log_g = jnp.maximum(log_g, -30.0 / float(log_clamp_subblock))
            if g_groups == 1:
              decay = jnp.exp(log_g)
            elif g_groups < dk:
              decay = jnp.repeat(jnp.exp(log_g), dk // g_groups, axis=-1)
            else:
              decay = jnp.exp(log_g)
            beta = jax.nn.sigmoid(b_r + 0.5)

          bsz = h_in.shape[0]
          init_s = jnp.zeros((bsz, hk, dk, dv_eff), jnp.float32)

          def _step(st, xs):
            qi, ki, vi, di, bi = xs
            st_d = st * di[..., None]
            ws = jnp.einsum("bhk,bhkv->bhv", ki, st_d)
            delta = (vi - ws) * bi[..., None]
            st_new = st_d + jnp.einsum("bhk,bhv->bhkv", ki, delta)
            if bf16_state:
              # Round recurrent state carry to BF16 every step (straight-through estimator for gradients)
              st_bf = st_new.astype(jnp.bfloat16).astype(jnp.float32)
              st_new = st_new + jax.lax.stop_gradient(st_bf - st_new)
            oi = jnp.einsum("bhk,bhkv->bhv", qi, st_new)
            return st_new, oi

          tr_t = lambda z: jnp.moveaxis(z, 1, 0)
          _, out_t = jax.lax.scan(_step, init_s, (tr_t(q_n), tr_t(k_n), tr_t(v_c), tr_t(decay), tr_t(beta)))
          core_y = jnp.moveaxis(out_t, 0, 1).reshape(bsz, seq_c, v_w)
          rms_y = jax.lax.rsqrt(jnp.mean(core_y * core_y, axis=-1, keepdims=True) + 1e-5)
          gated_y = (core_y * rms_y) * jax.nn.sigmoid(g_out_r)
          h_mid = h_in + jnp.dot(gated_y, lp["w_out"])

          rms_f = jax.lax.rsqrt(jnp.mean(h_mid * h_mid, axis=-1, keepdims=True) + 1e-5)
          gu = jnp.dot(h_mid * rms_f, lp["w_f1"])
          gf, uf = jnp.split(gu, 2, axis=-1)
          h_out = h_mid + jnp.dot(jax.nn.silu(gf) * uf, lp["w_f2_ffn"])
          return h_out, (k_n, decay, beta)

        def _forward_logits(p, x_tok):
          h = p["emb"][x_tok]
          shared_kgb = None
          for li, lp in enumerate(p["layers"]):
            use_shared = shared_kgb if (cross_layer_share_k_decay and (li % 2 == 1)) else None
            h, kgb_out = _apply_layer(h, lp, use_shared)
            if li % 2 == 0:
              shared_kgb = kgb_out
          rms_f = jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + 1e-5)
          return jnp.dot(h * rms_f, p["head"])

        def _loss_fn(p, x_tok, y_tok, a_m, i_m):
          logits = _forward_logits(p, x_tok)
          log_probs = jax.nn.log_softmax(logits, axis=-1)
          nll = -jnp.take_along_axis(log_probs, y_tok[..., None], axis=-1)[..., 0]
          w_mask = 1.0 + 4.0 * (a_m + i_m)
          return jnp.sum(nll * w_mask) / jnp.sum(w_mask)

        @jax.jit
        def _train_chunk(p_in, m_in, v_in, step_base, x_ch, y_ch, am_ch, im_ch):
          def _body(carry, xs_step):
            p, m, v, g_step = carry
            xb, yb, am, im = xs_step
            loss_val, grads = jax.value_and_grad(_loss_fn)(p, xb, yb, am, im)
            lr = 5e-3 * 0.5 * (1.0 + jnp.cos(jnp.pi * (g_step - 1) / n_steps))
            m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, grads)
            v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * (g_ * g_), v, grads)
            m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - 0.9**g_step), m)
            v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - 0.999**g_step), v)
            p = jax.tree_util.tree_map(
                lambda p_, mh_, vh_: p_ * (1.0 - lr * 1e-3) - lr * mh_ / (jnp.sqrt(vh_) + 1e-8), p, m_hat, v_hat
            )
            return (p, m, v, g_step + 1), loss_val

          (p_o, m_o, v_o, _), losses_o = jax.lax.scan(_body, (p_in, m_in, v_in, step_base + 1), (x_ch, y_ch, am_ch, im_ch))
          return (p_o, m_o, v_o), losses_o

        @jax.jit
        def _eval_fn(p_final, v_inp, v_tgt, v_am, v_im):
          logits_v = _forward_logits(p_final, v_inp)
          log_p = jax.nn.log_softmax(logits_v, axis=-1)
          tok_nll = -jnp.take_along_axis(log_p, v_tgt[..., None], axis=-1)[..., 0]
          pred_tok = jnp.argmax(logits_v, axis=-1)
          tok_acc = (pred_tok == v_tgt).astype(jnp.float32)

          val_nll = jnp.mean(tok_nll)
          assoc_nll = jnp.sum(tok_nll * v_am) / jnp.maximum(jnp.sum(v_am), 1.0)
          assoc_acc = jnp.sum(tok_acc * v_am) / jnp.maximum(jnp.sum(v_am), 1.0)
          ind_nll = jnp.sum(tok_nll * v_im) / jnp.maximum(jnp.sum(v_im), 1.0)
          ind_acc = jnp.sum(tok_acc * v_im) / jnp.maximum(jnp.sum(v_im), 1.0)
          return val_nll, assoc_nll, assoc_acc, ind_nll, ind_acc

        m_cur = jax.tree_util.tree_map(jnp.zeros_like, params)
        v_cur = jax.tree_util.tree_map(jnp.zeros_like, params)
        p_cur = params
        all_losses = []
        for ch_i in range(n_chunks):
          (p_cur, m_cur, v_cur), ch_losses = _train_chunk(
              p_cur, m_cur, v_cur, jnp.int32(ch_i * steps_per_chunk), tr_inp_chunk, tr_tgt_chunk, tr_assoc_chunk, tr_ind_chunk
          )
          all_losses.append(ch_losses)
        losses = jnp.concatenate(all_losses, axis=0)
        v_nll, a_nll, a_acc, i_nll, i_acc = _eval_fn(p_cur, val_inp, val_tgt, val_assoc_m, val_ind_m)
        jax.block_until_ready(v_nll)
        print(
            f"[{arm_name:44s}] params={n_params_total/1e3:6.1f}K | "
            f"train: s0={float(losses[0]):.4f} -> s150={float(losses[149]):.4f} -> s300={float(losses[-1]):.4f} | "
            f"val_nll={float(v_nll):.4f} | assoc={float(a_nll):.4f} ({100*float(a_acc):5.1f}%) | "
            f"ind={float(i_nll):.4f} ({100*float(i_acc):5.1f}%)",
            flush=True,
        )

      ablation_arms = [
          # (arm_name, hk, hv, dk, dv, g_groups, one_stage_gate, bf16_state, cross_layer_share, ffn_mult, log_clamp_sb)
          ("A04_SDN_G4_1StageFusedGate_FP32St", 4, 4, 32, 64, 4, True, False, False, 2.0, 0),
          ("A05_SDN_G4_1StageGate_BF16RecurrentState", 4, 4, 32, 64, 4, True, True, False, 2.0, 0),
          ("A06_SDN_G4_2to1_GVA_SharedKey_IsoValWidth", 2, 4, 32, 64, 4, True, False, False, 2.0, 0),
          ("A07_SDN_G4_Sym_HalvedValWidth_Plus_FFN", 4, 4, 32, 32, 4, True, False, False, 2.75, 0),
          ("A08_SDN_G4_CrossLayer_2to1_SharedKeyDecay", 4, 4, 32, 64, 4, True, False, True, 2.0, 0),
          ("A09_SDN_G4_UltraWideKey_2xDk_IsoValWidth", 2, 4, 64, 64, 4, True, False, False, 2.0, 0),
          ("A10_KDA_G32_LogClamped_SubBlock32_FP32St", 4, 4, 32, 64, 32, False, False, False, 2.0, 32),
      ]
      arms_to_run = getattr(self, "_arms_subset", ablation_arms[:4])
      for arm_cfg in arms_to_run:
        try:
          _run_6layer_loss_ablation_arm(*arm_cfg)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print(f"[{arm_cfg[0]:44s}] failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
      pass

  def test_10c_nextgen_linear_gdn_kda_v7x_loss_ablation_part2(self):
    """Shard 10C: Remaining 3 Arms (A08..A10) of 300-Step 2-Layer Pair Loss Ablation on TPU v7x."""
    self._arms_subset = [
        ("A08_SDN_G4_CrossLayer_2to1_SharedKeyDecay", 4, 4, 32, 64, 4, True, False, True, 2.0, 0),
        ("A09_SDN_G4_UltraWideKey_2xDk_IsoValWidth", 2, 4, 64, 64, 4, True, False, False, 2.0, 0),
        ("A10_KDA_G32_LogClamped_SubBlock32_FP32St", 4, 4, 32, 64, 32, False, False, False, 2.0, 32),
    ]
    try:
      self.test_10b_nextgen_linear_gdn_kda_v7x_loss_ablation()
    finally:
      delattr(self, "_arms_subset")


if __name__ == "__main__":
  absltest.main()
