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

"""Microbenchmark for Flint global causal SplashAttention on 4 TPU v7x chips.

Times the exact global-layer Splash fwd and fwd+bwd path used by the 64K/128K/
256K recipes (`flint_attention._splash_attention(is_global=True)`), with the
per-TensorCore shapes of those recipes, and reports efficiency vs the TPU v7x
BF16 peak (1153.5 TFLOPs/s per TC).

Env knobs (pass with `--test_env=KEY=VALUE`):
  MB_CASES: comma list of `<seq_len>:<cp_size>` (default
    `65536:1,131072:2,262144:4`, matching 64K (128,1,1), 128K (64,2,1),
    256K (32,4,1)).
  MB_OVERRIDES_JSON: JSON dict of FlintConfig overrides layered on top of the
    optimized recipe overrides (e.g. block sizes or new kernel knobs).
  MB_ITERS: timed iterations (default 5).
"""

from __future__ import annotations

import functools
import json
import os
import statistics
import time
from typing import Any, Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from absl.testing import absltest
from maxtext.experimental.flint40b import flint_attention
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import ops

_PEAK_TFLOPS_PER_TC = 1153.5
_GBS_FOR = {
    8192: 1024,
    16384: 512,
    32768: 256,
    65536: 128,
    131072: 64,
    262144: 32,
}
_FSDP_FOR = {1: 128, 2: 64, 4: 32, 8: 16}


def _time(fn: Callable[..., Any], *args: Any, iters: int) -> float:
  out = fn(*args)
  jax.block_until_ready(out)
  out = fn(*args)
  jax.block_until_ready(out)
  times = []
  for _ in range(iters):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    times.append(time.perf_counter() - t0)
  return float(statistics.median(times))


class FlintSplashMicrobenchTest(absltest.TestCase):

  def test_mxu_contraction_depth_probe(self):
    if not os.environ.get("MB_MXU_PROBE"):
      self.skipTest("MB_MXU_PROBE not set")

    reps = 64
    probe_dtype = jnp.dtype(os.environ.get("MB_MXU_PROBE_DTYPE", "bfloat16"))
    for mm, kk, nn in (
        (1024, 128, 512),
        (1024, 256, 512),
        (1024, 384, 512),
        (1024, 512, 512),
        (1024, 512, 128),
        (1024, 512, 256),
        (2048, 128, 1024),
        (2048, 1024, 128),
    ):

      def kern(a_ref, b_ref, o_ref, acc_ref):
        acc_ref[...] = jnp.zeros_like(acc_ref)

        def body(i, c):
          del i
          acc_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)
          return c

        jax.lax.fori_loop(0, reps, body, 0)
        o_ref[...] = acc_ref[...]

      grid_n = 64
      f = jax.jit(
          pl.pallas_call(
              kern,
              grid=(grid_n,),
              in_specs=[
                  pl.BlockSpec((mm, kk), lambda g: (0, 0)),
                  pl.BlockSpec((kk, nn), lambda g: (0, 0)),
              ],
              out_specs=pl.BlockSpec((mm, nn), lambda g: (g, 0)),
              out_shape=jax.ShapeDtypeStruct((grid_n * mm, nn), jnp.float32),
              scratch_shapes=[pltpu.VMEM((mm, nn), jnp.float32)],
          )
      )
      a = jnp.ones((mm, kk), probe_dtype)
      b = jnp.ones((kk, nn), probe_dtype)
      t = _time(f, a, b, iters=5)
      tf = 2.0 * mm * nn * kk * reps * grid_n / t / 1e12
      print(
          "MXU_PROBE "
          + json.dumps(
              {
                  "dtype": probe_dtype.name,
                  "m": mm,
                  "k": kk,
                  "n": nn,
                  "ms": round(t * 1e3, 4),
                  "tflops": round(tf, 1),
              }
          ),
          flush=True,
      )

  def test_global_splash_fwd_bwd_latency(self):
    cases = os.environ.get("MB_CASES", "65536:1,131072:2,262144:4")
    base_extra = json.loads(os.environ.get("MB_OVERRIDES_JSON", "{}") or "{}")
    variants = json.loads(os.environ.get("MB_VARIANTS", '[["base", {}]]') or '[["base", {}]]')
    iters = int(os.environ.get("MB_ITERS", "5"))
    results = []
    for (vname, vover), case in ((v, c) for v in variants for c in cases.split(",")):
      extra = {**base_extra, **vover}
      is_global = not bool(extra.pop("__local__", False))
      seq_s, cp_s = case.split(":")
      seq_len, cp = int(seq_s), int(cp_s)
      overrides = flint_config.optimized_64c_recipe_overrides(
          seq_len=seq_len,
          global_batch_size=_GBS_FOR[seq_len],
          fsdp_size=_FSDP_FOR[cp],
          cp_size=cp,
          remat_policy="global_selective",
      )
      overrides.update(extra)
      full_cfg = flint_config.flint_40b_config(**overrides)
      # Run the attention core on `cp` devices only (one sequence per TC).
      core_overrides: dict[str, Any] = dict(overrides)
      core_overrides.update(fsdp_size=1, cp_size=cp, ep_size=1, global_batch_size=1)
      cfg = flint_config.flint_40b_config(**core_overrides)
      del full_cfg
      mesh = flint_sharding.create_mesh(cfg.sharding, devices=jax.devices()[:cp])
      m = cfg.model
      hq, hkv, d = m.num_attention_heads, m.num_key_value_heads, m.head_dim
      key = jax.random.key(0)
      kq, kk, kv, kd = jax.random.split(key, 4)
      q = (jax.random.normal(kq, (1, hq, seq_len, d), jnp.float32) * (d**-0.5)).astype(jnp.bfloat16)
      k = jax.random.normal(kk, (1, hkv, seq_len, d), jnp.bfloat16)
      v = jax.random.normal(kv, (1, hkv, seq_len, d), jnp.bfloat16)
      do = jax.random.normal(kd, (1, seq_len, hq, d), jnp.bfloat16)
      sinks = jnp.zeros((hq,), jnp.float32)

      use_sdiag = cfg.kernels.splash_global_sdiag_gate if is_global else cfg.kernels.splash_local_sdiag_gate
      if use_sdiag and cp != 1:
        raise ValueError("sdiag_gate microbench requires cp=1 (aligned q/k).")

      def core(q_l, k_f, v_f):
        mlv = flint_attention.sdiag_max_logit(q_l, k_f, sinks, cp_size=1, reduce_q_over_cp=False) if use_sdiag else None
        return flint_attention._splash_attention(  # pylint: disable=protected-access
            q_l, k_f, v_f, None, None, sinks, mlv, cfg, is_global=is_global
        )

      smap = functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(P(None, None, "cp", None), P(), P()),
          out_specs=P(None, "cp", None, None),
          check_vma=False,
      )
      fwd = jax.jit(smap(core))

      def loss(q_l, k_f, v_f, do_l):
        return jnp.sum((smap(core)(q_l, k_f, v_f).astype(jnp.float32)) * do_l.astype(jnp.float32))

      fwd_bwd = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
      if os.environ.get("MB_PARITY"):
        outs = [np.asarray(fwd(q, k, v), np.float32)] + [np.asarray(g, np.float32) for g in fwd_bwd(q, k, v, do)]
        key_p = (seq_len, cp)
        if not hasattr(self, "_ref_outs"):
          self._ref_outs = {}
        if key_p not in self._ref_outs:
          self._ref_outs[key_p] = outs
        else:
          errs = []
          for name, a_o, r_o in zip(("o", "dq", "dk", "dv"), outs, self._ref_outs[key_p]):
            denom = float(np.max(np.abs(r_o))) or 1.0
            errs.append(
                (
                    name,
                    float(np.max(np.abs(a_o - r_o))) / denom,
                    float(np.linalg.norm(a_o - r_o) / (np.linalg.norm(r_o) or 1.0)),
                )
            )
          print(
              "MICROBENCH_PARITY " + json.dumps({"variant": vname, "seq_len": seq_len, "errs": errs}),
              flush=True,
          )
      t_fwd = _time(fwd, q, k, v, iters=iters)
      t_fb = _time(fwd_bwd, q, k, v, do, iters=iters)
      t_bwd = t_fb - t_fwd
      # Causal attention FLOPs per TC (per sequence split over cp TCs).
      if is_global:
        fwd_tf = 2.0 * seq_len * seq_len * hq * d / cp / 1e12
      else:
        w = m.sliding_window
        fwd_tf = 4.0 * (seq_len * w - w * w / 2) * hq * d / cp / 1e12
      res = {
          "variant": vname,
          "is_global": is_global,
          "seq_len": seq_len,
          "cp": cp,
          "fwd_ms": round(t_fwd * 1e3, 3),
          "bwd_ms": round(t_bwd * 1e3, 3),
          "fwd_bwd_ms": round(t_fb * 1e3, 3),
          "fwd_eff_pct": round(100 * fwd_tf / t_fwd / _PEAK_TFLOPS_PER_TC, 2),
          "bwd_eff_pct": round(100 * 2 * fwd_tf / max(t_bwd, 1e-9) / _PEAK_TFLOPS_PER_TC, 2),
          "fwd_bwd_eff_pct": round(100 * 3 * fwd_tf / t_fb / _PEAK_TFLOPS_PER_TC, 2),
          "x14_layers_ms": round(t_fb * 14e3, 1),
      }
      results.append(res)
      print("MICROBENCH_RESULT " + json.dumps(res), flush=True)
    print("MICROBENCH_ALL " + json.dumps(results), flush=True)

  def test_dense_fp8_latency(self):
    if not os.environ.get("MB_DENSE"):
      self.skipTest("MB_DENSE not set")

    shapes = (
        (65536, 2048, 6144, False),
        (65536, 4096, 2048, True),
        (65536, 2048, 2048, False),
        (65536, 1024, 2048, True),
    )
    iters = int(os.environ.get("MB_ITERS", "5"))
    for use_gmm in (False, True):
      overrides = flint_config.optimized_64c_recipe_overrides(
          seq_len=65536,
          global_batch_size=128,
          fsdp_size=128,
          cp_size=1,
          remat_policy="global_selective",
      )
      overrides.update(
          fsdp_size=1,
          cp_size=1,
          ep_size=1,
          global_batch_size=1,
          dense_fp8_gmm=use_gmm,
      )
      cfg = flint_config.flint_40b_config(**overrides)
      for m, k, n, transposed in shapes:
        kx, kw, kd = jax.random.split(jax.random.key(1), 3)
        x = jax.random.normal(kx, (m, k), jnp.bfloat16)
        w = jax.random.normal(kw, (n, k) if transposed else (k, n), jnp.float32) * (k**-0.5)
        do = jax.random.normal(kd, (m, n), jnp.bfloat16)

        def f(x_, w_):
          return ops.fp8_dense_dot_sharded(x_, w_, cfg, sharded_axis=0, rhs_is_transposed=transposed)

        def loss(x_, w_, do_):
          return jnp.sum(f(x_, w_).astype(jnp.float32) * do_.astype(jnp.float32))

        fwd = jax.jit(f)
        fb = jax.jit(jax.grad(loss, argnums=(0, 1)))
        out = np.asarray(fwd(x, w), np.float32)
        gx, gw = [np.asarray(g, np.float32) for g in fb(x, w, do)]
        key_p = (m, k, n)
        if not hasattr(self, "_dense_ref"):
          self._dense_ref = {}
        errs = None
        if key_p in self._dense_ref:
          errs = [
              float(np.linalg.norm(a - r) / (np.linalg.norm(r) or 1.0))
              for a, r in zip((out, gx, gw), self._dense_ref[key_p])
          ]
        else:
          self._dense_ref[key_p] = (out, gx, gw)
        t_f = _time(fwd, x, w, iters=iters)
        t_fb = _time(fb, x, w, do, iters=iters)
        print(
            "DENSE_RESULT "
            + json.dumps(
                {
                    "gmm": use_gmm,
                    "m": m,
                    "k": k,
                    "n": n,
                    "transposed": transposed,
                    "fwd_ms": round(t_f * 1e3, 3),
                    "fwd_bwd_ms": round(t_fb * 1e3, 3),
                    "rel_err_o_dx_dw": errs,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
  absltest.main()
