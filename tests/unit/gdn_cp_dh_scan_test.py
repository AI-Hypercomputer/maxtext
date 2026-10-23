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
"""The Pallas CP composition scans (`dh_scan`) against the XLA `cp_gdn` trees they replace on TPU, and their keys.

Runs the kernel in Pallas interpret mode. XLA:CPU computes the tree's f32 matmuls in plain f32, so the comparison
bounds the scans' 3-pass bf16 matmul error and the reduction order, not TPU numerics.

The keys gdn_cp_m_scan and gdn_cp_dh_scan (default False): the config validation; the gating, traced through
model_runner.run_gdn_kernel_layer on an abstract TPU7x context mesh (no scan pallas_call with both keys off, only the
forward m scan with gdn_cp_m_scan alone, both with both); and the CPU numerics through gdn_decoupled_conv1d, in a child
process with 2 CPU devices (sequence-sharded CP needs 2), so it runs whatever the parent's device count: keys off
bit-identical to the call without them, keys on close to keys off, forward and backward.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.gdn.gdn_bwd import cp_gdn
from maxtext.kernels.gdn.gdn_bwd import dh_scan

_NK, _NV, _D, _C, _NC = 2, 8, 128, 64, 4
_NUMERICS_DEVICES = 2
_PASS_TOKEN = "GDN_CP_SCAN_CPU_NUMERICS_CHECKS_PASSED"


def _inputs(seed=0):
  """One rank's K-slice conv output, gates and per-chunk T^{-1}, with weak decay."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 6)
  s = _NC * _C
  k_conv = jax.random.normal(keys[0], (1, s, _NK * _D), jnp.float32)
  b = jax.random.normal(keys[1], (1, s, _NV), jnp.float32).astype(jnp.bfloat16)
  # Weak decay keeps the product of the 4 chunk transitions far from 0.
  a = (jax.random.normal(keys[2], (1, s, _NV), jnp.float32) - 6.0).astype(jnp.bfloat16)
  a_log = jax.random.normal(keys[3], (_NV,), jnp.float32) * 0.3
  dt_bias = jax.random.normal(keys[4], (_NV,), jnp.float32) * 0.3
  t_inv = jnp.tril(jax.random.normal(keys[5], (1, _NC, _NV, _C, _C), jnp.float32) * 0.05, -1) + jnp.eye(_C)
  return k_conv, b, a, a_log, dt_bias, t_inv


def _tree_m(k_conv, b, a, a_log, dt_bias, t_inv, segment_ids, init_seg):
  """m_local of the XLA tree (f32 on CPU)."""
  m, _ = cp_gdn.compose_local_from_t_inv(
      k_conv,
      b,
      a,
      a_log,
      dt_bias,
      t_inv,
      num_k_heads=_NK,
      num_v_heads=_NV,
      head_k_dim=_D,
      head_v_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      s_ext_pass1=jnp.zeros((1, _NV, _D, _D), jnp.float32),
      segment_ids=segment_ids,
      init_seg=init_seg,
  )
  return m


def _scan_m(k_conv, b, a, a_log, dt_bias, t_inv, segment_ids, init_seg):
  """m_local of the Pallas scan, interpreted."""
  return dh_scan.gdn_cp_m_scan(
      k_conv,
      b,
      a,
      a_log,
      dt_bias,
      t_inv,
      num_v_heads=_NV,
      kq_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
      interpret=True,
  )


def _rel_err(got, ref):
  got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
  return np.abs(got - ref).max() / np.abs(ref).max()


# A document continuing from the previous rank and spanning the whole rank (the long single-document case), and a
# packed row: that document ends inside chunk 1, two more follow with boundaries inside chunks, then padding.
_SINGLE = np.ones((1, _NC * _C), np.int32)
_PACKED = np.zeros((1, _NC * _C), np.int32)
_PACKED[0, :70], _PACKED[0, 70:200], _PACKED[0, 200:240] = 1, 2, 3


@pytest.mark.parametrize("head_tile", [64, 4])  # one head group, two head groups
@pytest.mark.parametrize("layout", ["none", "single", "packed"])
def test_m_scan_matches_compose_local_tree(layout, head_tile, monkeypatch):
  monkeypatch.setattr(dh_scan, "_M_SCAN_HEAD_TILE", head_tile)
  segment_ids = {"none": None, "single": jnp.asarray(_SINGLE), "packed": jnp.asarray(_PACKED)}[layout]
  init_seg = None if segment_ids is None else jnp.ones((1,), jnp.float32)
  args = _inputs()
  ref = _tree_m(*args, segment_ids, init_seg)
  got = _scan_m(*args, segment_ids, init_seg)
  assert float(jnp.abs(ref).max()) > 1e-4, "degenerate reference: the transitions vanish"
  assert _rel_err(got, ref) < 5e-5


def test_m_scan_sees_chunk_order_and_segment_starts():
  """Negative controls: the comparison above can fail."""
  args = _inputs()
  segment_ids = jnp.asarray(_SINGLE)
  ref = _tree_m(*args, segment_ids, jnp.ones((1,), jnp.float32))
  k_conv, b, a, a_log, dt_bias, t_inv = args
  reversed_chunks = _scan_m(k_conv, b, a, a_log, dt_bias, t_inv[:, ::-1], segment_ids, jnp.ones((1,), jnp.float32))
  assert _rel_err(reversed_chunks, ref) > 1e-3
  # Without the incoming document, the rank's first token starts a new one, and nothing from the incoming state
  # reaches the rank's end state.
  fresh = _scan_m(*args, segment_ids, jnp.zeros((1,), jnp.float32))
  assert float(jnp.abs(fresh).max()) == 0.0


def _bwd_inputs(seed=1):
  """One rank's full conv output (q, k, v), gates, output cotangent and per-chunk T^{-1}."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 7)
  s = _NC * _C
  qkv_conv = jax.random.normal(keys[0], (1, s, 2 * _NK * _D + _NV * _D), jnp.float32).astype(jnp.bfloat16)
  b = jax.random.normal(keys[1], (1, s, _NV), jnp.float32).astype(jnp.bfloat16)
  a = (jax.random.normal(keys[2], (1, s, _NV), jnp.float32) - 2.0).astype(jnp.bfloat16)
  a_log = jax.random.normal(keys[3], (_NV,), jnp.float32) * 0.3
  dt_bias = jax.random.normal(keys[4], (_NV,), jnp.float32) * 0.3
  do = jax.random.normal(keys[5], (1, s, _NV, _D), jnp.float32).astype(jnp.bfloat16)
  t_inv = jnp.tril(jax.random.normal(keys[6], (1, _NC, _NV, _C, _C), jnp.float32) * 0.05, -1) + jnp.eye(_C)
  return qkv_conv, b, a, a_log, dt_bias, do, t_inv


@pytest.mark.parametrize("head_tile", [32, 4])  # one head group, two head groups
@pytest.mark.parametrize("layout", ["none", "single", "packed"])
def test_dh_scan_matches_compose_bwd_local_tree(layout, head_tile, monkeypatch):
  monkeypatch.setattr(dh_scan, "_DH_SCAN_HEAD_TILE", head_tile)
  segment_ids = {"none": None, "single": jnp.asarray(_SINGLE), "packed": jnp.asarray(_PACKED)}[layout]
  init_seg = None if segment_ids is None else jnp.ones((1,), jnp.float32)
  args = _bwd_inputs()
  _, ref = cp_gdn.compose_bwd_local_from_t_inv(
      *args,
      num_k_heads=_NK,
      num_v_heads=_NV,
      head_k_dim=_D,
      head_v_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
  )
  got = dh_scan.gdn_cp_dh_scan(
      *args,
      num_v_heads=_NV,
      kq_head_dim=_D,
      v_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
      interpret=True,
  )
  assert float(jnp.abs(ref).max()) > 1e-4, "degenerate reference"
  assert _rel_err(got, ref) < 5e-5
  # Negative control: a perturbed t_inv must show.
  qkv_conv, b, a, a_log, dt_bias, do, t_inv = args
  bad = dh_scan.gdn_cp_dh_scan(
      qkv_conv,
      b,
      a,
      a_log,
      dt_bias,
      do,
      t_inv * 1.01,
      num_v_heads=_NV,
      kq_head_dim=_D,
      v_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      segment_ids=segment_ids,
      init_seg=init_seg,
      interpret=True,
  )
  assert _rel_err(bad, ref) > 1e-3


def test_gdn_cp_scan_config_validation():
  """gdn_cp_m_scan and gdn_cp_dh_scan default to False and validate GDN kernel prerequisites."""
  from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
  from maxtext.configs.types import DecoderBlockType, MaxTextConfig  # pylint: disable=import-outside-toplevel

  cfg_ok = SimpleNamespace(
      decoder_block=DecoderBlockType.QWEN3_NEXT,
      gdn_conv1d_dense_token_stack=False,
      gdn_cp_m_scan=True,
      gdn_cp_dh_scan=True,
      ici_context_parallelism=2,
      dcn_context_parallelism=1,
      ici_context_usp_ulysses_parallelism=1,
      dcn_context_usp_ulysses_parallelism=1,
      use_gdn_kernel=True,
      gdn_conv1d_input_in_qkv_dtype=False,
      ici_tensor_parallelism=1,
      dcn_tensor_parallelism=1,
      gdn_chunk_size=64,
      gdn_key_head_dim=128,
      gdn_value_head_dim=128,
      use_qk_norm_in_gdn=True,
      packing=True,
      max_target_length=256,
      gdn_gated_norm_layout="auto",
      gdn_in_proj_kernel_order=False,
  )
  MaxTextConfig.validate_gdn_config(cfg_ok, is_rl_config=False)

  cfg_non_gdn = SimpleNamespace(**{**cfg_ok.__dict__, "decoder_block": DecoderBlockType.LLAMA2})
  with pytest.raises(ValueError, match="gdn_cp_m_scan=True / gdn_cp_dh_scan=True requires a decoder"):
    MaxTextConfig.validate_gdn_config(cfg_non_gdn, is_rl_config=False)

  cfg_no_kernel = SimpleNamespace(**{**cfg_ok.__dict__, "use_gdn_kernel": False})
  with pytest.raises(ValueError, match="requires use_gdn_kernel=True"):
    MaxTextConfig.validate_gdn_config(cfg_no_kernel, is_rl_config=False)

  cfg_bad_chunk = SimpleNamespace(**{**cfg_ok.__dict__, "gdn_chunk_size": 128})
  with pytest.raises(ValueError, match="requires gdn_chunk_size=64"):
    MaxTextConfig.validate_gdn_config(cfg_bad_chunk, is_rl_config=False)


def test_gdn_decoupled_conv1d_cp_scan_gating_and_tpu_lowering(monkeypatch):
  """Keys off: dh_scan is never called. On an abstract TPU7x mesh, m only: the m scan; both: both pallas_calls."""
  from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
  from jax._src import core as jax_core  # pylint: disable=import-outside-toplevel
  from maxtext.configs.types import ShardMode  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.gdn import model_runner  # pylint: disable=import-outside-toplevel

  calls = {"m_scan": 0, "dh_scan": 0}
  orig_m = dh_scan.gdn_cp_m_scan
  orig_dh = dh_scan.gdn_cp_dh_scan

  def spy_m(*a, **kw):
    calls["m_scan"] += 1
    return orig_m(*a, **kw)

  def spy_dh(*a, **kw):
    calls["dh_scan"] += 1
    return orig_dh(*a, **kw)

  monkeypatch.setattr(dh_scan, "gdn_cp_m_scan", spy_m)
  monkeypatch.setattr(dh_scan, "gdn_cp_dh_scan", spy_dh)

  nk, nv, d, c, ks = 2, 4, 128, 64, 4
  dim = 2 * nk * d + nv * d
  seq = 4 * c  # 1 chunk per rank across 4 CP ranks

  tpu_dev = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
  abs_mesh = jax.sharding.AbstractMesh(
      (4,), ("context",), axis_types=(jax.sharding.AxisType.Auto,), abstract_device=tpu_dev
  )

  class _Val:

    def __init__(self, v):
      self.value = v

    def __getitem__(self, idx):
      return self.value[idx]

  def pallas_names(closed_jaxpr):
    names = []

    def visit(jaxpr):
      for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
          names.append(eqn.params.get("name", ""))
        for sub in jax_core.jaxprs_in_params(eqn.params):
          visit(sub)

    visit(closed_jaxpr.jaxpr)
    return names

  def trace_layer(cp_m_scan: bool, cp_dh_scan: bool):
    cfg = SimpleNamespace(
        ici_context_parallelism=4,
        dcn_context_parallelism=1,
        ici_context_usp_ulysses_parallelism=1,
        dcn_context_usp_ulysses_parallelism=1,
        gdn_cp_mode="seq",
        gdn_cp_matmul_precision="high",
        shard_mode=ShardMode.AUTO,
        logical_axis_rules=[],
        gdn_conv_kernel_dim=ks,
        gdn_chunk_size=c,
        use_qk_norm_in_gdn=True,
        context_parallel_load_balance=False,
        gdn_conv1d_input_in_qkv_dtype=False,
        gdn_conv1d_dense_token_stack=False,
        gdn_cp_m_scan=cp_m_scan,
        gdn_cp_dh_scan=cp_dh_scan,
        gdn_bwd_natural_layout="off",
        gdn_conv1d_natural_layout=False,
        gdn_fwd_kernel_head_group_size=0,
    )

    def step(q, k, v_raw, b, a, cw, along, dtb):
      def loss_fn(q_x, k_x, v_x, b_x, a_x):
        layer = SimpleNamespace(
            config=cfg,
            mesh=abs_mesh,
            num_k_heads=nk,
            num_v_heads=nv,
            head_k_dim=d,
            head_v_dim=d,
            key_dim=nk * d,
            v_heads_per_k_head=nv // nk,
            conv1d=SimpleNamespace(kernel=_Val(cw), bias=None),
            A_log=_Val(along),
            dt_bias=_Val(dtb),
        )
        out, _, _ = model_runner.run_gdn_kernel_layer(
            layer=layer,
            query=q_x,
            key=k_x,
            value=v_x.reshape(1, seq, nv, d),
            value_raw=v_x,
            b=b_x,
            a=a_x,
            active_cache=None,
            decoder_segment_ids=None,
            model_mode="train",
            state_dtype=jnp.float32,
        )
        return jnp.sum(out.astype(jnp.float32) ** 2)

      loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v_raw, b, a)
      return loss + jnp.sum(grads[0].astype(jnp.float32))

    sds = jax.ShapeDtypeStruct
    args = (
        sds((1, seq, nk, d), jnp.bfloat16),
        sds((1, seq, nk, d), jnp.bfloat16),
        sds((1, seq, nv * d), jnp.bfloat16),
        sds((1, seq, nv), jnp.bfloat16),
        sds((1, seq, nv), jnp.bfloat16),
        sds((ks, 1, dim), jnp.float32),
        sds((nv,), jnp.float32),
        sds((nv,), jnp.float32),
    )
    with jax.sharding.use_abstract_mesh(abs_mesh):
      return jax.make_jaxpr(step)(*args)

  jaxpr_off = trace_layer(False, False)
  assert calls == {"m_scan": 0, "dh_scan": 0}
  names_off = pallas_names(jaxpr_off)
  assert "gdn_cp_m_scan" not in names_off
  assert "gdn_cp_dh_scan" not in names_off

  jaxpr_m_only = trace_layer(True, False)
  assert calls == {"m_scan": 1, "dh_scan": 0}
  names_m_only = pallas_names(jaxpr_m_only)
  assert "gdn_cp_m_scan" in names_m_only
  assert "gdn_cp_dh_scan" not in names_m_only

  jaxpr_both = trace_layer(True, True)
  assert calls == {"m_scan": 2, "dh_scan": 1}
  names_both = pallas_names(jaxpr_both)
  assert "gdn_cp_m_scan" in names_both
  assert "gdn_cp_dh_scan" in names_both


@pytest.mark.cpu_only
def test_gdn_decoupled_conv1d_cp_scan_cpu_numerics():
  """Runs `_cpu_numerics_checks` in a child process with 2 CPU devices, so it never skips for want of devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUMERICS_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _cpu_numerics_checks():
  """Key-OFF is bit-identical to default; Key-ON in CPU interpret mode matches forward and backward closely."""
  import functools  # pylint: disable=import-outside-toplevel
  from jax.sharding import PartitionSpec as P  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.gdn.gdn_bwd import api as gdn_api  # pylint: disable=import-outside-toplevel

  assert jax.device_count() >= _NUMERICS_DEVICES, jax.devices()

  mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("cp",))
  nk, nv, d, c, ks = 2, 4, 128, 64, 4
  dim = 2 * nk * d + nv * d
  s_total = 4 * c  # 2 chunks per rank across 2 CP ranks

  keys = jax.random.split(jax.random.PRNGKey(7), 7)
  qkv = jax.random.normal(keys[0], (1, s_total, dim), jnp.float32).astype(jnp.bfloat16)
  b = jax.random.normal(keys[1], (1, s_total, nv), jnp.float32).astype(jnp.bfloat16)
  a = (jax.random.normal(keys[2], (1, s_total, nv), jnp.float32) - 4.0).astype(jnp.bfloat16)
  cw = jax.random.normal(keys[3], (ks, 1, dim), jnp.float32).astype(jnp.bfloat16) * 0.1
  a_log = jax.random.normal(keys[4], (nv,), jnp.float32) * 0.2
  dt_bias = jax.random.normal(keys[5], (nv,), jnp.float32) * 0.2
  cot_out = jax.random.normal(keys[6], (1, s_total, nv, d), jnp.float32).astype(jnp.bfloat16)

  def run_fwd_bwd(mode: str):
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            P(None, "cp", None),
            P(None, "cp", None),
            P(None, "cp", None),
            P(),
            P(),
            P(),
            P(None, "cp", None, None),
        ),
        out_specs=(
            P(None, "cp", None, None),
            P(None, "cp", None),
            P(None, "cp", None),
            P(None, "cp", None),
            P(),
            P(),
            P(),
        ),
        check_vma=False,
    )
    def mapped(qkv_l, b_l, a_l, cw_l, alog_l, dtb_l, cot_l):
      def fwd(qkv_x, b_x, a_x, cw_x, alog_x, dtb_x):
        extra = () if mode == "default" else ((False, False) if mode == "off" else (True, True))
        out, _ = gdn_api.gdn_decoupled_conv1d(
            qkv_x,
            b_x,
            a_x,
            cw_x,
            None,
            alog_x,
            dtb_x,
            None,
            None,
            nk,
            nv,
            d,
            d,
            ks,
            c,
            True,
            jnp.float32,
            "cp",
            None,
            jax.lax.Precision.HIGH,
            False,
            False,
            *extra,
        )
        return out

      out, vjp_fn = jax.vjp(fwd, qkv_l, b_l, a_l, cw_l, alog_l, dtb_l)
      dqkv, db, da, dcw, dalog, ddtb = vjp_fn(cot_l)
      return out, dqkv, db, da, dcw, dalog, ddtb

    return jax.jit(mapped)(qkv, b, a, cw, a_log, dt_bias, cot_out)

  res_def = run_fwd_bwd("default")
  res_off = run_fwd_bwd("off")
  res_on = run_fwd_bwd("on")

  # Key-OFF must be bit-identical (0.0 diff) to default (omitted args).
  for t_def, t_off in zip(res_def, res_off):
    assert float(jnp.max(jnp.abs(t_def.astype(jnp.float32) - t_off.astype(jnp.float32)))) == 0.0

  # Key-ON (Pallas scans in CPU interpret mode) must match Key-OFF closely
  # (< 5e-4 for f32 gradients, < 5e-3 = < 1 bf16 ULP for bf16 activations/gradients).
  for t_off, t_on in zip(res_off, res_on):
    tol = 5e-4 if t_off.dtype == jnp.float32 else 5e-3
    assert _rel_err(t_on, t_off) < tol


if __name__ == "__main__":
  _cpu_numerics_checks()
  print(_PASS_TOKEN, flush=True)
