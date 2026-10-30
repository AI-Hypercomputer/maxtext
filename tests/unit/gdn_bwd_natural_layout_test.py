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
"""gdn_bwd_natural_layout: the GDN backward kernel with operand groups in their natural layouts against the relayout path.

The Pallas kernels run in interpret mode on CPU. Every natural-layout mode only changes HBM index maps of the backward
kernel (every VMEM block holds the same values), so the outputs must be bitwise equal; deliberately wrong index maps
must not be.
"""

import functools
import types

import jax
from jax.experimental import pallas as pl
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import ShardMode
from maxtext.kernels.gdn import model_runner
from maxtext.kernels.gdn.gdn_bwd import bwd_memory_ref
from maxtext.kernels.gdn.gdn_bwd import pallas_mosaic_tpu_bwd as bwd_kernel

_D, _C = 128, 64


def _bits(x):
  x = np.asarray(x)
  return x.view(np.uint16) if x.dtype == jnp.bfloat16 else x.view(np.uint32) if x.dtype == np.float32 else x


def _inputs(s, nk, nv, dtype, seed=0):
  dim = 2 * nk * _D + nv * _D
  nc = s // _C
  k = jax.random.split(jax.random.PRNGKey(seed), 9)
  return {
      "qkv_conv": jax.random.normal(k[0], (1, s, dim), jnp.float32).astype(dtype),
      "b": jax.random.normal(k[1], (1, s, nv), jnp.float32).astype(jnp.bfloat16),
      "a": (jax.random.normal(k[2], (1, s, nv), jnp.float32) - 2.0).astype(jnp.bfloat16),
      "a_log": jax.random.normal(k[3], (nv,), jnp.float32) * 0.3,
      "dt_bias": jax.random.normal(k[4], (nv,), jnp.float32) * 0.3,
      "do": jax.random.normal(k[5], (1, s, nv, _D), jnp.float32).astype(dtype),
      "chunk_states": jax.random.normal(k[6], (1, nc, nv, _D, _D), jnp.float32) * 0.05,
      "t_inv": jnp.tril(jax.random.normal(k[7], (1, nc, nv, _C, _C), jnp.float32) * 0.05, -1) + jnp.eye(_C),
      "d_recurrent_state": jax.random.normal(k[8], (1, nv, _D, _D), jnp.float32) * 0.05,
  }


def _packed_segments(s):
  x = np.zeros((1, s), np.int32)
  x[0, : s // 4], x[0, s // 4 : 3 * s // 4], x[0, 3 * s // 4 : s - 8] = 1, 2, 3
  return jnp.asarray(x)


def _run(x, nv, *, natural, head_tile, seg, dh0, recompute):
  """Runs the Pallas GDN backward kernel in interpret mode."""
  fn = functools.partial(
      bwd_kernel.pallas_gdn_bwd_kernel,
      num_v_heads=nv,
      kq_head_dim=_D,
      v_head_dim=_D,
      chunk_size=_C,
      use_qk_norm_in_gdn=True,
      head_tile=head_tile,
      segment_ids=seg,
      d_recurrent_state=x["d_recurrent_state"],
      return_dh0=dh0,
      interpret=True,
      recompute_t_inv=recompute,
      natural_layout=natural,
  )
  args = [x[n] for n in ("qkv_conv", "b", "a", "a_log", "dt_bias", "do", "chunk_states")]
  args.append(None if recompute else x["t_inv"])
  names = [str(e.params["name"]) for e in jax.make_jaxpr(fn)(*args).jaxpr.eqns if e.primitive.name == "pallas_call"]
  return jax.jit(fn)(*args), names


@pytest.mark.parametrize("mode", ["states", "all", "no_do", "inputs", "states_dy"])
@pytest.mark.parametrize(
    "nk,nv,head_tile,dtype",
    [(16, 64, None, jnp.bfloat16), (4, 16, 4, jnp.bfloat16), (4, 16, 8, jnp.float32)],
)
@pytest.mark.parametrize("seg_mode,dh0,recompute", [("none", False, False), ("packed", True, True)])
def test_natural_layout_is_bit_identical(nk, nv, head_tile, dtype, seg_mode, dh0, recompute, mode):
  s = 192
  x = _inputs(s, nk, nv, dtype)
  seg = _packed_segments(s) if seg_mode == "packed" else None
  ref, ref_names = _run(x, nv, natural="off", head_tile=head_tile, seg=seg, dh0=dh0, recompute=recompute)
  got, got_names = _run(x, nv, natural=mode, head_tile=head_tile, seg=seg, dh0=dh0, recompute=recompute)
  suffix = bwd_kernel.NATURAL_LAYOUT_MODES[mode][1]
  assert not any("_nat" in n for n in ref_names) and all(n.endswith(suffix) for n in got_names)
  assert len(ref) == len(got)
  for r, g in zip(ref, got):
    assert r.shape == g.shape and r.dtype == g.dtype
    np.testing.assert_array_equal(_bits(g), _bits(r))
  assert float(jnp.max(jnp.abs(ref[0].astype(jnp.float32)))) > 0.0


def test_unaligned_shape_keeps_the_relayout_path():
  x = _inputs(128, 2, 8, jnp.bfloat16)  # (q_size + k_size) = 512 is not a multiple of the 1024-wide v block
  ref, ref_names = _run(x, 8, natural="off", head_tile=None, seg=None, dh0=False, recompute=False)
  got, got_names = _run(x, 8, natural="all", head_tile=None, seg=None, dh0=False, recompute=False)
  assert got_names == ref_names == ["gdn_bwd_kernel"]
  for r, g in zip(ref, got):
    np.testing.assert_array_equal(_bits(g), _bits(r))


@pytest.mark.parametrize("wrong", ["k_window_one_block_low", "chunk_states_forward_order", "t_inv_group_0"])
def test_wrong_index_maps_are_detected(wrong, monkeypatch):
  """Negative controls: the comparison above must see a wrong natural-layout index map."""
  real = bwd_memory_ref.make_bwd_block_specs_natural

  def wrong_specs(**kw):
    ins, outs, ni, no = real(**kw)
    nc, k_off = kw["num_chunks"], kw["q_size"] // kw["tile_k_size"]
    ins = list(ins)
    if wrong == "k_window_one_block_low":
      ins[1] = pl.BlockSpec(ins[1].block_shape, lambda b, g, c: (b, nc - 1 - c, k_off - 1 + g))
    elif wrong == "chunk_states_forward_order":
      ins[6] = pl.BlockSpec(ins[6].block_shape, lambda b, g, c: (b, c, g, 0, 0))
    else:
      ins[7] = pl.BlockSpec(ins[7].block_shape, lambda b, g, c: (b, nc - 1 - c, 0, 0, 0))
    return ins, outs, ni, no

  s = 192
  x = _inputs(s, 4, 16, jnp.bfloat16, seed=1)
  ref, _ = _run(x, 16, natural="off", head_tile=8, seg=None, dh0=True, recompute=False)
  monkeypatch.setattr(bwd_kernel, "make_bwd_block_specs_natural", wrong_specs)
  jax.clear_caches()
  bad, _ = _run(x, 16, natural="all", head_tile=8, seg=None, dh0=True, recompute=False)
  jax.clear_caches()
  assert any(not np.array_equal(_bits(b), _bits(r)) for b, r in zip(bad, ref))


class _Value:

  def __init__(self, value):
    self.value = value

  def __getitem__(self, idx):
    return self.value[idx]


def _bwd_kernel_names_through_model_runner(config_fields):
  """Backward kernel names in the gradient of model_runner.run_gdn_kernel_layer, seq CP 4 on a TPU7x abstract mesh."""
  nk, nv, seq, cp = 16, 64, 256, 4
  device = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
  mesh = jax.sharding.AbstractMesh((cp,), ("context",), axis_types=(jax.sharding.AxisType.Auto,), abstract_device=device)
  cfg = types.SimpleNamespace(
      ici_context_parallelism=cp,
      dcn_context_parallelism=1,
      ici_context_usp_ulysses_parallelism=1,
      dcn_context_usp_ulysses_parallelism=1,
      gdn_cp_mode="seq",
      gdn_cp_matmul_precision="high",
      shard_mode=ShardMode.AUTO,
      logical_axis_rules=[],
      gdn_conv_kernel_dim=4,
      gdn_chunk_size=_C,
      use_qk_norm_in_gdn=True,
      context_parallel_load_balance=False,
      gdn_conv1d_input_in_qkv_dtype=True,
      gdn_conv1d_dense_token_stack=True,
      gdn_cp_m_scan=True,
      gdn_cp_dh_scan=True,
      gdn_cp_pass1_states_and_bf16_qkv=True,
      gdn_conv1d_natural_layout=False,
      gdn_fwd_kernel_head_group_size=0,
      **config_fields,
  )

  def loss(params, x, aux):
    layer = types.SimpleNamespace(
        config=cfg,
        mesh=mesh,
        num_k_heads=nk,
        num_v_heads=nv,
        head_k_dim=_D,
        head_v_dim=_D,
        key_dim=nk * _D,
        v_heads_per_k_head=nv // nk,
        conv1d=types.SimpleNamespace(kernel=_Value(params["conv_w"]), bias=None),
        A_log=_Value(params["a_log"]),
        dt_bias=_Value(params["dt_bias"]),
    )
    out, ncs, nrs = model_runner.run_gdn_kernel_layer(
        layer=layer,
        query=x["query"],
        key=x["key"],
        value=x["value_raw"].reshape(1, seq, nv, _D),
        value_raw=x["value_raw"],
        b=x["b"],
        a=x["a"],
        active_cache=None,
        decoder_segment_ids=aux["seg"],
        model_mode="train",
        state_dtype=jnp.float32,
    )
    return jnp.sum(out.astype(jnp.float32) * aux["do"]) + 0.5 * jnp.sum(ncs) + 0.25 * jnp.sum(nrs)

  a = jax.ShapeDtypeStruct
  params = {
      "conv_w": a((4, 1, (2 * nk + nv) * _D), jnp.float32),
      "a_log": a((nv,), jnp.float32),
      "dt_bias": a((nv,), jnp.float32),
  }
  x = {
      "query": a((1, seq, nk, _D), jnp.bfloat16),
      "key": a((1, seq, nk, _D), jnp.bfloat16),
      "value_raw": a((1, seq, nk, (nv // nk) * _D), jnp.bfloat16),
      "b": a((1, seq, nv), jnp.bfloat16),
      "a": a((1, seq, nv), jnp.bfloat16),
  }
  aux = {"seg": a((1, seq), jnp.int32), "do": a((1, seq, nv, _D), jnp.float32)}
  with jax.sharding.use_abstract_mesh(mesh):
    jaxpr = jax.make_jaxpr(jax.value_and_grad(loss, argnums=(0, 1)))(params, x, aux)
  names = []

  def visit(j):
    for e in j.eqns:
      if e.primitive.name == "pallas_call" and str(e.params.get("name", "")).startswith("gdn_bwd_kernel"):
        names.append(str(e.params["name"]))
      for sub in jax_core.jaxprs_in_params(e.params):
        visit(sub)

  visit(jaxpr.jaxpr)
  return names


def test_config_key_reaches_the_backward_kernel():
  assert _bwd_kernel_names_through_model_runner({"gdn_bwd_natural_layout": "off"}) == ["gdn_bwd_kernel"]
  assert _bwd_kernel_names_through_model_runner({"gdn_bwd_natural_layout": "all"}) == ["gdn_bwd_kernel_nat"]
  assert _bwd_kernel_names_through_model_runner({"gdn_bwd_natural_layout": "states"}) == ["gdn_bwd_kernel_nat_s"]
  # model_runner reads the key as an attribute: a config without it must fail loudly, not run the relayout path.
  with pytest.raises(AttributeError, match="gdn_bwd_natural_layout"):
    _bwd_kernel_names_through_model_runner({})


def test_config_default_is_off_and_modes_are_validated():
  from pydantic import TypeAdapter  # pylint: disable=import-outside-toplevel
  from maxtext.configs import types as config_types  # pylint: disable=import-outside-toplevel

  field = config_types.Qwen3Next.model_fields["gdn_bwd_natural_layout"]
  assert field.default == "off"
  for mode in bwd_kernel.NATURAL_LAYOUT_MODES:
    assert TypeAdapter(field.annotation).validate_python(mode) == mode
  with pytest.raises(Exception):
    TypeAdapter(field.annotation).validate_python("everything")
  with pytest.raises(ValueError, match="natural_layout must be"):
    bwd_kernel.natural_layout_mode("everything")
