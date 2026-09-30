# Copyright 2023–2026 Google LLC
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

# pylint: disable=redefined-outer-name,too-many-positional-arguments,protected-access

"""Layer-subgroup fwd/bwd parity: MaxText DeepSeek-V4 vs the vendored official reference (tiny, fp32, CPU).

Subgroups (see tests/utils/deepseek4_layerwise.py): S1 = hash prefix layers 0-2,
S2 = scanned blocks (layers 3-6), S3 = last scanned block (layers 7-8, via layer_map)
+ hc_head + output head + masked cross entropy. Weights move only through the
production converter tables. Run explicitly:

  PYTHONPATH=src:. JAX_PLATFORMS=cpu python -m pytest -v -s tests/unit/deepseek_v4_layerwise_fwd_bwd_test.py
"""

import re
import time

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from flax import nnx
import maxtext.layers.attention_compressed
import maxtext.layers.moe

try:
  import torch

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

from tests.utils import deepseek4_layerwise as L
from tests.utils import deepseek4_reference as R

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="torch not available"),
]

SEQ = 264  # R.TINY_SEQ_LEN: 2 HCA entries + remainder, 66 CSA entries (> index_topk).
PACKED_DOC = 264
WEIGHT_SEED, DATA_SEED = 0, 100
# R.tiny_args() has index_n_heads=2: with 2 ReLU heads ~17% of indexer rows select among
# blocks whose score is exactly 0.0 and torch.topk / lax.top_k break those ties differently
# (see test_indexer_zero_score_ties_two_heads). 16 heads remove exact ties.
INDEX_HEADS = 16
TOPK_TIE_ATOL = 1e-6

# Thresholds: 10x the max MaxText-fp32 vs reference-fp32 error over 3 seeds, never below the reference
# fp32 vs fp64 error (weight seeds 0-2, data seeds 100-102).
# dW / dh are the worst over K=4 cotangents and every subgroup key. Measured maxima over seeds:
#   quantity           | (i) ref32 vs ref64: rel_l2, 1-cos | (ii) MaxText vs ref32: rel_l2, 1-cos
#   S1/out             | 1.81e-07, 3.77e-15                | 1.09e-07, 6.00e-15
#   S1/dh              | 1.86e-07, 4.88e-15                | 1.39e-07, 9.66e-15
#   S1/dW              | 2.45e-04, 2.97e-08                | 6.44e-04, 2.06e-07
#   S2/out             | 2.32e-07, 4.77e-15                | 1.18e-07, 7.22e-15
#   S2/dh              | 2.36e-07, 5.88e-15                | 1.53e-07, 1.14e-14
#   S2/dW              | 9.43e-03, 4.43e-05                | 9.55e-03, 1.11e-05
#   S3/logits          | 1.67e-07, 1.37e-14                | 1.51e-07, 1.18e-14
#   S3/xent_sum        | 6.69e-08, 0                       | 8.58e-08, 0
#   S3/dh              | 3.10e-07, 4.77e-14                | 1.41e-07, 9.77e-15
#   S3/dW              | 2.32e-03, 1.31e-06                | 2.18e-03, 1.70e-06
#   schema B logits    | 1.89e-07, 1.82e-14                | 1.90e-07, 1.79e-14 (chain; full model identical)
#   schema B xent_sum  | 6.69e-08, 0                       | 8.58e-08, 0
#   packed doc2 out    | 3.87e-07, 7.11e-15                | 3.36e-03, 5.65e-06 (real mismatch, see xfail);
#                        doc_len=256 (128-aligned): 1.42e-07 -> uses the "out" threshold.
REL_L2_MAX = {
    "out": 1.2e-6,
    "dh": 1.5e-6,
    "dW": 9.6e-2,
    "logits": 1.5e-6,
    "xent_sum": 8.6e-7,
    "schema_b_logits": 1.9e-6,
    "schema_b_xent_sum": 8.6e-7,
    "packed_out": 1.2e-6,
}
ONE_MINUS_COS_MAX = {
    "out": 7.2e-14,
    "dh": 1.1e-13,
    "dW": 4.4e-5,
    "logits": 1.2e-13,
    "xent_sum": 0.0,
    "schema_b_logits": 1.8e-13,
    "schema_b_xent_sum": 0.0,
    "packed_out": 7.2e-14,
}


def _args(**overrides):
  return R.tiny_args(index_n_heads=INDEX_HEADS, **overrides)


def _layer_idx(key):
  m = re.match(r"^layers\.(\d+)\.", key)
  return int(m.group(1)) if m else None


def _expected_no_grad(key):
  """Native params that receive no gradient from the output cotangent in either implementation."""
  return ".attn.indexer." in key or key.endswith(("ffn.gate.bias", "ffn.gate.tid2eid"))


def _collection_arrays(model, prefix):
  return {k: np.asarray(v.get_value()) for k, v in L.flat_state(nnx.state(model)).items() if k.startswith(prefix)}


def _all_maxtext_arrays(model):
  out = {}
  for prefix in ("params-", "MoEBiasVar-", "Tid2EidVar-"):
    out.update(_collection_arrays(model, prefix))
  return out


def _check(metrics, quantity):
  m = metrics[quantity]
  family = quantity.split("/")[-1]
  assert m["rel_l2"] <= REL_L2_MAX[family], f"{quantity}: rel_l2 {m['rel_l2']:.3e} > {REL_L2_MAX[family]:.1e} ({m})"
  assert 1 - m["cos"] <= ONE_MINUS_COS_MAX[family], f"{quantity}: 1-cos {1 - m['cos']:.3e} ({m})"


@pytest.fixture(scope="module")
def ctx():
  """Module-scoped fixture building models and baseline outputs."""
  t0 = time.time()
  jax.config.update("jax_default_matmul_precision", "highest")
  suite = L.Suite(_args(), SEQ, WEIGHT_SEED, DATA_SEED)
  full = L.build_loaded_model(suite.args, SEQ, suite.args.n_layers, suite.ref_sd)
  mt = suite.maxtext_results()
  ref = suite.reference_results(torch.float32)
  metrics = L.compare_results(mt, ref, suite.subgroups, suite.ref_sd)
  print(f"\n[ctx] suite built + fwd/bwd both sides in {time.time() - t0:.1f}s")
  for q, m in metrics.items():
    print(
        f"[metrics] {q:12s} rel_l2={m['rel_l2']:.3e} 1-cos={1 - m['cos']:.3e} max_abs={m['max_abs']:.3e} worst={m['worst']}"
    )
  return {"suite": suite, "full": full, "mt": mt, "ref": ref, "metrics": metrics}


def test_static_config_diff_empty(ctx):
  suite = ctx["suite"]
  for name, lm in list(suite.models.items()) + [("full", ctx["full"])]:
    assert not L.static_config_diff(lm.cfg, suite.args), name


def test_transfer_accounting_and_roundtrip(ctx):
  suite = ctx["suite"]
  sources = {sg.name: {(sg.layer_map or {}).get(i, i) for i in range(sg.num_layers)} for sg in suite.subgroups}
  sources["full"] = set(range(suite.args.n_layers))
  for sg in suite.subgroups:
    assert set(sg.ref_layers) <= sources[sg.name]
  for name, lm in list(suite.models.items()) + [("full", ctx["full"])]:
    rep = lm.report
    print(f"[transfer] {name}: {rep.summary()}")
    assert not rep.left_at_init, (name, rep.left_at_init)
    assert rep.dummy and all("mhc_norm-scale" in k for k in rep.dummy) and rep.dummy_all_ones, (name, rep.dummy)
    expected = {k for k in suite.ref_sd if k.startswith("mtp.") or _layer_idx(k) not in sources[name] | {None}}
    assert set(rep.unconsumed) == expected, (name, sorted(set(rep.unconsumed) ^ expected))
    native = L.native_from_maxtext(_all_maxtext_arrays(lm.model), lm.cfg, lm.hf_cfg, layer_map=lm.layer_map)
    assert set(native) == rep.consumed, (name, sorted(set(native) ^ rep.consumed))
    for k in rep.consumed:
      want = suite.ref_sd[k].numpy()
      np.testing.assert_array_equal(native[k].astype(want.dtype), want, err_msg=f"{name}: {k}")


@pytest.mark.xfail(
    strict=True,
    raises=(TypeError, ValueError),
    reason=(
        "Production saver (conversion_utils.process_maxtext_param, used by to_huggingface) with scan_layers=True "
        "slices every 1-D target list on param_scan_axis=1: prefix-layer expert stacks [E, in, out] fail with "
        "'cannot reshape array of shape (8, 32) ... into shape [64 32]', scanned MoEBiasVar [blocks, E] fails with "
        "'cannot reshape array of size 3 into shape (8,)'. The harness passes the actual stacking axis "
        "(native_from_maxtext fix_stack_axis=True)."
    ),
)
def test_production_saver_unmodified(ctx):
  lm = ctx["full"]
  L.native_from_maxtext(_all_maxtext_arrays(lm.model), lm.cfg, lm.hf_cfg, fix_stack_axis=False)


def test_schema_a_forward(ctx):
  metrics, mt, ref = ctx["metrics"], ctx["mt"], ctx["ref"]
  for q in ("S1/out", "S2/out", "S3/logits", "S3/xent_sum"):
    _check(metrics, q)
  assert mt["S3"]["total_weights"] == ref["S3"]["total_weights"] == 2 * (SEQ - 8)


def test_backward(ctx):
  suite, metrics, mt, ref = ctx["suite"], ctx["metrics"], ctx["mt"], ctx["ref"]
  for sg in suite.subgroups:
    _check(metrics, f"{sg.name}/dh")
    _check(metrics, f"{sg.name}/dW")
    keys = L.subgroup_native_keys(sg, suite.ref_sd)
    no_grad = {k for k in keys if _expected_no_grad(k)}
    assert set(ref[sg.name]["no_grad"]) == no_grad, (sg.name, sorted(set(ref[sg.name]["no_grad"]) ^ no_grad))
    for i, (dw_mt, dw_ref) in enumerate(zip(mt[sg.name]["dW"], ref[sg.name]["dW"])):
      assert set(dw_ref) == set(keys) - no_grad
      missing = set(dw_ref) - set(dw_mt)
      assert not missing, (sg.name, i, sorted(missing))
      for k in no_grad:
        if k in dw_mt:  # Indexer params are Params; gate.bias / tid2eid live outside nnx.Param.
          assert not np.any(dw_mt[k]), (sg.name, i, k, float(np.abs(dw_mt[k]).max()))
    assert max(mt[sg.name]["dbias_max"]) == 0.0, sg.name


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MaxText mhc_norm (RMSNorm before the hyper-connection pre-mix, mhc.py) has a trainable scale with no "
        "native counterpart (converter maps it to None and loads ones); it gets a nonzero gradient, so SFT "
        "updates a parameter the reference lacks. See the printed grad norms."
    ),
)
def test_no_maxtext_only_trainable_params(ctx):
  norms = {}
  for sg in ctx["suite"].subgroups:
    for k, v in ctx["mt"][sg.name]["unmapped_grad_norm"][0].items():
      if v:
        norms[f"{sg.name}:{k}"] = v
  print(f"[mhc_norm] nonzero grad norms (ct0): {norms}")
  assert not norms


def _routing_records(ctx):
  """Captures forward routing decisions for all MoE and indexer layers."""
  suite, full = ctx["suite"], ctx["full"]
  x = suite.inputs
  mt_recs, ref_recs = [], []
  with L.maxtext_context(full.cfg, full.mesh), L.capture_maxtext_topk(mt_recs):
    jax.jit(full.views.full)(full.views.params, full.views.bias, x["tokens"], x["segs"], x["pos"])
    jax.effects_barrier()
  model = L.build_reference(suite.ref_sd, suite.args, torch.float32)
  tt = torch.tensor(x["tokens"])
  with torch.no_grad(), L.capture_reference_topk(model, ref_recs):
    L.ref_head_logits(model, L.ref_layers(model, range(suite.args.n_layers), L.ref_embed(model, tt), tt))
  return mt_recs, ref_recs


def test_routing_and_indexer_agreement(ctx):
  suite = ctx["suite"]
  mt_recs, ref_recs = _routing_records(ctx)
  n_indexer = sum(1 for r in suite.args.compress_ratios[: suite.args.n_layers] if r == 4)
  for kind, expected_n in (("moe", suite.args.n_layers), ("indexer", n_indexer)):
    a = [r for r in mt_recs if r[0] == kind]
    b = [r for r in ref_recs if r[0] == kind]
    assert len(a) == len(b) == expected_n, (kind, len(a), len(b))
    for i, (ra, rb) in enumerate(zip(a, b)):
      exact = L.topk_agreement(ra[2], rb[2])
      valid = L.valid_topk_fraction(ra[2], rb[2], rb[3], TOPK_TIE_ATOL)
      print(f"[routing] {kind} #{i} hash={ra[1]} exact_agreement={exact:.4f} valid_topk={valid:.4f}")
      assert ra[1] == rb[1], (kind, i)
      if kind == "moe" and rb[1]:
        assert exact == 1.0, (kind, i, exact)
      assert valid == 1.0, (kind, i, exact, valid)


def test_indexer_zero_score_ties_two_heads():
  """With R.tiny_args() (2 indexer heads) MaxText picks are valid top-k sets (ties only)."""
  jax.config.update("jax_default_matmul_precision", "highest")
  args = R.tiny_args()
  ref_sd = L.init_reference_state(args, WEIGHT_SEED)
  x = L.make_inputs(args, SEQ, DATA_SEED)
  p = args.n_hash_layers
  lm = L.build_loaded_model(args, SEQ, p + 2, ref_sd)
  h0 = L.reference_boundaries(ref_sd, args, x)[0]
  mt_recs, ref_recs = [], []
  with L.maxtext_context(lm.cfg, lm.mesh), L.capture_maxtext_topk(mt_recs):
    jax.jit(lambda h, pr, b: lm.views.s1(h, pr, b, x["tokens"], x["segs"], x["pos"]))(
        jnp.asarray(h0), lm.views.params, lm.views.bias
    )
    jax.effects_barrier()
  model = L.build_reference(ref_sd, args, torch.float32)
  with torch.no_grad(), L.capture_reference_topk(model, ref_recs):
    L.ref_layers(model, range(p), torch.tensor(h0), torch.tensor(x["tokens"]))
  (a,) = [r for r in mt_recs if r[0] == "indexer"]
  (b,) = [r for r in ref_recs if r[0] == "indexer"]
  exact = L.topk_agreement(a[2], b[2])
  valid = L.valid_topk_fraction(a[2], b[2], b[3], TOPK_TIE_ATOL)
  print(f"[indexer 2 heads] layer {p - 1}: exact_agreement={exact:.4f} valid_topk={valid:.4f}")
  assert valid == 1.0


def test_schema_b_chain_and_full_model(ctx):
  suite, full = ctx["suite"], ctx["full"]
  ref_logits, ref_xent = suite.reference_full(torch.float32)
  chain_logits, chain_xent = suite.maxtext_chain()
  full_logits = suite.maxtext_full_logits(full)
  results = {
      "chain/logits": L.compare(chain_logits, ref_logits),
      "chain/xent_sum": L.compare([chain_xent], [ref_xent]),
      "full/logits": L.compare(full_logits, ref_logits),
      "chain_vs_full/logits": L.compare(chain_logits, full_logits),
  }
  for q, m in results.items():
    print(f"[schema B] {q:22s} rel_l2={m['rel_l2']:.3e} 1-cos={1 - m['cos']:.3e} max_abs={m['max_abs']:.3e}")
  for q in ("chain/logits", "chain/xent_sum", "full/logits"):
    family = "schema_b_" + q.rsplit("/", maxsplit=1)[-1]
    assert results[q]["rel_l2"] <= REL_L2_MAX[family], (q, results[q])
    assert 1 - results[q]["cos"] <= ONE_MINUS_COS_MAX[family], (q, results[q])


def test_coverage(ctx):
  suite, full = ctx["suite"], ctx["full"]
  x = suite.inputs
  full_mt, parts_mt = set(), set()
  with L.maxtext_context(full.cfg, full.mesh), L.trace_maxtext_modules(full.model, full_mt):
    jax.eval_shape(
        lambda p, b: full.views.full(p, b, x["tokens"], x["segs"], x["pos"]), full.views.params, full.views.bias
    )
  for sg in suite.subgroups:
    lm = suite.models[sg.name]
    with L.maxtext_context(lm.cfg, lm.mesh), L.trace_maxtext_modules(lm.model, parts_mt):
      jax.eval_shape(suite.maxtext_fn(sg), jnp.asarray(suite.h_in(sg)), lm.views.params, lm.views.bias)
      if sg.name == "S1":
        jax.eval_shape(lambda p, b, v=lm.views: v.embed(p, b, x["tokens"], x["pos"]), lm.views.params, lm.views.bias)
  tt = torch.tensor(x["tokens"])
  full_pt, parts_pt = set(), set()
  model = L.build_reference(suite.ref_sd, suite.args, torch.float32)
  with torch.no_grad(), L.trace_torch_modules(model, full_pt):
    L.ref_head_logits(model, L.ref_layers(model, range(suite.args.n_layers), L.ref_embed(model, tt), tt))
  for sg in suite.subgroups:
    model = L.build_reference(suite.ref_sd, suite.args, torch.float32)
    with torch.no_grad(), L.trace_torch_modules(model, parts_pt):
      if sg.name == "S1":
        L.ref_embed(model, tt)
      y = L.ref_layers(model, sg.ref_layers, torch.tensor(suite.h_in(sg)), tt)
      if sg.head:
        L.ref_head_logits(model, y)
  print(f"[coverage] MaxText full={len(full_mt)} parts={len(parts_mt)}; torch full={len(full_pt)} parts={len(parts_pt)}")
  outer_wrappers = {("Transformer",), ("NNXDecoder",)}
  assert full_mt - outer_wrappers == parts_mt, sorted((full_mt - outer_wrappers) ^ parts_mt)
  assert full_pt == parts_pt, sorted(full_pt ^ parts_pt)
  for side, recs in (("maxtext", parts_mt), ("torch", parts_pt)):
    assert L.layer_types_of(recs) == L.LAYER_TYPES, (side, L.layer_types_of(recs))


@pytest.fixture(scope="module")
def packed():
  """Two 264-token docs packed in 528 (seq 528 only here); MaxText S1+S2 vs reference doc 2 alone."""
  jax.config.update("jax_default_matmul_precision", "highest")
  args = _args(max_seq_len=2 * PACKED_DOC)
  r = L.packed_isolation_run(args, L.init_reference_state(args, WEIGHT_SEED), PACKED_DOC, DATA_SEED)
  d = PACKED_DOC
  r["doc2_vs_ref"] = L.compare(r["out"][:, d:], r["ref_out"][torch.float32])
  r["leak_out"] = float(np.abs(r["out_p"][:, d:] - r["out"][:, d:]).max())
  r["leak_dh"] = float(np.abs(r["dh_p"][:, d:] - r["dh"][:, d:]).max())
  err = np.abs(r["out"][:, d:] - r["ref_out"][torch.float32]).max(axis=(0, 2, 3))
  r["first_bad_pos"] = int(np.argmax(err > 1e-5)) if np.any(err > 1e-5) else None
  print(
      f"\n[packed] perturb doc1 -> max|d h_out[doc2]|={r['leak_out']} max|d dh_in[doc2]|={r['leak_dh']}; "
      f"doc2 vs reference-alone {r['doc2_vs_ref']}, first doc2 position with err > 1e-5: {r['first_bad_pos']}"
  )
  return r


def test_packed_no_cross_document_leak(packed):
  assert packed["leak_out"] == 0.0 and packed["leak_dh"] == 0.0, (packed["leak_out"], packed["leak_dh"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Packed doc 2 starting at 264 (not a multiple of the HCA ratio 128): MaxText S1+S2 doc-2 output vs "
        "reference on doc 2 alone rel_l2 2.97e-3..3.36e-3 (3 seeds; ref fp32 vs fp64 3.9e-7). Error starts "
        "exactly at doc-local position 127, the first HCA compressed entry: MaxText compresses on absolute "
        "128-token windows rather than per-document windows. No leak (doc 1 perturbation changes doc 2 by 0.0). "
        "With doc_len=256 (aligned) rel_l2 = 1.42e-7."
    ),
)
def test_packed_doc2_matches_reference_alone(packed):
  m = packed["doc2_vs_ref"]
  assert m["rel_l2"] <= REL_L2_MAX["packed_out"], m
  assert 1 - m["cos"] <= ONE_MINUS_COS_MAX["packed_out"], m


def test_packed_doc2_matches_reference_alone_when_aligned():
  """Aligned packed doc 2 (doc_len=256, multiple of compress_ratio=128) matches reference alone."""
  jax.config.update("jax_default_matmul_precision", "highest")
  args = _args(max_seq_len=512)
  ref_sd = L.init_reference_state(args, WEIGHT_SEED)
  r = L.packed_isolation_run(args, ref_sd, 256, DATA_SEED)
  leak_out = float(np.abs(r["out_p"][:, 256:] - r["out"][:, 256:]).max())
  leak_dh = float(np.abs(r["dh_p"][:, 256:] - r["dh"][:, 256:]).max())
  assert leak_out == 0.0, f"leak_out {leak_out} != 0.0"
  assert leak_dh == 0.0, f"leak_dh {leak_dh} != 0.0"
  m = L.compare(r["out"][:, 256:], r["ref_out"][torch.float32])
  assert m["rel_l2"] <= REL_L2_MAX["packed_out"], m
  assert 1 - m["cos"] <= ONE_MINUS_COS_MAX["packed_out"], m


def _print_fault(name: str, target_metric: str, measured_rel_l2: float, threshold: float, aux: str = "") -> None:
  """Prints formatted [fault] line and asserts measured_rel_l2 / threshold > 10.0."""
  margin = measured_rel_l2 / threshold
  aux_str = f" {aux}" if aux else ""
  msg = (
      f"[fault] {name:30s} {target_metric:15s} rel_l2={measured_rel_l2:.3e} "
      f"thresh={threshold:.1e} margin={margin:.1f}x{aux_str}"
  )
  print(msg)
  assert margin > 10.0, f"{name}: margin {margin} <= 10.0"


def test_fault_injection_matrix(ctx):
  """Injects 9 historical/structural defects, verifying each fails its parity threshold with margin > 10x."""
  suite = ctx["suite"]
  ref = ctx["ref"]
  lm1 = suite.models["S1"]
  lm2 = suite.models["S2"]
  lm3 = suite.models["S3"]
  sg1 = suite.subgroups[0]
  sg2 = suite.subgroups[1]
  sg3 = suite.subgroups[2]
  h_in_s1 = jnp.asarray(suite.h_in(sg1))
  h_in_s2 = jnp.asarray(suite.h_in(sg2))
  h_in_s3 = jnp.asarray(suite.h_in(sg3))
  x = suite.inputs

  # 1. yarn_off (b/562177034)
  cfg_yo = L.make_config(suite.args, SEQ, "float32", sg1.num_layers, rope_type="default")
  m_yo, mesh_yo = L.build_maxtext_model(cfg_yo)
  hf_cfg_s1 = L.hf_config_from_ref_args(suite.args, sg1.num_layers)
  L.load_reference_into_maxtext(suite.ref_sd, m_yo, cfg_yo, hf_cfg_s1)
  v_yo = L.MaxTextSubgroups(m_yo, cfg_yo, mesh_yo)
  with L.maxtext_context(cfg_yo, mesh_yo):
    out_yo = v_yo.s1(h_in_s1, v_yo.params, v_yo.bias, x["tokens"], x["segs"], x["pos"])
  m_yo_comp = L.compare(np.asarray(out_yo), ref["S1"]["out"])
  _print_fault("yarn_off", "S1/out", m_yo_comp["rel_l2"], REL_L2_MAX["out"])

  # 2. tid2eid_zeroed (b/563048036)
  orig_tid2eid = {}
  for path, val in L.flat_state(lm1.views.rest).items():
    if "tid2eid" in path:
      orig_tid2eid[path] = np.asarray(val.get_value())
  try:
    for path, val in L.flat_state(lm1.views.rest).items():
      if "tid2eid" in path:
        val[...] = jnp.zeros_like(val.get_value())
    mt_recs = []
    with L.maxtext_context(lm1.cfg, lm1.mesh), L.capture_maxtext_topk(mt_recs):
      out_t2e = lm1.views.s1(h_in_s1, lm1.views.params, lm1.views.bias, x["tokens"], x["segs"], x["pos"])
      jax.effects_barrier()
    m_t2e = L.compare(np.asarray(out_t2e), ref["S1"]["out"])
    moe_recs = [r for r in mt_recs if r[0] == "moe" and r[1]]
    ref_recs = []
    m_ref_s1 = L.build_reference(suite.ref_sd, suite.args, torch.float32)
    with torch.no_grad(), L.capture_reference_topk(m_ref_s1, ref_recs):
      L.ref_layers(m_ref_s1, sg1.ref_layers, torch.tensor(suite.h_in(sg1)), torch.tensor(x["tokens"]))
    ref_moe = [r for r in ref_recs if r[0] == "moe" and r[1]]
    agreements = [L.topk_agreement(ra[2], rb[2]) for ra, rb in zip(moe_recs, ref_moe)]
    min_agree = min(agreements) if agreements else 1.0
    assert min_agree < 1.0, f"min_agree {min_agree} not < 1.0"
    _print_fault("tid2eid_zeroed", "S1/out", m_t2e["rel_l2"], REL_L2_MAX["out"], f"topk_agreement={min_agree:.4f}")
  finally:
    for path, val in L.flat_state(lm1.views.rest).items():
      if path in orig_tid2eid:
        val[...] = jnp.asarray(orig_tid2eid[path])

  # 3. rope_rotate_half_swapped
  orig_rope = maxtext.layers.attention_compressed.CompressedAttention._apply_rotary_embedding_v4

  def bad_rope(self, *a, **k):
    out = orig_rope(self, *a, **k)
    d = out.shape[-1] // 2
    return jnp.concatenate([out[..., d:], out[..., :d]], axis=-1)

  maxtext.layers.attention_compressed.CompressedAttention._apply_rotary_embedding_v4 = bad_rope
  try:
    with L.maxtext_context(lm1.cfg, lm1.mesh):
      out_rope = jax.jit(lambda h, p, b: lm1.views.s1(h, p, b, x["tokens"], x["segs"], x["pos"]))(
          h_in_s1, lm1.views.params, lm1.views.bias
      )
    m_rope = L.compare(np.asarray(out_rope), ref["S1"]["out"])
    _print_fault("rope_rotate_half_swapped", "S1/out", m_rope["rel_l2"], REL_L2_MAX["out"])
  finally:
    maxtext.layers.attention_compressed.CompressedAttention._apply_rotary_embedding_v4 = orig_rope

  # 4. attention_sink_dropped
  def drop_sinks(path, val):
    path_str = "/".join(str(p.key if hasattr(p, "key") else p) for p in path)
    if "sinks" in path_str:
      return jnp.full_like(val, -1e9)
    return val

  bad_params_s1 = jax.tree_util.tree_map_with_path(drop_sinks, lm1.views.params)
  with L.maxtext_context(lm1.cfg, lm1.mesh):
    out_sink = lm1.views.s1(h_in_s1, bad_params_s1, lm1.views.bias, x["tokens"], x["segs"], x["pos"])
  m_sink = L.compare(np.asarray(out_sink), ref["S1"]["out"])
  _print_fault("attention_sink_dropped", "S1/out", m_sink["rel_l2"], REL_L2_MAX["out"])

  # 5. stop_gradient_branch (backward-only defect)
  orig_moe_call = maxtext.layers.moe.RoutedMoE.__call__

  def bad_moe_call(self, *a, **k):
    out, lb_loss, bias_updates = orig_moe_call(self, *a, **k)
    return jax.lax.stop_gradient(out), lb_loss, bias_updates

  maxtext.layers.moe.RoutedMoE.__call__ = bad_moe_call
  try:
    ct0 = jax.tree.map(jnp.asarray, suite.cotangents["S1"][0])
    runner5 = L.vjp_runner(suite.maxtext_fn(sg1))
    with L.maxtext_context(lm1.cfg, lm1.mesh):
      out5, dh5, dp5, _ = runner5(h_in_s1, lm1.views.params, lm1.views.bias, ct0)
    m5_out = L.compare(np.asarray(out5), ref["S1"]["out"])
    assert m5_out["rel_l2"] <= REL_L2_MAX["out"], f"fwd failed: {m5_out}"
    m5_dh = L.compare(np.asarray(dh5), ref["S1"]["dh"][0])
    flat_dp5 = L.flat_arrays(dp5, "params")
    dw5 = L.native_from_maxtext(flat_dp5, lm1.cfg, lm1.hf_cfg, layer_map=lm1.layer_map)
    ref_dw_map = ref["S1"]["dW"][0]
    dw_comps = [
        L.compare(
            v,
            ref_dw_map[k].numpy() if hasattr(ref_dw_map[k], "numpy") else ref_dw_map[k],
        )
        for k, v in dw5.items()
        if k in ref_dw_map
    ]
    m5_dw_max = max(c["rel_l2"] for c in dw_comps)
    _print_fault(
        "stop_gradient_branch_dh", "S1/dh", m5_dh["rel_l2"], REL_L2_MAX["dh"], f"fwd_rel_l2={m5_out['rel_l2']:.3e}"
    )
    _print_fault("stop_gradient_branch_dW", "S1/dW", m5_dw_max, REL_L2_MAX["dW"])
  finally:
    maxtext.layers.moe.RoutedMoE.__call__ = orig_moe_call

  # 6. xent_zeroed (b/563046810)
  x_zero = dict(x)
  x_zero["target_segs"] = np.zeros_like(x_zero["target_segs"])
  with L.maxtext_context(lm3.cfg, lm3.mesh):
    logits6, xent_sum6 = lm3.views.s3(
        h_in_s3,
        lm3.views.params,
        lm3.views.bias,
        x_zero["tokens"],
        x_zero["segs"],
        x_zero["pos"],
        x_zero["targets"],
        x_zero["target_segs"],
    )
  m6_xent = abs(float(xent_sum6) - ref["S3"]["xent_sum"]) / (abs(ref["S3"]["xent_sum"]) + 1e-12)
  _print_fault("xent_zeroed_xent_sum", "S3/xent_sum", m6_xent, REL_L2_MAX["xent_sum"])

  def fn6(h, p, b):
    return lm3.views.s3(
        h, p, b, x_zero["tokens"], x_zero["segs"], x_zero["pos"], x_zero["targets"], x_zero["target_segs"]
    )

  runner6 = L.vjp_runner(fn6)
  loss_ct = (jnp.zeros_like(logits6), jnp.float32(1.0))
  with L.maxtext_context(lm3.cfg, lm3.mesh):
    _, dh6, _, _ = runner6(h_in_s3, lm3.views.params, lm3.views.bias, loss_ct)
  m_ref3 = L.build_reference(suite.ref_sd, suite.args, torch.float32)
  h_in_torch = torch.tensor(suite.h_in(sg3), requires_grad=True)
  tt_torch = torch.tensor(x["tokens"])
  y_torch = L.ref_layers(m_ref3, sg3.ref_layers, h_in_torch, tt_torch)
  log_torch = L.ref_head_logits(m_ref3, y_torch)
  xs_torch, _ = L.ref_masked_xent(log_torch, torch.tensor(x["targets"]), torch.tensor(x["target_segs"]))
  (ref_dh_loss_only,) = torch.autograd.grad(xs_torch, h_in_torch)
  m6_dh = L.compare(np.asarray(dh6), ref_dh_loss_only.detach().numpy())
  _print_fault("xent_zeroed_dh", "S3/dh", m6_dh["rel_l2"], REL_L2_MAX["dh"])

  # 7. moe_bias_not_restored (b/562177048)
  b_zero = jax.tree.map(jnp.zeros_like, lm2.views.bias)
  mt_recs7 = []
  with L.maxtext_context(lm2.cfg, lm2.mesh), L.capture_maxtext_topk(mt_recs7):
    out7 = lm2.views.blocks(h_in_s2, lm2.views.params, b_zero, x["tokens"], x["segs"], x["pos"])
    jax.effects_barrier()
  m7 = L.compare(np.asarray(out7), ref["S2"]["out"])
  moe_recs7 = [r for r in mt_recs7 if r[0] == "moe"]
  ref_recs7 = []
  m_ref_s2 = L.build_reference(suite.ref_sd, suite.args, torch.float32)
  with torch.no_grad(), L.capture_reference_topk(m_ref_s2, ref_recs7):
    L.ref_layers(m_ref_s2, sg2.ref_layers, torch.tensor(suite.h_in(sg2)), torch.tensor(x["tokens"]))
  ref_moe7 = [r for r in ref_recs7 if r[0] == "moe"]
  agree7 = [L.topk_agreement(ra[2], rb[2]) for ra, rb in zip(moe_recs7, ref_moe7)]
  min_agree7 = min(agree7) if agree7 else 1.0
  assert min_agree7 < 1.0, f"min_agree7 {min_agree7} not < 1.0"
  _print_fault("moe_bias_not_restored", "S2/out", m7["rel_l2"], REL_L2_MAX["out"], f"topk_agreement={min_agree7:.4f}")

  # 8. param_scan_axis_1d_missliced
  def misslice_pre_norm(path, val):
    path_str = "/".join(str(p.key if hasattr(p, "key") else p) for p in path)
    if "scanned_blocks" in path_str and "pre_self_attention_layer_norm" in path_str and "scale" in path_str:
      return jnp.broadcast_to(val[:1, :], val.shape)
    return val

  bad_params_s2 = jax.tree_util.tree_map_with_path(misslice_pre_norm, lm2.views.params)

  def fn8(h, p, b):
    return lm2.views.blocks(h, p, b, x["tokens"], x["segs"], x["pos"])

  runner8 = L.vjp_runner(fn8)
  ct0_s2 = jax.tree.map(jnp.asarray, suite.cotangents["S2"][0])
  with L.maxtext_context(lm2.cfg, lm2.mesh):
    out8, dh8, _, _ = runner8(h_in_s2, bad_params_s2, lm2.views.bias, ct0_s2)
  m8_out = L.compare(np.asarray(out8), ref["S2"]["out"])
  m8_dh = L.compare(np.asarray(dh8), ref["S2"]["dh"][0])
  _print_fault("param_scan_axis_1d_missliced_out", "S2/out", m8_out["rel_l2"], REL_L2_MAX["out"])
  _print_fault("param_scan_axis_1d_missliced_dh", "S2/dh", m8_dh["rel_l2"], REL_L2_MAX["dh"])

  # 9. sft_loss_mask_shifted
  x_rolled = dict(x)
  x_rolled["target_segs"] = np.roll(x_rolled["target_segs"], 1, axis=-1)
  with L.maxtext_context(lm3.cfg, lm3.mesh):
    _, xent_sum9 = lm3.views.s3(
        h_in_s3,
        lm3.views.params,
        lm3.views.bias,
        x_rolled["tokens"],
        x_rolled["segs"],
        x_rolled["pos"],
        x_rolled["targets"],
        x_rolled["target_segs"],
    )
  m9_xent = abs(float(xent_sum9) - ref["S3"]["xent_sum"]) / (abs(ref["S3"]["xent_sum"]) + 1e-12)
  _print_fault("sft_loss_mask_shifted", "S3/xent_sum", m9_xent, REL_L2_MAX["xent_sum"])
